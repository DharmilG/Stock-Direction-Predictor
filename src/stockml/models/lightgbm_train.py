from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.storages import RDBStorage
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

from ..config import load_settings
from ..utils.hash import stable_hash
from ..utils.io import atomic_write_json
from ..utils.logging import setup_logging
from ..utils.state import update_state
from ..validation.walk_forward import PurgedWalkForwardSplit
from ..validation.time_splits import three_way_date_split

TARGET = "target"
DROP_COLUMNS = {"open", "high", "low", "close", "volume", "target", "label_raw", "event_end", "event_start", "entry_price", "exit_price", "realized_return"}
# G1-Step1: bhav delivery columns exist only from 2020+ on the free NSE
# archive. NaN there means "no coverage", not "neutral" (B-07) — LightGBM
# handles missing natively, so these columns are exempt from the
# complete-row requirement instead of dropping all pre-2020 rows.
# G1-Step2/3: F-11 group columns are NaN for ungrouped symbols by design;
# sector_dispersion is NaN for single-member sectors; regime_bull needs 200d
# warmup. Same native-missing treatment — never zero-filled.
NAN_OK_COLUMNS = {
    "delivery_per", "delivery_z20", "turnover_lacs", "turnover_z20",
    "no_trades", "trades_z20",
    "ret_5_minus_sector", "ret_20_minus_sector", "sector_xs_rank",
    "sector_dispersion", "ret_5_minus_group", "ret_20_minus_group",
    "group_xs_rank", "group_dispersion", "regime_bull",
    # N6 news context: means/shocks/z-scores are NaN on quiet days by
    # design (B-07); counts/flags are true zeros. Native missing handling.
    "news_co_sent_mean", "news_co_count_z20", "news_co_sent_shock",
    "ann_co_days_since_last_ann",
    "ev_co_tone_mean", "ev_co_count_z20",
    "news_mkt_sent_mean", "news_mkt_count_z20",
}


def load_training_frame(settings) -> pd.DataFrame:
    frames = []
    for symbol in settings.symbols:
        safe = symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        path = settings.paths.data_processed / f"{safe}_dataset.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed dataset: {path}")
        df = pd.read_parquet(path).copy()
        df.attrs = {}
        df.index = pd.MultiIndex.from_arrays([pd.DatetimeIndex(df.index), [symbol] * len(df)], names=["timestamp", "symbol"])
        frames.append(df)
    if not frames:
        raise RuntimeError("No datasets found.")
    panel = pd.concat(frames).sort_index(kind="stable")
    return panel


def prepare_xy(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS and c != "symbol"]
    X = df[feature_cols].copy()
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    strict_cols = [c for c in X.columns if c not in NAN_OK_COLUMNS]
    valid = X[strict_cols].notna().all(axis=1) & df[TARGET].notna()
    X = X.loc[valid]
    y = df.loc[valid, TARGET].astype(int)
    events = df.loc[valid, "event_end"]
    return X, y, events, feature_cols


def _timestamps(index) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        return pd.DatetimeIndex(index.get_level_values("timestamp"))
    return pd.DatetimeIndex(index)


def _folds(X: pd.DataFrame, events: pd.Series, settings):
    splitter = PurgedWalkForwardSplit(settings.n_splits, settings.purge_days, settings.embargo_days)
    return list(splitter.split(_timestamps(X.index), events))


def _sample_weights(y: pd.Series, enabled: bool) -> np.ndarray | None:
    if not enabled:
        return None
    counts = y.value_counts().to_dict()
    total = float(len(y))
    n_classes = max(len(counts), 1)
    class_weight = {int(k): total / (n_classes * float(v)) for k, v in counts.items() if v}
    return y.map(class_weight).to_numpy(dtype=float)


def _selection_score(y_true, pred) -> float:
    bal = balanced_accuracy_score(y_true, pred)
    mcc = matthews_corrcoef(y_true, pred)
    mcc_norm = 0.5 * (mcc + 1.0)
    return float(0.5 * bal + 0.5 * mcc_norm)


def objective_factory(X: pd.DataFrame, y: pd.Series, events: pd.Series, settings, trial_dir: Path):
    folds = _folds(X, events, settings)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.55, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.55, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 20.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            "seed": settings.random_seed,
            "feature_fraction_seed": settings.random_seed,
            "bagging_seed": settings.random_seed,
            "data_random_seed": settings.random_seed,
        }
        scores = []
        for fold in folds:
            train_idx, valid_idx = fold.train_idx, fold.valid_idx
            w_train = _sample_weights(y.iloc[train_idx], settings.class_weighting)
            w_valid = _sample_weights(y.iloc[valid_idx], settings.class_weighting)
            dtrain = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx], weight=w_train, free_raw_data=False)
            dvalid = lgb.Dataset(X.iloc[valid_idx], label=y.iloc[valid_idx], weight=w_valid, reference=dtrain, free_raw_data=False)
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=min(settings.num_boost_round, 500),
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(min(settings.early_stopping_rounds, 50), verbose=False)],
            )
            proba = booster.predict(X.iloc[valid_idx], num_iteration=booster.best_iteration)
            pred = np.argmax(proba, axis=1)
            score = _selection_score(y.iloc[valid_idx], pred)
            scores.append(score)
            # D2: observe training per fold per class (logging only — no
            # behavior change). Answers: does UP fail in all eras or recent
            # ones? Which features win each fold (stability vs one lucky fold)?
            try:
                yv = y.iloc[valid_idx].to_numpy()
                for c, name in ((0, "down"), (1, "flat"), (2, "up")):
                    m = yv == c
                    trial.set_user_attr(f"fold{fold.fold_id}_recall_{name}",
                                        float((pred[m] == c).mean()) if m.any() else float("nan"))
                trial.set_user_attr(f"fold{fold.fold_id}_score", float(score))
                gain = pd.Series(booster.feature_importance(importance_type="gain"),
                                 index=list(X.columns)).sort_values(ascending=False)
                trial.set_user_attr(f"fold{fold.fold_id}_top15", ",".join(gain.head(15).index))
                diag_path = Path(trial_dir) / "fold_diagnostics.jsonl"
                diag_path.parent.mkdir(parents=True, exist_ok=True)
                with open(diag_path, "a", encoding="utf-8") as fh:
                    fh.write(__import__("json").dumps({
                        "trial": int(trial.number), "fold": int(fold.fold_id),
                        "score": float(score),
                        "recall": {name: float((pred[yv == c] == c).mean()) if (yv == c).any() else None
                                   for c, name in ((0, "down"), (1, "flat"), (2, "up"))},
                        "top15": list(gain.head(15).index),
                    }) + "\n")
            except Exception:
                pass  # diagnostics must never break optimization
            trial.report(float(np.mean(scores)), step=fold.fold_id)
            if trial.should_prune():
                logger = __import__("logging").getLogger(__name__)
                logger.debug("Optuna trial %s pruned after fold %s", trial.number, fold.fold_id)
                raise optuna.TrialPruned()
        return float(np.mean(scores)) if scores else -1.0

    return objective


class CheckpointCallback:
    def __init__(self, path: Path, every: int, metadata: dict[str, Any]):
        self.path = path
        self.meta_path = path.with_suffix(path.suffix + ".json")
        self.every = max(1, every)
        self.metadata = dict(metadata)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _atomic_save(self, model: lgb.Booster, iteration: int) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        model.save_model(str(tmp))
        os.replace(tmp, self.path)
        meta_tmp = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
        payload = {**self.metadata, "completed_rounds": int(iteration)}
        meta_tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(meta_tmp, self.meta_path)

    def __call__(self, env: lgb.callback.CallbackEnv) -> None:
        iteration = env.iteration + 1
        if iteration % self.every == 0:
            self._atomic_save(env.model, iteration)


def train_with_resume(X, y, params, settings, checkpoint_path: Path, final_path: Path, context_path: Path):
    dtrain = lgb.Dataset(X, label=y, weight=_sample_weights(y, settings.class_weighting), free_raw_data=False)
    params_hash = stable_hash(params)
    meta_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    existing = checkpoint_path if checkpoint_path.exists() else None
    completed_rounds = 0

    if existing:
        if not meta_path.exists():
            existing = None
        else:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                current_context = json.loads(context_path.read_text(encoding="utf-8"))
                compatible = (
                    meta.get("dataset_signature") == current_context.get("dataset_signature")
                    and meta.get("params_hash") == params_hash
                    and int(meta.get("num_features", -1)) == int(X.shape[1])
                )
                if not compatible:
                    existing = None
            except Exception:
                existing = None

    if existing:
        try:
            checkpoint_model = lgb.Booster(model_file=str(existing))
            completed_rounds = checkpoint_model.current_iteration()
        except Exception:
            existing = None
            completed_rounds = 0

    if final_path.exists() and completed_rounds >= settings.num_boost_round:
        return lgb.Booster(model_file=str(final_path))

    remaining = max(0, settings.num_boost_round - completed_rounds)
    if remaining == 0:
        booster = lgb.Booster(model_file=str(existing))
    else:
        checkpoint_metadata = {
            "dataset_signature": json.loads(context_path.read_text(encoding="utf-8")).get("dataset_signature"),
            "params_hash": params_hash,
            "num_features": int(X.shape[1]),
        }
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=remaining,
            init_model=str(existing) if existing else None,
            callbacks=[
                CheckpointCallback(checkpoint_path, settings.checkpoint_every_rounds, checkpoint_metadata),
            ],
        )

    tmp_final = final_path.with_suffix(final_path.suffix + ".tmp")
    booster.save_model(str(tmp_final))
    os.replace(tmp_final, final_path)
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    current_context = json.loads(context_path.read_text(encoding="utf-8"))
    tmp_meta.write_text(
        json.dumps({
            "dataset_signature": current_context.get("dataset_signature"),
            "params_hash": params_hash,
            "num_features": int(X.shape[1]),
            "completed_rounds": int(booster.current_iteration()),
        }, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_meta, meta_path)
    return booster

def save_temperature_calibrator(model: lgb.Booster, X_cal: pd.DataFrame, y_cal: pd.Series, path: Path):
    from scipy.optimize import minimize_scalar

    logits = np.log(np.clip(model.predict(X_cal), 1e-12, 1.0))
    y = y_cal.to_numpy(dtype=int)

    def nll(temp: float) -> float:
        scaled = logits / temp
        scaled -= scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))))

    result = minimize_scalar(nll, bounds=(0.25, 5.0), method="bounded")
    atomic_write_json(path, {"temperature": float(result.x), "method": "temperature_scaling_multiclass"})


def run_training() -> None:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "train.log")
    df = load_training_frame(settings)
    dataset_signature = stable_hash({
        "symbols": settings.symbols,
        "rows": int(len(df)),
        "first_timestamp": str(df.index.get_level_values("timestamp").min()) if isinstance(df.index, pd.MultiIndex) else str(df.index.min()),
        "last_timestamp": str(df.index.get_level_values("timestamp").max()) if isinstance(df.index, pd.MultiIndex) else str(df.index.max()),
        "columns": list(df.columns),
        "target": {"horizon": settings.triple_barrier_horizon, "up": settings.triple_barrier_up_mult, "down": settings.triple_barrier_down_mult, "flat_band": settings.triple_barrier_flat_band},
    })
    context_path = paths.artifacts / "training_context.json"
    previous_context = __import__("json").loads(context_path.read_text()) if context_path.exists() else {}
    checkpoint_path = paths.checkpoints / "lightgbm_final_checkpoint.txt"
    if settings.force_retrain or previous_context.get("dataset_signature") != dataset_signature:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Removed incompatible old final-model checkpoint.")
    atomic_write_json(context_path, {"dataset_signature": dataset_signature})
    X, y, events, feature_cols = prepare_xy(df)
    class_counts = y.value_counts().sort_index().to_dict()
    class_share = (y.value_counts(normalize=True).sort_index()).to_dict()
    logger.info("Training matrix: %d rows x %d features | class counts=%s | shares=%s", len(X), X.shape[1], class_counts, {int(k): round(float(v), 4) for k, v in class_share.items()})
    if set(class_counts) != {0, 1, 2}:
        raise RuntimeError(f"Training dataset must contain all three labels DOWN/FLAT/UP; found {sorted(class_counts)}")
    if max(class_share.values()) > 0.75:
        logger.warning("Severe target imbalance detected (max class share %.1f%%). Review triple-barrier parameters before trusting accuracy.", 100 * max(class_share.values()))

    # Optuna is persisted in SQLite so completed trials survive process crashes/restarts.
    study_db = paths.checkpoints / "optuna.db"
    storage = RDBStorage(url=f"sqlite:///{study_db}", heartbeat_interval=60, grace_period=300)
    pruner_name = str(settings.optuna_pruner).strip().lower()
    if pruner_name in {"median", "medianpruner"}:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=max(8, min(10, settings.optuna_trials // 3)), n_warmup_steps=2)
    elif pruner_name in {"none", "nop", "disabled"}:
        pruner = optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unsupported OPTUNA_PRUNER={settings.optuna_pruner!r}; use none or median")
    study = optuna.create_study(
        study_name=f"{settings.project_name}-lgbm-{dataset_signature[:12]}",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=settings.random_seed),
        pruner=pruner,
    )
    _, _, pretest_mask, _, test_start = three_way_date_split(_timestamps(X.index))
    opt_mask = ~pretest_mask
    X_opt, y_opt, events_opt = X.loc[opt_mask], y.loc[opt_mask], events.loc[opt_mask]
    objective = objective_factory(X_opt, y_opt, events_opt, settings, paths.checkpoints / "trials")
    remaining_trials = max(0, settings.optuna_trials - len(study.trials))
    if remaining_trials:
        logger.info("Optuna: running %d remaining trials", remaining_trials)
        study.optimize(objective, n_trials=remaining_trials, timeout=settings.optuna_timeout_seconds or None, gc_after_trial=True)
    else:
        logger.info("Optuna study already has requested number of trials; skipping.")

    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None
    if best_trial is not None and best_trial.state == optuna.trial.TrialState.COMPLETE:
        best_params = dict(best_trial.params)
    else:
        logger.warning("No completed Optuna trial is available; using conservative default LightGBM parameters.")
        best_params = {
            "learning_rate": 0.02,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "min_gain_to_split": 0.0,
        }
    best_params.update({
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss", "verbosity": -1,
        "seed": settings.random_seed, "feature_fraction_seed": settings.random_seed,
        "bagging_seed": settings.random_seed, "data_random_seed": settings.random_seed,
    })
    atomic_write_json(paths.artifacts / "best_params.json", best_params)

    # Final model protocol: 60% train, 20% calibration/validation, final 20% untouched test.
    train_mask, calibration_mask, test_mask, calibration_start, test_start = three_way_date_split(_timestamps(X.index))
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_cal, y_cal = X.loc[calibration_mask], y.loc[calibration_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    final_path = paths.artifacts / "models" / "lightgbm_final.txt"
    checkpoint_path = paths.checkpoints / "lightgbm_final_checkpoint.txt"
    if settings.force_retrain and final_path.exists():
        final_path.unlink()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Training final model with resumable LightGBM checkpointing.")
    model = train_with_resume(X_train, y_train, best_params, settings, checkpoint_path, final_path, context_path)
    # Anti-memorization tripwire (hard user rule): high accuracy from
    # remembering is failure, not success. 95% means the model memorized.
    # Log + metadata only — never blocks the pipeline, but a flagged model
    # must never be celebrated or shipped without review.
    train_acc = test_acc_final = gap_pp = None
    try:
        train_acc = float((np.argmax(model.predict(X_train), axis=1) == y_train.to_numpy()).mean()) if len(X_train) else None
        test_acc_final = float((np.argmax(model.predict(X_test), axis=1) == y_test.to_numpy()).mean()) if len(X_test) else None
        if train_acc is not None and test_acc_final is not None:
            gap_pp = (train_acc - test_acc_final) * 100.0
            if gap_pp > 20.0:
                logger.error("MEMORIZATION FLAG: train acc %.4f vs test acc %.4f (gap %.1fpp > 20pp) — model may be remembering, not learning.",
                             train_acc, test_acc_final, gap_pp)
            if test_acc_final > 0.65:
                logger.error("TOO-GOOD FLAG: test acc %.4f > 0.65 — verify against leakage checklist before trusting.", test_acc_final)
            else:
                logger.info("Generalization check: train acc %.4f vs test acc %.4f (gap %.1fpp).", train_acc, test_acc_final, gap_pp)
    except Exception as exc:
        logger.warning("Generalization check skipped: %s", exc)
    if len(X_cal) >= 20 and y_cal.nunique() == 3:
        save_temperature_calibrator(model, X_cal, y_cal, paths.artifacts / "calibration.json")
    else:
        atomic_write_json(paths.artifacts / "calibration.json", {"temperature": 1.0, "method": "identity"})

    metadata = {
        "feature_columns": feature_cols,
        "symbols": settings.symbols,
        "history_years": settings.history_years,
        "model_type": "lightgbm_multiclass",
        "classes": {"0": "DOWN", "1": "FLAT", "2": "UP"},
        "best_params_hash": stable_hash(best_params),
        "dataset_signature": dataset_signature,
        "training_rows": int(len(X_train)),
        "calibration_rows": int(len(X_cal)),
        "test_rows": int(len(X_test)),
        "train_accuracy": train_acc,
        "final_test_accuracy": test_acc_final,
        "train_test_gap_pp": gap_pp,
        "calibration_start": str(calibration_start.date()),
        "test_start": str(test_start.date()),
    }
    atomic_write_json(paths.artifacts / "model_metadata.json", metadata)

    # MLflow uses a DB-backed tracking store by default; the URI is configurable for remote production.
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_name="final-lightgbm") as run:
        mlflow.log_params(best_params)
        mlflow.log_metrics({"optuna_best_mcc": float(study.best_value) if best_trial is not None else 0.0})
        mlflow.log_artifact(str(final_path))
        mlflow.log_artifact(str(paths.artifacts / "model_metadata.json"))
        mlflow.set_tags({"model_type": "lightgbm_multiclass", "environment": settings.environment})
    logger.info("Model training complete. Best CV selection score=%.5f", float(study.best_value) if best_trial is not None else 0.0)
    update_state(paths.state, "train", status="complete", model=str(final_path), optuna_trials=len(study.trials))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run_training()
