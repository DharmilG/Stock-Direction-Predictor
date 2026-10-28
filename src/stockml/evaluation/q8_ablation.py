from __future__ import annotations

"""Phase 8 — Ablation gate on Pool D (sealed holdout).

CODE READY — UNTESTED
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_settings
from ..data.qwen_harness import HOLDOUT_START, STAR_CUTOFF
from ..evaluation.dm import up_vs_rest_dm
from ..features.qwen_features import build_qwen_features_panel
from ..models.lightgbm_train import load_training_frame, prepare_xy
from ..qpipeline import state as q_state
from ..qpipeline.contracts import load_q_track_config, phase_artifacts
from ..utils.io import atomic_write_json
from ..utils.logging import setup_logging


def _eligible_touch_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("eligible") is True:
            count += 1
    return count


def _load_calendar(settings):
    return pd.DatetimeIndex(
        pd.read_parquet(settings.paths.root / "data" / "reference" / "trading_calendar.parquet").iloc[:, 0]
    )


def _join_qwen_features(
    X: pd.DataFrame,
    qwen_panel: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Join per-symbol qwen features onto the training matrix."""
    if not qwen_panel:
        return X.copy()
    frames = []
    for symbol, feats in qwen_panel.items():
        if feats is None or feats.empty:
            continue
        feats = feats.copy()
        feats.index.name = "timestamp"
        feats["symbol"] = symbol
        frames.append(feats.reset_index())
    if not frames:
        return X.copy()
    qwen_long = pd.concat(frames, ignore_index=True)
    qwen_long["timestamp"] = pd.to_datetime(qwen_long["timestamp"]).dt.tz_localize(None).dt.normalize()
    if isinstance(X.index, pd.MultiIndex):
        X_reset = X.reset_index()
        X_reset["timestamp"] = pd.to_datetime(X_reset["timestamp"]).dt.tz_localize(None).dt.normalize()
        merged = X_reset.merge(
            qwen_long,
            on=["timestamp", "symbol"],
            how="left",
            validate="many_to_one",
        )
        merged = merged.set_index(["timestamp", "symbol"])
    else:
        merged = X.copy()
        merged["timestamp"] = pd.to_datetime(merged.index).tz_localize(None).normalize()
        merged = merged.merge(
            qwen_long.drop(columns=["symbol"]),
            left_on="timestamp",
            right_on="timestamp",
            how="left",
        )
        merged = merged.set_index("timestamp")
    return merged


def _train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, params: dict, seed: int = 42):
    import lightgbm as lgb

    params = dict(params)
    params.update(
        {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "verbosity": -1,
            "seed": seed,
        }
    )
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    booster = lgb.train(params, dtrain, num_boost_round=1000)
    return booster


def _up_auc(y_true: pd.Series, proba: np.ndarray) -> float:
    y = np.asarray(y_true)
    is_up = (y == 2).astype(int)
    if len(np.unique(is_up)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(is_up, proba[:, 2]))


def _bootstrap_se(y_true: pd.Series, proba_base: np.ndarray, proba_qwen: np.ndarray, n: int, seed: int) -> float:
    y = np.asarray(y_true)
    is_up = (y == 2).astype(int)
    if len(np.unique(is_up)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(is_up[idx])) < 2:
            continue
        auc_base = roc_auc_score(is_up[idx], proba_base[idx, 2])
        auc_qwen = roc_auc_score(is_up[idx], proba_qwen[idx, 2])
        diffs.append(auc_qwen - auc_base)
    if not diffs:
        return float("nan")
    return float(np.std(diffs, ddof=1))


def run_phase8(**_kwargs) -> dict:
    """Run the sealed Pool D ablation gate once."""
    cfg = load_q_track_config().get("phase8", {})
    artifacts = phase_artifacts()
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "q8_ablation.log")

    holdout_start = pd.Timestamp(cfg.get("holdout_start", HOLDOUT_START))
    cutoff = pd.Timestamp(load_q_track_config().get("star", {}).get("cutoff", STAR_CUTOFF))

    df = load_training_frame(settings)
    X_base, y, events, feature_cols = prepare_xy(df)
    X_base = X_base.sort_index()
    y = y.loc[X_base.index]
    calendar = _load_calendar(settings)
    timestamps = X_base.index.get_level_values("timestamp") if isinstance(X_base.index, pd.MultiIndex) else X_base.index
    symbols = (
        X_base.index.get_level_values("symbol").astype(str).tolist()
        if isinstance(X_base.index, pd.MultiIndex)
        else [settings.symbols[0]] * len(X_base)
    )
    unique_symbols = sorted(set(symbols))
    qwen_panel = build_qwen_features_panel(pd.DatetimeIndex(timestamps).unique(), calendar, unique_symbols)
    X_qwen = _join_qwen_features(X_base, qwen_panel).sort_index()

    train_mask = pd.DatetimeIndex(timestamps) < cutoff
    test_mask = pd.DatetimeIndex(timestamps) >= holdout_start
    if train_mask.sum() < 100 or test_mask.sum() < 100:
        raise RuntimeError("Not enough rows in train or Pool D test window.")

    Xb_train, Xb_test = X_base.loc[train_mask], X_base.loc[test_mask]
    Xq_train, Xq_test = X_qwen.loc[train_mask], X_qwen.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    try:
        best_params = json.loads((paths.artifacts / "best_params.json").read_text(encoding="utf-8"))
    except Exception:
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
        }

    booster_base = _train_lightgbm(Xb_train, y_train, best_params, seed=settings.random_seed)
    booster_qwen = _train_lightgbm(Xq_train, y_train, best_params, seed=settings.random_seed)
    proba_base = booster_base.predict(Xb_test)
    proba_qwen = booster_qwen.predict(Xq_test)

    auc_base = _up_auc(y_test, proba_base)
    auc_qwen = _up_auc(y_test, proba_qwen)
    delta = auc_qwen - auc_base
    n_bootstrap = int(cfg.get("n_bootstrap", 1000))
    se = _bootstrap_se(
        y_test,
        proba_base,
        proba_qwen,
        n=n_bootstrap,
        seed=int(cfg.get("bootstrap_seed", 42)),
    )
    dm = up_vs_rest_dm(
        np.asarray(y_test),
        np.argmax(proba_qwen, axis=1),
        np.argmax(proba_base, axis=1),
        lags=5,
    )

    touch_ledger = Path(artifacts["phase8"]["touch_ledger"])
    eligible = _eligible_touch_count(touch_ledger)
    alpha = float(cfg.get("alpha", 0.05))
    bonferroni_alpha = alpha / max(1, eligible)
    dm_p = dm.get("p_value")
    pass_gate = bool(
        np.isfinite(delta)
        and np.isfinite(se)
        and delta > 2 * se
        and dm_p is not None
        and float(dm_p) < bonferroni_alpha
    )

    qwen_features = [c for c in Xq_train.columns if c.startswith("qwen_")]
    report = {
        "phase": "phase8",
        "holdout_start": str(holdout_start.date()),
        "train_cutoff": str(cutoff.date()),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "auc_base": auc_base,
        "auc_qwen": auc_qwen,
        "delta": delta,
        "bootstrap_se": se,
        "bar_2se": 2 * se if np.isfinite(se) else None,
        "dm": dm,
        "eligible_touches": eligible,
        "bonferroni_alpha": bonferroni_alpha,
        "dm_p_value": dm_p,
        "pass": pass_gate,
        "qwen_features": qwen_features,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(Path(artifacts["phase8"]["results"]), report)
    # Sealed-holdout governance: log the touch even on a failed gate.
    from .touch_ledger import log_touch

    log_touch(
        settings.paths,
        track="Q",
        phase="ablation",
        description="Phase8 Pool D ablation (base vs +qwen)",
        rows_touched=int(test_mask.sum()),
        artifact_ref=str(artifacts["phase8"]["results"]),
        eligible=True,
    )
    if pass_gate:
        # The pass opens a separate economics retest, never an auto-unpause.
        logger.info("Phase8 PASS — economics retest may be opened separately.")
    else:
        logger.info("Phase8 FAIL — keep infra, delete adapters per spec.")
    q_state.set_phase_status("phase8", "passed" if pass_gate else "failed", result=report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 Pool D ablation.")
    args = parser.parse_args()
    print(run_phase8())


if __name__ == "__main__":
    main()
