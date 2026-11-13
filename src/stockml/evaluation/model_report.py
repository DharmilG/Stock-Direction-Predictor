"""Model behaviour report: train / calibration / test, from the fitted booster.

This stage answers "what does the model do, on the data it saw and on the data
it did not" and writes the artifacts the ``#/model`` page reads. It is
read-only with respect to the model: no fitting, no parameter change, no
threshold change. ``results/predictions.parquet`` is only read (as a
consistency check against ``evaluate``), never rewritten.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from ..api.model_diagnostics import (
    CLASSES,
    driver_rows,
    hypotheses,
    plate,
    threshold_sweep,
)
from ..config import Settings, load_settings
from ..models.lightgbm_train import _timestamps, load_training_frame, prepare_xy
from ..utils.hash import stable_hash
from ..utils.io import read_json
from ..utils.logging import setup_logging
from ..utils.state import update_state
from ..validation.time_splits import three_way_date_split
from .metrics import apply_temperature
from .flat_threshold import apply_flat_threshold

SPLITS = ("train", "calibration", "test")
SHAP_SAMPLE = 1000
EXPLAIN_SAMPLE = 600
TOP_K = 12
CURVE_WINDOW = 30


def _write_parquet(frame: pd.DataFrame, path) -> None:
    """Same atomic pattern as utils.io, but without persisting a row index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _split_frame(index, actual, proba, predicted, events: dict) -> pd.DataFrame:
    if isinstance(index, pd.MultiIndex):
        symbols = index.get_level_values("symbol").astype(str)
        dates = index.get_level_values("timestamp")
    else:
        symbols = pd.Series([""] * len(index))
        dates = index
    ordered = np.sort(proba, axis=1)
    frame = pd.DataFrame({
        "date": pd.to_datetime(pd.Index(dates)).date,
        "symbol": np.asarray(symbols),
        "actual_label": actual,
        "predicted_label": predicted,
        "p_down": proba[:, 0],
        "p_flat": proba[:, 1],
        "p_up": proba[:, 2],
        "max_p": proba.max(axis=1),
        "margin": ordered[:, -1] - ordered[:, -2],
        "correct": (actual == predicted).astype(int),
    }, index=index)
    for name, values in events.items():
        frame[name] = values
    return frame


def _stratified(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Keep rare predicted classes in the explanation sample."""
    if len(frame) <= n:
        return frame
    per_class = max(1, n // 3)
    chunks = []
    for c in range(3):
        chunk = frame[frame["predicted_label"] == c]
        if len(chunk):
            chunks.append(chunk.sample(min(per_class, len(chunk)), random_state=seed))
    if not chunks:
        return frame.head(n)
    return pd.concat(chunks).sort_index()


def _load_folds(path, limit: int = 400) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return rows[-limit:]


def _importance(paths, psi: dict, threshold: float, limit: int = 40) -> list[dict]:
    path = paths.results / "shap_feature_importance.csv"
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path).sort_values("mean_abs_shap", ascending=False).head(limit)
    except Exception:
        return []
    rows = []
    for rank, (_, row) in enumerate(frame.iterrows()):
        feature = str(row["feature"])
        value = psi.get(feature)
        rows.append({
            "feature": feature,
            "mean_abs_shap": float(row["mean_abs_shap"]),
            "psi": None if value is None else float(value),
            "drift": bool(value is not None and value >= threshold),
            "in_top15": rank < 15,
        })
    return rows


def _shap_matrix(explainer, sample_x: pd.DataFrame):
    """Normalise (n, f, 3) vs [list of 3 (n, f)] vs binary output."""
    values = explainer.shap_values(sample_x)
    if isinstance(values, list):
        matrix = np.stack([np.asarray(v) for v in values], axis=-1)
    else:
        matrix = np.asarray(values)
        if matrix.ndim == 2:
            matrix = matrix[..., None]
    base = explainer.expected_value
    base = list(np.asarray(base, dtype=float).reshape(-1)) if base is not None else [0.0]
    if len(base) == 1:
        base = base * max(1, matrix.shape[-1])
    return matrix, base


def dataset_signature(settings: Settings, frame: pd.DataFrame) -> str:
    """Fingerprint of the matrix a model was fitted on.

    Shared with the v2 trainer so both sides of a comparison are fingerprinted
    identically: a data refresh then shows up on whichever side is stale, and
    this report can tell whether its own panel still matches the fit.
    """
    index = frame.index
    first = index.get_level_values("timestamp").min() if isinstance(index, pd.MultiIndex) else index.min()
    last = index.get_level_values("timestamp").max() if isinstance(index, pd.MultiIndex) else index.max()
    return stable_hash({
        "symbols": settings.symbols,
        "rows": int(len(frame)),
        "first_timestamp": str(first),
        "last_timestamp": str(last),
        "columns": list(frame.columns),
        "target": {
            "horizon": settings.triple_barrier_horizon,
            "up": settings.triple_barrier_up_mult,
            "down": settings.triple_barrier_down_mult,
            "flat_band": settings.triple_barrier_flat_band,
        },
    })


def run_model_report(settings: Settings | None = None) -> None:
    """Behaviour report: plates, threshold sweeps, SHAP drivers, case samples.

    ``settings`` selects the workspace. Everything below reads the booster, the
    metadata and the calibration file out of ``settings.paths``, so the same code
    describes the v2 twin without a parallel implementation.
    """
    settings = settings or load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "report.log")
    model_path = paths.artifacts / "models" / "lightgbm_final.txt"
    if not model_path.exists():
        raise FileNotFoundError("Trained LightGBM model not found. Run train first.")
    import lightgbm as lgb
    import shap

    started = datetime.now()
    model = lgb.Booster(model_file=str(model_path))
    metadata = read_json(paths.artifacts / "model_metadata.json", {}) or {}
    metadata_features = list(metadata.get("feature_columns") or [])
    if not metadata_features:
        logger.warning("model_metadata.json has no feature_columns; scoring every numeric column in the store.")

    df = load_training_frame(settings)
    signature = dataset_signature(settings, df)
    X, y, _events, feature_cols = prepare_xy(df, columns=metadata_features)
    prepared_features = [c for c in df.columns if c not in ("symbol",)]
    alignment = {
        "model_features": len(metadata_features) if metadata_features else int(X.shape[1]),
        "scored_features": int(X.shape[1]),
        "store_features": len(prepared_features),
        "extra_in_store": max(0, len(prepared_features) - int(X.shape[1])) if metadata_features else 0,
        "missing_from_store": [c for c in (metadata_features or []) if c not in df.columns][:20],
        "pinned_to_metadata": bool(metadata_features),
    }
    train_mask, cal_mask, test_mask, calibration_start, test_start = three_way_date_split(_timestamps(X.index))
    masks = {"train": train_mask, "calibration": cal_mask, "test": test_mask}

    raw = model.predict(X)
    temperature = float(read_json(paths.artifacts / "calibration.json", {"temperature": 1.0}).get("temperature", 1.0))
    proba_all = apply_temperature(raw, temperature)
    pred_all = apply_flat_threshold(proba_all, float(settings.flat_threshold))

    event_cols = {}
    for col in ("event_start", "event_end", "entry_price", "exit_price", "realized_return"):
        if col in df.columns:
            event_cols[col] = df.loc[X.index, col].to_numpy()

    actual_all = y.to_numpy(dtype=int)
    frames = {}
    for name in SPLITS:
        mask = masks[name]
        sub = X.index[mask]
        positions = np.where(mask)[0]
        frame = _split_frame(
            sub,
            actual_all[positions],
            proba_all[positions],
            pred_all[positions],
            {key: values[positions] for key, values in event_cols.items()},
        )
        frames[name] = frame

    combined = pd.concat([frames[name].assign(split=name) for name in SPLITS], ignore_index=True)
    _write_parquet(combined, paths.results / "predictions_splits.parquet")
    logger.info("Wrote %d scored rows across %s.", len(combined), ", ".join(SPLITS))

    plates = {name: plate(frames[name], name, window=CURVE_WINDOW) for name in SPLITS}

    sweep = []
    for name in SPLITS:
        for row in threshold_sweep(frames[name]["actual_label"].to_numpy(), frames[name][["p_down", "p_flat", "p_up"]].to_numpy()):
            sweep.append({"split": name, **row})

    psi_threshold = float(getattr(settings, "drift_psi_threshold", 0.25) or 0.25)
    monitor = read_json(paths.results / "monitoring.json", {}) or {}
    psi = monitor.get("feature_psi") if isinstance(monitor.get("feature_psi"), dict) else {}
    importance = _importance(paths, psi, psi_threshold)

    # --- SHAP: signed contributions per class, and per-row explanations -------
    target_shap = _stratified(frames["test"], SHAP_SAMPLE, settings.random_seed)
    train_shap = _stratified(frames["train"], SHAP_SAMPLE, settings.random_seed + 1)
    shap_rows = []
    driver_table = pd.DataFrame(columns=["class", "feature", "mean_signed", "mean_abs"])
    explain_rows = []
    try:
        explainer = shap.TreeExplainer(model)
        per_class_signed = []
        for split_name, sample in (("test", target_shap), ("train", train_shap)):
            sample_x = X.loc[sample.index]
            matrix, base = _shap_matrix(explainer, sample_x)
            n_classes = matrix.shape[-1]
            means = [matrix[:, :, c].mean(axis=0) for c in range(n_classes)]
            per_class_signed.append(np.stack(means, axis=0))
            subset = sample.head(EXPLAIN_SAMPLE)
            row_positions = {label: pos for pos, label in enumerate(sample.index)}
            for label, row in subset.iterrows():
                pos = row_positions.get(label)
                if pos is None:
                    continue
                for c in range(min(n_classes, 3)):
                    contributions = matrix[pos, :, c]
                    top = np.argsort(-np.abs(contributions))[:TOP_K]
                    for rank, feature_pos in enumerate(top, start=1):
                        value = sample_x.iat[pos, feature_pos]
                        explain_rows.append({
                            "split": split_name,
                            "symbol": row["symbol"],
                            "date": row["date"],
                            "class": CLASSES[c],
                            "rank": rank,
                            "feature": str(X.columns[feature_pos]),
                            "feature_value": float(value) if np.isfinite(value) else None,
                            "contribution": float(contributions[feature_pos]),
                            "base_value": float(base[c]) if c < len(base) else None,
                            "predicted_label": int(row["predicted_label"]),
                            "actual_label": int(row["actual_label"]),
                            "p_down": float(row["p_down"]),
                            "p_flat": float(row["p_flat"]),
                            "p_up": float(row["p_up"]),
                            "max_p": float(row["max_p"]),
                        })
        signed = np.mean(np.stack(per_class_signed, axis=0), axis=0) if per_class_signed else None
        if signed is not None:
            rows = []
            for c in range(min(signed.shape[0], 3)):
                order = np.argsort(-np.abs(signed[c]))[:40]
                for feature_pos in order:
                    rows.append({
                        "class": CLASSES[c],
                        "feature": str(X.columns[feature_pos]),
                        "mean_signed": float(signed[c, feature_pos]),
                        "mean_abs": float(np.abs(signed[c, feature_pos])),
                    })
            driver_table = pd.DataFrame(rows)
            driver_table.to_csv(paths.results / "shap_by_class.csv", index=False)
    except Exception as exc:
        logger.warning("SHAP report skipped: %s", exc)

    if explain_rows:
        _write_parquet(pd.DataFrame(explain_rows), paths.results / "explanations_splits.parquet")
    drivers = driver_rows(driver_table, None, limit=8)

    # --- Consistency against evaluate ---------------------------------------
    consistency = {
        "dataset_signature": signature,
        "metadata_signature": None,
        "signature_matches": None,
        "predictions_max_abs_diff": None,
        "predictions_compared": 0,
        "notes": [],
    }
    consistency["metadata_signature"] = metadata.get("dataset_signature")
    if metadata.get("dataset_signature") is not None:
        consistency["signature_matches"] = metadata["dataset_signature"] == signature
        if not consistency["signature_matches"]:
            consistency["notes"].append(
                "The feature matrix signature differs from the one recorded at training time: "
                "the data changed after the model was fitted, so train-split rows may not be the rows it saw."
            )
    if alignment.get("missing_from_store"):
        consistency["notes"].append(
            f"{len(alignment['missing_from_store'])} trained feature(s) are absent from the store: "
            f"{', '.join(alignment['missing_from_store'][:5])}"
        )
    if alignment.get("pinned_to_metadata") and alignment.get("store_features", 0) > alignment.get("scored_features", 0):
        consistency["notes"].append(
            f"The feature store now has {alignment['store_features']} columns but the booster was trained on "
            f"{alignment['scored_features']}; this report is pinned to the trained set, so a retrain would see "
            "more inputs than the numbers below reflect."
        )
    consistency["feature_alignment"] = alignment
    try:
        reference_path = paths.results / "predictions.parquet"
        if reference_path.exists():
            reference = pd.read_parquet(reference_path)
            # Explicit key merge on a de-indexed copy: `symbol` is both a column
            # and an index level here, and index alignment would silently match
            # on that shared level name and cross-join the dates.
            test_flat = frames["test"].reset_index(drop=True)
            theirs = pd.DataFrame({
                "date": pd.DatetimeIndex(pd.to_datetime(list(reference.index.get_level_values(0)))).normalize(),
                "symbol": reference.index.get_level_values(1).astype(str),
                "p_down_evaluate": reference["p_down"].to_numpy(),
                "predicted_evaluate": reference["predicted_label"].to_numpy(),
            }).drop_duplicates(subset=["date", "symbol"])
            mine = pd.DataFrame({
                "date": pd.DatetimeIndex(pd.to_datetime(list(test_flat["date"]))).normalize(),
                "symbol": test_flat["symbol"].astype(str).to_numpy(),
                "p_down_report": test_flat["p_down"].to_numpy(),
                "predicted_report": test_flat["predicted_label"].to_numpy(),
            })
            joined = mine.merge(theirs, on=["date", "symbol"], how="inner")
            consistency["predictions_compared"] = int(len(joined))
            consistency["window"] = {
                "report_start": str(min(frames["test"]["date"])),
                "report_end": str(max(frames["test"]["date"])),
                "evaluate_start": str(theirs["date"].min().date()) if len(theirs) else None,
                "evaluate_end": str(theirs["date"].max().date()) if len(theirs) else None,
            }
            if len(joined):
                if len(joined) > max(len(mine), len(theirs)):
                    consistency["notes"].append("Duplicate (date, symbol) keys made the comparison ambiguous.")
                else:
                    diff = float(np.abs(joined["p_down_report"] - joined["p_down_evaluate"]).max())
                    mean_diff = float(np.abs(joined["p_down_report"] - joined["p_down_evaluate"]).mean())
                    differing = int((np.abs(joined["p_down_report"] - joined["p_down_evaluate"]) > 1e-6).sum())
                    agree = float((joined["predicted_report"] == joined["predicted_evaluate"]).mean())
                    consistency["predictions_max_abs_diff"] = diff
                    consistency["predictions_mean_abs_diff"] = round(mean_diff, 6)
                    consistency["rows_differing"] = differing
                    consistency["predicted_label_agreement"] = round(agree, 6)
                    consistency["same_window"] = bool(
                        consistency["window"]["report_start"] == consistency["window"]["evaluate_start"]
                        and consistency["window"]["report_end"] == consistency["window"]["evaluate_end"]
                    )
                    if differing:
                        consistency["notes"].append(
                            f"On the {len(joined)} overlapping rows {100.0 * differing / len(joined):.2f}% of "
                            f"probabilities differ (max {diff:.4f}, mean {mean_diff:.4f}) while "
                            f"{100.0 * agree:.2f}% of decisions agree: the same booster is scoring re-built "
                            "feature values, or a different model was used for one of the two runs."
                        )
                    if not consistency["same_window"]:
                        consistency["notes"].append(
                            "The report's test window differs from the evaluated one because the data moved after "
                            "training; the recorded metrics describe the earlier window, not these rows."
                        )
            else:
                consistency["notes"].append("No overlapping (date, symbol) rows with results/predictions.parquet.")
        else:
            consistency["notes"].append("results/predictions.parquet is missing; run evaluate to cross-check.")
    except Exception as exc:
        consistency["notes"].append(f"Consistency check skipped: {exc}")

    train_rule = plates["train"].get("rule") or {}
    test_rule = plates["test"].get("rule") or {}
    comparison = {
        "accuracy_gap_pp": (
            round((train_rule["accuracy_argmax"] - test_rule["accuracy_argmax"]) * 100.0, 2)
            if train_rule.get("accuracy_argmax") is not None and test_rule.get("accuracy_argmax") is not None
            else None
        ),
        "separation_gap": (
            round((plates["train"]["profile"].get("separation") or 0.0) - (plates["test"]["profile"].get("separation") or 0.0), 4)
        ),
        "recall_gap": {
            name: (
                round((plates["train"]["confusion"]["recall"].get(name) or 0.0) - (plates["test"]["confusion"]["recall"].get(name) or 0.0), 4)
            )
            for name in CLASSES
        },
    }

    hypotheses_rows = hypotheses(
        plates=plates,
        sweep=sweep,
        importance=importance,
        tau_flat=float(settings.flat_threshold),
        psi_threshold=psi_threshold,
        calibration=read_json(paths.artifacts / "calibration.json", {}) or {},
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round((datetime.now() - started).total_seconds(), 1),
        "meta": {
            "model_type": "lightgbm_multiclass",
            "classes": {str(k): v for k, v in ((0, "DOWN"), (1, "FLAT"), (2, "UP"))},
            "feature_count": int(X.shape[1]),
            "rows": {name: int(len(frames[name])) for name in SPLITS},
            "train_start": str(min(frames["train"]["date"])),
            "calibration_start": str(calibration_start.date()),
            "test_start": str(test_start.date()),
            "test_end": str(max(frames["test"]["date"])),
            "tau_flat": float(settings.flat_threshold),
            "temperature": temperature,
            "dataset_signature": signature,
            "best_params_hash": metadata.get("best_params_hash"),
            "trained_train_accuracy": metadata.get("train_accuracy"),
            "trained_test_accuracy": metadata.get("final_test_accuracy"),
            "psi_threshold": psi_threshold,
            "shap_method": "tree_shap",
            "shap_sample_rows": int(len(target_shap)),
            "explain_sample_rows": int(len(explain_rows)),
            "report_version": 1,
            "feature_columns": list(feature_cols),
        },
        "consistency": consistency,
        "splits": plates,
        "comparison": comparison,
        "thresholds": sweep,
        "folds": _load_folds(paths.checkpoints / "trials" / "fold_diagnostics.jsonl"),
        "importance": importance,
        "drivers": drivers,
        "hypotheses": hypotheses_rows,
    }
    paths.results.mkdir(parents=True, exist_ok=True)
    tmp = (paths.results / "model_report.json").with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, paths.results / "model_report.json")

    diff = consistency.get("predictions_max_abs_diff")
    if diff is None:
        agreement = "not compared"
    else:
        agreement = "consistent" if diff <= 1e-6 else "INCONSISTENT"
    logger.info(
        "Report: train %.4f | calibration %.4f | test %.4f (argmax) | mode=%s | %s vs evaluate",
        train_rule.get("accuracy_argmax") or 0.0,
        (plates["calibration"].get("rule") or {}).get("accuracy_argmax") or 0.0,
        test_rule.get("accuracy_argmax") or 0.0,
        plates["test"]["mode"]["mode"],
        agreement,
    )
    update_state(
        paths.state,
        "model_report",
        status="complete",
        report=str(paths.results / "model_report.json"),
        mode=plates["test"]["mode"]["mode"],
    )


if __name__ == "__main__":
    run_model_report()
