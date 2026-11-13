"""Per-row explanations for the latest scored row of each symbol.

The dashboard can only *show* a reason if one was computed and persisted, so
this command writes ``results/explanations_serving.parquet`` — the same shape
as the split-sampled explanations the ``model-report`` stage writes. It never
changes a prediction: probabilities are recomputed exactly as
``run_prediction`` does and are stored alongside the contributions so the page
can prove the waterfall matches the serving row.
"""

from __future__ import annotations

import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..api.model_diagnostics import CLASSES
from ..config import Settings, load_settings
from ..evaluation.flat_threshold import apply_flat_threshold
from ..evaluation.metrics import apply_temperature
from ..utils.io import read_json
from ..utils.logging import setup_logging
from ..utils.state import update_state
from .predict import _load_latest_features


def run_explain(symbol: str | None = None, top_k: int = 12, settings: Settings | None = None) -> pd.DataFrame:
    """Per-row SHAP waterfalls for the serving row.

    ``settings`` selects the workspace (default: the incumbent tree), so the v2
    stages can explain the v2 booster with the same code.
    """
    settings = settings or load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "explain.log")
    model_path = paths.artifacts / "models" / "lightgbm_final.txt"
    metadata_path = paths.artifacts / "model_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Trained model artifacts are missing. Run training first.")
    import shap

    model = lgb.Booster(model_file=str(model_path))
    metadata = read_json(metadata_path, {}) or {}
    feature_cols = metadata.get("feature_columns", [])
    temperature = float((read_json(paths.artifacts / "calibration.json", {"temperature": 1.0}) or {}).get("temperature", 1.0))
    explainer = shap.TreeExplainer(model)

    symbols = [symbol] if symbol else list(settings.symbols)
    records = []
    for name in symbols:
        try:
            row = _load_latest_features(settings, name)
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
            continue
        X = row[[c for c in feature_cols if c in row.columns]].select_dtypes(include=[np.number])
        missing = [c for c in feature_cols if c not in X.columns]
        if missing:
            logger.warning("Skipping %s: %d production feature(s) missing (%s)", name, len(missing), missing[:5])
            continue
        X = X[feature_cols].replace([np.inf, -np.inf], np.nan)
        if X.isna().all().all():
            logger.warning("Skipping %s: latest row has no usable features.", name)
            continue
        proba = apply_temperature(model.predict(X), temperature)[0]
        label_idx = int(apply_flat_threshold(proba.reshape(1, -1), float(settings.flat_threshold))[0])
        values = explainer.shap_values(X)
        matrix = np.stack([np.asarray(v) for v in values], axis=-1) if isinstance(values, list) else np.asarray(values)
        if matrix.ndim == 2:
            matrix = matrix[..., None]
        base_values = explainer.expected_value
        base_values = list(np.asarray(base_values, dtype=float).reshape(-1)) if base_values is not None else [0.0]
        if len(base_values) == 1:
            base_values = base_values * max(1, matrix.shape[-1])

        date = pd.Timestamp(row.index[-1]).date().isoformat()
        for c in range(min(matrix.shape[-1], 3)):
            contributions = matrix[0, :, c]
            top = np.argsort(-np.abs(contributions))[:top_k]
            for rank, position in enumerate(top, start=1):
                value = X.iat[0, position]
                records.append({
                    "split": "serving",
                    "symbol": name,
                    "date": date,
                    "class": CLASSES[c],
                    "rank": rank,
                    "feature": str(X.columns[position]),
                    "feature_value": float(value) if np.isfinite(value) else None,
                    "contribution": float(contributions[position]),
                    "base_value": float(base_values[c]) if c < len(base_values) else None,
                    "predicted_label": label_idx,
                    "actual_label": -1,
                    "max_p": float(proba.max()),
                    "p_down": float(proba[0]),
                    "p_flat": float(proba[1]),
                    "p_up": float(proba[2]),
                })

    out = pd.DataFrame(records)
    if out.empty:
        raise RuntimeError("No explanations were produced; check the feature store.")
    path = paths.results / "explanations_serving.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    logger.info("Wrote %d contribution rows for %d symbol(s) to %s.", len(out), out["symbol"].nunique(), path)
    update_state(paths.state, "explain", status="complete", output=str(path), symbols=int(out["symbol"].nunique()))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--top-k", type=int, default=12, dest="top_k")
    args = parser.parse_args()
    run_explain(symbol=args.symbol, top_k=args.top_k)
