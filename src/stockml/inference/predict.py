from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import load_settings
from ..models.lightgbm_train import prepare_xy
from ..evaluation.flat_threshold import apply_flat_threshold
from ..evaluation.metrics import apply_temperature
from ..utils.io import read_json
from ..utils.logging import setup_logging
from ..utils.state import update_state


def _load_latest_features(settings, symbol: str) -> pd.DataFrame:
    safe = symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
    path = settings.paths.data_features / f"{safe}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing features for {symbol}: {path}")
    return pd.read_parquet(path).sort_index().tail(1)


def run_prediction() -> pd.DataFrame:
    settings = load_settings()
    logger = setup_logging(settings.paths.logs / "predict.log")
    model_path = settings.paths.artifacts / "models" / "lightgbm_final.txt"
    metadata_path = settings.paths.artifacts / "model_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Trained model artifacts are missing. Run training first.")
    model = lgb.Booster(model_file=str(model_path))
    metadata = read_json(metadata_path, {})
    feature_cols = metadata.get("feature_columns", [])
    calibration = read_json(settings.paths.artifacts / "calibration.json", {"temperature": 1.0})

    records = []
    for symbol in settings.symbols:
        row = _load_latest_features(settings, symbol)
        X = row[[c for c in feature_cols if c in row.columns]].select_dtypes(include=[np.number])
        missing = [c for c in feature_cols if c not in X.columns]
        if missing:
            raise RuntimeError(f"Missing production features for {symbol}: {missing[:10]}")
        X = X[feature_cols].replace([np.inf, -np.inf], np.nan)
        # G1: group/delivery columns are NaN by design where coverage is
        # absent (B-07). Training rows carry the same NaNs and LightGBM
        # handles missing natively — predict must accept them too, else
        # train/serve skew. Only hard-fail when ALL values are missing.
        nan_cols = X.columns[X.isna().any()].tolist()
        if nan_cols:
            logger.warning("Predicting %s with %d NaN feature(s): %s", symbol, len(nan_cols), nan_cols[:8])
        if X.isna().all().all():
            raise RuntimeError(f"Latest row for {symbol} has missing features.")
        raw = model.predict(X)
        proba = apply_temperature(raw, float(calibration.get("temperature", 1.0)))[0]
        label_idx = int(apply_flat_threshold(proba.reshape(1, -1), float(settings.flat_threshold))[0])
        label = ["DOWN", "FLAT", "UP"][label_idx]
        records.append({
            "timestamp": row.index[-1],
            "symbol": symbol,
            "prediction": label,
            "p_down": float(proba[0]),
            "p_flat": float(proba[1]),
            "p_up": float(proba[2]),
            "confidence": float(proba.max()),
        })
    out = pd.DataFrame(records)
    out.to_csv(settings.paths.results / "latest_predictions.csv", index=False)
    logger.info("Generated %d current predictions.", len(out))
    update_state(settings.paths.state, "predict", status="complete", output=str(settings.paths.results / "latest_predictions.csv"))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", dest="print_output")
    args = parser.parse_args()
    result = run_prediction()
    if args.print_output:
        print(result.to_string(index=False))
