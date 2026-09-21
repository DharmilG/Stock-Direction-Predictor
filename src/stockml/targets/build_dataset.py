from __future__ import annotations

import argparse

import pandas as pd

from ..config import load_settings
from ..targets.triple_barrier import make_labeled_dataset
from ..utils.io import atomic_write_parquet
from ..utils.logging import setup_logging
from ..utils.state import update_state


def run_build_dataset() -> None:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "labels.log")
    completed = []
    for symbol in settings.symbols:
        safe = symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        feature_path = paths.data_features / f"{safe}_features.parquet"
        raw_path = paths.data_raw / f"{safe}.parquet"
        out = paths.data_processed / f"{safe}_dataset.parquet"
        if out.exists():
            try:
                feature_mtime = feature_path.stat().st_mtime
                if out.stat().st_mtime >= feature_mtime:
                    logger.info("Dataset for %s is current; skipping.", symbol)
                    completed.append(symbol)
                    continue
            except Exception:
                pass
        features = pd.read_parquet(feature_path)
        raw = pd.read_parquet(raw_path)
        dataset = make_labeled_dataset(features, raw, settings)
        atomic_write_parquet(dataset, out)
        completed.append(symbol)
        logger.info("Built %s: %d rows, %d columns", symbol, len(dataset), len(dataset.columns))
    update_state(paths.state, "labels", status="complete", symbols=completed)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run_build_dataset()
