from __future__ import annotations

"""I0-D1a: label-grid dry run (read-only, no training).

Relabels raw price data on a horizon × threshold grid using the existing
triple-barrier code and reports class balance + realized-return separation
per cell. No model, no predictions, no test-label reads of the sealed
holdout beyond what labels already encode (label generation only).

Grid: horizon ∈ {1, 5, 10, 20} × upper/lower multiplier ∈ {0.5, 1.0, 1.5},
volatility_window=20, flat_band=1.0 (current operating point).
"""

import argparse

import numpy as np
import pandas as pd

from ..config import load_settings
from ..targets.triple_barrier import triple_barrier_labels
from ..utils.io import atomic_write_json, read_json
from ..utils.logging import setup_logging
from ..utils.state import update_state

HORIZONS = [1, 5, 10, 20]
MULTS = [0.5, 1.0, 1.5]


def cell_stats(frames: dict[str, pd.DataFrame], horizon: int, mult: float) -> dict:
    """Aggregate label stats for one grid cell over price frames. Pure core."""
    counts = {0: 0, 1: 0, 2: 0}
    rets: dict[int, list] = {0: [], 1: [], 2: []}
    for symbol, price in frames.items():
        labels = triple_barrier_labels(
            price, horizon=int(horizon), volatility_window=20,
            upper_multiplier=float(mult), lower_multiplier=float(mult),
            flat_band=1.0,
        )
        mapped = labels["label_raw"].map({-1: 0, 0: 1, 1: 2}).dropna().astype(int)
        for k, v in mapped.value_counts().items():
            counts[int(k)] += int(v)
        sub = labels.loc[mapped.index]
        for c in (0, 1, 2):
            vals = pd.to_numeric(sub.loc[mapped == c, "realized_return"], errors="coerce").dropna()
            if len(vals):
                # Sample for artifact size (median/std only).
                rets[c].extend(vals.iloc[:: max(1, len(vals) // 2000)].tolist())
    total = sum(counts.values())
    out = {"horizon": int(horizon), "mult": float(mult), "n_total": int(total)}
    for c, name in ((0, "down"), (1, "flat"), (2, "up")):
        r = np.asarray(rets[c], dtype=float)
        out[f"{name}_share"] = float(counts[c] / total) if total else 0.0
        out[f"{name}_ret_med"] = float(np.median(r)) if len(r) else None
        out[f"{name}_ret_std"] = float(np.std(r)) if len(r) else None
    return out


def run_label_grid(horizons=None, mults=None) -> dict:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "label_grid.log")
    frames: dict[str, pd.DataFrame] = {}
    for symbol in settings.symbols:
        safe = symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        path = paths.data_raw / f"{safe}.parquet"
        if path.exists():
            frames[symbol] = pd.read_parquet(path)
    cells = []
    for h in (horizons or HORIZONS):
        for m in (mults or MULTS):
            cell = cell_stats(frames, h, m)
            cells.append(cell)
            logger.info("h=%d mult=%.1f n=%d down=%.3f flat=%.3f up=%.3f up_med=%s",
                        h, m, cell["n_total"], cell["down_share"], cell["flat_share"],
                        cell["up_share"], str(cell["up_ret_med"]))
    result = {"cells": cells, "symbols": len(frames)}
    atomic_write_json(paths.artifacts / "label_grid.json", result)
    update_state(paths.state, "label_grid", status="complete")
    logger.info("Label grid complete (%d cells).", len(cells))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="I0-D1a: label grid dry run.")
    parser.parse_args()
    run_label_grid()


if __name__ == "__main__":
    main()
