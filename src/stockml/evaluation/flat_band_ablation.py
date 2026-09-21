from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..config import load_settings
from ..targets.triple_barrier import triple_barrier_labels
from ..utils.logging import setup_logging
from ..utils.state import update_state

# B-02/F-23: the FLAT class is dead at flat_band=0.25 (3.7% of labels, F1=0.0).
# Sweep the band and pick the value giving ~25-40% FLAT (§8.2 Option A).
CANDIDATE_BANDS = [0.25, 0.5, 0.75, 1.0]
FLAT_TARGET_LO, FLAT_TARGET_HI = 0.25, 0.40


def scan_flat_bands(
    price_frames: dict[str, pd.DataFrame],
    bands: list[float] | None = None,
    horizon: int = 5,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Label-share scan per flat_band. Pure computation over price frames (testable)."""
    rows = []
    for band in (bands or CANDIDATE_BANDS):
        counts = {0: 0, 1: 0, 2: 0}
        total = 0
        for symbol, price in price_frames.items():
            labels = triple_barrier_labels(
                price, horizon=horizon, volatility_window=volatility_window, flat_band=float(band)
            )
            mapped = labels["label_raw"].map({-1: 0, 0: 1, 1: 2}).dropna().astype(int)
            for k, v in mapped.value_counts().items():
                counts[int(k)] += int(v)
            total += int(len(mapped))
        flat_share = counts[1] / total if total else 0.0
        rows.append({
            "flat_band": float(band),
            "n_total": total,
            "down_share": counts[0] / total if total else 0.0,
            "flat_share": flat_share,
            "up_share": counts[2] / total if total else 0.0,
        })
    return pd.DataFrame(rows)


def pick_flat_band(scan: pd.DataFrame) -> float:
    """Band whose FLAT share falls in [0.25, 0.40]; else closest to 0.30."""
    in_band = scan[(scan["flat_share"] >= FLAT_TARGET_LO) & (scan["flat_share"] <= FLAT_TARGET_HI)]
    if not in_band.empty:
        return float(in_band.iloc[0]["flat_band"])
    target = 0.5 * (FLAT_TARGET_LO + FLAT_TARGET_HI)
    return float(scan.iloc[(scan["flat_share"] - target).abs().argsort().iloc[0]]["flat_band"])


def run_flat_band_ablation() -> None:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "flat_ablation.log")
    frames: dict[str, pd.DataFrame] = {}
    for symbol in settings.symbols:
        safe = symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        raw_path = paths.data_raw / f"{safe}.parquet"
        if raw_path.exists():
            frames[symbol] = pd.read_parquet(raw_path)
    if not frames:
        raise RuntimeError("No raw price files found. Run fetch first.")
    scan = scan_flat_bands(
        frames,
        horizon=settings.triple_barrier_horizon,
        volatility_window=settings.triple_barrier_vol_window,
    )
    winner = pick_flat_band(scan)
    logger.info("flat_band scan:\n%s", scan.to_string(index=False))
    logger.info("Selected flat_band=%.2f", winner)

    cfg_path = paths.root / "config" / "config.yaml"
    text = cfg_path.read_text(encoding="utf-8")
    text, n = re.subn(r"(?m)^(\s*flat_band:\s*)[0-9.]+", rf"\g<1>{winner}", text)
    if n == 0:
        raise RuntimeError("Could not find flat_band in config.yaml")
    cfg_path.write_text(text, encoding="utf-8")

    # Label change invalidates every processed dataset: force a rebuild on next labels run.
    stale = list(paths.data_processed.glob("*_dataset.parquet"))
    for old in stale:
        old.unlink()
    logger.info("Cleared %d processed datasets; labels will rebuild with flat_band=%.2f.", len(stale), winner)
    update_state(paths.state, "flat_ablation", status="complete", flat_band=winner,
                 scan=scan.to_dict(orient="records"))


if __name__ == "__main__":
    run_flat_band_ablation()
