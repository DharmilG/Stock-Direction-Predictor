from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_settings
from ..utils.io import atomic_write_parquet
from ..utils.logging import setup_logging
from ..utils.state import update_state


# F-12 default rank sources. Overridable via config features.xs_rank_sources.
# All are stationary transforms; ranks are computed per session across the
# universe with NO shift (same-day closes all known by 15:30 IST, §7.3).
DEFAULT_XS_RANK_SOURCES = [
    "ret_5", "ret_20", "ret_60", "momentum_20",
    "volatility_20", "volatility_60",
    "relative_volume_20", "volume_z_20",
    "rsi_14", "macd_hist", "atr_pct_14", "bb_position", "obv_z20",
]


def compute_xs_ranks(panel: pd.DataFrame, sources: list[str] | None = None) -> pd.DataFrame:
    """Add `{source}_xs_rank` percentile ranks per timestamp. Pure function for tests."""
    out = panel.copy()
    for col in (sources or DEFAULT_XS_RANK_SOURCES):
        if col in out.columns:
            out[f"{col}_xs_rank"] = out.groupby("timestamp")[col].rank(pct=True)
    return out


def _load_membership(ref_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long (symbol, bucket, from, to) tables for sector and group maps.

    Missing files → empty tables (F-11 silently skips; old behavior kept).
    """
    sector_rows: list[dict] = []
    try:
        sm = json.loads((ref_dir / "sector_map.json").read_text(encoding="utf-8"))
        for sym, spec in sm.items():
            if str(sym).startswith("_") or not isinstance(spec, dict):
                continue
            sector_rows.append({
                "symbol": sym, "bucket": str(spec.get("sector", "Unknown")),
                "from": pd.Timestamp(spec.get("from") or "2010-01-01"),
                "to": pd.Timestamp(spec.get("to")) if spec.get("to") else pd.Timestamp.max,
            })
    except FileNotFoundError:
        pass
    group_rows: list[dict] = []
    try:
        gm = json.loads((ref_dir / "group_map.json").read_text(encoding="utf-8"))
        for gid, gspec in gm.items():
            if str(gid).startswith("_") or not isinstance(gspec, dict):
                continue
            for m in gspec.get("members", []):
                group_rows.append({
                    "symbol": m.get("symbol"), "bucket": str(gid),
                    "from": pd.Timestamp(m.get("from") or "2010-01-01"),
                    "to": pd.Timestamp(m.get("to")) if m.get("to") else pd.Timestamp.max,
                })
    except FileNotFoundError:
        pass
    return pd.DataFrame(sector_rows), pd.DataFrame(group_rows)


def add_group_sector_features(
    panel: pd.DataFrame,
    sector_members: pd.DataFrame,
    group_members: pd.DataFrame,
) -> pd.DataFrame:
    """F-11: sector/group-relative features. Pure function for tests.

    Point-in-time: membership rows carry from/to; a row only joins the bucket
    active at its timestamp. All inputs are same-day closes known by 15:30
    IST (§7.3) — no shift. Single-member buckets yield NaN dispersion (kept;
    LightGBM native missing).
    """
    out = panel.copy()
    if "timestamp" not in out.columns:
        return out
    out["_ts"] = pd.to_datetime(out["timestamp"])
    for kind, members, prefix in (
        ("sector", sector_members, "sector"),
        ("group", group_members, "group"),
    ):
        if members.empty or "ret_20" not in out.columns:
            continue
        m = members.copy()
        m["_from"] = pd.to_datetime(m["from"])
        m["_to"] = pd.to_datetime(m["to"])
        j = out.merge(m[["symbol", "bucket", "_from", "_to"]], on="symbol", how="left")
        active = (j["_ts"] >= j["_from"]) & (j["_ts"] <= j["_to"])
        j.loc[~active, "bucket"] = np.nan
        key = (j["timestamp"].astype(str) + " " + j["bucket"].astype(str))
        has_bucket = j["bucket"].notna().to_numpy()
        if "ret_5" in out.columns:
            mean5 = j.groupby(key)["ret_5"].transform("mean")
            out[f"ret_5_minus_{prefix}"] = np.where(has_bucket, j["ret_5"] - mean5, np.nan)
        mean20 = j.groupby(key)["ret_20"].transform("mean")
        out[f"ret_20_minus_{prefix}"] = np.where(has_bucket, j["ret_20"] - mean20, np.nan)
        out[f"{prefix}_xs_rank"] = np.where(
            has_bucket, j.groupby(key)["ret_20"].rank(pct=True).to_numpy(), np.nan)
        out[f"{prefix}_dispersion"] = np.where(
            has_bucket, j.groupby(key)["ret_20"].transform("std").to_numpy(), np.nan)
    out = out.drop(columns=["_ts"], errors="ignore")
    return out


def run_cross_sectional() -> None:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "cross_sectional.log")
    frames = []
    for symbol in settings.symbols:
        safe = symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        path = paths.data_features / f"{safe}_features.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path).copy()
        df.attrs = {}
        df["symbol"] = symbol
        frames.append(df)
    if not frames:
        raise RuntimeError("No feature files found for cross-sectional processing.")
    panel = pd.concat(frames, axis=0).reset_index().rename(columns={"index": "timestamp"})
    sources = list(settings.features.get("xs_rank_sources", DEFAULT_XS_RANK_SOURCES))
    panel = compute_xs_ranks(panel, sources)
    n_ranks = sum(1 for c in sources if f"{c}_xs_rank" in panel.columns)
    logger.info("Computed %d cross-sectional rank families over %d symbols.", n_ranks, len(frames))
    ref_dir = paths.data_raw.parent / "reference"
    sector_members, group_members = _load_membership(ref_dir)
    n_before = panel.shape[1]
    panel = add_group_sector_features(panel, sector_members, group_members)
    logger.info("F-11 group/sector features added (%d new cols; sector buckets=%d group buckets=%d).",
                panel.shape[1] - n_before,
                sector_members["bucket"].nunique() if not sector_members.empty else 0,
                group_members["bucket"].nunique() if not group_members.empty else 0)
    panel = panel.sort_values(["symbol", "timestamp"]).set_index(["timestamp", "symbol"])
    for symbol, sdf in panel.groupby(level="symbol", sort=False):
        out_symbol = symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        out = paths.data_features / f"{out_symbol}_features.parquet"
        sdf = sdf.droplevel("symbol")
        atomic_write_parquet(sdf, out)
        logger.info("Updated cross-sectional features for %s", symbol)
    update_state(paths.state, "cross_sectional", status="complete", symbols=settings.symbols)


if __name__ == "__main__":
    run_cross_sectional()
