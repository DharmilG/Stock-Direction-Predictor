"""Repair internal holes in the raw OHLCV panel.

``run_fetch`` only ever *appends* from ``last_date + 1``, so a session missing in
the middle of a series can never be healed by refreshing. This module finds those
holes and re-downloads a short window around them, merging the rows back in.
"""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from ..config import Paths, load_settings
from ..utils.io import atomic_write_parquet
from ..utils.logging import setup_logging
from ..utils.state import update_state
from .providers import get_provider

_POLITENESS_SECONDS = 3
DEFAULT_WINDOW_SESSIONS = 750


def safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")


def calendar_sessions(paths: Paths) -> pd.DatetimeIndex:
    path = paths.root / "data" / "reference" / "trading_calendar.parquet"
    if not path.is_file():
        return pd.DatetimeIndex([])
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return pd.DatetimeIndex([])
    if "session" not in frame.columns:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(frame["session"], errors="coerce")).dropna().normalize().sort_values()


def missing_sessions(calendar: pd.DatetimeIndex, dates: pd.DatetimeIndex, window_sessions: int) -> pd.DatetimeIndex:
    """Sessions a symbol lacks inside its own coverage (pre-listing/holidays excluded)."""
    if len(calendar) == 0 or len(dates) == 0:
        return pd.DatetimeIndex([])
    window = calendar[-window_sessions:] if window_sessions > 0 else calendar
    window = window[(window >= dates.min()) & (window <= dates.max())]
    return window.difference(dates)


def symbol_dates(path: Path) -> pd.DatetimeIndex:
    try:
        values = pq.read_table(path, columns=["timestamp"]).column(0).to_pandas()
    except Exception:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(values, errors="coerce")).dropna().normalize().sort_values()


def scan_holes(paths: Paths, window_sessions: int = DEFAULT_WINDOW_SESSIONS) -> list[dict[str, Any]]:
    """Every internal hole across the raw price panel (no network access)."""
    calendar = calendar_sessions(paths)
    holes: list[dict[str, Any]] = []
    for path in sorted(p for p in paths.data_raw.glob("*.parquet") if p.stem != "macro"):
        dates = symbol_dates(path)
        missing = missing_sessions(calendar, dates, window_sessions)
        if len(missing) == 0:
            continue
        holes.append({
            "symbol": path.stem,
            "file": path.name,
            "missing_sessions": int(len(missing)),
            "first_missing": missing.min().date().isoformat(),
            "last_missing": missing.max().date().isoformat(),
            "missing_dates": [d.date().isoformat() for d in missing[:20]],
            "window_sessions": int(min(len(calendar), window_sessions)),
        })
    return holes


def _merge(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([existing, fresh])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def run_heal_holes(window_sessions: int = DEFAULT_WINDOW_SESSIONS, dry_run: bool = False) -> dict[str, Any]:
    """Re-download a short window for each symbol with holes and merge it in."""
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "heal.log")
    holes = scan_holes(paths, window_sessions)
    if not holes:
        logger.info("No internal holes found in the raw price panel.")
        update_state(paths.state, "heal", status="complete", holes=0, repaired=0, unrepaired=0)
        return {"status": "complete", "holes": 0, "repaired": 0, "unrepaired": 0, "symbols": []}

    if dry_run:
        logger.info("Dry run: %d symbols with holes, nothing downloaded.", len(holes))
        return {"status": "dry-run", "holes": sum(h["missing_sessions"] for h in holes), "symbols": holes}

    provider = get_provider(settings.provider)
    repaired: list[dict[str, Any]] = []
    unrepaired: list[dict[str, Any]] = []

    for index, hole in enumerate(holes):
        path = paths.data_raw / hole["file"]
        start = pd.Timestamp(hole["first_missing"]).date() - timedelta(days=3)
        end = pd.Timestamp(hole["last_missing"]).date() + timedelta(days=1)
        try:
            existing = pd.read_parquet(path)
            fresh = provider.fetch_ohlcv(hole["symbol"], start, end)
            merged = _merge(existing, fresh)
            before = len(missing_sessions(calendar_sessions(paths), pd.DatetimeIndex(existing.index), window_sessions))
            after = len(missing_sessions(calendar_sessions(paths), pd.DatetimeIndex(merged.index), window_sessions))
            atomic_write_parquet(merged, path)
            record = {
                "symbol": hole["symbol"],
                "window": [start.isoformat(), end.isoformat()],
                "rows_added": int(len(merged) - len(existing)),
                "holes_before": int(before),
                "holes_after": int(after),
            }
            (repaired if after < before else unrepaired).append(record)
            logger.info("Healed %s: %d -> %d holes (%d rows added).", hole["symbol"], before, after, record["rows_added"])
        except Exception as exc:
            logger.warning("Heal failed for %s: %s", hole["symbol"], exc)
            unrepaired.append({"symbol": hole["symbol"], "error": str(exc)})
        if index < len(holes) - 1:
            time.sleep(_POLITENESS_SECONDS)

    summary = {
        "status": "complete" if not unrepaired else "partial",
        "holes": sum(h["missing_sessions"] for h in holes),
        "repaired": len(repaired),
        "unrepaired": len(unrepaired),
        "repaired_detail": repaired,
        "unrepaired_detail": unrepaired,
    }
    update_state(paths.state, "heal", status=summary["status"], **{
        "holes": summary["holes"],
        "repaired": summary["repaired"],
        "unrepaired": summary["unrepaired"],
    })
    logger.info("Heal complete: %d repaired, %d still open.", len(repaired), len(unrepaired))
    return summary
