from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from ..config import load_settings
from ..utils.io import atomic_write_json, atomic_write_parquet, read_json
from ..utils.logging import setup_logging
from ..utils.state import update_state
from .macro import fetch_macro
from .providers import get_provider


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "timestamp"
    return out[~out.index.duplicated(keep="last")].sort_index()


def _merge_history(existing: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    fresh = _normalize_frame(fresh)
    if existing is None or existing.empty:
        fresh.attrs = {}
        return fresh
    existing = _normalize_frame(existing)
    # Never propagate arbitrary user/provider metadata into concat inputs.
    # pandas compares attrs during concat finalization, and attrs may contain
    # non-scalar objects such as DataFrames.
    existing.attrs = {}
    fresh.attrs = {}
    combined = pd.concat([existing, fresh], axis=0)
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def _latest_data_date(path, fallback: date) -> date:
    if not path.exists():
        return fallback
    try:
        frame = pd.read_parquet(path)
        if frame.empty:
            return fallback
        return pd.Timestamp(frame.index.max()).date()
    except Exception:
        return fallback


def _fetch_incremental(provider, symbol: str, raw_path, action_path, initial_start: date, end: date, interval: str, logger):
    existing = None
    if raw_path.exists():
        try:
            existing = _normalize_frame(pd.read_parquet(raw_path))
        except Exception as exc:
            logger.warning("Could not read existing %s: %s; rebuilding.", raw_path, exc)

    if existing is not None and not existing.empty:
        last_date = existing.index.max().date()
        fetch_start = last_date + timedelta(days=1)
        if fetch_start > end:
            logger.info("%s is already current through %s; skipping.", symbol, last_date)
            return existing
    else:
        fetch_start = initial_start

    logger.info("Fetching %s from %s to %s", symbol, fetch_start, end)
    fresh = provider.fetch_ohlcv(symbol, fetch_start, end, interval)
    merged = _merge_history(existing, fresh)
    atomic_write_parquet(merged, raw_path)

    actions = provider.fetch_actions(symbol, initial_start, end)
    if actions is not None and not actions.empty:
        existing_actions = None
        if action_path.exists():
            try:
                existing_actions = pd.read_parquet(action_path)
            except Exception:
                existing_actions = None
        actions_merged = _merge_history(existing_actions, actions)
        atomic_write_parquet(actions_merged, action_path)
    elif not action_path.exists():
        atomic_write_parquet(pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp")), action_path)

    return merged


def _fetch_macro_incremental(start: date, end: date, path, logger) -> None:
    existing = None
    if path.exists():
        try:
            existing = _normalize_frame(pd.read_parquet(path))
        except Exception as exc:
            logger.warning("Could not read macro history: %s; rebuilding.", exc)
    fetch_start = start
    if existing is not None and not existing.empty:
        last = existing.index.max().date()
        if last >= end - timedelta(days=3):
            logger.info("Macro data is current; skipping.")
            return
        fetch_start = last + timedelta(days=1)
    logger.info("Fetching macro data %s -> %s", fetch_start, end)
    fresh = fetch_macro(fetch_start, end)
    merged = _merge_history(existing, fresh)
    atomic_write_parquet(merged, path)


def run_fetch() -> None:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "fetch.log")
    provider = get_provider(settings.provider)

    end = date.today()
    start = end - timedelta(days=round(settings.history_years * 365.25))
    manifest_path = paths.state / "data_manifest.json"
    manifest = read_json(manifest_path, default={}) or {}

    _fetch_macro_incremental(start, end, paths.data_raw / "macro.parquet", logger)
    actions_dir = paths.data_raw / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)

    for i, symbol in enumerate(settings.symbols):
        safe = _safe_symbol(symbol)
        out = paths.data_raw / f"{safe}.parquet"
        action_out = actions_dir / f"{safe}.parquet"
        frame = _fetch_incremental(provider, symbol, out, action_out, start, end, settings.interval, logger)
        if frame.empty:
            raise RuntimeError(f"No data available for {symbol}")
        manifest[symbol] = {
            "path": str(out.relative_to(paths.root)),
            "last_date": str(frame.index.max().date()),
            "rows": int(len(frame)),
        }
        atomic_write_json(manifest_path, manifest)
        # Politeness delay between symbols keeps rapid 16y backfills under the
        # provider rate limit (JSWSTEEL.NS tripped it at ~4s/symbol with no gap).
        if i < len(settings.symbols) - 1:
            __import__("time").sleep(3)

    update_state(paths.state, "fetch", status="complete", start=str(start), end=str(end), symbols=settings.symbols, manifest=str(manifest_path))
    logger.info("Data fetch complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch 14-18 years of daily market data and macro data.")
    parser.parse_args()
    run_fetch()


if __name__ == "__main__":
    main()
