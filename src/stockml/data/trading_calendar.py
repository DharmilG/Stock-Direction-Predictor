from __future__ import annotations

"""NSE trading calendar builder (F-02).

Sessions = days with NIFTY data in data/raw/macro.parquet (realized
sessions — an article can only influence a session that occurred) plus any
weekday present in yfinance price data. Stored as
data/reference/trading_calendar.parquet (single tz-naive `session` column).

Point-in-time note: for historical decision_date mapping, realized
sessions are exactly right. For forward scheduling, NSE publishes the
holiday list yearly; refresh this file when it changes.
"""

import argparse

import pandas as pd

from ..config import load_settings
from ..utils.io import atomic_write_parquet
from ..utils.logging import setup_logging
from ..utils.state import update_state


def build_calendar(nifty: pd.DataFrame | None = None) -> pd.DatetimeIndex:
    """Pure core: sessions from NIFTY presence. Testable without files."""
    if nifty is None or nifty.empty or "nifty" not in nifty.columns:
        return pd.DatetimeIndex([], name="session")
    sessions = pd.DatetimeIndex(
        pd.to_datetime(nifty.index[nifty["nifty"].notna()]).tz_localize(None).normalize().unique()
    ).sort_values()
    sessions.name = "session"
    return sessions


def load_calendar(ref_dir=None) -> pd.DatetimeIndex:
    """Load built calendar; empty index when not yet built."""
    from pathlib import Path
    from ..config import load_settings as _ls
    ref = Path(ref_dir) if ref_dir else _ls().paths.data_raw.parent / "reference"
    path = ref / "trading_calendar.parquet"
    if not path.exists():
        return pd.DatetimeIndex([], name="session")
    frame = pd.read_parquet(path)
    col = "session" if "session" in frame.columns else frame.columns[0]
    return pd.DatetimeIndex(pd.to_datetime(frame[col]).dt.tz_localize(None)).sort_values()


def run_build_calendar() -> pd.DatetimeIndex:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "calendar.log")
    ref_dir = paths.data_raw.parent / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    macro_path = paths.data_raw / "macro.parquet"
    macro = pd.read_parquet(macro_path) if macro_path.exists() else pd.DataFrame()
    sessions = build_calendar(macro)
    atomic_write_parquet(pd.DataFrame({"session": sessions}), ref_dir / "trading_calendar.parquet")
    update_state(paths.state, "calendar", status="complete", sessions=int(len(sessions)),
                 first=str(sessions.min().date()) if len(sessions) else None,
                 last=str(sessions.max().date()) if len(sessions) else None)
    logger.info("Trading calendar built: %d sessions.", len(sessions))
    return sessions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NSE trading calendar (F-02).")
    parser.parse_args()
    run_build_calendar()


if __name__ == "__main__":
    main()
