from __future__ import annotations

"""decision_date() news alignment (B-05/F-04, Track U).

Maps a publication timestamp to the FIRST NSE session it may influence:

    published_IST <= session @ 15:30  →  usable for THAT session's decision
    published_IST >  session @ 15:30  →  first usable NEXT session

Rules: CONVERT (never tz_localize(None) on aware stamps), compare against
the 15:30 cutoff, advance to the next real NSE session via
trading_calendar.parquet (weekends + holidays, e.g. Friday 16:00 before a
Monday holiday → Tuesday). Aggregate by decision_date with NO additional
shift(1) — the shift is baked in (double-shifting destroys a day, B-05).

This module defines + tests the mapping ONLY. aggregate_news() keeps its
current behavior until the U-ablation step flips it over deliberately.
"""

from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
CUTOFF = time(15, 30)


def decision_date(published_utc: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """First NSE session this article may influence. Pure.

    published_utc: tz-aware UTC (or naive assumed UTC). calendar: sorted
    tz-naive NSE sessions. Returns tz-naive session date.
    """
    ts = pd.Timestamp(published_utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ist = ts.tz_convert(IST)
    candidate = ist.normalize().tz_localize(None)
    if ist.time() > CUTOFF:
        candidate += pd.Timedelta(days=1)
    cal = pd.DatetimeIndex(pd.to_datetime(calendar).tz_localize(None)).sort_values()
    pos = cal.searchsorted(candidate, side="left")
    if pos >= len(cal):
        raise ValueError(f"No NSE session on/after {candidate.date()} in calendar.")
    return pd.Timestamp(cal[pos])


def decision_dates(frame: pd.DataFrame, calendar: pd.DatetimeIndex,
                   col: str = "published_utc") -> pd.Series:
    """Vectorized decision_date over a frame. Pure.

    Dates beyond the calendar (e.g. announcements newer than price history)
    map to NaT and are dropped by callers — they cannot join any trading
    row and must never shift history.
    """
    def _safe(ts):
        try:
            return decision_date(ts, calendar)
        except ValueError:
            return pd.NaT
    return frame[col].apply(_safe)
