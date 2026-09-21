from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_and_market_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.DatetimeIndex(out.index).tz_localize(None)
    month = idx.month
    quarter = idx.quarter
    out["calendar_month"] = month.astype(int)
    out["calendar_quarter"] = quarter.astype(int)
    out["calendar_weekofyear"] = idx.isocalendar().week.to_numpy(dtype=int)
    out["calendar_dayofweek"] = idx.dayofweek.astype(int)
    out["calendar_month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["calendar_month_cos"] = np.cos(2 * np.pi * month / 12.0)
    out["calendar_quarter_sin"] = np.sin(2 * np.pi * quarter / 4.0)
    out["calendar_quarter_cos"] = np.cos(2 * np.pi * quarter / 4.0)
    # Indian fiscal year: Apr-Mar.
    fiscal_q = ((month - 4) % 12) // 3 + 1
    out["fiscal_quarter"] = fiscal_q.astype(int)
    out["fiscal_year"] = (idx.year + (month >= 4).astype(int)).astype(int)
    out["days_to_month_end"] = (idx.to_period("M").to_timestamp(how="end").normalize() - idx.normalize()).days
    out["is_quarter_end"] = idx.is_quarter_end.astype(int)
    out["is_month_end"] = idx.is_month_end.astype(int)
    out["is_month_start"] = idx.is_month_start.astype(int)
    # Earnings-season proxies commonly used for Indian quarterly cycles.
    out["earnings_season_proxy"] = np.isin(month, [1, 4, 7, 10]).astype(int)
    return out
