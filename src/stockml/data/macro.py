from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

# Yahoo symbols are used only as a runnable fallback. A licensed macro provider can replace this adapter.
# India-only per the Master Plan (B-10): no US equities/rates as features.
# The "vix" key is INDIA VIX (^INDIAVIX), not CBOE VIX — downstream
# feature code (vix_regime_high) keys off this name, so keep it.
DEFAULT_MACRO_SYMBOLS = {
    "nifty": "^NSEI",
    "banknifty": "^NSEBANK",
    "vix": "^INDIAVIX",
    "usd_inr": "INR=X",
    "crude": "CL=F",
    "gold": "GC=F",
}


def fetch_macro(start: date, end: date) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, symbol in DEFAULT_MACRO_SYMBOLS.items():
        frame = yf.download(
            symbol,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
            multi_level_index=False,
            timeout=30,
        )
        if frame is None or frame.empty:
            continue
        if isinstance(frame.columns, pd.MultiIndex):
            close_candidates = [c for c in frame.columns if str(c[0]).lower() == "close"]
            if not close_candidates:
                continue
            close = frame[close_candidates[0]].rename(name)
        else:
            close = frame["Close"].rename(name) if "Close" in frame else frame["close"].rename(name)
        close.index = pd.to_datetime(close.index).tz_localize(None)
        frames.append(close)
    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp"))
    result = pd.concat(frames, axis=1).sort_index()
    result.index.name = "timestamp"
    return result
