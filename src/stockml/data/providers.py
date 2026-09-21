from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

# yfinance rate-limits rapid sequential downloads (HTTP 429 surfaces as an
# empty frame). Retry with backoff instead of failing the whole fetch.
_FETCH_ATTEMPTS = 4
_FETCH_BACKOFF_SECONDS = (5, 15, 30, 60)


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def fetch_actions(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError


class YahooFinanceProvider(MarketDataProvider):
    """Runnable fallback/development provider.

    For production, replace this with a licensed market-data provider and keep the same interface.
    """

    def fetch_ohlcv(self, symbol: str, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
        import logging
        import time

        logger = logging.getLogger(__name__)
        last_error: Exception | None = None
        for attempt in range(_FETCH_ATTEMPTS):
            try:
                frame = yf.download(
                    symbol,
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    interval=interval,
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    threads=False,
                    group_by="column",
                    multi_level_index=False,
                    timeout=60,
                )
            except Exception as exc:  # network/timeout/rate-limit — retryable
                last_error = exc
                frame = None
            if frame is not None and not frame.empty:
                break
            last_error = last_error or RuntimeError("empty response (likely rate-limited)")
            wait = _FETCH_BACKOFF_SECONDS[min(attempt, len(_FETCH_BACKOFF_SECONDS) - 1)]
            logger.warning("OHLCV fetch for %s attempt %d/%d failed (%s); retrying in %ds.",
                           symbol, attempt + 1, _FETCH_ATTEMPTS, last_error, wait)
            time.sleep(wait)
        else:
            frame = None
        if frame is None or frame.empty:
            raise RuntimeError(f"No OHLCV data returned for {symbol} after {_FETCH_ATTEMPTS} attempts: {last_error}")
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [str(c[0]).lower().replace(" ", "_") for c in frame.columns]
        else:
            frame = frame.rename(columns={c: str(c).lower().replace(" ", "_") for c in frame.columns})
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise RuntimeError(f"Missing columns for {symbol}: {missing}")
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame.index.name = "timestamp"
        return frame[required].sort_index()

    def fetch_actions(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        actions = ticker.actions
        if actions is None or actions.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp"))
        actions.index = pd.to_datetime(actions.index).tz_localize(None)
        actions = actions.loc[(actions.index.date >= start) & (actions.index.date <= end)]
        return actions.sort_index()


def get_provider(name: str) -> MarketDataProvider:
    key = name.lower().strip()
    if key == "yfinance":
        return YahooFinanceProvider()
    raise ValueError(
        f"Unknown provider '{name}'. Implement a MarketDataProvider adapter in providers.py and select it in .env/config.yaml."
    )
