from __future__ import annotations

"""Feature-store join (N6 storage split).

Build artifacts live separated:
  data/features/stocks/{SYMBOL}_features.parquet  (price-derived)
  data/features/news/{SYMBOL}_news_features.parquet (news-derived)
  data/features/market/market_features.parquet     (market-wide, one copy)
data/features/{SYMBOL}_features.parquet is the MATERIALIZED joined view —
every existing reader (train, monitor, predict, labels, backtest) keeps
working unchanged. _join_log.json records per-symbol input mtimes so
staleness is checkable without re-reading parquet.
"""

import pandas as pd

from ..utils.io import atomic_write_parquet


def join_symbol_features(stocks: pd.DataFrame, news: pd.DataFrame | None,
                         market: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join news + market onto stocks. Pure (testable without files)."""
    out = stocks.copy()
    for extra, tag in ((news, "news"), (market, "market")):
        if extra is None or extra.empty:
            continue
        overlap = [c for c in extra.columns if c in out.columns]
        if overlap:
            extra = extra.drop(columns=overlap)
        out = out.join(extra, how="left")
    out.index.name = "timestamp"
    return out
