from __future__ import annotations

"""N6 news-context feature builders (company + market scope).

Separated build artifacts: data/features/news/{SYMBOL}_news_features.parquet
and data/features/market/market_features.parquet. Joined onto stocks by
features/join.py — builders never touch price features.

Conventions (B-05/B-07): decision_date alignment (no extra shift), counts
are TRUE zeros, means/z-scores/shocks stay NaN when no data (LightGBM
native missing), every family carries an availability flag. Group/sector
fan-out needs entity_map.json — company+market only (locked scope).
"""

import pandas as pd

from .announcement_features import aggregate_announcements
from .decision_date import decision_dates


def _empty_like(index: pd.DatetimeIndex, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out.index.name = "timestamp"
    for c in cols:
        out[c] = 0.0 if not (c.endswith("_available") or "days_since" in c) else (0.0 if c.endswith("_available") else float("nan"))
    return out


def aggregate_scored_text(
    frame: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    symbol: str,
    calendar: pd.DatetimeIndex,
    prefix: str,
    score_col: str = "finbert_score",
) -> pd.DataFrame:
    """Decision-date text aggregates (GDELT/RSS/AlphaVantage corpus). Pure core."""
    cols = [f"{prefix}_count", f"{prefix}_sent_mean", f"{prefix}_count_z20",
            f"{prefix}_sent_shock", f"{prefix}_count_5d", f"{prefix}_count_20d",
            f"{prefix}_available", f"{prefix}_coverage_60d"]
    index = pd.DatetimeIndex(trading_index).tz_localize(None).normalize()
    base = pd.DataFrame(index=index)
    base.index.name = "timestamp"
    if frame is None or frame.empty or calendar is None or len(calendar) == 0:
        return _empty_like(index, cols)
    f = frame.copy()
    if "symbol" in f.columns:
        f = f[f["symbol"].astype(str) == str(symbol)]
    if f.empty:
        return _empty_like(index, cols)
    f = f.copy()
    f["decision_date"] = decision_dates(f, calendar, col="published_utc")
    f = f.dropna(subset=["decision_date"])
    if f.empty:
        return _empty_like(index, cols)
    f["decision_date"] = pd.to_datetime(f["decision_date"]).dt.tz_localize(None).dt.normalize()
    f["_score"] = pd.to_numeric(f[score_col] if score_col in f.columns else float("nan"), errors="coerce")
    daily_count = f.groupby("decision_date").size().rename("c")
    daily_sent = f.groupby("decision_date")["_score"].mean().rename("s")
    out = base.join(daily_count, how="left").join(daily_sent, how="left")
    out[f"{prefix}_count"] = out["c"].fillna(0.0)
    out[f"{prefix}_sent_mean"] = out["s"]
    _c = out[f"{prefix}_count"]
    out[f"{prefix}_count_5d"] = _c.rolling(5, min_periods=1).sum()
    out[f"{prefix}_count_20d"] = _c.rolling(20, min_periods=1).sum()
    rm, rs = _c.rolling(20, min_periods=5).mean(), _c.rolling(20, min_periods=5).std().replace(0, float("nan"))
    out[f"{prefix}_count_z20"] = (_c - rm) / rs
    sm = out[f"{prefix}_sent_mean"]
    out[f"{prefix}_sent_shock"] = sm - sm.rolling(20, min_periods=5).mean()
    out[f"{prefix}_available"] = (out[f"{prefix}_count"].rolling(60, min_periods=1).sum() > 0).astype(float)
    out[f"{prefix}_coverage_60d"] = out[f"{prefix}_count"].rolling(60, min_periods=1).sum()
    return out.drop(columns=["c", "s"], errors="ignore")[cols]


def aggregate_event_tone(
    frame: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    symbol: str,
    calendar: pd.DatetimeIndex,
    prefix: str = "ev_co",
) -> pd.DataFrame:
    """GDELT-native tone aggregates (tone column is GDELT's, never FinBERT). Pure core."""
    cols = [f"{prefix}_count", f"{prefix}_tone_mean", f"{prefix}_count_z20", f"{prefix}_available"]
    index = pd.DatetimeIndex(trading_index).tz_localize(None).normalize()
    base = pd.DataFrame(index=index)
    base.index.name = "timestamp"
    if frame is None or frame.empty:
        return _empty_like(index, cols)
    f = frame.copy()
    if "symbol" in f.columns:
        f = f[f["symbol"].astype(str) == str(symbol)]
    if f.empty:
        return _empty_like(index, cols)
    f = f.copy()
    f["dd"] = pd.to_datetime(f["published_utc"], errors="coerce").dt.tz_localize(None).dt.normalize()
    if calendar is not None and len(calendar):
        cal = pd.DatetimeIndex(pd.to_datetime(calendar).tz_localize(None)).sort_values()
        pos = cal.searchsorted(f["dd"].to_numpy(), side="left")
        pos = pd.Series(pos, index=f.index).clip(0, len(cal) - 1)
        # Dates before the calendar map to NaT (dropped); else next session.
        mapped = pd.Series(pd.NaT, index=f.index)
        valid = f["dd"].notna()
        mapped[valid] = cal[pos[valid]].to_numpy()
        f["dd"] = mapped
    f = f.dropna(subset=["dd"])
    if f.empty:
        return _empty_like(index, cols)
    f["_tone"] = pd.to_numeric(f["tone"] if "tone" in f.columns else float("nan"), errors="coerce")
    daily_count = f.groupby("dd").size().rename("c")
    daily_tone = f.groupby("dd")["_tone"].mean().rename("t")
    out = base.join(daily_count, how="left").join(daily_tone, how="left")
    out[f"{prefix}_count"] = out["c"].fillna(0.0)
    out[f"{prefix}_tone_mean"] = out["t"]
    _c = out[f"{prefix}_count"]
    rm, rs = _c.rolling(20, min_periods=5).mean(), _c.rolling(20, min_periods=5).std().replace(0, float("nan"))
    out[f"{prefix}_count_z20"] = (_c - rm) / rs
    out[f"{prefix}_available"] = (out[f"{prefix}_count"].rolling(60, min_periods=1).sum() > 0).astype(float)
    return out.drop(columns=["c", "t"], errors="ignore")[cols]


def aggregate_market(
    ann_all: pd.DataFrame,
    news_all: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Market-wide rows (identical for every symbol — stored once). Pure core."""
    cols = ["news_mkt_count", "news_mkt_sent_mean", "news_mkt_count_z20", "ann_mkt_count"]
    index = pd.DatetimeIndex(trading_index).tz_localize(None).normalize()
    base = pd.DataFrame(index=index)
    base.index.name = "timestamp"
    out = base.copy()
    for c in cols:
        out[c] = 0.0
    out["news_mkt_sent_mean"] = float("nan")
    if ann_all is not None and not ann_all.empty and calendar is not None and len(calendar):
        a = ann_all.copy()
        a["decision_date"] = decision_dates(a, calendar, col="published_utc")
        a = a.dropna(subset=["decision_date"])
        if not a.empty:
            a["decision_date"] = pd.to_datetime(a["decision_date"]).dt.tz_localize(None).dt.normalize()
            out["ann_mkt_count"] = a.groupby("decision_date").size().reindex(out.index).fillna(0.0).to_numpy()
    if news_all is not None and not news_all.empty and calendar is not None and len(calendar):
        n = news_all.copy()
        n["decision_date"] = decision_dates(n, calendar, col="published_utc")
        n = n.dropna(subset=["decision_date"])
        if not n.empty:
            n["decision_date"] = pd.to_datetime(n["decision_date"]).dt.tz_localize(None).dt.normalize()
            n["_s"] = pd.to_numeric(n["finbert_score"] if "finbert_score" in n.columns else float("nan"), errors="coerce")
            out["news_mkt_count"] = n.groupby("decision_date").size().reindex(out.index).fillna(0.0).to_numpy()
            out["news_mkt_sent_mean"] = n.groupby("decision_date")["_s"].mean().reindex(out.index).to_numpy()
    _c = out["news_mkt_count"]
    rm, rs = _c.rolling(20, min_periods=5).mean(), _c.rolling(20, min_periods=5).std().replace(0, float("nan"))
    out["news_mkt_count_z20"] = ((_c - rm) / rs).to_numpy()
    return out[cols]


def build_news_features_for_symbol(
    news_frame: pd.DataFrame | None,
    ann_frame: pd.DataFrame | None,
    event_frame: pd.DataFrame | None,
    trading_index: pd.DatetimeIndex,
    symbol: str,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Company-scope news context for one symbol (locked scope)."""
    parts = [aggregate_scored_text(news_frame, trading_index, symbol, calendar, "news_co")]
    ann_co = aggregate_announcements(ann_frame if ann_frame is not None else pd.DataFrame(),
                                     trading_index, symbol, calendar)
    ren = {c: "ann_co_" + c[len("ann_"):] for c in ann_co.columns if c.startswith("ann_")}
    if "days_since_last_ann" in ann_co.columns:
        ren["days_since_last_ann"] = "ann_co_days_since_last_ann"
    ann_co = ann_co.rename(columns=ren)
    parts.append(ann_co)
    parts.append(aggregate_event_tone(event_frame, trading_index, symbol, calendar, "ev_co"))
    out = pd.concat(parts, axis=1)
    out.index.name = "timestamp"
    return out
