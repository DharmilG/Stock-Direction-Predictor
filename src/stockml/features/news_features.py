from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _sentiment_value(row: pd.Series) -> float:
    value = row.get("sentiment")
    if pd.notna(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    label = str(row.get("sentiment_label") or "").lower()
    return {"positive": 1.0, "neutral": 0.0, "negative": -1.0}.get(label, 0.0)


def add_finbert_scores(news: pd.DataFrame, scorer) -> pd.DataFrame:
    if news.empty:
        return news
    frame = news.copy()
    texts = (frame["title"].fillna("").astype(str) + ". " + frame.get("description", "").fillna("").astype(str)).tolist()
    labels, scores = scorer(texts)
    frame["finbert_label"] = labels
    frame["finbert_score"] = scores
    frame["sentiment"] = frame["finbert_score"]
    return frame


def _day_floor(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.tz_localize(None).dt.normalize()


def aggregate_news(news: pd.DataFrame, trading_index: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
    """Aggregate point-in-time news into daily market-aligned features."""
    index = pd.DatetimeIndex(trading_index).tz_localize(None).normalize()
    base = pd.DataFrame(index=index)
    base.index.name = "timestamp"
    if news.empty:
        return base.assign(**{f"news_{x}": 0.0 for x in [
            "count", "sources", "sentiment_mean", "sentiment_std", "positive", "negative",
            "relevance_mean", "finbert_score_mean", "headline_length_mean",
        ]})

    frame = news.copy()
    frame = frame[frame["symbol"].astype(str) == str(symbol)] if "symbol" in frame.columns else frame
    frame["date"] = _day_floor(frame["published_utc"])
    frame["sentiment_value"] = frame.apply(_sentiment_value, axis=1)
    frame["headline_length"] = frame["title"].fillna("").astype(str).str.len()
    grouped = frame.groupby("date").agg(
        news_count=("article_id", "nunique"),
        news_sources=("source", "nunique"),
        news_sentiment_mean=("sentiment_value", "mean"),
        news_sentiment_std=("sentiment_value", "std"),
        news_positive=("sentiment_value", lambda s: float((s > 0).sum())),
        news_negative=("sentiment_value", lambda s: float((s < 0).sum())),
        news_relevance_mean=("relevance", "mean"),
        news_finbert_score_mean=("finbert_score", "mean") if "finbert_score" in frame.columns else ("sentiment_value", "mean"),
        news_headline_length_mean=("headline_length", "mean"),
    )
    out = base.join(grouped)
    numeric_cols = list(out.columns)
    out[numeric_cols] = out[numeric_cols].fillna(0.0)
    for w in (1, 3, 5, 10, 20):
        out[f"news_count_{w}d"] = out["news_count"].rolling(w, min_periods=1).sum().shift(1).fillna(0.0)
        out[f"news_sentiment_{w}d"] = out["news_sentiment_mean"].rolling(w, min_periods=1).mean().shift(1).fillna(0.0)
        out[f"news_negative_ratio_{w}d"] = (
            out["news_negative"].rolling(w, min_periods=1).sum()
            / out["news_count"].rolling(w, min_periods=1).sum().replace(0, np.nan)
        ).shift(1).fillna(0.0)
    out["news_sentiment_mean"] = out["news_sentiment_mean"].shift(1).fillna(0.0)
    out["news_sentiment_std"] = out["news_sentiment_std"].shift(1).fillna(0.0)
    out["news_relevance_mean"] = out["news_relevance_mean"].shift(1).fillna(0.0)
    out["news_finbert_score_mean"] = out["news_finbert_score_mean"].shift(1).fillna(0.0)
    out["news_headline_length_mean"] = out["news_headline_length_mean"].shift(1).fillna(0.0)
    # Surprise/novelty proxy: unusual count relative to trailing history.
    rolling_mean = out["news_count"].rolling(20, min_periods=5).mean()
    rolling_std = out["news_count"].rolling(20, min_periods=5).std().replace(0, np.nan)
    out["news_volume_z20"] = ((out["news_count"] - rolling_mean) / rolling_std).shift(1).fillna(0.0)
    out["news_negative_shock"] = (out["news_negative"] - out["news_positive"]).shift(1).fillna(0.0)
    return out
