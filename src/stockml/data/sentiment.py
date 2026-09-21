from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd
import requests


@dataclass
class SentimentRecord:
    timestamp: pd.Timestamp
    symbol: str
    score: float
    source: str


class GDELTNewsSentimentProvider:
    """Optional news adapter.

    GDELT is used here as a no-key integration point for experimentation. For production,
    use a licensed, point-in-time news feed and an inference worker that stores raw news,
    publication timestamps and model versions.
    """

    endpoint = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, max_records: int = 250):
        self.max_records = max_records

    def fetch(self, query: str, start: date, end: date) -> pd.DataFrame:
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": self.max_records,
            "startdatetime": start.strftime("%Y%m%d000000"),
            "enddatetime": end.strftime("%Y%m%d235959"),
            "sort": "datedesc",
        }
        response = requests.get(self.endpoint, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("articles", [])
        rows = []
        for article in articles:
            raw_date = article.get("seendate") or article.get("date")
            if not raw_date:
                continue
            try:
                ts = pd.to_datetime(raw_date, utc=True).tz_convert(None)
            except Exception:
                continue
            # GDELT does not provide a finance-specific sentiment score. We persist neutral 0.0
            # unless a downstream sentiment model is configured; this prevents accidental fake signal.
            rows.append({"timestamp": ts, "sentiment_score": 0.0, "url": article.get("url", "")})
        if not rows:
            return pd.DataFrame(columns=["timestamp", "sentiment_score", "url"]).set_index("timestamp")
        return pd.DataFrame(rows).set_index("timestamp").sort_index()
