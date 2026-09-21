from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


@dataclass(frozen=True)
class NewsArticle:
    article_id: str
    symbol: str
    published_utc: pd.Timestamp
    title: str
    description: str = ""
    url: str = ""
    source: str = ""
    provider: str = ""
    sentiment: float | None = None
    sentiment_label: str | None = None
    relevance: float | None = None


def _article_id(*parts: str) -> str:
    raw = "|".join(p.strip() for p in parts if p is not None)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class QuotaExhausted(RuntimeError):
    """Raised when the AlphaVantage daily call budget is spent."""


def _quota_check_and_take(quota_path: Path | None, daily_budget: int) -> None:
    today = date.today().isoformat()
    state = {"date": today, "used": 0}
    if quota_path is not None and quota_path.exists():
        try:
            state = json.loads(quota_path.read_text(encoding="utf-8"))
        except Exception:
            state = {"date": today, "used": 0}
    if state.get("date") != today:
        state = {"date": today, "used": 0}
    if int(state.get("used", 0)) >= daily_budget:
        raise QuotaExhausted(f"AlphaVantage budget spent ({state['used']}/{daily_budget} today). Resume tomorrow.")
    state["used"] = int(state.get("used", 0)) + 1
    if quota_path is not None:
        quota_path.parent.mkdir(parents=True, exist_ok=True)
        quota_path.write_text(json.dumps(state), encoding="utf-8")


class AlphaVantageNewsProvider:
    """Historical/live market news provider.

    Requires ALPHAVANTAGE_API_KEY. The provider is intentionally paginated and
    date-windowed so a long historical backfill can be resumed safely.
    """

    endpoint = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, pause_seconds: float = 0.25, session: requests.Session | None = None,
                 quota_path: Path | None = None, daily_budget: int = 25):
        if not api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY is required for Alpha Vantage news")
        self.api_key = api_key
        self.pause_seconds = pause_seconds
        self.session = session or requests.Session()
        # Free-tier burst guard: max `daily_budget` HTTP calls per UTC day,
        # persisted so separate runs share one budget. Raises QuotaExhausted
        # instead of burning the key or getting throttled mid-backfill.
        self.quota_path = quota_path
        self.daily_budget = daily_budget

    def fetch(self, symbol: str, query_symbol: str, start: date, end: date, limit: int = 1000) -> list[NewsArticle]:
        articles: list[NewsArticle] = []
        cursor_from = start.strftime("%Y%m%dT0000")
        cursor_to = end.strftime("%Y%m%dT2359")
        next_cursor_from = cursor_from
        while True:
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": query_symbol,
                "time_from": next_cursor_from,
                "time_to": cursor_to,
                "sort": "EARLIEST",
                "limit": min(1000, limit),
                "apikey": self.api_key,
            }
            _quota_check_and_take(self.quota_path, self.daily_budget)
            resp = self.session.get(self.endpoint, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                break
            feed = payload.get("feed") or []
            if not feed:
                break
            max_seen: str | None = None
            for item in feed:
                published = pd.to_datetime(item.get("time_published"), format="%Y%m%dT%H%M%S", utc=True, errors="coerce")
                if pd.isna(published):
                    continue
                published_naive = published.tz_convert(None)
                title = str(item.get("title") or "").strip()
                desc = str(item.get("summary") or "").strip()
                url = str(item.get("url") or "").strip()
                publisher = str((item.get("source") or "")).strip()
                score = None
                relevance = None
                try:
                    score = float(item.get("overall_sentiment_score"))
                except (TypeError, ValueError):
                    pass
                try:
                    relevance = float(item.get("overall_sentiment_relevance_score"))
                except (TypeError, ValueError):
                    pass
                articles.append(NewsArticle(
                    article_id=_article_id(symbol, url, title, str(published_naive)),
                    symbol=symbol,
                    published_utc=published_naive,
                    title=title,
                    description=desc,
                    url=url,
                    source=publisher,
                    provider="alphavantage",
                    sentiment=score,
                    sentiment_label=None,
                    relevance=relevance,
                ))
                value = str(item.get("time_published") or "")
                if value and (max_seen is None or value > max_seen):
                    max_seen = value
            if len(feed) < limit or not max_seen:
                break
            # The API doesn't expose an opaque cursor in this endpoint. Move the
            # lower bound past the latest returned article and deduplicate later.
            try:
                latest_dt = datetime.strptime(max_seen, "%Y%m%dT%H%M%S") + timedelta(seconds=1)
                next_cursor_from = latest_dt.strftime("%Y%m%dT%H%M")
            except ValueError:
                break
            if next_cursor_from >= cursor_to:
                break
            time.sleep(self.pause_seconds)
        return articles


class GDELTDocProvider:
    """Recent global-news provider using the GDELT DOC 2.0 API.

    GDELT's DOC API only exposes a recent rolling window, so this provider is
    deliberately used for recent/current enrichment rather than pretending it is
    a 16-year full-text archive. Hard limit observed 2026-09-20: HTTP 429 past
    ~1 request per 5 seconds (HTML "limit requests" page, not JSON).
    """

    endpoint = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, pause_seconds: float = 6.0, session: requests.Session | None = None):
        self.pause_seconds = max(6.0, pause_seconds)
        self.session = session or requests.Session()
        self._attempts = 4
        self._backoff = (10, 30, 60, 120)

    def fetch(self, symbol: str, query: str, start: datetime, end: datetime, max_records: int = 250) -> list[NewsArticle]:
        import time as _time
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": min(250, max_records),
            "sort": "datedesc",
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        }
        last_err: Exception | None = None
        for attempt in range(self._attempts):
            try:
                resp = self.session.get(self.endpoint, params=params, timeout=60)
                if resp.status_code == 429:
                    raise RuntimeError("GDELT 429 rate-limit (need >=5s between requests)")
                resp.raise_for_status()
                try:
                    payload = resp.json()
                except Exception:
                    raise RuntimeError(f"GDELT non-JSON response ({len(resp.content)} bytes, likely rate-limit page)")
                break
            except Exception as exc:
                last_err = exc
                _time.sleep(self._backoff[min(attempt, len(self._backoff) - 1)])
        else:
            raise RuntimeError(f"GDELT fetch failed for {symbol}: {last_err}")
        out: list[NewsArticle] = []
        for item in payload.get("articles", []) if isinstance(payload, dict) else []:
            published = pd.to_datetime(item.get("seendate") or item.get("date"), utc=True, errors="coerce")
            if pd.isna(published):
                continue
            published_naive = published.tz_convert(None)
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            source = str(item.get("domain") or item.get("sourcecountry") or "").strip()
            out.append(NewsArticle(
                article_id=_article_id(symbol, url, title, str(published_naive)),
                symbol=symbol,
                published_utc=published_naive,
                title=title,
                url=url,
                source=source,
                provider="gdelt",
            ))
        time.sleep(self.pause_seconds)
        return out


def articles_to_frame(articles: Iterable[NewsArticle]) -> pd.DataFrame:
    rows = []
    for a in articles:
        rows.append({
            "article_id": a.article_id,
            "symbol": a.symbol,
            "published_utc": pd.Timestamp(a.published_utc),
            "title": a.title,
            "description": a.description,
            "url": a.url,
            "source": a.source,
            "provider": a.provider,
            "sentiment": a.sentiment,
            "sentiment_label": a.sentiment_label,
            "relevance": a.relevance,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "article_id", "symbol", "published_utc", "title", "description", "url",
            "source", "provider", "sentiment", "sentiment_label", "relevance",
        ])
    frame = pd.DataFrame(rows)
    frame["published_utc"] = pd.to_datetime(frame["published_utc"], utc=True).dt.tz_convert(None)
    return frame.drop_duplicates("article_id").sort_values("published_utc").reset_index(drop=True)


def collect_news_for_symbol(
    symbol: str,
    start: date,
    end: date,
    output_path: Path,
    query: str | None = None,
    alphavantage_symbol: str | None = None,
    enable_alphavantage: bool = False,
    enable_gdelt_recent: bool = True,
    gdelt_days: int = 90,
) -> pd.DataFrame:
    """Resume-safe news collection into a local parquet file.

    The function deliberately stores raw provider fields. Feature aggregation is
    performed separately so the raw news corpus remains auditable.
    """
    existing = pd.DataFrame()
    if output_path.exists():
        try:
            existing = pd.read_parquet(output_path)
        except Exception:
            existing = pd.DataFrame()

    pieces = [existing] if not existing.empty else []
    av_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    if enable_alphavantage and av_key and alphavantage_symbol:
        from ..config import load_settings
        quota_path = load_settings().paths.state / "av_quota.json"
        provider = AlphaVantageNewsProvider(av_key, quota_path=quota_path)
        # Free-tier budget: 25 calls/day. Each monthly window costs >=1 call,
        # so cap history at AV_LOOKBACK_DAYS (default 2y — AV coverage is
        # ~2022+ anyway). Oldest-first is openly wasteful; start recent.
        try:
            _lookback = int(os.getenv("AV_LOOKBACK_DAYS", "730"))
        except ValueError:
            _lookback = 730
        # Incremental history: start after the last stored article date, with a
        # one-day overlap to catch late corrections/duplicates safely.
        cursor = max(start, end - timedelta(days=_lookback))
        if not existing.empty and "published_utc" in existing.columns:
            try:
                last = pd.to_datetime(existing["published_utc"], errors="coerce").max()
                if pd.notna(last):
                    cursor = max(start, last.date() - timedelta(days=1))
            except Exception:
                pass
        while cursor <= end:
            window_end = min(end, cursor + timedelta(days=29))
            try:
                chunk = provider.fetch(symbol, alphavantage_symbol, cursor, window_end)
                frame = articles_to_frame(chunk)
                if not frame.empty:
                    pieces.append(frame)
            except Exception:
                # Keep the raw corpus accumulated so far. The orchestrator can retry.
                pass
            cursor = window_end + timedelta(days=1)

    if enable_gdelt_recent and query:
        gdelt_start = max(start, end - timedelta(days=max(1, gdelt_days) - 1))
        provider = GDELTDocProvider()
        try:
            chunk = provider.fetch(symbol, query, datetime.combine(gdelt_start, datetime.min.time()), datetime.combine(end, datetime.max.time()))
            frame = articles_to_frame(chunk)
            if not frame.empty:
                pieces.append(frame)
        except Exception:
            pass

    if not pieces:
        frame = articles_to_frame([])
    else:
        frame = pd.concat(pieces, ignore_index=True)
        frame = frame.drop_duplicates("article_id").sort_values("published_utc").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, output_path)
    return frame
