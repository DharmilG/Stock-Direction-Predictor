from __future__ import annotations

"""RSS daily news ingestion (N2, Track N).

Free, live-only (no history — complements GDELT DOC's 90-day window).
Working feeds verified 2026-09-20 (MC-Markets 502, BS-Markets 403 dropped):
  ET-Markets, Mint-Markets, MC-TopNews.
Items are market-scope by default (RSS carries no scrip tags); company
routing via aliases is N6 scope-fan-out work, not ingestion work.
Schema matches the news corpus (article_id, symbol='', published_utc,
title, description, url, source, provider=rss) so downstream scoring and
aggregation reuse the same code paths.
"""

import argparse
import hashlib
from datetime import date, datetime, timezone

import feedparser
import pandas as pd

from ..config import load_settings
from ..utils.io import atomic_write_parquet
from ..utils.logging import setup_logging
from ..utils.state import update_state

FEEDS = {
    "ET-Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Mint-Markets": "https://www.livemint.com/rss/markets",
    "MC-TopNews": "https://www.moneycontrol.com/rss/MCtopnews.xml",
}


def _article_id(url: str, title: str) -> str:
    return hashlib.sha256(f"{url}|{title}".encode("utf-8", errors="replace")).hexdigest()[:32]


def normalize_entries(entries: list[dict], source: str) -> pd.DataFrame:
    """Feed entries → news-corpus frame. Pure (testable without network)."""
    rows = []
    for e in entries:
        title = str(e.get("title") or "").strip()
        if not title:
            continue
        url = str(e.get("link") or e.get("id") or "").strip()
        published = None
        for key in ("published_parsed", "updated_parsed"):
            if e.get(key):
                try:
                    published = datetime(*e[key][:6], tzinfo=timezone.utc)
                    break
                except Exception:
                    pass
        if published is None:
            try:
                published = pd.to_datetime(e.get("published") or e.get("updated"), utc=True)
            except Exception:
                continue
        if pd.isna(published):
            continue
        rows.append({
            "article_id": _article_id(url, title),
            "symbol": "",
            "published_utc": pd.Timestamp(published).tz_convert(None),
            "title": title,
            "description": str(e.get("summary") or e.get("description") or "").strip(),
            "url": url,
            "source": source,
            "provider": "rss",
        })
    if not rows:
        return pd.DataFrame(columns=["article_id", "symbol", "published_utc", "title",
                                     "description", "url", "source", "provider"])
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates("article_id").sort_values("published_utc").reset_index(drop=True)


def run_rss_fetch(day: date | None = None) -> dict:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "rss.log")
    out_dir = paths.data_raw / "rss"
    out_dir.mkdir(parents=True, exist_ok=True)
    day = day or date.today()
    frames = []
    for source, url in FEEDS.items():
        try:
            parsed = feedparser.parse(url)
            frame = normalize_entries(
                [{"title": e.get("title"), "link": e.get("link", e.get("id", "")),
                  "summary": e.get("summary", e.get("description", "")),
                  "published_parsed": e.get("published_parsed"), "updated_parsed": e.get("updated_parsed"),
                  "published": e.get("published"), "updated": e.get("updated")}
                 for e in parsed.entries], source)
            logger.info("RSS %s: %d entries.", source, len(frame))
            frames.append(frame)
        except Exception as exc:
            logger.warning("RSS %s failed: %s", source, exc)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not merged.empty:
        merged = merged.drop_duplicates("article_id").sort_values("published_utc").reset_index(drop=True)
    out = out_dir / f"rss_{day.isoformat()}.parquet"
    if out.exists():
        try:
            merged = pd.concat([pd.read_parquet(out), merged], ignore_index=True).drop_duplicates("article_id")
        except Exception:
            pass
    atomic_write_parquet(merged, out)
    update_state(paths.state, "rss", status="complete", day=str(day), rows=int(len(merged)))
    logger.info("RSS pull for %s: %d rows.", day, len(merged))
    return {"day": str(day), "rows": int(len(merged))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull RSS market news (N2).")
    parser.parse_args()
    run_rss_fetch()


if __name__ == "__main__":
    main()
