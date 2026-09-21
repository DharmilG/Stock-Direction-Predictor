from __future__ import annotations

"""NSE corporate-announcement ingestion (F-06, Track U backbone).

Source: https://www.nseindia.com/api/corporate-announcements?index=equities&symbol=XXX
Pre-tagged with the exact symbol — zero NLP/entity ambiguity (§5.2), the
highest-confidence context family. History on NSE reaches 2004.

Stored per symbol: data/raw/announcements/{SAFE}.parquet with
  article_id (NSE seq_id, stable) | published_utc | source=nse_ann |
  symbol | subject (NSE desc category) | text | url
Cursors: data/raw/announcements/_cursors/{SAFE}.json (per-source cursors §5.5).

BSE announcements are specced as the follow-up (no clean free JSON API);
NSE first. This module fetches + stores ONLY — aggregation into features
(B-06/F-05 fan-out) is the U-ablation step and stays untouched here.
"""

import argparse
import hashlib
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from ..config import load_settings
from ..utils.io import atomic_write_json, atomic_write_parquet, read_json
from ..utils.logging import setup_logging
from ..utils.state import update_state

IST = ZoneInfo("Asia/Kolkata")
URL = "https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={symbol}"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/",
}
_ATTEMPTS = 3
_BACKOFF = (5, 15, 45)
_DAY_GAP_SECONDS = 3


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=20)
    except Exception:
        pass
    return s


def parse_announcements(payload: list[dict], symbol_ns: str) -> pd.DataFrame:
    """NSE JSON → normalized frame. Pure (testable without network)."""
    rows = []
    for rec in payload:
        seq = str(rec.get("seq_id") or rec.get("dt") or "")
        if not seq:
            continue
        raw_ts = rec.get("an_dt") or rec.get("sort_date") or ""
        try:
            naive = pd.to_datetime(raw_ts, format="%d-%b-%Y %H:%M:%S")
        except (ValueError, TypeError):
            try:
                naive = pd.to_datetime(raw_ts)
            except (ValueError, TypeError):
                continue
        if pd.isna(naive):
            continue
        published_utc = pd.Timestamp(naive).tz_localize(IST).tz_convert("UTC").tz_localize(None)
        text = str(rec.get("attchmntText") or "")
        url = str(rec.get("attchmntFile") or "")
        rows.append({
            "article_id": f"nse_ann:{seq}",
            "published_utc": published_utc,
            "source": "nse_ann",
            "symbol": symbol_ns,
            "subject": str(rec.get("desc") or "General"),
            "text": text,
            "url": "" if url == "-" else url,
            "text_sha": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16],
        })
    if not rows:
        return pd.DataFrame(columns=["article_id", "published_utc", "source", "symbol", "subject", "text", "url", "text_sha"])
    frame = pd.DataFrame(rows).drop_duplicates(subset=["article_id"]).sort_values("published_utc")
    return frame


def fetch_symbol_announcements(base_symbol: str, session: requests.Session | None = None) -> pd.DataFrame:
    """Full announcement history for one NSE symbol (no .NS suffix)."""
    s = session or _session()
    url = URL.format(symbol=base_symbol)
    last_err: Exception | None = None
    for attempt in range(_ATTEMPTS):
        try:
            r = s.get(url, timeout=60)
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, list):
                raise RuntimeError(f"unexpected payload: {type(payload)}")
            return payload
        except Exception as exc:
            last_err = exc
            time.sleep(_BACKOFF[min(attempt, len(_BACKOFF) - 1)])
    raise RuntimeError(f"announcements fetch failed for {base_symbol}: {last_err}")


def _safe(symbol: str) -> str:
    return symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")


def run_announcement_fetch() -> dict:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "announcements.log")
    out_dir = paths.data_raw / "announcements"
    cursor_dir = out_dir / "_cursors"
    out_dir.mkdir(parents=True, exist_ok=True)
    cursor_dir.mkdir(parents=True, exist_ok=True)

    s = _session()
    summary: dict[str, int] = {}
    for i, symbol in enumerate(settings.symbols):
        if not symbol.endswith(".NS"):
            continue
        base = symbol.replace(".NS", "")
        safe = _safe(symbol)
        out = out_dir / f"{safe}.parquet"
        try:
            payload = fetch_symbol_announcements(base, s)
            frame = parse_announcements(payload, symbol)
        except Exception as exc:
            logger.warning("Announcements for %s failed: %s", symbol, exc)
            summary[symbol] = -1
            continue
        existing = None
        if out.exists():
            try:
                existing = pd.read_parquet(out)
            except Exception:
                existing = None
        merged = pd.concat([existing, frame]) if existing is not None and not existing.empty else frame
        merged = merged.drop_duplicates(subset=["article_id"]).sort_values("published_utc")
        atomic_write_parquet(merged, out)
        atomic_write_json(cursor_dir / f"{safe}.json", {
            "source": "nse_ann", "symbol": symbol,
            "last_run_utc": datetime.utcnow().isoformat() + "Z",
            "articles_total": int(len(merged)),
            "status": "complete" if len(merged) else "empty",
        })
        summary[symbol] = int(len(merged))
        logger.info("Announcements for %s: %d records.", symbol, len(merged))
        if i < len(settings.symbols) - 1:
            time.sleep(_DAY_GAP_SECONDS)
    update_state(paths.state, "announcements", status="complete", summary=summary)
    logger.info("Announcement fetch complete.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NSE corporate announcements (F-06).")
    parser.parse_args()
    run_announcement_fetch()


def run_ann_scoring(since: str = "2023-08-29", batch_size: int = 64) -> dict:
    """Score announcement texts with FinBERT, cached forever (§5.6).

    Only rows with decision-relevant dates (published_utc >= since) lacking
    finbert_score are scored; everything else is skipped (resume-safe).
    Scoring text content is point-in-time (no timing information added).
    """
    from .finbert import FinBERTScorer
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "ann_score.log")
    out_dir = paths.data_raw / "announcements"
    scorer: FinBERTScorer | None = None
    summary: dict[str, int] = {"scored": 0, "skipped": 0, "symbols": 0}
    for path in sorted(out_dir.glob("*.parquet")):
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if frame.empty:
            continue
        if "finbert_score" not in frame.columns:
            frame["finbert_score"] = float("nan")
            frame["finbert_label"] = ""
        todo = frame[(pd.to_datetime(frame["published_utc"]) >= since) & frame["finbert_score"].isna()]
        summary["skipped"] += int(len(frame) - len(todo))
        if todo.empty:
            continue
        if scorer is None:
            scorer = FinBERTScorer(model_name=settings.finbert_model,
                                   cache_dir=settings.finbert_cache_dir, device=-1)
        texts = [(str(r.get("subject") or "") + ". " + str(r.get("text") or "")).strip(". ")
                 for _, r in todo.iterrows()]
        labels_all: list[str] = []
        scores_all: list[float] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            blank = [not t.strip() for t in chunk]
            subt = [t if not b else "General corporate update." for t, b in zip(chunk, blank)]
            labs, scrs = scorer(subt)
            labels_all.extend(["neutral" if b else l for b, l in zip(blank, labs)])
            scores_all.extend([0.0 if b else s for b, s in zip(blank, scrs)])
        frame.loc[todo.index, "finbert_label"] = labels_all
        frame.loc[todo.index, "finbert_score"] = scores_all
        atomic_write_parquet(frame, path)
        summary["scored"] += int(len(todo))
        summary["symbols"] += 1
        logger.info("Scored %d announcements in %s.", len(todo), path.name)
    update_state(paths.state, "ann_score", status="complete", summary=summary)
    logger.info("Announcement scoring complete: %s.", summary)
    return summary


if __name__ == "__main__":
    main()
