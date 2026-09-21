from __future__ import annotations

"""GDELT BigQuery historical events (N4, Track N).

Billing truth probed 2026-09-20: gdeltv2.events is 0.4TB / 61 cols /
UNPARTITIONED — every query scans referenced columns over the whole table
(date filters do NOT prune bytes). GKG is 21.94TB: off-limits on Sandbox
quota. Therefore: ONE single-shot query (not monthly chunks), billed once
(~tens of GB), filtered client-side. Quota guard stops at 800GB/month.

Scope note: GDELT events are coded actor-action records (CAMEO) — good for
M&A/regulatory/conflict-type company news, weak for earnings/results chatter.
Stored per symbol: data/raw/gdelt_bq/{SAFE}.parquet
(article_id, published_utc, source=gdelt_bq, symbol, event_code,
tone, goldstein, mentions, url). Tone is GDELT-native (separate column
from FinBERT, never mixed).
"""

import argparse
import hashlib
from datetime import date

import pandas as pd

from ..config import load_settings
from ..utils.io import atomic_write_json, atomic_write_parquet, read_json
from ..utils.logging import setup_logging
from ..utils.state import update_state

PROJECT_FALLBACK = "stock-news-backfill"
MONTHLY_BYTE_BUDGET = 800_000_000_000

QUERY = """
SELECT SQLDATE, Actor1Name, Actor2Name, Actor1CountryCode, Actor2CountryCode,
       EventCode, EventRootCode, QuadClass, AvgTone, GoldsteinScale,
       NumMentions, SOURCEURL, DATEADDED
FROM `gdelt-bq.gdeltv2.events`
WHERE Actor1CountryCode = 'IND' OR Actor2CountryCode = 'IND'
"""


def _safe(symbol: str) -> str:
    return symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")


def _aliases(symbol: str, query: str) -> list[str]:
    """Uppercase match aliases from the GDELT query name + symbol stem."""
    base = {query.upper(), symbol.replace(".NS", "").replace("-", " ").upper()}
    out = set()
    for b in base:
        out.add(b)
        for stop in (" LIMITED", " LTD", " COMPANY", " CORPORATION", " INDUSTRIES"):
            if b.endswith(stop):
                out.add(b[: -len(stop)])
    return sorted(out, key=len, reverse=True)


def match_symbol(actor1: str, actor2: str, aliases: list[str]) -> bool:
    """Substring match on either actor (GDELT names are UPPER). Pure."""
    hay = f"{actor1 or ''} {actor2 or ''}".upper()
    return any(a and a in hay for a in aliases)


def frame_from_rows(rows: list[dict], symbol: str, aliases: list[str]) -> pd.DataFrame:
    """BigQuery rows → normalized per-symbol frame. Pure (testable offline)."""
    recs = []
    for r in rows:
        if not match_symbol(str(r.get("Actor1Name") or ""), str(r.get("Actor2Name") or ""), aliases):
            continue
        try:
            day = pd.to_datetime(str(int(r["SQLDATE"])), format="%Y%m%d")
        except (ValueError, TypeError, KeyError):
            continue
        url = str(r.get("SOURCEURL") or "")
        recs.append({
            "article_id": "gdelt_bq:" + hashlib.sha256(f"{symbol}|{url}|{day.date()}".encode()).hexdigest()[:32],
            "published_utc": day,
            "source": "gdelt_bq",
            "symbol": symbol,
            "event_code": str(r.get("EventCode") or ""),
            "event_root": str(r.get("EventRootCode") or ""),
            "tone": pd.to_numeric([r.get("AvgTone")], errors="coerce")[0],
            "goldstein": pd.to_numeric([r.get("GoldsteinScale")], errors="coerce")[0],
            "mentions": pd.to_numeric([r.get("NumMentions")], errors="coerce")[0],
            "url": url,
        })
    if not recs:
        return pd.DataFrame(columns=["article_id", "published_utc", "source", "symbol", "event_code",
                                     "event_root", "tone", "goldstein", "mentions", "url"])
    frame = pd.DataFrame(recs).drop_duplicates("article_id").sort_values("published_utc")
    return frame.reset_index(drop=True)


def _flush_buffers(buffers: dict[str, list[dict]], out_dir) -> None:
    """Merge buffered rows into per-symbol parquets. Pure I/O helper."""
    for symbol, rows in buffers.items():
        if not rows:
            continue
        out = out_dir / f"{_safe(symbol)}.parquet"
        frame = pd.DataFrame(rows).drop_duplicates("article_id").sort_values("published_utc")
        existing = None
        if out.exists():
            try:
                existing = pd.read_parquet(out)
            except Exception:
                existing = None
        merged = pd.concat([existing, frame]) if existing is not None and not existing.empty else frame
        merged = merged.drop_duplicates("article_id").sort_values("published_utc").reset_index(drop=True)
        atomic_write_parquet(merged, out)
        buffers[symbol] = []


def run_gdelt_bq_fetch(force: bool = False) -> dict:
    from google.cloud import bigquery
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "gdelt_bq.log")
    project = (getattr(settings, "gcp_project_id", "") or PROJECT_FALLBACK).strip() or PROJECT_FALLBACK
    quota_path = paths.state / "bq_quota.json"
    quota = read_json(quota_path, default={}) or {}
    month = date.today().strftime("%Y-%m")
    if quota.get("month") != month:
        quota = {"month": month, "billed_bytes": 0}
    if quota.get("billed_bytes", 0) >= MONTHLY_BYTE_BUDGET and not force:
        logger.info("BigQuery monthly byte budget spent; skipping.")
        return {"status": "quota-spent", **quota}
    done_marker = paths.state / "gdelt_bq_done.json"
    if done_marker.exists() and not force:
        logger.info("GDELT BigQuery backfill already complete; use force to re-run.")
        return {"status": "already-complete"}

    client = bigquery.Client(project=project)
    job = client.query(QUERY)
    # Stream pages (never list(): IND matches millions of rows and OOMs).
    # Match client-side per page, flush per symbol incrementally.
    from .news_fetch import _query_for
    out_dir = paths.data_raw / "gdelt_bq"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [(s, _aliases(s, _query_for(s))) for s in settings.symbols if s.endswith(".NS")]
    buffers: dict[str, list[dict]] = {s: [] for s, _ in targets}
    n_pages = n_rows = 0
    for page in job.result(page_size=20000).pages:
        n_pages += 1
        for r in page:
            n_rows += 1
            a1, a2 = str(r.get("Actor1Name") or ""), str(r.get("Actor2Name") or "")
            try:
                day = pd.to_datetime(str(int(r["SQLDATE"])), format="%Y%m%d")
            except (ValueError, TypeError, KeyError):
                continue
            url = str(r.get("SOURCEURL") or "")
            for symbol, aliases in targets:
                if not match_symbol(a1, a2, aliases):
                    continue
                buffers[symbol].append({
                    "article_id": "gdelt_bq:" + hashlib.sha256(f"{symbol}|{url}|{day.date()}".encode()).hexdigest()[:32],
                    "published_utc": day,
                    "source": "gdelt_bq",
                    "symbol": symbol,
                    "event_code": str(r.get("EventCode") or ""),
                    "event_root": str(r.get("EventRootCode") or ""),
                    "tone": pd.to_numeric([r.get("AvgTone")], errors="coerce")[0],
                    "goldstein": pd.to_numeric([r.get("GoldsteinScale")], errors="coerce")[0],
                    "mentions": pd.to_numeric([r.get("NumMentions")], errors="coerce")[0],
                    "url": url,
                })
        if n_pages % 25 == 0:
            logger.info("BigQuery streaming: %d pages, %d rows scanned.", n_pages, n_rows)
            _flush_buffers(buffers, out_dir)
    _flush_buffers(buffers, out_dir)
    billed = int(job.total_bytes_billed or 0)
    quota["billed_bytes"] = int(quota.get("billed_bytes", 0)) + billed
    atomic_write_json(quota_path, quota)
    logger.info("BigQuery IN-actor scan: %d rows scanned, %.1f GB billed.", n_rows, billed / 1e9)

    summary = {"rows_scanned": int(n_rows), "gb_billed": round(billed / 1e9, 2), "symbols": {}}
    for symbol, _ in targets:
        out = out_dir / f"{_safe(symbol)}.parquet"
        try:
            summary["symbols"][symbol] = int(len(pd.read_parquet(out))) if out.exists() else 0
        except Exception:
            summary["symbols"][symbol] = 0
    matched = sum(summary["symbols"].values())
    logger.info("Matched %d symbol-events across %d symbols.", matched, len(summary["symbols"]))
    atomic_write_json(done_marker, {"date": date.today().isoformat(), "summary": summary})
    update_state(paths.state, "gdelt_bq", status="complete", summary=summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="GDELT BigQuery historical backfill (single-shot).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_gdelt_bq_fetch(force=args.force)


if __name__ == "__main__":
    main()
