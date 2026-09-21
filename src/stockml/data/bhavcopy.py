from __future__ import annotations

"""NSE security-wise bhavcopy delivery/turnover fetch (free, local).

Source: https://archives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv
Columns: SYMBOL,SERIES,DATE1,PREV_CLOSE,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,
         LAST_PRICE,CLOSE_PRICE,AVG_PRICE,TTL_TRD_QNTY,TURNOVER_LACS,
         NO_OF_TRADES,DELIV_QTY,DELIV_PER

Archive truth (probed 2026-09-18):
- sec_bhavdata_full (with DELIV_QTY/DELIV_PER) exists from ~2020 onward.
  2015-2019 return HTTP 404 on the free archive.
- Old cm-JSON bhav zips exist back to 2010 but carry NO delivery columns.
So: turnover proxy (close*volume) covers 2010+ from yfinance OHLCV;
delivery_* covers 2020+ where the free archive has it, with
delivery_available=0/1 flag distinguishing missing from neutral (B-07 rule).
Circuit price-band history is not published free day-wise; we store a
lock proxy (close==high==low intraday lock) instead of inventing bands.
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from ..config import load_settings
from ..utils.io import atomic_write_parquet
from ..utils.logging import setup_logging
from ..utils.state import update_state

BASE_URL = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{dmy}.csv"
# NSE blocks bare clients; prime cookies with a homepage visit first.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/csv,*/*",
    "Referer": "https://www.nseindia.com/",
}
_ATTEMPTS = 3
_BACKOFF = (5, 15, 45)
# Politeness between daily requests; NSE rate-limits rapid sequential hits
# with HTTP 200 block pages (no SERIES column). Hammering through a block
# keeps every subsequent day failing — see logs 2026-09-18.
_DAY_GAP_SECONDS = 4
_COOLDOWN_AFTER_CONSEC_FAILS = 5
_COOLDOWN_SECONDS = 180
_ABORT_AFTER_CONSEC_FAILS = 15


class BhavBlocked(RuntimeError):
    """NSE returned a block/anti-bot page instead of CSV."""


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=20)
    except Exception:
        pass
    return s


def fetch_bhav_day(day: date, session: requests.Session | None = None) -> pd.DataFrame | None:
    """Download one day's sec bhavcopy. Returns EQ-series frame or None (holiday/404).

    Raises BhavBlocked when NSE returns an anti-bot page (HTTP 200 with HTML
    or CSV lacking the SERIES column) so the caller can cool down and renew
    the session instead of hammering through the block.
    """
    s = session or _session()
    dmy = day.strftime("%d%m%Y")
    url = BASE_URL.format(dmy=dmy)
    last_err: Exception | None = None
    for attempt in range(_ATTEMPTS):
        try:
            r = s.get(url, timeout=45)
            if r.status_code == 404:
                return None  # holiday or pre-2020 (no delivery archive)
            r.raise_for_status()
            if len(r.content) < 500:
                return None
            from io import StringIO
            txt = r.content.decode("utf-8", errors="replace")
            if "<html" in txt[:2000].lower() or not txt.lstrip().startswith("SYMBOL"):
                logger = __import__("logging").getLogger(__name__)
                logger.warning("bhav %s non-CSV head: %r", day, txt[:200])
                raise BhavBlocked(f"non-CSV response for {day} (likely rate-limited)")
            df = pd.read_csv(StringIO(txt), skipinitialspace=True)
            df.columns = [str(c).strip() for c in df.columns]
            if "SERIES" not in df.columns:
                logger = __import__("logging").getLogger(__name__)
                logger.warning("bhav %s cols: %r head: %r", day, list(df.columns)[:8], txt[:200])
                raise BhavBlocked(f"no SERIES column for {day} (likely rate-limited)")
            df = df[df["SERIES"] == "EQ"].copy()
            if df.empty:
                return None
            df["DATE1"] = pd.to_datetime(df["DATE1"]).dt.tz_localize(None)
            return df
        except BhavBlocked:
            raise
        except Exception as exc:
            last_err = exc
            time.sleep(_BACKOFF[min(attempt, len(_BACKOFF) - 1)])
    raise RuntimeError(f"bhavcopy fetch failed for {day}: {last_err}")


def _safe(symbol: str) -> str:
    return symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")


def run_bhavcopy_fetch(start: date | None = None, end: date | None = None) -> dict:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "bhavcopy.log")
    out_dir = paths.data_raw / "bhavcopy"
    out_dir.mkdir(parents=True, exist_ok=True)

    end_d = end or date.today()
    # Delivery archive starts ~2020; earlier would 404 every day. Start there.
    start_d = start or date(2020, 1, 1)
    # Per-symbol incremental: resume from existing parquet max date, unless an
    # explicit --start was given (chunked backfills must honor it).
    per_symbol_last: dict[str, date] = {}
    for symbol in settings.symbols:
        p = out_dir / f"{_safe(symbol)}.parquet"
        if p.exists():
            try:
                ex = pd.read_parquet(p)
                if not ex.empty:
                    per_symbol_last[symbol] = pd.Timestamp(ex.index.max()).date()
            except Exception:
                pass

    if start is not None:
        day = start_d
    elif per_symbol_last:
        day = min(per_symbol_last.values()) + timedelta(days=1)
        day = max(day, start_d)
    else:
        day = start_d

    s = _session()
    # Accumulate rows per symbol in memory, then merge with existing.
    buffers: dict[str, list[pd.DataFrame]] = {sym: [] for sym in settings.symbols}
    sym_base = {sym: sym.replace(".NS", "") for sym in settings.symbols}

    def _flush() -> None:
        for sym in settings.symbols:
            if not buffers[sym]:
                continue
            p = out_dir / f"{_safe(sym)}.parquet"
            existing = None
            if p.exists():
                try:
                    existing = pd.read_parquet(p)
                except Exception:
                    existing = None
            new = pd.concat(buffers[sym])
            new = new[~new.index.duplicated(keep="last")].sort_index()
            merged = pd.concat([existing, new]) if existing is not None and not existing.empty else new
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            atomic_write_parquet(merged, p)
            buffers[sym] = []

    cur = day
    ndays = 0
    nhit = 0
    consec_fails = 0
    while cur <= end_d:
        if cur.weekday() < 5:  # weekdays only; holidays return None cheaply
            try:
                frame = fetch_bhav_day(cur, s)
                consec_fails = 0
            except BhavBlocked as exc:
                logger.warning("bhav %s blocked: %s", cur, exc)
                frame = None
                consec_fails += 1
            except Exception as exc:
                logger.warning("bhav %s failed: %s", cur, exc)
                frame = None
                consec_fails += 1
            if consec_fails >= _ABORT_AFTER_CONSEC_FAILS:
                logger.error("Aborting: %d consecutive failures at %s (NSE block?). Flushing progress.", consec_fails, cur)
                break
            if consec_fails and consec_fails % _COOLDOWN_AFTER_CONSEC_FAILS == 0:
                logger.warning("Cooling down %ds after %d consecutive failures; renewing session.", _COOLDOWN_SECONDS, consec_fails)
                _flush()
                time.sleep(_COOLDOWN_SECONDS)
                s = _session()
            ndays += 1
            if frame is not None:
                nhit += 1
                for sym in settings.symbols:
                    base = sym_base[sym]
                    row = frame[frame["SYMBOL"] == base]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    buffers[sym].append(pd.DataFrame({
                        "delivery_qty": [pd.to_numeric([r.get("DELIV_QTY")], errors="coerce")[0]],
                        "delivery_per": [pd.to_numeric([r.get("DELIV_PER")], errors="coerce")[0]],
                        "turnover_lacs": [pd.to_numeric([r.get("TURNOVER_LACS")], errors="coerce")[0]],
                        "no_trades": [pd.to_numeric([r.get("NO_OF_TRADES")], errors="coerce")[0]],
                        "ttl_qty": [pd.to_numeric([r.get("TTL_TRD_QNTY")], errors="coerce")[0]],
                    }, index=pd.DatetimeIndex([pd.Timestamp(cur)], name="timestamp")))
            if ndays % 60 == 0:
                logger.info("bhavcopy progress %s hits=%d/%d", cur, nhit, ndays)
                _flush()
            time.sleep(_DAY_GAP_SECONDS)
        cur += timedelta(days=1)

    _flush()
    summary = {"days_scanned": ndays, "days_hit": nhit, "symbols": {}}
    for sym in settings.symbols:
        p = out_dir / f"{_safe(sym)}.parquet"
        if p.exists():
            try:
                summary["symbols"][sym] = int(len(pd.read_parquet(p)))
                continue
            except Exception:
                pass
        summary["symbols"][sym] = 0
    update_state(paths.state, "bhavcopy", status="complete", start=str(day), end=str(end_d), summary=summary)
    logger.info("Bhavcopy fetch complete: %d/%d days hit.", nhit, ndays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NSE delivery/turnover bhavcopy (2020+).")
    parser.add_argument("--start", default="", help="YYYY-MM-DD (default 2020-01-01 or resume)")
    parser.add_argument("--end", default="", help="YYYY-MM-DD (default today)")
    args = parser.parse_args()
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    run_bhavcopy_fetch(start, end)


if __name__ == "__main__":
    main()
