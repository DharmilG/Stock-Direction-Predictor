from __future__ import annotations

"""NSE bulk/block deal daily ingestion (I2, forward-only).

History spike 2026-09-20: NSE publishes bulk/block as CURRENT-DAY snapshots
only (archives.nseindia.com/content/equities/{bulk,block}.csv); no dated
free archive exists (all dated patterns 404; BSE is JS-gated, no direct
CSV). So no backfill is possible — this ingests daily going forward from
first run. I2-D1 stays pending-coverage until months accumulate; no
backtest claim can be made before then.

Schema per row: date | symbol (NSE, no .NS) | client | side | qty | price.
Symbol mapping to .NS: bulk/block symbols are NSE symbols already.
"""

import argparse
import time
from datetime import date

import pandas as pd
import requests

from ..config import load_settings
from ..utils.io import atomic_write_parquet
from ..utils.logging import setup_logging
from ..utils.state import update_state

SOURCES = {
    "bulk": "https://archives.nseindia.com/content/equities/bulk.csv",
    "block": "https://archives.nseindia.com/content/equities/block.csv",
}
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/csv,*/*",
    "Referer": "https://www.nseindia.com/",
}
_ATTEMPTS = 3
_BACKOFF = (5, 15, 45)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=20)
    except Exception:
        pass
    return s


def parse_deals(text: str, source: str) -> pd.DataFrame:
    """NSE bulk/block CSV text → normalized frame. Pure (testable offline)."""
    from io import StringIO
    try:
        df = pd.read_csv(StringIO(text), skipinitialspace=True)
    except Exception:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    needed = {"Date", "Symbol", "Buy/Sell", "Quantity Traded"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce").dt.normalize(),
        "symbol": df["Symbol"].astype(str).str.strip(),
        "client": df.get("Client Name", "").astype(str) if "Client Name" in df.columns else "",
        "side": df["Buy/Sell"].astype(str).str.upper().str.strip(),
        "qty": pd.to_numeric(df["Quantity Traded"], errors="coerce"),
        "price": pd.to_numeric(df.get("Trade Price / Wght. Avg. Price"), errors="coerce"),
        "source": source,
    })
    out = out.dropna(subset=["date", "symbol"]).query("side in ('BUY', 'SELL') and qty > 0")
    return out.reset_index(drop=True)


def fetch_deals_day(session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch today's bulk+block snapshots. Returns normalized concat."""
    s = session or _session()
    frames = []
    for source, url in SOURCES.items():
        last_err: Exception | None = None
        for attempt in range(_ATTEMPTS):
            try:
                r = s.get(url, timeout=45)
                r.raise_for_status()
                txt = r.content.decode("utf-8", errors="replace")
                if "Date,Symbol" not in txt[:500]:
                    raise RuntimeError(f"non-CSV response for {source}")
                frames.append(parse_deals(txt, source))
                break
            except Exception as exc:
                last_err = exc
                time.sleep(_BACKOFF[min(attempt, len(_BACKOFF) - 1)])
        else:
            raise RuntimeError(f"deals fetch failed for {source}: {last_err}")
        time.sleep(3)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_bulk_fetch() -> dict:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "bulkdeals.log")
    out_dir = paths.data_raw / "bulkdeals"
    out_dir.mkdir(parents=True, exist_ok=True)
    day = date.today()
    frame = fetch_deals_day()
    out = out_dir / f"deals_{day.isoformat()}.parquet"
    if out.exists():
        try:
            frame = pd.concat([pd.read_parquet(out), frame], ignore_index=True)
        except Exception:
            pass
    frame = frame.drop_duplicates(subset=["date", "symbol", "client", "side", "qty"])
    atomic_write_parquet(frame, out)
    update_state(paths.state, "bulkdeals", status="complete", day=str(day), rows=int(len(frame)))
    logger.info("Bulk/block deals for %s: %d rows.", day, len(frame))
    return {"day": str(day), "rows": int(len(frame))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NSE bulk/block deals (forward-only).")
    parser.parse_args()
    run_bulk_fetch()


if __name__ == "__main__":
    main()
