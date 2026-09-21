from __future__ import annotations

"""I0-D1b (earnings proximity) + I1-D1 (lead-lag) read-only slices.

Both join already-ingested data against FROZEN F-adopted predictions —
no retraining, no feature changes. Pre-registered bars (§0.2): every delta
is judged against 2×SE computed from the bucket sample sizes by a fixed
formula written here, not chosen after seeing numbers.

I0-D1b: days-since-last-earnings (announcement subject → earnings flag by
  decision_date) buckets 0-5/6-10/11-20/21+. Bar: UP-miss delta 0-5 vs 21+
  > 2×SE_diff. Tests PEAD directly.
I1-D1: per sector, leader (top-quintile trailing-60d-median turnover proxy)
  vs laggards (bottom quintile). Leader day-t-1 return regime (strong-up /
  other, scaled by its own 20d vol) vs laggard UP-miss at t. Bar: delta >
  2×SE_diff. Temporally distinct from same-day sector relatives (t-1 vs t).
"""

import argparse

import numpy as np
import pandas as pd

from ..config import load_settings
from ..utils.io import atomic_write_json
from ..utils.logging import setup_logging
from ..utils.state import update_state
from .touch_ledger import log_touch
from .u2_d1 import load_membership
from .up_autopsy import era_split

UP = 2
MIN_BUCKET_UP = 200


def se_2sample(p1: float, n1: int, p2: float, n2: int) -> float:
    """2×SE of a difference in proportions. Pure. THE pre-registered bar."""
    if n1 < 10 or n2 < 10:
        return float("inf")
    return 2.0 * float(np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))


def bucket_delta_report(buckets: dict[str, dict], base: str, focus: str) -> dict:
    """Delta focus-vs-base with pre-registered bar. Pure.

    buckets: name → {"up_rows": n, "miss_rate": p}. Returns delta, bar,
    verdict. Unreliable buckets (<MIN_BUCKET_UP UP rows) force FAIL-closed.
    """
    b, f = buckets.get(base, {}), buckets.get(focus, {})
    if b.get("up_rows", 0) < MIN_BUCKET_UP or f.get("up_rows", 0) < MIN_BUCKET_UP:
        return {"delta": None, "bar_2se": None, "pass": False, "reason": "insufficient rows"}
    delta = float(f["miss_rate"] - b["miss_rate"])
    bar = se_2sample(f["miss_rate"], f["up_rows"], b["miss_rate"], b["up_rows"])
    return {"delta": delta, "bar_2se": bar, "pass": bool(abs(delta) > bar)}


def earnings_dates(ann: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Decision-dates of earnings-flagged announcements per symbol. Pure core."""
    from ..features.announcement_features import map_event_flags
    from ..features.decision_date import decision_dates
    work = ann.copy()
    work["decision_date"] = decision_dates(work, calendar, col="published_utc")
    work = work.dropna(subset=["decision_date"])
    flags = map_event_flags(work["subject"] if "subject" in work.columns else pd.Series("", index=work.index))
    work = work[flags["ann_event_earnings"] > 0].copy()
    work["dd"] = pd.to_datetime(work["decision_date"]).dt.normalize()
    work["sym"] = work["symbol"].astype(str)
    return work[["dd", "sym"]].drop_duplicates()


def days_since_last(dates: pd.DatetimeIndex, events: pd.DatetimeIndex) -> np.ndarray:
    """For each date, days since latest event strictly before it. Pure."""
    ev = pd.DatetimeIndex(events).sort_values()
    pos = ev.searchsorted(dates, side="left") - 1
    out = np.full(len(dates), np.nan)
    has = pos >= 0
    out[has] = (dates[has] - ev[pos[has]]).days
    return out


def earnings_bucket(days: np.ndarray) -> np.ndarray:
    """0-5 / 6-10 / 11-20 / 21+ / none. Pure."""
    out = np.full(len(days), "none", dtype=object)
    out[(days >= 0) & (days <= 5)] = "0-5"
    out[(days >= 6) & (days <= 10)] = "6-10"
    out[(days >= 11) & (days <= 20)] = "11-20"
    out[days >= 21] = "21+"
    return out


def leadlag_buckets(pred: pd.DataFrame, price_ret: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """Laggard rows labeled by leader's day-t-1 regime. Pure core.

    price_ret: sessions × symbols daily returns. members: symbol→(sector,).
    Leader per sector = top-quintile trailing-60d-median turnover (passed in
    via members frame with a `leader` bool column). Regime: leader ret_{t-1}
    scaled by its trailing-20d vol: >=+0.5 → up, <=-0.5 → down, else flat.
    Returns bucket labels aligned to pred rows (NaN where unassigned).
    """
    out = pd.Series(pd.NA, index=pred.index, dtype=object)
    leaders = members[members["leader"]].groupby("sector")["symbol"].apply(list)
    laggards = members[~members["leader"]].groupby("sector")["symbol"].apply(list)
    vol = price_ret.rolling(20, min_periods=10).std().shift(1)
    for sector, lead_syms in leaders.items():
        lag_syms = laggards.get(sector, [])
        if not lead_syms or not lag_syms:
            continue
        lead_ret = price_ret[lead_syms].mean(axis=1)
        lead_vol = vol[lead_syms].mean(axis=1).replace(0, np.nan)
        regime = pd.Series(pd.NA, index=price_ret.index, dtype=object)
        z = lead_ret / lead_vol
        regime[z >= 0.5] = "leader-up"
        regime[z <= -0.5] = "leader-down"
        regime[z.abs() < 0.5] = "leader-flat"
        sub = pred[pred["symbol"].isin(lag_syms)].copy()
        if sub.empty:
            continue
        t_1 = pd.to_datetime(sub["timestamp"]).dt.normalize() - pd.Timedelta(days=1)
        # Map t-1 to previous SESSION (weekends/holidays have no leader bar).
        sessions = price_ret.index
        pos = sessions.searchsorted(t_1, side="right") - 1
        valid = pos >= 0
        reg = pd.Series(pd.NA, index=sub.index, dtype=object)
        reg[valid] = regime.iloc[pos[valid]].to_numpy()
        out.loc[sub.index] = reg
    return out


def run_i01_d1() -> dict:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "i01_d1.log")
    ref_dir = paths.data_raw.parent / "reference"
    pred = pd.read_parquet(paths.results / "predictions.parquet").reset_index(drop=True)
    is_up = (pred["actual_label"].to_numpy() == UP)
    missed_up = is_up & (pred["predicted_label"] != UP)
    eras = era_split(pred)
    result: dict = {"n_test": int(len(pred)), "i0_earnings": {}, "i1_leadlag": {}}

    # ---- I0-D1b: earnings proximity ----
    from ..data.trading_calendar import load_calendar
    from .u2_d1 import load_all_announcements
    calendar = load_calendar(ref_dir)
    ann = load_all_announcements(paths.data_raw / "announcements")
    earn = earnings_dates(ann, calendar)
    by_sym: dict[str, pd.DatetimeIndex] = {
        s: pd.DatetimeIndex(g["dd"].sort_values().unique())
        for s, g in earn.groupby("sym")
    }
    days = np.full(len(pred), np.nan)
    for sym, g in pred.groupby("symbol"):
        ev = by_sym.get(str(sym), pd.DatetimeIndex([]))
        if len(ev):
            dates = pd.DatetimeIndex(pd.to_datetime(g["timestamp"]).dt.normalize())
            days[g.index] = days_since_last(dates, ev)
    buckets = earnings_bucket(days)
    bstat: dict[str, dict] = {}
    for name in ("0-5", "6-10", "11-20", "21+", "none"):
        m = (buckets == name)
        up_m = m & is_up
        n_up = int(up_m.sum())
        bstat[name] = {"rows": int(m.sum()), "up_rows": n_up,
                       "miss_rate": float(missed_up.to_numpy()[up_m].mean()) if n_up else None}
    era_ok = True
    for name in ("0-5", "21+"):
        m = (buckets == name)
        era_rates = []
        for e in pd.unique(eras.to_numpy()):
            in_era = (eras.to_numpy() == e) & m & is_up
            if in_era.sum() >= MIN_BUCKET_UP // 4:
                era_rates.append(float(missed_up.to_numpy()[in_era].mean()))
        if len(era_rates) >= 3 and (max(era_rates) - min(era_rates)) >= 0.25:
            era_ok = False
    verdict = bucket_delta_report(bstat, "21+", "0-5")
    verdict["cross_era_stable"] = bool(era_ok)
    result["i0_earnings"] = {"buckets": bstat, "verdict": verdict}
    logger.info("I0 earnings 0-5 vs 21+: %s", verdict)

    # ---- I1-D1: lead-lag ----
    price_ret = {}
    for symbol in settings.symbols:
        safe = symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        path = paths.data_raw / f"{safe}.parquet"
        if path.exists():
            price_ret[symbol] = pd.read_parquet(path)["close"].pct_change()
    price_ret = pd.DataFrame(price_ret).sort_index()
    price_ret.index = pd.to_datetime(price_ret.index).tz_localize(None).normalize()
    turnover_med = {}
    for symbol in settings.symbols:
        safe = symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        path = paths.data_features / f"{safe}_features.parquet"
        if path.exists():
            f = pd.read_parquet(path)
            if "turnover_cr_proxy" in f.columns:
                idx = pd.to_datetime(f.index).tz_localize(None).normalize()
                turnover_med[symbol] = pd.Series(pd.to_numeric(f["turnover_cr_proxy"], errors="coerce").to_numpy(), index=idx).rolling(60, min_periods=20).median().shift(1)
    turnover_med = pd.DataFrame(turnover_med).sort_index()
    sectors, _ = load_membership(ref_dir)
    members = []
    for sym in settings.symbols:
        ts_all = pd.to_datetime(pred.loc[pred["symbol"] == sym, "timestamp"]).dt.normalize()
        if len(ts_all) == 0:
            continue
        b = None
        for bucket, fr, to in sectors.get(sym, []):
            if fr <= ts_all.max():
                b = bucket
        members.append({"symbol": sym, "sector": b or "Unknown"})
    members = pd.DataFrame(members)
    # Leader = top-quintile trailing turnover within sector (median over test window).
    members["turn"] = members["symbol"].map(
        {s: float(turnover_med[s].loc[turnover_med.index.intersection(pd.to_datetime(pred['timestamp']).dt.normalize())].median()) if s in turnover_med else float("nan")
         for s in members["symbol"]})
    members["leader"] = members.groupby("sector")["turn"].transform(
        lambda s: s >= s.quantile(0.8))
    lab = leadlag_buckets(pred, price_ret, members)
    lstat: dict[str, dict] = {}
    for name in ("leader-up", "leader-down", "leader-flat"):
        m = (lab == name).to_numpy()
        up_m = m & is_up
        n_up = int(up_m.sum())
        lstat[name] = {"rows": int(m.sum()), "up_rows": n_up,
                       "miss_rate": float(missed_up.to_numpy()[up_m].mean()) if n_up else None}
    verdict1 = bucket_delta_report(
        {k: v for k, v in lstat.items()}, "leader-flat", "leader-up")
    result["i1_leadlag"] = {"buckets": lstat, "verdict": verdict1,
                            "leaders": members[members["leader"]].groupby("sector")["symbol"].apply(list).to_dict()}
    logger.info("I1 leader-up vs flat: %s", verdict1)

    atomic_write_json(paths.results / "i01_d1.json", result)
    log_touch(paths, "I0", "D1", "earnings-proximity slice", int(len(pred)), "results/i01_d1.json")
    log_touch(paths, "I1", "D1", "lead-lag slice", int(len(pred)), "results/i01_d1.json")
    update_state(paths.state, "i01_d1", status="complete")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="I0-D1b + I1-D1 read-only slices.")
    parser.parse_args()
    run_i01_d1()


if __name__ == "__main__":
    main()
