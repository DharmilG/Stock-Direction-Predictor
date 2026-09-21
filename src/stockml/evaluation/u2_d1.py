from __future__ import annotations

"""U2-D1 (read-only): does broader-scope announcement activity predict UP misses?

Joins group/sector maps against the 92,960 ingested announcements and asks,
on TEST rows of the frozen predictions:
  * is the UP-miss rate different on days with group/sector/market-wide
    announcement activity vs days without?
  * is UP-vs-rest AUC better on the high-activity subset (where §5.1 theory
    predicts scope effects should show) than overall?

Same slicing discipline as D1: buckets need ≥200 rows, patterns must hold
across eras. No retrain, no feature change, no confound with anything.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..config import load_settings
from ..utils.io import atomic_write_json
from ..utils.logging import setup_logging
from ..utils.state import update_state
from .up_autopsy import era_split

UP = 2
MIN_SLICE_ROWS = 200


def load_all_announcements(raw_dir) -> pd.DataFrame:
    """Concat per-symbol announcement parquets. Pure-ish (reads only)."""
    from pathlib import Path
    frames = []
    for path in sorted(Path(raw_dir).glob("*.parquet")):
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_membership(ref_dir) -> tuple[dict, dict]:
    """symbol → [(bucket, from, to)] for sector and group. Pure."""
    from pathlib import Path
    ref_dir = Path(ref_dir)
    sectors: dict[str, list] = {}
    try:
        sm = json.loads((ref_dir / "sector_map.json").read_text(encoding="utf-8"))
        for sym, spec in sm.items():
            if str(sym).startswith("_") or not isinstance(spec, dict):
                continue
            sectors.setdefault(sym, []).append(
                (str(spec.get("sector", "Unknown")),
                 pd.Timestamp(spec.get("from") or "2010-01-01"),
                 pd.Timestamp(spec.get("to")) if spec.get("to") else pd.Timestamp.max))
    except FileNotFoundError:
        pass
    groups: dict[str, list] = {}
    try:
        gm = json.loads((ref_dir / "group_map.json").read_text(encoding="utf-8"))
        for gid, gspec in gm.items():
            if str(gid).startswith("_") or not isinstance(gspec, dict):
                continue
            for m in gspec.get("members", []):
                groups.setdefault(m.get("symbol"), []).append(
                    (str(gid),
                     pd.Timestamp(m.get("from") or "2010-01-01"),
                     pd.Timestamp(m.get("to")) if m.get("to") else pd.Timestamp.max))
    except FileNotFoundError:
        pass
    return sectors, groups


def bucket_at(memberships: list, ts: pd.Timestamp) -> str | None:
    """Active bucket for a symbol at a timestamp, else None. Pure."""
    for bucket, fr, to in (memberships or []):
        if fr <= ts <= to:
            return bucket
    return None


def scope_activity(ann: pd.DataFrame, sectors: dict, groups: dict) -> pd.DataFrame:
    """Per-announcement scope tags. Pure.

    Returns frame with decision_date, group_tag (group id or None),
    sector_tag. Market scope = every announcement (counted separately).
    Expects ann to already carry a decision_date column.
    """
    out = ann.copy()
    ts = pd.to_datetime(out["decision_date"])
    out["group_tag"] = [
        bucket_at(groups.get(sym, []), t) for sym, t in zip(out["symbol"].astype(str), ts)
    ]
    out["sector_tag"] = [
        bucket_at(sectors.get(sym, []), t) for sym, t in zip(out["symbol"].astype(str), ts)
    ]
    return out


def activity_for_rows(pred: pd.DataFrame, tagged: pd.DataFrame,
                      sectors: dict, groups: dict) -> pd.DataFrame:
    """Scope activity per test row. Pure.

    group_act  = # announcements that decision-day from OTHER members of the
                 row symbol's group (spillover, excl. self)
    sector_act = same across the row symbol's sector (excl. self)
    market_act = # announcements market-wide that decision-day
    """
    tagged = tagged.copy()
    tagged["dd"] = pd.to_datetime(tagged["decision_date"]).dt.normalize()
    by_group = tagged[tagged["group_tag"].notna()].groupby(["dd", "group_tag"]).size()
    by_sector = tagged[tagged["sector_tag"].notna()].groupby(["dd", "sector_tag"]).size()
    by_market = tagged.groupby("dd").size()
    # Self counts to subtract (own announcements, same decision day).
    own = tagged.groupby([tagged["dd"], tagged["symbol"].astype(str)]).size()

    pred = pred.copy()
    pred["_dd"] = pd.to_datetime(pred["timestamp"]).dt.normalize()
    sym_of = pred["symbol"].astype(str)
    ts_of = pd.to_datetime(pred["timestamp"])

    g_act, s_act, m_act = [], [], []
    for dd, sym, t in zip(pred["_dd"], sym_of, ts_of):
        g = bucket_at(groups.get(sym, []), t)
        s = bucket_at(sectors.get(sym, []), t)
        g_act.append(float(by_group.get((dd, g), 0) - own.get((dd, sym), 0)) if g else 0.0)
        s_act.append(float(by_sector.get((dd, s), 0) - own.get((dd, sym), 0)) if s else 0.0)
        m_act.append(float(by_market.get(dd, 0)))
    return pd.DataFrame({"group_act": g_act, "sector_act": s_act, "market_act": m_act}, index=pred.index)


def subset_report(name: str, mask: pd.Series, is_up: pd.Series,
                  missed_up: pd.Series, p_up: pd.Series,
                  eras: pd.Series) -> dict:
    """Miss rate + UP-AUC on a subset with cross-era check. Pure."""
    m = mask.fillna(False).astype(bool)
    n = int(m.sum())
    rep: dict = {"rows": n, "reliable": bool(n >= MIN_SLICE_ROWS)}
    up_m = m & is_up
    rep["up_rows"] = int(up_m.sum())
    rep["up_miss_rate"] = float(missed_up[up_m].mean()) if up_m.any() else None
    if up_m.sum() >= MIN_SLICE_ROWS and is_up[m].sum() >= 50 and (~is_up[m]).sum() >= 50:
        try:
            rep["up_auc"] = float(roc_auc_score(is_up[m].astype(int), p_up[m].to_numpy()))
        except ValueError:
            rep["up_auc"] = None
    else:
        rep["up_auc"] = None
    per_era = {}
    era_v = eras.to_numpy()
    is_up_v = is_up.to_numpy(dtype=bool)
    missed_v = missed_up.to_numpy(dtype=bool)
    for e in pd.unique(era_v):
        in_era = era_v == e
        denom = int((is_up_v & in_era).sum())
        if denom >= MIN_SLICE_ROWS // 4:
            per_era[str(e)] = {"rows": denom,
                               "miss_rate": float((missed_v & in_era).sum() / denom)}
    rep["per_era"] = per_era
    rates = [v["miss_rate"] for v in per_era.values()]
    rep["cross_era"] = bool(len(rates) >= 3 and (max(rates) - min(rates)) < 0.25)
    return {name: rep}


def run_u2_d1() -> dict:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "u2_d1.log")
    pred = pd.read_parquet(paths.results / "predictions.parquet")
    ref_dir = paths.data_raw.parent / "reference"

    from ..data.trading_calendar import load_calendar
    from ..features.decision_date import decision_dates
    calendar = load_calendar(ref_dir)
    ann = load_all_announcements(paths.data_raw / "announcements")
    if ann.empty or len(calendar) == 0:
        raise RuntimeError("Announcements or calendar missing; run announcements + calendar first.")
    ann = ann.copy()
    ann["decision_date"] = decision_dates(ann, calendar, col="published_utc")
    ann = ann.dropna(subset=["decision_date"])
    sectors, groups = load_membership(ref_dir)
    tagged = scope_activity(ann, sectors, groups)
    act = activity_for_rows(pred, tagged, sectors, groups)

    is_up = pred["actual_label"] == UP
    missed_up = is_up & (pred["predicted_label"] != UP)
    p_up = pred["p_up"]
    eras = era_split(pred)

    result: dict = {
        "n_test": int(len(pred)),
        "n_announcements": int(len(ann)),
        "up_miss_overall": float(missed_up[is_up].mean()),
        "subsets": {},
    }
    try:
        result["up_auc_overall"] = float(roc_auc_score(is_up.astype(int), p_up.to_numpy()))
    except ValueError:
        result["up_auc_overall"] = None
    mk = int(act["market_act"].quantile(0.80))
    result["market_high_threshold"] = mk
    result["subsets"].update(subset_report("group_active", act["group_act"] > 0, is_up, missed_up, p_up, eras))
    result["subsets"].update(subset_report("group_quiet", act["group_act"] == 0, is_up, missed_up, p_up, eras))
    result["subsets"].update(subset_report("sector_active", act["sector_act"] > 0, is_up, missed_up, p_up, eras))
    result["subsets"].update(subset_report("sector_quiet", act["sector_act"] == 0, is_up, missed_up, p_up, eras))
    result["subsets"].update(subset_report("market_high", act["market_act"] >= mk, is_up, missed_up, p_up, eras))
    result["subsets"].update(subset_report("market_low", act["market_act"] < mk, is_up, missed_up, p_up, eras))

    atomic_write_json(paths.results / "u2_d1.json", result)
    update_state(paths.state, "u2_d1", status="complete", report=str(paths.results / "u2_d1.json"))
    for name, rep in result["subsets"].items():
        logger.info("%s: rows=%d up_miss=%.3f up_auc=%s xera=%s",
                    name, rep["rows"], rep.get("up_miss_rate") or 0.0,
                    round(rep["up_auc"], 4) if rep.get("up_auc") else None, rep.get("cross_era"))
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="U2-D1: scope-activity vs UP-miss, read-only.")
    parser.parse_args()
    run_u2_d1()


if __name__ == "__main__":
    main()
