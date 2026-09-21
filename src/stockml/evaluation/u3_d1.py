from __future__ import annotations

"""U3-D1 (read-only): does announcement CONTENT predict UP misses?

U2-D1 tested presence/scope (null). This tests sentiment-signed content:
FinBERT scores (cached by `ann-score`, §5.6 score-once-cache-forever) joined
against frozen predictions, sliced by sentiment sign and magnitude on UP
rows specifically. Same join-query discipline: no retrain, cross-era check.

If sentiment slices also come back null, the structural-ceiling claim for
UP covers both presence AND content, and Door C for UP is airtight.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..config import load_settings
from ..utils.io import atomic_write_json
from ..utils.logging import setup_logging
from ..utils.state import update_state
from .u2_d1 import load_all_announcements
from .up_autopsy import era_split

UP = 2
MIN_SLICE_ROWS = 200
POS_THR = 0.2
NEG_THR = -0.2


def sentiment_context(pred: pd.DataFrame, ann: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-test-row sentiment of same-symbol announcements on decision_date. Pure core.

    Columns: sent_mean (NaN when none), sent_max, sent_min, sent_count.
    """
    from ..features.decision_date import decision_dates
    work = ann.copy()
    work["decision_date"] = decision_dates(work, calendar, col="published_utc")
    work = work.dropna(subset=["decision_date"])
    work["dd"] = pd.to_datetime(work["decision_date"]).dt.normalize()
    work["sym"] = work["symbol"].astype(str)
    sc = pd.to_numeric(work["finbert_score"], errors="coerce")
    grouped = work.assign(_sc=sc).groupby(["dd", "sym"])["_sc"]
    agg_mean = grouped.mean()
    agg_max = grouped.max()
    agg_min = grouped.min()
    agg_n = grouped.size()

    pred = pred.copy()
    pred["_dd"] = pd.to_datetime(pred["timestamp"]).dt.normalize()
    means, maxs, mins, ns = [], [], [], []
    for dd, sym in zip(pred["_dd"], pred["symbol"].astype(str)):
        key = (dd, sym)
        if key in agg_mean.index:
            means.append(float(agg_mean.loc[key]))
            maxs.append(float(agg_max.loc[key]))
            mins.append(float(agg_min.loc[key]))
            ns.append(int(agg_n.loc[key]))
        else:
            means.append(float("nan"))
            maxs.append(float("nan"))
            mins.append(float("nan"))
            ns.append(0)
    return pd.DataFrame({"sent_mean": means, "sent_max": maxs, "sent_min": mins, "sent_count": ns},
                        index=pred.index)


def sign_bucket(mean: float) -> str:
    """Sentiment sign bucket. Pure."""
    if pd.isna(mean):
        return "none"
    if mean >= POS_THR:
        return "positive"
    if mean <= NEG_THR:
        return "negative"
    return "neutral"


def run_u3_d1() -> dict:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "u3_d1.log")
    pred = pd.read_parquet(paths.results / "predictions.parquet")
    ann = load_all_announcements(paths.data_raw / "announcements")
    scored = ann[ann["finbert_score"].notna()] if "finbert_score" in ann.columns else ann.iloc[0:0]
    if scored.empty:
        raise RuntimeError("No scored announcements; run ann-score first.")
    from ..data.trading_calendar import load_calendar
    calendar = load_calendar(paths.data_raw.parent / "reference")

    ctx = sentiment_context(pred, scored, calendar)
    is_up = pred["actual_label"] == UP
    missed_up = is_up & (pred["predicted_label"] != UP)
    p_up = pred["p_up"]
    eras = era_split(pred)
    buckets = ctx["sent_mean"].apply(sign_bucket)

    try:
        overall_auc = float(roc_auc_score(is_up.astype(int), p_up.to_numpy()))
    except ValueError:
        overall_auc = None
    result: dict = {
        "n_test": int(len(pred)),
        "n_scored_ann": int(len(scored)),
        "rows_with_sentiment": int(ctx["sent_count"].gt(0).sum()),
        "up_miss_overall": float(missed_up[is_up].mean()),
        "up_auc_overall": overall_auc,
        "slices": {},
    }
    for name in ("positive", "negative", "neutral", "none"):
        m = (buckets == name).to_numpy()
        up_m = m & is_up.to_numpy()
        rep: dict = {"rows": int(m.sum()), "up_rows": int(up_m.sum()),
                     "reliable": bool(m.sum() >= MIN_SLICE_ROWS)}
        rep["up_miss_rate"] = float(missed_up.to_numpy()[up_m].mean()) if up_m.any() else None
        if up_m.sum() >= MIN_SLICE_ROWS and up_m.sum() < m.sum():
            try:
                rep["up_auc"] = float(roc_auc_score(is_up.to_numpy()[m], p_up.to_numpy()[m]))
            except ValueError:
                rep["up_auc"] = None
        else:
            rep["up_auc"] = None
        era_v = eras.to_numpy()
        per_era = {}
        for e in pd.unique(era_v):
            in_era = era_v == e
            denom = int((is_up.to_numpy() & in_era & m).sum())
            if denom >= MIN_SLICE_ROWS // 4:
                per_era[str(e)] = {"rows": denom,
                                   "miss_rate": float((missed_up.to_numpy() & in_era & m).sum() / denom)}
        rep["per_era"] = per_era
        rates = [v["miss_rate"] for v in per_era.values()]
        rep["cross_era"] = bool(len(rates) >= 3 and (max(rates) - min(rates)) < 0.25)
        result["slices"][name] = rep
        logger.info("sent=%s rows=%d up_miss=%s up_auc=%s xera=%s", name, rep["rows"],
                    round(rep["up_miss_rate"], 4) if rep["up_miss_rate"] else None,
                    round(rep["up_auc"], 4) if rep["up_auc"] else None, rep["cross_era"])
    # Magnitude check: strongly-signed (|mean|>=0.5) vs weakly-signed rows.
    strong = ctx["sent_mean"].abs() >= 0.5
    for name, mask in (("strong", strong.fillna(False).to_numpy()),
                       ("weak", ((ctx["sent_mean"].abs() < 0.5) & ctx["sent_count"].gt(0)).to_numpy())):
        up_m = mask & is_up.to_numpy()
        result["slices"][name] = {
            "rows": int(mask.sum()), "up_rows": int(up_m.sum()),
            "up_miss_rate": float(missed_up.to_numpy()[up_m].mean()) if up_m.any() else None,
        }
    atomic_write_json(paths.results / "u3_d1.json", result)
    update_state(paths.state, "u3_d1", status="complete", report=str(paths.results / "u3_d1.json"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="U3-D1: sentiment content vs UP-miss, read-only.")
    parser.parse_args()
    run_u3_d1()


if __name__ == "__main__":
    main()
