from __future__ import annotations

"""Announcement context features (Track U ablation, F-06/F-04).

ONE variable: NSE corporate announcements aggregated by decision_date
(features/decision_date.py — 15:30 IST cutoff, NSE sessions, NO extra
shift(1); the shift is baked into the mapping).

Family (9 features, company scope — pre-tagged scrip, §5.2):
  ann_count / ann_count_5d / ann_count_20d — trailing sums by decision_date
  days_since_last_ann — recency (NaN when none ever → NAN_OK, B-07)
  ann_event_earnings / acquisition / analyst / regulatory — subject-mapped
  ann_available — trailing-252d coverage flag (missing ≠ neutral, B-07)

Counts of zero are TRUE zeros (no announcement that day), not imputed
neutrals. Group/sector fan-out (B-06/F-05) is a separate later step —
company scope only in this ablation.
"""

import pandas as pd

from .decision_date import decision_dates

EVENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ann_event_earnings": ("result", "financial", "dividend", "bonus", "split", "buyback", "payout"),
    "ann_event_acquisition": ("acquis", "merger", "amalgamat", "stake", "takeover", "divest", "disposal", "slump sale"),
    "ann_event_analyst": ("analyst", "investor meet", "con. call", "concall", "presentation", "investor day"),
    "ann_event_regulatory": ("sebi", "disclosure under", "insider trading", "show-cause", "show cause", "penalty", "takeover regulations"),
}


def map_event_flags(subjects: pd.Series) -> pd.DataFrame:
    """NSE desc → coarse event flags. Pure (testable without files)."""
    lowered = subjects.fillna("").astype(str).str.lower()
    out = pd.DataFrame(index=subjects.index)
    for col, keywords in EVENT_KEYWORDS.items():
        out[col] = lowered.apply(lambda s: float(any(k in s for k in keywords)))
    return out


def aggregate_announcements(
    ann: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    symbol: str,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Decision-date announcement features aligned to trading sessions. Pure core."""
    index = pd.DatetimeIndex(trading_index).tz_localize(None).normalize()
    base = pd.DataFrame(index=index)
    base.index.name = "timestamp"
    cols = ["ann_count", "ann_count_5d", "ann_count_20d", "days_since_last_ann",
            *EVENT_KEYWORDS, "ann_available"]
    if ann is None or ann.empty or calendar is None or len(calendar) == 0:
        out = base.copy()
        out["ann_count"] = 0.0
        out["ann_count_5d"] = 0.0
        out["ann_count_20d"] = 0.0
        out["days_since_last_ann"] = float("nan")
        for col in EVENT_KEYWORDS:
            out[col] = 0.0
        out["ann_available"] = 0.0
        return out[cols]

    frame = ann.copy()
    if "symbol" in frame.columns:
        frame = frame[frame["symbol"].astype(str) == str(symbol)]
    if frame.empty:
        return aggregate_announcements(None, trading_index, symbol, calendar)
    frame = frame.copy()
    frame["decision_date"] = decision_dates(frame, calendar, col="published_utc")
    frame = frame.dropna(subset=["decision_date"])
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.tz_localize(None).dt.normalize()

    daily = frame.groupby("decision_date").size().rename("ann_count")
    out = base.join(daily, how="left")
    out["ann_count"] = out["ann_count"].fillna(0.0)
    out["ann_count_5d"] = out["ann_count"].rolling(5, min_periods=1).sum()
    out["ann_count_20d"] = out["ann_count"].rolling(20, min_periods=1).sum()
    last = frame.groupby("decision_date").size().index
    last_seen = pd.Series(pd.NaT, index=out.index)
    sorted_last = pd.DatetimeIndex(last).sort_values()
    pos = sorted_last.searchsorted(out.index, side="right") - 1
    has = pos >= 0
    last_seen[has] = sorted_last[pos[has]]
    out["days_since_last_ann"] = (out.index.to_series(index=out.index) - pd.to_datetime(last_seen)).dt.days.astype(float)
    out.loc[~has, "days_since_last_ann"] = float("nan")

    flags = map_event_flags(frame["subject"] if "subject" in frame.columns else pd.Series("", index=frame.index))
    frame = pd.concat([frame, flags], axis=1)
    for col in EVENT_KEYWORDS:
        out[col] = frame.groupby("decision_date")[col].max().reindex(out.index).fillna(0.0)
    out["ann_available"] = (out["ann_count"].rolling(252, min_periods=1).sum() > 0).astype(float)
    return out[cols]
