"""Q0 analytics: pure pandas over the rating ledger, its traces and the logs.

Nothing here imports FastAPI. Every entry point takes the paths or frames it
needs, so the numbers can be pinned down in unit tests against a synthetic
ledger instead of whatever the last run happened to produce.

Two rules the numbers obey:

* A field is either measured or absent. If the run never wrote GPU utilisation
  the timeline reports ``fields: ["temp_c"]`` and warns — it never draws a zero
  line to fill the gap.
* Every derived series is recomputed from the ledger or the logs, never copied
  from a summary file, so two sources can be compared instead of conflated.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..qpipeline.paths import p0_from

from .dashboard_support import age_seconds, file_meta

VOTES = ("DOWN", "FLAT", "UP")
LABELS = ("DOWN", "FLAT", "UP")
SAMPLE_SECONDS = 5
RESOURCE_FIELDS = (
    "temp_c", "util_pct", "mem_used_mb", "mem_total_mb",
    "power_w", "fan_pct", "clock_sm_mhz", "clock_max_sm_mhz",
)
ROLLING_CASES = 50
LIVE_SECONDS = 600
MAX_SERIES_POINTS = 2000

# Trace summaries are cached by path+mtime: reading 700 JSON files costs a second,
# and the analytics tab is polled.
_TRACE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def judged_mask(frame: pd.DataFrame) -> pd.Series:
    """A row is *judged* when it carries a usable vote and did not fail the harness."""
    mask = pd.Series(True, index=frame.index)
    if "qwen_vote" in frame.columns:
        mask &= frame["qwen_vote"].astype(str).str.upper().isin(VOTES)
    if "harness_fail" in frame.columns:
        mask &= ~frame["harness_fail"].fillna(False).astype(bool)
    return mask


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _rate(mask: pd.Series) -> float | None:
    return round(float(mask.mean()), 4) if len(mask) else None


def _matrix(rows: pd.Series, cols: pd.Series, row_labels: tuple[str, ...], col_labels: tuple[str, ...]) -> dict:
    counts = {label: {other: 0 for other in col_labels} for label in row_labels}
    for row_value, col_value in zip(rows.astype(str), cols.astype(str)):
        if row_value in counts and col_value in counts[row_value]:
            counts[row_value][col_value] += 1
    totals = {label: int(sum(counts[label].values())) for label in row_labels}
    return {
        "rows": list(row_labels),
        "cols": list(col_labels),
        "counts": [[counts[label][other] for other in col_labels] for label in row_labels],
        "row_totals": totals,
    }


def _quantiles(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if not len(values):
        return {"n": 0, "p50": None, "p90": None, "mean": None}
    return {
        "n": int(len(values)),
        "p50": round(float(values.quantile(0.50)), 2),
        "p90": round(float(values.quantile(0.90)), 2),
        "mean": round(float(values.mean()), 2),
    }


# --------------------------------------------------------------------------- #
# Trace metadata
# --------------------------------------------------------------------------- #
_TRACE_KEEP = (
    "model", "prompt_version", "num_ctx", "num_predict", "temperature", "top_p",
    "vote", "sentiment", "confidence", "truncated", "no_news", "harness_fail",
    "fail_reason", "seconds", "attempts",
)


def trace_summary(path: Path) -> dict[str, Any] | None:
    """Small, cached digest of one trace file. ``None`` when it is unreadable."""
    meta = file_meta(path)
    if not meta["available"]:
        return None
    key = str(path)
    stamp = (meta["modified_at"], meta["size_bytes"])
    cached = _TRACE_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    summary = {field: payload.get(field) for field in _TRACE_KEEP}
    thinking = payload.get("thinking") or ""
    content = payload.get("content") or ""
    summary["thinking_chars"] = len(thinking)
    summary["content_chars"] = len(content)
    summary["trace_len"] = len(thinking) + len(content)
    _TRACE_CACHE[key] = (stamp, summary)
    return summary


def trace_map(paths, frame: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    """``{case_id: trace summary}`` for every ledger row that has a trace file."""
    if frame is None or frame.empty or "case_id" not in frame.columns:
        return {}
    traces: dict[str, dict[str, Any]] = {}
    for case_id, raw in zip(frame["case_id"].astype(str), frame.get("trace_path", pd.Series(dtype=object)).astype(str)):
        if not raw or raw in {"nan", "None"}:
            continue
        summary = trace_summary(Path(raw))
        if summary is not None:
            traces[case_id] = summary
    return traces


def _seconds_series(frame: pd.DataFrame, traces: dict[str, dict[str, Any]]) -> pd.Series:
    """Per-row case duration, from the trace when present (the ledger has none)."""
    if frame.empty:
        return pd.Series(dtype=float)
    if traces:
        return pd.Series(
            [traces.get(str(case), {}).get("seconds") for case in frame["case_id"].astype(str)],
            index=frame.index,
            dtype=float,
        )
    return pd.Series(np.nan, index=frame.index, dtype=float)


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
def analytics(paths, frame: pd.DataFrame | None) -> dict[str, Any]:
    """Funnel, vote×label, FinBERT, calibration, disagreement, latency, cohorts."""
    if frame is None or frame.empty:
        return {"available": False}

    judged = frame[judged_mask(frame)]
    traces = trace_map(paths, frame)
    votes = frame["qwen_vote"].astype(str).str.upper() if "qwen_vote" in frame.columns else pd.Series("NONE", index=frame.index)
    finbert = frame["finbert_vote"].astype(str).str.upper() if "finbert_vote" in frame.columns else pd.Series("NONE", index=frame.index)
    labels = frame["historical_label"].astype(str).str.upper() if "historical_label" in frame.columns else pd.Series("NONE", index=frame.index)
    kept = frame["kept_for_training"].fillna(False).astype(bool) if "kept_for_training" in frame.columns else pd.Series(False, index=frame.index)
    fails = frame["harness_fail"].fillna(False).astype(bool) if "harness_fail" in frame.columns else pd.Series(False, index=frame.index)
    score = _numeric(frame, "finbert_score")
    conf = _numeric(frame, "qwen_conf")

    attempted = int(len(frame))
    rated = int(frame["case_id"].astype(str).nunique()) if "case_id" in frame.columns else attempted
    funnel = [
        {"step": "attempted", "label": "Cases attempted", "n": attempted},
        {"step": "rated", "label": "Rated (unique case ids)", "n": rated},
        {"step": "judged", "label": "Judged (usable vote, no harness fail)", "n": int(len(judged))},
        {"step": "kept", "label": "Kept for training", "n": int(kept.sum())},
    ]
    judged_rate = (len(judged) / attempted) if attempted else None
    fail_rate = (float(fails.mean()) if attempted else None)
    for index, step in enumerate(funnel):
        step["share_of_attempted"] = round(step["n"] / attempted, 4) if attempted else None
        step["retention"] = None
        if index:
            previous = funnel[index - 1]["n"]
            step["retention"] = round(step["n"] / previous, 4) if previous else None

    # Calibration: confidence says 0.7 — is the vote right 70% of the time?
    calibration = {"bins": [], "ece": None, "n": 0}
    confidence_hist: list[dict] = []
    if conf.notna().any():
        edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001]
        clipped = conf.clip(0.0, 1.0).dropna()
        counted = pd.cut(clipped, bins=edges, right=False, labels=False).value_counts().sort_index()
        total_conf = int(counted.sum())
        confidence_hist = [
            {"lo": round(edges[int(index)], 2), "hi": round(min(edges[int(index) + 1], 1.0), 2), "n": int(count), "share": round(int(count) / total_conf, 4)}
            for index, count in counted.items()
        ]
    if "vote_matches_label" in frame.columns and conf.notna().any():
        table = pd.DataFrame({"conf": conf.clip(0.0, 1.0), "hit": frame["vote_matches_label"].fillna(False).astype(float)})
        table = table[table["conf"].notna()]
        if len(table):
            edges = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001]
            table["bin"] = pd.cut(table["conf"], bins=edges, right=False, labels=False)
            bins, ece = [], 0.0
            for index, chunk in table.groupby("bin"):
                if not len(chunk):
                    continue
                low, high = edges[int(index)], edges[int(index) + 1]
                accuracy = float(chunk["hit"].mean())
                confidence = float(chunk["conf"].mean())
                bins.append({
                    "lo": round(low, 2),
                    "hi": round(min(high, 1.0), 2),
                    "n": int(len(chunk)),
                    "share": round(len(chunk) / len(table), 4),
                    "accuracy": round(accuracy, 4),
                    "confidence": round(confidence, 4),
                    "gap": round(accuracy - confidence, 4),
                })
                ece += (len(chunk) / len(table)) * abs(accuracy - confidence)
            calibration = {"bins": bins, "ece": round(ece, 4), "n": int(len(table)), "mean_confidence": round(float(table["conf"].mean()), 4)}

    # Disagreement: who follows the realised label, and where do the two raters split.
    both_agree = (votes == labels) & (finbert == labels)
    qwen_only = (votes == labels) & (finbert != labels)
    finbert_only = (votes != labels) & (finbert == labels)
    neither = (votes != labels) & (finbert != labels)
    judged_index = judged.index
    disagreement = {
        "both_match": int(both_agree.reindex(judged_index).fillna(False).sum()),
        "qwen_only": int(qwen_only.reindex(judged_index).fillna(False).sum()),
        "finbert_only": int(finbert_only.reindex(judged_index).fillna(False).sum()),
        "neither": int(neither.reindex(judged_index).fillna(False).sum()),
        "qwen_matches": _rate(votes.reindex(judged_index) == labels.reindex(judged_index)),
        "finbert_matches": _rate(finbert.reindex(judged_index) == labels.reindex(judged_index)),
        "finbert_vs_qwen": _matrix(finbert.reindex(judged_index), votes.reindex(judged_index), LABELS, LABELS),
        "quoted_on": "judged rows only",
    }

    seconds = _seconds_series(frame, traces)
    outcome = pd.Series(
        np.where(fails, "harness_fail", np.where(frame.get("vote_matches_label", pd.Series(False, index=frame.index)).fillna(False).astype(bool), "match", "mismatch")),
        index=frame.index,
    )
    latency = {
        "overall": _quantiles(seconds),
        "by_outcome": [
            {"outcome": name, **_quantiles(seconds[outcome == name])} for name in ("match", "mismatch", "harness_fail")
        ],
        "source": "track_q/phase_0/traces/*.json (seconds)",
        "available": bool(seconds.notna().any()),
    }

    no_news = frame["no_news"].fillna(False).astype(bool) if "no_news" in frame.columns else pd.Series(False, index=frame.index)
    cohorts = []
    for name, mask in (("news", ~no_news), ("no_news", no_news)):
        subset = judged[judged.index.isin(frame.index[mask])]
        cohorts.append({
            "cohort": name,
            "n": int(len(subset)),
            "qwen_matches": _rate(subset["vote_matches_label"].fillna(False).astype(bool)) if "vote_matches_label" in subset.columns else None,
            "finbert_matches": _rate(subset["finbert_matches_label"].fillna(False).astype(bool)) if "finbert_matches_label" in subset.columns else None,
            "mean_confidence": round(float(_numeric(subset, "qwen_conf").dropna().mean()), 4) if len(subset) else None,
        })

    per_symbol = []
    if "symbol" in frame.columns:
        for symbol, chunk in judged.groupby(judged["symbol"].astype(str)):
            per_symbol.append({
                "symbol": symbol,
                "n": int(len(chunk)),
                "qwen_matches": _rate(chunk["vote_matches_label"].fillna(False).astype(bool)) if "vote_matches_label" in chunk.columns else None,
                "mean_confidence": round(float(_numeric(chunk, "qwen_conf").dropna().mean()), 4) if len(chunk) else None,
                "mean_finbert": round(float(_numeric(chunk, "finbert_score").dropna().mean()), 4) if len(chunk) else None,
            })
        per_symbol.sort(key=lambda row: row["n"], reverse=True)

    return {
        "available": True,
        "rows": attempted,
        "funnel": funnel,
        "judged_rate": round(judged_rate, 4) if judged_rate is not None else None,
        "harness_fail_rate": round(fail_rate, 4) if fail_rate is not None else None,
        "votes_by_label": _matrix(labels.reindex(judged_index), votes.reindex(judged_index), LABELS, LABELS),
        "finbert_by_label": _matrix(labels.reindex(judged_index), finbert.reindex(judged_index), LABELS, LABELS),
        "calibration": calibration,
        "confidence_hist": confidence_hist,
        "disagreement": disagreement,
        "latency": latency,
        "cohorts": cohorts,
        "per_symbol": per_symbol,
        "finbert_score": _quantiles(score),
        "confidence": _quantiles(conf),
        "trace_note": "Latency and trace lengths come from the trace files; the ledger itself stores no duration.",
    }


# --------------------------------------------------------------------------- #
# Timeline (resource series, heartbeat history, ledger progress)
# --------------------------------------------------------------------------- #
def _window_seconds(window: str) -> float | None:
    match = re.fullmatch(r"\s*(\d+)\s*([mh])\s*", str(window or ""))
    if not match:
        return None
    amount, unit = int(match.group(1)), match.group(2)
    return float(amount * (60 if unit == "m" else 3600))


def _resource_series(paths, window: str, step: int) -> tuple[dict[str, Any], list[str]]:
    """GPU series from the superset file when it exists, else temperature only."""
    warnings: list[str] = []
    p0 = p0_from(paths.root)
    flat_path = p0["logs"] / "q0_resource_5s.csv"
    legacy_path = p0["logs"] / "gpu_5s.csv"
    path = flat_path if flat_path.exists() else legacy_path
    meta = file_meta(path)
    if not meta["available"]:
        warnings.append("Neither track_q/phase_0/logs/q0_resource_5s.csv nor track_q/phase_0/logs/gpu_5s.csv exists; no GPU series to draw.")
        return {"available": False, "fields": [], "series": [], "source": None}, warnings

    try:
        frame = pd.read_csv(path)
    except Exception:
        warnings.append(f"{path.name} is unreadable; no GPU series to draw.")
        return {"available": False, "fields": [], "series": [], "source": path.name}, warnings

    if path is legacy_path:
        warnings.append(
            "track_q/phase_0/logs/q0_resource_5s.csv is missing (written by q0_d1) — only temperature is available. "
            "Utilisation, power, fan and clocks are absent, not zero."
        )
    fields = [name for name in RESOURCE_FIELDS if name in frame.columns and pd.to_numeric(frame[name], errors="coerce").notna().any()]
    if not fields:
        warnings.append(f"{path.name} has no numeric resource columns.")
        return {"available": False, "fields": [], "series": [], "source": path.name}, warnings

    frame["_at"] = pd.to_datetime(frame.get("timestamp"), errors="coerce")
    frame = frame.dropna(subset=["_at"]).sort_values("_at")
    ceiling = _window_seconds(window)
    if ceiling is not None and len(frame):
        cutoff = frame["_at"].max() - pd.Timedelta(seconds=ceiling)
        frame = frame[frame["_at"] >= cutoff]
    stride = max(1, int(round(float(step or SAMPLE_SECONDS) / SAMPLE_SECONDS)))
    # "all" plus 5s samples is ~12k rows for a long run: keep the shape, not the
    # raw sample count, so one poll does not ship a megabyte.
    if len(frame) > MAX_SERIES_POINTS:
        stride = max(stride, -(-len(frame) // MAX_SERIES_POINTS))
    if stride > 1:
        frame = frame.iloc[::stride]

    series = []
    for _, row in frame.iterrows():
        entry = {"at": row["_at"].isoformat(timespec="seconds"), "case_id": str(row.get("case_id") or "")}
        for name in fields:
            value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
            entry[name] = None if pd.isna(value) else float(value)
        series.append(entry)

    return {
        "available": bool(series),
        "fields": fields,
        "series": series,
        "source": f"logs/{path.name}",
        "stride_seconds": stride * SAMPLE_SECONDS,
        "timezone": "naive-local (written by q0_d1)",
        "size_bytes": meta["size_bytes"],
        "supplementary": flat_path.exists(),
    }, warnings


def _history_rows(path: Path, limit: int = 4000) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    if not file_meta(path)["available"]:
        warnings.append("track_q/phase_0/logs/q0_history.jsonl is missing — written by q0_d1 once per rest/heartbeat (30s throttle).")
        return [], warnings
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        warnings.append("track_q/phase_0/logs/q0_history.jsonl is unreadable.")
        return [], warnings
    return rows[-limit:], warnings


def _progress_rows(frame: pd.DataFrame | None, seconds: pd.Series | None = None) -> list[dict]:
    """Cumulative rated / judged / kept, plus rolling rates, in ledger order.

    The ledger is append-only, so row order *is* time order. It carries no
    timestamp of its own, which is why the x axis is a case index and not a clock.
    ``seconds`` is the per-row case duration taken from the traces; when it is
    absent the duration fields stay ``None`` instead of being filled in.
    """
    if frame is None or frame.empty:
        return []
    judged = judged_mask(frame).astype(int)
    kept = frame["kept_for_training"].fillna(False).astype(bool).astype(int) if "kept_for_training" in frame.columns else pd.Series(0, index=frame.index)
    fail = frame["harness_fail"].fillna(False).astype(bool).astype(int) if "harness_fail" in frame.columns else pd.Series(0, index=frame.index)
    votes = frame["qwen_vote"].astype(str).str.upper() if "qwen_vote" in frame.columns else pd.Series("", index=frame.index)
    hit = frame["vote_matches_label"].fillna(False).astype(bool).astype(int) if "vote_matches_label" in frame.columns else pd.Series(0, index=frame.index)
    duration = pd.to_numeric(seconds, errors="coerce") if seconds is not None else pd.Series(np.nan, index=frame.index)

    rows = []
    for position in range(len(frame)):
        window = slice(max(0, position - ROLLING_CASES + 1), position + 1)
        sampled = position + 1
        rolling_hit = hit.iloc[window]
        agreement = float(rolling_hit.mean())
        standard_error = float(np.sqrt(max(agreement * (1 - agreement), 0.0) / max(len(rolling_hit), 1)))
        rolling_seconds = duration.iloc[window].dropna()
        rows.append({
            "seq": sampled,
            "rated": sampled,
            "judged": int(judged.iloc[: sampled].sum()),
            "kept": int(kept.iloc[: sampled].sum()),
            "harness_fail_rate": round(float(fail.iloc[window].mean()), 4),
            "qwen_matches": round(agreement, 4),
            "agreement_lo": round(max(0.0, agreement - 2 * standard_error), 4),
            "agreement_hi": round(min(1.0, agreement + 2 * standard_error), 4),
            "vote_up": round(float((votes.iloc[window] == "UP").mean()), 4),
            "vote_down": round(float((votes.iloc[window] == "DOWN").mean()), 4),
            "vote_flat": round(float((votes.iloc[window] == "FLAT").mean()), 4),
            "case_seconds_p50": round(float(rolling_seconds.quantile(0.50)), 1) if len(rolling_seconds) else None,
            "case_seconds_p95": round(float(rolling_seconds.quantile(0.95)), 1) if len(rolling_seconds) else None,
        })
    return rows


def _thin(rows: list[dict], limit: int) -> list[dict]:
    """Stride a long series down to ``limit`` points, always keeping the last one.

    The timeline is polled every few seconds during a run; sending 716 cumulative
    points on each poll costs more than it shows.
    """
    if len(rows) <= limit or limit <= 1:
        return rows
    stride = max(1, len(rows) // (limit - 1))
    thinned = rows[::stride]
    if thinned[-1] is not rows[-1]:
        thinned.append(rows[-1])
    return thinned


def timeline(paths, frame: pd.DataFrame | None, window: str = "2h", step: int = 5) -> dict[str, Any]:
    """Everything the Live tab plots, from the files the run actually wrote."""
    p0 = p0_from(paths.root)
    resource, warnings = _resource_series(paths, window, step)
    history, history_warnings = _history_rows(p0["logs"] / "q0_history.jsonl")
    warnings.extend(history_warnings)
    hb_path = p0["logs"] / "q0_heartbeat.json"
    heartbeat_meta = file_meta(hb_path)
    heartbeat = {}
    if heartbeat_meta["available"]:
        try:
            heartbeat = json.loads(hb_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            warnings.append("track_q/phase_0/logs/q0_heartbeat.json is unreadable.")
    age = age_seconds(heartbeat_meta["modified_at"])
    running = bool(heartbeat) and age is not None and age <= LIVE_SECONDS and str(heartbeat.get("phase") or "") not in {"complete", "done"}

    progress = _progress_rows(frame, _seconds_series(frame, trace_map(paths, frame)) if frame is not None else None)
    psi_path = p0["state"] / "psi_history.jsonl"
    psi_rows, psi_warnings = ([], [])
    if file_meta(psi_path)["available"]:
        try:
            psi_rows = [json.loads(line) for line in psi_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError):
            psi_warnings.append("artifacts/state/psi_history.jsonl is unreadable.")
    else:
        psi_warnings.append("artifacts/state/psi_history.jsonl is missing; drift has no trend yet (run: monitor).")
    warnings.extend(psi_warnings)

    return {
        "window": window,
        "step": int(step or SAMPLE_SECONDS),
        "resource": resource,
        "history": {"available": bool(history), "rows": _thin(history, 400), "source": "track_q/phase_0/logs/q0_history.jsonl"},
        "progress": {"available": bool(progress), "rows": _thin(progress, 300), "window": ROLLING_CASES, "source": "track_q/phase_0/state/qwen_rating_ledger.parquet"},
        "psi": {"available": bool(psi_rows), "rows": psi_rows[-60:], "source": "track_q/phase_0/state/psi_history.jsonl"},
        "live": {
            "running": running,
            "phase": heartbeat.get("phase"),
            "current_case": heartbeat.get("current_case"),
            "heartbeat_age_seconds": age,
            "limit_seconds": LIVE_SECONDS,
            "updated_at": heartbeat.get("updated_at"),
        },
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
def case_rows(
    frame: pd.DataFrame | None,
    *,
    limit: int = 50,
    offset: int = 0,
    symbol: str = "",
    vote: str = "all",
    outcome: str = "all",
) -> dict[str, Any]:
    """Paginated ledger rows, ordered newest decision date first."""
    if frame is None or frame.empty:
        return {"total": 0, "offset": offset, "limit": limit, "rows": []}

    work = frame.copy()
    if symbol:
        work = work[work["symbol"].astype(str).str.upper().str.contains(symbol.upper(), regex=False, na=False)]
    if vote and vote != "all":
        work = work[work["qwen_vote"].astype(str).str.upper() == vote.upper()]
    if outcome and outcome != "all" and "vote_matches_label" in work.columns:
        # A harness failure carries no vote, so it has no outcome to compare: it is
        # excluded from both "matches" and "misses" rather than counted as a miss.
        verdict = work["vote_matches_label"]
        work = work[verdict.notna() & (verdict.astype(bool) == (outcome == "match"))]
    if "decision_date" in work.columns:
        work = work.sort_values("decision_date", ascending=False)

    total = int(len(work))
    page = work.iloc[offset: offset + limit]
    rows = []
    for _, row in page.iterrows():
        rows.append({
            "case_id": str(row.get("case_id")),
            "symbol": str(row.get("symbol")),
            "decision_date": str(row.get("decision_date")),
            "pool": str(row.get("pool")),
            "model_stage": str(row.get("model_stage")),
            "prompt_version": str(row.get("prompt_version")),
            "vote": None if pd.isna(row.get("qwen_vote")) else str(row.get("qwen_vote")),
            "confidence": None if pd.isna(row.get("qwen_conf")) else round(float(row.get("qwen_conf")), 4),
            "finbert_vote": None if pd.isna(row.get("finbert_vote")) else str(row.get("finbert_vote")),
            "finbert_score": None if pd.isna(row.get("finbert_score")) else round(float(row.get("finbert_score")), 4),
            "label": None if pd.isna(row.get("historical_label")) else str(row.get("historical_label")),
            "matches_label": None if pd.isna(row.get("vote_matches_label")) else bool(row.get("vote_matches_label")),
            "kept_for_training": bool(row.get("kept_for_training")) if not pd.isna(row.get("kept_for_training")) else None,
            "drop_reason": None if pd.isna(row.get("drop_reason")) else str(row.get("drop_reason")),
            "harness_fail": bool(row.get("harness_fail")) if not pd.isna(row.get("harness_fail")) else None,
            "thermal_state": None if pd.isna(row.get("thermal_state")) else str(row.get("thermal_state")),
            "trace_available": bool(str(row.get("trace_path") or "")) and Path(str(row.get("trace_path"))).exists(),
        })
    return {"total": total, "offset": offset, "limit": limit, "rows": rows}


def safe_case_id(case_id: str) -> bool:
    """Reject anything that could escape the phase-0 trace directory."""
    if not case_id or len(case_id) > 200:
        return False
    return not ("/" in case_id or "\\" in case_id or ".." in case_id or ":" in case_id)


def case_detail(paths, frame: pd.DataFrame | None, case_id: str) -> dict[str, Any] | None:
    """One ledger row plus its full trace. ``None`` when the case is unknown."""
    if not safe_case_id(case_id):
        return None
    row = None
    trace_path: Path | None = None
    if frame is not None and not frame.empty and "case_id" in frame.columns:
        match = frame[frame["case_id"].astype(str) == case_id]
        if not match.empty:
            record = match.iloc[0]
            raw = record.get("trace_path")
            row = {key: (None if pd.isna(record.get(key)) else record.get(key)) for key in record.index}
            if isinstance(raw, str) and raw:
                trace_path = Path(raw)
    traces_dir = (p0_from(paths.root)["traces"]).resolve()
    if trace_path is None:
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", case_id)
        candidate = traces_dir / f"{slug}.json"
        trace_path = candidate if candidate.exists() else None
    elif not str(trace_path.resolve()).startswith(str(traces_dir)):
        return None  # a ledger row pointing outside the trace directory is not served

    trace = None
    if trace_path is not None and trace_path.exists():
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            trace = None

    if row is None and trace is None:
        return None
    return {
        "case_id": case_id,
        "ledger": row,
        "trace": trace,
        "trace_available": trace is not None,
        "trace_chars": (len((trace or {}).get("thinking") or "") + len((trace or {}).get("content") or "")) if trace else None,
    }
