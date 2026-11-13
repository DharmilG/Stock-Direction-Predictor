"""Read-only model behaviour endpoints.

Everything here is a view over artifacts written by ``model-report`` (and, for
serving explanations, by ``explain``). Nothing trains, retrains, scores, or
changes a threshold: the page exists to explain the model that already exists.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from .dashboard_support import (
    age_seconds,
    file_meta,
    frame_file,
    json_file,
    json_safe,
    now_ist,
    to_iso,
)
from .models import DEFAULT_MODEL, model_context, resolve

router = APIRouter(prefix="/dashboard/model", tags=["dashboard-model"])
explain_router = APIRouter(prefix="/dashboard", tags=["dashboard-model"])

REPORT_STALE_HOURS = 24 * 14
LIVE_HEARTBEAT_SECONDS = 120
CASE_PAGE_LIMIT = 500


def _envelope(data: Any, artifacts: list[str], *, as_of: Any = None, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "data": json_safe(data),
        "as_of": to_iso(as_of),
        "generated_at": now_ist().isoformat(),
        "source": {"kind": "artifact", "artifacts": artifacts},
        "warnings": warnings or [],
    }


def _stage_hint(model: str, stage: str) -> str:
    """The command that would actually rebuild this model's artefact.

    v2 lives under its own runner; pointing a v2 reader at
    ``python -m stockml.main model-report`` would silently rebuild the
    incumbent instead, so the hint has to follow the selected model.
    """
    if resolve(model).id == "v1":
        return f"python -m stockml.main {stage}"
    return f"python -m stockml.v2.stages {stage}"


def _missing(what: str, artifacts: list[str]) -> dict[str, Any]:
    return _envelope(None, artifacts, warnings=[f"{what} is not available yet."])


def _tail_jsonl(path, limit: int) -> list[dict]:
    """Read the last ``limit`` JSON lines; malformed lines are skipped, not fatal."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# --------------------------------------------------------------------------- #
# Behaviour report
# --------------------------------------------------------------------------- #
@router.get("/report")
def model_report(model: str = Query(default=DEFAULT_MODEL)) -> dict[str, Any]:
    settings, paths = model_context(model)
    path = paths.results / "model_report.json"
    artifacts = [
        "results/model_report.json",
        "results/predictions_splits.parquet",
        "results/shap_by_class.csv",
        "artifacts/model_metadata.json",
    ]
    report = json_file(path)
    if not report:
        return _missing(
            f"results/model_report.json (run: {_stage_hint(model, 'report')})",
            artifacts,
        )

    meta = file_meta(path)
    age = age_seconds(meta["modified_at"])
    warnings: list[str] = []
    if age is not None and age > REPORT_STALE_HOURS * 3600:
        warnings.append(
            f"The behaviour report is {age // 86400} day(s) old; re-run "
            f"`{_stage_hint(model, 'report')}` to refresh it."
        )
    consistency = report.get("consistency") or {}
    alignment = consistency.get("feature_alignment") or {}
    if alignment.get("pinned_to_metadata") and alignment.get("store_features", 0) > alignment.get("scored_features", 0):
        warnings.append(
            f"The feature store has {alignment['store_features']} columns; the booster was trained on "
            f"{alignment['scored_features']}. The report is pinned to the trained set."
        )
    if consistency.get("same_window") is False:
        warnings.append("The report covers a different test window than results/metrics.json; both are shown.")
    return _envelope(report, artifacts, as_of=meta["modified_at"], warnings=warnings)


# --------------------------------------------------------------------------- #
# Live training telemetry
# --------------------------------------------------------------------------- #
@router.get("/live")
def model_live(model: str = Query(default=DEFAULT_MODEL)) -> dict[str, Any]:
    settings, paths = model_context(model)
    heartbeat = json_file(paths.logs / "train_heartbeat.json")
    curve = _tail_jsonl(paths.checkpoints / "train_curve.jsonl", 4000)
    folds = _tail_jsonl(paths.checkpoints / "trials" / "fold_diagnostics.jsonl", 400)
    checkpoint_meta = json_file(paths.checkpoints / "lightgbm_final_checkpoint.txt.json")
    heartbeat_age = age_seconds(heartbeat.get("updated_at")) if heartbeat else None
    # A "done" heartbeat is a finished run, not a live one, however recent it is.
    running = str(heartbeat.get("status") or "") in {"optuna", "final_fit", "calibration"}
    training = bool(heartbeat) and running and heartbeat_age is not None and heartbeat_age <= LIVE_HEARTBEAT_SECONDS
    if curve:
        # One line, one run: a curve that mixed two signatures would draw a
        # step nobody trained.
        latest_run = curve[-1].get("run")
        curve = [point for point in curve if point.get("run") == latest_run][-400:]

    warnings: list[str] = []
    if not heartbeat:
        warnings.append(
            "logs/train_heartbeat.json does not exist: no training run has been instrumented yet. "
            "Start one with `python -m stockml.main train`."
        )
    elif not training:
        warnings.append(
            f"The last training heartbeat is {heartbeat_age}s old with status "
            f"{heartbeat.get('status')!r}; the run has finished or stopped."
        )
    if not curve:
        warnings.append("checkpoints/train_curve.jsonl has no recorded iterations yet.")
    if not folds:
        warnings.append("checkpoints/trials/fold_diagnostics.jsonl has no folds yet.")

    return _envelope(
        {
            "training": training,
            "heartbeat": heartbeat or None,
            "heartbeat_age_seconds": heartbeat_age,
            "curve": curve,
            "folds": folds,
            "checkpoint_rounds": (checkpoint_meta or {}).get("completed_rounds"),
            "limits": {"heartbeat_seconds": LIVE_HEARTBEAT_SECONDS},
        },
        [
            "logs/train_heartbeat.json",
            "checkpoints/train_curve.jsonl",
            "checkpoints/trials/fold_diagnostics.jsonl",
        ],
        as_of=(heartbeat or {}).get("updated_at"),
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Case browser
# --------------------------------------------------------------------------- #
@router.get("/cases")
def model_cases(
    split: str = Query("test", pattern="^(train|calibration|test|all)$"),
    predicted: str | None = Query(None, pattern="^(DOWN|FLAT|UP|all)$"),
    actual: str | None = Query(None, pattern="^(DOWN|FLAT|UP|all)$"),
    symbol: str | None = None,
    correct: str = Query("all", pattern="^(all|right|wrong)$"),
    limit: int = Query(50, ge=1, le=CASE_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
    model: str = Query(default=DEFAULT_MODEL),
) -> dict[str, Any]:
    settings, paths = model_context(model)
    artifacts = ["results/predictions_splits.parquet"]
    frame = frame_file(paths.results / "predictions_splits.parquet")
    if frame is None or frame.empty:
        return _missing(
            f"results/predictions_splits.parquet (run: {_stage_hint(model, 'report')})",
            artifacts,
        )

    names = {0: "DOWN", 1: "FLAT", 2: "UP"}
    work = frame
    if split != "all":
        work = work[work["split"] == split]
    if predicted and predicted != "all":
        work = work[work["predicted_label"] == {v: k for k, v in names.items()}[predicted]]
    if actual and actual != "all":
        work = work[work["actual_label"] == {v: k for k, v in names.items()}[actual]]
    if correct == "right":
        work = work[work["correct"] == 1]
    elif correct == "wrong":
        work = work[work["correct"] == 0]
    if symbol:
        work = work[work["symbol"].astype(str).str.upper() == symbol.upper()]

    total = int(len(work))
    if not total:
        warnings = ["No rows match those filters."]
        return _envelope({"total": 0, "offset": offset, "limit": limit, "rows": []}, artifacts, warnings=warnings)

    work = work.sort_values(["date", "symbol"], ascending=[False, True]).iloc[offset: offset + limit]
    rows = []
    for _, row in work.iterrows():
        rows.append({
            "split": str(row["split"]),
            "symbol": str(row["symbol"]),
            "date": str(row["date"]),
            "actual": names.get(int(row["actual_label"]), "?"),
            "predicted": names.get(int(row["predicted_label"]), "?"),
            "p_down": round(float(row["p_down"]), 4),
            "p_flat": round(float(row["p_flat"]), 4),
            "p_up": round(float(row["p_up"]), 4),
            "max_p": round(float(row["max_p"]), 4),
            "margin": round(float(row["margin"]), 4),
            "correct": bool(row["correct"]),
            "realized_return": None if pd.isna(row.get("realized_return")) else round(float(row["realized_return"]), 6),
        })
    return _envelope(
        {"total": total, "offset": offset, "limit": limit, "rows": rows},
        artifacts,
        as_of=file_meta(paths.results / "predictions_splits.parquet")["modified_at"],
    )


# --------------------------------------------------------------------------- #
# Per-row waterfall
# --------------------------------------------------------------------------- #
@explain_router.get("/explain/{symbol}")
def explain_symbol(
    symbol: str,
    date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    split: str | None = Query(None, pattern="^(serving|train|calibration|test)$"),
    model: str = Query(default=DEFAULT_MODEL),
) -> dict[str, Any]:
    settings, paths = model_context(model)
    if symbol not in settings.symbols:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} is not in the configured universe")

    # No split / no date means "today's serving row"; asking for a split reads the
    # sampled split file written by `model-report`.
    warnings: list[str] = []
    use_serving = split in (None, "serving") and date is None
    split = "test" if split in (None, "serving") else split
    path = paths.results / ("explanations_serving.parquet" if use_serving else "explanations_splits.parquet")
    fallback = paths.results / "explanations_splits.parquet"
    artifacts = ["results/explanations_serving.parquet", "results/explanations_splits.parquet"]

    frame = frame_file(path)
    if frame is None or frame.empty:
        if use_serving:
            warnings.append(
                "results/explanations_serving.parquet is missing (run: python -m stockml.main explain); "
                "showing a sampled split row instead."
            )
        path = fallback
        frame = frame_file(fallback)
        use_serving = False
    if frame is None or frame.empty:
        return _missing(
            "results/explanations_*.parquet (run: python -m stockml.main model-report, or explain)",
            artifacts,
        )

    work = frame[frame["symbol"].astype(str) == symbol]
    if not use_serving:
        work = work[work["split"] == split]
        if date:
            work = work[work["date"].astype(str) == date]
    if work.empty:
        available = sorted(frame["symbol"].astype(str).unique().tolist())[:8]
        raise HTTPException(
            status_code=404,
            detail=f"No explanation for {symbol} in that slice. Run `python -m stockml.main explain --symbol {symbol}`. "
                   f"Symbols present: {', '.join(available)}",
        )

    latest = sorted(work["date"].astype(str).unique())[-1]
    work = work[work["date"].astype(str) == latest]
    classes: dict[str, Any] = {}
    raw_scores: dict[str, float] = {}
    for name, chunk in work.groupby("class", sort=False):
        steps = []
        for _, row in chunk.sort_values("rank").iterrows():
            steps.append({
                "rank": int(row["rank"]),
                "feature": str(row["feature"]),
                "feature_value": None if pd.isna(row.get("feature_value")) else float(row["feature_value"]),
                "contribution": round(float(row["contribution"]), 6),
            })
        base = float(chunk["base_value"].iloc[0]) if pd.notna(chunk["base_value"].iloc[0]) else None
        closing = base + sum(s["contribution"] for s in steps) if base is not None else None
        # Tree SHAP is additive in raw (log-odds) space, so base + contributions is
        # exactly the class raw score the booster produced.
        if closing is not None:
            raw_scores[str(name)] = closing
        classes[str(name)] = {"base_value": base, "steps": steps, "raw_score": None if closing is None else round(closing, 6)}

    head = work.iloc[0]
    probabilities = {}
    for name, column in (("DOWN", "p_down"), ("FLAT", "p_flat"), ("UP", "p_up")):
        value = head.get(column)
        probabilities[name] = None if value is None or pd.isna(value) else round(float(value), 4)
    ordered = sorted(((v, k) for k, v in probabilities.items() if v is not None), reverse=True)
    decision_margin = round(ordered[0][0] - ordered[1][0], 4) if len(ordered) >= 2 else None

    # "Distance to flip": how far the runner-up must travel in raw-score units, and
    # how many multiples of its strongest contribution that represents. It is an
    # estimate of headroom, not a counterfactual — no feature is actually re-scored.
    flip = None
    if len(raw_scores) >= 2 and ordered:
        winner, runner_up = ordered[0][1], ordered[1][1]
        raw_gap = raw_scores.get(runner_up, 0.0) - raw_scores.get(winner, 0.0)
        lever = max(classes[runner_up]["steps"], key=lambda s: abs(s["contribution"]), default=None)
        flip = {
            "from": winner,
            "to": runner_up,
            "probability_gap": decision_margin,
            "raw_gap": round(raw_gap, 4),
            "lever": None if not lever or not lever["contribution"] else {
                "feature": lever["feature"],
                "class": runner_up,
                "contribution": lever["contribution"],
                "value": lever["feature_value"],
                "multiple_needed": round(abs(raw_gap / lever["contribution"]), 2),
            },
            "note": "Raw-score headroom; contributions are additive, the features behind them are not linear.",
        }

    meta = file_meta(path)
    return _envelope(
        {
            "source": "serving" if use_serving else "split",
            "symbol": symbol,
            "date": str(latest),
            "split": "serving" if use_serving else split,
            "prediction": {0: "DOWN", 1: "FLAT", 2: "UP"}.get(int(head["predicted_label"]), None),
            "actual": None if int(head.get("actual_label", -1)) < 0 else {0: "DOWN", 1: "FLAT", 2: "UP"}.get(int(head["actual_label"])),
            "probabilities": probabilities,
            "base_value": {name: entry["base_value"] for name, entry in classes.items()},
            "margin": decision_margin,
            "flip": flip,
            "method": "tree_shap",
            "classes": classes,
            "stale": (age_seconds(meta["modified_at"]) or 0) > 86400,
            "age_seconds": age_seconds(meta["modified_at"]),
        },
        artifacts,
        as_of=meta["modified_at"],
        warnings=warnings,
    )
