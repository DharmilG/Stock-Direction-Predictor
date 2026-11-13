"""Read-only endpoints for the v2 experiment.

Every number here comes from ``v2/artifacts/v2_report.json`` and
``v2/results/v2_predictions.parquet``, written by ``python -m stockml.v2.train``.
Nothing on this router trains, scores, or touches the incumbent: it is a view.
Each response carries ``METRIC_CONTRACT`` so the page cannot invent a
definition for a number it did not compute.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import APIRouter, Query

from ..v2.workspace import V2_ROOT
from .dashboard_support import file_meta, frame_file, json_file, json_safe, now_ist, to_iso

router = APIRouter(prefix="/dashboard/v2", tags=["dashboard-v2"])

REPORT_STALE_HOURS = 24 * 30
CASE_PAGE_LIMIT = 500
CLASS_NAMES = {0: "DOWN", 1: "FLAT", 2: "UP", -1: "ABSTAIN"}

# Denominator + the single decision each number may inform. The UI prints the
# denominator next to the value; a metric without one is not comparable.
METRIC_CONTRACT: dict[str, dict[str, str]] = {
    "coverage": {
        "denominator": "direction calls / all test rows",
        "decision": "How much of the book the rule is willing to trade.",
    },
    "accuracy_called": {
        "denominator": "correct calls / direction calls only",
        "decision": "Hit rate when the rule commits. Silence is excluded.",
    },
    "accuracy_all_rows": {
        "denominator": "correct / all test rows, FLAT and ABSTAIN counted as misses",
        "decision": "Headline directional accuracy with no free abstentions.",
    },
    "accuracy_movers": {
        "denominator": "correct / rows whose true class is DOWN or UP",
        "decision": "Directional accuracy on the rows that actually moved.",
    },
    "accuracy_all_penalised": {
        "denominator": "correct / all rows, every ABSTAIN counted wrong",
        "decision": "Whether abstaining still leaves the rule ahead of always-calling.",
    },
    "balanced_called": {
        "denominator": "mean of DOWN and UP recall / direction calls",
        "decision": "Whether the rule stopped leaning on one direction.",
    },
    "mcc_called": {
        "denominator": "Matthews correlation / direction calls",
        "decision": "Directional skill corrected for class mix.",
    },
    "three_class_accuracy": {
        "denominator": "exact class match / all test rows",
        "decision": "The label the model was trained on, scored as-is.",
    },
    "three_class_mcc": {
        "denominator": "Matthews correlation over all three classes / all rows",
        "decision": "3-class skill corrected for class mix.",
    },
    "log_loss": {
        "denominator": "mean -log p(true class) / all test rows",
        "decision": "Whether the probabilities improved, independent of the rule.",
    },
    "calibration_error": {
        "denominator": "confidence-binned ECE / all test rows",
        "decision": "Whether stated confidence matches observed frequency.",
    },
    "up_recall": {
        "denominator": "UP calls on true-UP rows / true-UP rows",
        "decision": "Whether UP is reachable at all.",
    },
    "flat_auc": {
        "denominator": "ranking of true-FLAT rows against the rest / all rows",
        "decision": "Whether FLAT is rankable even if it is not called.",
    },
    "mean_delta": {
        "denominator": "paired per-date mean of v2 minus incumbent / test dates",
        "decision": "Effect size on the independent unit (a date, not a row).",
    },
    "t": {
        "denominator": "date-clustered paired bootstrap of the delta",
        "decision": "Whether the gain survives correlation within a date.",
    },
}


def _paths():
    return V2_ROOT / "artifacts" / "v2_report.json", V2_ROOT / "results" / "v2_predictions.parquet"


def _envelope(data: Any, artifacts: list[str], *, as_of: Any = None, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "data": json_safe(data),
        "as_of": to_iso(as_of),
        "generated_at": now_ist().isoformat(),
        "source": {"kind": "artifact", "artifacts": artifacts, "contract": METRIC_CONTRACT},
        "warnings": warnings or [],
    }


def _report_or(what: str) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Load the report once; return it with its artifact list and its file meta."""
    report_path, predictions_path = _paths()
    artifacts = [
        "v2/artifacts/v2_report.json",
        "v2/results/v2_predictions.parquet",
        "v2/artifacts/v2_booster.txt",
        "v2/artifacts/v2_calibration.json",
    ]
    report = json_file(report_path)
    if not report:
        return {}, artifacts, {"missing": f"{what}: run `.venv\\Scripts\\python.exe -m stockml.v2.train`"}
    return report, artifacts, file_meta(report_path)


def _stale(meta: dict[str, Any]) -> list[str]:
    modified = meta.get("modified_at")
    if not modified:
        return []
    moment = datetime.fromisoformat(str(modified))
    if moment.tzinfo is None:
        return []
    age = now_ist() - moment
    hours = age.total_seconds() / 3600.0
    if hours <= REPORT_STALE_HOURS:
        return []
    return [f"The v2 report is {int(hours // 24)} day(s) old; re-run `python -m stockml.v2.train`."]


@router.get("/summary")
def v2_summary() -> dict[str, Any]:
    report, artifacts, meta = _report_or("v2 summary")
    if not report:
        return _envelope(None, artifacts, warnings=[meta["missing"]])

    warnings = _stale(meta)
    comparison = report.get("comparison") or {}
    if not comparison.get("promote"):
        warnings.append("v2 is not a candidate for promotion: see comparison.gates for the failing conditions.")
    variants = report.get("variants") or []
    if not any(variant.get("feasible") for variant in variants):
        warnings.append("No variant produced a feasible decision policy; the matrix is diagnostic only.")
    selection = report.get("selection") or {}
    if selection.get("was_tie"):
        warnings.append(
            "The selection score did not separate the top designs: "
            f"{', '.join(selection.get('contenders') or [])} were within {selection.get('tie_epsilon')} of each other "
            "and the tie was broken on calibration log loss."
        )

    data = {
        "generated_at": report.get("generated_at"),
        "protocol": report.get("protocol"),
        "population": report.get("population"),
        "incumbent": report.get("incumbent"),
        "selected_variant": report.get("selected_variant"),
        "selection": report.get("selection"),
        "variant_count": len(variants),
        "feasible_variants": [variant["id"] for variant in variants if variant.get("feasible")],
        "comparison": comparison,
        "metrics": report.get("metrics"),
    }
    return _envelope(data, artifacts, as_of=meta.get("modified_at"), warnings=warnings)


@router.get("/experiments")
def v2_experiments() -> dict[str, Any]:
    report, artifacts, meta = _report_or("v2 experiments")
    if not report:
        return _envelope(None, artifacts, warnings=[meta["missing"]])

    variants = []
    for variant in report.get("variants") or []:
        variants.append(
            {
                "id": variant.get("id"),
                "row_filter": variant.get("row_filter"),
                "class_weighting": variant.get("class_weighting"),
                "prior_adjustment": variant.get("prior_adjustment"),
                "rows_train": variant.get("rows_train"),
                "elapsed_s": variant.get("elapsed_s"),
                "feasible": variant.get("feasible"),
                "infeasible_reason": variant.get("infeasible_reason"),
                "selected_policy": variant.get("selected_policy"),
                "grid_size": variant.get("grid_size"),
                "feasible_policies": variant.get("feasible_policies"),
                "selection_score": variant.get("selection_score"),
                "coverage": variant.get("coverage"),
                "accuracy_called": variant.get("accuracy_called"),
                "accuracy_all_rows": variant.get("accuracy_all_rows"),
                "balanced_called": variant.get("balanced_called"),
                "log_loss": variant.get("log_loss"),
                "calibration_error": variant.get("calibration_error"),
                "up_recall": variant.get("up_recall"),
                # `mix` is what the rule actually emitted, abstentions included.
                # Without it the page would have to fall back to the incumbent's
                # distribution and would misreport v2's behaviour.
                "mix": variant.get("mix"),
                "branch_share": variant.get("branch_share"),
            }
        )
    data = {
        "variants": variants,
        "selected_variant": report.get("selected_variant"),
        "grid": {"size": (variants[0] or {}).get("grid_size"), "axes": ["tau_flat", "delta", "min_conf"]},
        "recall_floor": {"floor": 0.15, "applies_when_class_share": 0.10},
    }
    return _envelope(data, artifacts, as_of=meta.get("modified_at"), warnings=_stale(meta))


@router.get("/diagnostics")
def v2_diagnostics() -> dict[str, Any]:
    report, artifacts, meta = _report_or("v2 diagnostics")
    if not report:
        return _envelope(None, artifacts, warnings=[meta["missing"]])

    warnings = _stale(meta)
    if not report.get("folds"):
        warnings.append("Walk-forward fold bars were skipped in this run (`--no-folds`).")
    data = {
        "diagnostics": report.get("diagnostics") or {},
        "folds": report.get("folds") or [],
        "folds_note": report.get("folds_note"),
        "selected_variant": report.get("selected_variant"),
        "population": report.get("population"),
    }
    return _envelope(data, artifacts, as_of=meta.get("modified_at"), warnings=warnings)


@router.get("/reasons")
def v2_reasons() -> dict[str, Any]:
    report, artifacts, meta = _report_or("v2 reasons")
    if not report:
        return _envelope(None, artifacts, warnings=[meta["missing"]])

    variants = report.get("variants") or []
    data = {
        "reasons": report.get("reasons") or {},
        "branches": [
            {
                "id": "flat_rule",
                "meaning": "p_flat crossed tau_flat, so the row was labelled FLAT rather than a direction.",
            },
            {
                "id": "low_confidence",
                "meaning": "No class reached min_conf, so the rule abstained.",
            },
            {
                "id": "margin",
                "meaning": "UP and DOWN were within delta of each other, so the rule abstained instead of using a tie-break.",
            },
            {
                "id": "direction",
                "meaning": "A direction was called and the larger of UP/DOWN was emitted.",
            },
        ],
        "variant_shares": [
            {"id": variant.get("id"), "branch_share": variant.get("branch_share"), "mix": variant.get("mix")}
            for variant in variants
        ],
    }
    return _envelope(data, artifacts, as_of=meta.get("modified_at"), warnings=_stale(meta))


def _case_frame() -> pd.DataFrame | None:
    _, predictions_path = _paths()
    frame = frame_file(predictions_path)
    if frame is None or frame.empty:
        return None
    for column in ("pred_inc", "pred_v2", "y_true"):
        if column in frame.columns:
            frame[f"{column}_name"] = frame[column].map(CLASS_NAMES)
    return frame


@router.get("/cases")
def v2_cases(
    symbol: str = Query("", description="Exact NSE symbol, empty for all"),
    predicted: str = Query("all", description="DOWN | FLAT | UP | ABSTAIN | all, on the v2 rule"),
    actual: str = Query("all", description="DOWN | FLAT | UP | all"),
    correct: str = Query("all", description="all | v2 | incumbent | disagree"),
    limit: int = Query(50, ge=1, le=CASE_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    frame = _case_frame()
    artifacts = ["v2/results/v2_predictions.parquet"]
    if frame is None:
        return _envelope(
            None,
            artifacts,
            warnings=["v2 test rows are not available yet: run `python -m stockml.v2.train`."],
        )

    selected = frame
    if symbol:
        selected = selected[selected["symbol"] == symbol]
    if predicted != "all":
        selected = selected[selected["pred_v2_name"] == predicted.upper()]
    if actual != "all":
        selected = selected[selected["y_true_name"] == actual.upper()]
    if correct == "v2":
        selected = selected[selected["correct_v2"]]
    elif correct == "incumbent":
        selected = selected[selected["correct_inc"]]
    elif correct == "disagree":
        selected = selected[selected["pred_inc"] != selected["pred_v2"]]

    total = int(len(selected))
    page = selected.iloc[offset : offset + limit]
    rows = [
        {
            "symbol": str(row["symbol"]),
            "date": str(pd.Timestamp(row["timestamp"]).date()),
            "actual": row["y_true_name"],
            "incumbent": row["pred_inc_name"],
            "v2": row["pred_v2_name"],
            "branch": row["branch_v2"],
            "p_v2_up": float(row["p_v2_up"]),
            "p_v2_flat": float(row["p_v2_flat"]),
            "p_v2_down": float(row["p_v2_down"]),
            "correct_v2": bool(row["correct_v2"]),
            "correct_inc": bool(row["correct_inc"]),
        }
        for _, row in page.iterrows()
    ]
    data = {"total": total, "offset": offset, "limit": limit, "rows": rows}
    return _envelope(data, artifacts, warnings=[])
