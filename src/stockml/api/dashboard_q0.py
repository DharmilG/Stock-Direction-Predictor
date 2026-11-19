"""Read-only Q0 / LLM research telemetry.

The dashboard must report what the run actually wrote, never an estimate dressed
up as a measurement. Source priority is: runner heartbeat -> gate results ->
ledger -> supervisor -> GPU samples -> phase state -> setup state, and every
block records which file it came from plus how old that file is.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ..config import load_settings
from ..qpipeline.paths import p0_from
from . import q0_analytics as analytics_core
from .dashboard_support import (
    age_seconds,
    artifact_paths,
    file_meta,
    frame_file,
    json_file,
    json_safe,
    now_ist,
    to_date,
    to_iso,
)

router = APIRouter(prefix="/dashboard/q0", tags=["dashboard-q0"])

# Mirrors q0_status.ps1: the supervisor heartbeats every few seconds, the runner
# can legitimately go quiet while it rests between cases.
SUPERVISOR_STALE_SECONDS = 120
RUNNER_STALE_SECONDS = 600

RATED_VOTES = {"UP", "DOWN", "FLAT"}


def _envelope(data: Any, artifacts: list[str], *, as_of: Any = None, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "data": json_safe(data),
        "as_of": to_iso(as_of),
        "generated_at": now_ist().isoformat(),
        "source": {"kind": "runtime", "artifacts": artifacts},
        "warnings": warnings or [],
    }


def _source(name: str, path, used: bool) -> dict[str, Any]:
    meta = file_meta(path)
    return {
        "name": name,
        "path": path.name,
        "used": used,
        "available": meta["available"],
        "modified_at": meta["modified_at"],
        "age_seconds": age_seconds(meta["modified_at"]),
    }


def _parse_case(case_id: Any) -> dict[str, Any] | None:
    """``TMPV.NS|2023-04-11|179f49c9`` -> symbol / date / short hash."""
    if not isinstance(case_id, str) or not case_id:
        return None
    parts = case_id.split("|")
    if len(parts) < 3:
        return {"raw": case_id}
    return {"symbol": parts[0], "decision_date": parts[1], "case_hash": parts[2], "raw": case_id}


def _ledger_frame(paths) -> pd.DataFrame | None:
    frame = frame_file(artifact_paths(paths)["q0_ledger"])
    if frame is None or frame.empty:
        return None
    return frame


def _pool_stats(frame: pd.DataFrame | None) -> dict[str, Any]:
    """Rated = unique case ids in the ledger; judged = usable vote, no harness fail."""
    if frame is None:
        return {
            "rows": 0,
            "rated": 0,
            "judged": 0,
            "harness_fail": 0,
            "harness_fail_rate": None,
            "judged_rate": None,
            "kept_for_training": 0,
        }
    unique_ids = frame["case_id"].astype(str).nunique() if "case_id" in frame.columns else len(frame)
    votes = frame.get("qwen_vote")
    fail = frame.get("harness_fail")
    judged_mask = pd.Series(True, index=frame.index)
    if votes is not None:
        judged_mask &= votes.astype(str).str.upper().isin(RATED_VOTES)
    if fail is not None:
        judged_mask &= ~fail.fillna(False).astype(bool)
    judged = int(judged_mask.sum())
    fails = int(fail.fillna(False).astype(bool).sum()) if fail is not None else 0
    attempted = len(frame) or 1
    kept = int(frame["kept_for_training"].fillna(False).astype(bool).sum()) if "kept_for_training" in frame.columns else 0
    return {
        "rows": int(len(frame)),
        "rated": int(unique_ids),
        "judged": judged,
        "harness_fail": fails,
        "harness_fail_rate": round(fails / attempted, 6),
        "judged_rate": round(judged / attempted, 6),
        "kept_for_training": kept,
    }


def _gpu_trend(paths, limit: int = 120) -> dict[str, Any]:
    path = artifact_paths(paths)["q0_gpu"]
    meta = file_meta(path)
    if not meta["available"]:
        return {"available": False, "samples": []}
    try:
        frame = pd.read_csv(path, usecols=["timestamp", "temp_c"]).tail(limit)
    except Exception:
        return {"available": False, "samples": [], "note": "q0 gpu samples (track_q/phase_0/logs/gpu_5s.csv) unreadable"}
    samples = [
        {"at": str(row["timestamp"]), "temp_c": None if pd.isna(row["temp_c"]) else float(row["temp_c"])}
        for _, row in frame.iterrows()
    ]
    return {
        "available": True,
        "samples": samples,
        "latest_c": samples[-1]["temp_c"] if samples else None,
        "peak_c": max((s["temp_c"] for s in samples if s["temp_c"] is not None), default=None),
        # q0_d1.py logs naive local wall clock here, so it is NOT comparable to UTC.
        "timezone": "naive-local (written by q0_d1._append_gpu_sample)",
        "size_bytes": meta["size_bytes"],
    }


def _phases(paths) -> dict[str, Any]:
    state = json_file(artifact_paths(paths)["q_track_state"])
    try:
        from ..qpipeline.contracts import PHASES
    except Exception:
        PHASES = ("phase3", "phase6_probe", "phase4", "phase5", "phase6_train", "phase7", "phase8")
    if not state:
        return {
            "available": False,
            "phases": [
                {"name": name, "status": "pending", "updated_at": None, "note": None}
                for name in PHASES
            ],
            "note": "track_q/_common/phase_registry.json has not been written; the runner creates it when a phase starts.",
        }
    entries = []
    for name in PHASES:
        record = state.get(name) or {}
        entries.append({
            "name": name,
            "status": record.get("status", "pending"),
            "updated_at": record.get("updated_at"),
            "note": record.get("note") or record.get("message"),
        })
    return {"available": True, "phases": entries, "updated_at": state.get("updated_at")}


def _gate(paths) -> dict[str, Any]:
    verdict = json_file(artifact_paths(paths)["q0_verdict"])
    state = json_file(p0_from(paths.root)["state"] / "q0_d1.json")
    pool_total = verdict.get("pool_total") or state.get("pool_total")
    n = verdict.get("n")
    gate_pass = verdict.get("pass")
    complete = verdict.get("complete")
    # A partial run records a *running* pass flag; reporting that as FAIL would
    # claim a verdict the run never reached.
    if complete is False:
        label = "INCOMPLETE"
    elif gate_pass is True:
        label = "PASS"
    elif gate_pass is False:
        label = "FAIL"
    else:
        label = "UNKNOWN"
    return {
        "available": bool(verdict),
        "verdict": label,
        "recorded_pass": gate_pass if isinstance(gate_pass, bool) else None,
        "passed": gate_pass if (isinstance(gate_pass, bool) and complete is not False) else None,
        "complete": complete if isinstance(complete, bool) else None,
        "n": n,
        "pool_total": pool_total,
        "qwen_agreement": verdict.get("qwen_agreement"),
        "finbert_agreement": verdict.get("finbert_agreement"),
        "delta": verdict.get("delta"),
        "se_diff": verdict.get("se_diff"),
        "bar_2se": verdict.get("bar_2se"),
        "harness_fail_rate": verdict.get("harness_fail_rate"),
        "rated_total": verdict.get("rated_total"),
        "recorded_at": file_meta(artifact_paths(paths)["q0_verdict"])["modified_at"],
        "state_status": state.get("status"),
        "state_verdict_path": state.get("verdict"),
        "note": (
            "Gate pass/fail is copied verbatim from track_q/phase_0/verdict/q0_d1.json; nothing is recomputed here."
            if verdict
            else "No gate verdict has been recorded yet."
        ),
    }


@router.get("/status")
def q0_status() -> dict[str, Any]:
    settings = load_settings()
    paths = settings.paths
    artifacts = artifact_paths(paths)

    heartbeat = json_file(artifacts["q0_heartbeat"])
    supervisor = json_file(artifacts["q0_supervisor"])
    setup = json_file(artifacts["q0_setup"])
    ledger = _ledger_frame(paths)

    heartbeat_age = age_seconds(heartbeat.get("updated_at"))
    supervisor_age = age_seconds(supervisor.get("updated_at"))
    runner_stale = heartbeat_age is not None and heartbeat_age > RUNNER_STALE_SECONDS
    supervisor_stale = supervisor_age is not None and supervisor_age > SUPERVISOR_STALE_SECONDS

    pools = _pool_stats(ledger)
    stats = heartbeat.get("pool_stats") if isinstance(heartbeat.get("pool_stats"), dict) else {}

    warnings: list[str] = []
    if not heartbeat:
        warnings.append("No runner heartbeat (track_q/phase_0/logs/q0_heartbeat.json). The run is not active.")
    elif runner_stale:
        warnings.append(
            f"Runner heartbeat is {heartbeat_age}s old (limit {RUNNER_STALE_SECONDS}s): the run is stopped or resting."
        )
    if supervisor and supervisor_stale:
        warnings.append(f"Supervisor heartbeat is {supervisor_age}s old: the supervisor is not running.")
    if heartbeat and pools["rated"] == 0:
        warnings.append("The ledger has no rows yet; rated/judged counts will stay at zero until the first case lands.")

    status = "stale" if (not heartbeat or runner_stale or supervisor_stale) else "live"
    gate = _gate(paths)
    if gate["available"] and gate["passed"] is None:
        warnings.append("Gate verdict present but the pass field is missing or non-boolean.")

    return _envelope(
        {
            "runtime": {
                "status": status,
                "phase": heartbeat.get("phase"),
                "complete": heartbeat.get("complete"),
                "updated_at": heartbeat.get("updated_at"),
                "heartbeat_age_seconds": heartbeat_age,
                "runner_stale": runner_stale,
                "supervisor_stale": supervisor_stale,
                "pid": heartbeat.get("pid"),
                "run_started_at": heartbeat.get("run_started_at"),
                "current_case": _parse_case(heartbeat.get("current_case")),
                "stale_limits": {"runner_seconds": RUNNER_STALE_SECONDS, "supervisor_seconds": SUPERVISOR_STALE_SECONDS},
            },
            "progress": {
                "pool_total": heartbeat.get("pool_total") or gate.get("pool_total"),
                "ledger_total": heartbeat.get("ledger_total") or pools["rows"],
                "rated": heartbeat.get("rated_total") or pools["rated"],
                "judged": heartbeat.get("judged_total") or pools["judged"],
                "attempted": heartbeat.get("attempted"),
                "harness_fail_rate": heartbeat.get("harness_fail_rate", pools["harness_fail_rate"]),
                "eta_hours": heartbeat.get("eta_hours"),
                "avg_case_seconds": heartbeat.get("avg_case_seconds"),
                "counts_are_distinct": True,
                "definitions": {
                    "rated": "unique case ids in the ledger",
                    "judged": "rated cases with a usable UP/DOWN/FLAT vote and no harness failure",
                },
                "pool_stats_from_heartbeat": stats or None,
            },
            "thermal": {
                "state": heartbeat.get("thermal_state"),
                "temperature_c": heartbeat.get("temperature_c"),
                "max_temperature_c": heartbeat.get("max_temperature_c"),
                "clock_mhz": supervisor.get("clock_mhz"),
                "applied_clock_mhz": supervisor.get("applied_clock_mhz"),
                "cap_mhz": supervisor.get("clock_cap_mhz"),
                "rests": heartbeat.get("rests"),
                "hard_trips": heartbeat.get("hard_trips"),
                "cooldowns": heartbeat.get("cooldowns"),
                "alarms": heartbeat.get("alarms"),
                "prepares": heartbeat.get("prepares"),
                "aborts": heartbeat.get("aborts"),
            },
            "timing": {
                "run_started_at": heartbeat.get("run_started_at"),
                "last_checkpoint_at": heartbeat.get("last_checkpoint_at"),
                "work_seconds": heartbeat.get("work_seconds"),
                "total_model_seconds": heartbeat.get("total_model_seconds"),
                "proactive_target_seconds": heartbeat.get("proactive_target_seconds"),
            },
            "supervisor": {
                "available": bool(supervisor),
                "state": supervisor.get("state"),
                "message": supervisor.get("message"),
                "pid": supervisor.get("supervisor_pid"),
                "child_pid": supervisor.get("child_pid"),
                "child_alive": supervisor.get("child_alive"),
                "restart_in_seconds": supervisor.get("restart_in_seconds"),
                "last_exit_code": supervisor.get("last_exit_code"),
                "updated_at": supervisor.get("updated_at"),
                "age_seconds": supervisor_age,
                "mutex": supervisor.get("mutex_name"),
            },
            "gate": gate,
            "phases": _phases(paths),
            "setup": {
                "available": bool(setup),
                "phase": setup.get("phase"),
                "status": setup.get("status"),
                "model": setup.get("model") or setup.get("qwen_model"),
                "quantization": setup.get("quantization"),
                "ctx_pinned": setup.get("ctx_pinned"),
                "num_predict_min": setup.get("num_predict_min"),
                "corrected_fields": setup.get("corrections") or setup.get("reported_corrections"),
                "raw": setup or None,
            },
            "gpu": _gpu_trend(paths),
            "resume": {
                "note": "Read-only hint. The dashboard never starts or stops the run.",
                "command": "powershell -NoProfile -ExecutionPolicy Bypass -File .\\setup_q0_task.ps1",
                "alternative": "powershell -NoProfile -ExecutionPolicy Bypass -File .\\run_q0_overnight.ps1",
                "resumable": True,
                "ledger_is_append_only": True,
            },
            "sources": [
                _source("runner heartbeat", artifacts["q0_heartbeat"], bool(heartbeat)),
                _source("gate verdict", artifacts["q0_verdict"], bool(gate["available"])),
                _source("rating ledger", artifacts["q0_ledger"], ledger is not None),
                _source("supervisor heartbeat", artifacts["q0_supervisor"], bool(supervisor)),
                _source("gpu samples", artifacts["q0_gpu"], file_meta(artifacts["q0_gpu"])["available"]),
                _source("phase state", artifacts["q_track_state"], file_meta(artifacts["q_track_state"])["available"]),
                _source("setup state", artifacts["q0_setup"], bool(setup)),
            ],
        },
        [
            "track_q/phase_0/logs/q0_heartbeat.json",
            "track_q/phase_0/logs/q0_supervisor.json",
            "track_q/phase_0/logs/gpu_5s.csv",
            "track_q/phase_0/verdict/q0_d1.json",
            "track_q/phase_0/state/q0_d1.json",
            "track_q/_common/phase_registry.json",
            "artifacts/state/qwen_setup.json",
            "track_q/phase_0/state/qwen_rating_ledger.parquet",
        ],
        as_of=heartbeat.get("updated_at") or file_meta(artifacts["q0_ledger"])["modified_at"],
        warnings=warnings,
    )


@router.get("/ledger-summary")
def q0_ledger_summary() -> dict[str, Any]:
    settings = load_settings()
    paths = settings.paths
    frame = _ledger_frame(paths)
    if frame is None:
        raise HTTPException(status_code=404, detail="track_q/phase_0/state/qwen_rating_ledger.parquet is missing")

    pools = _pool_stats(frame)
    votes = frame["qwen_vote"].astype(str).str.upper() if "qwen_vote" in frame.columns else pd.Series(dtype=str)
    judged_mask = votes.isin(RATED_VOTES)
    if "harness_fail" in frame.columns:
        judged_mask &= ~frame["harness_fail"].fillna(False).astype(bool)
    judged = frame[judged_mask]

    def share(column: str) -> dict[str, int]:
        if column not in frame.columns:
            return {}
        return {str(k): int(v) for k, v in frame[column].fillna("(missing)").astype(str).value_counts().head(12).items()}

    def mean_of(column: str, source: pd.DataFrame = None) -> float | None:
        target = frame if source is None else source
        if column not in target.columns:
            return None
        series = pd.to_numeric(target[column], errors="coerce").dropna()
        return float(series.mean()) if len(series) else None

    dates = pd.to_datetime(frame["decision_date"], errors="coerce").dropna() if "decision_date" in frame.columns else pd.Series(dtype="datetime64[ns]")
    thermal = frame["thermal_state"].fillna("(missing)").astype(str).value_counts()
    return _envelope(
        {
            "rows": int(len(frame)),
            **pools,
            "pools": share("pool"),
            "models": share("model_stage"),
            "prompt_versions": share("prompt_version"),
            "votes": share("qwen_vote"),
            "drop_reasons": share("drop_reason"),
            "thermal_states": {str(k): int(v) for k, v in thermal.items()},
            "flags": {
                "truncated": int(frame["truncated"].fillna(False).astype(bool).sum()) if "truncated" in frame.columns else None,
                "no_news": int(frame["no_news"].fillna(False).astype(bool).sum()) if "no_news" in frame.columns else None,
                "harness_fail": pools["harness_fail"],
                "finbert_qwen_disagree": int(frame["finbert_qwen_disagree"].fillna(False).astype(bool).sum())
                if "finbert_qwen_disagree" in frame.columns
                else None,
            },
            "agreement": {
                "qwen_matches_label": mean_of("vote_matches_label", judged),
                "finbert_matches_label": mean_of("finbert_matches_label", judged),
                "finbert_qwen_disagree_rate": mean_of("finbert_qwen_disagree"),
                "judged_rows": int(len(judged)),
            },
            "confidence": {
                "qwen_conf_mean": mean_of("qwen_conf"),
                "qwen_conf_p10": float(pd.to_numeric(frame.get("qwen_conf"), errors="coerce").quantile(0.10))
                if "qwen_conf" in frame.columns
                else None,
                "qwen_conf_p90": float(pd.to_numeric(frame.get("qwen_conf"), errors="coerce").quantile(0.90))
                if "qwen_conf" in frame.columns
                else None,
                "finbert_score_mean": mean_of("finbert_score"),
            },
            "trace": {
                "mean_len": mean_of("qwen_trace_len"),
                "max_len": float(pd.to_numeric(frame.get("qwen_trace_len"), errors="coerce").max())
                if "qwen_trace_len" in frame.columns
                else None,
            },
            "gpu": {
                "temp_start_mean": mean_of("gpu_temp_start"),
                "temp_end_mean": mean_of("gpu_temp_end"),
                "temp_end_max": float(pd.to_numeric(frame.get("gpu_temp_end"), errors="coerce").max())
                if "gpu_temp_end" in frame.columns
                else None,
            },
            "range": {
                "start": to_date(dates.min()).isoformat() if len(dates) else None,
                "end": to_date(dates.max()).isoformat() if len(dates) else None,
                "symbols": int(frame["symbol"].astype(str).nunique()) if "symbol" in frame.columns else None,
            },
            "top_symbols": share("symbol"),
        },
        ["track_q/phase_0/state/qwen_rating_ledger.parquet"],
        as_of=file_meta(artifact_paths(paths)["q0_ledger"])["modified_at"],
    )


# --------------------------------------------------------------------------- #
# Live timeline / analytics / case browser
#
# Everything below is derived from the files the run writes — the ledger, the
# trace JSONs, the resource CSVs and the heartbeat history. A missing file is
# reported as ``data: null`` plus a warning (200); it is never smoothed over with
# a synthetic series, because a fake utilisation line is worse than no line.
# --------------------------------------------------------------------------- #
TIMELINE_ARTIFACTS = [
    "track_q/phase_0/logs/q0_resource_5s.csv",
    "track_q/phase_0/logs/gpu_5s.csv",
    "track_q/phase_0/logs/q0_history.jsonl",
    "track_q/phase_0/logs/q0_heartbeat.json",
    "track_q/phase_0/state/qwen_rating_ledger.parquet",
    "track_q/phase_0/state/psi_history.jsonl",
]
CASES_ARTIFACTS = ["track_q/phase_0/state/qwen_rating_ledger.parquet", "track_q/phase_0/traces/*.json"]


@router.get("/timeline")
def q0_timeline(
    window: str = Query("2h", pattern="^(30m|2h|all)$"),
    step: int = Query(5, ge=5, le=300),
) -> dict[str, Any]:
    settings = load_settings()
    paths = settings.paths
    payload = analytics_core.timeline(paths, _ledger_frame(paths), window=window, step=step)
    warnings = list(payload.pop("warnings", []))
    if not payload["resource"]["available"] and not payload["history"]["available"] and not payload["progress"]["available"]:
        return _envelope(None, TIMELINE_ARTIFACTS, warnings=warnings + ["No Q0 telemetry has been written yet."])
    return _envelope(
        payload,
        TIMELINE_ARTIFACTS,
        as_of=file_meta(p0_from(paths.root)["logs"] / "q0_history.jsonl")["modified_at"] or file_meta(artifact_paths(paths)["q0_gpu"])["modified_at"],
        warnings=warnings,
    )


@router.get("/analytics")
def q0_analytics() -> dict[str, Any]:
    settings = load_settings()
    paths = settings.paths
    frame = _ledger_frame(paths)
    if frame is None:
        return _envelope(
            None,
            CASES_ARTIFACTS,
            warnings=["track_q/phase_0/state/qwen_rating_ledger.parquet is missing; there is nothing to analyse yet."],
        )
    payload = analytics_core.analytics(paths, frame)
    warnings = []
    if not payload["latency"]["available"]:
        warnings.append("No trace files were readable, so latency is absent rather than estimated.")
    if payload["calibration"]["ece"] is None:
        warnings.append("Calibration needs both qwen_conf and vote_matches_label; one of them is missing.")
    return _envelope(
        payload,
        CASES_ARTIFACTS,
        as_of=file_meta(artifact_paths(paths)["q0_ledger"])["modified_at"],
        warnings=warnings,
    )


@router.get("/cases")
def q0_cases(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    symbol: str = Query("", max_length=40),
    vote: str = Query("all", pattern="^(all|UP|DOWN|FLAT)$"),
    outcome: str = Query("all", pattern="^(all|match|mismatch)$"),
) -> dict[str, Any]:
    settings = load_settings()
    paths = settings.paths
    frame = _ledger_frame(paths)
    if frame is None:
        return _envelope(
            None,
            CASES_ARTIFACTS,
            warnings=["track_q/phase_0/state/qwen_rating_ledger.parquet is missing; the case browser has no source."],
        )
    payload = analytics_core.case_rows(frame, limit=limit, offset=offset, symbol=symbol, vote=vote, outcome=outcome)
    warnings = [] if payload["total"] else ["No cases match those filters."]
    return _envelope(
        payload,
        CASES_ARTIFACTS,
        as_of=file_meta(artifact_paths(paths)["q0_ledger"])["modified_at"],
        warnings=warnings,
    )


@router.get("/case/{case_id}")
def q0_case(case_id: str) -> dict[str, Any]:
    settings = load_settings()
    paths = settings.paths
    # Path safety is checked before the ledger is even read: an id carrying a
    # separator is a traversal attempt, not a missing case, so it is a 404 here.
    if not analytics_core.safe_case_id(case_id):
        raise HTTPException(status_code=404, detail="Unknown case id")
    payload = analytics_core.case_detail(paths, _ledger_frame(paths), case_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"No ledger row or trace for {case_id}")
    warnings = [] if payload["trace_available"] else ["The trace file for this case is missing; only ledger fields are shown."]
    return _envelope(payload, CASES_ARTIFACTS, warnings=warnings)
