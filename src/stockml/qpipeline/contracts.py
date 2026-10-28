from __future__ import annotations

"""Shared constants and artifact paths for the Track Q pipeline.

CODE READY — UNTESTED
"""

from pathlib import Path

import yaml

from ..config import ROOT


Q_TRACK_CONFIG_PATH = ROOT / "config" / "q_track.yaml"
Q_TRACK_STATE_PATH = ROOT / "artifacts" / "state" / "q_track_state.json"
Q0_RESULTS_PATH = ROOT / "results" / "q0_d1.json"

PHASES = ("phase3", "phase6_probe", "phase4", "phase5", "phase6_train", "phase7", "phase8")


def load_q_track_config() -> dict:
    """Read config/q_track.yaml. Pure file read — no side effects."""
    if not Q_TRACK_CONFIG_PATH.exists():
        return {}
    data = yaml.safe_load(Q_TRACK_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("config/q_track.yaml must be a mapping")
    return data


def resolve_path(value: str) -> Path:
    """Resolve a path from config relative to project root."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def q0_gate_ok() -> bool:
    """True only when results/q0_d1.json reports complete=true and pass=true."""
    if not Q0_RESULTS_PATH.exists():
        return False
    try:
        import json

        data = json.loads(Q0_RESULTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("complete") is True and data.get("pass") is True)


def q0_summary() -> dict:
    if not Q0_RESULTS_PATH.exists():
        return {"complete": False, "pass": False, "source": str(Q0_RESULTS_PATH)}
    try:
        import json

        data = json.loads(Q0_RESULTS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"complete": False, "pass": False, "error": str(exc)}
    return {
        "complete": bool(data.get("complete")),
        "pass": bool(data.get("pass")),
        "rated_total": data.get("rated_total"),
        "judged_total": data.get("judged_total"),
        "delta": data.get("delta"),
        "bar_2se": data.get("bar_2se"),
        "source": str(Q0_RESULTS_PATH),
    }


def phase_artifacts() -> dict[str, dict[str, str]]:
    """Canonical artifact paths per phase, from config."""
    cfg = load_q_track_config()
    p3 = cfg.get("phase3", {})
    star = cfg.get("star", {})
    nightly = cfg.get("nightly", {})
    p6 = cfg.get("phase6", {})
    p8 = cfg.get("phase8", {})
    return {
        "phase3": {
            "ledger": str(resolve_path(p3.get("ledger_path", "data/llm/q3_thinking_cap_ledger.parquet"))),
            "results": str(resolve_path(p3.get("results_path", "results/q3_thinking_cap.json"))),
            "trace_dir": str(resolve_path(p3.get("trace_dir", "results/traces_q3"))),
        },
        "phase4": {
            "pool_a": str(resolve_path(star.get("pool_a_path", "data/llm/pool_a.parquet"))),
            "candidates": str(resolve_path(star.get("candidates_path", "data/llm/star_candidates.jsonl"))),
            "trace_dir": str(resolve_path(star.get("trace_dir", "results/traces_star"))),
        },
        "phase5": {
            "kept": str(resolve_path(star.get("kept_path", "data/llm/star_kept.jsonl"))),
            "dropped": str(resolve_path(star.get("dropped_path", "data/llm/star_dropped.jsonl"))),
            "audit_template": str(resolve_path(star.get("audit_template_path", "results/star_audit_template.csv"))),
            "audit_results": str(resolve_path(star.get("audit_results_path", "results/star_audit_results.csv"))),
            "report": str(resolve_path(star.get("report_path", "results/phase5_filter_audit.json"))),
        },
        "phase6_probe": {
            "results": str(resolve_path(p6.get("probe_results_path", "results/qwen_10step_probe.json"))),
        },
        "phase6_train": {
            "adapter_dir": str(resolve_path(p6.get("adapter_dir", "artifacts/models/qwen_lora_r1"))),
            "results": str(resolve_path(p6.get("results_path", "results/qwen_qlora.json"))),
        },
        "phase7": {
            "ledger": str(resolve_path(nightly.get("ledger_path", "data/llm/qwen_nightly_ledger.parquet"))),
            "results": str(resolve_path(nightly.get("results_path", "results/qwen_nightly.json"))),
            "trace_dir": str(resolve_path(nightly.get("trace_dir", "results/traces_q7"))),
        },
        "phase8": {
            "results": str(resolve_path(p8.get("results_path", "results/q8_ablation.json"))),
            "touch_ledger": str(resolve_path(p8.get("touch_ledger_path", "results/holdout_touches.jsonl"))),
        },
    }
