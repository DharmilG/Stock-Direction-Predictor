from __future__ import annotations

"""Atomic phase-state tracking for the Track Q pipeline.

CODE READY — UNTESTED
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import load_settings
from .contracts import PHASES


def state_path() -> Path:
    return Path(load_settings().paths.state) / "q_track_state.json"


def _empty_state() -> dict[str, Any]:
    return {
        "updated_at": None,
        "phases": {phase: {"status": "pending", "updated_at": None} for phase in PHASES},
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, default=str))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_state() -> dict[str, Any]:
    """Read the phase-state file. Missing file = all phases pending."""
    path = state_path()
    if not path.exists():
        return _empty_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_state()
    phases = data.get("phases") or {}
    merged = _empty_state()
    for phase, info in phases.items():
        if phase in merged["phases"]:
            merged["phases"][phase] = {
                "status": str(info.get("status", "pending")),
                "updated_at": info.get("updated_at"),
                **{k: v for k, v in info.items() if k not in {"status", "updated_at"}},
            }
    merged["updated_at"] = data.get("updated_at")
    return merged


def save_state(state: dict[str, Any]) -> None:
    state = dict(state)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(state_path(), state)


def phase_status(phase: str) -> str:
    if phase not in PHASES:
        raise ValueError(f"Unknown phase: {phase}")
    return str(load_state()["phases"].get(phase, {}).get("status", "pending"))


def set_phase_status(phase: str, status: str, **fields: Any) -> None:
    if phase not in PHASES:
        raise ValueError(f"Unknown phase: {phase}")
    if status not in {"pending", "running", "passed", "failed", "blocked"}:
        raise ValueError(f"Unsupported status: {status}")
    state = load_state()
    entry = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    state["phases"][phase] = entry
    save_state(state)


def reset_phase(phase: str) -> None:
    set_phase_status(phase, "pending")


def previous_phases(phase: str) -> tuple[str, ...]:
    idx = PHASES.index(phase)
    return PHASES[:idx]


def gate_ready(phase: str) -> tuple[bool, str]:
    """True only when every earlier phase has passed."""
    if phase not in PHASES:
        return False, f"Unknown phase: {phase}"
    for earlier in previous_phases(phase):
        status = phase_status(earlier)
        if status != "passed":
            return False, f"{earlier} must be passed before {phase} (currently {status})"
    return True, ""
