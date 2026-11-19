"""Single source of truth for every Track Q file path.

All phase runtime state lives under ``<repo>/track_q/``. Each phase owns one
folder: ``phase_0`` (the Q0 zero-shot gate) and ``phase_3`` (thinking-cap
ablation) exist today; later phases get their folder when the phase first runs.

Nothing else in the codebase should hardcode these paths -- import from here so
that a move only needs to change this file.
"""
from __future__ import annotations

from pathlib import Path

from ..config import ROOT

TRACK_Q_ROOT = ROOT / "track_q"
COMMON_DIR = TRACK_Q_ROOT / "_common"


def phase_dir(phase: str | int) -> Path:
    """Folder for a phase id: 0 -> phase_0, 3 -> phase_3, '6_probe' -> phase_6_probe."""
    suffix = str(phase)
    return TRACK_Q_ROOT / f"phase_{suffix}"


# --------------------------------------------------------------------------- #
# Common (shared across phases)
# --------------------------------------------------------------------------- #
PHASE_REGISTRY = COMMON_DIR / "phase_registry.json"   # gate status (was q_track_state.json)
SYNC_MANIFEST = COMMON_DIR / "sync_manifest.json"     # hashes + last completed unit
CONFIG_SNAPSHOT = COMMON_DIR / "config_snapshot.yaml" # frozen q_track.yaml copy
FIO = COMMON_DIR / "fio.py"                           # fault-tolerant IO helpers


# --------------------------------------------------------------------------- #
# Phase 0 -- Q0 zero-shot gate
# --------------------------------------------------------------------------- #
P0 = phase_dir(0)
P0_DATA = P0 / "data"
P0_STATE = P0 / "state"
P0_VERDICT = P0 / "verdict"
P0_TRACES = P0 / "traces"
P0_LOGS = P0 / "logs"
P0_SCRIPTS = P0 / "scripts"
# Q0_ aliases (the codebase uses the Q0_ prefix consistently)
Q0_TRACES = P0_TRACES

Q0_POOL = P0_DATA / "pool_c.parquet"
Q0_LEDGER = P0_STATE / "qwen_rating_ledger.parquet"
Q0_VERDICT_FILE = P0_VERDICT / "q0_d1.json"
Q0_STATE_FILE = P0_STATE / "q0_d1.json"
Q0_SETUP_STATE = P0_STATE / "qwen_setup.json"
Q0_PSI_HISTORY = P0_STATE / "psi_history.jsonl"
Q0_HEARTBEAT = P0_LOGS / "q0_heartbeat.json"
Q0_SUPERVISOR = P0_LOGS / "q0_supervisor.json"
Q0_HISTORY = P0_LOGS / "q0_history.jsonl"
Q0_GPU = P0_LOGS / "gpu_5s.csv"
Q0_RESOURCE = P0_LOGS / "q0_resource_5s.csv"
Q0_DONE = P0_LOGS / "q0_DONE.txt"
Q0_ALERT = P0_LOGS / "q0_ALERT.txt"
Q0_SETUP_RESULT = P0_LOGS / "q0_SETUP_RESULT.txt"
Q0_SETUP_LOG = P0_LOGS / "q0_setup.log"
Q0_SUPERVISOR_LOG = P0_LOGS / "q0_supervisor.log"
Q0_SUPERVISOR_EVENTS = P0_LOGS / "q0_supervisor_events.log"
Q0_SUPERVISOR_TASK = P0_LOGS / "q0_supervisor_task.log"
Q0_MAIN_LOG = P0_LOGS / "q0_d1.log"


# --------------------------------------------------------------------------- #
# Phase 3 -- thinking-cap ablation
# --------------------------------------------------------------------------- #
P3 = phase_dir(3)
P3_DATA = P3 / "data"
P3_RESULTS = P3 / "results"
P3_TRACES = P3 / "traces"
P3_LOGS = P3 / "logs"
P3_SCRIPTS = P3 / "scripts"

Q3_LEDGER = P3_DATA / "q3_thinking_cap_ledger.jsonl"
Q3_RESULTS = P3_RESULTS / "q3_thinking_cap.json"
Q3_HEARTBEAT = P3_LOGS / "q3_heartbeat.json"
Q3_GPU = P3_LOGS / "q3_gpu_5s.csv"
Q3_MAIN_LOG = P3_LOGS / "q3_thinking_cap.log"
Q3_DONE = P3_LOGS / "q3_DONE.txt"


def p0_from(root_or_paths):
    """Phase-0 paths anchored at a root (or a Paths-like object with .root).

    Also accepts a minimal stub whose ``results`` points at ``<root>/results``;
    in that case the repo root is ``results.parent``. This keeps unit tests that
    stub settings working with the new tree.
    """
    if isinstance(root_or_paths, Path):
        root = root_or_paths
    elif hasattr(root_or_paths, "results"):
        root = Path(root_or_paths.results).parent
    elif hasattr(root_or_paths, "root"):
        root = Path(root_or_paths.root)
    else:
        root = Path(root_or_paths)
    base = root / "track_q" / "phase_0"
    return {
        "logs": base / "logs", "state": base / "state", "traces": base / "traces",
        "verdict": base / "verdict" / "q0_d1.json", "data": base / "data",
    }


def ensure_dirs(phase: str | int | None = None) -> None:
    """Create the folder tree (all phases, or just one). Safe to call repeatedly."""
    dirs = [COMMON_DIR]
    if phase is None:
        for p in (0, 3):
            base = phase_dir(p)
            dirs += [base, base / "data", base / "state", base / "verdict",
                     base / "results", base / "traces", base / "logs", base / "scripts"]
    else:
        base = phase_dir(phase)
        dirs += [base, base / "data", base / "state", base / "verdict",
                 base / "results", base / "traces", base / "logs", base / "scripts"]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
