from __future__ import annotations

"""Q0-D1 zero-shot gate (Track Q, Phase 2.5).

Frozen base model (no LoRA) on Pool C (n=1500 stratified pre-holdout,
decision_date < 2023-08-24). Compares Qwen vote-agreement vs FinBERT
agreement on the identical cases. Bar: Δ > 2×SE (≈3.6pp at p≈0.47).

Resumable: appends rating-ledger rows to track_q/phase_0/state/qwen_rating_ledger.parquet;
re-runs skip completed case_ids. Pool C verdict → track_q/phase_0/verdict/q0_d1.json.

Usage:
    python -m stockml.data.q0_d1 --build-pool
    python -m stockml.data.q0_d1 --pilot 30
    python -m stockml.data.q0_d1 --full
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_settings
from ..qpipeline.paths import (
    Q0_LEDGER, Q0_POOL, Q0_VERDICT_FILE, Q0_STATE_FILE, Q0_TRACES,
    Q0_HEARTBEAT, Q0_GPU, Q0_RESOURCE, Q0_HISTORY, Q0_MAIN_LOG,
)
from ..utils.io import atomic_write_json, atomic_write_parquet
from ..utils.logging import setup_logging
from ..utils.state import update_state
from ..features.decision_date import decision_dates
from .qwen_harness import (
    HOLDOUT_START,
    MODEL,
    STAR_CUTOFF,
    build_prompt,
    rate_case,
    write_trace,
)

LABEL_TO_VOTE = {0: "DOWN", 1: "FLAT", 2: "UP"}
N_DOWN, N_FLAT, N_UP = 500, 250, 750

# --- Thermal guard (2026-10-20, user-specified ladder) -----------------------
# Continuous until 80C; 84C prepare, 85C alarm, 87C hard abort.  Every thermal
# rest unloads the model first.  Four hard stops inside one rolling hour force
# a 30-minute full cooldown (counters reset) rather than ending the run.
GPU_WATCH_C = 80
GPU_PREPARE_C = 84
GPU_ALARM_C = 85
GPU_ABORT_C = 87
GPU_REST_MINUTES = 20
GPU_COOLDOWN_MINUTES = 30
GPU_TRIPS_PER_WINDOW = 4
GPU_TRIP_WINDOW_MINUTES = 60
GPU_MONITOR_INTERVAL_SECONDS = 5
PROACTIVE_WORK_MINUTES = 60
PROACTIVE_REST_MINUTES = 15
SOFT_BREATHER_SECONDS = 30
THERMAL_SAMPLE_LOG = "gpu_5s.csv"
RESOURCE_SAMPLE_LOG = "q0_resource_5s.csv"
HISTORY_LOG = "q0_history.jsonl"
HISTORY_EVERY_SECONDS = 30.0
_GPU_SAMPLE_LOCK = threading.Lock()
_LAST_HISTORY_TS = [0.0]

# One batched query: eight fields, one nvidia-smi process. Asking for the same
# numbers in separate invocations costs ~200 ms each on Windows, and the guard
# runs every 5 s.
_GPU_QUERY = (
    "temperature.gpu,utilization.gpu,memory.used,memory.total,"
    "power.draw,fan.speed,clocks.sm,clocks.max.sm"
)
RESOURCE_FIELDS = (
    "temp_c", "util_pct", "mem_used_mb", "mem_total_mb",
    "power_w", "fan_pct", "clock_sm_mhz", "clock_max_sm_mhz",
)


@dataclass
class ThermalState:
    """Per-run thermal counters.  Not persisted; the ledger is the durable log."""

    hard_trips: int = 0
    rests: int = 0
    cooldowns: int = 0
    proactive_breaks: int = 0
    alarms: int = 0
    prepares: int = 0
    aborts: int = 0
    max_temp_c: float | None = None
    rest_minutes: float = 0.0
    trip_times: list[float] = field(default_factory=list)
    work_seconds: float = 0.0
    last_temp_c: float | None = None
    last_state: str = "green"


def _parse_temp(output: str) -> float | None:
    for line in (output or "").splitlines():
        line = line.strip()
        if line[:2].isdigit() or line[:3].replace(".", "", 1).isdigit():
            try:
                return float(line.split()[0])
            except ValueError:
                continue
    return None


def gpu_temp_c() -> float | None:
    """First GPU die temperature via nvidia-smi. None if unreadable."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return _parse_temp(out.stdout)


def _parse_row(output: str) -> list[str]:
    """First data row of a ``--format=csv,noheader`` query, split into fields."""
    for line in (output or "").splitlines():
        if line.strip():
            return [part.strip() for part in line.split(",")]
    return []


def _to_float(value: str) -> float | None:
    """nvidia-smi prints ``[N/A]`` for fields a card does not expose."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def gpu_sample() -> dict[str, float | None] | None:
    """Every GPU field the Q0 charts plot, from a single nvidia-smi call.

    ``None`` means the query itself failed (no nvidia-smi, driver error) and no
    series should be drawn at all. A ``None`` *value* inside the dict means the
    card answered but does not report that field — e.g. ``fan.speed`` on a
    passively cooled or laptop part — which must be shown as a gap, never as 0.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={_GPU_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    row = _parse_row(out.stdout)
    if not row:
        return None
    values = [_to_float(part) for part in row[: len(RESOURCE_FIELDS)]]
    values += [None] * (len(RESOURCE_FIELDS) - len(values))
    return dict(zip(RESOURCE_FIELDS, values))


def _track_temp(state: ThermalState, temp: float | None) -> None:
    if temp is None:
        return
    state.last_temp_c = temp
    if state.max_temp_c is None or temp > state.max_temp_c:
        state.max_temp_c = temp


def _fmt_temp(temp: float | None) -> str:
    return f"{temp:.0f}C" if temp is not None else "unknown"


def _append_gpu_sample(path: Path, case_id: str, temp: float) -> None:
    """Append one 5-second sample; cheap enough for ~30k rows over the run."""
    with _GPU_SAMPLE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        with path.open("a", encoding="utf-8") as fh:
            if new_file:
                fh.write("timestamp,case_id,temp_c\n")
            fh.write(f"{datetime.now().isoformat(timespec='seconds')},{case_id},{temp:g}\n")


def _append_resource_sample(path: Path, case_id: str, sample: dict[str, float | None]) -> None:
    """Superset of ``gpu_5s.csv``: one row per sample, empty cell = not reported.

    ``gpu_5s.csv`` keeps being written unchanged so existing readers and old runs
    stay valid; this file exists for the charts that need more than temperature.
    """
    stamp = datetime.now().isoformat(timespec="seconds")
    cells = [stamp, case_id]
    for name in RESOURCE_FIELDS:
        value = sample.get(name)
        cells.append("" if value is None else f"{value:g}")
    with _GPU_SAMPLE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        with path.open("a", encoding="utf-8") as fh:
            if new_file:
                fh.write("timestamp,case_id," + ",".join(RESOURCE_FIELDS) + "\n")
            fh.write(",".join(cells) + "\n")


def _append_history(path: Path, payload: dict) -> None:
    """Append a throttled heartbeat row so the run has a history, not only a now.

    The heartbeat file is overwritten in place (one snapshot); without this the
    timeline could only ever show the latest value. Rows are whole heartbeats,
    because the interesting patterns are co-movements (temperature vs throughput)
    that a per-field log would not preserve.
    """
    now = time.monotonic()
    with _GPU_SAMPLE_LOCK:
        if now - _LAST_HISTORY_TS[0] < HISTORY_EVERY_SECONDS:
            return
        _LAST_HISTORY_TS[0] = now
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
        except OSError:
            pass


def _write_heartbeat(path: Path, **fields) -> None:
    """Atomic per-case progress snapshot for unattended monitoring.

    Failure to write a heartbeat must never interrupt a run that is otherwise
    making progress, so this helper swallows all exceptions.
    """
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}
    payload.update(fields)
    try:
        atomic_write_json(Path(path), payload)
    except Exception:
        pass


def _pool_stats(ledger: pd.DataFrame) -> dict[str, float | int]:
    """Return rated/judged counts and harness-failure rate for Pool C rows."""
    empty = {"rated_total": 0, "judged_total": 0, "harness_fail_rate": 0.0}
    if ledger is None or len(ledger) == 0 or "pool" not in ledger.columns:
        return empty
    rows = ledger[ledger["pool"] == "C"]
    if len(rows) == 0:
        return empty
    rated_total = int(rows["case_id"].nunique()) if "case_id" in rows.columns else int(len(rows))
    if "harness_fail" in rows.columns:
        fail = rows["harness_fail"].fillna(False).astype(bool)
    else:
        fail = pd.Series(False, index=rows.index)
    harness_fail_rate = float(fail.mean()) if len(fail) else 0.0
    if "qwen_vote" in rows.columns:
        judged = rows[rows["qwen_vote"].isin(["UP", "DOWN", "FLAT"]) & ~fail]
    else:
        judged = rows.iloc[0:0]
    return {
        "rated_total": rated_total,
        "judged_total": int(len(judged)),
        "harness_fail_rate": harness_fail_rate,
    }


def unload_model(logger) -> bool:
    """Best-effort `ollama stop` before any rest. Never raises."""
    exe = shutil.which("ollama")
    if exe is None:
        logger.warning("ollama CLI not found — cannot unload model before rest.")
        return False
    try:
        out = subprocess.run([exe, "stop", MODEL], capture_output=True,
                             text=True, timeout=30)
    except Exception as exc:  # noqa: BLE001 — thermal path must never crash
        logger.warning("ollama stop failed: %s", exc)
        return False
    if out.returncode != 0:
        logger.warning("ollama stop rc=%s stderr=%s", out.returncode,
                       (out.stderr or "").strip()[:200])
        return False
    logger.warning("Unloaded %s before thermal rest.", MODEL)
    return True


def _rest(logger, state: ThermalState, minutes: int, reason: str) -> float | None:
    """Unload, sleep a fixed number of minutes, log the after temperature."""
    logger.warning("Thermal rest: %s — unloading and sleeping %d min.", reason, minutes)
    unload_model(logger)
    time.sleep(minutes * 60)
    state.rests += 1
    state.rest_minutes += minutes
    state.work_seconds = 0.0
    after = gpu_temp_c()
    _track_temp(state, after)
    logger.warning("Thermal rest done (%d min, now %s) — resuming.",
                   minutes, _fmt_temp(after))
    return after


def handle_hard_trip(logger, state: ThermalState, temp: float | None,
                     case_id: str, source: str) -> None:
    """Count an 87C stop; 4th inside the rolling hour -> 30-min cooldown."""
    now = time.monotonic()
    state.hard_trips += 1
    state.trip_times = [t for t in state.trip_times
                        if now - t < GPU_TRIP_WINDOW_MINUTES * 60]
    state.trip_times.append(now)
    if len(state.trip_times) >= GPU_TRIPS_PER_WINDOW:
        state.cooldowns += 1
        state.trip_times.clear()
        logger.warning(
            "HARD STOP %s at %s on %s — %d trips within %d min; "
            "30-min cooldown and full counter reset.",
            source, _fmt_temp(temp), case_id, GPU_TRIPS_PER_WINDOW,
            GPU_TRIP_WINDOW_MINUTES)
        _rest(logger, state, GPU_COOLDOWN_MINUTES, "4-per-hour cooldown")
    else:
        logger.warning(
            "HARD STOP %s at %s on %s — 20-min rest (%d/%d in window).",
            source, _fmt_temp(temp), case_id, len(state.trip_times),
            GPU_TRIPS_PER_WINDOW)
        _rest(logger, state, GPU_REST_MINUTES, "87C hard stop")


def maybe_proactive_rest(logger, state: ThermalState) -> bool:
    """15-minute unloaded break after 60 minutes of actual GPU work."""
    if state.work_seconds < PROACTIVE_WORK_MINUTES * 60:
        return False
    state.proactive_breaks += 1
    logger.warning("Proactive break: %d min of actual work completed — 15-min unloaded rest.",
                   PROACTIVE_WORK_MINUTES)
    unload_model(logger)
    time.sleep(PROACTIVE_REST_MINUTES * 60)
    state.rests += 1
    state.rest_minutes += PROACTIVE_REST_MINUTES
    state.work_seconds = 0.0
    after = gpu_temp_c()
    _track_temp(state, after)
    logger.warning("Proactive break done (now %s) — resuming.", _fmt_temp(after))
    return True


def thermal_gate_before_case(logger, state: ThermalState, case_id: str) -> float | None:
    """Blocking pre-case guard. Returns the temperature after any rest.

    - 87C+: hard stop, 20-min rest (or 30-min cooldown on the 4th trip/hour).
    - 85-86C: alarm only, keep going case-by-case.
    - 84C: prepare only, no unload while stable.
    - 80-83C: 30s breather only if the temperature rose since the last check.
    """
    maybe_proactive_rest(logger, state)
    previous = state.last_temp_c
    temp = gpu_temp_c()
    if temp is None:
        logger.warning("GPU temp unreadable (nvidia-smi missing?) — continuing without thermal gate.")
        state.last_state = "unknown"
        return None
    _track_temp(state, temp)

    if temp >= GPU_ABORT_C:
        handle_hard_trip(logger, state, temp, case_id, "pre-case")
        after = gpu_temp_c()
        _track_temp(state, after)
        state.last_state = "rest"
        return after
    if temp >= GPU_ALARM_C:
        state.alarms += 1
        state.last_state = "alarm"
        logger.warning("GPU %s — ALARM (85-86C); continuing case-by-case, hard stop at %dC.",
                       _fmt_temp(temp), GPU_ABORT_C)
        return temp
    if temp >= GPU_PREPARE_C:
        state.prepares += 1
        state.last_state = "prepare"
        logger.warning("GPU %s — PREPARE; no unload while stable at 84C.", _fmt_temp(temp))
        return temp
    if temp >= GPU_WATCH_C:
        state.last_state = "amber"
        if previous is not None and temp > previous:
            logger.warning("GPU %s rising from %s — 30s breather before case.",
                           _fmt_temp(temp), _fmt_temp(previous))
            time.sleep(SOFT_BREATHER_SECONDS)
        return temp
    state.last_state = "green"
    return temp


class ThermalMonitor(threading.Thread):
    """Samples the GPU every 5s while a case is in flight.

    One batched nvidia-smi query per tick feeds both logs: ``gpu_5s.csv`` (the
    original temperature-only file, unchanged) and ``q0_resource_5s.csv`` (the
    superset the Q0 charts read). At >=87C it sets ``abort_event`` and unloads
    the model.  ``rate_case`` sees the event and returns without retrying, so the
    case is discarded and re-rated after cooldown.  This keeps the die from
    reaching 90C.
    """

    def __init__(self, logger, case_id: str, sample_path: Path,
                 interval: int = GPU_MONITOR_INTERVAL_SECONDS,
                 abort_c: int = GPU_ABORT_C,
                 resource_path: Path | None = None):
        super().__init__(daemon=True, name=f"thermal-{case_id[:24]}")
        self.logger = logger
        self.case_id = case_id
        self.sample_path = sample_path
        self.resource_path = resource_path
        self.interval = interval
        self.abort_c = abort_c
        self.stop_event = threading.Event()
        self.abort_event = threading.Event()
        self.last_temp_c: float | None = None
        self.max_temp_c: float | None = None
        self.abort_temp_c: float | None = None

    def run(self) -> None:
        while not self.stop_event.is_set():
            sample = gpu_sample()
            temp = sample.get("temp_c") if sample else None
            if sample is not None and self.resource_path is not None:
                try:
                    _append_resource_sample(self.resource_path, self.case_id, sample)
                except Exception:  # logging must never kill the monitor
                    pass
            # An unsupported temperature field means no thermal verdict either
            # way: keep sampling, do not abort.
            if temp is not None:
                self.last_temp_c = temp
                if self.max_temp_c is None or temp > self.max_temp_c:
                    self.max_temp_c = temp
                try:
                    _append_gpu_sample(self.sample_path, self.case_id, temp)
                except Exception:  # logging must never kill the monitor
                    pass
                if temp >= self.abort_c:
                    self.abort_temp_c = temp
                    self.abort_event.set()
                    self.logger.warning(
                        "GPU %s >= %dC during %s — aborting in-flight case and unloading.",
                        _fmt_temp(temp), self.abort_c, self.case_id)
                    unload_model(self.logger)
                    return
            if self.stop_event.wait(self.interval):
                return

    def stop(self) -> None:
        self.stop_event.set()



def _finbert_vote(score: float) -> str:
    if pd.isna(score):
        return "FLAT"
    if score > 0.2:
        return "UP"
    if score < -0.2:
        return "DOWN"
    return "FLAT"


def build_pool(seed: int = 42) -> pd.DataFrame:
    """Sample Pool C case_ids from pre-holdout announcements + labels."""
    settings = load_settings()
    paths = settings.paths
    calendar = pd.DatetimeIndex(pd.read_parquet(
        paths.root / "data" / "reference" / "trading_calendar.parquet").iloc[:, 0])
    try:
        sector_map = json.loads((paths.root / "data" / "reference" / "sector_map.json").read_text())
    except Exception:
        sector_map = {}
    rows = []
    for symbol in settings.symbols:
        safe = symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        ann_path = paths.data_raw / "announcements" / f"{safe}.parquet"
        ds_path = paths.data_processed / f"{safe}_dataset.parquet"
        if not ann_path.exists() or not ds_path.exists():
            continue
        ann = pd.read_parquet(ann_path)
        if ann.empty or "published_utc" not in ann.columns:
            continue
        ds = pd.read_parquet(ds_path)
        if "target" not in ds.columns:
            continue
        labels = ds["target"].dropna().astype(int)
        label_by_date = {pd.Timestamp(t).normalize(): int(v) for t, v in labels.items()}
        dd = decision_dates(ann, calendar, col="published_utc")
        ann = ann.copy()
        ann["decision_date"] = pd.to_datetime(dd).dt.tz_localize(None).dt.normalize()
        ann = ann.dropna(subset=["decision_date"])
        ann = ann[ann["decision_date"] < pd.Timestamp(STAR_CUTOFF)]
        if ann.empty:
            continue
        price = ds[["close", "volume"]].copy() if {"close", "volume"} <= set(ds.columns) else None
        for idx, row in ann.iterrows():
            d = pd.Timestamp(row["decision_date"]).normalize()
            if d not in label_by_date:
                continue
            label = label_by_date[d]
            trend, volz = float("nan"), float("nan")
            if price is not None and d in price.index:
                pos = price.index.get_loc(d)
                if isinstance(pos, slice):
                    pos = int(np.argmax(price.index[pos] == d)) + 0
                try:
                    loc = price.index.get_indexer([d])[0]
                    if loc >= 6:
                        c0 = float(price["close"].iloc[loc - 5])
                        c1 = float(price["close"].iloc[loc])
                        trend = (c1 / c0 - 1.0) if c0 else float("nan")
                    if loc >= 21:
                        v = price["volume"].iloc[loc - 20:loc]
                        volz = float((price["volume"].iloc[loc] - v.mean()) / (v.std() or np.nan))
                except Exception:
                    pass
            text = (str(row.get("subject") or "") + ". " + str(row.get("text") or "")).strip(". ")
            fin = pd.to_numeric(pd.Series([row.get("finbert_score")]), errors="coerce").iloc[0]
            sector = ""
            for sec, members in (sector_map.items() if isinstance(sector_map, dict) else []):
                if symbol in (members or []):
                    sector = str(sec)
                    break
            key = f"{symbol}|{d.date()}|{hashlib.md5(text[:200].encode(errors='ignore')).hexdigest()[:8]}"
            rows.append({
                "case_id": key, "symbol": symbol, "sector": sector,
                "decision_date": d, "published_utc": str(row.get("published_utc")),
                "text": text, "finbert_score": float(fin) if pd.notna(fin) else float("nan"),
                "finbert_vote": _finbert_vote(fin),
                "trend_5d": trend, "volume_z": volz,
                "historical_label": LABEL_TO_VOTE[label],
            })
    pool = pd.DataFrame(rows).drop_duplicates(subset=["case_id"])
    rng = np.random.default_rng(seed)
    parts = []
    for lab, n in (("DOWN", N_DOWN), ("FLAT", N_FLAT), ("UP", N_UP)):
        sub = pool[pool["historical_label"] == lab]
        take = min(n, len(sub))
        parts.append(sub.sample(take, random_state=seed) if take else sub.iloc[0:0])
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out["pool"] = "C"
    return out


def run_gate(limit: int | None = None) -> dict:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(Q0_MAIN_LOG)
    ledger_path = Q0_LEDGER
    pool_cache = Q0_POOL
    assert STAR_CUTOFF < HOLDOUT_START, "STAR cutoff must precede holdout"
    if pool_cache.exists():
        pool = pd.read_parquet(pool_cache)
    else:
        pool = build_pool()
        atomic_write_parquet(pool, pool_cache)
        logger.info("Pool C built: %d cases.", len(pool))
    done: set[str] = set()
    existing = None
    if ledger_path.exists():
        existing = pd.read_parquet(ledger_path)
        done = set(existing["case_id"].astype(str))
    todo = pool[~pool["case_id"].astype(str).isin(done)]
    if limit is not None:
        todo = todo.head(limit)
    logger.info("Q0-D1: %d todo (%d done).", len(todo), len(done))
    trace_dir = Q0_TRACES
    sample_path = Q0_GPU
    resource_path = Q0_RESOURCE
    history_path = Q0_HISTORY
    heartbeat_path = Q0_HEARTBEAT
    state = ThermalState()
    run_started_at = datetime.now(timezone.utc).isoformat()
    attempted = 0
    total_model_seconds = 0.0
    last_checkpoint_at: str | None = None
    last_case_id: str | None = None

    def hb(**fields) -> None:
        """Write the unattended-run heartbeat, enriched with live thermal state."""
        ledger_total = int(len(existing)) if existing is not None else 0
        stats = _pool_stats(existing if existing is not None else pd.DataFrame())
        avg_seconds = (total_model_seconds / attempted) if attempted else 0.0
        remaining = max(0, int(len(pool)) - int(stats["rated_total"]))
        eta_hours = (remaining * avg_seconds / 3600.0) if avg_seconds > 0 else None
        base = {
            "run_started_at": run_started_at,
            "pool_total": int(len(pool)),
            "ledger_total": ledger_total,
            "rated_total": stats["rated_total"],
            "judged_total": stats["judged_total"],
            "harness_fail_rate": stats["harness_fail_rate"],
            "complete": bool(stats["rated_total"] >= len(pool)),
            "attempted": int(attempted),
            "last_case_id": last_case_id,
            "last_checkpoint_at": last_checkpoint_at,
            "total_model_seconds": round(total_model_seconds, 1),
            "avg_case_seconds": round(avg_seconds, 2),
            "eta_hours": round(eta_hours, 2) if eta_hours is not None else None,
            "work_seconds": round(state.work_seconds, 1),
            "proactive_target_seconds": PROACTIVE_WORK_MINUTES * 60,
            "thermal_state": state.last_state,
            "max_temperature_c": state.max_temp_c,
            "rests": int(state.rests),
            "hard_trips": int(state.hard_trips),
            "cooldowns": int(state.cooldowns),
            "proactive_breaks": int(state.proactive_breaks),
            "alarms": int(state.alarms),
            "prepares": int(state.prepares),
            "aborts": int(state.aborts),
            "rest_minutes": float(state.rest_minutes),
        }
        base.update(fields)
        _write_heartbeat(heartbeat_path, **base)
        # The heartbeat is a snapshot; the history is what the timeline plots.
        _append_history(history_path, {"timestamp": datetime.now(timezone.utc).isoformat(), **base})

    hb(phase="starting", current_case=None)

    for _, row in todo.iterrows():
        case_id = str(row["case_id"])
        # Blocking guard: may rest (20/30 min), then this same case proceeds.
        start_temp = thermal_gate_before_case(logger, state, case_id)
        hb(phase="running", current_case=case_id, temperature_c=start_temp)

        ctx = (f"5-day trend: {row['trend_5d']:.2%}, volume shock (z): {row['volume_z']:.2f}, "
               f"FinBERT prior: {row['finbert_score']:.2f}" if pd.notna(row["finbert_score"])
               else f"5-day trend: {row['trend_5d']:.2%}, NO_NEWS FinBERT prior")
        prompt, _ = build_prompt(row["symbol"], row["sector"] or "unknown", ctx, row["text"])

        monitor = ThermalMonitor(logger, case_id, sample_path, resource_path=resource_path)
        monitor.start()
        try:
            result = rate_case(row["symbol"], row["sector"] or "unknown", ctx,
                               row["text"], abort_event=monitor.abort_event)
        finally:
            monitor.stop()
            monitor.join(timeout=40)
        if monitor.max_temp_c is not None:
            _track_temp(state, monitor.max_temp_c)
        end_temp = monitor.last_temp_c if monitor.last_temp_c is not None else gpu_temp_c()
        _track_temp(state, end_temp)
        # Actual-work accounting: the 15-min break is earned by GPU seconds,
        # not by wall-clock time spent idle between cases.
        attempted += 1
        total_model_seconds += max(0.0, float(result.seconds or 0.0))
        state.work_seconds += max(0.0, float(result.seconds or 0.0))

        if result.thermal_abort or monitor.abort_event.is_set():
            state.aborts += 1
            trip_temp = monitor.abort_temp_c or end_temp
            result.thermal_abort = True
            result.thermal_temp_c = trip_temp
            result.fail_reason = result.fail_reason or "thermal-abort"
            write_trace(trace_dir, re.sub(r"[^A-Za-z0-9_.-]", "_", case_id), prompt, result)
            logger.warning(
                "Thermal abort on %s at %s — case discarded (not in ledger), will retry after rest.",
                case_id, _fmt_temp(trip_temp))
            last_case_id = case_id
            hb(phase="thermal_abort", current_case=case_id,
               temperature_c=trip_temp, last_case_seconds=round(result.seconds, 2),
               last_error="thermal-abort")
            handle_hard_trip(logger, state, trip_temp, case_id, "mid-case")
            continue

        trace_path = write_trace(trace_dir, re.sub(r"[^A-Za-z0-9_.-]", "_", case_id), prompt, result)
        new_row = {
            "case_id": row["case_id"], "symbol": row["symbol"],
            "decision_date": pd.Timestamp(row["decision_date"]).date().isoformat(),
            "pool": "C", "model_stage": "zero_shot",
            "prompt_version": "v1-skeleton",
            "qwen_vote": result.vote, "qwen_sent": result.sentiment,
            "qwen_conf": result.confidence, "qwen_trace_len": result.think_len,
            "finbert_score": row["finbert_score"],
            "finbert_qwen_disagree": (abs(float(row["finbert_score"]) - float(result.sentiment))
                                      if pd.notna(row["finbert_score"]) and result.sentiment is not None else float("nan")),
            "finbert_vote": row["finbert_vote"],
            "historical_label": row["historical_label"],
            "vote_matches_label": (result.vote == row["historical_label"]) if result.vote else False,
            "finbert_matches_label": (row["finbert_vote"] == row["historical_label"]),
            "kept_for_training": False, "drop_reason": "q0-gate-only",
            "truncated": result.truncated, "no_news": result.no_news,
            "harness_fail": result.harness_fail, "trace_path": str(trace_path),
            "gpu_temp_start": start_temp, "gpu_temp_end": end_temp,
            "thermal_state": state.last_state,
        }
        part = pd.DataFrame([new_row])
        existing = pd.concat([existing, part], ignore_index=True) if existing is not None else part
        # Per-case durable checkpoint: stopping at 87C never loses a completed case.
        atomic_write_parquet(existing, ledger_path)
        last_case_id = case_id
        last_checkpoint_at = datetime.now(timezone.utc).isoformat()
        logger.info("Q0-D1 checkpoint: %d rated.", len(existing))
        hb(phase="checkpoint", current_case=None, checkpoint_case=case_id,
           temperature_c=end_temp, last_case_seconds=round(result.seconds, 2),
           last_vote=result.vote, last_harness_fail=result.harness_fail,
           last_error=(result.fail_reason or None) if result.harness_fail else None)

    ledger = existing if existing is not None else pd.DataFrame()
    judged = pd.DataFrame()
    if len(ledger) and "pool" in ledger.columns:
        judged = ledger[ledger["pool"] == "C"]
        judged = judged[judged["qwen_vote"].isin(["UP", "DOWN", "FLAT"]) & ~judged["harness_fail"]]
    n = len(judged)
    p_q = float(judged["vote_matches_label"].mean()) if n else 0.0
    p_f = float(judged["finbert_matches_label"].mean()) if n else 0.0
    se = float(np.sqrt(p_q * (1 - p_q) / n + p_f * (1 - p_f) / n)) if n else float("nan")
    delta = p_q - p_f
    stats = _pool_stats(ledger)
    rated_total = int(stats["rated_total"])
    harness_fail_rate = float(stats["harness_fail_rate"])
    verdict = {
        "n": int(n), "qwen_agreement": p_q, "finbert_agreement": p_f,
        "delta": delta, "se_diff": se, "bar_2se": 2 * se if n else None,
        "pass": bool(n and delta > 2 * se),
        "harness_fail_rate": harness_fail_rate,
        "rated_total": rated_total,
        "ledger_total": int(len(ledger)),
        "judged_total": int(stats["judged_total"]),
        "pool_total": int(len(pool)),
        "complete": bool(rated_total >= len(pool)),
        "thermal_rests": int(state.rests),
        "thermal": {
            "hard_trips": int(state.hard_trips),
            "rests": int(state.rests),
            "cooldowns": int(state.cooldowns),
            "proactive_breaks": int(state.proactive_breaks),
            "alarms": int(state.alarms),
            "prepares": int(state.prepares),
            "aborts": int(state.aborts),
            "max_temp_c": state.max_temp_c,
            "rest_minutes": float(state.rest_minutes),
        },
    }
    atomic_write_json(Q0_VERDICT_FILE, verdict)
    logger.info("Q0-D1 verdict: %s", verdict)
    hb(phase="verdict", current_case=None, complete=verdict["complete"],
       n=int(n), qwen_agreement=p_q, finbert_agreement=p_f, delta=delta,
       se_diff=se if n else None, last_error=None,
       **{"pass": verdict["pass"]})
    if verdict["complete"]:
        status = "complete" if verdict["pass"] else "stopped"
    else:
        status = "partial"
    update_state(Q0_STATE_FILE.parent, "q0_d1", status=status,
                 verdict=str(Q0_VERDICT_FILE))
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description="Q0-D1 zero-shot gate.")
    parser.add_argument("--build-pool", action="store_true")
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    if args.build_pool:
        pool = build_pool()
        atomic_write_parquet(pool, settings.paths.root / "data" / "llm" / "pool_c.parquet")
        print(f"Pool C: {len(pool)} cases "
              f"({(pool['historical_label'] == 'DOWN').sum()}/{(pool['historical_label'] == 'FLAT').sum()}/{(pool['historical_label'] == 'UP').sum()})")
        return
    if args.pilot:
        print(run_gate(limit=args.pilot))
        return
    if args.full:
        print(run_gate())
        return
    parser.print_help()


if __name__ == "__main__":
    main()
