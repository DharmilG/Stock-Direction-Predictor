from __future__ import annotations

"""Track Q guard tests: holdout assertion + leakage scanner + harness validator.

No Ollama calls here — pure logic only, so these run on CPU in CI.
"""

import pandas as pd

from stockml.data.qwen_harness import (
    EMBARGO_DAYS,
    HOLDOUT_START,
    STAR_CUTOFF,
    validate_output,
)


def test_holdout_constants_are_pinned():
    from datetime import date

    assert HOLDOUT_START == date(2023, 8, 29)  # never 2023-12-20 stale run
    assert EMBARGO_DAYS == 5
    assert STAR_CUTOFF == date(2023, 8, 24)


def test_star_assertion_trips_on_contamination():
    from stockml.data.qwen_harness import STAR_CUTOFF

    clean = pd.DataFrame({"decision_date": pd.to_datetime(["2020-05-01", "2023-08-23"])})
    assert clean["decision_date"].max().date() < STAR_CUTOFF
    dirty = pd.DataFrame({"decision_date": pd.to_datetime(["2020-05-01", "2023-08-24"])})
    assert not (dirty["decision_date"].max().date() < STAR_CUTOFF)


def _leakage_hit(trace: str) -> bool:
    import re

    patterns = [
        r"triple.?barrier",
        r"upper barrier (was )?hit|lower barrier (was )?hit",
        r"label (is|was) (UP|DOWN|FLAT)",
        r"(rose|fell|gained|lost|closed at|settled at)\s+\d+(\.\d+)?\s*(%|percent|rs|inr)",
        r"i have seen (future|post|subsequent)",
    ]
    return any(re.search(p, trace, re.I) for p in patterns)


def test_scanner_trips_per_class():
    assert _leakage_hit("the upper barrier was hit on day 3")
    assert _leakage_hit("label is UP per the answer key")
    assert _leakage_hit("the stock rose 8% the following week")
    assert _leakage_hit("i have seen future prices")


def test_scanner_clean_same_day_no_trip():
    assert not _leakage_hit("published 2023-08-23, decision_date 2023-08-23: normal analysis")


def test_validator_accepts_good_output():
    thinking = "x" * 120 + " step 1 classify, step 2 base rate, step 3 positioning, step 4 commit"
    content = '{"sentiment": 0.7, "vote": "UP", "confidence": 0.8, "trace_steps": [1,2,3,4]}'
    parsed, reason = validate_output(content, thinking)
    assert reason == "" and parsed is not None and parsed["vote"] == "UP"


def test_thermal_constants():
    from stockml.data.q0_d1 import (
        GPU_ABORT_C,
        GPU_ALARM_C,
        GPU_COOLDOWN_MINUTES,
        GPU_PREPARE_C,
        GPU_REST_MINUTES,
        GPU_TRIPS_PER_WINDOW,
        GPU_WATCH_C,
    )

    assert (GPU_WATCH_C, GPU_PREPARE_C, GPU_ALARM_C, GPU_ABORT_C) == (80, 84, 85, 87)
    assert GPU_REST_MINUTES == 20
    assert GPU_COOLDOWN_MINUTES == 30
    assert GPU_TRIPS_PER_WINDOW == 4


def test_thermal_parse():
    from stockml.data.q0_d1 import _parse_temp

    assert _parse_temp("81\n") == 81.0
    assert _parse_temp("  85  \n") == 85.0
    assert _parse_temp("N/A\n") is None
    assert _parse_temp("") is None


def _quiet_logger():
    import logging

    return logging.getLogger("test-q0-thermal")


def test_hard_rest_and_fourth_trip_cooldown(monkeypatch):
    from stockml.data import q0_d1 as q

    sleeps: list[int] = []
    monkeypatch.setattr(q.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(q, "unload_model", lambda logger: True)
    monkeypatch.setattr(q, "gpu_temp_c", lambda: 55.0)
    state = q.ThermalState(work_seconds=123.0)

    for _ in range(4):
        q.handle_hard_trip(_quiet_logger(), state, 87.0, "case", "test")

    assert sleeps == [20 * 60, 20 * 60, 20 * 60, 30 * 60]
    assert state.hard_trips == 4
    assert state.cooldowns == 1
    assert state.trip_times == []
    assert state.work_seconds == 0.0  # every rest resets the actual-work clock


def test_hard_trip_window_expires(monkeypatch):
    from stockml.data import q0_d1 as q

    sleeps: list[int] = []
    monkeypatch.setattr(q.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(q, "unload_model", lambda logger: True)
    monkeypatch.setattr(q, "gpu_temp_c", lambda: 55.0)
    now = q.time.monotonic()
    state = q.ThermalState(trip_times=[now - 3601, now - 3500, now - 3400])

    q.handle_hard_trip(_quiet_logger(), state, 87.0, "case", "test")

    assert sleeps == [20 * 60]
    assert state.cooldowns == 0
    assert len(state.trip_times) == 3


def test_proactive_rest_after_60_min_actual_work(monkeypatch):
    from stockml.data import q0_d1 as q

    sleeps: list[int] = []
    monkeypatch.setattr(q.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(q, "unload_model", lambda logger: True)
    monkeypatch.setattr(q, "gpu_temp_c", lambda: 55.0)
    state = q.ThermalState(work_seconds=60 * 60)

    assert q.maybe_proactive_rest(_quiet_logger(), state) is True
    assert sleeps == [15 * 60]
    assert state.proactive_breaks == 1
    assert state.work_seconds == 0.0  # the break resets the actual-work clock


def test_no_proactive_rest_before_60_min_actual_work(monkeypatch):
    from stockml.data import q0_d1 as q

    sleeps: list[int] = []
    monkeypatch.setattr(q.time, "sleep", lambda seconds: sleeps.append(seconds))
    state = q.ThermalState(work_seconds=60 * 60 - 1)

    assert q.maybe_proactive_rest(_quiet_logger(), state) is False
    assert sleeps == []


def test_pool_stats_counts_judged_and_fail_rate():
    from stockml.data.q0_d1 import _pool_stats

    frame = pd.DataFrame({
        "pool": ["C", "C", "C", "C", "B"],
        "case_id": ["a", "b", "c", "d", "e"],
        "qwen_vote": ["UP", None, "DOWN", "FLAT", "UP"],
        "harness_fail": [False, True, False, False, False],
    })
    stats = _pool_stats(frame)
    assert stats["rated_total"] == 4
    assert stats["judged_total"] == 3  # a, c, d; b is a harness failure
    assert stats["harness_fail_rate"] == 1 / 4


def test_heartbeat_is_written_atomically(tmp_path):
    import json

    from stockml.data.q0_d1 import _write_heartbeat

    target = tmp_path / "q0_heartbeat.json"
    _write_heartbeat(target, phase="checkpoint", ledger_total=7)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["phase"] == "checkpoint"
    assert payload["ledger_total"] == 7
    assert payload["pid"] > 0
    assert "updated_at" in payload


def test_monitor_aborts_at_87(monkeypatch):
    from pathlib import Path

    from stockml.data import q0_d1 as q

    temps = iter([86.0, 87.0])
    monkeypatch.setattr(q, "gpu_temp_c", lambda: next(temps, 55.0))
    monkeypatch.setattr(q, "_append_gpu_sample", lambda *args: None)
    unloaded: list[int] = []
    monkeypatch.setattr(q, "unload_model", lambda logger: unloaded.append(1) or True)
    monitor = q.ThermalMonitor(_quiet_logger(), "case", Path("unused"), interval=0)
    monitor.start()
    monitor.join(timeout=2)

    assert monitor.abort_event.is_set()
    assert monitor.abort_temp_c == 87.0
    assert unloaded


def test_unload_model_command(monkeypatch):
    from stockml.data import q0_d1 as q

    calls = []

    class Result:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(q.shutil, "which", lambda name: "ollama")
    monkeypatch.setattr(q.subprocess, "run",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or Result())

    assert q.unload_model(_quiet_logger()) is True
    assert calls[0][0][0] == ["ollama", "stop", q.MODEL]


def test_gate_prepare_alarm_and_hard_rest(monkeypatch):
    from stockml.data import q0_d1 as q

    sleeps: list[int] = []
    temps = [84.0, 85.0, 87.0, 55.0]
    monkeypatch.setattr(q.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(q, "unload_model", lambda logger: True)
    monkeypatch.setattr(q, "gpu_temp_c",
                        lambda: temps.pop(0) if temps else 55.0)
    state = q.ThermalState()

    assert q.thermal_gate_before_case(_quiet_logger(), state, "c1") == 84.0
    assert state.last_state == "prepare"
    assert q.thermal_gate_before_case(_quiet_logger(), state, "c2") == 85.0
    assert state.last_state == "alarm"
    assert state.alarms == 1
    assert q.thermal_gate_before_case(_quiet_logger(), state, "c3") == 55.0
    assert state.last_state == "rest"
    assert state.hard_trips == 1
    assert sleeps == [20 * 60]


def test_soft_breather_when_rising(monkeypatch):
    from stockml.data import q0_d1 as q

    sleeps: list[int] = []
    monkeypatch.setattr(q.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(q, "gpu_temp_c", lambda: 82.0)
    state = q.ThermalState(last_temp_c=81.0)

    assert q.thermal_gate_before_case(_quiet_logger(), state, "c1") == 82.0
    assert state.last_state == "amber"
    assert sleeps == [30]


def test_validator_rejects_sign_mismatch_and_thin_thinking():
    ok_thinking = "y" * 120
    parsed, _ = validate_output(
        '{"sentiment": -0.6, "vote": "UP", "confidence": 0.5, "trace_steps": [1,2,3,4]}',
        ok_thinking,
    )
    assert parsed is None
    parsed, reason = validate_output(
        '{"sentiment": 0.5, "vote": "UP", "confidence": 0.5, "trace_steps": [1,2,3,4]}',
        "too short",
    )
    assert parsed is None and reason == "empty-thinking"
