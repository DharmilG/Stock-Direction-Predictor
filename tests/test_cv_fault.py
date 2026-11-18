"""Fault-tolerance primitives for the CV lane. No data or training required."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from CV.chartsense import fault


def test_atomic_write_bytes_leaves_no_temp(tmp_path):
    target = tmp_path / "nested" / "file.bin"
    fault.atomic_write_bytes(target, b"hello")
    assert target.read_bytes() == b"hello"
    assert list(target.parent.glob("*.tmp")) == []


def test_atomic_write_json_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(fault.state, "STATE_DIR", tmp_path)
    fault.write_state("demo", {"status": "running", "n": 1})
    assert fault.read_state("demo")["n"] == 1
    updated = fault.update_state("demo", status="done")
    assert updated == {"status": "done", "n": 1}


def test_ledger_skips_truncated_tail_and_rebuilds_csv(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        json.dumps({"key": "a", "status": "done", "filename": "a.png", "direction": "UP"}) + "\n"
        + json.dumps({"key": "b", "status": "failed", "error": "boom"}) + "\n"
        + '{"key": "c", "status": "do',
        encoding="utf-8",
    )
    ledger = fault.Ledger(path)
    assert ledger.bad_lines == 1
    assert ledger.is_done("a")
    assert not ledger.is_done("b")
    assert ledger.done_keys() == {"a"}

    ledger.mark_done("c", filename="c.png", direction="DOWN")
    assert ledger.is_done("c")
    assert ledger.attempts("c") == 1
    csv_path = tmp_path / "labels.csv"
    assert ledger.rebuild_csv(csv_path, ["filename", "direction"]) == 2
    assert "c.png" in csv_path.read_text(encoding="utf-8")


def test_lock_recovers_stale_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(fault.lock, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(fault.lock, "QUARANTINE_DIR", tmp_path / "quarantine")
    stale = tmp_path / "demo.lock"
    stale.write_text(
        json.dumps(
            {
                "pid": 99999999,
                "host": "ghost",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "heartbeat": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    with fault.Lock("demo") as acquired:
        assert acquired.path.exists()
    assert not stale.exists()


def test_lock_rejects_live_process(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(fault.lock, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(fault.lock, "QUARANTINE_DIR", tmp_path / "quarantine")
    live = tmp_path / "live.lock"
    live.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "host": "local",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "heartbeat": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(fault.LockBusy):
        fault.Lock("live").acquire()


def test_fingerprint_mismatch_refuses_resume():
    with pytest.raises(fault.FingerprintMismatch):
        fault.require_match("a" * 64, "b" * 64, "unit test")
    fault.require_match("a" * 64, "a" * 64, "unit test")


def test_graceful_stop_handler_and_restore():
    stop = fault.GracefulStop()
    stop.install()
    try:
        stop._handle(15, None)
        assert stop.requested
        with pytest.raises(KeyboardInterrupt):
            stop.check()
    finally:
        stop.restore()
