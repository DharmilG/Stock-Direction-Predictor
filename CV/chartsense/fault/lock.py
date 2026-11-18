"""Process lock with pid liveness and heartbeat-based stale recovery."""

from __future__ import annotations

import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from ..config import LOCK_DIR, QUARANTINE_DIR


class LockBusy(RuntimeError):
    """Another live process holds this lock."""


class Lock:
    def __init__(self, name: str, stale_seconds: int = 600, heartbeat_seconds: int = 60):
        self.path = LOCK_DIR / f"{name}.lock"
        self.stale_seconds = stale_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self._acquired = False
        self._last_heartbeat = 0.0

    def _payload(self) -> dict:
        return {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "heartbeat": datetime.now(timezone.utc).isoformat(),
        }

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _is_stale(self, payload: dict) -> bool:
        pid = payload.get("pid")
        if not pid:
            return True
        try:
            if not psutil.pid_exists(int(pid)):
                return True
        except (TypeError, ValueError):
            return True
        stamp = payload.get("heartbeat", "")
        try:
            age = time.time() - datetime.fromisoformat(stamp).timestamp()
        except (TypeError, ValueError):
            return True
        return age > self.stale_seconds

    def _write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())

    def acquire(self) -> "Lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                payload = self._read()
                if not self._is_stale(payload):
                    raise LockBusy(
                        f"{self.path} is held by pid {payload.get('pid')} "
                        f"on {payload.get('host')}. Wait for it or stop that process."
                    )
                QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
                dest = QUARANTINE_DIR / f"{self.path.name}.stale-{int(time.time())}"
                try:
                    os.replace(self.path, dest)
                except OSError:
                    continue
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._payload(), handle)
                handle.flush()
                os.fsync(handle.fileno())
            self._acquired = True
            self._last_heartbeat = time.time()
            return self
        raise LockBusy(f"could not acquire {self.path} after recovering a stale lock")

    def heartbeat(self, force: bool = False) -> None:
        if not self._acquired:
            return
        now = time.time()
        if not force and now - self._last_heartbeat < self.heartbeat_seconds:
            return
        payload = self._payload()
        payload["started_at"] = self._read().get("started_at", payload["started_at"])
        self._write(payload)
        self._last_heartbeat = now

    def release(self) -> None:
        if self._acquired:
            try:
                self.path.unlink(missing_ok=True)
            finally:
                self._acquired = False

    def __enter__(self) -> "Lock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


__all__ = ["Lock", "LockBusy"]
