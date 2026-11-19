"""Fault-tolerant file IO shared by all Track Q phases.

Two guarantees:
1. Writes are durable: data is flushed to OS and fsync'd to disk before the
   file is atomically renamed into place. A crash can never leave a half file.
2. Reads are defensive: a truncated/corrupt JSONL line is quarantined instead
   of crashing the whole resume.

This module lives under track_q/_common so it travels with the phase folders
when they are synced (git / Drive / Colab). It has no dependency on stockml.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


def atomic_write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_append_jsonl(path: Path | str, row: dict) -> None:
    """Append one JSON line durably (flush + fsync). O_APPEND keeps it race-safe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def read_jsonl_safe(path: Path | str, quarantine_dir: Path | str | None = None) -> list[dict]:
    """Read a JSONL file, quarantining malformed lines instead of crashing.

    Returns only the valid rows. A corrupt line is moved to ``quarantine_dir``
    so the evidence is kept, not silently deleted.
    """
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    bad: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    bad.append(line)
            except json.JSONDecodeError:
                bad.append(line)
    if bad:
        qdir = Path(quarantine_dir) if quarantine_dir else path.parent / "quarantine"
        qdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        good_path = path.with_suffix(path.suffix + ".repaired")
        with good_path.open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        (qdir / f"{path.name}.{stamp}.bad").write_text("\n".join(bad) + "\n", encoding="utf-8")
        os.replace(good_path, path)
    return rows


def file_fingerprint(paths: Iterable[Path | str]) -> str:
    """Stable short fingerprint of a set of files (content hash)."""
    import hashlib
    h = hashlib.sha256()
    for p in sorted(Path(p) for p in paths):
        h.update(str(p.name).encode("utf-8"))
        if p.exists():
            h.update(p.read_bytes())
        else:
            h.update(b"<missing>")
    return h.hexdigest()[:16]


def dict_fingerprint(payload: dict) -> str:
    import hashlib
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def write_text_atomic(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
