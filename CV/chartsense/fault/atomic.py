"""Atomic, fsync-backed writes.

Mirrors src/stockml/utils/io.py: write to a temp file in the same directory,
flush, fsync, then os.replace. A crash at any point leaves either the old file
or the new file, never a half-written one.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from stockml.utils.io import atomic_write_json, atomic_write_text


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_save_torch(obj: Any, path: Path | str) -> None:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    atomic_write_bytes(path, buffer.getvalue())


def atomic_save_image(image: Any, path: Path | str, fmt: str = "PNG") -> None:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    atomic_write_bytes(path, buffer.getvalue())


def atomic_save_csv(df: Any, path: Path | str) -> None:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    atomic_write_text(Path(path), buffer.getvalue())


def sweep_stale_temps(
    root: Path | str,
    max_age_seconds: int = 3600,
    quarantine_dir: Path | str | None = None,
) -> list[tuple[str, str]]:
    """Move orphaned temp/part files to quarantine. Returns (source, dest) pairs."""
    now = time.time()
    moved: list[tuple[str, str]] = []
    for candidate in Path(root).rglob("*"):
        if not candidate.is_file():
            continue
        if ".tmp" not in candidate.name and not candidate.name.endswith(".part"):
            continue
        try:
            age = now - candidate.stat().st_mtime
        except OSError:
            continue
        if age < max_age_seconds:
            continue
        dest_dir = Path(quarantine_dir) if quarantine_dir else candidate.parent
        dest = dest_dir / f"{candidate.name}.stale-{int(now)}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(candidate, dest)
            moved.append((str(candidate), str(dest)))
        except OSError:
            continue
    return moved


__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "atomic_save_torch",
    "atomic_save_image",
    "atomic_save_csv",
    "sweep_stale_temps",
]
