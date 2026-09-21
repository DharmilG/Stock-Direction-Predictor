from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import atomic_write_json, read_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_path(root: Path, step: str) -> Path:
    return root / f"{step}.json"


def get_state(root: Path, step: str) -> dict[str, Any]:
    return read_json(state_path(root, step), default={}) or {}


def update_state(root: Path, step: str, **kwargs: Any) -> dict[str, Any]:
    current = get_state(root, step)
    current.update(kwargs)
    current["updated_at"] = utc_now()
    atomic_write_json(state_path(root, step), current)
    return current
