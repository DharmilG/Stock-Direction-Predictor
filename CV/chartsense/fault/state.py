"""Resumable progress state, written atomically after every unit of work."""

from __future__ import annotations

from typing import Any

from stockml.utils.io import read_json

from ..config import STATE_DIR
from .atomic import atomic_write_json


def state_path(name: str):
    return STATE_DIR / f"{name}.json"


def read_state(name: str, default: Any = None) -> dict:
    payload = read_json(state_path(name), default=None)
    if isinstance(payload, dict):
        return payload
    return dict(default) if isinstance(default, dict) else {}


def write_state(name: str, payload: dict) -> None:
    atomic_write_json(state_path(name), payload)


def update_state(name: str, **changes: Any) -> dict:
    current = read_state(name, {})
    current.update(changes)
    write_state(name, current)
    return current


def initial_state(name: str, **defaults: Any) -> dict:
    current = read_state(name)
    if not current:
        current = dict(defaults)
        current.setdefault("status", "initialized")
        write_state(name, current)
    return current


__all__ = ["state_path", "read_state", "write_state", "update_state", "initial_state"]
