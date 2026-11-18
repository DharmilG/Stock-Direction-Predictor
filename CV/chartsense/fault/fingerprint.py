"""Fingerprints used to refuse unsafe resumes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def file_fingerprint(paths: Iterable[Path | str]) -> str:
    digest = hashlib.sha256()
    for raw in sorted(paths, key=lambda p: str(p)):
        path = Path(raw)
        digest.update(str(path.name).encode("utf-8"))
        if not path.exists():
            digest.update(b"|missing")
            continue
        stat = path.stat()
        digest.update(f"|{stat.st_mtime_ns}|{stat.st_size}".encode("ascii"))
    return digest.hexdigest()


def dict_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def code_fingerprint(root: Path | str) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class FingerprintMismatch(RuntimeError):
    """Raised when on-disk state was produced by a different dataset or code."""


def require_match(expected: str | None, actual: str | None, what: str) -> None:
    if expected and actual and expected != actual:
        raise FingerprintMismatch(
            f"{what} fingerprint changed ({expected[:12]} -> {actual[:12]}). "
            "Refusing to resume. Use --force-fresh to start over."
        )


__all__ = [
    "file_fingerprint",
    "dict_fingerprint",
    "code_fingerprint",
    "FingerprintMismatch",
    "require_match",
]
