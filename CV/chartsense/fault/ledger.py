"""Append-only JSONL ledger — the source of truth for resumable generation.

A labels.csv is a derived artifact and is rebuilt from the ledger. A truncated
or half-written CSV can therefore never destroy resume state.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .atomic import atomic_save_csv


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self, path: Path | str, key_field: str = "key"):
        self.path = Path(path)
        self.key_field = key_field
        self._records: dict[str, dict] = {}
        self._bad_lines = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Typically the tail of an interrupted append; skip it.
                    self._bad_lines += 1
                    continue
                if not isinstance(record, dict):
                    self._bad_lines += 1
                    continue
                key = record.get(self.key_field)
                if key is not None:
                    self._records[str(key)] = record  # last entry wins

    @property
    def bad_lines(self) -> int:
        return self._bad_lines

    def is_done(self, key: str) -> bool:
        record = self._records.get(str(key))
        return bool(record and record.get("status") == "done")

    def attempts(self, key: str) -> int:
        record = self._records.get(str(key))
        return int(record.get("attempts", 0)) if record else 0

    def get(self, key: str) -> dict | None:
        record = self._records.get(str(key))
        return dict(record) if record else None

    def record(self, record: dict) -> None:
        """Append one record with flush + fsync, then update the in-memory view."""
        if self.key_field not in record:
            raise ValueError(f"ledger record missing {self.key_field!r}")
        record = dict(record)
        record.setdefault("updated_at", _now())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._records[str(record[self.key_field])] = record

    def mark_done(self, key: str, **extra: Any) -> None:
        self.record({self.key_field: key, "status": "done", "attempts": self.attempts(key) + 1, **extra})

    def mark_failed(self, key: str, error: str, **extra: Any) -> None:
        self.record(
            {
                self.key_field: key,
                "status": "failed",
                "attempts": self.attempts(key) + 1,
                "error": str(error)[:500],
                **extra,
            }
        )

    def records(self, status: str | None = None) -> list[dict]:
        values = self._records.values()
        if status is not None:
            values = [r for r in values if r.get("status") == status]
        return [dict(r) for r in values]

    def done_keys(self) -> set[str]:
        return {key for key, record in self._records.items() if record.get("status") == "done"}

    def rebuild_csv(self, out_path: Path | str, fields: Iterable[str]) -> int:
        fields = list(fields)
        rows = [
            {field: record.get(field, "") for field in fields}
            for record in self._records.values()
            if record.get("status") == "done"
        ]
        import pandas as pd

        atomic_save_csv(pd.DataFrame(rows, columns=fields), out_path)
        return len(rows)


__all__ = ["Ledger"]
