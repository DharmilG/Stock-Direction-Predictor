from __future__ import annotations

import json
from datetime import date
from pathlib import Path


class UniverseProvider:
    """Loads a point-in-time universe when a constituent history file is supplied.

    Expected JSON shape:
    {
      "2024-01-02": ["RELIANCE.NS", "TCS.NS"],
      "2024-01-03": [...]
    }

    Without this file, configured symbols are treated as a static universe. That is suitable
    for a single instrument/index but is NOT a survivorship-bias-free historical NIFTY-500 study.
    """

    def __init__(self, path: Path, static_symbols: list[str]):
        self.path = path
        self.static_symbols = static_symbols
        self.history: dict[str, list[str]] = {}
        if path.exists():
            self.history = json.loads(path.read_text(encoding="utf-8"))

    @property
    def point_in_time_available(self) -> bool:
        return bool(self.history)

    def symbols_for_date(self, when: date) -> list[str]:
        key = when.isoformat()
        return self.history.get(key, self.static_symbols)
