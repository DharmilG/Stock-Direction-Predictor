from __future__ import annotations

"""Phase 5 leakage scanner (Appendix B v1).

CODE READY — UNTESTED
"""

import re
from datetime import date, datetime
from typing import Any

PATTERNS: dict[str, list[str]] = {
    "post_decision_date": [
        r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b",
        r"\b(next|following) (week|day|days)\b",
        r"\bfive days later\b",
        r"\bafter the event\b",
    ],
    "outcome_mentions": [
        r"\btriple.?barrier\b",
        r"\bupper barrier (was )?hit\b",
        r"\blower barrier (was )?hit\b",
        r"\btouched the (ceiling|floor)\b",
        r"\blabel (is|was) (UP|DOWN|FLAT)\b",
    ],
    "future_prices_returns": [
        r"\b(rose|fell|gained|lost|closed at|settled at)\s+\d+(?:\.\d+)?\s*(?:%|percent|rs|inr)\b",
        r"\b5-day return\b",
        r"\brealized return\b",
        r"\bevent_end\b",
    ],
    "self_incrimination": [
        r"\bi have seen (future|post|subsequent)\b",
        r"\bas an ai\b.*\bknowledge cutoff\b",
        r"\bbased on what happened (after|next)\b",
    ],
}

_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")


def _parse_decision_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd_ts(value).date()
    except Exception:
        return None


def pd_ts(value: Any):
    """Small local helper to avoid importing pandas at module import time."""
    import pandas as pd

    return pd.Timestamp(value)


def scan_text(text: str, decision_date: Any) -> list[dict[str, str]]:
    """Return scanner hits for one trace/article text."""
    text = text or ""
    dd = _parse_decision_date(decision_date)
    hits: list[dict[str, str]] = []
    for cls, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                snippet = text[max(0, match.start() - 60) : match.end() + 60].replace("\n", " ")
                if cls == "post_decision_date" and dd is not None:
                    iso = _ISO_DATE_RE.search(match.group(0))
                    if iso:
                        try:
                            hit_date = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
                            if hit_date <= dd:
                                continue
                        except ValueError:
                            pass
                hits.append(
                    {
                        "class": cls,
                        "pattern": pattern,
                        "snippet": snippet.strip(),
                        "start": str(match.start()),
                        "end": str(match.end()),
                    }
                )
    return hits


def scan_candidate(row: dict[str, Any]) -> list[dict[str, str]]:
    """Scan a STaR candidate row (thinking/content only)."""
    decision_date = row.get("decision_date")
    text = "\n".join(
        str(row.get(field) or "")
        for field in ("thinking", "content", "trace")
        if field in row
    )
    return scan_text(text, decision_date)


def scanner_version() -> str:
    """Bump only when the pattern list changes (never shrink without a commit)."""
    return "appendix-b-v1"
