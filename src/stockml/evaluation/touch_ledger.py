from __future__ import annotations

"""Test-touch ledger (Track I §0.1 governance).

Every event that reads the sealed test window's true labels — training,
evaluation, or a read-only slice join — appends one line to
results/holdout_touches.jsonl: date, track, phase (D1|ablation),
description, rows_touched, artifact_ref. D1 touches are lighter than
retrains but never zero; the count feeds PBO.

Backfill covers the 8 pre-ledger campaign touches from existing reports.
"""

import datetime as _dt
import json

LEDGER = "holdout_touches.jsonl"
N_TEST = 37292

# Reconstructed from reports/logs. Order oldest → newest.
BACKFILL = [
    {"date": "2026-09-15", "track": "G1", "phase": "ablation", "description": "Week-2 25-trial train + evaluate", "rows_touched": N_TEST, "artifact_ref": "artifacts/g1_baseline_freeze.json", "eligible": True},
    {"date": "2026-09-15", "track": "G1-R", "phase": "ablation", "description": "horizon-10 relabel + retrain + evaluate", "rows_touched": N_TEST, "artifact_ref": "logs/train.log", "eligible": True},
    {"date": "2026-09-16", "track": "G1-R", "phase": "D1", "description": "threshold sweep on frozen test predictions", "rows_touched": N_TEST, "artifact_ref": "results/confidence_accuracy.csv", "eligible": True},
    {"date": "2026-09-16", "track": "G1-R", "phase": "ablation", "description": "horizon-10 revert retrain + evaluate", "rows_touched": N_TEST, "artifact_ref": "logs/train.log", "eligible": True},
    {"date": "2026-09-18", "track": "G1", "phase": "ablation", "description": "attempt-2 features + retrain + evaluate", "rows_touched": N_TEST, "artifact_ref": "results/metrics.json", "eligible": True},
    {"date": "2026-09-19", "track": "H1", "phase": "ablation", "description": "delivery×turnover interaction + retrain", "rows_touched": N_TEST, "artifact_ref": "results/metrics.json", "eligible": True},
    {"date": "2026-09-19", "track": "H1b", "phase": "ablation", "description": "CLV×delivery + retrain (memorization flag)", "rows_touched": N_TEST, "artifact_ref": "results/metrics.json", "eligible": False, "note": "memorization flag"},
    {"date": "2026-09-19", "track": "Issue-1", "phase": "D1", "description": "up-autopsy slice joins on frozen predictions", "rows_touched": N_TEST, "artifact_ref": "results/up_autopsy.json", "eligible": True},
    {"date": "2026-09-19", "track": "F", "phase": "D1", "description": "tau_flat swept on calibration, judged once on test", "rows_touched": N_TEST, "artifact_ref": "results/flat_threshold.json", "eligible": True},
    {"date": "2026-09-19", "track": "U", "phase": "ablation", "description": "announcement counts + retrain", "rows_touched": N_TEST, "artifact_ref": "results/metrics.json", "eligible": True},
    {"date": "2026-09-19", "track": "U2", "phase": "D1", "description": "scope-activity slice join (u2_d1.json)", "rows_touched": N_TEST, "artifact_ref": "results/u2_d1.json", "eligible": True},
    {"date": "2026-09-19", "track": "U3", "phase": "D1", "description": "sentiment slice join (u3_d1.json)", "rows_touched": 15216, "artifact_ref": "results/u3_d1.json", "eligible": True},
]


def _read(path):
    if not path.exists():
        return []
    try:
        return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return []


def backfill(paths) -> list[dict]:
    """Ensure all BACKFILL touches exist; never delete logged touches.

    Legacy free-form stage entries from the pre-ledger file are dropped once
    (they overlap BACKFILL's coverage); everything with the full schema —
    including future log_touch() rows — is preserved. Idempotent.
    """
    path = paths.results / LEDGER
    existing = _read(path)
    keep = [e for e in existing
            if all(k in e for k in ("track", "phase", "description", "artifact_ref"))
            and not str(e.get("stage", "")).startswith("G3")]
    have = {(e.get("track"), e.get("phase"), e.get("description")) for e in keep}
    merged = list(keep)
    for b in BACKFILL:
        if (b["track"], b["phase"], b["description"]) not in have:
            merged.append(dict(b))
    # G3 economics touches keep their original detailed rows.
    for e in existing:
        if str(e.get("stage", "")).startswith("G3"):
            merged.append(e)
    # Normalize: every entry carries the full schema.
    for e in merged:
        e.setdefault("track", e.get("stage", "unknown"))
        e.setdefault("phase", "ablation")
        e.setdefault("description", e.get("stage", ""))
        e.setdefault("rows_touched", N_TEST)
        e.setdefault("artifact_ref", "")
        e.setdefault("eligible", True)
    path.write_text("\n".join(json.dumps(e) for e in merged) + "\n", encoding="utf-8")
    return merged


def log_touch(paths, track: str, phase: str, description: str,
              rows_touched: int, artifact_ref: str, eligible: bool = True) -> int:
    """Append one touch. Returns eligible D1+ablation count for PBO sizing."""
    entries = _read(paths.results / LEDGER)
    entries.append({"date": _dt.date.today().isoformat(), "track": track, "phase": phase,
                    "description": description, "rows_touched": int(rows_touched),
                    "artifact_ref": artifact_ref, "eligible": bool(eligible)})
    (paths.results / LEDGER).write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return sum(1 for e in entries if e.get("eligible"))
