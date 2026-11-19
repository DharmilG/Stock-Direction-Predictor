"""Read-only artifact access for the research dashboard API.

Nothing in this module mutates state: the dashboard must never fetch data,
retrain, or rewrite predictions. It only resolves artifact paths inside the
project, caches parsed files by ``(mtime, size)`` so a finished training run or
data refresh is visible without restarting the server, and derives freshness
from *data dates* rather than file timestamps.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import Paths, Settings, load_settings
from ..qpipeline.paths import p0_from, PHASE_REGISTRY

# NSE timestamps are stored as naive IST; a session's data only exists after the
# exchange settles, so anything before the cutoff refers to the previous session.
IST = timezone(timedelta(hours=5, minutes=30), "IST")
SESSION_DATA_CUTOFF = time(18, 30)

# Cache is keyed per file and invalidated by mtime/size, never by wall clock.
_CACHE: dict[str, tuple[tuple[str, int, int], Any]] = {}
MAX_CACHE_ENTRIES = 1024

# Age past which a model is merely "old", not wrong. This is informational.
TRAIN_STALE_DAYS = 30
# A refresh heartbeat older than this means the run died or never started.
REFRESH_HEARTBEAT_STALE_SECONDS = 900

_TIME_COLUMNS = (
    "timestamp",
    "session",
    "published_utc",
    "decision_date",
    "event_start",
    "event_end",
    "date",
    "datetime",
)


# --------------------------------------------------------------------------- #
# Generic cached readers
# --------------------------------------------------------------------------- #
def _stat_key(path: Path) -> tuple[str, int, int]:
    try:
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(path), 0, 0)


def _cached(path: Path, loader: Callable[[Path], Any], namespace: str = "raw") -> Any:
    if not path.is_file():
        return None
    key = _stat_key(path)
    slot = f"{namespace}:{path}"
    hit = _CACHE.get(slot)
    if hit is not None and hit[0] == key:
        return hit[1]
    try:
        value = loader(path)
    except Exception:
        value = None
    if len(_CACHE) >= MAX_CACHE_ENTRIES:
        _CACHE.clear()
    _CACHE[slot] = (key, value)
    return value


def json_file(path: Path) -> dict[str, Any]:
    """Parsed JSON object, or ``{}`` when the artifact is missing/invalid."""
    value = _cached(path, lambda p: json.loads(p.read_text(encoding="utf-8")))
    return value if isinstance(value, dict) else {}


def frame_file(path: Path) -> pd.DataFrame | None:
    return _cached(path, lambda p: pd.read_parquet(p))


def csv_file(path: Path) -> pd.DataFrame | None:
    return _cached(path, lambda p: pd.read_csv(p))


def file_meta(path: Path) -> dict[str, Any]:
    """Existence + mtime, without parsing the payload."""
    try:
        stat = path.stat()
    except OSError:
        return {"available": False, "modified_at": None, "size_bytes": None}
    return {
        "available": True,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, IST).isoformat(),
        "size_bytes": int(stat.st_size),
    }


# --------------------------------------------------------------------------- #
# JSON safety + scalar coercion
# --------------------------------------------------------------------------- #
def json_safe(value: Any) -> Any:
    """Coerce numpy/pandas scalars and non-finite floats into strict JSON."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return to_iso(value)
    if isinstance(value, pd.Timedelta):
        return str(value)
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            return str(value)
    if isinstance(value, (pd.Series, pd.Index)):
        return [json_safe(v) for v in value.tolist()]
    return str(value)


def to_iso(value: Any) -> str | None:
    """Date-only artifacts render as YYYY-MM-DD; timestamps keep their clock."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    try:
        stamp = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(stamp):
        return None
    if stamp.hour == stamp.minute == stamp.second == 0 and stamp.microsecond == 0:
        return stamp.date().isoformat()
    return stamp.isoformat()


def to_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        stamp = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(stamp):
        return None
    return stamp.date()


def parse_utc(value: Any) -> datetime | None:
    """Parse an ISO timestamp into an aware datetime (naive values mean UTC)."""
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def age_seconds(value: Any, now: datetime | None = None) -> int | None:
    parsed = parse_utc(value)
    if parsed is None:
        return None
    reference = now or datetime.now(timezone.utc)
    return max(0, int((reference - parsed).total_seconds()))


# --------------------------------------------------------------------------- #
# Trading sessions
# --------------------------------------------------------------------------- #
def now_ist() -> datetime:
    return datetime.now(IST)


def _roll_back_to_weekday(value: date) -> date:
    while value.weekday() >= 5:
        value -= timedelta(days=1)
    return value


def latest_weekday_session(now: datetime | None = None) -> date:
    """Weekday estimate of the most recent session whose data can exist."""
    moment = (now or now_ist()).astimezone(IST)
    reference = moment.date() if moment.time() >= SESSION_DATA_CUTOFF else moment.date() - timedelta(days=1)
    return _roll_back_to_weekday(reference)


def calendar_summary(paths: Paths) -> dict[str, Any]:
    path = paths.root / "data" / "reference" / "trading_calendar.parquet"
    frame = frame_file(path)
    if frame is None or frame.empty or "session" not in frame.columns:
        return {"available": False, "sessions": 0, "first": None, "last": None, "path": _rel(paths, path)}
    sessions = pd.to_datetime(frame["session"], errors="coerce").dropna()
    if sessions.empty:
        return {"available": False, "sessions": 0, "first": None, "last": None, "path": _rel(paths, path)}
    return {
        "available": True,
        "sessions": int(len(sessions)),
        "first": sessions.min().date().isoformat(),
        "last": sessions.max().date().isoformat(),
        "path": _rel(paths, path),
    }


def expected_session(paths: Paths, now: datetime | None = None) -> dict[str, Any]:
    """The session the dataset *should* reach, and whether that is estimated.

    The stored calendar is itself a stale artifact, so when it lags we fall back
    to a weekday estimate and say so instead of pretending to know NSE holidays.
    """
    calendar = calendar_summary(paths)
    estimate = latest_weekday_session(now)
    calendar_last = to_date(calendar.get("last"))
    if calendar_last is not None and calendar_last >= estimate:
        return {"session": calendar_last.isoformat(), "estimated": False, "calendar_last": calendar["last"]}
    return {"session": estimate.isoformat(), "estimated": True, "calendar_last": calendar["last"]}


def weekday_sessions_between(start: date | None, end: date | None) -> int:
    """Weekday count for ``start < d <= end``; an estimate, never a holiday map."""
    if start is None or end is None or end <= start:
        return 0
    span = pd.bdate_range(start + timedelta(days=1), end)
    return int(len(span))


def sessions_before(session: date | None, count: int) -> date | None:
    """Roll a session back by ``count`` weekday sessions (label horizon maths)."""
    if session is None or count <= 0:
        return session
    cursor = session
    remaining = count
    while remaining > 0:
        cursor -= timedelta(days=1)
        if cursor.weekday() < 5:
            remaining -= 1
    return cursor


# --------------------------------------------------------------------------- #
# Parquet spans
# --------------------------------------------------------------------------- #
def _span_from_statistics(handle: pq.ParquetFile, column: str) -> tuple[str, str] | None:
    """First/last from row-group statistics — no column data is read."""
    names = list(handle.schema_arrow.names)
    position = names.index(column)
    metadata = handle.metadata
    lows: list[Any] = []
    highs: list[Any] = []
    for index in range(metadata.num_row_groups):
        chunk = metadata.row_group(index).column(position)
        if chunk.path_in_schema != column:
            return None
        stats = chunk.statistics
        if stats is None or not stats.has_min_max:
            return None
        lows.append(stats.min)
        highs.append(stats.max)
    if not lows:
        return None
    low, high = min(lows), max(highs)
    try:
        low_stamp = pd.Timestamp(low, unit="ns") if isinstance(low, int) else pd.Timestamp(low)
        high_stamp = pd.Timestamp(high, unit="ns") if isinstance(high, int) else pd.Timestamp(high)
    except Exception:
        return None
    if pd.isna(low_stamp) or pd.isna(high_stamp):
        return None
    return low_stamp.date().isoformat(), high_stamp.date().isoformat()


def parquet_span(path: Path) -> dict[str, Any] | None:
    """Row count plus first/last timestamp from metadata, not payload."""
    try:
        handle = pq.ParquetFile(path)
    except Exception:
        return None
    names = list(handle.schema_arrow.names)
    rows = int(handle.metadata.num_rows)
    column = next((name for name in _TIME_COLUMNS if name in names), None)
    if column is None:
        for name in names:
            field_type = handle.schema_arrow.field(name).type
            if pa.types.is_timestamp(field_type) or pa.types.is_date(field_type):
                column = name
                break
    if column is None:
        return {"rows": rows, "first": None, "last": None, "column": None}

    span = _span_from_statistics(handle, column)
    if span is not None:
        return {"rows": rows, "first": span[0], "last": span[1], "column": column}
    # Statistics absent (rare, e.g. files written without them): read one column.
    try:
        values = pd.to_datetime(pq.read_table(path, columns=[column]).column(0).to_pandas(), errors="coerce").dropna()
    except Exception:
        return {"rows": rows, "first": None, "last": None, "column": column}
    if values.empty:
        return {"rows": rows, "first": None, "last": None, "column": column}
    return {
        "rows": rows,
        "first": values.min().date().isoformat(),
        "last": values.max().date().isoformat(),
        "column": column,
    }


def dataset_coverage(paths: Paths, files: Iterable[Path], expected: date | None, sample: int = 6) -> dict[str, Any]:
    """Aggregate spans across many parquet files without loading them fully."""
    file_list = sorted(files)
    spans: list[tuple[str, dict[str, Any]]] = []
    for path in file_list:
        span = _cached(path, parquet_span, namespace="span")
        if span:
            spans.append((path.stem, span))
    if not spans:
        return {
            "files": 0,
            "rows": 0,
            "first": None,
            "last": None,
            "last_min": None,
            "symbols_behind": 0,
            "behind": [],
            "empty": [],
        }
    lasts = [(name, to_date(span["last"])) for name, span in spans]
    behind = sorted(name for name, last in lasts if expected and (last is None or last < expected))
    totals = sum(span["rows"] for _, span in spans)
    firsts = [span["first"] for _, span in spans if span["first"]]
    last_values = [span["last"] for _, span in spans if span["last"]]
    return {
        "files": len(spans),
        "rows": int(totals),
        "first": min(firsts) if firsts else None,
        "last": max(last_values) if last_values else None,
        "last_min": min(last_values) if last_values else None,
        "symbols_behind": len(behind),
        "behind": behind[:sample],
        "empty": sorted(name for name, span in spans if span["rows"] == 0)[:sample],
    }


def _rel(paths: Paths, path: Path) -> str:
    try:
        return path.resolve().relative_to(paths.root.resolve()).as_posix()
    except Exception:
        return path.name


def symbol_dates(path: Path) -> pd.DatetimeIndex | None:
    """Every date a symbol actually has a bar for (cached numpy-backed index)."""
    values = _cached(path, lambda p: pq.read_table(p, columns=["timestamp"]).column(0).to_pandas(), namespace="dates")
    if values is None:
        return None
    index = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce")).dropna().normalize().unique().sort_values()
    return index


def holes_summary(paths: Paths, calendar: dict[str, Any], window_sessions: int = 750) -> dict[str, Any]:
    """Sessions a symbol should have inside its own coverage but does not.

    Legitimate absences (pre-listing, exchange holidays) are excluded by clipping
    the calendar to the symbol's own first/last date.
    """
    if not calendar.get("available"):
        return {"available": False, "reason": "trading calendar unavailable", "symbols_with_holes": 0, "total_holes": 0, "detail": []}
    calendar_frame = frame_file(paths.root / "data" / "reference" / "trading_calendar.parquet")
    if calendar_frame is None or "session" not in calendar_frame.columns:
        return {"available": False, "reason": "trading calendar unreadable", "symbols_with_holes": 0, "total_holes": 0, "detail": []}
    # Lazy import: the repair module pulls in the market-data provider stack.
    from ..data.heal import missing_sessions

    sessions = pd.DatetimeIndex(pd.to_datetime(calendar_frame["session"], errors="coerce")).dropna().normalize().sort_values()

    detail: list[dict[str, Any]] = []
    for path in sorted(p for p in paths.data_raw.glob("*.parquet") if p.stem != "macro"):
        dates = symbol_dates(path)
        if dates is None or dates.empty:
            continue
        missing = missing_sessions(sessions, dates, window_sessions)
        if len(missing):
            detail.append({
                "symbol": path.stem,
                "missing_sessions": int(len(missing)),
                "first_missing": missing.min().date().isoformat(),
                "last_missing": missing.max().date().isoformat(),
                "missing_dates": [d.date().isoformat() for d in missing[:10]],
            })
    return {
        "available": True,
        "window_sessions": int(min(len(sessions), window_sessions)),
        "window_start": sessions.min().date().isoformat() if len(sessions) else None,
        "symbols_checked": int(len(list(paths.data_raw.glob("*.parquet"))) - 1),
        "symbols_with_holes": len(detail),
        "total_holes": int(sum(item["missing_sessions"] for item in detail)),
        "repairable": True,
        "repair_command": "python -m stockml.main heal",
        "detail": sorted(detail, key=lambda item: -item["missing_sessions"])[:20],
    }


# --------------------------------------------------------------------------- #
# Artifact helpers shared by the routers
# --------------------------------------------------------------------------- #
def artifact_paths(paths: Paths) -> dict[str, Path]:
    return {
        "metrics": paths.results / "metrics.json",
        "backtest_metrics": paths.results / "backtest_metrics.json",
        "backtest_series": paths.results / "backtest_timeseries.parquet",
        "predictions": paths.results / "latest_predictions.csv",
        "prediction_history": paths.results / "predictions.parquet",
        "monitoring": paths.results / "monitoring.json",
        "shap": paths.results / "shap_feature_importance.csv",
        "model_metadata": paths.artifacts / "model_metadata.json",
        "calibration": paths.artifacts / "calibration.json",
        "training_context": paths.artifacts / "training_context.json",
        "best_params": paths.artifacts / "best_params.json",
        "checkpoint_index": paths.checkpoints / "lightgbm_final_checkpoint.txt.json",
        "checkpoint_model": paths.checkpoints / "lightgbm_final_checkpoint.txt",
        "q0_verdict": p0_from(paths.root)["verdict"],
        "q0_ledger": p0_from(paths.root)["state"] / "qwen_rating_ledger.parquet",
        "q0_heartbeat": p0_from(paths.root)["logs"] / "q0_heartbeat.json",
        "q0_supervisor": p0_from(paths.root)["logs"] / "q0_supervisor.json",
        "q0_gpu": p0_from(paths.root)["logs"] / "gpu_5s.csv",
        "q0_setup": p0_from(paths.root)["state"] / "qwen_setup.json",
        "q_track_state": PHASE_REGISTRY,
        "refresh_state": paths.state / "refresh.json",
    }


def predictions_frame(paths: Paths) -> pd.DataFrame | None:
    frame = csv_file(artifact_paths(paths)["predictions"])
    if frame is None or frame.empty:
        return None
    if "timestamp" not in frame.columns:
        return None
    frame = frame.copy()
    frame["_date"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame


# --------------------------------------------------------------------------- #
# Training progress (read-only inference from the trainer's own checkpoint)
# --------------------------------------------------------------------------- #
def training_status(paths: Paths, settings: Settings) -> dict[str, Any]:
    index = json_file(artifact_paths(paths)["checkpoint_index"])
    model_path = paths.artifacts / "models" / "lightgbm_final.txt"
    checkpoint_path = artifact_paths(paths)["checkpoint_model"]
    index_meta = file_meta(artifact_paths(paths)["checkpoint_index"])
    model_meta = file_meta(model_path)

    completed = index.get("completed_rounds")
    configured = int(getattr(settings, "num_boost_round", 1000) or 1000)
    checkpoint_age = age_seconds(index_meta.get("modified_at"))
    active = bool(checkpoint_age is not None and checkpoint_age < REFRESH_HEARTBEAT_STALE_SECONDS)
    progress = None
    if isinstance(completed, int) and configured > 0:
        progress = round(min(completed / configured, 1.0), 4)

    return {
        "lightgbm": {
            "available": bool(index or model_meta["available"]),
            "completed_rounds": completed if isinstance(completed, int) else None,
            "configured_rounds": configured,
            "progress": progress,
            "checkpoint_modified_at": index_meta.get("modified_at"),
            "checkpoint_age_seconds": checkpoint_age,
            "model_modified_at": model_meta.get("modified_at"),
            "active": active,
            "detection": "inferred from checkpoint index mtime (<15 min = running)",
            "source": _rel(paths, artifact_paths(paths)["checkpoint_index"]),
        },
        "qwen_qlora": {
            "available": False,
            "progress": None,
            "note": "Phase 6 emits results/qwen_qlora.json only on completion; no live step/loss telemetry is written.",
        },
    }


def _step(
    step: str,
    label: str,
    *,
    state: dict[str, Any],
    as_of: date | None,
    expected: date | None,
    backfillable: bool,
    command: str | None,
    note: str | None = None,
    age_based: bool = False,
    stale_days: int = TRAIN_STALE_DAYS,
) -> dict[str, Any]:
    """One row of the data-health table."""
    available = bool(state) or as_of is not None
    updated_at = state.get("updated_at")
    lag = weekday_sessions_between(as_of, expected) if expected else 0
    if not available:
        status = "missing"
    elif age_based:
        lag = 0
        days = age_seconds(updated_at)
        days = int(days / 86400) if days is not None else None
        status = "fresh" if (days is not None and days <= stale_days) else "stale"
    elif as_of is None:
        status = "fresh" if str(state.get("status")) == "complete" else "partial"
    else:
        status = "stale" if lag > 0 else "fresh"
    return {
        "step": step,
        "label": label,
        "status": status,
        "available": available,
        "as_of": as_of.isoformat() if as_of else None,
        "expected_as_of": expected.isoformat() if expected else None,
        "lag_sessions": lag,
        "state_status": state.get("status"),
        "updated_at": updated_at,
        "age_seconds": age_seconds(updated_at),
        "backfillable": backfillable,
        "command": command,
        "note": note,
        "age_based": age_based,
    }


def data_health(paths: Paths | None = None, settings: Settings | None = None) -> dict[str, Any]:
    """Freshness, coverage, consistency checks and the exact remediation command."""
    settings = settings or load_settings()
    paths = paths or settings.paths
    expected_info = expected_session(paths)
    expected = to_date(expected_info["session"])
    horizon = int(getattr(settings, "triple_barrier_horizon", 5) or 5)
    label_expected = sessions_before(expected, horizon)

    state_dir = paths.state
    states = {
        name: json_file(state_dir / f"{name}.json")
        for name in (
            "fetch",
            "bhavcopy",
            "calendar",
            "announcements",
            "ann_score",
            "news",
            "rss",
            "bulkdeals",
            "gdelt_bq",
            "features",
            "cross_sectional",
            "labels",
            "baseline",
            "train",
            "evaluate",
            "backtest",
            "monitor",
            "predict",
        )
    }

    raw_files = [p for p in (paths.data_raw).glob("*.parquet") if p.stem != "macro"]
    groups = {
        "raw_prices": dataset_coverage(paths, raw_files, expected),
        "macro": dataset_coverage(paths, [paths.data_raw / "macro.parquet"], expected),
        "bhavcopy": dataset_coverage(paths, (paths.data_raw / "bhavcopy").glob("*.parquet"), expected),
        "announcements": dataset_coverage(paths, (paths.data_raw / "announcements").glob("*.parquet"), expected),
        "news": dataset_coverage(paths, (paths.data_raw / "news").glob("*.parquet"), expected),
        "rss": dataset_coverage(paths, (paths.data_raw / "rss").glob("*.parquet"), expected),
        "bulkdeals": dataset_coverage(paths, (paths.data_raw / "bulkdeals").glob("*.parquet"), expected),
        "gdelt_bq": dataset_coverage(paths, (paths.data_raw / "gdelt_bq").glob("*.parquet"), expected),
        "features": dataset_coverage(paths, paths.data_features.glob("*_features.parquet"), expected),
        "processed": dataset_coverage(paths, paths.data_processed.glob("*.parquet"), label_expected),
    }

    def group_date(name: str) -> date | None:
        return to_date(groups[name].get("last"))

    steps = [
        _step(
            "fetch",
            "Raw OHLCV + macro",
            state=states["fetch"],
            as_of=min(filter(None, [group_date("raw_prices"), group_date("macro")]), default=None),
            expected=expected,
            backfillable=True,
            command="python -m stockml.main fetch",
        ),
        _step(
            "bhavcopy",
            "Bhavcopy / delivery",
            state=states["bhavcopy"],
            as_of=group_date("bhavcopy"),
            expected=expected,
            backfillable=True,
            command="python -m stockml.main bhavcopy",
            note="NSE archive; 404s before ~18:30 IST and on exchange holidays.",
        ),
        _step(
            "calendar",
            "Trading calendar",
            state=states["calendar"],
            as_of=to_date(states["calendar"].get("last")),
            expected=expected,
            backfillable=True,
            command="python -m stockml.main calendar",
            note="Derived from the nifty series, so run it after fetch.",
        ),
        _step(
            "announcements",
            "NSE announcements",
            state=states["announcements"],
            as_of=group_date("announcements"),
            expected=expected,
            backfillable=True,
            command="python -m stockml.main announcements",
            note="Full history is re-pulled per symbol and de-duplicated.",
        ),
        _step(
            "ann_score",
            "Announcement FinBERT scores",
            state=states["ann_score"],
            as_of=None,
            expected=None,
            backfillable=True,
            command="python -m stockml.main ann-score",
            note="Only unseen announcement texts are scored; cached forever.",
        ),
        _step(
            "news",
            "News (AlphaVantage + GDELT DOC)",
            state=states["news"],
            as_of=group_date("news"),
            expected=expected,
            backfillable=False,
            command="python -m stockml.main news",
            note="AlphaVantage is capped at 25 requests/day and GDELT DOC only covers a rolling ~3-month window, so a 5-week gap needs repeated runs.",
        ),
        _step(
            "gdelt_bq",
            "GDELT BigQuery events",
            state=states["gdelt_bq"],
            as_of=group_date("gdelt_bq"),
            expected=expected,
            backfillable=True,
            command="python -m stockml.main gdelt-bq --force",
            note="Single UNPARTITIONED scan (~164 GB). Check artifacts/state/bq_quota.json first.",
        ),
        _step(
            "rss",
            "RSS headlines",
            state=states["rss"],
            as_of=group_date("rss"),
            expected=expected,
            backfillable=False,
            command=None,
            note="Forward-only by design: one feed snapshot per day, nothing to backfill.",
        ),
        _step(
            "bulkdeals",
            "Bulk / block deals",
            state=states["bulkdeals"],
            as_of=group_date("bulkdeals"),
            expected=expected,
            backfillable=False,
            command=None,
            note="Forward-only by design; the NSE endpoint has no history.",
        ),
        _step(
            "features",
            "Feature panel",
            state=states["features"],
            as_of=group_date("features"),
            expected=expected,
            backfillable=True,
            command="python -m stockml.main features",
            note="Incremental: only symbols whose raw data moved are rebuilt.",
        ),
        _step(
            "cross_sectional",
            "Cross-sectional context",
            state=states["cross_sectional"],
            as_of=group_date("features"),
            expected=expected,
            backfillable=True,
            command="python -m stockml.main cross-sectional",
        ),
        _step(
            "labels",
            "Labels / processed datasets",
            state=states["labels"],
            as_of=group_date("processed"),
            expected=label_expected,
            backfillable=True,
            command="python -m stockml.main labels",
            note=f"Trails features by {horizon} sessions by design (label horizon), never a real gap.",
        ),
        _step(
            "monitor",
            "Drift monitoring",
            state=states["monitor"],
            as_of=to_date(json_file(artifact_paths(paths)["monitoring"]).get("latest_data")),
            expected=expected,
            backfillable=True,
            command="python -m stockml.main monitor",
        ),
        _step(
            "predict",
            "Latest predictions",
            state=states["predict"],
            as_of=to_date((predictions_frame(paths)["_date"].max() if predictions_frame(paths) is not None else None)),
            expected=expected,
            backfillable=True,
            command="python -m stockml.main predict",
            note="Predict reads the last row of the feature file, so its date can only advance after features — re-running predict alone never refreshes it.",
        ),
        _step(
            "baseline",
            "Baseline logistic model",
            state=states["baseline"],
            as_of=None,
            expected=None,
            backfillable=True,
            command="python -m stockml.main baseline",
            age_based=True,
        ),
        _step(
            "train",
            "LightGBM training",
            state=states["train"],
            as_of=to_date(states["train"].get("updated_at")),
            expected=None,
            backfillable=True,
            command="python -m stockml.main train",
            note="Retraining moves the holdout end date; log a holdout touch.",
            age_based=True,
        ),
        _step(
            "evaluate",
            "Evaluation",
            state=states["evaluate"],
            as_of=to_date(states["evaluate"].get("updated_at")),
            expected=None,
            backfillable=True,
            command="python -m stockml.main evaluate",
            age_based=True,
        ),
        _step(
            "backtest",
            "Backtest",
            state=states["backtest"],
            as_of=to_date(states["backtest"].get("updated_at")),
            expected=None,
            backfillable=True,
            command="python -m stockml.main backtest",
            age_based=True,
        ),
    ]

    checks = _consistency_checks(paths, groups, expected, label_expected, states, horizon)
    holes = holes_summary(paths, calendar_summary(paths))
    if holes.get("available"):
        checks.append({
            "id": "raw_series_has_no_holes",
            "ok": holes["total_holes"] == 0,
            "severity": "warn",
            "detail": (
                f"{holes['total_holes']} missing session(s) across {holes['symbols_with_holes']} symbols inside their own coverage window. "
                "Re-fetching a short window repairs these; appending from the last date does not."
                if holes["total_holes"]
                else "No internal holes: every symbol has a bar for each session inside its coverage window."
            ),
        })
    refresh = _refresh_block(paths)
    training = training_status(paths, settings)

    stale = [s for s in steps if s["status"] == "stale"]
    missing = [s for s in steps if s["status"] == "missing"]
    actionable = [
        {
            "step": s["step"],
            "label": s["label"],
            "status": s["status"],
            "lag_sessions": s["lag_sessions"],
            "backfillable": s["backfillable"],
            "command": s["command"],
        }
        for s in steps
        if s["status"] in {"stale", "missing"} and s["command"]
    ]

    return {
        "generated_at": now_ist().isoformat(),
        "expected_session": expected_info["session"],
        "expected_session_estimated": bool(expected_info["estimated"]),
        "market_close_passed": now_ist().time() >= SESSION_DATA_CUTOFF,
        "label_expected_session": label_expected.isoformat() if label_expected else None,
        "label_horizon_sessions": horizon,
        "calendar": calendar_summary(paths),
        "gap": {
            "gap_sessions_estimate": weekday_sessions_between(group_date("features"), expected),
            "estimated": True,
            "note": "Weekday count since the feature panel's last session; NSE holidays are not modelled.",
        },
        "steps": steps,
        "coverage": groups,
        "holes": holes,
        "checks": checks,
        "refresh_run": refresh,
        "training": training,
        "summary": {
            "stale_steps": [s["step"] for s in stale],
            "missing_steps": [s["step"] for s in missing],
            "actionable": actionable,
            "refresh_running": bool(refresh.get("running")),
            "fixable_command": (
                "python -m stockml.main refresh"
                if any(s["backfillable"] and s["status"] in {"stale", "missing"} for s in steps)
                else None
            ),
        },
    }


def _consistency_checks(
    paths: Paths,
    groups: dict[str, dict[str, Any]],
    expected: date | None,
    label_expected: date | None,
    states: dict[str, dict[str, Any]],
    horizon: int,
) -> list[dict[str, Any]]:
    """Cross-artifact checks a per-file timestamp can never catch."""
    checks: list[dict[str, Any]] = []

    features_last = to_date(groups["features"].get("last"))
    frame = predictions_frame(paths)
    prediction_date = to_date(frame["_date"].max()) if frame is not None else None
    if frame is None:
        checks.append({
            "id": "predictions_present",
            "ok": False,
            "severity": "error",
            "detail": "results/latest_predictions.csv is missing or unreadable.",
        })
    else:
        checks.append({
            "id": "predictions_match_features",
            "ok": bool(features_last and prediction_date == features_last),
            "severity": "error",
            "detail": (
                f"Predictions are dated {prediction_date or 'unknown'} while the feature panel ends {features_last or 'unknown'}. "
                "Because predict reads the last feature row, re-running predict cannot advance the date."
            ),
            "prediction_date": prediction_date.isoformat() if prediction_date else None,
            "features_date": features_last.isoformat() if features_last else None,
        })

    checks.append({
        "id": "features_reach_expected_session",
        "ok": bool(features_last and expected and features_last >= expected),
        "severity": "error",
        "detail": (
            f"Feature panel ends {features_last or 'unknown'}; expected session is {expected or 'unknown'}."
        ),
        "lag_sessions": weekday_sessions_between(features_last, expected),
    })

    processed_last = to_date(groups["processed"].get("last"))
    checks.append({
        "id": "labels_lag_by_horizon",
        "ok": bool(processed_last and label_expected and processed_last >= label_expected),
        "severity": "error",
        "detail": (
            f"Processed datasets end {processed_last or 'unknown'}; labels can only exist up to {label_expected or 'unknown'} "
            f"({horizon} sessions behind features)."
        ),
    })

    behind = groups["features"].get("behind") or []
    checks.append({
        "id": "features_cover_all_symbols",
        "ok": not behind,
        "severity": "warn",
        "detail": (
            f"{groups['features'].get('symbols_behind', 0)} symbol feature files are behind the expected session."
            if behind
            else "All symbol feature files reach the expected session."
        ),
        "symbols": behind[:10],
    })

    empty_news = groups["news"].get("empty") or []
    if empty_news:
        checks.append({
            "id": "news_symbols_without_rows",
            "ok": False,
            "severity": "warn",
            "detail": (
                f"{len(empty_news)} symbols have zero news rows in the current file set: {', '.join(empty_news)}. "
                "This is a provider/query coverage limit, not a date gap — backfill cannot invent rows."
            ),
            "symbols": empty_news,
        })

    missing_states = [name for name, value in states.items() if not value]
    if missing_states:
        checks.append({
            "id": "pipeline_state_missing",
            "ok": False,
            "severity": "warn",
            "detail": f"No state file recorded for: {', '.join(sorted(missing_states))}.",
        })
    return checks


def _refresh_block(paths: Paths) -> dict[str, Any]:
    state = json_file(artifact_paths(paths)["refresh_state"])
    if not state:
        return {"available": False, "running": False, "state": None}
    age = age_seconds(state.get("updated_at"))
    running = (
        str(state.get("status")) == "running"
        and age is not None
        and age < REFRESH_HEARTBEAT_STALE_SECONDS
    )
    return {
        "available": True,
        "running": running,
        "state": state.get("status"),
        "current_step": state.get("current_step"),
        "step_index": state.get("step_index"),
        "step_total": state.get("step_total"),
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "duration_seconds": state.get("duration_seconds"),
        "updated_at": state.get("updated_at"),
        "heartbeat_age_seconds": age,
        "steps": state.get("steps"),
        "failures": state.get("failures") or [],
        "retrain": state.get("retrain"),
        "options": state.get("options"),
        "source": "artifacts/state/refresh.json",
    }
