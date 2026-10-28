"""Refresh the data pipeline in the order the artifacts actually depend on.

``stockml.pipeline.run_all`` is the full research run and includes retraining.
That is the wrong default when you only want today's data: it moves the holdout
window. ``run_refresh`` refreshes data and predictions, and only touches the
model when ``retrain=True``.

Progress is written to ``artifacts/state/refresh.json`` before every step, which
is what lets the dashboard show a live step counter instead of a frozen panel.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import load_settings
from ..utils.io import atomic_write_json
from ..utils.logging import setup_logging
from ..utils.state import update_state
from .announcements import run_ann_scoring, run_announcement_fetch
from .bhavcopy import run_bhavcopy_fetch
from .bulkdeals import run_bulk_fetch
from .fetch_data import run_fetch
from .gdelt_bq import run_gdelt_bq_fetch
from .heal import run_heal_holes
from .news_fetch import run_news_fetch
from .rss import run_rss_fetch
from .trading_calendar import run_build_calendar
from ..evaluation.evaluate import run_evaluation
from ..features.cross_sectional import run_cross_sectional
from ..features.feature_engineering import run_feature_engineering
from ..inference.predict import run_prediction
from ..models.lightgbm_train import run_training
from ..monitoring.monitor import run_monitor
from ..targets.build_dataset import run_build_dataset
from ..backtest.backtest import run_backtest

LOCK_FILENAME = "refresh.lock"
STALE_LOCK_SECONDS = 6 * 60 * 60


@dataclass
class Step:
    name: str
    label: str
    func: Callable[[], Any]
    optional: bool = False


@dataclass
class RefreshOptions:
    skip_heal: bool = False
    skip_news: bool = False
    skip_gdelt: bool = True
    skip_announcements: bool = False
    # FinBERT re-scoring is only needed when the model is retrained; a data-only
    # refresh can leave it to a separate run.
    skip_ann_score: bool = True
    retrain: bool = False
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "skip_heal": self.skip_heal,
            "skip_news": self.skip_news,
            "skip_gdelt": self.skip_gdelt,
            "skip_announcements": self.skip_announcements,
            "skip_ann_score": self.skip_ann_score,
            "retrain": self.retrain,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_steps(options: RefreshOptions) -> list[Step]:
    steps: list[Step] = []
    if not options.skip_heal:
        steps.append(Step("heal", "Repair internal price holes", run_heal_holes))
    steps.extend([
        Step("fetch", "Incremental OHLCV + macro", run_fetch),
        # NSE's archive 404s before ~18:30 IST and on holidays; one bad day must
        # not abort the whole refresh.
        Step("bhavcopy", "Bhavcopy / delivery", run_bhavcopy_fetch, optional=True),
        Step("calendar", "Trading calendar", run_build_calendar, optional=True),
    ])
    if not options.skip_announcements:
        steps.append(Step("announcements", "NSE announcements", run_announcement_fetch, optional=True))
        if not options.skip_ann_score:
            steps.append(Step("ann-score", "Announcement FinBERT scores", run_ann_scoring, optional=True))
    if not options.skip_news:
        steps.append(Step("news", "News (AlphaVantage + GDELT DOC)", run_news_fetch, optional=True))
        steps.append(Step("rss", "RSS headlines (today)", run_rss_fetch, optional=True))
        steps.append(Step("bulkdeals", "Bulk / block deals (today)", run_bulk_fetch, optional=True))
    if not options.skip_gdelt:
        steps.append(Step("gdelt-bq", "GDELT BigQuery backfill", lambda: run_gdelt_bq_fetch(force=True), optional=True))
    steps.extend([
        Step("features", "Feature panel", run_feature_engineering),
        Step("cross-sectional", "Cross-sectional context", run_cross_sectional),
        Step("labels", "Labels / processed datasets", run_build_dataset),
    ])
    if options.retrain:
        steps.extend([
            Step("train", "LightGBM training", run_training),
            Step("evaluate", "Evaluation", run_evaluation),
            Step("backtest", "Backtest", run_backtest),
        ])
    steps.extend([
        Step("monitor", "Drift monitoring", run_monitor),
        Step("predict", "Latest predictions", run_prediction),
    ])
    return steps


def _summarize(result: Any) -> Any:
    try:
        import pandas as pd

        if isinstance(result, pd.DataFrame):
            return {"rows": int(len(result))}
        if isinstance(result, pd.Index):
            return {"rows": int(len(result))}
    except Exception:
        pass
    if isinstance(result, dict):
        keep = {k: v for k, v in result.items() if isinstance(v, (int, float, str, bool)) or v is None}
        return keep or {"keys": sorted(result)[:8]}
    if isinstance(result, (list, tuple)):
        return {"items": len(result)}
    if result is None:
        return None
    return str(result)[:200]


def _lock_path(paths) -> Path:
    return paths.state / LOCK_FILENAME


def _acquire_lock(paths) -> Path:
    """Refuse to run two refreshes at once: they would fight over the parquet files."""
    lock = _lock_path(paths)
    if lock.exists():
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:
            age = STALE_LOCK_SECONDS + 1
        if age < STALE_LOCK_SECONDS:
            raise RuntimeError(f"Another refresh appears to be running (lock {lock.name} is {int(age)}s old).")
    atomic_write_json(lock, {"pid": os.getpid(), "started_at": _now()})
    return lock


def _release_lock(lock: Path) -> None:
    try:
        lock.unlink()
    except OSError:
        pass


def run_refresh(
    *,
    skip_heal: bool = False,
    skip_news: bool = False,
    skip_gdelt: bool = True,
    skip_announcements: bool = False,
    skip_ann_score: bool = True,
    retrain: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "refresh.log")
    options = RefreshOptions(skip_heal, skip_news, skip_gdelt, skip_announcements, skip_ann_score, retrain, dry_run)
    steps = build_steps(options)

    if dry_run:
        plan = [{"index": i + 1, "name": s.name, "label": s.label, "optional": s.optional} for i, s in enumerate(steps)]
        logger.info("Dry run: %d steps planned, nothing executed.", len(steps))
        return {"status": "dry-run", "step_total": len(steps), "plan": plan, "options": options.as_dict()}

    lock = _acquire_lock(paths)
    started_at = _now()
    state: dict[str, Any] = {
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "current_step": None,
        "step_index": 0,
        "step_total": len(steps),
        "retrain": retrain,
        "options": options.as_dict(),
        "steps": [
            {"name": s.name, "label": s.label, "status": "pending", "optional": s.optional}
            for s in steps
        ],
        "failures": [],
    }
    update_state(paths.state, "refresh", **state)

    try:
        for index, step in enumerate(steps):
            state["current_step"] = step.name
            state["step_index"] = index + 1
            state["steps"][index]["status"] = "running"
            state["steps"][index]["started_at"] = _now()
            update_state(paths.state, "refresh", **state)
            logger.info("========== REFRESH %d/%d %s ==========", index + 1, len(steps), step.name)

            began = time.time()
            try:
                result = step.func()
                state["steps"][index]["status"] = "complete"
                state["steps"][index]["summary"] = _summarize(result)
            except Exception as exc:  # one bad source must not lose the earlier work
                message = f"{type(exc).__name__}: {exc}"
                state["steps"][index]["status"] = "skipped" if step.optional else "failed"
                state["steps"][index]["error"] = message
                state["failures"].append({"step": step.name, "error": message, "optional": step.optional})
                logger.warning("Refresh step %s failed: %s", step.name, message)
                if not step.optional:
                    state["status"] = "failed"
                    state["steps"][index]["duration_seconds"] = round(time.time() - began, 2)
                    state["steps"][index]["finished_at"] = _now()
                    break
            state["steps"][index]["duration_seconds"] = round(time.time() - began, 2)
            state["steps"][index]["finished_at"] = _now()
            update_state(paths.state, "refresh", **state)

        if state["status"] == "running":
            state["status"] = "partial" if state["failures"] else "complete"
    finally:
        state["current_step"] = None
        state["finished_at"] = _now()
        state["duration_seconds"] = round(
            (datetime.fromisoformat(state["finished_at"]) - datetime.fromisoformat(started_at)).total_seconds(), 2
        )
        update_state(paths.state, "refresh", **state)
        _release_lock(lock)

    logger.info("Refresh %s in %.1fs (%d step failure(s)).", state["status"], state["duration_seconds"], len(state["failures"]))
    return state
