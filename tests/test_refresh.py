"""Ordering and safety guarantees for the refresh chain.

``stockml.pipeline.run_all`` is the full research run and retrains. ``run_refresh``
exists so "update my data" does not silently move the holdout window; these tests
pin that contract down without touching the network or the artifact tree.
"""

from __future__ import annotations

import pandas as pd

from stockml.data.heal import missing_sessions, scan_holes
from stockml.data.refresh import RefreshOptions, build_steps, run_refresh


def _names(options: RefreshOptions) -> list[str]:
    return [step.name for step in build_steps(options)]


def test_default_plan_never_retrains():
    names = _names(RefreshOptions())
    assert "train" not in names
    assert "evaluate" not in names
    assert "backtest" not in names
    # Prediction is last: it reads the feature row the chain just rebuilt.
    assert names[-1] == "predict"
    assert names[0] == "heal"


def test_dependency_order_is_respected():
    names = _names(RefreshOptions())
    assert names.index("heal") < names.index("fetch")
    assert names.index("fetch") < names.index("features")
    assert names.index("features") < names.index("cross-sectional")
    assert names.index("cross-sectional") < names.index("labels")
    assert names.index("labels") < names.index("monitor")
    assert names.index("monitor") < names.index("predict")


def test_retrain_plan_inserts_model_steps_before_predict():
    names = _names(RefreshOptions(retrain=True))
    assert names.index("labels") < names.index("train")
    assert names.index("train") < names.index("evaluate")
    assert names.index("evaluate") < names.index("backtest")
    assert names.index("backtest") < names.index("predict")


def test_optional_sources_can_be_skipped():
    names = _names(RefreshOptions(skip_news=True, skip_announcements=True, skip_gdelt=True))
    assert not any(name in names for name in ("news", "rss", "bulkdeals", "announcements", "ann-score", "gdelt-bq"))
    with_gdelt = _names(RefreshOptions(skip_news=True, skip_gdelt=False))
    assert "gdelt-bq" in with_gdelt


def test_optional_steps_are_marked_so_one_source_cannot_abort_the_run():
    optional = {step.name: step.optional for step in build_steps(RefreshOptions())}
    # Raw prices, features, labels and prediction are required.
    for required in ("fetch", "features", "cross-sectional", "labels", "predict"):
        assert optional[required] is False, required
    # External, rate-limited sources are not.
    for soft in ("bhavcopy", "calendar", "announcements", "news", "rss", "bulkdeals"):
        assert optional[soft] is True, soft


def test_dry_run_plans_without_executing_or_writing_state():
    result = run_refresh(dry_run=True)
    assert result["status"] == "dry-run"
    assert result["step_total"] == len(result["plan"])
    assert [item["index"] for item in result["plan"]] == list(range(1, len(result["plan"]) + 1))
    assert result["options"]["retrain"] is False
    assert "step_index" not in result  # no heartbeat object was produced


# --------------------------------------------------------------------------- #
# Hole detection (pure, no network)
# --------------------------------------------------------------------------- #
def test_missing_sessions_ignores_pre_listing_dates():
    calendar = pd.bdate_range("2024-01-01", "2024-01-31")
    listed_from = calendar[10:]
    assert len(missing_sessions(calendar, listed_from, 750)) == 0


def test_missing_sessions_finds_internal_holes_only():
    calendar = pd.bdate_range("2024-01-01", "2024-03-29")
    dates = calendar.delete([20, 21, 40])
    missing = missing_sessions(calendar, dates, 750)
    assert list(missing) == [calendar[20], calendar[21], calendar[40]]


def test_missing_sessions_respects_the_window():
    calendar = pd.bdate_range("2020-01-01", "2024-01-31")
    dates = calendar.delete([0])  # a hole far outside a 60-session window
    assert len(missing_sessions(calendar, dates, 60)) == 0


def test_scan_holes_reads_the_raw_panel(tmp_path):
    from stockml.config import Paths

    root = tmp_path
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    reference = root / "data" / "reference"
    reference.mkdir(parents=True)

    calendar = pd.bdate_range("2024-01-01", "2024-02-29")
    pd.DataFrame({"session": list(calendar)}).to_parquet(reference / "trading_calendar.parquet", index=False)

    frame = pd.DataFrame({"timestamp": list(calendar.delete(30)), "close": 100.0})
    frame.to_parquet(raw / "AAA.NS.parquet", index=False)
    pd.DataFrame({"timestamp": list(calendar), "close": 100.0}).to_parquet(raw / "BBB.NS.parquet", index=False)
    # `macro` is a market series, not a tradable symbol: it must be ignored.
    pd.DataFrame({"timestamp": list(calendar.delete(15)), "close": 100.0}).to_parquet(raw / "macro.parquet", index=False)

    paths = Paths(root=root, data_raw=raw)
    holes = scan_holes(paths)
    assert [hole["symbol"] for hole in holes] == ["AAA.NS"]
    assert holes[0]["missing_dates"] == [calendar[30].date().isoformat()]


def test_scan_holes_is_safe_without_a_calendar(tmp_path):
    from stockml.config import Paths

    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame({"timestamp": pd.bdate_range("2024-01-01", periods=5), "close": 100.0}).to_parquet(raw / "AAA.NS.parquet", index=False)
    # No trading calendar on disk: an empty scan, not an exception.
    assert scan_holes(Paths(root=tmp_path, data_raw=raw)) == []
