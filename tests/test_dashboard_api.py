"""Contract tests for the read-only dashboard API.

The dashboard exists to *report* artifacts. These tests therefore guard two
properties above all: every route answers with the same envelope, and no route
mutates a pipeline artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockml.api import dashboard as dashboard_module
from stockml.api import dashboard_q0 as q0_module
from stockml.api.dashboard_support import data_health
from stockml.api.server import app
from stockml.config import Paths, load_settings

ENVELOPE_KEYS = {"data", "as_of", "generated_at", "source", "warnings"}

READ_ONLY_ROUTES = (
    "/dashboard/health",
    "/dashboard/symbols",
    "/dashboard/overview",
    "/dashboard/predictions",
    "/dashboard/metrics",
    "/dashboard/model-metadata",
    "/dashboard/shap",
    "/dashboard/backtest",
    "/dashboard/backtest/trades",
    "/dashboard/monitoring",
    "/dashboard/data-health",
    "/dashboard/activity",
    "/dashboard/q0/status",
)


def _local_paths(root):
    """A complete Paths tree under ``root`` — every field redirected."""
    return Paths(
        root=root,
        data_raw=root / "data" / "raw",
        data_processed=root / "data" / "processed",
        data_features=root / "data" / "features",
        data_features_stocks=root / "data" / "features" / "stocks",
        data_features_news=root / "data" / "features" / "news",
        data_features_market=root / "data" / "features" / "market",
        checkpoints=root / "checkpoints",
        artifacts=root / "artifacts",
        results=root / "results",
        logs=root / "logs",
        state=root / "artifacts" / "state",
        mlflow_artifacts=root / "artifacts" / "mlflow",
        registry=root / "artifacts" / "model_registry",
    )


def _stub_settings(root):
    paths = _local_paths(root)
    paths.ensure()
    for extra in (root / "data" / "reference", root / "data" / "llm", paths.artifacts / "models"):
        extra.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        paths=paths,
        symbols=["AAA.NS", "BBB.NS"],
        universe_tier="test",
        triple_barrier_horizon=5,
        triple_barrier_flat_band=0.25,
        num_boost_round=1000,
        drift_psi_threshold=0.2,
        drift_lookback=250,
        psi_buckets=10,
        backtest_confidence_threshold=0.55,
        flat_threshold=1.01,
        commission_bps=2.0,
        slippage_bps=5.0,
        other_cost_bps=3.0,
    )


@pytest.fixture()
def client():
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Envelope + read-only guarantees against the real artifact tree
# --------------------------------------------------------------------------- #
def test_every_dashboard_route_uses_the_same_envelope(client):
    settings = load_settings()
    if not (settings.paths.results / "latest_predictions.csv").is_file():
        pytest.skip("serving artifacts are not present in this checkout")
    for route in READ_ONLY_ROUTES:
        response = client.get(route)
        assert response.status_code == 200, f"{route} -> {response.status_code}: {response.text[:200]}"
        body = response.json()
        assert set(body) == ENVELOPE_KEYS, route
        assert body["source"]["kind"] in {"artifact", "runtime"}, route
        assert isinstance(body["warnings"], list), route


def test_serving_csv_is_never_rewritten(client):
    """`GET /dashboard/predictions` must not re-run inference."""
    settings = load_settings()
    csv_path = settings.paths.results / "latest_predictions.csv"
    if not csv_path.is_file():
        pytest.skip("serving artifacts are not present in this checkout")
    before = csv_path.stat().st_mtime_ns
    assert client.get("/dashboard/predictions").status_code == 200
    assert client.get("/dashboard/overview").status_code == 200
    assert csv_path.stat().st_mtime_ns == before, "a dashboard route rewrote results/latest_predictions.csv"


def test_unknown_symbol_and_path_traversal_are_rejected(client):
    assert client.get("/dashboard/stock/NOT_A_SYMBOL").status_code == 404
    escaped = client.get("/dashboard/stock/..%2F..%2Fetc%2Fpasswd")
    assert escaped.status_code in {400, 404}


# --------------------------------------------------------------------------- #
# Empty workspace: missing artifacts degrade to warnings, never 500
# --------------------------------------------------------------------------- #
def test_missing_artifacts_return_warnings_not_errors(tmp_path, monkeypatch):
    stub = _stub_settings(tmp_path)
    monkeypatch.setattr(dashboard_module, "load_settings", lambda: stub)
    monkeypatch.setattr(q0_module, "load_settings", lambda: stub)
    local = TestClient(app)

    for route in ("/dashboard/metrics", "/dashboard/predictions", "/dashboard/backtest", "/dashboard/monitoring", "/dashboard/shap"):
        response = local.get(route)
        assert response.status_code == 200, route
        body = response.json()
        assert body["data"] is None, route
        assert body["warnings"], route

    # The Q0 ledger is a parquet, not an optional artifact: a missing file is 404.
    assert local.get("/dashboard/q0/ledger-summary").status_code == 404
    # The status endpoint itself must still answer, with warnings.
    status = local.get("/dashboard/q0/status")
    assert status.status_code == 200
    assert status.json()["data"]["runtime"]["status"] == "stale"


def test_malformed_artifacts_do_not_raise(tmp_path, monkeypatch):
    stub = _stub_settings(tmp_path)
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results" / "metrics.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "results" / "monitoring.json").write_text("[]", encoding="utf-8")
    (tmp_path / "artifacts" / "model_metadata.json").write_text("null", encoding="utf-8")
    monkeypatch.setattr(dashboard_module, "load_settings", lambda: stub)
    local = TestClient(app)

    assert local.get("/dashboard/metrics").status_code == 200
    monitoring = local.get("/dashboard/monitoring")
    assert monitoring.status_code == 200
    assert monitoring.json()["data"] is None


# --------------------------------------------------------------------------- #
# Freshness: gap, holes and cross-artifact checks
# --------------------------------------------------------------------------- #
def _write_parquet(path: Path, dates):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "timestamp": list(dates),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
        }
    )
    frame.to_parquet(path, index=False)


def test_data_health_detects_gap_hole_and_matches_predictions(tmp_path):
    stub = _stub_settings(tmp_path)
    paths = stub.paths
    calendar = pd.bdate_range("2026-09-01", "2026-09-15")
    pd.DataFrame({"session": list(calendar)}).to_parquet(tmp_path / "data" / "reference" / "trading_calendar.parquet", index=False)

    holed = calendar.delete(5)  # one session missing in the middle
    features_span = calendar[:-3]
    _write_parquet(paths.data_raw / "AAA.NS.parquet", holed)
    _write_parquet(paths.data_features / "AAA.NS_features.parquet", features_span)
    _write_parquet(paths.data_processed / "AAA.NS_dataset.parquet", calendar[:-6])

    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "symbol": ["AAA.NS"],
            "timestamp": [features_span[-1]],
            "prediction": ["UP"],
            "p_down": [0.2],
            "p_flat": [0.2],
            "p_up": [0.6],
            "confidence": [0.6],
        }
    ).to_csv(tmp_path / "results" / "latest_predictions.csv", index=False)

    (tmp_path / "artifacts" / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "state" / "refresh.json").write_text(
        json.dumps(
            {
                "status": "running",
                "updated_at": pd.Timestamp.utcnow().isoformat(),
                "current_step": "fetch",
                "step_index": 2,
                "step_total": 13,
                "steps": [{"name": "fetch", "status": "running"}],
                "retrain": False,
            }
        ),
        encoding="utf-8",
    )

    health = data_health(paths, stub)

    assert health["expected_session_estimated"] is True
    assert health["gap"]["gap_sessions_estimate"] > 0
    assert health["label_expected_session"] < health["expected_session"]

    assert health["holes"]["total_holes"] == 1
    detail = health["holes"]["detail"][0]
    assert detail["symbol"] == "AAA.NS"
    assert detail["first_missing"] == calendar[5].date().isoformat()

    checks = {check["id"]: check for check in health["checks"]}
    assert checks["raw_series_has_no_holes"]["ok"] is False
    assert checks["predictions_match_features"]["ok"] is True

    steps = {step["step"]: step for step in health["steps"]}
    assert steps["features"]["status"] == "stale"
    assert steps["features"]["lag_sessions"] > 0

    assert health["refresh_run"]["running"] is True
    assert health["refresh_run"]["step_index"] == 2
    assert "features" in health["summary"]["stale_steps"]


def test_label_horizon_is_not_reported_as_a_gap(tmp_path):
    """Processed datasets stop ``horizon`` sessions early by design."""
    stub = _stub_settings(tmp_path)
    paths = stub.paths

    # Anchor the calendar on the current session so the label expectation is
    # measured against real "today", not against an arbitrary calendar end.
    end = pd.Timestamp.now().normalize()
    while end.weekday() > 4:
        end -= pd.Timedelta("1D")
    calendar = pd.bdate_range(end=end, periods=45)

    pd.DataFrame({"session": list(calendar)}).to_parquet(tmp_path / "data" / "reference" / "trading_calendar.parquet", index=False)
    _write_parquet(paths.data_raw / "AAA.NS.parquet", calendar)
    _write_parquet(paths.data_features / "AAA.NS_features.parquet", calendar)
    _write_parquet(paths.data_processed / "AAA.NS_dataset.parquet", calendar[:-5])

    health = data_health(paths, stub)
    assert health["expected_session"] == end.date().isoformat()
    assert health["gap"]["gap_sessions_estimate"] == 0

    steps = {step["step"]: step for step in health["steps"]}
    labels = steps["labels"]
    # expected_as_of is the label session, so a 5-session trail is current, not stale.
    assert labels["expected_as_of"] == health["label_expected_session"]
    assert labels["lag_sessions"] == 0
    assert labels["status"] == "fresh"
    assert health["holes"]["total_holes"] == 0


def test_stock_route_is_bounded_by_the_universe(tmp_path, monkeypatch):
    stub = _stub_settings(tmp_path)
    _write_parquet(stub.paths.data_raw / "AAA.NS.parquet", pd.bdate_range("2026-09-01", "2026-09-10"))
    monkeypatch.setattr(dashboard_module, "load_settings", lambda: stub)
    local = TestClient(app)

    ok = local.get("/dashboard/stock/AAA.NS")
    assert ok.status_code == 200
    body = ok.json()
    assert body["data"]["symbol"] == "AAA.NS"
    assert len(body["data"]["price"]) == 8

    assert local.get("/dashboard/stock/ZZZ.NS").status_code == 404
