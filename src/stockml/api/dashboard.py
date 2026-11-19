"""Read-only dashboard endpoints.

These routes exist for the research dashboard and nothing else: every response is
built from artifacts already on disk. No route here fetches data, retrains, or
rewrites ``results/latest_predictions.csv`` — the legacy ``/predictions`` route
still does that and is deliberately left untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ..config import Settings, load_settings
from ..qpipeline.paths import Q0_STATE_FILE, Q0_VERDICT_FILE
from .dashboard_support import (
    age_seconds,
    artifact_paths,
    csv_file,
    data_health,
    dataset_coverage,
    expected_session,
    file_meta,
    frame_file,
    json_file,
    json_safe,
    now_ist,
    predictions_frame,
    to_date,
    to_iso,
    training_status,
)
from .models import DEFAULT_MODEL, describe, model_context, registry, registry_warnings

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

VERSION = "1.0.0"

# Curated feature subset for the stock page; anything missing is simply omitted.
_STOCK_FEATURES = (
    "close",
    "volume",
    "ret_1",
    "ret_5",
    "ret_20",
    "volatility_20",
    "rsi_14",
    "atr_pct_14",
    "macd_hist",
    "bb_position",
    "relative_volume_20",
    "delivery_per",
    "news_count",
    "news_sentiment_mean",
    "macro_nifty_ret5",
    "macro_vix",
    "macro_usd_inr",
    "ret_20_xs_rank",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _envelope(data: Any, artifacts: list[str], *, as_of: Any = None, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "data": json_safe(data),
        "as_of": to_iso(as_of),
        "generated_at": now_ist().isoformat(),
        "source": {"kind": "artifact", "artifacts": artifacts},
        "warnings": warnings or [],
    }


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@router.get("/models")
def dashboard_models() -> dict[str, Any]:
    """Which models the header can offer, and how complete each one is.

    The page uses ``ready`` to disable a model that has not been built yet, and
    ``missing`` to say which artefact is absent rather than showing a blank page.
    """
    specs = registry()
    return _envelope(
        {
            "default": DEFAULT_MODEL,
            "models": [describe(spec) for spec in specs.values()],
            "stages_command": "python -m stockml.v2.stages all",
        },
        [],
        warnings=registry_warnings(),
    )


def _safe_name(symbol: str) -> str:
    return symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")


def _require_symbol(symbol: str, settings: Settings) -> str:
    if symbol not in settings.symbols:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} is not in the configured universe")
    return _safe_name(symbol)


def _inside(base: Path, name: str) -> Path:
    """Resolve ``name`` under ``base`` and refuse anything that escapes it."""
    candidate = (base / name).resolve()
    root = base.resolve()
    if root not in candidate.parents:
        raise HTTPException(status_code=400, detail="Refusing to read outside the data directory")
    return candidate


def _missing(what: str, artifacts: list[str]) -> dict[str, Any]:
    return _envelope(None, artifacts, warnings=[f"{what} is not available yet."])


# --------------------------------------------------------------------------- #
# Health / universe
# --------------------------------------------------------------------------- #
@router.get("/health")
def dashboard_health() -> dict[str, Any]:
    settings = load_settings()
    paths = settings.paths
    artifacts = artifact_paths(paths)
    return _envelope(
        {
            "status": "ok",
            "service": "stock-direction-dashboard",
            "version": VERSION,
            "artifacts": {
                "model": file_meta(paths.artifacts / "models" / "lightgbm_final.txt")["available"],
                "predictions": artifacts["predictions"].is_file(),
                "metrics": artifacts["metrics"].is_file(),
                "backtest": artifacts["backtest_series"].is_file(),
                "monitoring": artifacts["monitoring"].is_file(),
                "q0_ledger": artifacts["q0_ledger"].is_file(),
            },
        },
        ["artifacts/models/lightgbm_final.txt"],
    )


@router.get("/symbols")
def dashboard_symbols() -> dict[str, Any]:
    settings = load_settings()
    return _envelope(
        {
            "symbols": list(settings.symbols),
            "count": len(settings.symbols),
            "universe_tier": settings.universe_tier,
        },
        ["config/config.yaml"],
    )


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
@router.get("/overview")
def dashboard_overview() -> dict[str, Any]:
    settings = load_settings()
    paths = settings.paths
    artifacts = artifact_paths(paths)

    metadata = json_file(artifacts["model_metadata"])
    metrics = json_file(artifacts["metrics"])
    backtest = json_file(artifacts["backtest_metrics"])
    monitoring = json_file(artifacts["monitoring"])
    calibration = json_file(artifacts["calibration"])

    predictions = predictions_frame(paths)
    distribution: dict[str, int] = {}
    prediction_date = None
    mean_confidence = None
    if predictions is not None:
        distribution = {str(k): int(v) for k, v in predictions["prediction"].value_counts().items()}
        prediction_date = predictions["_date"].max()
        mean_confidence = float(predictions["confidence"].mean())

    features_coverage = dataset_coverage(paths, paths.data_features.glob("*_features.parquet"), None)
    processed_coverage = dataset_coverage(paths, paths.data_processed.glob("*.parquet"), None)
    expected = expected_session(paths)
    health = data_health(paths, settings)

    training = training_status(paths, settings)
    return _envelope(
        {
            "dataset": {
                "symbols": len(settings.symbols),
                "feature_rows": features_coverage.get("rows"),
                "feature_first": features_coverage.get("first"),
                "feature_last": features_coverage.get("last"),
                "dataset_rows": processed_coverage.get("rows"),
                "dataset_first": processed_coverage.get("first"),
                "dataset_last": processed_coverage.get("last"),
                "horizon_days": settings.triple_barrier_horizon,
                "flat_band": settings.triple_barrier_flat_band,
            },
            "predictions": {
                "as_of": to_iso(prediction_date),
                "rows": 0 if predictions is None else int(len(predictions)),
                "distribution": distribution,
                "mean_confidence": mean_confidence,
            },
            "evaluation": {
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "mcc": metrics.get("mcc"),
                "roc_auc_ovr_macro": metrics.get("roc_auc_ovr_macro"),
                "ece": metrics.get("ece"),
                "n_test": metrics.get("n_test"),
                "test_start": metrics.get("test_start"),
                "beats_majority_baseline": metrics.get("beats_majority_baseline"),
                "majority_baseline_accuracy": metrics.get("majority_baseline_accuracy"),
            },
            "backtest": {
                "total_return": backtest.get("total_return"),
                "max_drawdown": backtest.get("max_drawdown"),
                "sharpe": backtest.get("sharpe"),
                "trade_count": backtest.get("trade_count"),
                "segment": backtest.get("segment"),
            },
            "model": {
                "trained_at": metadata.get("trained_at") or metadata.get("created_at"),
                "training_rows": metadata.get("training_rows"),
                "test_rows": metadata.get("test_rows"),
                "feature_count": len(metadata.get("feature_columns") or []),
                "train_accuracy": metadata.get("train_accuracy"),
                "final_test_accuracy": metadata.get("final_test_accuracy"),
                "temperature": calibration.get("temperature"),
            },
            "drift": {
                "alerts": len(monitoring.get("alerts") or {}),
                "latest_data": monitoring.get("latest_data"),
                "missing_rate": monitoring.get("missing_rate"),
            },
            "freshness": {
                "expected_session": expected["session"],
                "expected_session_estimated": expected["estimated"],
                "features_as_of": features_coverage.get("last"),
                "predictions_as_of": to_iso(prediction_date),
                "stale_steps": health["summary"]["stale_steps"],
                "gap_sessions_estimate": health["gap"]["gap_sessions_estimate"],
                "holes": health["holes"]["total_holes"],
                "refresh_running": health["refresh_run"]["running"],
            },
            "training": training,
        },
        [
            "results/latest_predictions.csv",
            "results/metrics.json",
            "results/backtest_metrics.json",
            "results/monitoring.json",
            "artifacts/model_metadata.json",
            "artifacts/calibration.json",
            "data/features/*_features.parquet",
            "data/processed/*_dataset.parquet",
        ],
        as_of=features_coverage.get("last"),
        warnings=(
            [f"Feature data ends {features_coverage.get('last')} but the expected session is {expected['session']}."]
            if features_coverage.get("last") and features_coverage.get("last") < expected["session"]
            else []
        ),
    )


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
@router.get("/predictions")
def dashboard_predictions(
    limit: int = Query(default=100, ge=1, le=500),
    model: str = Query(default=DEFAULT_MODEL),
) -> dict[str, Any]:
    settings, paths = model_context(model)
    frame = predictions_frame(paths)
    if frame is None:
        return _missing("results/latest_predictions.csv", ["results/latest_predictions.csv"])

    distribution = {str(k): int(v) for k, v in frame["prediction"].value_counts().items()}
    mean_confidence = float(frame["confidence"].mean())
    as_of = frame["_date"].max()

    frame = frame.sort_values("confidence", ascending=False).head(limit)
    records = [
        {
            "symbol": row["symbol"],
            "date": to_iso(row["_date"]),
            "prediction": row["prediction"],
            "p_down": float(row["p_down"]),
            "p_flat": float(row["p_flat"]),
            "p_up": float(row["p_up"]),
            "confidence": float(row["confidence"]),
        }
        for _, row in frame.iterrows()
    ]
    expected = expected_session(paths)
    warnings = []
    if as_of is not None and str(as_of.date()) < expected["session"]:
        warnings.append(
            f"Predictions are dated {as_of.date()} while the expected session is {expected['session']}. "
            "The serving row is the last feature row, so refresh features before re-running predict."
        )
    return _envelope(
        {
            "rows": records,
            "count": len(records),
            "distribution": distribution,
            "mean_confidence": mean_confidence,
            "confidence_threshold": settings.backtest_confidence_threshold,
            "flat_threshold": settings.flat_threshold,
            "universe": len(settings.symbols),
        },
        ["results/latest_predictions.csv", "artifacts/calibration.json"],
        as_of=as_of,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@router.get("/metrics")
def dashboard_metrics(model: str = Query(default=DEFAULT_MODEL)) -> dict[str, Any]:
    settings, paths = model_context(model)
    artifacts = artifact_paths(paths)
    metrics = json_file(artifacts["metrics"])
    if not metrics:
        return _missing("results/metrics.json", ["results/metrics.json"])

    metadata = json_file(artifacts["model_metadata"])
    calibration = json_file(artifacts["calibration"])
    flat_threshold = settings.flat_threshold
    argmax_accuracy = metadata.get("final_test_accuracy")
    accuracy = metrics.get("accuracy")
    notes = [
        "metrics.json is scored after temperature calibration and the FLAT decision threshold, "
        "while model_metadata.json reports plain argmax on the same test rows.",
    ]
    if isinstance(accuracy, float) and isinstance(argmax_accuracy, float):
        notes.append(
            f"Decision-rule difference: {accuracy:.4f} (thresholded) vs {argmax_accuracy:.4f} (argmax) "
            f"= {abs(accuracy - argmax_accuracy) * 100:.2f} pp."
        )

    return _envelope(
        {
            "metrics": metrics,
            "confusion_matrix": metrics.get("confusion_matrix"),
            "labels": ["DOWN", "FLAT", "UP"],
            "classification_report": metrics.get("classification_report"),
            "calibration": calibration,
            "decision_rule": {
                "temperature": calibration.get("temperature"),
                "flat_threshold": flat_threshold,
                "confidence_threshold": settings.backtest_confidence_threshold,
                "argmax_accuracy": argmax_accuracy,
                "thresholded_accuracy": accuracy,
                "notes": notes,
            },
        },
        ["results/metrics.json", "artifacts/model_metadata.json", "artifacts/calibration.json"],
        as_of=metrics.get("test_start"),
    )


@router.get("/model-metadata")
def dashboard_model_metadata(model: str = Query(default=DEFAULT_MODEL)) -> dict[str, Any]:
    settings, paths = model_context(model)
    artifacts = artifact_paths(paths)
    metadata = json_file(artifacts["model_metadata"])
    if not metadata:
        return _missing("artifacts/model_metadata.json", ["artifacts/model_metadata.json"])
    return _envelope(
        {
            "metadata": metadata,
            "feature_columns": metadata.get("feature_columns") or [],
            "feature_count": len(metadata.get("feature_columns") or []),
            "training_context": json_file(artifacts["training_context"]),
            "best_params": json_file(artifacts["best_params"]),
            "calibration": json_file(artifacts["calibration"]),
            "checkpoint": json_file(artifacts["checkpoint_index"]),
        },
        [
            "artifacts/model_metadata.json",
            "artifacts/training_context.json",
            "artifacts/best_params.json",
            "artifacts/calibration.json",
            "checkpoints/lightgbm_final_checkpoint.txt.json",
        ],
    )


@router.get("/shap")
def dashboard_shap(
    limit: int = Query(default=25, ge=1, le=200),
    model: str = Query(default=DEFAULT_MODEL),
) -> dict[str, Any]:
    settings, paths = model_context(model)
    path = artifact_paths(paths)["shap"]
    frame = csv_file(path)
    if frame is None or frame.empty:
        return _missing("results/shap_feature_importance.csv", ["results/shap_feature_importance.csv"])
    frame = frame.head(limit)
    column = "mean_abs_shap" if "mean_abs_shap" in frame.columns else frame.columns[-1]
    total = float(frame[column].sum()) or 1.0
    rows = [
        {"feature": str(row["feature"]), "importance": float(row[column]), "share": float(row[column]) / total}
        for _, row in frame.iterrows()
    ]
    return _envelope(
        {"rows": rows, "count": len(rows), "metric": column},
        ["results/shap_feature_importance.csv"],
    )


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def _backtest_frame(paths) -> pd.DataFrame | None:
    frame = frame_file(artifact_paths(paths)["backtest_series"])
    if frame is None or frame.empty or "net_return" not in frame.columns:
        return None
    frame = frame.reset_index(drop=True)
    if "event_end" not in frame.columns:
        return None
    frame["_exit"] = pd.to_datetime(frame["event_end"], errors="coerce")
    frame["_entry"] = pd.to_datetime(frame.get("event_start"), errors="coerce")
    return frame.dropna(subset=["_exit"])


@router.get("/backtest")
def dashboard_backtest(model: str = Query(default=DEFAULT_MODEL)) -> dict[str, Any]:
    settings, paths = model_context(model)
    artifacts = artifact_paths(paths)
    metrics = json_file(artifacts["backtest_metrics"])
    frame = _backtest_frame(paths)
    if not metrics or frame is None:
        return _missing("results/backtest_metrics.json", ["results/backtest_metrics.json", "results/backtest_timeseries.parquet"])

    daily = frame.groupby(frame["_exit"].dt.normalize())["net_return"].sum().sort_index()
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    weekly_equity = equity.resample("W-FRI").last().dropna()
    weekly_drawdown = drawdown.resample("W-FRI").last().dropna()
    if len(weekly_equity) < 5:  # short history: keep the raw daily curve
        weekly_equity, weekly_drawdown = equity, drawdown

    curve = [
        {"date": index.date().isoformat(), "equity": float(value), "drawdown": float(weekly_drawdown.get(index, 0.0))}
        for index, value in weekly_equity.items()
    ]

    per_symbol = (
        frame.groupby("symbol")
        .agg(trades=("net_return", "size"), net_return=("net_return", "sum"), hit_rate=("correct", "mean"))
        .reset_index()
        .sort_values("net_return", ascending=False)
    )
    return _envelope(
        {
            "metrics": metrics,
            "curve": curve,
            "equity_points": int(len(equity)),
            "trading_days": int(len(daily)),
            "positive_days": int((daily > 0).sum()),
            "start": daily.index.min().date().isoformat() if len(daily) else None,
            "end": daily.index.max().date().isoformat() if len(daily) else None,
            "per_symbol": [
                {
                    "symbol": row["symbol"],
                    "trades": int(row["trades"]),
                    "net_return": float(row["net_return"]),
                    "hit_rate": None if pd.isna(row["hit_rate"]) else float(row["hit_rate"]),
                }
                for _, row in per_symbol.iterrows()
            ],
            "cost_model": {
                "cost_bps_round_trip": metrics.get("cost_bps_round_trip"),
                "commission_bps": settings.commission_bps,
                "slippage_bps": settings.slippage_bps,
                "other_cost_bps": settings.other_cost_bps,
                "segment": metrics.get("segment"),
                "overlap_policy": metrics.get("overlap_policy"),
            },
        },
        ["results/backtest_metrics.json", "results/backtest_timeseries.parquet"],
        as_of=metrics.get("test_start"),
    )


@router.get("/backtest/trades")
def dashboard_backtest_trades(
    symbol: str | None = Query(default=None),
    outcome: str = Query(default="all", pattern="^(all|winners|losers)$"),
    limit: int = Query(default=100, ge=1, le=1000),
    model: str = Query(default=DEFAULT_MODEL),
) -> dict[str, Any]:
    settings, paths = model_context(model)
    frame = _backtest_frame(paths)
    if frame is None:
        return _missing("results/backtest_timeseries.parquet", ["results/backtest_timeseries.parquet"])
    if symbol:
        frame = frame[frame["symbol"] == symbol]
    if outcome == "winners":
        frame = frame[frame["net_return"] > 0]
    elif outcome == "losers":
        frame = frame[frame["net_return"] < 0]

    # Counted before the `limit` cut: a footer that only counted the rows on
    # screen would call a 200-of-3000 sample a hit rate. This is the whole
    # filtered population, and `returned` says how much of it the table shows.
    total = int(len(frame))
    correct = frame["correct"] if "correct" in frame.columns else None
    hits = int(pd.Series(correct).fillna(False).astype(bool).sum()) if correct is not None else 0
    misses = total - hits
    summary = {
        "total": total,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total if total else None,
        "miss_rate": misses / total if total else None,
    }

    frame = frame.sort_values("_exit", ascending=False).head(limit)

    columns = [
        "symbol",
        "predicted_label",
        "actual_label",
        "p_down",
        "p_flat",
        "p_up",
        "entry_price",
        "exit_price",
        "realized_return",
        "gross_return",
        "cost",
        "net_return",
        "correct",
        "accepted",
    ]
    rows = []
    for _, row in frame.iterrows():
        record = {"event_start": to_iso(row.get("event_start")), "event_end": to_iso(row.get("event_end"))}
        for column in columns:
            if column in frame.columns:
                value = row[column]
                record[column] = None if pd.isna(value) else value
        rows.append(record)
    return _envelope(
        {
            "rows": rows,
            "count": len(rows),
            "summary": {**summary, "returned": len(rows)},
            "filter": {"symbol": symbol, "outcome": outcome},
        },
        ["results/backtest_timeseries.parquet"],
    )


# --------------------------------------------------------------------------- #
# Monitoring
# --------------------------------------------------------------------------- #
@router.get("/monitoring")
def dashboard_monitoring(model: str = Query(default=DEFAULT_MODEL)) -> dict[str, Any]:
    settings, paths = model_context(model)
    monitoring = json_file(artifact_paths(paths)["monitoring"])
    if not monitoring:
        return _missing("results/monitoring.json", ["results/monitoring.json"])

    alerts = monitoring.get("alerts") or []
    if isinstance(alerts, dict):  # tolerate the older dict-shaped artifact
        alerts = [{"feature": key, "psi": value} for key, value in alerts.items()]
    ordered = sorted(
        (
            {"feature": str(item.get("feature")), "psi": float(item.get("psi"))}
            for item in alerts
            if isinstance(item, dict) and isinstance(item.get("psi"), (int, float))
        ),
        key=lambda item: item["psi"],
        reverse=True,
    )
    psi = monitoring.get("feature_psi") or {}
    top_psi = sorted(
        ({"feature": key, "psi": float(value)} for key, value in psi.items() if isinstance(value, (int, float))),
        key=lambda item: item["psi"],
        reverse=True,
    )[:20]
    expected = expected_session(paths)
    latest = to_date(monitoring.get("latest_data"))
    warnings = []
    if latest and latest.isoformat() < expected["session"]:
        warnings.append(
            f"Drift was last computed on {latest.isoformat()}; the expected session is {expected['session']}."
        )
    return _envelope(
        {
            "latest_data": monitoring.get("latest_data"),
            "rows": monitoring.get("rows"),
            "missing_rate": monitoring.get("missing_rate"),
            "threshold": monitoring.get("threshold", settings.drift_psi_threshold),
            "alerts": ordered,
            "top_psi": top_psi,
            "psi_threshold": settings.drift_psi_threshold,
            "lookback": settings.drift_lookback,
            "buckets": settings.psi_buckets,
            "raw": monitoring,
        },
        ["results/monitoring.json"],
        as_of=monitoring.get("latest_data"),
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Stock detail
# --------------------------------------------------------------------------- #
@router.get("/stock/{symbol}")
def dashboard_stock(
    symbol: str,
    points: int = Query(default=180, ge=20, le=1000),
    model: str = Query(default=DEFAULT_MODEL),
) -> dict[str, Any]:
    settings, paths = model_context(model)
    safe = _require_symbol(symbol, settings)

    raw_path = _inside(paths.data_raw, f"{safe}.parquet")
    feature_path = _inside(paths.data_features, f"{safe}_features.parquet")
    raw = frame_file(raw_path)
    features = frame_file(feature_path)
    if raw is None and features is None:
        raise HTTPException(status_code=404, detail=f"No stored data for {symbol}")

    expected = expected_session(paths)
    raw_tail = raw.tail(points) if raw is not None else None
    feature_tail = features.tail(points) if features is not None else None

    price = []
    if raw_tail is not None:
        for stamp, row in raw_tail.iterrows():
            price.append({
                "date": to_iso(stamp),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })

    indicators: dict[str, list[dict[str, Any]]] = {}
    if feature_tail is not None:
        for column in _STOCK_FEATURES:
            if column not in feature_tail.columns:
                continue
            series = feature_tail[column]
            if series.dropna().empty:
                continue
            if series.dropna().nunique() <= 1:
                continue
            indicators[column] = [
                {"date": to_iso(index), "value": None if pd.isna(value) else float(value)}
                for index, value in series.items()
            ]

    predictions = predictions_frame(paths)
    latest_prediction = None
    if predictions is not None:
        match = predictions[predictions["symbol"] == symbol]
        if not match.empty:
            row = match.iloc[0]
            latest_prediction = {
                "date": to_iso(row["_date"]),
                "prediction": row["prediction"],
                "p_down": float(row["p_down"]),
                "p_flat": float(row["p_flat"]),
                "p_up": float(row["p_up"]),
                "confidence": float(row["confidence"]),
            }

    warnings: list[str] = []
    if raw is not None and not raw.empty:
        last_raw = to_iso(raw.index.max())
        if last_raw and last_raw < expected["session"]:
            warnings.append(f"{symbol} prices end {last_raw}; the expected session is {expected['session']}.")
    features_as_of = to_iso(features.index.max()) if features is not None and not features.empty else None
    return _envelope(
        {
            "symbol": symbol,
            "price": price,
            "indicators": indicators,
            "indicator_names": sorted(indicators),
            "latest_prediction": latest_prediction,
            "features_as_of": features_as_of,
            "price_as_of": to_iso(raw.index.max()) if raw is not None and not raw.empty else None,
            "rows": 0 if raw is None else int(len(raw)),
        },
        [f"data/raw/{safe}.parquet", f"data/features/{safe}_features.parquet", "results/latest_predictions.csv"],
        as_of=features_as_of,
        warnings=warnings,
    )


@router.get("/stock/{symbol}/predictions")
def dashboard_stock_predictions(
    symbol: str,
    points: int = Query(default=120, ge=10, le=1000),
    model: str = Query(default=DEFAULT_MODEL),
) -> dict[str, Any]:
    settings, paths = model_context(model)
    _require_symbol(symbol, settings)

    history = frame_file(artifact_paths(paths)["prediction_history"])
    if history is None or history.empty:
        return _missing("results/predictions.parquet", ["results/predictions.parquet"])

    # The parquet is indexed by (timestamp, symbol) *and* carries both as columns.
    frame = history.reset_index(drop=True)
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame[frame["symbol"] == symbol].sort_values("timestamp").tail(points)
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"No evaluation history for {symbol}")

    label_map = {0: "DOWN", 1: "FLAT", 2: "UP", -1: "ABSTAIN"}
    rows = [
        {
            "date": to_iso(row["timestamp"]),
            "predicted": label_map.get(int(row["predicted_label"]), str(row["predicted_label"])),
            "actual": label_map.get(int(row["actual_label"]), str(row["actual_label"])),
            "p_down": float(row["p_down"]),
            "p_flat": float(row["p_flat"]),
            "p_up": float(row["p_up"]),
            "correct": bool(row["correct"]),
            "realized_return": None if pd.isna(row.get("realized_return")) else float(row["realized_return"]),
            "rolling_accuracy": None if pd.isna(row.get("rolling_accuracy")) else float(row["rolling_accuracy"]),
        }
        for _, row in frame.iterrows()
    ]
    accuracy = float(frame["correct"].mean())
    return _envelope(
        {
            "rows": rows,
            "count": len(rows),
            "accuracy": accuracy,
            "start": rows[0]["date"],
            "end": rows[-1]["date"],
        },
        ["results/predictions.parquet"],
        as_of=rows[-1]["date"],
    )


# --------------------------------------------------------------------------- #
# Data health
# --------------------------------------------------------------------------- #
@router.get("/data-health")
def dashboard_data_health() -> dict[str, Any]:
    settings = load_settings()
    paths = settings.paths
    health = data_health(paths, settings)
    warnings: list[str] = []
    if health["summary"]["stale_steps"]:
        warnings.append(
            f"{len(health['summary']['stale_steps'])} pipeline steps are behind the expected session "
            f"{health['expected_session']}."
        )
    if health["holes"]["total_holes"]:
        warnings.append(
            f"{health['holes']['total_holes']} missing sessions were found inside symbol coverage windows."
        )
    return _envelope(
        health,
        [
            "artifacts/state/*.json",
            "artifacts/state/refresh.json",
            "data/reference/trading_calendar.parquet",
            "data/raw/*.parquet",
            "data/features/*_features.parquet",
            "data/processed/*_dataset.parquet",
        ],
        as_of=health["expected_session"],
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Small extras used by the shell
# --------------------------------------------------------------------------- #
@router.get("/activity")
def dashboard_activity(limit: int = Query(default=12, ge=1, le=50)) -> dict[str, Any]:
    """Most recent mtimes across the artifact set — a cheap 'what changed' feed."""
    settings = load_settings()
    paths = settings.paths
    candidates = [
        paths.results / "latest_predictions.csv",
        paths.results / "metrics.json",
        paths.results / "backtest_metrics.json",
        paths.results / "monitoring.json",
        paths.artifacts / "model_metadata.json",
        paths.checkpoints / "lightgbm_final_checkpoint.txt.json",
        paths.state / "refresh.json",
        Q0_STATE_FILE,
        Q0_VERDICT_FILE,
    ]
    entries = []
    for path in candidates:
        meta = file_meta(path)
        if meta["available"]:
            entries.append({
                "path": path.name,
                "modified_at": meta["modified_at"],
                "age_seconds": age_seconds(meta["modified_at"]),
                "size_bytes": meta["size_bytes"],
            })
    entries.sort(key=lambda item: item["age_seconds"] if item["age_seconds"] is not None else 10**9)
    return _envelope({"entries": entries[:limit]}, ["results/", "artifacts/", "checkpoints/"])
