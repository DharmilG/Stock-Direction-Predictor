from __future__ import annotations

from pathlib import Path

from .backtest.backtest import run_backtest
from .config import load_settings
from .data.fetch_data import run_fetch
from .data.news_fetch import run_news_fetch
from .evaluation.evaluate import run_evaluation
from .features.cross_sectional import run_cross_sectional
from .features.feature_engineering import run_feature_engineering
from .inference.predict import run_prediction
from .models.baseline import train_baseline
from .models.lightgbm_train import run_training
from .monitoring.monitor import run_monitor
from .targets.build_dataset import run_build_dataset
from .utils.logging import setup_logging
from .utils.state import get_state


def _latest_mtime(paths: list[Path]) -> float:
    existing = [p.stat().st_mtime for p in paths if p.exists()]
    return max(existing) if existing else 0.0


def should_run(name: str, settings) -> bool:
    paths = settings.paths
    if name in {"fetch", "cross-sectional", "labels"}:
        return True  # individual functions are freshness-aware and resumable per file.
    state_name = "cross_sectional" if name == "cross-sectional" else name
    state = get_state(paths.state, state_name)
    if state.get("status") != "complete":
        return True
    if settings.force_retrain and name in {"baseline", "train", "evaluate", "backtest", "monitor", "predict"}:
        return True
    if name == "news":
        if not settings.enable_news:
            return False
        news_paths = list((paths.data_raw / "news").glob("*.parquet"))
        if not news_paths:
            return True
        newest_news = _latest_mtime(news_paths)
        newest_raw = _latest_mtime(list(paths.data_raw.glob("*.parquet")))
        return newest_raw >= newest_news or newest_news == 0.0
    if name == "features":
        latest_raw = _latest_mtime(list(paths.data_raw.glob("*.parquet")) + list((paths.data_raw / "news").glob("*.parquet")))
        latest_features = _latest_mtime(list(paths.data_features.glob("*.parquet")))
        return latest_raw > latest_features
    if name == "baseline":
        return _latest_mtime(list(paths.data_processed.glob("*.parquet"))) > _latest_mtime([paths.artifacts / "models" / "baseline_logistic.joblib"])
    if name == "train":
        return _latest_mtime(list(paths.data_processed.glob("*.parquet"))) > _latest_mtime([paths.artifacts / "models" / "lightgbm_final.txt"])
    if name == "evaluate":
        return _latest_mtime([paths.artifacts / "models" / "lightgbm_final.txt"]) > _latest_mtime([paths.results / "predictions.parquet"])
    if name == "backtest":
        return _latest_mtime([paths.results / "predictions.parquet"]) > _latest_mtime([paths.results / "backtest_timeseries.parquet"])
    if name == "monitor":
        return _latest_mtime(list(paths.data_features.glob("*.parquet"))) > _latest_mtime([paths.results / "monitoring.json"])
    if name == "predict":
        return True  # latest prediction is cheap and intentionally refreshed every full run.
    return False


def run_all() -> None:
    settings = load_settings()
    logger = setup_logging(settings.paths.logs / "pipeline.log")
    steps = [
        ("fetch", run_fetch),
        ("news", run_news_fetch),
        ("features", run_feature_engineering),
        ("cross-sectional", run_cross_sectional),
        ("labels", run_build_dataset),
        ("baseline", train_baseline),
        ("train", run_training),
        ("evaluate", run_evaluation),
        ("backtest", run_backtest),
        ("monitor", run_monitor),
        ("predict", run_prediction),
    ]
    for name, func in steps:
        if not should_run(name, settings):
            logger.info("%s is already complete and current; skipping.", name)
            continue
        logger.info("========== START %s ==========", name)
        func()
        logger.info("========== COMPLETE %s ==========", name)
