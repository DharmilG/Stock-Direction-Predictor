from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
MLFLOW_DB = ROOT / "artifacts" / "mlflow.db"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data_raw: Path = ROOT / "data" / "raw"
    data_processed: Path = ROOT / "data" / "processed"
    data_features: Path = ROOT / "data" / "features"
    # N6 split store: separated build artifacts; data_features/ holds the
    # materialized joined view (same paths as before — readers unchanged).
    data_features_stocks: Path = ROOT / "data" / "features" / "stocks"
    data_features_news: Path = ROOT / "data" / "features" / "news"
    data_features_market: Path = ROOT / "data" / "features" / "market"
    checkpoints: Path = ROOT / "checkpoints"
    artifacts: Path = ROOT / "artifacts"
    results: Path = ROOT / "results"
    logs: Path = ROOT / "logs"
    state: Path = ROOT / "artifacts" / "state"
    mlflow_artifacts: Path = ROOT / "artifacts" / "mlflow"
    registry: Path = ROOT / "artifacts" / "model_registry"

    def ensure(self) -> None:
        for value in self.__dict__.values():
            if isinstance(value, Path):
                value.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    project_name: str = os.getenv("PROJECT_NAME", "stock-direction-platform")
    environment: str = os.getenv("ENVIRONMENT", "local")
    history_years: int = _env_int("HISTORY_YEARS", 16)
    interval: str = os.getenv("INTERVAL", "1d")
    provider: str = os.getenv("MARKET_DATA_PROVIDER", "yfinance")
    symbols: list[str] = field(default_factory=list)
    universe_tier: str = os.getenv("UNIVERSE_TIER", "single")

    enable_sentiment: bool = _env_bool("ENABLE_SENTIMENT", False)
    enable_news: bool = _env_bool("ENABLE_NEWS", True)
    news_provider: str = os.getenv("NEWS_PROVIDER", "alphavantage,gdelt")
    news_gdelt_days: int = _env_int("NEWS_GDELT_DAYS", 90)
    alphavantage_api_key: str = os.getenv("ALPHAVANTAGE_API_KEY", "")
    finbert_model: str = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
    finbert_device: int = _env_int("FINBERT_DEVICE", -1)
    finbert_cache_dir: str = os.getenv("FINBERT_CACHE_DIR", str(ROOT / "artifacts" / "hf_cache"))
    gcp_project_id: str = os.getenv("GCP_PROJECT_ID", "")
    gdelt_max_records: int = _env_int("GDELT_MAX_RECORDS", 250)
    optuna_pruner: str = os.getenv("OPTUNA_PRUNER", "none")
    class_weighting: bool = _env_bool("CLASS_WEIGHTING", True)

    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow_experiment: str = os.getenv("MLFLOW_EXPERIMENT", "stock-direction")
    mlflow_registered_model: str = os.getenv("MLFLOW_REGISTERED_MODEL", "stock-direction-lightgbm")

    train_start: str = os.getenv("TRAIN_START", "")
    validation_start: str = os.getenv("VALIDATION_START", "")
    test_start: str = os.getenv("TEST_START", "")

    n_splits: int = _env_int("N_SPLITS", 5)
    purge_days: int = _env_int("PURGE_DAYS", 5)
    embargo_days: int = _env_int("EMBARGO_DAYS", 5)
    optuna_trials: int = _env_int("OPTUNA_TRIALS", 25)
    optuna_timeout_seconds: int = _env_int("OPTUNA_TIMEOUT_SECONDS", 0)
    random_seed: int = _env_int("RANDOM_SEED", 42)
    num_boost_round: int = _env_int("NUM_BOOST_ROUND", 1000)
    early_stopping_rounds: int = _env_int("EARLY_STOPPING_ROUNDS", 100)
    checkpoint_every_rounds: int = _env_int("CHECKPOINT_EVERY_ROUNDS", 1)
    force_retrain: bool = _env_bool("FORCE_RETRAIN", False)

    label_method: str = os.getenv("LABEL_METHOD", "triple_barrier")
    triple_barrier_horizon: int = _env_int("TRIPLE_BARRIER_HORIZON", 5)
    triple_barrier_vol_window: int = _env_int("TRIPLE_BARRIER_VOL_WINDOW", 20)
    triple_barrier_up_mult: float = _env_float("TRIPLE_BARRIER_UP_MULT", 1.0)
    triple_barrier_down_mult: float = _env_float("TRIPLE_BARRIER_DOWN_MULT", 1.0)
    triple_barrier_flat_band: float = _env_float("TRIPLE_BARRIER_FLAT_BAND", 0.25)

    commission_bps: float = _env_float("COMMISSION_BPS", 2.0)
    slippage_bps: float = _env_float("SLIPPAGE_BPS", 5.0)
    other_cost_bps: float = _env_float("OTHER_COST_BPS", 3.0)
    backtest_segment: str = os.getenv("BACKTEST_SEGMENT", "delivery")
    backtest_confidence_threshold: float = _env_float("CONFIDENCE_THRESHOLD", 0.55)
    max_gross_exposure: float = _env_float("MAX_GROSS_EXPOSURE", 1.0)
    # Track F: FLAT decision threshold (predict FLAT iff p_flat >= tau,
    # else argmax(DOWN, UP)). Default >1.0 disables the FLAT class.
    flat_threshold: float = _env_float("FLAT_THRESHOLD", 1.01)

    drift_lookback: int = _env_int("DRIFT_LOOKBACK", 20)
    psi_buckets: int = _env_int("PSI_BUCKETS", 10)
    drift_psi_threshold: float = _env_float("DRIFT_PSI_THRESHOLD", 0.25)

    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = _env_int("API_PORT", 8000)

    features: dict[str, Any] = field(default_factory=dict)
    paths: Paths = field(default_factory=Paths)


def load_settings() -> Settings:
    cfg_path = ROOT / "config" / "config.yaml"
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    env_symbols = [s.strip() for s in os.getenv("SYMBOLS", "").split(",") if s.strip()]
    yaml_symbols = raw.get("symbols", ["^NSEI"])
    symbols = env_symbols or yaml_symbols
    s = Settings(symbols=symbols, features=raw.get("features", {}))
    if "UNIVERSE_TIER" not in os.environ:
        s.universe_tier = str(raw.get("universe_tier", s.universe_tier))
    if "ENABLE_NEWS" not in os.environ:
        s.enable_news = bool(raw.get("enable_news", s.enable_news))
    if "ENABLE_SENTIMENT" not in os.environ:
        s.enable_sentiment = bool(raw.get("enable_sentiment", s.enable_sentiment))
    if "NEWS_PROVIDER" not in os.environ:
        s.news_provider = str(raw.get("news_provider", s.news_provider))
    if "NEWS_GDELT_DAYS" not in os.environ:
        s.news_gdelt_days = int(raw.get("news_gdelt_days", s.news_gdelt_days))
    if "OPTUNA_PRUNER" not in os.environ:
        s.optuna_pruner = str(raw.get("training", {}).get("optuna_pruner", s.optuna_pruner))
    if "CLASS_WEIGHTING" not in os.environ:
        s.class_weighting = bool(raw.get("training", {}).get("class_weighting", s.class_weighting))

    target = raw.get("target", {})
    if "LABEL_METHOD" not in os.environ:
        s.label_method = target.get("method", s.label_method)
    if "TRIPLE_BARRIER_HORIZON" not in os.environ:
        s.triple_barrier_horizon = int(target.get("horizon_days", s.triple_barrier_horizon))
    if "TRIPLE_BARRIER_VOL_WINDOW" not in os.environ:
        s.triple_barrier_vol_window = int(target.get("volatility_window", s.triple_barrier_vol_window))
    if "TRIPLE_BARRIER_UP_MULT" not in os.environ:
        s.triple_barrier_up_mult = float(target.get("upper_multiplier", s.triple_barrier_up_mult))
    if "TRIPLE_BARRIER_DOWN_MULT" not in os.environ:
        s.triple_barrier_down_mult = float(target.get("lower_multiplier", s.triple_barrier_down_mult))
    if "TRIPLE_BARRIER_FLAT_BAND" not in os.environ:
        s.triple_barrier_flat_band = float(target.get("flat_band", s.triple_barrier_flat_band))

    validation = raw.get("validation", {})
    if "N_SPLITS" not in os.environ:
        s.n_splits = int(validation.get("n_splits", s.n_splits))
    if "PURGE_DAYS" not in os.environ:
        s.purge_days = int(validation.get("purge_days", s.purge_days))
    if "EMBARGO_DAYS" not in os.environ:
        s.embargo_days = int(validation.get("embargo_days", s.embargo_days))

    training = raw.get("training", {})
    if "OPTUNA_TRIALS" not in os.environ:
        s.optuna_trials = int(training.get("optuna_trials", s.optuna_trials))
    if "NUM_BOOST_ROUND" not in os.environ:
        s.num_boost_round = int(training.get("num_boost_round", s.num_boost_round))
    if "EARLY_STOPPING_ROUNDS" not in os.environ:
        s.early_stopping_rounds = int(training.get("early_stopping_rounds", s.early_stopping_rounds))
    if "CHECKPOINT_EVERY_ROUNDS" not in os.environ:
        s.checkpoint_every_rounds = int(training.get("checkpoint_every_rounds", s.checkpoint_every_rounds))

    backtest = raw.get("backtest", {})
    if "BACKTEST_SEGMENT" not in os.environ:
        s.backtest_segment = str(backtest.get("segment", s.backtest_segment)).lower()
    if "CONFIDENCE_THRESHOLD" not in os.environ:
        s.backtest_confidence_threshold = float(backtest.get("confidence_threshold", s.backtest_confidence_threshold))
    decision = raw.get("decision", {})
    if "FLAT_THRESHOLD" not in os.environ:
        s.flat_threshold = float(decision.get("flat_threshold", s.flat_threshold))
    if "MAX_GROSS_EXPOSURE" not in os.environ:
        s.max_gross_exposure = float(backtest.get("max_gross_exposure", s.max_gross_exposure))
    if "COMMISSION_BPS" not in os.environ:
        s.commission_bps = float(backtest.get("commission_bps", s.commission_bps))
    if "SLIPPAGE_BPS" not in os.environ:
        s.slippage_bps = float(backtest.get("slippage_bps", s.slippage_bps))
    if "OTHER_COST_BPS" not in os.environ:
        s.other_cost_bps = float(backtest.get("other_cost_bps", s.other_cost_bps))

    monitoring = raw.get("monitoring", {})
    if "DRIFT_LOOKBACK" not in os.environ:
        s.drift_lookback = int(monitoring.get("drift_lookback", s.drift_lookback))
    if "PSI_BUCKETS" not in os.environ:
        s.psi_buckets = int(monitoring.get("psi_buckets", s.psi_buckets))
    if "DRIFT_PSI_THRESHOLD" not in os.environ:
        s.drift_psi_threshold = float(monitoring.get("drift_psi_threshold", s.drift_psi_threshold))

    s.paths.ensure()
    return s
