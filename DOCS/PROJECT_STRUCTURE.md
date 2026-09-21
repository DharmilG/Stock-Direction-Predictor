# Project Structure Guide

**Purpose:** Single reference for humans and agents to answer: "What is in this folder, what does this file do, and where does data flow?"

**Stack:** Python 3.12, LightGBM, triple-barrier labeling, Optuna, NIFTY 50 universe.

---

## 1. Folder Map

```
stock_direction_platform/
|
|-- config/                        Runtime configuration
|-- src/stockml/                   Main Python package
|   |-- data/                      Data providers and ingestion
|   |-- features/                  Feature engineering and joins
|   |-- targets/                   Label generation
|   |-- models/                    LightGBM training and baselines
|   |-- validation/                Purged walk-forward splits
|   |-- evaluation/                Metrics, diagnostics, economics
|   |-- backtest/                  Cost-aware backtesting
|   |-- monitoring/                Drift detection, PSI
|   |-- inference/                 Batch prediction
|   |-- api/                       FastAPI inference endpoint
|   |-- risk/                      Position sizing
|   |-- registry/                  MLflow model logging
|   |-- utils/                     Hashing, I/O, logging, state
|
|-- tests/                         Unit tests (49)
|-- DOCS/                          Project documentation
|-- artifacts/                     Model files, state snapshots, freezes
|-- checkpoints/                   Optuna DB, booster checkpoints, fold diagnostics
|-- results/                       Metrics, predictions, ledgers, plots
|-- logs/                          One log per pipeline step
|-- data/                          Excluded from git, rebuilt via pipeline
|   |-- raw/                       Parquet cache (yfinance, bhavcopy, news)
|   |-- processed/                 Joined dataset
|   |-- features/                  Join log
|   |-- reference/                 Sector map, group map
|
|-- .env.example                   Key template (never commit .env)
|-- .gitignore                     Exclusions (venv, secrets, parquet, mlruns)
|-- pyproject.toml                 Package metadata
|-- requirements.txt               Core dependencies
|-- requirements-optional.txt      Optional dependencies
|-- Dockerfile                     Container build
|-- docker-compose.yml             Local dev stack
|-- start_run.bat                  Windows bootstrap
|-- start_run.sh                   Linux bootstrap
|-- start_run.py                   Python bootstrap
|-- COMMANDS.txt                   Cheat sheet for common commands
```

---

## 2. Root Files

| File | Purpose |
|------|---------|
| `.env.example` | Template for API keys and settings. Copy to `.env` and fill in. Never commit `.env`. |
| `.gitignore` | Excludes `.venv/`, `.env`, `*.parquet`, `mlruns/`, `artifacts/hf_cache/`, `*.png`. Keeps `logs/`, `results/`, `checkpoints/`. |
| `pyproject.toml` | Package metadata, build config, entry points. |
| `requirements.txt` | Core dependencies: lightgbm, optuna, pandas, scipy, transformers, torch. |
| `requirements-optional.txt` | Optional: fastapi, uvicorn, shap. |
| `Dockerfile` | Multi-stage build for production deployment. |
| `docker-compose.yml` | Local dev stack with API + training containers. |
| `start_run.bat` | Windows one-command pipeline launcher. |
| `start_run.sh` | Linux one-command pipeline launcher. |
| `start_run.py` | Python bootstrap: venv check, dependency install, pipeline run. |
| `COMMANDS.txt` | Cheat sheet for fetch, train, evaluate, backtest commands. |

---

## 3. config/

| File | Purpose |
|------|---------|
| `config.yaml` | All runtime settings: universe tier, symbol list, horizon, flat_band, costs, thresholds, training params, API config. |
| `universe_history.json.example` | Template for tracking universe tier changes over time. |

Key settings in `config.yaml`:
- `universe_tier: nifty50` -- stock universe
- `horizon_days: 5` -- prediction horizon
- `flat_band: 1.0` -- FLAT class threshold (percentage)
- `segment: delivery` -- cost model (delivery = 31.5 bps round-trip)
- `confidence_threshold: 0.55` -- minimum confidence for trade signals
- `optuna_trials: 25` -- hyperparameter search budget

---

## 4. src/stockml/ -- Main Package

### 4.1 Core

| File | Purpose |
|------|---------|
| `config.py` | `Settings` dataclass, `Paths` helper, all env/config loading. Single source of truth for paths. |
| `pipeline.py` | Orchestrator: runs steps in order, manages state JSONs, handles resume from checkpoints. |
| `main.py` | CLI entry point: parses args, calls pipeline. |

### 4.2 data/ -- Providers and Ingestion

| File | Purpose |
|------|---------|
| `providers.py` | Yahoo Finance adapter: download OHLCV + volume + delivery. |
| `fetch_data.py` | Download + cache to parquet. Respects date ranges, handles gaps. |
| `bhavcopy.py` | NSE bhavcopy: delivery quantity and turnover from NSE website. |
| `trading_calendar.py` | NSE trading holidays and session dates. IST cutoff alignment. |
| `universe.py` | NIFTY 50/200/500 membership lookup. |
| `macro.py` | Macro proxies: INR/USD, India VIX, crude oil (free sources). |

### 4.3 data/ -- News and NLP

| File | Purpose |
|------|---------|
| `news.py` | GDELT DOC + AlphaVantage news providers. |
| `news_fetch.py` | News retrieval orchestrator: coordinates providers, manages quotas. |
| `announcements.py` | NSE corporate announcements: fetch, parse, FinBERT score. |
| `bulkdeals.py` | NSE bulk/block deal ingestion (forward-only, no backfill). |
| `rss.py` | RSS feeds: Economic Times, Mint, MoneyControl. |
| `gdelt_bq.py` | BigQuery historical events (GDELT GKG). Quota-aware, 800 GB/mo limit. |
| `finbert.py` | FinBERT model loading and inference. |
| `finbert_sentiment.py` | FinBERT sentiment scoring pipeline. |
| `sentiment.py` | Sentiment aggregation: per-symbol, per-day, rolling windows. |

### 4.4 features/ -- Feature Engineering

| File | Purpose |
|------|---------|
| `feature_engineering.py` | Main builder: 136 columns, split store logic, gap detection. |
| `cross_sectional.py` | 13 rank families (F-12): percentile ranks within NIFTY 50 per day. |
| `news_context.py` | N6 news-family features: announcement counts, event counts, sentiment scores. |
| `announcement_features.py` | Announcement aggregation by decision_date: count, sentiment, keywords. |
| `decision_date.py` | IST cutoff alignment (B-05): ensures features use only pre-market data. |
| `join.py` | Stocks + news + market join. Produces `_join_log.json` coverage metadata. |
| `context.py` | Helper utilities for feature context. |

Feature families:
- Price/technical: returns, volatility, momentum, RSI, MACD, Bollinger
- Volume: turnover, delivery ratio, volume shock
- Cross-sectional: percentile ranks within universe (13 families)
- News: announcement sentiment, event counts, GDELT scores
- Macro: INR, VIX, crude proxies

### 4.5 targets/ -- Label Generation

| File | Purpose |
|------|---------|
| `triple_barrier.py` | Triple-barrier labeling: up/down/flat from price paths. |
| `build_dataset.py` | Feature + label assembly: joins features with labels, handles alignment. |

Label classes:
- UP: price crosses upper barrier within horizon
- DOWN: price crosses lower barrier within horizon
- FLAT: price stays within band (default 1.0% for 5-day horizon)

### 4.6 models/ -- Training

| File | Purpose |
|------|---------|
| `lightgbm_train.py` | Optuna hyperparameter search, purged fold training, anti-memorization tripwire, fold diagnostics logging. |
| `baseline.py` | Majority-class and logistic regression baselines for comparison. |

Key settings:
- `feature_fraction: 0.6-0.8` -- anti-memorization
- `class_weighting: true` -- balance UP/DOWN/FLAT
- Tripwire: train-vs-test gap >20pp triggers MEMORIZATION FLAG

### 4.7 validation/ -- Data Splits

| File | Purpose |
|------|---------|
| `time_splits.py` | Purged walk-forward splits with embargo gap. |
| `walk_forward.py` | Expanding window validation. |

Split parameters:
- `n_splits: 5` -- number of folds
- `embargo_days: 5` -- gap between train/test
- `purge_days: 5` -- remove overlapping labels

### 4.8 evaluation/ -- Diagnostics and Metrics

| File | Purpose |
|------|---------|
| `metrics.py` | Accuracy, AUC, F1, precision, recall computation. |
| `evaluate.py` | Fold-level evaluation: per-class metrics, confusion matrix. |
| `final_report.py` | Aggregate statistics across folds. |
| `dm.py` | Diebold-Mariano test with Newey-West correction. |
| `economics.py` | G3 cost-aware variants: threshold sweep, PBO calculation. |
| `touch_ledger.py` | Canonical touch log: BACKFILL (initial) + log_touch (each change). |
| `up_autopsy.py` | UP-to-DOWN cell diagnosis: why UP predictions fail. |
| `u2_d1.py` | Scope-activity diagnosis: delivery/volume/turnover analysis. |
| `u3_d1.py` | Sentiment-content diagnosis: news signal analysis. |
| `label_grid.py` | I0-D1a 12-cell grid: horizon x multiplier ablation. |
| `i01_d1.py` | Earnings proximity + lead-lag analysis. |
| `flat_threshold.py` | tau_flat sweep: FLAT threshold optimization. |
| `flat_band_ablation.py` | B-02 flat band width study. |
| `down_flag.py` | DOWN-risk flag reformulation experiment. |
| `ablation.py` | Generic ablation runner for feature studies. |

Diagnostic framework:
- Pre-registered 2xSE bars (statistical significance)
- 23+ logged touches in `holdout_touches.jsonl`
- Anti-memorization: train-test gap >20pp flagged
- DM test for model comparison (p<0.005 threshold)

### 4.9 backtest/

| File | Purpose |
|------|---------|
| `backtest.py` | Transaction-cost-aware backtest: confidence sizing, gross exposure limits, equity curve, drawdown. |

Cost model (delivery segment):
- Commission: 1.0 bps
- Slippage: 3.0 bps
- STT: 0.1% on sell
- Other: 1.0 bps
- Total round-trip: ~31.5 bps

### 4.10 monitoring/

| File | Purpose |
|------|---------|
| `monitor.py` | Production monitoring: compares latest predictions against historical baselines. |
| `drift.py` | Population Stability Index (PSI) calculator for feature distribution shifts. |

### 4.11 inference/

| File | Purpose |
|------|---------|
| `predict.py` | Batch prediction: loads trained model, runs on latest features, outputs predictions.parquet. |

### 4.12 api/

| File | Purpose |
|------|---------|
| `server.py` | FastAPI endpoint: `/predict` route loads model, runs prediction, returns JSON. |

### 4.13 risk/

| File | Purpose |
|------|---------|
| `portfolio.py` | Position sizing: confidence-scaled signal with max-position cap, gross exposure enforcement. |

### 4.14 registry/

| File | Purpose |
|------|---------|
| `mlflow_registry.py` | MLflow logging: model artifacts, params, metrics to local or remote tracking server. |

### 4.15 utils/

| File | Purpose |
|------|---------|
| `hash.py` | Content hashing for cache invalidation. |
| `io.py` | Parquet/JSON read/write helpers. |
| `logging.py` | Structured logging setup. |
| `state.py` | Pipeline state JSON read/write (resume support). |

---

## 5. tests/

| File | Purpose |
|------|---------|
| `test_core.py` | 49 unit tests covering: config loading, feature engineering, labeling, validation splits, metrics, economics, touch ledger, diagnostics. |

Run with: `python -m pytest tests/test_core.py -v`

---

## 6. artifacts/

| Path | Purpose |
|------|---------|
| `models/lightgbm_final.txt` | Trained LightGBM model (5.7 MB). |
| `models/baseline_logistic.joblib` | Baseline logistic model. |
| `state/*.json` | Per-step pipeline state (resume checkpoints). |
| `best_params.json` | Best Optuna hyperparameters. |
| `calibration.json` | Probability calibration parameters. |
| `g1_*.json` | G1 gate freeze snapshots. |
| `label_grid.json` | I0-D1a grid results. |
| `model_metadata.json` | Training metadata (features, folds, scores). |
| `training_context.json` | Training run context. |
| `mlflow/` | Local MLflow tracking data (excluded from git). |
| `hf_cache/` | HuggingFace model cache (excluded from git). |

---

## 7. checkpoints/

| File | Purpose |
|------|---------|
| `optuna.db` | Optuna study history: all trials, params, scores. |
| `lightgbm_final_checkpoint.txt` | Last LightGBM booster (resume training). |
| `lightgbm_final_checkpoint.txt.json` | Checkpoint metadata. |
| `trials/fold_diagnostics.jsonl` | Per-fold per-class recall + top-15 gain importances. |

---

## 8. results/

| File | Purpose |
|------|---------|
| `metrics.json` | Final model metrics (UP-AUC, accuracy, F1, etc.). |
| `backtest_metrics.json` | Cost-aware backtest results (Sharpe, CAGR, max DD). |
| `economics.json` | G3 economics: threshold sweep, best variant, PBO. |
| `holdout_touches.jsonl` | Canonical touch ledger (23+ entries). |
| `down_flag.json` | DOWN-flag reformulation results. |
| `flat_threshold.json` | tau_flat sweep results. |
| `up_autopsy.json` | UP-to-DOWN cell diagnosis. |
| `u2_d1.json` | Scope-activity diagnosis. |
| `u3_d1.json` | Sentiment-content diagnosis. |
| `i01_d1.json` | Earnings proximity + lead-lag results. |
| `monitoring.json` | Production monitoring snapshot. |
| `latest_predictions.csv` | Most recent predictions per symbol. |
| `predictions.parquet` | Full prediction dataset. |
| `confidence_accuracy.csv` | Confidence vs actual accuracy calibration. |
| `shap_feature_importance.csv` | SHAP feature importance rankings. |
| `worst_20.csv` | 20 worst predictions for debugging. |
| `*.png` | Plots (excluded from git): calibration, confusion matrix, equity curve, drawdown, SHAP summary. |

---

## 9. logs/

One log file per pipeline step, named `<step>.log`. Stdout captures in `<step>.out`.

| Log | Corresponding step |
|-----|-------------------|
| `fetch.log` | Data download |
| `bhavcopy.log` | Bhavcopy ingestion |
| `calendar.log` | Trading calendar |
| `news.log` | News fetch |
| `rss.log` | RSS ingestion |
| `gdelt_bq.log` | BigQuery events |
| `announcements.log` | NSE announcements |
| `ann_score.log` | FinBERT scoring |
| `bulkdeals.log` | Bulk/block deals |
| `features.log` | Feature engineering |
| `cross_sectional.log` | Cross-sectional ranks |
| `labels.log` | Label generation |
| `train.log` | Model training |
| `evaluate.log` | Evaluation |
| `backtest.log` | Backtesting |
| `predict.log` | Inference |
| `monitor.log` | Monitoring |
| `economics.log` | G3 economics |
| `down_flag.log` | DOWN-flag experiment |
| `flat_ablation.log` | Flat band ablation |
| `flat_threshold.log` | tau_flat sweep |
| `up_autopsy.log` | UP-to-DOWN diagnosis |
| `u2_d1.log` | Scope-activity diagnosis |
| `u3_d1.log` | Sentiment-content diagnosis |
| `i01_d1.log` | Earnings proximity |
| `label_grid.log` | Label grid ablation |

---

## 10. data/ (Excluded from Git)

Rebuilt via pipeline. Not versioned.

| Path | Contents |
|------|----------|
| `raw/*.parquet` | Downloaded OHLCV + delivery + volume per symbol. |
| `processed/` | Joined feature+label dataset. |
| `features/_join_log.json` | Per-symbol join metadata (row counts, date ranges, null rates). |
| `reference/sector_map.json` | NIFTY 50 sector classification. |
| `reference/group_map.json` | NIFTY 50 group classification. |

---

## 11. DOCS/

| File | Purpose |
|------|---------|
| `MASTER_PLAN.md` | Living project plan: architecture, decisions, task tracker (1242 lines). |
| `TRACK_I_INFORMATION.md` | Track I governance: news/information sources, read-only diagnosis rules. |
| `PROJECT_STRUCTURE.md` | This file. |

---

## 12. Data Flow

```
yfinance/NSE APIs  -->  data/raw/*.parquet  (fetch)
bhavcopy/NSE       -->  data/raw/*_delivery.parquet  (bhavcopy)
news providers     -->  data/raw/news/*.parquet  (news)
                   |
                   v
feature_engineering.py  -->  features (136 cols)
cross_sectional.py      -->  +13 rank families
news_context.py         -->  +21 news features
join.py                 -->  joined dataset + _join_log.json
                   |
                   v
triple_barrier.py  -->  labels (UP/DOWN/FLAT)
                   |
                   v
lightgbm_train.py  -->  model + metrics + predictions
                   |
                   v
backtest.py        -->  cost-aware PnL
economics.py       -->  threshold optimization
monitor.py         -->  drift detection
```

---

## 13. Key Constraints

- **No stacking/ensemble** in Phase 1 (MASTER_PLAN section 9.1). Single unified LightGBM.
- **feature_fraction 0.6-0.8** to prevent memorization.
- **Anti-memorization tripwire:** train-test gap >20pp triggers warning.
- **Pre-registered significance bars:** 2xSE for all ablation comparisons.
- **India-only scope:** NSE symbols, INR costs, IST timestamps.
- **Free data sources only:** no paid API subscriptions.
- **Delivery cost model:** 31.5 bps round-trip (realistic A-band).
