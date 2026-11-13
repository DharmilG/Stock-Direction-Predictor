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
|   |-- v2/                        Isolated second model: workspace, policy, training
|   |-- utils/                     Hashing, I/O, logging, state

|-- tests/                         Unit tests (core, dashboard API, diagnostics, Q0, refresh, v2 policy)
|-- DOCS/                          Project documentation
|-- artifacts/                     Model files, state snapshots, freezes
|-- checkpoints/                   Optuna DB, booster checkpoints, fold diagnostics
|-- results/                       Metrics, predictions, ledgers, plots
|-- logs/                          One log per pipeline step
|-- v2/                            v2 workspace (excluded from git): copied data, its own booster, report, logs
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
| `paired.py` | Date-clustered paired comparison: `clustered_delta()` resamples dates rather than rows (50 names share one day's move) and `mcnemar()` tests the discordant pairs. Shared by `v2/train.py` and `api/compare.py`. |
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
| `model_report.py` | Behaviour report: five-panel plate per split, mode, hypotheses, τ sweep, signed SHAP, explanations sample. Never trains. |
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
| `explain.py` | Per-row Tree SHAP waterfall for the latest serving row of each symbol → `results/explanations_serving.parquet`. Not part of `refresh`. |

### 4.12 api/

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app: mounts the routers, CORS allowlist, optional `dist/` static mount. |
| `models.py` | The model registry: `v1` (served incumbent) and `v2` (research twin, `servable=False`). `model_context(model)` gives a route the selected settings and paths, `GET /dashboard/models` reports each model's readiness, and `registry_warnings()` flags the pair when the incumbent was refit after v2 copied its dataset signature. |
| `compare.py` | `/dashboard/compare/*`: summary, segments, pairing, attribution. Joins the two `predictions.parquet` frames on `(timestamp, symbol)` and computes every paired delta on the rows both snapshots label the same way, so neither model is scored on the other's truth. |
| `dashboard.py` | Read-only artifact router: models, overview, predictions, metrics, model metadata, shap, backtest, monitoring, stock history. Every model-owned artefact is resolved through `models.model_context(model)`, so `?model=v2` is the only switch. |
| `dashboard_model.py` | `/dashboard/model/*` (report, live, cases) and `/dashboard/explain/{symbol}`. |
| `dashboard_q0.py` | `/dashboard/q0/*`: status, ledger summary, timeline, analytics, cases, case trace. |
| `dashboard_v2.py` | `/dashboard/v2/*`: summary, experiments, diagnostics, reasons, cases. Reads `v2/artifacts/v2_report.json` only; carries a `METRIC_CONTRACT` naming each metric's denominator. Since the model selector exists it is the *experiment record* feed — the comparison page's bottom half — and no longer the only way to see v2. |
| `dashboard_support.py` | Shared artifact IO: `json_file`, `frame_file`, `file_meta`, `age_seconds`, `json_safe`, envelope helpers. |
| `model_diagnostics.py` | FastAPI-free analysis: `distribution`, `confusion`, `class_profile`, `conviction`, `margins`, `threshold_sweep`, `rolling`, `mode_of`, `plate`, `hypotheses`, `driver_rows`, `waterfall`. |
| `q0_analytics.py` | FastAPI-free analysis: `analytics`, `timeline`, `case_rows`, `case_detail`, `safe_case_id`. |

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

### 4.16 v2/ -- Isolated Second Model

A second LightGBM that never touches the incumbent tree. It reads `artifacts/best_params.json`
read-only (no Optuna, no MLflow write), writes only under `v2/`, and is explicitly non-promotable: the
comparison page reports the gates that fail. Design and measured history: `DOCS/MASTER_PLAN.md` §14.2.1.

| File | Purpose |
|------|---------|
| `workspace.py` | `paths_v2()` builds a private `Paths` via `dataclasses.replace`; `provision()` copies `data/processed`, `data/features/**`, `data/reference` into `v2/data/` skipping unchanged files (`data/raw` is never copied); `read_only()` snapshots `artifacts/`, `results/`, `checkpoints/` and raises if any file changed on exit. |
| `policy.py` | The decision rule, kept out of the trainer: ordered branches `flat_rule → low_confidence → margin → direction`, `ABSTAIN = -1`, a 245-point threshold grid, a per-class recall floor, and `direction_only()` which reports both the all-rows and movers-only denominators. Feasibility failures are recorded, never raised. |
| `train.py` | `python -m stockml.v2.train`. Fits the 5-variant matrix (row filter × class weighting × prior adjustment), scores every variant and the incumbent on one shared test population, runs the date-clustered paired bootstrap, purged walk-forward folds, and writes `v2/artifacts/v2_report.json` + `v2/results/*`. |
| `stages.py` | `python -m stockml.v2.stages <evaluate\|backtest\|monitor\|report\|predict\|explain\|all>`. Points the incumbent's own stage implementations at `settings_v2()` so v2 produces the same artefact schema as v1 — SHAP, backtest, drift, a latest-date call and serving explanations — instead of a report only. `evaluate` is v2's own: it applies the real `policy.Policy` and records an `ABSTAIN` as `predicted_label = -1`. Every stage runs inside `read_only()`. |
| `__init__.py` | Package marker. |

The page that reads it is `stock-direction-dashboard/src/pages/ModelCompare.jsx` (`#/compare`, with
`#/v2` redirected). Its top half consumes `/dashboard/compare/*` — the paired scorecard, segment table,
agreement 2×2, flip matrix and two SHAP orderings — and its bottom half is the experiment record from
`/dashboard/v2/*`. It computes no metric of its own.

---

## 5. tests/

| File | Purpose |
|------|---------|
| `test_core.py` | Feature engineering, labeling, validation splits, metrics, economics, touch ledger, diagnostics, per-fold diagnostics. |
| `test_dashboard_api.py` | Envelope on every route, read-only guarantee (no artifact rewritten), traversal rejection, missing/malformed artifacts degrade to warnings, `data-health` gaps. |
| `test_model_diagnostics.py` | Plate numbers, `separation`, all five modes, hypothesis verdicts, waterfall additivity. |
| `test_q0_analytics.py` | Funnel, vote×label, calibration/ECE, disagreement, latency from traces, case filters, path safety, resource series fields, progress rows. |
| `test_refresh.py` | Refresh order: no retraining by default, optional sources cannot abort a run, dry-run executes nothing. |
| `test_v2_policy.py` | v2 decision policy, no artefacts needed: abstain on low confidence, branch precedence, both accuracy denominators, the recall floor yielding `feasible=False` instead of raising, selection ranking, and vectorised/scalar agreement. |
| `test_model_selector.py` | The registry and the paired comparison: v2's flat band comes from its own selection, an unknown id 404s, the comparability warning fires only when the incumbent moved, `clustered_delta`/`mcnemar` behave on known inputs, `_axes` reproduces `metrics.json` (point estimate and a bootstrap batch), and the paired join accounts for every row (`paired + mismatch == overlap`, `discarded == a_only + b_only + mismatch`). |
| `test_qwen_guards.py` | Q0 thermal guard, gate and vote guards (GPU sampled via monkeypatch). |

Run with: `python -m pytest tests -q`

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
| `train_curve.jsonl` | Per-iteration train vs calibration loss/accuracy, appended by the training callback (observability only). |

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
| `shap_feature_importance.csv` | SHAP feature importance rankings (all-class mean \|SHAP\|). |
| `shap_by_class.csv` | Signed mean SHAP per class — the source of plate panel E. |
| `model_report.json` | Behaviour report: plates per split, mode, hypotheses, τ sweep, folds, importance, consistency. |
| `predictions_splits.parquet` | The same booster scored on train / calibration / test (does not replace `predictions.parquet`). |
| `explanations_splits.parquet` | Sampled per-row SHAP contributions per class, with base values. |
| `explanations_serving.parquet` | Serving-row waterfalls, written by `explain`. |
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
| `report.log` | Behaviour report (`model-report`) |
| `explain.log` | Serving explanations (`explain`) |

The v2 experiment logs to its own workspace, not here: `v2/logs/v2_train.log`.

Telemetry files that live in `logs/` are not step logs: `train_heartbeat.json`
(current training progress), `gpu_5s.csv` (temperature), `q0_resource_5s.csv`
(the superset GPU sample), `q0_history.jsonl` (throttled Q0 heartbeats),
`q0_heartbeat.json` (current Q0 case). See `DOCS/TRACK_Q_TELEMETRY.md`.

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

## 10.1 v2/ (Excluded from Git)

The isolated workspace for the second model. `python -m stockml.v2.train` creates it and
`python -m stockml.v2.stages all` fills in the twin stages; nothing else writes here, and nothing here is
read by the serving path. Copied from `data/` by `workspace.provision()` (size+mtime skipping), which
never copies `data/raw/`.

| Path | Contents |
|------|----------|
| `data/processed/`, `data/features/`, `data/reference/` | Private copy of the dataset the trainer fits on. |
| `artifacts/v2_report.json` | The whole measurement: protocol, population, every variant, head-to-head, gates, significance, diagnostics, folds. This is the file `/dashboard/v2/*` serves. |
| `artifacts/v2_booster.txt`, `artifacts/v2_calibration.json`, `artifacts/v2_selection.json` | The selected variant's booster, its temperature, and how it was chosen — including `policy.tau_flat`, which the registry applies when it scores v2. |
| `results/v2_predictions.parquet`, `v2_folds.json`, `v2_metrics.json` | Per-row paired calls (what the comparison page's cases table pages through), fold table, metrics. |
| `results/predictions.parquet`, `metrics.json`, `shap_feature_importance.csv`, `backtest_*.json`, `monitoring.json`, `model_report.json`, `latest_predictions.csv` | The twin stages' output, in the incumbent's own schema. `results/predictions.parquet` is what `/dashboard/compare/*` joins against v1's. |
| `logs/v2_train.log`, `logs/v2_stages.log` | Run logs. |

---

## 11. DOCS/

| File | Purpose |
|------|---------|
| `MASTER_PLAN.md` | Living project plan: architecture, decisions, task tracker (1242 lines). |
| `MODEL_OBSERVABILITY.md` | Behaviour Plate, mode taxonomy, hypotheses, `model-report` artifacts, live training telemetry, per-case explanations, retrain policy. |
| `TRACK_Q_TELEMETRY.md` | Q0 telemetry files, the absent-never-zero rule, timeline/analytics/cases payloads. |
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
