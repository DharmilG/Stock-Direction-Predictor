# India Stock Direction Platform — Master Plan

**Project:** `stock_direction_platform`
**Scope:** India-only (NSE). No US/global equities as prediction targets.
**Cost constraint:** free data sources only, local development only.
**Hardware:** local PC, RTX 3050 6GB, CPU fallback for GBDT.
**Document status:** living document. Section 14 is the task tracker — update it as work completes.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Architecture Diagram](#2-architecture-diagram)
3. [What Is Already Built](#3-what-is-already-built)
4. [Data Layer — Sources, Storage, Schema](#4-data-layer--sources-storage-schema)
5. [The Hard Part — News & Context Engineering](#5-the-hard-part--news--context-engineering)
6. [Universe Expansion — Single Index → NIFTY 500](#6-universe-expansion--single-index--nifty-500)
7. [Feature Engineering](#7-feature-engineering)
8. [Labels & Targets](#8-labels--targets)
9. [Models — Algorithms and Why](#9-models--algorithms-and-why)
10. [Training, Checkpointing, Resume](#10-training-checkpointing-resume)
11. [Evaluation, Backtesting, Statistical Testing](#11-evaluation-backtesting-statistical-testing)
12. [Visualization & Reporting](#12-visualization--reporting)
13. [Bootstrap Scripts & One-Command Run](#13-bootstrap-scripts--one-command-run)
14. [Task Tracker — Done / Pending / Bugs](#14-task-tracker--done--pending--bugs)

---

## 1. What This System Does

Predicts the **direction of Indian equities over a short forward horizon**, using price/volume history, cross-sectional market structure, macro context, and dated news/corporate-announcement context.

### The daily contract

```
Day t, after NSE close (15:30 IST)
   → all features computed using ONLY information available by 15:30 IST on day t
   → model outputs P(down) / P(flat) / P(up) per symbol
   → a signal is emitted ONLY if calibrated confidence clears a threshold
   → hypothetical entry at day t+1 OPEN (never at day t close — you cannot
     compute from a close and transact at that same close)
   → exit when a barrier is touched, or at the horizon close
```

### What "good" looks like — set expectations correctly

Do **not** target "70% accuracy." That number, when seen publicly, almost always comes from one of five errors: wrong baseline (comparing to 50% instead of the majority class), temporal leakage from random train/test splits, selection bias across many attempts, silent lookahead in feature engineering, or no transaction costs.

Realistic and genuinely valuable targets:

| Metric | Target | Why this metric |
|---|---|---|
| ROC-AUC (OvR macro) | 0.55 – 0.62 | Measures ranking skill, which is what actually transfers |
| Accuracy vs **majority-class baseline** | beat it by a real, tested margin | The only accuracy comparison that means anything |
| MCC | > 0.05, ideally > 0.10 | Robust to class imbalance |
| Calibration (ECE) | < 0.05 | Required before any confidence-based sizing |
| Net Sharpe after real Indian costs | > 0 | The only economic test that matters |
| Accuracy on high-confidence subset | meaningfully above base rate | This is where real edge shows up |

The system is designed to **abstain** most days. A model that says "no signal" on 70–80% of days and is right more often on the rest is worth far more than one forced to call every day.

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            LAYER 1 — RAW INGESTION                            │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ PRICE/VOLUME   │  │ CORPORATE      │  │ NEWS           │  │ MACRO      │ │
│  │                │  │ ANNOUNCEMENTS  │  │                │  │ (India)    │ │
│  │ • yfinance     │  │ • NSE announce │  │ • GDELT DOC    │  │ • NIFTY50  │ │
│  │ • NSE bhavcopy │  │ • BSE announce │  │ • GDELT BQ     │  │ • BANKNIFTY│ │
│  │ • corp actions │  │ • results cal  │  │ • RSS (ET, MC, │  │ • INDIAVIX │ │
│  │                │  │ • board mtgs   │  │   Mint, BS)    │  │ • USDINR   │ │
│  │                │  │                │  │ • AlphaVantage │  │ • Crude    │ │
│  │                │  │                │  │   (free tier)  │  │ • Gold     │ │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘  └─────┬──────┘ │
└──────────┼───────────────────┼───────────────────┼─────────────────┼────────┘
           │                   │                   │                 │
           ▼                   ▼                   ▼                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     LAYER 2 — POINT-IN-TIME DATA STORE                        │
│                        (DuckDB + Parquet, local, free)                        │
│                                                                              │
│   Every row carries:  as_of_timestamp  │  ingested_at  │  source  │ revision │
│   HARD RULE: a feature for day t may only read rows with                      │
│              as_of_timestamp <= t 15:30 IST                                   │
│                                                                              │
│   Tables: prices · corp_actions · universe_history · announcements ·          │
│           news_raw · news_scored · entity_map · macro · trading_calendar      │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                 LAYER 3 — ENTITY RESOLUTION & CONTEXT FAN-OUT                 │
│                          (the hard, high-value part)                          │
│                                                                              │
│   Raw article ──► entity extraction ──► scope classification                  │
│                                              │                                │
│              ┌───────────────────────────────┼──────────────────────────┐     │
│              ▼                ▼              ▼              ▼           ▼     │
│        COMPANY-LEVEL     GROUP-LEVEL    SECTOR-LEVEL   MARKET-LEVEL  IGNORE   │
│        "Infosys Q3"      "Adani Group"  "IT sector"    "RBI repo"            │
│              │                │              │              │                 │
│              │                ▼              ▼              ▼                 │
│              │        fan out to all   fan out to all  applies to all         │
│              │        group members    sector members  symbols                │
│              └────────────────┴──────────────┴──────────────┘                 │
│                               │                                               │
│                               ▼                                               │
│              DECISION-TIME ALIGNMENT (24h news → tradeable session)            │
│              published <= t 15:30 IST  →  usable for day t decision           │
│              published >  t 15:30 IST  →  first usable for day t+1 decision   │
│                               │                                               │
│                               ▼                                               │
│              FinBERT scoring  +  event-type tagging  +  relevance weight       │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       LAYER 4 — FEATURE ENGINEERING                           │
│                                                                              │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌─────────┐ ┌────────┐ ┌────────┐ │
│  │ Price /  │ │ Volume   │ │ CROSS-     │ │ Regime  │ │ Macro  │ │ News / │ │
│  │ Technical│ │          │ │ SECTIONAL  │ │         │ │ (India)│ │ Event  │ │
│  │          │ │          │ │ (ranks)    │ │         │ │        │ │        │ │
│  │ returns  │ │ rel vol  │ │ ret rank   │ │ vix     │ │ nifty  │ │ sent   │ │
│  │ sma/ema  │ │ vol z    │ │ vol rank   │ │ regime  │ │ vix    │ │ count  │ │
│  │ rsi macd │ │ obv      │ │ vol-me rank│ │ bull/   │ │ usdinr │ │ event  │ │
│  │ atr bb   │ │          │ │ sector-rel │ │ bear    │ │ crude  │ │ tags   │ │
│  │ fracdiff │ │          │ │ group-rel  │ │ expiry  │ │ gold   │ │ shock  │ │
│  └──────────┘ └──────────┘ └────────────┘ └─────────┘ └────────┘ └────────┘ │
│                                                                              │
│         ALL joined on (timestamp, symbol) with strict point-in-time shift     │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 5 — LABELING                                   │
│         Triple-barrier · entry at t+1 OPEN · vol-scaled barriers              │
│         Output: label ∈ {DOWN, FLAT, UP} + event_start/end + entry/exit px    │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 6 — VALIDATION SPLITTING                             │
│         Purged + embargoed walk-forward (chronological, for training)         │
│         CPCV (combinatorial, for PBO estimation only — not for training)      │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 7 — MODELS                                     │
│                                                                              │
│   ┌────────────────────────┐        ┌──────────────────────────┐             │
│   │ PRIMARY: LightGBM      │        │ BASELINE: ElasticNet /   │             │
│   │ multiclass, one unified│  vs.   │ Logistic Regression      │             │
│   │ model, feature_fraction│        │ (mandatory sanity anchor)│             │
│   │ 0.6–0.8                │        │                          │             │
│   └───────────┬────────────┘        └────────────┬─────────────┘             │
│               │                                  │                           │
│               └──────────► Diebold-Mariano ◄─────┘                           │
│                     (is the gap real or noise?)                              │
│                                                                              │
│   DEFERRED (Phase 2 only, gated on primary beating majority baseline):        │
│   • Small TCN (<100k params) on stationary sequence inputs                    │
│   • Nonneg L2 logistic meta-learner over base model probabilities             │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  LAYER 8 — CALIBRATION & DECISION                             │
│   Temperature scaling (softmax-safe, ΣP=1 preserved) OR Dirichlet calibration │
│                               │                                               │
│                               ▼                                               │
│   Confidence threshold → UP / DOWN / FLAT / NO-SIGNAL                         │
│   Hard masks: circuit-limit proximity · illiquid · halted · missing data      │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│               LAYER 9 — EVALUATION / BACKTEST / REPORTING                     │
│                                                                              │
│  Statistical: accuracy vs MAJORITY baseline · bal-acc · MCC · AUC · ECE ·     │
│               CPCV/PBO · Diebold-Mariano · per-class report                   │
│  Economic:    Indian-cost-accurate backtest (STT, stamp, GST, brokerage,      │
│               slippage) · Sharpe · MaxDD · regime-split results               │
│  Diagnostic:  SHAP global + per-prediction · worst-20 confident misses ·      │
│               confusion matrix by sector/regime/volatility bucket             │
│  Ledger:      every prediction logged, outcome appended when resolved         │
│                               │                                               │
│                               ▼                                               │
│         GRAPHS: rolling accuracy · confidence-bucket accuracy ·               │
│         reliability diagram · confusion matrix · equity curve ·               │
│         drawdown · correct-vs-wrong bar · SHAP summary                        │
│                               │                                               │
│                               ▼                                               │
│         FINAL CONSOLE OUTPUT: "Correct N / Total M  (X.XX%)"                  │
│                               "Majority baseline:  (Y.YY%)"                   │
│                               "Edge over baseline: (+/-Z.ZZ pp)"              │
└──────────────────────────────────────────────────────────────────────────────┘

         ╔══════════════════════════════════════════════════════════╗
         ║  CROSS-CUTTING: CHECKPOINT / RESUME AT EVERY STAGE       ║
         ║  per-symbol parquet · step state JSON · Optuna SQLite ·   ║
         ║  LightGBM booster checkpoint + fingerprint manifest ·     ║
         ║  news cursor per source · atomic writes everywhere        ║
         ╚══════════════════════════════════════════════════════════╝
```

---

## 3. What Is Already Built

Verified by reading the actual code in `src/stockml/`. This is genuinely substantial — roughly 60–70% of the skeleton exists and works.

### 3.1 Working and correct

| Component | File | Notes |
|---|---|---|
| CLI + pipeline orchestration | `main.py`, `pipeline.py` | 12 commands, freshness-aware step skipping |
| Price fetch | `data/fetch_data.py` | Per-symbol parquet, staleness refresh |
| Macro fetch | `data/macro.py` | 8 series via yfinance |
| Feature engineering | `features/feature_engineering.py` | ~110 features: returns, SMA/EMA ratios, volatility, RSI, MACD, ATR, Bollinger, OBV, relative volume, fractional differentiation |
| **Macro point-in-time shift** | `feature_engineering.py` | `series.shift(1)` on every macro column — **correct**, avoids the classic overnight-leak trap |
| Calendar/fiscal context | `features/context.py` | Indian fiscal year (Apr–Mar), quarter/month cyclical encodings, earnings-season proxy |
| **Triple-barrier labeling** | `targets/triple_barrier.py` | **Correctly enters at `t+1` open**, vol-scaled barriers, deterministic both-barriers-touched rule |
| Cross-sectional ranking | `features/cross_sectional.py` | Correctly written groupby-timestamp percentile ranks — **dormant only because config has 1 symbol** |
| Purged/embargoed splits | `validation/` | `time_splits.py`, `walk_forward.py` |
| LightGBM training | `models/lightgbm_train.py` | Optuna study in SQLite, class weighting, early stopping |
| Linear baseline | `models/baseline.py` | Logistic regression |
| **Multiclass calibration** | `evaluation/metrics.py` | `apply_temperature()` — temperature scaling on log-probs, **preserves ΣP=1 correctly** |
| Metrics | `evaluation/metrics.py` | accuracy, bal-acc, MCC, macro P/R/F1, confusion matrix, OvR AUC, directional accuracy + coverage |
| **Majority baseline reporting** | `evaluation/evaluate.py` | Computes and records `majority_baseline_accuracy` — the honest comparison exists |
| Event backtest | `backtest/backtest.py` | No-overlapping-positions rule, gross exposure normalization, per-side costs |
| SHAP diagnostics | `evaluation/evaluate.py` | Feature importance + summary plot |
| Drift monitor | `monitoring/` | PSI computation framework |
| Checkpoint/resume | `utils/state.py`, `utils/io.py` | Atomic writes, step state JSON, booster checkpoint with dataset+param fingerprint |
| Bootstrap scripts | `start_run.py/.sh/.bat` | Venv creation, dependency repair, editable install, full pipeline run |
| FastAPI inference | `api/server.py` | `/health`, `/predictions`, `/predictions/{symbol}` |
| News infrastructure | `data/news.py`, `news_fetch.py`, `finbert.py` | Provider interface + FinBERT scorer + aggregation — **code exists, never populated** |
| News features | `features/news_features.py` | count, sources, sentiment mean/std, positive/negative, relevance, rolling 1/3/5/10/20d windows, volume z-score, negative shock |
| Universe provider | `data/universe.py` | Point-in-time constituent loader — **interface exists, no data file** |

### 3.2 Measured results from the last run (`results/metrics.json`)

```
test_start:                      2023-12-20
n_test:                          670
accuracy:                        0.4910
majority_baseline_accuracy:      0.5104   ← MODEL IS BELOW THIS
balanced_accuracy:               0.3466
MCC:                             0.0442
ROC-AUC (OvR macro):             0.5219
directional_accuracy:            0.5133  @ 99.4% coverage
FLAT class:                      25 actual, 4 predicted, 0 correct (F1 = 0.0)
backtest Sharpe:                 -1.78
backtest total return:           -5.06%
gross_return_sum:                 0.0077
cost_paid:                        0.0594   ← costs are 7.7× gross edge
```

**Read this honestly:** the current model does not beat always-predicting-the-majority-class. AUC 0.52 is near coin-flip. This is not a failure of the code — it is the expected result of training on **one instrument** with **zero news data** and a **dead FLAT class**. The fixes are in Section 14.

---

## 4. Data Layer — Sources, Storage, Schema

### 4.1 Free Indian data sources

| Data | Source | Cost | History | Notes |
|---|---|---|---|---|
| Daily OHLCV | `yfinance` (`.NS` suffix) | Free | 15–20y | Primary. Rate-limit with chunked batches |
| Daily OHLCV (backup) | NSE Bhavcopy CSV archive | Free | 20y+ | Official; use to cross-validate yfinance and fill gaps |
| Corporate actions | NSE corporate actions API / bhavcopy | Free | Long | **Critical** for split/bonus adjustment verification |
| Index constituents | NSE index factsheet archives + press releases | Free but manual | Partial | Point-in-time membership requires manual/scripted assembly — see §6.2 |
| Corporate announcements | NSE + BSE announcement APIs | Free | ~10y | **Highest-value news source.** Official, timestamped, entity-tagged by definition |
| Results calendar | NSE board meeting / results calendar | Free | ~10y | Gives forward-known event dates |
| News (recent) | GDELT DOC 2.0 API | Free, no key | ~3 months rolling | Good for live/daily operation |
| News (history) | GDELT BigQuery `gdeltv2` | Free tier quota | 2015+ | Main historical news backfill route |
| News (RSS) | Economic Times, Moneycontrol, Mint, Business Standard | Free | Live only | No history; use for daily ingestion going forward |
| News sentiment | Alpha Vantage `NEWS_SENTIMENT` | Free key | ~2022+ | 25 calls/day free tier — supplementary only |
| Macro India | `yfinance`: `^NSEI`, `^NSEBANK`, `^INDIAVIX`, `INR=X`, `CL=F`, `GC=F` | Free | 15y+ | |

### 4.2 Honest constraint: news history is not free for 16 years

**You will not get 16 years of Indian financial news for free.** Realistic coverage:

| Period | Price data | News coverage | Strategy |
|---|---|---|---|
| 2010–2015 | Full | Essentially none | Price/macro/cross-sectional only |
| 2015–2022 | Full | GDELT BigQuery (moderate) | Partial news features |
| 2022–present | Full | GDELT + AlphaVantage + RSS + NSE/BSE announcements | Full news features |

**Design consequence — this is important:** add an explicit `news_available` flag feature (0/1) plus `news_coverage_score` (articles in trailing 20d). The model then learns *"when news coverage exists, weight it; when absent, ignore it"* instead of silently treating "no news" as "neutral news." Without this flag, 2010–2015 rows poison the news features by making "no news" look like a real signal.

Also: **never impute sentiment.** Missing must stay structurally distinguishable from neutral. The existing code fills missing with `0.0`, which is indistinguishable from genuinely neutral sentiment — see Bug B-07.

### 4.3 Storage: DuckDB + Parquet

Free, local, zero-server, and reads parquet natively — ideal for this project.

```
data/
├── raw/
│   ├── prices/           {SYMBOL}.parquet          # per symbol, resumable
│   ├── corp_actions/     {SYMBOL}.parquet
│   ├── announcements/    {SYMBOL}.parquet          # NSE/BSE official
│   ├── news/             {SYMBOL}.parquet          # company-scoped
│   │   ├── _market/      market.parquet            # market-wide news
│   │   └── _group/       {GROUP_ID}.parquet        # group-level news
│   └── macro.parquet
├── reference/
│   ├── universe_history.json     # date → [symbols]  (point-in-time)
│   ├── entity_map.json           # alias → symbol
│   ├── group_map.json            # group → [symbols] with date ranges
│   ├── sector_map.json           # symbol → sector, with date ranges
│   └── trading_calendar.parquet  # NSE sessions + holidays
├── features/             {SYMBOL}_features.parquet
├── processed/            {SYMBOL}_labeled.parquet
└── warehouse.duckdb      # views over the parquet files for analysis
```

### 4.4 Core schema — point-in-time discipline

Every ingested row carries provenance:

```sql
-- news_raw
article_id        TEXT PRIMARY KEY   -- stable hash of (url + title)
published_utc     TIMESTAMP          -- ACTUAL publication time, not date
ingested_at       TIMESTAMP          -- when WE fetched it
source            TEXT
url               TEXT
title             TEXT
description       TEXT
language          TEXT
raw_payload       JSON

-- news_scored  (derived, recomputable)
article_id        TEXT
scope             TEXT      -- 'company' | 'group' | 'sector' | 'market' | 'ignore'
symbol            TEXT      -- fanned-out target symbol (NULL for market scope)
group_id          TEXT
sector            TEXT
relevance         DOUBLE    -- 0..1
event_type        TEXT      -- see §5.4 taxonomy
finbert_label     TEXT
finbert_score     DOUBLE    -- -1..+1
decision_date     DATE      -- FIRST session this article may influence (see §5.3)
```

**The one unbreakable rule:**

> A feature row for `(symbol, date t)` may only read source rows whose
> `published_utc <= t @ 15:30 IST`. Everything else is a leak.

---

## 5. The Hard Part — News & Context Engineering

This is the section that determines whether the project produces something meaningfully better than a price-only model. It is also where most of the engineering risk lives.

### 5.1 The four scopes of market-moving information

Not all news attaches to one ticker. The pipeline must classify and route each article:

| Scope | Example | Routing | Feature prefix |
|---|---|---|---|
| **Company** | "Infosys wins $1.5bn deal" | → `INFY.NS` only | `news_co_*` |
| **Group** | "SEBI probes Adani Group entities" | → *all* Adani-group members | `news_grp_*` |
| **Sector** | "IT sector hit by US visa rules" | → all IT-sector symbols | `news_sec_*` |
| **Market** | "RBI hikes repo rate 50bps" | → every symbol, same value | `news_mkt_*` |

The current code only handles company scope — `aggregate_news()` filters `frame["symbol"] == symbol` and drops everything else. Group and market news are silently discarded. **This is the single biggest missed opportunity in the current news design** (Bug B-06).

### 5.2 Entity resolution — mapping text to symbols

Build `reference/entity_map.json` and `reference/group_map.json`.

```jsonc
// entity_map.json — alias → canonical symbol
{
  "infosys":            "INFY.NS",
  "infosys ltd":        "INFY.NS",
  "infy":               "INFY.NS",
  "reliance industries":"RELIANCE.NS",
  "ril":                "RELIANCE.NS",
  "adani ports":        "ADANIPORTS.NS",
  "adani ports and sez":"ADANIPORTS.NS",
  "apsez":              "ADANIPORTS.NS"
}
```

```jsonc
// group_map.json — business groups, WITH date ranges (memberships change)
{
  "ADANI": {
    "display": "Adani Group",
    "aliases": ["adani group", "adani conglomerate", "gautam adani"],
    "members": [
      {"symbol": "ADANIENT.NS",   "from": "2010-01-01", "to": null},
      {"symbol": "ADANIPORTS.NS", "from": "2010-01-01", "to": null},
      {"symbol": "ADANIGREEN.NS", "from": "2018-06-18", "to": null},
      {"symbol": "ADANIPOWER.NS", "from": "2010-01-01", "to": null},
      {"symbol": "AWL.NS",        "from": "2022-02-08", "to": null},
      {"symbol": "ACC.NS",        "from": "2022-09-16", "to": null},
      {"symbol": "AMBUJACEM.NS",  "from": "2022-09-16", "to": null}
    ]
  },
  "TATA": {
    "display": "Tata Group",
    "aliases": ["tata group", "tata sons"],
    "members": [
      {"symbol": "TCS.NS",       "from": "2010-01-01", "to": null},
      {"symbol": "TATAMOTORS.NS","from": "2010-01-01", "to": null},
      {"symbol": "TATASTEEL.NS", "from": "2010-01-01", "to": null},
      {"symbol": "TITAN.NS",     "from": "2010-01-01", "to": null}
    ]
  }
}
```

**Why date ranges matter:** ACC and Ambuja became Adani companies in Sept 2022. Fanning 2015 Adani news to ACC would be factually wrong *and* a subtle lookahead — you'd be encoding knowledge of a 2022 acquisition into 2015 features.

**Resolution algorithm:**

```
1. Normalize title+description (lowercase, strip punctuation)
2. Exact alias match against entity_map  → company scope, relevance 1.0
3. Alias match against group_map aliases → group scope, relevance 0.7
4. Sector keyword match                  → sector scope, relevance 0.4
5. Market keyword match (RBI, repo, inflation, budget, FII/DII, Nifty,
   Sensex, GST, crude, rupee)            → market scope, relevance 0.5
6. No match                              → scope='ignore', dropped
```

For company/group scope, apply a **title-vs-body weight**: a match in the headline gets full relevance; body-only match gets 0.5×. A ticker mentioned in a market-wrap listicle is much weaker evidence than one in the headline.

**Use NSE/BSE announcements as ground truth.** They arrive pre-tagged with the exact scrip code — no NLP needed, zero ambiguity, and they are the actual legally-mandated disclosure channel. Treat `announcement_*` features as a separate, higher-confidence family from `news_*` features.

### 5.3 The 24-hour problem — mapping publication time to a tradeable session

News arrives at all hours. NSE trades 09:15–15:30 IST. The mapping from "published at X" to "first session this can influence" is the most error-prone part of the whole pipeline.

```
NSE trading day t:  09:15 ─────────────────────► 15:30
                      │                            │
                      │                            │
  ┌───────────────────┴────────────────────────────┴──────────────────────┐
  │                                                                       │
  │  Published 15:30 (t-1) → 15:30 (t)   →  usable for the DAY t decision │
  │  Published after 15:30 (t)           →  first usable for DAY t+1      │
  │                                                                       │
  └───────────────────────────────────────────────────────────────────────┘

decision_date(article) = first NSE session s such that
                         published_IST <= s @ 15:30
```

**Concretely:**

| Published (IST) | Decision date | Reasoning |
|---|---|---|
| Mon 10:00 | Mon | Within Monday's session, before 15:30 cutoff |
| Mon 15:29 | Mon | Just makes Monday's cutoff |
| Mon 15:31 | Tue | After cutoff → first influences Tuesday |
| Mon 21:00 | Tue | Evening news → Tuesday |
| Tue 02:00 | Tue | Overnight → Tuesday (before Tuesday 15:30) |
| Sat 11:00 | Mon | Weekend → next trading session |
| Fri 16:00 (before Mon holiday) | Tue | Must use the trading calendar, not `+1 day` |

**This requires the real NSE trading calendar**, not calendar arithmetic. Indian markets have ~15 holidays/year plus occasional special sessions (Muhurat trading). Build `reference/trading_calendar.parquet` from NSE holiday lists.

**Current implementation is wrong here** (Bug B-05). `news_features.py` does:

```python
frame["date"] = pd.to_datetime(values).dt.tz_localize(None).dt.normalize()   # calendar-day floor
...
out["news_sentiment_mean"] = out["news_sentiment_mean"].shift(1)             # blunt 1-session shift
```

Two problems:
1. `tz_localize(None)` on UTC timestamps **discards the timezone without converting** — a UTC timestamp of `2024-03-05 19:00` is 00:30 IST on March 6, but this code files it under March 5.
2. Calendar-day flooring + `shift(1)` is *approximately* safe but throws away real information: news published Monday 10:00 IST is legitimately usable for Monday's decision, but this code defers it to Tuesday. You lose an entire day of signal on every intraday article.

**Correct implementation:**

```python
IST = ZoneInfo("Asia/Kolkata")
CUTOFF = time(15, 30)

def decision_date(published_utc: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    ist = published_utc.tz_convert(IST)               # CONVERT, never tz_localize(None)
    same_day_ok = ist.time() <= CUTOFF
    candidate = ist.normalize().tz_localize(None)
    if not same_day_ok:
        candidate += pd.Timedelta(days=1)
    # advance to the next actual NSE session
    idx = calendar.searchsorted(candidate, side="left")
    return calendar[idx]
```

Then aggregate by `decision_date` with **no additional `shift(1)`** — the shift is already baked into the decision-date logic. Applying both double-shifts and destroys a day of signal.

### 5.4 Event-type tagging — giving the model "why"

Sentiment alone is a thin signal. "Why did the price drop" is better answered by *event type*. Tag each article/announcement into a taxonomy, then feed one-hot/count features:

| Event type | Keywords / announcement subject | Typical horizon |
|---|---|---|
| `earnings_result` | results, Q1/Q2/Q3/Q4, profit, revenue, PAT, EBITDA | 1–5d |
| `earnings_guidance` | guidance, outlook, forecast | 1–10d |
| `order_win` | order, contract, deal, wins, bags, LOI | 1–5d |
| `mna` | acquisition, merger, stake, takeover, divest | 1–20d |
| `regulatory` | SEBI, RBI, CCI, NCLT, probe, penalty, notice, show-cause | 1–20d |
| `rating_change` | upgrade, downgrade, CRISIL, ICRA, Moody's, S&P | 1–10d |
| `management` | CEO, CFO, MD, resign, appoint, board | 1–10d |
| `capital_action` | buyback, dividend, bonus, split, rights, QIP, FPO | 1–10d |
| `debt_credit` | default, NPA, debt, refinanc, bond, repayment | 1–20d |
| `litigation` | court, lawsuit, arbitration, tribunal | 1–20d |
| `expansion` | capex, plant, factory, expansion, capacity | 5–20d |
| `macro_policy` | repo rate, inflation, CPI, GDP, budget, GST | market scope |
| `short_seller` | short seller, Hindenburg, fraud allegation | 1–20d |

**Why this matters for your Adani example:** a Hindenburg-type report is `short_seller` + `regulatory`, **group scope**, extreme negative sentiment. A price-only model sees an unexplained cliff. A model with group-scope event tags sees "group-level short-seller allegation flag = 1, group sentiment = -0.9" on every Adani stock simultaneously — which is exactly the context that explains the crash.

**Feature construction per scope and event type:**

```
news_{scope}_{event}_count_{1,3,5,10,20}d
news_{scope}_{event}_sent_{1,3,5,10,20}d
news_{scope}_count_z20              # abnormal news volume
news_{scope}_sent_shock             # today's sentiment minus 20d mean
news_{scope}_neg_ratio_{5,20}d
announcement_{event}_flag           # from NSE/BSE, higher confidence
days_since_last_{event}             # recency decay
news_available                      # 0/1 — coverage flag (see §4.2)
news_coverage_score                 # articles in trailing 20d
```

### 5.5 Incremental news fetch with cursors

News backfill is slow and rate-limited. It must be resumable per source per symbol.

```
data/raw/news/_cursors/{source}_{symbol}.json
{
  "source": "gdelt_bq",
  "symbol": "ADANIPORTS.NS",
  "backfilled_from": "2015-02-19",
  "backfilled_to":   "2021-07-30",
  "last_run_utc":    "2026-09-14T11:02:31Z",
  "articles_total":  4127,
  "status":          "partial"
}
```

Backfill runs **backwards** in monthly chunks from today; daily incremental runs fetch forward from `backfilled_to`. A crash resumes from the cursor, never re-fetching completed months. Same pattern already proven in `fetch_data.py` for prices.

### 5.6 FinBERT scoring on 6GB VRAM

- Model: `ProsusAI/finbert` (~440MB) or `yiyanghkust/finbert-tone`
- Batch size 32–64 at fp16 fits comfortably in 6GB
- **Score once, cache forever** — `finbert_score` is written into `news_raw` parquet keyed by `article_id`. Never re-score an article.
- Scoring is embarrassingly parallel and resumable: process only rows where `finbert_score IS NULL`
- This is already correctly implemented in `news_fetch.py` (it checks `missing = frame["finbert_score"].isna()`)

---

## 6. Universe Expansion — Single Index → NIFTY 500

### 6.1 Staged rollout

Do not jump straight to 500. Each stage must complete and be validated before the next.

| Stage | Universe | Purpose | Expected rows |
|---|---|---|---|
| 0 | `^NSEI` only (current) | — | ~3,900 |
| 1 | NIFTY 50 | Prove cross-sectional code activates | ~190,000 |
| 2 | NIFTY 200 | Sector diversity, group coverage | ~750,000 |
| 3 | NIFTY 500 | Full cross-sectional richness | ~1,900,000 |

**Why this is the highest-impact change available.** Cross-sectional rank features are the strongest generalization lever in the design, and they are *structurally impossible* with one symbol — ranking one instrument against a universe of one produces a constant. The code in `cross_sectional.py` is already correct; it produces nothing only because `config.yaml` lists a single symbol.

### 6.2 Point-in-time universe (survivorship bias)

Taking today's NIFTY 500 and backfilling to 2010 silently guarantees every stock in your training set survived to 2026. That inflates backtest returns in a way no amount of modeling rigor can undo.

Free assembly route:
1. Scrape NSE index rebalance press releases (semi-annual) from the NSE archive
2. Build `universe_history.json`: `{"2019-03-29": [...], "2019-09-27": [...]}`
3. Forward-fill between rebalance dates
4. Keep delisted symbols' price data up to their delisting date, flagged `tradable=false` afterward
5. Pre-inclusion history is usable for **feature warmup only**, never as labeled training rows

The loader already exists (`data/universe.py`, `UniverseProvider`) and correctly documents this limitation in its docstring. It just needs the data file.

**Realistic caveat:** assembling clean point-in-time NIFTY 500 membership for 16 years from free sources is genuinely laborious. Acceptable interim compromise: do it properly for NIFTY 50/100 (fewer changes, better-documented), and for the 200–500 tail accept static membership *with the bias explicitly documented in every report*. Do not let it be forgotten silently.

### 6.3 Execution realism — costs, circuits, liquidity

**Indian transaction costs (verified, current as of April 2026):**

| Segment | STT | Sides | Round-trip STT |
|---|---|---|---|
| Equity delivery | 0.10% | Buy **and** sell | **20 bps** |
| Equity intraday | 0.025% | Sell only | 2.5 bps |
| Equity futures | 0.05% | Sell only | 5 bps |

Plus: brokerage, exchange turnover fee, stamp duty, GST on brokerage+fees, SEBI fee, and bid-ask slippage. Realistic all-in delivery round trip: **~25–35 bps**.

The current config uses `commission 2 + slippage 5 + other 3 = 10 bps/side = 20 bps round trip`, which roughly matches STT alone and **under-prices everything else** (Bug B-04).

**This is why horizon matters more than accuracy.** At a 1-day horizon, gross edge per trade is a few bps against a 25–35bps hurdle — arithmetically unwinnable. At 5–20 days the same per-day edge accumulates while cost stays fixed per round trip. Current config `horizon_days: 5` is reasonable; consider 10.

**Reconciling "NIFTY 500 for data" with "costs kill small caps":**

> **Train on NIFTY 500. Trade only the liquid subset.**

Train the model on the full 500 — more rows, richer cross-section, better rank features. But apply hard masks at the *decision* layer so signals are only emitted on symbols that are actually tradeable:

```
TRADEABLE(symbol, t) requires ALL of:
  ✓ in F&O list OR 20d median turnover >= ₹25 crore
  ✓ close price >= ₹50
  ✓ NOT within 0.5% of daily circuit band on day t
  ✓ no trading halt / series suspension
  ✓ price data present and non-stale
```

**Circuit filters matter and are currently missing entirely.** When a stock hits the upper circuit, ask-side liquidity vanishes — you *cannot buy it*. The backtest will happily fill the order anyway and book a phantom profit. In the NIFTY 500 tail this happens often enough to materially inflate returns (Bug B-08).

---

## 7. Feature Engineering

### 7.1 Families

| Family | Count | Status | Examples |
|---|---|---|---|
| Price/technical | ~45 | ✅ built | `ret_{1,3,5,10,20,60}`, `sma_ratio_*`, `ema_ratio_*`, `volatility_*`, `rsi_14`, `macd*`, `atr_pct_14`, `bb_position`, `bb_width`, `range_pct`, `body_pct`, `gap_pct` |
| Fractional differentiation | 2 | ✅ built | `fracdiff_0.3`, `fracdiff_0.5` — stationarity while preserving memory |
| Volume | ~8 | ✅ built | `relative_volume_*`, `volume_z_*`, `obv`, `obv_z20` |
| **Cross-sectional** | ~15 | ⚠️ code ready, **dormant** | `ret_20_xs_rank`, `volatility_20_xs_rank`, `relative_volume_20_xs_rank` — **expand to 12–15 ranks once universe > 1** |
| Sector-relative | ~6 | ❌ pending | `ret_20_minus_sector`, `sector_xs_rank`, `sector_momentum` |
| Group-relative | ~4 | ❌ pending | `ret_5_minus_group`, `group_dispersion` — captures group-wide shocks |
| Macro (India) | ~40 | ✅ built, needs trim | `macro_nifty*`, `macro_vix*`, `macro_usd_inr*`, `macro_crude*`, `macro_gold*` |
| Regime | ~6 | ⚠️ partial | `vix_regime_high` ✅; pending: bull/bear flag, expiry-week flag, vol-regime bucket |
| Calendar | ~14 | ✅ built | Indian fiscal quarter, month/quarter cyclical, earnings-season proxy |
| **News/event** | ~60 | ⚠️ code ready, **0% populated** | See §5.4 |
| Liquidity/tradability | ~5 | ❌ pending | turnover z, circuit proximity, halt flag, `tradable` |

### 7.2 India-only cleanup

Per the India-first decision, **remove US series as features**: `macro_sp500*`, `macro_nasdaq*`, `macro_us10y*` (~15 columns). Keep `macro_vix` only if it's India VIX (`^INDIAVIX`), not CBOE VIX.

**Add instead:** `^NSEBANK` (Bank Nifty), NIFTY midcap/smallcap indices, India 10Y G-Sec yield if freely available.

This also removes the entire class of overnight-timing bugs Gemini flagged — no US session means no US-session alignment risk.

### 7.3 Point-in-time rules per family

| Family | Shift rule | Rationale |
|---|---|---|
| Price/technical/volume | **No shift** | Computed from day `t` close, which is known at 15:30. Entry is `t+1` open, so no leak. Current code is correct and documents this. |
| Macro | `shift(1)` | Publication/settlement lag. Already correct. |
| Cross-sectional ranks | **No shift** | Computed from same-day `t` closes across the universe, all known at 15:30 |
| News | `decision_date` logic (§5.3) | **No additional shift** — double-shifting loses a day |
| Calendar | No shift | Deterministic, known in advance |

---

## 8. Labels & Targets

### 8.1 Current implementation (keep it)

`targets/triple_barrier.py` is correct:
- Entry at `t+1` **open** — the execution-timing fix, already present
- Upper barrier `entry × (1 + u·σ)`, lower `entry × (1 - l·σ)`, σ = 20d realized vol
- Intraday high/low tested against barriers
- Both barriers touched on the same day → labeled ambiguous, exits at time-barrier close (deterministic, no side-picking)
- Time barrier at `horizon_days`, then vol-scaled flat band decides the class
- Emits `event_start`, `event_end`, `entry_price`, `exit_price`, `realized_return` for the backtest

### 8.2 The FLAT class is broken — fix required

Current config: `flat_band: 0.25` → 25 FLAT samples out of 670 (3.7%). The model predicted FLAT 4 times and got 0 right. Precision, recall, F1 all zero.

A class the model never predicts is not a class. Two options:

**Option A — widen the band (preferred):** run an ablation over `flat_band ∈ {0.25, 0.5, 0.75, 1.0}` and pick the value giving roughly 25–40% FLAT. Select on balanced accuracy and MCC, not raw accuracy.

**Option B — go binary + abstention:** drop FLAT entirely, train UP/DOWN, and produce "no signal" from the calibrated confidence threshold rather than from a learned class. Cleaner, and arguably a better match for how the signal is actually used.

Run Option A's ablation first; it's cheap and informative either way.

---

## 9. Models — Algorithms and Why

### 9.1 Phase 1 — the only models that ship initially

| Model | Algorithm | Role |
|---|---|---|
| **Primary** | LightGBM multiclass (`multiclass`, 3 classes) | The backbone |
| **Baseline** | Logistic regression / ElasticNet | Mandatory sanity anchor |

**Why LightGBM, specifically:**

- The data is **tabular and heterogeneous** — returns, ranks, binary flags, sentiment scores, categorical sector IDs. GBDT handles mixed types and scales natively with no normalization step.
- Markets have **conditional interactions**: momentum predicts continuation in calm regimes and reversal in volatile ones. A linear model cannot express "if volatility is low, trust momentum" without you hand-engineering every such interaction in advance. Trees find them automatically.
- **Mechanically:** gradient boosting is not one tree enumerating every if/else case. It builds many shallow trees *sequentially*, each fit to the residual errors of the ensemble so far — gradient descent in function space. That sequential error-correction is what captures interactions cheaply.
- Robust to outliers and non-Gaussian returns — a tree split doesn't care about a single wild day.
- Trains in minutes on CPU, so retraining is practical.
- **SHAP interpretability** — required for the "why is it wrong" diagnostics in §11.3.
- Strong regularization controls (`max_depth`, `min_child_samples`, `lambda_l1/l2`, `feature_fraction`, `bagging_fraction`) — essential given how noisy financial labels are.

**Why not SVM:** training is roughly O(n²)–O(n³) in rows — impractical at 1.9M rows, and it needs Platt scaling bolted on to produce probabilities at all.

**Why not a deep net as the primary:** with ~2,500–3,000 *independent* trading days (same-day rows across 500 stocks are highly correlated, so raw row count massively overstates independent evidence), a multi-million-parameter network memorizes rather than generalizes.

**Why one unified LightGBM rather than the A/B split from earlier drafts:** running "price-only" and "price+news" models and stacking both into a meta-learner creates severe multicollinearity — one model's feature set is a strict subset of the other's, so their outputs are 0.9+ correlated and the meta-learner mostly partitions noise. Use `feature_fraction 0.6–0.8` on one model and let the trees decide whether news splits improve the objective. Keep A-vs-B as an **ablation comparison** (`main.py ablation` already exists), never as two stack members.

**Suggested starting hyperparameters** (Optuna will refine):

```yaml
objective: multiclass
num_class: 3
metric: multi_logloss
learning_rate: 0.02          # low + many rounds
num_leaves: 31..127
max_depth: 5..9
min_child_samples: 100..500  # HIGH — noisy labels
feature_fraction: 0.6..0.8
bagging_fraction: 0.7..0.9
bagging_freq: 1
lambda_l1: 0..5
lambda_l2: 0..10
num_boost_round: 3000
early_stopping_rounds: 200
class_weight: balanced
```

### 9.2 Phase 2 — deferred, gated

**Do not build these until the Phase-1 LightGBM beats the majority-class baseline with a Diebold-Mariano-confirmed margin.** Adding models to a system that loses to "always predict the majority class" adds complexity, not accuracy.

| Model | Gate condition |
|---|---|
| Small TCN (<100k params), 30d window, stationary inputs only (returns/gaps/vol-z, never raw price) | Phase 1 beats majority baseline |
| Nonneg L2 logistic meta-learner over base probabilities | TCN independently beats baseline **and** correlation with LightGBM < 0.9 |

Meta-learner rules if it ever ships: train only on **out-of-fold** predictions; always compare against a plain unweighted average; **kill rule** — if the learned stack doesn't beat both the best single model and the simple average by a DM-confirmed margin, ship the simpler option.

**Why nonneg L2 logistic and not GBDT as the combiner:** with ~4 highly correlated inputs there is almost no real structure to find and enormous room to fit fold noise. A constrained linear combiner can only learn "how much to trust each model," which is the actual well-posed question.

### 9.3 Calibration

`apply_temperature()` in `evaluation/metrics.py` already does this correctly — it divides log-probabilities by a temperature and re-softmaxes, which preserves ΣP=1 by construction. This is the right multiclass approach; naive per-class Platt/isotonic would break the sum-to-one constraint.

Fit temperature on a **held-out calibration slice** that is neither train nor test. Report Expected Calibration Error and a reliability diagram before/after.

---

## 10. Training, Checkpointing, Resume

### 10.1 Validation scheme

**Purged, embargoed, chronological walk-forward** for training and out-of-sample simulation:

```
Fold 1: train [2010──2017] │purge│embargo│ test [2018]
Fold 2: train [2010──2018] │purge│embargo│ test [2019]
Fold 3: train [2010──2019] │purge│embargo│ test [2020]
Fold 4: train [2010──2020] │purge│embargo│ test [2021]
Fold 5: train [2010──2021] │purge│embargo│ test [2022]
                                            ↓
                      FINAL SEALED HOLDOUT: [2023──present]
                      touched ONCE, at the very end
```

- **Purge** ≥ `horizon_days` — triple-barrier labels span multiple days, so training rows whose event window overlaps the test period must be removed
- **Embargo** ≥ 5 days after each test fold before training resumes
- Current config `purge_days: 5, embargo_days: 5` with `horizon_days: 5` is the minimum acceptable; prefer `purge = horizon + 2`

**CPCV is for PBO estimation only — never for training.** Walk-forward is strictly chronological; CPCV recombines time blocks non-chronologically. They answer different questions and both belong, in different roles. There is no conflict as long as roles stay separate.

### 10.2 Checkpoint & resume — already strong, extend it

Already working:

| Stage | Mechanism |
|---|---|
| Price fetch | Per-symbol parquet + staleness check |
| Features | Per-symbol parquet + mtime comparison vs raw+news |
| Labels | Per-symbol processed parquet |
| Optuna | SQLite study at `checkpoints/optuna.db` — completed trials survive restarts |
| LightGBM | `lightgbm_final_checkpoint.txt` + JSON manifest with **dataset fingerprint + param fingerprint + feature count**, so incompatible checkpoints are never resumed; `init_model` continues after a crash |
| Step state | `artifacts/state/*.json` per stage |
| Atomic writes | `utils/io.py` — temp file + replace, everywhere |

Verified in `VERIFICATION.txt`: a simulated crash at iteration 5 resumed correctly and completed iteration 12.

To add:

| Stage | Needed mechanism |
|---|---|
| News backfill | Per-source-per-symbol cursor JSON (§5.5) |
| FinBERT scoring | Already partial — extend so partial batches flush every N articles |
| Evaluation | Checkpoint SHAP computation (slow on 1.9M rows — sample 50k rows) |
| Backtest | Persist per-fold results so a crash doesn't re-run everything |

### 10.3 Training the expanded universe

At NIFTY 500 × 16 years ≈ 1.9M rows × ~200 features:

- Memory: use `float32`, categorical dtype for `symbol`/`sector`, LightGBM `Dataset` with `free_raw_data=True`
- Expect 15–40 min per full fit on CPU; Optuna 50 trials × 5 folds ≈ overnight
- GPU LightGBM is possible but often not faster at this scale — CPU is fine
- The RTX 3050 is for FinBERT scoring and (later) the TCN, not for GBDT

---

## 11. Evaluation, Backtesting, Statistical Testing

### 11.1 Tier 1 — Statistical correctness

| Metric | Status | Note |
|---|---|---|
| Accuracy | ✅ | **Always shown beside majority baseline** |
| **Majority-class baseline** | ✅ | The only accuracy comparison that means anything |
| Balanced accuracy | ✅ | |
| MCC | ✅ | Robust to imbalance |
| Macro P/R/F1 + per-class report | ✅ | Catches dead classes like the current FLAT |
| ROC-AUC (OvR macro) | ✅ | The primary ranking metric |
| Directional accuracy + coverage | ✅ | Both reported, correctly separated |
| **Calibration / ECE** | ⚠️ | Temperature scaling exists; ECE + reliability diagram need adding |
| **CPCV + PBO** | ❌ | Missing — needs porting from the earlier `v2/` project |
| **Diebold-Mariano** | ❌ | Missing — required for every model-vs-baseline claim |

**The default reported comparison must be against the majority baseline.** The current `lightgbm_minus_baseline_accuracy` field compares against the *linear* baseline (0.4522) and shows `+0.0388`, which reads as a win while the model is actually **below** the majority baseline (0.5104 vs 0.4910). That framing is how a losing model looks like a winning one (Bug B-03).

### 11.2 Tier 2 — Economic correctness

- Event backtest with no overlapping positions per symbol ✅ built
- **Indian-accurate cost model** ❌ — segment-aware STT + stamp + GST + brokerage + slippage
- **Circuit/liquidity masking** ❌ — see §6.3
- **Regime-split results** ❌ — Sharpe computed separately in bull / bear / high-vol / low-vol windows. A model that only works in one regime is regime-lucky, not working.
- Benchmarks: buy-and-hold NIFTY, always-majority-class, random signals at matched turnover

### 11.3 Tier 3 — Diagnostics ("why is it wrong")

- Global SHAP importance ✅ built
- **Per-prediction SHAP** ⚠️ — needed for the worst-20 review
- **Worst-20 confident-but-wrong report** ❌ — after every run, auto-pull the 20 highest-confidence wrong predictions with SHAP breakdown, news context for that date, and realized return. This is the single most useful diagnostic artifact in the whole system.
- **Confusion matrix sliced by sector / regime / volatility bucket** ❌ — one aggregate matrix hides "always wrong right before earnings"

### 11.4 Tier 4 — Prediction ledger

Persistent table, append-only, the single source of truth for every graph:

```sql
prediction_id   TEXT PRIMARY KEY
symbol          TEXT
decision_date   DATE
model_version   TEXT
p_down p_flat p_up  DOUBLE
predicted_class TEXT
confidence      DOUBLE
signal_emitted  BOOLEAN        -- FALSE when NO-SIGNAL / masked
mask_reason     TEXT           -- circuit / illiquid / low_conf / NULL
realized_class  TEXT           -- filled when the event resolves
correct         BOOLEAN
realized_return DOUBLE
resolved_at     TIMESTAMP
```

Never recompute "was it right" from scratch — always join against the ledger.

---

## 12. Visualization & Reporting

### 12.1 Required graphs

| Graph | File | Status | Shows |
|---|---|---|---|
| Correct vs wrong bar chart | `prediction_correctness.png` | ✅ built | Total right vs wrong |
| Rolling accuracy | `rolling_accuracy.png` | ✅ built | Accuracy over time (7/30/90d) — **must overlay the majority baseline as a horizontal line** |
| Confusion matrix | `confusion_matrix.png` | ✅ built | Per-class error structure |
| Confidence-bucket accuracy | `confidence_accuracy.csv` | ⚠️ CSV only | **Needs a chart** — accuracy per confidence bin. This is where real edge shows up |
| Equity curve | `equity_curve.png` | ✅ built | Net cumulative return |
| Drawdown | `drawdown.png` | ✅ built | |
| SHAP summary | `shap_summary.png` | ✅ built | |
| **Reliability diagram** | `calibration.png` | ❌ pending | Predicted vs realized probability |
| **Per-class ROC / PR curves** | `roc_pr.png` | ❌ pending | Ranking quality per class |
| **Regime-split performance** | `regime_performance.png` | ❌ pending | Bars per regime |
| **News coverage over time** | `news_coverage.png` | ❌ pending | Makes the 2010–2015 coverage gap visible |

### 12.2 Required final console output

Every full run must end with exactly this, so the headline number can never be read without its baseline:

```
══════════════════════════════════════════════════════════
  FINAL TEST RESULT
══════════════════════════════════════════════════════════
  Test period            : 2023-12-20 → 2026-09-15
  Total test samples     : 670

  CORRECT PREDICTIONS    : 329  /  670        (49.10%)
  WRONG PREDICTIONS      : 341  /  670        (50.90%)

  ── Reference baselines ─────────────────────────────────
  Majority-class baseline:                     (51.04%)
  Linear baseline        :                     (45.22%)

  EDGE OVER MAJORITY     :                     (-1.94 pp)   ✗ BELOW BASELINE

  ── Signal-only (confidence >= 0.55) ────────────────────
  Signals emitted        : 141  /  670        (21.04% coverage)
  CORRECT                :  73  /  141        (51.77%)

  ── Ranking & calibration ───────────────────────────────
  ROC-AUC (OvR macro)    : 0.5219
  MCC                    : 0.0442
  Balanced accuracy      : 0.3466
  ECE                    : n/a

  ── Economics (after Indian costs) ──────────────────────
  Net Sharpe             : -1.78
  Net total return       : -5.06%
  Gross return           :  0.77%
  Costs paid             :  5.94%   ← 7.7× gross
══════════════════════════════════════════════════════════
```

The `EDGE OVER MAJORITY` line with an explicit ✓/✗ is the single most important line in the entire system.

---

## 13. Bootstrap Scripts & One-Command Run

Already built: `start_run.py`, `start_run.sh`, `start_run.bat`. Current behavior — create venv if absent, verify dependency versions, install/repair missing or incompatible deps, install project editable, run full pipeline with the venv interpreter (no manual activation needed).

Required additions:

```
start_run.py  [--stage STAGE] [--force] [--universe {nifty50,nifty200,nifty500}]
              [--skip-news] [--report-only]

Flow:
 1. Check Python >= 3.11                                     ✅ built
 2. Create .venv if missing                                  ✅ built
 3. Verify + repair dependencies                             ✅ built
 4. Editable install                                         ✅ built
 5. Verify reference data present (entity_map, group_map,
    sector_map, trading_calendar, universe_history)          ❌ add
 6. Warn loudly if news coverage < 20%                       ❌ add
 7. Run pipeline stages with resume                          ✅ built
 8. Print the §12.2 final result block                       ❌ add
 9. Exit code 0 if model beats majority baseline, 1 if not   ❌ add
```

Step 9 makes "did it actually work" machine-checkable rather than a judgement call.

---

## 14. Task Tracker — Done / Pending / Bugs

Legend: ✅ done · 🟡 partial · ❌ not started · 🐛 bug

### 14.1 BUGS — fix these first

---

#### 🐛 B-01 — Config trains on a single symbol
**Severity:** CRITICAL · **File:** `config/config.yaml`
**Problem:** `symbols: [^NSEI]` — one instrument, ~3,900 rows. Cross-sectional features are structurally impossible (rank of one thing against itself is constant). Confirmed by PSI 0.0 on `ret_20_xs_rank`, `volatility_20_xs_rank`, `relative_volume_20_xs_rank`.
**Fix:** Replace with the NIFTY 50 constituent list first, validate, then expand to 200, then 500. Add `universe_tier` to config so tier is switchable.
**Verify:** `ret_20_xs_rank` in `monitoring.json` has non-zero PSI and non-constant values.
**Status:** ✅ DONE (2026-09-15) — 50-symbol Sept-2026 list in `config.yaml` + `universe_tier: nifty50` (static, bias documented per §6.2 interim). 188k rows fetched. 13 `*_xs_rank` families live and non-constant.

---

#### 🐛 B-02 — FLAT class is dead
**Severity:** HIGH · **File:** `config/config.yaml`, `targets/triple_barrier.py`
**Problem:** `flat_band: 0.25` → 25/670 FLAT samples (3.7%). Model predicted FLAT 4 times, got 0 right. Precision/recall/F1 all 0.0.
**Fix:** Ablate `flat_band ∈ {0.25, 0.5, 0.75, 1.0}`, choose the value producing 25–40% FLAT, select on balanced accuracy + MCC. Alternative: binary UP/DOWN with confidence-based abstention.
**Verify:** FLAT recall > 0.15 and FLAT F1 > 0.
**Status:** ✅ DONE with deviation (2026-09-15) — `evaluation/flat_band_ablation.py` ran the full sweep on 183k labels: FLAT share 5.2% → 7.2% across the range, so 25–40% is **structurally unreachable** (barrier-touch exits bypass the band; it only binds on time-barrier exits). Selected `flat_band=1.0`. The verify itself is MET: FLAT recall 0.203, F1 0.214 (was 0/0/0) on 12.3k FLAT samples. Option B (binary) stays a follow-up if FLAT degrades.

---

#### 🐛 B-03 — Misleading baseline comparison in reports
**Severity:** HIGH · **File:** `evaluation/evaluate.py`
**Problem:** `lightgbm_minus_baseline_accuracy` compares against the *linear* baseline (0.4522), reporting `+0.0388` — which reads as a win. The model is actually **below** the majority baseline (0.4910 vs 0.5104). Both numbers exist in the JSON; the wrong one is headlined.
**Fix:** Add `lightgbm_minus_majority_accuracy` and make it the primary reported field. Keep the linear comparison as secondary. Emit an explicit `beats_majority_baseline: true/false`.
**Verify:** `metrics.json` contains `beats_majority_baseline` and the console block from §12.2 prints the ✓/✗ line.
**Status:** ✅ DONE (2026-09-15) — `lightgbm_minus_majority_accuracy` + `beats_majority_baseline` added in `evaluation/evaluate.py`; linear delta kept as secondary; rolling chart overlays majority baseline; `metrics.json` backfilled (edge −1.94 pp, beats=false).

---

#### 🐛 B-04 — Transaction costs under-priced for Indian equities
**Severity:** HIGH · **File:** `config/config.yaml`, `backtest/backtest.py`
**Problem:** `2 + 5 + 3 = 10 bps/side` ≈ STT alone for delivery equity. Ignores stamp duty, GST, exchange fees, and realistic spread. Real all-in delivery round trip is ~25–35 bps.
**Fix:** Implement segment-aware costs — delivery: 0.1% STT both sides + stamp 0.015% buy + GST 18% on (brokerage+fees) + slippage; futures: 0.05% STT sell-side only. Add `segment: delivery|futures` to config.
**Verify:** A 5-day round trip in the backtest shows total cost ≈ 25–35 bps for delivery.
**Status:** ✅ DONE (2026-09-15) — `round_trip_cost_rate()` in `backtest.py`, `segment: delivery` in config. Verified: 31.5 bps round trip. Honest economics now visible: gross +38.0% vs costs 102.5% over the test window (turnover, not model error, is the binding constraint per §6.3).

---

#### 🐛 B-05 — News timezone handling drops a day, double-shift loses signal
**Severity:** HIGH · **File:** `features/news_features.py`
**Problem:** Two defects.
(a) `_day_floor()` calls `.dt.tz_localize(None)` on UTC timestamps — this **discards the timezone without converting**. A UTC time of `2024-03-05 19:00` is `00:30 IST on March 6` but gets filed under March 5.
(b) Calendar-day flooring plus an additional `.shift(1)` double-defers: news published Monday 10:00 IST is legitimately usable for Monday's decision but is pushed to Tuesday.
**Fix:** Implement `decision_date()` per §5.3 — `tz_convert("Asia/Kolkata")`, compare against the 15:30 cutoff, advance to the next **NSE session** using `trading_calendar.parquet`. Aggregate by `decision_date` and **remove the extra `shift(1)`**.
**Verify:** Unit test — article at `2024-03-05 09:50 UTC` (15:20 IST) → decision_date `2024-03-05`; article at `2024-03-05 10:10 UTC` (15:40 IST) → `2024-03-06`; Friday 16:00 IST before a Monday holiday → Tuesday.
**Status:** 🟡 CODE DONE (2026-09-19, `features/decision_date.py` + B-05 vectors green; `aggregate_news()` flip-over waits for U-ablation — no retrain triggered)

---

#### 🐛 B-06 — Group / sector / market news is silently discarded
**Severity:** HIGH · **File:** `features/news_features.py`
**Problem:** `aggregate_news()` filters `frame["symbol"] == symbol` and drops everything else. Group-level news (e.g. a short-seller report on an entire business group) and market-level news (RBI policy) never reach the model — exactly the events that explain the largest, most correlated price moves.
**Fix:** Implement the scope classifier and fan-out from §5.1–5.2. Produce separate feature families `news_co_*`, `news_grp_*`, `news_sec_*`, `news_mkt_*`. Build `entity_map.json`, `group_map.json`, `sector_map.json` with date-ranged memberships.
**Verify:** For an Adani-group-wide news date, every Adani member symbol shows identical non-zero `news_grp_*` values; for an RBI policy date, all symbols show identical `news_mkt_*`.
**Status:** ❌

---

#### 🐛 B-07 — Missing news imputed as neutral
**Severity:** MEDIUM · **File:** `features/news_features.py`
**Problem:** `.fillna(0.0)` everywhere makes "no news existed" indistinguishable from "news existed and was neutral." Given news coverage is ~0% before 2015, this teaches the model that a decade of history had genuinely neutral sentiment.
**Fix:** Add `news_available` (0/1) and `news_coverage_score` (trailing-20d article count). Keep the 0.0 fills but let the model condition on availability. Consider NaN + LightGBM's native missing handling instead.
**Verify:** `news_available` is 0 for pre-coverage dates and 1 afterwards; feature importance shows the model using it.
**Status:** ❌

---

#### 🐛 B-08 — No circuit-limit or liquidity masking
**Severity:** HIGH (once universe > NIFTY 50) · **File:** new `risk/tradability.py`
**Problem:** When a stock hits the upper circuit, ask liquidity is zero — you cannot buy. The backtest fills anyway and books a phantom profit. Frequent in the NIFTY 500 tail.
**Fix:** Implement `TRADEABLE(symbol, t)` per §6.3. Mask to NO-SIGNAL when within 0.5% of the circuit band, below the turnover floor, below ₹50, or halted. Log `mask_reason` in the ledger.
**Verify:** Backtest trade count drops and Sharpe changes measurably after masking; no trade exists on a day where the symbol closed at its circuit band.
**Status:** ❌

---

#### 🐛 B-09 — Drift monitor alarms on everything
**Severity:** MEDIUM · **File:** `monitoring/drift.py`, `monitoring/monitor.py`
**Problem:** PSI computed on raw price levels (`open`, `high`, `low`, `close` → PSI 10.7–11.0) against a 0.2 threshold. Prices trend over 16 years by nature, so ~100 features alert simultaneously and the monitor is ignored. A monitor that always fires is worse than none.
**Fix:** Compute PSI only on stationary transforms — returns, ratios, z-scores, ranks. Explicitly exclude raw level columns. Raise threshold to 0.25 and cap alerts to the top-20 SHAP features.
**Verify:** On healthy data, fewer than 5 features alert.
**Status:** 🟡 PARTIAL (2026-09-15) — level columns excluded, macro levels excluded (2nd pass), threshold 0.25, alerts capped to top-20 SHAP; live runs cut alerts ~100 → 16 → 12 → 11 on 50-symbol data. Remaining 11 are interpretable macro-regime shifts (gold vol/z, nifty_ret5, crude/vix transforms) plus atr/volatility — the monitor firing on a real regime, not noise. The <5 target is kept for review, not chased by over-tuning (Goodhart).

---

#### 🐛 B-10 — US macro features contradict the India-first decision
**Severity:** LOW (correctness), MEDIUM (noise) · **File:** `data/macro.py`, `config/config.yaml`
**Problem:** `sp500`, `nasdaq`, `us10y` are fetched and expanded into ~15 features. The project is now explicitly India-only. (Note: the existing `shift(1)` handling is *correct* and does not leak — this is a scope decision, not a bug in timing.)
**Fix:** Remove `sp500`, `nasdaq`, `us10y` from `DEFAULT_MACRO_SYMBOLS`. Add `^NSEBANK`, `^INDIAVIX`, NIFTY midcap/smallcap. Confirm `macro_vix` sources India VIX, not CBOE VIX.
**Verify:** No `macro_sp500*` / `macro_nasdaq*` / `macro_us10y*` columns in the feature parquet.
**Status:** ✅ DONE (2026-09-15) — `DEFAULT_MACRO_SYMBOLS` is now India-only (`nifty`, `banknifty=^NSEBANK`, `vix=^INDIAVIX`, `usd_inr`, `crude`, `gold`). Macro parquet refetched 2026-09-15; feature parquet rebuilt with zero US columns.

---

#### 🐛 B-11 — No point-in-time universe file → survivorship bias
**Severity:** HIGH (once universe > NIFTY 50) · **File:** `data/universe.py` + missing data
**Problem:** `UniverseProvider` correctly falls back to static symbols and documents the risk, but `universe_history.json` doesn't exist. Backfilling today's constituents to 2010 guarantees every training stock survived.
**Fix:** Scrape NSE index rebalance press releases, build `universe_history.json`, forward-fill between rebalances, keep delisted data with `tradable=false`.
**Verify:** `UniverseProvider.point_in_time_available == True`; a known 2018-delisted stock appears in 2017 training rows and not in 2019.
**Status:** ❌

---

#### 🐛 B-12 — Zero news data despite full news infrastructure
**Severity:** CRITICAL · **File:** data gap, not code
**Problem:** Every news feature has PSI exactly 0.0 — constant, never populated. `VERIFICATION.txt` confirms a live 16-year news backfill was never validated. The README states the platform never fabricates sentiment (correct behavior). So the "multimodal" system is currently a price-only model.
**Fix:** Build the news backfill per §5 — GDELT BigQuery historical, GDELT DOC for recent, NSE/BSE announcements as the high-confidence backbone, RSS for daily going forward. Implement per-source cursors.
**Verify:** `news_coverage_score` > 0 for > 60% of rows after 2018; `news_count` PSI non-zero.
**Status:** 🟡 INGESTION LIVE (2026-09-20) — NSE announcements 92,960 (back to 2004) + GDELT DOC 9,179 articles / 38 symbols / 90d window (all FinBERT-scored, cached) + RSS 103 rows/day (ET/Mint/MC, live-only) + bulk/block forward-only from 2026-09-20. AlphaVantage: key secured in git-ignored `.env`, quota guard (25/day, state-persisted) + 2y lookback fix live; day-one burst spent 25 calls on 16y-lookback windows (0 articles — fixed, resumes correctly tomorrow). GDELT DOC rate limit mapped (1 req/5s, escalating IP cooldown — 6s floor + backoff + retry in code, tested). Gaps: 12/50 GDELT symbols empty (re-fetch after cooldown), BigQuery 2015+ blocked on ADC (see N4). B-12 verify (coverage>60% post-2018, news_count PSI≠0) needs N6 feature rebuild — NOT done, no retrain triggered.
**N6 feature-wiring spec (spec only, no retrain per standing gates):** (1) replace `_day_floor`+`shift(1)` in `aggregate_news` with `decision_date()` (built+tested); (2) scope fan-out company/group/sector/market via maps (entity_map.json still missing — must be built first); (3) `news_available` + coverage-score flags, NaN + LightGBM native missing (B-07, never 0-fill); (4) families `news_co/grp/sec/mkt_*` per §5.4. Training stays behind the U-ablation gate.
**N6 BUILD DONE 2026-09-21, STOPPED COLD (no retrain).** Split store live: `features/stocks/` (137 pure-price cols, xs-free) + `features/news/` (21 cols: 8 news_co + 9 ann_co + 4 ev_co) + `features/market/` (4 cols) + materialized joined view (183 cols after cross-sectional) + `_join_log.json`. 52/52 green. Coverage honesty (RELIANCE post-2018): announcements 61%, events 81%, **GDELT text 0.1%** — B-12's >60% is met only via announcements/events; text news is a 90-day sliver until AlphaVantage/BigQuery-text.ipynb exist. U-ablation must be judged knowing text features are near-empty historically. Caught+fixed in build: `days_since` naming (NAN_OK-critical), stale-xs pollution in moved files, silent `except:pass` gap.
**N4 BigQuery 2015+ BLOCKED on auth 2026-09-20:** client lib installed, project `stock-news-backfill` set, but no Application Default Credentials on this PC and no `gcloud` CLI present (it is a system installer, not a pip package — cannot live in the venv). Unblocks with exactly one user action: install gcloud CLI + `gcloud auth application-default login`, OR create a service-account JSON (BigQuery Job User + Data Viewer) and point `GOOGLE_APPLICATION_CREDENTIALS` at it.
**N4 DONE with skew 2026-09-21 (auth completed by user):** billing truth probed first — events table 0.4TB/61col UNPARTITIONED (every query full-scans; GKG 21.94TB off-limits), so ONE single-shot IN-actor query ('IND' CAMEO code — 'IN' matched zero) instead of monthly chunks. Fixed two real bugs: OOM from `list()` (now page-streamed + incremental flush) and silent `except:pass` gap. Result: **88,798 events 2015–2026 across 11 symbols** (LT 22k, INFY 21.7k, TATASTEEL 12.5k, TMPV 11k, RELIANCE 10k, BEL 8.7k — govt-contract-heavy names; CAMEO codes 010/020/051… genuine). **39 symbols zero — real coverage skew, not a bug:** CAMEO event coding under-covers banks/FMCG/pharma (routine business isn't an "event"). Quota 329/800GB spent this month. Tone is GDELT-native (kept separate from FinBERT per standing rule).

---

### 14.2 PENDING FEATURES

| ID | Task | Priority | Depends on |
|---|---|---|---|
| F-01 | Expand universe to NIFTY 50 → 200 → 500 | P0 | B-01 | ✅ NIFTY 50 DONE (2026-09-15, ~188k rows); 200/500 pending |
| F-02 | Build `trading_calendar.parquet` (NSE sessions + holidays) | P0 | — | ✅ DONE (2026-09-19, 3,927 sessions 2010→2026 from NIFTY presence) |
| F-03 | Build `entity_map.json` / `group_map.json` / `sector_map.json` | P0 | — | 🟡 PARTIAL (group+sector maps live 2026-09-18; entity_map pending) |
| F-04 | Implement `decision_date()` news alignment | P0 | B-05, F-02 |
| F-05 | Implement news scope classifier + fan-out | P0 | B-06, F-03 |
| F-06 | NSE/BSE corporate announcement ingestion | P0 | F-02 | 🟡 INGESTION DONE (2026-09-19, NSE 92,960 records/50 symbols back to 2004, cursors; aggregation into features waits for U-ablation; BSE DEFERRED — no clean free JSON API, revisit if NSE-only U2 fails) |
| F-07 | GDELT BigQuery historical news backfill with cursors | P0 | B-12 |
| F-08 | Event-type taxonomy tagging | P1 | F-05 |
| F-09 | Indian-accurate segment-aware cost model | P0 | B-04 | ✅ DONE (2026-09-15, 31.5 bps delivery round trip) |
| F-10 | Circuit / liquidity tradability masking | P0 | B-08 |
| F-11 | Sector- and group-relative features | P1 | F-01, F-03 | ✅ DONE code-side (2026-09-18, maps + 8 relatives live) — weak SHAP (ranks 43–110), needs review |
| F-12 | Expand cross-sectional ranks to 12–15 features | P1 | F-01 | ✅ DONE (2026-09-15, 13 rank families) |
| F-13 | Regime features (bull/bear, vol bucket, expiry week) | P1 | — | ✅ DONE code-side (2026-09-18, regime_bull + expiry_week_flag) — near-unused by model (SHAP 100+), needs review |
| F-14 | Calibration: ECE metric + reliability diagram | P1 | — | ✅ DONE (2026-09-18, ECE 0.0424 <0.05, calibration.png) |
| F-15 | CPCV + PBO harness | P1 | — |
| F-16 | Diebold-Mariano test harness | P1 | — |
| F-17 | Prediction ledger table + resolution job | P1 | — |
| F-18 | Per-prediction SHAP + worst-20 report | P2 | — |
| F-19 | Sliced confusion matrices (sector/regime/vol) | P2 | F-01 |
| F-20 | Regime-split backtest | P2 | F-13 |
| F-21 | Missing graphs (calibration, ROC/PR, regime, coverage) | P2 | F-14 |
| F-22 | Final console result block + exit code | P1 | B-03 | ✅ DONE (2026-09-15) — `evaluation/final_report.py` + `report` command; `start_run.py` prints block and exits 1 below baseline |
| F-23 | `flat_band` ablation runner | P0 | B-02 | ✅ DONE (2026-09-15, `flat_band=1.0`, FLAT F1 0.214) |
| F-24 | DuckDB warehouse views over parquet | P2 | — |
| F-25 | Remove US macro, add Bank Nifty / India VIX / midcap | P1 | B-10 | ✅ DONE (2026-09-15, parquet verified clean) |

### 14.3 ALREADY DONE

✅ CLI + pipeline orchestration with freshness-aware skipping
✅ Per-symbol resumable price fetch
✅ Macro fetch (India-only since 2026-09-15: nifty/banknifty/India VIX/usd_inr/crude/gold)
✅ NSE bhavcopy delivery/turnover fetch since 2026-09-18 (sec_bhavdata_full, 2020+, per-symbol parquet + availability flag)
✅ ~110 engineered features including fractional differentiation
✅ Correct macro `shift(1)` point-in-time handling
✅ Indian fiscal calendar / earnings-season context
✅ Triple-barrier labeling with correct `t+1` **open** entry
✅ Cross-sectional ranking code (active since 2026-09-15: 13 rank families over NIFTY 50)
✅ Purged + embargoed walk-forward splits
✅ LightGBM training with Optuna in SQLite, class weighting, early stopping
✅ Logistic regression baseline
✅ Temperature-scaling multiclass calibration (ΣP=1 preserved)
✅ Full metric suite incl. majority baseline, directional accuracy + coverage
✅ Event backtest with no-overlap rule and exposure normalization
✅ SHAP global importance + summary plot
✅ PSI drift framework (needs B-09 fix)
✅ Atomic writes, step state, LightGBM checkpoint with fingerprint manifest
✅ Crash-resume verified by simulated crash test
✅ `start_run.py/.sh/.bat` bootstrap with venv + dependency repair
✅ FastAPI inference endpoints
✅ News/FinBERT infrastructure (code complete, data empty per B-12)
✅ 8 result graphs

### 14.4 Recommended execution order

```
WEEK 1 — Make the numbers honest
  B-03  majority-baseline reporting      ← do this first; everything else
  F-22  final console block                is judged against it
  B-09  fix drift monitor
  B-10 / F-25  India-only macro

WEEK 2 — Make the model learnable
  B-01 / F-01  universe → NIFTY 50
  F-12  expand cross-sectional ranks
  B-02 / F-23  flat_band ablation
  B-04 / F-09  real Indian costs
  ↳ CHECKPOINT: does it beat the majority baseline now?

WEEK 3–4 — Build the context layer
  F-02  trading calendar
  F-03  entity / group / sector maps
  B-05 / F-04  decision_date alignment
  B-06 / F-05  scope classifier + fan-out
  F-06  NSE/BSE announcements
  B-12 / F-07  GDELT historical backfill
  B-07  news availability flags
  F-08  event-type tagging

WEEK 5 — Scale and harden
  F-01  universe → 200 → 500
  B-11  point-in-time universe
  B-08 / F-10  circuit + liquidity masking
  F-11  sector/group-relative features
  F-13  regime features

WEEK 6 — Rigor
  F-14  calibration + ECE
  F-15  CPCV + PBO
  F-16  Diebold-Mariano
  F-17  prediction ledger
  F-18  worst-20 diagnostics
  F-19/20/21  sliced evaluation + graphs
```

### 14.5 Decision gates

Do not proceed past a gate until its condition is met.

| Gate | Condition | If it fails |
|---|---|---|
| **G1** after Week 2 | Model beats majority baseline on walk-forward folds | Do not build the news layer yet. Revisit labels, horizon, feature quality. |
**G1 verdict 2026-09-15: FAILED** — 46.41% vs majority 47.85% (edge −1.45 pp) on 37,292 test rows (NIFTY 50, 25 Optuna trials). Genuine progress inside the failure: AUC 0.5219 → **0.5599** (in the 0.55–0.62 target band), FLAT F1 0.0 → 0.214, signal-only 50.07% @ 20% coverage. Per the gate rule, Weeks 3–4 news work does NOT start until accuracy crosses majority — next: horizon/label review, confidence-threshold tuning, cost-aware turnover control.
**G1-recovery 2026-09-16 (zero new features — subtraction/reframing only):** (b) threshold sweep on frozen predictions: edge grows monotonically 0.55→0.75 (+2.2pp → +11.8pp), proving confidence ranks — adopted `confidence_threshold: 0.60` (8.4% coverage @ 51.54%); economics improved mechanically (Sharpe −2.65 → −1.58, costs 102% → 47%). (a) horizon-10 TESTED and LOST (edge −2.49pp, AUC 0.5520, MCC 0.015 — longer horizon skewed labels UP and halved independent events) — reverted to horizon 5. (d) PSI experiment: reference-window choice irrelevant (93 vs 91 alerts); the bug was deterministic calendar columns (PSI ~11) — excluded from monitoring + unit-tested; remaining 12 are macro-regime transforms, <5 text kept as-is. **Gate still closed** (−1.45 pp). 16/16 tests green.
**G1-attempt-2 2026-09-18 (price-only evidence, no news):** NSE `sec_bhavdata_full` delivery/turnover backfill 2020→today (50 symbols, ~1,740 rows each; pre-2020 delivery unavailable on free archive — turnover proxy covers 2010+, `delivery_available` flag); full `sector_map.json` + `group_map.json` with date ranges + F-11 relatives; F-13 `regime_bull` + `expiry_week_flag`; F-14 ECE 0.0424 (<0.05 ✓) + `calibration.png`. Full 25-trial retrain on 174k×136 matrix. Result: 46.62% vs 47.85% (**−1.24 pp**, +0.21pp vs freeze) — strict criteria FAILED on all three (headline<0; signal 50.18% @ 11.63% <52%; Sharpe −2.49 worse than −1.58). New turnover features ARE used (turnover_cr_proxy SHAP #4, delivery_per #21) and DOWN F1 rose 0.548→0.604, but UP recall collapsed 0.335→0.159 and threshold ranking flattened. **Gate still closed.** 21/21 tests green. Along the way fixed 3 real bugs the new data exposed: bhav CSV `skipinitialspace` parse, strict-mask row drop (test 37k→16k), predict NaN rejection vs LightGBM native missing.
**Issue #1 (UP→DOWN) D1/D2 2026-09-19:** `evaluation/up_autopsy.py` + `up-autopsy` command + `results/up_autopsy.json`/`worst_20.csv`. Findings: miss uniform 0.78–0.94 across all 15 sectors/8 groups/both regimes (cross-era ✓); UP OvR AUC 0.518 (ranking failure, not bias — per-class thresholds cut); labels symmetric (UP +1.41% vs DOWN −1.41% median — question healthy, reading faulty); delivery quintile strictly monotonic q1 0.787→q5 0.891. D2: per-fold per-class recall + top-15 into Optuna attrs + `fold_diagnostics.jsonl`; train-vs-test gap tripwire in code (>20pp MEMORIZATION FLAG, test>0.65 TOO-GOOD FLAG, stored in model_metadata.json).
**H1 (plain delivery×turnover) 2026-09-19: FAILED.** UP-AUC 0.518→0.5144 (bar was >0.53); dm_up −5.45 (p≈5e-8, worse than linear baseline); headline −1.29pp. UP recall rose 0.159→0.218 but ranking fell — redistribution, not learning. Reverted per one-at-a-time rule.
**H1b (CLV×delivery) 2026-09-19: FAILED + MEMORIZATION FLAG.** UP-AUC 0.5144→**0.5127** (worse monotonically across H1/H1b); dm_up −5.85 (p≈5e-9); train-vs-test gap **21.7pp > 20pp — model flagged, DISQUALIFIED regardless of headline**. Headline −0.80pp is a false positive (DOWN redistribution + memorization, UP ranking degraded) and must not be cited as progress. delivery_clv SHAP 22 (used, not helping). 27/27 tests green. **PAUSED per review:** uniform-everywhere miss + three failed feature rounds suggest a fundamental information gap for 5-day UP at 1σ from daily bars, not a feature gap. H1/H1b columns reverted; attempt-2 checkpoint restored (study 331cb0c3d8d4 intact, 25/25 trials, deterministic re-train reproduced −1.24pp, no flag). Flagged checkpoint replaced, must never be restored.
**Track F (Issue #2) 2026-09-19: ADOPTED as operating rule, G1 still closed.** Rationale recorded explicitly: per-class thresholds were REJECTED for UP (OvR AUC 0.518 — no ranking skill exists to convert; tuning would relocate a coin flip) and are CORRECT for FLAT (OvR AUC 0.622 best-of-three with F1 0.207 worst — ranking skill exists, decision boundary wrong). Same tool, opposite diagnosis — not the same idea twice. `evaluation/flat_threshold.py` + `flat-threshold` command: τ_flat swept on calibration slice only (winner 0.40, FLAT F1 0.1441 @ acc 0.4576 vs base 0.4516), applied once to test: headline 0.4662→**0.4714 (+0.52pp, edge −0.71pp)**, DM tau-vs-argmax stat +9.14 (p≈6e-20), FLAT F1 0.2069→0.2052 preserved, DOWN 0.6055 / UP 0.2443 no tradeoff, UP-AUC untouched 0.5181 (decision rule cannot move ranking, as predicted). Adopted into `config decision.flat_threshold` + `evaluate.py` + `predict.py` (serving matches judged rule). Trading set unchanged (rule binds below 0.60 confidence) so economics identical. 28/28 tests green.
**Track U ablation (company-scope announcement counts) 2026-09-19: FAILED on target metric, clean (no flag).** 9 features (counts 1/5/20d, recency, 4 event flags, availability) by decision_date, fresh 25-trial study on 174k×145. Result: UP-AUC 0.5181→**0.5140 (wrong direction)**; dm_up −6.72 (p≈2e-11, worse than linear); headline −0.71→−0.97pp; FLAT F1 0.2052→0.1896 (slight degradation). Tripwire: gap 17.6pp, NO flag — honest failure, not memorization. UP recall 0.219→0.318 is redistribution (precision flat), same false-positive shape as H1. Ann SHAP best 41 (weak). Counts alone do not answer UP. Next single variable if approved: **U2 scope fan-out** (group/sector/market routing per §5.1 — the actual Door B argument; counts were never its strong form), judged on UP-AUC again. 34/34 tests green.
**U2-D1 (read-only scope check) 2026-09-19: HOLD U2 training.** Joined maps × 92,873 dated announcements; UP-miss on active vs quiet days (all cross-era stable): group 0.6848 vs 0.6815 (Δ 0.003, nothing), sector 0.6734 vs 0.7043 (Δ −0.031, tiny), market-high 0.6863 vs low 0.6813 (nothing). UP-AUC on active subsets: sector 0.5185 vs quiet 0.5026 (+0.016, directionally favorable but an order of magnitude too small to lift aggregate UP-AUC by the needed +0.015); market-high 0.5002 (coin flip — high-activity days rank WORSE). Raw scope signal is ~null: no retrain justified. U2 training stays on hold.
**Redistribution pattern verdict (3 attempts):** UP-AUC across F/H1/H1b/U = 0.5181/0.5144/0.5127/0.5140, spread 0.0054 vs Hanley-McNeil SE ≈ 0.0030 (n_up 16,843) — all four within ~2 SE of a hard floor ≈ 0.515. Statistically indistinguishable from noise around a ceiling, not three independent misses. Price-side escalation for UP is paused; 5-day UP at 1σ from daily bars reads as information gap. Door C (redefine UP-side success) is now a legitimate question for UP only — not FLAT, not the project.
**Track I governance 2026-09-20:** `results/holdout_touches.jsonl` rebuilt as canonical ledger (19 touches: 12 backfilled campaign + 5 G3 + I0/I1 D1s; `evaluation/touch_ledger.py` backfill/log_touch, idempotent). Pre-registered 2×SE bars for all D1 deltas (`bucket_delta_report`, FAIL-closed on thin buckets).
**I0-D1a label grid 2026-09-20:** 12 cells (h∈{1,5,10,20}×mult∈{0.5,1.0,1.5}), cached `artifacts/label_grid.json`. Best balance h=1/mult=0.5 (37.5/29.6/32.9 — the only cell in the 25–40% FLAT band) with 1-day economics caveat; runner-up h=5/mult=0.5 (40.4/23.4/36.2, keeps 5-day economics). Horizons ≥10 saturate (±1pp vs h=5 — barriers touch anyway). Candidate cells queued, no ablation yet.
**I0-D1b earnings proximity 2026-09-20: FAIL-closed.** UP-miss 0–5d 0.8238 (n=613) vs 21+d 0.8377 (n=13,796): Δ=−0.014 vs bar 0.031. Wrong direction and insignificant — no PEAD edge in current predictions. No ablation.
**I1-D1 lead-lag 2026-09-20: FAIL-closed.** Turnover-quintile leaders per sector; laggard UP-miss on leader-up 0.8242 (n=3,055) vs leader-flat 0.8292: Δ=−0.005 vs bar 0.017. Information does not travel leaders→laggards for rallies (leader-down days miss least at 0.7785 — falls propagate, rallies don't; consistent with DOWN-working/UP-dead). No ablation. 43/43 tests green.
**I2 status 2026-09-20: forward-only ingestion live, D1 pending-coverage.** History spike proved NSE publishes bulk/block as current-day snapshots only (no dated free archive; BSE JS-gated). `data/bulkdeals.py` + `bulkdeals` command ingesting daily from 2026-09-20 (228 rows day one). No backtest claim possible until months accumulate.
**DOWN-flag reformulation 2026-09-20: FAIL on both locked bars.** τ_down swept on calibration in the operative region (<0.5 — first grid [0.5,0.8] was degenerate by construction since fallback already predicts DOWN when p_down≥p_up; caught by reading the flat grid, fixed, re-run). Calibration precision FLAT at 0.470–0.471 across τ 0.30→0.50 while coverage falls 92%→78%: threshold is inert — DOWN calls are precision-invariant, no ranking to convert (mirrors the UP threshold rejection). Test: precision 48.6% (< 55% bar), mean_net −0.086bps (negative), whipsaw 51%. The reframe fails; UP→NO-SIGNAL routing changes nothing about viability. Ledger at 22 touches. 45/45 tests green.
**Economics thread (flagged, separate from G1):** signal-only 51.73% @ 8.37% coverage with Sharpe −1.74 — accuracy without economics, true since the first sealed holdout. Trade-level autopsy 2026-09-19 (2,336 accepted trades): mean net per trade NEGATIVE in every confidence bucket (0.60–0.65: −0.0087%, 0.65–0.70: −0.0227%, 0.70+: −0.0198%) — confidence filtering cannot fix it. Cost per trade 0.024% ≈ 2× gross edge 0.012%; DOWN/UP legs lose identically. Scope for the dedicated G3 pass: (a) confidence-proportional sizing (Kelly-fractional, capped) instead of flat 1.0/trade; (b) holding-period sensitivity (fewer round trips amortize the fixed 31.5 bps); (c) F&O-only subset (15 bps round trip, liquid names) vs delivery; (d) turnover gate (min edge-per-trade multiple of cost). Tracked independently of UP-AUC work.
**G3-0 bucket gate 2026-09-20: SIZING FALSIFIED.** Restored F-adopted ledger (3,080 accepted): all buckets negative AND monotonically worse with confidence (0.60–0.65: −0.0098% n=2028; 0.65–0.70: −0.0223% n=785; 0.70+: −0.0328% n=267). No top bucket with positive edge at n≥150. G3a does not run. Effort reallocated to holding/F&O/gate.
**G3b (turnover/holding) 2026-09-20: all variants negative; best thr70 Sharpe −0.85.** V0 −2.49 (3,080 trades) → thr65 −1.90 (1,186, DM +3.4 p=0.0007) → thr70 −0.85 (338, DM +3.74 p=0.0002, Bonferroni-significant vs V0) → cap10 −2.26 / cap5 −1.87 (non-significant). PBO over variants 0.004 (low — thr70 dominates IS and OOS alike). But kill bar is Sharpe ≥ 0: thr70 net −4.55% (gross +2.39% vs costs 6.94%) FAILS it. G3d ≡ thr70 (cap non-binding at 0.45 trades/day — stated, not burned as a touch).
**G3c-proxy 2026-09-20: FAIL, kill-confirming (pre-registered asymmetric use).** Turnover-quartile liquid proxy + futures 15bps on frozen F-adopted predictions: Sharpe −0.28, net −4.2%, 1,076 trades, DM +5.56 vs V0. Best economics of the campaign and still negative — as the optimistic bound predicted. Per pre-registration this closes the F&O leg with real evidence and authorizes nothing (proxy, non-authoritative; real list still required before any go-signal could exist).
**G3 KILL VERDICT 2026-09-20: PAUSE — trading project becomes a research repo.** Best variant fo_proxy Sharpe −0.28 < 0 (kill bar). Significant-vs-V0 DMs (thr70 p=0.0002, fo_proxy p≈0, both < Bonferroni 0.05/13≈0.0038 over 15 logged touches / 13 eligible in `results/holdout_touches.jsonl`) prove variants differ from baseline, not that any makes money. Campaign PBO 0.0 < 0.3 (clean selection, no overfit — the failure is edge, not luck). Sizing falsified, thresholds bounded at −0.85, costs exceed edge 2:1 at every confidence level. The system cannot make money as specified; further variants would be luck-mining past a answered question. 39/39 tests green.
**U3-D1 (sentiment content) 2026-09-19: NULL — structural ceiling now covers presence AND content.** Scored 26,325 test-window announcements with FinBERT (CPU 90/s, cached forever; found + fixed transformers cache_dir passthrough bug). Join vs frozen predictions (15,216 test rows with sentiment): UP-miss positive 0.6875 / negative 0.6565 / neutral 0.6646 / none 0.6944 — no separation; UP-AUC positive **0.4864 (below coin flip)**, negative 0.5137, neutral 0.5132. Decisive: P(UP|positive-sent) = 43% ≈ P(UP|no-news) = 45% — sentiment-signed content carries no marginal information for THESE labels. Not a modeling failure; the information isn't in the conditional. Door C for UP is airtight: stop asking 5-day UP at 1σ from available data. 36/36 tests green.
| **G2** after Week 4 | News features measurably improve AUC via ablation, DM-confirmed | Keep news for interpretability but don't let it drive the model. |
| **G3** after Week 5 | Net Sharpe > 0 after real Indian costs | The binding constraint is costs/turnover, not accuracy — lengthen horizon or restrict to F&O names. |
| **G4** after Week 6 | PBO < 0.3 and DM significant vs baseline | The result is likely luck across trials. Simplify and re-test. |

---

## Appendix A — Realistic expectations

| Belief | Reality |
|---|---|
| "70% directional accuracy is achievable" | No. Public "70%" results almost always come from a wrong baseline, temporal leakage, selection bias across many attempts, silent lookahead, or no transaction costs. |
| "More features will fix it" | Usually not. Most feature families wash out. Universe size, label quality, and cost realism matter far more. |
| "Deep learning will beat LightGBM here" | Unlikely at this data scale. ~2,500–3,000 independent trading days does not support a large network. |
| "The model needs to be right most days" | No. A model that abstains 75% of the time and is modestly better than base rate on the rest is far more valuable. |
| "Accuracy is the goal" | AUC, calibration, and net-of-cost Sharpe are the goals. Accuracy is a sanity check against the majority baseline. |

**The honest target:** AUC 0.55–0.62, calibrated probabilities, positive net Sharpe after real Indian costs, and materially higher accuracy on the high-confidence subset. Your own backtest — where costs are 7.7× gross return — already shows that the binding constraint is execution economics, not model intelligence.

---

*End of Master Plan. Update Section 14 as tasks complete.*
