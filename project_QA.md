# Project Guide in Plain English

**Project:** `stock_direction_platform` — stock direction research for Indian (NSE) shares
**Written:** 28 October 2026
**Who this is for:** someone who has to understand this repo without reading all the code.

**How to use this file**

| If you want to… | Read |
| --- | --- |
| Know what the project does in one minute | §0 |
| Run something | §1 |
| Open the folder and not get lost | §2, §3 |
| Follow one prediction end to end (ranks, triple barrier, leakage) | **§4** |
| Understand the words (feature, label, SHAP, bps…) | §5 |
| Know why LightGBM and which libraries | §6, §7 |
| Know about the Qwen LLM and Phase Q0 | §8, §9, §10 |
| Know the real scores | §12 |
| Understand the dashboard, panel by panel | **§14** |
| Read any chart, and know what the *opposite* shape would prove | **§18** |
| See the whole system as one picture (inputs → models → decision) | **§19**, diagram in `architecture.md` |
| Know what is broken | §15 |

Simple rule for this file: **every technical word is explained once, in plain words, the first time it appears.** §5 is the full dictionary.

---

## 0. The project in 60 seconds

- **What it predicts.** For each of 50 large Indian companies (the NIFTY 50 list), it predicts whether the share price 5 trading days later will be **UP**, **DOWN** or **FLAT**.
- **How.** Daily prices and news go in. A table of numbers ("features") is built. Each row gets a correct answer ("label"). A machine learning model called **LightGBM** learns the link. Then the model is tested, calibrated, backtested and drawn on a dashboard.
- **The honest result today.** The model gets **47.1 %** of test rows right. A dumb rule ("always say DOWN", which is the most common answer) gets **47.9 %**. So the model is *slightly worse than a dumb rule*. The project says this openly everywhere; that is a feature, not a bug.
- **Why keep going.** Because the project is built to *measure* whether an idea works, not to pretend it does. Every "does it work?" question has a pre-agreed pass mark.
- **The second track (Track Q).** Instead of reading numbers, ask a local AI model (**Qwen3 8B**) to read the news and vote UP/DOWN/FLAT. **Phase Q0** is testing whether that AI is better than the current news model (FinBERT). It is running right now: **716 of 1,500 cases done**.

**In one line:** a careful research lab that answers "does this idea beat a dumb guess?" — and today the answer is "no".

---

## 1. Entry point: where the program starts

There are **six** ways to start something. Most people only need the first two.

| # | File | What it is | How to run it |
| --- | --- | --- | --- |
| 1 | `start_run.py` | The friendly front door. Creates `.venv`, installs dependencies, then runs the whole pipeline and the final report. | `python start_run.py` (or `start_run.bat` on Windows) |
| 2 | `src/stockml/main.py` | **The real entry point.** A command-line menu with 35 commands. Everything else is called from here. | `python -m stockml.main <command>` |
| 3 | `src/stockml/pipeline.py` | The `all` command. Runs steps in order and skips steps that are already up to date. | `python -m stockml.main all` |
| 4 | `src/stockml/api/server.py` | The read-only web API (FastAPI) that feeds the dashboard. | `uvicorn stockml.api.server:app --port 8000` |
| 5 | `stock-direction-dashboard/src/main.jsx` | The React dashboard in the browser. It only reads the API. | `npm run dev` (inside `stock-direction-dashboard/`) |
| 6 | `src/stockml/qpipeline/__main__.py` | The Track Q phase runner (the LLM research track). | `python -m stockml.qpipeline --status` |

### 1.1 What `python -m stockml.main all` actually does

`pipeline.py` checks, step by step, "is this step already newer than its input?" If yes, it skips. If no, it runs. The order is fixed:

1. **fetch** — daily price bars for 50 symbols (16 years of history) plus macro series.
2. **news** — news headlines and their sentiment scores.
3. **features** — turn prices and news into number columns.
4. **cross-sectional** — rank each stock against the other 49 on the same day.
5. **labels** — decide the correct answer (UP/DOWN/FLAT) for each row.
6. **baseline** — a simple logistic regression, as a sanity check.
7. **train** — tune and train LightGBM.
8. **evaluate** — score it on data it has never seen.
9. **backtest** — pretend to trade it, and charge real Indian fees.
10. **monitor** — check whether the input data has drifted.
11. **predict** — write today's predictions for all 50 symbols.

**`all` does not download announcements, bhavcopy, RSS or GDELT.** Those belong to the `refresh` chain, which is the fuller daily path:

`heal → fetch → bhavcopy → calendar → announcements → ann-score → news → rss → bulkdeals → [gdelt-bq, only if asked] → features → cross-sectional → labels → [train → evaluate → backtest, only with -Retrain] → predict`

`refresh` also writes a lock file, so two refreshes cannot run at the same time.

### 1.2 The 35 commands, grouped

You rarely need all of them. This is the map.

| Group | Commands |
| --- | --- |
| Run everything | `all`, `refresh`, `train-fresh` |
| Get data | `fetch`, `bhavcopy`, `calendar`, `announcements`, `ann-score`, `bulkdeals`, `news`, `rss`, `gdelt-bq`, `heal` |
| Build inputs | `features`, `cross-sectional`, `labels` |
| Train and score | `baseline`, `train`, `evaluate`, `backtest`, `monitor`, `predict`, `report` |
| Explain the model | `model-report`, `explain`, `ablation`, `flat-ablation`, `flat-threshold`, `up-autopsy`, `down-flag`, `label-grid` |
| One-off studies (finished) | `economics`, `i01-d1`, `u2-d1`, `u3-d1` |

**What the three "run" commands differ in**

| Command | Fetches data? | Retrains? | Moves the test window? |
| --- | --- | --- | --- |
| `refresh` | yes | no | no |
| `train-fresh` | yes | yes | **yes** — retraining pushes the "unseen data" end date forward |
| `all` | yes (skipping fresh parts) | only if data changed | yes, if it retrains |

The `train-fresh` / `all` row matters: every retrain spends some of your "unseen data" budget. See §12.4.

---

## 2. How to explore this repo in 10 minutes

Do these in order. Each step is small.

1. **`COMMANDS.txt`** — every command with a one-line note. Fastest way to see what exists.
2. **`config/config.yaml`** — the settings: the 50 symbols, the 5-day horizon, the flat band, the cost model, the thresholds. Nothing is hidden in code.
3. **`src/stockml/main.py` (lines 40–95)** — you see the 35 commands and which function each one calls. This is the whole control panel in one screen.
4. **`src/stockml/pipeline.py`** — how `all` decides to skip or run a step.
5. **`src/stockml/models/lightgbm_train.py`** — the model itself. This is the heart.
6. **`results/`** — the output of the last run. `metrics.json` is the scorecard.
7. **`DOCS/MASTER_PLAN.md`** — the long design document with the full background-task list (B-01…B-12, F-01…F-25).
8. **`stock-direction-dashboard/src/pages/`** — one file per dashboard page. Read `Overview.jsx` first; it is the simplest.

**Rule of thumb:** if you want to know "what numbers does the project believe?", read `results/*.json`. If you want to know "why?", read `DOCS/`.

---

## 3. Folder map: what is where

### 3.1 The root folder

| Item | What it is |
| --- | --- |
| `src/stockml/` | All Python code (§3.3) |
| `data/` | All data files, raw and processed (§3.2) |
| `results/` | Latest scores, predictions, charts, traces |
| `artifacts/` | Trained model files, tuned settings, small state files |
| `checkpoints/` | Training progress you can resume from |
| `logs/` | One log file per pipeline step, plus live heartbeats |
| `config/` | Settings: `config.yaml`, `q_track.yaml` |
| `stock-direction-dashboard/` | The web dashboard (React) |
| `DOCS/` | Long-form design documents |
| `tests/` | Automated tests |
| `scripts/` | Small helper scripts |
| `.env` | Secrets and keys (not committed) |
| `start_run.py`, `refresh_data.ps1`, `run_q_pipeline.ps1` | Launchers |

### 3.2 `data/` — which folder holds which data

| Folder | Contents | Example file |
| --- | --- | --- |
| `data/raw/` | One price file per symbol (daily bars: open, high, low, close, volume) | `ADANIPORTS.NS.parquet` |
| `data/raw/bhavcopy/` | NSE official end-of-day file, including **delivery %** | `ADANIPORTS.NS_bhavcopy.parquet` |
| `data/raw/actions/` | Corporate actions (splits, bonuses, dividends) | `ADANIPORTS.NS_actions.parquet` |
| `data/raw/announcements/` | NSE company announcements (text) | `ADANIPORTS.NS_announcements.parquet` |
| `data/raw/news/` | News headlines per symbol | `ADANIPORTS.NS_news.parquet` |
| `data/raw/rss/` | Raw RSS headline snapshot | `rss_headlines.parquet` |
| `data/raw/bulkdeals/` | Bulk and block deals from NSE | `bulk_block_deals.parquet` |
| `data/raw/gdelt_bq/` | Historical world news events (Google BigQuery) | `gdelt_events_2024.parquet` |
| `data/features/stocks/` | **The model inputs.** 183 number columns per symbol (news included) | `ADANIPORTS.NS_features.parquet` |
| `data/features/news/` | 21 news number columns per symbol | `ADANIPORTS.NS_news_features.parquet` |
| `data/features/market/` | 4 market-wide columns (same for all symbols) | `market_features.parquet` |
| `data/processed/` | **The training table:** features + the correct answer, one file per symbol | `ADANIPORTS.NS_dataset.parquet` |
| `data/llm/` | Track Q files: the 1,500 test cases and the rating ledger | `pool_c.parquet`, `qwen_rating_ledger.parquet` |
| `data/reference/` | Maps and calendars: sector map, group map, trading calendar | `sector_map.json` |

**One sentence:** `raw` = what we downloaded, `features` = numbers we computed, `processed` = ready-to-train table (numbers + answer).

#### What is a "bhavcopy"?

"Bhav" means *price* (the word NSE uses for a quote). A **bhavcopy** is the exchange's own end-of-day file: one row per listed security, with open, high, low, close, last price, average price, traded quantity, turnover and number of trades.

The reason we download it is the last two columns: **`DELIV_QTY`** and **`DELIV_PER`** — how many shares, and what percentage of the day's volume, were actually **delivered** (taken into a demat account) instead of being squared off the same day.

Why delivery % matters:

- A stock up 3 % on low delivery is mostly intraday trading — people are flipping, not holding.
- The same move on high delivery means buyers took the shares home, which is a stronger, more committed signal.
- Yahoo Finance does not publish delivery data at all, so this is information we cannot get anywhere else.

The honest limits, from the module itself (probed 2026-09-18):

| Limit | Consequence |
| --- | --- |
| The free NSE archive only has delivery columns from ~2020 | 2015–2019 return HTTP 404, so `delivery_*` exists for 2020 onward only |
| Missing is not the same as zero | A `delivery_available` 0/1 flag marks "we do not have it", so the model is not told "nobody took delivery" |
| NSE rate-limits hard | 4 s between daily requests, cool-down after 5 failures, abort after 15, cookie priming first |
| Circuit price bands are not published free day-wise | The code stores a *lock proxy* (`close == high == low`) instead of inventing band values |

**Where it lands:** `data/raw/bhavcopy/{symbol}.parquet` — one file per symbol, named after the symbol (`ADANIPORTS.NS.parquet`), merged day by day instead of rewritten from scratch. Those delivery columns are then folded into the feature store. See §15 for the twist: the columns are built but the current model does not read them.

#### Why `.parquet` and not CSV?

Parquet is a **column-wise, compressed** file format. CSV stores text row by row; parquet stores each column separately, compressed, with the data type kept.

Measured on one real file in this repo (`data/features/stocks/ADANIPORTS.NS_features.parquet`, 3,980 rows × 120 columns):

| Format | Size |
| --- | --- |
| Parquet | **3.1 MB** |
| The same table as CSV | **7.0 MB** |

Why it is the right choice here:

1. **Smaller.** All 50 stocks × 183 columns × 16 years is a lot of numbers; parquet keeps it manageable.
2. **Typed and exact.** A float stays a float, a date stays a date, NaN stays missing. CSV turns everything into text and quietly changes types on reload.
3. **Fast.** Reading only the columns you need is possible, so a 183-column table does not have to be fully materialised for every step.
4. **Safe writes.** `atomic_write_parquet` writes to a temporary file and then swaps it in, so a crash mid-write cannot leave a half-written dataset that looks valid.
5. **No quoting or delimiter traps.** Headlines contain commas and quotes; parquet does not care.

The cost: parquet is not human-readable. That is why the files a human must read — `metrics.json`, `latest_predictions.csv`, `model_report.json` — stay JSON/CSV, and only the big internal tables are parquet.

#### Why is `announcements` not part of `all`?

`all` is the short, reliable chain: prices → labels → model. NSE announcement scraping is the fragile part of the project (blocks, holidays, changed page layout), so it lives in the `refresh` chain as an **optional** step — `Step("announcements", ..., optional=True)`. A failed optional step is recorded as `skipped`, not `failed`, and the run continues (it can also be turned off with `skip_announcements`).

### 3.3 `src/stockml/` — which folder holds which code

| Folder | Job | Key file |
| --- | --- | --- |
| `data/` | Downloading and cleaning every data source | `fetch_data.py`, `news.py`, `announcements.py`, `bhavcopy.py` |
| `features/` | Turning raw rows into model columns | `feature_engineering.py`, `cross_sectional.py`, `decision_date.py` |
| `targets/` | Making the correct answer | `triple_barrier.py`, `build_dataset.py` |
| `validation/` | Splitting time safely, no peeking into the future | `time_splits.py`, `walk_forward.py` |
| `models/` | The models | `lightgbm_train.py`, `baseline.py` |
| `training/` | Qwen fine-tuning (LoRA/QLoRA) | `qwen_qlora.py`, `qwen_export.py` |
| `evaluation/` | Scoring, testing, backtesting, one-off studies | `evaluate.py`, `metrics.py`, `model_report.py`, `backtest` helpers |
| `backtest/` | The trading simulation with real costs | `backtest.py` |
| `monitoring/` | Drift checks | `drift.py`, `monitor.py` |
| `inference/` | Producing today's predictions, per-row explanations | `predict.py`, `explain.py` |
| `api/` | The read-only dashboard API | `server.py`, `dashboard*.py` |
| `qpipeline/` | Track Q phase runner with gates | `runner.py`, `state.py` |
| `v2/` | A newer decision rule (abstain option) | `policy.py` |
| `config.py`, `main.py`, `pipeline.py` | Settings, command menu, `all` | — |

### 3.4 The other folders

| Folder | Contents |
| --- | --- |
| `results/` | `metrics.json` (scorecard), `model_report.json` (behaviour report), `backtest_metrics.json`, `latest_predictions.csv`, `predictions.parquet`, `monitoring.json`, PNG charts, `traces/` (one JSON per LLM case), `holdout_touches.jsonl` (test-set usage log) |
| `artifacts/` | `models/lightgbm_final.txt` (the trained model), `best_params.json` (tuned settings), `model_metadata.json`, `calibration.json`, `state/*.json` (one small file per step) |
| `checkpoints/` | `lightgbm_final_checkpoint.txt` + `.json` (resume training), `optuna.db` (the tuning study), `trials/fold_diagnostics.jsonl` |
| `logs/` | `fetch.log`, `features.log`, `train.log`… plus `q0_heartbeat.json`, `gpu_5s.csv` (live telemetry) |
| `config/` | `config.yaml` (main settings), `q_track.yaml` (LLM track settings) |

---

## 4. One prediction, step by step

Follow one row: **ADANIPORTS on 14 Oct 2026**.

1. **Decide the decision moment.** `decision_date.py` turns a date into "the last moment you could really have known this". India closes at 15:30 IST, so anything after that counts as the next trading day. This prevents cheating with future information — §4.1 explains exactly what that cheating looks like.
2. **Collect numbers.** Close price, 5/10/20-day returns, moving averages, volatility, RSI, ATR, Bollinger width, volume ratios, delivery %, news counts, market index moves. → **183 columns: 60 price/volume/delivery, 51 news and announcement, 34 market, 21 cross-sectional, 17 calendar**.
3. **Rank across stocks.** "Is ADANIPORTS's 20-day return high compared with the other 49 today?" → a rank from 0 to 1. This removes the effect of the whole market going up.
4. **Make the answer.** Look 5 sessions ahead. Did the price move more than ±1 standard deviation of its own 20-day volatility? If no → **FLAT**. If yes → **UP** or **DOWN**. This is the "triple barrier" (§4.3, and the short definition in §5.1).
5. **Feed the model.** LightGBM reads 136 of the 183 columns and outputs three probabilities: p(DOWN), p(FLAT), p(UP).
6. **Calibrate.** Multiply the scores by a learned factor so that "70 % confident" really means about 70 % right. Factor in use: **T = 0.978**.
7. **Decide.** If p(FLAT) ≥ 0.40 → say **FLAT**. Otherwise pick the stronger of UP/DOWN. Only trade if the winner's probability ≥ **0.60**.
8. **Write the row.** `results/latest_predictions.csv`, one row per symbol, with all three probabilities and a confidence value.

**Warning that matters:** step 5 uses 136 columns, but 183 are built. The 47 left out are 42 news/announcement columns and 5 delivery/volume columns — they exist and are *not used by the current model*. This is a real gap, listed in §15.

### 4.1 What "cheating" means at step 1

**Cheating here means: using information that did not exist yet at the moment you claim to be deciding.**

One concrete week, all times IST:

| When the news was published | Which decision it may feed | Why |
| --- | --- | --- |
| Tuesday 09:00 | **Tuesday** | Before the 15:30 close, so it was on the screen while you decided |
| Tuesday 16:00 | **Wednesday** | The market had already closed — you could not have acted on it |
| Saturday 11:00 | **Monday** | Next real session |
| Friday 16:00, before a Monday holiday | **Tuesday** | Next real session, the holiday is skipped |

Now the failure. Suppose Tuesday's close fell 2 %, and at 17:00 a headline said *"Q2 profit misses estimates"*. If the code lets that headline count as **Tuesday** information, the training row becomes: *features of a day that already dropped 2 %* + *the news that explains the drop* → *label DOWN*. The model learns a rule that is true only after the fact. In the lab it looks excellent. Live, at 15:30 on Tuesday, that headline does not exist yet, so the model gets the same features **without** the explanation and the accuracy collapses. That is **look-ahead bias** — the general name is **leakage**.

What `decision_date.py` does, in order:

1. Reads the timestamp and **converts** it to IST. It never strips the timezone first, because stripping would invent a wrong hour.
2. Compares the clock time with the **15:30** cutoff.
3. If the time is later than 15:30, moves the date forward by one day.
4. Snaps to the **next real NSE session** from `trading_calendar.parquet`, so weekends and exchange holidays are skipped, not assumed.
5. Adds **no extra shift**. The shift is already inside the rule. A second `shift(1)` would silently throw away a whole day of news.
6. Maps anything newer than the calendar to `NaT`, and callers drop those rows. A headline cannot move price history it does not cover.

The rule is deliberately one-directional: news is pushed **later**, never earlier. Late news costs you a little data. Early news costs you the whole result.

### 4.2 Step 3 — why cross-sectional ranks?

"Cross-sectional" just means **across stocks, on the same day**. Instead of asking "is ADANIPORTS's 20-day return big?", it asks "is ADANIPORTS's 20-day return big **compared with the other 49 today**?"

The code (`features/cross_sectional.py`) does this for 13 source columns — `ret_5`, `ret_20`, `ret_60`, `momentum_20`, `volatility_20`, `volatility_60`, `relative_volume_20`, `volume_z_20`, `rsi_14`, `macd_hist`, `atr_pct_14`, `bb_position`, `obv_z20` — and writes a new column `{feature}_xs_rank` holding a percentile from 0 to 1:

```python
out[f"{col}_xs_rank"] = out.groupby("timestamp")[col].rank(pct=True)
```

It also builds sector and business-group relatives: `ret_5_minus_sector`, `ret_20_minus_sector`, `sector_xs_rank`, `sector_dispersion`, and the same four for groups. Membership is **point-in-time**: a row only joins the sector that was active on its own date, using the `from`/`to` fields in `data/reference/sector_map.json`.

Why it earns its place:

1. **It removes the market tide.** On a day the NIFTY rises 2 %, nearly every stock shows a positive 5-day return. Raw returns then measure "was the market up?", not "is this stock strong?". A rank says: 47th out of 50 — this stock lagged, whatever the index did.
2. **It is scale-free.** A 3 % move is large for a sleepy large-cap and small for a jumpy one. Percentiles are comparable across all 50 stocks and across all 16 years.
3. **It survives volatility regimes.** Raw return distributions widen and narrow over the years; a 0–1 rank does not.
4. **It is robust to outliers.** One bad print cannot dominate a rank the way it dominates a mean.
5. **It encodes the actual question.** A direction model that only looks at one stock's own past cannot tell "this stock is about to fall" from "this stock will fall less than the others". Rank features give it the second half.

**Why this is not leakage.** The ranks use the same day's close, with **no shift** — which looks suspicious until you line it up with the barrier: every close of day *t* is known at 15:30 on day *t*, and the trade enters at the **open of t+1**. So the rank is computed from data you legitimately hold at decision time, and the label starts strictly after it. Same-day ranking is fine; same-day ranking *of the label* would not be.

**Caveat.** A rank needs a fixed peer group. If the 50-symbol universe changes over 16 years, ranks from different eras are not perfectly comparable — which is why the universe history file is still an open item (B-11).

### 4.3 Step 4 — the triple barrier, line by line

First, the name. There are **three** exits, which is why it is called *triple*:

| Barrier | Meaning |
| --- | --- |
| **Upper** | Price rose enough → labelled **UP** |
| **Lower** | Price fell enough → labelled **DOWN** |
| **Time** | Nothing hit within the horizon → the answer is decided by the final return |

There is no such thing as a "ripple barrier" — that is a misreading of *triple* barrier. Nothing in the code ripples.

What `targets/triple_barrier.py` actually does for each row *t*:

1. **Volatility.** `sigma` = standard deviation of the last 20 **daily percentage returns**, from close prices. If 20 days are not available, or `sigma` is zero, the row is skipped.
2. **Entry price.** The **open of day t+1**, never the close of day t. The decision is made after *t* closes, so the earliest you can act is the next open.
3. **The two price barriers.** `upper = entry × (1 + 1.0 × sigma)`, `lower = entry × (1 − 1.0 × sigma)`. The 1.0 is `upper_multiplier` / `lower_multiplier` in the config.
4. **Walk forward, day by day**, up to `horizon = 5` sessions, and test using **high and low**, not closes — because a barrier is a real order that would fill intraday.
5. **First touch wins.** `high ≥ upper` → label **+1 (UP)** and exit at exactly the barrier price. `low ≤ lower` → **−1 (DOWN)**.
6. **Both barriers touched on the same day** — the code refuses to guess a side. It records `hit = 0` (FLAT) and exits at that day's close. Deterministic, no coin flip, no look-ahead into which came first.
7. **Nothing touched in 5 sessions** — exit at the last close, compute the return, and call it FLAT only if `|return| ≤ flat_band × sigma` (`flat_band = 1.0` in config). Otherwise the sign of the return decides.
8. **Outputs.** `label_raw` (−1/0/+1), `event_start`, `event_end`, `entry_price`, `exit_price`, `realized_return`. Then `make_labeled_dataset` maps −1→0 (DOWN), 0→1 (FLAT), +1→2 (UP) and drops rows missing any of those fields.

**Why FLAT is rare — and why that explains a lot.** ±1σ is roughly one ordinary day's range. Over five sessions, almost every path touches one of the two barriers, so the "nothing hit" case is uncommon: in one symbol's dataset, FLAT is **149 of 3,955 rows (~4 %)**, against 1,910 DOWN and 1,896 UP. A class that rare is hard for any model to learn, which is exactly why the project needs a separate decision rule for FLAT (§12) instead of trusting the raw argmax.

**One more thing the code gets right.** `event_end` is not decoration: it is the date the answer becomes known, and the walk-forward splitter uses it to purge training rows whose label would still be unwinding when the validation window starts (§4.6).

### 4.4 Step 4 — why standard deviation, and how it is used

**Why not a fixed ±3 %?** Because 3 % is a huge move for a utility and a boring day for a small-cap. One fixed number would label half the universe FLAT and the other half always UP or DOWN, and the model would learn the stock's volatility instead of its direction.

**Why σ?** σ measures "this stock's own normal daily wobble", so the barrier becomes *relative*:

- **Comparable across stocks** — each name is judged against its own noise.
- **Comparable across calm and wild periods** — the same stock in 2016 and 2020 is judged fairly.
- **It makes the label mean something tradeable** — "moved more than its normal noise" is closer to "there was a move worth acting on" than "moved 2 %".
- **It is dimensionless** — the multiplier is a number of σ, so the rule transfers to any price level.

**Why the standard deviation of returns, not of prices?** Price standard deviation grows with the price itself and drifts upward over 16 years; a ₹1,400 stock would look "more volatile" than the same company at ₹400. Percentage returns are stationary-ish, so σ stays comparable over time.

**How it is computed:**

```python
vol = close.pct_change().rolling(20, min_periods=20).std()
upper = entry * (1 + 1.0 * sigma)
lower = entry * (1 - 1.0 * sigma)
```

`20` is `volatility_window` (about one trading month — long enough to be stable, short enough to react). `std()` is the ordinary sample standard deviation. Both are config values, not magic numbers.

**The honest weakness.** σ is backward-looking. When volatility jumps, the barrier is still set from yesterday's calm, so a genuinely violent week can hit barriers "too easily". A forward-looking volatility estimate would be more accurate and less reproducible; the trade was made deliberately.

### 4.5 Step 5 — how the model actually predicts

Intuition first: LightGBM is a **booster** — a long list of small decision trees. Each tree looks at one or two numbers and asks a yes/no question ("is `ret_5` above 0.8?"). Every tree produces a small vote for each of the three classes, and the votes are **added up** into three scores. Predicting means: build the row, walk it down every tree, sum the votes, then turn the sums into probabilities.

The code path (`inference/predict.py`, 82 lines):

1. Load `artifacts/models/lightgbm_final.txt` (the trained booster) and `artifacts/model_metadata.json` (the **pinned** list of the 136 feature columns).
2. For each of the 50 symbols, keep only the **last row** of its feature file (`tail(1)`) — the current prediction date.
3. Select exactly the pinned columns, in the pinned order, and replace `±inf` with `NaN`.
4. If a pinned column is **entirely absent** → hard error, `Missing production features`. This is the train/serve-skew guard: a booster trained on 136 columns must never be scored on a different set. Individual `NaN`s *are* allowed, because LightGBM handles missing values natively and the training data carried the same holes; the run logs a warning.
5. `model.predict(X)` returns three raw scores per symbol.
6. **Calibration:** `apply_temperature(raw, 0.97797)` rescales the scores into three probabilities that sum to 1. This is the step that makes "0.7" mean roughly 70 %.
7. **Decision:** `apply_flat_threshold(proba, 0.40)` → **FLAT if `p_flat ≥ 0.40`**, otherwise whichever of `p_down` / `p_up` is larger. Note it is *not* a plain argmax: a stock can show UP with `p_up = 0.38`, as long as `p_flat` is below 0.40.
8. `confidence` = the single largest probability, so it may be `p_flat`. This is why a "FLAT at 62 % confidence" is a real, meaningful row.
9. Write `results/latest_predictions.csv` — one row per symbol, then mark the `predict` state complete.

So a prediction is: *136 pinned numbers → tree votes → three probabilities → a threshold rule → one of three words plus a confidence.*

### 4.6 How we check for leakage

There are two completely different jobs here, and it is worth keeping them apart.

**Job 1 — structural guards in code (these are the ones that protect the numbers):**

| Guard | Where | What it stops |
| --- | --- | --- |
| Date-ordered split, 60/20/20 | `validation/time_splits.py` | Random shuffling, which would mix future rows into training |
| Purged walk-forward | `validation/walk_forward.py` | Training rows whose 5-day label event overlaps the validation window |
| Calendar purge + embargo | `PurgedWalkForwardSplit(n_splits=5, purge_days=5, embargo_days=5)` | A 5-calendar-day gap before each window, so no training row can sit right up against it; the embargo is recorded for audit |
| `decision_date` alignment | `features/decision_date.py` | Yesterday's evening news used as if it were known this morning |
| Point-in-time membership | `sector_map.json` `from`/`to` | A stock being judged against a sector it had not joined yet |
| Pinned feature list at scoring | `models/lightgbm_train.py::prepare_xy` | Scoring a 136-feature booster on a 183-column matrix and calling it a measurement |
| No shift on same-day ranks | `features/cross_sectional.py` | Nothing — this is correct, because closes are known by 15:30 and the label starts at the next open |

The mechanism worth understanding is **purging**: a training row is dropped if `event_end` (the day its answer became known) is on or after the validation window's start. Without it, a row labelled on 10 March could sit in training while the model is tested on 8 March — the label would be describing a future it was scored against.

**Job 2 — the text scanner (`qpipeline/leakage_scanner.py`).** This one is a *reasoning* auditor for the LLM track, not a data guard. It greps the model's own `thinking` + `content` text with four pattern families:

| Class | Catches | Examples |
| --- | --- | --- |
| `post_decision_date` | Dates that come **after** the decision date | `2026-03-12`, "next week", "five days later", "after the event" |
| `outcome_mentions` | The answer being named | "upper barrier hit", "label was UP", "touched the floor" |
| `future_prices_returns` | Numbers from the future | "rose 4 %", "5-day return", "event_end" |
| `self_incrimination` | The model admitting it | "I have seen the future", "based on what happened after" |

Two details make it more than a naive grep:

- An ISO date is only reported if it is **strictly after** `decision_date`. A mention of *today* is normal and is skipped.
- Each hit keeps a ±60 character snippet plus start/end offsets, so a human can read the surrounding sentence instead of trusting a flag.

**Be honest about its status.** The file header says `CODE READY — UNTESTED`, and `scanner_version()` returns `appendix-b-v1`. It finds *explicit* textual evidence of cheating; it cannot prove there is none. The structural guards in Job 1 are what make the reported metrics trustworthy; the scanner is a filter for building clean fine-tuning data (Phase 5).

---

## 5. Every technical word, explained

Read this once. After that, everything else makes sense.

### 5.1 Data and labels

| Word | Plain meaning | Where you see it |
| --- | --- | --- |
| **Symbol / universe** | A stock ticker; the "universe" is the whole list of stocks we cover (50). | `config.yaml` |
| **Session** | One trading day on the exchange. 5 sessions ≈ one week. | everywhere |
| **Bar** | One day's price summary: open, high, low, close, volume. | `data/raw/` |
| **Feature** | One number column the model is allowed to read. Examples: `ret_5d`, `rsi_14`, `vol_20d`. | `data/features/` |
| **Feature store** | The folder where all feature files live. | `data/features/` |
| **Label / target** | The correct answer for a row, decided in advance by a fixed rule. | `data/processed/` |
| **Class** | One of the answers: 0 = DOWN, 1 = FLAT, 2 = UP. | everywhere |
| **Multiclass** | A problem where the answer is one of three options, not yes/no. | §6 |
| **Horizon** | How far ahead we look. Here: **5 sessions**. | `config.yaml` |
| **Triple barrier** | The labelling rule. Two price barriers (up and down) plus a time limit. Hit the top → UP, hit the bottom → DOWN, run out of time → FLAT. | `targets/triple_barrier.py` |
| **Flat band** | The "do nothing" zone, measured in units of normal price wobble (σ). Here: **±1σ**, i.e. `flat_band: 1.0`. | `config.yaml` |
| **σ (sigma) / volatility** | How much a stock normally jumps around. Used to scale the barriers so a quiet stock and a wild stock are judged fairly. | `vol_20d` |
| **Cross-sectional rank** | A stock's 0–1 position compared with the other 49 stocks **on the same day** (§4.2). | `_xs_rank` columns |
| **Bhavcopy** | NSE's own daily end-of-day file. We download it for one reason: the **delivery** columns (§3.2). | `data/raw/bhavcopy/` |
| **Delivery %** | Share of the day's volume actually taken into demat accounts — real buying instead of same-day flipping. | delivery features |
| **Parquet** | A compressed, column-wise table file. Smaller and typed, unlike CSV (§3.2). | `data/**/*.parquet` |
| **Point-in-time** | Using only information that existed at that moment. The opposite is **look-ahead bias** (cheating). | `decision_date.py` |
| **Leakage** | Accidentally letting future information into training. Makes results look great and be worthless. Two separate defences: structural guards in the splitting/alignment code, and a text scanner for the LLM's own reasoning (§4.6). | `walk_forward.py`, `decision_date.py`, `leakage_scanner.py` |
| **Drift** | When today's inputs stop looking like the inputs the model learned from. | `monitoring/` |
| **PSI** | *Population Stability Index*: a number that says how much a column's distribution shifted. Small = stable. Above ~0.25 = warn. | dashboard, `monitoring.json` |

### 5.2 Model and training

| Word | Plain meaning | Where you see it |
| --- | --- | --- |
| **Model** | A function that turns the feature numbers into probabilities. | `artifacts/models/` |
| **Train / fit** | Show the model examples so it can adjust its internal numbers. | `train` command |
| **Booster / weak learner** | One small decision tree. LightGBM builds hundreds of them in a row, each fixing the previous ones' mistakes. | `lightgbm_final.txt` |
| **Gradient boosting** | That "build many small trees, each correcting the last" idea. | §6 |
| **Decision tree** | A chain of yes/no questions on feature values. Easy to draw, easy to overfit. | §6 |
| **Ensemble** | A group of models whose answers are combined. Here: hundreds of trees. | §6 |
| **Hyperparameter** | A setting *you* choose before training (tree depth, learning rate). Not learned from data. | `best_params.json` |
| **Optuna** | A tool that tries many hyperparameter combinations and keeps the best. Here: 25 trials. | `checkpoints/optuna.db` |
| **Trial** | One attempt in that search. | dashboard Live tab |
| **Learning rate** | How big each correction step is. Small = slow but steady. Here: **0.035**. | `best_params.json` |
| **num_leaves / max_depth** | How complex one tree may get. Bigger = more powerful, easier to overfit. Here: 16 leaves, depth 6. | `best_params.json` |
| **min_data_in_leaf** | The smallest group a tree leaf may contain. Stops the tree memorising 3 rows. Here: **116**. | `best_params.json` |
| **Regularisation (L1 / L2)** | A penalty for complexity, to keep the model humble. | `best_params.json` |
| **Early stopping** | Stop training when the validation score stops improving. | `lightgbm_train.py` |
| **Round / iteration** | One tree added. `1000 rounds` = up to 1000 trees. | dashboard |
| **Log-loss (multi_logloss)** | The score the model minimises. Lower is better. | learning-curve charts |
| **Overfitting** | Memorising the training rows instead of learning a pattern. Shows up as high train score, low test score. | §12 |
| **Class weight** | Telling the model "FLAT rows are rare, count them more". | `lightgbm_train.py` |
| **Seed / signature** | A fingerprint of the exact data + settings used. If it changes, old results cannot be compared directly. | `dataset_signature` |
| **Checkpoint** | A saved copy of progress, so a crashed run can resume instead of restarting. | `checkpoints/` |
| **Holdout / test set** | The last slice of time kept completely unseen until the end, so the final score is honest. | 2023-10 to 2026-10 |
| **Walk-forward** | Testing time in slices: learn from the past, test on the next block, move forward, repeat. | `validation/` |
| **Purged / embargo** | Deleting the few days around each block boundary, because a 5-day label overlaps the next block. Without this, the test score is inflated. | `validation/` |
| **Fold** | One of those test slices. 5 folds = 5 rounds of the above. | dashboard Folds |
| **Calibration** | Making the stated probability match reality. "70 % sure" should be right ~70 % of the time. | `calibration.json` |
| **Temperature scaling** | The simple calibration method used: divide the scores by one learned number T. **T = 0.978** means the model was slightly under-confident. | §12 |
| **ECE** | *Expected Calibration Error*. Average gap between "how confident it said" and "how often it was right". **Lower is better**; 0.04 is good. | dashboard |

### 5.3 Numbers that score a model

| Word | Plain meaning | Good value looks like |
| --- | --- | --- |
| **Accuracy** | Share of rows answered correctly. | Higher than the majority baseline |
| **Majority baseline** | The dumb rule: always answer the most common class. **This is the bar to beat.** | 47.9 % (our bar) |
| **Balanced accuracy** | Accuracy averaged over the three classes, so a rare class counts as much as a common one. | 0.33 = random, 0.5+ = learning something |
| **Precision** | When it says "UP", how often is it actually UP? | Close to recall |
| **Recall** | Of all the real UP rows, how many did it find? | Not near 0 |
| **F1** | One number balancing precision and recall. | 0.5+ |
| **Confusion matrix** | A 3×3 table: true answer down the side, guess across the top. The diagonal is correct. | Big diagonal |
| **MCC** | *Matthews Correlation Coefficient*: one number from −1 to +1 for how well guesses match truth, fair to rare classes. **0 = random.** | 0.15+ is a usable signal |
| **ROC AUC** | How well the model *ranks* rows by probability, ignoring the threshold. 0.5 = coin flip, 1.0 = perfect. | 0.60+ |
| **Coverage** | The share of rows where the model is confident enough to speak. | 10–30 % is normal for a strong gate |
| **Directional accuracy** | Among the rows where it picked UP or DOWN, how often was the direction right? | 50 %+ |
| **Diebold–Mariano (DM) test** | A statistical test asking "is this difference real, or just luck?" A large negative number vs the baseline means "not better". | \|DM\| > 2 to matter |
| **p-value / SE** | *Standard error*: how much a score could wobble by chance. The gate asks for a gap **more than 2×SE**, i.e. clearly bigger than noise. | — |

### 5.4 Money numbers

| Word | Plain meaning | Good value |
| --- | --- | --- |
| **Backtest** | Replay history and pretend to trade each signal. | — |
| **bps** | *Basis points*: 1 bps = 0.01 %. Our round trip costs **31.5 bps = 0.315 %**. | — |
| **Round trip** | Buy + sell. Both sides pay fees. | — |
| **Gross vs net** | Before fees vs after fees. Only net is real. | Net positive |
| **Max drawdown** | Worst fall from a peak to the following low. | Under 20 % |
| **Sharpe ratio** | Return divided by how bumpy the ride was. **Above 1 is decent; negative means losing.** | > 0, ideally > 1 |
| **Hit rate** | Share of trades that made money. | 50 %+ |
| **Equity curve** | Your money over time, starting at 1.0. | Rising |

### 5.5 Checks and explanations

| Word | Plain meaning | Where |
| --- | --- | --- |
| **SHAP** | A way to split a prediction into per-feature contributions. "This row went UP because `ret_5d` added +0.3 and `rsi_14` took away −0.1." It explains *the model*, not the market. | Model behaviour → Cases |
| **Base value / raw score** | The model's starting score before looking at features, then the score after. Base + contributions = raw score. | Waterfall drawer |
| **Driver** | A feature that pushes a prediction up or down. | Behaviour plate, panel E |
| **Margin** | Gap between the top and second probability. A tiny margin means "the model is basically guessing". | Model behaviour |
| **Conviction** | How high the winning probability is. Low conviction = weak call. | Behaviour plate, panel D |
| **Attribution vs causation** | "The model used this feature" ≠ "this feature caused the price move". Always read SHAP as the first meaning. | — |
| **Behaviour mode** | A label for *how* the model is wrong. `collapsed_prior` = it just repeats the most common answer. | `model_report.json` |
| **Hypothesis** | A specific possible cause, with a stated pass/fail number. Six are tested every run. | Diagnosis tab |

### 5.6 LLM words (Track Q)

| Word | Plain meaning |
| --- | --- |
| **LLM** | *Large Language Model*: an AI that reads and writes text. Ours is Qwen3 8B. |
| **Qwen3 8B** | A model from Alibaba with 8 billion parameters, run locally on your own GPU. |
| **Quantisation (Q4_K_M)** | Squashing the model's numbers to 4 bits so it fits in 5.2 GB instead of ~16 GB. Costs a little accuracy. |
| **Ollama** | The local server that runs the model. Version 0.34.2. |
| **Context window** | How much text the model can hold at once. Here: **2048 tokens**. |
| **Token** | A chunk of text, roughly ¾ of a word. |
| **Prompt** | The instructions + text we give the model. |
| **Strict JSON** | We force the answer into a fixed shape (`{"vote":"UP","confidence":0.7,...}`) so code can read it. |
| **Harness** | Our wrapper that builds the prompt, calls the model, and checks the answer. A bad answer is rejected, not used. |
| **Vote** | The model's answer: UP, DOWN or FLAT. |
| **Zero-shot** | Ask the model with no training examples. That is what Q0 tests. |
| **FinBERT** | An older, smaller model that scores *sentiment* (positive/negative) of a headline. It is the opponent Qwen must beat. |
| **LoRA / QLoRA** | Fine-tuning tricks: train a tiny add-on (a few million numbers) instead of the whole 8-billion model. |
| **STaR** | *Self-Taught Reasoning*: keep the model's **correct** self-written explanations and train it on those. |
| **Gate** | A pass mark decided **before** the experiment. If the gate fails, the track stops. |
| **Pool C** | The 1,500 fixed test cases (500 DOWN, 250 FLAT, 750 UP) used for the Q0 gate. |
| **Trace** | A saved JSON file with the exact prompt, the model's private reasoning, and its answer. |
| **Ledger** | The append-only table of every case ever rated (`data/llm/qwen_rating_ledger.parquet`). Never edited, only appended. |

---

## 6. Why LightGBM?

LightGBM is a **gradient-boosting** library: it builds hundreds of small decision trees, one after another, each new tree trying to fix the mistakes of the ones before. For tables of numbers, it is usually the strongest simple choice.

**Six reasons it was chosen here**

1. **It fits the data shape.** Our rows are plain numbers in columns (price ratios, ranks, counts, percentages). Trees are built exactly for that. They do not need the numbers to be scaled or normalised.
2. **It finds non-linear patterns and interactions.** "RSI high *and* volume falling" can matter. A linear model cannot express that unless you build the product by hand.
3. **It is fast.** 100,000 training rows, 136 columns, 25 tuning trials, on a normal laptop. Fast enough to re-run after every data refresh.
4. **It handles messy real data.** Missing values, outliers, very skewed columns (volume, news counts) — trees cope without special treatment.
5. **It tells you what it used.** Built-in gain importance, plus SHAP values, tell you which columns drove each prediction. A neural net would not give that cheaply.
6. **It is reproducible and inspectable.** One text file (`lightgbm_final.txt`) is the whole model. You can diff it, archive it, and load it anywhere. No GPU, no framework version roulette.

**What was tried or considered, and why it lost**

| Alternative | Why not |
| --- | --- |
| Logistic regression | Kept, but only as the baseline to beat. Too rigid for interactions. |
| Deep neural net (MLP/TCN) | Needs far more data, GPU time, and tuning to beat trees on a table this size. Harder to explain. |
| Random forest | Similar accuracy, slower, and boosting usually wins on tabular data. |
| A+B+stacking ensemble | More complexity for a model that is already below the dumb baseline. Fix the data first. |
| An LLM as the decider | Too slow (~20 s per case), not reproducible, and no probability you can calibrate. It is used as a *feature* source instead (Track Q). |

### The rejected options, explained in plain words

**Logistic regression** — the simplest useful classifier. It gives every feature a weight, multiplies each feature by its weight, adds them up, and pushes the total through an S-shaped curve (the logistic function) so the output lands between 0 and 1. That curve is the "logistic" part; the 0–1 output is what makes it usable as a probability. Training means choosing those weights.

- What it can express: one straight dividing line per class pair — "more of this number → more likely UP".
- What it cannot: "RSI high **and** volume low" at the same time. To get that, you must compute the product yourself and feed it in.
- Role here: it is not thrown away. It is trained as `artifacts/models/baseline_logistic.joblib` and its accuracy (**0.48021**) is one of the two baselines the real model must beat (§12).

**MLP (multi-layer perceptron)** — a small neural network. Instead of one weighted sum it stacks **layers**: each neuron computes a weighted sum and then bends it with a non-linear function, and the next layer combines those outputs. One hidden layer can already draw curved boundaries; two or three can approximate very complex shapes.

- Why it lost: far more knobs to set (number of layers, width, learning rate, epochs, regularisation), it trains on CPU only slowly, and it is much harder to explain than a thousand small trees. On a table of this size, boosted trees usually match or beat it.

**TCN (temporal convolutional network)** — a neural network built for **sequences**. It slides learned filters along the time axis, so it can detect local shapes ("volume spike, then three quiet days") and, with dilation, look far back at low cost.

- Why it lost: it wants the **raw sequence** as input (the price path, the volume path). That would bypass the features we deliberately built and named; it needs more data and tuning; and its decisions are much harder to attribute to a specific column a user can understand.

**Ensemble** — a group of models whose answers are combined rather than trusting a single one. The hope is that their mistakes differ, so the combination is steadier than any member.

**A+B** — literally two models, A and B, combined. Simplest forms: average the probabilities, or let them vote.

**Stacking** — a smarter combination. Instead of a fixed average, you train a **third, small model** (the "meta-model") that learns *how much to trust* each base model, using their out-of-fold predictions as its inputs.

- Why it lost here: more moving parts; a second layer of folds that must itself be purged and validated, which is a fresh place for leakage to sneak in; three models to retrain, version and explain.

**The decisive reason for all four of the above:** the single model already scores **worse than the dumb majority guess** (accuracy 0.47139 vs 0.47855). Adding models to a weak signal does not create signal — it creates a more confident version of the same weak signal. Fix the data and the features first; then revisit model choice.

**Honest note:** LightGBM is a good choice for the *problem shape*. It is not a magic fix. The current model still loses to the majority baseline, and the reason is the *signal*, not the algorithm (§12).

---

## 7. The Python libraries, and what each one does here

You do not need to know these to read the results. This is the inventory.

### 7.1 Core data and maths

| Library | What it does in this project |
| --- | --- |
| `numpy` | Fast arrays and maths. Everything sits on this. |
| `pandas` | Tables with dates. Used for almost every data step. |
| `pyarrow` | Reads/writes `.parquet` files — the compact column format used for all data. |
| `scipy` | Statistics: t-tests, distributions, the Diebold–Mariano test. |
| `scikit-learn` | Metrics (accuracy, MCC, AUC, confusion matrix), calibration helpers, the baseline model. Pinned at `1.8.0`. |

### 7.2 Modelling and experiment tracking

| Library | What it does here |
| --- | --- |
| `lightgbm` | The main model (§6). |
| `optuna` | Searches hyperparameters; stores the study in `checkpoints/optuna.db` so it can resume. |
| `mlflow` | Logs runs, parameters and metrics; local database at `artifacts/mlflow.db`. |
| `shap` | Splits each prediction into per-feature contributions. |
| `joblib` | Saves/loads the small baseline model. |

### 7.3 Data sources

| Library | What it does here |
| --- | --- |
| `yfinance` | Downloads daily price bars from Yahoo Finance. |
| `requests` | Plain HTTP calls to NSE, Alpha Vantage, RSS feeds. |
| `feedparser` | Parses RSS headline feeds. |
| `google-cloud-bigquery` | Pulls the huge historical GDELT news-event table. |
| `transformers` | Runs **FinBERT**, the sentiment model for announcements and headlines. |
| `torch` | The engine FinBERT runs on. |

### 7.4 Serving, files and support

| Library | What it does here |
| --- | --- |
| `fastapi` + `uvicorn` | The read-only dashboard API and its web server. |
| `pydantic` + `pydantic-settings` | Typed settings loaded from `config.yaml` + `.env`. |
| `python-dotenv` | Reads `.env` secrets. |
| `PyYAML` | Reads `config.yaml` and `q_track.yaml`. |
| `psutil` | Process/CPU/memory telemetry for the live dashboards. |
| `matplotlib` + `seaborn` | The PNG charts saved in `results/`. |
| `packaging` | Version checks in `start_run.py`. |

### 7.5 The LLM side (separate environment)

| Library / tool | What it does here |
| --- | --- |
| `ollama` (Python client) | Talks to the local Qwen3 server. |
| `unsloth`, `peft`, `trl` | LoRA/QLoRA fine-tuning of Qwen, on WSL2. |
| `bitsandbytes`, `torchao` | 4-bit loading so an 8B model fits on one consumer GPU. |
| `datasets`, `accelerate`, `tokenizers` | Training data handling for the fine-tune. |

The LLM packages live in a **separate** environment (`requirements-llm.txt`) because they pin different versions of `torch` and `transformers`. That is deliberate: the main pipeline must never break because the fine-tuning stack moved.

### 7.6 Front end

| Library | What it does |
| --- | --- |
| React 19 + Vite 7 | The dashboard pages and dev server. |
| Tailwind CSS 4 | Styling, single monochrome design system. |
| Recharts | The line/area charts. |
| lucide-react | Icons. |

---

## 8. Are we embedding or chunking? — No.

Direct answer: **there are no embeddings, no chunking, no vector database, and no RAG anywhere in this project.** A repo-wide search for `sentence_transformers`, `faiss`, `chromadb`, `pinecone`, `qdrant`, `weaviate`, `text-embedding`, `vector database`, `RAG` returns **zero matches**.

What people usually mean by these words:

| Word | What it normally means | Do we do it? |
| --- | --- | --- |
| **Embedding** | Turning text into a list of numbers so you can compare meanings. | **No.** |
| **Chunking** | Cutting a long document into pieces before embedding it. | **No.** |
| **Vector database** | A store that finds "similar" text by distance. | **No.** |
| **RAG** | Retrieve relevant text, then let an LLM answer using it. | **No.** |

There are exactly **two** text paths, and neither embeds:

1. **FinBERT** — a small classifier. It reads a headline and outputs a score (positive/negative). A classifier is not an embedder: it gives a verdict, not a vector.
2. **Qwen3 8B** — reads the whole news text for one case and writes a JSON vote. It does not search a library; it reads what we hand it.

**Why chunking is unnecessary:** the documents are tiny. Median news text is **278 characters**, and the 95th percentile is **616** — a few lines. We cap input at **1,500 words**, which almost never cuts anything. One document = one prompt = one vote.

The word "chunk" does appear in the code once — as a lazy pandas `groupby` variable name. It has nothing to do with chunking text.

### Why we don't need any of it — and when we would

Each of the four techniques solves a problem this project does not have:

| Technique | The problem it solves | Why that problem is absent here |
| --- | --- | --- |
| **Embeddings** | "Find things that mean the same thing." | We never search. For a given symbol and date the news rows are already **linked** — the pool is known, not retrieved. Nothing to look up means nothing to embed. |
| **Chunking** | "A document is longer than the model's window." | Our documents are 278 characters at the median. Cutting one further produces fragments like "the company said on Monday" with no subject and no verb of consequence. |
| **Vector database** | "Compare thousands of vectors fast." | There is no corpus to query, so there is no index to keep. It would also tie every stored vector to one embedding model version — upgrade the embedder and the whole index is invalid. |
| **RAG** | "Read documents, then answer in words." | Our output is not words. It is `p_down / p_flat / p_up`, calibrated and scored. Retrieved prose cannot be dropped into a probability table. |

Two more practical reasons:

- **Cost and speed.** An embedding call per document, plus storage, plus a similarity search — all to reach the same decision the classifier already makes.
- **Reproducibility.** The current text path is one deterministic classifier (FinBERT) or one constrained JSON vote (Qwen), both re-runnable. A retrieval stack adds a second model whose version silently changes results.

**And the honest limit of this argument.** If any of the following became true, retrieval would earn its place and we would build it:

1. Each symbol-day grew to hundreds of long articles instead of a handful of short ones.
2. We wanted a feature like "how similar is today's news to the 20 past events that preceded big moves" — that *is* an embedding lookup.
3. The corpus outgrew the model's window (Qwen runs with a **2,048-token** context; our median document is 278 characters, nowhere near it).

None of those hold today, so retrieval here would be complexity with no measurable gain — exactly the failure mode this project exists to avoid.

---

## 9. The Qwen model: exact version and exact use

### 9.1 What is installed

| Item | Value |
| --- | --- |
| Model | **`qwen3:8b`** (Qwen3, 8 billion parameters) |
| Quantisation | **Q4_K_M** (4-bit) |
| Model file hash | `500a1f067a9f` |
| Size on disk | 5.2 GB |
| Server | **Ollama 0.34.2** (local, Windows) |
| Context window | **2048 tokens**, pinned on purpose |
| Speed | ~20.3 s per article, single case |

### 9.2 The exact settings used for every call

| Setting | Value | Why |
| --- | --- | --- |
| `temperature` | 0.6 | Some variety when re-reading a hard case; not random. |
| `top_p` | 0.95 | Standard nucleus sampling. |
| `num_ctx` | 2048 | **4096 crashes or stalls on this GPU.** Pinned to 2048 and locked. |
| `num_predict` | 1024 | Enough room for reasoning + JSON answer (512 was too small). |
| `thinking_budget` | 512 | Lets the model reason privately before answering. |
| `keep_alive` | `5m` | Keeps the model warm between cases. |
| `retries` | 3 | Re-ask if the answer breaks the format. |
| `timeout` | 300 s | Give up on a stuck call. |
| Max article length | 1,500 words | Fits the context window with room for instructions. |

### 9.3 How one case is processed

1. **Build the prompt** (`qwen_harness.py` → `build_prompt`). Frozen skeleton, version tag `v1-skeleton`: role, the price context, the news text, the output contract.
2. **Call Ollama** (`rate_case`). `think=True`, and we read `message.thinking` (the private reasoning) separately from `message.content` (the answer).
3. **Validate** (`validate_output`). The answer is thrown away if it fails: `no-json-vote`, `json-parse`, `bad-vote`, `bad-numbers`, `out-of-range`, `bad-trace-steps`, `sign-mismatch`, `empty-thinking`. A rejected case is recorded as a **harness failure**, never silently fixed.
4. **Save a trace.** One JSON in `results/traces/` with the exact prompt, the reasoning, the answer, timing.
5. **Append to the ledger.** `data/llm/qwen_rating_ledger.parquet` — append-only, one row per case, never rewritten.
6. **Hand the vote to the model as a feature.** Downstream, the vote and the confidence become columns that LightGBM can use. The LLM never makes the final decision; it votes, LightGBM decides.

### 9.4 Deviations from the plan (documented, not hidden)

| Planned | Actual | Reason |
| --- | --- | --- |
| Context 4096 | **2048** | 4096 crashes the GPU driver on this machine. |
| `num_predict` 512 | **1024** | 512 truncated the JSON answer. |
| Read answer from `content` | read `message.thinking` **and** `content` | Qwen3 hides reasoning in a separate field; parsing only `content` lost it. |

---

## 10. Phase Q0: what is running right now

### 10.1 What Q0 is testing

One question: **is Qwen3 8B, with no fine-tuning, better than FinBERT at reading the same news and calling the 5-day direction?**

- Test set: **Pool C**, 1,500 iterations, built to be balanced but honest (500 DOWN, 250 FLAT, 750 UP), all from before the holdout.
- Same case, same text, two raters: **Qwen3 8B** vs **FinBERT**.
- **Pass mark (set in advance):** Qwen's agreement with the real label must beat FinBERT's by **more than 2 × standard error** of the difference. A small win is not a win.

### 10.2 Live progress (28 Oct 2026)

| Measure | Value |
| --- | --- |
| Iterations rated | **716 of 1,500** |
| Iterations judged (usable vote) | **652** |
| Harness failure rate | **8.94 %** |
| Average time per case | **173 s** |
| Estimated time left | **~38 hours** |
| Hottest GPU temperature | **86 °C** (limit 87 °C) |
| Thermal state | **green** |

### 10.3 How it stays safe on one GPU

The run is long and hot, so it protects itself instead of hoping for the best.

| Guard | Rule |
| --- | --- |
| Temperature ladder | 80 °C watch → 84 °C prepare → 85 °C alarm → **87 °C abort** |
| Check frequency | Every 5 seconds, on a separate thread |
| After abort | Cool down, then resume from the ledger |
| Long run | 30-minute cooldown after 4 trips per hour |
| Work rhythm | 15-minute break per 60 minutes of GPU work |
| Clock | Locked to **1400 MHz** to stop thermal spikes |
| Supervisor | A watchdog process restarts the run if it dies (single instance enforced) |

### 10.4 The important warning about the verdict file

`results/q0_d1.json` currently says:

```json
{"n": 97, "delta": 0.0515, "se_diff": 0.0631, "pass": false, "complete": false, "rated_total": 111}
```

**This is a stale snapshot from an early 111-case run. It is not a verdict.** The gate needs 1,500 cases. `complete: false` means the gate was never reached. Do not quote this number as a result. The dashboard labels it `INCOMPLETE` for exactly this reason.

### 10.5 What happens when Q0 finishes

1. The runner writes the real `results/q0_d1.json` with n = 1,500.
2. If **PASS** → the track continues to Phase 4 (reliability), then LoRA work.
3. If **FAIL** → Track Q stops. The LLM is not used for direction, and the project goes back to improving the tabular model. **This is a real possible outcome and is planned for.**

---

## 11. Next phases (the Track Q roadmap)

Every phase has a pass mark. A phase cannot start until the previous one passed. `artifacts/state/q_track_state.json` does not exist yet, so the orchestrator refuses to start anything — that is correct behaviour, not a bug.

| Phase | What it does | Pass mark |
| --- | --- | --- |
| **Q0** (running) | Zero-shot gate: Qwen3 vs FinBERT on 1,500 iterations | Beat FinBERT by > 2×SE |
| **Phase 3** | Build the rating ledger at scale | Ledger complete, harness failures under control |
| **Phase 6 probe** | Check LoRA training is technically possible | Runs on a small sample |
| **Phase 4** | Reliability: does the same case get the same vote twice? | Agreement above the stated bar |
| **Phase 5** | Export training data (prompt + correct rationale) | Enough usable rows |
| **Phase 6 train** | QLoRA fine-tune a small adapter on Qwen3 8B | Loss improves, no collapse |
| **Phase 7** | Re-run the gate with the fine-tuned adapter | Beats the zero-shot result |
| **Phase 8** | One final evaluation, run once, never repeated | Pre-registered bar met |

**Beyond Track Q**, the other open work is not about LLMs at all: feed the news and delivery columns to LightGBM (they are built but unused), and re-run the evaluation so the two report files agree (§15).

---

## 12. The honest scoreboard

### 12.1 Test-set metrics (chronological holdout, 2023-10 → 2026-10)

| Metric | Value | Plain reading |
| --- | --- | --- |
| Accuracy | **0.4714** | 47.1 % of rows right |
| Majority baseline | **0.4785** | "Always DOWN" gets 47.9 % |
| Difference | **−0.71 pp** | **The model loses to a dumb rule** |
| Balanced accuracy | 0.3803 | Barely above random (0.33) |
| MCC | **0.0372** | Almost no signal |
| ROC AUC | **0.5544** | Weak ranking ability |
| ECE | **0.0424** | Probabilities are reasonably honest |
| Temperature | **0.978** | Almost no calibration was needed |
| Diebold–Mariano (UP) | **−4.83** | The gap vs baseline is real, not luck — in the wrong direction |

### 12.2 What the model actually does

Per-class recall on the test split:

| Class | Recall | Meaning |
| --- | --- | --- |
| DOWN | **0.80** | Finds 80 % of real down moves |
| FLAT | **0.15** | Misses most flat periods |
| UP | **0.16** | Misses most up moves |

It produces **DOWN 79.7 %** of the time, while only **49.5 %** of rows are actually DOWN.

The behaviour report labels this mode **`collapsed_prior`**: the model learned "DOWN is the most common answer" and mostly repeats it. That is exactly what the majority baseline does — which is why the scores are almost identical.

**Why this is a good sign for the project:** the diagnostic machinery caught it immediately, named it, and blocked the next phase. A weaker project would have reported "47 % accuracy, promising" and moved on.

### 12.3 Backtest (real Indian delivery costs)

| Metric | Value | Plain reading |
| --- | --- | --- |
| Trades | 3,080 | — |
| Gross return (summed) | **+15.4 %** | Before costs there is a small edge |
| Cost paid | **61.4 %** | 31.5 bps per round trip, paid 3,080 times |
| Net return | **−37.3 %** | After costs, a loss |
| Sharpe | **−2.49** | Losing, and bumpy |
| Max drawdown | large | — |

**Reading:** the gross edge is smaller than the tolls. Lowering the assumed cost would only hide this; the model has to get better.

### 12.4 Gate results so far

| Study | Verdict |
| --- | --- |
| G1 (original baseline gate) | **FAILED** |
| G1 recovery attempt | Failed |
| H1 / H1b hypotheses | Rejected, plus a memorisation flag |
| Track F | Adopted |
| U2-D1, U3-D1 | Failed to clear the bar |
| I0-D1b, I1-D1 | No improvement |
| DOWN-flag | Confirmed the collapse |
| **G3** | **KILL → PAUSE** — the tabular track is paused, not deleted |

One practical consequence: every retrain advances the "unseen data" end date. The holdout has been touched enough that **further retraining now costs real evaluation budget**. That is why `artifacts/holdout_touches.jsonl` exists.

---

## 13. What is built, and what is missing

### 13.1 Built and working

- Full data pipeline: prices, macro, delivery %, announcements, news, GDELT, RSS.
- Feature store: 183 columns, point-in-time safe, cross-sectional ranks.   
- Triple-barrier labels at a 5-session horizon.
- LightGBM training with Optuna tuning, checkpoint/resume, class weights.
- Temperature-scaling calibration and ECE.
- Purged walk-forward validation and a leakage scanner.
- Honest evaluation against a majority baseline with DM tests.
- Event backtest with an Indian delivery cost model.
- Drift monitoring (PSI) and a 12-check data-health panel.
- SHAP explanations plus a behaviour/diagnosis report with six hypotheses.
- Read-only FastAPI + React dashboard (8 pages, 40+ panels).
- Track Q scaffolding: harness, ledger, thermal guards, supervisor, phase runner.

### 13.2 Missing or blocked (the short list)

| # | Issue | Why it matters |
| --- | --- | --- |
| 1 | **136 features used, 183 built** | The model has never seen 42 news/announcement columns or 5 delivery/volume columns. |
| 2 | **`results/q0_d1.json` is stale** | A 97-case snapshot can be mistaken for a verdict. |
| 3 | **Report and evaluation disagree** | `model_report.json` and `metrics.json` were computed on different windows; the dashboard shows a red banner on purpose. |
| 4 | **Model loses to the majority baseline** | The core scientific result. |
| 5 | **Track Q phase state not written** | `q_track_state.json` missing → nothing after Q0 can start. |
| 6 | **News-blind predictions** | The serving file cannot react to today's news. |
| 7 | **Reference data incomplete** | Universe history, entity map, sector/group detail are still stubs. |

---

## 14. The dashboard, page by page

Eight pages, grouped in three sections: **Research** (Overview, Predictions, Stock analysis), **Model** (Model behaviour, Model evaluation, Backtesting), **LLM & data** (Q0 research, Data & integration).

### 14.1 How the dashboard gets its numbers

- It only calls `GET /dashboard/*`. No button trains, re-predicts or fetches new data; the one export is a CSV of the rows already on screen. That is deliberate: opening a page can never change a result.
- Every reply has the same envelope: `{ data, as_of, generated_at, source: { kind, artifacts }, warnings }`.
  - `as_of` = the date the data speaks to.
  - `generated_at` = when the file was written.
  - `source.kind` = `live` or `demo`.
- **Demo fallback:** if the API is unreachable and demo mode is on, panels fill from a bundled sample file and show a banner. A reachable API always wins. Demo numbers are never real results.
- **Header chips:** `● LIVE` / `◐ STALE` / `○ DISCONNECTED` (from `/dashboard/health`), the symbol count (50), and the universe tier.
- **Refresh speed:** 60 s for normal files, 5 s for Q0 and live training.
- Colours never encode direction. Direction uses a **glyph + text + badge weight**, so the page still reads on a greyscale printout.

---

### 14.2 Page 1 — Overview

*Purpose: is the platform healthy, and how did the last run score?*

**The four big numbers**

| Stat | What it shows | Plain meaning | Good looks like |
| --- | --- | --- | --- |
| **Expected session** | Newest trading day the data *should* reach | From the stored NSE calendar | Equals the last trading day. "Weekday estimate" hint = the calendar is lagging, so the value is a guess |
| **Feature panel** | Date of the newest feature row | How current the model's inputs are | Same as expected session; gap 0 |
| **Test accuracy** | Holdout accuracy + the bar | How often the model is right on unseen data | **Above** the majority baseline (47.9 %). Today: 47.1 %, so it says "Does not beat" |
| **Backtest total return** | Net return after costs, with drawdown and Sharpe | Would it have made money? | Positive, Sharpe > 0. Today: negative |

**Panels**

| Panel | What it shows | What it means | Good looks like |
| --- | --- | --- | --- |
| **Freshness** | Table: step, status, as-of date, lag in sessions, the one command that fixes it | Which pipeline step is behind | "Every pipeline step is current". Any row here = predictions are built on old data |
| **Serving → Latest prediction mix** | Bars of how many symbols got UP / DOWN / FLAT, plus mean confidence | The shape of today's output at a glance | A spread across the three labels. One bar filling the row = the model collapsed (see §12.2) |
| **Model → Training snapshot** | Rounds completed vs configured (1000), when the checkpoint was last touched, trained-at time, feature count, temperature | Is the model file the one you think it is? | Rounds = configured (or early-stopped), feature count 136, temperature ≈ 1.0 |
| **Dataset → Coverage** | Symbols, feature rows, feature date span, dataset rows, horizon, flat band | Size and age of the training data | 50 symbols, roughly 16 years, horizon **5**, flat band **1.0** |
| **Health → Checks and drift** | First five integrity checks, drift alert count, missing rate, latest data date | Are the files consistent with each other? | All checks ticked, drift alerts 0, missing rate near 0 |
| **Artifacts → Recently written files** | File name, modified time, age, size | What the pipeline last wrote | Fresh timestamps. A very old model file explains stale predictions |

---

### 14.3 Page 2 — Predictions

*Purpose: what is the model saying today, for all 50 stocks?*

| Stat | What it shows | Plain meaning | Good looks like |
| --- | --- | --- | --- |
| **Universe** | 50 | How many symbols are configured | 50 |
| **Rows returned / scored** | Visible rows out of all scored rows | The filter is working | Equal unless you filtered |
| **Mean confidence** | Average winning probability | How sure the model feels on average | 0.45–0.60. Very high average = over-confidence |
| **Above threshold** | Rows with confidence ≥ **60 %**, and their share | How selective the model is | A minority of symbols (e.g. 5–15 of 50). 50 of 50 means no selectivity |

**Panels**

- **Signals table** — symbol, as-of date, signal badge, probability bar (p DOWN / p FLAT / p UP), confidence, and an `act` badge when confidence clears 60 %. Sorted by confidence. The **date is the last feature row the model saw**, not today's date.
- **Mix → Predicted label distribution** — how many symbols got each label. Ideally a believable spread; today heavily DOWN.
- **Decision rule** — explains that probabilities are calibrated, that the label uses a FLAT band instead of a plain argmax, and that a row above 60 % is only a *candidate* (the backtest still loses on some).
- **Legend → Signal encoding** — what each glyph means: UP = up move beyond the flat band; DOWN = down move beyond it; FLAT = inside the band, i.e. no directional edge.

**Warning shown on this page:** if the serving date lags the expected session, the cause is upstream. Re-running only the prediction step cannot advance the date, because it reads the last feature row.

---

### 14.4 Page 3 — Stock analysis

*Purpose: one symbol, in detail.*

| Stat | What it shows | Plain meaning | Good looks like |
| --- | --- | --- | --- |
| **Last close** | Latest close price + date | — | Date is a recent session |
| **Change in window** | % change over the visible history | Direction of the chart | — |
| **Stored bars** | Rows in `data/raw` for this symbol | Data depth | Thousands of rows (years) |
| **Latest signal** | Today's label + confidence | What the model says about this stock | — |

**Panels**

- **Price → Close price** — the raw daily close line. This is the price the labels are built from.
- **Features** — a dropdown listing every feature the model consumes for this stock (monotonic or constant columns are hidden). This is the best place to eyeball whether a feature looks sane.
- **Serving → Latest prediction** — the probability bar plus P(up) and P(down) for this symbol only.
- **Realised → Prediction accuracy** — accuracy of this symbol's past scored predictions, with the date range and the number of events. Small sample: read it as a hint, not a verdict.
- **History → Every scored prediction** — one row per decision date: predicted vs realised, the three probabilities, the realised return, hit/miss.
- **Volume → Traded volume** — daily volume line. Sudden spikes often line up with news days.
- **Caveat — what this page will not tell you** — a missing session is a **data gap, not a zero return**; and the newest few sessions have no realised label yet, because the label needs 5 future sessions. Both are correct behaviour, not bugs.

---

### 14.5 Page 4 — Model behaviour

*Purpose: **why** does the model predict what it predicts? This is the most important page in the project.*

**Top of the page**

| Element | What it shows | Meaning |
| --- | --- | --- |
| Split selector | Train / Calibration / Test | Which slice the plate describes |
| **Compare** | All three splits side by side on the same axes | Spot overfitting instantly: if train looks great and test looks flat, the model memorised |
| **Consistency banner** | "Report and evaluation agree" or "do not line up", rows compared, max \|Δp_down\|, decision agreement %, "dataset signature moved" | Whether `model_report.json` and `metrics.json` were computed on the same rows. **It is red today, on purpose** |

**Tab 1 — Behaviour**

| Panel | What it shows | What it means | Good looks like |
| --- | --- | --- | --- |
| Plate header | Split, row count, a one-line headline, a **mode badge** | The report's own verdict on *how* the model is wrong | `balanced_discriminating` |
| **A · shares** | Predicted share vs actual share, per class | Is the model producing the same mix that really exists? | Bars roughly aligned |
| **B · confusion matrix** | 3×3 counts, true down the side | Where the errors are | Big diagonal |
| **C · class profile** | Mean probability per actual class, plus **separation** (0 = identical profiles, 1 = disjoint) | Can the model tell the classes apart at all? | Separation clearly above 0 |
| **D · conviction** | Histogram of the winning probability; share of calls with max p < 0.50; mean max p | How sure the calls are | Most mass above 0.5 |
| **E · drivers** | Signed mean SHAP toward the class it over-produces | *Which features* cause the over-prediction | A few plausible features, not one column doing everything |
| **Margins** | Median gap between the top two classes; share of calls below 0.10 | Near-zero margins = coin flips | Few calls under 0.10 |
| **τ sweep** | Accuracy, directional coverage and UP recall as the FLAT threshold τ changes; vertical line = τ in use (**0.40**) | What the current threshold costs you, and what another one would give | Accuracy peaks near the τ in use |
| **Rolling predicted share & accuracy** | Rolling share of DOWN/UP and rolling accuracy over the split, against a line for the actual DOWN share | Is the behaviour stable over time, or drifting? | The rolling DOWN line tracks the actual-share line |

**Tab 2 — Diagnosis**

Six hypothesis cards. Each card states a claim, shows the measured evidence and the threshold it is judged against, and gives a verdict:

| Glyph | Verdict | Meaning |
| --- | --- | --- |
| ● | Supported | The cause is real |
| ○ | Rejected | Not the cause — stop chasing it |
| ◐ | Inconclusive | Not enough evidence either way |
| ◇ | Context | Background, no verdict |

The value of this tab: it **stops the team guessing**. A rejected hypothesis is as useful as a supported one.

**Tab 3 — Cases**

- Filter by split, predicted class, actual class, correct/wrong, symbol.
- Click any row to open the **explanation drawer** for that one prediction:
  - the three probabilities, the **decision margin**, the split it came from;
  - a **waterfall per class**: base value → ranked feature contributions → raw score;
  - **"distance to flip"**: which feature would have to change, by what multiple, to change the answer. This tells you whether a wrong call was *close* or *hopeless*.
- Remember: contributions explain **the model's arithmetic**, not the market's cause.

**Tab 4 — Training**

| Panel | What it shows | Good looks like |
| --- | --- | --- |
| Model card | Model type, feature count (136), train rows, dataset signature, params hash, temperature, τ flat, train window, test window | Signature matches the one in the evaluation |
| Train / test accuracy | Train accuracy (re-scored) and test accuracy | A small gap. A 15-point gap = memorisation |
| Three splits | Rows, accuracy, balanced accuracy, MCC, ECE and mode per split | Test ≈ calibration; train can be higher, not wildly |
| Three confusion matrices | One per split, same axes | Test should not be dramatically worse than train |
| Folds → recall per fold | Heat strip per fold: DOWN / FLAT / UP recall, darker = higher | Stable across folds. One bright column = luck |
| Attribution → importance with PSI | Mean \|SHAP\| plus a PSI drift badge per feature | Top features make sense; PSI under 0.25 |
| Learning curve | Train loss vs calibration loss per iteration | Both fall, then flatten; calibration loss should not turn upward |

**Tab 5 — Live**

Shows the training run in flight, if one is running: phase, trial x/y, fold x/y, iteration x/1000 with ETA, a rounds progress bar, the per-iteration loss chart, and a fold table with score and per-class recall.

The calibration set is attached as a **watched set only** — it is never used to stop training early, so watching it cannot change the model.

---

### 14.6 Page 5 — Model evaluation

*Purpose: the scorecard. Are the numbers good, and are the probabilities honest?*

**The six stats**

| Stat | What it means | What would be good |
| --- | --- | --- |
| **Accuracy (thresholded)** | Accuracy under the real decision rule (FLAT band), with the plain argmax next to it | Above the 47.9 % baseline |
| **Balanced accuracy** | Accuracy averaged over the three classes; macro F1 beside it | 0.33 = random, 0.45+ = real skill |
| **MCC** | One number from −1 to +1, fair to rare classes; 0 = chance | 0.10+ |
| **ROC AUC (OvR macro)** | Ranking quality, one class versus the rest | 0.60+ |
| **Expected calibration error** | Average gap between claimed confidence and observed accuracy; temperature beside it | Under 0.05 is decent |
| **Directional accuracy** | Correctness among rows where it picked UP or DOWN, plus coverage | 50 %+ on a meaningful coverage |

**Panels**

- **Holdout confusion matrix** — same 3×3, with the majority baseline printed underneath. The note explains that the FLAT column dominates, which is exactly why accuracy alone is misleading.
- **Precision / recall per class** — plus test rows, test start, "LightGBM − baseline" in percentage points, and the **Diebold–Mariano** statistic.
- **Calibration → reliability diagram** — x = what the model claimed, y = what actually happened. A perfectly calibrated model sits on the diagonal. Below the diagonal = over-confident. Temperature scaling is fitted on the validation slice, not on this holdout.
- **Decision rule** — the notes plus temperature, flat threshold, confidence threshold, thresholded vs argmax accuracy, and how many predictions were actually taken.
- **SHAP feature importance** — mean absolute SHAP per feature, switchable between 15 / 25 / 40 rows. Read as "what the model used", never as "what caused the price".
- **Training metadata** — trained-at, training rows, test rows, features, train accuracy, final test accuracy, and chips of the first 14 feature names.
- **Training context** — the raw best-parameters JSON and the checkpoint index. Useful when a result must be reproduced exactly.

---

### 14.7 Page 6 — Backtesting

*Purpose: if you had traded every signal and paid real Indian fees, what would have happened?*

| Stat | What it shows | Plain meaning | Good looks like |
| --- | --- | --- | --- |
| **Total return** | Net %, with gross minus cost underneath, segment and bps | Money after fees | Positive |
| **Max drawdown** | Worst peak-to-trough fall of the equity curve | Worst pain endured | Small (< 20 %) |
| **Sharpe** | Return per unit of bumpiness | Risk-adjusted quality | Above 1; negative = losing |
| **Trades** | Count, hit rate, trading days, positive days | How much activity, how often right | Hit rate above 50 % |

**Panels**

- **Cumulative net equity** — equity over time, starting at 1.0, with a reference line at 1.0 and drawdown drawn underneath. Weekly points, reconstructed from daily events.
- **Per symbol** — trades, hit rate, net return and a small bar per symbol. Shows whether one stock carries the result (or sinks it).
- **Where the return goes** — gross return, cost paid, net return, round-trip cost. The callout says the important thing: when gross is positive but net is negative, the strategy has **no edge to spend**, and lowering the assumed cost would only hide it.
- **Cost model** — commission, slippage, other costs in bps, and the overlap policy (**no overlapping positions per symbol**, so you cannot stack 10 trades on the same day).
- **Trades** — one row per event: entry and exit date, symbol, predicted vs realised, entry and exit price, gross, cost, net, hit/miss. Filterable by symbol and by winners/losers.

---

### 14.8 Page 7 — Q0 / LLM research

*Purpose: watch the live LLM experiment. Read-only: this page can never start, stop or resume the run.*

Four tabs: **Status**, **Live**, **Analytics**, **Cases**.

**Tab 1 — Status**

| Panel | Key values | Meaning |
| --- | --- | --- |
| **Run** | Phase, complete yes/no, PID, started at; heartbeat age vs its limit; runner/supervisor stale flags; the current case (symbol + decision date + hash) | Is the process alive and what is it chewing on |
| **Big stats** | **Rated**, **Judged**, **Harness fail rate**, **ETA** | Rated = cases with any row; Judged = usable vote. They differ on purpose: a malformed vote is rated but not judged |
| **Ledger coverage** | Progress bar of rated vs 1,500, plus ledger rows and attempts | How far through Pool C |
| **Gate** | n, pool, **Δ**, **bar (2×SE)**, Qwen agreement, FinBERT agreement, SE of difference, recorded pass, complete, recorded at | The actual verdict, with the arithmetic that produced it |
| **Phases** | Each Track Q phase and its status | What is done, running, or blocked. Empty = `q_track_state.json` missing |
| **Thermal** | Temperature, SM clock, state | Whether the GPU is safe |
| **GPU temperature samples** | Temperature over time | Spikes and rests are visible |
| **Supervisor** | Supervisor PID, child PID, applied clock, mutex name | The watchdog is armed and single-instance |
| **Setup** | Model, quantisation, hash, context size, throughput | Proof of *which* model produced these votes |
| **Timing** | Measured wall clock | Reality check against the ETA |
| **Ledger distribution** | Votes, pools, drop reasons, top symbols | Where the votes go and why cases get dropped |
| **Agreement vs realised label** | Qwen and FinBERT agreement rates | The number the gate is built from |
| **Sources** | Where every field came from | Traceability |
| **Control** | "Not available here" | The read-only rule, stated on the page |

**Tab 2 — Live**

Phase, heartbeat age, current case, window selector; then **temperature and clock** (two axes), **utilisation and VRAM**, **power draw and fan**, **cumulative rated vs judged**, **rolling case duration**, **rolling agreement and harness failures**, **rolling vote mix**, **thermal history with ETA burn-down**, and **PSI drift history**.

Why it matters: a run that takes ~38 hours can silently stall. The heartbeat age and the rolling charts show a stall within minutes.

**Tab 3 — Analytics**

- **Funnel** — attempted → rated → judged → kept, with retention at each step. Tells you where cases are lost.
- **Qwen vote × realised label** and **FinBERT vote × label** — two cross-tabs side by side. The gate is the difference between these two.
- **Calibration** — does the model's stated confidence mean anything? ECE badge on the panel; below the diagonal = over-confident.
- **Confidence distribution** — where the confidence actually sits.
- **Disagreement quadrants** — both agree / Qwen only / FinBERT only / neither. "Neither" caps how far either model can go.
- **Latency by outcome** — p50, p90, mean seconds per case. Harness failures being slowest is expected: they are the retries that ran out.
- **Cohorts — news vs no-news** — the honest control. A rater that only wins on news days is reading text; one that wins on no-news days is exploiting the prior.
- **Per symbol** — agreement by stock. A single symbol driving the average is a red flag.

**Tab 4 — Cases**

Filter by vote and outcome, then click a row to see the stored trace: the ledger row, the vote and confidence, the **exact prompt** the model saw, the **private reasoning**, and the raw answer. Nothing is reconstructed; if the trace file is missing, the page says so.

---

### 14.9 Page 8 — Data & integration

*Purpose: is the data fresh, complete and consistent — and what one command fixes it?*

| Stat | What it shows | Good looks like |
| --- | --- | --- |
| **Expected session** | Newest session the data should reach | = latest trading day |
| **Gap** | Sessions behind | 0 |
| **Internal holes** | Missing sessions *inside* a symbol's own coverage | 0 |
| **Stale steps** | Pipeline steps that are behind | 0 |

**Panels**

- **Refresh → Pipeline run state** — if a refresh is running: step x of y, current step, how long it has been going.
- **Every pipeline step** — the master table: step, status (`fresh` / `stale` / `missing` / `partial`), data as-of, expected, lag, **backfill** (`yes` or `forward-only`), the exact repair command, and a note. Model steps are judged by *artifact age*, not session lag, because a model can legitimately be older than the data.
- **Coverage** — per dataset: files, rows, first date, last date, symbols behind.
- **Cross-artifact checks** — do the files agree with each other (row counts, dates, label horizon)? Each check is `ok`, or a severity.
- **Holes** — per symbol: how many sessions are missing, the first and last missing date, and the first ten dates. The callout explains why this matters: appending from "last date + 1" can never repair a hole in the middle, because the pipeline thinks the symbol is already current.
- **Runbook** — five copy-paste commands: refresh, refresh + retrain, heal holes, serve the API, run the dashboard.
- **Training** — what happens when you train: LightGBM rounds recorded, checkpoint time, detection method, whether QLoRA telemetry exists. The callout states the trap: `train` trains on whatever features already exist, so use `refresh_data.ps1 -Retrain` if you want fresh data.
- **Live → Q0 runtime** — status, phase, heartbeat age, gate verdict.
- **Integration → Read-only endpoints** — the 16 routes and what each returns, the shared envelope, and the promises: GET only, artifact allowlist, no path traversal, malformed files degrade to warnings.
- **Honesty — what cannot be backfilled** — RSS is one snapshot per day; bulk/block deals have no history; news is capped at 25 Alpha Vantage requests/day; GDELT BigQuery is backfillable but scans ~164 GB.
- **Behind per dataset** — which symbols are behind, with the last and minimum dates.

**Footer note worth remembering:** labels *intentionally* trail features by 5 sessions (the horizon), and lag counts are weekday estimates because the dashboard does not model NSE holidays.

---

### 14.10 The visual language, in one table

Every chart on these pages is taken apart individually in **§18** — what it actually draws, what it shows today, and what the opposite shape would mean.

| Thing | Meaning |
| --- | --- |
| `●` / `◐` / `○` on a signal | UP (solid) / DOWN (mid) / FLAT (dashed) |
| `●` / `○` / `◐` / `◇` on a hypothesis | Supported / Rejected / Inconclusive / Context |
| Badge `fresh` / `stale` / `missing` / `partial` | Current / behind / absent / partly written |
| Colour intensity | Relative size within that panel only — not comparable between panels |
| Numbers in mono font | Raw values, untransformed |
| "Empty" box | The artifact is missing (or has zero rows). It is never a zero score |
| Source tag | `live` vs `demo`, plus which files answered |
| "Read-only" notes | Real guarantees: the page cannot trigger work |

---

## 15. Known issues, in order of severity

1. **The model loses to a dumb rule.** 47.14 % vs 47.85 %. This is the honest headline.
2. **136 features used, 183 built.** 42 news/announcement columns and 5 delivery/volume columns are computed and then not passed to the booster. This is the single biggest fixable gap.
3. **`results/q0_d1.json` is stale.** It holds a 97-case snapshot marked `complete: false`. Never quote it as a verdict.
4. **Two report files disagree.** `metrics.json` and `model_report.json` cover different windows and use different dataset signatures. The dashboard shows a red consistency banner on purpose. The fix is to re-run `evaluate` and `model-report` on the same window.
5. **The model is news-blind in production.** `latest_predictions.csv` cannot react to today's headlines.
6. **Test budget is being spent.** Every retrain moves the holdout. `holdout_touches.jsonl` exists to make this visible.
7. **Track Q cannot proceed past Q0** until `artifacts/state/q_track_state.json` is written and Q0 actually passes.
8. **Duplicate `torch`/`transformers` pins** across `requirements.txt` and `requirements-optional.txt`. Harmless today, confusing next month.

---

## 16. Command cheat sheet

```powershell
# First run (creates .venv, installs, runs everything, then reports)
python start_run.py

# Daily refresh: new data + features + predictions. Never retrains.
.\refresh_data.ps1

# Refresh and retrain (moves the holdout end date forward)
.\refresh_data.ps1 -Retrain

# One step at a time
.\.venv\Scripts\python.exe -m stockml.main fetch
.\.venv\Scripts\python.exe -m stockml.main features
.\.venv\Scripts\python.exe -m stockml.main labels
.\.venv\Scripts\python.exe -m stockml.main train
.\.venv\Scripts\python.exe -m stockml.main evaluate
.\.venv\Scripts\python.exe -m stockml.main backtest
.\.venv\Scripts\python.exe -m stockml.main model-report
.\.venv\Scripts\python.exe -m stockml.main predict

# Repair holes inside a symbol's own history
.\.venv\Scripts\python.exe -m stockml.main heal --dry-run

# Read-only API, then dashboard
.\.venv\Scripts\python.exe -m uvicorn stockml.api.server:app --port 8000
cd stock-direction-dashboard ; npm run dev

# Track Q
.\.venv\Scripts\python.exe -m stockml.qpipeline --status
```

---

## 17. If you remember only ten things

1. The task is 3-class direction (**UP / DOWN / FLAT**) for 50 NIFTY stocks, 5 sessions ahead.
2. The model is **LightGBM**: hundreds of small trees, chosen for speed, tabular fit, and explainability.
3. The current model is **worse than the majority baseline** (47.1 % vs 47.9 %), and it mostly predicts DOWN.
4. Labels come from a **triple barrier** scaled by each stock's own volatility, with a ±1σ flat band.
5. Validation is **purged walk-forward**, so overlapping 5-day labels cannot leak the future into the past.
6. There are **no embeddings, no chunking and no vector database** anywhere.
7. The LLM is **Qwen3 8B, Q4_K_M, via Ollama, 2048-token context**, used as a *voter*, not a decider.
8. **Phase Q0** is running right now (716/1500) and its only question is "does Qwen beat FinBERT by more than 2×SE?"
9. The dashboard is **read-only** and every panel maps to a file on disk.
10. Everything here is designed so a failed idea is **visible and cheap**, not hidden.

---

## 18. Every graph: what it draws, and what the other shape would mean

Two kinds of graph live here, drawn by different code:

| Kind | Drawn by | Where | Look |
| --- | --- | --- | --- |
| **Live charts** | Recharts, `stock-direction-dashboard/src/components/charts.jsx` | the 8 dashboard pages | grayscale, hairline grid, tooltip on hover |
| **Saved images** | matplotlib, inside `evaluate.py` and `backtest.py` | `results/*.png` | 160 dpi, written fresh on every run |

Two rules before reading any of them:

1. **Colour is never the information.** Every live chart is grayscale by construction. Series identity is carried by **line weight and dash pattern** — solid first, long dashes second, dotted third. Read the legend, not the shade.
2. **Every panel has an `info` circle** whose tooltip says *what it shows · how it is read · what ideal looks like · when to worry*. This section is the long version of those tooltips, plus the part the tooltips cannot hold: **what the opposite shape would prove**.

Each graph below gets three things: **what it plots**, **what you are looking at today**, and **what the other shapes would mean**.

### 18.1 Equity curve — `EquityChart` (Backtesting) and `results/equity_curve.png`

**What it plots.** X = exit date. Y = account value starting at **1.00**. Area chart, solid line, faint wash underneath, **dashed horizontal reference at 1.00** (break-even). A second dotted line carries drawdown.

**How it is built** (`backtest.py`): each accepted trade's net return is summed **per exit date**, then compounded — `equity = (1 + daily_return).cumprod()`. So the line is what ₹1 would have become, taking every call the model made, paying costs, reinvesting everything. The last point equals `1 + total_return`.

**Today:** starts at 1.00, ends at **0.627** — a **−37.3 %** compounded loss over 3,080 trades, with the dashed break-even line above the curve for almost the whole window.

| If the shape were… | What would then be true |
| --- | --- |
| Rising steadily above 1.00 | The signal is paying for its own costs. Next questions: is it above the **majority baseline**, does it hold in every fold, and is the gain spread across symbols? |
| Flat around 1.00 | Gross edge ≈ costs. Either it trades rarely or the winners and losers cancel. Check the trade count before celebrating. |
| **Falling in a smooth line** (today) | Costs are the drag, not the direction: gross **+15.4 %**, cost paid **61.4 %**, net sum **−46.1 %**, compounded **−37.3 %**. |
| Falling with a few vertical cliffs | A handful of exit dates caused the damage. Read the per-symbol table and the trade log for those dates. |
| Rising with deep saw-tooth dips | Profitable but fragile. The drawdown chart, not this one, tells you whether you could hold it. |

**Small trap:** the Backtesting page shows both a *sum* (gross / cost / net, from the "Where the return goes" panel) and a *compound* (this curve). They will not match, and that is correct — sums add, the curve multiplies.

### 18.2 Drawdown — `results/drawdown.png` and the dotted line inside `EquityChart`

**What it plots.** X = exit date. Y = `equity ÷ highest_equity_so_far − 1`. It is always ≤ 0, and it snaps back to 0 every time a new high is printed.

**Today:** max drawdown **−38.0 %**, slightly deeper than the total loss — the signature of a path that never recovers.

| If the shape were… | What would then be true |
| --- | --- |
| Hugging 0 most of the time | The account sits at or near its best level — a healthy curve. |
| One sharp V, then recovery | A single bad episode. Find the symbol or event behind that date. |
| A long flat stretch near 0 | No new high for a long time: the strategy has stopped earning even if it is not falling. |
| **Stair-stepping down, never recovering** (today) | After the early part of the run, no new high is ever printed. |
| One vertical drop to −0.9 | Concentration or a cost bug, not a market event. Investigate before trusting anything else on the page. |

### 18.3 Rolling accuracy — `results/rolling_accuracy.png`

**What it plots.** X = test row (1…N, not a date). Y = accuracy over a rolling window of `min(50, max(10, N/10))` rows — **50 rows** here. A **dashed horizontal line** is the majority-class baseline (0.4785), deliberately not a flattering 0.5 line.

**A detail worth knowing:** rows are ordered by date then symbol, and there are 50 symbols per date, so a 50-row window is roughly **one trading day**. This line is nearly a daily hit-rate chart, and on days when all calls are DOWN it really measures "did the market fall?".

**Today:** the line crosses the baseline constantly and spends as much time below it as above; the overall mean is **0.4714**, i.e. below the line.

| If… | Then… |
| --- | --- |
| Consistently above the dashed line | A real edge. Then demand significance (the DM test exists for exactly this) and check it survives costs. |
| **Below the line for long stretches** (today) | The model adds noise rather than signal. |
| Jagged 0 → 1 swings | The window is too short for the number of rows. Widen it before reading anything into it. |
| Smooth, slowly decaying | Drift: the learned relationship is fading. Cross-check the PSI column on the Training tab. |
| One huge spike | One easy date (all DOWN, market fell). A spike is not evidence of skill. |

### 18.4 Prediction correctness strip — `results/prediction_correctness.png`

**What it plots.** X = test row. Y = the **predicted** class (0 = DOWN, 1 = FLAT, 2 = UP). Every row is one mark: **○ correct, × wrong**. It is a strip plot, so density is the message, not the height.

**Today:** a dense band at 0 that is mostly ○, and two thin bands at 1 and 2 that are mostly ×. Same story as the confusion matrix: it makes DOWN calls and misses most UP rows.

| If the marks were… | What would then be true |
| --- | --- |
| Spread evenly across the three bands | All three classes are in use. Judge accuracy separately — coverage says nothing about correctness. |
| One band, all ○ | Either an unusually easy period or leakage. Verify against the split dates before believing it. |
| × clustering in the last third | A regime change late in the test window. Compare feature PSI at those dates. |
| The band cycling 0 → 2 → 0 | The model is following momentum, not direction. The rolling-share chart will show the same oscillation. |

### 18.5 Confusion matrix — `ConfusionMatrix` (Model evaluation; Model behaviour; Training tab) and `results/confusion_matrix.png`

**What it plots.** A 3×3 grid. **Rows = actual class, columns = predicted class.** Cell shade is `count ÷ largest cell` (darker = more rows), and **the number is printed in every cell**, so nothing depends on the shade alone. The dashboard title spells the orientation out: `actual ↓ / predicted →`.

**How to read one cell.** Read the **row first**: the row is the truth, the column is the guess. Row DOWN, column UP = "rows that really were DOWN and the model called UP" — a miss, and an expensive one because the trade goes the wrong way.

**Today** (test set, 37,292 rows):

| actual ↓ \ predicted → | DOWN | FLAT | UP | row total |
| --- | --- | --- | --- | --- |
| **DOWN** | **14,339** | 649 | 2,858 | 17,846 |
| **FLAT** | 1,787 | **447** | 369 | 2,603 |
| **UP** | **13,392** | 658 | **2,793** | 16,843 |
| column total | 29,518 | 1,754 | 6,020 | 37,292 |

The single darkest cell is **actual UP → predicted DOWN: 13,392**. Four out of five UP rows are called DOWN. That one cell is the whole diagnosis of this model.

| If the grid looked like… | What would then be true |
| --- | --- |
| A dark diagonal, pale elsewhere | Real skill on all three classes. Then check costs, because a correct direction can still be too small to pay the round trip. |
| **One dark column** (today: DOWN, 29,518 of 37,292) | The model has collapsed onto a class. It is not classifying, it is guessing the most common answer and dressing it up as a probability. |
| A nearly empty FLAT row and column | The rare class is ignored. Expected here: FLAT is ~4 % of labels (§4.3). |
| Off-diagonals roughly equal in both directions | The model is symmetric but noisy — better than collapse, still no edge. |
| Numbers only on the diagonal and nowhere else | Either a very easy period or leakage. Check the split dates and the holdout log before celebrating. |
| Column totals ≈ row totals for all three | The predicted mix matches reality. That alone is not skill, but its absence is a red flag. |

### 18.6 Reliability diagram (calibration) — `CalibrationChart` (Model evaluation) and `results/calibration.png`

**What it plots.** X = **mean predicted confidence** inside a bin. Y = **realised accuracy** in that same bin. Dots are bins, joined by a line. A **dashed diagonal from (0,0) to (1,1)** is perfect calibration: "when I say 60 %, I am right 60 % of the time". The saved PNG adds a **red vertical stem per bin** — the visual size of each gap — and prints ECE in the title.

**Today** — this is the most informative chart on the evaluation page, because the numbers behind it are not what the headline ECE suggests:

| Confidence bin | Rows | Mean confidence | Realised accuracy | Gap |
| --- | --- | --- | --- | --- |
| 0.30–0.40 | 3,837 | 0.378 | 0.390 | **+1.1 pp** |
| 0.40–0.50 | 17,072 | 0.451 | 0.462 | **+1.0 pp** |
| 0.50–0.60 | 12,045 | 0.544 | 0.484 | **+6.0 pp** |
| 0.60–0.70 | 3,794 | 0.637 | 0.501 | **+13.6 pp** |
| 0.70–0.80 | 472 | 0.730 | 0.498 | **+23.2 pp** |
| 0.80–0.90 | 66 | 0.835 | 0.561 | **+27.4 pp** |
| 0.90–1.00 | 6 | 0.909 | 0.500 | **+40.9 pp** |

The curve therefore looks **honest on the left and flattens on the right**: accuracy rises to about 0.50 and then stops, while confidence keeps climbing past 0.90. Overall ECE is a decent-looking **0.042** only because 56 % of all rows sit in the carefully calibrated 0.40–0.50 bin.

**The conceptual point, which people get wrong all the time:** calibration measures whether the *number* is honest, not whether the *model* is good. A model that always says "47 % DOWN" and is right 47 % of the time is perfectly calibrated and completely useless. Today's model is roughly calibrated where the mass is, and badly over-confident in the tail.

| If the dots were… | What would then be true |
| --- | --- |
| **On the diagonal** | The confidence numbers can be taken at face value. |
| **Above the diagonal** | Under-confident: the model is better than it claims, and a sharper temperature would find accuracy. |
| **Below the diagonal** (today, past 0.50) | Over-confident: it promises more than it delivers. Any rule that says "act at ≥ 0.60" is built on a promise of ~60 % that historically delivered **~50 %**. |
| A **flat horizontal line** | The probability carries no ranking information at all: every confidence level is right about 47 % of the time. This is the shape to fear, and it is close to what the right-hand bins show. |
| A **downward-sloping** curve | Worse than useless: the model is most confident exactly where it is most wrong. |
| Only two or three dots | Not enough bins with data. Read the table, not the picture. |

**Cross-check the calibration panel against §18.7.** Good calibration with a flat top means "honest about being weak", which is exactly what this project reports.

### 18.7 The numbers panel beside the charts (precision, recall, MCC, AUC, DM)

Not a graph, but it is the panel people screenshot, so it gets the same treatment.

| Number | Plain meaning | Today |
| --- | --- | --- |
| **Precision** | Of the rows it labelled X, how many really were X | DOWN 0.486 · FLAT 0.255 · UP 0.464 |
| **Recall** | Of the rows that really were X, how many it found | DOWN **0.803** · FLAT **0.172** · UP **0.166** |
| **Macro F1** | Unweighted average of the three classes' F1 | 0.352 |
| **MCC** | One number for "better than chance", −1 to +1, 0 = guessing | **0.037** |
| **AUC (one-vs-rest)** | Ranking quality, 0.5 = coin flip | **0.554** |
| **Directional accuracy (when it commits)** | Accuracy on the 33,382 rows where it named a side | **0.513** |
| **Coverage** | Share of rows where it named a side rather than FLAT | 0.953 |
| **Diebold–Mariano on UP** | Is the difference from the baseline statistically real? | stat **−4.83**, p **1.4e-6** |

| If the number were… | What would then be true |
| --- | --- |
| **Both precision and recall high for a class** | The model genuinely recognises that class. |
| **High recall, middling precision** | It finds most of that class, but only half of its calls are right — exactly DOWN here (recall **0.803**, precision **0.486**). It is over-calling DOWN to harvest recall. |
| **High precision, low recall** | When it says X it is usually right, but it rarely says X. That is the shape you *want* before opening the threshold. |
| **Both low** | The class is effectively unlearned — FLAT (0.255 / 0.172) and UP (0.464 / 0.166). |
| **MCC ≈ 0** (today 0.037) | No agreement with reality beyond chance, whatever accuracy says. |
| **AUC in 0.50–0.55** (today 0.554) | The ranking of the three probabilities is barely better than random. |
| **DM stat positive with p < 0.05** | The model is *significantly better* than the baseline. Today the stat is negative with p ≈ 10⁻⁶, i.e. **significantly worse** — a clear result, just not the one anyone wanted. |
| **DM p > 0.05** | No detectable difference. "We cannot tell yet" is a legitimate answer and must not be written up as a win. |

### 18.8 Feature importance — `ImportanceBars` (Model evaluation, Training tab) and `results/shap_summary.png`

**What it plots.** Horizontal bars, length = **mean absolute SHAP value** on a 1,000-row sample of the test set. The page shows the top 15 by default (toggle 15 / 25 / 40); the saved PNG is the top 20, drawn with the largest bar at the top. On the Training tab the same ranking is drawn again with a **PSI badge** next to each feature.

**SHAP in one line:** each row's prediction is split into per-feature contributions; mean |contribution| over many rows says how much that feature *moves* the model, regardless of direction.

**Today:** importance is spread over many columns, and the PSI badges flag features whose live distribution has drifted away from training.

| If the shape were… | What would then be true |
| --- | --- |
| **Spread across many explainable features** | The model is using many small signals — the honest, if weak, situation we are in. |
| **One bar 30–40 % of the total** | Suspect a proxy for time (a row counter, an index level, a date-derived number) rather than market information. Check that feature against §5.1 "point-in-time". |
| **All bars almost equal, all small** | Nothing carries signal. Accuracy will sit at the majority class. |
| **A feature you cannot explain on top** | Either a genuine discovery or a leak. Prove it with the waterfall before trusting it. |
| **PSI above the threshold on top features** | The inputs moved underneath a model that has not been retrained. Any accuracy on old data is historical only. |

**What this chart cannot tell you:** direction. A big bar means "this column matters", not "high values push UP". For direction use the signed drivers (§18.14) or the waterfall (§18.16).

### 18.9 Predicted label distribution — `DistributionBars` (Overview, Predictions)

**What it plots.** Three horizontal bars, one per class, width = share of rows, with the raw count printed on the right. Plain divs, no chart library. On Predictions the same panel also prints "as of", the confidence threshold and the flat band.

**Today** — `results/latest_predictions.csv`, all 50 symbols on the latest date:

| Class | Rows | Share |
| --- | --- | --- |
| DOWN | **45** | 90 % |
| UP | 3 | 6 % |
| FLAT | 2 | 4 % |

Mean confidence **0.475**. The intended mix is roughly **48 % DOWN / 5 % FLAT / 47 % UP**, so this is a collapse: it is calling almost everything DOWN on a date where, historically, about 45 % of rows turn out UP.

| If the bars looked like… | What would then be true |
| --- | --- |
| **Close to 48 / 5 / 47** | The model is not collapsed. It says nothing yet about being right. |
| **Like today (90 % DOWN)** | Collapse onto one class. Useful as a tell, useless as a forecast. |
| **100 % FLAT** | τ = 0.40 has swallowed every row: no probability reached a decision. Lower τ, or the probabilities are too flat to use. |
| **100 % UP** | The opposite collapse — check the temperature and the class weights. |
| **A different mix every single day** | The serving path is unstable, or forecasts come from different model versions. Check the consistency banner (§14.5). |
| **Mix matches reality exactly** | Only possible if the model is trivial or something leaks. Verify before trusting. |

### 18.10 τ sweep — `SweepChart` (Model behaviour → Behaviour)

**What it plots.** X = **τ_flat**, the flat threshold, swept over a grid. Three lines on a 0–1 axis: **accuracy** (solid), **directional coverage** (long dashes), **recall for UP** (dotted). A vertical dashed reference marks **the τ actually in use (0.40)**.

**How the ends behave.** At τ = 0 nothing can be FLAT, so coverage = 1 and accuracy = pure directional accuracy. At τ = 1 almost everything is FLAT, so coverage → 0 and accuracy → the FLAT share (~7 %). Everything interesting is in between.

**Today:** at τ = 0.40 the coverage is **95.3 %** and accuracy-when-committed is **51.3 %** — the threshold is barely doing any work; only 4.7 % of rows become FLAT.

| If the curves were shaped… | What would then be true |
| --- | --- |
| **Accuracy rising as coverage falls** | Abstention is working: the model is buying accuracy by refusing to answer. This is the desirable shape, and it is *not* what we see. |
| **Accuracy flat across all τ** | The probabilities do not rank rows. Lowering or raising τ only moves how often it abstains, never how right it is. |
| **Coverage collapsing before accuracy improves** | The worst trade: you give up most of your calls and learn nothing extra. |
| **Recall UP rising with τ** | τ is spending coverage to rescue the UP class — a targeted fix for the collapse in §18.5. |
| **Everything pinned at 1.0** | Mostly FLAT predictions; the model refuses to decide. |
| **A sharp cliff at one τ** | A handful of rows sit exactly at that boundary. The curve is then a step function and small data changes will move it a lot. |

### 18.11 Behaviour plate, panel A — paired share bars (`PairedShareBars`)

**What it plots.** For each class, **two bars**: a faded one = share of **actual** rows, a solid one = share of **predicted** rows. The right-hand number is the difference in **percentage points**, bolded once it passes ±10 pp.

**Today** (test split):

| Class | Actual share | Predicted share | Bias |
| --- | --- | --- | --- |
| DOWN | 47.9 % | 79.2 % | **+31.3 pp** |
| FLAT | 7.0 % | 4.7 % | −2.3 pp |
| UP | 45.2 % | 16.1 % | **−29.0 pp** |

**That single panel is the whole model in three lines**: it issues 31 points too many DOWN calls and 29 points too few UP calls.

| If the bars were… | What would then be true |
| --- | --- |
| Equal in all three rows | The predicted mix matches reality. Necessary, nowhere near sufficient. |
| Solid longer than faded (today, DOWN) | Over-prediction of that class. |
| Solid shorter than faded (today, UP) | Under-prediction — rows that should be found are being missed. |
| All biases under ±5 pp but accuracy still poor | The model reproduces the *marginal* mix without telling rows apart: a prior-only model that learned the base rate and nothing else. |
| Bias flipping sign between train and test | The class balance shifted, or the model is unstable. Retraining on the same window will not fix it. |

### 18.12 Behaviour plate, panel C — class profile and separation (`GroupedProfileBars`)

**What it plots.** One row per **actual** class. Inside each row, three bars = the **mean predicted probability** of DOWN, FLAT and UP for the rows in that class. Below them the panel prints one number, **separation**: 0 = the three profiles are identical, 1 = completely disjoint.

**Ideal picture:** inside "actual UP", the third bar (p_up) is clearly the tallest; inside "actual DOWN", the first bar is. The three rows look different.

**Today:** because 79 % of all rows are called DOWN, the three rows look almost the same — the DOWN bar dominates in all of them. The panel's own numbers say it plainly: separation **0.0366** on test (0.0195 on calibration, 0.1875 on train), against a scale where 1.0 means the classes are completely distinguishable. The report also stamps the split with a mode name, and on both calibration and test that name is **`collapsed_prior`** — the code's own word for "it is reporting the base rate". The source comment says this out loud: *flat, overlapping bars mean the model says the same thing whatever is true.*

| If the profiles were… | What would then be true |
| --- | --- |
| Clearly different per row | The model separates the classes; combine with precision/recall to see how well. |
| **Nearly identical** (today) | The prediction barely depends on the inputs — it is reporting a prior, not a classification. |
| **Inverted** (actual UP rows get the highest p_down) | Anti-signal. Suspect a label/sign bug before believing it. |
| Only one bar non-zero everywhere | Total collapse onto that class. |
| One profile high, others flat at the base rate | Only one class is being learned; the rest are defaulted. |

### 18.13 Behaviour plate, panel D — conviction histogram (`ConvictionHistogram`)

**What it plots.** Bins of **max(p)** — the probability of the winning class, i.e. "how sure was it?" — with the share of rows in each bin. The footnote prints two numbers: the share of calls with max(p) < 0.50, and the mean max(p).

**Today:** mean max(p) = **0.4975**, and **55.6 %** of all test calls had max(p) below 0.50 (66.1 % on train, 58.7 % on calibration). In plain words: **more than half the predictions are decided by a coin-flip margin.** The winning class is often below 0.50 — closer to a three-way tie than to a decision.

| If the mass sat… | What would then be true |
| --- | --- |
| At 0.33–0.40 | The model sees three-way ties everywhere. There is no information in the probabilities. |
| **Spread with the bulk under 0.50** (today) | Weak conviction across the board; the model is guessing and saying so. |
| At 0.60+ *and* calibration confirms 60 % | Genuine confidence. This is what would make the ≥0.60 trade rule meaningful. |
| At 0.60+ *but* accuracy hits 50 % | Confident noise — today's tail (§18.6). The worst of all worlds. |
| In two clumps (near 1/3 and near 0.9) | Two behaviours mixed: rows it refuses to decide, and rows it is certain about. Split them and look at what the certain ones have in common. |

### 18.14 Behaviour plate, panel E — signed drivers (`SignedDriverBars`)

**What it plots.** For the class this split **over-produces** (the "dominant" class, DOWN today), the **mean signed SHAP** per feature. A bar to the right pushes toward that class; a bar to the left pushes away. The number is the mean contribution.

**Today:** the drivers explain *what makes the model say DOWN* — read them as "the evidence it thinks it sees", not as market truth.

| If the bars were… | What would then be true |
| --- | --- |
| Mostly one direction, a few strong | The class has a signature the model recognises, even if the class is over-called. |
| Balanced left and right, all small | The class is chosen by the *absence* of the others rather than by positive evidence — the typical fingerprint of a collapsed class. |
| Dominated by a time-like feature | Leak. Check it against §4.6 before using anything else on the page. |
| Different drivers in train and test | The learned relation is not stable across the split. Retraining on the same window will reproduce it, not fix it. |

### 18.15 Rolling predicted share and accuracy (`RollingLine`, Behaviour tab)

**What it plots.** Three rolling series over the split — **rolling share of DOWN calls**, **rolling share of UP calls**, **rolling accuracy** — plus a **dashed reference line** that is *not* 0.5 but the **actual DOWN share for the whole split** (about 48 %).

**Today:** the DOWN-share line runs far above the dashed reference almost everywhere; the UP-share line runs far below. This is §18.11 drawn as a time series, which is why it is worth looking at: it tells you *when* the over-calling happens, not just that it exists.

| If the lines were… | What would then be true |
| --- | --- |
| Predicted DOWN share tracking the dashed line | The mix adapts correctly over time. |
| **Pinned above the reference for the whole window** (today) | A structural bias, not a bad week. No amount of extra data will average it out. |
| Rolling accuracy decaying toward the right edge | Drift late in the test window; match those dates against PSI. |
| DOWN and UP shares oscillating in opposite phase | The model flips between "call everything DOWN" and "call everything UP" phases — the signature of a momentum follower. |
| All three lines flat and boring | The model is not reacting to anything; check that features actually vary by date. |

### 18.16 Waterfall — one case explained (`Waterfall`, case drawer)

**What it plots.** One row's journey from the model's **base value** to the **raw score** of one class. Each step is a feature contribution: **ink = pushes toward the class, grey = pushes away**, with the value printed on the right. The axis is symmetric around zero, so left and right are visually comparable.

**How it is built:** the steps were persisted sorted by |contribution| and are re-drawn by accumulating `from → to`, so the bars literally stack. The last line must equal the score the model used.

| If the picture showed… | What would then be true |
| --- | --- |
| Many medium steps, both directions | A normal, multi-feature decision. |
| **One giant step** | Effectively a single-feature decision. If that feature is a time proxy, the whole model is one clock. |
| Big positive and big negative steps that cancel | An unstable call: small noise would flip the answer. Do not act on it. |
| The base value already far from zero | The model starts biased toward this class before it looks at any feature; the inputs only nudge a decision that was already made. |
| Steps not adding up to the printed raw score | The report is stale relative to the model on disk. Re-run `model-report`. |

### 18.17 Fold recall heatmap (`FoldRecallHeatmap`, Training tab)

**What it plots.** The last 40 **folds** (each is one trial × one fold), drawn left to right oldest → newest. Each fold is three stacked cells: **DOWN / FLAT / UP recall**. Darker = higher recall. Hovering shows trial, fold and the fold score.

| If the strip looked like… | What would then be true |
| --- | --- |
| Three consistent, moderately dark rows | The model generalises across time — the strongest evidence this project can produce. |
| **The FLAT row pale everywhere** | The rare class is never learned. Expected (§4.3), and it is why a separate decision rule exists. |
| Dark on the left, pale on the right | Decay over time: drift or a regime change. Cross-check with the monitoring page. |
| Dark and pale alternating fold by fold | The fold result depends on which months landed in it, not on learning. A single fold score proves nothing. |
| Only the DOWN row ever dark | Collapse visible at training time, before any serving question arises. |

### 18.18 Loss curves — train vs calibration (`LineSeriesChart`, Training tab and Live tab)

**What it plots.** X = **iteration** (written every 10 rounds), Y = **multi-class log loss**. Two lines: **train** (solid) and **calibration** (long dashes). The same chart exists twice: for the finished run on disk, and for the run currently in flight.

**One thing that must not be misread:** the calibration set is attached as a *watched* set. It is drawn to inform you; it is not used as an early-stopping rule, so it cannot quietly shape the model.

| If the curves… | What would then be true |
| --- | --- |
| Both fall, then flatten | Healthy training. The flattening point is the useful round count. |
| Train keeps falling while calibration turns up | Overfitting in progress. The divergence point is the honest stopping point. |
| Both flat from round 1 | No signal, a learning rate that is too small, or features that do not vary. |
| Calibration line much noisier than train | A small calibration set. It cannot corrupt the model here, but do not read small wiggles as trend. |
| Both falling to a plateau well above the FLAT-only loss | The model is not even reaching a trivial solution. That is a data problem, not a tuning problem. |

### 18.19 Track Q category bars — `ShareBars` (Analytics tab)

**What it plots.** Four small panels — **votes**, **pools**, **drop reasons**, **top symbols**. Each is a sorted list: bar width = that category's **share of the panel total**, with the raw count on the right. Top 4–5 categories only.

| If a panel were… | What would then be true |
| --- | --- |
| Votes spread over DOWN / FLAT / UP | The model is actually choosing. |
| **Votes dominated by FLAT** | It is hedging — the safest answer under a strict JSON schema. Check the prompt and the thinking budget before calling it a model failure. |
| One pool dominating | The pool is unbalanced. Pool C should be roughly 500 DOWN / 250 FLAT / 750 UP. |
| One drop reason at 80 % | A harness or schema bug, not a judgement about the news. Fix the harness first. |
| Top symbols concentrated in three names | The sample is not diverse, so any gate result describes those three companies. |

### 18.20 Track Q progress chart (Analytics tab)

**What it plots.** X = sequence number, three lines: **rated**, **judged**, **kept** (dotted). Rated = a row exists; judged = a usable vote; kept = passed the quality filter.

| If the lines were… | What would then be true |
| --- | --- |
| Rated and judged rising together | The harness is accepting almost everything. |
| **Judged well below rated** (today, 652 vs 716) | ~9 % of outputs are malformed. Expected, and it is why both numbers are shown. |
| Kept much below judged | The quality filter is doing heavy work — check the drop-reason panel. |
| All three flat | The run has stalled. Check the heartbeat's `updated_at`, never the file's modified time. |
| Judged above rated | Impossible — a display or join bug. |

### 18.21 Track Q gate bar (Status tab)

**What it plots.** Not a chart library: a **horizontal bar for Δ** (model score minus baseline score) drawn against a **2×SE band**. The gate passes only when Δ clears that band *and* the recorded pass flag says so.

| If the bar were… | What would then be true |
| --- | --- |
| Δ positive and longer than the 2×SE band | A real win at the agreed bar. Record it with the version and the run id. |
| Δ positive but inside the band | **Inconclusive** — the honest label, and the one the dashboard uses. Do not round it up to "pass". |
| Δ negative | Rejected. The LLM is worse than FinBERT on this pool. |
| No bar at all | Not enough judged rows yet (652 of 1,500 today). Nothing can be concluded. |

### 18.22 Track Q telemetry — five live lines (Live tab)

All five are wall-clock line charts on a 5-second sample; the x-axis is local time, while the heartbeat timestamps are UTC. Convert before comparing the two.

| Chart | What it plots | Ideal shape | If it looked otherwise |
| --- | --- | --- | --- |
| **GPU temperature** | °C per sample, 5 s apart | A line that plateaus below the limit, with saw-tooth drops = the guard resting the card | A climbing line with no saw-tooth means the guard is not firing; a flat line at 0 means the probe is dead |
| **Temp vs clock** (double axis) | Temp (°C, left) against SM clock (MHz, right), with a reference at the memory-clock cap | Temp rising while clock is *capped* — thermal throttling contained on purpose | Clock falling *before* the temp limit = a policy change, not heat; both rising together = overheating in progress |
| **GPU utilisation** | % busy per sample | High and steady while rating, with dips during rests | Pinned at 100 % for hours = no cooldown, heat will climb; pinned at 0 = nothing is running |
| **VRAM used** | MB allocated, with a reference line at total | A flat band well under total | Climbing toward total = leak or a bigger batch than planned; exactly at total = the next call will OOM |
| **Power vs fan** (double axis) | Watts (left) and fan % (right) | Power steady, fan responding to temperature | Power high with fan stuck low = fan curve problem; both spiking in sync with temp = the guard resting the card, which is normal |

### 18.23 Phase rows (Status tab)

Each phase is a **horizontal progress bar** with its step count, gate and status. A green phase with a full bar has finished; a bar that has not moved between two refreshes is stalled, which is a supervisor question, not a model question.

### 18.24 Which chart to open for which question

| Your question | Open |
| --- | --- |
| "Is it any good?" | Confusion matrix (18.5), then precision/recall (18.7) — never accuracy alone |
| "Is it collapsed?" | Distribution bars (18.9), paired shares (18.11) |
| "Can I trust the confidence number?" | Reliability diagram (18.6), then the conviction histogram (18.13) |
| "Would it have made money?" | Equity curve (18.1), drawdown (18.2), per-symbol table |
| "Why did it say UP for this row?" | Waterfall (18.16) |
| "Is it overfitting or decaying?" | Loss curves (18.18), fold heatmap (18.17), rolling accuracy (18.3) |
| "Is the input data still the data it learned on?" | PSI badges (18.8), then the Data and integration page |
| "Will Phase Q0 pass?" | The gate bar (18.21) — and only after 1,500 iterations |

**One footnote for completeness:** `charts.jsx` also defines a tiny `Sparkline` component. No page currently imports it, so you will not see it on screen — it is available for future table cells.

---

## 19. The architecture on one page

Everything in §2–§14 is one system. This section draws it once, as a single diagram, and then answers the five questions the picture has to settle:

1. how input gets to the model,
2. which model is touched **first**,
3. how votes are counted,
4. when the LLM is used, and for what,
5. how a decision is actually taken.

The diagram itself is a **Mermaid block** — plain text, rendered by the editor — and it lives in [`architecture.md`](architecture.md), drawn top to bottom. If a step changes, the diagram is edited in the same commit as the code. No image to re-draw, and only one copy to keep current.

### 19.1 The diagram

The picture lives in its own file: **[`architecture.md`](architecture.md)**. It is the same 36 nodes, drawn **top to bottom** with one lane per block, and it carries the legend.

It is kept in one place on purpose. Two copies of a 48-edge diagram drift the first time a step changes, so `architecture.md` holds the drawing and the lane table, and this section holds the prose that explains it (§19.2–§19.6). Every node number below is the number printed in that diagram.

### 19.2 The five questions, answered by position

| Question | Short answer | Where to look |
| --- | --- | --- |
| How does input get to the model? | Six external sources are downloaded once per run, aligned to a `decision_date`, turned into 183 columns, then cut into the fixed 136-column matrix | 1–6 → 7 → 8 → 9 → 10 → 11 → 12 |
| Which model is touched **first**? | **FinBERT** (15). It runs during data preparation, long before training, and its output is *a feature*, not a prediction | thick arrow 15 → 9 |
| How are votes counted? | LightGBM: every tree casts a weighted vote per class; the votes are summed into 3 raw scores, then softened into probabilities | 16 → 18 → 19 |
| | Qwen: one JSON vote per iteration; the harness counts *usable* votes and compares agreement against FinBERT's | 31 → 32 → 35 |
| When is the LLM used, and for what? | Only in lane 5, only as a **rater to be measured** — never in the path that produces a prediction or a trade | entire lane 5 |
| How is a decision taken? | Two-step rule on the probabilities, then a confidence gate; separately, a research gate for whether a track continues | 19 → 21 → 22, and 25 / 35 |

### 19.3 Two different things are called "a decision"

The word hides two unrelated acts. Keeping them apart is what stops the LLM from being confused with the predictor.

|  | The per-row trading decision (21–22) | The research gate (25, 35) |
| --- | --- | --- |
| Asked of | one row: one symbol on one date | a whole experiment, once |
| Rule | FLAT if p(FLAT) ≥ 0.40, else the larger of p(DOWN) and p(UP); trade only if that probability ≥ 0.60 | a bar fixed **before** the run (G1/G2/G3, or "beat FinBERT by more than 2 × standard error") |
| Possible outputs | DOWN / FLAT / UP, plus a confidence number | PASS / FAIL / INCONCLUSIVE / PAUSE |
| Written to | `results/latest_predictions.csv` | `results/model_report.json`, `results/q0_d1.json`, the phase state file |
| Who can overrule it | nobody — it is arithmetic | the analyst, but only by recording why |

### 19.4 Where each lane lives in the code

| Lane | Code | Outputs |
| --- | --- | --- |
| 1 — INPUTS | `src/stockml/data/`: `fetch_data.py`, `bhavcopy.py`, `announcements.py`, `news.py`, `rss.py`, `gdelt_bq.py`; `config/universe*` | `data/raw/`, `data/reference/` |
| 2 — FEATURES, LABELS, SPLITS | `features/feature_engineering.py`, `features/cross_sectional.py`, `features/decision_date.py`; `targets/triple_barrier.py`; `validation/walk_forward.py`, `validation/time_splits.py` | `data/features/`, `data/processed/` |
| 3 — MODELS | `models/lightgbm_train.py`, `models/baseline.py`; `evaluation/flat_threshold.py` (the 0.40 rule) | `artifacts/models/`, `artifacts/best_params.json`, `checkpoints/` |
| 4 — DECISION, BACKTEST, SURFACE | `inference/predict.py`, `v2/policy.py`, `backtest/backtest.py`, `evaluation/*`, `api/server.py`, `stock-direction-dashboard/src/` | `results/`, and the browser |
| 5 — TRACK Q | `data/qwen_harness.py`, `data/qwen_rater.py`, `data/qwen_nightly.py`, `qpipeline/runner.py`, `data/leakage_scanner.py` | `data/llm/`, `results/traces/`, `results/q0_d1.json` |

### 19.5 What the picture deliberately does not contain

None of these exist in this system, so drawing them would be a lie: **embeddings, chunking, a vector database, RAG, retrieval, a message queue, Kafka, Kubernetes, cloud storage, a second database, a GPU cluster.** There is also **no fine-tuned LLM in the decision path** — QLoRA code exists (`training/qwen_qlora.py`) but nothing on the path to node 21 touches an adapter.

The reason is simple and worth stating once: this model reads **numbers in a fixed table**, not text passages. Retrieval only helps when the answer lives in documents that must be searched; here every input is already a column, and the one text source (announcements) is compressed to sentiment numbers by FinBERT at node 15.

### 19.6 Two corrections made while drawing this

1. **Announcement text enters from node 3, not node 1.** Yahoo Finance (node 1) publishes prices, never announcement text; the text comes from NSE announcements, so the arrow runs 3 → 15.
2. **The feature count is 183, not "137 + 21 + 4 = 190".** That sum is 162, and neither number matched the file. Measured on `data/features/stocks/ADANIPORTS.NS_features.parquet`: **183 columns** (60 price/volume/delivery, 51 news and announcement, 34 market, 21 cross-sectional, 17 calendar), of which **136 are pinned** for the booster and **47 are unused** (42 news/announcement + 5 delivery/volume). §3, §4, §13 and §15 now state 183 and 136 the same way, so the document agrees with itself.

