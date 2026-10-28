# Data freshness and backfill runbook

Answers two questions: **why does the data go stale**, and **which gaps can
actually be filled**. Everything here is measured against the artifact tree in
this repository; the dashboard's **Data & integration** page renders the same
facts live from `GET /dashboard/data-health`.

## 1. Short answer

| Problem | Repairable? | How |
| --- | --- | --- |
| Panel ends weeks before the last NSE session (a *tail* gap) | **Yes, fully** | `.\refresh_data.ps1` — `fetch` resumes at `last_date + 1` |
| Sessions missing in the *middle* of a series (a *hole*) | **Yes, fully** | `heal` runs first and re-downloads a window around each hole |
| Deterministic features, labels, cross-sectional context, predictions | **Yes, always** | Rebuilt from raw data by the same refresh |
| NSE announcements | Mostly | Paginated API, but bounded by NSE retention |
| News articles from AlphaVantage | **No** | 25 requests/day, newest-first; older articles are gone |
| RSS headlines | **No** | today's feeds only |
| Bulk / block deals | **No** | scraped per day, file-per-day, no history endpoint |
| GDELT news (DOC API) | Last 90 days only | `news_gdelt_days: 90` |
| GDELT BigQuery | Yes, at a cost | `--include-gdelt`, ~164 GB scan per run |

The one-line rule: **prices and everything derived from them are always
recoverable; "today's snapshot" sources are not.**

## 2. What "stale" means

`expected_session` is the last NSE session whose close is in the books: the
previous weekday before 18:30 IST, otherwise today. Each artifact is then
compared against *its own* expectation, not against `expected_session` blindly:

- raw prices, announcements, news → expected at `expected_session`
- the feature panel → `expected_session` (it is built from raw prices)
- processed datasets/labels → `expected_session - triple_barrier_horizon`
  (5 sessions), because a label needs the next five closes to exist

That last point matters: a label file that ends five sessions before the
features is **fresh**, not stale. `data-health` encodes this in
`label_expected_session` and a test pins it
(`tests/test_dashboard_api.py::test_label_horizon_is_not_reported_as_a_gap`).

If the stored trading calendar is itself behind, `expected_session` is
estimated from weekdays and the response sets
`expected_session_estimated: true`. The UI shows this, because an estimate and
a recorded calendar should not look identical.

## 3. Where the staleness came from

Diagnosed with `data-health` on 2026-10-28, before the repair run:

```text
expected_session : 2026-10-27   (calendar last stored session 2026-09-15, so estimated)
features as_of   : 2026-09-15   -> gap 30 sessions
raw prices       : 13 internal holes across the panel (calendar sessions inside a
                   symbol's own [first, last] range that the symbol was missing)
```

Two independent causes:

1. **No refresh ran for ~6 weeks.** `run_fetch` only appends forward, so nothing
   pulled 2026-09-16 → 2026-10-27.
2. **`run_fetch` cannot repair a hole.** It requests `last_date + 1 → today`. A
   session missing in the middle is never re-requested, no matter how often you
   refresh. That is why `heal` exists and why it runs *first*.

## 4. Fixing it

`refresh_data.ps1` wraps `python -m stockml.main refresh`:

```powershell
.\refresh_data.ps1                  # data + predictions; model and holdout untouched
.\refresh_data.ps1 -SkipNews        # fast path: skips AlphaVantage/RSS/bulk deals
.\refresh_data.ps1 -NoHeal          # forward-only top-up
.\refresh_data.ps1 -Retrain         # + train -> evaluate -> backtest, then predict
.\refresh_data.ps1 -DryRun          # print the step plan, touch nothing
```

Observed timings for a 30-session catch-up on this machine (2026-10-28):

| Step | Wall time | Notes |
| --- | --- | --- |
| `heal` | 47 s | 13 holes found, **13 repaired, 0 unrepaired** |
| `fetch` | 207 s | 2026-09-16 → 2026-10-28 per symbol, ~4 s/symbol |
| `bhavcopy` | 135 s | 28 archive days scanned, 28 hit |
| `calendar` | 0.05 s | 3 956 sessions |
| `announcements` | 286 s | ~1 000–5 000 rows per symbol |
| `news` | **hours**, not minutes | AlphaVantage quota + a full FinBERT re-score of the new corpus |
| `features`/`labels`/`predict` | seconds–minutes | deterministic rebuild |

The whole non-news path is ~12 minutes. News is measured in hours on the first
rebuild after a long gap (17 minutes for a single symbol on this machine,
because every newly collected article has to be scored), so run it separately:

```powershell
.\refresh_data.ps1 -SkipNews                    # close the price gap now (~12 min)
.venv\Scripts\python.exe -m stockml.main news   # let the text corpus catch up in the background
```

Skip the whole text branch on a schedule where the AlphaVantage quota (25
requests/day, shared by every symbol) matters, and skip `-SkipAnnouncements`
when the announcement fetch already ran today — the wrapper forwards both flags
to the CLI.

### Why clicking "train" alone did not fix it

`python -m stockml.main train` fits on whatever is already in
`data/processed/`. It never fetches. Training on stale features just produces a
stale model with a newer timestamp. Use:

```powershell
.\refresh_data.ps1 -Retrain     # refresh first, then train/evaluate/backtest/predict
```

`-Retrain` re-derives the 60/20/20 train/calibration/test split from the newly
extended dataset (`three_way_date_split`), so the sealed test window moves
forward and every reported metric changes. That is a modelling decision, not a
data refresh, which is why it is opt-in: the default refresh leaves the model and
the holdout window exactly where they are. The research-stage reads of that
window (i01-d1, down-flag, q8-ablation) are what append rows to
`results/holdout_touches.jsonl`; `train` itself does not.

## 5. Backfill matrix

| Dataset | Mechanism | Backfillable | Cost / limit |
| --- | --- | --- | --- |
| `data/raw/*.parquet` (OHLCV) | yfinance incremental append | Tail yes; holes only via `heal` | ~4 s/symbol; `heal` window is `750` sessions |
| `data/raw/macro.parquet` | same provider | Yes | excluded from hole scan (market series, not a symbol) |
| `data/raw/bhavcopy/*` | NSE archive, file per day | Yes from 2020-01-01 (`run_bhavcopy_fetch(start=…)`) | 404s before ~18:30 IST and on holidays → step is optional |
| `data/reference/trading_calendar.parquet` | reconstructed | Yes, fully | instant |
| `data/raw/announcements/*` | NSE paginated API | Mostly | bounded by NSE retention |
| `artifacts` ann scores (`ann-score`) | FinBERT over stored text | Yes, but slow | CPU/GPU minutes |
| `data/raw/news/*` | AlphaVantage + GDELT DOC | **No** (AV); 90 days (GDELT) | 25 AV req/day shared by all symbols |
| `data/raw/rss/*` | per-day RSS snapshot | **No** | today only |
| `data/raw/bulkdeals/*` | per-day NSE snapshot | **No** | today only |
| `data/raw/gdelt_bq*` | BigQuery export | Yes | ~164 GB scan per invocation |
| `data/features/*` | computed | **Yes, always** | rebuilds from raw |
| `data/processed/*` | computed | **Yes, always** | lags features by 5 sessions by design |
| `results/predictions*.parquet`, `latest_predictions.csv` | inference | **Yes, always** | last step of every refresh |

### Consequence for the missing 30 sessions

The price-derived features are complete after the refresh. The *text* features
for those sessions are the lossy part:

- AlphaVantage returned only its most recent articles, so per-symbol article
  counts for mid-September are lower than a same-day collection would have been.
- GDELT DOC covers the last 90 days, so the gap **is** inside its window and
  gets whatever it still has — but it is a rewritten index, not a replay.
- RSS and bulk deals for those days are permanently empty.

Downstream that means news-derived columns for the gap can be flat or NaN. If
they are features of the model, the model never saw that pattern in training and
the gap rows are weaker evidence than the surrounding ones. The honest handling
is to keep them and let the freshness panel flag the text steps as behind, not
to interpolate fabricated articles into the corpus.

## 6. Ordering guarantees

`build_steps()` fixes the order because the artifacts depend on it:

```text
heal -> fetch -> bhavcopy -> calendar -> announcements [-> ann-score]
     -> news -> rss -> bulkdeals [-> gdelt-bq]
     -> features -> cross-sectional -> labels
     [-> train -> evaluate -> backtest]
     -> monitor -> predict
```

- A step marked `optional` may fail without aborting the run; a required step
  failing stops the chain immediately (`status: "failed"`).
- `artifacts/state/refresh.json` is rewritten **before** each step, which is
  what lets the dashboard show a live step counter.
- `artifacts/state/refresh.lock` prevents two refreshes from writing the same
  parquet files. A lock older than 6 hours is treated as abandoned.
- `tests/test_refresh.py` pins the order, the optional/required split, and the
  "no retraining by default" rule.

## 7. Verifying the repair

1. `GET /dashboard/data-health` (or **Data & integration**):
   - `gap.gap_sessions_estimate == 0`
   - `holes.total_holes == 0`
   - `fetch`, `features`, `cross_sectional`, `labels`, `monitor`, `predict` report
     `status: "fresh"` with `lag_sessions: 0`
   - the `checks[]` block has no failing date check
2. `.\refresh_data.ps1 -DryRun` — the plan should list `heal` first.
3. `.venv\Scripts\python.exe -m stockml.main heal --dry-run` — should report no
   holes to repair.
4. `results/latest_predictions.csv` timestamp should equal the last feature
   session, and `data-health`'s `predictions_match_features` check should be
   `ok`. A mismatch means `predict` ran before `features` — rerun the refresh.

`summary.stale_steps` will usually be **non-empty even after a clean refresh**,
and that is correct. It names the things a data refresh deliberately does not
touch:

| Still listed | Why | Fix |
| --- | --- | --- |
| `train`, `evaluate`, `backtest`, `baseline` | judged by artifact *age* (30 days), not by session lag | `.\refresh_data.ps1 -Retrain` |
| `gdelt_bq` | skipped unless `-IncludeGdelt` | `python -m stockml.main gdelt-bq --force` (~164 GB) |
| `rss`, `bulkdeals` | today-only sources that were never collected | nothing — unrecoverable |
| `news` | the text corpus lags the price panel | `python -m stockml.main news` (hours) |

A measured end state after one data refresh on 2026-10-28:

```text
expected_session 2026-10-28   gap 0   holes 0
fresh : fetch, bhavcopy, calendar, announcements, features, cross_sectional,
        labels (2026-10-21, by horizon), monitor, predict
stale : news (3), gdelt_bq (28), rss (28), bulkdeals (28),
        baseline/train/evaluate/backtest (artifact age 38 d)
check : news_symbols_without_rows = false (6 symbols have no articles at all —
        a provider coverage limit, not a date gap)
```

## 8. Automation

The refresh is designed to be scheduled. Run it after 18:30 IST on weekdays so
the day's bhavcopy and the closing bar both exist:

```powershell
# Task Scheduler -> daily 19:00 IST
powershell -NoProfile -ExecutionPolicy Bypass -File "E:\books\Adaniports\stock_direction_platform\refresh_data.ps1" -SkipNews
```

Run the news path on a separate, less frequent schedule (weekly, or when the
AlphaVantage quota resets) because it is the only step that can burn an external
budget. `artifacts/state/av_quota.json` records the day's usage.

## 9. Failure playbook

| Symptom | Cause | Action |
| --- | --- | --- |
| `Another refresh appears to be running` | stale `refresh.lock` | confirm no python process is running, delete the lock, rerun |
| `status: "partial"` | an optional step raised | read `failures[]` in `refresh.json`; only rerun that step (`python -m stockml.main news`) |
| `status: "failed"` | a required step raised | fix the cause (calendar missing, disk full, model file absent), rerun from scratch — completed steps are idempotent |
| `heal` reports unrepaired holes | provider returned nothing for those dates | check whether the symbol was listed/suspended; NSE holidays are already excluded |
| gap reappears the next morning | the schedule is not running | check Task Scheduler history; the dashboard's stale counter is the alarm |
| `expected_session_estimated: true` forever | `calendar` step is failing | `python -m stockml.main calendar` |

## 10. What this runbook deliberately does not do

- It does not backdate `as_of` values. If an artifact is old, the dashboard says
  so; the fix is a refresh, not a cosmetic timestamp.
- It does not interpolate missing prices, articles or deals. A filled gap in the
  data and a *fabricated* gap look identical downstream, so only real rows are
  written.
- It does not run training implicitly. Retraining moves the holdout and changes
  every reported metric, so it stays behind `-Retrain`.
