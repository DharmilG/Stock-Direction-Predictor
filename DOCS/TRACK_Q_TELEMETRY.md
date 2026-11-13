# Track Q telemetry — what the Q0 page reads, and what it refuses to invent

**Purpose:** the Q0 page answers "what is the zero-shot rater actually doing, and
is the GPU healthy enough to keep it doing it?" from files the run itself writes.
The rule throughout: **a field that was not recorded is absent, never zero.** No
chart in this document synthesises a series it did not read.

The page keeps the v1 Status tab and adds **Live · Analytics · Cases**.

---

## 1. Where it lives

| Layer | File |
| --- | --- |
| Telemetry writers | `src/stockml/data/q0_d1.py`, `src/stockml/monitoring/monitor.py` |
| Analysis (FastAPI-free) | `src/stockml/api/q0_analytics.py` |
| API router | `src/stockml/api/dashboard_q0.py` (`/dashboard/q0/*`) |
| Page | `stock-direction-dashboard/src/pages/Q0Research.jsx` + `Q0Tabs.jsx` (`#/q0`) |

---

## 2. Telemetry files

| File | Written by | Shape |
| --- | --- | --- |
| `logs/gpu_5s.csv` | thermal guard, every 5 s | `timestamp,case_id,temp_c` — **unchanged**, so existing readers and old runs stay valid |
| `logs/q0_resource_5s.csv` | `ResourceSampler`, every 5 s | superset: `timestamp,case_id,temp_c,util_pct,mem_used_mb,mem_total_mb,power_w,fan_pct,clock_sm_mhz,clock_max_sm_mhz`. An empty cell means the card does not report that field |
| `logs/q0_history.jsonl` | `_append_history`, throttled 30 s | one whole heartbeat per row, so co-movements survive (`timestamp, phase, pool stats, thermal counters, …`) |
| `logs/q0_heartbeat.json` | `_write_heartbeat` (atomic) | the *current* snapshot only: `updated_at, pid, phase, current_case, …` |
| `artifacts/state/psi_history.jsonl` | `monitor`, one row per run | `latest_data, generated_at, rows, missing_rate, threshold, alerts, feature_count, top[{feature, psi}]` (top 15 only) |
| `data/llm/qwen_rating_ledger.parquet` | the rater | `case_id, symbol, historical_label, qwen_vote, qwen_conf, finbert_vote, finbert_score, vote_matches_label, finbert_matches_label, harness_fail, kept_for_training, no_news, …` — append-only, so row order *is* time order |
| `results/traces/*.json` | per-case harness | prompt, thinking, content, timings; the only source of case duration |

### `gpu_sample()`

One batched `nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,
memory.total,power.draw,fan.speed,clocks.sm,clocks.max.sm` call replaces the
scattered single-field invocations (each costs ~200 ms on Windows and the guard
runs every 5 s). The contract matters:

- `gpu_sample() is None` → the **query failed** (no `nvidia-smi`, driver error).
  No series at all.
- a `None` **value** inside the dict → the card answered but does not expose the
  field (e.g. `fan.speed` on a passively cooled or laptop part). That is a gap in
  the chart, never a zero.
- `gpu_temp_c()` keeps its exact previous semantics and is still used by the
  thermal gate. The instrumentation is additive.

`psi_history.jsonl` stores only the 15 leading features: the full matrix is ~190
columns per run and the dashboard never plots more than the leaders.

---

## 3. `GET /dashboard/q0/timeline?window=&step=`

`window` ∈ `30m | 2h | all` (default `2h`), `step` is the sampling stride in
seconds (5–300, default 5).

| Payload | Contents |
| --- | --- |
| `resource` | `available, fields[], series[{at, case_id, <field>: value\|null}], source, stride_seconds, timezone, size_bytes, supplementary` |
| `history` | `available, rows` (≤ 400, throttled heartbeats) |
| `progress` | `available, rows` (≤ 300), `window` = 50 cases |
| `psi` | `available, rows` (last 60 monitoring runs) |
| `live` | `running, phase, current_case, heartbeat_age_seconds, limit_seconds` (600 s), `updated_at` |

Source selection is explicit, and so is its consequence:

1. `logs/q0_resource_5s.csv` exists → every recorded field is offered.
2. only `logs/gpu_5s.csv` exists → `fields: ["temp_c"]` **plus a warning** that
   utilisation, power, fan and clocks are absent, not zero. No synthesised series.
3. neither → `data: null` with a warning.

`fields` is derived from columns that carry at least one numeric value, so an
all-blank column disappears from both `fields` and the row dicts. The UI then
renders a `Missing` card ("Power and fan were not recorded") rather than an empty
axis — and where a field is partly present it degrades to the fields that exist
(power without fan) with a note.

Size control: `MAX_SERIES_POINTS = 2000`. A long "all" window is strided down
(and `history`/`progress` thinned) so one poll does not ship a megabyte. Thinning
always keeps the last point.

`progress.rows` is cumulative rated / judged / kept with rolling (50-case) rates,
vote mix, case-duration p50/p95 and an agreement corridor of ±2·SE. The ledger
carries no timestamp of its own, so the x axis is a **case index, not a clock** —
printed as such.

---

## 4. `GET /dashboard/q0/analytics`

Computed from judged rows only (`judged_mask`: a usable vote and no harness
failure). Every block states its denominator.

| Block | Contents |
| --- | --- |
| `funnel` | `attempted → rated (unique case ids) → judged → kept`, each with `share_of_attempted` and step-to-step `retention`; plus `judged_rate`, `harness_fail_rate` |
| `votes_by_label` | Qwen vote × realised label matrix (rows = label, columns = call) |
| `finbert_by_label` | the same for FinBERT |
| `calibration` | confidence bins `0 · 0.4 · 0.5 · 0.6 · 0.7 · 0.8 · 0.9 · 1`, each with `accuracy, confidence, gap, n, share`, plus `ece`, `n`, `mean_confidence` |
| `confidence_hist` | 10 bins of `qwen_conf` |
| `disagreement` | `both_match, qwen_only, finbert_only, neither`, `qwen_matches`, `finbert_matches`, `finbert_vs_qwen` matrix |
| `latency` | `overall` and `by_outcome` (`match / mismatch / harness_fail`) p50/p90/mean seconds from the trace files; `available: false` when no trace was readable |
| `cohorts` | `news` vs `no_news`: n, Qwen match, FinBERT match, mean confidence |
| `per_symbol` | n, Qwen match, mean confidence, mean FinBERT score |

Two deliberate choices:

- **Latency is absent rather than estimated.** The ledger stores no duration, so
  if `results/traces/*.json` cannot be read the block says `available: false` and
  the route adds a warning. It does not fall back to a constant.
- **A harness failure has no outcome to compare.** It is excluded from both
  `match` and `mismatch` (its `vote_matches_label` is null), not counted as a
  miss, so `mismatch` never inflates the failure rate.

The `no_news` cohort is the honest control: a rater that only wins on news rows is
reading the text; one that "wins" on no-news rows is reading the prior.

---

## 5. `GET /dashboard/q0/cases` and `/case/{case_id}`

`cases?limit=1–500&offset=&symbol=&vote=all|UP|DOWN|FLAT&outcome=all|match|mismatch`
returns paginated ledger rows + `total`. Trace metadata is joined in from a cached
scan of `results/traces/` (cache keyed on the newest trace modification time, so it
invalidates itself).

`case/{case_id}` returns the full trace: prompt, thinking, content, vote,
confidence, temperatures, plus `trace_available`. Path safety is checked **before**
the ledger is read: an id containing a separator or `..` is a 404, because a
traversal attempt is not a missing case. A ledger row whose trace file is gone
still renders, with the ledger fields only.

---

## 6. Frontend `#/q0`

Tabs **Status · Live · Analytics · Cases** (Status is the v1 page, unchanged).

**Live** — window 30 m / 2 h / all. Panels: temperature **and** SM clock vs cap
(throttling), utilisation and VRAM, power draw and fan, cumulative rated vs judged,
rolling case duration p50/p95, rolling agreement with the ±2·SE corridor and
harness-failure rate, rolling vote mix, thermal-state band and ETA burn-down, PSI
history. Each `Missing` card names the file that would populate it.

**Analytics** — funnel with retention, vote × label and FinBERT × label matrices,
Qwen calibration with ECE, confidence distribution, disagreement quadrants, latency
by outcome, news vs no-news cohorts, agreement by symbol.

**Cases** — ledger table with vote/outcome/symbol filters → trace drawer (prompt,
thinking, content, vote, confidence, temperatures). A missing trace says so; nothing
is reconstructed.

---

## 7. What this telemetry cannot tell you

- `q0_history.jsonl` starts at the next Q0 run. There is no GPU-history backfill:
  a run that predates the instrumentation has temperature samples only.
- `psi_history.jsonl` is one row per `monitor` run, not per day.
- The trace files record no wall-clock start, so `progress` uses a case index.
- Ledger timestamps are absent by design; `as_of` comes from the artifact's
  modification time.
- Agreement is against the triple-barrier label, which is a **proxy** for a good
  rating, not a ground truth about the news.

---

## 8. Tests

```powershell
.venv\Scripts\python.exe -m pytest tests\test_q0_analytics.py tests\test_qwen_guards.py tests\test_dashboard_api.py -q
```

`tests/test_q0_analytics.py` pins, on synthetic ledgers and CSVs: funnel counts and
retention, the vote × label matrix, ECE on a known over-confident rater, the
disagreement quadrants summing to the judged count, latency absent without traces
and present with them, cohort and per-symbol grouping, `case_rows` filters
(including "a harness failure is not a mismatch"), six path-traversal rejections,
`case_detail` refusing a `trace_path` outside `results/traces`, trace-cache
invalidation, `fields: ["temp_c"]` + warning for the legacy file, seven fields for
the superset file with `fan_pct` absent when blank, stride capping, `_window_seconds`
parsing, `_thin`, and `_progress_rows` never inventing a duration.

`tests/test_qwen_guards.py` uses `monkeypatch` on `gpu_sample`, so the thermal-abort
test does not need a GPU.
