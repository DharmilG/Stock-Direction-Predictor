# Integration guide — Stock Direction Intelligence dashboard

The dashboard reads the research repository's *artifacts*. It never starts a
fetch, an inference run or a training job. This guide documents the contract
between `stock-direction-dashboard/` and `src/stockml/api/`, how to run both
halves locally, and what each status label means.

## 1. Architecture

```text
Browser (React 19 + Vite 7 + Tailwind 4)
   │  GET only, JSON envelope
   ▼
Vite dev server (5173) ──proxy /dashboard, /health──► FastAPI (127.0.0.1:8000)
   │                                                    │
   │                                                    ├── src/stockml/api/dashboard.py      read-only artifact router
   │                                                    ├── src/stockml/api/dashboard_q0.py   Q0 telemetry + ledger
   │                                                    └── src/stockml/api/dashboard_support.py  artifact IO, freshness
   │                                                    ▼
   │                                              results/ · artifacts/ · data/ · logs/ · checkpoints/
   └── src/lib/demo.js (bundled synthetic snapshot, only when the API is unreachable)
```

There is no server-side rendering, no database and no write path. `src/lib/api.js`
exposes `get`-style helpers only; the legacy `GET /predictions` route (which calls
`run_prediction()`) is deliberately *not* referenced anywhere in the frontend, so
opening a page can never regenerate `results/latest_predictions.csv`.

## 2. Run it locally

Backend, from the repository root:

```powershell
.venv\Scripts\python.exe -m uvicorn stockml.api.server:app --host 127.0.0.1 --port 8000
```

Frontend, from `stock-direction-dashboard/`:

```powershell
npm install
Copy-Item .env.example .env
npm run dev
```

Open `http://localhost:5173`. With an empty `.env`, requests go to the Vite proxy,
which forwards `/dashboard/*` and `/health` to `DASHBOARD_API_URL`
(default `http://127.0.0.1:8000`). This is the recommended setup: no CORS
preflight, no origin configuration, and the same relative URLs in production.

| Variable | Where it is read | Purpose |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `src/lib/api.js` (build time) | Absolute API origin. Leave blank to use the proxy. |
| `DASHBOARD_API_URL` | `vite.config.js` (server side) | Proxy target for `npm run dev` / `npm run preview`. |
| `DASHBOARD_ALLOWED_ORIGINS` | `src/stockml/api/server.py` | Comma-separated CORS allowlist; defaults to 5173/4173 on localhost. |
| `DASHBOARD_STATIC_DIR` | `src/stockml/api/server.py` | Serve `dist/` from FastAPI instead of a separate static host. |

## 3. Endpoint reference

Every route below is a `GET` and answers with the same envelope.

| Route | Query | Artifact(s) |
| --- | --- | --- |
| `/dashboard/health` | – | aggregate: model, features, predictions, refresh state |
| `/dashboard/symbols` | – | `config/config.yaml` universe |
| `/dashboard/overview` | – | several, aggregated for the landing page |
| `/dashboard/predictions` | `limit` 1–500 (100) | `results/latest_predictions.csv` |
| `/dashboard/metrics` | – | `results/metrics.json` |
| `/dashboard/model-metadata` | – | `artifacts/model_metadata.json`, `best_params.json`, checkpoint |
| `/dashboard/shap` | `limit` 1–200 (25) | `results/shap_feature_importance.csv` |
| `/dashboard/backtest` | – | `results/backtest_metrics.json`, `backtest_timeseries.parquet` |
| `/dashboard/backtest/trades` | `symbol`, `outcome` (`all\|winners\|losers`), `limit` 1–1000 (100) | `results/backtest_timeseries.parquet` |
| `/dashboard/monitoring` | – | `results/monitoring.json` |
| `/dashboard/stock/{symbol}` | `points` 20–1000 (180) | `data/raw/<symbol>.parquet` |
| `/dashboard/stock/{symbol}/predictions` | `points` 10–1000 (120) | `results/predictions.parquet` |
| `/dashboard/data-health` | – | calendar, raw/features/processed spans, `artifacts/state/refresh.json` |
| `/dashboard/activity` | `limit` 1–50 (12) | state files under `artifacts/state/` |
| `/dashboard/q0/status` | – | heartbeat, supervisor, `gpu_5s.csv`, setup, phases |
| `/dashboard/q0/ledger-summary` | – | Q0 rating ledger (404 when absent) |

`/health`, `/predictions` and `/predictions/{symbol}` are legacy routes kept for
backwards compatibility. The first is read-only and used by the shell; the other
two run inference and are never called by the dashboard.

### Envelope

```json
{
  "data": { "...": "route payload, or null" },
  "as_of": "2026-10-28",
  "generated_at": "2026-10-28T18:04:11.204311+05:30",
  "source": { "kind": "artifact", "artifacts": ["results/metrics.json"] },
  "warnings": ["results/metrics.json is missing"]
}
```

- `source.kind` is `artifact` (read from disk), `runtime` (derived from a live
  probe such as a heartbeat) or `demo` (bundled snapshot).
- A missing or malformed artifact yields `data: null` plus a warning and HTTP
  200 — the UI renders an empty state instead of an error page. Only the Q0
  ledger returns 404, because that file is the product of the run itself.
- `as_of` is the *data* date, `generated_at` is the moment of the read. They are
  different questions and the UI shows both.

## 4. Status semantics

| Label | Meaning |
| --- | --- |
| **LIVE** | A heartbeat/probe file was written within its staleness budget. |
| **SNAPSHOT** | Artifacts exist but no live process is reporting. |
| **STALE** | Artifacts exist and contradict the expected session, or a heartbeat exceeded its budget. |
| **DISCONNECTED** | The API did not answer; the UI is showing an error or the demo snapshot. |

Freshness is computed from **data dates**, never file mtimes: `expected_session`
is the last completed NSE session, and a session counts as complete after 18:30
IST. When the stored trading calendar itself is behind, the estimate falls back
to weekdays and the response is flagged `expected_session_estimated: true`.

Two date-only rules matter:

1. NSE sessions are calendar dates. They are formatted with a regex split of the
   `YYYY-MM-DD` string, never `new Date(...)`, so they cannot shift a day across
   timezones. Instants (heartbeats, run timestamps) are rendered with
   `Intl.DateTimeFormat` pinned to `Asia/Kolkata`.
2. Processed datasets stop `triple_barrier_horizon` (5) sessions before the
   feature panel **by design**. `data-health` compares each step against its own
   expectation, so the label trail is reported as fresh, not as a gap.

## 5. Polling

`src/hooks/useResource.js` is single-flight: a tick that fires while a request
is in flight is skipped, in-flight requests are aborted on unmount, and polling
pauses while `document.hidden`.

| Resource | Interval |
| --- | --- |
| `/dashboard/q0/status` | 5 s |
| `/dashboard/health`, `/dashboard/data-health` | 60 s |
| `/dashboard/symbols` | once per session |
| everything else | on mount, on filter change, or on **Refresh** |

**Refresh** re-reads artifacts; it does not run the pipeline. To move data, use
`refresh_data.ps1` (see §7).

## 6. Demo fallback

`src/lib/demo.js` builds a deterministic synthetic snapshot (180 sessions, 6
symbols) shaped exactly like the live payloads. It is used only when a request
fails to reach the API *and* the user has enabled demo mode from the sidebar. It
never overrides a reachable API, and every payload it produces carries
`source.kind: "demo"` plus a warning, which the shell renders as a banner. Treat
it as a UI preview, not as evidence about the model.

## 7. Refreshing data

The dashboard "Refresh" button re-reads files. Data itself is updated by the
pipeline:

```powershell
.\refresh_data.ps1                 # data + predictions, model untouched
.\refresh_data.ps1 -Retrain        # + train → evaluate → backtest (moves the holdout)
.\refresh_data.ps1 -DryRun         # print the plan, change nothing
```

The full freshness model, the backfill matrix and the safety rules are in
[`DOCS/DATA_FRESHNESS_AND_BACKFILL.md`](../../DOCS/DATA_FRESHNESS_AND_BACKFILL.md).
Progress lands in `artifacts/state/refresh.json` and is rendered live by
**Data & integration → Refresh run**.

## 8. Production build

```powershell
npm run build          # writes dist/
npm run preview        # serves dist/ with the same proxy, for a smoke test
```

Two deployment shapes:

1. **Separate static host + API.** Set `VITE_API_BASE_URL` at build time and add
   the dashboard origin to `DASHBOARD_ALLOWED_ORIGINS`.
2. **Single origin.** Serve `dist/` from FastAPI:

```powershell
$env:DASHBOARD_STATIC_DIR = "e:\books\Adaniports\stock_direction_platform\stock-direction-dashboard\dist"
.venv\Scripts\python.exe -m uvicorn stockml.api.server:app --host 127.0.0.1 --port 8000
```

`/` then returns `index.html` and `/assets/*` the bundle; API routes win because
the mount is registered last.

The read-only guarantee is structural: there is no POST/PUT/DELETE route and the
client cannot ask for one. Keep it that way when extending the router — a
training-control endpoint on this server would be a remote code execution
surface, since anyone who can reach the port could start a GPU job.

## 9. Tests

```powershell
.venv\Scripts\python.exe -m pytest tests\test_dashboard_api.py tests\test_refresh.py -q
cd stock-direction-dashboard; npm run lint; npm run build
```

`tests/test_dashboard_api.py` asserts the envelope on every route, that
`results/latest_predictions.csv` is never rewritten by a read, that unknown
symbols and `..` traversal are rejected, that missing/malformed artifacts
degrade to warnings, and that `data-health` reports gap, holes and the label
horizon correctly. `tests/test_refresh.py` pins the refresh order: no retraining
by default, optional sources cannot abort a run, dry-run executes nothing.

## 10. Troubleshooting

| Symptom | Check |
| --- | --- |
| `DISCONNECTED`, every panel empty | Is uvicorn running on 8000? `curl http://127.0.0.1:8000/dashboard/health` |
| Panels show `source.kind = demo` | API unreachable **and** demo mode on. Turn it off to see the real error. |
| `data: null` with a warning | The artifact genuinely does not exist yet; run the pipeline stage named in `source.artifacts`. |
| Gap grows every day | `refresh_data.ps1` was not run; see the backfill matrix before assuming it can be repaired. |
| Q0 page says STALE | `logs/q0_heartbeat.json` is older than its budget; the supervisor is not running. |
| CORS error in the console | Either use the proxy (blank `VITE_API_BASE_URL`) or add the origin to `DASHBOARD_ALLOWED_ORIGINS`. |
| Dates off by one day | A component is calling `new Date()` on a session date; use `src/lib/dates.js`. |
