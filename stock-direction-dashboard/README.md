# Stock Direction Intelligence Dashboard

A React 19 + Vite 7 + Tailwind CSS 4 research console for the stock-direction
platform. Monochrome only — direction is encoded by glyph, label and weight, never
by colour — every action is a 5 px bordered button.

The app is a **read-only view over the pipeline's artifacts**. It never fetches
data, runs inference or trains a model; those are pipeline commands (see
[`docs/INTEGRATION_GUIDE.md`](docs/INTEGRATION_GUIDE.md)).

## Pages

| Route | What it shows |
| --- | --- |
| `#/overview` | expected session, feature panel date, test accuracy, backtest return, stale-step table, prediction mix, training progress, dataset coverage, consistency checks |
| `#/predictions` | latest scored universe with per-class probabilities, signal filter, confidence threshold, CSV export |
| `#/stock` | one symbol: close series, volume, selected indicator, scored history vs realised label, latest prediction |
| `#/evaluation` | accuracy vs majority/argmax baselines, balanced accuracy, MCC, ROC AUC OvR, ECE, confusion matrix, per-class report, calibration curve, SHAP importance, model metadata |
| `#/backtesting` | return, drawdown, Sharpe, trades, cost model, equity curve reconstructed from the saved time series, per-symbol breakdown, trade table |
| `#/q0` | Track Q telemetry: heartbeat, current case, rated vs judged, gate verdict, phases, thermals, GPU trend, supervisor state, ledger distributions |
| `#/data` | data health: gap, holes, per-step freshness with the exact remediation command, coverage, runbook, endpoint reference |

## Requirements

- Node.js 20.19+ or 22.12+
- A reachable `stockml.api.server` for live data (optional — see demo fallback)

## Run locally

```powershell
npm install
Copy-Item .env.example .env
npm run dev
```

Open `http://localhost:5173`. With an empty `VITE_API_BASE_URL`, Vite proxies
`/dashboard` and `/health` to `DASHBOARD_API_URL` (default
`http://127.0.0.1:8000`), so no CORS setup is needed. Start the backend first:

```powershell
..\.venv\Scripts\python.exe -m uvicorn stockml.api.server:app --host 127.0.0.1 --port 8000
```

## Build

```powershell
npm run build     # dist/
npm run preview   # serves dist/ with the same proxy
npm run lint
```

FastAPI can serve the build itself: set `DASHBOARD_STATIC_DIR` to `dist/` before
starting uvicorn.

## Data honesty

- Every response carries `source.kind` (`artifact`, `runtime`, `demo`) and
  `as_of` (data date) next to `generated_at` (read time). They are different
  questions and both are shown.
- Session dates are never parsed with `new Date()`; they are formatted from the
  `YYYY-MM-DD` string so they cannot shift across timezones.
- Missing artifacts render an empty state plus the artifact path, not a
  fabricated number.
- The bundled snapshot (`src/lib/demo.js`) is used **only** when the API is
  unreachable *and* demo mode is on. It is stamped `source.kind = "demo"` and
  flagged by a banner.
- `results/metrics.json` (test accuracy) and `artifacts/model_metadata.json`
  disagree; the Evaluation page reports both instead of reconciling them.
- Track Q counts come from the run's own heartbeat and ledger. "Not
  instrumented" means exactly that.

This is a research visualization, not a trading system or investment advice.
