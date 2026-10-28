export const ROUTES = ['overview', 'predictions', 'stock', 'evaluation', 'backtesting', 'q0', 'data'];

export const NAV_SECTIONS = [
  {
    label: 'Research',
    items: [
      { id: 'overview', label: 'Overview', hint: 'Platform state at a glance' },
      { id: 'predictions', label: 'Predictions', hint: 'Latest serving rows' },
      { id: 'stock', label: 'Stock analysis', hint: 'Per-symbol price and features' },
    ],
  },
  {
    label: 'Model',
    items: [
      { id: 'evaluation', label: 'Model evaluation', hint: 'Test metrics and calibration' },
      { id: 'backtesting', label: 'Backtesting', hint: 'Net event backtest' },
    ],
  },
  {
    label: 'LLM & data',
    items: [
      { id: 'q0', label: 'Q0 / LLM research', hint: 'Zero-shot gate telemetry' },
      { id: 'data', label: 'Data & integration', hint: 'Freshness, gaps, endpoints' },
    ],
  },
];

/** Grayscale encodings: intensity plus a directional icon, never color. */
export const SIGNAL_META = {
  UP: { label: 'UP', tone: 'badge-solid', description: 'Predicted up move beyond the flat band' },
  DOWN: { label: 'DOWN', tone: 'badge-mid', description: 'Predicted down move beyond the flat band' },
  FLAT: { label: 'FLAT', tone: 'badge-dashed', description: 'Inside the flat band' },
};

export const POLL_INTERVALS = {
  q0: 5000,
  artifacts: 60000,
};

export const FRESHNESS_TONE = {
  fresh: 'badge-ghost',
  stale: 'badge-alert',
  missing: 'badge-alert',
  partial: 'badge-dashed',
};

export const STATUS_LABEL = {
  live: 'LIVE',
  stale: 'STALE',
  disconnected: 'DISCONNECTED',
  snapshot: 'SNAPSHOT',
  connecting: 'CONNECTING',
};

export const RUNBOOK = [
  {
    title: 'Refresh data and predictions',
    command: 'powershell -NoProfile -ExecutionPolicy Bypass -File .\\refresh_data.ps1',
    detail: 'Fetches prices, delivery data, announcements and news, rebuilds features, then regenerates predictions. Never retrains.',
  },
  {
    title: 'Refresh and retrain',
    command: 'powershell -NoProfile -ExecutionPolicy Bypass -File .\\refresh_data.ps1 -Retrain',
    detail: 'Same chain, then train → evaluate → backtest → predict. Moves the holdout end date forward.',
  },
  {
    title: 'Repair internal price holes',
    command: '.venv\\Scripts\\python.exe -m stockml.main heal',
    detail: 'Re-downloads a short window for symbols missing sessions inside their own coverage.',
  },
  {
    title: 'Serve the API',
    command: '.venv\\Scripts\\python.exe -m uvicorn stockml.api.server:app --host 127.0.0.1 --port 8000',
    detail: 'Read-only /dashboard/* routes plus the legacy /predictions route.',
  },
  {
    title: 'Run the dashboard',
    command: 'npm run dev',
    detail: 'From stock-direction-dashboard/. Proxies /dashboard and /health to port 8000.',
  },
];
