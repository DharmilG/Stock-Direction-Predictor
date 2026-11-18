// `v2` is kept recognised for one release: it was its own page before the model
// selector existed, and old links are redirected to `compare`.
export const ROUTES = ['overview', 'predictions', 'stock', 'model', 'evaluation', 'backtesting', 'q0', 'data', 'compare', 'v2', 'cv'];

export const LEGACY_ROUTES = { v2: 'compare' };

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
      { id: 'model', label: 'Model behaviour', hint: 'Why it predicts what it predicts' },
      { id: 'evaluation', label: 'Model evaluation', hint: 'Test metrics and calibration' },
      { id: 'backtesting', label: 'Backtesting', hint: 'Net event backtest' },
      { id: 'compare', label: 'Model comparison', hint: 'v1 vs v2, where each wins and why' },
    ],
  },
  {
    label: 'LLM & data',
    items: [
      { id: 'q0', label: 'Q0 / LLM research', hint: 'Zero-shot gate telemetry' },
      { id: 'data', label: 'Data & integration', hint: 'Freshness, gaps, endpoints' },
    ],
  },
  {
    label: 'Computer vision',
    items: [{ id: 'cv', label: 'Chart Vision', hint: 'Upload a chart, read pattern and direction' }],
  },
];

/**
 * Pages whose numbers come from one model's artefacts. The header selector is
 * live on these and inert everywhere else, rather than silently ignored.
 */
export const MODEL_SCOPED_ROUTES = ['predictions', 'stock', 'model', 'evaluation', 'backtesting'];

/** Grayscale encodings: intensity plus a directional icon, never color. */
export const SIGNAL_META = {
  UP: { label: 'UP', tone: 'badge-solid', description: 'Predicted up move beyond the flat band' },
  DOWN: { label: 'DOWN', tone: 'badge-mid', description: 'Predicted down move beyond the flat band' },
  FLAT: { label: 'FLAT', tone: 'badge-dashed', description: 'Inside the flat band' },
};

export const POLL_INTERVALS = {
  cv: 5000,
  q0: 5000,
  artifacts: 60000,
  modelLive: 5000,
};

/**
 * Hypothesis verdicts. A glyph carries the verdict so the state survives a
 * grayscale screen and a printed page; the badge tone only reinforces it.
 */
export const VERDICT_META = {
  SUPPORTED: { glyph: '●', tone: 'badge-solid', label: 'Supported' },
  REJECTED: { glyph: '○', tone: 'badge-ghost', label: 'Rejected' },
  INCONCLUSIVE: { glyph: '◐', tone: 'badge-dashed', label: 'Inconclusive' },
  INFO: { glyph: '◇', tone: 'badge-ghost', label: 'Context' },
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
    detail: 'From stock-direction-dashboard/. Proxies /dashboard, /health and /cv to port 8000.',
  },
  {
    title: 'Generate CV images (resumable)',
    command: 'cd CV && python -m chartsense.cli generate-all --resume',
    detail: 'Writes synthetic and own-data chart images to CV/data/. Safe to re-run; skips completed images.',
  },
  {
    title: 'Train the CV model (resumable)',
    command: 'cd CV && python -m chartsense.cli train-all --resume',
    detail: 'Stage 1 then stage 2. Checkpoints every epoch; re-running resumes from the last one.',
  },
];
