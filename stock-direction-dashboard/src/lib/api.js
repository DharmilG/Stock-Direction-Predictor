/**
 * Read-only API client.
 *
 * Only GET is exposed: the dashboard must never trigger inference, training or a
 * data fetch. The legacy mutating `GET /predictions` endpoint is deliberately
 * absent from this module.
 */

import { demoEnvelope } from './demo';

const RAW_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');
export const API_BASE = RAW_BASE;
export const DEFAULT_TIMEOUT_MS = 20000;

/**
 * Demo fallback. When enabled and the API is unreachable, the bundled snapshot
 * answers instead — always stamped `source.kind = "demo"` so the shell can label
 * it. Live responses always win; this never masks a reachable API.
 */
let demoFallback = false;

export function setDemoFallback(enabled) {
  demoFallback = Boolean(enabled);
}

export class ApiError extends Error {
  constructor(message, { status = 0, url = '', cause } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.url = url;
    this.cause = cause;
  }
}

function buildUrl(path, params) {
  const url = `${RAW_BASE}${path}`;
  if (!params) return url;
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') query.set(key, String(value));
  });
  const suffix = query.toString();
  return suffix ? `${url}?${suffix}` : url;
}

async function getJson(path, { params, signal, timeout = DEFAULT_TIMEOUT_MS } = {}) {
  const url = buildUrl(path, params);
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeout);

  const forward = () => controller.abort();
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener('abort', forward, { once: true });
  }

  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
      mode: 'cors',
      credentials: 'omit',
      cache: 'no-store',
    });

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
      } catch {
        /* non-JSON error body: keep the status text */
      }
      throw new ApiError(detail, { status: response.status, url });
    }

    const payload = await response.json();
    // Every /dashboard/* route answers with the same envelope.
    if (payload && typeof payload === 'object' && 'data' in payload) return payload;
    return { data: payload, as_of: null, generated_at: null, source: null, warnings: [] };
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (timedOut) throw new ApiError(`Request timed out after ${timeout}ms`, { url });
    if (error?.name === 'AbortError') throw error;
    if (demoFallback) {
      const snapshot = demoEnvelope(path, url.includes('?') ? url.slice(url.indexOf('?')) : '');
      if (snapshot) return snapshot;
    }
    throw new ApiError(
      RAW_BASE
        ? `Cannot reach ${RAW_BASE}. Is the API running?`
        : 'Cannot reach the API. Start it with: uvicorn stockml.api.server:app --port 8000',
      { url, cause: error },
    );
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener('abort', forward);
  }
}

/**
 * Model scoping.
 *
 * Every model-owned endpoint takes the model id as its first argument and
 * forwards it as `?model=`. The registry answers which ids exist, so the shell
 * never hardcodes them. A page that forgets the id is silently showing v1 while
 * the header says v2 — the single most likely bug in this design — which is why
 * the id is a required positional argument rather than part of the options bag.
 */
function scoped(model, options) {
  return { ...options, params: { ...(options?.params ?? {}), model } };
}

export const api = {
  health: (options) => getJson('/dashboard/health', options),
  symbols: (options) => getJson('/dashboard/symbols', options),
  models: (options) => getJson('/dashboard/models', options),
  overview: (options) => getJson('/dashboard/overview', options),
  predictions: (model, options) => getJson('/dashboard/predictions', scoped(model, { ...options, params: { limit: 500 } })),
  metrics: (model, options) => getJson('/dashboard/metrics', scoped(model, options)),
  modelMetadata: (model, options) => getJson('/dashboard/model-metadata', scoped(model, options)),
  shap: (model, limit = 25, options) =>
    getJson('/dashboard/shap', scoped(model, { ...options, params: { limit } })),
  backtest: (model, options) => getJson('/dashboard/backtest', scoped(model, options)),
  backtestTrades: (model, { symbol, outcome = 'all', limit = 100 } = {}, options) =>
    getJson('/dashboard/backtest/trades', scoped(model, { ...options, params: { symbol, outcome, limit } })),
  monitoring: (model, options) => getJson('/dashboard/monitoring', scoped(model, options)),
  dataHealth: (options) => getJson('/dashboard/data-health', options),
  activity: (options) => getJson('/dashboard/activity', options),
  stock: (model, symbol, points = 180, options) =>
    getJson(`/dashboard/stock/${encodeURIComponent(symbol)}`, scoped(model, { ...options, params: { points } })),
  stockPredictions: (model, symbol, points = 120, options) =>
    getJson(`/dashboard/stock/${encodeURIComponent(symbol)}/predictions`, scoped(model, { ...options, params: { points } })),
  q0Status: (options) => getJson('/dashboard/q0/status', options),
  q0Ledger: (options) => getJson('/dashboard/q0/ledger-summary', options),
  modelReport: (model, options) => getJson('/dashboard/model/report', scoped(model, options)),
  modelLive: (model, options) => getJson('/dashboard/model/live', scoped(model, options)),
  modelCases: (model, { split = 'test', predicted = 'all', actual = 'all', symbol = '', correct = 'all', limit = 50, offset = 0 } = {}, options) =>
    getJson('/dashboard/model/cases', scoped(model, { ...options, params: { split, predicted, actual, symbol, correct, limit, offset } })),
  explain: (model, symbol, { date, split } = {}, options) =>
    getJson(`/dashboard/explain/${encodeURIComponent(symbol)}`, scoped(model, { ...options, params: { date, split } })),
  q0Timeline: ({ window = '2h', step = 5 } = {}, options) => getJson('/dashboard/q0/timeline', { ...options, params: { window, step } }),
  q0Analytics: (options) => getJson('/dashboard/q0/analytics', options),
  q0Cases: ({ limit = 50, offset = 0, symbol = '', vote = 'all', outcome = 'all' } = {}, options) =>
    getJson('/dashboard/q0/cases', { ...options, params: { limit, offset, symbol, vote, outcome } }),
  q0Case: (caseId, options) => getJson(`/dashboard/q0/case/${encodeURIComponent(caseId)}`, options),

  // Paired comparison. `a` is the reference (default v1), `b` the challenger.
  compareSummary: ({ a = 'v1', b = 'v2' } = {}, options) =>
    getJson('/dashboard/compare/summary', { ...options, params: { a, b } }),
  compareSegments: ({ a = 'v1', b = 'v2', by = 'class' } = {}, options) =>
    getJson('/dashboard/compare/segments', { ...options, params: { a, b, by } }),
  comparePairing: ({ a = 'v1', b = 'v2' } = {}, options) =>
    getJson('/dashboard/compare/pairing', { ...options, params: { a, b } }),
  compareAttribution: ({ a = 'v1', b = 'v2' } = {}, options) =>
    getJson('/dashboard/compare/attribution', { ...options, params: { a, b } }),

  // The v2 report endpoints stay served for one release; the comparison page is
  // their replacement and reads the model-scoped routes above.
  v2Summary: (options) => getJson('/dashboard/v2/summary', options),
  v2Experiments: (options) => getJson('/dashboard/v2/experiments', options),
  v2Diagnostics: (options) => getJson('/dashboard/v2/diagnostics', options),
  v2Reasons: (options) => getJson('/dashboard/v2/reasons', options),
  v2Cases: ({ predicted = 'all', actual = 'all', correct = 'all', symbol = '', limit = 50, offset = 0 } = {}, options) =>
    getJson('/dashboard/v2/cases', { ...options, params: { predicted, actual, correct, symbol, limit, offset } }),
};
