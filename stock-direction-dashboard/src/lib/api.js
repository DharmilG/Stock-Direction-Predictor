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

export const api = {
  health: (options) => getJson('/dashboard/health', options),
  symbols: (options) => getJson('/dashboard/symbols', options),
  overview: (options) => getJson('/dashboard/overview', options),
  predictions: (options) => getJson('/dashboard/predictions', options),
  metrics: (options) => getJson('/dashboard/metrics', options),
  modelMetadata: (options) => getJson('/dashboard/model-metadata', options),
  shap: (limit = 25, options) => getJson('/dashboard/shap', { ...options, params: { limit } }),
  backtest: (options) => getJson('/dashboard/backtest', options),
  backtestTrades: ({ symbol, outcome = 'all', limit = 100 } = {}, options) =>
    getJson('/dashboard/backtest/trades', { ...options, params: { symbol, outcome, limit } }),
  monitoring: (options) => getJson('/dashboard/monitoring', options),
  dataHealth: (options) => getJson('/dashboard/data-health', options),
  activity: (options) => getJson('/dashboard/activity', options),
  stock: (symbol, points = 180, options) =>
    getJson(`/dashboard/stock/${encodeURIComponent(symbol)}`, { ...options, params: { points } }),
  stockPredictions: (symbol, points = 120, options) =>
    getJson(`/dashboard/stock/${encodeURIComponent(symbol)}/predictions`, { ...options, params: { points } }),
  q0Status: (options) => getJson('/dashboard/q0/status', options),
  q0Ledger: (options) => getJson('/dashboard/q0/ledger-summary', options),
};
