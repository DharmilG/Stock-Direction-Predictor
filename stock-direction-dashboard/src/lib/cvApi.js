/**
 * CV client. This is the only dashboard module allowed to POST: the rest of
 * the app stays read-only via lib/api.js.
 */

import { API_BASE, ApiError } from './api';

function withAbort(signal, timeout) {
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
  return {
    signal: controller.signal,
    cleanup: () => {
      clearTimeout(timer);
      if (signal) signal.removeEventListener('abort', forward);
    },
    timedOut: () => timedOut,
  };
}

async function readError(response, url) {
  let detail = `HTTP ${response.status}`;
  try {
    const body = await response.json();
    if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(detail, { status: response.status, url });
}

async function getJson(path, { signal, timeout = 10000 } = {}) {
  const url = `${API_BASE}${path}`;
  const abort = withAbort(signal, timeout);
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: abort.signal,
      credentials: 'omit',
      cache: 'no-store',
    });
    if (!response.ok) throw await readError(response, url);
    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (abort.timedOut()) throw new ApiError('CV health check timed out', { url });
    if (error?.name === 'AbortError') throw error;
    throw new ApiError('Cannot reach the CV API. Is the FastAPI service running?', { url, cause: error });
  } finally {
    abort.cleanup();
  }
}

async function postForm(path, file, { signal, timeout = 60000 } = {}) {
  const url = `${API_BASE}${path}`;
  const abort = withAbort(signal, timeout);
  const body = new FormData();
  body.append('file', file);
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { Accept: 'application/json' },
      signal: abort.signal,
      credentials: 'omit',
      body,
    });
    if (!response.ok) throw await readError(response, url);
    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (abort.timedOut()) throw new ApiError('CV analysis timed out', { url });
    if (error?.name === 'AbortError') throw error;
    throw new ApiError('Cannot reach the CV API. Is the FastAPI service running?', { url, cause: error });
  } finally {
    abort.cleanup();
  }
}

export const cvApi = {
  health: (options) => getJson('/cv/health', options),
  training: (options) => getJson('/cv/training', options),
  evaluation: (options) => getJson('/cv/evaluation', options),
  analyze: (file, options) => postForm('/cv/analyze', file, options),
};

export default cvApi;
