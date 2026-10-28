import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Single-flight GET with optional polling.
 *
 * Overlapping requests are impossible: while one call is in flight, extra ticks
 * return the same promise. Polling pauses while the tab is hidden and refreshes
 * immediately when it becomes visible again.
 */
export function useResource(loader, { key = '', enabled = true, pollMs = 0 } = {}) {
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  const [state, setState] = useState({
    data: null,
    envelope: null,
    error: null,
    loading: Boolean(enabled),
    updatedAt: null,
  });
  const inFlight = useRef(null);
  const abortRef = useRef(null);
  const alive = useRef(true);

  const run = useCallback(async ({ quiet = false } = {}) => {
    if (inFlight.current) return inFlight.current;
    const controller = new AbortController();
    abortRef.current = controller;

    const promise = (async () => {
      if (!quiet) {
        setState((previous) => ({ ...previous, loading: previous.data == null, error: null }));
      }
      try {
        const envelope = await loaderRef.current({ signal: controller.signal });
        if (!alive.current) return;
        setState({
          data: envelope?.data ?? null,
          envelope: envelope ?? null,
          error: null,
          loading: false,
          updatedAt: Date.now(),
        });
      } catch (error) {
        if (!alive.current) return;
        if (error?.name === 'AbortError') return;
        setState((previous) => ({ ...previous, error, loading: false }));
      } finally {
        inFlight.current = null;
      }
    })();

    inFlight.current = promise;
    return promise;
  }, []);

  useEffect(() => {
    alive.current = true;
    if (!enabled) {
      setState((previous) => ({ ...previous, loading: false }));
      return () => {};
    }

    run();

    const cleanups = [];
    if (pollMs > 0) {
      const timer = setInterval(() => {
        if (!document.hidden) run({ quiet: true });
      }, pollMs);
      const onVisible = () => {
        if (!document.hidden) run({ quiet: true });
      };
      document.addEventListener('visibilitychange', onVisible);
      cleanups.push(() => clearInterval(timer), () => document.removeEventListener('visibilitychange', onVisible));
    }

    return () => {
      alive.current = false;
      abortRef.current?.abort();
      cleanups.forEach((fn) => fn());
    };
  }, [key, enabled, pollMs, run]);

  const refresh = useCallback(() => run({ quiet: true }), [run]);

  return { ...state, refresh };
}
