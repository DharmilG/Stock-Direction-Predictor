import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { api, setDemoFallback } from '../lib/api';
import { POLL_INTERVALS } from '../lib/constants';
import { useResource } from '../hooks/useResource';

const DashboardContext = createContext(null);

/**
 * Owns the slow-moving signals every page needs: API reachability, the symbol
 * universe, data freshness, and Q0 telemetry. Pages fetch their own artifacts so
 * a slow endpoint never blocks the shell.
 */
export function DashboardProvider({ children }) {
  const [demoMode, setDemoModeState] = useState(false);

  const setDemoMode = useCallback((next) => {
    setDemoModeState((previous) => (typeof next === 'function' ? next(previous) : next));
  }, []);

  const health = useResource(api.health, { key: 'health', pollMs: POLL_INTERVALS.artifacts });
  const symbols = useResource(api.symbols, { key: 'symbols' });
  const dataHealth = useResource(api.dataHealth, { key: 'data-health', pollMs: POLL_INTERVALS.artifacts });
  const q0 = useResource(api.q0Status, { key: 'q0-status', pollMs: POLL_INTERVALS.q0 });

  // The API client keeps the demo flag in module scope so plain getJson calls can
  // honour it without threading a context through every page. Toggling it only
  // matters when the API is unreachable, so the refetch is guarded by a ref.
  const previousDemo = useRef(demoMode);
  const refreshAllRef = useRef(null);
  useEffect(() => {
    setDemoFallback(demoMode);
    if (previousDemo.current !== demoMode) {
      previousDemo.current = demoMode;
      refreshAllRef.current?.();
    }
  }, [demoMode]);

  const connection = useMemo(() => {
    if (health.error) return 'disconnected';
    if (!health.data && health.loading) return 'connecting';
    const healthPayload = health.data;
    if (!healthPayload) return 'connecting';
    const stale = (dataHealth.data?.summary?.stale_steps?.length ?? 0) > 0;
    const gap = dataHealth.data?.gap?.gap_sessions_estimate ?? 0;
    return stale || gap > 0 ? 'stale' : 'live';
  }, [health.error, health.data, health.loading, dataHealth.data]);

  const refreshAll = useCallback(() => {
    health.refresh();
    dataHealth.refresh();
    q0.refresh();
    symbols.refresh();
  }, [health, dataHealth, q0, symbols]);

  refreshAllRef.current = refreshAll;

  const value = useMemo(
    () => ({
      connection,
      demoMode,
      setDemoMode,
      health: health.data,
      healthError: health.error,
      healthUpdatedAt: health.updatedAt,
      symbols: symbols.data?.symbols ?? [],
      universeTier: symbols.data?.universe_tier ?? null,
      dataHealth: dataHealth.data,
      dataHealthEnvelope: dataHealth.envelope,
      dataHealthError: dataHealth.error,
      q0: q0.data,
      q0Envelope: q0.envelope,
      q0Error: q0.error,
      q0UpdatedAt: q0.updatedAt,
      refreshAll,
      refreshQ0: q0.refresh,
      refreshDataHealth: dataHealth.refresh,
    }),
    [connection, demoMode, setDemoMode, health, symbols, dataHealth, q0, refreshAll],
  );

  return <DashboardContext.Provider value={value}>{children}</DashboardContext.Provider>;
}

export function useDashboard() {
  const context = useContext(DashboardContext);
  if (!context) throw new Error('useDashboard must be used inside <DashboardProvider>');
  return context;
}
