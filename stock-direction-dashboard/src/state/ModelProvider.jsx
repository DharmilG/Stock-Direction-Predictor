import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { api } from '../lib/api';
import { useResource } from '../hooks/useResource';

const ModelContext = createContext(null);

const STORAGE_KEY = 'sdi.model';
const FALLBACK_ID = 'v1';

function readStored() {
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    // Private browsing and some embedded webviews throw on localStorage.
    return null;
  }
}

function writeStored(id) {
  try {
    window.localStorage.setItem(STORAGE_KEY, id);
  } catch {
    /* the choice simply does not survive a reload */
  }
}

/**
 * Which model the pages describe.
 *
 * The selection is a single id that every model-scoped page reads; that is why
 * this lives above the router rather than inside a page. A model that has not
 * been built yet is offered but disabled, and a stored id whose artefacts have
 * since disappeared falls back to the default instead of rendering an empty
 * dashboard with no explanation.
 */
export function ModelProvider({ children }) {
  const registry = useResource(api.models, { key: 'models' });
  // Memoised because `?? []` allocates a fresh array every render, which would
  // re-run the reconciliation effect below on every render, not on every poll.
  const models = useMemo(() => registry.data?.models ?? [], [registry.data]);
  const stagesCommand = registry.data?.stages_command ?? 'python -m stockml.v2.stages all';

  const available = useMemo(() => models.filter((model) => model.ready), [models]);
  const byId = useMemo(() => Object.fromEntries(models.map((model) => [model.id, model])), [models]);

  const [modelId, setModelId] = useState(FALLBACK_ID);

  // Reconcile the stored choice against the registry once it lands. Run on every
  // registry change, not just mount, because a v2 rebuild can make a previously
  // unavailable model selectable without a reload.
  useEffect(() => {
    if (!models.length) return;
    const stored = readStored();
    const candidate = stored && byId[stored]?.ready ? stored : null;
    const fallback = available[0]?.id ?? FALLBACK_ID;
    const next = candidate ?? fallback;
    setModelId((current) => (current === next ? current : next));
  }, [models, byId, available]);

  const setModel = useCallback(
    (next) => {
      const id = typeof next === 'function' ? next(modelId) : next;
      if (!byId[id]?.ready) return;
      writeStored(id);
      setModelId(id);
    },
    [modelId, byId],
  );

  const spec = byId[modelId] ?? null;

  const value = useMemo(
    () => ({
      modelId,
      setModel,
      models,
      available,
      spec,
      ready: Boolean(spec?.ready),
      servable: Boolean(spec?.servable),
      status: spec?.status ?? null,
      label: spec?.label ?? modelId,
      stagesCommand,
      missing: spec?.missing ?? [],
      modelWarnings: registry.envelope?.warnings ?? [],
      registryError: registry.error,
      refreshModels: registry.refresh,
      loading: registry.loading,
    }),
    [modelId, setModel, models, available, spec, stagesCommand, registry.envelope, registry.error, registry.refresh, registry.loading],
  );

  return <ModelContext.Provider value={value}>{children}</ModelContext.Provider>;
}

export function useModel() {
  const context = useContext(ModelContext);
  if (!context) throw new Error('useModel must be used inside <ModelProvider>');
  return context;
}

/**
 * Suffix for a `useResource` key. Two pages fetching `metrics` for different
 * models must not share a cache slot, and `useResource` only re-runs when its
 * key changes.
 */
export function modelKey(modelId, name) {
  return `${name}:${modelId}`;
}
