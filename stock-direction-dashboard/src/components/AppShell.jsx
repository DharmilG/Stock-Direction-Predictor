import { useEffect, useState } from 'react';
import { Activity, Database, FileJson, Image as ImageIcon, Layers, RefreshCw, Sparkles } from 'lucide-react';
import { NAV_SECTIONS, STATUS_LABEL } from '../lib/constants';
import { humanAge } from '../lib/dates';
import { useDashboard } from '../state/DashboardProvider';
import { useModel } from '../state/ModelProvider';
import { Badge, Button, ErrorNote } from './ui';

const SECTION_ICONS = {
  Research: Activity,
  Model: Layers,
  'LLM & data': Database,
  'Computer vision': ImageIcon,
};

function ConnectionChip({ connection, detail }) {
  const tone =
    connection === 'live' ? 'badge-solid' : connection === 'stale' ? 'badge-alert' : connection === 'connecting' ? 'badge-dashed' : 'badge-alert';
  return (
    <span
      className={`badge ${tone}`}
      title={detail ? `Derived from GET /dashboard/health\n${detail}` : 'Derived from GET /dashboard/health and the freshness of the latest artifacts'}
    >
      <span aria-hidden="true">{connection === 'live' ? '●' : connection === 'stale' ? '◐' : '○'}</span>
      {STATUS_LABEL[connection] ?? connection}
    </span>
  );
}

/**
 * Which model the pages describe, plus the way into the comparison.
 *
 * Everything model-scoped hangs off this one control, so it lives in the shell
 * rather than on a page: the choice survives navigation and a reload. A model
 * whose core artefacts are missing stays visible but disabled — hiding it would
 * make an unbuilt model look like an unplanned one — and its tooltip names the
 * command that would build it.
 */
function ModelSwitcher({ route, onNavigate }) {
  const { modelId, setModel, models, stagesCommand, loading } = useModel();

  if (!models.length) {
    return (
      <span className="mono text-[.875rem] text-[var(--ink-3)]" title={loading ? 'Reading GET /dashboard/models' : 'No model registry available'}>
        {loading ? 'model …' : 'no models'}
      </span>
    );
  }

  const selected = models.find((model) => model.id === modelId);

  return (
    <div className="flex flex-wrap items-center gap-2">
      <div className="segment" role="group" aria-label="Model">
        {models.map((model) => (
          <Button
            key={model.id}
            variant="secondary"
            aria-pressed={route !== 'compare' && model.id === modelId}
            disabled={!model.ready}
            title={
              model.ready
                ? `${model.label} · ${model.root}`
                : `${model.label} is not built yet: missing ${model.missing.join(', ')}. Run \`${stagesCommand}\`.`
            }
            onClick={() => setModel(model.id)}
          >
            {model.label}
          </Button>
        ))}
        <Button
          variant="secondary"
          aria-pressed={route === 'compare'}
          title="Both models side by side: scorecard, confusion matrices, segments, flips and why"
          onClick={() => onNavigate('compare')}
        >
          Compare
        </Button>
      </div>
      {selected && !selected.servable && (
        <Badge tone="badge-dashed" title="Research model. It is measured here next to the incumbent and is never served; nothing on this dashboard promotes it.">
          not served
        </Badge>
      )}
    </div>
  );
}

function Sidebar({ route, onNavigate }) {
  const { connection, universeTier, symbols, health } = useDashboard();
  return (
    <aside className="sidebar">
      <div className="border-b border-[var(--line)] px-4 py-4">
        <div className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-md border border-[var(--ink)] bg-[var(--ink)] text-white">
            <Sparkles size={16} strokeWidth={2.25} />
          </span>
          <p className="truncate text-[.9375rem] font-semibold leading-tight text-[var(--ink)]">Stock Direction</p>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <ConnectionChip connection={connection} />
          <Badge tone="badge-ghost" title="Universe configured in config/config.yaml">
            {symbols.length || '—'} symbols
          </Badge>
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto px-2 py-3 scroll-thin" aria-label="Sections">
        {NAV_SECTIONS.map((section) => {
          const Icon = SECTION_ICONS[section.label] ?? FileJson;
          return (
            <div key={section.label} className="mb-4">
              <p className="eyebrow flex items-center gap-1.5 px-2 py-1">
                <Icon size={12} strokeWidth={2.5} />
                {section.label}
              </p>
              <div className="space-y-0.5">
                {section.items.map((item) => (
                  <a
                    key={item.id}
                    href={`#${item.id}`}
                    className="nav-row"
                    aria-current={route === item.id ? 'page' : undefined}
                    onClick={(event) => {
                      event.preventDefault();
                      onNavigate(item.id);
                    }}
                    title={item.hint}
                  >
                    <span className="truncate">{item.label}</span>
                  </a>
                ))}
              </div>
            </div>
          );
        })}
      </nav>

      <div className="border-t border-[var(--line)] px-4 py-3">
        <p
          className="mono truncate text-[.875rem] text-[var(--ink-3)]"
          title={`Universe tier · API version ${health?.version ?? '—'} · read-only`}
        >
          {universeTier ?? 'no tier recorded'}
        </p>
      </div>
    </aside>
  );
}

function DemoSwitch() {
  const { demoMode, setDemoMode } = useDashboard();
  return (
    <Button
      size="sm"
      variant={demoMode ? 'primary' : 'quiet'}
      aria-pressed={demoMode}
      title="Demo mode labels every panel with the bundled sample payload instead of live artifacts"
      onClick={() => setDemoMode((value) => !value)}
    >
      {demoMode ? 'Demo data' : 'Live data'}
    </Button>
  );
}

export function AppShell({ route, onNavigate, children }) {
  const { connection, healthError, healthUpdatedAt, refreshAll, dataHealth } = useDashboard();
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setBusy(false), 900);
    return () => clearTimeout(timer);
  }, [healthUpdatedAt]);

  const staleSteps = dataHealth?.summary?.stale_steps ?? [];
  const gap = dataHealth?.gap?.gap_sessions_estimate ?? 0;
  const holes = dataHealth?.holes?.total_holes ?? 0;

  const staleDetail = [staleSteps.length ? `${staleSteps.length} stale step(s)` : '', gap ? `gap ${gap} session(s)` : '', holes ? `${holes} hole(s)` : '']
    .filter(Boolean)
    .join(' · ');

  const handleRefresh = () => {
    setBusy(true);
    refreshAll();
  };

  return (
    <div className="app-shell">
      <Sidebar route={route} onNavigate={onNavigate} />
      <div className="min-w-0">
        <header className="topbar">
          <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 sm:px-6">
            <div className="flex flex-wrap items-center gap-2">
              <ModelSwitcher route={route} onNavigate={onNavigate} />
              <span className="hidden h-5 w-px bg-[var(--line)] sm:block" aria-hidden="true" />
              <ConnectionChip connection={connection} detail={connection !== 'live' ? staleDetail : undefined} />
              <span className="hidden text-[.875rem] text-[var(--ink-3)] md:inline">
                {healthUpdatedAt ? `checked ${humanAge((Date.now() - healthUpdatedAt) / 1000)}` : 'not checked yet'}
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <DemoSwitch />
              <Button size="sm" variant="primary" icon={RefreshCw} onClick={handleRefresh} disabled={busy}>
                {busy ? 'Refreshing' : 'Refresh'}
              </Button>
            </div>
          </div>
        </header>

        <main className="page">
          {healthError && (
            <div className="mb-4">
              <ErrorNote
                error={healthError}
                label="Disconnected from the API"
                onRetry={handleRefresh}
              />
            </div>
          )}
          {children}
        </main>
      </div>
    </div>
  );
}
