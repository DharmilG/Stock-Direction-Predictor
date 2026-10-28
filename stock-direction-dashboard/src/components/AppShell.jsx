import { useEffect, useState } from 'react';
import { Activity, AlertTriangle, Database, FileJson, Layers, RefreshCw, Sparkles } from 'lucide-react';
import { NAV_SECTIONS, STATUS_LABEL } from '../lib/constants';
import { formatInstant, humanAge } from '../lib/dates';
import { useDashboard } from '../state/DashboardProvider';
import { Badge, Button, ErrorNote, IconButton } from './ui';

const SECTION_ICONS = {
  Research: Activity,
  Model: Layers,
  'LLM & data': Database,
};

function ConnectionChip({ connection }) {
  const tone =
    connection === 'live' ? 'badge-solid' : connection === 'stale' ? 'badge-alert' : connection === 'connecting' ? 'badge-dashed' : 'badge-alert';
  return (
    <span className={`badge ${tone}`} title="Derived from GET /dashboard/health and the freshness of the latest artifacts">
      <span aria-hidden="true">{connection === 'live' ? '●' : connection === 'stale' ? '◐' : '○'}</span>
      {STATUS_LABEL[connection] ?? connection}
    </span>
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
          <div className="min-w-0">
            <p className="truncate text-[.8125rem] font-semibold leading-tight text-[var(--ink)]">Stock Direction</p>
            <p className="truncate text-[.6875rem] text-[var(--ink-3)]">Research dashboard</p>
          </div>
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
            <div key={section.label} className="mb-3">
              <p className="eyebrow flex items-center gap-1.5 px-2 py-1">
                <Icon size={11} strokeWidth={2.5} />
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
        <p className="eyebrow">Universe tier</p>
        <p className="mono mt-1 truncate text-[.7rem] text-[var(--ink-2)]">{universeTier ?? 'not recorded'}</p>
        <p className="mt-2 text-[.625rem] leading-relaxed text-[var(--ink-faint)]">
          Version {health?.version ?? '—'} · read-only API
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
              <ConnectionChip connection={connection} />
              {connection !== 'live' && (staleSteps.length > 0 || gap > 0) && (
                <span className="hidden items-center gap-1.5 text-[.6875rem] text-[var(--ink-3)] sm:inline-flex">
                  <AlertTriangle size={12} strokeWidth={2.5} />
                  {staleSteps.length} stale step(s) · gap {gap} session(s){holes ? ` · ${holes} hole(s)` : ''}
                </span>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <span className="hidden text-[.6875rem] text-[var(--ink-3)] md:inline">
                {healthUpdatedAt ? `checked ${humanAge((Date.now() - healthUpdatedAt) / 1000)}` : 'not checked yet'}
              </span>
              <DemoSwitch />
              <IconButton label="Refresh dashboard data" icon={RefreshCw} onClick={handleRefresh} disabled={busy} />
              <Button size="sm" variant="primary" icon={RefreshCw} onClick={handleRefresh} disabled={busy}>
                Refresh
              </Button>
              {healthUpdatedAt && (
                <span className="hidden text-[.625rem] text-[var(--ink-faint)] xl:inline">{formatInstant(new Date(healthUpdatedAt))}</span>
              )}
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
