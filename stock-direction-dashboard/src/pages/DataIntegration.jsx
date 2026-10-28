import { useMemo, useState } from 'react';
import { Check, Copy, RefreshCw, Server, Terminal } from 'lucide-react';
import { API_BASE } from '../lib/api';
import { RUNBOOK } from '../lib/constants';
import { formatDate, formatInstant, humanDuration } from '../lib/dates';
import { int, titleize } from '../lib/format';
import { useDashboard } from '../state/DashboardProvider';
import {
  Badge,
  Button,
  Callout,
  Empty,
  ErrorNote,
  KeyValue,
  Loading,
  PageHeader,
  Panel,
  ProgressBar,
  SourceTag,
  Stat,
  StatGrid,
  Table,
} from '../components/ui';

const STATUS_TONE = { fresh: 'badge-ghost', stale: 'badge-alert', missing: 'badge-alert', partial: 'badge-dashed' };

const ENDPOINTS = [
  { path: '/dashboard/health', note: 'Artifact presence and service version' },
  { path: '/dashboard/symbols', note: 'Configured universe and tier' },
  { path: '/dashboard/overview', note: 'Dataset span, evaluation, backtest, freshness summary' },
  { path: '/dashboard/predictions?limit=100', note: 'Serving rows, ranked by confidence' },
  { path: '/dashboard/metrics', note: 'Holdout metrics, confusion matrix, decision rule' },
  { path: '/dashboard/model-metadata', note: 'Trained-at, features, best params, checkpoint' },
  { path: '/dashboard/shap?limit=25', note: 'Mean absolute SHAP importance' },
  { path: '/dashboard/backtest', note: 'Metrics, weekly equity curve, per-symbol breakdown' },
  { path: '/dashboard/backtest/trades?outcome=all', note: 'Event-level trades with costs' },
  { path: '/dashboard/monitoring', note: 'PSI drift alerts' },
  { path: '/dashboard/stock/{symbol}?points=180', note: 'Price plus a curated feature panel' },
  { path: '/dashboard/stock/{symbol}/predictions', note: 'Scored prediction history for one symbol' },
  { path: '/dashboard/data-health', note: 'Freshness, gaps, holes, consistency checks' },
  { path: '/dashboard/activity', note: 'Most recent artifact mtimes' },
  { path: '/dashboard/q0/status', note: 'Live Q0 runner, gate, thermal telemetry' },
  { path: '/dashboard/q0/ledger-summary', note: 'Rated / judged counts and distributions' },
];

function CopyButton({ value }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      size="sm"
      variant="quiet"
      icon={copied ? Check : Copy}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          setTimeout(() => setCopied(false), 1600);
        } catch {
          setCopied(false);
        }
      }}
    >
      {copied ? 'Copied' : 'Copy'}
    </Button>
  );
}

export default function DataIntegration() {
  const { dataHealth, dataHealthEnvelope, dataHealthError, refreshDataHealth, q0 } = useDashboard();
  const [onlyProblems, setOnlyProblems] = useState(false);

  const shownSteps = useMemo(() => {
    const all = dataHealth?.steps ?? [];
    return onlyProblems ? all.filter((step) => step.status !== 'fresh') : all;
  }, [dataHealth, onlyProblems]);
  const coverage = Object.entries(dataHealth?.coverage ?? {});
  const holes = dataHealth?.holes;
  const refreshRun = dataHealth?.refresh_run;
  const training = dataHealth?.training;

  return (
    <>
      <PageHeader
        eyebrow="Data & integration"
        title="Freshness, gaps and the repair path"
        description="The single place that answers: what is the latest session, which artifacts are behind it, what can be backfilled, and which command fixes it."
        actions={
          <>
            <SourceTag envelope={dataHealthEnvelope} prefix="expected" />
            <Button size="sm" variant="quiet" icon={RefreshCw} onClick={refreshDataHealth}>
              Recheck
            </Button>
          </>
        }
      />

      {dataHealthError && (
        <div className="mb-4">
          <ErrorNote error={dataHealthError} label="Data health unavailable" onRetry={refreshDataHealth} />
        </div>
      )}
      {!dataHealth && !dataHealthError && <Loading rows={6} label="Scanning artifacts" />}

      {dataHealth && (
        <>
          <StatGrid columns={4}>
            <Stat
              label="Expected session"
              value={formatDate(dataHealth.expected_session)}
              hint={dataHealth.expected_session_estimated ? 'Weekday estimate — the stored calendar lags' : 'From the stored trading calendar'}
              footnote={`Calendar last: ${formatDate(dataHealth.calendar?.last)}`}
            />
            <Stat
              label="Gap"
              value={int(dataHealth.gap?.gap_sessions_estimate)}
              hint="Weekday sessions since the feature panel ended"
              footnote="NSE holidays are not modelled in this estimate"
            />
            <Stat
              label="Internal holes"
              value={int(holes?.total_holes)}
              hint={`${int(holes?.symbols_with_holes)} symbol(s) missing a session inside their own coverage`}
              footnote={holes?.repair_command}
            />
            <Stat
              label="Stale steps"
              value={int(dataHealth.summary?.stale_steps?.length)}
              hint={`${int(dataHealth.summary?.missing_steps?.length)} missing · ${int(dataHealth.summary?.actionable?.length)} actionable`}
              footnote={dataHealth.summary?.fixable_command ?? 'Nothing actionable'}
            />
          </StatGrid>

          <Panel className="mt-4" eyebrow="Refresh" title="Pipeline run state" actions={<RefreshCw size={14} className="text-[var(--ink-faint)]" />}>
            {refreshRun?.available ? (
              <>
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone={refreshRun.running ? 'badge-alert' : refreshRun.state === 'complete' ? 'badge-solid' : 'badge-dashed'}>
                    {refreshRun.running ? 'running' : refreshRun.state ?? 'unknown'}
                  </Badge>
                  <span className="mono text-[.7rem] text-[var(--ink-3)]">
                    {refreshRun.retrain ? 'refresh + retrain' : 'data + predict only'}
                  </span>
                  {refreshRun.current_step && <Badge tone="badge-ghost">now: {refreshRun.current_step}</Badge>}
                </div>
                <div className="mt-3">
                  <ProgressBar
                    label="Steps"
                    value={refreshRun.step_index ?? 0}
                    max={refreshRun.step_total ?? 0}
                    detail={`${refreshRun.heartbeat_age_seconds != null ? `heartbeat ${humanDuration(refreshRun.heartbeat_age_seconds)} old` : ''} ${
                      refreshRun.duration_seconds ? `· last run ${humanDuration(refreshRun.duration_seconds)}` : ''
                    }`}
                  />
                </div>
                <ul className="mt-3 grid gap-1.5 sm:grid-cols-2">
                  {(refreshRun.steps ?? []).map((step) => (
                    <li key={step.name} className="flex items-center justify-between gap-2 border-t border-[var(--line)] pt-1.5">
                      <span className="mono truncate text-[.7rem] text-[var(--ink-2)]">{step.label ?? step.name}</span>
                      <span className="flex shrink-0 items-center gap-1.5">
                        {step.duration_seconds != null && <span className="text-[.625rem] text-[var(--ink-faint)]">{humanDuration(step.duration_seconds)}</span>}
                        <Badge
                          tone={
                            step.status === 'complete'
                              ? 'badge-ghost'
                              : step.status === 'running'
                                ? 'badge-alert'
                                : step.status === 'failed'
                                  ? 'badge-alert'
                                  : 'badge-dashed'
                          }
                        >
                          {step.status}
                        </Badge>
                      </span>
                    </li>
                  ))}
                </ul>
                {refreshRun.failures?.length > 0 && (
                  <div className="mt-3">
                    <Callout tone="warn" title={`${refreshRun.failures.length} step failure(s)`}>
                      {refreshRun.failures.map((failure) => `${failure.step}: ${failure.error}`).join(' · ')}
                    </Callout>
                  </div>
                )}
              </>
            ) : (
              <Empty
                title="No refresh has run yet"
                detail="artifacts/state/refresh.json appears the first time you run the refresh pipeline. It is the only way the dashboard can show step-level progress."
              />
            )}
          </Panel>

          <Panel
            className="mt-4"
            eyebrow="Freshness"
            title="Every pipeline step"
            flush
            actions={
              <Button size="sm" variant={onlyProblems ? 'primary' : 'quiet'} aria-pressed={onlyProblems} onClick={() => setOnlyProblems((value) => !value)}>
                Only problems
              </Button>
            }
          >
            <Table
              columns={[
                { key: 'label', header: 'Step', strong: true },
                { key: 'status', header: 'Status', render: (row) => <Badge tone={STATUS_TONE[row.status] ?? 'badge-ghost'}>{row.status}</Badge> },
                { key: 'as_of', header: 'Data as of', render: (row) => <span className="mono">{formatDate(row.as_of)}</span> },
                { key: 'expected_as_of', header: 'Expected', render: (row) => <span className="mono">{formatDate(row.expected_as_of)}</span> },
                {
                  key: 'lag_sessions',
                  header: 'Lag',
                  align: 'right',
                  render: (row) => {
                    // Model steps are judged by artifact age, not by session lag:
                    // a retrain can be older than the data without being wrong.
                    if (row.age_based) {
                      const days = row.age_seconds != null ? Math.floor(row.age_seconds / 86400) : null;
                      return <span title="Age-based: judged by artifact age, not by session lag">{days != null ? `${days} d` : '—'}</span>;
                    }
                    return row.lag_sessions ? `${row.lag_sessions}` : '—';
                  },
                },
                {
                  key: 'backfillable',
                  header: 'Backfill',
                  render: (row) => <Badge tone={row.backfillable ? 'badge-ghost' : 'badge-dashed'}>{row.backfillable ? 'yes' : 'forward-only'}</Badge>,
                },
                { key: 'command', header: 'Repair command', render: (row) => <code className="mono text-[.65rem] text-[var(--ink-3)]">{row.command ?? '—'}</code> },
                { key: 'note', header: 'Note', render: (row) => <span className="whitespace-normal text-[.7rem] text-[var(--ink-3)]">{row.note ?? '—'}</span> },
              ]}
              rows={shownSteps}
              rowKey={(row) => row.step}
              empty={<div className="p-4"><Empty title="Nothing to fix" detail="Every artifact reaches its expected session." /></div>}
            />
          </Panel>

          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            <Panel eyebrow="Coverage" title="Rows and spans per dataset" flush>
              <Table
                dense
                columns={[
                  { key: 'name', header: 'Dataset', strong: true },
                  { key: 'files', header: 'Files', align: 'right', render: ([, value]) => int(value.files) },
                  { key: 'rows', header: 'Rows', align: 'right', render: ([, value]) => int(value.rows) },
                  { key: 'first', header: 'First', render: ([, value]) => formatDate(value.first) },
                  { key: 'last', header: 'Last', render: ([, value]) => formatDate(value.last) },
                  { key: 'behind', header: 'Behind', align: 'right', render: ([, value]) => (value.symbols_behind ? `${value.symbols_behind}` : '—') },
                ]}
                rows={coverage}
                rowKey={([name]) => name}
                empty={<div className="p-4"><Empty title="No parquet datasets found" /></div>}
              />
            </Panel>

            <Panel eyebrow="Integrity" title="Cross-artifact checks">
              <ul className="space-y-2">
                {(dataHealth.checks ?? []).map((check) => (
                  <li key={check.id} className="flex items-start gap-2 border-t border-[var(--line)] pt-2 first:border-t-0 first:pt-0">
                    <Badge tone={check.ok ? 'badge-ghost' : check.severity === 'error' ? 'badge-alert' : 'badge-dashed'}>{check.ok ? 'ok' : check.severity}</Badge>
                    <div className="min-w-0">
                      <p className="mono text-[.7rem] font-semibold text-[var(--ink-2)]">{check.id}</p>
                      <p className="muted text-[.75rem]">{check.detail}</p>
                    </div>
                  </li>
                ))}
                {!dataHealth.checks?.length && <Empty title="No checks recorded" />}
              </ul>
            </Panel>
          </div>

          <Panel className="mt-4" eyebrow="Holes" title="Sessions missing inside a symbol's own coverage" flush>
            {holes?.available && holes.detail?.length ? (
              <>
                <Table
                  dense
                  columns={[
                    { key: 'symbol', header: 'Symbol', strong: true },
                    { key: 'missing_sessions', header: 'Missing', align: 'right', render: (row) => int(row.missing_sessions) },
                    { key: 'first_missing', header: 'First', render: (row) => formatDate(row.first_missing) },
                    { key: 'last_missing', header: 'Last', render: (row) => formatDate(row.last_missing) },
                    {
                      key: 'missing_dates',
                      header: 'Dates (first 10)',
                      render: (row) => <span className="mono whitespace-normal text-[.65rem] text-[var(--ink-3)]">{row.missing_dates?.join(', ')}</span>,
                    },
                  ]}
                  rows={holes.detail}
                  rowKey={(row) => row.symbol}
                />
                <div className="border-t border-[var(--line)] p-3">
                  <Callout tone="warn" title="Why this matters">
                    <span>
                      Appending from <code className="mono">last_date + 1</code> can never repair a hole in the middle of a series: the pipeline thinks
                      the symbol is current. <code className="mono">{holes.repair_command}</code> re-downloads a short window per affected symbol and
                      merges on the timestamp, which is safe because prices are idempotent per day.
                    </span>
                  </Callout>
                </div>
              </>
            ) : (
              <div className="p-4">
                <Empty
                  title={holes?.available ? 'No internal holes' : 'Hole scan unavailable'}
                  detail={holes?.available ? 'Every symbol has a bar for each session inside its own first/last window.' : holes?.reason}
                />
              </div>
            )}
          </Panel>

          <div className="mt-4 grid gap-4 lg:grid-cols-3">
            <Panel className="lg:col-span-2" eyebrow="Runbook" title="Commands that close the gap" actions={<Terminal size={14} className="text-[var(--ink-faint)]" />}>
              <ul className="space-y-3">
                {RUNBOOK.map((entry) => (
                  <li key={entry.title} className="border-t border-[var(--line)] pt-3 first:border-t-0 first:pt-0">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <p className="text-[.8125rem] font-semibold text-[var(--ink)]">{entry.title}</p>
                      <CopyButton value={entry.command} />
                    </div>
                    <code className="mono mt-1 block break-all rounded-md border border-[var(--line)] bg-[var(--surface-sunken)] px-2 py-1.5 text-[.7rem] text-[var(--ink-2)]">
                      {entry.command}
                    </code>
                    <p className="muted mt-1 text-[.75rem]">{entry.detail}</p>
                  </li>
                ))}
              </ul>
            </Panel>

            <div className="space-y-4">
              <Panel eyebrow="Training" title="What runs when you train">
                <KeyValue
                  columns={1}
                  items={[
                    {
                      label: 'LightGBM',
                      value: training?.lightgbm?.active
                        ? `active · ${int(training.lightgbm.completed_rounds)}/${int(training.lightgbm.configured_rounds)} rounds`
                        : `${int(training?.lightgbm?.completed_rounds)}/${int(training?.lightgbm?.configured_rounds)} rounds recorded`,
                      mono: true,
                    },
                    { label: 'Checkpoint touched', value: training?.lightgbm?.checkpoint_modified_at ? formatInstant(training.lightgbm.checkpoint_modified_at) : '—' },
                    { label: 'Detection', value: training?.lightgbm?.detection ?? '—' },
                    { label: 'QLoRA', value: training?.qwen_qlora?.available ? 'instrumented' : 'not instrumented' },
                  ]}
                />
                <div className="mt-3">
                  <Callout tone="info" title="Training does not refresh data by itself">
                    <span className="mono">python -m stockml.main train</span> trains on whatever features already exist. Use{' '}
                    <span className="mono">.\refresh_data.ps1 -Retrain</span> so fetch → features → labels complete before the model is fitted.
                  </Callout>
                </div>
              </Panel>

              <Panel eyebrow="Live" title="Q0 runtime" actions={<Server size={14} className="text-[var(--ink-faint)]" />}>
                <KeyValue
                  columns={1}
                  items={[
                    { label: 'Status', value: q0?.runtime?.status ?? '—', mono: true },
                    { label: 'Phase', value: q0?.runtime?.phase ?? '—', mono: true },
                    { label: 'Heartbeat age', value: humanDuration(q0?.runtime?.heartbeat_age_seconds), mono: true },
                    { label: 'Gate', value: q0?.gate?.verdict ?? '—', mono: true },
                  ]}
                />
              </Panel>
            </div>
          </div>

          <Panel className="mt-4" eyebrow="Integration" title="Read-only endpoints" flush>
            <Table
              dense
              columns={[
                { key: 'path', header: 'Route', strong: true, mono: true },
                { key: 'note', header: 'Returns', render: (row) => <span className="whitespace-normal text-[.75rem] text-[var(--ink-3)]">{row.note}</span> },
              ]}
              rows={ENDPOINTS}
              rowKey={(row) => row.path}
            />
            <div className="border-t border-[var(--line)] p-3">
              <p className="muted text-[.75rem]">
                Base URL <span className="mono">{API_BASE || 'same origin (Vite proxy)'}</span>. Every response uses the same envelope:{' '}
                <span className="mono">{'{ data, as_of, generated_at, source: { kind, artifacts }, warnings }'}</span>. There is no POST, PUT or DELETE
                route, and the legacy mutating <span className="mono">GET /predictions</span> is intentionally absent from the client.
              </p>
              <div className="mt-2 flex flex-wrap gap-1.5">
                <Badge tone="badge-ghost">GET only</Badge>
                <Badge tone="badge-ghost">artifact allowlist</Badge>
                <Badge tone="badge-ghost">no path traversal</Badge>
                <Badge tone="badge-ghost">malformed files degrade to warnings</Badge>
              </div>
            </div>
          </Panel>

          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            <Panel eyebrow="Honesty" title="What cannot be backfilled">
              <ul className="muted space-y-2">
                <li>
                  <strong className="text-[var(--ink)]">RSS headlines</strong> are one snapshot per day. A day not polled is gone.
                </li>
                <li>
                  <strong className="text-[var(--ink)]">Bulk / block deals</strong> come from an NSE endpoint with no history.
                </li>
                <li>
                  <strong className="text-[var(--ink)]">News</strong> is capped at 25 AlphaVantage requests per day and GDELT DOC only covers a rolling
                  window, so a long gap needs several runs.
                </li>
                <li>
                  <strong className="text-[var(--ink)]">GDELT BigQuery</strong> is backfillable but scans ~164 GB; check the quota state first.
                </li>
              </ul>
              <p className="mt-3 text-[.6875rem] text-[var(--ink-faint)]">
                Forward-only sources are marked as such in the table above and never counted as repairs the dashboard could trigger.
              </p>
            </Panel>

            <Panel eyebrow="Coverage detail" title="Behind per dataset">
              <ul className="space-y-2">
                {coverage
                  .filter(([, value]) => value.symbols_behind > 0)
                  .map(([name, value]) => (
                    <li key={name} className="border-t border-[var(--line)] pt-2 first:border-t-0 first:pt-0">
                      <div className="flex items-center justify-between">
                        <span className="text-[.8125rem] font-semibold text-[var(--ink)]">{titleize(name)}</span>
                        <Badge tone="badge-alert">{value.symbols_behind} behind</Badge>
                      </div>
                      <p className="mono mt-1 break-words text-[.65rem] text-[var(--ink-3)]">{value.behind?.join(', ')}</p>
                      <p className="text-[.6875rem] text-[var(--ink-faint)]">
                        last {formatDate(value.last)} · min {formatDate(value.last_min)}
                      </p>
                    </li>
                  ))}
                {!coverage.some(([, value]) => value.symbols_behind > 0) && <Empty title="Every dataset is current" />}
              </ul>
            </Panel>
          </div>

          <p className="mt-4 text-[.6875rem] leading-relaxed text-[var(--ink-faint)]">
            Labels intentionally trail features by {dataHealth.label_horizon_sessions} sessions (the triple-barrier horizon), so the label expected
            session is {formatDate(dataHealth.label_expected_session)} rather than {formatDate(dataHealth.expected_session)}. Lag counts are weekday
            estimates: NSE holidays are not modelled by the dashboard, and the stored calendar is itself an artifact that can lag.
          </p>
        </>
      )}
    </>
  );
}
