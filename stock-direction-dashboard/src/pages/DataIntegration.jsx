import { useMemo, useState } from 'react';
import { Check, Copy, RefreshCw } from 'lucide-react';
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
  Section,
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
          <Section eyebrow="Headline" title="How current is the data">
            <StatGrid columns={4}>
              <Stat
                label="Expected session"
                value={formatDate(dataHealth.expected_session)}
                info={{
                  what: 'The trading date the pipeline should have reached by now.',
                  reads: `${dataHealth.expected_session_estimated ? 'A weekday estimate, because the stored calendar lags.' : 'Taken from the stored trading calendar.'} That calendar currently ends ${formatDate(dataHealth.calendar?.last)}.`,
                  ideal: 'Equal to the newest date in the feature panel, so downstream steps have something current to read.',
                }}
              />
              <Stat
                label="Gap"
                value={int(dataHealth.gap?.gap_sessions_estimate)}
                info={{
                  what: 'How many sessions the feature panel sits behind the expected session.',
                  reads: 'Counted in weekdays since the last feature row. NSE holidays are not modelled, so the estimate can overstate the gap by a day or two.',
                  ideal: 'Zero. Anything above zero means the model is serving on stale features.',
                }}
              />
              <Stat
                label="Internal holes"
                value={int(holes?.total_holes)}
                info={{
                  what: 'Sessions missing from the middle of a symbol\'s own history.',
                  reads: `${int(holes?.symbols_with_holes)} symbol(s) have at least one missing session between their first and last bar.`,
                  ideal: 'Zero. A hole is a gap, not a zero return, and it distorts every rolling window that spans it.',
                  when: `${holes?.repair_command ?? 'The repair command'} re-downloads a short window per affected symbol.`,
                }}
              />
              <Stat
                label="Stale steps"
                value={int(dataHealth.summary?.stale_steps?.length)}
                info={{
                  what: 'Pipeline steps that have not written an artifact for the expected session.',
                  reads: `${int(dataHealth.summary?.missing_steps?.length)} missing and ${int(dataHealth.summary?.actionable?.length)} actionable.`,
                  ideal: 'Zero. One or two stale steps usually clear on the next refresh.',
                  when: dataHealth.summary?.fixable_command ?? 'Nothing is actionable right now.',
                }}
              />
            </StatGrid>
          </Section>

          <Section eyebrow="Pipeline" title="Refresh and step status">
            <Panel
              title="Pipeline run state"
              info={{
                what: 'Progress of the refresh that is running, or of the last one that finished.',
                reads: 'artifacts/state/refresh.json, which records each step and how long it took. Written only by the refresh pipeline.',
                ideal: 'Every step complete, with the heartbeat seconds old while a run is in flight.',
                when: 'A failed step lists its error. Fix the first failure: later steps usually failed because it did.',
              }}
            >
              {refreshRun?.available ? (
                <>
                  <div className="flex flex-wrap items-center gap-2.5">
                    <Badge tone={refreshRun.running ? 'badge-alert' : refreshRun.state === 'complete' ? 'badge-solid' : 'badge-dashed'}>
                      {refreshRun.running ? 'running' : refreshRun.state ?? 'unknown'}
                    </Badge>
                    <span className="mono text-[.8125rem] text-[var(--ink-3)]">
                      {refreshRun.retrain ? 'refresh + retrain' : 'data + predict only'}
                    </span>
                    {refreshRun.current_step && <Badge tone="badge-ghost">now: {refreshRun.current_step}</Badge>}
                  </div>
                  <div className="mt-4">
                    <ProgressBar
                      label="Steps"
                      value={refreshRun.step_index ?? 0}
                      max={refreshRun.step_total ?? 0}
                      detail={`${refreshRun.heartbeat_age_seconds != null ? `heartbeat ${humanDuration(refreshRun.heartbeat_age_seconds)} old` : ''} ${
                        refreshRun.duration_seconds ? `· last run ${humanDuration(refreshRun.duration_seconds)}` : ''
                      }`}
                    />
                  </div>
                  <ul className="mt-4 grid gap-2 sm:grid-cols-2">
                    {(refreshRun.steps ?? []).map((step) => (
                      <li key={step.name} className="flex items-center justify-between gap-3 border-t border-[var(--line)] pt-2">
                        <span className="mono truncate text-[.8125rem] text-[var(--ink-2)]">{step.label ?? step.name}</span>
                        <span className="flex shrink-0 items-center gap-2">
                          {step.duration_seconds != null && <span className="text-[.75rem] text-[var(--ink-3)]">{humanDuration(step.duration_seconds)}</span>}
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
                    <div className="mt-4">
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
              className="mt-6"
              title="Every pipeline step"
              info={{
                what: 'One row per artifact the pipeline writes, with its date and how far behind it is.',
                reads: `Data as of is the session the artifact covers; expected is the session it should cover; backfill says whether a repair is possible at all. Labels trail features by ${dataHealth.label_horizon_sessions} sessions (the triple-barrier horizon), so the label row is expected at ${formatDate(dataHealth.label_expected_session)} rather than ${formatDate(dataHealth.expected_session)}. Lag is counted in weekdays: NSE holidays are not modelled.`,
                ideal: 'Every row fresh, with an empty lag column.',
                when: 'Model steps are judged by artifact age rather than session lag, because a retrain can be a week old and still be correct.',
              }}
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
                { key: 'command', header: 'Repair command', render: (row) => <code className="mono text-[.75rem] text-[var(--ink-3)]">{row.command ?? '—'}</code> },
                { key: 'note', header: 'Note', render: (row) => <span className="whitespace-normal text-[.8125rem] text-[var(--ink-3)]">{row.note ?? '—'}</span> },
              ]}
              rows={shownSteps}
              rowKey={(row) => row.step}
              empty={<div className="p-4"><Empty title="Nothing to fix" detail="Every artifact reaches its expected session." /></div>}
            />
            </Panel>
          </Section>

          <Section eyebrow="Coverage" title="What is on disk and what agrees">
            <div className="grid gap-6 lg:grid-cols-2">
            <Panel
              title="Rows and spans per dataset"
              info={{
                what: 'How much data each dataset holds, over what span, and how many symbols are behind.',
                reads: 'File and row counts over data/processed plus the first and last date in each dataset.',
                ideal: 'Last dates agreeing with the expected session, and zero symbols behind.',
              }}
              flush
            >
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

            <Panel
              title="Cross-artifact checks"
              info={{
                what: 'Agreement tests between artifacts that must describe the same window.',
                reads: 'For example, whether the predictions file and the metrics file score the same rows, or whether a feature panel matches its raw source.',
                ideal: 'Every check ok. These tests exist to catch a half-finished pipeline run.',
                when: 'An error-severity check names both artifacts it compared; re-running the producing step is usually enough.',
              }}
            >
              <ul className="space-y-3">
                {(dataHealth.checks ?? []).map((check) => (
                  <li key={check.id} className="flex items-start gap-2.5 border-t border-[var(--line)] pt-3 first:border-t-0 first:pt-0">
                    <Badge tone={check.ok ? 'badge-ghost' : check.severity === 'error' ? 'badge-alert' : 'badge-dashed'}>{check.ok ? 'ok' : check.severity}</Badge>
                    <div className="min-w-0">
                      <p className="mono text-[.8125rem] font-semibold text-[var(--ink-2)]">{check.id}</p>
                      <p className="muted text-[.875rem]">{check.detail}</p>
                    </div>
                  </li>
                ))}
                {!dataHealth.checks?.length && <Empty title="No checks recorded" />}
              </ul>
            </Panel>
            </div>
          </Section>

          <Section eyebrow="Holes" title="Sessions missing from the middle of a series">
          <Panel
            title="Sessions missing inside a symbol's own coverage"
            info={{
              what: 'Sessions where a symbol has no bar, even though it has bars before and after.',
              reads: 'Each symbol\'s first and last bar are taken as its coverage, then every calendar session in between is checked for a row.',
              ideal: 'An empty table. A hole is a data gap, not a zero return.',
              when: 'Appending from the last date can never repair a hole in the middle: the pipeline thinks the symbol is current. Use the repair command it prints.',
            }}
            flush
          >
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
                      render: (row) => <span className="mono whitespace-normal text-[.75rem] text-[var(--ink-3)]">{row.missing_dates?.join(', ')}</span>,
                    },
                  ]}
                  rows={holes.detail}
                  rowKey={(row) => row.symbol}
                />
                <div className="border-t border-[var(--line)] p-4">
                  <p className="text-[.875rem] text-[var(--ink-2)]">
                    Repair with <code className="mono">{holes.repair_command}</code>, which re-downloads a short window per affected symbol and
                    merges on the timestamp.
                  </p>
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
          </Section>

          <Section eyebrow="Runbook" title="Commands that close the gap">
          <div className="grid gap-6 lg:grid-cols-3">
            <Panel
              className="lg:col-span-2"
              title="Commands that close the gap"
              info={{
                what: 'The commands that repair the problems above, in the order they should be run.',
                reads: 'Each entry prints the exact command and what it does. Copy puts it on the clipboard unchanged.',
                ideal: 'You rarely need this panel; when you do, the first entry is usually enough.',
              }}
            >
              <ul className="space-y-4">
                {RUNBOOK.map((entry) => (
                  <li key={entry.title} className="border-t border-[var(--line)] pt-4 first:border-t-0 first:pt-0">
                    <div className="flex flex-wrap items-center justify-between gap-2.5">
                      <p className="text-[.9375rem] font-semibold text-[var(--ink)]">{entry.title}</p>
                      <CopyButton value={entry.command} />
                    </div>
                    <code className="mono mt-2 block break-all rounded-md border border-[var(--line)] bg-[var(--surface-sunken)] px-3 py-2 text-[.8125rem] text-[var(--ink-2)]">
                      {entry.command}
                    </code>
                    <p className="muted mt-2 text-[.875rem]">{entry.detail}</p>
                  </li>
                ))}
              </ul>
            </Panel>

            <div className="space-y-6">
              <Panel
                title="What runs when you train"
                info={{
                  what: 'The booster training state and what training does and does not do.',
                  reads: 'Rounds recorded against the configured total, when the checkpoint was last touched, and how the run was detected.',
                  ideal: 'Round count equal to the configured total with a recent checkpoint.',
                  when: 'Training never refreshes data by itself: it fits on whatever features already exist.',
                }}
              >
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
                <p className="mt-4 text-[.875rem] text-[var(--ink-2)]">
                  <span className="mono">python -m stockml.main train</span> fits on whatever features already exist. Use{' '}
                  <span className="mono">.\refresh_data.ps1 -Retrain</span> so fetch → features → labels complete first.
                </p>
              </Panel>

              <Panel
                title="Q0 runtime"
                info={{
                  what: 'The background Q0 research runner, as it last reported.',
                  reads: 'Status and phase from its heartbeat file, how stale that heartbeat is, and the gate verdict it last wrote.',
                  ideal: 'Heartbeat seconds old and a phase that is advancing. A gate either passes or names the condition it failed.',
                  when: 'A heartbeat hours old means the runner stopped without saying so; the gate verdict then refers to an older run.',
                }}
              >
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

          <Panel
            className="mt-6"
            title="Read-only endpoints"
            info={{
              what: 'Every route the dashboard reads from, and what each one returns.',
              reads: `Base URL ${API_BASE || 'the same origin through the Vite proxy'}. Every response uses the same envelope, so freshness and warnings travel with the data.`,
              ideal: 'GET only, artifacts served from an allowlist, no path traversal, and a malformed file degrading to a warning instead of an error.',
              when: 'A route missing here cannot be reached by the client at all. The legacy mutating GET /predictions is deliberately absent.',
            }}
            flush
          >
            <Table
              dense
              columns={[
                { key: 'path', header: 'Route', strong: true, mono: true },
                { key: 'note', header: 'Returns', render: (row) => <span className="whitespace-normal text-[.875rem] text-[var(--ink-3)]">{row.note}</span> },
              ]}
              rows={ENDPOINTS}
              rowKey={(row) => row.path}
            />
            <div className="border-t border-[var(--line)] p-4">
              <p className="muted text-[.875rem]">
                Envelope:{' '}
                <span className="mono">{'{ data, as_of, generated_at, source: { kind, artifacts }, warnings }'}</span>
              </p>
              <div className="mt-3 flex flex-wrap gap-1.5">
                <Badge tone="badge-ghost">GET only</Badge>
                <Badge tone="badge-ghost">artifact allowlist</Badge>
                <Badge tone="badge-ghost">no path traversal</Badge>
                <Badge tone="badge-ghost">malformed files degrade to warnings</Badge>
              </div>
            </div>
          </Panel>
          </Section>

          <Section eyebrow="Caveats" title="What cannot be repaired, and what is behind">
            <div className="grid gap-6 lg:grid-cols-2">
              <Panel
                title="What cannot be backfilled"
                info={{
                  what: 'Sources where a missed session is gone for good, not merely late.',
                  reads: 'These rows never appear as repairs the dashboard could trigger. Forward-only sources are marked as such in the step table.',
                  ideal: 'An empty history of missed sessions. Once a day is unpolled for one of these, no command brings it back.',
                }}
              >
                <ul className="muted space-y-3">
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
              </Panel>

              <Panel
                title="Behind per dataset"
                info={{
                  what: 'Which datasets still have symbols short of the expected session.',
                  reads: 'Each entry names the dataset, how many symbols are behind, and the newest and oldest last-date among them.',
                  ideal: 'Nothing listed. Every dataset current means the feature panel can be rebuilt without forward-filling.',
                }}
              >
                <ul className="space-y-3">
                  {coverage
                    .filter(([, value]) => value.symbols_behind > 0)
                    .map(([name, value]) => (
                      <li key={name} className="border-t border-[var(--line)] pt-3 first:border-t-0 first:pt-0">
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-[.9375rem] font-semibold text-[var(--ink)]">{titleize(name)}</span>
                          <Badge tone="badge-alert">{value.symbols_behind} behind</Badge>
                        </div>
                        <p className="mono mt-1.5 break-words text-[.8125rem] text-[var(--ink-3)]">{value.behind?.join(', ')}</p>
                        <p className="mt-1 text-[.8125rem] text-[var(--ink-3)]">
                          last {formatDate(value.last)} · oldest {formatDate(value.last_min)}
                        </p>
                      </li>
                    ))}
                  {!coverage.some(([, value]) => value.symbols_behind > 0) && <Empty title="Every dataset is current" />}
                </ul>
              </Panel>
            </div>
          </Section>
        </>
      )}
    </>
  );
}
