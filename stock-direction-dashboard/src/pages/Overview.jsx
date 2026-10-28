import { AlertTriangle, CheckCircle2, Clock, Database } from 'lucide-react';
import { api } from '../lib/api';
import { POLL_INTERVALS } from '../lib/constants';
import { formatDate, formatInstant, humanAge, humanDuration } from '../lib/dates';
import { pct, int, signedPct, num } from '../lib/format';
import { useResource } from '../hooks/useResource';
import { useDashboard } from '../state/DashboardProvider';
import { DistributionBars } from '../components/charts';
import { SignalBadge } from '../components/Signal';
import {
  Badge,
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
  Warnings,
} from '../components/ui';

function StatusBadge({ status }) {
  const tone = status === 'fresh' ? 'badge-ghost' : status === 'stale' ? 'badge-alert' : 'badge-dashed';
  return <Badge tone={tone}>{status}</Badge>;
}

export default function Overview({ onNavigate }) {
  const { dataHealth, health } = useDashboard();
  const overview = useResource(api.overview, { key: 'overview', pollMs: POLL_INTERVALS.artifacts });
  const activity = useResource(api.activity, { key: 'activity', pollMs: POLL_INTERVALS.artifacts });

  const data = overview.data;
  const envelope = overview.envelope;
  const training = data?.training?.lightgbm;
  const fresh = data?.freshness;
  const actionable = dataHealth?.summary?.actionable ?? [];
  const refreshRun = dataHealth?.refresh_run;
  const gap = fresh?.gap_sessions_estimate ?? dataHealth?.gap?.gap_sessions_estimate ?? 0;
  const behind = (dataHealth?.steps ?? []).filter((step) => step.status !== 'fresh').slice(0, 6);

  return (
    <>
      <PageHeader
        eyebrow="Overview"
        title="Platform state at a glance"
        description="Every number below is read from a file the pipeline wrote. Nothing on this page triggers inference, training or a fetch."
        actions={<SourceTag envelope={envelope} />}
      />

      {overview.error && (
        <div className="mb-4">
          <ErrorNote error={overview.error} label="Overview unavailable" onRetry={overview.refresh} />
        </div>
      )}
      {envelope?.warnings?.length > 0 && (
        <div className="mb-4">
          <Warnings warnings={envelope.warnings} />
        </div>
      )}

      <StatGrid columns={4}>
        <Stat
          label="Expected session"
          value={formatDate(fresh?.expected_session)}
          hint={fresh?.expected_session_estimated ? 'Weekday estimate: the stored calendar lags behind' : 'From the stored trading calendar'}
          footnote={health?.status === 'ok' ? 'API reachable · artifacts present' : undefined}
        />
        <Stat
          label="Feature panel"
          value={formatDate(fresh?.features_as_of)}
          hint={gap > 0 ? `${gap} session(s) behind the expected session` : 'Current with the expected session'}
          footnote={data?.dataset?.feature_rows ? `${int(data.dataset.feature_rows)} feature rows` : undefined}
        />
        <Stat
          label="Test accuracy"
          value={pct(data?.evaluation?.accuracy)}
          hint={`${data?.evaluation?.beats_majority_baseline ? 'Beats' : 'Does not beat'} the majority baseline (${pct(
            data?.evaluation?.majority_baseline_accuracy,
          )})`}
          footnote={data?.evaluation?.n_test ? `${int(data.evaluation.n_test)} test rows from ${formatDate(data.evaluation.test_start)}` : undefined}
        />
        <Stat
          label="Backtest total return"
          value={signedPct(data?.backtest?.total_return)}
          hint={`Max drawdown ${pct(data?.backtest?.max_drawdown)} · Sharpe ${num(data?.backtest?.sharpe, 2)}`}
          footnote={data?.backtest?.trade_count ? `${int(data.backtest.trade_count)} trades · ${data.backtest.segment ?? 'segment n/a'}` : undefined}
        />
      </StatGrid>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Panel
          className="lg:col-span-2"
          eyebrow="Freshness"
          title={behind.length ? `${behind.length} step(s) behind the expected session` : 'Every pipeline step is current'}
          actions={
            <button type="button" className="btn btn-quiet" data-size="sm" onClick={() => onNavigate('data')}>
              Open data health
            </button>
          }
          flush
        >
          {behind.length ? (
            <Table
              columns={[
                { key: 'label', header: 'Step', strong: true },
                { key: 'status', header: 'Status', render: (row) => <StatusBadge status={row.status} /> },
                { key: 'as_of', header: 'As of', render: (row) => <span className="mono">{formatDate(row.as_of, { withYear: false })}</span> },
                { key: 'lag_sessions', header: 'Lag', align: 'right', render: (row) => (row.lag_sessions ? `${row.lag_sessions}` : '—') },
                { key: 'command', header: 'Fix', render: (row) => <span className="mono text-[.7rem] text-[var(--ink-3)]">{row.command ?? '—'}</span> },
              ]}
              rows={behind}
              rowKey={(row) => row.step}
            />
          ) : (
            <div className="p-4">
              <Empty title="No stale steps" detail="The feature panel and predictions both reach the expected session." />
            </div>
          )}
          <div className="space-y-2 border-t border-[var(--line)] p-4">
            {refreshRun?.running && (
              <Callout tone="info" title="A refresh is running right now">
                Step {refreshRun.step_index ?? '—'} of {refreshRun.step_total ?? '—'} · {refreshRun.current_step ?? 'starting'} · started{' '}
                {humanAge(refreshRun.age_seconds)}
              </Callout>
            )}
            {actionable.length > 0 && (
              <Callout tone="warn" title="One command closes the gap">
                <code className="mono">python -m stockml.main refresh</code> runs{' '}
                {actionable.map((item) => item.step).join(' → ')} in dependency order. It never retrains, so the holdout window does not move.
              </Callout>
            )}
          </div>
        </Panel>

        <div className="space-y-4">
          <Panel eyebrow="Serving" title="Latest prediction mix">
            {overview.loading && !data ? (
              <Loading rows={4} />
            ) : data?.predictions?.rows ? (
              <>
                <div className="mb-3 flex items-center justify-between">
                  <SignalBadge signal="UP" />
                  <span className="mono text-[.7rem] text-[var(--ink-3)]">mean confidence {pct(data.predictions.mean_confidence)}</span>
                </div>
                <DistributionBars distribution={data.predictions.distribution} />
                <p className="muted mt-3 text-[.6875rem]">
                  {int(data.predictions.rows)} rows dated {formatDate(data.predictions.as_of)}.
                </p>
              </>
            ) : (
              <Empty title="No predictions yet" detail="results/latest_predictions.csv is missing or empty." />
            )}
          </Panel>

          <Panel eyebrow="Model" title="Training snapshot">
            <ProgressBar
              label="LightGBM rounds"
              value={training?.completed_rounds ?? 0}
              max={training?.configured_rounds ?? 1000}
              detail={
                training?.active
                  ? `Active: checkpoint touched ${humanAge(Math.round((Date.now() - new Date(training.checkpoint_modified_at).getTime()) / 1000))}`
                  : training?.checkpoint_modified_at
                    ? `Last checkpoint ${humanAge(Math.round((Date.now() - new Date(training.checkpoint_modified_at).getTime()) / 1000))}`
                    : 'No checkpoint recorded'
              }
            />
            <div className="mt-3 space-y-2">
              <KeyValue
                columns={1}
                items={[
                  { label: 'Trained at', value: data?.model?.trained_at ? formatInstant(data.model.trained_at) : '—' },
                  { label: 'Feature count', value: data?.model?.feature_count ?? '—', mono: true },
                  { label: 'Temperature', value: data?.model?.temperature?.toFixed?.(4) ?? '—', mono: true },
                ]}
              />
              <Callout tone="info">
                QLoRA telemetry is {data?.training?.qwen_qlora?.available ? 'available' : 'not instrumented'}:{' '}
                {data?.training?.qwen_qlora?.note ?? 'no live step or loss data is written.'}
              </Callout>
            </div>
          </Panel>
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel eyebrow="Dataset" title="Coverage" actions={<Database size={14} className="text-[var(--ink-faint)]" />}>
          <KeyValue
            items={[
              { label: 'Universe', value: `${data?.dataset?.symbols ?? '—'} symbols`, mono: true },
              { label: 'Feature rows', value: int(data?.dataset?.feature_rows), mono: true },
              { label: 'Feature span', value: `${formatDate(data?.dataset?.feature_first)} → ${formatDate(data?.dataset?.feature_last)}` },
              { label: 'Dataset rows', value: int(data?.dataset?.dataset_rows), mono: true },
              { label: 'Label horizon', value: `${data?.dataset?.horizon_days ?? '—'} sessions`, mono: true },
              { label: 'Flat band', value: num(data?.dataset?.flat_band, 4), mono: true },
            ]}
          />
        </Panel>

        <Panel
          eyebrow="Health"
          title="Checks and drift"
          actions={
            <button type="button" className="btn btn-quiet" data-size="sm" onClick={() => onNavigate('data')}>
              Details
            </button>
          }
        >
          <ul className="space-y-2">
            {(dataHealth?.checks ?? []).slice(0, 5).map((check) => (
              <li key={check.id} className="flex items-start gap-2 border-t border-[var(--line)] pt-2 first:border-t-0 first:pt-0">
                {check.ok ? (
                  <CheckCircle2 size={14} className="mt-0.5 shrink-0 text-[var(--ink-3)]" strokeWidth={2.25} />
                ) : (
                  <AlertTriangle size={14} className="mt-0.5 shrink-0 text-[var(--ink)]" strokeWidth={2.5} />
                )}
                <div className="min-w-0">
                  <p className="mono text-[.7rem] font-semibold text-[var(--ink-2)]">{check.id}</p>
                  <p className="muted text-[.75rem]">{check.detail}</p>
                </div>
              </li>
            ))}
            {!dataHealth?.checks?.length && <Empty title="No checks recorded" />}
          </ul>
          <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-[var(--line)] pt-3">
            <Badge tone="badge-ghost">drift alerts {data?.drift?.alerts ?? 0}</Badge>
            <Badge tone="badge-ghost">missing rate {pct(data?.drift?.missing_rate)}</Badge>
            <Badge tone="badge-ghost">latest {formatDate(data?.drift?.latest_data, { withYear: false })}</Badge>
          </div>
        </Panel>
      </div>

      <Panel className="mt-4" eyebrow="Artifacts" title="Recently written files" flush actions={<Clock size={14} className="text-[var(--ink-faint)]" />}>
        <Table
          columns={[
            { key: 'path', header: 'File', strong: true, mono: true },
            { key: 'modified_at', header: 'Modified (IST)', render: (row) => formatInstant(row.modified_at) },
            { key: 'age_seconds', header: 'Age', align: 'right', render: (row) => humanDuration(row.age_seconds) },
            { key: 'size_bytes', header: 'Size', align: 'right', render: (row) => `${Math.max(1, Math.round(row.size_bytes / 1024))} KB` },
          ]}
          rows={activity.data?.entries ?? []}
          rowKey={(row) => row.path}
          empty={<div className="p-4"><Empty title="No artifacts found" detail="The pipeline has not written anything yet." /></div>}
        />
      </Panel>

      <p className="mt-4 text-[.6875rem] leading-relaxed text-[var(--ink-faint)]">
        Read-only view. The dashboard never calls <code className="mono">GET /predictions</code>; that legacy route re-runs inference and rewrites the
        serving CSV.
      </p>
    </>
  );
}
