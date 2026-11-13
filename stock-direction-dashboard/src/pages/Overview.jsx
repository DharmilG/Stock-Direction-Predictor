import { AlertTriangle, CheckCircle2 } from 'lucide-react';
import { api } from '../lib/api';
import { POLL_INTERVALS } from '../lib/constants';
import { formatDate, formatInstant, humanAge, humanDuration } from '../lib/dates';
import { pct, int, signedPct, num } from '../lib/format';
import { useResource } from '../hooks/useResource';
import { useDashboard } from '../state/DashboardProvider';
import { modelKey, useModel } from '../state/ModelProvider';
import { DistributionBars } from '../components/charts';
import { SignalBadge } from '../components/Signal';
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
  Warnings,
} from '../components/ui';

function StatusBadge({ status }) {
  const tone = status === 'fresh' ? 'badge-ghost' : status === 'stale' ? 'badge-alert' : 'badge-dashed';
  return <Badge tone={tone}>{status}</Badge>;
}

/**
 * One registered model, at a glance. Each card reads its own artefacts, so an
 * unavailable model says why instead of borrowing its neighbour's numbers.
 */
function ModelCard({ model }) {
  const metrics = useResource((options) => api.metrics(model.id, options), {
    key: modelKey(model.id, 'metrics'),
    pollMs: POLL_INTERVALS.artifacts,
    enabled: model.ready,
  });
  const m = metrics.data?.metrics ?? {};

  return (
    <Panel
      title={model.label}
      eyebrow={model.servable ? 'Serving' : 'Research'}
      actions={
        <Badge
          tone={model.servable ? 'badge-ghost' : 'badge-dashed'}
          title={model.servable ? 'This is the booster the pipeline serves.' : 'Measured here next to the incumbent; nothing on this dashboard promotes it.'}
        >
          {model.servable ? 'served' : 'not served'}
        </Badge>
      }
      info={{
        what: 'The headline holdout scores for this model, as its own evaluation wrote them.',
        reads: 'Accuracy is thresholded three-class accuracy; balanced accuracy gives each class equal weight, which is the honest number on an imbalanced target.',
        ideal: 'Accuracy above the majority baseline, and a balanced accuracy that is not carried by one class.',
        when: 'A missing model means its artefacts are not on disk; the detail in the card names what is missing.',
      }}
    >
      {!model.ready ? (
        <Empty title="Not built" detail={`Missing ${model.missing.join(', ') || 'its core artefacts'}.`} />
      ) : (
        <KeyValue
          items={[
            { label: 'Test accuracy', value: pct(m.accuracy), mono: true },
            { label: 'Balanced accuracy', value: pct(m.balanced_accuracy), mono: true },
            { label: 'MCC', value: num(m.mcc, 4), mono: true },
            { label: 'Calibration error', value: num(m.ece, 4), mono: true },
            { label: 'Holdout rows', value: int(m.n_test), mono: true },
            { label: 'Artefacts written', value: model.latest ? formatInstant(model.latest) : '—', mono: true },
          ]}
        />
      )}
    </Panel>
  );
}

export default function Overview({ onNavigate }) {
  const { dataHealth } = useDashboard();
  const { models } = useModel();
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
  const baseline = data?.evaluation?.majority_baseline_accuracy;

  return (
    <>
      <PageHeader
        eyebrow="Overview"
        title="Platform state at a glance"
        description="Every number here is read from a file the pipeline wrote. Nothing on this page triggers inference, training or a fetch."
        actions={<SourceTag envelope={envelope} />}
      />

      {overview.error && (
        <div className="mb-5">
          <ErrorNote error={overview.error} label="Overview unavailable" onRetry={overview.refresh} />
        </div>
      )}
      {envelope?.warnings?.length > 0 && (
        <div className="mb-5">
          <Warnings warnings={envelope.warnings} />
        </div>
      )}

      <Section
        eyebrow="Headline · v1 · incumbent"
        title="Four numbers that summarise the platform"
        note="The two model numbers here are the served incumbent's. Each model's own holdout scores sit in the section below, and Compare puts the two side by side on the rows they share."
      >
        <StatGrid columns={4}>
          <Stat
            label="Expected session"
            value={formatDate(fresh?.expected_session)}
            info={{
              what: 'The trading date the pipeline should have reached by now.',
              reads: 'The next weekday after the last entry in the stored trading calendar. It is marked as an estimate when that calendar file lags.',
              ideal: 'Equal to the newest date in the feature panel, which is the column beside it.',
            }}
          />
          <Stat
            label="Feature panel"
            value={formatDate(fresh?.features_as_of)}
            info={{
              what: 'The last session for which every symbol has a complete feature row.',
              reads: 'Newest row date in data/features/*_features.parquet. The freshness section below lists every step that is not current.',
              ideal: 'Within one session of the expected session.',
              when: gap > 0 ? `It is ${gap} session(s) behind right now, so predictions are serving on stale features.` : undefined,
            }}
          />
          <Stat
            label="Test accuracy"
            value={pct(data?.evaluation?.accuracy)}
            info={{
              what: 'Share of held-out rows where the predicted class matches the label.',
              reads: 'Three-class accuracy on the purged walk-forward test window, always compared with the majority-class baseline.',
              ideal: `Above the baseline of ${pct(baseline)} by more than two standard errors. Accuracy alone flatters an imbalanced target, so read it next to balanced recall.`,
              when: 'At or below the baseline, the model is no better than always naming the most common class.',
            }}
          />
          <Stat
            label="Backtest total return"
            value={signedPct(data?.backtest?.total_return)}
            info={{
              what: 'Cumulative return of the paper-trading segment, after trading costs.',
              reads: 'Sum of per-trade returns net of 31.5 bps per delivery trade, compounded over the backtest window.',
              ideal: 'Positive, climbing steadily, with a drawdown that stays shallow relative to the gain.',
              when: 'A flat or falling curve over many trades means the edge does not survive costs, not that the window was short.',
            }}
          />
        </StatGrid>
      </Section>

      <Section
        eyebrow="Models"
        title="What is on the platform, and which one serves"
        note="Two boosters live side by side. v1 is the served incumbent; v2 is a research twin with its own artefacts, measured against v1 and never promoted by this dashboard. The selector in the header switches every model-scoped page between them."
        actions={
          <Button size="sm" variant="quiet" onClick={() => onNavigate('compare')}>
            Compare both
          </Button>
        }
      >
        <div className="grid gap-6 lg:grid-cols-2">
          {models.map((model) => (
            <ModelCard key={model.id} model={model} />
          ))}
        </div>
      </Section>

      <Section eyebrow="Freshness" title="Where the pipeline stands">
        <div className="grid gap-6 lg:grid-cols-3">
          <Panel
            className="lg:col-span-2"
            title={behind.length ? `${behind.length} step(s) behind the expected session` : 'Every pipeline step is current'}
            info={{
              what: 'Which pipeline steps have not yet written an artifact for the expected session.',
              reads: 'Each step is compared with the calendar date it should have produced. The Fix column is the command that closes it.',
              ideal: 'The table is empty. A single stale step usually clears on the next refresh.',
            }}
            actions={
              <Button variant="quiet" size="sm" onClick={() => onNavigate('data')}>
                Open data health
              </Button>
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
                  { key: 'command', header: 'Fix', render: (row) => <span className="mono text-[.8125rem] text-[var(--ink-3)]">{row.command ?? '—'}</span> },
                ]}
                rows={behind}
                rowKey={(row) => row.step}
              />
            ) : (
              <div className="p-4">
                <Empty title="No stale steps" detail="The feature panel and predictions both reach the expected session." />
              </div>
            )}
            {(refreshRun?.running || actionable.length > 0) && (
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
            )}
          </Panel>

          <Panel
            title="Latest prediction mix"
            info={{
              what: 'How the freshest serving rows split across DOWN, FLAT and UP.',
              reads: 'Class shares over results/latest_predictions.csv for the single date shown at the bottom of the panel.',
              ideal: 'Close to the training label mix of roughly 48% DOWN, 5% FLAT and 47% UP. A one-sided mix means the model has collapsed onto a class.',
              when: 'A FLAT bar that dwarfs the others means the confidence filter is eating the signal.',
            }}
          >
            {overview.loading && !data ? (
              <Loading rows={4} />
            ) : data?.predictions?.rows ? (
              <>
                <div className="mb-4 flex items-center justify-between">
                  <SignalBadge signal="UP" />
                  <span className="mono text-[.8125rem] text-[var(--ink-3)]">mean confidence {pct(data.predictions.mean_confidence)}</span>
                </div>
                <DistributionBars distribution={data.predictions.distribution} />
                <p className="muted mt-4 text-[.8125rem]">
                  {int(data.predictions.rows)} rows dated {formatDate(data.predictions.as_of)}.
                </p>
              </>
            ) : (
              <Empty title="No predictions yet" detail="results/latest_predictions.csv is missing or empty." />
            )}
          </Panel>
        </div>
      </Section>

      <Section
        eyebrow="Model · v1 · incumbent"
        title="What is on disk"
        note="The checkpoint and the training panel belong to the booster the pipeline serves. v2 keeps its own copies under v2/, which its stage commands write."
      >
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel
            title="Training snapshot"
            info={{
              what: 'Progress of the LightGBM booster currently written to disk.',
              reads: 'Rounds completed against the configured round count, read from the checkpoint file and its sidecar metadata.',
              ideal: 'Round count equals the configured total and the checkpoint timestamp is recent.',
              when: 'Stalled mid-run, or a checkpoint far older than the configured rounds, means the last run was interrupted.',
            }}
          >
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
            <div className="mt-5">
              <KeyValue
                columns={1}
                items={[
                  { label: 'Trained at', value: data?.model?.trained_at ? formatInstant(data.model.trained_at) : '—' },
                  { label: 'Feature count', value: data?.model?.feature_count ?? '—', mono: true },
                  { label: 'Temperature', value: data?.model?.temperature?.toFixed?.(4) ?? '—', mono: true },
                ]}
              />
            </div>
          </Panel>

          <Panel
            title="Coverage"
            info={{
              what: 'The size and span of the dataset the model was trained on.',
              reads: 'Row counts and first/last dates over data/processed, plus the label settings used to build them.',
              ideal: 'Feature span covers the full configured history and the dataset row count is a large fraction of the feature rows.',
              when: 'A dataset far smaller than the feature panel means label construction dropped rows, usually at the window edges.',
            }}
          >
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
        </div>
      </Section>

      <Section eyebrow="Health" title="Checks, drift and artifacts">
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel
            title="Checks and drift"
            info={{
              what: 'Live health checks and drift counters written by the monitoring step.',
              reads: 'The first five checks from the data-health artifact, then the drift summary counters underneath.',
              ideal: 'Every check passes and drift alerts stay at zero, with a missing rate close to zero.',
              when: 'A failing check names the artifact it could not read; the panel title on the data page gives the command that rebuilds it.',
            }}
            actions={
              <Button variant="quiet" size="sm" onClick={() => onNavigate('data')}>
                Details
              </Button>
            }
          >
            <ul className="space-y-3">
              {(dataHealth?.checks ?? []).slice(0, 5).map((check) => (
                <li key={check.id} className="flex items-start gap-2.5 border-t border-[var(--line)] pt-3 first:border-t-0 first:pt-0">
                  {check.ok ? (
                    <CheckCircle2 size={15} className="mt-0.5 shrink-0 text-[var(--ink-3)]" strokeWidth={2.25} />
                  ) : (
                    <AlertTriangle size={15} className="mt-0.5 shrink-0 text-[var(--ink)]" strokeWidth={2.5} />
                  )}
                  <div className="min-w-0">
                    <p className="mono text-[.8125rem] font-semibold text-[var(--ink-2)]">{check.id}</p>
                    <p className="muted text-[.875rem]">{check.detail}</p>
                  </div>
                </li>
              ))}
              {!dataHealth?.checks?.length && <Empty title="No checks recorded" />}
            </ul>
            <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-[var(--line)] pt-4">
              <Badge tone="badge-ghost">drift alerts {data?.drift?.alerts ?? 0}</Badge>
              <Badge tone="badge-ghost">missing rate {pct(data?.drift?.missing_rate)}</Badge>
              <Badge tone="badge-ghost">latest {formatDate(data?.drift?.latest_data, { withYear: false })}</Badge>
            </div>
          </Panel>

          <Panel
            title="Recently written files"
            info={{
              what: 'Artifacts the pipeline touched most recently, newest first.',
              reads: 'File modification times under artifacts/, results/ and data/processed, with age and size.',
              ideal: 'The newest entries line up with the expected session and with the steps that just ran.',
              when: 'Nothing written for days while the connection chip says stale means a pipeline run never started.',
            }}
            flush
          >
            <Table
              columns={[
                { key: 'path', header: 'File', strong: true, mono: true },
                { key: 'modified_at', header: 'Modified (IST)', render: (row) => formatInstant(row.modified_at) },
                { key: 'age_seconds', header: 'Age', align: 'right', render: (row) => humanDuration(row.age_seconds) },
                { key: 'size_bytes', header: 'Size', align: 'right', render: (row) => `${Math.max(1, Math.round(row.size_bytes / 1024))} KB` },
              ]}
              rows={activity.data?.entries ?? []}
              // The API returns basenames only, so artifacts/state/q0_d1.json and
              // results/q0_d1.json collide on `path`. Pair it with the index.
              rowKey={(row, index) => `${row.path}-${index}`}
              empty={<div className="p-4"><Empty title="No artifacts found" detail="The pipeline has not written anything yet." /></div>}
            />
          </Panel>
        </div>
      </Section>
    </>
  );
}
