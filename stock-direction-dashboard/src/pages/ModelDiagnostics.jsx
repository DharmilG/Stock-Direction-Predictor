import { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, AlertTriangle, CheckCircle2, MinusCircle, RefreshCw, X } from 'lucide-react';
import { api } from '../lib/api';
import { POLL_INTERVALS, VERDICT_META } from '../lib/constants';
import { formatAny, formatDate } from '../lib/dates';
import { int, num, pct, signed } from '../lib/format';
import { useResource } from '../hooks/useResource';
import { modelKey, useModel } from '../state/ModelProvider';
import {
  BehaviourPlate,
  ConfusionMatrix,
  ConvictionHistogram,
  GroupedProfileBars,
  LineSeriesChart,
  RollingLine,
  SignedDriverBars,
  Waterfall,
} from '../components/charts';
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
  Segmented,
  SourceTag,
  Stat,
  Table,
  Warnings,
} from '../components/ui';

const TABS = [
  { value: 'behaviour', label: 'Behaviour' },
  { value: 'diagnosis', label: 'Diagnosis' },
  { value: 'cases', label: 'Cases' },
  { value: 'training', label: 'Training' },
  { value: 'live', label: 'Live' },
];

const SPLITS = [
  { value: 'train', label: 'Train' },
  { value: 'calibration', label: 'Calibration' },
  { value: 'test', label: 'Test' },
];

const CLASS_LABELS = ['DOWN', 'FLAT', 'UP'];
const CLASS_OPTIONS = [{ value: 'all', label: 'Any' }, ...CLASS_LABELS.map((label) => ({ value: label, label }))];
const CORRECT_OPTIONS = [
  { value: 'all', label: 'Any' },
  { value: 'right', label: 'Correct' },
  { value: 'wrong', label: 'Wrong' },
];

/* ------------------------------------------------------------------ */
/* Behaviour                                                           */
/* ------------------------------------------------------------------ */
function SweepChart({ rows, tau }) {
  if (!rows?.length) return <Empty title="No threshold sweep" detail="results/model_report.json has no τ sweep rows." />;
  const data = rows.map((row) => ({
    tau_flat: Number(row.tau_flat),
    coverage: row.coverage_direction,
    accuracy: row.accuracy,
    DOWN: row.recall?.DOWN,
    FLAT: row.recall?.FLAT,
    UP: row.recall?.UP,
  }));
  return (
    <LineSeriesChart
      data={data}
      xKey="tau_flat"
      height={220}
      yDomain={[0, 1]}
      yTickFormatter={(value) => pct(value, 0)}
      xTickFormatter={(value) => num(value, 2)}
      reference={tau ? { value: Number(tau), label: 'τ in use' } : undefined}
      series={[
        { key: 'accuracy', label: 'accuracy' },
        { key: 'coverage', label: 'directional coverage' },
        { key: 'UP', label: 'recall UP', dash: '2 3' },
      ]}
    />
  );
}

function BehaviourTab({ report, training }) {
  const [split, setSplit] = useState('test');
  const [compare, setCompare] = useState(false);
  const plate = report.splits?.[split];
  const sweep = useMemo(() => (report.thresholds ?? []).filter((row) => row.split === split), [report.thresholds, split]);

  return (
    <>
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <Segmented label="Split" options={SPLITS} value={split} onChange={setSplit} />
          {training && (
            <span className="text-[.8125rem] text-[var(--ink-3)]">last completed run · {formatAny(report.generated_at)}</span>
          )}
        </div>
        <Button size="sm" variant={compare ? 'primary' : 'secondary'} onClick={() => setCompare((value) => !value)}>
          {compare ? 'Comparing splits' : 'Compare splits'}
        </Button>
      </div>

      {compare ? (
        <div className="grid gap-6 xl:grid-cols-3">
          {SPLITS.map((option) => (
            <BehaviourPlate key={option.value} plate={report.splits?.[option.value]} drivers={report.drivers} compact />
          ))}
        </div>
      ) : (
        <BehaviourPlate plate={plate} drivers={report.drivers} />
      )}

      {!compare && plate && (
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Panel
            title="Confusion matrix"
            info={{
              what: 'Where the predicted classes land against the realised labels on this split.',
              reads: `Rows are truth, columns the prediction, shade the count. Accuracy ${pct(plate.confusion?.accuracy)}, balanced ${pct(plate.confusion?.balanced_accuracy)}, MCC ${num(plate.confusion?.mcc, 4)}.`,
              ideal: 'A dark diagonal. Because FLAT is rare, a wide FLAT column is the normal failure shape, not a sign of caution.',
            }}
          >
            <ConfusionMatrix matrix={plate.confusion?.counts} labels={CLASS_LABELS} />
          </Panel>
          <Panel
            title="How close are the top two classes?"
            info={{
              what: 'The gap between the winning probability and the runner-up, bin by bin.',
              reads: `Share of rows in each margin bucket. Here the median margin is ${num(plate.margins?.median, 4)} and ${pct(plate.margins?.share_below_0_10, 1)} of calls separate the top two classes by less than 0.10.`,
              ideal: 'Most mass in the wide buckets. A pile-up near zero means the decision flips on rounding, which will not replicate out of sample.',
            }}
          >
            <div className="space-y-3">
              {(plate.margins?.bins ?? []).map((bin) => (
                <div key={`${bin.lo}-${bin.hi}`} className="grid grid-cols-[4rem_minmax(0,1fr)_3.5rem] items-center gap-3">
                  <span className="mono text-[.8125rem] text-[var(--ink-3)]">
                    {bin.lo}–{bin.hi}
                  </span>
                  <div className="h-2 w-full rounded-sm bg-[var(--surface-sunken)]">
                    <div className="h-full rounded-sm bg-[var(--ink)]" style={{ width: `${(bin.share ?? 0) * 100}%` }} />
                  </div>
                  <span className="mono text-right text-[.8125rem] text-[var(--ink-3)]">{pct(bin.share, 1)}</span>
                </div>
              ))}
            </div>
          </Panel>
        </div>
      )}

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Panel
          title={`τ sweep · ${split}`}
          info={{
            what: 'What the current decision threshold costs, by re-running the rule at every other threshold.',
            reads: 'Each point re-applies the existing rule at a different τ: accuracy, directional coverage and UP recall plotted against τ. Nothing here changes the threshold in force.',
            ideal: 'A broad plateau where accuracy stays high as τ moves. A single sharp peak means the threshold was fitted to noise.',
            when: `The marker sits at τ = ${num(report.meta?.tau_flat, 2)}. If that marker is on a cliff edge, the live setting is fragile.`,
          }}
        >
          <SweepChart rows={sweep} tau={report.meta?.tau_flat} />
        </Panel>
        <Panel
          title="Rolling predicted share and accuracy"
          info={{
            what: 'How the predicted class mix and the hit rate drifted across the split.',
            reads: 'A rolling window over the scored dates: share of DOWN calls, share of UP calls and rolling accuracy, with the actual DOWN share as a dashed reference.',
            ideal: 'Predicted shares tracking the actual dashed line, and rolling accuracy that does not decay towards the end of the window.',
            when: 'A predicted share far above the actual share is the model producing a class that is not there.',
          }}
        >
          {plate?.rolling?.length ? (
            <RollingLine
              data={plate.rolling}
              series={[
                { key: 'roll_pred_down', label: 'rolling share DOWN' },
                { key: 'roll_pred_up', label: 'rolling share UP', dash: '6 3' },
                { key: 'roll_accuracy', label: 'rolling accuracy', dash: '2 3' },
              ]}
              reference={{ value: plate.distribution?.actual?.DOWN, label: 'actual DOWN share' }}
              height={220}
            />
          ) : (
            <Empty title="No rolling series" detail="The split has no scored dates." />
          )}
        </Panel>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Diagnosis                                                           */
/* ------------------------------------------------------------------ */
function VerdictBadge({ verdict }) {
  const meta = VERDICT_META[verdict] ?? VERDICT_META.INCONCLUSIVE;
  return (
    <Badge tone={meta.tone} title={meta.label}>
      <span className="text-[.9375rem] leading-none">{meta.glyph}</span>
      {meta.label}
    </Badge>
  );
}

function hypothesisChart(hypothesis, report) {
  const plate = report.splits?.[hypothesis.split] ?? report.splits?.test;
  if (!plate) return null;
  const dominant = plate.distribution?.dominant;
  if (hypothesis.chart === 'profile') return <GroupedProfileBars profile={plate.profile} />;
  if (hypothesis.chart === 'conviction') return <ConvictionHistogram conviction={plate.conviction} />;
  if (hypothesis.chart === 'distribution') return <PairedShareSummary plate={plate} />;
  if (hypothesis.chart === 'sweep') {
    const rows = (report.thresholds ?? []).filter((row) => row.split === hypothesis.split);
    return <SweepChart rows={rows} tau={report.meta?.tau_flat} />;
  }
  if (hypothesis.chart === 'split_table') {
    return (
      <Table
        dense
        columns={[
          { key: 'label', header: 'Split', strong: true },
          { key: 'accuracy', header: 'Accuracy', align: 'right', render: (row) => pct(row.accuracy) },
          { key: 'balanced', header: 'Balanced', align: 'right', render: (row) => pct(row.balanced) },
          { key: 'mcc', header: 'MCC', align: 'right', render: (row) => num(row.mcc, 4) },
        ]}
        rows={SPLITS.map((option) => {
          const item = report.splits?.[option.value];
          return {
            label: option.label,
            accuracy: item?.confusion?.accuracy,
            balanced: item?.confusion?.balanced_accuracy,
            mcc: item?.confusion?.mcc,
          };
        })}
        rowKey={(row) => row.label}
      />
    );
  }
  if (hypothesis.chart === 'importance') {
    return (
      <ul className="space-y-1">
        {(report.importance ?? []).slice(0, 6).map((row) => (
          <li key={row.feature} className="flex items-center justify-between gap-2 text-[.875rem]">
            <span className="mono truncate text-[var(--ink-2)]">{row.feature}</span>
            <span className="mono shrink-0 text-[var(--ink-3)]">PSI {num(row.psi, 3)}</span>
          </li>
        ))}
      </ul>
    );
  }
  const rows = (report.drivers ?? []).filter((row) => row.class === dominant);
  return <SignedDriverBars rows={rows} label={`push toward ${dominant ?? '—'}`} />;
}

function PairedShareSummary({ plate }) {
  return (
    <div className="space-y-1">
      {CLASS_LABELS.map((label) => (
        <div key={label} className="flex items-center justify-between gap-2 text-[.8125rem]">
          <span className="mono text-[var(--ink-2)]">{label}</span>
          <span className="mono text-[var(--ink-3)]">
            {pct(plate.distribution?.predicted?.[label], 1)} produced vs {pct(plate.distribution?.actual?.[label], 1)} present
          </span>
        </div>
      ))}
    </div>
  );
}

function DiagnosisTab({ report }) {
  const hypotheses = report.hypotheses ?? [];
  if (!hypotheses.length) return <Empty title="No hypotheses recorded" detail="Run the model-report stage to derive them." />;
  const counts = hypotheses.reduce((acc, item) => ({ ...acc, [item.verdict]: (acc[item.verdict] ?? 0) + 1 }), {});
  return (
    <>
      <div className="mb-6 flex flex-wrap items-center gap-2.5">
        {Object.entries(VERDICT_META).map(([verdict, meta]) => (
          <span key={verdict} className="badge badge-ghost">
            {meta.glyph} {meta.label} {counts[verdict] ?? 0}
          </span>
        ))}
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        {hypotheses.map((hypothesis) => (
          <Panel
            key={hypothesis.id}
            eyebrow={`Measured on ${hypothesis.split}`}
            title={hypothesis.title}
            info={{
              what: 'One candidate explanation for a weakness, with the numbers that support or reject it.',
              reads: 'Each verdict is decided by a stated threshold on numbers in this report, printed next to the value it was compared against.',
              ideal: 'A rejected verdict is a useful result: it stops the team chasing a cause that is not there.',
              when: 'A supporting verdict with a value close to its threshold is the one to re-check on another split before acting on it.',
            }}
            actions={<VerdictBadge verdict={hypothesis.verdict} />}
          >
            <p className="muted mb-4">{hypothesis.claim}</p>
            <div className="mb-4 grid gap-2">
              {(hypothesis.evidence ?? []).map((item) => (
                <div key={`${item.label}`} className="flex flex-wrap items-baseline justify-between gap-2 border-t border-[var(--line)] pt-2">
                  <span className="text-[.875rem] text-[var(--ink-2)]">{item.label}</span>
                  <span className="mono text-[.875rem] font-semibold text-[var(--ink)]">
                    {item.value}
                    {item.threshold ? <span className="ml-2 font-normal text-[var(--ink-faint)]">threshold {item.threshold}</span> : null}
                  </span>
                </div>
              ))}
            </div>
            {hypothesisChart(hypothesis, report)}
          </Panel>
        ))}
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Cases                                                               */
/* ------------------------------------------------------------------ */
function Drawer({ title, onClose, children }) {
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/25" role="dialog" aria-modal="true" aria-label={title}>
      <div className="flex h-full w-full max-w-2xl flex-col overflow-hidden border-l border-[var(--line)] bg-white">
        <header className="flex items-start justify-between gap-3 border-b border-[var(--line)] px-4 py-3">
          <div className="min-w-0">
            <p className="eyebrow">Explanation</p>
            <h2 className="section-title mt-0.5">{title}</h2>
          </div>
          <Button size="sm" variant="quiet" icon={X} onClick={onClose}>
            Close
          </Button>
        </header>
        <div className="flex-1 overflow-auto px-4 py-3 scroll-thin">{children}</div>
      </div>
    </div>
  );
}

function ExplanationBody({ symbol, split }) {
  const { modelId } = useModel();
  const resource = useResource((options) => api.explain(modelId, symbol, { split }, options), {
    key: modelKey(modelId, `explain-${symbol}-${split}`),
  });
  const data = resource.data;
  if (resource.error) return <ErrorNote error={resource.error} label="No explanation for this row" onRetry={resource.refresh} />;
  if (resource.loading || !data) return <Loading rows={5} />;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="badge-ghost">{data.source}</Badge>
        <Badge tone="badge-ghost">{formatDate(data.date)}</Badge>
        <Badge tone={data.prediction === 'UP' ? 'badge-solid' : 'badge-mid'}>predicted {data.prediction}</Badge>
        {data.actual && <Badge tone="badge-dashed">actual {data.actual}</Badge>}
        {data.stale && <Badge tone="badge-alert">stale artifact</Badge>}
      </div>
      <KeyValue
        columns={3}
        items={[
          { label: 'p(DOWN)', value: num(data.probabilities?.DOWN, 4), mono: true },
          { label: 'p(FLAT)', value: num(data.probabilities?.FLAT, 4), mono: true },
          { label: 'p(UP)', value: num(data.probabilities?.UP, 4), mono: true },
          { label: 'Decision margin', value: num(data.margin, 4), mono: true },
          { label: 'Method', value: data.method, mono: true },
          { label: 'Explained split', value: data.split, mono: true },
        ]}
      />
      {data.flip?.lever && (
        <Callout tone="info" title={`Distance to flip to ${data.flip.to}`}>
          Raw-score gap {num(data.flip.raw_gap, 3)}. The strongest {data.flip.to} lever is <span className="mono">{data.flip.lever.feature}</span> at{' '}
          {num(data.flip.lever.contribution, 3)}; it would need to grow {num(data.flip.lever.multiple_needed, 2)}×. {data.flip.note}
        </Callout>
      )}
      {Object.entries(data.classes ?? {}).map(([name, entry]) => (
        <Panel key={name} eyebrow={`Class ${name}`} title={`Base ${num(entry.base_value, 3)} → raw ${num(entry.raw_score, 3)}`}>
          <Waterfall base={entry.base_value} steps={entry.steps} closing={entry.raw_score} />
        </Panel>
      ))}
      <Callout tone="info">
        Contributions are Tree SHAP values sampled from a stored explanation file, not recomputed per request. Base + contributions reconstructs the raw
        class score; it is an attribution, not a causal claim.
      </Callout>
    </div>
  );
}

function CasesTab() {
  const { modelId } = useModel();
  const [split, setSplit] = useState('test');
  const [predicted, setPredicted] = useState('all');
  const [actual, setActual] = useState('all');
  const [correct, setCorrect] = useState('all');
  const [symbol, setSymbol] = useState('');
  const [offset, setOffset] = useState(0);
  const [active, setActive] = useState(null);
  const limit = 25;

  const filters = useMemo(() => ({ split, predicted, actual, correct, symbol, limit, offset }), [split, predicted, actual, correct, symbol, offset]);
  const resource = useResource((options) => api.modelCases(modelId, filters, options), {
    key: modelKey(modelId, `cases-${split}-${predicted}-${actual}-${correct}-${symbol}-${offset}`),
  });
  const data = resource.data;
  const reset = useCallback((setter) => (value) => { setter(value); setOffset(0); }, []);

  return (
    <>
      <Panel
        title="Scored rows"
        info={{
          what: 'Every row the report scored, filterable, with the outcome attached.',
          reads: 'Filters are applied server-side: split, predicted class, actual class, right or wrong, and symbol.',
          ideal: 'You reach a specific slice quickly and the row count above the table matches what you expect.',
        }}
        actions={<Badge tone="badge-ghost">{data ? `${int(data.total)} rows` : '—'}</Badge>}
      >
        <div className="flex flex-wrap items-end gap-4">
          <label className="block">
            <span className="eyebrow">Split</span>
            <div className="mt-2">
              <Segmented label="Split" options={[...SPLITS, { value: 'all', label: 'All' }]} value={split} onChange={reset(setSplit)} />
            </div>
          </label>
          <label className="block">
            <span className="eyebrow">Predicted</span>
            <div className="mt-2">
              <Segmented label="Predicted" options={CLASS_OPTIONS} value={predicted} onChange={reset(setPredicted)} />
            </div>
          </label>
          <label className="block">
            <span className="eyebrow">Actual</span>
            <div className="mt-2">
              <Segmented label="Actual" options={CLASS_OPTIONS} value={actual} onChange={reset(setActual)} />
            </div>
          </label>
          <label className="block">
            <span className="eyebrow">Outcome</span>
            <div className="mt-2">
              <Segmented label="Outcome" options={CORRECT_OPTIONS} value={correct} onChange={reset(setCorrect)} />
            </div>
          </label>
          <label className="block">
            <span className="eyebrow">Symbol</span>
            <input
              className="field mt-2 w-48"
              placeholder="ADANIPORTS.NS"
              value={symbol}
              onChange={(event) => reset(setSymbol)(event.target.value.toUpperCase())}
            />
          </label>
        </div>
      </Panel>

      <Panel
        className="mt-6"
        flush
        bodyClass="px-0 py-0"
        title="Click a row for its waterfall"
        info={{
          what: 'The individual rows behind the aggregate numbers, one line per symbol and date.',
          reads: 'max p is the winning probability, margin is the gap to the runner-up, realised is the triple-barrier return in percent.',
          ideal: 'A spread of margins. A column of near-zero margins means the calls sit on the decision boundary.',
        }}
      >
        {resource.error ? (
          <div className="p-4">
            <ErrorNote error={resource.error} label="Case table unavailable" onRetry={resource.refresh} />
          </div>
        ) : resource.loading && !data ? (
          <div className="p-4">
            <Loading rows={6} />
          </div>
        ) : (
          <>
            <Table
              dense
              columns={[
                { key: 'symbol', header: 'Symbol', strong: true, mono: true },
                { key: 'date', header: 'Date', mono: true, render: (row) => formatDate(row.date) },
                { key: 'actual', header: 'Actual' },
                { key: 'predicted', header: 'Predicted' },
                { key: 'max_p', header: 'max p', align: 'right', mono: true, render: (row) => num(row.max_p, 3) },
                { key: 'margin', header: 'margin', align: 'right', mono: true, render: (row) => num(row.margin, 3) },
                {
                  key: 'realized_return',
                  header: 'realised',
                  align: 'right',
                  mono: true,
                  render: (row) => (row.realized_return == null ? '—' : `${signed(row.realized_return * 100, 2)}%`),
                },
                {
                  key: 'correct',
                  header: '',
                  render: (row) =>
                    row.correct ? (
                      <CheckCircle2 size={13} className="text-[var(--ink-3)]" aria-label="correct" />
                    ) : (
                      <MinusCircle size={13} className="text-[var(--ink-faint)]" aria-label="wrong" />
                    ),
                },
              ]}
              rows={data?.rows ?? []}
              rowKey={(row) => `${row.symbol}-${row.date}-${row.split}`}
              onRowClick={setActive}
              empty={<Empty title="No rows match those filters" detail="Widen the filters; the table only shows rows the report scored." />}
            />
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[var(--line)] px-4 py-3">
              <span className="text-[.8125rem] text-[var(--ink-3)]">
                {data?.total ? `Showing ${offset + 1}–${Math.min(offset + limit, data.total)} of ${int(data.total)}` : 'No rows'}
              </span>
              <div className="flex gap-2">
                <Button size="sm" variant="quiet" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>
                  Previous
                </Button>
                <Button
                  size="sm"
                  variant="quiet"
                  disabled={(data?.total ?? 0) <= offset + limit}
                  onClick={() => setOffset(offset + limit)}
                >
                  Next
                </Button>
              </div>
            </div>
          </>
        )}
      </Panel>

      {active && (
        <Drawer title={`${active.symbol} · ${active.date}`} onClose={() => setActive(null)}>
          <ExplanationBody symbol={active.symbol} split={active.split} />
        </Drawer>
      )}
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Training                                                            */
/* ------------------------------------------------------------------ */
function SplitTable({ report }) {
  return (
    <Table
      dense
      columns={[
        { key: 'label', header: 'Split', strong: true },
        { key: 'rows', header: 'Rows', align: 'right', mono: true, render: (row) => int(row.rows) },
        { key: 'accuracy', header: 'Accuracy', align: 'right', mono: true, render: (row) => pct(row.accuracy) },
        { key: 'balanced', header: 'Balanced', align: 'right', mono: true, render: (row) => pct(row.balanced) },
        { key: 'mcc', header: 'MCC', align: 'right', mono: true, render: (row) => num(row.mcc, 4) },
        { key: 'ece', header: 'ECE', align: 'right', mono: true, render: (row) => num(row.ece, 4) },
        { key: 'mode', header: 'Mode', render: (row) => <span className="mono text-[.8125rem]">{row.mode}</span> },
      ]}
      rows={SPLITS.map((option) => {
        const plate = report.splits?.[option.value];
        return {
          label: option.label,
          rows: plate?.rows,
          accuracy: plate?.confusion?.accuracy,
          balanced: plate?.confusion?.balanced_accuracy,
          mcc: plate?.confusion?.mcc,
          ece: plate?.rule?.ece,
          mode: plate?.mode?.mode,
        };
      })}
      rowKey={(row) => row.label}
    />
  );
}

function FoldRecallHeatmap({ folds }) {
  const shown = (folds ?? []).slice(-40);
  if (!shown.length) return <Empty title="No fold diagnostics" detail="checkpoints/trials/fold_diagnostics.jsonl is empty." />;
  const cell = (value) => ({
    background: `rgba(17,17,16,${(0.06 + Math.max(0, Math.min(1, value ?? 0)) * 0.86).toFixed(3)})`,
    color: (value ?? 0) > 0.45 ? '#fff' : 'var(--ink)',
  });
  return (
    <div>
      <div className="flex items-end gap-0.5">
        {shown.map((row, index) => (
          <div key={`${row.trial}-${row.fold}-${index}`} className="flex-1 space-y-0.5" title={`trial ${row.trial} fold ${row.fold} · score ${num(row.score, 3)}`}>
            {['down', 'flat', 'up'].map((name) => (
              <div key={name} className="h-2.5 rounded-[1px]" style={cell(row.recall?.[name])} />
            ))}
          </div>
        ))}
      </div>
      <div className="mt-2.5 flex flex-wrap gap-x-4 gap-y-1 text-[.8125rem] text-[var(--ink-3)]">
        <span>rows: DOWN / FLAT / UP recall per fold (oldest → newest, last {shown.length})</span>
        <span>darker = higher recall</span>
      </div>
    </div>
  );
}

function TrainingTab({ report, live }) {
  const meta = report.meta ?? {};
  const curve = (live?.curve ?? []).filter((point) => point.split === 'train' || point.split === 'calibration');
  const curveByIteration = useMemo(() => {
    const map = new Map();
    curve.forEach((point) => {
      const row = map.get(point.iteration) ?? { iteration: point.iteration };
      row[point.split] = point.multi_logloss;
      row[`${point.split}_accuracy`] = point.accuracy;
      map.set(point.iteration, row);
    });
    return [...map.values()].sort((a, b) => a.iteration - b.iteration);
  }, [curve]);

  return (
    <>
      <div className="grid gap-6 lg:grid-cols-3">
        <Panel
          className="lg:col-span-2"
          title="What is being explained"
          info={{
            what: 'The exact model behind every number on this page.',
            reads: 'Model type, feature count, row counts, the dataset signature and the fitted temperature and flat threshold.',
            ideal: 'The dataset signature and parameter hash match the run you believe produced these artifacts; a mismatch means you are reading a stale report.',
          }}
        >
          <KeyValue
            columns={3}
            items={[
              { label: 'Model', value: meta.model_type, mono: true },
              { label: 'Features', value: int(meta.feature_count), mono: true },
              { label: 'Train rows', value: int(meta.rows?.train), mono: true },
              { label: 'Dataset signature', value: String(meta.dataset_signature ?? '').slice(0, 16), mono: true },
              { label: 'Params hash', value: String(meta.best_params_hash ?? '').slice(0, 16), mono: true },
              { label: 'Temperature', value: num(meta.temperature, 4), mono: true },
              { label: 'τ flat', value: num(meta.tau_flat, 2), mono: true },
              { label: 'Train window', value: `${formatDate(meta.train_start)} → ${formatDate(meta.calibration_start)}`, mono: true },
              { label: 'Test window', value: `${formatDate(meta.test_start)} → ${formatDate(meta.test_end)}`, mono: true },
            ]}
          />
        </Panel>
        <div className="space-y-6">
          <Stat
            label="Train accuracy"
            value={pct(meta.trained_train_accuracy)}
            info={{
              what: 'Accuracy when the stored booster re-scores its own training split.',
              reads: 'The train split scored again with this exact booster, not the number training printed while fitting.',
              ideal: 'Close to the test accuracy. A wide gap is memorisation.',
            }}
          />
          <Stat
            label="Test accuracy"
            value={pct(meta.trained_test_accuracy)}
            info={{
              what: 'Accuracy when the stored booster scores the held-out test split.',
              reads: `Temperature ${num(meta.temperature, 4)} is applied before the flat threshold.`,
              ideal: 'Within a few points of train accuracy, and above the majority baseline.',
            }}
          />
        </div>
      </div>

      <Panel
        className="mt-6"
        title="Accuracy, balanced accuracy, MCC and ECE"
        info={{
          what: 'The same four scores on train, calibration and test, side by side.',
          reads: 'One row per split, identical code and identical axes. The gap between rows is the point of the table.',
          ideal: 'Train and test close together, with ECE under 0.05 on the split you will act on.',
          when: 'A large train-to-test gap is memorisation; a large calibration-to-test gap means the temperature no longer suits the data.',
        }}
      >
        <SplitTable report={report} />
      </Panel>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        {SPLITS.map((option) => (
          <Panel
            key={option.value}
            eyebrow={option.value}
            title="Confusion"
            info={{
              what: 'Where the predictions land against the realised labels on this split.',
              reads: 'Rows are truth, columns the prediction, shade the count.',
              ideal: 'A dark diagonal. Off-diagonal mass concentrated in the FLAT column means the model hedges rather than calls.',
            }}
          >
            <ConfusionMatrix matrix={report.splits?.[option.value]?.confusion?.counts} labels={CLASS_LABELS} />
          </Panel>
        ))}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Panel
          title="Recall per fold"
          info={{
            what: 'Per-class recall for each Optuna fold, oldest to newest.',
            reads: 'Three stacked cells per fold: DOWN, FLAT and UP recall. Darker is higher.',
            ideal: 'Solid colour across folds. A fold that goes pale for one class means that class was learned by luck on the others.',
          }}
        >
          <FoldRecallHeatmap folds={report.folds} />
        </Panel>
        <Panel
          title="Importance with PSI"
          info={{
            what: 'Which features carry the model, and how far each one has drifted.',
            reads: `Mean absolute SHAP over a ${int(meta.shap_sample_rows)}-row sample. PSI is the population stability index, compared against ${num(meta.psi_threshold, 2)}.`,
            ideal: 'Importance spread across explainable features and PSI below the threshold.',
            when: 'A feature flagged for drift means the values it sees in production no longer look like the ones it learned from.',
          }}
        >
          {report.importance?.length ? (
            <ul className="space-y-2">
              {report.importance.slice(0, 12).map((row) => {
                const max = Math.max(...report.importance.slice(0, 12).map((item) => item.mean_abs_shap ?? 0), 1e-9);
                return (
                  <li key={row.feature} className="grid grid-cols-[minmax(0,8rem)_minmax(0,1fr)_auto] items-center gap-3">
                    <span className="mono truncate text-[.8125rem] text-[var(--ink-2)]">{row.feature}</span>
                    <div className="h-2 w-full rounded-sm bg-[var(--surface-sunken)]">
                      <div className="h-full rounded-sm bg-[var(--ink)]" style={{ width: `${((row.mean_abs_shap ?? 0) / max) * 100}%` }} />
                    </div>
                    <span className="flex shrink-0 items-center gap-1.5">
                      <span className="mono text-[.8125rem] text-[var(--ink-3)]">{num(row.mean_abs_shap, 4)}</span>
                      {row.psi != null && (
                        <Badge tone={row.drift ? 'badge-alert' : 'badge-ghost'} title="Population stability index">
                          PSI {num(row.psi, 2)}
                        </Badge>
                      )}
                    </span>
                  </li>
                );
              })}
            </ul>
          ) : (
            <Empty title="No importance recorded" detail="results/shap_by_class.csv is missing." />
          )}
        </Panel>
      </div>

      <Panel
        className="mt-6"
        title="Recorded per-iteration loss"
        info={{
          what: 'The loss the booster recorded on each iteration while it was being fitted.',
          reads: 'Train and calibration multi-class log loss per iteration, written every 10 rounds.',
          ideal: 'Both curves falling and then flattening, with calibration loss flattening first. A calibration curve that starts rising is overfitting.',
        }}
      >
        {curveByIteration.length ? (
          <LineSeriesChart
            data={curveByIteration}
            xKey="iteration"
            height={220}
            xTickFormatter={(value) => String(value)}
            yTickFormatter={(value) => num(value, 2)}
            series={[
              { key: 'train', label: 'train loss' },
              { key: 'calibration', label: 'calibration loss', dash: '6 3' },
            ]}
          />
        ) : (
          <Empty
            title="No learning curve recorded"
            detail="checkpoints/train_curve.jsonl is written by training. It appears after the next `python -m stockml.main train` run."
          />
        )}
      </Panel>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Live                                                                */
/* ------------------------------------------------------------------ */
function LiveTab({ live }) {
  const heartbeat = live?.heartbeat;
  if (!live) return <Loading rows={4} />;
  if (!heartbeat) {
    return (
      <Empty
        title="No training run instrumented"
        detail="The heartbeat file is written while training runs. Start one with `python -m stockml.main train` and this tab updates itself; nothing here scores a half-written booster."
        action={
          <div className="mt-2 text-left">
            <Callout tone="info">
              Watching: <span className="mono">logs/train_heartbeat.json</span>, <span className="mono">checkpoints/train_curve.jsonl</span>,{' '}
              <span className="mono">checkpoints/trials/fold_diagnostics.jsonl</span>.
            </Callout>
          </div>
        }
      />
    );
  }

  const curve = live.curve ?? [];
  const byIteration = new Map();
  curve.forEach((point) => {
    const row = byIteration.get(point.iteration) ?? { iteration: point.iteration };
    row[point.split] = point.multi_logloss;
    byIteration.set(point.iteration, row);
  });
  const lossRows = [...byIteration.values()].sort((a, b) => a.iteration - b.iteration);
  const folds = (live.folds ?? []).slice(-12);

  return (
    <>
      <Panel
        title={`${heartbeat.status} · ${formatAny(heartbeat.updated_at)}`}
        info={{
          what: 'Live telemetry from the training run currently in flight.',
          reads: 'The heartbeat file is rewritten every few seconds while a run is active, and this panel polls faster while it is.',
          ideal: 'A phase, trial and fold that keep advancing, with the seconds-since-beat staying small.',
          when: 'A beat that stops moving while the badge still says TRAINING means the run died without clearing its heartbeat.',
        }}
        actions={<Badge tone={live.training ? 'badge-solid' : 'badge-alert'}>{live.training ? 'TRAINING' : 'IDLE'}</Badge>}
      >
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat
            label="Phase"
            value={heartbeat.status}
            info={{
              what: 'Which stage of the run is executing.',
              reads: `Read from the heartbeat, last written ${int(live.heartbeat_age_seconds)} s ago.`,
              ideal: 'The phase changes over minutes. A phase that has not moved in hours is stuck.',
            }}
          />
          <Stat
            label="Trial"
            value={heartbeat.trial == null ? '—' : `${int(heartbeat.trial)} / ${int(heartbeat.trial_total)}`}
            info={{
              what: 'Which Optuna hyperparameter trial is running.',
              reads: 'Completed trials against the configured total.',
              ideal: 'Advancing steadily; each trial re-runs the full purged walk-forward.',
            }}
          />
          <Stat
            label="Fold"
            value={heartbeat.fold == null ? '—' : `${int(heartbeat.fold)} / ${int(heartbeat.fold_total)}`}
            info={{
              what: 'Which purged walk-forward fold of the current trial is running.',
              reads: 'Fold index against the fold count configured for the run.',
              ideal: 'Advancing within each trial, with a fresh fold every few thousand rounds.',
            }}
          />
          <Stat
            label="Iteration"
            value={`${int(heartbeat.iteration)} / ${int(heartbeat.num_boost_round)}`}
            info={{
              what: 'Boosting rounds completed against the round budget.',
              reads: `Timed from the heartbeat; the run estimates ${int(heartbeat.eta_s)} s remaining.`,
              ideal: 'Monotonic. A value that resets without the trial changing means the run restarted.',
            }}
          />
        </div>
        <div className="mt-5">
          <ProgressBar value={heartbeat.iteration ?? 0} max={heartbeat.num_boost_round ?? 0} label="Rounds completed" detail={`Elapsed ${int(heartbeat.elapsed_s)} s · checkpoint at ${int(live.checkpoint_rounds)} rounds`} />
        </div>
      </Panel>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Panel
          title="Train vs calibration loss"
          info={{
            what: 'The loss of the run in flight, iteration by iteration.',
            reads: 'Recorded every 10 rounds. The calibration set is attached as a watched set, never as an early-stopping rule, so it cannot change the model.',
            ideal: 'Both curves falling then flattening, with calibration flattening first. A calibration curve that turns upward is overfitting in progress.',
          }}
        >
          {lossRows.length ? (
            <LineSeriesChart
              data={lossRows}
              xKey="iteration"
              height={220}
              xTickFormatter={(value) => String(value)}
              yTickFormatter={(value) => num(value, 2)}
              series={[
                { key: 'train', label: 'train loss' },
                { key: 'calibration', label: 'calibration loss', dash: '6 3' },
              ]}
            />
          ) : (
            <Empty title="No points yet" detail="The curve is written every 10 rounds." />
          )}
        </Panel>
        <Panel
          title="Score and per-class recall"
          info={{
            what: 'The last few folds scored by the run in flight.',
            reads: 'Optuna objective score plus per-class recall for each fold.',
            ideal: 'Scores that improve and per-class recall that stays balanced rather than trading DOWN for UP.',
          }}
        >
          {folds.length ? (
            <Table
              dense
              columns={[
                { key: 'trial', header: 'Trial', mono: true },
                { key: 'fold', header: 'Fold', mono: true },
                { key: 'score', header: 'Score', align: 'right', mono: true, render: (row) => num(row.score, 4) },
                { key: 'down', header: 'DOWN', align: 'right', mono: true, render: (row) => num(row.recall?.down, 3) },
                { key: 'flat', header: 'FLAT', align: 'right', mono: true, render: (row) => num(row.recall?.flat, 3) },
                { key: 'up', header: 'UP', align: 'right', mono: true, render: (row) => num(row.recall?.up, 3) },
              ]}
              rows={folds}
              rowKey={(row, index) => `${row.trial}-${row.fold}-${index}`}
            />
          ) : (
            <Empty title="No folds yet" detail="fold_diagnostics.jsonl is written per fold during the Optuna phase." />
          )}
        </Panel>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */
export default function ModelDiagnostics() {
  const { modelId, label, servable } = useModel();
  const [tab, setTab] = useState('behaviour');
  // Training writes a heartbeat every couple of seconds, so poll fast while one
  // is in flight and fall back to the artifact cadence otherwise.
  const [training, setTraining] = useState(false);
  const report = useResource((options) => api.modelReport(modelId, options), {
    key: modelKey(modelId, 'model-report'),
    pollMs: POLL_INTERVALS.artifacts,
  });
  const liveResource = useResource((options) => api.modelLive(modelId, options), {
    key: modelKey(modelId, 'model-live'),
    pollMs: training ? POLL_INTERVALS.modelLive : POLL_INTERVALS.artifacts,
  });
  const live = liveResource.data;
  useEffect(() => {
    setTraining(Boolean(live?.training));
  }, [live]);

  const reload = useCallback(() => {
    report.refresh();
    liveResource.refresh();
  }, [report, liveResource]);

  const missing = !report.loading && report.data == null;

  return (
    <>
      <PageHeader
        eyebrow={`Model behaviour · ${label}`}
        title="Why this model predicts what it predicts"
        description="One plate per split: shares, confusion, class profile, conviction and drivers. The same five panels describe a collapsed prior, a genuine tilt, or a balanced-but-uninformative model — nothing here assumes the answer in advance."
        actions={
          <>
            {training && <Badge tone="badge-solid">TRAINING</Badge>}
            <Badge tone={servable ? 'badge-ghost' : 'badge-dashed'} title={servable ? 'Served model' : 'Research model: measured here, never served.'}>
              {servable ? 'served' : 'research'}
            </Badge>
            <SourceTag envelope={report.envelope} />
            <Button size="sm" variant="quiet" icon={RefreshCw} onClick={reload}>
              Reload
            </Button>
          </>
        }
      />

      {report.error && (
        <div className="mb-5">
          <ErrorNote error={report.error} label="Behaviour report unavailable" onRetry={report.refresh} />
        </div>
      )}
      {report.envelope?.warnings?.length > 0 && (
        <div className="mb-5">
          <Warnings warnings={report.envelope.warnings} />
        </div>
      )}

      <div className="mb-6">
        <Segmented label="Section" options={TABS} value={tab} onChange={setTab} />
      </div>

      {report.data?.consistency && (
        <div className="mb-6">
          <ConsistencyBanner consistency={report.data.consistency} />
        </div>
      )}

      {missing ? (
        <Panel
          title="The behaviour report has not been built"
          info={{
            what: 'The report every tab on this page reads from.',
            reads: 'results/model_report.json. The command below builds it from the booster already on disk.',
            ideal: 'Present and dated after the last training run, so the numbers describe the model you are actually serving.',
          }}
        >
          <Empty
            title="results/model_report.json is missing"
            detail="Run the report stage to score all three splits with the existing booster. It never retrains and never changes a threshold."
            action={
              <pre className="mono mt-3 rounded-md border border-[var(--line)] bg-[var(--surface-sunken)] px-3 py-2 text-[.8125rem]">
                python -m stockml.main model-report
              </pre>
            }
          />
        </Panel>
      ) : report.loading && !report.data ? (
        <Loading rows={6} />
      ) : (
        <>
          {tab === 'behaviour' && <BehaviourTab report={report.data} training={training} />}
          {tab === 'diagnosis' && <DiagnosisTab report={report.data} />}
          {tab === 'cases' && <CasesTab />}
          {tab === 'training' && <TrainingTab report={report.data} live={live} />}
          {tab === 'live' && <LiveTab live={live} />}
        </>
      )}

      {live?.training && tab !== 'live' && (
        <div className="mt-6 flex items-center gap-2.5 rounded-lg border border-[var(--line)] bg-[var(--surface-sunken)] px-4 py-3 text-[.875rem] text-[var(--ink-2)]">
          <Activity size={15} />
          A training run is in progress. The numbers above describe the last completed run, not the booster being fitted.
        </div>
      )}
      {liveResource.error && tab === 'live' && (
        <div className="mt-6">
          <ErrorNote error={liveResource.error} label="Live telemetry unavailable" onRetry={liveResource.refresh} />
        </div>
      )}
    </>
  );
}

function ConsistencyBanner({ consistency }) {
  const diff = consistency.predictions_max_abs_diff;
  const notes = consistency.notes ?? [];
  const ok = consistency.same_window !== false && (diff == null || diff <= 1e-6);
  return (
    <div className="rounded-lg border border-[var(--line)] bg-[var(--surface-sunken)] px-4 py-3">
      <div className="flex flex-wrap items-center gap-2.5">
        {ok ? <CheckCircle2 size={15} /> : <AlertTriangle size={15} />}
        <span className="text-[.9375rem] font-semibold text-[var(--ink)]">
          {ok ? 'Report and evaluation agree' : 'Report and evaluation do not line up'}
        </span>
        <Badge tone="badge-ghost">{int(consistency.predictions_compared)} rows compared</Badge>
        <Badge tone="badge-ghost">max |Δp_down| {diff == null ? 'not compared' : num(diff, 4)}</Badge>
        <Badge tone="badge-ghost">decision agreement {pct(consistency.predicted_label_agreement, 2)}</Badge>
        {consistency.signature_matches === false && <Badge tone="badge-alert">dataset signature moved</Badge>}
      </div>
      {notes.length > 0 && (
        <ul className="mt-2.5 space-y-1.5">
          {notes.map((note) => (
            <li key={note} className="text-[.875rem] leading-relaxed text-[var(--ink-3)]">
              {note}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
