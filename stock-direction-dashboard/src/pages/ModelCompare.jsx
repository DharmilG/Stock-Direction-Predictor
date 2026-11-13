import { useMemo, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import { api } from '../lib/api';
import { POLL_INTERVALS } from '../lib/constants';
import { int, num, pct, pp, signed, signedPct, titleize } from '../lib/format';
import { useResource } from '../hooks/useResource';
import { modelKey, useModel } from '../state/ModelProvider';
import { CalibrationChart, ConfusionMatrix, ImportanceBars, LineSeriesChart, PairedShareBars } from '../components/charts';
import {
  Badge,
  Button,
  Empty,
  ErrorNote,
  KeyValue,
  Loading,
  PageHeader,
  Panel,
  Section,
  Segmented,
  SourceTag,
  Stat,
  StatGrid,
  Table,
  Warnings,
} from '../components/ui';

/**
 * The two models side by side.
 *
 * Two records, deliberately separate. The sections from "Scorecard" to "Backtest"
 * come from `/dashboard/compare/*`, which pairs v1 and v2 on the rows both of them
 * scored and puts a date-clustered interval on every difference. The sections under
 * "Experiment record" are v2's own report — how the variant was chosen, what the
 * gates said — and are left exactly as the trainer wrote them, because that is the
 * evidence for the design rather than for the comparison.
 *
 * The split matters: the paired deltas answer "which is better", the record answers
 * "why this v2". Folding the second into the first would let the experiment's own
 * choice of population stand in for a fair comparison.
 */

const RATIO_KEYS = new Set([
  'accuracy_all_rows',
  'accuracy_movers',
  'directional_coverage',
  'accuracy_when_calling',
  'three_class_accuracy',
  'up_recall',
]);

const DASH = '—';

const CLASS_LABELS = ['DOWN', 'FLAT', 'UP'];

const SEGMENT_OPTIONS = [
  { value: 'class', label: 'True class' },
  { value: 'confidence', label: 'Confidence' },
  { value: 'volatility', label: 'Volatility' },
  { value: 'month', label: 'Month' },
  { value: 'agreement', label: 'Agreement' },
  { value: 'symbol', label: 'Symbol' },
];

const PREDICTED_OPTIONS = [
  { value: 'all', label: 'All' },
  { value: 'DOWN', label: 'DOWN' },
  { value: 'FLAT', label: 'FLAT' },
  { value: 'UP', label: 'UP' },
  { value: 'ABSTAIN', label: 'Abstain' },
];

const DISAGREE_OPTIONS = [
  { value: 'all', label: 'All rows' },
  { value: 'v2', label: 'v2 correct' },
  { value: 'incumbent', label: 'Incumbent correct' },
  { value: 'disagree', label: 'Models disagree' },
];

function formatMetric(key, value) {
  if (value == null) return DASH;
  return RATIO_KEYS.has(key) ? pct(value, 2) : num(value, 4);
}

function formatDelta(key, value) {
  if (value == null) return DASH;
  return RATIO_KEYS.has(key) ? pp(value * 100, 2) : signed(value, 4);
}

function reliabilityRows(bins) {
  return (bins ?? [])
    .filter((bin) => bin.count > 0 && bin.confidence != null && bin.accuracy != null)
    .map((bin) => ({ mean_predicted: bin.confidence, mean_observed: bin.accuracy, count: bin.count }));
}

function GateBadge({ gate }) {
  return (
    <Badge tone={gate.pass ? 'badge-solid' : 'badge-dashed'} title={`threshold ${gate.threshold}`}>
      {gate.pass ? 'PASS' : 'FAIL'}
    </Badge>
  );
}

/** A scorecard cell, in the unit the axis declares. */
function cell(unit, value) {
  if (value == null) return DASH;
  return unit === 'pct' ? pct(value, 2) : num(value, 4);
}

/** The paired difference: percentage points on a share, raw units otherwise. */
function deltaCell(row) {
  if (row.delta == null) return DASH;
  return row.unit === 'pct' ? pp(row.delta * 100, 2) : signed(row.delta, 4);
}

/** The date-clustered interval, in the unit of the delta it brackets. */
function intervalCell(row) {
  if (row.ci_low == null || row.ci_high == null) {
    return <span className="text-[var(--ink-3)]">no interval</span>;
  }
  const scale = row.unit === 'pct' ? 100 : 1;
  const digits = row.unit === 'pct' ? 2 : 4;
  return `${signed(row.ci_low * scale, digits)} … ${signed(row.ci_high * scale, digits)}`;
}

/** Who the interval favours, or the reason it decides nothing. */
function Decided({ row }) {
  if (row.significant == null) {
    return (
      <Badge tone="badge-dashed" title="A ranking statistic is not a sum over rows, so it gets no paired interval and no verdict.">
        no interval
      </Badge>
    );
  }
  if (!row.significant) {
    return (
      <Badge tone="badge-ghost" title="The 95% interval contains zero, so this axis does not separate the two models.">
        within noise
      </Badge>
    );
  }
  return <Badge tone="badge-solid">{row.better}</Badge>;
}

/** The winner of one slice, or why the slice proves nothing. */
function WinnerCell({ row }) {
  if (row.note) {
    return (
      <Badge tone="badge-dashed" title={row.note}>
        too thin
      </Badge>
    );
  }
  if (row.significant == null) return <span className="text-[var(--ink-3)]">counts only</span>;
  if (!row.significant) return <Badge tone="badge-ghost">within noise</Badge>;
  return <Badge tone="badge-solid">{row.winner}</Badge>;
}

export default function ModelCompare() {
  const { models } = useModel();
  const v2Ready = Boolean(models.find((model) => model.id === 'v2')?.ready);
  const registrySettled = models.length > 0;

  // The comparison: v1 is the reference, v2 the challenger. These are the only
  // two places on this page where a model id is written as a literal.
  const scorecard = useResource((options) => api.compareSummary({ a: 'v1', b: 'v2' }, options), {
    key: 'compare-summary',
    pollMs: POLL_INTERVALS.artifacts,
    enabled: !registrySettled || v2Ready,
  });
  const [by, setBy] = useState('class');
  const segments = useResource((options) => api.compareSegments({ a: 'v1', b: 'v2', by }, options), {
    key: `compare-segments-${by}`,
    pollMs: POLL_INTERVALS.artifacts,
    enabled: !registrySettled || v2Ready,
  });
  const pairing = useResource((options) => api.comparePairing({ a: 'v1', b: 'v2' }, options), {
    key: 'compare-pairing',
    pollMs: POLL_INTERVALS.artifacts,
    enabled: !registrySettled || v2Ready,
  });
  const attribution = useResource((options) => api.compareAttribution({ a: 'v1', b: 'v2' }, options), {
    key: 'compare-attribution',
    pollMs: POLL_INTERVALS.artifacts,
    enabled: !registrySettled || v2Ready,
  });

  // Per-model artefacts, read twice: one confusion matrix each and one backtest each.
  const metricsV1 = useResource((options) => api.metrics('v1', options), { key: modelKey('v1', 'metrics'), pollMs: POLL_INTERVALS.artifacts });
  const metricsV2 = useResource((options) => api.metrics('v2', options), {
    key: modelKey('v2', 'metrics'),
    pollMs: POLL_INTERVALS.artifacts,
    enabled: !registrySettled || v2Ready,
  });
  const backtestV1 = useResource((options) => api.backtest('v1', options), { key: modelKey('v1', 'backtest'), pollMs: POLL_INTERVALS.artifacts });
  const backtestV2 = useResource((options) => api.backtest('v2', options), {
    key: modelKey('v2', 'backtest'),
    pollMs: POLL_INTERVALS.artifacts,
    enabled: !registrySettled || v2Ready,
  });

  // The experiment record at the foot of the page is v2's own report.
  const summary = useResource(api.v2Summary, { key: 'v2-summary', pollMs: POLL_INTERVALS.artifacts });
  const experiments = useResource(api.v2Experiments, { key: 'v2-experiments', pollMs: POLL_INTERVALS.artifacts });
  const diagnostics = useResource(api.v2Diagnostics, { key: 'v2-diagnostics', pollMs: POLL_INTERVALS.artifacts });
  const reasons = useResource(api.v2Reasons, { key: 'v2-reasons', pollMs: POLL_INTERVALS.artifacts });

  const [predicted, setPredicted] = useState('all');
  const [correct, setCorrect] = useState('all');
  const [offset, setOffset] = useState(0);
  const cases = useResource((options) => api.v2Cases({ predicted, correct, limit: 25, offset }, options), {
    key: `v2-cases-${predicted}-${correct}-${offset}`,
  });

  const data = summary.data;
  const comparison = data?.comparison ?? {};
  const incumbent = data?.incumbent ?? {};
  const population = data?.population ?? {};
  const protocol = data?.protocol ?? {};
  const selected = data?.selected_variant;
  const selection = data?.selection ?? null;

  const selectedVariant = useMemo(() => {
    const rows = experiments.data?.variants ?? [];
    return rows.find((row) => row.id === selected) ?? null;
  }, [experiments.data, selected]);

  const diag = diagnostics.data?.diagnostics ?? {};
  const folds = diagnostics.data?.folds ?? [];
  const reasonData = reasons.data ?? {};

  const headToHead = comparison.rows ?? [];
  const gates = comparison.gates ?? [];
  const significance = comparison.significance ?? [];

  const directional = significance.find((item) => item.label?.startsWith('directional')) ?? null;

  // Derived per render rather than memoised: these are three-element folds over
  // small objects, and a memo keyed on a fresh `?? {}` would recompute anyway.
  const mixActual = population.class_share ?? {};
  // No fallback to the incumbent here: showing its distribution under a v2
  // heading would be a straight misreport. If the variant has no mix, the panel
  // says so instead.
  const mixPredicted = selectedVariant?.mix ?? {};
  const hasMix = Object.keys(mixPredicted).length > 0;
  const pairedMix = {
    actual: Object.fromEntries(CLASS_LABELS.map((label) => [label, mixActual[label] ?? 0])),
    predicted: Object.fromEntries(CLASS_LABELS.map((label) => [label, mixPredicted[label] ?? 0])),
    bias_pp: Object.fromEntries(
      CLASS_LABELS.map((label) => [label, ((mixPredicted[label] ?? 0) - (mixActual[label] ?? 0)) * 100]),
    ),
  };

  const riskRows = (diag.risk_coverage?.v2 ?? []).map((point) => ({
    coverage: point.coverage,
    accuracy_called: point.accuracy_called,
  }));

  const branchRows = Object.entries(reasonData.reasons?.selected?.branch_share ?? {}).map(([key, value]) => ({
    branch: titleize(key),
    share: value,
  }));

  // --- comparison payloads ------------------------------------------------ //
  const card = scorecard.data ?? {};
  const scoreRows = card.scorecard ?? [];
  const segmentData = segments.data ?? {};
  const segmentRows = segmentData.rows ?? [];
  const agree = pairing.data?.agreement ?? {};
  const flips = pairing.data?.flips ?? [];
  const attributeData = attribution.data ?? {};
  const divergent = attributeData.divergent ?? [];

  const backtestRows = useMemo(() => {
    const a = backtestV1.data?.metrics ?? {};
    const b = backtestV2.data?.metrics ?? {};
    const specs = [
      { key: 'total_return', label: 'Total return', format: signedPct, diff: (x, y) => pp((y - x) * 100, 2) },
      { key: 'max_drawdown', label: 'Max drawdown', format: (value) => pct(value), diff: (x, y) => pp((y - x) * 100, 2) },
      { key: 'sharpe', label: 'Sharpe', format: (value) => num(value, 2), diff: (x, y) => signed(y - x, 2) },
      { key: 'trade_count', label: 'Trades taken', format: (value) => int(value), diff: (x, y) => signed(y - x, 0) },
    ];
    return specs.map((spec) => {
      const left = a[spec.key];
      const right = b[spec.key];
      return {
        ...spec,
        a: left,
        b: right,
        delta: left == null || right == null ? null : spec.diff(left, right),
      };
    });
  }, [backtestV1.data, backtestV2.data]);

  // Registry warnings are about the pair, so they belong at the top of the page
  // rather than inside either model's own panels.
  const warnings = [
    ...new Set([
      ...(scorecard.envelope?.warnings ?? []),
      ...(segments.envelope?.warnings ?? []),
      ...(attribution.envelope?.warnings ?? []),
      ...(summary.envelope?.warnings ?? []),
      ...(diagnostics.envelope?.warnings ?? []),
    ]),
  ];

  const refreshAll = () => {
    [
      scorecard,
      segments,
      pairing,
      attribution,
      metricsV1,
      metricsV2,
      backtestV1,
      backtestV2,
      summary,
      experiments,
      diagnostics,
      reasons,
    ].forEach((resource) => resource.refresh());
  };

  const header = (
    <PageHeader
      eyebrow="Model comparison"
      title="v1 against v2, where each wins and why"
      description="Both models are scored once on the rows they share, so every difference below is paired. v2 lives entirely under v2/ and is never served: this page decides what each model is good for, not which one trades."
      actions={
        <>
          <SourceTag envelope={scorecard.envelope} />
          <Button variant="secondary" icon={RefreshCw} onClick={refreshAll}>
            Refresh
          </Button>
        </>
      }
    />
  );

  if (registrySettled && !v2Ready) {
    return (
      <>
        {header}
        <Empty
          title="v2 is not built yet"
          detail="There is nothing to compare until its artefacts exist. Run `python -m stockml.v2.train` and then `python -m stockml.v2.stages all`; the header will offer v2 as soon as the registry sees them."
        />
      </>
    );
  }

  if (scorecard.error && !scorecard.data) {
    return (
      <>
        {header}
        <ErrorNote error={scorecard.error} onRetry={scorecard.refresh} label="The two models could not be paired" />
      </>
    );
  }

  if (scorecard.loading && !scorecard.data) {
    return <Loading rows={6} label="Pairing v1 and v2 on the rows they share" />;
  }

  return (
    <>
      {header}
      <Warnings warnings={warnings} />

      <Section
        eyebrow="Scorecard"
        title="Which model is better, and on what"
        note="Every delta is measured on the rows both models scored, with a 95% interval from a bootstrap that resamples dates rather than rows: fifty names share one day's move, so a row is not an independent observation. The own-window columns are each model's whole holdout, shown so the size of the cut is visible."
      >
        <StatGrid columns={4}>
          <Stat
            label="Rows v1 scored"
            value={int(card.a_rows)}
            info={{
              what: 'Everything in v1s evaluation file.',
              reads: 'The served booster on its own holdout.',
              ideal: 'Equal to v2s count; when they differ the two evaluations used different test windows.',
            }}
          />
          <Stat
            label="Rows v2 scored"
            value={int(card.b_rows)}
            info={{
              what: 'Everything in v2s evaluation file.',
              reads: 'The research twin on its own holdout, which spans more history by design.',
              ideal: 'Close to v1s count, so the overlap is most of both files.',
            }}
          />
          <Stat
            label="Rows on both"
            value={int(card.overlap_rows)}
            info={{
              what: 'Rows present in both files, matched on date and symbol.',
              reads: 'The widest population on which a paired comparison is even possible.',
              ideal: 'Close to the smaller of the two row counts.',
            }}
          />
          <Stat
            label="Scoreable, paired"
            value={int(card.paired_rows)}
            info={{
              what: 'Rows that also agree about the true label.',
              reads: `The overlap minus ${int(card.label_mismatch)} row(s) whose label changed between the two snapshots.`,
              ideal: 'Equal to the overlap. A gap means one evaluation predates the current label build.',
              when: 'A large gap makes the deltas describe a smaller sample than the file sizes suggest.',
            }}
          />
        </StatGrid>

        <Panel
          className="mt-6"
          title="Paired scorecard"
          eyebrow={`${int(card.paired_rows)} rows · ${int(card.test_dates)} dates · ${int(card.reps)} draws`}
          info={{
            what: 'Every axis both models can be scored on, computed on identical rows.',
            reads: 'Own-window columns score each model on its full holdout; shared columns score both on the same rows; the delta and its interval are paired.',
            ideal: 'A delta whose interval excludes zero, in the direction that is better for that axis.',
            when: 'AUC rows carry no interval: a ranking statistic is not a sum over rows, so it cannot be resampled the same way.',
          }}
        >
          <Table
            columns={[
              { key: 'label', header: 'Axis', strong: true },
              { key: 'a', header: 'v1 · own', align: 'right', mono: true, render: (row) => cell(row.unit, row.a) },
              { key: 'b', header: 'v2 · own', align: 'right', mono: true, render: (row) => cell(row.unit, row.b) },
              { key: 'paired_a', header: 'v1 · shared', align: 'right', mono: true, render: (row) => cell(row.unit, row.paired_a) },
              { key: 'paired_b', header: 'v2 · shared', align: 'right', mono: true, render: (row) => cell(row.unit, row.paired_b) },
              { key: 'delta', header: 'v2 − v1', align: 'right', mono: true, render: deltaCell },
              { key: 'ci', header: '95% CI', align: 'right', mono: true, render: intervalCell },
              { key: 'decided', header: 'Decided', render: (row) => <Decided row={row} /> },
            ]}
            rows={scoreRows}
            rowKey={(row) => row.axis}
            empty={<Empty title="Nothing paired" detail="One of the two models has no evaluation on disk." />}
          />
          <p className="mt-4 border-l-2 border-[var(--ink)] pl-3 text-[.9375rem] leading-relaxed text-[var(--ink-2)]">{card.verdict}</p>
          {card.label_mismatch > 0 && (
            <div className="mt-3 space-y-1">
              {card.label_drift.map((row) => (
                <p key={`${row.a_label}-${row.b_label}`} className="mono text-[.8125rem] text-[var(--ink-3)]">
                  {int(row.n)} row(s): {row.a_label} in v1&rsquo;s snapshot, {row.b_label} in v2&rsquo;s
                </p>
              ))}
            </div>
          )}
        </Panel>
      </Section>

      <Section
        eyebrow="Where each wins"
        title="The same delta, sliced"
        note="One overall number hides the useful part: the two models fail on different rows. Each slice re-computes the paired accuracy difference inside that slice, with an interval that resamples the dates inside it."
        actions={<Segmented options={SEGMENT_OPTIONS} value={by} onChange={setBy} label="Split by" />}
      >
        <Panel
          title="Paired accuracy by slice"
          eyebrow={segmentData.definition ?? 'slicing'}
          info={{
            what: 'The paired accuracy difference, recomputed inside each slice of the shared population.',
            reads: 'Positive means v2 is more accurate on that slice. Percentages move in percentage points; the interval is over dates.',
            ideal: 'A clear split: v2 ahead where the classes it was built for are, v1 ahead elsewhere.',
            when: 'A slice below the row or date floor gets no interval, because resampling it would be guesswork.',
          }}
        >
          <Table
            columns={[
              { key: 'key', header: 'Slice', strong: true },
              { key: 'n', header: 'Rows', align: 'right', mono: true, render: (row) => int(row.n) },
              { key: 'dates', header: 'Dates', align: 'right', mono: true, render: (row) => int(row.dates) },
              { key: 'a_accuracy', header: 'v1', align: 'right', mono: true, render: (row) => pct(row.a_accuracy, 2) },
              { key: 'b_accuracy', header: 'v2', align: 'right', mono: true, render: (row) => pct(row.b_accuracy, 2) },
              { key: 'delta_pp', header: 'v2 − v1', align: 'right', mono: true, render: (row) => pp(row.delta_pp, 2) },
              {
                key: 'ci',
                header: '95% CI (pp)',
                align: 'right',
                mono: true,
                render: (row) => (row.ci_low == null ? DASH : `${signed(row.ci_low, 1)} … ${signed(row.ci_high, 1)}`),
              },
              { key: 'winner', header: 'Winner', render: (row) => <WinnerCell row={row} /> },
            ]}
            rows={segmentRows}
            rowKey={(row) => row.key}
            empty={<Empty title="No slices" detail="The paired population is too thin to split." />}
          />
          {(segmentData.notes ?? []).map((note) => (
            <p key={note} className="mt-3 text-[.9375rem] leading-relaxed text-[var(--ink-2)]">
              {note}
            </p>
          ))}
        </Panel>
      </Section>

      <Section
        eyebrow="Behaviour"
        title="Where they agree, and where they do not"
        note="Accuracy says how often each model is right. The two-by-two says how often they disagree, which is the only place a difference between them can come from."
      >
        <div className="grid gap-6 lg:grid-cols-3">
          <Panel
            title="The two-by-two"
            eyebrow={`${int(pairing.data?.paired_rows)} shared rows`}
            info={{
              what: 'Who was right, cross-tabbed, on the rows both models scored.',
              reads: 'The diagonal is agreement. Only the off-diagonal cells, the discordant rows, carry evidence about which model is better.',
              ideal: 'A large both-right cell, and a discordance that leans clearly one way.',
              when: 'Near-equal off-diagonals mean the two models trade wins row by row and the accuracy gap is a coincidence of the sample.',
            }}
          >
            <Table
              dense
              columns={[
                { key: 'cell', header: '' },
                { key: 'v2_right', header: 'v2 right', align: 'right', mono: true },
                { key: 'v2_wrong', header: 'v2 wrong', align: 'right', mono: true },
              ]}
              rows={[
                { cell: 'v1 right', v2_right: int(agree.both_right), v2_wrong: int(agree.a_only) },
                { cell: 'v1 wrong', v2_right: int(agree.b_only), v2_wrong: int(agree.both_wrong) },
              ]}
              rowKey={(row) => row.cell}
            />
            <div className="mt-4 border-t border-[var(--line)] pt-4">
              <KeyValue
                columns={1}
                items={[
                  { label: 'Discordant rows', value: int(pairing.data?.discordant_n), mono: true },
                  { label: 'McNemar p', value: num(pairing.data?.mcnemar_p, 4), mono: true },
                ]}
              />
            </div>
            <p className="mt-3 text-[.9375rem] leading-relaxed text-[var(--ink-2)]">{pairing.data?.reading}</p>
          </Panel>

          <Panel
            title="v1 confusion"
            eyebrow="own holdout"
            info={{
              what: 'Predicted class against true class for the served booster, over its whole holdout.',
              reads: 'Rows are the realised label, columns the prediction. This is not the paired population, so read the scorecard for the like-for-like numbers.',
              ideal: 'A dark diagonal across all three classes.',
              when: 'Nearly all mass in one column means accuracy is being bought with a constant answer.',
            }}
          >
            <ConfusionMatrix matrix={metricsV1.data?.metrics?.confusion_matrix} title="v1 confusion matrix" />
          </Panel>

          <Panel
            title="v2 confusion"
            eyebrow="own holdout"
            info={{
              what: 'The same matrix for the research twin, over its own, longer holdout.',
              reads: 'Compare the shape with v1 above: where the mass moves tells you what the retrain changed.',
              ideal: 'A thinner FLAT column than v1 with the diagonal still holding.',
              when: 'A diagonal that only holds because coverage fell means abstention, not skill.',
            }}
          >
            <ConfusionMatrix matrix={metricsV2.data?.metrics?.confusion_matrix} title="v2 confusion matrix" />
          </Panel>
        </div>

        <Panel
          className="mt-6"
          title="Where the call changed"
          eyebrow="Flips"
          info={{
            what: 'Every pair of different calls, with how often each side turned out to be right on that group of rows.',
            reads: 'Sorted by size. A big group where the new call is worse is a regression that the aggregate accuracy can hide.',
            ideal: 'The large groups favour v2, and the groups it loses are small.',
            when: 'A large group whose hit rate barely moves means the rule changed without changing the answer.',
          }}
        >
          <Table
            columns={[
              { key: 'a_pred', header: 'v1 said' },
              { key: 'b_pred', header: 'v2 said' },
              { key: 'n', header: 'Rows', align: 'right', mono: true, render: (row) => int(row.n) },
              { key: 'a_hit_rate', header: 'v1 hit', align: 'right', mono: true, render: (row) => pct(row.a_hit_rate, 1) },
              { key: 'b_hit_rate', header: 'v2 hit', align: 'right', mono: true, render: (row) => pct(row.b_hit_rate, 1) },
              {
                key: 'swing',
                header: 'Swing',
                align: 'right',
                mono: true,
                render: (row) => pp((row.b_hit_rate - row.a_hit_rate) * 100, 1),
              },
            ]}
            rows={flips}
            rowKey={(row) => `${row.a_pred}-${row.b_pred}`}
            empty={<Empty title="No flips" detail="The two rules called every shared row the same way." />}
          />
        </Panel>
      </Section>

      <Section eyebrow="Attribution" title="Whether they are looking at the same thing" note={attributeData.note}>
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel
            title="v1 top features"
            eyebrow={`${int(attributeData.a_features)} features`}
            info={{
              what: 'Mean absolute SHAP over v1s test rows, as a share of the total importance.',
              reads: 'Not a direction: it says how much a feature moves the output, not which way.',
              ideal: 'The features the strategy story expects: volatility, market context, turnover.',
            }}
          >
            <ImportanceBars rows={attributeData.a_top ?? []} valueKey="share" labelKey="feature" limit={15} formatter={(value) => pct(value, 1)} />
          </Panel>
          <Panel
            title="v2 top features"
            eyebrow={`${int(attributeData.b_features)} features`}
            info={{
              what: 'The same measure for the research twin, over its own rows.',
              reads: 'Measured on a different sample, so compare ranks rather than absolute shares.',
              ideal: 'A list that mostly repeats v1s, with the differences explained by the design.',
            }}
          >
            <ImportanceBars rows={attributeData.b_top ?? []} valueKey="share" labelKey="feature" limit={15} formatter={(value) => pct(value, 1)} />
          </Panel>
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-3">
          <Panel
            title="Rank agreement"
            eyebrow="Spearman"
            info={{
              what: 'How closely the two importance orderings agree across the features both models carry.',
              reads: 'One at 1.0 means identical orderings; near zero means the two models have nothing in common in what they lean on.',
              ideal: 'High, between 0.8 and 1.0: the retrain changed the mapping, not the information used.',
              when: 'A low value would mean v2 is a different model looking at different data, and the paired deltas would need re-reading.',
            }}
          >
            <KeyValue
              columns={1}
              items={[
                { label: 'Rank correlation', value: num(attributeData.rank_correlation, 3), mono: true },
                { label: 'Features in both', value: int(attributeData.shared_features), mono: true },
                { label: 'Top-15 only in v1', value: int(divergent.filter((row) => row.only_in === 'v1').length), mono: true },
                { label: 'Top-15 only in v2', value: int(divergent.filter((row) => row.only_in === 'v2').length), mono: true },
              ]}
            />
          </Panel>
          <Panel
            className="lg:col-span-2"
            title="Where the top lists diverge"
            eyebrow={`${int(divergent.length)} feature(s)`}
            info={{
              what: 'Features in one model top fifteen that are not in the other.',
              reads: 'A rank of null means the feature is not in that model at all.',
              ideal: 'A short list, all of it explicable by the difference in design.',
            }}
          >
            <Table
              dense
              columns={[
                { key: 'feature', header: 'Feature', mono: true, strong: true },
                { key: 'a_rank', header: 'v1 rank', align: 'right', mono: true, render: (row) => (row.a_rank == null ? DASH : int(row.a_rank + 1)) },
                { key: 'b_rank', header: 'v2 rank', align: 'right', mono: true, render: (row) => (row.b_rank == null ? DASH : int(row.b_rank + 1)) },
                { key: 'only_in', header: 'Only in', render: (row) => <Badge tone="badge-ghost">{row.only_in}</Badge> },
              ]}
              rows={divergent}
              rowKey={(row) => row.feature}
              empty={<Empty title="No divergence" detail="Both top-fifteen lists name the same features." />}
            />
          </Panel>
        </div>
      </Section>

      <Section
        eyebrow="Backtest"
        title="What each rule would have earned"
        note="Both curves charge the same round-trip cost on the same events. What differs is the rule, and therefore which events were taken: an accuracy gain that only removes trades does not necessarily make money."
      >
        <Panel
          title="Net of costs"
          eyebrow="v1 against v2"
          info={{
            what: 'Headline backtest figures for each model, read from its own backtest artefacts.',
            reads: 'The delta is v2 minus v1. Returns and drawdowns are percentages; Sharpe is a ratio and trades a count.',
            ideal: 'v2 ahead on return and Sharpe without a materially deeper drawdown.',
            when: 'A higher accuracy with a lower return means the model is better at rows the strategy never trades.',
          }}
        >
          <Table
            columns={[
              { key: 'label', header: 'Metric', strong: true },
              { key: 'a', header: 'v1', align: 'right', mono: true, render: (row) => (row.a == null ? DASH : row.format(row.a)) },
              { key: 'b', header: 'v2', align: 'right', mono: true, render: (row) => (row.b == null ? DASH : row.format(row.b)) },
              { key: 'delta', header: 'v2 − v1', align: 'right', mono: true, render: (row) => row.delta ?? DASH },
            ]}
            rows={backtestRows}
            rowKey={(row) => row.key}
            empty={<Empty title="No backtest" detail="One of the two models has no backtest artefacts on disk." />}
          />
          <p className="mt-4 text-[.9375rem] text-[var(--ink-3)]">
            v1 window {backtestV1.data?.start ?? DASH} → {backtestV1.data?.end ?? DASH} · v2 window {backtestV2.data?.start ?? DASH} →{' '}
            {backtestV2.data?.end ?? DASH}
          </p>
        </Panel>
      </Section>

      <Section
        eyebrow="Experiment record · verdict"
        title={comparison.promote ? 'The candidate clears every gate' : 'The candidate is not promotable'}
        note={
          comparison.promote
            ? 'Every promotion gate passed on the shared test rows.'
            : 'The gates below are the whole decision. A point estimate that beats the incumbent is not enough on its own, because rows on the same date are not independent observations.'
        }
      >
        <div className="grid gap-6 lg:grid-cols-3">
          <Panel
            title="Promotion gates"
            eyebrow="Decision"
            info={{
              what: 'Three conditions a candidate must meet before it may replace the served booster.',
              reads: 'Directional balance, AUC(UP) and the date-clustered t of the directional gain. All three must pass.',
              ideal: 'All three pass, and the t is above 2 on date-clustered errors, not row-level errors.',
              when: 'A failure here is the answer, not a bug: it means v2 is a research result, not a replacement.',
            }}
          >
            <ul className="space-y-3">
              {gates.map((gate) => (
                <li key={gate.id} className="flex items-center justify-between gap-4 border-b border-[var(--line)] pb-2.5 last:border-0 last:pb-0">
                  <div className="min-w-0">
                    <p className="text-[.9375rem] font-semibold text-[var(--ink)]">{gate.label}</p>
                    <p className="mt-0.5 text-[.875rem] text-[var(--ink-3)]">
                      requires {gate.direction} {num(gate.threshold, 2)} · measured {num(gate.value, 4)}
                    </p>
                  </div>
                  <GateBadge gate={gate} />
                </li>
              ))}
              {!gates.length && <Empty title="No gates recorded" detail="The run finished without a feasible variant." />}
            </ul>
            <p className="mt-4 border-l-2 border-[var(--ink)] pl-3 text-[.9375rem] leading-relaxed text-[var(--ink-2)]">
              {comparison.verdict}
            </p>
          </Panel>

          <Panel
            title="Headline"
            eyebrow="Selected variant"
            info={{
              what: 'The variant the matrix selected, and the three numbers that decide its fate.',
              reads: 'Directional accuracy counts every test row; the delta is v2 minus the incumbent in percentage points.',
              ideal: 'A positive delta that also survives the date-clustered test.',
              when: 'A positive delta with t below 2 means the gain is inside noise on this panel.',
            }}
          >
            <KeyValue
              items={[
                { label: 'Selected variant', value: selected ?? DASH },
                { label: 'Rows shared', value: int(population.test_rows) },
                { label: 'Test dates', value: int(population.test_dates) },
                { label: 'Coverage', value: pct(selectedVariant?.coverage) },
                { label: 'Round trip cost', value: '31.5 bps' },
                { label: 'Rounds fitted', value: int(protocol.rounds) },
              ]}
            />
            <p className="mt-4 text-[.9375rem] leading-relaxed text-[var(--ink-2)]">
              {directional
                ? `Directional accuracy moves ${formatDelta('accuracy_all_rows', directional.mean_delta)} with t = ${num(directional.t, 2)} over ${int(directional.dates)} dates.`
                : 'No significance test was recorded for this run.'}
            </p>
          </Panel>

          <Panel
            title="Protocol"
            eyebrow="How it was measured"
            info={{
              what: 'The rules that make the comparison paired and the policy selection honest.',
              reads: 'Both models are scored on one shared population; the decision policy is chosen on the calibration slice only, then applied to test once.',
              ideal: 'Shared population, policy selected without seeing test, and a single test pass.',
              when: 'If any of these were skipped the numbers stop being comparable and are not reported.',
            }}
          >
            <KeyValue
              items={[
                { label: 'Train ends', value: protocol.splits?.train_end ?? DASH },
                { label: 'Test window', value: `${protocol.splits?.test_start ?? DASH} → ${protocol.splits?.test_end ?? DASH}` },
                { label: 'Purge / embargo', value: `${protocol.purge_days} / ${protocol.embargo_days} days` },
                { label: 'Features (pinned)', value: int(protocol.features) },
                { label: 'Incumbent rule', value: `p_flat ≥ ${num(protocol.incumbent_rule?.tau_flat, 2)}, tie → DOWN` },
                { label: 'Policy picked on', value: protocol.policy_selected_on ?? DASH },
              ]}
            />
          </Panel>
        </div>

        <Panel
          className="mt-6"
          title="Head to head"
          eyebrow="Shared test rows"
          info={{
            what: 'Every metric that decides whether one model is better than the other, computed once on identical rows.',
            reads: 'The delta column is v2 minus the incumbent. Percentages move in percentage points; log loss and correlation move in raw units.',
            ideal: 'Positive deltas where higher is better, negative on log loss and calibration error.',
            when: 'A metric that improves while direction does not is a probability win, not a trading win.',
          }}
        >
          <Table
            rows={headToHead}
            rowKey={(row) => row.key}
            columns={[
              { key: 'metric', header: 'Metric' },
              { key: 'incumbent', header: 'Incumbent', align: 'right', mono: true, render: (row) => formatMetric(row.key, row.incumbent) },
              { key: 'v2', header: 'v2', align: 'right', mono: true, strong: true, render: (row) => formatMetric(row.key, row.v2) },
              { key: 'delta', header: 'Delta', align: 'right', mono: true, render: (row) => formatDelta(row.key, row.delta) },
            ]}
            empty={<Empty title="No comparison rows" detail="The run produced no feasible candidate to compare." />}
          />
        </Panel>

        <Panel
          className="mt-6"
          title="Why the delta can be trusted, or not"
          eyebrow="Significance"
          info={{
            what: 'Paired bootstrap of the difference, resampled over dates rather than rows.',
            reads: 'Fifty names on one day share a market move, so a row is not an independent observation. The independent unit is the date.',
            ideal: 'A confidence interval that excludes zero and a t of at least 2.',
            when: 'A row-level standard error on this panel is roughly two times too small; it is not printed here for that reason.',
          }}
        >
          <Table
            rows={significance}
            rowKey={(row) => row.label}
            columns={[
              { key: 'label', header: 'Quantity' },
              { key: 'mean_delta', header: 'v2 − incumbent', align: 'right', mono: true, render: (row) => signed(row.mean_delta, 4) },
              { key: 'ci', header: '95% CI', align: 'right', mono: true, render: (row) => `${signed(row.ci_low, 3)} … ${signed(row.ci_high, 3)}` },
              { key: 't', header: 't', align: 'right', mono: true, render: (row) => num(row.t, 2) },
              { key: 'dates', header: 'Dates', align: 'right', mono: true, render: (row) => int(row.dates) },
              { key: 'rows', header: 'Rows', align: 'right', mono: true, render: (row) => int(row.rows) },
            ]}
            empty={<Empty title="No significance test" detail="Needs a feasible variant." />}
          />
        </Panel>
      </Section>

      <Section
        eyebrow="Experiment record · calls"
        title="What the rule chooses to say"
        note="The incumbent almost always emits a direction. v2 may abstain, and the question is whether abstaining buys accuracy or just hides errors."
      >
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel
            title="Predicted against actual"
            eyebrow="Mix"
            info={{
              what: 'How often each class is actually true versus how often each model says it.',
              reads: 'Grey bars are the truth, dark bars are the calls. The bias column is calls minus truth in percentage points.',
              ideal: 'Predicted shares close to actual shares on all three classes.',
              when: 'A large bias on DOWN means the rule has collapsed into a single call and its accuracy is riding the class mix.',
            }}
          >
            <PairedShareBars distribution={pairedMix} />
            <p className="mt-4 text-[.9375rem] leading-relaxed text-[var(--ink-2)]">
              {hasMix
                ? `v2 abstains on ${pct(mixPredicted.ABSTAIN ?? 0)} of rows and emits FLAT on ${pct(mixPredicted.FLAT ?? 0)}. True FLAT is ${pct(mixActual.FLAT ?? 0)} of the test population.`
                : 'The selected variant recorded no call distribution, so the bars show the true mix against nothing. Re-run the trainer to fill them.'}
            </p>
          </Panel>

          <Panel
            title="Accuracy against coverage"
            eyebrow="Risk–coverage"
            info={{
              what: 'What happens to hit rate as the rule is allowed to stay silent on more rows.',
              reads: 'Each point raises the confidence bar: fewer calls to the left, higher accuracy among the rows still called.',
              ideal: 'A curve that rises steeply at first, so silence is spent on the rows that were going to be wrong.',
              when: 'A flat curve means confidence does not rank correctness, and abstention cannot help.',
            }}
          >
            <LineSeriesChart
              data={riskRows}
              xKey="coverage"
              height={240}
              yDomain={[0, 1]}
              yTickFormatter={(value) => pct(value, 0)}
              xTickFormatter={(value) => pct(value, 0)}
              series={[{ key: 'accuracy_called', name: 'Accuracy when calling' }]}
            />
            <p className="mt-3 text-[.9375rem] leading-relaxed text-[var(--ink-2)]">
              The incumbent operates at {pct(diag.risk_coverage?.incumbent_coverage)} coverage with{' '}
              {pct(diag.risk_coverage?.incumbent_accuracy_when_calling)} accuracy on the rows it calls.
            </p>
          </Panel>
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Panel
            title="Which branch fired"
            eyebrow="Attribution"
            info={{
              what: 'Share of test rows attributed to each branch of the decision rule, in the order the branches are evaluated.',
              reads: 'flat_rule then low_confidence then margin then direction. The first branch that matches wins, so shares sum to one.',
              ideal: 'A large direction share with a non-trivial abstain share: it calls when it knows and stays out otherwise.',
              when: 'An abstain share of nearly one means the rule has stopped making decisions; a direction share of one means it never abstains at all.',
            }}
          >
            <ul className="space-y-2.5">
              {branchRows.map((row) => (
                <li key={row.branch} className="grid grid-cols-[9rem_minmax(0,1fr)_4rem] items-center gap-3">
                  <span className="text-[.9375rem] text-[var(--ink-2)]">{row.branch}</span>
                  <span className="h-2 w-full bg-[var(--surface-sunken)]">
                    <span className="block h-2 bg-[var(--ink)]" style={{ width: `${Math.max(0, Math.min(1, row.share)) * 100}%` }} />
                  </span>
                  <span className="mono text-right text-[.9375rem]">{pct(row.share, 1)}</span>
                </li>
              ))}
              {!branchRows.length && <Empty title="No branch attribution" />}
            </ul>
            <p className="mt-4 text-[.9375rem] leading-relaxed text-[var(--ink-2)]">
              Selected policy: tau_flat {num(selectedVariant?.selected_policy?.tau_flat, 2)}, delta{' '}
              {num(selectedVariant?.selected_policy?.delta, 2)}, min_conf {num(selectedVariant?.selected_policy?.min_conf, 2)}.
            </p>
          </Panel>

          <Panel
            title="Branch meanings"
            eyebrow="Reference"
            info={{
              what: 'What each branch means, so a share can be read without the source.',
              ideal: 'No branch is a surprise; each one maps to a rule the policy actually sets.',
            }}
          >
            <ul className="space-y-3">
              {(reasonData.branches ?? []).map((branch) => (
                <li key={branch.id} className="border-b border-[var(--line)] pb-2.5 last:border-0 last:pb-0">
                  <p className="mono text-[.9375rem] font-semibold text-[var(--ink)]">{branch.id}</p>
                  <p className="mt-1 text-[.9375rem] leading-relaxed text-[var(--ink-2)]">{branch.meaning}</p>
                </li>
              ))}
            </ul>
          </Panel>
        </div>
      </Section>

      <Section
        eyebrow="Experiment record · errors"
        title="Confusion, reliability and time"
        note="Direction is the question, but the probabilities and the fold-by-fold behaviour say whether any of it is stable."
      >
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel
            title="Incumbent confusion"
            eyebrow="All test rows"
            info={{
              what: 'Predicted class against true class for the served booster, using its own rule.',
              reads: 'Rows are predictions, columns are truth. The diagonal is correct; a heavy first column is a DOWN collapse.',
              ideal: 'Mass on the diagonal, spread across all three classes.',
              when: 'Nearly all mass in one predicted row means accuracy is being bought with a constant answer.',
            }}
          >
            <ConfusionMatrix matrix={diag.confusion?.incumbent ?? incumbent.confusion_matrix} title="Incumbent confusion matrix" />
          </Panel>
          <Panel
            title="v2 confusion"
            eyebrow="Selected variant"
            info={{
              what: 'Predicted class against true class for the v2 rule, including its abstentions.',
              reads: 'ABSTAIN rows are reported as their own outcome in the branch panel; here they appear as neither DOWN nor UP.',
              ideal: 'A thinner DOWN column than the incumbent with the diagonal still holding.',
              when: 'A diagonal that only holds because coverage fell means abstention, not skill.',
            }}
          >
            <ConfusionMatrix matrix={diag.confusion?.v2} title="v2 confusion matrix" />
          </Panel>
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Panel
            title="Reliability, incumbent"
            eyebrow={`ECE ${num(diag.reliability?.incumbent?.ece, 4)}`}
            info={{
              what: 'Observed accuracy against stated confidence, in ten confidence bins.',
              reads: 'A point on the dashed diagonal means the stated probability was honest. Above it means under-confident.',
              ideal: 'Points hugging the diagonal, and a low expected calibration error.',
              when: 'Bins that flatten at the top mean high confidence does not carry higher accuracy.',
            }}
          >
            <CalibrationChart bins={reliabilityRows(diag.reliability?.incumbent?.bins)} />
          </Panel>
          <Panel
            title="Reliability, v2"
            eyebrow={`ECE ${num(diag.reliability?.v2?.ece, 4)}`}
            info={{
              what: 'The same reliability diagram for the v2 probabilities, after temperature scaling.',
              reads: 'Compare the shape, not only the ECE: a better ECE with a worse top bin can still hurt at the operating point.',
              ideal: 'Hugging the diagonal across the confidence range the rule actually uses.',
              when: 'A lower ECE that comes from a single crowded bin is not an improvement in information.',
            }}
          >
            <CalibrationChart bins={reliabilityRows(diag.reliability?.v2?.bins)} />
          </Panel>
        </div>

        <Panel
          className="mt-6"
          title="Fold by fold"
          eyebrow="Purged walk-forward"
          info={{
            what: 'The selected variant refitted on each walk-forward fold of the pre-test window and scored on the following block.',
            reads: 'The policy and temperature are frozen from the main fit, so these bars show stability, not a new search.',
            ideal: 'v2 ahead of the incumbent in most folds, with the gap not concentrated in one.',
            when: 'One good fold carrying the average means the result does not hold up over time.',
          }}
        >
          <Table
            rows={folds}
            rowKey={(row) => row.fold}
            columns={[
              { key: 'fold', header: 'Fold', mono: true, render: (row) => `#${row.fold}` },
              { key: 'window', header: 'Block', mono: true, render: (row) => `${row.start} → ${row.end}` },
              { key: 'rows', header: 'Rows', align: 'right', mono: true, render: (row) => int(row.rows) },
              { key: 'v2_coverage', header: 'Coverage', align: 'right', mono: true, render: (row) => pct(row.v2_coverage) },
              { key: 'v2_accuracy_all', header: 'Accuracy (all rows)', align: 'right', mono: true, render: (row) => pct(row.v2_accuracy_all) },
              { key: 'v2_accuracy_called', header: 'Accuracy (called)', align: 'right', mono: true, render: (row) => pct(row.v2_accuracy_called) },
            ]}
            empty={<Empty title="Fold bars not computed" detail="Re-run without --no-folds." />}
          />
          <p className="mt-4 text-[.9375rem] leading-relaxed text-[var(--ink-2)]">
            {diagnostics.data?.folds_note ??
              'v2 only — the incumbent is not refit per fold, so it is not scored here.'}
          </p>
        </Panel>
      </Section>

      <Section
        eyebrow="Experiment record · matrix"
        title="Which design survived"
        note="Five designs, one shared test population. A variant that cannot meet the recall floor is reported as infeasible rather than dropped, because that is the finding."
      >
        <Panel
          title="All variants"
          eyebrow="Feasibility"
          info={{
            what: 'Every design in the matrix with the policy it selected on the calibration slice and how it scored on test.',
            reads: 'Feasible means at least one threshold set reached 15% recall on DOWN and UP while each class held at least 10% of rows.',
            ideal: 'More than one feasible design, so the conclusion is not an artefact of a single setting.',
            when: 'Only one feasible design means the result rests on that design alone — which is what happened in the earlier run.',
          }}
        >
          <Table
            rows={experiments.data?.variants ?? []}
            rowKey={(row) => row.id}
            columns={[
              { key: 'id', header: 'Variant', mono: true, strong: true },
              { key: 'row_filter', header: 'Rows' },
              { key: 'class_weighting', header: 'Weighted', render: (row) => (row.class_weighting ? 'yes' : 'no') },
              { key: 'prior_adjustment', header: 'Prior', render: (row) => (row.prior_adjustment ? 'yes' : 'no') },
              { key: 'rows_train', header: 'Train rows', align: 'right', mono: true, render: (row) => int(row.rows_train) },
              {
                key: 'feasible',
                header: 'Feasible',
                render: (row) => (
                  <Badge tone={row.feasible ? 'badge-solid' : 'badge-dashed'}>{row.feasible ? 'yes' : 'no'}</Badge>
                ),
              },
              { key: 'coverage', header: 'Coverage', align: 'right', mono: true, render: (row) => pct(row.coverage) },
              { key: 'accuracy_called', header: 'Acc. called', align: 'right', mono: true, render: (row) => pct(row.accuracy_called) },
              { key: 'accuracy_all_rows', header: 'Acc. all', align: 'right', mono: true, render: (row) => pct(row.accuracy_all_rows) },
              { key: 'balanced_called', header: 'Balance', align: 'right', mono: true, render: (row) => num(row.balanced_called, 4) },
              { key: 'log_loss', header: 'Log loss', align: 'right', mono: true, render: (row) => num(row.log_loss, 4) },
              { key: 'feasible_policies', header: 'Policies', align: 'right', mono: true, render: (row) => `${int(row.feasible_policies)}/${int(row.grid_size)}` },
            ]}
            empty={<Empty title="No variants recorded" detail="Run the trainer to fill the matrix." />}
          />
          <div className="mt-4 space-y-2">
            {(experiments.data?.variants ?? [])
              .filter((row) => !row.feasible && row.infeasible_reason)
              .map((row) => (
                <p key={row.id} className="text-[.9375rem] leading-relaxed text-[var(--ink-2)]">
                  <span className="mono font-semibold text-[var(--ink)]">{row.id}</span>: {row.infeasible_reason}
                </p>
              ))}
          </div>
          {selection ? (
            <p className="mt-4 border-l-2 border-[var(--ink)] pl-3 text-[.9375rem] leading-relaxed text-[var(--ink-2)]">
              {selection.was_tie
                ? `The score did not separate ${selection.contenders.join(', ')} — they were within ${num(selection.tie_epsilon, 3)} of each other — so the tie was broken on ${selection.tie_break}, which is measured on the calibration slice and never on test.`
                : `${selected} ranked first on the selection score with no near-tie to break.`}
            </p>
          ) : null}
        </Panel>
      </Section>

      <Section
        eyebrow="Experiment record · cases"
        title="The rows behind the numbers"
        note="Every test row with both models' calls side by side, so a claim can be checked one date at a time."
      >
        <Panel
          title="Test rows"
          eyebrow="Paired calls"
          info={{
            what: 'Each test row with the true class, the incumbent call, the v2 call and the branch that produced it.',
            reads: 'Filter by what v2 said or by who was right. Dates are the independent unit, not the rows.',
            ideal: 'v2 correct where the incumbent is wrong, concentrated in the direction classes rather than in FLAT.',
            when: 'Improvement that comes only from abstaining shows up as rows where v2 has no call at all.',
          }}
          actions={
            <>
              <Segmented options={PREDICTED_OPTIONS} value={predicted} onChange={(value) => { setPredicted(value); setOffset(0); }} label="v2 call" />
              <Segmented options={DISAGREE_OPTIONS} value={correct} onChange={(value) => { setCorrect(value); setOffset(0); }} label="Outcome filter" />
            </>
          }
        >
          <Table
            rows={cases.data?.rows ?? []}
            rowKey={(row, index) => `${row.symbol}-${row.date}-${index}`}
            columns={[
              { key: 'date', header: 'Date', mono: true },
              { key: 'symbol', header: 'Symbol' },
              { key: 'actual', header: 'Actual' },
              { key: 'incumbent', header: 'Incumbent' },
              { key: 'v2', header: 'v2' },
              { key: 'branch', header: 'Branch' },
              { key: 'p_v2_up', header: 'p(UP)', align: 'right', mono: true, render: (row) => num(row.p_v2_up, 3) },
              { key: 'p_v2_flat', header: 'p(FLAT)', align: 'right', mono: true, render: (row) => num(row.p_v2_flat, 3) },
              { key: 'p_v2_down', header: 'p(DOWN)', align: 'right', mono: true, render: (row) => num(row.p_v2_down, 3) },
              {
                key: 'outcome',
                header: 'Outcome',
                render: (row) => (
                  <span className="flex gap-1.5">
                    <Badge tone={row.correct_v2 ? 'badge-solid' : 'badge-dashed'}>v2 {row.correct_v2 ? '✓' : '✕'}</Badge>
                    <Badge tone={row.correct_inc ? 'badge-solid' : 'badge-dashed'}>inc {row.correct_inc ? '✓' : '✕'}</Badge>
                  </span>
                ),
              },
            ]}
            empty={<Empty title="No rows match" detail="Widen the filters or re-run the trainer." />}
          />
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
            <p className="text-[.9375rem] text-[var(--ink-3)]">
              {int(cases.data?.total)} row(s) match · showing {int(cases.data?.offset)}–
              {int((cases.data?.offset ?? 0) + (cases.data?.rows?.length ?? 0))}
            </p>
            <div className="flex gap-2.5">
              <Button variant="secondary" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 25))}>
                Previous
              </Button>
              <Button
                variant="secondary"
                disabled={(cases.data?.rows?.length ?? 0) < 25}
                onClick={() => setOffset(offset + 25)}
              >
                Next
              </Button>
            </div>
          </div>
        </Panel>
        <p className="mt-4 text-[.9375rem] text-[var(--ink-3)]">
          <SourceTag envelope={summary.envelope} prefix="report written" />
        </p>
      </Section>
    </>
  );
}
