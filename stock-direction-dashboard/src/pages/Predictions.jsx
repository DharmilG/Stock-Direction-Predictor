import { useMemo, useState } from 'react';
import { Download, Search } from 'lucide-react';
import { api } from '../lib/api';
import { POLL_INTERVALS } from '../lib/constants';
import { formatDate } from '../lib/dates';
import { int, num, pct } from '../lib/format';
import { useResource } from '../hooks/useResource';
import { modelKey, useModel } from '../state/ModelProvider';
import { DistributionBars } from '../components/charts';
import { AgreementMark, ProbabilityBar, SignalBadge } from '../components/Signal';
import {
  Badge,
  Button,
  Empty,
  ErrorNote,
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

const FILTERS = [
  { value: 'all', label: 'All' },
  { value: 'UP', label: 'UP' },
  { value: 'DOWN', label: 'DOWN' },
  { value: 'FLAT', label: 'FLAT' },
];

function downloadCsv(rows) {
  const header = ['symbol', 'date', 'prediction', 'p_down', 'p_flat', 'p_up', 'confidence'];
  const body = rows.map((row) => header.map((key) => row[key]).join(','));
  const blob = new Blob([[header.join(','), ...body].join('\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'dashboard_predictions.csv';
  link.click();
  URL.revokeObjectURL(url);
}

export default function Predictions() {
  const { modelId, label } = useModel();
  const [filter, setFilter] = useState('all');
  const [query, setQuery] = useState('');
  const [onlyActionable, setOnlyActionable] = useState(false);

  const resource = useResource((options) => api.predictions(modelId, options), {
    key: modelKey(modelId, 'predictions'),
    pollMs: POLL_INTERVALS.artifacts,
  });

  const data = resource.data;
  const threshold = data?.confidence_threshold ?? 0.6;

  const rows = useMemo(() => {
    const all = data?.rows ?? [];
    const needle = query.trim().toUpperCase();
    return all.filter((row) => {
      if (filter !== 'all' && row.prediction !== filter) return false;
      if (onlyActionable && row.confidence < threshold) return false;
      if (needle && !row.symbol.toUpperCase().includes(needle)) return false;
      return true;
    });
  }, [data, filter, query, onlyActionable, threshold]);

  const aboveThreshold = (data?.rows ?? []).filter((row) => row.confidence >= threshold).length;

  return (
    <>
      <PageHeader
        eyebrow={`Predictions · ${label}`}
        title="Latest serving rows"
        description="The serving file is rewritten by the pipeline, not by this page. Rows are ranked by confidence; the date is the last feature row the model saw."
        actions={
          <>
            <SourceTag envelope={resource.envelope} />
            <Button size="sm" variant="quiet" icon={Download} onClick={() => downloadCsv(rows)} disabled={!rows.length}>
              CSV
            </Button>
          </>
        }
      />

      {resource.error && (
        <div className="mb-5">
          <ErrorNote error={resource.error} label="Predictions unavailable" onRetry={resource.refresh} />
        </div>
      )}
      {resource.envelope?.warnings?.length > 0 && (
        <div className="mb-5">
          <Warnings warnings={resource.envelope.warnings} />
        </div>
      )}

      <Section eyebrow="Headline" title="What the serving file holds">
        <StatGrid columns={4}>
          <Stat
            label="Universe"
            value={data?.universe ?? '—'}
            info={{
              what: 'How many symbols the model scores each session.',
              reads: 'The universe configured in config/config.yaml, not the number of rows that happen to be in the serving file.',
              ideal: 'Equal to the symbol count shown in the sidebar.',
            }}
          />
          <Stat
            label="Rows returned"
            value={int(data?.count)}
            info={{
              what: 'Scored rows carried in the serving file.',
              reads: 'One row per symbol for the last feature date. The page asks the API for at most 500 rows.',
              ideal: 'One row per symbol. Fewer means some symbols were dropped upstream.',
            }}
          />
          <Stat
            label="Mean confidence"
            value={pct(data?.mean_confidence)}
            info={{
              what: 'Average confidence — the largest of the three calibrated probabilities — across the scored rows.',
              reads: 'max(p) per row, averaged. Because it is a maximum it sits above one third by construction.',
              ideal: 'Roughly 0.5 to 0.6 at the current class mix. Mean confidence above 0.9 signals overconfidence, not accuracy.',
              when: 'Drifting towards 0.34 means the model has stopped separating the classes.',
            }}
          />
          <Stat
            label="Above threshold"
            value={int(aboveThreshold)}
            info={{
              what: 'Rows clearing the serving confidence threshold.',
              reads: `Count of rows at or above ${pct(threshold, 0)}, the same threshold the backtest applies.`,
              ideal: 'A minority of rows; the threshold exists to discard most candidates.',
              when: 'Close to zero above-threshold rows means the model offers nothing actionable this session.',
            }}
          />
        </StatGrid>
      </Section>

      <Section eyebrow="Serving rows" title="Every scored symbol">
        <div className="grid gap-6 lg:grid-cols-3">
          <Panel
            className="lg:col-span-2"
            title={`${rows.length} row(s)`}
            info={{
              what: 'One row per symbol with its calibrated class probabilities and the resulting signal.',
              reads: 'Filtered by the segmented control and the symbol search. An "act" badge marks rows that clear the threshold.',
              ideal: 'Confidences spread across the file, not one repeated value — repetition means the features are not moving.',
            }}
            flush
            actions={<Segmented label="Filter by signal" options={FILTERS} value={filter} onChange={setFilter} />}
          >
            <div className="flex flex-wrap items-center gap-2 border-b border-[var(--line)] px-4 py-3">
              <label className="relative min-w-[12rem] flex-1">
                <span className="sr-only">Search symbol</span>
                <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[var(--ink-faint)]" />
                <input
                  className="field pl-9"
                  placeholder="Filter symbol"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  spellCheck={false}
                />
              </label>
              <Button size="sm" variant={onlyActionable ? 'primary' : 'quiet'} aria-pressed={onlyActionable} onClick={() => setOnlyActionable((value) => !value)}>
                Confidence ≥ {pct(threshold, 0)}
              </Button>
            </div>

            {resource.loading && !data ? (
              <div className="p-4">
                <Loading rows={6} />
              </div>
            ) : (
              <Table
                columns={[
                  { key: 'symbol', header: 'Symbol', strong: true },
                  { key: 'date', header: 'As of', render: (row) => formatDate(row.date) },
                  { key: 'prediction', header: 'Signal', render: (row) => <SignalBadge signal={row.prediction} /> },
                  {
                    key: 'probabilities',
                    header: 'Distribution',
                    render: (row) => (
                      <div className="w-[190px]">
                        <ProbabilityBar pDown={row.p_down} pFlat={row.p_flat} pUp={row.p_up} />
                      </div>
                    ),
                  },
                  {
                    key: 'confidence',
                    header: 'Confidence',
                    align: 'right',
                    render: (row) => (
                      <span className="inline-flex items-center gap-2">
                        {row.confidence >= threshold && <Badge tone="badge-solid">act</Badge>}
                        <span className="mono">{pct(row.confidence)}</span>
                      </span>
                    ),
                  },
                ]}
                rows={rows}
                rowKey={(row) => `${row.symbol}-${row.date}`}
                empty={
                  <div className="p-4">
                    <Empty
                      title={data?.rows?.length ? 'No rows match the filter' : 'No predictions yet'}
                      detail={data?.rows?.length ? 'Clear the filter or lower the confidence requirement.' : 'Run the refresh pipeline to regenerate results/latest_predictions.csv.'}
                    />
                  </div>
                }
              />
            )}
          </Panel>

          <div className="space-y-6">
            <Panel
              title="Predicted label distribution"
              info={{
                what: 'How this session splits across DOWN, FLAT and UP.',
                reads: 'Share of scored rows per predicted class, using the calibrated argmax with the flat band applied.',
                ideal: 'Close to the training mix of roughly 48% DOWN, 5% FLAT and 47% UP. A single dominant bar means the model collapsed.',
                when: 'A FLAT majority means every row fell inside the band, so no directional call survives.',
              }}
            >
              <DistributionBars distribution={data?.distribution} />
              <div className="mt-5 space-y-3 border-t border-[var(--line)] pt-4">
                <div className="flex items-center justify-between gap-3">
                  <span className="muted">As of</span>
                  <span className="mono text-[.875rem]">{formatDate(resource.envelope?.as_of)}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="muted">Confidence threshold</span>
                  <span className="mono text-[.875rem]">{pct(threshold)}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="muted">Flat band</span>
                  <span className="mono text-[.875rem]">±{num(data?.flat_threshold, 4)}</span>
                </div>
              </div>
            </Panel>

            <Panel
              title="Decision rule"
              info={{
                what: 'The three rules that turn three probabilities into one signal.',
                reads: 'Calibrated probabilities, then a FLAT band, then a confidence threshold. Each rule can veto the one before it.',
              }}
            >
              <ul className="muted space-y-3">
                <li>
                  The label is the calibrated argmax with the flat band applied, so it can differ from a raw argmax.
                </li>
                <li>
                  Clearing the threshold makes a row a candidate, not a trade: the backtest applies the same threshold and still loses on some trades.
                </li>
                <li>
                  If the serving date lags the expected session, the cause is upstream. Run <code className="mono">.\refresh_data.ps1</code>; it rebuilds
                  features before it predicts.
                </li>
              </ul>
            </Panel>
          </div>
        </div>
      </Section>

      <Section eyebrow="Legend" title="Signal encoding">
        <Panel
          title="What each glyph means"
          info={{
            what: 'What the UP, DOWN and FLAT marks stand for.',
            reads: 'Each signal is the class with the highest calibrated probability after the flat band is applied.',
            ideal: 'Direction is never carried by colour alone: a glyph, a text label and a badge weight say the same thing, so the table survives greyscale and print.',
          }}
        >
          <div className="grid gap-4 sm:grid-cols-3">
            {['UP', 'DOWN', 'FLAT'].map((signal) => (
              <div key={signal} className="flex items-start gap-3 border-t border-[var(--line)] pt-3">
                <AgreementMark predicted={signal} actual={signal} />
                <p className="muted text-[.875rem]">
                  {signal === 'UP'
                    ? 'Probability of an up move beyond the flat band.'
                    : signal === 'DOWN'
                      ? 'Probability of a down move beyond the flat band.'
                      : 'Inside the flat band: no directional edge.'}
                </p>
              </div>
            ))}
          </div>
        </Panel>
      </Section>
    </>
  );
}
