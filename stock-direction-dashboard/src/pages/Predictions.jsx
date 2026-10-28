import { useMemo, useState } from 'react';
import { Download, Search } from 'lucide-react';
import { api } from '../lib/api';
import { POLL_INTERVALS } from '../lib/constants';
import { formatDate } from '../lib/dates';
import { int, num, pct } from '../lib/format';
import { useResource } from '../hooks/useResource';
import { DistributionBars } from '../components/charts';
import { AgreementMark, ProbabilityBar, SignalBadge } from '../components/Signal';
import {
  Badge,
  Button,
  Callout,
  Empty,
  ErrorNote,
  Loading,
  PageHeader,
  Panel,
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
  const [filter, setFilter] = useState('all');
  const [query, setQuery] = useState('');
  const [onlyActionable, setOnlyActionable] = useState(false);

  const resource = useResource((options) => api.predictions({ ...options, params: { limit: 500 } }), {
    key: 'predictions',
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
        eyebrow="Predictions"
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
        <div className="mb-4">
          <ErrorNote error={resource.error} label="Predictions unavailable" onRetry={resource.refresh} />
        </div>
      )}
      {resource.envelope?.warnings?.length > 0 && (
        <div className="mb-4">
          <Warnings warnings={resource.envelope.warnings} />
        </div>
      )}

      <StatGrid columns={4}>
        <Stat label="Universe" value={data?.universe ?? '—'} hint="Configured symbols" />
        <Stat label="Rows returned" value={int(data?.count)} hint={`of ${int(data?.rows?.length)} scored`} />
        <Stat label="Mean confidence" value={pct(data?.mean_confidence)} hint={`Confidence threshold ${pct(threshold, 0)}`} />
        <Stat
          label="Above threshold"
          value={int(aboveThreshold)}
          hint={`${pct(data?.rows?.length ? aboveThreshold / data.rows.length : null, 1)} of scored rows`}
          footnote={`Flat band ±${num(data?.flat_threshold, 4)}`}
        />
      </StatGrid>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Panel className="lg:col-span-2" eyebrow="Signals" title={`${rows.length} row(s)`} flush actions={<Segmented label="Filter by signal" options={FILTERS} value={filter} onChange={setFilter} />}>
          <div className="flex flex-wrap items-center gap-2 border-b border-[var(--line)] px-4 py-2.5">
            <label className="relative flex-1 min-w-[12rem]">
              <span className="sr-only">Search symbol</span>
              <Search size={13} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--ink-faint)]" />
              <input
                className="field pl-7"
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

        <div className="space-y-4">
          <Panel eyebrow="Mix" title="Predicted label distribution">
            <DistributionBars distribution={data?.distribution} />
            <div className="mt-4 space-y-2 border-t border-[var(--line)] pt-3">
              <div className="flex items-center justify-between">
                <span className="muted">As of</span>
                <span className="mono text-[.75rem]">{formatDate(resource.envelope?.as_of)}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="muted">Confidence threshold</span>
                <span className="mono text-[.75rem]">{pct(threshold)}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="muted">Flat band</span>
                <span className="mono text-[.75rem]">±{num(data?.flat_threshold, 4)}</span>
              </div>
            </div>
          </Panel>

          <Panel eyebrow="How to read this" title="Decision rule">
            <ul className="muted space-y-2">
              <li>
                Probabilities are temperature-calibrated. The label comes from the calibrated argmax with a FLAT band, so it can differ from a raw
                argmax.
              </li>
              <li>
                A row above the confidence threshold is only a candidate: the backtest applies the same threshold and still loses money on some
                trades.
              </li>
              <li>
                If the serving date lags the expected session, the cause is upstream. Re-running prediction alone cannot advance the date because it
                reads the last feature row.
              </li>
            </ul>
            <div className="mt-3">
              <Callout tone="warn" title="Stale serving file?">
                Run <code className="mono">.\refresh_data.ps1</code>. It rebuilds features before it predicts.
              </Callout>
            </div>
          </Panel>
        </div>
      </div>

      <Panel className="mt-4" eyebrow="Legend" title="Signal encoding">
        <div className="grid gap-3 sm:grid-cols-3">
          {['UP', 'DOWN', 'FLAT'].map((signal) => (
            <div key={signal} className="flex items-center gap-3 border-t border-[var(--line)] pt-2">
              <AgreementMark predicted={signal} actual={signal} />
              <p className="muted text-[.75rem]">
                {signal === 'UP' ? 'Probability of an up move beyond the flat band.' : signal === 'DOWN' ? 'Probability of a down move beyond the flat band.' : 'Inside the flat band: no directional edge.'}
              </p>
            </div>
          ))}
        </div>
        <p className="mt-3 text-[.6875rem] text-[var(--ink-faint)]">
          Direction is never encoded with colour. Each signal carries a glyph, a text label and a distinct badge weight, so the table stays readable
          in greyscale and in print.
        </p>
      </Panel>
    </>
  );
}
