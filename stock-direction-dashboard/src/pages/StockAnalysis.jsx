import { useMemo, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import { api } from '../lib/api';
import { POLL_INTERVALS } from '../lib/constants';
import { formatDate } from '../lib/dates';
import { compact, int, num, pct } from '../lib/format';
import { useResource } from '../hooks/useResource';
import { useDashboard } from '../state/DashboardProvider';
import { LineSeriesChart } from '../components/charts';
import { AgreementMark, ProbabilityBar, SignalBadge } from '../components/Signal';
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
  Segmented,
  SourceTag,
  Stat,
  StatGrid,
  Table,
  Warnings,
} from '../components/ui';

const RANGES = [
  { value: 90, label: '90' },
  { value: 180, label: '180' },
  { value: 360, label: '360' },
];

export default function StockAnalysis() {
  const { symbols } = useDashboard();
  const [symbol, setSymbol] = useState('');
  const [points, setPoints] = useState(180);
  const [indicator, setIndicator] = useState('');

  const activeSymbol = symbol || symbols[0] || '';
  const stock = useResource((options) => api.stock(activeSymbol, points, options), {
    key: `stock-${activeSymbol}-${points}`,
    enabled: Boolean(activeSymbol),
    pollMs: POLL_INTERVALS.artifacts,
  });
  const history = useResource((options) => api.stockPredictions(activeSymbol, 120, options), {
    key: `stock-pred-${activeSymbol}`,
    enabled: Boolean(activeSymbol),
    pollMs: POLL_INTERVALS.artifacts,
  });

  const data = stock.data;
  const predictionRows = history.data?.rows ?? [];
  const names = data?.indicator_names ?? [];
  const activeIndicator = indicator && names.includes(indicator) ? indicator : names[0];

  const priceSeries = useMemo(
    () => (data?.price ?? []).map((row) => ({ date: row.date, close: row.close, volume: row.volume })),
    [data],
  );

  const indicatorSeries = useMemo(() => {
    if (!data?.indicators?.[activeIndicator]) return [];
    return data.indicators[activeIndicator].map((row) => ({ date: row.date, value: row.value }));
  }, [data, activeIndicator]);

  const lastClose = priceSeries.length ? priceSeries[priceSeries.length - 1].close : null;
  const firstClose = priceSeries.length ? priceSeries[0].close : null;
  const change = lastClose != null && firstClose ? lastClose / firstClose - 1 : null;

  return (
    <>
      <PageHeader
        eyebrow="Stock analysis"
        title={activeSymbol || 'Pick a symbol'}
        description="Price series, the features the model actually consumes, and the realised outcome of every prediction made for this symbol."
        actions={
          <>
            <SourceTag envelope={stock.envelope} />
            <Button size="sm" variant="quiet" icon={RefreshCw} onClick={() => { stock.refresh(); history.refresh(); }}>
              Reload
            </Button>
          </>
        }
      />

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-2">
          <span className="eyebrow">Symbol</span>
          <select className="field w-[13rem]" value={activeSymbol} onChange={(event) => setSymbol(event.target.value)}>
            {symbols.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <div className="flex items-center gap-2">
          <span className="eyebrow">Sessions</span>
          <Segmented label="History length" options={RANGES} value={points} onChange={setPoints} />
        </div>
      </div>

      {stock.error && (
        <div className="mb-4">
          <ErrorNote error={stock.error} label="Symbol data unavailable" onRetry={stock.refresh} />
        </div>
      )}
      {stock.envelope?.warnings?.length > 0 && (
        <div className="mb-4">
          <Warnings warnings={stock.envelope.warnings} />
        </div>
      )}

      <StatGrid columns={4}>
        <Stat label="Last close" value={num(lastClose, 2)} hint={formatDate(data?.price_as_of)} />
        <Stat label="Change in window" value={change == null ? '—' : `${change > 0 ? '+' : ''}${pct(change)}`} hint={`${priceSeries.length} sessions shown`} />
        <Stat label="Stored bars" value={int(data?.rows)} hint="data/raw parquet rows" />
        <Stat
          label="Latest signal"
          value={data?.latest_prediction ? <SignalBadge signal={data.latest_prediction.prediction} /> : '—'}
          hint={data?.latest_prediction ? `${pct(data.latest_prediction.confidence)} confidence` : 'No serving row for this symbol'}
        />
      </StatGrid>

      <Panel
        className="mt-4"
        eyebrow="Price"
        title="Close price"
        actions={<Badge tone="badge-ghost">features as of {formatDate(data?.features_as_of, { withYear: false })}</Badge>}
      >
        {stock.loading && !data ? (
          <Loading rows={5} />
        ) : priceSeries.length ? (
          <LineSeriesChart
            data={priceSeries}
            series={[{ key: 'close', label: 'Close', width: 2 }]}
            height={280}
            yTickFormatter={(value) => num(value, 0)}
          />
        ) : (
          <Empty title="No price rows" detail="data/raw has no parquet for this symbol." />
        )}
      </Panel>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Panel
          className="lg:col-span-2"
          eyebrow="Features"
          title={activeIndicator ? activeIndicator : 'No indicator available'}
          actions={
            names.length ? (
              <select className="field w-[13rem]" value={activeIndicator} onChange={(event) => setIndicator(event.target.value)}>
                {names.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            ) : null
          }
        >
          {indicatorSeries.length ? (
            <LineSeriesChart
              data={indicatorSeries}
              series={[{ key: 'value', label: activeIndicator, width: 1.75 }]}
              height={240}
              yTickFormatter={(value) => num(value, Math.abs(value) < 10 ? 2 : 0)}
            />
          ) : (
            <Empty title="Feature series unavailable" detail="The feature file for this symbol has not been built yet." />
          )}
          <p className="mt-3 text-[.6875rem] text-[var(--ink-faint)]">
            Only monotonic or constant columns are hidden; every other feature the model consumes is listed in the selector.
          </p>
        </Panel>

        <div className="space-y-4">
          <Panel eyebrow="Serving" title="Latest prediction">
            {data?.latest_prediction ? (
              <>
                <div className="mb-3 flex items-center justify-between">
                  <SignalBadge signal={data.latest_prediction.prediction} />
                  <span className="mono text-[.7rem] text-[var(--ink-3)]">{formatDate(data.latest_prediction.date)}</span>
                </div>
                <ProbabilityBar
                  pDown={data.latest_prediction.p_down}
                  pFlat={data.latest_prediction.p_flat}
                  pUp={data.latest_prediction.p_up}
                  height={10}
                />
                <div className="mt-3">
                  <KeyValue
                    columns={1}
                    items={[
                      { label: 'Confidence', value: pct(data.latest_prediction.confidence), mono: true },
                      { label: 'P(up)', value: pct(data.latest_prediction.p_up), mono: true },
                      { label: 'P(down)', value: pct(data.latest_prediction.p_down), mono: true },
                    ]}
                  />
                </div>
              </>
            ) : (
              <Empty title="No serving row" detail="results/latest_predictions.csv has no row for this symbol." />
            )}
          </Panel>

          <Panel eyebrow="Realised" title="Prediction accuracy">
            {history.data ? (
              <>
                <Stat label="Accuracy in window" value={pct(history.data.accuracy)} hint={`${history.data.count} scored events`} />
                <p className="muted mt-2 text-[.6875rem]">
                  {formatDate(history.data.start)} → {formatDate(history.data.end)}
                </p>
              </>
            ) : (
              <p className="muted">No evaluation history recorded for this symbol.</p>
            )}
          </Panel>
        </div>
      </div>

      <Panel className="mt-4" eyebrow="History" title="Every scored prediction" flush>
        {history.error ? (
          <div className="p-4">
            <ErrorNote error={history.error} label="History unavailable" onRetry={history.refresh} />
          </div>
        ) : (
          <Table
            columns={[
              { key: 'date', header: 'Decision date', render: (row) => formatDate(row.date) },
              { key: 'labels', header: 'Predicted vs realised', render: (row) => <AgreementMark predicted={row.predicted} actual={row.actual} /> },
              { key: 'p_up', header: 'P(up)', align: 'right', render: (row) => num(row.p_up, 3) },
              { key: 'p_flat', header: 'P(flat)', align: 'right', render: (row) => num(row.p_flat, 3) },
              { key: 'p_down', header: 'P(down)', align: 'right', render: (row) => num(row.p_down, 3) },
              {
                key: 'realized_return',
                header: 'Realised',
                align: 'right',
                render: (row) => (row.realized_return == null ? '—' : `${row.realized_return > 0 ? '+' : ''}${pct(row.realized_return)}`),
              },
              {
                key: 'correct',
                header: 'Result',
                render: (row) => <Badge tone={row.correct ? 'badge-solid' : 'badge-dashed'}>{row.correct ? 'hit' : 'miss'}</Badge>,
              },
            ]}
            rows={[...predictionRows].reverse()}
            rowKey={(row) => row.date}
            empty={
              <div className="p-4">
                <Empty title="No scored predictions" detail="results/predictions.parquet has no rows for this symbol." />
              </div>
            }
          />
        )}
      </Panel>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel eyebrow="Volume" title="Traded volume">
          {priceSeries.length ? (
            <LineSeriesChart
              data={priceSeries}
              series={[{ key: 'volume', label: 'Volume', width: 1.5 }]}
              height={200}
              yTickFormatter={(value) => compact(value)}
            />
          ) : (
            <Empty />
          )}
        </Panel>
        <Panel eyebrow="Caveat" title="What this page will not tell you">
          <div className="space-y-2">
            <Callout tone="info">
              A missing session is a data gap, not a zero return. The pipeline repairs holes with a short re-fetch; see Data &amp; integration.
            </Callout>
            <Callout tone="warn">
              Labels trail features by the triple-barrier horizon, so the newest few sessions legitimately have no realised label yet.
            </Callout>
          </div>
        </Panel>
      </div>
    </>
  );
}
