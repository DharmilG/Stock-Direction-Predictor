import { useMemo, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import { api } from '../lib/api';
import { POLL_INTERVALS } from '../lib/constants';
import { formatDate } from '../lib/dates';
import { compact, int, num, pct } from '../lib/format';
import { useResource } from '../hooks/useResource';
import { useDashboard } from '../state/DashboardProvider';
import { modelKey, useModel } from '../state/ModelProvider';
import { LineSeriesChart } from '../components/charts';
import { AgreementMark, ProbabilityBar, SignalBadge } from '../components/Signal';
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

const RANGES = [
  { value: 90, label: '90' },
  { value: 180, label: '180' },
  { value: 360, label: '360' },
];

export default function StockAnalysis() {
  const { symbols } = useDashboard();
  const { modelId, label } = useModel();
  const [symbol, setSymbol] = useState('');
  const [points, setPoints] = useState(180);
  const [indicator, setIndicator] = useState('');

  const activeSymbol = symbol || symbols[0] || '';
  const stock = useResource((options) => api.stock(modelId, activeSymbol, points, options), {
    key: modelKey(modelId, `stock-${activeSymbol}-${points}`),
    enabled: Boolean(activeSymbol),
    pollMs: POLL_INTERVALS.artifacts,
  });
  const history = useResource((options) => api.stockPredictions(modelId, activeSymbol, 120, options), {
    key: modelKey(modelId, `stock-pred-${activeSymbol}`),
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
        eyebrow={`Stock analysis · ${label}`}
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

      <div className="mb-6 flex flex-wrap items-center gap-x-6 gap-y-3">
        <label className="flex items-center gap-2.5">
          <span className="eyebrow">Symbol</span>
          <select className="field w-[13rem]" value={activeSymbol} onChange={(event) => setSymbol(event.target.value)}>
            {symbols.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <div className="flex items-center gap-2.5">
          <span className="eyebrow">Sessions</span>
          <Segmented label="History length" options={RANGES} value={points} onChange={setPoints} />
        </div>
      </div>

      {stock.error && (
        <div className="mb-5">
          <ErrorNote error={stock.error} label="Symbol data unavailable" onRetry={stock.refresh} />
        </div>
      )}
      {stock.envelope?.warnings?.length > 0 && (
        <div className="mb-5">
          <Warnings warnings={stock.envelope.warnings} />
        </div>
      )}

      <Section eyebrow="Headline" title="The window you are looking at">
        <StatGrid columns={4}>
          <Stat
            label="Last close"
            value={num(lastClose, 2)}
            info={{
              what: 'The most recent closing price stored for this symbol, in rupees.',
              reads: `Last row of the ${points}-session window read from data/raw.`,
              ideal: 'Dated within one session of the expected session. An older date means the raw file is behind.',
            }}
          />
          <Stat
            label="Change in window"
            value={change == null ? '—' : `${change > 0 ? '+' : ''}${pct(change)}`}
            info={{
              what: 'How much the close moved from the first session to the last in this window.',
              reads: `Last close divided by the first close of the ${points} sessions on screen, minus one.`,
              ideal: 'Any value is normal; what matters is whether it comes from one jump or a gradual trend. Check the price chart for that.',
            }}
          />
          <Stat
            label="Stored bars"
            value={int(data?.rows)}
            info={{
              what: 'How many daily rows exist on disk for this symbol, in total.',
              reads: 'Row count of the symbol parquet in data/raw, not the number drawn on the chart.',
              ideal: 'Roughly one row per trading session since listing. A count well below that means sessions are missing, and a missing session is a gap, not a zero return.',
            }}
          />
          <Stat
            label="Latest signal"
            value={data?.latest_prediction ? <SignalBadge signal={data.latest_prediction.prediction} /> : '—'}
            info={{
              what: 'What the model currently says about this symbol.',
              reads: 'The serving row for this symbol in results/latest_predictions.csv, using the date shown in the Serving panel.',
              ideal: 'Confidence well above one third; see the distribution in the Serving panel for the full picture.',
              when: 'No row means the symbol is not in the serving file, which is a pipeline issue rather than a market view.',
            }}
          />
        </StatGrid>
      </Section>

      <Section eyebrow="Price" title="What the stock did">
        <Panel
          title="Close price"
          info={{
            what: 'The daily closing price of this symbol, in rupees, over the selected window.',
            reads: 'One point per stored session from data/raw. The sessions selector changes how many points are drawn, not the data behind them.',
            ideal: 'A continuous line with no vertical jumps. A jump is usually a split or an adjusted-price artefact; a long flat stretch is usually holidays stored as rows.',
          }}
          actions={<Badge tone="badge-ghost">features as of {formatDate(data?.features_as_of, { withYear: false })}</Badge>}
        >
          {stock.loading && !data ? (
            <Loading rows={5} />
          ) : priceSeries.length ? (
            <LineSeriesChart
              data={priceSeries}
              series={[{ key: 'close', label: 'Close', width: 2 }]}
              height={300}
              yTickFormatter={(value) => num(value, 0)}
            />
          ) : (
            <Empty title="No price rows" detail="data/raw has no parquet for this symbol." />
          )}
        </Panel>
      </Section>

      <Section eyebrow="Model input" title="The features behind the signal">
        <div className="grid gap-6 lg:grid-cols-3">
          <Panel
            className="lg:col-span-2"
            title={activeIndicator ? activeIndicator : 'No indicator available'}
            info={{
              what: 'One input column the model actually consumes, plotted over the same window as the price.',
              reads: 'Values from this symbol\'s feature parquet. Columns that are constant or strictly monotonic, such as a row counter, are hidden from the selector.',
              ideal: 'Moves with the price in a way you can explain. A feature that is flat for months carries no information, and one that only trends is a backdoor clock.',
            }}
            actions={
              names.length ? (
                <select className="field w-[15rem]" value={activeIndicator} onChange={(event) => setIndicator(event.target.value)}>
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
                height={260}
                yTickFormatter={(value) => num(value, Math.abs(value) < 10 ? 2 : 0)}
              />
            ) : (
              <Empty title="Feature series unavailable" detail="The feature file for this symbol has not been built yet." />
            )}
          </Panel>

          <div className="space-y-6">
            <Panel
              title="Latest prediction"
              info={{
                what: 'The model\'s view of this symbol on the most recent scored session.',
                reads: 'Three calibrated probabilities after the flat band is applied, plus the confidence, which is the largest of the three.',
                ideal: 'One probability clearly ahead of the other two. Three values near 0.33 mean the model has no opinion about this name today.',
                when: 'If the date here lags the price chart, the serving file is stale and the cause is upstream.',
              }}
            >
              {data?.latest_prediction ? (
                <>
                  <div className="mb-4 flex items-center justify-between">
                    <SignalBadge signal={data.latest_prediction.prediction} />
                    <span className="mono text-[.875rem] text-[var(--ink-3)]">{formatDate(data.latest_prediction.date)}</span>
                  </div>
                  <ProbabilityBar
                    pDown={data.latest_prediction.p_down}
                    pFlat={data.latest_prediction.p_flat}
                    pUp={data.latest_prediction.p_up}
                    height={10}
                  />
                  <div className="mt-4">
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

            <Panel
              title="Prediction accuracy"
              info={{
                what: 'How often the model was right about this one symbol once the outcome was known.',
                reads: `Share of scored predictions in the last 120 sessions where the predicted class matched the realised label. The panel reports ${history.data?.count ?? 0} scored events between the dates shown.`,
                ideal: 'Compare it with the overall test accuracy and with the class mix; a handful of events swings this number wildly.',
                when: 'A very low value on few events is noise, not evidence. Labels also trail features by the barrier horizon, so the newest sessions have no outcome yet.',
              }}
            >
              {history.data ? (
                <>
                  <p className="kpi-sm num">{pct(history.data.accuracy)}</p>
                  <p className="muted mt-2 text-[.875rem]">
                    {formatDate(history.data.start)} → {formatDate(history.data.end)} · {history.data.count} scored events
                  </p>
                </>
              ) : (
                <p className="muted">No evaluation history recorded for this symbol.</p>
              )}
            </Panel>
          </div>
        </div>
      </Section>

      <Section eyebrow="History" title="Every scored prediction">
        <Panel
          title="Prediction against outcome"
          info={{
            what: 'Each past prediction for this symbol, next to what actually happened.',
            reads: 'Newest first. "hit" means the predicted class matched the realised label; the Realised column is the triple-barrier return in percent.',
            ideal: 'Hits spread across time rather than clustered in one bullish stretch, and realised returns whose sign agrees with the signal.',
            when: 'A long run of misses on one signal means the model is systematically wrong about this name in this regime.',
          }}
          flush
        >
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
      </Section>

      <Section eyebrow="Activity" title="How much it traded">
        <Panel
          title="Traded volume"
          info={{
            what: 'How many shares changed hands each session.',
            reads: 'Daily traded quantity from the same parquet as the close price, on a compact scale.',
            ideal: 'Volume should spike on the days the price makes its largest moves. A big move on ordinary volume is often a data artefact rather than a real repricing.',
          }}
        >
          {priceSeries.length ? (
            <LineSeriesChart
              data={priceSeries}
              series={[{ key: 'volume', label: 'Volume', width: 1.5 }]}
              height={220}
              yTickFormatter={(value) => compact(value)}
            />
          ) : (
            <Empty />
          )}
        </Panel>
      </Section>
    </>
  );
}
