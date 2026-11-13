import { useMemo, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import { api } from '../lib/api';
import { POLL_INTERVALS } from '../lib/constants';
import { formatDate } from '../lib/dates';
import { int, num, pct, signedPct } from '../lib/format';
import { useResource } from '../hooks/useResource';
import { useDashboard } from '../state/DashboardProvider';
import { modelKey, useModel } from '../state/ModelProvider';
import { EquityChart, ChartLegend } from '../components/charts';
import { AgreementMark } from '../components/Signal';
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

const OUTCOMES = [
  { value: 'all', label: 'All' },
  { value: 'winners', label: 'Winners' },
  { value: 'losers', label: 'Losers' },
];

/**
 * Hit and miss counts under the trade table.
 *
 * The counts come from the API's `summary`, which is computed before the row
 * limit, so they describe every trade that matches the filter rather than the
 * page of rows on screen. A hit is the same thing the table's Result column
 * calls a hit: the predicted class matched the realised label. It is not the
 * same as a profitable trade, which is why the net-positive count is shown
 * beside it and can differ.
 */
function TradeTally({ summary }) {
  if (!summary?.total) return null;
  const blocked = summary.returned < summary.total;
  return (
    <div className="flex flex-wrap items-end gap-x-8 gap-y-3 border-t border-[var(--line)] px-4 py-3.5">
      <div>
        <p className="eyebrow">Hits</p>
        <p className="kpi-sm mono mt-1">{int(summary.hits)}</p>
      </div>
      <div>
        <p className="eyebrow">Misses</p>
        <p className="kpi-sm mono mt-1">{int(summary.misses)}</p>
      </div>
      <div>
        <p className="eyebrow">Hit rate</p>
        <p className="kpi-sm mono mt-1">{pct(summary.hit_rate, 1)}</p>
      </div>
      <div>
        <p className="eyebrow">Miss rate</p>
        <p className="kpi-sm mono mt-1">{pct(summary.miss_rate, 1)}</p>
      </div>
      <p className="muted mb-1 ml-auto text-[.8125rem]">
        Over {int(summary.total)} trade(s) matching the filter
        {blocked ? ` · showing the latest ${int(summary.returned)}` : ''}.
      </p>
    </div>
  );
}

export default function Backtesting() {
  const { symbols } = useDashboard();
  const { modelId, label } = useModel();
  const [outcome, setOutcome] = useState('all');
  const [symbol, setSymbol] = useState('');

  const backtest = useResource((options) => api.backtest(modelId, options), {
    key: modelKey(modelId, 'backtest'),
    pollMs: POLL_INTERVALS.artifacts,
  });
  const trades = useResource(
    (options) => api.backtestTrades(modelId, { symbol: symbol || undefined, outcome, limit: 200 }, options),
    { key: modelKey(modelId, `trades-${symbol}-${outcome}`), pollMs: POLL_INTERVALS.artifacts },
  );

  const data = backtest.data;
  const metrics = data?.metrics ?? {};
  const curve = data?.curve ?? [];

  const hitRate = useMemo(() => {
    const rows = data?.per_symbol ?? [];
    const weighted = rows.reduce((sum, row) => sum + (row.hit_rate ?? 0) * row.trades, 0);
    const total = rows.reduce((sum, row) => sum + row.trades, 0);
    return total ? weighted / total : null;
  }, [data]);

  return (
    <>
      <PageHeader
        eyebrow={`Backtesting · ${label}`}
        title="Net event backtest"
        description="Each event is traded from the decision close to the barrier exit and charged the configured round-trip cost. Overlapping positions are not taken per symbol."
        actions={
          <>
            <SourceTag envelope={backtest.envelope} />
            <Button size="sm" variant="quiet" icon={RefreshCw} onClick={() => { backtest.refresh(); trades.refresh(); }}>
              Reload
            </Button>
          </>
        }
      />

      {backtest.error && (
        <div className="mb-5">
          <ErrorNote error={backtest.error} label="Backtest unavailable" onRetry={backtest.refresh} />
        </div>
      )}
      {backtest.envelope?.warnings?.length > 0 && (
        <div className="mb-5">
          <Warnings warnings={backtest.envelope.warnings} />
        </div>
      )}

      <Section eyebrow="Headline" title="What the strategy earned after costs">
        <StatGrid columns={4}>
          <Stat
            label="Total return"
            value={signedPct(metrics.total_return)}
            info={{
              what: 'What the strategy made or lost over the backtest window, after every cost.',
              reads: `Gross ${signedPct(metrics.gross_return_sum)} minus ${pct(metrics.cost_paid)} of costs, on the ${metrics.segment ?? 'unrecorded'} segment at ${num(metrics.cost_bps_round_trip, 1)} bps per round trip.`,
              ideal: 'Positive and climbing. At a 31.5 bps round trip on a 1.6% typical move, the strategy needs roughly 57% directional accuracy just to break even.',
              when: 'Positive gross with negative net means the signal exists but is smaller than the cost of trading it.',
            }}
          />
          <Stat
            label="Max drawdown"
            value={pct(metrics.max_drawdown)}
            info={{
              what: 'The worst peak-to-trough fall of the equity curve.',
              reads: 'Measured on the daily equity curve, not on the weekly points drawn below.',
              ideal: 'Shallow relative to total return. A drawdown as large as the gain means one bad stretch decides the result.',
            }}
          />
          <Stat
            label="Sharpe"
            value={num(metrics.sharpe, 2)}
            info={{
              what: 'Return per unit of volatility over the backtest.',
              reads: 'Computed on the daily equity series, and not annualised unless the artifact says so.',
              ideal: 'Above 1 is respectable for a daily equity series; between 0 and 1 is thin but not meaningless.',
              when: 'Negative means the curve is going down. A very high value on few trades is noise.',
            }}
          />
          <Stat
            label="Trades"
            value={int(metrics.trade_count)}
            info={{
              what: 'How many events were actually traded.',
              reads: `Hit rate ${pct(hitRate, 1)}, over ${int(data?.trading_days)} trading days of which ${int(data?.positive_days)} were positive.`,
              ideal: 'Enough trades that the result is not one lucky event. A few dozen trades cannot distinguish an edge from chance.',
            }}
          />
        </StatGrid>
      </Section>

      <Section eyebrow="Equity" title="The curve over time">
        <Panel
          title="Cumulative net equity"
          info={{
            what: 'What one unit of capital would be worth after each event, costs included.',
            reads: 'Equity starts at 1.0 on the first event and compounds every trade. Drawn weekly from the daily series, with drawdown shaded underneath. The window is the line printed in the note at the bottom of the panel.',
            ideal: 'A rising line with shallow dips. A saw-tooth that ends flat is the shape of an edge eaten by costs.',
            when: 'A curve that rises then gives everything back usually means the later trades were taken in a regime the model did not learn.',
          }}
          actions={<ChartLegend items={[{ label: 'Equity' }, { label: 'Drawdown', color: '#6f6f68', opacity: 0.9 }]} />}
        >
          {backtest.loading && !data ? (
            <Loading rows={5} />
          ) : curve.length ? (
            <>
              <EquityChart data={curve} height={320} />
              <p className="muted mt-4 text-[.875rem]">
                {formatDate(data?.start)} → {formatDate(data?.end)} · {curve.length} weekly points reconstructed from{' '}
                {int(data?.equity_points)} trading days.
              </p>
            </>
          ) : (
            <Empty title="No equity curve" detail="results/backtest_timeseries.parquet has no usable events." />
          )}
        </Panel>
      </Section>

      <Section eyebrow="Attribution" title="Where the return came from">
        <div className="grid gap-6 lg:grid-cols-3">
          <Panel
            className="lg:col-span-2"
            title="Per symbol"
            info={{
              what: 'Net return contributed by each symbol.',
              reads: 'Trades, hit rate and net return per symbol over the whole window. The bar is scaled against the largest absolute net return.',
              ideal: 'Gains spread across many names. One symbol carrying the whole result is a concentration you would not reproduce live.',
            }}
            flush
          >
            <Table
              dense
              columns={[
                { key: 'symbol', header: 'Symbol', strong: true },
                { key: 'trades', header: 'Trades', align: 'right', render: (row) => int(row.trades) },
                { key: 'hit_rate', header: 'Hit rate', align: 'right', render: (row) => pct(row.hit_rate, 1) },
                { key: 'net_return', header: 'Net return', align: 'right', render: (row) => signedPct(row.net_return) },
                {
                  key: 'bar',
                  header: '',
                  render: (row) => {
                    const max = Math.max(...(data?.per_symbol ?? []).map((item) => Math.abs(item.net_return)), 1e-9);
                    const width = (Math.abs(row.net_return) / max) * 100;
                    return (
                      <div className="h-1.5 w-24 bg-[var(--surface-sunken)]" title={`${row.symbol} net ${pct(row.net_return)}`}>
                        <div className="h-full bg-[var(--ink)]" style={{ width: `${width}%` }} />
                      </div>
                    );
                  },
                },
              ]}
              rows={data?.per_symbol ?? []}
              rowKey={(row) => row.symbol}
              empty={<div className="p-4"><Empty title="No per-symbol breakdown" /></div>}
            />
          </Panel>

          <div className="space-y-6">
            <Panel
              title="Where the return goes"
              info={{
                what: 'Gross return, the cost paid to capture it, and what is left.',
                reads: 'Gross return sum minus cost paid equals the net return sum, per the cost model below.',
                ideal: 'Net positive, with cost a small share of gross. Cost above gross means the strategy is paying more to trade than the signal earns.',
                when: 'Gross positive but net negative: there is no edge to spend. Fixing the model is the only lever, because lowering the assumed cost would hide the problem.',
              }}
            >
              <KeyValue
                columns={1}
                items={[
                  { label: 'Gross return sum', value: signedPct(metrics.gross_return_sum), mono: true },
                  { label: 'Cost paid', value: pct(metrics.cost_paid), mono: true },
                  { label: 'Net return sum', value: signedPct(metrics.net_return_sum), mono: true },
                  { label: 'Round-trip cost', value: `${num(metrics.cost_bps_round_trip, 1)} bps`, mono: true },
                ]}
              />
            </Panel>

            <Panel
              title="Cost model"
              info={{
                what: 'The per-trade assumptions the net figures are computed from.',
                reads: 'Commission, slippage and other costs in basis points, plus the overlap policy applied per symbol.',
                ideal: 'Assumptions at or above what a delivery order really costs. Understating them flatters every number on this page.',
              }}
            >
              <KeyValue
                columns={1}
                items={[
                  { label: 'Commission', value: `${num(data?.cost_model?.commission_bps, 1)} bps`, mono: true },
                  { label: 'Slippage', value: `${num(data?.cost_model?.slippage_bps, 1)} bps`, mono: true },
                  { label: 'Other', value: `${num(data?.cost_model?.other_cost_bps, 1)} bps`, mono: true },
                  { label: 'Overlap policy', value: data?.cost_model?.overlap_policy ?? '—' },
                ]}
              />
            </Panel>
          </div>
        </div>
      </Section>

      <Section eyebrow="Trades" title="Every event, one row">
        <Panel
          title={`${int(trades.data?.summary?.total ?? trades.data?.count ?? 0)} event(s)`}
          info={{
            what: 'Each event that was traded, from entry to barrier exit.',
            reads: 'Entry and exit prices, gross and net return, and whether the predicted class matched the realised label. Filter by symbol or by outcome. The tally under the table counts every trade the filter matches, not only the rows on screen.',
            ideal: 'Hits whose net return is positive. A hit with a negative net return means the move was real but too small to pay the round trip.',
          }}
          flush
          actions={
            <div className="flex flex-wrap items-center gap-2.5">
              <select className="field w-[11rem]" value={symbol} onChange={(event) => setSymbol(event.target.value)}>
                <option value="">All symbols</option>
                {symbols.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
              <Segmented label="Outcome filter" options={OUTCOMES} value={outcome} onChange={setOutcome} />
            </div>
          }
        >
          {trades.error ? (
            <div className="p-4">
              <ErrorNote error={trades.error} label="Trades unavailable" onRetry={trades.refresh} />
            </div>
          ) : (
            <>
              <Table
                dense
                columns={[
                  { key: 'event_start', header: 'Entry', render: (row) => formatDate(row.event_start) },
                  { key: 'event_end', header: 'Exit', render: (row) => formatDate(row.event_end) },
                  { key: 'symbol', header: 'Symbol', strong: true },
                  { key: 'labels', header: 'Predicted vs realised', render: (row) => <AgreementMark predicted={row.predicted_label} actual={row.actual_label} /> },
                  { key: 'entry_price', header: 'Entry', align: 'right', render: (row) => num(row.entry_price, 2) },
                  { key: 'exit_price', header: 'Exit', align: 'right', render: (row) => num(row.exit_price, 2) },
                  { key: 'gross_return', header: 'Gross', align: 'right', render: (row) => signedPct(row.gross_return) },
                  { key: 'cost', header: 'Cost', align: 'right', render: (row) => pct(row.cost) },
                  {
                    key: 'net_return',
                    header: 'Net',
                    align: 'right',
                    render: (row) => <span className="mono font-semibold">{signedPct(row.net_return)}</span>,
                  },
                  {
                    key: 'correct',
                    header: 'Result',
                    render: (row) => <Badge tone={row.correct ? 'badge-solid' : 'badge-dashed'}>{row.correct ? 'hit' : 'miss'}</Badge>,
                  },
                ]}
                rows={trades.data?.rows ?? []}
                rowKey={(row) => `${row.symbol}-${row.event_start}-${row.event_end}`}
                empty={<div className="p-4"><Empty title="No trades match the filter" /></div>}
              />
              {!trades.loading && <TradeTally summary={trades.data?.summary} />}
            </>
          )}
        </Panel>
      </Section>
    </>
  );
}
