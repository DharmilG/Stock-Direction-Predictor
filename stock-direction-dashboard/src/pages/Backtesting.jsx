import { useMemo, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import { api } from '../lib/api';
import { POLL_INTERVALS } from '../lib/constants';
import { formatDate } from '../lib/dates';
import { int, num, pct, signedPct } from '../lib/format';
import { useResource } from '../hooks/useResource';
import { useDashboard } from '../state/DashboardProvider';
import { EquityChart, ChartLegend } from '../components/charts';
import { AgreementMark } from '../components/Signal';
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

const OUTCOMES = [
  { value: 'all', label: 'All' },
  { value: 'winners', label: 'Winners' },
  { value: 'losers', label: 'Losers' },
];

export default function Backtesting() {
  const { symbols } = useDashboard();
  const [outcome, setOutcome] = useState('all');
  const [symbol, setSymbol] = useState('');

  const backtest = useResource(api.backtest, { key: 'backtest', pollMs: POLL_INTERVALS.artifacts });
  const trades = useResource(
    (options) => api.backtestTrades({ symbol: symbol || undefined, outcome, limit: 200 }, options),
    { key: `trades-${symbol}-${outcome}`, pollMs: POLL_INTERVALS.artifacts },
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
        eyebrow="Backtesting"
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
        <div className="mb-4">
          <ErrorNote error={backtest.error} label="Backtest unavailable" onRetry={backtest.refresh} />
        </div>
      )}
      {backtest.envelope?.warnings?.length > 0 && (
        <div className="mb-4">
          <Warnings warnings={backtest.envelope.warnings} />
        </div>
      )}

      <StatGrid columns={4}>
        <Stat
          label="Total return"
          value={signedPct(metrics.total_return)}
          hint={`Gross ${signedPct(metrics.gross_return_sum)} − cost ${pct(metrics.cost_paid)}`}
          footnote={`${metrics.segment ?? 'segment n/a'} · ${num(metrics.cost_bps_round_trip, 1)} bps round trip`}
        />
        <Stat label="Max drawdown" value={pct(metrics.max_drawdown)} hint="Peak-to-trough on the daily equity curve" />
        <Stat label="Sharpe" value={num(metrics.sharpe, 2)} hint="Not annualised unless the artifact says so" />
        <Stat
          label="Trades"
          value={int(metrics.trade_count)}
          hint={`Hit rate ${pct(hitRate, 1)}`}
          footnote={`${int(data?.trading_days)} trading days · ${int(data?.positive_days)} positive`}
        />
      </StatGrid>

      <Panel
        className="mt-4"
        eyebrow="Equity"
        title="Cumulative net equity"
        actions={<ChartLegend items={[{ label: 'Equity' }, { label: 'Drawdown', color: '#78786f', opacity: 0.9 }]} />}
      >
        {backtest.loading && !data ? (
          <Loading rows={5} />
        ) : curve.length ? (
          <>
            <EquityChart data={curve} height={300} />
            <p className="mt-2 text-[.6875rem] text-[var(--ink-faint)]">
              {formatDate(data?.start)} → {formatDate(data?.end)} · {curve.length} weekly points reconstructed from{' '}
              {int(data?.equity_points)} trading days. Equity starts at 1.0 on the first event.
            </p>
          </>
        ) : (
          <Empty title="No equity curve" detail="results/backtest_timeseries.parquet has no usable events." />
        )}
      </Panel>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Panel className="lg:col-span-2" eyebrow="Attribution" title="Per symbol" flush>
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

        <div className="space-y-4">
          <Panel eyebrow="Costs" title="Where the return goes">
            <KeyValue
              columns={1}
              items={[
                { label: 'Gross return sum', value: signedPct(metrics.gross_return_sum), mono: true },
                { label: 'Cost paid', value: pct(metrics.cost_paid), mono: true },
                { label: 'Net return sum', value: signedPct(metrics.net_return_sum), mono: true },
                { label: 'Round-trip cost', value: `${num(metrics.cost_bps_round_trip, 1)} bps`, mono: true },
              ]}
            />
            <div className="mt-3">
              <Callout tone="warn" title="Costs dominate">
                When the gross sum is positive but the net sum is negative, the strategy has no edge to spend. Fixing the model is the only lever;
                lowering the assumed cost would hide the problem.
              </Callout>
            </div>
          </Panel>

          <Panel eyebrow="Configuration" title="Cost model">
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

      <Panel
        className="mt-4"
        eyebrow="Trades"
        title={`${trades.data?.count ?? 0} event(s)`}
        flush
        actions={
          <div className="flex flex-wrap items-center gap-2">
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
        )}
      </Panel>
    </>
  );
}
