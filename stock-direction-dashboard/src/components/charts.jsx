import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { formatDate, formatShortDate } from '../lib/dates';
import { num, pct } from '../lib/format';

/**
 * Charts are grayscale by construction. Line weight and dash patterns encode
 * series identity; the axis, grid and tooltip follow the same hairline palette
 * as the rest of the shell.
 */

const INK = '#111110';
const INK_2 = '#4a4a47';
const INK_3 = '#78786f';
const LINE = '#e5e5e2';

export function ChartTooltip({ active, payload, label, formatter, labelFormatter }) {
  if (!active || !payload?.length) return null;
  const title = labelFormatter ? labelFormatter(label) : label;
  return (
    <div className="chart-tooltip">
      {title != null && <p className="mb-1 font-semibold">{title}</p>}
      {payload.map((entry) => (
        <p key={entry.dataKey ?? entry.name} className="mono flex items-center gap-2 text-[.7rem]">
          <span className="inline-block h-1.5 w-3" style={{ background: entry.color ?? entry.stroke ?? INK }} />
          <span className="text-[var(--ink-3)]">{entry.name}</span>
          <span className="ml-auto font-semibold">{formatter ? formatter(entry.value, entry) : entry.value}</span>
        </p>
      ))}
    </div>
  );
}

function axes({ xKey, tickFormatter, yTickFormatter, yDomain, yWidth = 46, xInterval }) {
  return [
    <CartesianGrid key="grid" stroke={LINE} strokeDasharray="2 3" vertical={false} />,
    <XAxis
      key="x"
      dataKey={xKey}
      tick={{ fill: INK_3, fontSize: 11 }}
      tickFormatter={tickFormatter ?? ((value) => formatShortDate(value))}
      tickLine={false}
      axisLine={{ stroke: LINE }}
      minTickGap={18}
      interval={xInterval ?? 'preserveStartEnd'}
    />,
    <YAxis
      key="y"
      tick={{ fill: INK_3, fontSize: 11 }}
      tickFormatter={yTickFormatter}
      tickLine={false}
      axisLine={false}
      width={yWidth}
      domain={yDomain ?? ['auto', 'auto']}
    />,
  ];
}

export function EquityChart({ data, xKey = 'date', height = 260, showDrawdown = true }) {
  if (!data?.length) return <div className="skeleton" style={{ height }} />;
  const values = data.map((point) => point.equity).filter((value) => Number.isFinite(value));
  const low = Math.min(...values, 1);
  const high = Math.max(...values, 1);
  const pad = Math.max(0.02, (high - low) * 0.12);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="equityWash" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={INK} stopOpacity={0.16} />
            <stop offset="100%" stopColor={INK} stopOpacity={0.01} />
          </linearGradient>
        </defs>
        {axes({ xKey, yTickFormatter: (value) => num(value, 2), yDomain: [low - pad, high + pad], yWidth: 52 })}
        <ReferenceLine y={1} stroke={INK_3} strokeDasharray="4 4" />
        <Tooltip
          content={
            <ChartTooltip
              labelFormatter={formatDate}
              formatter={(value, entry) => (entry.dataKey === 'drawdown' ? pct(value) : num(value, 4))}
            />
          }
        />
        <Area type="monotone" dataKey="equity" name="Equity" stroke={INK} strokeWidth={2} fill="url(#equityWash)" dot={false} activeDot={{ r: 3, fill: INK }} />
        {showDrawdown && <Line type="monotone" dataKey="drawdown" name="Drawdown" stroke={INK_3} strokeWidth={1} strokeDasharray="3 3" dot={false} />}
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function LineSeriesChart({ data, xKey = 'date', series, height = 240, yTickFormatter, yDomain, xTickFormatter, unit = '' }) {
  if (!data?.length) return <div className="skeleton" style={{ height }} />;
  const strokes = [INK, INK_2, INK_3];
  const dashes = [undefined, '6 3', '2 3'];
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        {axes({ xKey, tickFormatter: xTickFormatter, yTickFormatter, yDomain })}
        <Tooltip
          content={
            <ChartTooltip
              labelFormatter={xTickFormatter ? (value) => value : formatDate}
              formatter={(value) => (value == null ? 'n/a' : `${num(value, Math.abs(value) < 10 ? 3 : 2)}${unit}`)}
            />
          }
        />
        {series.map((entry, index) => (
          <Line
            key={entry.key}
            type="monotone"
            dataKey={entry.key}
            name={entry.label}
            stroke={entry.color ?? strokes[index % strokes.length]}
            strokeWidth={entry.width ?? 1.75}
            strokeDasharray={entry.dash ?? dashes[index % dashes.length]}
            dot={false}
            connectNulls={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function BarSeriesChart({ data, xKey, yKey, height = 220, name, yTickFormatter, xTickFormatter, horizontal = false, tone = INK }) {
  if (!data?.length) return <div className="skeleton" style={{ height }} />;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout={horizontal ? 'vertical' : 'horizontal'} margin={{ top: 8, right: 12, bottom: 0, left: horizontal ? 8 : 0 }}>
        <CartesianGrid stroke={LINE} strokeDasharray="2 3" vertical={horizontal} horizontal={!horizontal} />
        {horizontal ? (
          <>
            <XAxis type="number" tick={{ fill: INK_3, fontSize: 11 }} tickFormatter={yTickFormatter} axisLine={{ stroke: LINE }} tickLine={false} />
            <YAxis type="category" dataKey={xKey} tick={{ fill: INK_3, fontSize: 11 }} width={140} axisLine={false} tickLine={false} />
          </>
        ) : (
          <>
            <XAxis dataKey={xKey} tick={{ fill: INK_3, fontSize: 11 }} tickFormatter={xTickFormatter ?? ((value) => formatShortDate(value))} axisLine={{ stroke: LINE }} tickLine={false} minTickGap={16} />
            <YAxis tick={{ fill: INK_3, fontSize: 11 }} tickFormatter={yTickFormatter} axisLine={false} tickLine={false} width={46} />
          </>
        )}
        <Tooltip content={<ChartTooltip labelFormatter={(value) => value} formatter={(value) => num(value, 3)} />} />
        <Bar dataKey={yKey} name={name ?? yKey} fill={tone} radius={[2, 2, 0, 0]} maxBarSize={38} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/**
 * Confusion matrix as a grayscale heat grid: darker cell = more rows. The value
 * is printed in every cell, so no information depends on the shade alone.
 */
export function ConfusionMatrix({ matrix, labels = ['DOWN', 'FLAT', 'UP'], title = 'Confusion matrix' }) {
  if (!matrix?.length) return <div className="skeleton h-48" />;
  const flat = matrix.flat().map((value) => Number(value) || 0);
  const max = Math.max(...flat, 1);
  return (
    <div>
      <table className="w-full border-collapse text-center">
        <caption className="sr-only">{title}</caption>
        <thead>
          <tr>
            <th className="p-2 text-left text-[.625rem] font-bold uppercase tracking-widest text-[var(--ink-3)]">actual ↓ / predicted →</th>
            {labels.map((label) => (
              <th key={label} className="p-2 text-[.625rem] font-bold uppercase tracking-widest text-[var(--ink-3)]">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={labels[rowIndex] ?? rowIndex}>
              <th className="p-2 text-left text-[.625rem] font-bold uppercase tracking-widest text-[var(--ink-3)]">{labels[rowIndex] ?? rowIndex}</th>
              {row.map((value, columnIndex) => {
                const count = Number(value) || 0;
                const ratio = count / max;
                return (
                  <td key={columnIndex} className="border border-white p-0">
                    <div
                      className="flex h-14 items-center justify-center"
                      style={{ background: `rgba(17,17,16,${(0.06 + ratio * 0.88).toFixed(3)})`, color: ratio > 0.45 ? '#fff' : INK }}
                      title={`${labels[rowIndex] ?? rowIndex} → ${labels[columnIndex] ?? columnIndex}: ${count}`}
                    >
                      <span className="mono text-sm font-semibold">{count}</span>
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Reliability diagram: predicted confidence versus observed accuracy. */
export function CalibrationChart({ bins, height = 240 }) {
  if (!bins?.length) return <div className="skeleton" style={{ height }} />;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={bins} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
        <CartesianGrid stroke={LINE} strokeDasharray="2 3" />
        <XAxis dataKey="mean_predicted" type="number" domain={[0, 1]} tick={{ fill: INK_3, fontSize: 11 }} tickFormatter={(value) => pct(value, 0)} axisLine={{ stroke: LINE }} tickLine={false} />
        <YAxis type="number" domain={[0, 1]} tick={{ fill: INK_3, fontSize: 11 }} tickFormatter={(value) => pct(value, 0)} axisLine={false} tickLine={false} width={42} />
        <Tooltip content={<ChartTooltip formatter={(value) => pct(value)} />} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={INK_3} strokeDasharray="4 4" />
        <Line type="monotone" dataKey="mean_observed" name="Observed accuracy" stroke={INK} strokeWidth={2} dot={{ r: 2.5, fill: INK }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Horizontal importance ranking drawn as plain divs — cheaper than a chart. */
export function ImportanceBars({ rows, valueKey = 'importance', labelKey = 'feature', limit = 15, formatter }) {
  if (!rows?.length) return <div className="skeleton h-40" />;
  const shown = rows.slice(0, limit);
  const max = Math.max(...shown.map((row) => Math.abs(row[valueKey]) || 0), 1e-9);
  return (
    <ul className="space-y-1.5">
      {shown.map((row) => (
        <li key={row[labelKey]} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2">
          <div className="min-w-0">
            <div className="flex items-baseline justify-between gap-2">
              <span className="mono truncate text-[.7rem] text-[var(--ink-2)]">{row[labelKey]}</span>
              <span className="mono text-[.7rem] font-semibold text-[var(--ink)]">{formatter ? formatter(row[valueKey]) : num(row[valueKey], 4)}</span>
            </div>
            <div className="mt-0.5 h-1.5 w-full rounded-sm bg-[var(--surface-sunken)]">
              <div className="h-full rounded-sm bg-[var(--ink)]" style={{ width: `${(Math.abs(row[valueKey]) / max) * 100}%` }} />
            </div>
          </div>
        </li>
      ))}
    </ul>
  );
}

/** Tiny sparkline used inside tables; deliberately axis-free. */
export function Sparkline({ values, height = 28, stroke = INK }) {
  const points = (values ?? []).filter((value) => Number.isFinite(value));
  if (points.length < 2) return <span className="text-[.625rem] text-[var(--ink-faint)]">n/a</span>;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const path = points
    .map((value, index) => {
      const x = (index / (points.length - 1)) * 100;
      const y = height - ((value - min) / span) * (height - 2) - 1;
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(' ');
  return (
    <svg viewBox={`0 0 100 ${height}`} width="100%" height={height} preserveAspectRatio="none" aria-hidden="true">
      <path d={path} fill="none" stroke={stroke} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/** Horizontal distribution of predicted labels, one bar per class. */
export function DistributionBars({ distribution, labels = ['DOWN', 'FLAT', 'UP'] }) {
  const total = labels.reduce((sum, label) => sum + (distribution?.[label] ?? 0), 0);
  if (!total) return <div className="skeleton h-16" />;
  const tones = { DOWN: '#3a3a37', FLAT: '#c9c9c4', UP: INK };
  return (
    <div className="space-y-2">
      {labels.map((label) => {
        const count = distribution?.[label] ?? 0;
        return (
          <div key={label} className="grid grid-cols-[3.5rem_minmax(0,1fr)_3.5rem] items-center gap-2">
            <span className="mono text-[.7rem] font-semibold text-[var(--ink-2)]">{label}</span>
            <div className="h-2 w-full rounded-sm bg-[var(--surface-sunken)]">
              <div className="h-full rounded-sm" style={{ width: `${(count / total) * 100}%`, background: tones[label] ?? INK }} />
            </div>
            <span className="mono text-right text-[.7rem] text-[var(--ink-3)]">{count}</span>
          </div>
        );
      })}
    </div>
  );
}

export function ChartLegend({ items }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
      {items.map((item) => (
        <span key={item.label} className="inline-flex items-center gap-1.5 text-[.6875rem] text-[var(--ink-3)]">
          <span className="inline-block h-[2px] w-4" style={{ background: item.color ?? INK, opacity: item.opacity ?? 1 }} />
          {item.label}
        </span>
      ))}
    </div>
  );
}

/** Cell helper for bar charts that need per-bar shades. */
export function ShadedCells({ values }) {
  const max = Math.max(...values.map((value) => Math.abs(value) || 0), 1e-9);
  return values.map((value, index) => (
    <Cell key={index} fill={`rgba(17,17,16,${(0.25 + (Math.abs(value) / max) * 0.7).toFixed(3)})`} />
  ));
}
