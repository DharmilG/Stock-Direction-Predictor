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
import { int, num, pct } from '../lib/format';
import { Badge } from './ui';

/**
 * Charts are grayscale by construction. Line weight and dash patterns encode
 * series identity; the axis, grid and tooltip follow the same hairline palette
 * as the rest of the shell.
 */

const INK = '#111110';
const INK_2 = '#4a4a47';
const INK_3 = '#6f6f68';
const LINE = '#e5e5e2';

/**
 * Axis ticks carry colour only; the font size lives in `.recharts-cartesian-axis-tick-value`
 * so the whole chart surface rescales from styles.css.
 */
const TICK = { fill: INK_3 };

export function ChartTooltip({ active, payload, label, formatter, labelFormatter }) {
  if (!active || !payload?.length) return null;
  const title = labelFormatter ? labelFormatter(label) : label;
  return (
    <div className="chart-tooltip">
      {title != null && <p className="mb-1 font-semibold">{title}</p>}
      {payload.map((entry) => (
        <p key={entry.dataKey ?? entry.name} className="mono flex items-center gap-2 text-[.8125rem]">
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
      tick={TICK}
      tickFormatter={tickFormatter ?? ((value) => formatShortDate(value))}
      tickLine={false}
      axisLine={{ stroke: LINE }}
      minTickGap={18}
      interval={xInterval ?? 'preserveStartEnd'}
    />,
    <YAxis
      key="y"
      tick={TICK}
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

export function LineSeriesChart({ data, xKey = 'date', series, height = 240, yTickFormatter, yDomain, xTickFormatter, unit = '', reference }) {
  if (!data?.length) return <div className="skeleton" style={{ height }} />;
  const strokes = [INK, INK_2, INK_3];
  const dashes = [undefined, '6 3', '2 3'];
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        {axes({ xKey, tickFormatter: xTickFormatter, yTickFormatter, yDomain })}
        {reference != null && <ReferenceLine y={reference.value ?? reference} stroke={INK_3} strokeDasharray="4 4" label={reference.label} />}
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
            <XAxis type="number" tick={TICK} tickFormatter={yTickFormatter} axisLine={{ stroke: LINE }} tickLine={false} />
            <YAxis type="category" dataKey={xKey} tick={TICK} width={140} axisLine={false} tickLine={false} />
          </>
        ) : (
          <>
            <XAxis dataKey={xKey} tick={TICK} tickFormatter={xTickFormatter ?? ((value) => formatShortDate(value))} axisLine={{ stroke: LINE }} tickLine={false} minTickGap={16} />
            <YAxis tick={TICK} tickFormatter={yTickFormatter} axisLine={false} tickLine={false} width={46} />
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
            <th className="p-2 text-left text-[.8125rem] font-bold uppercase tracking-wider text-[var(--ink-3)]">actual ↓ / predicted →</th>
            {labels.map((label) => (
              <th key={label} className="p-2 text-[.8125rem] font-bold uppercase tracking-wider text-[var(--ink-3)]">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={labels[rowIndex] ?? rowIndex}>
              <th className="p-2 text-left text-[.8125rem] font-bold uppercase tracking-wider text-[var(--ink-3)]">{labels[rowIndex] ?? rowIndex}</th>
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
        <XAxis dataKey="mean_predicted" type="number" domain={[0, 1]} tick={TICK} tickFormatter={(value) => pct(value, 0)} axisLine={{ stroke: LINE }} tickLine={false} />
        <YAxis type="number" domain={[0, 1]} tick={TICK} tickFormatter={(value) => pct(value, 0)} axisLine={false} tickLine={false} width={42} />
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
              <span className="mono truncate text-[.8125rem] text-[var(--ink-2)]">{row[labelKey]}</span>
              <span className="mono text-[.8125rem] font-semibold text-[var(--ink)]">{formatter ? formatter(row[valueKey]) : num(row[valueKey], 4)}</span>
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
  if (points.length < 2) return <span className="text-[.75rem] text-[var(--ink-faint)]">n/a</span>;
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
            <span className="mono text-[.8125rem] font-semibold text-[var(--ink-2)]">{label}</span>
            <div className="h-2 w-full rounded-sm bg-[var(--surface-sunken)]">
              <div className="h-full rounded-sm" style={{ width: `${(count / total) * 100}%`, background: tones[label] ?? INK }} />
            </div>
            <span className="mono text-right text-[.8125rem] text-[var(--ink-3)]">{count}</span>
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
        <span key={item.label} className="inline-flex items-center gap-1.5 text-[.8125rem] text-[var(--ink-3)]">
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

// --------------------------------------------------------------------------- //
// Behaviour plate panels
//
// Five panels, one legend, one card. Each panel is also usable on its own; the
// plate merely arranges them so one split can be read in one glance.
// --------------------------------------------------------------------------- //
const CLASS_TONES = { DOWN: '#3a3a37', FLAT: '#c9c9c4', UP: INK };
const CLASS_LABELS = ['DOWN', 'FLAT', 'UP'];

/**
 * A. Paired bars: share of actual rows against share of predicted rows.
 * The bias chip is the difference in percentage points, printed, not inferred
 * from bar length.
 */
export function PairedShareBars({ distribution, height = 148 }) {
  if (!distribution) return <div className="skeleton" style={{ height }} />;
  const rows = CLASS_LABELS.map((label) => ({
    label,
    actual: (distribution.actual?.[label] ?? 0) * 100,
    predicted: (distribution.predicted?.[label] ?? 0) * 100,
    bias: distribution.bias_pp?.[label] ?? 0,
  }));
  return (
    <div className="space-y-2.5">
      {rows.map((row) => (
        <div key={row.label} className="grid grid-cols-[2.6rem_minmax(0,1fr)_3.6rem] items-center gap-2">
          <span className="mono text-[.8125rem] font-semibold text-[var(--ink-2)]">{row.label}</span>
          <div className="space-y-1">
            <div className="flex h-2 w-full rounded-sm bg-[var(--surface-sunken)]" title={`actual ${row.actual.toFixed(1)}%`}>
              <div className="h-full rounded-sm" style={{ width: `${row.actual}%`, background: CLASS_TONES[row.label], opacity: 0.45 }} />
            </div>
            <div className="flex h-2 w-full rounded-sm bg-[var(--surface-sunken)]" title={`predicted ${row.predicted.toFixed(1)}%`}>
              <div className="h-full rounded-sm" style={{ width: `${row.predicted}%`, background: CLASS_TONES[row.label] }} />
            </div>
          </div>
          <span className={`mono text-right text-[.8125rem] font-semibold ${Math.abs(row.bias) >= 10 ? 'text-[var(--ink)]' : 'text-[var(--ink-3)]'}`}>
            {`${row.bias > 0 ? '+' : ''}${row.bias.toFixed(1)} pp`}
          </span>
        </div>
      ))}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-[var(--line)] pt-2">
        <span className="inline-flex items-center gap-1.5 text-[.75rem] text-[var(--ink-3)]">
          <span className="inline-block h-2 w-3 rounded-sm bg-[var(--ink)] opacity-40" /> actual
        </span>
        <span className="inline-flex items-center gap-1.5 text-[.75rem] text-[var(--ink-3)]">
          <span className="inline-block h-2 w-3 rounded-sm bg-[var(--ink)]" /> predicted
        </span>
      </div>
    </div>
  );
}

/**
 * C. Class profile: mean p(c) inside each actual class. Flat, overlapping bars
 * mean the model says the same thing whatever is true — that is the whole point
 * of the panel, so nothing is smoothed or hidden.
 */
export function GroupedProfileBars({ profile, height = 176 }) {
  const rows = profile?.rows ?? [];
  if (!rows.length) return <div className="skeleton" style={{ height }} />;
  return (
    <div className="space-y-2">
      {rows.map((row) => (
        <div key={row.actual}>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="mono text-[.8125rem] font-semibold text-[var(--ink-2)]">actual {row.actual}</span>
            <span className="mono text-[.75rem] text-[var(--ink-faint)]">n {row.n}</span>
          </div>
          <div className="flex gap-1">
            {CLASS_LABELS.map((label) => {
              const value = row.mean?.[label] ?? 0;
              return (
                <div key={label} className="flex-1" title={`p(${label}) = ${value.toFixed(3)}`}>
                  <div className="flex h-14 items-end rounded-sm bg-[var(--surface-sunken)]">
                    <div className="w-full rounded-sm" style={{ height: `${Math.max(2, value * 100)}%`, background: CLASS_TONES[label] }} />
                  </div>
                  <p className="mono mt-0.5 text-center text-[.5625rem] text-[var(--ink-3)]">{value.toFixed(2)}</p>
                </div>
              );
            })}
          </div>
        </div>
      ))}
      <p className="border-t border-[var(--line)] pt-2 text-[.75rem] text-[var(--ink-faint)]">
        separation {num(profile?.separation, 4)} · 0 = identical profiles, 1 = disjoint
      </p>
    </div>
  );
}

/** D. Conviction: how often the winning class barely won. */
export function ConvictionHistogram({ conviction, height = 160 }) {
  const bins = conviction?.bins ?? [];
  if (!bins.length) return <div className="skeleton" style={{ height }} />;
  const max = Math.max(...bins.map((bin) => bin.share ?? 0), 1e-9);
  return (
    <div>
      <div className="flex h-32 items-end gap-1.5">
        {bins.map((bin) => (
          <div key={`${bin.lo}-${bin.hi}`} className="flex-1" title={`${bin.lo}–${bin.hi}: ${int(bin.n)} rows (${pct(bin.share, 1)})`}>
            <div className="flex h-full items-end rounded-sm bg-[var(--surface-sunken)]">
              <div className="w-full rounded-sm bg-[var(--ink)]" style={{ height: `${Math.max(1.5, ((bin.share ?? 0) / max) * 100)}%` }} />
            </div>
            <p className="mono mt-1 text-center text-[.5625rem] text-[var(--ink-3)]">{bin.lo === 0 ? '<.34' : bin.hi === 1 ? '.65+' : `${bin.lo}`}</p>
          </div>
        ))}
      </div>
      <p className="mt-1 border-t border-[var(--line)] pt-2 text-[.75rem] text-[var(--ink-faint)]">
        {pct(conviction?.share_below_half, 1)} of calls had max(p) &lt; 0.50 · mean max(p) {num(conviction?.mean_max_p, 4)}
      </p>
    </div>
  );
}

/** E. Drivers: signed mean SHAP for the class this split over-produces. */
export function SignedDriverBars({ rows, height = 176, label = 'mean signed SHAP' }) {
  if (!rows?.length) return <div className="skeleton" style={{ height }} />;
  const max = Math.max(...rows.map((row) => Math.abs(row.mean_signed) || 0), 1e-9);
  return (
    <div>
      <p className="eyebrow mb-1.5">{label}</p>
      <ul className="space-y-1.5">
        {rows.map((row) => {
          const ratio = (row.mean_signed ?? 0) / max;
          return (
            <li key={row.feature} className="grid grid-cols-[minmax(0,7.5rem)_minmax(0,1fr)] items-center gap-2">
              <span className="mono truncate text-[.75rem] text-[var(--ink-2)]" title={row.feature}>
                {row.feature}
              </span>
              <div className="relative h-2.5 w-full rounded-sm bg-[var(--surface-sunken)]">
                <div className="absolute top-0 h-full bg-[var(--ink)]" style={{ width: `${Math.abs(ratio) * 100}%`, left: ratio < 0 ? 'auto' : 0, right: ratio < 0 ? 0 : 'auto' }} />
                <div className="absolute top-[-1px] h-[calc(100%+2px)] w-px bg-[var(--ink-faint)]" style={{ left: 0 }} />
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/**
 * One bordered card, five panels. `split` names the panel row so three plates
 * can sit side by side with the same reading order.
 */
export function BehaviourPlate({ plate, drivers, compact = false }) {
  if (!plate) return <div className="skeleton h-64" />;
  const dominant = plate.distribution?.dominant;
  const driverRows = (drivers ?? []).filter((row) => row.class === dominant);
  return (
    <div className="rounded-lg border border-[var(--line)] bg-white p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <p className="eyebrow">{plate.split} · {int(plate.rows)} rows</p>
          <p className="mt-0.5 text-[.8125rem] font-semibold leading-snug text-[var(--ink)]">{plate.mode?.headline}</p>
        </div>
        <Badge tone={plate.mode?.mode === 'balanced_discriminating' ? 'badge-ghost' : 'badge-dashed'}>{plate.mode?.mode}</Badge>
      </div>
      <div className={`grid gap-3 ${compact ? 'grid-cols-1' : 'sm:grid-cols-2'}`}>
        <figure className="border-t border-[var(--line)] pt-2">
          <figcaption className="eyebrow mb-1.5">A · shares</figcaption>
          <PairedShareBars distribution={plate.distribution} />
        </figure>
        <figure className="border-t border-[var(--line)] pt-2">
          <figcaption className="eyebrow mb-1.5">C · class profile</figcaption>
          <GroupedProfileBars profile={plate.profile} />
        </figure>
        <figure className="border-t border-[var(--line)] pt-2">
          <figcaption className="eyebrow mb-1.5">D · conviction</figcaption>
          <ConvictionHistogram conviction={plate.conviction} />
        </figure>
        <figure className="border-t border-[var(--line)] pt-2">
          <figcaption className="eyebrow mb-1.5">E · drivers → {dominant ?? '—'}</figcaption>
          {driverRows.length ? (
            <SignedDriverBars rows={driverRows} />
          ) : (
            <p className="muted text-[.8125rem]">No signed SHAP rows for this class were persisted.</p>
          )}
        </figure>
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5 border-t border-[var(--line)] pt-2">
        {(plate.mode?.evidence ?? []).map((item) => (
          <span key={`${item.label}-${item.hint ?? ''}`} className="badge badge-ghost" title={item.hint}>
            {item.label} {item.value}
            {item.hint ? ` · ${item.hint}` : ''}
          </span>
        ))}
        {(plate.mode?.flags ?? []).map((flag) => (
          <span key={flag} className="badge badge-dashed">{flag}</span>
        ))}
      </div>
    </div>
  );
}

/**
 * Waterfall from the base value through the contributions to the class raw
 * score. Steps are ordered by |contribution|, which is how they were persisted.
 */
export function Waterfall({ base, steps, closing, height = 230 }) {
  if (base == null || !steps?.length) return <div className="skeleton" style={{ height }} />;
  let running = base;
  const bars = steps.map((step) => {
    const from = running;
    running += step.contribution;
    return { ...step, from, to: running, end: closing ?? running };
  });
  const span = Math.max(...bars.map((bar) => Math.max(Math.abs(bar.from), Math.abs(bar.to))), Math.abs(base), 1e-6);
  const scale = (value) => `${((value + span) / (2 * span)) * 100}%`;
  return (
    <div className="space-y-1">
      <div className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_3.4rem] items-center gap-2 text-[.75rem] text-[var(--ink-3)]">
        <span className="eyebrow">base</span>
        <div className="relative h-2 rounded-sm bg-[var(--surface-sunken)]">
          <div className="absolute top-0 h-full bg-[var(--ink-faint)]" style={{ left: scale(base), width: '1px' }} />
        </div>
        <span className="mono text-right">{num(base, 3)}</span>
      </div>
      {bars.map((bar) => {
        const left = Math.min(bar.from, bar.to);
        const width = Math.abs(bar.to - bar.from);
        const positive = bar.to >= bar.from;
        return (
          <div key={`${bar.rank}-${bar.feature}`} className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_3.4rem] items-center gap-2">
            <span className="mono truncate text-[.75rem] text-[var(--ink-2)]" title={`${bar.feature} = ${num(bar.feature_value, 4)}`}>
              {bar.rank}. {bar.feature}
            </span>
            <div className="relative h-2.5 rounded-sm bg-[var(--surface-sunken)]">
              <div className="absolute top-0 h-full w-px bg-[var(--ink-faint)]" style={{ left: scale(0) }} />
              <div
                className="absolute top-0 h-full"
                style={{
                  left: scale(left),
                  width: `calc(${(width / (2 * span)) * 100}% )`,
                  background: positive ? INK : INK_3,
                  opacity: positive ? 0.9 : 1,
                }}
              />
            </div>
            <span className="mono text-right text-[.75rem] text-[var(--ink-3)]">{num(bar.contribution, 3)}</span>
          </div>
        );
      })}
      <div className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_3.4rem] items-center gap-2 border-t border-[var(--line)] pt-1">
        <span className="eyebrow">raw score</span>
        <div className="relative h-2 rounded-sm bg-[var(--surface-sunken)]">
          <div className="absolute top-0 h-full bg-[var(--ink)]" style={{ left: scale(closing ?? running), width: '1px' }} />
        </div>
        <span className="mono text-right text-[.75rem] font-semibold text-[var(--ink)]">{num(closing ?? running, 3)}</span>
      </div>
    </div>
  );
}

/** Rolling share line with an optional reference level. */
export function RollingLine({ data, xKey = 'date', series, height = 200, reference, yDomain = [0, 1], yTickFormatter = (value) => pct(value, 0) }) {
  if (!data?.length) return <div className="skeleton" style={{ height }} />;
  return <LineSeriesChart data={data} xKey={xKey} series={series} height={height} yDomain={yDomain} yTickFormatter={yTickFormatter} reference={reference} />;
}

/** Two series on independent axes, e.g. temperature and clock. */
export function DualAxisLine({ data, xKey, left, right, height = 220, leftDomain, rightDomain, leftFormatter, rightFormatter, xTickFormatter }) {
  if (!data?.length) return <div className="skeleton" style={{ height }} />;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={LINE} strokeDasharray="2 3" vertical={false} />
        <XAxis
          dataKey={xKey}
          tick={TICK}
          tickFormatter={xTickFormatter}
          tickLine={false}
          axisLine={{ stroke: LINE }}
          minTickGap={24}
        />
        <YAxis yAxisId="left" tick={TICK} tickFormatter={leftFormatter} domain={leftDomain ?? ['auto', 'auto']} tickLine={false} axisLine={false} width={46} />
        <YAxis yAxisId="right" orientation="right" tick={TICK} tickFormatter={rightFormatter} domain={rightDomain ?? ['auto', 'auto']} tickLine={false} axisLine={false} width={46} />
        <Tooltip content={<ChartTooltip labelFormatter={xTickFormatter ? (value) => value : formatDate} />} />
        <Line yAxisId="left" type="monotone" dataKey={left.key} name={left.label} stroke={INK} strokeWidth={1.75} dot={false} connectNulls={false} />
        <Line yAxisId="right" type="monotone" dataKey={right.key} name={right.label} stroke={INK_3} strokeWidth={1.25} strokeDasharray="5 3" dot={false} connectNulls={false} />
        {right.reference != null && <ReferenceLine yAxisId="right" y={right.reference} stroke={INK_3} strokeDasharray="2 4" />}
      </LineChart>
    </ResponsiveContainer>
  );
}
