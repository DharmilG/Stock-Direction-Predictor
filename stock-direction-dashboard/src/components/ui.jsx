import { AlertTriangle, Info, Minus } from 'lucide-react';
import { InfoTip } from './InfoTip';

/**
 * Shared primitives. Everything here is monochrome on purpose: hierarchy comes
 * from weight, spacing and border style, never from colour.
 */

export function Panel({ title, eyebrow, info, actions, children, bodyClass = '', className = '', flush = false }) {
  return (
    <section className={`panel ${className}`}>
      {(title || eyebrow || actions) && (
        <header className="panel-header">
          <div className="min-w-0">
            {eyebrow && <p className="eyebrow">{eyebrow}</p>}
            {title && (
              <h3 className="section-title mt-1">
                {info ? <InfoTip {...info} label={`About ${title}`}>{title}</InfoTip> : title}
              </h3>
            )}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2.5">{actions}</div>}
        </header>
      )}
      <div className={flush ? bodyClass : `panel-body ${bodyClass}`}>{children}</div>
    </section>
  );
}

export function PageHeader({ eyebrow, title, description, actions }) {
  return (
    <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1 className="page-title mt-2">{title}</h1>
        {description && <p className="muted mt-2.5 max-w-3xl">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2.5">{actions}</div>}
    </header>
  );
}

/** A page section: one heading, then the panels that belong to it. */
export function Section({ eyebrow, title, note, actions, children, className = '' }) {
  return (
    <section className={`mt-10 first:mt-0 ${className}`}>
      <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          {eyebrow && <p className="eyebrow">{eyebrow}</p>}
          {title && <h2 className="group-title mt-1.5">{title}</h2>}
          {note && <p className="muted mt-2 max-w-3xl">{note}</p>}
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2.5">{actions}</div>}
      </div>
      {children}
    </section>
  );
}

/**
 * One figure. Two lines only: a label and the value. Anything a reader needs in
 * order to interpret it belongs in `info`, which opens on hover over the label.
 */
export function Stat({ label, value, unit, info, mono = true }) {
  return (
    <div className="panel px-5 py-5">
      <p className="eyebrow">
        {info ? <InfoTip {...info} label={`About ${label}`}>{label}</InfoTip> : label}
      </p>
      <p className={`kpi mt-2.5 flex flex-wrap items-baseline gap-x-2 ${mono ? 'num' : ''}`}>
        {value ?? <Minus size={22} className="text-[var(--ink-faint)]" />}
        {unit && <span className="text-[.9375rem] font-medium tracking-normal text-[var(--ink-3)]">{unit}</span>}
      </p>
    </div>
  );
}

export function StatGrid({ children, columns = 4 }) {
  const cols = { 2: 'sm:grid-cols-2', 3: 'sm:grid-cols-2 lg:grid-cols-3', 4: 'sm:grid-cols-2 xl:grid-cols-4', 5: 'sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5' };
  return <div className={`grid gap-4 ${cols[columns] ?? cols[4]}`}>{children}</div>;
}

export function Badge({ tone = 'badge-ghost', children, title, icon: Icon }) {
  return (
    <span className={`badge ${tone}`} title={title}>
      {Icon && <Icon size={11} strokeWidth={2.5} />}
      {children}
    </span>
  );
}

export function Button({ variant = 'primary', size, icon: Icon, children, title, ...props }) {
  return (
    <button type="button" className={`btn btn-${variant}`} data-size={size} title={title} {...props}>
      {Icon && <Icon size={15} strokeWidth={2.25} />}
      {children}
    </button>
  );
}

export function IconButton({ label, icon: Icon, variant = 'quiet', ...props }) {
  return (
    <button type="button" className={`btn btn-${variant} btn-icon`} aria-label={label} title={label} {...props}>
      <Icon size={16} strokeWidth={2.25} />
    </button>
  );
}

/** Segmented control. Same Button component; only `aria-pressed` changes. */
export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="segment" role="group" aria-label={label}>
      {options.map((option) => (
        <Button
          key={option.value}
          variant="secondary"
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </Button>
      ))}
    </div>
  );
}

export function Empty({ title = 'Nothing to show yet', detail, action }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-14 text-center">
      <div className="flex h-11 w-11 items-center justify-center rounded-full border border-dashed border-[var(--line-strong)]">
        <Minus size={17} className="text-[var(--ink-faint)]" />
      </div>
      <p className="text-base font-semibold text-[var(--ink)]">{title}</p>
      {detail && <p className="muted max-w-md">{detail}</p>}
      {action}
    </div>
  );
}

export function Loading({ rows = 3, label = 'Reading artifacts' }) {
  return (
    <div className="space-y-2.5" role="status" aria-label={label}>
      {Array.from({ length: rows }).map((_, index) => (
        <div key={index} className="skeleton h-4" style={{ width: `${100 - index * 12}%` }} />
      ))}
    </div>
  );
}

export function ErrorNote({ error, onRetry, label = 'Request failed' }) {
  if (!error) return null;
  return (
    <div className="flex flex-wrap items-start gap-3 border-l-2 border-[var(--ink)] bg-[var(--surface)] px-4 py-3">
      <AlertTriangle size={16} className="mt-0.5 shrink-0" />
      <div className="min-w-0 flex-1">
        <p className="text-[.9375rem] font-semibold text-[var(--ink)]">{label}</p>
        <p className="muted mt-1 break-words">{error.message}</p>
      </div>
      {onRetry && (
        <Button size="sm" variant="secondary" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

export function Callout({ tone = 'info', title, children }) {
  const Icon = tone === 'warn' ? AlertTriangle : Info;
  return (
    <div
      className={`flex gap-3 border-l-2 px-4 py-3 ${
        tone === 'warn' ? 'border-[var(--ink)] bg-[var(--surface-sunken)]' : 'border-[var(--line-strong)] bg-white'
      }`}
    >
      <Icon size={15} className="mt-0.5 shrink-0 text-[var(--ink-3)]" />
      <div className="min-w-0">
        {title && <p className="text-[.9375rem] font-semibold text-[var(--ink)]">{title}</p>}
        <div className="muted">{children}</div>
      </div>
    </div>
  );
}

export function KeyValue({ items, columns = 2 }) {
  const visible = items.filter((item) => item && item.value !== undefined);
  if (!visible.length) return <Empty title="No fields recorded" />;
  return (
    <dl className={`grid gap-x-6 gap-y-4 ${columns === 3 ? 'sm:grid-cols-3' : columns === 1 ? '' : 'sm:grid-cols-2'}`}>
      {visible.map((item) => (
        <div key={item.label} className="min-w-0 border-t border-[var(--line)] pt-2.5">
          <dt className="eyebrow">{item.label}</dt>
          <dd className={`mt-1 text-[.9375rem] text-[var(--ink)] ${item.mono ? 'mono' : ''} break-words`}>{item.value}</dd>
          {item.hint && <p className="mt-1 text-[.8125rem] text-[var(--ink-3)]">{item.hint}</p>}
        </div>
      ))}
    </dl>
  );
}

export function ProgressBar({ value, max = 100, label, detail, showValue = true }) {
  const total = max > 0 ? max : 1;
  const ratio = Math.max(0, Math.min(1, (value ?? 0) / total));
  return (
    <div>
      {(label || showValue) && (
        <div className="mb-1.5 flex items-baseline justify-between gap-3">
          {label && <span className="text-[.875rem] font-semibold text-[var(--ink-2)]">{label}</span>}
          {showValue && (
            <span className="mono text-[.875rem] text-[var(--ink-3)]">
              {value ?? 0} / {max ?? 0}
            </span>
          )}
        </div>
      )}
      <div className="h-2 w-full overflow-hidden rounded-full border border-[var(--line-strong)] bg-[var(--surface-sunken)]" role="progressbar" aria-valuenow={value ?? 0} aria-valuemin={0} aria-valuemax={max ?? 0}>
        <div className="h-full bg-[var(--ink)]" style={{ width: `${ratio * 100}%` }} />
      </div>
      {detail && <p className="mt-1.5 text-[.8125rem] text-[var(--ink-3)]">{detail}</p>}
    </div>
  );
}

export function Table({ columns, rows, empty, rowKey, dense = false, onRowClick }) {
  if (!rows?.length) return empty ?? <Empty />;
  return (
    <div className="overflow-x-auto scroll-thin">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} style={column.align === 'right' ? { textAlign: 'right' } : undefined} className={column.headerClass}>
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr
              key={rowKey ? rowKey(row, index) : index}
              onClick={onRowClick ? () => onRowClick(row, index) : undefined}
              style={onRowClick ? { cursor: 'pointer' } : undefined}
            >
              {columns.map((column) => (
                <td
                  key={column.key}
                  className={`${column.align === 'right' ? 'text-right' : ''} ${column.strong ? 'strong' : ''} ${column.mono ? 'mono' : ''} ${
                    dense ? 'py-1.5' : ''
                  }`}
                  style={column.width ? { width: column.width } : undefined}
                >
                  {column.render ? column.render(row, index) : row[column.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Wraps a number in a tabular-nums span so columns line up. */
export function Num({ children, className = '' }) {
  return <span className={`num ${className}`}>{children}</span>;
}

export function SourceTag({ envelope, prefix = 'as of' }) {
  if (!envelope) return null;
  const parts = [];
  if (envelope.as_of) parts.push(`${prefix} ${envelope.as_of}`);
  if (envelope.source?.artifacts?.length) parts.push(`${envelope.source.artifacts.length} artifact(s)`);
  return (
    <span
      className="badge badge-ghost"
      title={(envelope.source?.artifacts ?? []).join('\n') || 'No artifacts recorded'}
    >
      {parts.join(' · ') || 'artifact'}
    </span>
  );
}

/**
 * Warnings collapse to one line. The first one is the headline; the rest wait
 * behind a disclosure, because a stack of boxes above a chart hides the chart.
 */
export function Warnings({ warnings }) {
  if (!warnings?.length) return null;
  const [first, ...rest] = warnings;
  if (!rest.length) {
    return (
      <p className="notice flex items-center gap-2.5 px-3.5 py-2.5 text-[.9375rem] text-[var(--ink-2)]">
        <AlertTriangle size={15} className="shrink-0" />
        {first}
      </p>
    );
  }
  return (
    <details className="notice">
      <summary>
        <AlertTriangle size={15} className="shrink-0" />
        {first}
        <span className="notice-more">{rest.length} more</span>
      </summary>
      <ul>
        {rest.map((warning) => (
          <li key={warning}>{warning}</li>
        ))}
      </ul>
    </details>
  );
}

export function Mono({ children }) {
  return <span className="mono text-[.875rem] text-[var(--ink-3)]">{children}</span>;
}
