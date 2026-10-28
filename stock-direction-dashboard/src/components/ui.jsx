import { AlertTriangle, Info, Minus } from 'lucide-react';

/**
 * Shared primitives. Everything here is monochrome on purpose: hierarchy comes
 * from weight, spacing and border style, never from colour.
 */

export function Panel({ title, eyebrow, actions, children, bodyClass = '', className = '', flush = false }) {
  return (
    <section className={`panel ${className}`}>
      {(title || eyebrow || actions) && (
        <header className="panel-header">
          <div className="min-w-0">
            {eyebrow && <p className="eyebrow">{eyebrow}</p>}
            {title && <h2 className="section-title mt-0.5">{title}</h2>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={flush ? bodyClass : `panel-body ${bodyClass}`}>{children}</div>
    </section>
  );
}

export function PageHeader({ eyebrow, title, description, actions }) {
  return (
    <header className="mb-5 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1 className="page-title mt-1">{title}</h1>
        {description && <p className="muted mt-1.5 max-w-3xl">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Stat({ label, value, hint, footnote, mono = true }) {
  return (
    <div className="panel px-4 py-3">
      <p className="eyebrow">{label}</p>
      <p className={`kpi mt-1.5 ${mono ? 'num' : ''}`}>{value ?? <Minus size={18} className="text-[var(--ink-faint)]" />}</p>
      {hint && <p className="muted mt-1 text-[.75rem]">{hint}</p>}
      {footnote && <p className="mt-1 text-[.6875rem] text-[var(--ink-faint)]">{footnote}</p>}
    </div>
  );
}

export function StatGrid({ children, columns = 4 }) {
  const cols = { 2: 'sm:grid-cols-2', 3: 'sm:grid-cols-2 lg:grid-cols-3', 4: 'sm:grid-cols-2 xl:grid-cols-4', 5: 'sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5' };
  return <div className={`grid gap-3 ${cols[columns] ?? cols[4]}`}>{children}</div>;
}

export function Badge({ tone = 'badge-ghost', children, title, icon: Icon }) {
  return (
    <span className={`badge ${tone}`} title={title}>
      {Icon && <Icon size={11} strokeWidth={2.5} />}
      {children}
    </span>
  );
}

export function Button({ variant = 'secondary', size, icon: Icon, children, title, ...props }) {
  return (
    <button type="button" className={`btn btn-${variant}`} data-size={size} title={title} {...props}>
      {Icon && <Icon size={14} strokeWidth={2.25} />}
      {children}
    </button>
  );
}

export function IconButton({ label, icon: Icon, variant = 'quiet', ...props }) {
  return (
    <button type="button" className={`btn btn-${variant} btn-icon`} aria-label={label} title={label} {...props}>
      <Icon size={15} strokeWidth={2.25} />
    </button>
  );
}

export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="segment" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className="btn btn-secondary"
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Empty({ title = 'Nothing to show yet', detail, action }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      <div className="flex h-10 w-10 items-center justify-center rounded-full border border-dashed border-[var(--line-strong)]">
        <Minus size={16} className="text-[var(--ink-faint)]" />
      </div>
      <p className="text-sm font-semibold text-[var(--ink)]">{title}</p>
      {detail && <p className="muted max-w-md">{detail}</p>}
      {action}
    </div>
  );
}

export function Loading({ rows = 3, label = 'Reading artifacts' }) {
  return (
    <div className="space-y-2" role="status" aria-label={label}>
      {Array.from({ length: rows }).map((_, index) => (
        <div key={index} className="skeleton h-4" style={{ width: `${100 - index * 12}%` }} />
      ))}
    </div>
  );
}

export function ErrorNote({ error, onRetry, label = 'Request failed' }) {
  if (!error) return null;
  return (
    <div className="flex flex-wrap items-start gap-3 border-l-[5px] border-[var(--ink)] bg-[var(--surface-sunken)] px-3 py-2">
      <AlertTriangle size={15} className="mt-0.5 shrink-0" />
      <div className="min-w-0 flex-1">
        <p className="text-[.8125rem] font-semibold text-[var(--ink)]">{label}</p>
        <p className="muted mt-0.5 break-words">{error.message}</p>
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
      className={`flex gap-2.5 border-l-[5px] px-3 py-2 ${
        tone === 'warn' ? 'border-[var(--ink)] bg-[var(--surface-sunken)]' : 'border-[var(--line-strong)] bg-white'
      }`}
    >
      <Icon size={14} className="mt-0.5 shrink-0 text-[var(--ink-3)]" />
      <div className="min-w-0">
        {title && <p className="text-[.8125rem] font-semibold text-[var(--ink)]">{title}</p>}
        <div className="muted">{children}</div>
      </div>
    </div>
  );
}

export function KeyValue({ items, columns = 2 }) {
  const visible = items.filter((item) => item && item.value !== undefined);
  if (!visible.length) return <Empty title="No fields recorded" />;
  return (
    <dl className={`grid gap-x-5 gap-y-3 ${columns === 3 ? 'sm:grid-cols-3' : columns === 1 ? '' : 'sm:grid-cols-2'}`}>
      {visible.map((item) => (
        <div key={item.label} className="min-w-0 border-t border-[var(--line)] pt-2">
          <dt className="eyebrow">{item.label}</dt>
          <dd className={`mt-0.5 text-[.875rem] text-[var(--ink)] ${item.mono ? 'mono' : ''} break-words`}>{item.value}</dd>
          {item.hint && <p className="mt-0.5 text-[.6875rem] text-[var(--ink-faint)]">{item.hint}</p>}
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
        <div className="mb-1 flex items-baseline justify-between gap-3">
          {label && <span className="text-[.75rem] font-semibold text-[var(--ink-2)]">{label}</span>}
          {showValue && (
            <span className="mono text-[.75rem] text-[var(--ink-3)]">
              {value ?? 0} / {max ?? 0}
            </span>
          )}
        </div>
      )}
      <div className="h-2 w-full overflow-hidden rounded-full border border-[var(--line-strong)] bg-[var(--surface-sunken)]" role="progressbar" aria-valuenow={value ?? 0} aria-valuemin={0} aria-valuemax={max ?? 0}>
        <div className="h-full bg-[var(--ink)]" style={{ width: `${ratio * 100}%` }} />
      </div>
      {detail && <p className="mt-1 text-[.6875rem] text-[var(--ink-faint)]">{detail}</p>}
    </div>
  );
}

export function Table({ columns, rows, empty, rowKey, dense = false }) {
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
            <tr key={rowKey ? rowKey(row, index) : index}>
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

export function Warnings({ warnings }) {
  if (!warnings?.length) return null;
  return (
    <ul className="space-y-1.5">
      {warnings.map((warning) => (
        <li key={warning}>
          <Callout tone="warn">{warning}</Callout>
        </li>
      ))}
    </ul>
  );
}

export function Mono({ children }) {
  return <span className="mono text-[.75rem] text-[var(--ink-3)]">{children}</span>;
}
