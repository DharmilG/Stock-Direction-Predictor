import { ArrowDown, ArrowUp, Minus } from 'lucide-react';
import { SIGNAL_META } from '../lib/constants';
import { pct } from '../lib/format';

const ICONS = { UP: ArrowUp, DOWN: ArrowDown, FLAT: Minus };

/**
 * The dashboard never encodes direction with colour. A signal is carried by its
 * glyph, its text label and the weight of its badge, so it survives greyscale,
 * colour-blindness and screenshot printing.
 */
export function SignalBadge({ signal, size = 'md' }) {
  const key = String(signal ?? '').toUpperCase();
  const meta = SIGNAL_META[key];
  if (!meta) {
    return <span className="badge badge-ghost">{signal ?? 'n/a'}</span>;
  }
  const Icon = ICONS[key];
  return (
    <span className={`badge ${meta.tone}`} title={meta.description} data-size={size}>
      <Icon size={11} strokeWidth={3} aria-hidden="true" />
      {meta.label}
    </span>
  );
}

export function SignalIcon({ signal, size = 16 }) {
  const Icon = ICONS[String(signal ?? '').toUpperCase()] ?? Minus;
  return <Icon size={size} strokeWidth={2.5} aria-hidden="true" />;
}

/** p_down / p_flat / p_up as three adjacent grey bands with a tick at 50%. */
export function ProbabilityBar({ pDown, pFlat, pUp, height = 8 }) {
  const down = Math.max(0, Math.min(1, pDown ?? 0));
  const flat = Math.max(0, Math.min(1, pFlat ?? 0));
  const up = Math.max(0, Math.min(1, pUp ?? 0));
  const total = down + flat + up || 1;
  const bands = [
    { key: 'down', share: down / total, fill: '#3a3a37', label: 'P(down)' },
    { key: 'flat', share: flat / total, fill: '#c9c9c4', label: 'P(flat)' },
    { key: 'up', share: up / total, fill: '#111110', label: 'P(up)' },
  ];
  return (
    <div className="w-full">
      <div className="flex overflow-hidden rounded-sm border border-[var(--line-strong)]" style={{ height }} role="img" aria-label={`P(down) ${pct(down)}, P(flat) ${pct(flat)}, P(up) ${pct(up)}`}>
        {bands.map((band) => (
          <div key={band.key} title={`${band.label} ${pct(band.share)}`} style={{ width: `${band.share * 100}%`, background: band.fill }} />
        ))}
      </div>
      <div className="mt-1 flex justify-between text-[.625rem] text-[var(--ink-faint)]">
        <span className="mono">D {pct(down)}</span>
        <span className="mono">F {pct(flat)}</span>
        <span className="mono">U {pct(up)}</span>
      </div>
    </div>
  );
}

/** Agreement between a predicted and a realised label, shown as a glyph pair. */
export function AgreementMark({ predicted, actual }) {
  const same = String(predicted).toUpperCase() === String(actual).toUpperCase();
  return (
    <span className="inline-flex items-center gap-1.5" title={same ? 'Prediction matched the realised label' : 'Prediction missed the realised label'}>
      <SignalBadge signal={predicted} size="sm" />
      <span className="text-[.625rem] text-[var(--ink-faint)]">{same ? '=' : '≠'}</span>
      <SignalBadge signal={actual} size="sm" />
    </span>
  );
}
