const DASH = '—';

export function num(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return DASH;
  return Number(value).toFixed(digits);
}

export function int(value) {
  if (value == null || Number.isNaN(Number(value))) return DASH;
  return new Intl.NumberFormat('en-IN').format(Math.round(Number(value)));
}

/** Ratio (0.4714) -> "47.14%". */
export function pct(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return DASH;
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

/** Already-percentage values (4.27 meaning 4.27%) -> "4.27 pp". */
export function pp(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return DASH;
  const sign = Number(value) > 0 ? '+' : '';
  return `${sign}${Number(value).toFixed(digits)} pp`;
}

export function signed(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return DASH;
  const n = Number(value);
  return `${n > 0 ? '+' : ''}${n.toFixed(digits)}`;
}

export function signedPct(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return DASH;
  const n = Number(value) * 100;
  return `${n > 0 ? '+' : ''}${n.toFixed(digits)}%`;
}

export function compact(value) {
  if (value == null || Number.isNaN(Number(value))) return DASH;
  const n = Number(value);
  if (Math.abs(n) >= 1e7) return `${(n / 1e7).toFixed(2)} Cr`;
  if (Math.abs(n) >= 1e5) return `${(n / 1e5).toFixed(2)} L`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return n.toFixed(0);
}

export function titleize(value) {
  if (value == null) return DASH;
  return String(value)
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function clamp01(value) {
  if (value == null || Number.isNaN(Number(value))) return 0;
  return Math.min(1, Math.max(0, Number(value)));
}

export function sortBy(rows, key, direction = 'desc') {
  const factor = direction === 'asc' ? 1 : -1;
  return [...rows].sort((a, b) => {
    const x = a?.[key];
    const y = b?.[key];
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    if (typeof x === 'number' && typeof y === 'number') return (x - y) * factor;
    return String(x).localeCompare(String(y)) * factor;
  });
}
