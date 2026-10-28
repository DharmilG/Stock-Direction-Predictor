/**
 * Date helpers.
 *
 * NSE sessions are calendar dates, not instants: "2026-09-15" means that trading
 * day in IST. Passing those strings through `new Date()` converts them to UTC
 * midnight and shifts the label a day in any negative-offset timezone, so
 * date-only values are formatted straight from their parts and never parsed.
 */

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const DATE_ONLY = /^(\d{4})-(\d{2})-(\d{2})$/;
const MIDNIGHT = /^(\d{4})-(\d{2})-(\d{2})[T ]00:00(:00(\.0+)?)?(Z|[+-]00:?00)?$/;

const IST_FORMAT = new Intl.DateTimeFormat('en-GB', {
  timeZone: 'Asia/Kolkata',
  year: 'numeric',
  month: 'short',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
});

/** Split a date-only value into parts without timezone conversion. */
export function dateParts(value) {
  if (value == null || value === '') return null;
  const raw = String(value).trim();
  const match = DATE_ONLY.exec(raw) || MIDNIGHT.exec(raw);
  if (!match) return null;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  if (!year || month < 1 || month > 12 || day < 1 || day > 31) return null;
  return { year, month, day };
}

export function isDateOnly(value) {
  return dateParts(value) != null;
}

/** "2026-09-15" -> "15 Sep 2026". Returns an em dash when unavailable. */
export function formatDate(value, { withYear = true } = {}) {
  const parts = dateParts(value);
  if (!parts) return '—';
  const label = `${String(parts.day).padStart(2, '0')} ${MONTHS[parts.month - 1]}`;
  return withYear ? `${label} ${parts.year}` : label;
}

/** "2026-09-15" -> "15/09". Compact axis/tick label. */
export function formatShortDate(value) {
  const parts = dateParts(value);
  if (!parts) return '—';
  return `${String(parts.day).padStart(2, '0')}/${String(parts.month).padStart(2, '0')}`;
}

/** "2026-09-15T00:00:00" -> "2026-09-15"; ISO instants pass through unchanged. */
export function toDateString(value) {
  const parts = dateParts(value);
  if (!parts) return null;
  return `${parts.year}-${String(parts.month).padStart(2, '0')}-${String(parts.day).padStart(2, '0')}`;
}

/**
 * Format an actual instant (heartbeat, file mtime) in IST. Only use this for
 * values that carry a real time of day and an offset.
 */
export function formatInstant(value, { seconds = false } = {}) {
  if (!value) return '—';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return '—';
  const base = IST_FORMAT.format(parsed);
  if (!seconds) return `${base} IST`;
  return `${base}:${String(parsed.getSeconds()).padStart(2, '0')} IST`;
}

/** "2026-09-15" or an instant -> the right formatter, chosen per value. */
export function formatAny(value) {
  if (value == null || value === '') return '—';
  return dateParts(value) ? formatDate(value) : formatInstant(value);
}

/** 9265 -> "2h 34m". Rounds down; returns "—" for nullish input. */
export function humanDuration(totalSeconds) {
  if (totalSeconds == null || Number.isNaN(Number(totalSeconds))) return '—';
  const seconds = Math.max(0, Math.round(Number(totalSeconds)));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours}h ${minutes % 60}m`;
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
}

export function humanAge(seconds) {
  if (seconds == null) return 'unknown';
  return `${humanDuration(seconds)} ago`;
}

/** Session counts read better than raw day counts for trading data. */
export function sessionLag(count) {
  if (count == null) return '—';
  if (count === 0) return 'current';
  return `${count} session${count === 1 ? '' : 's'} behind`;
}
