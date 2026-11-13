import { useCallback, useId, useLayoutEffect, useRef, useState } from 'react';
import { Info } from 'lucide-react';

/** Keep the note this far inside the viewport edges. */
const MARGIN = 12;
/** Vertical gap between the heading and the note; matches `.infotip`'s offset. */
const GAP = 9;
const UNSET = { dx: 0, top: undefined };

/**
 * Explains a heading in place. The trigger wraps the heading text, so hovering
 * the title itself opens the note; the glyph beside it takes focus for keyboard
 * readers. Four slots at most, so a panel never grows a wall of prose:
 *
 *   what   — what the number or axis is, in plain words
 *   reads  — how it is computed, including the denominator
 *   ideal  — what a healthy reading looks like (model panels)
 *   when   — what a specific bad reading means (model panels)
 *
 * The note is 24rem wide but the heading it hangs from can sit anywhere, so its
 * placement is measured rather than assumed: it slides left to clear the right
 * edge, and it sits under the heading unless that would leave the viewport, in
 * which case it goes above, and failing both it pins to the top margin and
 * scrolls inside its own max-height. Placement is re-measured on resize, on
 * scroll, and whenever the note itself changes size.
 */
export function InfoTip({ children, what, reads, ideal, when, label = 'What this shows' }) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const wrapRef = useRef(null);
  const tipRef = useRef(null);
  const [place, setPlace] = useState(UNSET);
  const rows = [
    ['How it is read', reads],
    ['Ideal', ideal],
    ['If it looks wrong', when],
  ].filter(([, text]) => Boolean(text));

  const reposition = useCallback(() => {
    const wrap = wrapRef.current;
    const tip = tipRef.current;
    if (!wrap || !tip) return;

    const anchor = wrap.getBoundingClientRect();
    // offsetWidth/Height ignore the translate already applied to the note.
    const width = tip.offsetWidth;
    const height = tip.offsetHeight;
    const viewWidth = document.documentElement.clientWidth;
    const viewHeight = window.innerHeight;

    let dx = 0;
    if (anchor.left + width > viewWidth - MARGIN) dx = viewWidth - MARGIN - (anchor.left + width);
    if (anchor.left + dx < MARGIN) dx = MARGIN - anchor.left;

    // Under the heading first, then above it, then pinned to the top margin.
    const below = anchor.height + GAP;
    const above = -height - GAP;
    let top = below;
    if (anchor.top + below + height > viewHeight - MARGIN) {
      top = anchor.top + above >= MARGIN ? above : MARGIN - anchor.top;
    }

    const next = { dx: Math.round(dx), top: Math.round(top) };
    setPlace((previous) => (previous.dx === next.dx && previous.top === next.top ? previous : next));
  }, []);

  useLayoutEffect(() => {
    if (!open) {
      setPlace((previous) => (previous.dx === UNSET.dx && previous.top === UNSET.top ? previous : UNSET));
      return undefined;
    }

    // The layout effect runs before paint, so the note never flashes unclamped.
    reposition();
    // One frame later, in case web fonts or wrapping settled after measuring.
    const frame = requestAnimationFrame(reposition);
    const observer = new ResizeObserver(reposition);
    if (tipRef.current) observer.observe(tipRef.current);
    window.addEventListener('resize', reposition);
    window.addEventListener('scroll', reposition, true);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      window.removeEventListener('resize', reposition);
      window.removeEventListener('scroll', reposition, true);
    };
  }, [open, reposition]);

  if (!what && !rows.length) return null;

  return (
    <span
      ref={wrapRef}
      className="infotip-wrap"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      {children && <span>{children}</span>}
      <button
        type="button"
        className="infotip-trigger"
        aria-label={label}
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(event) => {
          if (event.key === 'Escape') setOpen(false);
        }}
        onClick={() => setOpen((value) => !value)}
      >
        <Info size={11} strokeWidth={2.75} />
      </button>
      {open && (
        <span
          ref={tipRef}
          className="infotip"
          role="tooltip"
          id={id}
          style={{ transform: `translateX(${place.dx}px)`, top: place.top === undefined ? undefined : `${place.top}px` }}
        >
          <span className="infotip-title">{what}</span>
          {rows.map(([key, text]) => (
            <span className="infotip-row" key={key}>
              <span className="infotip-key">{key}</span>
              {text}
            </span>
          ))}
        </span>
      )}
    </span>
  );
}
