"""Paired comparisons between two models scored on the same rows.

Fifty names on the same day share a market move, so rows are not independent
observations. Every interval here resamples **dates**, not rows: a row-level
standard error on this panel is roughly half the size it should be, which is
exactly how a one-point difference turns into a fake result.

One implementation, used by the v2 trainer and by the comparison endpoints, so
the two can never disagree about what "the gap is 2.1 pp" means.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

BOOTSTRAP_REPS = 1000


def clustered_delta(
    dates,
    a,
    b,
    label: str = "b - a",
    *,
    reps: int = BOOTSTRAP_REPS,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired difference ``b - a``, bootstrapped over dates.

    ``a`` and ``b`` are per-row values (correctness flags, per-row losses, ...).
    They are averaged within each date first, so a date that happens to carry
    more names does not get more weight than a thin one.
    """
    frame = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates).tz_localize(None),
            "a": np.asarray(a, dtype=float),
            "b": np.asarray(b, dtype=float),
        }
    )
    per_date_frame = frame.groupby("date", sort=True)[["a", "b"]].mean()
    per_date = (per_date_frame["b"] - per_date_frame["a"]).to_numpy()
    n_dates = int(per_date.size)
    mean = float(per_date.mean()) if n_dates else 0.0
    se = float(per_date.std(ddof=1) / np.sqrt(n_dates)) if n_dates > 1 else 0.0
    rng = np.random.default_rng(seed)
    draws = (
        np.array([rng.choice(per_date, n_dates, replace=True).mean() for _ in range(reps)])
        if n_dates
        else np.array([0.0])
    )
    return {
        "label": label,
        "mean_delta": mean,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "t": (mean / se) if se > 0 else None,
        "dates": n_dates,
        "rows": int(len(frame)),
        "reps": int(reps),
        "method": "paired date-clustered bootstrap",
    }


def mcnemar(a_correct, b_correct) -> dict[str, Any]:
    """Exact McNemar test on the rows where the two models disagree.

    The only rows that carry information about *which* model is better are the
    discordant ones: rows both get right or both get wrong are silent. A model
    that wins by agreeing with the other is not winning.
    """
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(a_only, discordant, 0.5, alternative="two-sided").pvalue)
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "p_value": p_value,
        "method": "exact McNemar (binomial on discordant pairs)",
    }
