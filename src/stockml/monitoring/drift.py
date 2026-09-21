from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if len(ref) < 30 or len(cur) < 10 or ref.nunique() < 2:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
