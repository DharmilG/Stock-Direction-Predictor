from __future__ import annotations

"""Minimal Diebold-Mariano test (interim; full HAC harness is F-16).

Compares two forecasts via the loss differential series with a Newey-West
(Bartlett) long-run variance — required here because 5-day triple-barrier
events overlap, so per-row losses are serially correlated and a plain
variance would overstate significance.
"""

import numpy as np
from scipy.stats import norm


def loss_differential(losses_base: np.ndarray, losses_model: np.ndarray) -> np.ndarray:
    """Positive values ⇒ model better. Pure."""
    return np.asarray(losses_base, dtype=float) - np.asarray(losses_model, dtype=float)


def dm_statistic(d: np.ndarray, lags: int = 5) -> dict:
    """DM stat with Newey-West variance. H0: equal predictive accuracy. Pure."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"stat": None, "p_value": None, "n": int(n), "note": "too few samples"}
    mean_d = float(d.mean())
    gamma0 = float(((d - mean_d) ** 2).mean())
    lr_var = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        cov = float(((d[lag:] - mean_d) * (d[:-lag] - mean_d)).mean())
        lr_var += 2.0 * w * cov
    if lr_var <= 0:
        return {"stat": 0.0, "p_value": 1.0, "n": int(n), "note": "zero variance (identical losses)"}
    stat = float(mean_d / np.sqrt(lr_var / n))
    p_value = float(2.0 * norm.sf(abs(stat)))
    return {"stat": stat, "p_value": p_value, "n": int(n), "mean_d": mean_d}


def up_vs_rest_dm(y_true: np.ndarray, pred_model: np.ndarray, pred_base: np.ndarray, lags: int = 5) -> dict:
    """DM on UP-vs-rest 0/1 loss: model vs a competing forecast. Pure."""
    y_true = np.asarray(y_true, dtype=int)
    is_up = (y_true == 2).astype(int)
    loss_model = (is_up != (np.asarray(pred_model, dtype=int) == 2).astype(int)).astype(float)
    loss_base = (is_up != (np.asarray(pred_base, dtype=int) == 2).astype(int)).astype(float)
    out = dm_statistic(loss_differential(loss_base, loss_model), lags=lags)
    out["model_up_accuracy"] = float(1.0 - loss_model.mean())
    out["base_up_accuracy"] = float(1.0 - loss_base.mean())
    return out
