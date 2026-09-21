from __future__ import annotations

"""DOWN-flag reformulation (risk-off head, Workstream 2).

Framing: not "predicts direction" but "flags elevated down-move probability
for exposure reduction." No shorting leg (sidesteps short-sell frictions);
UP predictions route to NO-SIGNAL. Two specialist heads, one frozen
backbone — NOT a generic ensemble (§9.1 stays in force).

Cascade (FLAT head adopted first, unchanged):
  p_flat >= tau_flat (0.40) → FLAT
  elif p_down >= tau_down → DOWN (risk flag)
  else argmax(DOWN, UP)

Locked bar (precision PLUS economics — either alone fails):
  DOWN precision >= 55% on flagged test subset AND avoided-loss economics
  positive (mean over flags of -realized_return minus one-way exit cost).
A precise flag that loses money after exit costs is a failed flag.

Protocol: sweep tau_down on CALIBRATION slice only (max precision at
coverage >= 3%), judge ONCE on test. Ledger-logged as a D1-grade touch
(decision-only, no retrain) — still a read of test labels.
"""

import argparse

import numpy as np

from ..config import load_settings
from ..utils.io import atomic_write_json
from ..utils.logging import setup_logging
from ..utils.state import update_state
from .flat_threshold import _calibration_proba
from .touch_ledger import log_touch

DOWN = 0
FLAT = 1
# Operative region is BELOW 0.5: the fallback already predicts DOWN whenever
# p_down >= p_up, so tau_down >= 0.5 is a near-noop (verified: flat grid).
# The flag only bites where it overrules UP-leaning rows (p_down < p_up).
TAU_GRID = [round(x, 2) for x in np.arange(0.30, 0.51, 0.05)]
MIN_COVERAGE = 0.03
PRECISION_BAR = 0.55
# One-way delivery exit: half of commission/slippage/other + sell-side STT.
EXIT_COST = (1.0 + 3.0 + 1.0) / 2 / 10000.0 + 0.0010


def apply_down_flag(proba: np.ndarray, tau_flat: float = 0.40, tau_down: float = 0.60) -> np.ndarray:
    """Cascade decision rule. Pure."""
    proba = np.asarray(proba, dtype=float)
    pred = np.where(proba[:, 0] >= proba[:, 2], 0, 2)
    down_mask = proba[:, DOWN] >= tau_down
    pred = np.where(down_mask, DOWN, pred).astype(int)
    flat_mask = proba[:, FLAT] >= tau_flat
    return np.where(flat_mask, FLAT, pred).astype(int)


def sweep_down_threshold(y_true: np.ndarray, proba: np.ndarray,
                         tau_flat: float = 0.40, grid=None) -> dict:
    """Max DOWN precision at coverage floor on a tuning split. Pure."""
    from sklearn.metrics import precision_score
    y_true = np.asarray(y_true, dtype=int)
    rows = []
    for tau in (grid or TAU_GRID):
        pred = apply_down_flag(proba, tau_flat, float(tau))
        flagged = pred == DOWN
        cov = float(flagged.mean())
        prec = float(precision_score(y_true == DOWN, flagged, zero_division=0))
        rows.append({"tau_down": float(tau), "precision": prec,
                     "recall": float(((pred == DOWN) & (y_true == DOWN)).sum() / max((y_true == DOWN).sum(), 1)),
                     "coverage": cov})
    feasible = [r for r in rows if r["coverage"] >= MIN_COVERAGE]
    pool = feasible or rows
    winner = max(pool, key=lambda r: r["precision"])
    return {"winner": winner, "grid": rows}


def avoided_loss_economics(y_true: np.ndarray, pred: np.ndarray,
                           realized: np.ndarray, exit_cost: float = EXIT_COST) -> dict:
    """Risk-avoidance economics on flagged rows. Pure.

    Benefit per flag = -realized_return (loss dodged when DOWN, upside
    missed when wrong) minus one-way exit cost. Positive mean ⇒ the flag
    pays for its whipsaws.
    """
    flagged = np.asarray(pred, dtype=int) == DOWN
    n = int(flagged.sum())
    if n == 0:
        return {"n_flags": 0, "mean_net": None, "total_net": 0.0}
    per_flag = -np.asarray(realized, dtype=float)[flagged] - exit_cost
    return {"n_flags": n, "mean_net": float(per_flag.mean()),
            "total_net": float(per_flag.sum()),
            "whipsaw_rate": float((np.asarray(y_true)[flagged] != DOWN).mean())}


def run_down_flag() -> dict:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "down_flag.log")
    tau_flat = float(settings.flat_threshold)
    y_cal, p_cal, y_test, p_test = _calibration_proba(settings)
    sweep = sweep_down_threshold(y_cal, p_cal, tau_flat)
    tau = float(sweep["winner"]["tau_down"])
    logger.info("Calibration: tau_down=%.2f precision=%.4f coverage=%.3f.",
                tau, sweep["winner"]["precision"], sweep["winner"]["coverage"])
    pred_test = apply_down_flag(p_test, tau_flat, tau)
    test_pred_path = paths.results / "predictions.parquet"
    import pandas as pd
    frame = pd.read_parquet(test_pred_path)
    realized = frame["realized_return"].to_numpy()
    from sklearn.metrics import precision_score, recall_score
    prec = float(precision_score(y_test, pred_test, labels=[0, 1, 2], average=None, zero_division=0)[0])
    rec = float(recall_score(y_test, pred_test, labels=[0, 1, 2], average=None, zero_division=0)[0])
    econ = avoided_loss_economics(y_test, pred_test, realized)
    verdict = {
        "tau_flat": tau_flat, "tau_down": tau,
        "calibration_winner": sweep["winner"],
        "calibration_grid": sweep["grid"],
        "test": {"down_precision": prec, "down_recall": rec,
                 "coverage": float((pred_test == DOWN).mean()), **econ},
        "pass": bool(prec >= PRECISION_BAR and (econ["mean_net"] or 0.0) > 0),
    }
    atomic_write_json(paths.results / "down_flag.json", verdict)
    log_touch(paths, "DOWNFLAG", "D1", "tau_down sweep + test judgment", int(len(y_test)), "results/down_flag.json")
    update_state(paths.state, "down_flag", status="complete")
    logger.info("Test: DOWN precision %.4f recall %.4f coverage %.3f mean_net %s → %s.",
                prec, rec, verdict["test"]["coverage"], str(econ["mean_net"]),
                "PASS" if verdict["pass"] else "FAIL")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description="DOWN-flag reformulation sweep + judgment.")
    parser.parse_args()
    run_down_flag()


if __name__ == "__main__":
    main()
