from __future__ import annotations

"""Track F (Issue #2): per-class FLAT thresholds.

Diagnosis: FLAT OvR AUC 0.622 (best of three classes) with F1 0.207
(worst) = decision-boundary problem, not information problem. The model
ranks FLAT well but argmax draws the line wrong.

Why this is NOT the rejected UP-threshold idea: UP has no ranking skill
(AUC 0.518) — thresholds cannot create ranking. FLAT has ranking skill
without F1 — thresholds convert existing ranking into decisions. Same
tool, opposite diagnosis; judged separately.

Protocol (no leakage): sweep τ_flat on the frozen CALIBRATION slice
(temperature-fitting slice, never test), apply the winner to TEST once,
judge on test with headline + DM + per-class F1.
"""

import argparse

import numpy as np
import pandas as pd

from ..config import load_settings
from ..models.lightgbm_train import load_training_frame, prepare_xy, _timestamps
from ..utils.io import atomic_write_json, read_json
from ..utils.logging import setup_logging
from ..utils.state import update_state
from ..validation.time_splits import three_way_date_split
from .dm import dm_statistic, loss_differential
from .metrics import apply_temperature, classification_metrics

FLAT = 1
TAU_GRID = [round(x, 2) for x in np.arange(0.10, 0.61, 0.05)]


def apply_flat_threshold(proba: np.ndarray, tau_flat: float) -> np.ndarray:
    """Predict FLAT iff p_flat >= tau, else argmax(DOWN, UP). Pure."""
    proba = np.asarray(proba, dtype=float)
    pred = np.argmax(proba, axis=1)
    flat_mask = proba[:, FLAT] >= tau_flat
    fallback = np.where(proba[:, 0] >= proba[:, 2], 0, 2)
    return np.where(flat_mask, FLAT, fallback).astype(int)


def sweep_flat_threshold(y_true: np.ndarray, proba: np.ndarray, grid=None) -> list[dict]:
    """Sweep τ on a tuning split. Pure. Maximizes FLAT F1 subject to
    headline accuracy dropping no more than 0.5pp vs argmax."""
    from sklearn.metrics import f1_score, accuracy_score
    y_true = np.asarray(y_true, dtype=int)
    base_pred = np.argmax(np.asarray(proba), axis=1)
    base_acc = float(accuracy_score(y_true, base_pred))
    rows = []
    for tau in (grid or TAU_GRID):
        pred = apply_flat_threshold(proba, float(tau))
        rows.append({
            "tau_flat": float(tau),
            "flat_f1": float(f1_score(y_true, pred, labels=[0, 1, 2], average=None, zero_division=0)[1]),
            "accuracy": float(accuracy_score(y_true, pred)),
            "coverage_flat": float((pred == FLAT).mean()),
        })
    feasible = [r for r in rows if r["accuracy"] >= base_acc - 0.005]
    pool = feasible or rows
    winner = max(pool, key=lambda r: r["flat_f1"])
    return [{"base_accuracy": base_acc, "winner": winner, "grid": rows}]


def _calibration_proba(settings):
    import lightgbm as lgb
    paths = settings.paths
    model = lgb.Booster(model_file=str(paths.artifacts / "models" / "lightgbm_final.txt"))
    calibration = read_json(paths.artifacts / "calibration.json", {"temperature": 1.0})
    df = load_training_frame(settings)
    X, y, _, _ = prepare_xy(df)
    _, cal_mask, test_mask, _, _ = three_way_date_split(_timestamps(X.index))
    raw_cal = model.predict(X.loc[cal_mask])
    raw_test = model.predict(X.loc[test_mask])
    t = float(calibration.get("temperature", 1.0))
    return (y.loc[cal_mask].to_numpy(), apply_temperature(raw_cal, t),
            y.loc[test_mask].to_numpy(), apply_temperature(raw_test, t))


def run_flat_threshold() -> dict:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "flat_threshold.log")
    y_cal, p_cal, y_test, p_test = _calibration_proba(settings)
    sweep = sweep_flat_threshold(y_cal, p_cal)[0]
    tau = float(sweep["winner"]["tau_flat"])
    logger.info("Calibration: base acc %.4f → tau_flat=%.2f (FLAT F1 %.4f, acc %.4f).",
                sweep["base_accuracy"], tau,
                sweep["winner"]["flat_f1"], sweep["winner"]["accuracy"])
    # Apply ONCE to test. Judge with headline + FLAT F1 + DM vs argmax.
    pred_argmax = np.argmax(p_test, axis=1)
    pred_tau = apply_flat_threshold(p_test, tau)
    m_argmax = classification_metrics(y_test, pred_argmax, p_test)
    m_tau = classification_metrics(y_test, pred_tau, p_test)
    f1 = lambda m: m["classification_report"]["1"]["f1-score"]
    dm = dm_statistic(loss_differential((y_test != pred_argmax).astype(float),
                                       (y_test != pred_tau).astype(float)), lags=5)
    verdict = {
        "tau_flat": tau,
        "calibration": {"base_accuracy": sweep["base_accuracy"], "winner": sweep["winner"]},
        "test_argmax": {"accuracy": m_argmax["accuracy"], "flat_f1": f1(m_argmax),
                        "down_f1": m_argmax["classification_report"]["0"]["f1-score"],
                        "up_f1": m_argmax["classification_report"]["2"]["f1-score"],
                        "majority": m_argmax.get("majority_baseline_accuracy") if isinstance(m_argmax, dict) else None},
        "test_tau": {"accuracy": m_tau["accuracy"], "flat_f1": f1(m_tau),
                     "down_f1": m_tau["classification_report"]["0"]["f1-score"],
                     "up_f1": m_tau["classification_report"]["2"]["f1-score"]},
        "dm_tau_vs_argmax": dm,
    }
    # Majority baseline on test for edge computation.
    import pandas as pd
    counts = pd.Series(y_test).value_counts()
    verdict["test_majority"] = float(counts.max() / len(y_test))
    atomic_write_json(paths.results / "flat_threshold.json", verdict)
    update_state(paths.state, "flat_threshold", status="complete", report=str(paths.results / "flat_threshold.json"))
    logger.info("Test: argmax acc %.4f FLAT-F1 %.4f → tau acc %.4f FLAT-F1 %.4f (DM stat %.2f p %.4f).",
                verdict["test_argmax"]["accuracy"], verdict["test_argmax"]["flat_f1"],
                verdict["test_tau"]["accuracy"], verdict["test_tau"]["flat_f1"],
                dm.get("stat") or 0.0, dm.get("p_value") or 1.0)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description="Track F: tune FLAT threshold on calibration, judge once on test.")
    parser.parse_args()
    run_flat_threshold()


if __name__ == "__main__":
    main()
