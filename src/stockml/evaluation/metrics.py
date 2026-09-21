from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, proba=None) -> dict:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    y_arr = np.asarray(y_true, dtype=int)
    nonflat = np.isin(y_arr, [0, 2])
    if nonflat.any():
        result["directional_accuracy_nonflat_actual"] = float(accuracy_score(y_arr[nonflat], np.asarray(y_pred)[nonflat]))
        result["directional_samples_nonflat_actual"] = int(nonflat.sum())
    else:
        result["directional_accuracy_nonflat_actual"] = None
        result["directional_samples_nonflat_actual"] = 0
    pred_arr = np.asarray(y_pred, dtype=int)
    signal_mask = np.isin(pred_arr, [0, 2])
    evaluated_signal = signal_mask & nonflat
    if evaluated_signal.any():
        result["directional_accuracy_when_model_takes_direction"] = float(accuracy_score(y_arr[evaluated_signal], pred_arr[evaluated_signal]))
        result["directional_signal_coverage"] = float(signal_mask.mean())
        result["directional_samples_when_model_takes_direction"] = int(evaluated_signal.sum())
    else:
        result["directional_accuracy_when_model_takes_direction"] = None
        result["directional_signal_coverage"] = float(signal_mask.mean())
        result["directional_samples_when_model_takes_direction"] = 0

    if proba is not None:
        try:
            # Explicit one-vs-rest binarization avoids relying on the deprecated
            # LogisticRegression-style multiclass API and works across supported
            # scikit-learn releases.
            classes = np.array([0, 1, 2])
            y_arr = np.asarray(y_true, dtype=int)
            binary = np.column_stack([(y_arr == c).astype(int) for c in classes])
            result["roc_auc_ovr_macro"] = float(roc_auc_score(binary, proba, average="macro"))
        except ValueError:
            result["roc_auc_ovr_macro"] = None
    return result


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-6)
    logits = np.log(np.clip(proba, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> dict:
    """F-14: multiclass ECE on the predicted class.

    Bins by confidence (max prob); per-bin |accuracy - mean confidence|,
    weighted by bin mass. Returns ece plus the per-bin table for the
    reliability diagram. Pure computation (testable).
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = np.argmax(proba, axis=1)
    conf = proba.max(axis=1)
    correct = (y_true == pred).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({"bin": b, "lo": float(lo), "hi": float(hi), "count": 0,
                         "accuracy": None, "confidence": None, "gap": None})
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        gap = abs(acc - avg_conf)
        ece += (n / len(y_true)) * gap
        bins.append({"bin": b, "lo": float(lo), "hi": float(hi), "count": n,
                     "accuracy": acc, "confidence": avg_conf, "gap": gap})
    return {"ece": float(ece), "n_bins": int(n_bins), "bins": bins}
