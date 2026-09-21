from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, confusion_matrix

from ..config import load_settings
from ..models.lightgbm_train import load_training_frame, prepare_xy, _timestamps
from ..utils.io import atomic_write_json, atomic_write_parquet, read_json
from ..utils.logging import setup_logging
from ..utils.state import update_state
from .metrics import apply_temperature, classification_metrics, expected_calibration_error
from .dm import up_vs_rest_dm
from .flat_threshold import apply_flat_threshold
from ..validation.time_splits import three_way_date_split


def run_evaluation() -> None:
    settings = load_settings()
    paths = settings.paths
    logger = setup_logging(paths.logs / "evaluate.log")
    model_path = paths.artifacts / "models" / "lightgbm_final.txt"
    if not model_path.exists():
        raise FileNotFoundError("Trained LightGBM model not found. Run train first.")
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(model_path))

    df = load_training_frame(settings)
    X, y, _, feature_cols = prepare_xy(df)
    _, _, test_mask, _, test_start = three_way_date_split(_timestamps(X.index))
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    raw_proba = model.predict(X_test)
    temperature_meta = read_json(paths.artifacts / "calibration.json", {"temperature": 1.0})
    proba = apply_temperature(raw_proba, float(temperature_meta.get("temperature", 1.0)))
    # Track F decision rule (config decision.flat_threshold): FLAT iff
    # p_flat >= tau, else argmax(DOWN, UP).
    pred = apply_flat_threshold(proba, float(settings.flat_threshold))

    actual = y_test.to_numpy()
    result = classification_metrics(y_test, pred, proba)
    # F-14: ECE + reliability data on the sealed test split.
    try:
        ece_out = expected_calibration_error(actual, proba, n_bins=10)
        result["ece"] = ece_out["ece"]
        result["ece_bins"] = ece_out["bins"]
    except Exception as exc:
        logger.warning("ECE computation skipped: %s", exc)
        result["ece"] = None
        result["ece_bins"] = []
    class_counts = {int(k): int(v) for k, v in pd.Series(actual).value_counts().sort_index().items()}
    majority_class = int(pd.Series(actual).mode().iloc[0]) if len(actual) else None
    result["test_class_counts"] = class_counts
    result["majority_class"] = majority_class
    result["majority_baseline_accuracy"] = float(max(class_counts.values()) / len(actual)) if class_counts else None
    # B-03: the headlined comparison is vs the MAJORITY baseline, not the linear one.
    # The linear-baseline delta is kept only as a secondary diagnostic.
    if result["majority_baseline_accuracy"] is not None:
        result["lightgbm_minus_majority_accuracy"] = float(result["accuracy"] - result["majority_baseline_accuracy"])
        result["beats_majority_baseline"] = bool(result["lightgbm_minus_majority_accuracy"] > 0)
    else:
        result["lightgbm_minus_majority_accuracy"] = None
        result["beats_majority_baseline"] = False
    result["prediction_class_counts"] = {int(k): int(v) for k, v in pd.Series(pred).value_counts().sort_index().items()}
    actual_nonflat = np.isin(actual, [0, 2])
    pred_nonflat = np.isin(pred, [0, 2])
    directional_mask = actual_nonflat
    result["directional_accuracy_nonflat_actual"] = float(
        accuracy_score(actual[directional_mask], pred[directional_mask])
    ) if directional_mask.any() else None
    strict_direction_mask = actual_nonflat & pred_nonflat
    result["directional_accuracy_when_model_takes_direction"] = float(
        accuracy_score(actual[strict_direction_mask], pred[strict_direction_mask])
    ) if strict_direction_mask.any() else None
    result["directional_predictions_taken"] = int(strict_direction_mask.sum())
    baseline_path = paths.artifacts / "models" / "baseline_logistic.joblib"
    if baseline_path.exists():
        try:
            import joblib
            baseline = joblib.load(baseline_path)
            base_pred = baseline.predict(X_test)
            result["baseline_accuracy"] = float(accuracy_score(y_test, base_pred))
            result["lightgbm_minus_baseline_accuracy"] = result["accuracy"] - result["baseline_accuracy"]
            # H1 guard: DM specifically on UP-vs-rest loss (model vs linear
            # baseline), Newey-West lags for overlapping 5-day events. A
            # headline win via DOWN alone must NOT count as H1 working.
            try:
                result["dm_up"] = up_vs_rest_dm(actual, pred, np.asarray(base_pred, dtype=int), lags=5)
            except Exception as exc:
                logger.warning("UP DM test skipped: %s", exc)
                result["dm_up"] = None
        except Exception as exc:
            logger.warning("Baseline evaluation skipped: %s", exc)
    idx = X_test.index
    symbols = idx.get_level_values("symbol").astype(str) if isinstance(idx, pd.MultiIndex) else pd.Series([settings.symbols[0]] * len(idx))
    timestamps = idx.get_level_values("timestamp") if isinstance(idx, pd.MultiIndex) else idx
    event_cols = {}
    for col in ["event_start", "event_end", "entry_price", "exit_price", "realized_return"]:
        if col in df.columns:
            event_cols[col] = df.loc[X_test.index, col].to_numpy()

    df_pred = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": symbols,
        "actual_label": actual,
        "predicted_label": pred,
        "p_down": proba[:, 0],
        "p_flat": proba[:, 1],
        "p_up": proba[:, 2],
        "correct": (actual == pred).astype(int),
        **event_cols,
    }, index=idx)
    df_pred.index.names = ["timestamp", "symbol"] if isinstance(idx, pd.MultiIndex) else ["timestamp"]

    rolling_window = min(50, max(10, len(df_pred) // 10))
    df_pred["rolling_accuracy"] = df_pred["correct"].rolling(rolling_window, min_periods=5).mean()

    test_path = paths.results / "predictions.parquet"
    atomic_write_parquet(df_pred, test_path)
    atomic_write_json(paths.results / "metrics.json", {"test_start": str(test_start.date()), **result, "n_test": int(len(y_test))})

    # Correct/incorrect timeline.
    plt.figure(figsize=(16, 5))
    x = np.arange(len(df_pred))
    correct = df_pred["correct"].to_numpy(dtype=bool)
    plt.scatter(x[correct], df_pred.loc[correct, "predicted_label"], marker="o", label="Correct")
    plt.scatter(x[~correct], df_pred.loc[~correct, "predicted_label"], marker="x", label="Incorrect")
    plt.title("Test-set prediction correctness over time")
    plt.xlabel("Test observation")
    plt.ylabel("Predicted class (0=DOWN, 1=FLAT, 2=UP)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.results / "prediction_correctness.png", dpi=160)
    plt.close()

    plt.figure(figsize=(16, 5))
    plt.plot(df_pred["rolling_accuracy"].to_numpy())
    # Honest overlay: the majority-class baseline, not a fixed 0.5 line.
    plt.axhline(result.get("majority_baseline_accuracy") or 0.5, linestyle="--", linewidth=1,
                label=f"Majority baseline ({(result.get('majority_baseline_accuracy') or 0.0):.4f})")
    plt.legend()
    plt.title(f"Rolling directional accuracy (window={rolling_window})")
    plt.xlabel("Test observation")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(paths.results / "rolling_accuracy.png", dpi=160)
    plt.close()

    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1, 2], ["DOWN", "FLAT", "UP"])
    plt.yticks([0, 1, 2], ["DOWN", "FLAT", "UP"])
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(paths.results / "confusion_matrix.png", dpi=160)
    plt.close()

    # Confidence bucket diagnostic.
    df_pred["confidence"] = proba.max(axis=1)
    df_pred["confidence_bucket"] = pd.cut(df_pred["confidence"], bins=[0.0, .55, .60, .65, .70, .80, 1.01], right=False)
    bucket = df_pred.groupby("confidence_bucket", observed=False)["correct"].agg(["count", "mean"]).reset_index()
    bucket.to_csv(paths.results / "confidence_accuracy.csv", index=False)

    # F-14: reliability diagram (predicted confidence vs realized accuracy).
    try:
        bins = [b for b in (result.get("ece_bins") or []) if b.get("count")]
        if bins:
            plt.figure(figsize=(7, 6))
            xs = [(b["lo"] + b["hi"]) / 2 for b in bins]
            accs = [b["accuracy"] for b in bins]
            confs = [b["confidence"] for b in bins]
            plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
            plt.plot(confs, accs, marker="o", label="Model")
            for b in bins:
                plt.plot([b["confidence"], b["confidence"]], [b["accuracy"], b["confidence"]], color="red", linewidth=2, alpha=0.6)
            plt.xlabel("Mean predicted confidence")
            plt.ylabel("Realized accuracy")
            plt.title(f"Reliability diagram (ECE={result.get('ece') or float('nan'):.4f})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(paths.results / "calibration.png", dpi=160)
            plt.close()
    except Exception as exc:
        logger.warning("Reliability diagram skipped: %s", exc)

    # SHAP feature impact on a bounded sample to keep resource use predictable.
    try:
        sample = X_test.sample(min(1000, len(X_test)), random_state=settings.random_seed)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            mean_abs = np.mean([np.abs(v) for v in shap_values], axis=0).mean(axis=0)
        else:
            values = np.asarray(shap_values)
            if values.ndim == 3:
                mean_abs = np.abs(values).mean(axis=(0, 2))
            elif values.ndim == 2:
                mean_abs = np.abs(values).mean(axis=0)
            else:
                raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
        fi = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.asarray(mean_abs).reshape(-1)}).sort_values("mean_abs_shap", ascending=False)
        fi.to_csv(paths.results / "shap_feature_importance.csv", index=False)
        plt.figure(figsize=(10, 7))
        top = fi.head(20).sort_values("mean_abs_shap")
        plt.barh(top["feature"], top["mean_abs_shap"])
        plt.title("Top 20 SHAP feature impacts")
        plt.tight_layout()
        plt.savefig(paths.results / "shap_summary.png", dpi=160)
        plt.close()
    except Exception as exc:
        logger.warning("SHAP diagnostic skipped: %s", exc)

    logger.info(
        "3-class accuracy: %.4f | Balanced accuracy: %.4f | MCC: %.4f | "
        "Directional accuracy (non-flat actual): %s | Directional accuracy (model takes direction): %s | Coverage: %.4f",
        result["accuracy"], result["balanced_accuracy"], result["mcc"],
        f"{result.get('directional_accuracy_nonflat_actual'):.4f}" if result.get("directional_accuracy_nonflat_actual") is not None else "NA",
        f"{result.get('directional_accuracy_when_model_takes_direction'):.4f}" if result.get("directional_accuracy_when_model_takes_direction") is not None else "NA",
        result.get("directional_signal_coverage", 0.0),
    )
    if result.get("majority_baseline_accuracy") is not None:
        edge = result["accuracy"] - result["majority_baseline_accuracy"]
        logger.info(
            "EDGE OVER MAJORITY: %+.4f pp  %s",
            edge * 100.0, "✓ ABOVE BASELINE" if edge > 0 else "✗ BELOW BASELINE",
        )
    update_state(paths.state, "evaluate", status="complete", test_start=str(test_start.date()), metrics=str(paths.results / "metrics.json"))
    # F-22: every evaluate ends with the §12.2 block (economics show n/a until backtest runs).
    from .final_report import print_final_report as _print_final
    _print_final()


if __name__ == "__main__":
    run_evaluation()
