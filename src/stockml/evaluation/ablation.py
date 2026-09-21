from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef

from ..config import load_settings
from ..models.lightgbm_train import _sample_weights, _timestamps, load_training_frame, prepare_xy
from ..validation.time_splits import three_way_date_split
from ..validation.walk_forward import PurgedWalkForwardSplit


def _score(y_true, pred) -> float:
    balanced = balanced_accuracy_score(y_true, pred)
    mcc = matthews_corrcoef(y_true, pred)
    return float(0.5 * balanced + 0.25 * (mcc + 1.0))


def run_ablation() -> None:
    """Out-of-sample comparison of market-only vs market+news/context features."""
    settings = load_settings()
    df = load_training_frame(settings)
    X, y, events, _ = prepare_xy(df)
    splitter = PurgedWalkForwardSplit(settings.n_splits, settings.purge_days, settings.embargo_days)
    folds = list(splitter.split(_timestamps(X.index), events))

    news_cols = [c for c in X.columns if str(c).startswith("news_")]
    context_prefixes = ("calendar_", "fiscal_", "earnings_", "is_month_", "is_quarter_", "days_to_")
    context_cols = [c for c in X.columns if str(c).startswith(context_prefixes)]
    if not news_cols:
        raise RuntimeError("No news features are present. Run news ingestion and feature engineering first.")

    groups = {
        "market_only": [c for c in X.columns if c not in set(news_cols + context_cols)],
        "multimodal": X.columns.tolist(),
    }
    results = []
    for name, cols in groups.items():
        fold_scores, fold_acc = [], []
        for fold in folds:
            tr, va = fold.train_idx, fold.valid_idx
            train = lgb.Dataset(X.iloc[tr][cols], label=y.iloc[tr], weight=_sample_weights(y.iloc[tr], settings.class_weighting))
            params = {
                "objective": "multiclass", "num_class": 3, "verbosity": -1,
                "learning_rate": 0.02, "num_leaves": 31, "max_depth": 6,
                "min_data_in_leaf": 50, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 1,
                "lambda_l1": 0.1, "lambda_l2": 0.1, "seed": settings.random_seed,
            }
            booster = lgb.train(params, train, num_boost_round=min(settings.num_boost_round, 300))
            pred = np.argmax(booster.predict(X.iloc[va][cols]), axis=1)
            fold_scores.append(_score(y.iloc[va], pred))
            fold_acc.append(accuracy_score(y.iloc[va], pred))
        results.append({
            "model": name,
            "mean_selection_score": float(np.mean(fold_scores)),
            "mean_accuracy": float(np.mean(fold_acc)),
            "feature_count": len(cols),
        })

    out = settings.paths.results / "multimodal_ablation.json"
    out.write_text(json.dumps({
        "news_feature_count": len(news_cols),
        "context_feature_count": len(context_cols),
        "results": results,
    }, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_ablation()
