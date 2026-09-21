from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import load_settings
from ..models.lightgbm_train import load_training_frame, prepare_xy, _timestamps
from ..validation.time_splits import three_way_date_split
from ..utils.io import atomic_write_json
from ..utils.state import update_state


def train_baseline() -> None:
    settings = load_settings()
    df = load_training_frame(settings)
    X, y, _, _ = prepare_xy(df)
    train_mask, _, _, _, _ = three_way_date_split(_timestamps(X.index))
    model = Pipeline([
        # G1: delivery_* exist only from 2020+ (NaN = no coverage, B-07).
        # The linear anchor median-imputes; the primary LightGBM uses native
        # missing handling. Baseline stays a secondary diagnostic.
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=settings.random_seed)),
    ])
    model.fit(X.loc[train_mask], y.loc[train_mask])
    out = settings.paths.artifacts / "models" / "baseline_logistic.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    atomic_write_json(settings.paths.artifacts / "baseline_metadata.json", {"features": list(X.columns), "rows": int(train_mask.sum())})
    update_state(settings.paths.state, "baseline", status="complete", model=str(out))


if __name__ == "__main__":
    train_baseline()
