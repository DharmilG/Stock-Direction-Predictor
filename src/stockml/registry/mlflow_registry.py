from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow import MlflowClient

from ..config import load_settings


def configure_mlflow(settings):
    uri = settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    return uri


def list_model_versions(settings):
    configure_mlflow(settings)
    client = MlflowClient()
    try:
        return client.search_model_versions(f"name='{settings.mlflow_registered_model}'")
    except Exception:
        return []


def register_model_artifact(settings, model_path: Path) -> None:
    configure_mlflow(settings)
    # Native LightGBM model is logged as an artifact. Registration can be added to a remote MLflow
    # environment once a supported model flavor / serving format is selected for deployment.
    with mlflow.start_run(run_name="model-artifact-registration"):
        mlflow.log_artifact(str(model_path))
        mlflow.set_tag("registered_model_name", settings.mlflow_registered_model)
