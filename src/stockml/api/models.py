"""The model registry.

The dashboard describes one model at a time and the header picks which. Every
route that reads a model-owned artefact resolves its paths here rather than
taking ``load_settings().paths`` directly, so "show v2" is a query parameter and
never a second set of endpoints.

``v1`` is the served incumbent. ``v2`` is the research twin written by
``python -m stockml.v2.train`` and completed by ``python -m stockml.v2.stages``.
It is registered so it can be *measured* next to v1, never so it can be served:
``servable`` is false and nothing here promotes it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from ..config import Paths, Settings, load_settings
from ..v2.workspace import V2_ROOT, settings_v2
from .dashboard_support import artifact_paths, file_meta, json_file

DEFAULT_MODEL = "v1"

# Artefacts a model needs before any of its pages can render at all.
CORE = ("model", "metrics", "model_metadata", "prediction_history", "predictions")

# Artefacts an individual page can live without, in the order the v2 stages earn
# them: evaluate -> backtest -> monitor -> report -> predict.
OPTIONAL = ("shap", "backtest_metrics", "monitoring", "model_report", "explanations")

STAGES_COMMAND = "python -m stockml.v2.stages all"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    status: str  # "serving" | "research"
    servable: bool
    settings: Settings

    @property
    def paths(self) -> Paths:
        return self.settings.paths


def _files(paths: Paths) -> dict[str, Path]:
    """The model-owned artefacts, by the name the registry reports them under."""
    shared = artifact_paths(paths)
    return {
        "model": paths.artifacts / "models" / "lightgbm_final.txt",
        "metrics": shared["metrics"],
        "model_metadata": shared["model_metadata"],
        "prediction_history": shared["prediction_history"],
        "predictions": shared["predictions"],
        "shap": shared["shap"],
        "backtest_metrics": shared["backtest_metrics"],
        "monitoring": shared["monitoring"],
        "model_report": paths.results / "model_report.json",
        "explanations": paths.results / "explanations_splits.parquet",
    }


def _v2_settings(base: Settings) -> Settings:
    """v2 settings with the serving threshold taken from its own selection.

    The v2 workspace carries the incumbent's 0.4 flat band in ``config.yaml``,
    but the v2 rule is whatever ``v2/train.py`` selected (currently "never FLAT,
    argmax of DOWN and UP", which is tau_flat = 1.01). Serving the v2 booster
    under the incumbent's band would describe a model that was never fitted.
    """
    settings = settings_v2(base)
    selection = json_file(V2_ROOT / "artifacts" / "v2_selection.json") or {}
    tau = (selection.get("policy") or {}).get("tau_flat")
    if tau is None:
        return settings
    return replace(settings, flat_threshold=float(tau))


@lru_cache(maxsize=1)
def registry() -> dict[str, ModelSpec]:
    base = load_settings()
    return {
        "v1": ModelSpec("v1", "v1 · incumbent", "serving", True, base),
        "v2": ModelSpec("v2", "v2 · research", "research", False, _v2_settings(base)),
    }


def resolve(model: str | None = None) -> ModelSpec:
    key = (model or DEFAULT_MODEL).strip().lower()
    spec = registry().get(key)
    if spec is None:
        known = ", ".join(sorted(registry()))
        raise HTTPException(status_code=404, detail=f"Unknown model {model!r}. Known models: {known}")
    return spec


def model_context(model: str | None = None) -> tuple[Settings, Paths]:
    """``(settings, paths)`` for the selected model. Route bodies use this."""
    spec = resolve(model)
    return spec.settings, spec.paths


def describe(spec: ModelSpec) -> dict[str, Any]:
    """Per-model readiness, so the header can disable an unbuilt model."""
    files = _files(spec.paths)
    artifacts = {name: file_meta(path) for name, path in files.items()}
    missing = [name for name in CORE if not artifacts[name]["available"]]
    missing += [name for name in OPTIONAL if not artifacts[name]["available"]]
    core_ready = all(artifacts[name]["available"] for name in CORE)
    return {
        "id": spec.id,
        "label": spec.label,
        "status": spec.status,
        "servable": spec.servable,
        "root": str(spec.paths.root),
        "ready": core_ready,
        "missing": missing,
        "artifacts": artifacts,
        "latest": max((a["modified_at"] for a in artifacts.values() if a["modified_at"]), default=None),
    }


def registry_warnings() -> list[str]:
    """Comparability warnings for the pair, not for either model alone.

    A comparison is only like-for-like while both sides were fitted on the same
    dataset snapshot. The signature is copied into v2's metadata at fit time, so
    a platform data refresh or an incumbent retrain shows up here immediately.
    """
    specs = registry()
    incumbent = json_file(_files(specs["v1"].paths)["model_metadata"]) or {}
    candidate = json_file(_files(specs["v2"].paths)["model_metadata"]) or {}
    warnings: list[str] = []
    if not candidate:
        warnings.append(
            "v2 has not been fitted yet, so it cannot be selected. Run "
            f"`python -m stockml.v2.train` and then `{STAGES_COMMAND}`."
        )
        return warnings
    left = incumbent.get("dataset_signature")
    # v2's own ``dataset_signature`` hashes its frame, which deliberately spans
    # more history than the incumbent's did, so the two values can never match.
    # The meaningful question is whether the incumbent's snapshot moved *since v2
    # copied it* — which is what ``incumbent_signature`` records at fit time.
    at_fit = candidate.get("incumbent_signature")
    if left and at_fit and left != at_fit:
        warnings.append(
            "The incumbent has been refit since v2 was copied "
            f"({left[:12]} now, {at_fit[:12]} when v2 fitted). Re-run "
            "`python -m stockml.v2.train` before reading the comparison."
        )
    if left and not at_fit:
        warnings.append(
            "v2 does not record the incumbent's signature at fit time, so the pair "
            "cannot be proven comparable. Re-run `python -m stockml.v2.train`."
        )
    return warnings
