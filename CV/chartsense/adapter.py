"""Bridge between the CV model and the platform API / dashboard."""

from __future__ import annotations

from PIL import Image

from .config import BEST_MODEL_PATH, DIRECTIONS, PATTERNS, SIGNAL_MAP
from .fault import read_state
from .api.inference import ModelUnavailable, image_to_base64, load_cv_model, predict


def get_status() -> dict:
    """Read-only readiness/progress snapshot used by GET /cv/health."""
    return {
        "ready": BEST_MODEL_PATH.exists(),
        "model": str(BEST_MODEL_PATH),
        "signal_classes": [SIGNAL_MAP[d] for d in DIRECTIONS],
        "pattern_classes": list(PATTERNS),
        "stages": {
            name: read_state(name)
            for name in ("generate_synthetic", "generate_own", "train_stage1", "train_stage2", "evaluate")
        },
    }


def analyze_chart(image: Image.Image) -> dict:
    """Run one inference and return a dashboard-ready payload."""
    model = load_cv_model()
    prediction = predict(image, model)
    warnings: list[str] = []

    gradcam_base64 = None
    try:
        from .training.gradcam import get_gradcam

        gradcam_base64 = image_to_base64(get_gradcam(image, model=model))
    except Exception as exc:  # prediction is still useful without the heatmap
        warnings.append(f"GradCAM unavailable: {exc}")

    return {
        "signal": SIGNAL_MAP[prediction["direction"]],
        "signal_confidence": prediction["direction_confidence"],
        "pattern": prediction["pattern"],
        "pattern_confidence": prediction["pattern_confidence"],
        "pattern_scores": prediction["all_pattern_scores"],
        "direction_probs": {SIGNAL_MAP[key]: value for key, value in prediction["all_direction_scores"].items()},
        "gradcam_base64": gradcam_base64,
        "image_size": list(image.size),
        "warnings": warnings,
    }


__all__ = ["analyze_chart", "get_status", "ModelUnavailable"]
