"""Lazy, cached CV model loading and prediction.

Importing this module is cheap and safe: torch, torchvision and the model are
imported only when a prediction is actually requested. A missing or corrupt
checkpoint raises ModelUnavailable instead of crashing the API at startup.
"""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

from PIL import Image

from ..config import BEST_MODEL_PATH, DIRECTIONS, PATTERNS


class ModelUnavailable(RuntimeError):
    """The CV checkpoint is missing, unreadable or incompatible."""


@lru_cache(maxsize=4)
def _load_cached(path_str: str, mtime_ns: int, size: int):
    import torch

    from ..model.chartsense import ChartSenseModel

    payload = torch.load(path_str, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model = ChartSenseModel()
    model.load_state_dict(state)
    model.eval()
    return model


def clear_model_cache() -> None:
    _load_cached.cache_clear()


def load_cv_model(path: Path | str | None = None):
    """Load the CV model, rebuilding the cache when the checkpoint changes."""
    target = Path(path) if path is not None else BEST_MODEL_PATH
    if not target.exists():
        raise ModelUnavailable(
            f"CV checkpoint not found: {target}. "
            "Generate data and run training before using /cv/analyze."
        )
    try:
        stat = target.stat()
        return _load_cached(str(target), stat.st_mtime_ns, stat.st_size)
    except Exception as exc:
        raise ModelUnavailable(f"cannot load CV checkpoint {target}: {exc}") from exc


def predict(image: Image.Image, model) -> dict:
    import torch
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        pattern_out, direction_out = model(input_tensor)
    pattern_probs = torch.softmax(pattern_out, dim=1)[0]
    direction_probs = torch.softmax(direction_out, dim=1)[0]
    pattern_index = int(pattern_probs.argmax().item())
    direction_index = int(direction_probs.argmax().item())
    return {
        "pattern": PATTERNS[pattern_index],
        "pattern_confidence": round(float(pattern_probs[pattern_index].item()), 3),
        "direction": DIRECTIONS[direction_index],
        "direction_confidence": round(float(direction_probs[direction_index].item()), 3),
        "all_pattern_scores": {PATTERNS[i]: round(float(pattern_probs[i].item()), 3) for i in range(len(PATTERNS))},
        "all_direction_scores": {DIRECTIONS[i]: round(float(direction_probs[i].item()), 3) for i in range(len(DIRECTIONS))},
    }


def image_to_base64(image_array) -> str:
    from PIL import Image as PILImage

    image = PILImage.fromarray(image_array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def run_inference(image_source: Image.Image | Path | str):
    """Convenience helper used by scripts and the standalone CV app."""
    image = image_source if isinstance(image_source, Image.Image) else Image.open(image_source).convert("RGB")
    from ..training.gradcam import get_gradcam

    model = load_cv_model()
    prediction = predict(image, model)
    return {
        "prediction": prediction,
        "gradcam_image": image_to_base64(get_gradcam(image, model=model)),
    }


__all__ = ["ModelUnavailable", "load_cv_model", "clear_model_cache", "predict", "image_to_base64", "run_inference"]
