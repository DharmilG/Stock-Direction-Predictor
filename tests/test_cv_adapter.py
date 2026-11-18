"""Adapter contract tests. Uses fakes — no checkpoint, no real inference."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
CV_ROOT = REPO / "CV"
if str(CV_ROOT) not in sys.path:
    sys.path.insert(0, str(CV_ROOT))

from chartsense import adapter  # noqa: E402


def test_status_reports_not_ready_without_checkpoint(monkeypatch):
    monkeypatch.setattr(adapter, "BEST_MODEL_PATH", CV_ROOT / "models" / "definitely_missing.pth")
    status = adapter.get_status()
    assert status["ready"] is False
    assert set(status["signal_classes"]) == {"UP", "FLAT", "DOWN"}
    assert len(status["pattern_classes"]) == 6
    assert "train_stage1" in status["stages"]


def test_analyze_chart_maps_direction_and_survives_gradcam_failure(monkeypatch):
    monkeypatch.setattr(adapter, "load_cv_model", lambda: object())
    monkeypatch.setattr(
        adapter,
        "predict",
        lambda image, model: {
            "pattern": "bull_flag",
            "pattern_confidence": 0.91,
            "direction": "bullish",
            "direction_confidence": 0.68,
            "all_pattern_scores": {name: 0.1 for name in adapter.PATTERNS},
            "all_direction_scores": {"neutral": 0.1, "bullish": 0.68, "bearish": 0.22},
        },
    )
    image = Image.new("RGB", (32, 32), color="white")
    result = adapter.analyze_chart(image)
    assert result["signal"] == "UP"
    assert result["pattern"] == "bull_flag"
    assert result["direction_probs"]["DOWN"] == 0.22
    assert result["gradcam_base64"] is None
    assert result["warnings"], "a missing GradCAM dependency must produce a warning, not a crash"


def test_analyze_chart_returns_gradcam_when_available(monkeypatch):
    monkeypatch.setattr(adapter, "load_cv_model", lambda: object())
    monkeypatch.setattr(
        adapter,
        "predict",
        lambda image, model: {
            "pattern": "double_top",
            "pattern_confidence": 0.8,
            "direction": "bearish",
            "direction_confidence": 0.7,
            "all_pattern_scores": {name: 0.1 for name in adapter.PATTERNS},
            "all_direction_scores": {"neutral": 0.1, "bullish": 0.2, "bearish": 0.7},
        },
    )
    monkeypatch.setattr(adapter, "image_to_base64", lambda array: "AAAA")

    fake_parent = type(sys)("chartsense.training")
    fake_parent.__path__ = []
    fake_module = type(sys)("chartsense.training.gradcam")
    fake_module.get_gradcam = lambda image, model=None: "array"
    monkeypatch.setitem(sys.modules, "chartsense.training", fake_parent)
    monkeypatch.setitem(sys.modules, "chartsense.training.gradcam", fake_module)

    result = adapter.analyze_chart(Image.new("RGB", (16, 16)))
    assert result["signal"] == "DOWN"
    assert result["gradcam_base64"] == "AAAA"


def test_model_unavailable_is_a_clear_error():
    from chartsense.api.inference import ModelUnavailable

    with pytest.raises(ModelUnavailable):
        adapter.load_cv_model(CV_ROOT / "models" / "missing.pth")
