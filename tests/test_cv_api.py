"""CV endpoint tests. No checkpoint and no training required."""

from __future__ import annotations

import io

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from stockml.api.cv import _parse_image
from stockml.api.server import app


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 24), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_cv_health_is_available_without_a_checkpoint():
    client = TestClient(app)
    response = client.get("/cv/health")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"data", "as_of", "generated_at", "source", "warnings"}
    assert isinstance(payload["data"]["ready"], bool)
    assert payload["source"]["contract"] == "cv-prediction-v1"


def test_parse_image_validation():
    with pytest.raises(HTTPException) as empty:
        _parse_image(b"", "image/png")
    assert empty.value.status_code == 400

    with pytest.raises(HTTPException) as wrong_type:
        _parse_image(_png_bytes(), "text/plain")
    assert wrong_type.value.status_code == 400

    with pytest.raises(HTTPException) as huge:
        _parse_image(b"x" * (5 * 1024 * 1024 + 1), "image/png")
    assert huge.value.status_code == 400

    assert _parse_image(_png_bytes(), "image/png").size == (24, 24)


def test_cv_training_endpoint_returns_stage_history():
    client = TestClient(app)
    response = client.get("/cv/training")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"data", "as_of", "generated_at", "source", "warnings"}
    stages = payload["data"]["stages"]
    assert set(stages) == {"train_stage1", "train_stage2"}
    assert isinstance(stages["train_stage2"]["history"], list)


def test_cv_evaluation_endpoint_without_artifacts_reports_not_ready(tmp_path, monkeypatch):
    import stockml.api.cv as cv_module

    monkeypatch.setattr(cv_module, "_cv_root", lambda: tmp_path)
    client = TestClient(app)
    response = client.get("/cv/evaluation")
    assert response.status_code == 200
    assert response.json()["data"]["ready"] is False


def test_analyze_without_checkpoint_returns_503(tmp_path, monkeypatch):
    pytest.importorskip("multipart", reason="python-multipart is installed with the CV overlay")
    from stockml.api.cv import _cv_modules

    adapter = _cv_modules()
    monkeypatch.setattr(adapter, "BEST_MODEL_PATH", tmp_path / "missing.pth")
    inference_module = __import__("chartsense.api.inference", fromlist=["BEST_MODEL_PATH"])
    monkeypatch.setattr(inference_module, "BEST_MODEL_PATH", tmp_path / "missing.pth")
    client = TestClient(app)
    response = client.post("/cv/analyze", files={"file": ("chart.png", _png_bytes(), "image/png")})
    assert response.status_code == 503


def test_analyze_rejects_wrong_file_type():
    pytest.importorskip("multipart", reason="python-multipart is installed with the CV overlay")
    client = TestClient(app)
    response = client.post("/cv/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
