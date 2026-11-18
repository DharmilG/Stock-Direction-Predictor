"""Read-only-by-design CV endpoints.

Deliberately NOT under /dashboard/*: /dashboard routes only read artifacts,
while /cv/analyze is stateless inference on a user-supplied image. It writes
nothing, fetches nothing and never trains.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from PIL import Image, UnidentifiedImageError

from .dashboard_support import json_safe, now_ist, to_iso

router = APIRouter(prefix="/cv", tags=["cv"])

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}


def _envelope(data: Any, artifacts: list[str], *, as_of: Any = None, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "data": json_safe(data),
        "as_of": to_iso(as_of),
        "generated_at": now_ist().isoformat(),
        "source": {"kind": "runtime", "artifacts": artifacts, "contract": "cv-prediction-v1"},
        "warnings": warnings or [],
    }


def _cv_root() -> Path:
    return Path(__file__).resolve().parents[3] / "CV"


def _cv_modules():
    """Import the CV lane lazily so the API boots without CV dependencies."""
    cv_root = _cv_root()
    if str(cv_root) not in sys.path:
        sys.path.insert(0, str(cv_root))
    from chartsense import adapter

    return adapter


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        return []
    return records


def _parse_image(contents: bytes, content_type: str | None) -> Image.Image:
    if not contents:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="image exceeds 5 MB limit")
    if content_type and content_type.split(";")[0].strip().lower() not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="unsupported image type; use PNG, JPEG or WEBP")
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"cannot decode image: {exc}") from exc
    except Exception as exc:  # e.g. PIL DecompressionBombError
        raise HTTPException(status_code=400, detail=f"cannot decode image: {exc}") from exc


@router.get("/health")
def cv_health() -> dict[str, Any]:
    try:
        adapter = _cv_modules()
        status = adapter.get_status()
    except Exception as exc:
        return _envelope({"ready": False, "error": str(exc)}, [], warnings=["CV lane unavailable"])
    warnings = [] if status.get("ready") else ["CV checkpoint not trained yet"]
    return _envelope(status, [status.get("model", "")], warnings=warnings)


@router.get("/training")
def cv_training() -> dict[str, Any]:
    """Per-epoch training history and stage progress. Read-only."""
    root = _cv_root()
    stages: dict[str, Any] = {}
    for name in ("train_stage1", "train_stage2"):
        state = _read_json(root / "artifacts" / "state" / f"{name}.json")
        history = _read_jsonl(root / "artifacts" / "metrics" / f"{name}.jsonl")
        stages[name] = {
            "status": state.get("status", "not-started"),
            "epoch": state.get("epoch"),
            "best_metric": state.get("best_metric"),
            "fingerprint": state.get("fingerprint"),
            "history": history,
        }
    return _envelope({"stages": stages}, [str(root / "artifacts" / "metrics")])


@router.get("/evaluation")
def cv_evaluation() -> dict[str, Any]:
    """Evaluation metrics, confusion matrix and classification report. Read-only."""
    root = _cv_root()
    metrics_path = root / "evaluation" / "cv_metrics.json"
    if not metrics_path.exists():
        return _envelope({"ready": False}, [], warnings=["No CV evaluation has been run yet"])
    metrics = _read_json(metrics_path)
    matrix_path = root / "evaluation" / "cv_confusion_matrix.png"
    matrix_b64 = None
    if matrix_path.exists():
        try:
            matrix_b64 = base64.b64encode(matrix_path.read_bytes()).decode("utf-8")
        except OSError:
            matrix_b64 = None
    report_path = root / "evaluation" / "direction_report.txt"
    report = ""
    if report_path.exists():
        try:
            report = report_path.read_text(encoding="utf-8")
        except OSError:
            report = ""
    data = {"ready": True, **metrics, "confusion_matrix_base64": matrix_b64, "classification_report": report}
    return _envelope(data, [str(metrics_path)])


@router.post("/analyze")
async def cv_analyze(request: Request) -> dict[str, Any]:
    # Request-based parsing keeps route registration (and therefore API startup)
    # working even before python-multipart is installed.
    try:
        form = await request.form()
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="python-multipart is required for image upload") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid multipart body: {exc}") from exc
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="missing 'file' field")
    image = _parse_image(await upload.read(), getattr(upload, "content_type", None))
    try:
        adapter = _cv_modules()
        result = adapter.analyze_chart(image)
    except HTTPException:
        raise
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"CV dependencies are not installed: {exc}") from exc
    except Exception as exc:
        if type(exc).__name__ == "ModelUnavailable" or isinstance(exc, FileNotFoundError):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=f"CV inference failed: {exc}") from exc

    model_path = ""
    try:
        model_path = adapter.get_status().get("model", "")
    except Exception:
        pass
    return _envelope(result, [model_path], warnings=result.get("warnings") or [])
