"""Optional standalone FastAPI app for the CV lane.

The integrated platform uses stockml.api.cv instead. This app is kept for
developers who want to try the CV model on its own. It never loads a model at
import time, so it starts even when no checkpoint exists.
"""

from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from ..config import TEMPLATES_DIR
from ..adapter import get_status

app = FastAPI(title="ChartSense CV API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}


def _read_image(contents: bytes, content_type: str | None) -> Image.Image:
    if not contents:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="image exceeds 5 MB limit")
    if content_type and content_type.split(";")[0].strip().lower() not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="unsupported image type; use PNG, JPEG or WEBP")
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"cannot decode image: {exc}") from exc
    except Exception as exc:  # includes PIL decompression-bomb errors
        raise HTTPException(status_code=400, detail=f"cannot decode image: {exc}") from exc


@app.get("/health")
def health():
    return get_status()


@app.post("/predict")
async def predict_chart(file: UploadFile = File(...)):
    from ..adapter import analyze_chart
    from ..inference import ModelUnavailable

    image = _read_image(await file.read(), file.content_type)
    try:
        return analyze_chart(image)
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/app", response_class=HTMLResponse)
def frontend():
    index = TEMPLATES_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="CV frontend template missing")
    return index.read_text(encoding="utf-8")
