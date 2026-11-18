from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import load_settings
from ..inference.predict import run_prediction
from .compare import router as compare_router
from .cv import router as cv_router
from .dashboard import router as dashboard_router
from .dashboard_model import explain_router as dashboard_model_explain_router
from .dashboard_model import router as dashboard_model_router
from .dashboard_q0 import router as dashboard_q0_router
from .dashboard_v2 import router as dashboard_v2_router

app = FastAPI(title="Stock Direction ML API", version="1.1.0")

# The dashboard dev server runs on a different origin (Vite: 5173). Only the
# configured local origins are allowed; this is a research tool, not a public API.
_DEFAULT_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("DASHBOARD_ALLOWED_ORIGINS", ",".join(_DEFAULT_ORIGINS)).split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)
app.include_router(dashboard_q0_router)
app.include_router(dashboard_v2_router)
app.include_router(dashboard_model_router)
app.include_router(dashboard_model_explain_router)
app.include_router(compare_router)
app.include_router(cv_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predictions")
def predictions():
    try:
        return run_prediction().to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/predictions/{symbol}")
def prediction(symbol: str):
    settings = load_settings()
    if symbol not in settings.symbols:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} is not configured")
    try:
        rows = run_prediction()
        match = rows[rows["symbol"] == symbol]
        if match.empty:
            raise HTTPException(status_code=404, detail="No prediction available")
        return match.iloc[0].to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Optional: serve a production build of the dashboard from the same origin.
# Mounted last so the API routes above always win.
_STATIC_DIR = os.getenv("DASHBOARD_STATIC_DIR", "").strip()
if _STATIC_DIR:
    static_root = Path(_STATIC_DIR).expanduser().resolve()
    assets = static_root / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="dashboard-assets")

    @app.get("/", include_in_schema=False)
    def dashboard_index():
        index = static_root / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="Dashboard build not found")
        return FileResponse(index)
