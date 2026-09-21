from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ..config import load_settings
from ..inference.predict import run_prediction

app = FastAPI(title="Stock Direction ML API", version="1.0.0")


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
