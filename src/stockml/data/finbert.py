from __future__ import annotations

from pathlib import Path
from typing import Iterable


class FinBERTScorer:
    """Lazy-loaded financial sentiment model.

    Default model is ProsusAI/finbert, which exposes positive/negative/neutral
    probabilities. A local/cache directory may be supplied for reproducible
    deployment.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", cache_dir: str | Path | None = None, device: int = -1):
        import os
        from transformers import pipeline
        # transformers>=4.5x forwards unknown pipeline kwargs to the
        # tokenizer (which rejects cache_dir) — configure cache via env.
        if cache_dir:
            os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
        self.pipe = pipeline("text-classification", model=model_name, device=device)

    def __call__(self, texts: Iterable[str]) -> tuple[list[str], list[float]]:
        labels: list[str] = []
        scores: list[float] = []
        for item in self.pipe(list(texts), truncation=True, max_length=256, batch_size=16):
            label = str(item["label"]).lower()
            value = float(item["score"])
            if "positive" in label:
                signed = value
                normalized = "positive"
            elif "negative" in label:
                signed = -value
                normalized = "negative"
            else:
                signed = 0.0
                normalized = "neutral"
            labels.append(normalized)
            scores.append(signed)
        return labels, scores
