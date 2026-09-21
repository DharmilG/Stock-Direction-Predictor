from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class FinBERTSentiment:
    model_name: str = "ProsusAI/finbert"

    def __post_init__(self):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError("Install optional NLP dependencies from requirements-optional.txt to use FinBERT sentiment.") from exc
        self._pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            truncation=True,
            top_k=None,
        )

    def score_texts(self, texts: list[str]) -> np.ndarray:
        outputs = self._pipeline(texts, batch_size=16)
        scores = []
        for row in outputs:
            if isinstance(row, dict):
                row = [row]
            mapping = {item["label"].lower(): float(item["score"]) for item in row}
            scores.append(mapping.get("positive", 0.0) - mapping.get("negative", 0.0))
        return np.asarray(scores, dtype=float)

    def score_news_frame(self, news: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        if news.empty:
            return news.assign(sentiment_score=[])
        out = news.copy()
        out["sentiment_score"] = self.score_texts(out[text_column].astype(str).tolist())
        return out
