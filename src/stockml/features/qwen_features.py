from __future__ import annotations

"""Phase 7 feature layer using Qwen sentiment as the score column.

CODE READY — UNTESTED
"""

from pathlib import Path

import pandas as pd

from ..config import load_settings
from ..features.news_context import aggregate_scored_text
from ..qpipeline.contracts import load_q_track_config


def build_qwen_features_for_symbol(
    news_frame: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    symbol: str,
    calendar: pd.DatetimeIndex,
    prefix: str = "qwen_co",
) -> pd.DataFrame:
    """Company-scope Qwen features using the existing groupby path."""
    return aggregate_scored_text(
        news_frame,
        trading_index,
        symbol,
        calendar,
        prefix,
        score_col="qwen_score",
    )


def load_qwen_nightly_frame() -> pd.DataFrame:
    cfg = load_q_track_config().get("nightly", {})
    path = Path(cfg.get("ledger_path", "data/llm/qwen_nightly_ledger.parquet"))
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def build_qwen_features_panel(
    trading_index: pd.DatetimeIndex,
    calendar: pd.DatetimeIndex,
    symbols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build qwen features per symbol from the nightly ledger."""
    settings = load_settings()
    symbols = symbols or settings.symbols
    ledger = load_qwen_nightly_frame()
    if ledger.empty:
        return {symbol: aggregate_scored_text(
            pd.DataFrame(), trading_index, symbol, calendar, "qwen_co", score_col="qwen_score"
        ) for symbol in symbols}
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        sub = ledger[ledger["symbol"].astype(str) == str(symbol)]
        out[symbol] = build_qwen_features_for_symbol(sub, trading_index, symbol, calendar)
    return out


def save_qwen_features_panel(
    trading_index: pd.DatetimeIndex,
    calendar: pd.DatetimeIndex,
    symbols: list[str] | None = None,
) -> list[Path]:
    """Save per-symbol qwen features to data/features/qwen/."""
    from ..utils.io import atomic_write_parquet

    settings = load_settings()
    symbols = symbols or settings.symbols
    panel = build_qwen_features_panel(trading_index, calendar, symbols)
    out_dir = settings.paths.data_features / "qwen"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for symbol, frame in panel.items():
        safe = symbol.replace("^", "index_").replace("/", "_").replace("=", "_").replace(":", "_")
        path = out_dir / f"{safe}_qwen_features.parquet"
        atomic_write_parquet(frame, path)
        written.append(path)
    return written
