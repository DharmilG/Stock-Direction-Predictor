from __future__ import annotations

import numpy as np
import pandas as pd


def confidence_position_size(p_up: float, p_down: float, threshold: float = 0.55, max_position: float = 1.0) -> float:
    """Simple confidence-scaled signal used by the research backtest.

    Returns +position for UP confidence, -position for DOWN confidence, otherwise 0.
    This is a research policy, not a broker-specific order-sizing rule.
    """
    if p_up >= threshold and p_up > p_down:
        return float(max_position * min(1.0, (p_up - threshold) / max(1e-9, 1 - threshold)))
    if p_down >= threshold and p_down > p_up:
        return float(-max_position * min(1.0, (p_down - threshold) / max(1e-9, 1 - threshold)))
    return 0.0


def enforce_gross_exposure(signals: pd.Series, max_gross: float = 1.0) -> pd.Series:
    gross = signals.abs().sum()
    if gross <= max_gross or gross == 0:
        return signals
    return signals * (max_gross / gross)
