from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    price: pd.DataFrame,
    horizon: int = 5,
    volatility_window: int = 20,
    upper_multiplier: float = 1.0,
    lower_multiplier: float = 1.0,
    flat_band: float = 0.25,
) -> pd.DataFrame:
    """Create tradeable next-session-entry triple-barrier labels.

    Prediction timestamp t is assumed to be after t's close. The hypothetical
    trade enters at t+1 open and exits when the upper/lower barrier is first
    touched or at the horizon-day close. High/low are used for barrier tests.
    """
    df = price.sort_index().copy()
    required = {"open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")

    close = df["close"].astype(float)
    vol = close.pct_change().rolling(volatility_window, min_periods=volatility_window).std()
    labels = pd.Series(np.nan, index=df.index, dtype=float)
    event_end = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    event_start = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    entry_price = pd.Series(np.nan, index=df.index, dtype=float)
    exit_price = pd.Series(np.nan, index=df.index, dtype=float)
    realized_return = pd.Series(np.nan, index=df.index, dtype=float)

    for i in range(len(df) - horizon):
        sigma = vol.iloc[i]
        start_idx = i + 1
        if start_idx >= len(df) or not np.isfinite(sigma) or sigma <= 0:
            continue
        entry = float(df["open"].iloc[start_idx])
        if not np.isfinite(entry) or entry <= 0:
            continue

        upper = entry * (1.0 + upper_multiplier * sigma)
        lower = entry * (1.0 - lower_multiplier * sigma)
        future = df.iloc[start_idx : min(len(df), start_idx + horizon)]
        if future.empty:
            continue

        hit = None
        hit_idx = None
        exit_px = None
        # Deterministic rule for a day where both barriers are touched: mark as
        # ambiguous and use the time-barrier close instead of choosing a side.
        for ts, row in future.iterrows():
            high = float(row["high"])
            low = float(row["low"])
            hit_upper = high >= upper
            hit_lower = low <= lower
            if hit_upper and hit_lower:
                hit = 0
                hit_idx = ts
                exit_px = float(row["close"])
                break
            if hit_upper:
                hit = 1
                hit_idx = ts
                exit_px = upper
                break
            if hit_lower:
                hit = -1
                hit_idx = ts
                exit_px = lower
                break

        if hit is None:
            final = future.iloc[-1]
            exit_px = float(final["close"])
            hit_idx = future.index[-1]
            ret = exit_px / entry - 1.0
            flat_threshold = max(0.0, sigma * flat_band)
            hit = 0 if abs(ret) <= flat_threshold else (1 if ret > 0 else -1)

        labels.iloc[i] = hit
        event_start.iloc[i] = df.index[start_idx]
        event_end.iloc[i] = hit_idx
        entry_price.iloc[i] = entry
        exit_price.iloc[i] = float(exit_px)
        realized_return.iloc[i] = float(exit_px / entry - 1.0)

    return pd.DataFrame(
        {
            "label_raw": labels,
            "event_start": event_start,
            "event_end": event_end,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "realized_return": realized_return,
        },
        index=df.index,
    )


def make_labeled_dataset(features: pd.DataFrame, raw_price: pd.DataFrame, settings) -> pd.DataFrame:
    labels = triple_barrier_labels(
        raw_price,
        horizon=settings.triple_barrier_horizon,
        volatility_window=settings.triple_barrier_vol_window,
        upper_multiplier=settings.triple_barrier_up_mult,
        lower_multiplier=settings.triple_barrier_down_mult,
        flat_band=settings.triple_barrier_flat_band,
    )
    df = features.join(labels, how="left")
    df["target"] = df["label_raw"].map({-1: 0, 0: 1, 1: 2})
    return df.dropna(subset=["target", "event_start", "event_end", "entry_price", "exit_price"]).copy()
