from __future__ import annotations

import pandas as pd


def three_way_date_split(index: pd.DatetimeIndex, train_fraction: float = 0.60, validation_fraction: float = 0.20):
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(index).unique()))
    if len(dates) < 20:
        raise ValueError("At least 20 unique observations/dates are recommended for train/validation/test splitting.")
    train_pos = max(1, int(len(dates) * train_fraction))
    valid_pos = max(train_pos + 1, int(len(dates) * (train_fraction + validation_fraction)))
    train_end = dates[train_pos]
    valid_end = dates[min(valid_pos, len(dates) - 1)]
    dt = pd.DatetimeIndex(index)
    train_mask = dt < train_end
    validation_mask = (dt >= train_end) & (dt < valid_end)
    test_mask = dt >= valid_end
    return train_mask, validation_mask, test_mask, train_end, valid_end
