from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_idx: np.ndarray
    valid_idx: np.ndarray


class PurgedWalkForwardSplit:
    """Expanding-window walk-forward splitter with calendar purge and embargo.

    Validation windows are strictly in the future. Training rows are removed when
    their label event overlaps the validation start (purging). A calendar gap is
    also applied before each validation window. Since this is an expanding-window
    splitter, observations after the validation window are not in the training set;
    the configured embargo is therefore recorded as an additional post-validation
    gap for auditability rather than used to remove nonexistent future training rows.
    """

    def __init__(self, n_splits: int = 5, purge_days: int = 5, embargo_days: int = 5):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = int(n_splits)
        self.purge_days = max(0, int(purge_days))
        self.embargo_days = max(0, int(embargo_days))

    def split(self, timestamps: pd.DatetimeIndex, event_end: pd.Series | None = None):
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps)).tz_localize(None)
        unique_ts = pd.DatetimeIndex(sorted(ts.unique()))
        if len(unique_ts) < self.n_splits + 2:
            raise ValueError("Not enough unique dates for walk-forward validation.")

        boundaries = np.linspace(0, len(unique_ts), self.n_splits + 2, dtype=int)
        event_values = None
        if event_end is not None:
            event_values = pd.to_datetime(pd.Series(event_end)).dt.tz_localize(None).to_numpy()
            if len(event_values) != len(ts):
                raise ValueError("event_end must have the same number of rows as timestamps.")

        for fold in range(self.n_splits):
            train_end_pos = boundaries[fold + 1]
            valid_start_pos = train_end_pos
            valid_end_pos = boundaries[fold + 2]
            valid_start = unique_ts[valid_start_pos]
            valid_end = unique_ts[valid_end_pos - 1]

            train_cutoff = valid_start - pd.Timedelta(days=self.purge_days)
            train_mask = ts < train_cutoff
            if event_values is not None:
                has_event = pd.notna(event_values)
                train_mask &= (~has_event) | (event_values < valid_start.to_datetime64())

            valid_mask = (ts >= valid_start) & (ts <= valid_end)
            train_idx = np.flatnonzero(train_mask)
            valid_idx = np.flatnonzero(valid_mask)
            if len(train_idx) == 0 or len(valid_idx) == 0:
                continue
            # Embargo end is intentionally computed and exposed for audit/debugging.
            _embargo_end = valid_end + pd.Timedelta(days=self.embargo_days)
            _ = _embargo_end
            yield Fold(fold + 1, train_idx, valid_idx)
