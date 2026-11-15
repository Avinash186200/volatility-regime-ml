"""
Evaluation metrics and walk-forward splitting. Start here, extend as the
project needs -- see docs/project_brief.md for the fuller list (QLIKE,
IC/ICIR, Diebold-Mariano) worth adding as you go.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def mae_log(y_true: pd.Series, y_pred: pd.Series) -> float:
    """MAE on log variance -- the standard, proxy-robust way to score a variance
    forecast. Raw-variance MAE is dominated by a handful of spikes; logs behave."""
    aligned = pd.concat([y_true.rename("true"), y_pred.rename("pred")], axis=1).dropna()
    return float(np.mean(np.abs(np.log(aligned["true"]) - np.log(aligned["pred"]))))


def walk_forward_splits(
    index: pd.DatetimeIndex, train_size: int, test_size: int, step: int
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Expanding-window walk-forward splits. No shuffling, no random k-fold --
    this is time series, and lookahead is the default failure mode if you treat
    it like one."""
    start = train_size
    while start + test_size <= len(index):
        train_idx = index[:start]
        test_idx = index[start:start + test_size]
        yield train_idx, test_idx
        start += step
