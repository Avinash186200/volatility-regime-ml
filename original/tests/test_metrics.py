import numpy as np
import pandas as pd

from src.evaluation.metrics import mae_log, walk_forward_splits


def test_mae_log_zero_for_identical_series():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    y = pd.Series(np.linspace(0.01, 0.05, 10), index=idx)
    assert mae_log(y, y) == 0.0


def test_walk_forward_splits_no_overlap_train_test():
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    splits = list(walk_forward_splits(idx, train_size=20, test_size=5, step=5))
    assert len(splits) > 0
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()
