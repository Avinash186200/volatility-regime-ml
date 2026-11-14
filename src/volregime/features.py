"""Causal features and the forecast target.

Timing convention: a forecast is made at the close of day ``t``. Features on row ``t`` use
data up to and including day ``t`` (all rolling windows are trailing). The target on row
``t`` is the average realised variance over days ``t+1 .. t+h``, so it is strictly in the
future of every feature on that row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EWMA_LAMBDA = 0.94  # RiskMetrics value, fixed in advance and never tuned
FLOOR = 1e-8

#: Columns a model may use. ``true_regime`` and the target are deliberately absent.
FEATURES = [
    "ret",
    "rv_d",
    "rv_w",
    "rv_m",
    "ewma",
    "neg_ret",
    "log_rv_d",
    "log_rv_w",
    "log_rv_m",
    "log_ewma",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build trailing HAR components, EWMA and a leverage term from ``ret`` and ``rv``.

    Args:
        df: frame with ``ret`` and ``rv`` (annualised daily realised variance).

    Returns:
        Frame with ``FEATURES`` columns, same index. Early rows are NaN until the 22-day
        window is full.
    """
    rv = df["rv"].clip(lower=FLOOR)
    out = pd.DataFrame(index=df.index)
    out["ret"] = df["ret"]
    out["rv_d"] = rv
    out["rv_w"] = rv.rolling(5).mean()
    out["rv_m"] = rv.rolling(22).mean()
    out["ewma"] = rv.ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean()
    out["neg_ret"] = np.minimum(df["ret"], 0.0)
    for col in ["rv_d", "rv_w", "rv_m", "ewma"]:
        out[f"log_{col}"] = np.log(out[col])
    out.loc[out.index[:21], :] = np.nan  # EWMA is defined earlier, but keep one warm-up
    return out[FEATURES]


def make_target(rv: pd.Series, h: int) -> pd.Series:
    """Average realised variance over the next ``h`` days, aligned to the forecast day.

    ``y[t] = mean(rv[t+1], ..., rv[t+h])``. The last ``h`` rows are NaN.
    """
    if h < 1:
        raise ValueError("h must be >= 1")
    return rv.rolling(h).mean().shift(-h).rename(f"y_h{h}")
