"""Forecast losses, a Diebold-Mariano test, classifier scores and a vol-targeting check."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, roc_auc_score


def qlike_series(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Per-observation QLIKE, ``y/f - log(y/f) - 1``. Zero at a perfect forecast.

    QLIKE and MSE are the two losses that rank variance forecasts consistently even when
    the realised-variance proxy is noisy (Patton, 2011). QLIKE punishes under-prediction
    more and is not dominated by a few spikes the way MSE is.
    """
    ratio = np.asarray(y, dtype=float) / np.asarray(f, dtype=float)
    return ratio - np.log(ratio) - 1.0


def qlike(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.mean(qlike_series(y, f)))


def mse(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(f)) ** 2))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int) -> tuple[float, float]:
    """DM test of equal predictive accuracy with a Newey-West variance.

    Returns ``(t_stat, p_value)``. A negative t-stat means model A has the lower loss.
    ``lag`` should be at least ``h - 1`` because overlapping targets make losses serially
    correlated.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    dc = d - d.mean()
    var = float(dc @ dc) / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * float(dc[k:] @ dc[:-k]) / n
    if var <= 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt(var / n)
    return float(t), float(2 * (1 - norm.cdf(abs(t))))


def classifier_scores(z: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """AUC and Brier score of probabilities ``p`` for binary labels ``z``."""
    z = np.asarray(z).astype(int)
    if z.min() == z.max():
        return {"auc": np.nan, "brier": float(brier_score_loss(z, p))}
    return {"auc": float(roc_auc_score(z, p)), "brier": float(brier_score_loss(z, p))}


def vol_target_check(
    next_ret: np.ndarray, forecast: np.ndarray, target_vol: float = 0.10, window: int = 63
) -> dict[str, float]:
    """Scale tomorrow's return by ``target_vol / forecast_vol`` and see how steady the result is.

    A better one-day variance forecast should give a realised volatility closer to the target
    and a smaller spread of the rolling realised volatility around it. No costs or leverage
    limits are applied; it is a sanity check, not a strategy.
    """
    w = target_vol / np.sqrt(np.asarray(forecast))
    scaled = pd.Series(w * np.asarray(next_ret))
    roll = scaled.rolling(window).std() * np.sqrt(252)
    return {
        "realised_vol": float(scaled.std() * np.sqrt(252)),
        "rolling_vol_sd": float((roll / target_vol).std()),
    }
