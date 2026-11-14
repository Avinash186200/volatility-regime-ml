"""The forecasters. All share one interface so the comparison stays fair.

``fit(X_train, y_train)`` sees only training rows (already purged by the walk-forward).
``predict(X_ctx)`` receives a contiguous block of feature rows that starts at the first
training row and ends at the last test row, and returns a variance forecast for every row.
A forecast on row ``t`` may use rows ``<= t`` only. Row-wise models use just row ``t``; the
GARCH filter is a forward recursion. The walk-forward keeps only the test rows.

All forecasts are annualised variances of the average daily variance over ``t+1 .. t+h``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ANNUAL
from .features import FLOOR

LOG_HAR_COLS = ["log_rv_d", "log_rv_w", "log_rv_m", "neg_ret"]
CLF_COLS = ["log_rv_d", "log_rv_w", "log_rv_m", "log_ewma", "neg_ret", "ret"]


class HistoricalMean:
    """Benchmark: the training-window mean of the target. Uses no current information."""

    name = "hist_mean"

    def __init__(self, h: int = 1) -> None:
        self.h = h
        self.level = np.nan

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalMean":
        self.level = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.level)


class EWMA:
    """RiskMetrics EWMA of realised variance (lambda 0.94, fixed). Flat across horizons."""

    name = "ewma"

    def __init__(self, h: int = 1) -> None:
        self.h = h

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EWMA":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["ewma"].to_numpy()


def garch_filter(r: np.ndarray, omega: float, alpha: float, beta: float, s0: float) -> np.ndarray:
    """One-step-ahead GARCH(1,1) variances.

    ``out[k]`` is the variance for day ``k+1`` given returns up to day ``k``:
    ``out[k] = omega + alpha * r[k]**2 + beta * out[k-1]``, with ``out[-1] = s0``.
    """
    u = omega + alpha * r**2
    out, _ = lfilter([1.0], [1.0, -beta], u, zi=[beta * s0])
    return out


@dataclass
class GARCH11:
    """GARCH(1,1) with normal errors, fitted by maximum likelihood on training returns only.

    Returns are scaled to percent for the optimiser. The filter is initialised with the
    training-sample variance at the first training row, then run forward, so it is causal.
    """

    h: int = 1
    name: str = "garch"
    params: tuple[float, float, float] = field(default=(np.nan, np.nan, np.nan))
    s0: float = np.nan

    @staticmethod
    def _neg_loglik(theta: np.ndarray, r: np.ndarray, s0: float) -> float:
        omega, alpha, beta = theta
        if alpha + beta >= 0.999:
            return 1e10
        s_next = garch_filter(r, omega, alpha, beta, s0)
        s = np.concatenate([[s0], s_next[:-1]])
        s = np.maximum(s, 1e-12)
        return 0.5 * float(np.sum(np.log(s) + r**2 / s))

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "GARCH11":
        r = 100.0 * X["ret"].to_numpy()
        var = float(np.var(r))
        self.s0 = var
        x0 = np.array([0.05 * var, 0.08, 0.90])
        res = minimize(
            self._neg_loglik,
            x0,
            args=(r, var),
            method="L-BFGS-B",
            bounds=[(1e-6 * var, 10 * var), (1e-6, 0.5), (0.0, 0.998)],
        )
        self.params = tuple(float(v) for v in res.x)  # type: ignore[assignment]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        omega, alpha, beta = self.params
        r = 100.0 * X["ret"].to_numpy()
        s1 = garch_filter(r, omega, alpha, beta, self.s0)  # variance for t+1, known at t
        persist = alpha + beta
        vbar = omega / (1 - persist)
        # average of the k-step forecasts, k = 1..h
        k = np.arange(self.h)
        avg_decay = float(np.mean(persist**k))
        s_avg = vbar + avg_decay * (s1 - vbar)
        return s_avg / 1e4 * ANNUAL


class HAR:
    """HAR-RV (Corsi, 2009): OLS of the target on daily, weekly and monthly realised variance."""

    name = "har"
    cols = ["rv_d", "rv_w", "rv_m"]

    def __init__(self, h: int = 1) -> None:
        self.h = h
        self.model = LinearRegression()
        self.floor = FLOOR

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HAR":
        self.model.fit(X[self.cols], y)
        self.floor = max(float(y.min()) * 0.5, FLOOR)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.model.predict(X[self.cols]), self.floor)


class LogHAR:
    """HAR on log variance with a leverage term. Used as an ablation for the regime model."""

    name = "log_har"

    def __init__(self, h: int = 1) -> None:
        self.h = h
        self.model = LinearRegression()
        self.s2 = 0.0

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[LOG_HAR_COLS]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogHAR":
        Z = self._design(X)
        ly = np.log(np.maximum(y.to_numpy(), FLOOR))
        self.model.fit(Z, ly)
        self.s2 = float(np.var(ly - self.model.predict(Z)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # exp(E[log y] + s2/2) is the mean of a log-normal, removing the Jensen bias
        return np.exp(self.model.predict(self._design(X)) + 0.5 * self.s2)


class RegimeClassifier:
    """Logistic regression for P(next-h-day variance is in the high regime).

    "High" means above the 75th percentile of the *training* targets. The threshold is set
    from training rows only and stored, so the test period never shapes it.
    """

    def __init__(self, quantile: float = 0.75) -> None:
        self.quantile = quantile
        self.threshold = np.nan
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        self.constant: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, threshold: float | None = None) -> "RegimeClassifier":
        self.threshold = float(np.quantile(y, self.quantile)) if threshold is None else threshold
        z = (y.to_numpy() > self.threshold).astype(int)
        if z.min() == z.max():
            self.constant = float(z.mean())
        else:
            self.constant = None
            self.model.fit(X[CLF_COLS], z)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(X), self.constant)
        return self.model.predict_proba(X[CLF_COLS])[:, 1]


class RegimeHAR(LogHAR):
    """Log-HAR whose intercept and daily persistence shift with the regime probability.

    Design: ``[log_rv_d, log_rv_w, log_rv_m, neg_ret, p, p * log_rv_d]`` where ``p`` is the
    classifier's probability of the high regime. On training rows ``p`` is cross-fitted
    (contiguous blocks, purged by ``h`` rows each side), so the second stage never sees a
    probability the classifier produced for a row it was trained on.
    """

    name = "regime_har"

    def __init__(self, h: int = 1, n_blocks: int = 5) -> None:
        super().__init__(h)
        self.n_blocks = n_blocks
        self.clf = RegimeClassifier()
        self.last_proba: np.ndarray | None = None

    def _design_p(self, X: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
        Z = X[LOG_HAR_COLS].copy()
        Z["p_high"] = p
        Z["p_x_rv_d"] = p * X["log_rv_d"].to_numpy()
        return Z

    def _cross_fit(self, X: pd.DataFrame, y: pd.Series, threshold: float) -> np.ndarray:
        n = len(X)
        p = np.empty(n)
        edges = np.linspace(0, n, self.n_blocks + 1).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            keep = np.ones(n, dtype=bool)
            keep[max(0, a - self.h) : min(n, b + self.h)] = False
            clf = RegimeClassifier().fit(X[keep], y[keep], threshold=threshold)
            p[a:b] = clf.predict_proba(X.iloc[a:b])
        return p

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegimeHAR":
        self.clf.fit(X, y)
        p_oof = self._cross_fit(X, y, self.clf.threshold)
        Z = self._design_p(X, p_oof)
        ly = np.log(np.maximum(y.to_numpy(), FLOOR))
        self.model.fit(Z, ly)
        self.s2 = float(np.var(ly - self.model.predict(Z)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.clf.predict_proba(X)
        self.last_proba = p
        return np.exp(self.model.predict(self._design_p(X, p)) + 0.5 * self.s2)


MODEL_CLASSES = {
    "hist_mean": HistoricalMean,
    "ewma": EWMA,
    "garch": GARCH11,
    "har": HAR,
    "log_har": LogHAR,
    "regime_har": RegimeHAR,
}

#: Members of the ensemble: the four forecasters this project is about.
ENSEMBLE_MEMBERS = ["ewma", "garch", "har", "regime_har"]
