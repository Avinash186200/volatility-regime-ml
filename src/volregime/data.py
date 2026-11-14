"""Market data: a synthetic regime-switching generator and an optional public-data loader.

Every frame this module returns has the same columns, indexed by business date:

- ``close``: price level
- ``ret``: daily close-to-close log return
- ``rv``: daily realised variance, annualised (x252)

The synthetic generator also returns ``true_regime`` (0 calm, 1 stress). It exists for
diagnostics only and is never used as a model input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ANNUAL = 252
INTRADAY_BARS = 78  # 5-minute bars in a 6.5-hour session


def make_synthetic_market(
    n_days: int = 3000,
    seed: int = 7,
    regimes: bool = True,
    leverage: float = 0.08,
    vol_of_vol: float = 0.12,
    jump_prob: float = 0.01,
    start: str = "2012-01-02",
) -> pd.DataFrame:
    """Simulate daily returns and intraday realised variance with volatility regimes.

    The latent daily log-variance follows an AR(1) whose mean switches between a calm and a
    stress regime (a two-state Markov chain). Negative returns push variance up (leverage).
    Each day is built from ``INTRADAY_BARS`` normal increments, so ``rv`` is a genuine
    realised variance (sum of squared intraday returns) and ``ret`` is their sum. Rare jumps
    add to both.

    With ``regimes=False, leverage=0, vol_of_vol=0, jump_prob=0`` the variance is constant
    and returns are i.i.d., which is the pure-noise case used in the tests.

    Args:
        n_days: number of business days.
        seed: RNG seed.
        regimes: switch between calm (12% vol) and stress (35% vol) regimes if True.
        leverage: response of log-variance to a negative standardised return.
        vol_of_vol: standard deviation of the log-variance innovation.
        jump_prob: daily probability of a price jump.
        start: first business date.

    Returns:
        DataFrame with ``close, ret, rv, true_regime``.
    """
    rng = np.random.default_rng(seed)
    mu = np.log(np.array([0.12, 0.35]) ** 2 / ANNUAL)
    p_stay = np.array([0.99, 0.97])
    phi = 0.97

    state = np.zeros(n_days, dtype=int)
    h = np.empty(n_days)
    ret = np.empty(n_days)
    rv = np.empty(n_days)
    h_prev = mu[0]
    z_prev = 0.0
    for t in range(n_days):
        if regimes and t > 0:
            s_prev = state[t - 1]
            state[t] = s_prev if rng.random() < p_stay[s_prev] else 1 - s_prev
        m = mu[state[t]]
        lev = leverage * (max(-z_prev, 0.0) - 0.4)  # 0.4 ~ E[max(-z, 0)] keeps the mean stable
        h[t] = (1 - phi) * m + phi * h_prev + lev + vol_of_vol * rng.standard_normal()
        sigma2 = np.exp(h[t])
        incr = rng.standard_normal(INTRADAY_BARS) * np.sqrt(sigma2 / INTRADAY_BARS)
        jump = rng.normal(0.0, 3.0 * np.sqrt(sigma2)) if rng.random() < jump_prob else 0.0
        ret[t] = incr.sum() + jump
        rv[t] = (incr**2).sum() + jump**2
        z_prev = ret[t] / np.sqrt(sigma2)
        h_prev = h[t]

    idx = pd.bdate_range(start=start, periods=n_days, name="date")
    return pd.DataFrame(
        {
            "close": 100.0 * np.exp(np.cumsum(ret)),
            "ret": ret,
            "rv": rv * ANNUAL,
            "true_regime": state,
        },
        index=idx,
    )


def garman_klass_variance(ohlc: pd.DataFrame) -> pd.Series:
    """Daily Garman-Klass range variance (annualised) from open/high/low/close columns.

    It ignores the overnight gap, so its level is below close-to-close variance. The models
    that regress on it (HAR family) absorb that level difference; GARCH, which works on
    close-to-close returns, does not, which is noted in the methodology.
    """
    hl = np.log(ohlc["high"] / ohlc["low"])
    co = np.log(ohlc["close"] / ohlc["open"])
    gk = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    return gk.clip(lower=1e-10) * ANNUAL


def from_ohlc(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Convert a daily OHLC frame (lower-case columns, date index) into the standard frame."""
    ohlc = ohlc.sort_index()
    out = pd.DataFrame(index=ohlc.index)
    out["close"] = ohlc["close"]
    out["ret"] = np.log(ohlc["close"] / ohlc["close"].shift(1))
    out["rv"] = garman_klass_variance(ohlc)
    out.index.name = "date"
    return out.dropna()


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a user CSV with ``date`` plus either ``open, high, low, close`` or ``close, rv``."""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.columns = [c.lower() for c in df.columns]
    if {"open", "high", "low", "close"} <= set(df.columns):
        return from_ohlc(df)
    if {"close", "rv"} <= set(df.columns):
        out = df[["close", "rv"]].copy()
        out["ret"] = np.log(out["close"] / out["close"].shift(1))
        return out.dropna()[["close", "ret", "rv"]]
    raise ValueError("CSV needs date + open/high/low/close, or date + close/rv")


def load_public(ticker: str, start: str = "2005-01-01") -> pd.DataFrame:  # pragma: no cover - network
    """Download free daily OHLC data at runtime with ``yfinance`` (optional dependency).

    Nothing downloaded is written to the repo.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit("pip install yfinance to use --ticker") from exc
    raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    raw.index.name = "date"
    return from_ohlc(raw[["open", "high", "low", "close"]])
