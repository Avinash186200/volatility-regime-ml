"""Rolling-origin, purged walk-forward evaluation shared by every forecaster."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURES, build_features, make_target
from .metrics import qlike_series
from .models import ENSEMBLE_MEMBERS, MODEL_CLASSES


@dataclass(frozen=True)
class Fold:
    """Row positions (into the full frame) of one training window and the test block after it."""

    k: int
    train: np.ndarray
    test: np.ndarray


def make_folds(
    valid: np.ndarray, h: int, train_window: int, test_block: int, expanding: bool = False
) -> list[Fold]:
    """Build rolling-origin folds over the rows where features and target are both defined.

    For a test block starting at row ``e``, training rows are those whose target window
    ``t+1 .. t+h`` ends before ``e``: rows ``t <= e - 1 - h``. That purge of ``h`` rows is
    what stops a training label from overlapping the test period.

    Args:
        valid: sorted row positions with complete features and target.
        h: forecast horizon in days.
        train_window: number of rows in each training window (before the purge).
        test_block: number of rows per test block. Models are refitted every block.
        expanding: grow the training window from the first valid row instead of rolling it.
    """
    folds: list[Fold] = []
    first = int(valid[0])
    e = first + train_window
    last = int(valid[-1])
    k = 0
    while e <= last:
        lo = first if expanding else e - train_window
        train = valid[(valid >= lo) & (valid <= e - 1 - h)]
        test = valid[(valid >= e) & (valid < e + test_block)]
        if len(test) and len(train) > 50:
            folds.append(Fold(k, train, test))
            k += 1
        e += test_block
    return folds


def run_walk_forward(
    df: pd.DataFrame,
    h: int,
    train_window: int = 1000,
    test_block: int = 63,
    expanding: bool = False,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Fit every model on each training window, forecast the following block, collect results.

    Two ensembles are added: an equal-weight average of ``ENSEMBLE_MEMBERS`` and an
    inverse-QLIKE weighted average whose weights come only from *earlier* folds' test losses
    (equal weights in the first fold). Neither looks at the block it is forecasting.

    Returns:
        One row per out-of-sample forecast with ``fold, y, z_high, p_high`` and a column per
        model, indexed by forecast date.
    """
    models = models or list(MODEL_CLASSES)
    X = build_features(df)
    y = make_target(df["rv"], h)
    ok = X[FEATURES].notna().all(axis=1).to_numpy() & y.notna().to_numpy()
    valid = np.flatnonzero(ok)
    folds = make_folds(valid, h, train_window, test_block, expanding)
    if not folds:
        raise ValueError("not enough data for one fold; lower train_window")

    members = [m for m in ENSEMBLE_MEMBERS if m in models]
    past_loss: dict[str, list[float]] = {m: [] for m in members}
    rows = []
    for fold in folds:
        Xtr, ytr = X.iloc[fold.train], y.iloc[fold.train]
        ctx = X.iloc[fold.train[0] : fold.test[-1] + 1]
        test_pos = fold.test - fold.train[0]
        out = pd.DataFrame(index=X.index[fold.test])
        out["fold"] = fold.k
        out["y"] = y.iloc[fold.test].to_numpy()
        for name in models:
            model = MODEL_CLASSES[name](h=h).fit(Xtr, ytr)
            out[name] = model.predict(ctx)[test_pos]
            if name == "regime_har":
                out["p_high"] = model.last_proba[test_pos]
                out["z_high"] = (out["y"] > model.clf.threshold).astype(int)
        if members:
            F = out[members].to_numpy()
            out["ens_equal"] = F.mean(axis=1)
            if all(past_loss[m] for m in members):
                inv = np.array([1.0 / np.mean(past_loss[m]) for m in members])
                w = inv / inv.sum()
            else:
                w = np.full(len(members), 1.0 / len(members))
            out["ens_invloss"] = F @ w
            for m in members:
                past_loss[m].append(float(np.mean(qlike_series(out["y"], out[m]))))
        rows.append(out)
    return pd.concat(rows)
