import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volregime.data import from_ohlc, make_synthetic_market  # noqa: E402
from volregime.features import FEATURES, build_features, make_target  # noqa: E402
from volregime.metrics import classifier_scores, diebold_mariano, qlike, qlike_series  # noqa: E402
from volregime.models import GARCH11, MODEL_CLASSES, garch_filter  # noqa: E402
from volregime.walkforward import make_folds, run_walk_forward  # noqa: E402

FAST = dict(train_window=600, test_block=126)


@pytest.fixture(scope="module")
def market():
    return make_synthetic_market(n_days=1500, seed=3)


@pytest.fixture(scope="module")
def noise():
    return make_synthetic_market(
        n_days=1500, seed=4, regimes=False, leverage=0.0, vol_of_vol=0.0, jump_prob=0.0
    )


def test_qlike_zero_at_perfect_forecast_and_positive_otherwise():
    y = np.array([0.01, 0.04, 0.09])
    assert qlike(y, y) == 0.0
    assert (qlike_series(y, y * 1.3) > 0).all()
    assert (qlike_series(y, y * 0.7) > 0).all()


def test_dm_sign():
    rng = np.random.default_rng(0)
    a = rng.random(500)
    t, p = diebold_mariano(a, a + 0.1, lag=1)
    assert t < 0 and p < 0.01


def test_target_is_strictly_future_average(market):
    rv = market["rv"]
    y = make_target(rv, 5)
    t = 100
    assert y.iloc[t] == pytest.approx(rv.iloc[t + 1 : t + 6].mean())
    assert y.iloc[-5:].isna().all()


def test_true_regime_is_never_a_feature(market):
    assert "true_regime" not in FEATURES
    assert set(build_features(market).columns) == set(FEATURES)


def test_features_do_not_change_when_the_future_changes(market):
    t = 700
    altered = market.copy()
    altered.iloc[t + 1 :, altered.columns.get_loc("rv")] *= 5.0
    altered.iloc[t + 1 :, altered.columns.get_loc("ret")] *= -3.0
    a, b = build_features(market), build_features(altered)
    pd.testing.assert_frame_equal(a.iloc[: t + 1], b.iloc[: t + 1])


@pytest.mark.parametrize("name", list(MODEL_CLASSES))
def test_every_forecast_uses_only_past_rows(market, name):
    """Fit on rows 30..529, then perturb rows after 800: forecasts up to row 800 must not move."""
    X = build_features(market)
    y = make_target(market["rv"], 5)
    Xtr, ytr = X.iloc[30:530], y.iloc[30:530]
    model = MODEL_CLASSES[name](h=5).fit(Xtr, ytr)
    ctx = X.iloc[30:1000]
    base = model.predict(ctx)

    altered = market.copy()
    altered.iloc[801:, altered.columns.get_loc("rv")] *= 4.0
    altered.iloc[801:, altered.columns.get_loc("ret")] *= -2.0
    pert = model.predict(build_features(altered).iloc[30:1000])
    cut = 801 - 30
    np.testing.assert_allclose(base[:cut], pert[:cut])
    if name != "hist_mean":  # the benchmark ignores current data by design
        assert not np.allclose(base[cut:], pert[cut:])


@pytest.mark.parametrize("h", [1, 5, 22])
def test_training_labels_end_before_test_block(h):
    valid = np.arange(25, 1400 - h)
    for fold in make_folds(valid, h, train_window=500, test_block=100):
        last_label_day = fold.train.max() + h  # target of row t spans t+1..t+h
        assert last_label_day < fold.test.min()
        assert fold.train.max() < fold.test.min()


def test_forecasts_identical_when_later_data_is_removed(market):
    """Strong leakage check: deleting everything after a fold cannot change that fold."""
    full = run_walk_forward(market, h=5, **FAST)
    short = run_walk_forward(market.iloc[:1250], h=5, **FAST)
    common = short.index[short["fold"] < short["fold"].max()]  # drop the truncated last fold
    cols = ["y", "ewma", "garch", "har", "log_har", "regime_har", "ens_equal", "ens_invloss"]
    pd.testing.assert_frame_equal(full.loc[common, cols], short.loc[common, cols])


def test_inverse_loss_weights_use_earlier_folds_only(market):
    res = run_walk_forward(market, h=1, **FAST)
    first = res[res["fold"] == 0]
    np.testing.assert_allclose(first["ens_invloss"], first["ens_equal"])


def test_garch_recovers_known_parameters():
    rng = np.random.default_rng(11)
    n, omega, alpha, beta = 6000, 0.02, 0.08, 0.90
    r = np.empty(n)
    s = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s) * rng.standard_normal()
        s = omega + alpha * r[t] ** 2 + beta * s
    X = pd.DataFrame({"ret": r / 100.0})
    a_hat, b_hat = GARCH11().fit(X).params[1:]
    assert abs(a_hat - alpha) < 0.03 and abs(b_hat - beta) < 0.04


def test_garch_filter_matches_loop():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(50)
    out = garch_filter(r, 0.1, 0.1, 0.8, 1.0)
    s, ref = 1.0, []
    for x in r:
        s = 0.1 + 0.1 * x**2 + 0.8 * s
        ref.append(s)
    np.testing.assert_allclose(out, ref)


def test_models_beat_the_historical_mean_when_there_is_signal(market):
    res = run_walk_forward(market, h=1, **FAST)
    base = qlike(res["y"], res["hist_mean"])
    for m in ["ewma", "garch", "har", "regime_har", "ens_equal"]:
        assert qlike(res["y"], res[m]) < 0.6 * base, m


def test_noise_shows_no_edge(noise):
    """Constant variance: no forecaster beats the historical mean, and the classifier is a coin."""
    res = run_walk_forward(noise, h=5, **FAST)
    base = qlike(res["y"], res["hist_mean"])
    best = min(qlike(res["y"], res[m]) for m in ["ewma", "garch", "har", "log_har", "regime_har"])
    assert best > 0.97 * base
    auc = classifier_scores(res["z_high"], res["p_high"])["auc"]
    assert abs(auc - 0.5) < 0.08


def test_ohlc_loader_columns():
    idx = pd.bdate_range("2020-01-01", periods=30, name="date")
    rng = np.random.default_rng(2)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 30)))
    ohlc = pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "close": close}, index=idx
    )
    out = from_ohlc(ohlc)
    assert list(out.columns) == ["close", "ret", "rv"]
    assert (out["rv"] > 0).all() and len(out) == 29
