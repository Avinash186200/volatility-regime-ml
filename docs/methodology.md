# Methodology and bias audit

## Background

There is no single correct way to forecast volatility. The project brief I wrote asked for a small model zoo that spans the spectrum (smoothing, econometric, regression, a classifier), a fair head-to-head comparison against HAR-RV, an explanation of *why* models differ, a combined forecast with a verdict on whether combining helps, and a rough economic check. This repo is my implementation of that brief.

## Data contract

| Column | Meaning |
|---|---|
| `close` | Daily close |
| `ret` | Close-to-close log return of that day |
| `rv` | Realised variance of that day, annualised (x252) |
| `true_regime` | Synthetic data only. Diagnostics only, never a model input (tested) |

Sources:

- **Synthetic** (`make_synthetic_market`): a two-state Markov regime (calm 12%, stress 35% annualised vol, staying probabilities 0.99 and 0.97), log-variance AR(1) with persistence 0.97, a leverage term on negative standardised returns, and jumps with probability 1% a day. Each day has 78 intraday increments, so `rv` is a sum of squared intraday returns.
- **Public daily OHLC** (`--ticker`, downloaded at runtime with `yfinance`, never saved into the repo): `rv` is the Garman-Klass estimator. It ignores the overnight gap, so it sits below close-to-close variance. The HAR family absorbs that level difference through its intercept and slopes. GARCH and EWMA do not, which tilts the comparison on real data.
- **User CSV** (`--csv`): `date` plus `open, high, low, close`, or `date` plus `close, rv`.

## Timing

A forecast is made at the close of day `t`.

- Features on row `t` use days `<= t` only. All rolling windows are trailing. A test perturbs every day after `t` and checks that no feature up to `t` changes.
- The target on row `t` is `mean(rv[t+1 .. t+h])`.
- Every model's `predict` is checked the same way: perturb the data after row `j`, and the forecasts up to `j` must not move.

## Models

- **Historical mean**: the training-window mean of the target.
- **EWMA**: `ewma[t] = 0.94 * ewma[t-1] + 0.06 * rv[t]`. Lambda is the RiskMetrics value, fixed in advance. Its forecast is flat across horizons.
- **GARCH(1,1)**: normal-likelihood MLE on the training window's daily returns (in percent, L-BFGS-B, `alpha + beta < 0.999`). The filter starts at the training-sample variance on the first training row and runs forward. The `h`-day forecast is the average of the `k`-step forecasts, `vbar + (alpha+beta)^(k-1) * (s[t+1] - vbar)`. A test recovers known parameters from a simulated GARCH path.
- **HAR-RV**: OLS of the target on `rv_d`, `rv_w` (5-day mean) and `rv_m` (22-day mean). Forecasts are floored at half the smallest training target.
- **Log-HAR**: OLS of `log y` on the logged HAR terms and the day's negative return, then `exp(fit + s2/2)` to undo the log-normal bias.
- **Regime classifier**: a standardised logistic regression for `P(y[t] > q75)`, where `q75` is the 75th percentile of the **training** targets. Inputs: the logged HAR terms, the logged EWMA, the negative return and the return.
- **Regime HAR**: Log-HAR plus `p_high` and `p_high * log rv_d`, so the intercept and daily persistence can shift with the regime. On training rows `p_high` is cross-fitted (5 contiguous blocks, with `h` rows purged on each side), so the second stage never sees a probability for a row the classifier was trained on.
- **Ensembles**: an equal-weight average of EWMA, GARCH, HAR and Regime HAR. The inverse-loss version weights each member by `1 / mean QLIKE` over all earlier folds' test blocks, with equal weights in the first fold. A test checks that the first fold's two ensembles are identical.

## Validation design

- A rolling origin with a 1,000-day training window, refitted every 63 days (`--expanding` grows the window instead). Every model sees exactly the same rows in every fold.
- **Purge**: a training row `t` is used only if `t + h` comes before the first test day. A test checks this for h = 1, 5 and 22.
- **Truncation test**: running the whole walk-forward on the data cut at day 1,250 must give bit-identical forecasts for every fold that ends before the cut. Any use of later data, in features, fitting, thresholds or ensemble weights, would break it.
- **Losses**: QLIKE `y/f - log(y/f) - 1` and MSE. Both rank variance forecasts consistently under a noisy but unbiased proxy (Patton, 2011). QLIKE is the headline because it is not dominated by a few spikes.
- **Diebold-Mariano** test of each model against HAR on QLIKE, with a Newey-West variance using `h` lags, because overlapping targets make the losses autocorrelated.
- **Classifier**: AUC and Brier, against the score you get from simply ranking the current EWMA level.
- **Economic sanity check** (h = 1): scale the next day's return by `0.10 / sqrt(forecast)` and measure the realised vol and the spread of its 63-day rolling value around the target. There are no costs, no leverage cap and no implied-vol comparison, so this is not a trading result.

## Bias audit of the original starter code

| # | Issue in the original | Effect | Fix here |
|---|---|---|---|
| 1 | A single split at a fixed date. The target was `rv.shift(-1)`, so the last training row's label was the first test day's variance. At horizon `h` the overlap is `h` days | Training labels from inside the test period | A purge of `h` rows before every test block |
| 2 | The naive "yesterday = tomorrow" baseline was evaluated on `rv[t]` at row `t`, while HAR was evaluated on `rv[t+1]` at row `t`, then the two were filtered to the same index | The two models were scored on target days one day apart, so the reported ~22% improvement was not like-for-like | One target series and one index for every model |
| 3 | GARCH and HAR were estimated once, on one training period, and scored on one test window | The result depended on one window | Rolling-origin refits (32 folds by default) |
| 4 | RMSE on annualised squared daily returns | A very noisy proxy, and RMSE in levels is driven by a few spikes | QLIKE and MSE, a DM test, intraday `rv` on synthetic data and Garman-Klass on OHLC |
| 5 | HAR in levels, with no floor | Negative variance forecasts are possible | Floor at half the smallest training target |
| 6 | No rule on where ensemble weights, thresholds or smoothing constants come from | Easy to tune on the test period without noticing | Ensemble weights from earlier folds, the threshold from the training quantile, and EWMA lambda fixed |
| 7 | (New code) a regime probability used inside a second-stage regression | An in-sample probability looks better than it is | Cross-fitting with a purge |
| 8 | Only next-day results | Horizon cherry-picking | 1, 5 and 22 days always reported together |

The GARCH code in the original notebook was causal: parameters were fitted on training returns, and the filter was then run with those fixed parameters. It is kept in spirit, and refitted per fold.

## Findings (synthetic data)

1. HAR beats EWMA and GARCH at every horizon. GARCH's handicap is information: it sees one squared return per day, while the RV-based models see 78 intraday returns.
2. Regime HAR beats HAR, but Log-HAR without the regime probability does at least as well. The improvement comes from the log transform and the leverage term, a feature-construction gain, not from modelling regimes. The brief asked for exactly this kind of check.
3. The regime classifier is good at its own job (AUC 0.98 at h = 1, against 0.92 for the current-level ranking). Its information is already in the log RV terms as far as variance forecasting is concerned.
4. Neither ensemble beats the best single model. Equal weighting is hurt by weak members. Inverse-loss weighting from past folds helps, and both ensembles beat HAR at 22 days.
5. On constant-variance noise, nothing beats the historical mean and the classifier's AUC is about 0.5 (a CI test).

## Not done, and why

- **Implied volatility.** The brief's decomposition into realised and implied, and the "is it already priced in" question, needs option-implied data. The synthetic market has none, and I did not want to rely on a downloaded index whose licence I would have to check per user. This is the main open item.
- **Jump/continuous split** (bipower variation): it needs intraday returns, so it only applies to the synthetic data. Left out to keep the repo focused.
- **Gradient boosting and sequence models**: the brief allowed them, but I have not implemented them here. They add tuning freedom, which is a selection-bias risk, so I kept the zoo small. The `fit` / `predict` interface makes them a drop-in addition.
- **The exploration notebook and the contributor workflow docs** from the original project were not ported.

## References

- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*.
- Patton, A. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*.
- Andersen, T., Bollerslev, T., Diebold, F. and Labys, P. (2003). Modeling and forecasting realized volatility. *Econometrica*.
- Garman, M. and Klass, M. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*.
- Diebold, F. and Mariano, R. (1995). Comparing predictive accuracy. *Journal of Business and Economic Statistics*.
