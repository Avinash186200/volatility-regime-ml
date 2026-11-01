# Volatility Regime Forecasting — an ML case study

A write-up of an applied ML project: using engineered volatility features and a model ensemble
to forecast market volatility regimes, with a strong focus on **rigorous, leak-free validation**.

This is a **case study**, not a code repo — it describes the problem, the modeling approach, and
(most importantly) the validation process. No proprietary code, data, API keys, or client details
are included. Figures below are rounded/representative of the real results, not exact production
numbers.

---

## The problem

Short-term volatility is hard to forecast directly — it's noisy, and naive models tend to produce
backtests that look great until you check them properly. The brief: build a model that estimates
the probability of a market volatility regime shift, usable as a risk-management signal rather
than a price predictor.

## Feature engineering

The breakthrough came from reframing the target. Instead of predicting the volatility index
directly, the model predicts off the **relationship between implied and realized volatility**:

- **IV/RV ratio** — implied vol (market-priced) vs. realized vol (what actually happened).
  IV > RV → calm regime. IV < RV → stress regime (the market under-priced recent realized moves).
- Realized volatility at multiple windows (5d / 10d / 21d / 63d), with momentum ratios between them
- Volatility term-structure features (short-dated vs. longer-dated implied vol)
- Cross-sectional dispersion (how correlated/uncorrelated the underlying constituents are)

Reframing the target this way — relative mispricing instead of an absolute level — took the
strongest feature's information coefficient from **not statistically significant to a 6x
improvement**, confirmed with a formal significance test (p < 0.001).

## Modeling

An ensemble of diverse architectures, deliberately choosing **architecture and horizon diversity**
over single-model tuning:

- Logistic regression (regime classification) and gradient-boosted trees (HistGradientBoosting,
  LightGBM) for regression, each trained at multiple forecast horizons
- Systematic ablation: each candidate model/feature-set was tested **solo** before being added to
  the ensemble — only kept if it added genuinely uncorrelated signal (tracked via out-of-sample
  correlation between model outputs, not just raw performance)
- Dozens of architecture/feature combinations tested and most **rejected** — the ensemble that
  shipped kept ~4 models out of 10+ tried, specifically the ones that were both performant *and*
  weakly correlated with the rest

## The part that matters most: catching our own mistakes

Two real bugs were caught mid-project, both the kind that quietly inflate a backtest:

1. **Look-ahead bias.** An early version scored very well — until an audit found ~30 of 41
   features used same-day closing prices that would not actually have been available at the time
   the signal needed to fire. Re-running with every feature correctly lagged dropped the headline
   number substantially. Shipped the honest version anyway.
2. **Target contamination.** A separate model used a 3-day-forward target that leaked information
   from inside the window. Switching to a clean 1-day-forward target with a strict expanding
   walk-forward split produced a realistic, defensible result instead of an inflated one.

## Validation methodology

The validation bar, applied to every candidate before it was allowed to ship:

- **Walk-forward, expanding-window** splits only — train strictly on the past, never shuffle time series
- **Permutation testing** — shuffle the labels, re-run, confirm the real result beats the shuffled
  distribution (p < 0.05 required)
- **Bootstrap confidence intervals** on the headline metric, not a single point estimate
- **Deflated Sharpe / multiple-testing awareness** — when dozens of model variants are tried, the
  "best" one is expected to look good by chance alone; the bar accounts for how many variants were
  tested, not just the winner's raw score
- A hard rule: **a result that can't clear permutation + bootstrap + multiple-testing checks does
  not ship**, no matter how good the headline number looks

## The honest outcome

The direct "predict next-day volatility direction" model, after full leak-free validation,
**did not show a reliable edge** — despite promising single-feature signals and earlier backtests
that looked strong before the bias was removed. That finding was reported plainly rather than
reframed to look better.

The validated, shippable result was different in kind: not a price predictor, but a **statistically
significant risk-timing overlay** — using the same features to flag regime shifts and scale
exposure down ahead of them. That version cleared every validation gate (significant under
permutation testing, positive bootstrap confidence interval, stable across sub-periods) and was
productionized with automated daily scoring, data-freshness checks, and fail-safe alerting.

## What this demonstrates

- Feature engineering that measurably moved a model's signal (not just hyperparameter tuning)
- Comfort with an ensemble of model families (linear, boosted trees, multiple architectures) and
  disciplined ablation to decide what actually belongs in it
- A validation methodology (walk-forward, purging, permutation tests, bootstrap CI, multiple-testing
  correction) rigorous enough to catch two real bugs before they reached a client
- The judgement to report a negative result honestly, and to reframe a failed hypothesis into a
  different, validated, shippable product rather than force the original idea to work

---

*Companion project: [AI Engineering from Scratch](https://github.com/Avinash186200/ai-engineering-from-scratch) —
a public course on building production LLM/agent systems.*
