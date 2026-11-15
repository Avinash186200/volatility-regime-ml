# Project Brief: Forecasting Volatility

## Motivation

There's no single "correct" way to forecast volatility — different models capture different things,
and reasonable people decompose the problem differently. This project explores that space directly:
build several genuinely different approaches, understand *why* they disagree (not just which one
scores best), and figure out how to combine them into one usable signal.

Two things worth knowing going in:

- Volatility forecasting has a very strong, very simple benchmark — a basic statistical model using
  recent history (HAR-RV) — that is notoriously hard to beat. "My fancy model doesn't beat the
  simple one" is a legitimate, useful finding here, not a failure.
- Forecast accuracy and trading value aren't the same thing. A model can look accurate and still not
  be worth trading once you account for how much of the "obvious" information is already priced in
  by the market.

This is your internship project, and it's real work: the models and signal you build here are meant
to be used in our own experiments and in the products we build on top of them, not filed away once
the internship ends.

## Phase 1 — Volatility decomposition

Before building models, spend real time on how you're going to decompose volatility itself. There's
no single right way to split it — that's part of what makes this a research question rather than an
engineering task. A few angles worth trying:

- **Short-term vs. long-term components** — recent volatility vs. a slower-moving baseline.
- **Realized vs. implied** — what actually happened vs. what the options market is currently
  pricing in.
- **Continuous vs. jump-driven** — steady drift vs. sudden, discontinuous moves. These two pieces
  tend to have very different forecastability, and lumping them together can hide that.

Pick at least one decomposition and carry it through the rest of the project — it will shape what
your features and models actually need to predict.

## Phase 2 — Baseline

Build the HAR-RV baseline first (`src/models/baseline_har.py`). Regress next-period realized
variance on the daily, weekly, and monthly rolling averages of past realized variance. Three
coefficients, ordinary least squares. This is the floor — it has resisted a lot of attempts to
displace it, and every other model in this project needs to be measured against it, not against
"nothing."

## Phase 3 — Build out the model zoo

Spanning a spectrum:

- **Simple statistical / econometric models** — HAR variants, a basic GARCH model.
- **Regularized regression** — ridge, LASSO, or elastic net on a wider feature set.
- **One or two ML approaches** — gradient boosting is a reasonable, pragmatic choice; a simple
  sequence model if you want to push further.

Run every model through the *same* evaluation: same data, same train/test splits, same metrics — so
that differences in the results are attributable to the model, not to an accidental advantage in
setup. `src/models/base.py` defines the shared `fit` / `predict` interface every model should
implement.

## Phase 4 — Understand *why*, not just *which*

Don't stop at a leaderboard. For any model that beats HAR-RV, dig into where the advantage actually
comes from:

- Is it capturing asymmetric reactions to bad news vs. good news?
- Is it picking up a regime shift or threshold effect?
- Is it exploiting an interaction between signals that a linear model can't represent unless you
  hand-build it as a feature?

If you can build the interaction explicitly and hand it to a simple model, and the "advantage" of
the fancier model disappears, that's a real, useful finding — it tells you the gain was about
feature construction, not about the model being smarter.

## Phase 5 — Multiple horizons

Test at more than one forecast horizon — next day vs. next few weeks, for instance. A model can be
strong at one horizon and mediocre at another; report both rather than picking whichever one looks
best.

## Phase 6 — Combine

Combine your models into a single forecast. Start with a simple average — it's a famously strong
baseline in forecast combination and is worth beating deliberately, not by default. Try one or two
smarter schemes (inverse-error weighting, a simple regression-based combination) and see whether the
added complexity actually earns its keep.

## Phase 7 — Does it matter economically?

Translate the final combined forecast into a rough economic evaluation. Would it actually have been
worth trading, once you account for the fact that a lot of "obvious" volatility information is
already priced into the market? A model that forecasts well but adds nothing over what the market
already knows is an honest and useful conclusion.

## Deliverables

- A short model zoo with an honest head-to-head comparison against the HAR-RV baseline.
- A clear explanation of *why* any differences between models exist, not just a leaderboard.
- One combined forecast, and a verdict on whether combining actually helped.
- A rough economic sanity check on the final signal.

## Data

There's an internal Polygon-based data pipeline elsewhere in the org (the `Data-Fetch` repo) for
higher-frequency data — ask the team if you need it. `src/data/loaders.py` pulls free public data
(VIX, index/ETF prices via `yfinance`) so you can start immediately.
