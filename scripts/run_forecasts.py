"""Run the full forecaster comparison and write the results.

Examples:
    python scripts/run_forecasts.py                      # synthetic data, seed 7
    python scripts/run_forecasts.py --ticker SPY         # free public data (needs yfinance)
    python scripts/run_forecasts.py --csv my_ohlc.csv    # your own daily OHLC
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volregime.data import load_csv, load_public, make_synthetic_market  # noqa: E402
from volregime.metrics import (  # noqa: E402
    classifier_scores,
    diebold_mariano,
    mse,
    qlike,
    qlike_series,
    vol_target_check,
)
from volregime.walkforward import run_walk_forward  # noqa: E402

REPORT = ["hist_mean", "ewma", "garch", "har", "log_har", "regime_har", "ens_equal", "ens_invloss"]


def leaderboard(res: pd.DataFrame, h: int) -> pd.DataFrame:
    """QLIKE, MSE and a DM test against HAR for every model on the same rows."""
    y = res["y"].to_numpy()
    base = qlike_series(y, res["har"].to_numpy())
    rows = []
    for m in REPORT:
        f = res[m].to_numpy()
        t, p = diebold_mariano(qlike_series(y, f), base, lag=max(h, 1))
        rows.append(
            {
                "model": m,
                "qlike": qlike(y, f),
                "qlike_vs_har": qlike(y, f) / qlike(y, res["har"].to_numpy()),
                "mse_vs_har": mse(y, f) / mse(y, res["har"].to_numpy()),
                "dm_t_vs_har": t if m != "har" else np.nan,
                "dm_p": p if m != "har" else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("model")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--ticker", help="download free daily OHLC at runtime (yfinance)")
    src.add_argument("--csv", help="daily CSV: date + open/high/low/close, or date + close/rv")
    ap.add_argument("--n-days", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 22])
    ap.add_argument("--train-window", type=int, default=1000)
    ap.add_argument("--test-block", type=int, default=63)
    ap.add_argument("--expanding", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.ticker:
        df, label = load_public(args.ticker), args.ticker
    elif args.csv:
        df, label = load_csv(args.csv), Path(args.csv).name
    else:
        df, label = make_synthetic_market(n_days=args.n_days, seed=args.seed), f"synthetic seed {args.seed}"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 120)
    print(f"Data: {label}, {len(df)} days, {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Rolling window {args.train_window}, refit every {args.test_block} days, purge = horizon\n")

    for h in args.horizons:
        res = run_walk_forward(
            df, h, train_window=args.train_window, test_block=args.test_block, expanding=args.expanding
        )
        res.to_csv(out / f"forecasts_h{h}.csv")
        board = leaderboard(res, h)
        board.to_csv(out / f"leaderboard_h{h}.csv")
        print(f"== horizon {h} day(s): {len(res)} out-of-sample forecasts, {res['fold'].nunique()} folds")
        print(board.round(3).to_string())

        clf = classifier_scores(res["z_high"], res["p_high"])
        naive = classifier_scores(res["z_high"], res["ewma"].rank(pct=True))  # current vol level as the score
        print(
            f"   regime classifier: AUC {clf['auc']:.3f}, Brier {clf['brier']:.3f} "
            f"| current-level (EWMA) score AUC {naive['auc']:.3f}"
        )
        if h == 1:
            nxt = df["ret"].shift(-1).reindex(res.index).to_numpy()
            print("   vol targeting at 10% (h=1): realised vol | sd of 63d vol / target")
            for m in ["hist_mean", "ewma", "garch", "har", "regime_har", "ens_equal"]:
                v = vol_target_check(nxt, res[m].to_numpy())
                print(f"     {m:<11} {v['realised_vol']:.3f} | {v['rolling_vol_sd']:.3f}")
        print()


if __name__ == "__main__":
    main()
