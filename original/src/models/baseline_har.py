"""
HAR-RV: the standard benchmark in volatility forecasting (Corsi, 2009).

Regresses next-period realized variance on the daily, weekly (5-day), and
monthly (21-day) rolling averages of past realized variance. Three
coefficients, ordinary least squares. This is the floor every other model in
this project has to beat -- if a fancier model doesn't clear this, that's a
real result, not a failure.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression


def build_har_features(realized_var: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "rv_daily": realized_var,
        "rv_weekly": realized_var.rolling(5).mean(),
        "rv_monthly": realized_var.rolling(21).mean(),
    })


class HARModel:
    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARModel":
        aligned = pd.concat([X, y.rename("target")], axis=1).dropna()
        self.model.fit(aligned[X.columns], aligned["target"])
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(X.ffill())
        return pd.Series(preds, index=X.index)
