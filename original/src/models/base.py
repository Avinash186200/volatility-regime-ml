"""
Contract every model in this repo follows: fit on a training window, predict
on the next one. Keeping this interface identical across models is what makes
the head-to-head comparison in the project brief fair -- no model gets a
bespoke evaluation path.
"""
from typing import Protocol

import pandas as pd


class Forecaster(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster": ...

    def predict(self, X: pd.DataFrame) -> pd.Series: ...
