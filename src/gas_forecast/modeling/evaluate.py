from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    """Return mean absolute error."""
    true = pd.Series(y_true, dtype="float64")
    pred = pd.Series(y_pred, dtype="float64")
    return float((true - pred).abs().mean())


def rmse(y_true, y_pred) -> float:
    """Return root mean squared error."""
    true = pd.Series(y_true, dtype="float64")
    pred = pd.Series(y_pred, dtype="float64")
    return float(np.sqrt(((true - pred) ** 2).mean()))


def bias(y_true, y_pred) -> float:
    """Return average forecast error, actual minus predicted."""
    true = pd.Series(y_true, dtype="float64")
    pred = pd.Series(y_pred, dtype="float64")
    return float((true - pred).mean())
