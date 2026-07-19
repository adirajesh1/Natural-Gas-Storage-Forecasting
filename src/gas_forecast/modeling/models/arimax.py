"""Sklearn-compatible one-step ARIMAX challenger."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAXRegressor(RegressorMixin, BaseEstimator):
    """Dynamic regression with ARMA errors for chronological one-step tests."""

    supports_recursive = False

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 1),
        trend: str = "c",
        maxiter: int = 100,
    ) -> None:
        self.order = order
        self.trend = trend
        self.maxiter = maxiter

    def fit(self, X: pd.DataFrame, y: pd.Series):
        features = pd.DataFrame(X).astype(float)
        target = pd.Series(y).astype(float)
        if features.isna().any().any() or target.isna().any():
            raise ValueError("ARIMAXRegressor does not accept missing values.")
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        self.n_features_in_ = features.shape[1]
        self._fitted = SARIMAX(
            target.to_numpy(),
            exog=features.to_numpy(),
            order=self.order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=self.maxiter)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "_fitted"):
            raise RuntimeError("Call fit() before predict().")
        features = pd.DataFrame(X).astype(float)
        if features.shape[1] != self.n_features_in_:
            raise ValueError("ARIMAX feature count does not match fitted data.")
        return np.asarray(
            self._fitted.get_forecast(
                steps=len(features),
                exog=features.to_numpy(),
            ).predicted_mean
        )

