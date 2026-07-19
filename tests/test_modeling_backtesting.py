import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from gas_forecast.modeling.splitters import ExpandingWindowSplitter
from gas_forecast.modeling.backtesting import run_backtest


class MeanRegressor(BaseEstimator, RegressorMixin):
    fit_calls = 0

    def fit(self, X, y):
        type(self).fit_calls += 1
        self.mean_ = float(y.mean())
        return self

    def predict(self, X):
        return [self.mean_] * len(X)


class OverlappingSplitter:
    def split(self, df):
        yield [0, 1, 2, 3], [3, 4]


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-05", periods=8, freq="W-FRI"),
            "feature": [1.0, 2.0, 3.0, 4.0, None, 6.0, 7.0, 8.0],
            "weekly_change_bcf": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
            "weekly_change_lag1": [None, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0],
        }
    )


def test_run_backtest_clones_model_and_returns_predictions_and_metrics():
    MeanRegressor.fit_calls = 0
    df = _training_frame()
    splitter = ExpandingWindowSplitter(
        "date",
        initial_train_start="2024-01-05",
        initial_train_end="2024-01-26",
        val_weeks=1,
        step_weeks=1,
    )

    predictions, metrics = run_backtest(
        df,
        feature_cols=["feature"],
        target_col="weekly_change_bcf",
        date_col="date",
        model=MeanRegressor(),
        splitter=splitter,
    )

    assert MeanRegressor.fit_calls == 3
    assert predictions["fold"].tolist() == [1, 2, 3]
    assert predictions["date"].tolist() == [
        pd.Timestamp("2024-02-09"),
        pd.Timestamp("2024-02-16"),
        pd.Timestamp("2024-02-23"),
    ]
    assert {
        "date",
        "weekly_change_bcf",
        "fold",
        "predicted_weekly_change",
        "forecast_deviation",
    }.issubset(predictions.columns)
    assert metrics["fold"].tolist() == [1, 2, 3, "overall"]
    assert not metrics[["mae", "rmse", "bias"]].isna().any().any()
    assert metrics.loc[0, "mae"] == pytest.approx(7.0)


def test_run_backtest_does_not_mutate_input_or_create_lag_features():
    df = _training_frame()
    original = df.copy(deep=True)
    splitter = ExpandingWindowSplitter(
        "date",
        initial_train_start="2024-01-05",
        initial_train_end="2024-01-26",
        val_weeks=1,
        step_weeks=1,
    )

    predictions, _ = run_backtest(
        df,
        feature_cols=["feature"],
        target_col="weekly_change_bcf",
        date_col="date",
        model=MeanRegressor(),
        splitter=splitter,
    )

    pd.testing.assert_frame_equal(df, original)
    assert "weekly_change_lag1" in df.columns
    assert "weekly_change_lag1" not in predictions.columns


def test_run_backtest_rejects_splitter_with_training_validation_overlap():
    with pytest.raises(ValueError, match="must precede"):
        run_backtest(
            _training_frame(),
            feature_cols=["feature"],
            target_col="weekly_change_bcf",
            date_col="date",
            model=MeanRegressor(),
            splitter=OverlappingSplitter(),
        )


def test_run_backtest_with_lightgbm_and_xgboost_if_installed():
    from gas_forecast.modeling.config import sklearn_model_configs

    df = _training_frame()
    splitter = ExpandingWindowSplitter(
        "date",
        initial_train_start="2024-01-05",
        initial_train_end="2024-01-26",
        val_weeks=1,
        step_weeks=1,
    )

    configs = {c.key: c for c in sklearn_model_configs()}

    if "lightgbm" in configs:
        predictions, metrics = run_backtest(
            df,
            feature_cols=["feature"],
            target_col="weekly_change_bcf",
            date_col="date",
            model=configs["lightgbm"].build(),
            splitter=splitter,
        )
        assert not predictions.empty
        assert not metrics.empty

    if "xgboost" in configs:
        predictions, metrics = run_backtest(
            df,
            feature_cols=["feature"],
            target_col="weekly_change_bcf",
            date_col="date",
            model=configs["xgboost"].build(),
            splitter=splitter,
        )
        assert not predictions.empty
        assert not metrics.empty
