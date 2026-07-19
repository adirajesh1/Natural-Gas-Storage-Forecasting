import pandas as pd
import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from gas_forecast.modeling import RecursiveForecaster, run_recursive_backtest
from gas_forecast.modeling.splitters import ExpandingWindowSplitter
from gas_forecast.data.features import DEFAULT_WEATHER_MODEL_FEATURES, add_storage_features


class HddRegressor(BaseEstimator, RegressorMixin):
    """Small test estimator that exposes the weather input passed to a forecast."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["hdd"].to_numpy()


class StorageDifferenceRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["storage_vs_last_year"].to_numpy()


class StorageAverageDifferenceRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["storage_vs_5yr_avg"].to_numpy()

def _create_mock_features() -> pd.DataFrame:
    # Build 100 weeks of mock data
    dates = pd.date_range(start="2020-01-01", periods=100, freq="W-FRI")
    
    df = pd.DataFrame({
        "date": dates,
        "week_of_year": dates.isocalendar().week.astype(int),
        "weekly_change_bcf": np.random.uniform(-50, 50, 100),
        "storage_bcf": np.cumsum(np.random.uniform(-50, 50, 100)) + 3000.0,
        "temperature_f": np.random.uniform(30, 80, 100),
        "hdd": np.random.uniform(0, 100, 100),
        "cdd": np.random.uniform(0, 50, 100),
        "weather_days": [7.0] * 100,
        "storage_5yr_avg": [3000.0] * 100,
        "storage_bcf_lag52": [2900.0] * 100,
    })
    
    # Add calendar/seasonal encodings
    week_angle = 2 * np.pi * df["week_of_year"] / 52.0
    month_angle = 2 * np.pi * dates.month / 12.0
    df["week_sin"] = np.sin(week_angle)
    df["week_cos"] = np.cos(week_angle)
    df["month_sin"] = np.sin(month_angle)
    df["month_cos"] = np.cos(month_angle)
    df["is_injection_season"] = ((dates.month >= 4) & (dates.month <= 10)).astype(int)
    
    # Add lag weather/storage columns to fit the 14 weather model features
    df["hdd_lag1"] = df["hdd"].shift(1).fillna(50.0)
    df["cdd_lag1"] = df["cdd"].shift(1).fillna(10.0)
    df["weekly_change_lag1"] = df["weekly_change_bcf"].shift(1).fillna(0.0)
    
    # Add rolling weather/storage columns
    df["hdd_rolling_4wk"] = df["hdd"].rolling(window=4, min_periods=1).mean().shift(1).fillna(50.0)
    df["cdd_rolling_4wk"] = df["cdd"].rolling(window=4, min_periods=1).mean().shift(1).fillna(10.0)
    df["weekly_change_rolling_4wk"] = df["weekly_change_bcf"].rolling(window=4, min_periods=1).mean().shift(1).fillna(0.0)
    
    # Add storage comparisons (aligned to t-1 to match features.py)
    df["storage_bcf_lag1"] = df["storage_bcf"].shift(1).fillna(3000.0)
    df["storage_vs_5yr_avg"] = df["storage_bcf_lag1"] - df["storage_5yr_avg"].shift(1).fillna(3000.0)
    df["storage_vs_last_year"] = df["storage_bcf_lag1"] - df["storage_bcf"].shift(53).fillna(2900.0)
    
    return df

def test_recursive_forecaster():
    df = _create_mock_features()
    feature_cols = list(DEFAULT_WEATHER_MODEL_FEATURES)
    target_col = "weekly_change_bcf"
    
    # Fit model on first 80 weeks
    train_df = df.iloc[:80]
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    model = LinearRegression().fit(X_train, y_train)
    
    # Instantiate RecursiveForecaster
    forecaster = RecursiveForecaster(model, feature_cols)
    
    # Run 4 weeks forecast starting at week 80
    start_date = df.iloc[80]["date"]
    res = forecaster.predict_horizon(df, start_date=start_date, horizon_weeks=4)
    
    assert len(res) == 4
    assert {
        "date",
        "forecast_origin",
        "region",
        "horizon",
        "model_key",
        "reconciliation_method",
        "weather_provider",
        "weather_run",
        "p10",
        "p50",
        "p90",
        "predicted_weekly_change",
        "projected_storage",
        "actual_weekly_change",
        "actual_storage",
        "week_of_year",
    }.issubset(res.columns)
    assert res["horizon"].tolist() == [1, 2, 3, 4]
    # Check that projected storage accumulates correctly
    expected_storage_1 = df.iloc[79]["storage_bcf"] + res.iloc[0]["predicted_weekly_change"]
    assert pytest.approx(res.iloc[0]["projected_storage"]) == expected_storage_1
    expected_storage_2 = expected_storage_1 + res.iloc[1]["predicted_weekly_change"]
    assert pytest.approx(res.iloc[1]["projected_storage"]) == expected_storage_2

def test_recursive_backtester():
    df = _create_mock_features()
    feature_cols = list(DEFAULT_WEATHER_MODEL_FEATURES)
    target_col = "weekly_change_bcf"
    
    model = LinearRegression()
    
    # Use expanding window: initial train up to week 60, validation window of 8 weeks, step size 8 weeks
    # For 100 weeks, this will yield 4 folds:
    # Fold 1: train 60, val 8 (60-67)
    # Fold 2: train 68, val 8 (68-75)
    # Fold 3: train 76, val 8 (76-83)
    # Fold 4: train 84, val 8 (84-91)
    splitter = ExpandingWindowSplitter(
        date_col="date",
        initial_train_start=df.iloc[0]["date"],
        initial_train_end=df.iloc[59]["date"],
        val_weeks=8,
        step_weeks=8,
    )
    
    predictions_df, metrics_df = run_recursive_backtest(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        date_col="date",
        model=model,
        splitter=splitter,
        horizon_weeks=4,
        interval_coverage=0.80,
        min_calibration_samples=1,
    )
    
    assert not predictions_df.empty
    assert not metrics_df.empty
    assert "horizon_weeks_ahead" in metrics_df.columns
    assert "mae" in metrics_df.columns
    assert "rmse" in metrics_df.columns
    assert "bias" in metrics_df.columns
    assert {"interval_lower", "interval_upper", "interval_radius"}.issubset(
        predictions_df.columns
    )
    assert {"empirical_coverage", "interval_samples"}.issubset(metrics_df.columns)
    
    # Ensure no leakage: validation starts strictly after training ends
    for fold in predictions_df["fold"].unique():
        fold_preds = predictions_df[predictions_df["fold"] == fold]
        # Get train end for this fold splitter split
        for f, (train_idx, val_idx) in enumerate(splitter.split(df), start=1):
            if f == fold:
                train_dates = df.iloc[train_idx]["date"]
                val_dates = df.iloc[val_idx]["date"]
                assert val_dates.min() > train_dates.max()


def test_recursive_forecaster_defaults_to_seasonal_future_inputs():
    dates = pd.date_range("2023-01-06", periods=104, freq="W-FRI")
    week_numbers = dates.isocalendar().week.astype(int).to_numpy()
    df = pd.DataFrame(
        {
            "date": dates,
            "week_of_year": week_numbers,
            "weekly_change_bcf": np.zeros(len(dates)),
            "storage_bcf": np.full(len(dates), 3_000.0),
            "temperature_f": np.full(len(dates), 50.0),
            "hdd": week_numbers.astype(float),
            "cdd": np.zeros(len(dates)),
            "weather_days": np.full(len(dates), 7.0),
        }
    )
    start_date = dates[52]
    df.loc[52, "hdd"] = 999.0

    forecaster = RecursiveForecaster(HddRegressor(), ["hdd"])
    seasonal = forecaster.predict_horizon(df, start_date, horizon_weeks=1)
    observed = forecaster.predict_horizon(
        df,
        start_date,
        horizon_weeks=1,
        forecast_input_mode="observed",
    )

    assert seasonal.loc[0, "predicted_weekly_change"] == pytest.approx(1.0)
    assert observed.loc[0, "predicted_weekly_change"] == pytest.approx(999.0)


def test_recursive_forecaster_uses_weather_scenario_available_at_origin():
    dates = pd.date_range("2023-01-06", periods=104, freq="W-FRI")
    week_numbers = dates.isocalendar().week.astype(int).to_numpy()
    df = pd.DataFrame(
        {
            "date": dates,
            "duoarea": "R48",
            "week_of_year": week_numbers,
            "weekly_change_bcf": np.zeros(len(dates)),
            "storage_bcf": np.full(len(dates), 3_000.0),
            "temperature_f": np.full(len(dates), 50.0),
            "hdd": week_numbers.astype(float),
            "cdd": np.zeros(len(dates)),
            "weather_days": np.full(len(dates), 7.0),
        }
    )
    start_date = dates[52]
    scenarios = pd.DataFrame(
        {
            "date": [start_date, start_date],
            "duoarea": ["R48", "R48"],
            "issued_at": [start_date - pd.Timedelta(days=8), start_date + pd.Timedelta(days=1)],
            "temperature_f": [40.0, 60.0],
            "hdd": [77.0, 999.0],
            "cdd": [0.0, 0.0],
            "weather_days": [7.0, 7.0],
        }
    )

    forecast = RecursiveForecaster(HddRegressor(), ["hdd"]).predict_horizon(
        df,
        start_date,
        horizon_weeks=1,
        forecast_input_mode="scenario",
        weather_scenario=scenarios,
    )

    assert forecast.loc[0, "predicted_weekly_change"] == pytest.approx(77.0)


def test_recursive_backtest_uses_seasonal_inputs_by_default():
    dates = pd.date_range("2023-01-06", periods=60, freq="W-FRI")
    week_numbers = dates.isocalendar().week.astype(int).to_numpy()
    df = pd.DataFrame(
        {
            "date": dates,
            "week_of_year": week_numbers,
            "weekly_change_bcf": np.zeros(len(dates)),
            "storage_bcf": np.full(len(dates), 3_000.0),
            "temperature_f": np.full(len(dates), 50.0),
            "hdd": week_numbers.astype(float),
            "cdd": np.zeros(len(dates)),
            "weather_days": np.full(len(dates), 7.0),
        }
    )
    df.loc[52, "hdd"] = 999.0
    splitter = ExpandingWindowSplitter(
        date_col="date",
        initial_train_start=dates[0],
        initial_train_end=dates[51],
        val_weeks=1,
        step_weeks=8,
    )

    predictions, _ = run_recursive_backtest(
        df=df,
        feature_cols=["hdd"],
        target_col="weekly_change_bcf",
        date_col="date",
        model=HddRegressor(),
        splitter=splitter,
        horizon_weeks=1,
    )

    first_prediction = predictions.loc[predictions["fold"] == 1].iloc[0]
    assert first_prediction["predicted_weekly_change"] == pytest.approx(1.0)
    assert first_prediction["forecast_input_mode"] == "seasonal"


def test_recursive_forecaster_aligns_year_ago_storage_to_prior_week_state():
    dates = pd.date_range("2023-01-06", periods=60, freq="W-FRI")
    df = pd.DataFrame(
        {
            "date": dates,
            "week_of_year": dates.isocalendar().week.astype(int),
            "weekly_change_bcf": np.ones(len(dates)),
            "storage_bcf": np.arange(100.0, 160.0),
            "temperature_f": np.full(len(dates), 50.0),
            "hdd": np.full(len(dates), 20.0),
            "cdd": np.zeros(len(dates)),
            "weather_days": np.full(len(dates), 7.0),
        }
    )

    forecast = RecursiveForecaster(
        StorageDifferenceRegressor(),
        ["storage_vs_last_year"],
    ).predict_horizon(df, dates[53], horizon_weeks=1)

    # At target index 53, the feature is S_52 - S_0 = 152 - 100.
    assert forecast.loc[0, "predicted_weekly_change"] == pytest.approx(52.0)


def test_recursive_forecaster_rebuilds_training_storage_vs_five_year_average():
    dates = pd.date_range("2018-01-05", periods=330, freq="W-FRI")
    df = pd.DataFrame(
        {
            "date": dates,
            "week_of_year": dates.isocalendar().week.astype(int).to_numpy(),
            "weekly_change_bcf": np.ones(len(dates)),
            "storage_bcf": np.arange(1_000.0, 1_000.0 + len(dates)),
            "temperature_f": np.full(len(dates), 50.0),
            "hdd": np.full(len(dates), 20.0),
            "cdd": np.zeros(len(dates)),
            "weather_days": np.full(len(dates), 7.0),
        }
    )
    target_date = dates[320]
    expected = add_storage_features(df).loc[
        lambda frame: frame["date"] == target_date,
        "storage_vs_5yr_avg",
    ].iloc[0]

    forecast = RecursiveForecaster(
        StorageAverageDifferenceRegressor(),
        ["storage_vs_5yr_avg"],
    ).predict_horizon(df, target_date, horizon_weeks=1)

    assert forecast.loc[0, "predicted_weekly_change"] == pytest.approx(expected)


def test_recursive_forecaster_rejects_mixed_region_input():
    df = _create_mock_features().iloc[:20].copy()
    df["duoarea"] = ["R31", "R32"] * 10

    with pytest.raises(ValueError, match="one region"):
        RecursiveForecaster(HddRegressor(), ["hdd"]).predict_horizon(
            df,
            df.iloc[10]["date"],
            horizon_weeks=1,
        )


def test_recursive_backtest_rejects_unknown_input_mode():
    df = _create_mock_features()
    splitter = ExpandingWindowSplitter(
        date_col="date",
        initial_train_start=df.iloc[0]["date"],
        initial_train_end=df.iloc[59]["date"],
        val_weeks=1,
        step_weeks=1,
    )

    with pytest.raises(ValueError, match="forecast_input_mode"):
        run_recursive_backtest(
            df=df,
            feature_cols=["hdd"],
            target_col="weekly_change_bcf",
            date_col="date",
            model=HddRegressor(),
            splitter=splitter,
            forecast_input_mode="unknown",
        )
