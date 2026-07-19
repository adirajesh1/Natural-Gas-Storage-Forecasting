import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from power_forecast.backtesting import run_power_backtest
from power_forecast.pipelines import _component_history
from power_forecast.fundamentals import (
    GAS_HEAT_CONTENT_MMBTU_PER_BCF,
    build_physical_stack,
)
from power_forecast.models.correction import _rolling_predictions, fit_predict_correction


def test_physical_stack_balances_and_bounds_gas_generation():
    hours = pd.date_range("2026-07-13T01:00:00Z", periods=24, freq="h")
    forecast = pd.DataFrame(
        {
            "forecast_origin": pd.Timestamp("2026-07-13T00:00:00Z"),
            "delivery_hour": hours,
            "load_forecast_mw": 70_000.0,
            "wind_forecast_mw": 15_000.0,
            "solar_forecast_mw": np.maximum(0, np.sin(np.linspace(-1, 2, 24))) * 10_000,
            "nuclear_mw": 5_000.0,
            "hydro_mw": 1_000.0,
            "other_nonthermal_mw": 500.0,
            "net_imports_mw": 200.0,
            "battery_net_discharge_mw": 300.0,
            "available_capacity_mw": 90_000.0,
            "conventional_outage_mw": 3_000.0,
        }
    )
    result = build_physical_stack(forecast, heat_rate=7.5)
    assert np.abs(result["balance_error_mw"]).max() < 1e-8
    assert (result["gas_generation_mw"] >= 0).all()
    assert (result["gas_generation_mw"] <= result["dispatchable_thermal_mw"]).all()
    expected = result["gas_generation_mw"] * 7.5 / GAS_HEAT_CONTENT_MMBTU_PER_BCF
    np.testing.assert_allclose(result["gas_burn_bcf_base"], expected)
    assert result["gas_burn_bcf_base"].sum() == pytest.approx(expected.sum())


def test_negative_residual_becomes_curtailment_and_still_balances():
    forecast = pd.DataFrame(
        {
            "forecast_origin": [pd.Timestamp("2026-04-01T00:00:00Z")],
            "delivery_hour": [pd.Timestamp("2026-04-01T01:00:00Z")],
            "load_forecast_mw": [10_000.0],
            "wind_forecast_mw": [12_000.0],
            "solar_forecast_mw": [5_000.0],
            "nuclear_mw": [4_000.0],
            "hydro_mw": [0.0],
            "other_nonthermal_mw": [0.0],
            "net_imports_mw": [0.0],
            "battery_net_discharge_mw": [0.0],
        }
    )
    result = build_physical_stack(forecast)
    assert result.loc[0, "dispatchable_thermal_mw"] == 0
    assert result.loc[0, "curtailment_mw"] == pytest.approx(11_000.0)
    assert result.loc[0, "balance_error_mw"] == pytest.approx(0.0)


def test_correction_is_not_promoted_when_it_does_not_beat_baseline():
    origins = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
    rows = []
    for origin in origins:
        for horizon in range(1, 49):
            baseline = 50_000.0 + horizon
            rows.append(
                {
                    "forecast_origin": origin,
                    "delivery_hour": origin + pd.Timedelta(hours=horizon),
                    "baseline_mw": baseline,
                    "actual_mw": baseline,
                    "horizon_hour": horizon,
                }
            )
    history = pd.DataFrame(rows)
    future = history.loc[history["forecast_origin"] == origins[-1]].copy()
    result, selection = fit_predict_correction(history, future)
    assert not selection.promoted
    assert selection.model_name == "ercot_baseline"
    np.testing.assert_allclose(result["forecast_mw"], result["baseline_mw"])

    history["component"] = "load"
    predictions, metrics = run_power_backtest(
        history, components=("load",)
    )
    assert not predictions.empty
    assert set(metrics["model"]) == {"ercot_baseline", "hour_of_week_fallback"}


def test_rolling_correction_training_excludes_undelivered_labels():
    fit_sizes = []

    class RecordingRegressor(RegressorMixin, BaseEstimator):
        def fit(self, features, target):
            fit_sizes.append(len(features))
            return self

        def predict(self, features):
            return np.zeros(len(features))

    origins = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    rows = []
    for origin in origins:
        for horizon in (*range(1, 41), 72):
            rows.append(
                {
                    "forecast_origin": origin,
                    "delivery_hour": origin + pd.Timedelta(hours=horizon),
                    "horizon_bucket": "test",
                    "baseline_mw": 100.0,
                    "actual_mw": 100.0,
                }
            )
    _rolling_predictions(
        pd.DataFrame(rows),
        RecordingRegressor(),
        ["baseline_mw"],
    )
    assert fit_sizes == [103]


def test_rolling_correction_training_excludes_late_retrievals():
    fit_sizes = []

    class RecordingRegressor(RegressorMixin, BaseEstimator):
        def fit(self, features, target):
            fit_sizes.append(len(features))
            return self

        def predict(self, features):
            return np.zeros(len(features))

    origins = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    rows = []
    for origin_number, origin in enumerate(origins):
        retrieved_at = (
            origins[-1] + pd.Timedelta(hours=1)
            if origin_number == 2
            else origin
        )
        for horizon in range(1, 61):
            rows.append(
                {
                    "forecast_origin": origin,
                    "retrieved_at": retrieved_at,
                    "delivery_hour": origin + pd.Timedelta(hours=horizon),
                    "horizon_bucket": "test",
                    "baseline_mw": 100.0,
                    "actual_mw": 100.0,
                }
            )

    _rolling_predictions(
        pd.DataFrame(rows),
        RecordingRegressor(),
        ["baseline_mw"],
    )

    assert fit_sizes == [107]


def test_gas_generation_share_excludes_actuals_after_forecast_origin():
    origin = pd.Timestamp("2026-07-13T00:00:00Z")
    forecast = pd.DataFrame(
        {
            "forecast_origin": [origin],
            "delivery_hour": [origin + pd.Timedelta(hours=1)],
            "load_forecast_mw": [100.0],
            "wind_forecast_mw": [0.0],
            "solar_forecast_mw": [0.0],
            "nuclear_mw": [0.0],
            "hydro_mw": [0.0],
            "other_nonthermal_mw": [0.0],
            "net_imports_mw": [0.0],
            "battery_net_discharge_mw": [0.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "valid_at": [origin - pd.Timedelta(hours=1), origin + pd.Timedelta(hours=1)],
            "gas_generation_actual_mw": [20.0, 90.0],
            "coal_generation_actual_mw": [80.0, 10.0],
        }
    )

    result = build_physical_stack(forecast, actuals)

    assert result.loc[0, "gas_generation_mw"] == pytest.approx(20.0)


def test_recent_error_uses_latest_forecast_once_per_delivery_hour():
    target_origin = pd.Timestamp("2026-01-02T03:00:00Z")
    rows = []
    for delivery in pd.to_datetime(
        ["2026-01-02T01:00:00Z", "2026-01-02T02:00:00Z"]
    ):
        rows.extend(
            [
                {
                    "component": "load",
                    "issued_at": pd.Timestamp("2026-01-01T00:00:00Z"),
                    "valid_at": delivery,
                    "baseline_mw": 0.0,
                    "actual_mw": 100.0,
                },
                {
                    "component": "load",
                    "issued_at": pd.Timestamp("2026-01-02T00:00:00Z"),
                    "valid_at": delivery,
                    "baseline_mw": 100.0,
                    "actual_mw": 100.0,
                },
            ]
        )
    rows.append(
        {
            "component": "load",
            "issued_at": target_origin,
            "valid_at": target_origin + pd.Timedelta(hours=1),
            "baseline_mw": 100.0,
            "actual_mw": np.nan,
        }
    )

    result = _component_history(pd.DataFrame(rows), pd.DataFrame(), "load")

    target = result.loc[result["forecast_origin"] == target_origin]
    assert target["recent_error_mw"].eq(0.0).all()
