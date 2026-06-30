import pandas as pd
import pytest

from gas_forecast.plotting import plot_weekly_change_forecast


def test_plot_weekly_change_forecast_computes_missing_deviation_and_sorts_dates():
    forecast = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-12", "2024-01-05"]),
            "weekly_change_bcf": [12.0, 10.0],
            "predicted_weekly_change": [9.0, 11.0],
        }
    )

    fig = plot_weekly_change_forecast(forecast)

    assert len(fig.data) == 3
    assert list(fig.data[0].x) == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-12"),
    ]
    assert list(fig.data[2].y) == [-1.0, 3.0]
    assert "forecast_deviation" not in forecast.columns


def test_plot_weekly_change_forecast_adds_band_and_outlier_traces():
    forecast = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-12"]),
            "weekly_change_bcf": [10.0, 20.0],
            "predicted_weekly_change": [11.0, 12.0],
            "forecast_deviation": [-1.0, 8.0],
            "lower_band": [9.0, 10.0],
            "upper_band": [13.0, 14.0],
            "outside_band": [False, True],
        }
    )

    fig = plot_weekly_change_forecast(forecast)

    trace_names = [trace.name for trace in fig.data]
    assert "+/- 1 std range" in trace_names
    assert "Outside +/- 1 std" in trace_names


def test_plot_weekly_change_forecast_validates_required_columns():
    forecast = pd.DataFrame({"date": pd.to_datetime(["2024-01-05"])})

    with pytest.raises(ValueError, match="missing required columns"):
        plot_weekly_change_forecast(forecast)
