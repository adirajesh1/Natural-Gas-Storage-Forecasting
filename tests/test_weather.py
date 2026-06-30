import pandas as pd

from gas_forecast.data.weather import (
    aggregate_weather_to_storage_weeks,
    assign_storage_week_end,
)


def test_assign_storage_week_end_maps_saturday_through_friday_to_friday():
    dates = pd.Series(pd.date_range("2024-01-06", periods=7))

    result = assign_storage_week_end(dates)

    assert result.tolist() == [pd.Timestamp("2024-01-12")] * 7


def test_aggregate_weather_to_storage_weeks_drops_incomplete_weeks():
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-06", periods=10),
            "temperature_f": range(10),
            "hdd": [1.0] * 10,
            "cdd": [2.0] * 10,
        }
    )

    weekly = aggregate_weather_to_storage_weeks(daily)

    assert len(weekly) == 1
    assert weekly.loc[0, "date"] == pd.Timestamp("2024-01-12")
    assert weekly.loc[0, "weather_days"] == 7
    assert weekly.loc[0, "hdd"] == 7.0
    assert weekly.loc[0, "cdd"] == 14.0
