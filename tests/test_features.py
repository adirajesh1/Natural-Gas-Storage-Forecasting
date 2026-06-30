import pandas as pd

from gas_forecast.data.features import build_weekly_model_features


def test_build_weekly_model_features_keeps_lags_within_region():
    dates = pd.to_datetime(["2024-01-05", "2024-01-12"] * 2)
    storage = pd.DataFrame(
        {
            "date": dates,
            "duoarea": ["R01", "R01", "R02", "R02"],
            "storage_bcf": [100, 110, 200, 190],
            "weekly_change_bcf": [5, 10, -2, -10],
            "year": [2024, 2024, 2024, 2024],
            "month": [1, 1, 1, 1],
            "week_of_year": [1, 2, 1, 2],
        }
    )
    weather = pd.DataFrame(
        {
            "date": dates,
            "duoarea": ["R01", "R01", "R02", "R02"],
            "temperature_f": [30, 31, 70, 71],
            "hdd": [35, 34, 0, 0],
            "cdd": [0, 0, 5, 6],
            "weather_days": [7, 7, 7, 7],
        }
    )

    features = build_weekly_model_features(storage, weather)

    first_rows = features.groupby("duoarea").head(1)
    assert first_rows["weekly_change_lag1"].isna().all()
    assert first_rows["hdd_lag1"].isna().all()

    second_rows = features.groupby("duoarea").tail(1).set_index("duoarea")
    assert second_rows.loc["R01", "weekly_change_lag1"] == 5
    assert second_rows.loc["R02", "weekly_change_lag1"] == -2
