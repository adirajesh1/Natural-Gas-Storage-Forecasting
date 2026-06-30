import pandas as pd

from gas_forecast.data.storage import calculate_weekly_storage_change


def test_calculate_weekly_storage_change_is_grouped_by_region():
    storage = pd.DataFrame(
        {
            "duoarea": ["R01", "R01", "R02", "R02"],
            "period": pd.to_datetime(
                ["2024-01-05", "2024-01-12", "2024-01-05", "2024-01-12"]
            ),
            "value": [100, 110, 200, 190],
        }
    )

    result = calculate_weekly_storage_change(storage)

    assert pd.isna(result.loc[0, "weekly_change_bcf"])
    assert result.loc[1, "weekly_change_bcf"] == 10.0
    assert pd.isna(result.loc[2, "weekly_change_bcf"])
    assert result.loc[3, "weekly_change_bcf"] == -10.0
