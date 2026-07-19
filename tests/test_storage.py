import pandas as pd
import pytest

from gas_forecast.data.storage import (
    calculate_weekly_storage_change,
    validate_storage_region,
)


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


def test_validate_storage_region_rejects_missing_week_before_change_calculation():
    storage = pd.DataFrame(
        {
            "duoarea": ["R48", "R48"],
            "period": pd.to_datetime(["2024-01-05", "2024-01-19"]),
            "value": [3_000.0, 3_020.0],
        }
    )

    with pytest.raises(ValueError, match="consecutive weekly periods"):
        validate_storage_region(storage)
