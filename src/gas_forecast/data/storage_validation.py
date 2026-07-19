from __future__ import annotations

import pandas as pd


def validate_weekly_storage(storage: pd.DataFrame) -> None:
    required_columns = {
        "period",
        "value",
        "series-description",
        "duoarea",
        "year",
        "month",
        "week_of_year",
    }

    missing_cols = required_columns - set(storage.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if storage["period"].isna().any():
        raise ValueError("Found missing values in 'period'.")

    if storage["value"].isna().any():
        n_missing = storage["value"].isna().sum()
        raise ValueError(f"Found {n_missing} missing values in 'value'.")

    if (storage["value"] < 0).any():
        raise ValueError("Found negative values in 'value'.")

    if not storage.sort_values(["duoarea", "period"]).index.equals(storage.index):
        raise ValueError("Storage data is not sorted by ['duoarea', 'period'].")


def validate_storage_region(df: pd.DataFrame) -> None:
    for _, group in df.groupby("duoarea"):
        if group["period"].duplicated().any():
            raise ValueError("Selected region has duplicate periods.")

        if not group["period"].is_monotonic_increasing:
            raise ValueError("Selected region is not sorted by period.")

        period_gaps = group["period"].diff().dropna()
        if not period_gaps.eq(pd.Timedelta(weeks=1)).all():
            raise ValueError("Selected region does not have consecutive weekly periods.")
