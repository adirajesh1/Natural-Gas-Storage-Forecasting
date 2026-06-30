from __future__ import annotations

import pandas as pd

from gas_forecast.data.storage_validation import (
    validate_storage_region,
    validate_weekly_storage,
)


def clean_weekly_storage(raw_df, start_date=None, end_date=None):
    df = raw_df.copy()

    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series-description"] = df["series-description"].astype(str)

    if start_date is not None:
        df = df[df["period"] >= pd.to_datetime(start_date)]

    if end_date is not None:
        df = df[df["period"] <= pd.to_datetime(end_date)]

    df["year"] = df["period"].dt.year
    df["month"] = df["period"].dt.month
    df["week_of_year"] = df["period"].dt.isocalendar().week.astype(int)

    dedupe_cols = (
        ["duoarea", "period", "series"]
        if "series" in df.columns
        else ["duoarea", "period"]
    )
    df = df.drop_duplicates(subset=dedupe_cols, keep="last")

    df = df.sort_values(["duoarea", "period"]).reset_index(drop=True)

    validate_weekly_storage(df)

    return df


def select_region(storage: pd.DataFrame, region: str | list[str]) -> pd.DataFrame:
    if isinstance(region, str):
        regions = [region]
    else:
        regions = region

    available_regions = set(storage["duoarea"].unique())
    missing_regions = [r for r in regions if r not in available_regions]
    if missing_regions:
        raise ValueError(f"Region(s) {missing_regions} not found in storage['duoarea']")

    selected_storage = storage.loc[storage["duoarea"].isin(regions)]
    validate_storage_region(selected_storage)

    return selected_storage


def calculate_weekly_storage_change(storage: pd.DataFrame) -> pd.DataFrame:
    df = storage.copy()
    if "duoarea" in df.columns:
        df["weekly_change_bcf"] = df.groupby("duoarea")["value"].diff()
    else:
        df["weekly_change_bcf"] = df["value"].diff()
    return df


def prepare_storage_model_data(storage: pd.DataFrame) -> pd.DataFrame:
    df = storage.copy()

    df = df.rename(
        columns={
            "period": "date",
            "value": "storage_bcf",
        }
    )

    df = df[
        [
            "date",
            "storage_bcf",
            "weekly_change_bcf",
            "year",
            "month",
            "week_of_year",
            "duoarea",
        ]
    ]

    df = df.dropna(subset=["weekly_change_bcf"])

    return df.reset_index(drop=True)
