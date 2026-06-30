from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "weekly_change_bcf"

_WEATHER_JOIN_COLUMNS = (
    "date",
    "duoarea",
    "temperature_f",
    "hdd",
    "cdd",
    "weather_days",
)

_STORAGE_JOIN_COLUMNS = (
    "date",
    "duoarea",
    "storage_bcf",
    "weekly_change_bcf",
    "year",
    "month",
    "week_of_year",
)

BASE_MODEL_COLUMNS = _STORAGE_JOIN_COLUMNS + (
    "temperature_f",
    "hdd",
    "cdd",
    "weather_days",
)

CALENDAR_FEATURE_COLUMNS = (
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
)

WEATHER_FEATURE_COLUMNS = (
    "hdd_per_day",
    "cdd_per_day",
    "hdd_lag1",
    "hdd_lag4",
    "cdd_lag1",
    "cdd_lag4",
)

STORAGE_FEATURE_COLUMNS = (
    "weekly_change_lag1",
    "weekly_change_lag4",
    "storage_bcf_lag52",
    "weekly_change_yoy",
)

ENGINEERED_FEATURE_COLUMNS = (
    CALENDAR_FEATURE_COLUMNS
    + WEATHER_FEATURE_COLUMNS
    + STORAGE_FEATURE_COLUMNS
)

DEFAULT_WEATHER_MODEL_FEATURES = (
    *CALENDAR_FEATURE_COLUMNS,
    "hdd",
    "cdd",
    "hdd_lag1",
    "weekly_change_lag1",
)


def _lag_within_region(
    frame: pd.DataFrame,
    column: str,
    lag: int,
    *,
    group_col: str = "duoarea",
) -> pd.Series:
    """Shift a column within each region to avoid cross-region leakage."""
    if group_col in frame.columns:
        return frame.groupby(group_col, group_keys=False)[column].shift(lag)
    return frame[column].shift(lag)


def join_weather_storage(
    storage: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Join weekly storage and weather on EIA week-ending Friday and region."""
    missing_storage = {"date", "duoarea"} - set(storage.columns)
    if missing_storage:
        raise ValueError(
            f"Storage missing required columns: {sorted(missing_storage)}"
        )

    missing_weather = set(_WEATHER_JOIN_COLUMNS) - set(weather.columns)
    if missing_weather:
        raise ValueError(
            f"Weather missing required columns: {sorted(missing_weather)}"
        )

    weather_subset = weather[list(_WEATHER_JOIN_COLUMNS)]
    return storage.merge(
        weather_subset,
        on=["date", "duoarea"],
        how="inner",
        validate="one_to_one",
    )


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical week-of-year and month encodings."""
    df = frame.copy()
    dates = pd.to_datetime(df["date"])
    week_angle = 2 * np.pi * df["week_of_year"] / 52.0
    month_angle = 2 * np.pi * dates.dt.month / 12.0
    df["week_sin"] = np.sin(week_angle)
    df["week_cos"] = np.cos(week_angle)
    df["month_sin"] = np.sin(month_angle)
    df["month_cos"] = np.cos(month_angle)
    return df


def add_weather_features(
    frame: pd.DataFrame,
    *,
    lag_weeks: tuple[int, ...] = (1, 4),
) -> pd.DataFrame:
    """Add normalized weather metrics and lagged HDD/CDD."""
    df = frame.copy()
    df["hdd_per_day"] = df["hdd"] / df["weather_days"]
    df["cdd_per_day"] = df["cdd"] / df["weather_days"]

    for lag in lag_weeks:
        df[f"hdd_lag{lag}"] = _lag_within_region(df, "hdd", lag)
        df[f"cdd_lag{lag}"] = _lag_within_region(df, "cdd", lag)

    return df


def add_storage_features(
    frame: pd.DataFrame,
    *,
    lag_weeks: tuple[int, ...] = (1, 4),
    yoy_lag: int = 52,
) -> pd.DataFrame:
    """Add lagged storage changes and year-over-year comparisons."""
    df = frame.copy()

    for lag in lag_weeks:
        df[f"weekly_change_lag{lag}"] = _lag_within_region(
            df, TARGET_COLUMN, lag
        )

    df["storage_bcf_lag52"] = _lag_within_region(df, "storage_bcf", yoy_lag)
    df["weekly_change_yoy"] = df[TARGET_COLUMN] - _lag_within_region(
        df, TARGET_COLUMN, yoy_lag
    )

    return df


def build_weekly_model_features(
    storage: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    region: str | None = None,
    lag_weeks: tuple[int, ...] = (1, 4),
) -> pd.DataFrame:
    """Join storage and weather, then build a model-ready weekly feature table."""
    if region is not None:
        storage = storage.loc[storage["duoarea"] == region].copy()
        weather = weather.loc[weather["duoarea"] == region].copy()

    joined = join_weather_storage(storage, weather)
    df = add_calendar_features(joined)
    df = add_weather_features(df, lag_weeks=lag_weeks)
    df = add_storage_features(df, lag_weeks=lag_weeks)

    return df.sort_values(["duoarea", "date"]).reset_index(drop=True)


def validate_weekly_model_features(
    frame: pd.DataFrame,
    *,
    region: str | None = None,
) -> None:
    """Validate a weekly model feature table."""
    required = set(BASE_MODEL_COLUMNS) | set(ENGINEERED_FEATURE_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if frame["date"].isna().any():
        raise ValueError("Found missing values in 'date'.")

    if frame.duplicated(subset=["date", "duoarea"]).any():
        raise ValueError("Found duplicate (date, duoarea) rows.")

    for _, group in frame.groupby("duoarea"):
        if not group["date"].is_monotonic_increasing:
            raise ValueError("Feature data is not sorted by date within region.")

    if region is not None:
        regions = frame["duoarea"].unique()
        if len(regions) != 1 or regions[0] != region:
            raise ValueError(
                f"Expected duoarea {region!r}, found {list(regions)}."
            )

    if (frame["weather_days"] <= 0).any():
        raise ValueError("Found non-positive weather_days values.")
