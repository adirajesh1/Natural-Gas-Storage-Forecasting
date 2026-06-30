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
    "hdd_rolling_4wk",
    "hdd_rolling_8wk",
    "cdd_rolling_4wk",
    "cdd_rolling_8wk",
    "hdd_vs_4wk_avg",
    "cdd_vs_4wk_avg",
)

STORAGE_FEATURE_COLUMNS = (
    "weekly_change_lag1",
    "weekly_change_lag4",
    "weekly_change_rolling_4wk",
    "weekly_change_rolling_8wk",
    "storage_bcf_lag52",
    "storage_vs_last_year",
    "storage_5yr_avg",
    "storage_vs_5yr_avg",
    "weekly_change_yoy",
)

SEASON_FEATURE_COLUMNS = (
    "is_injection_season",
    "is_withdrawal_season",
    "is_shoulder_month",
)

ENGINEERED_FEATURE_COLUMNS = (
    CALENDAR_FEATURE_COLUMNS
    + WEATHER_FEATURE_COLUMNS
    + STORAGE_FEATURE_COLUMNS
    + SEASON_FEATURE_COLUMNS
)

DEFAULT_WEATHER_MODEL_FEATURES = (
    *CALENDAR_FEATURE_COLUMNS,
    "hdd",
    "cdd",
    "hdd_lag1",
    "hdd_rolling_4wk",
    "cdd_rolling_4wk",
    "weekly_change_lag1",
    "weekly_change_rolling_4wk",
    "storage_vs_5yr_avg",
    "storage_vs_last_year",
    "is_injection_season",
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


def _rolling_mean_within_region(
    frame: pd.DataFrame,
    column: str,
    window: int,
    *,
    group_col: str = "duoarea",
) -> pd.Series:
    """Trailing rolling mean using only prior rows within each region."""
    shifted = _lag_within_region(frame, column, 1, group_col=group_col)
    if group_col in frame.columns:
        return shifted.groupby(frame[group_col], group_keys=False).rolling(
            window=window,
            min_periods=window,
        ).mean().reset_index(level=0, drop=True)
    return shifted.rolling(window=window, min_periods=window).mean()


def _same_week_history_mean(
    frame: pd.DataFrame,
    column: str,
    *,
    years: int = 5,
    group_col: str = "duoarea",
) -> pd.Series:
    """Mean of prior same-week observations within each region."""
    if group_col in frame.columns:
        return frame.groupby([group_col, "week_of_year"], group_keys=False)[
            column
        ].transform(lambda values: values.shift(1).rolling(years, min_periods=1).mean())
    return frame.groupby("week_of_year")[column].transform(
        lambda values: values.shift(1).rolling(years, min_periods=1).mean()
    )


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
    """Add cyclical calendar encodings and gas-season flags."""
    df = frame.copy()
    dates = pd.to_datetime(df["date"])
    week_angle = 2 * np.pi * df["week_of_year"] / 52.0
    month_angle = 2 * np.pi * dates.dt.month / 12.0
    df["week_sin"] = np.sin(week_angle)
    df["week_cos"] = np.cos(week_angle)
    df["month_sin"] = np.sin(month_angle)
    df["month_cos"] = np.cos(month_angle)
    df["is_injection_season"] = dates.dt.month.between(4, 10).astype(int)
    df["is_withdrawal_season"] = dates.dt.month.isin([11, 12, 1, 2, 3]).astype(int)
    df["is_shoulder_month"] = dates.dt.month.isin([4, 5, 9, 10]).astype(int)
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

    for window in (4, 8):
        df[f"hdd_rolling_{window}wk"] = _rolling_mean_within_region(
            df, "hdd", window
        )
        df[f"cdd_rolling_{window}wk"] = _rolling_mean_within_region(
            df, "cdd", window
        )

    df["hdd_vs_4wk_avg"] = df["hdd"] - df["hdd_rolling_4wk"]
    df["cdd_vs_4wk_avg"] = df["cdd"] - df["cdd_rolling_4wk"]

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

    for window in (4, 8):
        df[f"weekly_change_rolling_{window}wk"] = _rolling_mean_within_region(
            df, TARGET_COLUMN, window
        )

    df["storage_bcf_lag52"] = _lag_within_region(df, "storage_bcf", yoy_lag)
    df["storage_vs_last_year"] = df["storage_bcf"] - df["storage_bcf_lag52"]
    df["storage_5yr_avg"] = _same_week_history_mean(df, "storage_bcf", years=5)
    df["storage_vs_5yr_avg"] = df["storage_bcf"] - df["storage_5yr_avg"]
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
