from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_hdd_cdd(temperatures: pd.DataFrame) -> pd.DataFrame:
    """Calculate heating and cooling degree days from daily temperatures."""
    df = temperatures.copy()
    df["hdd"] = np.maximum(0, 65 - df["temperature_f"])
    df["cdd"] = np.maximum(0, df["temperature_f"] - 65)
    return df


def aggregate_population_weighted_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate state-level daily weather to a population-weighted regional index."""
    required = {"date", "temperature_f", "hdd", "cdd", "population_weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    weighted = df.assign(
        w_temp=df["temperature_f"] * df["population_weight"],
        w_hdd=df["hdd"] * df["population_weight"],
        w_cdd=df["cdd"] * df["population_weight"],
    )
    return (
        weighted.groupby("date", as_index=False)
        .agg(
            temperature_f=("w_temp", "sum"),
            hdd=("w_hdd", "sum"),
            cdd=("w_cdd", "sum"),
        )
        .reset_index(drop=True)
    )


def prepare_weather_model_data(
    weather: pd.DataFrame,
    duoarea: str,
) -> pd.DataFrame:
    """Select and format regional daily weather for modeling."""
    df = weather.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["duoarea"] = duoarea
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    return df[
        [
            "date",
            "temperature_f",
            "hdd",
            "cdd",
            "year",
            "month",
            "day_of_year",
            "duoarea",
        ]
    ].reset_index(drop=True)


def assign_storage_week_end(dates: pd.Series) -> pd.Series:
    """
    Map each calendar day to the EIA storage week-ending Friday.

    Each week spans Saturday through Friday (inclusive), matching the
    convention used for weekly natural gas storage reporting.
    """
    dates = pd.to_datetime(dates)
    if isinstance(dates, pd.DatetimeIndex):
        weekdays = dates.weekday
    else:
        weekdays = dates.dt.weekday
    days_to_friday = (4 - weekdays) % 7
    return dates + pd.to_timedelta(days_to_friday, unit="D")


def aggregate_weather_to_storage_weeks(
    daily: pd.DataFrame,
    *,
    drop_incomplete: bool = True,
    expected_days: int = 7,
) -> pd.DataFrame:
    """Aggregate regional daily weather into EIA storage weeks (Sat-Fri)."""
    required = {"date", "temperature_f", "hdd", "cdd"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["storage_week_end"] = assign_storage_week_end(df["date"])

    weekly = (
        df.groupby("storage_week_end", as_index=False)
        .agg(
            temperature_f=("temperature_f", "mean"),
            hdd=("hdd", "sum"),
            cdd=("cdd", "sum"),
            weather_days=("date", "count"),
        )
        .rename(columns={"storage_week_end": "date"})
    )

    if drop_incomplete:
        weekly = weekly.loc[weekly["weather_days"] == expected_days].copy()

    return weekly.reset_index(drop=True)


def prepare_weekly_weather_model_data(
    weather: pd.DataFrame,
    duoarea: str,
) -> pd.DataFrame:
    """Select and format regional weekly weather aligned to storage weeks."""
    df = weather.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["duoarea"] = duoarea
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    return df[
        [
            "date",
            "temperature_f",
            "hdd",
            "cdd",
            "weather_days",
            "year",
            "month",
            "week_of_year",
            "duoarea",
        ]
    ].reset_index(drop=True)
