from __future__ import annotations

import numpy as np
import pandas as pd

_LOCATION_COLUMNS = {
    "STATEFP",
    "STNAME",
    "POPULATION",
    "LATITUDE",
    "LONGITUDE",
    "WEIGHT",
}


def validate_weekly_weather(weather: pd.DataFrame) -> None:
    """Validate weekly weather aligned to storage reporting weeks."""
    required = {
        "date",
        "temperature_f",
        "hdd",
        "cdd",
        "weather_days",
        "duoarea",
    }
    missing_cols = required - set(weather.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if weather["date"].duplicated().any():
        raise ValueError("Found duplicate weekly weather dates.")

    if not weather["date"].dt.weekday.eq(4).all():
        raise ValueError("Weekly weather dates must fall on Friday.")

    if (weather["weather_days"] != 7).any():
        n_bad = (weather["weather_days"] != 7).sum()
        raise ValueError(f"Found {n_bad} weeks without exactly 7 weather days.")


def validate_weather_locations(
    locations: pd.DataFrame,
    *,
    expected_states: frozenset[str] | set[str] | None = None,
) -> None:
    """Validate a locations table used for weather downloads."""
    missing_cols = _LOCATION_COLUMNS - set(locations.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if locations["STNAME"].duplicated().any():
        raise ValueError("Duplicate state names in locations.")

    weight_sum = locations["WEIGHT"].sum()
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(f"Location weights must sum to 1, got {weight_sum}.")

    if expected_states is not None:
        actual_states = set(locations["STNAME"].astype(str))
        if actual_states != set(expected_states):
            missing = sorted(set(expected_states) - actual_states)
            extra = sorted(actual_states - set(expected_states))
            raise ValueError(
                f"Location states do not match region. "
                f"Missing: {missing}. Extra: {extra}."
            )


def validate_state_daily_weather(
    weather: pd.DataFrame,
    *,
    expected_states: frozenset[str] | set[str] | None = None,
) -> None:
    """Validate state-level daily weather before regional aggregation."""
    required = {
        "date",
        "temperature_f",
        "state",
        "population",
        "population_weight",
    }
    missing_cols = required - set(weather.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if weather["temperature_f"].isna().any():
        raise ValueError("Found missing values in 'temperature_f'.")

    if weather["date"].isna().any():
        raise ValueError("Found missing values in 'date'.")

    if weather.duplicated(subset=["date", "state"]).any():
        raise ValueError("Found duplicate date/state rows.")

    if expected_states is not None:
        actual_states = set(weather["state"].astype(str))
        if actual_states != set(expected_states):
            missing = sorted(set(expected_states) - actual_states)
            extra = sorted(actual_states - set(expected_states))
            raise ValueError(
                f"Weather states do not match region. "
                f"Missing: {missing}. Extra: {extra}."
            )

    expected_count = (
        len(expected_states)
        if expected_states is not None
        else weather["state"].nunique()
    )
    states_per_date = weather.groupby("date")["state"].nunique()
    if states_per_date.ne(expected_count).any():
        raise ValueError("Daily weather has incomplete state coverage for some dates.")
