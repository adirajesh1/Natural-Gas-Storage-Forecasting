"""Selection and validation of archived weekly weather forecast scenarios."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


WEATHER_SCENARIO_COLUMNS = (
    "date",
    "duoarea",
    "issued_at",
    "temperature_f",
    "hdd",
    "cdd",
    "weather_days",
)

_WEATHER_VALUE_COLUMNS = (
    "temperature_f",
    "hdd",
    "cdd",
    "weather_days",
)


def _as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """Return one timestamp in UTC for availability comparisons."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("A valid as-of timestamp is required.")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _normalize_dates(values: Iterable[object]) -> pd.DatetimeIndex:
    """Convert scenario target dates to normalized calendar days."""
    dates = pd.to_datetime(list(values), errors="coerce")
    if dates.isna().any():
        raise ValueError("Scenario target dates must be valid timestamps.")
    return pd.DatetimeIndex(dates).normalize()


def validate_weather_scenarios(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a versioned regional weekly weather scenario table.

    Each row represents a forecast for one EIA week-ending date. Multiple
    forecast versions are allowed, but their availability must be recorded in
    ``issued_at`` so callers can select the version known at an origin.
    """
    missing = sorted(set(WEATHER_SCENARIO_COLUMNS) - set(scenarios.columns))
    if missing:
        raise ValueError(f"Weather scenarios missing required columns: {missing}")

    data = scenarios.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["issued_at"] = pd.to_datetime(
        data["issued_at"],
        errors="coerce",
        utc=True,
    )
    if data["date"].isna().any():
        raise ValueError("Weather scenarios contain invalid target dates.")
    if data["issued_at"].isna().any():
        raise ValueError("Weather scenarios contain invalid issued_at timestamps.")
    if data["duoarea"].isna().any() or data["duoarea"].astype(str).eq("").any():
        raise ValueError("Weather scenarios require a non-empty duoarea on every row.")

    for column in _WEATHER_VALUE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            raise ValueError(f"Weather scenarios contain missing {column!r} values.")

    if (data["weather_days"] <= 0).any():
        raise ValueError("Weather scenario weather_days must be positive.")
    version_columns = [
        column
        for column in ("provider", "model", "model_run")
        if column in data.columns
    ]
    identity_columns = ["date", "duoarea", "issued_at", *version_columns]
    if data.duplicated(subset=identity_columns).any():
        raise ValueError(
            "Weather scenarios contain duplicate forecast-version rows."
        )

    return data.sort_values(["duoarea", "date", "issued_at", *version_columns]).reset_index(
        drop=True
    )


def select_weather_scenario_as_of(
    scenarios: pd.DataFrame,
    as_of: str | pd.Timestamp,
    *,
    region: str,
    target_dates: Iterable[object] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """Select the latest scenario version available at a forecast origin.

    ``issued_at`` is compared in UTC. When ``target_dates`` is supplied, every
    requested date must have a selected scenario, which prevents a recursive
    forecast from silently falling back to realized weather or a seasonal norm.
    """
    data = validate_weather_scenarios(scenarios)
    available = data.loc[
        (data["duoarea"] == region)
        & (data["issued_at"] <= _as_utc_timestamp(as_of))
    ].copy()
    if provider is not None:
        if "provider" not in available.columns:
            raise ValueError("Weather scenarios do not contain provider metadata.")
        available = available.loc[available["provider"] == provider].copy()
    if model is not None:
        if "model" not in available.columns:
            raise ValueError("Weather scenarios do not contain model metadata.")
        available = available.loc[available["model"] == model].copy()

    version_columns = [
        column for column in ("provider", "model") if column in available.columns
    ]
    if version_columns and provider is None and model is None and not available.empty:
        latest = available.loc[
            available["issued_at"]
            == available.groupby(["date", "duoarea"])["issued_at"].transform("max")
        ]
        if latest.groupby(["date", "duoarea"])[version_columns].size().gt(1).any():
            raise ValueError(
                "Multiple weather models are available; select provider= or model=."
            )
    selected = available.sort_values("issued_at").drop_duplicates(
        subset=["date", "duoarea"],
        keep="last",
    )

    if target_dates is not None:
        expected_dates = _normalize_dates(target_dates)
        selected = selected.loc[selected["date"].isin(expected_dates)].copy()
        missing_dates = expected_dates.difference(pd.DatetimeIndex(selected["date"]))
        if len(missing_dates):
            formatted = ", ".join(date.strftime("%Y-%m-%d") for date in missing_dates)
            raise ValueError(
                "No weather scenario is available at the requested origin for "
                f"{region} on: {formatted}"
            )
        selected = selected.set_index("date").reindex(expected_dates)
        selected.index.name = "date"
        selected = selected.reset_index()

    return selected.sort_values("date").reset_index(drop=True)
