"""Point-in-time ensemble weather forecast ingestion and aggregation."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import requests

from gas_forecast.data.weather_features import assign_storage_week_end
from gas_forecast.data.weather_scenarios import select_weather_scenario_as_of


OPEN_METEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

WEATHER_FORECAST_ARCHIVE_COLUMNS = (
    "provider",
    "model",
    "model_run",
    "ensemble_member",
    "issued_at",
    "valid_start",
    "valid_end",
    "state",
    "duoarea",
    "temperature_f",
    "hdd",
    "cdd",
    "coverage",
)

_IDENTITY_COLUMNS = (
    "provider",
    "model",
    "model_run",
    "ensemble_member",
    "issued_at",
    "valid_start",
    "state",
    "duoarea",
)


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("A valid UTC timestamp is required.")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def validate_weather_forecast_archive(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate member-level daily regional weather forecast vintages."""
    missing = sorted(set(WEATHER_FORECAST_ARCHIVE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Weather forecast archive missing columns: {missing}")

    data = frame.copy()
    for column in ("issued_at", "valid_start", "valid_end"):
        data[column] = pd.to_datetime(data[column], errors="coerce", utc=True)
    if data[["issued_at", "valid_start", "valid_end"]].isna().any().any():
        raise ValueError("Weather forecast archive contains invalid timestamps.")
    if (data["valid_end"] <= data["valid_start"]).any():
        raise ValueError("valid_end must be later than valid_start.")
    if (data["issued_at"] >= data["valid_end"]).any():
        raise ValueError("Weather forecasts cannot be issued after their valid period.")

    text_columns = (
        "provider",
        "model",
        "model_run",
        "ensemble_member",
        "state",
        "duoarea",
    )
    for column in text_columns:
        if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Weather forecast archive requires non-empty {column!r}.")
        data[column] = data[column].astype(str)

    for column in ("temperature_f", "hdd", "cdd", "coverage"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            raise ValueError(f"Weather forecast archive contains invalid {column!r}.")
    if not data["coverage"].between(0.0, 1.0).all():
        raise ValueError("Weather forecast coverage must be between zero and one.")
    if data.duplicated(subset=list(_IDENTITY_COLUMNS)).any():
        raise ValueError("Weather forecast archive contains duplicate member vintages.")

    return data.sort_values(list(_IDENTITY_COLUMNS)).reset_index(drop=True)


def _member_name(variable: str) -> str:
    prefix = "temperature_2m_mean"
    suffix = variable.removeprefix(prefix).lstrip("_")
    return suffix or "control"


def _open_meteo_results(
    locations: pd.DataFrame,
    payload: list[dict[str, Any]] | dict[str, Any],
) -> list[dict[str, Any]]:
    required = {"STNAME", "duoarea"}
    missing = required - set(locations.columns)
    if missing:
        raise ValueError(f"Forecast locations missing columns: {sorted(missing)}")
    results = payload if isinstance(payload, list) else [payload]
    if len(results) != len(locations):
        raise ValueError("Open-Meteo results do not match requested locations.")
    return results


def _archive_member_frame(
    location: pd.Series,
    *,
    valid_start: pd.DatetimeIndex,
    temperatures: pd.Series,
    issued_at: pd.Timestamp | pd.DatetimeIndex,
    provider: str,
    model: str,
    member: str,
    coverage: float | pd.Series,
) -> pd.DataFrame:
    model_run = (
        issued_at.isoformat()
        if isinstance(issued_at, pd.Timestamp)
        else issued_at.map(lambda value: value.isoformat())
    )
    frame = pd.DataFrame(
        {
            "provider": provider,
            "model": model,
            "model_run": model_run,
            "ensemble_member": member,
            "issued_at": issued_at,
            "valid_start": valid_start,
            "valid_end": valid_start + pd.Timedelta(days=1),
            "state": str(location["STNAME"]),
            "duoarea": str(location["duoarea"]),
            "temperature_f": temperatures.to_numpy(),
            "coverage": coverage,
        }
    )
    frame["hdd"] = np.maximum(0.0, 65.0 - frame["temperature_f"])
    frame["cdd"] = np.maximum(0.0, frame["temperature_f"] - 65.0)
    return frame


def parse_open_meteo_ensemble_response(
    locations: pd.DataFrame,
    payload: list[dict[str, Any]] | dict[str, Any],
    *,
    issued_at: object,
    provider: str = "open-meteo",
    model: str = "gfs_seamless",
) -> pd.DataFrame:
    """Parse Open-Meteo daily ensemble temperatures into archive rows."""
    results = _open_meteo_results(locations, payload)
    issued = _utc(issued_at)
    rows: list[pd.DataFrame] = []
    for (_, location), result in zip(locations.iterrows(), results, strict=True):
        daily = result.get("daily")
        if not isinstance(daily, dict) or "time" not in daily:
            raise ValueError(f"No daily ensemble data returned for {location['STNAME']}")
        temperature_vars = sorted(
            key for key in daily if key.startswith("temperature_2m_mean")
        )
        if not temperature_vars:
            raise ValueError("Open-Meteo response has no daily mean temperature members.")

        valid_start = pd.to_datetime(daily["time"], errors="coerce", utc=True)
        if valid_start.isna().any():
            raise ValueError("Open-Meteo response contains invalid forecast dates.")
        for variable in temperature_vars:
            temperatures = pd.to_numeric(pd.Series(daily[variable]), errors="coerce")
            if len(temperatures) != len(valid_start) or temperatures.isna().any():
                raise ValueError(f"Open-Meteo returned invalid values for {variable!r}.")
            valid_end = valid_start + pd.Timedelta(days=1)
            seconds_available = (
                pd.Series(valid_end)
                - pd.Series(valid_start).where(
                    valid_start > issued,
                    issued,
                )
            ).dt.total_seconds()
            rows.append(
                _archive_member_frame(
                    location,
                    valid_start=valid_start,
                    temperatures=temperatures,
                    issued_at=issued,
                    provider=provider,
                    model=model,
                    member=_member_name(variable),
                    coverage=(seconds_available / 86_400.0).clip(0.0, 1.0),
                )
            )

    return validate_weather_forecast_archive(pd.concat(rows, ignore_index=True))


def fetch_open_meteo_gefs_ensemble(
    locations: pd.DataFrame,
    *,
    issued_at: object,
    forecast_days: int = 16,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """Fetch the free live NOAA GEFS ensemble through Open-Meteo."""
    required = {"LATITUDE", "LONGITUDE", "STNAME", "duoarea"}
    missing = required - set(locations.columns)
    if missing:
        raise ValueError(f"Forecast locations missing columns: {sorted(missing)}")
    if not 1 <= forecast_days <= 35:
        raise ValueError("forecast_days must be between 1 and 35.")

    response = requests.get(
        OPEN_METEO_ENSEMBLE_URL,
        params={
            "latitude": ",".join(locations["LATITUDE"].astype(str)),
            "longitude": ",".join(locations["LONGITUDE"].astype(str)),
            "models": "gfs_seamless",
            "daily": "temperature_2m_mean",
            "temperature_unit": "fahrenheit",
            "timezone": "GMT",
            "forecast_days": forecast_days,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_open_meteo_ensemble_response(
        locations,
        response.json(),
        issued_at=issued_at,
    )


def parse_open_meteo_previous_runs_response(
    locations: pd.DataFrame,
    payload: list[dict[str, Any]] | dict[str, Any],
    *,
    provider: str = "open-meteo",
    model: str = "gfs_seamless",
) -> pd.DataFrame:
    """Parse fixed-lead GFS temperatures and reconstruct their issue dates."""
    results = _open_meteo_results(locations, payload)
    rows: list[pd.DataFrame] = []
    pattern = re.compile(r"^temperature_2m_mean_previous_day(\d+)$")
    for (_, location), result in zip(locations.iterrows(), results, strict=True):
        daily = result.get("daily")
        if not isinstance(daily, dict) or "time" not in daily:
            raise ValueError(f"No previous-run data returned for {location['STNAME']}")
        valid_start = pd.to_datetime(daily["time"], errors="coerce", utc=True)
        variables = [(key, pattern.match(key)) for key in daily]
        variables = [(key, match) for key, match in variables if match is not None]
        if not variables:
            raise ValueError("Open-Meteo response has no fixed-lead temperature fields.")
        for variable, match in variables:
            lead_days = int(match.group(1))
            temperatures = pd.to_numeric(pd.Series(daily[variable]), errors="coerce")
            if len(temperatures) != len(valid_start):
                raise ValueError(f"Open-Meteo returned invalid values for {variable!r}.")
            available = temperatures.notna()
            if not available.any():
                continue
            temperatures = temperatures.loc[available].reset_index(drop=True)
            member_valid_start = valid_start[available]
            issued = member_valid_start - pd.Timedelta(days=lead_days)
            rows.append(
                _archive_member_frame(
                    location,
                    valid_start=member_valid_start,
                    temperatures=temperatures,
                    issued_at=issued,
                    provider=provider,
                    model=model,
                    member="deterministic",
                    coverage=1.0,
                )
            )
    return validate_weather_forecast_archive(pd.concat(rows, ignore_index=True))


def fetch_open_meteo_previous_runs(
    locations: pd.DataFrame,
    start_date: str,
    end_date: str,
    *,
    max_lead_days: int = 7,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """Fetch archived GFS temperatures at fixed zero-to-seven-day leads."""
    required = {"LATITUDE", "LONGITUDE", "STNAME", "duoarea"}
    missing = required - set(locations.columns)
    if missing:
        raise ValueError(f"Forecast locations missing columns: {sorted(missing)}")
    if not 0 <= max_lead_days <= 7:
        raise ValueError("max_lead_days must be between zero and seven.")
    variables = ",".join(
        f"temperature_2m_mean_previous_day{lead}"
        for lead in range(max_lead_days + 1)
    )
    response = requests.get(
        OPEN_METEO_PREVIOUS_RUNS_URL,
        params={
            "latitude": ",".join(locations["LATITUDE"].astype(str)),
            "longitude": ",".join(locations["LONGITUDE"].astype(str)),
            "start_date": start_date,
            "end_date": end_date,
            "models": "gfs_seamless",
            "daily": variables,
            "temperature_unit": "fahrenheit",
            "timezone": "GMT",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_open_meteo_previous_runs_response(locations, response.json())


def select_state_weights_as_of(
    history: pd.DataFrame,
    as_of: object,
    *,
    weight_col: str,
) -> pd.DataFrame:
    """Select and normalize the latest state gas-load weights known at an origin."""
    required = {"state", "duoarea", "available_at", weight_col}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"State weight history missing columns: {sorted(missing)}")
    data = history.copy()
    data["available_at"] = pd.to_datetime(data["available_at"], errors="coerce", utc=True)
    data[weight_col] = pd.to_numeric(data[weight_col], errors="coerce")
    if data[["available_at", weight_col]].isna().any().any():
        raise ValueError("State weight history contains invalid values.")
    data = data.loc[data["available_at"] <= _utc(as_of)].copy()
    selected = data.sort_values("available_at").drop_duplicates(
        subset=["duoarea", "state"], keep="last"
    )
    if selected.empty:
        raise ValueError("No state weights were available at the requested origin.")
    if (selected[weight_col] < 0).any():
        raise ValueError("State weights cannot be negative.")
    totals = selected.groupby("duoarea")[weight_col].transform("sum")
    if (totals <= 0).any():
        raise ValueError("Each region must have positive total state weight.")
    selected["weather_weight"] = selected[weight_col] / totals
    return selected[["state", "duoarea", "available_at", "weather_weight"]].reset_index(
        drop=True
    )


def aggregate_state_forecasts(
    archive: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate state forecasts into gas-load- or population-weighted regions."""
    data = validate_weather_forecast_archive(archive)
    required_weights = {"state", "duoarea", "weather_weight"}
    missing = required_weights - set(weights.columns)
    if missing:
        raise ValueError(f"Weather weights missing columns: {sorted(missing)}")
    totals = weights.groupby("duoarea")["weather_weight"].sum()
    if not np.allclose(totals.to_numpy(dtype=float), 1.0):
        raise ValueError("Weather weights must sum to one within each region.")
    merged = data.merge(
        weights[list(required_weights)],
        on=["state", "duoarea"],
        how="left",
        validate="many_to_one",
    )
    if merged["weather_weight"].isna().any():
        missing_states = sorted(merged.loc[merged["weather_weight"].isna(), "state"].unique())
        raise ValueError(f"Missing weather weights for states: {missing_states}")

    group_cols = [
        "provider",
        "model",
        "model_run",
        "ensemble_member",
        "issued_at",
        "valid_start",
        "valid_end",
        "duoarea",
    ]
    for column in ("temperature_f", "hdd", "cdd", "coverage"):
        merged[f"weighted_{column}"] = merged[column] * merged["weather_weight"]
    regional = (
        merged.groupby(group_cols, as_index=False)
        .agg(
            temperature_f=("weighted_temperature_f", "sum"),
            hdd=("weighted_hdd", "sum"),
            cdd=("weighted_cdd", "sum"),
            coverage=("weighted_coverage", "sum"),
        )
    )
    regional["state"] = "REGIONAL"
    return validate_weather_forecast_archive(regional)


def aggregate_state_forecasts_with_weight_history(
    archive: pd.DataFrame,
    weight_history: pd.DataFrame,
    *,
    weight_col: str,
) -> pd.DataFrame:
    """Aggregate each forecast run with only the state weights known then."""
    data = validate_weather_forecast_archive(archive)
    outputs: list[pd.DataFrame] = []
    for issued_at, vintage in data.groupby("issued_at", sort=True):
        weights = select_state_weights_as_of(
            weight_history,
            issued_at,
            weight_col=weight_col,
        )
        outputs.append(aggregate_state_forecasts(vintage, weights))
    return validate_weather_forecast_archive(pd.concat(outputs, ignore_index=True))


def aggregate_forecasts_to_weekly_scenarios(
    archive: pd.DataFrame,
    *,
    require_complete_weeks: bool = True,
) -> pd.DataFrame:
    """Aggregate daily member forecasts into weekly ensemble scenario features."""
    data = validate_weather_forecast_archive(archive)
    data["date"] = assign_storage_week_end(data["valid_start"].dt.tz_localize(None))
    member_keys = [
        "provider",
        "model",
        "model_run",
        "ensemble_member",
        "issued_at",
        "duoarea",
        "date",
    ]
    member_weekly = (
        data.groupby(member_keys, as_index=False)
        .agg(
            temperature_f=("temperature_f", "mean"),
            hdd=("hdd", "sum"),
            cdd=("cdd", "sum"),
            weather_days=("valid_start", "nunique"),
            coverage=("coverage", "mean"),
        )
    )
    if require_complete_weeks:
        member_weekly = member_weekly.loc[
            (member_weekly["weather_days"] == 7)
            & (member_weekly["coverage"] >= 0.999)
        ].copy()
    if member_weekly.empty:
        return pd.DataFrame()

    scenario_keys = ["provider", "model", "model_run", "issued_at", "duoarea", "date"]
    scenarios = member_weekly.groupby(scenario_keys, as_index=False).agg(
        temperature_f=("temperature_f", "mean"),
        hdd=("hdd", "mean"),
        cdd=("cdd", "mean"),
        weather_days=("weather_days", "min"),
        coverage=("coverage", "mean"),
        ensemble_members=("ensemble_member", "nunique"),
        hdd_p10=("hdd", lambda values: values.quantile(0.10)),
        hdd_p90=("hdd", lambda values: values.quantile(0.90)),
        cdd_p10=("cdd", lambda values: values.quantile(0.10)),
        cdd_p90=("cdd", lambda values: values.quantile(0.90)),
    )
    scenarios["hdd_spread"] = scenarios["hdd_p90"] - scenarios["hdd_p10"]
    scenarios["cdd_spread"] = scenarios["cdd_p90"] - scenarios["cdd_p10"]
    return scenarios.sort_values(["duoarea", "date", "issued_at"]).reset_index(drop=True)


def build_asof_weather_features(
    origins: pd.DataFrame,
    scenarios: pd.DataFrame,
    *,
    horizon_weeks: int = 4,
) -> pd.DataFrame:
    """Build one leak-free ensemble-weather feature row per origin and horizon."""
    required = {"date", "duoarea"}
    missing = required - set(origins.columns)
    if missing:
        raise ValueError(f"Forecast origins missing columns: {sorted(missing)}")
    if horizon_weeks < 1:
        raise ValueError("horizon_weeks must be at least 1.")

    rows: list[pd.DataFrame] = []
    for origin in origins[list(required)].itertuples(index=False):
        origin_date = pd.Timestamp(origin.date).normalize()
        target_dates = pd.date_range(origin_date, periods=horizon_weeks, freq="W-FRI")
        selected = select_weather_scenario_as_of(
            scenarios,
            origin_date,
            region=str(origin.duoarea),
            target_dates=target_dates,
        )
        selected.insert(0, "origin_date", origin_date)
        selected["horizon"] = np.arange(1, len(selected) + 1)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)
