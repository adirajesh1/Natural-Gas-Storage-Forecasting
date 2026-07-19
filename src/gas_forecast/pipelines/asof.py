"""Pipelines that materialize point-in-time forecasting inputs from vintages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gas_forecast.data.balance_asof import build_asof_balance_features
from gas_forecast.data.export import save_versioned_parquet
from gas_forecast.data.paths import DEFAULT_PROCESSED_DIR, latest_processed_path
from gas_forecast.data.regions import region_slug
from gas_forecast.data.weather_scenarios import select_weather_scenario_as_of
from gas_forecast.data.weather_api import _date_periods
from gas_forecast.data.weather_forecasts import (
    aggregate_forecasts_to_weekly_scenarios,
    aggregate_state_forecasts,
    aggregate_state_forecasts_with_weight_history,
    fetch_open_meteo_gefs_ensemble,
    fetch_open_meteo_previous_runs,
)
from gas_forecast.data.weather_locations import (
    load_census_state_locations,
    select_weather_locations,
)
from gas_forecast.pipelines.data import PipelineOutputs


def _load_origins(
    region: str,
    *,
    processed_dir: Path,
    origins_path: str | Path | None,
) -> pd.DataFrame:
    """Load and scope model-table forecast origins to one storage region."""
    path = (
        Path(origins_path)
        if origins_path is not None
        else latest_processed_path(region, "weekly_model_features", processed_dir)
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing balance feature origins: {path}")

    origins = pd.read_parquet(path)
    if "duoarea" not in origins.columns:
        raise ValueError(f"Balance feature origins in {path} are missing 'duoarea'.")
    scoped = origins.loc[origins["duoarea"] == region].copy()
    if scoped.empty:
        raise ValueError(f"No origins for region {region!r} were found in {path}.")
    return scoped


def run_weather_scenario_pipeline(
    region: str,
    *,
    scenarios_path: str | Path,
    as_of: str | pd.Timestamp,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> Path:
    """Select and save the weather forecast version available at one origin."""
    scenarios_path = Path(scenarios_path)
    if not scenarios_path.exists():
        raise FileNotFoundError(f"Missing weather scenario archive: {scenarios_path}")

    selected = select_weather_scenario_as_of(
        pd.read_parquet(scenarios_path),
        as_of,
        region=region,
    )
    if selected.empty:
        raise ValueError(
            f"No weather scenarios for {region!r} were available at {as_of!r}."
        )

    return save_versioned_parquet(
        selected,
        processed_dir,
        f"{region_slug(region)}_weekly_weather_scenario",
        save_latest=True,
    )


def run_asof_balance_pipeline(
    region: str,
    *,
    vintages_path: str | Path,
    origins_path: str | Path | None = None,
    as_of_col: str | None = "date",
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> Path:
    """Build and save point-in-time balance lag features for a storage region.

    The input vintages must retain every historical version and its real
    ``available_at`` timestamp. The retrospective balance artifact produced by
    ``gas-data balance`` is not a substitute for that archive.
    """
    vintages_path = Path(vintages_path)
    if not vintages_path.exists():
        raise FileNotFoundError(f"Missing balance vintage archive: {vintages_path}")

    processed_dir = Path(processed_dir)
    origins = _load_origins(
        region,
        processed_dir=processed_dir,
        origins_path=origins_path,
    )
    vintages = pd.read_parquet(vintages_path)
    if "duoarea" in vintages.columns and not vintages["duoarea"].eq(region).any():
        raise ValueError(
            f"No balance vintages for region {region!r} were found in {vintages_path}."
        )
    features = build_asof_balance_features(
        origins,
        vintages,
        as_of_col=as_of_col,
    )
    return save_versioned_parquet(
        features,
        processed_dir,
        f"{region_slug(region)}_weekly_asof_balance_features",
        save_latest=True,
    )


def _forecast_locations(region: str) -> pd.DataFrame:
    locations = select_weather_locations(load_census_state_locations(), region)
    locations["duoarea"] = region
    return locations


def _aggregate_forecast_members(
    archive: pd.DataFrame,
    locations: pd.DataFrame,
    *,
    weight_history_path: str | Path | None,
    weight_col: str,
) -> pd.DataFrame:
    if weight_history_path is not None:
        history = pd.read_parquet(weight_history_path)
        return aggregate_state_forecasts_with_weight_history(
            archive,
            history,
            weight_col=weight_col,
        )
    weights = locations[["STNAME", "duoarea", "WEIGHT"]].rename(
        columns={"STNAME": "state", "WEIGHT": "weather_weight"}
    )
    return aggregate_state_forecasts(archive, weights)


def _save_forecast_outputs(
    region: str,
    state_archive: pd.DataFrame,
    regional_archive: pd.DataFrame,
    *,
    processed_dir: str | Path,
) -> PipelineOutputs:
    slug = region_slug(region)
    state_path = save_versioned_parquet(
        state_archive,
        processed_dir,
        f"{slug}_state_weather_forecast_archive",
        save_latest=True,
    )
    regional_path = save_versioned_parquet(
        regional_archive,
        processed_dir,
        f"{slug}_regional_weather_forecast_archive",
        save_latest=True,
    )
    weekly = aggregate_forecasts_to_weekly_scenarios(regional_archive)
    if weekly.empty:
        raise ValueError("Forecast archive did not contain a complete EIA storage week.")
    weekly_path = save_versioned_parquet(
        weekly,
        processed_dir,
        f"{slug}_weekly_weather_forecast_scenarios",
        save_latest=True,
    )
    return PipelineOutputs(
        region=region,
        paths={
            "state_forecast_archive": state_path,
            "regional_forecast_archive": regional_path,
            "weekly_weather_scenarios": weekly_path,
        },
        frames={
            "regional_forecast_archive": regional_archive,
            "weekly_weather_scenarios": weekly,
        },
    )


def run_live_weather_forecast_pipeline(
    region: str,
    *,
    issued_at: str | pd.Timestamp,
    forecast_days: int = 16,
    weight_history_path: str | Path | None = None,
    weight_col: str = "gas_load",
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> PipelineOutputs:
    """Fetch, weight, aggregate, and archive the current free GEFS ensemble."""
    locations = _forecast_locations(region)
    state_archive = fetch_open_meteo_gefs_ensemble(
        locations,
        issued_at=issued_at,
        forecast_days=forecast_days,
    )
    regional = _aggregate_forecast_members(
        state_archive,
        locations,
        weight_history_path=weight_history_path,
        weight_col=weight_col,
    )
    return _save_forecast_outputs(
        region,
        state_archive,
        regional,
        processed_dir=processed_dir,
    )


def run_historical_weather_forecast_pipeline(
    region: str,
    *,
    start_date: str,
    end_date: str,
    max_lead_days: int = 7,
    weight_history_path: str | Path | None = None,
    weight_col: str = "gas_load",
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> PipelineOutputs:
    """Materialize fixed-lead historical GFS runs for honest week-one tests."""
    locations = _forecast_locations(region)
    state_archive = pd.concat(
        [
            fetch_open_meteo_previous_runs(
                locations,
                period_start,
                period_end,
                max_lead_days=max_lead_days,
            )
            for period_start, period_end in _date_periods(start_date, end_date)
        ],
        ignore_index=True,
    )
    regional = _aggregate_forecast_members(
        state_archive,
        locations,
        weight_history_path=weight_history_path,
        weight_col=weight_col,
    )
    return _save_forecast_outputs(
        region,
        state_archive,
        regional,
        processed_dir=processed_dir,
    )
