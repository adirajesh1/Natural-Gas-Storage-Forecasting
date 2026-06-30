from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from gas_forecast.data.cache import DEFAULT_CACHE_DIR
from gas_forecast.data.export import save_versioned_parquet
from gas_forecast.data.features import (
    build_weekly_model_features,
    validate_weekly_model_features,
)
from gas_forecast.data.paths import (
    DEFAULT_LEGACY_WEATHER_CACHE_DIR,
    DEFAULT_PROCESSED_DIR,
    latest_processed_path,
    weather_cache_dir,
)
from gas_forecast.data.regions import region_slug, region_states, supported_storage_regions
from gas_forecast.data.storage_api import fetch_weekly_storage_incremental
from gas_forecast.data.storage_transforms import (
    calculate_weekly_storage_change,
    clean_weekly_storage,
    prepare_storage_model_data,
    select_region,
)
from gas_forecast.data.weather_cache import (
    fetch_all_state_temperatures,
    migrate_weather_chunk_cache,
)
from gas_forecast.data.weather_features import (
    aggregate_population_weighted_weather,
    aggregate_weather_to_storage_weeks,
    calculate_hdd_cdd,
    prepare_weather_model_data,
    prepare_weekly_weather_model_data,
)
from gas_forecast.data.weather_locations import (
    load_census_state_locations,
    select_weather_locations,
)
from gas_forecast.data.weather_validation import (
    validate_state_daily_weather,
    validate_weekly_weather,
)

PipelineStage = Literal["storage", "weather", "features"]
ALL_STAGES: tuple[PipelineStage, ...] = ("storage", "weather", "features")


@dataclass
class PipelineOutputs:
    """Paths and in-memory frames produced by a data pipeline run."""

    region: str
    paths: dict[str, Path] = field(default_factory=dict)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)


def _resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key

    try:
        from dotenv import load_dotenv

        load_dotenv("local.env")
    except ImportError:
        pass

    resolved = os.getenv("EIA_API_KEY")
    if not resolved:
        raise ValueError(
            "EIA API key required. Pass api_key= or set EIA_API_KEY in the "
            "environment (optionally via local.env with python-dotenv installed)."
        )
    return resolved


def _load_storage_for_region(processed_dir: Path, region: str) -> pd.DataFrame:
    path = latest_processed_path(region, "weekly_storage", processed_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed storage for {region}: {path}. "
            "Run the storage pipeline first."
        )
    storage = pd.read_parquet(path)
    if not (storage["duoarea"] == region).all():
        raise ValueError(f"Storage file {path} does not match region {region!r}.")
    return storage


def run_storage_pipeline(
    region: str,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    api_key: str | None = None,
    min_start_date: str = "2010-01-01",
    revision_weeks: int = 8,
) -> PipelineOutputs:
    """Download, clean, and export weekly storage for one EIA region."""
    cache_dir = Path(cache_dir)
    processed_dir = Path(processed_dir)
    slug = region_slug(region)
    key = _resolve_api_key(api_key)

    storage_raw = fetch_weekly_storage_incremental(
        key,
        cache_dir=cache_dir,
        revision_weeks=revision_weeks,
    )
    storage = clean_weekly_storage(storage_raw, start_date=min_start_date)
    region_storage = select_region(storage, region)
    region_storage = calculate_weekly_storage_change(region_storage)
    region_weekly_storage = prepare_storage_model_data(region_storage)

    versioned_path = save_versioned_parquet(
        region_weekly_storage,
        processed_dir,
        f"{slug}_weekly_storage",
        save_latest=True,
    )

    return PipelineOutputs(
        region=region,
        paths={
            "weekly_storage": versioned_path,
            "weekly_storage_latest": latest_processed_path(
                region, "weekly_storage", processed_dir
            ),
        },
        frames={"weekly_storage": region_weekly_storage},
    )


def run_weather_pipeline(
    region: str,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    legacy_weather_cache_dir: str | Path = DEFAULT_LEGACY_WEATHER_CACHE_DIR,
    storage: pd.DataFrame | None = None,
    pause_seconds: float = 3.0,
    migrate_legacy_cache: bool = True,
) -> PipelineOutputs:
    """Download weather, aggregate to regional daily/weekly series, and export."""
    cache_dir = Path(cache_dir)
    processed_dir = Path(processed_dir)
    w_cache_dir = weather_cache_dir(cache_dir)
    slug = region_slug(region)

    if migrate_legacy_cache:
        migrate_weather_chunk_cache(legacy_weather_cache_dir, w_cache_dir)

    if storage is None:
        storage = _load_storage_for_region(processed_dir, region)

    start_date = storage["date"].min().strftime("%Y-%m-%d")
    end_date = storage["date"].max().strftime("%Y-%m-%d")

    locations = select_weather_locations(load_census_state_locations(), region)
    state_daily_weather = fetch_all_state_temperatures(
        locations=locations,
        start_date=start_date,
        end_date=end_date,
        cache_dir=w_cache_dir,
        incremental=True,
        pause_seconds=pause_seconds,
    )
    validate_state_daily_weather(
        state_daily_weather,
        expected_states=region_states(region),
    )

    state_daily_path = save_versioned_parquet(
        state_daily_weather,
        processed_dir,
        f"{slug}_state_daily_weather",
        save_latest=True,
    )

    state_degrees = calculate_hdd_cdd(state_daily_weather)
    regional_weather = aggregate_population_weighted_weather(state_degrees)
    regional_weather_model = prepare_weather_model_data(regional_weather, region)
    daily_weather_path = save_versioned_parquet(
        regional_weather_model,
        processed_dir,
        f"{slug}_daily_weather",
        save_latest=True,
    )

    regional_weather_weekly = aggregate_weather_to_storage_weeks(regional_weather)
    regional_weather_weekly_model = prepare_weekly_weather_model_data(
        regional_weather_weekly,
        region,
    )
    validate_weekly_weather(regional_weather_weekly_model)
    weekly_weather_path = save_versioned_parquet(
        regional_weather_weekly_model,
        processed_dir,
        f"{slug}_weekly_weather",
        save_latest=True,
    )

    return PipelineOutputs(
        region=region,
        paths={
            "state_daily_weather": state_daily_path,
            "state_daily_weather_latest": latest_processed_path(
                region, "state_daily_weather", processed_dir
            ),
            "daily_weather": daily_weather_path,
            "daily_weather_latest": latest_processed_path(
                region, "daily_weather", processed_dir
            ),
            "weekly_weather": weekly_weather_path,
            "weekly_weather_latest": latest_processed_path(
                region, "weekly_weather", processed_dir
            ),
        },
        frames={"weekly_weather": regional_weather_weekly_model},
    )


def run_features_pipeline(
    region: str,
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    storage: pd.DataFrame | None = None,
    weekly_weather: pd.DataFrame | None = None,
) -> PipelineOutputs:
    """Join storage and weather, engineer model features, and export."""
    processed_dir = Path(processed_dir)
    slug = region_slug(region)

    if storage is None:
        storage = _load_storage_for_region(processed_dir, region)
    if weekly_weather is None:
        weather_path = latest_processed_path(region, "weekly_weather", processed_dir)
        if not weather_path.exists():
            raise FileNotFoundError(
                f"Missing processed weather for {region}: {weather_path}. "
                "Run the weather pipeline first."
            )
        weekly_weather = pd.read_parquet(weather_path)
        if not (weekly_weather["duoarea"] == region).all():
            raise ValueError(
                f"Weather file {weather_path} does not match region {region!r}."
            )

    weekly_model_features = build_weekly_model_features(
        storage,
        weekly_weather,
        region=region,
    )
    validate_weekly_model_features(weekly_model_features, region=region)

    versioned_path = save_versioned_parquet(
        weekly_model_features,
        processed_dir,
        f"{slug}_weekly_model_features",
        save_latest=True,
    )

    return PipelineOutputs(
        region=region,
        paths={
            "weekly_model_features": versioned_path,
            "weekly_model_features_latest": latest_processed_path(
                region, "weekly_model_features", processed_dir
            ),
        },
        frames={"weekly_model_features": weekly_model_features},
    )


def run_data_pipeline(
    region: str,
    *,
    stages: tuple[PipelineStage, ...] = ALL_STAGES,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    api_key: str | None = None,
    legacy_weather_cache_dir: str | Path = DEFAULT_LEGACY_WEATHER_CACHE_DIR,
    min_start_date: str = "2010-01-01",
    revision_weeks: int = 8,
    pause_seconds: float = 3.0,
    migrate_legacy_cache: bool = True,
) -> PipelineOutputs:
    """Run one or more data pipeline stages for an EIA storage region."""
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise ValueError(f"Unknown pipeline stages: {sorted(unknown)}")

    combined = PipelineOutputs(region=region)
    storage_df: pd.DataFrame | None = None
    weekly_weather_df: pd.DataFrame | None = None

    if "storage" in stages:
        storage_result = run_storage_pipeline(
            region,
            cache_dir=cache_dir,
            processed_dir=processed_dir,
            api_key=api_key,
            min_start_date=min_start_date,
            revision_weeks=revision_weeks,
        )
        combined.paths.update(storage_result.paths)
        combined.frames.update(storage_result.frames)
        storage_df = storage_result.frames["weekly_storage"]

    if "weather" in stages:
        weather_result = run_weather_pipeline(
            region,
            cache_dir=cache_dir,
            processed_dir=processed_dir,
            legacy_weather_cache_dir=legacy_weather_cache_dir,
            storage=storage_df,
            pause_seconds=pause_seconds,
            migrate_legacy_cache=migrate_legacy_cache,
        )
        combined.paths.update(weather_result.paths)
        combined.frames.update(weather_result.frames)
        weekly_weather_df = weather_result.frames["weekly_weather"]

    if "features" in stages:
        features_result = run_features_pipeline(
            region,
            processed_dir=processed_dir,
            storage=storage_df,
            weekly_weather=weekly_weather_df,
        )
        combined.paths.update(features_result.paths)
        combined.frames.update(features_result.frames)

    return combined


def run_all_regions(
    *,
    stages: tuple[PipelineStage, ...] = ALL_STAGES,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    api_key: str | None = None,
    legacy_weather_cache_dir: str | Path = DEFAULT_LEGACY_WEATHER_CACHE_DIR,
    min_start_date: str = "2010-01-01",
    revision_weeks: int = 8,
    pause_seconds: float = 3.0,
    migrate_legacy_cache: bool = True,
) -> dict[str, PipelineOutputs]:
    """Run the data pipeline for every supported EIA storage region."""
    return {
        region: run_data_pipeline(
            region,
            stages=stages,
            cache_dir=cache_dir,
            processed_dir=processed_dir,
            api_key=api_key,
            legacy_weather_cache_dir=legacy_weather_cache_dir,
            min_start_date=min_start_date,
            revision_weeks=revision_weeks,
            pause_seconds=pause_seconds,
            migrate_legacy_cache=migrate_legacy_cache,
        )
        for region in supported_storage_regions()
    }
