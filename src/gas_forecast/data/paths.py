from __future__ import annotations

from pathlib import Path

from gas_forecast.data.cache import DEFAULT_CACHE_DIR
from gas_forecast.data.regions import region_slug

DEFAULT_PROCESSED_DIR = Path("datasets/processed")
DEFAULT_LEGACY_WEATHER_CACHE_DIR = Path("datasets/raw/weather_cache")


def weather_cache_dir(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    """Return the per-state weather incremental cache directory."""
    return Path(cache_dir) / "weather"


def latest_processed_path(
    region: str,
    dataset: str,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> Path:
    """Return the latest processed parquet path for a region and dataset name."""
    slug = region_slug(region)
    return Path(processed_dir) / f"{slug}_{dataset}_latest.parquet"
