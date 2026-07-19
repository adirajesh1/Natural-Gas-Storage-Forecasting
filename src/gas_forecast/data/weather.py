"""Compatibility facade for weather data helpers.

New code should prefer the focused modules:

- `weather_api` for Open-Meteo request/response handling;
- `weather_cache` for incremental per-state cache behavior;
- `weather_locations` for Census location loading and region selection;
- `weather_features` for HDD/CDD and storage-week aggregation;
- `weather_validation` for dataframe validators.
"""

from gas_forecast.data.weather_api import (
    OPEN_METEO_ARCHIVE_URL,
    _date_periods,
    fetch_temperature_chunk,
)
from gas_forecast.data.weather_cache import (
    fetch_all_state_temperatures,
    migrate_weather_chunk_cache,
)
from gas_forecast.data.weather_features import (
    aggregate_population_weighted_weather,
    aggregate_weather_to_storage_weeks,
    assign_storage_week_end,
    calculate_hdd_cdd,
    prepare_weather_model_data,
    prepare_weekly_weather_model_data,
)
from gas_forecast.data.weather_locations import (
    CENSUS_POP_URL,
    load_census_state_locations,
    select_weather_locations,
)
from gas_forecast.data.weather_validation import (
    validate_state_daily_weather,
    validate_weather_locations,
    validate_weekly_weather,
)
from gas_forecast.data.weather_forecasts import (
    OPEN_METEO_ENSEMBLE_URL,
    OPEN_METEO_PREVIOUS_RUNS_URL,
    WEATHER_FORECAST_ARCHIVE_COLUMNS,
    aggregate_forecasts_to_weekly_scenarios,
    aggregate_state_forecasts,
    aggregate_state_forecasts_with_weight_history,
    build_asof_weather_features,
    fetch_open_meteo_gefs_ensemble,
    fetch_open_meteo_previous_runs,
    parse_open_meteo_ensemble_response,
    parse_open_meteo_previous_runs_response,
    select_state_weights_as_of,
    validate_weather_forecast_archive,
)

__all__ = [
    "CENSUS_POP_URL",
    "OPEN_METEO_ARCHIVE_URL",
    "OPEN_METEO_ENSEMBLE_URL",
    "OPEN_METEO_PREVIOUS_RUNS_URL",
    "WEATHER_FORECAST_ARCHIVE_COLUMNS",
    "aggregate_forecasts_to_weekly_scenarios",
    "aggregate_state_forecasts",
    "aggregate_state_forecasts_with_weight_history",
    "build_asof_weather_features",
    "fetch_open_meteo_gefs_ensemble",
    "fetch_open_meteo_previous_runs",
    "parse_open_meteo_ensemble_response",
    "parse_open_meteo_previous_runs_response",
    "select_state_weights_as_of",
    "validate_weather_forecast_archive",
    "_date_periods",
    "aggregate_population_weighted_weather",
    "aggregate_weather_to_storage_weeks",
    "assign_storage_week_end",
    "calculate_hdd_cdd",
    "fetch_all_state_temperatures",
    "fetch_temperature_chunk",
    "load_census_state_locations",
    "migrate_weather_chunk_cache",
    "prepare_weather_model_data",
    "prepare_weekly_weather_model_data",
    "select_weather_locations",
    "validate_state_daily_weather",
    "validate_weather_locations",
    "validate_weekly_weather",
]
