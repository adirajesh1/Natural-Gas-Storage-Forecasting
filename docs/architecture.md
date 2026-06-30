# Project Architecture

## Purpose

This document describes how the Gas Market Platform codebase is organized, where major responsibilities live, and how data moves from external sources into model-ready datasets.

Update this document when:

- a major module is added, removed, or renamed;
- a workflow entry point changes;
- responsibility moves between modules;
- a new external data source or generated artifact type is introduced.

## High-level flow

```text
External sources
  EIA weekly storage API
  Open-Meteo archive API
  Census state population centroids
        |
        v
Raw incremental cache
  datasets/cache/storage/
  datasets/cache/weather/by_state/
        |
        v
Processed parquet exports
  datasets/processed/
        |
        v
Model feature table
  weekly storage + population-weighted weekly weather + engineered lags
        |
        v
Forecast models, evaluation, and plots
```

The code is packaged under `src/gas_forecast` so notebooks and command-line workflows can share the same implementation.

## Repository layout

```text
src/gas_forecast/        Reusable package code
notebooks/               Exploratory and narrative analysis
datasets/                Local generated data, ignored by git
plots/                   Generated visual artifacts
docs/                    Architecture notes and decision records
dashboard/               Reserved for an app/dashboard layer
db/                      Reserved for persistent database work
models/                  Reserved for trained model artifacts
```

The empty top-level `dashboard`, `db`, and `models` directories are placeholders. If they remain unused, either document their intended ownership or remove them to reduce ambiguity.

## Package modules

| Module | Responsibility |
| --- | --- |
| `gas_forecast.cli` | Command-line entry point for data refresh workflows. |
| `gas_forecast.pipelines.data` | Orchestrates storage, weather, and feature pipelines. |
| `gas_forecast.data.cache` | Shared parquet cache loading, atomic writing, time-series merging, and date-gap detection. |
| `gas_forecast.data.paths` | Canonical local paths for cache and processed artifacts. |
| `gas_forecast.data.regions` | EIA storage-region definitions, state membership, and filesystem-safe slugs. |
| `gas_forecast.data.storage` | Compatibility facade that keeps older storage imports working. |
| `gas_forecast.data.storage_api` | EIA storage API access, pagination, and raw incremental cache refresh. |
| `gas_forecast.data.storage_transforms` | Storage cleaning, region selection, weekly change calculation, and model-data formatting. |
| `gas_forecast.data.storage_validation` | Storage dataframe validators. |
| `gas_forecast.data.weather` | Compatibility facade that keeps older weather imports working. |
| `gas_forecast.data.weather_api` | Open-Meteo request/response handling and legacy chunk cache paths. |
| `gas_forecast.data.weather_cache` | Incremental per-state weather cache behavior and legacy cache migration. |
| `gas_forecast.data.weather_locations` | Census state centroid loading and region-specific location selection. |
| `gas_forecast.data.weather_features` | HDD/CDD calculation, population-weighted aggregation, and storage-week alignment. |
| `gas_forecast.data.weather_validation` | Weather/location dataframe validators. |
| `gas_forecast.data.features` | Joins weekly storage/weather data and builds model-ready calendar, weather, and storage lag features. |
| `gas_forecast.data.export` | Versioned parquet export with optional latest-file aliases. |
| `gas_forecast.models` | Forecast model interface and implementations. |
| `gas_forecast.evaluation` | Fits models for an evaluation year and returns forecast diagnostics. |
| `gas_forecast.plotting` | Standard Plotly forecast visualizations. |

## Entry points

### CLI

The installable script is defined in `pyproject.toml`:

```text
gas-data = gas_forecast.cli:main
```

Current command:

```text
gas-data refresh --region R48
gas-data refresh --all-regions
```

Optional flags control stage selection, cache directories, processed output directories, storage revision windows, Open-Meteo request pacing, and legacy weather-cache migration.

### Python API

The main callable workflows live in `gas_forecast.pipelines.data`:

| Function | Purpose |
| --- | --- |
| `run_storage_pipeline` | Download/cache EIA storage, clean one region, calculate weekly change, export processed storage. |
| `run_weather_pipeline` | Load storage date range, download/cache state weather, aggregate daily and weekly weather, export processed weather. |
| `run_features_pipeline` | Join storage and weekly weather, build engineered model features, export feature table. |
| `run_data_pipeline` | Run one or more stages for one region. |
| `run_all_regions` | Run the selected stages for every supported region. |

### Notebooks

Notebooks are best treated as analysis consumers of package code. They can call lower-level functions while exploring, but stable workflows should eventually call the pipeline functions so notebook behavior does not drift from CLI behavior.

## Data artifacts

### Raw incremental cache

```text
datasets/cache/
  storage/
    weekly_storage_raw.parquet
  weather/
    by_state/
      Alabama.parquet
      ...
```

Storage cache behavior:

- first run backfills available history;
- later runs re-fetch a configurable recent tail window to capture EIA revisions;
- cache rows are merged and deduplicated by region, period, and series.

Weather cache behavior:

- one parquet file per state;
- requested ranges are compared against cached date coverage;
- only missing prefix/suffix gaps are fetched;
- large gaps are split into API-friendly request periods.

### Processed exports

Processed files are written to `datasets/processed` with timestamped names and latest aliases:

```text
{region_slug}_{dataset}_{timestamp}.parquet
{region_slug}_{dataset}_latest.parquet
```

Examples:

```text
lower48_weekly_storage_latest.parquet
lower48_weekly_weather_latest.parquet
lower48_weekly_model_features_latest.parquet
```

Use `weekly_model_features` as the canonical feature-table dataset name. Older `weekly_features` files may exist from previous iterations and should be treated as legacy artifacts.

## Modeling architecture

All forecast models implement `WeeklyChangeForecastModel`:

```text
fit(storage) -> model
predict(evaluation) -> predictions
```

Current implementations:

- `FiveYearWeeklyAverageModel`
- `WeeklyChangeLinearRegressionModel`
- `WeeklyChangeFourierRegressionModel`
- `WeeklyChangeSARIMAModel`

`evaluate_forecast` selects the requested evaluation year, fits the model using data no later than that year, and attaches predictions, deviations, and optional band/outside-band diagnostics.

## Testing

The test suite lives under `tests/` and is configured in `pyproject.toml`.

Current test focus:

- cache date-gap detection;
- storage-week Friday alignment;
- incomplete weather-week dropping;
- grouped storage change calculation;
- feature lags that do not leak across regions;
- evaluation-year handling that excludes future years.

Run tests with:

```text
python -m pytest
```

Install test dependencies with:

```text
python -m pip install -e ".[dev]"
```

## Design conventions

- Keep reusable behavior in `src/gas_forecast`, not notebooks.
- Keep generated parquet artifacts out of git.
- Validate data at workflow boundaries.
- Group time-series operations by `duoarea` when multiple regions may be present.
- Treat EIA storage week dates as Friday week-ending dates.
- Prefer pipeline functions for repeatable refreshes.
- Record non-obvious design choices in `docs/decisions.md`.

## Known improvement areas

- Notebook orchestration should keep moving toward pipeline calls.
- Placeholder top-level folders should either gain documented ownership or be removed.
