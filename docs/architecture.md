# Platform Architecture

## Purpose

This repository is a local, file-backed forecasting platform for three energy
domains:

- weekly U.S. natural-gas storage;
- hourly ERCOT power fundamentals;
- weekly U.S. crude-oil inventories.

The packages share timing, artifact, evaluation, and interval utilities while
keeping domain ingestion and physical models separate. There is no database or
long-running application service. Commands materialize parquet artifacts, and
the Streamlit dashboard reads those artifacts.

## System flow

```text
External APIs and explicit vintage archives
                |
                v
       datasets/cache/                 append-only or incremental raw data
                |
                v
       domain pipeline code            validation, timing, transformation
                |
                v
       datasets/processed/             versioned parquet + latest aliases
                |
                v
       backtests and forecasts         point, interval, and hierarchy outputs
                |
                v
       Streamlit dashboard / analysis
```

Generated data under `datasets/` is intentionally ignored by git. Reproducible
behavior lives in `src/`, tests, configuration, and documentation.

## Package boundaries

### `energy_forecast`

Domain-neutral infrastructure:

- append-only artifact and vintage handling;
- exact `issued_at`/`valid_at` as-of selection;
- chronological split and backtest helpers;
- common error metrics;
- conformal interval calibration.

Domain packages may use these primitives but should not put gas-, oil-, or
power-specific schemas in this package.

### `gas_forecast`

The gas package owns the primary weekly storage product.

| Area | Responsibility |
| --- | --- |
| `data/regions.py` | Canonical R48 and R31-R35 geography, labels, and state membership. |
| `data/storage_*` | EIA weekly storage ingestion, cleaning, changes, and validation. |
| `data/weather_*` | Realized weather, forecast vintages, regional weighting, HDD/CDD, and storage-week alignment. |
| `data/features.py` | Leakage-safe calendar, weather, inventory, lag, and rolling features. |
| `data/balance_*` | Retrospective physical gas balance and explicit point-in-time balance vintages. |
| `pipelines/data.py` | Realized storage/weather/feature materialization. |
| `pipelines/asof.py` | Live GEFS, historical fixed-lead GFS, and as-of balance materialization. |
| `pipelines/modeling.py` | Six-region backtests and saved hierarchy outputs. |
| `modeling/` | Estimator configuration, recursive forecasting, backtests, reconciliation, intervals, and promotion tests. |
| `llm/explain.py` | Optional narrative generation over already-computed results. |

Compatibility modules such as `data/weather.py` and `data/storage.py` re-export
focused modules for older consumers. New code should import the focused module
that owns the behavior.

### `power_forecast`

The power package owns public ERCOT/EIA-930 ingestion, archived hourly
vintages, residual correction, the 168-hour physical supply stack, and implied
gas-burn scenarios. Its canonical timestamps are UTC. See
[`power_fundamentals.md`](power_fundamentals.md).

### `oil_forecast`

The oil package owns EIA weekly crude components, physical balance forecasts,
inventory-change backtests, and its CLI workflow. See
[`oil_fundamentals.md`](oil_fundamentals.md).

## Gas data paths

### Realized-history path

```text
EIA weekly storage ----------------------+
                                         |
Open-Meteo realized state weather        +--> weekly feature table
  -> regional weights                    |       -> model fitting/backtests
  -> HDD/CDD                              |
  -> Saturday-Friday aggregation --------+
```

The canonical artifact for a region is:

```text
datasets/processed/{region_slug}_weekly_model_features_latest.parquet
```

Every artifact is validated against its requested `duoarea`. This prevents a
stale region file—for example, Pacific data labeled Mountain—from entering a
hierarchical run.

### Forecast-weather path

```text
Open-Meteo live GEFS or historical GFS fixed leads
  -> member/state daily archive
  -> population or point-in-time gas-load weighting
  -> regional member archive
  -> complete EIA-week ensemble summaries
  -> as-of scenario selection
```

Member archives retain provider, model, run, member, issue time, valid period,
state, region, temperature, HDD, CDD, and coverage. Weekly scenarios retain
ensemble means, P10, P90, spread, and member count.

Historical fixed-lead GFS data supports honest week-one replay from 2021.
Longer historical ensemble horizons require a separately retained GEFS archive
or a prepared NOAA reforecast dataset; the system does not manufacture ensemble
dispersion from deterministic history.

### Physical-balance path

Monthly state fundamentals are downscaled to weekly regional estimates for
production, sector consumption, fuel use, prices, and local balance. This is an
analytical context product. It is excluded from the default storage model unless
the caller supplies real historical vintages with availability timestamps.

## Modeling and hierarchy flow

```text
R48 feature table ----------------------> direct R48 base forecast
R31-R35 feature tables -----------------> five regional base forecasts
                                               |
                     +-------------------------+--------------------+
                     |                         |                    |
                  direct R48               bottom-up          MinT-shrink
                                                                      |
                                               coherent published outputs
```

All six series use the same estimator, feature contract, splitter, horizon, and
weather input mode in a hierarchy experiment. Reconciliation only uses origins
where all six forecasts exist. MinT covariance uses residuals whose target dates
are earlier than the current origin.

## Entry points

Installed commands:

```text
gas-data    gas storage, weather, balance, and hierarchy workflows
power-data  ERCOT power workflows
oil-data    crude-oil workflows
```

Important gas commands:

```bash
gas-data refresh --region R48
gas-data refresh --all-regions
gas-data weather-forecast --region R48 --issued-at 2026-07-18T00:00:00Z
gas-data weather-forecast --region R48 --start-date 2021-03-23 --end-date 2026-01-01
gas-data regional-backtest --model ridge --weather-input seasonal
gas-data regional-backtest --model ridge --weather-input scenario --weather-scenarios-path weather.parquet
```

The dashboard is launched with:

```bash
python -m streamlit run dashboard/Gas_Fundamentals.py
```

## Artifact conventions

Processed outputs use timestamped immutable files and a replaceable latest
alias:

```text
{name}_{UTC timestamp}_{content hash}.parquet
{name}_latest.parquet
```

Forecast outputs identify their origin and information set with fields such as
`forecast_origin`, `region`, `horizon`, `model_key`, `weather_provider`,
`weather_model`, `weather_run`, and `reconciliation_method`.

## Timing invariants

- Training dates must precede validation dates.
- Forecast vintages must have `issued_at <= forecast_origin`.
- Recursive scenario forecasts require every target week; they never fall back
  silently to realized weather.
- Realized future weather is allowed only in the explicitly named `observed`
  oracle mode.
- Balance features require actual `available_at` vintages.
- MinT covariance and conformal intervals use earlier out-of-fold errors only.
- Bottom-up and MinT point and available quantile outputs sum exactly to R48.

## Testing strategy

The test suite covers API parsing, date alignment, region invariants, feature
timing, recursive state transitions, as-of selection, interval calibration,
hierarchy coherence, promotion gates, and full domain pipelines.

Run it with an explicit writable temporary directory on restricted Windows
environments:

```bash
python -m pytest --basetemp datasets/cache/pytest-run
```

## Design rules

- Put stable behavior in packages, not notebooks or dashboard callbacks.
- Preserve raw vintages; select the information set at read time.
- Validate schemas and geography at pipeline boundaries.
- Prefer explicit small functions over generic workflow frameworks.
- Keep optional heavy models outside base dependencies.
- Do not promote a more complex model without paired operational evidence.
