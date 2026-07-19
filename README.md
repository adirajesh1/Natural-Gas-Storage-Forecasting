# Natural Gas Storage Forecasting

An open, reproducible project for answering a practical market question: **how much will U.S. natural gas storage change next week?**

The repository started as a Lower 48 storage model and has grown into a small energy-fundamentals lab. Natural gas is still the main event, with companion pipelines for ERCOT power and U.S. crude oil. Everything is built around public data, point-in-time evaluation, and models that are simple enough to inspect.

This is a research project, not a trading signal or investment recommendation.

## What is in the repo?

| Market | Forecast | Main inputs |
| --- | --- | --- |
| Natural gas | Weekly storage change for Lower 48 and five EIA regions | EIA storage, realized weather, archived GFS/GEFS forecasts, balance data |
| ERCOT power | 168-hour system fundamentals and implied gas burn | Load, wind, solar, generation, outages, weather |
| U.S. crude oil | Next weekly commercial inventory change | Production, trade, refinery inputs, SPR movements, balance adjustment |

The gas workflow can produce direct Lower 48 forecasts, sum regional forecasts from the bottom up, or reconcile the full hierarchy with MinT shrinkage. Forecasts include P10, P50, and P90 estimates where the selected model supports them.

## Quick start

You will need Python 3.10 or newer and an [EIA Open Data API key](https://www.eia.gov/opendata/register.php).

```bash
git clone https://github.com/adirajesh1/Natural-Gas-Storage-Forecasting.git
cd Natural-Gas-Storage-Forecasting
python -m venv .venv
python -m pip install -e ".[dev]"
```

Activate the virtual environment using the usual command for your shell, then expose your EIA key as `EIA_API_KEY`.

Build the Lower 48 dataset:

```bash
gas-data refresh --region R48
```

Launch the dashboard:

```bash
python -m streamlit run dashboard/Gas_Fundamentals.py
```

The Streamlit app is multipage, so the gas, power, and oil views are available from the same local session.

## The idea behind the gas model

Storage is the result of a physical balance, but next week's change is heavily shaped by weather. The pipeline combines those two views:

```text
EIA storage + forecast weather + known balance data
                         |
                         v
              weekly regional features
                         |
                         v
             time-aware model backtests
                         |
                         v
        direct / bottom-up / MinT forecasts
```

One rule matters more than model choice: a forecast may only use information that existed at its forecast origin. Realized weather is useful as an oracle diagnostic, but it is never treated as an operational result.

The model ladder deliberately starts small:

- seasonal and weather-normal baselines;
- linear regression, Ridge, and ElasticNet;
- HistGradientBoosting and Random Forest;
- a prior-fold linear/tree ensemble;
- ARIMAX as a one-week challenger;
- an optional pooled N-HiTS experiment across all six storage series.

The neural model is isolated in an optional dependency group and is not the production default:

```bash
python -m pip install -e ".[neural]"
```

More complexity only earns its place when it improves the out-of-sample week-one error, preserves regional performance, and keeps interval coverage calibrated.

## Useful commands

Refresh every EIA storage region:

```bash
gas-data refresh --all-regions
```

Run only part of the gas data pipeline:

```bash
gas-data refresh --region R48 --only storage,weather,features
```

Archive a live GEFS forecast:

```bash
gas-data weather-forecast --region R48 --issued-at 2026-07-18T00:00:00Z
```

Build historical fixed-lead GFS inputs for operational backtests:

```bash
gas-data weather-forecast --region R48 --start-date 2021-03-23 --end-date 2026-01-01
```

Compare direct, bottom-up, and reconciled regional forecasts:

```bash
gas-data regional-backtest --model hist_gradient_boosting --weather-input seasonal
```

Replay an archived point-in-time weather dataset:

```bash
gas-data regional-backtest \
  --model ridge \
  --weather-input scenario \
  --weather-scenarios-path weather_scenarios.parquet
```

Every command has built-in help:

```bash
gas-data --help
gas-data regional-backtest --help
```

## Power and oil

The adjacent market models use the same principles: explicit physical identities, vintage-aware inputs, and time-ordered backtests.

```bash
# ERCOT power
power-data refresh
power-data forecast --horizon-hours 168
power-data backtest

# U.S. crude oil
oil-data refresh --start 2010-01-01
oil-data forecast
oil-data backtest --initial-train-weeks 156
```

ERCOT ingestion also requires the credentials described in [the power fundamentals guide](docs/power_fundamentals.md). Oil uses the EIA key.

## Project layout

```text
dashboard/                  Streamlit application
docs/                       Architecture and model notes
notebooks/                  Guided analysis and experiments
src/energy_forecast/        Shared artifacts, splits, metrics, and intervals
src/gas_forecast/           Gas ingestion, features, models, and pipelines
src/power_forecast/         ERCOT fundamentals pipeline
src/oil_forecast/           U.S. crude fundamentals pipeline
tests/                      Deterministic unit and integration tests
```

Generated data is written under `datasets/` and intentionally ignored by Git. The main gas feature artifact is:

```text
datasets/processed/lower48_weekly_model_features_latest.parquet
```

## Documentation

- [Architecture](docs/architecture.md) — package boundaries, data flow, artifacts, and timing rules
- [Models](docs/models.md) — model contracts, forecast modes, reconciliation, intervals, and promotion gates
- [Power fundamentals](docs/power_fundamentals.md) — ERCOT inputs and assumptions
- [Oil fundamentals](docs/oil_fundamentals.md) — weekly crude balance and evaluation
- [Decision log](docs/decisions.md) — important design choices and their rationale

For a guided tour, start with [`notebooks/00_project_walkthrough.ipynb`](notebooks/00_project_walkthrough.ipynb).

## Development

Run the complete test suite with:

```bash
python -m pytest
```

The tests cover data validation, forecast-vintage selection, recursive forecasting, regional coherence, model experiments, and the gas, power, and oil pipelines.

## A few honest limitations

- This forecasts storage changes and physical balances, not commodity prices.
- Public historical weather archives do not preserve every live ensemble member indefinitely; true ensemble replays require retained vintages or a prepared reforecast archive.
- Weekly regional storage history is a small dataset. Simple baselines are often hard to beat, and that is a useful result rather than a failure.
- Prediction intervals are calibrated from historical forecast errors. They should be monitored by horizon and season as new data arrives.

If you are exploring the project for the first time, start with the dashboard or the walkthrough notebook. If you are changing the forecasting logic, start with the timing rules in the architecture document—they are the easiest place to accidentally make a backtest look better than reality.
