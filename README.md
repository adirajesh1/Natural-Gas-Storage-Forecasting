# Gas Forecast

Weekly natural gas storage forecasting with reproducible data pipelines, weather-aware feature engineering, and time-aware model backtests.

The repository also contains an ERCOT system-wide hourly power-fundamentals MVP.
It archives public forecast vintages, applies leakage-safe load/wind/solar error
corrections, builds a balanced 168-hour supply stack, and derives an explicit
heat-rate-based gas-burn scenario. See
[`docs/power_fundamentals.md`](docs/power_fundamentals.md).

It also contains a weekly U.S. crude-oil fundamentals MVP. The model forecasts
production, imports, exports, refinery inputs, SPR movement, and the observed
balance adjustment, then combines them through the commercial crude balance
identity to forecast the next weekly inventory change. See
[`docs/oil_fundamentals.md`](docs/oil_fundamentals.md).

## Project Goal

This project forecasts weekly changes in U.S. natural gas storage, measured in billion cubic feet:

```text
weekly_change_bcf = this week's storage_bcf - last week's storage_bcf
```

Weekly storage changes are a useful gas-market signal because they reflect seasonality, weather-driven demand, supply conditions, and inventory balance.

## What This Project Does

1. Pulls weekly storage data from the EIA API.
2. Pulls historical daily weather from Open-Meteo.
3. Aggregates state weather into EIA storage regions using population weights.
4. Aligns weather to EIA Friday storage weeks.
5. Builds model-ready weekly features.
6. Backtests multiple forecasting models with time-aware splits.
7. Selects archived weather scenarios and balance vintages at a forecast origin.
8. Produces point metrics, calibrated intervals, coverage diagnostics, and forecast-error plots.

## Architecture

```text
EIA weekly storage API        Open-Meteo daily weather
          |                            |
          v                            v
   storage cache              state weather cache
          |                            |
          v                            v
 cleaned weekly storage       population-weighted weather
          |                            |
          +------------+---------------+
                       |
                       v
          weekly model feature table
                       |
                       v
      time-aware model backtests and plots
```

Core package layout:

```text
src/gas_forecast/
  cli.py                  Command-line data refresh entry point
  pipelines/data.py       Storage, weather, and feature pipeline orchestration
  data/                   API access, cache behavior, transforms, validation, features
  modeling/               Unified modeling package (concrete models, splitters, backtest, metrics, configs)
  plotting.py             Plotly forecast diagnostics
```

## Main Data Artifact

The main modeling table is:

```text
datasets/processed/lower48_weekly_model_features_latest.parquet
```

It combines weekly storage, weekly weather, and engineered features such as:

- cyclical week/month features;
- HDD/CDD and weather lags;
- rolling weather averages;
- lagged storage changes;
- storage surplus/deficit versus last year and trailing same-week averages;
- injection/withdrawal season flags.

## Modeling Approach

The data-availability contract for forecasts and backtests is documented in
[`docs/modeling_assumptions.md`](docs/modeling_assumptions.md).

- `gas_forecast.modeling`: the unified modeling package containing concrete model implementations, splitters, backtest runners, evaluation metrics, and configuration grids.

The sklearn-style layer supports any estimator with:

```python
fit(X, y)
predict(X)
```

Current configured models include:

- Linear Regression
- Ridge
- ElasticNet
- Random Forest
- HistGradientBoosting
- Quantile HistGradientBoosting variants for P10/P90 forecast ranges

## Backtesting

Random train/test splits are not appropriate for this project because the data is time ordered. The project instead includes:

- `HoldoutSplitter`: one historical train period and one later validation period.
- `ExpandingWindowSplitter`: the training window grows each fold.
- `RollingWindowSplitter`: the train and validation windows both move forward.

The main backtest function is:

```python
from gas_forecast.modeling import run_backtest
```

It returns:

- `predictions_df`: dates, actuals, predictions, fold IDs, and forecast deviations.
- `metrics_df`: MAE, RMSE, bias, and fold sizes.

For multi-week paths, use `run_recursive_backtest`. Its default `seasonal`
input mode uses only information available before each forecast origin. Its
`observed` mode is an explicit realized-weather diagnostic and should not be
reported as an operational forecast result.

Use `forecast_input_mode="scenario"` with a versioned weekly weather archive to
replay the actual forecast information set. Each archive row needs `date`,
`duoarea`, `issued_at`, `temperature_f`, `hdd`, `cdd`, and `weather_days`.
Only the latest version with `issued_at <= forecast origin` is used.

Pass `interval_coverage=0.80` to either backtest runner to attach symmetric
conformal intervals. They are calibrated from earlier out-of-fold residuals
only. The returned predictions include interval bounds and calibration sample
counts; the metrics include empirical coverage, tail-miss rates, and width.

## How To Run

Install in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

Refresh data for Lower 48:

```bash
gas-data refresh --region R48
```

Refresh all supported EIA storage regions:

```bash
gas-data refresh --all-regions
```

Run only selected pipeline stages:

```bash
gas-data refresh --region R48 --only storage,weather,features
```

Refresh, forecast, and backtest weekly crude fundamentals:

```bash
oil-data refresh --start 2010-01-01
oil-data forecast
oil-data backtest --initial-train-weeks 156
```

Launch the multipage Streamlit dashboard (Gas, Power, and Oil):

```bash
python -m streamlit run dashboard/Gas_Fundamentals.py
```

Select a materialized weather scenario from a parquet archive:

```bash
gas-data weather-scenario --region R48 --scenarios-path weather_vintages.parquet --as-of 2025-01-03T00:00:00Z
```

Build point-in-time balance features from a genuine historical vintage archive:

```bash
gas-data balance-asof --region R48 --vintages-path balance_vintages.parquet
```

The balance archive must include `date`, `duoarea`, `available_at`,
`local_balance`, and `net_inflow_balancing`. It is intentionally separate from
the retrospective output of `gas-data balance`.

## Recommended Walkthrough

Start with:

```text
notebooks/00_project_walkthrough.ipynb
```

That notebook is intended as the clean project narrative:

1. load the processed feature table;
2. inspect the target and default features;
3. run a compact model comparison;
4. plot forecast errors;
5. inspect feature importance;
6. summarize limitations and next steps.

## Results To Look For

The most useful result is not just one model score. It is the comparison between:

- a simple seasonal baseline;
- a regularized linear model;
- a nonlinear tree-based model.

That comparison shows whether added complexity improves the forecast enough to justify itself.

Example one-step retrospective result on the current Lower 48 feature table:

| Model | MAE | RMSE | Bias | Validation rows |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 13.27 | 17.58 | 0.14 | 286 |
| HistGradientBoosting | 15.18 | 20.98 | -2.08 | 286 |
| Ridge | 15.47 | 21.07 | -0.41 | 286 |
| Random Forest | 15.52 | 20.77 | -3.69 | 286 |

These scores use realized target-week weather from the historical feature table,
so they are diagnostic upper bounds rather than operational forecast scores. In
this run, the simple linear model performs best. That is still useful evidence:
for this weekly dataset and default feature set, added nonlinear complexity does
not automatically improve error.

## Limitations

- The core weather pipeline still downloads realized historical weather. Live
  weather scenarios require a separately collected, versioned provider archive.
- The current project forecasts weekly storage changes, not natural gas prices.
- The balance disaggregation output remains retrospective. The as-of pipeline
  is usable only when source vintages retain real `available_at` timestamps.
- Conformal intervals describe empirical error coverage, not a structural
  probability model; coverage should be monitored by horizon and refreshed.
- Weekly data gives a relatively small sample size, so simple baselines remain important.
