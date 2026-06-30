# Gas Forecast

Weekly natural gas storage forecasting with reproducible data pipelines, weather-aware feature engineering, and time-aware model backtests.

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
7. Produces metrics and forecast-error plots.

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
  models/                 Interpretable baseline forecast classes
  modeling/               Sklearn-style splitters, trainer, metrics, configs
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

The project uses two modeling layers:

- `gas_forecast.models`: interpretable baseline model classes, useful for explaining the forecasting problem.
- `gas_forecast.modeling`: the preferred sklearn-style backtesting layer for comparing models on the shared feature table.

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

Example expanding-window result on the current Lower 48 feature table:

| Model | MAE | RMSE | Bias | Validation rows |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 13.27 | 17.58 | 0.14 | 286 |
| HistGradientBoosting | 15.18 | 20.98 | -2.08 | 286 |
| Ridge | 15.47 | 21.07 | -0.41 | 286 |
| Random Forest | 15.52 | 20.77 | -3.69 | 286 |

In this run, the simple linear model performs best. That is a useful modeling result: for this weekly dataset and default feature set, added nonlinear complexity does not automatically improve error.

## Limitations

- Weather inputs are historical weather, not weather forecasts.
- The current project forecasts weekly storage changes, not natural gas prices.
- The model does not yet include production, LNG exports, pipeline flows, power burn, or Henry Hub prices.
- Recursive multi-step forecasting is intentionally not implemented yet.
- Weekly data gives a relatively small sample size, so simple baselines remain important.
