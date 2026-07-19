# Model Specification

## Forecast product

The primary gas target is weekly EIA working-storage change:

```text
weekly_change_bcf[t] = storage_bcf[t] - storage_bcf[t-1]
```

The primary score is Lower-48 week-one MAE from expanding, point-in-time origins
beginning in 2021. Weeks two through four, regional accuracy, bias, RMSE,
interval coverage, and hierarchy coherence are secondary diagnostics.

The published forecast schema includes:

```text
date, forecast_origin, region, horizon, model_key,
weather_provider, weather_model, weather_run,
reconciliation_method, predicted_weekly_change,
p10, p50, p90, projected_storage
```

## Storage-model inputs

Default features are deliberately compact:

- cyclical week and month encodings;
- target-week HDD and CDD from the selected weather information set;
- lagged and four-week average HDD/CDD;
- lagged and four-week average storage change;
- inventory versus trailing same-week and year-ago levels;
- injection-season flag.

Feature creation is grouped by `duoarea` and sorted by date. Target-week storage
change is never used as an input. Current-week realized HDD/CDD in the standard
historical feature table make a one-step test an oracle upper bound, not an
operational score.

## Weather information sets

`RecursiveForecaster` has three explicit modes:

| Mode | Meaning |
| --- | --- |
| `seasonal` | Target-week weather is a profile estimated only from history before the origin. Deployable, but weak during weather shocks. |
| `scenario` | Uses the newest archived weather forecast issued no later than the origin. This is the preferred operational test. |
| `observed` | Uses realized target-window weather. Oracle diagnostic only. |

Live GEFS members are aggregated into regional weekly HDD/CDD mean, P10, P90,
and spread. State weights can be population-based or derived from gas demand.
Gas-load weights are selected using their own `available_at` timestamps at each
weather run, preventing future demand revisions from changing old forecasts.

## Classical model ladder

All challengers use identical folds and features.

| Model | Role |
| --- | --- |
| Five-year weekly average | Seasonal reference with no weather response. |
| Linear regression | Transparent low-variance default and coefficient benchmark. |
| Ridge | Regularized linear default when correlated lag features are unstable. |
| ElasticNet | Sparse regularized linear challenger. |
| Random forest | Nonlinear interaction benchmark. |
| HistGradientBoosting | Main nonlinear tabular challenger. |
| Quantile HistGradientBoosting | Standalone P10 and P90 conditional estimates. |
| Linear + HGB voting ensemble | Simple average fitted only on each fold's training rows. |
| XGBoost / LightGBM | Optional tree challengers when installed. |
| ARIMAX | Dynamic regression with ARMA errors; evaluated only one step ahead. |

ARIMAX is intentionally absent from recursive model choices. Repeatedly asking
a stateless fitted SARIMAX object for its first step would not be a valid
four-week state simulation.

## Optional pooled N-HiTS

The neural challenger is one pooled model, not six small regional networks:

- series: R48 and R31-R35;
- history: 104 weeks;
- output: four weeks;
- future covariates: ensemble HDD/CDD and their P10/P90 ranges;
- loss: P10/P50/P90 multi-quantile loss;
- dependency group: `.[neural]`.

Pooling gives the network roughly six times as many related observations and
lets it learn shared seasonal behavior. N-HiTS remains optional and non-default
until it passes the same promotion gate as classical models.

## Recursive forecast state

For target week `t`, the model predicts change and advances inventory:

```text
predicted_change[t] = model(features[t])
projected_storage[t] = projected_storage[t-1] + predicted_change[t]
```

The next step uses the prior prediction to update:

- `weekly_change_lag1`;
- rolling storage-change averages;
- projected storage versus seasonal and year-ago levels;
- weather lags and rolling averages;
- optional lagged balance residuals.

Only the explicitly enumerated recursive feature columns are accepted. This
prevents a newly added contemporaneous feature from silently reading future
values during a multiweek simulation.

## Regional hierarchy

Base models are fitted independently for R48 and the five EIA regions.

### Direct

Publishes only the direct R48 prediction. Independent regional predictions are
not presented beside it as though they were coherent.

### Bottom-up

Publishes the five regional forecasts and defines:

```text
R48 = R31 + R32 + R33 + R34 + R35
```

The identity is enforced for the point forecast and available quantiles.

### MinT-shrink

MinT combines all six base forecasts using a Ledoit-Wolf shrinkage estimate of
their residual covariance. For each origin, covariance is estimated only from
complete residual vectors with target dates earlier than that origin. The
reconciled regional outputs are finalized by summing them to an exactly coherent
R48 value.

Hierarchy evaluation uses only origin/target/horizon combinations where all six
series exist. Individual base backtests retain their longer histories.

## Prediction intervals

Backtests can add rolling conformal intervals. The radius for a forecast is
calibrated from earlier out-of-fold absolute errors only, separately by horizon
for recursive forecasts. Until the minimum calibration history exists, P10/P90
remain unavailable rather than being filled with artificial certainty.

Reported diagnostics include empirical coverage, upper/lower miss rates, and
average width. These measure historical calibration; they are not guarantees of
future regime coverage.

## Evaluation and promotion

Every candidate is evaluated with chronological expanding origins and a frozen
information set. The ablation table separates:

- seasonal, archived, and observed weather;
- population and gas-load weather weighting;
- direct, bottom-up, and MinT hierarchy paths;
- model family, region, and horizon.

A challenger is promoted only when all conditions hold:

1. Lower-48 week-one MAE improves by at least 1 Bcf.
2. The lower bound of a paired four-week block-bootstrap improvement interval is positive.
3. No region's week-one MAE degrades by more than 10%.
4. Empirical 80% interval coverage is within five percentage points of target.

Model complexity alone is not evidence. The production default remains the
simplest candidate that passes this gate.

## Physical gas-balance model

The dashboard also builds a weekly analytical balance:

```text
local_balance = dry_production
                - residential_commercial
                - power_burn
                - industrial
                - fuel_use

net_inflow_balancing = actual_storage_change - local_balance
```

Monthly EIA state values are converted to daily rates, modeled or interpolated,
then aggregated into Saturday-Friday storage weeks. Weather-sensitive demand is
estimated from HDD/CDD; national production and fuel components are allocated
regionally using observed state shares.

This reconstruction uses delayed and revised monthly data. It is physical
context, not a default forecasting covariate. Balance lags may enter a model only
from an archive that retains their real `available_at` vintages.

Centered seasonal balance norms and market-tightness measures use surrounding
years and are ex-post context only. They must not enter a forecast or backtest.

## Power and oil models

Power and oil are separate physical products rather than feature extensions of
the gas storage estimator:

- [`power_fundamentals.md`](power_fundamentals.md) describes ERCOT load,
  renewables, outages, dispatch stack, and implied gas burn.
- [`oil_fundamentals.md`](oil_fundamentals.md) describes weekly crude component
  forecasts and the inventory balance identity.

They share generic timing and evaluation utilities through `energy_forecast`,
but each domain retains its own schemas and promotion logic.

## Known limits

- Historical Open-Meteo GFS fixed leads do not contain retained GEFS member
  dispersion.
- Weekly regional samples are small; neural and high-capacity tree results have
  high overfit risk.
- Supply-demand balance history is not point-in-time unless explicit vintages
  are supplied.
- The platform forecasts storage quantities, not natural-gas prices.

