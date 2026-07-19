from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from gas_forecast.modeling.evaluation import bias, mae, rmse
from gas_forecast.modeling.forecaster import ForecastInputMode, RecursiveForecaster
from gas_forecast.modeling.intervals import (
    add_rolling_conformal_intervals,
    interval_metrics,
)


def _prepare_model_rows(
    df: pd.DataFrame,
    *,
    feature_cols: list[str] | tuple[str, ...],
    target_col: str,
    date_col: str,
) -> pd.DataFrame:
    """Validate and return chronological rows usable by a tabular estimator."""
    if target_col in feature_cols:
        raise ValueError("target_col cannot also be a model feature.")

    required_cols = list(dict.fromkeys([date_col, target_col, *feature_cols]))
    missing = sorted(set(required_cols) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.loc[:, required_cols].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    return data.sort_values(date_col).dropna(subset=required_cols).reset_index(drop=True)


def _overall_metrics(
    predictions: pd.DataFrame,
    *,
    target_col: str,
) -> dict[str, float | int | str]:
    """Calculate metrics over all concatenated validation predictions."""
    return {
        "fold": "overall",
        "mae": mae(predictions[target_col], predictions["predicted_weekly_change"]),
        "rmse": rmse(predictions[target_col], predictions["predicted_weekly_change"]),
        "bias": bias(predictions[target_col], predictions["predicted_weekly_change"]),
        "n_train": pd.NA,
        "n_val": len(predictions),
    }


def _validate_chronological_fold(
    data: pd.DataFrame,
    train_idx: list[int],
    val_idx: list[int],
    *,
    date_col: str,
) -> None:
    """Reject splitter output that exposes validation dates during fitting."""
    if not train_idx or not val_idx:
        raise ValueError("Backtest folds require non-empty train and validation rows.")
    train_end = data.iloc[train_idx][date_col].max()
    validation_start = data.iloc[val_idx][date_col].min()
    if train_end >= validation_start:
        raise ValueError("Training dates must precede validation dates in every fold.")


def _add_interval_metrics(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_col: str,
    coverage: float,
    group_col: str,
) -> pd.DataFrame:
    """Append coverage diagnostics for each metrics group and any overall row."""
    result = metrics.copy()
    for index, group_value in result[group_col].items():
        rows = (
            predictions
            if group_value == "overall"
            else predictions.loc[predictions[group_col] == group_value]
        )
        stats = interval_metrics(
            rows[target_col],
            rows["interval_lower"],
            rows["interval_upper"],
            coverage=coverage,
        )
        for name, value in stats.items():
            result.loc[index, name] = value
    return result


def run_backtest(
    df: pd.DataFrame,
    feature_cols: list[str] | tuple[str, ...],
    target_col: str,
    date_col: str,
    model,
    splitter,
    interval_coverage: float | None = None,
    min_calibration_samples: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a chronological sklearn-style backtest over prebuilt feature rows.

    This evaluates one-step predictions with the feature values already present
    in ``df``. Use :func:`run_recursive_backtest` for multi-week simulations.
    """
    data = _prepare_model_rows(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        date_col=date_col,
    )

    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, float | int | str]] = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(data), start=1):
        _validate_chronological_fold(
            data,
            train_idx,
            val_idx,
            date_col=date_col,
        )
        X_train = data.iloc[train_idx][list(feature_cols)]
        y_train = data.iloc[train_idx][target_col]
        X_val = data.iloc[val_idx][list(feature_cols)]
        y_val = data.iloc[val_idx][target_col]

        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)
        y_pred = np.asarray(fitted_model.predict(X_val)).ravel()

        fold_predictions = data.iloc[val_idx][[date_col, target_col]].copy()
        fold_predictions["fold"] = fold
        fold_predictions["predicted_weekly_change"] = y_pred
        fold_predictions["forecast_deviation"] = (
            fold_predictions[target_col] - fold_predictions["predicted_weekly_change"]
        )
        predictions.append(fold_predictions)

        metrics.append(
            {
                "fold": fold,
                "mae": mae(y_val, y_pred),
                "rmse": rmse(y_val, y_pred),
                "bias": bias(y_val, y_pred),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            }
        )

    if not predictions:
        raise ValueError("Backtest produced no validation folds.")

    predictions_df = pd.concat(predictions, ignore_index=True)
    metrics_df = pd.DataFrame([*metrics, _overall_metrics(predictions_df, target_col=target_col)])
    if interval_coverage is not None:
        predictions_df = add_rolling_conformal_intervals(
            predictions_df,
            target_col=target_col,
            date_col=date_col,
            coverage=interval_coverage,
            min_calibration_samples=min_calibration_samples,
        )
        metrics_df = _add_interval_metrics(
            metrics_df,
            predictions_df,
            target_col=target_col,
            coverage=interval_coverage,
            group_col="fold",
        )
    return predictions_df, metrics_df


def run_recursive_backtest(
    df: pd.DataFrame,
    feature_cols: list[str] | tuple[str, ...],
    target_col: str,
    date_col: str,
    model,
    splitter,
    horizon_weeks: int = 4,
    forecast_input_mode: ForecastInputMode = "seasonal",
    weather_scenarios: pd.DataFrame | None = None,
    region: str | None = None,
    model_key: str | None = None,
    interval_coverage: float | None = None,
    min_calibration_samples: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run rolling-origin recursive forecasts and report metrics by horizon.

    In the default ``"seasonal"`` mode, each validation origin uses seasonal
    weather and local-balance inputs derived only from earlier history.
    ``"scenario"`` selects versioned weather forecasts as of each origin, and
    ``"observed"`` is an oracle diagnostic rather than an operational result.
    Set ``interval_coverage`` to add expanding, out-of-fold conformal intervals
    and empirical coverage metrics.
    """
    if horizon_weeks < 1:
        raise ValueError("horizon_weeks must be at least 1.")
    if date_col not in df.columns:
        raise ValueError(f"Missing required column: {date_col!r}")
    if forecast_input_mode not in {"seasonal", "observed", "scenario"}:
        raise ValueError(
            "forecast_input_mode must be 'seasonal', 'observed', or 'scenario'."
        )
    if forecast_input_mode == "scenario" and weather_scenarios is None:
        raise ValueError("scenario mode requires weather_scenarios.")
    if forecast_input_mode != "scenario" and weather_scenarios is not None:
        raise ValueError(
            "weather_scenarios is only used when forecast_input_mode='scenario'."
        )

    source_data = df.copy()
    source_data[date_col] = pd.to_datetime(source_data[date_col])
    source_data = source_data.sort_values(date_col).reset_index(drop=True)
    model_data = _prepare_model_rows(
        source_data,
        feature_cols=feature_cols,
        target_col=target_col,
        date_col=date_col,
    )

    predictions: list[pd.DataFrame] = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(model_data), start=1):
        _validate_chronological_fold(
            model_data,
            train_idx,
            val_idx,
            date_col=date_col,
        )
        train_data = model_data.iloc[train_idx]
        validation_dates = model_data.iloc[val_idx][date_col].sort_values()
        if validation_dates.empty:
            continue

        fitted_model = clone(model)
        fitted_model.fit(
            train_data[list(feature_cols)],
            train_data[target_col],
        )

        forecaster = RecursiveForecaster(
            fitted_model,
            feature_cols,
            date_col=date_col,
            target_col=target_col,
            model_key=model_key,
        )
        projection = forecaster.predict_horizon(
            features_df=source_data,
            start_date=validation_dates.iloc[0],
            horizon_weeks=min(horizon_weeks, len(validation_dates)),
            forecast_input_mode=forecast_input_mode,
            weather_scenario=weather_scenarios,
            region=region,
            as_of=train_data[date_col].max(),
        )
        if projection.empty:
            continue

        projection["fold"] = fold
        projection["horizon_weeks_ahead"] = np.arange(1, len(projection) + 1)
        projection["forecast_input_mode"] = forecast_input_mode
        predictions.append(projection)

    if not predictions:
        raise ValueError("Backtest produced no validation folds.")

    predictions_df = pd.concat(predictions, ignore_index=True)
    metrics: list[dict[str, float | int]] = []
    for horizon in sorted(predictions_df["horizon_weeks_ahead"].unique()):
        horizon_rows = predictions_df.loc[
            predictions_df["horizon_weeks_ahead"] == horizon
        ].dropna(subset=["actual_weekly_change"])
        if horizon_rows.empty:
            continue

        metrics.append(
            {
                "horizon_weeks_ahead": int(horizon),
                "mae": mae(
                    horizon_rows["actual_weekly_change"],
                    horizon_rows["predicted_weekly_change"],
                ),
                "rmse": rmse(
                    horizon_rows["actual_weekly_change"],
                    horizon_rows["predicted_weekly_change"],
                ),
                "bias": bias(
                    horizon_rows["actual_weekly_change"],
                    horizon_rows["predicted_weekly_change"],
                ),
                "n_samples": len(horizon_rows),
            }
        )

    metrics_df = pd.DataFrame(metrics)
    if interval_coverage is not None:
        predictions_df = add_rolling_conformal_intervals(
            predictions_df,
            target_col="actual_weekly_change",
            date_col="date",
            coverage=interval_coverage,
            min_calibration_samples=min_calibration_samples,
            group_cols=("horizon_weeks_ahead",),
        )
        metrics_df = _add_interval_metrics(
            metrics_df,
            predictions_df,
            target_col="actual_weekly_change",
            coverage=interval_coverage,
            group_col="horizon_weeks_ahead",
        )
        if np.isclose(interval_coverage, 0.80):
            predictions_df["p10"] = predictions_df["interval_lower"]
            predictions_df["p90"] = predictions_df["interval_upper"]

    return predictions_df, metrics_df
