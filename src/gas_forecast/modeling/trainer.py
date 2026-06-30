from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.base import clone

from gas_forecast.modeling.evaluate import bias, mae, rmse


def run_backtest(
    df: pd.DataFrame,
    feature_cols: list[str] | tuple[str, ...],
    target_col: str,
    date_col: str,
    model,
    splitter,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a generic sklearn-style backtest over prebuilt feature rows.

    The trainer only sorts, splits, fits, predicts, and evaluates. Feature
    engineering, lag creation, target shifting, and weather joins must happen
    before this function is called.
    """
    required_cols = list(dict.fromkeys([date_col, target_col, *feature_cols]))
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = (
        df.loc[:, required_cols]
        .copy()
        .sort_values(date_col)
        .dropna(subset=required_cols)
        .reset_index(drop=True)
    )

    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, float | int | str]] = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(data), start=1):
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
            fold_predictions[target_col]
            - fold_predictions["predicted_weekly_change"]
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
    metrics_df = pd.DataFrame(metrics)
    metrics_df = pd.concat(
        [
            metrics_df,
            pd.DataFrame(
                [
                    {
                        "fold": "overall",
                        "mae": mae(
                            predictions_df[target_col],
                            predictions_df["predicted_weekly_change"],
                        ),
                        "rmse": rmse(
                            predictions_df[target_col],
                            predictions_df["predicted_weekly_change"],
                        ),
                        "bias": bias(
                            predictions_df[target_col],
                            predictions_df["predicted_weekly_change"],
                        ),
                        "n_train": pd.NA,
                        "n_val": len(predictions_df),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    return predictions_df, metrics_df
