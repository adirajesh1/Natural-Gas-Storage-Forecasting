import pandas as pd
import numpy as np

from gas_forecast.models.base import WeeklyChangeForecastModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_forecast(
    storage: pd.DataFrame,
    model: WeeklyChangeForecastModel,
    *,
    year: int | None = None,
) -> pd.DataFrame:
    """Fit a model and compare its predictions to actual weekly changes."""
    if year is None:
        year = storage["year"].max()

    training_storage = storage.loc[storage["year"] <= year].copy()
    actuals = storage.loc[
        storage["year"] == year, ["date", "week_of_year", "weekly_change_bcf"]
    ].copy()

    model.fit(training_storage)
    preds = model.predict(actuals)

    result = actuals.copy()
    for col in preds.columns:
        result[col] = preds[col].values

    result["forecast_deviation"] = (
        result["weekly_change_bcf"] - result["predicted_weekly_change"]
    )

    if {"lower_band", "upper_band"}.issubset(result.columns):
        result["outside_band"] = (
            (result["weekly_change_bcf"] > result["upper_band"])
            | (result["weekly_change_bcf"] < result["lower_band"])
        )
    else:
        result["outside_band"] = False

    return result

def error_metrics(forecast: pd.DataFrame) -> pd.DataFrame:
    """Calculate error metrics for a forecast."""
    mae = mean_absolute_error(forecast["weekly_change_bcf"], forecast["predicted_weekly_change"])
    rmse = np.sqrt(mean_squared_error(forecast["weekly_change_bcf"], forecast["predicted_weekly_change"]))
    r2 = r2_score(forecast["weekly_change_bcf"], forecast["predicted_weekly_change"])
    return pd.DataFrame({"Model": [mae, rmse, r2]}, index=["MAE", "RMSE", "R2"])
