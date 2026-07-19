from __future__ import annotations

import pandas as pd

from energy_forecast.evaluation import bias, mae, rmse
from power_forecast.models.correction import fit_predict_correction


def _fallback_predictions(
    component_history: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.Series:
    output = pd.Series(index=validation.index, dtype=float)
    history = component_history.dropna(subset=["actual_mw"]).copy()
    history["delivery_hour"] = pd.to_datetime(history["delivery_hour"], utc=True)
    for origin, group in validation.groupby("forecast_origin"):
        known_mask = (
            (history["forecast_origin"] < origin)
            & (history["delivery_hour"] < origin)
        )
        if "retrieved_at" in history:
            known_mask &= pd.to_datetime(history["retrieved_at"], utc=True) <= origin
        known = (
            history.loc[known_mask]
            .sort_values(["delivery_hour", "forecast_origin"])
            .drop_duplicates("delivery_hour", keep="last")
        )
        if known.empty:
            output.loc[group.index] = group["baseline_mw"]
            continue
        known_local = known["delivery_hour"].dt.tz_convert("America/Chicago")
        known = known.assign(hour_of_week=known_local.dt.dayofweek * 24 + known_local.dt.hour)
        profile = known.groupby("hour_of_week")["actual_mw"].mean()
        recent = known.tail(24)
        adjustment = float(
            (recent["actual_mw"] - recent["hour_of_week"].map(profile)).mean()
        )
        valid_local = group["delivery_hour"].dt.tz_convert("America/Chicago")
        valid_hour = valid_local.dt.dayofweek * 24 + valid_local.dt.hour
        output.loc[group.index] = (
            valid_hour.map(profile).fillna(float(known["actual_mw"].mean())) + adjustment
        ).clip(lower=0.0)
    return output


def run_power_backtest(
    history: pd.DataFrame,
    *,
    components: tuple[str, ...] = ("load", "wind", "solar"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate ERCOT baselines and promoted corrections by component/horizon."""
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    for component in components:
        component_history = history.loc[history["component"] == component].copy()
        if component_history.empty:
            continue
        latest_origin = component_history["forecast_origin"].max()
        latest = component_history.loc[
            component_history["forecast_origin"] == latest_origin
        ].head(1)
        _, selection = fit_predict_correction(component_history, latest)
        validation = selection.validation_predictions.copy()
        if validation.empty:
            continue
        validation["fallback_mw"] = _fallback_predictions(
            component_history, validation
        )
        validation["component"] = component
        validation["selected_model"] = selection.model_name
        predictions.append(validation)
        for bucket, group in validation.groupby("horizon_bucket"):
            evaluated = [
                ("ercot_baseline", "baseline_mw", False),
                ("hour_of_week_fallback", "fallback_mw", False),
            ]
            if selection.promoted:
                evaluated.append((selection.model_name, "corrected_mw", True))
            for model_name, column, promoted in evaluated:
                metrics.append(
                    {
                        "component": component,
                        "horizon_bucket": bucket,
                        "model": model_name,
                        "mae": mae(group["actual_mw"], group[column]),
                        "rmse": rmse(group["actual_mw"], group[column]),
                        "bias": bias(group["actual_mw"], group[column]),
                        "n_samples": len(group),
                        "promoted": promoted,
                    }
                )
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(metrics),
    )
