from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from energy_forecast.evaluation import mae
from power_forecast.schemas import horizon_bucket


DEFAULT_CORRECTION_FEATURES = (
    "baseline_mw",
    "horizon_hour",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "recent_error_mw",
    "temperature_f",
    "wind_speed_100m_mph",
    "shortwave_radiation_wm2",
    "cloud_cover_pct",
)


def add_forecast_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["forecast_origin"] = pd.to_datetime(data["forecast_origin"], utc=True)
    data["delivery_hour"] = pd.to_datetime(data["delivery_hour"], utc=True)
    if "horizon_hour" not in data:
        data["horizon_hour"] = (
            (data["delivery_hour"] - data["forecast_origin"]).dt.total_seconds() / 3600
        ).round().astype(int)
    local = data["delivery_hour"].dt.tz_convert("America/Chicago")
    data["hour_sin"] = np.sin(2 * np.pi * local.dt.hour / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * local.dt.hour / 24.0)
    data["dow_sin"] = np.sin(2 * np.pi * local.dt.dayofweek / 7.0)
    data["dow_cos"] = np.cos(2 * np.pi * local.dt.dayofweek / 7.0)
    data["horizon_bucket"] = horizon_bucket(data["horizon_hour"])
    if "recent_error_mw" not in data:
        data["recent_error_mw"] = 0.0
    return data


def _candidate_models() -> dict[str, object]:
    return {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=200,
                l2_regularization=0.1,
                random_state=42,
            ),
        ),
    }


def _metrics(predictions: pd.DataFrame, prediction_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bucket, group in predictions.groupby("horizon_bucket", dropna=False):
        rows.append(
            {
                "horizon_bucket": bucket,
                "baseline_mae": mae(group["actual_mw"], group["baseline_mw"]),
                "corrected_mae": mae(group["actual_mw"], group[prediction_col]),
                "n_samples": len(group),
            }
        )
    rows.append(
        {
            "horizon_bucket": "overall",
            "baseline_mae": mae(predictions["actual_mw"], predictions["baseline_mw"]),
            "corrected_mae": mae(predictions["actual_mw"], predictions[prediction_col]),
            "n_samples": len(predictions),
        }
    )
    return pd.DataFrame(rows)


def _is_promotable(metrics: pd.DataFrame) -> bool:
    overall = metrics.loc[metrics["horizon_bucket"] == "overall"].iloc[0]
    if not overall["corrected_mae"] < overall["baseline_mae"]:
        return False
    buckets = metrics.loc[metrics["horizon_bucket"] != "overall"]
    return bool((buckets["corrected_mae"] <= buckets["baseline_mae"] * 1.05).all())


def _rolling_predictions(
    history: pd.DataFrame,
    model,
    feature_cols: list[str],
) -> pd.DataFrame:
    origins = pd.DatetimeIndex(sorted(history["forecast_origin"].unique()))
    if len(origins) < 4:
        return pd.DataFrame()
    initial = max(2, int(np.ceil(len(origins) * 0.60)))
    outputs: list[pd.DataFrame] = []
    for origin in origins[initial:]:
        train_mask = (
            (history["forecast_origin"] < origin)
            & (history["delivery_hour"] < origin)
        )
        if "retrieved_at" in history:
            train_mask &= pd.to_datetime(history["retrieved_at"], utc=True) <= origin
        train = history.loc[train_mask].dropna(
            subset=["actual_mw", "baseline_mw"]
        )
        validation = history.loc[history["forecast_origin"] == origin].dropna(
            subset=["actual_mw", "baseline_mw"]
        )
        if len(train) < 100 or validation.empty:
            continue
        fitted = clone(model).fit(
            train[feature_cols], train["actual_mw"] - train["baseline_mw"]
        )
        output = validation[
            ["forecast_origin", "delivery_hour", "horizon_bucket", "baseline_mw", "actual_mw"]
        ].copy()
        output["corrected_mw"] = validation["baseline_mw"].to_numpy() + fitted.predict(
            validation[feature_cols]
        )
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


@dataclass(frozen=True)
class CorrectionSelection:
    model_name: str
    promoted: bool
    metrics: pd.DataFrame
    validation_predictions: pd.DataFrame


def fit_predict_correction(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...] = DEFAULT_CORRECTION_FEATURES,
) -> tuple[pd.DataFrame, CorrectionSelection]:
    """Select and fit a residual correction without using later origins."""
    historical = add_forecast_features(history)
    future = add_forecast_features(forecast)
    usable_features = [column for column in feature_cols if column in historical and column in future]
    if "baseline_mw" not in usable_features:
        raise ValueError("Correction features require baseline_mw.")
    forecast_cutoff = future["forecast_origin"].min()
    clean_history = historical.loc[
        (historical["forecast_origin"] < forecast_cutoff)
        & (historical["delivery_hour"] < forecast_cutoff)
    ].dropna(subset=["actual_mw", "baseline_mw"])
    if "retrieved_at" in clean_history:
        clean_history = clean_history.loc[
            pd.to_datetime(clean_history["retrieved_at"], utc=True) <= forecast_cutoff
        ]
    origins = pd.DatetimeIndex(sorted(clean_history["forecast_origin"].unique()))
    baseline_validation = pd.DataFrame()
    if len(origins) >= 4:
        initial = max(2, int(np.ceil(len(origins) * 0.60)))
        baseline_validation = clean_history.loc[
            clean_history["forecast_origin"].isin(origins[initial:]),
            ["forecast_origin", "delivery_hour", "horizon_bucket", "baseline_mw", "actual_mw"],
        ].copy()
        baseline_validation["corrected_mw"] = baseline_validation["baseline_mw"]
    candidates: list[tuple[str, object, pd.DataFrame, pd.DataFrame]] = []
    for name, model in _candidate_models().items():
        predictions = _rolling_predictions(clean_history, model, usable_features)
        if predictions.empty:
            continue
        metrics = _metrics(predictions, "corrected_mw")
        if _is_promotable(metrics):
            candidates.append((name, model, metrics, predictions))

    result = future.copy()
    if not candidates:
        result["forecast_mw"] = result["baseline_mw"]
        result["forecast_source"] = "ercot_baseline"
        empty_metrics = pd.DataFrame(
            columns=["horizon_bucket", "baseline_mae", "corrected_mae", "n_samples"]
        )
        return result, CorrectionSelection(
            "ercot_baseline", False, empty_metrics, baseline_validation
        )

    name, model, metrics, predictions = min(
        candidates,
        key=lambda item: float(
            item[2].loc[item[2]["horizon_bucket"] == "overall", "corrected_mae"].iloc[0]
        ),
    )
    fitted = clone(model).fit(
        clean_history[usable_features],
        clean_history["actual_mw"] - clean_history["baseline_mw"],
    )
    result["forecast_mw"] = result["baseline_mw"] + fitted.predict(result[usable_features])
    result["forecast_mw"] = result["forecast_mw"].clip(lower=0.0)
    result["forecast_source"] = f"ercot_plus_{name}_correction"
    return result, CorrectionSelection(name, True, metrics, predictions)
