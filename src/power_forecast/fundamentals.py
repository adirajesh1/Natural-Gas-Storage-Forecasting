from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_HEAT_RATE = 7.5
GAS_HEAT_CONTENT_MMBTU_PER_BCF = 1_037_000.0


def _hour_of_week(values: pd.Series) -> pd.Series:
    local = pd.to_datetime(values, utc=True).dt.tz_convert("America/Chicago")
    return local.dt.dayofweek * 24 + local.dt.hour


def forecast_hour_of_week_profile(
    actual_history: pd.DataFrame,
    *,
    value_col: str,
    delivery_hours: pd.Series,
    origin: object,
    nonnegative: bool = True,
) -> tuple[np.ndarray, str]:
    """Forecast a stable component using a trailing profile and latest-level adjustment."""
    if value_col not in actual_history:
        return np.zeros(len(delivery_hours)), "zero_missing_history"
    history = actual_history.copy()
    history["valid_at"] = pd.to_datetime(history["valid_at"], utc=True)
    origin_utc = pd.Timestamp(origin)
    origin_utc = origin_utc.tz_localize("UTC") if origin_utc.tzinfo is None else origin_utc.tz_convert("UTC")
    history = history.loc[(history["valid_at"] < origin_utc) & history[value_col].notna()].copy()
    if history.empty:
        return np.zeros(len(delivery_hours)), "zero_missing_history"
    history["hour_of_week"] = _hour_of_week(history["valid_at"])
    profile = history.groupby("hour_of_week")[value_col].mean()
    recent = history.tail(24)
    recent_profile = recent["hour_of_week"].map(profile)
    adjustment = float((recent[value_col].to_numpy() - recent_profile.to_numpy()).mean())
    future_hours = _hour_of_week(delivery_hours)
    fallback = float(history[value_col].mean())
    values = future_hours.map(profile).fillna(fallback).to_numpy(dtype=float) + adjustment
    if nonnegative:
        values = np.maximum(values, 0.0)
    return values, "hour_of_week_adjusted"


def fill_missing_baseline(
    forecast: pd.DataFrame,
    actual_history: pd.DataFrame,
    *,
    actual_col: str,
) -> pd.DataFrame:
    data = forecast.copy()
    missing = data["baseline_mw"].isna()
    if not missing.any():
        data["baseline_source"] = "ercot"
        return data
    profile, source = forecast_hour_of_week_profile(
        actual_history,
        value_col=actual_col,
        delivery_hours=data.loc[missing, "delivery_hour"],
        origin=data["forecast_origin"].iloc[0],
    )
    data["baseline_source"] = "ercot"
    data.loc[missing, "baseline_mw"] = profile
    data.loc[missing, "baseline_source"] = f"fallback_{source}"
    return data


def _gas_generation_forecast(
    stack: pd.DataFrame,
    actual_history: pd.DataFrame,
    *,
    gas_price: float,
) -> tuple[np.ndarray, str]:
    history = actual_history.copy()
    if "valid_at" in history and "forecast_origin" in stack:
        origin = pd.to_datetime(stack["forecast_origin"], utc=True).min()
        valid_at = pd.to_datetime(history["valid_at"], utc=True)
        history = history.loc[valid_at < origin].copy()
        history = history.sort_values("valid_at")
    required = {
        "gas_generation_actual_mw",
        "coal_generation_actual_mw",
    }
    if not required.issubset(history.columns):
        return stack["dispatchable_thermal_mw"].to_numpy() * 0.5, "recent_share_fallback"
    history = history.dropna(subset=list(required)).copy()
    if history.empty:
        return stack["dispatchable_thermal_mw"].to_numpy() * 0.5, "recent_share_fallback"
    denominator = history["gas_generation_actual_mw"] + history["coal_generation_actual_mw"]
    recent_share = (
        history.loc[denominator > 0, "gas_generation_actual_mw"] / denominator[denominator > 0]
    ).tail(24 * 30)
    share = float(recent_share.mean()) if not recent_share.empty else 0.5

    target = history["gas_generation_actual_mw"]
    if "dispatchable_thermal_actual_mw" not in history or len(history) < 24 * 30:
        return stack["dispatchable_thermal_mw"].to_numpy() * share, "recent_share_fallback"
    local_history = pd.to_datetime(history["valid_at"], utc=True).dt.tz_convert("America/Chicago")
    history["hour"] = local_history.dt.hour
    history["hour_sin"] = np.sin(2 * np.pi * history["hour"] / 24.0)
    history["hour_cos"] = np.cos(2 * np.pi * history["hour"] / 24.0)
    history["gas_price"] = history.get("gas_price", gas_price)
    history["conventional_outage_mw"] = history.get("conventional_outage_mw", 0.0)
    features = [
        "dispatchable_thermal_actual_mw",
        "hour_sin",
        "hour_cos",
        "conventional_outage_mw",
        "gas_price",
    ]
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    model.fit(history[features], target)
    local = pd.to_datetime(stack["delivery_hour"], utc=True).dt.tz_convert("America/Chicago")
    future = pd.DataFrame(
        {
            "dispatchable_thermal_actual_mw": stack["dispatchable_thermal_mw"],
            "hour_sin": np.sin(2 * np.pi * local.dt.hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * local.dt.hour / 24.0),
            "conventional_outage_mw": stack.get("conventional_outage_mw", 0.0),
            "gas_price": gas_price,
        }
    )
    return model.predict(future), "ridge_fuel_split"


def build_physical_stack(
    forecast: pd.DataFrame,
    actual_history: pd.DataFrame | None = None,
    *,
    heat_rate: float = DEFAULT_HEAT_RATE,
    gas_price: float = 3.0,
) -> pd.DataFrame:
    """Build an hourly, physically balanced ERCOT system stack."""
    required = {"delivery_hour", "load_forecast_mw", "wind_forecast_mw", "solar_forecast_mw"}
    missing = sorted(required - set(forecast.columns))
    if missing:
        raise ValueError(f"Power forecast missing stack columns: {missing}")
    if heat_rate <= 0:
        raise ValueError("heat_rate must be positive.")
    data = forecast.copy()
    history = pd.DataFrame() if actual_history is None else actual_history.copy()
    origin = data["forecast_origin"].iloc[0] if "forecast_origin" in data else data["delivery_hour"].min()
    profile_columns = {
        "nuclear_mw": "nuclear_actual_mw",
        "hydro_mw": "hydro_actual_mw",
        "other_nonthermal_mw": "other_nonthermal_actual_mw",
        "net_imports_mw": "net_imports_actual_mw",
        "battery_net_discharge_mw": "battery_net_discharge_actual_mw",
    }
    profile_sources: list[str] = []
    for output, actual in profile_columns.items():
        if output not in data:
            values, source = forecast_hour_of_week_profile(
                history,
                value_col=actual,
                delivery_hours=data["delivery_hour"],
                origin=origin,
                nonnegative=output not in {"net_imports_mw", "battery_net_discharge_mw"},
            )
            data[output] = values
            profile_sources.append(f"{output}:{source}")
    data["profile_sources"] = ";".join(profile_sources)
    data["net_load_mw"] = data["load_forecast_mw"] - data["wind_forecast_mw"] - data["solar_forecast_mw"]
    nonthermal = (
        data["nuclear_mw"]
        + data["hydro_mw"]
        + data["other_nonthermal_mw"]
        + data["net_imports_mw"]
        + data["battery_net_discharge_mw"]
    )
    data["raw_residual_mw"] = data["net_load_mw"] - nonthermal
    data["dispatchable_thermal_mw"] = data["raw_residual_mw"].clip(lower=0.0)
    data["curtailment_mw"] = (-data["raw_residual_mw"]).clip(lower=0.0)
    predicted_gas, gas_source = _gas_generation_forecast(data, history, gas_price=gas_price)
    data["gas_generation_mw"] = np.clip(
        predicted_gas,
        0.0,
        data["dispatchable_thermal_mw"].to_numpy(),
    )
    data["gas_generation_source"] = gas_source
    data["coal_other_generation_mw"] = data["dispatchable_thermal_mw"] - data["gas_generation_mw"]
    data["heat_rate_mmbtu_per_mwh"] = float(heat_rate)
    for label, rate in {"low": 7.0, "base": heat_rate, "high": 8.0}.items():
        data[f"gas_burn_bcf_{label}"] = (
            data["gas_generation_mw"] * float(rate) / GAS_HEAT_CONTENT_MMBTU_PER_BCF
        )
    if "available_capacity_mw" in data:
        data["capacity_margin_mw"] = data["available_capacity_mw"] - data["load_forecast_mw"]
        data["capacity_shortfall_mw"] = (-data["capacity_margin_mw"]).clip(lower=0.0)
    else:
        data["capacity_margin_mw"] = np.nan
        data["capacity_shortfall_mw"] = np.nan
    data["balanced_supply_mw"] = (
        data["wind_forecast_mw"]
        + data["solar_forecast_mw"]
        + nonthermal
        + data["gas_generation_mw"]
        + data["coal_other_generation_mw"]
        - data["curtailment_mw"]
    )
    data["balance_error_mw"] = data["balanced_supply_mw"] - data["load_forecast_mw"]
    return data
