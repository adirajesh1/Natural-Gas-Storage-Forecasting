from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from energy_forecast.artifacts import (
    append_vintage_parquet,
    save_versioned_parquet,
    write_parquet_cache,
)
from energy_forecast.asof import select_as_of
from energy_forecast.intervals import conformal_quantile
from power_forecast.data import (
    ERCOT_PRODUCTS,
    Eia930Client,
    ErcotApiClient,
    OpenMeteoForecastClient,
    normalize_adequacy,
    normalize_actual_load,
    normalize_component_product,
    normalize_eia930,
    normalize_outages,
    normalize_weather_forecasts,
)
from power_forecast.fundamentals import (
    DEFAULT_HEAT_RATE,
    build_physical_stack,
    fill_missing_baseline,
)
from power_forecast.models.correction import fit_predict_correction
from power_forecast.schemas import horizon_bucket
from power_forecast.schemas import frame_hash


DEFAULT_CACHE_DIR = Path("datasets/cache/power")
DEFAULT_PROCESSED_DIR = Path("datasets/processed")
COMPONENTS = ("load", "wind", "solar")


def _utc(value: object | None) -> pd.Timestamp:
    timestamp = pd.Timestamp(value or datetime.now(timezone.utc))
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _latest_path(processed_dir: str | Path, name: str) -> Path:
    return Path(processed_dir) / f"ercot_{name}_latest.parquet"


def _load_optional(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _load_eia_actual_history(processed_dir: Path) -> pd.DataFrame:
    """Load latest EIA actuals, recovering lost coverage from vintage materialization."""
    existing = _load_optional(_latest_path(processed_dir, "eia930_actuals"))
    vintages = _load_optional(_latest_path(processed_dir, "eia930_actual_vintages"))
    if vintages.empty:
        return existing
    vintages = vintages.copy()
    for column in ("valid_at", "issued_at", "retrieved_at"):
        vintages[column] = pd.to_datetime(vintages[column], utc=True)
    keys = [column for column in ("valid_at", "geography") if column in vintages]
    recovered = (
        vintages.sort_values(["issued_at", "retrieved_at"])
        .drop_duplicates(subset=keys, keep="last")
        .drop(
            columns=[
                column
                for column in (
                    "issued_at",
                    "retrieved_at",
                    "source",
                    "product_id",
                    "source_hash",
                )
                if column in vintages
            ]
        )
    )
    if existing.empty:
        return recovered.sort_values("valid_at").reset_index(drop=True)
    existing = existing.copy()
    existing["valid_at"] = pd.to_datetime(existing["valid_at"], utc=True)
    combined = pd.concat([recovered, existing], ignore_index=True)
    return (
        combined.drop_duplicates(subset=keys, keep="last")
        .sort_values("valid_at")
        .reset_index(drop=True)
    )


def _append_and_materialize(
    frame: pd.DataFrame,
    *,
    dataset: str,
    cache_dir: Path,
    processed_dir: Path,
) -> None:
    if frame.empty:
        return
    append_vintage_parquet(frame, cache_dir / "vintages" / dataset)
    latest_path = _latest_path(processed_dir, dataset)
    existing = _load_optional(latest_path)
    combined = pd.concat([existing, frame], ignore_index=True) if not existing.empty else frame.copy()
    dedupe = [column for column in ("product_id", "component", "issued_at", "valid_at", "geography") if column in combined]
    combined = combined.drop_duplicates(subset=dedupe, keep="last")
    combined = combined.sort_values([column for column in ("issued_at", "valid_at") if column in combined])
    write_parquet_cache(combined, latest_path)


def _extract_frame_and_issue(value: object) -> tuple[pd.DataFrame, object | None]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, pd.DataFrame):
        return value, None
    raise TypeError("ERCOT frame values must be DataFrames or (DataFrame, issued_at) tuples.")


def _ercot_live_params(name: str, retrieval_time: pd.Timestamp) -> dict[str, object]:
    """Bound live API pulls to recent operational data and genuine publications."""
    local = retrieval_time.tz_convert("America/Chicago")
    if name == "load_actual":
        return {
            "operatingDayFrom": (local - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
            "operatingDayTo": local.strftime("%Y-%m-%d"),
        }
    return {
        "postedDatetimeFrom": (local - pd.Timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S"),
        "postedDatetimeTo": local.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_power_data_pipeline(
    origin: object | None = None,
    *,
    ercot_frames: Mapping[str, object] | None = None,
    eia930_frame: pd.DataFrame | None = None,
    weather_frame: pd.DataFrame | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> dict[str, Path]:
    """Refresh and materialize normalized ERCOT power inputs.

    ``ercot_frames`` is an injection seam for downloaded files and tests. Its
    keys are load, load_actual, wind, solar, outages, and adequacy. A value may
    be a DataFrame containing an issuance timestamp or ``(frame, issued_at)``.
    """
    live_mode = ercot_frames is None
    query_time = _utc(origin)
    # A requested historical cutoff is not the time a live response was
    # actually observed. Injected frames retain the supplied timestamp so
    # fixture-based and replay workflows remain deterministic.
    retrieval_time = _utc(None) if live_mode else query_time
    cache_dir = Path(cache_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    if ercot_frames is None:
        client = ErcotApiClient()
        ercot_frames = {}
        for name, product in ERCOT_PRODUCTS.items():
            ercot_frames[name] = client.fetch_product_records(
                product,
                params=_ercot_live_params(name, query_time),
            )

    component_vintages: list[pd.DataFrame] = []
    for component in COMPONENTS:
        if component not in ercot_frames:
            continue
        raw, explicit_issue = _extract_frame_and_issue(ercot_frames[component])
        normalized = normalize_component_product(
            raw,
            component=component,
            product_id=ERCOT_PRODUCTS[component],
            issued_at=explicit_issue,
            retrieved_at=retrieval_time,
        )
        component_vintages.append(normalized)
    if component_vintages:
        components = pd.concat(component_vintages, ignore_index=True)
        _append_and_materialize(
            components,
            dataset="power_component_vintages",
            cache_dir=cache_dir,
            processed_dir=processed_dir,
        )

    if "load_actual" in ercot_frames:
        raw, explicit_issue = _extract_frame_and_issue(ercot_frames["load_actual"])
        actual_load = normalize_actual_load(
            raw, issued_at=explicit_issue, retrieved_at=retrieval_time
        )
        _append_and_materialize(
            actual_load,
            dataset="actual_load_vintages",
            cache_dir=cache_dir,
            processed_dir=processed_dir,
        )

    for name, normalizer in (("outages", normalize_outages), ("adequacy", normalize_adequacy)):
        if name not in ercot_frames:
            continue
        raw, explicit_issue = _extract_frame_and_issue(ercot_frames[name])
        normalized = normalizer(raw, issued_at=explicit_issue, retrieved_at=retrieval_time)
        _append_and_materialize(
            normalized,
            dataset=f"{name}_vintages",
            cache_dir=cache_dir,
            processed_dir=processed_dir,
        )

    existing_eia = _load_eia_actual_history(processed_dir)
    if eia930_frame is None:
        if not existing_eia.empty and "valid_at" in existing_eia:
            existing_times = pd.to_datetime(existing_eia["valid_at"], utc=True)
            known_times = existing_times.loc[existing_times <= query_time]
            start_at = (
                known_times.max() - pd.Timedelta(hours=48)
                if not known_times.empty
                else query_time - pd.Timedelta(days=180)
            )
        else:
            start_at = query_time - pd.Timedelta(days=180)
        start = start_at.strftime("%Y-%m-%dT%H")
        end = query_time.strftime("%Y-%m-%dT%H")
        eia930_frame = Eia930Client().fetch_actuals(start=start, end=end)
    actuals = normalize_eia930(eia930_frame) if "valid_at" not in eia930_frame else eia930_frame.copy()
    if not actuals.empty:
        actuals["valid_at"] = pd.to_datetime(actuals["valid_at"], utc=True)
        actual_vintages = actuals.copy()
        actual_vintages["issued_at"] = retrieval_time
        actual_vintages["retrieved_at"] = retrieval_time
        actual_vintages["source"] = "EIA"
        actual_vintages["product_id"] = "EIA-930"
        actual_vintages["geography"] = actual_vintages.get("geography", "ERCOT")
        actual_vintages["source_hash"] = frame_hash(eia930_frame)
        _append_and_materialize(
            actual_vintages,
            dataset="eia930_actual_vintages",
            cache_dir=cache_dir,
            processed_dir=processed_dir,
        )
        materialized_actuals = actuals.copy()
        if not existing_eia.empty:
            existing_eia = existing_eia.copy()
            existing_eia["valid_at"] = pd.to_datetime(
                existing_eia["valid_at"], utc=True
            )
            materialized_actuals = pd.concat(
                [existing_eia, materialized_actuals], ignore_index=True
            )
            actual_keys = [
                column
                for column in ("valid_at", "geography")
                if column in materialized_actuals
            ]
            materialized_actuals = materialized_actuals.drop_duplicates(
                subset=actual_keys, keep="last"
            )
        materialized_actuals = materialized_actuals.sort_values("valid_at").reset_index(
            drop=True
        )
        write_parquet_cache(
            materialized_actuals,
            _latest_path(processed_dir, "eia930_actuals"),
        )

    if weather_frame is None and live_mode:
        weather_frame = OpenMeteoForecastClient().fetch_ercot_forecast(
            issued_at=retrieval_time
        )
    if weather_frame is not None:
        weather = normalize_weather_forecasts(weather_frame)
        weather["retrieved_at"] = retrieval_time
        weather["source"] = "Open-Meteo"
        weather["product_id"] = "historical-forecast"
        weather["source_hash"] = frame_hash(weather_frame)
        _append_and_materialize(
            weather,
            dataset="weather_vintages",
            cache_dir=cache_dir,
            processed_dir=processed_dir,
        )

    return {
        name: path
        for name in (
            "power_component_vintages",
            "actual_load_vintages",
            "outages_vintages",
            "adequacy_vintages",
            "eia930_actuals",
            "eia930_actual_vintages",
            "weather_vintages",
        )
        if (path := _latest_path(processed_dir, name)).exists()
    }


def _actual_history(processed_dir: Path) -> pd.DataFrame:
    actuals = _load_eia_actual_history(processed_dir)
    if not actuals.empty:
        actuals["valid_at"] = pd.to_datetime(actuals["valid_at"], utc=True)
    load_vintages = _load_optional(_latest_path(processed_dir, "actual_load_vintages"))
    if not load_vintages.empty:
        load_vintages["issued_at"] = pd.to_datetime(load_vintages["issued_at"], utc=True)
        load_vintages["valid_at"] = pd.to_datetime(load_vintages["valid_at"], utc=True)
        latest_load = (
            load_vintages.sort_values("issued_at")
            .drop_duplicates("valid_at", keep="last")[["valid_at", "load_actual_mw"]]
        )
        actuals = latest_load if actuals.empty else actuals.merge(latest_load, on="valid_at", how="outer", suffixes=("", "_ercot"))
        if "load_actual_mw_ercot" in actuals:
            actuals["load_actual_mw"] = actuals["load_actual_mw_ercot"].combine_first(actuals.get("load_actual_mw"))
            actuals = actuals.drop(columns="load_actual_mw_ercot")
    return actuals.sort_values("valid_at").reset_index(drop=True) if not actuals.empty else actuals


def _component_actual_column(component: str) -> str:
    return f"{component}_actual_mw" if component != "load" else "load_actual_mw"


def _component_history(
    vintages: pd.DataFrame,
    actuals: pd.DataFrame,
    component: str,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    data = vintages.loc[vintages["component"] == component].copy()
    data = data.rename(columns={"issued_at": "forecast_origin", "valid_at": "delivery_hour"})
    if "retrieved_at" in data:
        data["retrieved_at"] = pd.to_datetime(data["retrieved_at"], utc=True)
    actual_col = _component_actual_column(component)
    if not actuals.empty and actual_col in actuals:
        data = data.merge(
            actuals[["valid_at", actual_col]].rename(columns={"valid_at": "delivery_hour", actual_col: "actual_from_eia"}),
            on="delivery_hour",
            how="left",
        )
        data["actual_mw"] = data["actual_from_eia"].combine_first(data["actual_mw"])
        data = data.drop(columns="actual_from_eia")
    data["horizon_hour"] = (
        (data["delivery_hour"] - data["forecast_origin"]).dt.total_seconds() / 3600
    ).round().astype(int)
    data = data.loc[data["horizon_hour"].between(1, 168)].copy()
    data["horizon_bucket"] = horizon_bucket(data["horizon_hour"])
    data["recent_error_mw"] = 0.0
    for origin in sorted(data["forecast_origin"].unique()):
        known_mask = (
            (data["forecast_origin"] < origin)
            & (data["delivery_hour"] < origin)
            & data["actual_mw"].notna()
        )
        if "retrieved_at" in data:
            known_mask &= data["retrieved_at"] <= origin
        known = data.loc[known_mask].sort_values(["delivery_hour", "forecast_origin"])
        known = known.drop_duplicates("delivery_hour", keep="last")
        if not known.empty:
            recent = float((known["actual_mw"] - known["baseline_mw"]).tail(24).mean())
            data.loc[data["forecast_origin"] == origin, "recent_error_mw"] = recent
    return _merge_weather_asof(data, weather)


def _merge_weather_asof(
    forecasts: pd.DataFrame,
    weather: pd.DataFrame | None,
) -> pd.DataFrame:
    if weather is None or weather.empty:
        return forecasts
    weather = weather.copy()
    weather["issued_at"] = pd.to_datetime(weather["issued_at"], utc=True)
    weather["valid_at"] = pd.to_datetime(weather["valid_at"], utc=True)
    if "retrieved_at" in weather:
        weather["retrieved_at"] = pd.to_datetime(weather["retrieved_at"], utc=True)
    outputs: list[pd.DataFrame] = []
    for origin, group in forecasts.groupby("forecast_origin", sort=False):
        eligible = weather.loc[weather["issued_at"] <= origin]
        sort_columns = ["issued_at"]
        if "retrieved_at" in eligible:
            eligible = eligible.loc[eligible["retrieved_at"] <= origin]
            sort_columns.append("retrieved_at")
        latest = (
            eligible.sort_values(sort_columns)
            .drop_duplicates("valid_at", keep="last")
            .drop(columns=[column for column in ("issued_at", "retrieved_at", "source", "product_id", "source_hash", "geography") if column in eligible])
            .rename(columns={"valid_at": "delivery_hour"})
        )
        outputs.append(group.merge(latest, on="delivery_hour", how="left"))
    return pd.concat(outputs, ignore_index=True) if outputs else forecasts


def load_power_history(
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    processed_dir = Path(processed_dir)
    vintages = _load_optional(_latest_path(processed_dir, "power_component_vintages"))
    if vintages.empty:
        raise FileNotFoundError("No ERCOT power component vintages have been materialized.")
    for column in ("issued_at", "retrieved_at", "valid_at"):
        vintages[column] = pd.to_datetime(vintages[column], utc=True)
    actuals = _actual_history(processed_dir)
    weather = _load_optional(_latest_path(processed_dir, "weather_vintages"))
    return pd.concat(
        [_component_history(vintages, actuals, component, weather) for component in COMPONENTS],
        ignore_index=True,
    )


def _select_optional_vintage(
    frame: pd.DataFrame,
    origin: pd.Timestamp,
    delivery_hours: pd.DatetimeIndex,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame({"delivery_hour": delivery_hours})
    for column in ("issued_at", "valid_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    selected = select_as_of(
        frame,
        origin,
        entity_keys=["geography"],
        valid_time_col="valid_at",
        issued_time_col="issued_at",
    ).rename(columns={"valid_at": "delivery_hour"})
    return pd.DataFrame({"delivery_hour": delivery_hours}).merge(selected, on="delivery_hour", how="left")


def _current_component(
    vintages: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    component: str,
    origin: pd.Timestamp,
    delivery_hours: pd.DatetimeIndex,
    weather: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, object]:
    subset = vintages.loc[vintages["component"] == component].copy()
    selected = select_as_of(
        subset,
        origin,
        entity_keys=["component", "geography"],
        valid_time_col="valid_at",
        issued_time_col="issued_at",
    )
    selected = selected.rename(columns={"valid_at": "delivery_hour"})
    future = pd.DataFrame({"delivery_hour": delivery_hours}).merge(
        selected[["delivery_hour", "baseline_mw", "issued_at", "source_hash"]],
        on="delivery_hour",
        how="left",
    )
    future["forecast_origin"] = origin
    future["horizon_hour"] = np.arange(1, len(future) + 1)
    future["actual_mw"] = np.nan
    future = fill_missing_baseline(
        future,
        actuals,
        actual_col=_component_actual_column(component),
    )
    history = _component_history(vintages, actuals, component, weather)
    known_mask = (
        (history["forecast_origin"] < origin)
        & (history["delivery_hour"] < origin)
        & history["actual_mw"].notna()
    )
    if "retrieved_at" in history:
        known_mask &= history["retrieved_at"] <= origin
    known = history.loc[known_mask].sort_values(
        ["delivery_hour", "forecast_origin"]
    )
    known = known.drop_duplicates("delivery_hour", keep="last")
    future["recent_error_mw"] = (
        float((known["actual_mw"] - known["baseline_mw"]).tail(24).mean())
        if not known.empty
        else 0.0
    )
    future = _merge_weather_asof(future, weather)
    corrected, selection = fit_predict_correction(history, future)
    corrected["component"] = component
    return corrected, selection


def _attach_current_intervals(
    forecast: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    prefix: str,
    coverage: float = 0.80,
) -> pd.DataFrame:
    data = forecast.copy()
    data[f"{prefix}_lower_mw"] = np.nan
    data[f"{prefix}_upper_mw"] = np.nan
    if predictions.empty:
        return data
    predictions = predictions.copy()
    predictions["horizon_bucket"] = horizon_bucket(
        ((predictions["delivery_hour"] - predictions["forecast_origin"]).dt.total_seconds() / 3600).round().astype(int)
    )
    predictions["abs_error"] = (predictions["actual_mw"] - predictions["corrected_mw"]).abs()
    for bucket, group in predictions.groupby("horizon_bucket"):
        radius = conformal_quantile(group["abs_error"], coverage)
        mask = data["horizon_bucket"] == bucket
        data.loc[mask, f"{prefix}_lower_mw"] = (data.loc[mask, f"{prefix}_forecast_mw"] - radius).clip(lower=0.0)
        data.loc[mask, f"{prefix}_upper_mw"] = data.loc[mask, f"{prefix}_forecast_mw"] + radius
    return data


def _attach_net_load_interval(
    forecast: pd.DataFrame,
    selections: dict[str, object],
) -> pd.DataFrame:
    data = forecast.copy()
    data["net_load_lower_mw"] = np.nan
    data["net_load_upper_mw"] = np.nan
    merged: pd.DataFrame | None = None
    for component in COMPONENTS:
        predictions = selections[component].validation_predictions
        if predictions.empty:
            return data
        part = predictions[
            ["forecast_origin", "delivery_hour", "horizon_bucket", "actual_mw", "corrected_mw"]
        ].rename(
            columns={
                "actual_mw": f"{component}_actual_mw",
                "corrected_mw": f"{component}_corrected_mw",
            }
        )
        keys = ["forecast_origin", "delivery_hour", "horizon_bucket"]
        merged = part if merged is None else merged.merge(part, on=keys, how="inner")
    if merged is None or merged.empty:
        return data
    merged["actual_net_load_mw"] = merged["load_actual_mw"] - merged["wind_actual_mw"] - merged["solar_actual_mw"]
    merged["predicted_net_load_mw"] = merged["load_corrected_mw"] - merged["wind_corrected_mw"] - merged["solar_corrected_mw"]
    merged["abs_error"] = (merged["actual_net_load_mw"] - merged["predicted_net_load_mw"]).abs()
    for bucket, group in merged.groupby("horizon_bucket"):
        radius = conformal_quantile(group["abs_error"], 0.80)
        mask = data["horizon_bucket"] == bucket
        data.loc[mask, "net_load_lower_mw"] = data.loc[mask, "net_load_mw"] - radius
        data.loc[mask, "net_load_upper_mw"] = data.loc[mask, "net_load_mw"] + radius
    return data


def _attach_gas_intervals(
    forecast: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    data = forecast.copy()
    data["gas_generation_lower_mw"] = np.nan
    data["gas_generation_upper_mw"] = np.nan
    required = {"gas_generation_actual_mw", "coal_generation_actual_mw"}
    if actuals.empty or not required.issubset(actuals.columns):
        return data
    history = actuals.dropna(subset=list(required)).copy()
    if "forecast_origin" in data and "valid_at" in history:
        origin = pd.to_datetime(data["forecast_origin"], utc=True).min()
        history = history.loc[pd.to_datetime(history["valid_at"], utc=True) < origin]
    history = history.sort_values("valid_at")
    dispatchable = history["gas_generation_actual_mw"] + history["coal_generation_actual_mw"]
    share = history["gas_generation_actual_mw"].div(dispatchable.where(dispatchable > 0))
    trailing_share = share.rolling(24 * 30, min_periods=24).mean()
    shifts = {"h001_024": 12, "h025_072": 48, "h073_168": 120}
    heat_rate = float(data["heat_rate_mmbtu_per_mwh"].iloc[0])
    for bucket, shift in shifts.items():
        predicted = dispatchable * trailing_share.shift(shift)
        residuals = (history["gas_generation_actual_mw"] - predicted).abs().dropna()
        if residuals.empty:
            continue
        radius = conformal_quantile(residuals, 0.80)
        mask = data["horizon_bucket"] == bucket
        data.loc[mask, "gas_generation_lower_mw"] = (
            data.loc[mask, "gas_generation_mw"] - radius
        ).clip(lower=0.0)
        data.loc[mask, "gas_generation_upper_mw"] = np.minimum(
            data.loc[mask, "gas_generation_mw"] + radius,
            data.loc[mask, "dispatchable_thermal_mw"],
        )
    data["gas_burn_bcf_lower"] = (
        data["gas_generation_lower_mw"] * heat_rate / 1_037_000.0
    )
    data["gas_burn_bcf_upper"] = (
        data["gas_generation_upper_mw"] * heat_rate / 1_037_000.0
    )
    return data


def build_power_forecast(
    origin: object | None = None,
    horizon_hours: int = 168,
    *,
    heat_rate: float = DEFAULT_HEAT_RATE,
    gas_price: float | None = None,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """Build and materialize a system-wide hourly ERCOT fundamentals forecast."""
    if horizon_hours < 1 or horizon_hours > 168:
        raise ValueError("horizon_hours must be between 1 and 168.")
    processed_dir = Path(processed_dir)
    vintages = _load_optional(_latest_path(processed_dir, "power_component_vintages"))
    if vintages.empty:
        raise FileNotFoundError("Run run_power_data_pipeline before building a forecast.")
    for column in ("issued_at", "retrieved_at", "valid_at"):
        vintages[column] = pd.to_datetime(vintages[column], utc=True)
    origin_utc = (
        _utc(origin)
        if origin is not None
        else pd.to_datetime(vintages["retrieved_at"], utc=True).max()
    )
    first_delivery = origin_utc.floor("h") + pd.Timedelta(hours=1)
    delivery_hours = pd.date_range(first_delivery, periods=horizon_hours, freq="h", tz="UTC")
    actuals = _actual_history(processed_dir)
    weather = _load_optional(_latest_path(processed_dir, "weather_vintages"))
    if gas_price is None:
        price_candidates = (
            processed_dir.parent / "cache" / "balance" / "daily_spot_price.parquet",
            Path("datasets/cache/balance/daily_spot_price.parquet"),
        )
        gas_price = 3.0
        for path in price_candidates:
            if path.exists():
                prices = pd.read_parquet(path)
                if not prices.empty and "value" in prices:
                    gas_price = float(pd.to_numeric(prices["value"], errors="coerce").dropna().iloc[-1])
                    break

    current: dict[str, pd.DataFrame] = {}
    selections: dict[str, object] = {}
    for component in COMPONENTS:
        current[component], selections[component] = _current_component(
            vintages,
            actuals,
            component=component,
            origin=origin_utc,
            delivery_hours=delivery_hours,
            weather=weather,
        )

    forecast = pd.DataFrame(
        {
            "forecast_origin": origin_utc,
            "delivery_hour": delivery_hours,
            "horizon_hour": np.arange(1, horizon_hours + 1),
        }
    )
    forecast["horizon_bucket"] = horizon_bucket(forecast["horizon_hour"])
    for component, frame in current.items():
        forecast[f"{component}_baseline_mw"] = frame["baseline_mw"].to_numpy()
        forecast[f"{component}_forecast_mw"] = frame["forecast_mw"].to_numpy()
        forecast[f"{component}_forecast_source"] = frame["forecast_source"].to_numpy()
        forecast[f"{component}_baseline_source"] = frame["baseline_source"].to_numpy()
        forecast[f"{component}_issued_at"] = frame["issued_at"].to_numpy()
        forecast = _attach_current_intervals(
            forecast,
            selections[component].validation_predictions,
            prefix=component,
        )

    outages = _select_optional_vintage(
        _load_optional(_latest_path(processed_dir, "outages_vintages")), origin_utc, delivery_hours
    )
    adequacy = _select_optional_vintage(
        _load_optional(_latest_path(processed_dir, "adequacy_vintages")), origin_utc, delivery_hours
    )
    for column in ("conventional_outage_mw", "renewable_outage_mw", "new_equipment_outage_mw"):
        forecast[column] = outages[column].fillna(0.0).to_numpy() if column in outages else 0.0
    forecast["available_capacity_mw"] = adequacy["available_capacity_mw"].to_numpy() if "available_capacity_mw" in adequacy else np.nan
    forecast["geography"] = "ERCOT"
    eligible_retrievals = vintages.loc[
        vintages["retrieved_at"] <= origin_utc, "retrieved_at"
    ]
    forecast["retrieved_at"] = (
        eligible_retrievals.max() if not eligible_retrievals.empty else pd.NaT
    )
    forecast = build_physical_stack(
        forecast,
        actuals,
        heat_rate=heat_rate,
        gas_price=gas_price,
    )
    forecast = _attach_net_load_interval(forecast, selections)
    forecast = _attach_gas_intervals(forecast, actuals)
    save_versioned_parquet(
        forecast,
        processed_dir,
        "ercot_hourly_power_forecast",
        save_latest=True,
    )
    return forecast
