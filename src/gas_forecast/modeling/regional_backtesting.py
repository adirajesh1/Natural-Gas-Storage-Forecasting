"""Regional base-model backtests and point-in-time hierarchical reconciliation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from gas_forecast.modeling.backtesting import run_recursive_backtest
from gas_forecast.modeling.experiments import summarize_ablation
from gas_forecast.modeling.reconciliation import (
    ALL_STORAGE_REGIONS,
    bottom_up_reconcile,
    direct_lower48_forecast,
    mint_shrink_reconcile,
)


def reconcile_backtest_predictions(base_predictions: pd.DataFrame) -> pd.DataFrame:
    """Build direct, bottom-up, and rolling-origin MinT forecast paths."""
    required = {
        "date",
        "forecast_origin",
        "region",
        "horizon",
        "predicted_weekly_change",
        "actual_weekly_change",
    }
    missing = sorted(required - set(base_predictions.columns))
    if missing:
        raise ValueError(f"Base regional predictions missing columns: {missing}")
    data = base_predictions.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["forecast_origin"] = pd.to_datetime(data["forecast_origin"])
    hierarchy_keys = ["forecast_origin", "date", "horizon"]
    complete_keys = (
        data.groupby(hierarchy_keys)["region"]
        .nunique()
        .loc[lambda counts: counts == len(ALL_STORAGE_REGIONS)]
        .rename("region_count")
        .reset_index()[hierarchy_keys]
    )
    data = data.merge(complete_keys, on=hierarchy_keys, how="inner")
    if data.empty:
        raise ValueError("No forecast origins contain all six storage regions.")
    data["residual"] = (
        data["actual_weekly_change"] - data["predicted_weekly_change"]
    )

    outputs: list[pd.DataFrame] = []
    for origin, current in data.groupby("forecast_origin", sort=True):
        current = current.copy()
        actuals = current[["date", "horizon", "region", "actual_weekly_change"]]
        value_cols = ["predicted_weekly_change"]
        if {"p10", "p50", "p90"}.issubset(current.columns):
            if current[["p10", "p50", "p90"]].notna().all().all():
                value_cols.extend(["p10", "p50", "p90"])
            else:
                current[["p10", "p50", "p90"]] = pd.NA

        direct = direct_lower48_forecast(current)
        bottom_up = bottom_up_reconcile(current, value_cols=value_cols)
        paths = [direct, bottom_up]
        try:
            paths.append(
                mint_shrink_reconcile(
                    current,
                    data.dropna(subset=["residual"]),
                    as_of=origin,
                    value_cols=value_cols,
                )
            )
        except ValueError as exc:
            if "at least two complete residual vectors" not in str(exc):
                raise

        for path in paths:
            path = path.drop(columns=["actual_weekly_change"], errors="ignore").merge(
                actuals,
                on=["date", "horizon", "region"],
                how="left",
                validate="one_to_one",
            )
            path["forecast_origin"] = origin
            outputs.append(path)
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["forecast_origin", "reconciliation_method", "date", "region"]
    ).reset_index(drop=True)


def run_hierarchical_recursive_backtest(
    feature_tables: Mapping[str, pd.DataFrame],
    *,
    feature_cols: Sequence[str],
    model,
    splitter,
    model_key: str,
    horizon_weeks: int = 4,
    forecast_input_mode: str = "seasonal",
    weather_scenarios: pd.DataFrame | None = None,
    interval_coverage: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest one base model for all regions, then reconcile every origin."""
    missing = sorted(set(ALL_STORAGE_REGIONS) - set(feature_tables))
    if missing:
        raise ValueError(f"Hierarchical backtest missing feature tables: {missing}")
    predictions: list[pd.DataFrame] = []
    for region in ALL_STORAGE_REGIONS:
        region_predictions, _ = run_recursive_backtest(
            feature_tables[region],
            feature_cols=list(feature_cols),
            target_col="weekly_change_bcf",
            date_col="date",
            model=model,
            splitter=splitter,
            horizon_weeks=horizon_weeks,
            forecast_input_mode=forecast_input_mode,  # type: ignore[arg-type]
            weather_scenarios=weather_scenarios,
            region=region,
            model_key=model_key,
            interval_coverage=interval_coverage,
        )
        predictions.append(region_predictions)
    base = pd.concat(predictions, ignore_index=True)
    reconciled = reconcile_backtest_predictions(base)
    scored = reconciled.assign(
        weather_input=forecast_input_mode,
        weather_weighting="archive",
    )
    metrics = summarize_ablation(
        scored,
        group_cols=("reconciliation_method", "model_key", "region", "horizon"),
    )
    return base, reconciled, metrics
