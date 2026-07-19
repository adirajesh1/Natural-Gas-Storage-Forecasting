"""Materialized regional gas-storage model challenge runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gas_forecast.data.export import save_versioned_parquet
from gas_forecast.data.paths import DEFAULT_PROCESSED_DIR, latest_processed_path
from gas_forecast.data.regions import supported_storage_regions
from gas_forecast.modeling.config import DEFAULT_FEATURE_COLUMNS, sklearn_model_configs
from gas_forecast.modeling.regional_backtesting import run_hierarchical_recursive_backtest
from gas_forecast.modeling.splitters import ExpandingWindowSplitter
from gas_forecast.pipelines.data import PipelineOutputs


def _load_feature_tables(processed_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for region in supported_storage_regions():
        path = latest_processed_path(region, "weekly_model_features", processed_dir)
        if not path.exists():
            raise FileNotFoundError(f"Missing regional feature table: {path}")
        frame = pd.read_parquet(path)
        if "duoarea" not in frame or not frame["duoarea"].eq(region).all():
            raise ValueError(f"Feature table {path} does not match region {region!r}.")
        tables[region] = frame
    return tables


def run_regional_model_backtest(
    *,
    model_key: str = "hist_gradient_boosting",
    forecast_input_mode: str = "seasonal",
    weather_scenarios_path: str | Path | None = None,
    initial_train_start: str = "2010-01-01",
    initial_train_end: str = "2020-12-31",
    horizon_weeks: int = 4,
    step_weeks: int = 4,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> PipelineOutputs:
    """Run and save one comparable six-region operational model experiment."""
    processed_dir = Path(processed_dir)
    configs = {config.key: config for config in sklearn_model_configs()}
    if model_key not in configs:
        raise ValueError(f"Unknown recursive model {model_key!r}. Available: {sorted(configs)}")
    if forecast_input_mode not in {"seasonal", "scenario", "observed"}:
        raise ValueError("forecast_input_mode must be seasonal, scenario, or observed.")
    if forecast_input_mode == "scenario" and weather_scenarios_path is None:
        raise ValueError("Scenario backtests require weather_scenarios_path.")
    if forecast_input_mode != "scenario" and weather_scenarios_path is not None:
        raise ValueError("weather_scenarios_path is only valid in scenario mode.")

    scenarios = (
        pd.read_parquet(weather_scenarios_path)
        if weather_scenarios_path is not None
        else None
    )
    splitter = ExpandingWindowSplitter(
        date_col="date",
        initial_train_start=initial_train_start,
        initial_train_end=initial_train_end,
        val_weeks=horizon_weeks,
        step_weeks=step_weeks,
    )
    base, reconciled, metrics = run_hierarchical_recursive_backtest(
        _load_feature_tables(processed_dir),
        feature_cols=DEFAULT_FEATURE_COLUMNS,
        model=configs[model_key].build(),
        splitter=splitter,
        model_key=model_key,
        horizon_weeks=horizon_weeks,
        forecast_input_mode=forecast_input_mode,
        weather_scenarios=scenarios,
    )
    stem = f"gas_{model_key}_{forecast_input_mode}_regional_backtest"
    base_path = save_versioned_parquet(
        base,
        processed_dir,
        f"{stem}_base",
        save_latest=True,
    )
    reconciled_path = save_versioned_parquet(
        reconciled,
        processed_dir,
        f"{stem}_reconciled",
        save_latest=True,
    )
    metrics_path = save_versioned_parquet(
        metrics,
        processed_dir,
        f"{stem}_metrics",
        save_latest=True,
    )
    return PipelineOutputs(
        region="hierarchy",
        paths={
            "base_predictions": base_path,
            "reconciled_predictions": reconciled_path,
            "metrics": metrics_path,
        },
        frames={
            "base_predictions": base,
            "reconciled_predictions": reconciled,
            "metrics": metrics,
        },
    )

