"""Model ablation reporting and challenger promotion gates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from gas_forecast.modeling.evaluation import bias, mae, rmse


def paired_block_bootstrap_mae_improvement(
    actual: Sequence[float],
    baseline: Sequence[float],
    challenger: Sequence[float],
    *,
    block_size: int = 4,
    n_bootstrap: int = 2_000,
    random_state: int = 42,
) -> dict[str, float]:
    """Estimate baseline-minus-challenger MAE and a paired block interval."""
    actual_values = np.asarray(actual, dtype=float)
    baseline_values = np.asarray(baseline, dtype=float)
    challenger_values = np.asarray(challenger, dtype=float)
    if not (
        len(actual_values) == len(baseline_values) == len(challenger_values)
    ):
        raise ValueError("Paired prediction arrays must have equal lengths.")
    if len(actual_values) < 2 or block_size < 1 or n_bootstrap < 1:
        raise ValueError("Bootstrap requires at least two rows and positive settings.")
    errors = np.abs(actual_values - baseline_values) - np.abs(
        actual_values - challenger_values
    )
    rng = np.random.default_rng(random_state)
    starts = np.arange(len(errors))
    draws = np.empty(n_bootstrap)
    blocks_needed = int(np.ceil(len(errors) / block_size))
    for index in range(n_bootstrap):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sampled = np.concatenate(
            [
                errors[(start + np.arange(block_size)) % len(errors)]
                for start in sampled_starts
            ]
        )[: len(errors)]
        draws[index] = sampled.mean()
    return {
        "mae_improvement": float(errors.mean()),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
    }


def summarize_ablation(
    predictions: pd.DataFrame,
    *,
    group_cols: Sequence[str] = (
        "weather_input",
        "weather_weighting",
        "reconciliation_method",
        "model_key",
        "region",
        "horizon",
    ),
    target_col: str = "actual_weekly_change",
    prediction_col: str = "predicted_weekly_change",
) -> pd.DataFrame:
    """Summarize one consistent metric table across experiment dimensions."""
    required = {*group_cols, target_col, prediction_col}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Ablation predictions missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for keys, group in predictions.dropna(subset=[target_col, prediction_col]).groupby(
        list(group_cols), dropna=False
    ):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values, strict=True))
        row.update(
            mae=mae(group[target_col], group[prediction_col]),
            rmse=rmse(group[target_col], group[prediction_col]),
            bias=bias(group[target_col], group[prediction_col]),
            n_samples=len(group),
        )
        if {"p10", "p90"}.issubset(group.columns):
            intervals = group.dropna(subset=["p10", "p90"])
            row["interval_coverage"] = (
                float(
                    (
                        (intervals[target_col] >= intervals["p10"])
                        & (intervals[target_col] <= intervals["p90"])
                    ).mean()
                )
                if not intervals.empty
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_challenger_promotion(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
    *,
    min_improvement_bcf: float = 1.0,
    max_region_degradation: float = 0.10,
    target_coverage: float = 0.80,
    coverage_tolerance: float = 0.05,
) -> dict[str, object]:
    """Apply the agreed week-one accuracy, stability, and calibration gate."""
    keys = ["date", "region", "horizon"]
    required = {*keys, "actual_weekly_change", "predicted_weekly_change"}
    for name, frame in (("baseline", baseline), ("challenger", challenger)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} predictions missing columns: {missing}")
    merged = baseline.merge(
        challenger,
        on=keys,
        suffixes=("_baseline", "_challenger"),
        validate="one_to_one",
    )
    primary = merged.loc[(merged["region"] == "R48") & (merged["horizon"] == 1)]
    if len(primary) < 2:
        raise ValueError("Promotion requires at least two paired Lower-48 week-one rows.")
    if not np.allclose(
        primary["actual_weekly_change_baseline"],
        primary["actual_weekly_change_challenger"],
    ):
        raise ValueError("Baseline and challenger actuals do not match.")
    bootstrap = paired_block_bootstrap_mae_improvement(
        primary["actual_weekly_change_baseline"],
        primary["predicted_weekly_change_baseline"],
        primary["predicted_weekly_change_challenger"],
    )

    degradation: dict[str, float] = {}
    for region, group in merged.loc[merged["horizon"] == 1].groupby("region"):
        baseline_mae = mae(
            group["actual_weekly_change_baseline"],
            group["predicted_weekly_change_baseline"],
        )
        challenger_mae = mae(
            group["actual_weekly_change_challenger"],
            group["predicted_weekly_change_challenger"],
        )
        degradation[region] = (
            (challenger_mae - baseline_mae) / baseline_mae
            if baseline_mae > 0
            else float(challenger_mae > 0)
        )

    coverage = np.nan
    if {"p10_challenger", "p90_challenger"}.issubset(primary.columns):
        intervals = primary.dropna(subset=["p10_challenger", "p90_challenger"])
        if not intervals.empty:
            coverage = float(
                (
                    (intervals["actual_weekly_change_challenger"] >= intervals["p10_challenger"])
                    & (intervals["actual_weekly_change_challenger"] <= intervals["p90_challenger"])
                ).mean()
            )
    coverage_pass = not np.isnan(coverage) and abs(coverage - target_coverage) <= coverage_tolerance
    region_pass = max(degradation.values(), default=0.0) <= max_region_degradation
    promote = bool(
        bootstrap["mae_improvement"] >= min_improvement_bcf
        and bootstrap["ci_lower"] > 0
        and region_pass
        and coverage_pass
    )
    return {
        **bootstrap,
        "max_region_degradation": max(degradation.values(), default=0.0),
        "regional_degradation": degradation,
        "interval_coverage": coverage,
        "accuracy_pass": bootstrap["mae_improvement"] >= min_improvement_bcf,
        "significance_pass": bootstrap["ci_lower"] > 0,
        "region_pass": region_pass,
        "coverage_pass": coverage_pass,
        "promote": promote,
    }

