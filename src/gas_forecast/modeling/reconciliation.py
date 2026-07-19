"""Coherent regional natural-gas storage forecast reconciliation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


LOWER48_REGION = "R48"
STORAGE_REGIONS = ("R31", "R32", "R33", "R34", "R35")
ALL_STORAGE_REGIONS = (LOWER48_REGION, *STORAGE_REGIONS)


def _validate_forecasts(
    forecasts: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    required = {"region", *group_cols, *value_cols}
    missing = sorted(required - set(forecasts.columns))
    if missing:
        raise ValueError(f"Regional forecasts missing columns: {missing}")
    data = forecasts.copy()
    for column in value_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[list(value_cols)].isna().any().any():
        raise ValueError("Regional forecasts contain missing prediction values.")
    if data.duplicated(subset=[*group_cols, "region"]).any():
        raise ValueError("Regional forecasts require one row per group and region.")
    return data


def _require_bottom_regions(group: pd.DataFrame) -> pd.DataFrame:
    indexed = group.set_index("region")
    missing = sorted(set(STORAGE_REGIONS) - set(indexed.index))
    if missing:
        raise ValueError(f"Regional forecasts missing bottom regions: {missing}")
    return indexed


def _finalize_reconciled(
    frame: pd.DataFrame,
    *,
    value_cols: Sequence[str],
    group_cols: Sequence[str],
) -> pd.DataFrame:
    quantiles = ["p10", "p50", "p90"]
    data = frame.copy()
    has_quantiles = set(quantiles).issubset(data.columns)
    if has_quantiles:
        bottom_mask = data["region"].isin(STORAGE_REGIONS)
        complete_bottom = bottom_mask & data[quantiles].notna().all(axis=1)
        data.loc[complete_bottom, quantiles] = np.sort(
            data.loc[complete_bottom, quantiles].to_numpy(dtype=float), axis=1
        )
    for _, group in data.groupby(list(group_cols), sort=False):
        parent_index = group.index[group["region"] == LOWER48_REGION]
        if len(parent_index) != 1:
            continue
        children = group.loc[group["region"].isin(STORAGE_REGIONS)]
        for column in value_cols:
            data.loc[parent_index[0], column] = children[column].sum()
        if has_quantiles and not set(quantiles).issubset(value_cols):
            data.loc[parent_index[0], quantiles] = np.nan
    return data


def direct_lower48_forecast(
    forecasts: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("date", "horizon"),
) -> pd.DataFrame:
    """Return the direct aggregate forecast without publishing incoherent children."""
    required = {"region", *group_cols}
    missing = sorted(required - set(forecasts.columns))
    if missing:
        raise ValueError(f"Regional forecasts missing columns: {missing}")
    direct = forecasts.loc[forecasts["region"] == LOWER48_REGION].copy()
    if direct.duplicated(subset=list(group_cols)).any():
        raise ValueError("Direct Lower-48 forecasts require one row per group.")
    direct["reconciliation_method"] = "direct"
    return direct.reset_index(drop=True)


def bottom_up_reconcile(
    forecasts: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("date", "horizon"),
    value_cols: Sequence[str] = ("predicted_weekly_change",),
) -> pd.DataFrame:
    """Publish bottom-region forecasts and set Lower-48 to their exact sum."""
    data = _validate_forecasts(forecasts, group_cols=group_cols, value_cols=value_cols)
    output: list[pd.DataFrame] = []
    for _, group in data.groupby(list(group_cols), sort=False):
        indexed = _require_bottom_regions(group)
        children = indexed.loc[list(STORAGE_REGIONS)].reset_index()
        parent = children.iloc[[0]].copy()
        parent["region"] = LOWER48_REGION
        parent.loc[:, list(value_cols)] = children[list(value_cols)].sum().to_numpy()
        output.append(pd.concat([parent, children], ignore_index=True))
    reconciled = pd.concat(output, ignore_index=True)
    reconciled["reconciliation_method"] = "bottom_up"
    return _finalize_reconciled(
        reconciled,
        value_cols=value_cols,
        group_cols=group_cols,
    )


def _residual_covariance(
    residuals: pd.DataFrame,
    *,
    residual_col: str,
    residual_group_cols: Sequence[str],
) -> np.ndarray:
    required = {"region", residual_col, *residual_group_cols}
    missing = sorted(required - set(residuals.columns))
    if missing:
        raise ValueError(f"Reconciliation residuals missing columns: {missing}")
    matrix = residuals.pivot_table(
        index=list(residual_group_cols),
        columns="region",
        values=residual_col,
        aggfunc="first",
    ).reindex(columns=ALL_STORAGE_REGIONS)
    matrix = matrix.dropna()
    if len(matrix) < 2:
        raise ValueError("MinT reconciliation requires at least two complete residual vectors.")
    return LedoitWolf().fit(matrix.to_numpy(dtype=float)).covariance_


def mint_shrink_reconcile(
    forecasts: pd.DataFrame,
    residuals: pd.DataFrame,
    *,
    as_of: object,
    group_cols: Sequence[str] = ("date", "horizon"),
    residual_group_cols: Sequence[str] = ("date", "horizon"),
    value_cols: Sequence[str] = ("predicted_weekly_change",),
    residual_col: str = "residual",
) -> pd.DataFrame:
    """Apply MinT reconciliation using shrinkage covariance from prior residuals."""
    data = _validate_forecasts(forecasts, group_cols=group_cols, value_cols=value_cols)
    if "date" not in residuals.columns:
        raise ValueError("Reconciliation residuals require a date column.")
    origin = pd.Timestamp(as_of).tz_localize(None)
    residual_dates = pd.to_datetime(residuals["date"], errors="coerce").dt.tz_localize(None)
    if residual_dates.isna().any():
        raise ValueError("Reconciliation residuals contain invalid dates.")
    prior_residuals = residuals.loc[residual_dates < origin].copy()
    covariance = _residual_covariance(
        prior_residuals,
        residual_col=residual_col,
        residual_group_cols=residual_group_cols,
    )

    summing = np.vstack([np.ones(len(STORAGE_REGIONS)), np.eye(len(STORAGE_REGIONS))])
    precision = np.linalg.pinv(covariance)
    projection = summing @ np.linalg.pinv(summing.T @ precision @ summing) @ summing.T @ precision

    output: list[pd.DataFrame] = []
    for _, group in data.groupby(list(group_cols), sort=False):
        indexed = group.set_index("region")
        missing = sorted(set(ALL_STORAGE_REGIONS) - set(indexed.index))
        if missing:
            raise ValueError(f"MinT forecasts missing regions: {missing}")
        reconciled = indexed.loc[list(ALL_STORAGE_REGIONS)].copy()
        for column in value_cols:
            values = indexed.loc[list(ALL_STORAGE_REGIONS), column].to_numpy(dtype=float)
            reconciled[column] = projection @ values
        output.append(reconciled.reset_index())
    result = pd.concat(output, ignore_index=True)
    result["reconciliation_method"] = "mint_shrink"
    return _finalize_reconciled(
        result,
        value_cols=value_cols,
        group_cols=group_cols,
    )


def reconciliation_error(
    forecasts: pd.DataFrame,
    *,
    value_col: str = "predicted_weekly_change",
    group_cols: Sequence[str] = ("date", "horizon"),
) -> pd.DataFrame:
    """Return signed Lower-48 minus bottom-region sums for each forecast group."""
    data = _validate_forecasts(
        forecasts,
        group_cols=group_cols,
        value_cols=(value_col,),
    )
    rows: list[dict[str, object]] = []
    for keys, group in data.groupby(list(group_cols), sort=False):
        indexed = group.set_index("region")
        missing = sorted(set(ALL_STORAGE_REGIONS) - set(indexed.index))
        if missing:
            raise ValueError(f"Coherence check missing regions: {missing}")
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values, strict=True))
        row["reconciliation_error"] = float(
            indexed.loc[LOWER48_REGION, value_col]
            - indexed.loc[list(STORAGE_REGIONS), value_col].sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)
