from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence

import numpy as np
import pandas as pd


RAW_COMPONENTS = (
    "production_kbpd",
    "refinery_inputs_kbpd",
    "imports_kbpd",
    "exports_kbpd",
    "commercial_stocks_kb",
    "spr_stocks_kb",
)
FORECAST_COMPONENTS = (
    "production_mmbbl",
    "imports_mmbbl",
    "refinery_inputs_mmbbl",
    "exports_mmbbl",
    "spr_stock_change_mmbbl",
    "balance_adjustment_mmbbl",
)
NONNEGATIVE_COMPONENTS = frozenset(
    {
        "production_mmbbl",
        "imports_mmbbl",
        "refinery_inputs_mmbbl",
        "exports_mmbbl",
    }
)


def build_weekly_crude_balance(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert long EIA weekly series into a U.S. commercial crude balance."""
    required = {"period", "component", "value"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Weekly crude data missing required columns: {missing}")
    data = raw.copy()
    data["period"] = pd.to_datetime(data["period"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    if data[["period", "value"]].isna().any().any():
        raise ValueError("Weekly crude data contains invalid dates or values.")
    if not np.isfinite(data["value"]).all():
        raise ValueError("Weekly crude data contains non-finite values.")
    if data["period"].dt.dayofweek.ne(4).any():
        raise ValueError("Weekly crude data must use Friday week-ending dates.")
    available = set(data["component"])
    absent = sorted(set(RAW_COMPONENTS) - available)
    if absent:
        raise ValueError(f"Weekly crude data missing components: {absent}")
    data = data.loc[data["component"].isin(RAW_COMPONENTS)]
    if data["value"].lt(0).any():
        raise ValueError("Weekly crude data contains negative values.")
    if data.duplicated(["period", "component"]).any():
        raise ValueError("Weekly crude data has duplicate period/component rows.")

    wide = data.pivot(index="period", columns="component", values="value")
    if wide[list(RAW_COMPONENTS)].isna().any().any():
        raise ValueError("Weekly crude data has incomplete component rows.")
    wide = wide.sort_index().reset_index()
    if len(wide) > 1 and not wide["period"].diff().dropna().eq(
        pd.Timedelta(days=7)
    ).all():
        raise ValueError("Weekly crude components do not form consecutive weeks.")
    result = pd.DataFrame({"date": wide["period"]})
    for component in (
        "production",
        "imports",
        "refinery_inputs",
        "exports",
    ):
        result[f"{component}_mmbbl"] = wide[f"{component}_kbpd"] * 7.0 / 1000.0
    result["commercial_stocks_mmbbl"] = wide["commercial_stocks_kb"] / 1000.0
    result["spr_stocks_mmbbl"] = wide["spr_stocks_kb"] / 1000.0
    result["commercial_stock_change_mmbbl"] = result[
        "commercial_stocks_mmbbl"
    ].diff()
    result["spr_stock_change_mmbbl"] = result["spr_stocks_mmbbl"].diff()
    result["fundamental_balance_mmbbl"] = (
        result["production_mmbbl"]
        + result["imports_mmbbl"]
        - result["refinery_inputs_mmbbl"]
        - result["exports_mmbbl"]
        - result["spr_stock_change_mmbbl"]
    )
    result["balance_adjustment_mmbbl"] = (
        result["commercial_stock_change_mmbbl"]
        - result["fundamental_balance_mmbbl"]
    )
    return result


def _validate_balance(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "commercial_stock_change_mmbbl", *FORECAST_COMPONENTS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Crude balance missing required columns: {missing}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        raise ValueError("Crude balance contains invalid dates.")
    if data["date"].dt.dayofweek.ne(4).any():
        raise ValueError("Crude balance must use Friday week-ending dates.")
    if data["date"].duplicated().any():
        raise ValueError("Crude balance contains duplicate dates.")
    for column in required - {"date"}:
        converted = pd.to_numeric(data[column], errors="coerce")
        if (converted.isna() & data[column].notna()).any():
            raise ValueError(f"Crude balance contains invalid values in {column}.")
        if not np.isfinite(converted.dropna()).all():
            raise ValueError(f"Crude balance contains non-finite values in {column}.")
        data[column] = converted
    data = data.sort_values("date").reset_index(drop=True)
    initially_nullable = {
        "commercial_stock_change_mmbbl",
        "spr_stock_change_mmbbl",
        "balance_adjustment_mmbbl",
    }
    for column in required - {"date"}:
        missing_values = data[column].isna()
        if column in initially_nullable:
            missing_values = missing_values.iloc[1:]
        if missing_values.any():
            raise ValueError(f"Crude balance contains missing values in {column}.")
        if column in NONNEGATIVE_COMPONENTS and data[column].lt(0).any():
            raise ValueError(f"Crude balance contains negative values in {column}.")
    if len(data) > 1 and not data["date"].diff().dropna().eq(
        pd.Timedelta(days=7)
    ).all():
        raise ValueError("Crude balance must contain consecutive weekly rows.")
    return data


@dataclass
class OilFundamentalsModel:
    """One-week crude-stock model built from forecast physical components."""

    lookback_years: int = 5
    recent_weeks: int = 4
    _history: pd.DataFrame = field(init=False, repr=False)

    def fit(self, history: pd.DataFrame) -> OilFundamentalsModel:
        if self.lookback_years < 1 or self.recent_weeks < 1:
            raise ValueError("lookback_years and recent_weeks must be positive.")
        self._history = _validate_balance(history)
        if self._history.empty:
            raise ValueError("Cannot fit oil fundamentals model on empty history.")
        return self

    def _forecast_component(
        self,
        history: pd.DataFrame,
        component: str,
        target_date: pd.Timestamp,
    ) -> float:
        values = history[["date", component]].dropna().tail(self.lookback_years * 53)
        if values.empty:
            raise ValueError(f"No history available for {component}.")
        weeks = values["date"].dt.isocalendar().week.astype(int)
        seasonal = values.assign(_week=weeks).groupby("_week")[component].mean()
        target_week = int(target_date.isocalendar().week)
        base = float(seasonal.get(target_week, values[component].mean()))
        recent = values.tail(self.recent_weeks).copy()
        recent_weeks = recent["date"].dt.isocalendar().week.astype(int)
        expected = recent_weeks.map(seasonal).fillna(base).to_numpy(dtype=float)
        level_adjustment = float((recent[component].to_numpy() - expected).mean())
        forecast = base + level_adjustment
        return max(forecast, 0.0) if component in NONNEGATIVE_COMPONENTS else forecast

    def predict(self, target_dates: Sequence[object]) -> pd.DataFrame:
        if not hasattr(self, "_history"):
            raise ValueError("Fit the oil fundamentals model before predicting.")
        rows: list[dict[str, object]] = []
        for value in target_dates:
            target_date = pd.Timestamp(value).normalize()
            eligible = self._history.loc[self._history["date"] < target_date]
            if eligible.empty:
                raise ValueError(f"No history is available before {target_date.date()}.")
            forecasts = {
                component: self._forecast_component(eligible, component, target_date)
                for component in FORECAST_COMPONENTS
            }
            fundamental = (
                forecasts["production_mmbbl"]
                + forecasts["imports_mmbbl"]
                - forecasts["refinery_inputs_mmbbl"]
                - forecasts["exports_mmbbl"]
                - forecasts["spr_stock_change_mmbbl"]
            )
            row: dict[str, object] = {
                "forecast_origin": eligible["date"].max(),
                "date": target_date,
                "model": "seasonal_level_fundamentals",
                "fundamental_balance_forecast_mmbbl": fundamental,
                "prediction_mmbbl": fundamental
                + forecasts["balance_adjustment_mmbbl"],
                "last_change_baseline_mmbbl": eligible[
                    "commercial_stock_change_mmbbl"
                ].dropna().iloc[-1],
            }
            row.update(
                {
                    f"{component.removesuffix('_mmbbl')}_forecast_mmbbl": forecast
                    for component, forecast in forecasts.items()
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)


def forecast_next_week(balance: pd.DataFrame, **model_kwargs: object) -> pd.DataFrame:
    data = _validate_balance(balance)
    target_date = data["date"].max() + pd.Timedelta(days=7)
    return OilFundamentalsModel(**model_kwargs).fit(data).predict([target_date])
