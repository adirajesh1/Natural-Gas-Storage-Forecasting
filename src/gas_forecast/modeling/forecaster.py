from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from gas_forecast.data.weather_scenarios import select_weather_scenario_as_of

ForecastInputMode = Literal["seasonal", "observed", "scenario"]

# These are the columns whose future values can be reconstructed from the
# forecast state. Keep this list explicit so recursive forecasts never silently
# consume a feature that is only available after the target week has closed.
RECURSIVE_FEATURE_COLUMNS = frozenset(
    {
        "week_sin",
        "week_cos",
        "month_sin",
        "month_cos",
        "hdd",
        "cdd",
        "hdd_lag1",
        "hdd_rolling_4wk",
        "cdd_rolling_4wk",
        "weekly_change_lag1",
        "weekly_change_rolling_4wk",
        "storage_vs_5yr_avg",
        "storage_vs_last_year",
        "is_injection_season",
        "net_inflow_balancing_lag1",
        "net_inflow_balancing_rolling_4wk",
        "local_balance_lag1",
    }
)

_BALANCE_FEATURE_COLUMNS = frozenset(
    {
        "net_inflow_balancing_lag1",
        "net_inflow_balancing_rolling_4wk",
        "local_balance_lag1",
    }
)


@dataclass
class _ForecastState:
    """Mutable history used to construct one recursive forecast step."""

    storage_bcf: float
    hdd_history: list[float]
    cdd_history: list[float]
    weekly_change_history: list[float]
    storage_by_date: dict[pd.Timestamp, float]
    local_balance_history: list[float]
    net_inflow_history: list[float]


@dataclass(frozen=True)
class _WeatherScenario:
    """Target-week weather inputs used by one recursive forecast step."""

    temperature_f: float
    hdd: float
    cdd: float
    weather_days: float


def _week_of_year(date: pd.Timestamp) -> int:
    return int(date.isocalendar().week)


def _seasonal_profiles(
    history: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Average available historical inputs by ISO week of year."""
    return history.groupby("week_of_year")[columns].mean()


def _seasonal_value(
    profile: pd.DataFrame,
    column: str,
    week_of_year: int,
    *,
    fallback: float,
) -> float:
    """Read a seasonal value, falling back to the profile-wide mean."""
    if week_of_year in profile.index:
        value = profile.loc[week_of_year, column]
        if pd.notna(value):
            return float(value)

    available = profile[column].dropna()
    if not available.empty:
        return float(available.mean())
    return fallback


def _observed_value(
    observed: pd.Series | None,
    column: str,
) -> float | None:
    """Return a non-null realized value from the target row, if present."""
    if observed is None or column not in observed:
        return None
    value = observed[column]
    if pd.isna(value):
        return None
    return float(value)


def _observed_or_default(
    observed: pd.Series | None,
    column: str,
    default: float,
) -> float:
    """Prefer an observed scenario value without treating zero as missing."""
    value = _observed_value(observed, column)
    return default if value is None else value


def _scenario_value(scenario: pd.Series | None, column: str) -> float:
    """Return a required value from an externally supplied weather scenario."""
    value = _observed_value(scenario, column)
    if value is None:
        raise ValueError(f"Weather scenario is missing a usable {column!r} value.")
    return value


def _trailing_mean(values: list[float], window: int) -> float:
    """Return the mean of the available trailing history."""
    if not values:
        raise ValueError("Cannot construct a lag feature without historical values.")
    return float(np.mean(values[-window:]))


def _same_week_storage_average(
    history: pd.DataFrame,
    week_of_year: int,
    *,
    fallback: float,
    years: int = 5,
    before_date: pd.Timestamp | None = None,
    date_col: str = "date",
) -> float:
    """Return the trailing same-week storage average known at forecast origin."""
    eligible = history["week_of_year"] == week_of_year
    if before_date is not None:
        eligible &= history[date_col] < before_date
    values = history.loc[eligible, "storage_bcf"].dropna()
    if values.empty:
        return fallback
    return float(values.tail(years).mean())


def _weather_scenario(
    weather_profile: pd.DataFrame,
    observed: pd.Series | None,
    scenario: pd.Series | None,
    *,
    week_of_year: int,
    input_mode: ForecastInputMode,
    temp_anomaly: float,
) -> _WeatherScenario:
    """Build target-week weather from a seasonal profile or observed diagnostic."""
    temperature_f = _seasonal_value(
        weather_profile,
        "temperature_f",
        week_of_year,
        fallback=50.0,
    )
    hdd = _seasonal_value(weather_profile, "hdd", week_of_year, fallback=0.0)
    cdd = _seasonal_value(weather_profile, "cdd", week_of_year, fallback=0.0)
    weather_days = _seasonal_value(
        weather_profile,
        "weather_days",
        week_of_year,
        fallback=7.0,
    )

    if input_mode == "scenario":
        temperature_f = _scenario_value(scenario, "temperature_f")
        hdd = _scenario_value(scenario, "hdd")
        cdd = _scenario_value(scenario, "cdd")
        weather_days = _scenario_value(scenario, "weather_days")
    elif input_mode == "observed":
        temperature_f = _observed_or_default(observed, "temperature_f", temperature_f)
        hdd = _observed_or_default(observed, "hdd", hdd)
        cdd = _observed_or_default(observed, "cdd", cdd)
        weather_days = _observed_or_default(observed, "weather_days", weather_days)

    weather_days = max(weather_days, 1.0)
    if temp_anomaly:
        adjusted_temperature = temperature_f + temp_anomaly
        hdd = max(0.0, 65.0 - adjusted_temperature) * weather_days
        cdd = max(0.0, adjusted_temperature - 65.0) * weather_days

    return _WeatherScenario(
        temperature_f=temperature_f,
        hdd=hdd,
        cdd=cdd,
        weather_days=weather_days,
    )


def _local_balance_scenario(
    balance_profile: pd.DataFrame | None,
    observed: pd.Series | None,
    *,
    week_of_year: int,
    input_mode: ForecastInputMode,
) -> float:
    """Build the optional target-week local-balance scenario."""
    if balance_profile is None:
        return 0.0

    local_balance = _seasonal_value(
        balance_profile,
        "local_balance",
        week_of_year,
        fallback=0.0,
    )
    if input_mode == "observed":
        return _observed_or_default(observed, "local_balance", local_balance)
    return local_balance


def _recursive_feature_values(
    state: _ForecastState,
    history: pd.DataFrame,
    *,
    target_date: pd.Timestamp,
    weather: _WeatherScenario,
    needs_balance_features: bool,
    date_col: str,
) -> dict[str, float | int]:
    """Rebuild all supported model inputs before predicting one target week."""
    week_of_year = _week_of_year(target_date)
    previous_week = target_date - pd.Timedelta(weeks=1)
    storage_5yr_average = _same_week_storage_average(
        history,
        _week_of_year(previous_week),
        fallback=state.storage_bcf,
        before_date=previous_week,
        date_col=date_col,
    )

    prev_iso = previous_week.isocalendar()
    last_year_date = next(
        (
            d
            for d in state.storage_by_date.keys()
            if d.isocalendar().year == prev_iso.year - 1
            and d.isocalendar().week == prev_iso.week
        ),
        None,
    )
    if last_year_date is None:
        last_year_date = target_date - pd.Timedelta(weeks=53)

    last_year_storage = state.storage_by_date.get(last_year_date)
    if last_year_storage is None:
        last_year_storage = _same_week_storage_average(
            history,
            _week_of_year(last_year_date),
            fallback=state.storage_bcf,
            before_date=previous_week,
            date_col=date_col,
        )

    week_angle = 2 * np.pi * week_of_year / 52.0
    month_angle = 2 * np.pi * target_date.month / 12.0
    return {
        "week_sin": np.sin(week_angle),
        "week_cos": np.cos(week_angle),
        "month_sin": np.sin(month_angle),
        "month_cos": np.cos(month_angle),
        "hdd": weather.hdd,
        "cdd": weather.cdd,
        "hdd_lag1": state.hdd_history[-1],
        "hdd_rolling_4wk": _trailing_mean(state.hdd_history, 4),
        "cdd_rolling_4wk": _trailing_mean(state.cdd_history, 4),
        "weekly_change_lag1": state.weekly_change_history[-1],
        "weekly_change_rolling_4wk": _trailing_mean(
            state.weekly_change_history,
            4,
        ),
        "storage_vs_5yr_avg": state.storage_bcf - storage_5yr_average,
        "storage_vs_last_year": state.storage_bcf - last_year_storage,
        "is_injection_season": int(4 <= target_date.month <= 10),
        "net_inflow_balancing_lag1": (
            state.net_inflow_history[-1] if needs_balance_features else 0.0
        ),
        "net_inflow_balancing_rolling_4wk": (
            _trailing_mean(state.net_inflow_history, 4)
            if needs_balance_features
            else 0.0
        ),
        "local_balance_lag1": (
            state.local_balance_history[-1] if needs_balance_features else 0.0
        ),
    }


def _advance_state(
    state: _ForecastState,
    *,
    target_date: pd.Timestamp,
    weather: _WeatherScenario,
    predicted_change: float,
    local_balance: float,
    needs_balance_features: bool,
) -> float:
    """Append simulated inputs and return the next projected storage level."""
    projected_storage = state.storage_bcf + predicted_change
    state.hdd_history.append(weather.hdd)
    state.cdd_history.append(weather.cdd)
    state.weekly_change_history.append(predicted_change)
    state.storage_bcf = projected_storage
    state.storage_by_date[target_date] = projected_storage
    if needs_balance_features:
        state.local_balance_history.append(local_balance)
        state.net_inflow_history.append(predicted_change - local_balance)
    return projected_storage


class RecursiveForecaster:
    """Simulate multi-week storage projections from a fitted sklearn-style model.

    The default ``forecast_input_mode="seasonal"`` uses only observations before
    ``start_date`` to form target-week weather and local-balance scenarios.
    ``"scenario"`` uses a versioned, as-of weather forecast table, while
    ``"observed"`` is only a diagnostic that intentionally assumes realized
    future inputs are available.
    """

    def __init__(
        self,
        model,
        feature_cols: list[str] | tuple[str, ...],
        *,
        date_col: str = "date",
        target_col: str = "weekly_change_bcf",
        model_key: str | None = None,
    ) -> None:
        self.model = model
        self.feature_cols = list(feature_cols)
        self.date_col = date_col
        self.target_col = target_col
        self.model_key = model_key or type(model).__name__

    def _validate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        unsupported = sorted(set(self.feature_cols) - RECURSIVE_FEATURE_COLUMNS)
        if unsupported:
            raise ValueError(
                "RecursiveForecaster cannot reconstruct future values for "
                f"feature columns: {unsupported}"
            )

        required = {
            self.date_col,
            self.target_col,
            "storage_bcf",
            "temperature_f",
            "hdd",
            "cdd",
            "weather_days",
            "week_of_year",
        }
        if set(self.feature_cols) & _BALANCE_FEATURE_COLUMNS:
            required.add("local_balance")

        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing required forecast columns: {missing}")

        data = frame.copy()
        data[self.date_col] = pd.to_datetime(data[self.date_col])
        data = data.sort_values(self.date_col).reset_index(drop=True)

        if data[self.date_col].duplicated().any():
            raise ValueError("Recursive forecasting requires one row per date.")
        if "duoarea" in data.columns and data["duoarea"].nunique(dropna=True) > 1:
            raise ValueError(
                "Recursive forecasting requires one region at a time. "
                "Filter the feature table before forecasting."
            )

        if "net_inflow_balancing" not in data.columns and {
            "net_inflow_balancing_lag1",
            "net_inflow_balancing_rolling_4wk",
        } & set(self.feature_cols):
            data["net_inflow_balancing"] = (
                data[self.target_col] - data["local_balance"]
            )

        return data

    def _initial_state(
        self,
        history: pd.DataFrame,
        *,
        needs_balance_features: bool,
    ) -> _ForecastState:
        actual_storage = history.dropna(subset=["storage_bcf"])
        if actual_storage.empty:
            raise ValueError("No historical storage is available before start_date.")

        hdd_history = history["hdd"].dropna().astype(float).tolist()
        cdd_history = history["cdd"].dropna().astype(float).tolist()
        change_history = history[self.target_col].dropna().astype(float).tolist()
        if not hdd_history or not cdd_history or not change_history:
            raise ValueError(
                "Historical weather and weekly storage changes are required before start_date."
            )

        storage_by_date = {
            pd.Timestamp(date): float(storage)
            for date, storage in actual_storage[[self.date_col, "storage_bcf"]].itertuples(
                index=False,
                name=None,
            )
        }

        local_balance_history: list[float] = []
        net_inflow_history: list[float] = []
        if needs_balance_features:
            local_balance_history = (
                history["local_balance"].dropna().astype(float).tolist()
            )
            net_inflow_history = (
                history["net_inflow_balancing"].dropna().astype(float).tolist()
            )
            if not local_balance_history or not net_inflow_history:
                raise ValueError(
                    "Balance lag features require historical local_balance and "
                    "net_inflow_balancing values."
                )

        return _ForecastState(
            storage_bcf=float(actual_storage.iloc[-1]["storage_bcf"]),
            hdd_history=hdd_history,
            cdd_history=cdd_history,
            weekly_change_history=change_history,
            storage_by_date=storage_by_date,
            local_balance_history=local_balance_history,
            net_inflow_history=net_inflow_history,
        )

    def predict_horizon(
        self,
        features_df: pd.DataFrame,
        start_date: pd.Timestamp | str,
        horizon_weeks: int,
        temp_anomaly: float = 0.0,
        forecast_input_mode: ForecastInputMode = "seasonal",
        weather_scenario: pd.DataFrame | None = None,
        region: str | None = None,
        as_of: pd.Timestamp | str | None = None,
        weather_provider: str | None = None,
        weather_model: str | None = None,
        reconciliation_method: str = "unreconciled",
    ) -> pd.DataFrame:
        """Project weekly changes and storage levels from ``start_date`` forward.

        ``start_date`` must be the first Friday week-ending date to forecast.
        Historical rows before that date establish the state. Rows at and after
        that date are used only for optional observed-input diagnostics and for
        returned actuals; seasonal mode never reads their feature values. In
        ``"scenario"`` mode, the supplied weekly weather forecasts are filtered
        by their ``issued_at`` timestamp at ``as_of``. If omitted, ``as_of`` is
        the last historical date before ``start_date``.
        """
        if horizon_weeks < 1:
            raise ValueError("horizon_weeks must be at least 1.")
        if forecast_input_mode not in {"seasonal", "observed", "scenario"}:
            raise ValueError(
                "forecast_input_mode must be 'seasonal', 'observed', or 'scenario'."
            )
        if forecast_input_mode == "scenario" and weather_scenario is None:
            raise ValueError("scenario mode requires a weather_scenario table.")
        if forecast_input_mode != "scenario" and weather_scenario is not None:
            raise ValueError(
                "weather_scenario is only used when forecast_input_mode='scenario'."
            )

        data = self._validate_frame(features_df)
        data_region: str | None = None
        if "duoarea" in data.columns:
            regions = data["duoarea"].dropna().astype(str).unique()
            if len(regions) == 1:
                data_region = regions[0]
        if region is not None and data_region is not None and region != data_region:
            raise ValueError(
                f"Requested scenario region {region!r} does not match feature data "
                f"region {data_region!r}."
            )
        scenario_region = region or data_region
        if forecast_input_mode == "scenario" and scenario_region is None:
            raise ValueError(
                "scenario mode requires region= when the feature table has no duoarea."
            )
        start = pd.Timestamp(start_date).normalize()
        if start.weekday() != 4:
            raise ValueError("start_date must be a Friday week-ending date.")

        history = data.loc[data[self.date_col] < start].copy()
        if history.empty:
            raise ValueError(f"No historical actuals found prior to start date {start}")
        forecast_origin = (
            pd.Timestamp(as_of)
            if as_of is not None
            else pd.Timestamp(history[self.date_col].max())
        )
        if forecast_origin >= start:
            raise ValueError("as_of must precede the first forecast target date.")

        needs_balance_features = bool(set(self.feature_cols) & _BALANCE_FEATURE_COLUMNS)
        state = self._initial_state(
            history,
            needs_balance_features=needs_balance_features,
        )

        weather_profile = _seasonal_profiles(
            history,
            ["temperature_f", "hdd", "cdd", "weather_days"],
        )
        balance_profile = (
            _seasonal_profiles(history, ["local_balance"])
            if needs_balance_features
            else None
        )
        observed_by_date = data.set_index(self.date_col)
        projection_dates = pd.date_range(
            start=start,
            periods=horizon_weeks,
            freq="W-FRI",
        )
        scenario_by_date: pd.DataFrame | None = None
        if forecast_input_mode == "scenario":
            selected_scenario = select_weather_scenario_as_of(
                weather_scenario,
                as_of=forecast_origin,
                region=scenario_region,
                target_dates=projection_dates,
                provider=weather_provider,
                model=weather_model,
            )
            scenario_by_date = selected_scenario.set_index("date")

        projections: list[dict[str, object]] = []
        for horizon, target_date in enumerate(projection_dates, start=1):
            week_of_year = _week_of_year(target_date)
            observed = (
                observed_by_date.loc[target_date]
                if target_date in observed_by_date.index
                else None
            )
            scenario = (
                scenario_by_date.loc[target_date]
                if scenario_by_date is not None
                else None
            )

            weather = _weather_scenario(
                weather_profile,
                observed,
                scenario,
                week_of_year=week_of_year,
                input_mode=forecast_input_mode,
                temp_anomaly=temp_anomaly,
            )
            local_balance = _local_balance_scenario(
                balance_profile,
                observed,
                week_of_year=week_of_year,
                input_mode=forecast_input_mode,
            )
            feature_values = _recursive_feature_values(
                state,
                history,
                target_date=target_date,
                weather=weather,
                needs_balance_features=needs_balance_features,
                date_col=self.date_col,
            )
            feature_row = pd.DataFrame(
                [[feature_values[column] for column in self.feature_cols]],
                columns=self.feature_cols,
            )
            predicted_change = float(self.model.predict(feature_row)[0])
            projected_storage = _advance_state(
                state,
                target_date=target_date,
                weather=weather,
                predicted_change=predicted_change,
                local_balance=local_balance,
                needs_balance_features=needs_balance_features,
            )

            projections.append(
                {
                    "date": target_date,
                    "forecast_origin": forecast_origin,
                    "region": scenario_region,
                    "horizon": horizon,
                    "model_key": self.model_key,
                    "reconciliation_method": reconciliation_method,
                    "weather_provider": (
                        scenario.get("provider", "archived_forecast")
                        if scenario is not None
                        else (
                            "observed_oracle"
                            if forecast_input_mode == "observed"
                            else "seasonal_norm"
                        )
                    ),
                    "weather_model": (
                        scenario.get("model", pd.NA)
                        if scenario is not None
                        else pd.NA
                    ),
                    "weather_run": (
                        scenario.get("model_run", scenario.get("issued_at", pd.NA))
                        if scenario is not None
                        else pd.NA
                    ),
                    "predicted_weekly_change": predicted_change,
                    "p10": np.nan,
                    "p50": predicted_change,
                    "p90": np.nan,
                    "projected_storage": projected_storage,
                    "actual_weekly_change": _observed_value(
                        observed,
                        self.target_col,
                    ),
                    "actual_storage": _observed_value(observed, "storage_bcf"),
                    "week_of_year": week_of_year,
                }
            )

        return pd.DataFrame(projections)
