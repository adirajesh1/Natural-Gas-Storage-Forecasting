from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import pandas as pd
import requests

from power_forecast.config import load_local_env

from power_forecast.schemas import ERCOT_GEOGRAPHY, find_column, normalized_name


EIA930_BASE = "https://api.eia.gov/v2/electricity/rto"


def normalize_eia930(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize EIA-930 ERCOT actuals to one row per UTC hour."""
    data = frame.copy()
    region_col = find_column(data, ["respondent", "region", "balancing_authority"], required=False)
    if region_col:
        mask = data[region_col].astype(str).str.upper().isin({"ERCO", "ERCOT"})
        data = data.loc[mask].copy()
    time_col = find_column(data, ["valid_at", "period", "timestamp", "datetime"])
    data["valid_at"] = pd.to_datetime(data[time_col], utc=True, errors="coerce")
    if data["valid_at"].isna().any():
        raise ValueError("EIA-930 data contains invalid hourly timestamps.")

    long_type = find_column(data, ["type_name", "type", "fueltype", "fuel_type"], required=False)
    value_col = find_column(data, ["value", "mw", "generation_mw"], required=False)
    if long_type and value_col:
        values = pd.to_numeric(data[value_col], errors="coerce")
        pivot = data.assign(_value=values).pivot_table(
            index="valid_at", columns=long_type, values="_value", aggfunc="sum"
        )
        pivot.columns = [str(column).lower().replace(" ", "_") for column in pivot.columns]
        data = pivot.reset_index()

    aliases = {
        "load_actual_mw": ["load_actual_mw", "demand", "actual_demand", "load"],
        "wind_actual_mw": ["wind_actual_mw", "wind"],
        "solar_actual_mw": ["solar_actual_mw", "solar"],
        "gas_generation_actual_mw": ["gas_generation_actual_mw", "natural_gas", "naturalgas", "gas"],
        "coal_generation_actual_mw": ["coal_generation_actual_mw", "coal"],
        "nuclear_actual_mw": ["nuclear_actual_mw", "nuclear"],
        "hydro_actual_mw": ["hydro_actual_mw", "hydro", "hydroelectric"],
        "other_nonthermal_actual_mw": ["other_nonthermal_actual_mw", "other", "other_energy_sources"],
        "net_imports_actual_mw": [
            "net_imports_actual_mw",
            "net_imports",
            "total_interchange",
            "interchange",
        ],
        "battery_net_discharge_actual_mw": ["battery_net_discharge_actual_mw", "battery", "battery_storage"],
    }
    result = data[["valid_at"]].copy()
    for output, candidates in aliases.items():
        column = find_column(data, candidates, required=False)
        values = pd.to_numeric(data[column], errors="coerce") if column else np.nan
        if output == "net_imports_actual_mw" and column is not None:
            if normalized_name(column) in {"totalinterchange", "interchange"}:
                values = -values
        result[output] = values
    result["geography"] = ERCOT_GEOGRAPHY
    return result.groupby(["valid_at", "geography"], as_index=False).last()


@dataclass
class Eia930Client:
    api_key: str | None = None
    timeout: float = 30.0

    def __post_init__(self) -> None:
        load_local_env()
        self.api_key = self.api_key or os.getenv("EIA_API_KEY")
        if not self.api_key:
            raise ValueError("EIA-930 access requires EIA_API_KEY.")

    def _fetch(self, route: str, *, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        params: dict[str, Any] = {
            "api_key": self.api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": "ERCO",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        records: list[dict[str, Any]] = []
        while True:
            response = requests.get(
                f"{EIA930_BASE}/{route}/data/", params=params, timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json().get("response", {})
            page = payload.get("data", [])
            records.extend(page)
            if len(page) < params["length"]:
                break
            params["offset"] += params["length"]
        return pd.DataFrame(records)

    def fetch_actuals(self, *, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        fuel = self._fetch("fuel-type-data", start=start, end=end)
        region = self._fetch("region-data", start=start, end=end)
        fuel_actuals = normalize_eia930(fuel) if not fuel.empty else pd.DataFrame()
        region_actuals = normalize_eia930(region) if not region.empty else pd.DataFrame()
        if fuel_actuals.empty:
            return region_actuals
        if region_actuals.empty:
            return fuel_actuals
        merged = fuel_actuals.merge(
            region_actuals,
            on=["valid_at", "geography"],
            how="outer",
            suffixes=("", "_region"),
        )
        for regional_column in [column for column in merged if column.endswith("_region")]:
            base_column = regional_column.removesuffix("_region")
            if base_column in merged:
                merged[base_column] = merged[base_column].combine_first(
                    merged[regional_column]
                )
                merged = merged.drop(columns=regional_column)
            else:
                merged = merged.rename(columns={regional_column: base_column})
        return merged
