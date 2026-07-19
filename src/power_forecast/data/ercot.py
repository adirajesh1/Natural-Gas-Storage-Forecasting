from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from power_forecast.config import load_local_env

from power_forecast.schemas import (
    ERCOT_GEOGRAPHY,
    find_column,
    frame_hash,
    validate_vintage_frame,
)


ERCOT_PRODUCTS = {
    "load": "NP3-560-CD",
    "load_actual": "NP6-345-CD",
    "wind": "NP4-732-CD",
    "solar": "NP4-745-CD",
    "outages": "NP3-233-CD",
    "adequacy": "NP3-763-CD",
}


def _payload_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return records from both current Public API and legacy HAL responses."""
    fields = payload.get("fields") or []
    rows = payload.get("data")
    if isinstance(rows, list):
        if not rows:
            return []
        if isinstance(rows[0], dict):
            return rows
        names = [field.get("name") for field in fields if isinstance(field, dict)]
        if names:
            return [dict(zip(names, row, strict=False)) for row in rows]

    embedded = payload.get("_embedded", {})
    return next((value for value in embedded.values() if isinstance(value, list)), [])


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _ercot_datetime_series(values: pd.Series) -> pd.Series:
    """Parse API datetimes, whose offset-free values are ERCOT local time."""
    timestamps = values.map(pd.Timestamp)
    aware = timestamps.map(lambda value: value is not pd.NaT and value.tzinfo is not None)
    if aware.any() and not aware.all():
        raise ValueError("ERCOT timestamps cannot mix offset-aware and offset-free values.")
    parsed = pd.to_datetime(values, errors="coerce", utc=bool(aware.all()))
    if parsed.isna().any():
        raise ValueError("ERCOT data contains invalid timestamps.")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(
            "America/Chicago",
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    return parsed.dt.tz_convert("UTC")


def _local_hour_end(day: pd.Series, hour_ending: pd.Series) -> pd.Series:
    days = pd.to_datetime(day, errors="coerce")
    hours = pd.to_numeric(hour_ending, errors="coerce")
    if hours.isna().any():
        hour_text = hour_ending.astype(str).str.extract(r"^\s*(\d{1,2})(?::\d{2})?\s*$", expand=False)
        hours = hours.fillna(pd.to_numeric(hour_text, errors="coerce"))
    if days.isna().any() or hours.isna().any():
        raise ValueError("Invalid ERCOT delivery date or hour-ending value.")
    naive = days + pd.to_timedelta(hours, unit="h")
    try:
        local = naive.dt.tz_localize(
            "America/Chicago",
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    except pd.errors.AmbiguousTimeError as exc:
        raise ValueError(
            "Ambiguous ERCOT local hours require ordered duplicate rows or an explicit UTC timestamp."
        ) from exc
    return local.dt.tz_convert("UTC")


def _valid_times(frame: pd.DataFrame) -> pd.Series:
    timestamp_col = find_column(
        frame,
        ["valid_at", "delivery_hour", "timestamp", "datetime", "delivery_datetime"],
        required=False,
    )
    if timestamp_col:
        return _ercot_datetime_series(frame[timestamp_col])
    day_col = find_column(
        frame,
        ["delivery_date", "deliverydate", "operating_date", "operating_day", "operatingday", "date"],
    )
    hour_col = find_column(frame, ["hour_ending", "hourending", "he", "hour"])
    return _local_hour_end(frame[day_col], frame[hour_col])


def _issued_times(
    frame: pd.DataFrame,
    *,
    issued_at: object | None,
    retrieved_at: pd.Timestamp,
) -> pd.Series:
    column = find_column(
        frame,
        ["issued_at", "posted_at", "posted_datetime", "posteddatetime", "publish_time", "created_at"],
        required=False,
    )
    if column:
        return _ercot_datetime_series(frame[column])
    if issued_at is None:
        raise ValueError(
            "ERCOT source did not include a publication timestamp; pass issued_at explicitly."
        )
    return pd.Series([_utc(issued_at)] * len(frame), index=frame.index)


def _system_rows(frame: pd.DataFrame) -> pd.DataFrame:
    geography_col = find_column(
        frame,
        ["geography", "region", "zone", "forecast_zone", "weather_zone", "load_zone"],
        required=False,
    )
    if geography_col is None:
        return frame.copy()
    normalized = frame[geography_col].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    mask = normalized.isin({"ERCOT", "TOTAL", "SYSTEM", "SYSTEMWIDE", "ERCOTTOTAL"})
    return frame.loc[mask].copy() if mask.any() else frame.copy()


def normalize_component_product(
    frame: pd.DataFrame,
    *,
    component: str,
    product_id: str,
    issued_at: object | None = None,
    retrieved_at: object | None = None,
) -> pd.DataFrame:
    """Normalize an ERCOT load/wind/solar report to one vintage contract."""
    if component not in {"load", "wind", "solar"}:
        raise ValueError("component must be load, wind, or solar.")
    data = _system_rows(frame)
    retrieved = _utc(retrieved_at or datetime.now(timezone.utc))
    baseline_aliases = {
        "load": ["baseline_mw", "forecast_mw", "system_total", "load_forecast", "forecast", "mw"],
        "wind": ["baseline_mw", "stwpf_system_wide", "stwpf", "wgrpp_system_wide", "wgrpp", "wind_forecast", "forecast", "mw"],
        "solar": ["baseline_mw", "stppf_system_wide", "stppf", "pvgrpp_system_wide", "pvgrpp", "solar_forecast", "forecast", "mw"],
    }
    actual_aliases = {
        "load": ["actual_mw", "actual_load", "system_load", "actual"],
        "wind": ["actual_mw", "gen_system_wide", "gen", "wind_generation", "actual"],
        "solar": ["actual_mw", "gen_system_wide", "gen", "solar_generation", "actual"],
    }
    baseline_col = find_column(data, baseline_aliases[component])
    actual_col = find_column(data, actual_aliases[component], required=False)
    result = pd.DataFrame(index=data.index)
    result["valid_at"] = _valid_times(data)
    result["issued_at"] = _issued_times(data, issued_at=issued_at, retrieved_at=retrieved)
    result["retrieved_at"] = retrieved
    result["source"] = "ERCOT"
    result["product_id"] = product_id.upper()
    result["geography"] = ERCOT_GEOGRAPHY
    result["source_hash"] = frame_hash(frame)
    result["component"] = component
    result["baseline_mw"] = pd.to_numeric(data[baseline_col], errors="coerce")
    result["actual_mw"] = (
        pd.to_numeric(data[actual_col], errors="coerce") if actual_col else np.nan
    )
    if result["baseline_mw"].isna().any():
        raise ValueError(f"ERCOT {component} data contains invalid baseline values.")
    result = validate_vintage_frame(result)
    keys = ["product_id", "issued_at", "valid_at", "component"]
    if result.duplicated(keys).any():
        provenance = [
            "source",
            "product_id",
            "issued_at",
            "retrieved_at",
            "valid_at",
            "geography",
            "source_hash",
            "component",
        ]
        result = result.groupby(provenance, as_index=False, dropna=False).agg(
            baseline_mw=("baseline_mw", "sum"),
            actual_mw=("actual_mw", lambda values: values.sum(min_count=1)),
        )
    return result.drop_duplicates(subset=keys, keep="last")


def normalize_actual_load(
    frame: pd.DataFrame,
    *,
    issued_at: object | None = None,
    retrieved_at: object | None = None,
) -> pd.DataFrame:
    """Normalize the ERCOT actual-system-load report as a revisable vintage."""
    data = _system_rows(frame)
    retrieved = _utc(retrieved_at or datetime.now(timezone.utc))
    value_col = find_column(
        data,
        ["actual_mw", "actual_load", "system_load", "system_total", "total", "load", "mw"],
    )
    result = pd.DataFrame(index=data.index)
    result["valid_at"] = _valid_times(data)
    result["issued_at"] = _issued_times(data, issued_at=issued_at, retrieved_at=retrieved)
    result["retrieved_at"] = retrieved
    result["source"] = "ERCOT"
    result["product_id"] = ERCOT_PRODUCTS["load_actual"]
    result["geography"] = ERCOT_GEOGRAPHY
    result["source_hash"] = frame_hash(frame)
    result["load_actual_mw"] = pd.to_numeric(data[value_col], errors="coerce")
    if result["load_actual_mw"].isna().any():
        raise ValueError("ERCOT actual load contains invalid values.")
    return validate_vintage_frame(result).drop_duplicates(
        subset=["product_id", "issued_at", "valid_at"], keep="last"
    )


def _normalize_capacity_product(
    frame: pd.DataFrame,
    *,
    product_id: str,
    value_aliases: dict[str, list[str]],
    issued_at: object | None,
    retrieved_at: object | None,
) -> pd.DataFrame:
    data = _system_rows(frame)
    retrieved = _utc(retrieved_at or datetime.now(timezone.utc))
    result = pd.DataFrame(index=data.index)
    result["valid_at"] = _valid_times(data)
    result["issued_at"] = _issued_times(data, issued_at=issued_at, retrieved_at=retrieved)
    result["retrieved_at"] = retrieved
    result["source"] = "ERCOT"
    result["product_id"] = product_id.upper()
    result["geography"] = ERCOT_GEOGRAPHY
    result["source_hash"] = frame_hash(frame)
    found_value = False
    for output, aliases in value_aliases.items():
        column = find_column(data, aliases, required=False)
        found_value = found_value or column is not None
        result[output] = pd.to_numeric(data[column], errors="coerce") if column else 0.0
    if not found_value:
        raise ValueError(f"ERCOT product {product_id} has no recognized capacity values.")
    result = validate_vintage_frame(result)
    keys = ["product_id", "issued_at", "valid_at"]
    if result.duplicated(keys).any():
        provenance = [
            "source",
            "product_id",
            "issued_at",
            "retrieved_at",
            "valid_at",
            "geography",
            "source_hash",
        ]
        aggregations = {column: "sum" for column in value_aliases}
        result = result.groupby(provenance, as_index=False, dropna=False).agg(aggregations)
    return result.drop_duplicates(subset=keys, keep="last")


def normalize_outages(
    frame: pd.DataFrame,
    *,
    issued_at: object | None = None,
    retrieved_at: object | None = None,
) -> pd.DataFrame:
    data = frame.copy()
    groups = {
        "conventional_outage_mw": "totalresourcemwzone",
        "renewable_outage_mw": "totalirrmwzone",
        "new_equipment_outage_mw": "totalnewequipmentresourcemwzone",
    }
    normalized_columns = {
        str(column).lower().replace("_", ""): column for column in data.columns
    }
    for output, prefix in groups.items():
        matches = [
            column
            for normalized, column in normalized_columns.items()
            if normalized.startswith(prefix)
        ]
        # The live API abbreviates New Equipment as NewEquip.
        if output == "new_equipment_outage_mw" and not matches:
            matches = [
                column
                for normalized, column in normalized_columns.items()
                if normalized.startswith("totalnewequipresourcemwzone")
            ]
        if matches:
            data[output] = data[matches].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    return _normalize_capacity_product(
        data,
        product_id=ERCOT_PRODUCTS["outages"],
        issued_at=issued_at,
        retrieved_at=retrieved_at,
        value_aliases={
            "conventional_outage_mw": ["conventional_outage_mw", "total_resource_out_mw", "resource_outage", "total_outage"],
            "renewable_outage_mw": ["renewable_outage_mw", "irr_out_mw", "irr_outage"],
            "new_equipment_outage_mw": ["new_equipment_outage_mw", "new_equip_out_mw", "new_equipment_outage"],
        },
    )


def normalize_adequacy(
    frame: pd.DataFrame,
    *,
    issued_at: object | None = None,
    retrieved_at: object | None = None,
) -> pd.DataFrame:
    return _normalize_capacity_product(
        frame,
        product_id=ERCOT_PRODUCTS["adequacy"],
        issued_at=issued_at,
        retrieved_at=retrieved_at,
        value_aliases={
            "available_capacity_mw": ["available_capacity_mw", "avail_cap_gen", "total_cap_gen", "total_generation_capacity", "available_generation"],
        },
    )


@dataclass
class ErcotApiClient:
    """Small client for ERCOT's authenticated public-report API."""

    subscription_key: str | None = None
    id_token: str | None = None
    base_url: str = "https://api.ercot.com/api/public-reports"
    timeout: float = 30.0
    page_size: int = 5000
    min_request_interval: float = 2.05
    max_retries: int = 5
    _last_request_at: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        load_local_env()
        self.subscription_key = self.subscription_key or os.getenv("ERCOT_SUBSCRIPTION_KEY")
        self.id_token = self.id_token or os.getenv("ERCOT_ID_TOKEN")
        if not self.subscription_key or not self.id_token:
            raise ValueError(
                "ERCOT Public API requires ERCOT_SUBSCRIPTION_KEY and ERCOT_ID_TOKEN."
            )

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.id_token}",
            "Ocp-Apim-Subscription-Key": str(self.subscription_key),
        }

    def _get(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
            self._last_request_at = time.monotonic()
            response = requests.get(
                url, headers=self.headers, params=params, timeout=self.timeout
            )
            retryable = response.status_code == 429 or response.status_code in {
                500,
                502,
                503,
                504,
            }
            if retryable and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else 2.0 ** (attempt + 1)
                except ValueError:
                    delay = 2.0 ** (attempt + 1)
                time.sleep(max(self.min_request_interval, min(delay, 60.0)))
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("ERCOT request retry loop exited unexpectedly.")

    def product_metadata(self, product_id: str) -> dict[str, Any]:
        return self._get(f"{self.base_url}/{product_id.lower()}")

    def fetch_product_records(
        self,
        product_id: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        metadata = self.product_metadata(product_id)
        artifacts = metadata.get("artifacts") or metadata.get("_embedded", {}).get("artifacts") or []
        if not artifacts:
            for value in metadata.get("_embedded", {}).values():
                candidates = value if isinstance(value, list) else [value]
                product = next(
                    (
                        candidate
                        for candidate in candidates
                        if isinstance(candidate, dict) and candidate.get("artifacts")
                    ),
                    None,
                )
                if product:
                    metadata = product
                    artifacts = product["artifacts"]
                    break
        if not artifacts:
            raise ValueError(f"ERCOT product {product_id} exposes no API artifact.")
        url = artifacts[0].get("_links", {}).get("endpoint", {}).get("href")
        if not url:
            raise ValueError(f"ERCOT product {product_id} artifact has no endpoint.")
        records: list[dict[str, Any]] = []
        request_params = dict(params or {})
        request_params.setdefault("size", self.page_size)
        while url:
            payload = self._get(url, params=request_params)
            records.extend(_payload_records(payload))
            meta = payload.get("_meta", {})
            current_page = int(meta.get("currentPage") or 1)
            total_pages = int(meta.get("totalPages") or 1)
            if current_page < total_pages:
                request_params["page"] = current_page + 1
                continue
            next_url = payload.get("_links", {}).get("next", {}).get("href")
            if next_url:
                url = next_url
                request_params = {}
                continue
            break
        frame = pd.DataFrame(records)
        if not frame.empty and not any(
            str(column).lower() in {"issued_at", "posted_at", "posteddatetime", "posted_datetime"}
            for column in frame.columns
        ):
            posted = metadata.get("lastPostDatetime") or metadata.get("last_post_datetime")
            if posted:
                frame["issued_at"] = posted
        return frame
