from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd
import requests

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _date_periods(
    start_date: str,
    end_date: str,
    *,
    freq: str = "YS",
) -> list[tuple[str, str]]:
    """Split a date range into inclusive start/end pairs."""
    range_start = pd.Timestamp(start_date)
    range_end = pd.Timestamp(end_date)

    if freq != "YS":
        raise ValueError(f"Unsupported freq: {freq}")

    periods: list[tuple[str, str]] = []
    for year in range(range_start.year, range_end.year + 1):
        period_start = max(range_start, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(range_end, pd.Timestamp(year=year, month=12, day=31))
        if period_start <= period_end:
            periods.append(
                (
                    period_start.strftime("%Y-%m-%d"),
                    period_end.strftime("%Y-%m-%d"),
                )
            )

    return periods


def _parse_temperature_response(
    locations: pd.DataFrame,
    payload: list[dict] | dict,
) -> pd.DataFrame:
    if isinstance(payload, dict):
        payload = [payload]

    if len(payload) != len(locations):
        raise ValueError(
            "Number of API results does not match number of requested locations."
        )

    frames: list[pd.DataFrame] = []
    for (_, location), result in zip(locations.iterrows(), payload, strict=True):
        daily = result.get("daily")
        if daily is None:
            raise ValueError(f"No daily data returned for {location['STNAME']}")

        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(daily["time"]),
                "temperature_f": daily["temperature_2m_mean"],
            }
        )
        frame["state"] = location["STNAME"]
        frame["population"] = location["POPULATION"]
        if "WEIGHT" in location:
            frame["population_weight"] = location["WEIGHT"]
        elif "population_weight" in location:
            frame["population_weight"] = location["population_weight"]

        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _cache_path(
    cache_dir: Path,
    locations: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> Path:
    state_key = "-".join(sorted(locations["STNAME"].astype(str)))
    digest = hashlib.sha256(
        f"{state_key}|{start_date}|{end_date}".encode()
    ).hexdigest()[:16]
    return cache_dir / f"temp_{digest}.parquet"


def fetch_temperature_chunk(
    locations: pd.DataFrame,
    start_date: str,
    end_date: str,
    *,
    cache_dir: Path | None = None,
    max_retries: int = 8,
    base_delay: float = 5.0,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """Fetch daily mean temperature for a small location/date chunk."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = _cache_path(cache_dir, locations, start_date, end_date)
        if cache_file.exists():
            return pd.read_parquet(cache_file)

    latitude_string = ",".join(locations["LATITUDE"].astype(str))
    longitude_string = ",".join(locations["LONGITUDE"].astype(str))
    params = {
        "latitude": latitude_string,
        "longitude": longitude_string,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
        "models": "era5_land",
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        response = requests.get(
            OPEN_METEO_ARCHIVE_URL,
            params=params,
            timeout=timeout,
        )

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                delay = float(retry_after)
            else:
                delay = base_delay * (2**attempt)
            time.sleep(delay)
            last_error = requests.HTTPError(
                "429 Too Many Requests",
                response=response,
            )
            continue

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            last_error = exc
            if response.status_code >= 500 and attempt + 1 < max_retries:
                time.sleep(base_delay * (2**attempt))
                continue
            raise

        frame = _parse_temperature_response(locations, response.json())
        if cache_dir is not None:
            frame.to_parquet(cache_file, index=False)
        return frame

    raise RuntimeError(
        f"Open-Meteo rate limit persisted after {max_retries} retries "
        f"for {start_date} to {end_date}."
    ) from last_error
