from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from gas_forecast.data.cache import (
    compute_date_gaps,
    load_parquet_cache,
    merge_timeseries,
    write_parquet_cache,
)
from gas_forecast.data.regions import lower48_excluded_states, region_states

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

CENSUS_POP_URL = (
    "https://www2.census.gov/geo/docs/reference/"
    "cenpop2020/CenPop2020_Mean_ST.txt"
)

_LOCATION_COLUMNS = {
    "STATEFP",
    "STNAME",
    "POPULATION",
    "LATITUDE",
    "LONGITUDE",
    "WEIGHT",
}


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


def _state_cache_path(cache_dir: Path, state_name: str) -> Path:
    safe_name = re.sub(r"[^\w\-]+", "_", state_name.strip())
    return cache_dir / "by_state" / f"{safe_name}.parquet"


def _gap_fetch_periods(gap_start: str, gap_end: str) -> list[tuple[str, str]]:
    """Split a date gap into API-friendly request periods."""
    start = pd.Timestamp(gap_start)
    end = pd.Timestamp(gap_end)
    days = (end - start).days + 1
    if days <= 31:
        return [(gap_start, gap_end)]
    return _date_periods(gap_start, gap_end, freq="YS")


def _apply_location_metadata(
    frame: pd.DataFrame,
    location: pd.Series,
) -> pd.DataFrame:
    result = frame.copy()
    result["state"] = location["STNAME"]
    result["population"] = location["POPULATION"]
    result["population_weight"] = location["WEIGHT"]
    return result


def _fetch_state_temperatures_incremental(
    location: pd.Series,
    start_date: str,
    end_date: str,
    *,
    cache_dir: Path,
    force_refresh: bool = False,
    pause_seconds: float = 3.0,
    max_retries: int = 8,
    base_delay: float = 5.0,
) -> pd.DataFrame:
    """Fetch daily temperatures for one state using a rolling per-state cache."""
    state_name = str(location["STNAME"])
    cache_path = _state_cache_path(cache_dir, state_name)
    location_df = pd.DataFrame([location])

    if force_refresh:
        cached = pd.DataFrame()
    else:
        cached = load_parquet_cache(cache_path)

    if not cached.empty:
        cached["date"] = pd.to_datetime(cached["date"])

    gaps = compute_date_gaps(
        cached["date"] if not cached.empty else None,
        start_date,
        end_date,
    )

    new_frames: list[pd.DataFrame] = []
    request_periods: list[tuple[str, str, str]] = []
    for gap_start, gap_end in gaps:
        for period_start, period_end in _gap_fetch_periods(gap_start, gap_end):
            request_periods.append((state_name, period_start, period_end))

    for index, (_, period_start, period_end) in enumerate(request_periods):
        print(f"{state_name}: fetching {period_start} to {period_end}")
        chunk = fetch_temperature_chunk(
            location_df,
            period_start,
            period_end,
            cache_dir=None,
            max_retries=max_retries,
            base_delay=base_delay,
        )
        new_frames.append(chunk)
        if index + 1 < len(request_periods):
            time.sleep(pause_seconds)

    if new_frames:
        fresh = pd.concat(new_frames, ignore_index=True)
    else:
        fresh = pd.DataFrame()

    merged = merge_timeseries(
        cached,
        fresh,
        key_cols=["state", "date"],
        date_col="date",
    )
    merged = _apply_location_metadata(merged, location)

    if not merged.empty:
        write_parquet_cache(merged, cache_path)

    range_start = pd.Timestamp(start_date)
    range_end = pd.Timestamp(end_date)
    return merged[
        (merged["date"] >= range_start) & (merged["date"] <= range_end)
    ].reset_index(drop=True)


def migrate_weather_chunk_cache(
    old_cache_dir: str | Path,
    new_cache_dir: str | Path,
) -> int:
    """
    Merge legacy hash-keyed weather chunk files into per-state caches.

    Returns the number of legacy chunk files processed.
    """
    old_dir = Path(old_cache_dir)
    new_dir = Path(new_cache_dir)
    if not old_dir.exists():
        return 0

    chunk_files = sorted(old_dir.glob("temp_*.parquet"))
    if not chunk_files:
        return 0

    for chunk_file in chunk_files:
        chunk = pd.read_parquet(chunk_file)
        if chunk.empty or "state" not in chunk.columns:
            continue

        for state_name, state_frame in chunk.groupby("state"):
            cache_path = _state_cache_path(new_dir, str(state_name))
            existing = load_parquet_cache(cache_path)
            merged = merge_timeseries(
                existing,
                state_frame,
                key_cols=["state", "date"],
                date_col="date",
            )
            write_parquet_cache(merged, cache_path)

    return len(chunk_files)


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


def fetch_all_state_temperatures(
    locations: pd.DataFrame,
    start_date: str,
    end_date: str,
    *,
    location_batch_size: int = 1,
    date_freq: str = "YS",
    pause_seconds: float = 3.0,
    cache_dir: Path | None = None,
    incremental: bool = True,
    force_refresh: bool = False,
    max_retries: int = 8,
    base_delay: float = 5.0,
) -> pd.DataFrame:
    """Retrieve daily temperatures for every requested state."""
    if location_batch_size < 1:
        raise ValueError("location_batch_size must be at least 1.")

    if incremental:
        if cache_dir is None:
            raise ValueError("cache_dir is required when incremental=True.")
        cache_dir.mkdir(parents=True, exist_ok=True)

        frames: list[pd.DataFrame] = []
        for index, (_, location) in enumerate(locations.iterrows()):
            state_frame = _fetch_state_temperatures_incremental(
                location,
                start_date,
                end_date,
                cache_dir=cache_dir,
                force_refresh=force_refresh,
                pause_seconds=pause_seconds,
                max_retries=max_retries,
                base_delay=base_delay,
            )
            frames.append(state_frame)
            if index + 1 < len(locations):
                time.sleep(pause_seconds)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    frames: list[pd.DataFrame] = []
    date_periods = _date_periods(start_date, end_date, freq=date_freq)
    total_chunks = (
        (len(locations) + location_batch_size - 1) // location_batch_size
    ) * len(date_periods)
    chunk_number = 0

    for period_start, period_end in date_periods:
        for start in range(0, len(locations), location_batch_size):
            chunk_number += 1
            batch = locations.iloc[start : start + location_batch_size].copy()
            state_names = ", ".join(batch["STNAME"].tolist())

            print(
                f"[{chunk_number}/{total_chunks}] "
                f"{period_start} to {period_end} | {state_names}"
            )

            chunk = fetch_temperature_chunk(
                batch,
                period_start,
                period_end,
                cache_dir=cache_dir,
                max_retries=max_retries,
                base_delay=base_delay,
            )
            frames.append(chunk)

            if chunk_number < total_chunks:
                time.sleep(pause_seconds)

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

def load_census_state_locations(
    url: str = CENSUS_POP_URL,
    *,
    exclude: frozenset[str] | set[str] | None = None,
) -> pd.DataFrame:
    """Load census state centroids and compute national population weights."""
    if exclude is None:
        exclude = lower48_excluded_states()

    locations = pd.read_csv(url, dtype={"STATEFP": str})
    locations = locations.loc[~locations["STNAME"].isin(exclude)].copy()
    locations["WEIGHT"] = (
        locations["POPULATION"] / locations["POPULATION"].sum()
    )
    return locations.reset_index(drop=True)


def select_weather_locations(
    locations: pd.DataFrame,
    duoarea: str,
) -> pd.DataFrame:
    """Filter census locations to an EIA storage region and renormalize weights."""
    states = region_states(duoarea)
    available_states = set(locations["STNAME"].astype(str))
    missing_states = sorted(states - available_states)
    if missing_states:
        raise ValueError(
            f"Region {duoarea} requires states missing from locations: "
            f"{missing_states}"
        )

    selected = locations.loc[locations["STNAME"].isin(states)].copy()
    selected["WEIGHT"] = selected["POPULATION"] / selected["POPULATION"].sum()
    validate_weather_locations(selected, expected_states=states)
    return selected.reset_index(drop=True)


def calculate_hdd_cdd(temperatures: pd.DataFrame) -> pd.DataFrame:
    """Calculate heating and cooling degree days from daily temperatures."""
    df = temperatures.copy()
    df["hdd"] = np.maximum(0, 65 - df["temperature_f"])
    df["cdd"] = np.maximum(0, df["temperature_f"] - 65)
    return df


def aggregate_population_weighted_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate state-level daily weather to a population-weighted regional index."""
    required = {"date", "temperature_f", "hdd", "cdd", "population_weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    weighted = df.assign(
        w_temp=df["temperature_f"] * df["population_weight"],
        w_hdd=df["hdd"] * df["population_weight"],
        w_cdd=df["cdd"] * df["population_weight"],
    )
    return (
        weighted.groupby("date", as_index=False)
        .agg(
            temperature_f=("w_temp", "sum"),
            hdd=("w_hdd", "sum"),
            cdd=("w_cdd", "sum"),
        )
        .reset_index(drop=True)
    )


def prepare_weather_model_data(
    weather: pd.DataFrame,
    duoarea: str,
) -> pd.DataFrame:
    """Select and format regional daily weather for modeling."""
    df = weather.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["duoarea"] = duoarea
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    return df[
        [
            "date",
            "temperature_f",
            "hdd",
            "cdd",
            "year",
            "month",
            "day_of_year",
            "duoarea",
        ]
    ].reset_index(drop=True)


def validate_weather_locations(
    locations: pd.DataFrame,
    *,
    expected_states: frozenset[str] | set[str] | None = None,
) -> None:
    """Validate a locations table used for weather downloads."""
    missing_cols = _LOCATION_COLUMNS - set(locations.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if locations["STNAME"].duplicated().any():
        raise ValueError("Duplicate state names in locations.")

    weight_sum = locations["WEIGHT"].sum()
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(f"Location weights must sum to 1, got {weight_sum}.")

    if expected_states is not None:
        actual_states = set(locations["STNAME"].astype(str))
        if actual_states != set(expected_states):
            missing = sorted(set(expected_states) - actual_states)
            extra = sorted(actual_states - set(expected_states))
            raise ValueError(
                f"Location states do not match region. "
                f"Missing: {missing}. Extra: {extra}."
            )


def validate_state_daily_weather(
    weather: pd.DataFrame,
    *,
    expected_states: frozenset[str] | set[str] | None = None,
) -> None:
    """Validate state-level daily weather before regional aggregation."""
    required = {
        "date",
        "temperature_f",
        "state",
        "population",
        "population_weight",
    }
    missing_cols = required - set(weather.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if weather["temperature_f"].isna().any():
        raise ValueError("Found missing values in 'temperature_f'.")

    if weather.duplicated(subset=["date", "state"]).any():
        raise ValueError("Found duplicate date/state rows.")

    if expected_states is not None:
        actual_states = set(weather["state"].astype(str))
        if actual_states != set(expected_states):
            missing = sorted(set(expected_states) - actual_states)
            extra = sorted(actual_states - set(expected_states))
            raise ValueError(
                f"Weather states do not match region. "
                f"Missing: {missing}. Extra: {extra}."
            )
