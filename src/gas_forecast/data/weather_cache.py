from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd

from gas_forecast.data.cache import (
    compute_date_gaps,
    load_parquet_cache,
    merge_timeseries,
    write_parquet_cache,
)
from gas_forecast.data.weather_api import _date_periods, fetch_temperature_chunk


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
) -> tuple[pd.DataFrame, bool]:
    """Fetch daily temperatures for one state using a rolling per-state cache.

    Returns the requested date-range frame and whether any API requests were made.
    """
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

    fetched = bool(request_periods)

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
    result = merged[
        (merged["date"] >= range_start) & (merged["date"] <= range_end)
    ].reset_index(drop=True)
    return result, fetched


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
            state_frame, fetched = _fetch_state_temperatures_incremental(
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
            if fetched and index + 1 < len(locations):
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
