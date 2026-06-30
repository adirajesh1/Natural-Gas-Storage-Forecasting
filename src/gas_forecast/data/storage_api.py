from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from gas_forecast.data.cache import (
    DEFAULT_CACHE_DIR,
    load_parquet_cache,
    merge_timeseries,
    write_parquet_cache,
)

EIA_WEEKLY_STORAGE_URL = (
    "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
)

STORAGE_CACHE_FILENAME = "weekly_storage_raw.parquet"
STORAGE_MERGE_KEY_COLS = ["duoarea", "period", "series"]


def _storage_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "storage" / STORAGE_CACHE_FILENAME


def fetch_weekly_storage_raw(
    api_key: str,
    *,
    start: str | None = None,
    end: str | None = None,
    offset: int = 0,
    length: int = 5000,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """
    Download weekly natural gas storage data from the EIA API.

    Parameters
    ----------
    api_key:
        EIA API key.
    start:
        Optional inclusive start date (YYYY-MM-DD).
    end:
        Optional inclusive end date (YYYY-MM-DD).
    offset:
        Row offset for pagination.
    length:
        Maximum number of records requested from the API.
    timeout:
        Maximum number of seconds to wait for the API response.
    """
    if not api_key:
        raise ValueError(
            "EIA API key is missing. Set EIA_API_KEY in local.env."
        )

    params: dict[str, str | int] = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": offset,
        "length": length,
    }
    if start is not None:
        params["start"] = start
    if end is not None:
        params["end"] = end

    response = requests.get(
        EIA_WEEKLY_STORAGE_URL,
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()

    try:
        records = payload["response"]["data"]
    except KeyError as exc:
        raise KeyError(
            "The EIA response did not contain response.data."
        ) from exc

    return pd.DataFrame(records)


def fetch_weekly_storage_paginated(
    api_key: str,
    *,
    start: str | None = None,
    end: str | None = None,
    length: int = 5000,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Download all weekly storage rows for a date window, paginating as needed."""
    frames: list[pd.DataFrame] = []
    offset = 0

    while True:
        chunk = fetch_weekly_storage_raw(
            api_key,
            start=start,
            end=end,
            offset=offset,
            length=length,
            timeout=timeout,
        )
        if chunk.empty:
            break

        frames.append(chunk)
        if len(chunk) < length:
            break

        offset += length

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def fetch_weekly_storage_incremental(
    api_key: str,
    *,
    cache_dir: str | Path | None = None,
    revision_weeks: int = 8,
    min_start_date: str = "2010-01-01",
    force_refresh: bool = False,
    length: int = 5000,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """
    Load weekly storage from cache and fetch only missing or revised rows.

    On first run, backfills the full history. On later runs, re-fetches the
    most recent ``revision_weeks`` to capture EIA revisions and appends any
    new periods.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_path = _storage_cache_path(Path(cache_dir))
    cached = pd.DataFrame() if force_refresh else load_parquet_cache(cache_path)

    if cached.empty:
        fresh = fetch_weekly_storage_paginated(
            api_key,
            length=length,
            timeout=timeout,
        )
        merged = _finalize_storage_cache(fresh, min_start_date=min_start_date)
        write_parquet_cache(merged, cache_path)
        return merged

    cached["period"] = pd.to_datetime(cached["period"])
    max_period = cached["period"].max()
    fetch_start = (max_period - pd.Timedelta(weeks=revision_weeks)).strftime(
        "%Y-%m-%d"
    )
    fetch_end = pd.Timestamp.today().strftime("%Y-%m-%d")

    fresh = fetch_weekly_storage_paginated(
        api_key,
        start=fetch_start,
        end=fetch_end,
        length=length,
        timeout=timeout,
    )
    merged = merge_timeseries(
        cached,
        fresh,
        key_cols=STORAGE_MERGE_KEY_COLS,
        date_col="period",
    )
    merged = _finalize_storage_cache(merged, min_start_date=min_start_date)
    write_parquet_cache(merged, cache_path)
    return merged


def _finalize_storage_cache(
    df: pd.DataFrame,
    *,
    min_start_date: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()
    result["period"] = pd.to_datetime(result["period"])
    result = result[result["period"] >= pd.to_datetime(min_start_date)]
    result = result.drop_duplicates(subset=STORAGE_MERGE_KEY_COLS, keep="last")
    return result.sort_values(STORAGE_MERGE_KEY_COLS).reset_index(drop=True)
