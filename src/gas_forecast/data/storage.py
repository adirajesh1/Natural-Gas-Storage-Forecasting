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

    Returns
    -------
    pd.DataFrame
        Weekly storage records returned by the EIA API.

    Raises
    ------
    ValueError
        If the API key is missing.
    requests.HTTPError
        If the EIA API returns an unsuccessful HTTP response.
    KeyError
        If the expected data is missing from the API response.
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


def clean_weekly_storage(raw_df, start_date=None, end_date=None):
    df = raw_df.copy()

    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series-description"] = df["series-description"].astype(str)

    if start_date is not None:
        df = df[df["period"] >= pd.to_datetime(start_date)]

    if end_date is not None:
        df = df[df["period"] <= pd.to_datetime(end_date)]

    df["year"] = df["period"].dt.year
    df["month"] = df["period"].dt.month
    df["week_of_year"] = df["period"].dt.isocalendar().week.astype(int)

    dedupe_cols = (
        ["duoarea", "period", "series"]
        if "series" in df.columns
        else ["duoarea", "period"]
    )
    df = df.drop_duplicates(subset=dedupe_cols, keep="last")

    df = df.sort_values(["duoarea", "period"]).reset_index(drop=True)

    validate_weekly_storage(df)

    return df


def select_region(storage: pd.DataFrame, region: str | list[str]) -> pd.DataFrame:
    if isinstance(region, str):
        regions = [region]
    else:
        regions = region

    available_regions = set(storage["duoarea"].unique())
    missing_regions = [r for r in regions if r not in available_regions]
    if missing_regions:
        raise ValueError(f"Region(s) {missing_regions} not found in storage['duoarea']")

    selected_storage = storage.loc[storage["duoarea"].isin(regions)]
    validate_storage_region(selected_storage)

    return selected_storage


def calculate_weekly_storage_change(storage: pd.DataFrame) -> pd.DataFrame:
    storage['weekly_change_bcf'] = storage['value'].diff()
    return storage


def prepare_storage_model_data(storage: pd.DataFrame) -> pd.DataFrame:
    df = storage.copy()

    df = df.rename(columns={
        "period": "date",
        "value": "storage_bcf",
    })

    df = df[
        ["date", "storage_bcf", "weekly_change_bcf", "year", "month", "week_of_year", "duoarea"]
    ]

    df = df.dropna(subset=["weekly_change_bcf"])

    return df.reset_index(drop=True)


def validate_weekly_storage(storage: pd.DataFrame) -> None:
    required_columns = {
        "period",
        "value",
        "series-description",
        "duoarea",
        "year",
        "month",
        "week_of_year",
    }

    missing_cols = required_columns - set(storage.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if storage["period"].isna().any():
        raise ValueError("Found missing values in 'period'.")

    if storage["value"].isna().any():
        n_missing = storage["value"].isna().sum()
        raise ValueError(f"Found {n_missing} missing values in 'value'.")

    if (storage["value"] < 0).any():
        raise ValueError("Found negative values in 'value'.")

    if not storage.sort_values(["duoarea", "period"]).index.equals(storage.index):
        raise ValueError("Storage data is not sorted by ['duoarea', 'period'].")


def validate_storage_region(df: pd.DataFrame) -> None:
    for _, group in df.groupby("duoarea"):
        if group["period"].duplicated().any():
            raise ValueError("Selected region has duplicate periods.")

        if not group["period"].is_monotonic_increasing:
            raise ValueError("Selected region is not sorted by period.")
