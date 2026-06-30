from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path("data/cache")


def load_parquet_cache(path: Path) -> pd.DataFrame:
    """Load a parquet cache file or return an empty DataFrame if missing."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def write_parquet_cache(df: pd.DataFrame, path: Path) -> None:
    """Atomically write a parquet cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def merge_timeseries(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    *,
    key_cols: list[str],
    date_col: str,
    keep: str = "last",
) -> pd.DataFrame:
    """Concatenate and deduplicate time-series rows, sorted by date."""
    if existing.empty:
        merged = new.copy()
    elif new.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_cols, keep=keep)

    if merged.empty:
        return merged

    merged[date_col] = pd.to_datetime(merged[date_col])
    sort_cols = list(dict.fromkeys(key_cols + [date_col]))
    return merged.sort_values(sort_cols).reset_index(drop=True)


def compute_date_gaps(
    cached_dates: pd.Series | None,
    start_date: str,
    end_date: str,
) -> list[tuple[str, str]]:
    """Return inclusive date gaps between cached coverage and a requested range."""
    range_start = pd.Timestamp(start_date).normalize()
    range_end = pd.Timestamp(end_date).normalize()

    if range_start > range_end:
        raise ValueError(
            f"start_date {start_date} is after end_date {end_date}."
        )

    if cached_dates is None or len(cached_dates) == 0:
        return [(range_start.strftime("%Y-%m-%d"), range_end.strftime("%Y-%m-%d"))]

    cached = pd.Series(pd.to_datetime(cached_dates)).dt.normalize()
    cache_min = cached.min()
    cache_max = cached.max()

    gaps: list[tuple[str, str]] = []

    if range_start < cache_min:
        prefix_end = min(range_end, cache_min - pd.Timedelta(days=1))
        if range_start <= prefix_end:
            gaps.append(
                (
                    range_start.strftime("%Y-%m-%d"),
                    prefix_end.strftime("%Y-%m-%d"),
                )
            )

    if range_end > cache_max:
        suffix_start = max(range_start, cache_max + pd.Timedelta(days=1))
        if suffix_start <= range_end:
            gaps.append(
                (
                    suffix_start.strftime("%Y-%m-%d"),
                    range_end.strftime("%Y-%m-%d"),
                )
            )

    return gaps


def split_gap_into_periods(
    gap_start: str,
    gap_end: str,
    *,
    max_days: int = 31,
) -> list[tuple[str, str]]:
    """Split a gap into chunks no larger than max_days (inclusive)."""
    start = pd.Timestamp(gap_start)
    end = pd.Timestamp(gap_end)
    total_days = (end - start).days + 1

    if total_days <= max_days:
        return [(gap_start, gap_end)]

    periods: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=max_days - 1), end)
        periods.append(
            (
                cursor.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
        )
        cursor = chunk_end + pd.Timedelta(days=1)

    return periods
