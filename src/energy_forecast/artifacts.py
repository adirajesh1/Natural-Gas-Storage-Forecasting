from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import uuid

import pandas as pd


REQUIRED_VINTAGE_COLUMNS = frozenset(
    {
        "source",
        "product_id",
        "issued_at",
        "retrieved_at",
        "valid_at",
        "geography",
        "source_hash",
    }
)


def load_parquet_cache(path: str | Path) -> pd.DataFrame:
    """Load one parquet file or an append-only parquet dataset."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def write_parquet_cache(df: pd.DataFrame, path: str | Path) -> None:
    """Atomically replace a parquet cache file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def append_vintage_parquet(
    frame: pd.DataFrame,
    dataset_dir: str | Path,
    *,
    required_columns: set[str] | frozenset[str] = REQUIRED_VINTAGE_COLUMNS,
) -> Path:
    """Append an immutable parquet part to a vintage dataset.

    The caller owns semantic deduplication. A unique part is always written so
    prior source publications are never silently replaced.
    """
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Vintage data missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Cannot append an empty vintage frame.")

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    retrieved = pd.to_datetime(frame["retrieved_at"], utc=True, errors="coerce")
    if retrieved.isna().any():
        raise ValueError("Vintage data contains invalid retrieved_at timestamps.")
    stamp = retrieved.max().strftime("%Y%m%dT%H%M%S%fZ")
    path = dataset_dir / f"part-{stamp}-{uuid.uuid4().hex[:12]}.parquet"
    tmp_path = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)
    return path


def save_versioned_parquet(
    df: pd.DataFrame,
    output_dir: str | Path,
    dataset_name: str | None = None,
    *,
    save_latest: bool = True,
) -> Path:
    """Write a timestamped parquet artifact and an optional latest alias."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_name is None or dataset_name == "":
        dataset_base = f"df_{id(df)}"
    else:
        dataset_base = dataset_name

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    versioned = output_dir / f"{dataset_base}_{stamp}_{uuid.uuid4().hex[:12]}.parquet"
    write_parquet_cache(df, versioned)
    if save_latest:
        write_parquet_cache(df, output_dir / f"{dataset_base}_latest.parquet")
    return versioned


def merge_timeseries(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    *,
    key_cols: list[str],
    date_col: str,
    keep: str = "last",
) -> pd.DataFrame:
    """Concatenate, deduplicate, and sort time-series rows."""
    if existing.empty:
        merged = new.copy()
    elif new.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new], ignore_index=True)
    if merged.empty:
        return merged
    merged[date_col] = pd.to_datetime(merged[date_col])
    merged = merged.drop_duplicates(subset=key_cols, keep=keep)
    return merged.sort_values(list(dict.fromkeys(key_cols + [date_col]))).reset_index(
        drop=True
    )


def compute_date_gaps(
    cached_dates: pd.Series | None,
    start_date: str,
    end_date: str,
) -> list[tuple[str, str]]:
    """Return contiguous missing ranges within requested daily coverage."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError(f"start_date {start_date} is after end_date {end_date}.")
    if cached_dates is None or len(cached_dates) == 0:
        return [(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))]
    cached = pd.DatetimeIndex(pd.to_datetime(cached_dates)).normalize().unique()
    missing = pd.date_range(start, end, freq="D").difference(cached)
    if missing.empty:
        return []

    gaps: list[tuple[str, str]] = []
    gap_start = previous = missing[0]
    for current in missing[1:]:
        if current != previous + pd.Timedelta(days=1):
            gaps.append(
                (gap_start.strftime("%Y-%m-%d"), previous.strftime("%Y-%m-%d"))
            )
            gap_start = current
        previous = current
    gaps.append((gap_start.strftime("%Y-%m-%d"), previous.strftime("%Y-%m-%d")))
    return gaps

