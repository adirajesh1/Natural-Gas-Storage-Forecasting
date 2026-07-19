from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


IndexSplit = tuple[list[int], list[int]]


def _normalize_dates(df: pd.DataFrame, date_col: str) -> pd.Series:
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col!r}")
    return pd.to_datetime(df[date_col]).dt.normalize()


def _date_or_none(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value).normalize()


def _positions(mask: pd.Series) -> list[int]:
    return np.flatnonzero(mask.to_numpy()).tolist()


def _require_nonempty(train_idx: list[int], val_idx: list[int]) -> None:
    if not train_idx:
        raise ValueError("Splitter produced an empty training window.")
    if not val_idx:
        raise ValueError("Splitter produced an empty validation window.")


@dataclass(frozen=True)
class HoldoutSplitter:
    """One train/validation split using inclusive date boundaries."""

    date_col: str
    train_start: str | pd.Timestamp | None = None
    train_end: str | pd.Timestamp | None = None
    val_start: str | pd.Timestamp | None = None
    val_end: str | pd.Timestamp | None = None

    def split(self, df: pd.DataFrame) -> Iterator[IndexSplit]:
        dates = _normalize_dates(df, self.date_col)
        train_start = _date_or_none(self.train_start)
        train_end = _date_or_none(self.train_end)
        val_start = _date_or_none(self.val_start)
        val_end = _date_or_none(self.val_end)

        if train_end is None and val_start is None:
            raise ValueError("HoldoutSplitter requires train_end or val_start.")

        if train_end is None and val_start is not None:
            train_end = val_start - pd.Timedelta(days=1)
        if val_start is None and train_end is not None:
            val_start = train_end + pd.Timedelta(days=1)

        if train_start is None:
            train_start = dates.min()
        if val_end is None:
            val_end = dates.max()

        if train_start > train_end:
            raise ValueError("train_start must not be after train_end.")
        if val_start > val_end:
            raise ValueError("val_start must not be after val_end.")
        if train_end >= val_start:
            raise ValueError("Training dates must precede validation dates.")

        train_mask = (dates >= train_start) & (dates <= train_end)
        val_mask = (dates >= val_start) & (dates <= val_end)
        train_idx = _positions(train_mask)
        val_idx = _positions(val_mask)
        _require_nonempty(train_idx, val_idx)
        yield train_idx, val_idx


@dataclass(frozen=True)
class ExpandingWindowSplitter:
    """Expanding-window validation with fixed-size validation windows."""

    date_col: str
    initial_train_start: str | pd.Timestamp
    initial_train_end: str | pd.Timestamp
    val_weeks: int
    step_weeks: int

    def split(self, df: pd.DataFrame) -> Iterator[IndexSplit]:
        if self.val_weeks < 1:
            raise ValueError("val_weeks must be at least 1.")
        if self.step_weeks < 1:
            raise ValueError("step_weeks must be at least 1.")

        dates = _normalize_dates(df, self.date_col)
        train_start = pd.Timestamp(self.initial_train_start).normalize()
        train_end = pd.Timestamp(self.initial_train_end).normalize()
        max_date = dates.max()

        while True:
            val_start = train_end + pd.Timedelta(days=1)
            val_end = val_start + pd.Timedelta(weeks=self.val_weeks) - pd.Timedelta(days=1)
            if val_start > max_date:
                break

            train_idx = _positions((dates >= train_start) & (dates <= train_end))
            val_idx = _positions((dates >= val_start) & (dates <= val_end))
            if val_idx:
                _require_nonempty(train_idx, val_idx)
                yield train_idx, val_idx

            train_end = train_end + pd.Timedelta(weeks=self.step_weeks)


@dataclass(frozen=True)
class RollingWindowSplitter:
    """Rolling-window validation with a fixed-size training window."""

    date_col: str
    initial_train_start: str | pd.Timestamp
    initial_train_end: str | pd.Timestamp
    val_weeks: int
    step_weeks: int

    def split(self, df: pd.DataFrame) -> Iterator[IndexSplit]:
        if self.val_weeks < 1:
            raise ValueError("val_weeks must be at least 1.")
        if self.step_weeks < 1:
            raise ValueError("step_weeks must be at least 1.")

        dates = _normalize_dates(df, self.date_col)
        train_start = pd.Timestamp(self.initial_train_start).normalize()
        train_end = pd.Timestamp(self.initial_train_end).normalize()
        max_date = dates.max()

        while True:
            val_start = train_end + pd.Timedelta(days=1)
            val_end = val_start + pd.Timedelta(weeks=self.val_weeks) - pd.Timedelta(days=1)
            if val_start > max_date:
                break

            train_idx = _positions((dates >= train_start) & (dates <= train_end))
            val_idx = _positions((dates >= val_start) & (dates <= val_end))
            if val_idx:
                _require_nonempty(train_idx, val_idx)
                yield train_idx, val_idx

            step = pd.Timedelta(weeks=self.step_weeks)
            train_start = train_start + step
            train_end = train_end + step
