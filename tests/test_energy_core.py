from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from energy_forecast.artifacts import append_vintage_parquet
from energy_forecast.artifacts import save_versioned_parquet
from energy_forecast.asof import select_as_of
from energy_forecast.intervals import add_horizon_conformal_intervals
from energy_forecast.splitters import RollingOriginSplitter


def test_select_as_of_excludes_later_publications():
    frame = pd.DataFrame(
        {
            "component": ["load", "load"],
            "valid_at": ["2026-07-14T01:00:00Z"] * 2,
            "issued_at": ["2026-07-13T00:00:00Z", "2026-07-13T02:00:00Z"],
            "value": [10.0, 99.0],
        }
    )
    selected = select_as_of(
        frame,
        "2026-07-13T01:00:00Z",
        entity_keys=["component"],
    )
    assert selected.loc[0, "value"] == pytest.approx(10.0)


def test_select_as_of_excludes_rows_retrieved_after_origin():
    frame = pd.DataFrame(
        {
            "component": ["load", "load"],
            "valid_at": ["2026-07-14T01:00:00Z"] * 2,
            "issued_at": ["2026-07-13T00:00:00Z"] * 2,
            "retrieved_at": ["2026-07-13T00:30:00Z", "2026-07-13T02:00:00Z"],
            "value": [10.0, 99.0],
        }
    )
    selected = select_as_of(
        frame,
        "2026-07-13T01:00:00Z",
        entity_keys=["component"],
    )
    assert selected.loc[0, "value"] == pytest.approx(10.0)


def test_hourly_rolling_origin_splitter_preserves_intraday_boundaries():
    frame = pd.DataFrame(
        {"time": pd.date_range("2026-01-01T00:00:00Z", periods=10, freq="h")}
    )
    splitter = RollingOriginSplitter(
        "time",
        initial_train_end="2026-01-01T04:00:00Z",
        validation_horizon="2h",
        step="2h",
    )
    splits = list(splitter.split(frame))
    assert splits[0] == ([0, 1, 2, 3], [4, 5])
    assert splits[1] == ([0, 1, 2, 3, 4, 5], [6, 7])


def test_append_vintages_never_overwrites_prior_part():
    frame = pd.DataFrame(
        {
            "source": ["ERCOT"],
            "product_id": ["TEST"],
            "issued_at": ["2026-01-01T00:00:00Z"],
            "retrieved_at": ["2026-01-01T00:01:00Z"],
            "valid_at": ["2026-01-01T01:00:00Z"],
            "geography": ["ERCOT"],
            "source_hash": ["abc"],
        }
    )
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        first = append_vintage_parquet(frame, temp_dir)
        second = append_vintage_parquet(frame, temp_dir)
        assert first != second
        assert len(list(Path(temp_dir).glob("*.parquet"))) == 2


def test_versioned_artifacts_never_overwrite_same_second(monkeypatch):
    class FixedDatetime:
        @classmethod
        def now(cls, _timezone):
            return pd.Timestamp("2026-01-01T00:00:00Z").to_pydatetime()

    monkeypatch.setattr("energy_forecast.artifacts.datetime", FixedDatetime)
    frame = pd.DataFrame({"value": [1.0]})
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        first = save_versioned_parquet(frame, temp_dir, "example", save_latest=False)
        second = save_versioned_parquet(frame, temp_dir, "example", save_latest=False)

        assert first != second
        assert len(list(Path(temp_dir).glob("example_*.parquet"))) == 2


def test_conformal_intervals_do_not_use_same_origin_residuals():
    frame = pd.DataFrame(
        {
            "forecast_origin": ["2026-01-01T00:00:00Z"] * 2 + ["2026-01-02T00:00:00Z"],
            "horizon_bucket": ["short"] * 3,
            "actual": [10.0, 30.0, 20.0],
            "prediction": [0.0, 0.0, 10.0],
        }
    )
    result = add_horizon_conformal_intervals(
        frame,
        actual_col="actual",
        prediction_col="prediction",
        min_calibration=2,
    )
    assert result.loc[0, "calibration_count"] == 0
    assert result.loc[1, "calibration_count"] == 0
    assert result.loc[2, "calibration_count"] == 2
    assert result.loc[2, "upper_bound"] > result.loc[2, "prediction"]
