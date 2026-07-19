from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import power_forecast.pipelines as power_pipelines

from power_forecast.pipelines import build_power_forecast, run_power_data_pipeline


def _component_frame(component: str, origins: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for origin_number, origin in enumerate(origins):
        for horizon in range(1, 169):
            delivery = origin + pd.Timedelta(hours=horizon)
            hour = delivery.tz_convert("America/Chicago").hour
            if component == "load":
                baseline = 55_000 + 8_000 * np.sin(2 * np.pi * hour / 24)
            elif component == "wind":
                baseline = 12_000 + 2_000 * np.cos(2 * np.pi * hour / 24)
            else:
                baseline = max(0.0, 10_000 * np.sin(np.pi * (hour - 6) / 12))
            rows.append(
                {
                    "valid_at": delivery,
                    "issued_at": origin,
                    "baseline_mw": baseline + origin_number,
                    "actual_mw": baseline,
                }
            )
    return pd.DataFrame(rows)


def test_end_to_end_power_pipeline_materializes_168_balanced_hours():
    origins = pd.date_range("2026-06-01T00:00:00Z", periods=5, freq="D")
    all_hours = pd.date_range(origins.min() + pd.Timedelta(hours=1), origins.max() + pd.Timedelta(hours=168), freq="h")
    actuals = pd.DataFrame(
        {
            "valid_at": all_hours,
            "load_actual_mw": 55_000.0,
            "wind_actual_mw": 12_000.0,
            "solar_actual_mw": 5_000.0,
            "gas_generation_actual_mw": 25_000.0,
            "coal_generation_actual_mw": 10_000.0,
            "nuclear_actual_mw": 5_000.0,
            "hydro_actual_mw": 500.0,
            "other_nonthermal_actual_mw": 250.0,
            "net_imports_actual_mw": 100.0,
            "battery_net_discharge_actual_mw": 0.0,
            "geography": "ERCOT",
        }
    )
    delivery = pd.date_range(origins[-1] + pd.Timedelta(hours=1), periods=168, freq="h")
    issued = origins[-1]
    ercot_frames = {
        component: _component_frame(component, origins)
        for component in ("load", "wind", "solar")
    }
    ercot_frames["load_actual"] = pd.DataFrame(
        {"valid_at": all_hours, "issued_at": origins[-1], "actual_mw": 55_000.0}
    )
    ercot_frames["outages"] = pd.DataFrame(
        {
            "valid_at": delivery,
            "issued_at": issued,
            "conventional_outage_mw": 3_000.0,
        }
    )
    ercot_frames["adequacy"] = pd.DataFrame(
        {
            "valid_at": delivery,
            "issued_at": issued,
            "available_capacity_mw": 90_000.0,
        }
    )

    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp = Path(temp_dir)
        processed = temp / "processed"
        run_power_data_pipeline(
            origins[-1] + pd.Timedelta(minutes=1),
            ercot_frames=ercot_frames,
            eia930_frame=actuals,
            cache_dir=temp / "cache",
            processed_dir=processed,
        )
        forecast = build_power_forecast(origins[-1], processed_dir=processed)

        assert len(forecast) == 168
        assert forecast["delivery_hour"].is_unique
        assert np.abs(forecast["balance_error_mw"]).max() < 1e-8
        assert (processed / "ercot_hourly_power_forecast_latest.parquet").exists()


def test_incremental_eia_refresh_preserves_prior_history():
    first_hours = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    first = pd.DataFrame(
        {
            "valid_at": first_hours,
            "load_actual_mw": [1.0, 2.0, 3.0, 4.0],
            "geography": "ERCOT",
        }
    )
    second = pd.DataFrame(
        {
            "valid_at": [first_hours[-1], first_hours[-1] + pd.Timedelta(hours=1)],
            "load_actual_mw": [40.0, 5.0],
            "geography": "ERCOT",
        }
    )
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp = Path(temp_dir)
        processed = temp / "processed"
        common = {
            "ercot_frames": {},
            "cache_dir": temp / "cache",
            "processed_dir": processed,
        }
        run_power_data_pipeline(
            "2026-01-02T00:00:00Z", eia930_frame=first, **common
        )
        run_power_data_pipeline(
            "2026-01-03T00:00:00Z", eia930_frame=second, **common
        )
        latest_path = processed / "ercot_eia930_actuals_latest.parquet"
        result = pd.read_parquet(latest_path)

        # Existing installations may already have a truncated latest file;
        # the append-only vintages must be able to restore its coverage.
        result.tail(2).to_parquet(latest_path, index=False)
        recovered = power_pipelines._load_eia_actual_history(processed)

    assert len(result) == 5
    assert len(recovered) == 5
    assert result.loc[result["valid_at"] == first_hours[0], "load_actual_mw"].iloc[0] == 1.0
    assert result.loc[result["valid_at"] == first_hours[-1], "load_actual_mw"].iloc[0] == 40.0


def test_live_refresh_uses_wall_clock_retrieval_time(monkeypatch):
    monkeypatch.setattr(power_pipelines, "ERCOT_PRODUCTS", {})
    monkeypatch.setattr(power_pipelines, "ErcotApiClient", lambda: object())
    weather = pd.DataFrame(
        {
            "valid_at": ["2025-01-01T01:00:00Z"],
            "issued_at": ["2025-01-01T00:00:00Z"],
            "temperature_f": [50.0],
        }
    )
    empty_actuals = pd.DataFrame(
        {"valid_at": pd.Series(dtype="datetime64[ns, UTC]")}
    )
    before = pd.Timestamp.now(tz="UTC")
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp = Path(temp_dir)
        processed = temp / "processed"
        run_power_data_pipeline(
            "2025-01-01T00:00:00Z",
            eia930_frame=empty_actuals,
            weather_frame=weather,
            cache_dir=temp / "cache",
            processed_dir=processed,
        )
        result = pd.read_parquet(processed / "ercot_weather_vintages_latest.parquet")
    after = pd.Timestamp.now(tz="UTC")

    retrieved = pd.to_datetime(result["retrieved_at"], utc=True).iloc[0]
    assert before <= retrieved <= after


def test_default_forecast_origin_includes_latest_retrieved_baselines():
    issued = pd.Timestamp("2026-07-13T00:00:00Z")
    retrieval = issued + pd.Timedelta(minutes=1)
    delivery = pd.date_range(issued + pd.Timedelta(hours=1), periods=168, freq="h")
    ercot_frames = {
        component: pd.DataFrame(
            {
                "valid_at": delivery,
                "issued_at": issued,
                "baseline_mw": value,
            }
        )
        for component, value in (("load", 60_000.0), ("wind", 10_000.0), ("solar", 5_000.0))
    }
    empty_actuals = pd.DataFrame(
        {"valid_at": pd.Series(dtype="datetime64[ns, UTC]")}
    )

    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp = Path(temp_dir)
        processed = temp / "processed"
        run_power_data_pipeline(
            retrieval,
            ercot_frames=ercot_frames,
            eia930_frame=empty_actuals,
            cache_dir=temp / "cache",
            processed_dir=processed,
        )
        forecast = build_power_forecast(processed_dir=processed)

    assert forecast["forecast_origin"].iloc[0] == retrieval
    assert forecast["load_baseline_source"].eq("ercot").all()
    assert forecast["wind_baseline_source"].eq("ercot").all()
    assert forecast["solar_baseline_source"].eq("ercot").all()
