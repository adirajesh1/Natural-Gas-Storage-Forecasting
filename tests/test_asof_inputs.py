from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from gas_forecast.data.balance_asof import (
    build_asof_balance_features,
    latest_balance_as_of,
)
from gas_forecast.data.weather_scenarios import select_weather_scenario_as_of
from gas_forecast.pipelines.asof import (
    run_asof_balance_pipeline,
    run_live_weather_forecast_pipeline,
    run_weather_scenario_pipeline,
)
from gas_forecast.data.weather_forecasts import parse_open_meteo_ensemble_response


def test_weather_scenario_selection_uses_only_versions_available_at_origin():
    scenarios = pd.DataFrame(
        {
            "date": ["2024-02-02", "2024-02-02", "2024-02-02"],
            "duoarea": ["R48", "R48", "R31"],
            "issued_at": [
                "2024-01-30T12:00:00Z",
                "2024-02-03T12:00:00Z",
                "2024-01-30T12:00:00Z",
            ],
            "temperature_f": [45.0, 60.0, 40.0],
            "hdd": [70.0, 35.0, 80.0],
            "cdd": [0.0, 0.0, 0.0],
            "weather_days": [7.0, 7.0, 7.0],
        }
    )

    early = select_weather_scenario_as_of(
        scenarios,
        "2024-02-02T00:00:00Z",
        region="R48",
        target_dates=["2024-02-02"],
    )
    late = select_weather_scenario_as_of(
        scenarios,
        "2024-02-04T00:00:00Z",
        region="R48",
        target_dates=["2024-02-02"],
    )

    assert early.loc[0, "hdd"] == pytest.approx(70.0)
    assert late.loc[0, "hdd"] == pytest.approx(35.0)


def test_asof_balance_features_exclude_unavailable_revisions():
    source_dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    vintages = pd.DataFrame(
        {
            "date": list(source_dates) + [source_dates[-1]],
            "duoarea": ["R48"] * 5,
            "available_at": [
                "2024-02-01T00:00:00Z",
                "2024-02-01T00:00:00Z",
                "2024-02-01T00:00:00Z",
                "2024-02-01T00:00:00Z",
                "2024-02-03T00:00:00Z",
            ],
            "local_balance": [4.0, 5.0, 6.0, 10.0, 99.0],
            "net_inflow_balancing": [1.0, 2.0, 3.0, 4.0, 88.0],
        }
    )
    origins = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-02-02")],
            "duoarea": ["R48"],
            "as_of": ["2024-02-02T00:00:00Z"],
        }
    )

    features = build_asof_balance_features(origins, vintages, as_of_col="as_of")
    early = latest_balance_as_of(vintages, "2024-02-02T00:00:00Z", region="R48")
    late = latest_balance_as_of(vintages, "2024-02-04T00:00:00Z", region="R48")

    assert features.loc[0, "local_balance_lag1"] == pytest.approx(10.0)
    assert features.loc[0, "net_inflow_balancing_lag1"] == pytest.approx(4.0)
    assert features.loc[0, "net_inflow_balancing_rolling_4wk"] == pytest.approx(2.5)
    assert features.loc[0, "balance_history_weeks"] == 4
    assert early.iloc[-1]["local_balance"] == pytest.approx(10.0)
    assert late.iloc[-1]["local_balance"] == pytest.approx(99.0)


def test_asof_pipelines_materialize_processed_artifacts():
    scenarios = pd.DataFrame(
        {
            "date": ["2024-02-02"],
            "duoarea": ["R48"],
            "issued_at": ["2024-02-01T00:00:00Z"],
            "temperature_f": [45.0],
            "hdd": [70.0],
            "cdd": [0.0],
            "weather_days": [7.0],
        }
    )
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        tmp_path = Path(temp_dir)
        scenarios_path = tmp_path / "weather_vintages.parquet"
        scenarios.to_parquet(scenarios_path, index=False)

        weather_path = run_weather_scenario_pipeline(
            "R48",
            scenarios_path=scenarios_path,
            as_of="2024-02-02T00:00:00Z",
            processed_dir=tmp_path,
        )

        source_dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
        vintages = pd.DataFrame(
            {
                "date": source_dates,
                "duoarea": ["R48"] * 4,
                "available_at": ["2024-02-01T00:00:00Z"] * 4,
                "local_balance": [4.0, 5.0, 6.0, 10.0],
                "net_inflow_balancing": [1.0, 2.0, 3.0, 4.0],
            }
        )
        vintages_path = tmp_path / "balance_vintages.parquet"
        vintages.to_parquet(vintages_path, index=False)
        origins_path = tmp_path / "origins.parquet"
        pd.DataFrame(
            {"date": [pd.Timestamp("2024-02-02")], "duoarea": ["R48"]}
        ).to_parquet(origins_path, index=False)

        balance_path = run_asof_balance_pipeline(
            "R48",
            vintages_path=vintages_path,
            origins_path=origins_path,
            processed_dir=tmp_path,
        )

        assert pd.read_parquet(weather_path).loc[0, "hdd"] == pytest.approx(70.0)
        assert pd.read_parquet(balance_path).loc[
            0, "local_balance_lag1"
        ] == pytest.approx(10.0)


def test_live_forecast_pipeline_materializes_member_and_weekly_archives(
    monkeypatch,
):
    locations = pd.DataFrame(
        {
            "STNAME": ["Alpha"],
            "duoarea": ["R31"],
            "LATITUDE": [40.0],
            "LONGITUDE": [-75.0],
            "WEIGHT": [1.0],
        }
    )
    dates = pd.date_range("2024-01-06", periods=7, freq="D").strftime("%Y-%m-%d").tolist()
    archive = parse_open_meteo_ensemble_response(
        locations,
        {
            "daily": {
                "time": dates,
                "temperature_2m_mean_member01": [40.0] * 7,
                "temperature_2m_mean_member02": [50.0] * 7,
            }
        },
        issued_at="2024-01-05T00:00:00Z",
    )
    monkeypatch.setattr(
        "gas_forecast.pipelines.asof._forecast_locations",
        lambda region: locations,
    )
    monkeypatch.setattr(
        "gas_forecast.pipelines.asof.fetch_open_meteo_gefs_ensemble",
        lambda *args, **kwargs: archive,
    )

    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        outputs = run_live_weather_forecast_pipeline(
            "R31",
            issued_at="2024-01-05T00:00:00Z",
            processed_dir=temp_dir,
        )

        assert set(outputs.paths) == {
            "state_forecast_archive",
            "regional_forecast_archive",
            "weekly_weather_scenarios",
        }
        weekly = pd.read_parquet(outputs.paths["weekly_weather_scenarios"])
        assert weekly.loc[0, "ensemble_members"] == 2
        assert weekly.loc[0, "duoarea"] == "R31"
