import pandas as pd
import pytest

from gas_forecast.data.weather_forecasts import (
    aggregate_forecasts_to_weekly_scenarios,
    aggregate_state_forecasts,
    build_asof_weather_features,
    parse_open_meteo_ensemble_response,
    parse_open_meteo_previous_runs_response,
    select_state_weights_as_of,
    validate_weather_forecast_archive,
)


def _locations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STNAME": ["Alpha", "Beta"],
            "duoarea": ["R31", "R31"],
            "LATITUDE": [40.0, 41.0],
            "LONGITUDE": [-75.0, -76.0],
        }
    )


def _payload() -> list[dict]:
    dates = pd.date_range("2024-01-06", periods=7, freq="D").strftime("%Y-%m-%d").tolist()
    return [
        {
            "daily": {
                "time": dates,
                "temperature_2m_mean_member01": [40.0] * 7,
                "temperature_2m_mean_member02": [50.0] * 7,
            }
        },
        {
            "daily": {
                "time": dates,
                "temperature_2m_mean_member01": [60.0] * 7,
                "temperature_2m_mean_member02": [70.0] * 7,
            }
        },
    ]


def test_ensemble_archive_aggregates_state_weights_and_weekly_quantiles():
    archive = parse_open_meteo_ensemble_response(
        _locations(),
        _payload(),
        issued_at="2024-01-05T00:00:00Z",
    )
    weights = pd.DataFrame(
        {
            "state": ["Alpha", "Beta"],
            "duoarea": ["R31", "R31"],
            "weather_weight": [0.75, 0.25],
        }
    )

    regional = aggregate_state_forecasts(archive, weights)
    scenarios = aggregate_forecasts_to_weekly_scenarios(regional)

    assert len(archive) == 28
    assert len(regional) == 14
    assert len(scenarios) == 1
    assert scenarios.loc[0, "date"] == pd.Timestamp("2024-01-12")
    assert scenarios.loc[0, "temperature_f"] == pytest.approx(50.0)
    assert scenarios.loc[0, "hdd"] == pytest.approx(109.375)
    assert scenarios.loc[0, "ensemble_members"] == 2
    assert scenarios.loc[0, "hdd_spread"] > 0


def test_gas_demand_weights_exclude_future_revisions():
    history = pd.DataFrame(
        {
            "state": ["Alpha", "Beta", "Alpha"],
            "duoarea": ["R31", "R31", "R31"],
            "available_at": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
                "2024-02-01T00:00:00Z",
            ],
            "gas_load": [75.0, 25.0, 1.0],
        }
    )

    weights = select_state_weights_as_of(
        history,
        "2024-01-15T00:00:00Z",
        weight_col="gas_load",
    )

    assert weights.set_index("state").loc["Alpha", "weather_weight"] == pytest.approx(0.75)
    assert weights["weather_weight"].sum() == pytest.approx(1.0)


def test_archive_rejects_forecast_issued_after_valid_time():
    archive = parse_open_meteo_ensemble_response(
        _locations().iloc[[0]],
        _payload()[0],
        issued_at="2024-01-05T00:00:00Z",
    )
    archive["issued_at"] = "2024-01-07T00:00:00Z"

    with pytest.raises(ValueError, match="issued after"):
        validate_weather_forecast_archive(archive)


def test_asof_weather_features_never_select_future_run():
    scenarios = pd.DataFrame(
        {
            "date": ["2024-01-12", "2024-01-12"],
            "duoarea": ["R31", "R31"],
            "issued_at": ["2024-01-11T00:00:00Z", "2024-01-13T00:00:00Z"],
            "temperature_f": [40.0, 70.0],
            "hdd": [175.0, 0.0],
            "cdd": [0.0, 35.0],
            "weather_days": [7.0, 7.0],
            "hdd_p10": [160.0, 0.0],
            "hdd_p90": [190.0, 0.0],
        }
    )
    origins = pd.DataFrame({"date": ["2024-01-12"], "duoarea": ["R31"]})

    features = build_asof_weather_features(origins, scenarios, horizon_weeks=1)

    assert features.loc[0, "hdd"] == pytest.approx(175.0)
    assert features.loc[0, "horizon"] == 1


def test_previous_runs_reconstruct_one_complete_week_from_fixed_leads():
    dates = pd.date_range("2024-01-05", periods=8, freq="D")
    daily = {"time": dates.strftime("%Y-%m-%d").tolist()}
    for lead in range(8):
        values = [None] * 8
        values[lead] = 50.0
        daily[f"temperature_2m_mean_previous_day{lead}"] = values

    archive = parse_open_meteo_previous_runs_response(
        _locations().iloc[[0]],
        {"daily": daily},
    )
    scenarios = aggregate_forecasts_to_weekly_scenarios(archive)

    assert len(archive) == 8
    assert len(scenarios) == 1
    assert scenarios.loc[0, "date"] == pd.Timestamp("2024-01-12")
    assert scenarios.loc[0, "hdd"] == pytest.approx(105.0)
