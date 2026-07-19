from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import numpy as np
import pytest

from gas_forecast.data import balance_api
from gas_forecast.data.balance_api import NATIONAL_BASELINE_SERIES, STATE_TO_ABBR
from gas_forecast.pipelines.balance import aggregate_price_to_weeks
from gas_forecast.modeling.models import StructuralDisaggregator

def test_state_abbreviations():
    assert STATE_TO_ABBR["Texas"] == "TX"
    assert STATE_TO_ABBR["California"] == "CA"
    assert STATE_TO_ABBR["Florida"] == "FL"

def test_aggregate_price_to_weeks():
    # Create daily prices
    dates = pd.date_range(start="2026-01-01", end="2026-01-15")
    prices = pd.DataFrame({
        "period": dates,
        "value": np.linspace(2.0, 3.0, len(dates))
    })
    
    # Weekly Friday dates
    weekly_dates = pd.Series([
        pd.Timestamp("2026-01-09"),
        pd.Timestamp("2026-01-16")
    ])
    
    res = aggregate_price_to_weeks(prices, weekly_dates)
    assert len(res) == 2
    assert res.iloc[0]["period"] == pd.Timestamp("2026-01-09")
    # For 2026-01-09, the 7-day range is 2026-01-03 to 2026-01-09
    expected_mean = prices[(prices["period"] >= "2026-01-03") & (prices["period"] <= "2026-01-09")]["value"].mean()
    assert pytest.approx(res.iloc[0]["value"]) == expected_mean


def test_monthly_fetch_requests_national_baselines_separately(monkeypatch):
    requested_series = []

    def fake_fetch(url, params, timeout=30.0):
        requested_series.append(tuple(params["facets[series][]"]))
        return pd.DataFrame(
            {
                "period": ["2024-01"],
                "series": [params["facets[series][]"][0]],
                "value": [1.0],
            }
        )

    monkeypatch.setattr(balance_api, "fetch_eia_api_paginated", fake_fetch)

    balance_api.fetch_monthly_state_data_raw("test-key", ["Texas"])

    assert len(requested_series) == 2
    assert set(requested_series[-1]) == set(NATIONAL_BASELINE_SERIES)
    assert not set(requested_series[0]) & set(NATIONAL_BASELINE_SERIES)


def test_incomplete_monthly_cache_is_refetched(monkeypatch):
    refreshed = False

    def fake_fetch(api_key, states, start_date="2010-01-01", end_date=None):
        nonlocal refreshed
        refreshed = True
        series = [
            "N3010TX2",
            "N3020TX2",
            "N3035TX2",
            "N3045TX2",
            "N9050TX2",
            *NATIONAL_BASELINE_SERIES,
        ]
        return pd.DataFrame(
            {
                "period": ["2024-01"] * len(series),
                "series": series,
                "value": [1.0] * len(series),
            }
        )

    monkeypatch.setattr(balance_api, "fetch_monthly_state_data_raw", fake_fetch)

    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        cache_dir = Path(temp_dir)
        cache_path = cache_dir / "balance" / "state_monthly_TX.parquet"
        cache_path.parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "period": [pd.Timestamp("2024-01-01")],
                "series": ["N3010TX2"],
                "value": [1.0],
            }
        ).to_parquet(cache_path, index=False)

        result = balance_api.get_monthly_state_data(
            "test-key",
            ["Texas"],
            cache_dir=cache_dir,
        )

    assert refreshed
    assert set(NATIONAL_BASELINE_SERIES).issubset(set(result["series"]))


def test_monthly_cache_requires_each_requested_states_consumption_inputs():
    texas_series = [
        "N3010TX2",
        "N3020TX2",
        "N3035TX2",
        "N3045TX2",
        "N9050TX2",
        *NATIONAL_BASELINE_SERIES,
    ]
    cache = pd.DataFrame(
        {
            "period": [pd.Timestamp("2024-01-01")] * len(texas_series),
            "series": texas_series,
            "value": [1.0] * len(texas_series),
        }
    )

    assert not balance_api._monthly_cache_is_usable(
        cache,
        ["Texas", "Oklahoma"],
    )


def test_monthly_cache_accepts_eia_dry_production_series_identifier():
    california_series = [
        "N3010CA2",
        "N3020CA2",
        "N3035CA2",
        "N3045CA2",
        "NA1160_SCA_2",
        *NATIONAL_BASELINE_SERIES,
    ]
    cache = pd.DataFrame(
        {
            "period": [pd.Timestamp("2024-01-01")] * len(california_series),
            "series": california_series,
            "value": [1.0] * len(california_series),
        }
    )

    assert balance_api._monthly_cache_is_usable(cache, ["California"])

def test_structural_disaggregator_fit_predict():
    # 1. Create mock monthly EIA data for 2 states (CA, TX)
    periods = pd.date_range(start="2024-01-01", end="2024-12-01", freq="MS")
    records = []
    for p in periods:
        # National baseline series
        records.extend([
            {"period": p.strftime("%Y-%m"), "series": "N9070US2", "value": 200000.0, "duoarea": "NUS"},
            {"period": p.strftime("%Y-%m"), "series": "N9050US2", "value": 200000.0, "duoarea": "NUS"},
            {"period": p.strftime("%Y-%m"), "series": "N9140US2", "value": 200000.0, "duoarea": "NUS"},
            {"period": p.strftime("%Y-%m"), "series": "N9160US2", "value": 4000.0, "duoarea": "NUS"},
            {"period": p.strftime("%Y-%m"), "series": "N9170US2", "value": 2000.0, "duoarea": "NUS"},
        ])
        for state in ["CA", "TX"]:
            records.extend([
                {
                    "period": p.strftime("%Y-%m"),
                    "series": (
                        "NA1160_SCA_2" if state == "CA" else f"N9050{state}2"
                    ),
                    "value": 100000.0,
                    "duoarea": f"S{state}",
                },  # State dry or marketed production (100 Bcf)
                {"period": p.strftime("%Y-%m"), "series": f"N3010{state}2", "value": 20000.0, "duoarea": f"S{state}"},      # Res (20 Bcf)
                {"period": p.strftime("%Y-%m"), "series": f"N3020{state}2", "value": 10000.0, "duoarea": f"S{state}"},      # Com (10 Bcf)
                {"period": p.strftime("%Y-%m"), "series": f"N3035{state}2", "value": 30000.0, "duoarea": f"S{state}"},      # Ind (30 Bcf)
                {"period": p.strftime("%Y-%m"), "series": f"N3045{state}2", "value": 40000.0, "duoarea": f"S{state}"},      # Power (40 Bcf)
            ])
    monthly_df = pd.DataFrame(records)
    monthly_df["period"] = pd.to_datetime(monthly_df["period"] + "-01")
    monthly_df["value"] = pd.to_numeric(monthly_df["value"])

    # 2. Create mock daily weather
    daily_dates = pd.date_range(start="2024-01-01", end="2024-12-31")
    daily_weather = pd.DataFrame({
        "date": daily_dates,
        "hdd": np.random.uniform(5, 25, len(daily_dates)),
        "cdd": np.random.uniform(0, 10, len(daily_dates)),
        "temperature_f": np.random.uniform(30, 80, len(daily_dates))
    })

    # 3. Create mock daily prices
    daily_price = pd.DataFrame({
        "period": daily_dates,
        "value": np.random.uniform(2.0, 4.0, len(daily_dates))
    })

    # 4. Instantiate and fit disaggregator
    disagg = StructuralDisaggregator()
    states = ["California", "Texas"]
    disagg.fit(monthly_df, daily_weather, daily_price, states)

    # 5. Predict on weekly weather
    weekly_dates = pd.date_range(start="2024-01-05", end="2024-12-27", freq="W-FRI")
    weekly_weather = pd.DataFrame({
        "date": weekly_dates,
        "hdd": np.random.uniform(50, 150, len(weekly_dates)),
        "cdd": np.random.uniform(0, 50, len(weekly_dates)),
        "temperature_f": np.random.uniform(35, 75, len(weekly_dates)),
        "month": weekly_dates.month,
        "week_of_year": weekly_dates.isocalendar().week,
        "duoarea": "R48"
    })
    
    weekly_price = pd.DataFrame({
        "period": weekly_dates,
        "value": np.random.uniform(2.0, 4.0, len(weekly_dates))
    })

    weekly_balance = disagg.predict_weekly(weekly_weather, weekly_price)

    assert not weekly_balance.empty
    assert "res_com" in weekly_balance.columns
    assert "power_burn" in weekly_balance.columns
    assert "industrial" in weekly_balance.columns
    assert "dry_production" in weekly_balance.columns
    assert "local_balance" in weekly_balance.columns

    # Verify dry production is roughly around expected scale
    # Total monthly was 200 Bcf (2 states * 100 Bcf each). Daily ~6.6 Bcf. Weekly ~46 Bcf.
    assert weekly_balance["dry_production"].mean() == pytest.approx(46.66, rel=0.1)
