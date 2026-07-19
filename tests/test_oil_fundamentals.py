import numpy as np
import pandas as pd
import pytest

from oil_forecast.backtesting import run_oil_backtest
from oil_forecast.fundamentals import (
    OilFundamentalsModel,
    build_weekly_crude_balance,
)


def _long_week(period, **values):
    return [
        {"period": period, "component": component, "value": value}
        for component, value in values.items()
    ]


def _synthetic_balance(periods=180):
    dates = pd.date_range("2022-01-07", periods=periods, freq="W-FRI")
    seasonal = np.sin(2 * np.pi * np.arange(periods) / 52.0)
    production = 90.0 + seasonal
    imports = 42.0 + 2.0 * seasonal
    refinery = 112.0 + 4.0 * seasonal
    exports = 28.0 - seasonal
    spr_change = np.zeros(periods)
    fundamental = production + imports - refinery - exports - spr_change
    adjustment = 7.0 + 0.5 * seasonal
    actual = fundamental + adjustment
    return pd.DataFrame(
        {
            "date": dates,
            "production_mmbbl": production,
            "imports_mmbbl": imports,
            "refinery_inputs_mmbbl": refinery,
            "exports_mmbbl": exports,
            "spr_stock_change_mmbbl": spr_change,
            "balance_adjustment_mmbbl": adjustment,
            "commercial_stock_change_mmbbl": actual,
        }
    )


def test_build_weekly_crude_balance_converts_units_and_closes_identity():
    rows = []
    rows.extend(
        _long_week(
            "2026-01-02",
            production_kbpd=10_000,
            imports_kbpd=5_000,
            refinery_inputs_kbpd=12_000,
            exports_kbpd=2_000,
            commercial_stocks_kb=400_000,
            spr_stocks_kb=100_000,
        )
    )
    rows.extend(
        _long_week(
            "2026-01-09",
            production_kbpd=10_000,
            imports_kbpd=5_000,
            refinery_inputs_kbpd=12_000,
            exports_kbpd=2_000,
            commercial_stocks_kb=406_000,
            spr_stocks_kb=102_000,
        )
    )

    result = build_weekly_crude_balance(pd.DataFrame(rows))

    assert result.loc[1, "production_mmbbl"] == pytest.approx(70.0)
    assert result.loc[1, "commercial_stock_change_mmbbl"] == pytest.approx(6.0)
    assert result.loc[1, "spr_stock_change_mmbbl"] == pytest.approx(2.0)
    assert result.loc[1, "fundamental_balance_mmbbl"] == pytest.approx(5.0)
    assert result.loc[1, "balance_adjustment_mmbbl"] == pytest.approx(1.0)


def test_build_weekly_crude_balance_rejects_incomplete_week():
    rows = _long_week(
        "2026-01-02",
        production_kbpd=10_000,
        imports_kbpd=5_000,
        refinery_inputs_kbpd=12_000,
        exports_kbpd=2_000,
        commercial_stocks_kb=400_000,
        spr_stocks_kb=100_000,
    )
    rows.extend(
        _long_week(
            "2026-01-09",
            production_kbpd=10_000,
            imports_kbpd=5_000,
            refinery_inputs_kbpd=12_000,
            exports_kbpd=2_000,
            commercial_stocks_kb=406_000,
        )
    )

    with pytest.raises(ValueError, match="incomplete component rows"):
        build_weekly_crude_balance(pd.DataFrame(rows))


def test_build_weekly_crude_balance_rejects_non_friday_periods():
    rows = _long_week(
        "2026-01-01",
        production_kbpd=10_000,
        imports_kbpd=5_000,
        refinery_inputs_kbpd=12_000,
        exports_kbpd=2_000,
        commercial_stocks_kb=400_000,
        spr_stocks_kb=100_000,
    )

    with pytest.raises(ValueError, match="Friday week-ending dates"):
        build_weekly_crude_balance(pd.DataFrame(rows))


def test_build_weekly_crude_balance_rejects_negative_raw_value():
    rows = _long_week(
        "2026-01-02",
        production_kbpd=10_000,
        imports_kbpd=-1,
        refinery_inputs_kbpd=12_000,
        exports_kbpd=2_000,
        commercial_stocks_kb=400_000,
        spr_stocks_kb=100_000,
    )

    with pytest.raises(ValueError, match="negative values"):
        build_weekly_crude_balance(pd.DataFrame(rows))


def test_oil_model_combines_forecast_components_through_balance_identity():
    balance = _synthetic_balance(120)
    target = balance["date"].max() + pd.Timedelta(days=7)

    forecast = OilFundamentalsModel().fit(balance).predict([target]).iloc[0]

    expected = (
        forecast["production_forecast_mmbbl"]
        + forecast["imports_forecast_mmbbl"]
        - forecast["refinery_inputs_forecast_mmbbl"]
        - forecast["exports_forecast_mmbbl"]
        - forecast["spr_stock_change_forecast_mmbbl"]
    )
    assert forecast["fundamental_balance_forecast_mmbbl"] == pytest.approx(expected)
    assert forecast["prediction_mmbbl"] == pytest.approx(
        expected + forecast["balance_adjustment_forecast_mmbbl"]
    )


def test_oil_model_rejects_nonconsecutive_weekly_balance():
    balance = _synthetic_balance(60).drop(index=30)

    with pytest.raises(ValueError, match="consecutive weekly rows"):
        OilFundamentalsModel().fit(balance)


def test_oil_model_rejects_non_friday_dates():
    balance = _synthetic_balance(60)
    balance["date"] -= pd.Timedelta(days=1)

    with pytest.raises(ValueError, match="Friday week-ending dates"):
        OilFundamentalsModel().fit(balance)


def test_oil_model_rejects_interior_missing_component_value():
    balance = _synthetic_balance(60)
    balance.loc[30, "imports_mmbbl"] = np.nan

    with pytest.raises(ValueError, match="missing values in imports_mmbbl"):
        OilFundamentalsModel().fit(balance)


def test_oil_model_rejects_non_finite_component_value():
    balance = _synthetic_balance(60)
    balance.loc[30, "production_mmbbl"] = np.inf

    with pytest.raises(ValueError, match="non-finite values in production_mmbbl"):
        OilFundamentalsModel().fit(balance)


def test_oil_model_rejects_negative_physical_flow():
    balance = _synthetic_balance(60)
    balance.loc[30, "imports_mmbbl"] = -1.0

    with pytest.raises(ValueError, match="negative values in imports_mmbbl"):
        OilFundamentalsModel().fit(balance)


def test_backtest_prediction_does_not_use_target_week_components():
    balance = _synthetic_balance()
    target_date = balance.loc[80, "date"]
    baseline_predictions, _ = run_oil_backtest(
        balance,
        initial_train_weeks=60,
        min_calibration=5,
    )
    changed = balance.copy()
    changed.loc[80, [
        "production_mmbbl",
        "imports_mmbbl",
        "refinery_inputs_mmbbl",
        "exports_mmbbl",
        "balance_adjustment_mmbbl",
    ]] = 99_999.0
    changed.loc[80, "commercial_stock_change_mmbbl"] = -99_999.0
    changed_predictions, _ = run_oil_backtest(
        changed,
        initial_train_weeks=60,
        min_calibration=5,
    )

    baseline_value = baseline_predictions.loc[
        baseline_predictions["date"] == target_date.tz_localize("UTC"),
        "prediction_mmbbl",
    ].iloc[0]
    changed_value = changed_predictions.loc[
        changed_predictions["date"] == target_date.tz_localize("UTC"),
        "prediction_mmbbl",
    ].iloc[0]
    assert changed_value == pytest.approx(baseline_value)


def test_oil_backtest_reports_baseline_and_calibrates_from_prior_origins():
    predictions, metrics = run_oil_backtest(
        _synthetic_balance(),
        initial_train_weeks=60,
        min_calibration=5,
    )

    assert set(metrics["model"]) == {
        "seasonal_level_fundamentals",
        "last_change_baseline",
    }
    assert predictions.loc[0, "calibration_count"] == 0
    assert predictions["lower_bound_mmbbl"].notna().any()
