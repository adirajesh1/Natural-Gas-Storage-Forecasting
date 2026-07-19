import numpy as np
import pandas as pd
import pytest

from gas_forecast.modeling.reconciliation import ALL_STORAGE_REGIONS, STORAGE_REGIONS
from gas_forecast.modeling.regional_backtesting import reconcile_backtest_predictions


def _base_predictions() -> pd.DataFrame:
    rows = []
    for origin_index, origin in enumerate(
        pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    ):
        target = origin + pd.Timedelta(weeks=1)
        child_actuals = np.asarray([10.0, 11.0, 12.0, 9.0, 8.0]) + origin_index
        actuals = np.r_[child_actuals.sum(), child_actuals]
        errors = np.asarray([2.0, 1.0, -1.0, 0.5, -0.5, 0.25]) * (origin_index + 1)
        for region, actual, error in zip(
            ALL_STORAGE_REGIONS, actuals, errors, strict=True
        ):
            rows.append(
                {
                    "date": target,
                    "forecast_origin": origin,
                    "region": region,
                    "horizon": 1,
                    "model_key": "ridge",
                    "predicted_weekly_change": actual - error,
                    "actual_weekly_change": actual,
                    "p10": actual - error - 5.0,
                    "p50": actual - error,
                    "p90": actual - error + 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_reconciled_backtest_adds_all_paths_without_using_future_errors():
    reconciled = reconcile_backtest_predictions(_base_predictions())

    assert {"direct", "bottom_up", "mint_shrink"}.issubset(
        set(reconciled["reconciliation_method"])
    )
    for (_, method), group in reconciled.groupby(
        ["forecast_origin", "reconciliation_method"]
    ):
        if method == "direct":
            assert group["region"].tolist() == ["R48"]
            continue
        indexed = group.set_index("region")
        assert indexed.loc["R48", "predicted_weekly_change"] == pytest.approx(
            indexed.loc[list(STORAGE_REGIONS), "predicted_weekly_change"].sum()
        )
        assert indexed.loc["R48", "p10"] == pytest.approx(
            indexed.loc[list(STORAGE_REGIONS), "p10"].sum()
        )


def test_future_residuals_do_not_change_earlier_mint_forecasts():
    base = _base_predictions()
    original = reconcile_backtest_predictions(base)
    future = base.loc[base["forecast_origin"] == base["forecast_origin"].max()].copy()
    future["forecast_origin"] = pd.Timestamp("2025-01-03")
    future["date"] = pd.Timestamp("2025-01-10")
    future["actual_weekly_change"] += 1_000_000.0
    extended = reconcile_backtest_predictions(pd.concat([base, future], ignore_index=True))

    cutoff = base["forecast_origin"].max()
    expected = original.loc[original["forecast_origin"] <= cutoff].reset_index(drop=True)
    actual = extended.loc[extended["forecast_origin"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual)


def test_partial_latest_origin_is_excluded_from_hierarchy_paths():
    base = _base_predictions()
    partial = base.iloc[[0]].copy()
    partial["forecast_origin"] = pd.Timestamp("2024-03-01")
    partial["date"] = pd.Timestamp("2024-03-08")

    reconciled = reconcile_backtest_predictions(pd.concat([base, partial], ignore_index=True))

    assert pd.Timestamp("2024-03-01") not in set(reconciled["forecast_origin"])
