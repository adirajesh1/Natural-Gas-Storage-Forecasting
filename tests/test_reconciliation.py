import numpy as np
import pandas as pd
import pytest

from gas_forecast.modeling.reconciliation import (
    ALL_STORAGE_REGIONS,
    STORAGE_REGIONS,
    bottom_up_reconcile,
    direct_lower48_forecast,
    mint_shrink_reconcile,
    reconciliation_error,
)


def _forecasts() -> pd.DataFrame:
    values = [52.0, 10.0, 11.0, 12.0, 9.0, 8.0]
    return pd.DataFrame(
        {
            "date": pd.Timestamp("2024-03-01"),
            "horizon": 1,
            "region": ALL_STORAGE_REGIONS,
            "predicted_weekly_change": values,
            "p10": np.asarray(values) - 3.0,
            "p50": values,
            "p90": np.asarray(values) + 3.0,
            "model_key": "ridge",
        }
    )


def _residuals() -> pd.DataFrame:
    rows = []
    for date_index, date in enumerate(pd.date_range("2023-12-01", periods=10, freq="W-FRI")):
        bottom = np.asarray([1.0, -1.0, 0.5, -0.5, 0.25]) * (date_index + 1)
        values = np.r_[bottom.sum() + (-1) ** date_index, bottom]
        for region, value in zip(ALL_STORAGE_REGIONS, values, strict=True):
            rows.append({"date": date, "horizon": 1, "region": region, "residual": value})
    return pd.DataFrame(rows)


def test_direct_path_only_publishes_lower48():
    direct = direct_lower48_forecast(_forecasts())

    assert direct["region"].tolist() == ["R48"]
    assert direct.loc[0, "reconciliation_method"] == "direct"


def test_bottom_up_forecasts_are_exactly_coherent():
    reconciled = bottom_up_reconcile(
        _forecasts(),
        value_cols=("predicted_weekly_change", "p10", "p50", "p90"),
    )

    assert set(reconciled["region"]) == set(ALL_STORAGE_REGIONS)
    for column in ("predicted_weekly_change", "p10", "p50", "p90"):
        parent = reconciled.set_index("region").loc["R48", column]
        children = reconciled.set_index("region").loc[list(STORAGE_REGIONS), column].sum()
        assert parent == children
    assert reconciliation_error(reconciled).loc[0, "reconciliation_error"] == pytest.approx(0.0)


def test_mint_shrink_is_coherent_and_ignores_future_residuals():
    historical = _residuals()
    future = historical.iloc[[0]].copy()
    future["date"] = "2025-01-03"
    future["residual"] = 1_000_000.0

    expected = mint_shrink_reconcile(
        _forecasts(),
        historical,
        as_of="2024-02-23",
        value_cols=("predicted_weekly_change", "p10", "p50", "p90"),
    )
    actual = mint_shrink_reconcile(
        _forecasts(),
        pd.concat([historical, future], ignore_index=True),
        as_of="2024-02-23",
        value_cols=("predicted_weekly_change", "p10", "p50", "p90"),
    )

    pd.testing.assert_frame_equal(expected, actual)
    assert reconciliation_error(actual).loc[0, "reconciliation_error"] == 0.0
    assert (actual["p10"] <= actual["p50"]).all()
    assert (actual["p50"] <= actual["p90"]).all()


def test_bottom_up_allows_intervals_before_calibration_is_available():
    forecasts = _forecasts()
    forecasts[["p10", "p90"]] = pd.NA

    reconciled = bottom_up_reconcile(forecasts)

    assert reconciled["p10"].isna().all()
    assert reconciled["p90"].isna().all()
    assert reconciliation_error(reconciled).loc[0, "reconciliation_error"] == pytest.approx(0.0)
