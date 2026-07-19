import builtins

import numpy as np
import pandas as pd
import pytest

from gas_forecast.modeling.experiments import (
    evaluate_challenger_promotion,
    summarize_ablation,
)
from gas_forecast.modeling.models import ARIMAXRegressor, PooledNHITSForecaster
from gas_forecast.modeling.reconciliation import ALL_STORAGE_REGIONS


def _promotion_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for region in ALL_STORAGE_REGIONS:
        for index, date in enumerate(pd.date_range("2024-01-05", periods=10, freq="W-FRI")):
            rows.append(
                {
                    "date": date,
                    "region": region,
                    "horizon": 1,
                    "actual_weekly_change": 0.0,
                    "predicted_weekly_change": 5.0,
                    "p10": -1.0 if index < 8 else 1.0,
                    "p90": 1.0 if index < 8 else 2.0,
                }
            )
    baseline = pd.DataFrame(rows)
    challenger = baseline.copy()
    challenger["predicted_weekly_change"] = 3.0
    return baseline, challenger


def test_promotion_gate_requires_accuracy_significance_stability_and_coverage():
    baseline, challenger = _promotion_frames()

    result = evaluate_challenger_promotion(baseline, challenger)

    assert result["mae_improvement"] == pytest.approx(2.0)
    assert result["interval_coverage"] == pytest.approx(0.8)
    assert result["promote"] is True


def test_ablation_summary_groups_the_experiment_dimensions():
    baseline, _ = _promotion_frames()
    baseline = baseline.assign(
        weather_input="archived",
        weather_weighting="gas_load",
        reconciliation_method="mint_shrink",
        model_key="ridge",
    )

    summary = summarize_ablation(baseline)

    assert len(summary) == len(ALL_STORAGE_REGIONS)
    assert summary["mae"].eq(5.0).all()
    assert summary["interval_coverage"].eq(0.8).all()


def test_arimax_is_a_cloneable_one_step_challenger():
    dates = pd.date_range("2023-01-06", periods=40, freq="W-FRI")
    X = pd.DataFrame({"hdd": np.linspace(50.0, 5.0, len(dates))})
    y = 2.0 + 0.5 * X["hdd"]

    model = ARIMAXRegressor(order=(1, 0, 0), maxiter=20).fit(X.iloc[:-4], y.iloc[:-4])
    predictions = model.predict(X.iloc[-4:])

    assert model.supports_recursive is False
    assert len(predictions) == 4
    assert np.isfinite(predictions).all()


def test_nhits_dependency_is_optional_and_actionable(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("neuralforecast"):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    panel = pd.DataFrame(
        {
            "unique_id": ["R48"],
            "ds": ["2024-01-05"],
            "y": [1.0],
            "hdd": [10.0],
            "cdd": [0.0],
            "hdd_p10": [8.0],
            "hdd_p90": [12.0],
            "cdd_p10": [0.0],
            "cdd_p90": [1.0],
        }
    )

    with pytest.raises(ImportError, match="optional neural dependencies"):
        PooledNHITSForecaster().fit(panel)

