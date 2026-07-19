from sklearn.base import clone

from gas_forecast.modeling.config import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_FOURIER_HARMONICS,
    DEFAULT_LOOKBACK_YEARS,
    DEFAULT_TARGET_COLUMN,
    FOURIER_HARMONIC_GRID,
    build_fourier_model,
    legacy_forecast_model_configs,
    one_step_model_configs,
    sklearn_model_configs,
)
from gas_forecast.modeling.models import (
    FiveYearWeeklyAverageModel,
    WeeklyChangeFourierRegressionModel,
)


def test_legacy_forecast_model_configs_build_expected_models():
    configs = {config.key: config for config in legacy_forecast_model_configs()}

    baseline = configs["five_year_average"].build()
    fourier = configs["fourier"].build()

    assert isinstance(baseline, FiveYearWeeklyAverageModel)
    assert baseline.lookback_years == DEFAULT_LOOKBACK_YEARS
    assert isinstance(fourier, WeeklyChangeFourierRegressionModel)
    assert fourier.n_harmonics == DEFAULT_FOURIER_HARMONICS


def test_build_fourier_model_uses_shared_lookback_config():
    model = build_fourier_model(3)

    assert model.lookback_years == DEFAULT_LOOKBACK_YEARS
    assert model.n_harmonics == 3
    assert 3 in FOURIER_HARMONIC_GRID


def test_sklearn_model_configs_build_cloneable_estimators():
    configs = sklearn_model_configs()
    keys = {config.key for config in configs}

    assert {
        "linear_regression",
        "ridge",
        "elastic_net",
        "random_forest",
        "hist_gradient_boosting",
        "hist_gradient_boosting_p10",
        "hist_gradient_boosting_p90",
        "linear_hgb_ensemble",
    }.issubset(keys)

    for config in configs:
        estimator = config.build()
        clone(estimator)
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict")


def test_arimax_is_only_in_the_one_step_model_ladder():
    assert "arimax" not in {config.key for config in sklearn_model_configs()}
    assert "arimax" in {config.key for config in one_step_model_configs()}


def test_default_feature_and_target_config_are_nonempty():
    assert DEFAULT_TARGET_COLUMN == "weekly_change_bcf"
    assert DEFAULT_FEATURE_COLUMNS


def test_dynamic_model_configs_presence():
    configs = sklearn_model_configs()
    keys = {config.key for config in configs}

    # Verify presence based on actual importability
    try:
        import lightgbm  # noqa: F401
        has_lgb = True
    except ImportError:
        has_lgb = False

    try:
        import xgboost  # noqa: F401
        has_xgb = True
    except ImportError:
        has_xgb = False

    assert ("lightgbm" in keys) == has_lgb
    assert ("xgboost" in keys) == has_xgb

    # If present, check their attributes
    configs_dict = {config.key: config for config in configs}
    if has_lgb:
        lgb_config = configs_dict["lightgbm"]
        assert lgb_config.label == "LightGBM Regressor"
        estimator = lgb_config.build()
        assert estimator.n_estimators == 300
        assert estimator.learning_rate == 0.05
        assert estimator.random_state == 42
        assert estimator.n_jobs == -1

    if has_xgb:
        xgb_config = configs_dict["xgboost"]
        assert xgb_config.label == "XGBoost Regressor"
        estimator = xgb_config.build()
        assert estimator.n_estimators == 300
        assert estimator.learning_rate == 0.05
        assert estimator.random_state == 42
        assert estimator.n_jobs == -1
