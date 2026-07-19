from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gas_forecast.data.features import DEFAULT_WEATHER_MODEL_FEATURES, TARGET_COLUMN
from gas_forecast.modeling.models import (
    FiveYearWeeklyAverageModel,
    ARIMAXRegressor,
    WeeklyChangeForecastModel,
    WeeklyChangeFourierRegressionModel,
    WeeklyChangeLinearRegressionModel,
    WeeklyChangeSARIMAModel,
)


@dataclass(frozen=True)
class ForecastModelConfig:
    """Named configuration for legacy forecast model classes."""

    key: str
    label: str
    factory: Callable[[], WeeklyChangeForecastModel]

    def build(self) -> WeeklyChangeForecastModel:
        return self.factory()


@dataclass(frozen=True)
class SklearnModelConfig:
    """Named configuration for sklearn-compatible estimators."""

    key: str
    label: str
    factory: Callable[[], object]

    def build(self):
        return self.factory()


DEFAULT_LOOKBACK_YEARS = 5
DEFAULT_FOURIER_HARMONICS = 7
FOURIER_HARMONIC_GRID = tuple(range(1, 11))
DEFAULT_TARGET_COLUMN = TARGET_COLUMN
DEFAULT_FEATURE_COLUMNS = DEFAULT_WEATHER_MODEL_FEATURES


LEGACY_FORECAST_MODEL_CONFIGS: tuple[ForecastModelConfig, ...] = (
    ForecastModelConfig(
        key="five_year_average",
        label="5-Year Avg",
        factory=lambda: FiveYearWeeklyAverageModel(
            lookback_years=DEFAULT_LOOKBACK_YEARS
        ),
    ),
    ForecastModelConfig(
        key="linear_regression",
        label="Linear Reg",
        factory=lambda: WeeklyChangeLinearRegressionModel(
            lookback_years=DEFAULT_LOOKBACK_YEARS
        ),
    ),
    ForecastModelConfig(
        key="fourier",
        label="Fourier",
        factory=lambda: WeeklyChangeFourierRegressionModel(
            lookback_years=DEFAULT_LOOKBACK_YEARS,
            n_harmonics=DEFAULT_FOURIER_HARMONICS,
        ),
    ),
    ForecastModelConfig(
        key="sarima",
        label="SARIMA",
        factory=lambda: WeeklyChangeSARIMAModel(
            lookback_years=DEFAULT_LOOKBACK_YEARS
        ),
    ),
)

SKLEARN_MODEL_CONFIGS: tuple[SklearnModelConfig, ...] = (
    SklearnModelConfig(
        key="linear_regression",
        label="Linear Regression",
        factory=LinearRegression,
    ),
    SklearnModelConfig(
        key="ridge",
        label="Ridge",
        factory=lambda: make_pipeline(
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
    ),
    SklearnModelConfig(
        key="elastic_net",
        label="ElasticNet",
        factory=lambda: make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42),
        ),
    ),
    SklearnModelConfig(
        key="random_forest",
        label="Random Forest",
        factory=lambda: RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    ),
    SklearnModelConfig(
        key="hist_gradient_boosting",
        label="HistGradientBoosting",
        factory=lambda: HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=0.1,
            random_state=42,
        ),
    ),
    SklearnModelConfig(
        key="hist_gradient_boosting_p10",
        label="HistGradientBoosting P10",
        factory=lambda: HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.10,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=0.1,
            random_state=42,
        ),
    ),
    SklearnModelConfig(
        key="hist_gradient_boosting_p90",
        label="HistGradientBoosting P90",
        factory=lambda: HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.90,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=0.1,
            random_state=42,
        ),
    ),
    SklearnModelConfig(
        key="linear_hgb_ensemble",
        label="Linear + HistGradientBoosting",
        factory=lambda: VotingRegressor(
            estimators=[
                ("linear", LinearRegression()),
                (
                    "hgb",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=300,
                        l2_regularization=0.1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    ),
)

ONE_STEP_ONLY_MODEL_CONFIGS: tuple[SklearnModelConfig, ...] = (
    SklearnModelConfig(
        key="arimax",
        label="ARIMAX (one-step)",
        factory=ARIMAXRegressor,
    ),
)


def legacy_forecast_model_configs() -> tuple[ForecastModelConfig, ...]:
    """Return default model configs used by the baseline forecast notebook."""
    return LEGACY_FORECAST_MODEL_CONFIGS


def _build_lightgbm():
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        reg_lambda=0.1,
        verbosity=-1,
        random_state=42,
        n_jobs=-1,
    )


def _build_xgboost():
    import xgboost as xgb
    return xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        reg_lambda=0.1,
        verbosity=0,
        random_state=42,
        n_jobs=-1,
    )


def sklearn_model_configs() -> tuple[SklearnModelConfig, ...]:
    """Return default sklearn-style model configs for feature-table backtests."""
    configs = list(SKLEARN_MODEL_CONFIGS)

    try:
        import lightgbm  # noqa: F401
        configs.append(
            SklearnModelConfig(
                key="lightgbm",
                label="LightGBM Regressor",
                factory=_build_lightgbm,
            )
        )
    except ImportError:
        pass

    try:
        import xgboost  # noqa: F401
        configs.append(
            SklearnModelConfig(
                key="xgboost",
                label="XGBoost Regressor",
                factory=_build_xgboost,
            )
        )
    except ImportError:
        pass

    return tuple(configs)


def one_step_model_configs() -> tuple[SklearnModelConfig, ...]:
    """Return the full one-step ladder, including non-recursive ARIMAX."""
    return (*sklearn_model_configs(), *ONE_STEP_ONLY_MODEL_CONFIGS)


def build_fourier_model(n_harmonics: int) -> WeeklyChangeFourierRegressionModel:
    """Build a Fourier regression model for harmonic-grid exploration."""
    return WeeklyChangeFourierRegressionModel(
        lookback_years=DEFAULT_LOOKBACK_YEARS,
        n_harmonics=n_harmonics,
    )
