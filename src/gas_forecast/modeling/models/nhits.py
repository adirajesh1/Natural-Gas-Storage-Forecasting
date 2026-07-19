"""Optional pooled regional N-HiTS challenger."""

from __future__ import annotations

import pandas as pd


def _load_neuralforecast():
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss
        from neuralforecast.models import NHITS
    except ImportError as exc:
        raise ImportError(
            "PooledNHITSForecaster requires the optional neural dependencies. "
            "Install the project with `pip install -e '.[neural]'`."
        ) from exc
    return NeuralForecast, NHITS, MQLoss


class PooledNHITSForecaster:
    """Four-week probabilistic N-HiTS model fitted across all storage regions."""

    def __init__(
        self,
        *,
        horizon: int = 4,
        input_size: int = 104,
        future_exog_cols: tuple[str, ...] = (
            "hdd",
            "cdd",
            "hdd_p10",
            "hdd_p90",
            "cdd_p10",
            "cdd_p90",
        ),
        max_steps: int = 500,
        random_seed: int = 42,
    ) -> None:
        self.horizon = horizon
        self.input_size = input_size
        self.future_exog_cols = future_exog_cols
        self.max_steps = max_steps
        self.random_seed = random_seed

    def _validate(self, frame: pd.DataFrame, *, require_target: bool) -> pd.DataFrame:
        required = {"unique_id", "ds", *self.future_exog_cols}
        if require_target:
            required.add("y")
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"N-HiTS panel missing columns: {missing}")
        data = frame.copy()
        data["ds"] = pd.to_datetime(data["ds"], errors="coerce")
        if data["ds"].isna().any():
            raise ValueError("N-HiTS panel contains invalid dates.")
        return data

    def fit(self, panel: pd.DataFrame) -> "PooledNHITSForecaster":
        data = self._validate(panel, require_target=True)
        NeuralForecast, NHITS, MQLoss = _load_neuralforecast()
        model = NHITS(
            h=self.horizon,
            input_size=self.input_size,
            futr_exog_list=list(self.future_exog_cols),
            loss=MQLoss(level=[80]),
            max_steps=self.max_steps,
            random_seed=self.random_seed,
        )
        self._forecast = NeuralForecast(models=[model], freq="W-FRI")
        self._forecast.fit(df=data)
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "_forecast"):
            raise RuntimeError("Call fit() before predict().")
        data = self._validate(future, require_target=False)
        result = self._forecast.predict(futr_df=data).reset_index()
        lower = next((column for column in result if "lo-80" in column), None)
        median = next((column for column in result if "median" in column), None)
        upper = next((column for column in result if "hi-80" in column), None)
        if lower is None or median is None or upper is None:
            raise RuntimeError("N-HiTS output did not contain the requested 10/50/90 quantiles.")
        return result.rename(columns={lower: "p10", median: "p50", upper: "p90"})[
            ["unique_id", "ds", "p10", "p50", "p90"]
        ]

