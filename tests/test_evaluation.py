import pandas as pd

from gas_forecast.evaluation import evaluate_forecast
from gas_forecast.models.base import WeeklyChangeForecastModel


class RecordingModel(WeeklyChangeForecastModel):
    def __init__(self):
        self.fit_years = None

    @property
    def name(self) -> str:
        return "Recording"

    def fit(self, storage: pd.DataFrame) -> "RecordingModel":
        self.fit_years = sorted(storage["year"].unique().tolist())
        return self

    def predict(self, evaluation: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"predicted_weekly_change": [0.0] * len(evaluation)})


def test_evaluate_forecast_does_not_fit_on_years_after_requested_year():
    storage = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-07", "2023-01-06", "2024-01-05"]),
            "year": [2022, 2023, 2024],
            "week_of_year": [1, 1, 1],
            "weekly_change_bcf": [1.0, 2.0, 3.0],
        }
    )
    model = RecordingModel()

    evaluate_forecast(storage, model, year=2023)

    assert model.fit_years == [2022, 2023]
