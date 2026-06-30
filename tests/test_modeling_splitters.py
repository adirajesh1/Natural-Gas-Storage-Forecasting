import pandas as pd

from gas_forecast.modeling.splitters import (
    ExpandingWindowSplitter,
    HoldoutSplitter,
    RollingWindowSplitter,
)


def _weekly_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"date": pd.date_range("2024-01-05", periods=8, freq="W-FRI")},
        index=[10, 20, 30, 40, 50, 60, 70, 80],
    )


def test_holdout_split_returns_expected_date_ranges_and_iloc_positions():
    df = _weekly_frame()
    splitter = HoldoutSplitter(
        "date",
        train_start="2024-01-05",
        train_end="2024-01-26",
        val_start="2024-02-02",
        val_end="2024-02-09",
    )

    train_idx, val_idx = next(splitter.split(df))

    assert train_idx == [0, 1, 2, 3]
    assert val_idx == [4, 5]
    assert df.iloc[train_idx]["date"].max() == pd.Timestamp("2024-01-26")
    assert df.iloc[val_idx]["date"].tolist() == [
        pd.Timestamp("2024-02-02"),
        pd.Timestamp("2024-02-09"),
    ]


def test_expanding_window_split_grows_training_and_steps_validation():
    df = _weekly_frame()
    splitter = ExpandingWindowSplitter(
        "date",
        initial_train_start="2024-01-05",
        initial_train_end="2024-01-26",
        val_weeks=1,
        step_weeks=1,
    )

    splits = list(splitter.split(df))

    assert splits[0] == ([0, 1, 2, 3], [4])
    assert splits[1] == ([0, 1, 2, 3, 4], [5])
    assert splits[2] == ([0, 1, 2, 3, 4, 5], [6])


def test_rolling_window_split_advances_train_and_validation_windows():
    df = _weekly_frame()
    splitter = RollingWindowSplitter(
        "date",
        initial_train_start="2024-01-05",
        initial_train_end="2024-01-26",
        val_weeks=1,
        step_weeks=1,
    )

    splits = list(splitter.split(df))

    assert splits[0] == ([0, 1, 2, 3], [4])
    assert splits[1] == ([1, 2, 3, 4], [5])
    assert splits[2] == ([2, 3, 4, 5], [6])
