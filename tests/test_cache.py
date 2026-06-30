import pandas as pd
import pytest

from gas_forecast.data.cache import compute_date_gaps


def test_compute_date_gaps_returns_full_range_when_cache_is_empty():
    assert compute_date_gaps(None, "2024-01-01", "2024-01-03") == [
        ("2024-01-01", "2024-01-03")
    ]


def test_compute_date_gaps_returns_prefix_and_suffix_only():
    cached_dates = pd.Series(pd.date_range("2024-01-03", "2024-01-05"))

    assert compute_date_gaps(cached_dates, "2024-01-01", "2024-01-07") == [
        ("2024-01-01", "2024-01-02"),
        ("2024-01-06", "2024-01-07"),
    ]


def test_compute_date_gaps_rejects_reversed_ranges():
    with pytest.raises(ValueError, match="after end_date"):
        compute_date_gaps(None, "2024-01-07", "2024-01-01")
