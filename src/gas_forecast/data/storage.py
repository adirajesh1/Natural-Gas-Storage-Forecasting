"""Compatibility facade for storage data helpers.

New code should prefer the focused modules:

- `storage_api` for EIA API access and raw incremental cache behavior;
- `storage_transforms` for cleaning, region selection, and model formatting;
- `storage_validation` for dataframe validators.
"""

from gas_forecast.data.storage_api import (
    EIA_WEEKLY_STORAGE_URL,
    STORAGE_CACHE_FILENAME,
    STORAGE_MERGE_KEY_COLS,
    fetch_weekly_storage_incremental,
    fetch_weekly_storage_paginated,
    fetch_weekly_storage_raw,
)
from gas_forecast.data.storage_transforms import (
    calculate_weekly_storage_change,
    clean_weekly_storage,
    prepare_storage_model_data,
    select_region,
)
from gas_forecast.data.storage_validation import (
    validate_storage_region,
    validate_weekly_storage,
)

__all__ = [
    "EIA_WEEKLY_STORAGE_URL",
    "STORAGE_CACHE_FILENAME",
    "STORAGE_MERGE_KEY_COLS",
    "calculate_weekly_storage_change",
    "clean_weekly_storage",
    "fetch_weekly_storage_incremental",
    "fetch_weekly_storage_paginated",
    "fetch_weekly_storage_raw",
    "prepare_storage_model_data",
    "select_region",
    "validate_storage_region",
    "validate_weekly_storage",
]
