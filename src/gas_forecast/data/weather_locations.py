from __future__ import annotations

import pandas as pd

from gas_forecast.data.regions import lower48_excluded_states, region_states
from gas_forecast.data.weather_validation import validate_weather_locations

CENSUS_POP_URL = (
    "https://www2.census.gov/geo/docs/reference/"
    "cenpop2020/CenPop2020_Mean_ST.txt"
)


def load_census_state_locations(
    url: str = CENSUS_POP_URL,
    *,
    exclude: frozenset[str] | set[str] | None = None,
) -> pd.DataFrame:
    """Load census state centroids and compute national population weights."""
    if exclude is None:
        exclude = lower48_excluded_states()

    locations = pd.read_csv(url, dtype={"STATEFP": str})
    locations = locations.loc[~locations["STNAME"].isin(exclude)].copy()
    locations["WEIGHT"] = (
        locations["POPULATION"] / locations["POPULATION"].sum()
    )
    return locations.reset_index(drop=True)


def select_weather_locations(
    locations: pd.DataFrame,
    duoarea: str,
) -> pd.DataFrame:
    """Filter census locations to an EIA storage region and renormalize weights."""
    states = region_states(duoarea)
    available_states = set(locations["STNAME"].astype(str))
    missing_states = sorted(states - available_states)
    if missing_states:
        raise ValueError(
            f"Region {duoarea} requires states missing from locations: "
            f"{missing_states}"
        )

    selected = locations.loc[locations["STNAME"].isin(states)].copy()
    selected["WEIGHT"] = selected["POPULATION"] / selected["POPULATION"].sum()
    validate_weather_locations(selected, expected_states=states)
    return selected.reset_index(drop=True)
