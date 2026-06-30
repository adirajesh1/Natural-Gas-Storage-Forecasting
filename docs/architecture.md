# Project Architecture

## Purpose of this document

This document describes how the codebase is organized, what each major module is responsible for, and how data moves through the system.

It should be updated when:

- a new major module is created;
- responsibility moves between modules;
- the main execution flow changes;
- a new external system or data source is introduced.

## High-level architecture

The intended architecture separates the project into four stages:

```text
1. Data collection
        |
        v
2. Data cleaning and validation
        |
        v
3. Feature engineering
        |
        v
4. Modeling and evaluation
```



## Current execution flow

### Storage (01a)

```text
REGION config
        |
        v
fetch_weekly_storage_incremental (data/cache/storage/)
        |
        v
clean → select_region → calculate change → export processed parquet
```

### Weather (01b)

```text
REGION config + storage date range
        |
        v
load census → select_weather_locations
        |
        v
fetch_all_state_temperatures incremental (data/cache/weather/by_state/)
        |
        v
HDD/CDD → aggregate → export processed parquet
```

### Raw cache layout

```text
data/cache/
  storage/weekly_storage_raw.parquet
  weather/by_state/{State}.parquet
```

The known weather-data flow is:

```text
Requested date range and region
        |
        v
Split range into calendar-year periods
        |
        v
Build Open-Meteo API requests
        |
        v
Send HTTP requests
        |
        v
Parse API responses
        |
        v
Store or cache downloaded weather data
        |
        v
Combine downloaded periods
        |
        v
Return weather data for later processing
```



## Known modules



### Weather downloader

Path: src/data/weather.py

Purpose:

> Download historical weather data from Open-Meteo in smaller, manageable request periods.

Known dependencies:

- `pandas`: date handling and tabular data;
- `requests`: HTTP requests;
- `pathlib.Path`: filesystem paths;
- `hashlib`: likely generation of stable cache identifiers;
- `time`: likely delays between retries or requests.

Known external system:

- Open-Meteo Archive API

API endpoint:

```text
https://archive-api.open-meteo.com/v1/archive
```

Known responsibilities:

- accept a requested historical date range;
- divide that range into yearly periods;
- make API requests;
- possibly cache API results;
- possibly retry failed requests;
- return or save weather observations.

Responsibilities that should remain outside this module:

- calculating HDD and CDD;
- aggregating weather into gas weeks;
- combining weather with EIA storage data;
- fitting forecasting models;
- evaluating forecast accuracy.



#### `_date_periods`

Purpose:

> Split an inclusive date range into calendar-year request periods.

Inputs:

- `start_date`: beginning of the full requested period;
- `end_date`: end of the full requested period;
- `freq`: currently supports yearly periods through `"YS"`.

Output:

```python
list[tuple[str, str]]
```

Each tuple contains an inclusive start and end date.

Example:

```python
_date_periods("2024-06-01", "2025-02-01")
```

Expected conceptual output:

```python
[
    ("2024-06-01", "2024-12-31"),
    ("2025-01-01", "2025-02-01"),
]
```

Important behavior:

- partial first and final years are preserved;
- intermediate years cover January 1 through December 31;
- unsupported frequencies raise a `ValueError`.

Reason for existence:

Long weather histories are easier to retry, cache, inspect, and resume when divided into smaller requests.

## Module inventory

This table will be populated after reviewing the folder tree.


| Module                | Purpose                     | Inputs                      | Outputs                     | Called by |
| --------------------- | --------------------------- | --------------------------- | --------------------------- | --------- |
| `TODO weather module` | Download historical weather | Dates, locations, variables | Daily weather data or files | `TODO`    |
| `TODO EIA module`     | Download storage data       | EIA request parameters      | Weekly storage records      | `TODO`    |
| `TODO feature module` | Create model features       | Clean source datasets       | Weekly feature table        | `TODO`    |
| `TODO model module`   | Train and evaluate model    | Weekly feature table        | Predictions and metrics     | `TODO`    |




## Entry points

An entry point is a file or function that starts a meaningful workflow.

Current entry points:


| Entry point | Workflow started            | Status               |
| ----------- | --------------------------- | -------------------- |
| `TODO`      | Download historical weather | Needs identification |
| `TODO`      | Download EIA storage data   | Needs identification |
| `TODO`      | Build features              | May not exist yet    |
| `TODO`      | Train model                 | May not exist yet    |




## Shared configuration

The following items need to be located and documented:

- [ ] requested historical date range;
- [ ] weather locations;
- [ ] weather variables;
- [ ] raw data directory;
- [ ] processed data directory;
- [ ] API keys;
- [ ] request timeouts;
- [ ] retry limits;
- [ ] cache settings;
- [ ] logging settings;
- [ ] model parameters.



## Open architecture questions

- Where does the user currently start the weather download?
- Is there one pipeline script or several separate scripts?
- Does the weather downloader save raw API responses or processed tables?
- How is the cache key generated?
- Can an interrupted download resume without repeating completed requests?
- Where are weather locations defined?
- Where will HDD and CDD calculations live?
- Is the project installed as a Python package?
- Are notebooks calling reusable functions from the package?

