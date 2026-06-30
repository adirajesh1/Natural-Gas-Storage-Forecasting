# Decision Log

## 2026-06-29 — Refactor Storage and Weather Functions to be generalizable

### Decision

Storage/Weather functions are moved to central location and have been made generalizable. Formatting has been changed to mimic one another's workflows (select region, then orchestrator will handle).

### Reason

Generalizable functions are better when models can be used to analyze different regions.

### Alternatives considered

- keeping in notebook/non-generalizable

### Tradeoff

Codebase becomes larger and mroe difficult to wrangle with

### Revisit when

Unlikely to revisit.

## 2026-06-29 — Split weather downloads by state and calendar year

### Decision

Historical weather requests are divided into state+calendar-year periods for yearly weather pull.

### Reason

Smaller requests are easier to cache, retry, inspect, and resume after a failure.

### Alternatives considered

- Request the entire history at once.
- Divide the history into monthly requests.



### Tradeoff

Yearly requests create more API calls and local files than one large request, but significantly fewer than monthly requests. With free API limits, this will fail (unless time increased etc.)

### Revisit when

If yearly requests remain too slow, exceed API limits, or regularly fail.

## 2026-06-29 — Incremental raw data cache for storage and weather

### Decision

Raw API responses are cached under `data/cache/` with incremental merge semantics:

- **Storage:** single `storage/weekly_storage_raw.parquet`; tail re-fetch of the last 8 weeks on each update to capture EIA revisions.
- **Weather:** per-state `weather/by_state/{state}.parquet`; gap-based fetch for missing prefix/suffix dates only.
- Shared helpers live in `src/gas_forecast/data/cache.py` (load, atomic write, merge, gap detection).

Processed exports in `data/processed/` remain versioned and are rebuilt from cache each notebook run.

### Reason

Weekly refresh should not re-download full histories. Storage revisions require re-fetching a short tail window, not append-only updates.

### Alternatives considered

- Append-only storage updates (misses EIA revisions).
- Hash-keyed weather chunks keyed by full date range (re-downloads partial years when `END_DATE` moves).

### Tradeoff

More code and on-disk cache files; weather first run is still one request per state per gap chunk.

### Revisit when

If Open-Meteo historical revisions become important (add `force_refresh` for weather history).