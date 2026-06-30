# Decision Log

## 2026-06-30 - Split broad storage and weather modules

### Decision

Split the broad `gas_forecast.data.weather` implementation into focused API, cache, location, feature, and validation modules. Split the broad `gas_forecast.data.storage` implementation into API/cache, transform, and validation modules.

The original `weather.py` and `storage.py` files remain as compatibility facades so existing notebook and package imports continue to work.

### Reason

The data modules had accumulated multiple responsibilities. Smaller modules make it clearer where API behavior, cache behavior, transformations, and validation belong without forcing downstream code to migrate all at once.

### Alternatives considered

- Keep the broad modules until they became harder to navigate.
- Move all existing imports immediately to the new modules and remove the old paths.

### Tradeoff

There are more files to understand, but each file has a tighter responsibility. The compatibility facades add a small layer of indirection while protecting existing notebooks.

### Revisit when

If notebooks and downstream code fully migrate to the focused modules, decide whether the facades should remain as stable public API or become deprecated.

## 2026-06-30 - Add architecture guardrails and focused tests

### Decision

Add a pytest suite for the highest-risk date, region, feature, and evaluation-year behavior. Update the architecture document to match the current `src/gas_forecast` package layout.

### Reason

The project has moved from notebook-only exploration toward reusable package and CLI workflows. Focused tests help preserve assumptions around storage weeks, grouped regional calculations, cache gaps, and time-based model evaluation.

### Alternatives considered

- Wait to add tests until after a larger refactor.
- Only update docs without executable checks.

### Tradeoff

There is a small amount of extra maintenance, but the covered behavior is central enough that regressions would be costly and hard to spot visually.

### Revisit when

When weather and storage modules are split into smaller files, move or expand the tests to follow the new module boundaries.

## 2026-06-29 - Refactor Storage and Weather Functions to be generalizable

### Decision

Storage/weather functions were moved to a central package location and made generalizable. Formatting was changed to make the workflows mirror one another: select a region, then let orchestration handle the workflow.

### Reason

Generalizable functions are better when models need to analyze different regions.

### Alternatives considered

- Keep region-specific, notebook-only functions.

### Tradeoff

The codebase becomes larger and more difficult to wrangle with, but the reusable workflow is easier to trust and extend.

### Revisit when

If the generalized functions become too abstract for the project scale.

## 2026-06-29 - Split weather downloads by state and calendar year

### Decision

Historical weather requests are divided into state and calendar-year periods for yearly weather pulls.

### Reason

Smaller requests are easier to cache, retry, inspect, and resume after a failure.

### Alternatives considered

- Request the entire history at once.
- Divide the history into monthly requests.

### Tradeoff

Yearly requests create more API calls and local files than one large request, but significantly fewer than monthly requests. With free API limits, this can still fail unless request pacing is increased.

### Revisit when

If yearly requests remain too slow, exceed API limits, or regularly fail.

## 2026-06-29 - Incremental raw data cache for storage and weather

### Decision

Raw API responses are cached under `datasets/cache/` with incremental merge semantics:

- Storage: single `storage/weekly_storage_raw.parquet`; tail re-fetch of the last 8 weeks on each update to capture EIA revisions.
- Weather: per-state `weather/by_state/{state}.parquet`; gap-based fetch for missing prefix/suffix dates only.
- Shared helpers live in `src/gas_forecast/data/cache.py` for load, atomic write, merge, and gap detection.

Processed exports in `datasets/processed/` remain versioned and are rebuilt from cache each notebook or pipeline run.

### Reason

Weekly refresh should not re-download full histories. Storage revisions require re-fetching a short tail window, not append-only updates.

### Alternatives considered

- Append-only storage updates, which would miss EIA revisions.
- Hash-keyed weather chunks keyed by full date range, which re-downloads partial years when `END_DATE` moves.

### Tradeoff

More code and more on-disk cache files; the first weather run is still one request per state per gap chunk.

### Revisit when

If Open-Meteo historical revisions become important, add `force_refresh` for weather history.
