# Gas Market Platform - Documentation Directory

Welcome to the documentation folder for the Gas Market Platform. This directory contains detailed architectural guides, mathematical formulations, modeling decisions, and domain-specific documentation.

## Documentation Index

Below is a summary of the available documentation files in this directory:

### 1. [Architecture Overview](architecture.md)
*   **Filename**: `architecture.md`
*   **Purpose**: Outlines the overall codebase layout, system flow (from raw caching to feature table generation and model execution), entry points (both CLI and Python APIs), data artifacts, and package module responsibilities.
*   **When to read**: When onboarding to the project, adding/renaming modules, or understanding how the pipeline components connect.

### 2. [Mathematical Methodology](METHODOLOGY.md)
*   **Filename**: `METHODOLOGY.md`
*   **Purpose**: Details the mathematical formulations, OLS regressions (Residential & Commercial, Power Burn), downscaling procedures for dry production and lease/plant/pipeline fuel, basis spread projections, trend ratios, market tightness indicators, and the recursive state simulation algorithm.
*   **When to read**: When investigating the physical/mathematical logic behind monthly-to-weekly data disaggregation, local balance sheets, or state transitions in multi-step forecast simulation.

### 3. [Modeling Assumptions & Timing Contracts](modeling_assumptions.md)
*   **Filename**: `modeling_assumptions.md`
*   **Purpose**: Defines the strict contracts for data availability, recursive forecasting modes (seasonal, scenario, observed), feature timing, balance sheet inputs, prediction interval calibration (conformal prediction), and interpretation limitations to prevent data leakage and look-ahead bias.
*   **When to read**: Before modifying model training loops, expanding features, or designing backtests.

### 4. [Power Fundamentals MVP](power_fundamentals.md)
*   **Filename**: `power_fundamentals.md`
*   **Purpose**: Details the ERCOT system-wide hourly power-fundamentals MVP, outlining the data contract, 168-hour physical stack identities, heat-rate based implied gas-burn scenario calculations, and CLI commands.
*   **When to read**: When working with the `power_forecast` package, adding weather/price scenarios, or modifying the Streamlit dashboard fundamentals view.

### 5. [Oil Fundamentals MVP](oil_fundamentals.md)
*   **Filename**: `oil_fundamentals.md`
*   **Purpose**: Documents the U.S. weekly crude balance, component forecasts, inventory-change backtest, CLI commands, and timing limitations.
*   **When to read**: When working with `oil_forecast`, interpreting the oil dashboard, or changing the EIA weekly-series contract.

### 6. [Decision Log](decisions.md)
*   **Filename**: `decisions.md`
*   **Purpose**: A chronological record of key design and architecture decisions, their justifications, alternatives considered, tradeoffs, and revisit conditions.
*   **When to read**: To understand *why* certain structural patterns (e.g. splitting weather downloads, adding explicit vintages, implementing conformal intervals) were adopted.

---

## Directory Conventions

*   **Documentation Integrity**: Maintain consistent definitions and update these files whenever structural changes occur.
*   **Referencing Files**: When linking between markdown files or package modules, use relative paths to keep links clickable and functional.
