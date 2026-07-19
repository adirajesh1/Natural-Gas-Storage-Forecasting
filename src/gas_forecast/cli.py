from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gas_forecast.data.regions import supported_storage_regions
from gas_forecast.data.paths import (
    DEFAULT_LEGACY_WEATHER_CACHE_DIR,
    DEFAULT_PROCESSED_DIR,
)
from gas_forecast.pipelines.data import (
    ALL_STAGES,
    PipelineOutputs,
    PipelineStage,
    run_all_regions,
    run_data_pipeline,
)


def _parse_stages(value: str | None) -> tuple[PipelineStage, ...]:
    if not value:
        return ALL_STAGES

    stages = tuple(stage.strip() for stage in value.split(",") if stage.strip())
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        valid = ", ".join(ALL_STAGES)
        raise argparse.ArgumentTypeError(
            f"Unknown stage(s): {', '.join(sorted(unknown))}. Valid: {valid}"
        )
    return stages  # type: ignore[return-value]


def _print_outputs(outputs: PipelineOutputs) -> None:
    print(f"Region: {outputs.region}")
    for name, path in sorted(outputs.paths.items()):
        print(f"  {name}: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gas-data",
        description="Run Gas Market Platform data pipelines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    refresh = subparsers.add_parser(
        "refresh",
        help="Refresh cached raw data and rebuild processed datasets.",
    )
    refresh.add_argument(
        "--region",
        help="EIA duoarea code (R48, R31, R32, R33, R34, R35).",
    )
    refresh.add_argument(
        "--all-regions",
        action="store_true",
        help="Run for every supported storage region.",
    )
    refresh.add_argument(
        "--only",
        type=_parse_stages,
        default=ALL_STAGES,
        metavar="STAGES",
        help="Comma-separated stages to run: storage,weather,features.",
    )
    refresh.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("datasets/cache"),
        help="Raw incremental cache directory.",
    )
    refresh.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Processed parquet output directory.",
    )
    refresh.add_argument(
        "--legacy-weather-cache-dir",
        type=Path,
        default=DEFAULT_LEGACY_WEATHER_CACHE_DIR,
        help="Legacy hash-keyed weather chunk cache to migrate once.",
    )
    refresh.add_argument(
        "--min-start-date",
        default="2010-01-01",
        help="Earliest storage period to retain after cleaning.",
    )
    refresh.add_argument(
        "--revision-weeks",
        type=int,
        default=8,
        help="Storage tail window to re-fetch for EIA revisions.",
    )
    refresh.add_argument(
        "--pause-seconds",
        type=float,
        default=3.0,
        help="Delay between Open-Meteo API requests when fetching gaps.",
    )
    refresh.add_argument(
        "--skip-legacy-migration",
        action="store_true",
        help="Do not migrate legacy weather chunk cache files.",
    )

    balance = subparsers.add_parser(
        "balance",
        help="Run weekly supply-demand balance disaggregation pipeline.",
    )
    balance.add_argument(
        "--region",
        required=True,
        help="EIA duoarea code (R48, R31, R32, R33, R34, R35).",
    )
    balance.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Processed parquet output directory.",
    )
    balance.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh raw monthly EIA data cache.",
    )

    weather_scenario = subparsers.add_parser(
        "weather-scenario",
        help="Select an archived weekly weather forecast as of one origin.",
    )
    weather_scenario.add_argument(
        "--region",
        required=True,
        help="EIA duoarea code for the scenario table.",
    )
    weather_scenario.add_argument(
        "--scenarios-path",
        required=True,
        type=Path,
        help="Parquet archive with date, duoarea, issued_at, and weekly weather values.",
    )
    weather_scenario.add_argument(
        "--as-of",
        required=True,
        help="Forecast-origin timestamp used to select the latest available version.",
    )
    weather_scenario.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Processed parquet output directory.",
    )

    balance_asof = subparsers.add_parser(
        "balance-asof",
        help="Build balance lag features from an explicit historical vintage archive.",
    )
    balance_asof.add_argument(
        "--region",
        required=True,
        help="EIA duoarea code (R48, R31, R32, R33, R34, R35).",
    )
    balance_asof.add_argument(
        "--vintages-path",
        required=True,
        type=Path,
        help="Parquet archive with date, duoarea, available_at, and balance values.",
    )
    balance_asof.add_argument(
        "--origins-path",
        type=Path,
        help="Optional feature/origin parquet; defaults to the region's model features.",
    )
    balance_asof.add_argument(
        "--as-of-column",
        default="date",
        help="Origin timestamp column; date is interpreted as midnight UTC.",
    )
    balance_asof.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Processed parquet output directory.",
    )

    weather_forecast = subparsers.add_parser(
        "weather-forecast",
        help="Archive live GEFS or historical fixed-lead GFS weather forecasts.",
    )
    weather_forecast.add_argument("--region", required=True)
    weather_forecast.add_argument(
        "--issued-at",
        help="Live forecast issue timestamp. Required unless start/end dates are supplied.",
    )
    weather_forecast.add_argument("--start-date")
    weather_forecast.add_argument("--end-date")
    weather_forecast.add_argument("--forecast-days", type=int, default=16)
    weather_forecast.add_argument("--max-lead-days", type=int, default=7)
    weather_forecast.add_argument("--weight-history-path", type=Path)
    weather_forecast.add_argument("--weight-column", default="gas_load")
    weather_forecast.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )

    regional_backtest = subparsers.add_parser(
        "regional-backtest",
        help="Backtest all storage regions and compare direct, bottom-up, and MinT paths.",
    )
    regional_backtest.add_argument("--model", default="hist_gradient_boosting")
    regional_backtest.add_argument(
        "--weather-input",
        choices=("seasonal", "scenario", "observed"),
        default="seasonal",
    )
    regional_backtest.add_argument("--weather-scenarios-path", type=Path)
    regional_backtest.add_argument("--initial-train-start", default="2010-01-01")
    regional_backtest.add_argument("--initial-train-end", default="2020-12-31")
    regional_backtest.add_argument("--horizon-weeks", type=int, default=4)
    regional_backtest.add_argument("--step-weeks", type=int, default=4)
    regional_backtest.add_argument(
        "--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "regional-backtest":
        from gas_forecast.pipelines.modeling import run_regional_model_backtest

        outputs = run_regional_model_backtest(
            model_key=args.model,
            forecast_input_mode=args.weather_input,
            weather_scenarios_path=args.weather_scenarios_path,
            initial_train_start=args.initial_train_start,
            initial_train_end=args.initial_train_end,
            horizon_weeks=args.horizon_weeks,
            step_weeks=args.step_weeks,
            processed_dir=args.processed_dir,
        )
        _print_outputs(outputs)
        return 0

    if args.command == "weather-forecast":
        from gas_forecast.pipelines.asof import (
            run_historical_weather_forecast_pipeline,
            run_live_weather_forecast_pipeline,
        )

        if bool(args.start_date) != bool(args.end_date):
            parser.error("Provide both --start-date and --end-date.")
        if args.start_date:
            outputs = run_historical_weather_forecast_pipeline(
                args.region,
                start_date=args.start_date,
                end_date=args.end_date,
                max_lead_days=args.max_lead_days,
                weight_history_path=args.weight_history_path,
                weight_col=args.weight_column,
                processed_dir=args.processed_dir,
            )
        else:
            if not args.issued_at:
                parser.error("Live weather forecasts require --issued-at.")
            outputs = run_live_weather_forecast_pipeline(
                args.region,
                issued_at=args.issued_at,
                forecast_days=args.forecast_days,
                weight_history_path=args.weight_history_path,
                weight_col=args.weight_column,
                processed_dir=args.processed_dir,
            )
        _print_outputs(outputs)
        return 0

    if args.command == "weather-scenario":
        from gas_forecast.pipelines.asof import run_weather_scenario_pipeline

        path = run_weather_scenario_pipeline(
            region=args.region,
            scenarios_path=args.scenarios_path,
            as_of=args.as_of,
            processed_dir=args.processed_dir,
        )
        print(path)
        return 0

    if args.command == "balance-asof":
        from gas_forecast.pipelines.asof import run_asof_balance_pipeline

        path = run_asof_balance_pipeline(
            region=args.region,
            vintages_path=args.vintages_path,
            origins_path=args.origins_path,
            as_of_col=args.as_of_column,
            processed_dir=args.processed_dir,
        )
        print(path)
        return 0

    if args.command == "balance":
        from gas_forecast.pipelines.balance import run_balance_pipeline
        run_balance_pipeline(
            region=args.region,
            processed_dir=args.processed_dir,
            force_refresh=args.force_refresh,
        )
        return 0

    if args.command != "refresh":
        parser.error(f"Unsupported command: {args.command}")

    if args.all_regions and args.region:
        parser.error("Use either --region or --all-regions, not both.")
    if not args.all_regions and not args.region:
        parser.error("Specify --region REGION or --all-regions.")

    common_kwargs = {
        "stages": args.only,
        "cache_dir": args.cache_dir,
        "processed_dir": args.processed_dir,
        "legacy_weather_cache_dir": args.legacy_weather_cache_dir,
        "min_start_date": args.min_start_date,
        "revision_weeks": args.revision_weeks,
        "pause_seconds": args.pause_seconds,
        "migrate_legacy_cache": not args.skip_legacy_migration,
    }

    if args.all_regions:
        results = run_all_regions(**common_kwargs)
        for region in supported_storage_regions():
            _print_outputs(results[region])
            print()
    else:
        _print_outputs(run_data_pipeline(args.region, **common_kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
