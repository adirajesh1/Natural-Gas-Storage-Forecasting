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
        help="EIA duoarea code (R48, R01, R02, R03, R04, R05).",
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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
