"""CLI entrypoint for dataset discovery."""

from __future__ import annotations

import argparse
import logging

from ..pipelines.discover_datasets import run_discovery


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover regression datasets and build manifest.")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of datasets to fetch.")
    parser.add_argument("--source", type=str, default=None, help="Dataset source (currently only 'openml').")
    parser.add_argument("--force", action="store_true", help="Re-process datasets even if already present.")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args(argv)
    result = run_discovery(max_datasets=args.max, source=args.source, force=args.force)
    logging.info("Discovery complete: %s", result)


if __name__ == "__main__":
    main()
