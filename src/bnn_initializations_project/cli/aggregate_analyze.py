"""CLI entrypoint for aggregation and analysis."""

from __future__ import annotations

import argparse
import logging

from ..pipelines.run_analysis import run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate results and compute summary analysis.")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    _ = build_parser().parse_args(argv)
    run_analysis()
    logging.info("Analysis complete.")


if __name__ == "__main__":
    main()
