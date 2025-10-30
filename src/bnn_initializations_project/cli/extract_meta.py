"""CLI entrypoint for meta-feature extraction."""

from __future__ import annotations

import argparse
import logging

from ..pipelines.run_meta_features import run_meta_feature_extraction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract meta-features for discovered datasets.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs.")
    parser.add_argument("--force", action="store_true", help="Recompute even when already present.")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args(argv)
    result = run_meta_feature_extraction(jobs=args.jobs, force=args.force)
    logging.info("Meta-feature extraction summary: %s", result)


if __name__ == "__main__": 
    main()