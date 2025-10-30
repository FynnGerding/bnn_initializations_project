"""Command line entrypoints for the BNN initializations project."""

from __future__ import annotations

from .aggregate_analyze import build_parser as build_aggregate_analyze_parser, main as aggregate_analyze_main
from .discover import build_parser as build_discover_parser, main as discover_main
from .extract_meta import build_parser as build_extract_meta_parser, main as extract_meta_main
from .train_eval import build_parser as build_train_eval_parser, main as train_eval_main

__all__ = [
    "aggregate_analyze_main",
    "build_aggregate_analyze_parser",
    "build_discover_parser",
    "build_extract_meta_parser",
    "build_train_eval_parser",
    "discover_main",
    "extract_meta_main",
    "train_eval_main",
]
