"""High-level orchestration pipelines."""

from __future__ import annotations

def run_discovery(*args, **kwargs):
    from .discover_datasets import run_discovery as _run_discovery

    return _run_discovery(*args, **kwargs)


def run_meta_feature_extraction(*args, **kwargs):
    from .run_meta_features import run_meta_feature_extraction as _run_mfe

    return _run_mfe(*args, **kwargs)


def run_training(*args, **kwargs):
    from .run_training import run_training as _run_training

    return _run_training(*args, **kwargs)


def run_analysis(*args, **kwargs):
    from .run_analysis import run_analysis as _run_analysis

    return _run_analysis(*args, **kwargs)

from .run_training import _process_dataset

__all__ = [
    "run_analysis",
    "run_discovery",
    "run_meta_feature_extraction",
    "run_training",
    "_process_dataset",
]
