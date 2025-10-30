"""Data input/output utilities for dataset discovery and persistence."""

from __future__ import annotations

from .manifest import MANIFEST_COLUMNS, manifest_path, read_manifest, upsert_manifest_rows
from .splits import SplitSummary, make_splits
from .storage import (
    artifacts_path,
    atomic_write_npz,
    atomic_write_parquet,
    dataset_npz_path,
    ensure_artifact_dirs,
    load_npz_arrays,
)

__all__ = [
    "MANIFEST_COLUMNS",
    "SplitSummary",
    "artifacts_path",
    "atomic_write_npz",
    "atomic_write_parquet",
    "dataset_npz_path",
    "ensure_artifact_dirs",
    "load_npz_arrays",
    "make_splits",
    "manifest_path",
    "read_manifest",
    "upsert_manifest_rows",
]
