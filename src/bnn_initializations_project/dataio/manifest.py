"""Manifest persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .storage import artifacts_path, atomic_write_parquet

__all__ = ["MANIFEST_COLUMNS", "manifest_path", "read_manifest", "upsert_manifest_rows"]

MANIFEST_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "source",
    "name",
    "n_rows",
    "n_features",
    "task_type",
    "target_name",
    "status",
    "error_msg",
)


def manifest_path() -> Path:
    """Return the filesystem path for the manifest table."""
    return artifacts_path("manifest.parquet")


def _empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def read_manifest(status: str | None = "ok") -> pd.DataFrame:
    """Read the manifest, optionally filtering by ``status``."""
    path = manifest_path()
    if not path.exists():
        df = _empty_manifest()
    else:
        df = pd.read_parquet(path)
    if status is None or "status" not in df.columns:
        return df
    return df[df["status"] == status].reset_index(drop=True)


def upsert_manifest_rows(rows: Iterable[dict]) -> None:
    """Merge rows into the manifest keyed by ``dataset_id``."""
    rows = list(rows)
    if not rows:
        return

    df_new = pd.DataFrame(rows)
    if "dataset_id" not in df_new.columns:
        raise ValueError("Manifest rows must include 'dataset_id'.")

    path = manifest_path()
    if path.exists():
        df_old = pd.read_parquet(path)
        combined = pd.concat([df_old, df_new], ignore_index=True, sort=False)
    else:
        combined = df_new

    combined = combined.sort_values(by=["dataset_id"], kind="stable")
    combined = combined.drop_duplicates(subset=["dataset_id"], keep="last")

    for col in MANIFEST_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA

    combined = combined[list(dict.fromkeys([*MANIFEST_COLUMNS, *combined.columns]))]
    atomic_write_parquet(combined, path)
