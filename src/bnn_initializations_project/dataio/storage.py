"""Stable artifact storage helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "artifacts_path",
    "atomic_write_npz",
    "atomic_write_parquet",
    "dataset_npz_path",
    "ensure_artifact_dirs",
    "load_npz_arrays",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifacts_path(*parts: str | os.PathLike[str]) -> Path:
    """Return a path inside the repository-level ``artifacts`` directory."""
    root = _project_root() / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    if not parts:
        return root
    sub_path = Path(*parts)
    full_path = root / sub_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    return full_path


def ensure_artifact_dirs(paths: Iterable[str | os.PathLike[str]]) -> None:
    """Ensure a collection of artifact-relative directories exist."""
    for rel in paths:
        artifacts_path(rel)


def dataset_npz_path(dataset_id: str) -> Path:
    """Return the canonical path for persisted dataset arrays."""
    filename = f"{dataset_id}.npz"
    return artifacts_path("datasets", filename)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomically write a DataFrame to parquet to avoid partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Atomically persist arrays to NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".npz") as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez(tmp_path, **arrays)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load arrays from an NPZ file."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
