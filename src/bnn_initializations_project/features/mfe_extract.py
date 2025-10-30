"""PyMFE wrapper utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:  # pragma: no cover - optional dependency handled at runtime
    from pymfe.mfe import MFE
except ImportError:  # pragma: no cover
    MFE = None

__all__ = ["compute_mfe"]


def compute_mfe(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[str] = ("general", "statistical", "info-theory"),
    summary: Iterable[str] = ("mean", "min", "max"),
) -> dict[str, float]:
    """Compute PyMFE meta-features for a dataset."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    mfe = MFE(groups=tuple(groups), summary=tuple(summary))
    mfe.fit(X, y)
    names, values = mfe.extract()
    return {name: float(value) for name, value in zip(names, values)}
