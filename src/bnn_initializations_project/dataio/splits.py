"""Train/ID/OOD split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

__all__ = ["SplitSummary", "make_splits"]


@dataclass(frozen=True)
class SplitSummary:
    """Description of the chosen out-of-distribution split."""

    ood_feature: int
    q10: float
    q90: float


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    *,
    id_ratio: float = 0.7,
    ood_tail_fraction: float = 0.1,
    ood_feature: int | None = None,
) -> Dict[str, Any]:
    """Construct train/ID/OOD splits.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        seed: RNG seed controlling the ID random split.
        id_ratio: Fraction of in-distribution data reserved for training.
        ood_tail_fraction: Tail mass per side for the OOD split.
        ood_feature: Optional pre-selected feature index for OOD splitting.

    Returns:
        Dictionary with arrays and ``SplitSummary`` metadata.
    """

    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim not in (1, 2):
        raise ValueError("y must be 1D or 2D.")

    y_flat = y.reshape(-1)
    if X.shape[0] != y_flat.shape[0]:
        raise ValueError(f"X and y must have matching first dimension.\nX shape: {X.shape} \ny shape: {y.shape}")

    n_samples, n_features = X.shape
    if n_samples == 0:
        raise ValueError("No samples provided.")

    ood_feature = int(ood_feature) if ood_feature is not None else int(np.argmax(np.var(X, axis=0)))
    feature_values = X[:, ood_feature]
    q_lo = float(np.quantile(feature_values, ood_tail_fraction))
    q_hi = float(np.quantile(feature_values, 1.0 - ood_tail_fraction))

    central_mask = (feature_values >= q_lo) & (feature_values <= q_hi)
    ood_mask = ~central_mask

    central_indices = np.nonzero(central_mask)[0]
    ood_indices = np.nonzero(ood_mask)[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(central_indices)

    n_train = max(1, int(np.floor(id_ratio * len(central_indices))))
    train_idx = central_indices[:n_train]
    id_idx = central_indices[n_train:]

    def _take(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return X[idx], y_flat[idx]

    X_train, y_train = _take(train_idx)
    X_id, y_id = _take(id_idx) if len(id_idx) else (np.empty((0, n_features)), np.empty((0,), dtype=y_flat.dtype))
    X_ood, y_ood = _take(ood_indices) if len(ood_indices) else (np.empty((0, n_features)), np.empty((0,), dtype=y_flat.dtype))

    summary = SplitSummary(ood_feature=ood_feature, q10=q_lo, q90=q_hi)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_id": X_id,
        "y_id": y_id,
        "X_ood": X_ood,
        "y_ood": y_ood,
        "summary": summary,
    }
