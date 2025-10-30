"""Additional meta-feature utilities."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

__all__ = ["mean_abs_corr", "nonlinearity_index", "outlier_rate_y"]


def nonlinearity_index(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Return R²(RandomForest) - R²(ElasticNet) on the training data."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must align on the first dimension.")
    if X.shape[0] < 5:
        return 0.0
    if X.shape[1] == 0:
        return 0.0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    enet = ElasticNetCV(
        l1_ratio=(0.1, 0.5, 0.9),
        max_iter=5000,
        alphas=50,
        cv=3,
        random_state=seed,
    )
    try:
        enet.fit(X_scaled, y)
    except ValueError:
        return 0.0
    y_enet = enet.predict(X_scaled)
    r2_enet = float(r2_score(y, y_enet))

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=seed,
        n_jobs=1,
    )
    rf.fit(X, y)
    y_rf = rf.predict(X)
    r2_rf = float(r2_score(y, y_rf))

    return r2_rf - r2_enet


def outlier_rate_y(y: np.ndarray, threshold: float = 3.0) -> float:
    """Fraction of targets with |z| greater than threshold."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return 0.0
    mu = float(np.mean(y))
    sigma = float(np.std(y))
    if sigma < 1e-12:
        return 0.0
    z = (y - mu) / sigma
    return float(np.mean(np.abs(z) > threshold))


def mean_abs_corr(X: np.ndarray) -> float:
    """Mean absolute pairwise feature correlation."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 2:
        return 0.0
    corr = np.corrcoef(X, rowvar=False)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    values = np.abs(corr[mask])
    if values.size == 0:
        return 0.0
    return float(np.nanmean(np.where(np.isfinite(values), values, np.nan)))
