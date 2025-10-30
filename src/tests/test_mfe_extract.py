import numpy as np
import pytest

from bnn_initializations_project.features.mfe_extract import compute_mfe


def test_compute_mfe_returns_features():
    X = np.random.RandomState(0).normal(size=(30, 5))
    y = np.random.RandomState(1).normal(size=(30,))
    feats = compute_mfe(X, y, groups=("general",), summary=("mean",))
    assert isinstance(feats, dict)
    assert feats
