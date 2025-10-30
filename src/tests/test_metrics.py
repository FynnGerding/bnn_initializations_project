import numpy as np

from bnn_initializations_project.metrics.predictive import coverage


def test_coverage_near_nominal_for_gaussian_draws():
    rng = np.random.default_rng(0)
    mu = rng.normal(loc=0.0, scale=1.0, size=1000)
    draws = mu + rng.normal(loc=0.0, scale=1.0, size=(500, mu.size))
    y_true = mu + rng.normal(loc=0.0, scale=1.0, size=mu.size)
    cov, _ = coverage(y_true, draws, alpha=0.1)
    assert 0.85 <= cov <= 0.95
