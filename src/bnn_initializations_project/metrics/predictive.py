"""Predictive performance metrics."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jrnd
import numpy as np

from bnn_initializations_project.models.bnn import Bayesian_MLP

__all__ = [
    "coverage",
    "nll_gaussian",
    "posterior_predictive_draws",
    "rmse",
]


def posterior_predictive_draws(
    samples_theta: jnp.ndarray,
    unravel_fn: Callable[[jnp.ndarray], any],
    X: np.ndarray,
    lik_sigma: float,
    activation: Callable[[jnp.ndarray], jnp.ndarray],
    key: jax.Array,
    S: int | None = None,
) -> jnp.ndarray:
    """Generate posterior predictive draws with Gaussian observation noise."""
    samples_theta = jnp.asarray(samples_theta)
    num_draws = samples_theta.shape[0]
    if num_draws == 0:
        raise ValueError("No posterior samples provided.")

    if S is None or S >= num_draws:
        indices = jnp.arange(num_draws)
        S = num_draws
    else:
        key_indices, key_noise = jrnd.split(key)
        indices = jrnd.choice(key_indices, num_draws, shape=(S,), replace=False)
        key = key_noise

    draw_keys = jrnd.split(key, S)
    X_jax = jnp.asarray(X, dtype=jnp.float32)

    def _single(idx: jnp.ndarray, draw_key: jax.Array) -> jnp.ndarray:
        theta = samples_theta[idx]
        params = unravel_fn(theta)
        mean = Bayesian_MLP.forward(params, X_jax, activation=activation)
        noise = lik_sigma * jrnd.normal(draw_key, shape=mean.shape, dtype=mean.dtype)
        return mean + noise

    draws = jax.vmap(_single)(indices, draw_keys)
    return draws


def nll_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    """Negative log-likelihood under a fixed Gaussian sigma."""
    y = jnp.asarray(y_true, jnp.float32).reshape(1, -1)
    mu = jnp.asarray(mu, jnp.float32)
    if mu.ndim == 1:
        mu = mu.reshape(1, -1)
    S = mu.shape[0]
    sig2 = (jnp.asarray(sigma, jnp.float32) + 1e-12) ** 2
    log_comp = -0.5*jnp.log(2*jnp.pi*sig2) - 0.5*((y - mu)**2)/sig2
    log_mix = jax.nn.logsumexp(log_comp, axis=0) - jnp.log(S)
    return float(-jnp.mean(log_mix))


def rmse(y_true: np.ndarray, mu: np.ndarray) -> float:
    """Root mean squared error."""
    y_true = jnp.asarray(y_true).reshape(-1)
    mu = jnp.asarray(mu).reshape(-1)
    return float(jnp.sqrt(jnp.mean((y_true - mu) ** 2)))


def coverage(
    y_true: np.ndarray,
    draws: jnp.ndarray,
    alpha: float = 0.1,
) -> tuple[float, float]:
    """Return (coverage, mean interval width) for two-sided credible intervals."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")

    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q

    interval = jnp.quantile(draws, jnp.array([lower_q, upper_q]), axis=0)
    lower, upper = interval[0], interval[1]

    y_true = jnp.asarray(y_true)
    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage_prob = float(jnp.mean(in_interval.astype(jnp.float32)))
    interval_width = float(jnp.mean(upper - lower))
    return coverage_prob, interval_width
