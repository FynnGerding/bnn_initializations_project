"""Inference routines for Bayesian neural networks."""
# `metropolis_hastings` AND `proposal_dist` code taken from my A1MCMC.ipynb

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrnd

from .bnn import Bayesian_MLP, build_log_posterior_fn, flatten_params

__all__ = ["FitResult", "fit_bnn", "metropolis_hastings", "proposal_dist"]


def proposal_dist(key: jax.Array, state: jnp.ndarray, *, step_size: float) -> jnp.ndarray:
    """Gaussian random walk proposal."""
    return state + step_size * jrnd.normal(key, shape=state.shape, dtype=state.dtype)


def metropolis_hastings(
    log_prob_fn: Callable[[jnp.ndarray], jnp.ndarray],
    proposal_fn: Callable[[jax.Array, jnp.ndarray, float], jnp.ndarray],
    initial_state: jnp.ndarray,
    *,
    num_samples: int,
    num_burn_in: int,
    key: jax.Array,
    step_size: float,
) -> Tuple[jnp.ndarray, float]:
    """Run a single-chain Metropolis–Hastings sampler."""
    total = num_samples + num_burn_in

    @jax.jit
    def step(carry, k):
        state, logp, accepted = carry
        key_prop, key_acc = jrnd.split(k)
        proposed = proposal_fn(key_prop, state, step_size)
        logp_prop = log_prob_fn(proposed)
        log_alpha = logp_prop - logp
        accept = jnp.log(jrnd.uniform(key_acc)) < log_alpha
        next_state = jnp.where(accept, proposed, state)
        next_logp = jnp.where(accept, logp_prop, logp)
        accepted = accepted + accept.astype(jnp.int32)
        return (next_state, next_logp, accepted), next_state

    initial_logp = log_prob_fn(initial_state)
    carry0 = (initial_state, initial_logp, jnp.array(0, dtype=jnp.int32))
    keys = jrnd.split(key, total)
    (final_state, final_logp, accepted), states = jax.lax.scan(step, carry0, keys)
    samples = states[num_burn_in:]
    accept_rate = (accepted.astype(jnp.float32) / total).item()
    return samples, accept_rate


@dataclass
class FitResult:
    """Container for posterior samples and metadata."""

    samples: jnp.ndarray
    diagnostics: dict
    unravel_fn: Callable[[jnp.ndarray], any]


def fit_bnn(
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    *,
    layer_widths,
    activation,
    prior_name: str,
    nu: float,
    lik_sigma: float,
    init_scheme: str = "isotropic_gaussian",
    num_samples: int = 3000,
    num_burn_in: int = 1000,
    step_size: float = 0.02,
    seed: int = 0,
    prior_scale: float = 1.0,
) -> FitResult:
    """Fit a Bayesian MLP using random-walk Metropolis-Hastings."""

    key = jrnd.PRNGKey(seed)
    net = Bayesian_MLP(layer_widths, init_scheme, activation=activation, rng_key=key)
    theta0, unravel = flatten_params(net.params)
    log_post = build_log_posterior_fn(
        unravel,
        layer_widths,
        sigma=lik_sigma,
        activation=activation,
        prior_name=prior_name,
        nu=nu,
        prior_scale=prior_scale,
    )

    X_train = jnp.asarray(X_train, dtype=jnp.float32)
    y_train = jnp.asarray(y_train, dtype=jnp.float32).reshape(-1, 1)

    def _log_prob(theta):
        return log_post(theta, X_train, y_train)

    key_run, key_sampler = jrnd.split(key)
    samples, accept_rate = metropolis_hastings(
        _log_prob,
        lambda k, state, step_size: proposal_dist(k, state, step_size=step_size),
        theta0,
        num_samples=num_samples,
        num_burn_in=num_burn_in,
        key=key_sampler,
        step_size=step_size,
    )

    diagnostics = {
        "accept_rate": accept_rate,
        "num_draws": int(samples.shape[0]),
        "seed": seed,
        "prior_name": prior_name,
        "nu": nu,
        "step_size": step_size,
    }

    return FitResult(samples=samples, diagnostics=diagnostics, unravel_fn=unravel)
