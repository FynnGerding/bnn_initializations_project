"""Bayesian MLP definitions and log-posterior construction."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, TypedDict

import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree
from jax.random import PRNGKey

from .priors_impl import PriorName, layer_logprior

__all__ = [
    "BayesianMLPConfig",
    "Bayesian_MLP",
    "LayerParams",
    "PyTreeParams",
    "_xavier_std",
    "build_log_posterior_fn",
    "flatten_params",
    "initialize_prior",
]


class LayerParams(TypedDict):
    w: jnp.ndarray
    b: jnp.ndarray


PyTreeParams = Tuple[LayerParams, ...]


def _xavier_std(fan_in: int, fan_out: int, dtype=jnp.float32) -> jnp.ndarray:
    return jnp.sqrt(jnp.asarray(2.0 / (fan_in + fan_out), dtype=dtype))


def _zeros_bias(out_dim: int, dtype=jnp.float32) -> jnp.ndarray:
    return jnp.zeros((out_dim,), dtype=dtype)


def _eps(dtype) -> jnp.ndarray:
    return jnp.asarray(1e-12, dtype=dtype)


def _layer_key(base_key: PRNGKey, i: int) -> PRNGKey:
    return jrnd.fold_in(base_key, i)


def _sample_gaussian(key: PRNGKey, shape: Sequence[int], std: jnp.ndarray, dtype=jnp.float32) -> jnp.ndarray:
    return std * jrnd.normal(key, shape, dtype=dtype)


def _sample_laplace(key: PRNGKey, shape: Sequence[int], target_std: jnp.ndarray, dtype=jnp.float32) -> jnp.ndarray:
    b = target_std / jnp.sqrt(jnp.asarray(2.0, dtype=dtype))
    return b * jrnd.laplace(key, shape, dtype=dtype)


def _sample_student_t(
    key: PRNGKey,
    shape: Sequence[int],
    target_std: jnp.ndarray,
    *,
    nu: float = 5.0,
    dtype=jnp.float32,
) -> jnp.ndarray:
    nu = jnp.asarray(nu, dtype=dtype)
    s = target_std * jnp.sqrt((nu - 2.0) / nu)
    key_norm, key_gamma = jrnd.split(key)
    z = jrnd.normal(key_norm, shape, dtype=dtype)
    g = 2.0 * jrnd.gamma(key_gamma, nu * 0.5, shape=shape, dtype=dtype)
    return s * z / jnp.sqrt(g / nu + _eps(dtype))


def _half_cauchy(key: PRNGKey, shape: Sequence[int], scale: float = 1.0, dtype=jnp.float32) -> jnp.ndarray:
    u = jrnd.uniform(key, shape, dtype=dtype, minval=_eps(dtype), maxval=1.0 - _eps(dtype))
    return jnp.asarray(scale, dtype=dtype) * jnp.abs(jnp.tan(jnp.pi * (u - 0.5)))


def _sample_horseshoe_regularized(
    key: PRNGKey,
    shape: Sequence[int],
    target_std: jnp.ndarray,
    dtype=jnp.float32,
) -> jnp.ndarray:
    key_loc, key_glob, key_eps = jrnd.split(key, 3)
    lam = _half_cauchy(key_loc, shape, dtype=dtype)
    tau0 = target_std
    tau = _half_cauchy(key_glob, (), dtype=dtype) * jnp.maximum(tau0, _eps(dtype))
    c = jnp.asarray(1.0, dtype=dtype)
    shrink = (tau * lam * c) / jnp.sqrt(c * c + (tau * tau) * (lam * lam) + _eps(dtype))
    base = jrnd.normal(key_eps, shape, dtype=dtype)
    return target_std * shrink * base


def _bartlett_lower(key: PRNGKey, p: int, df: int, dtype=jnp.float32) -> jnp.ndarray:
    key_norm, key_gamma = jrnd.split(key)
    normals = jrnd.normal(key_norm, (p, p), dtype=dtype)
    alphas = 0.5 * (df - jnp.arange(p, dtype=dtype))
    gam = jrnd.gamma(key_gamma, alphas, shape=(p,), dtype=dtype)
    diag = jnp.sqrt(2.0 * gam + _eps(dtype))
    A = jnp.tril(normals, k=-1) + jnp.diag(diag)
    return A


def _wishart_weight_col(key: PRNGKey, p: int, df: int, dtype=jnp.float32) -> jnp.ndarray:
    key_A, key_z = jrnd.split(key)
    A = _bartlett_lower(key_A, p, df, dtype)
    z = jrnd.normal(key_z, (p,), dtype=dtype)
    return jsp.linalg.solve_triangular(A, z, lower=True)


def _sample_wishart_layer(
    key: PRNGKey,
    fan_in: int,
    fan_out: int,
    target_std: jnp.ndarray,
    dtype=jnp.float32,
) -> jnp.ndarray:
    df = int(fan_in + 1)
    keys = jrnd.split(key, fan_out)
    col_sampler = jax.vmap(lambda k: _wishart_weight_col(k, fan_in, df, dtype))
    w = jnp.stack(col_sampler(keys), axis=1)
    s_emp = jnp.std(w) + _eps(dtype)
    alpha = target_std / s_emp
    return alpha * w


def _init_layer(key: PRNGKey, fan_in: int, fan_out: int, init_scheme: str, dtype=jnp.float32) -> LayerParams:
    std = _xavier_std(fan_in, fan_out, dtype)
    if init_scheme == "isotropic_gaussian":
        w = _sample_gaussian(key, (fan_in, fan_out), std, dtype)
    elif init_scheme == "laplace":
        w = _sample_laplace(key, (fan_in, fan_out), std, dtype)
    elif init_scheme == "student-t":
        w = _sample_student_t(key, (fan_in, fan_out), std, dtype=dtype)
    elif init_scheme == "sparse_horse-shoe":
        w = _sample_horseshoe_regularized(key, (fan_in, fan_out), std, dtype=dtype)
    elif init_scheme == "wishart_hierarchical_hyperprior":
        w = _sample_wishart_layer(key, fan_in, fan_out, std, dtype=dtype)
    else:
        raise ValueError(f"Unsupported init scheme: {init_scheme}")
    b = _zeros_bias(fan_out, dtype)
    return LayerParams(w=w, b=b)


def initialize_prior(
    layer_widths: Sequence[int],
    init_scheme: str,
    rng_key: PRNGKey,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> PyTreeParams:
    """Initialize network parameters under the specified prior."""
    if len(layer_widths) < 2:
        raise ValueError("layer_widths must contain at least input and output dimension.")
    params: List[LayerParams] = []
    for i, (fan_in, fan_out) in enumerate(zip(layer_widths[:-1], layer_widths[1:])):
        layer_key = _layer_key(rng_key, i)
        params.append(_init_layer(layer_key, fan_in, fan_out, init_scheme, dtype=dtype))
    return tuple(params)


@dataclass
class BayesianMLPConfig:
    """Configuration for building a Bayesian MLP."""

    layer_widths: Sequence[int]
    init_scheme: str = "isotropic_gaussian"
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh
    dtype: jnp.dtype = jnp.float32


class Bayesian_MLP:
    """Simple Bayesian MLP wrapper around pure-JAX operations."""

    def __init__(
        self,
        layer_widths: Sequence[int],
        init_scheme: str,
        *,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh,
        rng_key: PRNGKey = jrnd.PRNGKey(0),
        name: Optional[str] = None,
    ):
        if len(layer_widths) < 2:
            raise ValueError("layer_widths must be at least [in_dim, out_dim].")
        self.name = name
        self.layer_widths = tuple(int(w) for w in layer_widths)
        self.init_scheme = init_scheme
        self.activation = activation
        self.rng_key = rng_key
        self.params = initialize_prior(self.layer_widths, init_scheme, rng_key)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("activation",))
    def forward(
        params: PyTreeParams,
        x: jnp.ndarray,
        *,
        activation: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> jnp.ndarray:
        z = x
        last_idx = len(params) - 1
        for i, layer in enumerate(params):
            w, b = layer["w"], layer["b"]
            z = z @ w + b
            if i < last_idx:
                z = activation(z)
        return z

    def apply(self, x: jnp.ndarray, params: Optional[PyTreeParams] = None) -> jnp.ndarray:
        p = self.params if params is None else params
        return Bayesian_MLP.forward(p, x, activation=self.activation)

    def shapes(self) -> List[dict[str, tuple[int, ...]]]:
        return [{"w": tuple(layer["w"].shape), "b": tuple(layer["b"].shape)} for layer in self.params]

    def __call__(self, x: jnp.ndarray, params: Optional[PyTreeParams] = None) -> jnp.ndarray:
        return self.apply(x, params=params)


def flatten_params(params: PyTreeParams):
    """Flatten parameters into a vector together with an unravel function."""
    flat, unravel = ravel_pytree(params)
    return flat, unravel


def build_log_posterior_fn(
    unravel_fn: Callable[[jnp.ndarray], PyTreeParams],
    layer_widths: Sequence[int],
    *,
    sigma: float = 1.0,
    dtype=jnp.float32,
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh,
    prior_name: PriorName = "isotropic_gaussian",
    nu: float = 5.0,
    prior_scale: float = 1.0,
    prior_weight: float = 1.0,
    lik_weight: float = 1.0,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Return log p(theta | X, y) for the specified prior."""

    layer_widths = tuple(int(w) for w in layer_widths)
    sigma_arr = jnp.asarray(sigma, dtype=dtype)
    log_sigma = jnp.log(sigma_arr + 1e-12)
    log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=dtype))

    @jax.jit
    def logpost(theta: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        params = unravel_fn(theta)

        log_prior = jnp.array(0.0, dtype=dtype)
        for layer in params:
            log_prior = log_prior + layer_logprior(
                layer["w"],
                layer["b"],
                prior_name=prior_name,
                dtype=dtype,
                prior_scale=prior_scale,
                nu=nu,
            )

        preds = Bayesian_MLP.forward(params, X, activation=activation)
        resid = (y - preds).reshape(-1)
        n = resid.size
        log_lik = -0.5 * jnp.sum((resid / sigma_arr) ** 2) - n * (log_sigma + 0.5 * log_two_pi)
        return prior_weight * log_prior + lik_weight * log_lik

    return logpost
