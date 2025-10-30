"""Weight prior log-density utilities."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import jax.scipy as jsp

PriorName = Literal["isotropic_gaussian", "gaussian", "laplace", "student_t", "student-t"]

__all__ = [
    "PriorName",
    "layer_logprior",
    "logpdf_laplace",
    "logpdf_normal",
    "logpdf_student_t",
]


def logpdf_normal(x: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    """Element-wise Gaussian log-density summed over all entries.

    Supports scalar or per-parameter std that broadcasts to x.
    """
    x = jnp.asarray(x)
    std = jnp.asarray(std)
    z = x / std
    n = x.size
    return -0.5 * jnp.sum(z**2) - jnp.sum(jnp.log(std + 1e-12)) - 0.5 * n * jnp.log(2.0 * jnp.pi)


def logpdf_laplace(x: jnp.ndarray, target_std: jnp.ndarray) -> jnp.ndarray:
    """Element-wise Laplace(logistic) log-density with variance matching.

    For Laplace, Var = 2 b^2, so b = target_std / sqrt(2).
    Supports scalar or per-parameter target_std that broadcasts to x.
    """
    x = jnp.asarray(x)
    target_std = jnp.asarray(target_std)
    b = target_std / jnp.sqrt(2.0)
    return -jnp.sum(jnp.abs(x) / (b + 1e-12)) - jnp.sum(jnp.log(2.0 * b + 1e-12))


def logpdf_student_t(x: jnp.ndarray, target_std: jnp.ndarray, nu: float = 5.0) -> jnp.ndarray:
    """Element-wise Student-t log-density matched to the provided variance.

    For df v > 2, Var = v s^2 / (v - 2). We set s so that the target std matches.
    Supports scalar or per-parameter target_std that broadcasts to x.
    """
    x = jnp.asarray(x)
    target_std = jnp.asarray(target_std)
    if isinstance(nu, (int, float)) and nu <= 2.0:
        raise ValueError(f"Student-t prior requires nu > 2, got {nu}")

    nu = jnp.asarray(nu, dtype=x.dtype)

    s = target_std * jnp.sqrt((nu - 2.0) / nu)
    z = x / s
    c = (
        jsp.special.gammaln(0.5 * (nu + 1.0))
        - jsp.special.gammaln(0.5 * nu)
        - 0.5 * jnp.log(nu * jnp.pi)
    )
    log_det = jnp.sum(jnp.log(s + 1e-12))
    return (
        x.size * c
        - log_det
        - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p((z**2) / nu))
    )


def layer_logprior(
    w: jnp.ndarray,
    b: jnp.ndarray,
    prior_name: PriorName,
    *,
    dtype=jnp.float32,
    prior_scale: float = 1.0,
    nu: float = 5.0,
) -> jnp.ndarray:
    """Return the joint log-prior for a single layer."""
    fan_in, fan_out = w.shape
    w_std = prior_scale * jnp.sqrt(jnp.asarray(2.0 / (fan_in + fan_out), dtype=dtype))
    b_std = 0.1 * w_std

    if prior_name == "isotropic_gaussian":
        lp_w = logpdf_normal(w, w_std)
        lp_b = logpdf_normal(b, b_std)
    elif prior_name == "laplace":
        lp_w = logpdf_laplace(w, w_std)
        lp_b = logpdf_laplace(b, b_std)
    elif prior_name == "student_t":
        lp_w = logpdf_student_t(w, w_std, nu=nu)
        lp_b = logpdf_student_t(b, b_std, nu=nu)
    else:
        raise ValueError(f"Unsupported prior: {prior_name}")

    return lp_w + lp_b
