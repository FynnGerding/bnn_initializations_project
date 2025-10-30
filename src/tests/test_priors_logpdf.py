import jax.numpy as jnp

from bnn_initializations_project.models.priors_impl import (
    layer_logprior,
    logpdf_laplace,
    logpdf_normal,
    logpdf_student_t,
)


def test_logpdfs_are_finite():
    x = jnp.array([1.0, -0.5, 0.25], dtype=jnp.float32)
    std = jnp.array(1.0, dtype=jnp.float32)
    assert jnp.isfinite(logpdf_normal(x, std))
    assert jnp.isfinite(logpdf_laplace(x, std))
    assert jnp.isfinite(logpdf_student_t(x, std, nu=5.0))


def test_logpdf_disagreement_between_priors():
    x = jnp.array([1.0, 2.0, -1.5], dtype=jnp.float32)
    std = jnp.array(1.0, dtype=jnp.float32)
    lp_gauss = logpdf_normal(x, std)
    lp_laplace = logpdf_laplace(x, std)
    lp_student = logpdf_student_t(x, std, nu=5.0)
    assert not jnp.isclose(lp_gauss, lp_laplace)
    assert not jnp.isclose(lp_gauss, lp_student)


def test_layer_logprior_switches_with_prior():
    w = jnp.ones((2, 3), dtype=jnp.float32)
    b = jnp.zeros((3,), dtype=jnp.float32)
    lp_gauss = layer_logprior(w, b, "isotropic_gaussian")
    lp_laplace = layer_logprior(w, b, "laplace")
    assert not jnp.isclose(lp_gauss, lp_laplace)
