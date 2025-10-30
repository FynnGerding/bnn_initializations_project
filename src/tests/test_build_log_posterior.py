import jax.numpy as jnp
import jax.random as jrnd

from bnn_initializations_project.models.bnn import (
    Bayesian_MLP,
    build_log_posterior_fn,
    flatten_params,
    initialize_prior,
)


def test_build_log_posterior_differs_by_prior():
    layer_widths = [3, 4, 1]
    key = jrnd.PRNGKey(0)
    params = initialize_prior(layer_widths, "isotropic_gaussian", key)
    theta, unravel = flatten_params(params)
    X = jnp.ones((8, 3), dtype=jnp.float32)
    y = jnp.ones((8, 1), dtype=jnp.float32)

    logpost_gauss = build_log_posterior_fn(
        unravel,
        layer_widths,
        sigma=0.1,
        prior_name="isotropic_gaussian",
    )
    logpost_laplace = build_log_posterior_fn(
        unravel,
        layer_widths,
        sigma=0.1,
        prior_name="laplace",
    )

    lp_gauss = float(logpost_gauss(theta, X, y))
    lp_laplace = float(logpost_laplace(theta, X, y))
    assert lp_gauss != lp_laplace
