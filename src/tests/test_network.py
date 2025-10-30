import math
import pytest
import jax
import jax.numpy as jnp
import jax.random as jrnd

from bnn_initializations_project import _xavier_std, initialize_prior, Bayesian_MLP

@pytest.mark.parametrize("fan_in,fan_out", [(7, 11), (64, 64), (128, 32)])
def test_xavier_std_matches_formula(fan_in, fan_out):
    s = _xavier_std(fan_in, fan_out)
    assert s.dtype == jnp.float32
    expected = math.sqrt(2.0 / (fan_in + fan_out))
    assert jnp.allclose(s, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("scheme", [
    "isotropic_gaussian",
    "laplace",
    "student-t",
    "sparse_horse-shoe",
    "wishart_hierarchical_hyperprior",
])
def test_initialize_prior_shapes_and_bias_zero(scheme):
    layer_widths = [50, 40, 30]  # two layers
    key = jrnd.PRNGKey(0)
    params = initialize_prior(layer_widths, scheme, key)
    assert isinstance(params, tuple)
    assert len(params) == len(layer_widths) - 1

    for (fan_in, fan_out), layer in zip(zip(layer_widths[:-1], layer_widths[1:]), params):
        assert layer["w"].shape == (fan_in, fan_out)
        assert layer["b"].shape == (fan_out,)
        assert jnp.allclose(layer["b"], 0.0)


def test_initialize_prior_reproducible_and_key_sensitive():
    layer_widths = [16, 8, 4]
    key1 = jrnd.PRNGKey(42)
    key2 = jrnd.PRNGKey(43)

    p1 = initialize_prior(layer_widths, "isotropic_gaussian", key1)
    p1b = initialize_prior(layer_widths, "isotropic_gaussian", key1)
    p2 = initialize_prior(layer_widths, "isotropic_gaussian", key2)

    for a, b in zip(p1, p1b):
        assert jnp.array_equal(a["w"], b["w"])
        assert jnp.array_equal(a["b"], b["b"])

    diffs = [jnp.any(a["w"] != b["w"]) for a, b in zip(p1, p2)]
    assert any(bool(d) for d in diffs)


@pytest.mark.parametrize("scheme", ["isotropic_gaussian", "laplace", "student-t", "wishart_hierarchical_hyperprior"])
def test_weight_empirical_std_matches_target(scheme):
    fan_in, fan_out = 200, 300
    layer_widths = [fan_in, fan_out]
    key = jrnd.PRNGKey(0)

    params = initialize_prior(layer_widths, scheme, key)
    (layer,) = params
    W = layer["w"]

    target = _xavier_std(fan_in, fan_out)
    emp = jnp.std(W)
    assert jnp.allclose(emp, target, rtol=0.25, atol=0.0), f"emp={float(emp)} target={float(target)}"


def test_bayesian_mlp_shapes_and_call_equivalence():
    mlp = Bayesian_MLP([5, 4, 3], "isotropic_gaussian", activation=jax.nn.tanh, rng_key=jrnd.PRNGKey(0))
    x = jnp.ones((10, 5), dtype=jnp.float32)

    y1 = mlp.apply(x)
    y2 = mlp(x)  # __call__
    assert y1.shape == (10, 3)
    assert jnp.allclose(y1, y2)

    shapes = mlp.shapes()
    assert shapes == [{"w": (5, 4), "b": (4,)}, {"w": (4, 3), "b": (3,)}]


def test_forward_no_activation_on_last_layer():
    params = (
        {"w": jnp.array([[-1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32), "b": jnp.zeros((2,), jnp.float32)},
        {"w": jnp.array([[1.0], [-3.0]], dtype=jnp.float32), "b": jnp.zeros((1,), jnp.float32)},
    )
    x = jnp.array([1.0, 2.0], dtype=jnp.float32)
    out = Bayesian_MLP.forward(params, x, activation=jax.nn.relu)
    assert out.shape == (1,)
    assert jnp.allclose(out, -6.0)
