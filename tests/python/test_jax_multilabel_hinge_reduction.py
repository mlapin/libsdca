"""JAX scalar hinge uses wide row values before the final FP32 rounding."""

import jax
import jax.numpy as jnp
import libsdca.jax as J
import numpy as np
import pytest


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_mapped_hinge_reduces_before_rounding(reduction: J.Reduction) -> None:
    # The two groups have different scalar values. In the first group the wide row
    # loss 2**24 + 1 would be prematurely rounded down to 2**24 in FP32.
    z = jnp.array([[[0, 2**24], [0, 1]], [[1, 0], [0, 0]]], jnp.float32)
    mask = jnp.array([[True, False], [True, False]])

    def loss(x: jax.Array) -> jax.Array:
        return J.multilabel_hinge_loss(x, mask, reduction=reduction)

    mapped = jax.vmap(loss)
    scale = jnp.array([-1.5, 0.25], jnp.float32)

    @jax.jit
    def apply(x: jax.Array, cotangent: jax.Array) -> tuple[jax.Array, jax.Array]:
        value, pb = jax.vjp(mapped, x)
        return value, pb(cotangent)[0]

    divisor = 2 if reduction == "mean" else 1
    expected = np.array([(2**24 + 3) / divisor, 1 / divisor], np.float32)
    expected_grad = np.array([[[-1, 1], [-1, 1]], [[-0.5, 0.5], [-1, 1]]], np.float32) / divisor
    value, gradient = apply(z, scale)
    assert value.dtype == gradient.dtype == jnp.float32
    np.testing.assert_array_equal(jax.jit(mapped)(z), expected)
    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(gradient, expected_grad * np.asarray(scale)[:, None, None])
