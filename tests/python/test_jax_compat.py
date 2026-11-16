"""Numerical fences remain identities through mixed-type JAX transformations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from libsdca import _jax_compat
from libsdca._jax_compat import optimization_barrier


@pytest.mark.parametrize("force_legacy", [False, True])
def test_barrier_transforms_and_lowering(force_legacy: bool, monkeypatch: pytest.MonkeyPatch):
    if force_legacy:
        monkeypatch.setattr(_jax_compat, "_LEGACY", _jax_compat._legacy_barrier())

    def loss(x: jax.Array):
        fenced = optimization_barrier({"value": x, "index": jnp.int32(7), "empty": ()})
        return (fenced["value"] ** 3).sum() + fenced["index"] * 0

    x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 4
    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(loss)))(x)
    np.testing.assert_array_equal(values, (x**3).sum(-1))
    np.testing.assert_array_equal(gradients, 3 * x**2)
    hessian = jax.jit(jax.jacfwd(jax.grad(loss)))(x[1])
    np.testing.assert_array_equal(hessian, np.diag(6 * np.asarray(x[1])))
    assert "optimization_barrier" in jax.jit(loss).lower(x).as_text()
    assert "optimization_barrier" in jax.jit(jax.grad(loss)).lower(x).as_text()
    assert optimization_barrier(()) == ()


def test_nextafter_derivative_broadcast():
    x = jnp.asarray([0.25, 1.0, 2.0], jnp.float32)
    y = jnp.zeros((2, 3), jnp.float32)
    value, derivative = jax.jvp(_jax_compat.nextafter, (x, y), (jnp.ones_like(x), y))
    np.testing.assert_array_equal(value, jnp.nextafter(x, y))
    np.testing.assert_array_equal(derivative, np.ones((2, 3), np.float32))
    gradient = jax.jit(jax.grad(lambda a: _jax_compat.nextafter(a, 0.0).sum()))(x)
    np.testing.assert_array_equal(gradient, np.ones(3, np.float32))
