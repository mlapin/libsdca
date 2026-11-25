"""CPU JAX distillation sums and means keep the row statistics wide until the scalar result."""

import math

import jax
import jax.numpy as jnp
import libsdca.jax as J
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(jax.default_backend() != "cpu", reason="requires CPU kernels")


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_uniform_distillation_rounds_once(reduction: J.Reduction) -> None:
    scores = jnp.zeros((31, 3), jnp.float32)
    expected = np.float32(math.log(3) * (31 if reduction == "sum" else 1))

    def loss(x: jax.Array) -> jax.Array:
        return J.distillation_loss(x, scores, reduction=reduction)

    np.testing.assert_array_equal(jax.jit(loss)(scores), expected)
    value, grad = jax.jit(jax.value_and_grad(loss))(scores)
    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(grad, jnp.zeros_like(scores))


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_mapped_distillation_reduces_each_example(reduction: J.Reduction) -> None:
    rng = np.random.default_rng(71)
    scores, teachers = rng.standard_normal((2, 2, 3, 5, 37), dtype=np.float32)

    def loss(x: jax.Array, t: jax.Array) -> jax.Array:
        return J.distillation_loss(x, t, temperature=2, reduction=reduction)

    single = jax.jit(jax.value_and_grad(loss))
    values, grads = jax.jit(jax.vmap(jax.vmap(single)))(scores, teachers)
    for index in np.ndindex(2, 3):
        value, grad = single(scores[index], teachers[index])
        np.testing.assert_array_equal(values[index], value)
        np.testing.assert_array_equal(grads[index], grad)


def test_native_teacher_recovers_signed_small_moment() -> None:
    from libsdca import _jax_ops

    if _jax_ops._handler()[0] is None:
        pytest.skip("optional native teacher FFI unavailable")
    tiny = 2.0**-100
    z = np.tile(np.array([[1, 0, -1, tiny], [tiny, -1, 0, 1]], np.float32), (16, 1))
    t = jnp.zeros_like(z)
    u = np.tile(np.array([-3, 2.5], np.float32), 16)

    def pull(teacher: jax.Array, seed: jax.Array) -> jax.Array:
        def loss(x: jax.Array) -> jax.Array:
            return J.distillation_loss(jnp.asarray(z), x, backend="kernel", reduction="none")

        _, back = jax.vjp(loss, teacher)
        return back(seed)[0]

    expected = ((tiny / 4 - z.astype(np.float64)) * (u[:, None] / 4)).astype(np.float32)
    np.testing.assert_array_equal(jax.jit(pull)(t, u), expected)
