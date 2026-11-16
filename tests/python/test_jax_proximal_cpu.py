"""Native proximal-distillation CPU dispatch, storage and row semantics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike
from libsdca import _jax_ffi
from libsdca.jax import Reduction, proximal_distillation_loss
from numpy_api import proximal_distillation_loss_and_grad

jax.config.update("jax_enable_x64", True)
pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu" or _jax_ffi.missing_tools() is not None,
    reason="requires CPU FFI build tools",
)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("alpha", [0.001, 0.5, 1000.0])
@pytest.mark.parametrize("precision", [None, "float64"])
def test_native_rows_and_scaled_gradient(
    monkeypatch: pytest.MonkeyPatch,
    dtype: DTypeLike,
    reduction: Reduction,
    alpha: float,
    precision: str | None,
) -> None:
    monkeypatch.setenv("SDCA_JAX_FFI", "on")
    rng = np.random.default_rng(713)
    student = jnp.asarray(rng.normal(size=(3, 11)), dtype)
    teacher = jnp.asarray(rng.normal(size=(3, 11)), dtype).at[0].set(student[0])
    storage = np.float64 if dtype == jnp.float64 or precision == "float64" else np.float32
    value, gradient = proximal_distillation_loss_and_grad(
        np.asarray(student, storage), np.asarray(teacher, storage), temperature=2, alpha=alpha
    )
    expected_value = jnp.asarray(value, storage)
    upstream = jnp.asarray(
        [0.25, 2.0, 0.5] if reduction == "none" else 0.25, jnp.result_type(dtype, jnp.float32)
    )
    if reduction != "none":
        expected_value = getattr(expected_value, reduction)()
    factor = upstream if reduction == "none" else jnp.full((3,), upstream, storage)
    if reduction == "mean":
        factor /= 3
    expected_gradient = (jnp.asarray(gradient) * factor[:, None]).astype(dtype)

    def loss(z: jax.Array) -> jax.Array:
        return proximal_distillation_loss(
            z, teacher, temperature=2, alpha=alpha, reduction=reduction, precision=precision
        )

    call = jax.jit(loss)
    assert _jax_ffi.PROXIMAL_DISTILLATION in call.lower(student).as_text()
    result, pullback = jax.vjp(call, student)
    expected_value = expected_value.astype(jnp.result_type(dtype, jnp.float32))
    if dtype == jnp.float64:
        # The portable scalar oracle and shared SIMD solve use different reduction
        # orders; independent 80-digit references test the solve separately. Keep
        # exact storage/scaling checks below for every narrower dtype.
        np.testing.assert_allclose(
            result, expected_value, rtol=16 * np.finfo(np.float64).eps, atol=0
        )
        np.testing.assert_allclose(
            pullback(upstream)[0], expected_gradient, rtol=16 * np.finfo(np.float64).eps, atol=0
        )
    else:
        np.testing.assert_array_equal(result, expected_value)
        np.testing.assert_array_equal(pullback(upstream)[0], expected_gradient)
    np.testing.assert_array_equal(pullback(upstream)[0][0], 0)


def test_explicit_float_work_retains_array_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SDCA_JAX_FFI", "on")
    z = jnp.asarray([[1.0, 2.0, 3.0]], jnp.float32)
    call = jax.jit(lambda s: proximal_distillation_loss(s, -s, precision="float32"))
    assert _jax_ffi.PROXIMAL_DISTILLATION not in call.lower(z).as_text()
    assert bool(jnp.isfinite(call(z)))


@pytest.mark.parametrize("precision", [None, "float64"])
def test_overflow_loss_preserves_finite_gradient(
    monkeypatch: pytest.MonkeyPatch, precision: str | None
) -> None:
    monkeypatch.setenv("SDCA_JAX_FFI", "on")
    maximum = np.finfo(np.float64).max
    student = jnp.asarray([[-maximum, maximum, 0.0]], jnp.float64)
    teacher = -student
    expected_gradient = np.asarray([[-1.0, 1.0, 0.0]])
    value, gradient = proximal_distillation_loss_and_grad(
        np.asarray(student), np.asarray(teacher), temperature=1, alpha=1
    )
    assert np.isposinf(value).all()
    np.testing.assert_array_equal(gradient, expected_gradient)
    value, gradient = jax.jit(
        jax.value_and_grad(
            lambda s: proximal_distillation_loss(
                s, teacher, temperature=1, alpha=1, precision=precision
            )
        )
    )(student)
    assert np.isposinf(value)
    np.testing.assert_array_equal(gradient, expected_gradient)


@pytest.mark.parametrize("precision", [None, "float64"])
def test_native_hessian_matches_array(
    monkeypatch: pytest.MonkeyPatch, precision: str | None
) -> None:
    student = jnp.asarray([[0.25, -0.5, 1.0], [0.0, 0.0, -1.0]], jnp.float32)
    teacher = jnp.asarray([[0.0, 0.5, -0.5], [0.0, 0.0, -1.0]], jnp.float32)

    def loss(s: jax.Array) -> jax.Array:
        return proximal_distillation_loss(s, teacher, temperature=2, alpha=0.5, precision=precision)

    monkeypatch.setenv("SDCA_JAX_FFI", "off")
    expected = jax.jit(jax.hessian(loss))(student)
    monkeypatch.setenv("SDCA_JAX_FFI", "on")
    actual = jax.jit(jax.hessian(lambda s: loss(s)))(student)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)
    with pytest.raises(TypeError):
        jax.jvp(loss, (student,), (jnp.ones_like(student),))
