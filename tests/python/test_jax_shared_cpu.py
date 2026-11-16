"""Shared CPU dispatch, first derivatives and JAX transformations."""

import math
import os
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from libsdca import _jax_cpu, _jax_ffi
from libsdca import jax as losses
from libsdca.jax import Reduction
from test_jax import x64

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu" or not hasattr(jax, "ffi"),
    reason="shared CPU kernels require JAX FFI",
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_shared_cpu_dispatch_and_derivatives(
    dtype: type[np.float32] | type[np.float64], reduction: Reduction
):
    rng = np.random.default_rng(3)
    s, t = (rng.normal(size=(3, 7)).astype(dtype) for _ in range(2))
    with x64(dtype == np.float64), mock.patch.dict(os.environ, {"SDCA_JAX_FFI": "on"}):
        z, teacher = jnp.asarray(s), jnp.asarray(t)

        def loss(a: jax.Array):
            return losses.distillation_loss(a, teacher, temperature=2.0, reduction=reduction)

        compiled = jax.jit(loss)
        assert _jax_ffi.DISTILLATION in compiled.lower(z).as_text()
        value = compiled(z)

        def total(a: jax.Array):
            return loss(a).sum()

        gradient = jax.jit(jax.grad(total))(z)
        hessian = jax.jit(jax.jacfwd(jax.grad(total)))(z)
        with mock.patch.object(_jax_cpu, "distillation", return_value=None):
            expected = jax.jit(
                lambda a: losses.distillation_loss(a, teacher, temperature=2.0, reduction=reduction)
            )(z)

            def reference(a: jax.Array):
                return losses._distillation_reference(a, teacher, 2.0, False, reduction).sum()

            expected_hessian = jax.jit(jax.jacfwd(jax.grad(reference)))(z)
            expected_gradient = jax.jit(jax.grad(reference))(z)
        eps = np.finfo(dtype).eps
        np.testing.assert_allclose(value, expected, rtol=64 * eps, atol=0)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=0, atol=64 * eps)
        np.testing.assert_allclose(hessian, expected_hessian, rtol=0, atol=128 * eps)


def test_shared_cpu_vmap_and_far_tail():
    s = np.zeros((3, 1000), np.float32)
    s[:, 1:] = -np.array([88, 90, 92], np.float32)[:, None]
    t = s.copy()
    t[:, 1:] -= 1
    with x64(False), mock.patch.dict(os.environ, {"SDCA_JAX_FFI": "on"}):

        def loss(a: jax.Array, b: jax.Array):
            return losses.distillation_loss(a, b, reduction="sum")

        batched = jax.jit(jax.value_and_grad(loss))(s, t)
        values, gradients = jax.jit(jax.vmap(jax.value_and_grad(loss)))(s, t)
        np.testing.assert_allclose(values.sum(), batched[0], rtol=2e-6)
        np.testing.assert_array_equal(gradients, batched[1])
        assert np.all(np.asarray(values) > 0)
        assert np.all(np.asarray(gradients)[:, 0] < 0)


def test_shared_cpu_float64_reduction():
    rng = np.random.default_rng(53)
    s, t = (rng.normal(size=(8192, 2)) * 4 for _ in range(2))
    with x64(True), mock.patch.dict(os.environ, {"SDCA_JAX_FFI": "on"}):
        rows = jax.jit(lambda a, b: losses.distillation_loss(a, b, reduction="none"))(s, t)
        expected = math.fsum(np.asarray(rows))
        cases: tuple[tuple[Reduction, int], ...] = (("sum", 1), ("mean", len(s)))
        for reduction, divisor in cases:

            def loss(a: jax.Array, b: jax.Array, reduction: Reduction = reduction) -> jax.Array:
                return losses.distillation_loss(a, b, reduction=reduction)

            actual = jax.jit(loss)(s, t)
            assert float(actual) == pytest.approx(expected / divisor, rel=4 * np.finfo(float).eps)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_shared_statistics_match_torch(
    dtype: str, precision: str, reduction: Reduction, monkeypatch: pytest.MonkeyPatch
):
    # This checks identical shared SIMD arithmetic, not the separate portable
    # fallback selected by the coverage harness's extension-off pass.
    monkeypatch.setenv("SDCA_CPU_EXTENSION", "on")
    torch = pytest.importorskip("torch")
    from libsdca import torch as losses_torch

    rng = np.random.default_rng(73)
    s, t = (torch.tensor(rng.normal(size=(4, 67)), dtype=getattr(torch, dtype)) for _ in range(2))
    t[1] = s[1]
    s[2, 1:], t[2, 1:] = -30, -32
    s.requires_grad_()
    expected = losses_torch.distillation_loss(
        s, t, temperature=2.0, reduction=reduction, precision=precision
    )
    up = torch.linspace(0.3, 1.7, 4) if reduction == "none" else torch.tensor(1.7)
    (gradient,) = torch.autograd.grad(expected, s, up)
    with x64(precision == "float64"), mock.patch.dict(os.environ, {"SDCA_JAX_FFI": "on"}):
        z = jnp.asarray(s.detach().float().numpy(), dtype=getattr(jnp, dtype))
        teacher = jnp.asarray(t.float().numpy(), dtype=z.dtype)

        def loss(a: jax.Array):
            return losses.distillation_loss(
                a, teacher, temperature=2.0, reduction=reduction, precision=precision
            )

        def run(a: jax.Array):
            value, backward = jax.vjp(loss, a)
            return value, backward(jnp.asarray(up.numpy()))[0]

        compiled = jax.jit(run)
        if dtype != "float32" or precision == "float64":
            assert _jax_ffi.DISTILLATION_STATS in compiled.lower(z).as_text()
            assert _jax_ffi.DISTILLATION_BACKWARD in compiled.lower(z).as_text()
        value, actual = compiled(z)
        np.testing.assert_allclose(value, expected.detach().numpy(), rtol=2e-7, atol=0)
        np.testing.assert_array_equal(np.asarray(actual, np.float32), gradient.float().numpy())
        # The opaque saved statistics must not prevent higher derivatives or batching.
        hessian = jax.jit(jax.jacfwd(jax.grad(lambda a: loss(a).sum())))(z)
        assert np.isfinite(np.asarray(hessian, np.float32)).all()


def test_shared_cpu_extreme_temperature_backward(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SDCA_CPU_EXTENSION", "on")
    torch = pytest.importorskip("torch")
    from libsdca import torch as losses_torch

    # T * N overflows float32 when T = 8e37 and N = 8, while (q - p) * ct / (N * T) is O(0.1).
    s = np.tile(np.array([[-8e37, 8e37]], dtype=np.float32), (8, 1))
    t = -s
    st = torch.tensor(s, requires_grad=True)
    expected = losses_torch.distillation_loss(
        st, torch.tensor(t), temperature=8e37, reduction="mean"
    )
    (expected_grad,) = torch.autograd.grad(expected, st, torch.tensor(8e37, dtype=torch.float32))
    with x64(False), mock.patch.dict(os.environ, {"SDCA_JAX_FFI": "on"}):
        _, pullback = jax.jit(
            lambda a: jax.vjp(
                lambda x: losses.distillation_loss(
                    x, jnp.asarray(t), temperature=8e37, reduction="mean"
                ),
                a,
            )
        )(jnp.asarray(s))
        (actual_grad,) = pullback(jnp.asarray(8e37, dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(actual_grad), expected_grad.numpy(), rtol=2e-7, atol=0)
    assert float(np.max(np.abs(actual_grad))) > 0.05
