"""Measured JAX fp32 defaults agree across native and array paths."""

from typing import Any

import jax
import jax.numpy as jnp
import named_losses
import numpy as np
import pytest
from libsdca import _jax_ffi
from libsdca import jax as J
from libsdca._jax_fusion import _precision_scope


@pytest.mark.parametrize(
    "loss,classes,k",
    [
        ("proximal_cross_entropy", 13, 1),
        ("topk_cross_entropy", 10, 5),
        ("topk_cross_entropy", 64, 63),
    ],
)
@pytest.mark.parametrize("backend", ["auto", "native", "jax"])
@pytest.mark.parametrize("ffi", ["on", "off"])
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_default_fp32_work(
    loss: str,
    classes: int,
    k: int,
    backend: str,
    ffi: str,
    x64: bool,
    dtype: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    if ffi == "on" and backend != "jax" and (reason := _jax_ffi.missing_tools()):
        pytest.skip(f"forced native FFI unavailable: {reason}")
    monkeypatch.setenv("SDCA_JAX_FFI", ffi)
    with _precision_scope(x64):
        z = jnp.asarray(np.random.default_rng(1729).normal(size=(7, classes)), dtype=dtype)
        y = jnp.arange(7)

        def evaluate(precision: str | None) -> tuple[jax.Array, jax.Array]:
            def evaluate_loss(x: jax.Array) -> jax.Array:
                return named_losses.loss(J, loss, x, y, k=k, backend=backend, precision=precision)

            return jax.jit(jax.value_and_grad(evaluate_loss))(z)

        default, narrow = evaluate(None), evaluate("float32")
        for actual, expected in zip(default, narrow, strict=True):
            np.testing.assert_array_equal(actual, expected)
        if x64:
            wide = evaluate("float64")
            for actual, expected in zip(default, wide, strict=True):
                np.testing.assert_allclose(
                    np.asarray(actual, dtype=np.float64),
                    np.asarray(expected, dtype=np.float64),
                    rtol=0.02 if dtype == jnp.bfloat16 else 1e-5,
                    atol=0.002 if dtype == jnp.bfloat16 else 1e-7,
                )
        else:
            # Every backend: float64 work needs jax_enable_x64.
            with pytest.raises(ValueError, match="jax_enable_x64"):
                evaluate("float64")
    jax.clear_caches()


@pytest.mark.parametrize("shape", [(256, 10), (4096, 10), (4096, 32), (256, 65)])
@pytest.mark.parametrize("threads", ["1", "18"])
def test_topk_default_boundary(
    shape: tuple[int, int], threads: str, monkeypatch: pytest.MonkeyPatch
):
    """Wide solves and the measured Mac native scheduling exception stay explicit."""
    import sys

    monkeypatch.setenv("SDCA_NUM_THREADS", threads)
    monkeypatch.setenv("OMP_NUM_THREADS", threads)
    with _precision_scope(True):
        n, c = shape
        z = jnp.asarray(np.random.default_rng(1847).normal(size=shape), jnp.float32)
        y = jnp.arange(n) % c
        wide = c > 64 or (sys.platform == "darwin" and n * c > 16384 and c < 32 and threads != "1")

        def evaluate(precision: str | None):
            return jax.jit(
                jax.value_and_grad(lambda x: J.topk_cross_entropy(x, y, k=5, precision=precision))
            )(z)

        for actual, expected in zip(
            evaluate(None), evaluate("float64" if wide else "float32"), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
    jax.clear_caches()
