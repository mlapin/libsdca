"""JAX CUDA top-1 hinges use shared Triton kernels and preserve tiny score gaps."""

from typing import Any

import jax
import jax.numpy as jnp
import named_losses
import numpy as np
import pytest
from hinge_order_cases import rows
from libsdca import _jax_cuda
from libsdca import jax as J
from libsdca._jax_fusion import _precision_scope


@pytest.mark.parametrize("classes", [3, 129, 2049, 131073])
@pytest.mark.parametrize("power", [-29, -60])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.bfloat16])
@pytest.mark.parametrize("precision", [None, "float64"])
def test_jax_cuda_hinge_distinct_scores(
    classes: int, power: int, dtype: Any, precision: str | None, monkeypatch: pytest.MonkeyPatch
):
    if jax.default_backend() != "gpu":
        pytest.skip("needs JAX CUDA")
    accepted: list[tuple[str, bool]] = []
    ready = _jax_cuda._ready

    def observed(spec: _jax_cuda.Spec, x: jax.Array) -> bool:
        result = ready(spec, x)
        accepted.append((spec.family, result))
        return result

    monkeypatch.setattr(_jax_cuda, "_ready", observed)
    with _precision_scope(True):
        z, labels, expected = rows(classes, power)
        x, y = jnp.asarray(z, dtype=dtype), jnp.asarray(labels)
        for loss in ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta"):

            def f(v: jax.Array) -> jax.Array:
                return named_losses.loss(J, loss, v, y, precision=precision, reduction="sum")

            gradient = jax.jit(jax.grad(f))(x)
            np.testing.assert_array_equal(gradient, expected)
        assert ("hinge", True) in accepted, accepted
    jax.clear_caches()


def test_smooth_hinge_fp32_dispatch_guards_gamma_times_classes():
    route_ok = _jax_cuda._route(
        "topk_hinge_beta_smooth", 1, True, "float32", None, (2, 100), 1e35, 1.0
    )
    route_overflow = _jax_cuda._route(
        "topk_hinge_beta_smooth", 1, True, "float32", None, (2, 100), 5e37, 1.0
    )
    assert route_ok == ("hinge", (True, 1e35, True))
    assert route_overflow == ("hinge", (True, 5e37, False))
    torch = pytest.importorskip("torch")
    from libsdca import torch as T

    logits = torch.zeros((2, 100, 3), dtype=torch.float32)
    assert T._hinge_fp32(logits, None, 1e35, 100)
    assert not T._hinge_fp32(logits, None, 5e37, 100)
