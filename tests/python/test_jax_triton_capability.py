"""The optional CUDA launcher is selected by its actual named read/write reference contract."""

from typing import Any

import jax.numpy as jnp
import pytest


def test_supported_optional_jax_triton() -> None:
    pytest.importorskip("triton")
    transport = pytest.importorskip("jax_triton", minversion="0.5")
    from libsdca.triton._jax_transport import compatible

    assert compatible(transport.triton_call)


def test_incompatible_jax_triton_calls() -> None:
    pytest.importorskip("triton")
    from libsdca.triton._jax_transport import compatible

    def ignores_keywords(**kwargs: Any) -> list[Any]:
        return []

    def reads_only(**kwargs: Any) -> list[Any]:
        jnp.add(kwargs["source"], kwargs["destination"][...])
        return []

    def positional_only(a: Any, b: Any, /, **kwargs: Any) -> list[Any]:
        b[...] = a
        return []

    for call in (ignores_keywords, reads_only, positional_only):
        assert not compatible(call)


# One in-place plan per family whose launches bind the gradient over the scores (G = Z) next to
# other written arrays: label rows, narrow, register and streamed distillation, chunked top-k
# and multiclass hinges, proximal losses and multilabel rows.
_ALIASED = [
    ("cross_entropy", "bfloat16", 512, 10),
    ("cross_entropy", "float32", 64, 40000),
    ("topk_cross_entropy", "float32", 64, 1000),
    ("distillation", "bfloat16", 512, 10),
    ("distillation", "bfloat16", 512, 100),
    ("distillation", "float32", 64, 5000),
    ("topk_hinge", "float32", 16, 40000),
    ("multiclass_hinge", "bfloat16", 64, 40000),
    ("proximal_cross_entropy", "float32", 256, 100),
    ("proximal_distillation", "float32", 64, 5000),
    ("multilabel_cross_entropy", "bfloat16", 512, 10),
]


@pytest.mark.cuda
@pytest.mark.parametrize(("loss", "dtype", "n", "c"), _ALIASED)
def test_jax_triton_inplace_gradient(loss: str, dtype: str, n: int, c: int) -> None:
    """Under jax-triton a gradient written over temporary or donated logits equals the
    separate-buffer gradient, every coordinate (a G bound to Z is a full-size output)."""
    pytest.importorskip("triton")
    pytest.importorskip("jax_triton", minversion="0.5")
    import jax
    import libsdca.jax as lj
    import numpy as np
    from libsdca.triton import launch_jax

    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("needs a CUDA device")
    if launch_jax.available() is not None or launch_jax._jax_triton() is None:
        pytest.skip("this jax-triton is not the selected transport")
    assert launch_jax.route() == " through jax-triton"
    key = jax.random.PRNGKey
    z = (jax.random.normal(key(1), (n, c), jnp.float32) * 3).astype(dtype)
    if "distillation" in loss:
        t = (jax.random.normal(key(2), (n, c), jnp.float32) * 3).astype(dtype)
    elif loss == "multilabel_cross_entropy":
        t = jax.random.uniform(key(2), (n, c)) < 0.1
    else:
        t = jax.random.randint(key(2), (n,), 0, c)
    options = {
        "topk_cross_entropy": {"k": 2},
        "topk_hinge": {"k": 2},
        "distillation": {"temperature": 2.0},
        "proximal_cross_entropy": {"gamma": 0.5},
        "proximal_distillation": {"temperature": 2.0, "gamma": 0.5},
    }.get(loss, {})
    name = {
        "distillation": "distillation_loss",
        "proximal_distillation": "proximal_distillation_loss",
    }
    function = getattr(lj, name.get(loss, loss + ("_loss" if "hinge" in loss else "")))

    def f(z: Any, t: Any) -> Any:
        return function(z, t, **options)

    separate = jax.jit(jax.value_and_grad(f))(z, t)
    temporary = jax.jit(jax.value_and_grad(lambda z, t: f(z * jnp.ones((), z.dtype), t)))(z, t)
    donated = jax.jit(jax.value_and_grad(f), donate_argnums=0)(jnp.array(z, copy=True), t)
    for value, grad in (temporary, donated):
        np.testing.assert_array_equal(np.asarray(value), np.asarray(separate[0]))
        np.testing.assert_array_equal(
            np.asarray(grad, np.float32), np.asarray(separate[1], np.float32)
        )
