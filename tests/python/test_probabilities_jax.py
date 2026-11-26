"""libsdca.jax.cross_entropy of probability targets: the CPU kernel against the mpmath reference
and jax.numpy, under jit, vmap and higher derivatives."""

import re
from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
import libsdca.jax as J
import numpy as np
import pytest
from libsdca._frontend import FIRST_ORDER
from probability_references import Family, families, gradient_ulps, loss_ulps, reference

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dtype", ["float64", "float32", "bfloat16"])
@pytest.mark.parametrize("family", list(families()), ids=lambda f: f[0])
def test_kernel_accuracy_against_mpmath(family: Family, dtype: str):
    """As the PyTorch layer's kernel (test_probabilities_torch), under jit."""
    name, z, t, options = family
    kind = jnp.dtype(dtype)
    x, p = jnp.asarray(z).astype(kind), jnp.asarray(t).astype(kind)
    w = options.get("weight")
    w = None if w is None else jnp.asarray(w).astype(kind)
    eps = options.get("label_smoothing", 0.0)
    want, want_gradient, terms = reference(
        np.asarray(x, np.float64),
        np.asarray(p, np.float64),
        eps,
        None if w is None else np.asarray(w, np.float64),
    )

    def f(a: jax.Array) -> jax.Array:
        return J.cross_entropy(a, p, weight=w, label_smoothing=eps, reduction="none")

    values, pullback = jax.vjp(jax.jit(f), x)
    (gradient,) = pullback(jnp.ones_like(values))
    g = np.asarray(gradient.astype(jnp.float32) if dtype == "bfloat16" else gradient, np.float64)
    value_dtype = "float64" if dtype == "float64" else "float32"
    assert loss_ulps(np.asarray(values, np.float64), want, value_dtype) <= 8
    assert gradient_ulps(g, want_gradient, terms, dtype) <= 8
    if name != "softmax":
        largest = np.abs(want_gradient).max(1, keepdims=True)
        assert gradient_ulps(g, want_gradient, largest, dtype) <= 8


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_kernel_matches_jax_numpy(reduction: str):
    rng = np.random.default_rng(5)
    z = jnp.asarray(rng.normal(size=(2, 6, 3)))
    t = jax.nn.softmax(jnp.asarray(rng.normal(size=(2, 6, 3))), axis=1)
    w = jnp.asarray(rng.uniform(0.5, 1.5, 6))
    kw: dict[str, Any] = dict(weight=w, label_smoothing=0.1, reduction=reduction, axis=1)

    def loss(backend: J.Backend) -> Callable[[jax.Array], jax.Array]:
        return lambda a: J.cross_entropy(a, t, backend=backend, **kw).sum()

    for transform in (lambda f: f, jax.grad, lambda f: jax.jit(jax.grad(f))):
        np.testing.assert_allclose(
            transform(loss("auto"))(z), transform(loss("composed"))(z), rtol=1e-12, atol=1e-14
        )


def test_kernels_are_first_order():
    """Differentiating a kernel gradient raises PyTorch's error (one text), forward mode JAX's;
    composed takes both."""
    z, t = jnp.zeros((2, 3)), jnp.full((2, 3), 1 / 3)
    for second in (jax.hessian, lambda f: jax.grad(lambda a: jax.grad(f)(a).sum())):
        with pytest.raises(RuntimeError, match=re.escape(FIRST_ORDER)):
            second(lambda a: J.cross_entropy(a, t).sum())(z)
    with pytest.raises(TypeError, match="forward-mode"):
        jax.jacfwd(lambda a: J.cross_entropy(a, t))(z)
    composed = jax.hessian(lambda a: J.cross_entropy(a, t, backend="composed"))(z)
    assert composed.shape == (2, 3, 2, 3)


def test_backends():
    z, t = jnp.zeros((2, 3)), jnp.full((2, 3), 1 / 3)
    for backend in ("auto", "kernel"):
        assert J.explain_backend(J.cross_entropy, z, t, backend=backend)[0] == "kernel"
    assert J.explain_backend(J.cross_entropy, z, t, backend="composed")[0] == "composed"


def test_vmap_and_traced_smoothing():
    rng = np.random.default_rng(6)
    z, t = (
        jnp.asarray(rng.normal(size=(3, 4, 5))),
        jax.nn.softmax(jnp.asarray(rng.normal(size=(3, 4, 5)))),
    )
    batched = jax.vmap(lambda a, b: J.cross_entropy(a, b))(z, t)
    each = jnp.stack([J.cross_entropy(z[i], t[i], backend="composed") for i in range(3)])
    np.testing.assert_allclose(batched, each, rtol=1e-13)
    traced = jax.jit(lambda e: J.cross_entropy(z[0], t[0], label_smoothing=e))(0.25)
    np.testing.assert_allclose(
        traced, J.cross_entropy(z[0], t[0], label_smoothing=0.25), rtol=1e-13
    )
    kernel = jax.jit(lambda e: J.cross_entropy(z[0], t[0], label_smoothing=e, backend="kernel"))
    np.testing.assert_allclose(kernel(0.25), traced, rtol=1e-13)


def test_invalid_values():
    """A negative target or a row of masked classes makes its row NaN, eager or traced."""
    z, t = jnp.zeros((2, 3)), jnp.full((2, 3), 1 / 3)
    backend: Literal["auto", "composed"]
    for backend in ("auto", "composed"):
        loss = jax.jit(lambda a, p: J.cross_entropy(a, p, reduction="none", backend=backend))
        for f in (loss, lambda a, p: J.cross_entropy(a, p, reduction="none", backend=backend)):
            value = f(z, t.at[0, 1].set(-0.5))
            assert np.isnan(value[0]) and np.isfinite(value[1])
            value = f(z.at[1].set(-jnp.inf), t)
            assert np.isfinite(value[0]) and np.isnan(value[1])
