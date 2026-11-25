"""At k = 1 the top-k losses take the dedicated kernels: topk_cross_entropy is cross_entropy
and both top-k hinges are multiclass_hinge with the same gamma. Each interface must return
the dedicated loss's exact values and gradients and report the same kernel."""

from __future__ import annotations

import libsdca
import numpy as np
import pytest
from libsdca import _frontend as _f

# Ordinary rows plus a tied maximum, a label tied with the largest competitor and a large
# hinge, so the nonsmooth subgradient and the smoothed projection both see their edge cases.
ROWS, CLASSES = 37, 133
GAMMAS = [0.0, 0.5]


def _inputs(dtype: str = "float32") -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    z = rng.standard_normal((ROWS, CLASSES)).astype(np.float32)
    y = rng.integers(0, CLASSES, ROWS)
    z[1, :4] = [2.0, 2.0, 0.0, 2.0]
    y[1] = 2
    z[2, y[2]] = z[2].max()
    z[3, y[3]] = -40.0
    return z, y


def _pairs() -> list[tuple[str, dict, str, dict]]:
    """(top-k loss, its options at k = 1, dedicated loss, its options)."""
    pairs: list[tuple[str, dict, str, dict]] = [
        ("topk_cross_entropy", {"k": 1}, "cross_entropy", {})
    ]
    for gamma in GAMMAS:
        for variant in ("alpha", "beta"):
            pairs.append(
                (
                    "topk_hinge_loss",
                    {"k": 1, "variant": variant, "gamma": gamma},
                    "multiclass_hinge_loss",
                    {"gamma": gamma},
                )
            )
    return pairs


PAIRS = _pairs()
IDS = [f"{a}-{o.get('variant', '')}-{o.get('gamma', '')}" for a, o, _, _ in PAIRS]


def test_kernel_kind():
    kinds = _f.KINDS
    assert _f.kernel_kind(kinds["topk_cross_entropy"], 1) == kinds["cross_entropy"]
    assert _f.kernel_kind(kinds["topk_hinge_alpha"], 1) == kinds["multiclass_hinge"]
    assert _f.kernel_kind(kinds["topk_hinge_beta"], 1) == kinds["multiclass_hinge"]
    for name in ("topk_cross_entropy", "topk_hinge_alpha", "topk_hinge_beta"):
        assert _f.kernel_kind(kinds[name], 2) == kinds[name]
    for name, kind in kinds.items():
        if not name.startswith("topk_"):
            assert _f.kernel_kind(kind, 1) == kind


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("variant", ["alpha", "beta"])
def test_numpy_hinge(dtype: str, gamma: float, variant: str):
    z, y = _inputs()
    z = z.astype(dtype)
    top = libsdca.loss_and_grad(z, y, libsdca.TopKHinge(k=1, variant=variant, gamma=gamma))
    own = libsdca.loss_and_grad(z, y, libsdca.MulticlassHinge(gamma=gamma))
    for a, b in zip(top, own, strict=False):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_numpy_cross_entropy(dtype: str):
    z, y = _inputs()
    z = z.astype(dtype)
    top = libsdca.loss_and_grad(z, y, libsdca.TopKCrossEntropy(k=1))
    own = libsdca.loss_and_grad(z, y, libsdca.CrossEntropy())
    for a, b in zip(top, own, strict=False):
        np.testing.assert_array_equal(a, b)


def _torch_devices() -> list[str]:
    torch = pytest.importorskip("torch")
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float64"])
@pytest.mark.parametrize("reduction", ["mean", "none"])
@pytest.mark.parametrize("topk,top_options,own,own_options", PAIRS, ids=IDS)
def test_torch(
    dtype: str, reduction: str, topk: str, top_options: dict, own: str, own_options: dict
):
    torch = pytest.importorskip("torch")
    import libsdca.torch as L

    z0, y0 = _inputs()
    for device in _torch_devices():
        z = torch.tensor(z0, device=device).to(getattr(torch, dtype))
        y = torch.tensor(y0, device=device)
        outputs, routes = [], []
        for name, options in ((topk, top_options), (own, own_options)):
            loss = getattr(L, name)
            routes.append(L.explain_backend(loss, z, y, reduction=reduction, **options))
            x = z.clone().requires_grad_()
            value = loss(x, y, reduction=reduction, **options)
            (grad,) = torch.autograd.grad(value.sum(), x)
            outputs.append((value, grad))
        assert routes[0] == routes[1], device
        for a, b in zip(*outputs, strict=False):
            assert torch.equal(a, b), (device, dtype)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("topk,top_options,own,own_options", PAIRS, ids=IDS)
def test_jax(dtype: str, topk: str, top_options: dict, own: str, own_options: dict):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import libsdca.jax as J

    z0, y0 = _inputs()
    z, y = jnp.asarray(z0).astype(getattr(jnp, dtype)), jnp.asarray(y0)
    outputs, routes = [], []
    for name, options in ((topk, top_options), (own, own_options)):
        loss = getattr(J, name)
        routes.append(J.explain_backend(loss, z, y, **options))
        value, grad = jax.jit(jax.value_and_grad(lambda x: loss(x, y, **options)))(z)
        outputs.append(
            (np.asarray(value.astype(jnp.float32)), np.asarray(grad.astype(jnp.float32)))
        )
    assert routes[0] == routes[1]
    for a, b in zip(*outputs, strict=False):
        np.testing.assert_array_equal(a, b)
