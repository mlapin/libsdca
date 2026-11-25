"""float64 hinge rows whose score spread exceeds DBL_MAX with the label (or every positive)
far above the other classes: their margins of -inf (or near -DBL_MAX) contribute nothing, so
the loss is the limit 0 with a zero gradient in every interface (docs/API.md, Known limits)."""

from typing import Any

import libsdca
import named_losses
import numpy as np
import pytest

ROWS = [
    ([1.7e308, -1.7e308, 0.0, 3.0], 0),  # a margin of -inf beside large finite ones
    ([1.7e308, -1.7e308, -1.6e308, -1e308], 0),  # margins near -DBL_MAX whose sums overflow
    ([5.0, -1.7e308, 1.7e308, 2.0], 2),
]
LOSSES = [
    ("multiclass_hinge", {}),
    ("topk_hinge_alpha", {"k": 2}),
    ("topk_hinge_beta", {"k": 2}),
]


def _losses():
    for name, options in LOSSES:
        for gamma in (0.0, 1.0):
            yield name, {**options, "gamma": gamma}


def _mask_rows():
    z = np.array([[1.7e308, -1.7e308, 1.6e308, 0.0], [1.7e308, -1e308, 2.0, -1.7e308]])
    mask = np.array([[True, False, True, False], [True, False, False, False]])
    return z, mask


@pytest.mark.parametrize(("name", "options"), list(_losses()), ids=lambda v: str(v))
def test_numpy_gives_the_limit(name: str, options: dict):
    loss = named_losses.numpy_loss(name, **options)
    for z, label in ROWS:
        values, gradient = libsdca.loss_and_grad(np.array([z]), np.array([label]), loss)
        assert values[0] == 0.0, (z, values)
        assert not gradient.any() and np.isfinite(gradient).all(), (z, gradient)


SMOOTH_SPREAD = "[top-k] the smoothed multilabel hinge gives NaN at spreads near finfo.max"


@pytest.mark.parametrize(
    "gamma", (0.0, pytest.param(1.0, marks=pytest.mark.xfail(strict=True, reason=SMOOTH_SPREAD)))
)
def test_numpy_multilabel_hinge_gives_the_limit(gamma: float):
    z, mask = _mask_rows()
    values, gradient = libsdca.loss_and_grad(z, mask, libsdca.MultilabelHinge(gamma=gamma))
    np.testing.assert_array_equal(values, 0.0)
    assert not gradient.any() and np.isfinite(gradient).all()


@pytest.mark.parametrize(("name", "options"), list(_losses()), ids=lambda v: str(v))
def test_pytorch_and_jax_agree_with_numpy(name: str, options: dict):
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import libsdca.jax as lj
    import libsdca.torch as lt

    loss = named_losses.numpy_loss(name, **options)
    gamma = options["gamma"]
    k = options.get("k")
    variant = {"topk_hinge_alpha": "alpha", "topk_hinge_beta": "beta"}.get(name)

    def call(module: Any, x: Any, t: Any) -> Any:
        if variant is None:
            return module.multiclass_hinge_loss(x, t, gamma=gamma, reduction="sum")
        return module.topk_hinge_loss(x, t, k=k, variant=variant, gamma=gamma, reduction="sum")

    jax.config.update("jax_enable_x64", True)  # as test_masked_classes_jax
    for z, label in ROWS:
        _, expected = libsdca.loss_and_grad(np.array([z]), np.array([label]), loss)
        x = torch.tensor([z], dtype=torch.float64, requires_grad=True)
        value = call(lt, x, torch.tensor([label]))
        (gradient,) = torch.autograd.grad(value, x)
        assert value.item() == 0.0
        np.testing.assert_array_equal(gradient.numpy(), expected)
        jv, jg = jax.value_and_grad(lambda a: call(lj, a, jnp.asarray([label])))(
            jnp.asarray([z], dtype=jnp.float64)
        )
        assert float(jv) == 0.0
        np.testing.assert_array_equal(np.asarray(jg), expected)


def test_composed_hinges_have_zero_second_derivatives():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import libsdca.jax as lj
    import libsdca.torch as lt

    for name, options in (
        ("multiclass_hinge_loss", {}),
        ("topk_hinge_loss", {"k": 2}),
        ("multilabel_hinge_loss", {}),
    ):

        def loss(module: Any, z: Any, y: Any) -> Any:
            target = z == 0 if name == "multilabel_hinge_loss" else y
            return getattr(module, name)(z, target, backend="composed", **options)

        row = [[0.0, 1.0, 1.0]]
        x = torch.tensor(row, requires_grad=True)
        (g,) = torch.autograd.grad(loss(lt, x, torch.tensor([0])), x, create_graph=True)
        (h,) = torch.autograd.grad(g.sum(), x)
        torch.testing.assert_close(h, torch.zeros_like(x))
        hj = jax.hessian(lambda z: loss(lj, z, jnp.array([0])))(jnp.array(row))
        assert (np.asarray(hj) == 0).all()


@pytest.mark.parametrize("k", [2, 3, 5])
def test_composed_alpha_preserves_the_sign_before_averaging(k: int):
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import libsdca.jax as lj
    import libsdca.torch as lt

    jax.config.update("jax_enable_x64", True)
    big, tiny = np.finfo(np.float64).max, 2.0**-80
    # The large selected scores cancel before the unit offset. The other
    # two rows have exactly opposite margins much smaller than one ulp at 1.
    rows = [[0.0, big, -big, *([0.0] * (k - 2))], [1.0, *([tiny] * k)], [1.0, *([-tiny] * k)]]
    labels = [0, 0, 0]
    expected = np.array([1.0, tiny, 0.0])
    active = [-1.0, *([1.0 / k] * k)]
    gradient = np.array([active, active, [0.0] * (k + 1)])

    def loss(module: Any, scores: Any, targets: Any) -> Any:
        return module.topk_hinge_loss(
            scores, targets, k=k, variant="alpha", reduction="none", backend="composed"
        )

    x = torch.tensor(rows, dtype=torch.float64, requires_grad=True)
    vt = loss(lt, x, torch.tensor(labels))
    (gt,) = torch.autograd.grad(vt.sum(), x)
    xj = jnp.asarray(rows)
    vj = jax.jit(lambda z: loss(lj, z, jnp.asarray(labels)))(xj)
    gj = jax.jit(jax.grad(lambda z: loss(lj, z, jnp.asarray(labels)).sum()))(xj)
    for value, grad in ((vt.detach().numpy(), gt.numpy()), (vj, gj)):
        np.testing.assert_array_equal(value, expected)
        np.testing.assert_allclose(grad, gradient, rtol=2e-15, atol=0)
