"""Gradients of probability targets of cross_entropy (_composed._targets, one formula for the
kernels of libsdca.torch and libsdca.jax and the composed losses): PyTorch's autograd, with
class weights, label smoothing and every reduction; the values and logits gradients are those
of a target without gradient."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import libsdca.jax as J  # noqa: E402
import libsdca.torch as L  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from jax import Array  # noqa: E402

jax.config.update("jax_enable_x64", True)

OPTIONS = [
    {},
    {"weight": True},
    {"label_smoothing": 0.2},
    {"weight": True, "label_smoothing": 0.3},
]
REDUCTIONS = ("mean", "sum", "none")


def _inputs(shape: tuple[int, ...] = (5, 4), seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=shape)
    t = rng.uniform(0.05, 1.0, size=shape)
    w = rng.uniform(0.2, 2.0, size=shape[1])
    return z, t, w


def _torch_grads(fn: Any, z: np.ndarray, t: np.ndarray, **options: Any) -> tuple[Any, ...]:
    zt = torch.tensor(z, requires_grad=True)
    tt = torch.tensor(t, requires_grad=True)
    value = fn(zt, tt, **options)
    value.sum().backward()
    assert zt.grad is not None and tt.grad is not None
    return value.detach().numpy(), zt.grad.numpy(), tt.grad.numpy()


def _options(case: dict, w: np.ndarray, reduction: str, convert: Any) -> dict:
    out: dict[str, Any] = {"reduction": reduction}
    if case.get("weight"):
        out["weight"] = convert(w)
    if "label_smoothing" in case:
        out["label_smoothing"] = case["label_smoothing"]
    return out


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("case", OPTIONS, ids=str)
def test_torch_targets_match_autograd(case: dict, reduction: str):
    z, t, w = _inputs()
    options = _options(case, w, reduction, torch.tensor)
    expected = _torch_grads(F.cross_entropy, z, t, **options)
    for backend in ("kernel", "composed"):
        found = _torch_grads(L.cross_entropy, z, t, backend=backend, **options)
        for a, b in zip(found, expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("case", OPTIONS, ids=str)
def test_torch_gradcheck(case: dict):
    z, t, w = _inputs((3, 4), seed=1)
    for reduction in REDUCTIONS:
        options = _options(case, w, reduction, torch.tensor)
        inputs = (torch.tensor(z, requires_grad=True), torch.tensor(t, requires_grad=True))
        assert torch.autograd.gradcheck(lambda z, t: L.cross_entropy(z, t, **options), inputs)


def test_torch_values_and_logit_gradients_do_not_change():
    z, t, w = _inputs((6, 5), seed=2)
    for options in ({}, {"weight": torch.tensor(w), "label_smoothing": 0.1}):
        with_target = _torch_grads(L.cross_entropy, z, t, **options)
        zt = torch.tensor(z, requires_grad=True)
        value = L.cross_entropy(zt, torch.tensor(t), **options)
        value.backward()
        assert torch.equal(value.detach(), torch.tensor(with_target[0]))
        assert zt.grad is not None and torch.equal(zt.grad, torch.tensor(with_target[1]))


def test_torch_class_dimension_and_dtypes():
    z, t, _ = _inputs((2, 4, 3), seed=3)  # classes on dim 1
    expected = _torch_grads(F.cross_entropy, z, t)
    np.testing.assert_allclose(_torch_grads(L.cross_entropy, z, t)[2], expected[2], rtol=1e-13)
    # float32 targets of float64 logits get a float32 gradient.
    zt = torch.tensor(z, requires_grad=True)
    tt = torch.tensor(t, dtype=torch.float32, requires_grad=True)
    L.cross_entropy(zt, tt).backward()
    assert tt.grad is not None and tt.grad.dtype == torch.float32
    np.testing.assert_allclose(tt.grad.numpy(), expected[2], rtol=1e-6)


def _jax_grads(fn: Any, z: np.ndarray, t: np.ndarray, **options: Any) -> tuple[Any, ...]:
    def total(z: Any, t: Any) -> Any:
        return fn(z, t, **options).sum()

    value = fn(jnp.asarray(z), jnp.asarray(t), **options)
    gz, gt = jax.grad(total, argnums=(0, 1))(jnp.asarray(z), jnp.asarray(t))
    return np.asarray(value), np.asarray(gz), np.asarray(gt)


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("case", OPTIONS, ids=str)
def test_jax_targets_match_torch(case: dict, reduction: str):
    z, t, w = _inputs(seed=4)
    expected = _torch_grads(F.cross_entropy, z, t, **_options(case, w, reduction, torch.tensor))
    options = _options(case, w, reduction, jnp.asarray)
    for backend in ("kernel", "composed"):
        found = _jax_grads(J.cross_entropy, z, t, backend=backend, **options)
        for a, b in zip(found, expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-15)
        # libsdca.torch and libsdca.jax agree.
        ours = _torch_grads(
            L.cross_entropy, z, t, backend=backend, **_options(case, w, reduction, torch.tensor)
        )
        np.testing.assert_allclose(found[2], ours[2], rtol=1e-13, atol=1e-15)


def test_jax_transformations():
    z, t, w = _inputs((4, 5), seed=5)
    weight = jnp.asarray(w)

    def total(z: Any, t: Any, eps: Any) -> Any:
        return J.cross_entropy(z, t, weight=weight, label_smoothing=eps)

    expected = _torch_grads(F.cross_entropy, z, t, weight=torch.tensor(w), label_smoothing=0.25)[2]
    # A traced label_smoothing, under jit.
    gt = jax.jit(jax.grad(total, argnums=1))(jnp.asarray(z), jnp.asarray(t), 0.25)
    np.testing.assert_allclose(np.asarray(gt), expected, rtol=1e-13)
    batched = jax.vmap(jax.grad(lambda z, t: J.cross_entropy(z, t), argnums=1))(
        jnp.stack([jnp.asarray(z)] * 3), jnp.stack([jnp.asarray(t)] * 3)
    )
    single = jax.grad(lambda z, t: J.cross_entropy(z, t), argnums=1)(jnp.asarray(z), jnp.asarray(t))
    np.testing.assert_allclose(np.asarray(batched), np.stack([np.asarray(single)] * 3), rtol=1e-14)


def test_masked_classes():
    z = np.array([[1.0, -np.inf, 0.5, -1.0], [0.3, 0.2, -np.inf, 2.0]])
    t = np.array([[0.7, 0.0, 0.3, 0.0], [0.1, 0.4, 0.0, 0.5]])
    expected = _torch_grads(F.cross_entropy, z, t, reduction="sum")[2]
    assert np.all(expected[np.isinf(z)] == np.inf)
    for backend in ("kernel", "composed"):
        gt = _torch_grads(L.cross_entropy, z, t, reduction="sum", backend=backend)[2]
        np.testing.assert_allclose(gt, expected, rtol=1e-13)
        gt = _jax_grads(J.cross_entropy, z, t, reduction="sum", backend=backend)[2]
        np.testing.assert_allclose(gt, expected, rtol=1e-13)


def _teachers(seed: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(5, 6)), rng.normal(size=(5, 6))


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("temperature", (1.0, 2.5))
def test_teachers_match_autograd(temperature: float, reduction: str):
    s, t = _teachers()

    def reference(a: Any, b: Any) -> Any:
        target = torch.softmax(b / temperature, 1)
        return F.cross_entropy(a / temperature, target, reduction=reduction)

    expected = _torch_grads(reference, s, t)
    options = {"temperature": temperature, "reduction": reduction}
    for backend in ("kernel", "composed"):
        found = _torch_grads(L.distillation_loss, s, t, backend=backend, **options)
        for a, b in zip(found, expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-15)
        found = _jax_grads(J.distillation_loss, s, t, backend=backend, **options)
        for a, b in zip(found, expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("backend", ("kernel", "composed"))
def test_teacher_gradcheck(backend: str):
    s, t = _teachers(seed=7)
    losses = (
        lambda a, b: L.distillation_loss(a, b, temperature=1.5, backend=backend),
        lambda a, b: L.proximal_distillation_loss(
            a, b, temperature=1.5, gamma=0.7, reduction="sum", backend=backend
        ),
    )
    for loss in losses:
        inputs = (torch.tensor(s, requires_grad=True), torch.tensor(t, requires_grad=True))
        assert torch.autograd.gradcheck(loss, inputs)


def test_proximal_teachers_agree():
    s, t = _teachers(seed=8)
    options = {"temperature": 2.0, "gamma": 0.5, "reduction": "none"}
    found = [
        _torch_grads(L.proximal_distillation_loss, s, t, backend=b, **options)[2]
        for b in ("kernel", "composed")
    ]
    found += [
        _jax_grads(J.proximal_distillation_loss, s, t, backend=b, **options)[2]
        for b in ("kernel", "composed")
    ]
    for gt in found[1:]:
        np.testing.assert_allclose(gt, found[0], rtol=1e-12, atol=1e-15)


def test_teacher_masked_classes():
    s, t = _teachers(seed=9)
    t[0, 2] = -np.inf  # the teacher masks a class: probability 0
    s[1, 3] = t[1, 3] = -np.inf  # both mask it: removed
    expected = _torch_grads(
        lambda a, b: F.cross_entropy(a, torch.softmax(b, 1), reduction="sum"), s, t
    )[2]
    for backend in ("kernel", "composed"):
        gt = _torch_grads(L.distillation_loss, s, t, reduction="sum", backend=backend)[2]
        assert gt[0, 2] == 0 and gt[1, 3] == 0
        np.testing.assert_allclose(gt[2:], expected[2:], rtol=1e-13, atol=1e-15)
        gj = _jax_grads(J.distillation_loss, s, t, reduction="sum", backend=backend)[2]
        np.testing.assert_allclose(gj, gt, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize(
    "loss", [L.cross_entropy, L.distillation_loss, L.proximal_distillation_loss]
)
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_kernel_target_only_backward(monkeypatch: pytest.MonkeyPatch, loss: Any, reduction: str):
    from libsdca import _torch_ops

    # The dispatcher runs the autograd Function when the optional extension is unavailable.
    monkeypatch.setattr(_torch_ops, "eager", lambda: None)
    z, t = _teachers()
    if loss is L.cross_entropy:
        t = torch.tensor(t).softmax(1).numpy()
    expected = _torch_grads(loss, z, t, reduction=reduction)[2]
    target = torch.tensor(t, requires_grad=True)
    value = loss(torch.tensor(z), target, reduction=reduction)
    (result,) = torch.autograd.grad(value.sum(), target)
    np.testing.assert_array_equal(result.numpy(), expected)


def test_teacher_gradients_leave_the_rest_unchanged():
    s, t = _teachers(seed=10)
    for loss in (L.distillation_loss, L.proximal_distillation_loss):
        with_teacher = _torch_grads(loss, s, t)
        zt = torch.tensor(s, requires_grad=True)
        value = loss(zt, torch.tensor(t))
        value.backward()
        assert torch.equal(value.detach(), torch.tensor(with_teacher[0]))
        assert zt.grad is not None and torch.equal(zt.grad, torch.tensor(with_teacher[1]))
    for jax_loss in (J.distillation_loss, J.proximal_distillation_loss):
        both = jax.grad(lambda a, b: jax_loss(a, b), argnums=(0, 1))(jnp.asarray(s), jnp.asarray(t))
        student = jax.grad(lambda a: jax_loss(a, jnp.asarray(t)))(jnp.asarray(s))
        assert np.array_equal(np.asarray(both[0]), np.asarray(student))


@pytest.mark.parametrize("classes", (7, 4097))
def test_composed_distillation_forward_and_mixed_derivatives(classes: int) -> None:
    rng = np.random.default_rng(141)
    student, teacher, ds, dt = (jnp.asarray(x) for x in rng.normal(size=(4, 2, classes)))
    temperature = 2.0

    def loss(z: Array, t: Array) -> Array:
        return J.distillation_loss(
            z, t, temperature=temperature, backend="composed", reduction="sum"
        )

    def reference(z: Array, t: Array) -> Array:
        p = jax.nn.softmax(t / temperature)
        return -(p * jax.nn.log_softmax(z / temperature)).sum()

    def derivatives(
        function: Callable[[Array, Array], Array],
    ) -> tuple[Array, Array, Array]:
        # Both mixed blocks remain nonzero at matching student and teacher logits.
        z = teacher
        _, direction = jax.jvp(function, (z, teacher), (ds, dt))
        _, student_mixed = jax.jvp(lambda t: jax.grad(function)(z, t), (teacher,), (dt,))
        _, teacher_mixed = jax.jvp(lambda x: jax.grad(function, argnums=1)(x, teacher), (z,), (ds,))
        return direction, student_mixed, teacher_mixed

    for found, target in zip(derivatives(loss), derivatives(reference), strict=True):
        np.testing.assert_allclose(found, target, rtol=2e-12, atol=2e-15)

    # Both differentiable arguments also support a batched pullback under an enclosing jit.
    batch = jax.jit(jax.vmap(jax.grad(loss, argnums=(0, 1))))
    expected = jax.vmap(jax.grad(reference, argnums=(0, 1)))(student, teacher)
    for found, target in zip(batch(student, teacher), expected, strict=True):
        np.testing.assert_allclose(found, target, rtol=2e-12, atol=2e-15)
