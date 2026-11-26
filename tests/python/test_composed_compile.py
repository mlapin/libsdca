"""The composed losses (backend="composed") compile as one graph: torch.compile(fullgraph=True)
gives their eager losses and gradients, with the root solves of the proximal losses in one
traced loop when PyTorch exports the higher-order operator."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
from libsdca import torch as L
from libsdca._composed import xp_torch

B: dict[str, Any] = {"backend": "composed", "reduction": "none"}
# name: (loss of logits and target, target kind)
CASES: dict[str, tuple[Callable[..., torch.Tensor], str]] = {
    "cross_entropy": (lambda z, t: L.cross_entropy(z, t, **B), "labels"),
    "topk_cross_entropy": (lambda z, t: L.topk_cross_entropy(z, t, k=3, **B), "labels"),
    "multiclass_hinge": (lambda z, t: L.multiclass_hinge_loss(z, t, gamma=1.0, **B), "labels"),
    "topk_hinge_alpha": (
        lambda z, t: L.topk_hinge_loss(z, t, k=3, variant="alpha", gamma=1.0, **B),
        "labels",
    ),
    "topk_hinge_beta": (lambda z, t: L.topk_hinge_loss(z, t, k=3, gamma=0.5, **B), "labels"),
    "multilabel_hinge": (lambda z, t: L.multilabel_hinge_loss(z, t, gamma=1.0, **B), "mask"),
    "multilabel_cross_entropy": (lambda z, t: L.multilabel_cross_entropy(z, t, **B), "mask"),
    "proximal_cross_entropy": (
        lambda z, t: L.proximal_cross_entropy(z, t, gamma=0.5, **B),
        "labels",
    ),
    "distillation": (lambda z, t: L.distillation_loss(z, t, temperature=2.0, **B), "teacher"),
    "proximal_distillation": (
        lambda z, t: L.proximal_distillation_loss(z, t, temperature=0.5, gamma=2.0, **B),
        "teacher",
    ),
}


def rows(kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits of random rows, a masked class, ties and a large margin, and their target."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(6, 7, generator=g, dtype=torch.float64) * 3
    z[1, 2] = -math.inf
    z[2] = 1.0
    z[3, 0] = 1e8
    if kind == "labels":
        t = torch.tensor([4, 0, 3, 0, 6, 1])
    elif kind == "mask":
        t = torch.rand(6, 7, generator=g) < 0.4
        t[:, 0], t[1, 2] = True, False
    else:
        t = torch.randn(6, 7, generator=g, dtype=torch.float64)
        t[1, 2] = -math.inf
    return z, t


def loss_and_grad(fn: Callable[..., torch.Tensor], z: torch.Tensor, t: torch.Tensor) -> Any:
    x = z.clone().requires_grad_(True)
    value = fn(x, t)
    (grad,) = torch.autograd.grad(value.sum(), x)
    return value.detach(), grad


@pytest.mark.skipif(
    not xp_torch._TRACED_LOOPS, reason="this PyTorch installation unrolls the root solves"
)
@pytest.mark.parametrize("name", sorted(CASES))
def test_one_graph(name: str):
    """aot_eager runs the traced graph on eager kernels: the same losses and gradients."""
    fn, kind = CASES[name]
    z, t = rows(kind)
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
    for got, want in zip(loss_and_grad(compiled, z, t), loss_and_grad(fn, z, t), strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.skipif(
    not xp_torch._TRACED_LOOPS, reason="this PyTorch installation unrolls the root solves"
)
@pytest.mark.parametrize("name", ["proximal_cross_entropy", "proximal_distillation"])
def test_fixed_count(name: str, monkeypatch: pytest.MonkeyPatch):
    """Off the CPU the compiled solves run every root step on a CPU count (no condition read
    from the device): finished rows stay fixed, so the results are the eager ones."""
    monkeypatch.setattr(xp_torch, "_host_condition", lambda device: False)
    fn, kind = CASES[name]
    z, t = rows(kind)
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
    for got, want in zip(loss_and_grad(compiled, z, t), loss_and_grad(fn, z, t), strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_no_recompile():
    """A second call on other data reuses the graph: the frontend's caches are not read while
    compiling."""
    fn, kind = CASES["proximal_cross_entropy"]
    z, t = rows(kind)
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
    loss_and_grad(compiled, z, t)
    with torch._dynamo.config.patch(error_on_recompile=True):
        loss_and_grad(compiled, z * 2 + 1, t)


@pytest.mark.slow
@pytest.mark.skipif(
    not xp_torch._TRACED_LOOPS, reason="this PyTorch installation unrolls the root solves"
)
@pytest.mark.parametrize("name", ["proximal_cross_entropy", "proximal_distillation"])
def test_inductor(name: str):
    fn, kind = CASES[name]
    z, t = rows(kind)
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)
    for got, want in zip(loss_and_grad(compiled, z, t), loss_and_grad(fn, z, t), strict=True):
        torch.testing.assert_close(got, want, rtol=1e-13, atol=1e-15)


@pytest.mark.skipif(not xp_torch._TRACED_LOOPS, reason="requires tensor control flow")
@pytest.mark.parametrize("shape", [(3,), (1,), (2, 1), (2, 3, 1), (1, 3, 1), (0, 1), (0, 3)])
def test_traced_loop_row_layout(shape: tuple[int, ...]) -> None:
    count = torch.tensor(0)
    z = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    column = len(shape) > 1 and shape[-1] == 1
    if column:
        # The size-one axis has a leading-axis stride in this view.
        z = z.squeeze(-1).unsqueeze(0).movedim(0, -1)

    def fn(i: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        def more(state: tuple[torch.Tensor, ...]) -> torch.Tensor:
            return state[0] < 2

        def body(state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            value = state[1] + 1
            if column:
                value = torch.stack((value.squeeze(-1),), dim=-1)
            return state[0] + 1, value

        return xp_torch.while_loop(more, body, (i, x), 4)[1]

    compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(fn(count, z), z + 2, rtol=0, atol=0)
    torch.testing.assert_close(compiled(count, z), z + 2, rtol=0, atol=0)


@pytest.mark.skipif(not xp_torch._TRACED_LOOPS, reason="requires tensor control flow")
def test_proximal_compile_noncontiguous_scores() -> None:
    fn, _ = CASES["proximal_cross_entropy"]
    z = torch.tensor([[1.0, 0.5], [2.0, 0.0], [-1.0, -1.0]]).T.requires_grad_()
    assert not z.is_contiguous()
    labels = torch.tensor([0, 1])
    seed = torch.tensor([1.5, -0.25])
    value = fn(z, labels)
    (gradient,) = torch.autograd.grad(value, z, seed)
    compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
    actual = compiled(z, labels)
    (actual_gradient,) = torch.autograd.grad(actual, z, seed)
    torch.testing.assert_close(actual, value, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, gradient, rtol=0, atol=0)
