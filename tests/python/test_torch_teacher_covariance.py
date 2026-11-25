"""The eager FP32 CPU teacher VJP, its retained inputs, and its transform fallbacks."""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any

import pytest
import torch
from libsdca import _torch_ops
from libsdca import torch as losses


@pytest.fixture
def native_teacher() -> ModuleType:
    extension = _torch_ops.eager()
    if extension is None:
        pytest.skip("the optional PyTorch CPU extension is unavailable")
    return extension


def _grads(
    z: torch.Tensor,
    t: torch.Tensor,
    upstream: torch.Tensor,
    reduction: losses.Reduction = "none",
    **options: Any,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    value = losses.distillation_loss(z, t, reduction=reduction, **options)
    inputs = (z, t) if z.requires_grad else (t,)
    return value.detach(), torch.autograd.grad(value, inputs, upstream)


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_teacher_recovers_small_moment_and_scales_before_rounding(
    native_teacher: ModuleType, reduction: losses.Reduction, joint: bool
) -> None:
    # The zero score's gradient is 2^-104 in either ordering of the first
    # row. Multiplying the last row's unit FP32 derivative would erase it.
    u = 2.0**-100
    z = torch.tensor([[1, 0, -1, u], [u, -1, 0, 1], [0, 1, -1, 1]], requires_grad=joint)
    t = torch.tensor([[0.0] * 4, [0.0] * 4, [-110.0, 0.0, -110.0, -math.inf]], requires_grad=True)
    up = torch.tensor([2.5, -3.25, 2.0**30]) if reduction == "none" else torch.tensor(-(2.0**30))
    value, gradients = _grads(z, t, up, reduction, backend="kernel")
    factors = up.expand(3).double() / (3 if reduction == "mean" else 1)
    exp = math.exp(-110)
    norm = 1 + 2 * exp
    # Center on the winner: z_i - 1 is -1, 0, -2 on the three supported classes.
    expected = (u / 4 - z.detach().double()) / 4
    expected[2] = (
        torch.tensor(
            [exp * (1 - 3 * exp / norm), -3 * exp / norm, exp * (2 - 3 * exp / norm), 0],
            dtype=torch.float64,
        )
        / norm
    )
    torch.testing.assert_close(gradients[-1], (expected * factors[:, None]).float(), rtol=0, atol=0)
    if joint:
        v = losses.distillation_loss(z, t.detach(), reduction=reduction, backend="kernel")
        torch.testing.assert_close(value, v, rtol=0, atol=0)
        torch.testing.assert_close(gradients[0], torch.autograd.grad(v, z, up)[0], rtol=0, atol=0)


def test_teacher_exceptional_rows_keep_composed_ieee_result(native_teacher: ModuleType) -> None:
    z = torch.tensor([[0.0, 1, -1], [1, -math.inf, 2], [2, math.inf, 1], [4, 5, 6], [1, 0, 1]])
    t = torch.tensor([[-110.0, 0, -110], [2, 0, 1], [0, 0, 0], [math.nan, 1, 0], [0, 1, 2]])
    t.requires_grad_()
    up = torch.tensor([2.0**30, -2.5, 0.0, 1.0, math.inf])
    _, (observed,) = _grads(z, t, up, backend="kernel")
    _, (ref,) = _grads(z[1:], t[1:], up[1:], backend="composed")
    torch.testing.assert_close(observed[1:], ref, rtol=0, atol=0, equal_nan=True)
    assert observed[0, 0] > 0 and observed[0, 1] < 0 and observed[0, 2] > 0


def test_teacher_retains_inputs_and_dynamic_signed_cotangents(native_teacher: ModuleType) -> None:
    z = torch.tensor([[1.0, 0, -1], [0, 2, 1]], requires_grad=True)
    t = torch.tensor([[0.0, 1, 2], [-1, 2, 0]], requires_grad=True)
    value = losses.distillation_loss(z, t, reduction="none")
    up = torch.tensor([1.5, -3.0])
    saved = z.detach().clone(), t.detach().clone()
    a, b = torch.autograd.grad(value, (z, t), up, retain_graph=True)
    c, d = torch.autograd.grad(value, (z, t), -up, retain_graph=True)
    for actual, expected in ((a, -c), (b, -d), (z, saved[0]), (t, saved[1])):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with torch.no_grad():
        t.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(value, t, up)


def test_teacher_transform_and_higher_derivative_use_previous_pullback(
    native_teacher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args: Any) -> None:
        pytest.fail("the ordinary native VJP was selected for a transform")

    monkeypatch.setattr(native_teacher, "teacher_backward", unavailable)
    z = torch.tensor([[0.0, 1, -1], [0, 0, 1]])
    t = torch.tensor([[0.0, 0, 1], [1, 0, 1]], requires_grad=True)

    def f(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return losses.distillation_loss(a, b, reduction="none", backend="kernel")

    ref = losses.distillation_loss(z, t, reduction="sum", backend="composed")
    expected = torch.autograd.grad(ref, t)[0]
    wrapped = torch.func.grad(lambda b: f(z, b).sum())(t)
    torch.testing.assert_close(wrapped, expected)
    vals = f(z, t)
    weights = torch.tensor([[1.0, -2.0], [-3.0, 0.5]])
    (batched,) = torch.autograd.grad(vals, t, weights, is_grads_batched=True, retain_graph=True)
    torch.testing.assert_close(batched, weights[:, :, None] * expected)
    up = torch.ones_like(vals, requires_grad=True)
    (with_graph,) = torch.autograd.grad(vals, t, up, create_graph=True)
    with pytest.raises(RuntimeError, match="double backward is not supported"):
        torch.autograd.grad(with_graph.sum(), up)


@pytest.mark.parametrize(
    "options",
    [{"precision": "float64"}, {"temperature": 2.0**-160}, {"temperature": 2.0**160}],
)
def test_teacher_alternate_precision_and_extreme_temperature_follow_previous_path(
    native_teacher: ModuleType, monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]
) -> None:
    def unavailable(*args: Any) -> None:
        pytest.fail(
            "the ordinary native VJP was selected outside its precision or temperature range"
        )

    monkeypatch.setattr(native_teacher, "teacher_backward", unavailable)
    z = torch.tensor([[0.0, 0.5, -1]])
    t = torch.tensor([[0.0, 1, 0]], requires_grad=True)
    up = torch.tensor(-2.5)
    _, (observed,) = _grads(z, t, up, "sum", backend="kernel", **options)
    monkeypatch.setattr(_torch_ops, "eager", lambda: None)
    _, (expected,) = _grads(z, t, up, "sum", backend="kernel", **options)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0, equal_nan=True)


def test_compiled_teacher_uses_previous_pullback(
    native_teacher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args: Any) -> None:
        pytest.fail("the ordinary native VJP was selected while compiling")

    monkeypatch.setattr(native_teacher, "teacher_backward", unavailable)
    z = torch.tensor([[1.0, 0.5, -1], [0, 2, -1]])
    t = torch.tensor([[0.0, 1, 0], [-0.3, 1, 2.5]], requires_grad=True)

    def f(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return losses.distillation_loss(z, t, temperature=0.75, backend="kernel")

    compiled = torch.compile(f, backend="eager", fullgraph=True)
    observed = torch.autograd.grad(compiled(z, t), t)[0]
    ref = losses.distillation_loss(z, t, temperature=0.75, backend="composed")
    torch.testing.assert_close(observed, torch.autograd.grad(ref, t)[0])


def test_teacher_batch_matches_independent_rows_with_signed_and_invalid_upstreams(
    native_teacher: ModuleType,
) -> None:
    # Row ranges can be assigned to different CPU workers. A row with an
    # exceptional teacher or upstream must only replace its own derivative.
    student = torch.zeros(4, 64)
    teacher = torch.full((4, 64), -math.inf)
    student[:, :4] = torch.tensor([[1.0, 0, -1, 2.0**-100], [0, 1, -1, 1]] * 2)
    teacher[:, :4] = torch.tensor([[0.0] * 4, [-110.0, 0, -110.0, -math.inf]] * 2)
    teacher[2, 0] = math.nan
    upstream = torch.tensor([1.5, -(2.0**30), -0.0, math.inf])
    references = []
    for i in range(4):
        t = teacher[i : i + 1].detach().requires_grad_()
        _, (g,) = _grads(student[i : i + 1], t, upstream[i : i + 1], backend="kernel")
        references.append(g)
    idx = (torch.arange(2048) * 3) % 4
    t = teacher[idx].requires_grad_()
    _, (grad,) = _grads(student[idx], t, upstream[idx], backend="kernel")
    torch.testing.assert_close(grad, torch.cat(references)[idx], rtol=0, atol=0, equal_nan=True)
