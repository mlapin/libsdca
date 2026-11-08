"""Chunked linear CE retains full-range reductions and global weighted normalization."""

import math

import mpmath as mp
import pytest
import torch
from libsdca.torch import linear_cross_entropy

CASES = ("aggregate", "individual", "weight", "tiny_weight", "tail", "weighted_tail")


def data(case: str):
    h, w, y, cw = [1.0] * 4, [0.0, 1e308], [0] * 4, None
    reduction = "mean"
    if case == "individual":
        h, w, y = [1.0, 0.0, 0.0, 0.0], [1e308, -1e308], [1] * 4
    elif case == "weight":
        h, w, y, cw = [1.0, 2.0, 0.0, -1.0], [0.0, 1.0], [0, 1, -100, 0], [1e308, 5e307]
    elif case == "tiny_weight":
        y, cw = [0, 1, 1, 1], [1e-100, 1e308]
    elif case in ("tail", "weighted_tail"):
        w, reduction = [0.0, -746.0 if case == "tail" else -1000.0], "sum"
        if case == "weighted_tail":
            cw = [1e308, 1e308]
    return h, w, y, cw, reduction


def oracle(h: list[float], w: list[float], y: list[int], cw: list[float] | None, reduction: str):
    with mp.workdps(120):
        weights = [mp.mpf(0 if label == -100 else 1 if cw is None else cw[label]) for label in y]
        divisor = sum(weights) if reduction == "mean" else mp.mpf(1)
        value, grad = mp.mpf(0), []
        for hidden, label, weight in zip(h, y, weights, strict=True):
            label = max(label, 0)
            gap = mp.mpf(hidden) * (mp.mpf(w[1 - label]) - mp.mpf(w[label]))
            tail = mp.exp(-abs(gap)) if abs(gap) < 10000 else mp.mpf(0)
            loss = max(gap, 0) + mp.log1p(tail)
            mass = 1 / (1 + tail) if gap >= 0 else tail / (1 + tail)
            value += loss * weight / divisor
            # The API rounds each logit derivative before its GEMM.
            g = float(mass * weight / divisor)
            grad.append([-g, g] if label == 0 else [g, -g])
        return float(value), torch.tensor(grad, dtype=torch.float64)


@pytest.mark.parametrize(
    "device,backend",
    [("cpu", "auto"), ("cpu", "native"), ("cpu", "torch"), ("cuda", "auto"), ("cuda", "torch")],
)
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("case", CASES)
def test_linear_full_range(case: str, recompute: bool, device: str, backend: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    h, w, y, cw, reduction = data(case)
    expected, g = oracle(h, w, y, cw, reduction)
    hidden = torch.tensor(h, device=device, dtype=torch.float64)[:, None].requires_grad_()
    weight = torch.tensor(w, device=device, dtype=torch.float64)[:, None].requires_grad_()
    target = torch.tensor(y, device=device, dtype=torch.int64)
    class_weight = None if cw is None else torch.tensor(cw, device=device, dtype=torch.float64)
    value = linear_cross_entropy(
        hidden,
        weight,
        target,
        class_weight=class_weight,
        reduction=reduction,
        backend=backend,
        recompute=recompute,
        chunk_bytes=80,
    )
    assert math.isfinite(value.item())
    assert abs(value.item() - expected) <= 1024 * torch.finfo(torch.float64).eps * abs(expected)
    actual = torch.autograd.grad(value, (hidden, weight))
    references = (g @ weight.detach().cpu(), g.T @ hidden.detach().cpu())
    for a, e in zip(actual, references, strict=True):
        assert torch.allclose(a.cpu(), e, rtol=1e-12, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("recompute", [False, True])
def test_float32_logits_double_weights(device: str, recompute: bool):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    h = torch.ones((4, 1), device=device, requires_grad=True)
    w = torch.tensor([[0.0], [1.0]], device=device, requires_grad=True)
    y = torch.zeros(4, device=device, dtype=torch.int64)
    cw = torch.full((2,), 1e308, device=device, dtype=torch.float64)
    value = linear_cross_entropy(h, w, y, class_weight=cw, recompute=recompute, chunk_bytes=8)
    assert abs(value.item() - math.log1p(math.e)) < 2e-7
    gh, gw = torch.autograd.grad(value, (h, w))
    p = 1 / (1 + math.exp(-1))
    assert torch.allclose(gh, torch.full_like(gh, p / 4), atol=0, rtol=2e-6)
    assert torch.allclose(gw, torch.tensor([[-p], [p]], device=device), atol=0, rtol=2e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_recomputed_global_weight_scale(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    h = torch.ones((2, 1), dtype=torch.float64, device=device, requires_grad=True)
    w = torch.tensor([[0.0], [1.0]], dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tensor([0, 1], device=device)
    cw = torch.tensor([1e-100, 1e308], dtype=torch.float64, device=device)
    value = linear_cross_entropy(h, w, y, class_weight=cw, recompute=True, chunk_bytes=16)
    (gh,) = torch.autograd.grad(
        value, (h,), torch.tensor(1e308, dtype=torch.float64, device=device)
    )
    p = 1 / (1 + math.exp(-1))
    expected = torch.tensor([[p * 1e-100], [(p - 1) * 1e308]], device=device, dtype=torch.float64)
    assert torch.allclose(gh, expected, atol=0, rtol=3e-13)


@pytest.mark.parametrize("backend", ["auto", "torch"])
@pytest.mark.parametrize("bad", [-1.0, math.nan])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_full_range_invalid_cuda_weight(backend: str, bad: float, reduction: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    h = torch.ones((4, 1), device="cuda", dtype=torch.float64)
    w = torch.zeros((3, 1), device="cuda", dtype=torch.float64)
    y = torch.zeros(4, device="cuda", dtype=torch.int64)
    cw = torch.tensor([1.0, 1.0, bad], device="cuda", dtype=torch.float64)
    value = linear_cross_entropy(
        h, w, y, class_weight=cw, backend=backend, reduction=reduction, chunk_bytes=24
    )
    assert value.isnan().all()
