"""Wide-batch mask entropy: common and cancellation-sensitive rows share one batch."""

import functools
import math

import mpmath as mp
import pytest
import torch
from libsdca.torch import multilabel_cross_entropy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
PATTERNS = (
    ((-1.0, 0.0, 0.0), (True, False, False)),
    ((5.0, 5.0, -20.0), (True, True, False)),
    ((3.0, 3.0 + 1e-6, -100.0), (True, True, False)),
    ((0.0, 0.0, 0.0), (True, True, True)),
    ((0.0, 0.0, 0.0), (False, True, True)),
    ((-10000.0, 10000.0, 0.0), (True, True, False)),
    ((100.0, 0.0, 0.0), (True, False, False)),
)


@functools.cache
def reference(dtype: torch.dtype, classes: int, case: int):
    scores, flags = PATTERNS[case]
    represented = torch.tensor(scores, dtype=dtype).double().tolist()
    with mp.workdps(120):
        z = [mp.mpf(v) for v in represented]
        counts = [1, 1, classes - 2]
        top = max(z)
        e = [mp.exp(v - top) for v in z]
        den = sum(n * x for n, x in zip(counts, e, strict=True))
        p = sum(n for n, pos in zip(counts, flags, strict=True) if pos)
        value = (
            sum(
                n * (mp.log(den) + top - v)
                for n, v, pos in zip(counts, z, flags, strict=True)
                if pos
            )
            / p
        )
        gradient = [
            float(x / den - (mp.mpf(1) / p if pos else 0)) for x, pos in zip(e, flags, strict=True)
        ]
        return float(value), gradient


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("classes", [62501, 65537])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_split_mask_oracle(dtype: torch.dtype, classes: int, reduction: str):
    n = 2048
    cases = [i % len(PATTERNS) for i in range(n)]
    scores = torch.tensor([PATTERNS[i][0] for i in cases], dtype=dtype, device="cuda")
    flags = torch.tensor([PATTERNS[i][1] for i in cases], device="cuda")
    z = scores[:, 2, None].expand(n, classes).clone()
    z[:, :2] = scores[:, :2]
    mask = flags[:, 2, None].expand(n, classes).clone()
    mask[:, :2] = flags[:, :2]
    losses, gradients = zip(*(reference(dtype, classes, i) for i in cases), strict=True)
    z.requires_grad_()
    value = multilabel_cross_entropy(z, mask, reduction=reduction)
    (gradient,) = torch.autograd.grad(value.sum(), (z,))
    expected = torch.tensor(losses, dtype=torch.float64, device="cuda")
    if reduction != "none":
        expected = expected.mean() if reduction == "mean" else expected.sum()
    eps = torch.finfo(torch.float32).eps
    quantum = torch.finfo(torch.float32).tiny * eps
    assert torch.all((value.double() - expected).abs() <= 8 * eps * expected.abs() + quantum)
    ref = torch.tensor(gradients, dtype=torch.float64, device="cuda")
    if reduction == "mean":
        ref = ref / n
    actual = gradient[:, [0, 1, classes - 1]].double()
    info = torch.finfo(dtype)
    budget = (info.eps + 8 * eps) * ref.abs().amax(1, keepdim=True) + info.tiny * info.eps
    assert torch.all((actual - ref).abs() <= budget)


@pytest.mark.parametrize("bad", ["mask", "nan", "inf"])
def test_split_mask_invalid(bad: str):
    z = torch.zeros((2048, 62501), device="cuda")
    mask = torch.zeros_like(z, dtype=torch.bool)
    mask[:, 0] = True
    if bad == "mask":
        mask[-1] = False
    else:
        z[-1, -1] = math.nan if bad == "nan" else math.inf
    value = multilabel_cross_entropy(z, mask, reduction="none")
    assert value[:-1].isfinite().all()
    assert value[-1].isnan()


def test_split_mask_graph_reclassifies_rows():
    """Replay changes which residual kernel owns a row, including the saved backward."""
    z = torch.zeros((2048, 62501), device="cuda", requires_grad=True)
    mask = torch.zeros_like(z, dtype=torch.bool)
    mask[:, 0] = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            value = multilabel_cross_entropy(z, mask)
            torch.autograd.grad(value, z)
            del value
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        value = multilabel_cross_entropy(z, mask)
        (gradient,) = torch.autograd.grad(value, z)
    for positive_top in (False, True):
        with torch.no_grad():
            z[:, 0] = 20.0 if positive_top else -3.0
        expected_input = z.detach().clone().requires_grad_()
        expected_value = multilabel_cross_entropy(expected_input, mask)
        (expected_gradient,) = torch.autograd.grad(expected_value, expected_input)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
