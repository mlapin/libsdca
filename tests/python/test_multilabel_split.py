"""Wide-batch mask entropy: common and cancellation-sensitive rows share one batch."""

import functools
import math

import pytest
import torch
from conformance.references import Block, Loss
from conformance.references import reference as shared_reference
from libsdca.torch import Reduction, multilabel_cross_entropy

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


def test_wide_positive_near_softmax_crossing_matches_centered_row() -> None:
    classes, positives = 32000, 31885
    scores = torch.zeros((1, classes), device="cuda", dtype=torch.float32)
    scores[0, 0] = 0.004196599591523409
    scores[0, 3] = 3
    mask = torch.arange(classes, device=scores.device)[None, :] < positives

    def first_gradient(x: torch.Tensor, members: torch.Tensor) -> torch.Tensor:
        x.requires_grad_()
        value = multilabel_cross_entropy(x, members, reduction="sum")
        (gradient,) = torch.autograd.grad(value, x)
        return gradient[0, 0]

    # The extra nonmember contributes exp(-128), below a binary64 ULP of this
    # partition, and takes the centered path; an unguarded tile misses by many
    # float32 ulps.
    centered = first_gradient(
        torch.nn.functional.pad(scores, (0, 1), value=-128),
        torch.nn.functional.pad(mask, (0, 1), value=False),
    )
    actual = first_gradient(scores, mask)
    assert centered < 0
    torch.testing.assert_close(actual, centered, rtol=0, atol=0)


@functools.cache
def reference(dtype: torch.dtype, classes: int, case: int):
    """The row of a pattern at its width (conformance.references): value and the gradient
    entries of its first, second and every other class."""
    scores, flags = PATTERNS[case]
    z = torch.tensor(scores, dtype=dtype).double().tolist()
    counts = [1, 1, classes - 2]
    row = tuple(Block(v, n, f) for v, n, f in zip(z, counts, flags, strict=True))
    ref = shared_reference(Loss("multilabel_cross_entropy"), row)
    return float(ref.value), [float(g) for g in ref.gradient]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("classes", [62501, 65537])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_split_mask_oracle(dtype: torch.dtype, classes: int, reduction: Reduction):
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
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
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
