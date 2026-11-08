"""CUDA linear CE preserves mixed-dtype GEMM accuracy across chunks and fallback APIs."""

import contextlib
from unittest import mock

import pytest
import torch
from libsdca import _linear_cross_entropy as lce
from libsdca.torch import linear_cross_entropy
from torch.nn import functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_gradient_rounding(dtype: torch.dtype):
    """Fusion preserves multiplication before conversion, including exceptional values."""
    pytest.importorskip("triton")
    from libsdca._triton_linear import scaled

    torch.manual_seed(1781)
    x = torch.randn(131073, device="cuda")
    x[:13] = torch.tensor(
        [
            0.0,
            -0.0,
            2**-149,
            -(2**-149),
            2**-126,
            -(2**-126),
            1e-30,
            -1e-30,
            1e30,
            -1e30,
            float("inf"),
            -float("inf"),
            float("nan"),
        ],
        device="cuda",
    )
    for factor in (
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.37,
        2**-50,
        2**50,
        1e-20,
        1e20,
        float("inf"),
        float("nan"),
    ):
        up = torch.tensor(factor, device="cuda")
        actual, expected = scaled(x, up, dtype), (x * up).to(dtype)
        same = (actual.view(torch.int16) == expected.view(torch.int16)) | (
            actual.isnan() & expected.isnan()
        )
        assert bool(same.all()), (dtype, factor)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_saved_gradient_multiple_upstreams(dtype: torch.dtype):
    """Retaining the graph must preserve FP32 saved gradients between scalings."""
    torch.manual_seed(1782)
    h = torch.randn(37, 13, device="cuda", dtype=dtype).requires_grad_()
    w = (torch.randn(53, 13, device="cuda", dtype=dtype) * 0.1).requires_grad_()
    y = torch.randint(53, (37,), device="cuda")
    value = linear_cross_entropy(h, w, y)
    for factor in (0.37, -2.0, 0.0, 1.0):
        up = torch.tensor(factor, device="cuda")
        actual = torch.autograd.grad(value, (h, w), up, retain_graph=True)
        with mock.patch.object(lce, "_fused_scale", return_value=None):
            reference = linear_cross_entropy(h, w, y)
            expected = torch.autograd.grad(reference, (h, w), up)
        for a, e in zip(actual, expected, strict=True):
            assert torch.equal(a, e), (dtype, factor)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fallback", [False, True])
def test_mixed_gemm_accumulation(dtype: torch.dtype, fallback: bool):
    torch.manual_seed(191)
    total = torch.full((17, 13), float("nan"), device="cuda")
    expected = torch.zeros_like(total, dtype=torch.float64)
    magnitude = torch.zeros_like(expected)
    context = (
        mock.patch.multiple(lce, _MM_DTYPE_OUT=False, _ADDMM_DTYPE_OUT=False)
        if fallback
        else contextlib.nullcontext()
    )
    with context:
        for step in range(4):
            # Transposed A is the weight-gradient GEMM's actual layout.
            a = torch.randn(7, 17, device="cuda", dtype=dtype).T
            b = torch.randn(7, 13, device="cuda", dtype=dtype)
            lce._accumulate(total, a, b, step == 0)
            expected += a.double() @ b.double()
            magnitude += a.double().abs() @ b.double().abs()
    assert torch.all(
        (total.double() - expected).abs() <= 16 * torch.finfo(torch.float32).eps * magnitude
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("recompute", [False, True])
def test_cuda_linear_chunks(dtype: torch.dtype, reduction: str, recompute: bool):
    torch.manual_seed(731)
    h = torch.randn(37, 11, device="cuda", dtype=dtype)
    w = torch.randn(53, 11, device="cuda", dtype=dtype) * 0.1
    bias = torch.randn(53, device="cuda", dtype=dtype) * 0.1
    target = torch.randint(53, (37,), device="cuda")
    target[::7] = -100
    class_weight = (torch.rand(53, device="cuda") + 0.5).to(dtype)

    def evaluate(reference: bool):
        leaves = [v.detach().clone().requires_grad_() for v in (h, w, bias)]
        if reference:
            value = F.cross_entropy(
                F.linear(*leaves).float(), target, weight=class_weight.float(), reduction=reduction
            )
        else:
            value = linear_cross_entropy(
                leaves[0],
                leaves[1],
                target,
                leaves[2],
                class_weight=class_weight,
                reduction=reduction,
                recompute=recompute,
                chunk_bytes=7 * 53 * dtype.itemsize,
            )
        upstream = torch.full_like(value, 2.0**-8)
        return (value.detach(), *torch.autograd.grad(value, leaves, upstream))

    for actual, expected in zip(evaluate(False), evaluate(True), strict=True):
        eps = torch.finfo(dtype).eps
        quantum = torch.finfo(dtype).tiny * eps
        assert torch.max((actual.double() - expected.double()).abs()) <= (
            8 * eps * expected.double().abs().max() + quantum
        )


_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def _class_weight(weights: str | None, dtype: torch.dtype, classes: int) -> torch.Tensor | None:
    """None, or class weights in [0.5, 1.5) with class 0 weighing 0, in dtype ("same") or
    float64 (which selects the full-range path)."""
    if weights is None:
        return None
    w = torch.rand(classes, device="cuda", dtype=torch.float64) + 0.5
    w[0] = 0
    return w.to(dtype if weights == "same" else torch.float64)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("weights", [None, "same", "float64"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("recompute", [False, True])
def test_invalid_row_makes_every_reduced_gradient_nan(
    dtype: torch.dtype, weights: str | None, reduction: str, recompute: bool
):
    """One out-of-range target in the last of several chunks, beside an ignored row and a
    valid row of class weight 0: NaN loss and every gradient NaN, saved or recomputed, as
    cross_entropy on CUDA; the per-row reduction keeps the other rows."""
    torch.manual_seed(733)
    h = torch.randn(21, 11, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(53, 11, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(53, device="cuda", dtype=dtype, requires_grad=True)
    target = torch.randint(1, 53, (21,), device="cuda")
    target[0], target[1], target[-1] = -100, 0, 53
    options = {"class_weight": _class_weight(weights, dtype, 53), "chunk_bytes": 7 * 53 * 4}
    value = linear_cross_entropy(
        h, w, target, b, reduction=reduction, recompute=recompute, **options
    )
    grads = torch.autograd.grad(value, (h, w, b))
    assert value.isnan().item()
    assert all(g.isnan().all().item() for g in grads)
    rows = linear_cross_entropy(h, w, target, b, reduction="none", **options)
    (grad_hidden,) = torch.autograd.grad(rows[:-1].sum(), (h,))
    assert grad_hidden[:-1].isfinite().all().item()


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("weights", ["same", "float64"])
@pytest.mark.parametrize("backend", ["auto", "torch"])
@pytest.mark.parametrize("recompute", [False, True])
def test_weightless_mean_is_undefined(
    dtype: torch.dtype, weights: str, backend: str, recompute: bool
):
    """The mean over rows of total class weight 0 has NaN gradients, as F.linear +
    F.cross_entropy; ignored rows keep zero gradients."""
    h = torch.tensor([[1.0, 2.0], [3.0, -1.0]], device="cuda", dtype=dtype, requires_grad=True)
    w = torch.tensor([[2.0, 1.0], [1.0, 1.0]], device="cuda", dtype=dtype, requires_grad=True)
    y = torch.tensor([0, -100], device="cuda")
    weight = torch.tensor([0.0, 1.0], device="cuda", dtype=torch.float64)
    class_weight = weight.to(dtype if weights == "same" else torch.float64)
    value = linear_cross_entropy(
        h, w, y, class_weight=class_weight, backend=backend, recompute=recompute
    )
    got = torch.autograd.grad(value, (h, w))
    h64, w64 = (t.detach().double().requires_grad_() for t in (h, w))
    expected = torch.autograd.grad(
        F.cross_entropy(F.linear(h64, w64), y, weight=weight), (h64, w64)
    )
    for g, e in zip(got, expected, strict=True):
        torch.testing.assert_close(g.double(), e, equal_nan=True)
    assert got[0][0].isnan().all().item() and (got[0][1] == 0).all().item()
