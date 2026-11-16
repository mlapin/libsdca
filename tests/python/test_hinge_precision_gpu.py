"""Selectable hinge work must retain finite-input range and explicit double precision."""

import pytest
import torch
from libsdca import _cuda_launch
from libsdca.torch import sdca_loss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("classes", [3, 129, 2049])
@pytest.mark.parametrize("gamma", [0.0, 0.5])
@pytest.mark.parametrize("launcher", ["on", "off"])
def test_cuda_hinge_fp32_extreme_rows(
    classes: int, gamma: float, launcher: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("SDCA_CUDA_LAUNCHER", launcher)
    _cuda_launch.status.cache_clear()
    maximum = torch.finfo(torch.float32).max
    z = torch.full((19, classes), -maximum, device="cuda")
    z[:, :3] = torch.tensor([maximum, -maximum, 0.0], device="cuda")
    z[1, 2] = maximum
    y = torch.ones(19, dtype=torch.long, device="cuda")
    x = z.clone().requires_grad_()
    value = sdca_loss(x, y, "topk_hinge_alpha", gamma=gamma, reduction="none")
    gradient = torch.autograd.grad(value.sum(), x)[0]
    ref = z.double().requires_grad_()
    exact = sdca_loss(
        ref, y, "topk_hinge_alpha", gamma=gamma, reduction="none", precision="float64"
    )
    expected = torch.autograd.grad(exact.sum(), ref)[0]
    torch.testing.assert_close(value, exact.float(), rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected.float(), rtol=0, atol=0)
    assert torch.isfinite(gradient).all()
    _cuda_launch.status.cache_clear()


@pytest.mark.parametrize("classes", [3, 129, 2049, 131073])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("launcher", ["on", "off"])
@pytest.mark.parametrize("precision", [None, "float64"])
@pytest.mark.parametrize("power", [-29, -60])
def test_cuda_hinge_distinct_scores_survive_margin_rounding(
    classes: int,
    dtype: torch.dtype,
    launcher: str,
    precision: str | None,
    power: int,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SDCA_CUDA_LAUNCHER", launcher)
    _cuda_launch.status.cache_clear()
    x = torch.full((19, classes), -100.0, dtype=dtype, device="cuda")
    x[:, 0], x[:, 1], x[:, -1] = 0.0, 2.0**power, 2.0 ** (power + 1)
    x.requires_grad_()
    y = torch.zeros(19, dtype=torch.long, device="cuda")
    value = sdca_loss(x, y, "multiclass_hinge", reduction="none", precision=precision)
    gradient = torch.autograd.grad(value.sum(), x)[0]
    expected = torch.zeros_like(x)
    expected[:, 0], expected[:, -1] = -1.0, 1.0
    torch.testing.assert_close(value, torch.ones_like(value), rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
    _cuda_launch.status.cache_clear()


@pytest.mark.parametrize("classes", [3, 129, 2049, 131073])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_nonsmooth_hinge_scalar_double_matches_explicit_work(classes: int, dtype: torch.dtype):
    # The raw-score maximum is exact in storage precision. Only its margin needs
    # double arithmetic, so fp32 and fp64 requests can share the same result.
    z = torch.full((5, classes), -100.0, dtype=dtype, device="cuda")
    largest = torch.finfo(dtype).max
    z[:, 0] = torch.tensor([largest, -largest, 2**24, 0, 1], device="cuda", dtype=dtype)
    z[:, -1] = torch.tensor([largest, largest, 2**24, 2.0**-60, 0], device="cuda", dtype=dtype)
    y = torch.zeros(5, device="cuda", dtype=torch.long)
    results = []
    for precision in (None, "float32", "float64"):
        x = z.clone().requires_grad_()
        value = sdca_loss(x, y, "multiclass_hinge", precision=precision, reduction="none")
        results.append((value, torch.autograd.grad(value.sum(), x)[0]))
    for value, gradient in results[1:]:
        torch.testing.assert_close(value, results[0][0], rtol=0, atol=0)
        torch.testing.assert_close(gradient, results[0][1], rtol=0, atol=0)
    reference = z.double().requires_grad_()
    value = sdca_loss(reference, y, "multiclass_hinge", reduction="none")
    gradient = torch.autograd.grad(value.sum(), reference)[0]
    torch.testing.assert_close(results[0][0], value.float(), rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], gradient.to(dtype), rtol=0, atol=0)


def test_cuda_smooth_hinge_large_gamma_times_classes_promotes():
    # gamma fits normal float32, while C * gamma overflows the Michelot sum in float32.
    z = torch.zeros((2, 100), dtype=torch.float32, device="cuda")
    z[:, 1:11] = 2e37
    z[:, 11:] = -2e37
    y = torch.zeros(2, dtype=torch.long, device="cuda")
    x = z.clone().requires_grad_()
    value = sdca_loss(x, y, "topk_hinge_beta", gamma=5e37, reduction="none")
    gradient = torch.autograd.grad(value.sum(), x)[0]
    ref = z.double().requires_grad_()
    exact = sdca_loss(ref, y, "topk_hinge_beta", gamma=5e37, reduction="none", precision="float64")
    expected = torch.autograd.grad(exact.sum(), ref)[0]
    torch.testing.assert_close(value, exact.float(), rtol=1e-6, atol=0)
    torch.testing.assert_close(gradient, expected.float(), rtol=1e-6, atol=0)
