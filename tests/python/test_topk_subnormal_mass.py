"""Top-k entropy keeps target mass when every capped competitor rounds to zero."""

import contextlib
import math

import mpmath as mp
import pytest
import torch
from libsdca.torch import Backend, topk_cross_entropy


@pytest.mark.parametrize(
    "device,backend",
    [("cpu", "auto"), ("cpu", "composed"), ("cuda", "auto"), ("cuda", "composed")],
)
@pytest.mark.parametrize("classes", [3, 4, 65, 4097])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_topk_target_subnormal(device: str, backend: Backend, classes: int, dtype: torch.dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    info = torch.finfo(dtype)
    quantum = info.tiny * info.eps
    gap = math.log(classes - 1) - math.log(quantum) - math.log(0.75)
    x = torch.zeros((1, classes), device=device, dtype=dtype)
    x[0, 0] = gap
    x.requires_grad_()
    y = torch.zeros(1, device=device, dtype=torch.int64)
    context = contextlib.nullcontext()
    with context:
        value = topk_cross_entropy(x, y, k=2, backend=backend)
        (gradient,) = torch.autograd.grad(value, (x,))
    with mp.workdps(100):
        tail = (classes - 1) * mp.exp(-mp.mpf(x[0, 0].item()))
        expected_loss = float(mp.log1p(tail))
        expected_mass = float(tail / (1 + tail))
    expected_loss = torch.tensor(expected_loss, dtype=dtype).item()
    expected_mass = torch.tensor(expected_mass, dtype=dtype).item()
    assert expected_mass > 0
    assert value.item() == expected_loss
    assert gradient[0, 0].item() == -expected_mass
    assert torch.count_nonzero(gradient[0, 1:]) == 0


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize(
    "device,backend",
    [("cpu", "auto"), ("cpu", "composed"), ("cuda", "auto"), ("cuda", "composed")],
)
def test_all_capped_mean_of_scores_near_finfo_max(
    device: str, backend: Backend, dtype: torch.dtype
):
    # k = C - 1 caps every competitor: the loss is the mean of the score differences, whose
    # terms (1.5 S here) overflow although the mean (5 S / 6) does not.
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    big = torch.finfo(dtype).max
    x = torch.tensor([[-big / 2, big, big, -big]], device=device, dtype=dtype, requires_grad=True)
    y = torch.zeros(1, device=device, dtype=torch.int64)
    context = contextlib.nullcontext()
    with context:
        value = topk_cross_entropy(x, y, k=3, backend=backend)
        (gradient,) = torch.autograd.grad(value, (x,))
    eps = torch.finfo(dtype).eps
    assert value.item() == pytest.approx(5 * (big / 6), rel=64 * eps)
    expected = torch.tensor([[-1, 1 / 3, 1 / 3, 1 / 3]], dtype=torch.float64)
    torch.testing.assert_close(gradient.cpu().double(), expected, rtol=64 * eps, atol=0)


@pytest.mark.parametrize(
    "device,backend",
    [("cpu", "auto"), ("cpu", "composed"), ("cuda", "auto"), ("cuda", "composed")],
)
def test_all_capped_mean_beyond_double_double_limit(device: str, backend: Backend):
    # The exact mean margin is 1/4. Losing the unit term gives log(5) in documented
    # cases on the composed path. The CPU and CUDA kernels preserve it.
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    h, m = 2.0**200, 2.0**100
    x = torch.tensor([[m / 4, h, m, 1, -h]], device=device, dtype=torch.float64)
    y = torch.zeros(1, device=device, dtype=torch.int64)
    value = topk_cross_entropy(x, y, k=4, backend=backend).item()
    exact = math.log1p(4 * math.exp(0.25))
    if backend == "auto":
        assert value == pytest.approx(exact, rel=2e-15)
    else:
        assert abs(value - exact) <= exact - math.log(5) + 1e-15


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_composed_all_capped_mask_at_finite_extremes(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    big = torch.finfo(torch.float64).max
    x = torch.tensor(
        [[-math.inf, -big, -big, -big], [-math.inf, -big, -math.inf, -big]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    y = torch.zeros(2, dtype=torch.int64, device=device)
    value = topk_cross_entropy(x, y, k=3, backend="composed", reduction="none")
    (gradient,) = torch.autograd.grad(value.sum(), x)
    torch.testing.assert_close(value, x.new_tensor([math.inf, 0.0]))
    torch.testing.assert_close(
        gradient, x.new_tensor([[-1.0, 1 / 3, 1 / 3, 1 / 3], [0.0] * 4]), atol=0, rtol=0
    )
