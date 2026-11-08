"""Bounded CUDA expm1 accuracy and distillation's +/-2 simple-branch boundary."""

import importlib
import math

import mpmath as mp
import pytest
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    pytest.skip("Triton required", allow_module_level=True)

kernels = importlib.import_module("libsdca._triton_distillation")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@triton.jit
def _evaluate(X, Y, N: tl.constexpr, B: tl.constexpr):  # pyright: ignore[reportMissingParameterType]  # Triton JIT
    j = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + j, j < N, other=0.0)
    tl.store(Y + j, kernels._bounded_expm1(x), j < N)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bounded_expm1_preserves_accuracy_and_tiny_arguments(dtype: torch.dtype):
    info = torch.finfo(dtype)
    device = torch.device("cuda")
    anchors = torch.tensor(
        [
            -2.0,
            -1.0,
            -0.5,
            -info.tiny,
            -info.tiny * info.eps,
            -0.0,
            0.0,
            info.tiny * info.eps,
            info.tiny,
            0.5,
            1.0,
            2.0,
        ],
        dtype=dtype,
        device=device,
    )
    boundaries = torch.cat(
        [
            anchors,
            torch.nextafter(anchors, torch.full_like(anchors, -math.inf)),
            torch.nextafter(anchors, torch.full_like(anchors, math.inf)),
        ]
    )
    boundaries = boundaries[(boundaries >= -2) & (boundaries <= 2)]
    generator = torch.Generator(device=device).manual_seed(1729)
    random = 4 * torch.rand(65536, dtype=dtype, device=device, generator=generator) - 2
    near_zero = torch.linspace(-1e-7, 1e-7, 1025, dtype=dtype, device=device)
    x = torch.cat([boundaries, random, near_zero])
    actual = torch.empty_like(x)
    _evaluate[(triton.cdiv(x.numel(), 256),)](
        x,
        actual,
        x.numel(),
        256,
        enable_reflect_ftz=False,
    )
    expected = torch.expm1(x.double())
    budget = 8 * info.eps * expected.abs() + info.tiny * info.eps
    error = (actual.double() - expected).abs()
    assert torch.isfinite(actual).all()
    assert (error <= budget).all(), (float(error.max()), float((error / budget).max()))
    tiny = x.abs() <= info.tiny
    assert torch.equal(actual[tiny], x[tiny])
    assert torch.equal(torch.signbit(actual[x == 0]), torch.signbit(x[x == 0]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_distillation_simple_branch_boundary(dtype: torch.dtype, temperature: float):
    # Uniform teachers and equal-size student groups make centered dn exactly +/-a.
    # The three rows sit immediately below, at, and above the |dn| <= 2 threshold.
    classes = 8192
    two = torch.tensor(2.0, dtype=dtype, device="cuda")
    a = torch.stack(
        [
            torch.nextafter(two, torch.full_like(two, -math.inf)),
            two,
            torch.nextafter(two, torch.full_like(two, math.inf)),
        ]
    )
    x = a[:, None].expand(-1, classes).clone() * temperature
    x[:, classes // 2 :] *= -1
    teacher = torch.zeros_like(x)
    value, stats = kernels.forward(x, teacher, temperature, False, "none")
    gradient = kernels.backward(
        x, teacher, stats, torch.ones_like(value), temperature, False, "none"
    )
    n = x.shape[0]
    assert (stats[12 * n : 13 * n] < 0).tolist() == [True, True, False]
    with mp.workdps(100):
        represented = [mp.mpf(float(v)) for v in a.double().cpu()]
        expected_value = torch.tensor(
            [float(mp.log(classes) + mp.log(mp.cosh(v))) for v in represented],
            dtype=torch.float64,
            device="cuda",
        )
        positive = torch.tensor(
            [float(mp.tanh(v) / (classes * temperature)) for v in represented],
            dtype=torch.float64,
            device="cuda",
        )
    expected_gradient = positive[:, None].expand(-1, classes).clone()
    expected_gradient[:, classes // 2 :] *= -1
    value_eps = torch.finfo(value.dtype).eps
    gradient_eps = torch.finfo(dtype).eps
    torch.testing.assert_close(value.double(), expected_value, rtol=64 * value_eps, atol=0)
    torch.testing.assert_close(gradient.double(), expected_gradient, rtol=64 * gradient_eps, atol=0)
