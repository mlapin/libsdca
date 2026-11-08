"""CUDA value-only distillation: aggregate tails, wide weighted sums and FP64 fallback."""

from typing import Any

import mpmath as mp
import pytest
import torch
from libsdca.torch import distillation_loss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _grouped_value(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> Any:
    """Independent two-group reference from the represented first/other scores."""
    s0, s1 = map(mp.mpf, student[:2].double().tolist())
    t0, t1 = map(mp.mpf, teacher[:2].double().tolist())
    gap = (s0 - s1) / temperature
    count = student.numel() - 1
    mass = count * mp.exp((t1 - t0) / temperature)
    return mp.log1p(count * mp.exp(-gap)) + (mass / (1 + mass)) * gap


def _assert_contract(actual: torch.Tensor, expected: Any) -> None:
    info = torch.finfo(actual.dtype)
    budget = 64 * mp.mpf(info.eps) * abs(expected) + mp.mpf(info.tiny) * info.eps
    assert abs(mp.mpf(actual.item()) - expected) <= budget


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_two_pass_keeps_tiny_loss_and_large_weighted_sum(dtype: torch.dtype):
    pytest.importorskip("triton")
    classes = 4097
    student = torch.full((3, classes), -110.0, dtype=dtype, device="cuda")
    teacher = torch.full_like(student, -10000.0)
    student[:, 0], teacher[:, 0] = 0, 0
    # A finite mean whose unnormalized FP32 sum would overflow.
    student[1, 1:] = -torch.finfo(dtype).max / 4
    teacher[1] = 0
    # Centering overflows in FP32 at T=1, but the corresponding teacher mass is zero.
    large = torch.finfo(dtype).max * 0.75
    student[2] = -large
    student[2, 0] = large
    actual = distillation_loss(student, teacher, 1.0, reduction="none")
    assert actual[0] > 0
    assert torch.isfinite(actual).all()
    with mp.workdps(110):
        for row in range(3):
            _assert_contract(actual[row], _grouped_value(student[row], teacher[row], 1.0))


def test_two_pass_propagates_nonfinite_rows():
    pytest.importorskip("triton")
    student = torch.zeros(4, 4097, device="cuda")
    teacher = student.clone()
    student[0, 0], student[1, -1] = float("nan"), float("inf")
    teacher[2, 0], teacher[3, -1] = float("nan"), -float("inf")
    assert torch.isnan(distillation_loss(student, teacher, 2.0, reduction="none")).all()


@pytest.mark.parametrize("dtype,temperature", [(torch.float64, 1.0), (torch.float32, 1e300)])
def test_value_only_keeps_float64_work_fallback(dtype: torch.dtype, temperature: float):
    pytest.importorskip("triton")
    student = torch.full((1, 4097), -torch.finfo(dtype).max / 4, dtype=dtype, device="cuda")
    student[0, 0] = 0
    teacher = torch.zeros_like(student)
    # Float64's unnormalized sum can overflow too, so normalize its terms before summing.
    actual = distillation_loss(student, teacher, temperature, reduction="none")
    assert torch.isfinite(actual).all()
    with mp.workdps(110):
        _assert_contract(actual[0], _grouped_value(student[0], teacher[0], temperature))
