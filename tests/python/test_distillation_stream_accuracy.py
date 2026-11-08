"""Wide CUDA distillation rows against an independent 120-digit definition."""

import numpy as np
import pytest
import torch
from libsdca.torch import distillation_loss
from test_distillation_references import reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("classes", [4097, 32000])
def test_wide_rows_against_mpmath(dtype: torch.dtype, classes: int):
    rng = np.random.default_rng(1927)
    student = 3 * rng.normal(size=(5, classes))
    noise = rng.normal(size=student.shape)
    teacher = student + noise * np.array(
        [[0], [1e-6 if dtype == torch.float32 else 1e-12], [0.5], [4], [0]]
    )
    # Exactly represented common shifts also have identical distributions.
    student[4] = np.round(student[4] * 4) / 4
    teacher[4] = student[4] + 3
    x = torch.tensor(student, dtype=dtype, device="cuda", requires_grad=True)
    t = torch.tensor(teacher, dtype=dtype, device="cuda")
    expected_value, expected_gradient = reference(x.detach().cpu().numpy(), t.cpu().numpy(), 2.0)
    value = distillation_loss(x, t, 2.0, reduction="none")
    (gradient,) = torch.autograd.grad(value.sum(), x)
    eps = torch.finfo(dtype).eps
    actual_value = value.detach().double().cpu().numpy()
    actual_gradient = gradient.double().cpu().numpy()
    np.testing.assert_allclose(actual_value, expected_value, rtol=64 * eps, atol=0)
    error = np.abs(actual_gradient - expected_gradient)
    scale = np.abs(expected_gradient).max(axis=1, keepdims=True)
    assert (error <= 64 * eps * scale + 1e-60).all(), [
        (i, float(error[i].max()), float(scale[i, 0])) for i in range(5)
    ]
    assert not torch.count_nonzero(gradient[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("large_temperature", [False, True])
def test_streamed_overflow_rescue(dtype: torch.dtype, large_temperature: bool):
    huge = 3e38 if dtype == torch.float32 else 1e308
    x = torch.full((1, 4097), -huge, dtype=dtype, device="cuda")
    x[0, 0] = huge
    teacher = -x
    x.requires_grad_()
    temperature = huge if large_temperature else 4.0
    expected_value, expected_gradient = reference(
        x.detach().cpu().numpy(), teacher.cpu().numpy(), temperature
    )
    value = distillation_loss(x, teacher, temperature, reduction="none")
    (gradient,) = torch.autograd.grad(value.sum(), x)
    eps = torch.finfo(dtype).eps
    np.testing.assert_allclose(value.detach().cpu(), expected_value, rtol=64 * eps, atol=0)
    tiny = float(np.nextafter(np.float32(0), np.float32(1))) if dtype == torch.float32 else 5e-324
    np.testing.assert_allclose(gradient.cpu(), expected_gradient, rtol=64 * eps, atol=tiny)
