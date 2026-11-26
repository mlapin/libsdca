"""Wide CUDA distillation rows against the mpmath references (conformance.references)."""

import numpy as np
import pytest
import torch
from conformance.references import Loss, evaluate
from libsdca.torch import distillation_loss


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
    expected_value, expected_gradient = evaluate(
        Loss("distillation", temperature=2.0), x.detach().cpu().numpy(), teacher=t.cpu().numpy()
    )
    value = distillation_loss(x, t, temperature=2.0, reduction="none")
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
    expected_value, expected_gradient = evaluate(
        Loss("distillation", temperature=temperature),
        x.detach().cpu().numpy(),
        teacher=teacher.cpu().numpy(),
    )
    value = distillation_loss(x, teacher, temperature=temperature, reduction="none")
    (gradient,) = torch.autograd.grad(value.sum(), x)
    eps = torch.finfo(dtype).eps
    np.testing.assert_allclose(value.detach().cpu(), expected_value, rtol=64 * eps, atol=0)
    tiny = float(np.nextafter(np.float32(0), np.float32(1))) if dtype == torch.float32 else 5e-324
    np.testing.assert_allclose(gradient.cpu(), expected_gradient, rtol=64 * eps, atol=tiny)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("classes", [97, 4097, 32000])
def test_distillation_signed_partition_residual(dtype: torch.dtype, classes: int):
    scores = torch.zeros((2, classes), dtype=dtype, device="cuda")
    scores[0, 1] = 1
    scores[1, 0] = 13.25
    teacher = scores.clone()
    teacher[0, 0], teacher[0, 1] = 1, 0
    teacher[1, 1], teacher[1, 2] = 2**-23, -(2**-23)
    scores = scores.repeat(2, 1)
    teacher = teacher.repeat(2, 1)
    teacher[2] += 4
    if dtype != torch.bfloat16:
        # This shift preserves both perturbations exactly in FP32 and FP64.
        teacher[3] += 0.125
    scores.requires_grad_()
    values = distillation_loss(scores, teacher, temperature=2.0, reduction="none")
    for upstream in (1.0, -3.0):
        (gradient,) = torch.autograd.grad(
            values, scores, torch.full_like(values, upstream), retain_graph=True
        )
        # Permuting classes leaves their partition unchanged. Opposing tiny changes
        # instead increase the teacher partition by 4*sinh(2**-25)**2.
        grad = gradient.double().cpu().numpy() * np.sign(upstream)
        assert np.all(grad[::2, 0] < 0) and np.all(grad[::2, 1] > 0)
        assert not np.count_nonzero(grad[::2, 2:])
        assert np.all(grad[1::2, 0] > 0) and np.all(grad[1::2, 1] < 0)
        assert np.all(grad[1::2, 2:] > 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streamed_double_shifts_across_the_bounded_row_cutoff():
    classes = 8193
    code = torch.arange(classes, device="cuda")
    student = ((code % 13) - 6).double() / 4
    teacher = student + ((code % 3) - 1).double() / 8
    shifts = torch.tensor([0, 32, 2**40], device="cuda", dtype=torch.float64)[:, None]
    # The three exactly represented shifts take both sides of the 16*T score cutoff.
    # Repeat them with a shared mask, which selects the general support calculation.
    x = (student + shifts).repeat(2, 1)
    t = (teacher - shifts).repeat(2, 1)
    x[3:, 5] = -torch.inf
    t[3:, 5] = -torch.inf
    x.requires_grad_()
    values = distillation_loss(x, t, temperature=2.0, reduction="none")
    weights = torch.tensor([0.75] * 3 + [-0.25] * 3, device="cuda", dtype=torch.float64)
    (gradient,) = torch.autograd.grad(values, x, weights)
    for i in (0, 3):
        assert torch.isfinite(values[i])
        assert torch.equal(values[i : i + 3], values[i].expand(3))
        assert torch.equal(gradient[i : i + 3], gradient[i].expand(3, classes))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streamed_shifted_permutations_keep_zero_gradients():
    classes = 32000
    column = torch.arange(classes, device="cuda") % 32
    logits = (column.double() - 11) / 8
    # A permutation leaves the centered partition function unchanged even when every
    # teacher logit has an additive offset. Unmoved classes have exactly zero gradient.
    swap = torch.where(column == 1, 2, torch.where(column == 2, 1, column))
    teacher = (swap.double() - 11) / 8 + 0.375
    scores = logits.unsqueeze(0).requires_grad_()
    value = distillation_loss(scores, teacher[None, :], temperature=2)
    (grad,) = torch.autograd.grad(value, scores)
    assert torch.isfinite(value)
    assert torch.count_nonzero(grad[0, (column != 1) & (column != 2)]).item() == 0
    assert grad[0, 1] < 0 < grad[0, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streamed_float_distillation_distributed_changes():
    # At this batch size CUDA trains from saved row statistics. Check a spread of
    # small probability differences and a sparse permutation in the same launch.
    classes, groups = 8192, 32
    rng = np.random.default_rng(738)
    s = np.round(32 * rng.normal(0, 3, groups)) / 32
    t = s + np.round(32 * rng.normal(0, 0.5, groups)) / 32
    permuted = s.copy()
    permuted[:2] = s[:2][::-1]
    teachers = np.stack([t, permuted])
    y = torch.from_numpy(teachers.astype(np.float32)).cuda().repeat(256, classes // groups)
    x = torch.from_numpy(s.astype(np.float32)).cuda().repeat(512, classes // groups)
    x.requires_grad_()
    values = distillation_loss(x, y, temperature=2, reduction="none")
    up = torch.tensor([1.0, -0.75], device="cuda").repeat(256)
    (gradient,) = torch.autograd.grad(values, x, up)
    q = np.exp(s / 2)
    q /= q.sum()
    p = np.exp(teachers / 2)
    p /= p.sum(1, keepdims=True)
    expected_values = np.log(np.exp(s / 2).sum() * (classes // groups)) - (p * s / 2).sum(1)
    multiplier = np.array([1.0, -0.75])[:, None] / (2 * (classes // groups))
    expected = (q - p) * multiplier
    quantum = np.finfo(np.float32).eps
    np.testing.assert_allclose(values[:2].detach().cpu(), expected_values, rtol=2 * quantum)
    got = gradient[:2, :groups].cpu().numpy()
    assert np.all(
        np.abs(got - expected)
        <= 3 * quantum * np.max((q + p) * abs(multiplier), axis=1, keepdims=True)
    )
    assert np.count_nonzero(got[1, 2:]) == 0
