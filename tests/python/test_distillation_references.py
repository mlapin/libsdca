"""Distillation gradients against softmax differences in 120-digit arithmetic."""

import itertools
from typing import Any

import mpmath as mp
import numpy as np
import pytest

# Relative error budget in working eps, as numerics.md: the CPU paths and JAX measure at most
# 0.15 eps on these cases.
BUDGET = 64


def _finfo(dtype: type[np.floating[Any]]):
    return np.finfo(dtype)


def cases(dtype: type[np.floating[Any]], temperature: float) -> tuple[np.ndarray, np.ndarray]:
    """Tiny dominant-class changes coexist with a far-moving negligible class."""
    tiny, gap = (1e-10, 30.0) if dtype == np.float32 else (1e-20, 50.0)
    # The second teacher tail underflows in the working dtype; exp(delta/T)
    # also overflows, although their mathematical product remains small.
    far, huge = (120.0, 1e20) if dtype == np.float32 else (1000.0, 1e300)
    pairs = [([tiny, 0.0, -gap], [0.0, 0.0, -distance]) for distance in (2 * gap, far, huge)]
    students, teachers = [], []
    for student, teacher in pairs:
        for order in itertools.permutations(range(3)):
            students.append([student[j] * temperature for j in order])
            teachers.append([teacher[j] * temperature for j in order])
    return np.array(students, dtype=dtype), np.array(teachers, dtype=dtype)


def reference(
    student: np.ndarray, teacher: np.ndarray, temperature: float
) -> tuple[np.ndarray, np.ndarray]:
    """Independent definition, using the exact represented inputs and T."""
    values, gradients = [], []
    with mp.workdps(120):
        scale = mp.mpf(float(temperature))
        for student_row, teacher_row in zip(student, teacher, strict=True):
            s = [mp.mpf(float(x)) / scale for x in student_row]
            t = [mp.mpf(float(x)) / scale for x in teacher_row]
            s_max, t_max = max(s), max(t)
            es, et = [mp.exp(x - s_max) for x in s], [mp.exp(x - t_max) for x in t]
            s_sum, t_sum = sum(es), sum(et)
            q, p = [x / s_sum for x in es], [x / t_sum for x in et]
            values.append(float(-sum(a * mp.log(b) for a, b in zip(p, q, strict=True))))
            gradients.append([float((a - b) / scale) for a, b in zip(q, p, strict=True)])
    return np.array(values), np.array(gradients)


def proximal_reference(
    student: np.ndarray, teacher: np.ndarray, temperature: float, alpha: float, digits: int = 80
) -> tuple[np.ndarray, np.ndarray]:
    """Envelope value and gradient (pi - p) / T: pi_j = W(alpha exp(s_j + alpha p_j - L)) / alpha
    with L the root of sum(pi) = 1, found by Newton steps from the left of [L0 - alpha, L0]."""
    values, gradients = [], []
    with mp.workdps(digits):
        scale, a = mp.mpf(float(temperature)), mp.mpf(float(alpha))
        for student_row, teacher_row in zip(student, teacher, strict=True):
            s = [mp.mpf(float(x)) / scale for x in student_row]
            t = [mp.mpf(float(x)) / scale for x in teacher_row]
            et = [mp.exp(x - max(t)) for x in t]
            p = [x / sum(et) for x in et]
            x = [si + a * pi for si, pi in zip(s, p, strict=True)]
            top = max(x)
            level = top + mp.log(sum(mp.exp(v - top) for v in x)) - a
            for _ in range(1000):
                w = [mp.lambertw(a * mp.exp(v - level)).real for v in x]
                step = (sum(w) / a - 1) / (sum(v / (1 + v) for v in w) / a)
                level += step
                if abs(step) < mp.mpf(10) ** (10 - digits) * max(1, abs(level)):
                    break
            w = [mp.lambertw(a * mp.exp(v - level)).real for v in x]
            pi = [v / a for v in w]
            cross = -sum(b * (mp.log(v) - mp.log(a)) for b, v in zip(p, w, strict=True) if b)
            penalty = a * sum((u - b) ** 2 for u, b in zip(pi, p, strict=True)) / 2
            values.append(float(cross + penalty))
            gradients.append([float((u - b) / scale) for u, b in zip(pi, p, strict=True)])
    return np.array(values), np.array(gradients)


def check_reference(
    values: np.ndarray,
    gradients: np.ndarray,
    student: np.ndarray,
    teacher: np.ndarray,
    temperature: float,
) -> None:
    expected_value, expected_gradient = reference(student, teacher, temperature)
    info = np.finfo(student.dtype)
    np.testing.assert_allclose(values, expected_value, rtol=float(BUDGET * info.eps), atol=0)
    # Check every component relative to its own magnitude, including the
    # tiny tail; a unit-scale absolute tolerance would hide this regression.
    np.testing.assert_allclose(
        gradients,
        expected_gradient,
        rtol=float(BUDGET * info.eps),
        atol=float(info.smallest_subnormal),
    )
    np.testing.assert_array_equal(np.signbit(gradients), np.signbit(expected_gradient))


class DistillationReferenceTests:
    def check_temperature_range(self, subtests: pytest.Subtests, device: str) -> None:
        import torch
        from libsdca.torch import distillation_loss

        for dtype, large in ((np.float32, 3e38), (np.float64, 1e308)):
            large = float(dtype(large))
            for temperature in (4.0, large):
                students = np.array([[large, -large], [large, -large]], dtype=dtype)
                teachers = np.array([[large, -large], [-large, large]], dtype=dtype)
                # Opposing common offsets also exercise near-equal distributions:
                # raw student-teacher differences overflow but class gaps do not.
                if temperature == large:
                    near = np.nextafter(dtype(large), dtype(0))
                    students = np.concatenate((students, [[large, near]]))
                    teachers = np.concatenate((teachers, [[-near, -large]]))
                    students, teachers = students.astype(dtype), teachers.astype(dtype)
                with subtests.test(device=device, dtype=dtype, temperature=temperature):
                    x = torch.tensor(students, device=device, requires_grad=True)
                    t = torch.tensor(teachers, device=device)
                    value = distillation_loss(x, t, temperature, reduction="none")
                    (gradient,) = torch.autograd.grad(value.sum(), x)
                    expected_value, expected_gradient = reference(students, teachers, temperature)
                    info = _finfo(dtype)
                    np.testing.assert_allclose(
                        value.detach().cpu().numpy(),
                        expected_value,
                        rtol=float(BUDGET * info.eps),
                        atol=0,
                    )
                    np.testing.assert_allclose(
                        gradient.cpu().numpy(),
                        expected_gradient,
                        rtol=float(BUDGET * info.eps),
                        atol=float(info.smallest_subnormal),
                    )

    def test_torch_cpu_temperature_range(self, subtests: pytest.Subtests):
        self.check_temperature_range(subtests, "cpu")

    def test_torch_cuda_temperature_range(self, subtests: pytest.Subtests):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        self.check_temperature_range(subtests, "cuda")

    def check_torch(self, subtests: pytest.Subtests, device: str) -> None:
        import torch
        from libsdca.torch import distillation_loss

        for dtype in (np.float32, np.float64):
            for temperature in (0.25, 1.0, 4.0):
                student, teacher = cases(dtype, temperature)
                precisions = ("float32", "float64") if dtype == np.float32 else ("float64",)
                for precision in precisions:
                    with subtests.test(
                        device=device, dtype=dtype, temperature=temperature, precision=precision
                    ):
                        z = torch.tensor(student, device=device, requires_grad=True)
                        t = torch.tensor(teacher, device=device)
                        value = distillation_loss(
                            z, t, temperature, reduction="none", precision=precision
                        )
                        (gradient,) = torch.autograd.grad(value.sum(), z)
                        check_reference(
                            value.detach().cpu().numpy(),
                            gradient.cpu().numpy(),
                            student,
                            teacher,
                            temperature,
                        )

    def test_torch_cpu(self, subtests: pytest.Subtests):
        self.check_torch(subtests, "cpu")

    def test_torch_cuda(self, subtests: pytest.Subtests):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        self.check_torch(subtests, "cuda")


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x.astype(np.float64) - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@pytest.mark.parametrize(
    "device,extension,precision",
    [
        ("cpu", "on", None),
        ("cpu", "off", None),
        ("cuda", "auto", None),
        ("cuda", "auto", "float64"),
    ],
)
@pytest.mark.parametrize("classes", [4, 40])
def test_float32_rows_with_a_large_shift(
    device: str, extension: str, precision: str | None, classes: int
):
    """A shift near 100: s + shift exceeds the 88 where the float32 exponentials clamp, so the
    CPU kernels take float64 for these rows (the extension's small-row lanes and the native
    library's rows). Float32 work on CUDA keeps a float32 log-ratio: the entries where student
    and teacher nearly agree (2e-9 beside p = 4.5e-5) are within the budget of (p + q) / T."""
    import os
    from unittest import mock

    import torch
    from libsdca.torch import distillation_loss

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    student = np.full((2, classes), -100.0, dtype=np.float32)
    teacher = np.full((2, classes), -100.0, dtype=np.float32)
    student[:, :4] = [[0.0, -10.0, -100.0, -21.5], [-21.5, -100.0, -10.0, 0.0]]
    teacher[:, :4] = [[-100.0, -10.0, 0.0, -10.0], [-10.0, 0.0, -10.0, -100.0]]
    with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": extension}):
        z = torch.tensor(student, device=device, requires_grad=True)
        target = torch.tensor(teacher, device=device)
        value = distillation_loss(z, target, reduction="none", precision=precision)
        (gradient,) = torch.autograd.grad(value.sum(), z)
    values, gradients = value.detach().cpu().numpy(), gradient.cpu().numpy()
    if device == "cpu" or precision == "float64":
        check_reference(values, gradients, student, teacher, temperature=1.0)
        return
    expected_value, expected_gradient = reference(student, teacher, 1.0)
    info = np.finfo(np.float32)
    budget = BUDGET * float(info.eps)
    np.testing.assert_allclose(values, expected_value, rtol=budget, atol=0)
    bound = budget * (_softmax(student) + _softmax(teacher)) + float(info.smallest_subnormal)
    assert np.all(np.abs(gradients - expected_gradient) <= bound)
