"""Distillation values and gradients against the mpmath references (conformance.references)."""

import itertools
from typing import Any

import numpy as np
import pytest
from conformance.references import Loss, evaluate

# Relative error budget in working eps (docs/API.md, Known limits): the CPU paths and JAX measure at most
# 0.15 eps on these cases.
BUDGET = 64


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
    """Distillation of the represented rows by conformance.references."""
    return evaluate(Loss("distillation", temperature=temperature), student, teacher=teacher)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x.astype(np.float64) - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def check_reference(
    values: np.ndarray,
    gradients: np.ndarray | None,
    student: np.ndarray,
    teacher: np.ndarray,
    temperature: float,
    work: type[np.floating[Any]] | None = None,
) -> None:
    """values and gradients (unless None) within the documented budget of the reference:
    64 eps of each entry in float64 work, of (p + q) / T in float32 work (docs/API.md, Known
    limits)."""
    expected_value, expected_gradient = reference(student, teacher, temperature)
    info = np.finfo(student.dtype)
    eps = float(np.finfo(work or student.dtype).eps)
    np.testing.assert_allclose(values, expected_value, rtol=float(BUDGET * info.eps), atol=0)
    if gradients is None:
        return
    if eps == float(np.finfo(np.float64).eps):
        scale = np.abs(expected_gradient)
    else:
        scale = (_softmax(student / temperature) + _softmax(teacher / temperature)) / temperature
    bound = BUDGET * max(eps, float(info.eps)) * scale + float(info.smallest_subnormal)
    assert np.all(np.abs(gradients - expected_gradient) <= bound), (gradients, expected_gradient)


def check_temperature_case(
    device: str, dtype: type[np.floating[Any]], large: float, temperature: float | None
) -> None:
    """Rows at +-large (dtype's) with temperature, or large itself where None."""
    import torch
    from libsdca.torch import distillation_loss

    large = float(dtype(large))
    temperature = large if temperature is None else temperature
    students = np.array([[large, -large], [large, -large]], dtype=dtype)
    teachers = np.array([[large, -large], [-large, large]], dtype=dtype)
    # Opposing common offsets also exercise near-equal distributions:
    # raw student-teacher differences overflow but class gaps do not.
    if temperature == large:
        near = np.nextafter(dtype(large), dtype(0))
        students = np.concatenate((students, [[large, near]]))
        teachers = np.concatenate((teachers, [[-near, -large]]))
        students, teachers = students.astype(dtype), teachers.astype(dtype)
    x = torch.tensor(students, device=device, requires_grad=True)
    t = torch.tensor(teachers, device=device)
    value = distillation_loss(x, t, temperature=temperature, reduction="none")
    (gradient,) = torch.autograd.grad(value.sum(), x)
    values, gradients = value.detach().cpu().numpy(), gradient.cpu().numpy()
    # A row whose spread overflows the work type may be invalid: then its
    # value and every gradient entry are NaN (docs/API.md, Known limits).
    with np.errstate(over="ignore"):
        spread = np.maximum(np.ptp(students, 1), np.ptp(teachers, 1))
    overflows = ~np.isfinite(spread.astype(dtype))
    invalid = overflows & np.isnan(values)
    assert np.isnan(gradients[invalid]).all(), gradients
    keep = ~invalid
    check_reference(values[keep], gradients[keep], students[keep], teachers[keep], temperature)


# (dtype, large, temperature; None: large)
TEMPERATURE_CASES = [
    (dtype, large, temperature)
    for dtype, large in ((np.float32, 3e38), (np.float64, 1e308))
    for temperature in (4.0, None)
]


def _temperature_params() -> list[Any]:
    out = []
    for dtype, large, temperature in TEMPERATURE_CASES:
        key = f"{np.dtype(dtype).name}-{'max' if temperature is None else temperature}"
        out.append(pytest.param(dtype, large, temperature, id=key))
    return out


class DistillationReferenceTests:
    @pytest.mark.parametrize("dtype,large,temperature", _temperature_params())
    def test_torch_cpu_temperature_range(
        self, dtype: type[np.floating[Any]], large: float, temperature: float | None
    ):
        check_temperature_case("cpu", dtype, large, temperature)

    def test_torch_cuda_temperature_range(self, subtests: pytest.Subtests):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        for dtype, large, temperature in TEMPERATURE_CASES:
            with subtests.test(dtype=dtype, temperature=temperature):
                check_temperature_case("cuda", dtype, large, temperature)

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
                            z, t, temperature=temperature, reduction="none", precision=precision
                        )
                        (gradient,) = torch.autograd.grad(value.sum(), z)
                        check_reference(
                            value.detach().cpu().numpy(),
                            gradient.cpu().numpy(),
                            student,
                            teacher,
                            temperature,
                            np.float64 if precision == "float64" else np.float32,
                        )

    def test_torch_cpu(self, subtests: pytest.Subtests):
        self.check_torch(subtests, "cpu")

    def test_torch_cuda(self, subtests: pytest.Subtests):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        self.check_torch(subtests, "cuda")


@pytest.mark.parametrize(
    "device,precision", [("cpu", None), ("cpu", "float64"), ("cuda", None), ("cuda", "float64")]
)
@pytest.mark.parametrize("classes", [4, 40])
def test_float32_rows_with_a_large_shift(device: str, precision: str | None, classes: int):
    """A shift near 100, beyond the 88 where float32 exponentials leave the normal range: the
    entries where student and teacher nearly agree (2e-9 beside p = 4.5e-5) are within the
    budget of (p + q) / T in float32 work, of each entry in float64 work."""
    import torch
    from libsdca.torch import distillation_loss

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    student = np.full((2, classes), -100.0, dtype=np.float32)
    teacher = np.full((2, classes), -100.0, dtype=np.float32)
    student[:, :4] = [[0.0, -10.0, -100.0, -21.5], [-21.5, -100.0, -10.0, 0.0]]
    teacher[:, :4] = [[-100.0, -10.0, 0.0, -10.0], [-10.0, 0.0, -10.0, -100.0]]
    z = torch.tensor(student, device=device, requires_grad=True)
    target = torch.tensor(teacher, device=device)
    value = distillation_loss(z, target, reduction="none", precision=precision)
    (gradient,) = torch.autograd.grad(value.sum(), z)
    work = np.float64 if precision == "float64" else np.float32
    check_reference(
        value.detach().cpu().numpy(), gradient.cpu().numpy(), student, teacher, 1.0, work
    )


@pytest.mark.parametrize("dtype,gap", [(np.float32, -88.0), (np.float64, -715.0)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_proximal_distillation_adjacent_tail_scores(
    dtype: type[np.floating[Any]], gap: float, device: str
) -> None:
    """The pivot collects a representable subnormal difference of individually equal tails."""
    import torch
    from libsdca.torch import proximal_distillation_loss

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    student = np.full((2, 1000), gap, dtype=dtype)
    student[:, 0] = 0
    teacher = student.copy()
    for row, direction in enumerate((np.inf, -np.inf)):
        teacher[row, 1:] = np.nextafter(dtype(gap), dtype(direction))
    _, expected = evaluate(Loss("proximal_distillation", alpha=0.01), student, teacher=teacher)
    x = torch.tensor(student, device=device, requires_grad=True)
    value = proximal_distillation_loss(
        x, torch.tensor(teacher, device=device), backend="composed", gamma=0.01, reduction="sum"
    )
    (gradient,) = torch.autograd.grad(value, x)
    pivot = gradient[:, 0].double().cpu().numpy()
    np.testing.assert_allclose(
        pivot, expected[:, 0], atol=float(np.finfo(dtype).smallest_subnormal), rtol=0
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_proximal_distillation_large_penalty_tail(device: str) -> None:
    """A subnormal probability can retain a significant penalty in log space."""
    import torch
    from libsdca.torch import proximal_distillation_loss

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    student = np.array([[0.0, -88.0, -88.0]], dtype=np.float32)
    teacher = np.array([[0.0, -89.0, -89.0]], dtype=np.float32)
    gamma = 1e32
    _, expected = evaluate(Loss("proximal_distillation", alpha=gamma), student, teacher=teacher)
    x = torch.tensor(student, device=device, requires_grad=True)
    value = proximal_distillation_loss(
        x, torch.tensor(teacher, device=device), backend="composed", gamma=gamma
    )
    (gradient,) = torch.autograd.grad(value, x)
    np.testing.assert_allclose(
        gradient.double().cpu().numpy(),
        expected,
        atol=8 * float(np.finfo(np.float32).smallest_subnormal),
        rtol=0,
    )
