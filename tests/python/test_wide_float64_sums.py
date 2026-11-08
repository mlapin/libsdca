"""Float64 rows of many equal competitors against 100-digit two-group references.

Sequential float64 sums of C - 1 equal terms drift by about C/2 epsilons; every CPU path
must stay within 32 (cross_entropy, distillation_loss and proximal_distillation_loss).
"""

import os
from unittest import mock

import mpmath as mp
import pytest
import torch
from libsdca import _cpu_entropy
from libsdca.torch import cross_entropy, distillation_loss, proximal_distillation_loss
from mpmath_types import MPReal

BUDGET = 32 * torch.finfo(torch.float64).eps
PATHS = {"cpu-native": "off", **({"cpu-kernel": "on"} if _cpu_entropy.buildable() else {})}


def two_group(first: float, other: float, classes: int):
    """Probabilities of the first entry and of each of the classes - 1 others."""
    e = mp.exp(mp.mpf(other) - mp.mpf(first))
    return 1 / (1 + (classes - 1) * e), e / (1 + (classes - 1) * e)


def proximal_two_group(first: MPReal, other: MPReal, classes: int, alpha: MPReal):
    """pi of the first entry and of each other one for scores x (first, then classes - 1 equal):
    pi_j = W(alpha exp(x_j - L)) / alpha with L the root of sum(pi) = 1."""

    def mass(level: MPReal):
        w0 = mp.lambertw(alpha * mp.exp(first - level)).real
        w1 = mp.lambertw(alpha * mp.exp(other - level)).real
        return (w0 + (classes - 1) * w1) / alpha - 1

    top = max(first, other)
    level = top + mp.log(mp.exp(first - top) + (classes - 1) * mp.exp(other - top)) - alpha
    level = mp.findroot(mass, level)
    return [mp.lambertw(alpha * mp.exp(x - level)).real / alpha for x in (first, other)]


class WideFloat64SumTests:
    def check(self, actual: float, expected: MPReal | float, what: str) -> None:
        error = abs(mp.mpf(actual) - expected) / abs(expected)
        assert error <= BUDGET, f"{what}: {actual} vs {mp.nstr(expected, 17)}"

    def test_cross_entropy(self, subtests: pytest.Subtests):
        for name, flag in PATHS.items():
            for classes in (4095, 32769):
                for gap in (1.0, 20.0):
                    with subtests.test(path=name, classes=classes, gap=gap):
                        x = torch.full((1, classes), -gap, dtype=torch.float64)
                        x[0, 0] = 0
                        x.requires_grad_()
                        with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": flag}):
                            loss = cross_entropy(x, torch.tensor([1]))
                            loss.backward()
                        assert x.grad is not None
                        with mp.workdps(100):
                            first, other = two_group(0, -gap, classes)
                            self.check(loss.item(), -mp.log(other), "loss")
                            self.check(x.grad[0, 0].item(), first, "max entry")
                            self.check(x.grad[0, 1].item(), other - 1, "target")

    def test_distillation(self, subtests: pytest.Subtests):
        for name, flag in PATHS.items():
            for classes in (4095, 32769):
                with subtests.test(path=name, classes=classes):
                    student = torch.full((1, classes), -1.0, dtype=torch.float64)
                    teacher = torch.zeros(1, classes, dtype=torch.float64)
                    student[0, 0] = 0
                    student.requires_grad_()
                    with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": flag}):
                        loss = distillation_loss(student, teacher, 2.0)
                        loss.backward()
                    assert student.grad is not None
                    with mp.workdps(100):
                        p, p_other = two_group(0, -0.5, classes)
                        q = mp.mpf(1) / classes
                        value = -(q * mp.log(p) + (classes - 1) * q * mp.log(p_other))
                        self.check(loss.item(), value, "loss")
                        self.check(student.grad[0, 0].item(), (p - q) / 2, "first")
                        self.check(student.grad[0, 1].item(), (p_other - q) / 2, "other")

    def test_proximal_distillation(self, subtests: pytest.Subtests):
        temperature = 2.0
        for name, flag in PATHS.items():
            for classes in (4095, 32769):
                for alpha in (0.5, 100.0):
                    with subtests.test(path=name, classes=classes, alpha=alpha):
                        student = torch.full((1, classes), -1.0, dtype=torch.float64)
                        teacher = torch.zeros(1, classes, dtype=torch.float64)
                        student[0, 0] = 0
                        student.requires_grad_()
                        with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": flag}):
                            loss = proximal_distillation_loss(student, teacher, temperature, alpha)
                            loss.backward()
                        assert student.grad is not None
                        with mp.workdps(100):
                            a, q = mp.mpf(alpha), mp.mpf(1) / classes
                            # x = s / T + alpha p for the uniform teacher p = q.
                            first, other = proximal_two_group(
                                a * q, mp.mpf(-1) / temperature + a * q, classes, a
                            )
                            cross = -(mp.log(first) + (classes - 1) * mp.log(other)) * q
                            squares = (first - q) ** 2 + (classes - 1) * (other - q) ** 2
                            self.check(loss.item(), cross + a * squares / 2, "loss")
                            self.check(student.grad[0, 0].item(), (first - q) / 2, "first")
                            self.check(student.grad[0, 1].item(), (other - q) / 2, "other")
