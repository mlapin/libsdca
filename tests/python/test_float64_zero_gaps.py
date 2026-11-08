"""float64 logits [0] * p + [-gap] * (C - p) (the first p positive) at gaps 700 to 752, where
the losses and entries become float64 subnormals and then zero, against 60-digit references.

Every entry is within the release budget (numerics.md, how it is verified: 64 output eps plus
one subnormal quantum), with the CPU extension on and off. Beyond the spread of normal
exponentials (-708) the boolean-mask rows once summed exp(z - max) of every competitor, each
rounded into subnormals first: the extension's loss and positive entries were off by up to
2e4 quanta at 1,000 classes, and the native library's rows of two positives by up to 124.
"""

import functools
import os
from unittest import mock

import mpmath as mp
import pytest
import torch
from libsdca import _cpu_entropy
from libsdca.torch import sdca_loss
from mpmath_types import MPReal

PATHS = {"cpu-native": "off", **({"cpu-kernel": "on"} if _cpu_entropy.buildable() else {})}
# name -> (loss, mask target)
LOSSES = {
    "cross_entropy": ("cross_entropy", False),
    "cross_entropy mask": ("cross_entropy", True),
    "multilabel_cross_entropy": ("multilabel_cross_entropy", True),
}
GAPS = [700 + i / 4 for i in range(209)]
EPS, QUANTUM = torch.finfo(torch.float64).eps, 2.0**-1074


def _evaluate(
    name: str, path: str, classes: int, positives: int
) -> tuple[list[float], list[float], list[float]]:
    """Losses, first positive entries and last competitor entries of one row per gap."""
    loss, mask = LOSSES[name]
    x = torch.zeros(len(GAPS), classes, dtype=torch.float64)
    x[:, positives:] = -torch.tensor(GAPS, dtype=torch.float64).unsqueeze(1)
    x.requires_grad_()
    target = torch.zeros(len(GAPS), dtype=torch.long)
    if mask:
        target = torch.zeros(len(GAPS), classes, dtype=torch.bool)
        target[:, :positives] = True
    with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": PATHS[path]}):
        value = sdca_loss(x, target, loss, reduction="none")
        (gradient,) = torch.autograd.grad(value.sum(), x)
    return value.tolist(), gradient[:, 0].tolist(), gradient[:, -1].tolist()


@functools.cache
def _reference(gap: float, classes: int, positives: int) -> tuple[MPReal, MPReal, MPReal]:
    """60-digit loss log(p + t), positive entry 1 / (p + t) - 1 / p and competitor entry
    exp(-gap) / (p + t), t = (classes - p) exp(-gap), without their cancellations."""
    with mp.workdps(60):
        e = mp.exp(-mp.mpf(gap))
        tail = (classes - positives) * e
        total = positives + tail
        loss = mp.log(positives) + mp.log1p(tail / positives)
        return loss, -tail / (positives * total), e / total


def _within(actual: float, expected: MPReal) -> bool:
    with mp.workdps(60):
        return abs(mp.mpf(actual) - expected) <= 64 * EPS * abs(expected) + QUANTUM


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("name", list(LOSSES))
def test_float64_near_underflow_gaps(name: str, path: str, subtests: pytest.Subtests):
    for classes in (40, 1000):
        for positives in (1, 2) if LOSSES[name][1] else (1,):
            losses, firsts, others = _evaluate(name, path, classes, positives)
            for i, gap in enumerate(GAPS):
                loss, first, other = _reference(gap, classes, positives)
                with subtests.test(classes=classes, positives=positives, gap=gap):
                    assert _within(losses[i], loss), (losses[i], loss)
                    assert _within(firsts[i], first), (firsts[i], first)
                    assert _within(others[i], other), (others[i], other)
