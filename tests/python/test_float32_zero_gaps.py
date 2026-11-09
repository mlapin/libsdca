"""float32 logits [0, -gap, ..., -gap] (target 0) at gaps 80 to 112, where the loss and the
target entry become float32 subnormals and then zero, against 100-digit references.

Every entry is within the release budget (numerics.md, how it is verified: 64 output eps plus
one subnormal quantum), with the CPU extension on and off. The float32-exponential top-k rows
once summed their float32 competitor entries into the target entry: below exp(-87.3) those
lose bits and from a gap of 104 they vanish, so at 1,000 classes the target entry was off by
up to 2x near 103.9 and zero from 104 where cross_entropy keeps it to 110.94 (ranges report
§3.5); the native library's k = C - 1 rows summed their stored entries likewise.
"""

import functools
import math
import os
from unittest import mock

import mpmath as mp
import pytest
import torch
from libsdca import _cpu_entropy
from libsdca.torch import sdca_loss
from mpmath_types import MPReal

PATHS = {"cpu-native": "off", **({"cpu-kernel": "on"} if _cpu_entropy.buildable() else {})}
# name -> (loss, mask target, k); the float32-exponential rows of the extension and theirs.
LOSSES = {
    "cross_entropy": ("cross_entropy", False, 1),
    "cross_entropy mask": ("cross_entropy", True, 1),
    "multilabel_cross_entropy": ("multilabel_cross_entropy", True, 1),
    **{f"topk_cross_entropy k={k}": ("topk_cross_entropy", False, k) for k in (5, 64, 65)},
}
GAPS = [80 + i / 4 for i in range(129)]
EPS, TINY = torch.finfo(torch.float32).eps, torch.finfo(torch.float32).tiny


def _evaluate(
    name: str, path: str, gaps: list[float], classes: int, reduction: str
) -> tuple[list[float], list[float], list[float]]:
    """Losses, target entries and last competitor entries of one row per gap."""
    loss, mask, k = LOSSES[name]
    x = torch.zeros(len(gaps), classes)
    x[:, 1:] = -torch.tensor(gaps).unsqueeze(1)
    x.requires_grad_()
    target = torch.zeros(len(gaps), dtype=torch.long)
    if mask:
        target = torch.zeros(len(gaps), classes, dtype=torch.bool)
        target[:, 0] = True
    with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": PATHS[path]}):
        value = sdca_loss(x, target, loss, k=k, reduction=reduction)
        (gradient,) = torch.autograd.grad(value.sum(), x)
    assert value.dtype == gradient.dtype == torch.float32
    return value.tolist() if value.dim() else [], gradient[:, 0].tolist(), gradient[:, -1].tolist()


@functools.cache
def _reference(gap: float, classes: int) -> tuple[MPReal, MPReal, MPReal]:
    """100-digit loss log1p(t), target entry -t / (1 + t) and competitor entry
    exp(-gap) / (1 + t), t = (classes - 1) exp(-gap); top-k entropy of k < classes - 1
    equal competitors is cross_entropy."""
    with mp.workdps(100):
        e = mp.exp(-mp.mpf(gap))
        tail = (classes - 1) * e
        return mp.log1p(tail), -tail / (1 + tail), e / (1 + tail)


def _within(actual: float, expected: MPReal) -> bool:
    with mp.workdps(100):
        return abs(mp.mpf(actual) - expected) <= 64 * EPS * abs(expected) + mp.mpf(TINY) * EPS


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("name", list(LOSSES))
def test_float32_near_underflow_gaps(name: str, path: str, subtests: pytest.Subtests):
    for classes in (2, 66, 1000, 4097):
        if LOSSES[name][2] >= classes:
            continue
        for reduction in ("none", "mean"):
            losses, targets, others = _evaluate(name, path, GAPS, classes, reduction)
            scale = 1 / mp.mpf(len(GAPS)) if reduction == "mean" else 1
            for i, gap in enumerate(GAPS):
                loss, target, other = _reference(gap, classes)
                with subtests.test(classes=classes, reduction=reduction, gap=gap):
                    assert not losses or _within(losses[i], loss), (losses[i], loss)
                    assert _within(targets[i], target * scale), (targets[i], target * scale)
                    assert _within(others[i], other * scale), (others[i], other * scale)


def _first_zero(name: str, path: str) -> tuple[float, float]:
    """The adjacent float32 gaps (last nonzero, first zero) of the target entry at 1,000
    classes."""

    def target(gap: float) -> float:
        return _evaluate(name, path, [gap], 1000, "none")[1][0]

    lo, hi = 100.0, 120.0
    assert target(lo) != 0 and target(hi) == 0
    while True:
        mid = torch.tensor(lo + (hi - lo) / 2, dtype=torch.float32).item()
        if mid in (lo, hi):
            return lo, hi
        lo, hi = (lo, mid) if target(mid) == 0 else (mid, hi)


@pytest.mark.parametrize("path", list(PATHS))
def test_topk_target_entry_vanishes_with_cross_entropys(path: str):
    lo, hi = _first_zero("topk_cross_entropy k=5", path)
    assert (lo, hi) == _first_zero("cross_entropy", path)
    # One rounding of the reference: nonzero below, zero at the first zero (110.9375 on the
    # 1/16 grid of the ranges report).
    half = mp.mpf(TINY) * EPS / 2
    assert abs(_reference(lo, 1000)[1]) > half >= abs(_reference(hi, 1000)[1])
    assert math.ceil(hi * 16) / 16 == 110.9375
