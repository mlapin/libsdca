"""The documented 16-bit zero gaps: the first represented gap at which the loss and the target
gradient of logits [0, -gap, ..., -gap] (target 0) are zero.

numerics.md (representation limits) gives them: with two
classes the float32 loss of 16-bit logits reaches zero at 104 and the 16-bit target entry at
17.34 (float16) and 93 (bfloat16); with 999 equal competitors at 110.94 / 24.25 (float16) and
111 / 100 (bfloat16). Each is a representation limit, not a kernel cutoff: at the last nonzero
gap the 100-digit reference rounds to a nonzero value, at the first zero gap to zero.
Top-k entropy (k = 5) has those of cross_entropy: its target entry is minus the unrounded
competitor sum, which stays nonzero after each competitor entry rounds to zero.
"""

import os
from unittest import mock

import mpmath as mp
import pytest
import torch
from libsdca import _cpu_entropy
from libsdca.torch import sdca_loss
from mpmath_types import MPReal

# The default CPU paths (the tensor path of 16-bit logits works in float32: its float32 losses
# keep less than the subnormal resolution, see numerics.md).
PATHS = {"cpu-native": "off", **({"cpu-kernel": "on"} if _cpu_entropy.buildable() else {})}
# (dtype, classes) -> documented first zero gaps of (loss, target entry).
DOCUMENTED = {
    (torch.float16, 2): (104.0, 17.34375),
    (torch.float16, 1000): (110.9375, 24.25),
    (torch.bfloat16, 2): (104.0, 93.0),
    (torch.bfloat16, 1000): (111.0, 100.0),
}
# name -> (loss, mask target, k)
LOSSES = {
    "cross_entropy": ("cross_entropy", False, 1),
    "cross_entropy mask": ("cross_entropy", True, 1),
    "multilabel_cross_entropy": ("multilabel_cross_entropy", True, 1),
    "topk_cross_entropy k=5": ("topk_cross_entropy", False, 5),
}


def _at(dtype: torch.dtype, classes: int, gap: float, name: str, path: str) -> tuple[float, float]:
    """(loss, target entry) at logits [0, -gap, ...] (target 0), as float64 values."""
    loss, mask, k = LOSSES[name]
    x = torch.full((1, classes), -gap, dtype=dtype)
    x[0, 0] = 0
    x.requires_grad_()
    target = torch.zeros(1, dtype=torch.long)
    if mask:
        target = torch.zeros(1, classes, dtype=torch.bool)
        target[0, 0] = True
    with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": PATHS[path]}):
        value = sdca_loss(x, target, loss, k=k)
        (gradient,) = torch.autograd.grad(value, x)
    assert value.dtype == torch.float32 and gradient.dtype == dtype
    return value.item(), gradient[0, 0].item()


def _first_zero(dtype: torch.dtype, classes: int, name: str, path: str, index: int):
    """The adjacent represented gaps (last nonzero, first zero) of the loss or target entry."""
    lo, hi = 0.0, 120.0
    assert _at(dtype, classes, lo, name, path)[index] != 0
    assert _at(dtype, classes, hi, name, path)[index] == 0
    while True:
        mid = torch.tensor(lo + (hi - lo) / 2, dtype=dtype).item()
        if mid in (lo, hi):
            return lo, hi
        if _at(dtype, classes, mid, name, path)[index] == 0:
            hi = mid
        else:
            lo = mid


def _reference(gap: float, classes: int) -> tuple[MPReal, MPReal]:
    """100-digit loss log1p(t) and target entry -t / (1 + t), t = (classes - 1) exp(-gap)."""
    with mp.workdps(100):
        tail = (classes - 1) * mp.exp(-mp.mpf(gap))
        return mp.log1p(tail), -tail / (1 + tail)


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("name", list(LOSSES))
class HalfZeroGapTests:
    def test_first_zero_gaps(self, name: str, path: str, subtests: pytest.Subtests):
        for (dtype, classes), documented in DOCUMENTED.items():
            if LOSSES[name][2] >= classes:
                continue
            for index, quantity in enumerate(("loss", "target")):
                with subtests.test(dtype=dtype, classes=classes, quantity=quantity):
                    lo, hi = _first_zero(dtype, classes, name, path, index)
                    assert hi == documented[index], (lo, hi)
                    # One rounding of the reference: nonzero below, zero at the first zero.
                    out = torch.float32 if quantity == "loss" else dtype
                    half = mp.mpf(torch.finfo(out).tiny) * torch.finfo(out).eps / 2
                    assert abs(_reference(lo, classes)[index]) > half
                    assert abs(_reference(hi, classes)[index]) <= half
