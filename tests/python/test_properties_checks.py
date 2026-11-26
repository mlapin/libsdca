"""Unit tests of the checks of the property tests (properties.py), without a framework."""

from __future__ import annotations

import math

import numpy as np
import properties as P
from storage_limits import LIMITS


def _p1(batch: P.Batch, values: list[float], gradient: list[list[float]]) -> P.Outcome:
    return P.check_p1(
        "cross_entropy",
        batch,
        np.array(values),
        np.array(gradient),
        out="float32",
        work="float32",
        framework="numpy",
    )


def test_reduced_nan_fails_beside_a_tolerated_entry() -> None:
    """A NaN sum fails even where a row's gradient took an allowance (a lone subnormal):
    only rows outside the documented range permit a NaN reduction."""
    batch = P.Batch.of([0.0, 1.0], [0], reduction="sum")
    eta = LIMITS["float32"].tiny * LIMITS["float32"].eps
    outcome = _p1(batch, [math.nan], [[eta, 0.0]])
    assert outcome.documented  # the lone entry is within its allowance
    assert outcome.failures == ["sum of the rows: loss nan"]


def test_reduced_nan_passes_beside_a_documented_row() -> None:
    batch = P.Batch.of([[1e308, -1e308], [0.0, 1.0]], [1, 0], reduction="sum")
    batch = P.dataclasses.replace(batch, positives=np.array([[False, True], [True, False]]))
    outcome = P.check_p1(
        "multilabel_cross_entropy",
        batch,
        np.array([math.nan]),
        np.array([[math.nan, math.nan], [-0.5, 0.5]]),
        out="float64",
        work="float64",
        framework="numpy",
    )
    assert not outcome.failures, outcome.failures


def test_negative_loss_fails_p3() -> None:
    """A finite negative loss with a zero-sum gradient passes P1 and fails P3."""
    batch = P.Batch.of([0.0, -2.0, 1.0], [0])
    outcome = _p1(batch, [-1e-3], [[-1.0, 0.0, 1.0]])
    assert any("negative loss" in f for f in outcome.failures)


def test_rounding_below_zero_passes_p3() -> None:
    batch = P.Batch.of([0.0, -100.0], [0])
    outcome = _p1(batch, [-1e-7], [[0.0, 0.0]])
    assert not outcome.failures


def test_zero_sum_bound_and_c() -> None:
    eps = LIMITS["float32"].eps
    g = np.array([-1.0, 0.5, 0.5 + eps])
    total, bound, c = P.zero_sum_bound(g, "float32", flush=False)
    assert total == eps and math.isclose(c, 0.5, rel_tol=1e-6) and bound >= total


def test_cuda_constant_is_separate() -> None:
    """A float32 row 16 eps off zero fails the CPU bound and passes the CUDA one (C_CUDA)."""
    eps = LIMITS["float32"].eps
    g = np.array([-1.0, 1.0 - 16 * eps])
    total, cpu, _ = P.zero_sum_bound(g, "float32", flush=False)
    _, cuda, _ = P.zero_sum_bound(g, "float32", flush=False, cuda=True)
    assert cpu < total <= cuda


def test_cuda_kernel_coverage() -> None:
    wide = P.Batch.of([[0.0] * 1000], [0], k=999, gamma=0.5)
    assert P.outside_kernel_coverage("topk_hinge_beta", wide, ("torch", "cuda", "kernel"))
    assert not P.outside_kernel_coverage("topk_hinge_beta", wide, ("torch", "cpu", "kernel"))
    assert not P.outside_kernel_coverage("topk_cross_entropy", wide, ("torch", "cuda", "kernel"))


def _p2(z: list[float], g: list[float]) -> P.Outcome:
    batch = P.Batch.of([z], [0], probabilities=[P.softmax(z).tolist()])
    return P.check_p2(
        P.PROBABILITIES, batch, np.array([0.0]), np.array([g]), out="float64", storage="float64"
    )


def test_p2_nan_gradient_fails() -> None:
    """A NaN entry fails P2 (a NaN excess compares false with the floor)."""
    assert _p2([0.0, 1.0], [math.nan, 1.0]).failures


def test_p2_zero_mass_entry_keeps_the_others() -> None:
    """At [max, -max] the second entry has t = q = 0 and an infinite condition number: it
    must not turn the floor of the row into NaN and hide the wrong first entry."""
    big = float(np.finfo(np.float64).max)
    assert _p2([big, -big], [1.0, 0.0]).failures
    assert not _p2([big, -big], [0.0, 0.0]).failures
    assert not _p2([0.0, 1.0], (P.softmax([0.0, 1.0]) - P.softmax([0.0, 1.0])).tolist()).failures
