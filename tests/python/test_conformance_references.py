"""The conformance references (tests/python/conformance/references.py) against the oracles of
the other suites: the two-group closed forms of release_numerics.py, the capped
top-k rows of topk_rows, the row references of test_topk_kernels, test_multilabel_kernels and
test_distillation_references, and the float64 Moreau envelope of common_inputs."""

import importlib.util
import random
from pathlib import Path

import common_inputs
import mpmath as mp
import numpy as np
import pytest
import test_distillation_references as distillation
import test_multilabel_kernels as multilabel
import test_topk_kernels as topk
import torch
from conformance.references import Block, Loss, Row, reference
from topk_rows import CAPPED_GAPS, capped_reference

ROOT = Path(__file__).resolve().parents[2]


def _release_numerics():
    spec = importlib.util.spec_from_file_location(
        "release_numerics", ROOT / "tests" / "python" / "release_numerics.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(z: list[float], positive: list[bool], teacher: list[float] | None = None) -> Row:
    t = teacher or [0.0] * len(z)
    return tuple(Block(v, 1, p, s) for v, p, s in zip(z, positive, t, strict=True))


def _close(actual: object, expected: float, ulps: float = 2) -> bool:
    return abs(float(actual) - expected) <= ulps * np.spacing(abs(expected))  # pyright: ignore[reportArgumentType]


def _rows(seed: int):
    rng = random.Random(seed)
    for _ in range(12):
        c = rng.choice([2, 3, 5, 10])
        yield c, [rng.gauss(0, 3) for _ in range(c)], rng.randrange(c), rng


@pytest.mark.parametrize("width", [2, 65, 4097])
def test_two_group_rows_match_release_numerics(width: int):
    numerics = _release_numerics()
    patterns = [(0.0, -110.0), (0.0, -750.0), (1e16, 1e16 - 2), (3e38, -3e38), (-3e38, 3e38)]
    with mp.workdps(100):
        for first, other in patterns:
            for label in (0, 1):
                row: Row = (Block(first, 1, label == 0), Block(other, 1, label == 1))
                row += (Block(other, width - 2),) if width > 2 else ()
                for k in (1, 5, 64):
                    if k >= width or k > 1 and abs(first) > 1e30:
                        continue
                    loss = Loss("cross_entropy") if k == 1 else Loss("topk_cross_entropy", k=k)
                    ref = reference(loss, row)
                    closed = numerics.reference if k == 1 else numerics.topk_reference
                    args = (first, other, width, label) + ((k,) if k > 1 else ())
                    value, target, *_ = closed(*args)
                    # The closed forms round exp(-6e38) to 0: below every subnormal.
                    tiny = mp.mpf(10) ** -400
                    assert abs(ref.value - value) <= mp.mpf(10) ** -45 * abs(value) + tiny
                    error = abs(ref.gradient[label] - target)
                    assert error <= mp.mpf(10) ** -45 * abs(target) + tiny


@pytest.mark.parametrize("width", [10, 100, 4096])
def test_capped_rows_match_topk_rows(width: int):
    for gap in CAPPED_GAPS[::5]:
        for k in (2, 3):
            row = (Block(0.0, 1, True), Block(-gap), Block(-gap - 40, width - 2))
            ref = reference(Loss("topk_cross_entropy", k=k), row)
            expected = capped_reference(gap, width, k)
            for actual, exact in zip((ref.value, *ref.gradient), expected, strict=True):
                assert abs(actual - exact) <= mp.mpf(10) ** -45 * abs(exact)


def test_rows_match_the_kernel_test_references():
    for c, z, y, rng in _rows(1):
        label = [j == y for j in range(c)]
        for k in sorted({1, 2, c - 1} - {c}):
            if k < 1 or k >= c:
                continue
            ref = reference(Loss("topk_cross_entropy", k=k), _row(z, label))
            value, gradient = topk.reference(z, y, "topk_cross_entropy", k)
            assert _close(ref.value, value) and all(map(_close, ref.gradient, gradient))
            for name in ("topk_hinge_alpha", "topk_hinge_beta"):
                for gamma in (0.0, 0.5):
                    kind = name + ("_smooth" if gamma else "")
                    ref = reference(Loss(name, k=k, gamma=gamma), _row(z, label))
                    value, gradient = topk.reference(z, y, kind, k, gamma)
                    assert _close(ref.value, value)
        mask = [rng.random() < 0.5 for _ in range(c)]
        mask[y] = True
        ref = reference(Loss("multilabel_cross_entropy"), _row(z, mask))
        value, gradient = multilabel.exact(z, mask, "cross_entropy", 1.0)
        assert _close(ref.value, value, 4) and all(
            _close(a, b, 4) for a, b in zip(ref.gradient, gradient, strict=True)
        )
        if not all(mask):
            for gamma in (0.0, 0.5):
                kind = "multilabel_hinge_smooth" if gamma else "multilabel_hinge"
                ref = reference(Loss("multilabel_hinge", gamma=gamma), _row(z, mask))
                value, gradient = multilabel.exact(z, mask, kind, gamma)
                assert _close(ref.value, value) and all(map(_close, ref.gradient, gradient))


@pytest.mark.parametrize("temperature", [0.5, 1.0, 4.0])
def test_distillation_matches_test_distillation_references(temperature: float):
    for c, z, _, rng in _rows(2):
        t = [rng.gauss(0, 3) for _ in range(c)]
        row = _row(z, [False] * c, t)
        ref = reference(Loss("distillation", temperature=temperature), row)
        values, gradients = distillation.reference(np.array([z]), np.array([t]), temperature)
        assert _close(ref.value, values[0]) and all(map(_close, ref.gradient, gradients[0]))
        for alpha in (0.01, 1.0, 30.0):
            loss = Loss("proximal_distillation", temperature=temperature, alpha=alpha)
            ref = reference(loss, row)
            values, gradients = distillation.proximal_reference(
                np.array([z]), np.array([t]), temperature, alpha
            )
            assert _close(ref.value, values[0], 4)
            assert all(_close(a, b, 4) for a, b in zip(ref.gradient, gradients[0], strict=True))


@pytest.mark.parametrize("alpha", [1e-3, 0.1, 1.0, 10.0, 100.0])
def test_proximal_entropy_matches_common_inputs(alpha: float):
    for c, z, y, _ in _rows(3):
        ref = reference(
            Loss("proximal_cross_entropy", alpha=alpha), _row(z, [j == y for j in range(c)])
        )
        target = torch.nn.functional.one_hot(torch.tensor([y]), c).double()
        value, gradient = common_inputs._envelope(
            torch.tensor([z], dtype=torch.float64), target, alpha
        )
        scale = 1 + max(map(abs, z))
        assert abs(float(ref.value) - value.item()) <= 1e-12 * scale
        assert np.allclose(
            [float(g) for g in ref.gradient], gradient[0].numpy(), rtol=0, atol=1e-12
        )
