"""The root solves of the composed proximal losses (_composed._entropy): float-key bisection
closes any bracket, every finite row is solved within core::root_steps, and a row that is not
solved is NaN, never an unconverged value."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pytest
import torch
from libsdca._composed import _entropy, _losses, xp_torch
from libsdca._limits import limits

DTYPES = [torch.float32, torch.float64]


@pytest.mark.parametrize("dtype", DTYPES)
def test_bisect_closes_any_bracket(dtype: torch.dtype):
    lim, info = limits(dtype), torch.finfo(dtype)
    lo = torch.tensor([[-info.max], [-1.0], [0.0], [-info.tiny], [3.0]], dtype=dtype)
    hi = torch.tensor([[info.max], [1e-30], [info.tiny], [info.tiny], [3.0]], dtype=dtype)
    for _ in range(lim.key_bisection_steps):
        middle = _entropy.bisect(xp_torch, lo, hi)
        assert bool(((lo <= middle) & (middle <= hi)).all())
        lo = torch.where(middle == lo, lo, middle)
        hi = torch.where(middle == lo, lo, hi)
    assert bool((_entropy.bisect(xp_torch, lo, hi) == lo).all())
    zero = torch.tensor([[-0.0]], dtype=dtype)
    assert torch.equal(_entropy.bisect(xp_torch, zero, -zero).signbit(), zero.signbit())


def _counted(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The iterations of every composed while_loop (eager, condition tested every step)."""
    counts: list[int] = []

    def while_loop(more: Any, body: Any, state: Any, steps: int) -> Any:
        i = 0
        while i < steps and bool(more(state)):
            state = body(state)
            i += 1
        counts.append(i)
        return state

    monkeypatch.setattr(xp_torch, "while_loop", while_loop)
    return counts


def _rows(dtype: torch.dtype, scale: float, kind: str, classes: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if kind == "normal":
        z = torch.randn(16, classes, generator=g, dtype=torch.float64) * scale
    elif kind == "ties":
        z = torch.randint(0, 3, (16, classes), generator=g).double() * scale
    elif kind == "far":
        z = torch.randn(16, classes, generator=g, dtype=torch.float64)
        z[:, 0] += scale
    else:
        z = torch.randn(16, classes, generator=g, dtype=torch.float64)
        z[:, 1:] += scale
    z = z.to(dtype)
    if classes > 3:
        z[::3, 2] = -math.inf
    return z


@pytest.mark.parametrize("dtype", DTYPES)
def test_finite_rows_are_solved_within_root_steps(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
):
    """Extreme scales, steps, ties and masked classes: every valid row is solved (finite) in
    at most root_steps iterations."""
    counts = _counted(monkeypatch)
    scales = [1e-6, 1, 100, 1e4] + ([1e30] if dtype == torch.float64 else [1e15])
    gammas = [1e-30 if dtype == torch.float64 else 1e-20, 1e-3, 1, 1e3, 1e30]
    for seed, (scale, gamma, kind, classes) in enumerate(
        itertools.product(scales, gammas, ["normal", "ties", "far", "low"], [2, 5, 100])
    ):
        z = _rows(dtype, scale, kind, classes, seed)
        g = torch.Generator().manual_seed(seed)
        labels = torch.randint(0, classes, (16,), generator=g)
        labels = torch.where(torch.isinf(z.gather(1, labels[:, None]))[:, 0], 0, labels)
        teacher = _rows(dtype, scale, kind, classes, seed + 1)
        teacher = torch.where(z == -math.inf, -math.inf, teacher)
        for name, target, options in (
            ("proximal_cross_entropy", labels, {"gamma": gamma}),
            ("proximal_distillation", teacher, {"gamma": gamma, "temperature": 0.5}),
        ):
            x = z.clone().requires_grad_()
            value = getattr(_losses, name)(xp_torch, x, target, **options)
            (grad,) = torch.autograd.grad(value.sum(), x)
            label = f"{name} {dtype} scale={scale} gamma={gamma} {kind} C={classes}"
            assert bool(torch.isfinite(value).all()), label
            assert bool(torch.isfinite(grad[z > -math.inf]).all()), label
    # The loop's steps follow the first evaluation.
    assert counts and max(counts) + 1 <= limits(dtype).root_steps, max(counts)


@pytest.mark.parametrize("dtype", DTYPES)
def test_unsolved_rows_are_nan(dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch):
    """A residual that stays NaN leaves its row unsolved; proximal distillation returns such a
    row as NaN (value and gradient) and solves the others."""
    x = torch.zeros(3, 1, dtype=dtype)
    lo, hi = x - 1, x + 1
    shift = torch.tensor([[0.25], [math.nan], [-0.5]], dtype=dtype)

    def evaluate(t: torch.Tensor) -> tuple[torch.Tensor, ...]:
        f = t - shift
        return f, f == 0, -f, f

    root, done, f = _entropy._root(xp_torch, evaluate, x, lo, hi, x != x)
    assert done[:, 0].tolist() == [True, False, True]
    assert root[0, 0] == 0.25 and root[2, 0] == -0.5
    assert f[0, 0] == 0 and f[2, 0] == 0

    solve = _entropy._root

    def unsolved(*args: Any) -> tuple[torch.Tensor, ...]:
        omega, done, *rest = solve(*args)
        return omega, done & (torch.arange(done.shape[0])[:, None] != 1), *rest

    monkeypatch.setattr(_entropy, "_root", unsolved)
    rng = np.random.default_rng(0)
    z = torch.tensor(rng.normal(size=(3, 5)), dtype=dtype, requires_grad=True)
    teacher = torch.tensor(rng.normal(size=(3, 5)), dtype=dtype)
    value = _losses.proximal_distillation(xp_torch, z, teacher, gamma=1.0)
    (grad,) = torch.autograd.grad(value.sum(), z)
    assert value.isnan().tolist() == [False, True, False]
    assert grad.isnan().all(-1).tolist() == [False, True, False]
