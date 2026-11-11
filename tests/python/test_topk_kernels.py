"""Fused integer-target top-k kernels (1 < k <= 32) against loss.h and 60-digit references.

On CUDA the losses go through sdca_loss; with TRITON_INTERPRET=1 and no GPU the same
kernels run on CPU tensors.
"""

import contextlib
import os
import types
from collections.abc import Sequence
from typing import Any
from unittest import mock

import libsdca
import mpmath as mp
import pytest
import torch
import torch.nn.functional as F
from libsdca import _cpu_entropy, _cuda_launch, _topk
from libsdca import torch as sdca_torch
from libsdca.torch import explain_backend, sdca_loss
from mpmath_types import MPReal
from topk_rows import CAPPED_GAPS, capped_reference, capped_rows

try:
    import triton  # noqa: F401
    from libsdca import _triton_topk as T
except ImportError:
    T = None

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else ("cpu" if INTERPRET else None)
if T is not None and DEVICE == "cpu":
    import interpreter  # noqa: F401  # the libdevice shims of the interpreter
# The kernels' kinds: a _smooth hinge is the loss smoothed by gamma (see _public).
LOSSES = (
    "topk_cross_entropy",
    "topk_hinge_alpha",
    "topk_hinge_beta",
    "topk_hinge_alpha_smooth",
    "topk_hinge_beta_smooth",
)
SMOOTH = ("topk_hinge_alpha_smooth", "topk_hinge_beta_smooth")


def _public(kind: str, gamma: float = 0.5) -> tuple[str, float]:
    """The loss name and gamma of a kernel kind: gamma smooths only a _smooth kind."""
    return kind.removesuffix("_smooth"), gamma if kind.endswith("_smooth") else 0.0


@contextlib.contextmanager
def _setting(value: str):
    """SDCA_CUDA_LAUNCHER=value, reread by the launcher status (as test_cuda_launch)."""
    with mock.patch.dict(os.environ, {_cuda_launch.ENV: value}):
        _cuda_launch.status.cache_clear()
        try:
            yield
        finally:
            _cuda_launch.status.cache_clear()


@contextlib.contextmanager
def _python_launches():
    """Counts the losses autograd runs in Python, i.e. launched through Triton."""
    with mock.patch.object(_topk._TopK, "apply", wraps=_topk._TopK.apply) as apply:
        yield apply


def _triton():
    """The Triton top-k module; the classes that use it skip without it."""
    assert T is not None
    return T


def _upstream(z: torch.Tensor, reduction: str, scale: float = 1.0) -> torch.Tensor:
    shape = (z.shape[0],) if reduction == "none" else ()
    dtype = torch.promote_types(z.dtype, torch.float32)  # the loss value dtype
    return torch.full(shape, scale, dtype=dtype, device=z.device)


def fused(
    z: torch.Tensor,
    y: torch.Tensor,
    loss: str,
    k: int,
    gamma: float = 0.5,
    reduction: str = "none",
    upstream: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Value and gradient of the fused kernels (through sdca_loss on CUDA) in their default
    arithmetic: float32 exponentials of non-float64 logits."""
    upstream = _upstream(z, reduction) if upstream is None else upstream
    if z.is_cuda:
        x = z.clone().requires_grad_()
        name, gamma = _public(loss, gamma)
        value = sdca_loss(x, y, name, k=k, gamma=gamma, reduction=reduction)
        value.backward(upstream)
        assert x.grad is not None
        return value.detach(), x.grad
    kernels = _triton()
    fp32 = z.dtype != torch.float64
    value, stats = kernels.forward(z, y, reduction, loss, k, gamma, fp32)
    return value, kernels.backward(z, y, stats, upstream, reduction, loss, k, gamma, fp32)


def _scaled(
    values: torch.Tensor,
    gradients: torch.Tensor,
    y: torch.Tensor,
    dtype: torch.dtype,
    reduction: str,
    upstream: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce float64 row values and scale gradients, rounding each entry once as the kernels
    do: the target entry is minus the unrounded sum of the others."""
    n = values.shape[0]
    u = upstream.double().cpu()
    weight = u if reduction == "none" else u.expand(n) / (n if reduction == "mean" else 1)
    value = {"none": values, "sum": values.sum(), "mean": values.mean()}[reduction]
    others = (gradients * weight[:, None]).scatter(1, y[:, None], 0)
    target = -others.sum(1, keepdim=True)
    out = torch.promote_types(dtype, torch.float32)
    return value.to(out), others.scatter(1, y[:, None], target).to(dtype)


def native(
    z: torch.Tensor,
    y: torch.Tensor,
    loss: str,
    k: int,
    gamma: float = 0.5,
    reduction: str = "none",
    upstream: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The same from include/sdca/loss.h in float64."""
    upstream = _upstream(z, reduction) if upstream is None else upstream
    y = y.cpu()
    name, gamma = _public(loss, gamma)
    values, gradients = libsdca.loss_and_grad(
        z.double().cpu().numpy(), y.numpy(), name, k=k, gamma=gamma
    )
    value, gradient = _scaled(
        torch.from_numpy(values), torch.from_numpy(gradients), y, z.dtype, reduction, upstream
    )
    return value.to(z.device), gradient.to(z.device)


def _knapsack(h: list[MPReal], cap: MPReal, rhs: MPReal):
    """t with sum(clip(h - t, 0, cap)) = rhs, from the breakpoints of the mass."""

    def mass(t: MPReal):
        return sum(min(max(x - t, 0), cap) for x in h)

    points = sorted(set(h) | {x - cap for x in h})
    for lo, hi in zip(points, points[1:], strict=False):
        if mass(lo) >= rhs >= mass(hi):
            span = mass(lo) - mass(hi)
            return lo if span == 0 else lo + (mass(lo) - rhs) * (hi - lo) / span
    return points[0] - (rhs - mass(points[0])) / len(h)


def _cone(h: list[MPReal], k: int):
    """Projection of h onto 0 <= x_i <= sum(x)/k, by its KKT conditions over the sorted h."""
    s = sorted(h, reverse=True)
    if sum(s[:k]) <= 0:
        return [mp.mpf(0)] * len(h)
    c = sum(s[:k]) / k
    if k == len(s) or s[k] <= s[k - 1] - c:
        return [min(max(x - (s[k - 1] - c), 0), c) for x in h]
    tol = mp.mpf(10) ** (-40) * (1 + max(abs(x) for x in s))
    for u in range(k):
        for m in range(len(s) - u + 1):
            head, rest = sum(s[:u]), sum(s[u : u + m])
            d = (k - u) ** 2 + m * u
            if d == 0:
                continue
            t, c = (u * rest - (k - u) * head) / d, (m * head + (k - u) * rest) / d
            ok = c > 0 and all(x - t >= c - tol for x in s[:u])
            ok = ok and all(-tol <= x - t <= c + tol for x in s[u : u + m])
            if ok and all(x <= t + tol for x in s[u + m :]):
                return [min(max(x - t, 0), c) for x in h]
    raise AssertionError("no cone active set")


def reference(row: Sequence[float], y: int, loss: str, k: int, gamma: float = 0.5):
    """Value and gradient of one row in 60 digits, from the losses' definitions."""
    with mp.workdps(60):
        z = [mp.mpf(float(x)) for x in row]
        others = [j for j in range(len(z)) if j != y]
        gradient = [mp.mpf(0)] * len(z)
        if loss == "topk_cross_entropy":
            v = {j: z[j] - z[y] for j in others}
            order = sorted(others, key=lambda j: -v[j])
            u, lse = 0, mp.mpf(0)  # k >= 1 rebinds both
            for u in range(k):
                lse = mp.log(sum(mp.exp(v[j]) for j in order[u:]))
                if v[order[u]] <= lse - mp.log(k - u):
                    break
            odds = (sum(v[j] for j in order[:u]) + (k - u) * (lse - mp.log(k - u))) / k
            odds += mp.log(k)
            mass = 1 / (1 + mp.exp(-odds))
            for i, j in enumerate(order):
                gradient[j] = mass / k if i < u else mass * (k - u) / k * mp.exp(v[j] - lse)
            value = mp.log1p(mp.exp(odds))
        else:
            h = [1 + z[j] - z[y] for j in others]
            if loss in ("topk_hinge_alpha", "topk_hinge_beta"):
                order = sorted(range(len(h)), key=lambda i: -h[i])[:k]
                terms = [max(h[i], 0) if loss == "topk_hinge_beta" else h[i] for i in order]
                value = max(sum(terms) / k, 0)
                a = [mp.mpf(0)] * len(h)
                for i in order:
                    if value > 0 and (loss == "topk_hinge_alpha" or h[i] > 0):
                        a[i] = mp.mpf(1) / k
            else:
                cap = mp.mpf(gamma) / k
                if loss == "topk_hinge_alpha_smooth":
                    a = _cone(h, k)
                    full = sum(a) > gamma
                else:
                    a = [min(max(x, 0), cap) for x in h]
                    full = sum(a) > gamma
                if full:
                    t = _knapsack(h, cap, mp.mpf(gamma))
                    a = [min(max(x - t, 0), cap) for x in h]
                value = sum(x * b - b * b / 2 for x, b in zip(h, a, strict=True)) / gamma
                a = [b / gamma for b in a]
            for i, j in enumerate(others):
                gradient[j] = a[i]
        gradient[y] = -sum(gradient, mp.mpf(0))
        return float(value), [float(g) for g in gradient]


def _close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    message: str,
) -> None:
    """Elementwise relative tolerance, with an absolute floor at rtol times the largest entry."""
    floor = rtol * expected.abs().max().item() if expected.numel() else 0.0
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=rtol, atol=max(floor, 1e-300), msg=message
    )


@pytest.mark.skipif(
    not (T is not None and DEVICE), reason="needs CUDA with Triton, or TRITON_INTERPRET=1"
)
class TopkKernelTests:
    def setup_method(self):
        self.gen = torch.Generator().manual_seed(0)

    def _data(
        self, n: int, c: int, scale: float = 3.0, dtype: torch.dtype = torch.float64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = scale * torch.randn(n, c, dtype=torch.float64, generator=self.gen)
        y = torch.randint(0, c, (n,), generator=self.gen)
        return z.to(dtype).to(DEVICE), y.to(DEVICE)

    def assert_native(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        loss: str,
        k: int,
        rtol: float = 1e-12,
        **options: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value, gradient = fused(z, y, loss, k, **options)
        ref_value, ref_gradient = native(z, y, loss, k, **options)
        message = f"{loss} k={k} {tuple(z.shape)} {z.dtype} {options}"
        _close(value, ref_value, rtol, message)
        _close(gradient, ref_gradient, rtol, message)
        return value, gradient

    def test_random_rows_against_native(self):
        shapes = (
            ((4, 4), (16, 10), (8, 100), (3, 1025))
            if INTERPRET
            else ((4, 4), (64, 10), (33, 100), (8, 1025), (5, 4099))
        )
        for n, c in shapes:
            z, y = self._data(n, c)
            for k in sorted({2, 3, min(5, c - 1), min(32, c - 1)}):
                for loss in LOSSES:
                    if loss == "topk_cross_entropy" and k == c - 1:
                        continue
                    for scale in (3.0, 0.1):
                        self.assert_native(scale / 3 * z, y, loss, k)
        z, y = self._data(37, 20)
        for loss in LOSSES:
            for reduction in ("none", "sum", "mean"):
                for scale in (0.25, -3.0):
                    upstream = _upstream(z, reduction, scale)
                    self.assert_native(z, y, loss, 4, reduction=reduction, upstream=upstream)

    def test_ties_at_the_kth_value(self):
        z = torch.tensor(
            [
                [0.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, -1.0],
                [2.0, 2.0, 0.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [3.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0],
            ],
            dtype=torch.float64,
            device=DEVICE,
        )
        y = torch.tensor([0, 2, 3, 0], device=DEVICE)
        for k in (2, 3, 4, 5, 6):
            for loss in LOSSES:
                _, gradient = self.assert_native(z, y, loss, k)
                if loss in ("topk_hinge_alpha", "topk_hinge_beta"):
                    # Exactly the stable-sort choice: ties take the first indices.
                    others = F.one_hot(y, 8).bool().logical_not()
                    expected = native(z, y, loss, k)[1]
                    assert torch.equal(gradient[others], expected[others]), loss
                    for i in range(4):
                        ref = reference(z[i].tolist(), int(y[i]), loss, k)[1]
                        chosen = [g > 0 for g in gradient[i].tolist()]
                        assert chosen == [g > 0 for g in ref], f"{loss} k={k} row {i}"

    def test_dispatch_bounds(self):
        """The row programs take rows * classes offsets in int32: below 2**31 logits."""
        for n, c, covered in (
            (2**16 - 1, 2**15, True),
            (2**16, 2**15, False),
            (_cuda_launch.ROWS, 4, True),
            (_cuda_launch.ROWS + 1, 4, False),
        ):
            logits = types.SimpleNamespace(is_cuda=True, shape=(n, c))
            # A stand-in with the only attributes covers reads; a real tensor this wide is huge.
            covers = _triton().covers(logits, "topk_hinge_alpha", 2)  # pyright: ignore[reportArgumentType]
            assert covers == covered, (n, c)

    def test_k_near_the_class_count(self):
        for c in (3, 4, 9, 33):
            z, y = self._data(6, c)
            for k in range(max(2, c - 3), c):
                for loss in LOSSES:
                    if loss == "topk_cross_entropy" and k == c - 1:
                        if z.is_cuda:
                            assert not _triton().covers(z, loss, k)
                        continue
                    for gamma in (0.5, 1e-6, 1e6):
                        self.assert_native(z, y, loss, k, gamma=gamma)

    def test_entropy_caps(self):
        # Dominant competitors cap 1 to k - 1 scores; far tails underflow relative to them.
        rows = [
            [0.0, 10.0, 9.0, 8.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0],
            [0.0, 40.0, 39.0, 38.5, 38.0, -700.0, -800.0, -5.0, 1.0, 2.0],
            [0.0, 800.0, 30.0, -800.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [-50.0, 10.0, 10.0, 10.0, 9.0, 9.0, 0.0, 0.0, 0.0, 0.0],
            [1e3, 1e3 + 5, 1e3 + 4, 1e3 - 2, 1e3 - 3, 1e3, 1e3, 1e3, 1e3, 1e3],
            # The k-th score lies 795 below the second: its tail relative to the k-th vanishes.
            [0.0, 5.0, -5.0, -800.0, -801.0, -802.0, -803.0, -804.0, -805.0, -806.0],
        ]
        z = torch.tensor(rows, dtype=torch.float64, device=DEVICE)
        y = torch.zeros(len(rows), dtype=torch.long, device=DEVICE)
        for k in (2, 3, 4, 5, 8):
            value, gradient = self.assert_native(z, y, "topk_cross_entropy", k, rtol=1e-11)
            for i, row in enumerate(rows):
                ref_value, ref_gradient = reference(row, 0, "topk_cross_entropy", k)
                message = f"k={k} row {i}"
                assert round(abs(value[i].item() / ref_value - 1.0), 12) == 0, message
                _close(
                    gradient[i],
                    torch.tensor(ref_gradient, dtype=torch.float64),
                    1e-11,
                    message,
                )

    def test_entropy_free_scores_far_below_a_capped_competitor(self):
        # Relative to the maximum, the free scores' differences round away from a gap of
        # about 1e8 on; relative to the first free score they stay exact.
        rows = []
        for gap in (50.0, 1e4, 1e8, 1e16, 1e30):
            rows.append([-gap, 1.0, gap, -0.3, 0.2, -1.5])
            rows.append([0.0, 1.0, gap, -0.3, 0.2, gap - 3.0])
            rows.append([0.0, gap, gap, 1.0, -0.3, 0.2])
            rows.append([gap, -gap, 2.0, 2.0, 0.5, 1.0])
        z = torch.tensor(rows, dtype=torch.float64, device=DEVICE)
        y = torch.zeros(len(rows), dtype=torch.long, device=DEVICE)
        for k in (2, 3, 4):
            value, gradient = fused(z, y, "topk_cross_entropy", k)
            for i, row in enumerate(rows):
                ref_value, ref_gradient = reference(row, 0, "topk_cross_entropy", k)
                message = f"k={k} row {i} {row}"
                expected = torch.tensor(ref_value, dtype=torch.float64)
                _close(value[i], expected, 1e-13, message)
                expected = torch.tensor(ref_gradient, dtype=torch.float64)
                _close(gradient[i], expected, 1e-12, message)
        # As test_tensor_losses: the gradient at any gap is the native one at 700.
        y = torch.zeros(1, dtype=torch.long, device=DEVICE)
        for k in (2, 3):
            near = torch.tensor([[-700.0, 1.0, 700.0, -0.3, 0.2]], dtype=torch.float64)
            expected = native(near, y.cpu(), "topk_cross_entropy", k)[1]
            for gap in (1e16, 1e30):
                z = torch.tensor([[-gap, 1.0, gap, -0.3, 0.2]], dtype=torch.float64, device=DEVICE)
                _close(fused(z, y, "topk_cross_entropy", k)[1], expected, 3e-13, f"{k} {gap}")

    def test_smooth_caps_cone_and_knapsack(self):
        z, y = self._data(12, 15)
        # The cone's zero and full cases, and a large spread with exactly k on top.
        z[0] = torch.linspace(-10, -9, 15, dtype=torch.float64)
        z[0, y[0]] = 5
        z[1] = torch.tensor([0.0] * 12 + [50.0, 49.0, 48.0], dtype=torch.float64)
        z[2] = torch.tensor([0.0, 1e-9, -1e-9, 2e-9] + [-3.0] * 11, dtype=torch.float64)
        for gamma in (1e-300, 1e-20, 1e-8, 0.25, 1.0, 3.0, 1e8, 1e300):
            for k in (2, 3, 7):
                for loss in SMOOTH:
                    self.assert_native(z, y, loss, k, gamma=gamma, rtol=1e-11)

    def test_offsets_and_tiny_margins(self):
        rows = [[1.0, 1e-20, -5.0, 0.3, 1e-20, -1.0]]
        for offset in (1e8, -1e15):
            rows.append([offset + d for d in (0.0, -1.0, -5.0, 0.5, -2.0, 3.0)])
        z = torch.tensor(rows, dtype=torch.float64, device=DEVICE)
        y = torch.zeros(len(rows), dtype=torch.long, device=DEVICE)
        for k in (2, 3):
            for loss in LOSSES:
                for gamma in (1e-30, 0.5):
                    value, gradient = self.assert_native(z, y, loss, k, gamma=gamma)
                    for i, row in enumerate(rows):
                        ref_value, ref_gradient = reference(row, 0, loss, k, gamma)
                        message = f"{loss} k={k} gamma={gamma} row {i}"
                        _close(
                            value[i],
                            torch.tensor(ref_value, dtype=torch.float64),
                            1e-11,
                            message,
                        )
                        _close(
                            gradient[i],
                            torch.tensor(ref_gradient, dtype=torch.float64),
                            1e-11,
                            message,
                        )

    def test_random_rows_against_mpmath(self):
        z, y = self._data(6, 12)
        z[3] = 0.01 * z[3]
        for k in (2, 4, 11):
            for loss in LOSSES:
                if loss == "topk_cross_entropy" and k == 11:
                    continue
                for gamma in (0.05, 0.5, 5.0):
                    value, gradient = fused(z, y, loss, k, gamma=gamma)
                    for i in range(z.shape[0]):
                        ref_value, ref_gradient = reference(
                            z[i].tolist(), int(y[i]), loss, k, gamma
                        )
                        message = f"{loss} k={k} gamma={gamma} row {i}"
                        _close(
                            value[i],
                            torch.tensor(ref_value, dtype=torch.float64),
                            1e-12,
                            message,
                        )
                        _close(
                            gradient[i],
                            torch.tensor(ref_gradient, dtype=torch.float64),
                            1e-12,
                            message,
                        )

    def test_dtypes_and_upstream_scaling(self):
        z64, y = self._data(16, 30)
        dtypes = [torch.float32] + ([torch.float16, torch.bfloat16] if z64.is_cuda else [])
        for dtype in dtypes:
            z = z64.to(dtype)
            # Both round the float64 results once: values to float32, gradients to the dtype.
            eps = torch.finfo(dtype).eps
            for loss in LOSSES:
                for reduction, scale in (("none", 3.0), ("none", 1e-30), ("mean", 1.0)):
                    if dtype != torch.float32 and scale < 1e-10:
                        continue
                    upstream = _upstream(z, reduction, scale)
                    options = {"reduction": reduction, "upstream": upstream}
                    self.assert_native(z, y, loss, 3, rtol=2 * eps, **options)

    def test_float32_subnormal_target_entry(self):
        # Rows [0, -gap, ...] (target 0) of 1,000 classes whose target entry is a float32
        # subnormal, within 64 eps plus one subnormal quantum: from float32 exponentials it is
        # the competitors' mass, not the sum of their float32 entries (zero from a gap of 104).
        kernels = _triton()
        gaps = (90.0, 100.0, 103.9, 104.0, 108.0, 110.5, 111.0)
        z = torch.zeros(len(gaps), 1000)
        z[:, 1:] = -torch.tensor(gaps)[:, None]
        z, y = z.to(DEVICE), torch.zeros(len(gaps), dtype=torch.long, device=DEVICE)
        upstream = torch.ones(len(gaps), device=DEVICE)
        info = torch.finfo(torch.float32)
        for fp32 in (False, True):
            _, stats = kernels.forward(z, y, "none", "topk_cross_entropy", 5, 0.0, fp32)
            g = kernels.backward(z, y, stats, upstream, "none", "topk_cross_entropy", 5, 0.0, fp32)
            for i, gap in enumerate(gaps):
                with mp.workdps(60):
                    tail = 999 * mp.exp(-mp.mpf(gap))
                    expected = -tail / (1 + tail)
                    error = abs(g[i, 0].item() - expected)
                    budget = 64 * info.eps * abs(expected) + mp.mpf(info.tiny) * info.eps
                    assert error <= budget, (fp32, gap, g[i, 0].item(), expected)

    def test_entropy_capped_confident_rows_within_64_eps(self):
        # As test_tensor_losses: log-domain odds and shifts were up to 339 eps off at m = 649.5.
        eps = torch.finfo(torch.float64).eps
        for classes, k in [(40, 2), (1000, 5)]:
            z = torch.tensor(capped_rows(classes), device=DEVICE)
            y = torch.zeros(len(z), dtype=torch.long, device=DEVICE)
            value, gradient = fused(z, y, "topk_cross_entropy", k)
            for i, gap in enumerate(CAPPED_GAPS):
                actual = (value[i], gradient[i, 0], gradient[i, 1], gradient[i, -1])
                names = ("loss", "target", "capped", "free")
                expected = capped_reference(gap, classes, k)
                for name, got, ref in zip(names, actual, expected, strict=True):
                    error = abs(mp.mpf(got.item()) - ref) / abs(ref)
                    assert error <= 64 * eps, (name, classes, gap, error / eps)

    def test_row_flags_are_zero_or_one(self):
        # Row statistics keep 0 or 1 in the invalid-row slot; the reduction needs no more.
        z, y = self._data(9, 12)
        z[0] = torch.linspace(-10, -9, 12, dtype=torch.float64)
        z[0, y[0]] = 5
        for loss in LOSSES:
            for gamma in (1e-8, 0.5, 1e8):
                _, stats = _triton().forward(z, y, "mean", loss, 3, gamma)
                assert torch.equal(stats[27:36], torch.zeros_like(stats[27:36])), loss
        z[4, 2] = float("nan")
        if z.is_cuda:
            out, stats = _triton().forward(z, y, "none", "topk_hinge_alpha_smooth", 3, 0.5)
            assert stats[27 + 4].item() == 1.0
            assert bool(out[4].isnan())
        else:
            with pytest.raises(ValueError):
                _triton().forward(z, y, "none", "topk_hinge_alpha_smooth", 3, 0.5)

    def test_validation(self):
        for loss in LOSSES:
            for z, y in (
                ([[float("nan"), 0.0, 1.0, 2.0]], [0]),
                ([[0.0, float("inf"), 1.0, 2.0]], [0]),
                ([[0.0, 1.0, 2.0, float("-inf")]], [1]),
                ([[0.0, 0.0, 0.0, 0.0]], [4]),
                ([[0.0, 0.0, 0.0, 0.0]], [-1]),
            ):
                z = torch.tensor(z, dtype=torch.float64, device=DEVICE)
                y = torch.tensor(y, device=DEVICE)
                if z.is_cuda:
                    v, g = fused(z, y, loss, 2)
                    assert bool(v.isnan().all()), f"{loss} {z} {y}"
                    assert bool(g.isnan().all()), f"{loss} {z} {y}"
                else:
                    with pytest.raises(ValueError):
                        fused(z, y, loss, 2)

    def test_short_row_packing_and_mixed_invalid_rows(self):
        kernels = _triton()
        for loss_id in (1, 2, 3, 4):
            for c in (3, 10, 31, 64, 100, 128):
                assert kernels._rows_per_cta(c, loss_id, 2) > 1
            assert kernels._rows_per_cta(129, loss_id, 2) == 1
        # Non-multiple batch sizes and scale sweep on packed C = 10 and C = 100.
        for c, k in ((10, 3), (100, 5)):
            for n in (1, 3, 15, 16, 17):
                z, y = self._data(n, c)
                for loss in LOSSES[1:]:
                    self.assert_native(z, y, loss, k)
        # Scale sweep across 60 orders of magnitude (1e-30 to 1e30), and agreement within a few
        # float64 eps against both loss.h and the 1-row kernel (whose top-only value shortcut has
        # the same nonzero terms summed in sorted rather than column order).
        from libsdca import _triton_reduce as R

        eps = torch.finfo(torch.float64).eps
        z, y = self._data(9, 10)
        for exp in (-30, 0, 30):
            scale = 10.0**exp
            zs, gamma = z * scale, 0.5 * scale
            for loss in SMOOTH:
                vp, _ = self.assert_native(zs, y, loss, 3, gamma=gamma, rtol=16 * eps)
                code = kernels.LOSSES[loss]
                block, _, kb = kernels._shape(10, 3, code)
                s1 = torch.empty(R.size(9, 4), device=DEVICE, dtype=torch.float64)
                v1 = torch.empty(9, device=DEVICE, dtype=torch.float64)
                kernels._forward[(9,)](
                    zs,
                    y,
                    s1,
                    kernels._gamma(zs.device, gamma),
                    9,
                    10,
                    block,
                    code,
                    3,
                    kb,
                    False,
                    1,
                    num_warps=4,
                )
                R.launch(s1, v1, 9, 0, kernels._LAYOUT)
                _close(vp, v1, 8 * eps, f"{loss} exp={exp} packed vs 1-row")
        # In-place backward (G == Z) and mixed invalid rows within one packed CTA.

        clean, cy = self._data(9, 10)
        mixed, my = clean.clone(), cy.clone()
        mixed[2, (int(cy[2]) + 1) % 10] = float("nan")
        my[6] = 10
        u = torch.ones(9, dtype=torch.float64, device=DEVICE)
        for loss in LOSSES[1:]:
            gamma = 0.5 if loss in SMOOTH else 0.0
            code = kernels.LOSSES[loss]
            ref_v, ref_s = kernels.forward(clean, cy, "none", loss, 3, gamma)
            ref_g = kernels.backward(clean, cy, ref_s, u, "none", loss, 3, gamma)
            inplace = clean.clone()
            block, warps, kb = kernels._shape(10, 3, code)
            rows = kernels._rows_per_cta(10, code, 3)
            grid = ((9 + rows - 1) // rows,)
            kernels._backward[grid](
                inplace,
                cy,
                ref_s,
                u,
                inplace,
                kernels._gamma(clean.device, gamma),
                9,
                10,
                block,
                code,
                3,
                0,
                False,
                rows,
                num_warps=warps,
            )
            torch.testing.assert_close(inplace, ref_g, rtol=0, atol=0)
            stats = torch.empty(R.size(9, 4), device=DEVICE, dtype=torch.float64)
            out = torch.empty(9, device=DEVICE, dtype=torch.float64)
            kernels._forward[grid](
                mixed,
                my,
                stats,
                kernels._gamma(mixed.device, gamma),
                9,
                10,
                block,
                code,
                3,
                kb,
                False,
                rows,
                num_warps=warps,
            )
            R.launch(stats, out, 9, 0, kernels._LAYOUT)
            grad = kernels.backward(mixed, my, stats, u, "none", loss, 3, gamma)
            for r in range(9):
                if r in (2, 6):
                    assert bool(out[r].isnan()) and bool(grad[r].isnan().all()), (loss, r)
                else:
                    torch.testing.assert_close(out[r], ref_v[r], rtol=0, atol=0)
                    torch.testing.assert_close(grad[r], ref_g[r], rtol=0, atol=0)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and T is not None), reason="CUDA with Triton is not available"
)
class TopkDispatchTests:
    def setup_method(self):
        self.gen = torch.Generator(device="cuda").manual_seed(0)

    def teardown_method(self):
        _cuda_launch.status.cache_clear()

    def _data(
        self, n: int, c: int, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = 3 * torch.randn(n, c, device="cuda", dtype=dtype, generator=self.gen)
        return z, torch.randint(0, c, (n,), device="cuda", generator=self.gen)

    def test_explain_backend_and_fallbacks(self):
        z, y = self._data(4, 40)
        host, _ = _cuda_launch.status()
        fused_name = "cuda-launcher" if host is not None else "triton"
        for loss in LOSSES:
            for k in (2, 32):
                assert (
                    explain_backend(z, y, _public(loss)[0], gamma=_public(loss)[1], k=k)[0]
                    == fused_name
                ), loss
            name, reason = explain_backend(z, y, _public(loss)[0], gamma=_public(loss)[1], k=33)
            assert name == "torch"
            assert "k <= 32" in reason
            assert (
                explain_backend(
                    z, y, _public(loss)[0], gamma=_public(loss)[1], k=2, precision="float32"
                )[0]
                == "torch"
            )
            mask = F.one_hot(y, 40).bool()
            assert (
                explain_backend(z, mask, _public(loss)[0], gamma=_public(loss)[1], k=2)[0]
                == "torch"
            )
        assert explain_backend(z, y, "topk_cross_entropy", k=39)[0] == "torch"
        # Outside the kernels the tensor path still computes the loss.
        for loss, k in (("topk_cross_entropy", 39), ("topk_hinge_alpha", 33)):
            value = sdca_loss(z.double(), y, _public(loss)[0], gamma=_public(loss)[1], k=k)
            mask = F.one_hot(y, 40).bool()
            expected = sdca_loss(
                z.double(), mask, _public(loss)[0], gamma=_public(loss)[1], k=k, backend="torch"
            )
            torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_variable_batch_sizes_and_forced_triton_launch(self):
        for setting in ("auto", "off"):
            with _setting(setting), _python_launches() as apply:
                host, _ = _cuda_launch.status()
                expected = "cuda-launcher" if host is not None else "triton"
                z, y = self._data(5, 50, torch.float64)
                assert explain_backend(z, y, "topk_hinge_alpha", k=3)[0] == expected, setting
                for loss in LOSSES:
                    for reduction in ("none", "mean"):
                        fused(*self._data(5, 50, torch.float64), loss, 3, reduction=reduction)
                before = _triton()._compiled.cache_info().currsize
                for n in (1, 2, 17, 2047, 2049, 5000, 20000):
                    z, y = self._data(n, 50, torch.float64)
                    for loss in LOSSES:
                        for reduction in ("none", "mean"):
                            value, gradient = fused(z, y, loss, 3, reduction=reduction)
                            ref_value, ref_gradient = native(z, y, loss, 3, reduction=reduction)
                            message = f"{setting} {loss} n={n} {reduction}"
                            _close(value, ref_value, 1e-12, message)
                            _close(gradient, ref_gradient, 1e-12, message)
                assert _triton()._compiled.cache_info().currsize == before, setting
                # The C++ launcher runs its own autograd; the Triton launch Python's.
                assert apply.called == (host is None), setting
        z, y = self._data(300, 1000)
        for loss in LOSSES:
            with _python_launches() as apply:
                default = fused(z, y, loss, 5)
                launched = apply.called
            with _setting("off"), _python_launches() as apply:
                name, reason = explain_backend(z, y, _public(loss)[0], gamma=_public(loss)[1], k=5)
                assert name == "triton", loss
                assert "SDCA_CUDA_LAUNCHER=off" in reason
                forced = fused(z, y, loss, 5)
                assert apply.called, loss
            assert launched == (_cuda_launch.status()[0] is None), loss
            # Both launches give identical results.
            assert torch.equal(default[0], forced[0]), loss
            assert torch.equal(default[1], forced[1]), loss

    def test_large_class_counts(self):
        for c in (32000, 32768, 32769, 128256):
            z, y = self._data(6, c)
            for loss in LOSSES:
                for k in (2, 10, 32):
                    value, gradient = fused(z, y, loss, k)
                    ref_value, ref_gradient = native(z, y, loss, k)
                    message = f"{loss} c={c} k={k}"
                    _close(value, ref_value, 2.4e-7, message)
                    _close(gradient, ref_gradient, 2.4e-7, message)

    def test_first_order_only_and_layout(self):
        z = torch.randn(4, 5, 7, dtype=torch.float64, device="cuda", generator=self.gen)
        y = torch.randint(0, 5, (4, 7), device="cuda", generator=self.gen)
        for loss in LOSSES:
            a = z.clone().requires_grad_()
            b = z.clone().requires_grad_()
            torch.testing.assert_close(
                sdca_loss(
                    a, y, _public(loss)[0], gamma=_public(loss)[1], k=2, dim=1, reduction="none"
                ),
                sdca_loss(
                    b.movedim(1, -1),
                    y,
                    _public(loss)[0],
                    gamma=_public(loss)[1],
                    k=2,
                    reduction="none",
                ),
            )
            x = z.movedim(1, -1).clone().requires_grad_()
            (g,) = torch.autograd.grad(
                sdca_loss(x, y, _public(loss)[0], gamma=_public(loss)[1], k=2), x, create_graph=True
            )
            with pytest.raises(RuntimeError):
                torch.autograd.grad(g.sum(), x)

    def test_short_row_zero_spills_and_cuda_graphs(self):
        kernels = _triton()
        z = torch.empty((1, 1), device="cuda", dtype=torch.float32)
        y = torch.empty(1, device="cuda", dtype=torch.int64)
        s = torch.empty(1, device="cuda", dtype=torch.float64)
        g = torch.empty(1, device="cuda", dtype=torch.float64)
        for c, k in ((3, 2), (10, 5), (31, 5), (64, 16), (100, 5), (128, 16)):
            for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
                zd = z.to(dtype)
                for loss in LOSSES[1:]:
                    code = kernels.LOSSES[loss]
                    block, warps, kb = kernels._shape(c, k, code)
                    rows = kernels._rows_per_cta(c, code, k)
                    kf = kernels._forward.warmup(
                        zd,
                        y,
                        s,
                        g,
                        2,
                        c,
                        block,
                        code,
                        k,
                        kb,
                        False,
                        rows,
                        num_warps=warps,
                        grid=(1,),
                    )
                    kb_kern = kernels._backward.warmup(
                        zd,
                        y,
                        s,
                        g,
                        zd,
                        g,
                        2,
                        c,
                        block,
                        code,
                        k,
                        1,
                        False,
                        rows,
                        num_warps=warps,
                        grid=(1,),
                    )
                    kf._init_handles()
                    kb_kern._init_handles()
                    assert kf.n_spills == 0, (c, k, dtype, loss, kf.n_spills)
                    assert kb_kern.n_spills == 0, (c, k, dtype, loss, kb_kern.n_spills)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        x = torch.randn(257, 10, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        yt = torch.randint(0, 10, (257,), device="cuda")
        for loss in LOSSES[1:]:
            name, gamma = _public(loss)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    (warm,) = torch.autograd.grad(sdca_loss(x, yt, name, k=3, gamma=gamma), x)
                    del warm
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                val = sdca_loss(x, yt, name, k=3, gamma=gamma)
                (captured,) = torch.autograd.grad(val, x)
            with torch.no_grad():
                x.normal_()
            ref_val = sdca_loss(x, yt, name, k=3, gamma=gamma)
            (expected,) = torch.autograd.grad(ref_val, x)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(val, ref_val, rtol=0, atol=0)
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@pytest.mark.skipif(not _cpu_entropy.buildable(), reason="the CPU extension does not build here")
class CpuTopkExtensionTests:
    """CPU C++ ATen extension (SDCA_CPU_EXTENSION=on) for k > 1 single-target losses."""

    def test_cpu_topk_extension_matches_native_and_reference(self):
        gen = torch.Generator().manual_seed(19)
        with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": "on"}):
            for n, c in ((8, 5), (23, 17), (5, 70)):
                z64 = 3 * torch.randn(n, c, dtype=torch.float64, generator=gen)
                y = torch.randint(0, c, (n,), generator=gen)
                for k in sorted({2, 4, min(65, c - 1), c - 1}):
                    for loss in LOSSES:
                        name, reason = explain_backend(
                            z64, y, _public(loss)[0], gamma=_public(loss)[1], k=k
                        )
                        assert name == "cpu-kernel", f"{loss} k={k}: {reason}"
                        for reduction in ("none", "mean", "sum"):
                            u = _upstream(z64, reduction, -0.75)
                            x = z64.clone().requires_grad_()
                            val = sdca_loss(
                                x,
                                y,
                                _public(loss)[0],
                                k=k,
                                gamma=_public(loss)[1],
                                reduction=reduction,
                            )
                            val.backward(u)
                            ref_v, ref_g = native(
                                z64, y, loss, k, gamma=0.5, reduction=reduction, upstream=u
                            )
                            _close(val.detach(), ref_v, 1e-12, f"{loss} k={k} {reduction}")
                            assert x.grad is not None
                            _close(x.grad, ref_g, 1e-12, f"{loss} k={k} {reduction}")
            # All four storage dtypes (including float16 and bfloat16).
            z64 = 3 * torch.randn(16, 20, dtype=torch.float64, generator=gen)
            y = torch.randint(0, 20, (16,), generator=gen)
            for dtype, rtol in (
                (torch.float32, 1e-6),
                (torch.float16, 2e-3),
                (torch.bfloat16, 2e-2),
            ):
                z = z64.to(dtype)
                for loss in LOSSES:
                    assert (
                        explain_backend(z, y, _public(loss)[0], gamma=_public(loss)[1], k=3)[0]
                        == "cpu-kernel"
                    )
                    x = z.clone().requires_grad_()
                    val = sdca_loss(
                        x, y, _public(loss)[0], k=3, gamma=_public(loss)[1], reduction="mean"
                    )
                    val.backward()
                    ref_v, ref_g = native(z.double(), y, loss, 3, gamma=0.5, reduction="mean")
                    _close(val.detach().double(), ref_v, rtol, f"{dtype} {loss}")
                    assert x.grad is not None
                    _close(x.grad.double(), ref_g, rtol, f"{dtype} {loss}")
            # Ties, extreme gaps, and invalid inputs.
            ties = torch.tensor(
                [[0.0, 2.0, 2.0, 2.0, 1.0, -1.0], [1e16, 1e16 + 1.0, 1e16 - 0.5, 0.0, 1.0, -1.0]],
                dtype=torch.float64,
            )
            ty = torch.tensor([0, 3])
            for loss in LOSSES:
                for k in (2, 3, 5):
                    x = ties.clone().requires_grad_()
                    val = sdca_loss(
                        x, ty, _public(loss)[0], k=k, gamma=_public(loss)[1], reduction="none"
                    )
                    val.sum().backward()
                    assert x.grad is not None
                    for i in range(len(ties)):
                        rv, rg = reference(ties[i].tolist(), int(ty[i]), loss, k, 0.5)
                        _close(
                            val[i].detach(),
                            torch.tensor(rv, dtype=torch.float64),
                            1e-11,
                            f"{loss} k={k} row {i}",
                        )
                        _close(
                            x.grad[i],
                            torch.tensor(rg, dtype=torch.float64),
                            1e-11,
                            f"{loss} k={k} row {i}",
                        )
                for bad_z, bad_y in (
                    ([[float("nan"), 0.0, 1.0]], [0]),
                    ([[0.0, float("inf"), 1.0]], [0]),
                    ([[0.0, 1.0, 2.0]], [3]),
                    ([[0.0, 1.0, 2.0]], [-1]),
                ):
                    with pytest.raises(ValueError):
                        sdca_loss(
                            torch.tensor(bad_z, dtype=torch.float64),
                            torch.tensor(bad_y),
                            _public(loss)[0],
                            k=2,
                        )


def test_dispatch_takes_float32_exponentials_by_default():
    """The CUDA dispatch (on CPU tensors): non-float64 logits ask the kernels for float32
    exponentials unless precision="float64" (the kernels apply it to entropy only)."""
    seen: list[bool] = []

    def topk(flat: torch.Tensor, *args: Any) -> torch.Tensor:
        seen.append(args[-1])
        return flat.new_zeros((), dtype=torch.float32)

    y = torch.tensor([0, 1])
    with (
        mock.patch.object(_topk, "covers", lambda flat, objective, k: True),
        mock.patch.object(_topk, "topk", topk),
    ):
        for dtype, precision in [
            (torch.bfloat16, None),
            (torch.float32, None),
            (torch.float32, "float64"),
            (torch.float64, None),
        ]:
            z = torch.zeros(2, 5, dtype=dtype)
            args = ("topk_cross_entropy", 2, 0.0, 1.0, "mean", -1, precision, "auto")
            sdca_torch._device_loss(z, z, y, *args)
    assert seen == [True, True, False, False]
