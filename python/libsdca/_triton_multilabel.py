"""Fused CUDA boolean-mask losses in float64, as include/sdca/loss.h: entropy on label sets,
multilabel_hinge and multilabel_hinge_smooth. With fp32, entropy takes its elementwise
exponentials in float32 and keeps sums and row statistics in float64.

One program streams a row in chunks of at most _CHUNK columns, so any class count fits.
Forward passes keep row statistics; backward recomputes the gradient from them per element.
Quotients by gamma are float64 divisions (div.rn.f64), never a reciprocal that a subnormal
gamma would overflow.
"""

import functools
import os
from typing import Any

import torch
import triton
import triton.language as tl

from . import _cuda_launch, _triton_reduce

_P, _I = _cuda_launch.POINTER, _cuda_launch.INT32

_REDUCTIONS = {"none": 0, "mean": 1, "sum": 2}
# Widest chunk of a row in one pass (narrower for entropy's lane statistics), and the
# backward's column block.
_CHUNK = 2048
_ENTROPY_CHUNK = 1024
_BACKWARD = 1024
# Per objective: (VALUE, LOG, BAD, STATUS) of the row statistics; see _triton_reduce.
_LAYOUTS = {
    "cross_entropy": (0, -1, 5, 6),
    "multilabel_hinge": (0, -1, 4, 5),
    "multilabel_hinge_smooth": (0, -1, 8, 9),
}
# Status codes, in the order of torch._target_mask's checks: the largest is raised.
MESSAGES = {
    3: "logits must be finite floating-point values",
    2: "Every example needs a positive label",
    1: "This loss needs a negative label",
}
_gammas: dict[tuple[int, float, int | None], torch.Tensor] = {}

if os.environ.get("TRITON_INTERPRET") == "1":
    # The interpreter (CPU, for tests) has no libdevice.
    @triton.jit
    def _exp(x):
        return tl.exp(x)

    @triton.jit
    def _exp32(x):
        return tl.exp(x)

    @triton.jit
    def _log1p(x):
        u = 1 + x
        return tl.where((u == 1) | (u == float("inf")), x, tl.log(u) * (x / (u - 1)))

    @triton.jit
    def _expm1(x):
        u = tl.exp(x)
        kahan = (u - 1) * (x / tl.log(u))
        return tl.where(u == 1, x, tl.where((u - 1 == -1) | (u == float("inf")), u - 1, kahan))

    @triton.jit
    def _sqrt(x):
        return tl.sqrt(x)

else:
    from triton.language.extra import libdevice

    @triton.jit
    def _exp(x):
        return libdevice.exp(x)

    @triton.jit
    def _exp32(x):
        t = x * 1.4426950408889634
        r = tl.exp2(t)
        d = tl.fma(t, -0.693145751953125, x)
        d = tl.fma(t, -1.4286067653301870e-06, d)
        return tl.where(t < -150, 0.0, tl.fma(r, d, r))

    @triton.jit
    def _log1p(x):
        return libdevice.log1p(x)

    @triton.jit
    def _expm1(x):
        return libdevice.expm1(x)

    @triton.jit
    def _sqrt(x):
        return libdevice.sqrt(x)


@triton.jit
def _exp_difference(x, origin, half: tl.constexpr = False):
    """Float32 exponential of a represented difference, retaining its rounding residual."""
    gap = x - origin
    rounded = gap - x
    low = (x - (gap - rounded)) + (-origin - rounded)
    # Beyond this range the float32 exponential (or squared half) is zero. Also
    # discard the undefined residual of an overflowing finite-score difference.
    low = tl.where(tl.abs(gap) < 256, low, 0.0)
    if half:
        gap, low = gap * 0.5, low * 0.5
    value = _exp(gap)
    # |low| <= 2^-16 here: exp(low) = 1 + low to substantially better than float eps.
    return value + value * low


@triton.jit
def _work(x, FP32: tl.constexpr):
    """x in the elementwise working type."""
    return x.to(tl.float32 if FP32 else tl.float64)


@triton.jit
def _load(Z, M, row, k, C: tl.constexpr, FP32: tl.constexpr):
    """Scores (working type) and positive flags of columns k, with their bounds and finiteness."""
    valid = k < C
    z = _work(tl.load(Z + row * C + k, valid, other=0), FP32)
    pos = (tl.load(M + row * C + k, valid, other=0) != 0) & valid
    return z, pos, valid, valid & ((z - z) == 0)


@triton.jit
def _chunk(
    Z, M, row, start, j, z1, p1, v1, f1, C: tl.constexpr, B: tl.constexpr, FP32: tl.constexpr
):
    """_load of columns start + j; a row of one chunk reuses its first load (z1, ...)."""
    z, pos, valid, finite = z1, p1, v1, f1
    if B < C:
        z, pos, valid, finite = _load(Z, M, row, start + j, C, FP32)
    return z, pos, valid, finite


@triton.jit
def _code(nonfinite, count, C: tl.constexpr, NEGATIVE: tl.constexpr):
    """3 for nonfinite scores, 2 without a positive, 1 without a negative (if NEGATIVE)."""
    code = tl.where(nonfinite > 0, 3, tl.where(count == 0, 2, 0))
    if NEGATIVE:
        code = tl.where(code == 0, tl.where(count == C, 1, 0), code)
    return code


@triton.jit
def _weight(U, row, N, REDUCTION: tl.constexpr):
    if REDUCTION == 0:
        weight = tl.load(U + row).to(tl.float64)
    else:
        weight = tl.load(U).to(tl.float64)
    if REDUCTION == 1:
        weight = weight / N
    return weight


@triton.jit
def _margin(z, low):
    """1 + z - low; when it cancels, with the low parts of both roundings (TwoSum), as loss.h."""
    d = z - low
    r = d - z
    lower = (z - (d - r)) + (-low - r)
    h = d + 1
    r = h - d
    lower = lower + ((d - (h - r)) + (1 - r))
    cancels = (tl.abs(h) < 0.0625 * tl.abs(d)) & ((lower - lower) == 0)
    return tl.where(cancels, h + lower, h)


@triton.jit
def _extremes(Z, M, row, z1, p1, v1, f1, C: tl.constexpr, B: tl.constexpr):
    """Least positive and largest negative score with their first indices, the first
    positive, the positive count and the nonfinite count."""
    j = tl.arange(0, B)
    low = tl.full([], float("inf"), tl.float64)
    high = tl.full([], -float("inf"), tl.float64)
    i_low = tl.zeros([], tl.int32)
    i_high = tl.zeros([], tl.int32)
    first = tl.full([], C, tl.int32)
    count = tl.zeros([], tl.int32)
    nonfinite = tl.zeros([], tl.int32)
    for start in range(0, C, B):
        k = start + j
        z, pos, valid, finite = _chunk(Z, M, row, start, j, z1, p1, v1, f1, C, B, False)
        nonfinite += tl.sum((valid & ~finite).to(tl.int32), 0)
        count += tl.sum(pos.to(tl.int32), 0)
        first = tl.minimum(first, tl.min(tl.where(pos, k, C), 0))
        zp = tl.where(pos & finite, z, float("inf"))
        zn = tl.where(finite & ~pos, z, -float("inf"))
        lo = tl.min(zp, 0)
        hi = tl.max(zn, 0)
        # The first extreme: the first in its chunk, and the earliest chunk.
        i_low = tl.where(lo < low, tl.min(tl.where(zp == lo, k, C), 0), i_low)
        i_high = tl.where(hi > high, tl.min(tl.where(zn == hi, k, C), 0), i_high)
        low = tl.minimum(low, lo)
        high = tl.maximum(high, hi)
    return low, i_low, high, i_high, first, count, nonfinite


@triton.jit
def _centered(e, e0, dz):
    """exp(z - m) - exp(z0 - m) from dz = z - z0, as math/entropy.h centers the positives."""
    em = _expm1(-tl.abs(dz))
    return tl.where(dz >= 0, -e * em, e0 * em)


@triton.jit
def _entropy_rows_forward(
    Z, M, S, N, C: tl.constexpr, B: tl.constexpr, FP32: tl.constexpr, ROWS: tl.constexpr
):
    """_entropy_forward across a [ROWS, B] tile of short rows (B >= C)."""
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    row_valid = row < N
    j = tl.arange(0, B)[None, :]
    valid = row_valid[:, None] & (j < C)
    z = _work(tl.load(Z + row[:, None] * C + j, valid, other=0), FP32)
    pos = (tl.load(M + row[:, None] * C + j, valid, other=0) != 0) & valid
    finite = valid & ((z - z) == 0)
    count = tl.sum(pos.to(tl.int32), 1)
    first = tl.min(tl.where(pos, j, C), 1)
    zm = tl.where(valid, tl.where(finite, z, float("inf")), -float("inf"))
    top_w = tl.max(zm, 1)
    index = tl.min(tl.where(zm == top_w[:, None], j, C), 1)
    second_w = tl.max(tl.where(j == index[:, None], -float("inf"), zm), 1)
    nonfinite = (top_w == float("inf")).to(tl.int32)
    top = top_w.to(tl.float64)
    scale = _exp(second_w.to(tl.float64) - top)
    scale_w = _work(scale, FP32)
    p = count.to(tl.float64)
    rp_w = _work(1 / p, FP32)
    anchor = tl.sum(tl.where(j == tl.minimum(first, C - 1)[:, None], z.to(tl.float64), 0.0), 1)
    anchor_w = _work(anchor, FP32)
    e0 = _exp(anchor_w - top_w)
    top_negative = (
        tl.sum(tl.where(j == tl.minimum(index, C - 1)[:, None], pos.to(tl.int32), 0), 1) == 0
    )
    ordinary = FP32 & top_negative
    other = valid & (j != index[:, None])
    e = _exp(tl.where(other, z - second_w[:, None], -float("inf")))
    if FP32:
        if tl.max((~ordinary).to(tl.int32), 0) > 0:
            e_comp = tl.where(other, _exp_difference(z, second_w[:, None]), 0.0)
            e = tl.where(ordinary[:, None], e, e_comp)
    e64 = e.to(tl.float64)
    tail = scale * tl.sum(e64, 1)
    raw_gap = top_w[:, None] - z
    scaled_gap = tl.where(
        raw_gap < float("inf"),
        raw_gap * rp_w[:, None],
        top_w[:, None] * rp_w[:, None] - z * rp_w[:, None],
    )
    gap = tl.sum(tl.where(pos, scaled_gap, 0.0).to(tl.float64), 1)
    negatives = tl.zeros([ROWS], tl.float64)
    differences = tl.zeros([ROWS], tl.float64)
    if tl.max((~ordinary).to(tl.int32), 0) > 0:
        negatives = scale * tl.sum(tl.where(pos, 0.0, e64), 1) + tl.where(top_negative, 1.0, 0.0)
        e_top = tl.where(j == index[:, None], 1.0, e * scale_w[:, None])
        centered = _centered(e_top, e0[:, None], z - anchor_w[:, None])
        differences = tl.sum(tl.where(pos, centered, 0.0).to(tl.float64), 1)
    den = 1 + tail
    code = _code(nonfinite, count, C, False)
    bad = code != 0
    nan = float("nan")
    tl.store(S + row, tl.where(bad, nan, gap + _log1p(tail)), row_valid)
    tl.store(S + N + row, tl.where(bad, nan, top), row_valid)
    tl.store(S + 2 * N + row, den, row_valid)
    tl.store(S + 3 * N + row, tl.where(ordinary, float("inf"), anchor), row_valid)
    tl.store(
        S + 4 * N + row,
        tl.where(ordinary, 1 / p, (differences / den + negatives / den) / p),
        row_valid,
    )
    tl.store(S + 5 * N + row, code.to(tl.float64), row_valid)


@triton.jit(do_not_specialize=["N"])
def _entropy_forward(
    Z, M, S, N, C: tl.constexpr, B: tl.constexpr, FP32: tl.constexpr, ROWS: tl.constexpr = 1
):
    """gap / p + log1p(tail) of uniform targets on the positives (detail::entropy_loss_gradient).

    The tail sums exp(z - second) over all but the first top, then scales by exp(second - top).
    Gaps and centered differences are only formed on chunks with a positive. FP32 takes the
    elementwise exponentials in float32; sums and row statistics stay float64.
    """
    if B >= C:
        _entropy_rows_forward(Z, M, S, N, C, B, FP32, ROWS)
    else:
        row = tl.program_id(0).to(tl.int64)
        j = tl.arange(0, B)
        z1, p1, v1, f1 = _load(Z, M, row, j, C, FP32)
        # Per lane: the top with its first column, the runner-up (a tie included), the positives.
        # Nonfinite scores count as +inf: a row is invalid when its top is.
        lane_top = _work(tl.full([B], -float("inf"), tl.float64), FP32)
        lane_second = _work(tl.full([B], -float("inf"), tl.float64), FP32)
        lane_index = tl.full([B], C, tl.int32)
        lane_count = tl.zeros([B], tl.int32)
        lane_first = tl.full([B], C, tl.int32)
        for start in range(0, C, B):
            k = start + j
            z, pos, valid, finite = _chunk(Z, M, row, start, j, z1, p1, v1, f1, C, B, FP32)
            lane_count += pos.to(tl.int32)
            lane_first = tl.minimum(lane_first, tl.where(pos, k, C))
            zm = tl.where(valid, tl.where(finite, z, float("inf")), -float("inf"))
            rises = zm > lane_top
            lane_second = tl.where(rises, lane_top, tl.maximum(lane_second, zm))
            lane_index = tl.where(rises, k, lane_index)
            lane_top = tl.where(rises, zm, lane_top)
        top_w = tl.max(lane_top, 0)
        index = tl.min(tl.where(lane_top == top_w, lane_index, C), 0)
        others_top = tl.max(tl.where(lane_index == index, -float("inf"), lane_top), 0)
        second_w = tl.maximum(tl.max(lane_second, 0), others_top)
        count = tl.sum(lane_count, 0)
        nonfinite = (top_w == float("inf")).to(tl.int32)
        first = tl.min(lane_first, 0)
        top = top_w.to(tl.float64)
        scale = _exp(second_w.to(tl.float64) - top)
        scale_w = _work(scale, FP32)
        p = count.to(tl.float64)
        rp_w = _work(1 / p, FP32)
        lane_tail = tl.zeros([B], tl.float64)
        lane_negatives = tl.zeros([B], tl.float64)
        gap = tl.zeros([B], tl.float64)
        differences = tl.zeros([B], tl.float64)
        anchor_w = _work(tl.load(Z + row * C + tl.minimum(first, C - 1)), FP32)
        e0 = _exp(anchor_w - top_w)
        top_negative = tl.load(M + row * C + tl.minimum(index, C - 1)) == 0
        # A negative maximum has gradient q_max, so the row gradient norm is at least
        # q_max. Direct positive q - 1/p subtraction then has bounded normwise error;
        # well-fit rows with positive maxima keep the cancellation-resistant formula.
        ordinary = FP32 & top_negative
        for start in range(0, C, B):
            k = start + j
            z, pos, valid, finite = _chunk(Z, M, row, start, j, z1, p1, v1, f1, C, B, FP32)
            e = _exp(tl.where(valid & (k != index), z - second_w, -float("inf")))
            if FP32 and not ordinary:
                e = tl.where(valid & (k != index), _exp_difference(z, second_w), 0.0)
            e64 = e.to(tl.float64)
            lane_tail += e64
            if ~ordinary:
                lane_negatives += tl.where(pos, 0.0, e64)
            raw_gap = top_w - z
            scaled_gap = tl.where(raw_gap < float("inf"), raw_gap * rp_w, top_w * rp_w - z * rp_w)
            gap += tl.where(pos, scaled_gap, 0.0).to(tl.float64)
            if ~ordinary:
                e_top = tl.where(k == index, 1.0, e * scale_w)
                centered = _centered(e_top, e0, z - anchor_w)
                differences += tl.where(pos, centered, 0.0).to(tl.float64)
        gap, differences = tl.sum(gap, 0), tl.sum(differences, 0)
        tail = scale * tl.sum(lane_tail, 0)
        negatives = scale * tl.sum(lane_negatives, 0) + tl.where(top_negative, 1.0, 0.0)
        den = 1 + tail
        anchor = tl.load(Z + row * C + tl.minimum(first, C - 1)).to(tl.float64)
        code = _code(nonfinite, count, C, False)
        bad = code != 0
        nan = float("nan")
        tl.store(S + row, tl.where(bad, nan, gap + _log1p(tail)))
        tl.store(S + N + row, tl.where(bad, nan, top))
        tl.store(S + 2 * N + row, den)
        # Finite logits cannot have an infinite anchor: this marks the direct residual.
        tl.store(S + 3 * N + row, tl.where(ordinary, float("inf"), anchor))
        # The positives' share of the zero sum: (differences + negatives) / (den p).
        tl.store(
            S + 4 * N + row, tl.where(ordinary, 1 / p, (differences / den + negatives / den) / p)
        )
        tl.store(S + 5 * N + row, code.to(tl.float64))


@triton.jit(do_not_specialize=["N"])
def _entropy_backward(
    Z,
    M,
    S,
    U,
    G,
    N,
    C: tl.constexpr,
    B: tl.constexpr,
    REDUCTION: tl.constexpr,
    FP32: tl.constexpr,
    ROWS: tl.constexpr = 1,
):
    if ROWS > 1:
        row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        row_valid = row < N
        k = tl.arange(0, B)[None, :]
        valid = row_valid[:, None] & (k < C)
        z = _work(tl.load(Z + row[:, None] * C + k, valid, other=0), FP32)
        pos = (tl.load(M + row[:, None] * C + k, valid, other=0) != 0) & valid
        top = tl.load(S + N + row, row_valid, other=0.0)[:, None]
        den = tl.load(S + 2 * N + row, row_valid, other=1.0)[:, None]
        anchor_1d = tl.load(S + 3 * N + row, row_valid, other=float("inf"))
        anchor = anchor_1d[:, None]
        share = tl.load(S + 4 * N + row, row_valid, other=0.0)[:, None]
        w = (
            tl.load(U + row, row_valid, other=0.0).to(tl.float64)[:, None]
            if REDUCTION == 0
            else tl.load(U).to(tl.float64)
        )
        if REDUCTION == 1:
            w = w / N
        rden = 1 / den
        root = _work(_sqrt(tl.abs(w) * rden), FP32)
        sign = _work(tl.where(w < 0, -1.0, tl.where(w > 0, 1.0, 0.0)), FP32)
        top_w = _work(top, FP32)
        anchor_w = _work(anchor, FP32)
        h = (
            _exp_difference(z, top_w, True)
            if FP32 and Z.dtype.element_ty == tl.float32
            else _exp((z - top_w) * 0.5)
        )
        half = h * root
        gradient = (half * half * sign).to(G.dtype.element_ty)
        direct = ((h.to(tl.float64) * h.to(tl.float64) * rden - share) * w).to(G.dtype.element_ty)
        if tl.max((anchor_1d != float("inf")).to(tl.int32), 0) > 0:
            centered = _centered(h * h, _exp(anchor_w - top_w), z - anchor_w).to(tl.float64)
            sens = ((centered * rden - share) * w).to(G.dtype.element_ty)
            positive = tl.where(anchor == float("inf"), direct, sens)
        else:
            positive = direct
        gradient = tl.where(pos, positive, gradient)
        tl.store(G + row[:, None] * C + k, gradient, valid)
    else:
        row = tl.program_id(0).to(tl.int64)
        k = tl.program_id(1) * B + tl.arange(0, B)
        z, pos, valid, finite = _load(Z, M, row, k, C, FP32)
        top = tl.load(S + N + row)
        den = tl.load(S + 2 * N + row)
        anchor = tl.load(S + 3 * N + row)
        share = tl.load(S + 4 * N + row)
        w = _weight(U, row, N, REDUCTION)
        rden = 1 / den
        # Negatives: w exp(z - top) / den as (exp((z - top) / 2) sqrt(|w| / den))^2, in range.
        root = _work(_sqrt(tl.abs(w) * rden), FP32)
        sign = _work(tl.where(w < 0, -1.0, tl.where(w > 0, 1.0, 0.0)), FP32)
        top_w = _work(top, FP32)
        anchor_w = _work(anchor, FP32)
        # The gap residual matters at float32 storage precision. For 16-bit
        # gradients its error is below their final storage rounding allowance.
        h = (
            _exp_difference(z, top_w, True)
            if FP32 and Z.dtype.element_ty == tl.float32
            else _exp((z - top_w) * 0.5)
        )
        half = h * root
        gradient = (half * half * sign).to(G.dtype.element_ty)
        if tl.max(pos.to(tl.int32), 0) > 0:
            if anchor == float("inf"):
                positive = ((h.to(tl.float64) * h.to(tl.float64) * rden - share) * w).to(
                    G.dtype.element_ty
                )
            else:
                centered = _centered(h * h, _exp(anchor_w - top_w), z - anchor_w).to(tl.float64)
                positive = ((centered * rden - share) * w).to(G.dtype.element_ty)
            gradient = tl.where(pos, positive, gradient)
        tl.store(G + row * C + k, gradient, valid)


@triton.jit(do_not_specialize=["N"])
def _hinge_forward(Z, M, S, N, C: tl.constexpr, B: tl.constexpr):
    """max(0, 1 + largest negative - least positive score), the first of tied extremes."""
    row = tl.program_id(0).to(tl.int64)
    z1, p1, v1, f1 = _load(Z, M, row, tl.arange(0, B), C, False)
    low, i_low, high, i_high, first, count, nonfinite = _extremes(Z, M, row, z1, p1, v1, f1, C, B)
    value = tl.maximum(_margin(high, low), 0.0)
    code = _code(nonfinite, count, C, True)
    bad = code != 0
    nan = float("nan")
    tl.store(S + row, tl.where(bad, nan, value))
    tl.store(S + N + row, i_low.to(tl.float64))
    tl.store(S + 2 * N + row, i_high.to(tl.float64))
    tl.store(S + 3 * N + row, tl.where(bad, nan, (value > 0).to(tl.float64)))
    tl.store(S + 4 * N + row, code.to(tl.float64))


@triton.jit(do_not_specialize=["N"])
def _hinge_backward(S, U, G, N, C: tl.constexpr, B: tl.constexpr, REDUCTION: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1) * B + tl.arange(0, B)
    i_low = tl.load(S + N + row).to(tl.int32)
    i_high = tl.load(S + 2 * N + row).to(tl.int32)
    w = _weight(U, row, N, REDUCTION) * tl.load(S + 3 * N + row)
    gradient = tl.where(k == i_high, w, tl.where(k == i_low, -w, 0.0))
    tl.store(G + row * C + k, gradient.to(G.dtype.element_ty), k < C)


@triton.jit
def _smooth_margins(z, pos, z0, low, recenter):
    """Margins 1/2 -+ (z - z0), or -(z - low) and 1 + z - low when the largest pairwise
    margin nearly cancels (loss.h multilabel_hinge_loss)."""
    d = z - z0
    plain = tl.where(pos, 0.5 - d, 0.5 + d)
    shifted = tl.where(pos, low - z, _margin(z, low))
    return tl.where(recenter, shifted, plain)


@triton.jit
def _margins_at(
    Z, M, row, start, j, z1, p1, v1, f1, h1, z0, low, recenter, C: tl.constexpr, B: tl.constexpr
):
    """Positive flags, bounds and smooth margins of columns start + j (cached for one chunk)."""
    z, pos, valid, finite = _chunk(Z, M, row, start, j, z1, p1, v1, f1, C, B, False)
    h = h1
    if B < C:
        h = _smooth_margins(z, pos, z0, low, recenter)
    return pos, valid, h


@triton.jit
def _superset(h, side, top, count, total, gamma):
    """Michelot's first superset {g > -gamma} of the gaps g = h - top to a running top."""
    new = tl.maximum(top, tl.max(tl.where(side, h, -float("inf")), 0))
    # A gain of gamma leaves the old gaps all <= -gamma (and a first top resets).
    shift = new - top
    keep = shift < gamma
    total = tl.where(keep, total - count * shift, 0.0)
    count = tl.where(keep, count, 0)
    inside = side & (h - new > -gamma)
    count += tl.sum(inside.to(tl.int32), 0)
    total += tl.sum(tl.where(inside, h - new, 0.0), 0)
    return new, count, total


@triton.jit
def _projection(h, pos, oa, ca, ob, cb, gamma):
    """The projected entries clip(h - origin - correction, 0, gamma) of each side."""
    x = tl.where(pos, (h - oa) - ca, (h - ob) - cb)
    return tl.minimum(tl.maximum(x, 0.0), gamma)


@triton.jit
def _terms(h, x, valid, gamma):
    """Sum of (h x - x^2/2) / gamma, scaled by x / gamma first outside the normal range."""
    hx = h * x
    xx = x * x / 2
    v = tl.abs(hx - xx)
    normal = (v <= 1.7976931348623157e308) & (v >= 2.2250738585072014e-308 / 2.220446049250313e-16)
    term = tl.where(normal, (hx - xx) / gamma, (h - x / 2) * (x / gamma))
    return tl.sum(tl.where(valid & (x > 0), term, 0.0), 0)


@triton.jit(do_not_specialize=["N"])
def _smooth_forward(Z, M, S, GAMMA, N, C: tl.constexpr, B: tl.constexpr, EPS: tl.constexpr):
    """Margins projected onto x >= 0 with equal positive and negative sums <= gamma
    (prox/two_simplex.h): two simplex projections, or a balanced threshold."""
    N = N.to(tl.int64)  # 8 * N reaches 2**31 at ROWS
    row = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, B)
    gamma = tl.load(GAMMA)
    z1, p1, v1, f1 = _load(Z, M, row, j, C, False)
    low, i_low, high, i_high, first, count, nonfinite = _extremes(Z, M, row, z1, p1, v1, f1, C, B)
    code = _code(nonfinite, count, C, True)
    z0 = tl.load(Z + row * C + tl.minimum(first, C - 1)).to(tl.float64)
    recenter = tl.abs((high - low) + 1) < 0.0625 * (1 + tl.abs(low - z0) + tl.abs(high - z0))
    h1 = _smooth_margins(z1, p1, z0, low, recenter)

    # Simplex projections of each side onto sum = gamma, by Michelot from a superset.
    top_a = tl.full([], -float("inf"), tl.float64)
    top_b = tl.full([], -float("inf"), tl.float64)
    n_a = tl.zeros([], tl.int32)
    n_b = tl.zeros([], tl.int32)
    sum_a = tl.zeros([], tl.float64)
    sum_b = tl.zeros([], tl.float64)
    for start in range(0, C, B):
        pos, valid, h = _margins_at(
            Z, M, row, start, j, z1, p1, v1, f1, h1, z0, low, recenter, C, B
        )
        top_a, n_a, sum_a = _superset(h, pos, top_a, n_a, sum_a, gamma)
        top_b, n_b, sum_b = _superset(h, valid & ~pos, top_b, n_b, sum_b, gamma)
    s_a = tl.maximum((sum_a - gamma) / n_a, -gamma)
    s_b = tl.maximum((sum_b - gamma) / n_b, -gamma)
    searching = code == 0
    steps = 0
    while searching:
        n_a = tl.zeros([], tl.int32)
        n_b = tl.zeros([], tl.int32)
        sum_a = tl.zeros([], tl.float64)
        sum_b = tl.zeros([], tl.float64)
        least_a = tl.full([], float("inf"), tl.float64)
        least_b = tl.full([], float("inf"), tl.float64)
        for start in range(0, C, B):
            pos, valid, h = _margins_at(
                Z, M, row, start, j, z1, p1, v1, f1, h1, z0, low, recenter, C, B
            )
            in_a = pos & (h - top_a > s_a)
            in_b = valid & ~pos & (h - top_b > s_b)
            n_a += tl.sum(in_a.to(tl.int32), 0)
            n_b += tl.sum(in_b.to(tl.int32), 0)
            sum_a += tl.sum(tl.where(in_a, h - top_a, 0.0), 0)
            sum_b += tl.sum(tl.where(in_b, h - top_b, 0.0), 0)
            least_a = tl.minimum(least_a, tl.min(tl.where(in_a, h - top_a, float("inf")), 0))
            least_b = tl.minimum(least_b, tl.min(tl.where(in_b, h - top_b, float("inf")), 0))
        s_a = (sum_a - gamma) / n_a
        s_b = (sum_b - gamma) / n_b
        # Each set is stable once its gaps all stay above the new threshold.
        steps += 1
        searching = ((least_a <= s_a) | (least_b <= s_b)) & (steps <= C)

    # The sum is active unless the thresholds add up below zero, to their own rounding.
    terms = tl.maximum(gamma, (tl.abs(top_a) + tl.abs(top_b)) + (tl.abs(s_a) + tl.abs(s_b)))
    active = (top_a + top_b) + (s_a + s_b) >= -EPS * terms
    oa = top_a
    ca = s_a
    ob = top_b
    cb = s_b
    if (~active) & (code == 0):
        # Equal sums below gamma: x = max(0, a - o - t), y = max(0, b + o + t) over the
        # supports, fixing entries as two_simplex.h (Kiwiel) from the simplex supports.
        origin = top_a * 0.5 - top_b * 0.5
        scale = tl.zeros([], tl.float64)
        sum_a = tl.zeros([], tl.float64)
        sum_b = tl.zeros([], tl.float64)
        m = tl.zeros([], tl.int32)
        for start in range(0, C, B):
            pos, valid, h = _margins_at(
                Z, M, row, start, j, z1, p1, v1, f1, h1, z0, low, recenter, C, B
            )
            in_a = pos & (h - top_a > s_a)
            in_b = valid & ~pos & (h - top_b > s_b)
            da = h - origin
            db = h + origin
            scale += tl.sum(tl.where(in_a, tl.abs(da), 0.0) + tl.where(in_b, tl.abs(db), 0.0), 0)
            sum_a += tl.sum(tl.where(in_a, da, 0.0), 0)
            sum_b += tl.sum(tl.where(in_b, db, 0.0), 0)
            m += tl.sum((in_a | in_b).to(tl.int32), 0)
        tol = 8 * EPS * scale
        t = (sum_a - sum_b) / m.to(tl.float64)
        cut_a = tl.full([], -float("inf"), tl.float64)
        cut_b = tl.full([], -float("inf"), tl.float64)
        fixing = m > 0
        fixes = 0
        while fixing:
            sum_x = tl.zeros([], tl.float64)
            sum_y = tl.zeros([], tl.float64)
            n_x = tl.zeros([], tl.int32)
            n_y = tl.zeros([], tl.int32)
            for start in range(0, C, B):
                pos, valid, h = _margins_at(
                    Z, M, row, start, j, z1, p1, v1, f1, h1, z0, low, recenter, C, B
                )
                da = h - origin
                db = h + origin
                in_a = pos & (h - top_a > s_a) & (da > cut_a)
                in_b = valid & ~pos & (h - top_b > s_b) & (db > cut_b)
                out_x = in_a & ~(da > t)
                out_y = in_b & ~(db > -t)
                sum_x += tl.sum(tl.where(out_x, da, 0.0), 0)
                sum_y += tl.sum(tl.where(out_y, db, 0.0), 0)
                n_x += tl.sum(out_x.to(tl.int32), 0)
                n_y += tl.sum(out_y.to(tl.int32), 0)
            infeasible = (sum_x - sum_y) - (n_x + n_y).to(tl.float64) * t
            drop_y = (n_y > 0) & (infeasible > tol)
            drop_x = (~drop_y) & (n_x > 0) & (infeasible < -tol)
            mt = m.to(tl.float64) * t
            following = tl.where(drop_y, mt + sum_y, mt - sum_x)
            cut_a = tl.where(drop_x, tl.maximum(cut_a, t), cut_a)
            cut_b = tl.where(drop_y, tl.maximum(cut_b, -t), cut_b)
            m = tl.where(drop_y, m - n_y, tl.where(drop_x, m - n_x, m))
            t = tl.where((drop_x | drop_y) & (m > 0), following / m.to(tl.float64), t)
            fixes += 1
            fixing = (drop_x | drop_y) & (m > 0) & (fixes <= C)
        oa = origin
        ca = t
        ob = -origin
        cb = -t

    value = tl.zeros([], tl.float64)
    for start in range(0, C, B):
        pos, valid, h = _margins_at(
            Z, M, row, start, j, z1, p1, v1, f1, h1, z0, low, recenter, C, B
        )
        value += _terms(h, _projection(h, pos, oa, ca, ob, cb, gamma), valid, gamma)
    bad = code != 0
    nan = float("nan")
    tl.store(S + row, tl.where(bad, nan, value))
    tl.store(S + N + row, z0)
    tl.store(S + 2 * N + row, low)
    tl.store(S + 3 * N + row, recenter.to(tl.float64))
    tl.store(S + 4 * N + row, tl.where(bad, nan, oa))
    tl.store(S + 5 * N + row, ca)
    tl.store(S + 6 * N + row, ob)
    tl.store(S + 7 * N + row, cb)
    tl.store(S + 8 * N + row, code.to(tl.float64))


@triton.jit(do_not_specialize=["N"])
def _smooth_backward(
    Z, M, S, U, G, GAMMA, N, C: tl.constexpr, B: tl.constexpr, REDUCTION: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1) * B + tl.arange(0, B)
    z, pos, valid, finite = _load(Z, M, row, k, C, False)
    gamma = tl.load(GAMMA)
    z0 = tl.load(S + N + row)
    low = tl.load(S + 2 * N + row)
    recenter = tl.load(S + 3 * N + row) != 0
    oa = tl.load(S + 4 * N + row)
    ca = tl.load(S + 5 * N + row)
    ob = tl.load(S + 6 * N + row)
    cb = tl.load(S + 7 * N + row)
    h = _smooth_margins(z, pos, z0, low, recenter)
    x = _projection(h, pos, oa, ca, ob, cb, gamma) / gamma
    gradient = tl.where(pos, -x, x) * _weight(U, row, N, REDUCTION)
    tl.store(G + row * C + k, gradient.to(G.dtype.element_ty), valid)


def _gamma(device: torch.device, gamma: float) -> torch.Tensor:
    # A float64 device scalar: Triton passes Python floats as float32.
    stream = torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None
    key = (device.index or 0, gamma, stream)
    if key not in _gammas:
        if len(_gammas) > 64:
            _gammas.clear()
        with torch.inference_mode(False):
            _gammas[key] = torch.full((1,), gamma, device=device, dtype=torch.float64)
    return _gammas[key]


def _rows_per_cta(c: int, objective: str) -> int:
    """Rows per program for short cross-entropy rows (c <= 256), else 1."""
    block = max(16, triton.next_power_of_2(c))
    if objective != "cross_entropy" or c > 256 or block > min(_CHUNK, _ENTROPY_CHUNK, _BACKWARD):
        return 1
    return min(32, 512 // block)


def _shape(c: int, objective: str) -> tuple[int, int]:
    """Chunk width and warps of the forward program of a row of c columns."""
    rows = _rows_per_cta(c, objective)
    if rows > 1:
        block = max(16, triton.next_power_of_2(c))
        return block, 4
    chunk = min(_CHUNK, _ENTROPY_CHUNK) if objective == "cross_entropy" else _CHUNK
    block = min(chunk, triton.next_power_of_2(c))
    return block, min(8, max(4, block // 256))


def _backward_block(c: int, objective: str = "") -> int:
    if _rows_per_cta(c, objective) > 1:
        return max(16, triton.next_power_of_2(c))
    return min(_BACKWARD, max(128, triton.next_power_of_2(c)))


def covers(classes: int) -> bool:
    """Whether the backward grid holds the columns (one program per row forward)."""
    return triton.cdiv(classes, _BACKWARD) < 65536


# Columns and warps of _entropy_online's tiles, and the narrowest row it streams.
_ONLINE = (1024, 4)
_ONLINE_MIN = 16384


def _online(c: int, objective: str, fp32: bool) -> bool:
    return objective == "cross_entropy" and fp32 and c >= _ONLINE_MIN


def forward(
    logits: torch.Tensor,
    mask: torch.Tensor,
    objective: str,
    reduction: str,
    gamma: float,
    fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss and row statistics of contiguous (N, C) logits and a boolean mask.

    Invalid CUDA values propagate NaNs without host synchronization. fp32 selects float32
    exponentials in the entropy kernels.
    """
    n, c = logits.shape
    with _cuda_launch.device_guard(logits):
        layout = _LAYOUTS[objective]
        code = _REDUCTIONS[reduction]
        stats = torch.empty(
            _triton_reduce.size(n, layout[3]), device=logits.device, dtype=torch.float64
        )
        dtype = torch.promote_types(logits.dtype, torch.float32)  # loss values: >= float32
        output = torch.empty((n,) if reduction == "none" else (), device=logits.device, dtype=dtype)
        flags = mask.view(torch.uint8)
        block, warps = _shape(c, objective)
        rows = _rows_per_cta(c, objective)
        if _online(c, objective, fp32):
            block, warps = _ONLINE
            _entropy_online[(n,)](logits, flags, stats, n, c, block, num_warps=warps)
        elif objective == "cross_entropy":
            _entropy_forward[(triton.cdiv(n, rows),)](
                logits, flags, stats, n, c, block, fp32, rows, num_warps=warps
            )
        elif objective == "multilabel_hinge":
            _hinge_forward[(n,)](logits, flags, stats, n, c, block, num_warps=warps)
        else:
            eps = torch.finfo(torch.float64).eps
            g = _gamma(logits.device, gamma)
            _smooth_forward[(n,)](logits, flags, stats, g, n, c, block, eps, num_warps=warps)
        _triton_reduce.launch(stats, output, n, code, layout)
        if not logits.is_cuda:
            status = int(stats[layout[3] * n].item())
            if status:
                raise ValueError(MESSAGES[status])
        return output, stats


def backward(
    logits: torch.Tensor,
    mask: torch.Tensor,
    stats: torch.Tensor,
    upstream: torch.Tensor,
    objective: str,
    reduction: str,
    gamma: float,
    fp32: bool = False,
) -> torch.Tensor:
    """Gradient in the logits dtype, scaled by the upstream derivative in float64."""
    n, c = logits.shape
    with _cuda_launch.device_guard(logits):
        code = _REDUCTIONS[reduction]
        gradient = torch.empty_like(logits)
        block = _backward_block(c, objective)
        rows = _rows_per_cta(c, objective)
        grid = (triton.cdiv(n, rows), triton.cdiv(c, block))
        flags = mask.view(torch.uint8)
        u = upstream.contiguous()
        if objective == "cross_entropy":
            _entropy_backward[grid](
                logits, flags, stats, u, gradient, n, c, block, code, fp32, rows, num_warps=4
            )
        elif objective == "multilabel_hinge":
            _hinge_backward[grid](stats, u, gradient, n, c, block, code, num_warps=4)
        else:
            g = _gamma(logits.device, gamma)
            _smooth_backward[grid](
                logits, flags, stats, u, gradient, g, n, c, block, code, num_warps=4
            )
        return gradient


@functools.lru_cache(maxsize=_cuda_launch.CACHE)
def _compiled(
    device: int,
    dtype: torch.dtype,
    c: int,
    code: int,
    objective: str,
    fp32: bool,
) -> torch.Tensor | str:
    """C++ launch handles for (device, dtype, c, reduction, objective, fp32), or why not."""
    with torch.cuda.device(device):
        z = torch.empty((1, 1), device="cuda", dtype=dtype)
        m = torch.empty((1, 1), device="cuda", dtype=torch.uint8)
        s = torch.empty(1, device="cuda", dtype=torch.float64)
        g = torch.empty(1, device="cuda", dtype=torch.float64)
        out_dtype = torch.promote_types(dtype, torch.float32)
        out = torch.empty(1, device="cuda", dtype=out_dtype)
        (block, warps), b_bwd = _shape(c, objective), _backward_block(c, objective)
        rows = _rows_per_cta(c, objective)
        layout = _LAYOUTS[objective]
        f_sig: tuple[int, ...]
        b_sig: tuple[int, ...]
        if objective == "cross_entropy":
            if _online(c, objective, fp32):
                k_fwd = _entropy_online.warmup(
                    z, m, s, 2, c, _ONLINE[0], num_warps=_ONLINE[1], grid=(1,)
                )
            else:
                k_fwd = _entropy_forward.warmup(
                    z, m, s, 2, c, block, fp32, rows, num_warps=warps, grid=(1,)
                )
            k_bwd = _entropy_backward.warmup(
                z, m, s, g, z, 2, c, b_bwd, code, fp32, rows, num_warps=4, grid=(1, 1)
            )
            f_sig, b_sig = (_P, _P, _P, _I), (_P, _P, _P, _P, _P, _I)
        elif objective == "multilabel_hinge":
            k_fwd = _hinge_forward.warmup(z, m, s, 2, c, block, num_warps=warps, grid=(1,))
            k_bwd = _hinge_backward.warmup(s, g, z, 2, c, b_bwd, code, num_warps=4, grid=(1, 1))
            f_sig, b_sig = (_P, _P, _P, _I), (_P, _P, _P, _I)
        else:
            eps = torch.finfo(torch.float64).eps
            k_fwd = _smooth_forward.warmup(z, m, s, g, 2, c, block, eps, num_warps=warps, grid=(1,))
            k_bwd = _smooth_backward.warmup(
                z, m, s, g, z, g, 2, c, b_bwd, code, num_warps=4, grid=(1, 1)
            )
            f_sig, b_sig = (_P, _P, _P, _P, _I), (_P, _P, _P, _P, _P, _P, _I)
        reduce = _triton_reduce.warmup(s, out, code, layout)
        return _cuda_launch.handles(
            [(k_fwd, f_sig)] + [(kernel, (_P, _P, _I)) for kernel in reduce] + [(k_bwd, b_sig)],
            _triton_reduce.BLOCK,
            _triton_reduce.DIRECT_ROWS,
            b_bwd,
            rows,
            rows,
        )


def launcher(
    logits: torch.Tensor, code: int, objective: str, fp32: bool = False
) -> tuple[Any, torch.Tensor | str]:
    """(extension, kernel handles) for the C++ launch, else (None, why not)."""
    host, reason = _cuda_launch.status()
    if host is None:
        return None, reason
    device = logits.device.index or 0
    handles = _compiled(device, logits.dtype, logits.shape[1], code, objective, fp32)
    return (None, handles) if isinstance(handles, str) else (host, handles)


def apply(
    logits: torch.Tensor,
    mask: torch.Tensor,
    objective: str,
    reduction: str,
    gamma: float,
    fp32: bool = False,
) -> torch.Tensor | None:
    """C++ autograd and launches when the launcher runs, else None."""
    code = _REDUCTIONS[reduction]
    host, handles = launcher(logits, code, objective, fp32)
    if host is None:
        return None
    obj_id = 0 if objective == "cross_entropy" else (1 if objective == "multilabel_hinge" else 2)
    return host.multilabel_cuda_apply(
        logits, mask.view(torch.uint8), _gamma(logits.device, gamma), obj_id, code, handles
    )


@triton.jit
def _residuals(
    Z, M, S, N, row, top_w, second_w, index, count, first, nonfinite,
    C: tl.constexpr, B: tl.constexpr,
):  # fmt: skip
    """The FP32-work residual scan of _entropy_forward from a row's float32 extremes,
    storing its statistics. Its nonnegative sums take _pairwise partials; the signed
    differences widen per column."""
    K: tl.constexpr = 8
    R: tl.constexpr = B // K
    j = tl.arange(0, R)[:, None] * K + tl.arange(0, K)[None, :]
    top = top_w.to(tl.float64)
    scale = _exp(second_w.to(tl.float64) - top)
    scale_w = scale.to(tl.float32)
    p = count.to(tl.float64)
    rp_w = (1 / p).to(tl.float32)
    lane_tail = tl.zeros([R], tl.float64)
    lane_negatives = tl.zeros([R], tl.float64)
    gap = tl.zeros([R], tl.float64)
    differences = tl.zeros([R, K], tl.float64)
    anchor_w = tl.load(Z + row * C + tl.minimum(first, C - 1)).to(tl.float32)
    e0 = _exp(anchor_w - top_w)
    # A negative maximum has gradient q_max, so the row gradient norm is at least
    # q_max. Direct positive q - 1/p subtraction then has bounded normwise error;
    # well-fit rows with positive maxima keep the cancellation-resistant formula.
    top_negative = tl.load(M + row * C + tl.minimum(index, C - 1)) == 0
    ordinary = top_negative
    for start in range(0, C, B):
        k = start + j
        z, pos, valid, finite = _load(Z, M, row, k, C, True)
        e = tl.where(valid & (k != index), _exp_difference(z, second_w), 0.0)
        lane_tail += _pairwise(e).to(tl.float64)
        lane_negatives += _pairwise(tl.where(pos, 0.0, e)).to(tl.float64)
        raw_gap = top_w - z
        scaled_gap = tl.where(raw_gap < float("inf"), raw_gap * rp_w, top_w * rp_w - z * rp_w)
        gap += _pairwise(tl.where(pos, scaled_gap, 0.0)).to(tl.float64)
        e_top = tl.where(k == index, 1.0, e * scale_w)
        centered = _centered(e_top, e0, z - anchor_w)
        differences += tl.where(pos, centered, 0.0).to(tl.float64)
    gap, differences = tl.sum(gap, 0), tl.sum(tl.reshape(differences, [B]), 0)
    tail = scale * tl.sum(lane_tail, 0)
    negatives = scale * tl.sum(lane_negatives, 0) + tl.where(top_negative, 1.0, 0.0)
    den = 1 + tail
    anchor = tl.load(Z + row * C + tl.minimum(first, C - 1)).to(tl.float64)
    code = _code(nonfinite, count, C, False)
    bad = code != 0
    nan = float("nan")
    tl.store(S + row, tl.where(bad, nan, gap + _log1p(tail)))
    tl.store(S + N + row, tl.where(bad, nan, top))
    tl.store(S + 2 * N + row, den)
    # Finite logits cannot have an infinite anchor: this marks the direct residual.
    tl.store(S + 3 * N + row, tl.where(ordinary, float("inf"), anchor))
    # The positives' share of the zero sum: (differences + negatives) / (den p).
    tl.store(S + 4 * N + row, tl.where(ordinary, 1 / p, (differences / den + negatives / den) / p))
    tl.store(S + 5 * N + row, code.to(tl.float64))


@triton.jit
def _halve(x):
    a, b = tl.split(tl.reshape(x, [x.shape[0], x.shape[1] // 2, 2]))
    return a + b


@triton.jit
def _pairwise(x):
    """Row sums of x [R, 8] by halving: three roundings."""
    return tl.reshape(_halve(_halve(_halve(x))), [x.shape[0]])


@triton.jit
def _online_tile(
    Z, M, row, k, C: tl.constexpr, MASKED: tl.constexpr,
    lane_top, lane_second, lane_index, lane_count, lane_first, check, total, gap, top,
):  # fmt: skip
    """_entropy_online's state after the columns k (MASKED: some at or beyond C)."""
    if MASKED:
        valid = k < C
        z = tl.load(Z + row * C + k, valid, other=0).to(tl.float32)
        pos = tl.load(M + row * C + k, valid, other=0) != 0
        zm = tl.where(valid, z, -float("inf"))
    else:
        z = tl.load(Z + row * C + k).to(tl.float32)
        pos = tl.load(M + row * C + k) != 0
        zm = z
    # z * 0 is NaN exactly for nonfinite z. Maxima skip NaNs; the check marks the row.
    check += z * 0.0
    rises = zm > lane_top
    lane_second = tl.where(rises, lane_top, tl.maximum(lane_second, zm))
    lane_index = tl.where(rises, k, lane_index)
    lane_top = tl.where(rises, zm, lane_top)
    new = tl.maximum(top, tl.max(tl.max(zm, 1), 0))
    if new > top:
        old = top.to(tl.float64)
        shift = new.to(tl.float64) - old
        counts = tl.sum(lane_count, 1)
        gap += tl.where(counts > 0, counts.to(tl.float64) * shift, 0.0)
        total = total * _exp(old - new.to(tl.float64))
        top = new
    e = _exp(zm - top) if MASKED else _exp32(zm - top)
    total += _pairwise(e).to(tl.float64)
    gap += _pairwise(tl.where(pos, top - z, 0.0)).to(tl.float64)
    lane_count += pos.to(tl.int32)
    lane_first = tl.minimum(lane_first, tl.where(pos, k, C))
    return lane_top, lane_second, lane_index, lane_count, lane_first, check, total, gap, top


@triton.jit(do_not_specialize=["N"])
def _entropy_online(Z, M, S, N, C: tl.constexpr, B: tl.constexpr):
    """One scan of FP32-work rows whose maximum is a negative class, else _residuals.

    The exponential sum and the positives' gaps follow the running maximum, rescaled by
    float64 factors when it rises. Such a row's loss is at least log 2 (every positive q is
    at most q_max <= 1/2 of the mass), so log(den) needs no tail beside the maximum.
    Each thread sums 8 adjacent columns pairwise in float32 before widening the sums (three
    roundings of nonnegative terms). A gap beyond float32 range sends the row to _residuals.
    """
    row = tl.program_id(0).to(tl.int64)
    K: tl.constexpr = 8
    R: tl.constexpr = B // K
    FULL: tl.constexpr = C // B * B
    j = tl.arange(0, R)[:, None] * K + tl.arange(0, K)[None, :]
    neg_inf: tl.constexpr = -float("inf")
    # Per lane: the top with its first column, the runner-up (a tie included), the
    # positives, and NaN once a score is nonfinite; per row of lanes, the exponential sum
    # and the positives' gaps below the running top.
    lane_top = tl.full([R, K], neg_inf, tl.float32)
    lane_second = lane_top
    lane_index = tl.full([R, K], C, tl.int32)
    lane_count = tl.zeros([R, K], tl.int32)
    lane_first = tl.full([R, K], C, tl.int32)
    check = tl.zeros([R, K], tl.float32)
    total = tl.zeros([R], tl.float64)
    gap = tl.zeros([R], tl.float64)
    top = tl.full([], neg_inf, tl.float32)
    for start in range(0, FULL, B):
        lane_top, lane_second, lane_index, lane_count, lane_first, check, total, gap, top = (
            _online_tile(
                Z, M, row, start + j, C, False,
                lane_top, lane_second, lane_index, lane_count, lane_first, check, total, gap, top,
            )
        )  # fmt: skip
    if FULL < C:
        lane_top, lane_second, lane_index, lane_count, lane_first, check, total, gap, top = (
            _online_tile(
                Z, M, row, FULL + j, C, True,
                lane_top, lane_second, lane_index, lane_count, lane_first, check, total, gap, top,
            )
        )  # fmt: skip
    lane_top = tl.reshape(lane_top, [B])
    lane_second = tl.reshape(lane_second, [B])
    lane_index = tl.reshape(lane_index, [B])
    lane_count = tl.reshape(lane_count, [B])
    lane_first = tl.reshape(lane_first, [B])
    gap = tl.sum(gap, 0)
    index = tl.min(tl.where(lane_top == top, lane_index, C), 0)
    others_top = tl.max(tl.where(lane_index == index, neg_inf, lane_top), 0)
    second = tl.maximum(tl.max(lane_second, 0), others_top)
    count = tl.sum(lane_count, 0)
    first = tl.min(lane_first, 0)
    nonfinite = (tl.sum(tl.reshape(check, [B]), 0) != 0).to(tl.int32)
    done = (tl.load(M + row * C + tl.minimum(index, C - 1)) == 0) & (gap < float("inf"))
    if done:
        den = tl.sum(total, 0)
        p = count.to(tl.float64)
        code = _code(nonfinite, count, C, False)
        bad = code != 0
        nan = float("nan")
        tl.store(S + row, tl.where(bad, nan, gap / p + tl.log(den)))
        tl.store(S + N + row, tl.where(bad, nan, top.to(tl.float64)))
        tl.store(S + 2 * N + row, den)
        tl.store(S + 3 * N + row, float("inf"))
        tl.store(S + 4 * N + row, 1 / p)
        tl.store(S + 5 * N + row, code.to(tl.float64))
    else:
        _residuals(Z, M, S, N, row, top, second, index, count, first, nonfinite, C, B)
