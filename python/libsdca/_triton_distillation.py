"""Fused CUDA distillation_loss and proximal_distillation_loss in Triton.

One program per example holds its row (up to 4096 classes) in registers, or streams
tiles of wider rows, a pass per statistic (and per proximal step), in working precision
(float32 or float64). Losses are float32 (float64 for float64 logits). distillation_loss
saves row statistics and recomputes its gradient in backward, scaled by the upstream
derivative and rounded once into the student's dtype; proximal_distillation_loss saves its
gradient in promote(student's dtype, float32), which backward scales before rounding. The
proximal kernel adds the safeguarded Halley solve of _tensor_prox.proximal_distillation on
the residual helpers of _triton_prox.
"""

from __future__ import annotations

import functools
import math
import os
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from . import _cuda_launch, _triton_reduce
from ._first_order import first_order
from ._triton_prox import _lambert_w_exp, _log, _near_residual, _probability

_CHUNK = 4096
# Wider proximal rows stream tiles of _STREAM_BLOCK columns.
_STREAM_BLOCK, _STREAM_WARPS = 2048, 16

if os.environ.get("TRITON_INTERPRET") == "1":

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

else:

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
def _bounded_expm1(x):
    """expm1 on [-2, 2]; float32 fits (expm1(x) - x) / x**2, float64 keeps libdevice."""
    if x.dtype == tl.float32:
        # Degree-8 fit, then x * (1 + x * p(x)); exact signed zero and tiny x.
        p = tl.full(x.shape, 2.9499494758056244e-07, tl.float32)
        p = tl.fma(p, x, 2.9900852496211883e-06)
        p = tl.fma(p, x, 2.4742717869230546e-05)
        p = tl.fma(p, x, 0.00019770115613937378)
        p = tl.fma(p, x, 0.0013889545807614923)
        p = tl.fma(p, x, 0.008334128186106682)
        p = tl.fma(p, x, 0.04166664555668831)
        p = tl.fma(p, x, 0.1666664332151413)
        p = tl.fma(p, x, 0.5)
        return x * tl.fma(x, p, 1.0)
    else:
        return _expm1(x)


@triton.jit
def _finite(x):
    return (x - x) == 0


@triton.jit
def _work(x, FP32: tl.constexpr):
    return x.to(tl.float32 if FP32 else tl.float64)


@triton.jit
def _temp_diff(v, origin, temp, LARGE_T: tl.constexpr):
    diff = v - origin
    scaled = diff / temp
    if LARGE_T:
        return tl.where(_finite(diff), scaled, (v / temp) - (origin / temp))
    return scaled


@triton.jit
def _lognorm_subnormal(top, sum_e, EPS: tl.constexpr):
    half = _exp(top * 0.5)
    tail = half * (half * sum_e)
    return tl.where(tail < EPS * 0.5, tail, _log1p(tail))


@triton.jit
def _load_pair(S_ptr, T_ptr, offsets, valid, FP32: tl.constexpr):
    """Student and teacher logits in working precision (nonfinite ones as 0) and the
    nonfinite count."""
    cs = _work(tl.load(S_ptr + offsets, mask=valid, other=0.0), FP32)
    ct = _work(tl.load(T_ptr + offsets, mask=valid, other=0.0), FP32)
    bad = tl.sum((valid & (~_finite(cs) | ~_finite(ct))).to(tl.int32), 0)
    return tl.where(valid & _finite(cs), cs, 0.0), tl.where(valid & _finite(ct), ct, 0.0), bad


@triton.jit
def _scaled(cs, ct, valid, s_max, t_max, temp, LARGE_T: tl.constexpr):
    """Centered, temperature-scaled student and teacher logits (-inf off the row)."""
    neg_inf: tl.constexpr = -float("inf")
    s = tl.where(valid, _temp_diff(cs, s_max, temp, LARGE_T), neg_inf)
    t = tl.where(valid, _temp_diff(ct, t_max, temp, LARGE_T), neg_inf)
    return s, t


@triton.jit
def _offset_delta(cs, ct, s_max, t_max):
    """Difference of centered logits, including both subtraction rounding errors."""
    a = cs - s_max
    b = ct - t_max
    ea = (cs - (a - (a - cs))) + (-s_max - (a - cs))
    eb = (ct - (b - (b - ct))) + (-t_max - (b - ct))
    return (a - b) + (ea - eb)


@triton.jit
def _dn(cs, ct, s, t, s_max, t_max, mean_delta, rel_mean, over, temp, LARGE_T: tl.constexpr):
    """The centered scaled difference of the represented logits."""
    dn = (_offset_delta(cs, ct, s_max, t_max) - mean_delta) / temp
    if LARGE_T:
        dn = tl.where(over, (s - t) - rel_mean, dn)
    return dn


@triton.jit
def _correction_term(dn, p, s, shift):
    """exp(log(p) + dn) - p, from the student logits where |dn| >= 1."""
    clamped_dn = tl.minimum(tl.maximum(dn, -1.0), 1.0)
    return tl.where(tl.abs(dn) < 1.0, p * _bounded_expm1(clamped_dn), _exp(s + shift) - p)


@triton.jit
def _shift(s_max, t_max, mean_delta, rel_mean, over, t_lognorm, temp, LARGE_T: tl.constexpr):
    shift = (-mean_delta / temp) - t_lognorm
    if LARGE_T:
        shift = tl.where(over, -rel_mean - t_lognorm, shift)
    return shift


@triton.jit
def _u(logratio, p, q, valid):
    """q - p, as p expm1(log(q / p)) where the log-ratio is small."""
    clamped_lr = tl.minimum(tl.maximum(logratio, -1.0), 1.0)
    u = tl.where(tl.abs(logratio) < 1.0, p * _bounded_expm1(clamped_lr), q - p)
    return tl.where(valid, u, 0.0)


@triton.jit
def _stats(
    S_ptr,
    T_ptr,
    temp,
    row,
    C: tl.constexpr,
    B: tl.constexpr,
    FP32: tl.constexpr,
    LARGE_T: tl.constexpr,
    EPS: tl.constexpr,
):
    """Invalid row, loss, p, q, log q, log p, log(q / p) and q - p, as torch._distillation_stats."""
    j = tl.arange(0, B)
    valid = j < C
    cs, ct, bad = _load_pair(S_ptr, T_ptr, row * C + j, valid, FP32)
    bad = bad > 0

    neg_inf: tl.constexpr = -float("inf")
    s_max = tl.max(tl.where(valid, cs, neg_inf), 0)
    t_max = tl.max(tl.where(valid, ct, neg_inf), 0)
    s, t = _scaled(cs, ct, valid, s_max, t_max, temp, LARGE_T)

    s_idx = tl.argmax(s, 0)
    t_idx = tl.argmax(t, 0)
    s_other = tl.where(j == s_idx, neg_inf, s)
    t_other = tl.where(j == t_idx, neg_inf, t)
    s_top = tl.max(s_other, 0)
    t_top = tl.max(t_other, 0)
    s_orig = tl.where(_finite(s_top), s_top, 0.0)
    t_orig = tl.where(_finite(t_top), t_top, 0.0)
    s_lognorm = _lognorm_subnormal(s_top, tl.sum(_exp(s_other - s_orig), 0), EPS)
    t_lognorm = _lognorm_subnormal(t_top, tl.sum(_exp(t_other - t_orig), 0), EPS)

    es = _exp(s)
    et = _exp(t)
    q = es / tl.sum(es, 0)
    p = et / tl.sum(et, 0)
    value = tl.sum(tl.where(valid & (p > 0), p * ((0.0 - s) + s_lognorm), 0.0), 0)

    p_sum = tl.sum(tl.where(valid & (p > 0), p, 0.0), 0)
    mean_delta = (
        tl.sum(tl.where(valid & (p > 0), p * _offset_delta(cs, ct, s_max, t_max), 0.0), 0) / p_sum
    )
    rel_mean = mean_delta
    if LARGE_T:
        rel_mean = tl.sum(tl.where(valid & (p > 0), p * (s - t), 0.0), 0) / p_sum
    over = ~_finite(mean_delta)
    dn = _dn(cs, ct, s, t, s_max, t_max, mean_delta, rel_mean, over, temp, LARGE_T)
    shift = _shift(s_max, t_max, mean_delta, rel_mean, over, t_lognorm, temp, LARGE_T)
    correction = tl.sum(tl.where(valid, _correction_term(dn, p, s, shift), 0.0), 0)
    logratio = dn - _log1p(correction)
    u = _u(logratio, p, q, valid)
    return bad, value, p, q, s - s_lognorm, t - t_lognorm, logratio, u


# distillation_loss saves row statistics, not its gradient: float64 slots of N rows each
# hold the loss, an invalid flag, the maxima, exponential sums and log-normalizers of s
# and t, the mean differences, the correction, sum(u) and the pivot, then the status word
# and the reduction partials (_triton_reduce).
_LAYOUT = (0, -1, 1, 13)
_SLOTS = _LAYOUT[3]


@triton.jit
def _register_lognorm(s, j, EPS: tl.constexpr):
    """log(sum(exp(s))) of a row with max(s) = 0, from its tail beside the first maximum."""
    other = tl.where(j == tl.argmax(s, 0), -float("inf"), s)
    top = tl.max(other, 0)
    origin = tl.where(_finite(top), top, 0.0)
    return _lognorm_subnormal(top, tl.sum(_exp(other - origin), 0), EPS)


@triton.jit
def _store_row(
    W, N, bad, value, s_max, t_max, s_sum, t_sum, s_lognorm, t_lognorm, mean_delta, rel_mean,
    correction, u_sum, pivot, mask=None,
):  # fmt: skip
    """Store a row's loss (NaN for an invalid row) and statistics at W = stats + row."""
    N = N.to(tl.int64)  # 12 * N exceeds int32 within ROWS
    tl.store(W, tl.where(bad, float("nan"), value).to(tl.float64), mask=mask)
    tl.store(W + N, bad.to(tl.float64), mask=mask)
    tl.store(W + 2 * N, s_max.to(tl.float64), mask=mask)
    tl.store(W + 3 * N, t_max.to(tl.float64), mask=mask)
    tl.store(W + 4 * N, s_sum.to(tl.float64), mask=mask)
    tl.store(W + 5 * N, t_sum.to(tl.float64), mask=mask)
    tl.store(W + 6 * N, s_lognorm.to(tl.float64), mask=mask)
    tl.store(W + 7 * N, t_lognorm.to(tl.float64), mask=mask)
    tl.store(W + 8 * N, mean_delta.to(tl.float64), mask=mask)
    tl.store(W + 9 * N, rel_mean.to(tl.float64), mask=mask)
    tl.store(W + 10 * N, correction.to(tl.float64), mask=mask)
    tl.store(W + 11 * N, u_sum.to(tl.float64), mask=mask)
    tl.store(W + 12 * N, pivot.to(tl.float64), mask=mask)


@triton.jit
def _register_rows(
    S_ptr, T_ptr, ST_ptr, N, temp, C: tl.constexpr, B: tl.constexpr, FP32: tl.constexpr,
    LARGE_T: tl.constexpr, EPS: tl.constexpr, VALUE_ONLY: tl.constexpr, ROWS: tl.constexpr,
):  # fmt: skip
    """Loss and statistics of ROWS rows held in registers [ROWS, B]."""
    neg_inf: tl.constexpr = -float("inf")
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    row_valid = row < N
    j = tl.arange(0, B)[None, :]
    valid = row_valid[:, None] & (j < C)
    offsets = row[:, None] * C + j
    cs = _work(tl.load(S_ptr + offsets, mask=valid, other=0.0), FP32)
    ct = _work(tl.load(T_ptr + offsets, mask=valid, other=0.0), FP32)
    bad = tl.sum((valid & (~_finite(cs) | ~_finite(ct))).to(tl.int32), 1) > 0
    cs = tl.where(valid & _finite(cs), cs, 0.0)
    ct = tl.where(valid & _finite(ct), ct, 0.0)
    s_max = tl.max(tl.where(valid, cs, neg_inf), 1)
    t_max = tl.max(tl.where(valid, ct, neg_inf), 1)
    s, t = _scaled(cs, ct, valid, s_max[:, None], t_max[:, None], temp, LARGE_T)
    s_idx = tl.argmax(s, 1)[:, None]
    s_other = tl.where(j == s_idx, neg_inf, s)
    s_top = tl.max(s_other, 1)
    s_orig = tl.where(_finite(s_top), s_top, 0.0)
    s_lognorm = _lognorm_subnormal(s_top, tl.sum(_exp(s_other - s_orig[:, None]), 1), EPS)
    et = _exp(t)
    t_sum = tl.sum(et, 1)
    p = et / t_sum[:, None]
    mass = valid & (p > 0)
    value = tl.sum(tl.where(mass, p * ((0.0 - s) + s_lognorm[:, None]), 0.0), 1)
    if VALUE_ONLY:
        tl.store(ST_ptr + row, tl.where(bad, float("nan"), value).to(tl.float64), mask=row_valid)
        tl.store(ST_ptr + N.to(tl.int64) + row, bad.to(tl.float64), mask=row_valid)
    else:
        es = _exp(s)
        s_sum = tl.sum(es, 1)
        p_sum = tl.sum(tl.where(mass, p, 0.0), 1)
        delta = _offset_delta(cs, ct, s_max[:, None], t_max[:, None])
        mean_delta = tl.sum(tl.where(mass, p * delta, 0.0), 1) / p_sum
        dn = (delta - mean_delta[:, None]) / temp
        simple = row < 0
        pivot = s_max.to(tl.float32) * 0.0
        u_simple = s_max * 0.0
        if C > 16:
            low = (tl.min(tl.where(valid, delta, float("inf")), 1) - mean_delta) / temp
            high = (tl.max(tl.where(valid, delta, -float("inf")), 1) - mean_delta) / temp
            simple = _finite(mean_delta) & (low >= -2) & (high <= 2)
            s_min = tl.min(tl.where(valid, cs, float("inf")), 1)
            t_min = tl.min(tl.where(valid, ct, float("inf")), 1)
            ranged = _finite((s_max - s_min) + (t_max - t_min))
            pivot = 0.5 * ranged.to(tl.float32) - simple.to(tl.float32)
            u_simple = tl.sum(tl.where(valid, p * _bounded_expm1(dn), 0.0), 1)
        if C > 16 and tl.max((~simple).to(tl.int32), 0) == 0:
            _store_row(
                ST_ptr + row, N, bad, value, s_max, t_max, s_sum, t_sum, s_lognorm, s_lognorm,
                mean_delta, mean_delta, s_max * 0.0, u_simple, pivot, row_valid,
            )  # fmt: skip
        else:
            t_idx = tl.argmax(t, 1)[:, None]
            t_other = tl.where(j == t_idx, neg_inf, t)
            t_top = tl.max(t_other, 1)
            t_orig = tl.where(_finite(t_top), t_top, 0.0)
            t_lognorm = _lognorm_subnormal(t_top, tl.sum(_exp(t_other - t_orig[:, None]), 1), EPS)
            q = es / s_sum[:, None]
            rel_mean = mean_delta
            if LARGE_T:
                rel_mean = tl.sum(tl.where(mass, p * (s - t), 0.0), 1) / p_sum
            over = ~_finite(mean_delta)
            if LARGE_T:
                dn = tl.where(over[:, None], (s - t) - rel_mean[:, None], dn)
            shift = _shift(s_max, t_max, mean_delta, rel_mean, over, t_lognorm, temp, LARGE_T)
            correction = tl.sum(tl.where(valid, _correction_term(dn, p, s, shift[:, None]), 0.0), 1)
            u_sum = tl.sum(_u(dn - _log1p(correction)[:, None], p, q, valid), 1)
            if C <= 16:
                pivot = tl.argmax(tl.where(valid, p + q, neg_inf), 1).to(tl.float32)
            else:
                u_sum = tl.where(simple, u_simple, u_sum)
                correction = tl.where(simple, 0.0, correction)
            _store_row(
                ST_ptr + row, N, bad, value, s_max, t_max, s_sum, t_sum, s_lognorm, t_lognorm,
                mean_delta, rel_mean, correction, u_sum, pivot, row_valid,
            )  # fmt: skip


@triton.jit
def _register_row(
    S_ptr,
    T_ptr,
    W,
    N,
    temp,
    row,
    C: tl.constexpr,
    B: tl.constexpr,
    FP32: tl.constexpr,
    LARGE_T: tl.constexpr,
    EPS: tl.constexpr,
    VALUE_ONLY: tl.constexpr = False,
):
    """Loss and statistics of a row held in registers, as _stats."""
    neg_inf: tl.constexpr = -float("inf")
    j = tl.arange(0, B)
    valid = j < C
    cs, ct, bad = _load_pair(S_ptr, T_ptr, row * C + j, valid, FP32)
    s_max = tl.max(tl.where(valid, cs, neg_inf), 0)
    t_max = tl.max(tl.where(valid, ct, neg_inf), 0)
    s, t = _scaled(cs, ct, valid, s_max, t_max, temp, LARGE_T)
    s_lognorm = _register_lognorm(s, j, EPS)
    et = _exp(t)
    t_sum = tl.sum(et, 0)
    p = et / t_sum
    mass = valid & (p > 0)
    value = tl.sum(tl.where(mass, p * ((0.0 - s) + s_lognorm), 0.0), 0)
    if VALUE_ONLY:
        tl.store(W, tl.where(bad > 0, float("nan"), value).to(tl.float64))
        tl.store(W + N.to(tl.int64), (bad > 0).to(tl.float64))
    else:
        p_sum = tl.sum(tl.where(mass, p, 0.0), 0)
        delta = _offset_delta(cs, ct, s_max, t_max)
        mean_delta = tl.sum(tl.where(mass, p * delta, 0.0), 0) / p_sum
        es = _exp(s)
        s_sum = tl.sum(es, 0)
        if C <= 16:
            t_lognorm = _register_lognorm(t, j, EPS)
            q = es / s_sum
            rel_mean = mean_delta
            if LARGE_T:
                rel_mean = tl.sum(tl.where(mass, p * (s - t), 0.0), 0) / p_sum
            over = ~_finite(mean_delta)
            dn = _dn(cs, ct, s, t, s_max, t_max, mean_delta, rel_mean, over, temp, LARGE_T)
            shift = _shift(s_max, t_max, mean_delta, rel_mean, over, t_lognorm, temp, LARGE_T)
            correction = tl.sum(tl.where(valid, _correction_term(dn, p, s, shift), 0.0), 0)
            u_sum = tl.sum(_u(dn - _log1p(correction), p, q, valid), 0)
            pivot = tl.argmax(tl.where(valid, p + q, neg_inf), 0).to(tl.float32)
        else:
            low = (tl.min(tl.where(valid, delta, float("inf")), 0) - mean_delta) / temp
            high = (tl.max(tl.where(valid, delta, -float("inf")), 0) - mean_delta) / temp
            simple = _finite(mean_delta) & (low >= -2) & (high <= 2)
            s_min = tl.min(tl.where(valid, cs, float("inf")), 0)
            t_min = tl.min(tl.where(valid, ct, float("inf")), 0)
            ranged = _finite((s_max - s_min) + (t_max - t_min))
            pivot = 0.5 * ranged.to(tl.float32) - simple.to(tl.float32)
            if simple:
                t_lognorm = temp * 0
                rel_mean = mean_delta
                correction = temp * 0
                dn = (delta - mean_delta) / temp
                u_sum = tl.sum(tl.where(valid, p * _bounded_expm1(dn), 0.0), 0)
            else:
                t_lognorm = _register_lognorm(t, j, EPS)
                q = es / s_sum
                rel_mean = mean_delta
                if LARGE_T:
                    rel_mean = tl.sum(tl.where(mass, p * (s - t), 0.0), 0) / p_sum
                over = ~_finite(mean_delta)
                dn = _dn(cs, ct, s, t, s_max, t_max, mean_delta, rel_mean, over, temp, LARGE_T)
                shift = _shift(s_max, t_max, mean_delta, rel_mean, over, t_lognorm, temp, LARGE_T)
                correction = tl.sum(tl.where(valid, _correction_term(dn, p, s, shift), 0.0), 0)
                u_sum = tl.sum(_u(dn - _log1p(correction), p, q, valid), 0)
        _store_row(
            W, N, bad > 0, value, s_max, t_max, s_sum, t_sum, s_lognorm, t_lognorm, mean_delta,
            rel_mean, correction, u_sum, pivot,
        )  # fmt: skip


@triton.jit
def _value_tile(
    S_ptr, T_ptr, base, start, C, rtemp, s_max, t_max, rs, rt, s_lognorm, mean_delta,
    rel_mean, over, log1p_c, simple, B: tl.constexpr, FP32: tl.constexpr,
    LARGE_T: tl.constexpr, MASKED: tl.constexpr, VALUE_ONLY: tl.constexpr,
):  # fmt: skip
    """A lean tile's terms p log q of the value and its residuals u."""
    k, valid, p, s, logratio = _lean_tile(
        S_ptr, T_ptr, base, start, C, rtemp, s_max, t_max, rt, mean_delta, rel_mean, over,
        log1p_c, B, FP32, LARGE_T, MASKED,
    )  # fmt: skip
    value = tl.where(p > 0, p * (s - s_lognorm), 0.0)
    u = value * 0
    if not VALUE_ONLY:
        # Let d be the centered, unnormalized log-ratio. Then
        # u = p expm1(d) = q (1 + sum(u)) - p, so u - q sum(u) is q - p.
        # Bounding |d| by 2 keeps this cancellation-resistant residual well scaled.
        if simple:
            u = p * _bounded_expm1(logratio)
            if MASKED:
                u = tl.where(valid, u, 0.0)
        else:
            es = _exp32(s) if FP32 and not MASKED else _exp(s)
            u = _u(logratio, p, es * rs, valid)
    return value, u


@triton.jit
def _stream_value(
    S_ptr, T_ptr, base, C: tl.constexpr, temp, s_max, t_max, s_sum, t_sum, s_lognorm,
    t_lognorm, mean_delta, rel_mean, simple, B: tl.constexpr, FP32: tl.constexpr,
    LARGE_T: tl.constexpr, VALUE_ONLY: tl.constexpr,
):  # fmt: skip
    """The value, sum(u) (float64 lane sums) and the correction of a streamed row."""
    over = ~_finite(mean_delta)
    correction = temp * 0
    if not VALUE_ONLY:
        if ~simple:
            correction = _stream_correction(
                S_ptr, T_ptr, base, C, temp, s_max, t_max, t_sum, t_lognorm, mean_delta,
                rel_mean, over, B, FP32, LARGE_T,
            )  # fmt: skip
    # Wide rows form every gradient entry independently, so no pivot is needed.
    rtemp, rs, rt = _reciprocal(temp), _reciprocal(s_sum), _reciprocal(t_sum)
    log1p_c = _log1p(correction)
    value = tl.zeros([B], tl.float64)
    u_sum = value
    FULL: tl.constexpr = C // B * B
    for start in range(0, FULL, B):
        v, u = _value_tile(
            S_ptr, T_ptr, base, start, C, rtemp, s_max, t_max, rs, rt, s_lognorm, mean_delta,
            rel_mean, over, log1p_c, simple, B, FP32, LARGE_T, False, VALUE_ONLY,
        )  # fmt: skip
        value -= v.to(tl.float64)
        u_sum += u.to(tl.float64)
    if FULL < C:
        v, u = _value_tile(
            S_ptr, T_ptr, base, FULL, C, rtemp, s_max, t_max, rs, rt, s_lognorm, mean_delta,
            rel_mean, over, log1p_c, simple, B, FP32, LARGE_T, True, VALUE_ONLY,
        )  # fmt: skip
        value -= v.to(tl.float64)
        u_sum += u.to(tl.float64)
    return tl.sum(value, 0), u_sum, correction


@triton.jit
def _streamed_row(
    S_ptr,
    T_ptr,
    W,
    N,
    temp,
    row,
    C: tl.constexpr,
    B: tl.constexpr,
    FP32: tl.constexpr,
    LARGE_T: tl.constexpr,
    EPS: tl.constexpr,
    VALUE_ONLY: tl.constexpr = False,
):
    """Streamed loss and optional backward statistics, sharing the proximal row helpers.

    Value-only mode leaves derivative statistics unused so the compiler can omit their work.
    """
    base = row * C
    if VALUE_ONLY and FP32:
        # FP64 work keeps the normalized-per-term scan: its unnormalized sum could overflow.
        bad, _sm, _tm, _ss, _ts, _sln, _tln, value = _stream_normalizers(
            S_ptr, T_ptr, base, C, temp, B, FP32, LARGE_T, EPS, False, True, True
        )
    else:
        (
            bad, s_max, t_max, s_sum, t_sum, s_lognorm, t_lognorm, mean_delta, rel_mean, simple,
            ranged,
        ) = _stream_normalizers(
            S_ptr, T_ptr, base, C, temp, B, FP32, LARGE_T, EPS, True, False, True
        )  # fmt: skip
        if LARGE_T:
            if ranged:
                value, u_sum, correction = _stream_value(
                    S_ptr, T_ptr, base, C, temp, s_max, t_max, s_sum, t_sum, s_lognorm,
                    t_lognorm, mean_delta, rel_mean, simple, B, FP32, False, VALUE_ONLY,
                )  # fmt: skip
            else:
                value, u_sum, correction = _stream_value(
                    S_ptr, T_ptr, base, C, temp, s_max, t_max, s_sum, t_sum, s_lognorm,
                    t_lognorm, mean_delta, rel_mean, simple, B, FP32, True, VALUE_ONLY,
                )  # fmt: skip
        else:
            value, u_sum, correction = _stream_value(
                S_ptr, T_ptr, base, C, temp, s_max, t_max, s_sum, t_sum, s_lognorm, t_lognorm,
                mean_delta, rel_mean, simple, B, FP32, False, VALUE_ONLY,
            )  # fmt: skip
        if not VALUE_ONLY:
            u_sum = tl.sum(u_sum, 0)
            # Wide rows do not use a pivot: its slot records which residual backward recomputes
            # (negative for simple rows), and a half that no large-temperature fallback is
            # needed.
            pivot = 0.5 * ranged.to(tl.float32) - simple.to(tl.float32)
            _store_row(
                W, N, bad, value, s_max, t_max, s_sum, t_sum, s_lognorm, t_lognorm, mean_delta,
                rel_mean, correction, u_sum, pivot,
            )  # fmt: skip
    if VALUE_ONLY:
        tl.store(W, tl.where(bad, float("nan"), value))
        tl.store(W + N.to(tl.int64), bad.to(tl.float64))


@triton.jit(do_not_specialize=["N"])
def _forward(
    S_ptr,
    T_ptr,
    TEMP_ptr,
    ST_ptr,
    N,
    C: tl.constexpr,
    B: tl.constexpr,
    FP32: tl.constexpr,
    LARGE_T: tl.constexpr,
    EPS: tl.constexpr,
    VALUE_ONLY: tl.constexpr = False,
    ROWS: tl.constexpr = 1,
):
    dtype: tl.constexpr = tl.float32 if FP32 else tl.float64
    temp = tl.load(TEMP_ptr).to(dtype)
    if ROWS > 1:
        _register_rows(
            S_ptr, T_ptr, ST_ptr, N.to(tl.int64), temp, C, B, FP32, LARGE_T, EPS, VALUE_ONLY, ROWS
        )
    else:
        row = tl.program_id(0).to(tl.int64)
        if B < C:
            _streamed_row(
                S_ptr, T_ptr, ST_ptr + row, N.to(tl.int64), temp, row, C, B, FP32, LARGE_T,
                EPS, VALUE_ONLY,
            )  # fmt: skip
        else:
            _register_row(
                S_ptr, T_ptr, ST_ptr + row, N.to(tl.int64), temp, row, C, B, FP32, LARGE_T,
                EPS, VALUE_ONLY,
            )  # fmt: skip


@triton.jit
def _slot(W, N, slot: tl.constexpr, dtype: tl.constexpr):
    N = N.to(tl.int64)
    return tl.load(W + slot * N).to(dtype)


@triton.jit
def _gradient_tile(
    S_ptr, T_ptr, G_ptr, base, start, C, rtemp, s_max, t_max, rs, rt, mean_delta, rel_mean,
    over, log1p_c, simple, ratio, u_sum, bad, weight, B: tl.constexpr, FP32: tl.constexpr,
    LARGE_T: tl.constexpr, MASKED: tl.constexpr,
):  # fmt: skip
    """Store a lean tile's gradient (u - q sum(u)) / T, scaled by the upstream weight. Simple
    rows with a positive ratio form q as (p + u) ratio, without exp(s)."""
    k, valid, p, s, logratio = _lean_tile(
        S_ptr, T_ptr, base, start, C, rtemp, s_max, t_max, rt, mean_delta, rel_mean, over,
        log1p_c, B, FP32, LARGE_T, MASKED,
    )  # fmt: skip
    if ratio > 0:
        u = p * _bounded_expm1(logratio)
        q = (p + u) * ratio
    else:
        es = _exp32(s) if FP32 and not MASKED else _exp(s)
        q = es * rs
        if simple:
            u = p * _bounded_expm1(logratio)
        else:
            u = _u(logratio, p, q, valid)
    g = tl.where(bad, float("nan"), ((u - q * u_sum) * rtemp) * weight)
    g = g.to(G_ptr.dtype.element_ty)
    if MASKED:
        tl.store(G_ptr + base + k, g, mask=valid)
    else:
        tl.store(G_ptr + base + k, g)


@triton.jit
def _gradient_scan(
    S_ptr, T_ptr, G_ptr, base, C: tl.constexpr, temp, s_max, t_max, s_sum, t_sum, mean_delta,
    rel_mean, over, correction, simple, u_sum, bad, weight, B: tl.constexpr,
    FP32: tl.constexpr, LARGE_T: tl.constexpr,
):  # fmt: skip
    """Store a streamed row's gradient over unmasked full tiles and a masked tail."""
    rtemp, rs, rt = _reciprocal(temp), _reciprocal(s_sum), _reciprocal(t_sum)
    log1p_c = _log1p(correction)
    # In simple rows (no correction), q / p = exp(dn) ratio. A bounded ratio keeps q below a
    # negligible multiple of an underflowed p.
    ratio = _exp(mean_delta.to(tl.float64) / temp.to(tl.float64)) * t_sum.to(tl.float64) / s_sum
    ratio = tl.where(simple & (ratio >= 2.0**-20) & (ratio <= 2.0**20), ratio, 0.0).to(temp.dtype)
    FULL: tl.constexpr = C // B * B
    for start in range(0, FULL, B):
        _gradient_tile(
            S_ptr, T_ptr, G_ptr, base, start, C, rtemp, s_max, t_max, rs, rt, mean_delta,
            rel_mean, over, log1p_c, simple, ratio, u_sum, bad, weight, B, FP32, LARGE_T, False,
        )  # fmt: skip
    if FULL < C:
        _gradient_tile(
            S_ptr, T_ptr, G_ptr, base, FULL, C, rtemp, s_max, t_max, rs, rt, mean_delta,
            rel_mean, over, log1p_c, simple, ratio, u_sum, bad, weight, B, FP32, LARGE_T, True,
        )  # fmt: skip


@triton.jit
def _backward_rows(
    S_ptr, T_ptr, TEMP_ptr, ST_ptr, U_ptr, G_ptr, N, C: tl.constexpr, B: tl.constexpr,
    FP32: tl.constexpr, LARGE_T: tl.constexpr, REDUCTION: tl.constexpr, ROWS: tl.constexpr,
):  # fmt: skip
    """_backward across a [ROWS, B] tile of short rows (B >= C)."""
    dtype: tl.constexpr = tl.float32 if FP32 else tl.float64
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    row_valid = row < N
    n = N.to(tl.int64)
    W = ST_ptr + row
    temp = tl.load(TEMP_ptr).to(dtype)
    bad = (tl.load(W + n, mask=row_valid, other=0.0) != 0)[:, None]
    s_max = tl.load(W + 2 * n, mask=row_valid, other=0.0).to(dtype)[:, None]
    t_max = tl.load(W + 3 * n, mask=row_valid, other=0.0).to(dtype)[:, None]
    s_sum = tl.load(W + 4 * n, mask=row_valid, other=1.0).to(dtype)[:, None]
    t_sum = tl.load(W + 5 * n, mask=row_valid, other=1.0).to(dtype)[:, None]
    mean_delta = tl.load(W + 8 * n, mask=row_valid, other=0.0).to(dtype)[:, None]
    u_sum = tl.load(W + 11 * n, mask=row_valid, other=0.0).to(dtype)[:, None]
    code_1d = tl.load(W + 12 * n, mask=row_valid, other=-0.5)
    code = code_1d[:, None]
    if REDUCTION == 0:
        weight = tl.load(U_ptr + row, mask=row_valid, other=0.0).to(dtype)[:, None]
    else:
        weight = tl.load(U_ptr).to(dtype)
    if REDUCTION == 1:
        weight = weight / N
    j = tl.arange(0, B)[None, :]
    valid = row_valid[:, None] & (j < C)
    offsets = row[:, None] * C + j
    cs = _work(tl.load(S_ptr + offsets, mask=valid, other=0.0), FP32)
    ct = _work(tl.load(T_ptr + offsets, mask=valid, other=0.0), FP32)
    cs = tl.where(valid & _finite(cs), cs, 0.0)
    ct = tl.where(valid & _finite(ct), ct, 0.0)
    if C > 16 and tl.max((code_1d >= 0).to(tl.int32), 0) == 0:
        if not LARGE_T or tl.max(((code_1d != 0.5) & (code_1d != -0.5)).to(tl.int32), 0) == 0:
            s, t = _scaled(cs, ct, valid, s_max, t_max, temp, False)
            dn = _dn(cs, ct, s, t, s_max, t_max, mean_delta, mean_delta, False, temp, False)
        else:
            s, t = _scaled(cs, ct, valid, s_max, t_max, temp, LARGE_T)
            dn = _dn(cs, ct, s, t, s_max, t_max, mean_delta, mean_delta, False, temp, LARGE_T)
        q = _exp(s) / s_sum
        p = _exp(t) / t_sum
        u = tl.where(valid, p * _bounded_expm1(dn), 0.0)
    else:
        rel_mean = tl.load(W + 9 * n, mask=row_valid, other=0.0).to(dtype)[:, None]
        correction = tl.load(W + 10 * n, mask=row_valid, other=0.0).to(dtype)[:, None]
        over = ~_finite(mean_delta)
        s, t = _scaled(cs, ct, valid, s_max, t_max, temp, LARGE_T)
        q = _exp(s) / s_sum
        p = _exp(t) / t_sum
        dn = _dn(cs, ct, s, t, s_max, t_max, mean_delta, rel_mean, over, temp, LARGE_T)
        logratio = dn - _log1p(correction)
        u = _u(logratio, p, q, valid)
        if C > 16:
            u = tl.where(code < 0, tl.where(valid, p * _bounded_expm1(logratio), 0.0), u)
    g = u - q * u_sum
    if C <= 16:
        pivot = tl.minimum(tl.maximum(code, 0.0), C - 1).to(tl.int64)
        keep = valid & (j != pivot)
        others = tl.sum(tl.where(keep, g, 0.0), 1)[:, None]
        g = tl.where(j == pivot, -others, g)
    g = tl.where(bad, float("nan"), (g / temp) * weight)
    tl.store(G_ptr + offsets, g.to(G_ptr.dtype.element_ty), mask=valid)


@triton.jit(do_not_specialize=["N"])
def _backward(
    S_ptr,
    T_ptr,
    TEMP_ptr,
    ST_ptr,
    U_ptr,
    G_ptr,
    N,
    C: tl.constexpr,
    B: tl.constexpr,
    FP32: tl.constexpr,
    LARGE_T: tl.constexpr,
    REDUCTION: tl.constexpr,
    ROWS: tl.constexpr = 1,
):
    """Gradient u - q sum(u) from the row statistics; the pivot entry closes the zero sum."""
    if ROWS > 1:
        _backward_rows(
            S_ptr, T_ptr, TEMP_ptr, ST_ptr, U_ptr, G_ptr, N, C, B, FP32, LARGE_T, REDUCTION, ROWS
        )
    else:
        dtype: tl.constexpr = tl.float32 if FP32 else tl.float64
        row = tl.program_id(0).to(tl.int64)
        n = N.to(tl.int64)
        W = ST_ptr + row
        temp = tl.load(TEMP_ptr).to(dtype)
        bad = tl.load(W + n) != 0
        s_max, t_max = _slot(W, n, 2, dtype), _slot(W, n, 3, dtype)
        s_sum, t_sum = _slot(W, n, 4, dtype), _slot(W, n, 5, dtype)
        s_lognorm, t_lognorm = _slot(W, n, 6, dtype), _slot(W, n, 7, dtype)
        mean_delta, rel_mean = _slot(W, n, 8, dtype), _slot(W, n, 9, dtype)
        correction, u_sum = _slot(W, n, 10, dtype), _slot(W, n, 11, dtype)
        pivot = tl.minimum(tl.maximum(tl.load(W + 12 * n), 0.0), C - 1).to(tl.int64)
        over = ~_finite(mean_delta)
        if REDUCTION == 0:
            weight = tl.load(U_ptr + row).to(dtype)
        else:
            weight = tl.load(U_ptr).to(dtype)
        if REDUCTION == 1:
            weight = weight / N
        if C <= 16:
            others = tl.zeros([B], dtype)
            for start in range(0, C, B):
                k, valid, p, q, _, _, logratio = _stream_tile(
                    S_ptr, T_ptr, row * C, start, C, temp, s_max, t_max, s_sum, t_sum, s_lognorm,
                    t_lognorm, mean_delta, rel_mean, over, correction, B, FP32, LARGE_T,
                )  # fmt: skip
                g = _u(logratio, p, q, valid) - q * u_sum
                keep = valid & (k != pivot)
                others += tl.where(keep, g, 0.0)
                g = tl.where(bad, float("nan"), (g / temp) * weight)
                tl.store(G_ptr + row * C + k, g.to(G_ptr.dtype.element_ty), mask=keep)
            g = -tl.sum(others, 0)
            g = tl.where(bad, float("nan"), (g / temp) * weight)
            tl.store(G_ptr + row * C + pivot, g.to(G_ptr.dtype.element_ty))
        else:
            code = tl.load(W + 12 * n)
            simple = code < 0
            if LARGE_T:
                if (code == 0.5) | (code == -0.5):
                    _gradient_scan(
                        S_ptr, T_ptr, G_ptr, row * C, C, temp, s_max, t_max, s_sum, t_sum,
                        mean_delta, rel_mean, over, correction, simple, u_sum, bad, weight, B,
                        FP32, False,
                    )  # fmt: skip
                else:
                    _gradient_scan(
                        S_ptr, T_ptr, G_ptr, row * C, C, temp, s_max, t_max, s_sum, t_sum,
                        mean_delta, rel_mean, over, correction, simple, u_sum, bad, weight, B,
                        FP32, True,
                    )  # fmt: skip
            else:
                _gradient_scan(
                    S_ptr, T_ptr, G_ptr, row * C, C, temp, s_max, t_max, s_sum, t_sum, mean_delta,
                    rel_mean, over, correction, simple, u_sum, bad, weight, B, FP32, False,
                )  # fmt: skip


@triton.jit
def _prox_terms(
    omega,
    p,
    log_p,
    log_q,
    logratio,
    alpha,
    log_alpha,
    valid,
    EPS: tl.constexpr,
    TINY: tl.constexpr,
    CUT: tl.constexpr,
    LARGE: tl.constexpr,
    STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
    FP32: tl.constexpr,
):
    """Residuals r = p - pi and pi at omega, as _tensor_prox.proximal_distillation."""
    arg = omega - logratio
    h = (log_q + alpha * p) - omega
    prob = tl.minimum(_probability(h, alpha, log_alpha, CUT, LARGE, STEPS, FP32), 1.0)
    half = alpha * p / 2
    near = (p > TINY) & (arg <= half + 0.6931471805599453) & (arg >= -half - 0.4054651081081644)
    # The near range is _residual's: its far-range probability would be the prob above.
    r_near = _near_residual(
        tl.where(near, arg, 0.0),
        alpha,
        tl.where(near, p, 1.0),
        tl.where(near, log_p, 0.0),
        EPS,
        RES_STEPS,
        FP32,
    )
    # A zero argument has a zero residual, also where p is subnormal.
    r = tl.where(near, r_near, tl.where(arg == 0, 0.0, p - prob))
    pi = tl.where(near, p - r_near, prob)
    return tl.where(valid, r, 0.0), tl.where(valid, pi, 0.0)


@triton.jit
def _log_pi(
    r, p, log_p, log_q, omega, alpha, log_alpha, CUT: tl.constexpr, LARGE: tl.constexpr,
    STEPS: tl.constexpr, FP32: tl.constexpr,
):  # fmt: skip
    """log pi: log1p where r is small, else h - W."""
    h = (log_q + alpha * p) - omega
    far = h - _lambert_w_exp(h + log_alpha, CUT, LARGE, STEPS, FP32)
    near = log_p + _log1p(-r / tl.where(p > 0, p, 1.0))
    # Far above a tiny p, -r / p overflows: log pi = log(p - r) directly.
    above = r < -p / 2
    near = tl.where(above, _log(tl.where(above, p - r, 1.0)), near)
    return tl.where(r <= p / 2, near, far)


@triton.jit
def _omega_sums(r, pi, alpha):
    """f = sum r, its magnitude and the first two derivatives in omega."""
    denom = 1 + alpha * pi
    d1 = tl.sum(pi / denom, 0)
    d2 = -tl.sum(pi / (denom * denom * denom), 0)
    return tl.sum(r, 0), tl.sum(tl.abs(r), 0), d1, d2


@triton.jit
def _omega_step(omega, f, magnitude, d1, d2, lo, hi, EPS: tl.constexpr, TINY: tl.constexpr):
    """One safeguarded Halley step: the next omega, the bracket and done."""
    done = tl.abs(f) <= 8 * EPS * tl.maximum(magnitude, TINY)
    lo = tl.where(~done & (f < 0), omega, lo)
    hi = tl.where(~done & (f >= 0), omega, hi)
    newton = -f / d1
    correction = (f * d2) / (2 * d1 * d1)
    clamped = tl.minimum(tl.maximum(correction, -0.5), 0.5)
    delta = tl.where(_finite(correction), newton / (1 - clamped), newton)
    delta = tl.where((d1 > 0) & _finite(d1) & _finite(newton), delta, 0.0)
    following = omega + delta
    # A nonzero step below the resolution of omega: omega is the closest root.
    done = done | ((delta != 0) & (following == omega))
    inside = (lo <= following) & (following <= hi) & (following != omega) & _finite(following)
    following = tl.where(inside, following, lo + (hi - lo) / 2)
    done = done | (following == omega)
    return tl.where(done, omega, following), lo, hi, done


@triton.jit(do_not_specialize=["N"])
def _prox_kernel(
    S_ptr,
    T_ptr,
    PARAMS_ptr,
    LOSS_ptr,
    GRAD_ptr,
    N,
    C: tl.constexpr,
    B: tl.constexpr,
    FP32: tl.constexpr,
    LARGE_T: tl.constexpr,
    EPS: tl.constexpr,
    TINY: tl.constexpr,
    CUT: tl.constexpr,
    LARGE: tl.constexpr,
    STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
):
    dtype: tl.constexpr = tl.float32 if FP32 else tl.float64
    row = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, B)
    valid = j < C
    temp = tl.load(PARAMS_ptr).to(dtype)
    alpha = tl.load(PARAMS_ptr + 1).to(dtype)
    log_alpha = tl.load(PARAMS_ptr + 2).to(dtype)
    bad, _, p, q, log_q, log_p, logratio, u = _stats(
        S_ptr, T_ptr, temp, row, C, B, FP32, LARGE_T, EPS
    )
    # Safeguarded Halley steps on sum(r(omega)) = 0 in the bracket [lo, hi], which holds 0.
    lo = -alpha * tl.maximum(tl.max(tl.where(valid, u, -float("inf")), 0), 0.0)
    hi = -alpha * tl.minimum(tl.min(tl.where(valid, u, float("inf")), 0), 0.0)
    hi = tl.where(lo <= hi, hi, lo)
    omega = lo * 0
    done = bad | ((lo == 0) & (hi == 0))
    step = 0
    while ~done & (step < 100):
        r, pi = _prox_terms(
            omega, p, log_p, log_q, logratio, alpha, log_alpha, valid,
            EPS, TINY, CUT, LARGE, STEPS, RES_STEPS, FP32,
        )  # fmt: skip
        f, magnitude, d1, d2 = _omega_sums(r, pi, alpha)
        omega, lo, hi, done = _omega_step(omega, f, magnitude, d1, d2, lo, hi, EPS, TINY)
        step += 1
    r, _ = _prox_terms(
        omega, p, log_p, log_q, logratio, alpha, log_alpha, valid,
        EPS, TINY, CUT, LARGE, STEPS, RES_STEPS, FP32,
    )  # fmt: skip
    pivot = tl.argmax(tl.where(valid, p + q, -float("inf")), 0)
    r = tl.where(j == pivot, -tl.sum(tl.where(j == pivot, 0.0, r), 0), r)
    # H from log pi (log1p where r is small), and alpha r_j first in the penalty.
    log_pi = _log_pi(r, p, log_p, log_q, omega, alpha, log_alpha, CUT, LARGE, STEPS, FP32)
    cross = -tl.sum(tl.where(valid & (p > 0), p * log_pi, 0.0), 0)
    value = cross + tl.sum((alpha * r) * r, 0) / 2

    nan = float("nan")
    tl.store(LOSS_ptr + row, tl.where(bad, nan, value).to(LOSS_ptr.dtype.element_ty))
    grad = tl.where(bad, nan, -r / temp).to(GRAD_ptr.dtype.element_ty)
    tl.store(GRAD_ptr + row * C + j, grad, mask=valid)


@triton.jit
def _stream_tile(
    S_ptr, T_ptr, base, start, C, temp, s_max, t_max, s_sum, t_sum, s_lognorm, t_lognorm,
    mean_delta, rel_mean, over, correction, B: tl.constexpr, FP32: tl.constexpr,
    LARGE_T: tl.constexpr,
):  # fmt: skip
    """Columns, p, q, log q, log p and log(q / p) of a tile from the row statistics."""
    k = start + tl.arange(0, B)
    valid = k < C
    cs, ct, _ = _load_pair(S_ptr, T_ptr, base + k, valid, FP32)
    s, t = _scaled(cs, ct, valid, s_max, t_max, temp, LARGE_T)
    q = _exp(s) / s_sum
    p = _exp(t) / t_sum
    dn = _dn(cs, ct, s, t, s_max, t_max, mean_delta, rel_mean, over, temp, LARGE_T)
    return k, valid, p, q, s - s_lognorm, t - t_lognorm, dn - _log1p(correction)


@triton.jit
def _lean_pair(S_ptr, T_ptr, offsets, valid, FP32: tl.constexpr, MASKED: tl.constexpr):
    """Student and teacher logits in working precision; nonfinite ones pass through, as
    their rows are invalid."""
    if MASKED:
        cs = tl.load(S_ptr + offsets, mask=valid, other=0.0)
        ct = tl.load(T_ptr + offsets, mask=valid, other=0.0)
    else:
        cs = tl.load(S_ptr + offsets)
        ct = tl.load(T_ptr + offsets)
    return _work(cs, FP32), _work(ct, FP32)


@triton.jit
def _lean_scaled(
    cs, ct, valid, s_max, t_max, rtemp, LARGE_T: tl.constexpr, MASKED: tl.constexpr
):  # fmt: skip
    """_scaled by the reciprocal temperature."""
    ds = cs - s_max
    dt = ct - t_max
    s = ds * rtemp
    t = dt * rtemp
    if LARGE_T:
        s = tl.where(_finite(ds), s, cs * rtemp - s_max * rtemp)
        t = tl.where(_finite(dt), t, ct * rtemp - t_max * rtemp)
    if MASKED:
        s = tl.where(valid, s, -float("inf"))
        t = tl.where(valid, t, -float("inf"))
    return s, t


@triton.jit
def _lean_columns(start, C, B: tl.constexpr, MASKED: tl.constexpr):
    """Columns of a tile and their validity, constant for a full tile."""
    k = start + tl.arange(0, B)
    if MASKED:
        valid = k < C
    else:
        valid = tl.full([B], True, tl.int1)
    return k, valid


@triton.jit
def _reciprocal(x):
    """1 / x rounded once from float64."""
    return (1.0 / x.to(tl.float64)).to(x.dtype)


@triton.jit
def _lean_tile(
    S_ptr, T_ptr, base, start, C, rtemp, s_max, t_max, rt, mean_delta, rel_mean, over, log1p_c,
    B: tl.constexpr, FP32: tl.constexpr, LARGE_T: tl.constexpr, MASKED: tl.constexpr,
):  # fmt: skip
    """_stream_tile by the reciprocal temperature and teacher sum, with unmasked full tiles:
    columns, p, s and log(q / p); q = exp(s) / s_sum is left to the callers that need it."""
    k, valid = _lean_columns(start, C, B, MASKED)
    cs, ct = _lean_pair(S_ptr, T_ptr, base + k, valid, FP32, MASKED)
    s, t = _lean_scaled(cs, ct, valid, s_max, t_max, rtemp, LARGE_T, MASKED)
    et = _exp32(t) if FP32 and not MASKED else _exp(t)
    p = et * rt
    dn = (_offset_delta(cs, ct, s_max, t_max) - mean_delta) * rtemp
    if LARGE_T:
        dn = tl.where(over, (s - t) - rel_mean, dn)
    return k, valid, p, s, dn - log1p_c


@triton.jit
def _top_two(m1, i1, m2, x, k):
    """Keep two maxima per lane; defer communication until the entire row is read."""
    rise = x > m1
    return tl.maximum(m1, x), tl.where(rise, k, i1), tl.maximum(m2, tl.minimum(m1, x))


@triton.jit
def _finish_top_two(m1, i1, m2, C: tl.constexpr):
    maximum = tl.max(m1, 0)
    index = tl.min(tl.where(m1 == maximum, i1, C), 0)
    other = tl.max(tl.where(i1 == index, m2, m1), 0)
    return maximum, index, other


@triton.jit
def _stream_relative_mean(
    S_ptr,
    T_ptr,
    base,
    C,
    temp,
    s_max,
    t_max,
    t_sum,
    B: tl.constexpr,
    FP32: tl.constexpr,
    LARGE_T: tl.constexpr,
):
    j = tl.arange(0, B)
    total = tl.zeros([B], tl.float64)
    for start in range(0, C, B):
        valid = start + j < C
        cs, ct, _ = _load_pair(S_ptr, T_ptr, base + start + j, valid, FP32)
        s, t = _scaled(cs, ct, valid, s_max, t_max, temp, LARGE_T)
        p = _exp(t) / t_sum
        total += tl.where(valid & (p > 0), p * (s - t), 0.0).to(tl.float64)
    return tl.sum(total, 0).to(temp.dtype)


@triton.jit
def _normalizer_tile(
    S_ptr, T_ptr, base, start, C, temp, rtemp, s_max, t_max, s_orig, t_orig, s_idx, t_idx,
    s_other, t_other, weighted_gap, delta_sum, low, high, B: tl.constexpr, FP32: tl.constexpr,
    LARGE_T: tl.constexpr, WITH_MEAN: tl.constexpr, WITH_VALUE: tl.constexpr,
    LEAN: tl.constexpr, MASKED: tl.constexpr,
):  # fmt: skip
    """Add a tile to the lane sums of the tails, the value gap and the mean difference, and
    to the difference bounds; lean tiles scale by rtemp = 1 / temp."""
    k, valid = _lean_columns(start, C, B, MASKED)
    if LEAN:
        cs, ct = _lean_pair(S_ptr, T_ptr, base + k, valid, FP32, MASKED)
        s, t = _lean_scaled(cs, ct, valid, s_max, t_max, rtemp, LARGE_T, MASKED)
    else:
        cs, ct, _ = _load_pair(S_ptr, T_ptr, base + k, valid, FP32)
        s, t = _scaled(cs, ct, valid, s_max, t_max, temp, LARGE_T)
    if LEAN and FP32 and not MASKED:
        es, et = _exp32(s), _exp32(t)
    else:
        es, et = _exp(s), _exp(t)
    if s_orig == 0:
        s_terms = es
    else:
        s_terms = _exp(s - s_orig)
    if t_orig == 0:
        t_terms = et
    else:
        t_terms = _exp(t - t_orig)
    s_other += tl.where(k == s_idx, 0.0, s_terms).to(tl.float64)
    t_other += tl.where(k == t_idx, 0.0, t_terms).to(tl.float64)
    if WITH_VALUE:
        # Used only with FP32 work: widening before the product/sum keeps every finite
        # term and their C-term total in range. Guard zero teacher mass beside inf gaps.
        term = et.to(tl.float64) * (0.0 - s.to(tl.float64))
        weighted_gap += tl.where(valid & (et > 0), term, 0.0)
    if WITH_MEAN:
        delta = _offset_delta(cs, ct, s_max, t_max)
        low = tl.minimum(low, tl.where(valid, delta, float("inf")))
        high = tl.maximum(high, tl.where(valid, delta, -float("inf")))
        delta_sum += tl.where(valid & (et > 0), et * delta, 0.0).to(tl.float64)
    return s_other, t_other, weighted_gap, delta_sum, low, high


@triton.jit
def _lean_normalizers(
    S_ptr, T_ptr, base, C: tl.constexpr, temp, s_max, t_max, s_orig, t_orig, s_idx, t_idx,
    s_other, t_other, weighted_gap, delta_sum, low, high, B: tl.constexpr, FP32: tl.constexpr,
    LARGE_T: tl.constexpr, WITH_MEAN: tl.constexpr, WITH_VALUE: tl.constexpr,
):  # fmt: skip
    """The normalizer pass over unmasked full tiles and a masked tail."""
    rtemp = _reciprocal(temp)
    FULL: tl.constexpr = C // B * B
    for start in range(0, FULL, B):
        s_other, t_other, weighted_gap, delta_sum, low, high = _normalizer_tile(
            S_ptr, T_ptr, base, start, C, temp, rtemp, s_max, t_max, s_orig, t_orig,
            s_idx, t_idx, s_other, t_other, weighted_gap, delta_sum, low, high,
            B, FP32, LARGE_T, WITH_MEAN, WITH_VALUE, True, False,
        )  # fmt: skip
    if FULL < C:
        s_other, t_other, weighted_gap, delta_sum, low, high = _normalizer_tile(
            S_ptr, T_ptr, base, FULL, C, temp, rtemp, s_max, t_max, s_orig, t_orig,
            s_idx, t_idx, s_other, t_other, weighted_gap, delta_sum, low, high,
            B, FP32, LARGE_T, WITH_MEAN, WITH_VALUE, True, True,
        )  # fmt: skip
    return s_other, t_other, weighted_gap, delta_sum, low, high


@triton.jit
def _stream_normalizers(
    S_ptr, T_ptr, base, C, temp, B: tl.constexpr, FP32: tl.constexpr, LARGE_T: tl.constexpr,
    EPS: tl.constexpr, WITH_MEAN: tl.constexpr = False, WITH_VALUE: tl.constexpr = False,
    LEAN: tl.constexpr = False,
):  # fmt: skip
    """Accumulate lane statistics through the row, then reduce once per statistic. Lean
    rows (C constant) skip the nonfinite selects after the first pass and scale by 1 / temp."""
    neg_inf: tl.constexpr = -float("inf")
    j = tl.arange(0, B)
    zero = temp * 0
    s_first = tl.full([B], neg_inf, zero.dtype)
    t_first, s_top, t_top = s_first, s_first, s_first
    s_idx = tl.full([B], C, tl.int32)
    t_idx = s_idx
    bad = tl.full([B], False, tl.int1)
    s_least = tl.full([B], float("inf"), zero.dtype)
    t_least = s_least
    for start in range(0, C, B):
        k = start + j
        valid = k < C
        cs = _work(tl.load(S_ptr + base + k, mask=valid, other=0.0), FP32)
        ct = _work(tl.load(T_ptr + base + k, mask=valid, other=0.0), FP32)
        bad = bad | (valid & (~_finite(cs) | ~_finite(ct)))
        cs = tl.where(_finite(cs), cs, 0.0)
        ct = tl.where(_finite(ct), ct, 0.0)
        s_first, s_idx, s_top = _top_two(s_first, s_idx, s_top, tl.where(valid, cs, neg_inf), k)
        t_first, t_idx, t_top = _top_two(t_first, t_idx, t_top, tl.where(valid, ct, neg_inf), k)
        if LEAN and LARGE_T:
            s_least = tl.minimum(s_least, tl.where(valid, cs, float("inf")))
            t_least = tl.minimum(t_least, tl.where(valid, ct, float("inf")))
    bad = tl.sum(bad.to(tl.int32), 0) > 0
    s_max, s_idx, s_top = _finish_top_two(s_first, s_idx, s_top, C)
    t_max, t_idx, t_top = _finish_top_two(t_first, t_idx, t_top, C)
    s_top = _temp_diff(s_top, s_max, temp, LARGE_T)
    t_top = _temp_diff(t_top, t_max, temp, LARGE_T)
    # Within finite summed ranges, centered logits and their differences are finite: the
    # large-temperature fallbacks are not needed.
    ranged = _finite((s_max - tl.min(s_least, 0)) + (t_max - tl.min(t_least, 0)))
    # Normal tails reuse exp(s), exp(t); far tails retain scaling before subnormal rounding.
    cutoff: tl.constexpr = -40.0 if FP32 else -350.0
    s_orig = tl.where(_finite(s_top) & (s_top < cutoff), s_top, 0.0)
    t_orig = tl.where(_finite(t_top) & (t_top < cutoff), t_top, 0.0)
    s_other = tl.zeros([B], tl.float64)
    t_other = s_other
    delta_sum = s_other
    weighted_gap = s_other
    low = tl.full([B], float("inf"), zero.dtype)
    high = -low
    if LEAN:
        if LARGE_T:
            if ranged:
                s_other, t_other, weighted_gap, delta_sum, low, high = _lean_normalizers(
                    S_ptr, T_ptr, base, C, temp, s_max, t_max, s_orig, t_orig, s_idx, t_idx,
                    s_other, t_other, weighted_gap, delta_sum, low, high, B, FP32, False,
                    WITH_MEAN, WITH_VALUE,
                )  # fmt: skip
            else:
                s_other, t_other, weighted_gap, delta_sum, low, high = _lean_normalizers(
                    S_ptr, T_ptr, base, C, temp, s_max, t_max, s_orig, t_orig, s_idx, t_idx,
                    s_other, t_other, weighted_gap, delta_sum, low, high, B, FP32, True,
                    WITH_MEAN, WITH_VALUE,
                )  # fmt: skip
        else:
            s_other, t_other, weighted_gap, delta_sum, low, high = _lean_normalizers(
                S_ptr, T_ptr, base, C, temp, s_max, t_max, s_orig, t_orig, s_idx, t_idx,
                s_other, t_other, weighted_gap, delta_sum, low, high, B, FP32, False,
                WITH_MEAN, WITH_VALUE,
            )  # fmt: skip
    else:
        for start in range(0, C, B):
            s_other, t_other, weighted_gap, delta_sum, low, high = _normalizer_tile(
                S_ptr, T_ptr, base, start, C, temp, temp, s_max, t_max, s_orig, t_orig,
                s_idx, t_idx, s_other, t_other, weighted_gap, delta_sum, low, high,
                B, FP32, LARGE_T, WITH_MEAN, WITH_VALUE, False, True,
            )  # fmt: skip
    s_half, t_half = _exp(s_orig * 0.5), _exp(t_orig * 0.5)
    s_tail = s_half * (s_half * tl.sum(s_other, 0).to(zero.dtype))
    t_tail = t_half * (t_half * tl.sum(t_other, 0).to(zero.dtype))
    s_lognorm = tl.where(s_tail < EPS * 0.5, s_tail, _log1p(s_tail))
    t_lognorm = tl.where(t_tail < EPS * 0.5, t_tail, _log1p(t_tail))
    s_sum, t_sum = 1 + s_tail, 1 + t_tail
    if WITH_VALUE:
        value = s_lognorm.to(tl.float64) + tl.sum(weighted_gap, 0) / t_sum.to(tl.float64)
        return bad, s_max, t_max, s_sum, t_sum, s_lognorm, t_lognorm, value
    elif WITH_MEAN:
        mean = (tl.sum(delta_sum, 0) / t_sum).to(zero.dtype)
        relative = mean
        if LARGE_T:
            if ~_finite(mean):
                relative = _stream_relative_mean(
                    S_ptr, T_ptr, base, C, temp, s_max, t_max, t_sum, B, FP32, LARGE_T
                )
        lower = (tl.min(low, 0) - mean) / temp
        upper = (tl.max(high, 0) - mean) / temp
        simple = _finite(mean) & (lower >= -2) & (upper <= 2)
        return bad, s_max, t_max, s_sum, t_sum, s_lognorm, t_lognorm, mean, relative, simple, ranged
    else:
        return bad, s_max, t_max, s_sum, t_sum, s_lognorm, t_lognorm


@triton.jit
def _stream_correction(
    S_ptr,
    T_ptr,
    base,
    C,
    temp,
    s_max,
    t_max,
    t_sum,
    t_lognorm,
    mean_delta,
    rel_mean,
    over,
    B: tl.constexpr,
    FP32: tl.constexpr,
    LARGE_T: tl.constexpr,
):
    """The log-ratio correction of a row without a pivot, over lean tiles."""
    shift = _shift(s_max, t_max, mean_delta, rel_mean, over, t_lognorm, temp, LARGE_T)
    rtemp, rt = _reciprocal(temp), _reciprocal(t_sum)
    correction = tl.zeros([B], tl.float64)
    FULL: tl.constexpr = C // B * B
    for start in range(0, FULL, B):
        _, _, p, s, dn = _lean_tile(
            S_ptr, T_ptr, base, start, C, rtemp, s_max, t_max, rt, mean_delta, rel_mean, over,
            0.0, B, FP32, LARGE_T, False,
        )  # fmt: skip
        correction += _correction_term(dn, p, s, shift).to(tl.float64)
    if FULL < C:
        _, valid, p, s, dn = _lean_tile(
            S_ptr, T_ptr, base, FULL, C, rtemp, s_max, t_max, rt, mean_delta, rel_mean, over,
            0.0, B, FP32, LARGE_T, True,
        )  # fmt: skip
        correction += tl.where(valid, _correction_term(dn, p, s, shift), 0.0).to(tl.float64)
    return tl.sum(correction, 0).to(temp.dtype)


@triton.jit
def _stream_ratio(
    S_ptr, T_ptr, base, C, temp, s_max, t_max, t_sum, t_lognorm, B: tl.constexpr,
    FP32: tl.constexpr, LARGE_T: tl.constexpr,
):  # fmt: skip
    """The teacher-weighted mean differences and the log-ratio correction, as _stats; the
    sums over tiles accumulate in float64."""
    j = tl.arange(0, B)
    zero = temp * 0
    p_sum = tl.zeros([B], tl.float64)
    p_delta = p_sum
    p_rel = p_sum
    for start in range(0, C, B):
        valid = start + j < C
        cs, ct, _ = _load_pair(S_ptr, T_ptr, base + start + j, valid, FP32)
        s, t = _scaled(cs, ct, valid, s_max, t_max, temp, LARGE_T)
        p = _exp(t) / t_sum
        weighted = valid & (p > 0)
        p_sum += tl.where(weighted, p, 0.0).to(tl.float64)
        p_delta += tl.where(weighted, p * _offset_delta(cs, ct, s_max, t_max), 0.0).to(tl.float64)
        if LARGE_T:
            p_rel += tl.where(weighted, p * (s - t), 0.0).to(tl.float64)
    p_sum, p_delta, p_rel = tl.sum(p_sum, 0), tl.sum(p_delta, 0), tl.sum(p_rel, 0)
    mean_delta = (p_delta / p_sum).to(zero.dtype)
    rel_mean = mean_delta
    if LARGE_T:
        rel_mean = (p_rel / p_sum).to(zero.dtype)
    over = ~_finite(mean_delta)
    shift = _shift(s_max, t_max, mean_delta, rel_mean, over, t_lognorm, temp, LARGE_T)
    correction = tl.zeros([B], tl.float64)
    for start in range(0, C, B):
        valid = start + j < C
        cs, ct, _ = _load_pair(S_ptr, T_ptr, base + start + j, valid, FP32)
        s, t = _scaled(cs, ct, valid, s_max, t_max, temp, LARGE_T)
        dn = _dn(cs, ct, s, t, s_max, t_max, mean_delta, rel_mean, over, temp, LARGE_T)
        p = _exp(t) / t_sum
        term = _correction_term(dn, p, s, shift)
        correction += tl.where(valid, term, 0.0).to(tl.float64)
    return mean_delta, rel_mean, over, tl.sum(correction, 0).to(zero.dtype)


@triton.jit
def _stream_residuals(
    S_ptr, T_ptr, GRAD_ptr, base, C, pivot, bad, omega, alpha, log_alpha, temp, s_max, t_max,
    s_sum, t_sum, s_lognorm, t_lognorm, mean_delta, rel_mean, over, correction,
    B: tl.constexpr, FP32: tl.constexpr, LARGE_T: tl.constexpr, EPS: tl.constexpr,
    TINY: tl.constexpr, CUT: tl.constexpr, LARGE: tl.constexpr, STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
):  # fmt: skip
    """Write -r / T at omega and return the loss; the pivot balances the others, last.
    The sums over tiles accumulate in float64."""
    nan = float("nan")
    rest = tl.zeros([], tl.float64)
    cross = rest
    penalty = rest
    for start in range(0, C, B):
        k, valid, p, q, log_q, log_p, logratio = _stream_tile(
            S_ptr, T_ptr, base, start, C, temp, s_max, t_max, s_sum, t_sum, s_lognorm,
            t_lognorm, mean_delta, rel_mean, over, correction, B, FP32, LARGE_T,
        )  # fmt: skip
        r, _ = _prox_terms(
            omega, p, log_p, log_q, logratio, alpha, log_alpha, valid,
            EPS, TINY, CUT, LARGE, STEPS, RES_STEPS, FP32,
        )  # fmt: skip
        others = valid & (k != pivot)
        rest += tl.sum(tl.where(others, r, 0.0), 0).to(tl.float64)
        log_pi = _log_pi(r, p, log_p, log_q, omega, alpha, log_alpha, CUT, LARGE, STEPS, FP32)
        cross -= tl.sum(tl.where(others & (p > 0), p * log_pi, 0.0), 0).to(tl.float64)
        penalty += tl.sum(tl.where(others, (alpha * r) * r, 0.0), 0).to(tl.float64)
        grad = tl.where(bad, nan, -r / temp).to(GRAD_ptr.dtype.element_ty)
        tl.store(GRAD_ptr + base + k, grad, mask=valid)
    # The pivot's column alone, as a one-column tile.
    k, valid, p, q, log_q, log_p, logratio = _stream_tile(
        S_ptr, T_ptr, base, pivot, pivot + 1, temp, s_max, t_max, s_sum, t_sum, s_lognorm,
        t_lognorm, mean_delta, rel_mean, over, correction, 1, FP32, LARGE_T,
    )  # fmt: skip
    r = (-rest).to(omega.dtype) + tl.zeros([1], omega.dtype)
    log_pi = _log_pi(r, p, log_p, log_q, omega, alpha, log_alpha, CUT, LARGE, STEPS, FP32)
    cross -= tl.sum(tl.where(p > 0, p * log_pi, 0.0), 0).to(tl.float64)
    value = (cross + (penalty + tl.sum((alpha * r) * r, 0).to(tl.float64)) / 2).to(omega.dtype)
    # Every thread's tile stores land before the pivot's.
    tl.debug_barrier()
    grad = tl.where(bad, nan, -r / temp).to(GRAD_ptr.dtype.element_ty)
    tl.store(GRAD_ptr + base + k, grad)
    return value


@triton.jit(do_not_specialize=["N", "C"])
def _prox_stream_kernel(
    S_ptr,
    T_ptr,
    PARAMS_ptr,
    LOSS_ptr,
    GRAD_ptr,
    N,
    C,
    B: tl.constexpr,
    FP32: tl.constexpr,
    LARGE_T: tl.constexpr,
    EPS: tl.constexpr,
    TINY: tl.constexpr,
    CUT: tl.constexpr,
    LARGE: tl.constexpr,
    STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
):
    """_prox_kernel over tiles of B columns: the statistics and each step are passes."""
    dtype: tl.constexpr = tl.float32 if FP32 else tl.float64
    neg_inf: tl.constexpr = -float("inf")
    row = tl.program_id(0).to(tl.int64)
    base = row * C
    temp = tl.load(PARAMS_ptr).to(dtype)
    alpha = tl.load(PARAMS_ptr + 1).to(dtype)
    log_alpha = tl.load(PARAMS_ptr + 2).to(dtype)
    zero = temp * 0
    bad, s_max, t_max, s_sum, t_sum, s_lognorm, t_lognorm = _stream_normalizers(
        S_ptr, T_ptr, base, C, temp, B, FP32, LARGE_T, EPS
    )
    mean_delta, rel_mean, over, correction = _stream_ratio(
        S_ptr, T_ptr, base, C, temp, s_max, t_max, t_sum, t_lognorm, B, FP32, LARGE_T
    )
    # The bracket of omega and the pivot (the first largest p + q).
    u_max = zero + neg_inf
    u_min = zero - neg_inf
    pq_max = zero + neg_inf
    pivot = C * 0
    for start in range(0, C, B):
        k, valid, p, q, log_q, log_p, logratio = _stream_tile(
            S_ptr, T_ptr, base, start, C, temp, s_max, t_max, s_sum, t_sum, s_lognorm,
            t_lognorm, mean_delta, rel_mean, over, correction, B, FP32, LARGE_T,
        )  # fmt: skip
        u = _u(logratio, p, q, valid)
        u_max = tl.maximum(u_max, tl.max(tl.where(valid, u, neg_inf), 0))
        u_min = tl.minimum(u_min, tl.min(tl.where(valid, u, -neg_inf), 0))
        pq = tl.where(valid, p + q, neg_inf)
        largest = tl.max(pq, 0)
        pivot = tl.where(largest > pq_max, start + tl.argmax(pq, 0), pivot)
        pq_max = tl.maximum(pq_max, largest)

    # Safeguarded Halley steps on sum(r(omega)) = 0 in the bracket [lo, hi], which holds 0.
    lo = -alpha * tl.maximum(u_max, 0.0)
    hi = -alpha * tl.minimum(u_min, 0.0)
    hi = tl.where(lo <= hi, hi, lo)
    omega = lo * 0
    done = bad | ((lo == 0) & (hi == 0))
    step = 0
    while ~done & (step < 100):
        # The sums over tiles accumulate in float64.
        f = tl.zeros([], tl.float64)
        magnitude = f
        d1 = f
        d2 = f
        for start in range(0, C, B):
            k, valid, p, q, log_q, log_p, logratio = _stream_tile(
                S_ptr, T_ptr, base, start, C, temp, s_max, t_max, s_sum, t_sum, s_lognorm,
                t_lognorm, mean_delta, rel_mean, over, correction, B, FP32, LARGE_T,
            )  # fmt: skip
            r, pi = _prox_terms(
                omega, p, log_p, log_q, logratio, alpha, log_alpha, valid,
                EPS, TINY, CUT, LARGE, STEPS, RES_STEPS, FP32,
            )  # fmt: skip
            tf, tm, t1, t2 = _omega_sums(r, pi, alpha)
            f += tf.to(tl.float64)
            magnitude += tm.to(tl.float64)
            d1 += t1.to(tl.float64)
            d2 += t2.to(tl.float64)
        omega, lo, hi, done = _omega_step(
            omega, f.to(dtype), magnitude.to(dtype), d1.to(dtype), d2.to(dtype), lo, hi, EPS, TINY
        )
        step += 1

    value = _stream_residuals(
        S_ptr, T_ptr, GRAD_ptr, base, C, pivot, bad, omega, alpha, log_alpha, temp, s_max,
        t_max, s_sum, t_sum, s_lognorm, t_lognorm, mean_delta, rel_mean, over, correction,
        B, FP32, LARGE_T, EPS, TINY, CUT, LARGE, STEPS, RES_STEPS,
    )  # fmt: skip
    tl.store(LOSS_ptr + row, tl.where(bad, float("nan"), value).to(LOSS_ptr.dtype.element_ty))


_scalars: dict[tuple[int, torch.dtype, tuple[float, ...], int | None], torch.Tensor] = {}


def _scalar_tensor(
    device: torch.device, work: torch.dtype, values: tuple[float, ...]
) -> torch.Tensor:
    """Device scalars in the work dtype (float arguments would be float32), cached per stream."""
    stream = torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None
    key = (device.index or 0, work, values, stream)
    if key not in _scalars:
        if len(_scalars) > 64:
            _scalars.clear()
        with torch.inference_mode(False):
            t = torch.empty(len(values), dtype=work, device=device)
            for i, value in enumerate(values):
                t[i].fill_(value)
            _scalars[key] = t
    return _scalars[key]


def _work_dtype(dtype: torch.dtype, temperature: float, double: bool) -> torch.dtype:
    """float32 for non-float64 logits unless double, or float64 where T or 1/T underflows."""
    work = torch.float64 if double or dtype == torch.float64 else torch.float32
    tiny = torch.finfo(work).tiny
    return work if tiny <= temperature <= 1.0 / tiny else torch.float64


# Columns and warps of the tiles of streamed rows, and of the backward chunks.
_STREAM = (256, 4)
_BACKWARD = (1024, 4)


def _rows_per_cta(c: int) -> int:
    """Rows per program for short distillation rows (c <= min(256, _CHUNK)), else 1."""
    block = max(16, triton.next_power_of_2(c))
    if c > 256 or block > _CHUNK:
        return 1
    return min(16, 256 // block)


def _shapes(c: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Forward and backward (block, warps) of rows of c columns; wider rows than _CHUNK
    stream tiles."""
    block = max(16, triton.next_power_of_2(c))
    if block > _CHUNK:
        return _STREAM, _BACKWARD
    rows = _rows_per_cta(c)
    if rows > 1:
        return (block, 2), (block, 2)
    backward = min(block, _BACKWARD[0])
    return (block, max(1, min(4, block // 512))), (backward, max(1, backward // 256))


_REDUCTIONS = {"none": 0, "mean": 1, "sum": 2}
_P, _I = _cuda_launch.POINTER, _cuda_launch.INT32


def forward(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
    double: bool,
    reduction: str,
    value_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss and row statistics of contiguous (N, C) CUDA logits; invalid rows give NaN.

    value_only omits backward statistics. Callers must select it before entering
    autograd.Function.forward or a custom operator, whose grad mode is disabled.
    """
    n, c = student.shape
    with _cuda_launch.device_guard(student):
        work = _work_dtype(student.dtype, temperature, double)
        stats = torch.empty(
            _triton_reduce.size(n, _SLOTS), device=student.device, dtype=torch.float64
        )
        shape = (n,) if reduction == "none" else ()
        dtype = torch.promote_types(student.dtype, torch.float32)
        out = torch.empty(shape, device=student.device, dtype=dtype)
        (block, warps), _ = _shapes(c)
        rows = _rows_per_cta(c)
        _forward[(triton.cdiv(n, rows),)](
            student, teacher, _scalar_tensor(student.device, work, (temperature,)), stats, n, c,
            block, work == torch.float32, temperature > 1.0, torch.finfo(work).eps, value_only,
            rows, num_warps=warps, enable_reflect_ftz=False,
        )  # fmt: skip
        _triton_reduce.launch(stats, out, n, _REDUCTIONS[reduction], _LAYOUT)
        return out, stats


def backward(
    student: torch.Tensor,
    teacher: torch.Tensor,
    stats: torch.Tensor,
    upstream: torch.Tensor,
    temperature: float,
    double: bool,
    reduction: str,
) -> torch.Tensor:
    """Gradient in the student's dtype, scaled by the upstream derivative."""
    n, c = student.shape
    with _cuda_launch.device_guard(student):
        work = _work_dtype(student.dtype, temperature, double)
        gradient = torch.empty_like(student)
        upstream = upstream.to(torch.promote_types(student.dtype, torch.float32)).contiguous()
        _, (block, warps) = _shapes(c)
        rows = _rows_per_cta(c)
        _backward[(triton.cdiv(n, rows),)](
            student, teacher, _scalar_tensor(student.device, work, (temperature,)), stats,
            upstream, gradient, n, c, block, work == torch.float32, temperature > 1.0,
            _REDUCTIONS[reduction], rows, num_warps=warps, enable_reflect_ftz=False,
        )  # fmt: skip
        return gradient


class _Fused(torch.autograd.Function):
    """Triton-launched distillation_loss: saves the row statistics, not the gradient."""

    @staticmethod
    def forward(
        ctx: Any,
        student: torch.Tensor,
        teacher: torch.Tensor,
        temperature: float,
        double: bool,
        reduction: str,
    ) -> torch.Tensor:
        out, stats = forward(student, teacher, temperature, double, reduction)
        ctx.save_for_backward(student, teacher, stats)
        ctx.options = (temperature, double, reduction)
        return out

    @staticmethod
    @first_order
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        student, teacher, stats = ctx.saved_tensors
        gradient = backward(student, teacher, stats, grad_outputs[0], *ctx.options)
        return gradient, None, None, None, None


@functools.lru_cache(maxsize=_cuda_launch.CACHE)
def _compiled(
    device: int,
    dtype: torch.dtype,
    c: int,
    work: torch.dtype,
    large_t: bool,
    code: int,
    value_only: bool = False,
) -> Any:
    """C++ launch handles keyed by shape, arithmetic, reduction and value-only mode."""
    with torch.cuda.device(device):
        z = torch.empty((1, 1), device="cuda", dtype=dtype)
        temp = torch.empty(1, device="cuda", dtype=work)
        stats = torch.empty(1, device="cuda", dtype=torch.float64)
        out = torch.empty(1, device="cuda", dtype=torch.promote_types(dtype, torch.float32))
        (block, warps), (b_block, b_warps) = _shapes(c)
        rows = _rows_per_cta(c)
        fp32, eps = work == torch.float32, torch.finfo(work).eps
        # Any int32 N: the kernels do not specialize on it.
        k_fwd = _forward.warmup(
            z, z, temp, stats, 2, c, block, fp32, large_t, eps, value_only, rows,
            num_warps=warps, enable_reflect_ftz=False, grid=(1,),
        )  # fmt: skip
        k_bwd = _backward.warmup(
            z, z, temp, stats, out, z, 2, c, b_block, fp32, large_t, code, rows,
            num_warps=b_warps, enable_reflect_ftz=False, grid=(1,),
        )  # fmt: skip
        reduce = _triton_reduce.warmup(stats, out, code, _LAYOUT)
        return _cuda_launch.handles(
            [(k_fwd, (_P, _P, _P, _P, _I))]
            + [(k, (_P, _P, _I)) for k in reduce]
            + [(k_bwd, (_P, _P, _P, _P, _P, _P, _I))],
            _triton_reduce.BLOCK,
            _triton_reduce.DIRECT_ROWS,
            0,
            rows,
            rows,
        )


def distillation_loss(
    student: torch.Tensor, teacher: torch.Tensor, temperature: float, double: bool, reduction: str
) -> torch.Tensor:
    """Reduced distillation_loss of contiguous (N, C) CUDA logits, with autograd."""
    value_only = not (torch.is_grad_enabled() and student.requires_grad)
    host = _cuda_launch.host()
    if host is not None:
        work = _work_dtype(student.dtype, temperature, double)
        code = _REDUCTIONS[reduction]
        handles = _compiled(
            student.device.index or 0,
            student.dtype,
            student.shape[1],
            work,
            temperature > 1,
            code,
            value_only,
        )
        if not isinstance(handles, str):
            temp = _scalar_tensor(student.device, work, (temperature,))
            return host.distillation_cuda_apply(student, teacher, temp, code, handles)
    if value_only:
        return forward(student, teacher, temperature, double, reduction, True)[0]
    return _Fused.apply(student, teacher, temperature, double, reduction)


def proximal_distillation(
    student: torch.Tensor, teacher: torch.Tensor, temperature: float, alpha: float, double: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss (N,) and gradient (N, C) of proximal_distillation_loss at contiguous CUDA logits."""
    n, c = student.shape
    with _cuda_launch.device_guard(student):
        tiny = torch.finfo(torch.float32).tiny
        holds = all(tiny <= v <= 1.0 / tiny for v in (temperature, alpha))
        fp32 = not double and student.dtype != torch.float64 and holds
        work = torch.float32 if fp32 else torch.float64
        cut, large, steps, res_steps = (-18.0, 1e7, 2, 4) if fp32 else (-36.0, 1e16, 3, 6)
        params = _scalar_tensor(student.device, work, (temperature, alpha, math.log(alpha)))
        out_dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
        loss = torch.empty(n, dtype=out_dtype, device=student.device)
        # float32 for 16-bit logits: backward scales it before the only rounding.
        grad = torch.empty(student.shape, dtype=out_dtype, device=student.device)
        if n > 0:
            stream = c > _CHUNK
            block = _STREAM_BLOCK if stream else max(16, triton.next_power_of_2(c))
            warps = _STREAM_WARPS if stream else max(1, min(16, block // 64))
            (_prox_stream_kernel if stream else _prox_kernel)[(n,)](
                student,
                teacher,
                params,
                loss,
                grad,
                n,
                c,
                B=block,
                FP32=fp32,
                LARGE_T=temperature > 1.0,
                EPS=torch.finfo(work).eps,
                TINY=torch.finfo(work).tiny,
                CUT=cut,
                LARGE=large,
                STEPS=steps,
                RES_STEPS=res_steps,
                num_warps=warps,
                enable_reflect_ftz=False,
            )
        return loss, grad
