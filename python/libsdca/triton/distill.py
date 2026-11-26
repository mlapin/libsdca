"""distillation and proximal_distillation, as include/sdca/loss/distill.h, on student and
teacher logits z and t with p = softmax(t / T) and q = softmax(z / T).

distillation: rows (ROWS short rows per program, rows up to 4096 classes in registers, wider
ones in passes over chunks) give H(p, q) = log1p(tail) - sum p s, with s = (z - max z) / T and
the tail of exp(s) beside one largest entry, and the gradient (q - p) / T times the row factor.
The logarithm of the ratio of the two normalizers comes from the sum of differences of
unnormalized probabilities, using expm1 for close student and teacher logits. Each close pair
of probabilities contributes p expm1(log(q / p)) to the gradient. On bounded float32 rows
whose largest probabilities are at most one half, tile sums use exactly summable high parts
and their residuals. Near-equal normalizers use the exponential-difference computation for
the gradient. Rows in registers subtract the residual sum times q. Identical logits give an
exact zero gradient.

proximal_distillation: one program per row solves sum_j r_j(omega) = 0 for the residuals
r = p - pi of the proximal point pi by safeguarded Halley steps (a pass over the row per step)
and writes the loss and the gradient -r / T, whose zero sum the pivot (the largest p + q)
closes. Float32 work takes float64 sums; a temperature or gamma outside float32's normal range
selects float64 work. Rows of one tile stay in registers, and a training forward saves the unit
gradient, which backward scales. Float32 streamed rows take chunk-parallel passes (_px_plan):
two reads gather the normalizers, then the head classes and tail moments on which a program
per row takes the Halley steps; the forward saves a small row state and backward writes the
gradient from it in one read.

A class the teacher masks (-inf) has teacher probability zero; one both mask is removed (its
gradient is +0.0). Teacher probability on a masked student class gives infinite loss and
the limiting gradient. NaN, +inf, or a student or teacher row of masked classes only gives
a NaN row.
"""

from __future__ import annotations

import math
from typing import Any

import triton
import triton.language as tl

from .._limits import FLOAT32
from . import DISTILLATION_REGISTER_CLASSES, common, rows
from .common import Builder, Call, pow2
from .math import (
    DBL_MAX,
    DBL_MIN,
    EXP_UNWIDE,
    EXP_WIDE,
    add_parts,
    bits,
    bounded_expm1,
    bounded_tile_parts,
    bounded_tile_sum,
    doubles,
    exp,
    exp32,
    exp_parts,
    expm1,
    f64,
    log,
    log1p,
    normal_exp32,
    scale,
    wide_exp,
    work,
)
from .prox import (
    _FLOAT32,
    _FLOAT64,
    failed,
    halley,
    lambert_w_exp,
    near_residual,
    probability,
    saved_replay,
    saved_rows,
    saved_stride,
)
from .prox import (
    _TAIL as _SERIES_TAIL,
)
from .prox import (
    _series as series,
)
from .prox import (
    _series_square as series_square,
)
from .rows import chunks as row_chunks

_REGISTER = DISTILLATION_REGISTER_CLASSES
# Streamed rows: the statistics pass's chunk and warps, the gradient's columns and warps per
# program; the proximal solve's chunk and warps.
_CHUNK = (512, 4)
_BACKWARD = (2048, 4)
_STREAM = (2048, 16)
_PROXIMAL_COPY = (2048, 4)
_SERIES_HEAD = 256  # head classes of _series_solve


@triton.jit
def _pair(Z, Y, row, live, start, C: tl.constexpr, B: tl.constexpr, FP32: tl.constexpr):
    """Student and teacher logits of columns start + j in the work type (NaN, +inf and columns
    off the row as 0; masked classes stay -inf), their bounds and the entries the losses
    refuse."""
    k = start + tl.arange(0, B)[None, :]
    valid = live[:, None] & (k < C)
    offsets = row[:, None] * C + k
    cs = work(tl.load(Z + offsets, valid, other=0), FP32)
    ct = work(tl.load(Y + offsets, valid, other=0), FP32)
    admitted = (cs < float("inf")) & (ct < float("inf"))
    bad = valid & ~admitted
    cs = tl.where(valid & (cs < float("inf")), cs, 0.0)
    ct = tl.where(valid & (ct < float("inf")), ct, 0.0)
    return k, valid, cs, ct, bad


@triton.jit
def _per_t(TEMP, FP32: tl.constexpr):
    """What _over takes for the temperature: its reciprocal in float32 work, rounded once
    (TEMP is then in float32's normal range), else TEMP, whose reciprocal can overflow."""
    if FP32:
        t = work(1.0 / f64(TEMP), FP32)
    else:
        t = f64(TEMP)
    return t


@triton.jit
def _divide_temperature(x, temperature, RECIPROCAL: tl.constexpr):
    # An exact normal power-of-two reciprocal gives the same rounded FP64 quotient,
    # including subnormal and overflowing results.
    if RECIPROCAL != 0.0:
        return x * RECIPROCAL
    else:
        return x / temperature


@triton.jit
def _over(x, t, RECIPROCAL: tl.constexpr = 0.0):
    """x / T from _per_t's t."""
    if t.dtype == tl.float32:
        y = x * t
    else:
        y = _divide_temperature(x, t, RECIPROCAL)
    return y


@triton.jit
def _factor(x, t):
    """A row factor over T as _gradient takes it: in float64 work the gradient divides by T
    last (_finish), so that its zero entries stay zero where 1/T overflows."""
    if t.dtype == tl.float32:
        y = x * t
    else:
        y = x
    return y


@triton.jit
def _finish(g, t):
    if t.dtype == tl.float32:
        y = g
    else:
        y = g / t
    return y


@triton.jit
def _scaled(cs, ct, valid, s_max, t_max, per_t, ORDINARY: tl.constexpr = False,
            RECIPROCAL: tl.constexpr = 0.0):  # fmt: skip
    """Centered, temperature-scaled logits (-inf off the row) and their difference, with both
    subtractions' rounding errors (NaN at masked classes)."""
    a = cs - s_max
    b = ct - t_max
    ea = (cs - (a - (a - cs))) + (-s_max - (a - cs))
    eb = (ct - (b - (b - ct))) + (-t_max - (b - ct))
    s = tl.where(valid, _over(a, per_t, RECIPROCAL), -float("inf"))
    t = tl.where(valid, _over(b, per_t, RECIPROCAL), -float("inf"))
    d = _over((a - b) + (ea - eb), per_t, RECIPROCAL)
    if ORDINARY:
        return s, t, d
    if per_t.dtype == tl.float64:
        # Centering two finite scores of opposite signs can overflow before division by T.
        s = tl.where(
            valid & (cs > -float("inf")) & (a == -float("inf")),
            _divide_temperature(0.5 * cs - 0.5 * s_max, 0.5 * per_t, 2.0 * RECIPROCAL),
            s,
        )
        t = tl.where(
            valid & (ct > -float("inf")) & (b == -float("inf")),
            _divide_temperature(0.5 * ct - 0.5 * t_max, 0.5 * per_t, 2.0 * RECIPROCAL),
            t,
        )
    else:
        s = tl.where(
            valid & (cs > -float("inf")) & (a == -float("inf")),
            ((0.5 * cs - 0.5 * s_max) * per_t) * 2,
            s,
        )
        t = tl.where(
            valid & (ct > -float("inf")) & (b == -float("inf")),
            ((0.5 * ct - 0.5 * t_max) * per_t) * 2,
            t,
        )
    d = tl.where(d == d, d, tl.where(s == t, 0.0, s - t))
    return s, t, d


@triton.jit
def _add(total, error, piece, COMPENSATE: tl.constexpr):
    """Add a tile's sum, retaining lost low bits across many tiles."""
    if COMPENSATE:
        piece -= error
        result = total + piece
        error = (result - total) - piece
    else:
        result = total + piece
    return result, error


@triton.jit
def _double_expm1(x):
    """Shared CPU range reduction for double expm1 on x <= 1; x <= -40 rounds to -1."""
    x = tl.where(x < -40.0, -40.0, x)
    n = tl.floor(x * 1.44269504088896338700 + 0.5)
    r = tl.fma(-n, 6.93147180369123816490e-1, x)
    r = tl.fma(-n, 1.90821492927058770002e-10, r)
    r2 = r * r
    r4 = r2 * r2
    p12 = tl.fma(r, 1.0 / 6227020800.0, 1.0 / 479001600.0)
    p10 = tl.fma(r, 1.0 / 39916800.0, 1.0 / 3628800.0)
    p8 = tl.fma(r, 1.0 / 362880.0, 1.0 / 40320.0)
    p6 = tl.fma(r, 1.0 / 5040.0, 1.0 / 720.0)
    p4 = tl.fma(r, 1.0 / 120.0, 1.0 / 24.0)
    p2 = tl.fma(r, 1.0 / 6.0, 0.5)
    q10 = tl.fma(p12, r2, p10)
    q6 = tl.fma(p8, r2, p6)
    q2 = tl.fma(p4, r2, p2)
    m = tl.fma(tl.fma(tl.fma(q10, r4, q6), r4, q2), r2, r)
    scale = ((n.to(tl.int64) + 1023) << 52).to(tl.float64, bitcast=True)
    return tl.fma(scale, m, scale - 1.0)


@triton.jit
def _sparse_mass_difference(x, y, peak, temperature, WIDE: tl.constexpr = False):
    """Signed exp differences with separate linear and quadratic terms for small changes.
    WIDE: both terms scaled by 2**64 (wide_exp), so that masses below exp(-708) keep their
    precision instead of rounding to subnormals before the sums."""
    top = tl.maximum(x, y)
    width = tl.abs(x - y) / temperature
    sign = tl.where(x >= y, 1.0, -1.0).to(tl.float64)
    # Select the arguments before the transcendentals. Large changes use the
    # top mass and expm1(-width). Small changes around zero use a common mass
    # and keep the shift from that origin in their quadratic residual.
    small = width < 0.125
    anchor = tl.where(small & (tl.abs(top) <= 0.5 * temperature), 0.0, top)
    if WIDE:
        mass = wide_exp((anchor - peak) / temperature)
    else:
        mass = exp((anchor - peak) / temperature)
    shift = expm1(tl.where(small, (top - anchor) / temperature, -width))
    u = -width
    poly = tl.fma(u, 1.0 / 362880.0, 1.0 / 40320.0)
    poly = tl.fma(u, poly, 1.0 / 5040.0)
    poly = tl.fma(u, poly, 1.0 / 720.0)
    poly = tl.fma(u, poly, 1.0 / 120.0)
    poly = tl.fma(u, poly, 1.0 / 24.0)
    poly = tl.fma(u, poly, 1.0 / 6.0)
    poly = tl.fma(u, poly, 0.5)
    remainder = -(width * width) * poly
    leading = (sign * mass) * tl.where(small, width, -shift)
    residual = sign * mass * tl.fma(1.0 + shift, remainder, width * shift)
    return leading, tl.where(small, residual, 0.0)


@triton.jit
def _difference_error(a, b, delta):
    term = delta - a
    return (a - (delta - term)) - (b + term)


@triton.jit
def _aligned_difference(Z, Y, row, live, per_t, s_max, t_max, valid1, cs1, ct1, ratio, ts,
                        ROWS: tl.constexpr, C: tl.constexpr, B: tl.constexpr, FP32: tl.constexpr):  # fmt: skip
    """Sum opposing mass changes at a common, exactly translated score origin."""
    # When the centered partitions nearly agree, their rounding error can
    # determine the signs of otherwise identical class probabilities.
    threshold: tl.constexpr = 2.0**-23 if FP32 else 2.0**-52
    close = live & (tl.abs(ratio) <= threshold)
    if tl.sum(close.to(tl.int32), 0) != 0:
        peak = s_max.to(tl.float64)
        teacher = t_max.to(tl.float64)
        shift = teacher - peak
        exact = close & (_difference_error(teacher, peak, shift) == 0.0)
        hi = tl.zeros([ROWS], tl.float64)
        lo = hi
        temperature = 1.0 / per_t.to(tl.float64) if FP32 else per_t
        for start in range(0, C, B):
            if B >= C:
                valid, cs, ct = valid1, cs1, ct1
            else:
                _, valid, cs, ct, _ = _pair(Z, Y, row, live, start, C, B, FP32)
            x, y = cs.to(tl.float64), ct.to(tl.float64)
            aligned = y - shift[:, None]
            lost = (y > -float("inf")) & (_difference_error(y, shift[:, None], aligned) != 0.0)
            if not FP32:
                # The scaled general path handles the overflow of a raw subtraction.
                lost |= (x > -float("inf")) & (tl.abs(x) > DBL_MAX / 2.0)
                lost |= (aligned > -float("inf")) & (tl.abs(aligned) > DBL_MAX / 2.0)
            exact &= tl.max((valid & lost).to(tl.int32), 1) == 0
            leading, residual = _sparse_mass_difference(
                x, aligned, peak[:, None], temperature, WIDE=True
            )
            use = valid & exact[:, None] & (x != aligned)
            pair = (tl.where(use, leading, 0.0), tl.where(use, residual, 0.0))
            leading, residual = tl.reduce(pair, 1, add_parts)
            hi, lo = add_parts(hi, lo, leading, residual)
        # The sums are of masses scaled by 2**64: a far tail's difference (common peaks with
        # every other class beyond exp(-708)) is then one rounding of the exact sum, not a sum
        # of subnormal terms (which loses a fraction of a percent of it near exp(-740)).
        ratio = tl.where(exact, ((hi + lo) / ts) * EXP_UNWIDE, ratio)
    return ratio


@triton.jit
def _precise_stats(Z, Y, row, live, per_t, s_max, t_max, cross_scale, valid1, cs1, ct1,
                   invalid_scores, ROWS: tl.constexpr, C: tl.constexpr, B: tl.constexpr,
                   FP32: tl.constexpr, GRAD: tl.constexpr, RECIPROCAL: tl.constexpr,
                   FINITE_DOUBLE: tl.constexpr = False, RECIP_MASS: tl.constexpr = False):  # fmt: skip
    """Accumulate separate maxima, masked mass and the small normalizer difference."""
    sties = tl.zeros([ROWS], tl.int32)
    tties = sties
    count = sties
    stail = tl.zeros([ROWS], tl.float64)
    ttail = stail
    delta = stail
    s_error = stail
    t_error = stail
    d_error = stail
    cross = stail
    cross_error = stail
    lost = stail
    for start in range(0, C, B):
        if B >= C:
            valid, cs, ct = valid1, cs1, ct1
        else:
            k, valid, cs, ct, bad = _pair(Z, Y, row, live, start, C, B, FP32)
        s, t, d = _scaled(
            cs,
            ct,
            valid,
            s_max[:, None],
            t_max[:, None],
            per_t,
            ORDINARY=FINITE_DOUBLE,
            RECIPROCAL=RECIPROCAL,
        )
        if FINITE_DOUBLE:
            es = exp(s) * EXP_WIDE
            te = exp(t)
            tf = 1.0
            et = te * EXP_WIDE
            kept = valid
        else:
            es = wide_exp(s)
            te, tf = exp_parts(t)
            et = (te * EXP_WIDE) * tf
            kept = valid & (cs > -float("inf"))
            count += tl.sum(kept.to(tl.int32), 1)
            lost += tl.sum(tl.where(kept, 0.0, et).to(tl.float64), 1)
        sties += tl.sum((valid & (s == 0)).to(tl.int32), 1)
        tties += tl.sum((valid & (t == 0)).to(tl.int32), 1)
        st = tl.sum(tl.where(s < 0, es, 0.0).to(tl.float64), 1)
        tt = tl.sum(tl.where(t < 0, et, 0.0).to(tl.float64), 1)
        stail, s_error = _add(stail, s_error, st, B < C and not FP32)
        ttail, t_error = _add(ttail, t_error, tt, B < C and not FP32)
        piece = tl.sum(
            tl.where(
                kept & (te > 0), (s.to(tl.float64) * te) * (tf * cross_scale[:, None]), 0.0
            ).to(tl.float64),
            1,
        )
        cross, cross_error = _add(cross, cross_error, piece, B < C and not FP32)
        if GRAD:
            near = tl.abs(d) < 1
            if FP32:
                v = tl.where(near, et * bounded_expm1(tl.where(near, d, 0.0)), es - et)
            else:
                # Reversing the scores must negate the term exactly. This keeps a
                # permutation with equal normalizers from giving untouched classes a gradient.
                relative = _double_expm1(tl.where(near, -tl.abs(d), 0.0))
                v = tl.where(near, tl.where(d > 0, -es, et) * relative, es - et)
            delta, d_error = _add(delta, d_error, tl.sum(v.to(tl.float64), 1), B < C and not FP32)
    if FINITE_DOUBLE:
        count = tl.full([ROWS], C, tl.int32)
    stail = stail * EXP_UNWIDE + (sties - 1).to(tl.float64)
    ttail = ttail * EXP_UNWIDE + (tties - 1).to(tl.float64)
    s_norm = log1p(stail)
    t_norm = log1p(ttail)
    ts = 1.0 + ttail
    c = (delta * EXP_UNWIDE) / ts
    if GRAD and B >= C:
        c = _aligned_difference(
            Z, Y, row, live, per_t, s_max, t_max, valid1, cs1, ct1, c, ts, ROWS, C, B, FP32
        )
    # log(sum(exp(s)) / sum(exp(t))) when c approaches -1.
    log_c = tl.where(c < -0.5, log((1.0 + stail) / ts), log1p(c))
    missing = (lost * EXP_UNWIDE) / ts
    value = tl.where(missing > 0, float("inf"), s_norm - cross * ((1.0 / cross_scale) / ts))
    bad = invalid_scores | (sties == 0) | (tties == 0)
    if RECIP_MASS:
        # Normalize the centered masses directly. Rounding a log partition before
        # subtracting it in every lane perturbs small probability differences.
        s_norm, t_norm = 1.0 / (1.0 + stail), 1.0 / ts
    return s_norm, t_norm, log_c, missing, count, value, bad


@triton.jit
def _mass_difference(d, es, et):
    near = tl.abs(d) < 1
    return tl.where(near, et * bounded_expm1(tl.where(near, d, 0.0)), es - et)


@triton.jit
def _close_normalizers(cs, ct, valid, s_max, t_max, per_t, ts, B: tl.constexpr):
    """The near-zero log ratio on bounded rows, from per-class exponential differences."""
    s, t, d = _scaled(cs, ct, valid, s_max[:, None], t_max[:, None], per_t, True)
    kept = valid & (cs > -float("inf"))
    es = tl.where(kept, normal_exp32(s), 0.0)
    et = tl.where(kept, normal_exp32(t), 0.0)
    near = tl.abs(d) < 1.0
    piece = tl.where(near, et * bounded_expm1(tl.where(near, d, 0.0)), es - et)
    delta = bounded_tile_sum(piece, 0.0, B, 1)
    # Keep the full-precision difference path for almost-identical distributions.
    distinct = tl.max(tl.abs(es - et), 1) >= 0.125
    return log1p(delta / ts), distinct


@triton.jit
def _bounded_stats(Z, Y, row, live, per_t, s_max, t_max, valid1, cs1, ct1,
                   C: tl.constexpr, B: tl.constexpr, GRAD: tl.constexpr, ORIGIN: tl.constexpr = False):  # fmt: skip
    """Float32 tile sums on shared-support rows with both centered ranges at most 16."""
    ss = tl.zeros([1], tl.float64)
    ts = tl.zeros([1], tl.float64)
    cross = tl.zeros([1], tl.float64)
    delta = tl.zeros([1], tl.float64)
    distinct = tl.zeros([1], tl.float32)
    invalid = tl.zeros([1, B], tl.int1)
    max_s = tl.full([1, B], -float("inf"), tl.float32)
    max_t = max_s
    for start in range(0, C, B):
        if B >= C:
            valid, cs, ct = valid1, cs1, ct1
            bad = tl.full([1, B], False, tl.int1)
        else:
            k, valid, cs, ct, bad = _pair(Z, Y, row, live, start, C, B, True)
        if ORIGIN:
            invalid |= bad | (
                valid & ((cs == ct) | (tl.abs(cs * per_t) > 8) | (tl.abs(ct * per_t) > 8))
            )
            max_s = tl.maximum(max_s, tl.where(valid, cs * per_t, -float("inf")))
            max_t = tl.maximum(max_t, tl.where(valid, ct * per_t, -float("inf")))
        valid = valid & (cs > -float("inf"))
        s = tl.where(valid, (cs - s_max[:, None]) * per_t, 0.0)
        t = tl.where(valid, (ct - t_max[:, None]) * per_t, 0.0)
        es = tl.where(valid, normal_exp32(s), 0.0)
        et = tl.where(valid, normal_exp32(t), 0.0)
        if ORIGIN:
            # e**8 < 4096. Scaling both masses by the same exact power of two
            # keeps the existing bounded tile sums and their error limits.
            es *= 0.000244140625
            et *= 0.000244140625
        ss += bounded_tile_sum(es, 0.0, B, 1)
        ts += bounded_tile_sum(et, 0.0, B, 1)
        if ORIGIN:
            _, _, d = _scaled(cs, ct, valid, s_max[:, None], t_max[:, None], per_t, True)
            piece = _mass_difference(d, es, et)
            delta += bounded_tile_sum(tl.where(valid, piece, 0.0), 0.0, B, 1)
            distinct = tl.maximum(distinct, tl.max(tl.abs(d), 1))
        dot = s * et
        cross += bounded_tile_sum(dot, tl.fma(s, et, -dot), B, 16)
    # On bounded rows, a negative student reciprocal identifies the fast backward.
    # Dominant or nearly identical distributions retain the wider statistics.
    accept = (ss >= 2.0) & (ts >= 2.0)
    log_ratio = log(ss / ts)
    if GRAD:
        separate = tl.abs(ss - ts) >= 0.125 * ts
        if B >= C:
            if tl.reshape(accept & ~separate, ()):
                log_ratio, separate = _close_normalizers(
                    cs1, ct1, valid1, s_max, t_max, per_t, ts, B
                )
        if ORIGIN:
            log_ratio = tl.where(separate, log_ratio, log1p(delta / ts))
            separate |= distinct >= 0.125
        accept &= separate
    if ORIGIN:
        accept &= (tl.max(invalid.to(tl.int32), 1) == 0) & (tl.abs(log_ratio) > 1.0e-4)
        accept &= tl.abs(log_ratio - (tl.max(max_s, 1) - tl.max(max_t, 1))) > 1.0e-4
        value = (log(ss) + 8.3177661667193437130) - cross / ts
        return accept, -0.000244140625 / ss, -0.000244140625 / ts, log_ratio, value
    return accept, -1.0 / ss, 1.0 / ts, log_ratio, log(ss) - cross / ts


@triton.jit
def _centered_stats(Z, Y, row, live, per_t, ROWS: tl.constexpr, C: tl.constexpr, B: tl.constexpr,
           FP32: tl.constexpr, GRAD: tl.constexpr, ORDINARY_ROWS: tl.constexpr = False,
           RECIPROCAL: tl.constexpr = 0.0, RECIP_MASS: tl.constexpr = False):  # fmt: skip
    """The two log normalizers and their difference. Accumulate exp(s) - exp(t)
    as exp(t) expm1(s - t) for close entries; the leading ones never cancel.
    The subtractive log ratio also applies when a single class is masked."""
    CHECK: tl.constexpr = ORDINARY_ROWS and ROWS == 1 and FP32
    k1, valid1, cs1, ct1, bad1 = _pair(Z, Y, row, live, 0, C, B, FP32)
    ms = tl.where(valid1, cs1, -float("inf"))
    mt = tl.where(valid1, ct1, -float("inf"))
    # Shared masks carry no mass. A one-sided mask leaves that side's minimum at -inf
    # and sends the row to the general support and missing-mass calculation.
    support = valid1 & ((cs1 > -float("inf")) | (ct1 > -float("inf")))
    smin = tl.where(support, cs1, float("inf"))
    tmin = tl.where(support, ct1, float("inf"))
    MARK_DOUBLE: tl.constexpr = not FP32 and RECIPROCAL != 0.0 and B < C and ROWS == 1
    flags = bad1.to(tl.int32)
    bound_score: tl.constexpr = 16.0 / RECIPROCAL if MARK_DOUBLE else 0.0
    if MARK_DOUBLE:
        flags |= ((tl.abs(cs1) > bound_score) | (tl.abs(ct1) > bound_score)).to(tl.int32) * 4
    # With every |z| below this bound, sum(2**64 * p * (z - max(z)) / T) fits.
    bound = ((0.5 * DBL_MAX * EXP_UNWIDE) / C) * per_t
    if not FP32:
        flags |= ((tl.abs(cs1) > bound) & (cs1 > -float("inf"))).to(tl.int32) * 2
    for start in range(B, C, B):
        k, valid, cs, ct, bad = _pair(Z, Y, row, live, start, C, B, FP32)
        ms = tl.maximum(ms, tl.where(valid, cs, -float("inf")))
        mt = tl.maximum(mt, tl.where(valid, ct, -float("inf")))
        if CHECK:
            support = valid & ((cs > -float("inf")) | (ct > -float("inf")))
            smin = tl.minimum(smin, tl.where(support, cs, float("inf")))
            tmin = tl.minimum(tmin, tl.where(support, ct, float("inf")))
        flags = flags | bad.to(tl.int32)
        if MARK_DOUBLE:
            flags |= ((tl.abs(cs) > bound_score) | (tl.abs(ct) > bound_score)).to(tl.int32) * 4
        if not FP32:
            flags |= ((tl.abs(cs) > bound) & (cs > -float("inf"))).to(tl.int32) * 2
    s_max = tl.max(ms, 1)
    t_max = tl.max(mt, 1)
    invalid_scores = tl.max(flags & 1, 1) != 0
    if FP32:
        cross_scale = tl.full([ROWS], EXP_WIDE, tl.float64)
    else:
        # With huge gaps, averaging the cross term first avoids overflowing its row sum.
        cross_scale = tl.where(
            tl.max(flags & 2, 1) != 0, 1.0 / triton.next_power_of_2(C), EXP_WIDE
        ).to(tl.float64)
    args = (Z, Y, row, live, per_t, s_max, t_max, cross_scale, valid1, cs1, ct1, invalid_scores)
    if CHECK:
        normal = ((s_max - tl.min(smin, 1)) * per_t <= 16) & (
            (t_max - tl.min(tmin, 1)) * per_t <= 16
        )
        normal &= ~invalid_scores & (s_max > -float("inf")) & (t_max > -float("inf"))
        if tl.reshape(normal, ()):
            normal, s_norm, t_norm, log_c, value = _bounded_stats(
                Z, Y, row, live, per_t, s_max, t_max, valid1, cs1, ct1, C, B, GRAD
            )
        else:
            zero = tl.zeros([ROWS], tl.float64)
            s_norm, t_norm, log_c, value = zero, zero, zero, zero
        if tl.reshape(normal, ()):
            missing = tl.zeros([ROWS], tl.float64)
            count = tl.full([ROWS], C, tl.int32)
            bad = invalid_scores
        else:
            s_norm, t_norm, log_c, missing, count, value, bad = _precise_stats(
                *args, ROWS, C, B, FP32, GRAD, RECIPROCAL
            )
    elif MARK_DOUBLE:
        if tl.max(flags & 5) == 0:
            s_norm, t_norm, log_c, missing, count, value, bad = _precise_stats(
                *args, ROWS, C, B, FP32, GRAD, RECIPROCAL, True
            )
            # The positive tail on these bounded diffuse rows makes the sign an
            # unambiguous marker for the backward kernel.
            s_norm = -s_norm
        else:
            s_norm, t_norm, log_c, missing, count, value, bad = _precise_stats(
                *args, ROWS, C, B, FP32, GRAD, RECIPROCAL
            )
    else:
        s_norm, t_norm, log_c, missing, count, value, bad = _precise_stats(
            *args, ROWS, C, B, FP32, GRAD, RECIPROCAL, RECIP_MASS=RECIP_MASS
        )
    return s_max, t_max, s_norm, t_norm, log_c, missing, count, value, bad, valid1, cs1, ct1, k1


@triton.jit
def _partition_value(ss, serr, ts, terr, cross, cerr):
    """Round log(ss) - cross / ts after splitting the binary exponent from the logarithm."""
    magnitude = (ss.to(tl.int64, bitcast=True) >> 52) - 1023
    scale = ((1023 - magnitude) << 52).to(tl.float64, bitcast=True)
    exponent = magnitude.to(tl.float64)
    anchor = exponent * 6.93147180369123816490e-01
    log_mantissa = log(ss * scale) + exponent * 1.90821492927058770002e-10
    q = cross / ts
    q_error = (tl.fma(-ts, q, cross) + (q * terr - cerr)) / ts
    high = anchor - q
    err = (anchor - (high - (high - anchor))) + (-q - (high - anchor))
    return high + (log_mantissa + ((err - q_error) - serr / ss))


@triton.jit
def _stats(Z, Y, row, live, per_t, ROWS: tl.constexpr, C: tl.constexpr, B: tl.constexpr,
           FP32: tl.constexpr, GRAD: tl.constexpr, ORDINARY_ROWS: tl.constexpr = False,
           RECIPROCAL: tl.constexpr = 0.0, STREAM_FP32: tl.constexpr = False, RECIP_MASS: tl.constexpr = False):  # fmt: skip
    """Try one read of bounded FP64 scores, retaining the centered rescue for
    missing support, nearly equal normalizers and concentrated distributions."""
    if not FP32 and RECIPROCAL != 0.0 and ROWS == 1 and B < C:
        # Unchanged logits can carry a small residual determined wholly by the difference
        # of normalizers. Keep those rows, masked rows and extreme rows on the centered path.
        _, valid0, s0, t0, bad0 = _pair(Z, Y, row, live, 0, C, B, False)
        reject = bad0 | (
            valid0
            & ((s0 == t0) | (tl.abs(s0 * RECIPROCAL) > 16.0) | (tl.abs(t0 * RECIPROCAL) > 16.0))
        )
        if tl.max(reject.to(tl.int32)) == 0:
            ss = tl.zeros([1], tl.float64)
            ts = ss
            cross = ss
            serr, terr, cerr = ss, ss, ss
            delta, derr = ss, ss
            flags = tl.full([1, B], False, tl.int1)
            mind = tl.full([1, B], float("inf"), tl.float64)
            maxd = tl.full([1, B], -float("inf"), tl.float64)
            smax = tl.full([1, B], -float("inf"), tl.float64)
            tmax = smax
            for start in range(0, C, B):
                k, valid, cs, ct, bad = _pair(Z, Y, row, live, start, C, B, False)
                s = cs * RECIPROCAL
                t = ct * RECIPROCAL
                flags |= (
                    bad | (valid & (cs == ct)) | (valid & ((tl.abs(s) > 16.0) | (tl.abs(t) > 16.0)))
                )
                smax = tl.maximum(smax, tl.where(valid, s, -float("inf")))
                tmax = tl.maximum(tmax, tl.where(valid, t, -float("inf")))
                d = s - t
                mind = tl.minimum(mind, tl.where(valid, d, float("inf")))
                maxd = tl.maximum(maxd, tl.where(valid, d, -float("inf")))
                es = tl.where(valid, exp(s), 0.0)
                et = tl.where(valid, exp(t), 0.0)
                ss, serr = _add(ss, serr, tl.sum(es, 1), True)
                ts, terr = _add(ts, terr, tl.sum(et, 1), True)
                product = s * et
                cross, cerr = _add(cross, cerr, tl.sum(product, 1), True)
                cerr -= tl.sum(tl.fma(s, et, -product), 1)
                delta, derr = _add(delta, derr, tl.sum(es - et, 1), True)
            ratio = delta / ts
            log_c = log1p(ratio)
            value = _partition_value(ss, serr, ts, terr, cross, cerr)
            sn = 1.0 / ss
            tn = -1.0 / ts
            use = (tl.max(flags.to(tl.int32), 1) == 0) & (ss > 2) & (ts > 2)
            use &= (tl.max(maxd, 1) - tl.min(mind, 1) > 0.25) & (tl.abs(ratio) > 1.0e-4)
            use &= (tl.abs(ratio) < 0.5) & (value > 2.0)
            use &= tl.abs(log_c - (tl.max(smax, 1) - tl.max(tmax, 1))) > 1.0e-5
            # Negative student and teacher reciprocals identify the one-read row in backward.
            if tl.reshape(use, ()):
                k1, valid1, cs1, ct1, bad1 = _pair(Z, Y, row, live, 0, C, B, False)
                zero = tl.zeros([1], tl.float64)
                return (
                    zero,
                    zero,
                    -sn,
                    tn,
                    log_c,
                    zero,
                    tl.full([1], C, tl.int32),
                    value,
                    ~use,
                    valid1,
                    cs1,
                    ct1,
                    k1,
                )
    if STREAM_FP32 and GRAD and FP32 and ROWS == 1 and B < C:
        k, valid, cs, ct, bad = _pair(Z, Y, row, live, 0, C, B, True)
        reject = bad | (valid & ((cs == ct) | (tl.abs(cs * per_t) > 8) | (tl.abs(ct * per_t) > 8)))
        # Sparse changes make a few classes determine the normalizer difference.
        # Keep those, shared values, and exceptional support on the centered path.
        diff = (cs - ct) * per_t
        adjacent = tl.gather(diff, (tl.arange(0, B)[None, :] ^ 1), 1)
        varied = tl.sum((tl.abs(diff - adjacent) > 0.03125).to(tl.int32)) >= B // 8
        if (tl.max(reject.to(tl.int32)) == 0) & varied:
            zero = tl.zeros([1], tl.float32)
            accept, sn, tn, log_c, value = _bounded_stats(
                Z, Y, row, live, per_t, zero, zero, valid, cs, ct, C, B, GRAD, True
            )
            if tl.reshape(accept, ()):
                return (
                    zero,
                    zero,
                    sn,
                    tn,
                    log_c,
                    zero.to(tl.float64),
                    tl.full([1], C, tl.int32),
                    value,
                    ~accept,
                    valid,
                    cs,
                    ct,
                    k,
                )
    return _centered_stats(
        Z,
        Y,
        row,
        live,
        per_t,
        ROWS,
        C,
        B,
        FP32,
        GRAD,
        ORDINARY_ROWS,
        RECIPROCAL,
        RECIP_MASS,
    )


@triton.jit
def _gradient(cs, ct, valid, s_max, t_max, s_norm, t_norm, log_c, bad, factor, per_t,
              FULL_ROW: tl.constexpr, ORDINARY: tl.constexpr = False,
              RECIPROCAL: tl.constexpr = 0.0, FINITE_DOUBLE: tl.constexpr = False, ORIGIN: tl.constexpr = False,
              LONG_RATIO: tl.constexpr = False, RECIP_MASS: tl.constexpr = False):  # fmt: skip
    """(q-p) / T using log(q/p) near equality, even when both round to one."""
    s, t, d = _scaled(cs, ct, valid, s_max, t_max, per_t, ORDINARY or FINITE_DOUBLE, RECIPROCAL)
    if per_t.dtype == tl.float64 and not FULL_ROW:
        # Write the difference using one of its two probabilities. For a
        # positive large ratio it is -q expm1(-ratio), avoiding overflow.
        ratio = d - log_c
        close = (tl.abs(d) < 1) & (tl.abs(log_c) < 1)
        positive = (ratio >= 1) & ~close
        power = tl.where(positive, s - s_norm, t - t_norm)
        if ORIGIN:
            mass = (exp(tl.where(positive, s, t)) * tl.where(positive, s_norm, -t_norm)) * EXP_WIDE
        else:
            mass = exp(power) * EXP_WIDE if FINITE_DOUBLE else wide_exp(power)
        e = _double_expm1(tl.where(close, d, tl.where(positive, -ratio, ratio)))
        c = _double_expm1(tl.where(tl.abs(log_c) < 1, log_c, 0.0))
        direct = tl.where(e == c, mass * ratio, (mass * (e - c)) / (1.0 + c))
        g = tl.where(close, direct, tl.where(positive, -mass, mass) * e)
        g = tl.where(valid, g, 0.0)
        f = _divide_temperature(factor, per_t, RECIPROCAL)
        wide_factor = f * EXP_UNWIDE
        usable = (tl.abs(f) < float("inf")) & (tl.abs(wide_factor) >= DBL_MIN)
        scaled = tl.where(
            usable, g * wide_factor, _divide_temperature(g * EXP_UNWIDE * factor, per_t, RECIPROCAL)
        )
        return tl.where(bad, float("nan"), scaled + 0.0)
    if ORDINARY:
        valid = valid & (cs > -float("inf"))
        p = normal_exp32(tl.where(valid, t, 0.0)) * (-t_norm if LONG_RATIO else t_norm)
        q = normal_exp32(tl.where(valid, s, 0.0)) * s_norm
        unwide = 1.0
    elif RECIP_MASS:
        p = wide_exp(t) * t_norm
        q = wide_exp(s) * s_norm
        unwide = EXP_UNWIDE
    else:
        p = wide_exp(t - t_norm)
        q = wide_exp(s - s_norm)
        unwide = EXP_UNWIDE
    ratio = d - log_c
    near = tl.abs(ratio) < 1
    if LONG_RATIO:
        near = tl.abs(ratio) < 1.0
        relative = bounded_expm1(tl.where(near, -tl.abs(ratio), 0.0))
        g = tl.where(near, tl.where(ratio > 0.0, -q, p) * relative, q - p)
    elif ORDINARY:
        # The sixth-degree Taylor remainder on |r| < 1/8 is below 1e-9 relative.
        near = tl.abs(ratio) < 0.125
        r = tl.where(near, ratio, 0.0)
        series = tl.fma(r, 1.0 / 720, 1.0 / 120)
        series = tl.fma(r, series, 1.0 / 24)
        series = tl.fma(r, series, 1.0 / 6)
        series = tl.fma(r, series, 0.5)
        g = tl.where(near, p * tl.fma(r * r, series, r), q - p)
    else:
        g = tl.where(near, p * bounded_expm1(tl.where(near, ratio, 0.0)), q - p)
    g = tl.where(valid, g, 0.0)
    # Redistributing rounding noise would give a swapped pair (the partitions coincide) a
    # derivative at every untouched class.
    threshold: tl.constexpr = 64 * (2.0**-23 if per_t.dtype == tl.float32 else 2.0**-52)
    if FULL_ROW and per_t.dtype == tl.float32:
        total = bounded_tile_sum(g, 0.0, g.shape[1], 1) if ORDINARY else tl.sum(g.to(tl.float64), 1)
        # The signed normalizer already contains the row-sum correction when the peaks
        # coincide (to float32 work's rounding).
        reliable = (s_max == t_max) | (tl.abs(log_c) <= threshold)
        g -= tl.where(reliable, 0.0, (q * unwide) * total.to(g.dtype)[:, None])
    if ORDINARY:
        # Zero a shared mask after the full-row pivot correction as well.
        scaled = tl.where(valid, g, 0.0) * factor
    elif per_t.dtype == tl.float64:
        f = factor / per_t
        wide_factor = f * unwide
        usable = (tl.abs(f) < float("inf")) & (tl.abs(wide_factor) >= DBL_MIN)
        scaled = tl.where(usable, g * wide_factor, (g * unwide * factor) / per_t)
        if FULL_ROW:
            # The written entries' sum (exact two-part), less q times it, one rounding each:
            # it then sums to zero within the entries' own roundings (half an ulp each),
            # common peaks included. A log ratio of the partitions far above the row's
            # entries (common peaks, log_c 0.5 at gradients 1e-4 of it) leaves each entry
            # one rounding of it times q, which this also removes.
            high, low = tl.reduce((scaled, scaled * 0.0), 1, add_parts)
            total = (high + low)[:, None]
            fix = ~(tl.abs(log_c) <= threshold) & (tl.abs(total) < float("inf"))
            scaled = tl.where(fix, tl.fma(-(q * unwide), total, scaled), scaled)
    else:
        scaled = (g.to(tl.float64) * unwide) * factor
    return tl.where(bad, float("nan"), scaled + 0.0)


@triton.jit
def _short_sum(x, error, B: tl.constexpr, bound: tl.constexpr):
    high, low = bounded_tile_parts(x, error, B, bound)
    total = high + low
    return total, (high - total) + low


@triton.jit
def _short_log(high, low):
    """Log of a positive two-part sum, using log(m) = 2 atanh((m-1)/(m+1))."""
    # Shift the mantissa into [1/sqrt(2), sqrt(2)]. The next term beyond degree
    # 13 contributes less than 5e-13. Keep the division residual beside its quotient.
    exponent = ((high.to(tl.int32, bitcast=True) + 0x4AFB0D) >> 23) - 127
    power = ((127 - exponent) << 23).to(tl.float32, bitcast=True)
    m, rem = high * power, low * power
    num, den = m - 1.0, m + 1.0
    den_error = (m - (den - 1.0)) + rem
    inv = 1.0 / den
    u = num * inv
    u_error = (tl.fma(-u, den, num) + (rem - u * den_error)) * inv
    u2 = u * u
    poly = tl.fma(u2, 1.0 / 13, 1.0 / 11)
    poly = tl.fma(poly, u2, 1.0 / 9)
    poly = tl.fma(poly, u2, 1.0 / 7)
    poly = tl.fma(poly, u2, 1.0 / 5)
    poly = tl.fma(poly, u2, 1.0 / 3)
    tail = tl.fma(2.0 * u * u2, poly, 2.0 * u_error)
    return (
        exponent.to(tl.float64) * 0.6931471805599453
        + (2.0 * u).to(tl.float64)
        + tail.to(tl.float64)
    )


@triton.jit
def _short_partition_ratio(ss, sl, ts, tlow, sinv, tinv, terr):
    """Compute log1p of a nonnegative relative difference to preserve close partitions."""
    delta, rem = add_parts(ss, sl, -ts, -tlow)
    negative = delta < 0.0
    serr = sinv * (tl.fma(-ss, sinv, 1.0) - sl * sinv)
    inv = tl.where(negative, sinv, tinv)
    error = tl.where(negative, serr, terr)
    sign = tl.where(negative, -1.0, 1.0)
    relative = tl.fma(sign * delta, inv, tl.fma(sign * delta, error, sign * rem * inv))
    return sign * log1p(relative)


@triton.jit
def _short_scale(g, factor, live, G):
    """A normal power-of-two scale is exact in float32 for accepted short rows."""
    f = factor.to(tl.float32)
    power = (f.to(tl.int32, bitcast=True) & 0x7FFFFF) == 0
    power &= (tl.abs(f) >= 2.0**-70) & (tl.abs(f) <= 2.0**80) & (factor == f.to(tl.float64))
    if tl.sum((live & ~power).to(tl.int32), 0) == 0:
        output = (g * f[:, None]).to(G.dtype.element_ty)
    else:
        # Per row, as a one-row launch would choose.
        wide = (g.to(tl.float64) * factor[:, None]).to(G.dtype.element_ty)
        output = tl.where(power[:, None], (g * f[:, None]).to(G.dtype.element_ty), wide)
    return output


@triton.jit
def _sparse_gradient(x, y, peak, temperature, valid, teacher, teacher_total, student_total):
    hi, lo = _sparse_mass_difference(x, y, peak[:, None], temperature)
    hi, lo = tl.where(valid, hi, 0.0), tl.where(valid, lo, 0.0)
    dh, dl = tl.reduce((hi, lo), 1, add_parts)
    prob = teacher / teacher_total[:, None]
    residual = tl.fma(-prob, dh[:, None], hi) + tl.fma(-prob, dl[:, None], lo)
    # For a dominant class form both complements directly, excluding that class.
    dominant = prob > 0.5
    oh, ol = tl.reduce((tl.where(dominant, 0.0, hi), tl.where(dominant, 0.0, lo)), 1, add_parts)
    other = (tl.sum(tl.where(dominant, 0.0, teacher), 1) / teacher_total)[:, None]
    pivot = tl.fma(-prob, oh[:, None], other * hi) + tl.fma(-prob, ol[:, None], other * lo)
    return tl.where(dominant, pivot, residual) / student_total[:, None]


@triton.jit
def _sparse_bfloat_rows(Z, Y, V, G, R, D, row, live, temperature, up,
                        C: tl.constexpr, B: tl.constexpr, ROWWISE: tl.constexpr,
                        DIVIDED: tl.constexpr, VALUES: tl.constexpr, GRAD: tl.constexpr):  # fmt: skip
    """Handle finite short BF16 rows with few differing scores and bounded partition scales."""
    k, valid, x, y, bad = _pair(Z, Y, row, live, 0, C, B, False)
    sm = tl.max(tl.where(valid, x, -float("inf")), 1)
    tm = tl.max(tl.where(valid, y, -float("inf")), 1)
    peak = tl.maximum(sm, tm)
    low = tl.min(tl.where(valid, tl.minimum(x, y), float("inf")), 1)
    take = live & (tl.sum((valid & (x != y)).to(tl.int32), 1) <= C // 4)
    take &= (tl.max(bad.to(tl.int32), 1) == 0) & ((peak - low) <= 128.0 * temperature)
    shift = tl.abs(sm - tm)
    # An arbitrarily small change of maximum can straddle the range boundary.
    # Retain the tighter range for substantial changes of dominant score.
    tiny_shift = (shift <= 1.0e-6 * temperature) & (
        (tl.minimum(sm, tm) - low) <= 16.0 * temperature
    )
    take &= (
        (sm == tm)
        | tiny_shift
        | ((shift <= 4.0 * temperature) & ((peak - low) <= 16.0 * temperature))
    )
    if tl.sum(take.to(tl.int32), 0) != 0:
        s = (x - sm[:, None]) / temperature
        teacher = tl.where(valid, exp((y - peak[:, None]) / temperature), 0.0)
        teacher_total = tl.sum(teacher, 1)
        if VALUES:
            first = tl.min(tl.where(valid & (x == sm[:, None]), k, B), 1)
            tail = tl.sum(tl.where(valid & (k != first[:, None]), exp(s), 0.0), 1)
            v = log1p(tail) - tl.sum(tl.where(valid, s * teacher, 0.0), 1) / teacher_total
            tl.store(V + row, v.to(V.dtype.element_ty), take)
        if GRAD:
            student_total = tl.sum(tl.where(valid, exp((x - peak[:, None]) / temperature), 0.0), 1)
            g = _sparse_gradient(
                x, y, peak, temperature, valid, teacher, teacher_total, student_total
            )
            factor = scale(up, R, D, row, take, ROWWISE, DIVIDED) / temperature
            g = g * factor[:, None] + 0.0
            tl.store(G + row[:, None] * C + k, g.to(G.dtype.element_ty), take[:, None] & valid)
    return live & ~take


@triton.jit
def _max2(a, b, c, d):
    return tl.maximum(a, c), tl.maximum(b, d)


@triton.jit
def _bfloat_sums(a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7):
    """Two two-part sums, two minima, an any-flag and a count in one reduction."""
    return (
        a0 + b0, a1 + b1, a2 + b2, a3 + b3,
        tl.minimum(a4, b4), tl.minimum(a5, b5), tl.maximum(a6, b6), a7 + b7,
    )  # fmt: skip


@triton.jit
def _sum2(a, b, c, d):
    return a + c, b + d


@triton.jit
def _tile_parts(x, error, B: tl.constexpr, BOUND: tl.constexpr):
    """bounded_tile_parts before its row sums (the same rounding, summed by the caller)."""
    bias: tl.constexpr = 1.5 * B * BOUND
    high = (x + bias) - bias
    low = (x - high) + error
    return high, low


@triton.jit
def _two_part(high, low):
    total = high + low
    return total, (high - total) + low


@triton.jit
def _bfloat_rows(Z, Y, V, G, R, D, row, live, per_t, up,
                       C: tl.constexpr, B: tl.constexpr, ROWWISE: tl.constexpr,
                       DIVIDED: tl.constexpr, VALUES: tl.constexpr, GRAD: tl.constexpr):  # fmt: skip
    """Handle finite, diffuse bfloat16 rows whose near-equal probabilities remain distinct.
    The row reductions share a few tuple reductions (one cross-warp exchange each)."""
    k, valid, cs, ct, invalid = _pair(Z, Y, row, live, 0, C, B, True)
    sm, tm = tl.reduce(
        (tl.where(valid, cs, -float("inf")), tl.where(valid, ct, -float("inf"))), 1, _max2
    )
    s = tl.where(valid, (cs - sm[:, None]) * per_t, 0.0)
    t = tl.where(valid, (ct - tm[:, None]) * per_t, 0.0)
    es = tl.where(valid, exp(s), 0.0)
    et = tl.where(valid, exp(t), 0.0)
    # The float32 sums need what they assume, not a fixed spread: exp(s) and exp(t) are at
    # most 1 (bounded_tile_parts with bound 1) and normal above -80, and the cross sum
    # takes |s exp(t)| <= 16. Rows outside take the general path.
    wide = (s < -80.0) | (t < -80.0) | (tl.abs(s * et) > 15.0)
    invalid |= valid & (wide | (cs == -float("inf")) | (ct == -float("inf")))
    sh, sl0 = _tile_parts(es, 0.0, B, 1)
    th, tl0 = _tile_parts(et, 0.0, B, 1)
    sh, sl0, th, tl0, smin, tmin, any_invalid, differing = tl.reduce(
        (
            sh, sl0, th, tl0,
            tl.where(valid, s, float("inf")), tl.where(valid, t, float("inf")),
            invalid.to(tl.int32), (valid & (cs != ct)).to(tl.int32),
        ),
        1,
        _bfloat_sums,
    )  # fmt: skip
    ss, sl = _two_part(sh, sl0)
    ts, tlow = _two_part(th, tl0)
    sinv, tinv = 1.0 / ss, 1.0 / ts
    terr = tinv * (tl.fma(-ts, tinv, 1.0) - tlow * tinv)
    log_c = _short_partition_ratio(ss, sl, ts, tlow, sinv, tinv, terr)
    # An ulp of a shared tail probability can affect many equal classes.
    # The full row arithmetic retains their signed residuals.
    rs, rt = tl.reduce(
        (
            (valid & ((s == smin[:, None]) | (s == 0.0))).to(tl.int32),
            (valid & ((t == tmin[:, None]) | (t == 0.0))).to(tl.int32),
        ),
        1,
        _sum2,
    )
    repeated = (rs > C // 2) | (rt > C // 2)
    # Rows with a dominant class (ss or ts below 1.25) stay: log(ss) and the cross sum are
    # both nonnegative, and the dominant entry's p expm1(ratio) keeps its relative accuracy.
    ordinary = ~repeated
    ordinary &= any_invalid == 0
    ordinary &= differing > C // 4
    # Each row is accepted or rejected on its own (masks, not block scalars), so grouping
    # rows in one program gives every row the result of a one-row launch.
    value_accepted = live & ordinary
    # Near-zero gradient lanes may require the general signed normalizer. Keep
    # their scalar on the same route as a forward without a gradient.
    if VALUES:
        if tl.sum(value_accepted.to(tl.int32), 0) != 0:
            dot = s * et
            dh, dl = _tile_parts(dot, tl.fma(s, et, -dot), B, 16)
            dh, dl = tl.reduce((dh, dl), 1, _sum2)
            cross, remainder = _two_part(dh, dl)
            weighted = cross * tinv
            error = tl.fma(cross, tinv, -weighted) + tl.fma(cross, terr, remainder * tinv)
            value = (_short_log(ss, sl) - weighted.to(tl.float64)) - error.to(tl.float64)
            tl.store(V + row, value.to(V.dtype.element_ty), value_accepted)
    accepted = value_accepted
    if GRAD:
        ratio = (s - t) - log_c[:, None]
        close = valid & (tl.abs(ratio) < 2.0e-6)
        accepted &= tl.sum(close.to(tl.int32), 1) == 0
        if tl.sum(accepted.to(tl.int32), 0) != 0:
            q = es * sinv[:, None]
            p = et * tinv[:, None]
            near = tl.abs(ratio) < 1.0
            g = tl.where(near, p * bounded_expm1(tl.where(near, ratio, 0.0)), q - p)
            factor = _factor(scale(up, R, D, row, accepted, ROWWISE, DIVIDED), per_t)
            output = _short_scale(g, factor, accepted, G)
            tl.store(G + row[:, None] * C + k, output, accepted[:, None] & valid)
    return accepted, value_accepted


# One-read streamed rows (float32 work), with s = z / T, t = t / T, p = softmax(t) and
# q = softmax(s). One pass gives the running maxima, both normalizers (rescaled float64 sums),
# W = sum exp(t - max t) (max s - s), whose terms stay nonnegative as the student's running
# maximum moves (the sum shifts exactly with it), and R0 = sum p expm1(x0), x0 = s - t - r0,
# about the running log ratio r0 of the normalizers (each tile's terms about the ratio with
# that tile; the float64 sum moves exactly to each new ratio). Then
#   H(p, q) = log sum exp(s - max s) + W / sum exp(t - max t)
# adds two nonnegative terms. The exact ratio is r0 + log1p(R0); about its float32 rounding r,
#   q - p = p (expm1(x) - R) / (1 + R),  x = s - t - r,  R = expm1(r0 - r + log1p(R0)),
# with |R| near float32's resolution, and identical logits give a zero gradient exactly.
# Masked, nonfinite and extreme rows take the general path (_forward with ONLINE).
# Saved row state (_STRIDE float64): max t, log sum exp(t - max t), r, R, unused, kind.
_ONLINE = (2048, 4)
_BFLOAT_LANES = 64  # lanes of a grouped BF16 register program
_BFLOAT_SPLIT = 1024  # register rows this wide run the fast and rescue passes separately
_STRIDE = 6
_STATE = tl.constexpr(_STRIDE)
_ONE_READ = tl.constexpr(0.0)
_FAR_READ = tl.constexpr(1.0)  # one-read rows with terms beyond exp(_FAR) (_far_residual)
_FAR_SLICE = tl.constexpr(256)  # lanes per slice of _far_residual
_GENERAL = tl.constexpr(2.0)
_EXTREME = tl.constexpr(60.0)  # |x0| beyond this: exp(x0) leaves float32's comfortable range
# 0 < max |x| below this: the gradient's second-order part (about x^2,
# as at classes the two rows share) is below the float32 rounding of the first-order sums
# (eps32 |x|), so the general path computes the row in float64.
_NEAR = tl.constexpr(2.0**-8)
# Float32 exponentials below exp(_FAR) near the end of float32's normal range (exp(-87.3)):
# in a chunk with such terms (one flag per chunk, then per row: _FAR_READ) a teacher term under
# a larger student term is written from the student's, and the gradient's exponential is taken
# in halves, whose product rounds once into the subnormals; the other chunks and rows run the
# plain arithmetic. A chunk with terms beyond it loses at most 2^-118.7 of the loss per class
# (the term times its score gap, at most 155 there with reach <= _EXTREME), C 2^-118.7 in all:
# within the bound's C eps |loss| from a loss of 2^-95.7. Smaller losses of such rows take the
# general path.
_FAR = tl.constexpr(-80.0)
_TINY_LOSS = tl.constexpr(2.0**-90)


@triton.jit
def _expm1_32(x):
    """expm1 in float32: Taylor series through x**9 for |x| < 1/2, else exp(x) - 1."""
    small = tl.abs(x) < 0.5
    y = tl.where(small, x, 0.0)
    p = tl.fma(y, 1.0 / 362880, 1.0 / 40320)
    p = tl.fma(y, p, 1.0 / 5040)
    p = tl.fma(y, p, 1.0 / 720)
    p = tl.fma(y, p, 1.0 / 120)
    p = tl.fma(y, p, 1.0 / 24)
    p = tl.fma(y, p, 1.0 / 6)
    p = tl.fma(y, p, 0.5)
    return tl.where(small, tl.fma(y * y, p, y), tl.exp(tl.where(small, 0.0, x)) - 1.0)


@triton.jit
def _centered(cs, ct, per_t, r):
    """(cs - ct) / T - r, keeping the rounding error of the difference (two-sum)."""
    d = cs - ct
    back = d - cs
    error = (cs - (d - back)) + (-ct - back)
    return tl.fma(d, per_t, -r) + error * per_t


_PARTIAL = tl.constexpr(12)  # float64 words of a chunk's partial statistics


@triton.jit(do_not_specialize=["TEMP"])
def _online_partial(
    Z, Y, P, TEMP, C: tl.constexpr, B: tl.constexpr, CHUNKS: tl.constexpr, TAIL: tl.constexpr = 0
):
    """A row chunk's statistics, independent of the other chunks: maxima, exact counts at
    them and float32 sums of the rest, W about the chunk's student maximum, the chunk's
    log ratio r_c of the normalizers and m = sum exp(t - max t) expm1(s - t - r_c).
    TAIL: the last chunk holds at most TAIL columns, and the program of the chunk before it
    also takes them, on a tile of TAIL lanes (its partial goes to the last slot)."""
    row = tl.program_id(0).to(tl.int64)
    P = doubles(P)
    per_t = _per_t(TEMP, True)
    shift, shared = rows.shared_shift(Z, Y, row, C, C)
    chunk = tl.program_id(1)
    _online_block(Z, Y, P, row, chunk, chunk, shift, shared, per_t, C, B, CHUNKS)
    if TAIL:
        if chunk == CHUNKS - 2:
            # The tail's window positions (CHUNKS - 1) B ... on TAIL lanes.
            tail = (CHUNKS - 1) * (B // TAIL)
            _online_block(Z, Y, P, row, tail, CHUNKS - 1, shift, shared, per_t, C, TAIL, CHUNKS)


@triton.jit
def _online_block(Z, Y, P, row, block, slot, shift, shared, per_t, C: tl.constexpr,
                  B: tl.constexpr, CHUNKS: tl.constexpr):  # fmt: skip
    """_online_partial of window block `block` (B lanes) into partial slot `slot`."""
    j, _, valid, inside = rows.span(block, shift, shared, B, C)
    if inside:
        cs = rows.load(Z, row, j, shift, valid, C, False).to(tl.float32)
        ct = rows.load(Y, row, j, shift, valid, C, False).to(tl.float32)
    else:
        cs = rows.load(Z, row, j, shift, valid, C, True).to(tl.float32)
        ct = rows.load(Y, row, j, shift, valid, C, True).to(tl.float32)
    finite = (cs < float("inf")) & (ct < float("inf"))  # NaN fails too
    general = valid & (~finite | (cs == -float("inf")) | (ct == -float("inf")))
    use = valid & ~general
    s = tl.where(use, cs * per_t, -float("inf"))
    t = tl.where(use, ct * per_t, -float("inf"))
    sm = tl.max(s, 1)
    tm = tl.max(t, 1)
    at_s = use & (s == sm[:, None])
    at_t = use & (t == tm[:, None])
    ds = s - sm[:, None]
    dt = t - tm[:, None]
    es = tl.where(use & ~at_s, tl.exp(ds), 0.0)
    et = tl.where(use & ~at_t, tl.exp(dt), 0.0)
    # 2: the general path's entries; 1: terms beyond exp(_FAR) of the chunk's maxima.
    flags = tl.max(tl.where(general, 2, (use & (tl.minimum(ds, dt) < _FAR)).to(tl.int32)), 1)
    count_s = tl.sum(at_s.to(tl.int32), 1)
    count_t = tl.sum(at_t.to(tl.int32), 1)
    rest_s = tl.sum(es, 1)
    rest_t = tl.sum(et, 1)
    r = (sm + tl.log(count_s.to(tl.float32) + rest_s)) - (
        tm + tl.log(count_t.to(tl.float32) + rest_t)
    )
    r = tl.where(tl.abs(r) < float("inf"), r, 0.0)
    et = tl.where(at_t, 1.0, et)
    w = tl.sum(et * tl.where(use, sm[:, None] - s, 0.0), 1)
    x = tl.where(use, _centered(cs, ct, per_t, r[:, None]), 0.0)
    if tl.reshape(flags, ()) == 1:
        m = _far_residual(Z, Y, row, block, shift, per_t, tm, r, C, B)
    else:
        m = tl.sum(et * _expm1_32(x), 1)
    out = P + (row * CHUNKS + slot) * _PARTIAL + tl.arange(0, 1)
    tl.store(out, sm.to(tl.float64))
    tl.store(out + 1, tm.to(tl.float64))
    tl.store(out + 2, count_s.to(tl.float64))
    tl.store(out + 3, rest_s.to(tl.float64))
    tl.store(out + 4, count_t.to(tl.float64))
    tl.store(out + 5, rest_t.to(tl.float64))
    tl.store(out + 6, w.to(tl.float64))
    tl.store(out + 7, r.to(tl.float64))
    tl.store(out + 8, m.to(tl.float64))
    tl.store(out + 9, tl.max(tl.abs(x), 1).to(tl.float64))
    tl.store(out + 10, flags.to(tl.float64))


@triton.jit
def _far_residual(Z, Y, row, block, shift, per_t, tm, r, C: tl.constexpr, B: tl.constexpr):
    """m of _online_block for a chunk with terms beyond exp(_FAR): a teacher term there under
    a larger student term gives p expm1(x) as q (-expm1(-x)), q = exp((t - tm) + x) (p e^x,
    where p would round to zero). It rereads the chunk (cached) in slices of _FAR_SLICE lanes,
    so that this branch needs no more registers than the common one."""
    m = tl.zeros([1], tl.float32)
    for start in tl.static_range(0, B, _FAR_SLICE):
        j = (block * B + start + tl.arange(0, _FAR_SLICE))[None, :]
        k = j - shift
        valid = (k >= 0) & (k < C)
        cs = rows.load(Z, row, j, shift, valid, C, True).to(tl.float32)
        ct = rows.load(Y, row, j, shift, valid, C, True).to(tl.float32)
        use = valid & (cs < float("inf")) & (ct < float("inf"))
        use &= (cs > -float("inf")) & (ct > -float("inf"))
        dt = ct * per_t - tm[:, None]
        et = tl.where(use, tl.where(dt == 0, 1.0, tl.exp(dt)), 0.0)
        x = tl.where(use, _centered(cs, ct, per_t, r[:, None]), 0.0)
        flip = use & (dt < _FAR) & (x > 0)
        e1 = _expm1_32(tl.where(flip, -x, x))
        m += tl.sum(tl.where(flip, -tl.exp(dt + x) * e1, et * e1), 1)
    return m


@triton.jit
def _sum_tail(count, rest, w, top, top_all):
    """A row's sum minus one from its chunks (rescaled by w; counts at the row's maximum
    exactly) in float64."""
    at = top == top_all
    exact = tl.sum(tl.where(at, count, 0.0), 0) - 1.0
    return exact + tl.sum(tl.where(at, 0.0, count * w) + rest * w, 0)


@triton.jit(do_not_specialize=["N"])
def _online(P, V, S, N, CHUNKS: tl.constexpr, CP: tl.constexpr, VALUES: tl.constexpr):
    """Merge a row's chunk statistics exactly (float64): its loss and gradient state, or the
    general path's mark."""
    row = tl.program_id(0).to(tl.int64)
    S = doubles(S)
    P = doubles(P)
    c = tl.arange(0, CP)
    here = c < CHUNKS
    part = P + (row * CHUNKS + c) * _PARTIAL
    ninf = -float("inf")
    count_s = tl.load(part + 2, here, other=0.0)
    count_t = tl.load(part + 4, here, other=0.0)
    some = here & (count_s + tl.load(part + 3, here, other=0.0) > 0)
    sm = tl.where(some, tl.load(part, here, other=0.0), ninf)
    tm = tl.where(some, tl.load(part + 1, here, other=0.0), ninf)
    rest_s = tl.load(part + 3, here, other=0.0)
    rest_t = tl.load(part + 5, here, other=0.0)
    SM = tl.max(sm, 0)
    TM = tl.max(tm, 0)
    ws = tl.where(some, tl.exp(sm - SM), 0.0)
    wt = tl.where(some, tl.exp(tm - TM), 0.0)
    ts = _sum_tail(count_s, rest_s, ws, sm, SM)
    tt = _sum_tail(count_t, rest_t, wt, tm, TM)
    zt = 1.0 + tt
    zt_c = (count_t + rest_t) * wt
    w = tl.sum(
        tl.where(
            some, (tl.load(part + 6, here, other=0.0) + (SM - sm) * (count_t + rest_t)) * wt, 0.0
        ),
        0,
    )
    loss = log1p(ts) + w / zt
    gap = SM - TM
    ratio = (gap + log1p(ts)) - log1p(tt)
    r = ratio.to(tl.float32)
    rc = tl.load(part + 7, here, other=0.0)
    move = tl.where(some, rc - r.to(tl.float64), 0.0)
    m = tl.load(part + 8, here, other=0.0) * wt
    residual = tl.sum(expm1(move) * zt_c + exp(move) * m, 0) / zt
    # Equal tails (as for permuted or shifted classes): the ratio is the maxima's difference.
    # Equal float64 sums do not prove equal tails (tails below 2^-53 of tied maxima round
    # away): the shortcut keeps the rows whose chunks' residual is exactly zero as well
    # (identical rows), and the others with equal sums take the general path. Keeping their
    # computed residual instead would give a permutation float32 noise where x = 0.
    equal = ts == tt
    ambiguous = equal & (residual != 0)
    r = tl.where(equal, gap.to(tl.float32), r)
    residual = tl.where(equal, expm1(gap - r.to(tl.float64)), residual)
    reach = tl.max(tl.where(some, tl.load(part + 9, here, other=0.0) + tl.abs(move), 0.0), 0)
    flags = tl.max(tl.load(part + 10, here, other=0.0), 0)
    ok = (flags < 2) & (reach <= _EXTREME) & (tl.abs(loss) < float("inf"))
    ok &= (reach == 0) | (reach >= _NEAR)
    ok &= (flags == 0) | (loss >= _TINY_LOSS)
    ok &= ~ambiguous
    ok &= tl.abs(residual) < float("inf")
    kind = tl.where(ok, tl.where(flags == 0, _ONE_READ, _FAR_READ), _GENERAL)
    state = S + row * _STATE
    tl.store(state, TM)
    tl.store(state + 1, log1p(tt))
    tl.store(state + 2, r.to(tl.float64))
    tl.store(state + 3, residual)
    tl.store(state + 5, kind)
    if VALUES:
        if ok:
            tl.store(V + row, loss.to(V.dtype.element_ty))


@triton.jit
def _online_gradient(state, cs, ct, valid, per_t, factor, FAR: tl.constexpr = False):
    """(q - p) / T times the row factor from _online's state: p (expm1(x) - R) / (1 + R).
    FAR (_FAR_READ rows): beyond exp(_FAR), q (-expm1(-x) - R e^-x) / (1 + R) with q = p e^x."""
    tm = tl.load(state).to(tl.float32)[:, None]
    zt = tl.load(state + 1).to(tl.float32)[:, None]
    r = tl.load(state + 2).to(tl.float32)[:, None]
    big = tl.load(state + 3)[:, None]
    if FAR:
        g = _far_gradient(cs, ct, valid, per_t, factor, tm, zt, r, big)
    else:
        p = tl.where(valid, tl.exp((ct * per_t - tm) - zt), 0.0)
        e = _expm1_32(tl.where(valid, _centered(cs, ct, per_t, r), 0.0))
        g = p * ((e - big.to(tl.float32)) * (1.0 / (1.0 + big)).to(tl.float32))
        g = g * factor.to(tl.float32)
    return tl.where(valid, g, 0.0) + 0.0


@triton.jit
def _far_gradient(cs, ct, valid, per_t, factor, tm, zt, r, big):
    """_online_gradient of a _FAR_READ row before its mask: beyond exp(_FAR) a smaller teacher
    term gives q (-expm1(-x) - R e^-x), q = p e^x, and the exponential is taken in halves,
    whose product rounds once into the subnormals."""
    a = (ct * per_t - tm) - zt
    x = tl.where(valid, _centered(cs, ct, per_t, r), 0.0)
    flip = (a < _FAR) & (x > 0)
    arg = tl.where(flip, a + x, a)
    half = arg < _FAR
    p = tl.where(valid, tl.exp(tl.where(half, 0.5 * arg, arg)), 0.0)
    e = _expm1_32(tl.where(flip, -x, x))
    rest = big.to(tl.float32)
    f = tl.where(flip, -e - rest * (1.0 + e), e - rest) * (1.0 / (1.0 + big)).to(tl.float32)
    up = factor.to(tl.float32)
    return tl.where(half, (p * (f * up)) * p, (p * f) * up)


@triton.jit(do_not_specialize=["N", "UP", "TEMP"])
def _online_backward(Z, Y, R, D, S, G, N, UP, TEMP, C: tl.constexpr, B: tl.constexpr,
                     ROWWISE: tl.constexpr, DIVIDED: tl.constexpr, CHUNKS: tl.constexpr = 0,
                     TAIL: tl.constexpr = 0, ALIAS: tl.constexpr = False):  # fmt: skip
    """The gradient of _online's rows in aligned row chunks (rows.py; G may be Z); rows it
    left to the general path are _backward's. TAIL: as _online_partial, the program of the
    chunk before the last also takes the last chunk's (at most TAIL) columns."""
    if ALIAS:
        Z = G
    S = doubles(S)
    row = tl.program_id(0).to(tl.int64)
    rows_ = row + tl.arange(0, 1)
    state = S + rows_ * _STATE
    kind = tl.reshape(tl.load(state + 5), ())
    if kind == _GENERAL:
        return
    per_t = _per_t(TEMP, True)
    factor = _factor(scale(UP, R, D, rows_, rows_ < N, ROWWISE, DIVIDED)[:, None], per_t)
    shift, shared = rows.shared_shift(Z, Y, row, C, C)
    _, same = rows.shared_shift(Z, G, row, C, C)
    shared &= same
    shift = tl.where(shared, shift, 0)
    chunk = tl.program_id(1)
    if kind == _ONE_READ:
        _online_window(Z, Y, G, state, row, chunk, shift, shared, per_t, factor, C, B)
        if TAIL:
            if chunk == CHUNKS - 2:
                tail = (CHUNKS - 1) * (B // TAIL)
                _online_window(Z, Y, G, state, row, tail, shift, shared, per_t, factor, C, TAIL)
    else:
        # _FAR_READ: the far terms' arithmetic (_far_gradient).
        _online_window(Z, Y, G, state, row, chunk, shift, shared, per_t, factor, C, B, True)
        if TAIL:
            if chunk == CHUNKS - 2:
                tail = (CHUNKS - 1) * (B // TAIL)
                _online_window(
                    Z, Y, G, state, row, tail, shift, shared, per_t, factor, C, TAIL, True
                )


@triton.jit
def _online_window(Z, Y, G, state, row, block, shift, shared, per_t, factor, C: tl.constexpr,
                   B: tl.constexpr, FAR: tl.constexpr = False):  # fmt: skip
    """_online_backward's gradient at window block `block` of B lanes."""
    j, _, valid, inside = rows.span(block, shift, shared, B, C)
    if inside:
        _online_chunk(Z, Y, G, state, row, j, shift, valid, per_t, factor, C, False, FAR)
    else:
        _online_chunk(Z, Y, G, state, row, j, shift, valid, per_t, factor, C, True, FAR)


@triton.jit
def _online_chunk(Z, Y, G, state, row, j, shift, valid, per_t, factor, C: tl.constexpr,
                  MASKED: tl.constexpr, FAR: tl.constexpr = False):  # fmt: skip
    cs = rows.load(Z, row, j, shift, valid, C, MASKED).to(tl.float32)
    ct = rows.load(Y, row, j, shift, valid, C, MASKED).to(tl.float32)
    g = _online_gradient(state, cs, ct, valid, per_t, factor, FAR)
    rows.store(G, row, j, shift, valid, g, C, MASKED)


# Rows below _NARROW classes (float32 work): one row per thread, _NARROW_ROWS rows per program
# (_narrow). A register tile spreads such a row over up to 16 lanes, and each of them repeats
# the row's float64 statistics; here the float64 row sums and logarithms run once per row.
# The arithmetic is the general path's (_precise_stats and _gradient on a full row), on the
# rows it takes: finite rows (BF16: whose scores differ in more than a quarter of the
# classes, as _sparse_bfloat_rows takes the others), and for a gradient normalizers further
# apart than _aligned_difference's threshold. _narrow_rescue runs the general path on the
# rows it leaves. Its column loops are unrolled, and the compile time grows steeply with C
# (minutes at 20 to 31 classes on an A100, in every process through jax-triton): from 12
# classes the general path takes the rows, as fast or faster except float32 forward calls.
_NARROW = 12
_NARROW_ROWS, _NARROW_WARPS = 128, 4
_GATE = 1024  # rows per _narrow_rescue program (a program none of whose rows is marked returns)
_CHUNK_FLAGS = tl.constexpr(128)  # flags _narrow_rescue scans at a time
_SUM_PARTS = tl.constexpr(1024)  # program sums the last rescue program adds at a time


@triton.jit
def _narrow_scaled(Z, Y, base, k, live, s_max, t_max, per_t):
    """_scaled of column k: the centered, temperature-scaled scores and their difference."""
    cs = tl.load(Z + base + k, live, other=0).to(tl.float32)
    ct = tl.load(Y + base + k, live, other=0).to(tl.float32)
    a = cs - s_max
    b = ct - t_max
    ea = (cs - (a - (a - cs))) + (-s_max - (a - cs))
    eb = (ct - (b - (b - ct))) + (-t_max - (b - ct))
    return a * per_t, b * per_t, ((a - b) + (ea - eb)) * per_t


@triton.jit
def _narrow_gradient(Z, Y, base, k, live, s_max, t_max, s_norm, t_norm, log_c, per_t,
                     RECIP_MASS: tl.constexpr):  # fmt: skip
    """_gradient's q (wide) and unscaled difference of column k."""
    s, t, d = _narrow_scaled(Z, Y, base, k, live, s_max, t_max, per_t)
    if RECIP_MASS:
        p = wide_exp(t) * t_norm
        q = wide_exp(s) * s_norm
    else:
        p = wide_exp(t - t_norm)
        q = wide_exp(s - s_norm)
    ratio = d - log_c
    near = tl.abs(ratio) < 1
    return tl.where(near, p * bounded_expm1(tl.where(near, ratio, 0.0)), q - p), q


@triton.jit(do_not_specialize=["N", "UP", "TEMP"])
def _narrow(Z, Y, R, D, V, G, F, N, UP, TEMP, K, C: tl.constexpr, ROWS: tl.constexpr,
            ROWWISE: tl.constexpr, DIVIDED: tl.constexpr, VALUES: tl.constexpr,
            GRAD: tl.constexpr, RECIP_MASS: tl.constexpr, SPARSE: tl.constexpr,
            REDUCE: tl.constexpr = False, ALIAS: tl.constexpr = False):  # fmt: skip
    """Values and gradients of rows of C classes, one row per thread; F marks the rows left
    to _narrow_rescue, 2 where their value is stored (G may be Z: each column is read before
    it is written)."""
    if ALIAS:
        Z = G
    _reset_count(K, REDUCE)
    per_t = _per_t(TEMP, True)
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = row < N
    base = row * C
    s_max = tl.full([ROWS], -float("inf"), tl.float32)
    t_max = s_max
    other = row < 0
    differing = tl.zeros([ROWS], tl.int32)
    for k in tl.static_range(C):
        cs = tl.load(Z + base + k, live, other=0).to(tl.float32)
        ct = tl.load(Y + base + k, live, other=0).to(tl.float32)
        # NaN, infinities and masked classes take the general path.
        other |= ~((tl.abs(cs) < float("inf")) & (tl.abs(ct) < float("inf")))
        s_max = tl.maximum(s_max, cs)
        t_max = tl.maximum(t_max, ct)
        if SPARSE:
            differing += (cs != ct).to(tl.int32)
    if SPARSE:
        other |= differing <= C // 4
    sties = tl.zeros([ROWS], tl.int32)
    tties = sties
    st = tl.zeros([ROWS], tl.float64)
    tt = st
    cross = st
    delta = st
    for k in tl.static_range(C):
        s, t, d = _narrow_scaled(Z, Y, base, k, live, s_max, t_max, per_t)
        # Centered differences beyond float32 take the general path's guards.
        other |= ~((tl.abs(s) < float("inf")) & (tl.abs(t) < float("inf")))
        other |= ~(tl.abs(d) < float("inf"))
        es = wide_exp(s)
        te, tf = exp_parts(t)
        et = (te * EXP_WIDE) * tf
        sties += (s == 0).to(tl.int32)
        tties += (t == 0).to(tl.int32)
        st += tl.where(s < 0, es, 0.0).to(tl.float64)
        tt += tl.where(t < 0, et, 0.0).to(tl.float64)
        if VALUES:
            cross += tl.where(te > 0, (s.to(tl.float64) * te) * (tf.to(tl.float64) * EXP_WIDE), 0.0)
        if GRAD:
            near = tl.abs(d) < 1
            v = tl.where(near, et * bounded_expm1(tl.where(near, d, 0.0)), es - et)
            delta += v.to(tl.float64)
    stail = st * EXP_UNWIDE + (sties - 1).to(tl.float64)
    ttail = tt * EXP_UNWIDE + (tties - 1).to(tl.float64)
    ts = 1.0 + ttail
    s_norm = log1p(stail)
    other &= live
    mine = live & ~other
    if VALUES:
        value = s_norm - cross * (EXP_UNWIDE / ts)
        tl.store(V + row, value.to(V.dtype.element_ty), mine)
    if GRAD:
        c = (delta * EXP_UNWIDE) / ts
        # _aligned_difference refines normalizers this close (the general path); the value
        # stays this kernel's, as without a gradient (F 2).
        close = mine & (tl.abs(c) <= 2.0**-23)
        mine &= ~close
        tl.store(F + row, tl.where(close, 2, other.to(tl.int32)).to(tl.uint8), live)
    else:
        tl.store(F + row, other.to(tl.uint8), live)
    if GRAD:
        log_c = tl.where(c < -0.5, log((1.0 + stail) / ts), log1p(c)).to(tl.float32)  # pyright: ignore[reportPossiblyUnboundVariable] -- set under GRAD above
        if RECIP_MASS:
            sn = (1.0 / (1.0 + stail)).to(tl.float32)
            tn = (1.0 / ts).to(tl.float32)
        else:
            sn = s_norm.to(tl.float32)
            tn = log1p(ttail).to(tl.float32)
        factor = _factor(scale(UP, R, D, row, live, ROWWISE, DIVIDED), per_t).to(tl.float32)
        total = tl.zeros([ROWS], tl.float64)
        for k in tl.static_range(C):
            g, q = _narrow_gradient(
                Z, Y, base, k, live, s_max, t_max, sn, tn, log_c, per_t, RECIP_MASS
            )
            total += g.to(tl.float64)
        # _gradient's full-row correction: the signed normalizer already holds it where the
        # peaks or the partitions coincide.
        threshold: tl.constexpr = 64 * 2.0**-23
        reliable = (s_max == t_max) | (tl.abs(log_c) <= threshold)
        total = tl.where(reliable, 0.0, total).to(tl.float32)
        for k in tl.static_range(C):
            g, q = _narrow_gradient(
                Z, Y, base, k, live, s_max, t_max, sn, tn, log_c, per_t, RECIP_MASS
            )
            g -= (q * EXP_UNWIDE) * total
            out = (g.to(tl.float64) * EXP_UNWIDE) * factor + 0.0
            tl.store(G + base + k, out.to(G.dtype.element_ty), mine)


@triton.jit(do_not_specialize=["N", "UP", "TEMP"])
def _narrow_rescue(Z, Y, R, D, V, G, F, N, UP, TEMP, L, T, C: tl.constexpr,
                   B: tl.constexpr, ROWS: tl.constexpr, GATE: tl.constexpr,
                   ROWWISE: tl.constexpr, DIVIDED: tl.constexpr, VALUES: tl.constexpr,
                   GRAD: tl.constexpr, RECIP_MASS: tl.constexpr, SPARSE: tl.constexpr,
                   REDUCE: tl.constexpr = False, MEAN: tl.constexpr = False,
                   PARTS: tl.constexpr = 1, ALIAS: tl.constexpr = False):  # fmt: skip
    """The rows _narrow marked, through the general register path (BF16: the sparse rows
    first), ROWS at a time; a program of GATE rows that _narrow left none of skips that.
    F 2: the row's value is stored. REDUCE: each program then writes the sum of its rows'
    values to T and counts itself in T's first word (which _narrow zeroed); the last one
    adds the PARTS sums in order into L (their sum, or their mean over the N rows): no
    reduction launch."""
    if ALIAS:
        Z = G
    start = tl.program_id(0).to(tl.int64) * GATE
    gate = start + tl.arange(0, GATE)
    if tl.max(tl.load(F + gate, gate < N, other=0).to(tl.int32), 0) != 0:
        _rescue_gate(
            Z, Y, R, D, V, G, F, N, UP, TEMP, start, C, B, ROWS, GATE, ROWWISE, DIVIDED,
            VALUES, GRAD, RECIP_MASS, SPARSE,
        )  # fmt: skip
    if REDUCE:
        _last_program_sum(V, L, T, gate, N, MEAN, PARTS)


@triton.jit
def _reset_count(K, REDUCE: tl.constexpr):
    """Zero the count of _narrow_rescue's programs (_last_program_sum: the first word of its
    sums array K), before they run."""
    if REDUCE:
        if tl.program_id(0) == 0:
            tl.store(K.to(tl.pointer_type(tl.int32), bitcast=True), 0)


@triton.jit
def _last_program_sum(V, L, T, gate, N, MEAN: tl.constexpr, PARTS: tl.constexpr):
    """Write this program's sum of V[gate] to T[1 + program] and count the program in T's
    first word (an int32; one array, as the native plan replay has few buffer slots); the
    program that counts last adds the PARTS sums in index order (deterministic) into L."""
    V = doubles(V)
    count = T.to(tl.pointer_type(tl.int32), bitcast=True)
    T = doubles(T) + 1
    pid = tl.program_id(0)
    tl.store(T + pid, tl.sum(tl.load(V + gate, gate < N, other=0.0), 0))
    # acq_rel: this program's sum is visible before its count; the last reads every sum.
    if tl.atomic_add(count, 1, sem="acq_rel") == tl.num_programs(0) - 1:
        total = tl.zeros([], tl.float64)
        for begin in range(0, PARTS, _SUM_PARTS):
            part = begin + tl.arange(0, _SUM_PARTS)
            total += tl.sum(tl.load(T + part, part < PARTS, other=0.0, cache_modifier=".cg"), 0)
        if MEAN:
            total /= tl.cast(N, tl.float64)
        tl.store(L, total.to(L.dtype.element_ty))


@triton.jit
def _rescue_gate(Z, Y, R, D, V, G, F, N, UP, TEMP, start, C: tl.constexpr, B: tl.constexpr,
                 ROWS: tl.constexpr, GATE: tl.constexpr, ROWWISE: tl.constexpr,
                 DIVIDED: tl.constexpr, VALUES: tl.constexpr, GRAD: tl.constexpr,
                 RECIP_MASS: tl.constexpr, SPARSE: tl.constexpr):  # fmt: skip
    """_narrow_rescue's rows: the general path on the marked rows of one gate."""
    per_t = _per_t(TEMP, True)
    for chunk in range(0, GATE, _CHUNK_FLAGS):
        index = start + chunk + tl.arange(0, _CHUNK_FLAGS)
        pending = tl.load(F + index, index < N, other=0) != 0
        # The marked rows only: each step takes the block of ROWS rows from the first one.
        while tl.max(pending.to(tl.int32), 0) != 0:
            first = tl.min(tl.where(pending, index, 2**62), 0)
            pending &= index >= first + ROWS
            row = first + tl.arange(0, ROWS)
            # A block ends at its chunk: the next chunk (or program) owns the rows past it.
            live = (row < N) & (row < start + chunk + _CHUNK_FLAGS)
            flag = tl.load(F + row, live, other=0)
            live &= flag != 0
            if SPARSE:
                live = _sparse_bfloat_rows(
                    Z, Y, V, G, R, D, row, live, f64(TEMP), UP, C, B, ROWWISE, DIVIDED, VALUES,
                    GRAD,
                )  # fmt: skip
            if tl.sum(live.to(tl.int32), 0) != 0:
                _general_rows(
                    Z, Y, R, D, V, G, row, live, flag == 2, per_t, UP, C, B, ROWS, True,
                    ROWWISE, DIVIDED, VALUES, GRAD, False, 0.0, False, RECIP_MASS,
                )  # fmt: skip


# 16-bit rows of _NARROW to _REGISTER classes (float32 work): a [NB, ROWS, W] tile puts each
# row on W / VEC lanes (_MEDIUM_ELEMS columns per thread, blocks in registers), and the
# element loop is a float32 core: with d = max - x (exact for 16-bit scores) and k = log2(e) / T,
#   S = count(d = 0) + sum_{d > 0} 2^(-k d)  (student and teacher),  W = sum 2^(-k d_t) d_s,
#   H(p, q) = log S_s + W / (T S_t),
# two nonnegative terms, and q - p = 2^(-k d_s - log2 S_s) - 2^(-k d_t - log2 S_t). Two
# cross-lane reductions; each row finishes in float64 (log S as log1p(S - 1)). Rows with
# non-finite or masked entries, centered scores closer than _MEDIUM_GAP T (the float32 gradient's
# error would exceed its allowance), or (BF16) changes in at most a quarter of the classes
# are marked, and _narrow_rescue runs the general path on them. A forward that saves state
# stores (max z, max t, log2 S_s, log2 S_t) per row (NaN first: a marked row); backward
# reads it.
_MEDIUM_ELEMS = 16
_MEDIUM_GAP = tl.constexpr(0.0625)
_MEDIUM_STATE = tl.constexpr(4)
_MEDIUM_CLOSE = tl.constexpr(2.0**-20)  # normalizers this close take the general path


@triton.jit
def _sum6(a, b, c, d, e, f, g, h, i, j, k, m):
    return a + g, b + h, c + i, d + j, e + k, f + m


_MEDIUM_PAD = tl.constexpr(-1.0e30)  # scores off the row: below any row _medium takes
# The padding must lie this far below both maxima (log2 units of the scaled scores): its
# exponentials are then exactly 0 in float32 and it adds nothing to any row statistic.
_MEDIUM_FAR = tl.constexpr(192.0)


@triton.jit
def _medium_tile(Z, Y, row, live, C: tl.constexpr, NB: tl.constexpr, W: tl.constexpr):
    k = tl.arange(0, NB)[:, None, None] * W + tl.arange(0, W)[None, None, :]
    valid = live[None, :, None] & (k < C)
    offsets = row[None, :, None] * C + k
    z = tl.load(Z + offsets, valid, other=_MEDIUM_PAD).to(tl.float32)
    y = tl.load(Y + offsets, valid, other=_MEDIUM_PAD).to(tl.float32)
    return offsets, valid, z, y


@triton.jit
def _row_closed(q, p):
    """q - p with the full-row correction of _gradient: a dominant class's q and p can both
    round to 1 and lose its entry (student [0, 31 x -30], teacher [0, 31 x -31] at T = 1:
    -1.8e-12), so the row sum of the differences, taken back in proportion to q, restores
    it and closes the row sum."""
    g = q - p
    total = tl.sum(tl.sum(g, 2), 0)
    return g - q * total[None, :, None]


@triton.jit
def _padding_near(zm, ym, temperature):
    """Rows whose padding (_MEDIUM_PAD) is not _MEDIUM_FAR below both maxima: at very high
    temperatures its exponentials would enter the sums."""
    k2 = 1.4426950408889634 / temperature
    near = (zm.to(tl.float64) - _MEDIUM_PAD) * k2 < _MEDIUM_FAR
    return near | ((ym.to(tl.float64) - _MEDIUM_PAD) * k2 < _MEDIUM_FAR)


@triton.jit
def _medium_marks(z, y, zm, ym, gap, check, value, temperature, live, C: tl.constexpr):
    """The rows _medium leaves to the general path."""
    other = ~(check == 0) | ~(tl.abs(value) < float("inf"))
    other |= ~(gap >= temperature.to(tl.float32) * _MEDIUM_GAP)
    other |= (zm <= _MEDIUM_PAD) | (ym <= _MEDIUM_PAD) | _padding_near(zm, ym, temperature)
    differing = tl.sum(tl.sum((z != y).to(tl.int32), 2), 0)
    other |= differing <= C // 4
    return other & live


@triton.jit
def _medium_gradient(es, et, inv_s, inv_t, factor):
    """(q - p) times the row factor with _gradient's full-row correction: the float64 row sum
    of the float32 differences, taken back in proportion to q, closes the zero sum."""
    q = es * inv_s[None, :, None]
    p = et * inv_t[None, :, None]
    g = tl.fma(es, inv_s[None, :, None], -p)
    total = tl.sum(tl.sum(g.to(tl.float64), 2), 0).to(tl.float32)
    g -= q * total[None, :, None]
    return (g.to(tl.float64) * factor[None, :, None]) + 0.0


@triton.jit
def _medium_precise(V, S, G, F, R, D, UP, row, live, offsets, valid, z, y, zm, ym, gap, ds, dt,
                    temperature, C: tl.constexpr, ROWS: tl.constexpr, ROWWISE: tl.constexpr,
                    DIVIDED: tl.constexpr, VALUES: tl.constexpr, GRAD: tl.constexpr,
                    SAVE: tl.constexpr):  # fmt: skip
    """_medium for float32 scores: the differences d to the maxima round once, exponentials are
    exp32 (corrected, subnormal results kept), the row sums are float64, the state holds the
    reciprocal normalizers (q = e / S, as the general path), and the gradient takes the
    full-row correction (zero sum at float32 outputs). Rows with changes in at most a quarter
    of the classes take the general path, whose sparse arithmetic keeps untouched classes."""
    per_t = (1.0 / temperature).to(tl.float32)
    es = exp32(-(ds * per_t))
    et = exp32(-(dt * per_t))
    top_s = ds == 0
    top_t = dt == 0
    # Exponentials below 2^-64 (a spread beyond about 44 T) can give subnormal gradient
    # entries, which q = e / S would round more than once: the general path's scaled
    # exponentials take those rows.
    tiny = ((es < 2.0**-64) | (et < 2.0**-64)) & (z > 0.5 * _MEDIUM_PAD)
    # The tails and the cross sum add in float64; the counts are exact in float32.
    count_s, tail_s, count_t, tail_t, cross, check = tl.reduce(
        tl.reduce(
            (
                top_s.to(tl.float32), tl.where(top_s, 0.0, es).to(tl.float64),
                top_t.to(tl.float32), tl.where(top_t, 0.0, et).to(tl.float64),
                (et * ds).to(tl.float64), (ds - ds) + (dt - dt) + tiny.to(tl.float32),
            ),
            2,
            _sum6,
        ),
        0,
        _sum6,
    )  # fmt: skip
    ss = count_s.to(tl.float64) + tail_s
    ts = count_t.to(tl.float64) + tail_t
    value = log1p((count_s - 1).to(tl.float64) + tail_s) + cross / (temperature * ts)
    other = _medium_marks(z, y, zm, ym, gap, check, value, temperature, live, C)
    # Normalizers this close may be equal (a shifted permutation): the general path's
    # aligned difference keeps the untouched classes' gradient at zero.
    other |= live & (tl.abs(ss - ts) <= _MEDIUM_CLOSE * ts)
    mine = live & ~other
    tl.store(F + row, other.to(tl.uint8), live)
    if VALUES:
        tl.store(V + row, value.to(V.dtype.element_ty), mine)
    inv_s = (1.0 / ss).to(tl.float32)
    inv_t = (1.0 / ts).to(tl.float32)
    if SAVE:
        state = tl.join(tl.join(tl.where(other, float("nan"), zm), inv_s), tl.join(ym, inv_t))
        state = tl.reshape(state.to(tl.float64), (ROWS, _MEDIUM_STATE))
        column = tl.arange(0, _MEDIUM_STATE)[None, :]
        tl.store(S + row[:, None] * _MEDIUM_STATE + column, state, live[:, None])
    if GRAD:
        factor = scale(UP, R, D, row, live, ROWWISE, DIVIDED) / temperature
        g = _medium_gradient(es, et, inv_s, inv_t, factor)
        tl.store(G + offsets, g.to(G.dtype.element_ty), valid & mine[None, :, None])


@triton.jit(do_not_specialize=["N", "UP", "TEMP"])
def _medium(Z, Y, R, D, V, S, G, F, N, UP, TEMP, K, C: tl.constexpr, NB: tl.constexpr,
            W: tl.constexpr, ROWS: tl.constexpr, ROWWISE: tl.constexpr, DIVIDED: tl.constexpr,
            VALUES: tl.constexpr, GRAD: tl.constexpr, SAVE: tl.constexpr,
            SPARSE: tl.constexpr, PRECISE: tl.constexpr = False, REDUCE: tl.constexpr = False,
            ALIAS: tl.constexpr = False):  # fmt: skip
    """Values, a joint gradient (GRAD) or saved state (SAVE) of 16-bit rows; F marks the rows
    left to _narrow_rescue. Columns off the row read _MEDIUM_PAD, which adds nothing to the
    sums, so the element loop needs no masks; non-finite entries and differences to the
    maxima beyond float32 make the row's sum of (ds - ds) + (dt - dt) NaN."""
    if ALIAS:
        Z = G
    _reset_count(K, REDUCE)
    S = doubles(S)
    temperature = f64(TEMP)
    k2 = (1.4426950408889634 / temperature).to(tl.float32)
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = row < N
    offsets, valid, z, y = _medium_tile(Z, Y, row, live, C, NB, W)
    zm, ym = tl.reduce(tl.reduce((z, y), 2, _max2), 0, _max2)
    ds = zm[None, :, None] - z
    dt = ym[None, :, None] - y
    # The largest change of the centered scores (invariant to shifting either row): the
    # float32 gradient's error bound holds for rows where it is at least _MEDIUM_GAP T.
    gap = tl.max(tl.max(tl.where(valid, tl.abs(ds - dt), 0.0), 2), 0)
    if PRECISE:
        _medium_precise(
            V, S, G, F, R, D, UP, row, live, offsets, valid, z, y, zm, ym, gap, ds, dt,
            temperature, C, ROWS, ROWWISE, DIVIDED, VALUES, GRAD, SAVE,
        )  # fmt: skip
        return
    es = tl.exp2(-k2 * ds)
    et = tl.exp2(-k2 * dt)
    top_s = ds == 0
    top_t = dt == 0
    count_s, tail_s, count_t, tail_t, cross, check = tl.reduce(
        tl.reduce(
            (
                top_s.to(tl.float32), tl.where(top_s, 0.0, es),
                top_t.to(tl.float32), tl.where(top_t, 0.0, et),
                et * ds, (ds - ds) + (dt - dt),
            ),
            2,
            _sum6,
        ),
        0,
        _sum6,
    )  # fmt: skip
    # log S as log1p(S - 1): a dominant class leaves tails far below 1 ulp of S. Both terms
    # of the value are nonnegative, so float32 keeps it within a few ulps.
    log_s = log1p((count_s - 1) + tail_s)
    ts = count_t + tail_t
    # Once per row in float64: T times the normalizer can exceed float32's range.
    value = log_s.to(tl.float64) + cross.to(tl.float64) / (temperature * ts.to(tl.float64))
    other = ~(check == 0) | ~(tl.abs(value) < float("inf"))
    other |= ~(gap >= temperature.to(tl.float32) * _MEDIUM_GAP)
    other |= (zm <= _MEDIUM_PAD) | (ym <= _MEDIUM_PAD) | _padding_near(zm, ym, temperature)
    other |= tl.abs((count_s + tail_s) - ts) <= _MEDIUM_CLOSE * ts
    if SPARSE:
        differing = tl.sum(tl.sum((z != y).to(tl.int32), 2), 0)
        other |= differing <= C // 4
    other &= live
    mine = live & ~other
    tl.store(F + row, other.to(tl.uint8), live)
    if VALUES:
        tl.store(V + row, value.to(V.dtype.element_ty), mine)
    if SAVE or GRAD:
        ls = log_s * 1.4426950408889634
        lt = log1p((count_t - 1) + tail_t) * 1.4426950408889634
    if SAVE:
        # join(join(a, c), join(b, d)) lays out a, b, c, d.
        state = tl.join(tl.join(tl.where(other, float("nan"), zm), ls), tl.join(ym, lt)).to(  # pyright: ignore[reportPossiblyUnboundVariable] -- set under SAVE or GRAD
            tl.float64
        )
        state = tl.reshape(state, (ROWS, _MEDIUM_STATE))
        column = tl.arange(0, _MEDIUM_STATE)[None, :]
        tl.store(S + row[:, None] * _MEDIUM_STATE + column, state, live[:, None])
    if GRAD:
        factor = scale(UP, R, D, row, live, ROWWISE, DIVIDED) / temperature
        q = tl.exp2(-k2 * ds - ls[None, :, None])  # pyright: ignore[reportPossiblyUnboundVariable] -- set under SAVE or GRAD
        p = tl.exp2(-k2 * dt - lt[None, :, None])  # pyright: ignore[reportPossiblyUnboundVariable] -- set under SAVE or GRAD
        g = _row_closed(q, p).to(tl.float64) * factor[None, :, None] + 0.0
        tl.store(G + offsets, g.to(G.dtype.element_ty), valid & mine[None, :, None])


@triton.jit(do_not_specialize=["N", "UP", "TEMP"])
def _medium_backward(Z, Y, R, D, S, G, F, N, UP, TEMP, C: tl.constexpr, NB: tl.constexpr,
                     W: tl.constexpr, ROWS: tl.constexpr, ROWWISE: tl.constexpr,
                     DIVIDED: tl.constexpr, PRECISE: tl.constexpr = False,
                     ALIAS: tl.constexpr = False):  # fmt: skip
    """The gradient from _medium's saved state: one read of the rows, one write; F marks the
    rows the state marks, for _narrow_rescue."""
    if ALIAS:
        Z = G
    S = doubles(S)
    temperature = f64(TEMP)
    k2 = (1.4426950408889634 / temperature).to(tl.float32)
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = row < N
    zm = tl.load(S + row * _MEDIUM_STATE, live, other=0.0)
    other = live & (zm != zm)
    tl.store(F + row, other.to(tl.uint8), live)
    mine = live & ~other
    ym = tl.load(S + row * _MEDIUM_STATE + 1, live, other=0.0).to(tl.float32)
    ls = tl.load(S + row * _MEDIUM_STATE + 2, live, other=0.0).to(tl.float32)
    lt = tl.load(S + row * _MEDIUM_STATE + 3, live, other=0.0).to(tl.float32)
    offsets, valid, z, y = _medium_tile(Z, Y, row, mine, C, NB, W)
    if PRECISE:
        # The state holds 1 / S_s and 1 / S_t (_medium_precise).
        per_t = (1.0 / temperature).to(tl.float32)
        es = exp32(-((zm.to(tl.float32)[None, :, None] - z) * per_t))
        et = exp32(-((ym[None, :, None] - y) * per_t))
        factor = scale(UP, R, D, row, live, ROWWISE, DIVIDED) / temperature
        g = _medium_gradient(es, et, ls, lt, factor)
    else:
        factor = scale(UP, R, D, row, live, ROWWISE, DIVIDED) / temperature
        q = tl.exp2(-k2 * (zm.to(tl.float32)[None, :, None] - z) - ls[None, :, None])
        p = tl.exp2(-k2 * (ym[None, :, None] - y) - lt[None, :, None])
        g = _row_closed(q, p).to(tl.float64) * factor[None, :, None] + 0.0
    tl.store(G + offsets, g.to(G.dtype.element_ty), valid)


@triton.jit(do_not_specialize=["N", "UP", "TEMP"])
def _forward(
    Z,
    Y,
    R,
    D,
    V,
    S,
    G,
    F,
    N,
    UP,
    TEMP,
    C: tl.constexpr,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    FP32: tl.constexpr,
    ROWWISE: tl.constexpr,
    DIVIDED: tl.constexpr,
    VALUES: tl.constexpr,
    GRAD: tl.constexpr,
    SPLIT: tl.constexpr = 0,
    ORDINARY_ROWS: tl.constexpr = False,
    SAVE_STATS: tl.constexpr = False,
    ALIAS: tl.constexpr = False,
    FIXED_T: tl.constexpr = 0.0,
    RECIPROCAL: tl.constexpr = 0.0,
    STREAM_FP32: tl.constexpr = False,
    RECIP_MASS: tl.constexpr = False,
    REGISTER_BFLOAT: tl.constexpr = False,
    ONLINE: tl.constexpr = False,
    STRIDE: tl.constexpr = 5,
):
    """Row values, and either the register-row gradient or the five saved statistics:
    two maxima, two normalizers (negative first reciprocal on bounded rows, logs elsewhere)
    and their log ratio. ONLINE: only the rows _online left to the general path."""
    if ALIAS:
        Z = G
    S = doubles(S)
    per_t = tl.full((), FIXED_T, tl.float64) if FIXED_T != 0.0 else _per_t(TEMP, FP32)
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = row < N
    if ONLINE:
        if tl.reshape(tl.load(S + row * STRIDE + 5), ()) != _GENERAL:
            return
    value_accepted = row < 0
    if REGISTER_BFLOAT:
        if SPLIT == 2:
            # Rescue pass: only the rows the fast pass flagged (2: value stored).
            flag = tl.load(F + row, live, other=0)
            value_accepted = flag == 2
            live &= flag != 0
        else:
            handled, value_accepted = _bfloat_rows(
                Z, Y, V, G, R, D, row, live, per_t, UP, C, B, ROWWISE, DIVIDED, VALUES, GRAD
            )
            live &= ~handled
        if SPLIT == 1:
            # Fast pass: flag the remaining rows for the rescue pass.
            flag = tl.where(live, tl.where(value_accepted, 2, 1), 0).to(tl.uint8)
            tl.store(F + row, flag, row < N)
            return
        if tl.sum(live.to(tl.int32), 0) == 0:
            return
        live = _sparse_bfloat_rows(
            Z, Y, V, G, R, D, row, live, f64(TEMP), UP, C, B, ROWWISE, DIVIDED, VALUES, GRAD
        )
        if tl.sum(live.to(tl.int32), 0) == 0:
            return
    if not SAVE_STATS:
        _general_rows(
            Z, Y, R, D, V, G, row, live, value_accepted, per_t, UP, C, B, ROWS, FP32, ROWWISE,
            DIVIDED, VALUES, GRAD, ORDINARY_ROWS, RECIPROCAL, STREAM_FP32, RECIP_MASS,
        )  # fmt: skip
    else:
        s_max, t_max, sn, tn, log_c, lost, count, value, bad, valid, cs, ct, k = _stats(
            Z,
            Y,
            row,
            live,
            per_t,
            ROWS,
            C,
            B,
            FP32,
            GRAD,
            ORDINARY_ROWS,
            RECIPROCAL=RECIPROCAL,
            STREAM_FP32=STREAM_FP32,
            RECIP_MASS=RECIP_MASS,
        )
        nan = float("nan")
        if VALUES:
            tl.store(
                V + row, tl.where(bad, nan, value).to(V.dtype.element_ty), live & ~value_accepted
            )
        tl.store(S + row * STRIDE, s_max.to(tl.float64), live)
        tl.store(S + row * STRIDE + 1, t_max.to(tl.float64), live)
        tl.store(S + row * STRIDE + 2, sn, live)
        tl.store(S + row * STRIDE + 3, tn, live)
        tl.store(S + row * STRIDE + 4, tl.where(bad, nan, log_c), live)


@triton.jit
def _general_rows(Z, Y, R, D, V, G, row, live, value_accepted, per_t, UP,
                  C: tl.constexpr, B: tl.constexpr, ROWS: tl.constexpr, FP32: tl.constexpr,
                  ROWWISE: tl.constexpr, DIVIDED: tl.constexpr, VALUES: tl.constexpr,
                  GRAD: tl.constexpr, ORDINARY_ROWS: tl.constexpr, RECIPROCAL: tl.constexpr,
                  STREAM_FP32: tl.constexpr, RECIP_MASS: tl.constexpr):  # fmt: skip
    """The general path's values (but of the rows value_accepted) and register-row gradients
    of the rows live."""
    dtype = per_t.dtype
    s_max, t_max, sn, tn, log_c, lost, count, value, bad, valid, cs, ct, k = _stats(
        Z,
        Y,
        row,
        live,
        per_t,
        ROWS,
        C,
        B,
        FP32,
        GRAD,
        ORDINARY_ROWS,
        RECIPROCAL=RECIPROCAL,
        STREAM_FP32=STREAM_FP32,
        RECIP_MASS=RECIP_MASS,
    )
    nan = float("nan")
    if VALUES:
        tl.store(V + row, tl.where(bad, nan, value).to(V.dtype.element_ty), live & ~value_accepted)
    if GRAD:
        factor = _factor(scale(UP, R, D, row, live, ROWWISE, DIVIDED), per_t).to(dtype)[:, None]
        args = (
            cs,
            ct,
            valid,
            s_max[:, None],
            t_max[:, None],
            tl.abs(sn).to(dtype)[:, None],
            tn.to(dtype)[:, None],
            log_c.to(dtype)[:, None],
            bad[:, None],
            factor,
            per_t,
        )
        g = _gradient(*args, True, RECIP_MASS=RECIP_MASS).to(G.dtype.element_ty)
        tl.store(G + row[:, None] * C + k, g.to(G.dtype.element_ty), valid)


@triton.jit(do_not_specialize=["TEMP"])
def _rescue_partition(Z, Y, S, TEMP, C: tl.constexpr, B: tl.constexpr, FP32: tl.constexpr,
                      ONLINE: tl.constexpr = False, STRIDE: tl.constexpr = 5):  # fmt: skip
    """Refine nearly equal streamed partitions in the saved state before the pullback."""
    row = tl.program_id(0).to(tl.int64) + tl.arange(0, 1)
    S = doubles(S)
    if ONLINE:
        if tl.reshape(tl.load(S + row * STRIDE + 5), ()) != _GENERAL:
            return
    log_ratio = tl.load(S + row * STRIDE + 4)
    # A one-ulp relative change at unit mass is the resolution of a normalized
    # working-precision sum. Equal maxima alone say nothing about its error.
    epsilon: tl.constexpr = 2.0**-23 if FP32 else 2.0**-52
    if tl.reshape(tl.abs(log_ratio) <= epsilon, ()):
        sn = tl.load(S + row * STRIDE + 2)
        tn = tl.load(S + row * STRIDE + 3)
        # The one-read paths store reciprocals; their own admission already
        # requires well-separated partitions. The centered path stores logs.
        centered = (sn >= 0) if FP32 else (tn >= 0)
        if tl.reshape(centered, ()):
            sm = tl.load(S + row * STRIDE)
            tm = tl.load(S + row * STRIDE + 1)
            ratio = expm1(log_ratio)
            # Register rows already hold their scores. Streamed rows reload each
            # tile only inside this rescue and require no per-class scratch.
            empty = tl.full((1, 1), 0.0, tl.float32 if FP32 else tl.float64)
            valid = tl.full((1, 1), False, tl.int1)
            new = _aligned_difference(
                Z,
                Y,
                row,
                row >= 0,
                _per_t(TEMP, FP32),
                sm,
                tm,
                valid,
                empty,
                empty,
                ratio,
                exp(tn),
                1,
                C,
                B,
                FP32,
            )
            tl.store(S + row * STRIDE + 4, tl.where(new == ratio, log_ratio, log1p(new)))


@triton.jit(do_not_specialize=["N", "UP", "TEMP"])
def _backward(
    Z,
    Y,
    R,
    D,
    S,
    G,
    N,
    UP,
    TEMP,
    C: tl.constexpr,
    B: tl.constexpr,
    FP32: tl.constexpr,
    ROWWISE: tl.constexpr,
    DIVIDED: tl.constexpr,
    FULL_ROW: tl.constexpr = False,
    ALIAS: tl.constexpr = False,
    FIXED_T: tl.constexpr = 0.0,
    RECIPROCAL: tl.constexpr = 0.0,
    LONG_RATIO: tl.constexpr = False,
    ONLINE: tl.constexpr = False,
    STRIDE: tl.constexpr = 5,
):
    """Gradient from five saved statistics, B columns per program (G may be Z).
    A full register row also corrects its gradient sum. ONLINE: _online's rows take
    _online_gradient."""
    if ALIAS:
        Z = G
    S = doubles(S)
    per_t = tl.full((), FIXED_T, tl.float64) if FIXED_T != 0.0 else _per_t(TEMP, FP32)
    dtype = per_t.dtype
    row = tl.program_id(0).to(tl.int64) + tl.arange(0, 1)
    live = row < N
    if ONLINE:
        if tl.reshape(tl.load(S + row * STRIDE + 5), ()) != _GENERAL:
            return
    # ONLINE: one program per row walks its blocks (only the rows _online left run).
    first = 0 if ONLINE else tl.program_id(1) * B
    for begin in range(first, C if ONLINE else first + B, B):
        k, valid, cs, ct, bad_ = _pair(Z, Y, row, live, begin, C, B, FP32)
        stat = S + row * STRIDE
        log_c = tl.load(stat + 4, live, other=0.0).to(dtype)[:, None]
        sn = tl.load(stat + 2, live, other=1.0).to(dtype)[:, None]
        s_max = tl.load(stat, live, other=0.0).to(dtype)[:, None]
        t_max = tl.load(stat + 1, live, other=0.0).to(dtype)[:, None]
        s_norm = tl.abs(sn)
        tn = tl.load(stat + 3, live, other=1.0).to(dtype)[:, None]
        args = (
            cs,
            ct,
            valid,
            s_max,
            t_max,
            s_norm,
            tn,
            log_c,
            log_c != log_c,
            _factor(scale(UP, R, D, row, live, ROWWISE, DIVIDED)[:, None], per_t).to(dtype),
            per_t,
        )
        if FP32 and tl.reshape(sn < 0, ()):
            if LONG_RATIO and tl.reshape(tn < 0, ()):
                g = _gradient(*args, FULL_ROW, True, LONG_RATIO=True).to(G.dtype.element_ty)
            else:
                g = _gradient(*args, FULL_ROW, True).to(G.dtype.element_ty)
        elif not FP32 and RECIPROCAL != 0.0 and tl.reshape(sn < 0, ()):
            if tl.reshape(tn < 0, ()):
                g = _gradient(
                    *args,
                    FULL_ROW,
                    RECIPROCAL=RECIPROCAL,
                    FINITE_DOUBLE=True,
                    ORIGIN=True,
                ).to(G.dtype.element_ty)
            else:
                g = _gradient(*args, FULL_ROW, RECIPROCAL=RECIPROCAL, FINITE_DOUBLE=True).to(
                    G.dtype.element_ty
                )
        else:
            g = _gradient(*args, FULL_ROW, RECIPROCAL=RECIPROCAL).to(G.dtype.element_ty)
        tl.store(G + row[:, None] * C + k, g.to(G.dtype.element_ty), valid)


@triton.jit
def _terms(
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
):
    """Residuals r = p - pi and pi at omega."""
    arg = omega - logratio
    h = (log_q + alpha * p) - omega
    prob = tl.minimum(probability(h, alpha, log_alpha, CUT, LARGE, STEPS), 1.0)
    half = alpha * p / 2
    near = (p > TINY) & (arg <= half + 0.6931471805599453) & (arg >= -half - 0.4054651081081644)
    r_near = near_residual(
        tl.where(near, arg, 0.0),
        alpha,
        tl.where(near, p, 1.0),
        tl.where(near, log_p, 0.0),
        EPS,
        RES_STEPS,
    )
    # A zero argument has a zero residual, also where p is subnormal.
    r = tl.where(near, r_near, tl.where(arg == 0, 0.0, p - prob))
    pi = tl.where(near, p - r_near, prob)
    return tl.where(valid, r, 0.0), tl.where(valid, pi, 0.0)


@triton.jit
def _log_pi(
    r,
    p,
    log_p,
    log_q,
    omega,
    alpha,
    log_alpha,
    CUT: tl.constexpr,
    LARGE: tl.constexpr,
    STEPS: tl.constexpr,
):
    """log pi: log1p where r is small (log(p - r) far above a tiny p), else h - W."""
    h = (log_q + alpha * p) - omega
    far = h - lambert_w_exp(h + log_alpha, CUT, LARGE, STEPS)
    near = log_p + log1p(-r / tl.where(p > 0, p, 1.0))
    above = r < -p / 2
    near = tl.where(above, log(tl.where(above, p - r, 1.0)), near)
    return tl.where(r <= p / 2, near, far)


@triton.jit
def _tile(
    Z,
    Y,
    row,
    start,
    s_max,
    t_max,
    s_norm,
    t_norm,
    log_c,
    extra,
    per_t,
    end,
    C: tl.constexpr,
    B: tl.constexpr,
    FP32: tl.constexpr,
):
    """Columns before end and probabilities for the proximal solve on kept student classes.
    Each kept class absorbs an equal share of the teacher mass on masked student classes."""
    k = start + tl.arange(0, B)
    valid = k < end
    cs = work(tl.load(Z + row * C + k, valid, other=0), FP32)
    ct = work(tl.load(Y + row * C + k, valid, other=0), FP32)
    s, t, d = _scaled(cs, ct, valid, s_max, t_max, per_t)
    log_q = s - s_norm
    log_p = t - t_norm
    q = exp(log_q)
    p = exp(log_p)
    kept = cs > -float("inf")
    p = tl.where(kept, p + extra, p)
    log_p = tl.where(kept & (extra > 0), log(p), log_p)
    ratio = tl.where(kept & (extra > 0), log_q - log_p, d - log_c)
    return k, valid, kept, p, q, log_p, log_q, ratio


@triton.jit
def _tail_series(omega, shift, moments, n, alpha):
    """The tail classes' pi sum and its first two derivatives in h at omega, from their
    moments sum (alpha x)^n at omega = shift (float64)."""
    a64 = alpha.to(tl.float64)
    nf = n.to(tl.float64)
    terms = series(nf) * exp((shift.to(tl.float64) - omega.to(tl.float64)) * nf) * moments
    return tl.sum(terms, 0) / a64, tl.sum(terms * nf, 0) / a64, tl.sum(terms * (nf * nf), 0) / a64


@triton.jit
def _series_solve(
    Z, Y, H, row, omega, lo, hi, done, s_max, t_max, s_norm, t_norm, log_c, per_t, alpha,
    log_alpha, C: tl.constexpr, B: tl.constexpr, K: tl.constexpr, EPS: tl.constexpr,
    TINY: tl.constexpr, CUT: tl.constexpr, LARGE: tl.constexpr, STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
):  # fmt: skip
    """The Halley steps of a row on its head classes and tail moments (a row without masked
    student classes): one pass collects the classes with alpha x > TAIL at the bracket's
    lowest omega (x = p exp(ratio + alpha p - lo), pi's limit there) into H and the moments
    of the others, whose pi is then the series of prox._fast_measure. Returns omega and
    whether the steps converged (else the caller's passes solve the row)."""
    shift = lo
    a64 = alpha.to(tl.float64)
    n = tl.arange(0, 8) + 1
    moments = tl.zeros([8], tl.float64)
    total = tl.zeros([], tl.float64)
    spread = total
    count = tl.zeros([], tl.int32)
    zero = tl.zeros([], tl.float32)
    for start in range(0, C, B):
        k, valid, kept, p, q, log_p, log_q, ratio = _tile(
            Z, Y, row, start, s_max, t_max, s_norm, t_norm, log_c, zero, per_t, C, C, B, True
        )
        live = valid & kept
        p64 = exp(log_p.to(tl.float64))
        y = a64 * p64 * exp((ratio.to(tl.float64) + a64 * p64) - shift.to(tl.float64))
        y = tl.where(live, y, 0.0)
        head = live & (y > _SERIES_TAIL)
        y = tl.where(head, 0.0, y)
        power = y
        for m in tl.static_range(8):
            moments += tl.where(n == m + 1, tl.sum(power, 0), 0.0)
            power = power * y
        total += tl.sum(tl.where(live & ~head, p64, 0.0), 0)
        # Sum |r| of the tail at omega = 0 (only the tolerance reads it).
        pi0 = _tail_pi(y, shift, a64)
        spread += tl.sum(tl.where(live & ~head, tl.abs(p64 - pi0), 0.0), 0)
        slot = count + tl.cumsum(head.to(tl.int32), 0) - 1
        keep = head & (slot < K)
        at = (row * K + slot) * 4
        tl.store(H + at, p, keep)
        tl.store(H + at + 1, log_p, keep)
        tl.store(H + at + 2, log_q, keep)
        tl.store(H + at + 3, ratio, keep)
        count += tl.sum(head.to(tl.int32), 0)
    fits = count <= K
    tl.debug_barrier()
    j = tl.arange(0, K)
    held = j < count
    at = (row * K + j) * 4
    p = tl.load(H + at, held, other=0.0)
    log_p = tl.load(H + at + 1, held, other=0.0)
    log_q = tl.load(H + at + 2, held, other=0.0)
    ratio = tl.load(H + at + 3, held, other=0.0)
    resolved = done & False
    f = tl.zeros([], tl.float32)
    magnitude = f
    step = 0
    while ~done & fits & (step < 100):
        r, pi = _terms(
            omega, p, log_p, log_q, ratio, alpha, log_alpha, held, EPS, TINY, CUT, LARGE,
            STEPS, RES_STEPS,
        )  # fmt: skip
        tail, tail_d1, tail_d2 = _tail_series(omega, shift, moments, n, alpha)
        denom = 1 + alpha * pi
        f64_ = tl.sum(r.to(tl.float64), 0) + (total - tail)
        f = f64_.to(tl.float32)
        magnitude = (tl.sum(tl.abs(r).to(tl.float64), 0) + spread).to(tl.float32)
        d1 = (tl.sum((pi / denom).to(tl.float64), 0) + tail_d1).to(tl.float32)
        d2 = (-tl.sum((pi / (denom * denom * denom)).to(tl.float64), 0) - tail_d2).to(tl.float32)
        following, lo, hi, resolved, done = halley(
            omega, f, d1, d2, magnitude, lo, hi, resolved, -float("inf"), EPS, TINY
        )
        omega = tl.where(done, omega, following)
        step += 1
    ok = ~failed(f, magnitude, lo, hi, resolved, EPS, TINY) | (step == 0)
    return omega, fits & ok


@triton.jit
def _tail_pi(y, shift, a64):
    """pi of tail classes at omega = 0 from y = alpha x at omega = shift (float64)."""
    w = y * exp(shift.to(tl.float64))
    poly = 117649.0 / 5040.0
    poly = poly * w - 10.8
    poly = poly * w + 125.0 / 24.0
    poly = poly * w - 8.0 / 3.0
    poly = poly * w + 1.5
    poly = poly * w - 1.0
    return (w + (w * w) * poly) / a64


@triton.jit
def _unit(g, per_t, SAVE: tl.constexpr):
    """A gradient entry over T (_finish), or as a forward saves it: before the factor and T."""
    if SAVE:
        y = g
    else:
        y = _finish(g, per_t)
    return y


@triton.jit(do_not_specialize=["UP", "TEMP", "ALPHA", "LOG_ALPHA"])
def _proximal(  # noqa: C901 - constexpr specializations of one Triton kernel
    Z,
    Y,
    R,
    D,
    S,
    V,
    G,
    H,
    CS,
    UP,
    TEMP,
    ALPHA,
    LOG_ALPHA,
    C: tl.constexpr,
    B: tl.constexpr,
    FP32: tl.constexpr,
    ROWWISE: tl.constexpr,
    DIVIDED: tl.constexpr,
    VALUES: tl.constexpr,
    GRAD: tl.constexpr,
    EPS: tl.constexpr,
    TINY: tl.constexpr,
    CUT: tl.constexpr,
    LARGE: tl.constexpr,
    STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
    ALIAS: tl.constexpr = False,
    SAVE: tl.constexpr = False,
    REPLAY: tl.constexpr = False,
    SERIES: tl.constexpr = 0,
    GATED: tl.constexpr = False,
    SW: tl.constexpr = 0,
):
    """One row per program: the solve for omega, then the loss and the gradient. SAVE and
    REPLAY as prox._kernel: S holds (extra - r or -p) before the factor and T. SERIES > 0
    (float32 work) solves rows without masked student classes on their head classes, at
    most SERIES in H, and tail moments (_series_solve)."""
    if ALIAS:
        Z = G
    row = tl.program_id(0).to(tl.int64)
    if REPLAY:
        if not saved_replay(S, row, C, CS, FP32):
            return
    if GATED:
        # Rows the chunk passes solved (_px_solve).
        if tl.load(doubles(S) + row * SW + 1) != 0:
            return
    if SAVE:
        G = saved_rows(S, FP32) + row * (CS - C)
    per_t = _per_t(TEMP, FP32)
    alpha = work(f64(ALPHA), FP32)
    log_alpha = work(f64(LOG_ALPHA), FP32)
    dtype = per_t.dtype
    rows = row + tl.arange(0, 1)
    s_max, t_max, sn, tn, log_c, lost, count, value, bad, valid1, cs1, ct1, k1 = _stats(
        Z, Y, rows, rows >= 0, per_t, 1, C, B, FP32, True
    )
    s_max, t_max = tl.sum(s_max, 0), tl.sum(t_max, 0)
    s_norm, t_norm = tl.sum(sn, 0).to(dtype), tl.sum(tn, 0).to(dtype)
    log_c = tl.sum(log_c, 0).to(dtype)
    missing = tl.sum(lost, 0).to(dtype)
    extra = missing / tl.sum(count, 0)
    bad = tl.max(bad.to(tl.int32), 0) > 0
    # A row of one tile stays in registers: its probabilities are formed once (as public's
    # register rows), not again in every pass.
    ONE: tl.constexpr = B >= C
    tile0 = _tile(Z, Y, row, 0, s_max, t_max, s_norm, t_norm, log_c, extra, per_t, C, C, B, FP32)
    # The bracket of omega and the pivot (the first largest p + q).
    u_max = tl.full([], -float("inf"), dtype)
    u_min = -u_max
    best = u_max
    pivot = C * 0
    for start in range(0, C, B):
        if ONE:
            k, valid, kept, p, q, log_p, log_q, ratio = tile0
        else:
            k, valid, kept, p, q, log_p, log_q, ratio = _tile(
                Z,
                Y,
                row,
                start,
                s_max,
                t_max,
                s_norm,
                t_norm,
                log_c,
                extra,
                per_t,
                C,
                C,
                B,
                FP32,
            )
        u = tl.where(
            valid,
            tl.where(
                tl.abs(ratio) < 1.0,
                p * bounded_expm1(tl.minimum(tl.maximum(ratio, -1.0), 1.0)),
                q - p,
            ),
            0.0,
        )
        u_max = tl.maximum(u_max, tl.max(tl.where(valid, u, -float("inf")), 0))
        u_min = tl.minimum(u_min, tl.min(tl.where(valid, u, float("inf")), 0))
        pq = tl.where(valid & kept, p + q, -float("inf"))
        largest = tl.max(pq, 0)
        pivot = tl.where(largest > best, start + tl.argmax(pq, 0), pivot)
        best = tl.maximum(best, largest)
    # When both distributions concentrate on the same class, the Hessian of the loss
    # is bounded by twice the student's total tail. The proximal correction then falls
    # below rounding, while forming each teacher probability would round the subnormal
    # tails before they accumulate into the dominant entry.
    tail = tl.maximum(s_norm, t_norm)
    same_peak = (u_max < 0.5) & (u_min > -0.5)
    if ~bad & (missing == 0) & same_peak & (tail < EPS * EPS) & (alpha * tail < EPS / 8):
        if VALUES:
            tl.store(V + rows, value.to(V.dtype.element_ty))
        if SAVE:
            tl.store(G + row * C + C, 1.0)
        elif GRAD:
            factor = _factor(scale(UP, R, D, rows, rows >= 0, ROWWISE, DIVIDED), per_t)[:, None]
            for start in range(0, C, B):
                k, valid, cs, ct, unused = _pair(Z, Y, rows, rows >= 0, start, C, B, FP32)
                g = _gradient(
                    cs,
                    ct,
                    valid,
                    s_max,
                    t_max,
                    s_norm,
                    t_norm,
                    log_c,
                    bad,
                    factor,
                    per_t,
                    B >= C,
                )
                tl.store(G + rows[:, None] * C + k, g.to(G.dtype.element_ty), valid)
        return
    # Safeguarded Halley steps on sum(r(omega)) = 0 in the bracket [lo, hi], which holds 0.
    lo = -alpha * tl.maximum(u_max, 0.0)
    hi = -alpha * tl.minimum(u_min, 0.0)
    hi = tl.where(lo <= hi, hi, lo)
    omega = lo * 0
    done = bad | ((lo == 0) & (hi == 0))
    if SERIES > 0:
        if ~done & (missing == 0):
            omega, done = _series_solve(
                Z, Y, H, row, omega, lo, hi, done, s_max, t_max, s_norm, t_norm, log_c,
                per_t, alpha, log_alpha, C, B, SERIES, EPS, TINY, CUT, LARGE, STEPS,
                RES_STEPS,
            )  # fmt: skip
            omega = tl.where(done, omega, 0.0)
    resolved = done & False
    step = 0
    while ~done & (step < 100):
        f = tl.zeros([], tl.float64)
        magnitude = f
        d1 = f
        d2 = f
        for start in range(0, C, B):
            if ONE:
                k, valid, kept, p, q, log_p, log_q, ratio = tile0
            else:
                k, valid, kept, p, q, log_p, log_q, ratio = _tile(
                    Z,
                    Y,
                    row,
                    start,
                    s_max,
                    t_max,
                    s_norm,
                    t_norm,
                    log_c,
                    extra,
                    per_t,
                    C,
                    C,
                    B,
                    FP32,
                )
            r, pi = _terms(
                omega,
                p,
                log_p,
                log_q,
                ratio,
                alpha,
                log_alpha,
                valid,
                EPS,
                TINY,
                CUT,
                LARGE,
                STEPS,
                RES_STEPS,
            )
            r = tl.where(kept, r, 0.0)
            denom = 1 + alpha * pi
            f += tl.sum(r, 0).to(tl.float64)
            magnitude += tl.sum(tl.abs(r), 0).to(tl.float64)
            d1 += tl.sum(pi / denom, 0).to(tl.float64)
            d2 -= tl.sum(pi / (denom * denom * denom), 0).to(tl.float64)
        following, lo, hi, resolved, done = halley(
            omega,
            f.to(dtype),
            d1.to(dtype),
            d2.to(dtype),
            magnitude.to(dtype),
            lo,
            hi,
            resolved,
            -float("inf"),
            EPS,
            TINY,
        )
        omega = tl.where(done, omega, following)
        step += 1
    # The residuals at omega; the pivot balances the others, written last.
    factor = _factor(scale(UP, R, D, row, row >= 0, ROWWISE, DIVIDED), per_t).to(dtype)
    if SAVE:
        factor = tl.full([], 1.0, dtype)
    rest = tl.zeros([], tl.float64)
    cross = rest
    penalty = rest
    nan = float("nan")
    for start in range(0, C, B):
        if ONE:
            k, valid, kept, p, q, log_p, log_q, ratio = tile0
        else:
            k, valid, kept, p, q, log_p, log_q, ratio = _tile(
                Z,
                Y,
                row,
                start,
                s_max,
                t_max,
                s_norm,
                t_norm,
                log_c,
                extra,
                per_t,
                C,
                C,
                B,
                FP32,
            )
        r, pi = _terms(
            omega,
            p,
            log_p,
            log_q,
            ratio,
            alpha,
            log_alpha,
            valid,
            EPS,
            TINY,
            CUT,
            LARGE,
            STEPS,
            RES_STEPS,
        )
        others = valid & (k != pivot)
        rest += tl.sum(tl.where(others & kept, r, 0.0), 0).to(tl.float64)
        log_pi = _log_pi(r, p, log_p, log_q, omega, alpha, log_alpha, CUT, LARGE, STEPS)
        cross -= tl.sum(tl.where(others & (p > 0), p * log_pi, 0.0), 0).to(tl.float64)
        penalty += tl.sum(tl.where(others, (alpha * r) * r, 0.0), 0).to(tl.float64)
        if GRAD:
            g = tl.where(
                bad,
                nan,
                _unit(tl.where(kept, extra - r, -p) * factor, per_t, SAVE) + 0.0,
            )
            tl.store(G + row * C + k, g.to(G.dtype.element_ty), others)
    k, valid, kept, p, q, log_p, log_q, ratio = _tile(
        Z,
        Y,
        row,
        pivot,
        s_max,
        t_max,
        s_norm,
        t_norm,
        log_c,
        extra,
        per_t,
        pivot + 1,
        C,
        1,
        FP32,
    )
    r = (-rest).to(dtype) + tl.zeros([1], dtype)
    if FP32:
        # Closing the sum protects a dominant class's tiny residual; on ordinary float32 rows
        # it would gather every entry's rounding error into one.
        direct, pi = _terms(
            omega,
            p,
            log_p,
            log_q,
            ratio,
            alpha,
            log_alpha,
            valid,
            EPS,
            TINY,
            CUT,
            LARGE,
            STEPS,
            RES_STEPS,
        )
        r = tl.where(p + q > 1.0, r, direct)
    log_pi = _log_pi(r, p, log_p, log_q, omega, alpha, log_alpha, CUT, LARGE, STEPS)
    cross -= tl.sum(tl.where(p > 0, p * log_pi, 0.0), 0).to(tl.float64)
    value = tl.where(
        missing > 0, float("inf"), cross + (penalty + tl.sum((alpha * r) * r, 0).to(tl.float64)) / 2
    )
    if GRAD:
        # Every thread's tile stores land before the pivot's.
        tl.debug_barrier()
        g = tl.where(bad, nan, _unit(tl.where(kept, extra - r, -p) * factor, per_t, SAVE) + 0.0)
        tl.store(G + row * C + k, g.to(G.dtype.element_ty), valid)
    if VALUES:
        tl.store(V + row, tl.where(bad, nan, value).to(V.dtype.element_ty))
    if SAVE:
        tl.store(G + row * C + C, 0.0)


@triton.jit(do_not_specialize=["UP", "TEMP"])
def _proximal_scaled(
    R,
    D,
    S,
    G,
    CS,
    UP,
    TEMP,
    C: tl.constexpr,
    B: tl.constexpr,
    FP32: tl.constexpr,
    ROWWISE: tl.constexpr,
    DIVIDED: tl.constexpr,
    ALIAS: tl.constexpr = False,
):
    """The gradient of rows a forward saved, B columns per program (axis 1), as _proximal
    writes it; rows it marks are left to the replay. G may be the scores (ALIAS)."""
    row = tl.program_id(0).to(tl.int64)
    if saved_replay(S, row, C, CS, FP32):
        return
    per_t = _per_t(TEMP, FP32)
    factor = _factor(scale(UP, R, D, row, row >= 0, ROWWISE, DIVIDED), per_t).to(per_t.dtype)
    k = tl.program_id(1) * B + tl.arange(0, B)
    valid = k < C
    g = tl.load(saved_rows(S, FP32) + row * CS + k, valid, other=0.0)
    g = _finish(g * factor, per_t) + 0.0
    tl.store(G + row * C + k, g.to(G.dtype.element_ty), valid)


# Streamed float32-work proximal rows without masked or non-finite entries (C > _REGISTER)
# take chunk-parallel passes, one program per (row, chunk), with exact float64 merges:
#  1. _px_partial: the chunk's maxima and normalizer sums (counts at the maxima kept exact);
#  2. _px_prepare (a program per row): the log normalizers and the lowest shift of omega,
#     -gamma max q, which bounds the solve's bracket below;
#  3. _px_moments: per chunk the bracket's terms max/min (q - p), the pivot (the first largest
#     p + q), the head classes (y = gamma q exp(gamma p - shift) > TAIL: their p, log p, log q,
#     the log ratio and the column) and the tail's moments sum y^n and sum p y^n, its p mass,
#     sum p log q and sum p^2;
#  4. _px_solve (a program per row): Halley steps on the head classes and tail series (as
#     _series_solve) and the loss: exact on head classes, from the moments on the tail
#     (log pi = log q + gamma p - omega - gamma pi there);
#  5. _px_gradient: -r times the row factor from the row state, and the chunk's sums of r
#     beside the pivot, which _px_close turns into the pivot's entry where p + q > 1.
# Near rows, whose residuals are small beside the float32 moments' rounding (about 1e-7 of the
# tail's mass, which moves omega by as much), take one exact Newton step after 4 (_px_refine,
# a pass over those rows only).
# A training forward runs 1-4 and saves the row state (S); backward runs 5. Rows the passes
# leave (flagged, a head overflow, a failed solve) take _proximal, gated on the state.
_PX = (2048, 8)  # chunk columns and warps of the passes
_PX_HEAD = 256  # head slots per row, split over its chunks (at least 16 each)
_PX_Q = tl.constexpr(10)  # float64 words of a chunk's statistics
_PX_P = tl.constexpr(26)  # float64 words of a chunk's moments
# Row state: 0 taken, 1 solved, 2 max z, 3 max t, 4 log normalizer of z, 5 of t, 6 their
# difference, 7 shift, 8 omega, 9 pivot, 10 whether the pivot closes the sum, 11 whether
# _px_refine corrects omega, then the head count of each chunk (SW words in all).
_PX_SW = 12
_PX_HEADS = tl.constexpr(_PX_SW)


@triton.jit
def _px_load(Z, Y, row, j, shift, valid, C: tl.constexpr, MASKED: tl.constexpr):
    cs = rows.load(Z, row, j, shift, valid, C, MASKED).to(tl.float32)
    ct = rows.load(Y, row, j, shift, valid, C, MASKED).to(tl.float32)
    return cs, ct


@triton.jit(do_not_specialize=["TEMP"])
def _px_partial(Z, Y, Q, TEMP, C: tl.constexpr, B: tl.constexpr, NCH: tl.constexpr):
    """Pass 1: a chunk's maxima, exact counts at them, float32 sums of the other exp((z -
    max) / T), its smallest entries and whether it holds an entry the passes refuse."""
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    Q = doubles(Q)
    per_t = _per_t(TEMP, True)
    shift, shared = rows.shared_shift(Z, Y, row, C, C)
    j, _, valid, inside = rows.span(chunk, shift, shared, B, C)
    if inside:
        cs, ct = _px_load(Z, Y, row, j, shift, valid, C, False)
    else:
        cs, ct = _px_load(Z, Y, row, j, shift, valid, C, True)
    inf = float("inf")
    ok = (tl.abs(cs) < inf) & (tl.abs(ct) < inf)  # NaN fails too
    refused = tl.max((valid & ~ok).to(tl.int32))
    use = valid & ok
    sm = tl.max(tl.where(use, cs, -inf))
    tm = tl.max(tl.where(use, ct, -inf))
    at_s = use & (cs == sm)
    at_t = use & (ct == tm)
    es = tl.where(use & ~at_s, tl.exp((cs - sm) * per_t), 0.0)
    et = tl.where(use & ~at_t, tl.exp((ct - tm) * per_t), 0.0)
    out = Q + (row * NCH + chunk) * _PX_Q
    tl.store(out, sm.to(tl.float64))
    tl.store(out + 1, tm.to(tl.float64))
    tl.store(out + 2, tl.sum(at_s.to(tl.int32)).to(tl.float64))
    tl.store(out + 3, tl.sum(es).to(tl.float64))
    tl.store(out + 4, tl.sum(at_t.to(tl.int32)).to(tl.float64))
    tl.store(out + 5, tl.sum(et).to(tl.float64))
    tl.store(out + 6, refused.to(tl.float64))
    tl.store(out + 7, tl.min(tl.where(use, cs, inf)).to(tl.float64))
    tl.store(out + 8, tl.min(tl.where(use, ct, inf)).to(tl.float64))


@triton.jit(do_not_specialize=["TEMP", "ALPHA"])
def _px_prepare(
    Q, S, TEMP, ALPHA, NCH: tl.constexpr, CP: tl.constexpr, EPS: tl.constexpr,
    SW: tl.constexpr,
):  # fmt: skip
    """Merge a row's chunks (float64): whether the passes take it, its maxima, log
    normalizers and their difference, and the lowest shift of omega."""
    row = tl.program_id(0).to(tl.int64)
    Q = doubles(Q)
    state = doubles(S) + row * SW
    per_t = _per_t(TEMP, True).to(tl.float64)
    c = tl.arange(0, CP)
    here = c < NCH
    part = Q + (row * NCH + c) * _PX_Q
    inf = float("inf")
    count_s = tl.load(part + 2, here, other=0.0)
    count_t = tl.load(part + 4, here, other=0.0)
    some = here & (count_s > 0)
    sm = tl.where(some, tl.load(part, here, other=0.0), -inf)
    tm = tl.where(some, tl.load(part + 1, here, other=0.0), -inf)
    SM = tl.max(sm, 0)
    TM = tl.max(tm, 0)
    ws = tl.where(some, exp((sm - SM) * per_t), 0.0)
    wt = tl.where(some, exp((tm - TM) * per_t), 0.0)
    tail_s = _sum_tail(count_s, tl.load(part + 3, here, other=0.0), ws, sm, SM)
    tail_t = _sum_tail(count_t, tl.load(part + 5, here, other=0.0), wt, tm, TM)
    refused = tl.max(tl.load(part + 6, here, other=0.0), 0) != 0
    smin = tl.min(tl.load(part + 7, here, other=inf), 0)
    tmin = tl.min(tl.load(part + 8, here, other=inf), 0)
    s_norm = log1p(tail_s)
    t_norm = log1p(tail_t)
    # The centered scores stay finite in float32; both distributions confident beyond EPS
    # (and the confident limit of _proximal) stay on the general path.
    spread = (SM.to(tl.float32) - smin.to(tl.float32)) * per_t.to(tl.float32)
    spread += (TM.to(tl.float32) - tmin.to(tl.float32)) * per_t.to(tl.float32)
    fast = ~refused & (spread < inf) & (tl.maximum(s_norm, t_norm) >= EPS)
    fast &= (s_norm < inf) & (t_norm < inf)
    # omega >= -gamma max(q - p) >= -gamma max q, a little lower for rounding.
    alpha = f64(ALPHA)
    lowest = (-alpha * exp(-s_norm) * (1.0 + 2.0**-12)).to(tl.float32)
    tl.store(state, fast.to(tl.float64))
    tl.store(state + 1, 0.0)
    tl.store(state + 2, SM)
    tl.store(state + 3, TM)
    tl.store(state + 4, s_norm)
    tl.store(state + 5, t_norm)
    tl.store(state + 6, s_norm - t_norm)
    tl.store(state + 7, lowest.to(tl.float64))


@triton.jit
def _px_elements(cs, ct, valid, state, per_t):
    """p, q, log p, log q and the log ratio of a chunk's entries (as _tile), from the state."""
    s_max = tl.load(state + 2).to(tl.float32)
    t_max = tl.load(state + 3).to(tl.float32)
    s_norm = tl.load(state + 4).to(tl.float32)
    t_norm = tl.load(state + 5).to(tl.float32)
    log_c = tl.load(state + 6).to(tl.float32)
    s, t, d = _scaled(cs, ct, valid, s_max, t_max, per_t, ORDINARY=True)
    log_q = s - s_norm
    log_p = t - t_norm
    q = tl.where(valid, exp(log_q), 0.0)
    p = tl.where(valid, exp(log_p), 0.0)
    return p, q, log_p, log_q, d - log_c


@triton.jit
def _px_store4(out, i, j, k, m, a, b, c, d):
    """Store the sums of four columns at out + i, j, k, m (float64) from one reduction."""
    t = tl.sum(tl.join(tl.join(a, b), tl.join(c, d)), 0)
    ab, cd = tl.split(t)
    sa, sb = tl.split(ab)
    sc, sd = tl.split(cd)
    if m != 0:
        tl.store(out + m, sd.to(tl.float64))
    tl.store(out + k, sc.to(tl.float64))
    tl.store(out + j, sb.to(tl.float64))
    tl.store(out + i, sa.to(tl.float64))


@triton.jit
def _px_moments_body(
    Z, Y, P, HB, row, chunk, j, shift, valid, state, per_t, alpha, log_alpha,
    C: tl.constexpr, NCH: tl.constexpr, K: tl.constexpr, MASKED: tl.constexpr,
):  # fmt: skip
    cs, ct = _px_load(Z, Y, row, j, shift, valid, C, MASKED)
    k = j - shift
    p, q, log_p, log_q, ratio = _px_elements(cs, ct, valid, state, per_t)
    inf = float("inf")
    u = tl.where(
        tl.abs(ratio) < 1.0,
        p * bounded_expm1(tl.minimum(tl.maximum(ratio, -1.0), 1.0)),
        q - p,
    )
    pq = tl.where(valid, p + q, -inf)
    best = tl.max(pq)
    # The chunk's own bound of omega: pi e^(gamma pi) = x e^-omega with x = q e^(gamma p),
    # W(y) / y falls and the chunk's pi sum to at most 1, so omega >= log X - gamma max x / X
    # (X the chunk's sum of x): far tighter than -gamma max q where one class dominates.
    log_x = tl.where(valid, (log_p + ratio) + alpha * p, -inf)
    top_x = tl.max(log_x)
    mass = tl.sum(tl.where(valid, exp(log_x - top_x), 0.0))
    local = (top_x + log(mass)) - alpha / mass
    local = local - 1e-5 * (1 + tl.abs(local))
    lowest = tl.load(state + 7).to(tl.float32)
    lowest = tl.where(local > lowest, local, lowest)
    # y = gamma p e^(ratio + gamma p - shift): p times the exponential of a small argument on
    # near rows, rounded relative to p as the gradient's residuals are (the exponential of
    # log x, about log(1 / C), would shift every y by the rounding of the quantized logits).
    y = tl.where(valid, (alpha * p) * exp((ratio + alpha * p) - lowest), 0.0)
    head = valid & (y > _SERIES_TAIL)
    tail = valid & ~head
    yt = tl.where(tail, y, 0.0)
    pt = tl.where(tail, p, 0.0)
    # pi of the tail at omega = 0 (only the tolerance reads sum |p - pi|).
    w = yt * exp(lowest)
    poly = 117649.0 / 5040.0
    poly = poly * w - 10.8
    poly = poly * w + 125.0 / 24.0
    poly = poly * w - 8.0 / 3.0
    poly = poly * w + 1.5
    poly = poly * w - 1.0
    pi0 = (w + (w * w) * poly) / alpha
    out = doubles(P) + (row * NCH + chunk) * _PX_P
    tl.store(out + 1, tl.max(tl.where(valid, u, -inf)).to(tl.float64))
    tl.store(out + 2, tl.min(tl.where(valid, u, inf)).to(tl.float64))
    tl.store(out + 3, best.to(tl.float64))
    tl.store(out + 4, tl.min(tl.where(pq == best, k, C)).to(tl.float64))
    tl.store(out + 25, lowest.to(tl.float64))
    # Sums four at a time (one cross-warp reduction each). The tail's mass and first moments
    # are summed in float64: the solve takes their difference, sum (p - pi), which is far
    # below either on near rows.
    B: tl.constexpr = k.shape[1]
    zero = tl.zeros([B], tl.float64)
    _px_store4(out, 5, 9, 17, 0, tl.reshape(pt.to(tl.float64), [B]),
               tl.reshape(yt.to(tl.float64), [B]), tl.reshape((pt * yt).to(tl.float64), [B]),
               zero)  # fmt: skip
    y2 = yt * yt
    y3 = y2 * yt
    y4 = y2 * y2
    _px_store4(out, 0, 6, 7, 8, tl.reshape(head.to(tl.float32), [B]),
               tl.reshape(tl.where(tail, p * log_q, 0.0), [B]), tl.reshape(pt * pt, [B]),
               tl.reshape(tl.where(tail, tl.abs(p - pi0), 0.0), [B]))  # fmt: skip
    _px_store4(out, 10, 11, 12, 13, tl.reshape(y2, [B]), tl.reshape(y3, [B]),
               tl.reshape(y4, [B]), tl.reshape(y4 * yt, [B]))  # fmt: skip
    _px_store4(out, 14, 15, 16, 18, tl.reshape(y4 * y2, [B]), tl.reshape(y4 * y3, [B]),
               tl.reshape(y4 * y4, [B]), tl.reshape(pt * y2, [B]))  # fmt: skip
    _px_store4(out, 19, 20, 21, 22, tl.reshape(pt * y3, [B]), tl.reshape(pt * y4, [B]),
               tl.reshape(pt * (y4 * yt), [B]), tl.reshape(pt * (y4 * y2), [B]))  # fmt: skip
    _px_store4(out, 23, 24, 23, 24, tl.reshape(pt * (y4 * y3), [B]),
               tl.reshape(pt * (y4 * y4), [B]), tl.reshape(pt * (y4 * y3), [B]),
               tl.reshape(pt * (y4 * y4), [B]))  # fmt: skip
    count = tl.sum(head.to(tl.int32))
    if count > 0:
        slot = tl.cumsum(head.to(tl.int32), 1) - 1
        keep = head & (slot < K)
        at = ((row * NCH + chunk) * K + slot) * 5
        tl.store(HB + at, p, keep)
        tl.store(HB + at + 1, log_p, keep)
        tl.store(HB + at + 2, log_q, keep)
        tl.store(HB + at + 3, ratio, keep)
        tl.store(HB + at + 4, k.to(tl.float32, bitcast=True), keep)


@triton.jit(do_not_specialize=["TEMP", "ALPHA", "LOG_ALPHA"])
def _px_moments(
    Z, Y, S, P, HB, TEMP, ALPHA, LOG_ALPHA, C: tl.constexpr, B: tl.constexpr,
    NCH: tl.constexpr, K: tl.constexpr, SW: tl.constexpr,
):  # fmt: skip
    """Pass 3: a chunk's bracket terms, pivot candidate, head classes and tail moments."""
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    state = doubles(S) + row * SW
    if tl.load(state) == 0:
        return
    per_t = _per_t(TEMP, True)
    alpha = f64(ALPHA).to(tl.float32)
    log_alpha = f64(LOG_ALPHA).to(tl.float32)
    shift, shared = rows.shared_shift(Z, Y, row, C, C)
    j, _, valid, inside = rows.span(chunk, shift, shared, B, C)
    if inside:
        _px_moments_body(Z, Y, P, HB, row, chunk, j, shift, valid, state, per_t, alpha,
                         log_alpha, C, NCH, K, False)  # fmt: skip
    else:
        _px_moments_body(Z, Y, P, HB, row, chunk, j, shift, valid, state, per_t, alpha,
                         log_alpha, C, NCH, K, True)  # fmt: skip


@triton.jit(do_not_specialize=["ALPHA", "LOG_ALPHA"])
def _px_solve(
    P, HB, S, V, ALPHA, LOG_ALPHA, NCH: tl.constexpr, CP: tl.constexpr, HP: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr, VALUES: tl.constexpr, EPS: tl.constexpr,
    TINY: tl.constexpr, CUT: tl.constexpr, LARGE: tl.constexpr, STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr, SW: tl.constexpr,
):  # fmt: skip
    """Pass 4: the Halley steps for omega on the head classes and tail series, and the loss;
    state 1 says the passes solved the row."""
    row = tl.program_id(0).to(tl.int64)
    state = doubles(S) + row * SW
    if tl.load(state) == 0:
        return
    P = doubles(P)
    c = tl.arange(0, CP)
    here = c < NCH
    part = P + (row * NCH + c) * _PX_P
    counts = tl.load(part, here, other=0.0)
    if tl.max(counts, 0) > K:
        return
    inf = float("inf")
    u_max = tl.max(tl.load(part + 1, here, other=-inf), 0).to(tl.float32)
    u_min = tl.min(tl.load(part + 2, here, other=inf), 0).to(tl.float32)
    best = tl.load(part + 3, here, other=-inf)
    top = tl.max(best, 0)
    pivot = tl.min(tl.where(best == top, tl.load(part + 4, here, other=C), C), 0).to(tl.int32)
    total = tl.sum(tl.load(part + 5, here, other=0.0), 0)
    cross_log_q = tl.sum(tl.load(part + 6, here, other=0.0), 0)
    square = tl.sum(tl.load(part + 7, here, other=0.0), 0)
    spread = tl.sum(tl.load(part + 8, here, other=0.0), 0)
    # Each chunk's moments about its own shift, rescaled to the largest (a bound of omega).
    shifts = tl.where(here & (best > -inf), tl.load(part + 25, here, other=-inf), -inf)
    lowest = tl.max(shifts, 0)
    n = tl.arange(0, 8) + 1
    words = part[:, None] + n[None, :]
    rescale = tl.where(
        (shifts > -inf)[:, None], exp((shifts - lowest)[:, None] * n[None, :].to(tl.float64)), 0.0
    )
    moments = tl.sum(tl.load(words + 8, here[:, None], other=0.0) * rescale, 0)
    mixed = tl.sum(tl.load(words + 16, here[:, None], other=0.0) * rescale, 0)
    lowest = lowest.to(tl.float32)
    i = tl.arange(0, HP)
    held = (i // K < NCH) & ((i % K).to(tl.float64) < tl.load(
        P + (row * NCH + i // K) * _PX_P, i // K < NCH, other=0.0))  # fmt: skip
    at = (row * NCH * K + i) * 5
    p = tl.load(HB + at, held, other=0.0)
    log_p = tl.load(HB + at + 1, held, other=0.0)
    log_q = tl.load(HB + at + 2, held, other=0.0)
    ratio = tl.load(HB + at + 3, held, other=0.0)
    column = tl.load(HB + at + 4, held, other=0.0).to(tl.int32, bitcast=True)
    alpha = f64(ALPHA).to(tl.float32)
    log_alpha = f64(LOG_ALPHA).to(tl.float32)
    lo = -alpha * tl.maximum(u_max, 0.0)
    hi = -alpha * tl.minimum(u_min, 0.0)
    lo = tl.maximum(lo, lowest)
    hi = tl.where(lo <= hi, hi, lo)
    omega = lo * 0
    done = (lo == 0) & (hi == 0)
    resolved = done & False
    f = tl.zeros([], tl.float32)
    magnitude = f
    step = 0
    while ~done & (step < 100):
        r, pi = _terms(
            omega, p, log_p, log_q, ratio, alpha, log_alpha, held, EPS, TINY, CUT, LARGE,
            STEPS, RES_STEPS,
        )  # fmt: skip
        tail, tail_d1, tail_d2 = _tail_series(omega, lowest, moments, n, alpha)
        denom = 1 + alpha * pi
        f = (tl.sum(r.to(tl.float64), 0) + (total - tail)).to(tl.float32)
        magnitude = (tl.sum(tl.abs(r).to(tl.float64), 0) + spread).to(tl.float32)
        d1 = (tl.sum((pi / denom).to(tl.float64), 0) + tail_d1).to(tl.float32)
        d2 = (-tl.sum((pi / (denom * denom * denom)).to(tl.float64), 0) - tail_d2).to(tl.float32)
        following, lo, hi, resolved, done = halley(
            omega, f, d1, d2, magnitude, lo, hi, resolved, -float("inf"), EPS, TINY
        )
        omega = tl.where(done, omega, following)
        step += 1
    if failed(f, magnitude, lo, hi, resolved, EPS, TINY) & (step > 0):
        return
    # The loss: exact on the head classes (the pivot closing the sum where p + q > 1), from
    # the moments on the tail.
    r, pi = _terms(
        omega, p, log_p, log_q, ratio, alpha, log_alpha, held, EPS, TINY, CUT, LARGE, STEPS,
        RES_STEPS,
    )  # fmt: skip
    tail, _, _ = _tail_series(omega, lowest, moments, n, alpha)
    at_pivot = held & (column == pivot)
    close = top > 1.0
    rest = tl.sum(tl.where(at_pivot, 0.0, r).to(tl.float64), 0) + (total - tail)
    # Nested: Triton's interpreter cannot AND a scalar comparison into a block.
    r = tl.where(at_pivot, tl.where(close, (-rest).to(tl.float32), r), r)
    log_pi = _log_pi(r, p, log_p, log_q, omega, alpha, log_alpha, CUT, LARGE, STEPS)
    cross = -tl.sum(tl.where(held & (p > 0), p * log_pi, 0.0).to(tl.float64), 0)
    penalty = tl.sum(tl.where(held, (alpha * r) * r, 0.0).to(tl.float64), 0)
    a64 = alpha.to(tl.float64)
    nf = n.to(tl.float64)
    sigma = exp((lowest.to(tl.float64) - omega.to(tl.float64)) * nf)
    p_pi = tl.sum(series(nf) * sigma * mixed, 0) / a64
    pi_pi = tl.sum(series_square(nf) * sigma * moments, 0) / (a64 * a64)
    # log pi = log q + gamma p - omega - gamma pi on the tail.
    cross -= cross_log_q + a64 * square - omega.to(tl.float64) * total - a64 * p_pi
    penalty += a64 * ((square - 2.0 * p_pi) + pi_pi)
    tl.store(state + 8, omega.to(tl.float64))
    tl.store(state + 9, pivot.to(tl.float64))
    tl.store(state + 10, close.to(tl.float64))
    # The series' f is exact to about 1e-7 of the tail's mass (float32 moments): where the
    # residuals' 1-norm is below 2^-10 of that mass, omega takes _px_refine's exact step.
    refine = (step > 0) & (magnitude.to(tl.float64) < 2.0**-10 * total)
    tl.store(state + 11, refine.to(tl.float64))
    tl.store(state + _PX_HEADS + c, counts, here)
    tl.store(state + 1, 1.0)
    if VALUES:
        tl.store(V + row, (cross + penalty / 2).to(V.dtype.element_ty))


@triton.jit(do_not_specialize=["TEMP", "ALPHA", "LOG_ALPHA"])
def _px_refine(
    Z, Y, S, TEMP, ALPHA, LOG_ALPHA, C: tl.constexpr, B: tl.constexpr, NCH: tl.constexpr,
    SW: tl.constexpr, EPS: tl.constexpr, TINY: tl.constexpr, CUT: tl.constexpr,
    LARGE: tl.constexpr, STEPS: tl.constexpr, RES_STEPS: tl.constexpr,
):  # fmt: skip
    """One Newton step for omega of a solved row that _px_solve flagged (state 11): f(omega) =
    sum r and f' = sum pi / (1 + gamma pi) over the row, each class exact (_terms), so f keeps
    the residuals' own accuracy, not the float32 moments' (release numerics: near rows of 4097
    to 32769 classes, omega 1e-8 to 5e-7 off, the others' gradient up to 0.17 of itself; and
    the bracket of -gamma max(q - p), which assumes the computed sum q equals sum p, could
    exclude the root). Other rows return at once."""
    row = tl.program_id(0).to(tl.int64)
    state = doubles(S) + row * SW
    if (tl.load(state + 1) == 0) | (tl.load(state + 11) == 0):
        return
    per_t = _per_t(TEMP, True)
    alpha = f64(ALPHA).to(tl.float32)
    log_alpha = f64(LOG_ALPHA).to(tl.float32)
    omega = tl.load(state + 8).to(tl.float32)
    shift, shared = rows.shared_shift(Z, Y, row, C, C)
    shift = tl.where(shared, shift, 0)
    f = tl.zeros([], tl.float64)
    d1 = tl.zeros([], tl.float64)
    for chunk in range(0, NCH):
        j, k_unused, valid, inside_unused = rows.span(chunk, shift, shared, B, C)
        cs, ct = _px_load(Z, Y, row, j, shift, valid, C, True)
        p, q, log_p, log_q, ratio = _px_elements(cs, ct, valid, state, per_t)
        r, pi = _terms(
            omega, p, log_p, log_q, ratio, alpha, log_alpha, valid, EPS, TINY, CUT, LARGE,
            STEPS, RES_STEPS,
        )  # fmt: skip
        f += tl.sum(tl.sum(r.to(tl.float64), 1), 0)
        d1 += tl.sum(tl.sum((pi / (1 + alpha * pi)).to(tl.float64), 1), 0)
    step = tl.where(d1 > 0, f / d1, 0.0)
    tl.store(state + 8, omega.to(tl.float64) - step)


@triton.jit
def _px_tail_residual(p, ratio, omega, alpha):
    """r of a tail class (y = gamma pi's limit <= TAIL): pi = p e^(ratio + gamma p - omega)
    P(y) with P(y) = W(y) / y, so r = -p expm1(ratio - omega + gamma p + log P(y)), relative
    to r on near rows; zero where omega equals the log ratio (pi = p)."""
    arg = omega - ratio
    y = (alpha * p) * exp(alpha * p - arg)
    poly = 117649.0 / 5040.0
    poly = poly * y - 10.8
    poly = poly * y + 125.0 / 24.0
    poly = poly * y - 8.0 / 3.0
    poly = poly * y + 1.5
    poly = poly * y - 1.0
    level = (alpha * p - arg) + log1p(y * poly)
    small = tl.abs(level) < 1.0
    r = -p * tl.where(small, bounded_expm1(tl.where(small, level, 0.0)), expm1(level))
    return tl.where(arg != 0, r, 0.0)


@triton.jit
def _px_gradient_body(
    Z, Y, LP, G, row, chunk, j, shift, valid, state, per_t, alpha, factor,
    C: tl.constexpr, NCH: tl.constexpr, MASKED: tl.constexpr,
):  # fmt: skip
    cs, ct = _px_load(Z, Y, row, j, shift, valid, C, MASKED)
    p, q, log_p, log_q, ratio = _px_elements(cs, ct, valid, state, per_t)
    omega = tl.load(state + 8).to(tl.float32)
    pivot = tl.load(state + 9).to(tl.int32)
    r = tl.where(valid, _px_tail_residual(p, ratio, omega, alpha), 0.0)
    tl.store(LP + row * NCH + chunk, tl.sum(tl.where(j - shift == pivot, 0.0, r).to(tl.float64)))
    rows.store(G, row, j, shift, valid, (0.0 - r) * factor + 0.0, C, MASKED)


@triton.jit(do_not_specialize=["UP", "TEMP", "ALPHA"])
def _px_gradient(
    Z, Y, R, D, S, LP, G, UP, TEMP, ALPHA, C: tl.constexpr, B: tl.constexpr,
    NCH: tl.constexpr, ROWWISE: tl.constexpr, DIVIDED: tl.constexpr, SW: tl.constexpr,
    ALIAS: tl.constexpr = False,
):  # fmt: skip
    """Pass 5: -r times the row factor (as _proximal writes it) at a chunk of a solved row
    without head classes, and the chunk's float64 sum of r beside the pivot (LP). G may be
    the scores (ALIAS): a chunk reads its entries before it writes them."""
    if ALIAS:
        Z = G
    LP = doubles(LP)
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    state = doubles(S) + row * SW
    if (tl.load(state + 1) == 0) | (tl.load(state + _PX_HEADS + chunk) != 0):
        return
    per_t = _per_t(TEMP, True)
    alpha = f64(ALPHA).to(tl.float32)
    factor = _factor(scale(UP, R, D, row, row >= 0, ROWWISE, DIVIDED), per_t).to(tl.float32)
    shift, shared = rows.shared_shift(Z, Y, row, C, C)
    _, same = rows.shared_shift(Z, G, row, C, C)
    shared &= same
    shift = tl.where(shared, shift, 0)
    j, _, valid, inside = rows.span(chunk, shift, shared, B, C)
    if inside:
        _px_gradient_body(Z, Y, LP, G, row, chunk, j, shift, valid, state, per_t, alpha,
                          factor, C, NCH, False)  # fmt: skip
    else:
        _px_gradient_body(Z, Y, LP, G, row, chunk, j, shift, valid, state, per_t, alpha,
                          factor, C, NCH, True)  # fmt: skip


@triton.jit(do_not_specialize=["UP", "TEMP", "ALPHA", "LOG_ALPHA"])
def _px_head_gradient(
    Z, Y, R, D, S, LP, G, UP, TEMP, ALPHA, LOG_ALPHA, C: tl.constexpr, B: tl.constexpr,
    NCH: tl.constexpr, ROWWISE: tl.constexpr, DIVIDED: tl.constexpr, SW: tl.constexpr,
    EPS: tl.constexpr, TINY: tl.constexpr, CUT: tl.constexpr, LARGE: tl.constexpr,
    STEPS: tl.constexpr, RES_STEPS: tl.constexpr, ALIAS: tl.constexpr = False,
):  # fmt: skip
    """Pass 5 at the chunks of a solved row that hold head classes (a program per row,
    looping over them): their exact residuals (_terms), the tail's as _px_gradient."""
    if ALIAS:
        Z = G
    LP = doubles(LP)
    row = tl.program_id(0).to(tl.int64)
    state = doubles(S) + row * SW
    if tl.load(state + 1) == 0:
        return
    per_t = _per_t(TEMP, True)
    alpha = f64(ALPHA).to(tl.float32)
    log_alpha = f64(LOG_ALPHA).to(tl.float32)
    factor = _factor(scale(UP, R, D, row, row >= 0, ROWWISE, DIVIDED), per_t).to(tl.float32)
    omega = tl.load(state + 8).to(tl.float32)
    pivot = tl.load(state + 9).to(tl.int32)
    shift, shared = rows.shared_shift(Z, Y, row, C, C)
    _, same = rows.shared_shift(Z, G, row, C, C)
    shift = tl.where(shared & same, shift, 0)
    for chunk in range(0, NCH):
        if tl.load(state + _PX_HEADS + chunk) != 0:
            j, k_unused, valid, inside_unused = rows.span(chunk, shift, shared & same, B, C)
            cs, ct = _px_load(Z, Y, row, j, shift, valid, C, True)
            p, q, log_p, log_q, ratio = _px_elements(cs, ct, valid, state, per_t)
            exact, pi_unused = _terms(
                omega, p, log_p, log_q, ratio, alpha, log_alpha, valid, EPS, TINY, CUT, LARGE,
                STEPS, RES_STEPS,
            )  # fmt: skip
            y = (alpha * p) * exp(alpha * p - (omega - ratio))
            r = tl.where(y > _SERIES_TAIL, exact, _px_tail_residual(p, ratio, omega, alpha))
            r = tl.where(valid, r, 0.0)
            tl.store(
                LP + row * NCH + chunk, tl.sum(tl.where(j - shift == pivot, 0.0, r).to(tl.float64))
            )
            rows.store(G, row, j, shift, valid, (0.0 - r) * factor + 0.0, C, True)


@triton.jit(do_not_specialize=["UP", "TEMP"])
def _px_close(
    R, D, S, LP, G, UP, TEMP, C: tl.constexpr, NCH: tl.constexpr, CP: tl.constexpr,
    ROWWISE: tl.constexpr, DIVIDED: tl.constexpr, SW: tl.constexpr,
):  # fmt: skip
    """The pivot's entry of a solved row whose p + q > 1: minus the others' r as written."""
    LP = doubles(LP)
    row = tl.program_id(0).to(tl.int64)
    state = doubles(S) + row * SW
    if (tl.load(state + 1) == 0) | (tl.load(state + 10) == 0):
        return
    c = tl.arange(0, CP)
    r = (-tl.sum(tl.load(LP + row * NCH + c, c < NCH, other=0.0), 0)).to(tl.float32)
    per_t = _per_t(TEMP, True)
    factor = _factor(scale(UP, R, D, row, row >= 0, ROWWISE, DIVIDED), per_t).to(tl.float32)
    pivot = tl.load(state + 9).to(tl.int64)
    tl.store(G + row * C + pivot, ((0.0 - r) * factor + 0.0).to(G.dtype.element_ty))


def _px_plan(call: Call, b: Builder, args: dict, meta: dict, warps: int) -> common.Plan:
    """The chunk passes of streamed float32-work proximal_distillation and the gated general
    solve of the rows they leave."""
    n, c = call.n, call.c
    width, pass_warps = _PX
    nch = row_chunks(c, width)
    head = max(16, pow2(_PX_HEAD // nch))
    saved = call.saved == common.SAVED
    save = call.saved == common.SAVE
    values = ("V",) if call.values else ()
    sw = _PX_SW + 2 * ((nch + 1) // 2)
    args["S"] = common.Ref("S") if saved else b.array("S", (n, sw), "float64", not save)
    shapes: dict[str, Any] = dict(C=c, NCH=nch, CP=pow2(nch), SW=sw)
    numerics = {k: meta[k] for k in ("EPS", "TINY", "CUT", "LARGE", "STEPS", "RES_STEPS")}
    if not saved:
        args["Q"] = b.array("Q", (n, nch, _PX_Q.value), "float64")
        args["P"] = b.array("P", (n, nch, _PX_P.value), "float64")
        args["HB"] = b.array("HB", (n, nch * head, 5), "float32")
        b.launch(_px_partial, (n, nch), args, pass_warps, B=width, **shapes)
        b.launch(_px_prepare, (n,), args, 1, EPS=meta["EPS"], **shapes)
        b.launch(
            _px_moments, (n, nch), args, pass_warps, state=("S", "P", "HB"), B=width, K=head,
            enable_reflect_ftz=False, **shapes,
        )  # fmt: skip
        hp = pow2(nch * head)
        b.launch(
            _px_solve, (n,), args, max(1, min(8, hp // 256)), state=("S",) + values, HP=hp,
            K=head, VALUES=call.values, enable_reflect_ftz=False, **shapes, **numerics,
        )  # fmt: skip
        b.launch(
            _px_refine, (n,), args, pass_warps, state=("S",), B=width,
            enable_reflect_ftz=False, C=c, NCH=nch, SW=sw, **numerics,
        )  # fmt: skip
    if call.grad:
        args["LP"] = b.array("LP", (n, nch), "float64")
        factors: dict[str, Any] = dict(ROWWISE=call.rowwise, DIVIDED=call.divided)
        b.launch(
            _px_gradient, (n, nch), args, pass_warps, B=width, enable_reflect_ftz=False,
            **shapes, **factors,
        )  # fmt: skip
        b.launch(
            _px_head_gradient, (n,), args, pass_warps, state=("LP", "G"), B=width,
            enable_reflect_ftz=False, **shapes, **factors, **numerics,
        )  # fmt: skip
        b.launch(_px_close, (n,), args, 1, state=("G",), **shapes, **factors)
    meta = dict(meta, SW=sw)
    args["H"] = b.array("H", (n, _SERIES_HEAD, 4), "float32")
    state = (() if saved else values) + (("G",) if call.grad else ()) + ("H",)
    rest = dict(meta, VALUES=call.values and not saved, GRAD=call.grad, GATED=True)
    b.launch(
        _proximal, (n,), args, warps, state=state, enable_reflect_ftz=False,
        SERIES=_SERIES_HEAD, **rest,
    )  # fmt: skip
    return b.plan()


def _rescue_reduction(b: Builder, call: Call, args: dict) -> tuple[dict, dict]:
    """The sum or mean of the row values in _narrow_rescue's last program (_last_program_sum)
    instead of a reduction launch, for one float32 result: the constexprs of the row kernel
    and of the rescue. Other reductions (float64 results, vmapped groups) keep reduce.add."""
    args.update(K=common.Ref("Z"), T=common.Ref("Z"), L=common.Ref("Z"))
    # One array holds the count (first word) and the program sums: the native eager replay
    # binds at most 12 buffers.
    one = call.reduction and call.values and not call.reduction_rows
    if not (one and call.value_dtype == "float32" and call.saved != common.SAVED):
        return {}, {}
    gates = common.cdiv(call.n, _GATE)
    args["T"] = b.array("T", (gates + 1,), "float64")
    args["K"] = args["T"]
    args["L"] = b.array("L", (), call.value_dtype, internal=False)
    b.reduced = True
    rescue = dict(REDUCE=True, MEAN=call.reduction == common.MEAN, PARTS=gates)
    return dict(REDUCE=True), rescue


def _chunk_width(c: int, width: int) -> tuple[int, int, int]:
    """(B, chunks, tail) of a streamed row: chunks of B = width columns, or of half that
    where it pads the row's lanes by at least a tenth less. A last chunk of at most B / 8
    columns is a tail of pow2(its columns) lanes, which the program before it takes (no
    program of its own): 8193 columns are 4 chunks of 2048 and a tail of 16 lanes."""

    def layout(b: int) -> tuple[int, int, int, int]:
        chunks = row_chunks(c, b)
        rest = c + rows.SLACK - (chunks - 1) * b
        tail = pow2(rest) if chunks > 1 and pow2(rest) <= b // 8 else 0
        return (chunks - 1) * b + (tail or b), b, chunks, tail

    wide, narrow = layout(width), layout(width // 2)
    return (narrow if 10 * narrow[0] <= 9 * wide[0] else wide)[1:]


def _fp32(call: Call, *parameters: float) -> bool:
    """Float32 work, unless a parameter (or its reciprocal) is outside float32's normal range."""
    return call.fp32 and all(FLOAT32.tiny <= v <= 1 / FLOAT32.tiny for v in parameters)


def plan(call: Call) -> common.Plan:  # noqa: C901 - one launch plan per path
    """V and G of distillation and proximal_distillation."""
    n, c = call.n, call.c
    b = Builder(call)
    args = dict(b.inputs(), **b.results(), C=c, TEMP=bits(call.temperature))
    if call.kind == common.PROXIMAL_DISTILLATION:
        if not (math.isfinite(call.gamma) and call.gamma > 0):
            raise ValueError("gamma must be finite and positive")
        fp32 = _fp32(call, call.temperature, call.gamma)
        eps, tiny, cut, large, steps, res_steps = _FLOAT32 if fp32 else _FLOAT64
        block = max(16, pow2(c))
        # Register rows at most 8 warps: 16 caps a thread at 128 registers, which spills the
        # solve.
        block, warps = _STREAM if c > _REGISTER else (block, max(1, min(8, block // 64)))
        args.update(ALPHA=bits(call.gamma), LOG_ALPHA=bits(math.log(call.gamma)))
        meta = dict(
            b.flags(),
            B=block,
            FP32=fp32,
            EPS=eps,
            TINY=tiny,
            CUT=cut,
            LARGE=large,
            STEPS=steps,
            RES_STEPS=res_steps,
        )
        cs = saved_stride(c)
        args["CS"] = cs
        if fp32 and c > _REGISTER:
            return _px_plan(call, b, args, meta, warps)
        if call.saved == common.SAVED:
            args["S"] = common.Ref("S")
        elif call.saved == common.SAVE:
            args["S"] = b.array("S", (n, cs), "float32" if fp32 else "float64", False)
        else:
            args["S"] = args["Z"]
        # The series solve pays where a Halley step is a pass over a streamed row.
        series_head = _SERIES_HEAD if fp32 and c > _REGISTER else 0
        args["H"] = args["Z"]
        if call.saved != common.SAVED:
            save = call.saved == common.SAVE
            if series_head:
                args["H"] = b.array("H", (n, series_head, 4), "float32")
            solve = dict(meta, GRAD=call.grad or save, SAVE=save, SERIES=series_head)
            b.launch(_proximal, (n,), args, warps, enable_reflect_ftz=False, **solve)
            if not (save and call.grad):
                return b.plan()
        width, copy_warps = _PROXIMAL_COPY
        b.launch(
            _proximal_scaled, (n, common.cdiv(c, width)), args, copy_warps, C=c, B=width,
            FP32=fp32, ROWWISE=call.rowwise, DIVIDED=call.divided,
        )  # fmt: skip
        replay = dict(meta, VALUES=False, REPLAY=True)
        b.launch(_proximal, (n,), args, warps, state=("G",), enable_reflect_ftz=False, **replay)
        return b.plan()
    fp32 = _fp32(call, call.temperature)
    # Keep each register row intact; stream larger rows through fixed-size tiles.
    streamed = c > _REGISTER
    stats = streamed and (bool(call.saved) or call.grad)
    # Streamed float32 work: one read per row, the general path for the rows it leaves.
    online = streamed and fp32
    stride = _STRIDE if online else 5
    if online:
        args["S"] = b.stats(stride) if stats else b.array("S", (n, stride), "float64")
    else:
        args["S"] = b.stats(stride) if stats else args["Z"]
    shared = ("S", "V") if online and call.values else ("S",) if online else ()
    register_bfloat = not streamed and fp32 and call.dtype == call.target == "bfloat16"
    stream_float = streamed and fp32 and call.dtype == "float32"
    if not streamed and fp32 and c < _NARROW:
        # One row per thread, and the general register path for the rows it leaves.
        grid = (common.cdiv(n, _NARROW_ROWS),)
        flags = dict(args, F=b.array("F", (n,), "uint8"))
        fused, folded = _rescue_reduction(b, call, flags)
        narrow = dict(
            C=c, ROWWISE=call.rowwise, DIVIDED=call.divided, VALUES=call.values,
            GRAD=call.grad, RECIP_MASS=call.dtype == "float32", SPARSE=register_bfloat,
            enable_reflect_ftz=False,
        )  # fmt: skip
        # Every output as state: a gradient written over the scores (G = Z) is then bound as
        # the full donated buffer by binders that allocate the declared writes.
        written = tuple(k for k, on in (("V", call.values), ("F", True), ("G", call.grad)) if on)
        written += ("K",) if fused else ()
        b.launch(
            _narrow, grid, flags, _NARROW_WARPS, state=written, ROWS=_NARROW_ROWS, **fused,
            **narrow,
        )  # fmt: skip
        block = max(16, pow2(c))
        # Both launches write rows of V and G (state: JAX threads the partial outputs), and
        # the rescue counts its programs in T (_last_program_sum).
        partial = tuple(k for k, on in (("V", call.values), ("G", call.grad)) if on)
        partial += ("T",) if fused else ()
        b.launch(
            _narrow_rescue, (common.cdiv(n, _GATE),), flags, 2, state=partial, B=block,
            ROWS=max(1, min(16, 256 // block)), GATE=_GATE, **folded, **narrow,
        )  # fmt: skip
        return b.plan()
    medium_types = ("bfloat16", "float16", "float32")
    if not streamed and fp32 and call.dtype in medium_types and call.target in medium_types:
        precise = "float32" in (call.dtype, call.target)
        # Rows of _NARROW to _REGISTER classes: _medium, its saved state and backward, and
        # the general register path for the rows they mark.
        block = max(16, pow2(c))
        lanes = max(1, block // _MEDIUM_ELEMS)
        width = lanes * min(8, c & -c)
        warps, rows = (lanes // 32, 1) if lanes >= 32 else (4, 128 // lanes)
        geometry = dict(C=c, NB=block // width, W=width, ROWS=rows)
        grid = (common.cdiv(n, rows),)
        flags = dict(args, F=b.array("F", (n,), "uint8"))
        fused, folded = _rescue_reduction(b, call, flags)
        if call.saved == common.SAVED:
            flags["S"] = common.Ref("S")
            b.launch(
                _medium_backward, grid, flags, warps, state=("F", "G"), ROWWISE=call.rowwise,
                DIVIDED=call.divided, PRECISE=precise, **geometry,
            )  # fmt: skip
        else:
            save = call.saved == common.SAVE
            flags["S"] = b.stats(_MEDIUM_STATE.value) if save else args["Z"]
            written = (("V", call.values), ("S", save), ("F", True), ("G", call.grad))
            written += (("K", bool(fused)),)
            b.launch(
                _medium, grid, flags, warps, state=tuple(k for k, on in written if on),
                ROWWISE=call.rowwise, DIVIDED=call.divided,
                VALUES=call.values, GRAD=call.grad, SAVE=save, SPARSE=register_bfloat,
                PRECISE=precise, enable_reflect_ftz=False, **fused, **geometry,
            )  # fmt: skip
        values = call.values and call.saved != common.SAVED
        partial = tuple(k for k, on in (("V", values), ("G", call.grad)) if on)
        partial += ("T",) if fused else ()
        if partial:
            b.launch(
                _narrow_rescue, (common.cdiv(n, _GATE),), flags,
                max(1, min(8, block // 512)), state=partial, C=c, B=block,
                ROWS=max(1, _BFLOAT_LANES // block), GATE=_GATE, ROWWISE=call.rowwise,
                DIVIDED=call.divided, VALUES=values, GRAD=call.grad,
                RECIP_MASS=call.dtype == "float32", SPARSE=register_bfloat,
                enable_reflect_ftz=False, **folded,
            )  # fmt: skip
        return b.plan()
    if streamed:
        rows = 1
        block, warps = (256, 4) if not fp32 else _CHUNK
    else:
        block = max(16, pow2(c))
        # BF16 acceptance and the sparse rescue are row-local.
        # BF16 acceptance is per row (masks), so register rows group like the others.
        if register_bfloat:
            # Row-local acceptance (per-row masks): short rows share a one-warp program
            # (64 lanes, two per thread), wider rows use few warps so that their many row
            # reductions stay mostly within a warp.
            rows = max(1, _BFLOAT_LANES // block)
            warps = max(1, min(8, block // 512))
        else:
            rows = max(1, min(16, 256 // block))
            warps = 2 if rows > 1 else max(1, min(16, block // 128))
    fixed_t = call.temperature if not fp32 and math.frexp(call.temperature)[0] == 0.5 else 0.0
    reciprocal = 1.0 / fixed_t if streamed and 2.0**-1000 <= fixed_t <= 2.0**1000 else 0.0
    meta = dict(b.flags(), FP32=fp32, ROWS=rows, FIXED_T=fixed_t, RECIPROCAL=reciprocal)
    meta["GRAD"] = call.grad or stats
    if not (stats and call.saved == common.SAVED):
        first = dict(args, G=args["Z"]) if stats else args
        if online:
            width, online_warps = _ONLINE
            width, chunks, tail = _chunk_width(c, width)
            partial = b.array("Q", (n, chunks, 12), "float64")
            b.launch(
                _online_partial, (n, chunks - (1 if tail else 0)), dict(args, P=partial),
                online_warps, C=c, B=width, CHUNKS=chunks, TAIL=tail,
            )  # fmt: skip
            b.launch(
                _online, (n,), dict(args, P=partial), 1, CHUNKS=chunks, CP=pow2(chunks),
                VALUES=call.values,
            )  # fmt: skip
        forward = dict(
            ONLINE=online,
            STRIDE=stride,
            B=block,
            SAVE_STATS=stats,
            ORDINARY_ROWS=stream_float,
            REGISTER_BFLOAT=register_bfloat,
            RECIP_MASS=not streamed and fp32 and call.dtype == "float32",
            STREAM_FP32=stream_float,
            enable_reflect_ftz=False,
            **meta,
        )
        grid = (common.cdiv(n, rows),)
        if register_bfloat and block >= _BFLOAT_SPLIT:
            # Wide rows: a lean fast pass flags the rows it leaves, and the rescue pass reads
            # n bytes of flags and runs the general path on the flagged rows only (on narrow
            # rows the second pass over every program costs more than it saves).
            flags = dict(first, F=b.array("F", (n,), "uint8"))
            b.launch(_forward, grid, flags, warps, state=shared, SPLIT=1, **forward)
            # Both passes write rows of V and G (state: JAX threads the partial outputs).
            partial = tuple(k for k, on in (("V", call.values), ("G", call.grad)) if on)
            b.launch(_forward, grid, flags, warps, state=shared + partial, SPLIT=2, **forward)
        else:
            b.launch(_forward, grid, dict(first, F=common.Ref("Z")), warps, state=shared, **forward)
        if stats:
            b.launch(
                _rescue_partition,
                (n,),
                first,
                warps,
                state=("S",),
                C=c,
                B=block,
                FP32=fp32,
                ONLINE=online,
                STRIDE=stride,
                enable_reflect_ftz=False,
            )
    if call.grad and stats:
        width, num_warps = _BACKWARD if fp32 else (1024, 4)
        if online:
            chunk, chunks, tail = _chunk_width(c, width)
            b.launch(
                _online_backward,
                (n, chunks - (1 if tail else 0)),
                args,
                num_warps,
                C=c,
                B=chunk,
                ROWWISE=call.rowwise,
                DIVIDED=call.divided,
                CHUNKS=chunks,
                TAIL=tail,
            )
        b.launch(
            _backward,
            (n, 1 if online else common.cdiv(c, width)),
            args,
            num_warps,
            state=("G",) if online else (),
            B=width,
            FULL_ROW=False,
            LONG_RATIO=stream_float,
            ONLINE=online,
            STRIDE=stride,
            enable_reflect_ftz=False,
            **meta,
        )
    return b.plan()
