"""proximal_cross_entropy, as include/sdca/loss/proximal.h, and the solves the proximal losses
share: W(exp(x)), the residual of a label and the safeguarded Halley step.

Ordinary float32-work label rows take the series path (see _TAIL): chunk-parallel passes
gather the row's statistics, then the tail moments and few head classes, so that a program per
row takes the Halley steps for the shift dt of the proximal point on them; a last chunk pass
writes the gradient from a small row state, which a training forward saves for its backward.

Other rows: one program per row solves by Halley steps, a pass over the row per step (rows up
to 4096 classes load once and stay in registers). It writes the row's loss and its gradient
-a, whose zero sum the pivot (the label, else the largest |a|) closes; a training forward
saves this unit gradient and backward scales it. Float32 work keeps float64 sums over tiles; a
row whose spread float32 cannot hold is solved in float64.

A score of -inf at a class that is not a label masks it (probability and gradient zero).
For a masked label the loss is infinite and its gradient is -1 at the label and the
proximal probabilities on the kept classes. NaN, +inf, an invalid label, an all-masked row
and a solve that does not converge give NaN rows.
"""

from __future__ import annotations

import math
from typing import Any

import triton
import triton.language as tl

from .._limits import FLOAT32, FLOAT64
from . import common
from .common import Builder, Call, pow2
from .math import (
    DBL_EPS,
    DBL_MIN,
    below,
    bits,
    doubles,
    exp,
    f64,
    finite,
    log,
    log1p,
    scale,
    target_row,
)

_REGISTER = 4096
_DOUBLE_TAIL = tl.constexpr(-FLOAT64.log_denorm_min)
_STREAM = (2048, 16)
_COPY = (2048, 4)  # the scaled copy of a saved gradient: columns and warps per program
# Work-type constants: (eps, tiny, W cut, W large, Newton steps of W, residual steps).
_FLOAT32 = (FLOAT32.eps, FLOAT32.tiny, -18.0, 1e7, 2, 4)
_FLOAT64 = (FLOAT64.eps, FLOAT64.tiny, -36.0, 1e16, 3, 6)


def saved_stride(c: int) -> int:
    """Columns of a saved row: the unit gradient, its replay flag, padding to 16 bytes."""
    return (c + 1 + 3) // 4 * 4


@triton.jit
def saved_rows(S, FP32: tl.constexpr):
    """The saved unit gradients in the work type (float64 also as 32-bit word pairs)."""
    if FP32:
        P = S
    else:
        P = doubles(S)
    return P


@triton.jit
def saved_replay(S, row, C, CS, FP32: tl.constexpr):
    """Whether a forward left this row's gradient to a replay of its solve."""
    return tl.load(saved_rows(S, FP32) + row * CS + C) != 0


@triton.jit
def scaled_chunk(S, G, row, block, factor, zero, C, CS, B: tl.constexpr, FP32: tl.constexpr):
    """Chunk `block` of a row's gradient from its saved unit gradient: g * factor rounded once
    into G's dtype (+0.0 at zeros), as the solve writes it; zeros where `zero`."""
    k = block * B + tl.arange(0, B)
    valid = k < C
    P = saved_rows(S, FP32)
    g = tl.load(P + row * CS + k, valid, other=0.0)
    out = tl.where(zero, 0.0, g * factor.to(g.dtype) + 0.0)
    tl.store(G + row * C + k, out.to(G.dtype.element_ty), valid)


@triton.jit
def lambert_w_exp(x, CUT: tl.constexpr, LARGE: tl.constexpr, STEPS: tl.constexpr):
    """W(exp(x)): the w of w + log(w) = x."""
    u = log(tl.where(x > 30, x, log1p(exp(x))))
    for _ in tl.static_range(STEPS):
        e = exp(u)
        g = (e - x) + u
        u = u - g / ((e + 1) - g * e / (2 * (e + 1)))
    w = exp(u)
    if x.dtype == tl.float64:
        w = w - ((w - x) + log(w)) * (w / (w + 1))
    w = tl.where(x > LARGE, x - log(x), w)
    return tl.where(x < CUT, exp(x), tl.where(x == float("inf"), x, w))


@triton.jit
def probability(h, alpha, log_alpha, CUT: tl.constexpr, LARGE: tl.constexpr, STEPS: tl.constexpr):
    """W(alpha exp(h)) / alpha, exp(h - W) where the argument is far below."""
    arg = h + log_alpha
    w = lambert_w_exp(arg, CUT, LARGE, STEPS)
    if h.dtype == tl.float32:
        cut = -9.0
    else:
        cut = -18.0
    return tl.where(arg < cut, exp(h - w), w / alpha)


@triton.jit
def near_residual(s, alpha, q, log_q, EPS: tl.constexpr, RES_STEPS: tl.constexpr):
    """r in [q - 1, q) with alpha r - log1p(-r / q) = s, for s in the near range
    [-alpha q / 2 - log 1.5, alpha q / 2 + log 2] (or zero): safeguarded Halley steps."""
    lo = q - 1
    inv_q = 1 / q
    linear = s / (alpha + inv_q)
    r = linear / (1 + (linear * inv_q) / (2 * (1 + alpha * q)))
    r = tl.minimum(
        tl.maximum(r, lo, propagate_nan=tl.PropagateNan.ALL),
        below(q),
        propagate_nan=tl.PropagateNan.ALL,
    )
    low = lo + r * 0
    high = q + r * 0
    tol = 8 * EPS * tl.abs(s)
    for _ in tl.static_range(RES_STEPS):
        f = (alpha * r - log1p(-r * inv_q)) - s
        inv_qr = 1 / (q - r)
        d1 = alpha + inv_qr
        denom = d1 - (f * inv_qr * inv_qr) / (2 * d1)
        step = tl.where(denom > 0, r - f / denom, r - f / d1)
        # A step below the resolution of r ends it: a subnormal r misses tol.
        active = (tl.abs(f) > tol) & (step != r)
        low = tl.where(active & (f < 0), r, low)
        high = tl.where(active & (f >= 0), r, high)
        step = tl.where((low < step) & (step < high), step, (low + high) / 2)
        r = tl.where(active, step, r)
    return tl.where(s == 0, 0.0, tl.where(s <= alpha * lo + log_q, lo, r))


@triton.jit
def residual(
    t,
    alpha,
    q,
    log_alpha,
    log_q,
    EPS: tl.constexpr,
    CUT: tl.constexpr,
    LARGE: tl.constexpr,
    STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
):
    """r in [q - 1, q) with alpha r - log1p(-r / q) = t: the near solve, or q minus the
    proximal probability far from zero."""
    lo = q - 1
    half = alpha * q / 2
    far = (t > half + 0.6931471805599453) | (t < -half - 0.4054651081081644)
    prob = probability(log_q + (alpha * q - t), alpha, log_alpha, CUT, LARGE, STEPS)
    outer = tl.minimum(
        tl.maximum(q - prob, lo, propagate_nan=tl.PropagateNan.ALL),
        q,
        propagate_nan=tl.PropagateNan.ALL,
    )
    r = near_residual(tl.where(far, 0.0, t), alpha, q, log_q, EPS, RES_STEPS)
    r = tl.where(far, outer, r)
    return tl.where(t == 0, 0.0, tl.where(t <= alpha * lo + log_q, lo, r))


@triton.jit
def halley(x, f, d1, d2, magnitude, lo, hi, resolved, jump, EPS: tl.constexpr, TINY: tl.constexpr):
    """One safeguarded Halley step on f(x) = 0 in [lo, hi] (jump: a larger step to take, or 0):
    the next x, the bracket, whether x is the closest root, and done."""
    done = tl.abs(f) <= 8 * EPS * tl.maximum(magnitude, TINY)
    lo = tl.where(~done & (f < 0), x, lo)
    hi = tl.where(~done & (f >= 0), x, hi)
    newton = -f / d1
    correction = (f * d2) / (2 * d1 * d1)
    clamped = tl.minimum(tl.maximum(correction, -0.5), 0.5)
    delta = tl.where(finite(correction), newton / (1 - clamped), newton)
    delta = tl.where(jump > delta, jump, delta)
    delta = tl.where((d1 > 0) & finite(d1) & finite(newton), delta, 0.0)
    following = x + delta
    # A nonzero step below the resolution of x: x is the closest root.
    resolved = resolved | (~done & (delta != 0) & (following == x))
    done = done | resolved
    inside = (lo <= following) & (following <= hi) & (following != x) & finite(following)
    following = tl.where(inside, following, lo + (hi - lo) / 2)
    done = done | (following == x)
    return following, lo, hi, resolved, done


@triton.jit
def failed(f, magnitude, lo, hi, resolved, EPS: tl.constexpr, TINY: tl.constexpr):
    """Whether a solve stopped short of a root (the resolution of x and f is eps relative to
    at least tiny: they can underflow)."""
    width = 64 * EPS * tl.maximum(tl.maximum(tl.abs(lo), tl.abs(hi)), TINY)
    far = tl.abs(f) > 256 * EPS * tl.maximum(magnitude, TINY)
    return ~finite(f) | (~resolved & far & (hi - lo > width))


@triton.jit
def _tile(Z, Y, base, row, start, C, B: tl.constexpr, dtype, masked):
    """Columns, validity, the label, other classes and scores of a tile."""
    k = start + tl.arange(0, B)
    valid = k < C
    z = tl.load(Z + base + k, valid, other=0.0).to(dtype)
    # A masked label takes the proximal step toward a uniform target on the kept classes.
    positive = valid & tl.where(masked, z > -float("inf"), k == tl.load(Y + row))
    return k, valid, positive, valid & ~positive, z


@triton.jit
def _centered(z, zmax, valid):
    """Scores less the largest, zero off the row; a masked class keeps its -inf."""
    v = z - zmax
    return tl.where(valid & (v == v), v, 0.0)


@triton.jit
def _evaluate(
    dt,
    v,
    ref,
    r1,
    count,
    positive,
    others,
    alpha,
    log_alpha,
    q,
    log_q,
    clip,
    EPS: tl.constexpr,
    CUT: tl.constexpr,
    LARGE: tl.constexpr,
    STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
):
    """Per tile: f, f', f'', magnitude and other mass, and the lanes' label residuals and other
    probabilities; r1 is the residual of a one-label row's label."""
    argument = (ref - v) + dt
    if count == 1:
        r_label = tl.where(positive, r1, 0.0)
    else:
        r_label = residual(argument, alpha, q, log_alpha, log_q, EPS, CUT, LARGE, STEPS, RES_STEPS)
        r_label = tl.where(positive, r_label, 0.0)
    prob = probability(((v - ref) - dt) + log_q, alpha, log_alpha, CUT, LARGE, STEPS)
    counted = prob <= 1
    prob = tl.where(others, tl.minimum(prob, 1.0, propagate_nan=tl.PropagateNan.ALL), 0.0)
    slope = q - r_label
    label_denom = 1 + alpha * slope
    denom = 1 + alpha * prob
    label_d1 = tl.where(positive & (argument >= clip), slope / label_denom, 0.0)
    d1 = tl.where(counted & others, prob / denom, 0.0)
    f = tl.sum(r_label, 0) - tl.sum(prob, 0)
    derivative = tl.sum(label_d1, 0) + tl.sum(d1, 0)
    curvature = -tl.sum(label_d1 / (label_denom * label_denom), 0) - tl.sum(d1 / (denom * denom), 0)
    magnitude = tl.sum(tl.abs(r_label), 0) + tl.sum(prob, 0)
    return f, derivative, curvature, magnitude, tl.sum(prob, 0), r_label, prob


@triton.jit
def _label_log_probability(
    h, r, q, log_q, log_alpha, CUT: tl.constexpr, LARGE: tl.constexpr, STEPS: tl.constexpr
):
    """log p of a label: log1p where r is small, else log(W / alpha) = h - W."""
    far = h - lambert_w_exp(h + log_alpha, CUT, LARGE, STEPS)
    return tl.where(r <= q / 2, log_q + log1p(-r / q), far)


@triton.jit
def _measure(
    Z,
    Y,
    base,
    row,
    C,
    zmax,
    dt,
    ref,
    label_max,
    count,
    alpha,
    log_alpha,
    q,
    log_q,
    clip,
    positive1,
    others1,
    v1,
    masked,
    B: tl.constexpr,
    SINGLE: tl.constexpr,
    EPS: tl.constexpr,
    CUT: tl.constexpr,
    LARGE: tl.constexpr,
    STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
):
    """f, f', f'', magnitude and other mass of a row at dt: float64 sums over its tiles (a
    float32 sum over many tiles misplaces the root)."""
    dtype = dt.dtype
    r1 = dt * 0
    if count == 1:
        r1 = residual(
            (ref - label_max) + dt, alpha, q, log_alpha, log_q, EPS, CUT, LARGE, STEPS, RES_STEPS
        )
    f = tl.zeros([], tl.float64)
    derivative = f
    curvature = f
    magnitude = f
    mass = f
    for start in range(0, C, B):
        if SINGLE:
            positive, others, v = positive1, others1, v1
        else:
            _, valid, positive, others, z = _tile(Z, Y, base, row, start, C, B, dtype, masked)
            v = _centered(z, zmax, valid)
        tf, td, tc, tm, tmass, _, _ = _evaluate(
            dt,
            v,
            ref,
            r1,
            count,
            positive,
            others,
            alpha,
            log_alpha,
            q,
            log_q,
            clip,
            EPS,
            CUT,
            LARGE,
            STEPS,
            RES_STEPS,
        )
        f += tf.to(tl.float64)
        derivative += td.to(tl.float64)
        curvature += tc.to(tl.float64)
        magnitude += tm.to(tl.float64)
        mass += tmass.to(tl.float64)
    return (
        f.to(dtype),
        derivative.to(dtype),
        curvature.to(dtype),
        magnitude.to(dtype),
        mass.to(dtype),
    )


@triton.jit
def _overflowing_gap(
    Z, G, base, y, label_z, other_z, alpha, factor, C, B: tl.constexpr, GRAD: tl.constexpr
):
    """The entropy envelope for an isolated group of equal winners beyond the binary64 range."""
    half_gap = 0.5 * other_z - 0.5 * label_z
    wins = tl.zeros([], tl.int32)
    runner = tl.full([], -float("inf"), tl.float64)
    for start in range(0, C, B):
        k = start + tl.arange(0, B)
        z = tl.load(Z + base + k, (k < C) & (k != y), other=-float("inf")).to(tl.float64)
        wins += tl.sum((z == other_z).to(tl.int32), 0)
        runner = tl.maximum(runner, tl.max(tl.where(z < other_z, z, -float("inf")), 0))
    share = 1.0 / wins.to(tl.float64)
    # For total competitor mass p split equally at the winners, the quadratic dual is
    # p * gap - alpha * p**2 * (1 + share) / 2. Its omitted entropy is at most log(C).
    mass = tl.minimum(tl.maximum((half_gap / alpha) * (2.0 / (1.0 + share)), 0.0), 1.0)
    half_step = (0.5 * alpha) * (mass * share)
    separation = (0.5 * other_z - 0.5 * runner) - half_step
    # The excluded mass, even after scaling a gradient, rounds to zero here.
    slack = _DOUBLE_TAIL + log(tl.cast(C, tl.float64)) + 8.0 * DBL_EPS * half_step
    closed = (half_gap < 0) | (separation > slack)
    value = tl.where(
        mass == 1.0, (half_gap - (0.25 * alpha) * (1.0 + share)) * 2.0, mass * half_gap
    )
    if GRAD:
        for start in range(0, C, B):
            k = start + tl.arange(0, B)
            z = tl.load(Z + base + k, k < C, other=-float("inf")).to(tl.float64)
            g = tl.where(k == y, -mass, tl.where(z == other_z, mass * share, 0.0))
            tl.store(G + base + k, _out(g, factor, ~closed, G), k < C)
    return tl.maximum(value, 0.0), ~closed


@triton.jit
def _solve(
    Z,
    Y,
    G,
    row,
    C,
    alpha,
    log_alpha,
    factor,
    masked,
    B: tl.constexpr,
    SINGLE: tl.constexpr,
    GRAD: tl.constexpr,
    EPS: tl.constexpr,
    TINY: tl.constexpr,
    CUT: tl.constexpr,
    LARGE: tl.constexpr,
    STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
):
    """Solve a row in the dtype of alpha: its loss, whether it is invalid and whether it took
    a special path (whose gradient folds in factor or is float64); writes its gradient times
    factor."""
    dtype = alpha.dtype
    base = row * C
    inf = float("inf")
    # The largest score, the largest label and other score, the first largest column, the
    # first label, the label count and whether the row is outside the domain.
    zmax = tl.full([], -inf, dtype)
    label_z = zmax
    other_z = zmax
    first = C * 0
    first_label = C
    count = C * 0
    refused = C * 0
    for start in range(0, C, B):
        k, valid, positive, others, z = _tile(Z, Y, base, row, start, C, B, dtype, masked)
        count += tl.sum(positive.to(tl.int32), 0)
        refused += tl.sum((valid & ~(tl.where(positive, tl.abs(z), z) < inf)).to(tl.int32), 0)
        scores = tl.where(valid, z, -inf)
        largest = tl.max(scores, 0)
        first = tl.where(largest > zmax, start + tl.argmax(scores, 0), first)
        zmax = tl.maximum(zmax, largest)
        label_z = tl.maximum(label_z, tl.max(tl.where(positive, z, -inf), 0))
        other_z = tl.maximum(other_z, tl.max(tl.where(others, z, -inf), 0))
        first_label = tl.minimum(first_label, tl.min(tl.where(positive, k, C), 0))
    # A finite score whose gap to the maximum overflows has zero exponential mass.
    # Keep its original label membership when forming the residual and objective.
    bad = (count < 1) | (count >= C) | (refused > 0)
    if dtype == tl.float64:
        if ~bad & ~masked & finite(other_z) & ~finite(other_z - label_z):
            value, invalid = _overflowing_gap(
                Z, G, base, first_label, label_z, other_z, alpha, factor, C, B, GRAD
            )
            return value, invalid, True
    # In a confident one-label row, the proximal probability correction is bounded
    # by O(alpha * tail**2). Accumulate the competitors before their probabilities
    # underflow when that correction is below the work type's rounding.
    gap = other_z.to(tl.float64) - label_z.to(tl.float64)
    bound = gap + log(tl.cast(C, tl.float64))
    log_eps = log(tl.full([], EPS, tl.float64))
    invisible = (log_alpha + bound < log_eps - 2.0) | (
        log_alpha + 2.0 * bound < log_eps + log(tl.full([], TINY, tl.float64)) - 2.0
    )
    if (count == 1) & ~masked & ~bad & finite(other_z) & (bound < 2.0 * log_eps) & invisible:
        total = tl.zeros([B], tl.float64)
        for start in range(0, C, B):
            k = start + tl.arange(0, B)
            other = (k < C) & (k != first_label)
            z = tl.load(Z + base + k, other, other=-float("inf")).to(tl.float64)
            total += exp(z - other_z.to(tl.float64))
        off = tl.sum(total, 0)
        half = exp(0.5 * gap)
        tau = half * (half * off)
        if GRAD:
            weight = factor.to(tl.float64) / (1.0 + tau)
            gy = -(half * off) * (half * weight)
            for start in range(0, C, B):
                k = start + tl.arange(0, B)
                z = tl.load(Z + base + k, k < C, other=-float("inf")).to(tl.float64)
                e = exp(0.5 * (z - label_z.to(tl.float64)))
                g = tl.where(k == first_label, gy, e * (weight * e))
                tl.store(G + base + k, (g + 0.0).to(G.dtype.element_ty), k < C)
        return log1p(tau), bad, True
    q = 1 / count.to(dtype)
    log_q = log(q)
    clip = alpha * (q - 1) + log_q
    label_max = label_z - zmax
    other_max = other_z - zmax
    ref = tl.maximum(label_max, other_max)
    k1, valid1, positive1, others1, z1 = _tile(Z, Y, base, row, 0, C, B, dtype, masked)
    v1 = _centered(z1, zmax, valid1)
    tail = tl.zeros([B], dtype)
    for start in range(0, C, B):
        if SINGLE:
            k, valid, v = k1, valid1, v1
        else:
            k, valid, _, _, z = _tile(Z, Y, base, row, start, C, B, dtype, masked)
            v = _centered(z, zmax, valid)
        tail += tl.where(valid & (k != first), exp(v - ref), 0.0)
    # The bracket of dt and the softmax-limit start inside it.
    lo = tl.maximum(label_max - alpha * (1 - q), other_max - alpha) + log_q
    hi = tl.maximum(label_max + alpha * q, other_max) + log_q + log(tl.cast(C, dtype))
    hi = tl.where(lo <= hi, hi, lo)
    dt = log_q + log1p(tl.sum(tail, 0))
    dt = tl.minimum(
        tl.maximum(dt, lo, propagate_nan=tl.PropagateNan.ALL), hi, propagate_nan=tl.PropagateNan.ALL
    )
    others_count = (C - count).to(dtype)
    f, derivative, curvature, magnitude, mass = _measure(
        Z,
        Y,
        base,
        row,
        C,
        zmax,
        dt,
        ref,
        label_max,
        count,
        alpha,
        log_alpha,
        q,
        log_q,
        clip,
        positive1,
        others1,
        v1,
        masked,
        B,
        SINGLE,
        EPS,
        CUT,
        LARGE,
        STEPS,
        RES_STEPS,
    )
    done = bad
    resolved = count < 0
    step = 0
    while ~done & (step < 200):
        # Halley, or a step in the log of the other mass where it exceeds one.
        jump = log(mass) + alpha * (mass - 1) / others_count
        jump = tl.where((f < 0) & (mass > 1) & finite(jump), jump, -inf)
        following, lo, hi, resolved, done = halley(
            dt, f, derivative, curvature, magnitude, lo, hi, resolved, jump, EPS, TINY
        )
        if ~done:
            dt = following
            f, derivative, curvature, magnitude, mass = _measure(
                Z,
                Y,
                base,
                row,
                C,
                zmax,
                dt,
                ref,
                label_max,
                count,
                alpha,
                log_alpha,
                q,
                log_q,
                clip,
                positive1,
                others1,
                v1,
                masked,
                B,
                SINGLE,
                EPS,
                CUT,
                LARGE,
                STEPS,
                RES_STEPS,
            )
        step += 1
    invalid = bad | failed(f, magnitude, lo, hi, resolved, EPS, TINY)
    loss = _finish(
        Z,
        Y,
        G,
        base,
        row,
        C,
        zmax,
        dt,
        ref,
        label_max,
        count,
        first_label,
        invalid,
        alpha,
        log_alpha,
        q,
        log_q,
        clip,
        factor,
        masked,
        B,
        GRAD,
        EPS,
        CUT,
        LARGE,
        STEPS,
        RES_STEPS,
    )
    return loss.to(tl.float64), invalid, False


@triton.jit
def _finish(
    Z,
    Y,
    G,
    base,
    row,
    C,
    zmax,
    dt,
    ref,
    label_max,
    count,
    first_label,
    invalid,
    alpha,
    log_alpha,
    q,
    log_q,
    clip,
    factor,
    masked,
    B: tl.constexpr,
    GRAD: tl.constexpr,
    EPS: tl.constexpr,
    CUT: tl.constexpr,
    LARGE: tl.constexpr,
    STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
):
    """Write -a factor at dt and return the loss. The pivot (the label, else the first largest
    |a|) is left out of the float64 sums and written last, balancing the others."""
    dtype = dt.dtype
    factor = factor.to(dtype)
    r1 = dt * 0
    if count == 1:
        r1 = residual(
            (ref - label_max) + dt, alpha, q, log_alpha, log_q, EPS, CUT, LARGE, STEPS, RES_STEPS
        )
    labels = tl.full([], count, dtype)
    rest = tl.zeros([], tl.float64)
    penalty = rest
    mass = rest
    log_p = rest
    log_pq = rest
    pivot_a = dt * 0
    pivot = count * 0 - 1
    nonzero = count * 0
    for start in range(0, C, B):
        k, valid, positive, others, z = _tile(Z, Y, base, row, start, C, B, dtype, masked)
        v = _centered(z, zmax, valid)
        _, _, _, _, _, r_label, prob = _evaluate(
            dt,
            v,
            ref,
            r1,
            count,
            positive,
            others,
            alpha,
            log_alpha,
            q,
            log_q,
            clip,
            EPS,
            CUT,
            LARGE,
            STEPS,
            RES_STEPS,
        )
        a = tl.where(positive, r_label, -prob)
        mass += tl.sum(tl.where(positive, 0.0, a), 0).to(tl.float64)
        nonzero += tl.sum((positive & (a != 0)).to(tl.int32), 0)
        if count != 1:
            h = log_q + (alpha * q - ((ref - v) + dt))
            lp = _label_log_probability(h, r_label, q, log_q, log_alpha, CUT, LARGE, STEPS)
            lp = tl.where(positive, lp, 0.0)
            log_p += tl.sum(lp, 0).to(tl.float64)
            log_pq += tl.sum(lp * q, 0).to(tl.float64)
        eligible = valid & (~masked | positive)
        rank = tl.where(eligible, tl.abs(a), -float("inf"))
        candidate = tl.where(count == 1, first_label, start + tl.argmax(rank, 0))
        at = k == candidate
        candidate_a = tl.sum(tl.where(at, a, 0.0), 0)
        better = tl.where(
            count == 1,
            (start <= candidate) & (candidate < start + B),
            (pivot < 0) | (tl.abs(candidate_a) > tl.abs(pivot_a)),
        )
        better &= tl.sum((at & eligible).to(tl.int32), 0) != 0
        dropped = tl.where(better, pivot_a, candidate_a)
        rest += (dropped + tl.sum(tl.where(at, 0.0, a), 0)).to(tl.float64)
        penalty += ((alpha * dropped) * dropped + tl.sum(tl.where(at, 0.0, (alpha * a) * a), 0)).to(
            tl.float64
        )
        pivot_a = tl.where(better, candidate_a, pivot_a)
        pivot = tl.where(better, candidate, pivot)
        if GRAD:
            g = tl.where(masked, tl.where(positive, q - a, -(k == tl.load(Y + row)).to(dtype)), -a)
            tl.store(G + base + k, _out(g, factor, invalid, G), mask=valid)
    # Labels that are all zero share the recovered mass of the others; the first balances.
    if (count > 1) & (nonzero == 0) & (mass != 0):
        shared = (-mass / labels).to(dtype)
        grad = _out(-shared, factor, invalid, G)
        for start in range(0, C, B):
            k = start + tl.arange(0, B)
            positive = k == tl.load(Y + row)
            if GRAD:
                tl.store(G + base + k, grad, mask=(k < C) & positive & (k != first_label))
        others = labels - 1
        penalty += ((alpha * pivot_a) * pivot_a + others * ((alpha * shared) * shared)).to(
            tl.float64
        )
        rest = mass + (others * shared).to(tl.float64)
        pivot = first_label
    pivot_a = (-rest).to(dtype)
    penalty += ((alpha * pivot_a) * pivot_a).to(tl.float64)
    # H from the label probabilities (the recovered one-label mass: the root residual may
    # round to zero), and alpha a_j first in the penalty.
    if count == 1:
        h = log_q + (alpha * q - ((ref - label_max) + dt))
        lp = _label_log_probability(h, pivot_a, q, log_q, log_alpha, CUT, LARGE, STEPS)
        log_p = lp.to(tl.float64)
        log_pq = (lp * q).to(tl.float64)
    cross = -log_p * q
    if ~finite(cross):
        cross = -log_pq
    # Every thread's tile stores land before the pivot's.
    if GRAD:
        tl.debug_barrier()
        g = tl.where(masked, q - pivot_a, -pivot_a)
        tl.store(G + base + pivot, _out(g, factor, invalid, G), (pivot >= 0) & (pivot < C))
    return tl.where(masked, float("inf"), cross + penalty / 2)


@triton.jit
def _out(g, factor, invalid, G):
    """A gradient entry times the row's factor, rounded once into G's dtype (+0.0 at zeros)."""
    return tl.where(invalid, float("nan"), g * factor + 0.0).to(G.dtype.element_ty)


@triton.jit(do_not_specialize=["C", "CS", "SW", "GROUP", "IGNORE", "UP", "ALPHA", "LOG_ALPHA"])
def _kernel(  # noqa: C901 - constexpr specializations of one Triton kernel
    Z,
    Y,
    W,
    R,
    D,
    S,
    V,
    G,
    C,
    CS,
    SW,
    GROUP,
    IGNORE,
    UP,
    ALPHA,
    LOG_ALPHA,
    B: tl.constexpr,
    SINGLE: tl.constexpr,
    FP32: tl.constexpr,
    WEIGHTED: tl.constexpr,
    IGNORED: tl.constexpr,
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
    GATED: tl.constexpr = False,
):
    """One row per program. Float32 work solves a row whose float32 spread overflows in
    float64 (a branch per row: ordinary rows keep the float32 solve).

    SAVE writes the unit gradient (the work type's g before the row factor) into S, rows of
    CS columns, and at column C whether a backward must replay the row (its gradient is not
    g times the factor in the work type). REPLAY solves only the rows so marked."""
    if ALIAS:
        Z = G
    row = tl.program_id(0).to(tl.int64)
    if REPLAY:
        if not saved_replay(S, row, C, CS, FP32):
            return
    if GATED:
        if fast_done(S, row, SW):
            return
    if SAVE:
        G = saved_rows(S, FP32) + row * (CS - C)
    live = row >= 0
    alpha = f64(ALPHA)
    log_alpha = f64(LOG_ALPHA)
    y, inside, ignored, w = target_row(Y, W, row, live, C, IGNORE, GROUP, WEIGHTED, IGNORED)
    masked = inside & (tl.load(Z + row * C + y) == -float("inf"))
    factor = scale(UP, R, D, row, live, ROWWISE, DIVIDED) * w
    if SAVE:
        factor = tl.full([], 1.0, tl.float64)
    loss = tl.zeros([], tl.float64)
    invalid = row < 0
    special = row < 0
    if ignored:
        if GRAD:
            for start in range(0, C, B):
                k = start + tl.arange(0, B)
                tl.store(G + row * C + k, tl.zeros([B], G.dtype.element_ty), k < C)
    elif FP32 and Z.dtype.element_ty != tl.float16:
        j = tl.arange(0, B)
        # Keep the general masked-label limit and steps outside float32's normal range.
        alpha32 = alpha.to(tl.float32)
        include_masks = masked | ~finite(alpha32) | (alpha32 < TINY)
        largest = tl.full([], -float("inf"), tl.float32)
        smallest = tl.full([], float("inf"), tl.float32)
        for start in range(0, C, B):
            valid = start + j < C
            z = tl.load(Z + row * C + start + j, mask=valid, other=0.0).to(tl.float32)
            largest = tl.maximum(largest, tl.max(tl.where(valid, z, -float("inf")), 0))
            # A masked competitor has no mass and adds nothing to the spread.
            kept = valid & ((z > -float("inf")) | include_masks)
            smallest = tl.minimum(smallest, tl.min(tl.where(kept, z, float("inf")), 0))
        if finite(largest - smallest):
            loss, invalid, special = _solve(
                Z,
                Y,
                G,
                row,
                C,
                alpha.to(tl.float32),
                log_alpha.to(tl.float32),
                factor,
                masked,
                B,
                SINGLE,
                GRAD,
                EPS,
                TINY,
                CUT,
                LARGE,
                STEPS,
                RES_STEPS,
            )
        else:
            loss, invalid, special = _solve(
                Z,
                Y,
                G,
                row,
                C,
                alpha,
                log_alpha,
                factor,
                masked,
                B,
                SINGLE,
                GRAD,
                DBL_EPS,
                DBL_MIN,
                -36.0,
                1e16,
                3,
                6,
            )
            special = True | special  # a float64 g in float32 work
    elif FP32:
        loss, invalid, special = _solve(
            Z,
            Y,
            G,
            row,
            C,
            alpha.to(tl.float32),
            log_alpha.to(tl.float32),
            factor,
            masked,
            B,
            SINGLE,
            GRAD,
            EPS,
            TINY,
            CUT,
            LARGE,
            STEPS,
            RES_STEPS,
        )
    else:
        loss, invalid, special = _solve(
            Z,
            Y,
            G,
            row,
            C,
            alpha,
            log_alpha,
            factor,
            masked,
            B,
            SINGLE,
            GRAD,
            EPS,
            TINY,
            CUT,
            LARGE,
            STEPS,
            RES_STEPS,
        )
    if VALUES:
        value = tl.where(w == 0, 0.0, loss * w)
        tl.store(V + row, tl.where(invalid, float("nan"), value).to(V.dtype.element_ty))
    if SAVE:
        tl.store(G + row * C + C, special.to(G.dtype.element_ty))


@triton.jit(do_not_specialize=["C", "CS", "GROUP", "IGNORE", "UP"])
def _scaled(
    Y,
    W,
    R,
    D,
    S,
    G,
    C,
    CS,
    GROUP,
    IGNORE,
    UP,
    B: tl.constexpr,
    FP32: tl.constexpr,
    WEIGHTED: tl.constexpr,
    IGNORED: tl.constexpr,
    ROWWISE: tl.constexpr,
    DIVIDED: tl.constexpr,
    ALIAS: tl.constexpr = False,
):
    """The gradient of rows a forward saved, a chunk of B columns per program (axis 1); the
    rows it marks are left to the replay. G may be the scores (ALIAS), which it does not read."""
    row = tl.program_id(0).to(tl.int64)
    if saved_replay(S, row, C, CS, FP32):
        return
    live = row >= 0
    _, _, ignored, w = target_row(Y, W, row, live, C, IGNORE, GROUP, WEIGHTED, IGNORED)
    factor = scale(UP, R, D, row, live, ROWWISE, DIVIDED) * w
    scaled_chunk(S, G, row, tl.program_id(1), factor, ignored, C, CS, B, FP32)


# The series path of ordinary float32 one-label rows. A class whose limit probability at the
# lowest shift the solve can take, x = exp(a - dt_min), has alpha x <= TAIL (and x <= 1) is a
# tail class: its probability W(alpha x sigma) / alpha, sigma = exp(dt_min - dt) <= 1, is the
# series sum_n c_n alpha^(n-1) (x sigma)^n, which _MOMENTS terms give to about 1e-10
# relative (the next term is c_9 TAIL^8). The solve then iterates on the moments
# P_n = sum (alpha x)^n of the tail and on the head classes (a capacity per chunk), so a
# Halley step costs O(head) instead of a pass over the row: the values read the row twice
# (statistics, moments). The row state S (_STATE + one head count per chunk, float64) is
# all a gradient needs besides the row: one more pass writes it, with each head class's
# probability exact (in chunks that hold one). Other rows take the general solve.
_TAIL = tl.constexpr(1.0 / 32.0)
_MOMENTS = tl.constexpr(8)
_HEAD = 64  # head slots per row, split over its chunks (at least _HEAD_CHUNK each)
_HEAD_CHUNK = 16
_PASS = (2048, 8)  # columns and warps of the series path's chunk passes
_LOWER_STEPS = tl.constexpr(48)
_GUESS_MARGIN = tl.constexpr(0.5)
# Row state: 0 taken, 1 max z, 2 ref, 3 label_max, 4 dt_min, 5 lo, 6 hi, 7 start, 8 solved,
# 9 dt, 10 the label's unit gradient, then the head count of each chunk.
_STATE_COLUMNS = 12
_STATE = tl.constexpr(_STATE_COLUMNS)


def state_width(c: int) -> int:
    """Float64 columns of a series row's state."""
    return _STATE_COLUMNS + 2 * ((triton.cdiv(c, min(_PASS[0], max(16, pow2(c)))) + 1) // 2)


@triton.jit
def _series(n):
    """c_n = (-n)^(n-1) / n!, the coefficients of W(y) = sum c_n y^n (n = 1..8)."""
    return tl.where(
        n == 1, 1.0, tl.where(
        n == 2, -1.0, tl.where(
        n == 3, 1.5, tl.where(
        n == 4, -8.0 / 3.0, tl.where(
        n == 5, 125.0 / 24.0, tl.where(
        n == 6, -10.8, tl.where(
        n == 7, 117649.0 / 5040.0, -2097152.0 / 40320.0))))))
    )  # fmt: skip


@triton.jit
def _series_square(n):
    """The coefficients of W(y)^2 = sum (2 / n) (-n)^(n-2) / (n-2)! y^n (n = 2..8; 0 at 1)."""
    return tl.where(
        n == 1, 0.0, tl.where(
        n == 2, 1.0, tl.where(
        n == 3, -2.0, tl.where(
        n == 4, 4.0, tl.where(
        n == 5, -25.0 / 3.0, tl.where(
        n == 6, 18.0, tl.where(
        n == 7, -2.0 * 16807.0 / 840.0, 262144.0 / 2880.0))))))
    )  # fmt: skip


@triton.jit
def _state(S, row, SW):
    """Row `row` of the series rows' state (float64, also as 32-bit word pairs)."""
    return doubles(S) + row * SW


@triton.jit
def fast_done(S, row, SW):
    """Whether the series path solved this row."""
    return tl.load(_state(S, row, SW) + 8) != 0


@triton.jit
def _tail_probability(x, alpha):
    """W(alpha x) / alpha for alpha x <= TAIL: x times the series in y = alpha x (float32)."""
    y = alpha * x
    poly = 117649.0 / 5040.0
    poly = poly * y - 10.8
    poly = poly * y + 125.0 / 24.0
    poly = poly * y - 8.0 / 3.0
    poly = poly * y + 1.5
    poly = poly * y - 1.0
    return x + (x * y) * poly


@triton.jit(do_not_specialize=["C", "NCH"])
def _fast_stats(Z, Y, Q, C, NCH, B: tl.constexpr):
    """Per chunk: the largest other score, the other scores' sums of e and e^2 relative to
    it (NaN where an entry is NaN or +inf) and the smallest finite score."""
    Q = doubles(Q)
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    k = chunk * B + tl.arange(0, B)
    valid = k < C
    y = tl.load(Y + row).to(tl.int64)
    z = tl.load(Z + row * C + k, valid, other=0.0).to(tl.float32)
    label = k == y
    others = valid & ~label
    inf = float("inf")
    omax = tl.max(tl.where(others, z, -inf), 0)
    shift = tl.where(omax > -inf, omax, 0.0)
    e = tl.where(others, exp(z - shift), 0.0)
    two = tl.arange(0, 2)[None, :]
    sums = tl.sum(tl.where(two == 0, e[:, None], (e * e)[:, None]), 0)
    smallest = tl.min(tl.where(valid & (z > -inf), z, inf), 0)
    base = Q + (row * NCH + chunk) * 8
    tl.store(base + 1, omax.to(tl.float64))
    tl.store(base + 2 + tl.arange(0, 2), sums.to(tl.float64))
    tl.store(base + 4, smallest.to(tl.float64))


@triton.jit
def _lower(x, b_label, b_other, n_other, alpha, log_alpha):
    """h(x) = (W(alpha e^(b_label - x)) + n_other W(alpha e^(b_other - x))) / alpha, a lower
    bound of the step's mass at shift x, and its derivative (float64)."""
    wl = lambert_w_exp(log_alpha + (b_label - x), -36.0, 1e16, 3)
    wo = lambert_w_exp(log_alpha + (b_other - x), -36.0, 1e16, 3)
    h = (wl + n_other * wo) / alpha
    d = -(wl / (1 + wl) + n_other * (wo / (1 + wo))) / alpha
    return h, d


@triton.jit(do_not_specialize=["C", "NCH", "SW", "GROUP", "IGNORE", "ALPHA", "LOG_ALPHA"])
def _fast_prepare(
    Z, Y, W, Q, S, C, NCH, SW, GROUP, IGNORE, ALPHA, LOG_ALPHA,
    CP: tl.constexpr, WEIGHTED: tl.constexpr, IGNORED: tl.constexpr,
    EPS: tl.constexpr, TINY: tl.constexpr,
):  # fmt: skip
    """Merge a row's chunks: whether the series path takes it, its centering, the lowest
    shift dt_min of the solve (below the root), the bracket and the start."""
    Q = doubles(Q)
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, CP)
    live = c < NCH
    base = Q + (row * NCH + c) * 8
    inf = float("inf")
    omax = tl.load(base + 1, live, other=-inf)
    other_z = tl.max(omax, 0)
    osum = tl.load(base + 2, live, other=0.0)
    shift = tl.where(other_z > -inf, other_z, 0.0)
    n_other = tl.sum(tl.where(omax > -inf, osum * exp(omax - shift), 0.0), 0)
    osq = tl.load(base + 3, live, other=0.0)
    n_square = tl.sum(tl.where(omax > -inf, osq * exp(2.0 * (omax - shift)), 0.0), 0)
    smallest = tl.min(tl.load(base + 4, live, other=inf), 0)
    y, inside, ignored, w = target_row(Y, W, row, row >= 0, C, IGNORE, GROUP, WEIGHTED, IGNORED)
    label_z = tl.load(Z + row * C + y).to(tl.float32)
    alpha = f64(ALPHA).to(tl.float32)
    log_alpha = f64(LOG_ALPHA).to(tl.float32)
    # A NaN or +inf entry makes n_other NaN; one at the label leaves the label not finite.
    largest = tl.maximum(other_z.to(tl.float32), label_z)
    spread = largest - smallest.to(tl.float32)
    other32 = other_z.to(tl.float32)
    # _solve's limit of confident rows stays on the general path.
    gap = other_z - label_z.to(tl.float64)
    bound = gap + log(tl.cast(C, tl.float64))
    log_eps = log(tl.full([], EPS, tl.float64))
    invisible = (log_alpha.to(tl.float64) + bound < log_eps - 2.0) | (
        log_alpha.to(tl.float64) + 2.0 * bound < log_eps + log(tl.full([], TINY, tl.float64)) - 2.0
    )
    confident = (bound < 2.0 * log_eps) & invisible
    fast = (
        inside & ~ignored & finite(label_z) & finite(n_other) & finite(spread)
        & finite(other32) & ~confident
    )  # fmt: skip
    zmax = tl.maximum(label_z, other32)
    label_max = label_z - zmax
    other_max = other32 - zmax
    ref = tl.maximum(label_max, other_max)
    q = tl.full([], 1.0, tl.float32)
    log_q = log(q)
    lo = tl.maximum(label_max - alpha * (1 - q), other_max - alpha) + log_q
    hi = tl.maximum(label_max + alpha * q, other_max) + log_q + log(tl.cast(C, tl.float32))
    hi = tl.where(lo <= hi, hi, lo)
    # The mass of the step is at least h(x) of _lower (W(alpha x) / (alpha x) falls with x):
    # the largest x with h(x) >= 1 bounds the root below. Newton steps on log h in a bracket.
    a64 = f64(ALPHA)
    b_label = (label_max - ref).to(tl.float64) + a64
    b_other = (other_max - ref).to(tl.float64)
    n_other = tl.where(fast, n_other, 1.0)
    n_square = tl.where(fast, n_square, 1.0)
    lower = lo.to(tl.float64)
    upper = hi.to(tl.float64)
    x = lower
    moving = fast
    step = 0
    while moving & (step < _LOWER_STEPS):
        h, d = _lower(x, b_label, b_other, n_other, a64, f64(LOG_ALPHA))
        g = log(h)
        lower = tl.where(g >= 0, x, lower)
        upper = tl.where(g >= 0, upper, x)
        following = x - g * h / d
        inner = (lower < following) & (following < upper)
        following = tl.where(inner, following, lower + (upper - lower) / 2)
        # Far finer than the margin below.
        moving = tl.abs(following - x) > 1e-7 * (1 + tl.abs(x))
        x = following
        step += 1
    # The root of the second-order mass sum x - alpha sum x^2 = 1 (x = e^(b - dt)) less a
    # margin, where it is above the bound: fewer head classes, and _fast_solve checks that
    # the root is above it (else the general path solves the row).
    s1 = n_other * exp(b_other) + exp(b_label - a64)
    s2 = n_square * exp(2.0 * b_other) + exp(2.0 * (b_label - a64))
    disc = s1 * s1 - 4.0 * a64 * s2
    estimate = tl.where(disc > 0, -log(2.0 / (s1 + tl.sqrt(tl.maximum(disc, 0.0)))), lower)
    lower = tl.maximum(lower, estimate - _GUESS_MARGIN)
    # A margin below, rounded down into float32.
    dt_min = (lower - 1e-4 * (1 + tl.abs(lower))).to(tl.float32)
    dt_min = tl.where(dt_min.to(tl.float64) <= lower, dt_min, dt_min - tl.abs(dt_min) * 2.4e-7)
    lo = tl.maximum(lo, dt_min)
    hi = tl.where(lo <= hi, hi, lo)
    start = log(n_other * exp(b_other) + exp(b_label - a64)).to(tl.float32) + log_q
    start = tl.minimum(tl.maximum(start, lo), hi)
    out = _state(S, row, SW)
    tl.store(out + 8, 0.0)
    tl.store(out, fast.to(tl.float64))
    tl.store(out + 1, zmax.to(tl.float64))
    tl.store(out + 2, ref.to(tl.float64))
    tl.store(out + 3, label_max.to(tl.float64))
    tl.store(out + 4, dt_min.to(tl.float64))
    tl.store(out + 5, lo.to(tl.float64))
    tl.store(out + 6, hi.to(tl.float64))
    tl.store(out + 7, start.to(tl.float64))


@triton.jit
def _centered_logits(Z, Y, state, row, k, valid, C):
    """a = (z - max z) - ref at columns k, the other classes and the row's dt_min."""
    zmax = tl.load(state + 1).to(tl.float32)
    ref = tl.load(state + 2).to(tl.float32)
    dt_min = tl.load(state + 4).to(tl.float32)
    y = tl.load(Y + row).to(tl.int64)
    z = tl.load(Z + row * C + k, valid, other=0.0).to(tl.float32)
    a = (z - zmax) - ref
    others = valid & (k != y)
    return a, others, dt_min


@triton.jit
def _is_head(x, alpha):
    return (alpha * x > _TAIL) | (x > 1.0)


@triton.jit(do_not_specialize=["C", "NCH", "SW", "ALPHA"])
def _fast_moments(Z, Y, S, P, H, C, NCH, SW, ALPHA, B: tl.constexpr, K: tl.constexpr):
    """Per chunk: the tail moments sum (alpha x)^n, the head count (also in the row state)
    and the head classes' a."""
    P = doubles(P)
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    state = _state(S, row, SW)
    if tl.load(state) == 0:
        return
    k = chunk * B + tl.arange(0, B)
    valid = k < C
    a, others, dt_min = _centered_logits(Z, Y, state, row, k, valid, C)
    alpha = f64(ALPHA).to(tl.float32)
    x = exp(a - dt_min)
    head = others & _is_head(x, alpha)
    # Float32 chunk sums of y <= TAIL, float64 across chunks (as _measure's tiles).
    yv = tl.where(others & ~head, alpha * x, 0.0)
    out = P + (row * NCH + chunk) * 9
    power = yv
    for m in tl.static_range(8):
        tl.store(out + 1 + m, tl.sum(power, 0).to(tl.float64))
        power = power * yv
    count = tl.sum(head.to(tl.int32), 0)
    tl.store(out, count.to(tl.float64))
    tl.store(state + _STATE + chunk, count.to(tl.float64))
    if count > 0:
        slot = tl.cumsum(head.to(tl.int32), 0) - 1
        keep = head & (slot < K)
        at = (row * NCH + chunk) * K + slot
        tl.store(H + at, a, keep)


@triton.jit
def _fast_measure(
    dt, a_head, head, moments, n, dt_min, label_arg, alpha, log_alpha, q, log_q, clip,
    EPS: tl.constexpr, CUT: tl.constexpr, LARGE: tl.constexpr, STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
):  # fmt: skip
    """_measure of a series row: f, f', f'', magnitude and other mass at dt (float32), the
    head probabilities and the tail's mass and sum of squares (float64)."""
    argument = label_arg + dt
    r1 = residual(argument, alpha, q, log_alpha, log_q, EPS, CUT, LARGE, STEPS, RES_STEPS)
    prob = probability((a_head - dt) + log_q, alpha, log_alpha, CUT, LARGE, STEPS)
    counted = prob <= 1
    prob = tl.where(head, tl.minimum(prob, 1.0, propagate_nan=tl.PropagateNan.ALL), 0.0)
    denom = 1 + alpha * prob
    d1 = tl.where(counted & head, prob / denom, 0.0)
    slope = q - r1
    label_denom = 1 + alpha * slope
    label_d1 = tl.where(argument >= clip, slope / label_denom, 0.0)
    a64 = alpha.to(tl.float64)
    nf = n.to(tl.float64)
    sigma = exp((dt_min.to(tl.float64) - dt.to(tl.float64)) * nf)
    terms = _series(nf) * sigma * moments
    tail = tl.sum(terms, 0) / a64
    tail_d1 = tl.sum(terms * nf, 0) / a64
    tail_d2 = tl.sum(terms * (nf * nf), 0) / a64
    tail_sq = tl.sum(_series_square(nf) * sigma * moments, 0) / (a64 * a64)
    mass = tl.sum(prob.to(tl.float64), 0) + tail
    f = r1.to(tl.float64) - mass
    derivative = label_d1.to(tl.float64) + tl.sum(d1.to(tl.float64), 0) + tail_d1
    curvature = (
        -(label_d1 / (label_denom * label_denom)).to(tl.float64)
        - tl.sum((d1 / (denom * denom)).to(tl.float64), 0)
        - tail_d2
    )
    magnitude = tl.abs(r1).to(tl.float64) + mass
    dtype = dt.dtype
    return (
        f.to(dtype), derivative.to(dtype), curvature.to(dtype), magnitude.to(dtype),
        mass.to(dtype), prob, tail, tail_sq,
    )  # fmt: skip


@triton.jit
def _head_slots(P, row, NCH, HP: tl.constexpr, K: tl.constexpr):
    """The head slots of a row (i over its chunks' K slots) and which hold a class."""
    i = tl.arange(0, HP)
    chunk = i // K
    count = tl.load(P + (row * NCH + chunk) * 9, chunk < NCH, other=0.0)
    return i, (chunk < NCH) & ((i % K).to(tl.float64) < count)


@triton.jit(do_not_specialize=["C", "NCH", "SW", "GROUP", "IGNORE", "ALPHA", "LOG_ALPHA"])
def _fast_solve(
    Y, W, P, H, S, V, C, NCH, SW, GROUP, IGNORE, ALPHA, LOG_ALPHA,
    CP: tl.constexpr, HP: tl.constexpr, K: tl.constexpr, WEIGHTED: tl.constexpr,
    IGNORED: tl.constexpr, VALUES: tl.constexpr, EPS: tl.constexpr, TINY: tl.constexpr,
    CUT: tl.constexpr, LARGE: tl.constexpr, STEPS: tl.constexpr, RES_STEPS: tl.constexpr,
):  # fmt: skip
    """The Halley solve of a series row on its head classes and tail moments (_solve's
    steps), its loss, dt and the label's unit gradient; state 8 says it solved the row."""
    P = doubles(P)
    row = tl.program_id(0).to(tl.int64)
    state = _state(S, row, SW)
    if tl.load(state) == 0:
        return
    c = tl.arange(0, CP)
    counts = tl.load(P + (row * NCH + c) * 9, c < NCH, other=0.0)
    if tl.max(counts, 0) > K:
        return
    n = tl.arange(0, 8) + 1
    moments = tl.sum(
        tl.load(P + (row * NCH + c[:, None]) * 9 + n[None, :], (c < NCH)[:, None], other=0.0), 0
    )
    i, head = _head_slots(P, row, NCH, HP, K)
    a_head = tl.load(H + row * NCH * K + i, head, other=0.0)
    alpha = f64(ALPHA).to(tl.float32)
    log_alpha = f64(LOG_ALPHA).to(tl.float32)
    ref = tl.load(state + 2).to(tl.float32)
    label_max = tl.load(state + 3).to(tl.float32)
    dt_min = tl.load(state + 4).to(tl.float32)
    lo = tl.load(state + 5).to(tl.float32)
    hi = tl.load(state + 6).to(tl.float32)
    dt = tl.load(state + 7).to(tl.float32)
    q = tl.full([], 1.0, tl.float32)
    log_q = log(q)
    clip = alpha * (q - 1) + log_q
    label_arg = ref - label_max
    others_count = tl.cast(C - 1, tl.float32)
    f, derivative, curvature, magnitude, mass, prob, tail, tail_sq = _fast_measure(
        dt, a_head, head, moments, n, dt_min, label_arg, alpha, log_alpha, q, log_q, clip,
        EPS, CUT, LARGE, STEPS, RES_STEPS,
    )  # fmt: skip
    if f > 0:
        # The start is above the root: check that dt_min is below it.
        low, _, _, _, _, _, _, _ = _fast_measure(
            lo, a_head, head, moments, n, dt_min, label_arg, alpha, log_alpha, q, log_q,
            clip, EPS, CUT, LARGE, STEPS, RES_STEPS,
        )  # fmt: skip
        if low > 0:
            return
    done = row < 0
    resolved = done
    step = 0
    while ~done & (step < 200):
        jump = log(mass) + alpha * (mass - 1) / others_count
        jump = tl.where((f < 0) & (mass > 1) & finite(jump), jump, -float("inf"))
        following, lo, hi, resolved, done = halley(
            dt, f, derivative, curvature, magnitude, lo, hi, resolved, jump, EPS, TINY
        )
        if ~done:
            dt = following
            f, derivative, curvature, magnitude, mass, prob, tail, tail_sq = _fast_measure(
                dt, a_head, head, moments, n, dt_min, label_arg, alpha, log_alpha, q, log_q,
                clip, EPS, CUT, LARGE, STEPS, RES_STEPS,
            )  # fmt: skip
        step += 1
    if failed(f, magnitude, lo, hi, resolved, EPS, TINY):
        return
    # _finish of a one-label row: the label balances the others, alpha a_j^2 in the penalty.
    a = -prob
    rest = tl.sum(a.to(tl.float64), 0) - tail
    penalty = tl.sum(((alpha * a) * a).to(tl.float64), 0) + alpha.to(tl.float64) * tail_sq
    pivot_a = (-rest).to(tl.float32)
    penalty += ((alpha * pivot_a) * pivot_a).to(tl.float64)
    h = log_q + (alpha * q - (label_arg + dt))
    lp = _label_log_probability(h, pivot_a, q, log_q, log_alpha, CUT, LARGE, STEPS)
    loss = -lp.to(tl.float64) + penalty / 2
    tl.store(state + 9, dt.to(tl.float64))
    tl.store(state + 10, (-pivot_a).to(tl.float64))
    tl.store(state + 8, 1.0)
    if VALUES:
        _, _, _, w = target_row(Y, W, row, row >= 0, C, IGNORE, GROUP, WEIGHTED, IGNORED)
        value = tl.where(w == 0, 0.0, loss * w)
        tl.store(V + row, value.to(V.dtype.element_ty))


@triton.jit
def _fast_factor(Y, W, R, D, row, C, GROUP, IGNORE, UP, ROWWISE: tl.constexpr,
                 DIVIDED: tl.constexpr, WEIGHTED: tl.constexpr, IGNORED: tl.constexpr,
                 ):  # fmt: skip
    """The row's gradient factor in float32 (as _finish rounds it)."""
    _, _, _, w = target_row(Y, W, row, row >= 0, C, IGNORE, GROUP, WEIGHTED, IGNORED)
    return (scale(UP, R, D, row, row >= 0, ROWWISE, DIVIDED) * w).to(tl.float32)


@triton.jit(do_not_specialize=["C", "NCH", "SW", "GROUP", "IGNORE", "UP", "ALPHA", "LOG_ALPHA"])
def _fast_gradient(
    Z, Y, W, R, D, S, L, G, C, NCH, SW, GROUP, IGNORE, UP, ALPHA, LOG_ALPHA,
    B: tl.constexpr, ROWWISE: tl.constexpr, DIVIDED: tl.constexpr, WEIGHTED: tl.constexpr,
    IGNORED: tl.constexpr, CUT: tl.constexpr, LARGE: tl.constexpr, STEPS: tl.constexpr,
    ALIAS: tl.constexpr = False,
):  # fmt: skip
    """The gradient of a solved series row's other classes from its state, a chunk per
    program: the tail's p by the series, a head class's exactly (chunks that hold one), times
    the row factor (rounded once into G's dtype, +0.0 at zeros); and the chunk's float64 sum
    of these p (L), which _fast_label closes."""
    L = doubles(L)
    if ALIAS:
        Z = G
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    state = _state(S, row, SW)
    if tl.load(state + 8) == 0:
        return
    k = chunk * B + tl.arange(0, B)
    valid = k < C
    a, others, dt_min = _centered_logits(Z, Y, state, row, k, valid, C)
    alpha = f64(ALPHA).to(tl.float32)
    dt = tl.load(state + 9).to(tl.float32)
    prob = _tail_probability(exp(a - dt), alpha)
    if tl.load(state + _STATE + chunk) > 0:
        head = _is_head(exp(a - dt_min), alpha)
        log_alpha = f64(LOG_ALPHA).to(tl.float32)
        exact = probability(a - dt, alpha, log_alpha, CUT, LARGE, STEPS)
        exact = tl.minimum(exact, 1.0, propagate_nan=tl.PropagateNan.ALL)
        prob = tl.where(head, exact, prob)
    prob = tl.where(others, prob, 0.0)
    tl.store(L + row * NCH + chunk, tl.sum(prob.to(tl.float64), 0))
    factor = _fast_factor(Y, W, R, D, row, C, GROUP, IGNORE, UP, ROWWISE, DIVIDED, WEIGHTED,
                          IGNORED)  # fmt: skip
    tl.store(G + row * C + k, (prob * factor + 0.0).to(G.dtype.element_ty), others)


@triton.jit(do_not_specialize=["C", "NCH", "SW", "GROUP", "IGNORE", "UP"])
def _fast_label(
    Y, W, R, D, S, L, G, C, NCH, SW, GROUP, IGNORE, UP, CP: tl.constexpr,
    ROWWISE: tl.constexpr, DIVIDED: tl.constexpr, WEIGHTED: tl.constexpr,
    IGNORED: tl.constexpr, ALIAS: tl.constexpr = False,
):  # fmt: skip
    """The label's gradient of a solved series row: minus the sum of the others' p as
    written (_finish's pivot), times the row factor."""
    L = doubles(L)
    row = tl.program_id(0).to(tl.int64)
    if tl.load(_state(S, row, SW) + 8) == 0:
        return
    c = tl.arange(0, CP)
    pivot = tl.sum(tl.load(L + row * NCH + c, c < NCH, other=0.0), 0).to(tl.float32)
    factor = _fast_factor(Y, W, R, D, row, C, GROUP, IGNORE, UP, ROWWISE, DIVIDED, WEIGHTED,
                          IGNORED)  # fmt: skip
    y = tl.load(Y + row).to(tl.int64)
    tl.store(G + row * C + y, (-pivot * factor + 0.0).to(G.dtype.element_ty))


def _shape(c: int) -> tuple[int, int]:
    if c > _REGISTER:
        return _STREAM
    block = max(16, pow2(c))
    return block, max(1, min(16, block // 64))


def _series_path(call: Call) -> bool:
    """Float32 work on label rows with a step in float32's normal range: the series path
    takes the ordinary rows."""
    return (
        call.fp32
        and not call.dense
        and not call.mask
        and call.c >= 2
        and FLOAT32.tiny <= call.gamma <= FLOAT32.max
    )


def plan(call: Call) -> common.Plan:
    """V and G of proximal_cross_entropy.

    Float32 label rows take the series path: chunk statistics, moments and head classes,
    a solve per row (V and the row state S), then the gradient's chunks from S. The rows it
    leaves take one general solve per row (gated on S). A forward that saves keeps S; its
    gradient is the last pass and, for the rows left, their general solve.

    Otherwise every row takes the general solve, and a forward that saves writes the unit
    gradient (S, with a replay flag per row); the gradient from it scales S, replaying the
    flagged rows."""
    if not (math.isfinite(call.gamma) and call.gamma > 0):
        raise ValueError("gamma must be finite and positive")
    b = Builder(call)
    n, c = call.n, call.c
    eps, tiny, cut, large, steps, res_steps = _FLOAT32 if call.fp32 else _FLOAT64
    block, warps = _shape(c)
    cs = saved_stride(c)
    args = dict(
        b.inputs(),
        **b.results(),
        C=c,
        CS=cs,
        SW=0,
        ALPHA=bits(call.gamma),
        LOG_ALPHA=bits(math.log(call.gamma)),
    )
    meta = dict(
        b.flags(),
        B=block,
        SINGLE=c <= _REGISTER,
        EPS=eps,
        TINY=tiny,
        CUT=cut,
        LARGE=large,
        STEPS=steps,
        RES_STEPS=res_steps,
    )
    save = call.saved == common.SAVE
    saved = call.saved == common.SAVED
    values = ("V",) if call.values else ()
    grad = ("G",) if call.grad else ()
    if _series_path(call):
        width, pass_warps = _PASS
        width = min(width, max(16, pow2(c)))
        nch = common.cdiv(c, width)
        sw = state_width(c)
        head = max(_HEAD_CHUNK, pow2(_HEAD // nch))
        args.update(SW=sw, NCH=nch)
        args["S"] = common.Ref("S") if saved else b.array("S", (n, sw), "float64", not save)
        flags: dict[str, Any] = dict(WEIGHTED=call.weighted, IGNORED=call.ignore)
        if not saved:
            args.update(
                Q=b.array("Q", (n, nch, 8), "float64"),
                P=b.array("P", (n, nch, 9), "float64"),
                H=b.array("H", (n, nch, head), "float32"),
            )
            hp = pow2(nch * head)
            b.launch(_fast_stats, (n, nch), args, pass_warps, B=width)
            b.launch(
                _fast_prepare, (n,), args, 1, CP=pow2(nch), EPS=eps, TINY=tiny,
                enable_reflect_ftz=False, **flags,
            )  # fmt: skip
            b.launch(
                _fast_moments, (n, nch), args, pass_warps, state=("S", "P", "H"), B=width,
                K=head, enable_reflect_ftz=False,
            )  # fmt: skip
            b.launch(
                _fast_solve, (n,), args, max(1, min(8, hp // 256)), state=("S",) + values,
                CP=pow2(nch), HP=hp, K=head, VALUES=call.values, EPS=eps, TINY=tiny, CUT=cut,
                LARGE=large, STEPS=steps, RES_STEPS=res_steps, enable_reflect_ftz=False,
                **flags,
            )  # fmt: skip
        if call.grad:
            factors = dict(flags, ROWWISE=call.rowwise, DIVIDED=call.divided)
            args["L"] = b.array("L", (n, nch), "float64")
            b.launch(
                _fast_gradient, (n, nch), args, pass_warps, B=width, CUT=cut, LARGE=large,
                STEPS=steps, enable_reflect_ftz=False, **factors,
            )  # fmt: skip
            b.launch(_fast_label, (n,), args, 1, state=("G",), CP=pow2(nch), **factors)
        # libdevice keeps float32 subnormals (the probabilities of far classes), as on the CPU.
        rest = dict(meta, VALUES=call.values and not saved, GATED=True)
        state = (() if saved else values) + grad
        b.launch(_kernel, (n,), args, warps, state=state, enable_reflect_ftz=False, **rest)
        return b.plan()
    if saved:
        args["S"] = common.Ref("S")
    elif save:
        args["S"] = b.array("S", (n, cs), "float32" if call.fp32 else "float64", False)
    else:
        args["S"] = args["Z"]
    if not saved:
        solve = dict(meta, GRAD=call.grad or save, SAVE=save)
        b.launch(_kernel, (n,), args, warps, enable_reflect_ftz=False, **solve)
        if not (save and call.grad):
            return b.plan()
    width, copy_warps = _COPY
    b.launch(
        _scaled,
        (n, common.cdiv(c, width)),
        args,
        copy_warps,
        B=width,
        FP32=call.fp32,
        WEIGHTED=call.weighted,
        IGNORED=call.ignore,
        ROWWISE=call.rowwise,
        DIVIDED=call.divided,
    )
    replay = dict(meta, VALUES=False, REPLAY=True)
    b.launch(_kernel, (n,), args, warps, state=("G",), enable_reflect_ftz=False, **replay)
    return b.plan()
