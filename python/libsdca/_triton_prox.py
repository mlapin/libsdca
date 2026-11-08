"""proximal_entropy on CUDA in one Triton kernel: the solve of _tensor_prox
with one program per example, so a call is a single launch. Rows up to
REGISTER_CLASSES stay in registers; wider ones stream tiles, a pass per solve step.
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from . import _cuda_launch

if os.environ.get("TRITON_INTERPRET") == "1":
    # The interpreter (CPU, for tests) has no libdevice.
    @triton.jit
    def _exp(x):
        return tl.exp(x)

    @triton.jit
    def _log(x):
        return tl.log(x)

    @triton.jit
    def _log1p(x):
        u = 1 + x
        return tl.where(u == 1, x, tl.log(u) * (x / (u - 1)))

    @triton.jit
    def _below(x, FP32: tl.constexpr):
        # x (1 - eps/2) rounds to the next float/double toward zero of a positive x.
        return x * (0.99999994 if FP32 else 0.9999999999999999)

else:

    @triton.jit
    def _exp(x):
        return libdevice.exp(x)

    @triton.jit
    def _log(x):
        return libdevice.log(x)

    @triton.jit
    def _log1p(x):
        return libdevice.log1p(x)

    @triton.jit
    def _below(x, FP32: tl.constexpr):
        return libdevice.nextafter(x, x * 0)


@triton.jit
def _finite(x):
    return x - x == 0


@triton.jit
def _lambert_w_exp(
    x, CUT: tl.constexpr, LARGE: tl.constexpr, STEPS: tl.constexpr, FP32: tl.constexpr
):
    """W(exp(x)), as _tensor_prox._lambert_w_exp."""
    u = _log(tl.where(x > 30, x, _log1p(_exp(x))))
    for _ in tl.static_range(STEPS):
        e = _exp(u)
        g = (e - x) + u
        u = u - g / ((e + 1) - g * e / (2 * (e + 1)))
    w = _exp(u)
    if not FP32:
        w = w - ((w - x) + _log(w)) * (w / (w + 1))
    w = tl.where(x > LARGE, x - _log(x), w)
    return tl.where(x < CUT, _exp(x), tl.where(x == float("inf"), x, w))


@triton.jit
def _probability(
    h,
    alpha,
    log_alpha,
    CUT: tl.constexpr,
    LARGE: tl.constexpr,
    STEPS: tl.constexpr,
    FP32: tl.constexpr,
):
    arg = h + log_alpha
    w = _lambert_w_exp(arg, CUT, LARGE, STEPS, FP32)
    prob_cut: tl.constexpr = -9.0 if FP32 else -18.0
    return tl.where(arg < prob_cut, _exp(h - w), w / alpha)


@triton.jit
def _residual(
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
    FP32: tl.constexpr,
):
    """r in [q - 1, q) with alpha r - log1p(-r / q) = t."""
    lo = q - 1
    half = alpha * q / 2
    far = (t > half + 0.6931471805599453) | (t < -half - 0.4054651081081644)
    prob = _probability(log_q + (alpha * q - t), alpha, log_alpha, CUT, LARGE, STEPS, FP32)
    outer = tl.minimum(
        tl.maximum(q - prob, lo, propagate_nan=tl.PropagateNan.ALL),
        q,
        propagate_nan=tl.PropagateNan.ALL,
    )
    r = _near_residual(tl.where(far, 0.0, t), alpha, q, log_q, EPS, RES_STEPS, FP32)
    r = tl.where(far, outer, r)
    return tl.where(t == 0, 0.0, tl.where(t <= alpha * lo + log_q, lo, r))


@triton.jit
def _near_residual(
    s, alpha, q, log_q, EPS: tl.constexpr, RES_STEPS: tl.constexpr, FP32: tl.constexpr
):
    """_residual of a t inside the near range [-alpha q / 2 - log 1.5, alpha q / 2 + log 2]
    (or zero), without the Lambert-W probability of the far range."""
    lo = q - 1
    inv_q = 1 / q
    linear = s / (alpha + inv_q)
    r = linear / (1 + (linear * inv_q) / (2 * (1 + alpha * q)))
    r = tl.minimum(
        tl.maximum(r, lo, propagate_nan=tl.PropagateNan.ALL),
        _below(q, FP32),
        propagate_nan=tl.PropagateNan.ALL,
    )
    low = lo + r * 0
    high = q + r * 0
    tol = 8 * EPS * tl.abs(s)
    for _ in tl.static_range(RES_STEPS):
        f = (alpha * r - _log1p(-r * inv_q)) - s
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
def _single_residual(
    dt,
    ref,
    label_max,
    count,
    alpha,
    log_alpha,
    q,
    log_q,
    EPS: tl.constexpr,
    CUT: tl.constexpr,
    LARGE: tl.constexpr,
    STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
    FP32: tl.constexpr,
):
    """The label residual of a one-label row, once per row on a scalar (else zero)."""
    r1 = dt * 0
    if count == 1:
        r1 = _residual(
            (ref - label_max) + dt,
            alpha,
            q,
            log_alpha,
            log_q,
            EPS,
            CUT,
            LARGE,
            STEPS,
            RES_STEPS,
            FP32,
        )
    return r1


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
    FP32: tl.constexpr,
):
    """f, f', f'', magnitude, other mass, label residuals and other probabilities of
    the lanes; r1 is _single_residual."""
    argument = (ref - v) + dt
    if count == 1:
        r_label = tl.where(positive, r1, 0.0)
    else:
        r_label = tl.where(
            positive,
            _residual(
                argument, alpha, q, log_alpha, log_q, EPS, CUT, LARGE, STEPS, RES_STEPS, FP32
            ),
            0.0,
        )
    prob = _probability(((v - ref) - dt) + log_q, alpha, log_alpha, CUT, LARGE, STEPS, FP32)
    counted = prob <= 1
    prob = tl.where(others, tl.minimum(prob, 1.0, propagate_nan=tl.PropagateNan.ALL), 0.0)
    mass = tl.sum(prob, 0)
    slope = q - r_label
    label_denom = 1 + alpha * slope
    denom = 1 + alpha * prob
    label_d1 = tl.where(positive & (argument >= clip), slope / label_denom, 0.0)
    d1 = tl.where(counted & others, prob / denom, 0.0)
    derivative = tl.sum(label_d1, 0) + tl.sum(d1, 0)
    curvature = -tl.sum(label_d1 / (label_denom * label_denom), 0) - tl.sum(d1 / (denom * denom), 0)
    magnitude = tl.sum(tl.abs(r_label), 0) + mass
    f = tl.sum(r_label, 0) - mass
    return f, derivative, curvature, magnitude, mass, r_label, prob


@triton.jit
def _start(
    label_max, other_max, ref, tail, count, alpha, q, log_q, log_classes, FP32: tl.constexpr
):
    """The bracket [lo, hi] of dt and the softmax-limit start inside it."""
    label = label_max - ref
    other = other_max - ref
    lo = tl.maximum(label - alpha * (1 - q), other - alpha) + log_q
    hi = tl.maximum(label + alpha * q, other) + log_q + log_classes
    hi = tl.where(lo <= hi, hi, lo)
    tiny_cut: tl.constexpr = -7.971192359924316 if FP32 else -18.021826694558577
    tiny = (count == 1) & (label_max == ref) & (other + log_classes < tiny_cut)
    dt = tl.where(tiny, log_q, log_q + _log1p(tail))
    dt = tl.minimum(
        tl.maximum(dt, lo, propagate_nan=tl.PropagateNan.ALL), hi, propagate_nan=tl.PropagateNan.ALL
    )
    return lo, hi, dt


@triton.jit
def _advance(
    dt, f, derivative, curvature, magnitude, mass, lo, hi, resolved, alpha, others, EPS, TINY
):
    """One safeguarded step: the next dt, the bracket, resolved and done."""
    done = tl.abs(f) <= 8 * EPS * tl.maximum(magnitude, TINY)
    lo = tl.where(~done & (f < 0), dt, lo)
    hi = tl.where(~done & (f >= 0), dt, hi)
    # Halley, or a step in the log of the other mass where it exceeds one.
    newton = -f / derivative
    correction = (f * curvature) / (2 * derivative * derivative)
    clamped = tl.minimum(tl.maximum(correction, -0.5), 0.5)
    delta = tl.where(_finite(correction), newton / (1 - clamped), newton)
    log_step = _log(mass) + alpha * (mass - 1) / others
    jump = (f < 0) & (mass > 1) & _finite(log_step) & (log_step > delta)
    delta = tl.where(jump, log_step, delta)
    delta = tl.where((derivative > 0) & _finite(derivative) & _finite(newton), delta, 0.0)
    following = dt + delta
    # A nonzero step below the resolution of dt: dt is the closest root.
    resolved = resolved | (~done & (delta != 0) & (following == dt))
    done = done | resolved
    inside = (lo <= following) & (following <= hi) & (following != dt) & _finite(following)
    following = tl.where(inside, following, lo + (hi - lo) / 2)
    done = done | (following == dt)
    return following, lo, hi, resolved, done


@triton.jit
def _failed(f, magnitude, lo, hi, resolved, EPS, TINY):
    # The resolution of dt and of f is eps relative to at least tiny: they can underflow.
    width = 64 * EPS * tl.maximum(tl.maximum(tl.abs(lo), tl.abs(hi)), TINY)
    far = tl.abs(f) > 256 * EPS * tl.maximum(magnitude, TINY)
    return ~_finite(f) | (~resolved & far & (hi - lo > width))


@triton.jit
def _label_log_probability(
    h, r_label, q, log_q, log_alpha, CUT: tl.constexpr, LARGE: tl.constexpr,
    STEPS: tl.constexpr, FP32: tl.constexpr,
):  # fmt: skip
    """log p of a label: log1p where r is small, else log(W / alpha) = h - W."""
    far = h - _lambert_w_exp(h + log_alpha, CUT, LARGE, STEPS, FP32)
    near = log_q + _log1p(-r_label / q)
    return tl.where(r_label <= q / 2, near, far)


@triton.jit
def _tiny_tail(gap, count, log_alpha, classes, FP32: tl.constexpr):
    log_tiny: tl.constexpr = -87.3365447505531 if FP32 else -708.3964185322641
    log_bound: tl.constexpr = -18.021826694558577 if FP32 else -38.12309493079699
    # Absolute error < one subnormal quantum, or relative error < working eps.
    tail = gap + _log((classes - 1).to(gap.dtype))
    error = tl.maximum(log_alpha, 0.0) + tail
    negligible = (error < log_bound) | (error + tail < log_tiny + log_bound)
    return (count == 1) & _finite(gap) & (gap < log_tiny) & negligible


@triton.jit
def _tiny_entropy(
    z_ptr, positive_ptr, loss_ptr, gradient_ptr, status_ptr, row, classes, zmax,
    label_max, other_max, BLOCK: tl.constexpr, FP32: tl.constexpr,
):  # fmt: skip
    """Sum before rounding into subnormals; the proximal correction is negligible."""
    dtype: tl.constexpr = tl.float32 if FP32 else tl.float64
    base = row * classes
    j = tl.arange(0, BLOCK)
    total = tl.zeros([BLOCK], tl.float64)
    label = classes
    half = _exp(((other_max - label_max).to(tl.float64)) / 2)
    for start in range(0, classes, BLOCK):
        k = start + j
        valid = k < classes
        z = tl.load(z_ptr + base + k, mask=valid, other=0.0).to(dtype)
        positive = tl.load(positive_ptr + base + k, mask=valid, other=0) != 0
        label = tl.minimum(label, tl.min(tl.where(positive & valid, k, classes), 0))
        relative = ((z - zmax) - other_max).to(tl.float64)
        term = _exp(tl.where(valid & ~positive, relative, -float("inf")))
        total += term
        tl.store(gradient_ptr + base + k, (half * term) * half, mask=valid)
    mass = half * (half * tl.sum(total, 0))
    tl.debug_barrier()
    tl.store(gradient_ptr + base + label, -mass)
    tl.store(loss_ptr + row, mass)
    tl.store(status_ptr + row, 0)


@triton.jit(do_not_specialize=["classes"])
def _solve_kernel(
    z_ptr,
    positive_ptr,
    params_ptr,
    loss_ptr,
    gradient_ptr,
    status_ptr,
    classes,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    TINY: tl.constexpr,
    CUT: tl.constexpr,
    LARGE: tl.constexpr,
    STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
    FP32: tl.constexpr,
):
    dtype: tl.constexpr = tl.float32 if FP32 else tl.float64
    row = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, BLOCK)
    valid = j < classes
    alpha = tl.load(params_ptr).to(dtype)
    log_alpha = tl.load(params_ptr + 1).to(dtype)
    z = tl.load(z_ptr + row * classes + j, mask=valid, other=0.0).to(dtype)
    positive = (tl.load(positive_ptr + row * classes + j, mask=valid, other=0) != 0) & valid
    others = valid & ~positive
    count = tl.sum(positive.to(tl.int32), 0)
    bad = (count < 1) | (count >= classes) | (tl.sum((valid & ~_finite(z)).to(tl.int32), 0) > 0)
    v = tl.where(valid, z - tl.max(tl.where(valid, z, -float("inf")), 0), -float("inf"))
    # Unsupported centered differences must fail, not become equal-score lanes.
    bad = bad | (tl.sum((valid & ~_finite(v)).to(tl.int32), 0) > 0)
    v = tl.where(_finite(v) | ~valid, v, 0.0)

    q = 1 / count.to(dtype)
    log_q = _log(q)
    clip = alpha * (q - 1) + log_q
    label_max = tl.max(tl.where(positive, v, -float("inf")), 0)
    other_max = tl.max(tl.where(others, v, -float("inf")), 0)
    ref = tl.maximum(label_max, other_max)
    log_classes = _log(classes.to(dtype))
    if ~bad & _tiny_tail(other_max - label_max, count, log_alpha, classes, FP32):
        _tiny_entropy(
            z_ptr, positive_ptr, loss_ptr, gradient_ptr, status_ptr, row, classes,
            tl.max(tl.where(valid, z, -float("inf")), 0), label_max, other_max, BLOCK, FP32,
        )  # fmt: skip
    else:
        shifted = v - ref
        first = tl.argmax(shifted, 0)
        tail = tl.sum(tl.where(j == first, 0.0, _exp(shifted)), 0)
        lo, hi, dt = _start(
            label_max, other_max, ref, tail, count, alpha, q, log_q, log_classes, FP32
        )

        r1 = _single_residual(
            dt, ref, label_max, count, alpha, log_alpha, q, log_q,
            EPS, CUT, LARGE, STEPS, RES_STEPS, FP32,
        )  # fmt: skip
        f, derivative, curvature, magnitude, mass, r_label, prob = _evaluate(
            dt, v, ref, r1, count, positive, others, alpha, log_alpha, q, log_q, clip,
            EPS, CUT, LARGE, STEPS, RES_STEPS, FP32,
        )  # fmt: skip
        done = bad
        resolved = count < 0
        step = 0
        while ~done & (step < 200):
            following, lo, hi, resolved, done = _advance(
                dt, f, derivative, curvature, magnitude, mass, lo, hi, resolved, alpha,
                (classes - count).to(dtype), EPS, TINY,
            )  # fmt: skip
            if ~done:
                dt = following
                r1 = _single_residual(
                    dt, ref, label_max, count, alpha, log_alpha, q, log_q,
                    EPS, CUT, LARGE, STEPS, RES_STEPS, FP32,
                )  # fmt: skip
                f, derivative, curvature, magnitude, mass, r_label, prob = _evaluate(
                    dt, v, ref, r1, count, positive, others, alpha, log_alpha, q, log_q, clip,
                    EPS, CUT, LARGE, STEPS, RES_STEPS, FP32,
                )  # fmt: skip
            step += 1
        failed = _failed(f, magnitude, lo, hi, resolved, EPS, TINY)
        tl.store(status_ptr + row, tl.where(bad, 1, tl.where(failed, 2, 0)).to(tl.int8))

        # Zero-sum to one rounding: the label (or the largest entry) balances the
        # others. Labels that are all zero share the recovered mass of the others.
        a = tl.where(positive, r_label, -prob)
        mass = tl.sum(tl.where(positive, 0.0, a), 0)
        labels_zero = tl.sum((positive & (a != 0)).to(tl.int32), 0) == 0
        share = (count > 1) & labels_zero & (mass != 0)
        a = tl.where(share & positive, -mass / count.to(dtype), a)
        first = tl.argmax(positive.to(tl.int32), 0)
        pivot = tl.where((count == 1) | share, first, tl.argmax(tl.abs(a), 0))
        a = tl.where(j == pivot, -tl.sum(tl.where(j == pivot, 0.0, a), 0), a)
        # H from the label probabilities, and alpha a_j first in the penalty.
        h = log_q + (alpha * q - ((ref - v) + dt))
        # Use the recovered one-label mass: the root residual may round to zero.
        r_value = tl.where(count == 1, -mass, r_label)
        log_p = _label_log_probability(h, r_value, q, log_q, log_alpha, CUT, LARGE, STEPS, FP32)
        log_p = tl.where(positive, log_p, 0.0)
        cross = -tl.sum(log_p, 0) * q
        if ~_finite(cross):
            cross = -tl.sum(log_p * q, 0)
        loss = cross + tl.sum(tl.where(valid, (alpha * a) * a, 0.0), 0) / 2
        invalid = bad | failed
        loss = tl.where(invalid, float("nan"), loss)
        grad = tl.where(invalid, float("nan"), -a).to(gradient_ptr.dtype.element_ty)
        tl.store(loss_ptr + row, loss)
        tl.store(gradient_ptr + row * classes + j, grad, mask=valid)


@triton.jit
def _tile(z_ptr, positive_ptr, base, start, classes, zmax, BLOCK: tl.constexpr, dtype):
    """Columns, labels, other classes and centered scores (zero off the row) of a tile."""
    k = start + tl.arange(0, BLOCK)
    valid = k < classes
    z = tl.load(z_ptr + base + k, mask=valid, other=0.0).to(dtype)
    positive = (tl.load(positive_ptr + base + k, mask=valid, other=0) != 0) & valid
    v = z - zmax
    return k, valid, positive, valid & ~positive, tl.where(valid & _finite(v), v, 0.0)


@triton.jit
def _stream_evaluate(
    z_ptr, positive_ptr, base, classes, zmax, dt, ref, label_max, count, alpha, log_alpha,
    q, log_q, clip, BLOCK: tl.constexpr, EPS: tl.constexpr, CUT: tl.constexpr,
    LARGE: tl.constexpr, STEPS: tl.constexpr, RES_STEPS: tl.constexpr, FP32: tl.constexpr,
):  # fmt: skip
    """f, f', f'', magnitude and other mass of a row, one tile at a time.

    The tile sums accumulate in float64: a float32 sum over many tiles misplaces the root.
    """
    r1 = _single_residual(
        dt, ref, label_max, count, alpha, log_alpha, q, log_q,
        EPS, CUT, LARGE, STEPS, RES_STEPS, FP32,
    )  # fmt: skip
    f = tl.zeros([], tl.float64)
    derivative = f
    curvature = f
    magnitude = f
    mass = f
    for start in range(0, classes, BLOCK):
        _, _, positive, others, v = _tile(
            z_ptr, positive_ptr, base, start, classes, zmax, BLOCK, dt.dtype
        )
        tf, td, tc, tm, tmass, _, _ = _evaluate(
            dt, v, ref, r1, count, positive, others, alpha, log_alpha, q, log_q, clip,
            EPS, CUT, LARGE, STEPS, RES_STEPS, FP32,
        )  # fmt: skip
        f += tf.to(tl.float64)
        derivative += td.to(tl.float64)
        curvature += tc.to(tl.float64)
        magnitude += tm.to(tl.float64)
        mass += tmass.to(tl.float64)
    dtype = dt.dtype
    return (
        f.to(dtype),
        derivative.to(dtype),
        curvature.to(dtype),
        magnitude.to(dtype),
        mass.to(dtype),
    )


@triton.jit
def _stream_stats(z_ptr, positive_ptr, base, classes, BLOCK: tl.constexpr, dtype):
    """The largest score, the largest label and other score, the first largest column,
    the first label, the label count and whether the row is outside the domain."""
    inf = float("inf")
    j = tl.arange(0, BLOCK)
    zmax = tl.full([], -inf, dtype)
    zmin = tl.full([], inf, dtype)
    label_z = zmax
    other_z = zmax
    first = classes * 0
    first_label = classes
    count = classes * 0
    nonfinite = classes * 0
    for start in range(0, classes, BLOCK):
        k = start + j
        valid = k < classes
        z = tl.load(z_ptr + base + k, mask=valid, other=0.0).to(dtype)
        positive = (tl.load(positive_ptr + base + k, mask=valid, other=0) != 0) & valid
        count += tl.sum(positive.to(tl.int32), 0)
        nonfinite += tl.sum((valid & ~_finite(z)).to(tl.int32), 0)
        scores = tl.where(valid, z, -inf)
        largest = tl.max(scores, 0)
        first = tl.where(largest > zmax, start + tl.argmax(scores, 0), first)
        zmax = tl.maximum(zmax, largest)
        zmin = tl.minimum(zmin, tl.min(tl.where(valid, z, inf), 0))
        label_z = tl.maximum(label_z, tl.max(tl.where(positive, z, -inf), 0))
        other_z = tl.maximum(other_z, tl.max(tl.where(valid & ~positive, z, -inf), 0))
        first_label = tl.minimum(first_label, tl.min(tl.where(positive, k, classes), 0))
    # Rounding is monotonic: zmin - zmax overflows if and only if a centered score does.
    bad = (count < 1) | (count >= classes) | (nonfinite > 0) | ~_finite(zmin - zmax)
    return zmax, label_z, other_z, first, first_label, count, bad


@triton.jit
def _stream_finish(
    z_ptr, positive_ptr, gradient_ptr, base, classes, zmax, dt, ref, label_max, count,
    first_label, invalid, alpha, log_alpha, q, log_q, clip, BLOCK: tl.constexpr,
    EPS: tl.constexpr, CUT: tl.constexpr, LARGE: tl.constexpr, STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr, FP32: tl.constexpr,
):  # fmt: skip
    """Write -a at dt and return the loss. The pivot (the label, else the first largest
    |a|) is left out of the sums and written last, balancing the others. The sums over
    tiles accumulate in float64."""
    r1 = _single_residual(
        dt, ref, label_max, count, alpha, log_alpha, q, log_q,
        EPS, CUT, LARGE, STEPS, RES_STEPS, FP32,
    )  # fmt: skip
    labels = count.to(dt.dtype)
    rest = tl.zeros([], tl.float64)
    penalty = rest
    pivot_a = dt * 0
    pivot = count * 0 - 1
    mass = rest
    log_p = rest
    log_pq = rest
    nonzero = count * 0
    for start in range(0, classes, BLOCK):
        k, valid, positive, others, v = _tile(
            z_ptr, positive_ptr, base, start, classes, zmax, BLOCK, dt.dtype
        )
        _, _, _, _, _, r_label, prob = _evaluate(
            dt, v, ref, r1, count, positive, others, alpha, log_alpha, q, log_q, clip,
            EPS, CUT, LARGE, STEPS, RES_STEPS, FP32,
        )  # fmt: skip
        a = tl.where(positive, r_label, -prob)
        mass += tl.sum(tl.where(positive, 0.0, a), 0).to(tl.float64)
        nonzero += tl.sum((positive & (a != 0)).to(tl.int32), 0)
        if count != 1:
            h = log_q + (alpha * q - ((ref - v) + dt))
            lp = _label_log_probability(h, r_label, q, log_q, log_alpha, CUT, LARGE, STEPS, FP32)
            lp = tl.where(positive, lp, 0.0)
            log_p += tl.sum(lp, 0).to(tl.float64)
            log_pq += tl.sum(lp * q, 0).to(tl.float64)
        candidate = tl.where(count == 1, first_label, start + tl.argmax(tl.abs(a), 0))
        at = k == candidate
        candidate_a = tl.sum(tl.where(at, a, 0.0), 0)
        better = tl.where(
            count == 1,
            (start <= candidate) & (candidate < start + BLOCK),
            (pivot < 0) | (tl.abs(candidate_a) > tl.abs(pivot_a)),
        )
        dropped = tl.where(better, pivot_a, candidate_a)
        rest += (dropped + tl.sum(tl.where(at, 0.0, a), 0)).to(tl.float64)
        penalty += ((alpha * dropped) * dropped + tl.sum(tl.where(at, 0.0, (alpha * a) * a), 0)).to(
            tl.float64
        )
        pivot_a = tl.where(better, candidate_a, pivot_a)
        pivot = tl.where(better, candidate, pivot)
        grad = tl.where(invalid, float("nan"), -a).to(gradient_ptr.dtype.element_ty)
        tl.store(gradient_ptr + base + k, grad, mask=valid)
    # Labels that are all zero share the recovered mass of the others; the first balances.
    if (count > 1) & (nonzero == 0) & (mass != 0):
        shared = (-mass / labels).to(dt.dtype)
        grad = tl.where(invalid, float("nan"), -shared).to(gradient_ptr.dtype.element_ty)
        for start in range(0, classes, BLOCK):
            k = start + tl.arange(0, BLOCK)
            positive = tl.load(positive_ptr + base + k, mask=k < classes, other=0) != 0
            tl.store(gradient_ptr + base + k, grad, mask=positive & (k != first_label))
        others = labels - 1
        penalty += ((alpha * pivot_a) * pivot_a + others * ((alpha * shared) * shared)).to(
            tl.float64
        )
        rest = mass + (others * shared).to(tl.float64)
        pivot = first_label
    pivot_a = (-rest).to(dt.dtype)
    penalty += ((alpha * pivot_a) * pivot_a).to(tl.float64)
    # H from the label probabilities, and alpha a_j first in the penalty.
    if count == 1:
        h = log_q + (alpha * q - ((ref - label_max) + dt))
        lp = _label_log_probability(h, pivot_a, q, log_q, log_alpha, CUT, LARGE, STEPS, FP32)
        log_p = lp.to(tl.float64)
        log_pq = (lp * q).to(tl.float64)
    cross = -log_p * q
    if ~_finite(cross):
        cross = -log_pq
    # Every thread's tile stores land before the pivot's.
    tl.debug_barrier()
    grad = tl.where(invalid, float("nan"), -pivot_a).to(gradient_ptr.dtype.element_ty)
    tl.store(gradient_ptr + base + pivot, grad)
    return (cross + penalty / 2).to(dt.dtype)


@triton.jit(do_not_specialize=["classes"])
def _stream_solve_kernel(
    z_ptr,
    positive_ptr,
    params_ptr,
    loss_ptr,
    gradient_ptr,
    status_ptr,
    classes,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    TINY: tl.constexpr,
    CUT: tl.constexpr,
    LARGE: tl.constexpr,
    STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
    FP32: tl.constexpr,
):
    """_solve_kernel over tiles of BLOCK columns: each solve step is one pass over the row."""
    dtype: tl.constexpr = tl.float32 if FP32 else tl.float64
    row = tl.program_id(0).to(tl.int64)
    base = row * classes
    alpha = tl.load(params_ptr).to(dtype)
    log_alpha = tl.load(params_ptr + 1).to(dtype)
    zmax, label_z, other_z, first, first_label, count, bad = _stream_stats(
        z_ptr, positive_ptr, base, classes, BLOCK, dtype
    )
    q = 1 / count.to(dtype)
    log_q = _log(q)
    clip = alpha * (q - 1) + log_q
    label_max = label_z - zmax
    other_max = other_z - zmax
    ref = tl.maximum(label_max, other_max)
    log_classes = _log(classes.to(dtype))
    if ~bad & _tiny_tail(other_max - label_max, count, log_alpha, classes, FP32):
        _tiny_entropy(
            z_ptr, positive_ptr, loss_ptr, gradient_ptr, status_ptr, row, classes,
            zmax, label_max, other_max, BLOCK, FP32,
        )  # fmt: skip
    else:
        tail = tl.zeros([BLOCK], dtype)
        for start in range(0, classes, BLOCK):
            k, valid, _, _, v = _tile(z_ptr, positive_ptr, base, start, classes, zmax, BLOCK, dtype)
            tail += tl.where(valid & (k != first), _exp(v - ref), 0.0)
        lo, hi, dt = _start(
            label_max, other_max, ref, tl.sum(tail, 0), count, alpha, q, log_q, log_classes, FP32
        )

        f, derivative, curvature, magnitude, mass = _stream_evaluate(
            z_ptr, positive_ptr, base, classes, zmax, dt, ref, label_max, count, alpha, log_alpha,
            q, log_q, clip, BLOCK, EPS, CUT, LARGE, STEPS, RES_STEPS, FP32,
        )  # fmt: skip
        done = bad
        resolved = count < 0
        step = 0
        while ~done & (step < 200):
            following, lo, hi, resolved, done = _advance(
                dt, f, derivative, curvature, magnitude, mass, lo, hi, resolved, alpha,
                (classes - count).to(dtype), EPS, TINY,
            )  # fmt: skip
            if ~done:
                dt = following
                f, derivative, curvature, magnitude, mass = _stream_evaluate(
                    z_ptr, positive_ptr, base, classes, zmax, dt, ref, label_max, count, alpha,
                    log_alpha, q, log_q, clip, BLOCK, EPS, CUT, LARGE, STEPS, RES_STEPS, FP32,
                )  # fmt: skip
            step += 1
        failed = _failed(f, magnitude, lo, hi, resolved, EPS, TINY)
        tl.store(status_ptr + row, tl.where(bad, 1, tl.where(failed, 2, 0)).to(tl.int8))
        invalid = bad | failed
        loss = _stream_finish(
            z_ptr, positive_ptr, gradient_ptr, base, classes, zmax, dt, ref, label_max, count,
            first_label, invalid, alpha, log_alpha, q, log_q, clip, BLOCK,
            EPS, CUT, LARGE, STEPS, RES_STEPS, FP32,
        )  # fmt: skip
        tl.store(loss_ptr + row, tl.where(invalid, float("nan"), loss))


@triton.jit
def _solve(
    z_ptr, positive_ptr, params_ptr, loss_ptr, gradient_ptr, status_ptr, classes,
    BLOCK: tl.constexpr, EPS: tl.constexpr, TINY: tl.constexpr, CUT: tl.constexpr,
    LARGE: tl.constexpr, STEPS: tl.constexpr, RES_STEPS: tl.constexpr, FP32: tl.constexpr,
    STREAM: tl.constexpr,
):  # fmt: skip
    if STREAM:
        _stream_solve_kernel(
            z_ptr, positive_ptr, params_ptr, loss_ptr, gradient_ptr, status_ptr, classes,
            BLOCK, EPS, TINY, CUT, LARGE, STEPS, RES_STEPS, FP32,
        )  # fmt: skip
    else:
        _solve_kernel(
            z_ptr, positive_ptr, params_ptr, loss_ptr, gradient_ptr, status_ptr, classes,
            BLOCK, EPS, TINY, CUT, LARGE, STEPS, RES_STEPS, FP32,
        )  # fmt: skip


@triton.jit(do_not_specialize=["classes"])
def _kernel(
    z_ptr,
    positive_ptr,
    params_ptr,
    loss_ptr,
    gradient_ptr,
    status_ptr,
    classes,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    TINY: tl.constexpr,
    CUT: tl.constexpr,
    LARGE: tl.constexpr,
    STEPS: tl.constexpr,
    RES_STEPS: tl.constexpr,
    FP32: tl.constexpr,
    STREAM: tl.constexpr,
):
    if FP32 and z_ptr.dtype.element_ty != tl.float16:
        row = tl.program_id(0).to(tl.int64)
        j = tl.arange(0, BLOCK)
        largest = tl.full([], -float("inf"), tl.float32)
        smallest = tl.full([], float("inf"), tl.float32)
        for start in range(0, classes if STREAM else 1, BLOCK):
            valid = start + j < classes
            z = tl.load(z_ptr + row * classes + start + j, mask=valid, other=0.0).to(tl.float32)
            largest = tl.maximum(largest, tl.max(tl.where(valid, z, -float("inf")), 0))
            smallest = tl.minimum(smallest, tl.min(tl.where(valid, z, float("inf")), 0))
        # Finite float32/bfloat16 inputs can have an unrepresentable difference.
        # Branch per row on the device: ordinary rows retain the float32 solve.
        if ~_finite(largest - smallest):
            _solve(
                z_ptr, positive_ptr, params_ptr, loss_ptr, gradient_ptr, status_ptr, classes,
                BLOCK, 2.220446049250313e-16, 2.2250738585072014e-308, -36.0, 1e16, 3, 6,
                False, STREAM,
            )  # fmt: skip
        else:
            _solve(
                z_ptr, positive_ptr, params_ptr, loss_ptr, gradient_ptr, status_ptr, classes,
                BLOCK, EPS, TINY, CUT, LARGE, STEPS, RES_STEPS, True, STREAM,
            )  # fmt: skip
    else:
        _solve(
            z_ptr, positive_ptr, params_ptr, loss_ptr, gradient_ptr, status_ptr, classes,
            BLOCK, EPS, TINY, CUT, LARGE, STEPS, RES_STEPS, FP32, STREAM,
        )  # fmt: skip


# The widest rows held in registers; wider ones stream tiles of _STREAM_BLOCK columns.
REGISTER_CLASSES = 4096
_STREAM_BLOCK, _STREAM_WARPS = 2048, 16


_params: dict[tuple[int, torch.dtype, float, int | None], torch.Tensor] = {}


def _alpha_params(device: torch.device, work: torch.dtype, alpha: float) -> torch.Tensor:
    # Scalar float arguments would be float32 in the kernel; cache per device, dtype and alpha.
    # A scalar initialized on another stream may not be ready for this launch.
    stream = torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None
    key = (device.index or 0, work, alpha, stream)
    if key not in _params:
        if len(_params) > 64:
            _params.clear()
        with torch.inference_mode(False):
            # Device fills also work on the first call inside a captured graph;
            # torch.tensor([...], device=device) copies pageable CPU memory.
            params = torch.empty(2, dtype=work, device=device)
            params[0].fill_(alpha)
            params[1].fill_(math.log(alpha))
            _params[key] = params
    return _params[key]


def proximal_entropy(
    z: torch.Tensor, positive: torch.Tensor, alpha: float, fp32: bool | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss (examples,) and gradient -a at CUDA scores (examples, classes).

    The solve is in float32 if fp32 (default: float32 scores), else float64.
    Rows whose spread overflows float32 promote on the device. The loss is in
    the requested work type and the gradient in promote(scores' dtype, float32): backward
    scales it by the upstream derivative before the only rounding to 16 bits.
    """
    if not (math.isfinite(alpha) and alpha > 0):
        raise ValueError("alpha must be finite and positive")
    examples, classes = z.shape
    with _cuda_launch.device_guard(z):
        fp32 = z.dtype == torch.float32 if fp32 is None else fp32
        work = torch.float32 if fp32 else torch.float64
        eps = torch.finfo(work).eps
        cut, large, steps, res_steps = (-18.0, 1e7, 2, 4) if fp32 else (-36.0, 1e16, 3, 6)
        z, positive = z.contiguous(), positive.contiguous().view(torch.uint8)
        # Keep the unrounded parameters for a row's possible float64 fallback.
        params = _alpha_params(z.device, torch.float64, alpha)
        loss = torch.empty(examples, dtype=work, device=z.device)
        gradient = torch.empty(
            z.shape, dtype=torch.promote_types(z.dtype, torch.float32), device=z.device
        )
        status = torch.empty(examples, dtype=torch.int8, device=z.device)
        if examples:
            stream = classes > REGISTER_CLASSES
            block = _STREAM_BLOCK if stream else max(16, triton.next_power_of_2(classes))
            warps = _STREAM_WARPS if stream else max(1, min(16, block // 64))
            _kernel[(examples,)](
                z,
                positive,
                params,
                loss,
                gradient,
                status,
                classes,
                BLOCK=block,
                EPS=eps,
                TINY=torch.finfo(work).tiny,
                CUT=cut,
                LARGE=large,
                STEPS=steps,
                RES_STEPS=res_steps,
                FP32=fp32,
                STREAM=stream,
                num_warps=warps,
                # libdevice keeps float32 subnormals (the probabilities of far classes), as on the CPU.
                enable_reflect_ftz=False,
            )
            if not z.is_cuda:
                worst = int(status.max())
                if worst == 1:
                    raise ValueError(
                        "proximal_cross_entropy needs finite scores, labels and a non-target class"
                    )
                if worst == 2:
                    raise RuntimeError("Entropy normalization did not converge")
        return loss, gradient
