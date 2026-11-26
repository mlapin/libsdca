"""Device math of every Triton kernel: one exp, log, log1p, expm1 per work type, the
compensated row arithmetic the losses share (margins, smooth-hinge terms), and the row target
and gradient factor of the launch contract.

Under TRITON_INTERPRET=1 (CPU tests) libdevice is unavailable; the tl functions stand in.
"""

from __future__ import annotations

import os
import struct

import triton
import triton.language as tl

from .._limits import FLOAT32, FLOAT64

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
FLT_MAX = tl.constexpr(FLOAT32.max)
DBL_MAX = tl.constexpr(FLOAT64.max)
DBL_EPS = tl.constexpr(FLOAT64.eps)
DBL_MIN = tl.constexpr(FLOAT64.tiny)
_FLT_ONE_BELOW = tl.constexpr(FLOAT32.one_below)
_DBL_ONE_BELOW = tl.constexpr(FLOAT64.one_below)
_FLT_MIN_OVER_EPS = tl.constexpr(FLOAT32.min_over_eps)
_DBL_MIN_OVER_EPS = tl.constexpr(FLOAT64.min_over_eps)
# exp2(t) rounds to zero in float32 below t = log2(denorm_min) - 1.
_EXP2_MIN_NORMAL = tl.constexpr(FLOAT32.min_exponent - 1)
_EXP2_SHIFT = tl.constexpr(FLOAT32.digits)
_EXP2_RESCALE = tl.constexpr(2.0**-FLOAT32.digits)
_EXP2_ZERO = tl.constexpr(FLOAT32.min_exponent - FLOAT32.digits - 1)


def bits(value: float) -> int:
    """A float64 as the int64 of its bits: kernels read float parameters with `f64`."""
    return struct.unpack("<q", struct.pack("<d", float(value)))[0]


if INTERPRET:

    @triton.jit
    def exp(x):
        return tl.exp(x)

    @triton.jit
    def log(x):
        return tl.log(x)

    @triton.jit
    def sqrt(x):
        return tl.sqrt(x)

    @triton.jit
    def log1p(x):
        u = 1 + x
        return tl.where((u == 1) | (u == float("inf")), x, tl.log(u) * (x / (u - 1)))

    @triton.jit
    def expm1(x):
        u = tl.exp(x)
        kahan = (u - 1) * (x / tl.log(u))
        return tl.where(u == 1, x, tl.where((u - 1 == -1) | (u == float("inf")), u - 1, kahan))

    @triton.jit
    def below(x):
        """The next value toward zero of a positive x."""
        if x.dtype == tl.float32:
            y = x * _FLT_ONE_BELOW
        else:
            y = x * _DBL_ONE_BELOW
        return y

else:
    from triton.language.extra import libdevice

    @triton.jit
    def exp(x):
        return libdevice.exp(x)

    @triton.jit
    def log(x):
        return libdevice.log(x)

    @triton.jit
    def sqrt(x):
        return libdevice.sqrt(x)

    @triton.jit
    def log1p(x):
        return libdevice.log1p(x)

    @triton.jit
    def expm1(x):
        return libdevice.expm1(x)

    @triton.jit
    def below(x):
        """The next value toward zero of a positive x."""
        return libdevice.nextafter(x, x * 0)


@triton.jit
def _exp_correction(x, t, r):
    d = tl.fma(t, -0.693145751953125, x)
    d = tl.fma(t, -1.4286067653301870e-06, d)
    return tl.fma(r, d, r)


@triton.jit
def normal_exp32(x):
    """The corrected CUDA exponential where the result is normal."""
    t = x * 1.4426950408889634
    return _exp_correction(x, t, tl.exp2(t))


@triton.jit
def bounded_tile_parts(x, error, B: tl.constexpr, BOUND: tl.constexpr):
    """Two float32 partial sums of B entries bounded by BOUND, including product residuals."""
    bias: tl.constexpr = 1.5 * B * BOUND
    high = (x + bias) - bias
    low = (x - high) + error
    return tl.sum(high, 1), tl.sum(low, 1)


@triton.jit
def bounded_tile_sum(x, error, B: tl.constexpr, BOUND: tl.constexpr):
    """Round B numbers bounded by BOUND to multiples of ulp(bias), whose sum is exact:
    each integer coefficient has magnitude at most 2**23 / B. error is the product's
    low part for a dot product. Only two row scalars need double addition."""
    high, low = bounded_tile_parts(x, error, B, BOUND)
    return high.to(tl.float64) + low.to(tl.float64)


@triton.jit
def exp32(x):
    """exp of float32 x <= 0: exp2 of t = x log2(e) times 1 + d, d = x - t ln(2) from a two-part
    ln(2) (|d| <= 2**-23 |x|: d**2 / 2 is below float32 rounding). 0 below t = -150 (x = -inf
    included: _EXP2_ZERO); NaN stays NaN."""
    t = x * 1.4426950408889634
    # CUDA exp2.approx flushes a subnormal result. Round the corrected normal
    # exponential first, then scale back into the subnormal range.
    small = t < _EXP2_MIN_NORMAL
    r = tl.exp2(tl.where(small, t + _EXP2_SHIFT, t))
    corrected = _exp_correction(x, t, r)
    return tl.where(t < _EXP2_ZERO, 0.0, corrected * tl.where(small, _EXP2_RESCALE, 1.0))


@triton.jit
def exp_nonpositive(x):
    """exp of x <= 0 (or NaN): exp32 in float32, libdevice in float64."""
    if x.dtype == tl.float32:
        y = exp32(x)
    else:
        y = exp(x)
    return y


EXP_WIDE = tl.constexpr(18446744073709551616.0)
EXP_UNWIDE = tl.constexpr(5.421010862427522e-20)


@triton.jit
def exp_parts(x):
    """Factors whose product is exp(x), with half exponents in the far tail."""
    if x.dtype == tl.float32:
        tiny = x < -75
    else:
        tiny = x < -700
    e = exp_nonpositive(tl.where(tiny, 0.5 * x, x))
    return e, tl.where(tiny, e, 1.0)


@triton.jit
def add_parts(a, ea, b, eb):
    """Add two leading values and their rounding errors."""
    total = a + b
    back = total - a
    carry = (a - (total - back)) + (b - back)
    return total, (ea + eb) + carry


@triton.jit
def wide_exp(x):
    """exp(x) scaled by 2**64 before its small entries round to subnormals."""
    e, f = exp_parts(x)
    return (e * EXP_WIDE) * f


@triton.jit
def bounded_expm1(x):
    """expm1 on [-2, 2]: a degree-8 Chebyshev fit of (expm1(x) - x) / x**2 in float32 (exact
    signed zero and tiny x), libdevice in float64."""
    if x.dtype == tl.float32:
        p = tl.full(x.shape, 2.9499494758056244e-07, tl.float32)
        p = tl.fma(p, x, 2.9900852496211883e-06)
        p = tl.fma(p, x, 2.4742717869230546e-05)
        p = tl.fma(p, x, 0.00019770115613937378)
        p = tl.fma(p, x, 0.0013889545807614923)
        p = tl.fma(p, x, 0.008334128186106682)
        p = tl.fma(p, x, 0.04166664555668831)
        p = tl.fma(p, x, 0.1666664332151413)
        p = tl.fma(p, x, 0.5)
        y = x * tl.fma(x, p, 1.0)
    else:
        y = expm1(x)
    return y


@triton.jit
def f64(word):
    """A float64 parameter passed as the int64 of its bits (`bits`): Triton passes Python
    floats as float32."""
    return tl.cast(word, tl.int64).to(tl.float64, bitcast=True)


@triton.jit
def doubles(pointer):
    """A float64 buffer, also when it reaches the kernel as 32-bit words (JAX without x64)."""
    if pointer.dtype.element_ty != tl.float64:
        pointer = pointer.to(tl.pointer_type(tl.float64), bitcast=True)
    return pointer


@triton.jit
def work(x, FP32: tl.constexpr):
    """x in the work type: float32 or float64."""
    if FP32:
        y = x.to(tl.float32)
    else:
        y = x.to(tl.float64)
    return y


@triton.jit
def finite(x):
    return (x - x) == 0


@triton.jit
def softplus(x):
    """log(1 + exp(x)), exp(x) below -36 and x above 36 (+inf and -inf included)."""
    mid = log1p(exp(tl.minimum(tl.maximum(x, -40.0), 40.0)))
    return tl.where(x < -36, exp(x), tl.where(x > 36, x, mid))


@triton.jit
def label_statistics(gap, total, SINGLE_LOG: tl.constexpr = False):
    """Loss, off-label mass and inverse partition at the largest score.
    Split exp(-abs(gap)) so a representable tail is rounded after multiplication by total."""
    half = exp(-0.5 * tl.abs(gap))
    above = gap >= 0
    odds = half * (half * total)
    partition = tl.where(above, total + half * half, 1.0 + odds)
    first = 1.0 / partition
    # Above the label the competitors carry at least half the probability. Their
    # mass is well conditioned as one minus the label probability; FMA avoids
    # rounding the large numerator and its reciprocal separately.
    complement = tl.fma(-tl.where(above, half * half, 1.0), first, 1.0)
    mass = tl.where(above | (odds >= 1.0), complement, odds * first)
    if SINGLE_LOG:
        # All lanes of a one-row program agree on the sign. Run just the applicable
        # libdevice logarithm on throughput-bound grids of these programs.
        if tl.reshape(above, ()):
            value = gap + log(partition)
        else:
            value = log1p(odds)
    else:
        value = tl.where(above, gap + log(partition), log1p(odds))
    return value, mass, first


@triton.jit
def margin(z, label):
    """1 + z - label; where it cancels, with the low parts of both roundings (TwoSum)."""
    d = z - label
    r = d - z
    low = (z - (d - r)) + (-label - r)
    h = d + 1
    r = h - d
    low = low + ((d - (h - r)) + (1 - r))
    cancels = ((h.dtype == tl.float32) | (tl.abs(h) < 0.0625 * tl.abs(d))) & finite(low)
    return tl.where(cancels, h + low, h)


@triton.jit
def smooth_terms(h, keep, a, gamma, AXIS: tl.constexpr):
    """Sum over a > 0 of (h a - a**2 / 2) / gamma: the smooth hinges' value at the projection a,
    scaled by a / gamma first where the product leaves the normal range."""
    ha = h * a
    aa = a * a / 2
    v = tl.abs(ha - aa)
    if h.dtype == tl.float32:
        normal = (v <= FLT_MAX) & (v >= _FLT_MIN_OVER_EPS)
    else:
        normal = (v <= DBL_MAX) & (v >= _DBL_MIN_OVER_EPS)
    term = tl.where(normal, (ha - aa) / gamma, (h - a / 2) * (a / gamma))
    return tl.sum(tl.where(keep & (a > 0), term, 0.0), AXIS)


@triton.jit
def scale(UP, R, D, row, live, ROWWISE: tl.constexpr, DIVIDED: tl.constexpr):
    """A row's gradient factor (float64): upstream (UP, float64 bits) times row_upstream
    (ROWWISE 1: R[row], 2: R[0]; any float dtype), over D[0] where DIVIDED."""
    s = f64(UP) + tl.zeros(row.shape, tl.float64)
    if ROWWISE == 1:
        s = s * tl.load(R + row, live, other=0).to(tl.float64)
    if ROWWISE == 2:
        s = s * tl.load(R).to(tl.float64)
    if DIVIDED:
        s = s / tl.load(D).to(tl.float64)
    return s


@triton.jit
def target_row(
    Y, W, row, live, C: tl.constexpr, IGNORE, GROUP, WEIGHTED: tl.constexpr, IGNORED: tl.constexpr
):
    """A row's label clamped into [0, C), whether it is a valid target (in range, or ignored),
    whether it is ignored, and its weight: W[label] of the row's group of GROUP rows (1 without
    weights), 0 where ignored."""
    y = tl.load(Y + row, live, other=0).to(tl.int64)
    ignored = (y == IGNORE) & IGNORED
    inside = ((y >= 0) & (y < C)) | ignored
    y = tl.minimum(tl.maximum(y, 0), C - 1).to(tl.int32)
    if WEIGHTED:
        w = tl.load(W + (row // GROUP) * C + y, live, other=0).to(tl.float64)
    else:
        w = tl.zeros(row.shape, tl.float64) + 1.0
    return y, inside, ignored, tl.where(ignored, 0.0, w)
