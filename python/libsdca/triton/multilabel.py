"""multilabel_hinge, nonsmooth and smooth (gamma > 0), as include/sdca/loss/multilabel.h, in
float64 on boolean masks.

Rows wider than _CHUNK columns of float32 or 16-bit scores run as parallel aligned row chunks
(rows.py) whose extremes and smooth candidates a float64 merge per row combines; narrower rows
(and float64 scores) stream in one program. The backward recomputes the gradient per element
from the row statistics, or only writes it where the nonsmooth extremes are unique. The smooth hinge
projects the margins onto x >= 0 with equal positive and negative sums <= gamma (the
two-simplex): two simplex projections by Michelot's method from a running superset, or a
balanced threshold whose entries are fixed as Kiwiel's method fixes them. Quotients by gamma
are float64 divisions, never a reciprocal that a subnormal gamma would overflow.

A score of -inf at a negative class masks it (margin -inf: no term, gradient +0.0); a row whose
negatives are all masked has loss zero if any positive is finite. A masked positive against
a finite negative gives infinite loss. NaN, +inf, all-masked rows and rows without a positive
or a negative give NaN rows.
"""

from __future__ import annotations

import triton
import triton.language as tl

from . import common, rows
from .common import Builder, Call, pow2
from .math import DBL_EPS, bits, doubles, f64, finite, margin, scale, smooth_terms
from .rows import chunks as row_chunks

_CHUNK = 2048
_BACKWARD = 1024


@triton.jit
def _load(Z, M, row, k, C: tl.constexpr):
    """Scores (float64) and positive flags of columns k, their bounds, and whether a score is
    below +inf (a masked positive shows in the least positive score)."""
    valid = k < C
    z = tl.load(Z + row * C + k, valid, other=0).to(tl.float64)
    pos = (tl.load(M + row * C + k, valid, other=0) != 0) & valid
    return z, pos, valid, valid & (z < float("inf"))


@triton.jit
def _extremes(Z, M, row, z1, p1, v1, f1, C: tl.constexpr, B: tl.constexpr, TIES: tl.constexpr):
    """Least positive and largest negative score with their multiplicities, the first positive,
    and whether the row is invalid (NaN, +inf, all masked, no positive or negative).
    Masked classes are no negatives: without another negative the largest is -inf."""
    j = tl.arange(0, B)
    low = tl.full([], float("inf"), tl.float64)
    high = tl.full([], -float("inf"), tl.float64)
    n_low = tl.zeros([], tl.int32)
    n_high = n_low
    first = tl.full([], C, tl.int32)
    count = n_low
    masked_pos = n_low
    nonfinite = n_low
    for start in range(0, C, B):
        k = start + j
        z, pos, valid, finite = z1, p1, v1, f1
        if B < C:
            z, pos, valid, finite = _load(Z, M, row, k, C)
        nonfinite += tl.sum((valid & ~finite).to(tl.int32), 0)
        count += tl.sum(pos.to(tl.int32), 0)
        masked_pos += tl.sum((pos & (z == -float("inf"))).to(tl.int32), 0)
        first = tl.minimum(first, tl.min(tl.where(pos, k, C), 0))
        zp = tl.where(pos & finite, z, float("inf"))
        zn = tl.where(finite & ~pos, z, -float("inf"))
        lo = tl.min(zp, 0)
        hi = tl.max(zn, 0)
        if TIES:
            lows = tl.sum((pos & (z == lo)).to(tl.int32), 0)
            highs = tl.sum((valid & ~pos & (z == hi)).to(tl.int32), 0)
            n_low = tl.where(lo < low, lows, n_low + tl.where(lo == low, lows, 0))
            n_high = tl.where(hi > high, highs, n_high + tl.where(hi == high, highs, 0))
        low = tl.minimum(low, lo)
        high = tl.maximum(high, hi)
    all_masked = (high == -float("inf")) & (masked_pos == count)
    bad = (nonfinite > 0) | all_masked | (count == 0) | (count == C)
    return low, n_low, high, n_high, first, bad


@triton.jit
def _smooth_margins(z, pos, z0, low, recenter):
    """Margins 1/2 -+ (z - z0), or -(z - low) and 1 + z - low where the largest pairwise margin
    nearly cancels."""
    d = z - z0
    plain = tl.where(pos, 0.5 - d, 0.5 + d)
    shifted = tl.where(pos, low - z, margin(z, low))
    # Masked positives share the positive simplex. The negatives project their gaps to
    # the largest negative, independently of the finite positive scores.
    limit = tl.where(pos, tl.where(z == -float("inf"), 0.0, -float("inf")), d)
    separate = tl.where(pos, low - z, d)
    return tl.where(
        low == -float("inf"),
        limit,
        tl.where(recenter == 2, separate, tl.where(recenter == 1, shifted, plain)),
    )


@triton.jit
def _margins_at(
    Z, M, row, start, j, p1, v1, h1, z0, low, recenter, C: tl.constexpr, B: tl.constexpr
):
    """Positive flags, bounds and smooth margins of columns start + j."""
    pos, valid, h = p1, v1, h1
    if B < C:
        z, pos, valid, _ = _load(Z, M, row, start + j, C)
        h = _smooth_margins(z, pos, z0, low, recenter)
    return pos, valid, h


@triton.jit
def _superset(h, side, top, count, total, gamma):
    """Michelot's first superset {g > -gamma} of the gaps g = h - top to a running top: a gain
    of gamma leaves the old gaps all <= -gamma (a first top resets)."""
    new = tl.maximum(top, tl.max(tl.where(side, h, -float("inf")), 0))
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
def _thresholds(
    Z, M, row, j, p1, v1, h1, z0, low, recenter, skip, gamma, C: tl.constexpr, B: tl.constexpr
):
    """Origins and corrections (oa, ca) of the positives and (ob, cb) of the negatives."""
    top_a = tl.full([], -float("inf"), tl.float64)
    top_b = top_a
    n_a = tl.zeros([], tl.int32)
    n_b = n_a
    sum_a = tl.zeros([], tl.float64)
    sum_b = sum_a
    for start in range(0, C, B):
        pos, valid, h = _margins_at(Z, M, row, start, j, p1, v1, h1, z0, low, recenter, C, B)
        top_a, n_a, sum_a = _superset(h, pos, top_a, n_a, sum_a, gamma)
        top_b, n_b, sum_b = _superset(h, valid & ~pos, top_b, n_b, sum_b, gamma)
    s_a = tl.maximum((sum_a - gamma) / n_a, -gamma)
    s_b = tl.maximum((sum_b - gamma) / n_b, -gamma)
    searching = ~skip
    steps = 0
    while searching:
        n_a = tl.zeros([], tl.int32)
        n_b = n_a
        sum_a = tl.zeros([], tl.float64)
        sum_b = sum_a
        least_a = tl.full([], float("inf"), tl.float64)
        least_b = least_a
        for start in range(0, C, B):
            pos, valid, h = _margins_at(Z, M, row, start, j, p1, v1, h1, z0, low, recenter, C, B)
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
    oa, ca, ob, cb = top_a, s_a, top_b, s_b
    # The sum is active unless the thresholds add up below zero, to their own rounding.
    terms = tl.maximum(gamma, (tl.abs(top_a) + tl.abs(top_b)) + (tl.abs(s_a) + tl.abs(s_b)))
    if (
        ~skip
        & (low > -float("inf"))
        & (recenter != 2)
        & ((top_a + top_b) + (s_a + s_b) < -DBL_EPS * terms)
    ):
        # Equal sums below gamma: x = max(0, a - o - t), y = max(0, b + o + t) over the
        # supports, entries fixed as two_simplex.h (Kiwiel) from the simplex supports.
        origin = top_a * 0.5 - top_b * 0.5
        size = tl.zeros([], tl.float64)
        sum_a = size
        sum_b = size
        m = tl.zeros([], tl.int32)
        for start in range(0, C, B):
            pos, valid, h = _margins_at(Z, M, row, start, j, p1, v1, h1, z0, low, recenter, C, B)
            in_a = pos & (h - top_a > s_a)
            in_b = valid & ~pos & (h - top_b > s_b)
            da = h - origin
            db = h + origin
            size += tl.sum(tl.where(in_a, tl.abs(da), 0.0) + tl.where(in_b, tl.abs(db), 0.0), 0)
            sum_a += tl.sum(tl.where(in_a, da, 0.0), 0)
            sum_b += tl.sum(tl.where(in_b, db, 0.0), 0)
            m += tl.sum((in_a | in_b).to(tl.int32), 0)
        tol = 8 * DBL_EPS * size
        t = (sum_a - sum_b) / m.to(tl.float64)
        cut_a = tl.full([], -float("inf"), tl.float64)
        cut_b = cut_a
        fixing = m > 0
        fixes = 0
        while fixing:
            sum_x = tl.zeros([], tl.float64)
            sum_y = sum_x
            n_x = tl.zeros([], tl.int32)
            n_y = n_x
            for start in range(0, C, B):
                pos, valid, h = _margins_at(
                    Z, M, row, start, j, p1, v1, h1, z0, low, recenter, C, B
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
            drop_x = ~drop_y & (n_x > 0) & (infeasible < -tol)
            mt = m.to(tl.float64) * t
            following = tl.where(drop_y, mt + sum_y, mt - sum_x)
            cut_a = tl.where(drop_x, tl.maximum(cut_a, t), cut_a)
            cut_b = tl.where(drop_y, tl.maximum(cut_b, -t), cut_b)
            m = tl.where(drop_y, m - n_y, tl.where(drop_x, m - n_x, m))
            t = tl.where((drop_x | drop_y) & (m > 0), following / m.to(tl.float64), t)
            fixes += 1
            fixing = (drop_x | drop_y) & (m > 0) & (fixes <= C)
        oa, ca, ob, cb = origin, t, -origin, -t
    return oa, ca, ob, cb


@triton.jit(do_not_specialize=["N", "GAMMA"])
def _forward(
    Z,
    Y,
    V,
    S,
    N,
    GAMMA,
    C: tl.constexpr,
    B: tl.constexpr,
    SMOOTH: tl.constexpr,
    VALUES: tl.constexpr,
    GRAD: tl.constexpr,
    F=None,
    GENERAL: tl.constexpr = False,
    STRIDE: tl.constexpr = 7,
):
    """Row values and statistics: (low, high, positive and negative shares) or, smooth, (z0, low,
    recenter, oa, ca, ob, cb); the last slot is NaN for an invalid row. GENERAL: only the
    rows F marks (_smooth_merge left them), whose state has STRIDE words (then NaN)."""
    S = doubles(S)
    gamma = f64(GAMMA)
    row = tl.program_id(0).to(tl.int64)
    if GENERAL:
        if tl.load(doubles(F) + row) == 0:
            return
    j = tl.arange(0, B)
    z1, p1, v1, f1 = _load(Z, Y, row, j, C)
    low, n_low, high, n_high, first, bad = _extremes(
        Z, Y, row, z1, p1, v1, f1, C, B, GRAD and not SMOOTH
    )
    nan = float("nan")
    if SMOOTH:
        # A row whose negatives are all masked has no pair of scores: the origin +inf of its
        # positives and the margins -inf of its negatives project to zero.
        alone = ~bad & (high == -float("inf"))
        z0 = tl.load(Z + row * C + tl.minimum(first, C - 1)).to(tl.float64)
        z0 = tl.where(low == -float("inf"), tl.where(alone, 0.0, high), z0)
        recenter = tl.abs((high - low) + 1) < 0.0625 * (1 + tl.abs(low - z0) + tl.abs(high - z0))
        # At mass gamma each side's simplex threshold is at most gamma below
        # its largest margin. A pair gap of at least 2 gamma saturates both sides.
        # Center the two projections independently to retain small score differences.
        pair = margin(high, low)
        # A nonpositive largest margin projects both simplices to zero.
        alone = alone | (pair <= 0)
        separate = finite(low) & finite(high) & (pair >= 2 * gamma)
        recenter = tl.where(separate, 2, recenter.to(tl.int32))
        z0 = tl.where(separate, high, z0)
        h1 = _smooth_margins(z1, p1, z0, low, recenter)
        oa, ca, ob, cb = _thresholds(
            Z, Y, row, j, p1, v1, h1, z0, low, recenter, bad | alone, gamma, C, B
        )
        oa = tl.where(alone, float("inf"), oa)
        ca = tl.where(alone, 0.0, ca)
        ob = tl.where(alone, float("inf"), ob)
        cb = tl.where(alone, 0.0, cb)
        value = tl.zeros([], tl.float64)
        for start in range(0, C, B):
            pos, valid, h = _margins_at(Z, Y, row, start, j, p1, v1, h1, z0, low, recenter, C, B)
            value += smooth_terms(h, valid, _projection(h, pos, oa, ca, ob, cb, gamma), gamma, 0)
        value += tl.where(separate, pair, 0.0)
        value = tl.where(alone, 0.0, tl.where(low == -float("inf"), float("inf"), value))
        if GRAD:
            tl.store(S + row * STRIDE, z0)
            tl.store(S + row * STRIDE + 1, low)
            tl.store(S + row * STRIDE + 2, recenter.to(tl.float64))
            tl.store(S + row * STRIDE + 3, oa)
            tl.store(S + row * STRIDE + 4, ca)
            tl.store(S + row * STRIDE + 5, ob)
            tl.store(S + row * STRIDE + 6, tl.where(bad, nan, cb))
            if STRIDE > 7:
                tl.store(S + row * STRIDE + 7, nan)
    else:
        h = tl.where(high == -float("inf"), -float("inf"), margin(high, low))
        value = tl.maximum(h, 0.0)
        if GRAD:
            slope = tl.where(h == 0, 0.5, (h > 0).to(tl.float64))
            tl.store(S + row * 4, low)
            tl.store(S + row * 4 + 1, high)
            tl.store(S + row * 4 + 2, -slope / tl.maximum(n_low, 1).to(tl.float64))
            tl.store(
                S + row * 4 + 3, tl.where(bad, nan, slope / tl.maximum(n_high, 1).to(tl.float64))
            )
    if VALUES:
        tl.store(V + row, tl.where(bad, nan, value).to(V.dtype.element_ty))


@triton.jit(do_not_specialize=["N", "UP", "GAMMA"])
def _backward(
    Z,
    Y,
    R,
    D,
    S,
    G,
    N,
    UP,
    GAMMA,
    C: tl.constexpr,
    B: tl.constexpr,
    SMOOTH: tl.constexpr,
    ROWWISE: tl.constexpr,
    DIVIDED: tl.constexpr,
    ALIAS: tl.constexpr = False,
):
    """B columns of a row's gradient per program (G may be Z)."""
    if ALIAS:
        Z = G
    S = doubles(S)
    row = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1) * B + tl.arange(0, B)
    s = scale(UP, R, D, row, row >= 0, ROWWISE, DIVIDED)
    if SMOOTH:
        gamma = f64(GAMMA)
        z, pos, valid, _ = _load(Z, Y, row, k, C)
        z0 = tl.load(S + row * 7)
        low = tl.load(S + row * 7 + 1)
        recenter = tl.load(S + row * 7 + 2).to(tl.int32)
        cb = tl.load(S + row * 7 + 6)
        h = _smooth_margins(z, pos, z0, low, recenter)
        x = (
            _projection(
                h,
                pos,
                tl.load(S + row * 7 + 3),
                tl.load(S + row * 7 + 4),
                tl.load(S + row * 7 + 5),
                cb,
                gamma,
            )
            / gamma
        )
        g = tl.where(cb != cb, float("nan"), tl.where(pos, -x, x) * s)
    else:
        z, pos, valid, _ = _load(Z, Y, row, k, C)
        low = tl.load(S + row * 4)
        high = tl.load(S + row * 4 + 1)
        wp = tl.load(S + row * 4 + 2)
        wn = tl.load(S + row * 4 + 3)
        g = tl.where(
            wn != wn,
            float("nan"),
            tl.where(pos & (z == low), wp, tl.where(~pos & (z == high), wn, 0.0)) * s,
        )
    # + 0.0: zero entries are +0.0 for either sign of the upstream derivative.
    tl.store(G + row * C + k, (g + 0.0).to(G.dtype.element_ty), valid)


# Wide nonsmooth rows: parallel aligned row chunks (rows.py) write their extremes
# (_HINGE_PART float64 words), one program per row merges them, and the gradient of a row with
# a unique least positive and largest negative only writes (two entries, zeros elsewhere).
# Rows with ties there read their chunks to find every tied score (_hinge_write).
_PARTS = (2048, 4)  # at most 4095 columns: _hinge_partial's 12-bit counts
_WRITE = (4096, 4)
_HINGE_PART = tl.constexpr(10)
_HINGE_STATE = tl.constexpr(6)  # low, high, positive and negative shares, unique columns
# Smooth rows: each chunk also keeps up to _CANDIDATES scores per side within gamma (and a
# little more) of its own least positive or largest negative, a superset of the row's
# entries within gamma of its extremes, where the projections take every step after the
# first superset (_thresholds). The merge runs them in registers; rows with more candidates
# or with infinite extremes take _forward (GENERAL), one program per row.
_CANDIDATES = 16
_SLACK = tl.constexpr(2.0**-20)  # relative widening of the candidates' window
_SMOOTH_STATE = tl.constexpr(8)  # _forward's 7, then the largest negative of an ordinary row


@triton.jit
def _window(gamma):
    return gamma + gamma * _SLACK


@triton.jit(do_not_specialize=["GAMMA"])
def _hinge_partial(Z, Y, P, Q, GAMMA, C: tl.constexpr, B: tl.constexpr, CHUNKS: tl.constexpr,
                   SMOOTH: tl.constexpr = False, K: tl.constexpr = 16):  # fmt: skip
    """A chunk's least finite positive and largest finite negative (masked classes are no
    negatives), their counts and first columns, its NaN or +inf scores, positives and
    masked positives; SMOOTH: its first positive and its candidates (Q: K float32 scores per
    side, then their two int32 counters)."""
    row, block = rows.place(CHUNKS)
    shift, shared = rows.mask_shift(Z, Y, row, C, C)
    j, k, valid, inside = rows.span(block, shift, shared, B, C)
    VEC: tl.constexpr = 128 // Z.dtype.element_ty.primitive_bitwidth
    if inside:
        z = rows.load(Z, row, j, shift, valid, C, False)
        pos = rows.load_mask(Y, row, j, shift, valid, VEC, C, False)
    else:
        z = rows.load(Z, row, j, shift, valid, C, True)
        pos = rows.load_mask(Y, row, j, shift, valid, VEC, C, True)
    finite = valid & (z < float("inf"))
    lo = tl.min(tl.where(pos & finite, z, float("inf")), 1)
    hi = tl.max(tl.where(finite & ~pos, z, -float("inf")), 1)
    at_lo = pos & (z == lo[:, None])
    at_hi = valid & ~pos & (z == hi[:, None])
    out = doubles(P) + (row * CHUNKS + block) * _HINGE_PART + tl.arange(0, 1)
    tl.store(out, lo.to(tl.float64))
    tl.store(out + 1, tl.sum(at_lo.to(tl.int32), 1).to(tl.float64))
    tl.store(out + 2, tl.min(tl.where(at_lo, k, C), 1).to(tl.float64))
    tl.store(out + 3, hi.to(tl.float64))
    tl.store(out + 4, tl.sum(at_hi.to(tl.int32), 1).to(tl.float64))
    tl.store(out + 5, tl.min(tl.where(at_hi, k, C), 1).to(tl.float64))
    tl.store(out + 6, tl.sum((valid & ~finite).to(tl.int32), 1).to(tl.float64))
    count = tl.sum(pos.to(tl.int32), 1) + 65536 * tl.sum(
        (pos & (z == -float("inf"))).to(tl.int32), 1
    )
    tl.store(out + 7, count.to(tl.float64))
    if SMOOTH:
        first = tl.min(tl.where(pos, k, C), 1)
        # Float32 differences of float32 scores round within the window's widening.
        window = _window(f64(GAMMA)).to(tl.float32)
        z = z.to(tl.float32)
        near_a = pos & finite & (z - lo.to(tl.float32)[:, None] <= window)
        near_b = finite & ~pos & (hi.to(tl.float32)[:, None] - z <= window)
        near_count = tl.sum(near_a.to(tl.int32) + 65536 * near_b.to(tl.int32), 1)
    if SMOOTH:
        tl.store(out + 8, first.to(tl.float64))  # pyright: ignore[reportPossiblyUnboundVariable] -- set under SMOOTH above
        # The few candidates take slots from two counters (atomics on their lanes only).
        base = Q + (row * CHUNKS + block) * (2 * K + 2)
        counters = base.to(tl.pointer_type(tl.int32), bitcast=True) + 2 * K
        tl.store(counters + tl.arange(0, 2), tl.zeros([2], tl.int32))
        tl.debug_barrier()
        one = tl.full(z.shape, 1, tl.int32)
        zero = tl.zeros(z.shape, tl.int32)
        slot_a = tl.atomic_add(counters + zero, one, mask=near_a, sem="relaxed")  # pyright: ignore[reportPossiblyUnboundVariable] -- set under SMOOTH above
        slot_b = tl.atomic_add(counters + 1 + zero, one, mask=near_b, sem="relaxed")  # pyright: ignore[reportPossiblyUnboundVariable] -- set under SMOOTH above
        tl.store(base + slot_a, z, near_a & (slot_a < K))  # pyright: ignore[reportPossiblyUnboundVariable] -- set under SMOOTH above
        tl.store(base + K + slot_b, z, near_b & (slot_b < K))  # pyright: ignore[reportPossiblyUnboundVariable] -- set under SMOOTH above
        tl.store(out + 9, near_count.to(tl.float64))  # pyright: ignore[reportPossiblyUnboundVariable] -- set under SMOOTH above


@triton.jit(do_not_specialize=["N"])
def _hinge_merge(P, V, S, N, C: tl.constexpr, CHUNKS: tl.constexpr, CP: tl.constexpr,
                 VALUES: tl.constexpr, GRAD: tl.constexpr):  # fmt: skip
    """_forward's nonsmooth value and state of a row from its chunks, with the columns of a
    unique least positive and largest negative (else -1)."""
    row = tl.program_id(0).to(tl.int64)
    S = doubles(S)
    c = tl.arange(0, CP)
    here = c < CHUNKS
    part = doubles(P) + (row * CHUNKS + c) * _HINGE_PART
    lo = tl.load(part, here, other=float("inf"))
    hi = tl.load(part + 3, here, other=-float("inf"))
    low = tl.min(lo, 0)
    high = tl.max(hi, 0)
    n_low = tl.sum(tl.where(here & (lo == low), tl.load(part + 1, here, other=0.0), 0.0), 0)
    n_high = tl.sum(tl.where(here & (hi == high), tl.load(part + 4, here, other=0.0), 0.0), 0)
    i_low = tl.min(tl.where(here & (lo == low), tl.load(part + 2, here, other=C), C), 0)
    i_high = tl.min(tl.where(here & (hi == high), tl.load(part + 5, here, other=C), C), 0)
    nonfinite = tl.sum(tl.load(part + 6, here, other=0.0), 0)
    packed = tl.load(part + 7, here, other=0.0)
    masked_pos = tl.sum(tl.floor(packed / 65536), 0)
    count = tl.sum(packed - 65536 * tl.floor(packed / 65536), 0)
    all_masked = (high == -float("inf")) & (masked_pos == count)
    bad = (nonfinite > 0) | all_masked | (count == 0) | (count == C)
    h = tl.where(high == -float("inf"), -float("inf"), margin(high, low))
    value = tl.maximum(h, 0.0)
    if GRAD:
        slope = tl.where(h == 0, 0.5, (h > 0).to(tl.float64))
        unique = (n_low == 1) & (n_high == 1) & ~bad
        tl.store(S + row * _HINGE_STATE, low)
        tl.store(S + row * _HINGE_STATE + 1, high)
        tl.store(S + row * _HINGE_STATE + 2, -slope / tl.maximum(n_low, 1.0))
        tl.store(
            S + row * _HINGE_STATE + 3, tl.where(bad, float("nan"), slope / tl.maximum(n_high, 1.0))
        )
        tl.store(S + row * _HINGE_STATE + 4, tl.where(unique, i_low, -1.0))
        tl.store(S + row * _HINGE_STATE + 5, tl.where(unique, i_high, -1.0))
    if VALUES:
        tl.store(V + row, tl.where(bad, float("nan"), value).to(V.dtype.element_ty))


@triton.jit(do_not_specialize=["N", "GAMMA"])
def _smooth_merge(Z, P, Q, V, S, F, N, GAMMA, C: tl.constexpr, CHUNKS: tl.constexpr,
                  CP: tl.constexpr, K: tl.constexpr, VALUES: tl.constexpr, GRAD: tl.constexpr):  # fmt: skip
    """_forward's smooth value and state of an ordinary row from its chunks' extremes and
    candidates; F marks the rows _forward (GENERAL) computes instead."""
    row = tl.program_id(0).to(tl.int64)
    S = doubles(S)
    gamma = f64(GAMMA)
    c = tl.arange(0, CP)
    here = c < CHUNKS
    part = doubles(P) + (row * CHUNKS + c) * _HINGE_PART
    low = tl.min(tl.load(part, here, other=float("inf")), 0)
    high = tl.max(tl.load(part + 3, here, other=-float("inf")), 0)
    nonfinite = tl.sum(tl.load(part + 6, here, other=0.0), 0)
    packed = tl.load(part + 7, here, other=0.0)
    count = tl.sum(packed - 65536 * tl.floor(packed / 65536), 0)
    first = tl.min(tl.load(part + 8, here, other=C), 0).to(tl.int64)
    found = tl.load(part + 9, here, other=0.0)
    over = tl.max(tl.maximum(found - 65536 * tl.floor(found / 65536), tl.floor(found / 65536)), 0)
    bad = (nonfinite > 0) | (count == 0) | (count == C)
    z0 = tl.load(Z + row * C + tl.minimum(first, C - 1)).to(tl.float64)
    # Margins round to 2**-48 of these magnitudes: within the window's widening.
    reach = 2.0**-48 * (1.0 + tl.abs(z0) + tl.abs(low) + tl.abs(high))
    general = ~bad & ((over > K) | ~finite(low) | ~finite(high) | (reach > gamma * _SLACK))
    tl.store(doubles(F) + row, general.to(tl.float64))
    if general:
        return
    recenter = tl.abs((high - low) + 1) < 0.0625 * (1 + tl.abs(low - z0) + tl.abs(high - z0))
    pair = margin(high, low)
    alone = pair <= 0
    separate = pair >= 2 * gamma
    recenter = tl.where(separate, 2, recenter.to(tl.int32))
    z0 = tl.where(separate, high, z0)
    T: tl.constexpr = CP * 2 * K
    i = tl.arange(0, T)
    chunk = i // (2 * K)
    side = (i // K) % 2
    slot = i % K
    live = chunk < CHUNKS
    got = tl.load(doubles(P) + (row * CHUNKS + chunk) * _HINGE_PART + 9, live, other=0.0)
    n = tl.where(side == 0, got - 65536 * tl.floor(got / 65536), tl.floor(got / 65536))
    v1 = live & (slot < n)
    z1 = tl.load(Q + (row * CHUNKS + chunk) * (2 * K + 2) + side * K + slot, v1, other=0.0)
    z1 = z1.to(tl.float64)
    p1 = v1 & (side == 0)
    h1 = _smooth_margins(z1, p1, z0, low, recenter)
    j = tl.arange(0, T)
    oa, ca, ob, cb = _thresholds(
        Z, Z, row, j, p1, v1, h1, z0, low, recenter, bad | alone, gamma, T, T
    )
    oa = tl.where(alone, float("inf"), oa)
    ca = tl.where(alone, 0.0, ca)
    ob = tl.where(alone, float("inf"), ob)
    cb = tl.where(alone, 0.0, cb)
    value = smooth_terms(h1, v1, _projection(h1, p1, oa, ca, ob, cb, gamma), gamma, 0)
    value = tl.where(alone, 0.0, value + tl.where(separate, pair, 0.0))
    nan = float("nan")
    if GRAD:
        state = S + row * _SMOOTH_STATE
        tl.store(state, z0)
        tl.store(state + 1, low)
        tl.store(state + 2, recenter.to(tl.float64))
        tl.store(state + 3, oa)
        tl.store(state + 4, ca)
        tl.store(state + 5, ob)
        tl.store(state + 6, tl.where(bad, nan, cb))
        tl.store(state + 7, tl.where(bad | alone, nan, high))
    if VALUES:
        tl.store(V + row, tl.where(bad, nan, value).to(V.dtype.element_ty))


@triton.jit
def _smooth_chunk(Z, M, G, row, j, k, shift, valid, state, s, gamma, lean_a, lean_b,
                  C: tl.constexpr, LEAN: tl.constexpr, MASKED: tl.constexpr):  # fmt: skip
    """LEAN: x = clip(lean_a - z, 0, gamma) on positives and clip(z - lean_b, 0, gamma) on
    negatives in float32 (an ordinary row of 16-bit storage), else _backward's float64."""
    z = rows.load(Z, row, j, shift, valid, C, MASKED)
    VEC: tl.constexpr = 128 // Z.dtype.element_ty.primitive_bitwidth
    pos = rows.load_mask(M, row, j, shift, valid, VEC, C, MASKED)
    if LEAN:
        zf = z.to(tl.float32)
        g32 = gamma.to(tl.float32)
        x = tl.where(pos, lean_a.to(tl.float32)[:, None] - zf, zf - lean_b.to(tl.float32)[:, None])
        x = tl.minimum(tl.maximum(x, 0.0), g32)
        g = tl.where(pos, -x, x) * (s / gamma).to(tl.float32)[:, None] + 0.0
    else:
        zz = z.to(tl.float64)
        h = _smooth_margins(zz, pos, tl.load(state)[:, None], tl.load(state + 1)[:, None],
                            tl.load(state + 2).to(tl.int32)[:, None])  # fmt: skip
        cb = tl.load(state + 6)[:, None]
        x = _projection(h, pos, tl.load(state + 3)[:, None], tl.load(state + 4)[:, None],
                        tl.load(state + 5)[:, None], cb, gamma) / gamma  # fmt: skip
        g = tl.where(cb != cb, float("nan"), tl.where(pos, -x, x) * s[:, None]) + 0.0
    rows.store(G, row, j, shift, valid, g, C, MASKED)


@triton.jit(do_not_specialize=["N", "UP", "GAMMA"])
def _smooth_backward(Z, Y, R, D, S, G, N, UP, GAMMA, C: tl.constexpr, B: tl.constexpr,
                     CHUNKS: tl.constexpr, ROWWISE: tl.constexpr, DIVIDED: tl.constexpr,
                     LEAN: tl.constexpr, ALIAS: tl.constexpr = False):  # fmt: skip
    """_backward's smooth gradient over aligned chunks of the scores. An ordinary row of
    16-bit storage (LEAN) projects in float32 from two row thresholds, as accurate as the
    storage (within 2**-12 gamma); other rows recompute the float64 margins."""
    if ALIAS:
        Z = G
    S = doubles(S)
    row, block = rows.place(CHUNKS)
    one = row + tl.arange(0, 1)
    state = S + one * _SMOOTH_STATE
    gamma = f64(GAMMA)
    s = scale(UP, R, D, one, one >= 0, ROWWISE, DIVIDED)
    z0 = tl.load(state)
    low = tl.load(state + 1)
    recenter = tl.load(state + 2)
    oa = tl.load(state + 3) + tl.load(state + 4)
    ob = tl.load(state + 5) + tl.load(state + 6)
    # x = h - o - c in each margin form (_smooth_margins): clip(a - z) on the positives and
    # clip(z - b) on the negatives.
    lean_a = tl.where(recenter == 0, 0.5 + z0, low) - oa
    lean_b = tl.where(recenter == 0, z0 - 0.5, tl.where(recenter == 1, low - 1, z0)) + ob
    reach = tl.abs(tl.where(oa < float("inf"), lean_a, 0.0)) + tl.abs(
        tl.where(ob < float("inf"), lean_b, 0.0)
    )
    lean = (tl.load(state + 7) == tl.load(state + 7)) & (reach <= gamma * 4096.0)
    shift, shared = rows.shared_shift(Z, G, row, C, C)
    _, aligned = rows.mask_shift(Z, Y, row, C, C)
    shared &= aligned
    shift = tl.where(shared, shift, 0)
    j, k, valid, inside = rows.span(block, shift, shared, B, C)
    if LEAN and tl.reshape(lean, ()):
        if inside:
            _smooth_chunk(Z, Y, G, row, j, k, shift, valid, state, s, gamma, lean_a, lean_b,
                          C, True, False)  # fmt: skip
        else:
            _smooth_chunk(Z, Y, G, row, j, k, shift, valid, state, s, gamma, lean_a, lean_b,
                          C, True, True)  # fmt: skip
    else:
        if inside:
            _smooth_chunk(Z, Y, G, row, j, k, shift, valid, state, s, gamma, lean_a, lean_b,
                          C, False, False)  # fmt: skip
        else:
            _smooth_chunk(Z, Y, G, row, j, k, shift, valid, state, s, gamma, lean_a, lean_b,
                          C, False, True)  # fmt: skip


@triton.jit
def _hinge_chunk(Z, M, G, row, j, k, shift, valid, low, high, wp, wn, i_low, i_high, s,
                 C: tl.constexpr, TIES: tl.constexpr, MASKED: tl.constexpr):  # fmt: skip
    if TIES:
        z = rows.load(Z, row, j, shift, valid, C, MASKED).to(tl.float32)
        VEC: tl.constexpr = 128 // Z.dtype.element_ty.primitive_bitwidth
        pos = rows.load_mask(M, row, j, shift, valid, VEC, C, MASKED)
        at_low = pos & (z == low.to(tl.float32)[:, None])
        at_high = valid & ~pos & (z == high.to(tl.float32)[:, None])
    else:
        kk = k.to(tl.float64)
        at_low = kk == i_low[:, None]
        at_high = kk == i_high[:, None]
    g = tl.where(at_high, wn[:, None], tl.where(at_low, wp[:, None], 0.0)) * s[:, None]
    g = tl.where(wn[:, None] != wn[:, None], float("nan"), g) + 0.0
    rows.store(G, row, j, shift, valid, g, C, MASKED)


@triton.jit(do_not_specialize=["N", "UP"])
def _hinge_write(Z, Y, R, D, S, G, N, UP, C: tl.constexpr, B: tl.constexpr,
                 CHUNKS: tl.constexpr, ROWWISE: tl.constexpr, DIVIDED: tl.constexpr,
                 ALIAS: tl.constexpr = False):  # fmt: skip
    """The nonsmooth gradient over aligned chunks: a row whose extremes are unique (or an
    invalid row: NaN) is written without reading the scores; tied extremes share their side's
    slope (G may be Z: each chunk is read first)."""
    if ALIAS:
        Z = G
    S = doubles(S)
    row, block = rows.place(CHUNKS)
    one = row + tl.arange(0, 1)
    state = S + one * _HINGE_STATE
    low = tl.load(state)
    high = tl.load(state + 1)
    wp = tl.load(state + 2)
    wn = tl.load(state + 3)
    i_low = tl.load(state + 4)
    i_high = tl.load(state + 5)
    s = scale(UP, R, D, one, one >= 0, ROWWISE, DIVIDED)
    shift, shared = rows.shared_shift(Z, G, row, C, C)
    _, aligned = rows.mask_shift(Z, Y, row, C, C)
    shared &= aligned
    shift = tl.where(shared, shift, 0)
    j, k, valid, inside = rows.span(block, shift, shared, B, C)
    if tl.reshape((i_low < 0) & (wn == wn), ()):
        if inside:
            _hinge_chunk(Z, Y, G, row, j, k, shift, valid, low, high, wp, wn, i_low, i_high, s,
                         C, True, False)  # fmt: skip
        else:
            _hinge_chunk(Z, Y, G, row, j, k, shift, valid, low, high, wp, wn, i_low, i_high, s,
                         C, True, True)  # fmt: skip
    else:
        if inside:
            _hinge_chunk(Z, Y, G, row, j, k, shift, valid, low, high, wp, wn, i_low, i_high, s,
                         C, False, False)  # fmt: skip
        else:
            _hinge_chunk(Z, Y, G, row, j, k, shift, valid, low, high, wp, wn, i_low, i_high, s,
                         C, False, True)  # fmt: skip


def _chunked(call: Call) -> bool:
    """Wide rows of storage that float64 holds exactly in float32-sized chunk words."""
    if call.gamma > 0 and not 2.0**-60 <= call.gamma <= 2.0**60:
        return False
    return call.c > _CHUNK and call.dtype != "float64"


def _chunked_plan(b: Builder, args: dict, meta: dict) -> common.Plan:
    call = b.call
    n, c = call.n, call.c
    smooth = call.gamma > 0
    stats = call.grad or call.saved == common.SAVE
    if call.saved != common.SAVED:
        width, warps = _PARTS
        chunks = row_chunks(c, width)
        partial = b.array("P", (n, chunks, 10), "float64")
        cand = b.array("Q", (n, chunks, 2 * _CANDIDATES + 2), "float32") if smooth else args["Z"]
        part = dict(args, P=partial, Q=cand)
        b.launch(_hinge_partial, (n * chunks,), part, warps, C=c, B=width, CHUNKS=chunks,
                 SMOOTH=smooth, K=_CANDIDATES)  # fmt: skip
        if smooth:
            flags = b.array("F", (n,), "float64")
            b.launch(_smooth_merge, (n,), dict(part, F=flags), 4, C=c, CHUNKS=chunks,
                     CP=pow2(chunks), K=_CANDIDATES, VALUES=call.values, GRAD=stats)  # fmt: skip
            shared = tuple(k for k, on in (("V", call.values), ("S", stats)) if on)
            b.launch(_forward, (n,), dict(args, F=flags), min(8, max(4, _CHUNK // 256)),
                     state=shared, B=_CHUNK, GENERAL=True, STRIDE=8,
                     **dict(meta, GRAD=stats))  # fmt: skip
        else:
            b.launch(_hinge_merge, (n,), part, 1, C=c, CHUNKS=chunks, CP=pow2(chunks),
                     VALUES=call.values, GRAD=stats)  # fmt: skip
    if call.grad:
        width, warps = _WRITE
        chunks = row_chunks(c, width)
        if smooth:
            b.launch(_smooth_backward, (n * chunks,), args, warps, B=width, CHUNKS=chunks,
                     LEAN=call.dtype != "float32", **meta)  # fmt: skip
        else:
            b.launch(_hinge_write, (n * chunks,), args, warps, B=width, CHUNKS=chunks, **meta)
    return b.plan()


def covers(call: Call) -> str | None:
    if common.cdiv(call.c, _BACKWARD) >= 65536:
        return f"the multilabel kernels take at most {65535 * _BACKWARD} classes"
    return None


def plan(call: Call) -> common.Plan:
    """V and G of multilabel_hinge."""
    n, c = call.n, call.c
    smooth = call.gamma > 0
    b = Builder(call)
    args = dict(b.inputs(), **b.results(), GAMMA=bits(call.gamma))
    args["F"] = args["Z"]  # _forward's general-row marks, where a launch reads them
    stats = call.grad or call.saved == common.SAVE
    meta = dict(b.flags(), SMOOTH=smooth)
    if _chunked(call):
        args["S"] = b.stats(8 if smooth else 6) if stats else args["Z"]
        return _chunked_plan(b, args, meta)
    args["S"] = b.stats(7 if smooth else 4) if stats else args["Z"]
    block = min(_CHUNK, pow2(c))
    if call.saved != common.SAVED:
        b.launch(_forward, (n,), args, min(8, max(4, block // 256)), B=block,
                 **dict(meta, GRAD=stats))  # fmt: skip
    if call.grad:
        block = min(_BACKWARD, max(128, pow2(c)))
        b.launch(_backward, (n, common.cdiv(c, block)), args, 4, B=block, **meta)
    return b.plan()
