"""Fused CUDA integer-target top-k losses for 1 < k <= MAX_K, in float64 as include/sdca/loss.h.

One program streams a row in chunks of B columns. It keeps the k (or k + 1) largest
margins, or scores for entropy_topk, in a short vector merged chunk by chunk, then solves
the row from those: the capped entropy thresholds, the top-k cone or simplex, or the
capped knapsack, each from a search over the at most k capped entries and Michelot's
method for the free ones. A row sum whose terms vanish at the short vector's last entry
sums the short vector; others pass over the row again. FP32 takes the entropy's
elementwise exponentials in float32; row statistics stay in float64.
"""

import functools
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from . import _cuda_launch, _triton_reduce
from ._triton_hinge import _gamma, _margin, _terms

LOSSES = {
    "topk_cross_entropy": 0,
    "topk_hinge_alpha": 1,
    "topk_hinge_beta": 2,
    "topk_hinge_alpha_smooth": 3,
    "topk_hinge_beta_smooth": 4,
}
# The largest k of the kernels: they keep k + 1 scores per row.
MAX_K = 32
_REDUCTIONS = {"none": 0, "mean": 1, "sum": 2}
# Row values, two statistics per row and invalid flags, then the status word: the k = 1
# hinge layout, so the C++ hinge launch runs these kernels too.
_LAYOUT = (0, -1, 3, 4)
_P, _I = _cuda_launch.POINTER, _cuda_launch.INT32
# Row chunk and the columns per warp: short chunks and few warps keep more rows in flight
# (an H100 sweep of 256 to 4096 columns at 32000 classes).
_CHUNK = 512
_WARP_COLUMNS = 128
# Double's epsilon (jit functions read only constexpr globals).
_EPS: tl.constexpr = tl.constexpr(2.220446049250313e-16)


@triton.jit
def _at(top, kk, i):
    """Entry i of a short vector."""
    return tl.max(tl.where(kk == i, top, -float("inf")), 0)


@triton.jit
def _softplus(x):
    return tl.maximum(x, 0.0) + libdevice.log1p(libdevice.exp(-tl.abs(x)))


@triton.jit
def _target(Z, Y, row, C: tl.constexpr):
    """Row offset, clamped target, its score in float64 and the target's invalid flag."""
    offset = row.to(tl.int64) * C
    y = tl.load(Y + row)
    in_range = (y >= 0) & (y < C)
    y = tl.minimum(tl.maximum(y, 0), C - 1)
    label = tl.load(Z + offset + y).to(tl.float64)
    return offset, y, label, (~in_range) | ((label - label) != 0)


@triton.jit
def _load(Z, offset, y, k, C: tl.constexpr, FP32: tl.constexpr):
    """Other scores of columns k (nonfinite ones +inf, -inf elsewhere) and the other mask."""
    valid = k < C
    z = tl.load(Z + offset + k, valid, other=0).to(tl.float32 if FP32 else tl.float64)
    other = valid & (k != y)
    return tl.where(other, tl.where((z - z) == 0, z, float("inf")), -float("inf")), other


@triton.jit
def _margins(Z, offset, y, label, k, C: tl.constexpr):
    """Other scores of columns k, the other mask and their largest margin.

    Margins grow with scores: a chunk whose largest margin adds no term adds none.
    """
    x, other = _load(Z, offset, y, k, C, Z.dtype.element_ty != tl.float64)
    return x, other, _margin(tl.max(x, 0).to(tl.float64), label)


@triton.jit
def _merge(val, idx, filled, least, x, start, j, evicted, K1: tl.constexpr, TAIL: tl.constexpr):
    """Slots (val, idx) with the entries of x above least, taken largest and leftmost first.

    The first K1 slots hold the largest entries in the stable order (descending, earlier
    columns first), the first filled of them taken; least is their smallest (-inf until
    all are). Returns the slots, filled, least, the taken mask and, for TAIL, evicted plus
    the evicted entries' exp(value - least), relative to least.
    """
    kk = tl.arange(0, val.shape[0])
    live = kk < K1
    taken = j < 0
    best, at = tl.max(x, 0, return_indices=True, return_indices_tie_break_left=True)
    while best > least:
        if filled < K1:
            slot = kk == filled
        else:
            # The last slot in the stable order: the latest column at the least value.
            latest = tl.max(tl.where(live & (val == least), idx, -(2**31)), 0)
            slot = live & (val == least) & (idx == latest)
        filled = tl.minimum(filled + 1, K1)
        val = tl.where(slot, best, val)
        idx = tl.where(slot, start + at, idx)
        taken = taken | (j == at)
        new = least
        if filled == K1:
            new = tl.min(tl.where(live, val, float("inf")), 0)
        if TAIL:
            if least > -float("inf"):
                shift = least.to(tl.float64) - new.to(tl.float64)
                evicted = (evicted + 1.0) * libdevice.exp(shift)
        least = new
        best, at = tl.max(
            tl.where(taken, -float("inf"), x),
            0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
    return val, idx, filled, least, taken, evicted


@triton.jit
def _sorted(val, idx, KB: tl.constexpr):
    """Slots in the stable order (-inf pads last): float64 values and their columns."""
    kk = tl.arange(0, KB)
    before = (val[None, :] > val[:, None]) | (
        (val[None, :] == val[:, None]) & (idx[None, :] < idx[:, None])
    )
    rank = tl.sum(before.to(tl.int32), 1)
    hit = rank[:, None] == kk[None, :]
    top = tl.max(tl.where(hit, val[:, None], -float("inf")), 0).to(tl.float64)
    return top, tl.max(tl.where(hit, idx[:, None], -1), 0)


@triton.jit
def _kth(lanes, K1: tl.constexpr):
    """The K1-th largest of lanes, ties repeated (-inf if fewer)."""
    value = tl.max(lanes, 0)
    count = tl.sum((lanes == value).to(tl.int32), 0)
    # Each pass takes the next distinct value; -inf at the target ends it.
    while count < K1:
        value = tl.max(tl.where(lanes < value, lanes, -float("inf")), 0)
        count += tl.sum((lanes == value).to(tl.int32), 0)
    return value


@triton.jit
def _margin_lanes(Z, offset, y, label, cap, C: tl.constexpr, B: tl.constexpr, BOX: tl.constexpr):
    """Largest other score of each lane j (columns j mod B) and, for BOX, the sum of
    clip(h, 0, cap) over the others."""
    j = tl.arange(0, B)
    lanes = tl.full(
        [B], -float("inf"), tl.float32 if Z.dtype.element_ty != tl.float64 else tl.float64
    )
    box = tl.zeros([B], tl.float64)
    for start in range(0, C, B):
        x, other = _load(Z, offset, y, start + j, C, Z.dtype.element_ty != tl.float64)
        lanes = tl.maximum(lanes, x)
        if BOX:
            h = _margin(x.to(tl.float64), label)
            box += tl.where(other, tl.minimum(tl.maximum(h, 0.0), cap), 0.0)
    return lanes, tl.sum(box, 0)


@triton.jit
def _margin_top(
    Z, offset, y, label, lanes, C: tl.constexpr, B: tl.constexpr, K1: tl.constexpr, KB: tl.constexpr
):
    """The K1 largest margins in the stable order (-inf pads) and their columns.

    K1 lanes hold a score at least the K1-th lane maximum, so only margins at least its
    margin are candidates (margins grow with scores).
    """
    j = tl.arange(0, B)
    kk = tl.arange(0, KB)
    bound = _margin(_kth(lanes, K1).to(tl.float64), label)
    val = tl.full([KB], -float("inf"), tl.float64)
    idx = -1 - kk
    filled = tl.zeros([], tl.int32)
    least = tl.full([], -float("inf"), tl.float64)
    evicted = tl.zeros([], tl.float64)
    for start in range(0, C, B):
        x, other = _load(Z, offset, y, start + j, C, Z.dtype.element_ty != tl.float64)
        m = _margin(tl.max(x, 0).to(tl.float64), label)
        if (m >= bound) & (m > least):
            h = _margin(x.to(tl.float64), label)
            h = tl.where(h >= bound, h, -float("inf"))
            val, idx, filled, least, _, evicted = _merge(
                val, idx, filled, least, h, start, j, evicted, K1, False
            )
    return _sorted(val, idx, KB)


@triton.jit
def _above(Z, offset, y, label, level, top, live, last, C: tl.constexpr, B: tl.constexpr):
    """Sum of max(h - level, 0) over the others."""
    if tl.maximum(last - level, 0.0) > 0:
        j = tl.arange(0, B)
        acc = tl.zeros([B], tl.float64)
        for start in range(0, C, B):
            x, other, m = _margins(Z, offset, y, label, start + j, C)
            if m > level:
                h = _margin(x.to(tl.float64), label)
                acc += tl.where(other, tl.maximum(h - level, 0.0), 0.0)
        total = tl.sum(acc, 0)
    else:
        total = tl.sum(tl.where(live, tl.maximum(top - level, 0.0), 0.0), 0)
    return total


@triton.jit
def _clipped(Z, offset, y, label, level, cap, top, live, last, C: tl.constexpr, B: tl.constexpr):
    """Sum of clip((h - level) + cap, 0, cap) over the others."""
    if tl.minimum(tl.maximum((last - level) + cap, 0.0), cap) > 0:
        j = tl.arange(0, B)
        acc = tl.zeros([B], tl.float64)
        for start in range(0, C, B):
            x, other, m = _margins(Z, offset, y, label, start + j, C)
            if (m - level) + cap > 0:
                h = _margin(x.to(tl.float64), label)
                acc += tl.where(other, tl.minimum(tl.maximum((h - level) + cap, 0.0), cap), 0.0)
        total = tl.sum(acc, 0)
    else:
        total = tl.sum(
            tl.where(live, tl.minimum(tl.maximum((top - level) + cap, 0.0), cap), 0.0), 0
        )
    return total


@triton.jit
def _free(
    Z, offset, y, label, bound, origin, level, top, live, last, C: tl.constexpr, B: tl.constexpr
):
    """Count, sum and least of g = h - origin over the others with h <= bound and g > level."""
    if (last - origin) > level:
        j = tl.arange(0, B)
        count = tl.zeros([B], tl.int32)
        total = tl.zeros([B], tl.float64)
        least = tl.full([B], float("inf"), tl.float64)
        for start in range(0, C, B):
            x, other, m = _margins(Z, offset, y, label, start + j, C)
            if (m - origin) > level:
                h = _margin(x.to(tl.float64), label)
                g = h - origin
                inside = other & (h <= bound) & (g > level)
                count += inside.to(tl.int32)
                total += tl.where(inside, g, 0.0)
                least = tl.minimum(least, tl.where(inside, g, float("inf")))
        n = tl.sum(count, 0)
        s = tl.sum(total, 0)
        m = tl.min(least, 0)
    else:
        gap = top - origin
        kept = live & (top <= bound) & (gap > level)
        n = tl.sum(kept.to(tl.int32), 0)
        s = tl.sum(tl.where(kept, gap, 0.0), 0)
        m = tl.min(tl.where(kept, gap, float("inf")), 0)
    return n, s, m


@triton.jit
def _knapsack(
    Z,
    offset,
    y,
    label,
    top,
    kk,
    live,
    last,
    cap,
    rhs,
    C: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
):
    """Origin and correction of a = clip((h - origin) - correction, 0, cap) with sum(a) = rhs.

    Bisection on the clipped mass at t = top[j - 1] - cap counts the u <= K largest at the
    cap (ties never split); Michelot's method from the superset {h > top[u] - cap} of the
    free entries then gives the correction relative to the largest free margin.
    """
    lo = 0
    hi = K
    while lo < hi:
        mid = (lo + hi + 1) // 2
        level = _at(top, kk, mid - 1)
        ok = _clipped(Z, offset, y, label, level, cap, top, live, last, C, B) <= rhs
        lo = tl.where(ok, mid, lo)
        hi = tl.where(ok, hi, mid - 1)
    origin = _at(top, kk, tl.minimum(lo, K - 1))
    rem = rhs - cap * lo
    # With all K at the cap no entry is free: t = top[K - 1] - cap. Either correction is
    # <= 0 (free margins lie at or below origin, rem > 0), which _projected relies on.
    s = -cap
    if lo < K:
        level = -cap
        count, total, least = _free(
            Z, offset, y, label, origin, origin, level, top, live, last, C, B
        )
        s = (total - rem) / count
        level = tl.maximum(level, s)
        # The set {g > level} only shrinks, so rounding cannot cycle (_triton_hinge._threshold);
        # it is stable once its least gap stays above the level.
        while least <= level:
            count, total, least = _free(
                Z, offset, y, label, origin, origin, level, top, live, last, C, B
            )
            s = (total - rem) / count
            level = tl.maximum(level, s)
    return origin, s


@triton.jit
def _projected(h, first, second, gamma, K: tl.constexpr):
    """The smooth hinges' a from two row statistics: a positive second is the cone's cap at
    threshold first; else first and second (<= 0) are a knapsack's origin and correction."""
    cone = second > 0
    shifted = tl.where(cone, h - first, (h - first) - second)
    return tl.minimum(tl.maximum(shifted, 0.0), tl.where(cone, second, gamma / K))


@triton.jit
def _cone(
    Z, offset, y, label, top, kk, live, last, C: tl.constexpr, B: tl.constexpr, K: tl.constexpr
):
    """Threshold t and cap c of the projection clip(h - t, 0, c) onto 0 <= x_i <= sum(x)/K.

    As _tensor_losses._topk_cone: exactly the K largest are capped when the rest lie below their
    threshold; otherwise bisection on the stationarity residual counts the u capped, and
    Michelot's method finds the free entries of t = (u rest - (K - u) head) / d.
    """
    # The K largest in double-double, and 16 eps relative to scores below one.
    whole = tl.sum(tl.where(kk < 0, top, 0.0), 0)
    low = whole
    scale = whole
    for i in tl.static_range(K):
        x = _at(top, kk, i)
        total = whole + x
        rounded = total - whole
        low += (whole - (total - rounded)) + (x - rounded)
        whole = total
        scale += tl.abs(x)
    whole += low
    tol = 16 * _EPS * tl.minimum(scale, 1.0)
    c = whole / K
    t = _at(top, kk, K - 1) - c
    if (t < _at(top, kk, K) - tol) & (whole > tol):
        # u is one past the last j whose level top[j] lies above the solution's cap level.
        lo = -1
        hi = K - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            level = _at(top, kk, mid)
            shift = -_above(Z, offset, y, label, level, top, live, last, C, B) / K
            ok = _above(Z, offset, y, label, shift, top, live, last, C, B) / K + 2 * shift < level
            lo = tl.where(ok, mid, lo)
            hi = tl.where(ok, hi, mid - 1)
        u = lo + 1
        if u < K:
            head = tl.sum(tl.where(kk < u, top, 0.0), 0)
            bound = _at(top, kk, u)
            # Michelot from all free margins; the set {h > level} only shrinks (as _knapsack).
            level = tl.full([], -float("inf"), tl.float64)
            m, rest, least = _free(Z, offset, y, label, bound, 0.0, level, top, live, last, C, B)
            d = ((K - u) * (K - u) + m * u).to(tl.float64)
            t = (u * rest - (K - u) * head) / d
            level = t
            while least <= level:
                m, rest, least = _free(
                    Z, offset, y, label, bound, 0.0, level, top, live, last, C, B
                )
                d = ((K - u) * (K - u) + m * u).to(tl.float64)
                t = (u * rest - (K - u) * head) / d
                level = tl.maximum(level, t)
            c = (m * head + (K - u) * rest) / d
    zero = whole <= tol
    return tl.where(zero, 0.0, t), tl.where(zero, 0.0, c)


@triton.jit
def _smooth(
    Z,
    offset,
    y,
    label,
    flag,
    gamma,
    C: tl.constexpr,
    B: tl.constexpr,
    LOSS: tl.constexpr,
    K: tl.constexpr,
    K1: tl.constexpr,
    KB: tl.constexpr,
):
    """Value, the two statistics and the flag of a smooth hinge (K1 = K + 1 for the cone)."""
    cap = gamma / K
    lanes, box = _margin_lanes(Z, offset, y, label, cap, C, B, LOSS == 4)
    flag = flag | (tl.max(lanes, 0) == float("inf"))
    kk = tl.arange(0, KB)
    live = kk < K1
    origin = tl.zeros([], tl.float64)
    correction = tl.zeros([], tl.float64)
    value = tl.zeros([], tl.float64)
    if ~flag:
        top = tl.full([KB], -float("inf"), tl.float64)
        if LOSS == 3:
            top, _ = _margin_top(Z, offset, y, label, lanes, C, B, K1, KB)
        elif box > gamma:
            top, _ = _margin_top(Z, offset, y, label, lanes, C, B, K1, KB)
        # Without a projection (box <= gamma) the value pass takes the whole row.
        last = tl.where((LOSS == 3) | (box > gamma), _at(top, kk, K1 - 1), float("inf"))
        if LOSS == 3:
            # The cone's projection unless its sum k c exceeds gamma: then sum(a) = gamma.
            t, c = _cone(Z, offset, y, label, top, kk, live, last, C, B, K)
            if K * c > gamma:
                origin, correction = _knapsack(
                    Z, offset, y, label, top, kk, live, last, cap, gamma, C, B, K
                )
            else:
                # The cap c as the second statistic; a zero cone keeps no margin.
                origin = tl.where(c > 0, t, float("inf"))
                correction = c
        else:
            if box > gamma:
                origin, correction = _knapsack(
                    Z, offset, y, label, top, kk, live, last, cap, gamma, C, B, K
                )
        if _projected(last, origin, correction, gamma, K) > 0:
            j = tl.arange(0, B)
            for start in range(0, C, B):
                x, other, m = _margins(Z, offset, y, label, start + j, C)
                if _projected(m, origin, correction, gamma, K) > 0:
                    h = _margin(x.to(tl.float64), label)
                    a = _projected(h, origin, correction, gamma, K)
                    value += _terms(h, other, a, gamma)
        else:
            value = _terms(top, live, _projected(top, origin, correction, gamma, K), gamma)
    return value, origin, correction, flag


@triton.jit
def _entropy_top(
    Z,
    offset,
    y,
    bound,
    C: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    KB: tl.constexpr,
    FP32: tl.constexpr,
):
    """The K largest other scores in float64 (descending, ties repeated) and the sum of
    exp(z - last) over the rest, last the K-th; the K largest are at least bound."""
    j = tl.arange(0, B)
    kk = tl.arange(0, KB)
    val = tl.full([KB], -float("inf"), tl.float32 if FP32 else tl.float64)
    idx = -1 - kk
    filled = tl.zeros([], tl.int32)
    least = tl.full([], -float("inf"), tl.float32 if FP32 else tl.float64)
    # The tail relative to max(least, bound), which every entry outside the slots lies below.
    ref = bound
    tail = tl.zeros([B], tl.float64)
    evicted = tl.zeros([], tl.float64)
    for start in range(0, C, B):
        x, other = _load(Z, offset, y, start + j, C, FP32)
        taken = j < 0
        candidates = tl.where(x >= bound, x, -float("inf"))
        if tl.max(candidates, 0) > least:
            val, idx, filled, least, taken, evicted = _merge(
                val, idx, filled, least, candidates, start, j, evicted, K, True
            )
            if least > ref:
                tail = tail * libdevice.exp(ref.to(tl.float64) - least.to(tl.float64))
                ref = least
        e = tl.where(taken | (x == -float("inf")), 0.0, libdevice.exp(x - ref))
        tail += e.to(tl.float64)
    top, _ = _sorted(val, idx, KB)
    return top, tl.sum(tail, 0) + evicted


@triton.jit
def _entropy_below(Z, offset, y, free, C: tl.constexpr, B: tl.constexpr, FP32: tl.constexpr):
    """The others equal to free and the sum of exp(z - free) over those below it."""
    j = tl.arange(0, B)
    level = free.to(tl.float32 if FP32 else tl.float64)
    copies = tl.zeros([B], tl.int32)
    below = tl.zeros([B], tl.float64)
    for start in range(0, C, B):
        x, other = _load(Z, offset, y, start + j, C, FP32)
        copies += (x == level).to(tl.int32)
        below += tl.where(x < level, libdevice.exp(x - level), 0.0).to(tl.float64)
    return tl.sum(copies, 0), tl.sum(below, 0)


@triton.jit
def _entropy_rows(
    Z, Y, S, N, C: tl.constexpr, B: tl.constexpr, K: tl.constexpr, KB: tl.constexpr,
    FP32: tl.constexpr, ROWS: tl.constexpr,
):  # fmt: skip
    """Top-k entropy across a [ROWS, B] register tile (B >= C)."""
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    row_valid = row < N
    offset = row * C
    y = tl.load(Y + row, mask=row_valid, other=0)
    in_range = row_valid & (y >= 0) & (y < C)
    y = tl.minimum(tl.maximum(y, 0), C - 1)
    label = tl.load(Z + offset + y, mask=row_valid, other=0).to(tl.float64)
    flag = (~in_range) | ((label - label) != 0)
    j = tl.arange(0, B)[None, :]
    valid = row_valid[:, None] & (j < C)
    z = tl.load(Z + offset[:, None] + j, mask=valid, other=0).to(tl.float32 if FP32 else tl.float64)
    other = valid & (j != y[:, None])
    x = tl.where(other, tl.where((z - z) == 0, z, float("inf")), -float("inf"))
    largest = tl.max(x, 1)
    maximum = largest.to(tl.float64)
    flag = flag | (maximum == float("inf"))
    ties = tl.sum((other & (x == largest[:, None])).to(tl.int32), 1)
    acc = tl.sum(
        tl.where(x < largest[:, None], libdevice.exp(x - largest[:, None]), 0.0).to(tl.float64), 1
    )
    others = acc + (ties - 1).to(tl.float64)
    eps = 32 * _EPS
    lse = libdevice.log1p(others)
    log_k = libdevice.log(tl.full([ROWS], K, tl.float64))
    threshold = lse - log_k
    active = tl.zeros([ROWS], tl.int32)
    head = tl.zeros([ROWS], tl.float64)
    free = maximum
    above = tl.zeros([ROWS], tl.int32)
    spread = others
    need_cap = (threshold + eps * tl.maximum(tl.abs(threshold), 1.0) < 0) & (~flag)
    if tl.max(need_cap.to(tl.int32), 0) > 0:
        kk = tl.arange(0, KB)[None, :]
        top = tl.full([ROWS, KB], -float("inf"), tl.float64)
        cand = x
        last_w = largest
        for i in tl.static_range(K):
            best, at = tl.max(cand, 1, return_indices=True, return_indices_tie_break_left=True)
            top = tl.where(kk == i, best.to(tl.float64)[:, None], top)
            cand = tl.where(j == at[:, None], -float("inf"), cand)
            if i == K - 1:
                last_w = best
        last = last_w.to(tl.float64)
        tail = tl.sum(
            tl.where(cand > -float("inf"), libdevice.exp(cand - last_w[:, None]), 0.0).to(
                tl.float64
            ),
            1,
        )
        running = need_cap
        for i in tl.static_range(1, K):
            cand_free = tl.max(tl.where(kk == i, top, -float("inf")), 1)
            scale = libdevice.exp(last - cand_free)
            ok_tail = (scale > 0) & ((tail - tail) == 0)
            mid = tl.where((kk > i) & (kk < K), libdevice.exp(top - cand_free[:, None]), 0.0)
            capped_ties = tl.where((kk < i) & (top == cand_free[:, None]), 1.0, 0.0)
            cand_lse = libdevice.log1p(tl.sum(mid, 1) + scale * tail)
            cand_spread = tl.sum(mid + capped_ties, 1) + scale * tail
            if tl.max((running & ~ok_tail).to(tl.int32), 0) > 0:
                level = cand_free.to(tl.float32 if FP32 else tl.float64)[:, None]
                copies = tl.sum((x == level).to(tl.int32), 1)
                below = tl.sum(tl.where(x < level, libdevice.exp(x - level), 0.0).to(tl.float64), 1)
                tied = tl.sum(((kk < i) & (top == cand_free[:, None])).to(tl.int32), 1)
                cand_lse = tl.where(
                    ok_tail, cand_lse, libdevice.log1p(below + (copies - tied - 1).to(tl.float64))
                )
                cand_spread = tl.where(ok_tail, cand_spread, below + (copies - 1).to(tl.float64))
            active = tl.where(running, i, active)
            free = tl.where(running, cand_free, free)
            lse = tl.where(running, cand_lse, lse)
            spread = tl.where(running, cand_spread, spread)
            cand_thresh = cand_lse - libdevice.log(tl.full([ROWS], K - i, tl.float64))
            capped = cand_thresh + eps * tl.maximum(tl.abs(cand_thresh), 1.0) < 0
            if i < K - 1:
                running = running & capped
                head = tl.where(running, head + (cand_free - maximum), head)
        above = tl.where(
            need_cap, tl.sum(((kk < K) & (top > free[:, None])).to(tl.int32), 1), above
        )
    remaining = (K - active).to(tl.float64)
    rest = remaining * ((free - maximum) + lse - libdevice.log(remaining))
    gap = maximum - label
    mean = (head + rest) / K
    odds = gap + mean + log_k
    positive = tl.maximum(gap, 0.0)
    half = libdevice.exp((gap - positive) * 0.5)
    capped_scale = libdevice.exp(tl.minimum(mean + positive, 0.0)) * K
    small = half * (half * tl.where(active == 0, 1 + others, capped_scale))
    loss = tl.where(odds < -36, libdevice.log1p(small), _softplus(odds))
    tl.store(S + row, tl.where(flag, float("nan"), loss), mask=row_valid)
    tl.store(S + N + row, free, mask=row_valid)
    tl.store(S + 2 * N + row, (1 + spread) / (K - above).to(tl.float64), mask=row_valid)
    tl.store(S + 3 * N + row, flag.to(tl.float64), mask=row_valid)


@triton.jit
def _entropy_rows_backward(
    Z, Y, S, U, G, N, C: tl.constexpr, B: tl.constexpr, K: tl.constexpr,
    REDUCTION: tl.constexpr, FP32: tl.constexpr, ROWS: tl.constexpr,
):  # fmt: skip
    """_backward of top-k entropy across a [ROWS, B] register tile (B >= C)."""
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    row_valid = row < N
    j = tl.arange(0, B)[None, :]
    valid = row_valid[:, None] & (j < C)
    offset = row * C
    y = tl.load(Y + row, mask=row_valid, other=0)
    y = tl.minimum(tl.maximum(y, 0), C - 1)
    z = tl.load(Z + offset[:, None] + j, mask=valid, other=0).to(tl.float32 if FP32 else tl.float64)
    weight = (
        tl.load(U + row, mask=row_valid, other=0.0).to(tl.float64)
        if REDUCTION == 0
        else tl.load(U).to(tl.float64)
    )
    if REDUCTION == 1:
        weight = weight / N
    flag = (tl.load(S + 3 * N + row, mask=row_valid, other=0.0) != 0)[:, None]
    first = tl.load(S + N + row, mask=row_valid, other=0.0)[:, None]
    second = tl.load(S + 2 * N + row, mask=row_valid, other=0.0)[:, None]
    loss = tl.load(S + row, mask=row_valid, other=0.0)[:, None]
    w = weight[:, None] if REDUCTION == 0 else weight
    mass = -libdevice.expm1(-loss)
    cap = mass / K
    spread = tl.where(second > 0, second, 1.0)
    if FP32:
        sign = tl.where(w < 0, -1.0, 1.0).to(tl.float32)
        factor = (cap / spread * tl.abs(w)).to(tl.float32)
        bound = (cap * tl.abs(w)).to(tl.float32)
        d = z - first.to(tl.float32)
        e = tl.where(d > 0, bound, tl.minimum(libdevice.exp(d) * factor, bound))
        entry = (sign * e).to(tl.float64)
    else:
        log_spread = libdevice.log(spread)
        e = libdevice.exp(tl.minimum(z - first, log_spread))
        entry = cap * tl.minimum(e / spread, 1.0) * w
    target = -(w * mass)
    entry = tl.where(j == y[:, None], target, entry)
    grad = tl.where(flag, float("nan"), entry).to(G.dtype.element_ty)
    tl.store(G + offset[:, None] + j, grad, mask=valid)


@triton.jit
def _entropy(
    Z,
    Y,
    S,
    N,
    row,
    C: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    KB: tl.constexpr,
    FP32: tl.constexpr,
):
    """Top-k entropy as loss.h topk_entropy_loss, with the free scores relative to the first
    free one (as _tensor_losses._entropy_topk): free scores far below a capped one keep their
    differences, which scores relative to the maximum round away."""
    offset, y, label, flag = _target(Z, Y, row, C)
    j = tl.arange(0, B)
    kk = tl.arange(0, KB)
    # The largest other score and the sum of exp(z - maximum) over the others but one copy of it.
    largest = tl.full([], -float("inf"), tl.float32 if FP32 else tl.float64)
    lanes = tl.full([B], -float("inf"), tl.float32 if FP32 else tl.float64)
    acc = tl.zeros([B], tl.float64)
    ties = tl.zeros([B], tl.int32)
    for start in range(0, C, B):
        x, other = _load(Z, offset, y, start + j, C, FP32)
        lanes = tl.maximum(lanes, x)
        new = tl.maximum(largest, tl.max(x, 0))
        if new > largest:
            factor = libdevice.exp(largest.to(tl.float64) - new.to(tl.float64))
            acc = (acc + ties.to(tl.float64)) * factor
            ties = tl.zeros([B], tl.int32)
            largest = new
        acc += tl.where(x < largest, libdevice.exp(x - largest), 0.0).to(tl.float64)
        ties += (other & (x == largest)).to(tl.int32)
    maximum = largest.to(tl.float64)
    flag = flag | (maximum == float("inf"))
    others = tl.sum(acc, 0) + (tl.sum(ties, 0) - 1).to(tl.float64)
    zero = tl.zeros([], tl.float64)
    eps = 32 * _EPS
    # No score capped (the fast check of the thresholds): the odds of entropy.
    lse = libdevice.log1p(others)
    threshold = lse - libdevice.log(zero + K)
    active = tl.zeros([], tl.int32)
    head = zero
    free = maximum
    # The backward's capped count and sum of exp(z - free) less one, ties at free included.
    above = tl.zeros([], tl.int32)
    spread = others
    if (threshold + eps * tl.maximum(tl.abs(threshold), 1.0) < 0) & (~flag):
        top, tail = _entropy_top(Z, offset, y, _kth(lanes, K), C, B, K, KB, FP32)
        last = _at(top, kk, K - 1)
        stop = active * 0
        active += 1
        while stop == 0:
            free = _at(top, kk, active)
            scale = libdevice.exp(last - free)
            # The log-sum-exp of the scores from position active on, relative to free.
            if (scale > 0) & ((tail - tail) == 0):
                mid = tl.where((kk > active) & (kk < K), libdevice.exp(top - free), 0.0)
                lse = libdevice.log1p(tl.sum(mid, 0) + scale * tail)
                capped_ties = tl.where((kk < active) & (top == free), 1.0, 0.0)
                spread = tl.sum(mid + capped_ties, 0) + scale * tail
            else:
                # The entries below free and the later copies of it.
                count, below = _entropy_below(Z, offset, y, free, C, B, FP32)
                tied = tl.sum(((kk < active) & (top == free)).to(tl.int32), 0)
                lse = libdevice.log1p(below + (count - tied - 1).to(tl.float64))
                spread = below + (count - 1).to(tl.float64)
            # The native active-set test relative to free, as _tensor_losses._entropy_topk.
            threshold = lse - libdevice.log((K - active).to(tl.float64))
            capped = threshold + eps * tl.maximum(tl.abs(threshold), 1.0) < 0
            if capped & (active < K - 1):
                head += free - maximum
                active += 1
            else:
                stop += 1
        above = tl.sum(((kk < K) & (top > free)).to(tl.int32), 0)
    remaining = (K - active).to(tl.float64)
    log_k = libdevice.log(zero + K)
    rest = remaining * ((free - maximum) + lse - libdevice.log(remaining))
    gap = maximum - label
    mean = (head + rest) / K
    odds = gap + mean + log_k
    # As _tensor_losses._topk_tail: a tail below exp(-36) from squared half exponentials,
    # rounded once; a positive gap joins the mean. Rows of no capped score are entropy's.
    positive = tl.maximum(gap, 0.0)
    half = libdevice.exp((gap - positive) * 0.5)
    capped_scale = libdevice.exp(tl.minimum(mean + positive, 0.0)) * K
    small = half * (half * tl.where(active == 0, 1 + others, capped_scale))
    loss = tl.where(odds < -36, libdevice.log1p(small), _softplus(odds))
    # The capped map cap * min(1, exp(z - free) / D): the scores above free sit at the cap,
    # the rest share the remaining mass; D is their sum of exp(z - free) per remaining slot
    # (a log-domain shift would round at the magnitude of log(mass)).
    tl.store(S + N + row, free)
    tl.store(S + 2 * N + row, (1 + spread) / (K - above).to(tl.float64))
    tl.store(S + 3 * N + row, flag.to(tl.float64))
    return tl.where(flag, float("nan"), loss)


@triton.jit(do_not_specialize=["N"])
def _forward(
    Z,
    Y,
    S,
    GAMMA,
    N,
    C: tl.constexpr,
    B: tl.constexpr,
    LOSS: tl.constexpr,
    K: tl.constexpr,
    KB: tl.constexpr,
    FP32: tl.constexpr,
    ROWS: tl.constexpr = 1,
):
    if LOSS == 0 and B >= C:
        _entropy_rows(Z, Y, S, N, C, B, K, KB, FP32, ROWS)
    else:
        row = tl.program_id(0)
        if LOSS == 0:
            value = _entropy(Z, Y, S, N, row, C, B, K, KB, FP32)
        else:
            offset, y, label, flag = _target(Z, Y, row, C)
            kk = tl.arange(0, KB)
            if LOSS <= 2:
                lanes, _ = _margin_lanes(Z, offset, y, label, 0.0, C, B, False)
                flag = flag | (tl.max(lanes, 0) == float("inf"))
                top, index = _margin_top(Z, offset, y, label, lanes, C, B, K, KB)
                # As loss.h: the k largest (positive parts) over k, summed in order.
                value = tl.zeros([], tl.float64)
                for i in tl.static_range(K):
                    x = _at(top, kk, i)
                    if LOSS == 2:
                        x = tl.maximum(x, 0.0)
                    value += x / K
                value = tl.maximum(value, 0.0)
                # Ties at the k-th margin take the first columns, as loss.h's stable sort.
                first = _at(top, kk, K - 1)
                second = tl.max(tl.where(kk == K - 1, index, -1), 0).to(tl.float64)
            else:
                gamma = tl.load(GAMMA)
                if LOSS == 3:
                    value, first, second, flag = _smooth(
                        Z, offset, y, label, flag, gamma, C, B, LOSS, K, K + 1, KB
                    )
                else:
                    value, first, second, flag = _smooth(
                        Z, offset, y, label, flag, gamma, C, B, LOSS, K, K, KB
                    )
            tl.store(S + N + row, first)
            tl.store(S + 2 * N + row, second)
            tl.store(S + 3 * N + row, flag.to(tl.float64))
            value = tl.where(flag, float("nan"), value)
        tl.store(S + row, value)


@triton.jit(do_not_specialize=["N"])
def _backward(
    Z,
    Y,
    S,
    U,
    G,
    GAMMA,
    N,
    C: tl.constexpr,
    B: tl.constexpr,
    LOSS: tl.constexpr,
    K: tl.constexpr,
    REDUCTION: tl.constexpr,
    FP32: tl.constexpr,
    ROWS: tl.constexpr = 1,
):
    if LOSS == 0 and B >= C:
        _entropy_rows_backward(Z, Y, S, U, G, N, C, B, K, REDUCTION, FP32, ROWS)
    else:
        row = tl.program_id(0)
        j = tl.arange(0, B)
        if REDUCTION == 0:
            weight = tl.load(U + row).to(tl.float64)
        else:
            weight = tl.load(U).to(tl.float64)
        if REDUCTION == 1:
            weight = weight / N
        offset, y, label, _ = _target(Z, Y, row, C)
        flag = tl.load(S + 3 * N + row) != 0
        first = tl.load(S + N + row)
        second = tl.load(S + 2 * N + row)
        gamma = tl.load(GAMMA)
        loss = tl.load(S + row)
        # Entropy: the mass 1 - exp(-loss) of the loss log(1 + tail) and the cap mass / K;
        # second = D (the hinges' second is no sum).
        mass = -libdevice.expm1(-loss)
        cap = mass / K
        spread = tl.where(second > 0, second, 1.0)
        log_spread = libdevice.log(spread)
        # FP32: |w| cap exp(z - free) / D below the cap |w| cap, which holds above free.
        sign = tl.where(weight < 0, -1.0, 1.0).to(tl.float32)
        factor = (cap / spread * tl.abs(weight)).to(tl.float32)
        bound = (cap * tl.abs(weight)).to(tl.float32)
        # Hinges: a zero loss has a zero subgradient; the chosen entries take weight / K.
        step = (tl.full([], 1.0, tl.float64) / K) * tl.where(loss > 0, weight, 0.0)
        total = tl.zeros([B], tl.float64)
        for start in range(0, C, B):
            k = start + j
            # entry: the unrounded gradient entry, rounded once when stored.
            if LOSS == 0:
                z, other = _load(Z, offset, y, k, C, FP32)
                if FP32:
                    d = z - first.to(tl.float32)
                    e = tl.where(d > 0, bound, tl.minimum(libdevice.exp(d) * factor, bound))
                    entry = (sign * e).to(tl.float64)
                else:
                    e = libdevice.exp(tl.minimum(z - first, log_spread))
                    entry = cap * tl.minimum(e / spread, 1.0) * weight
            else:
                z, other, m = _margins(Z, offset, y, label, k, C)
                entry = tl.zeros([B], tl.float64)
                if LOSS <= 2:
                    if m >= first:
                        h = _margin(z.to(tl.float64), label)
                        chosen = (h > first) | ((h == first) & (k <= second.to(tl.int32)))
                        if LOSS == 2:
                            chosen = chosen & (h > 0)
                        entry = tl.where(chosen, step, 0.0)
                else:
                    if _projected(m, first, second, gamma, K) > 0:
                        h = _margin(z.to(tl.float64), label)
                        x = _projected(h, first, second, gamma, K) / gamma
                        entry = x * weight
            entry = tl.where(other, entry, 0.0)
            total += entry
            grad = tl.where(flag, float("nan"), entry).to(G.dtype.element_ty)
            tl.store(G + offset + k, grad, (k < C) & (k != y))
        # The target entry: entropy's -weight mass (as loss.h; a sum of the entries would lose
        # those below the normal range), the hinges' minus the competitors' unrounded sum,
        # rounded once.
        if LOSS == 0:
            mass = weight * mass
        else:
            mass = tl.sum(total, 0)
        target = tl.where(flag, float("nan"), -mass).to(G.dtype.element_ty)
        tl.store(G + offset + y, target)


def _rows_per_cta(c: int, loss: int = 0) -> int:
    """Rows per program for short top-k entropy rows (c <= 256), else 1."""
    if loss != 0 or c > 256:
        return 1
    return min(16, 256 // max(16, triton.next_power_of_2(c)))


def _shape(c: int, k: int, loss: int = 0) -> tuple[int, int, int]:
    """Chunk width, warps and the short vector of the k (or k + 1) largest."""
    block = max(16, min(triton.next_power_of_2(c), _CHUNK))
    rows = _rows_per_cta(c, loss)
    warps = 2 if rows > 1 else max(4, block // _WARP_COLUMNS)
    return block, warps, triton.next_power_of_2(k + 1)


def _fp32(logits: torch.Tensor, objective: str, fp32: bool) -> bool:
    """Float32 exponentials: only the entropy has elementwise ones, never for float64 logits."""
    return fp32 and objective == "topk_cross_entropy" and logits.dtype != torch.float64


def describe(logits: torch.Tensor, objective: str, fp32: bool) -> str:
    """The arithmetic of the kernels."""
    return "float32 exponentials" if _fp32(logits, objective, fp32) else "float64"


def covers(logits: torch.Tensor, objective: str, k: int) -> bool:
    """Whether one program per row computes this loss (top-1 losses have their own kernels).

    k = classes - 1 entropy_topk is the mean of all scores, outside these kernels.
    """
    n, c = logits.shape
    return (
        logits.is_cuda
        and objective in LOSSES
        and 1 < k <= MAX_K
        and (objective != "topk_cross_entropy" or k < c - 1)
        and n <= _cuda_launch.ROWS
        and n * c < 2**31
    )


def forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    reduction: str,
    objective: str,
    k: int,
    gamma: float,
    fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss and row statistics; CUDA invalid values propagate NaNs asynchronously."""
    n, c = logits.shape
    with _cuda_launch.device_guard(logits):
        stats = torch.empty(_triton_reduce.size(n, 4), device=logits.device, dtype=torch.float64)
        # Loss values are at least float32.
        dtype = torch.promote_types(logits.dtype, torch.float32)
        output = torch.empty((n,) if reduction == "none" else (), device=logits.device, dtype=dtype)
        loss, fp32 = LOSSES[objective], _fp32(logits, objective, fp32)
        block, warps, kb = _shape(c, k, loss)
        rows = _rows_per_cta(c, loss)
        g = _gamma(logits.device, gamma)
        _forward[(triton.cdiv(n, rows),)](
            logits, target, stats, g, n, c, block, loss, k, kb, fp32, rows, num_warps=warps
        )
        _triton_reduce.launch(stats, output, n, _REDUCTIONS[reduction], _LAYOUT)
        if not logits.is_cuda and stats[4 * n].item():
            raise ValueError("logits must be finite and target indices in range")
        return output, stats


def backward(
    logits: torch.Tensor,
    target: torch.Tensor,
    stats: torch.Tensor,
    upstream: torch.Tensor,
    reduction: str,
    objective: str,
    k: int,
    gamma: float,
    fp32: bool = False,
) -> torch.Tensor:
    """Gradient in the logits dtype, scaled by the upstream derivative in float64."""
    n, c = logits.shape
    with _cuda_launch.device_guard(logits):
        gradient = torch.empty_like(logits)
        loss = LOSSES[objective]
        block, warps, _ = _shape(c, k, loss)
        rows = _rows_per_cta(c, loss)
        _backward[(triton.cdiv(n, rows),)](
            logits,
            target,
            stats,
            upstream.contiguous(),
            gradient,
            _gamma(logits.device, gamma),
            n,
            c,
            block,
            loss,
            k,
            _REDUCTIONS[reduction],
            _fp32(logits, objective, fp32),
            rows,
            num_warps=warps,
        )
        return gradient


@functools.lru_cache(maxsize=_cuda_launch.CACHE)
def _compiled(
    device: int, dtype: torch.dtype, c: int, code: int, loss: int, k: int, fp32: bool
) -> torch.Tensor | str:
    """C++ launch handles for (device, dtype, c, reduction, loss, k, fp32), or why not."""
    with torch.cuda.device(device):
        z = torch.empty((1, 1), device="cuda", dtype=dtype)
        y = torch.empty(1, device="cuda", dtype=torch.int64)
        s = torch.empty(1, device="cuda", dtype=torch.float64)
        g = torch.empty(1, device="cuda", dtype=torch.float64)
        out = torch.empty(1, device="cuda", dtype=torch.promote_types(dtype, torch.float32))
        block, warps, kb = _shape(c, k, loss)
        rows = _rows_per_cta(c, loss)
        # Any int32 N: the kernels do not specialize on it. The C++ launcher passes a
        # float64 upstream derivative.
        k_fwd = _forward.warmup(
            z, y, s, g, 2, c, block, loss, k, kb, fp32, rows, num_warps=warps, grid=(1,)
        )
        k_bwd = _backward.warmup(
            z, y, s, g, z, g, 2, c, block, loss, k, code, fp32, rows, num_warps=warps, grid=(1,)
        )
        reduce = _triton_reduce.warmup(s, out, code, _LAYOUT)
        return _cuda_launch.handles(
            [(k_fwd, (_P, _P, _P, _P, _I))]
            + [(kernel, (_P, _P, _I)) for kernel in reduce]
            + [(k_bwd, (_P, _P, _P, _P, _P, _P, _I))],
            _triton_reduce.BLOCK,
            _triton_reduce.DIRECT_ROWS,
            0,
            rows,
            rows,
        )


def launcher(
    logits: torch.Tensor, code: int, objective: str, k: int, fp32: bool = False
) -> tuple[Any, torch.Tensor | str]:
    """(extension, kernel handles) for the C++ launch, else (None, why not)."""
    host, reason = _cuda_launch.status()
    if host is None:
        return None, reason
    device = logits.device.index or 0
    handles = _compiled(
        device,
        logits.dtype,
        logits.shape[1],
        code,
        LOSSES[objective],
        k,
        _fp32(logits, objective, fp32),
    )
    return (None, handles) if isinstance(handles, str) else (host, handles)


def apply(
    logits: torch.Tensor,
    target: torch.Tensor,
    reduction: str,
    objective: str,
    k: int,
    gamma: float,
    fp32: bool = False,
) -> torch.Tensor | None:
    """C++ autograd and launches (the k = 1 hinge entry) when the launcher runs, else None."""
    code = _REDUCTIONS[reduction]
    host, handles = launcher(logits, code, objective, k, fp32)
    if host is None:
        return None
    return host.hinge_cuda_apply(logits, target, _gamma(logits.device, gamma), code, handles)
