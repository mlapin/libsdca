"""The top-k losses, as include/sdca/loss/topk.h: topk_cross_entropy (1 < k; k = 1 is
cross_entropy's, in ce.py), multiclass_hinge and topk_hinge of both variants, nonsmooth and
smooth (gamma > 0).

A program holds ROWS short rows in a register tile, or streams one row in chunks of B columns
(a single chunk is loaded once). It keeps the k (or k + 1) largest other scores per row in a
short vector merged chunk by chunk, after a pass over per-lane maxima bounds the candidates of
streamed rows; scores order the classes as their margins h_j = 1 + z_j - z_y do, ties to the
first column. The smooth hinges project h onto their sets: a capped knapsack (bisection over
the capped count, Michelot's method for the free entries) and, for variant alpha, the top-k
cone first. Row sums whose terms vanish beyond the short vector sum it; others pass over the
row. Margins and projections are float64; top-k entropy takes float32 exponentials in float32
work, with float64 sums.

Streamed rows in float32 work (the smooth hinges; the nonsmooth top-k hinges on wide rows)
read the scores once in parallel aligned chunks (_candidates: each chunk's peak, and the
smooth beta hinge's box sums). The chunks whose peaks reach the KS-th largest peak hold the
KS largest scores; _gather reads only those again, keeping each one's largest distinct scores
with their counts, and _solve runs the solvers above on the merged candidates as on a
register row. Where a score outside them could change the result (a projection's support
reaching the largest score they leave, a tie past them, a masked label), _forward's general
path takes the row in a gated launch. The gradient of streamed rows runs in parallel aligned
windows (_chunk_backward) with the label entries last (_label_entry): rows whose nonzero
entries lie in the chosen chunks clear the other windows without reading them and compute
the chosen chunks alone (_chunk_heavy), and an indexed hinge's unique winner writes its two
entries without reading the scores.

Streamed top-k entropy rows (k <= _ENTROPY_K, float32 work) read the scores once in parallel
aligned chunks (_entropy_partial: each chunk's k largest with their counts, a level below them
and the sum of exp(z - level) under it); _entropy_merge combines them in float64 and solves
the cap as _entropy does, leaving rows it cannot decide (a masked label, a tail out of range)
to _forward's gated general path. Their gradient takes the parallel windows.

A score of -inf at a class that is not the label masks it: it is no other score and its
gradient is +0.0; with fewer than k other scores left, short vectors hold -inf pads (the limits
of docs/API.md). NaN, +inf and a nonfinite or out-of-range label give NaN rows.
"""

from __future__ import annotations

import triton
import triton.language as tl

from . import common, rows
from .common import Builder, Call, pow2
from .math import (
    DBL_EPS,
    add_parts,
    bits,
    doubles,
    exp,
    exp32,
    expm1,
    f64,
    finite,
    label_statistics,
    log,
    log1p,
    margin,
    scale,
    smooth_terms,
    target_row,
)
from .rows import chunks as row_chunks

MAX_K = 256
MAX_SMOOTH_K = 32
ENTROPY, ALPHA, BETA = tl.constexpr(0), tl.constexpr(1), tl.constexpr(2)
ALPHA_SMOOTH, BETA_SMOOTH = tl.constexpr(3), tl.constexpr(4)
ABOVE, CLIPPED, FREE = tl.constexpr(0), tl.constexpr(1), tl.constexpr(2)
# Row chunks and columns per warp: short chunks and few warps keep more rows in flight.
_CHUNK = 512
_WARP_COLUMNS = 128
# Top-k cross-entropy with k <= _SMALL_K over _WARP_ROW < classes <= _CAP_ROW searches its cap
# in one register row; below _FILLING_ROWS rows more warps share each row (_shape).
_SMALL_K, _WARP_ROW, _CAP_ROW, _FILLING_ROWS = 8, 128, 1024, 256
# Columns and warps of a streamed row's gradient chunks.
_BACKWARD = (512, 1)
# Columns from which the top-k hinges (k > 1; nonsmooth, smooth) take the candidates' forward.
_CANDIDATE_WIDTH, _SMOOTH_WIDTH = 24576, 2048
# Windows per program of the gradient of rows the candidates decided (mostly cleared).
_FILL_LOOP = 1
# Columns and warps of a streamed row's candidate chunks, and the largest k + 1 they take.
_SELECT, _CANDIDATE_K = (256, 8), 16
# Top-k entropy rows from parallel chunks (_entropy_partial): the largest k, the narrowest
# rows, the chunks' columns and warps, the gradient windows' columns and warps, and the
# merge's warps.
_ENTROPY_K, _ENTROPY_WIDTH, _ENTROPY_CHUNK = 5, 2048, (2048, 4)
_ENTROPY_BACKWARD, _ENTROPY_MERGE_WARPS = (512, 4), 2


@triton.jit
def _scores(Z, row, live, y, start, C: tl.constexpr, B: tl.constexpr, XT: tl.constexpr):
    """Other scores of columns start + j (NaN and +inf as +inf, -inf elsewhere and at masked
    classes) of ROWS rows."""
    k = start + tl.arange(0, B)[None, :]
    valid = live[:, None] & (k < C)
    z = tl.load(Z + row[:, None] * C + k, valid, other=0).to(XT)
    other = valid & (k != y[:, None])
    return tl.where(other, tl.where(z < float("inf"), z, float("inf")), -float("inf"))


@triton.jit
def _at(top, kk, i):
    """Entry i of each row's short vector."""
    return tl.max(tl.where(kk == i, top, -float("inf")), 1)


@triton.jit
def _atv(top, kk, i):
    """Entry i[r] of row r's short vector."""
    return tl.max(tl.where(kk == i[:, None], top, -float("inf")), 1)


@triton.jit
def _kth(lanes, K1: tl.constexpr):
    """The K1-th largest of each row's lanes, ties repeated (-inf if fewer)."""
    value = tl.max(lanes, 1)
    count = tl.sum((lanes == value[:, None]).to(tl.int32), 1)
    while tl.min(count, 0) < K1:
        following = tl.max(tl.where(lanes < value[:, None], lanes, -float("inf")), 1)
        more = count < K1
        value = tl.where(more, following, value)
        count += tl.where(more, tl.sum((lanes == following[:, None]).to(tl.int32), 1), 0)
    return value


@triton.jit
def _merge(val, idx, filled, least, x, start, evicted, K1: tl.constexpr, TAIL: tl.constexpr):
    """Slots (val, idx) with the entries of x above least, largest and leftmost first: the
    first K1 slots hold the largest entries (least is their smallest, -inf until all are
    filled), a full row evicting its latest column at least. Returns the slots, the taken mask
    and, for TAIL, evicted plus the evicted entries' exp(value - least), relative to least."""
    kk = tl.arange(0, val.shape[1])[None, :]
    lane = tl.arange(0, x.shape[1])[None, :]
    live = kk < K1
    taken = x < -float("inf")
    best, at = tl.max(x, 1, return_indices=True, return_indices_tie_break_left=True)
    go = best > least
    while tl.max(go.to(tl.int32), 0) > 0:
        full = filled >= K1
        at_least = live & (val == least[:, None])
        latest = tl.max(tl.where(at_least, idx, -(2**31)), 1)
        slot = tl.where(full[:, None], at_least & (idx == latest[:, None]), kk == filled[:, None])
        slot = slot & go[:, None]
        val = tl.where(slot, best[:, None], val)
        idx = tl.where(slot, start + at[:, None], idx)
        filled = tl.where(go, tl.minimum(filled + 1, K1), filled)
        taken = taken | ((lane == at[:, None]) & go[:, None])
        new = tl.where(filled == K1, tl.min(tl.where(live, val, float("inf")), 1), least)
        if TAIL:
            shift = exp(least.to(tl.float64) - new.to(tl.float64))
            evicted = tl.where(go & full, (evicted + 1.0) * shift, evicted)
        least = tl.where(go, new, least)
        best, at = tl.max(
            tl.where(taken, -float("inf"), x),
            1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        go = best > least
    return val, idx, filled, least, taken, evicted


@triton.jit
def _select(
    Z,
    row,
    live,
    y,
    x1,
    ROWS: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    KB: tl.constexpr,
    K1: tl.constexpr,
    XT: tl.constexpr,
    TAIL: tl.constexpr,
    COUNT_CUTOFF: tl.constexpr = False,
):
    """The K1 largest other scores of each row in decreasing order, the largest
    lane, optional cutoff multiplicity and, for TAIL, the sum of exp(x - top[K1 - 1]) over the other scores. Streamed rows
    take only candidates at least the K1-th largest lane maximum."""
    SINGLE: tl.constexpr = B >= C
    tied = tl.zeros([ROWS], tl.int32)
    if SINGLE and TAIL and K1 <= 8:
        # A register row needs only the leading scores and the uncapped tail.
        # Select one distinct lane at a time so equal scores keep their multiplicity.
        lane = tl.arange(0, B)[None, :]
        kk = tl.arange(0, KB)[None, :]
        top = tl.full([ROWS, KB], -float("inf"), tl.float64)
        remaining = x1
        largest = tl.max(x1, 1)
        last = largest
        for i in tl.static_range(K1):
            last, at = tl.max(remaining, 1, return_indices=True, return_indices_tie_break_left=True)
            top = tl.where(kk == i, last.to(tl.float64)[:, None], top)
            remaining = tl.where(lane == at[:, None], -float("inf"), remaining)
        terms = tl.where(remaining > -float("inf"), exp(remaining - last[:, None]), 0.0)
        return top, largest, tl.sum(terms.to(tl.float64), 1), tied
    lanes = x1
    if SINGLE:
        bound = tl.full([ROWS], -float("inf"), XT)
    else:
        for start in range(B, C, B):
            x = _scores(Z, row, live, y, start, C, B, XT)
            lanes = tl.maximum(lanes, x)
        if K1 > 32:
            bound = _at(tl.sort(lanes, descending=True), tl.arange(0, B)[None, :], K1 - 1)
        else:
            bound = _kth(lanes, K1)
    kk = tl.arange(0, KB)[None, :]
    val = tl.full([ROWS, KB], -float("inf"), XT)
    idx = -1 - kk + tl.zeros([ROWS, KB], tl.int32)
    filled = tl.zeros([ROWS], tl.int32)
    least = tl.full([ROWS], -float("inf"), XT)
    evicted = tl.zeros([ROWS], tl.float64)
    ref = bound
    tail = tl.zeros([ROWS, B], tl.float64)
    for start in range(0, C, B):
        if SINGLE:
            x = x1
        else:
            x = _scores(Z, row, live, y, start, C, B, XT)
        candidates = tl.where(x >= bound[:, None], x, -float("inf"))
        taken = x < -float("inf")
        previous_slots = val
        previous_cutoff = least
        if tl.max((tl.max(candidates, 1) > least).to(tl.int32), 0) > 0:
            val, idx, filled, least, taken, evicted = _merge(
                val, idx, filled, least, candidates, start, evicted, K1, TAIL
            )
        if COUNT_CUTOFF:
            # _merge admits only scores strictly above the old cutoff. When it rises,
            # all earlier occurrences of the new cutoff are still in previous_slots.
            # At an unchanged cutoff, keep the full count including discarded ties.
            retained = tl.sum(((kk < K1) & (previous_slots == least[:, None])).to(tl.int32), 1)
            tied = tl.where(least > previous_cutoff, retained, tied)
            tied += tl.sum((x == least[:, None]).to(tl.int32), 1)
        if TAIL:
            new = tl.maximum(ref, least)
            factor = tl.where(new > ref, exp(ref.to(tl.float64) - new.to(tl.float64)), 1.0)
            tail = tail * factor[:, None]
            ref = new
            e = tl.where(taken | (x == -float("inf")), 0.0, exp(x - ref[:, None]))
            tail += e.to(tl.float64)
    top = tl.sort(val, descending=True).to(tl.float64)
    return top, tl.max(lanes, 1), tl.sum(tail, 1) + evicted, tied


@triton.jit
def _rule(h, keep, MODE: tl.constexpr, p1, p2, p3, AXIS: tl.constexpr):
    """Over margins h where keep: ABOVE: sum max(h - p1, 0); CLIPPED: sum clip(h - p1 + p2, 0,
    p2); FREE: the count, sum and least of g = h - p2 over h <= p1 and g > p3."""
    if MODE == FREE:
        g = h - p2
        inside = keep & (h <= p1) & (g > p3)
        count = tl.sum(inside.to(tl.int32), AXIS)
        total = tl.sum(tl.where(inside, g, 0.0), AXIS)
        least = tl.min(tl.where(inside, g, float("inf")), AXIS)
    else:
        if MODE == ABOVE:
            v = tl.maximum(h - p1, 0.0)
        else:
            v = tl.minimum(tl.maximum((h - p1) + p2, 0.0), p2)
        total = tl.sum(tl.where(keep, v, 0.0), AXIS)
        count = total.to(tl.int32) * 0
        least = total
    return count, total, least


@triton.jit
def _reaches(m, MODE: tl.constexpr, p1, p2, p3):
    """Whether margins up to m can add a term to _rule."""
    if MODE == ABOVE:
        r = m > p1
    elif MODE == CLIPPED:
        r = (m - p1) + p2 > 0
    else:
        r = (m - p2) > p3
    return r


@triton.jit
def _pass(
    Z,
    row,
    live,
    y,
    z_y,
    x1,
    top,
    kk,
    K1: tl.constexpr,
    MODE: tl.constexpr,
    p1,
    p2,
    p3,
    ROWS: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    XT: tl.constexpr,
):
    """_rule over each row's other margins: from the short vector where the terms vanish beyond
    it (its K1-th entry adds none), else over the row (chunks whose largest margin adds none
    skipped)."""
    SINGLE: tl.constexpr = B >= C
    zero = tl.zeros([ROWS], tl.float64)
    p1, p2, p3 = p1 + zero, p2 + zero, p3 + zero
    q1, q2, q3 = p1[:, None], p2[:, None], p3[:, None]
    count, total, least = _rule(top, kk < K1, MODE, q1, q2, q3, 1)
    whole = _reaches(_at(top, kk, K1 - 1), MODE, p1, p2, p3) | SINGLE
    if tl.max(whole.to(tl.int32), 0) > 0:
        n = tl.zeros([ROWS], tl.int32)
        s = tl.zeros([ROWS], tl.float64)
        m = tl.full([ROWS], float("inf"), tl.float64)
        for start in range(0, C, B):
            if SINGLE:
                x = x1
            else:
                x = _scores(Z, row, live, y, start, C, B, XT)
            reach = _reaches(margin(tl.max(x, 1).to(tl.float64), z_y), MODE, p1, p2, p3)
            if tl.max((whole & reach).to(tl.int32), 0) > 0:
                h = margin(x.to(tl.float64), z_y[:, None])
                cn, cs, cm = _rule(h, x > -float("inf"), MODE, q1, q2, q3, 1)
                n += cn
                s += cs
                m = tl.minimum(m, cm)
        count = tl.where(whole, n, count)
        total = tl.where(whole, s, total)
        least = tl.where(whole, m, least)
    return count, total, least


@triton.jit
def _knapsack(
    Z,
    row,
    live,
    y,
    z_y,
    x1,
    top,
    kk,
    active,
    cap,
    rhs,
    ROWS: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    K1: tl.constexpr,
    XT: tl.constexpr,
):
    """Origin and correction of a = clip((h - origin) - correction, 0, cap), sum(a) = rhs, of
    the active rows: bisection on the clipped mass at top[j - 1] counts the u <= K largest at
    the cap (ties never split); Michelot's method from the superset {h > top[u] - cap} of the
    free entries gives the correction (<= 0) relative to the largest free margin."""
    lo = tl.zeros([ROWS], tl.int32)
    hi = tl.where(active, K, 0)
    unused = cap * 0
    while tl.max(hi - lo, 0) > 0:
        mid = (lo + hi + 1) // 2
        level = _atv(top, kk, mid - 1)
        clip_n, clipped, clip_m = _pass(
            Z, row, live, y, z_y, x1, top, kk, K1, CLIPPED, level, cap, unused, ROWS, C, B, XT
        )
        ok = clipped <= rhs
        searching = lo < hi
        lo = tl.where(searching & ok, mid, lo)
        hi = tl.where(searching & ~ok, mid - 1, hi)
    origin = _atv(top, kk, tl.minimum(lo, K - 1))
    remainder = rhs - cap * lo.to(tl.float64)
    level = tl.zeros([ROWS], tl.float64) - cap
    s = level
    running = active & (lo < K)
    if tl.max(running.to(tl.int32), 0) > 0:
        # The set {g > level} only shrinks: rounding cannot cycle; it is stable once its least
        # gap stays above the level.
        n, total, least = _pass(
            Z, row, live, y, z_y, x1, top, kk, K1, FREE, origin, origin, level, ROWS, C, B, XT
        )
        s = tl.where(running, (total - remainder) / n, s)
        level = tl.where(running, tl.maximum(level, s), level)
        running = running & (least <= level)
        while tl.max(running.to(tl.int32), 0) > 0:
            n, total, least = _pass(
                Z, row, live, y, z_y, x1, top, kk, K1, FREE, origin, origin, level, ROWS, C, B, XT
            )
            s = tl.where(running, (total - remainder) / n, s)
            level = tl.where(running, tl.maximum(level, s), level)
            running = running & (least <= level)
    return origin, s


@triton.jit
def _cone(
    Z,
    row,
    live,
    y,
    z_y,
    x1,
    top,
    kk,
    active,
    ROWS: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    K1: tl.constexpr,
    XT: tl.constexpr,
):
    """Threshold t and cap c of the projection clip(h - t, 0, c) onto 0 <= x_i <= sum(x) / K:
    exactly the K largest are capped when the rest lie below their threshold; else bisection
    on the stationarity residual counts the u capped, and Michelot's method finds the free
    entries of t = (u rest - (K - u) head) / d."""
    # The K largest in double-double, and 16 eps relative to scores below one.
    whole = tl.zeros([ROWS], tl.float64)
    low = whole
    scale = whole
    for i in tl.static_range(K):
        x = _at(top, kk, i)
        total = whole + x
        rounded = total - whole
        low += tl.where(finite(total), (whole - (total - rounded)) + (x - rounded), 0.0)
        whole = total
        scale += tl.abs(x)
    whole += low
    tol = 16 * DBL_EPS * tl.minimum(scale, 1.0)
    c = whole / K
    t = _at(top, kk, K - 1) - c
    unused = c * 0
    cone = active & (t < _at(top, kk, K) - tol) & (whole > tol)
    if tl.max(cone.to(tl.int32), 0) > 0:
        # u is one past the last j whose level top[j] lies above the solution's cap level.
        lo = tl.full([ROWS], -1, tl.int32)
        hi = tl.where(cone, K - 1, -1)
        while tl.max(hi - lo, 0) > 0:
            mid = (lo + hi + 1) // 2
            level = _atv(top, kk, mid)
            above_n, above, above_m = _pass(
                Z, row, live, y, z_y, x1, top, kk, K1, ABOVE, level, unused, unused, ROWS, C, B, XT
            )
            shift = -above / K
            above_n, above, above_m = _pass(
                Z, row, live, y, z_y, x1, top, kk, K1, ABOVE, shift, unused, unused, ROWS, C, B, XT
            )
            ok = above / K + 2 * shift < level
            searching = lo < hi
            lo = tl.where(searching & ok, mid, lo)
            hi = tl.where(searching & ~ok, mid - 1, hi)
        u = lo + 1
        running = cone & (u < K)
        if tl.max(running.to(tl.int32), 0) > 0:
            head = tl.sum(tl.where(kk < u[:, None], top, 0.0), 1)
            bound = _atv(top, kk, u)
            level = tl.full([ROWS], -float("inf"), tl.float64)
            m = tl.zeros([ROWS], tl.int32)
            rest = unused
            while tl.max(running.to(tl.int32), 0) > 0:
                n, total, least = _pass(
                    Z,
                    row,
                    live,
                    y,
                    z_y,
                    x1,
                    top,
                    kk,
                    K1,
                    FREE,
                    bound,
                    unused,
                    level,
                    ROWS,
                    C,
                    B,
                    XT,
                )
                m = tl.where(running, n, m)
                rest = tl.where(running, total, rest)
                d = ((K - u) * (K - u) + m * u).to(tl.float64)
                following = (u * rest - (K - u) * head) / d
                t = tl.where(running, following, t)
                level = tl.where(running, tl.maximum(level, following), level)
                running = running & (least <= level)
            free = cone & (u < K)
            d = ((K - u) * (K - u) + m * u).to(tl.float64)
            c = tl.where(free, (m * head + (K - u) * rest) / d, c)
    zero = ~active | (whole <= tol)
    return tl.where(zero, 0.0, t), tl.where(zero, 0.0, c)


@triton.jit
def _projected(h, first, second, gamma, K: tl.constexpr):
    """The smooth hinges' a from two row statistics: a positive second is the cone's cap at
    threshold first; else first and second (<= 0) are a knapsack's origin and correction."""
    cone = second > 0
    shifted = tl.where(cone, h - first, (h - first) - second)
    return tl.minimum(tl.maximum(shifted, 0.0), tl.where(cone, second, gamma / K))


@triton.jit
def _simplex_shift(gaps, gamma, B: tl.constexpr):
    """Threshold relative to the maximum score; only gaps above -gamma can be active."""
    if B <= 16:
        ordered = tl.sort(tl.maximum(gaps, -gamma), descending=True)
        ranks = (1 + tl.arange(0, B)).to(tl.float64)
        return tl.max((tl.cumsum(ordered, axis=1) - gamma) / ranks[None, :], 1)
    else:
        # Michelot's projection from the public multiclass hinge. Removing gaps below
        # the current level can only increase it. The first stable active set is exact.
        active = gaps > -gamma
        count = tl.sum(active.to(tl.int32), 1)
        shift = (tl.sum(tl.where(active, gaps, 0.0), 1) - gamma) / tl.maximum(count, 1)
        least = tl.min(tl.where(active, gaps, float("inf")), 1)
        while tl.max((least <= shift).to(tl.int32), 0) > 0:
            active = active & (gaps > shift[:, None])
            count = tl.sum(active.to(tl.int32), 1)
            next_shift = (tl.sum(tl.where(active, gaps, 0.0), 1) - gamma) / tl.maximum(count, 1)
            shift = tl.maximum(shift, next_shift)
            least = tl.min(tl.where(active, gaps, float("inf")), 1)
        return shift


@triton.jit
def _stored_midpoint(value, magnitude):
    # Only positive finite projected values occur. Keep the original arithmetic
    # near a single-precision rounding boundary, including at a binade edge.
    nearest = value.to(tl.float32)
    word = nearest.to(tl.uint32, bitcast=True)
    adjacent = tl.where(value >= nearest.to(tl.float64), word + 1, word - 1)
    halfway = 0.5 * (nearest.to(tl.float64) + adjacent.to(tl.float32, bitcast=True).to(tl.float64))
    return finite(value) & (tl.abs(value - halfway) <= (64 * DBL_EPS) * magnitude)


@triton.jit
def _checked_short_shift(gaps, gamma, B: tl.constexpr):
    """Select on FP32 gaps, then verify the support using the FP64 threshold."""
    fg = gaps.to(tl.float32)
    amount = gamma.to(tl.float32)
    active = fg > -amount
    count = tl.sum(active.to(tl.int32), 1)
    edge = (tl.sum(tl.where(active, fg, 0.0), 1) - amount) / tl.maximum(count, 1)
    least = tl.min(tl.where(active, fg, float("inf")), 1)
    while tl.max((least <= edge).to(tl.int32), 0) > 0:
        active = active & (fg > edge[:, None])
        old_count = count
        count = tl.sum(active.to(tl.int32), 1)
        edge = (tl.sum(tl.where(active, fg, 0.0), 1) - amount) / tl.maximum(count, 1)
        least = tl.where(
            count < old_count, tl.min(tl.where(active, fg, float("inf")), 1), float("inf")
        )
    shift = (tl.sum(tl.where(active, gaps, 0.0), 1) - gamma) / tl.maximum(count, 1)
    certificate = (
        (count > 0)
        & (tl.min(tl.where(active, gaps, float("inf")), 1) > shift)
        & (tl.max(tl.where(active, -float("inf"), gaps), 1) <= shift)
    )
    if tl.min(certificate.to(tl.int32), 0) == 0:
        shift = _simplex_shift(gaps, gamma, B)
    return shift


@triton.jit
def _short_simplex(x, label, bad, gamma, B: tl.constexpr):
    """Project a register-resident row around its best competitor."""
    largest = tl.max(x, 1).to(tl.float64)
    peak = tl.where(finite(largest), largest, 0.0)
    highest = x == largest[:, None]
    runner = tl.max(tl.where(highest, -float("inf"), x), 1).to(tl.float64)
    best = margin(peak, label)
    isolated = (
        (tl.sum(highest.to(tl.int32), 1) == 1)
        & finite(largest)
        & (peak - runner >= gamma)
        & (best > gamma)
    )
    value = best - 0.5 * gamma
    if tl.min((isolated & ~_stored_midpoint(value, tl.abs(best) + gamma)).to(tl.int32), 0) > 0:
        # A positive second field tags this unique solution; projection shifts are nonpositive.
        return (
            value,
            peak,
            gamma + tl.zeros(peak.shape, tl.float64),
            tl.full(peak.shape, float("inf"), tl.float64),
            bad,
        )
    gaps = tl.where(finite(x), x.to(tl.float64) - peak[:, None], -float("inf"))
    shift = _checked_short_shift(gaps, gamma, B)
    projected = finite(largest) & (margin(peak, label) > -shift)
    if tl.min(projected.to(tl.int32)) > 0:
        # On the simplex h = h_peak + shift + a at each active entry;
        # sum(a) = gamma. Work and gamma bounds keep the squares finite.
        a = tl.minimum(tl.maximum(gaps - shift[:, None], 0.0), gamma)
        best = margin(peak, label)
        value = (best + shift) + (0.5 * tl.sum(a * a, 1)) / gamma
        midpoint = _stored_midpoint(value, tl.abs(best) + gamma)
        if tl.max(midpoint.to(tl.int32), 0) > 0:
            h = margin(x.to(tl.float64), label[:, None])
            value = tl.where(midpoint, smooth_terms(h, finite(x), a, gamma, 1), value)
    else:
        h = margin(x.to(tl.float64), label[:, None])
        a = tl.where(projected[:, None], gaps - shift[:, None], h)
        a = tl.minimum(tl.maximum(a, 0.0), gamma)
        value = smooth_terms(h, finite(x), a, gamma, 1)
    return (
        value,
        tl.where(projected, peak, 0.0),
        tl.where(projected, shift, 0.0),
        tl.where(projected, float("inf"), label),
        bad | (largest == float("inf")),
    )


@triton.jit
def _smooth_k1(Z, row, live, y, z_y, bad, gamma, ROWS: tl.constexpr, C: tl.constexpr,
               B: tl.constexpr, XT: tl.constexpr):  # fmt: skip
    """Value, origin and correction of the smooth hinge at k = 1 over rows wider than B: the
    projection of the margins h onto {0 <= a <= gamma, sum a <= gamma} in three kinds of
    pass. The first sums the box clip(h, 0, gamma) and keeps Michelot's superset
    {h - top > -gamma} against the running largest margin top; each further pass refines
    the threshold s on the gaps h - top (only shrinking the set, so rounding cannot cycle);
    the last sums the value over the chunks that hold support. a = clip((h - top) - s, 0,
    gamma), the knapsack form of _projected (origin top, correction s <= 0)."""
    top = tl.full([ROWS], -float("inf"), tl.float64)
    largest = tl.full([ROWS], -float("inf"), XT)
    box = tl.zeros([ROWS], tl.float64)
    count = tl.zeros([ROWS], tl.int32)
    total = tl.zeros([ROWS], tl.float64)
    for start in range(0, C, B):
        x = _scores(Z, row, live, y, start, C, B, XT)
        largest = tl.maximum(largest, tl.max(x, 1))
        other = x > -float("inf")
        h = margin(x.to(tl.float64), z_y[:, None])
        box += tl.sum(tl.where(other, tl.minimum(tl.maximum(h, 0.0), gamma), 0.0), 1)
        new = tl.maximum(top, tl.max(tl.where(other, h, -float("inf")), 1))
        # Shift the kept gaps to the new largest margin; a rise of gamma drops them all.
        shift = new - top
        keep = shift < gamma
        total = tl.where(keep, total - count.to(tl.float64) * shift, 0.0)
        count = tl.where(keep, count, 0)
        g = h - new[:, None]
        inside = other & (g > -gamma)
        count += tl.sum(inside.to(tl.int32), 1)
        total += tl.sum(tl.where(inside, g, 0.0), 1)
        top = new
    bad = bad | (largest == float("inf"))
    project = live & ~bad & (box > gamma)
    level = tl.zeros([ROWS], tl.float64) - gamma
    s = tl.maximum((total - gamma) / tl.maximum(count, 1).to(tl.float64), level)
    level = tl.maximum(level, s)
    running = project
    while tl.max(running.to(tl.int32), 0) > 0:
        count = tl.zeros([ROWS], tl.int32)
        total = tl.zeros([ROWS], tl.float64)
        least = tl.full([ROWS], float("inf"), tl.float64)
        for start in range(0, C, B):
            x = _scores(Z, row, live, y, start, C, B, XT)
            g = margin(x.to(tl.float64), z_y[:, None]) - top[:, None]
            inside = (x > -float("inf")) & (g > level[:, None])
            count += tl.sum(inside.to(tl.int32), 1)
            total += tl.sum(tl.where(inside, g, 0.0), 1)
            least = tl.minimum(least, tl.min(tl.where(inside, g, float("inf")), 1))
        fresh = (total - gamma) / tl.maximum(count, 1).to(tl.float64)
        s = tl.where(running, fresh, s)
        level = tl.where(running, tl.maximum(level, fresh), level)
        # The set is stable once its least gap stays above the threshold.
        running = running & (least <= level)
    origin = tl.where(project, top, 0.0)
    correction = tl.where(project, s, 0.0)
    value = tl.zeros([ROWS], tl.float64)
    for start in range(0, C, B):
        x = _scores(Z, row, live, y, start, C, B, XT)
        m = margin(tl.max(x, 1).to(tl.float64), z_y)
        # Chunks without support (most, once projected) skip the float64 divisions.
        if tl.max((live & (_projected(m, origin, correction, gamma, 1) > 0)).to(tl.int32), 0) > 0:
            h = margin(x.to(tl.float64), z_y[:, None])
            a = _projected(h, origin[:, None], correction[:, None], gamma, 1)
            value += smooth_terms(h, x > -float("inf"), a, gamma, 1)
    return value, origin, correction, z_y, bad


@triton.jit
def _smooth(
    Z,
    row,
    live,
    y,
    z_y,
    x1,
    bad,
    gamma,
    ROWS: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    KB: tl.constexpr,
    LOSS: tl.constexpr,
    XT: tl.constexpr,
):
    """Value and the two statistics of a smooth hinge."""
    SINGLE: tl.constexpr = B >= C
    K1: tl.constexpr = K + 1
    cap = gamma / K
    kk = tl.arange(0, KB)[None, :]
    top, largest, tail, _ = _select(Z, row, live, y, x1, ROWS, C, B, KB, K1, XT, False)
    cutoff = _at(top, kk, K - 1)
    next_score = _at(top, kk, K)
    hi, lo = _score_sum(top, (kk < K) & finite(top), K)
    mean = _mean_margin(hi, lo, z_y, K)
    # With a gap below the k-th score wider than its projected mass, only
    # the k largest can be active. They all take the same mass on the cone.
    head_mass = tl.minimum(tl.maximum(mean, 0.0), cap)
    isolated = finite(cutoff) & (cutoff - next_score >= head_mass) & (cutoff > next_score)
    if LOSS == BETA_SMOOTH:
        isolated = isolated & (margin(cutoff, z_y) >= cap)
    isolated_value = tl.where(
        head_mass > 0, (mean - 0.5 * head_mass) * ((head_mass / gamma) * K), 0.0
    )
    direct_step = head_mass / gamma
    if XT == tl.float64:
        # Several identical winners can lie more than DBL_MAX above the label.
        # Their common margin overflows, but their relative score gaps determine
        # a finite simplex projection. Check whether the whole tied group is isolated.
        overflow = (
            (mean == float("inf")) & (largest == cutoff) & (cutoff == next_score) & finite(cutoff)
        )
        if tl.max((live & overflow).to(tl.int32), 0) > 0:
            count = tl.zeros([ROWS], tl.int32)
            runner = tl.full([ROWS], -float("inf"), tl.float64)
            for start in range(0, C, B):
                if SINGLE:
                    scores = x1
                else:
                    scores = _scores(Z, row, live, y, start, C, B, XT)
                count += tl.sum((scores == cutoff[:, None]).to(tl.int32), 1)
                runner = tl.maximum(
                    runner, tl.max(tl.where(scores < cutoff[:, None], scores, -float("inf")), 1)
                )
            share = 1.0 / tl.maximum(count, 1).to(tl.float64)
            closed = overflow & (cutoff - runner >= gamma * share)
            isolated = isolated | closed
            direct_step = tl.where(closed, share, direct_step)
            isolated_value = tl.where(
                closed, (cutoff - gamma * share * 0.5) - z_y + 1, isolated_value
            )
    top = margin(top, z_y[:, None])
    bad = bad | (largest == float("inf"))
    origin = tl.zeros([ROWS], tl.float64)
    correction = origin
    active = live & ~bad & ~isolated
    if LOSS == BETA_SMOOTH:
        unused = cap * 0 + origin
        box_n, box, box_m = _pass(
            Z, row, live, y, z_y, x1, top, kk, K1, CLIPPED, cap, cap, unused, ROWS, C, B, XT
        )
        project = active & (box > gamma)
        if tl.max(project.to(tl.int32), 0) > 0:
            o, s = _knapsack(
                Z, row, live, y, z_y, x1, top, kk, project, cap, gamma, ROWS, C, B, K, K1, XT
            )
            origin = tl.where(project, o, origin)
            correction = tl.where(project, s, correction)
        last = tl.where(project, _at(top, kk, K1 - 1), float("inf"))
    else:
        # The cone's projection unless its sum k c exceeds gamma: then sum(a) = gamma.
        t, c = _cone(Z, row, live, y, z_y, x1, top, kk, active, ROWS, C, B, K, K1, XT)
        full = active & (K * c > gamma)
        origin = tl.where(c > 0, t, float("inf"))
        correction = c
        if tl.max(full.to(tl.int32), 0) > 0:
            o, s = _knapsack(
                Z, row, live, y, z_y, x1, top, kk, full, cap, gamma, ROWS, C, B, K, K1, XT
            )
            origin = tl.where(full, o, origin)
            correction = tl.where(full, s, correction)
        # Masked classes may leave fewer than K margins: the cone is zero.
        few = _at(top, kk, K - 1) == -float("inf")
        origin = tl.where(few | ~active, float("inf"), origin)
        correction = tl.where(few | ~active, 0.0, correction)
        last = _at(top, kk, K1 - 1)
    value = smooth_terms(
        top, kk < K1, _projected(top, origin[:, None], correction[:, None], gamma, K), gamma, 1
    )
    whole = (_projected(last, origin, correction, gamma, K) > 0) | SINGLE
    if tl.max(whole.to(tl.int32), 0) > 0:
        total = tl.zeros([ROWS], tl.float64)
        for start in range(0, C, B):
            if SINGLE:
                x = x1
            else:
                x = _scores(Z, row, live, y, start, C, B, XT)
            m = margin(tl.max(x, 1).to(tl.float64), z_y)
            if (
                tl.max((whole & (_projected(m, origin, correction, gamma, K) > 0)).to(tl.int32), 0)
                > 0
            ):
                h = margin(x.to(tl.float64), z_y[:, None])
                a = _projected(h, origin[:, None], correction[:, None], gamma, K)
                total += smooth_terms(h, x > -float("inf"), a, gamma, 1)
        value = tl.where(whole, total, value)
    return (
        tl.where(isolated, isolated_value, value),
        tl.where(isolated, cutoff, origin),
        tl.where(isolated, direct_step, correction),
        tl.where(isolated, -float("inf"), z_y),
        bad,
    )


@triton.jit
def _score_sum(x, keep, K: tl.constexpr):
    """A two-part sum scaled so K scores as large as finfo.max fit."""
    scale: tl.constexpr = 0.5 / triton.next_power_of_2(K)
    entries = tl.where(keep, x.to(tl.float64) * scale, 0.0)
    return tl.reduce((entries, tl.full(entries.shape, 0.0, tl.float64)), 1, add_parts)


@triton.jit
def _mean_margin(total, carry, z_y, K: tl.constexpr):
    """Mean competitor score minus the label, with a unit hinge offset before cancellation."""
    power: tl.constexpr = 2.0 * triton.next_power_of_2(K)
    total, carry = add_parts(total, carry, tl.full(total.shape, K / power, tl.float64), 0.0)
    return (tl.fma(-z_y / power, K, total) + carry) * (power / K)


@triton.jit
def _add_capped(a, ma, la, b, mb, lb):
    """Retain the sum of both correction terms when three score scales cancel."""
    high, error = add_parts(a, 0.0, b, 0.0)
    middle, fine = add_parts(ma, la, mb, lb)
    middle, fine = add_parts(middle, fine, error, 0.0)
    return high, middle, fine


@triton.jit
def _entropy_mean(total, carry, fine, z_y, K: tl.constexpr):
    """Score mean and its exponential correction in the negative tail."""
    unit: tl.constexpr = K / (2.0 * triton.next_power_of_2(K))
    high = tl.fma(-z_y, unit, total)
    mean = ((high + carry) + fine) / unit
    error = ((tl.fma(-mean, unit, high) + carry) + fine) / unit
    return mean, exp(tl.where((mean < 0) & (mean > -1500), error, 0.0))


@triton.jit
def _all_capped(Z, row, live, y, z_y, x1, bad, ROWS: tl.constexpr, C: tl.constexpr,
                B: tl.constexpr, K: tl.constexpr, XT: tl.constexpr):  # fmt: skip
    """Every competitor holds mass / K. A single mask leaves fewer than K."""
    count = tl.zeros([ROWS], tl.int32)
    total = tl.zeros([ROWS], tl.float64)
    carry, fine = total, total
    for start in range(0, C, B):
        if B >= C:
            x = x1
        else:
            x = _scores(Z, row, live, y, start, C, B, XT)
        keep = finite(x)
        count += tl.sum(keep.to(tl.int32), 1)
        entries = tl.where(keep, x.to(tl.float64) * (0.5 / triton.next_power_of_2(K)), 0.0)
        zero = tl.full(entries.shape, 0.0, tl.float64)
        tile, mid, low = tl.reduce((entries, zero, zero), 1, _add_capped)
        total, carry, fine = _add_capped(total, carry, fine, tile, mid, low)
        bad = bad | (tl.max(x, 1) == float("inf"))
    mean, factor = _entropy_mean(total, carry, fine, z_y, K)
    loss, _, _ = label_statistics(mean, K * factor)
    few = count < K
    zero = tl.zeros([ROWS], tl.float64)
    return tl.where(few, 0.0, loss), zero, zero + 1.0, bad


@triton.jit
def _bounded_cap(top, last, tail, kk, K: tl.constexpr):
    """Solve small-k caps together when the open head fits in one exponential range."""
    weights = tl.where(kk < K, exp(top - last[:, None]), 0.0)
    # Sum only the classes strictly to the right of the candidate in score order.
    # This avoids subtracting a possibly dominant head from the whole partition.
    suffix = tl.sum(tl.where(kk[:, None, :] > kk[:, :, None], weights[:, None, :], 0.0), 2)
    tail_by = (suffix + tail[:, None]) / tl.where(kk < K, weights, 1.0)
    threshold = (1.0 + tail_by) - (K - kk)
    within = (kk > 0) & (kk < K)
    sensitive = tl.sum((within & (tl.abs(threshold) < (64.0 * DBL_EPS * K))).to(tl.int32), 1)
    active = tl.min(tl.where(within & (threshold >= 0.0), kk, K - 1), 1)
    free = _atv(top, kk, active)
    below = _atv(tail_by, kk, active)
    copies = tl.sum(((kk < active[:, None]) & (top == free[:, None])).to(tl.int32), 1)
    return active, free, log1p(below), below + copies.to(tl.float64), sensitive == 0


@triton.jit
def _entropy(
    Z,
    row,
    live,
    y,
    z_y,
    x1,
    bad,
    ROWS: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    KB: tl.constexpr,
    XT: tl.constexpr,
):
    """Top-k entropy, with the free scores relative to the first free one: free scores far
    below a capped one keep their differences, which scores relative to the maximum round
    away. Returns the loss and the backward's free score and D (zero and one where fewer than
    K other scores are left: no loss)."""
    SINGLE: tl.constexpr = B >= C
    # The largest other score and the sum of exp(z - largest) over the others but one copy.
    largest = tl.full([ROWS], -float("inf"), XT)
    acc = tl.zeros([ROWS, B], tl.float64)
    ties = tl.zeros([ROWS, B], tl.int32)
    for start in range(0, C, B):
        if SINGLE:
            x = x1
        else:
            x = _scores(Z, row, live, y, start, C, B, XT)
        new = tl.maximum(largest, tl.max(x, 1))
        rises = new > largest
        factor = tl.where(rises, exp(largest.to(tl.float64) - new.to(tl.float64)), 1.0)
        acc = tl.where(rises[:, None], (acc + ties.to(tl.float64)) * factor[:, None], acc)
        ties = tl.where(rises[:, None], 0, ties)
        largest = new
        acc += tl.where(x < largest[:, None], exp(x - largest[:, None]), 0.0).to(tl.float64)
        ties += ((x == largest[:, None]) & (x > -float("inf"))).to(tl.int32)
    maximum = largest.to(tl.float64)
    others = tl.sum(acc, 1) + (tl.sum(ties, 1) - 1).to(tl.float64)
    loss, free, second, bad, _ = _entropy_capped(
        Z, row, live, y, z_y, x1, bad, maximum, others, maximum, maximum, maximum, ROWS, C, B,
        K, KB, XT, False,
    )  # fmt: skip
    return loss, free, second, bad


@triton.jit
def _entropy_capped(Z, row, live, y, z_y, x1, bad, maximum, others, rest, peaks, left,
                    ROWS: tl.constexpr, C: tl.constexpr, B: tl.constexpr, K: tl.constexpr,
                    KB: tl.constexpr, XT: tl.constexpr, CANDIDATES: tl.constexpr):  # fmt: skip
    """_entropy from the largest other score and the sum of exp(z - largest) over the others
    but one copy. CANDIDATES: x1 holds the row chunks' candidates, rest and peaks their chunks'
    sums of exp(z - peak) over the scores they leave and their peaks, left the largest score
    they leave; returns whether the candidates may not decide the row (a score they leave
    among the K largest, a tail below the float64 range) as well."""
    SINGLE: tl.constexpr = B >= C
    kk = tl.arange(0, KB)[None, :]
    bad = bad | (maximum == float("inf"))
    few = maximum == -float("inf")
    unsure = live & (maximum != maximum)
    eps = 32 * DBL_EPS
    log_k = log(tl.full([ROWS], K, tl.float64))
    # No score capped (the fast check of the thresholds): the odds of entropy.
    lse = log1p(others)
    threshold = lse - log_k
    active = tl.zeros([ROWS], tl.int32)
    mean_gap = tl.zeros([ROWS], tl.float64)
    mean_factor = tl.full([ROWS], 1.0, tl.float64)
    free = maximum
    # The backward's capped count and sum of exp(z - free) less one, ties at free included.
    above = active
    spread = others
    need = (threshold + eps * tl.maximum(tl.abs(threshold), 1.0) < 0) & ~bad & live
    if SINGLE and ROWS == 1 and not CANDIDATES:
        if not tl.reshape(need, ()):
            loss, _, _ = label_statistics(maximum - z_y, 1 + others, True)
            return (
                tl.where(few, 0.0, loss),
                tl.where(few, 0.0, free),
                tl.where(few, 1.0, (1 + others) / K),
                bad,
                unsure,
            )
    if tl.max(need.to(tl.int32), 0) > 0:
        if CANDIDATES:
            top, largest, _, _ = _select(Z, row, live, y, x1, ROWS, C, B, KB, K, XT, False)
            last = _at(top, kk, K - 1)
            unsure = unsure | (need & ~(left <= last))
            # The tail beyond the K largest: candidates past them and the chunks' rests.
            ordered = tl.sort(x1, 1, descending=True).to(tl.float64)
            past = (tl.arange(0, B)[None, :] >= K) & (ordered > -float("inf"))
            reach = finite(last)[:, None]
            tail = tl.sum(tl.where(past & reach, exp(ordered - last[:, None]), 0.0), 1)
            moved = tl.where((peaks > -float("inf")) & reach, exp(peaks - last[:, None]), 0.0)
            tail += tl.sum(rest * moved, 1)
        else:
            top, largest, tail, _ = _select(Z, row, live, y, x1, ROWS, C, B, KB, K, XT, True)
            last = _at(top, kk, K - 1)
        few = few | (need & (last == -float("inf")))
        active = tl.where(need, 1, 0)
        running = need
        if SINGLE and ROWS == 1 and K <= 8 and XT == tl.float32:
            if tl.reshape(finite(last) & (maximum - last <= 32.0), ()):
                index, score, norm, remainder, ready = _bounded_cap(top, last, tail, kk, K)
                active = tl.where(ready, index, active)
                free = tl.where(ready, score, free)
                lse = tl.where(ready, norm, lse)
                spread = tl.where(ready, remainder, spread)
                running = running & ~ready
        while tl.max(running.to(tl.int32), 0) > 0:
            candidate = _atv(top, kk, active)
            scale = exp(last - candidate)
            # The log-sum-exp of the scores from position active on, relative to candidate.
            mid = tl.where((kk > active[:, None]) & (kk < K), exp(top - candidate[:, None]), 0.0)
            ties_above = (kk < active[:, None]) & (top == candidate[:, None])
            c_lse = log1p(tl.sum(mid, 1) + scale * tail)
            c_spread = tl.sum(mid + ties_above.to(tl.float64), 1) + scale * tail
            tailed = (scale > 0) & finite(tail)
            if CANDIDATES:
                unsure = unsure | (running & ~tailed)
            elif tl.max((running & ~tailed).to(tl.int32), 0) > 0:
                # The entries below the candidate and its later copies.
                level = candidate.to(XT)[:, None]
                copies = tl.zeros([ROWS], tl.int32)
                below = tl.zeros([ROWS], tl.float64)
                for start in range(0, C, B):
                    if SINGLE:
                        x = x1
                    else:
                        x = _scores(Z, row, live, y, start, C, B, XT)
                    copies += tl.sum((x == level).to(tl.int32), 1)
                    below += tl.sum(tl.where(x < level, exp(x - level), 0.0).to(tl.float64), 1)
                tied = tl.sum(ties_above.to(tl.int32), 1)
                c_lse = tl.where(tailed, c_lse, log1p(below + (copies - tied - 1).to(tl.float64)))
                c_spread = tl.where(tailed, c_spread, below + (copies - 1).to(tl.float64))
            free = tl.where(running, candidate, free)
            lse = tl.where(running, c_lse, lse)
            spread = tl.where(running, c_spread, spread)
            threshold = c_lse - log((K - active).to(tl.float64))
            capped = threshold + eps * tl.maximum(tl.abs(threshold), 1.0) < 0
            running = running & capped & (active < K - 1)
            active = tl.where(running, active + 1, active)
        above = tl.where(need, tl.sum(((kk < K) & (top > free[:, None])).to(tl.int32), 1), 0)
        # Separate the score mean from the free group's log-mass before they cancel
        # at large spreads. There are K-active copies of the free threshold in the mean.
        scores = tl.where(kk < active[:, None], top, free[:, None])
        total, carry = _score_sum(scores, (kk < K) & finite(scores), K)
        mean_gap, mean_factor = _entropy_mean(total, carry, 0.0, z_y, K)
    remaining = (K - active).to(tl.float64)
    # Keep the bounded free-group correction as a multiplier of exp(mean_gap).
    # Adding it to a large negative score gap would lose bits before exponentiation.
    free_weight = (K * mean_factor) * exp((remaining / K) * (lse - log(remaining)))
    capped_loss, _, _ = label_statistics(mean_gap, free_weight)
    uncapped, _, _ = label_statistics(maximum - z_y, 1 + others)
    loss = tl.where(active == 0, uncapped, capped_loss)
    second = (1 + spread) / (K - above).to(tl.float64)
    return (
        tl.where(few, 0.0, loss),
        tl.where(few, 0.0, free),
        tl.where(few, 1.0, second),
        bad,
        unsure,
    )


@triton.jit
def _label_limit(Z, row, live, y, z_y, x1, gamma, ROWS: tl.constexpr, C: tl.constexpr,
                 B: tl.constexpr, K: tl.constexpr, LOSS: tl.constexpr, XT: tl.constexpr):  # fmt: skip
    """For masked labels, count remaining competitors and place the label below the hinge cap."""
    masked = live & (z_y == -float("inf"))
    count = tl.full([ROWS], C - 1, tl.int32)
    if tl.max(masked.to(tl.int32), 0) != 0:
        count = tl.zeros([ROWS], tl.int32)
        low = tl.full([ROWS], float("inf"), tl.float64)
        for start in range(0, C, B):
            if B >= C:
                x = x1
            else:
                x = _scores(Z, row, masked, y, start, C, B, XT)
            left = x > -float("inf")
            count += tl.sum(left.to(tl.int32), 1)
            if LOSS != ENTROPY:
                low = tl.minimum(low, tl.min(tl.where(left, x.to(tl.float64), float("inf")), 1))
        if LOSS != ENTROPY:
            z_y = tl.where(masked, low - gamma / K, z_y)
    enough = count > 0 if LOSS == BETA or LOSS == BETA_SMOOTH else count >= K
    return z_y, masked & enough, masked & (count == 0)


@triton.jit
def _max_hinge(Z, row, live, y, z_y, x1, limit, bad, ROWS: tl.constexpr, C: tl.constexpr,
               B: tl.constexpr, XT: tl.constexpr, GRAD: tl.constexpr, INDEX: tl.constexpr):  # fmt: skip
    """Maximum competitor, its tie share and the hinge value for both k=1 losses."""
    largest = tl.full([ROWS], -float("inf"), XT)
    tied = tl.zeros([ROWS], tl.int32)
    winner = tl.zeros([ROWS], tl.int32)
    for start in range(0, C, B):
        if B >= C:
            x = x1
        else:
            x = _scores(Z, row, live, y, start, C, B, XT)
        best = tl.max(x, 1)
        if INDEX and GRAD:
            col = start + tl.arange(0, B)[None, :]
            # Only a unique maximum uses this index. Tied maxima use the full count.
            position = tl.max(tl.where(x == best[:, None], col, 0), 1)
            winner = tl.where(best > largest, position, winner)
        if GRAD:
            count = tl.sum((x == best[:, None]).to(tl.int32), 1)
            tied = tl.where(best > largest, count, tied + tl.where(best == largest, count, 0))
        largest = tl.maximum(largest, best)
    bad = bad | (largest == float("inf")) | (limit & (largest == -float("inf")))
    first = largest.to(tl.float64)
    h = margin(first, z_y)
    value = tl.maximum(h, 0.0)
    second = 1.0 / tl.maximum(tied + (h == 0).to(tl.int32), 1).to(tl.float64)
    third = (h >= 0).to(tl.float64)
    if B >= C:
        # Store the common competitor slope and the total mass separately.
        # At the hinge boundary the label shares the tie with the competitors.
        second = tl.where(h >= 0, second, 0.0)
        third = tl.where(h == 0, tied.to(tl.float64) / (tied + 1), (h > 0).to(tl.float64))
    if INDEX and GRAD:
        unique = (tied == 1) & finite(first)
        second = tl.where(unique, (-winner - 1).to(tl.float64), second)
        third = tl.where(unique & (h == 0), 0.5, third)
    return value, first, second, third, bad


@triton.jit
def _write_summary(V, S, row, live, w, ignored, limit, bad, value, first, second, third,
                   VALUES: tl.constexpr, GRAD: tl.constexpr, STRIDE: tl.constexpr = 3):  # fmt: skip
    """Store the weighted row and the three fields consumed by its backward."""
    value = tl.where(limit, float("inf"), value)
    bad = (bad | ((w == 0) & (value == float("inf")))) & ~ignored
    nan = float("nan")
    if VALUES:
        value = tl.where(ignored | (w == 0), 0.0, value * w)
        tl.store(V + row, tl.where(bad, nan, value).to(V.dtype.element_ty), live)
    if GRAD:
        tl.store(S + row * STRIDE, first, live)
        tl.store(S + row * STRIDE + 1, second, live)
        tl.store(S + row * STRIDE + 2, tl.where(bad, nan, tl.where(ignored, 0.0, third)), live)


@triton.jit(do_not_specialize=["N", "GROUP", "IGNORE", "GAMMA"])
def _forward(
    Z,
    Y,
    W,
    V,
    S,
    F,
    N,
    GROUP,
    IGNORE,
    GAMMA,
    C: tl.constexpr,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    KB: tl.constexpr,
    LOSS: tl.constexpr,
    FP32: tl.constexpr,
    WEIGHTED: tl.constexpr,
    IGNORED: tl.constexpr,
    VALUES: tl.constexpr,
    GRAD: tl.constexpr,
    INDEX: tl.constexpr = False,
    SHORT_SIMPLEX: tl.constexpr = False,
    GATED: tl.constexpr = False,
    STRIDE: tl.constexpr = 3,
):
    """Row values (times the label's weight) and three statistics per row (NaN third for
    invalid rows): entropy (free, D, loss), nonsmooth hinges (k-th score, cutoff share,
    hinge slope; beta at k > 1: the label score) or, for a single maximum in registers,
    (score, competitor slope, total mass).
    Smooth hinges use (origin, correction, effective label score) or, for an isolated top-k
    support, (score cutoff, common gradient, -inf). GATED: only the rows _solve marked in F
    (one row per program)."""
    if GATED:
        if tl.load(doubles(F) + tl.program_id(0)) == 0:
            return
    S = doubles(S)
    gamma = f64(GAMMA)
    XT: tl.constexpr = (
        tl.float32 if Z.dtype.element_ty != tl.float64 and (FP32 or LOSS != ENTROPY) else tl.float64
    )
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = row < N
    y, inside, ignored, w = target_row(Y, W, row, live, C, IGNORE, GROUP, WEIGHTED, IGNORED)
    z_y = tl.load(Z + row * C + y, live, other=0).to(tl.float64)
    bad = ~inside | ~(z_y < float("inf")) | ~finite(w)
    x1 = _scores(Z, row, live, y, 0, C, B, XT)
    if (LOSS == ALPHA or LOSS == BETA) and K == 1:
        limit = live & (z_y == -float("inf"))
    else:
        z_y, limit, empty = _label_limit(Z, row, live, y, z_y, x1, gamma, ROWS, C, B, K, LOSS, XT)
        bad = bad | empty
    if LOSS == ENTROPY:
        if K == C - 1:
            value, first, second, bad = _all_capped(
                Z, row, live, y, z_y, x1, bad, ROWS, C, B, K, XT
            )
        else:
            value, first, second, bad = _entropy(
                Z, row, live, y, z_y, x1, bad, ROWS, C, B, K, KB, XT
            )
        third = value
    elif (LOSS == ALPHA or LOSS == BETA) and K == 1:
        value, first, second, third, bad = _max_hinge(
            Z, row, live, y, z_y, x1, limit, bad, ROWS, C, B, XT, GRAD, INDEX
        )
    elif LOSS == ALPHA or LOSS == BETA:
        kk = tl.arange(0, KB)[None, :]
        top, largest, _, tied = _select(
            Z, row, live, y, x1, ROWS, C, B, KB, K, XT, False, GRAD and B < C
        )
        bad = bad | (largest == float("inf"))
        h = margin(top, z_y[:, None])
        if LOSS == BETA:
            # A positive margin can overflow before its share is divided by K.
            terms = tl.where(
                h == float("inf"), top / K - z_y[:, None] / K + 1.0 / K, tl.maximum(h, 0.0) / K
            )
            total = tl.sum(tl.where(kk < K, terms, 0.0), 1)
        else:
            high, low = _score_sum(top, (kk < K) & finite(top), K)
            total = tl.where(
                _at(top, kk, K - 1) > -float("inf"),
                _mean_margin(high, low, z_y, K),
                -float("inf"),
            )
        value = tl.maximum(total, 0.0)
        first = _at(top, kk, K - 1)
        second = tl.full([ROWS], 1.0, tl.float64)
        if GRAD:
            places = tl.sum(((kk < K) & (top == first[:, None])).to(tl.int32), 1)
            if B >= C:
                tied = tl.sum((x1 == first[:, None]).to(tl.int32), 1)
            second = places.to(tl.float64) / tl.maximum(tied, 1).to(tl.float64)
        if LOSS == ALPHA:
            third = tl.where(total == 0, 0.5, (total > 0).to(tl.float64))
        else:
            # Beta's entries take their own slopes: third keeps the label score for them.
            third = z_y
    elif SHORT_SIMPLEX:
        value, first, second, third, bad = _short_simplex(x1, z_y, bad, gamma, B)
    elif K == 1 and B < C and XT != tl.float64:
        # float64 scores can overflow their margins; the general solver handles that.
        value, first, second, third, bad = _smooth_k1(
            Z, row, live, y, z_y, bad, gamma, ROWS, C, B, XT
        )
    else:
        value, first, second, third, bad = _smooth(
            Z, row, live, y, z_y, x1, bad, gamma, ROWS, C, B, K, KB, LOSS, XT
        )
    _write_summary(
        V, S, row, live, w, ignored, limit, bad, value, first, second, third, VALUES, GRAD, STRIDE
    )


# A chunk's statistics (_candidates): its peak and two sums.
_PEAK, _SUM, _VALUE = tl.constexpr(0), tl.constexpr(1), tl.constexpr(2)
_FIELDS = tl.constexpr(3)


@triton.jit(do_not_specialize=["GAMMA"])
def _candidates(Z, Y, P, GAMMA, C: tl.constexpr, B: tl.constexpr, R: tl.constexpr,
                K: tl.constexpr, LOSS: tl.constexpr, CHUNKS: tl.constexpr):  # fmt: skip
    """Statistics of R row chunks of B columns per program (one warp each, a row tile [R, B]
    in aligned windows of R B columns, rows.py): each chunk's largest other score (NaN and
    +inf as +inf), the entropy's sum of exp(z - peak) over its other scores (peak its
    largest), and for the smooth beta hinge (k = 1 included) the sums of the box a = clip(h,
    0, gamma / K) and of gamma times the value at it over its other margins."""
    P = doubles(P)
    gamma = f64(GAMMA)
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    y = tl.load(Y + row + tl.arange(0, 1)).to(tl.int64)
    y = tl.minimum(tl.maximum(y, 0), C - 1).to(tl.int32)
    if LOSS == BETA_SMOOTH:
        z_y = tl.load(Z + row * C + y).to(tl.float64)
    else:
        z_y = tl.zeros([1], tl.float64)
    shift = rows.misaligned(Z, row, C)
    _, _, _, inside = rows.span(block, shift, shift == shift, R * B, C)
    j = block * (R * B) + tl.arange(0, R)[:, None] * B + tl.arange(0, B)[None, :]
    k = j - shift
    valid = (k >= 0) & (k < C)
    if inside:
        _candidate_chunk(Z, P, row, block, j, k, shift, valid, y, z_y, gamma, C, B, R, K,
                         LOSS, CHUNKS, False)  # fmt: skip
    else:
        _candidate_chunk(Z, P, row, block, j, k, shift, valid, y, z_y, gamma, C, B, R, K,
                         LOSS, CHUNKS, True)  # fmt: skip


@triton.jit
def _candidate_chunk(Z, P, row, block, j, k, shift, valid, y, z_y, gamma, C: tl.constexpr,
                     B: tl.constexpr, R: tl.constexpr, K: tl.constexpr, LOSS: tl.constexpr,
                     CHUNKS: tl.constexpr, MASKED: tl.constexpr):  # fmt: skip
    z = rows.load(Z, row, j, shift, valid, C, MASKED).to(tl.float32)
    other = valid & (k != y[:, None])
    x = tl.where(other, tl.where(z < float("inf"), z, float("inf")), -float("inf"))
    peak = tl.max(x, 1)
    field = P + (row * CHUNKS + block * R + tl.arange(0, R)[:, None]) * _FIELDS
    tl.store(field + _PEAK, peak.to(tl.float64)[:, None])
    # Float32 sums over a chunk (pairwise, at most B terms), float64 across chunks.
    if LOSS == ENTROPY:
        e = tl.where(x > -float("inf"), exp32(x - peak[:, None]), 0.0)
        tl.store(field + _SUM, tl.sum(e, 1).to(tl.float64)[:, None])
    if LOSS == BETA_SMOOTH:
        # Float32 work: the margins, the box and its terms per entry, float64 sums.
        h = x - (z_y - 1.0).to(tl.float32)[:, None]
        cap = (gamma / K).to(tl.float32)
        a = tl.where(x > -float("inf"), tl.minimum(tl.maximum(h, 0.0), cap), 0.0)
        terms = tl.where(a > 0, a * (h - 0.5 * a), 0.0)
        tl.store(field + _SUM, tl.sum(a, 1).to(tl.float64)[:, None])
        tl.store(field + _VALUE, tl.sum(terms, 1).to(tl.float64)[:, None])


@triton.jit
def _chosen(P, row, live, CHUNKS: tl.constexpr, CP: tl.constexpr, KS: tl.constexpr,
            SEL: tl.constexpr):  # fmt: skip
    """A row's chunk peaks, which exist, and the chunks it reads again: those above the KS-th
    largest peak (fewer than KS) and the first at it, up to SEL (copies of the level a full
    set leaves count among the scores the candidates leave)."""
    p = tl.arange(0, CP)[None, :]
    there = live[:, None] & (p < CHUNKS)
    peaks = tl.load(P + row[:, None] * (CHUNKS * _FIELDS) + p * _FIELDS + _PEAK, there,
                    other=-float("inf"))  # fmt: skip
    level = _kth(peaks, KS)[:, None]
    above = there & (peaks > level)
    tied = there & (peaks == level)
    room = SEL - tl.sum(above.to(tl.int32), 1)[:, None]
    chosen = above | (tied & (tl.cumsum(tied.to(tl.int32), 1) <= room))
    return p, there, peaks, chosen


@triton.jit
def _gather(Z, Y, P, Q, N, C: tl.constexpr, B: tl.constexpr, KS: tl.constexpr,
            E: tl.constexpr, SEL: tl.constexpr, GS: tl.constexpr, LOSS: tl.constexpr,
            CHUNKS: tl.constexpr, CP: tl.constexpr):  # fmt: skip
    """The E largest distinct other scores of a row's chosen chunks (GS slots per program,
    one per warp: the slot-th chosen chunk; none past the chosen) with their counts and
    columns (one of the copies), the largest score they leave and, for the entropy, the sum
    of exp(z - it) over the scores they leave."""
    Q = doubles(Q)
    P = doubles(P)
    row = tl.program_id(0).to(tl.int64) + tl.arange(0, 1)
    slot = tl.program_id(1) * GS + tl.arange(0, GS)[:, None]
    live = row < N
    p, there, peaks, chosen = _chosen(P, row, live, CHUNKS, CP, KS, SEL)
    rank = tl.cumsum(chosen.to(tl.int32), 1) - 1
    chunk = tl.max(tl.where(chosen & (rank == slot), p, -1), 1)[:, None]
    y = tl.load(Y + row, live, other=0).to(tl.int64)
    y = tl.minimum(tl.maximum(y, 0), C - 1).to(tl.int32)
    shift = rows.misaligned(Z, tl.sum(row, 0), C)
    lane = tl.arange(0, B)[None, :]
    k = chunk * B + lane - shift
    ok = live[:, None] & (chunk >= 0) & (k >= 0) & (k < C) & (k != y[:, None])
    z = tl.load(Z + row[:, None] * C + k, ok, other=0).to(tl.float32)
    x = tl.where(ok, tl.where(z < float("inf"), z, float("inf")), -float("inf"))
    mm = tl.arange(0, E)[None, :]
    vals = tl.full([GS, E], -float("inf"), tl.float32)
    counts = tl.zeros([GS, E], tl.int32)
    columns = tl.full([GS, E], -1, tl.int32)
    for i in tl.static_range(E):
        v = tl.max(x, 1)[:, None]
        hit = (x == v) & (v > -float("inf"))
        vals = tl.where(mm == i, v, vals)
        counts = tl.where(mm == i, tl.sum(hit.to(tl.int32), 1)[:, None], counts)
        at = tl.max(tl.where(hit, lane, -1), 1)[:, None]
        columns = tl.where(mm == i, tl.where(at >= 0, chunk * B + at - shift, -1), columns)
        x = tl.where(hit, -float("inf"), x)
    rest = tl.max(x, 1)
    out = Q + (row[:, None] * SEL + slot) * (3 * E + 2)
    tl.store(out + mm, vals.to(tl.float64), live[:, None])
    tl.store(out + E + mm, counts.to(tl.float64), live[:, None])
    tl.store(out + 2 * E + mm, columns.to(tl.float64), live[:, None])
    tl.store(out + 3 * E, rest.to(tl.float64)[:, None], live[:, None])
    if LOSS == ENTROPY:
        tail = tl.where(x > -float("inf"), exp32(x - rest[:, None]), 0.0)
        tl.store(out + 3 * E + 1, tl.sum(tail.to(tl.float64), 1)[:, None], live[:, None])


@triton.jit(do_not_specialize=["N", "GROUP", "IGNORE", "GAMMA"])
def _solve(Z, Y, W, P, Q, V, S, F, N, GROUP, IGNORE, GAMMA, C: tl.constexpr, B: tl.constexpr,
           K: tl.constexpr, KB: tl.constexpr, KS: tl.constexpr, E: tl.constexpr, M: tl.constexpr,
           SEL: tl.constexpr, LOSS: tl.constexpr, CHUNKS: tl.constexpr, CP: tl.constexpr,
           WEIGHTED: tl.constexpr, IGNORED: tl.constexpr, VALUES: tl.constexpr,
           GRAD: tl.constexpr, INDEX: tl.constexpr, STRIDE: tl.constexpr):  # fmt: skip
    """A streamed row's value and statistics from its chunks (_candidates), as _forward
    computes them over the whole row. The chunks whose peaks reach the KS-th largest peak
    hold the KS largest scores: _gather reads them again (at most SEL chunks), and the
    solvers run on their M largest as on a register row. Where scores outside them could
    change the result (a support reaching them, more tied peaks, a masked label), F marks
    the row for _forward's general path."""
    S = doubles(S)
    P = doubles(P)
    Q = doubles(Q)
    F = doubles(F)
    gamma = f64(GAMMA)
    row = tl.program_id(0).to(tl.int64) + tl.arange(0, 1)
    live = row < N
    y, inside, ignored, w = target_row(Y, W, row, live, C, IGNORE, GROUP, WEIGHTED, IGNORED)
    z_y = tl.load(Z + row * C + y, live, other=0).to(tl.float64)
    bad = ~inside | ~(z_y < float("inf")) | ~finite(w)
    p, there, peaks, chosen = _chosen(P, row, live, CHUNKS, CP, KS, SEL)
    fields = P + row[:, None] * (CHUNKS * _FIELDS) + p * _FIELDS
    # The chosen chunks' distinct values and counts (_gather), merged: the M largest scores
    # x1 (ties repeated), and the copies of the merged values in all chosen chunks.
    sl = tl.arange(0, SEL)[:, None]
    ee = tl.arange(0, E)[None, :]
    mm = tl.arange(0, M)[None, :]
    record = Q + (row[:, None] * SEL + sl) * (3 * E + 2)
    pool = tl.load(record + ee, live[:, None], other=-float("inf")).to(tl.float32)
    counts = tl.load(record + E + ee, live[:, None], other=0.0).to(tl.int32)
    columns = tl.load(record + 2 * E + ee, live[:, None], other=-1.0).to(tl.int32)
    rests = tl.load(record + 3 * E, live[:, None], other=-float("inf"))
    # The column of each candidate where it is the only copy among the chosen chunks.
    spots = tl.full([1, M], -1, tl.int32)
    x1 = tl.full([1, M], -float("inf"), tl.float32)
    distinct = tl.full([1, M], -float("inf"), tl.float32)
    copies = tl.zeros([1, M], tl.int32)
    filled = tl.zeros([1, 1], tl.int32)
    # Copies past the M slots: the largest such value, and the sum of exp(z - it) over them.
    spill = tl.full([1, 1], -float("inf"), tl.float32)
    spilled = tl.zeros([1, 1], tl.float64)
    for i in tl.static_range(M):
        v = tl.max(tl.max(pool, 1), 0)
        c = tl.sum(tl.sum(tl.where(pool == v, counts, 0), 1), 0)
        x1 = tl.where((mm >= filled) & (mm < filled + c), v, x1)
        only = tl.max(tl.max(tl.where(pool == v, columns, -1), 1), 0)
        spots = tl.where((mm == filled) & (c == 1), only, spots)
        distinct = tl.where(mm == i, v, distinct)
        copies = tl.where(mm == i, c, copies)
        over = tl.minimum(tl.maximum(filled + c - M, 0), c)
        spill = tl.where((over > 0) & (spill == -float("inf")), v, spill)
        if LOSS == ENTROPY:
            gap = tl.where(over > 0, v.to(tl.float64) - spill.to(tl.float64), 0.0)
            spilled += tl.where(over > 0, over.to(tl.float64) * exp(gap), 0.0)
        filled += c
        pool = tl.where(pool == v, -float("inf"), pool)
    # The largest score the candidates leave: it decides every threshold below them.
    leftover = tl.max(tl.max(pool, 1), 0)
    rest_peak = tl.maximum(tl.max(tl.max(rests, 1), 0), tl.maximum(leftover, spill).to(tl.float64))
    rest_peak = tl.reshape(rest_peak, (1,))
    left = tl.maximum(tl.max(tl.where(chosen, -float("inf"), peaks), 1), rest_peak)
    limit = live & (z_y == -float("inf"))
    if LOSS == ENTROPY:
        c = tl.arange(0, M)[None, :]
        largest, at = tl.max(x1, 1, return_indices=True, return_indices_tie_break_left=True)
        maximum = largest.to(tl.float64)
        # Outside the candidates: the other chunks and the chosen chunks' rest (in the
        # slot of the first chosen chunk), as peaks and sums of exp(z - peak).
        sums = tl.load(fields + _SUM, there, other=0.0)
        first_chosen = tl.min(tl.where(chosen, p, CP), 1)[:, None]
        tails = tl.load(record + 3 * E + 1, live[:, None], other=0.0)
        moved = tl.where(rests > -float("inf"), exp(rests - rest_peak[:, None]), 0.0)
        tail = tl.sum(tl.sum(tails * moved, 1), 0)
        spare = tl.where(pool > -float("inf"), exp(pool.to(tl.float64) - rest_peak[:, None]), 0.0)
        tail += tl.sum(tl.sum(spare * counts.to(tl.float64), 1), 0)
        overflow = spill.to(tl.float64) - rest_peak[:, None]
        tail += tl.sum(tl.where(spill > -float("inf"), exp(overflow), 0.0) * spilled, 1)
        peaks = tl.where(p == first_chosen, rest_peak, tl.where(chosen, -float("inf"), peaks))
        rest = tl.where(p == first_chosen, tail, tl.where(chosen, 0.0, sums))
        # The sum of exp(z - maximum) over the others but one copy of the maximum.
        e = tl.where(
            (x1 > -float("inf")) & (c != at[:, None]),
            exp(x1.to(tl.float64) - maximum[:, None]),
            0.0,
        )
        moved = tl.where(peaks > -float("inf"), exp(peaks - maximum[:, None]), 0.0)
        others = tl.sum(e, 1) + tl.sum(rest * moved, 1)
        value, first, second, bad, unsure = _entropy_capped(
            Z, row, live, y, z_y, x1, bad, maximum, others, rest, peaks, left, 1, M, M, K,
            KB, tl.float32, True,
        )  # fmt: skip
        third = value
        # A masked label needs the rest's count.
        general = limit | (live & ~bad & unsure)
    elif (LOSS == ALPHA or LOSS == BETA) and K == 1:
        largest = tl.max(x1, 1)
        tied = tl.sum(tl.where(distinct == largest[:, None], copies, 0), 1)
        bad = bad | (largest == float("inf")) | (limit & (largest == -float("inf")))
        first = largest.to(tl.float64)
        h = margin(first, z_y)
        value = tl.maximum(h, 0.0)
        second = 1.0 / tl.maximum(tied + (h == 0).to(tl.int32), 1).to(tl.float64)
        third = (h >= 0).to(tl.float64)
        if INDEX:
            winner = tl.max(tl.where(mm == 0, spots, -1), 1).to(tl.float64)
            unique = (tied == 1) & finite(first)
            second = tl.where(unique, -winner - 1.0, second)
            third = tl.where(unique & (h == 0), 0.5, third)
        # Copies of the maximum outside the chosen chunks would share its slope.
        unchosen = tl.max(tl.where(chosen, -float("inf"), peaks), 1)
        general = live & ~bad & ~(unchosen < first) & (largest > -float("inf"))
    elif LOSS == ALPHA or LOSS == BETA:
        kk = tl.arange(0, KB)[None, :]
        top, largest, _, _ = _select(Z, row, live, y, x1, 1, M, M, KB, K, tl.float32, False)
        bad = bad | (largest == float("inf"))
        h = margin(top, z_y[:, None])
        if LOSS == BETA:
            terms = tl.where(
                h == float("inf"), top / K - z_y[:, None] / K + 1.0 / K, tl.maximum(h, 0.0) / K
            )
            total = tl.sum(tl.where(kk < K, terms, 0.0), 1)
        else:
            high, low = _score_sum(top, (kk < K) & finite(top), K)
            total = tl.where(
                _at(top, kk, K - 1) > -float("inf"),
                _mean_margin(high, low, z_y, K),
                -float("inf"),
            )
        value = tl.maximum(total, 0.0)
        first = _at(top, kk, K - 1)
        places = tl.sum(((kk < K) & (top == first[:, None])).to(tl.int32), 1)
        tied = tl.sum(tl.where(distinct == first.to(tl.float32)[:, None], copies, 0), 1)
        second = places.to(tl.float64) / tl.maximum(tied, 1).to(tl.float64)
        if LOSS == ALPHA:
            third = tl.where(total == 0, 0.5, (total > 0).to(tl.float64))
        else:
            third = z_y
        # A masked label needs the rest's count; copies of a finite cutoff outside the
        # chosen chunks would share its slope.
        unchosen = tl.max(tl.where(chosen, -float("inf"), peaks), 1)
        general = limit | (live & ~bad & ~(unchosen < first) & (first > -float("inf")))
    else:
        value, first, second, third, bad, unsure = _smooth_candidates(
            Z, row, live, y, z_y, x1, bad, left, fields, there, gamma, M, K, KB, LOSS
        )
        general = limit | (live & ~bad & unsure)
    _write_summary(
        V, S, row, live, w, ignored, limit, bad, value, first, second, third, VALUES, GRAD, STRIDE
    )
    tl.store(F + row, general.to(tl.float64), live)
    if GRAD and STRIDE > 3:
        # Rows the candidates decide have their nonzero entries in the chosen chunks: the
        # gradient clears the others unread and computes these (_chunk_heavy).
        base = S + row[:, None] * STRIDE + 3
        lead = rows.misaligned(Z, tl.sum(row, 0), C).to(tl.float64)
        # An indexed hinge's unique winner (a negative second) writes in _chunk_backward.
        decided = ~general & ~bad
        if INDEX:
            decided = decided & ~(second < 0)
        tl.store(base, decided.to(tl.float64)[:, None], live[:, None])
        tl.store(base + 1, lead + tl.zeros([1, 1], tl.float64), live[:, None])
        rank = tl.cumsum(chosen.to(tl.int32), 1) - 1
        slots = tl.arange(0, SEL)[None, :]
        listed = tl.max(tl.where(chosen & (rank == tl.arange(0, SEL)[:, None]), p, -1), 1)
        tl.store(base + 2 + slots, tl.reshape(listed, (1, SEL)).to(tl.float64), live[:, None])


@triton.jit
def _entropy_partial(Z, Y, P, C: tl.constexpr, B: tl.constexpr, K: tl.constexpr,
                     KP: tl.constexpr, CHUNKS: tl.constexpr):  # fmt: skip
    """A row chunk's statistics for the top-k entropy (one program per aligned chunk of B
    columns, rows.py): the K largest distinct other scores at least a level (NaN and +inf as
    +inf; -inf pads) with their counts, which are at least K together, the level and the float32
    sum of exp(z - level) over the scores below it. The K largest of a row are among its
    chunks' scores at their levels."""
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    y = tl.load(Y + row + tl.arange(0, 1)).to(tl.int64)
    y = tl.minimum(tl.maximum(y, 0), C - 1).to(tl.int32)
    shift = rows.misaligned(Z, row, C)
    j, k, valid, inside = rows.span(block, shift, shift == shift, B, C)
    if inside:
        _entropy_chunk(Z, P, row, block, j, k, shift, valid, y, C, B, K, KP, CHUNKS, False)
    else:
        _entropy_chunk(Z, P, row, block, j, k, shift, valid, y, C, B, K, KP, CHUNKS, True)


@triton.jit
def _distinct(lanes, K: tl.constexpr, KP: tl.constexpr):
    """The K largest distinct finite values of lanes ([1, L]) at least the level, the K-th
    largest lane (ties repeated; the smallest finite lane if fewer, -inf if none): values,
    their counts ([1, KP]) and the number of lanes at least the level."""
    kk = tl.arange(0, KP)[None, :]
    value = tl.max(lanes, 1)
    filled = tl.sum((lanes == value[:, None]).to(tl.int32), 1)
    filled = tl.where(value > -float("inf"), filled, 0)
    values = tl.where(kk == 0, value[:, None], -float("inf"))
    counts = tl.where(kk == 0, filled[:, None], 0)
    for i in tl.static_range(1, K):
        following = tl.max(tl.where(lanes < value[:, None], lanes, -float("inf")), 1)
        go = (filled < K) & (following > -float("inf"))
        c = tl.sum((lanes == following[:, None]).to(tl.int32), 1)
        values = tl.where((kk == i) & go[:, None], following[:, None], values)
        counts = tl.where((kk == i) & go[:, None], c[:, None], counts)
        value = tl.where(go, following, value)
        filled += tl.where(go, c, 0)
    return values, counts, value, filled


@triton.jit
def _entropy_chunk(Z, P, row, block, j, k, shift, valid, y, C: tl.constexpr, B: tl.constexpr,
                   K: tl.constexpr, KP: tl.constexpr, CHUNKS: tl.constexpr,
                   MASKED: tl.constexpr):  # fmt: skip
    """The K largest maxima of the chunk's vectors (G scores of 16 bytes, maxima within a
    thread) set the level; where no vector holds a second score at it or above, the scores
    at least the level are those maxima. Else (ties, close scores) the K largest scores."""
    G: tl.constexpr = 16 // (Z.dtype.element_ty.primitive_bitwidth // 8)
    z = rows.load(Z, row, j, shift, valid, C, MASKED).to(tl.float32)
    x = tl.where(valid & (k != y[:, None]), tl.where(z < float("inf"), z, float("inf")),
                 -float("inf"))  # fmt: skip
    lanes = tl.reshape(tl.max(tl.reshape(x, [B // G, G]), 1), [1, B // G])
    values, counts, level, filled = _distinct(lanes, K, KP)
    keep = (x >= level[:, None]) & (x > -float("inf"))
    if tl.reshape(tl.sum(keep.to(tl.int32), 1) != filled, ()):
        values, counts, level, filled = _distinct(x, K, KP)
        keep = (x >= level[:, None]) & (x > -float("inf"))
    rest = tl.sum(tl.where(keep | (x == -float("inf")), 0.0, tl.exp(x - level[:, None])), 1)
    out = P + (row * CHUNKS + block) * (2 * K + 2)
    kk = tl.arange(0, KP)[None, :]
    tl.store(out + kk, values, kk < K)
    tl.store(out + K + kk, counts.to(tl.float32), kk < K)
    tl.store(out + 2 * K + tl.arange(0, 1), level)
    tl.store(out + 2 * K + 1 + tl.arange(0, 1), rest)


@triton.jit(do_not_specialize=["N", "GROUP", "IGNORE"])
def _entropy_merge(Z, Y, W, P, V, S, F, N, GROUP, IGNORE, C: tl.constexpr, K: tl.constexpr,
                   KB: tl.constexpr, KP: tl.constexpr, CHUNKS: tl.constexpr, CP: tl.constexpr,
                   WEIGHTED: tl.constexpr, IGNORED: tl.constexpr, VALUES: tl.constexpr,
                   GRAD: tl.constexpr, STRIDE: tl.constexpr):  # fmt: skip
    """A row's top-k entropy from its chunks (_entropy_partial), merged in float64: the K
    largest other scores (ties repeated) and the sum of exp(z - left) over the others, left
    the largest score they leave, solved as _forward's _entropy solves a row. F marks the rows
    the merge cannot decide (a masked label, a tail out of float64 range) for _forward."""
    S = doubles(S)
    F = doubles(F)
    row = tl.program_id(0).to(tl.int64) + tl.arange(0, 1)
    live = row < N
    y, inside, ignored, w = target_row(Y, W, row, live, C, IGNORE, GROUP, WEIGHTED, IGNORED)
    z_y = tl.load(Z + row * C + y, live, other=0).to(tl.float64)
    bad = ~inside | ~(z_y < float("inf")) | ~finite(w)
    p = tl.arange(0, CP)[:, None]
    i = tl.arange(0, KP)[None, :]
    base = P + (tl.program_id(0).to(tl.int64) * CHUNKS + p) * (2 * K + 2)
    there = (p < CHUNKS) & (i < K)
    pool = tl.load(base + i, there, other=-float("inf"))
    counts = tl.load(base + K + i, there, other=0.0)
    peaks = tl.load(base + 2 * K, p < CHUNKS, other=-float("inf"))
    sums = tl.load(base + 2 * K + 1, p < CHUNKS, other=0.0)
    # The K largest with their copies; copies past K spill (relative to the first spilled).
    kk = tl.arange(0, KB)[None, :]
    x1 = tl.full([1, KB], -float("inf"), tl.float32)
    filled = tl.zeros([1, 1], tl.float32)
    spill = tl.full([1, 1], -float("inf"), tl.float32)
    spilled = tl.zeros([1, 1], tl.float64)
    for _ in tl.static_range(K):
        v = tl.max(tl.max(pool, 1, keep_dims=True), 0, keep_dims=True)
        hit = pool == v
        c = tl.sum(tl.sum(tl.where(hit, counts, 0.0), 1, keep_dims=True), 0, keep_dims=True)
        c = tl.where(v > -float("inf"), c, 0.0)
        take = tl.minimum(c, K - filled)
        x1 = tl.where((kk >= filled) & (kk < filled + take), v, x1)
        over = c - take
        spill = tl.where((over > 0) & (spill == -float("inf")), v, spill)
        gap = tl.where(over > 0, v.to(tl.float64) - spill.to(tl.float64), 0.0)
        spilled += tl.where(over > 0, over.to(tl.float64) * exp(gap), 0.0)
        filled += take
        pool = tl.where(hit, -float("inf"), pool)
    # The largest score the K largest leave (or a chunk's level), and the sum of exp(z - it)
    # over those scores: float32 exponentials, float64 sums.
    low = tl.max(tl.max(pool, 1, keep_dims=True), 0, keep_dims=True)
    low = tl.maximum(tl.maximum(low, tl.max(peaks, 0, keep_dims=True)), spill)
    some = low > -float("inf")
    moved = tl.where((pool > -float("inf")) & some, tl.exp(pool - low) * counts, 0.0)
    tail = tl.sum(tl.sum(moved.to(tl.float64), 1, keep_dims=True), 0, keep_dims=True)
    moved = tl.where((peaks > -float("inf")) & some, tl.exp(peaks - low), 0.0).to(tl.float64)
    tail += tl.sum(moved * sums.to(tl.float64), 0, keep_dims=True)
    left = low.to(tl.float64)
    tail += (
        tl.where((spill > -float("inf")) & some, exp(spill.to(tl.float64) - left), 0.0) * spilled
    )
    largest = tl.max(x1, 1)
    maximum = largest.to(tl.float64)
    # The sum of exp(z - maximum) over the others but one copy of the maximum (x1[0]).
    e = tl.where((x1 > -float("inf")) & (kk > 0), exp(x1.to(tl.float64) - maximum[:, None]), 0.0)
    reach = tl.where(some, exp(left - maximum[:, None]), 0.0)
    others = tl.sum(e, 1) + tl.sum(tail * reach, 1)
    left = tl.reshape(left, (1,))
    value, first, second, bad, unsure = _entropy_capped(
        Z, row, live, y, z_y, x1, bad, maximum, others, tail, tl.reshape(left, (1, 1)), left, 1,
        KB, KB, K, KB, tl.float32, True,
    )  # fmt: skip
    limit = live & (z_y == -float("inf"))
    general = limit | (live & ~bad & unsure)
    _write_summary(
        V, S, row, live, w, ignored, limit, bad, value, first, second, value, VALUES, GRAD, STRIDE
    )
    tl.store(F + row, general.to(tl.float64), live)


@triton.jit
def _smooth_candidates(Z, row, live, y, z_y, x1, bad, left, fields, there, gamma,
                       CM: tl.constexpr, K: tl.constexpr, KB: tl.constexpr,
                       LOSS: tl.constexpr):  # fmt: skip
    """_smooth over a row's candidates x1, the beta box and its value from the chunk sums.
    Returns _smooth's results and whether the candidates may not decide them: a projection
    whose support reaches the largest score they leave (left)."""
    K1: tl.constexpr = K + 1
    cap = gamma / K
    kk = tl.arange(0, KB)[None, :]
    top, largest, _, _ = _select(Z, row, live, y, x1, 1, CM, CM, KB, K1, tl.float32, False)
    cutoff = _at(top, kk, K - 1)
    next_score = _at(top, kk, K)
    # The K + 1 largest are the candidates' where no score they leave exceeds them.
    inexact = live & ~(left <= next_score)
    hi, lo = _score_sum(top, (kk < K) & finite(top), K)
    mean = _mean_margin(hi, lo, z_y, K)
    head_mass = tl.minimum(tl.maximum(mean, 0.0), cap)
    isolated = finite(cutoff) & (cutoff - next_score >= head_mass) & (cutoff > next_score)
    if LOSS == BETA_SMOOTH:
        isolated = isolated & (margin(cutoff, z_y) >= cap)
    isolated_value = tl.where(
        head_mass > 0, (mean - 0.5 * head_mass) * ((head_mass / gamma) * K), 0.0
    )
    direct_step = head_mass / gamma
    top = margin(top, z_y[:, None])
    reach = margin(left, z_y)
    bad = bad | (largest == float("inf"))
    origin = tl.zeros([1], tl.float64)
    correction = origin
    active = live & ~bad & ~isolated
    unsure = inexact
    if LOSS == BETA_SMOOTH:
        box = tl.sum(tl.load(fields + _SUM, there, other=0.0), 1)
        project = active & (box > gamma)
        if tl.max(project.to(tl.int32), 0) > 0:
            o, s = _knapsack(
                Z, row, live, y, z_y, x1, top, kk, project, cap, gamma, 1, CM, CM, K, K1,
                tl.float32,
            )  # fmt: skip
            origin = tl.where(project, o, origin)
            correction = tl.where(project, s, correction)
            unsure = unsure | (project & (_projected(reach, origin, correction, gamma, K) > 0))
        boxed = tl.sum(tl.load(fields + _VALUE, there, other=0.0), 1) / gamma
        # Float32 box terms beyond their range leave the row to the general path.
        unsure = unsure | (active & ~(finite(box) & finite(boxed)))
    else:
        # The cone's projection unless its sum k c exceeds gamma: then sum(a) = gamma. The
        # candidates decide the cone where its support stays above the scores they leave;
        # else the knapsack's derivative in gamma decides: the cone's sum exceeds gamma
        # where t + sum over the capped of (h - t - cap) / K > 0 at its threshold t.
        t, c = _cone(Z, row, live, y, z_y, x1, top, kk, active, 1, CM, CM, K, K1, tl.float32)
        cone_sure = (c <= 0) | (reach <= t)
        full = active & (K * c > gamma)
        trial = active & (full | ~cone_sure)
        o = origin
        s = correction
        if tl.max(trial.to(tl.int32), 0) > 0:
            o, s = _knapsack(
                Z, row, live, y, z_y, x1, top, kk, trial, cap, gamma, 1, CM, CM, K, K1,
                tl.float32,
            )  # fmt: skip
        knap_sure = _projected(reach, o, s, gamma, K) <= 0
        level = o + s
        a = _projected(top, o[:, None], s[:, None], gamma, K)
        capped = (kk < K1) & (a >= cap) & finite(top)
        slope = level + tl.sum(tl.where(capped, (top - level[:, None]) - cap, 0.0), 1) / K
        knap = tl.where(cone_sure, full, slope > 0)
        unsure = unsure | (active & tl.where(knap, ~knap_sure, ~cone_sure))
        origin = tl.where(knap, o, tl.where(c > 0, t, float("inf")))
        correction = tl.where(knap, s, c)
        few = _at(top, kk, K - 1) == -float("inf")
        origin = tl.where(few | ~active, float("inf"), origin)
        correction = tl.where(few | ~active, 0.0, correction)
        boxed = origin * 0
        project = active
    h = margin(x1.to(tl.float64), z_y[:, None])
    a = _projected(h, origin[:, None], correction[:, None], gamma, K)
    value = tl.where(project, smooth_terms(h, x1 > -float("inf"), a, gamma, 1), boxed)
    return (
        tl.where(isolated, isolated_value, value),
        tl.where(isolated, cutoff, origin),
        tl.where(isolated, direct_step, correction),
        tl.where(isolated, -float("inf"), z_y),
        bad,
        unsure,
    )


@triton.jit
def _entries(
    Z,
    row,
    live,
    y,
    z_y,
    k,
    valid,
    s,
    first,
    second,
    third,
    gamma,
    C: tl.constexpr,
    K: tl.constexpr,
    LOSS: tl.constexpr,
    FP32: tl.constexpr,
    SHORT_SIMPLEX: tl.constexpr,
):
    """The gradient entries of columns k (not the label) of ROWS rows, unrounded."""
    z = tl.load(Z + row * C + k, valid, other=0)
    return _entry_values(
        z, valid & (k != y), z_y, s, first, second, third, gamma, C, K, LOSS, FP32, SHORT_SIMPLEX
    )


@triton.jit
def _entry_values(
    z,
    other,
    z_y,
    s,
    first,
    second,
    third,
    gamma,
    C: tl.constexpr,
    K: tl.constexpr,
    LOSS: tl.constexpr,
    FP32: tl.constexpr,
    SHORT_SIMPLEX: tl.constexpr,
):
    """The gradient entries of loaded scores z where other (not the label), unrounded."""
    if LOSS == ENTROPY and K == C - 1:
        entry = -expm1(-third) / K * s
    elif LOSS == ENTROPY:
        # The capped map cap min(1, exp(z - free) / D): scores above free sit at the cap.
        mass = -expm1(-third)
        cap = mass / K
        spread = tl.where(second > 0, second, 1.0)
        x = tl.where(other, z, -float("inf"))
        if FP32 and z.dtype != tl.float64:
            sign = tl.where(s < 0, -1.0, 1.0).to(tl.float32)
            factor = (cap / spread * tl.abs(s)).to(tl.float32)
            bound = (cap * tl.abs(s)).to(tl.float32)
            d = x.to(tl.float32) - first.to(tl.float32)
            e = tl.where(d > 0, bound, tl.minimum(exp(d) * factor, bound))
            entry = (sign * e).to(tl.float64)
        else:
            e = exp(tl.minimum(x.to(tl.float64) - first, log(spread)))
            entry = cap * tl.minimum(e / spread, 1.0) * s
    elif LOSS == ALPHA or LOSS == BETA:
        zt = z.to(tl.float64)
        weight = tl.where(zt > first, 1.0, tl.where(zt == first, second, 0.0))
        if LOSS == BETA and K > 1:
            h = margin(zt, z_y)
            third = tl.where(h == 0, 0.5, (h > 0).to(tl.float64))
        entry = tl.where(zt > -float("inf"), s * third * weight / K, 0.0)
    elif SHORT_SIMPLEX:
        zt = z.to(tl.float64)
        h = tl.where(third == float("inf"), (zt - first) - second, margin(zt, z_y))
        entry = (tl.minimum(tl.maximum(h, 0.0), gamma) / gamma) * s
    else:
        h = margin(z.to(tl.float64), z_y)
        projected = _projected(h, first, second, gamma, K) / gamma
        projected = tl.where(
            third == -float("inf"), tl.where(z.to(tl.float64) >= first, second, 0.0), projected
        )
        entry = projected * s
    return tl.where(other, entry, 0.0)


@triton.jit
def _quick_margin(x, label):
    """margin(x, label) of float32 scores x in two float64 operations, rounded once where it
    cancels (x near label - 1): label - 1 is exact for labels in [0.5, 2]; else x is no
    smaller than 0.5 there and x - label of float32 values within 2**29 of each other is
    exact, as is adding one; a label below 2**-28 adds last, after the exact x + 1."""
    t = x.to(tl.float64)
    near = (label >= 0.5) & (label <= 2.0)
    tiny = tl.abs(label) < 3.725290298461914e-09
    return tl.where(near, t - (label - 1.0), tl.where(tiny, (t + 1.0) - label, (t - label) + 1.0))


@triton.jit
def _lean_entries(z, other, z_y, s, first, second, third, gamma, C: tl.constexpr,
                  K: tl.constexpr, LOSS: tl.constexpr, FP32: tl.constexpr):  # fmt: skip
    """_entry_values with the row factors taken once per row (s / K, s / gamma) and float32
    compares of float32 scores: no division per entry. Float32 entropy entries stay float32."""
    if LOSS == ENTROPY and K < C - 1 and FP32 and z.dtype != tl.float64:
        mass = -expm1(-third)
        cap = mass / K
        spread = tl.where(second > 0, second, 1.0)
        sign = tl.where(s < 0, -1.0, 1.0).to(tl.float32)
        factor = (cap / spread * tl.abs(s)).to(tl.float32)
        bound = (cap * tl.abs(s)).to(tl.float32)
        d = tl.where(other, z.to(tl.float32), -float("inf")) - first.to(tl.float32)
        e = tl.where(d > 0, bound, tl.minimum(exp32(tl.minimum(d, 0.0)) * factor, bound))
        entry = tl.where(other, sign * e, 0.0)
    elif (LOSS == ALPHA or LOSS == BETA) and z.dtype != tl.float64:
        x = z.to(tl.float32)
        cutoff = first.to(tl.float32)
        unit = s / K
        if LOSS == BETA and K > 1:
            h = _quick_margin(x, z_y)
            slope = tl.where(h == 0, 0.5, (h > 0).to(tl.float64))
            above = unit * slope
            tie = (unit * second) * slope
        else:
            above = unit * third
            tie = (unit * second) * third
        entry = tl.where(x > cutoff, above, tl.where(x == cutoff, tie, unit * 0.0))
        entry = tl.where(other & (x > -float("inf")), entry, 0.0)
    elif LOSS == ALPHA_SMOOTH or LOSS == BETA_SMOOTH:
        zt = z.to(tl.float64)
        h = _quick_margin(z.to(tl.float32), z_y)
        projected = _projected(h, first, second, gamma, K) * (s / gamma)
        entry = tl.where(third == -float("inf"), tl.where(zt >= first, second * s, 0.0), projected)
        entry = tl.where(other, entry, 0.0)
    else:
        entry = _entry_values(
            z, other, z_y, s, first, second, third, gamma, C, K, LOSS, FP32, False
        )
    return entry


@triton.jit
def _unique_hinge(y, second, mass, s, k):
    column = (-second - 1.0).to(tl.int32)
    amount = s * mass
    return tl.where(k == y[:, None], -amount, tl.where(k == column, amount, 0.0))


@triton.jit
def _hinge_entries(Z, row, y, first, second, mass, s, k, valid, live,
                   C: tl.constexpr, ROWS: tl.constexpr):  # fmt: skip
    """Read the scores to share exact ties and scale finite nonwinners correctly."""
    z = tl.load(Z + row[:, None] * C + k, valid, other=0)
    other = valid & (k != y[:, None]) & (z > -float("inf"))
    winner = other & (z == first.to(z.dtype))
    entry = tl.where(winner, s * second, 0.0)
    target = -s * mass
    if tl.max((~finite(tl.reshape(s, (ROWS,))) & live).to(tl.int32), 0) != 0:
        entry = tl.where(other, s * tl.where(winner, second, 0.0), 0.0)
        target = -tl.sum(entry, 1)[:, None]
    return tl.where(k == y[:, None], target, entry)


@triton.jit(do_not_specialize=["N", "GROUP", "IGNORE", "UP", "GAMMA"])
def _backward(
    Z,
    Y,
    W,
    R,
    D,
    S,
    G,
    N,
    GROUP,
    IGNORE,
    UP,
    GAMMA,
    C: tl.constexpr,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    LOSS: tl.constexpr,
    FP32: tl.constexpr,
    WEIGHTED: tl.constexpr,
    IGNORED: tl.constexpr,
    ROWWISE: tl.constexpr,
    DIVIDED: tl.constexpr,
    ALIAS: tl.constexpr = False,
    INDEX: tl.constexpr = False,
    SHORT_SIMPLEX: tl.constexpr = False,
):
    """The gradient from the row statistics, rounded once; the label entry is minus the
    others' unrounded sum (entropy: -s mass), written last (G may be Z)."""
    if ALIAS:
        Z = G
    S = doubles(S)
    gamma = f64(GAMMA)
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = row < N
    y, inside, ignored, w = target_row(Y, W, row, live, C, IGNORE, GROUP, WEIGHTED, IGNORED)
    z_y = tl.load(Z + row * C + y, live, other=0).to(tl.float64)[:, None]
    s = (scale(UP, R, D, row, live, ROWWISE, DIVIDED) * w)[:, None]
    first = tl.load(S + row * 3, live, other=0.0)[:, None]
    second = tl.load(S + row * 3 + 1, live, other=0.0)[:, None]
    third = tl.load(S + row * 3 + 2, live, other=0.0)[:, None]
    if LOSS == ALPHA_SMOOTH or LOSS == BETA_SMOOTH:
        z_y = third
    bad = third != third
    zero = (ignored | (w == 0))[:, None] & ~bad
    if INDEX:
        if tl.reshape((second < 0.0) & finite(s), ()):
            # The published backward writes a unique winner from its index. The
            # row owner streams every output once and never rereads competitor scores.
            for start in range(0, C, B):
                k = start + tl.arange(0, B)[None, :]
                g = _unique_hinge(y, second, third, s, k)
                g = tl.where(bad, float("nan"), tl.where(zero, 0.0, g) + 0.0)
                tl.store(
                    G + row[:, None] * C + k, g.to(G.dtype.element_ty), live[:, None] & (k < C)
                )
            return
        # The general streamed formula multiplies share by total mass. A unique
        # winner has unit share. The register formula expects its final slope.
        second = tl.where(second < 0.0, third if B >= C else 1.0, second)
    if (LOSS == ALPHA or LOSS == BETA) and K == 1 and B >= C:
        k = tl.arange(0, B)[None, :]
        valid = live[:, None] & (k < C)
        g = _hinge_entries(Z, row, y, first, second, third, s, k, valid, live, C, ROWS)
        g = tl.where(bad, float("nan"), tl.where(zero, 0.0, g) + 0.0)
        tl.store(G + row[:, None] * C + k, g.to(G.dtype.element_ty), valid)
        return
    if SHORT_SIMPLEX:
        isolated = (third == float("inf")) & (second > 0)
        second = tl.where(isolated, -second, second)
        k = tl.arange(0, B)[None, :]
        valid = live[:, None] & (k < C)
        if tl.min(((third == float("inf")) & finite(s) | ~live[:, None]).to(tl.int32)) > 0:
            # A projected competitor simplex has total mass one.
            z = tl.load(Z + row[:, None] * C + k, valid, other=0).to(tl.float64)
            if tl.min(isolated.to(tl.int32)) > 0:
                grad = tl.where(k == y[:, None], -s, tl.where(z == first, s, 0.0))
            else:
                a = tl.minimum(tl.maximum((z - first) - second, 0.0), gamma)
                grad = tl.where(k == y[:, None], -s, (a / gamma) * s)
        else:
            entry = _entries(
                Z,
                row[:, None],
                live,
                y[:, None],
                z_y,
                k,
                valid,
                s,
                first,
                second,
                third,
                gamma,
                C,
                K,
                LOSS,
                FP32,
                SHORT_SIMPLEX,
            )
            grad = tl.where(k == y[:, None], -tl.sum(entry, 1)[:, None], entry)
        grad = tl.where(bad, float("nan"), tl.where(zero, 0.0, grad) + 0.0)
        tl.store(G + row[:, None] * C + k, grad.to(G.dtype.element_ty), valid)
        return
    total = tl.zeros([ROWS, B], tl.float64)
    for start in range(0, C, B):
        k = start + tl.arange(0, B)[None, :]
        valid = live[:, None] & (k < C)
        entry = _entries(
            Z,
            row[:, None],
            live,
            y[:, None],
            z_y,
            k,
            valid,
            s,
            first,
            second,
            third,
            gamma,
            C,
            K,
            LOSS,
            FP32,
            SHORT_SIMPLEX,
        )
        total += entry
        # + 0.0: zero entries are +0.0 for either sign of the upstream derivative.
        g = tl.where(bad, float("nan"), tl.where(zero, 0.0, entry) + 0.0)
        tl.store(G + row[:, None] * C + k, g.to(G.dtype.element_ty), valid & (k != y[:, None]))
    if LOSS == ENTROPY:
        target = 0.0 - s * -expm1(-third)
    else:
        target = -tl.sum(total, 1)[:, None]
    target = tl.where(bad, float("nan"), tl.where(zero, 0.0, target) + 0.0)
    tl.store(G + row[:, None] * C + y[:, None], target.to(G.dtype.element_ty), live[:, None])


@triton.jit
def _row_state(Z, Y, W, R, D, S, row, live, N, GROUP, IGNORE, UP, C: tl.constexpr,
               LOSS: tl.constexpr, WEIGHTED: tl.constexpr, IGNORED: tl.constexpr,
               ROWWISE: tl.constexpr, DIVIDED: tl.constexpr, STRIDE: tl.constexpr):  # fmt: skip
    """A streamed row's label, factor s, three statistics, effective label score (the smooth
    hinges': third), NaN mark and zero mark, as _backward reads them."""
    y, inside, ignored, w = target_row(Y, W, row, live, C, IGNORE, GROUP, WEIGHTED, IGNORED)
    s = (scale(UP, R, D, row, live, ROWWISE, DIVIDED) * w)[:, None]
    first = tl.load(S + row * STRIDE, live, other=0.0)[:, None]
    second = tl.load(S + row * STRIDE + 1, live, other=0.0)[:, None]
    third = tl.load(S + row * STRIDE + 2, live, other=0.0)[:, None]
    # The smooth hinges' and (K > 1) beta's third is the effective label score.
    z_y = third
    bad = third != third
    zero = (ignored | (w == 0))[:, None] & ~bad
    return y, s, first, second, third, z_y, bad, zero


@triton.jit(do_not_specialize=["N", "GROUP", "IGNORE", "UP", "GAMMA"])
def _chunk_backward(Z, Y, W, R, D, S, G, P, N, GROUP, IGNORE, UP, GAMMA, C: tl.constexpr,
                    B: tl.constexpr, RB: tl.constexpr, K: tl.constexpr, LOSS: tl.constexpr,
                    FP32: tl.constexpr, WEIGHTED: tl.constexpr, IGNORED: tl.constexpr,
                    ROWWISE: tl.constexpr, DIVIDED: tl.constexpr, CHUNKS: tl.constexpr,
                    STRIDE: tl.constexpr, WINDOWS: tl.constexpr, LOOP: tl.constexpr = 1,
                    SEL: tl.constexpr = 1, INDEX: tl.constexpr = False,
                    ALIAS: tl.constexpr = False):  # fmt: skip
    """The gradient of a streamed row in aligned windows of RB B columns (rows.py), LOOP
    windows per program (a tile [RB, B] each), from the row statistics: every column but the
    label, whose entry _label_entry overwrites from the windows' unrounded sums (P, CHUNKS
    per row). No window reads the label's score (the statistics hold it), so G may be Z.
    Rows that need no scores write without reading them: an indexed hinge's unique winner,
    and rows whose nonzero entries _solve placed in its chosen chunks (STRIDE > 3; B the
    chunks' width), whose other columns clear here while _chunk_heavy computes the chosen."""
    if ALIAS:
        Z = G
    S = doubles(S)
    P = doubles(P)
    row = tl.program_id(0).to(tl.int64)
    rows_ = row + tl.arange(0, 1)
    shift, shared = rows.shared_shift(Z, G, row, C, C)
    factor = scale(UP, R, D, rows_, rows_ < N, ROWWISE, DIVIDED)
    unique = tl.reshape(rows_ < 0, ())
    amount = factor
    winner = tl.zeros([1, 1], tl.int32)
    if INDEX:
        # A unique winner (a negative second) takes s times the mass; the rest is zero.
        second = tl.load(S + rows_ * STRIDE + 1)
        mass = tl.load(S + rows_ * STRIDE + 2)
        unique = tl.reshape((second < 0) & finite(factor) & (mass == mass), ())
        winner = (-second - 1.0).to(tl.int32)[:, None]
        amount = factor * mass
    listed = tl.reshape(rows_ < 0, ())
    chosen = tl.full([1, SEL], -1, tl.int32)
    if STRIDE > 3:
        base = S + row * STRIDE + 3
        flag = (tl.load(base + tl.arange(0, 1)) == 1) & shared
        listed = tl.reshape(
            flag & (tl.load(base + 1 + tl.arange(0, 1)) == shift) & finite(factor), ()
        )
        chosen = tl.load(base + 2 + tl.arange(0, SEL))[None, :].to(tl.int32)
    for t in range(LOOP):
        block = tl.program_id(1) * LOOP + t
        if block < WINDOWS:
            _, _, _, inside = rows.span(block, shift, shared, RB * B, C)
            j = block * (RB * B) + tl.arange(0, RB)[:, None] * B + tl.arange(0, B)[None, :]
            k = j - shift
            valid = (k >= 0) & (k < C)
            out = P + row * CHUNKS + block + tl.arange(0, 1)
            if unique:
                hit = k == winner
                g = tl.where(hit, amount.to(G.dtype.element_ty)[:, None], 0.0)
                if inside:
                    rows.store(G, row, j, shift, valid, g, C, False)
                else:
                    rows.store(G, row, j, shift, valid, g, C, True)
                mine = tl.max(tl.max((hit & valid).to(tl.int32), 1), 0) > 0
                tl.store(out, tl.where(mine, amount, 0.0))
            elif listed:
                own = block * RB + tl.arange(0, RB)[:, None]
                busy = tl.max((own == chosen).to(tl.int32), 1)[:, None] > 0
                blank = tl.zeros(j.shape, tl.float32)
                if tl.max(tl.max(busy.to(tl.int32), 1), 0) > 0:
                    rows.store(G, row, j, shift, valid & ~busy, blank, C, True)
                elif inside:
                    rows.store(G, row, j, shift, valid, blank, C, False)
                else:
                    rows.store(G, row, j, shift, valid, blank, C, True)
                tl.store(out, tl.zeros([1], tl.float64))
            else:
                if inside:
                    total = _chunk_grad(Z, Y, W, R, D, S, G, N, GROUP, IGNORE, UP, GAMMA, row,
                                        j, k, shift, valid, C, K, LOSS, FP32, WEIGHTED,
                                        IGNORED, ROWWISE, DIVIDED, STRIDE, False)  # fmt: skip
                else:
                    total = _chunk_grad(Z, Y, W, R, D, S, G, N, GROUP, IGNORE, UP, GAMMA, row,
                                        j, k, shift, valid, C, K, LOSS, FP32, WEIGHTED,
                                        IGNORED, ROWWISE, DIVIDED, STRIDE, True)  # fmt: skip
                if LOSS != ENTROPY:
                    tl.store(out, total)


@triton.jit(do_not_specialize=["N", "GROUP", "IGNORE", "UP", "GAMMA"])
def _chunk_heavy(Z, Y, W, R, D, S, G, P, N, GROUP, IGNORE, UP, GAMMA, C: tl.constexpr,
                 B: tl.constexpr, K: tl.constexpr, LOSS: tl.constexpr, FP32: tl.constexpr,
                 WEIGHTED: tl.constexpr, IGNORED: tl.constexpr, ROWWISE: tl.constexpr,
                 DIVIDED: tl.constexpr, STRIDE: tl.constexpr, SEL: tl.constexpr,
                 CHUNKS: tl.constexpr, ALIAS: tl.constexpr = False):  # fmt: skip
    """The entries of a listed row's slot-th chosen chunk (B columns from the forward's
    shift; one program per (row, slot)), with their unrounded sum in P (zero elsewhere)."""
    if ALIAS:
        Z = G
    S = doubles(S)
    P = doubles(P)
    gamma = f64(GAMMA)
    row = tl.program_id(0).to(tl.int64)
    slot = tl.program_id(1)
    rows_ = row + tl.arange(0, 1)
    base = S + row * STRIDE + 3
    shift, shared = rows.shared_shift(Z, G, row, C, C)
    lead = tl.load(base + 1).to(tl.int32)
    chunk = tl.load(base + 2 + slot).to(tl.int32)
    y, s, first, second, third, z_y, bad, zero = _row_state(
        Z, Y, W, R, D, S, rows_, rows_ < N, N, GROUP, IGNORE, UP, C, LOSS, WEIGHTED, IGNORED,
        ROWWISE, DIVIDED, STRIDE,
    )  # fmt: skip
    if (LOSS == ALPHA or LOSS == BETA) and K == 1:
        second = tl.where(second < 0.0, 1.0, second)
    listed = (tl.load(base + tl.arange(0, 1)) == 1) & shared & (lead == shift) & (chunk >= 0)
    listed = listed & tl.reshape(finite(s), (1,))
    total = tl.zeros([1], tl.float64)
    if tl.reshape(listed, ()):
        k = (chunk * B + tl.arange(0, B) - lead)[None, :]
        valid = (k >= 0) & (k < C)
        z = tl.load(Z + row * C + k, valid, other=0)
        other = valid & (k != y[:, None])
        entry = _lean_entries(z, other, z_y, s, first, second, third, gamma, C, K, LOSS, FP32)
        total = tl.sum(tl.where(other, entry, 0.0).to(tl.float64), 1)
        g = tl.where(zero, 0.0, entry) + 0.0
        tl.store(G + row * C + k, g.to(G.dtype.element_ty), valid)
    tl.store(P + row * (CHUNKS + SEL) + CHUNKS + slot + tl.arange(0, 1), total)


@triton.jit
def _chunk_grad(Z, Y, W, R, D, S, G, N, GROUP, IGNORE, UP, GAMMA, row, j, k, shift, valid,
                C: tl.constexpr, K: tl.constexpr, LOSS: tl.constexpr, FP32: tl.constexpr,
                WEIGHTED: tl.constexpr, IGNORED: tl.constexpr, ROWWISE: tl.constexpr,
                DIVIDED: tl.constexpr, STRIDE: tl.constexpr, MASKED: tl.constexpr):  # fmt: skip
    """Store a window's entries (the label's column too, which _label_entry overwrites);
    returns the unrounded sum of the others."""
    gamma = f64(GAMMA)
    rows_ = row + tl.arange(0, 1)
    y, s, first, second, third, z_y, bad, zero = _row_state(
        Z, Y, W, R, D, S, rows_, rows_ < N, N, GROUP, IGNORE, UP, C, LOSS, WEIGHTED, IGNORED,
        ROWWISE, DIVIDED, STRIDE,
    )  # fmt: skip
    if (LOSS == ALPHA or LOSS == BETA) and K == 1:
        # The indexed hinge's unique winner (a negative second) has unit share.
        second = tl.where(second < 0.0, 1.0, second)
    z = rows.load(Z, row, j, shift, valid, C, MASKED)
    other = valid & (k != y[:, None])
    quiet = tl.zeros([1], tl.int1)
    if LOSS != ENTROPY:
        # Entries vanish below the cutoff (a zero entry is +0.0 for finite factors); the
        # label's column, which _label_entry overwrites, may only keep a window busy.
        best = tl.max(tl.max(tl.where(valid, z, -float("inf")), 1), 0).to(tl.float64)
        if LOSS == ALPHA or LOSS == BETA:
            none = best < first
        else:
            isolated = third == -float("inf")
            reach = _projected(margin(best, z_y), first, second, gamma, K)
            none = tl.where(isolated, best < first, reach <= 0)
        quiet = tl.reshape(none & finite(s), (1,))
    if tl.reshape(quiet, ()):
        total = tl.zeros([1], tl.float64)
        blank = tl.where(bad, float("nan"), 0.0).to(tl.float32) + tl.zeros(z.shape, tl.float32)
        rows.store(G, row, j, shift, valid, blank, C, MASKED)
    else:
        entry = _lean_entries(z, other, z_y, s, first, second, third, gamma, C, K, LOSS, FP32)
        total = tl.sum(tl.sum(tl.where(other, entry, 0.0).to(tl.float64), 1), 0)[None]
        g = tl.where(bad, float("nan"), tl.where(zero, 0.0, entry) + 0.0)
        rows.store(G, row, j, shift, valid, g, C, MASKED)
    return total


@triton.jit(do_not_specialize=["N", "GROUP", "IGNORE", "UP"])
def _label_entry(Z, Y, W, R, D, S, G, P, N, GROUP, IGNORE, UP, C: tl.constexpr,
                 LOSS: tl.constexpr, WEIGHTED: tl.constexpr, IGNORED: tl.constexpr,
                 ROWWISE: tl.constexpr, DIVIDED: tl.constexpr, CHUNKS: tl.constexpr,
                 CP: tl.constexpr, STRIDE: tl.constexpr, SEL: tl.constexpr = 1,
                 ALIAS: tl.constexpr = False):  # fmt: skip
    """The label entries of _chunk_backward's rows: minus the others' unrounded sum (entropy:
    -s mass), rounded once."""
    if ALIAS:
        Z = G
    S = doubles(S)
    P = doubles(P)
    row = tl.program_id(0).to(tl.int64) + tl.arange(0, 1)
    live = row < N
    y, s, first, second, third, z_y, bad, zero = _row_state(
        Z, Y, W, R, D, S, row, live, N, GROUP, IGNORE, UP, C, ENTROPY, WEIGHTED, IGNORED,
        ROWWISE, DIVIDED, STRIDE,
    )  # fmt: skip
    if LOSS == ENTROPY:
        target = 0.0 - s * -expm1(-third)
    else:
        c = tl.arange(0, CP)[None, :]
        parts = tl.load(P + row[:, None] * CHUNKS + c, live[:, None] & (c < CHUNKS), other=0.0)
        if STRIDE > 3:
            # _chunk_heavy's sums follow the windows' (CHUNKS + SEL per row).
            parts = tl.load(
                P + row[:, None] * (CHUNKS + SEL) + c, live[:, None] & (c < CHUNKS + SEL), other=0.0
            )
        target = -tl.sum(parts, 1)[:, None]
    target = tl.where(bad, float("nan"), tl.where(zero, 0.0, target) + 0.0)
    tl.store(G + row[:, None] * C + y[:, None], target.to(G.dtype.element_ty), live[:, None])


def _code(call: Call) -> int:
    if call.kind == common.TOPK_CROSS_ENTROPY:
        return 0
    code = 1 if call.kind == common.TOPK_HINGE_ALPHA else 2
    return code + 2 if call.gamma > 0 else code


def _k(call: Call) -> int:
    return 1 if call.kind == common.MULTICLASS_HINGE else call.k


def covers(call: Call) -> str | None:
    """None where these kernels compute the call (topk_cross_entropy at k = classes - 1
    spreads the mass evenly), else why not."""
    k = _k(call)
    limit = MAX_SMOOTH_K if call.gamma > 0 and call.kind != common.TOPK_CROSS_ENTROPY else MAX_K
    if k > limit and not (call.kind == common.TOPK_CROSS_ENTROPY and k == call.c - 1):
        return f"the top-k kernels take k <= {limit}"
    return None


def _shape(n: int, c: int, k: int, code: int) -> tuple[int, int, int, int]:
    """Chunk, warps, rows per program and the short vector of the k + 1 largest."""
    kb = 1 if code == 0 and k == c - 1 else pow2(k + 1)
    block = max(16, min(pow2(c), _CHUNK))
    if code == 0 and k <= _SMALL_K and k < c - 1 and _WARP_ROW < c <= _CAP_ROW:
        # A small-k cap search over one register row keeps its reductions within a warp;
        # with too few rows to fill the device, more warps share each row.
        return pow2(c), 1 if n >= _FILLING_ROWS else 4, 1, kb
    limit = 256 if code == 0 else 128
    rows = 1
    if c <= limit and pow2(c) <= _CHUNK and not (code >= 3 and kb > block):
        cap = 16 if code < 3 or block <= 16 else 4
        rows = max(1, min(cap, 256 // block, 4096 // (kb * kb)))
    if rows > 1:
        warps = 2 if code < 3 or c <= 8 or block >= 128 else 4
    else:
        warps = max(4, block // _WARP_COLUMNS)
    return block, warps, rows, kb


def _candidate_sizes(k: int, code: int) -> tuple[int, int, int]:
    """The peaks a row's chosen chunks reach (the k largest scores; the smooth hinges' supports
    may reach a few past the k + 1 largest), the distinct scores taken per chosen chunk and
    the candidates the solvers take."""
    if code >= 3:
        return max(k + 3, 5), max(4, pow2(k + 1)), max(8, 2 * pow2(k + 1))
    return k, max(4, pow2(k + 1)), max(4, pow2(k + 1))


def plan(call: Call) -> common.Plan:
    """V and G of the top-k losses."""
    code, k = _code(call), _k(call)
    block, warps, rows, kb = _shape(call.n, call.c, k, code)
    # A single program owns the row, including all chunks when the row streams.
    index = call.indexed_hinge and rows == 1
    b = Builder(call)
    args = dict(b.inputs(), **b.results(), GAMMA=bits(call.gamma))
    stats = call.grad or call.saved == common.SAVE
    streamed = rows == 1 and block < call.c and not call.register_smooth_hinge
    # Streamed rows from their chunks' candidates: float32 work on scores up to float32 and a
    # short candidate list per chunk; _forward's general path takes the rows they leave open.
    # The smooth hinges' projections gain most; the nonsmooth top-k hinges gain on wide rows
    # (their gradient then skips the chunks below the cutoff); the k = 1 hinge keeps the
    # one-program rows, whose single streamed pass is as fast. The entropy, which sums the
    # whole row anyway, takes each chunk's k largest in the same read (chunked below).
    wide = call.c >= (_SMOOTH_WIDTH if code >= 3 else _CANDIDATE_WIDTH) and code > 0 and k > 1
    wide = wide or (code >= 3 and k == 1)
    candidates = (
        streamed and wide and call.fp32 and call.dtype != "float64" and k + 1 <= _CANDIDATE_K
    )
    # Streamed entropy rows from their chunks' k largest and sums: one read in parallel.
    chunked = (
        code == 0 and streamed and call.fp32 and call.dtype != "float64" and k <= _ENTROPY_K
        and k < call.c - 1 and call.c >= _ENTROPY_WIDTH
    )  # fmt: skip
    width, per = _SELECT
    ks, e, m = _candidate_sizes(k, code)
    sel = pow2(ks + 2)
    # The candidates' hinge rows also keep whether their chosen chunks hold every nonzero
    # gradient entry, the chunks' shift and the chunks.
    stride = 5 + sel if candidates and code != 0 else 3
    args["S"] = b.stats(stride) if stats else args["Z"]
    short = call.register_smooth_hinge
    if short:
        block = max(16, pow2(call.c))
    meta = dict(
        b.flags(),
        B=block,
        ROWS=rows,
        K=k,
        KB=kb,
        LOSS=code,
        INDEX=index,
        SHORT_SIMPLEX=short,
        STRIDE=stride,
    )
    grid = (common.cdiv(call.n, rows),)
    args["F"] = args["Z"]
    if call.saved != common.SAVED and candidates:
        windows = row_chunks(call.c, width * per)
        chunks = windows * per
        cargs = dict(args, P=b.array("P", (call.n, chunks * _FIELDS.value), "float64"))
        cargs["F"] = b.array("F", (call.n,), "float64")
        b.launch(_candidates, (call.n, windows), cargs, per,
                 **dict(meta, B=width, R=per, CHUNKS=chunks))  # fmt: skip
        cargs["Q"] = b.array("Q", (call.n, sel * (3 * e + 2)), "float64")
        smeta = dict(
            meta, GRAD=stats, B=width, KS=ks, E=e, M=m, SEL=sel, CHUNKS=chunks, CP=pow2(chunks)
        )
        b.launch(_gather, (call.n, sel), cargs, 1, **dict(smeta, GS=1))
        b.launch(_solve, (call.n,), cargs, 4, **smeta)
        shared = tuple(n for n in ("S", "V") if cargs[n] != args["Z"])
        b.launch(_forward, grid, cargs, warps, state=shared,
                 **dict(meta, GRAD=stats, GATED=True))  # fmt: skip
    elif call.saved != common.SAVED and chunked:
        width, per = _ENTROPY_CHUNK
        chunks = row_chunks(call.c, width)
        cargs = dict(args, P=b.array("P", (call.n, chunks * (2 * k + 2)), "float32"))
        cargs["F"] = b.array("F", (call.n,), "float64")
        b.launch(_entropy_partial, (call.n, chunks), cargs, per,
                 **dict(meta, B=width, KP=pow2(k), CHUNKS=chunks))  # fmt: skip
        b.launch(_entropy_merge, (call.n,), cargs, _ENTROPY_MERGE_WARPS,
                 **dict(meta, GRAD=stats, KP=pow2(k), CHUNKS=chunks, CP=pow2(chunks)))  # fmt: skip
        shared = tuple(n for n in ("S", "V") if cargs[n] != args["Z"])
        b.launch(_forward, grid, cargs, warps, state=shared,
                 **dict(meta, GRAD=stats, GATED=True))  # fmt: skip
    elif call.saved != common.SAVED:
        b.launch(_forward, grid, args, warps, **dict(meta, GRAD=stats))
    if call.grad and streamed and (code != 0 or chunked):
        # Streamed hinge and chunked entropy rows: the gradient in parallel aligned windows,
        # then the label entries.
        if stride > 3:
            # Rows the candidates decided: the windows of their chunks, cleared but where
            # chosen, and the chosen chunks.
            chunks = row_chunks(call.c, width * per)
            sums = b.array("H", (call.n, chunks + sel), "float64")
            bargs = dict(args, P=sums)
            loop = min(_FILL_LOOP, chunks)
            b.launch(_chunk_backward, (call.n, common.cdiv(chunks, loop)), bargs, per,
                     **dict(meta, B=width, RB=per, CHUNKS=chunks + sel, SEL=sel,
                            WINDOWS=chunks, LOOP=loop))  # fmt: skip
            hargs = dict(args, P=common.Ref("H"))
            b.launch(_chunk_heavy, (call.n, sel), hargs, 1, state=("G", "P"),
                     **dict(meta, B=width, SEL=sel, CHUNKS=chunks))  # fmt: skip
        else:
            width, per = _ENTROPY_BACKWARD if chunked else _BACKWARD
            chunks = row_chunks(call.c, width * per)
            sums = b.array("H", (call.n, chunks), "float64")
            bargs = dict(args, P=sums)
            b.launch(_chunk_backward, (call.n, chunks), bargs, per,
                     **dict(meta, B=width, RB=per, CHUNKS=chunks, WINDOWS=chunks))  # fmt: skip
        b.launch(_label_entry, (call.n,), bargs, 1, state=("G",),
                 **dict(meta, CHUNKS=chunks, CP=pow2(chunks + (sel if stride > 3 else 0)),
                        SEL=sel))  # fmt: skip
    elif call.grad:
        b.launch(_backward, grid, args, warps, **meta)
    return b.plan()


def describe(call: Call) -> str:
    entropy = call.kind == common.TOPK_CROSS_ENTROPY
    return "float32 exponentials" if entropy and call.fp32 else "float64"
