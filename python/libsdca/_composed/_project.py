"""Batched projections of the smoothed hinges onto simplex-like sets (docs/API.md,
Projections), on (rows, classes) arrays of any namespace. valid marks the entries a row
projects; the others are zero in the result.
"""

from __future__ import annotations

from collections.abc import Callable

from ._xp import Array, Namespace


def search(
    xp: Namespace, lo: Array, hi: Array, size: int, holds: Callable[[Array], Array]
) -> Array:
    """Last index in [lo, hi] (hi - lo <= size) of a monotone predicate that holds at lo."""

    def step(_: int, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
        lo, hi = bounds
        mid = (lo + hi + 1) // 2
        ok = holds(mid)
        return xp.where(ok, mid, lo), xp.where(ok, hi, mid - 1)

    return xp.fori(size.bit_length(), step, (lo, hi))[0]


def following(xp: Namespace, points: Array, index: Array, last: Array) -> Array:
    """points[index + 1], or -inf past last."""
    after = xp.take(points, xp.minimum(index + 1, points.shape[-1] - 1))
    return xp.where(index >= last, -float("inf"), after)


def exact_sum(xp: Namespace, x: Array, low: Array | None = None) -> Array:
    """Row sums (plus low) in double-double, by pairwise TwoSum."""
    n = x.shape[-1]
    x = xp.pad(x, 1 << (n - 1).bit_length())
    low = xp.full_like(x[:, :1], 0.0) if low is None else low
    while x.shape[-1] > 1:
        a, b = x[:, 0::2], x[:, 1::2]
        x = a + b
        rounded = x - a
        # An infinite sum has no rounding error to carry (and inf - inf is nan).
        error = xp.where(xp.isfinite(x), (a - (x - rounded)) + (b - rounded), 0.0)
        low = low + xp.sum(error)
    return x + low


def descending(xp: Namespace, h: Array, valid: Array) -> Array:
    return xp.sort(xp.where(valid, h, -float("inf")))


def knapsack(
    xp: Namespace,
    h: Array,
    valid: Array,
    cap: Array | float,
    rhs: Array | float,
    ordered: Array,
    max_u: int | None = None,
) -> tuple[Array, Array]:
    """Origin and correction of t with sum(clip(h - t, 0, cap)) = rhs over the valid entries,
    from their sorted values ordered; max_u bounds the capped count (cap (max_u + 1) > rhs).

    Bisection counts the capped entries, then the free ones, from the clipped mass relative
    to a represented score: a cap below the resolution of the scores still counts, and
    saturated far-off entries cannot cancel the free mass.
    """
    count = xp.count(valid)
    size = ordered.shape[-1]

    def within(j: Array, shift: Array | float) -> Array:
        # The mass at t = ordered[j - 1] + shift is at most rhs.
        origin = xp.take(ordered, xp.maximum(j - 1, 0))
        parts = xp.where(valid, xp.clip((h - origin) - shift, 0.0, cap), 0.0)
        return xp.sum(parts) <= rhs

    zero = count * 0
    if max_u == 0:
        u = zero
    else:
        span = size if max_u is None else min(size, max_u)
        hi = count if max_u is None else xp.minimum(count, max_u)
        u = search(xp, zero, hi, span, lambda j: (j == 0) | within(j, -cap))
    end = search(xp, u, count, size, lambda j: (j == u) | within(j, 0.0))
    origin = xp.take(ordered, xp.maximum(end - 1, 0))
    position = xp.arange(size, ordered) + 1
    free = (position > u) & (position <= end)
    excess = xp.sum(xp.where(free, ordered - origin, 0.0))
    excess = excess - (rhs - cap * xp.astype(u, h.dtype))
    correction = xp.minimum(excess / xp.astype(xp.maximum(end - u, 1), h.dtype), 0.0)
    correction = xp.maximum(correction, following(xp, ordered, end - 1, count - 1) - origin)
    # Without free entries the mass is flat: t = origin - cap caps the first u.
    return origin, xp.where(end > u, correction, -cap)


def knapsack_le(
    xp: Namespace, h: Array, valid: Array, cap: float, rhs: float, max_u: int | None = None
) -> Array:
    """Projection onto 0 <= x_i <= cap, sum(x) <= rhs."""
    box = xp.where(valid, xp.clip(h, 0.0, cap), 0.0)
    origin, correction = knapsack(xp, h, valid, cap, rhs, descending(xp, h, valid), max_u)
    full = xp.where(valid, xp.clip((h - origin) - correction, 0.0, cap), 0.0)
    return xp.where(xp.sum(box) <= rhs, box, full)


def topk_cone(xp: Namespace, h: Array, valid: Array, k: int, ordered: Array) -> tuple[Array, Array]:
    """Threshold t and cap c of the projection clip(h - t, 0, c) onto 0 <= x_i <= sum(x)/k,
    from the sorted valid entries ordered (at least k of them; at most all but one entry).

    Bisection on the monotone stationarity residuals finds the number u of capped entries,
    then the number m of free ones; (t, c) solve the stationarity equations of (u, m).
    """
    dtype, size = h.dtype, h.shape[-1] - 1
    count = xp.count(valid)

    def above(t: Array) -> Array:
        return xp.sum(xp.where(valid, xp.maximum(h - t, 0.0), 0.0))

    def capped(j: Array) -> Array:
        # The cap level t + c = ordered[j] lies above the solution's.
        level = xp.take(ordered, xp.maximum(j, 0))
        t = -above(level) / k
        return (j < 0) | (above(t) / k + 2 * t < level)

    start = count * 0 - 1
    u = search(xp, start, start + k, k, capped) + 1
    uf = xp.astype(u, dtype)
    position = xp.arange(h.shape[-1], h)
    head = xp.sum(xp.where(position < u, ordered, 0.0))

    def free(j: Array) -> Array:
        # The threshold t = ordered[j] lies above the solution's.
        t = xp.take(ordered, xp.maximum(j, 0))
        return (j < u) | (above(t) / k + 2 * t < (k * t + head) / xp.maximum(uf, 1.0))

    index = search(xp, u - 1, xp.maximum(count - 1, u - 1), size, free)
    m = xp.where(u == 0, xp.count(ordered > 0), index - u + 1)
    rest = xp.sum(xp.where((position >= u) & (position < u + m), ordered, 0.0))
    ku, mf = xp.astype(k - u, dtype), xp.astype(m, dtype)
    d = ku * ku + mf * uf
    t = (uf * rest - ku * head) / d
    c = (mf * head + ku * rest) / d
    # Exactly the k largest are capped when the rest lie below their threshold.
    whole = exact_sum(xp, ordered[:, :k])
    top = ordered[:, k - 1 : k] - whole / k
    # 16 eps relative to scores below one keeps tiny margins.
    tol = 16 * xp.finfo(dtype).eps * xp.minimum(xp.sum(xp.abs(ordered[:, :k])), 1.0)
    full = (u == k) | (top >= ordered[:, k : k + 1] - tol)
    t, c = xp.where(full, top, t), xp.where(full, whole / k, c)
    zero = whole <= tol
    return xp.where(zero, 0.0, t), xp.where(zero, 0.0, c)


def topk_simplex(xp: Namespace, h: Array, valid: Array, k: int, rhs: float) -> Array:
    """Projection onto 0 <= x_i <= sum(x)/k, sum(x) <= rhs: the cone's unless its sum exceeds
    rhs. Rows of fewer than k valid entries project to zero."""
    ordered = descending(xp, h, valid)
    t, c = topk_cone(xp, h, valid, k, ordered)
    origin, correction = knapsack(xp, h, valid, rhs / k, rhs, ordered, k)
    cone = xp.minimum(xp.maximum(h - t, 0.0), c)
    full = xp.clip((h - origin) - correction, 0.0, rhs / k)
    enough = xp.count(valid) >= k
    return xp.where(valid & enough, xp.where(k * c > rhs, full, cone), 0.0)


def two_simplex(xp: Namespace, h: Array, positive: Array, negative: Array, rhs: float) -> Array:
    """Projection onto x >= 0 whose positive and negative entries have equal sums <= rhs."""
    # A cap of 2 rhs leaves no entry capped.
    oa, ca = knapsack(xp, h, positive, 2 * rhs, rhs, descending(xp, h, positive), 0)
    ob, cb = knapsack(xp, h, negative, 2 * rhs, rhs, descending(xp, h, negative), 0)
    simplex = xp.where(positive, (h - oa) - ca, (h - ob) - cb)
    # Below rhs, sum(max(h - t, 0)) over the positives = sum(max(h + t, 0)) over the others.
    valid = positive | negative
    points = xp.sort(xp.where(positive, h, xp.where(negative, -h, -float("inf"))))

    def balance(t: Array) -> Array:
        parts = xp.where(positive, xp.maximum(h - t, 0.0), -xp.maximum(h + t, 0.0))
        return xp.sum(xp.where(valid, parts, 0.0))

    last = xp.count(valid) - 1
    index = search(xp, last * 0, last, h.shape[-1], lambda j: balance(xp.take(points, j)) <= 0)
    origin, below = xp.take(points, index), following(xp, points, index, last)
    left, right = positive & (h > below), negative & (-h <= below)
    parts = xp.where(left, h - origin, 0.0) - xp.where(right, h + origin, 0.0)
    t = xp.sum(parts) / xp.astype(xp.maximum(xp.count(left) + xp.count(right), 1), h.dtype)
    t = xp.maximum(xp.minimum(t, 0.0), below - origin)
    equal = xp.where(positive, (h - origin) - t, (h + origin) + t)
    # Rounding of the thresholds' own terms, not a fixed floor, which would swallow tiny margins.
    scale = xp.maximum((xp.abs(oa) + xp.abs(ob)) + (xp.abs(ca) + xp.abs(cb)), rhs)
    active = ((oa + ob) + (ca + cb) >= -xp.finfo(h.dtype).eps * scale) & (xp.count(negative) > 0)
    return xp.where(valid, xp.clip(xp.where(active, simplex, equal), 0.0, rhs), 0.0)
