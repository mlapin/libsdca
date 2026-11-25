"""Exact references of the hinges from their definitions: the top-k hinges (alpha, beta; their
smoothed forms by the KKT conditions of their projections) in mpmath, and the smoothed
multilabel hinge's two-simplex projection in rationals."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import mpmath as mp
from mpmath_types import MPReal


def _positive(x: MPReal) -> MPReal:
    """max(x, 0)."""
    return x if x > 0 else mp.mpf(0)


def _clip(x: MPReal, cap: MPReal) -> MPReal:
    """min(max(x, 0), cap)."""
    return cap if x > cap else _positive(x)


def _knapsack(h: list[MPReal], cap: MPReal, rhs: MPReal) -> MPReal:
    """t with sum(clip(h - t, 0, cap)) = rhs, from the breakpoints of the mass."""

    def mass(t: MPReal) -> MPReal:
        # mpf of the sum: mpmath's fsum is inferred to return int (its start).
        return mp.mpf(mp.fsum(_clip(x - t, cap) for x in h))

    points = sorted(set(h) | {x - cap for x in h})
    for lo, hi in zip(points, points[1:], strict=False):
        if mass(lo) >= rhs >= mass(hi):
            span = mass(lo) - mass(hi)
            return lo if span == 0 else lo + (mass(lo) - rhs) * (hi - lo) / span
    return points[0] - (rhs - mass(points[0])) / len(h)


def _cone(h: list[MPReal], k: int) -> list[MPReal]:
    """Projection of h onto 0 <= x_i <= sum(x) / k, by its KKT conditions over the sorted h."""
    s = sorted(h, reverse=True)
    if mp.fsum(s[:k]) <= 0:
        return [mp.mpf(0)] * len(h)
    c = mp.fsum(s[:k]) / k
    if k == len(s) or s[k] <= s[k - 1] - c:
        return [_clip(x - (s[k - 1] - c), c) for x in h]
    tol = mp.mpf(10) ** (20 - mp.mp.dps) * (1 + max(abs(x) for x in s))
    for u in range(k):
        for m in range(len(s) - u + 1):
            head, rest = mp.fsum(s[:u]), mp.fsum(s[u : u + m])
            d = (k - u) ** 2 + m * u
            if d == 0:
                continue
            t, c = (u * rest - (k - u) * head) / d, (m * head + (k - u) * rest) / d
            ok = c > 0 and all(x - t >= c - tol for x in s[:u])
            ok = ok and all(-tol <= x - t <= c + tol for x in s[u : u + m])
            if ok and all(x <= t + tol for x in s[u + m :]):
                return [_clip(x - t, c) for x in h]
    raise AssertionError("no cone active set")


def topk_hinge(
    z: Sequence[float], y: int, variant: str, k: int, gamma: float
) -> tuple[MPReal, list[MPReal]]:
    """Value and gradient of topk_hinge (variant alpha or beta; multiclass_hinge is alpha at
    k = 1) of label y, smoothed for gamma > 0, in the working digits."""
    zs = [mp.mpf(x) for x in z]
    others = [j for j in range(len(zs)) if j != y]
    h = [1 + zs[j] - zs[y] for j in others]
    if gamma == 0:
        # Kinks take the mean of the one-sided subgradients: the k-th place splits evenly among
        # the margins tied at it, a hinge at zero counts half; at k = 1 a largest margin of zero
        # ties with the hinge's zero (1 / (tied + 1) each, as optax).
        scores = [zs[j] for j in others]
        order = sorted(range(len(h)), key=lambda i: -scores[i])[:k]
        terms = [_positive(h[i]) if variant == "beta" else h[i] for i in order]
        value = _positive(mp.fsum(terms) / k)
        last = scores[order[-1]] if len(order) == k else mp.mpf("-inf")
        above = sum(1 for x in scores if x > last)
        tied = sum(1 for x in scores if x == last)
        share = mp.mpf(k - above) / tied if tied else mp.mpf(0)

        def half(x: MPReal) -> MPReal:
            return mp.mpf(1) if x > 0 else (mp.mpf(1) / 2 if x == 0 else mp.mpf(0))

        mean = mp.fsum(h[i] for i in order) / k
        a = [mp.mpf(0)] * len(h)
        for i, x in enumerate(h):
            if k == 1 and h[order[0]] == 0:
                a[i] = mp.mpf(1) / (tied + 1) if scores[i] == last else mp.mpf(0)
            elif scores[i] >= last:
                slope = half(x) if variant == "beta" else half(mean)
                a[i] = (1 if scores[i] > last else share) * slope / k
    else:
        g = mp.mpf(gamma)
        cap = g / k
        a = _cone(h, k) if variant == "alpha" else [_clip(x, cap) for x in h]
        if mp.fsum(a) > g:
            t = _knapsack(h, cap, g)
            a = [_clip(x - t, cap) for x in h]
        value = mp.fsum(x * b - b * b / 2 for x, b in zip(h, a, strict=True)) / g
        a = [b / g for b in a]
    gradient = [mp.mpf(0)] * len(zs)
    for i, j in enumerate(others):
        gradient[j] = a[i]
    gradient[y] = -mp.fsum(a)
    return value, gradient


def _simplex(a: list[Fraction], rhs: Fraction) -> Fraction:
    """Threshold t of the projection max(0, a - t) onto sum = rhs, by sorting."""
    s = sorted(a, reverse=True)
    total = Fraction(0)
    for i, v in enumerate(s):
        total += v
        t = (total - rhs) / (i + 1)
        if i + 1 == len(s) or s[i + 1] <= t:
            return t
    raise ValueError("empty vector")


def two_simplex(
    a: list[Fraction], b: list[Fraction], rhs: Fraction
) -> tuple[list[Fraction], list[Fraction]]:
    """Projection of (a, b) onto x, y >= 0 with sum(x) = sum(y) <= rhs (KKT, exact
    breakpoints)."""
    zero = Fraction(0)

    def f(t: Fraction) -> Fraction:
        return sum((max(v - t, zero) for v in a), zero) - sum((max(v + t, zero) for v in b), zero)

    points = sorted(set(a) | {-v for v in b})
    values = [f(t) for t in points]
    if values[0] <= 0:
        t = points[0] + values[0] / len(a)
    elif values[-1] >= 0:
        t = points[-1] + values[-1] / len(b)
    else:
        i = next(i for i in range(len(points) - 1) if values[i] >= 0 >= values[i + 1])
        lo, hi = points[i], points[i + 1]
        t = lo + values[i] / (values[i] - values[i + 1]) * (hi - lo)
    x = [max(v - t, zero) for v in a]
    if sum(x) <= rhs:
        return x, [max(v + t, zero) for v in b]
    ta, tb = _simplex(a, rhs), _simplex(b, rhs)
    return [max(v - ta, zero) for v in a], [max(v - tb, zero) for v in b]
