"""The LOSSES in jax.numpy on (examples, classes) rows: a port of the tensor
backend of libsdca.torch (include/sdca/loss.h, solved in the dtype of the
scores) for any device. Rows with an invalid target, or a proximal_entropy
solve that does not converge, give nan; there is no host check.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import DTypeLike

_INT = jnp.int32
# Smooth hinges solve from this many of the largest margins, and from all of them only when a
# row's solution reaches past them.
_CANDIDATES = 16
# Margins below this fraction of their terms cancel (sdca/math/compensated.h).
_MARGIN_CANCELLATION = 1 / 16
# Binary searches over rows of at most this many classes run as loops (see _search).
_UNROLL = 64
_SINGLE_TARGET = frozenset(
    [
        "topk_cross_entropy",
        "topk_hinge_alpha",
        "topk_hinge_beta",
        "topk_hinge_alpha_smooth",
        "topk_hinge_beta_smooth",
    ]
)


def _take(x: jax.Array, index: jax.Array) -> jax.Array:
    return jnp.take_along_axis(x, index, axis=-1, mode="clip")


def _put(x: jax.Array, index: jax.Array, values: jax.Array) -> jax.Array:
    """x with x[i, index[i, j]] = values[i, j]."""
    rows = jnp.arange(x.shape[0])[:, None]
    return x.at[rows, index].set(values, mode="drop")


def _descending(x: jax.Array) -> jax.Array:
    return -jnp.sort(-x, axis=-1)


def _cpu() -> bool:
    """Whether the computation being traced is for XLA:CPU, which takes the jaxlib 0.6
    workarounds below: off the CPU each lax.cond would wait for its predicate on the host."""
    return jax.default_backend() == "cpu"


def _float32_top_k(x: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    # The fence keeps XLA:CPU (jaxlib 0.6) from moving the slice of the result into a
    # conditional, which turns top_k into a sort of whole rows (50 times slower).
    values, index = lax.optimization_barrier(lax.top_k(x, k))
    return values, index


def top_k(x: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    """lax.top_k(x, k) (ties take the lowest index), on the CPU from float32 keys: XLA:CPU of
    jaxlib 0.6 sorts whole rows for lax.top_k of any other dtype, 50 to 100 times slower.

    Narrower types are exact in float32. Wider ones take k + 1 candidates by their rounded
    keys, ordered exactly; they hold the k largest unless rounding ties the k-th key to an
    unequal entry outside them, and then only the rows' exact top_k is taken."""
    if not _cpu():
        return lax.top_k(x, k)
    if x.dtype == jnp.float32:
        return _float32_top_k(x, k)
    if jnp.dtype(x.dtype).itemsize <= 2:
        index = _float32_top_k(x.astype(jnp.float32), k)[1]
        return _take(x, index), index
    if k >= x.shape[-1]:
        return lax.top_k(x, k)
    keys = x.astype(jnp.float32)
    width = k + 1
    candidates, index = _float32_top_k(keys, width)
    negated, index = lax.sort((-_take(x, index), index), num_keys=2)
    top, index = -negated[:, :k], index[:, :k]
    if width == x.shape[-1]:
        return top, index
    # Every entry outside the candidates has a key at most the last candidate's: below the
    # k-th key, it is below the k-th entry.
    kth = top[:, -1:].astype(jnp.float32)
    separated = (candidates[:, -1:] < kth).all()

    def checked() -> tuple[jax.Array, jax.Array]:
        # Keys tied at the k-th only between equal entries: the lowest indices are exact.
        equal = _count(keys == kth) == _count(x == top[:, -1:])
        return lax.cond(equal.all(), lambda: (top, index), lambda: tuple(lax.top_k(x, k)))

    return lax.cond(separated, lambda: (top, index), checked)


def over(x: jax.Array, scalar: float) -> jax.Array:
    """x / scalar, divided: XLA multiplies by the reciprocal of a constant divisor, which
    overflows below 5.6e-309, and XLA:CPU reads subnormals as zero. A divisor below 2^-1000
    is scaled up by 2^64 and the quotient back, which is exact. A power of two whose
    reciprocal is normal in the dtype of x multiplies by it, which is exact (T = 1: x)."""
    info = jnp.finfo(x.dtype)
    if math.frexp(scalar)[0] == 0.5 and float(info.tiny) <= min(scalar, 1 / scalar):
        return x if scalar == 1 else x * jnp.asarray(1 / scalar, x.dtype)
    shift = 64 if scalar < 2.0**-1000 else 0
    divisor = lax.optimization_barrier(jnp.asarray(scalar * 2.0**shift, x.dtype))
    quotient = x / divisor
    return quotient * 2.0**shift if shift else quotient


def difference(a: jax.Array, b: jax.Array | float) -> tuple[jax.Array, jax.Array]:
    """a - b and its rounding error (TwoSum)."""
    d = a - b
    rounded = d - a
    return d, (a - (d - rounded)) + (-b - rounded)


def _sum(x: jax.Array) -> jax.Array:
    return x.sum(-1, keepdims=True)


# The longest run of a row that row_sums adds in order.
_RUN = 256


def row_sums(terms: tuple[jax.Array, ...], masks: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    """Row sums of the terms and row counts of the masks (in the dtype of the first term),
    keepdims: the masks in the first term's pass.

    On XLA:CPU a plain row reduction of a computed or boolean array first writes it to memory
    and reduces it in a reduce-window (jaxlib 0.6 for every reduction over more than 32
    entries, 0.12 for integer and boolean ones): a pass and a temporary of the size of the
    scores. A variadic reduction fuses its operands, and jaxlib 0.12 emits it as fast as a
    plain one for one floating-point operand and any integer ones (four to six times slower
    for two floating-point ones). It adds in order, so it runs over contiguous runs of at
    most _RUN entries (the longest that divides the row), and plain sums add the runs:
    rounding as a pairwise sum. Elsewhere, plain reductions."""
    first = _row_sums(terms[0], tuple(m.astype(_INT) for m in masks))
    rest = tuple(_row_sums(t, ())[0] for t in terms[1:])
    return (first[0], *rest, *first[1:])


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _row_sums(term: jax.Array, counts: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    """The row sum of term and those of the integer counts (in its dtype), keepdims, in one
    pass."""
    if not _cpu():
        # Counts contain only 0/1, so their sum fits int32 below 2**31 classes.
        # On GPU, an explicit dtype avoids widening when x64 is enabled.
        classes = term.shape[-1]
        bounded = isinstance(classes, int) and classes < 2**31
        dtype = _INT if jax.default_backend() == "gpu" and bounded else None
        counted = tuple(c.sum(-1, keepdims=True, dtype=dtype).astype(term.dtype) for c in counts)
        return (_sum(term), *counted)
    # Alone, the sum takes the count of its nan terms along: a plain reduction would write
    # its operand to memory first.
    alone = not counts
    if alone:
        counts = ((term != term).astype(_INT),)
    classes = term.shape[-1]
    run = max(r for r in range(1, min(classes, _RUN) + 1) if classes % r == 0)
    shape = (*term.shape[:-1], classes // run, run)
    operands = (term.reshape(shape), *(c.reshape(shape) for c in counts))

    def add(a: tuple[jax.Array, ...], b: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        return tuple(x + y for x, y in zip(a, b, strict=True))

    inits = tuple(jnp.zeros((), o.dtype) for o in operands)
    out = lax.reduce(operands, inits, add, (len(shape) - 1,))
    total, counted = _sum(out[0]), tuple(_count(c).astype(term.dtype) for c in out[1:])
    if alone:
        return (jnp.where(counted[0] > 0, jnp.nan, total),)
    return (total, *counted)


@_row_sums.defjvp
def _row_sums_jvp(
    counts: tuple[jax.Array, ...], primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    out = _row_sums(primals[0], counts)
    return out, (_sum(tangents[0]), *(jnp.zeros_like(c) for c in out[1:]))


def row_first_max(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The row maximum of x (without derivatives) and its first class, keepdims, in one pass:
    on XLA:CPU one variadic reduction (see row_sums)."""
    x = lax.stop_gradient(x)
    classes = x.shape[-1]
    index = jnp.arange(classes, dtype=_INT)
    if not _cpu():
        top = x.max(-1, keepdims=True)
        return top, jnp.where(x == top, index, classes).min(-1, keepdims=True)

    def pick(a: tuple[jax.Array, ...], b: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        top = lax.max(a[0], b[0])
        first = lax.min(
            jnp.where(a[0] == top, a[1], classes), jnp.where(b[0] == top, b[1], classes)
        )
        return top, first

    inits = (jnp.asarray(-jnp.inf, x.dtype), jnp.asarray(classes, _INT))
    top, first = lax.reduce((x, jnp.broadcast_to(index, x.shape)), inits, pick, (x.ndim - 1,))
    return top[..., None], first[..., None]


def _count(x: jax.Array) -> jax.Array:
    return x.sum(-1, keepdims=True, dtype=_INT)


def _softplus(x: jax.Array) -> jax.Array:
    """log1p(exp(x)) without a linear cutoff."""
    return jnp.where(x > 0, x + jnp.log1p(jnp.exp(-x)), jnp.log1p(jnp.exp(x)))


def _search(
    lo: jax.Array,
    hi: jax.Array,
    size: int,
    holds: Callable[[jax.Array], jax.Array],
    classes: int,
) -> jax.Array:
    """Last index in [lo, hi] (hi - lo <= size) of a monotone predicate that holds at lo.

    Unrolled steps for rows of more than _UNROLL classes; a loop for narrower ones, whose
    unrolled searches XLA:CPU duplicates into each other until compiling takes minutes."""

    def step(_: int, bounds: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        lo, hi = bounds
        mid = (lo + hi + 1) // 2
        ok = holds(mid)
        return jnp.where(ok, mid, lo), jnp.where(ok, hi, mid - 1)

    if classes <= _UNROLL:
        return lax.fori_loop(0, size.bit_length(), step, (lo, hi))[0]
    for i in range(size.bit_length()):
        lo, hi = step(i, (lo, hi))
    return lo


def _next(points: jax.Array, index: jax.Array, last: jax.Array) -> jax.Array:
    """points[index + 1], or -inf past last."""
    following = _take(points, jnp.minimum(index + 1, points.shape[-1] - 1))
    return jnp.where(index >= last, -jnp.inf, following)


def _largest(h: jax.Array, valid: jax.Array, key: jax.Array, width: int | None) -> jax.Array:
    """The width largest valid h, descending (-inf past the valid ones): all of them for None,
    else those at the largest keys, which h does not decrease along."""
    masked = jnp.where(valid, h, -jnp.inf)
    if width is None:
        return _descending(masked)
    index = top_k(jnp.where(valid, key, -jnp.inf), width)[1]
    return _descending(_take(masked, index))


def _knapsack(
    h: jax.Array,
    valid: jax.Array,
    cap: jax.Array,
    rhs: jax.Array,
    ordered: jax.Array,
    max_u: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Origin and correction of t with sum(clip(h - t, 0, cap)) = rhs over the valid entries,
    from the sorted valid h or their largest ones (max_u below that width), and whether the
    prefix held the free entries and the next one."""
    count = _count(valid)
    size = ordered.shape[-1]
    within_prefix = jnp.minimum(count, size)

    def within(j: jax.Array, shift: jax.Array | float) -> jax.Array:
        # The mass at t = ordered[j - 1] + shift is at most rhs.
        origin = _take(ordered, jnp.maximum(j - 1, 0))
        parts = jnp.where(valid, jnp.clip((h - origin) - shift, 0, cap), 0.0)
        return _sum(parts) <= rhs

    zero = jnp.zeros_like(count)
    if max_u == 0:
        u = zero
    else:
        span = size if max_u is None else min(size, max_u)
        hi = within_prefix if max_u is None else jnp.minimum(within_prefix, max_u)
        u = _search(zero, hi, span, lambda j: (j == 0) | within(j, -cap), h.shape[-1])
    end = _search(u, within_prefix, size, lambda j: (j == u) | within(j, 0.0), h.shape[-1])
    origin = _take(ordered, jnp.maximum(end - 1, 0))
    position = jnp.arange(1, size + 1)
    free = (position > u) & (position <= end)
    excess = _sum(jnp.where(free, ordered - origin, 0.0))
    excess = excess - (rhs - cap * u.astype(h.dtype))
    correction = jnp.minimum(excess / jnp.maximum(end - u, 1).astype(h.dtype), 0)
    correction = jnp.maximum(correction, _next(ordered, end - 1, within_prefix - 1) - origin)
    # Without free entries the mass is flat: t = origin - cap caps the first u.
    held = (end < size) | (count <= size)
    return origin, jnp.where(end > u, correction, -cap), held


def _exact_sum(x: jax.Array, low: jax.Array | None = None) -> jax.Array:
    """Sum (plus low) over the last axis in double-double, by pairwise TwoSum."""
    n = x.shape[-1]
    x = jnp.pad(x, ((0, 0), (0, (1 << (n - 1).bit_length()) - n)))
    low = jnp.zeros_like(x[:, :1]) if low is None else low
    while x.shape[-1] > 1:
        a, b = x[:, 0::2], x[:, 1::2]
        x = a + b
        rounded = x - a
        # An infinite sum has no rounding error to carry (and inf - inf is nan).
        error = jnp.where(jnp.isfinite(x), (a - (x - rounded)) + (b - rounded), 0.0)
        low = low + _sum(error)
    return x + low


def _topk_cone(
    h: jax.Array, valid: jax.Array, k: int, ordered: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Threshold t, cap c of the projection clip(h - t, 0, c) onto 0 <= x_i <= sum(x)/k (all
    entries but one valid) from the sorted valid h or their largest ones; whether that prefix
    held the free entries and the next one, and whether c is at most the full solve's
    (nothing capped: c is the sum of the positive entries over k)."""
    dtype = h.dtype
    width = ordered.shape[-1]
    size = min(h.shape[-1] - 1, width)

    def above(t: jax.Array) -> jax.Array:
        return _sum(jnp.where(valid, jnp.maximum(h - t, 0), 0.0))

    def capped(j: jax.Array) -> jax.Array:
        # The cap level t + c = ordered[j] lies above the solution's.
        level = _take(ordered, jnp.maximum(j, 0))
        t = -above(level) / k
        return (j < 0) | (above(t) / k + 2 * t < level)

    start = jnp.full((h.shape[0], 1), -1, _INT)
    u = _search(start, start + k, k, capped, h.shape[-1]) + 1
    position = jnp.arange(width)
    head = _sum(jnp.where(position < u, ordered, 0.0))
    uf = u.astype(dtype)

    def free(j: jax.Array) -> jax.Array:
        # The threshold t = ordered[j] lies above the solution's.
        t = _take(ordered, jnp.maximum(j, 0))
        return (j < u) | (above(t) / k + 2 * t < (k * t + head) / jnp.maximum(uf, 1))

    index = _search(u - 1, jnp.full_like(u, size - 1), size, free, h.shape[-1])
    m = jnp.where(u == 0, _count(ordered > 0), index - u + 1)
    # Without capped entries every positive one is free: the prefix must end at or below 0.
    held = (h.shape[-1] - 1 <= width) | jnp.where(
        u == 0, ordered[:, width - 1 :] <= 0, index < width - 1
    )
    rest = _sum(jnp.where((position >= u) & (position < u + m), ordered, 0.0))
    ku, mf = (k - u).astype(dtype), m.astype(dtype)
    d = ku * ku + mf * uf
    t = (uf * rest - ku * head) / d
    c = (mf * head + ku * rest) / d
    # Exactly the k largest are capped when the rest lie below their threshold.
    whole = _exact_sum(ordered[:, :k])
    top = ordered[:, k - 1 : k] - whole / k
    # As topk_cone_tolerance: 16 eps relative to scores below one keeps tiny margins.
    eps = jnp.finfo(dtype).eps
    tol = 16 * eps * jnp.minimum(_sum(jnp.abs(ordered[:, :k])), 1)
    full = (u == k) | (top >= ordered[:, k : k + 1] - tol)
    t, c = jnp.where(full, top, t), jnp.where(full, whole / k, c)
    zero = whole <= tol
    return jnp.where(zero, 0, t), jnp.where(zero, 0, c), held, u == 0


def _topk_simplex(
    h: jax.Array, valid: jax.Array, k: int, rhs: jax.Array, cap: jax.Array, ordered: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Projection onto 0 <= x_i <= sum(x)/k, sum(x) <= rhs (cap = rhs/k): the cone's unless its
    sum exceeds rhs; and whether the prefix ordered held both solutions."""
    t, c, cone_held, below = _topk_cone(h, valid, k, ordered)
    origin, correction, held = _knapsack(h, valid, cap, rhs, ordered, k)
    cone = jnp.minimum(jnp.maximum(h - t, 0), c)
    full = jnp.clip((h - origin) - correction, 0, cap)
    # A prefix's c at most the full one's that already exceeds rhs/k selects the same branch.
    exceeds = k * c > rhs
    held = jnp.where(exceeds, held & (cone_held | below), cone_held)
    return jnp.where(valid, jnp.where(exceeds, full, cone), 0), held


def _knapsack_le(
    h: jax.Array, valid: jax.Array, cap: jax.Array, rhs: jax.Array, ordered: jax.Array, max_u: int
) -> tuple[jax.Array, jax.Array]:
    """Projection onto 0 <= x_i <= cap, sum(x) <= rhs over the valid entries, and whether the
    prefix ordered held the solution."""
    box = jnp.where(valid, jnp.clip(h, 0, cap), 0.0)
    origin, correction, held = _knapsack(h, valid, cap, rhs, ordered, max_u)
    full = jnp.where(valid, jnp.clip((h - origin) - correction, 0, cap), 0.0)
    within = _sum(box) <= rhs
    return jnp.where(within, box, full), within | held


def _two_simplex(
    h: jax.Array, positive: jax.Array, rhs: jax.Array, key: jax.Array, width: int | None
) -> tuple[jax.Array, jax.Array]:
    """Projection onto x >= 0 whose positive and other entries have equal sums <= rhs, from the
    width largest entries of each part (h falls along key at the positives and rises at the
    others; all for None), and whether they held the solution."""
    # A cap of 2 rhs leaves no entry capped.
    positives = _largest(h, positive, -key, width)
    others = _largest(h, ~positive, key, width)
    oa, ca, held_a = _knapsack(h, positive, 2 * rhs, rhs, positives, max_u=0)
    ob, cb, held_b = _knapsack(h, ~positive, 2 * rhs, rhs, others, 0)
    simplex = jnp.where(positive, (h - oa) - ca, (h - ob) - cb)
    # Below rhs, sum(max(h - t, 0)) over the positives = sum(max(h + t, 0)) over the others.
    if width is None:
        points = _descending(jnp.where(positive, h, -h))
        floor, ceiling = -jnp.inf, jnp.inf  # no omitted points
    else:
        # The points h of the largest positives and -h of the largest others: every point
        # between the smallest such h and the largest such -h (of parts wider than width).
        negated = jnp.where(jnp.isfinite(others), -others, -jnp.inf)
        points = _descending(jnp.concatenate([positives, negated], -1))
        floor = jnp.where(_count(positive) > width, positives[:, -1:], -jnp.inf)
        ceiling = jnp.where(_count(~positive) > width, negated[:, -1:], jnp.inf)
    size = points.shape[-1]

    def balance(t: jax.Array) -> jax.Array:
        return _sum(jnp.where(positive, jnp.maximum(h - t, 0), -jnp.maximum(h + t, 0)))

    last = jnp.full((h.shape[0], 1), size - 1, _INT)
    index = _search(
        jnp.zeros_like(last), last, size, lambda j: balance(_take(points, j)) <= 0, h.shape[-1]
    )
    origin, below = _take(points, index), _next(points, index, last)
    left, right = positive & (h > below), ~positive & (-h <= below)
    excess = _sum(jnp.where(left, h - origin, 0.0) - jnp.where(right, h + origin, 0.0))
    t = excess / jnp.maximum(_count(left) + _count(right), 1).astype(h.dtype)
    t = jnp.maximum(jnp.minimum(t, 0), below - origin)
    equal = jnp.where(positive, (h - origin) - t, (h + origin) + t)
    # Rounding of the thresholds' own terms, not a fixed floor, which would swallow tiny margins.
    scale = jnp.maximum((jnp.abs(oa) + jnp.abs(ob)) + (jnp.abs(ca) + jnp.abs(cb)), rhs)
    # In units of eps (a power of two): XLA:CPU flushes subnormal tolerances to zero.
    active = ((oa + ob) + (ca + cb)) / jnp.finfo(h.dtype).eps >= -scale
    held = held_a & held_b
    if width is not None:
        # The crossing and its neighbours lie between the omitted points.
        bracketed = (balance(points[:, :1]) <= 0) & (origin <= ceiling) & (below >= floor)
        held = held & (active | (bracketed & (index < size - 1)))
    return jnp.clip(jnp.where(active, simplex, equal), 0, rhs), held


def _offset_mean(z: jax.Array, positive: jax.Array, z0: jax.Array) -> jax.Array:
    """Mean of z - z0 over the non-target entries, accumulated in double-double. The scores are
    scaled by a power of two first: differences of scores near finfo.max and their sum
    overflow."""
    scale = 2.0 ** -(2 * z.shape[-1]).bit_length()
    total, low = difference(z * scale, z0 * scale)
    low = jnp.where(positive | jnp.isinf(total), 0, low)
    total = jnp.where(positive, 0, total)
    return _exact_sum(total, _sum(low)) / (z.shape[-1] - 1) / scale


def _entropy_topk(
    z: jax.Array, positive: jax.Array, z0: jax.Array, k: int, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Top-k entropy loss and its non-target gradient, the capped probabilities; key is z in
    the logits dtype (its order)."""
    n, dtype = z.shape[-1], z.dtype
    if k == n - 1:
        # Every competitor is capped: only their mean matters.
        mean = _offset_mean(z, positive, z0)
        loss, mass = _topk_tail(jnp.minimum(mean, 0), mean + math.log(k), jnp.full_like(mean, k))
        cap = jnp.broadcast_to(mass / k, z.shape)
        return loss[:, 0], jnp.where(positive, 0, cap)
    others = jnp.where(positive, -jnp.inf, z)
    # The k largest competitors; the others enter only through sums over the row.
    top = _take(others, top_k(jnp.where(positive, -jnp.inf, key), k)[1])
    largest, last = top[:, :1], top[:, k - 1 :]
    position = jnp.arange(k)
    # One pass over the row: the competitors below the k-th largest, relative to it, and those
    # equal to it beyond the k.
    below = _sum(jnp.where(others < last, jnp.exp(others - last), 0.0))
    extra = (_count(others == last) - _count(top == last)).astype(dtype)

    def tail(u: jax.Array) -> tuple[jax.Array, jax.Array]:
        # top[u] and the sum of the exponentials of the competitors after it relative to it
        # (its ties after position u count one each): free scores far below a capped one
        # keep their differences.
        free = _take(top, u)
        after = position > u
        terms = _sum(jnp.where(after & (top < free), jnp.exp(top - free), 0.0))
        ties = _count(after & (top == free)).astype(dtype)
        outside = jnp.exp(last - free) * (below + jnp.where(free > last, extra, 0.0))
        ties = ties + jnp.where(free == last, extra, 0.0)
        return free, (terms + outside) + ties

    def capped(j: jax.Array) -> jax.Array:
        # The native active-set test relative to top[j]: it exceeds the threshold.
        threshold = jnp.log1p(tail(jnp.maximum(j, 0))[1]) - jnp.log((k - j).astype(dtype))
        eps = jnp.finfo(dtype).eps
        return (j < 0) | (threshold + 32 * eps * jnp.maximum(jnp.abs(threshold), 1) < 0)

    start = jnp.full((z.shape[0], 1), -1, _INT)
    u = _search(start, start + k - 1, k - 1, capped, n) + 1
    free, sums = tail(u)
    lse = jnp.log1p(sums)
    head = _sum(jnp.where(position < u, top - largest, 0.0))
    rest, offset, total = (k - u).astype(dtype), largest - z0, sums + 1
    mean = (head + rest * ((free - largest) + lse - jnp.log(rest))) / k
    # As _tensor_losses._entropy_topk: rows of no capped competitor are cross-entropy's, a
    # positive offset joins the mean, and each competitor is cap * exp(z - free) / spread.
    above = jnp.maximum(offset, 0)
    scale = jnp.where(u == 0, total, jnp.exp(jnp.minimum(mean + above, 0)) * k)
    loss, mass = _topk_tail(offset - above, offset + mean + math.log(k), scale)
    cap, spread = mass / k, total / rest
    share = jnp.minimum(jnp.exp(jnp.minimum(others - free, jnp.log(spread))) / spread, 1)
    return loss[:, 0], cap * share


def _topk_tail(offset: jax.Array, odds: jax.Array, scale: jax.Array) -> tuple[jax.Array, jax.Array]:
    """As _tensor_losses._topk_tail: the loss and mass of the odds, offset + log(scale)
    (offset <= 0) where they are below -36, from squared half exponentials."""
    half = jnp.exp(offset * 0.5)
    tail = half * (half * scale)
    far = odds < -36
    loss = jnp.where(far, tail, _softplus(odds))
    return loss, jnp.where(far, tail / (1 + tail), jnp.exp(-_softplus(-odds)))


def _margin(z: jax.Array, z0: jax.Array) -> jax.Array:
    """1 + z - z0; when it cancels, with the low parts of both roundings (TwoSum), as in loss.h."""
    d, low = difference(z, z0)
    h, rounding = difference(d, -1.0)
    low = low + rounding
    cancels = (jnp.abs(h) < _MARGIN_CANCELLATION * jnp.abs(d)) & jnp.isfinite(low)
    return jnp.where(cancels, h + low, h)


def _smooth_value(h: jax.Array, a: jax.Array, gamma: jax.Array) -> jax.Array:
    """Sum of (h a - a^2/2) / gamma, scaled by a / gamma first where the products leave the normal
    range (overflow or underflow)."""
    info = jnp.finfo(h.dtype)
    ha, aa = h * a, a * a / 2
    value, tiny = jnp.abs(ha - aa), float(info.tiny) / float(info.eps)
    normal = (value <= float(info.max)) & (value >= tiny)
    return jnp.where(normal, (ha - aa) / gamma, (h - a / 2) * (a / gamma)).sum(-1)


def _hinge_topk(
    h: jax.Array, positive: jax.Array, key: jax.Array, objective: str, k: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """hinge_topk or topk_hinge: the loss, the k chosen classes, their gradient entries and
    whether candidates at the largest keys held them."""
    masked = jnp.where(positive, -jnp.inf, h)
    held = jnp.ones_like(positive[:, :1])
    # Ties take the lowest index, as a stable sort.
    if k == 1:
        top, order = masked.max(-1, keepdims=True), masked.argmax(-1, keepdims=True)
    elif masked.dtype == key.dtype or h.shape[-1] <= 2 * (k + _CANDIDATES):
        top, order = top_k(masked, k)
    else:
        # Wide margins of narrow scores: the k largest among candidates selected in the
        # narrow dtype, unless the last candidate ties the k-th largest.
        index = top_k(jnp.where(positive, -jnp.inf, key), k + _CANDIDATES)[1]
        negated, order = lax.sort((-_take(masked, index), index), num_keys=2)
        top, order = -negated[:, :k], order[:, :k]
        held = -negated[:, -1:] < top[:, -1:]
    return _hinge_entries(top, order, held, objective, k)


def _hinge_entries(
    top: jax.Array, order: jax.Array, held: jax.Array, objective: str, k: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The loss of the k largest margins top (classes order), their gradient entries and held."""
    if objective == "topk_hinge_beta":
        top = jnp.maximum(top, 0)
    value = jnp.maximum((top / k).sum(-1), 0)
    chosen = jnp.broadcast_to(value[:, None] > 0, top.shape)
    if objective == "topk_hinge_beta":
        chosen = chosen & (top > 0)
    return value, order, chosen.astype(top.dtype) / k, held


def _hinge_scores(
    z: jax.Array,
    z0: jax.Array,
    positive: jax.Array,
    work: DTypeLike,
    objective: str,
    k: int,
    first: tuple[jax.Array, jax.Array] | None = None,
    positive_of: Callable[[], jax.Array] | None = None,
) -> tuple[tuple[jax.Array, ...], Callable[[], jax.Array] | None]:
    """_hinge_topk from the k largest scores z (and the next), with margins at those only, and
    for k = 1 a finer test of the rows that do not hold.

    Margins grow with the scores, so the k largest scores (ties at the lowest index) have the k
    largest margins, unless rounding makes the k-th margin equal to that of a smaller score:
    held is false for those rows. The scores at and below the k-th then have smaller margins
    when the next candidate's does, or when it ties the k-th score and the next smaller score
    has a smaller margin. For k = 1 held tests the next smaller representable score, and the
    finer test the next smaller score of the row (another pass). first: the largest masked
    score and its first class, for k = 1 (else the plain maximum and a pass for the class,
    which the losses alone skip); positive_of: positive anew, for the finer test (a
    conditional's branch would otherwise take a copy of positive)."""
    masked = jnp.where(positive, -jnp.inf, z)
    recheck: Callable[[], jax.Array] | None = None
    if k == 1:
        if first is None:
            top, order = masked.max(-1, keepdims=True), row_first_max(masked)[1]
        else:
            top, order = first
        h = _margin(top.astype(work), z0)
        held = _margin(jnp.nextafter(top, -jnp.inf).astype(work), z0) < h

        def below() -> jax.Array:
            again = masked if positive_of is None else jnp.where(positive_of(), -jnp.inf, z)
            smaller = jnp.where(again < top, again, -jnp.inf).max(-1, keepdims=True)
            return _margin(smaller.astype(work), z0) < h

        recheck = below
    else:
        scores, order = top_k(masked, k + 1)
        h = _margin(scores.astype(work), z0)
        smaller = _margin(jnp.nextafter(scores[:, k - 1 : k], -jnp.inf).astype(work), z0)
        tie = scores[:, k : k + 1] == scores[:, k - 1 : k]
        held = (h[:, k : k + 1] < h[:, k - 1 : k]) | (tie & (smaller < h[:, k - 1 : k]))
        h, order = h[:, :k], order[:, :k]
    return _hinge_entries(h, order, held, objective, k), recheck


def _smooth_single(
    h: jax.Array,
    positive: jax.Array,
    key: jax.Array,
    objective: str,
    k: int,
    gamma: jax.Array,
    cap: jax.Array,
    width: int | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A smooth top-k hinge loss, its gradient at the other classes, and whether the width
    largest margins (at the largest keys) held the solution; cap = gamma / k."""
    valid = ~positive
    ordered = _largest(h, valid, key, width)
    if objective == "topk_hinge_alpha_smooth":
        a, held = _topk_simplex(h, valid, k, gamma, cap, ordered)
    else:
        a, held = _knapsack_le(h, valid, cap, gamma, ordered, k)
    h = jnp.where(positive, 0, h)
    return _smooth_value(h, a, gamma), a / gamma, held


def _multilabel_hinge(
    z: jax.Array,
    positive: jax.Array,
    gamma: jax.Array,
    smooth: bool,
    key: jax.Array,
    width: int | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Multilabel hinge loss and gradient, relative to the first positive score, and whether
    the width entries at the extreme keys held the solution."""
    low = jnp.where(positive, z, jnp.inf).argmin(-1, keepdims=True)
    high = jnp.where(positive, -jnp.inf, z).argmax(-1, keepdims=True)
    zlow, zhigh = _take(z, low), _take(z, high)
    position = jnp.arange(z.shape[-1])
    if not smooth:
        value = jnp.maximum(_margin(zhigh, zlow), 0)
        sign = (value > 0).astype(z.dtype)
        gradient = jnp.where(position == high, sign, jnp.where(position == low, -sign, 0.0))
        return value[:, 0], gradient, jnp.ones_like(positive[:, :1])
    z0 = _take(z, positive.argmax(-1, keepdims=True))
    d = z - z0
    h = jnp.where(positive, 0.5 - d, 0.5 + d)
    # A nearly cancelling largest pairwise margin goes in one entry, as in loss.h.
    scale = 1 + jnp.abs(zlow - z0) + jnp.abs(zhigh - z0)
    recenter = jnp.abs((zhigh - zlow) + 1) < _MARGIN_CANCELLATION * scale
    h = jnp.where(recenter, jnp.where(positive, zlow - z, _margin(z, zlow)), h)
    a, held = _two_simplex(h, positive, gamma, key, width)
    return _smooth_value(h, a, gamma), jnp.where(positive, -a, a) / gamma, held


# proximal_entropy: the block solve of include/sdca/math/entropy.h as in
# libsdca._tensor_prox, with lax.while_loop in place of host checks.


def _lambert_w_exp(x: jax.Array) -> jax.Array:
    """W(exp(x)): Halley steps on exp(u) + u = x for u = log W from log(softplus(x))."""
    fp32 = x.dtype == jnp.float32
    cut, large = (-18.0, 1e7) if fp32 else (-36.0, 1e16)

    def halley(u: jax.Array, x: jax.Array) -> jax.Array:
        e = jnp.exp(u)
        g = (e - x) + u
        return u - g / ((e + 1) - g * e / (2 * (e + 1)))

    # Two steps in float32 (from log(softplus(x)), to float32 resolution), then one in the
    # dtype of x: Halley's cubic convergence takes 1e-7 to below float64 resolution, with a
    # third of the wide exponentials of three wide steps (XLA:CPU's float64 exp and log cost
    # five times float32's).
    x32 = jnp.clip(x, cut, large).astype(jnp.float32)
    u = jnp.log(jnp.where(x32 > 30, x32, jnp.log1p(jnp.exp(x32))))
    for _ in range(2):
        u = halley(u, x32)
    u = u.astype(x.dtype)
    if not fp32:
        u = halley(u, x)
    w = jnp.exp(u)
    if not fp32:
        # A last Newton step on W itself: exp(u) loses the low bits of a large u.
        w = w - ((w - x) + jnp.log(w)) * (w / (w + 1))
    # Far below, W(exp(x)) = exp(x) to the last bit; far above, x - log x.
    w = jnp.where(x > large, x - jnp.log(x), w)
    return jnp.where(x < cut, jnp.exp(x), jnp.where(x == jnp.inf, x, w))


def _probability(h: jax.Array, alpha: jax.Array | float, log_alpha: jax.Array | float) -> jax.Array:
    """W(exp(log alpha + h)) / alpha without an underflowing W."""
    arg = h + log_alpha
    w = _lambert_w_exp(arg)
    cut = -9.0 if h.dtype == jnp.float32 else -18.0
    return jnp.where(arg < cut, jnp.exp(h - w), w / alpha)


def _residual(
    t: jax.Array,
    alpha: jax.Array | float,
    q: jax.Array,
    log_alpha: jax.Array | float,
    log_q: jax.Array,
) -> jax.Array:
    """r in [q - 1, q) with alpha r - log1p(-r / q) = t (entropy_residual)."""
    # XLA:CPU flushes subnormals: the tolerance 8 eps |s| scales f by a power of two instead,
    # and f within the change over the smallest normal r is converged (r rounds to zero).
    info = jnp.finfo(t.dtype)
    inv_tol = 1 / (8 * float(info.eps))
    steps = 4 if t.dtype == jnp.float32 else 6
    lo = q - 1
    half = alpha * q / 2
    far = (t > half + math.log(2)) | (t < -half - math.log(1.5))
    prob = _probability(log_q + (alpha * q - t), alpha, log_alpha)
    outer = jnp.minimum(jnp.maximum(q - prob, lo), q)
    # Near zero: a Pade start and Halley steps on log1p(-r / q), bracketed.
    s = jnp.where(far, 0.0, t)
    inv_q = 1 / q
    linear = s / (alpha + inv_q)
    r = linear / (1 + (linear * inv_q) / (2 * (1 + alpha * q)))
    r = jnp.minimum(jnp.maximum(r, lo), jnp.nextafter(q, jnp.zeros_like(q)))
    low, high = jnp.broadcast_to(lo, r.shape), jnp.broadcast_to(q, r.shape)
    floor = (alpha + inv_q) * float(info.tiny)
    for _ in range(steps):
        f = (alpha * r - jnp.log1p(-r * inv_q)) - s
        inv_qr = 1 / (q - r)
        d1 = alpha + inv_qr
        denom = d1 - (f * inv_qr * inv_qr) / (2 * d1)
        step = jnp.where(denom > 0, r - f / denom, r - f / d1)
        # A step below the resolution of r ends it (also one that underflows, where the
        # Halley denominator overflows): bisecting [q - 1, q) instead would lose r.
        active = (jnp.abs(f) * inv_tol > jnp.abs(s)) & (jnp.abs(f) > floor) & (step != r)
        low = jnp.where(active & (f < 0), r, low)
        high = jnp.where(active & (f >= 0), r, high)
        step = jnp.where((low < step) & (step < high), step, (low + high) / 2)
        r = jnp.where(active, step, r)
    r = jnp.where(far, outer, r)
    return jnp.where(t == 0, 0.0, jnp.where(t <= alpha * lo + log_q, lo, r))


class _Block:
    """The residual sum f(dt) of a batch of entropy blocks and its derivatives."""

    def __init__(
        self, v: jax.Array, positive: jax.Array, alpha: jax.Array, log_alpha: jax.Array, width: int
    ) -> None:
        self.v, self.positive, self.alpha, self.log_alpha = v, positive, alpha, log_alpha
        self.count = _count(positive)
        self.q = 1 / self.count.astype(v.dtype)
        self.log_q = jnp.log(self.q)
        self.clip = alpha * (self.q - 1) + self.log_q
        self.label_max = jnp.where(positive, v, -jnp.inf).max(-1, keepdims=True)
        self.other_max = jnp.where(positive, -jnp.inf, v).max(-1, keepdims=True)
        self.ref = jnp.maximum(self.label_max, self.other_max)
        finite = jnp.isfinite(v).all(-1, keepdims=True)
        self.invalid = (self.count < 1) | (self.count >= v.shape[-1]) | ~finite
        # The labels first, padded with other classes up to width (at least the label count).
        if width == 1:
            self.labels = positive.argmax(-1, keepdims=True)
        elif width < v.shape[-1]:
            self.labels = top_k(positive.astype(jnp.int8), width)[1]
        else:
            self.labels = jnp.broadcast_to(jnp.arange(width), positive.shape)
        self.used = _take(positive, self.labels)
        self.label_v = _take(v, self.labels)

    def evaluate(self, dt: jax.Array) -> tuple[jax.Array, ...]:
        """f, f', f'', magnitude, other mass, label residuals and other probabilities."""
        alpha = self.alpha
        argument = (self.ref - self.label_v) + dt
        r_label = _residual(argument, alpha, self.q, self.log_alpha, self.log_q)
        r_label = jnp.where(self.used, r_label, 0.0)
        prob = _probability(((self.v - self.ref) - dt) + self.log_q, alpha, self.log_alpha)
        counted = prob <= 1
        prob = jnp.where(self.positive, 0.0, jnp.minimum(prob, 1))
        mass = _sum(prob)
        slope = self.q - r_label
        label_denom, denom = 1 + alpha * slope, 1 + alpha * prob
        label_d1 = jnp.where(self.used & (argument >= self.clip), slope / label_denom, 0.0)
        d1 = jnp.where(counted, prob / denom, 0.0)
        derivative = _sum(label_d1) + _sum(d1)
        curvature = -_sum(label_d1 / label_denom**2) - _sum(d1 / denom**2)
        magnitude = _sum(jnp.abs(r_label)) + mass
        f = _sum(r_label) - mass
        return f, derivative, curvature, magnitude, mass, r_label, prob

    def label_log_probability(self, r_label: jax.Array, dt: jax.Array) -> jax.Array:
        """log p of the labels: log1p where r is small, else log(W / alpha) = h - W."""
        h = self.log_q + (self.alpha * self.q - ((self.ref - self.label_v) + dt))
        far = h - _lambert_w_exp(h + self.log_alpha)
        near = self.log_q + jnp.log1p(-r_label / self.q)
        return jnp.where(self.used, jnp.where(r_label <= self.q / 2, near, far), 0.0)

    def residuals(self, r_label: jax.Array, prob: jax.Array) -> jax.Array:
        """The residuals of all classes: r at the labels, -p at the others."""
        r = -prob
        return _put(r, self.labels, jnp.where(self.used, r_label, _take(r, self.labels)))

    def step(
        self, f: jax.Array, derivative: jax.Array, curvature: jax.Array, mass: jax.Array
    ) -> jax.Array:
        """Halley step, or a step in the log of the other mass where it exceeds one."""
        newton = -f / derivative
        correction = (f * curvature) / (2 * derivative * derivative)
        result = jnp.where(
            jnp.isfinite(correction), newton / (1 - jnp.clip(correction, -0.5, 0.5)), newton
        )
        others = (self.v.shape[-1] - self.count).astype(f.dtype)
        log_step = jnp.log(mass) + self.alpha * (mass - 1) / others
        jump = (f < 0) & (mass > 1) & jnp.isfinite(log_step) & (log_step > result)
        result = jnp.where(jump, log_step, result)
        valid = (derivative > 0) & jnp.isfinite(derivative) & jnp.isfinite(newton)
        return jnp.where(valid, result, 0.0)

    def bracket(self) -> tuple[jax.Array, jax.Array]:
        """dt where the largest probability is one (f <= 0) and every one <= 1/n (f >= 0)."""
        a, label, other = self.alpha, self.label_max - self.ref, self.other_max - self.ref
        lo = jnp.maximum(label - a * (1 - self.q), other - a) + self.log_q
        hi = jnp.maximum(label + a * self.q, other) + self.log_q + math.log(self.v.shape[-1])
        return lo, jnp.where(lo <= hi, hi, lo)

    def start(self) -> jax.Array:
        """The softmax-limit threshold; log q for a confident single label."""
        eps = jnp.finfo(self.v.dtype).eps
        shifted = self.v - self.ref
        first = jnp.arange(self.v.shape[-1]) == shifted.argmax(-1, keepdims=True)
        tail = _sum(jnp.where(first, 0.0, jnp.exp(shifted)))
        tiny = (
            (self.count == 1)
            & (self.label_max == self.ref)
            & ((self.other_max - self.ref) + math.log(self.v.shape[-1]) < math.log(eps) / 2)
        )
        return jnp.where(tiny, self.log_q, self.log_q + jnp.log1p(tail))


_STEPS = 200


def _solve(block: _Block) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The root dt of every block, its label residuals, other probabilities and failures."""
    # Tolerances in units of eps (powers of two) scale the residual, not the magnitude:
    # XLA:CPU flushes subnormals, so 8 eps of a tiny magnitude would be zero.
    eps = float(jnp.finfo(block.v.dtype).eps)
    lo, hi = block.bracket()
    dt = jnp.minimum(jnp.maximum(block.start(), lo), hi)
    done = block.invalid
    # A nonzero step below the resolution of dt: dt is the closest root.
    resolved = jnp.zeros_like(done)
    zero = jnp.zeros_like(dt)
    evaluation = (zero, zero, jnp.zeros_like(block.label_v), jnp.zeros_like(block.v))

    def body(state: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        step, dt, lo, hi, done, resolved = state[:6]
        f, derivative, curvature, magnitude, mass, r_label, prob = block.evaluate(dt)
        done = done | (jnp.abs(f) * (1 / (8 * eps)) <= magnitude)
        lo = jnp.where(~done & (f < 0), dt, lo)
        hi = jnp.where(~done & (f >= 0), dt, hi)
        delta = block.step(f, derivative, curvature, mass)
        following = dt + delta
        resolved = resolved | (~done & (delta != 0) & (following == dt))
        done = done | resolved
        inside = (lo <= following) & (following <= hi) & (following != dt) & jnp.isfinite(following)
        following = jnp.where(inside, following, lo + (hi - lo) / 2)
        done = done | (following == dt)
        # The last step keeps dt, so the evaluation is always at the final dt.
        dt = jnp.where(done | (step == _STEPS - 1), dt, following)
        return step + 1, dt, lo, hi, done, resolved, f, magnitude, r_label, prob

    def more(state: tuple[jax.Array, ...]) -> jax.Array:
        return (state[0] < _STEPS) & ~state[4].all()

    state: tuple[jax.Array, ...] = (jnp.zeros((), _INT), dt, lo, hi, done, resolved, *evaluation)
    _, dt, lo, hi, _, resolved, f, magnitude, r_label, prob = lax.while_loop(more, body, state)
    wide = (hi - lo) * (1 / (64 * eps)) > jnp.maximum(jnp.abs(lo), jnp.abs(hi))
    failed = ~jnp.isfinite(f) | (~resolved & (jnp.abs(f) * (1 / (256 * eps)) > magnitude) & wide)
    return dt, r_label, prob, failed | block.invalid


def _recover_tiny_tail(
    block: _Block, value: jax.Array, a: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Aggregate small tails before the final rounding, where prox equals CE to precision."""
    info = jnp.finfo(block.v.dtype)
    gap = block.other_max - block.label_max
    tail = gap + math.log(block.v.shape[-1] - 1)
    error = jnp.maximum(0.0, block.log_alpha) + tail
    bound = math.log(float(info.eps) / 8)
    tiny = (
        (block.count == 1)
        & jnp.isfinite(gap)
        & (gap < math.log(float(info.tiny)))
        & ((error < bound) | (error + tail < math.log(float(info.tiny)) + bound))
    )
    origin = jnp.where(tiny, block.other_max, 0.0)
    terms = jnp.exp(jnp.where(block.positive | ~tiny, -jnp.inf, block.v - origin))
    half = jnp.exp(jnp.where(tiny, gap, 0.0) / 2)
    mass = half * cast(jax.Array, lax.optimization_barrier(half * _sum(terms)))
    hidden = cast(jax.Array, lax.optimization_barrier(half * terms))
    recovered = jnp.where(block.positive, mass, -hidden * half)
    return jnp.where(tiny[:, 0], mass[:, 0], value), jnp.where(tiny, recovered, a)


def proximal_entropy(
    z: jax.Array, positive: jax.Array, alpha: jax.Array, log_alpha: jax.Array, width: int
) -> tuple[jax.Array, jax.Array]:
    """Loss (examples,) and gradient -a of proximal_entropy at float64 or float32 scores z.

    width bounds the label count of every row. nan where a row has no positive
    or no negative label, a nonfinite centered score, or does not converge.
    """
    v = z - z.max(-1, keepdims=True)
    block = _Block(v, positive, alpha, log_alpha, width)
    dt, r_label, prob, failed = _solve(block)
    a = block.residuals(r_label, prob)
    # Zero-sum to one rounding: the label (or the largest entry) balances the
    # others. Labels that are all zero share the recovered mass of the others.
    mass = _sum(jnp.where(positive, 0.0, a))
    labels_zero = (jnp.where(positive, a, 0.0) == 0).all(-1, keepdims=True)
    share = (block.count > 1) & labels_zero & (mass != 0)
    a = jnp.where(share & positive, -mass / block.count.astype(a.dtype), a)
    pivot = jnp.where(
        (block.count == 1) | share,
        positive.argmax(-1, keepdims=True),
        jnp.abs(a).argmax(-1, keepdims=True),
    )
    at_pivot = jnp.arange(a.shape[-1]) == pivot
    others = jnp.where(at_pivot, 0.0, a)
    a = jnp.where(at_pivot, -_sum(others), others)
    # H from the label probabilities (u = v + alpha a cancels), and alpha a_j
    # first in the penalty (a_j^2 can underflow).
    # Match the loss to the corrected one-label gradient mass.
    r_value = jnp.where(block.count == 1, -mass, r_label)
    logs = block.label_log_probability(r_value, dt)
    cross = -logs.sum(-1) * block.q[:, 0]
    # The sum can overflow while its mean is representable. Preserve ordinary
    # rounding, and keep XLA from moving this exceptional scaling past the sum.
    scaled = -lax.optimization_barrier(logs * block.q).sum(-1)
    cross = jnp.where(jnp.isfinite(cross), cross, scaled)
    value = cross + ((alpha * a) * a).sum(-1) / 2
    value, a = _recover_tiny_tail(block, value, a)
    return jnp.where(failed[:, 0], jnp.nan, value), jnp.where(failed, jnp.nan, -a)


def proximal_distillation(
    log_p: jax.Array,
    log_q: jax.Array,
    p: jax.Array,
    q: jax.Array,
    logratio: jax.Array,
    u0: jax.Array,
    alpha: float,
    invalid: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Moreau envelope min_u { H(p, softmax(u)) + |u - s|^2 / (2 alpha) } and pi - p.

    As libsdca._tensor_prox.proximal_distillation: a zero log-ratio log(q / p) gives a zero
    gradient. omega is the root of sum(r) = 0 for the residuals r = p - pi.
    """
    log_alpha = math.log(alpha)
    info = jnp.finfo(log_q.dtype)
    eps, tiny = float(info.eps), float(info.tiny)
    lo = -alpha * jnp.maximum(u0.max(-1, keepdims=True), 0.0)
    hi = -alpha * jnp.minimum(u0.min(-1, keepdims=True), 0.0)
    hi = jnp.where(lo <= hi, hi, lo)

    def evaluate(w: jax.Array) -> tuple[jax.Array, jax.Array]:
        arg = w - logratio
        prob = jnp.minimum(_probability((log_q + alpha * p) - w, alpha, log_alpha), 1.0)
        half = alpha * p / 2
        near = (p > tiny) & (arg <= half + math.log(2)) & (arg >= -half - math.log(1.5))
        p_safe, log_p_safe = jnp.where(near, p, 1.0), jnp.where(near, log_p, 0.0)
        r_near = _residual(jnp.where(near, arg, 0.0), alpha, p_safe, log_alpha, log_p_safe)
        r = jnp.where(near, r_near, jnp.where(arg == 0, 0.0, p - prob))
        return r, jnp.where(near, p - r_near, prob)

    def body(state: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        step, omega, lo, hi, done = state
        r, pi = evaluate(omega)
        f = _sum(r)
        done = done | (jnp.abs(f) * (1 / (8 * eps)) <= _sum(jnp.abs(r)))
        lo = jnp.where(~done & (f < 0), omega, lo)
        hi = jnp.where(~done & (f >= 0), omega, hi)
        denom = 1 + alpha * pi
        d1, d2 = _sum(pi / denom), -_sum(pi / denom**3)
        newton = -f / d1
        correction = (f * d2) / (2 * d1 * d1)
        delta = jnp.where(
            jnp.isfinite(correction), newton / (1 - jnp.clip(correction, -0.5, 0.5)), newton
        )
        delta = jnp.where((d1 > 0) & jnp.isfinite(d1) & jnp.isfinite(newton), delta, 0.0)
        following = omega + delta
        # A nonzero step below the resolution of omega: omega is the closest root.
        done = done | (~done & (delta != 0) & (following == omega))
        inside = (lo <= following) & (following <= hi) & (following != omega)
        following = jnp.where(inside & jnp.isfinite(following), following, lo + (hi - lo) / 2)
        done = done | (following == omega)
        return step + 1, jnp.where(done, omega, following), lo, hi, done

    def more(state: tuple[jax.Array, ...]) -> jax.Array:
        return (state[0] < _STEPS) & ~state[4].all()

    omega = jnp.clip(jnp.zeros_like(lo), lo, hi)
    state = (jnp.zeros((), _INT), omega, lo, hi, invalid | ((lo == 0) & (hi == 0)))
    omega = lax.while_loop(more, body, state)[1]
    r, _ = evaluate(omega)
    at_pivot = jnp.arange(p.shape[-1]) == (p + q).argmax(-1, keepdims=True)
    others = jnp.where(at_pivot, 0.0, r)
    r = jnp.where(at_pivot, -_sum(others), others)
    h = (log_q + alpha * p) - omega
    far_log_pi = h - _lambert_w_exp(h + log_alpha)
    near_log_pi = log_p + jnp.log1p(-r / jnp.where(p > 0, p, 1.0))
    # Far above a tiny p, -r / p overflows: log pi = log(p - r) directly.
    above = r < -p / 2
    near_log_pi = jnp.where(above, jnp.log(jnp.where(above, p - r, 1.0)), near_log_pi)
    log_pi = jnp.where(r <= p / 2, near_log_pi, far_log_pi)
    cross = -jnp.where(p > 0, p * log_pi, 0.0).sum(-1)
    return cross + ((alpha * r) * r).sum(-1) / 2, -r


def _topk_hinge(
    z: jax.Array,
    z0: jax.Array,
    label: jax.Array,
    positive: jax.Array,
    invalid: jax.Array,
    work: DTypeLike,
    objective: str,
    k: int,
    first: tuple[jax.Array, jax.Array] | None = None,
    positive_of: Callable[[], jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """The nonsmooth top-k hinge loss (in work) and gradient (in the dtype of z) of scores z,
    target scores z0 (in work) and their classes label; first and positive_of as in
    _hinge_scores."""
    dtype, position = z.dtype, jnp.arange(z.shape[-1])

    def gradient_of(order: jax.Array, entries: jax.Array) -> jax.Array:
        # The k entries in the logits dtype; the target restores their zero sum. Each
        # is 0 or one stored 1/k: the sum is their count times it, rounded once.
        stored = entries.astype(dtype)
        one = (jnp.ones((), work) / k).astype(dtype).astype(work)
        total = (-(_count(stored != 0).astype(work) * one)).astype(dtype)
        if k == 1:
            gradient = jnp.where(position == order, stored, 0)
        else:
            gradient = _put(jnp.zeros_like(z), order, stored)
        return jnp.where(position == label, total, gradient)

    if not _cpu():
        # Every margin, without the conditionals that spare that pass on the CPU.
        h = _margin(z.astype(work), z0)

        def hinge(key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            value, order, entries, held = _hinge_topk(h, positive, key, objective, k)
            return value, gradient_of(order, entries), held

        value, gradient, held = hinge(z)
        if k > 1 and z.dtype != work:
            value, gradient = lax.cond(
                (held | invalid).all(), lambda: (value, gradient), lambda: hinge(h)[:2]
            )
    else:

        def every() -> tuple[jax.Array, jax.Array]:
            # Margins of every class, ordered by themselves (converted here: XLA
            # would otherwise store the converted scores for the branch).
            h = _margin(z.astype(work), z0)
            again = positive if positive_of is None else positive_of()
            order, entries = _hinge_topk(h, again, h, objective, k)[1:3]
            return order.astype(_INT), entries

        # The k largest margins are those of the k largest scores, whichever classes
        # tie: only the subgradient may need every margin.
        scores, recheck = _hinge_scores(z, z0, positive, work, objective, k, first, positive_of)
        value, order, entries, held = scores

        def candidates() -> tuple[jax.Array, jax.Array]:
            return order.astype(_INT), entries

        def finer() -> tuple[jax.Array, jax.Array]:
            assert recheck is not None
            return lax.cond((recheck() | invalid).all(), candidates, every)

        # The branches choose the k classes; the gradient is written outside them, where
        # XLA fuses it with its uses.
        fallback = every if recheck is None else finer
        gradient = gradient_of(*lax.cond((held | invalid).all(), candidates, fallback))
    return value, gradient


def _nonfinite(z: jax.Array) -> jax.Array:
    # The largest and smallest scores (nan propagates): reductions of the scores themselves,
    # which XLA:CPU reads in place (an isfinite reduction writes its mask to memory and, on
    # jaxlib 0.12, reduces it in a slow reduce-window).
    largest, smallest = z.max(-1, keepdims=True), z.min(-1, keepdims=True)
    return ~(jnp.isfinite(largest) & jnp.isfinite(smallest))


# One computation per shape and static option, for any gamma and alpha.
@functools.partial(jax.jit, static_argnames=("objective", "k", "width", "values_only"))
def loss_and_gradient(
    z: jax.Array,
    target: jax.Array,
    scalars: jax.Array,
    objective: str,
    k: int,
    width: int,
    values_only: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Losses (examples,) and gradients (examples, classes) of float32/float64 scores z and
    integer labels (examples,) or a label mask (examples, classes), solved in the dtype of
    scalars = [gamma, gamma / k, alpha, log alpha] (as torch._tensor_loss): losses in that
    dtype, gradients rounded to the dtype of z. values_only: for the losses alone, which
    may take passes better suited to them."""
    dtype, work = z.dtype, scalars.dtype
    w = z.astype(work)
    # Opaque to XLA, which would multiply by the reciprocal of a constant divisor (infinite
    # for gamma below 5.6e-309, 0 * inf = nan) instead of dividing.
    gamma, cap, alpha, log_alpha = lax.optimization_barrier(scalars)
    position = jnp.arange(z.shape[-1])
    if target.ndim == 1:
        # Labels need no pass over a mask: out of range, they mark no class.
        label = target[:, None].astype(jnp.promote_types(target.dtype, _INT))
        positive = position == label
        count = ((label >= 0) & (label < z.shape[-1])).astype(_INT)
        # Anew in conditionals' branches, which would otherwise take a copy of positive.
        positive_of: Callable[[], jax.Array] | None
        positive_of = functools.partial(jnp.equal, position, label)
    else:
        positive, count = target, _count(target)
        label = positive.argmax(-1, keepdims=True)
        positive_of = None
    if objective == "proximal_cross_entropy":
        value, gradient = proximal_entropy(w, positive, alpha, log_alpha, width)
        return value, gradient.astype(dtype)
    if objective in _SINGLE_TARGET:
        hinge = objective in ("topk_hinge_alpha", "topk_hinge_beta")
        if hinge and k == 1 and _cpu() and not values_only:
            # z * 0 is nan exactly at a nonfinite score: the masked maximum with its first
            # class (one variadic pass, which the gradient needs) then checks every score,
            # the target's too. The losses alone take the plain maximum and check apart,
            # faster on XLA:CPU for long rows.
            first = row_first_max(jnp.where(positive, -jnp.inf, z) + z * 0)
            nonfinite = ~jnp.isfinite(first[0])
        elif hinge and k == 1 and _cpu():
            # The losses alone: the masked maximum (the hinge's own pass) is not finite
            # where another score is nan or +inf, the smallest score where one is -inf.
            top = jnp.where(positive, -jnp.inf, z).max(-1, keepdims=True)
            target_score = _take(z, label)
            finite = jnp.isfinite(top) & jnp.isfinite(z.min(-1, keepdims=True))
            first, nonfinite = None, ~(finite & jnp.isfinite(target_score))
        else:
            first, nonfinite = None, _nonfinite(z)
        invalid = (count != 1) | nonfinite
        z0 = _take(z, label).astype(work)
        if objective in ("topk_hinge_alpha", "topk_hinge_beta"):
            value, gradient = _topk_hinge(
                z, z0, label, positive, invalid, work, objective, k, first, positive_of
            )
        else:
            if objective == "topk_cross_entropy":
                value, others = _entropy_topk(w, positive, z0, k, z)
            else:
                h = _margin(w, z0)

                def smooth(width: int | None) -> tuple[jax.Array, jax.Array, jax.Array]:
                    return _smooth_single(h, positive, z, objective, k, gamma, cap, width)

                value, others = _widening(smooth, _candidates(z, k), invalid)
            # Keep entropy's target mass independently of rounded competitor entries.
            stored = others.astype(dtype)
            total = (
                -jnp.expm1(-value[:, None])
                if objective == "topk_cross_entropy"
                else _sum(stored.astype(work))
            )
            gradient = jnp.where(positive, (-total).astype(dtype), stored)
    else:
        invalid = (count == 0) | (count == z.shape[-1]) | _nonfinite(z)
        is_smooth = objective.endswith("_smooth")

        def multilabel(width: int | None) -> tuple[jax.Array, jax.Array, jax.Array]:
            return _multilabel_hinge(w, positive, gamma, is_smooth, z, width)

        value, gradient = _widening(multilabel, _candidates(z, 1) if is_smooth else (), invalid)
        gradient = gradient.astype(dtype)
    value = jnp.where(invalid[:, 0], jnp.nan, value)
    return value, jnp.where(invalid, jnp.nan, gradient)


def _candidates(z: jax.Array, k: int) -> tuple[int, ...]:
    """How many of the largest margins to try before solving from all of them."""
    return tuple(
        w for w in (max(_CANDIDATES, k + 1), 4 * max(_CANDIDATES, k + 1)) if 2 * w < z.shape[-1]
    )


def _widening(
    solve: Callable[[int | None], tuple[jax.Array, ...]],
    widths: tuple[int, ...],
    invalid: jax.Array,
) -> tuple[jax.Array, ...]:
    """solve from the first width whose candidates hold every row, else from every class; the
    wider solves run only when needed (and always under vmap, where lax.cond selects)."""
    if not widths:
        return solve(None)[:-1]
    *results, held = solve(widths[0])
    return lax.cond(
        (held | invalid).all(),
        lambda: tuple(results),
        lambda: _widening(solve, widths[1:], invalid),
    )
