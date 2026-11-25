"""The losses of docs/API.md on (rows, classes) logits of any namespace, per row.

Cross-entropy and the nonsmooth hinges are expressions the framework differentiates at any
order. Cross-entropy is formed as the kernels form it (include/sdca/loss/ce.h): the loss
from log1p of the classes below the largest, so that it stays positive past the precision
cliff, and its derivative at the label (or the largest class) as minus the mass of the
others, so that the gradient sums to zero (Barron et al., 2026: no Softmax Collapse).
Multilabel cross-entropy and distillation attach their gradient formed the same way, from
an anchor and from the log-ratio of student and teacher (_attach_smooth): the gradient keeps
its own derivatives, so they differentiate at any order too. The smoothed hinges, top-k
entropy and the proximal losses solve for their maximizer or envelope point and attach its
known gradient (_attach): first order only.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, overload

from .._limits import limits
from ._entropy import all_capped, difference, proximal_entropy, topk_entropy
from ._entropy import proximal_distillation as _proximal_distillation
from ._project import exact_sum, knapsack_le, topk_simplex, two_simplex
from ._xp import Array, Namespace

WIDE_DISTILLATION_CLASSES = 2048

_INF = float("inf")
_NAN = float("nan")
# Margins below this fraction of their terms cancel.
# The smoothed multilabel hinge recentres its margins on the lowest positive where the largest
# pairwise margin falls below this fraction of its terms (multilabel.h: kRecenter).
_RECENTER = 1 / 16


def _labels(xp: Namespace, z: Array, target: Array) -> tuple[Array, Array, Array]:
    """Labels (rows, 1) clipped into range, whether they were in range, and their one-hot
    mask."""
    n = z.shape[-1]
    label = target[:, None]
    checked = xp.eager_labels(label, z)
    if checked is not None:
        return checked, True, xp.arange(n, z) == checked
    inside = (label >= 0) & (label < n)
    label = xp.where(inside, label, 0)
    return label, inside, (xp.arange(n, z) == label) & inside


def _maximum(xp: Namespace, z: Array) -> Array:
    """The largest score per row, with NaN or +inf propagated."""
    # Some XLA CPU maximum reductions discard NaNs inside a vector. Map those lanes to
    # +inf, which also invalidates the row and survives the same maximum reduction.
    if xp.name == "jax":
        z = xp.where(xp.isnan(z), _INF, z)
    return xp.max(z)


def _open(xp: Namespace, z: Array) -> Array:
    """Rows (rows, 1) without NaN or +inf, and with a finite score."""
    high = _maximum(xp, z)
    return (high < _INF) & (high > -_INF)


def _finite_at(xp: Namespace, z: Array, mask: Array) -> Array:
    """Rows whose scores at mask are not masked."""
    return xp.all(~mask | (z > -_INF))


def _scores(xp: Namespace, z: Array, valid: Array) -> Array:
    """z where its derivative is taken apart from the loss: masked classes of valid rows as
    0 (their gradient is +0.0), every entry of invalid rows as given (a NaN gradient)."""
    return xp.where(valid & ~xp.isfinite(z), 0.0, z)


def _finish(xp: Namespace, value: Array, z: Array, valid: Array) -> Array:
    """value (rows,) of a differentiated loss of z: NaN at the invalid rows, with NaN
    gradients, and +0.0 at the gradient entries that every path leaves zero (-0.0 from a
    negative cotangent plus this +0.0)."""
    poison = xp.stop(xp.where(valid, 0.0, _NAN))
    extra = xp.sum(_scores(xp, z, valid) * poison + xp.where(xp.isnan(z), z, 0.0))
    return xp.where(valid[:, 0], value, _NAN) + extra[:, 0]


def _attach(xp: Namespace, value: Array, z: Array, gradient: Array, valid: Array) -> Array:
    """value (rows,) with the gradient (rows, classes) at z, first order; NaN rows where not
    valid."""
    g = xp.stop(xp.where(valid, xp.where(gradient == 0, 0.0, gradient), _NAN))
    return xp.attach(xp.stop(xp.where(valid[:, 0], value, _NAN)), z, g)


def _infinite(xp: Namespace, z: Array, gradient: Array, limit: Array, value: Array) -> Array:
    """value (rows,), but +inf with the attached gradient at the rows limit (rows, 1): a masked
    label, positive class or student class of teacher probability (docs/API.md, Masked
    classes)."""
    g = xp.stop(xp.where(limit, gradient, 0.0))
    infinite = xp.attach(xp.stop(xp.full_like(value, _INF)), z, g)
    return xp.where(limit[:, 0], infinite, value)


def _stand_in(xp: Namespace, z: Array, at: Array, offset: float) -> Array:
    """z with the classes at given a finite logit offset below the row's lowest finite one, in
    the regime of the limit as they go to -inf."""
    low = xp.min(xp.where(xp.isfinite(z), z, _INF))
    return xp.where(at, low - offset, z)


def _margin(xp: Namespace, x: Array, y: Array) -> Array:
    """1 + x - y with the rounding errors of x - y and of the sum (TwoSum) added back, as
    topk.h's margin: within an ulp of the exact margin, its derivatives those of 1 + x - y."""
    d = x - y
    h = d + 1
    rd, rh = d - x, h - d
    error = ((x - (d - rd)) + (-y - rd)) + ((d - (h - rh)) + (1 - rh))
    return xp.where(xp.isfinite(h), h + error, h)


def mean_margin(xp: Namespace, best: Array, label: Array) -> tuple[Array, Array]:
    """Average signed margin and a sum with the same sign before rounding the average.
    Retain the unit when selected scores cancel, and compute small margins about label - 1.
    A power-of-two scaling bounds differences and their sum on extreme rows."""
    k = best.shape[-1]
    origin, bias = difference(label, 1.0)
    bound = xp.finfo(best.dtype).max / (2 * k)
    large = (best[:, :1] > bound) | (best[:, -1:] < -bound) | (xp.abs(origin) > bound)
    scale = xp.where(large, 2.0 ** -(2 * k).bit_length(), 1.0)
    high, low = difference(best * scale, origin * scale)
    low = xp.sum(xp.where(xp.isfinite(high), low, 0.0)) - bias * (k * scale)
    total = exact_sum(xp, high, low)
    return total / (k * scale), total


def _attach_smooth(xp: Namespace, value: Array, z: Array, gradient: Array, valid: Array) -> Array:
    """value (rows,) with the gradient (rows, classes) at z, which keeps its derivatives in z;
    NaN rows where not valid."""
    g = xp.where(valid, gradient, _NAN)
    return xp.attach_smooth(xp.stop(xp.where(valid[:, 0], value, _NAN)), z, g)


def _attach_hinge(xp: Namespace, value: Array, z: Array, gradient: Array, valid: Array) -> Array:
    """The chosen piecewise-constant slope has zero higher derivatives. Keep its dependence
    on the scores for eager double backward, including finite substitutes at masked entries."""
    anchor = z[:, :1]
    g = gradient + xp.where(xp.isfinite(anchor), anchor, 0.0) * 0.0
    return _attach_smooth(xp, value, z, g, valid)


def _tail(xp: Namespace, x: Array, e: Array, scale: float = 1.0) -> Array:
    """sum_j exp(x_j) - 1 from e = scale * exp(x), x <= 0 the scores less their maximum.
    Sum before undoing the scale so subnormal entries can combine. The classes tied at the
    maximum add exactly 1 each."""
    rest = xp.sum(xp.where(x < 0, e, 0.0))
    if scale != 1.0:
        rest = rest / scale
    return rest + (xp.astype(xp.count(x == 0), x.dtype) - 1)


def _ties(xp: Namespace, x: Array, at: Array) -> Array:
    """1 / (their count) at the entries at of each row, 0 elsewhere (without derivatives)."""
    return xp.stop(xp.where(at, 1.0 / xp.maximum(xp.astype(xp.count(at), x.dtype), 1.0), 0.0))


def _smooth_value(xp: Namespace, h: Array, a: Array, gamma: float) -> Array:
    """Sum of (h a - a^2/2) / gamma, scaled by a / gamma first where the products leave the
    normal range."""
    lim = limits(h.dtype)
    ha, aa = h * a, a * a / 2
    value = xp.abs(ha - aa)
    normal = (value <= lim.max) & (value >= lim.min_over_eps)
    return xp.sum(xp.where(normal, (ha - aa) / gamma, (h - a / 2) * (a / gamma)))[:, 0]


def _log1p(xp: Namespace, x: Array) -> Array:
    """log1p(x), x itself below the smallest normal (PyTorch's CPU log1p loses it there)."""
    return xp.where(x < limits(x.dtype).tiny, x, xp.log1p(x))


def _scaled_log1p(xp: Namespace, half: Array, s: Array, weight: Array | None) -> Array:
    """weight log1p(t) (rows, 1) of t = half (half s) in [0, 1], rounded once from normal
    factors where exp(gap) = half^2 is subnormal; below the smallest normal, where
    log1p(t) = t, (half weight) (half s), whose derivatives round once too (XLA flushes a
    subnormal t)."""
    t = half * (half * s)
    small = t < limits(t.dtype).tiny
    if weight is None:
        return xp.where(small, t, xp.log1p(t))
    return xp.where(small, (half * weight) * (half * s), weight * xp.log1p(t))


def _competitors(xp: Namespace, z: Array, label: Array, positive: Array) -> tuple[Array, ...]:
    """exp(z_j - m) of the classes but the label y, from their largest score m (0 at y), their
    sum s in [1, C) and the gap m - z_y; m carries no derivative (the losses are shift
    invariant). A row without a finite other class has s = 0."""
    others = xp.where(positive, -_INF, z)
    top = xp.stop(xp.max(others))
    e = xp.exp(others - xp.where(top > -_INF, top, 0.0))
    s = xp.sum(e)
    return e, s, xp.where(s > 0, top - xp.take(z, label), 0.0)


@overload
def _label_entropy(
    xp: Namespace, s: Array, gap: Array, weight: Array | None = None, *, mass: Literal[False] = ...
) -> Array: ...


@overload
def _label_entropy(
    xp: Namespace, s: Array, gap: Array, weight: Array | None = None, *, mass: Literal[True]
) -> tuple[Array, Array]: ...


def _label_entropy(
    xp: Namespace, s: Array, gap: Array, weight: Array | None = None, *, mass: bool = False
) -> Array | tuple[Array, Array]:
    """weight log(1 + t) (rows, 1) of t = exp(gap) s, with s and gap of _competitors the loss
    log(1 + sum_{j != y} exp(z_j - z_y)) of labels y (ce.h, label_statistics): its derivative
    is minus the others' mass t / (1 + t) at the label and each other class's own exponential
    elsewhere, never 1 - softmax(z)_y. With mass, also t / (1 + t)."""
    odds = gap + xp.log(xp.where(s > 0, s, 1.0))
    near = odds <= 0
    half = xp.exp(xp.where(near, gap, 0.0) / 2)
    far = xp.where(near, 0.0, odds)
    above = far + xp.log1p(xp.exp(-far))
    value = xp.where(
        near, _scaled_log1p(xp, half, s, weight), above if weight is None else weight * above
    )
    if not mass:
        return value
    t = half * (half * s)
    return value, xp.where(near, t / (1 + t), 1 / (1 + xp.exp(-far)))


def _target_entropy(xp: Namespace, z: Array, u: Array) -> Array:
    """-sum_j u_j log softmax(z)_j (rows, 1) of target weights u >= 0 (rows, classes), as
    sum_j u_j (m - z_j) + T log1p(sum_{j != j0} exp(z_j - m)) from the first largest score
    m = z_j0 and T = sum_j u_j (ce.h, target_row): its derivative at j0 is the others' target
    weight minus T times their mass. The sum is exp(m2 - m) s from the second largest score
    m2 (as _label_entropy). A term of target weight zero is zero, also at a masked class."""
    index = xp.argmax(z)
    first = xp.arange(z.shape[-1], z) == index
    top = xp.take(z, index)
    _, s, gap = _competitors(xp, z, index, first)
    gaps = xp.sum(xp.where((u != 0) & ~first, u * (z - top), 0.0))
    return _scaled_log1p(xp, xp.exp(gap / 2), s, xp.sum(u)) - gaps


def cross_entropy(
    xp: Namespace,
    logits: Array,
    target: Array,
    *,
    weight: Array | None = None,
    ignore_index: int | None = -100,
    label_smoothing: float | Array = 0.0,
) -> Array:
    """torch.nn.functional.cross_entropy per row of labels (rows,) or probabilities (rows,
    classes): w_y (1 - eps) (lse(z) - z_y) + eps / C sum_j w_j (lse(z) - z_j); rows of
    ignore_index are 0 (label_weights gives the divisor of their mean)."""
    z = xp.work(logits)
    if target.shape == logits.shape:
        return _probability_targets(xp, z, target, weight, label_smoothing)
    n = z.shape[-1]
    ignored = (target[:, None] == ignore_index) if ignore_index is not None else None
    label, inside, positive = _labels(
        xp, z, target if ignored is None else xp.where(ignored[:, 0], 0, target)
    )
    valid = inside & _open(xp, z)
    if ignored is not None:
        valid = valid | ignored
        z = xp.where(ignored, 0.0, z)
    w = None if weight is None else xp.astype(weight, z.dtype)[None, :]
    if _zero(label_smoothing):
        _, s, gap = _competitors(xp, z, label, positive)
        loss = _label_entropy(xp, s, gap, None if w is None else xp.take(w, label))
    else:
        # Smoothing as target weights: w_j eps / C at every class, (1 - eps) w_y more at y.
        u = xp.astype(positive, z.dtype) * (1 - label_smoothing) + label_smoothing / n
        loss = _target_entropy(xp, z, u if w is None else u * w)
    if ignored is not None:
        loss = xp.where(ignored, 0.0, loss)
    # A zero class weight times an infinite loss (a masked label) is NaN: an invalid row.
    return _finish(xp, loss[:, 0], z, valid & ~xp.isnan(loss))


def label_weights(
    xp: Namespace,
    logits: Array,
    target: Array,
    *,
    weight: Array | None = None,
    ignore_index: int | None = -100,
) -> Array:
    """The weight (rows,) of each label row in the mean of cross_entropy: w_y, 0 where
    ignored."""
    z = xp.work(logits)
    label = xp.clip(target[:, None], 0, z.shape[-1] - 1)
    w = (
        xp.full_like(label, 1.0, dtype=z.dtype)
        if weight is None
        else xp.take(xp.astype(weight, z.dtype)[None, :], label)
    )
    if ignore_index is not None:
        w = xp.where(target[:, None] == ignore_index, 0.0, w)
    return w[:, 0]


def _zero(x: float | Array) -> bool:
    return (isinstance(x, int) or isinstance(x, float)) and x == 0


def _probability_targets(
    xp: Namespace, z: Array, t: Array, weight: Array | None, eps: float | Array
) -> Array:
    """-sum_j u_j log softmax(z)_j with u = w ((1 - eps) t + eps / C); a term of target
    weight zero is zero, also at a masked class."""
    t = xp.astype(t, z.dtype)
    valid = _open(xp, z) & xp.all(xp.isfinite(t) & (t >= 0))
    u = t if _zero(eps) else t * (1 - eps) + eps / z.shape[-1]
    if weight is not None:
        u = u * xp.astype(weight, z.dtype)[None, :]
    return _finish(xp, _target_entropy(xp, z, u)[:, 0], z, valid)


def topk_cross_entropy(xp: Namespace, logits: Array, target: Array, *, k: int = 1) -> Array:
    """Top-k entropy of labels (rows,): cross_entropy at k = 1."""
    if k == 1:
        return cross_entropy(xp, logits, target, ignore_index=None)
    z = xp.work(logits)
    label, inside, positive = _labels(xp, z, target)
    open_ = inside & _open(xp, z)
    r = xp.count(~positive & (z > -_INF))
    masked = open_ & (xp.take(z, label) == -_INF)
    valid = open_ & ((r > 0) | ~masked)
    if k == z.shape[-1] - 1:
        # Every competitor has the same share, so neither sorting nor a cap search is needed.
        zs = xp.stop(xp.where(valid, z, 0.0))
        target_score = xp.where(masked, 0.0, xp.take(zs, label))
        value, mass = all_capped(xp, zs, ~positive & (zs > -_INF), target_score, r)
        mass = xp.where(r == k, xp.where(masked, 1.0, mass), 0.0)
        value = xp.where(r == k, xp.where(masked, _INF, value), 0.0)
        gradient = xp.where(positive, -mass, mass / k)
        return _attach(xp, value[:, 0], z, gradient, valid)
    # A masked label: a stand-in far below the others gives the limit's gradient (mass 1).
    # exp(-gap) of a gap beyond -log(denorm_min) is zero: the mass is 1, the limit's.
    deep = 1 - limits(z.dtype).log_denorm_min
    zs = xp.stop(_stand_in(xp, xp.where(valid, z, 0.0), positive & masked, deep))
    value, others = topk_entropy(xp, zs, ~positive & (zs > -_INF), xp.take(zs, label), k)
    # Where the largest other class takes at most a k-th of the others' mass, nothing is
    # capped: cross-entropy, from its label statistics (as topk.h).
    e, s, gap = _competitors(xp, zs, label, positive)
    ce, mass = _label_entropy(xp, s, gap, mass=True)
    uncapped = s >= k
    others = xp.where(uncapped, e * (mass / xp.where(uncapped, s, 1.0)), others)
    # Else the label balances the others, unless they may be subnormal (XLA flushes them).
    total = xp.sum(others)
    flushed = total < limits(z.dtype).tiny * z.shape[-1]
    label_entry = xp.where(uncapped, -mass, xp.where(flushed, xp.expm1(-value)[:, None], -total))
    gradient = xp.where(positive, label_entry, others)
    value = xp.where(uncapped[:, 0], ce[:, 0], value)
    value = xp.where((masked & valid & (r >= k))[:, 0], _INF, value)
    return _attach(xp, value, z, gradient, valid)


def multiclass_hinge(xp: Namespace, logits: Array, target: Array, *, gamma: float = 0.0) -> Array:
    """max(0, 1 + max_{j != y} z_j - z_y) of labels (rows,), smoothed by gamma > 0."""
    return topk_hinge(xp, logits, target, k=1, gamma=gamma)


def topk_hinge(
    xp: Namespace,
    logits: Array,
    target: Array,
    *,
    k: int = 1,
    variant: str = "beta",
    gamma: float = 0.0,
) -> Array:
    """Top-k hinge of labels (rows,) on the k largest margins 1 + z_j - z_y (j != y), of
    variant "alpha" (their mean) or "beta" (the mean of their hinges), smoothed by
    gamma > 0."""
    z = xp.work(logits)
    label, inside, positive = _labels(xp, z, target)
    open_ = inside & _open(xp, z)
    others = ~positive & (z > -_INF)
    target_score = xp.take(z, label)
    masked = open_ & (target_score == -_INF)
    # The limit as the label's logit goes to -inf: alpha with fewer than k others stays 0.
    zero = masked & (xp.count(others) < k) if variant == "alpha" and k > 1 else masked & False
    limit = masked & ~zero
    valid = open_ & (target_score > -_INF)
    if gamma > 0:
        # A stand-in label below the others by more than the cap: every margin exceeds it.
        zs = _stand_in(xp, xp.where(valid | masked, z, 0.0), positive & masked, gamma / k + 1)
        return _smooth_topk_hinge(
            xp, z, zs, label, positive, valid | masked, limit, k, variant, gamma
        )
    scores = xp.where(others, z, -_INF)
    best = xp.max(scores) if k == 1 else xp.topk(scores, k, stable=False)[0]
    zy = xp.where(valid, target_score, 0.0)
    if k == 1:
        signed = total = _margin(xp, best, zy)
    elif variant == "beta":
        top = _margin(xp, best, zy)
        # A difference can overflow before its finite share of the mean is taken.
        share = xp.where(xp.isfinite(top), top / k, best / k - zy / k + 1.0 / k)
        total = xp.sum(xp.maximum(share, 0.0))
        signed = _margin(xp, scores, zy)
    else:
        total, signed = xp.mean_margin(best, zy)
    value = xp.where(limit, _INF, xp.where(zero, 0.0, xp.maximum(total, 0.0)))
    # Select by score before rounding any margins. A zero hinge shares its derivative
    # equally with the active branch, including all co-winners when k is one.
    cutoff = best[:, -1:]
    tied = others & (scores == cutoff)
    count = xp.astype(xp.count(tied), z.dtype)
    if k == 1:
        weights = xp.where(tied, 1.0 / xp.maximum(count, 1.0), 0.0)
        grad = xp.where(
            signed == 0,
            xp.where(tied, 1.0 / (count + 1.0), 0.0),
            xp.where(signed > 0, weights, 0.0),
        )
    else:
        above = scores > cutoff
        places = xp.astype(k - xp.count(above), z.dtype)
        weights = xp.where(above, 1.0, xp.where(tied, places / xp.maximum(count, 1.0), 0.0)) / k
        grad = weights * xp.where(signed > 0, 1.0, xp.where(signed == 0, 0.5, 0.0))
    grad = xp.where(limit, weights, xp.where(zero, 0.0, grad))
    return _attach_hinge(
        xp, value[:, 0], z, xp.where(positive, -xp.sum(grad), grad), valid | masked
    )


def _smooth_topk_hinge(
    xp: Namespace,
    z: Array,
    zs: Array,
    label: Array,
    positive: Array,
    valid: Array,
    limit: Array,
    k: int,
    variant: str,
    gamma: float,
) -> Array:
    """max_p <h, p> - gamma/2 |p|^2 over the top-k simplex (alpha) or the capped simplex
    {0 <= p <= 1/k, sum(p) <= 1} (beta); the gradient is the maximizer p. zs: z with a
    stand-in for a masked label at the rows limit, whose loss is +inf."""
    zs = xp.stop(xp.where(valid, zs, 0.0))
    others = ~positive & (zs > -_INF)
    h = xp.where(others, _margin(xp, zs, xp.take(zs, label)), -_INF)
    if variant == "alpha" and k > 1:
        a = topk_simplex(xp, h, others, k, gamma)
    else:
        a = knapsack_le(xp, h, others, gamma / k, gamma, 0 if k == 1 else k)
    value = xp.where(limit[:, 0], _INF, _smooth_value(xp, xp.where(others, h, 0.0), a, gamma))
    g = a / gamma
    return _attach(xp, value, z, xp.where(positive, -xp.sum(g), g), valid)


def multilabel_hinge(xp: Namespace, logits: Array, target: Array, *, gamma: float = 0.0) -> Array:
    """max(0, 1 + max_{j not in Y} z_j - min_{y in Y} z_y) of label masks (rows, classes),
    smoothed by gamma > 0."""
    z = xp.work(logits)
    positive = target
    sized = _open(xp, z) & (xp.count(positive) > 0) & (xp.count(~positive) > 0)
    valid = sized & _finite_at(xp, z, positive)
    # Masked positives: +inf where a negative is left (the limit as they go to -inf
    # together), else 0.
    masked = positive & (z == -_INF)
    negative = ~positive & (z > -_INF)
    some = sized & ~valid
    limit = some & (xp.count(negative) > 0)
    zero = some & ~limit
    if gamma > 0:
        # The negatives' projection onto {y >= 0, sum y = gamma}: lifted so the largest
        # exceeds gamma, the bound sum y <= gamma holds with equality.
        scores = xp.stop(xp.where(negative, z, -_INF))
        lift = xp.where(negative, scores - xp.max(scores) + (gamma + 1), 0.0)
        a = knapsack_le(xp, lift, negative, gamma, gamma)
        tail = xp.where(negative, a / gamma, -1.0 / xp.astype(xp.count(masked), z.dtype))
        lifted = xp.where(masked | negative, tail, 0.0)
    else:
        highs = xp.where(negative, z, -_INF)
        lifted = _ties(xp, z, negative & (z == xp.max(highs))) - _ties(xp, z, masked)
    zz = xp.where(valid, z, 0.0)
    lows, highs = xp.where(positive, zz, _INF), xp.where(positive, -_INF, zz)
    zlow, zhigh = xp.min(lows), xp.max(highs)
    if gamma <= 0:
        h = _margin(xp, zhigh, zlow)
        slope = xp.where(h > 0, 1.0, xp.where(h == 0, 0.5, 0.0))
        grad = slope * (
            _ties(xp, z, negative & (zz == zhigh)) - _ties(xp, z, positive & (zz == zlow))
        )
        grad = xp.where(limit, lifted, xp.where(zero, 0.0, grad))
        value = xp.where(limit, _INF, xp.where(zero, 0.0, xp.maximum(h, 0.0)))
        return _attach_hinge(xp, value[:, 0], z, grad, valid | some)
    zs, zlow, zhigh = xp.stop(zz), xp.stop(zlow), xp.stop(zhigh)
    negative = ~positive & (zs > -_INF)
    z0 = xp.take(zs, xp.argmax(positive))
    d = zs - z0
    h = xp.where(positive, 0.5 - d, 0.5 + d)
    # A nearly cancelling largest pairwise margin goes in one entry.
    scale = 1 + xp.abs(zlow - z0) + xp.abs(zhigh - z0)
    recenter = xp.abs((zhigh - zlow) + 1) < _RECENTER * scale
    h = xp.where(recenter, xp.where(positive, zlow - zs, _margin(xp, zs, zlow)), h)
    a = two_simplex(xp, h, positive, negative, gamma)
    value = _smooth_value(xp, xp.where(positive | negative, h, 0.0), a, gamma)
    value = xp.where(limit[:, 0], _INF, xp.where(zero[:, 0], 0.0, value))
    gradient = xp.where(some, xp.where(limit, lifted, 0.0), xp.where(positive, -a, a) / gamma)
    return _attach(xp, value, z, gradient, valid | some)


def multilabel_cross_entropy(xp: Namespace, logits: Array, target: Array) -> Array:
    """lse(z) - mean_{y in Y} z_y of label masks (rows, classes): mean_{y in Y} (m - z_y) +
    log1p(tail) from the largest score m, and its gradient softmax(z) - 1 / |Y| at the
    positives on the anchor a = m + log((1 + tail) / |Y|) (ce.h, MaskWeights): there the
    u_j = exp(z_j - a) sum to |Y| + S with S small, and a positive's entry is
    (expm1(z_j - a) - S / |Y|) / (|Y| + S), without the cancellation of softmax(z)_j - 1 / |Y|.
    The entries are exact for any anchor, which carries no derivative."""
    z = xp.work(logits)
    positive = target
    highest = xp.stop(_maximum(xp, z))
    count = xp.astype(xp.count(positive), z.dtype)
    valid = (highest < _INF) & (highest > -_INF) & (count > 0)
    x = xp.where(valid, z - highest, 0.0)
    # Scale the exponential sum before individual negative-class probabilities round to
    # subnormals. Ordinary exponents are unchanged; a tail is the product of two normal
    # half-exponentials, with the exact power of two applied before the product.
    scale = 2.0**64
    far = x < limits(z.dtype).log_min
    e = xp.exp(xp.where(far, x * 0.5, x))
    scaled = e * scale
    wide = xp.where(far, scaled * e, scaled)
    tail = _tail(xp, xp.stop(x), xp.stop(wide), scale)
    # Compute the mean positive gap on a common power-of-two scale. Finite scores
    # at opposite storage extremes can overflow before their mean is taken.
    gap_scale = 1.0 / scale
    gaps = highest * gap_scale - z * gap_scale
    value = xp.sum(xp.where(positive, gaps, 0.0)) / (count * gap_scale) + _log1p(xp, tail)
    # log((1 + tail) / |Y|), relative to its size where the positives lead: 0 for |Y| tied
    # positives far above the others.
    shift = xp.log1p((tail - (count - 1)) / count)
    # Round the common anchor once before subtracting it from each positive score.
    # This preserves small differences between near-equal positive probabilities.
    pivot = highest + shift
    rest = xp.expm1(xp.where(valid, z - pivot, 0.0))
    u = xp.where(positive, rest * scale, wide * xp.exp(highest - pivot))
    s = xp.sum(u)
    gradient = ((u - xp.where(positive, s / count, 0.0)) / (count + s / scale)) / scale
    return _attach_smooth(xp, value[:, 0], z, gradient, valid)


def proximal_cross_entropy(
    xp: Namespace, logits: Array, target: Array, *, gamma: float = 1.0
) -> Array:
    """The Moreau envelope of cross-entropy with step gamma, of labels (rows,) or label
    masks (rows, classes) (the uniform target on the labels). A row of several labels needs
    another class that is not masked."""
    z = xp.work(logits)
    n = z.shape[-1]
    if target.shape == z.shape:
        positive = target
        labels = xp.arange(n, z)[None, :]
        count = xp.count(positive)
        others = xp.count(~positive & (z > -_INF))
        valid = (count == 1) | ((count > 0) & (others > 0))
    else:
        labels, valid, positive = _labels(xp, z, target)
    base = valid & _open(xp, z)
    valid = base & _finite_at(xp, z, positive)
    # Masked labels beside a class left: a stand-in 2 gamma - log(denorm_min) + 1 below the
    # classes left, whose pi is then below exp(log(denorm_min) - 1), zero in the work type,
    # gives the limit's gradient (loss +inf).
    limit = base & ~valid & (xp.count(~positive & (z > -_INF)) > 0)
    masked = positive & (z == -_INF) & limit
    zs = xp.where(valid | limit, z, 0.0)
    zs = xp.stop(_stand_in(xp, zs, masked, 2 * gamma + 1 - limits(z.dtype).log_denorm_min))
    value, gradient, failed = proximal_entropy(xp, zs, positive, labels, gamma, ~(valid | limit))
    value = xp.where(limit[:, 0], _INF, value)
    return _attach(xp, value, z, gradient, (valid | limit) & ~failed)


class _Rows(NamedTuple):
    """The rows of distillation (include/sdca/loss/distill.h) at temperature T, with the row
    maxima m_z and m_t (m_z without derivative: what depends on z is shift invariant)."""

    z: Array  # the student's work logits
    valid: Array  # finite rows whose student masks no class the teacher keeps
    limit: Array  # a masked student class has teacher mass
    s: Array | None  # (z - m_z) / T for the proximal solve
    e: Array | None  # exp(s) for the proximal solve
    lse: Array  # log sum exp(s) = log1p(the sum without the first largest term)
    cross: Array  # sum_j p_j s_j
    q: Array  # softmax(z / T)
    log_p: Array | None  # log softmax(t / T) for the proximal solve
    p: Array  # softmax(t / T)
    lognorm: Array  # student log partition for the proximal solve
    logratio: Array  # log(q / p) from the difference of the unnormalized partitions


def _proximal_logratio(
    xp: Namespace,
    e: Array,
    et: Array,
    t_tail: Array,
    delta: Array,
    temperature: float,
    p: Array,
    shared: Array,
    lse_t: Array,
    lse: Array,
    s: Array,
    tn: Array,
) -> tuple[Array, Array]:
    """Center the ratio on the teacher's mean displacement for the proximal solve."""
    mean = xp.sum(xp.where(et > 0, et * delta, 0.0)) / (1 + t_tail)
    dn = (delta - mean) / temperature
    small = (xp.abs(dn) < 1) & shared
    em = xp.expm1(xp.where(small, dn, 0.0))
    shift = -mean / temperature - lse_t
    # exp(shift) sum exp(s) over |dn| >= 1, from shift + log(sum) where exp(shift) overflows;
    # C is +inf where the term does, with finite factors for the derivatives.
    far = xp.sum(xp.where(small, 0.0, e))
    bound = limits(shift.dtype).log_quarter_max
    log_far = shift + xp.log(xp.where(far > 0, far, 1.0))
    huge = shift > bound
    term = xp.where(
        huge,
        xp.where(far > 0, xp.exp(xp.minimum(log_far, bound)), 0.0),
        xp.exp(xp.minimum(shift, bound)) * far,
    )
    c = xp.where(
        huge & (far > 0) & (log_far > bound), _INF, xp.sum(xp.where(small, p * em, -p)) + term
    )
    lc = xp.log1p(xp.stop(c))
    finite = xp.isfinite(lc)
    lognorm = xp.where(finite, lc - shift, lse)
    logratio = xp.where(finite, xp.stop(dn) - lc, (s - lognorm) - (tn - lse_t))
    return lognorm, logratio


def _tempered_gap(xp: Namespace, z: Array, top: Array, temperature: float) -> Array:
    """(z - top) / T, including finite gaps whose unscaled subtraction overflows."""
    if isinstance(temperature, int) or isinstance(temperature, float):
        if temperature <= 1:
            return (z - top) / temperature
        # Halving both finite endpoints first puts their difference in range.
        return (z * 0.5 - top * 0.5) / (temperature * 0.5)
    gap = z - top
    overflow = (gap == -_INF) & xp.isfinite(z) & (temperature > 1)
    return xp.where(overflow, z / temperature - top / temperature, gap / temperature)


def _distillation_rows(
    xp: Namespace, logits: Array, teacher: Array, temperature: float, *, proximal: bool = False
) -> _Rows:
    """Centered student and teacher distributions. Sum the difference of their
    unnormalized probabilities without subtracting near-equal exponentials."""
    dtype = xp.result_type(logits, teacher)
    z = xp.work(xp.astype(logits, dtype))
    t = xp.work(xp.astype(teacher, dtype))
    if proximal or xp.name != "jax":
        t = xp.stop(t)
    student_max = _maximum(xp, z)
    teacher_max = _maximum(xp, t)
    valid = xp.isfinite(student_max) & xp.isfinite(teacher_max)
    student_kept, teacher_kept = z > -_INF, t > -_INF
    limit = valid & ~xp.all(~teacher_kept | student_kept)
    zz, tt = xp.where(valid, z, 0.0), xp.where(valid, t, 0.0)
    top = xp.stop(xp.where(valid, student_max, 0.0))
    high = xp.where(valid, teacher_max, 0.0)
    wide = not proximal and z.dtype == xp.float64 and z.shape[1] >= WIDE_DISTILLATION_CLASSES
    student_base, teacher_base = top, high
    if wide:
        # A common origin preserves the signed difference between closely normalized logits.
        # Keep the two exponent scales within a factor of e.
        joint = xp.abs(top - high) <= temperature
        center = xp.stop(xp.maximum(top, high))
        student_base, teacher_base = xp.where(joint, center, top), xp.where(joint, center, high)
    sz = _tempered_gap(xp, zz, student_base, temperature)
    tn = _tempered_gap(xp, tt, teacher_base, temperature)
    e, et = xp.exp(sz), xp.exp(tn)
    q = e / xp.sum(e)
    if wide:
        # Sum before restoring the student's maximum to one. Its exact ties and the tail
        # remain separate so a concentrated distribution can retain a subnormal loss.
        shift = (student_base - top) / temperature
        s = sz + shift
        rest = xp.sum(xp.where(zz < top, xp.stop(e), 0.0)) * xp.exp(shift)
        tail = rest + (xp.astype(xp.count(zz == top), z.dtype) - 1)
        lse = _log1p(xp, tail)
        teacher_mass = xp.sum(et)
        t_tail = teacher_mass - 1
    else:
        s = sz
        lse = _log1p(xp, _tail(xp, xp.stop(s), xp.stop(e)))
        t_tail = _tail(xp, tn, et)
        teacher_mass = 1 + t_tail
    p = et / teacher_mass
    # (z - t) - (m_z - m_t) with the rounding errors of both differences: exact where the
    # logits agree, and z - t rounds where the student's offset is far from the teacher's.
    if xp.name == "torch":
        # Reusing the original wide masks slows down Inductor. Compute these equivalent
        # masks from the already sanitized logits for PyTorch's reduction.
        student_kept, teacher_kept = xp.isfinite(zz), xp.isfinite(tt)
    shared = student_kept & teacher_kept
    d, error = difference(xp.where(shared, zz, 0.0), xp.where(shared, tt, 0.0))
    offset, offset_error = difference(student_base, teacher_base)
    delta = xp.where(shared, (d - offset) + (error - offset_error), 0.0)
    cross = xp.sum(xp.where(et > 0, et * s, 0.0)) / teacher_mass
    log_p = None
    if proximal:
        lse_t = xp.log1p(t_tail)
        log_p = tn - lse_t
        lognorm, logratio = _proximal_logratio(
            xp, e, et, t_tail, delta, temperature, p, shared, lse_t, lse, s, tn
        )
    else:
        lognorm = lse
        # Raw, finite logits of opposite signs can overflow their difference.
        # Centering each distribution first gives the limiting log ratio there.
        delta = xp.where(xp.isnan(delta), xp.where(s == tn, 0.0, s - tn), delta / temperature)
        delta = xp.where(shared, delta, xp.where(student_kept, _INF, -_INF))
        # For each class exp(s) - exp(tn) is the larger exponential times the
        # signed relative difference, even when the two scores nearly agree.
        above = delta >= 0
        relative = xp.expm1(xp.where(above, -delta, delta))
        difference_sum = xp.sum(xp.where(above, -e * relative, et * relative))
        logratio = delta - xp.log1p(difference_sum / teacher_mass)
    return _Rows(
        z,
        valid,
        limit,
        s if proximal else None,
        e if proximal else None,
        lse,
        cross,
        q,
        log_p,
        p,
        lognorm,
        logratio,
    )


def distillation(xp: Namespace, logits: Array, target: Array, *, temperature: float = 1.0) -> Array:
    """Cross-entropy of the teacher and student, with gradient (q - p) / T.
    Compute each difference from the log-ratio, then distribute its rounding residual
    over the student probabilities. A confident maximum retains the mass of its tail."""
    r = _distillation_rows(xp, logits, target, temperature)
    value = r.lse - r.cross
    above = r.logratio >= 0
    relative = xp.expm1(xp.where(above, -r.logratio, r.logratio))
    u = xp.where(above, -r.q * relative, r.p * relative)
    residual = xp.sum(u)
    balance = r.q
    if r.z.dtype == xp.float64 and r.z.shape[1] >= WIDE_DISTILLATION_CLASSES:
        # Project a resolved, diffuse difference relative to each entry's magnitude. Scale
        # the residual once per row to protect near-zero entries without normalizing a
        # matrix of weights. The weights have no derivative in the exact zero residual.
        magnitude = xp.stop(xp.abs(u))
        total = xp.sum(magnitude)
        diffuse = (r.lse > 0.7) & (total > 1.0 / 16)
        balance = xp.stop(xp.where(diffuse, magnitude, r.q))
        residual = xp.where(diffuse, residual / xp.maximum(total, 1.0 / 16), residual)
    gradient = (u - balance * residual) / temperature
    gradient = xp.where(r.limit, (r.q - r.p) / temperature, gradient)
    value = xp.where(r.limit, _INF, value)
    return _attach_smooth(xp, value[:, 0], r.z, gradient, r.valid)


def proximal_distillation(
    xp: Namespace,
    logits: Array,
    target: Array,
    *,
    temperature: float = 1.0,
    gamma: float = 1.0,
) -> Array:
    """The Moreau envelope with step gamma of H(softmax(t / T), softmax(.)) at z / T, of
    teacher logits t (rows, classes); its gradient (pi - p) / T is exactly zero for equal
    logits."""
    r = _distillation_rows(xp, logits, target, temperature, proximal=True)
    assert r.s is not None and r.e is not None and r.log_p is not None
    s = xp.stop(r.s)
    log_q, log_p = s - r.lognorm, r.log_p
    q = xp.stop(r.e) * xp.exp(-r.lognorm)
    logratio = xp.stop(r.logratio)
    removed = (log_q == -_INF) & (log_p == -_INF)
    logratio = xp.where(r.p > 0, logratio, xp.where(removed, 0.0, _INF))
    # On limit rows, spread the missing teacher mass across the kept classes.
    kept = r.s > -_INF
    missing = xp.sum(xp.where(kept, 0.0, r.p))
    extra = xp.where(r.limit, missing / xp.astype(xp.count(kept), r.p.dtype), 0.0)
    p = xp.where(r.limit, xp.where(kept, r.p + extra, 0.0), r.p)
    log_p = xp.where(r.limit, xp.log(p), log_p)
    log_q = xp.where(r.limit, s - r.lse, log_q)
    q = xp.where(r.limit, r.q, q)
    logratio = xp.where(r.limit, xp.where(kept, log_q - log_p, 0.0), logratio)
    value, gradient = _proximal_distillation(xp, log_p, log_q, p, q, logratio, gamma, ~r.valid)
    gradient = xp.where(r.limit, gradient + xp.where(kept, extra, -r.p), gradient)
    value = xp.where(r.limit[:, 0], _INF, value)
    return _attach(xp, value, r.z, gradient / temperature, r.valid)
