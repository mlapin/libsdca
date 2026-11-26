"""The entropy problems of the composed losses: top-k entropy in closed form, and the Moreau
envelopes of cross-entropy (the block solve of proximal_cross_entropy) and of distillation,
batched over rows by safeguarded Halley steps.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .._limits import FLOAT32, FLOAT64, LAMBERT_RATIO, limits
from ._project import exact_sum, search
from ._xp import Array, Namespace

_INF = float("inf")
_NAN = float("nan")
# log(u / 4) of each work type's digits (computed here: compiled graphs cannot trace it).
_LOG_QUARTER_ROUNDOFF = {lim.digits: lim.log_pow2(-(lim.digits + 2)) for lim in (FLOAT32, FLOAT64)}


def softplus(xp: Namespace, x: Array) -> Array:
    """log1p(exp(x)) without a linear cutoff."""
    return xp.where(x > 0, x + xp.log1p(xp.exp(-x)), xp.log1p(xp.exp(x)))


def difference(a: Array, b: Array) -> tuple[Array, Array]:
    """a - b and its rounding error (TwoSum)."""
    d = a - b
    rounded = d - a
    return d, (a - (d - rounded)) + (-b - rounded)


def _offset_mean(xp: Namespace, z: Array, valid: Array, z0: Array, count: Array) -> Array:
    """Mean of z - z0 over the valid entries, accumulated in double-double. The scores are
    scaled by a power of two first: differences of scores near finfo.max and their sum
    overflow."""
    scale = 2.0 ** -(2 * z.shape[-1]).bit_length()
    total, low = difference(z * scale, z0 * scale)
    low = xp.where(valid & xp.isfinite(total), low, 0.0)
    total = xp.where(valid, total, 0.0)
    mean = exact_sum(xp, total, xp.sum(low)) / xp.astype(xp.maximum(count, 1), z.dtype)
    return mean / scale


def _topk_tail(xp: Namespace, offset: Array, odds: Array, scale: Array) -> tuple[Array, Array]:
    """Top-k entropy loss and mass of the odds, offset + log(scale) (offset <= 0) where they
    are below log eps: such a tail is its own loss to half an eps, from squared half
    exponentials rounded once; the log-domain odds round at the magnitude of the offset."""
    half = xp.exp(offset * 0.5)
    tail = half * (half * scale)
    far = odds < limits(odds.dtype).log_eps
    loss = xp.where(far, tail, softplus(xp, odds))
    return loss, xp.where(far, tail / (1 + tail), xp.exp(-softplus(xp, -odds)))


def all_capped(
    xp: Namespace, z: Array, valid: Array, z0: Array, count: Array
) -> tuple[Array, Array]:
    """Loss and total competitor mass when k = C - 1 and all competitors have equal share."""
    k = z.shape[-1] - 1
    # Take the mean about a score close to it. Its offset from the label and the
    # competitors' mean residual stay separate in exp(gap) * exp(residual);
    # prematurely rounding their sum near -700 loses hundreds of relative eps.
    scale = 2.0 ** -(2 * z.shape[-1]).bit_length()
    signed = xp.sum(xp.where(valid, z * scale, 0.0)) / xp.astype(xp.maximum(count, 1), z.dtype)
    pivot = signed / scale
    pivot = xp.where(xp.isfinite(pivot - z0), pivot, z0)
    gap, low = difference(pivot, z0)
    rest = _offset_mean(xp, z, valid, pivot, count) + low
    total = gap + rest
    split = (gap < 0) & (xp.abs(rest) < limits(z.dtype).log_max - math.log(k) - 1)
    offset = xp.where(split, gap, total)
    factor = k * xp.exp(xp.where(split, rest, 0.0))
    return _topk_tail(xp, xp.minimum(offset, 0.0), total + math.log(k), factor)


def topk_entropy(xp: Namespace, z: Array, valid: Array, z0: Array, k: int) -> tuple[Array, Array]:
    """Top-k entropy loss (rows,) and its gradient at the competitors valid, the capped
    probabilities, for target scores z0 (rows, 1) and k > 1.

    Rows of fewer than k competitors have the limit 0; rows of exactly k cap every one, and
    only their mean matters.
    """
    dtype, eps = z.dtype, xp.finfo(z.dtype).eps
    r = xp.count(valid)
    others = xp.where(valid, z, -_INF)
    # The k largest competitors in order, and the sum of the exponentials of the others
    # relative to the k-th: no full sort.
    ordered, index = xp.topk(others, k)
    last = ordered[:, k - 1 :]
    rest = xp.sum(xp.exp(xp.put(others, index, -_INF) - last))
    largest = ordered[:, :1]
    position = xp.arange(k, z)

    def tail(u: Array) -> tuple[Array, Array]:
        # ordered[u] and the sum of the exponentials of the competitors below it, relative
        # to it: free scores far below a capped one keep their differences.
        free = xp.take(ordered, u)
        terms = xp.sum(xp.where(position > u, xp.exp(ordered - free), 0.0))
        return free, terms + xp.exp(last - free) * rest

    def capped(j: Array) -> Array:
        # The active-set test relative to ordered[j]: it exceeds the threshold.
        threshold = xp.log1p(tail(xp.maximum(j, 0))[1]) - xp.log(xp.astype(k - j, dtype))
        return (j < 0) | (threshold + 32 * eps * xp.maximum(xp.abs(threshold), 1.0) < 0)

    start = r * 0 - 1
    u = search(xp, start, start + k - 1, k - 1, capped) + 1
    free, sums = tail(u)
    head = xp.sum(xp.where(position < u, ordered - largest, 0.0))
    uncapped, offset, total = xp.astype(k - u, dtype), largest - z0, sums + 1
    mean = (head + uncapped * ((free - largest) + xp.log1p(sums) - xp.log(uncapped))) / k
    # Rows of no capped competitor are cross-entropy's; a positive offset (a capped
    # competitor far above the free ones) joins the mean.
    above = xp.maximum(offset, 0.0)
    scale = xp.where(u == 0, total, xp.exp(xp.minimum(mean + above, 0.0)) * k)
    loss, mass = _topk_tail(xp, offset - above, offset + mean + math.log(k), scale)
    # Each competitor cap * exp(z - free) / spread, at most cap, spread the free ones' sum of
    # exp(z - free) per uncapped slot: a log-domain shift would round at the magnitude of
    # log(mass).
    spread = total / uncapped
    share = xp.minimum(xp.exp(xp.minimum(others - free, xp.log(spread))) / spread, 1.0)
    gradient = (mass / k) * share

    every = _offset_mean(xp, z, valid, z0, r)
    all_loss, all_mass = _topk_tail(
        xp, xp.minimum(every, 0.0), every + math.log(k), xp.full_like(every, float(k))
    )
    exact = r == k
    loss = xp.where(exact, all_loss, loss)
    gradient = xp.where(exact, all_mass / k, gradient)
    few = r < k
    loss = xp.where(few, 0.0, loss)
    gradient = xp.where(valid & ~few, gradient, 0.0)
    return loss[:, 0], gradient


def lambert_w_exp(xp: Namespace, x: Array) -> Array:
    """W(exp(x)): Halley steps on exp(u) + u = x for u = log W from log(softplus(x)).

    Two steps in float32 reach its resolution; float64 takes one more and a Newton step on
    W itself (exp(u) loses the low bits of a large u).
    """
    fp32 = x.dtype == xp.float32
    lim = limits(x.dtype)
    # Below log u, W(e^x) = e^x (1 - e^x + ...) rounds to e^x; above 1/eps, to x - log x.
    cut, large = lim.log_roundoff, lim.inv_eps

    def halley(u: Array, x: Array) -> Array:
        e = xp.exp(u)
        g = (e - x) + u
        return u - g / ((e + 1) - g * e / (2 * (e + 1)))

    x32 = xp.astype(xp.clip(x, cut, large), xp.float32)
    u = xp.log(xp.where(x32 > 30, x32, xp.log1p(xp.exp(x32))))
    for _ in range(2):
        u = halley(u, x32)
    u = xp.astype(u, x.dtype)
    if not fp32:
        u = halley(u, x)
    w = xp.exp(u)
    if not fp32:
        w = w - ((w - x) + xp.log(w)) * (w / (w + 1))
    # Far below, W(exp(x)) = exp(x) to the last bit; far above, x - log x.
    w = xp.where(x > large, x - xp.log(xp.maximum(x, 1.0)), w)
    return xp.where(x < cut, xp.exp(x), xp.where(x == _INF, x, w))


def _lambert_ratio(xp: Namespace, y: Array, terms: int) -> Array:
    """W(y) / y for 0 <= y <= 2**-5 by its series (core::lambert_ratio_small)."""
    g = y * LAMBERT_RATIO[terms - 1] + LAMBERT_RATIO[terms - 2]
    for n in range(terms - 3, -1, -1):
        g = g * y + LAMBERT_RATIO[n]
    return g


def probability(xp: Namespace, h: Array, alpha: float, log_alpha: float) -> Array:
    """W(alpha exp(h)) / alpha (core::lambert_probability): exp(h) W(y) / y at y = alpha
    exp(h) on the series' domain (exp(h) <= max / 4), where W / alpha errs by W's error and
    exp(h - W) by u |h - W|; else W / alpha, and exp(h - W) on the series' domain beyond
    exp(h) <= max / 4."""
    lim = limits(h.dtype)
    arg = h + log_alpha
    small = arg <= lim.lambert_ratio_arg
    series = small & (h <= lim.log_quarter_max)
    e = xp.exp(xp.where(series, h, 0.0))
    ratio = _lambert_ratio(xp, alpha * e, lim.lambert_ratio_terms)
    # Below y = u / 4 the series rounds to one.
    near = xp.where(arg < _LOG_QUARTER_ROUNDOFF[lim.digits], e, e * ratio)
    w = lambert_w_exp(xp, arg)
    return xp.where(series, near, xp.where(small, xp.exp(h - w), w / alpha))


def residual(
    xp: Namespace, t: Array, alpha: float, q: Array, log_alpha: float, log_q: Array
) -> Array:
    """r in [q - 1, q) with alpha r - log1p(-r / q) = t."""
    # A flushed subnormal tolerance 8 eps |s| scales f by a power of two instead, and f
    # within the change over the smallest normal r is converged (r rounds to zero).
    lim = limits(t.dtype)
    inv_tol = lim.inv_eps / 8
    lo = q - 1
    half = alpha * q / 2
    far = (t > half + math.log(2)) | (t < -half - math.log(1.5))
    prob = probability(xp, log_q + (alpha * q - t), alpha, log_alpha)
    outer = xp.minimum(xp.maximum(q - prob, lo), q)
    # Near zero: a Pade start and Halley steps on log1p(-r / q), bracketed.
    s = xp.where(far, 0.0, t)
    inv_q = 1 / q
    linear = s / (alpha + inv_q)
    r = linear / (1 + (linear * inv_q) / (2 * (1 + alpha * q)))
    below = xp.nextafter(q, q * 0)
    r = xp.minimum(xp.maximum(r, lo), below)
    low, high = lo + r * 0, below + r * 0
    floor = (alpha + inv_q) * lim.tiny
    done = xp.isnan(r)
    # Halley steps from the start's 4 bits; a step below the resolution of r ends it (also
    # one that underflows, where the Halley denominator overflows).
    for _ in range(lim.steps_of_order(3, 4)):
        f = (alpha * r - xp.log1p(-r * inv_q)) - s
        inv_qr = 1 / (q - r)
        d1 = alpha + inv_qr
        denom = d1 - (f * inv_qr * inv_qr) / (2 * d1)
        step = xp.where(denom > 0, -f / denom, -f / d1)
        met = (xp.abs(f) * inv_tol <= xp.abs(s)) | (xp.abs(f) <= floor)
        r, low, high, done = _advance(xp, r, f, step, low, high, done | met)
    r = xp.where(far, outer, r)
    return xp.where(t == 0, 0.0, xp.where(t <= alpha * lo + log_q, lo, r))


def _usable(xp: Namespace, f: Array, derivative: Array) -> Array:
    """Where a Newton-type step -f / f' is defined: f' positive and finite."""
    return (derivative > 0) & xp.isfinite(derivative) & xp.isfinite(-f / derivative)


def _halley(f: Array, derivative: Array, curvature: Array, xp: Namespace) -> Array:
    """The safeguarded Halley step of a root solve; 0 where it is not _usable."""
    newton = -f / derivative
    correction = (f * curvature) / (2 * derivative * derivative)
    step = xp.where(xp.isfinite(correction), newton / (1 - xp.clip(correction, -0.5, 0.5)), newton)
    return xp.where(_usable(xp, f, derivative), step, 0.0)


def _key(xp: Namespace, x: Array) -> Array:
    """The float key of x as a signed integer, increasing with x (core::to_key, offset by
    half the key range): -0.0 and +0.0 are adjacent."""
    bits = xp.bits(x)
    return xp.where(bits < 0, ~(bits & _max_int(bits)), bits)


def _from_key(xp: Namespace, key: Array, like: Array) -> Array:
    return xp.from_bits(xp.where(key < 0, ~key | ~_max_int(key), key), like)


def _max_int(bits: Array) -> int:
    return 2 ** (8 * bits.dtype.itemsize - 1) - 1


def bisect(xp: Namespace, lo: Array, hi: Array) -> Array:
    """The midpoint of the float keys of lo <= hi (core::bisect), which closes any bracket of
    finite values to adjacent ones in limits.key_bisection_steps halvings; lo where lo == hi."""
    a, b = _key(xp, lo), _key(xp, hi)
    middle = (a >> 1) + (b >> 1) + (a & b & 1)
    return xp.where(lo < hi, _from_key(xp, middle, lo), lo)


def _advance(
    xp: Namespace,
    x: Array,
    f: Array,
    step: Array,
    lo: Array,
    hi: Array,
    done: Array,
    free: bool | Array = True,
) -> tuple[Array, Array, Array, Array]:
    """One step of a safeguarded search for the root of an increasing residual, f at x: the
    end of the bracket [lo, hi] on f's side moves to x, and the next iterate is x + step
    where free and inside the bracket, else its bisection. Rows are done where the bracket is
    adjacent or a free nonzero step is below the resolution of x (x is the closest root);
    done rows keep x and the bracket. Returns x, lo, hi and done."""
    lo = xp.where(~done & (f < 0), x, lo)
    hi = xp.where(~done & (f >= 0), x, hi)
    middle = bisect(xp, lo, hi)
    nxt = x + step
    done = done | ((step != 0) & (nxt == x) & free) | (middle == lo)
    inside = (lo <= nxt) & (nxt <= hi) & (nxt != x) & xp.isfinite(nxt) & free
    nxt = xp.where(inside, nxt, middle)
    return xp.where(done, x, nxt), lo, hi, done


def _root(
    xp: Namespace,
    evaluate: Callable[[Array], tuple[Array, ...]],
    x: Array,
    lo: Array,
    hi: Array,
    done: Array,
) -> tuple[Array, ...]:
    """The root of an increasing residual in [lo, hi] from x, for the rows not done, by
    core::root_steps: free steps (evaluate(x) gives the residual f, whether it meets its
    tolerance, a step, 0 where there is none, and any further values), then bisections, which
    close a finite bracket to adjacent values: every row whose residuals are not NaN is done
    within root_steps. Returns the root, done and the further values at the root."""
    lim = limits(x.dtype)

    def body(state: tuple[Array, ...]) -> tuple[Array, ...]:
        i, x, lo, hi, done = state[:5]
        f, met, step, *rest = evaluate(x)
        free = i < lim.root_free_steps
        return (i + 1, *_advance(xp, x, f, step, lo, hi, done | met, free), *rest)

    def more(state: tuple[Array, ...]) -> Array:
        return ~state[4].all()

    # The first evaluation gives the carry of the further values; done rows keep x, so the
    # last evaluation of a done row is at its root.
    count = xp.astype(xp.arange(1, x)[0] * 0, xp.int_)
    state = xp.while_loop(more, body, body((count, x, lo, hi, done)), lim.root_steps - 1)
    return state[1], state[4], *state[5:]


class _Block:
    """The residual sum f(dt) of a batch of entropy blocks and its derivatives: the labels of
    a row form its block; others are its valid other classes."""

    def __init__(
        self,
        xp: Namespace,
        v: Array,
        positive: Array,
        others: Array,
        labels: Array,
        alpha: float,
    ) -> None:
        self.xp, self.v, self.positive, self.alpha = xp, v, positive, alpha
        self.log_alpha = math.log(alpha)
        self.count = xp.count(positive)
        self.others = xp.astype(xp.count(others), v.dtype)
        self.q = 1 / xp.astype(self.count, v.dtype)
        self.log_q = xp.log(self.q)
        self.clip = alpha * (self.q - 1) + self.log_q
        self.label_max = xp.max(xp.where(positive, v, -_INF))
        self.other_max = xp.max(xp.where(positive, -_INF, v))
        self.other_min = xp.min(xp.where(others, v, _INF))
        self.ref = xp.maximum(self.label_max, self.other_max)
        self.labels = labels
        self.used = xp.take(positive, labels)
        self.label_v = xp.take(v, labels)

    def evaluate(self, dt: Array) -> tuple[Array, ...]:
        """f, f', f'', magnitude, other mass, label residuals and other probabilities."""
        xp, alpha = self.xp, self.alpha
        argument = (self.ref - self.label_v) + dt
        r_label = residual(xp, argument, alpha, self.q, self.log_alpha, self.log_q)
        r_label = xp.where(self.used, r_label, 0.0)
        prob = probability(xp, ((self.v - self.ref) - dt) + self.log_q, alpha, self.log_alpha)
        counted = prob <= 1
        prob = xp.where(self.positive, 0.0, xp.minimum(prob, 1.0))
        mass = xp.sum(prob)
        slope = self.q - r_label
        label_denom, denom = 1 + alpha * slope, 1 + alpha * prob
        label_d1 = xp.where(self.used & (argument >= self.clip), slope / label_denom, 0.0)
        d1 = xp.where(counted, prob / denom, 0.0)
        derivative = xp.sum(label_d1) + xp.sum(d1)
        curvature = -xp.sum(label_d1 / (label_denom * label_denom)) - xp.sum(d1 / (denom * denom))
        magnitude = xp.sum(xp.abs(r_label)) + mass
        f = xp.sum(r_label) - mass
        return f, derivative, curvature, magnitude, mass, r_label, prob

    def label_log_probability(self, r_label: Array, dt: Array) -> Array:
        """log p of the labels: log1p where r is small, else log(W / alpha) = h - W."""
        xp = self.xp
        h = self.log_q + (self.alpha * self.q - ((self.ref - self.label_v) + dt))
        far = h - lambert_w_exp(xp, h + self.log_alpha)
        near = self.log_q + xp.log1p(-r_label / self.q)
        return xp.where(self.used, xp.where(r_label <= self.q / 2, near, far), 0.0)

    def residuals(self, r_label: Array, prob: Array) -> Array:
        """The residuals of all classes: r at the labels, -p at the others."""
        xp = self.xp
        r = -prob
        return xp.put(r, self.labels, xp.where(self.used, r_label, xp.take(r, self.labels)))

    def step(self, f: Array, derivative: Array, curvature: Array, mass: Array) -> Array:
        """Halley step, or a step in the log of the other mass where it exceeds one."""
        xp = self.xp
        result = _halley(f, derivative, curvature, xp)
        log_step = xp.log(mass) + self.alpha * (mass - 1) / self.others
        jump = (f < 0) & (mass > 1) & xp.isfinite(log_step) & (log_step > result)
        return xp.where(_usable(xp, f, derivative) & jump, log_step, result)

    def bracket(self) -> tuple[Array, Array]:
        """dt where the largest probability is one (f <= 0) and every one <= 1/n (f >= 0)."""
        xp = self.xp
        a, label, other = self.alpha, self.label_max - self.ref, self.other_max - self.ref
        lo = xp.maximum(label - a * (1 - self.q), other - a) + self.log_q
        hi = xp.maximum(label + a * self.q, other) + self.log_q + math.log(self.v.shape[-1])
        return lo, xp.where(lo <= hi, hi, lo)

    def start(self) -> Array:
        """The softmax-limit threshold; log q for a confident single label."""
        xp = self.xp
        eps = xp.finfo(self.v.dtype).eps
        shifted = self.v - self.ref
        first = xp.arange(self.v.shape[-1], self.v) == xp.argmax(shifted)
        tail = xp.sum(xp.where(first, 0.0, xp.exp(shifted)))
        tiny = (
            (self.count == 1)
            & (self.label_max == self.ref)
            & ((self.other_max - self.ref) + math.log(self.v.shape[-1]) < math.log(eps) / 2)
        )
        return xp.where(tiny, self.log_q, self.log_q + xp.log1p(tail))


def _solve(block: _Block, skip: Array) -> tuple[Array, Array, Array, Array]:
    """The root dt of every block, its label residuals, other probabilities and failures (the
    rows not done within root_steps, or of a residual that is not finite)."""
    # The tolerance scales the residual, not the magnitude: a tiny magnitude times the
    # tolerance may flush to zero.
    xp = block.xp
    inv_tol = 1 / limits(block.v.dtype).prox_tolerance
    lo, hi = block.bracket()

    def evaluate(dt: Array) -> tuple[Array, ...]:
        f, derivative, curvature, magnitude, mass, r_label, prob = block.evaluate(dt)
        met = xp.abs(f) * inv_tol <= magnitude
        return f, met, block.step(f, derivative, curvature, mass), f, r_label, prob

    start = xp.minimum(xp.maximum(block.start(), lo), hi)
    dt, done, f, r_label, prob = _root(xp, evaluate, start, lo, hi, skip)
    return dt, r_label, prob, ~done | ~xp.isfinite(f)


def _exp_sum(xp: Namespace, logs: Array) -> Array:
    """The row sums (rows, 1) of exp(logs), relative to the largest and scaled back once by
    half exponentials: subnormal terms keep their bits."""
    top = xp.max(logs)
    top = xp.where(xp.isfinite(top), top, 0.0)
    half = xp.exp(top / 2)
    return half * (half * xp.sum(xp.exp(logs - top)))


def _zero_sum(xp: Namespace, a: Array, pivot: Array) -> Array:
    """a with its entry at pivot (rows, 1) balancing the others: a zero sum to one rounding."""
    at_pivot = xp.arange(a.shape[-1], a) == pivot
    rest = xp.where(at_pivot, 0.0, a)
    return xp.where(at_pivot, -xp.sum(rest), rest)


def _tiny_tail(block: _Block, value: Array, a: Array, single: Array) -> tuple[Array, Array]:
    """The loss and gradient of single-label rows whose other classes lie so far below that
    the proximal perturbation is below precision, and one below the normal range: their tail
    aggregated before rounding (the solve's probabilities of those may flush to zero)."""
    xp = block.xp
    info = xp.finfo(block.v.dtype)
    gap = block.other_max - block.label_max
    tail = gap + xp.log(xp.maximum(block.others, 1.0))
    error = max(0.0, block.log_alpha) + tail
    bound = math.log(float(info.eps) / 8)
    tiny = (
        single
        & xp.isfinite(gap)
        & (block.other_min - block.label_max < math.log(float(info.tiny)))
        & ((error < bound) | (error + tail < math.log(float(info.tiny)) + bound))
    )
    origin = xp.where(tiny, block.other_max, 0.0)
    terms = xp.exp(xp.where(block.positive | ~tiny, -_INF, block.v - origin))
    half = xp.exp(xp.where(tiny, gap, 0.0) / 2)
    mass = half * (half * xp.sum(terms))
    recovered = xp.where(block.positive, mass, -(half * terms) * half)
    return xp.where(tiny[:, 0], mass[:, 0], value), xp.where(tiny, recovered, a)


def proximal_entropy(
    xp: Namespace, z: Array, positive: Array, labels: Array, alpha: float, skip: Array
) -> tuple[Array, Array, Array]:
    """min_u H(y, softmax(u)) + |u - z|^2 / (2 alpha) per row for the uniform target y on the
    labels: the loss (rows,), its gradient (rows, classes) and the rows whose solve failed;
    the rows skip are not solved.

    labels (rows, width) index every label of a row (padded with other classes); masked
    classes (-inf) are removed, and a row whose other classes are all masked has loss 0.
    """
    v = z - xp.max(z)
    others = ~positive & (v > -_INF)
    alone = xp.count(others) == 0
    v = xp.where(alone | skip, 0.0, v)
    block = _Block(xp, v, positive, others, labels, alpha)
    dt, r_label, prob, failed = _solve(block, alone | skip)
    a = block.residuals(r_label, prob)
    # Zero-sum to one rounding: the label (or the largest entry) balances the others. Labels
    # that are all zero share the recovered mass of the others.
    mass = xp.sum(xp.where(positive, 0.0, a))
    labels_zero = xp.all(xp.where(positive, a, 0.0) == 0)
    count = block.count
    share = (count > 1) & labels_zero & (mass != 0)
    a = xp.where(share & positive, -mass / xp.astype(count, a.dtype), a)
    pivot = xp.where((count == 1) | share, xp.argmax(positive), xp.argmax(xp.abs(a)))
    a = _zero_sum(xp, a, pivot)
    # H from the label probabilities (u = v + alpha a cancels), and alpha a_j first in the
    # penalty (a_j^2 can underflow). The one-label loss takes the corrected mass.
    r_value = xp.where(count == 1, -mass, r_label)
    logs = block.label_log_probability(r_value, dt)
    cross = -xp.sum(logs)[:, 0] * block.q[:, 0]
    # The sum can overflow while its mean is representable.
    cross = xp.where(xp.isfinite(cross), cross, -xp.sum(logs * block.q)[:, 0])
    value = cross + xp.sum((alpha * a) * a)[:, 0] / 2
    value, a = _tiny_tail(block, value, a, count == 1)
    value = xp.where(alone[:, 0], 0.0, value)
    return value, xp.where(alone, 0.0, -a), failed & ~alone


def proximal_distillation(
    xp: Namespace,
    log_p: Array,
    log_q: Array,
    p: Array,
    q: Array,
    logratio: Array,
    alpha: float,
    skip: Array,
) -> tuple[Array, Array]:
    """min_u H(p, softmax(u)) + |u - s|^2 / (2 alpha) per row and its gradient pi - p in s,
    from the teacher's p = softmax(t) and the student's q = softmax(s) and the log-ratio
    log(q / p): a zero log-ratio gives a zero gradient. omega is the root of sum(r) = 0 for
    the residuals r = p - pi. Rows not solved within root_steps are NaN.
    """
    log_alpha = math.log(alpha)
    lim = limits(log_q.dtype)
    inv_tol, tiny = 1 / lim.prox_tolerance, lim.tiny
    # q - p from the ratio where q is near p: their difference would cancel.
    near_ratio = xp.abs(logratio) < 1
    u0 = xp.where(near_ratio, p * xp.expm1(xp.where(near_ratio, logratio, 0.0)), q - p)
    lo = -alpha * xp.maximum(xp.max(u0), 0.0)
    hi = -alpha * xp.minimum(xp.min(u0), 0.0)
    hi = xp.where(lo <= hi, hi, lo)

    def evaluate(w: Array) -> tuple[Array, Array]:
        arg = w - logratio
        prob = xp.minimum(probability(xp, (log_q + alpha * p) - w, alpha, log_alpha), 1.0)
        half = alpha * p / 2
        near = (p > tiny) & (arg <= half + math.log(2)) & (arg >= -half - math.log(1.5))
        p_safe, log_p_safe = xp.where(near, p, 1.0), xp.where(near, log_p, 0.0)
        r_near = residual(xp, xp.where(near, arg, 0.0), alpha, p_safe, log_alpha, log_p_safe)
        r = xp.where(near, r_near, xp.where(arg == 0, 0.0, p - prob))
        return r, xp.where(near, p - r_near, prob)

    def residuals(w: Array) -> tuple[Array, ...]:
        r, pi = evaluate(w)
        f = xp.sum(r)
        denom = 1 + alpha * pi
        d1, d2 = xp.sum(pi / denom), -xp.sum(pi / (denom * denom * denom))
        return f, xp.abs(f) * inv_tol <= xp.sum(xp.abs(r)), _halley(f, d1, d2, xp), r, pi

    start = xp.clip(lo * 0, lo, hi)
    omega, done, r, pi = _root(xp, residuals, start, lo, hi, skip | ((lo == 0) & (hi == 0)))
    pivot = xp.argmax(p + q)
    r = _zero_sum(xp, r, pivot)
    h = (log_q + alpha * p) - omega
    small_subnormal_penalty = alpha <= lim.roundoff / tiny
    if small_subnormal_penalty:
        # Below tiny, alpha*pi is less than one unit roundoff in absolute value.
        # The root already produced every representable normal probability.
        normal_pi = pi >= tiny
        far_log_pi = xp.where(normal_pi, xp.log(xp.where(normal_pi, pi, 1.0)), h)
        if alpha > 1:
            # Use log(pi) = h - alpha*pi where the subtraction is well conditioned.
            far_log_pi = xp.where(alpha * pi <= 1, h - alpha * pi, far_log_pi)
    else:
        far_log_pi = h - lambert_w_exp(xp, h + log_alpha)
    near_log_pi = log_p + xp.log1p(-r / xp.where(p > 0, p, 1.0))
    # Far above a tiny p, -r / p overflows: log pi = log(p - r) directly, but for a subnormal
    # p - r (a few bits), use the solved logarithm.
    above = r < -p / 2
    direct = xp.where(p - r >= tiny, xp.log(xp.where(above, p - r, 1.0)), far_log_pi)
    near_log_pi = xp.where(above, direct, near_log_pi)
    log_pi = xp.where(r <= p / 2, near_log_pi, far_log_pi)
    # Where the pivot holds most of pi, log1p of the others' mass: its log_p rounds at the
    # scale of one (log p = -1e-19 is 0), which p log pi would carry in full.
    at_pivot = xp.arange(p.shape[-1], p) == pivot
    others = _exp_sum(xp, xp.where(at_pivot, -_INF, log_pi))
    log_pi = xp.where(at_pivot & (others <= 0.5), xp.log1p(-others), log_pi)
    # The terms of subnormal p round to a few bits (or vanish): those from log p in the log
    # domain (-log pi >= 0).
    normal = p >= tiny
    tail = (log_p > -_INF) & ~normal & (log_pi < 0)
    logs = xp.where(tail, log_p + xp.log(xp.where(tail, -log_pi, 1.0)), -_INF)
    cross = _exp_sum(xp, logs) - xp.sum(xp.where(normal, p * log_pi, 0.0))
    value = cross[:, 0] + xp.sum((alpha * r) * r)[:, 0] / 2
    if small_subnormal_penalty:
        # Preserve the residual when both probabilities are subnormal. Scale the
        # difference before summing the tail and round the balancing pivot once.
        unit = math.sqrt(tiny)
        small = (log_p < lim.log_min) & (far_log_pi < lim.log_min)
        distance = xp.where(log_p == far_log_pi, 0.0, -xp.abs(log_p - far_log_pi))
        relative = xp.expm1(distance)
        half = xp.exp(xp.maximum(log_p, far_log_pi) / 2)
        scaled = (half / unit) * (half * xp.where(log_p < far_log_pi, relative, -relative))
        scaled_residual = xp.where(small, scaled, r / unit)
        balanced = -xp.sum(xp.where(at_pivot, 0.0, scaled_residual)) * unit
        r = xp.where(at_pivot, balanced, xp.where(small, scaled * unit, r))
    return xp.where(done[:, 0], value, _NAN), xp.where(done, -r, _NAN)
