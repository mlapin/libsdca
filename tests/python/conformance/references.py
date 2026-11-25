"""mpmath references of the libsdca losses on rows given by runs of equal entries.

A Row is a tuple of Blocks: runs of entries that share a logit, a teacher logit and a
positive flag. Every loss is symmetric in such entries, so a reference costs a few operations
per block however wide the row (two for a 32,768-class far-tail row) and has one gradient
entry per block.

Values and gradient entries keep DIGITS significant digits. The formulas do not cancel:

- soft-target entropy (cross_entropy, multilabel_cross_entropy, distillation) is
  H(t) + sum_{t > 0} t phi(w) + sum_{t = 0} q with phi(w) = expm1(w) - w >= 0 and
  w = log q - log t, and its gradient q - t is t expm1(w);
- the Moreau envelopes (proximal_cross_entropy, proximal_distillation) follow from their
  optimality conditions pi_j = t_j exp(w_j), w_j + alpha t_j expm1(w_j) = c_j (and
  log pi_j + alpha pi_j = d_j where t_j = 0), solved by Newton steps for each w and for the
  normalizer, with the value of the entropy above plus alpha |pi - t|^2 / 2;
- topk_cross_entropy is the capped-set closed form of release_numerics.topk_reference, taken
  over blocks: tied competitors are capped or free together.

The working digits add those of the row's largest magnitude and of its spread (up to 1000),
so differences of represented inputs and tails down to exp(-1000) keep DIGITS digits.

The hinges take their definitions (conformance.hinges: mpmath, and the multilabel two-simplex
in rationals) on the row with each block cut to k + 1 entries. A cut block of a smooth hinge
must come out with a zero gradient: then the cut row's KKT point is the full row's.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

import mpmath as mp
import numpy as np
from mpmath_types import MPReal

from conformance import hinges

DIGITS = 50
HINGES = ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge")
PROXIMAL = ("proximal_cross_entropy", "proximal_distillation")
DISTILLATION = ("distillation", "proximal_distillation")


@dataclass(frozen=True)
class Block:
    """count entries of logit value (and teacher logit teacher; a positive label if positive;
    the target probability mass of cross-entropy of probability targets, if not None)."""

    value: float
    count: int = 1
    positive: bool = False
    teacher: float = 0.0
    mass: float | None = None


Row = tuple[Block, ...]


@dataclass(frozen=True)
class Loss:
    """A loss and its options; label names the configuration in test ids."""

    name: str
    k: int = 1
    gamma: float = 0.0
    alpha: float = 1.0
    temperature: float = 1.0
    recompute: bool = False

    @property
    def label(self) -> str:
        options = [f"k={self.k}"] if self.name.startswith("topk") else []
        options += [f"gamma={self.gamma:g}"] if self.name in HINGES else []
        options += [f"T={self.temperature:g}"] if self.name in DISTILLATION else []
        options += [f"alpha={self.alpha:g}"] if self.name in PROXIMAL else []
        options += ["recompute"] if self.recompute else []
        return self.name + (f"[{','.join(options)}]" if options else "")


def named(
    name: str, k: int = 1, gamma: float = 0.0, temperature: float = 1.0, alpha: float = 1.0
) -> Loss:
    """The Loss of a shared name with the parameters it takes (the others at their defaults)."""
    return Loss(
        name,
        k=k if name.startswith("topk") else 1,
        gamma=gamma if name in HINGES else 0.0,
        alpha=alpha if name in PROXIMAL else 1.0,
        temperature=temperature if name in DISTILLATION else 1.0,
    )


@dataclass(frozen=True)
class Reference:
    """A row's loss, one gradient entry per block and, for distillation, (p + q) / T per
    block; kink is a nonsmooth hinge's distance from a kink in units of the scores."""

    value: MPReal
    gradient: tuple[MPReal, ...]
    scale: tuple[MPReal, ...] = ()
    kink: float = math.inf


def width(row: Row) -> int:
    return sum(b.count for b in row)


def entries(row: Row) -> list[float]:
    return [b.value for b in row for _ in range(b.count)]


def _digits(loss: Loss, row: Row) -> int:
    """DIGITS, those of the largest input, and those of each input's largest gap to its
    maximum up to 760 (smaller terms are below every dtype's subnormals)."""
    inputs = [[b.value for b in row if math.isfinite(b.value)]]
    if loss.name in DISTILLATION:
        inputs.append([b.teacher for b in row if math.isfinite(b.teacher)])
    inputs = [values for values in inputs if values]
    big = max((abs(v) for values in inputs for v in values), default=0.0)
    with np.errstate(over="ignore"):
        gaps = [(max(v) - x) / loss.temperature for v in inputs for x in v]
    spread = max((g for g in gaps if g <= 760), default=0.0)
    return DIGITS + 10 + int(math.log10(1 + big)) + int(spread / math.log(10))


def fraction(x: float | MPReal) -> Fraction:
    """A represented input (float or mpf) as an exact rational."""
    if not isinstance(x, float):
        man, exp = mp.mpf(x).man_exp
        value = Fraction(man) * Fraction(2) ** exp
        return -value if x < 0 else value
    return Fraction(x)


def _exact(*terms: float) -> MPReal:
    """The sum of represented inputs, rounded once to the working digits."""
    total = sum(map(fraction, terms), Fraction(0))
    return mp.mpf(total.numerator) / total.denominator


def _phi(w: MPReal) -> MPReal:
    """expm1(w) - w without its cancellation near 0."""
    if abs(w) >= 0.5:
        return mp.expm1(w) - w
    term, total, n = w * w / 2, mp.mpf(0), 2
    while term and abs(term) > mp.eps * abs(total) / 16:
        total += term
        n += 1
        term = term * w / n
    return total


def _log_targets(loss: Loss, row: Row) -> list[MPReal | None]:
    """log t of each block (None where t = 0): the label, the positives or softmax(teacher/T)."""
    if loss.name in DISTILLATION:
        # A teacher's -inf masks its class: t = 0 there.
        top = max(b.teacher for b in row)
        shifted = [
            None if b.teacher == -math.inf else _exact(b.teacher, -top) / loss.temperature
            for b in row
        ]
        ties = sum(b.count for b in row if b.teacher == top)
        rest = mp.fsum(
            b.count * mp.exp(s)
            for b, s in zip(row, shifted, strict=True)
            if s is not None and s < 0
        )
        normalizer = mp.log1p(ties - 1 + rest)
        return [None if s is None else s - normalizer for s in shifted]
    positives = sum(b.count for b in row if b.positive)
    if positives == 0:
        raise ValueError("a row without a positive label")
    return [-mp.log(positives) if b.positive else None for b in row]


def _offsets(loss: Loss, row: Row, anchor: int) -> list[MPReal]:
    """(x_j - x_a) - (log t_j - log t_a) with x = z / T: exact differences of inputs, so that
    a student near its teacher keeps the digits of their difference."""
    a = row[anchor]
    if loss.name in DISTILLATION:
        return [
            (
                _exact(b.value, -a.value)
                if b.teacher == -math.inf
                else _exact(b.value, -a.value, -b.teacher, a.teacher)
            )
            / loss.temperature
            for b in row
        ]
    return [_exact(b.value, -a.value) / loss.temperature for b in row]


def _softmax(loss: Loss, row: Row) -> tuple[list[MPReal], MPReal, int, MPReal]:
    """x_j - max(x) with x = z / T, lse(x) - max(x), the anchor a (the largest target) and the
    lift lse(x) - x_a + log t_a: log q_j - log t_j = c_j - lift with c_j of _offsets."""
    logt = _log_targets(loss, row)
    top = max(b.value for b in row)
    shifted = [_exact(b.value, -top) / loss.temperature for b in row]
    ties = sum(b.count for b in row if b.value == top)
    rest = mp.fsum(b.count * mp.exp(s) for b, s in zip(row, shifted, strict=True) if s < 0)
    normalizer = mp.log1p(ties - 1 + rest)
    anchor, t_anchor = -1, None
    for j, t in enumerate(logt):
        if t is not None and (t_anchor is None or t > t_anchor):
            anchor, t_anchor = j, t
    assert t_anchor is not None
    return shifted, normalizer, anchor, normalizer - shifted[anchor] + t_anchor


def _entropy(loss: Loss, row: Row) -> Reference:
    logt = _log_targets(loss, row)
    shifted, normalizer, anchor, lift = _softmax(loss, row)
    value, gradient, scale = mp.mpf(0), [], []
    for b, t, s, c in zip(row, logt, shifted, _offsets(loss, row, anchor), strict=True):
        if t is None:
            q = mp.exp(s - normalizer)
            value += b.count * q
            gradient.append(q / loss.temperature)
            scale.append(q / loss.temperature)
            continue
        w, p = c - lift, mp.exp(t)
        value += b.count * p * (_phi(w) - t)
        gradient.append(p * mp.expm1(w) / loss.temperature)
        scale.append(p * (1 + mp.exp(w)) / loss.temperature)
    return Reference(value, tuple(gradient), tuple(scale))


def _tolerance() -> MPReal:
    """Relative accuracy of the solves: 15 digits short of the working digits."""
    return mp.mpf(10) ** (15 - mp.mp.dps)


def _newton(f: Callable[[MPReal], tuple[MPReal, MPReal]], start: MPReal) -> MPReal:
    """Root of an increasing convex f (value, slope) from a start right of it. The steps
    decrease monotonically; a step to the right is rounding at the root."""
    w = start
    for _ in range(2000):
        value, slope = f(w)
        step = value / slope
        if step <= 0:
            return w
        w -= step
        if step <= _tolerance() * abs(w):
            return w
    raise ArithmeticError("Newton steps did not converge")


def _log_ratio(c: MPReal, kappa: MPReal, positive: bool) -> MPReal:
    """w with w + kappa expm1(w) = c (positive: t > 0), else w + kappa exp(w) = c."""
    if positive:
        start = mp.mpf(0) if c < 0 else min(c, mp.log1p(c / kappa))
        return _newton(lambda w: (w + kappa * mp.expm1(w) - c, 1 + kappa * mp.exp(w)), start)
    start = c if c <= 0 else min(c, max(mp.mpf(0), mp.log(c / kappa)))
    return _newton(lambda w: (w + kappa * mp.exp(w) - c, 1 + kappa * mp.exp(w)), start)


def _normalizer(
    mass: Callable[[MPReal], tuple[MPReal, MPReal]], start: MPReal, radius: MPReal
) -> MPReal:
    """Root of a decreasing mass (value, slope) within radius of start: Newton steps that
    bisect where they leave the bracket or halve it too slowly (rtsafe)."""
    low, high = start - radius, start + radius
    if mass(low)[0] < 0 or mass(high)[0] > 0:
        raise ArithmeticError("the normalizer is outside its bracket")
    x, previous = start, 2 * radius
    value, slope = mass(x)
    low, high = (x, high) if value > 0 else (low, x)
    for _ in range(4000):
        if value == 0:
            return x
        guess = x - value / slope
        if abs(guess - x) <= _tolerance() * abs(x):
            return guess
        if not low < guess < high or abs(guess - x) > previous / 2:
            guess = (low + high) / 2
        previous, x = abs(guess - x), guess
        if previous <= _tolerance() * abs(x) or x in (low, high):
            return x
        value, slope = mass(x)
        low, high = (x, high) if value > 0 else (low, x)
    raise ArithmeticError("the normalizer did not converge")


def _envelope(loss: Loss, row: Row) -> Reference:
    """min_u H(t, softmax(u)) + |u - x|^2 / (2 alpha), x = z / T, and (pi - t) / T."""
    logt = _log_targets(loss, row)
    alpha = mp.mpf(loss.alpha)
    _, _, anchor, lift = _softmax(loss, row)
    offsets = _offsets(loss, row, anchor)
    t_anchor = logt[anchor]
    assert t_anchor is not None
    # L = x_a - log t_a + shift: c_j = offset_j - shift, d_j = offset_j + log t_a - shift.
    targets = [None if t is None else mp.exp(t) for t in logt]

    def solve(shift: MPReal) -> list[MPReal]:
        logs = []
        for t, o in zip(targets, offsets, strict=True):
            if t is None:
                logs.append(_log_ratio(o + t_anchor - shift, alpha, False))
            else:
                logs.append(_log_ratio(o - shift, alpha * t, True))
        return logs

    def mass(shift: MPReal) -> tuple[MPReal, MPReal]:
        value, slope = mp.mpf(0), mp.mpf(0)
        for b, t, w in zip(row, targets, solve(shift), strict=True):
            e = mp.exp(w)
            if t is None:
                value += b.count * e
                slope -= b.count * e / (1 + alpha * e)
            else:
                value += b.count * t * mp.expm1(w)
                slope -= b.count * t * e / (1 + alpha * t * e)
        return value, slope

    # L = lse(u) with |u - x| <= alpha: the shift is within alpha of the lift.
    logs = solve(_normalizer(mass, lift, alpha * (1 + mp.mpf(2) ** -20)))
    value, gradient = mp.mpf(0), []
    for b, t, lt, w in zip(row, targets, logt, logs, strict=True):
        if t is None or lt is None:
            g = mp.exp(w)
            value += b.count * g
        else:
            g = t * mp.expm1(w)
            value += b.count * t * (_phi(w) - lt)
        value += b.count * alpha * g * g / 2
        gradient.append(g / loss.temperature)
    return Reference(value, tuple(gradient))


def _topk(loss: Loss, row: Row) -> Reference:
    """Top-k entropy: the u competitors capped at s / k and the free rest by their softmax."""
    target = [j for j, b in enumerate(row) if b.positive]
    if len(target) != 1 or row[target[0]].count != 1:
        raise ValueError("top-k entropy takes one label")
    y, k = target[0], loss.k
    v = {j: _exact(b.value, -row[y].value) for j, b in enumerate(row) if j != y}
    order = sorted(v, key=lambda j: -v[j])
    top = v[order[0]]
    after = [mp.mpf(0)] * (len(order) + 1)  # sum of n exp(v - top) after each position
    for i in reversed(range(len(order))):
        after[i] = after[i + 1] + row[order[i]].count * mp.exp(v[order[i]] - top)
    capped, start, free = [], 0, order
    for i, j in enumerate(order):
        n = row[j].count
        rest = after[i + 1] * mp.exp(top - v[j])
        if start + n >= k or (k - start - n) <= rest:
            free = order[i:]
            break
        capped.append(j)
        start += n
    top = v[free[0]]
    lse = top + mp.log(mp.fsum(row[j].count * mp.exp(v[j] - top) for j in free))
    odds = mp.fsum(row[j].count * v[j] for j in capped) + (k - start) * (lse - mp.log(k - start))
    odds = odds / k + mp.log(k)
    mass = 1 / (1 + mp.exp(-odds))
    gradient = []
    for j in range(len(row)):
        if j == y:
            gradient.append(-mass)
        elif j in capped:
            gradient.append(mass / k)
        else:
            gradient.append(mass * (k - start) / k * mp.exp(v[j] - lse))
    return Reference(mp.log1p(mp.exp(odds)), tuple(gradient))


class CutBlockActive(ValueError):
    """A smoothed hinge's cut row is not the full row's: a cut block is active."""


def _multilabel_hinge(
    z: list[float], mask: list[bool], gamma: float
) -> tuple[MPReal, list[MPReal]]:
    """The multilabel hinge in rationals, rounded to the working digits (its float value
    overflows at finfo.max)."""
    zs = [fraction(v) for v in z]
    pos = [v for v, b in zip(zs, mask, strict=True) if b]
    neg = [v for v, b in zip(zs, mask, strict=True) if not b]

    def real(x: Fraction) -> MPReal:
        return mp.mpf(x.numerator) / x.denominator

    if gamma == 0:
        # Ties split evenly among the lowest positives and the largest negatives, and a margin
        # of zero counts half: the mean of the one-sided subgradients.
        low, high = min(pos), max(neg)
        margin = 1 + high - low
        value = max(Fraction(0), margin)
        factor = Fraction(1) if margin > 0 else (Fraction(1, 2) if margin == 0 else Fraction(0))
        lows = sum(1 for i, v in enumerate(zs) if mask[i] and v == low)
        highs = sum(1 for i, v in enumerate(zs) if not mask[i] and v == high)
        gradient = [mp.mpf(0)] * len(z)
        for i, v in enumerate(zs):
            if mask[i] and v == low:
                gradient[i] = real(-factor / lows)
            elif not mask[i] and v == high:
                gradient[i] = real(factor / highs)
        return real(value), gradient
    g = Fraction(gamma)
    a = [Fraction(1, 2) - (v - pos[0]) for v in pos]
    b = [Fraction(1, 2) + (v - pos[0]) for v in neg]
    x, y = hinges.two_simplex(a, b, g)
    value = sum((h * v - v * v / 2 for h, v in zip(a + b, x + y, strict=True)), Fraction(0)) / g
    xs, ys = iter(x), iter(y)
    return real(value), [real(-next(xs) / g if m else next(ys) / g) for m in mask]


def _hinge(loss: Loss, row: Row) -> Reference:
    """The hinge references; rows wider than 64 classes cut to k + 1 entries per block."""
    keep = loss.k + 1 if loss.name.startswith("topk") else 2
    cut = [min(b.count, keep) if width(row) > 64 else b.count for b in row]
    first = np.cumsum([0, *cut[:-1]]).tolist()
    z = [b.value for b, n in zip(row, cut, strict=True) for _ in range(n)]
    mask = [b.positive for b, n in zip(row, cut, strict=True) for _ in range(n)]
    smooth = loss.gamma > 0
    if loss.name == "multilabel_hinge":
        value, gradient = _multilabel_hinge(z, mask, loss.gamma)
    else:
        variant = "beta" if loss.name == "topk_hinge_beta" else "alpha"
        k = 1 if loss.name == "multiclass_hinge" else loss.k
        value, gradient = hinges.topk_hinge(z, mask.index(True), variant, k, loss.gamma)
    for b, n, i in zip(row, cut, first, strict=True):
        if smooth and n < b.count and gradient[i] != 0:
            raise CutBlockActive(f"a cut block is active: {loss.label} on {row}")
    kink = math.inf if smooth else _kink(loss, z, mask)
    return Reference(mp.mpf(value), tuple(mp.mpf(gradient[i]) for i in first), kink=kink)


def _kink(loss: Loss, z: list[float], mask: list[bool]) -> float:
    """loss_references.nonsmooth_gap in rationals: the distance of the scores from a kink
    (where the subgradients differ), exact where float64 margins would round."""
    zs = [fraction(v) for v in z]
    if loss.name == "multilabel_hinge":
        low = sorted(v for v, b in zip(zs, mask, strict=True) if b)
        high = sorted((v for v, b in zip(zs, mask, strict=True) if not b), reverse=True)
        gaps = [abs(1 + high[0] - low[0])]
        gaps += [low[1] - low[0]] if len(low) > 1 else []
        gaps += [high[0] - high[1]] if len(high) > 1 else []
        return _float(min(gaps))
    k = 1 if loss.name == "multiclass_hinge" else loss.k
    y = mask.index(True)
    h = sorted((1 + v - zs[y] for j, v in enumerate(zs) if j != y), reverse=True)
    gaps = [abs(h[k - 1] - h[k])] if len(h) > k else []
    if loss.name == "topk_hinge_beta":
        gaps.append(min(abs(x) for x in h[:k]))
    else:
        gaps.append(abs(sum(h[:k], Fraction(0)) / k))
    return _float(min(gaps))


def _float(x: Fraction) -> float:
    """x rounded to float64, inf beyond its range."""
    return float(mp.mpf(x.numerator) / x.denominator)


def has_reference(loss: Loss, row: Row) -> bool:
    try:
        reference(loss, row)
    except CutBlockActive:
        return False
    return True


def _soft(row: Row) -> Reference:
    """-sum_j u_j log softmax(z)_j of target masses u and its gradient |u| softmax(z)_j - u_j;
    a class of logit -inf is masked (its term zero where u_j is, +inf where u_j > 0)."""
    finite = [b for b in row if math.isfinite(b.value)]
    top = max(b.value for b in finite)
    lse = mp.log(mp.fsum(b.count * mp.exp(_exact(b.value, -top)) for b in finite))
    mass = mp.fsum(b.count * mp.mpf(b.mass or 0.0) for b in row)
    value, gradient = mp.mpf(0), []
    for b in row:
        u = mp.mpf(b.mass or 0.0)
        if not math.isfinite(b.value):
            value += mp.inf if u > 0 else 0
            gradient.append(-u)
            continue
        if u != 0:
            value += b.count * u * (lse - _exact(b.value, -top))
        gradient.append(mass * mp.exp(_exact(b.value, -top) - lse) - u)
    return Reference(value, tuple(gradient))


def reference(loss: Loss, row: Row) -> Reference:
    """The loss of one row and its gradient per block."""
    return _reference(loss, row, _signs(row))


def _signs(row: Row) -> tuple[float, ...]:
    """The signs of a row's floats: Blocks of 0.0 and -0.0 compare (and hash) equal."""
    floats = [x for b in row for x in (b.value, b.teacher, b.mass) if x is not None]
    return tuple(math.copysign(1.0, x) for x in floats)


@functools.cache
def _reference(loss: Loss, row: Row, signs: tuple[float, ...]) -> Reference:
    with mp.workdps(_digits(loss, row)):
        if any(b.mass is not None for b in row):
            return _soft(row)
        if loss.name in HINGES:
            return _hinge(loss, row)
        if loss.name in PROXIMAL:
            return _envelope(loss, row)
        if loss.name == "topk_cross_entropy" and loss.k > 1:
            return _topk(loss, row)
        return _entropy(loss, row)


def reduced(loss: Loss, rows: Sequence[Row], reduction: str) -> tuple[MPReal, MPReal]:
    """The reduced value of rows and the factor of each row's gradient."""
    with mp.workdps(DIGITS + 10):
        total: MPReal = mp.mpf(0)
        for row in rows:
            total += reference(loss, row).value
        if reduction == "mean":
            return total / len(rows), 1 / mp.mpf(len(rows))
        return total, mp.mpf(1)


def rows(logits: np.ndarray, target: np.ndarray | None = None, teacher: np.ndarray | None = None):
    """Each row of 2-D logits as blocks of its distinct entries (logit, positive flag or target
    mass, teacher logit), and the block of each entry."""
    z = np.asarray(logits, np.float64)
    t = None if target is None else np.asarray(target)
    for i in range(z.shape[0]):
        positive = np.zeros(z.shape[1], bool)
        mass = None
        if t is not None and t.ndim == 1:
            positive[int(t[i])] = True
        elif t is not None and t.dtype == bool:
            positive = t[i]
        elif t is not None:
            mass = np.asarray(t[i], np.float64)
        keys = [
            (
                float(z[i, j]),
                bool(positive[j]),
                0.0 if teacher is None else float(teacher[i, j]),
                None if mass is None else float(mass[j]),
            )
            for j in range(z.shape[1])
        ]
        order: dict[tuple, int] = {}
        counts: list[int] = []
        index = []
        for key in keys:
            if key not in order:
                order[key] = len(order)
                counts.append(0)
            counts[order[key]] += 1
            index.append(order[key])
        row = tuple(Block(k[0], n, k[1], k[2], k[3]) for k, n in zip(order, counts, strict=True))
        yield row, index


def evaluate(
    loss: Loss,
    logits: np.ndarray,
    target: np.ndarray | None = None,
    teacher: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """float64 values (n,) and gradients (n, C), each rounded once, of 2-D logits and their
    targets: labels (n,), a positive mask (n, C) or target masses (n, C) of cross-entropy;
    distillation takes teacher logits (n, C) instead."""
    values, gradients = [], []
    for row, index in rows(logits, target, teacher):
        ref = reference(loss, row)
        values.append(float(ref.value))
        gradients.append([float(ref.gradient[b]) for b in index])
    return np.array(values, np.float64), np.array(gradients, np.float64).reshape(np.shape(logits))


def kinks(loss: Loss, logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per row, a nonsmooth hinge's distance from a kink (where its subgradients differ), in
    units of the scores; inf for the smooth losses."""
    return np.array([reference(loss, row).kink for row, _ in rows(logits, target, None)])


def primal(
    loss: Loss,
    x: np.ndarray,
    w: np.ndarray,
    target: np.ndarray,
    c: float,
    center: np.ndarray | None = None,
) -> MPReal:
    """The solver's primal objective 0.5 ||W - W0||^2 + C sum_i loss(x_i W) at weights W (d, m)
    and center W0 (or 0), with the scores x W exact."""
    x, w = np.asarray(x, np.float64), np.asarray(w, np.float64)
    labels = np.asarray(target)
    with mp.workdps(DIGITS + 60):
        xs = [[mp.mpf(float(v)) for v in row] for row in x]
        ws = [[mp.mpf(float(v)) for v in row] for row in w]
        total: MPReal = mp.mpf(0)
        for i, row in enumerate(xs):
            scores = [
                mp.fsum(a * ws[j][col] for j, a in enumerate(row)) for col in range(w.shape[1])
            ]
            positive = labels[i] if labels.dtype == bool else np.arange(w.shape[1]) == labels[i]
            blocks = tuple(Block(v, 1, bool(p)) for v, p in zip(scores, positive, strict=True))
            total += reference(loss, blocks).value
        w0 = np.zeros_like(w) if center is None else np.asarray(center, np.float64)
        squares = mp.fsum(
            (mp.mpf(float(a)) - mp.mpf(float(b))) ** 2
            for a, b in zip(w.ravel(), w0.ravel(), strict=True)
        )
        return squares / 2 + mp.mpf(c) * total


# The solver's top-k entropy dual steps (label first).


def all_capped_step(scores: np.ndarray, previous: np.ndarray, c: float, norm2: float = 1.0):
    """The solver's top-k entropy dual step of an all-capped block (k = C - 1, label first) at
    scores and previous duals, in 160 digits."""
    # Average exact represented inputs before forming any rounded differences.
    # Solve logit(s) + alpha*(1+1/k)*s = mean + log(k) by bisection.
    with mp.workdps(160):
        k = len(scores) - 1
        z, a = [mp.mpf(float(x)) for x in scores], [mp.mpf(float(x)) for x in previous]
        mean = sum(z[1:]) / k - z[0] - mp.mpf(norm2) * (sum(a[1:]) / k - a[0])
        coefficient = mp.mpf(c) * norm2 * (1 + mp.mpf(1) / k)
        rhs = mean + mp.log(k)
        lo, hi = rhs - coefficient, rhs
        # Include the bracket's exponent: 1e308 needs over 1024 halvings
        # merely to reach unit width, before resolving the actual root.
        for _ in range(max(600, int(mp.log(coefficient, 2)) + 600)):
            mid = (lo + hi) / 2
            if mid == lo or mid == hi:
                break
            mass = 1 / (1 + mp.exp(-mid))
            if mid + coefficient * mass < rhs:
                lo = mid
            else:
                hi = mid
            if hi == lo:
                break
        mass = mp.mpf(c) / (1 + mp.exp(-(lo + hi) / 2))
        return np.array([float(mass)] + [-float(mass / k)] * k)


def capped_softmax_dual(z: np.ndarray, a: np.ndarray, c: float, k: int):
    """C*[mass, -p] of the capped softmax at the updated scores z + a, in 90 digits."""
    with mp.workdps(90):
        v = [mp.mpf(float(z[j])) + mp.mpf(float(a[j])) for j in range(len(z))]
        v = [x - v[0] for x in v[1:]]
        order = sorted(range(len(v)), key=lambda j: -v[j])
        u, free, lse = 0, order, mp.mpf("nan")  # rebound by the loop: k >= 1
        for u in range(k):
            free = order[u:]
            lse = v[order[u]] + mp.log(sum(mp.exp(v[j] - v[order[u]]) for j in free))
            if v[order[u]] <= lse - mp.log(k - u):
                break
        odds = (sum(v[j] for j in order[:u]) + (k - u) * (lse - mp.log(k - u))) / k + mp.log(k)
        mass = 1 / (1 + mp.exp(-odds))
        p = [mp.mpf(0)] * len(v)
        for j in order[:u]:
            p[j] = mass / k
        for j in free:
            p[j] = mass * (1 - mp.mpf(u) / k) * mp.exp(v[j] - lse)
        return np.array([float(c * mass)] + [-float(c * x) for x in p])
