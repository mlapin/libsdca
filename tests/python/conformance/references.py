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
- topk_cross_entropy is the capped-set closed form of test_topk_kernels.reference and
  release_numerics.topk_reference, taken over blocks: tied competitors are
  capped or free together.

The working digits add those of the row's largest magnitude and of its spread (up to 1000),
so differences of represented inputs and tails down to exp(-1000) keep DIGITS digits.

The hinges reuse test_topk_kernels.reference (60 digits) and test_multilabel_kernels.exact
(rationals) on the row with each block cut to k + 1 entries. A cut block of a smooth hinge
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

DIGITS = 50
HINGES = ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge")
PROXIMAL = ("proximal_cross_entropy", "proximal_distillation")
DISTILLATION = ("distillation", "proximal_distillation")


@dataclass(frozen=True)
class Block:
    """count entries of logit value (and teacher logit teacher; a positive label if positive)."""

    value: float
    count: int = 1
    positive: bool = False
    teacher: float = 0.0


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
    inputs = [[b.value for b in row]]
    inputs += [[b.teacher for b in row]] if loss.name in DISTILLATION else []
    big = max(abs(v) for values in inputs for v in values)
    with np.errstate(over="ignore"):
        gaps = [(max(v) - x) / loss.temperature for v in inputs for x in v]
    spread = max(g for g in gaps if g <= 760)
    return DIGITS + 10 + int(math.log10(1 + big)) + int(spread / math.log(10))


def _exact(*terms: float) -> MPReal:
    """The sum of represented inputs, rounded once to the working digits."""
    total = sum(map(Fraction, terms), Fraction(0))
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
        top = max(b.teacher for b in row)
        shifted = [_exact(b.teacher, -top) / loss.temperature for b in row]
        ties = sum(b.count for b in row if b.teacher == top)
        rest = mp.fsum(b.count * mp.exp(s) for b, s in zip(row, shifted, strict=True) if s < 0)
        normalizer = mp.log1p(ties - 1 + rest)
        return [s - normalizer for s in shifted]
    positives = sum(b.count for b in row if b.positive)
    if positives == 0:
        raise ValueError("a row without a positive label")
    return [-mp.log(positives) if b.positive else None for b in row]


def _offsets(loss: Loss, row: Row, anchor: int) -> list[MPReal]:
    """(x_j - x_a) - (log t_j - log t_a) with x = z / T: exact differences of inputs, so that
    a student near its teacher keeps the digits of their difference."""
    a = row[anchor]
    if loss.name in DISTILLATION:
        return [_exact(b.value, -a.value, -b.teacher, a.teacher) / loss.temperature for b in row]
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
    """test_multilabel_kernels.exact in rationals, rounded to the working digits (its float
    value overflows at finfo.max)."""
    import test_multilabel_kernels as multilabel

    zs = [Fraction(v) for v in z]
    pos = [v for v, b in zip(zs, mask, strict=True) if b]
    neg = [v for v, b in zip(zs, mask, strict=True) if not b]

    def real(x: Fraction) -> MPReal:
        return mp.mpf(x.numerator) / x.denominator

    if gamma == 0:
        low, high = min(pos), max(neg)
        value = max(Fraction(0), 1 + high - low)
        gradient = [mp.mpf(0)] * len(z)
        if value > 0:
            gradient[next(i for i, v in enumerate(zs) if mask[i] and v == low)] = mp.mpf(-1)
            gradient[next(i for i, v in enumerate(zs) if not mask[i] and v == high)] = mp.mpf(1)
        return real(value), gradient
    g = Fraction(gamma)
    a = [Fraction(1, 2) - (v - pos[0]) for v in pos]
    b = [Fraction(1, 2) + (v - pos[0]) for v in neg]
    x, y = multilabel._two_simplex(a, b, g)
    value = sum((h * v - v * v / 2 for h, v in zip(a + b, x + y, strict=True)), Fraction(0)) / g
    xs, ys = iter(x), iter(y)
    return real(value), [real(-next(xs) / g if m else next(ys) / g) for m in mask]


def _hinge(loss: Loss, row: Row) -> Reference:
    """The hinge references of the kernel tests; rows wider than 64 classes cut to k + 1
    entries per block."""
    import test_topk_kernels as topk

    keep = loss.k + 1 if loss.name.startswith("topk") else 2
    cut = [min(b.count, keep) if width(row) > 64 else b.count for b in row]
    first = np.cumsum([0, *cut[:-1]]).tolist()
    z = [b.value for b, n in zip(row, cut, strict=True) for _ in range(n)]
    mask = [b.positive for b, n in zip(row, cut, strict=True) for _ in range(n)]
    smooth = loss.gamma > 0
    if loss.name == "multilabel_hinge":
        value, gradient = _multilabel_hinge(z, mask, loss.gamma)
    else:
        name = "topk_hinge_alpha" if loss.name == "multiclass_hinge" else loss.name
        k = 1 if loss.name == "multiclass_hinge" else loss.k
        kind = name + ("_smooth" if smooth else "")
        value, gradient = topk.reference(z, mask.index(True), kind, k, loss.gamma)
    for b, n, i in zip(row, cut, first, strict=True):
        if smooth and n < b.count and gradient[i] != 0:
            raise CutBlockActive(f"a cut block is active: {loss.label} on {row}")
    kink = math.inf if smooth else _kink(loss, z, mask)
    return Reference(mp.mpf(value), tuple(mp.mpf(gradient[i]) for i in first), kink=kink)


def _kink(loss: Loss, z: list[float], mask: list[bool]) -> float:
    """loss_references.nonsmooth_gap in rationals: the distance of the scores from a kink
    (where the subgradients differ), exact where float64 margins would round."""
    zs = [Fraction(v) for v in z]
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


@functools.cache
def reference(loss: Loss, row: Row) -> Reference:
    """The loss of one row and its gradient per block."""
    with mp.workdps(_digits(loss, row)):
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
