"""Float64 NumPy references of the losses, independent of libsdca's solvers.

Every hinge is max_{b in B} <h, b> - gamma/2 ||b||^2 over margins h and its feasible set B
(gamma = 0: the nonsmooth hinge); b = proj_B(h / gamma) is found here by bisection on the
multipliers, not by the sorting algorithms of libsdca. The entropy losses are log-sum-exp
compositions. Top-k entropy and proximal entropy have no reference here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]
REFERENCED = (
    "cross_entropy",
    "multiclass_hinge",
    "topk_hinge_alpha",
    "topk_hinge_beta",
    "multilabel_hinge",
    "multilabel_cross_entropy",
)
_STEPS = 120


def _bisect(f: Callable[[Array], Array], low: Array, high: Array) -> Array:
    """The root of each decreasing f between low and high, per row."""
    for _ in range(_STEPS):
        mid = 0.5 * (low + high)
        right = f(mid) > 0
        low, high = np.where(right, mid, low), np.where(right, high, mid)
    return 0.5 * (low + high)


def _clip(x: Array, valid: Array, t: Array | float, cap: Array) -> Array:
    return np.where(valid, np.clip(x - t, 0, cap), 0)


def _threshold(x: Array, valid: Array, total: Array, cap: Array) -> Array:
    """t of the projection clip(x - t, 0, cap) of each row of x (valid entries) onto
    {0 <= b <= cap, sum b = total}."""
    span = np.abs(np.where(valid, x, 0)).max(-1, keepdims=True) + cap + 1

    def excess(t: Array) -> Array:
        return _clip(x, valid, t, cap).sum(-1, keepdims=True) - total

    return _bisect(excess, -span, span)


def _projection(x: Array, valid: Array, kind: str, k: int, positive: Array) -> Array:
    """proj_B(x) for the hinge sets: kind 'beta' (0 <= b <= 1/k, sum <= 1), 'alpha' (the top-k
    simplex: 0 <= b <= sum/k, sum <= 1) and 'two' (sums over the positive and the other valid
    entries equal and <= 1).

    'alpha' and 'two' bisect on s = sum(b) for a zero derivative of the convex distance from x
    to its projection at sum s: minus the multipliers of the sums and caps."""
    one = np.ones((len(x), 1))
    if kind == "beta":
        free = _clip(x, valid, 0.0, one / k)
        tight = _clip(x, valid, _threshold(x, valid, one, one / k), one / k)
        return np.where(free.sum(-1, keepdims=True) > 1, tight, free)
    negative = valid & ~positive

    def slope(s: Array) -> Array:  # minus the derivative, decreasing in s
        if kind == "two":
            return _threshold(x, positive, s, 2 + s) + _threshold(x, negative, s, 2 + s)
        t = _threshold(x, valid, s, s / k)
        capped = valid & (x - t >= s / k)
        return t + np.where(capped, x - t - s / k, 0).sum(-1, keepdims=True) / k

    s = _bisect(slope, 0 * one, one)
    if kind == "two":
        b = _clip(x, positive, _threshold(x, positive, s, 2 + s), 2 + s)
        return b + _clip(x, negative, _threshold(x, negative, s, 2 + s), 2 + s)
    return _clip(x, valid, _threshold(x, valid, s, s / k), s / k)


def _smooth(
    h: Array, valid: Array, gamma: float, kind: str, k: int, positive: Array
) -> tuple[Array, Array]:
    """Value and maximizer b of max_{b in B} <h, b> - gamma/2 ||b||^2 per row."""
    h = np.where(valid, h, 0.0)
    b = _projection(h / gamma, valid, kind, k, positive)
    return (h * b).sum(-1) - 0.5 * gamma * (b * b).sum(-1), b


def _topk_nonsmooth(h: Array, kind: str, k: int) -> tuple[Array, Array]:
    """Value and the subgradient weights b of the nonsmooth top-k hinges of margins h (-inf at
    the target; ties: lowest index)."""
    order = np.argsort(-h, axis=-1, kind="stable")[:, :k]
    top = np.take_along_axis(h, order, -1)
    b = np.zeros_like(h)
    rows = np.arange(len(h))[:, None]
    if kind == "alpha":
        value = np.maximum(top.mean(-1), 0)
        b[rows, order] = np.where(value[:, None] > 0, 1.0 / k, 0.0)
    else:
        value = np.maximum(top, 0).mean(-1)
        b[rows, order] = np.where(top > 0, 1.0 / k, 0.0)
    return value, b


def _entropy(z: Array, positive: Array) -> tuple[Array, Array]:
    """Cross-entropy to the uniform distribution over the positive classes."""
    shift = z.max(-1, keepdims=True)
    e = np.exp(z - shift)
    lse = np.log(e.sum(-1)) + shift[:, 0]
    count = positive.sum(-1, keepdims=True)
    value = lse - np.where(positive, z, 0).sum(-1) / count[:, 0]
    return value, e / e.sum(-1, keepdims=True) - positive / count


def reference(
    z: npt.ArrayLike, positive: npt.ArrayLike, loss: str, k: int = 1, gamma: float = 0.0
) -> tuple[Array, Array]:
    """Float64 per-row losses (n,) and gradients (n, m) of scores z and a positive mask.

    Nonsmooth gradients are the subgradients whose ties go to the lowest index; compare them
    only where nonsmooth_gap is clear of rounding.
    """
    z = np.asarray(z, np.float64)
    mask = np.asarray(positive, bool)
    if loss in ("cross_entropy", "multilabel_cross_entropy"):
        return _entropy(z, mask)
    rows = np.arange(len(z))
    if loss == "multilabel_hinge":
        if gamma == 0:
            low = np.where(mask, z, np.inf).argmin(-1)
            high = np.where(mask, -np.inf, z).argmax(-1)
            value = np.maximum(1 + z[rows, high] - z[rows, low], 0)
            g = np.zeros_like(z)
            g[rows, high] += value > 0
            g[rows, low] -= value > 0
            return value, g
        # The loss of z less a constant: margins of the order of the spread.
        c = z - z.mean(-1, keepdims=True)
        h = np.where(mask, 0.5 - c, 0.5 + c)
        value, b = _smooth(h, np.ones_like(mask), gamma, "two", 1, mask)
        return value, np.where(mask, -b, b)
    if loss not in ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta"):
        raise ValueError(f"no reference for {loss}")
    k = 1 if loss == "multiclass_hinge" else k
    # At k = 1 the alpha set is the beta set: the simplex.
    kind = "alpha" if loss == "topk_hinge_alpha" and k > 1 else "beta"
    label = mask.argmax(-1)
    h = 1 + z - z[rows, label][:, None]
    valid = ~mask
    if gamma == 0:
        value, b = _topk_nonsmooth(np.where(valid, h, -np.inf), kind, k)
    else:
        value, b = _smooth(h, valid, gamma, kind, k, mask)
    b = np.where(valid, b, 0)
    b[rows, label] = -b.sum(-1)
    return value, b


def nonsmooth_gap(z: npt.ArrayLike, positive: npt.ArrayLike, loss: str, k: int = 1) -> Array:
    """Per row, the distance of a nonsmooth hinge from a kink (where subgradients differ), in
    units of the scores."""
    z = np.asarray(z, np.float64)
    mask = np.asarray(positive, bool)
    if loss == "multilabel_hinge":
        low = np.sort(np.where(mask, z, np.inf), -1)
        high = -np.sort(np.where(mask, np.inf, -z), -1)
        gaps = [np.abs(1 + high[:, 0] - low[:, 0])]
        if z.shape[1] > 2:
            gaps.append(np.where(mask.sum(-1) > 1, low[:, 1] - low[:, 0], np.inf))
            gaps.append(np.where((~mask).sum(-1) > 1, high[:, 0] - high[:, 1], np.inf))
        return np.minimum.reduce(gaps)
    k = 1 if loss == "multiclass_hinge" else k
    label = mask.argmax(-1)
    h = np.where(mask, -np.inf, 1 + z - z[np.arange(len(z)), label][:, None])
    s = -np.sort(-h, -1)
    gaps = [np.abs(s[:, k - 1] - s[:, k]) if s.shape[1] > k + 1 else np.full(len(z), np.inf)]
    if loss == "topk_hinge_beta":
        gaps.append(np.abs(s[:, :k]).min(-1))
    else:
        gaps.append(np.abs(s[:, :k].mean(-1)))
    return np.minimum.reduce(gaps)


def primal(
    x: npt.ArrayLike, w: npt.ArrayLike, y: npt.ArrayLike, objective: str, C: float, **options: Any
) -> float:
    """0.5 ||W||^2 + C sum loss(X W) in float64 at weights W (d, m), for labels or a mask y;
    the losses without a reference here from float64 libsdca.loss_and_grad."""
    from libsdca import Solver

    x, w = np.asarray(x, np.float64), np.asarray(w, np.float64)
    scores = x @ w
    labels = np.asarray(y)
    positive = labels if labels.dtype == bool else np.eye(w.shape[1], dtype=bool)[labels]
    if objective in REFERENCED:
        values = reference(scores, positive, objective, **options)[0]
    else:
        values = Solver("float64").loss_and_grad(scores, positive, objective, **options)[0]
    return 0.5 * float(np.sum(w * w)) + C * float(values.sum())
