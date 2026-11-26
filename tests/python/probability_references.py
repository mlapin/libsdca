"""Cross-entropy of probability targets: the rows the accuracy tests draw, their references
(conformance.references) and their error measures.

The loss is -sum_j u_j log softmax(z)_j of the target weights u = w ((1 - eps) t + eps / C)
and its gradient T softmax(z) - u (T = sum_j u_j); a class of -inf is masked (its term zero
where u_j is, +inf where u_j > 0).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import storage_limits
from conformance.references import Loss, evaluate

CLASSES = 37
ROWS = 8
# (name, logits, targets, options: label_smoothing and weight)
Family = tuple[str, np.ndarray, np.ndarray, dict[str, Any]]


def families(seed: int = 7) -> Iterator[Family]:
    """(name, logits, targets, options) of rows of ordinary, wide and confident logits,
    targets equal to the softmax (whose float64 gradient is rounding noise), masked classes of
    target zero, targets that sum to three, and label smoothing with class weights."""
    rng = np.random.default_rng(seed)
    n, c = ROWS, CLASSES

    def dirichlet(alpha: float = 1.0) -> np.ndarray:
        return rng.dirichlet(np.full(c, alpha), n)

    yield "ordinary", 3 * rng.standard_normal((n, c)), dirichlet(), {}
    yield "wide", 300 * rng.standard_normal((n, c)), dirichlet(), {}
    z = rng.standard_normal((n, c))
    y = rng.integers(c, size=n)
    z[np.arange(n), y] += 40
    t = np.zeros((n, c))
    t[np.arange(n), y] = 1
    yield "confident", z, t, {}
    z = 3 * rng.standard_normal((n, c))
    e = np.exp(z - z.max(1, keepdims=True))
    yield "softmax", z, e / e.sum(1, keepdims=True), {}
    z, t = 3 * rng.standard_normal((n, c)), dirichlet(0.3)
    hide = rng.random((n, c)) < 0.2
    hide[:, 0] = False
    z[hide], t[hide] = -np.inf, 0
    yield "masked", z, t, {}
    yield "unnormalized", 3 * rng.standard_normal((n, c)), 3 * dirichlet(), {}
    options = {"label_smoothing": 0.1, "weight": rng.uniform(0.25, 2, c)}
    yield "smoothed", 3 * rng.standard_normal((n, c)), dirichlet(), options


def reference(
    z: np.ndarray, t: np.ndarray, eps: float = 0.0, w: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The loss (n,) and gradient (n, C) rounded to float64 (conformance.references, of the
    target weights u), and each row's largest term max_j max(T p_j, u_j) (n, 1), of float64
    logits, targets and weights."""
    c = z.shape[1]
    u = t * (1 - eps) + eps / c if eps else t.astype(np.float64)
    if w is not None:
        u = u * w
    values, gradients = evaluate(Loss("cross_entropy"), z, u)
    with np.errstate(invalid="ignore", over="ignore"):
        p = np.exp(z - z.max(1, keepdims=True))
        p = p / p.sum(1, keepdims=True)
    terms = np.maximum(u.sum(1, keepdims=True) * p, u).max(1, keepdims=True)
    return values, gradients, terms


EPS = storage_limits.EPS


def loss_ulps(value: np.ndarray, want: np.ndarray, dtype: str) -> float:
    """The largest relative error of finite losses in ulps of dtype (absolute at zero)."""
    finite = np.isfinite(want)
    assert (value[~finite] == want[~finite]).all()
    v, w = value[finite], want[finite]
    error = np.abs(v - w) / np.where(w != 0, np.abs(w), 1)
    return float(error.max(initial=0) / EPS[dtype])


def gradient_ulps(gradient: np.ndarray, want: np.ndarray, scale: np.ndarray, dtype: str) -> float:
    """The largest error of the gradient rows relative to scale (n, 1), in ulps of dtype."""
    error = np.abs(gradient - want) / np.where(scale > 0, scale, 1)
    return float(error.max(initial=0) / EPS[dtype])
