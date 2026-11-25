"""The derivatives of differentiable targets, written once for every path that computes their
losses (the kernels of libsdca.torch and libsdca.jax and the composed losses): class
probabilities of cross_entropy and teacher logits of the distillation losses."""

from __future__ import annotations

from typing import Any

from ._xp import Array, Namespace


def probability_gradient(
    xp: Namespace, logits: Array, weight: Array | None, label_smoothing: Any
) -> Array:
    """d/dt of cross_entropy of probability targets t (rows, classes) per unit of each row's
    loss: -(1 - eps) w_c log softmax(z)_c (+inf at a masked class), as PyTorch's autograd.
    In the work dtype of the logits."""
    z = xp.work(logits)
    nll = -xp.log_softmax(z)
    plain = (
        isinstance(label_smoothing, int) or isinstance(label_smoothing, float)
    ) and label_smoothing == 0
    gradient = nll if plain else (1 - label_smoothing) * nll
    if weight is not None:
        gradient = gradient * xp.astype(weight, z.dtype)[None, :]
    return gradient


def teacher_gradient(
    xp: Namespace,
    logits: Array,
    teacher: Array,
    temperature: float,
    student: Array | None = None,
    gamma: float = 0.0,
) -> Array:
    """d/dt of distillation H(p, q), p = softmax(t / T), q = softmax(z / T), in the teacher
    logits t per unit of each row's loss: p_c (-log q_c - H(p, q)) / T, as PyTorch's autograd
    of cross_entropy(z / T, softmax(t / T)); 0 where p_c = 0 (a class the teacher masks).

    proximal_distillation (student: its gradient (pi - p) / T per unit of loss; gamma): by
    the envelope theorem the same with q = pi = softmax(u*), the optimum u* = z / T -
    gamma T (pi - p) / T of its strongly convex inner problem. In the work dtype of the logits."""
    z = xp.work(logits)
    u = z / temperature
    if student is not None:
        u = u - (gamma * temperature) * xp.astype(student, z.dtype)
    p = xp.exp(xp.log_softmax(xp.astype(teacher, z.dtype) / temperature))
    kept = p > 0
    nll = xp.where(kept, -xp.log_softmax(u), 0.0)
    loss = xp.sum(p * nll)
    return xp.where(kept, p * (nll - loss) / temperature, 0.0)


def cotangent(xp: Namespace, gradient: Array, upstream: Array, scale: float, dtype: Any) -> Array:
    """The target's cotangent: gradient (rows, classes, per unit of loss) times upstream (one
    factor per row, or one for all) times scale, in the target's dtype."""
    factor = xp.astype(upstream, gradient.dtype)
    if scale != 1:
        factor = factor * scale
    return xp.astype(factor.reshape((*upstream.shape, 1)) * gradient, dtype)
