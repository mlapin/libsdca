"""optax's classification losses on libsdca's kernels: the signatures and per-example outputs
of optax 0.2.9's softmax_cross_entropy_with_integer_labels, softmax_cross_entropy,
safe_softmax_cross_entropy and multiclass_hinge_loss, computed by libsdca.jax (jit, grad and
vmap; the kernels where libsdca.jax has them).

Results have the dtype optax returns, computed in float32 work for 16-bit logits (float64 for
float64) and rounded once. Where optax and libsdca differ (docs/JAX.md, optax losses):

- a logit of -inf masks its class (docs/API.md, Masked classes): a class without target mass
  is removed, so rows with -inf give finite values where optax's softmax_cross_entropy and
  multiclass_hinge_loss give NaN; target mass on a masked class gives +inf;
- a class-shaped ``where`` masks classes in the same way, so a label or target mass on an
  excluded class gives +inf (optax leaves it out of the sum);
- targets are finite and nonnegative (other rows give NaN), labels in [0, classes) (others:
  NaN).
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from . import _frontend as _f
from . import jax as _losses

__all__ = [
    "multiclass_hinge_loss",
    "safe_softmax_cross_entropy",
    "softmax_cross_entropy",
    "softmax_cross_entropy_with_integer_labels",
]

Axis = int | tuple[int, ...] | None


def _axes(axis: object, ndim: int) -> tuple[int, ...]:
    """The class axes, in [0, ndim) and in order: one, several (a tuple) or all (None)."""
    if axis is None:
        return tuple(range(ndim))
    if not isinstance(axis, tuple):
        return (_f.class_axis(_losses.JAX, ndim, _f.check_axis(axis, "axis")),)
    axes = tuple(_f.class_axis(_losses.JAX, ndim, _f.check_axis(a, "axis")) for a in axis)
    if len(set(axes)) != len(axes):
        raise ValueError(f"axis {axis} repeats an axis")
    return axes


def _class_axis(x: jax.Array, axes: tuple[int, ...]) -> tuple[jax.Array, int]:
    """x and the class axis for libsdca.jax: one class axis as it is, several moved to the
    end and flattened into one (optax's order)."""
    if len(axes) == 1:
        return x, axes[0]
    batch = tuple(a for a in range(x.ndim) if a not in axes)
    x = x.transpose(batch + axes)
    return x.reshape(x.shape[: len(batch)] + (math.prod(x.shape[len(batch) :]),)), -1


def _mask(where: ArrayLike, shape: tuple[int, ...], axes: tuple[int, ...]) -> Any:
    """optax's where as a class mask of the logits' shape: given so, or of the batch shape
    (whole examples) and broadcast over the class axes."""
    mask = jnp.asarray(where, dtype=bool)
    if mask.ndim != len(shape):
        mask = jnp.expand_dims(mask, axes)
    return _class_axis(jnp.broadcast_to(mask, shape), axes)[0]


def _cross_entropy(
    z: jax.Array,
    target: jax.Array,
    axes: tuple[int, ...],
    where: ArrayLike | None,
    dtype: Any,
    empty: float,
) -> jax.Array:
    """Per-example cross_entropy of libsdca.jax in optax's dtype (in float64 work where that
    is float64) of labels or probabilities (of the logits' shape), with the classes where
    excludes at -inf and empty (what optax gives) for a row of no class."""
    mask = None if where is None else _mask(where, z.shape, axes)
    if jnp.issubdtype(target.dtype, jnp.floating):
        target = _class_axis(target, axes)[0]
    z, axis = _class_axis(z, axes)
    precision = _f.work(dtype, None)

    def loss(z: jax.Array) -> jax.Array:
        return _losses.cross_entropy(z, target, reduction="none", axis=axis, precision=precision)

    if mask is None:
        return loss(z).astype(dtype)
    keep = mask.any(axis)
    if target.ndim != z.ndim:
        target = jnp.where(keep, target, 0)
    z = jnp.where(mask, z, -jnp.inf)
    z = jnp.where(jnp.expand_dims(keep, axis), z, jnp.zeros((), z.dtype))
    return jnp.where(keep, loss(z), empty).astype(dtype)


def softmax_cross_entropy_with_integer_labels(
    logits: ArrayLike,
    labels: ArrayLike,
    axis: int | tuple[int, ...] = -1,
    where: ArrayLike | None = None,
) -> jax.Array:
    """Softmax cross-entropy of integer labels, as optax: logsumexp(logits) - logits[label]
    over the class axis or axes (a label is a flat index into a tuple's axes), per example.

    where: None, a class mask of the logits' shape (excluded classes are masked; an excluded
    label gives +inf) or an example mask of the labels' shape. A row that where leaves no
    class gives 0.0, as optax. Returns losses of the labels' shape in the logits' dtype.
    """
    z, y = jnp.asarray(logits), jnp.asarray(labels)
    if not jnp.issubdtype(y.dtype, jnp.integer):
        raise TypeError("labels must have an integer dtype")
    axes = _axes(axis, z.ndim)
    expected = tuple(n for i, n in enumerate(z.shape) if i not in axes)
    if y.shape != expected:
        raise ValueError(f"labels must have shape {expected}, got {y.shape}")
    return _cross_entropy(z, y, axes, where, z.dtype, 0.0)


def softmax_cross_entropy(
    logits: ArrayLike,
    labels: ArrayLike,
    axis: Axis = -1,
    where: ArrayLike | None = None,
) -> jax.Array:
    """Softmax cross-entropy of target probabilities (labels of the logits' shape, not
    normalized), as optax: -sum(labels * log_softmax(logits)) over the class axis or axes
    (None: every axis), per example; differentiable in the logits and the labels.

    where: None, a class mask of the logits' shape (excluded classes are masked: no target
    mass there gives 0, target mass +inf) or an example mask of the batch shape. A row that
    where leaves no class gives -0.0, as optax (whose labels' gradient there is NaN; zero
    here). A logit of -inf with no target mass gives 0, where optax gives NaN. Returns losses
    of the batch shape in promote_types(logits, labels).
    """
    z, t = jnp.asarray(logits), jnp.asarray(labels)
    dtype = jnp.result_type(z, t)
    if not jnp.issubdtype(t.dtype, jnp.floating):
        t = t.astype(dtype)
    _f.check_probabilities(t.shape, z.shape)
    return _cross_entropy(z, t, _axes(axis, z.ndim), where, dtype, -0.0)


def safe_softmax_cross_entropy(logits: ArrayLike, labels: ArrayLike) -> jax.Array:
    """softmax_cross_entropy over the last axis with 0 * log 0 = 0, as optax: a logit of -inf
    with no target mass adds nothing, with target mass gives +inf."""
    return softmax_cross_entropy(logits, labels)


def multiclass_hinge_loss(scores: ArrayLike, labels: ArrayLike) -> jax.Array:
    """Multiclass hinge loss of integer labels (Crammer and Singer), as optax:
    max(scores + 1 - one_hot(labels)) - scores[label] over the last axis, per example.

    A score of -inf masks its class (optax: NaN). At a kink (a tie, or a margin of exactly
    zero) the gradient averages the one-sided subgradients, as optax. Returns losses of the
    labels' shape in the default float dtype (float32; float64 with jax_enable_x64), as optax.
    """
    s = jnp.asarray(scores)
    dtype = jax.dtypes.canonicalize_dtype(jnp.promote_types(s.dtype, jnp.float64))
    values = _losses.multiclass_hinge_loss(
        s, labels, reduction="none", precision=_f.work(dtype, None)
    )
    return values.astype(dtype)
