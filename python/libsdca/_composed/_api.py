"""loss(name, scores, target, ...): the composed losses as the frontends call them."""

from __future__ import annotations

from typing import Any

from .. import _frontend as _f
from . import _losses, _targets
from ._xp import Array, Namespace

NAMES = frozenset(_f.KINDS)
_TEACHER = _f.DISTILLATION
_SINGLE_LABEL = NAMES - _f.MULTILABEL - _TEACHER - {"cross_entropy"}


def namespace(x: Any) -> Namespace:
    """xp_torch for PyTorch tensors, else xp_jax."""
    if type(x).__module__.startswith("torch"):
        from . import xp_torch as xp
    else:
        from . import xp_jax as xp
    return xp


def loss(
    name: str,
    scores: Array,
    target: Array,
    *,
    k: int = 1,
    gamma: float = 0.0,
    temperature: float = 1.0,
    label_smoothing: float | Array = 0.0,
    weight: Array | None = None,
    ignore_index: int | None = None,
    reduction: str = "mean",
    precision: str | None = None,
) -> Array:
    """The loss name of class-last scores (..., C) and their target: labels (...) of any
    integer dtype, a bool mask (..., C), probabilities (..., C) (cross_entropy) or teacher
    logits (..., C) (probabilities and teachers with the gradients of _targets). Per example (...) for reduction "none", else their "sum"
    or "mean", in promote(scores.dtype, float32).

    weight (C,) and ignore_index apply to every single-label loss as torch's cross-entropy
    takes them: w[label] times each example's loss, 0 where ignored, the mean dividing by the
    sum of those weights (_frontend.mean_divisor; cross_entropy also weighs its label
    smoothing, as PyTorch). precision "float64" computes in float64; None and "float32" in
    float32 for 16- and 32-bit scores. Options are checked and weights given
    _frontend.nan_weights by the caller.
    """
    if name not in NAMES:
        raise ValueError(f"unknown loss {name!r}")
    xp = namespace(scores)
    shape, classes = tuple(scores.shape[:-1]), scores.shape[-1]
    z = scores.reshape(-1, classes)
    z = xp.astype(z, xp.float64) if precision == "float64" else xp.work(z)
    labels = name in _SINGLE_LABEL or (name == "cross_entropy" and target.shape != scores.shape)
    probabilities = name == "cross_entropy" and not labels
    value_dtype = _f.value_dtype(xp, scores.dtype, target.dtype if probabilities else None)
    t = target.reshape(-1) if labels else target.reshape(-1, classes)
    if labels:
        t = xp.astype(t, xp.int_)
    elif name in _TEACHER and precision == "float64":
        t = xp.astype(t, xp.float64)
    class_weight = weight
    if weight is not None:
        weight = xp.astype(weight, z.dtype)

    if name == "cross_entropy":
        values = _losses.cross_entropy(
            xp,
            z,
            t if labels else xp.stop(t),
            weight=weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        if not labels:
            gradient = _targets.probability_gradient
            values = xp.target_gradient(
                values, t, gradient, 1.0, xp.stop(z), weight, label_smoothing
            )
    elif name in _SINGLE_LABEL:
        values = _single_label(xp, name, z, t, k, gamma, weight, ignore_index)
    elif name == "multilabel_cross_entropy":
        values = _losses.multilabel_cross_entropy(xp, z, t)
    elif name == "multilabel_hinge":
        values = _losses.multilabel_hinge(xp, z, t, gamma=gamma)
    elif name == "distillation":
        values = _losses.distillation(xp, z, t, temperature=temperature)
        if xp.name == "jax":
            from .xp_jax import attach_teacher

            values = attach_teacher(values, z, t, temperature)
        else:
            values = xp.target_gradient(
                values, t, _targets.teacher_gradient, 1.0, xp.stop(z), xp.stop(t), temperature
            )
    else:
        values = _losses.proximal_distillation(xp, z, t, temperature=temperature, gamma=gamma)
        values = xp.target_gradient(
            values, t, _proximal_teacher, 1.0, xp.stop(z), xp.stop(t), temperature, gamma
        )

    if reduction == "none":
        return xp.astype(values, value_dtype).reshape(shape)
    wide = xp.astype(values, xp.wide())
    total = wide.sum()
    if reduction == "mean":
        total = total / _f.mean_divisor(xp, t, class_weight, ignore_index, labels)
    return xp.astype(total, value_dtype)


def _proximal_teacher(xp: Namespace, z: Array, t: Array, temperature: float, gamma: float) -> Array:
    """_targets.teacher_gradient of proximal_distillation, of the composed student gradient."""

    def loss(z: Array) -> Array:
        return _losses.proximal_distillation(xp, z, t, temperature=temperature, gamma=gamma)

    student = xp.grad(loss, z)
    return _targets.teacher_gradient(xp, z, t, temperature, student, gamma)


def _single_label(
    xp: Namespace,
    name: str,
    z: Array,
    t: Array,
    k: int,
    gamma: float,
    weight: Array | None,
    ignore_index: int | None,
) -> Array:
    """A single-label loss other than cross_entropy, weighted and with ignored rows of
    label 0 and zero scores (neither read nor checked, with a +0.0 gradient)."""
    ignored = None
    if ignore_index is not None:
        ignored = t == ignore_index
        z = xp.where(ignored[:, None], 0.0, z)
        t = xp.where(ignored, 0, t)
    if name == "topk_cross_entropy":
        values = _losses.topk_cross_entropy(xp, z, t, k=k)
    elif name == "multiclass_hinge":
        values = _losses.multiclass_hinge(xp, z, t, gamma=gamma)
    elif name == "proximal_cross_entropy":
        values = _losses.proximal_cross_entropy(xp, z, t, gamma=gamma)
    else:
        variant = name.removeprefix("topk_hinge_")
        values = _losses.topk_hinge(xp, z, t, k=k, variant=variant, gamma=gamma)
    if weight is not None:
        values = values * _losses.label_weights(xp, z, t, weight=weight, ignore_index=None)
    return values if ignored is None else xp.where(ignored, 0.0, values)
