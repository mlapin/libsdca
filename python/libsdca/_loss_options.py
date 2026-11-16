"""The options of the PyTorch and JAX losses, validated once (constructors and calls alike).

Wrong types raise TypeError, bad values ValueError, before any computation. The tensor kinds
are the losses the tensor backends, the CPU extension and the native library compute:
multiclass_hinge is the top-k alpha hinge at k = 1, multilabel_cross_entropy the entropy of a
boolean mask, and a hinge smoothed by gamma > 0 takes the suffix _smooth (gamma is then 1 for
the nonsmooth kinds).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any, NamedTuple

REDUCTIONS = ("none", "sum", "mean")
VARIANTS = ("alpha", "beta")
# The losses of one class label per example, and of a boolean mask.
SINGLE_LABEL = frozenset(
    (
        "cross_entropy",
        "topk_cross_entropy",
        "multiclass_hinge",
        "topk_hinge_alpha",
        "topk_hinge_beta",
        "proximal_cross_entropy",
    )
)
MULTILABEL = frozenset(("multilabel_cross_entropy", "multilabel_hinge"))
_HINGES = frozenset(("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge"))
_BASE_KINDS = {"multiclass_hinge": "topk_hinge_alpha", "multilabel_cross_entropy": "cross_entropy"}


class Spec(NamedTuple):
    """A loss with checked options: its name, tensor kind, and the kind's k, gamma and alpha."""

    name: str
    objective: str
    k: int
    gamma: float
    alpha: float

    @property
    def multilabel(self) -> bool:
        return self.name in MULTILABEL


def spec(name: str, k: int = 1, gamma: float = 0.0, alpha: float = 1.0) -> Spec:
    """The Spec of a loss from checked options (check_k, check_gamma, check_positive); the kind
    of gamma 0 (and of a loss without gamma) takes gamma 1, which no kernel reads."""
    base = _BASE_KINDS.get(name, name)
    k = 1 if name == "multiclass_hinge" else k
    if name in _HINGES and gamma > 0:
        return Spec(name, base + "_smooth", k, gamma, alpha)
    return Spec(name, base, k, 1.0, alpha)


def loss_of_kind(kind: str) -> tuple[str, float | None]:
    """The loss of the native library computing a tensor kind, and gamma 0 for a nonsmooth
    kind (None: the given gamma)."""
    return kind.removesuffix("_smooth"), None if kind.endswith("_smooth") else 0.0


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    return float(value)


def check_k(k: object) -> int:
    """A top-k parameter: an integer >= 1 (k < classes is checked with the logits)."""
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise TypeError(f"k must be an integer, not {type(k).__name__}")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return int(k)


def check_classes(k: int, classes: int) -> None:
    """At least two classes, and k in [1, classes - 1]."""
    if classes < 2:
        raise ValueError("At least two classes are required")
    if k >= classes:
        raise ValueError(f"k must be in [1, classes - 1] = [1, {classes - 1}], got {k}")


def check_gamma(gamma: object) -> float:
    value = _real(gamma, "gamma")
    if not (math.isfinite(value) and value >= 0):
        raise ValueError(f"gamma must be finite and nonnegative, got {gamma}")
    return value


def check_positive(value: object, name: str) -> float:
    """alpha or temperature: finite and positive."""
    out = _real(value, name)
    if not (math.isfinite(out) and out > 0):
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return out


def check_label_smoothing(label_smoothing: object) -> float:
    value = _real(label_smoothing, "label_smoothing")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1], got {label_smoothing}")
    return value


def check_ignore_index(ignore_index: object) -> int | None:
    """None (no ignored label) or an int64 label."""
    if ignore_index is None:
        return None
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral):
        raise TypeError(
            f"ignore_index must be None or an integer, not {type(ignore_index).__name__}"
        )
    value = int(ignore_index)
    if not -(2**63) <= value < 2**63:
        raise ValueError("ignore_index must fit in int64")
    return value


def check_reduction(reduction: object) -> str:
    if not isinstance(reduction, str):
        raise TypeError(f"reduction must be a string, not {type(reduction).__name__}")
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be 'none', 'sum' or 'mean', got {reduction!r}")
    return reduction


def check_variant(variant: object) -> str:
    if not isinstance(variant, str):
        raise TypeError(f"variant must be a string, not {type(variant).__name__}")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be 'alpha' or 'beta', got {variant!r}")
    return variant


def check_backend(backend: object, allowed: tuple[str, ...]) -> str:
    if not isinstance(backend, str):
        raise TypeError(f"backend must be a string, not {type(backend).__name__}")
    if backend not in allowed:
        raise ValueError(f"backend must be one of {', '.join(map(repr, allowed))}, got {backend!r}")
    return backend


def check_axis(axis: object, name: str) -> int:
    """The class dimension dim (PyTorch) or axis (JAX): an integer."""
    if isinstance(axis, bool) or not isinstance(axis, Integral):
        raise TypeError(f"{name} must be an integer, not {type(axis).__name__}")
    return int(axis)


def check_precision_name(
    precision: object, dtypes: Mapping[Any, str], framework: str
) -> str | None:
    """None, "float32", "float64" or a framework dtype of dtypes (to its name)."""
    if precision is None:
        return None
    if isinstance(precision, str):
        if precision in ("float32", "float64"):
            return precision
        raise ValueError(
            f"precision must be None, 'float32' or 'float64' (or {framework}.float32/float64), "
            f"got {precision!r}"
        )
    try:
        name = dtypes.get(precision)
    except TypeError:  # unhashable
        name = None
    if name is not None:
        return name
    if any(type(precision) is type(key) for key in dtypes):
        raise ValueError(
            f"precision must be None, 'float32', 'float64', {framework}.float32 or "
            f"{framework}.float64, got {precision!r}"
        )
    raise TypeError(
        f"precision must be None, 'float32', 'float64', {framework}.float32 or "
        f"{framework}.float64, not {precision!r}"
    )
