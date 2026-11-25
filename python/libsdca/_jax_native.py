"""The CPU rows of libsdca.jax where its FFI handler (_jax_cpu.cpp) is not built: the same
kernels in the native library (sdca_loss_rows), called on the host as the handler calls them
(under vmap, one call per group of rows of its own class weights and label smoothing).

jax.experimental.buffer_callback hands the callback XLA's own buffers, and the gradient may
take the logits' buffer, as with the handler; JAX before it has jax.pure_callback, which
copies the arguments and results.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import jax
import numpy as np

from . import _native
from ._frontend import DISTILLATION, NAMES, Rows

try:
    from jax.experimental.buffer_callback import buffer_callback
except ImportError:
    buffer_callback = None

# explain_backend's route (_frontend.explanation).
ROUTE = " through jax.pure_callback" if buffer_callback is None else " through buffer_callback"
_TEACHERS = frozenset(kind for kind, name in NAMES.items() if name in DISTILLATION)
_LABELS = (np.int32, np.uint32, np.int64)


def call(
    shape: jax.ShapeDtypeStruct,
    scores: jax.Array,
    target: jax.Array,
    weight: jax.Array,
    eps: jax.Array,
    upstream: jax.Array,
    rows: Rows,
    gradient: bool,
    both: bool = False,
    inplace: bool = True,
    reduction: str = "none",
) -> Any:
    """The host equivalent of the FFI call, including its fused value and gradient."""
    args = scores, target, weight, eps, upstream
    outputs = (shape, jax.ShapeDtypeStruct(scores.shape, scores.dtype)) if both else shape
    options = {"rows": rows, "gradient": gradient, "both": both, "reduction": reduction}
    if buffer_callback is None:
        # A mapped callback receives additional leading axes. Preserve them when reducing
        # all row axes of each original example.
        axes = scores.ndim if reduction != "none" else 1
        copied = functools.partial(_copied, dtype=shape.dtype, axes=axes, **options)
        return jax.pure_callback(copied, outputs, *args, vmap_method="broadcast_all")
    in_place = functools.partial(_in_place, **options)
    aliases = {0: int(both)} if (gradient or both) and inplace else None
    run = buffer_callback(
        in_place, outputs, vmap_method="broadcast_all", input_output_aliases=aliases
    )
    return run(*args)


def _in_place(
    ctx: Any, out: Any, *args: Any, rows: Rows, gradient: bool, both: bool, reduction: str
) -> None:
    values, grads = out if both else ((None, out) if gradient else (out, None))
    _rows(
        None if values is None else np.asarray(values),
        None if grads is None else np.asarray(grads),
        *map(np.asarray, args),
        rows=rows,
        reduction=reduction,
    )


def _copied(
    *args: np.ndarray,
    dtype: Any,
    axes: int,
    rows: Rows,
    gradient: bool,
    both: bool,
    reduction: str,
) -> Any:
    """Allocate results with the mapped leading dimensions supplied by the callback."""
    scores = args[0]
    values = np.empty(scores.shape[:-axes], dtype) if both or not gradient else None
    grads = np.empty(scores.shape, scores.dtype) if gradient or both else None
    _rows(values, grads, *args, rows=rows, reduction=reduction)
    return (values, grads) if both else (grads if gradient else values)


def _role(target: np.ndarray, kind: int, n: int) -> str:
    if target.dtype == np.bool_:
        return "mask"
    if kind in _TEACHERS:
        return "teacher"
    return "labels" if target.size == n and target.dtype in _LABELS else "probabilities"


def _row_target(target: np.ndarray, kind: int, n: int) -> tuple[str, np.ndarray]:
    role = _role(target, kind, n)
    if role == "labels":
        target = np.ascontiguousarray(target, np.int64)
    elif role == "mask":
        target = np.ascontiguousarray(target).view(np.uint8)
    else:
        target = np.ascontiguousarray(target)
    return role, target


def _mapped_groups(n: int, weights: int, smoothing: int, upstream: int) -> int:
    scales = upstream if upstream < n else 0
    groups = max(weights, smoothing, scales, 1)
    if (
        any(1 < count != groups for count in (weights, smoothing, scales))
        or n % groups
        or upstream > n
    ):
        raise ValueError("libsdca: mapped weight, label_smoothing and scale must share the rows")
    return groups


def _store_values(
    out: np.ndarray, values: np.ndarray, reduction: str, terms: np.ndarray | None = None
) -> None:
    """Run the same reducer and IEEE output conversion as the compiled FFI handler."""
    mode = ("none", "sum", "mean").index(reduction)
    groups = out.size if mode else 1
    per = values.size // groups if groups else 0
    ptr = None if terms is None else terms.ctypes.data
    if mode and groups != 1:
        _native.reduce_row_groups(
            out.dtype.name, mode, groups, per, values.ctypes.data, ptr, out.ctypes.data
        )
    else:
        _native.reduce_rows(
            out.dtype.name, mode, values.size, values.ctypes.data, ptr, out.ctypes.data
        )


def _label_terms(
    target: np.ndarray, w: np.ndarray, classes: int, rows: Rows, reduction: str, role: str
) -> np.ndarray | None:
    if reduction != "mean" or role != "labels" or not (w.size or rows.ignore):
        return None
    y = target.reshape(-1)
    if w.size:
        weights = w.size // classes
        wi = np.arange(y.size) // (y.size // weights) * classes
        terms = w[wi + np.clip(y, 0, classes - 1)]
    else:
        terms = np.ones(y.size, np.float64)
    if rows.ignore:
        terms[y == rows.ignore_index] = 0
    return terms


def _group_scale(up: np.ndarray, n: int, group: int, row0: int) -> tuple[float, int | None]:
    if up.size == n:
        return 1.0, up.ctypes.data + row0 * up.itemsize
    if up.size:
        return float(up[group if up.size > 1 else 0]), None
    return 1.0, None


def _rows(
    value_out: np.ndarray | None,
    grads: np.ndarray | None,
    scores: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    eps: np.ndarray,
    upstream: np.ndarray,
    *,
    rows: Rows,
    reduction: str = "none",
) -> None:
    """Into out (C-contiguous; may be scores): the values, rounded once into its dtype, or the
    gradient times the per-row upstream factors, of class-last scores (rows: every other
    axis)."""
    if scores.size == 0:
        if value_out is not None:
            _store_values(value_out, np.empty(0, np.float64), reduction)
        return
    scores = np.ascontiguousarray(scores)
    classes = scores.shape[-1]
    n = scores.size // classes
    w = np.ascontiguousarray(weight, np.float64).reshape(-1)
    e = np.asarray(eps, np.float64).reshape(-1)
    up = np.ascontiguousarray(upstream, np.float64).reshape(-1)
    weights = w.size // classes
    groups = _mapped_groups(n, weights, e.size, up.size)
    if any(not out.flags.c_contiguous for out in (value_out, grads) if out is not None):
        raise ValueError("libsdca: the result must be C-contiguous")
    role, target = _row_target(target, rows.kind, n)
    values = None if value_out is None else np.empty(n, np.float64)
    per = n // groups
    row_bytes = classes * scores.itemsize
    target_bytes = target.itemsize * (1 if role == "labels" else classes)
    for i in range(groups):
        factor, factors = _group_scale(up, n, i, i * per)
        # A static smoothing (float64, as the FFI handler takes it) wins; e carries a traced one
        # in the logits' work type.
        traced = e.size and math.isnan(rows.label_smoothing)
        r = rows._replace(label_smoothing=float(e[i if e.size > 1 else 0])) if traced else rows
        row0 = i * per
        _native.loss_rows(
            r,
            0,
            scores.dtype.name,
            per,
            classes,
            scores.ctypes.data + row0 * row_bytes,
            target.ctypes.data + row0 * target_bytes,
            role,
            target.dtype.name if role == "probabilities" else "float64",
            None
            if weights == 0
            else w.ctypes.data + (i if weights > 1 else 0) * classes * w.itemsize,
            factor,
            factors,
            None if values is None else values.ctypes.data + row0 * values.itemsize,
            grads.ctypes.data + row0 * row_bytes if grads is not None else None,
        )
    if value_out is not None and values is not None:
        terms = _label_terms(target, w, classes, rows, reduction, role)
        _store_values(value_out, values, reduction, terms)
