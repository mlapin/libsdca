"""The CPU kernels of the libsdca::loss and libsdca::loss_ operators (_torch_ops) where the
CPU extension (_torch_cpu.cpp) is not built: the same kernels in the native library, through
sdca_loss_rows on the tensors' own buffers, with PyTorch's intra-op threads.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

from . import _native
from ._frontend import DISTILLATION, NAMES, Rows

_DTYPES = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}
_TEACHERS = frozenset(kind for kind, name in NAMES.items() if name in DISTILLATION)
_lock = threading.Lock()


def _check(
    scores: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    rows: torch.Tensor | None,
) -> None:
    """The checks of the extension's operators (_torch_cpu.cpp, check)."""
    if scores.dim() != 2 or not scores.is_contiguous() or scores.dtype not in _DTYPES:
        raise RuntimeError("libsdca: scores must be (N, C)")
    if not target.is_contiguous() or target.shape[0] != scores.shape[0]:
        raise RuntimeError("libsdca: target must be contiguous with a row per score row")
    if weight is not None and not (
        weight.dtype == torch.float64
        and weight.is_contiguous()
        and weight.numel() in (0, scores.shape[1])
    ):
        raise RuntimeError("libsdca: weight must be float64 (C)")
    if rows is not None and not (
        rows.dtype == torch.float64 and rows.is_contiguous() and rows.numel() == scores.shape[0]
    ):
        raise RuntimeError("libsdca: row_upstream must be float64 (N)")


def _role(target: torch.Tensor, kind: int) -> str:
    if target.dtype == torch.int64:
        return "labels"
    if target.dtype == torch.bool:
        return "mask"
    return "teacher" if kind in _TEACHERS else "probabilities"


def _evaluate(
    scores: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    row_upstream: torch.Tensor | None,
    rows: Rows,
    upstream: float,
    values: torch.Tensor,
    grads: torch.Tensor | None,
) -> None:
    n, classes = scores.shape
    role = _role(target, rows.kind)
    probabilities = _DTYPES.get(target.dtype, "float64") if role == "probabilities" else "float64"
    _native.loss_rows(
        rows,
        torch.get_num_threads(),
        _DTYPES[scores.dtype],
        n,
        classes,
        scores.data_ptr(),
        target.data_ptr(),
        role,
        probabilities,
        weight.data_ptr() if weight is not None and weight.numel() else None,
        upstream,
        None if row_upstream is None else row_upstream.data_ptr(),
        values.data_ptr() if values.numel() else None,
        None if grads is None else grads.data_ptr(),
    )


def _loss(
    scores: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    row_upstream: torch.Tensor | None,
    *args: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check(scores, target, weight, row_upstream)
    *options, upstream, values, grad = args
    out = scores.new_empty((scores.shape[0] if values else 0,), dtype=torch.float64)
    g = torch.empty_like(scores) if grad else scores.new_empty((0,))
    grads = g if grad and scores.numel() else None
    _evaluate(scores, target, weight, row_upstream, Rows(*options), upstream, out, grads)
    return out, g


def _loss_(
    scores: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    row_upstream: torch.Tensor | None,
    *args: Any,
) -> torch.Tensor:
    _check(scores, target, weight, row_upstream)
    *options, upstream, values, grad = args
    if not grad:
        raise RuntimeError("libsdca: loss_ writes the gradient")
    out = scores.new_empty((scores.shape[0] if values else 0,), dtype=torch.float64)
    grads = scores if scores.numel() else None
    _evaluate(scores, target, weight, row_upstream, Rows(*options), upstream, out, grads)
    return out


def _reduce_rows(
    values: torch.Tensor, terms: torch.Tensor | None, dtype: torch.dtype, reduction: int
) -> torch.Tensor:
    out = values.new_empty(values.shape[:-1], dtype=dtype)
    ptr = None if terms is None else terms.data_ptr()
    if values.ndim == 2:
        _native.reduce_row_groups(
            _DTYPES[dtype],
            reduction,
            values.shape[0],
            values.shape[1],
            values.data_ptr(),
            ptr,
            out.data_ptr(),
        )
    else:
        _native.reduce_rows(
            _DTYPES[dtype], reduction, values.numel(), values.data_ptr(), ptr, out.data_ptr()
        )
    return out


def register(library: torch.library.Library) -> None:
    """The CPU kernels of the operators of library (libsdca), once."""
    with _lock:
        if not torch._C._dispatch_has_kernel_for_dispatch_key("libsdca::loss", "CPU"):
            library.impl("reduce_rows", _reduce_rows, "CPU")
            library.impl("loss", _loss, "CPU")
            library.impl("loss_", _loss_, "CPU")
