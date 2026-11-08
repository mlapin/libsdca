"""CPU losses: the extension, native and tensor kernels behind one autograd Function, and one
torch.library operator pair that keeps them opaque (no graph breaks) under torch.compile.

A kernel computes (loss, saved) in forward and the gradient from saved in backward. Stats
kernels save row statistics and take the upstream derivative in their backward; gradient
kernels save the first-order gradient (float32 for 16-bit logits) and scale it. Every
caller resolves its kernel, implementation and options on the host first: the operators
receive them as (kind, ints, floats).
"""

from __future__ import annotations

import functools
import math
import os
import struct
import threading
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from . import _core, _cpu_entropy
from ._first_order import guard

REDUCTIONS = {"none": 0, "mean": 1, "sum": 2}
_REDUCTION_NAMES = tuple(REDUCTIONS)
# The objectives of the generic kernel (the tensor kinds of _core._KINDS).
OBJECTIVES = (
    "cross_entropy",
    "topk_cross_entropy",
    "topk_hinge_alpha",
    "topk_hinge_beta",
    "topk_hinge_alpha_smooth",
    "topk_hinge_beta_smooth",
    "multilabel_hinge",
    "multilabel_hinge_smooth",
    "proximal_cross_entropy",
)
PRECISIONS = (None, "float32", "float64")
# Implementations of the kernels.
TENSOR, NATIVE, EXTENSION = 0, 1, 2

# Storage codes of the native tensor entry points.
DTYPES = {
    torch.float32: _core.DTYPE_FLOAT32,
    torch.float64: _core.DTYPE_FLOAT64,
    torch.float16: _core.DTYPE_FLOAT16,
    torch.bfloat16: _core.DTYPE_BFLOAT16,
}
# The torch layer's messages for the native library's.
_MESSAGES = {
    "Logits must be finite": "logits must be finite floating-point values",
    "Labels must lie in [0, num_classes)": "Target index out of range",
    "Every row needs a positive label": "Every example needs a positive label",
    "This loss takes one positive label per row": "This loss needs exactly one target",
    "This loss needs a negative label per row": "This loss needs a negative label",
}


def native(function: Callable[..., None], *args: Any, **options: Any) -> None:
    """A native tensor entry point, with the torch layer's messages."""
    try:
        function(*args, num_threads=threads(), **options)
    except ValueError as error:
        message = _MESSAGES.get(str(error))
        if message is None:
            raise
        raise ValueError(message) from None


Ints = Sequence[int]
Floats = Sequence[float]


@dataclass(frozen=True)
class Kernel:
    """forward(logits, target, extra, ints, floats, need) -> (loss, saved; empty unless need),
    backward(logits, target, extra, saved, upstream, ints, floats, inplace) -> gradient
    (inplace: saved is not read again), fake(logits, target, extra, ints, floats) -> fake
    (loss, saved), and higher(logits, target, extra, upstream, ints, floats): a gradient
    that create_graph differentiates (None: first order)."""

    forward: Callable[..., tuple[Tensor, Tensor]]
    backward: Callable[..., Tensor]
    fake: Callable[..., tuple[Tensor, Tensor]]
    higher: Callable[..., Tensor] | None = None


KERNELS: dict[str, Kernel] = {}


def threads() -> int | None:
    """The native thread count: SDCA_NUM_THREADS if set, else PyTorch's intra-op threads."""
    return None if "SDCA_NUM_THREADS" in os.environ else torch.get_num_threads()


def value_dtype(logits: Tensor) -> torch.dtype:
    """promote(logits dtype, float32): the dtype of every loss value."""
    return torch.promote_types(logits.dtype, torch.float32)


def saved_dtype(logits: Tensor) -> torch.dtype:
    """The dtype of a saved first-order gradient: float32 for 16-bit logits."""
    return torch.promote_types(logits.dtype, torch.float32)


def keep_graph() -> bool:
    """Whether the running backward keeps its graph (retain_graph, create_graph or unknown)."""
    autograd = getattr(torch._C, "_autograd", None)
    keep = getattr(autograd, "_get_current_graph_task_keep_graph", None)
    return True if keep is None else bool(keep())


def scaled(gradient: Tensor, upstream: Tensor, dtype: torch.dtype, inplace: bool) -> Tensor:
    """gradient * upstream[:, None] rounded once into dtype.

    16-bit gradients are saved in float32, so the upstream factor (65536 under an fp16
    GradScaler) multiplies before the only rounding. The upstream is cast to the gradient
    dtype first: no logits-sized float64 product. inplace reuses the saved gradient.
    """
    scale = upstream.to(gradient.dtype)
    scale = scale[:, None] if scale.ndim == 1 else scale
    product = gradient.mul_(scale) if inplace else gradient * scale
    return product.to(dtype)


def _values(logits: Tensor, reduction: int, dtype: torch.dtype) -> Tensor:
    """A fake loss: per row (reduction none) or reduced."""
    shape = (logits.shape[0],) if reduction == 0 else ()
    return logits.new_empty(shape, dtype=dtype)


def _rows(logits: Tensor, slots: int) -> Tensor:
    """Fake row statistics (n, slots) in float64."""
    return logits.new_empty((logits.shape[0], slots), dtype=torch.float64)


# Gradient kernels: forward returns per-example values in float64 (callers reduce, then
# round once to the value dtype) and the gradient in saved_dtype.


def _gradient_backward(
    logits: Tensor,
    target: Tensor,
    extra: Tensor | None,
    saved: Tensor,
    upstream: Tensor,
    ints: Ints,
    floats: Floats,
    inplace: bool,
) -> Tensor:
    return scaled(saved, upstream, logits.dtype, inplace)


def _gradient_fake(
    logits: Tensor, target: Tensor, extra: Tensor | None, ints: Ints, floats: Floats
) -> tuple[Tensor, Tensor]:
    values = logits.new_empty((logits.shape[0],), dtype=torch.float64)
    return values, torch.empty_like(logits, dtype=saved_dtype(logits))


def _rows_and_gradient(values: Tensor, gradient: Tensor, logits: Tensor) -> tuple[Tensor, Tensor]:
    return values.to(torch.float64), gradient.to(saved_dtype(logits))


# A gradient kernel's call without gradient runs on row blocks of the largest of 1/_BLOCKS
# of the rows, _BLOCK_BYTES of gradient and _THREAD_ROWS rows per thread of the kernel.
_BLOCKS, _BLOCK_BYTES, _THREAD_ROWS = 4, 32 << 20, 4

Rows = Callable[[Tensor, Tensor, Tensor | None], tuple[Tensor, Tensor]]


def native_threads() -> int:
    """The threads a native call may take: threads(), else the count of SDCA_NUM_THREADS (0
    or not a count: automatic, capped by the OpenMP runtime's count, PyTorch's threads)."""
    count = threads()
    if count is None:
        text = os.environ["SDCA_NUM_THREADS"]
        count = int(text) if text.isascii() and text.isdigit() and int(text) < 4096 else 0
    return count or torch.get_num_threads()


def _values_only(
    forward: Rows, logits: Tensor, target: Tensor, parallel: int
) -> tuple[Tensor, Tensor]:
    """The values of forward(logits, target, out) -> (values, gradient) on row blocks (see
    _BLOCKS) for a kernel of parallel threads, one block if that covers every row; forward
    writes into out, one buffer of a block that every block reuses, where it can. Nothing saved."""
    n, classes = logits.shape
    width = classes * (8 if logits.dtype == torch.float64 else 4)
    rows = max(-(-n // _BLOCKS), _BLOCK_BYTES // width, _THREAD_ROWS * parallel)
    if rows >= n:
        return forward(logits, target, None)[0], _no_stats(logits)
    values = torch.empty(n, dtype=torch.float64)
    out = torch.empty((rows, classes), dtype=saved_dtype(logits))
    for begin in range(0, n, rows):
        end = min(n, begin + rows)
        values[begin:end] = forward(logits[begin:end], target[begin:end], out[: end - begin])[0]
    return values, _no_stats(logits)


def _loss_forward(
    logits: Tensor,
    target: Tensor,
    extra: Tensor | None,
    ints: Ints,
    floats: Floats,
    need: bool,
) -> tuple[Tensor, Tensor]:
    """A loss of _core (ints: objective, k, implementation, precision; floats: gamma, alpha)
    of int64 labels or a positive mask: the native library, or the extension's
    proximal_cross_entropy."""
    from . import torch as _torch

    objective, k, implementation = OBJECTIVES[ints[0]], ints[1], ints[2]
    precision = PRECISIONS[ints[3]]
    gamma, alpha = floats
    if implementation == NATIVE:

        def forward(z: Tensor, t: Tensor, out: Tensor | None) -> tuple[Tensor, Tensor]:
            return _native_loss(z, t, objective, k, gamma, alpha, precision, out)

    else:
        # Of all rows: blocks of a call without gradient take the same arithmetic.
        fp32 = precision == "float32" and _torch._float32_holds(logits, alpha)
        operators = _cpu_entropy.operators()

        def forward(z: Tensor, t: Tensor, out: Tensor | None) -> tuple[Tensor, Tensor]:
            values, grad = operators.prox_entropy_forward(z, t, alpha, fp32)
            return _rows_and_gradient(values, grad, z)

    if need:
        return forward(logits, target, None)
    parallel = native_threads() if implementation == NATIVE else torch.get_num_threads()
    return _values_only(forward, logits, target, parallel)


def _native_loss(
    logits: Tensor,
    target: Tensor,
    objective: str,
    k: int,
    gamma: float,
    alpha: float,
    precision: str | None,
    out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Values (float64) and gradients (saved dtype, into out if given) of int64 labels or a
    positive mask by the native library, in float arithmetic for 16-bit logits or
    precision="float32"."""
    from ._core import _loss_of_kind

    n, classes = logits.shape
    name, nonsmooth = _loss_of_kind(objective)
    values = torch.empty(n, dtype=torch.float64)
    gradient = torch.empty(logits.shape, dtype=saved_dtype(logits)) if out is None else out
    labels = target.data_ptr() if target.dtype == torch.int64 else None
    positive = None if labels is not None else target.data_ptr()
    native(
        _core._loss_gradient_rows,
        logits.data_ptr(),
        DTYPES[logits.dtype],
        n,
        classes,
        labels,
        positive,
        values.data_ptr(),
        gradient.data_ptr(),
        loss=name,
        k=k,
        gamma=gamma if nonsmooth is None else nonsmooth,
        alpha=alpha,
        fp32=precision == "float32" and logits.dtype != torch.float64,
    )
    return values, gradient


KERNELS["loss"] = Kernel(_loss_forward, _gradient_backward, _gradient_fake)


def _native_distillation(
    function: Callable[..., None], logits: Tensor, teacher: Tensor, *args: Any, **options: Any
) -> None:
    """A native distillation entry point on contiguous student and teacher logits."""
    try:
        function(
            logits.data_ptr(),
            teacher.data_ptr(),
            DTYPES[logits.dtype],
            *logits.shape,
            *args,
            num_threads=threads(),
            **options,
        )
    except ValueError as error:
        if str(error) != "Logits must be finite":
            raise
        raise ValueError("Logits must be finite floating-point tensors") from None


# The distillation kernels reduce their rows (ints[3]: a REDUCTIONS code) into the value dtype
# themselves: one autograd node per call, and a scalar upstream derivative for mean and sum.


def _reduced(values: Tensor, reduction: int, logits: Tensor) -> Tensor:
    """float64 row values reduced as REDUCTIONS code reduction, rounded once into the value
    dtype."""
    if reduction == REDUCTIONS["mean"]:
        values = values.mean()
    elif reduction == REDUCTIONS["sum"]:
        values = values.sum()
    return values.to(value_dtype(logits))


def _row_upstream(upstream: Tensor, n: int, reduction: int) -> Tensor:
    """The float64 upstream derivative of each row (a scalar for mean and sum)."""
    scale = upstream.to(torch.float64)
    return scale / n if reduction == REDUCTIONS["mean"] else scale


def _distillation_fake(statistics: bool) -> Callable[..., tuple[Tensor, Tensor]]:
    """The fake of a distillation kernel: the reduced loss and the saved gradient (or, with
    statistics, the row statistics)."""

    def fake(
        logits: Tensor, target: Tensor, extra: Tensor | None, ints: Ints, floats: Floats
    ) -> tuple[Tensor, Tensor]:
        values = _values(logits, ints[3], value_dtype(logits))
        if statistics:
            return values, _rows(logits, _DISTILLATION_STATS)
        return values, torch.empty_like(logits, dtype=saved_dtype(logits))

    return fake


def _distillation_forward(
    logits: Tensor,
    target: Tensor,
    extra: Tensor | None,
    ints: Ints,
    floats: Floats,
    need: bool,
) -> tuple[Tensor, Tensor]:
    """proximal_distillation_loss (ints: implementation, double, proximal, reduction; floats:
    temperature, alpha) of logits and the teacher's logits (target), in float64 arithmetic."""
    temperature, alpha = floats
    if ints[0] == EXTENSION:
        operators = _cpu_entropy.operators()

        def forward(z: Tensor, t: Tensor, out: Tensor | None) -> tuple[Tensor, Tensor]:
            values, grad = operators.prox_distillation_forward(z, t, temperature, alpha)
            return _rows_and_gradient(values, grad, z)

    else:

        def forward(z: Tensor, t: Tensor, out: Tensor | None) -> tuple[Tensor, Tensor]:
            values = torch.empty(z.shape[0], dtype=torch.float64)
            gradient = torch.empty(z.shape, dtype=saved_dtype(z)) if out is None else out
            _native_distillation(
                _core._proximal_distillation_rows,
                z,
                t,
                temperature,
                alpha,
                values.data_ptr(),
                gradient.data_ptr(),
            )
            return values, gradient

    if need:
        values, gradient = forward(logits, target, None)
    else:
        parallel = torch.get_num_threads() if ints[0] == EXTENSION else native_threads()
        values, gradient = _values_only(forward, logits, target, parallel)
    return _reduced(values, ints[3], logits), gradient


def _distillation_backward(
    logits: Tensor,
    target: Tensor,
    extra: Tensor | None,
    saved: Tensor,
    upstream: Tensor,
    ints: Ints,
    floats: Floats,
    inplace: bool,
) -> Tensor:
    rows = _row_upstream(upstream, logits.shape[0], ints[3])
    return scaled(saved, rows, logits.dtype, inplace)


_DISTILLATION_STATS = 7
KERNELS["distillation"] = Kernel(
    _distillation_forward, _distillation_backward, _distillation_fake(False)
)


def _distillation_float(logits: Tensor, double: bool) -> bool:
    """Whether native distillation takes float exponentials (not precision="float64")."""
    return not double and logits.dtype != torch.float64


def _distillation_stats_forward(
    logits: Tensor, target: Tensor, extra: Tensor | None, ints: Ints, floats: Floats, need: bool
) -> tuple[Tensor, Tensor]:
    """distillation_loss (ints: implementation, double, proximal, reduction; floats:
    temperature, alpha) by the extension or the native library, saving row statistics."""
    implementation, double = ints[0], bool(ints[1])
    if implementation == EXTENSION:
        values, stats = _cpu_entropy.operators().distillation_forward(
            logits, target, floats[0], double, need
        )
        loss = _reduced(values.to(torch.float64), ints[3], logits)
        return loss, _no_stats(logits) if stats is None else stats
    n = logits.shape[0]
    values = torch.empty(n, dtype=torch.float64)
    stats = torch.empty((n, _DISTILLATION_STATS), dtype=torch.float64) if need else None
    _native_distillation(
        _core._distillation_forward,
        logits,
        target,
        floats[0],
        values.data_ptr(),
        None if stats is None else stats.data_ptr(),
        fp32=_distillation_float(logits, double),
    )
    return _reduced(values, ints[3], logits), _no_stats(logits) if stats is None else stats


def _distillation_stats_backward(
    logits: Tensor,
    target: Tensor,
    extra: Tensor | None,
    saved: Tensor,
    upstream: Tensor,
    ints: Ints,
    floats: Floats,
    inplace: bool,
) -> Tensor:
    implementation, double = ints[0], bool(ints[1])
    n = logits.shape[0]
    scale = _row_upstream(upstream, n, ints[3])
    if implementation == EXTENSION:
        return _cpu_entropy.operators().distillation_backward(
            logits, target, saved, scale.expand(n), floats[0], double
        )
    if scale.ndim and scale.stride(0) != 0:
        scale = scale.contiguous()
    gradient = torch.empty_like(logits, memory_format=torch.contiguous_format)
    _native_distillation(
        _core._distillation_backward,
        logits,
        target,
        floats[0],
        saved.data_ptr(),
        scale.data_ptr(),
        1 if scale.ndim and scale.stride(0) != 0 else 0,
        gradient.data_ptr(),
        fp32=_distillation_float(logits, double),
    )
    return gradient


KERNELS["distillation_stats"] = Kernel(
    _distillation_stats_forward, _distillation_stats_backward, _distillation_fake(True)
)


def _distillation_grad_forward(
    logits: Tensor, target: Tensor, extra: Tensor | None, ints: Ints, floats: Floats, need: bool
) -> tuple[Tensor, Tensor]:
    """distillation_loss by the extension, saving its rows' gradient times the temperature in
    the logits' dtype (float32 or float64): its exponentials are the forward's, where the
    statistics kernel recomputes them backward."""
    values, saved = _cpu_entropy.operators().distillation_forward_grad(logits, target, floats[0])
    return _reduced(values, ints[3], logits), saved


def _distillation_grad_backward(
    logits: Tensor,
    target: Tensor,
    extra: Tensor | None,
    saved: Tensor,
    upstream: Tensor,
    ints: Ints,
    floats: Floats,
    inplace: bool,
) -> Tensor:
    """The saved gradient times upstream / T, the factor rounded once into its dtype (or, where
    that overflows, over T first)."""
    temperature, n = floats[0], logits.shape[0]
    if upstream.numel() == 1:
        # One factor: a Python float (as _row_upstream's float64), rounded once into the saved dtype.
        scalar = float(upstream) / (n if ints[3] == REDUCTIONS["mean"] else 1) / temperature
        if abs(scalar) <= torch.finfo(saved.dtype).max:
            product = saved.mul_(scalar) if inplace else saved * scalar
            return product.to(logits.dtype)
    rows = _row_upstream(upstream, n, ints[3])
    factor = rows / temperature
    if bool(torch.isfinite(factor.to(saved.dtype)).all()):
        return scaled(saved, factor, logits.dtype, inplace)
    inverse = torch.tensor(1 / temperature, dtype=saved.dtype)
    over = saved.mul_(inverse) if inplace else saved * inverse
    return scaled(over, rows, logits.dtype, True)


def distillation_grad(logits: Tensor, temperature: float, double: bool) -> bool:
    """Whether a differentiable distillation_loss call saves its gradient (the extension's
    distillation_forward_grad): float64 logits, or float32 logits in float32 work, at a
    temperature whose reciprocal their dtype holds (backward multiplies by it). That gradient is
    the backward's own buffer; 16-bit logits keep the statistics kernel, whose backward rounds
    once without a float32 buffer twice their size."""
    if logits.dtype != torch.float64 and (logits.dtype != torch.float32 or double):
        return False
    tiny = torch.finfo(logits.dtype).tiny
    return tiny <= temperature <= 1 / tiny


KERNELS["distillation_grad"] = Kernel(
    _distillation_grad_forward, _distillation_grad_backward, _distillation_fake(False)
)


# Integer-target cross-entropy: the extension or the native library.

# The ignore_index of calls without one: every other label is validated.
NO_IGNORE = -(2**63)
# Row statistics of the native entropy (float64 per row; the extension saves 3 n + 1).
_NATIVE_STATS = 5


# The widest float64 rows of the extension's integer entropy (entropy64_forward).
DOUBLE_CLASSES = 32


def entropy_implementation(logits: Tensor, extension: bool, classes: int) -> int:
    """The implementation of integer entropy: the extension where it is available, except
    for float64 rows of more than DOUBLE_CLASSES classes, which the native library takes.
    float64 rows keep the native library's statistics and gradients either way, whose tails
    and weights beyond double's range take a log-domain tensor path."""
    wide = logits.dtype == torch.float64 and classes > DOUBLE_CLASSES
    return EXTENSION if extension and not wide else NATIVE


def _no_stats(logits: Tensor) -> Tensor:
    return logits.new_empty((0,), dtype=torch.float64)


def _entropy_tensor_value(
    logits: Tensor, target: Tensor, weight: Tensor | None, reduction: int, ignore: int
) -> Tensor:
    """The loss by float64 tensor operations, in log domain where it leaves double's range."""
    from . import _cross_entropy, _entropy

    name = _REDUCTION_NAMES[reduction]
    ignored = target == ignore
    labels = target.masked_fill(ignored, 0)
    stats = _entropy._statistics(logits, labels)
    if weight is None and ignore == NO_IGNORE:
        value = _entropy._values(stats, name)
    else:
        rows = _row_weights(weight, labels, ignored)
        value = _cross_entropy._weighted_value(stats, rows, name)
    return value.to(value_dtype(logits))


def _row_weights(weight: Tensor | None, labels: Tensor, ignored: Tensor) -> Tensor:
    """float64 weights of the rows: weight[label], 0 where ignored."""
    rows = (
        torch.ones(labels.shape, dtype=torch.float64) if weight is None else weight.double()[labels]
    )
    return rows.masked_fill(ignored, 0)


def _entropy_forward(
    logits: Tensor, target: Tensor, extra: Tensor | None, ints: Ints, floats: Floats, need: bool
) -> tuple[Tensor, Tensor]:
    """Cross-entropy of int64 labels (ints: reduction, ignore_index, fp32, implementation)
    with class weights extra (or None), as torch.nn.functional.cross_entropy."""
    reduction, ignore, fp32, implementation = ints
    if implementation == EXTENSION and logits.dtype != torch.float64:
        loss, stats, represented = _cpu_entropy.operators().forward(
            logits, target, reduction, ignore, extra, bool(fp32), need
        )
        if not represented:
            loss = _entropy_tensor_value(logits, target, extra, reduction, ignore)
        return loss, _no_stats(logits) if stats is None else stats
    return _native_entropy(logits, target, extra, ints, need)


def _entropy_backward(
    logits: Tensor,
    target: Tensor,
    extra: Tensor | None,
    saved: Tensor,
    upstream: Tensor,
    ints: Ints,
    floats: Floats,
    inplace: bool,
) -> Tensor:
    reduction, ignore, fp32, implementation = ints
    if implementation == EXTENSION and logits.dtype != torch.float64:
        return _cpu_entropy.operators().backward(
            logits, target, saved, upstream, reduction, ignore, extra, bool(fp32)
        )
    n, classes = logits.shape
    stats, totals = saved[: n * _NATIVE_STATS], saved[n * _NATIVE_STATS :]
    scale = upstream.to(torch.float64)
    if reduction == REDUCTIONS["mean"]:
        scale = scale / totals[1]
    if logits.dtype == torch.float64 and _beyond_double(stats, totals, scale, extra):
        # Derivatives below double's normal range, scaled up (by the upstream derivative or
        # a class weight), or weight sums beyond it: the log-domain tensor gradient.
        with torch.no_grad():
            return _entropy_higher(logits, target, extra, upstream, ints, floats)
    if implementation == EXTENSION:
        return _cpu_entropy.operators().entropy64_backward(
            logits, target, saved, upstream, reduction, ignore, extra
        )
    scale = scale.contiguous()
    gradient = torch.empty_like(logits, memory_format=torch.contiguous_format)
    weights = _weights(extra)
    native(
        _core._cross_entropy_backward,
        logits.data_ptr(),
        DTYPES[logits.dtype],
        n,
        classes,
        target.data_ptr(),
        ignore,
        _address(weights),
        stats.data_ptr(),
        scale.data_ptr(),
        1 if scale.ndim else 0,
        gradient.data_ptr(),
        fp32=bool(fp32),
    )
    return gradient


_weight_cache: dict[int, tuple[Tensor, Tensor, int]] = {}
_weight_lock = threading.Lock()


def _weights(weight: Tensor | None) -> Tensor | None:
    """Float64 class weights, cached until the next other weight. Callers hold the result
    over the native call: another thread's weight can evict it while that runs."""
    if weight is None:
        return None
    with _weight_lock:
        cached = _weight_cache.get(id(weight))
        if cached is None or cached[0] is not weight or weight._version != cached[2]:
            _weight_cache.clear()
            cached = (weight, weight.to(torch.float64).contiguous(), weight._version)
            _weight_cache[id(weight)] = cached
        return cached[1]


def _address(tensor: Tensor | None) -> int | None:
    return None if tensor is None else tensor.data_ptr()


def _beyond_double(stats: Tensor, totals: Tensor, scale: Tensor, extra: Tensor | None) -> bool:
    """Whether a float64 gradient needs the log-domain tensor path: derivatives below
    double's normal range scaled up, or weights and their sums beyond its range."""
    # NumPy views: a few small-tensor operators would cost more than the kernel at 256 rows.
    largest = float(np.abs(scale.detach().numpy()).max(initial=0.0))
    if extra is not None:
        largest *= float(extra.abs().amax())
    tiny = bool(np.any(stats.detach().numpy()[_NATIVE_STATS - 1 :: _NATIVE_STATS] != 0))
    finite = bool(np.isfinite(totals.detach().numpy()).all())
    return (largest > 1 and tiny) or not finite or not math.isfinite(largest)


def _native_entropy(
    logits: Tensor, target: Tensor, weight: Tensor | None, ints: Ints, need: bool
) -> tuple[Tensor, Tensor]:
    """The loss and saved row statistics (with the weight totals) by the native library, or
    of float64 rows by the extension's entropy64_forward (the same statistics)."""
    reduction, ignore, fp32, implementation = ints
    n, classes = logits.shape
    if implementation == EXTENSION:
        values, saved = _cpu_entropy.operators().entropy64_forward(
            logits, target, ignore, weight, need
        )
        totals = saved[-2:]
    else:
        saved = torch.empty(n * _NATIVE_STATS + 2 if need else 2, dtype=torch.float64)
        totals = saved[-2:]
        values = torch.empty(n, dtype=torch.float64)
        weights = _weights(weight)
        native(
            _core._cross_entropy_forward,
            logits.data_ptr(),
            DTYPES[logits.dtype],
            n,
            classes,
            target.data_ptr(),
            ignore,
            _address(weights),
            values.data_ptr(),
            saved.data_ptr() if need else None,
            totals.data_ptr(),
            fp32=bool(fp32),
        )
    if logits.dtype == torch.float64:
        tiny = torch.finfo(torch.float64).tiny
        rows = values.numpy()
        small = rows < tiny
        rare = not bool(np.isfinite(rows).all())
        if not rare and small.any():
            rare = bool((small & (target.numpy() != ignore)).any())
        if reduction != REDUCTIONS["none"]:
            rare = rare or not math.isfinite(float(totals[0]))
        if rare:
            # Losses (or their sum) below double's normal range or beyond it: the log-domain
            # tensor value.
            loss = _entropy_tensor_value(logits, target, weight, reduction, ignore)
            return loss, saved if need else _no_stats(logits)
    if reduction == REDUCTIONS["none"]:
        loss = values
    elif reduction == REDUCTIONS["sum"]:
        loss = totals[0]
    else:
        loss = totals[0] / totals[1]
    return loss.to(value_dtype(logits)), saved if need else _no_stats(logits)


def _entropy_fake(
    logits: Tensor, target: Tensor, extra: Tensor | None, ints: Ints, floats: Floats
) -> tuple[Tensor, Tensor]:
    n = logits.shape[0]
    stats = ints[3] == EXTENSION and logits.dtype != torch.float64
    size = 3 * n + 1 if stats else n * _NATIVE_STATS + 2
    saved = logits.new_empty((size,), dtype=torch.float64)
    return _values(logits, ints[0], value_dtype(logits)), saved


def _entropy_higher(
    logits: Tensor,
    target: Tensor,
    extra: Tensor | None,
    upstream: Tensor,
    ints: Ints,
    floats: Floats,
) -> Tensor:
    """The differentiable float64 tensor gradient (create_graph)."""
    from . import _cross_entropy, _entropy

    reduction, ignore = ints[0], ints[1]
    name = _REDUCTION_NAMES[reduction]
    ignored = target == ignore
    labels = target.masked_fill(ignored, 0)
    stats = _entropy._statistics(logits, labels)
    if extra is None and ignore == NO_IGNORE:
        return _entropy._gradient(logits, labels, stats, upstream, name)
    rows = _row_weights(extra, labels, ignored)
    grad = _cross_entropy._weighted_gradient(logits, labels, stats, rows, upstream, name)
    if reduction == REDUCTIONS["mean"]:
        undefined = ~ignored & ~(rows > 0).any()
        grad = grad.masked_fill(undefined[:, None], torch.nan)
    return grad


KERNELS["entropy"] = Kernel(_entropy_forward, _entropy_backward, _entropy_fake, _entropy_higher)


# Stats kernels of the extension (k = 1 hinges, top-k, boolean masks).


def _hinge_forward(
    logits: Tensor, target: Tensor, extra: Tensor | None, ints: Ints, floats: Floats, need: bool
) -> tuple[Tensor, Tensor]:
    """k = 1 hinges (ints: reduction, smooth; floats: gamma)."""
    return _cpu_entropy.operators().hinge_forward(logits, target, ints[0], bool(ints[1]), floats[0])


def _hinge_backward(
    logits: Tensor,
    target: Tensor,
    extra: Tensor | None,
    saved: Tensor,
    upstream: Tensor,
    ints: Ints,
    floats: Floats,
    inplace: bool,
) -> Tensor:
    return _cpu_entropy.operators().hinge_backward(
        logits, target, saved, upstream, ints[0], bool(ints[1]), floats[0]
    )


def _stats_fake(slots: int) -> Callable[..., tuple[Tensor, Tensor]]:
    def fake(
        logits: Tensor, target: Tensor, extra: Tensor | None, ints: Ints, floats: Floats
    ) -> tuple[Tensor, Tensor]:
        return _values(logits, ints[0], value_dtype(logits)), _rows(logits, slots)

    return fake


KERNELS["hinge"] = Kernel(_hinge_forward, _hinge_backward, _stats_fake(2))


def _topk_forward(
    logits: Tensor, target: Tensor, extra: Tensor | None, ints: Ints, floats: Floats, need: bool
) -> tuple[Tensor, Tensor]:
    """1 < k top-k losses (ints: reduction, loss code, k, fp32; floats: gamma)."""
    return _cpu_entropy.operators().topk_forward(
        logits, target, ints[0], ints[1], ints[2], floats[0], bool(ints[3])
    )


def _topk_backward(
    logits: Tensor,
    target: Tensor,
    extra: Tensor | None,
    saved: Tensor,
    upstream: Tensor,
    ints: Ints,
    floats: Floats,
    inplace: bool,
) -> Tensor:
    return _cpu_entropy.operators().topk_backward(
        logits, target, saved, upstream, ints[0], ints[1], ints[2], floats[0], bool(ints[3])
    )


KERNELS["topk"] = Kernel(_topk_forward, _topk_backward, _stats_fake(3))


def _multilabel_forward(
    logits: Tensor, target: Tensor, extra: Tensor | None, ints: Ints, floats: Floats, need: bool
) -> tuple[Tensor, Tensor]:
    """Boolean-mask entropy and hinges (ints: reduction, objective code, float32 exponentials;
    floats: gamma)."""
    mask = target.view(torch.uint8)
    return _cpu_entropy.operators().multilabel_forward(
        logits, mask, ints[0], ints[1], floats[0], bool(ints[2]), need
    )


def _multilabel_backward(
    logits: Tensor,
    target: Tensor,
    extra: Tensor | None,
    saved: Tensor,
    upstream: Tensor,
    ints: Ints,
    floats: Floats,
    inplace: bool,
) -> Tensor:
    mask = target.view(torch.uint8)
    return _cpu_entropy.operators().multilabel_backward(
        logits, mask, saved, upstream, ints[0], ints[1], floats[0]
    )


def _multilabel_higher(
    logits: Tensor,
    target: Tensor,
    extra: Tensor | None,
    upstream: Tensor,
    ints: Ints,
    floats: Floats,
) -> Tensor:
    """The differentiable float64 tensor gradient of boolean-mask entropy (objective 0)."""
    from ._multilabel import _EntropySaved
    from .torch import _TensorEntropy

    rows = upstream.expand(logits.shape[0])
    if ints[0] == REDUCTIONS["mean"]:
        rows = rows / logits.shape[0]
    saved = _EntropySaved(saved_tensors=(logits, target), double=True)
    return _TensorEntropy.backward(saved, rows)[0]


def _multilabel_kernel() -> Kernel:
    # Only entropy (objective 0) has higher derivatives; _higher checks the objective.
    return Kernel(_multilabel_forward, _multilabel_backward, _stats_fake(7), _multilabel_higher)


KERNELS["multilabel"] = _multilabel_kernel()
# Objectives of KERNELS["multilabel"] with higher derivatives.
_HIGHER_MULTILABEL = 0


def _has_higher(kind: str, ints: Ints) -> bool:
    kernel = KERNELS[kind]
    if kernel.higher is None:
        return False
    return kind != "multilabel" or ints[1] == _HIGHER_MULTILABEL


def gradient(
    kind: str,
    logits: Tensor,
    target: Tensor,
    extra: Tensor | None,
    saved: Tensor,
    upstream: Tensor,
    ints: Ints,
    floats: Floats,
    inplace: bool,
) -> Tensor:
    """The gradient of a kernel's loss; under create_graph differentiable, or first order."""
    kernel = KERNELS[kind]
    if not torch.is_grad_enabled():
        return kernel.backward(logits, target, extra, saved, upstream, ints, floats, inplace)
    if _has_higher(kind, ints):
        assert kernel.higher is not None
        return kernel.higher(logits, target, extra, upstream, ints, floats)
    with torch.no_grad():
        first = kernel.backward(logits, target, extra, saved, upstream, ints, floats, False)
    return guard(first, logits, upstream)


class _Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        kind: str,
        logits: Tensor,
        target: Tensor,
        extra: Tensor | None,
        ints: list[int],
        floats: list[float],
    ) -> Tensor:
        loss, saved = KERNELS[kind].forward(logits, target, extra, ints, floats, True)
        ctx.save_for_backward(logits, target, extra, saved)
        ctx.kind, ctx.ints, ctx.floats = kind, ints, floats
        return loss

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor | None, ...]:
        (upstream,) = grad_outputs
        logits, target, extra, saved = ctx.saved_tensors
        inplace = not keep_graph()
        grad = gradient(
            ctx.kind, logits, target, extra, saved, upstream, ctx.ints, ctx.floats, inplace
        )
        return None, grad, None, None, None, None


# The kinds whose extension calls take their autograd in C++ (no Python between forward and
# backward), by their code in the extension's apply.
APPLIED = ("entropy", "hinge", "topk", "multilabel", "distillation_stats", "distillation_grad")
_CODES = {kind: code for code, kind in enumerate(APPLIED)}


def _applies(kind: str, ints: Ints) -> bool:
    """Whether a kernel call runs in the extension with its autograd in C++."""
    if kind == "entropy":
        return ints[3] == EXTENSION
    if kind == "distillation_stats":
        return ints[0] == EXTENSION
    return kind in _CODES


@functools.cache
def _applied() -> Callable[..., Tensor]:
    """The extension's apply; a create_graph backward calls gradient."""
    operators = _cpu_entropy.operators()

    def higher(
        code: int,
        logits: Tensor,
        target: Tensor,
        extra: Tensor | None,
        saved: Tensor,
        upstream: Tensor,
        ints: list[int],
        floats: list[float],
    ) -> Tensor:
        return gradient(APPLIED[code], logits, target, extra, saved, upstream, ints, floats, False)

    operators.set_gradient(higher, _entropy_tensor_value)
    return typing.cast("Callable[..., Tensor]", operators.apply)


def apply(
    kind: str, logits: Tensor, target: Tensor, extra: Tensor | None, ints: Ints, floats: Floats
) -> Tensor:
    """The loss of an extension kernel of APPLIED, differentiable in logits; its autograd in
    C++."""
    return _applied()(_CODES[kind], logits, target, extra, ints, floats)


_is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", lambda: False)
# torch.library.custom_op with register_fake and register_autograd (PyTorch 2.4).
AVAILABLE = hasattr(torch.library, "custom_op") and hasattr(torch.library, "register_autograd")


# A constant of the traced call: older Dynamo (PyTorch 2.6) breaks the graph at struct.pack.
@_cpu_entropy._constant
def _bits(floats: Sequence[float]) -> list[int]:
    """The float64 bit patterns of floats: the operators take int lists, since a float
    list argument equal to the upstream derivative of a sum (1.0) miscompiles under
    AOTAutograd (torch 2.15)."""
    return list(struct.unpack(f"<{len(floats)}q", struct.pack(f"<{len(floats)}d", *floats)))


def _floats(bits: Sequence[int]) -> list[float]:
    return list(struct.unpack(f"<{len(bits)}d", struct.pack(f"<{len(bits)}q", *bits)))


def loss(
    kind: str,
    logits: Tensor,
    target: Tensor,
    extra: Tensor | None,
    ints: list[int],
    floats: list[float],
) -> Tensor:
    """The loss of a CPU kernel; differentiable in logits."""
    if AVAILABLE and _is_compiling():
        return torch.ops.libsdca.cpu_loss(kind, logits, target, extra, ints, _bits(floats))[0]
    if _applies(kind, ints):
        return apply(kind, logits, target, extra, ints, floats)
    if torch.is_grad_enabled() and logits.requires_grad:
        return _Function.apply(kind, logits, target, extra, ints, floats)
    return KERNELS[kind].forward(logits, target, extra, ints, floats, False)[0]


if AVAILABLE:

    @torch.library.custom_op("libsdca::cpu_loss", mutates_args=(), device_types="cpu")
    def _cpu_loss(
        kind: str,
        logits: Tensor,
        target: Tensor,
        extra: typing.Optional[Tensor],  # noqa: UP007 - the schema needs typing.Optional
        ints: typing.List[int],  # noqa: UP006 - and typing.List before PyTorch 2.6
        bits: typing.List[int],  # noqa: UP006
    ) -> tuple[Tensor, Tensor]:
        return KERNELS[kind].forward(logits, target, extra, ints, _floats(bits), True)

    @torch.library.custom_op("libsdca::cpu_loss_backward", mutates_args=(), device_types="cpu")
    def _cpu_loss_backward(
        kind: str,
        logits: Tensor,
        target: Tensor,
        extra: typing.Optional[Tensor],  # noqa: UP007
        saved: Tensor,
        upstream: Tensor,
        ints: typing.List[int],  # noqa: UP006
        bits: typing.List[int],  # noqa: UP006
    ) -> Tensor:
        floats = _floats(bits)
        return KERNELS[kind].backward(logits, target, extra, saved, upstream, ints, floats, False)

    @_cpu_loss.register_fake
    def _(
        kind: str,
        logits: Tensor,
        target: Tensor,
        extra: Tensor | None,
        ints: list[int],
        bits: list[int],
    ) -> tuple[Tensor, Tensor]:
        return KERNELS[kind].fake(logits, target, extra, ints, _floats(bits))

    @_cpu_loss_backward.register_fake
    def _(
        kind: str,
        logits: Tensor,
        target: Tensor,
        extra: Tensor | None,
        saved: Tensor,
        upstream: Tensor,
        ints: list[int],
        bits: list[int],
    ) -> Tensor:
        return torch.empty_like(logits)

    def _setup(ctx: Any, inputs: tuple[Any, ...], output: tuple[Tensor, Tensor]) -> None:
        kind, logits, target, extra, ints, bits = inputs
        ctx.save_for_backward(logits, target, extra, output[1])
        ctx.mark_non_differentiable(output[1])
        ctx.set_materialize_grads(False)
        ctx.kind, ctx.ints, ctx.bits = kind, ints, bits

    def _rule(ctx: Any, upstream: Tensor | None, _: Any) -> tuple[Tensor | None, ...]:
        if upstream is None:
            return (None,) * 6
        logits, target, extra, saved = ctx.saved_tensors
        if torch.is_grad_enabled():
            floats = _floats(ctx.bits)
            grad = gradient(
                ctx.kind, logits, target, extra, saved, upstream, ctx.ints, floats, False
            )
        else:
            grad = torch.ops.libsdca.cpu_loss_backward(
                ctx.kind, logits, target, extra, saved, upstream, ctx.ints, ctx.bits
            )
        return None, grad, None, None, None, None

    _cpu_loss.register_autograd(_rule, setup_context=_setup)


def reduce(values: Tensor, reduction: str) -> Tensor:
    """Per-example values reduced: none, sum or mean."""
    if reduction == "none":
        return values
    return values.sum() if reduction == "sum" else values.mean()
