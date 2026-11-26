"""PyTorch losses: one function and one module per loss (cross_entropy and CrossEntropyLoss,
topk_cross_entropy, the hinges, the multilabel losses, proximal_cross_entropy, the
distillation losses and linear_cross_entropy).

backend="auto" (and its synonym "kernel") runs libsdca's kernels: the CPU kernels built on
first use (_torch_cpu.cpp; without them the same kernels of the native library,
_torch_native) or the Triton kernels on CUDA (triton/launch_torch.py), through the
libsdca::loss operator (_torch_ops), or raises NotImplementedError naming
backend="composed", which runs the composed tensor operations (_composed) on any device.
Invalid rows (NaN or +inf logits, invalid class weights, rows of -inf
only) give NaN losses and gradients, without an error or a host synchronization; a masked
label or positive class gives +inf (docs/API.md, Masked classes).
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, ClassVar, Literal, NamedTuple

import torch
from torch import nn

from . import _frontend as _f
from . import _torch_ops as _ops
from ._torch_compat import is_compiling

__all__ = [
    "CrossEntropyLoss",
    "DistillationLoss",
    "LinearCrossEntropyLoss",
    "MulticlassHingeLoss",
    "MultilabelCrossEntropyLoss",
    "MultilabelHingeLoss",
    "ProximalCrossEntropyLoss",
    "ProximalDistillationLoss",
    "TopKCrossEntropyLoss",
    "TopKHingeLoss",
    "cross_entropy",
    "distillation_loss",
    "explain_backend",
    "linear_cross_entropy",
    "multiclass_hinge_loss",
    "multilabel_cross_entropy",
    "multilabel_hinge_loss",
    "proximal_cross_entropy",
    "proximal_distillation_loss",
    "topk_cross_entropy",
    "topk_hinge_loss",
]

Reduction = Literal["none", "sum", "mean"]
Backend = Literal["auto", "kernel", "composed"]
TORCH = _f.Framework(
    "dim", "dimension", {torch.float32: "float32", torch.float64: "float64"}, "torch"
)


_KINDS: dict[torch.dtype, str] = {}


def kind_of(t: torch.Tensor) -> str:
    """The dtype kind of a tensor for _frontend's checks."""
    dtype = t.dtype
    # Compiled graphs would guard on the cache's contents (and recompile once it fills).
    compiling = is_compiling()
    kind = None if compiling else _KINDS.get(dtype)
    if kind is None:
        complex_kind = "c" if dtype.is_complex else "i"
        kind = "b" if dtype == torch.bool else "f" if dtype.is_floating_point else complex_kind
        if not compiling:
            _KINDS[dtype] = kind
    return kind


def full(*tensors: Any) -> tuple[Any, ...]:
    """DTensor inputs gathered in full on every rank (differentiably); others unchanged."""
    return tuple(t.full_tensor() if type(t).__name__ == "DTensor" else t for t in tensors)


def _tensor(t: object, name: str) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, not {type(t).__name__}")
    return t


def autocast_enabled(device: str) -> bool:
    try:
        return torch.is_autocast_enabled(device)
    except TypeError:  # PyTorch 2.1 predates the device argument
        return torch.is_autocast_cpu_enabled() if device == "cpu" else torch.is_autocast_enabled()


class Prepared(NamedTuple):
    """A checked call: class-last scores and targets, the class weights (float64, NaN where
    invalid) and the shape of the per-example losses."""

    z: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor | None
    shape: torch.Size


def prepare(
    call: _f.Call,
    logits: object,
    target: object,
    weight: object,
    axis: int,
    label_smoothing: float = 0.0,
) -> Prepared:
    """Check the tensors of a call; TypeError for wrong types, ValueError for bad shapes."""
    name = call.name
    distillation = name in _f.DISTILLATION
    logits = _tensor(logits, "student" if distillation else "logits")
    target = _tensor(target, "teacher" if distillation else "target")
    _f.check_logits(call, kind_of(logits), logits.dtype)
    ndim = logits.ndim
    axis = _f.class_axis(TORCH, ndim, axis)
    last = axis == ndim - 1
    if target.device != logits.device:
        raise device_error("target", "logits")
    role = _f.target_role(
        TORCH, call, kind_of(target), target.dtype, target.shape, logits.shape, axis
    )
    if role == "labels":
        target = target if target.dtype == torch.int64 else target.long()
    else:
        if role == "teacher" and target.dtype != logits.dtype:
            common = torch.promote_types(logits.dtype, target.dtype)
            logits, target = logits.to(common), target.to(common)
        elif role == "probabilities" and _f.probability_dtype(target.dtype, logits.dtype):
            target = target.float()
        target = target if last else target.movedim(axis, -1)
    z = logits if last else logits.movedim(axis, -1)
    _f.check_classes(call.k, z.shape[-1])
    _f.check_precision(logits.dtype, call.precision)
    w = None
    if weight is not None:
        weight = _tensor(weight, "weight")
        _f.check_weight(kind_of(weight), weight.dtype, weight.shape, z.shape[-1])
        if weight.device != logits.device:
            raise device_error("weight", "logits")
        w = _ops.nan_weights(weight)
    return Prepared(z, target, w, z.shape[:-1])


def _flat(t: torch.Tensor, classes: int, labels: bool, scores: bool = False) -> torch.Tensor:
    if t.ndim != (1 if labels else 2):
        t = t.reshape(-1) if labels else t.reshape(-1, classes)
    return _rows(t) if scores else t.contiguous()


def _rows(scores: torch.Tensor) -> torch.Tensor:
    """(N, C) scores for the row kernels: CUDA rows of unit column stride as they are (a
    slice x[:, a:b] of wider rows; launch_torch copies them only for kernels without row
    strides), else contiguous."""
    if scores.is_cuda and scores.dim() == 2 and scores.stride(-1) == 1:
        if scores.stride(0) >= scores.shape[1]:
            return scores
    return scores.contiguous()


def _why_not(p: Prepared, call: _f.Call) -> str | None:
    """None where a kernel computes a prepared call, else why not."""
    return _ops.kernels(p.z.device, p.z.dtype, p.shape.numel(), p.z.shape[-1], call.kind, call.k)


def explained(p: Prepared, call: _f.Call, label_smoothing: float = 0.0) -> tuple[str, str]:
    """(backend, reason) of a prepared call; NotImplementedError where no kernel computes it."""
    if not call.composed:
        reason = _why_not(p, call)
        if reason is not None:
            raise _f.no_kernel(reason)
    work = _f.work(p.z.dtype, call.precision)
    return _f.explanation(
        call.composed,
        p.z.device.type,
        work,
        "composed tensor operations",
        fallback=_ops.fallback(p.z.device) or _ops.launcher(p.z.device),
    )


def run(call: _f.Call, p: Prepared, label_smoothing: float = 0.0) -> torch.Tensor:
    """The loss of a prepared call."""
    # Read the tuple field directly so Dynamo can specialize this Python branch.
    if call.backend == "composed":
        from ._composed import loss

        return loss(call.name, p.z, p.target, weight=p.weight, **call.options(label_smoothing))
    if p.z.is_cuda and _ops.is_compiling():
        # The kernel specializes on the class width. Batch dimensions can vary.
        torch._dynamo.mark_static(p.z, -1)
    reason = _why_not(p, call)
    if reason is not None:
        raise _f.no_kernel(reason)
    z = p.z
    classes = z.shape[-1]
    if p.shape.numel() == 0:
        dtype = _ops.value_dtype(z.dtype, _ops.target_dtype(p.target))
        return _reduce(z.sum(-1).to(dtype), call.reduction)
    labels = p.target.dtype == torch.int64
    scores = _flat(z, classes, False, scores=True)
    rows = call.rows(label_smoothing)
    target = _flat(p.target, classes, labels)
    reduction = _ops.REDUCTIONS[call.reduction]
    # Probability targets and teachers that need a gradient: its backward reads the logits,
    # which the extension must then not overwrite.
    target_grad = torch.is_grad_enabled() and target.requires_grad
    extension = _ops.eager()
    if extension is not None:
        # The extension divides a mean by the label weights or the rows itself.
        reuse = call.reuse_logits and not target_grad
        value = extension.apply(scores, target, p.weight, None, *rows, reduction, reuse)
    else:
        _ops.warn_launcher(scores)
        grad = torch.is_grad_enabled() and z.requires_grad
        if grad and scores.is_cuda and not _ops._transformed():
            value = _ops.CudaLoss.apply(scores, target, p.weight, rows, reduction)
        else:
            reuse = grad and call.reuse_logits and _ops.compiled_cpu_labels(z, target, rows)
            # The target has a separate pullback below. A fixed score tensor needs
            # no kernel backward even when the target requires a derivative.
            kernel_target = target.detach() if target_grad else target
            value, _ = _ops.Loss.apply(
                scores, kernel_target, p.weight, rows, reduction, grad, reuse, 0
            )
    if target_grad:
        from ._composed import xp_torch

        mean = call.reduction == "mean"
        scale = 1.0 / _f.mean_divisor(xp_torch, target, p.weight, None, False) if mean else 1.0
        function, operands = _target_gradient(call, rows, label_smoothing, scores, target, p.weight)
        attach = xp_torch.target_gradient
        if _native_teacher(call, extension, scores, target):
            from ._torch_teacher import attach

        value = attach(value, target, function, scale, *operands)
    return value.reshape(p.shape) if call.reduction == "none" else value


def _native_teacher(
    call: _f.Call, extension: Any, scores: torch.Tensor, target: torch.Tensor
) -> bool:
    # For temperatures in the finite, nonzero FP32 range the complete VJP
    # factor and products of finite FP32 inputs fit safely in binary64.
    return (
        extension is not None
        and call.name == "distillation"
        and scores.device.type == "cpu"
        and scores.dtype == target.dtype == torch.float32
        and call.precision != "float64"
        and 2.0**-149 <= call.temperature <= torch.finfo(torch.float32).max
        and not (scores.is_neg() or target.is_neg())
    )


def _target_gradient(
    call: _f.Call,
    rows: _f.Rows,
    label_smoothing: float,
    scores: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
) -> tuple[Any, tuple[Any, ...]]:
    """The gradient function (_composed._targets) and operands of a target's cotangent."""
    from ._composed import _targets

    logits = scores.detach()
    if call.precision == "float64":
        logits = logits.double()
    if call.name == "cross_entropy":
        return _targets.probability_gradient, (logits, weight, label_smoothing)
    teacher = target.detach().to(logits.dtype)
    if call.name == "distillation":
        return _targets.teacher_gradient, (logits, teacher, call.temperature)
    return _proximal_teacher, (logits, teacher, call.temperature, call.gamma, rows)


def _proximal_teacher(
    xp: Any,
    logits: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
    gamma: float,
    rows: _f.Rows,
) -> torch.Tensor:
    """_targets.teacher_gradient of proximal_distillation, of the kernels' student gradient
    per unit of loss, in the work dtype."""
    from ._composed import _targets

    z, t = xp.work(logits), xp.work(teacher)
    student = torch.ops.libsdca.loss(z, t, None, None, *rows, 1.0, False, True)[1]
    return _targets.teacher_gradient(xp, z, t, temperature, student, gamma)


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return loss
    return loss.sum() if reduction == "sum" else loss.mean()


# The checked extension arguments for repeated, already class-last matrices. Only tensor
# metadata and loss options are retained; the extension receives the current inputs.
_MATRIX_ARGS: dict[tuple[Any, ...], tuple[Any, ...]] = {}


class _MatrixCall(NamedTuple):
    extension: Any
    apply: Any


_CPU_MATRICES: dict[tuple[_f.Call, float], _MatrixCall] = {}


def _remember_matrix(
    key: tuple[Any, ...],
    call: _f.Call,
    p: Prepared,
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
) -> None:
    if (
        len(_MATRIX_ARGS) >= 512
        or p.z is not logits
        or p.target is not target
        or not p.shape.numel()
    ):
        return
    extension = _ops.eager()
    if extension is None:
        return
    args = (*call.rows(label_smoothing), _ops.REDUCTIONS[call.reduction], call.reuse_logits)
    if not torch.overrides.has_torch_function((logits, target)) and logits.is_cpu:
        bind = getattr(extension, "bind_matrix", None)
        if bind is None:
            return
        options = (call, label_smoothing)
        saved = _CPU_MATRICES.get(options)
        if saved is None or saved.extension is not extension:
            saved = _MatrixCall(extension, bind(*args))
            _CPU_MATRICES[options] = saved
        saved.apply(logits, target, remember=True)
    _MATRIX_ARGS[key] = args


def _loss(
    call: _f.Call, logits: Any, target: Any, weight: Any = None, label_smoothing: float = 0.0
) -> torch.Tensor:
    plain = type(logits) is torch.Tensor and type(target) is torch.Tensor
    key = None
    if not (plain and (weight is None or type(weight) is torch.Tensor)):
        logits, target, weight = full(logits, target, weight)
    elif weight is None and not is_compiling() and not call.composed:
        cached = _CPU_MATRICES.get((call, label_smoothing))
        if (
            cached is not None
            and not torch.overrides.has_torch_function((logits, target))
            and logits.is_cpu
            and cached.extension is _ops.eager()
        ):
            result = cached.apply(logits, target)
            if result is not None:
                return result
        if logits.ndim == 2 and not target.requires_grad:
            key = (
                call,
                label_smoothing,
                logits.shape,
                logits.dtype,
                logits.device,
                target.shape,
                target.dtype,
                target.device,
            )
            args = _MATRIX_ARGS.get(key)
            if args is not None:
                extension = _ops.eager()
                if extension is not None:
                    return extension.apply(_rows(logits), target.contiguous(), None, None, *args)
    p = prepare(call, logits, target, weight, call.axis, label_smoothing)
    value = run(call, p, label_smoothing)
    if key is not None:
        _remember_matrix(key, call, p, logits, target, label_smoothing)
    return value


def _bind(name: str, *, dim: object = -1, **options: object) -> _f.Call:
    memo = not is_compiling()
    return _f.bind(TORCH, name, axis=dim, memo=memo, **options)  # type: ignore[arg-type]


def _class_dim(logits: object) -> int:
    return 0 if isinstance(logits, torch.Tensor) and logits.ndim == 1 else 1


def _cross_entropy_call(
    logits: object,
    weight: object,
    ignore_index: object,
    reduction: object,
    label_smoothing: object,
    backend: object,
    precision: object,
    reuse_logits: object = True,
) -> tuple[_f.Call, float]:
    dim = _class_dim(logits)
    # The options already checked; types tell apart equal values of which one is invalid.
    key: tuple[object, ...] | None = (
        ignore_index,
        type(ignore_index),
        reduction,
        label_smoothing,
        type(label_smoothing),
        backend,
        precision,
        reuse_logits,
        type(reuse_logits),
        dim,
    )
    try:
        cached = _CE_CALLS.get(key)  # type: ignore[arg-type]
    except TypeError:  # an unhashable option: checked below
        cached, key = None, None
    if cached is not None:
        if weight is not None:
            _check_ce_weight(weight, logits)
        return cached
    _f.check_ignore_index(ignore_index, optional=False)
    eps = _f.check_label_smoothing(label_smoothing)
    if weight is not None:
        _check_ce_weight(weight, logits)
    call = _bind(
        "cross_entropy",
        ignore_index=ignore_index,
        reduction=reduction,
        dim=dim,
        backend=backend,
        precision=precision,
        reuse_logits=reuse_logits,
    )
    if key is not None and len(_CE_CALLS) < 4096:
        _CE_CALLS[key] = (call, eps)
    return call, eps


_CE_CALLS: dict[tuple[object, ...], tuple[_f.Call, float]] = {}


def _check_ce_weight(weight: object, logits: object) -> None:
    weight = _tensor(weight, "weight")
    if weight.requires_grad:
        raise ValueError("weight must not require gradients")
    if isinstance(logits, torch.Tensor):
        _check_ce_weight_dtype(weight, logits)


def _check_ce_weight_dtype(weight: torch.Tensor, logits: torch.Tensor) -> None:
    """Class weights of the logits' dtype (float32 for 16-bit logits under autocast)."""
    autocast = (
        logits.dtype in _ops.HALF
        and weight.dtype == torch.float32
        and autocast_enabled(logits.device.type)
    )
    if weight.is_floating_point() and weight.dtype != logits.dtype and not autocast:
        raise TypeError(f"weight must have the logits' dtype {logits.dtype}, not {weight.dtype}")


@_f.loss_function
def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    backend: str = "auto",
    precision: object = None,
    reuse_logits: bool = True,
) -> torch.Tensor:
    """Cross-entropy of (C,), (N, C) or (N, C, ...) logits, as
    torch.nn.functional.cross_entropy (classes on dim 1).

    target holds class labels (any integer dtype) of the logits' shape without the class
    dimension, or class probabilities of the logits' shape (floating point). weight,
    ignore_index (labels only), reduction and label_smoothing in [0, 1] follow PyTorch,
    including the weighted mean. 16-bit inputs return float32 losses and 16-bit gradients.

    backend: "auto" or "kernel" (libsdca's kernels; NotImplementedError where none computes
    the call) or "composed" (tensor operations). precision: None, "float32" or "float64" (or
    torch.float32, torch.float64). reuse_logits: as in topk_cross_entropy.

    Raises:
        TypeError: For arguments of the wrong type (a boolean target).
        ValueError: For invalid shapes or options. Invalid values (NaN or +inf logits, rows
            of -inf only, invalid class weights or target
            probabilities) give NaN losses and gradients.
        IndexError: For a label outside the classes in synchronous CPU calls.
        NotImplementedError: For backend "auto" or "kernel" where no kernel computes the call.
    """
    call, eps = _cross_entropy_call(
        input, weight, ignore_index, reduction, label_smoothing, backend, precision, reuse_logits
    )
    return _loss(call, input, target, weight, eps)


@_f.loss_function
def topk_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    k: int,
    weight: torch.Tensor | None = None,
    ignore_index: int | None = None,
    reduction: str = "mean",
    dim: int | None = None,
    backend: str = "auto",
    precision: object = None,
    reuse_logits: bool = True,
) -> torch.Tensor:
    """Top-k cross-entropy (the top-k entropy loss of Lapin et al., 2016); cross-entropy at
    k = 1. k is an integer in [1, classes - 1].
    """
    call = _bind(
        "topk_cross_entropy",
        k=k,
        ignore_index=ignore_index,
        reduction=reduction,
        dim=_class_dim(logits) if dim is None else dim,
        backend=backend,
        precision=precision,
        reuse_logits=reuse_logits,
    )
    return _loss(call, logits, target, weight)


@_f.loss_function
def multiclass_hinge_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 0.0,
    weight: torch.Tensor | None = None,
    ignore_index: int | None = None,
    reduction: str = "mean",
    dim: int | None = None,
    backend: str = "auto",
    precision: object = None,
    reuse_logits: bool = True,
) -> torch.Tensor:
    """Multiclass hinge loss of Crammer and Singer (2001), smoothed by gamma > 0 (finite);
    hinge values at kinks take deterministic subgradients.
    """
    call = _bind(
        "multiclass_hinge",
        gamma=gamma,
        ignore_index=ignore_index,
        reduction=reduction,
        dim=_class_dim(logits) if dim is None else dim,
        backend=backend,
        precision=precision,
        reuse_logits=reuse_logits,
    )
    return _loss(call, logits, target, weight)


@_f.loss_function
def topk_hinge_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    k: int,
    variant: Literal["alpha", "beta"] = "beta",
    gamma: float = 0.0,
    weight: torch.Tensor | None = None,
    ignore_index: int | None = None,
    reduction: str = "mean",
    dim: int | None = None,
    backend: str = "auto",
    precision: object = None,
    reuse_logits: bool = True,
) -> torch.Tensor:
    """Top-k hinge loss (Lapin et al., 2015), smoothed by gamma > 0 (finite). Variant "beta"
    averages the k largest hinges; "alpha" is the hinge of the average of the k largest
    margins. k is an integer in [1, classes - 1].
    """
    call = _bind(
        "topk_hinge",
        k=k,
        variant=variant,
        gamma=gamma,
        ignore_index=ignore_index,
        reduction=reduction,
        dim=_class_dim(logits) if dim is None else dim,
        backend=backend,
        precision=precision,
        reuse_logits=reuse_logits,
    )
    return _loss(call, logits, target, weight)


@_f.loss_function
def proximal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 1.0,
    weight: torch.Tensor | None = None,
    ignore_index: int | None = None,
    reduction: str = "mean",
    dim: int | None = None,
    backend: str = "auto",
    precision: object = None,
    reuse_logits: bool = True,
) -> torch.Tensor:
    """Moreau envelope of the cross-entropy with step gamma (finite and positive)."""
    call = _bind(
        "proximal_cross_entropy",
        gamma=gamma,
        ignore_index=ignore_index,
        reduction=reduction,
        dim=_class_dim(logits) if dim is None else dim,
        backend=backend,
        precision=precision,
        reuse_logits=reuse_logits,
    )
    return _loss(call, logits, target, weight)


@_f.loss_function
def multilabel_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    dim: int | None = None,
    backend: str = "auto",
    precision: object = None,
    reuse_logits: bool = True,
) -> torch.Tensor:
    """Multilabel cross-entropy: uniform target probabilities on the positive labels."""
    call = _bind(
        "multilabel_cross_entropy",
        reduction=reduction,
        dim=_class_dim(logits) if dim is None else dim,
        backend=backend,
        precision=precision,
        reuse_logits=reuse_logits,
    )
    return _loss(call, logits, target)


@_f.loss_function
def multilabel_hinge_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 0.0,
    reduction: str = "mean",
    dim: int | None = None,
    backend: str = "auto",
    precision: object = None,
    reuse_logits: bool = True,
) -> torch.Tensor:
    """Multilabel hinge loss, smoothed by gamma > 0 (finite)."""
    call = _bind(
        "multilabel_hinge",
        gamma=gamma,
        reduction=reduction,
        dim=_class_dim(logits) if dim is None else dim,
        backend=backend,
        precision=precision,
        reuse_logits=reuse_logits,
    )
    return _loss(call, logits, target)


@_f.loss_function
def distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    temperature: float = 1.0,
    reduction: str = "mean",
    dim: int | None = None,
    backend: str = "auto",
    precision: object = None,
    reuse_logits: bool = True,
) -> torch.Tensor:
    """Cross-entropy with a fixed teacher distribution softmax(teacher / T).

    Both arguments are LOGITS. No T² multiplier is applied. The student's gradient is exactly
    zero at equal represented logits. A teacher that requires gradients gets PyTorch's, that
    of cross_entropy(student / T, softmax(teacher / T)): p (-log q - loss) / T per unit of
    loss, with p and q the teacher's and the student's softmax at T.

    Args:
        student: Floating-point logits with the class dimension dim (at least two classes).
        teacher: Logits of the same shape and device; logits of another floating-point
            dtype are promoted with the student's (torch.promote_types).
        temperature: T, finite and positive.
        reduction: "none", "sum" or "mean".
        dim: Class dimension of the logits (default 1, or 0 for 1-D logits, as in
            torch.nn.functional.cross_entropy).
        backend: "auto" or "kernel" (libsdca's kernels; NotImplementedError where none
            computes the call) or "composed" (tensor operations).
        precision: None, "float32" or "float64" (or torch.float32, torch.float64).
        reuse_logits: As in topk_cross_entropy (for the student logits).

    Returns:
        Per-example losses (student.shape without dim) or their reduction.

    Raises:
        TypeError: For arguments of the wrong type.
        ValueError: For invalid shapes or options. A logit of -inf masks a class where the
            teacher's is -inf too; -inf in the teacher alone is teacher probability zero,
            in the student alone the loss +inf (docs/API.md, Masked classes); invalid rows
            give NaN.
    """
    call = _bind(
        "distillation",
        temperature=temperature,
        reduction=reduction,
        dim=_class_dim(student) if dim is None else dim,
        backend=backend,
        precision=precision,
        reuse_logits=reuse_logits,
    )
    return _loss(call, student, teacher)


@_f.loss_function
def proximal_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    temperature: float = 1.0,
    gamma: float = 1.0,
    reduction: str = "mean",
    dim: int | None = None,
    backend: str = "auto",
    precision: object = None,
    reuse_logits: bool = True,
) -> torch.Tensor:
    """Moreau envelope of distillation cross-entropy with step gamma and temperature T.

    Computes min_u { H(softmax(teacher/T), softmax(u)) + |u - student/T|^2 / (2 gamma) }.
    As gamma -> 0+, this converges to distillation_loss(student, teacher, temperature=T).
    gamma is finite and positive; other arguments, the result and errors as in
    distillation_loss.
    """
    call = _bind(
        "proximal_distillation",
        temperature=temperature,
        gamma=gamma,
        reduction=reduction,
        dim=_class_dim(student) if dim is None else dim,
        backend=backend,
        precision=precision,
        reuse_logits=reuse_logits,
    )
    return _loss(call, student, teacher)


_SINGLE_LABEL_DOC = """
    Args:
        logits: Floating-point scores with the class dimension dim (at least k + 1 classes;
            no examples give the torch.nn.functional results: empty, a zero sum or a NaN
            mean).
        target: Integer class labels of logits.shape without dim (any integer dtype), on
            the device of logits.
        weight: None or class weights (classes,), finite and nonnegative: each example's
            loss is multiplied by the weight of its label, and the mean divides by the sum
            of those weights (as torch.nn.functional.cross_entropy).
        ignore_index: None or a label whose examples contribute nothing (zero loss and
            gradient; the mean leaves them out).
        reduction: "none", "sum" or "mean".
        dim: Class dimension of logits (default 1, or 0 for 1-D logits, as in
            torch.nn.functional.cross_entropy).
        backend: "auto" or "kernel" (libsdca's kernels: the CPU kernels or the Triton
            kernels on CUDA; NotImplementedError where none computes the call) or
            "composed" (tensor operations, any device).
        precision: None (float32 work for 16-bit logits, else their dtype's), "float32" or
            "float64" (or torch.float32, torch.float64).
        reuse_logits: Whether the kernels' backward writes the gradient over logits computed
            by the autograd graph (never a leaf) where nothing else references them. Where
            something does, it allocates a new gradient and warns once per process. False
            never reuses them and never warns.

    Returns:
        Per-example losses (logits.shape without dim) or their reduction, in
        promote_types(logits.dtype, float32).

    Raises:
        TypeError: For arguments of the wrong type (a boolean or floating-point target).
        ValueError: For invalid shapes or options. Invalid logit rows or class weights give NaN losses.
        IndexError: For a label outside the classes in synchronous CPU calls.
        NotImplementedError: For backend "auto" or "kernel" where no kernel computes the call.
"""

_MULTILABEL_DOC = """
    Args:
        logits: Floating-point scores with the class dimension dim (at least two classes).
        target: Boolean mask of logits.shape: the positive labels of each example (at
            least one, and for multilabel_hinge_loss at least one negative).
        reduction, dim, backend, precision, reuse_logits: As in topk_cross_entropy.

    Returns:
        Per-example losses (logits.shape without dim) or their reduction.

    Raises:
        TypeError: For arguments of the wrong type (a target that is not boolean).
        ValueError: For invalid shapes or options. Rows without the labels they need, or
            with nonfinite logits, give NaN.
"""

for _function in (
    topk_cross_entropy,
    multiclass_hinge_loss,
    topk_hinge_loss,
    proximal_cross_entropy,
):
    _function.__doc__ = (_function.__doc__ or "") + _SINGLE_LABEL_DOC
for _function in (multilabel_cross_entropy, multilabel_hinge_loss):
    _function.__doc__ = (_function.__doc__ or "") + _MULTILABEL_DOC


def _explain_cross_entropy(
    input: Any,
    target: Any,
    weight: Any = None,
    ignore_index: object = -100,
    reduction: object = "mean",
    label_smoothing: object = 0.0,
    backend: object = "auto",
    precision: object = None,
    reuse_logits: object = True,
) -> tuple[str, str]:
    call, eps = _cross_entropy_call(
        input, weight, ignore_index, reduction, label_smoothing, backend, precision, reuse_logits
    )
    input, target, weight = full(input, target, weight)
    return explained(prepare(call, input, target, weight, call.axis, eps), call, eps)


def _explainer(name: str) -> Callable[..., tuple[str, str]]:
    """explain_backend of a loss function other than cross_entropy, of its bound arguments."""

    def explain(**arguments: Any) -> tuple[str, str]:
        logits, target = (arguments.pop(key) for key in list(arguments)[:2])
        weight = arguments.pop("weight", None)
        if arguments["dim"] is None:
            arguments["dim"] = _class_dim(logits)
        call = _bind(name, **arguments)
        logits, target, weight = full(logits, target, weight)
        return explained(prepare(call, logits, target, weight, call.axis), call)

    return explain


def explain_backend(
    loss: Callable[..., torch.Tensor], /, *args: Any, **kwargs: Any
) -> tuple[str, str]:
    """(backend, reason): the path that loss(*args, **kwargs) takes, without computing it.

    loss is a function of libsdca.torch (topk_hinge_loss, cross_entropy, ...) or one of its
    loss modules (TopKHingeLoss(k=2), ...), called with these arguments. The backend is the
    value of the backend option that takes this path: "kernel" (libsdca's kernels; the
    reason says which) or "composed" (tensor operations). Like the first such call, it may
    build the CPU kernels.

    Raises:
        TypeError: If loss is not a libsdca.torch loss, or for arguments the call rejects.
        ValueError: For arguments the call rejects.
        NotImplementedError: Where backend "auto" or "kernel" has no kernel for the call.
    """
    if isinstance(loss, nn.Module):
        explain = getattr(loss, "_explain", None)
        if explain is None:
            raise TypeError(f"{type(loss).__name__} is not a libsdca.torch loss module")
        return explain(*args, **kwargs)
    explain = _EXPLAINERS.get(loss) if callable(loss) else None
    if explain is None:
        raise TypeError(f"{loss!r} is not a libsdca.torch loss function")
    signature = inspect.signature(loss)
    _f.check_keywords(_f.loss_name(loss), kwargs, signature.parameters)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return explain(**bound.arguments)


class _Loss(nn.Module):
    """A loss function as a module: its options after the tensors (checked here as each call
    checks them), with class weights as a buffer moved by .to() and saved in state_dict()."""

    _function: Callable[..., torch.Tensor]
    _signature: ClassVar[inspect.Signature]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        _f.check_keywords(_f.loss_name(self._function), kwargs, self._signature.parameters)
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        options = dict(bound.arguments)
        name = self._weight_name()
        if name in options:
            weight = options.pop(name)
            if weight is not None:
                _check_module_weight(weight)
            self.register_buffer(name, weight)
        for key, value in options.items():
            setattr(self, key, value)
        self._check(options)

    @classmethod
    def _weight_name(cls) -> str:
        return "weight"

    def _check(self, options: dict[str, Any]) -> None:
        if options.get("dim", 1) is None:
            options["dim"] = 1
        _bind(_f.loss_name(self._function), **options)

    def _options(self) -> dict[str, Any]:
        names = list(self._signature.parameters)
        return {key: getattr(self, key) for key in names}

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        """The loss of the tensors with the stored options (the call checks their values; their
        names are the signature's, so the unchecked function takes them)."""
        cls = type(self)
        names, weight = _module_layout(cls)
        attributes = self.__dict__
        options = {key: attributes[key] for key in names}
        if weight is not None:
            options[weight] = self._buffers[weight]
        return _unchecked(cls._function)(*tensors, **options)

    def _explain(self, *tensors: torch.Tensor) -> tuple[str, str]:
        return explain_backend(type(self)._function, *tensors, **self._options())

    def extra_repr(self) -> str:
        options = self._options()
        options.pop(self._weight_name(), None)
        return ", ".join(f"{key}={value!r}" for key, value in options.items())


@functools.cache
def _module_layout(cls: type[_Loss]) -> tuple[tuple[str, ...], str | None]:
    """The option names a module class stores as attributes, and its class weight buffer's."""
    name = cls._weight_name()
    names = tuple(cls._signature.parameters)
    return tuple(key for key in names if key != name), name if name in names else None


def _unchecked(function: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """A loss function without the keyword check of _frontend.loss_function."""
    return getattr(function, "__wrapped__", function)


def _check_module_weight(weight: object) -> None:
    """A module's class weights: None or a 1-D floating-point tensor (its length is checked
    with the logits)."""
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be None or a tensor, not {type(weight).__name__}")
    _f.check_weight(kind_of(weight), weight.dtype, weight.shape, None)


def device_error(*names: str) -> ValueError:
    """The error of tensors of a call (names) on different devices."""
    return ValueError(f"{', '.join(names[:-1])} and {names[-1]} must be on the same device")


def _module(function: Callable[..., torch.Tensor]) -> type[_Loss]:
    """The module class of a loss function: its parameters after the two tensors."""
    parameters = list(inspect.signature(function).parameters.values())[2:]
    signature = inspect.Signature(parameters)
    namespace = {
        "_function": staticmethod(function),
        "_signature": signature,
        "__signature__": signature,
        "__doc__": f"{function.__name__} as a module; see it for the arguments.",
    }
    return type("_Module", (_Loss,), namespace)


class CrossEntropyLoss(_module(cross_entropy)):
    """cross_entropy as a module (torch.nn.CrossEntropyLoss's arguments); weight is a buffer
    moved by .to() and saved in state_dict()."""

    def _check(self, options: dict[str, Any]) -> None:
        _cross_entropy_call(None, self.weight, **options)


class TopKCrossEntropyLoss(_module(topk_cross_entropy)):
    """topk_cross_entropy as a module; see it for the arguments."""


class MulticlassHingeLoss(_module(multiclass_hinge_loss)):
    """multiclass_hinge_loss as a module; see it for the arguments."""


class TopKHingeLoss(_module(topk_hinge_loss)):
    """topk_hinge_loss as a module; see it for the arguments."""


class ProximalCrossEntropyLoss(_module(proximal_cross_entropy)):
    """proximal_cross_entropy as a module; see it for the arguments."""


class MultilabelCrossEntropyLoss(_module(multilabel_cross_entropy)):
    """multilabel_cross_entropy as a module; see it for the arguments."""


class MultilabelHingeLoss(_module(multilabel_hinge_loss)):
    """multilabel_hinge_loss as a module; see it for the arguments."""


class DistillationLoss(_module(distillation_loss)):
    """distillation_loss as a module; see it for the arguments."""


class ProximalDistillationLoss(_module(proximal_distillation_loss)):
    """proximal_distillation_loss as a module; see it for the arguments."""


from ._torch_linear import LinearCrossEntropyLoss, linear_cross_entropy  # noqa: E402
from ._torch_linear import explain as _explain_linear  # noqa: E402

_EXPLAINERS: dict[Callable[..., Any], Callable[..., tuple[str, str]]] = {
    cross_entropy: _explain_cross_entropy,
    topk_cross_entropy: _explainer("topk_cross_entropy"),
    multiclass_hinge_loss: _explainer("multiclass_hinge"),
    topk_hinge_loss: _explainer("topk_hinge"),
    proximal_cross_entropy: _explainer("proximal_cross_entropy"),
    multilabel_cross_entropy: _explainer("multilabel_cross_entropy"),
    multilabel_hinge_loss: _explainer("multilabel_hinge"),
    distillation_loss: _explainer("distillation"),
    proximal_distillation_loss: _explainer("proximal_distillation"),
    linear_cross_entropy: _explain_linear,
}
