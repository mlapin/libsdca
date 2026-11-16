"""PyTorch-shaped integer-target cross-entropy with stable weighted reductions."""

from __future__ import annotations

import functools
import math
from types import ModuleType
from typing import Any, Protocol

import torch
from torch import nn

from . import _cpu, _cpu_entropy, _entropy, _ops
from . import _loss_options as _opt
from ._torch_distributed import call_with_dtensors, has_dtensor


def _torch() -> ModuleType:
    """libsdca.torch (which imports this module): imported here, as Dynamo
    reads a module cached in a global as None (PyTorch 2.6 and older)."""
    from . import torch as mod

    return mod


class _Context(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]
    reduction: str
    ignore_index: int
    has_weight: bool
    fp32: bool
    cuda_context: int | None

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class _TransformContext(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]
    reduction: str
    value_dtype: torch.dtype

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...

    def save_for_forward(self, *tensors: torch.Tensor) -> None: ...

    def mark_non_differentiable(self, *tensors: torch.Tensor) -> None: ...


def _weighted_value(
    stats: tuple[torch.Tensor, ...], weight: torch.Tensor, reduction: str
) -> torch.Tensor:
    values = _entropy._values(stats, "none")
    tiny = torch.finfo(torch.float64).tiny
    # Exact parts of the weighted values: tails and products beyond double's normal range.
    m, e = _entropy._parts(stats, weight)
    row_nan = values.isnan() | weight.isnan()
    if reduction == "none":
        ordinary = torch.where(weight > 0, values * weight, 0.0)
        res = torch.where(
            (values < tiny) | ~torch.isfinite(ordinary), _entropy._ldexp(m, e), ordinary
        )
        return res.masked_fill(row_nan, math.nan)
    normalized = _normalized_weight(weight) if reduction == "mean" else weight
    # Scale both factors before summing: a finite weighted mean must not overflow.
    scale = torch.where(weight > 0, values, 0.0).max()
    scale = torch.where(torch.isfinite(scale) & (scale > 0), scale, 1)
    terms = torch.where(weight > 0, values / scale, 0.0) * normalized
    ordinary = terms.sum() * scale
    exact = _entropy._reduce_parts(m, e, reduction, weight)
    underflow = (((terms.abs() < tiny) | (values < tiny)) & (weight > 0)).any()
    res = torch.where(underflow | ~torch.isfinite(ordinary), exact, ordinary)
    return torch.where(row_nan.any(), math.nan, res)


def _normalized_weight(weight: torch.Tensor) -> torch.Tensor:
    largest = weight.max()
    scaled = weight / torch.where(largest > 0, largest, 1)
    return scaled / scaled.sum()


@functools.cache
def _weighted_kernel() -> ModuleType | None:
    if _entropy._kernel() is None:
        return None
    from . import _triton_weighted_entropy

    return _triton_weighted_entropy


class _FusedWeightedEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _Context,
        logits: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None,
        ignore_index: int,
        reduction: str,
        fp32: bool = False,
    ) -> torch.Tensor:
        kernel = _weighted_kernel()
        assert kernel is not None
        value, stats, ctx.cuda_context = kernel.forward(
            logits, target, weight, ignore_index, reduction, fp32
        )
        ctx.save_for_backward(logits, target, target if weight is None else weight, stats)
        ctx.reduction, ctx.ignore_index, ctx.has_weight, ctx.fp32 = (
            reduction,
            ignore_index,
            weight is not None,
            fp32,
        )
        return value

    @staticmethod
    def backward(
        ctx: _Context, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        (upstream,) = grad_outputs
        logits, target, class_weight, saved = ctx.saved_tensors
        out_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        if not torch.is_grad_enabled() and upstream.dtype == out_dtype:
            kernel = _weighted_kernel()
            assert kernel is not None
            gradient = kernel.backward(
                logits,
                target,
                class_weight if ctx.has_weight else None,
                saved,
                upstream,
                ctx.ignore_index,
                ctx.reduction,
                ctx.cuda_context,
                ctx.fp32,
            )
            return gradient, None, None, None, None, None
        # Higher derivatives retain the existing differentiable double tensor path.
        ignored = target == ctx.ignore_index
        labels = target.masked_fill(ignored, 0)
        c = logits.shape[1]
        safe_labels = labels.clamp(0, c - 1)
        weight = (
            class_weight.double()[safe_labels]
            if ctx.has_weight
            else torch.ones_like(labels, dtype=torch.float64)
        )
        weight = weight.masked_fill(ignored, 0)
        # Reuse forward's device-side status for higher derivatives: there is
        # no need to scan logits or all class weights a second time.
        n = target.numel()
        invalid = (saved[3 : 4 * n : 4] < 0) | (saved[5 * n] == 3)
        if ctx.reduction != "none":
            invalid = (saved[5 * n] > 0).expand(n)
        stats = _entropy._statistics(logits, labels, invalid)
        gradient = _weighted_gradient(logits, labels, stats, weight, upstream, ctx.reduction)
        if ctx.reduction == "mean":
            undefined = (~ignored) & (saved[5 * n + 1] == 0)
            gradient = gradient.masked_fill(undefined[:, None], torch.nan)
        return gradient, None, None, None, None, None


def _fused_weighted(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    ignore_index: int,
    reduction: str,
    dim: int,
    fp32: bool = False,
) -> torch.Tensor | None:
    if not _can_fuse_weighted(input, target, ignore_index, dim):
        return None
    classes = input.shape[dim]
    z = (
        input.contiguous()
        if input.ndim == 2
        else input.movedim(dim, -1).reshape(-1, classes).contiguous()
    )
    y = (target if target.ndim == 1 else target.reshape(-1)).contiguous()
    w = None if weight is None else weight.contiguous()
    if _ops.compiling():
        return _ops.weighted_entropy(z, y, w, ignore_index, reduction, fp32)
    kernel = _weighted_kernel()
    assert kernel is not None
    out = kernel.apply(z, y, w, ignore_index, reduction, fp32)
    if out is not None:
        return out
    return _FusedWeightedEntropy.apply(
        z,
        y,
        w,
        ignore_index,
        reduction,
        fp32,
    )


def _can_fuse_weighted(
    input: torch.Tensor, target: torch.Tensor, ignore_index: int, dim: int
) -> bool:
    classes = input.shape[dim]
    if not (
        input.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        and input.is_cuda
        and target.numel() > 0
        and -(2**63) <= ignore_index < 2**63
        and _entropy.covers(input.device, target.numel(), classes)
    ):
        return False
    kernel = _weighted_kernel()
    return kernel is not None and classes <= kernel.CLASSES


class _WeightedEntropy(torch.autograd.Function):
    """Float64 tensor statistics; composes with torch.func transforms."""

    generate_vmap_rule = True

    @staticmethod
    def forward(
        logits: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        ignored: torch.Tensor,
        reduction: str,
    ) -> tuple[torch.Tensor, ...]:
        stats = _entropy._statistics(logits, target)
        dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
        return _weighted_value(stats, weight, reduction).to(dtype), *stats

    @staticmethod
    def setup_context(
        ctx: _TransformContext, inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]
    ) -> None:
        logits, target, weight, ignored, reduction = inputs
        value, *stats = output
        ctx.mark_non_differentiable(*stats)
        ctx.save_for_backward(logits, target, weight, ignored, *stats)
        ctx.reduction, ctx.value_dtype = reduction, value.dtype

    @staticmethod
    def backward(
        ctx: _TransformContext, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        return _WeightedEntropy._gradient(ctx, grad_outputs[0]), None, None, None, None

    @staticmethod
    def _gradient(ctx: _TransformContext, upstream: torch.Tensor) -> torch.Tensor:
        logits, target, weight, ignored, *saved = ctx.saved_tensors
        stats = (
            _entropy._statistics(logits, target, saved[0].isnan())
            if torch.is_grad_enabled()
            else tuple(saved)
        )
        gradient = _weighted_gradient(logits, target, stats, weight, upstream, ctx.reduction)
        if ctx.reduction == "mean":
            undefined = (~ignored) & ~(weight > 0).any()
            gradient = gradient.masked_fill(undefined[:, None], torch.nan)
        return gradient


class _TransformedEntropy(_WeightedEntropy):
    """_WeightedEntropy with forward mode (jvp, hessian): Dynamo cannot trace a custom jvp."""

    @staticmethod
    def setup_context(
        ctx: _TransformContext, inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]
    ) -> None:
        _WeightedEntropy.setup_context(ctx, inputs, output)
        ctx.save_for_forward(*inputs[:4], *output[1:])

    @staticmethod
    def jvp(
        ctx: _TransformContext, *tangents: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, ...]:
        tangent, nones = tangents[0], (None,) * 4
        if tangent is None:
            return None, *nones
        rows = () if ctx.reduction != "none" else tangent.shape[:1]
        upstream = tangent.new_ones(rows, dtype=torch.float64)
        product = _WeightedEntropy._gradient(ctx, upstream).double() * tangent.double()
        value = product.sum(-1) if ctx.reduction == "none" else product.sum()
        return value.to(ctx.value_dtype), *nones


def _weighted_gradient(
    logits: torch.Tensor,
    target: torch.Tensor,
    stats: tuple[torch.Tensor, ...],
    weight: torch.Tensor,
    upstream: torch.Tensor,
    reduction: str,
    *,
    normalizer: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if normalizer is not None:
        largest, total = normalizer
        factor = (weight / torch.where(largest > 0, largest, 1)) / total
    else:
        factor = _normalized_weight(weight) if reduction == "mean" else weight
    bounded = logits.dtype != torch.float64 and upstream.dtype != torch.float64
    if bounded:
        scale = upstream.double().expand_as(weight) * factor
        scale = torch.where(weight > 0, scale, 0.0)
        gradient = _entropy._gradient(logits, target, stats, scale, "none", bounded_scale=True)
        row_nan = stats[0].isnan() | weight.isnan()
        if reduction != "none":
            row_nan = row_nan.any().expand_as(row_nan)
        return gradient.masked_fill(row_nan[:, None], math.nan)
    safe_target = target.clamp(0, logits.shape[1] - 1)[:, None]
    upstream = upstream.double().expand_as(weight)
    scale = upstream * factor
    tiny = torch.finfo(torch.float64).tiny
    rare = ~torch.isfinite(scale) | (scale.abs() < tiny) | (factor.abs() < tiny)
    rare = rare & (weight > 0) & (upstream != 0)
    safe_scale = torch.where((weight > 0) & ~rare, scale, 0.0)
    ordinary = _entropy._gradient(logits, target, stats, safe_scale, "none")
    maximum, _, log_sum, relative = stats
    log_mass = -torch.logaddexp(torch.zeros_like(relative), -relative)
    logs = (logits.double() - maximum[:, None]) - log_sum[:, None] + log_mass[:, None]
    logs = logs.scatter(1, safe_target, log_mass[:, None])
    log_weight = weight.log()
    if reduction == "mean":
        denominator = (
            torch.logsumexp(log_weight, 0)
            if normalizer is None
            else normalizer[0].log() + normalizer[1].log()
        )
        log_weight = log_weight - torch.where(torch.isfinite(denominator), denominator, 0)
    # Combine probability and weight before either can underflow or overflow.
    root = ((logs + log_weight[:, None]) * 0.5).exp()
    exceptional = root * (root * upstream[:, None])
    exceptional = exceptional.scatter(1, safe_target, -exceptional.gather(1, safe_target))
    out = torch.where(rare[:, None], exceptional.to(logits.dtype), ordinary)
    row_nan = stats[0].isnan() | weight.isnan()
    if reduction != "none":
        row_nan = row_nan.any().expand_as(row_nan)
    return out.masked_fill(row_nan[:, None], math.nan)


def _options(
    weight: torch.Tensor | None, ignore_index: int, reduction: str, label_smoothing: float
) -> float:
    """Check the options of cross_entropy; label_smoothing as a float."""
    if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
        raise TypeError(f"ignore_index must be an integer, not {type(ignore_index).__name__}")
    if not -(2**63) <= ignore_index < 2**63:
        raise ValueError("ignore_index must fit in int64")
    _opt.check_reduction(reduction)
    eps = _opt.check_label_smoothing(label_smoothing)
    if weight is not None:
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"weight must be None or a tensor, not {type(weight).__name__}")
        if weight.requires_grad:
            raise ValueError("weight must not require gradients")
    return eps


_BACKENDS = ("auto", "kernel", "native", "torch")


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
) -> torch.Tensor:
    """Stable cross-entropy of (C,), (N, C) or (N, C, ...) logits, as
    torch.nn.functional.cross_entropy.

    target holds class labels (any integer dtype) of the logits' shape without the class
    dimension, or class probabilities of the logits' shape (floating point). weight,
    ignore_index (labels only), reduction and label_smoothing in [0, 1] follow PyTorch,
    including the weighted mean. Half inputs return float32 losses. Integer labels without
    smoothing run the fused kernels: every call keeps float64 row statistics and supports
    double backward; on the CPU the CPU extension or the native library computes weighted and
    ignored rows too, and auto CUDA fuses rows of every supported dtype up to 2**25 classes.
    Label smoothing adds (eps / C) times the smoothing term (a logsumexp in tensor
    operations) to (1 - eps) times that loss; probability targets take log_softmax in
    tensor operations. Both work in float32 for 16-bit logits (else the logits' dtype, or
    float64 for precision="float64").

    backend: "auto", "kernel" (as "auto", raising NotImplementedError where no fused kernel
    runs), "native" (the native library, CPU tensors) or "torch" (tensor operations).
    precision: None, "float32" or "float64" (or torch.float32, torch.float64).

    Raises:
        TypeError: For arguments of the wrong type (a boolean target).
        ValueError: For invalid shapes or options; on the CPU also for nonfinite logits,
            labels out of range or invalid class weights (CUDA and torch.func transforms give
            NaN instead).
    """
    fast = _fast_cpu(
        input, target, weight, ignore_index, reduction, label_smoothing, backend, precision
    )
    if fast is not None:
        return fast
    if has_dtensor(input, target, weight):
        return call_with_dtensors(
            cross_entropy,
            input,
            target,
            weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
            backend=backend,
            precision=precision,
        )
    eps = _options(weight, ignore_index, reduction, label_smoothing)
    mod = _torch()
    mod._check_tensors(input, target)
    canonical = mod._check_precision(input, precision)
    _opt.check_backend(backend, _BACKENDS)
    dim = 0 if input.ndim == 1 else 1
    classes = input.shape[dim]
    if classes < 2:
        raise ValueError("At least two classes are required")
    if target.is_floating_point():
        return _probabilities(input, target, weight, ignore_index, reduction, eps, canonical)
    target = _class_labels(input, target, dim)
    if eps > 0:
        nll = cross_entropy(
            input,
            target,
            weight,
            ignore_index=ignore_index,
            reduction=reduction,
            backend=backend,
            precision=canonical,
        )
        smooth = _smoothing(input, target, weight, ignore_index, reduction, canonical)
        return ((1 - eps) * nll + (eps / classes) * smooth).to(nll.dtype)
    if backend == "kernel":
        mod._require_kernel(_label_route(input, target, weight, ignore_index, "auto", canonical))
        backend = "auto"
    return _labels_loss(input, target, weight, ignore_index, reduction, backend, canonical)


def _class_labels(input: torch.Tensor, target: torch.Tensor, dim: int) -> torch.Tensor:
    """int64 class labels of the logits' shape without dim (any integer dtype is cast)."""
    if target.dtype != torch.int64:
        if target.dtype == torch.bool or target.is_complex():
            hint = (
                "; multilabel_cross_entropy takes boolean masks"
                if target.dtype == torch.bool
                else ""
            )
            raise TypeError(
                f"cross_entropy takes integer class labels or floating-point class "
                f"probabilities, not {target.dtype}{hint}"
            )
        target = target.long()
    shape = input.shape[:dim] + input.shape[dim + 1 :]
    if target.shape != shape:
        raise ValueError(
            f"class labels must have shape {tuple(shape)} (the logits' shape without the "
            f"class dimension 1), not {tuple(target.shape)}"
        )
    return target


def _work_dtype(input: torch.Tensor, precision: str | None) -> torch.dtype:
    """float32 work for 16-bit and float32 logits, float64 for float64 logits or
    precision="float64"."""
    if precision == "float64" or input.dtype == torch.float64:
        return torch.float64
    return torch.float32


def _class_weight(weight: torch.Tensor, input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Checked class weights in dtype, NaN where invalid off the checked CPU path."""
    classes = input.shape[0 if input.ndim == 1 else 1]
    checked = not (input.is_cuda or _ops.is_compiling() or _ops.transformed())
    _check_weight(weight, input, classes, check_values=checked)
    w = weight.to(dtype)
    if not checked:
        valid = torch.isfinite(w).all() & (w >= 0).all()
        w = w.masked_fill(~valid, math.nan)
    return w


def _smoothing(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    ignore_index: int,
    reduction: str,
    precision: str | None,
) -> torch.Tensor:
    """PyTorch's label-smoothing term: -sum_j w_j log p_j per example (w = 1 without class
    weights), zero where ignored, summed, or for the mean divided by the sum of the label
    weights (the count of examples without class weights)."""
    dim = 0 if input.ndim == 1 else 1
    work = _work_dtype(input, precision)
    z = input.movedim(dim, -1).to(work)
    lse = torch.logsumexp(z, -1)
    ignored = target == ignore_index
    if weight is None:
        smooth = z.shape[-1] * lse - z.sum(-1)
        label_weight = (~ignored).to(work)
    else:
        w = _class_weight(weight, input, work)
        smooth = w.sum() * lse - (z * w).sum(-1)
        labels = target.masked_fill(ignored, 0).clamp(0, z.shape[-1] - 1)
        label_weight = w[labels].masked_fill(ignored, 0)
    smooth = smooth.masked_fill(ignored, 0)
    if reduction == "none":
        return smooth
    if reduction == "sum":
        return smooth.sum()
    return smooth.sum() / label_weight.sum()


def _probabilities(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    ignore_index: int,
    reduction: str,
    eps: float,
    precision: str | None,
) -> torch.Tensor:
    """Cross-entropy of class probabilities, as PyTorch: -sum_j w_j t_j log p_j per example
    (t smoothed to (1 - eps) t + eps / C); the mean divides by the number of examples."""
    if target.shape != input.shape:
        raise ValueError(
            f"class probabilities must have the logits' shape {tuple(input.shape)}, not "
            f"{tuple(target.shape)}"
        )
    if ignore_index >= 0:
        raise ValueError("ignore_index is not supported with class probabilities")
    dim = 0 if input.ndim == 1 else 1
    classes = input.shape[dim]
    work = _work_dtype(input, precision)
    log_p = torch.log_softmax(input.to(work), dim)
    t = target.to(work)
    if eps > 0:
        t = t * (1 - eps) + eps / classes
    product = log_p * t
    if weight is not None:
        shape = [1] * input.ndim
        shape[dim] = classes
        product = product * _class_weight(weight, input, work).reshape(shape)
    value_dtype = _cpu.value_dtype(input)
    if reduction == "none":
        return (-product.sum(dim)).to(value_dtype)
    total = -product.sum()
    if reduction == "sum":
        return total.to(value_dtype)
    return (total / (input.numel() // classes)).to(value_dtype)


def _labels_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    ignore_index: int,
    reduction: str,
    backend: str,
    canonical: str | None,
) -> torch.Tensor:
    """cross_entropy of checked int64 labels without smoothing (backend not "kernel")."""
    mod = _torch()
    transformed = _ops.transformed()
    dim = 0 if input.ndim == 1 else 1
    classes = input.shape[dim]
    shape = target.shape
    checked = not (input.is_cuda or _ops.is_compiling() or transformed)
    _check_weight(weight, input, classes, check_values=checked)
    if target.numel() == 0:
        return _empty(input, weight, reduction, dim)
    z = input.movedim(dim, -1) if input.ndim > 2 else input
    if input.device.type == "cpu" and backend != "torch" and not transformed:
        value = _cpu_cross_entropy(z, target, weight, ignore_index, reduction, backend, canonical)
        return value.reshape(shape) if reduction == "none" else value
    # Without ignored rows (a data-dependent test, not traced) the shared entropy op applies.
    if checked and weight is None and not bool((target == ignore_index).any()):
        return mod._resolved_loss(
            input, target, "cross_entropy", backend, 1, 1.0, 1.0, reduction, dim, canonical, 1
        )
    if backend == "auto" and not transformed:
        fp32 = canonical != "float64" and input.dtype != torch.float64
        fused_value = _fused_weighted(input, target, weight, ignore_index, reduction, dim, fp32)
        if fused_value is not None:
            return fused_value.reshape(shape) if reduction == "none" else fused_value
    return _tensor_labels_loss(
        input, target, weight, ignore_index, reduction, backend, checked, transformed
    )


def _tensor_labels_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    ignore_index: int,
    reduction: str,
    backend: str,
    checked: bool,
    transformed: bool,
) -> torch.Tensor:
    """_labels_loss in float64 tensor operations (checked: with the synchronizing row
    checks; transformed: under torch.func transforms)."""
    dim = 0 if input.ndim == 1 else 1
    classes = input.shape[dim]
    shape = target.shape
    ignored = target == ignore_index
    if backend == "native":
        raise ValueError(
            "Native losses require CPU tensors outside torch.func transforms; "
            "use backend='torch' otherwise"
        )
    labels = target.masked_fill(ignored, 0).reshape(-1)
    z = input.movedim(dim, -1).reshape(-1, classes)
    if checked:
        z = _checked_rows(z, labels, ignored.reshape(-1))
    safe_labels = labels.clamp(0, classes - 1)
    row_weight = (
        torch.ones_like(labels, dtype=torch.float64)
        if weight is None
        else weight.double()[safe_labels]
    )
    row_weight = row_weight.masked_fill(ignored.reshape(-1), 0)
    if not checked:
        # Without a synchronizing check (CUDA, traced or transformed): invalid rows give NaN.
        bad_row = (~ignored.reshape(-1)) & ((labels < 0) | (labels >= classes))
        if weight is not None:
            valid_w = torch.isfinite(weight).all() & (weight >= 0).all()
            bad_row = bad_row | ~valid_w
        row_weight = row_weight.masked_fill(bad_row, math.nan)
    entropy = _TransformedEntropy if transformed else _WeightedEntropy
    value = entropy.apply(z, labels, row_weight, ignored.reshape(-1), reduction)[0]
    return value.reshape(shape) if reduction == "none" else value


def _label_route(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    ignore_index: int,
    backend: str,
    canonical: str | None,
) -> tuple[str, str]:
    """explain_backend's (backend, reason) of _labels_loss (backend "auto", "native" or
    "torch")."""
    mod = _torch()
    dim = 0 if input.ndim == 1 else 1
    transformed = _ops.transformed()
    fp32 = canonical != "float64" and input.dtype != torch.float64
    arithmetic = "float32 exponentials, float64 row statistics" if fp32 else "float64"
    if input.device.type == "cpu" and backend != "torch" and not transformed:
        return _cpu_label_route(input, backend, dim, arithmetic)
    checked = not (input.is_cuda or _ops.is_compiling() or transformed)
    if checked and weight is None and target.numel() and not bool((target == ignore_index).any()):
        call = mod._bind("topk_cross_entropy", k=1, dim=dim, backend=backend, precision=canonical)
        return mod._explain_loss(call, input, target)
    if backend == "auto" and not transformed and target.numel():
        if _can_fuse_weighted(input, target, ignore_index, dim):
            return "kernel", f"fused Triton weighted-entropy kernels ({arithmetic})"
    return _tensor_label_route(backend, transformed)


def _cpu_label_route(
    input: torch.Tensor, backend: str, dim: int, arithmetic: str
) -> tuple[str, str]:
    """_label_route on CPU outside torch.func transforms (backend "auto" or "native"):
    the CPU extension's kernel or the native library."""
    classes = input.shape[dim]
    extension = backend == "auto" and _cpu_entropy.enabled()
    z = input.movedim(dim, -1) if input.ndim > 2 else input
    if _cpu.entropy_implementation(z, extension, classes) == _cpu.EXTENSION:
        return "kernel", f"the CPU extension's cross-entropy kernel ({arithmetic})"
    why = (
        ""
        if backend == "native"
        else f"; the CPU extension is unavailable ({_cpu_entropy.why_not()})"
    )
    if backend == "auto" and extension:
        why = " (float64 rows wider than the extension's)"
    return "native", f"the native library ({arithmetic}){why}"


def _tensor_label_route(backend: str, transformed: bool) -> tuple[str, str]:
    """_label_route of the tensor operations; backend "native" raises."""
    if backend == "native":
        raise ValueError("Native losses require CPU tensors; use backend='torch' otherwise")
    why = "backend='torch'" if backend == "torch" else "no fused kernel applies"
    if transformed:
        why = "torch.func transforms"
    return "torch", f"float64 tensor operations ({why})"


def explain(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    backend: str = "auto",
    precision: object = None,
) -> tuple[str, str]:
    """explain_backend of cross_entropy (its arguments)."""
    mod = _torch()
    if has_dtensor(input, target, weight):
        mod._dtensor_mesh(input, target, weight)
        explained = explain(
            mod._stand_in(input),
            mod._stand_in(target),
            mod._stand_in(weight),
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
            backend=backend,
            precision=precision,
        )
        return mod._gathered(explained)
    eps = _options(weight, ignore_index, reduction, label_smoothing)
    mod._check_tensors(input, target)
    canonical = mod._check_precision(input, precision)
    _opt.check_backend(backend, _BACKENDS)
    dim = 0 if input.ndim == 1 else 1
    if input.shape[dim] < 2:
        raise ValueError("At least two classes are required")
    if target.is_floating_point():
        work = str(_work_dtype(input, canonical)).removeprefix("torch.")
        return "torch", f"class probabilities: log_softmax in {work} tensor operations"
    target = _class_labels(input, target, dim)
    if weight is not None:
        _check_weight(weight, input, input.shape[dim], check_values=False)
    route = "auto" if backend == "kernel" else backend
    name, reason = _label_route(input, target, weight, ignore_index, route, canonical)
    if eps > 0:
        reason += "; label smoothing adds a logsumexp term in tensor operations"
    return name, reason


def _checked_rows(z: torch.Tensor, labels: torch.Tensor, ignored: torch.Tensor) -> torch.Tensor:
    """z (rows, classes) after the CPU value checks, with zero ignored rows if they are
    nonfinite: as the native losses, ignored rows may hold any values and get zero gradients."""
    mod, counted = _torch(), z
    if bool(ignored.any()):
        counted = z[~ignored]
        if not bool(mod._finite(z[ignored])):
            z = z.masked_fill(ignored[:, None], 0)
    mod._raise_failed(
        [
            (~mod._finite(counted), "logits must be finite floating-point values"),
            (((labels < 0) | (labels >= z.shape[1])).any(), "Target index out of range"),
        ]
    )
    return z


_FAST_DTYPES = (torch.float32, torch.bfloat16, torch.float16, torch.float64)


def _fast_cpu(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    ignore_index: int,
    reduction: str,
    label_smoothing: float,
    backend: str,
    precision: object,
) -> torch.Tensor | None:
    """_cpu_cross_entropy of ordinary (N, C) CPU tensors, or None: where it applies every check
    of cross_entropy passes but the value checks, which the kernels make with the same messages
    in the same order (class weights only in the extension: the native library takes them
    checked)."""
    if not (
        type(input) is torch.Tensor
        and type(target) is torch.Tensor
        and _plain_options(ignore_index, reduction, label_smoothing, backend)
        and _plain_logits(input, precision)
        and (weight is None or _extension_rows(input, backend))
        and _plain_target(target, input)
        and not _ops.is_compiling()
        and not _ops.transformed()
    ):
        return None
    extension = backend == "auto" and _cpu_entropy.enabled()
    implementation = _cpu.entropy_implementation(input, extension, input.shape[1])
    if weight is not None and not (
        _plain_weight(weight, input) and implementation == _cpu.EXTENSION
    ):
        return None
    # As _cpu_cross_entropy: float32 exponentials for non-double logits.
    fp32 = int(input.dtype != torch.float64)
    ints = [_cpu.REDUCTIONS[reduction], ignore_index, fp32, implementation]
    w = None if weight is None else weight.contiguous()
    return _cpu.loss("entropy", input.contiguous(), target.contiguous(), w, ints, [])


def _plain_options(ignore_index: int, reduction: str, label_smoothing: float, backend: str) -> bool:
    """Whether the options of a cross_entropy call pass its checks, on a CPU kernel."""
    return (
        backend in ("auto", "native")
        and label_smoothing == 0
        and type(ignore_index) is int
        and -(2**63) <= ignore_index < 2**63
        and reduction in ("none", "mean", "sum")
    )


def _plain_logits(input: torch.Tensor, precision: object) -> bool:
    """Whether input is (N, C) CPU logits of at least two classes in the default work."""
    return (
        input.is_cpu
        and input.dtype in _FAST_DTYPES
        and (precision is None or (precision == "float32" and input.dtype != torch.float64))
        and input.ndim == 2
        and input.shape[1] >= 2
    )


def _extension_rows(input: torch.Tensor, backend: str) -> bool:
    """Whether the (N, C) logits can take the extension (float64 rows of at most
    _cpu.DOUBLE_CLASSES classes)."""
    return backend == "auto" and (
        input.dtype != torch.float64 or input.shape[1] <= _cpu.DOUBLE_CLASSES
    )


def _plain_target(target: torch.Tensor, input: torch.Tensor) -> bool:
    """Whether target holds the int64 CPU labels of every row of input (at least one)."""
    return (
        target.dtype == torch.int64
        and target.ndim == 1
        and target.is_cpu
        and 0 < target.shape[0] == input.shape[0]
    )


def _plain_weight(weight: torch.Tensor, input: torch.Tensor) -> bool:
    """Whether weight is a constant (C,) CPU tensor of the dtype of input."""
    return (
        type(weight) is torch.Tensor
        and not weight.requires_grad
        and weight.dtype == input.dtype
        and weight.is_cpu
        and weight.shape == (input.shape[1],)
    )


def _empty(
    input: torch.Tensor, weight: torch.Tensor | None, reduction: str, dim: int
) -> torch.Tensor:
    """The loss of no examples, connected to the graph: empty, a zero sum or a NaN mean."""
    res = _torch()._empty_loss(input.movedim(dim, -1), reduction, _cpu.value_dtype(input))
    if weight is not None and input.is_cuda:
        valid_w = torch.isfinite(weight).all() & (weight >= 0).all()
        res = torch.where(valid_w, res, math.nan)
    return res


def _cpu_cross_entropy(
    z: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    ignore_index: int,
    reduction: str,
    backend: str,
    precision: str | None,
) -> torch.Tensor:
    """The CPU kernel (the extension, else the native library) of class-last logits z."""
    mod = _torch()
    fp32 = precision != "float64" and z.dtype != torch.float64
    extension = backend == "auto" and _cpu_entropy.enabled()
    implementation = _cpu.entropy_implementation(z, extension, z.shape[-1])
    ints = [_cpu.REDUCTIONS[reduction], ignore_index, int(fp32), implementation]
    extra = None if weight is None else weight.contiguous()
    return _cpu.loss("entropy", mod._flat(z), mod._labels(target), extra, ints, [])


def _check_weight(
    weight: torch.Tensor | None, logits: torch.Tensor, classes: int, *, check_values: bool = True
) -> None:
    """Class weights: a (classes,) tensor on the logits' device in their dtype (float32 for
    16-bit logits under autocast); finite and nonnegative where check_values."""
    if weight is None:
        return
    autocast_weight = (
        logits.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == torch.float32
        and _autocast_enabled(logits.device.type)
    )
    if weight.dtype != logits.dtype and not autocast_weight:
        raise TypeError(f"weight must have the logits' dtype {logits.dtype}, not {weight.dtype}")
    if weight.shape != (classes,):
        raise ValueError(f"weight must have shape ({classes},), not {tuple(weight.shape)}")
    if weight.device != logits.device:
        raise ValueError("weight and logits must be on the same device")
    if check_values and not bool(torch.isfinite(weight).all() & (weight >= 0).all()):
        raise ValueError("weight must be finite and nonnegative")


def _autocast_enabled(device: str) -> bool:
    # PyTorch 2.1 predates the device argument on is_autocast_enabled.
    try:
        return torch.is_autocast_enabled(device)
    except TypeError:
        return torch.is_autocast_cpu_enabled() if device == "cpu" else torch.is_autocast_enabled()


class CrossEntropyLoss(nn.Module):
    """cross_entropy as a module (torch.nn.CrossEntropyLoss's arguments); weight is a buffer
    moved by .to() and saved in state_dict()."""

    weight: torch.Tensor | None

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        *,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__()
        _options(weight, ignore_index, reduction, label_smoothing)
        _opt.check_backend(backend, _BACKENDS)
        _torch()._precision(precision)
        self.register_buffer("weight", weight)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.backend = backend
        self.precision = precision

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Evaluate cross_entropy with the stored options."""
        return cross_entropy(
            input,
            target,
            self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            backend=self.backend,
            precision=self.precision,
        )

    def _options(self) -> dict[str, Any]:
        return dict(
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            backend=self.backend,
            precision=self.precision,
        )

    def _explain(self, input: torch.Tensor, target: torch.Tensor) -> tuple[str, str]:
        return explain(input, target, self.weight, **self._options())

    def extra_repr(self) -> str:
        return ", ".join(f"{key}={value!r}" for key, value in self._options().items())
