"""PyTorch losses: one function and one module per loss (cross_entropy and CrossEntropyLoss,
topk_cross_entropy, the hinges, the multilabel losses, proximal_cross_entropy, the
distillation losses and linear_cross_entropy). CPU losses run in the CPU extension (built on
first use, _cpu_entropy) or the native library (_cpu), CUDA losses in fused Triton kernels;
the tensor backend runs every loss on either device, and its entropy has higher derivatives.
"""

from __future__ import annotations

import functools
import inspect
import math
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, overload

import torch
from torch import nn
from torch.autograd.function import FunctionCtx

from . import _cpu, _cpu_entropy, _cuda_launch, _entropy, _hinge, _multilabel, _ops, _topk
from . import _cross_entropy as _ce
from . import _linear_cross_entropy as _lce
from . import _loss_options as _opt
from ._cross_entropy import CrossEntropyLoss as CrossEntropyLoss
from ._cross_entropy import cross_entropy as cross_entropy
from ._entropy import entropy as _indexed_entropy
from ._entropy import fused as _fused_entropy
from ._first_order import MESSAGE as _FIRST_ORDER
from ._first_order import first_order as _first_order
from ._linear_cross_entropy import LinearCrossEntropyLoss as LinearCrossEntropyLoss
from ._linear_cross_entropy import linear_cross_entropy as linear_cross_entropy
from ._loss_options import Spec as _Spec
from ._tensor_losses import (
    _SINGLE_TARGET,
    _first,
    _multilabel_hinge,
    _normalizer,
    _over,
    _single_target,
    _TensorEntropy,
    _work,
)
from ._tensor_losses import _scalars as _scalars
from ._tensor_prox import proximal_distillation as _proximal_distillation
from ._tensor_prox import proximal_entropy as _tensor_proximal_entropy
from ._torch_distributed import call_with_dtensors as _call_with_dtensors
from ._torch_distributed import dtensor_mesh as _dtensor_mesh
from ._torch_distributed import has_dtensor as _has_dtensor

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
Backend = Literal["auto", "kernel", "native", "torch"]
# backend="kernel" is "auto" raising NotImplementedError where no fused kernel runs.
BACKENDS = ("auto", "kernel", "native", "torch")
_PRECISIONS = {torch.float32: "float32", torch.float64: "float64"}
# explain_backend's names of the internal routes.
_PUBLIC = {"cpu-kernel": "kernel", "cuda-launcher": "kernel", "triton": "kernel"}


def _precision(precision: object) -> str | None:
    """None, "float32" or "float64" (or torch.float32, torch.float64); else raise."""
    if precision is None:
        return None
    return _opt.check_precision_name(precision, _PRECISIONS, "torch")


# The widest register-resident rows of the fused Triton kernels; wider ones stream tiles.
_KERNEL_CLASSES = 4096


def _empty_loss(z: torch.Tensor, reduction: str, dtype: torch.dtype) -> torch.Tensor:
    """The loss of no examples, as torch.nn.functional: empty per-example losses, a zero sum
    and a NaN mean, connected to the graph of the (class-last) logits z."""
    return _reduce(z.sum(-1).to(dtype), reduction)


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError("reduction must be none, sum, or mean")


_HALF = (torch.float16, torch.bfloat16)


def _value_dtype(logits: torch.Tensor) -> torch.dtype:
    """The dtype of every loss value: promote(logits dtype, float32), as under autocast."""
    return torch.promote_types(logits.dtype, torch.float32)


def _working(logits: torch.Tensor, precision: str | None, prox_kernel: bool = False) -> str | None:
    """The precision a call computes in: None is float32 for float16/bfloat16 logits, and for
    float32 logits on a proximal_cross_entropy kernel (its float32 kernels meet the proximal
    bound; CUDA k=1 hinges instead use _hinge_fp32, while tensor and native paths keep float64)."""
    if precision is None and (
        logits.dtype in _HALF or (prox_kernel and logits.dtype == torch.float32)
    ):
        return "float32"
    return precision


def _device_working(
    logits: torch.Tensor, z: torch.Tensor, canonical: str | None, objective: str, fused: bool
) -> str | None:
    """_working of a call off the CPU kernels: float32 logits of proximal_cross_entropy take
    float32 on its kernel (fused, where the kernel covers the rows of class-last z)."""
    kernel = objective == "proximal_cross_entropy" and fused
    return _working(
        logits, canonical, kernel and _prox_kernel(z.reshape(-1, z.shape[-1])) is not None
    )


def _contract(logits: torch.Tensor, canonical: str | None) -> str | None:
    """precision="float32" of non-float64 logits is None: both promise float32 work and take
    the same (fastest) route; float64 logits keep "float32" for its error."""
    return None if canonical == "float32" and logits.dtype != torch.float64 else canonical


def _check_precision(logits: torch.Tensor, precision: object) -> str | None:
    """float64 logits take float64 (or None); other dtypes take float32 or float64 (or None)."""
    canonical = _precision(precision)
    if logits.dtype == torch.float64 and canonical == "float32":
        raise ValueError("float64 logits require precision='float64' or None")
    return canonical


def _finite(t: torch.Tensor) -> torch.Tensor:
    """0-d bool: every entry is finite. NaN propagates through amax; no full-size temporaries."""
    if t.numel() == 0:
        return torch.ones((), dtype=torch.bool, device=t.device)
    return torch.isfinite(t.amax()) & torch.isfinite(t.amin())


@functools.cache
def _kernel_prox() -> Callable[..., tuple[torch.Tensor, torch.Tensor]] | None:
    """The Triton proximal_entropy, or None without Triton >= 3.3 or an NVIDIA GPU."""
    if torch.version.cuda is None:
        return None
    try:
        import triton

        from ._triton_prox import proximal_entropy as kernel
    except ImportError:
        return None
    return kernel if _cuda_launch.kernels_supported(triton.__version__) else None


def _prox_kernel(logits: torch.Tensor) -> Callable[..., tuple[torch.Tensor, torch.Tensor]] | None:
    """The Triton proximal_entropy for CUDA rows, else None.

    One program per row: fewer than 2**31 rows (the grid limit).
    """
    if logits.is_cuda and logits.shape[0] < 2**31:
        return _kernel_prox()
    return None


def _float32_holds(z: torch.Tensor, *scales: float, check_spread: bool = True) -> bool:
    """Whether float32 safely holds the scales, reciprocals and row spreads.

    CUDA tensor operations cannot choose their dtype from device values without
    synchronizing. Conservatively promote storage types whose finite range can
    overflow a float32 difference. A kernel that checks and promotes individual
    rows on the device can pass check_spread=False.
    """
    tiny = torch.finfo(torch.float32).tiny
    if not all(tiny <= s <= 1 / tiny for s in scales):
        return False
    if not check_spread or z.dtype == torch.float16:
        return True
    if z.is_cuda:
        return False
    spread = z.amax(-1).float() - z.amin(-1).float()
    return bool(spread.isfinite().all())


def _tensor_loss(
    logits: torch.Tensor,
    positive: torch.Tensor,
    objective: str,
    k: int,
    gamma: float,
    alpha: float,
    precision: str | None,
    single: bool = False,
    fused: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Losses and gradients (float32 for 16-bit logits, else their dtype), computed as
    include/sdca/loss.h in float64; fused=False keeps proximal_cross_entropy off its kernel.

    precision="float32" computes in float32 where float32 holds gamma, alpha and the spread.
    CUDA tensor fallbacks conservatively promote float32/bfloat16/float64 storage;
    the proximal kernel instead checks each row and promotes on the device.
    The proximal_entropy kernel reads the logits and writes the gradient in the saved dtype.
    """
    kernel = _prox_kernel(logits) if fused and objective == "proximal_cross_entropy" else None
    fp32 = precision == "float32" and _float32_holds(
        logits, gamma, alpha, check_spread=kernel is None
    )
    if objective == "proximal_cross_entropy":
        if kernel is not None:
            run = _ops.proximal_entropy if _ops.compiling() else kernel
            return run(logits, positive, alpha, fp32)
        z = logits.float() if fp32 else logits.double()
        value, gradient = _tensor_proximal_entropy(z, positive, alpha, single=single)
        return value, gradient.to(_cpu.saved_dtype(logits))
    z = logits.float() if fp32 else logits.double()
    store = _cpu.saved_dtype(logits)
    if objective in _SINGLE_TARGET:
        value, others = _single_target(z, positive, objective, k, gamma)
        # Entropy's target mass survives even when every stored competitor vanishes.
        stored = others.to(store)
        total = (
            -torch.expm1(-value[:, None])
            if objective == "topk_cross_entropy"
            else stored.sum(-1, keepdim=True, dtype=torch.float64)
        )
        gradient = torch.where(positive, (-total).to(store), stored)
    else:
        value, grad64 = _multilabel_hinge(z, positive, gamma, objective.endswith("_smooth"))
        gradient = grad64.to(store)
    if logits.is_cuda:
        count = positive.sum(-1, keepdim=True)
        finite = z.isfinite().all(-1, keepdim=True)
        if objective in _SINGLE_TARGET:
            invalid = (count != 1) | ~finite
        else:
            invalid = (count == 0) | (count == z.shape[-1]) | ~finite
        value = value.masked_fill(invalid.squeeze(-1), math.nan)
        gradient = gradient.masked_fill(invalid, math.nan)
    return value, gradient


class _SavedGradient(torch.autograd.Function):
    """A first-order loss whose forward saves (gradient, logits): backward scales the gradient.

    Subclasses define forward; the first input is the logits.
    """

    @staticmethod
    @_first_order
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        gradient, logits = ctx.saved_tensors
        inplace = not (_ops.is_compiling() or _cpu.keep_graph())
        nones = (None,) * (len(ctx.needs_input_grad) - 1)
        return (_cpu.scaled(gradient, grad_output, logits.dtype, inplace), *nones)


class _Loss(_SavedGradient):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        logits: torch.Tensor,
        positive: torch.Tensor,
        objective: str,
        k: int,
        gamma: float,
        alpha: float,
        precision: str | None,
        single: bool = False,
        fused: bool = True,
    ) -> torch.Tensor:
        value, gradient = _tensor_loss(
            logits.detach(), positive, objective, k, gamma, alpha, precision, single, fused
        )
        # The logits only link a create_graph gradient to them (first_order).
        ctx.save_for_backward(gradient, logits)
        return value


def _check_device(t: torch.Tensor) -> None:
    if not (t.is_cpu or t.is_cuda):
        raise ValueError(f"libsdca losses take CPU or CUDA tensors, not {t.device.type}")


def _check_tensors(logits: torch.Tensor, target: torch.Tensor) -> None:
    """Tensors (TypeError) on one CPU or CUDA device, logits with a class dimension
    (ValueError)."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a tensor, not {type(logits).__name__}")
    if not logits.is_floating_point():
        raise TypeError(f"logits must be floating-point, not {logits.dtype}")
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"target must be a tensor, not {type(target).__name__}")
    if logits.ndim < 1:
        raise ValueError("logits must have a class dimension")
    _check_device(logits)
    if target.device != logits.device:
        raise ValueError("target and logits must be on the same device")


def _labels_of(spec: _Spec, logits: torch.Tensor, target: torch.Tensor, dim: int) -> torch.Tensor:
    """The checked target of a loss: int64 class labels (logits.shape without dim; any integer
    dtype is cast) for the single-label losses, a boolean mask of logits.shape for the
    multilabel ones. A target of the other kind raises TypeError, a wrong shape ValueError."""
    if spec.multilabel:
        if target.dtype != torch.bool:
            raise TypeError(
                f"{spec.name} takes a boolean mask of the logits' shape, not {target.dtype} targets"
            )
        if target.shape != logits.shape:
            raise ValueError(
                f"target must have the logits' shape {tuple(logits.shape)}, "
                f"not {tuple(target.shape)}"
            )
        return target
    if target.dtype != torch.int64:
        if target.dtype == torch.bool or target.is_floating_point() or target.is_complex():
            hint = "; multilabel losses take boolean masks" if target.dtype == torch.bool else ""
            raise TypeError(f"{spec.name} takes integer class labels, not {target.dtype}{hint}")
        target = target.long()
    if target.ndim != logits.ndim - 1 or target.shape != _without(logits.shape, dim):
        raise ValueError(
            f"target must have shape {tuple(_without(logits.shape, dim))} (the logits' shape "
            f"without dim), not {tuple(target.shape)}"
        )
    return target


def _without(shape: torch.Size, dim: int) -> torch.Size:
    """shape without the dimension dim (ValueError if out of range)."""
    if not -len(shape) <= dim < len(shape):
        raise ValueError(f"dim {dim} is out of range for {len(shape)}-D logits")
    dim %= len(shape)
    return shape[:dim] + shape[dim + 1 :]


def _raise_failed(checks: list[tuple[torch.Tensor, str]]) -> None:
    """Raise ValueError for the first failed CPU value check."""
    failed = torch.stack([flag for flag, _ in checks]).tolist()
    for flag, (_, message) in zip(failed, checks, strict=False):
        if flag:
            raise ValueError(message)


def _target_mask(
    target: torch.Tensor, logits: torch.Tensor, dim: int, objective: str
) -> torch.Tensor:
    """Boolean positive-label mask (examples, classes) from int64 indices or a mask.

    On CPU, value checks and the finiteness of logits share one check. On CUDA,
    out-of-range targets leave an all-false row so invalid rows propagate NaN without
    synchronizing the device.
    """
    moved = logits.movedim(dim, -1).shape
    shape, classes = moved[:-1], moved[-1]
    index = False
    if target.dtype == torch.bool and target.shape == logits.shape:
        positive = target.movedim(dim, -1).reshape(-1, classes).contiguous()
    else:
        if target.dtype != torch.int64 or target.shape != shape:
            raise ValueError("target must contain int64 class indices or be a boolean mask")
        in_range = (target >= 0) & (target < classes)
        positive = torch.zeros((target.numel(), classes), dtype=torch.bool, device=logits.device)
        positive.scatter_(1, target.reshape(-1, 1).clamp(0, classes - 1), in_range.reshape(-1, 1))
        index = True
    if logits.is_cuda:
        return positive
    checks = [(~_finite(logits), "logits must be finite floating-point values")]
    if index:
        checks.append((((target < 0) | (target >= classes)).any(), "Target index out of range"))
    else:
        checks.append(((~positive.any(-1)).any(), "Every example needs a positive label"))
        if objective in _SINGLE_TARGET:
            rest = positive.scatter(-1, _first(positive), False)
            checks.append((rest.any(), "This loss needs exactly one target"))
    if objective not in _SINGLE_TARGET and objective != "cross_entropy":
        checks.append((positive.all(-1).any(), "This loss needs a negative label"))
    if not _ops.is_compiling():
        _raise_failed(checks)
    return positive


_TOPK = frozenset(("topk_cross_entropy", "topk_hinge_alpha", "topk_hinge_beta"))


class _Call(NamedTuple):
    """A loss with checked options: what a module holds, and a function checks per call."""

    spec: _Spec
    reduction: str
    dim: int
    backend: str
    precision: str | None
    ignore_index: int | None = None


# The _Call of each distinct set of options (all hashable), checked once.
_CALLS: dict[tuple[object, ...], _Call] = {}


def _bind(
    name: str,
    *,
    k: object = 1,
    gamma: object = 0.0,
    alpha: object = 1.0,
    ignore_index: object = None,
    reduction: object = "mean",
    dim: object = -1,
    backend: object = "auto",
    precision: object = None,
) -> _Call:
    """The checked _Call of loss name: TypeError for wrong types, ValueError for bad values."""
    key: tuple[object, ...] | None = (
        name,
        k,
        type(k),
        gamma,
        type(gamma),
        alpha,
        type(alpha),
        ignore_index,
        type(ignore_index),
        reduction,
        dim,
        type(dim),
        backend,
        precision,
    )
    try:
        call = _CALLS.get(key)  # type: ignore[arg-type]
    except TypeError:  # an unhashable option: checked below
        call, key = None, None
    if call is not None:
        return call
    spec = _opt.spec(
        name,
        _opt.check_k(k) if name in _TOPK else 1,
        _opt.check_gamma(gamma),
        _opt.check_positive(alpha, "alpha"),
    )
    call = _Call(
        spec,
        _opt.check_reduction(reduction),
        _opt.check_axis(dim, "dim"),
        _opt.check_backend(backend, BACKENDS),
        _precision(precision),
        _opt.check_ignore_index(ignore_index),
    )
    if key is not None and len(_CALLS) < 4096:
        _CALLS[key] = call
    return call


def _topk_hinge(variant: object) -> str:
    return "topk_hinge_" + _opt.check_variant(variant)


def _check_weight(weight: object, logits: torch.Tensor, classes: int) -> None:
    """A (classes,) floating-point tensor on the logits' device; its values are checked with
    the losses (_row_weights)."""
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be None or a tensor, not {type(weight).__name__}")
    if not weight.is_floating_point():
        raise TypeError(f"weight must be floating-point, not {weight.dtype}")
    if weight.shape != (classes,):
        raise ValueError(f"weight must have shape ({classes},), not {tuple(weight.shape)}")
    if weight.device != logits.device:
        raise ValueError("weight and logits must be on the same device")


def _loss(
    call: _Call, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """A named loss of checked options (_bind): per example (logits.shape without dim) or
    reduced."""
    if _has_dtensor(logits, target, weight):
        return _call_with_dtensors(_loss, call, logits, target, weight)
    _check_tensors(logits, target)
    target = _labels_of(call.spec, logits, target, call.dim)
    if weight is not None or call.ignore_index is not None:
        return _weighted(call, logits, target, weight)
    if _ops.transformed():
        return _transformed(call, logits, target)
    backend = call.backend
    if backend == "kernel":
        _require_kernel(_explain_loss(call, logits, target))
        backend = "auto"
    spec = call.spec
    return _resolved_loss(
        logits,
        target,
        spec.objective,
        backend,
        spec.k,
        spec.gamma,
        spec.alpha,
        call.reduction,
        call.dim,
        call.precision,
        spec.k,
    )


def _require_kernel(explained: tuple[str, str]) -> None:
    """NotImplementedError for backend="kernel" where no fused kernel computes the call."""
    backend, reason = explained
    if backend != "kernel":
        raise NotImplementedError(
            f"backend='kernel': no fused kernel computes this call ({reason}); "
            "backend='auto' takes that path"
        )


def _weighted(
    call: _Call, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None
) -> torch.Tensor:
    """A single-label loss with class weights or an ignored label, as torch's cross-entropy:
    w[label] times each example's loss (0 where ignored); the mean divides the sum by the sum
    of those weights. Ignored examples take label 0 and zero logits (so their values are
    neither read nor checked, and their gradient is zero)."""
    _without(logits.shape, call.dim)
    dim = call.dim % logits.ndim
    classes = logits.shape[dim]
    if weight is not None:
        _check_weight(weight, logits, classes)
    z = logits.movedim(dim, -1)
    ignored = None
    if call.ignore_index is not None:
        ignored = target == call.ignore_index
        z = z.masked_fill(ignored[..., None], 0)
        target = target.masked_fill(ignored, 0)
    rows = call._replace(reduction="none", dim=-1, ignore_index=None)
    values = _loss(rows, z, target)
    row_weight = _row_weights(weight, target, ignored, classes, values)
    weighted = values * row_weight
    if call.reduction == "none":
        return weighted
    if call.reduction == "sum":
        return weighted.sum()
    return weighted.sum() / row_weight.sum()


def _row_weights(
    weight: torch.Tensor | None,
    target: torch.Tensor,
    ignored: torch.Tensor | None,
    classes: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """w[label] per example in the values' dtype (1 without weights, 0 where ignored). Class
    weights must be finite and nonnegative: ValueError on the CPU, NaN rows elsewhere."""
    if weight is None:
        row = torch.ones_like(values)
    else:
        w = weight.to(values.dtype)
        valid = torch.isfinite(w).all() & (w >= 0).all()
        if w.is_cuda or _ops.is_compiling() or _ops.transformed():
            w = w.masked_fill(~valid, math.nan)
        elif not bool(valid):
            raise ValueError("weight must be finite and nonnegative")
        row = w[target.clamp(0, classes - 1)]
    return row if ignored is None else row.masked_fill(ignored, 0)


def _transformed(call: _Call, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """A named loss under torch.func transforms: the eager loss of the rows (_Transformed)."""
    spec = call.spec
    _without(logits.shape, call.dim)
    z = logits.movedim(call.dim, -1)
    shape, classes = z.shape[:-1], z.shape[-1]
    _opt.check_classes(spec.k, classes)
    flat = z.reshape(-1, classes)
    if spec.multilabel:
        rows = target.movedim(call.dim, -1).reshape(-1, classes)
    else:
        rows = target.reshape(-1)
    run = functools.partial(_rows_loss, call)
    value = _Transformed.apply(flat, rows, run)[0]
    return _reduce(value.reshape(shape), call.reduction)


def _rows_loss(call: _Call, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The eager per-example loss of (rows, classes) logits (_transformed)."""
    spec = call.spec
    backend = "auto" if call.backend == "kernel" else call.backend
    return _resolved_loss(
        logits,
        target,
        spec.objective,
        backend,
        spec.k,
        spec.gamma,
        spec.alpha,
        "none",
        -1,
        call.precision,
        spec.k,
    )


class _Guard(torch.autograd.Function):
    """The gradient of a first-order loss under torch.func transforms, linked to the logits
    through a node that raises when it is differentiated (a hessian, jacrev of a gradient)."""

    generate_vmap_rule = True

    @staticmethod
    def forward(gradient: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        return gradient.clone()

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        pass

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, ...]:
        raise RuntimeError(_FIRST_ORDER)


class _Transformed(torch.autograd.Function):
    """A first-order loss of (rows, classes) logits under torch.func transforms (grad, vjp,
    vmap, jacrev): forward runs the eager loss of the rows, run(logits, other), with its
    gradient; vmap merges the mapped dimension into the rows. Forward mode raises."""

    @staticmethod
    def forward(
        logits: torch.Tensor, other: torch.Tensor, run: Callable[..., torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            leaf = logits.detach().requires_grad_()
            value = run(leaf, other)
            if value.numel() == 0:
                return value.detach(), torch.zeros_like(leaf)
            (gradient,) = torch.autograd.grad(value, leaf, torch.ones_like(value))
        return value.detach(), gradient

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]) -> None:
        ctx.mark_non_differentiable(output[1])
        ctx.save_for_backward(output[1], inputs[0])

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        gradient, logits = ctx.saved_tensors
        upstream = grad_outputs[0]
        out = (gradient.to(upstream.dtype) * upstream[:, None]).to(logits.dtype)
        if torch.is_grad_enabled():
            out = _Guard.apply(out, upstream, logits)
        return out, None, None

    @staticmethod
    def jvp(ctx: Any, *tangents: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        raise RuntimeError(_FIRST_ORDER + " (nor forward mode: jvp, jacfwd, hessian)")

    @staticmethod
    def vmap(  # pyright: ignore[reportIncompatibleMethodOverride]
        info: Any,
        in_dims: tuple[int | None, ...],
        logits: torch.Tensor,
        other: torch.Tensor,
        run: Callable[..., torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
        size = info.batch_size

        def front(x: torch.Tensor, d: int | None) -> torch.Tensor:
            return x.movedim(d, 0) if d is not None else x.expand(size, *x.shape)

        z, o = front(logits, in_dims[0]), front(other, in_dims[1])
        rows = z.shape[1]
        value, gradient = _Transformed.apply(
            z.reshape(size * rows, z.shape[-1]), o.reshape(size * rows, *o.shape[2:]), run
        )
        return (value.reshape(size, rows), gradient.reshape(z.shape)), (0, 0)


def _resolved_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    objective: str,
    backend: str,
    k: int,
    gamma: float,
    alpha: float,
    reduction: str,
    dim: int,
    canonical: str | None,
    given_k: int,
) -> torch.Tensor:
    """sdca_loss of a tensor-backend kind, with checked tensors and canonical precision.

    k is the k of the kind; given_k, the k the caller passed, is the one checked."""
    canonical = _contract(logits, canonical)
    fast = _fast_cpu_loss(
        logits, target, objective, backend, k, gamma, alpha, reduction, dim, canonical, given_k
    )
    if fast is not None:
        return fast
    if logits.dtype == torch.float64 and canonical == "float32":
        raise ValueError("float64 logits require precision='float64' or None")
    # Avoid identity view nodes in the common (N,C) autograd graph.
    z = (
        logits
        if isinstance(dim, int) and not isinstance(dim, bool) and dim in (-1, logits.ndim - 1)
        else logits.movedim(dim, -1)
    )
    shape, classes = z.shape[:-1], z.shape[-1]
    _opt.check_classes(given_k, classes)
    if z.numel() == 0:
        return _empty_loss(z, reduction, _value_dtype(logits))
    if z.device.type == "cpu":
        plan = _cpu_plan(logits, shape, target, objective, backend, k, reduction, canonical)
        if plan is not None:
            return _run_cpu(plan, logits, z, target, dim, reduction, gamma, alpha)
    if backend == "native":
        raise ValueError("Native losses require CPU tensors; use backend='torch' on other devices")
    return _device_loss(
        logits, z, target, objective, k, gamma, alpha, reduction, dim, canonical, backend
    )


# The _cpu_plan of ordinary CPU calls by the options it depends on (_fast_cpu_loss).
_PLANS: dict[tuple[object, ...], _CpuPlan] = {}


def _fast_cpu_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    objective: str,
    backend: str,
    k: int,
    gamma: float,
    alpha: float,
    reduction: str,
    dim: int,
    canonical: str | None,
    given_k: int,
) -> torch.Tensor | None:
    """_resolved_loss of ordinary (N, C) CPU tensors with their _cpu_plan cached, or None:
    where it applies every check of _resolved_loss passes but the value checks, which the
    kernels make with the same messages."""
    if not _ordinary_cpu(logits, target, backend, canonical, dim, reduction):
        return None
    n, classes = logits.shape
    # _check_options' fast case.
    if not (
        n > 0
        and type(given_k) is int
        and 1 <= given_k < classes
        and type(gamma) is float
        and type(alpha) is float
        and 0.0 <= gamma < math.inf
        and 0.0 < alpha < math.inf
    ):
        return None
    indexed = target.dtype == torch.int64 and target.ndim == 1 and target.shape[0] == n
    if not (indexed or (target.dtype == torch.bool and target.shape == logits.shape)):
        return None
    wide = logits.dtype == torch.float64 and classes > _cpu.DOUBLE_CLASSES
    key = (objective, k, reduction, logits.dtype, indexed, wide, _cpu_entropy.enabled())
    plan = _PLANS.get(key)
    if plan is None:
        plan = _cpu_plan(logits, logits.shape[:-1], target, objective, backend, k, reduction, None)
        if plan is None:
            return None
        _PLANS[key] = plan
    return _run_cpu(plan, logits, logits, target, dim, reduction, gamma, alpha)


def _ordinary_cpu(
    logits: torch.Tensor,
    target: torch.Tensor,
    backend: str,
    canonical: str | None,
    dim: int,
    reduction: str,
) -> bool:
    """Whether a call has (N, C) CPU logits of a CPU kernel's dtype, a CPU target, the class
    dimension last and the defaults that _fast_cpu_loss plans for, outside a compiled graph."""
    return (
        type(logits) is torch.Tensor
        and type(target) is torch.Tensor
        and backend == "auto"
        and canonical is None
        and logits.ndim == 2
        and type(dim) is int
        and dim in (-1, 1)
        and logits.is_cpu
        and target.is_cpu
        and logits.dtype in _FAST_DTYPES
        and reduction in _cpu.REDUCTIONS
        and not _ops.is_compiling()
    )


def _hinge_fp32(logits: torch.Tensor, canonical: str | None, gamma: float, classes: int) -> bool:
    """Shared dispatch/diagnostics policy; the kernel guards each tile's score range, while
    host dispatch promotes when C * gamma can overflow the float32 Michelot sum."""
    return (
        canonical != "float64"
        and logits.dtype != torch.float64
        and (gamma == 0 or _float32_holds(logits, gamma, gamma * classes, check_spread=False))
    )


def _device_loss(
    logits: torch.Tensor,
    z: torch.Tensor,
    target: torch.Tensor,
    objective: str,
    k: int,
    gamma: float,
    alpha: float,
    reduction: str,
    dim: int,
    canonical: str | None,
    backend: str,
) -> torch.Tensor:
    """The fused CUDA kernels ("auto") or tensor operations ("torch", or beyond the kernels)
    on CPU or CUDA, of class-last scores z."""
    shape, classes = z.shape[:-1], z.shape[-1]
    fused = backend != "torch"
    # Fused CUDA entropy (integer targets or boolean masks) uses float32 exponentials + float64
    # row statistics by default for non-float64 logits, and full float64 when
    # precision="float64" or logits are float64.
    # CUDA k=1 hinges follow the requested work precision.
    work = _device_working(logits, z, canonical, objective, fused)
    indexed = fused and target.dtype == torch.int64
    labels_fit = target.shape == shape and reduction in ("none", "sum", "mean")
    if _indexed_entropy_path(target, objective, k):
        return _device_entropy(logits, z, target, reduction, canonical, fused)
    if indexed and k == 1 and objective in _hinge.SMOOTH and labels_fit:
        flat = _flat(z)
        if _hinge.covers(flat):
            value = _hinge.hinge(
                flat,
                _labels(target),
                objective,
                reduction,
                gamma,
                _hinge_fp32(logits, canonical, gamma, flat.shape[1]),
            )
            return value.reshape(shape) if reduction == "none" else value
    if _multilabel.applies(objective, logits, target, classes, backend, canonical, reduction):
        flat = _flat(z)
        mask = target.movedim(dim, -1).reshape(-1, classes).contiguous()
        value = _multilabel.loss(flat, mask, objective, reduction, gamma, canonical)
        return value.reshape(shape) if reduction == "none" else value
    if indexed and k > 1 and objective in _SINGLE_TARGET and labels_fit:
        flat = _flat(z)
        if _topk.covers(flat, objective, k):
            fp32 = _topk_fp32(logits, canonical)
            value = _topk.topk(flat, _labels(target), objective, reduction, k, gamma, fp32)
            return value.reshape(shape) if reduction == "none" else value
    positive = _target_mask(target, logits, dim, objective)
    flat = z.reshape(-1, classes).contiguous()
    single = target.dtype == torch.int64
    if objective == "cross_entropy":
        values = _TensorEntropy.apply(flat, positive, work == "float64")
    else:
        values = _Loss.apply(flat, positive, objective, k, gamma, alpha, work, single, fused)
    return _reduce(values.reshape(shape), reduction).to(_value_dtype(logits))


def _device_entropy(
    logits: torch.Tensor,
    z: torch.Tensor,
    target: torch.Tensor,
    reduction: str,
    canonical: str | None,
    allow_fused: bool = True,
) -> torch.Tensor:
    """Integer entropy of class-last scores z: fused CUDA kernels where allow_fused, else
    tensor operations."""
    shape, classes = z.shape[:-1], z.shape[-1]
    if target.shape != shape:
        raise ValueError("target must contain int64 class indices or be a boolean mask")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("reduction must be none, sum, or mean")
    flat = _flat(z)
    fused = allow_fused and _fused_entropy(flat)
    if not flat.is_cuda and not _ops.is_compiling():
        _raise_failed(
            [
                (~_finite(logits), "logits must be finite floating-point values"),
                (((target < 0) | (target >= classes)).any(), "Target index out of range"),
            ]
        )
    fp32 = fused and canonical != "float64" and logits.dtype != torch.float64
    value = _indexed_entropy(flat, _labels(target), reduction, fp32=fp32, is_fused=fused)
    return value.reshape(shape) if reduction == "none" else value


def _topk_fp32(logits: torch.Tensor, canonical: str | None) -> bool:
    """Whether the fused top-k entropy takes float32 exponentials, as the fused entropy: for
    non-float64 logits unless precision="float64"."""
    return canonical != "float64" and logits.dtype != torch.float64


def _flat(z: torch.Tensor) -> torch.Tensor:
    """Contiguous (examples, classes) class-last scores."""
    return (z if z.ndim == 2 else z.reshape(-1, z.shape[-1])).contiguous()


def _labels(target: torch.Tensor) -> torch.Tensor:
    """Contiguous (examples,) int64 targets."""
    return (target if target.ndim == 1 else target.reshape(-1)).contiguous()


class _CpuPlan(NamedTuple):
    """A CPU kernel call: the backend name and why, the kernel (_cpu.KERNELS) and its
    integer options, and whether it reduces itself."""

    backend: str
    reason: str
    kind: str
    ints: list[int]
    reduced: bool = True


def _cpu_plan(
    logits: torch.Tensor,
    shape: torch.Size,
    target: torch.Tensor,
    objective: str,
    backend: str,
    k: int,
    reduction: str,
    canonical: str | None,
) -> _CpuPlan | None:
    """The CPU kernel of a loss call (backend "auto" or "native") of logits whose shape
    without the class dimension is shape, or None (backend "torch") for tensor operations;
    the call that explain_backend reports."""
    if backend == "torch":
        return None
    if reduction not in _cpu.REDUCTIONS:
        raise ValueError("reduction must be none, sum, or mean")
    code = _cpu.REDUCTIONS[reduction]
    extension = backend == "auto" and _cpu_entropy.enabled()
    unavailable = ""
    if backend == "auto" and not extension:
        unavailable = f"; the CPU extension is unavailable ({_cpu_entropy.why_not()})"
    indexed, masked = _target_kind(target, logits, shape)
    if indexed and (objective == "cross_entropy" or (objective == "topk_cross_entropy" and k == 1)):
        classes = logits.numel() // max(1, shape.numel())
        return _cpu_entropy_plan(logits, classes, code, extension, canonical, unavailable)
    fp32 = _extension_float(objective, k, masked, canonical, logits)
    if extension and (canonical != "float32" or fp32):
        plan = _extension_plan(objective, k, code, indexed, masked, fp32)
        if plan is not None:
            return plan
    return _loss_plan(logits, objective, k, extension, canonical, unavailable)


def _target_kind(
    target: torch.Tensor, logits: torch.Tensor, shape: torch.Size
) -> tuple[bool, bool]:
    """(int64 indices, boolean mask) of a CPU loss call's target; ValueError for neither."""
    indexed = target.dtype == torch.int64 and target.shape == shape
    masked = target.dtype == torch.bool and target.shape == logits.shape
    if not (indexed or masked):
        raise ValueError("target must contain int64 class indices or be a boolean mask")
    return indexed, masked


def _extension_float(
    objective: str, k: int, masked: bool, canonical: str | None, logits: torch.Tensor
) -> bool:
    """Whether the extension takes float32 exponentials (no float64 work, as cross_entropy):
    top-k entropy rows of k <= 64, and by default boolean-mask entropy."""
    if canonical == "float64" or logits.dtype == torch.float64:
        return False
    if objective == "topk_cross_entropy":
        return k <= _TOPK_FLOAT_K
    return masked and objective == "cross_entropy" and canonical is None


def _loss_plan(
    logits: torch.Tensor,
    objective: str,
    k: int,
    extension: bool,
    canonical: str | None,
    unavailable: str,
) -> _CpuPlan:
    """The generic loss kernel: the extension's proximal_cross_entropy, else the native
    library."""
    work = _working(logits, canonical, objective == "proximal_cross_entropy" and extension)
    working = _tensor_arithmetic(objective, logits, work)
    if objective == "proximal_cross_entropy" and extension:
        implementation = _cpu.EXTENSION
        plan = ("cpu-kernel", f"the CPU extension's proximal_cross_entropy kernel ({working})")
    else:
        implementation = _cpu.NATIVE
        plan = ("native", f"the native library ({working}){unavailable}")
    ints = [_cpu.OBJECTIVES.index(objective), k, implementation, _cpu.PRECISIONS.index(work)]
    return _CpuPlan(*plan, "loss", ints, reduced=False)


# The largest k of the extension's float32 top-k entropy rows (kFloatTopK there).
_TOPK_FLOAT_K = 64


def _cpu_entropy_plan(
    logits: torch.Tensor,
    classes: int,
    code: int,
    extension: bool,
    canonical: str | None,
    unavailable: str,
) -> _CpuPlan:
    """Integer entropy on the CPU: the extension, or the native library."""
    fp32 = canonical != "float64" and logits.dtype != torch.float64
    arithmetic = "float32 exponentials, float64 row statistics" if fp32 else "float64"
    implementation = _cpu.entropy_implementation(logits, extension, classes)
    ints = [code, _cpu.NO_IGNORE, int(fp32), implementation]
    if implementation == _cpu.EXTENSION:
        return _CpuPlan("cpu-kernel", f"the CPU extension ({arithmetic})", "entropy", ints)
    return _CpuPlan("native", f"the native library ({arithmetic}){unavailable}", "entropy", ints)


def _extension_plan(
    objective: str, k: int, code: int, indexed: bool, masked: bool, fp32: bool
) -> _CpuPlan | None:
    """The kernels of the extension: k = 1 hinges, top-k losses and boolean masks, in float64
    but for fp32 top-k and boolean-mask entropy (float32 exponentials, float64 row
    statistics)."""
    why = "the CPU extension's {} kernels (float64)"
    if indexed and k == 1 and objective in _hinge.SMOOTH:
        ints = [code, int(_hinge.SMOOTH[objective])]
        return _CpuPlan("cpu-kernel", why.format("k = 1 hinge"), "hinge", ints)
    if indexed and k > 1 and objective in _topk.LOSSES:
        ints = [code, _topk.LOSSES[objective], k, int(fp32)]
        if fp32:
            why = "the CPU extension's {} kernels (float32 exponentials, float64 row statistics)"
        return _CpuPlan("cpu-kernel", why.format("top-k"), "topk", ints)
    if masked and objective in _multilabel.OBJECTIVES:
        fp32 = fp32 and objective == "cross_entropy"
        ints = [code, _multilabel.OBJECTIVE_IDS[objective], int(fp32)]
        if fp32:
            why = "the CPU extension's {} kernels (float32 exponentials, float64 row statistics)"
        return _CpuPlan("cpu-kernel", why.format("boolean-mask"), "multilabel", ints)
    return None


def _run_cpu(
    plan: _CpuPlan,
    logits: torch.Tensor,
    z: torch.Tensor,
    target: torch.Tensor,
    dim: int,
    reduction: str,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    """The loss of a CPU kernel plan of class-last scores z; the kernels check the values."""
    shape, classes = z.shape[:-1], z.shape[-1]
    if target.dtype == torch.bool:
        mask = target if target.ndim == 2 and dim in (-1, 1) else target.movedim(dim, -1)
        labels = (mask if mask.ndim == 2 else mask.reshape(-1, classes)).contiguous()
    else:
        labels = _labels(target)
    floats = [gamma, alpha] if plan.kind == "loss" else [gamma]
    value = _cpu.loss(plan.kind, _flat(z), labels, None, plan.ints, floats)
    if plan.reduced:
        return value.reshape(shape) if reduction == "none" else value
    return _reduce(value.reshape(shape), reduction).to(_value_dtype(logits))


def _indexed_entropy_path(target: torch.Tensor, objective: str, k: int) -> bool:
    """Integer-target entropy (or top-1 entropy): the fused float32 kernels where they apply,
    else float64 (whose storage-limit guarantees the float32 tensor entropy lacks)."""
    return target.dtype == torch.int64 and (
        objective == "cross_entropy" or (objective == "topk_cross_entropy" and k == 1)
    )


def _fused_cuda(launcher: tuple[object, object], detail: str = "") -> tuple[str, str]:
    host, reason = launcher
    detail = f" ({detail})" if detail else ""
    if host is not None:
        return "cuda-launcher", f"fused Triton kernels launched from C++ autograd{detail}"
    return "triton", f"fused Triton kernels launched by Triton{detail}: {reason}"


def _explain_topk(
    flat: torch.Tensor, objective: str, k: int, reduction: str, indexed: bool, fp32: bool
) -> tuple[str, str] | None:
    """The backend of CUDA integer-target top-k losses at k > 1; None for other calls."""
    if not (indexed and k > 1 and objective in _SINGLE_TARGET):
        return None
    if reduction in ("none", "sum", "mean") and _topk.covers(flat, objective, k):
        kernel = _topk._kernel()
        assert kernel is not None
        launch = kernel.launcher(flat, kernel._REDUCTIONS[reduction], objective, k, fp32)
        return _fused_cuda(launch, kernel.describe(flat, objective, fp32))
    why = "k <= 32 (topk_cross_entropy: k < classes - 1) and 2**31 logits"
    return "torch", f"tensor operations: the fused top-k kernels need Triton 3.3, {why}"


def _unfused(objective: str, hinge: bool, target: torch.Tensor) -> str:
    """Why a call takes the tensor operations; hinge: a k = 1 hinge the CUDA kernels skip."""
    if hinge:
        return "the fused k = 1 hinges need Triton 3.3 and at most 2**28 rows"
    if objective in _hinge.SMOOTH or objective == "topk_cross_entropy":
        return "fused paths need int64 targets and k = 1 (on CUDA also k > 1)"
    return f"{objective} has no fused integer-target path"


def _explain_loss(call: _Call, logits: torch.Tensor, target: torch.Tensor) -> tuple[str, str]:
    """(backend, reason) of a named loss with checked tensors and target (_loss), in the
    names explain_backend reports."""
    name, reason = _explain_route(call, logits, target)
    return _PUBLIC.get(name, name), reason


def _explain_route(call: _Call, logits: torch.Tensor, target: torch.Tensor) -> tuple[str, str]:
    """The internal route (cpu-kernel, cuda-launcher, triton, native or torch) of _loss."""
    spec = call.spec
    objective, k, gamma, reduction = spec.objective, spec.k, spec.gamma, call.reduction
    backend = "auto" if call.backend == "kernel" else call.backend
    if logits.dtype == torch.float64 and call.precision == "float32":
        raise ValueError("float64 logits require precision='float64' or None")
    canonical = _contract(logits, call.precision)
    _without(logits.shape, call.dim)
    z = logits.movedim(call.dim, -1)
    shape, classes = z.shape[:-1], z.shape[-1]
    _opt.check_classes(k, classes)
    flat = z.reshape(-1, classes)
    work = _device_working(logits, flat, canonical, objective, backend != "torch")
    if flat.device.type == "cpu":
        plan = _cpu_plan(logits, shape, target, objective, backend, k, reduction, canonical)
        if plan is not None:
            return plan.backend, plan.reason
    if backend == "native":
        raise ValueError("Native losses require CPU tensors; use backend='torch' on other devices")
    if backend == "torch":
        return _explain_torch(z, flat, target, objective, k, work)
    indexed = target.dtype == torch.int64 and target.shape == shape
    if target.shape == shape and _indexed_entropy_path(target, objective, k):
        return _explain_entropy(logits, flat, canonical, reduction)
    fast = indexed and k == 1 and objective in _hinge.SMOOTH
    if fast and _hinge.covers(flat):
        return _explain_hinge(logits, flat, objective, gamma, reduction, canonical)
    reason = _multilabel.explain(objective, logits, target, classes, backend, canonical, reduction)
    if reason is not None:
        return reason
    fp32_topk = _topk_fp32(logits, canonical)
    explained = (
        _explain_topk(flat, objective, k, reduction, indexed, fp32_topk) if flat.is_cuda else None
    )
    if explained is not None:
        return explained
    return _explain_tensor(objective, flat, work, fast and flat.is_cuda, target)


def _explain_entropy(
    logits: torch.Tensor, flat: torch.Tensor, canonical: str | None, reduction: str
) -> tuple[str, str]:
    """_explain_route of integer targets on the entropy path: the fused CUDA kernel or the
    tensor operations."""
    if _fused_entropy(flat):
        kernel = _entropy._kernel()
        assert kernel is not None
        fp32_entropy = canonical != "float64" and logits.dtype != torch.float64
        launch = kernel.launcher(flat, kernel._REDUCTIONS[reduction], fp32_entropy)
        return _fused_cuda(launch, kernel.describe(flat.shape[1], fp32_entropy))
    why = "needs Triton 3.3, at most 2**28 rows and 2**25 classes" if flat.is_cuda else "CPU"
    return "torch", f"integer entropy in float64 tensor operations ({why})"


def _explain_hinge(
    logits: torch.Tensor,
    flat: torch.Tensor,
    objective: str,
    gamma: float,
    reduction: str,
    canonical: str | None,
) -> tuple[str, str]:
    """_explain_route of a k = 1 hinge the fused CUDA kernels cover."""
    kernel = _hinge._kernel()
    assert kernel is not None
    code = kernel._REDUCTIONS[reduction]
    fp32 = _hinge_fp32(logits, canonical, gamma, flat.shape[1])
    return _fused_cuda(
        kernel.launcher(flat, code, _hinge.SMOOTH[objective], fp32),
        "float32" if fp32 else "float64",
    )


def _explain_tensor(
    objective: str, flat: torch.Tensor, work: str | None, hinge: bool, target: torch.Tensor
) -> tuple[str, str]:
    """_explain_route without a fused path: proximal_cross_entropy's Triton kernel or the
    tensor operations (hinge: a k = 1 hinge the CUDA kernels skip)."""
    arithmetic = _tensor_arithmetic(objective, flat, work)
    if objective == "proximal_cross_entropy" and _prox_kernel(flat) is not None:
        return (
            "triton",
            f"proximal_cross_entropy runs one Triton kernel (Python launch, {arithmetic})",
        )
    why = _unfused(objective, hinge, target)
    return "torch", f"tensor operations in {arithmetic}: {why}"


@overload
def _stand_in(x: torch.Tensor) -> torch.Tensor: ...
@overload
def _stand_in(x: torch.Tensor | None) -> torch.Tensor | None: ...
def _stand_in(x: torch.Tensor | None) -> torch.Tensor | None:
    """The full (gathered) tensor's stand-in for explaining a DTensor call: an uninitialized
    tensor of its global shape on the local device; other values unchanged."""
    if x is not None and _has_dtensor(x):
        return torch.empty(x.shape, dtype=x.dtype, device=x.device)
    return x


def _gathered(explained: tuple[str, str]) -> tuple[str, str]:
    name, reason = explained
    return name, f"DTensor inputs, gathered in full on every rank: {reason}"


def _explain_call(
    call: _Call, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None
) -> tuple[str, str]:
    """explain_backend of _loss(call, logits, target, weight)."""
    if _has_dtensor(logits, target, weight):
        _dtensor_mesh(logits, target, weight)
        stand = _stand_in(logits), _stand_in(target), _stand_in(weight)
        return _gathered(_explain_call(call, *stand))
    _check_tensors(logits, target)
    target = _labels_of(call.spec, logits, target, call.dim)
    if weight is None and call.ignore_index is None:
        return _explain_loss(call, logits, target)
    _without(logits.shape, call.dim)
    if weight is not None:
        _check_weight(weight, logits, logits.shape[call.dim])
    if call.ignore_index is not None:
        target = target.masked_fill(target == call.ignore_index, 0)
    rows = call._replace(reduction="none", ignore_index=None)
    name, reason = _explain_loss(rows, logits, target)
    return name, f"{reason}; class weights and ignore_index weigh its per-example losses"


def explain_backend(
    loss: Callable[..., torch.Tensor], /, *args: Any, **kwargs: Any
) -> tuple[str, str]:
    """(backend, reason): the path that loss(*args, **kwargs) takes, without computing it.

    loss is a function of libsdca.torch (topk_hinge_loss, cross_entropy, ...) or one of its
    loss modules (TopKHingeLoss(k=2), ...), called with these arguments. The backend is a
    value of the backend option that takes this path: "kernel" (the CPU extension's kernels
    or fused CUDA kernels; the reason says which, and how they launch), "native" (the
    native library) or "torch" (tensor operations). DTensor inputs report the path of the
    gathered full inputs. Like the first such call, it may build the CPU extension or the
    CUDA launcher and compile kernels.

    Raises:
        TypeError: If loss is not a libsdca.torch loss, or for arguments the call rejects.
        ValueError: For arguments the call rejects.
    """
    if isinstance(loss, nn.Module):
        explain = getattr(loss, "_explain", None)
        if explain is None:
            raise TypeError(f"{type(loss).__name__} is not a libsdca.torch loss module")
        explained: tuple[str, str] = explain(*args, **kwargs)
        return explained
    explain = _EXPLAINERS.get(loss) if callable(loss) else None
    if explain is None:
        raise TypeError(f"{loss!r} is not a libsdca.torch loss function")
    bound = inspect.signature(loss).bind(*args, **kwargs)
    bound.apply_defaults()
    explained = explain(**bound.arguments)
    return explained


def _explain_torch(
    z: torch.Tensor,
    flat: torch.Tensor,
    target: torch.Tensor,
    objective: str,
    k: int,
    work: str | None,
) -> tuple[str, str]:
    """explain_backend of backend="torch": the tensor operations on every device."""
    entropy = _indexed_entropy_path(target, objective, k)
    if target.shape == z.shape[:-1] and entropy:
        return "torch", "integer entropy in float64 tensor operations (backend='torch')"
    arithmetic = _tensor_arithmetic(objective, flat, work)
    return "torch", f"tensor operations in {arithmetic} (backend='torch')"


def _tensor_arithmetic(objective: str, logits: torch.Tensor, work: str | None) -> str:
    """The arithmetic of _per_example for these logits and working precision."""
    if objective == "cross_entropy":
        return "float64" if work == "float64" or logits.dtype == torch.float64 else "float32"
    if work == "float32":
        return "float32, or float64 where float32 cannot hold gamma, alpha or a row's spread"
    return "float64"


@functools.cache
def _kernel_distill(proximal: bool = False) -> Callable[..., Any] | None:
    """The Triton distillation_loss (or proximal_distillation_loss), or None without
    Triton >= 3.3 or an NVIDIA GPU."""
    if torch.version.cuda is None:
        return None
    try:
        import triton

        from . import _triton_distillation as kernels
    except ImportError:
        return None
    if not _cuda_launch.kernels_supported(triton.__version__):
        return None
    return kernels.proximal_distillation if proximal else kernels.distillation_loss


def _two_difference(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """a - b and its rounding error (TwoSum; the error is 0 where the difference is not
    finite)."""
    d = a - b
    rounded = d - a
    error = (a - (d - rounded)) + (-b - rounded)
    return d, torch.where(d.isfinite(), error, 0.0)


def _difference(
    cs: torch.Tensor, ct: torch.Tensor, student_max: torch.Tensor, teacher_max: torch.Tensor
) -> torch.Tensor:
    """(cs - ct) - (student_max - teacher_max), nearly exactly."""
    d, error = _two_difference(cs, ct)
    offset, offset_error = _two_difference(student_max, teacher_max)
    return (d - offset) + (error - offset_error)


def _distillation_stats(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
    double: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Shared temperature-scaled log-probabilities, probabilities, log-ratio, and difference q - p."""
    cs = _distillation_work(student, temperature, double)
    ct = _distillation_work(teacher, temperature, double)
    invalid = ~(cs.isfinite().all(-1, keepdim=True) & ct.isfinite().all(-1, keepdim=True))
    # Center before scaling: a small temperature would overflow the logits.
    student_max, teacher_max = cs.amax(-1, keepdim=True), ct.amax(-1, keepdim=True)
    # The represented differences relative to the common offset of the maxima:
    # (cs - ct) - (student_max - teacher_max) with the rounding errors of both differences,
    # so neither a large common offset (logits near 1e4) nor a small difference far from
    # the maximum rounds away.
    delta = _difference(cs, ct, student_max, teacher_max)
    s = _temperature_difference(cs, student_max, temperature)
    t = _temperature_difference(ct, teacher_max, temperature)
    maximum, _, lognorm = _normalizer(s)
    _, _, teacher_lognorm = _normalizer(t)
    log_q = (s - maximum) - lognorm
    log_p = t - teacher_lognorm
    p, q = torch.softmax(t, -1), torch.softmax(s, -1)
    value = torch.where(p > 0, p * ((maximum - s) + lognorm), 0.0).sum(-1)
    # Barron's log-ratio formula must select its fallback per class: a
    # distant negligible class must not erase small differences elsewhere.
    p64, d64 = p.double(), delta.double()
    mean_delta = torch.where(p > 0, p64 * d64, 0.0).sum(-1, keepdim=True) / p64.sum(
        -1, keepdim=True
    )
    dn64 = _over(d64 - mean_delta, temperature)
    shift = _over(-mean_delta, temperature).to(cs.dtype) - teacher_lognorm
    if temperature > 1:
        # An overflowing raw teacher/student offset can hide a small
        # difference between their distributions. Centered scaled scores
        # remove that common offset before the cancellation correction.
        relative_delta = s.double() - t.double()
        relative_mean = torch.where(p > 0, p64 * relative_delta, 0.0).sum(
            -1, keepdim=True
        ) / p64.sum(-1, keepdim=True)
        overflow = ~mean_delta.isfinite()
        dn64 = torch.where(overflow, relative_delta - relative_mean, dn64)
        shift = torch.where(overflow, (-relative_mean).to(cs.dtype) - teacher_lognorm, shift)
    dn = dn64.to(cs.dtype)
    # exp(log(p) + d) = exp(s + shift), expressed using student logits
    # so a huge teacher tail cannot cancel d. This also works when p
    # underflows or expm1(d) alone overflows.
    correction = torch.where(
        dn.abs() < 1,
        p * torch.expm1(dn.clamp(-1, 1)),
        (s + shift).exp() - p,
    ).sum(-1, keepdim=True, dtype=torch.float64)
    logratio = (dn64 - torch.log1p(correction)).to(cs.dtype)
    u = torch.where(
        logratio.abs() < 1,
        p * torch.expm1(logratio.clamp(-1, 1)),
        q - p,
    )
    return value, log_p, log_q, p, q, logratio, u, invalid


def _distillation_rows(
    student: torch.Tensor, teacher: torch.Tensor, temperature: float, double: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """distillation_loss per row (in the working dtype) and its gradient (saved dtype) by
    tensor operations."""
    value, _, _, _, q, _, u, invalid = _distillation_stats(student, teacher, temperature, double)
    gradient = u - q * u.sum(-1, keepdim=True, dtype=torch.float64).to(u.dtype)
    gradient = _over(gradient, temperature).masked_fill(invalid, math.nan)
    value = value.masked_fill(invalid.squeeze(-1), math.nan).to(_value_dtype(student))
    return value, gradient.to(_cpu.saved_dtype(student))


def _proximal_distillation_rows(
    student: torch.Tensor, teacher: torch.Tensor, temperature: float, alpha: float, double: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """proximal_distillation_loss per row and its gradient (saved dtype) by tensor operations."""
    tiny = torch.finfo(torch.float32).tiny
    force_double = double or not (tiny <= alpha <= 1 / tiny)
    _, log_p, log_q, p, q, logratio, u, invalid = _distillation_stats(
        student, teacher, temperature, force_double
    )
    value, grad_unscaled = _proximal_distillation(log_p, log_q, p, q, logratio, u, alpha, invalid)
    gradient = _over(grad_unscaled, temperature).to(_cpu.saved_dtype(student))
    return value.to(_value_dtype(student)), gradient


class _Distillation(_SavedGradient):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        student: torch.Tensor,
        teacher: torch.Tensor,
        temperature: float,
        double: bool,
    ) -> torch.Tensor:
        value, gradient = _distillation_rows(student, teacher, temperature, double)
        ctx.save_for_backward(gradient, student)
        return value


class _ProximalDistillation(_SavedGradient):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        student: torch.Tensor,
        teacher: torch.Tensor,
        temperature: float,
        alpha: float,
        double: bool,
        fused: bool = True,
    ) -> torch.Tensor:
        kernel = _kernel_distill(True) if fused and student.is_cuda else None
        if kernel is not None:
            kernel = _ops.proximal_distillation if _ops.compiling() else kernel
            value, gradient = kernel(
                student.contiguous(), teacher.contiguous(), temperature, alpha, double
            )
        else:
            value, gradient = _proximal_distillation_rows(
                student, teacher, temperature, alpha, double
            )
        ctx.save_for_backward(gradient, student)
        return value


def _fused_distillation(
    student: torch.Tensor, teacher: torch.Tensor, temperature: float, double: bool, reduction: str
) -> torch.Tensor | None:
    """The fused CUDA distillation_loss of (N, C) logits, or None where it does not run."""
    kernel = _kernel_distill() if student.is_cuda else None
    if kernel is None or student.shape[0] > _cuda_launch.ROWS:
        return None
    if _ops.compiling():
        kernel = _ops.distillation
    return kernel(student.contiguous(), teacher.contiguous(), temperature, double, reduction)


def _distillation_work(z: torch.Tensor, temperature: float, double: bool) -> torch.Tensor:
    """The working logits: float64 if double or where the dtype cannot hold T and 1/T."""
    work = _work(z, double)
    tiny = torch.finfo(work.dtype).tiny
    return work if tiny <= temperature <= 1 / tiny else work.double()


def _temperature_difference(
    logits: torch.Tensor, origin: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Center then scale, rescuing gaps that overflow before a large temperature."""
    difference = logits - origin
    scaled = _over(difference, temperature)
    if temperature > 1:
        separated = _over(logits, temperature) - _over(origin, temperature)
        return torch.where(difference.isfinite(), scaled, separated)
    return scaled


class _Distill(NamedTuple):
    """A distillation loss with checked options (alpha None: distillation_loss)."""

    temperature: float
    alpha: float | None
    reduction: str
    dim: int
    backend: str
    precision: str | None


def _bind_distillation(
    temperature: object,
    alpha: object,
    reduction: object,
    dim: object,
    backend: object,
    precision: object,
) -> _Distill:
    return _Distill(
        _opt.check_positive(temperature, "temperature"),
        None if alpha is None else _opt.check_positive(alpha, "alpha"),
        _opt.check_reduction(reduction),
        _opt.check_axis(dim, "dim"),
        _opt.check_backend(backend, BACKENDS),
        _precision(precision),
    )


def distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    temperature: float = 1.0,
    reduction: str = "mean",
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """Cross-entropy with a fixed teacher distribution softmax(teacher / T).

    Both arguments are LOGITS. The teacher is detached by definition. No T² multiplier is
    applied. This first-order gradient is exactly zero at equal represented logits; near
    equality it avoids probability subtraction.

    Args:
        student: Floating-point logits with the class dimension dim (at least two classes;
            no examples give the torch.nn.functional results).
        teacher: Logits of the same shape and device; logits of another floating-point
            dtype are promoted with the student's (torch.promote_types).
        temperature: T, finite and positive.
        reduction: "none", "sum" or "mean".
        dim: Class dimension of the logits.
        backend: "auto" (the CPU extension, else the native library, on the CPU; fused
            kernels on CUDA; else tensor operations), "kernel" (as "auto", raising
            NotImplementedError where no fused kernel runs), "native" (the native library,
            CPU tensors) or "torch" (tensor operations).
        precision: None or "float32" (float32 arithmetic for non-float64 logits) or
            "float64" (or torch.float32, torch.float64). Temperatures float32 cannot hold
            use float64.

    Returns:
        Per-example losses (student.shape without dim) or their reduction.

    Raises:
        TypeError: For arguments of the wrong type.
        ValueError: For invalid shapes or options, and on the CPU for nonfinite logits.
    """
    options = _bind_distillation(temperature, None, reduction, dim, backend, precision)
    return _distill(options, student, teacher)


def proximal_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    temperature: float = 1.0,
    alpha: float = 1.0,
    reduction: str = "mean",
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """Moreau envelope of distillation cross-entropy with step alpha and temperature T.

    Computes min_u { H(softmax(teacher/T), softmax(u)) + |u - student/T|^2 / (2 alpha) }.
    As alpha -> 0+, this converges to distillation_loss(student, teacher, temperature=T).
    The first-order gradient is identically zero at equal represented student and teacher
    logits. alpha is finite and positive; other arguments, the result and errors as in
    distillation_loss.
    """
    options = _bind_distillation(temperature, alpha, reduction, dim, backend, precision)
    return _distill(options, student, teacher)


def _distill(options: _Distill, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """distillation_loss or proximal_distillation_loss of checked options."""
    fast = _fast_distillation(options, student, teacher)
    if fast is not None:
        return fast
    if _has_dtensor(student, teacher):
        return _call_with_dtensors(_distill, options, student, teacher)
    student, teacher = _check_distillation(student, teacher, options)
    if _ops.transformed():
        return _transformed_distillation(options, student, teacher)
    if options.backend == "kernel":
        _require_kernel(_explain_distillation_route(options, student, teacher))
        options = options._replace(backend="auto")
    return _distillation(options, student, teacher)


_FAST_DTYPES = (torch.float32, torch.bfloat16, torch.float16, torch.float64)


def _fast_distillation(
    options: _Distill, student: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor | None:
    """_cpu_distillation of ordinary (N, C) CPU tensors, or None: where it applies every check
    of distillation_loss passes but finiteness, which the kernels check."""
    if (
        options.alpha is None
        and options.backend == "auto"
        and type(student) is torch.Tensor
        and type(teacher) is torch.Tensor
        and student.is_cpu
        and student.ndim == 2
        and options.dim in (-1, 1)
        and student.dtype in _FAST_DTYPES
        and (
            options.precision is None
            or (options.precision == "float32" and student.dtype != torch.float64)
        )
        and teacher.dtype == student.dtype
        and teacher.is_cpu
        and teacher.shape == student.shape
        and student.shape[0] > 0
        and student.shape[1] >= 2
        and not _ops.is_compiling()
        and not _ops.transformed()
    ):
        return _cpu_distillation(
            student, teacher.detach(), options.temperature, None, False, options.reduction
        )
    return None


def _check_distillation(
    student: torch.Tensor, teacher: torch.Tensor, options: _Distill
) -> tuple[torch.Tensor, torch.Tensor]:
    """Checked logits of a distillation loss, promoted to one dtype."""
    for name, t in (("student", student), ("teacher", teacher)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, not {type(t).__name__}")
        if not t.is_floating_point():
            raise TypeError(f"{name} logits must be floating-point, not {t.dtype}")
    if student.shape != teacher.shape:
        raise ValueError(
            f"student and teacher must have one shape, not {tuple(student.shape)} and "
            f"{tuple(teacher.shape)}"
        )
    if student.device != teacher.device:
        raise ValueError("student and teacher must be on the same device")
    if student.ndim < 1:
        raise ValueError("logits must have a class dimension")
    _check_device(student)
    _without(student.shape, options.dim)
    if student.dtype != teacher.dtype:
        common = torch.promote_types(student.dtype, teacher.dtype)
        student, teacher = student.to(common), teacher.to(common)
    if student.dtype == torch.float64 and options.precision == "float32":
        raise ValueError("float64 logits require precision='float64' or None")
    return student, teacher


def _cpu_distillation(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
    alpha: float | None,
    double: bool,
    reduction: str,
    native: bool = False,
) -> torch.Tensor:
    """The distillation loss of (N, C) CPU logits, reduced, by the extension (unless native)
    or the native library."""
    # Both check finiteness in their own passes.
    extension = not native and _cpu_entropy.enabled()
    implementation = _cpu.EXTENSION if extension else _cpu.NATIVE
    ints = [implementation, int(double), int(alpha is not None), _cpu.REDUCTIONS[reduction]]
    floats = [temperature, 1.0 if alpha is None else alpha]
    kind = "distillation_stats" if alpha is None else "distillation"
    if (
        alpha is None
        and implementation == _cpu.EXTENSION
        and torch.is_grad_enabled()
        and student.requires_grad
        and _cpu.distillation_grad(student, temperature, double)
    ):
        kind = "distillation_grad"
    return _cpu.loss(kind, student.contiguous(), teacher.contiguous(), None, ints, floats)


def _distillation(options: _Distill, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """distillation_loss, or proximal_distillation_loss with step alpha, of checked logits."""
    temperature, alpha, reduction, dim = options[:4]
    double = options.precision == "float64"
    backend = options.backend
    s, t = student, teacher.detach()
    if dim not in (-1, s.ndim - 1):
        s, t = s.movedim(dim, -1), t.movedim(dim, -1)
    if s.shape[-1] < 2:
        raise ValueError("At least two classes are required")
    shape, dtype = s.shape[:-1], _value_dtype(student)
    if s.numel() == 0:
        return _empty_loss(s, reduction, dtype)
    s2, t2 = (s, t) if s.ndim == 2 else (s.reshape(-1, s.shape[-1]), t.reshape(-1, t.shape[-1]))
    if student.device.type == "cpu" and backend != "torch":
        loss = _cpu_distillation(s2, t2, temperature, alpha, double, reduction, backend == "native")
        return loss.reshape(shape) if reduction == "none" else loss
    if backend == "native":
        raise ValueError("Native losses require CPU tensors; use backend='torch' on other devices")
    fused = backend == "auto"
    if alpha is None:
        if fused:
            value = _fused_distillation(s2, t2, temperature, double, reduction)
            if value is not None:
                return value.reshape(shape) if reduction == "none" else value
        values = _Distillation.apply(s2, t2, temperature, double)
    else:
        values = _ProximalDistillation.apply(s2, t2, temperature, alpha, double, fused)
    return _reduce(values.reshape(shape), reduction)


def _transformed_distillation(
    options: _Distill, student: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    """A distillation loss under torch.func transforms: the eager loss of the rows."""
    s = student.movedim(options.dim, -1)
    t = teacher.detach().movedim(options.dim, -1)
    shape, classes = s.shape[:-1], s.shape[-1]
    if classes < 2:
        raise ValueError("At least two classes are required")
    rows = options._replace(reduction="none", dim=-1)
    if rows.backend == "kernel":
        rows = rows._replace(backend="auto")
    run = functools.partial(_distillation, rows)
    value = _Transformed.apply(s.reshape(-1, classes), t.reshape(-1, classes), run)[0]
    return _reduce(value.reshape(shape), options.reduction)


def _explain_distillation(
    options: _Distill, student: torch.Tensor, teacher: torch.Tensor
) -> tuple[str, str]:
    """explain_backend of a distillation loss."""
    if _has_dtensor(student, teacher):
        _dtensor_mesh(student, teacher)
        stand = _stand_in(student), _stand_in(teacher)
        return _gathered(_explain_distillation(options, *stand))
    student, teacher = _check_distillation(student, teacher, options)
    return _explain_distillation_route(options, student, teacher)


def _explain_distillation_route(
    options: _Distill, student: torch.Tensor, teacher: torch.Tensor
) -> tuple[str, str]:
    """The path of _distillation (backend "kernel" as "auto")."""
    backend = "auto" if options.backend == "kernel" else options.backend
    proximal = options.alpha is not None
    name = "proximal_distillation" if proximal else "distillation"
    if student.shape[options.dim] < 2:
        raise ValueError("At least two classes are required")
    work = (
        "float64" if options.precision == "float64" or student.dtype == torch.float64 else "float32"
    )
    if student.device.type == "cpu" and backend != "torch":
        if backend == "auto" and _cpu_entropy.enabled():
            return "kernel", f"the CPU extension's {name} kernel ({work})"
        why = (
            ""
            if backend == "native"
            else f"; the CPU extension is unavailable ({_cpu_entropy.why_not()})"
        )
        return "native", f"the native library ({work}){why}"
    if backend == "native":
        raise ValueError("Native losses require CPU tensors; use backend='torch' on other devices")
    rows = student.numel() // student.shape[options.dim]
    if backend == "auto" and student.is_cuda:
        if _kernel_distill(proximal) is not None and (proximal or rows <= _cuda_launch.ROWS):
            return "kernel", f"fused Triton {name} kernels ({work})"
        return "torch", f"tensor operations ({work}): the fused {name} kernels need Triton 3.3"
    why = "backend='torch'" if backend == "torch" else "no fused kernel off CUDA"
    return "torch", f"tensor operations ({work}): {why}"


class DistillationLoss(nn.Module):
    """distillation_loss as a module; see it for the arguments."""

    temperature: float
    reduction: str
    dim: int
    backend: str
    precision: object

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__()
        _bind_distillation(temperature, None, reduction, dim, backend, precision)
        self.temperature = temperature
        self.reduction = reduction
        self.dim = dim
        self.backend = backend
        self.precision = precision

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """distillation_loss(student, teacher) with the stored options."""
        return distillation_loss(student, teacher, **self._options())

    def _options(self) -> dict[str, Any]:
        return dict(
            temperature=self.temperature,
            reduction=self.reduction,
            dim=self.dim,
            backend=self.backend,
            precision=self.precision,
        )

    def _explain(self, student: torch.Tensor, teacher: torch.Tensor) -> tuple[str, str]:
        return explain_backend(distillation_loss, student, teacher, **self._options())

    def extra_repr(self) -> str:
        return _repr(self._options())


class ProximalDistillationLoss(nn.Module):
    """proximal_distillation_loss as a module; see it for the arguments."""

    temperature: float
    alpha: float
    reduction: str
    dim: int
    backend: str
    precision: object

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        alpha: float = 1.0,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__()
        _bind_distillation(temperature, alpha, reduction, dim, backend, precision)
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        self.dim = dim
        self.backend = backend
        self.precision = precision

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """proximal_distillation_loss(student, teacher) with the stored options."""
        return proximal_distillation_loss(student, teacher, **self._options())

    def _options(self) -> dict[str, Any]:
        return dict(
            temperature=self.temperature,
            alpha=self.alpha,
            reduction=self.reduction,
            dim=self.dim,
            backend=self.backend,
            precision=self.precision,
        )

    def _explain(self, student: torch.Tensor, teacher: torch.Tensor) -> tuple[str, str]:
        return explain_backend(proximal_distillation_loss, student, teacher, **self._options())

    def extra_repr(self) -> str:
        return _repr(self._options())


def _repr(options: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in options.items())


def _check_module_weight(weight: object) -> None:
    """A module's class weights: None or a 1-D floating-point tensor (its length and values
    are checked with the logits)."""
    if weight is None:
        return
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be None or a tensor, not {type(weight).__name__}")
    if not weight.is_floating_point():
        raise TypeError(f"weight must be floating-point, not {weight.dtype}")
    if weight.ndim != 1:
        raise ValueError(f"weight must be 1-D (classes,), not {tuple(weight.shape)}")


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
        dim: Class dimension of logits.
        backend: "auto" (on the CPU the CPU extension, else the native library; fused
            kernels on CUDA; else tensor operations), "kernel" (as "auto", raising
            NotImplementedError where no fused kernel runs), "native" (the native library,
            CPU tensors) or "torch" (tensor operations, CPU or CUDA).
        precision: None (float32 work for 16-bit logits, else their dtype's), "float32" or
            "float64" (or torch.float32, torch.float64); see TORCH.md.

    Returns:
        Per-example losses (logits.shape without dim) or their reduction, in
        promote_types(logits.dtype, float32).

    Raises:
        TypeError: For arguments of the wrong type (a boolean or floating-point target).
        ValueError: For invalid shapes or options; on the CPU also for nonfinite logits,
            labels out of range or invalid class weights (CUDA and torch.func transforms
            give NaN instead).
"""

_MULTILABEL_DOC = """
    Args:
        logits: Floating-point scores with the class dimension dim (at least two classes).
        target: Boolean mask of logits.shape: the positive labels of each example (at
            least one, and for multilabel_hinge_loss at least one negative).
        reduction, dim, backend, precision: As in topk_cross_entropy.

    Returns:
        Per-example losses (logits.shape without dim) or their reduction.

    Raises:
        TypeError: For arguments of the wrong type (a target that is not boolean).
        ValueError: For invalid shapes or options; on the CPU also for nonfinite logits or
            rows without the labels they need (CUDA and transforms give NaN instead).
"""


def topk_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    k: int,
    weight: torch.Tensor | None = None,
    ignore_index: int | None = None,
    reduction: str = "mean",
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """Top-k cross-entropy (the top-k entropy loss of Lapin et al., 2016); cross-entropy at
    k = 1. k is an integer in [1, classes - 1].
    """
    call = _bind(
        "topk_cross_entropy",
        k=k,
        ignore_index=ignore_index,
        reduction=reduction,
        dim=dim,
        backend=backend,
        precision=precision,
    )
    return _loss(call, logits, target, weight)


def multiclass_hinge_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 0.0,
    weight: torch.Tensor | None = None,
    ignore_index: int | None = None,
    reduction: str = "mean",
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """Multiclass hinge loss of Crammer and Singer (2001), smoothed by gamma > 0 (finite);
    hinge values at kinks take deterministic subgradients.
    """
    call = _bind(
        "multiclass_hinge",
        gamma=gamma,
        ignore_index=ignore_index,
        reduction=reduction,
        dim=dim,
        backend=backend,
        precision=precision,
    )
    return _loss(call, logits, target, weight)


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
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """Top-k hinge loss (Lapin et al., 2015), smoothed by gamma > 0 (finite). Variant "beta"
    averages the k largest hinges; "alpha" is the hinge of the average of the k largest
    margins. k is an integer in [1, classes - 1].
    """
    call = _bind(
        _topk_hinge(variant),
        k=k,
        gamma=gamma,
        ignore_index=ignore_index,
        reduction=reduction,
        dim=dim,
        backend=backend,
        precision=precision,
    )
    return _loss(call, logits, target, weight)


def proximal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 1.0,
    weight: torch.Tensor | None = None,
    ignore_index: int | None = None,
    reduction: str = "mean",
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """Moreau envelope of the cross-entropy with step alpha (finite and positive).

    Raises RuntimeError if a CPU normalization does not converge.
    """
    call = _bind(
        "proximal_cross_entropy",
        alpha=alpha,
        ignore_index=ignore_index,
        reduction=reduction,
        dim=dim,
        backend=backend,
        precision=precision,
    )
    return _loss(call, logits, target, weight)


def multilabel_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """Multilabel cross-entropy: uniform target probabilities on the positive labels."""
    call = _bind(
        "multilabel_cross_entropy",
        reduction=reduction,
        dim=dim,
        backend=backend,
        precision=precision,
    )
    return _loss(call, logits, target)


def multilabel_hinge_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 0.0,
    reduction: str = "mean",
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """Multilabel hinge loss, smoothed by gamma > 0 (finite)."""
    call = _bind(
        "multilabel_hinge",
        gamma=gamma,
        reduction=reduction,
        dim=dim,
        backend=backend,
        precision=precision,
    )
    return _loss(call, logits, target)


for _function in (
    topk_cross_entropy,
    multiclass_hinge_loss,
    topk_hinge_loss,
    proximal_cross_entropy,
):
    _function.__doc__ = (_function.__doc__ or "") + _SINGLE_LABEL_DOC
for _function in (multilabel_cross_entropy, multilabel_hinge_loss):
    _function.__doc__ = (_function.__doc__ or "") + _MULTILABEL_DOC


class _SingleLabel(nn.Module):
    """The common parts of the single-label loss modules: class weights (a buffer moved by
    .to() and saved in state_dict()), ignore_index and the call options. Subclasses define
    _function and _options; forward checks the current attributes (a cached check)."""

    weight: torch.Tensor | None
    _function: Callable[..., torch.Tensor]
    ignore_index: int | None
    reduction: str
    dim: int
    backend: str
    precision: object

    def _setup(self, weight: torch.Tensor | None, ignore_index: int | None, **options: Any) -> None:
        _check_module_weight(weight)
        for key, value in options.items():
            setattr(self, key, value)
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight)
        # The options, checked now as each call checks them.
        _bind(self._name(), **self._call_options())

    def _name(self) -> str:
        return self._function.__name__

    def _call_options(self) -> dict[str, Any]:
        return {key: value for key, value in self._options().items() if key != "variant"}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """The loss of logits and integer class labels with the stored options."""
        return type(self)._function(logits, target, weight=self.weight, **self._options())

    def _options(self) -> dict[str, Any]:
        return {**self._own(), "ignore_index": self.ignore_index, **self._common()}

    def _own(self) -> dict[str, Any]:
        return {}

    def _common(self) -> dict[str, Any]:
        return dict(
            reduction=self.reduction, dim=self.dim, backend=self.backend, precision=self.precision
        )

    def _explain(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[str, str]:
        function = type(self)._function
        return explain_backend(function, logits, target, weight=self.weight, **self._options())

    def extra_repr(self) -> str:
        return _repr(self._options())


class TopKCrossEntropyLoss(_SingleLabel):
    """topk_cross_entropy as a module; see it for the arguments."""

    _function = staticmethod(topk_cross_entropy)
    k: int

    def __init__(
        self,
        *,
        k: int,
        weight: torch.Tensor | None = None,
        ignore_index: int | None = None,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__()
        self._setup(
            weight,
            ignore_index,
            k=k,
            reduction=reduction,
            dim=dim,
            backend=backend,
            precision=precision,
        )

    def _own(self) -> dict[str, Any]:
        return {"k": self.k}


class MulticlassHingeLoss(_SingleLabel):
    """multiclass_hinge_loss as a module; see it for the arguments."""

    _function = staticmethod(multiclass_hinge_loss)
    gamma: float

    def __init__(
        self,
        *,
        gamma: float = 0.0,
        weight: torch.Tensor | None = None,
        ignore_index: int | None = None,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__()
        self._setup(
            weight,
            ignore_index,
            gamma=gamma,
            reduction=reduction,
            dim=dim,
            backend=backend,
            precision=precision,
        )

    def _name(self) -> str:
        return "multiclass_hinge"

    def _own(self) -> dict[str, Any]:
        return {"gamma": self.gamma}


class TopKHingeLoss(_SingleLabel):
    """topk_hinge_loss as a module; see it for the arguments."""

    _function = staticmethod(topk_hinge_loss)
    k: int
    variant: Literal["alpha", "beta"]
    gamma: float

    def __init__(
        self,
        *,
        k: int,
        variant: Literal["alpha", "beta"] = "beta",
        gamma: float = 0.0,
        weight: torch.Tensor | None = None,
        ignore_index: int | None = None,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__()
        _topk_hinge(variant)
        self._setup(
            weight,
            ignore_index,
            k=k,
            variant=variant,
            gamma=gamma,
            reduction=reduction,
            dim=dim,
            backend=backend,
            precision=precision,
        )

    def _name(self) -> str:
        return _topk_hinge(self.variant)

    def _own(self) -> dict[str, Any]:
        return {"k": self.k, "variant": self.variant, "gamma": self.gamma}


class ProximalCrossEntropyLoss(_SingleLabel):
    """proximal_cross_entropy as a module; see it for the arguments."""

    _function = staticmethod(proximal_cross_entropy)
    alpha: float

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        weight: torch.Tensor | None = None,
        ignore_index: int | None = None,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__()
        self._setup(
            weight,
            ignore_index,
            alpha=alpha,
            reduction=reduction,
            dim=dim,
            backend=backend,
            precision=precision,
        )

    def _own(self) -> dict[str, Any]:
        return {"alpha": self.alpha}


class _Multilabel(nn.Module):
    """The common parts of the multilabel loss modules."""

    reduction: str
    dim: int
    backend: str
    precision: object
    _function: Callable[..., torch.Tensor]

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """The loss of logits and a boolean label mask with the stored options."""
        return type(self)._function(logits, target, **self._options())

    def _options(self) -> dict[str, Any]:
        raise NotImplementedError

    def _explain(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[str, str]:
        return explain_backend(type(self)._function, logits, target, **self._options())

    def extra_repr(self) -> str:
        return _repr(self._options())


class MultilabelCrossEntropyLoss(_Multilabel):
    """multilabel_cross_entropy as a module; see it for the arguments."""

    _function = staticmethod(multilabel_cross_entropy)

    def __init__(
        self,
        *,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__()
        _bind(
            "multilabel_cross_entropy",
            reduction=reduction,
            dim=dim,
            backend=backend,
            precision=precision,
        )
        self.reduction = reduction
        self.dim = dim
        self.backend = backend
        self.precision = precision

    def _options(self) -> dict[str, Any]:
        return dict(
            reduction=self.reduction, dim=self.dim, backend=self.backend, precision=self.precision
        )


class MultilabelHingeLoss(_Multilabel):
    """multilabel_hinge_loss as a module; see it for the arguments."""

    _function = staticmethod(multilabel_hinge_loss)
    gamma: float

    def __init__(
        self,
        *,
        gamma: float = 0.0,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__()
        _bind(
            "multilabel_hinge",
            gamma=gamma,
            reduction=reduction,
            dim=dim,
            backend=backend,
            precision=precision,
        )
        self.gamma = gamma
        self.reduction = reduction
        self.dim = dim
        self.backend = backend
        self.precision = precision

    def _options(self) -> dict[str, Any]:
        return dict(
            gamma=self.gamma,
            reduction=self.reduction,
            dim=self.dim,
            backend=self.backend,
            precision=self.precision,
        )


def _explain_single_label(name: str) -> Callable[..., tuple[str, str]]:
    def explain(
        logits: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        variant: object = None,
        **options: Any,
    ) -> tuple[str, str]:
        loss = _topk_hinge(variant) if name == "topk_hinge" else name
        return _explain_call(_bind(loss, **options), logits, target, weight)

    return explain


def _explain_distillation_call(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: object = 1.0,
    alpha: object = None,
    reduction: object = "mean",
    dim: object = -1,
    backend: object = "auto",
    precision: object = None,
) -> tuple[str, str]:
    options = _bind_distillation(temperature, alpha, reduction, dim, backend, precision)
    return _explain_distillation(options, student, teacher)


# explain_backend of each loss function, called with its bound arguments.
_EXPLAINERS: dict[Callable[..., Any], Callable[..., tuple[str, str]]] = {
    topk_cross_entropy: _explain_single_label("topk_cross_entropy"),
    multiclass_hinge_loss: _explain_single_label("multiclass_hinge"),
    topk_hinge_loss: _explain_single_label("topk_hinge"),
    proximal_cross_entropy: _explain_single_label("proximal_cross_entropy"),
    multilabel_cross_entropy: _explain_single_label("multilabel_cross_entropy"),
    multilabel_hinge_loss: _explain_single_label("multilabel_hinge"),
    distillation_loss: _explain_distillation_call,
    proximal_distillation_loss: _explain_distillation_call,
    cross_entropy: _ce.explain,
    linear_cross_entropy: _lce.explain,
}
