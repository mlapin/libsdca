"""PyTorch loss layers: SDCALoss and sdca_loss (the LOSSES at the input
scores) and DistillationLoss and distillation_loss (cross-entropy to a
teacher's softmax). CPU losses run in the CPU extension (built on first use,
_cpu_entropy) or the native library (_cpu), CUDA losses in fused Triton kernels;
the tensor backend runs every loss on any device, and its entropy has higher
derivatives.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.autograd.function import FunctionCtx

from . import _cpu, _cpu_entropy, _cuda_launch, _entropy, _hinge, _multilabel, _ops, _topk
from ._core import (
    _KINDS,
    _canonical_precision,
    _check_options,
    _Entry,
    _entry,
    _kind_of,
    _tensor_kind,
    _topk_hinge_name,
)
from ._cross_entropy import CrossEntropyLoss as CrossEntropyLoss
from ._cross_entropy import cross_entropy as cross_entropy
from ._entropy import entropy as _indexed_entropy
from ._entropy import fused as _fused_entropy
from ._first_order import first_order as _first_order
from ._linear_cross_entropy import LinearCrossEntropyLoss as LinearCrossEntropyLoss
from ._linear_cross_entropy import linear_cross_entropy as linear_cross_entropy
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
    "SDCALoss",
    "TopKCrossEntropyLoss",
    "TopKHingeAlphaLoss",
    "TopKHingeBetaLoss",
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
    "sdca_loss",
    "topk_cross_entropy",
    "topk_hinge_alpha_loss",
    "topk_hinge_beta_loss",
    "topk_hinge_loss",
]

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


def _working(logits: torch.Tensor, precision: str | None) -> str | None:
    """The precision a call computes in: None is float32 for float16/bfloat16 logits."""
    return "float32" if precision is None and logits.dtype in _HALF else precision


def _check_precision(logits: torch.Tensor, precision: object) -> str | None:
    """float64 logits take float64 (or None); other dtypes take float32 or float64 (or None)."""
    canonical = _canonical_precision(precision)
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


def _check_tensors(logits: torch.Tensor, target: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor) or logits.ndim < 1:
        raise ValueError("logits must be a tensor with a class dimension")
    if not logits.is_floating_point():
        raise ValueError("logits must be finite floating-point values")
    if not isinstance(target, torch.Tensor):
        raise ValueError("target must be a tensor")
    if target.device != logits.device:
        raise ValueError("target and logits must be on the same device")


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


def sdca_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss: str = "cross_entropy",
    *,
    backend: str = "auto",
    k: int = 1,
    gamma: float = 0.0,
    alpha: float = 1.0,
    reduction: str = "mean",
    dim: int = -1,
    precision: object = None,
) -> torch.Tensor:
    """A loss of the scores (not a differentiable call to libsdca.fit).

    Gradients are first-order only (differentiating them raises), except
    that entropy supports double backward (integer entropy on every backend,
    boolean-mask entropy on the tensor backend and the CPU extension). Hinge
    values at kinks use deterministic subgradients.

    Args:
        logits: Finite floating-point scores with the class dimension dim
            (at least two classes; without examples the result is that of
            torch.nn.functional: empty, a zero sum or a NaN mean).
        target: int64 class indices of logits.shape without dim, or a boolean
            mask of logits.shape: uniformly weighted positive labels for
            the cross-entropies, positive sets for multilabel hinge. Other
            losses require one target. Must be on the device of logits.
        loss: One of libsdca.LOSSES.
        backend: "auto" (on the CPU the CPU extension, else the native
            library; fused kernels on CUDA; else torch), "native" (the native
            library, CPU tensors) or "torch" (tensor operations on any device).
        k: Top-k parameter in [1, classes-1]; only the topk losses use it.
        gamma: Hinge smoothing, finite and >= 0 (0: nonsmooth).
        alpha: proximal_cross_entropy step, finite and positive.
        reduction: "none", "sum" or "mean".
        dim: Class dimension of logits.
        precision: None (the default of the dtype), "float32" or "float64"
            (or a matching dtype object); see TORCH.md for what each guarantees.

    Returns:
        Per-example losses (logits.shape without dim) or their reduction.

    Raises:
        ValueError: For invalid inputs or options.
        RuntimeError: If a proximal_cross_entropy normalization does not converge on CPU.
    """
    return _bound_loss(
        _entry(loss), loss, logits, target, backend, k, gamma, alpha, reduction, dim, precision
    )


def _bound_loss(
    entry: _Entry,
    loss: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    backend: str,
    k: int,
    gamma: float,
    alpha: float,
    reduction: str,
    dim: int,
    precision: object,
) -> torch.Tensor:
    """sdca_loss of the loss bound in entry (named loss)."""
    if _has_dtensor(logits, target):
        return _call_with_dtensors(
            sdca_loss,
            logits,
            target,
            loss,
            backend=backend,
            k=k,
            gamma=gamma,
            alpha=alpha,
            reduction=reduction,
            dim=dim,
            precision=precision,
        )
    # The private kind of the tensor backends; gamma is 1 for a nonsmooth hinge.
    objective, kind_k, gamma = _kind_of(entry, k, gamma)
    if backend not in ("auto", "native", "torch"):
        raise ValueError("Unknown backend")
    _check_tensors(logits, target)
    canonical = _canonical_precision(precision)
    return _resolved_loss(
        logits, target, objective, backend, kind_k, gamma, alpha, reduction, dim, canonical, k
    )


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
    if classes < 2:
        raise ValueError("At least two classes are required")
    _check_options(given_k, gamma, alpha, classes)
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
    on any device, of class-last scores z."""
    shape, classes = z.shape[:-1], z.shape[-1]
    fused = backend != "torch"
    # Fused CUDA entropy (integer targets or boolean masks) uses float32 exponentials + float64
    # row statistics by default for non-float64 logits, and full float64 when
    # precision="float64" or logits are float64.
    # The fused hinges work in float64 unless an explicit precision="float32" selects tensors.
    explicit = canonical == "float32"
    work = _working(logits, canonical)
    indexed = fused and target.dtype == torch.int64 and not explicit
    labels_fit = target.shape == shape and reduction in ("none", "sum", "mean")
    if _indexed_entropy_path(z, target, objective, k, explicit, fused):
        return _device_entropy(logits, z, target, reduction, canonical, fused)
    if indexed and k == 1 and objective in _hinge.SMOOTH and labels_fit:
        flat = _flat(z)
        if _hinge.covers(flat):
            value = _hinge.hinge(flat, _labels(target), objective, reduction, gamma)
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
    explicit = canonical == "float32"
    fp32 = canonical != "float64" and logits.dtype != torch.float64 if fused else explicit
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
    work = _working(logits, canonical)
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


# The named wrappers' losses, bound once at import.
_TOPK_CROSS_ENTROPY = _KINDS["topk_cross_entropy"]
_MULTICLASS_HINGE = _KINDS["multiclass_hinge"]
_TOPK_HINGE_ALPHA = _KINDS["topk_hinge_alpha"]
_TOPK_HINGE_BETA = _KINDS["topk_hinge_beta"]
_MULTILABEL_CROSS_ENTROPY = _KINDS["multilabel_cross_entropy"]
_MULTILABEL_HINGE = _KINDS["multilabel_hinge"]
_PROXIMAL_CROSS_ENTROPY = _KINDS["proximal_cross_entropy"]


def _indexed_entropy_path(
    z: torch.Tensor,
    target: torch.Tensor,
    objective: str,
    k: int,
    explicit: bool,
    fused: bool = True,
) -> bool:
    """Integer-target entropy (or top-1 entropy) of class-last z in float64 or the fused float32
    kernels (where fused). Default float16/bfloat16 work without those kernels keeps the float64
    path, whose storage-limit guarantees the float32 tensor entropy lacks."""
    if target.dtype != torch.int64 or not (
        objective == "cross_entropy" or (objective == "topk_cross_entropy" and k == 1)
    ):
        return False
    if not explicit:
        return True
    return (
        fused
        and z.is_cuda
        and z.shape[-1] <= _entropy.CLASSES
        and z.numel() <= z.shape[-1] * _entropy._cuda_launch.ROWS
        and _entropy._kernel() is not None
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


def _unfused(objective: str, hinge: bool, fp32: bool, target: torch.Tensor) -> str:
    """Why a call takes the tensor operations; hinge: a k = 1 hinge the CUDA kernels skip."""
    if hinge:
        return "the fused k = 1 hinges need Triton 3.3 and at most 2**28 rows"
    if objective in _hinge.SMOOTH or objective == "topk_cross_entropy":
        return (
            "fused paths need int64 targets, no precision='float32' and k = 1 (on CUDA also k > 1)"
        )
    if objective == "cross_entropy" and fp32 and target.dtype == torch.int64:
        return "float32 work fuses integer entropy only on CUDA with Triton"
    return f"{objective} has no fused integer-target path"


def explain_backend(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss: str = "cross_entropy",
    *,
    backend: str = "auto",
    k: int = 1,
    gamma: float = 0.0,
    reduction: str = "mean",
    dim: int = -1,
    precision: object = None,
) -> tuple[str, str]:
    """(backend, reason) that sdca_loss takes for these arguments, without computing the loss.

    The backend is "cuda-launcher" (fused Triton kernels launched from C++), "triton"
    (Triton kernels launched by Triton), "cpu-kernel" (the CPU extension), "native" or
    "torch", or "dtensor" (a full-input distributed fallback). Like the first such
    call, it may build the CPU extension or the launcher and compile kernels. Arguments
    as in sdca_loss.
    """
    objective, kind_k, gamma = _tensor_kind(loss, k, gamma)
    if backend not in ("auto", "native", "torch"):
        raise ValueError("Unknown backend")
    canonical = _check_precision(logits, precision)
    if _has_dtensor(logits, target):
        _dtensor_mesh(logits, target)
        if backend == "native" and logits.device.type != "cpu":
            raise ValueError(
                "Native losses require CPU tensors; use backend='torch' on other devices"
            )
        return "dtensor", "gathers full inputs on every rank and returns a replicated global loss"
    z = logits.movedim(dim, -1)
    shape, classes = z.shape[:-1], z.shape[-1]
    if classes < 2:
        raise ValueError("At least two classes are required")
    _check_options(k, gamma, 1.0, classes)
    k = kind_k
    flat = z.reshape(-1, classes)
    work = _working(logits, canonical)
    if flat.device.type == "cpu":
        plan = _cpu_plan(logits, shape, target, objective, backend, k, reduction, canonical)
        if plan is not None:
            return plan.backend, plan.reason
    if backend == "native":
        raise ValueError("Native losses require CPU tensors; use backend='torch' on other devices")
    explicit = canonical == "float32"
    if backend == "torch":
        return _explain_torch(z, flat, target, objective, k, explicit, work)
    indexed = target.dtype == torch.int64 and not explicit and target.shape == shape
    if target.shape == shape and _indexed_entropy_path(z, target, objective, k, explicit):
        if _fused_entropy(flat):
            kernel = _entropy._kernel()
            assert kernel is not None
            fp32_entropy = canonical != "float64" and logits.dtype != torch.float64
            launch = kernel.launcher(flat, kernel._REDUCTIONS[reduction], fp32_entropy)
            return _fused_cuda(launch, kernel.describe(classes, fp32_entropy))
        why = "needs Triton 3.3, at most 2**28 rows and 2**25 classes" if flat.is_cuda else "CPU"
        return "torch", f"integer entropy in float64 tensor operations ({why})"
    fast = indexed and k == 1 and objective in _hinge.SMOOTH
    if fast and reduction in ("none", "sum", "mean") and _hinge.covers(flat):
        kernel = _hinge._kernel()
        assert kernel is not None
        code = kernel._REDUCTIONS[reduction]
        return _fused_cuda(kernel.launcher(flat, code, _hinge.SMOOTH[objective]), "float64")
    reason = _multilabel.explain(objective, logits, target, classes, backend, canonical, reduction)
    if reason is not None:
        return reason
    fp32_topk = _topk_fp32(logits, canonical)
    explained = (
        _explain_topk(flat, objective, k, reduction, indexed, fp32_topk) if flat.is_cuda else None
    )
    if explained is not None:
        return explained
    arithmetic = _tensor_arithmetic(objective, flat, work)
    if objective == "proximal_cross_entropy" and _prox_kernel(flat) is not None:
        return (
            "triton",
            f"proximal_cross_entropy runs one Triton kernel (Python launch, {arithmetic})",
        )
    why = _unfused(objective, fast and flat.is_cuda, work == "float32", target)
    return "torch", f"tensor operations in {arithmetic}: {why}"


def _explain_torch(
    z: torch.Tensor,
    flat: torch.Tensor,
    target: torch.Tensor,
    objective: str,
    k: int,
    explicit: bool,
    work: str | None,
) -> tuple[str, str]:
    """explain_backend of backend="torch": the tensor operations on every device."""
    entropy = _indexed_entropy_path(z, target, objective, k, explicit, fused=False)
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


# SDCALoss options resolved by _bind when set.
_STATIC = frozenset(("loss", "backend", "k", "gamma", "precision"))


class SDCALoss(nn.Module):
    """A libsdca loss as a module; see sdca_loss for the arguments.

    Example:
        >>> criterion = SDCALoss("topk_hinge_beta", k=2, gamma=0.5)
        >>> criterion(torch.randn(8, 5), torch.randint(0, 5, (8,))).shape
        torch.Size([])
    """

    # Resolved by _resolve from the options; forward only reads them.
    _objective: str
    _k: int
    _gamma: float
    _canonical: str | None

    def __init__(
        self,
        loss: str = "cross_entropy",
        *,
        backend: str = "auto",
        k: int = 1,
        gamma: float = 0.0,
        alpha: float = 1.0,
        reduction: str = "mean",
        dim: int = -1,
        precision: object = None,
    ) -> None:
        super().__init__()
        self.loss = loss
        self.backend = backend
        self.k = k
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.dim = dim
        self.precision = precision
        self._bind()

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _STATIC and "_objective" in self.__dict__:
            # Validate before assigning, so a rejected value leaves the module unchanged.
            options = {key: getattr(self, key) for key in _STATIC}
            options[name] = value
            bound = self._resolve(**options)
            super().__setattr__(name, value)
            self.__dict__.update(bound)
        else:
            super().__setattr__(name, value)

    @staticmethod
    def _resolve(
        loss: str, backend: str, k: int, gamma: float, precision: object
    ) -> dict[str, Any]:
        """The loss, k, gamma, backend and precision resolved once; forward only reads them."""
        objective, kind_k, kind_gamma = _tensor_kind(loss, k, gamma)
        if backend not in ("auto", "native", "torch"):
            raise ValueError("Unknown backend")
        return {
            "_objective": objective,
            "_k": kind_k,
            "_gamma": kind_gamma,
            "_canonical": _canonical_precision(precision),
        }

    def _bind(self) -> None:
        self.__dict__.update(self._resolve(**{key: getattr(self, key) for key in _STATIC}))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """sdca_loss(logits, target) with the stored loss and options."""
        if _has_dtensor(logits, target):
            return sdca_loss(
                logits,
                target,
                self.loss,
                backend=self.backend,
                k=self.k,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=self.reduction,
                dim=self.dim,
                precision=self.precision,
            )
        _check_tensors(logits, target)
        return _resolved_loss(
            logits,
            target,
            self._objective,
            self.backend,
            self._k,
            self._gamma,
            self.alpha,
            self.reduction,
            self.dim,
            self._canonical,
            self.k,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.loss!r}, backend={self.backend!r}, k={self.k}, gamma={self.gamma}, "
            f"alpha={self.alpha}, reduction={self.reduction!r}, dim={self.dim}, "
            f"precision={self.precision!r}"
        )


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
    ) -> torch.Tensor:
        kernel = _kernel_distill(True) if student.is_cuda else None
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


def distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float = 1.0,
    *,
    reduction: str = "mean",
    dim: int = -1,
    precision: object = None,
) -> torch.Tensor:
    """Cross-entropy with a fixed teacher distribution softmax(teacher/T).

    Both arguments are LOGITS. The teacher is detached by definition. No T²
    multiplier is applied. This first-order gradient is exactly zero at equal
    represented logits; near equality it avoids probability subtraction.

    Args:
        student: Finite floating-point logits with the class dimension dim
            (at least two classes; no examples give the torch.nn.functional
            results).
        teacher: Finite logits of the same shape, device and dtype.
        temperature: T, finite and positive.
        reduction: "none", "sum" or "mean".
        dim: Class dimension of the logits.
        precision: None or "float32" (float32 arithmetic for non-float64
            logits) or "float64" (float64 arithmetic, or matching dtype
            objects). Temperatures float32 cannot hold use float64.

    Returns:
        Per-example losses (student.shape without dim) or their reduction.

    Raises:
        ValueError: For invalid inputs or options.
    """
    fast = _fast_distillation(student, teacher, temperature, reduction, dim, precision)
    if fast is not None:
        return fast
    if _has_dtensor(student, teacher):
        return _call_with_dtensors(
            distillation_loss,
            student,
            teacher,
            temperature,
            reduction=reduction,
            dim=dim,
            precision=precision,
        )
    return _distillation(student, teacher, temperature, None, reduction, dim, precision)


_FAST_DTYPES = (torch.float32, torch.bfloat16, torch.float16, torch.float64)


def _fast_distillation(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
    reduction: str,
    dim: int,
    precision: object,
) -> torch.Tensor | None:
    """_cpu_distillation of ordinary (N, C) CPU tensors, or None: where it applies every check
    of distillation_loss passes but finiteness, which the kernels check."""
    if (
        type(student) is torch.Tensor
        and type(teacher) is torch.Tensor
        and type(temperature) in (float, int)
        and 0 < temperature < math.inf
        and reduction in ("none", "mean", "sum")
        and student.is_cpu
        and student.ndim == 2
        and dim in (-1, 1)
        and student.dtype in _FAST_DTYPES
        and (precision is None or (precision == "float32" and student.dtype != torch.float64))
        and teacher.dtype == student.dtype
        and teacher.is_cpu
        and teacher.shape == student.shape
        and student.shape[0] > 0
        and student.shape[1] >= 2
        and not _ops.is_compiling()
    ):
        return _cpu_distillation(student, teacher.detach(), temperature, None, False, reduction)
    return None


def _check_distillation(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
    alpha: float | None,
    reduction: str,
    precision: object,
) -> bool:
    """Check the arguments of a distillation loss; whether it computes in float64."""
    if (
        student.shape != teacher.shape
        or student.device != teacher.device
        or student.dtype != teacher.dtype
        or student.ndim < 1
    ):
        raise ValueError("Student and teacher must have matching shape, device, and dtype")
    if not student.is_floating_point():
        raise ValueError("Logits must be finite floating-point tensors")
    double = _check_precision(student, precision) == "float64"
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if alpha is not None and (not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("alpha must be finite and positive")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("reduction must be none, sum, or mean")
    return double


def _cpu_distillation(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
    alpha: float | None,
    double: bool,
    reduction: str,
) -> torch.Tensor | None:
    """The distillation loss of (N, C) CPU logits, reduced, by the extension or the native
    library (None: the tensor operations, which tests select)."""
    # Both check finiteness in their own passes.
    implementation = _cpu.EXTENSION if _cpu_entropy.enabled() else _cpu.NATIVE
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


def _distillation(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
    alpha: float | None,
    reduction: str,
    dim: int,
    precision: object,
) -> torch.Tensor:
    """distillation_loss, or proximal_distillation_loss with step alpha."""
    double = _check_distillation(student, teacher, temperature, alpha, reduction, precision)
    s, t = student, teacher.detach()
    if dim not in (-1, s.ndim - 1):
        s, t = s.movedim(dim, -1), t.movedim(dim, -1)
    if s.shape[-1] < 2:
        raise ValueError("At least two classes are required")
    shape, dtype = s.shape[:-1], _value_dtype(student)
    if s.numel() == 0:
        return _empty_loss(s, reduction, dtype)
    s2, t2 = (s, t) if s.ndim == 2 else (s.reshape(-1, s.shape[-1]), t.reshape(-1, t.shape[-1]))
    if student.device.type == "cpu":
        loss = _cpu_distillation(s2, t2, temperature, alpha, double, reduction)
        if loss is not None:
            return loss.reshape(shape) if reduction == "none" else loss
    if alpha is None:
        fused = _fused_distillation(s2, t2, temperature, double, reduction)
        if fused is not None:
            return fused.reshape(shape) if reduction == "none" else fused
        values = _Distillation.apply(s2, t2, temperature, double)
    else:
        values = _ProximalDistillation.apply(s2, t2, temperature, alpha, double)
    return _reduce(values.reshape(shape), reduction)


class DistillationLoss(nn.Module):
    """distillation_loss as a module; see it for the arguments."""

    def __init__(
        self,
        temperature: float = 1.0,
        *,
        reduction: str = "mean",
        dim: int = -1,
        precision: object = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.dim = dim
        self.precision = precision

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """distillation_loss(student, teacher) with the stored options."""
        return distillation_loss(
            student,
            teacher,
            temperature=self.temperature,
            reduction=self.reduction,
            dim=self.dim,
            precision=self.precision,
        )

    def extra_repr(self) -> str:
        return (
            f"temperature={self.temperature}, reduction={self.reduction!r}, dim={self.dim}, "
            f"precision={self.precision!r}"
        )


def proximal_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float = 1.0,
    alpha: float = 1.0,
    *,
    reduction: str = "mean",
    dim: int = -1,
    precision: object = None,
) -> torch.Tensor:
    """Moreau envelope of distillation cross-entropy with step alpha and temperature T.

    Computes min_u { H(softmax(teacher/T), softmax(u)) + |u - student/T|^2 / (2 alpha) }.
    As alpha -> 0+, this converges to distillation_loss(student, teacher, temperature).
    The first-order gradient is identically zero at equal represented student and teacher logits.
    alpha is finite and positive; other arguments, the result and errors as in distillation_loss.
    """
    if _has_dtensor(student, teacher):
        return _call_with_dtensors(
            proximal_distillation_loss,
            student,
            teacher,
            temperature,
            alpha,
            reduction=reduction,
            dim=dim,
            precision=precision,
        )
    return _distillation(student, teacher, temperature, alpha, reduction, dim, precision)


class ProximalDistillationLoss(nn.Module):
    """proximal_distillation_loss as a module; see it for the arguments."""

    def __init__(
        self,
        temperature: float = 1.0,
        alpha: float = 1.0,
        *,
        reduction: str = "mean",
        dim: int = -1,
        precision: object = None,
    ) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        self.dim = dim
        self.precision = precision

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """proximal_distillation_loss(student, teacher) with the stored options."""
        return proximal_distillation_loss(
            student,
            teacher,
            temperature=self.temperature,
            alpha=self.alpha,
            reduction=self.reduction,
            dim=self.dim,
            precision=self.precision,
        )

    def extra_repr(self) -> str:
        return (
            f"temperature={self.temperature}, alpha={self.alpha}, "
            f"reduction={self.reduction!r}, dim={self.dim}, precision={self.precision!r}"
        )


def topk_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    k: int = 1,
    *,
    reduction: str = "mean",
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """Top-k cross-entropy (the top-k entropy loss); cross_entropy at k = 1.

    Other arguments, the result and errors as in sdca_loss.
    """
    return _bound_loss(
        _TOPK_CROSS_ENTROPY,
        "topk_cross_entropy",
        logits,
        target,
        backend,
        k,
        0.0,
        1.0,
        reduction,
        dim,
        precision,
    )


class TopKCrossEntropyLoss(SDCALoss):
    """topk_cross_entropy as a module; one target per example."""

    def __init__(
        self,
        k: int = 1,
        *,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__(
            "topk_cross_entropy",
            k=k,
            reduction=reduction,
            dim=dim,
            backend=backend,
            precision=precision,
        )


def multiclass_hinge_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 0.0,
    *,
    reduction: str = "mean",
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """Multiclass hinge loss of Crammer and Singer, smoothed by gamma > 0.

    Other arguments, the result and errors as in sdca_loss.
    """
    return _bound_loss(
        _MULTICLASS_HINGE,
        "multiclass_hinge",
        logits,
        target,
        backend,
        1,
        gamma,
        1.0,
        reduction,
        dim,
        precision,
    )


class MulticlassHingeLoss(SDCALoss):
    """multiclass_hinge_loss as a module."""

    def __init__(
        self,
        gamma: float = 0.0,
        *,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__(
            "multiclass_hinge",
            gamma=gamma,
            reduction=reduction,
            dim=dim,
            backend=backend,
            precision=precision,
        )


def topk_hinge_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    k: int = 1,
    *,
    variant: str = "beta",
    gamma: float = 0.0,
    reduction: str = "mean",
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """Top-k hinge loss, smoothed by gamma > 0. Variant "beta" averages the k
    largest hinges; "alpha" is the hinge of the average of the k largest margins.

    variant is "alpha" or "beta"; other arguments, the result and errors as in sdca_loss.
    """
    if variant == "beta":
        entry, name = _TOPK_HINGE_BETA, "topk_hinge_beta"
    elif variant == "alpha":
        entry, name = _TOPK_HINGE_ALPHA, "topk_hinge_alpha"
    else:
        raise ValueError("variant must be 'alpha' or 'beta'")
    return _bound_loss(
        entry, name, logits, target, backend, k, gamma, 1.0, reduction, dim, precision
    )


def topk_hinge_alpha_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    k: int = 1,
    *,
    gamma: float = 0.0,
    reduction: str = "mean",
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """topk_hinge_loss with variant="alpha".

    Other arguments, the result and errors as in sdca_loss.
    """
    return _bound_loss(
        _TOPK_HINGE_ALPHA,
        "topk_hinge_alpha",
        logits,
        target,
        backend,
        k,
        gamma,
        1.0,
        reduction,
        dim,
        precision,
    )


def topk_hinge_beta_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    k: int = 1,
    *,
    gamma: float = 0.0,
    reduction: str = "mean",
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """topk_hinge_loss with variant="beta".

    Other arguments, the result and errors as in sdca_loss.
    """
    return _bound_loss(
        _TOPK_HINGE_BETA,
        "topk_hinge_beta",
        logits,
        target,
        backend,
        k,
        gamma,
        1.0,
        reduction,
        dim,
        precision,
    )


class TopKHingeLoss(SDCALoss):
    """topk_hinge_loss as a module (variant "beta" by default)."""

    def __init__(
        self,
        k: int = 1,
        *,
        variant: str = "beta",
        gamma: float = 0.0,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__(
            _topk_hinge_name(variant),
            k=k,
            gamma=gamma,
            reduction=reduction,
            dim=dim,
            backend=backend,
            precision=precision,
        )


class TopKHingeAlphaLoss(TopKHingeLoss):
    """topk_hinge_alpha_loss as a module."""

    def __init__(
        self,
        k: int = 1,
        *,
        gamma: float = 0.0,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__(
            k,
            variant="alpha",
            gamma=gamma,
            reduction=reduction,
            dim=dim,
            backend=backend,
            precision=precision,
        )


class TopKHingeBetaLoss(TopKHingeLoss):
    """topk_hinge_beta_loss as a module."""

    def __init__(
        self,
        k: int = 1,
        *,
        gamma: float = 0.0,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__(
            k,
            variant="beta",
            gamma=gamma,
            reduction=reduction,
            dim=dim,
            backend=backend,
            precision=precision,
        )


def multilabel_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """Multilabel cross-entropy: uniform target probabilities on the positive labels.

    Other arguments, the result and errors as in sdca_loss.
    """
    return _bound_loss(
        _MULTILABEL_CROSS_ENTROPY,
        "multilabel_cross_entropy",
        logits,
        target,
        backend,
        1,
        0.0,
        1.0,
        reduction,
        dim,
        precision,
    )


class MultilabelCrossEntropyLoss(SDCALoss):
    """multilabel_cross_entropy as a module."""

    def __init__(
        self,
        *,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__(
            "multilabel_cross_entropy",
            reduction=reduction,
            dim=dim,
            backend=backend,
            precision=precision,
        )


def multilabel_hinge_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 0.0,
    *,
    reduction: str = "mean",
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """Multilabel hinge loss, smoothed by gamma > 0.

    Other arguments, the result and errors as in sdca_loss.
    """
    return _bound_loss(
        _MULTILABEL_HINGE,
        "multilabel_hinge",
        logits,
        target,
        backend,
        1,
        gamma,
        1.0,
        reduction,
        dim,
        precision,
    )


class MultilabelHingeLoss(SDCALoss):
    """multilabel_hinge_loss as a module."""

    def __init__(
        self,
        gamma: float = 0.0,
        *,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__(
            "multilabel_hinge",
            gamma=gamma,
            reduction=reduction,
            dim=dim,
            backend=backend,
            precision=precision,
        )


def proximal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    *,
    reduction: str = "mean",
    dim: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> torch.Tensor:
    """Moreau envelope of the cross-entropy with step alpha.

    Other arguments, the result and errors as in sdca_loss.
    """
    return _bound_loss(
        _PROXIMAL_CROSS_ENTROPY,
        "proximal_cross_entropy",
        logits,
        target,
        backend,
        1,
        0.0,
        alpha,
        reduction,
        dim,
        precision,
    )


class ProximalCrossEntropyLoss(SDCALoss):
    """proximal_cross_entropy as a module."""

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        reduction: str = "mean",
        dim: int = -1,
        backend: str = "auto",
        precision: object = None,
    ) -> None:
        super().__init__(
            "proximal_cross_entropy",
            alpha=alpha,
            reduction=reduction,
            dim=dim,
            backend=backend,
            precision=precision,
        )
