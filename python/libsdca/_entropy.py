"""Integer-target entropy with log-domain tails and scaled reverse derivatives."""

from __future__ import annotations

import functools
import math
from types import ModuleType
from typing import Protocol

import torch

from . import _cuda_launch, _ops


class _Context(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]
    reduction: str
    fused: bool
    fp32: bool

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


def _power(e: torch.Tensor) -> torch.Tensor:
    """2**e exactly, of integers e in [-1022, 1023]."""
    return ((e.to(torch.int64) + 1023) << 52).view(torch.float64)


def _ldexp(m: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """m 2**e of float64 m in [2**-5, 1) (or 0) and integers e, rounded once."""
    first = e.clamp(-1016, 1023)
    return m * _power(first) * _power((e - first).clamp(-1022, 1023))


def _parts(
    stats: tuple[torch.Tensor, ...], weight: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every row's loss (times weight) as m 2**e with m in [2**-5, 1) or 0: a tail from the
    mantissas of its half exponential, its sum and the weight, none of which leaves double's
    normal range; the log of a tail would round its exponential's argument instead."""
    maximum, label, log_sum, relative = stats
    tail = relative < -36
    large = torch.where(torch.isfinite(relative), relative * 0.5, maximum * 0.5 - label * 0.5)
    half = torch.where(relative > 36, large, relative.clamp(-36, 36).exp().log1p() * 0.5)
    m_half, e_half = torch.frexp(torch.where(tail, ((maximum - label) * 0.5).exp(), 1.0))
    m, e = torch.frexp(torch.where(tail, log_sum.exp(), half))
    m, e = torch.where(tail, m_half * m_half * m, m), torch.where(tail, 2 * e_half + e, e + 1)
    if weight is not None:
        m_weight, e_weight = torch.frexp(weight)
        m, e = m * m_weight, e + e_weight
    return m, e


def _reduce_parts(
    m: torch.Tensor, e: torch.Tensor, reduction: str, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """The sum, or the mean over weight's sum (else the row count), of the rows m 2**e of
    _parts, scaled by the largest exponent and rounded once."""
    top = torch.where(m != 0, e, -(2**20)).amax()
    total = _ldexp(m, e - top).sum()
    if reduction == "mean":
        if weight is None:
            total = total / m.numel()
        else:
            m_weight, e_weight = torch.frexp(weight)
            weight_top = torch.where(m_weight != 0, e_weight, -(2**20)).amax()
            total = total / _ldexp(m_weight, e_weight - weight_top).sum()
            top = top - weight_top
    m_total, e_total = torch.frexp(total)
    return _ldexp(m_total, e_total + top)


def _statistics(
    z: torch.Tensor, target: torch.Tensor, invalid: torch.Tensor | None = None
) -> tuple[torch.Tensor, ...]:
    z = z.double()
    c = z.shape[1]
    safe = target.clamp(0, c - 1)[:, None]
    label = z.gather(1, safe).squeeze(1)
    masked = z.scatter(1, safe, -torch.inf)
    other_max = masked.max(1).values
    log_sum = (masked - other_max[:, None]).exp().sum(1).log()
    relative = (other_max - label) + log_sum
    if invalid is None:
        invalid = (target < 0) | (target >= c) | ~z.isfinite().all(1)
    return (
        other_max.masked_fill(invalid, math.nan),
        label.masked_fill(invalid, math.nan),
        log_sum.masked_fill(invalid, math.nan),
        relative.masked_fill(invalid, math.nan),
    )


def _values(stats: tuple[torch.Tensor, ...], reduction: str) -> torch.Tensor:
    maximum, label, log_sum, relative = stats
    bounded = relative.clamp(-36, 36)
    central = bounded.exp().log1p()
    half_tail = ((maximum - label) * 0.5).exp()
    tail = half_tail * (half_tail * log_sum.exp())
    value = torch.where(relative < -36, tail, torch.where(relative > 36, relative, central))
    if reduction == "none":
        return value
    total = value.sum()
    ordinary = total if reduction == "sum" else total / value.numel()
    # Normal values with a finite total sum directly. Rounded subnormal tails, or values
    # beyond double's range whose mean is representable, take their exact parts.
    normal = (value.min() >= torch.finfo(torch.float64).tiny) & torch.isfinite(total)
    if not value.is_cuda and bool(normal):
        return ordinary
    exact = _reduce_parts(*_parts(stats), reduction)
    return torch.where(value.isnan().any(), math.nan, torch.where(normal, ordinary, exact))


@functools.cache
def _kernel() -> ModuleType | None:
    if torch.version.cuda is None:
        return None
    try:
        import triton

        from . import _triton_entropy
    except ImportError:
        return None
    return _triton_entropy if _cuda_launch.kernels_supported(triton.__version__) else None


# Widest rows of the CUDA kernels (_triton_entropy.CLASSES, without importing Triton).
CLASSES = 2**25


def covers(device: torch.device, rows: int, classes: int) -> bool:
    """Whether the optional CUDA kernels cover logits of this device and shape."""
    return (
        device.type == "cuda"
        and rows <= _cuda_launch.ROWS
        and classes <= CLASSES
        and _kernel() is not None
    )


def fused(logits: torch.Tensor) -> bool:
    """Whether the optional CUDA kernel covers this input shape."""
    return (
        logits.is_cuda
        and logits.shape[0] <= _cuda_launch.ROWS
        and logits.shape[1] <= CLASSES
        and _kernel() is not None
    )


def _gradient(
    logits: torch.Tensor,
    target: torch.Tensor,
    stats: tuple[torch.Tensor, ...],
    upstream: torch.Tensor,
    reduction: str,
    *,
    bounded_scale: bool = False,
) -> torch.Tensor:
    bounded = bounded_scale or (logits.dtype != torch.float64 and upstream.dtype != torch.float64)
    maximum, _, log_sum, relative = stats
    safe = target.clamp(0, logits.shape[1] - 1)[:, None]
    # log of total non-target mass, computed without first forming that mass.
    log_mass = -torch.logaddexp(torch.zeros_like(relative), -relative)
    log_gradient = (logits.double() - maximum[:, None]) + (log_mass - log_sum)[:, None]
    log_gradient = log_gradient.scatter(1, safe, log_mass[:, None])
    upstream = upstream.double().expand(logits.shape[0])
    root = (log_gradient * 0.5).exp()
    divisor = math.sqrt(logits.shape[0]) if reduction == "mean" else 1.0
    if reduction == "mean":
        root = root / divisor
    # Linear in the upstream derivative even at zero, including double
    # backward. The upstream multiplication precedes the final rounding.
    gradient = root * (root * upstream[:, None])
    gradient = gradient.scatter(1, safe, -gradient.gather(1, safe))
    if bounded:
        # Log-domain rounding is far below float32 storage error. Keep the
        # factored roots: later derivatives can rescue initially vanished tails.
        return gradient.to(logits.dtype)
    # Without far rows the selection below returns gradient: skipped where checking synchronizes
    # (CUDA) or is not traced (compilation).
    if not (logits.is_cuda or _ops.is_compiling()) and not bool((relative < -37).any()):
        return gradient.to(logits.dtype)
    # Keep an exactly represented gap separate from log(class count). Taking
    # exp(gap + log(count)) loses relative digits for tiny but normal tails.
    label = stats[1]
    far = relative < -37
    half = torch.where(far[:, None], (logits.double() - label[:, None]) * 0.5, -torch.inf).exp()
    if reduction == "mean":
        half = half / divisor
    far_gradient = half * (half * upstream[:, None])
    half_mass = torch.where(far, (maximum - label) * 0.5, -torch.inf).exp()
    if reduction == "mean":
        half_mass = half_mass / divisor
    target_mass = half_mass * ((half_mass * upstream) * log_sum.exp())
    far_gradient = far_gradient.scatter(1, safe, -target_mass[:, None])
    return torch.where(far[:, None], far_gradient, gradient).to(logits.dtype)


class _Entropy(torch.autograd.Function):
    """Fused CUDA rows (where use_kernel), else float64 tensor statistics; higher derivatives
    differentiate the float64 tensor gradient."""

    @staticmethod
    def forward(
        ctx: _Context,
        logits: torch.Tensor,
        target: torch.Tensor,
        reduction: str,
        fp32: bool,
        use_kernel: bool,
    ) -> torch.Tensor:
        ctx.reduction, ctx.fused, ctx.fp32 = reduction, use_kernel, fp32
        if ctx.fused:
            kernel = _kernel()
            assert kernel is not None
            output, kernel_stats = kernel.forward(logits, target, reduction, fp32)
            ctx.save_for_backward(logits, target, kernel_stats)
            return output
        stats = _statistics(logits, target)
        ctx.save_for_backward(logits, target, *stats)
        dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
        return _values(stats, reduction).to(dtype)

    @staticmethod
    def backward(
        ctx: _Context, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        (upstream,) = grad_outputs
        logits, target, *saved = ctx.saved_tensors
        if ctx.fused and not torch.is_grad_enabled():
            kernel = _kernel()
            assert kernel is not None
            gradient = kernel.backward(logits, target, saved[0], upstream, ctx.reduction, ctx.fp32)
            return gradient, None, None, None, None
        if torch.is_grad_enabled():
            # Higher derivatives differentiate the float64 gradient.
            n = logits.shape[0]
            invalid = saved[0][5 * n : 6 * n] != 0 if ctx.fused else saved[0].isnan()
            stats = _statistics(logits, target, invalid)
        else:
            stats = tuple(saved)
        return _gradient(logits, target, stats, upstream, ctx.reduction), None, None, None, None


def entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    reduction: str,
    fp32: bool = False,
    is_fused: bool | None = None,
) -> torch.Tensor:
    """Validated contiguous (N,C) logits and (N,) int64 targets; first/higher derivatives.

    fp32 selects the float32 CUDA kernels; callers pass it only where fused() holds.
    is_fused=False keeps the tensor operations (backend="torch"); None checks fused().
    """
    use_kernel = logits.is_cuda and (fused(logits) if is_fused is None else is_fused)
    if use_kernel:
        if _ops.compiling():
            return _ops.entropy(logits, target, reduction, fp32)
        kernel = _kernel()
        assert kernel is not None
        out = kernel.apply(logits, target, reduction, fp32)
        if out is not None:
            return out
    return _Entropy.apply(logits, target, reduction, fp32, use_kernel)
