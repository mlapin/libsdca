"""Boolean-mask entropy and multilabel hinges in fused CUDA kernels (Triton launch).

The CPU kernels are in _cpu (OBJECTIVE_IDS holds the codes of both)."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol

import torch

from . import _cuda_launch, _ops
from ._first_order import first_order

OBJECTIVES = ("cross_entropy", "multilabel_hinge", "multilabel_hinge_smooth")
OBJECTIVE_IDS = {"cross_entropy": 0, "multilabel_hinge": 1, "multilabel_hinge_smooth": 2}


class _Context(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]
    objective: str
    reduction: str
    gamma: float
    fp32: bool

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


@dataclass
class _EntropySaved:
    """What torch._TensorEntropy.backward reads from its context."""

    saved_tensors: tuple[torch.Tensor, ...]
    double: bool


@functools.cache
def _kernel() -> ModuleType | None:
    if torch.version.cuda is None:
        return None
    try:
        import triton

        from . import _triton_multilabel
    except ImportError:
        return None
    return _triton_multilabel if _cuda_launch.kernels_supported(triton.__version__) else None


def fp32(objective: str, logits: torch.Tensor, precision: str | None) -> bool:
    """Whether entropy takes float32 exponentials: non-float64 logits, unless
    precision="float64"."""
    return objective == "cross_entropy" and precision != "float64" and logits.dtype != torch.float64


def applies(
    objective: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    classes: int,
    backend: str,
    precision: str | None,
    reduction: str,
) -> bool:
    """Whether the fused CUDA kernels compute this call: backend "auto", logits with a boolean
    mask of their shape, no precision="float32" and a known reduction."""
    if objective not in OBJECTIVES or backend != "auto" or precision == "float32":
        return False
    if target.dtype != torch.bool or target.shape != logits.shape or not logits.is_cuda:
        return False
    if reduction not in ("none", "sum", "mean") or logits.numel() > classes * _cuda_launch.ROWS:
        return False
    kernel = _kernel()
    return kernel is not None and kernel.covers(classes)


def explain(
    objective: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    classes: int,
    backend: str,
    precision: str | None,
    reduction: str,
) -> tuple[str, str] | None:
    """explain_backend of a boolean mask on CUDA, else None."""
    if objective not in OBJECTIVES or target.dtype != torch.bool or not logits.is_cuda:
        return None
    if applies(objective, logits, target, classes, backend, precision, reduction):
        kernel = _kernel()
        assert kernel is not None
        single = fp32(objective, logits, precision)
        host, reason = kernel.launcher(
            logits.reshape(-1, classes), kernel._REDUCTIONS[reduction], objective, single
        )
        work = "float32 exponentials" if single else "float64"
        if host is not None:
            return (
                "cuda-launcher",
                "boolean-mask entropy and multilabel hinges in fused Triton kernels launched "
                f"from C++ autograd ({work})",
            )
        return (
            "triton",
            f"boolean-mask entropy and multilabel hinges in fused Triton kernels ({work}): {reason}",
        )
    return (
        "torch",
        "tensor operations: fused boolean-mask paths need Triton 3.3, no precision='float32', "
        "at most 2**28 rows and 67,107,840 classes",
    )


class _Hinge(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _Context,
        logits: torch.Tensor,
        mask: torch.Tensor,
        objective: str,
        reduction: str,
        gamma: float,
        fp32: bool,
    ) -> torch.Tensor:
        kernel = _kernel()
        assert kernel is not None
        output, stats = kernel.forward(logits, mask, objective, reduction, gamma, fp32)
        ctx.save_for_backward(logits, mask, stats)
        ctx.objective, ctx.reduction, ctx.gamma, ctx.fp32 = objective, reduction, gamma, fp32
        return output

    @staticmethod
    @first_order
    def backward(
        ctx: _Context, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        return _gradient(ctx, grad_outputs[0]), None, None, None, None, None


class _Entropy(_Hinge):
    @staticmethod
    def backward(
        ctx: _Context, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        (upstream,) = grad_outputs
        if not torch.is_grad_enabled():
            return _gradient(ctx, upstream), None, None, None, None, None
        # Higher derivatives: the differentiable tensor backward in float64.
        from .torch import _TensorEntropy

        logits, mask, _ = ctx.saved_tensors
        rows = upstream.expand(logits.shape[0])
        if ctx.reduction == "mean":
            rows = rows / logits.shape[0]
        saved = _EntropySaved(saved_tensors=(logits, mask), double=True)
        return _TensorEntropy.backward(saved, rows)[0], None, None, None, None, None


def _gradient(ctx: _Context, upstream: torch.Tensor) -> torch.Tensor:
    logits, mask, stats = ctx.saved_tensors
    kernel = _kernel()
    assert kernel is not None
    return kernel.backward(
        logits, mask, stats, upstream, ctx.objective, ctx.reduction, ctx.gamma, ctx.fp32
    )


def loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    objective: str,
    reduction: str,
    gamma: float,
    precision: str | None = None,
) -> torch.Tensor:
    """The objective of contiguous (N, C) CUDA logits and a boolean mask of their shape."""
    single = fp32(objective, logits, precision)
    if _ops.compiling():
        return _ops.multilabel(logits, mask, objective, reduction, gamma, single)
    kernel = _kernel()
    assert kernel is not None
    out = kernel.apply(logits, mask, objective, reduction, gamma, single)
    if out is not None:
        return out
    function = _Entropy if objective == "cross_entropy" else _Hinge
    return function.apply(logits, mask, objective, reduction, gamma, single)
