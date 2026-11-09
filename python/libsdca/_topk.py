"""Integer-target top-k losses for 1 < k <= 32 in fused CUDA kernels; first-order autograd.

The CPU kernels are in _cpu (LOSSES holds the loss codes of both)."""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Protocol

import torch

from . import _cuda_launch, _ops
from ._first_order import first_order

LOSSES = {
    "topk_cross_entropy": 0,
    "topk_hinge_alpha": 1,
    "topk_hinge_beta": 2,
    "topk_hinge_alpha_smooth": 3,
    "topk_hinge_beta_smooth": 4,
}


class _Context(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]
    options: tuple[str, str, int, float, bool]

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


@functools.cache
def _kernel() -> ModuleType | None:
    if torch.version.cuda is None:
        return None
    try:
        import triton

        from . import _triton_topk
    except ImportError:
        return None
    return _triton_topk if _cuda_launch.kernels_supported(triton.__version__) else None


class _TopK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _Context,
        logits: torch.Tensor,
        target: torch.Tensor,
        reduction: str,
        objective: str,
        k: int,
        gamma: float,
        fp32: bool,
    ) -> torch.Tensor:
        kernel = _kernel()
        assert kernel is not None
        output, stats = kernel.forward(logits, target, reduction, objective, k, gamma, fp32)
        ctx.save_for_backward(logits, target, stats)
        ctx.options = (reduction, objective, k, gamma, fp32)
        return output

    @staticmethod
    @first_order
    def backward(
        ctx: _Context, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        (upstream,) = grad_outputs
        logits, target, stats = ctx.saved_tensors
        kernel = _kernel()
        assert kernel is not None
        gradient = kernel.backward(logits, target, stats, upstream, *ctx.options)
        return gradient, None, None, None, None, None, None


def covers(logits: torch.Tensor, objective: str, k: int) -> bool:
    """Whether a fused CUDA kernel computes this loss of contiguous (N, C) logits."""
    if not logits.is_cuda:
        return False
    kernel = _kernel()
    return kernel is not None and kernel.covers(logits, objective, k)


def topk(
    logits: torch.Tensor,
    target: torch.Tensor,
    objective: str,
    reduction: str,
    k: int,
    gamma: float,
    fp32: bool = False,
) -> torch.Tensor:
    """The top-k loss of contiguous (N, C) CUDA logits and (N,) int64 targets, where covers
    holds. fp32 takes the CUDA entropy's elementwise exponentials in float32.
    """
    if _ops.compiling():
        return _ops.topk(logits, target, reduction, objective, k, gamma, fp32)
    kernel = _kernel()
    assert kernel is not None
    out = kernel.apply(logits, target, reduction, objective, k, gamma, fp32)
    if out is not None:
        return out
    return _TopK.apply(logits, target, reduction, objective, k, gamma, fp32)
