"""Integer-target hinge losses at k = 1 in fused CUDA kernels; first-order autograd.

The CPU kernels are in _cpu (SMOOTH is shared)."""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Protocol

import torch

from . import _cuda_launch, _ops
from ._first_order import first_order

# At k = 1 each pair is one loss: the max margin, or its projection onto sum a <= gamma.
SMOOTH = {
    "topk_hinge_alpha": False,
    "topk_hinge_beta": False,
    "topk_hinge_alpha_smooth": True,
    "topk_hinge_beta_smooth": True,
}


class _Context(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]
    reduction: str
    smooth: bool
    gamma: float

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


@functools.cache
def _kernel() -> ModuleType | None:
    if torch.version.cuda is None:
        return None
    try:
        import triton

        from . import _triton_hinge
    except ImportError:
        return None
    return _triton_hinge if _cuda_launch.kernels_supported(triton.__version__) else None


class _Hinge(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _Context,
        logits: torch.Tensor,
        target: torch.Tensor,
        reduction: str,
        smooth: bool,
        gamma: float,
    ) -> torch.Tensor:
        kernel = _kernel()
        assert kernel is not None
        output, stats = kernel.forward(logits, target, reduction, smooth, gamma)
        ctx.save_for_backward(logits, target, stats)
        ctx.reduction, ctx.smooth, ctx.gamma = reduction, smooth, gamma
        return output

    @staticmethod
    @first_order
    def backward(
        ctx: _Context, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        (upstream,) = grad_outputs
        logits, target, stats = ctx.saved_tensors
        kernel = _kernel()
        assert kernel is not None
        gradient = kernel.backward(
            logits, target, stats, upstream, ctx.reduction, ctx.smooth, ctx.gamma
        )
        return gradient, None, None, None, None


def covers(logits: torch.Tensor) -> bool:
    """Whether a fused CUDA kernel computes these contiguous (N, C) logits."""
    if not logits.is_cuda:
        return False
    kernel = _kernel()
    return kernel is not None and kernel.covers(logits)


def hinge(
    logits: torch.Tensor, target: torch.Tensor, objective: str, reduction: str, gamma: float
) -> torch.Tensor:
    """The k = 1 hinge objective of contiguous (N, C) CUDA logits and (N,) int64 targets."""
    if _ops.compiling():
        return _ops.hinge(logits, target, reduction, SMOOTH[objective], gamma)
    kernel = _kernel()
    assert kernel is not None
    out = kernel.apply(logits, target, reduction, SMOOTH[objective], gamma)
    if out is not None:
        return out
    return _Hinge.apply(logits, target, reduction, SMOOTH[objective], gamma)
