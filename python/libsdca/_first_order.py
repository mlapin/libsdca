"""The backward of first-order autograd Functions: differentiating its gradient raises."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

MESSAGE = (
    "This loss has first derivatives only: double backward is not supported. Only "
    "cross_entropy, multilabel_cross_entropy and topk_cross_entropy at k = 1 have higher "
    "derivatives (with boolean-mask targets, not in the native library)"
)


class _Raise(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gradient: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        return gradient.detach()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, ...]:
        raise RuntimeError(MESSAGE)


def guard(gradient: torch.Tensor, *inputs: torch.Tensor | None) -> torch.Tensor:
    """gradient, linked to the inputs that require gradients through a node that raises
    when it is differentiated (create_graph)."""
    linked = [t for t in inputs if t is not None and t.requires_grad]
    return _Raise.apply(gradient, *linked) if linked else gradient


def first_order(backward: Callable[..., Any]) -> Callable[..., Any]:
    """In place of once_differentiable: under create_graph the gradients depend on the
    saved inputs and the upstream gradients through a node that raises, so a gradient
    used inside another term (a penalty) cannot enter as a constant.
    """

    @functools.wraps(backward)
    def wrapper(ctx: Any, *grad_outputs: torch.Tensor) -> Any:
        with torch.no_grad():
            grads = backward(ctx, *grad_outputs)
        if not torch.is_grad_enabled():
            return grads
        inputs = (*ctx.saved_tensors, *grad_outputs)
        return tuple(guard(g, *inputs) if isinstance(g, torch.Tensor) else g for g in grads)

    return wrapper
