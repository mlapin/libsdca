"""Direct CPU first-order VJP for a differentiable FP32 teacher."""

from __future__ import annotations

from typing import Any

import torch

from . import _torch_ops
from ._composed import xp_torch
from ._composed._targets import cotangent

_LEGACY_BATCHED = getattr(getattr(torch._C, "_functorch", None), "is_legacy_batchedtensor", None)


class _TeacherGradient(xp_torch._TargetGradient):
    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (upstream,) = grad_outputs
        if (
            not ctx.needs_input_grad[1]
            or torch.is_grad_enabled()
            or type(upstream) is not torch.Tensor
            or upstream.is_neg()
            or (_LEGACY_BATCHED is not None and _LEGACY_BATCHED(upstream))
            or (extension := _torch_ops.eager()) is None
        ):
            return xp_torch._TargetGradient.backward(ctx, upstream)

        z, teacher = ctx.saved_tensors
        temperature = ctx.operands[-1]
        grad, exceptional = extension.teacher_backward(z, teacher, upstream, ctx.scale, temperature)
        if exceptional:
            # Preserve the existing IEEE results for each invalid row or nonfinite upstream.
            indices = torch.tensor(exceptional, dtype=torch.long, device=teacher.device)
            unit = ctx.gradient(xp_torch, z[indices], teacher[indices], temperature)
            factors = upstream[indices] if upstream.ndim else upstream
            grad[indices] = cotangent(xp_torch, unit, factors, ctx.scale, ctx.dtype)
        return upstream, grad, None, None, None, None, None


def attach(
    value: torch.Tensor, target: torch.Tensor, gradient: Any, scale: float, *operands: Any
) -> torch.Tensor:
    return _TeacherGradient.apply(value, target, gradient, scale, *operands)
