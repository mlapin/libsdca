"""torch.library operators of the fused CUDA losses, for graph-break-free torch.compile.

Eager calls keep their direct launches (C++ autograd or Triton). Under torch.compile the
dispatchers call these operators instead. Each one is opaque to Dynamo: the launcher status,
kernel handles and the choice of C++ or Triton launch run inside it. A loss registers

* a forward operator returning the loss and the saved row statistics,
* a backward operator returning the gradient (both with fake implementations), and
* register_autograd: the backward operator, or under create_graph the eager rule (the
  float64 tensor gradient of the entropies; first-order losses raise).

Losses whose kernel returns the gradient with the loss register one operator that their
autograd Function calls.
"""

import typing
from typing import Any

import torch
from torch import Tensor

from . import _cuda_launch
from ._first_order import first_order

# torch.library.custom_op with register_fake and register_autograd (PyTorch 2.4). Only
# CUDA builds register: registration costs import time.
AVAILABLE = (
    torch.version.cuda is not None
    and hasattr(torch.library, "custom_op")
    and hasattr(torch.library, "register_autograd")
)
_is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", lambda: False)


def is_compiling() -> bool:
    """Whether Dynamo traces this call."""
    return _is_compiling()


def compiling() -> bool:
    """Whether Dynamo traces this call and the operators are registered."""
    return AVAILABLE and _is_compiling()


def transformed() -> bool:
    """Whether a torch.func transform (grad, vmap, jacrev, ...) wraps this call."""
    return torch._C._are_functorch_transforms_active()


def _values(logits: Tensor, reduction: str, dtype: torch.dtype) -> Tensor:
    """Fake loss: per row or reduced."""
    shape = (logits.shape[0],) if reduction == "none" else ()
    return logits.new_empty(shape, dtype=dtype)


def _stats(logits: Tensor, slots: int) -> Tensor:
    """Fake row statistics of the kernels reduced by _triton_reduce."""
    from . import _triton_reduce

    return logits.new_empty(_triton_reduce.size(logits.shape[0], slots), dtype=torch.float64)


def _context(logits: Tensor) -> int:
    """The CUDA context of logits, made current: autograd may run backward without one."""
    return _cuda_launch.ensure_context(logits.get_device())


def _register(forward: Any, backward: Any, fake: Any, fake_backward: Any, rule: Any) -> None:
    """Fakes and autograd of a forward (logits, target, *options) -> (loss, stats)."""
    forward.register_fake(fake)
    backward.register_fake(fake_backward)

    def setup(ctx: Any, inputs: tuple[Any, ...], output: tuple[Tensor, Tensor]) -> None:
        # rule reads (logits, target, stats) and the options; stats is not differentiable.
        ctx.save_for_backward(inputs[0], inputs[1], output[1])
        ctx.mark_non_differentiable(output[1])
        ctx.set_materialize_grads(False)
        ctx.options = inputs[2:]

    forward.register_autograd(rule, setup_context=setup)


def _empty_like(logits: Tensor, *args: Any) -> Tensor:
    return torch.empty_like(logits)


if AVAILABLE:
    # Integer-target entropy (_triton_entropy).

    @torch.library.custom_op("libsdca::entropy", mutates_args=(), device_types="cuda")
    def _entropy(
        logits: Tensor, target: Tensor, reduction: str, fp32: bool
    ) -> tuple[Tensor, Tensor]:
        from . import _triton_entropy

        return _triton_entropy.forward(logits, target, reduction, fp32)

    @torch.library.custom_op("libsdca::entropy_backward", mutates_args=(), device_types="cuda")
    def _entropy_backward(
        logits: Tensor, target: Tensor, stats: Tensor, upstream: Tensor, reduction: str, fp32: bool
    ) -> Tensor:
        from . import _triton_entropy

        _context(logits)
        return _triton_entropy.backward(logits, target, stats, upstream, reduction, fp32)

    def _entropy_fake(logits: Tensor, target: Tensor, reduction: str, fp32: bool) -> Any:
        from . import _triton_entropy

        half = logits.dtype in (torch.float16, torch.bfloat16)
        dtype = torch.float32 if half else logits.dtype
        return _values(logits, reduction, dtype), _stats(logits, _triton_entropy._LAYOUT[3])

    def _entropy_rule(ctx: Any, upstream: Tensor | None, _: Any) -> tuple[Tensor | None, ...]:
        if upstream is None:
            return None, None, None, None
        logits, target, stats = ctx.saved_tensors
        reduction, fp32 = ctx.options
        if torch.is_grad_enabled():
            # Higher derivatives: the eager float64 tensor gradient.
            from ._entropy import _Entropy

            ctx.reduction, ctx.fused = reduction, True
            return _Entropy.backward(ctx, upstream)[:4]
        gradient = _entropy_backward(logits, target, stats, upstream, reduction, fp32)
        return gradient, None, None, None

    _register(_entropy, _entropy_backward, _entropy_fake, _empty_like, _entropy_rule)

    # Weighted / ignored-label entropy (_triton_weighted_entropy).

    @torch.library.custom_op("libsdca::weighted_entropy", mutates_args=(), device_types="cuda")
    def _weighted(
        logits: Tensor,
        target: Tensor,
        weight: typing.Optional[Tensor],  # noqa: UP007 - the schema needs typing.Optional
        ignore_index: int,
        reduction: str,
        fp32: bool,
    ) -> tuple[Tensor, Tensor]:
        from . import _triton_weighted_entropy

        value, stats, _ = _triton_weighted_entropy.forward(
            logits, target, weight, ignore_index, reduction, fp32
        )
        return value, stats

    @torch.library.custom_op(
        "libsdca::weighted_entropy_backward", mutates_args=(), device_types="cuda"
    )
    def _weighted_backward(
        logits: Tensor,
        target: Tensor,
        weight: typing.Optional[Tensor],  # noqa: UP007
        stats: Tensor,
        upstream: Tensor,
        ignore_index: int,
        reduction: str,
        fp32: bool,
    ) -> Tensor:
        from . import _triton_weighted_entropy

        context = _context(logits)
        return _triton_weighted_entropy.backward(
            logits, target, weight, stats, upstream, ignore_index, reduction, context, fp32
        )

    def _weighted_fake(
        logits: Tensor,
        target: Tensor,
        weight: Tensor | None,
        ignore_index: int,
        reduction: str,
        fp32: bool,
    ) -> Any:
        from . import _triton_weighted_entropy

        full64 = logits.dtype == torch.float64
        size = _triton_weighted_entropy.stats_size(logits.shape[0], full64)
        stats = logits.new_empty(size, dtype=torch.float64)
        dtype = torch.float64 if full64 else torch.float32
        return _values(logits, reduction, dtype), stats

    def _weighted_setup(ctx: Any, inputs: tuple[Any, ...], output: tuple[Tensor, Tensor]) -> None:
        logits, target, weight, ignore_index, reduction, fp32 = inputs
        # As _cross_entropy._FusedWeightedEntropy, whose backward handles create_graph.
        ctx.save_for_backward(logits, target, target if weight is None else weight, output[1])
        ctx.mark_non_differentiable(output[1])
        ctx.set_materialize_grads(False)
        ctx.reduction, ctx.ignore_index, ctx.has_weight, ctx.fp32 = (
            reduction,
            ignore_index,
            weight is not None,
            fp32,
        )
        ctx.cuda_context = None

    def _weighted_rule(ctx: Any, upstream: Tensor | None, _: Any) -> tuple[Tensor | None, ...]:
        if upstream is None:
            return (None,) * 6
        logits, target, weight, stats = ctx.saved_tensors
        out_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        if torch.is_grad_enabled() or upstream.dtype != out_dtype:
            from ._cross_entropy import _FusedWeightedEntropy

            return _FusedWeightedEntropy.backward(ctx, upstream)
        gradient = _weighted_backward(
            logits,
            target,
            weight if ctx.has_weight else None,
            stats,
            upstream,
            ctx.ignore_index,
            ctx.reduction,
            ctx.fp32,
        )
        return gradient, None, None, None, None, None

    _weighted.register_fake(_weighted_fake)
    _weighted_backward.register_fake(_empty_like)
    _weighted.register_autograd(_weighted_rule, setup_context=_weighted_setup)

    # k = 1 hinges (_triton_hinge), first order.

    @torch.library.custom_op("libsdca::hinge", mutates_args=(), device_types="cuda")
    def _hinge(
        logits: Tensor, target: Tensor, reduction: str, smooth: bool, gamma: float
    ) -> tuple[Tensor, Tensor]:
        from . import _triton_hinge as kernel

        code = kernel._REDUCTIONS[reduction]
        host, handles = kernel.launcher(logits, code, smooth)
        if host is None:
            return kernel.forward(logits, target, reduction, smooth, gamma)
        g = kernel._gamma(logits.device, gamma)
        return host.hinge_cuda_forward(logits, target, g, code, handles)

    @torch.library.custom_op("libsdca::hinge_backward", mutates_args=(), device_types="cuda")
    def _hinge_backward(
        logits: Tensor,
        target: Tensor,
        stats: Tensor,
        upstream: Tensor,
        reduction: str,
        smooth: bool,
        gamma: float,
    ) -> Tensor:
        from . import _triton_hinge as kernel

        host, handles = kernel.launcher(logits, kernel._REDUCTIONS[reduction], smooth)
        if host is None:
            return kernel.backward(logits, target, stats, upstream, reduction, smooth, gamma)
        _context(logits)
        g = kernel._gamma(logits.device, gamma)
        return host.hinge_cuda_backward(logits, target, stats, upstream, g, handles)

    def _hinge_fake(logits: Tensor, target: Tensor, reduction: str, *args: Any) -> Any:
        from . import _triton_hinge

        dtype = torch.promote_types(logits.dtype, torch.float32)
        return _values(logits, reduction, dtype), _stats(logits, _triton_hinge._LAYOUT[3])

    @first_order
    def _hinge_gradient(ctx: Any, upstream: Tensor) -> tuple[Tensor | None, ...]:
        logits, target, stats = ctx.saved_tensors
        return (
            _hinge_backward(logits, target, stats, upstream, *ctx.options),
            None,
            None,
            None,
            None,
        )

    def _hinge_rule(ctx: Any, upstream: Tensor | None, _: Any) -> tuple[Tensor | None, ...]:
        return (None,) * 5 if upstream is None else _hinge_gradient(ctx, upstream)

    _register(_hinge, _hinge_backward, _hinge_fake, _empty_like, _hinge_rule)

    # 1 < k top-k losses (_triton_topk), first order; the C++ launch is the hinge entry.

    @torch.library.custom_op("libsdca::topk", mutates_args=(), device_types="cuda")
    def _topk(
        logits: Tensor,
        target: Tensor,
        reduction: str,
        objective: str,
        k: int,
        gamma: float,
        fp32: bool,
    ) -> tuple[Tensor, Tensor]:
        from . import _triton_topk as kernel

        code = kernel._REDUCTIONS[reduction]
        host, handles = kernel.launcher(logits, code, objective, k, fp32)
        if host is None:
            return kernel.forward(logits, target, reduction, objective, k, gamma, fp32)
        g = kernel._gamma(logits.device, gamma)
        return host.hinge_cuda_forward(logits, target, g, code, handles)

    @torch.library.custom_op("libsdca::topk_backward", mutates_args=(), device_types="cuda")
    def _topk_backward(
        logits: Tensor,
        target: Tensor,
        stats: Tensor,
        upstream: Tensor,
        reduction: str,
        objective: str,
        k: int,
        gamma: float,
        fp32: bool,
    ) -> Tensor:
        from . import _triton_topk as kernel

        code = kernel._REDUCTIONS[reduction]
        host, handles = kernel.launcher(logits, code, objective, k, fp32)
        if host is None:
            args = (reduction, objective, k, gamma, fp32)
            return kernel.backward(logits, target, stats, upstream, *args)
        _context(logits)
        g = kernel._gamma(logits.device, gamma)
        return host.hinge_cuda_backward(logits, target, stats, upstream, g, handles)

    def _topk_fake(logits: Tensor, target: Tensor, reduction: str, *args: Any) -> Any:
        from . import _triton_topk

        dtype = torch.promote_types(logits.dtype, torch.float32)
        return _values(logits, reduction, dtype), _stats(logits, _triton_topk._LAYOUT[3])

    @first_order
    def _topk_gradient(ctx: Any, upstream: Tensor) -> tuple[Tensor | None, ...]:
        logits, target, stats = ctx.saved_tensors
        gradient = _topk_backward(logits, target, stats, upstream, *ctx.options)
        return gradient, None, None, None, None, None, None

    def _topk_rule(ctx: Any, upstream: Tensor | None, _: Any) -> tuple[Tensor | None, ...]:
        return (None,) * 7 if upstream is None else _topk_gradient(ctx, upstream)

    _register(_topk, _topk_backward, _topk_fake, _empty_like, _topk_rule)

    # Boolean-mask entropy and multilabel hinges (_triton_multilabel).

    def _objective_id(objective: str) -> int:
        return {"cross_entropy": 0, "multilabel_hinge": 1}.get(objective, 2)

    @torch.library.custom_op("libsdca::multilabel", mutates_args=(), device_types="cuda")
    def _multilabel(
        logits: Tensor, mask: Tensor, objective: str, reduction: str, gamma: float, fp32: bool
    ) -> tuple[Tensor, Tensor]:
        from . import _triton_multilabel as kernel

        code = kernel._REDUCTIONS[reduction]
        host, handles = kernel.launcher(logits, code, objective, fp32)
        if host is None:
            return kernel.forward(logits, mask, objective, reduction, gamma, fp32)
        g, flags = kernel._gamma(logits.device, gamma), mask.view(torch.uint8)
        return host.multilabel_cuda_forward(
            logits, flags, g, _objective_id(objective), code, handles
        )

    @torch.library.custom_op("libsdca::multilabel_backward", mutates_args=(), device_types="cuda")
    def _multilabel_backward(
        logits: Tensor,
        mask: Tensor,
        stats: Tensor,
        upstream: Tensor,
        objective: str,
        reduction: str,
        gamma: float,
        fp32: bool,
    ) -> Tensor:
        from . import _triton_multilabel as kernel

        host, handles = kernel.launcher(logits, kernel._REDUCTIONS[reduction], objective, fp32)
        if host is None:
            args = (objective, reduction, gamma, fp32)
            return kernel.backward(logits, mask, stats, upstream, *args)
        _context(logits)
        g, flags = kernel._gamma(logits.device, gamma), mask.view(torch.uint8)
        return host.multilabel_cuda_backward(
            logits, flags, stats, upstream, g, _objective_id(objective), handles
        )

    def _multilabel_fake(
        logits: Tensor, mask: Tensor, objective: str, reduction: str, gamma: float, fp32: bool
    ) -> Any:
        from . import _triton_multilabel

        dtype = torch.promote_types(logits.dtype, torch.float32)
        slots = _triton_multilabel._LAYOUTS[objective][3]
        return _values(logits, reduction, dtype), _stats(logits, slots)

    @first_order
    def _multilabel_gradient(ctx: Any, upstream: Tensor) -> tuple[Tensor | None, ...]:
        logits, mask, stats = ctx.saved_tensors
        gradient = _multilabel_backward(logits, mask, stats, upstream, *ctx.options)
        return gradient, None, None, None, None, None

    def _multilabel_rule(ctx: Any, upstream: Tensor | None, _: Any) -> tuple[Tensor | None, ...]:
        if upstream is None:
            return (None,) * 6
        objective, reduction = ctx.options[:2]
        if objective == "cross_entropy" and torch.is_grad_enabled():
            from ._multilabel import _Entropy

            ctx.reduction = reduction
            return _Entropy.backward(ctx, upstream)
        return _multilabel_gradient(ctx, upstream)

    _register(_multilabel, _multilabel_backward, _multilabel_fake, _empty_like, _multilabel_rule)

    # distillation_loss (_triton_distillation), first order: 13 row statistics per row.

    @torch.library.custom_op("libsdca::distillation", mutates_args=(), device_types="cuda")
    def _distillation(
        student: Tensor, teacher: Tensor, temperature: float, double: bool, reduction: str
    ) -> tuple[Tensor, Tensor]:
        from . import _triton_distillation

        return _triton_distillation.forward(student, teacher, temperature, double, reduction)

    @torch.library.custom_op("libsdca::distillation_value", mutates_args=(), device_types="cuda")
    def _distillation_value(
        student: Tensor, teacher: Tensor, temperature: float, double: bool, reduction: str
    ) -> Tensor:
        from . import _triton_distillation

        return _triton_distillation.forward(student, teacher, temperature, double, reduction, True)[
            0
        ]

    @_distillation_value.register_fake
    def _distillation_value_fake(student: Tensor, teacher: Tensor, *args: Any) -> Tensor:
        return _values(student, args[-1], torch.promote_types(student.dtype, torch.float32))

    @torch.library.custom_op("libsdca::distillation_backward", mutates_args=(), device_types="cuda")
    def _distillation_backward(
        student: Tensor,
        teacher: Tensor,
        stats: Tensor,
        upstream: Tensor,
        temperature: float,
        double: bool,
        reduction: str,
    ) -> Tensor:
        from . import _triton_distillation

        args = (temperature, double, reduction)
        return _triton_distillation.backward(student, teacher, stats, upstream, *args)

    def _distillation_fake(student: Tensor, teacher: Tensor, *args: Any) -> Any:
        from . import _triton_distillation

        dtype = torch.promote_types(student.dtype, torch.float32)
        stats = _stats(student, _triton_distillation._SLOTS)
        return _values(student, args[-1], dtype), stats

    @first_order
    def _distillation_gradient(ctx: Any, upstream: Tensor) -> tuple[Tensor | None, ...]:
        student, teacher, stats = ctx.saved_tensors
        gradient = _distillation_backward(student, teacher, stats, upstream, *ctx.options)
        return gradient, None, None, None, None

    def _distillation_rule(ctx: Any, upstream: Tensor | None, _: Any) -> tuple[Tensor | None, ...]:
        return (None,) * 5 if upstream is None else _distillation_gradient(ctx, upstream)

    _register(
        _distillation, _distillation_backward, _distillation_fake, _empty_like, _distillation_rule
    )

    # Kernels returning the loss and its gradient, called by autograd Functions.

    def _loss_gradient_fake(logits: Tensor, *args: Any) -> Any:
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        return logits.new_empty(logits.shape[0], dtype=dtype), torch.empty_like(logits, dtype=dtype)

    @torch.library.custom_op("libsdca::proximal_distillation", mutates_args=(), device_types="cuda")
    def _proximal_distillation(
        student: Tensor, teacher: Tensor, temperature: float, alpha: float, double: bool
    ) -> tuple[Tensor, Tensor]:
        from . import _triton_distillation

        return _triton_distillation.proximal_distillation(
            student, teacher, temperature, alpha, double
        )

    @torch.library.custom_op("libsdca::proximal_entropy", mutates_args=(), device_types="cuda")
    def _proximal_entropy(
        logits: Tensor, positive: Tensor, alpha: float, fp32: bool
    ) -> tuple[Tensor, Tensor]:
        from . import _triton_prox

        return _triton_prox.proximal_entropy(logits, positive, alpha, fp32)

    def _proximal_entropy_fake(logits: Tensor, positive: Tensor, alpha: float, fp32: bool) -> Any:
        dtype = torch.float32 if fp32 else torch.float64
        saved = torch.promote_types(logits.dtype, torch.float32)
        return logits.new_empty(logits.shape[0], dtype=dtype), torch.empty_like(logits, dtype=saved)

    _proximal_distillation.register_fake(_loss_gradient_fake)
    _proximal_entropy.register_fake(_proximal_entropy_fake)


def entropy(logits: Tensor, target: Tensor, reduction: str, fp32: bool) -> Tensor:
    return torch.ops.libsdca.entropy(logits, target, reduction, fp32)[0]


def weighted_entropy(
    logits: Tensor,
    target: Tensor,
    weight: Tensor | None,
    ignore_index: int,
    reduction: str,
    fp32: bool,
) -> Tensor:
    op = torch.ops.libsdca.weighted_entropy
    return op(logits, target, weight, ignore_index, reduction, fp32)[0]


def hinge(logits: Tensor, target: Tensor, reduction: str, smooth: bool, gamma: float) -> Tensor:
    return torch.ops.libsdca.hinge(logits, target, reduction, smooth, gamma)[0]


def topk(
    logits: Tensor,
    target: Tensor,
    reduction: str,
    objective: str,
    k: int,
    gamma: float,
    fp32: bool,
) -> Tensor:
    return torch.ops.libsdca.topk(logits, target, reduction, objective, k, gamma, fp32)[0]


def multilabel(
    logits: Tensor, mask: Tensor, objective: str, reduction: str, gamma: float, fp32: bool
) -> Tensor:
    return torch.ops.libsdca.multilabel(logits, mask, objective, reduction, gamma, fp32)[0]


def distillation(
    student: Tensor, teacher: Tensor, temperature: float, double: bool, reduction: str
) -> Tensor:
    # Choose outside the custom operator: its implementation runs with grad mode off.
    from . import _triton_distillation

    if student.shape[1] > _triton_distillation._CHUNK and not (
        torch.is_grad_enabled() and student.requires_grad
    ):
        return torch.ops.libsdca.distillation_value(
            student, teacher, temperature, double, reduction
        )
    return torch.ops.libsdca.distillation(student, teacher, temperature, double, reduction)[0]


def proximal_distillation(
    student: Tensor, teacher: Tensor, temperature: float, alpha: float, double: bool
) -> tuple[Tensor, Tensor]:
    return torch.ops.libsdca.proximal_distillation(student, teacher, temperature, alpha, double)


def proximal_entropy(
    logits: Tensor, positive: Tensor, alpha: float, fp32: bool
) -> tuple[Tensor, Tensor]:
    return torch.ops.libsdca.proximal_entropy(logits, positive, alpha, fp32)
