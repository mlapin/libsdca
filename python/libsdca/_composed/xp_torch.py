"""The composed-loss namespace (_xp.Namespace) over PyTorch tensors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import torch
from torch._C._functorch import is_functorch_wrapped_tensor
from torch.nn import functional as F

from .._torch_compat import is_compiling
from ._xp import FIRST_ORDER, cpu_labels

State = TypeVar("State")
Carry = tuple[torch.Tensor, ...]

name = "torch"
float32 = torch.float32
float64 = torch.float64
int_ = torch.int64


def finfo(dtype: torch.dtype) -> torch.finfo:
    return torch.finfo(dtype)


def work(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def wide() -> torch.dtype:
    return torch.float64


def astype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.to(dtype)


def result_type(a: torch.Tensor, b: torch.Tensor) -> torch.dtype:
    return torch.promote_types(a.dtype, b.dtype)


def full_like(x: torch.Tensor, value: float, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.full_like(x, value, dtype=dtype)


def eager_labels(label: torch.Tensor, scores: torch.Tensor) -> torch.Tensor | None:
    if (
        is_compiling()
        or scores.device.type != "cpu"
        or is_functorch_wrapped_tensor(label)
        or is_functorch_wrapped_tensor(scores)
        or type(label) is not torch.Tensor
        or type(scores) is not torch.Tensor
    ):
        return None
    return torch.from_numpy(cpu_labels(label.numpy(), scores.shape[-1]))


def arange(n: int, like: torch.Tensor) -> torch.Tensor:
    return torch.arange(n, device=like.device)


def where(c: torch.Tensor, a: Any, b: Any) -> torch.Tensor:
    return torch.where(c, a, b)


exp = torch.exp
expm1 = torch.expm1
log = torch.log
log1p = torch.log1p
abs = torch.abs  # noqa: A001
isfinite = torch.isfinite
isnan = torch.isnan
nextafter = torch.nextafter
_SIGNED = {torch.float32: torch.int32, torch.float64: torch.int64}


def bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(_SIGNED[x.dtype])


def from_bits(bits: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return bits.view(like.dtype)


def maximum(a: torch.Tensor, b: Any) -> torch.Tensor:
    return torch.maximum(a, b) if isinstance(b, torch.Tensor) else a.clamp(min=b)


def minimum(a: torch.Tensor, b: Any) -> torch.Tensor:
    return torch.minimum(a, b) if isinstance(b, torch.Tensor) else a.clamp(max=b)


def clip(x: torch.Tensor, lo: Any, hi: Any) -> torch.Tensor:
    return minimum(maximum(x, lo), hi)


def sum(x: torch.Tensor) -> torch.Tensor:  # noqa: A001
    return x.sum(-1, keepdim=True)


def max(x: torch.Tensor) -> torch.Tensor:  # noqa: A001
    return x.amax(-1, keepdim=True)


def min(x: torch.Tensor) -> torch.Tensor:  # noqa: A001
    return x.amin(-1, keepdim=True)


def argmax(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bool:
        x = x.to(torch.uint8)
    return x.argmax(-1, keepdim=True)


def argmin(x: torch.Tensor) -> torch.Tensor:
    return x.argmin(-1, keepdim=True)


def count(mask: torch.Tensor) -> torch.Tensor:
    return mask.sum(-1, keepdim=True)


def all(mask: torch.Tensor) -> torch.Tensor:  # noqa: A001
    return mask.all(-1, keepdim=True)


def logsumexp(x: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(x, -1, keepdim=True)


def softmax(x: torch.Tensor) -> torch.Tensor:
    # As the derivative of logsumexp computes it.
    return (x - logsumexp(x)).exp()


def log_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(x, -1)


def sort(x: torch.Tensor) -> torch.Tensor:
    return x.sort(dim=-1, descending=True).values


def mean_margin(best: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from . import _losses, xp_torch

    return _losses.mean_margin(xp_torch, best, label)


def topk(x: torch.Tensor, k: int, *, stable: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    if not stable:
        values, index = torch.topk(x, k, dim=-1)
        return values, index
    if k == 1:
        index = argmax(x)
        return take(x, index), index
    values, index = x.sort(dim=-1, descending=True, stable=True)
    return values[..., :k], index[..., :k]


class _TakeOne(torch.autograd.Function):
    """One column per row. CUDA gather's atomic-add backward flushes subnormal gradients.
    Each row has a single destination, so its derivative can write it directly."""

    generate_vmap_rule = True

    @staticmethod
    def forward(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return x.gather(-1, index)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        x, index = inputs
        ctx.shape = x.shape
        ctx.save_for_backward(index)
        ctx.save_for_forward(index)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        (upstream,) = grad_outputs
        (index,) = ctx.saved_tensors
        zero = upstream.new_zeros(()).expand(ctx.shape)
        return zero.scatter(-1, index, upstream), None


class _TakeOneForward(_TakeOne):
    @staticmethod
    def jvp(ctx: Any, *tangents: torch.Tensor) -> torch.Tensor:
        dx, _ = tangents
        (index,) = ctx.saved_tensors
        return dx.gather(-1, index)


def take(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    rows = torch.broadcast_shapes(x.shape[:-1], index.shape[:-1])
    index = index.long().expand(*rows, index.shape[-1])
    x = x.expand(*rows, x.shape[-1])
    if x.is_cuda and x.dtype == torch.float32 and index.shape[-1] == 1:
        if is_compiling():
            return _TakeOne.apply(x, index)
        return _TakeOneForward.apply(x, index)
    return x.gather(-1, index)


def put(x: torch.Tensor, index: torch.Tensor, values: Any) -> torch.Tensor:
    index = index.long().expand(*x.shape[:-1], index.shape[-1])
    if isinstance(values, torch.Tensor):
        return x.scatter(-1, index, values.expand(index.shape))
    return x.scatter(-1, index, float(values))


def concat(xs: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(list(xs), -1)


def pad(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.pad(x, (0, size - x.shape[-1]))


def stop(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


class _Raise(torch.autograd.Function):
    """gradient, through a node whose derivatives raise."""

    generate_vmap_rule = True

    @staticmethod
    def forward(gradient: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        return gradient.detach().clone()

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        pass

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, ...]:
        raise RuntimeError(FIRST_ORDER)

    @staticmethod
    def jvp(ctx: Any, *tangents: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(FIRST_ORDER)


class _Attach(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(value: torch.Tensor, z: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        return value.detach().clone()

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        _, z, gradient = inputs
        ctx.save_for_backward(z, gradient)
        ctx.save_for_forward(z, gradient)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (upstream,) = grad_outputs
        z, gradient = ctx.saved_tensors
        if torch.is_grad_enabled():
            gradient = _Raise.apply(gradient, z, upstream)
        product = upstream[:, None] * gradient
        return None, torch.where(product == 0, 0.0, product), None


class _AttachForward(_Attach):
    """_Attach with forward mode: compiled graphs cannot trace a Function that defines jvp."""

    @staticmethod
    def jvp(ctx: Any, *tangents: torch.Tensor) -> torch.Tensor:
        z, gradient = ctx.saved_tensors
        return (_Raise.apply(gradient, z) * tangents[1]).sum(-1)


def attach(value: torch.Tensor, z: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    if is_compiling():
        return _Attach.apply(value, z, gradient)
    return _AttachForward.apply(value, z, gradient)


class _AttachSmooth(torch.autograd.Function):
    """_Attach whose gradient keeps the derivatives it has in z: a double backward
    differentiates it."""

    generate_vmap_rule = True

    @staticmethod
    def forward(value: torch.Tensor, z: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        return value.detach().clone()

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        ctx.save_for_backward(inputs[2])
        ctx.save_for_forward(inputs[2])

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (upstream,) = grad_outputs
        (gradient,) = ctx.saved_tensors
        product = upstream[:, None] * gradient
        return None, product + 0.0, None


class _AttachSmoothForward(_AttachSmooth):
    @staticmethod
    def jvp(ctx: Any, *tangents: torch.Tensor) -> torch.Tensor:
        (gradient,) = ctx.saved_tensors
        return (gradient * tangents[1]).sum(-1)


def attach_smooth(value: torch.Tensor, z: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    if is_compiling():
        return _AttachSmooth.apply(value, z, gradient)
    return _AttachSmoothForward.apply(value, z, gradient)


_SAVED = object()  # an operand saved for backward


class _TargetGradient(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(value: torch.Tensor, target: torch.Tensor, *args: Any) -> torch.Tensor:
        return value.view_as(value)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        _, target, gradient, scale, *operands = inputs
        tensors = [o for o in operands if isinstance(o, torch.Tensor)]
        ctx.save_for_backward(*tensors)
        ctx.operands = [_SAVED if isinstance(o, torch.Tensor) else o for o in operands]
        ctx.gradient, ctx.scale, ctx.dtype = gradient, scale, target.dtype

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (upstream,) = grad_outputs
        unused = (None,) * (len(ctx.operands) + 2)
        if not ctx.needs_input_grad[1]:
            return upstream, None, *unused
        from . import xp_torch
        from ._targets import cotangent

        tensors = iter(ctx.saved_tensors)
        operands = [next(tensors) if o is _SAVED else o for o in ctx.operands]
        gradient = ctx.gradient(xp_torch, *operands)
        if torch.is_grad_enabled():
            gradient = _Raise.apply(gradient, upstream)
        return upstream, cotangent(xp_torch, gradient, upstream, ctx.scale, ctx.dtype), *unused


def target_gradient(
    value: torch.Tensor, target: torch.Tensor, gradient: Any, scale: float, *operands: Any
) -> torch.Tensor:
    return _TargetGradient.apply(value, target, gradient, scale, *operands)


def grad(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x = x.detach().requires_grad_()
        (g,) = torch.autograd.grad(f(x).sum(), x)
    return g


def fori(n: int, body: Callable[[int, State], State], state: State) -> State:
    for i in range(n):
        state = body(i, state)
    return state


def _host_checks(state: Carry) -> int:
    """Steps between host checks of a loop's condition, 0 where they cannot run: compiled
    graphs, torch.func transforms and CUDA graph capture."""
    if is_compiling() or any(is_functorch_wrapped_tensor(t) for t in state):
        return 0
    if state[0].is_cuda:
        return 0 if torch.cuda.is_current_stream_capturing() else 16
    return 1


_TRACED_LOOPS = callable(getattr(torch, "while_loop", None))


def _host_condition(device: torch.device) -> bool:
    """Whether a compiled loop on device stops early: its condition is read without a
    synchronization."""
    return device.type == "cpu"


def while_loop(
    more: Callable[[Carry], torch.Tensor],
    body: Callable[[Carry], Carry],
    state: Carry,
    steps: int,
) -> Carry:
    if _TRACED_LOOPS and is_compiling():
        # The compiled loop reads its condition on the host every step: on another device
        # that synchronizes, so there it runs every step on a CPU count.
        early = _host_condition(state[0].device)

        # Inductor requires matching strides on entry to and return from the body.
        # An (N, 1) row reduction can have strides (1, N) or (1, 1). Carry the
        # row values as a view without that axis and restore it for the solve.
        row_column = tuple(t.ndim > 1 and t.shape[-1] == 1 for t in state)

        def pack(carry: Carry) -> Carry:
            return tuple(
                t.squeeze(-1) if column else t for t, column in zip(carry, row_column, strict=True)
            )

        def unpack(carry: Carry) -> Carry:
            return tuple(
                t.unsqueeze(-1) if column else t
                for t, column in zip(carry, row_column, strict=True)
            )

        def going(i: torch.Tensor, *carry: torch.Tensor) -> torch.Tensor:
            return (i < steps) & more(unpack(carry)) if early else i < steps

        def step(i: torch.Tensor, *carry: torch.Tensor) -> Carry:
            return (i + 1, *pack(body(unpack(carry))))

        device = state[0].device if early else torch.device("cpu")
        count = torch.zeros((), dtype=torch.int64, device=device)
        return unpack(tuple(torch.while_loop(going, step, (count, *pack(state)))[1:]))
    every = _host_checks(state)
    # On CUDA the first check comes after every steps: one before them only synchronizes.
    for i in range(steps):
        if every and (i + 1) % every == 0 and not bool(more(state)):
            break
        state = body(state)
    return state
