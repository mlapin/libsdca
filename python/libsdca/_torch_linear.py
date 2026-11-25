"""linear_cross_entropy: cross_entropy(hidden @ weight.T + bias, target) without the (N, V)
logits.

Rows go through in chunks of at most chunk_bytes of logits. A chunk's logits (one GEMM in
the compute dtype) become, by the cross-entropy row kernel (libsdca::loss_), their row losses
and in place their gradient (class weights, label smoothing, ignored rows and the upstream
factor applied before its one rounding), which adds grad_hidden = G @ W, grad_weight += G^T @
h and grad_bias += sum(G) in float32 (float64 for float64 logits). Where a mean or sum needs
gradients the forward computes them at upstream derivative 1 and the backward scales them;
otherwise the backward recomputes each chunk.
"""

from __future__ import annotations

import contextlib
import functools
import typing
from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import _frontend as _f
from . import _torch_ops as _ops

CHUNK_BYTES, CHUNK_ROWS, CUDA_BF16_ROWS = 256 << 20, 2048, 8192
# cuBLAS GEMMs with a class count that is not a multiple of 8 leave its fast tensor-core
# kernels and run several times slower. On CUDA such weights are copied once per call into
# _PAD_CLASSES-aligned rows of zeros; the padded logits are exactly zero, the row kernel sees
# the (rows, classes) view (its gradient leaves them zero), and the weight and bias gradients
# are row slices of padded sums.
_PAD_CLASSES = 64
# Float32 copies of 16-bit CPU weights (whose 16-bit GEMMs are slow): blocks of this size.
_BLOCK_BYTES = 64 << 20
_LEGACY_BATCHED = getattr(getattr(torch._C, "_functorch", None), "is_legacy_batchedtensor", None)
_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

Grads = tuple[Tensor | None, Tensor | None, Tensor | None]


def chunk_rows(classes: int, dtype: torch.dtype, cuda: bool, budget: int) -> int:
    rows = (budget or CHUNK_BYTES) // (classes * dtype.itemsize)
    if budget == 0:
        rows = max(rows, CUDA_BF16_ROWS if cuda and dtype == torch.bfloat16 else CHUNK_ROWS)
    return max(1, rows)


def _blocks(classes: int, hidden: int) -> Iterator[slice]:
    rows = max(1, _BLOCK_BYTES // (4 * hidden))
    for begin in range(0, classes, rows):
        yield slice(begin, min(classes, begin + rows))


def _accumulate(total: Tensor, a: Tensor, b: Tensor, first: bool) -> None:
    """total (+)= a @ b, the product accumulated in total's dtype where the GEMM can."""
    if a.dtype != total.dtype:
        if a.is_cuda and a.dtype in _ops.HALF and hasattr(torch.ops.aten.mm, "dtype"):
            if first and hasattr(torch.ops.aten.mm, "dtype_out"):
                torch.mm(a, b, out_dtype=total.dtype, out=total)
                return
            product = torch.mm(a, b, out_dtype=total.dtype)
        else:
            product = torch.mm(a, b).to(total.dtype)
        total.copy_(product) if first else total.add_(product)
    elif first:
        torch.mm(a, b, out=total)
    else:
        total.addmm_(a, b)


class Options(typing.NamedTuple):
    """A call's ints for the operators: ignore_index, reduction, precision, chunk budget,
    the gradients the forward saves (a mask of hidden, weight, bias) and the dtype index."""

    ignore_index: int
    reduction: int
    precision: int
    budget: int
    save: int
    dtype: int


class _Chunks:
    """The chunk loop of one call: each chunk's logits, row losses and gradient (in place)."""

    def __init__(
        self,
        hidden: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        target: Tensor,
        class_weight: Tensor | None,
        o: Options,
        eps: float,
    ) -> None:
        self.hidden, self.target, self.class_weight, self.o, self.eps = (
            hidden,
            target,
            class_weight,
            o,
            eps,
        )
        self.dtype = _DTYPES[o.dtype]
        self.wide = hidden.device.type == "cpu" and self.dtype in _ops.HALF
        self.rows = chunk_rows(weight.shape[0], self.dtype, hidden.is_cuda, o.budget)
        self.accumulator = torch.float64 if self.dtype == torch.float64 else torch.float32
        self.w, self.b = weight.to(self.dtype), None if bias is None else bias.to(self.dtype)
        self.classes = classes = weight.shape[0]
        self.full: Tensor | None = None
        self.sums: Grads = (None, None, None)
        if hidden.is_cuda and classes % 8 and self.dtype != torch.float64:
            padded = -(-classes // _PAD_CLASSES) * _PAD_CLASSES
            w = self.w.new_zeros((padded, self.w.shape[1]))
            w[:classes].copy_(self.w)
            self.w = w
            if self.b is not None:
                b = self.b.new_zeros(padded)
                b[:classes].copy_(self.b)
                self.b = b

    def zeros(self, needs: tuple[bool, bool, bool], hidden_dtype: torch.dtype) -> Grads:
        """The gradient accumulators: the weight's and bias's as (classes,) row slices of the
        padded sums, which add() accumulates (self.sums)."""
        device = self.hidden.device
        self.sums = (
            torch.zeros(self.hidden.shape, dtype=hidden_dtype, device=device) if needs[0] else None,
            torch.empty(self.w.shape, dtype=self.accumulator, device=device) if needs[1] else None,
            torch.empty(self.w.shape[0], dtype=self.accumulator, device=device)
            if needs[2] and self.b is not None
            else None,
        )
        gh, gw, gb = self.sums
        n = self.classes
        return gh, None if gw is None else gw[:n], None if gb is None else gb[:n]

    def logits(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """A chunk's logits in the compute dtype, and its rows as the GEMMs take them."""
        h = h.to(self.dtype)
        if not self.wide:
            self.full = F.linear(h, self.w, self.b)
            return self.full[:, : self.classes], h
        h = h.float()
        z = h.new_empty((h.shape[0], self.w.shape[0]), dtype=self.dtype)
        for v in _blocks(self.w.shape[0], self.w.shape[1]):
            b = None if self.b is None else self.b[v].float()
            z[:, v] = F.linear(h, self.w[v].float(), b)
        return z, h

    def run(
        self, values: Tensor | None, upstream: float, rows: Tensor | None, grads: Grads
    ) -> None:
        """Each chunk's row losses into values (unless None) and its gradients (times
        upstream and rows, a float64 factor per row) added into grads."""
        gradient = any(g is not None for g in grads)
        first = True
        context = (
            torch.autocast(self.hidden.device.type, enabled=False)
            if _autocast(self.hidden.device.type)
            else contextlib.nullcontext()
        )
        with context, torch.no_grad():
            for part in self.parts():
                self.chunk(part, values, upstream, rows, grads, gradient, first)
                first = False
            if first:
                for g in grads[1:]:
                    if g is not None:
                        g.zero_()

    def parts(self) -> Iterator[slice | Tensor]:
        """Row selections of at most self.rows rows: slices, or on the CPU the indices of the
        rows not ignored where some are (CUDA passes them to the kernels, without a host
        synchronization)."""
        n = self.target.shape[0]
        if not self.target.is_cuda:
            ignored = self.target == self.o.ignore_index
            if bool(ignored.any()):
                index = (~ignored).nonzero().squeeze(1)
                for begin in range(0, index.numel(), self.rows):
                    yield index[begin : begin + self.rows]
                return
        for begin in range(0, n, self.rows):
            yield slice(begin, min(n, begin + self.rows))

    def chunk(
        self,
        part: slice | Tensor,
        values: Tensor | None,
        upstream: float,
        rows: Tensor | None,
        grads: Grads,
        gradient: bool,
        first: bool,
    ) -> None:
        z, h = self.logits(self.hidden[part])
        y = self.target[part].contiguous()
        args = (
            y,
            self.class_weight,
            None if rows is None else rows[part].contiguous(),
            0,
            1,
            0.0,
            1.0,
            self.eps,
            self.o.ignore_index,
            True,
            self.o.precision,
            upstream,
            values is not None,
        )
        if gradient:
            row_values = torch.ops.libsdca.loss_(z, *args, True)
        else:
            row_values = torch.ops.libsdca.loss(z, *args, False)[0]
        if values is not None:
            values[part] = row_values
        if gradient:
            self.add(grads, part, z, h, first)

    def add(self, grads: Grads, part: slice | Tensor, g: Tensor, h: Tensor, first: bool) -> None:
        if not self.wide:
            # The padded GEMM operands: the full logits (zero gradient in the padding) and
            # the padded sums of which grads holds the (classes,) slices.
            grad_hidden, grad_weight, grad_bias = self.sums
            g = g if self.full is None else self.full
            if grad_hidden is not None:
                if g.is_cuda and isinstance(part, slice):
                    _accumulate(grad_hidden[part], g, self.w, True)
                else:
                    grad_hidden[part] = torch.mm(g, self.w).to(grad_hidden.dtype)
            if grad_weight is not None:
                _accumulate(grad_weight, g.T, h, first)
            if grad_bias is not None:
                summed = g.sum(0, dtype=self.accumulator)
                grad_bias.copy_(summed) if first else grad_bias.add_(summed)
            return
        grad_hidden, grad_weight, grad_bias = grads
        out: Tensor | None = None
        for v in _blocks(self.w.shape[0], self.w.shape[1]):
            w, gv = self.w[v].float(), g[:, v].float()
            if grad_hidden is not None:
                out = torch.mm(gv, w) if out is None else out.addmm_(gv, w)
            if grad_weight is not None:
                _accumulate(grad_weight[v], gv.T, h, first)
            if grad_bias is not None:
                summed = gv.sum(0, dtype=self.accumulator)
                grad_bias[v].copy_(summed) if first else grad_bias[v].add_(summed)
        if grad_hidden is not None and out is not None:
            grad_hidden[part] = out.to(grad_hidden.dtype)


def _autocast(device: str) -> bool:
    from .torch import autocast_enabled

    return autocast_enabled(device)


def _saved(save: int) -> tuple[bool, bool, bool]:
    return bool(save & 1), bool(save & 2), bool(save & 4)


def _rows_options(o: Options) -> _f.Rows:
    return _f.Rows(0, ignore_index=o.ignore_index, ignore=True, precision=o.precision)


def _divisor(target: Tensor, class_weight: Tensor | None, o: Options) -> Tensor | float:
    if o.reduction != _ops.REDUCTIONS["mean"]:
        return 1.0
    return _ops.divisor(target, class_weight, _rows_options(o), target.shape[0])


def _forward(
    hidden: Tensor,
    weight: Tensor,
    bias: typing.Optional[Tensor],  # noqa: UP007, UP045 - the schema needs typing.Optional
    target: Tensor,
    class_weight: typing.Optional[Tensor],  # noqa: UP007, UP045
    ints: typing.List[int],  # noqa: UP006
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The loss, the mean's divisor (1 otherwise) and the gradients the forward saves (at
    upstream derivative 1, in the accumulator dtype; empty where not saved)."""
    o = Options(*ints)
    chunks = _Chunks(hidden, weight, bias, target, class_weight, o, eps)
    total = _divisor(target, class_weight, o)
    n = target.shape[0]
    values = torch.zeros(n, dtype=torch.float64, device=hidden.device)
    grads = chunks.zeros(_saved(o.save), chunks.accumulator)
    inverse = 1.0 / total
    # Only saved gradients consume these factors. Recompute applies the cotangent in backward.
    upstream, rows = _ops.row_scale(inverse, n) if o.save else (1.0, None)
    chunks.run(values, upstream, rows, grads)
    dtype = _ops.value_dtype(chunks.dtype)
    if o.reduction == _ops.REDUCTIONS["none"]:
        loss = values.to(dtype)
    else:
        loss = values.sum()
        loss = (loss * inverse if o.reduction == _ops.REDUCTIONS["mean"] else loss).to(dtype)
    divisor = torch.as_tensor(total, dtype=torch.float64, device=hidden.device)
    gh, gw, gb = (hidden.new_empty(0) if g is None else g for g in grads)
    return loss, divisor, gh, gw, gb


def _forward_fake(
    hidden: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    target: Tensor,
    class_weight: Tensor | None,
    ints: list[int],
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    o = Options(*ints)
    dtype = _DTYPES[o.dtype]
    shape = target.shape if o.reduction == _ops.REDUCTIONS["none"] else ()
    accumulator = torch.float64 if dtype == torch.float64 else torch.float32
    saved = _saved(o.save)

    def like(t: Tensor | None, save: bool) -> Tensor:
        return (
            hidden.new_empty(0)
            if t is None or not save
            else t.new_empty(t.shape, dtype=accumulator)
        )

    return (
        hidden.new_empty(shape, dtype=_ops.value_dtype(dtype)),
        hidden.new_empty((), dtype=torch.float64),
        like(hidden, saved[0]),
        like(weight, saved[1]),
        like(bias, saved[2]),
    )


def _recomputed(
    hidden: Tensor,
    weight: Tensor,
    bias: typing.Optional[Tensor],  # noqa: UP007, UP045
    target: Tensor,
    class_weight: typing.Optional[Tensor],  # noqa: UP007, UP045
    divisor: Tensor,
    upstream: Tensor,
    ints: typing.List[int],  # noqa: UP006
    eps: float,
    needs: typing.List[bool],  # noqa: UP006
) -> tuple[Tensor, Tensor, Tensor]:
    """The gradients of the forward's loss, each chunk's logits recomputed."""
    o = Options(*ints)
    chunks = _Chunks(hidden, weight, bias, target, class_weight, o, eps)
    n = target.shape[0]
    scale = upstream.to(torch.float64) / divisor
    up, rows = _ops.row_scale(scale.reshape(-1) if scale.ndim else scale, n)
    grads = chunks.zeros((needs[0], needs[1], needs[2]), hidden.dtype)
    chunks.run(None, up, rows, grads)
    gh, gw, gb = _finished(grads, weight, bias)
    return (
        hidden.new_empty(0) if gh is None else gh,
        hidden.new_empty(0) if gw is None else gw,
        hidden.new_empty(0) if gb is None else gb,
    )


def _recomputed_fake(
    hidden: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    target: Tensor,
    class_weight: Tensor | None,
    divisor: Tensor,
    upstream: Tensor,
    ints: list[int],
    eps: float,
    needs: list[bool],
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(hidden) if needs[0] else hidden.new_empty(0),
        torch.empty_like(weight) if needs[1] else hidden.new_empty(0),
        torch.empty_like(bias) if bias is not None and needs[2] else hidden.new_empty(0),
    )


def _finished(grads: Grads, weight: Tensor, bias: Tensor | None) -> Grads:
    """Accumulated gradients rounded once into the weight's and the bias's dtypes."""
    gh, gw, gb = grads
    return (
        gh,
        None if gw is None else gw.to(weight.dtype),
        None if gb is None or bias is None else gb.to(bias.dtype),
    )


def _setup(ctx: Any, inputs: tuple[Any, ...], output: tuple[Tensor, ...]) -> None:
    hidden, weight, bias, target, class_weight, ints, eps = inputs
    ctx.save_for_backward(hidden, weight, bias, target, class_weight, *output[1:])
    ctx.mark_non_differentiable(*output[1:])
    ctx.set_materialize_grads(False)
    ctx.ints, ctx.eps = ints, eps


@functools.cache
def _bfloat_scale_traffic(device: int) -> int | None:
    """Small conversions fit in L2 and benefit from ATen's lower launch overhead."""
    from .triton.common import native_bfloat_conversion

    if not native_bfloat_conversion(torch.cuda):
        return None
    capacity = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 0)
    return capacity if capacity > 0 else None


def _dense_bfloat_scale(gradient: Tensor, dtype: torch.dtype) -> bool:
    if not (
        not _ops.is_compiling()
        and gradient.dtype == torch.float32
        and dtype == torch.bfloat16
        and gradient.is_contiguous()
        and 0 < gradient.numel() < (1 << 31)  # Triton offsets use signed int32.
        and gradient.device.index == torch.cuda.current_device()
        and torch.version.cuda is not None
    ):
        return False
    try:
        from .triton.launch_torch import available
    except ImportError:
        return False
    threshold = _bfloat_scale_traffic(torch.cuda.current_device())
    # Each FP32 input is read once and packed to one BF16 output.
    return threshold is not None and 6 * gradient.numel() >= threshold and available() is None


def _scale_saved(gradient: Tensor, upstream: Tensor, dtype: torch.dtype) -> Tensor:
    """Keep the multiply in the accumulator dtype and round into the returned gradient."""
    factor = upstream.to(gradient.dtype)
    # Out variants lack a batching rule, and autograd needs a functional multiply
    # when it traces higher-order derivatives.
    if (
        gradient.is_cuda
        and gradient.dtype != dtype
        and not torch.is_grad_enabled()
        and _LEGACY_BATCHED is not None
        and not _LEGACY_BATCHED(upstream)
        and not _ops._transformed()
    ):
        if _dense_bfloat_scale(gradient, dtype):
            from .triton.linear import scale_bfloat

            return scale_bfloat(gradient, factor)
        return torch.mul(gradient, factor, out=torch.empty_like(gradient, dtype=dtype))
    return (gradient * factor).to(dtype)


def _backward(ctx: Any, upstream: Tensor | None, *_: Any) -> tuple[Tensor | None, ...]:
    if upstream is None:
        return (None,) * 7
    hidden, weight, bias, target, class_weight, divisor = ctx.saved_tensors[:6]
    o = Options(*ctx.ints)
    mask = _saved(o.save)
    if any(mask):
        up = upstream.reshape(())
        out = tuple(
            _scale_saved(g, up, inp.dtype) if keep and inp is not None else None
            for g, inp, keep in zip(
                ctx.saved_tensors[6:], (hidden, weight, bias), mask, strict=True
            )
        )
    else:
        needs = [bool(ctx.needs_input_grad[i]) for i in range(3)]
        gh, gw, gb = _linear_backward(
            hidden, weight, bias, target, class_weight, divisor, upstream, ctx.ints, ctx.eps, needs
        )
        out = tuple(g if need else None for g, need in zip((gh, gw, gb), needs, strict=True))
    out = tuple(None if g is None else _ops.guard(g, upstream) for g in out)
    return (*out, None, None, None, None)


if hasattr(torch.library, "custom_op"):
    _linear_op = torch.library.custom_op("libsdca::linear_cross_entropy", mutates_args=())(_forward)
    _linear_op.register_fake(_forward_fake)
    _linear_op.register_autograd(_backward, setup_context=_setup)
    _linear_grad_op = torch.library.custom_op(
        "libsdca::linear_cross_entropy_backward", mutates_args=()
    )(_recomputed)
    _linear_grad_op.register_fake(_recomputed_fake)
    _linear_forward = torch.ops.libsdca.linear_cross_entropy
    _linear_backward = torch.ops.libsdca.linear_cross_entropy_backward
else:
    # Before the public custom-op API, eager mode uses these same chunk routines
    # through the conventional autograd.Function interface.
    class _LinearFunction(torch.autograd.Function):
        forward = staticmethod(_forward)
        setup_context = staticmethod(_setup)
        backward = staticmethod(_backward)

    _linear_forward = _LinearFunction.apply
    _linear_backward = _recomputed


def compute_dtype(hidden: Tensor, weight: Tensor, bias: Tensor | None) -> torch.dtype:
    """The dtype of the logits: that of hidden, weight and bias, or under autocast the
    autocast dtype for those that are not float64 (as torch.nn.functional.linear)."""
    tensors = [hidden, weight] if bias is None else [hidden, weight, bias]
    dtypes = {t.dtype for t in tensors}
    if _autocast(hidden.device.type):
        get = getattr(torch, "get_autocast_dtype", None)
        cast = (
            get(hidden.device.type)
            if get is not None
            else (
                torch.get_autocast_cpu_dtype() if hidden.is_cpu else torch.get_autocast_gpu_dtype()
            )
        )
        dtypes = {d if d == torch.float64 or not d.is_floating_point else cast for d in dtypes}
    if len(dtypes) != 1 or (dtype := dtypes.pop()) not in _DTYPES:
        raise ValueError("hidden, weight and bias must share one floating-point dtype")
    return dtype


def _check_types(
    hidden: object,
    weight: object,
    target: object,
    bias: object,
    chunk_bytes: object,
    recompute: object,
) -> None:
    from .torch import _tensor

    for name, t in (("hidden", hidden), ("weight", weight), ("target", target)):
        _tensor(t, name)
    if bias is not None:
        _tensor(bias, "bias")
    if chunk_bytes is not None:
        _f.check_integer(chunk_bytes, "chunk_bytes", 1)
    _f.check_flag(recompute, "recompute")


def _check(hidden: Tensor, weight: Tensor, target: Tensor, bias: Tensor | None) -> None:
    """The shapes and devices of the linear layer; ValueError where invalid."""
    from .torch import device_error

    if hidden.ndim < 1 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[1]:
        raise ValueError("hidden must be (..., H) and weight (classes, H)")
    if bias is not None and bias.shape != weight.shape[:1]:
        raise ValueError("bias must have shape (classes,)")
    tensors = (hidden, weight, target) if bias is None else (hidden, weight, target, bias)
    if any(t.device != hidden.device for t in tensors):
        raise device_error("hidden", "weight", "target", "bias")
    _f.check_classes(1, weight.shape[0])


def _full_logits(
    hidden: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    target: Tensor,
    class_weight: Tensor | None,
    options: dict[str, Any],
    probabilities: bool,
) -> Tensor:
    """cross_entropy of the full logits (class probabilities, or backend="composed")."""
    from .torch import cross_entropy

    classes = weight.shape[0]
    logits = F.linear(hidden, weight, bias).reshape(-1, classes)
    rows = target.reshape(-1, classes) if probabilities else target.reshape(-1)
    value = cross_entropy(logits, rows, class_weight, **options)
    shape = target.shape[:-1] if probabilities else target.shape
    return value.reshape(shape) if options["reduction"] == "none" else value


def _prepare(
    hidden: Any,
    weight: Any,
    target: Any,
    bias: Any,
    class_weight: Any,
    options: dict[str, Any],
    chunk_bytes: Any,
    recompute: Any,
) -> tuple[_f.Call, float, tuple[Any, ...], bool]:
    from .torch import TORCH, _cross_entropy_call, device_error, full, kind_of

    call, eps = _cross_entropy_call(
        None, class_weight, options["ignore_index"], options["reduction"],
        options["label_smoothing"], options["backend"], options["precision"],
    )  # fmt: skip
    tensors = full(hidden, weight, target, bias, class_weight)
    hidden, weight, target, bias, class_weight = tensors
    _check_types(hidden, weight, target, bias, chunk_bytes, recompute)
    _check(hidden, weight, target, bias)
    logits = (*hidden.shape[:-1], weight.shape[0])
    role = _f.target_role(
        TORCH, call, kind_of(target), target.dtype, target.shape, logits, len(logits) - 1
    )
    probabilities = role == "probabilities"
    if class_weight is not None:
        _f.check_weight(kind_of(class_weight), class_weight.dtype, class_weight.shape,
                        weight.shape[0])  # fmt: skip
        if class_weight.device != hidden.device:
            raise device_error("weight", "logits")
    _f.check_precision(compute_dtype(hidden, weight, bias), call.precision)
    return call, eps, tensors, probabilities


def _check_kernel(hidden: Tensor, weight: Tensor, dtype: torch.dtype) -> None:
    n = hidden.numel() // hidden.shape[-1]
    reason = _ops.kernels(hidden.device, dtype, n, weight.shape[0], 0, 1)
    if reason is not None:
        raise _f.no_kernel(reason)


def _route(call: _f.Call, hidden: Tensor, weight: Tensor, bias: Tensor | None) -> tuple[str, str]:
    dtype = compute_dtype(hidden, weight, bias)
    name = str(dtype).removeprefix("torch.")
    if not call.composed:
        _check_kernel(hidden, weight, dtype)
    return _f.explanation(
        call.composed,
        hidden.device.type,
        _f.work(dtype, call.precision),
        f"composed cross-entropy of the full {name} logits",
        f" on chunks of the {name} logits between GEMMs",
        _ops.fallback(hidden.device),
    )


def linear_cross_entropy(
    hidden: Tensor,
    weight: Tensor,
    target: Tensor,
    bias: Tensor | None = None,
    *,
    class_weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    backend: str = "auto",
    precision: object = None,
    chunk_bytes: int | None = None,
    recompute: bool = False,
) -> Tensor:
    """cross_entropy(F.linear(hidden, weight, bias), target) without the (N, classes) logits.

    Rows go through in chunks of at most chunk_bytes of logits (at least one row; by default
    256 MiB or 2048 rows, whichever is more; CUDA BF16 uses at least 8192 rows), each turned
    into its gradient in place. With gradients to compute and a mean or sum, the forward
    computes them too (at upstream derivative 1): three GEMMs of the logits' size. Otherwise
    (recompute=True, reduction "none", float16 logits) the backward recomputes each chunk's
    logits: four GEMMs. Class probabilities (a floating-point target of shape (...,
    classes)) and backend="composed" compute the logits in full.

    Args:
        hidden: (..., H) floating-point features.
        weight: (classes, H) weights of the linear layer.
        target: Class labels (any integer dtype) of shape hidden.shape[:-1], or class
            probabilities of shape (*hidden.shape[:-1], classes).
        bias: (classes,) bias of the linear layer, or None.
        class_weight, ignore_index, reduction, label_smoothing, backend, precision: as the
            weight and the other options of cross_entropy.
        chunk_bytes: The memory budget of a chunk's logits in bytes (None: the default).
        recompute: Keep only the loss between forward and backward (one GEMM more).

    hidden, weight and bias share their dtype, which the logits take; under autocast the
    autocast dtype (float64 tensors keep theirs). The loss has dtype promote(that dtype,
    float32); the gradients have their inputs' dtypes, accumulated in float32 (float64 for
    float64 logits). Gradients are first order.
    """
    options = dict(
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
        backend=backend,
        precision=precision,
    )
    call, eps, tensors, probabilities = _prepare(
        hidden, weight, target, bias, class_weight, options, chunk_bytes, recompute
    )
    hidden, weight, target, bias, class_weight = tensors
    if probabilities or call.composed:
        return _full_logits(hidden, weight, bias, target, class_weight, options, probabilities)
    dtype = compute_dtype(hidden, weight, bias)
    _check_kernel(hidden, weight, dtype)
    shape = hidden.shape[:-1]
    if shape.numel() == 0:
        return _full_logits(hidden, weight, bias, target, class_weight, options, False)
    labels = (target if target.dtype == torch.int64 else target.long()).reshape(-1).contiguous()
    save = not recompute and reduction != "none" and dtype != torch.float16
    mask = 0
    if save and torch.is_grad_enabled():
        mask = int(hidden.requires_grad) | int(weight.requires_grad) << 1
        mask |= int(bias is not None and bias.requires_grad) << 2
    o = Options(
        typing.cast(int, call.ignore_index),
        _ops.REDUCTIONS[reduction],
        _f.PRECISIONS[call.precision],
        chunk_bytes or 0,
        mask,
        _DTYPES.index(dtype),
    )
    cw = None if class_weight is None else _ops.nan_weights(class_weight)
    flat = hidden.reshape(-1, hidden.shape[-1])
    loss = _linear_forward(flat, weight, bias, labels, cw, list(o), eps)[0]
    return loss.reshape(shape) if reduction == "none" else loss


def explain(
    hidden: Tensor,
    weight: Tensor,
    target: Tensor,
    bias: Tensor | None = None,
    class_weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    backend: str = "auto",
    precision: object = None,
    chunk_bytes: int | None = None,
    recompute: bool = False,
) -> tuple[str, str]:
    """explain_backend of linear_cross_entropy (its arguments)."""
    options = dict(
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
        backend=backend,
        precision=precision,
    )
    call, _, tensors, probabilities = _prepare(
        hidden, weight, target, bias, class_weight, options, chunk_bytes, recompute
    )
    hidden, weight, target, bias, class_weight = tensors
    name, reason = _route(call, hidden, weight, bias)
    if probabilities:
        return name, f"class probabilities: cross_entropy of the full logits ({reason})"
    return name, reason


class LinearCrossEntropyLoss(nn.Module):
    """linear_cross_entropy as a module; class_weight is a buffer moved by .to()."""

    class_weight: Tensor | None

    def __init__(
        self,
        class_weight: Tensor | None = None,
        *,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        backend: str = "auto",
        precision: object = None,
        chunk_bytes: int | None = None,
        recompute: bool = False,
    ) -> None:
        super().__init__()
        from .torch import _cross_entropy_call

        _cross_entropy_call(
            None, class_weight, ignore_index, reduction, label_smoothing, backend, precision
        )
        _check_types(torch.empty(0), torch.empty(0), torch.empty(0), None, chunk_bytes, recompute)
        self.register_buffer("class_weight", class_weight)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.backend = backend
        self.precision = precision
        self.chunk_bytes = chunk_bytes
        self.recompute = recompute

    def forward(
        self, hidden: Tensor, weight: Tensor, target: Tensor, bias: Tensor | None = None
    ) -> Tensor:
        """linear_cross_entropy(hidden, weight, target, bias) with the stored options."""
        return linear_cross_entropy(
            hidden, weight, target, bias, class_weight=self.class_weight, **self._options()
        )

    def _options(self) -> dict[str, Any]:
        return dict(
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            backend=self.backend,
            precision=self.precision,
            chunk_bytes=self.chunk_bytes,
            recompute=self.recompute,
        )

    def _explain(
        self, hidden: Tensor, weight: Tensor, target: Tensor, bias: Tensor | None = None
    ) -> tuple[str, str]:
        return explain(
            hidden, weight, target, bias, class_weight=self.class_weight, **self._options()
        )

    def extra_repr(self) -> str:
        return ", ".join(f"{key}={value!r}" for key, value in self._options().items())
