"""Fused linear cross-entropy: cross_entropy(hidden @ weight.T + bias, target) without the
(N, V) logits.

Rows go through in chunks of at most chunk_bytes of logits. A chunk's logits (a GEMM of the
compute dtype's operands with float32 accumulation, rounded once into that dtype as
torch.nn.functional.linear rounds them) give their row losses in float64 and row statistics by
cross_entropy's row kernels (the CPU extension, the native library, Triton or tensor
operations); the logits then become their gradient in place (scaled by the upstream derivative
and the class weight before its one rounding into the compute dtype), which adds
grad_hidden = G @ W, grad_weight += G^T @ h and grad_bias += sum(G) in float32 (float64 for
float64), each rounded once into its input's dtype. Where a mean or sum needs gradients, the
forward does all of it at upstream derivative 1 and the backward multiplies the gradients by
the upstream derivative (exact at 1); otherwise the forward keeps only the loss and the mean's
divisor, and the backward recomputes each chunk.

On the CPU, 16-bit GEMMs run as float32 GEMMs of the exactly converted 16-bit operands (ATen's
16-bit CPU GEMMs without AMX are several times slower), the weight's a block of classes at a
time; the logits are rounded into the 16-bit dtype before the loss, as a 16-bit GEMM rounds
them. Rows of the ignored label are skipped on
the CPU (on CUDA they pass through the kernels, which give them zero gradients, without
synchronizing).
"""

from __future__ import annotations

import contextlib
import math
import typing
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import _core, _cpu, _cpu_entropy, _cross_entropy, _cuda_launch, _entropy, _ops
from ._core import _canonical_precision
from ._first_order import first_order, guard
from ._linear_full_range import FullRange
from ._torch_distributed import has_dtensor

# The row kernels of a chunk; KERNELS names them as explain_backend does.
TORCH, NATIVE, EXTENSION, TRITON = 0, 1, 2, 3
KERNELS = ("torch", "native", "cpu-kernel", "triton")
_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_HALF = (torch.float16, torch.bfloat16)
# The default chunk: CHUNK_BYTES of logits (with their float32 copies on the CPU), or CHUNK_ROWS
# rows where those take more; shorter chunks slowed the GEMMs by up to 18% (32 CPU threads,
# 523 rows of 128256 classes).
CHUNK_BYTES, CHUNK_ROWS = 256 << 20, 2048
# Larger BF16 CUDA GEMMs amortize launches and weight-gradient updates:
# 8192 rows improved A100 BF16 timings at 32k and 128k classes. Explicit budgets
# still bound logits to the requested bytes (at least one row).
CUDA_BF16_ROWS = 8192
# Bytes per logit of the tensor kernel's float64 temporaries.
_TENSOR_BYTES = 32


def _torch() -> ModuleType:
    """libsdca.torch (which imports this module)."""
    from . import torch as mod

    return mod


def chunk_rows(classes: int, dtype: torch.dtype, kernel: int, budget: int) -> int:
    """Rows per chunk: the most whose logits (and the tensor kernel's float64 temporaries) fit
    in budget bytes, at least one; budget 0 is the default (CHUNK_BYTES, at least CHUNK_ROWS
    rows; CUDA BF16 GEMMs use at least CUDA_BF16_ROWS)."""
    per = dtype.itemsize + (_TENSOR_BYTES if kernel == TORCH else 0)
    rows = (budget or CHUNK_BYTES) // (classes * per)
    if budget == 0:
        minimum = CUDA_BF16_ROWS if kernel == TRITON and dtype == torch.bfloat16 else CHUNK_ROWS
        rows = max(rows, minimum)
    return max(1, min(rows, _cuda_launch.ROWS))


def _gemm16(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether the GEMMs run in float32 on exact copies of 16-bit operands (the CPU)."""
    return device.type == "cpu" and dtype in _HALF


# Float32 copies of 16-bit weights on the CPU: blocks of at most this many bytes.
_BLOCK_BYTES = 64 << 20


def _vocabulary(w: Tensor) -> Iterator[slice]:
    """Blocks of the rows (classes) of w whose float32 copies take at most _BLOCK_BYTES."""
    classes, rows = w.shape[0], max(1, _BLOCK_BYTES // (4 * w.shape[1]))
    for begin in range(0, classes, rows):
        yield slice(begin, min(classes, begin + rows))


def _logits(
    hidden: Tensor, w: Tensor, b: Tensor | None, dtype: torch.dtype, wide: bool
) -> tuple[Tensor, Tensor]:
    """A chunk's logits in the compute dtype and its hidden rows as the GEMMs take them (wide:
    float32 GEMMs of float32 copies of the operands, the weight's a block at a time, rounded
    into the 16-bit dtype)."""
    h = hidden.to(dtype)
    if not wide:
        return F.linear(h, w, b), h
    h = h.float()
    z = h.new_empty((h.shape[0], w.shape[0]), dtype=dtype)
    for v in _vocabulary(w):
        z[:, v] = F.linear(h, w[v].float(), None if b is None else b[v].float())
    return z, h


def _chunks(target: Tensor, ignore: int, rows: int) -> Iterator[slice | Tensor]:
    """Row selections of at most rows rows: slices, or on the CPU the indices of the rows not
    ignored where some are."""
    n = target.shape[0]
    if not target.is_cuda and bool((target == ignore).any()):
        index = (target != ignore).nonzero().squeeze(1)
        for begin in range(0, index.numel(), rows):
            yield index[begin : begin + rows]
        return
    for begin in range(0, n, rows):
        yield slice(begin, min(n, begin + rows))


def _row_weights(target: Tensor, weight: Tensor | None, ignore: int) -> Tensor:
    """float64 weights of the rows: weight[label] (1 without class weights), 0 where ignored."""
    ignored = target == ignore
    if weight is None:
        rows = torch.ones(target.shape, dtype=torch.float64, device=target.device)
    else:
        labels = target.masked_fill(ignored, 0).clamp(0, weight.shape[0] - 1)
        rows = weight.double()[labels]
    return rows.masked_fill(ignored, 0)


def _rows(
    kernel: int, z: Tensor, y: Tensor, weight: Tensor | None, ignore: int, fp32: bool
) -> tuple[Tensor, Any]:
    """Row losses of logits z (float64, times the class weight, 0 where ignored; NaN for invalid
    CUDA rows) and the statistics _gradient_ takes. The CPU kernels raise for invalid rows."""
    n, c = z.shape
    if kernel == EXTENSION:
        _, stats, _ = _cpu_entropy.operators().forward(z, y, 0, ignore, weight, fp32, True)
        maximum, label, total = stats[: 3 * n].view(n, 3).unbind(1)
        log_sum = total.log()
        loss = _entropy._values((maximum, label, log_sum, (maximum - label) + log_sum), "none")
        rows = _row_weights(y, weight, ignore)
        return torch.where(rows > 0, loss * rows, 0.0), stats
    if kernel == NATIVE:
        values = torch.empty(n, dtype=torch.float64)
        stats = torch.empty(n * 5 + 2, dtype=torch.float64)
        weights = _cpu._weights(weight)
        _cpu.native(
            _core._cross_entropy_forward,
            z.data_ptr(),
            _cpu.DTYPES[z.dtype],
            n,
            c,
            y.data_ptr(),
            ignore,
            _cpu._address(weights),
            values.data_ptr(),
            stats.data_ptr(),
            stats[5 * n :].data_ptr(),
            fp32=fp32,
        )
        return values, stats
    if kernel == TRITON:
        from . import _triton_weighted_entropy

        with _cuda_launch.device_guard(z):
            _, stats, _ = _triton_weighted_entropy._triton_forward(z, y, weight, ignore, 0, fp32)
        values = stats[4 * n : 5 * n]
        if weight is not None:
            # Status 3: an invalid class weight, which makes every row NaN.
            values = values.masked_fill(stats[5 * n] == 3, math.nan)
        return values, stats
    return _tensor_rows(z, y, weight, ignore)


def _tensor_rows(
    z: Tensor, y: Tensor, weight: Tensor | None, ignore: int
) -> tuple[Tensor, tuple[Any, ...]]:
    """_rows by float64 tensor operations (cross_entropy's backend="torch")."""
    mod, c = _torch(), z.shape[1]
    ignored = y == ignore
    labels = y.masked_fill(ignored, 0)
    if not z.is_cuda:
        mod._raise_failed(
            [
                (~mod._finite(z), "logits must be finite floating-point values"),
                (((labels < 0) | (labels >= c)).any(), "Target index out of range"),
            ]
        )
    rows = _row_weights(y, weight, ignore)
    if z.is_cuda:
        bad = ~ignored & ((labels < 0) | (labels >= c))
        if weight is not None:
            bad = bad | ~(torch.isfinite(weight).all() & (weight >= 0).all())
        rows = rows.masked_fill(bad, math.nan)
    labels = labels.clamp(0, c - 1)
    stats = _entropy._statistics(z, labels)
    return _cross_entropy._weighted_value(stats, rows, "none"), (labels, stats, rows)


def _gradient_(
    kernel: int,
    z: Tensor,
    y: Tensor,
    weight: Tensor | None,
    ignore: int,
    fp32: bool,
    stats: Any,
    scale: Tensor,
) -> None:
    """Logits z become the gradient of their rows' losses times scale (float64 (n,)) and the
    class weight, each entry rounded once into z's dtype; zero rows where ignored."""
    n, c = z.shape
    if kernel == EXTENSION:
        _cpu_entropy.operators().backward(z, y, stats, scale, 0, ignore, weight, fp32, z)
    elif kernel == NATIVE:
        weights = _cpu._weights(weight)
        _cpu.native(
            _core._cross_entropy_backward,
            z.data_ptr(),
            _cpu.DTYPES[z.dtype],
            n,
            c,
            y.data_ptr(),
            ignore,
            _cpu._address(weights),
            stats.data_ptr(),
            scale.data_ptr(),
            1,
            z.data_ptr(),
            fp32=fp32,
        )
    elif kernel == TRITON:
        from . import _triton_entropy

        with _cuda_launch.device_guard(z):
            _triton_entropy.launch_backward(
                z, y, stats, scale, 0, fp32, ignore, weight is not None, True, out=z
            )
    else:
        labels, row_stats, rows = stats
        z.copy_(_cross_entropy._weighted_gradient(z, labels, row_stats, rows, scale, "none"))


# torch.mm(..., out_dtype=) of 16-bit CUDA operands (PyTorch 2.8; no CPU kernel).
_MM_DTYPE = hasattr(torch.ops.aten.mm, "dtype")
_MM_DTYPE_OUT = hasattr(torch.ops.aten.mm, "dtype_out")
_ADDMM_DTYPE_OUT = hasattr(torch.ops.aten.addmm, "dtype_out")


def _mm(a: Tensor, b: Tensor, dtype: torch.dtype) -> Tensor:
    """a @ b, accumulated in float32 at least, rounded once into dtype where the GEMM can."""
    if a.dtype == dtype:
        return torch.mm(a, b)
    if _MM_DTYPE and a.is_cuda and a.dtype in _HALF and dtype == torch.float32:
        return torch.mm(a, b, out_dtype=dtype)
    return torch.mm(a, b).to(dtype)


def _add(total: Tensor, part: Tensor, first: bool) -> None:
    if first:
        total.copy_(part)
    else:
        total.add_(part)


def _cuda_accumulate(total: Tensor, a: Tensor, b: Tensor, first: bool) -> bool:
    """Write a half-input CUDA GEMM directly into its float32 accumulator where supported."""
    if not (a.is_cuda and a.dtype in _HALF and total.dtype == torch.float32):
        return False
    if first and _MM_DTYPE_OUT:
        torch.mm(a, b, out_dtype=total.dtype, out=total)
    elif not first and _ADDMM_DTYPE_OUT:
        torch.addmm(total, a, b, out_dtype=total.dtype, out=total)
    else:
        return False
    return True


def _accumulate(total: Tensor, a: Tensor, b: Tensor, first: bool) -> None:
    """total += a @ b (total = a @ b where first: no zeroed accumulator to read); 16-bit CUDA
    operands round their product once first."""
    if a.dtype != total.dtype:
        if not _cuda_accumulate(total, a, b, first):
            _add(total, _mm(a, b, total.dtype), first)
    elif first:
        torch.mm(a, b, out=total)
    else:
        total.addmm_(a, b)


def _no_autocast(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    """The GEMMs take the resolved dtype: no autocast inside the operators."""
    if _cross_entropy._autocast_enabled(device.type):
        return torch.autocast(device.type, enabled=False)
    return contextlib.nullcontext()


def _check_class_weight(weight: Tensor | None) -> None:
    if (
        weight is not None
        and not weight.is_cuda
        and not bool(torch.isfinite(weight).all() & (weight >= 0).all())
    ):
        raise ValueError("weight must be finite and nonnegative")


Grads = tuple[Tensor | None, Tensor | None, Tensor | None]


class _Chunks:
    """The chunk loop of one call: each chunk's logits, row losses and statistics, and the
    gradients that its rows add (G @ W into grad_hidden's rows, G^T @ h and sum(G) into
    float32 accumulators, float64 for float64 logits)."""

    def __init__(
        self,
        hidden: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        target: Tensor,
        class_weight: Tensor | None,
        ints: list[int],
    ) -> None:
        ignore, self.reduction, self.kernel, flag, code, budget, _ = ints
        self.hidden, self.target, self.ignore = hidden, target, ignore
        self.class_weight = class_weight
        # A 0-d flag set for a recomputed reduction: every row's gradient is NaN (see _poisoned).
        self.poison: Tensor | None = None
        self.dtype, self.fp32 = _DTYPES[code], bool(flag)
        full = self.dtype == torch.float64 or (
            class_weight is not None and class_weight.dtype == torch.float64
        )
        self.full = (
            FullRange(
                _row_weights(target, class_weight, ignore),
                next(k for k, v in _cpu.REDUCTIONS.items() if v == self.reduction),
                ignore,
            )
            if full
            else None
        )
        self.full_gradient = self.full is not None and class_weight is not None
        device = hidden.device
        self.wide = _gemm16(device, self.dtype)
        self.rows = chunk_rows(weight.shape[0], self.dtype, self.kernel, budget)
        self.accumulator = torch.float64 if self.dtype == torch.float64 else torch.float32
        self.weight, self.bias = weight, bias
        # The operands rounded into the compute dtype.
        self.w, self.b = weight.to(self.dtype), None if bias is None else bias.to(self.dtype)

    def zeros(self, needs: tuple[bool, bool, bool], hidden_dtype: torch.dtype) -> Grads:
        """Gradients where needs (grad_hidden zero, in hidden_dtype; the accumulators are
        written by the first chunk, or zeroed by run without one)."""
        device, classes = self.hidden.device, self.weight.shape[0]
        return (
            torch.zeros(self.hidden.shape, dtype=hidden_dtype, device=device) if needs[0] else None,
            torch.empty(self.weight.shape, dtype=self.accumulator, device=device)
            if needs[1]
            else None,
            torch.empty(classes, dtype=self.accumulator, device=device)
            if needs[2] and self.bias is not None
            else None,
        )

    def run(self, values: Tensor | None, scale: Tensor | None, grads: Grads) -> None:
        """Each chunk's row losses into values (unless None) and, given the rows' float64
        upstream factors scale ((n,) or a scalar), its gradients added into grads."""
        self.first = True
        self.invalid: Tensor | None = None
        # Rows of class weight 0 under an infinite factor (the mean of a batch whose rows weigh
        # 0 in total) have the undefined gradient 0 * inf, as in PyTorch, but the kernels skip
        # zero-weight rows. The CPU kernels' check, once per call: no device sync.
        self.undefined = (
            self.class_weight is not None
            and self.kernel in (NATIVE, EXTENSION)
            and scale is not None
            and scale.device.type == "cpu"
            and not bool(torch.isfinite(scale).all())
        )
        with _no_autocast(self.hidden.device), torch.no_grad():
            for part in _chunks(self.target, self.ignore, self.rows):
                # A call per chunk: its tensors are freed before the next chunk's logits.
                self.chunk(part, values, scale, grads)
            if self.first:
                for g in grads[1:]:
                    if g is not None:
                        g.zero_()

    def chunk(
        self, part: slice | Tensor, values: Tensor | None, scale: Tensor | None, grads: Grads
    ) -> None:
        z, h = _logits(self.hidden[part], self.w, self.b, self.dtype, self.wide)
        y = self.target[part].contiguous()
        rows, stats = _rows(self.kernel, z, y, self.class_weight, self.ignore, self.fp32)
        if rows.is_cuda:
            nan = rows.isnan().any()
            self.invalid = nan if self.invalid is None else self.invalid | nan
        full_stats = None
        if self.full is not None:
            full_stats = self.full.statistics(self.kernel, z, y, rows, stats)
        if values is not None:
            if self.full is not None and full_stats is not None:
                self.full.save(part, full_stats)
            else:
                values[part] = rows
        if scale is None:
            return
        factor = (scale[part] if scale.ndim else scale.expand(z.shape[0])).contiguous()
        if self.full_gradient and self.full is not None and full_stats is not None:
            z.copy_(self.full.gradient(z, y, full_stats, part, factor, self.poison))
        else:
            if self.kernel in (TORCH, TRITON):
                self.mark(z, y, stats, factor)
            _gradient_(self.kernel, z, y, self.class_weight, self.ignore, self.fp32, stats, factor)
        if self.undefined and self.class_weight is not None:
            # Before the GEMMs, so that it reaches every gradient.
            weightless = _row_weights(y, self.class_weight, self.ignore) == 0
            undefined = weightless & (y != self.ignore) & ~torch.isfinite(factor)
            z.masked_fill_(undefined[:, None], math.nan)
        del stats
        if self.wide:
            self.add_blocks(grads, part, z, h)
        else:
            self.add(grads, part, z, h)
        self.first = False

    def mark(self, z: Tensor, y: Tensor, stats: Tensor, factor: Tensor) -> None:
        """Mark the rows whose gradient is NaN in the weights the tensor (NaN) or Triton
        (negative) gradient takes from stats, also where it would skip the row: a pass over the
        rows, not the logits."""
        tensor = self.kernel == TORCH
        weights = stats[2] if tensor else stats[: 4 * z.shape[0]].view(-1, 4)[:, 3]
        undefined = self.poison
        if self.class_weight is not None:
            weightless = (weights == 0) & (y != self.ignore) & ~factor.isfinite()
            undefined = weightless if undefined is None else weightless | undefined
        if undefined is not None:
            weights.masked_fill_(undefined, math.nan if tensor else -1)

    def add(self, grads: Grads, part: slice | Tensor, g: Tensor, h: Tensor) -> None:
        grad_hidden, grad_weight, grad_bias = grads
        if grad_hidden is not None:
            grad_hidden[part] = _mm(g, self.w, grad_hidden.dtype)
        if grad_weight is not None:
            _accumulate(grad_weight, g.T, h, self.first)
        if grad_bias is not None:
            _add(grad_bias, g.sum(0, dtype=self.accumulator), self.first)

    def add_blocks(self, grads: Grads, part: slice | Tensor, g: Tensor, h: Tensor) -> None:
        """add of the 16-bit gradient g on the CPU: float32 GEMMs of float32 copies of g's and
        the weight's blocks of classes; grad_hidden's rows summed in float32, rounded once."""
        grad_hidden, grad_weight, grad_bias = grads
        rows: Tensor | None = None
        for v in _vocabulary(self.w):
            w, gv = self.w[v].float(), g[:, v].float()
            if grad_hidden is not None:
                rows = torch.mm(gv, w) if rows is None else rows.addmm_(gv, w)
            if grad_weight is not None:
                _accumulate(grad_weight[v], gv.T, h, self.first)
            if grad_bias is not None:
                _add(grad_bias[v], gv.sum(0, dtype=self.accumulator), self.first)
        if grad_hidden is not None and rows is not None:
            grad_hidden[part] = rows.to(grad_hidden.dtype)


def _saved(ints: list[int]) -> tuple[bool, bool, bool]:
    """The gradients a forward computes (ints[6], a mask of hidden, weight and bias)."""
    return bool(ints[6] & 1), bool(ints[6] & 2), bool(ints[6] & 4)


def _forward(
    hidden: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    target: Tensor,
    class_weight: Tensor | None,
    ints: list[int],
) -> tuple[Tensor, Tensor, Grads]:
    """The loss (reduced as ints[1]), the mean's divisor (the weight of the rows not ignored;
    1 for none and sum; NaN for a reduced loss with an invalid CUDA row, see _poisoned) and
    the gradients of _saved(ints) at upstream derivative 1, in the accumulator dtype. ints:
    ignore_index, reduction code, kernel, fp32, compute dtype (_DTYPES index), chunk bytes,
    the gradients to compute (_saved)."""
    _check_class_weight(class_weight)
    chunks = _Chunks(hidden, weight, bias, target, class_weight, ints)
    device = hidden.device
    divisor = torch.ones((), dtype=torch.float64, device=device)
    if chunks.reduction == _cpu.REDUCTIONS["mean"]:
        divisor = _row_weights(target, class_weight, chunks.ignore).sum()
    values = torch.zeros(hidden.shape[0], dtype=torch.float64, device=device)
    saved = _saved(ints)
    grads = chunks.zeros(saved, chunks.accumulator)
    scale = torch.ones_like(divisor) if chunks.full_gradient else 1 / divisor
    chunks.run(values, scale if any(saved) else None, grads)
    if chunks.invalid is not None and chunks.reduction != _cpu.REDUCTIONS["none"]:
        divisor = torch.where(chunks.invalid, math.nan, divisor)
    if chunks.full is not None:
        values = chunks.full.value()
    elif chunks.reduction != _cpu.REDUCTIONS["none"]:
        # This branch returns at most float32. A float64 tree sum has ample
        # guard digits; float64 inputs/weights keep FullRange's scaled sum.
        values = values.sum() / divisor
    return values.to(torch.promote_types(chunks.dtype, torch.float32)), divisor, grads


def _backward(
    hidden: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    target: Tensor,
    class_weight: Tensor | None,
    divisor: Tensor,
    upstream: Tensor,
    ints: list[int],
    needs: tuple[bool, bool, bool],
) -> Grads:
    """The gradients of _forward's loss in hidden, weight and bias (None where not needs),
    each chunk's logits recomputed."""
    chunks = _Chunks(hidden, weight, bias, target, class_weight, ints)
    scale = _poisoned(upstream.to(torch.float64), divisor)
    none = chunks.reduction == _cpu.REDUCTIONS["none"]
    grads = chunks.zeros(needs, hidden.dtype)
    factor = scale if chunks.full_gradient else scale / divisor
    if divisor.is_cuda and not none:
        chunks.poison = divisor.isnan()
    chunks.run(None, scale.reshape(-1) if none else factor, grads)
    return _finished(grads, weight, bias)


def _poisoned(upstream: Tensor, divisor: Tensor) -> Tensor:
    """upstream, or NaN where _forward's divisor is: an invalid CUDA row makes the reduced loss
    NaN and, as in the fused cross_entropy kernels, every gradient (the CPU raises instead).
    Folded into the factor that multiplies each gradient once: no pass of its own (a
    recomputation also marks the rows it would skip, see _Chunks.poison)."""
    return torch.where(divisor.isnan(), divisor.to(upstream.dtype), upstream)


def _finished(grads: Grads, weight: Tensor, bias: Tensor | None) -> Grads:
    """Accumulated gradients rounded once into the weight's and the bias's dtypes."""
    grad_hidden, grad_weight, grad_bias = grads
    return (
        grad_hidden,
        None if grad_weight is None else grad_weight.to(weight.dtype),
        None if grad_bias is None or bias is None else grad_bias.to(bias.dtype),
    )


def _fused_scale(
    grads: Grads, upstream: Tensor | None, hidden: Tensor, weight: Tensor, bias: Tensor | None
) -> Grads | None:
    """Fuse CUDA 16-bit output conversion with scaling; None keeps the tensor path."""
    inputs = (hidden, weight, bias)
    if (
        upstream is not None
        and not _ops.is_compiling()
        and hidden.is_cuda
        and _entropy._kernel() is not None
        and all(
            g is None
            or (
                x is not None
                and x.dtype in _HALF
                and g.dtype == torch.float32
                and g.is_contiguous()
                and g.device == upstream.device
            )
            for g, x in zip(grads, inputs, strict=False)
        )
    ):
        from ._triton_linear import scaled as fused_scaled

        return typing.cast(
            Grads,
            tuple(
                None if g is None or x is None else fused_scaled(g, upstream, x.dtype)
                for g, x in zip(grads, inputs, strict=False)
            ),
        )
    return None


def _scaled(
    grads: Grads,
    upstream: Tensor | None,
    hidden: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    inplace: bool,
) -> Grads:
    """A forward's gradients times the upstream derivative (None: 1), each rounded once into its
    input's dtype (multiplying by 1 is exact); inplace reuses them."""
    fused = _fused_scale(grads, upstream, hidden, weight, bias)
    if fused is not None:
        return fused
    if upstream is None:
        scaled = list(grads) if inplace else [None if g is None else g.clone() for g in grads]
    else:
        factor = upstream.reshape(())
        scaled = [
            None if g is None else g.mul_(factor.to(g.dtype)) if inplace else g * factor.to(g.dtype)
            for g in grads
        ]
    grad_hidden = None if scaled[0] is None else scaled[0].to(hidden.dtype)
    return _finished((grad_hidden, scaled[1], scaled[2]), weight, bias)


class _Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        target: Tensor,
        class_weight: Tensor | None,
        ints: list[int],
    ) -> Tensor:
        loss, divisor, grads = _forward(hidden, weight, bias, target, class_weight, ints)
        ctx.save_for_backward(hidden, weight, bias, target, class_weight, divisor)
        ctx.ints, ctx.grads = ints, grads
        return loss

    @staticmethod
    @first_order
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor | None, ...]:
        (upstream,) = grad_outputs
        hidden, weight, bias, target, class_weight, divisor = ctx.saved_tensors[:6]
        if any(_saved(ctx.ints)):
            inplace = not (_ops.is_compiling() or _cpu.keep_graph())
            # The CPU skips the exact multiplication by 1 (a pass over the weight's gradient).
            unit = not upstream.is_cuda and bool(upstream == 1)
            up = None if unit else _poisoned(upstream, divisor)
            grads = _scaled(ctx.grads, up, hidden, weight, bias, inplace)
            if inplace:
                ctx.grads = (None, None, None)
            return (*grads, None, None, None)
        needs = (ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2])
        grads = _backward(
            hidden, weight, bias, target, class_weight, divisor, upstream, ctx.ints, needs
        )
        return (*grads, None, None, None)


# torch.library.custom_op with register_fake and register_autograd (PyTorch 2.4).
AVAILABLE = hasattr(torch.library, "custom_op") and hasattr(torch.library, "register_autograd")

if AVAILABLE:

    @torch.library.custom_op("libsdca::linear_cross_entropy", mutates_args=())
    def _op(
        hidden: Tensor,
        weight: Tensor,
        bias: typing.Optional[Tensor],  # noqa: UP007 - the schema needs typing.Optional
        target: Tensor,
        class_weight: typing.Optional[Tensor],  # noqa: UP007
        ints: typing.List[int],  # noqa: UP006 - and typing.List before PyTorch 2.6
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        loss, divisor, grads = _forward(hidden, weight, bias, target, class_weight, ints)
        gh, gw, gb = (hidden.new_empty(0) if g is None else g for g in grads)
        return loss, divisor, gh, gw, gb

    @torch.library.custom_op("libsdca::linear_cross_entropy_backward", mutates_args=())
    def _op_backward(
        hidden: Tensor,
        weight: Tensor,
        bias: typing.Optional[Tensor],  # noqa: UP007
        target: Tensor,
        class_weight: typing.Optional[Tensor],  # noqa: UP007
        divisor: Tensor,
        upstream: Tensor,
        ints: typing.List[int],  # noqa: UP006
        needs: typing.List[bool],  # noqa: UP006
    ) -> tuple[Tensor, Tensor, Tensor]:
        gh, gw, gb = _backward(
            hidden,
            weight,
            bias,
            target,
            class_weight,
            divisor,
            upstream,
            ints,
            (needs[0], needs[1], needs[2]),
        )
        # Distinct empty outputs: operator outputs may not alias.
        return (
            hidden.new_empty(0) if gh is None else gh,
            hidden.new_empty(0) if gw is None else gw,
            hidden.new_empty(0) if gb is None else gb,
        )

    @_op.register_fake
    def _(
        hidden: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        target: Tensor,
        class_weight: Tensor | None,
        ints: list[int],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        shape = hidden.shape[:1] if ints[1] == _cpu.REDUCTIONS["none"] else ()
        dtype = torch.promote_types(_DTYPES[ints[4]], torch.float32)
        accumulator = torch.float64 if _DTYPES[ints[4]] == torch.float64 else torch.float32
        saved = _saved(ints)

        def like(t: Tensor | None, save: bool) -> Tensor:
            if t is None or not save:
                return hidden.new_empty(0)
            return t.new_empty(t.shape, dtype=accumulator)

        return (
            hidden.new_empty(shape, dtype=dtype),
            hidden.new_empty((), dtype=torch.float64),
            like(hidden, saved[0]),
            like(weight, saved[1]),
            like(bias, saved[2]),
        )

    @_op_backward.register_fake
    def _(
        hidden: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        target: Tensor,
        class_weight: Tensor | None,
        divisor: Tensor,
        upstream: Tensor,
        ints: list[int],
        needs: list[bool],
    ) -> tuple[Tensor, Tensor, Tensor]:
        return (
            torch.empty_like(hidden) if needs[0] else hidden.new_empty(0),
            torch.empty_like(weight) if needs[1] else hidden.new_empty(0),
            torch.empty_like(bias) if bias is not None and needs[2] else hidden.new_empty(0),
        )

    def _setup(ctx: Any, inputs: tuple[Any, ...], output: tuple[Tensor, ...]) -> None:
        hidden, weight, bias, target, class_weight, ints = inputs
        ctx.save_for_backward(hidden, weight, bias, target, class_weight, *output[1:])
        ctx.mark_non_differentiable(*output[1:])
        ctx.set_materialize_grads(False)
        ctx.ints = ints

    def _rule(ctx: Any, upstream: Tensor | None, *_: Any) -> tuple[Tensor | None, ...]:
        if upstream is None:
            return (None,) * 6
        saved, saved_grads = ctx.saved_tensors[:6], ctx.saved_tensors[6:]
        mask = _saved(ctx.ints)
        if any(mask):
            gh, gw, gb = (g if save else None for g, save in zip(saved_grads, mask, strict=True))
            up = _poisoned(upstream, saved[5])
            grads = _scaled((gh, gw, gb), up, saved[0], saved[1], saved[2], False)
            if torch.is_grad_enabled():
                # Differentiating these gradients raises.
                out = tuple(None if g is None else guard(g, *saved, upstream) for g in grads)
                return (*out, None, None, None)
            return (*grads, None, None, None)
        needs = [bool(ctx.needs_input_grad[i]) for i in range(3)]
        if torch.is_grad_enabled():
            ctx.grads = (None, None, None)
            return _Function.backward(ctx, upstream)
        grads = torch.ops.libsdca.linear_cross_entropy_backward(*saved, upstream, ctx.ints, needs)
        return (
            *(g if need else None for g, need in zip(grads, needs, strict=True)),
            None,
            None,
            None,
        )

    _op.register_autograd(_rule, setup_context=_setup)


def _autocast_dtype(device: str) -> torch.dtype:
    get = getattr(torch, "get_autocast_dtype", None)
    if get is not None:
        return get(device)
    # PyTorch 2.3 and older.
    return torch.get_autocast_cpu_dtype() if device == "cpu" else torch.get_autocast_gpu_dtype()


def compute_dtype(hidden: Tensor, weight: Tensor, bias: Tensor | None) -> torch.dtype:
    """The dtype of the logits: that of hidden, weight and bias, or under autocast the autocast
    dtype for those that are not float64 (as torch.nn.functional.linear casts them)."""
    tensors = [hidden, weight] if bias is None else [hidden, weight, bias]
    dtypes = {t.dtype for t in tensors}
    if _cross_entropy._autocast_enabled(hidden.device.type):
        cast = _autocast_dtype(hidden.device.type)
        dtypes = {d if d == torch.float64 or not d.is_floating_point else cast for d in dtypes}
    if len(dtypes) != 1 or (dtype := dtypes.pop()) not in _DTYPES:
        raise ValueError("hidden, weight and bias must share one floating-point dtype")
    return dtype


def _kernel(device: torch.device, dtype: torch.dtype, classes: int, backend: str) -> int:
    """The row kernel of backend ("auto", "native" or "torch") for these logits."""
    if backend not in ("auto", "native", "torch"):
        raise ValueError("Unknown backend")
    if device.type == "cpu":
        if backend == "torch":
            return TORCH
        auto = backend == "auto" and dtype != torch.float64 and _cpu_entropy.enabled()
        return EXTENSION if auto else NATIVE
    if backend == "native":
        raise ValueError("Native losses require CPU tensors; use backend='torch' on other devices")
    fused = backend == "auto" and device.type == "cuda" and classes <= _entropy.CLASSES
    return TRITON if fused and _cross_entropy._weighted_kernel() is not None else TORCH


def _check(
    hidden: Tensor, weight: Tensor, target: Tensor, bias: Tensor | None
) -> tuple[torch.Size, int]:
    """The shape of the examples and the class count; ValueError for invalid tensors."""
    tensors = (hidden, weight, target) if bias is None else (hidden, weight, target, bias)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError("hidden, weight, target and bias must be tensors")
    if hidden.ndim < 1 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[1]:
        raise ValueError("hidden must be (..., H) and weight (classes, H)")
    if bias is not None and bias.shape != weight.shape[:1]:
        raise ValueError("bias must have shape (classes,)")
    if any(t.device != hidden.device for t in tensors):
        raise ValueError("hidden, weight, target and bias must be on the same device")
    if weight.shape[0] < 2:
        raise ValueError("At least two classes are required")
    shape = hidden.shape[:-1]
    if target.dtype != torch.int64 or target.shape != shape:
        raise ValueError("target must contain int64 class indices of hidden.shape[:-1]")
    return shape, weight.shape[0]


def _check_options(
    class_weight: Tensor | None, classes: int, device: torch.device, chunk_bytes: int | None
) -> int:
    """The chunk budget (0: the default); ValueError for invalid class weights or budgets."""
    if class_weight is not None and (
        class_weight.shape != (classes,)
        or class_weight.device != device
        or not class_weight.is_floating_point()
    ):
        raise ValueError("weight must have shape (classes,) and the logits' device")
    if chunk_bytes is None:
        return 0
    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 1:
        raise ValueError("chunk_bytes must be a positive integer")
    return chunk_bytes


def _gradient_mask(hidden: Tensor, weight: Tensor, bias: Tensor | None) -> int:
    """The gradients a backward needs, as _saved's mask (0 without gradients)."""
    if not torch.is_grad_enabled():
        return 0
    has_bias = bias is not None and bias.requires_grad
    return int(hidden.requires_grad) | int(weight.requires_grad) << 1 | int(has_bias) << 2


def _composed(
    hidden: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    target: Tensor,
    class_weight: Tensor | None,
    options: dict[str, Any],
) -> Tensor:
    """cross_entropy of the full (N, classes) logits."""
    logits = F.linear(hidden, weight, bias).reshape(-1, weight.shape[0])
    value = _cross_entropy.cross_entropy(logits, target.reshape(-1), class_weight, **options)
    return value.reshape(target.shape) if options["reduction"] == "none" else value


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
    256 MiB or 2048 rows, whichever is more; CUDA BF16 uses at least 8192 rows), each
    turned into its gradient in place. With gradients to compute and a mean or sum, the
    forward computes them too (at upstream derivative 1; backward multiplies them by the
    upstream derivative): three GEMMs of the logits' size, as PyTorch's. Otherwise
    (recompute=True, reduction "none", float16 logits, whose gradient would be rounded before a
    GradScaler factor), the forward keeps only the loss and the backward recomputes each
    chunk's logits: four GEMMs. The loss and the logits' gradient carry cross_entropy's
    guarantees relative to the logits computed (no softmax collapse, entries rounded once);
    the GEMMs and their rounding are PyTorch's.

    Args:
        hidden: (..., H) floating-point features.
        weight: (classes, H) weights of the linear layer.
        target: int64 class indices of shape hidden.shape[:-1].
        bias: (classes,) bias of the linear layer, or None.
        class_weight, ignore_index, reduction, label_smoothing, backend, precision: as the
            weight and the other options of cross_entropy. backend "auto" takes the CPU
            extension (float64: the native library) or the fused CUDA kernels.
        chunk_bytes: The memory budget of a chunk's logits in bytes (None: the default).
        recompute: Keep only the loss between forward and backward (one GEMM more).

    hidden, weight and bias share their dtype, which the logits take; under autocast the
    autocast dtype (float64 tensors keep theirs). The loss has dtype promote(that dtype,
    float32); the gradients have their inputs' dtypes, accumulated in float32 (float64 for
    float64 logits). Gradients are first order.

    Raises:
        ValueError: For invalid inputs or options (on the CPU also for nonfinite logits or
            targets out of range; on CUDA those give NaN).
    """
    _cross_entropy._options(class_weight, ignore_index, reduction, label_smoothing)
    options: dict[str, Any] = dict(
        ignore_index=ignore_index, reduction=reduction, backend=backend, precision=precision
    )
    if has_dtensor(hidden, weight, target, bias, class_weight):
        # The full logits: correct, not memory-efficient.
        return _composed(hidden, weight, bias, target, class_weight, options)
    shape, classes = _check(hidden, weight, target, bias)
    budget = _check_options(class_weight, classes, hidden.device, chunk_bytes)
    dtype = compute_dtype(hidden, weight, bias)
    canonical = _canonical_precision(precision)
    if dtype == torch.float64 and canonical == "float32":
        raise ValueError("float64 logits require precision='float64' or None")
    kernel = _kernel(hidden.device, dtype, classes, backend)
    if shape.numel() == 0:
        return _composed(hidden, weight, bias, target, class_weight, options)
    fp32 = canonical != "float64" and dtype != torch.float64
    ints = [ignore_index, _cpu.REDUCTIONS[reduction], kernel, int(fp32), _DTYPES.index(dtype)]
    save = not recompute and reduction != "none" and dtype != torch.float16
    ints += [budget, _gradient_mask(hidden, weight, bias) if save else 0]
    flat = hidden.reshape(-1, hidden.shape[-1])
    labels = target.reshape(-1).contiguous()
    if AVAILABLE and _ops.is_compiling():
        loss = torch.ops.libsdca.linear_cross_entropy(
            flat, weight, bias, labels, class_weight, ints
        )[0]
    else:
        loss = _Function.apply(flat, weight, bias, labels, class_weight, ints)
    return loss.reshape(shape) if reduction == "none" else loss


def explain(
    hidden: Tensor, weight: Tensor, bias: Tensor | None = None, *, backend: str = "auto"
) -> tuple[str, str]:
    """(row kernel, compute dtype) of a linear_cross_entropy call of these tensors."""
    dtype = compute_dtype(hidden, weight, bias)
    kernel = _kernel(hidden.device, dtype, weight.shape[0], backend)
    return KERNELS[kernel], str(dtype).removeprefix("torch.")


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
        _cross_entropy._options(class_weight, ignore_index, reduction, label_smoothing)
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
            hidden,
            weight,
            target,
            bias,
            class_weight=self.class_weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            backend=self.backend,
            precision=self.precision,
            chunk_bytes=self.chunk_bytes,
            recompute=self.recompute,
        )

    def extra_repr(self) -> str:
        return (
            f"ignore_index={self.ignore_index}, reduction={self.reduction!r}, "
            f"backend={self.backend!r}, precision={self.precision!r}, "
            f"chunk_bytes={self.chunk_bytes}, recompute={self.recompute}"
        )
