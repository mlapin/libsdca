"""The operators of libsdca.torch.

libsdca::loss and libsdca::loss_ are the row kernels every device implements (the CPU kernel
in _torch_cpu.cpp, built on first use; the CUDA kernel in triton/launch_torch.py): per-row
float64 values and the gradient of each row, scaled by upstream * row_upstream (loss_ writes
it into the scores). In eager mode the extension's apply computes a loss: forward the
values (on the CPU, with the gradient of a sum or mean at an upstream of 1), backward the
gradient of other upstreams, written over the logits where nothing else can read them.
Loss does in compiled graphs and under torch.func. Low-precision backward supplies the
cotangent to the row kernels before they round the result into the logits' dtype. Both are first order (differentiating the gradient raises
RuntimeError, forward mode too).
"""

from __future__ import annotations

import functools
import importlib.util
import operator
import sys
import threading
from pathlib import Path
from typing import Any

import torch

from . import _extensions
from . import _frontend as _f
from ._composed import xp_torch
from ._frontend import FIRST_ORDER, Rows
from ._torch_compat import is_compiling

_OPTIONS = (
    "int kind, int k, float gamma, float temperature, float label_smoothing, int ignore_index, "
    "bool ignore, int precision"
)
_ROW_ARGS = (
    f"Tensor? weight, Tensor? row_upstream, {_OPTIONS}, float upstream, bool values, bool grad, "
    "Tensor? divisor=None, Tensor? stats=None"
)
_lib = torch.library.Library("libsdca", "FRAGMENT")
_lib.define(f"loss(Tensor scores, Tensor target, {_ROW_ARGS}) -> (Tensor, Tensor)")
_lib.define(f"loss_(Tensor(a!) scores, Tensor target, {_ROW_ARGS}) -> Tensor")
_lib.define("reduce_rows(Tensor values, Tensor? terms, ScalarType dtype, int reduction) -> Tensor")
# The CUDA forward of the extension's apply: the values or their reduction (value dtype), a
# mean's divisor of label weights and the statistics its gradient reads (both else empty).
_lib.define(
    f"forward(Tensor scores, Tensor target, Tensor? weight, {_OPTIONS}, int reduction, "
    "bool save) -> (Tensor, Tensor, Tensor)"
)

REDUCTIONS = {"none": 0, "sum": 1, "mean": 2}
HALF = (torch.float16, torch.bfloat16)


def value_dtype(dtype: torch.dtype, target: torch.dtype | None = None) -> torch.dtype:
    return _f.value_dtype(torch, dtype, target)


def target_dtype(target: torch.Tensor) -> torch.dtype | None:
    """The dtype that a target adds to value_dtype: that of class probabilities or teachers."""
    return target.dtype if target.is_floating_point() else None


def _loss_fake(scores: torch.Tensor, target: torch.Tensor, *args: Any) -> tuple[torch.Tensor, ...]:
    values, grad = args[11], args[12]
    n = scores.shape[0] if values else 0
    g = scores.new_empty(scores.shape if grad else (0,))
    return scores.new_empty((n,), dtype=torch.float64), g


def _loss_inplace_fake(scores: torch.Tensor, target: torch.Tensor, *args: Any) -> torch.Tensor:
    return scores.new_empty((scores.shape[0] if args[11] else 0,), dtype=torch.float64)


def _reduce_fake(
    values: torch.Tensor, terms: torch.Tensor | None, dtype: torch.dtype, reduction: int
) -> torch.Tensor:
    return values.new_empty(values.shape[:-1], dtype=dtype)


_lib.impl("reduce_rows", _reduce_fake, "Meta")
_lib.impl("loss", _loss_fake, "Meta")
_lib.impl("loss_", _loss_inplace_fake, "Meta")


def _cpu_build() -> _extensions.Build:
    from torch.utils import cpp_extension

    def link() -> list[str]:
        dirs = cpp_extension.library_paths()
        flags = [f"-L{d}" for d in dirs] + [f"-Wl,-rpath,{d}" for d in dirs]
        flags += ["-lc10", "-ltorch_cpu", "-ltorch", "-ltorch_python"]
        # Python's symbols come from the interpreter that loads the module.
        return flags + (["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else [])

    def key() -> list[str]:
        abi = str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
        return [torch.__version__, torch.version.git_version, abi]

    def missing() -> str | None:
        if not (Path(_extensions.python_include()) / "Python.h").is_file():
            return "the Python development headers (Python.h) are not installed"
        if any((Path(d) / "torch/library.h").is_file() for d in cpp_extension.include_paths()):
            return None
        return "PyTorch ships no C++ headers (torch/library.h)"

    return _extensions.Build(
        env="SDCA_CPU_EXTENSION",
        source="_torch_cpu.cpp",
        what="the CPU kernels of libsdca.torch",
        cflags=lambda: [
            _standard(cpp_extension),
            *(["-DLIBSDCA_KEEP_GRAPH"] if _KEEP_GRAPH is not None else []),
            *(f"-I{d}" for d in cpp_extension.include_paths()),
            f"-I{_extensions.python_include()}",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            f'-DLIBSDCA_FIRST_ORDER="{FIRST_ORDER}"',
        ],
        link=link,
        key=key,
        missing=missing,
    )


def _standard(cpp_extension: Any) -> str:
    """The C++ standard of PyTorch's own extension builds."""
    source = Path(cpp_extension.__file__).read_text()
    return "-std=c++20" if "-std=c++20" in source else "-std=c++17"


def _import(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.cache
def _extension() -> tuple[Any, str]:
    return _extensions.load(_cpu_build(), _import)


@functools.cache
def cpu_kernels(kind: int | None = None) -> str | None:
    """None where the CPU kernels are registered (and compute loss kind), else why not: those
    of the extension, else of the native library (_frontend.cpu_route)."""
    module, reason = _extension()
    if module is None:
        return _f.cpu_route(reason, _native_kernels, "SDCA_CPU_EXTENSION", GUIDE)
    return None if kind is None else _f.missing_kernels(module.kinds(), kind)


GUIDE = "docs/TORCH.md#build-tools"


@functools.cache
def _native_kernels() -> str | None:
    """None where the native library's kernels are registered, else why not."""
    from . import _native, _torch_native

    reason = _native.rows_reason()
    if reason is None:
        _torch_native.register(_lib)
    return reason


def fallback(device: torch.device) -> str | None:
    """Why the CPU extension does not load where the native library computes the CPU calls
    (explain_backend's fallback), else None."""
    module, reason = _extension()
    return reason if device.type == "cpu" and module is None else None


def launcher(device: torch.device) -> str | None:
    """Why the extension does not load where Python launches the CUDA kernels (explain_backend),
    else None."""
    module, reason = _extension()
    return reason if device.type == "cuda" and module is None else None


def warn_launcher(scores: torch.Tensor) -> None:
    """A warning, once per process, where Python launches a CUDA call (scores on CUDA) because
    the extension does not load: unless its switch is off, and never in compiled graphs or
    torch.func transforms (which launch through Python by design)."""
    if not scores.is_cuda or is_compiling() or _transformed():
        return
    module, reason = _extension()
    if module is None and reason != "SDCA_CPU_EXTENSION=off":
        _f.warn_python_launcher(reason, "SDCA_CPU_EXTENSION", GUIDE)


def eager() -> Any:
    """The extension's eager autograd (apply), or None: in compiled graphs, under torch.func
    transforms and without the extension."""
    if is_compiling() or _transformed():
        return None
    return _extension()[0]


@functools.cache
def _cpu_reason(dtype: torch.dtype, kind: int) -> str | None:
    return _f.kernel_reason(dtype, "cpu", lambda: cpu_kernels(kind), lambda: None)


def kernels(
    device: torch.device, dtype: torch.dtype, n: int, classes: int, kind: int, k: int
) -> str | None:
    # CUDA coverage depends on a row limit. Keep the batch size out of the constant probe.
    probe_rows = _f.CUDA_MAX_ROWS + 1 if device.type == "cuda" and n > _f.CUDA_MAX_ROWS else 1
    return _kernel_reason(device, dtype, probe_rows, classes, kind, k)


@torch.compiler.assume_constant_result
def _kernel_reason(
    device: torch.device, dtype: torch.dtype, n: int, classes: int, kind: int, k: int
) -> str | None:
    """None where a row kernel computes (n, classes) scores, else why not."""
    platform = device.type
    if platform == "cpu":
        return _cpu_reason(dtype, kind)
    return _f.kernel_reason(
        dtype,
        platform,
        lambda: cpu_kernels(kind),
        lambda: _f.triton_reason("launch_torch", kind, k, n, classes, dtype),
    )


def divisor(
    target: torch.Tensor, weight: torch.Tensor | None, rows: Rows, n: int
) -> torch.Tensor | float:
    """The divisor of a mean of n rows (_frontend.mean_divisor; the extension's apply computes
    the same in C++)."""
    ignore_index = rows.ignore_index if rows.ignore else None
    labels = target.dtype == torch.int64
    total = _f.mean_divisor(xp_torch, target, weight, ignore_index, labels)
    return float(n) if isinstance(total, int) else total


def row_scale(scale: torch.Tensor | float, n: int) -> tuple[float, torch.Tensor | None]:
    """(upstream, row_upstream) of the row kernel for a gradient scale: a number, or a 0-d or
    (n,) tensor (no host synchronization)."""
    if not isinstance(scale, torch.Tensor):
        return float(scale), None
    return 1.0, scale.to(torch.float64).expand(n).contiguous()


try:
    from torch._dynamo.comptime import comptime
except ImportError:  # Earlier or alternative compilers keep the functional operator.
    comptime = None

_score_origin = threading.local()
_constant_result = getattr(torch.compiler, "assume_constant_result", None)


def _record_score_origin(context: Any) -> None:
    """Record the producer of _linear_scores' own argument during Dynamo capture."""
    _score_origin.linear = False
    try:
        node = context.get_local("scores").as_proxy().node
    except (AttributeError, KeyError, NotImplementedError):
        return
    if node.op == "call_method":
        _score_origin.linear = node.target in ("mm", "matmul")
    elif node.op == "call_function":
        _score_origin.linear = node.target in (
            operator.matmul,
            torch.matmul,
            torch.mm,
            torch.nn.functional.linear,
        )


def _take_score_origin() -> bool:
    linear = getattr(_score_origin, "linear", False)
    _score_origin.linear = False
    return linear


if _constant_result is not None:
    _take_score_origin = _constant_result(_take_score_origin)


def _linear_scores(scores: torch.Tensor) -> bool:
    # comptime callbacks return None. Read their result immediately on the capturing
    # thread. A placeholder, a view, or an unavailable callback uses the functional op.
    if comptime is None or _constant_result is None:
        return False
    comptime(_record_score_origin)
    return _take_score_origin()


def compiled_cpu_labels(scores: torch.Tensor, target: torch.Tensor, rows: Rows) -> bool:
    """Whether Inductor can reclaim a produced FP32 CPU label-CE score buffer."""
    return (
        is_compiling()
        and scores.device.type == "cpu"
        and scores.dtype == torch.float32
        and target.dtype == torch.int64
        and rows.kind == _f.KINDS["cross_entropy"]
        and not scores.is_leaf
        and not _transformed()
        and _linear_scores(scores)
    )


def evaluate(
    scores: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    rows: Rows,
    reduction: int,
    grad: bool,
    reuse: bool = False,
    groups: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The loss of (N, C) scores, reduced, and its gradient (empty unless grad)."""
    n = scores.shape[0]
    mean = reduction == REDUCTIONS["mean"]
    ignore = rows.ignore_index if rows.ignore else None
    terms = (
        _f.mean_terms(xp_torch, target, weight, ignore, target.dtype == torch.int64)
        if mean
        else None
    )
    per = n // groups if groups else n
    if terms is None:
        total = float(per) if mean else 1.0
    elif groups:
        total = terms.reshape(groups, per).sum(1).repeat_interleave(per)
    else:
        total = terms.sum()
    upstream, row_upstream = row_scale(1.0 / total if mean else 1.0, n)
    if reuse:
        # Inductor can elide this clone only when the original scores have no other reader.
        gradient = scores.clone(memory_format=torch.contiguous_format)
        values = torch.ops.libsdca.loss_(
            gradient, target, weight, row_upstream, *rows, upstream, True, True
        )
    else:
        values, gradient = torch.ops.libsdca.loss(
            scores, target, weight, row_upstream, *rows, upstream, True, grad
        )
    dtype = value_dtype(scores.dtype, target_dtype(target))
    if reduction == REDUCTIONS["none"]:
        return values.to(dtype), gradient
    if groups:
        values = values.reshape(groups, per)
    return torch.ops.libsdca.reduce_rows(values, terms, dtype, reduction), gradient


class _RowPullback(torch.autograd.Function):
    """Keep the actual cotangent in the row kernel under vmap(grad(loss))."""

    @staticmethod
    def forward(
        scores: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None,
        factors: torch.Tensor,
        rows: Rows,
    ) -> torch.Tensor:
        _, gradient = torch.ops.libsdca.loss(
            scores, target, weight, factors, *rows, 1.0, False, True
        )
        return gradient

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        pass

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, ...]:
        raise RuntimeError(FIRST_ORDER)

    @staticmethod
    def vmap(  # pyright: ignore[reportIncompatibleMethodOverride]
        info: Any,
        in_dims: tuple[Any, ...],
        scores: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None,
        factors: torch.Tensor,
        rows: Rows,
    ) -> tuple[torch.Tensor, int]:
        if in_dims[2] is not None:
            raise NotImplementedError("libsdca: class weights cannot be mapped (vmap)")

        def batch(x: torch.Tensor, dim: int | None) -> torch.Tensor:
            return x.movedim(dim, 0) if dim is not None else x.expand(info.batch_size, *x.shape)

        z = batch(scores, in_dims[0])
        y = batch(target, in_dims[1])
        u = batch(factors, in_dims[3])
        grad = _RowPullback.apply(
            z.flatten(0, 1).contiguous(),
            y.flatten(0, 1).contiguous(),
            weight,
            u.flatten(0, 1).contiguous(),
            rows,
        )
        return grad.reshape(z.shape), 0


def _weighted_gradient(
    scores: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    rows: Rows,
    reduction: int,
    upstream: torch.Tensor,
    groups: int = 0,
) -> torch.Tensor:
    """Apply the actual cotangent before rounding the row gradient into the logits' dtype."""
    n = scores.shape[0]
    factor = upstream.to(torch.float64)
    if reduction == REDUCTIONS["mean"]:
        if groups:
            per = n // groups
            ignore = rows.ignore_index if rows.ignore else None
            terms = _f.mean_terms(xp_torch, target, weight, ignore, target.dtype == torch.int64)
            total = float(per) if terms is None else terms.reshape(groups, per).sum(1)
            factor = factor / total
        else:
            factor = factor / divisor(target, weight, rows, n)
    if groups:
        factor = factor.repeat_interleave(n // groups)
    scale, factors = row_scale(factor, n)
    if _transformed() and factors is not None:
        return _RowPullback.apply(scores, target, weight, factors, rows)
    _, gradient = torch.ops.libsdca.loss(scores, target, weight, factors, *rows, scale, False, True)
    return gradient


class _Guard(torch.autograd.Function):
    """A first-order gradient: differentiating it raises."""

    generate_vmap_rule = True

    @staticmethod
    def forward(gradient: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        return gradient.view_as(gradient)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        pass

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, ...]:
        raise RuntimeError(FIRST_ORDER)


def guard(gradient: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
    """gradient, linked to inputs where a backward builds a graph (create_graph=True)."""
    return _Guard.apply(gradient, *inputs) if torch.is_grad_enabled() else gradient


def scaled(
    gradient: torch.Tensor, upstream: torch.Tensor, dtype: torch.dtype, factor_first: bool = False
) -> torch.Tensor:
    """gradient * upstream (a scalar or one factor per row), rounded once into dtype, with
    +0.0 at its zeros for either sign of upstream."""
    factor = upstream.to(gradient.dtype)
    factor = factor[:, None] if factor.ndim == 1 else factor
    # Inductor factors addcmul's unit coefficient into its first multiplicand.
    # Scaling the cotangent there avoids a separate pass over a saved score-sized tensor.
    a, b = (factor, gradient) if factor_first else (gradient, factor)
    return torch.addcmul(gradient.new_zeros(()), a, b).to(dtype)


_ACTIVE = getattr(torch._C, "_are_functorch_transforms_active", None)
_KEEP_GRAPH = getattr(
    getattr(torch._C, "_autograd", None), "_get_current_graph_task_keep_graph", None
)


def _keeps_graph() -> bool:
    """Whether the running backward keeps the graph (retain_graph, create_graph; unknown)."""
    return _KEEP_GRAPH is None or bool(_KEEP_GRAPH())


def _transformed() -> bool:
    return _ACTIVE is not None and bool(_ACTIVE())


class CudaLoss(torch.autograd.Function):
    """Reduced CUDA losses whose backward reads the original rows and their statistics."""

    @staticmethod
    def forward(
        ctx: Any,
        scores: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None,
        rows: Rows,
        reduction: int,
    ) -> torch.Tensor:
        mean = reduction == REDUCTIONS["mean"]
        value, total, stats = torch.ops.libsdca.forward(
            scores, target, weight, *rows, reduction, True
        )
        denominator = total if total.numel() else None
        ctx.save_for_backward(scores, target, weight, denominator, stats)
        ctx.rows = rows
        ctx.row_mean = mean and denominator is None
        return value

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
        (upstream,) = grad_outputs
        scores, target, weight, divisor, stats = ctx.saved_tensors
        scale = 1.0 / scores.shape[0] if ctx.row_mean else 1.0
        factor = upstream
        if ctx.row_mean and is_compiling():
            # A float custom-op argument specializes on its value. Carry the dynamic
            # row-mean scale in the existing tensor cotangent, at the kernel's precision.
            factor = upstream.to(torch.float64) * scale
            scale = 1.0
        _, gradient = torch.ops.libsdca.loss(
            scores,
            target,
            weight,
            factor.contiguous(),
            *ctx.rows,
            scale,
            False,
            True,
            divisor,
            stats if stats.numel() else None,
        )
        return guard(gradient, upstream, scores), None, None, None, None


class Loss(torch.autograd.Function):
    """A first-order row loss with an optional saved unit gradient. vmap folds the rows."""

    @staticmethod
    def forward(
        scores: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None,
        rows: Rows,
        reduction: int,
        grad: bool,
        reuse: bool = False,
        groups: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scores.is_cuda or (scores.dtype in HALF and (is_compiling() or _transformed())):
            # Checking a CUDA cotangent would wait on the device. In a transform
            # the cotangent may be traced. Compute the needed gradient in backward.
            grad = False
        return evaluate(scores, target, weight, rows, reduction, grad, reuse, groups)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]) -> None:
        ctx.mark_non_differentiable(output[1])
        ctx.set_materialize_grads(False)
        scores, target, weight, rows, reduction, _, ctx.reuse, ctx.groups = inputs
        ctx.dtype, ctx.reduction = scores.dtype, reduction
        ctx.rows = rows if scores.is_cuda or scores.dtype in HALF else None
        # A non-unit cotangent can revive a tail that would round to zero in
        # storage. Keep the original row inputs to apply it in the kernel.
        operands = (scores, target, weight) if ctx.rows is not None else (scores,)
        if is_compiling() or _transformed():
            # Under torch.func the scores link the gradient to them, so that differentiating
            # it raises.
            ctx.save_for_backward(
                output[1], *(operands if ctx.rows is not None or _transformed() else ())
            )
        else:
            # Eager backward hands the gradient over as its only reference, which autograd
            # then takes as the logits' gradient instead of copying it; the scores link it
            # to them under create_graph.
            ctx.gradient = output[1]
            ctx.save_for_backward(*operands)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
        upstream = grad_outputs[0]
        if hasattr(ctx, "gradient"):
            gradient, operands = ctx.gradient, ctx.saved_tensors
            if not _keeps_graph():
                del ctx.gradient
        else:
            gradient, *operands = ctx.saved_tensors
        if gradient.numel() == 0 and ctx.rows is None:
            raise RuntimeError("libsdca: the forward of this loss computed no gradient")
        unit = (
            gradient.numel() > 0
            and gradient.dtype == ctx.dtype
            and upstream.device.type == "cpu"
            and not is_compiling()
            and not _transformed()
            and bool((upstream == 1).all())
        )
        if unit:
            out = gradient
        elif ctx.rows is not None:
            scores, target, weight = operands
            out = _weighted_gradient(
                scores, target, weight, ctx.rows, ctx.reduction, upstream, ctx.groups
            )
        else:
            if ctx.groups:
                upstream = upstream.repeat_interleave(gradient.shape[0] // ctx.groups)
            out = scaled(gradient, upstream, ctx.dtype, ctx.reuse)
        return guard(out, upstream, *operands[:1]), None, None, None, None, None, None, None

    @staticmethod
    def vmap(  # pyright: ignore[reportIncompatibleMethodOverride]
        info: Any,
        in_dims: tuple[Any, ...],
        scores: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None,
        rows: Rows,
        reduction: int,
        grad: bool,
        reuse: bool = False,
        groups: int = 0,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
        if in_dims[2] is not None:
            raise NotImplementedError("libsdca: class weights cannot be mapped (vmap)")
        size = info.batch_size

        def front(x: torch.Tensor, d: int | None) -> torch.Tensor:
            return x.movedim(d, 0) if d is not None else x.expand(size, *x.shape)

        z, t = front(scores, in_dims[0]), front(target, in_dims[1])
        # The mapped wrapper can hide requires_grad from the public frontend;
        # the unwrapped input carries its enclosing vjp's requirement.
        grad = grad or z.requires_grad
        n, classes = z.shape[1], z.shape[2]
        flat_z = z.reshape(size * n, classes).contiguous()
        flat_t = t.reshape(size * n, *t.shape[2:]).contiguous()
        if size == 0:
            out_shape = (
                (size, n)
                if reduction == REDUCTIONS["none"]
                else ((size, groups) if groups else (size,))
            )
            dtype = value_dtype(scores.dtype, target_dtype(target))
            return (z.new_empty(out_shape, dtype=dtype), z.new_empty((size, 0))), (0, 0)
        count = size * (groups or 1) if reduction != REDUCTIONS["none"] else 0
        values, gradient = Loss.apply(flat_z, flat_t, weight, rows, reduction, grad, False, count)
        if reduction == REDUCTIONS["none"]:
            values = values.reshape(size, n)
        else:
            values = values.reshape((size, groups) if groups else (size,))
        gradient = (
            gradient.reshape(size, n, classes) if gradient.numel() else gradient.reshape(size, 0)
        )
        return (values, gradient), (0, 0)


def nan_weights(weight: torch.Tensor) -> torch.Tensor:
    """float64 class weights for the kernels (_frontend.nan_weights)."""
    return _f.nan_weights(xp_torch, weight.detach().to(torch.float64).contiguous())
