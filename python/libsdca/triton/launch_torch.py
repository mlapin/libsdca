"""The CUDA kernels of the torch.library operators libsdca::loss and libsdca::loss_ (schemas,
fakes, autograd and vmap in _torch_ops.py): per-row float64 values and the gradient times
upstream * row_upstream, launched by Triton on the scores' device and current stream with no
host synchronization. loss_ writes the gradient into the scores: every kernel reads a block
before it writes it, so the caller may hand over logits nothing else references.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any

import torch

from . import common
from .common import Call, Ref
from .math import bits

_NAMES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.bool: "bool",
    torch.uint8: "uint8",
}
_TORCH = {name: dtype for dtype, name in _NAMES.items()}
_smoothing: dict[tuple[int, float, int], torch.Tensor] = {}


@functools.cache
def available() -> str | None:
    """None where the kernels run, else why not."""
    try:
        import triton  # noqa: F401
    except ImportError:
        return "Triton is not installed (pip install 'libsdca[torch]' on Linux, or triton)"
    reason = common.triton_available()
    if reason is not None:
        return reason
    if not torch.cuda.is_available():
        return "no CUDA device"
    return None


def covers(kind: int, k: int, n: int, classes: int, dtype: torch.dtype) -> str | None:
    """None where the kernels compute a call, else why not."""
    return common.covers(kind, k, n, classes, _NAMES.get(dtype))


def _eps(device: torch.device, value: float) -> torch.Tensor:
    """Label smoothing as a float64 device scalar (the kernels read it from memory, as JAX's
    traced smoothing), cached per device, value and stream."""
    stream = torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else 0
    key = (device.index or 0, value, stream)
    if key not in _smoothing:
        if len(_smoothing) > 64:
            _smoothing.clear()
        with torch.inference_mode(False):
            _smoothing[key] = torch.full((), value, dtype=torch.float64, device=device)
    return _smoothing[key]


class _Launch:
    """Execute a warmed plan through Triton's public grid launcher."""

    def __init__(self, launch: common.Launch) -> None:
        self.kernel, self.grid = launch.kernel, (*launch.grid, 1, 1)[:3]
        self.args, self.meta = dict(launch.args), dict(launch.meta)

    def __call__(self, env: dict[str, Any], up: int) -> Any:
        args = {k: env[v.name] if isinstance(v, Ref) else v for k, v in self.args.items()}
        if "UP" in args:
            args["UP"] = up
        return self.kernel[self.grid](**args, **self.meta)


_SLOTS = {name: i for i, name in enumerate(("Z", "Y", "W", "R", "D", "S", "V", "L", "G", "E"))}


def _install_plan(module, arrays, launches, kernels, up):
    """Record the warmed launch ABI in eager autograd if the driver can run these kernels."""
    if any(k is None for k in kernels):
        return
    slots = dict(_SLOTS)
    allocation = []
    for a in arrays:
        slot = slots.setdefault(a.name, len(slots))
        if slot >= 12:
            return
        allocation.append((slot, _TORCH[a.dtype], a.shape))
    program = []
    for launch, kernel in zip(launches, kernels, strict=True):
        m = getattr(kernel, "metadata", None)
        if m is None or not hasattr(launch.kernel, "params") or not common.direct_launch(m):
            return
        signature = getattr(getattr(kernel, "src", None), "signature", {})
        cells, buffers, sizes = [], [], []
        for param in launch.kernel.params:
            if param.is_constexpr:
                continue
            value = launch.args[param.name]
            kind = signature.get(param.name, "")
            if isinstance(value, Ref) and kind.startswith("*"):
                cells.append(0)
                buffers.append(slots[value.name])
                sizes.append(8)
            elif not isinstance(value, Ref) and kind in ("i32", "u32", "i64", "u64"):
                cells.append(up if param.name == "UP" else value)
                buffers.append(-1)
                sizes.append(4 if kind.endswith("32") else 8)
            else:
                return
        program.append(
            ((kernel.function, *launch.grid, m.num_warps * 32, m.shared), cells, buffers, sizes)
        )
    module.cuda_install_plan(kernels, allocation, program)


@functools.lru_cache(maxsize=1024)
def _prepared(call: Call) -> tuple[tuple[common.Array, ...], tuple[_Launch, ...]]:
    plan = common.plan(call)
    return plan.arrays, tuple(_Launch(launch) for launch in plan.launches)


def run(
    scores: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    row_upstream: torch.Tensor | None,
    kind: int,
    k: int,
    gamma: float,
    temperature: float,
    label_smoothing: float,
    ignore_index: int,
    ignore: bool,
    precision: int,
    upstream: float,
    values: bool,
    grad: bool,
    divisor: torch.Tensor | None = None,
    stats: torch.Tensor | None = None,
    inplace: bool = False,
    reduction: int = 0,
    save: bool = False,
    value_dtype: str = "float64",
    shape_only: bool = False,
) -> dict[str, Any]:
    """The row kernel of (N, C) CUDA scores, contiguous or row-strided (rows of unit column
    stride: copied first unless the loss family's kernels take strides): the plan's arrays
    by name (V, L, D, S, G; the scores as G when inplace). row_upstream: one factor per row,
    a 0-d factor for every row or None (any float dtype); divisor: a 0-d float64 the
    gradient factors divide by; stats: what a forward with save returned as S; reduction: L
    holds the sum or mean of the values (value_dtype, also V's without a reduction), D a
    mean's divisor of label weights."""
    n, c = scores.shape
    stride = 0 if shape_only else row_stride(scores)
    # Kernels without strides take a contiguous copy (an inplace gradient goes back into the
    # caller's scores); its launches then do not install into the C++ replay, which would
    # bind the caller's scores instead.
    original = None
    if stride and not _strided(kind, k):
        original, scores, stride = scores, scores.contiguous(), 0
    rowwise = (
        common.UNIFORM if row_upstream is None
        else common.SCALAR if row_upstream.dim() == 0 else common.PER_ROW
    )  # fmt: skip
    saved = common.SAVE if save else common.SAVED if stats is not None and stats.numel() else 0
    # Fake tensors can have symbolic row counts, which cannot key a concrete launch cache.
    make_call = _call.__wrapped__ if shape_only else _call
    call = make_call(
        n, c, scores.dtype, target.dtype, kind, k, gamma, temperature, label_smoothing,
        ignore_index, ignore, precision, weighted=weight is not None, rowwise=rowwise,
        divided=divisor is not None, saved=saved, reduction=reduction, values=values, grad=grad,
        value_dtype=value_dtype, inplace=inplace, stride=stride,
    )  # fmt: skip
    env: dict[str, Any] = {"Z": scores, "Y": target, "W": weight, "R": row_upstream}
    if divisor is not None:
        env["D"] = divisor
    if saved == common.SAVED:
        env["S"] = stats
    prepare = _prepared.__wrapped__ if shape_only else _prepared
    arrays, launches = prepare(call)
    for array in arrays:
        env[array.name] = torch.empty(array.shape, dtype=_TORCH[array.dtype], device=scores.device)
    if shape_only:
        return env
    if call.smoothed:
        env["E"] = _eps(scores.device, label_smoothing)
    # CPU tensors run under TRITON_INTERPRET=1 (tests); Triton launches on the current device.
    other = scores.is_cuda and scores.device.index != torch.cuda.current_device()
    guard = torch.cuda.device(scores.device) if other else contextlib.nullcontext()
    if n > 0:
        up = bits(upstream)
        with guard:
            from .. import _torch_ops

            module = (
                _torch_ops._extension()[0]
                if scores.is_cuda and _torch_ops._extension.cache_info().currsize
                else None
            )
            if (
                module is not None
                and original is None
                and hasattr(module, "cuda_plan_requested")
                and module.cuda_plan_requested()
            ):
                compiled = [launch(env, up) for launch in launches]
                _install_plan(module, arrays, launches, compiled, up)
            else:
                for launch in launches:
                    launch(env, up)
    if inplace:
        if original is not None:
            original.copy_(scores)
            scores = original
        env["G"] = scores
    return env


@functools.lru_cache(maxsize=1024)
def _call(n, c, dtype, target, *options, weighted, rowwise, divided, saved, reduction, values,
          grad, value_dtype, inplace, stride=0) -> Call:  # fmt: skip
    """The Call of run's arguments (upstream is a launch argument, not part of the plan)."""
    return Call.of(
        n, c, _NAMES[dtype], _NAMES.get(target, "float32"), *options, weighted=weighted,
        rowwise=rowwise, divided=divided, saved=saved, reduction=reduction, values=values,
        grad=grad, value_dtype=value_dtype, inplace=inplace, stride=stride,
    )  # fmt: skip


def row_stride(scores: torch.Tensor) -> int:
    """The row stride of (N, C) scores whose rows are not contiguous with each other but have
    unit column stride (e.g. a slice x[:, a:b] of wider rows), else 0."""
    if scores.is_contiguous() or scores.dim() != 2:
        return 0
    n, c = scores.shape
    s0, s1 = scores.stride()
    return s0 if (s1 == 1 or c == 1) and s0 >= c and n > 1 else 0


@functools.lru_cache(maxsize=64)
def _strided(kind: int, k: int) -> bool:
    """Whether the kernels of a loss take row-strided scores."""
    return common.strided(Call(kind, 1, 2, "float32", "int64", k=k))


def _empty(scores: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return scores.new_empty((0,), dtype=dtype)


def _loss(scores: torch.Tensor, target: torch.Tensor, *args: Any) -> tuple[torch.Tensor, ...]:
    env = run(scores, target, *args)
    values, grad = args[11], args[12]
    v = env["V"] if values else _empty(scores, torch.float64)
    return v, env["G"] if grad else _empty(scores, scores.dtype)


def _loss_(scores: torch.Tensor, target: torch.Tensor, *args: Any) -> torch.Tensor:
    env = run(scores, target, *args, inplace=True)
    return env["V"] if args[11] else _empty(scores, torch.float64)


def _forward(
    scores: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    kind: int,
    k: int,
    gamma: float,
    temperature: float,
    label_smoothing: float,
    ignore_index: int,
    ignore: bool,
    precision: int,
    reduction: int,
    save: bool,
    *,
    shape_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """libsdca::forward: the values (value dtype) or their reduction, a mean's divisor of
    label weights (else empty) and the statistics a gradient reads (else empty)."""
    env = run(
        scores, target, weight, None, kind, k, gamma, temperature, label_smoothing, ignore_index,
        ignore, precision, 1.0, True, False, reduction=reduction, save=save, shape_only=shape_only,
        value_dtype="float64" if torch.float64 in (scores.dtype, target.dtype) else "float32",
    )  # fmt: skip
    # The operator schema promises three distinct outputs, even for empty auxiliaries.
    total = env["D"] if "D" in env else _empty(scores, torch.float64)
    stats = env["S"] if "S" in env else _empty(scores, torch.float64)
    return env["L" if reduction else "V"], total, stats


@functools.lru_cache(maxsize=256)
def _reduction_launch(
    n: int, rows: int, dtype: torch.dtype, reduction: int, terms: bool
) -> _Launch:
    from . import reduce

    # Only the reduction portion of the same plan is needed when the row values
    # are an existing operator output (e.g. inside a captured training graph).
    call = Call(
        0,
        n,
        1,
        "float64",
        "float64",
        reduction=reduction,
        reduction_rows=rows,
        value_dtype=_NAMES[dtype],
    )
    builder = common.Builder(call)
    reduce.add(builder, terms=terms)
    return _Launch(builder.launches[0])


def _reduce_rows(
    values: torch.Tensor, terms: torch.Tensor | None, dtype: torch.dtype, reduction: int
) -> torch.Tensor:
    out = values.new_empty(values.shape[:-1], dtype=dtype)
    n = values.numel()
    if n == 0:
        return out.fill_(float("nan") if reduction == common.MEAN else 0.0)
    divisor = torch.empty_like(out, dtype=torch.float64) if terms is not None else values
    env = {
        "V": values,
        "Z": values,
        "Y": terms if terms is not None else values,
        "L": out,
        "D": divisor,
    }
    with torch.cuda.device(values.device):
        _reduction_launch(
            n, values.shape[-1] if values.ndim == 2 else 0, dtype, reduction, terms is not None
        )(env, 0)
    return out


def register() -> None:
    """Register the CUDA kernels of libsdca::loss and libsdca::loss_ (defined by _torch_ops),
    once."""
    if not getattr(register, "done", False):
        torch.library.register_kernel("libsdca::reduce_rows", "cuda", _reduce_rows)
        torch.library.register_kernel("libsdca::loss", "cuda", _loss)
        torch.library.register_kernel("libsdca::loss_", "cuda", _loss_)
        torch.library.register_kernel("libsdca::forward", "cuda", _forward)
        torch.library.register_fake(
            "libsdca::forward", functools.partial(_forward, shape_only=True)
        )
        register.done = True  # pyright: ignore[reportFunctionMemberAccess]


if hasattr(torch.ops.libsdca, "loss"):
    register()
