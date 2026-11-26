"""The row kernels of libsdca.jax: one XLA FFI target, libsdca_loss, registered for the CPU by
_jax_cpu.cpp (built on first use) and for CUDA by triton/launch_jax.py, in a custom_vjp of
the reduced loss whose forward computes the values (on the CPU, with the gradient of a sum or
mean at a unit cotangent) and whose backward the gradient of other cotangents, scaled before
its one rounding (first order; forward mode raises).
"""

from __future__ import annotations

import ctypes
import functools
import importlib.metadata
import math
from pathlib import Path
from typing import Any

import jax
import jax.core
import jax.numpy as jnp
import numpy as np

from . import _extensions, _jax_alias, _jax_native, _jax_results, _native
from . import _frontend as _f
from ._frontend import KINDS, PRECISIONS, Rows

TARGET = "libsdca_loss"


@functools.cache
def _build() -> _extensions.Build:
    def missing() -> str | None:
        ffi = getattr(jax, "ffi", None)
        if ffi is None or not hasattr(ffi, "include_dir"):
            return f"jax {jax.__version__} has no jax.ffi (upgrade jax)"
        if not (Path(ffi.include_dir()) / "xla/ffi/api/ffi.h").is_file():
            return "jaxlib ships no XLA FFI headers"
        return None

    def key() -> list[str]:
        return [jax.__version__, importlib.metadata.version("jaxlib")]

    return _extensions.Build(
        env="SDCA_JAX_FFI",
        source="_jax_cpu.cpp",
        what="the CPU kernels of libsdca.jax",
        cflags=lambda: ["-std=c++17", f"-I{jax.ffi.include_dir()}"],
        link=lambda: [],
        key=key,
        missing=missing,
    )


def _register(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.LibsdcaKinds.restype = ctypes.c_uint64
    jax.ffi.register_ffi_target(TARGET, jax.ffi.pycapsule(library.LibsdcaLoss), platform="cpu")
    jax.ffi.register_ffi_target(
        "libsdca_teacher_vjp", jax.ffi.pycapsule(library.LibsdcaTeacherVjp), platform="cpu"
    )
    return library


@functools.cache
def _handler() -> tuple[Any, str]:
    """The loaded CPU handler (built on first use), or None and why not."""
    return _extensions.load(_build(), _register)


def cpu_kernels(kind: int) -> str | None:
    """None where the CPU handler is registered and computes loss kind, else why not, or
    None where the native library computes it (_frontend.cpu_route, _jax_native)."""
    library, reason = _handler()
    if library is None:
        return _f.cpu_route(reason, _native.rows_reason, "SDCA_JAX_FFI", "docs/JAX.md#build-tools")
    return _f.missing_kernels(library.LibsdcaKinds(), kind)


def route(device: str) -> tuple[str, str | None]:
    """The route and fallback of explain_backend (_frontend.explanation) on the device."""
    if device == "cpu":
        library, reason = _handler()
        if library is None:
            return _jax_native.ROUTE, reason
    else:
        from .triton import launch_jax

        return launch_jax.route(), None
    return " through the FFI target", None


@functools.cache
def cuda_architecture() -> int:
    """The common compute capability of the local CUDA devices, or zero for a mixed set."""
    architectures = {
        str(device.compute_capability).replace(".", "")
        for device in jax.local_devices(backend="gpu")
    }
    return int(architectures.pop()) if len(architectures) == 1 else 0


def platform(x: jax.Array) -> str:
    """The platform of x, using the active default device when tracing hides its placement."""
    if isinstance(x, jax.core.Tracer):
        device = jax.config.jax_default_device
        if device is None:
            return jax.default_backend()
        return device if isinstance(device, str) else device.platform
    return next(iter(x.devices())).platform


def kernels(x: jax.Array, n: int, classes: int, kind: int, k: int) -> str | None:
    """None where a row kernel computes (n, classes) scores like x, else why not."""
    return _f.kernel_reason(
        x.dtype,
        platform(x),
        lambda: cpu_kernels(kind),
        lambda: _f.triton_reason("launch_jax", kind, k, n, classes, x.dtype),
    )


def value_dtype(dtype: Any, target: Any = None) -> Any:
    return _f.value_dtype(jnp, dtype, target)


def labels(target: jax.Array) -> jax.Array:
    """Class labels in an integer dtype of the FFI target."""
    if target.dtype in (jnp.int32, jnp.uint32, jnp.int64):
        return target
    return target.astype(jnp.int32)


def _vary_like_scores(output: Any, scores: jax.Array) -> Any:
    """Inside shard_map, FFI row outputs vary along the score batch's manual axes."""
    if not isinstance(scores, jax.core.Tracer):
        return output
    pcast = getattr(jax.lax, "pcast", None)
    typeof = getattr(jax, "typeof", None)
    if pcast is None or typeof is None:
        return output
    axis_type = getattr(typeof(scores), "manual_axis_type", None)
    if axis_type is None or not axis_type.varying:
        return output
    axes = axis_type.varying

    def varying(value: jax.Array) -> jax.Array:
        missing = axes - typeof(value).manual_axis_type.varying
        return pcast(value, tuple(missing), to="varying") if missing else value

    return jax.tree.map(varying, output)


def _call(
    scores: jax.Array,
    target: jax.Array,
    weight: jax.Array,
    eps: jax.Array,
    upstream: jax.Array,
    rows: Rows,
    gradient: bool,
    stats: jax.Array | None = None,
    save: bool = False,
    both: bool = False,
    reduction: str = "none",
    total: jax.Array | None = None,
) -> Any:
    """The FFI call: per-row values (value dtype), or the gradient (scores' dtype) times the
    per-row upstream factors, in the scores' buffer where XLA can spare it, or (both) the
    values and the gradient from one pass. Rows are every axis but the last. On CUDA a
    forward that saves also returns the row statistics its gradient then reads instead of
    computing them again ((0,) elsewhere)."""
    stats = _no_stats() if stats is None else stats

    def run(inplace: bool) -> Any:
        return _dispatch(rows, gradient, save, both, reduction, inplace)(
            scores, target, weight, eps, upstream, stats, total
        )

    result = _jax_alias.select(scores, run) if gradient or both else run(False)
    return _vary_like_scores(result, scores)


def _second_order(primals: Any, tangents: Any) -> Any:
    raise RuntimeError(_f.FIRST_ORDER)


def _launch_jax() -> Any:
    try:
        from .triton import launch_jax
    except ImportError:
        return None
    return launch_jax if launch_jax.available() is None else None


def _no_stats() -> jax.Array:
    return jnp.zeros((0,), jnp.uint32)


@functools.cache
def _dispatch(
    rows: Rows, gradient: bool, save: bool, both: bool, reduction: str, inplace: bool = True
) -> Any:
    """The call on the platform it is lowered for: under eager grad the scores are tracers
    without a device, and eager platform_dependent follows the default backend."""
    launch = _launch_jax()

    def cpu(
        scores: Any, target: Any, weight: Any, eps: Any, upstream: Any, stats: Any, total: Any
    ) -> Any:
        return _cpu(scores, target, weight, eps, upstream, stats, rows, gradient,
                    save, both, inplace, reduction)  # fmt: skip

    branches = {"cpu": cpu}
    if launch is not None:
        branches["cuda"] = functools.partial(
            _cuda,
            launch,
            rows=rows,
            gradient=gradient,
            save=save,
            both=both,
            reduction=reduction,
            inplace=inplace,
        )
    kernel = jax.custom_jvp(jax.jit(lambda *args: jax.lax.platform_dependent(*args, **branches)))
    kernel.defjvp(_second_order)
    return kernel


def _cuda(
    launch: Any,
    scores: jax.Array,
    target: jax.Array,
    weight: jax.Array,
    eps: jax.Array,
    upstream: jax.Array,
    stats: jax.Array,
    total: jax.Array | None,
    *,
    rows: Rows,
    reduction: str,
    gradient: bool,
    save: bool,
    both: bool,
    inplace: bool,
) -> Any:
    args = (scores, target, weight, eps, upstream.astype(value_dtype(scores.dtype)), stats)
    if both and reduction != "none":
        return launch.call(
            *args, rows=rows, gradient=True, reduction=reduction, both=True, inplace=inplace
        )
    out = launch.call(
        *args, rows=rows, gradient=gradient, save=save, reduction=reduction, inplace=inplace
    )
    if gradient or reduction != "none":
        return out
    if save:
        values, saved = out
        return values.astype(_loss_dtype(scores, target)), saved
    values = out.astype(_loss_dtype(scores, target))
    if both:
        return values, launch.call(*args, rows=rows, gradient=True, save=False, inplace=inplace)
    return values


def _cpu(
    scores: jax.Array,
    target: jax.Array,
    weight: jax.Array,
    eps: jax.Array,
    upstream: jax.Array,
    stats: jax.Array,
    rows: Rows,
    gradient: bool,
    save: bool,
    both: bool = False,
    inplace: bool = True,
    reduction: str = "none",
) -> Any:
    if save:
        values = _cpu(
            scores, target, weight, eps, upstream, stats, rows, gradient, False, reduction=reduction
        )
        launch = _launch_jax()
        struct = (
            launch.saved(scores, target, weight, launch.attributes(rows, gradient))
            if launch is not None
            else _no_stats()
        )
        return values, jnp.zeros(struct.shape, struct.dtype)
    cpu_kernels(rows.kind)
    n = scores.shape[:-1]
    gradient = gradient or both
    if gradient and upstream.ndim == 0 and rows.kind != KINDS["multilabel_hinge"]:
        upstream = jnp.broadcast_to(upstream, n)
    value_shape = () if reduction != "none" else n
    dtype = _loss_dtype(scores, target) if reduction != "none" else upstream.dtype
    shapes = (
        jax.ShapeDtypeStruct(value_shape if both or not gradient else (0,), dtype),
        jax.ShapeDtypeStruct(scores.shape if gradient else (0,), scores.dtype),
    )
    if _handler()[0] is None:
        shape = shapes[1] if gradient and not both else shapes[0]
        return _jax_native.call(
            shape, scores, target, weight, eps, upstream, rows, gradient, both, inplace, reduction
        )
    call = jax.ffi.ffi_call(
        TARGET,
        shapes,
        vmap_method="broadcast_all",
        input_output_aliases={0: 1} if gradient and inplace else None,
    )
    values, grad = call(
        scores,
        target,
        weight,
        eps,
        upstream,
        kind=np.int64(rows.kind),
        k=np.int64(rows.k),
        gamma=np.float64(rows.gamma),
        temperature=np.float64(rows.temperature),
        label_smoothing=np.float64(rows.label_smoothing),
        ignore_index=np.int64(rows.ignore_index),
        ignore=np.int64(rows.ignore),
        precision=np.int64(rows.precision),
        reduction=np.int64(_f.REDUCTIONS.index(reduction)),
    )
    if both:
        return values, grad
    return grad if gradient else values


def _row_dtype(scores: jax.Array, rows: Rows, reduction: str) -> Any:
    if reduction != "none" and rows.precision == PRECISIONS["float64"] and jax.config.x64_enabled:
        return jnp.float64
    return value_dtype(scores.dtype)


def _loss_dtype(scores: jax.Array, target: jax.Array) -> Any:
    """The dtype of the losses: value_dtype of the scores and of class probabilities."""
    floating = jnp.issubdtype(target.dtype, jnp.floating)
    return value_dtype(scores.dtype, target.dtype if floating else None)


def _row_factors(upstream: jax.Array, dtype: Any, reduction: str, total: Any) -> jax.Array:
    """Transpose the reduction, keeping a scalar cotangent of a sum or mean scalar."""
    factor = upstream.astype(dtype)
    if reduction == "mean":
        divisor = jnp.asarray(total).sum().astype(dtype)
        factor = factor / divisor
    return factor


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def loss(
    scores: jax.Array,
    target: jax.Array,
    weight: jax.Array,
    eps: jax.Array,
    total: Any,
    rows: Rows,
    reduction: str,
) -> jax.Array:
    """The losses of class-last scores (rows: every other axis) by the row kernels, reduced
    ("none", "sum" or "mean": divided by total)."""
    value, _, _ = _value(scores, target, weight, eps, total, rows, reduction, False)
    return value


def _value(
    scores: jax.Array,
    target: jax.Array,
    weight: jax.Array,
    eps: jax.Array,
    total: jax.Array,
    rows: Rows,
    reduction: str,
    save: bool,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Keep row values wide until the device kernel has reduced each mapped example."""
    unit = jnp.zeros((0,), _loss_dtype(scores, target))
    out = _call(scores, target, weight, eps, unit, rows, False, save=save, reduction=reduction)
    value, stats = out if save else (out, None)
    return value, value, stats


def _joint_distillation(scores: jax.Array, target: jax.Array, rows: Rows) -> bool:
    if not (
        _jax_results.available
        and scores.dtype == target.dtype == jnp.bfloat16
        and rows.precision != PRECISIONS["float64"]
        and platform(scores) == "gpu"
    ):
        return False
    from .triton import DISTILLATION_REGISTER_CLASSES

    # The register kernel computes the row value and its gradient together.
    # The streaming kernel saves its normalizers for the separate backward.
    return scores.shape[-1] <= DISTILLATION_REGISTER_CLASSES


def _joint_cuda(scores: jax.Array, target: jax.Array, rows: Rows) -> bool:
    return rows.kind == KINDS["distillation"] and _joint_distillation(scores, target, rows)


def _forward(
    scores: Any, target: Any, weight: Any, eps: Any, total: Any, rows: Rows, reduction: str
) -> tuple[jax.Array, tuple[Any, ...]]:
    """A reduced CPU loss or a selected CUDA row kernel can compute the unit gradient
    with its value. Other CUDA forwards save row statistics for backward."""
    s, t, w, e, d = scores.value, target.value, weight.value, eps.value, total.value
    dtype = _loss_dtype(s, t)
    row_dtype = _row_dtype(s, rows, reduction)
    gradient = stats = None
    gpu_joint = _joint_cuda(s, t, rows)
    if reduction != "none" and scores.perturbed and (gpu_joint or platform(s) == "cpu"):
        unit = _row_factors(jnp.ones((), dtype), row_dtype, reduction, d)
        value, gradient = _call(s, t, w, e, unit, rows, False, both=True, reduction=reduction)
        values = value
        if gpu_joint and rows.kind == KINDS["distillation"] and isinstance(s, jax.core.Tracer):
            separate, _, _ = _value(s, t, w, e, d, rows, reduction, False)
            value, gradient = _jax_results.select(value, gradient, separate)
            values = value
    else:
        value, values, stats = _value(s, t, w, e, d, rows, reduction, scores.perturbed)
    residuals = (s, t, w, e, d, values, stats, gradient, scores.perturbed, target.perturbed)
    return value, residuals


def _recompute_gradient(
    rows: Rows,
    scores: jax.Array,
    target: jax.Array,
    weight: jax.Array,
    eps: jax.Array,
    factors: jax.Array,
    values: jax.Array,
    stats: jax.Array | None,
) -> jax.Array:
    # Only a staged call can reuse the logits buffer. A concrete array already
    # selects a separate gradient; a device barrier would copy the entire input.
    z = scores
    if isinstance(scores, jax.core.Tracer):
        z, _ = _jax_alias.barrier((scores, values))
    return _call(z, target, weight, eps, factors, rows, True, stats)


@functools.partial(jax.jit, static_argnums=(0,), inline=True)
def _select_gradient(
    rows: Rows,
    scores: jax.Array,
    target: jax.Array,
    weight: jax.Array,
    eps: jax.Array,
    factors: jax.Array,
    values: jax.Array,
    stats: jax.Array | None,
    saved: jax.Array,
    upstream: jax.Array,
) -> jax.Array:
    """Cache the eager pullback's conditional across calls and inline it under an outer jit."""
    return jax.lax.cond(
        jnp.all(upstream == 1),
        lambda: saved,
        lambda: _recompute_gradient(rows, scores, target, weight, eps, factors, values, stats),
    )


def _share_cpu_gradient(saved: jax.Array) -> jax.Array:
    """Give the caller its own array handle so donation preserves the retained pullback."""
    # The CPU host view keeps the underlying storage alive. device_put can share it
    # and must preserve the original placement, including an uncommitted CPU array.
    with jax.default_device(next(iter(saved.devices()))):
        return jax.device_put(np.asarray(saved), saved.sharding if saved.committed else None)


def _backward(
    rows: Rows, reduction: str, residuals: tuple[Any, ...], upstream: jax.Array
) -> tuple[Any, ...]:
    """The gradient of the saved scores with the upstream factors applied before its one
    rounding: the forward's for a unit cotangent, else computed again (from the forward's row
    statistics on CUDA). Only for the arguments differentiated (symbolic zeros)."""
    scores, target, weight, eps, total, values, stats, saved, score_grad, target_grad = residuals
    factors = _row_factors(upstream, _row_dtype(scores, rows, reduction), reduction, total)
    target_gradient = None
    if target_grad:
        target_gradient = _target_gradient(rows, scores, target, weight, eps, factors)
    if not score_grad:
        return None, target_gradient, None, None, None

    args = (rows, scores, target, weight, eps, factors, values, stats)
    if saved is None:
        gradient = _recompute_gradient(*args)
    elif (
        not isinstance(upstream, jax.core.Tracer)
        and not isinstance(saved, jax.core.Tracer)
        and upstream.ndim == 0
        and upstream.is_fully_addressable
        and len(saved.devices()) == 1
        and platform(saved) == platform(upstream) == "cpu"
    ):
        # A retained CPU pullback can return its existing unit gradient directly.
        # Passing the same buffer through an XLA conditional copies it.
        gradient = (
            _share_cpu_gradient(saved) if np.asarray(upstream) == 1 else _recompute_gradient(*args)
        )
    elif (
        rows.kind == KINDS["distillation"]
        and isinstance(upstream, jax.core.Tracer)
        and _joint_cuda(scores, target, rows)
    ):
        gradient = saved if _jax_results.unit_literal(upstream) else _recompute_gradient(*args)
    else:
        gradient = _select_gradient(*args, saved, upstream)
    return gradient, target_gradient, None, None, None


def _teacher_formula(
    z: jax.Array, t: jax.Array, factors: jax.Array, temperature: float
) -> jax.Array:
    from ._composed import _targets, xp_jax

    result = _targets.teacher_gradient(xp_jax, z, t, temperature)
    return _targets.cotangent(xp_jax, result, factors, 1.0, t.dtype)


def _cpu_teacher(z: jax.Array, t: jax.Array, factors: jax.Array, temperature: float) -> jax.Array:
    shapes = (jax.ShapeDtypeStruct(z.shape, z.dtype), jax.ShapeDtypeStruct(z.shape[:-1], jnp.uint8))
    call = jax.ffi.ffi_call("libsdca_teacher_vjp", shapes, vmap_method="sequential")
    gradient, exceptional = call(
        z, t, factors.astype(jnp.float32), temperature=np.float64(temperature)
    )
    return jax.lax.cond(
        jnp.any(exceptional),
        lambda: jnp.where(
            exceptional[..., None] != 0, _teacher_formula(z, t, factors, temperature), gradient
        ),
        lambda: gradient,
    )


def _target_gradient(
    rows: Rows,
    scores: jax.Array,
    target: jax.Array,
    weight: jax.Array,
    eps: jax.Array,
    factors: jax.Array,
) -> Any:
    """The cotangent of a differentiated target (_composed._targets): class probabilities of
    cross_entropy, teacher logits of the distillation losses."""
    from ._composed import _targets, xp_jax

    z = scores.astype(jnp.float64) if rows.precision == PRECISIONS["float64"] else scores
    if rows.kind == KINDS["cross_entropy"]:
        w = weight if weight.shape[0] else None
        gradient = _targets.probability_gradient(xp_jax, z, w, eps)
    elif rows.kind == KINDS["distillation"]:
        if (
            z.dtype == target.dtype == jnp.float32
            and rows.precision != PRECISIONS["float64"]
            and 2.0**-149 <= rows.temperature <= np.finfo(np.float32).max
            and platform(scores) == "cpu"
            and _handler()[0] is not None
        ):
            # When tracing hides placement, select the typed CPU custom call only
            # on the platform that will execute the compiled program.
            return jax.lax.platform_dependent(
                z,
                target,
                factors,
                cpu=functools.partial(_cpu_teacher, temperature=rows.temperature),
                default=functools.partial(_teacher_formula, temperature=rows.temperature),
            )
        gradient = _targets.teacher_gradient(xp_jax, z, target.astype(z.dtype), rows.temperature)
    elif rows.kind == KINDS["proximal_distillation"]:
        z = xp_jax.work(z)
        t = target.astype(z.dtype)
        unit = jnp.ones(z.shape[:-1], value_dtype(z.dtype))
        student = _call(z, t, weight, eps, unit, rows, True)
        gradient = _targets.teacher_gradient(xp_jax, z, t, rows.temperature, student, rows.gamma)
    else:
        return None
    return _targets.cotangent(xp_jax, gradient, factors, 1.0, target.dtype)


loss.defvjp(_forward, _backward, symbolic_zeros=True)
_compiled_loss = jax.jit(loss, static_argnums=(5, 6), inline=True)


def _concrete_primal(scores: jax.Array) -> bool:
    """Whether autodiff and batching wrap a concrete array outside a staged computation."""
    value: object = scores
    while isinstance(value, jax.core.Tracer):
        value = getattr(value, "primal", getattr(value, "val", None))
    return isinstance(value, jax.Array)


def evaluate(
    scores: jax.Array,
    target: jax.Array,
    weight: jax.Array | None,
    eps: Any,
    rows: Rows,
    reduction: str = "none",
    total: Any = 1,
) -> jax.Array:
    """The losses of checked class-last arrays (labels as labels()) in the loss dtype: per row
    (scores.shape[:-1]), or their sum or mean (divided by total)."""
    wdtype = jnp.float64 if scores.dtype == jnp.float64 else jnp.float32
    w = jnp.zeros((0,), wdtype) if weight is None else jax.lax.stop_gradient(weight).astype(wdtype)
    e = jax.lax.stop_gradient(jnp.asarray(eps, wdtype))
    # A traced smoothing comes from e. The CPU accepts a static smoothing in float64.
    static = math.nan if isinstance(eps, jax.core.Tracer) else float(eps)
    total = jax.lax.stop_gradient(jnp.asarray(total))
    # Traced CPU calls keep input ownership visible to their caller. A traced short
    # CUDA distillation also exposes a literal unit cotangent to the transpose.
    small_eager = not isinstance(total, jax.core.Tracer) and math.prod(scores.shape[:-1]) <= 64
    cache = reduction != "none" and (
        _concrete_primal(scores)
        if platform(scores) == "cpu"
        else small_eager
        or (
            _joint_cuda(scores, target, rows)
            and (rows.kind != KINDS["distillation"] or not _jax_results.outer_trace(scores))
        )
    )
    kernel = _compiled_loss if cache else loss
    try:
        return kernel(scores, target, w, e, total, rows._replace(label_smoothing=static), reduction)
    except RuntimeError as error:
        # XLA prefixes the CPU handler's synchronous error with its status.
        _f.raise_label_error(error)
        raise
