"""JAX adapters for the portable CPU row kernels shared with PyTorch."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import _jax_cuda, _jax_ffi


@dataclass(frozen=True)
class Distillation:
    temperature: float
    double: bool
    reduction: str
    reference: Callable[..., jax.Array]
    gradient: Callable[..., jax.Array]
    need_gradient: bool = False


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def _pair(s: jax.Array, t: jax.Array, spec: Distillation) -> tuple[jax.Array, jax.Array]:
    shape = s.shape[:-1] if spec.reduction == "none" else ()
    result = (
        jax.ShapeDtypeStruct(shape, s.dtype),
        jax.ShapeDtypeStruct(s.shape if spec.need_gradient else (0,), s.dtype),
    )
    call = jax.ffi.ffi_call(_jax_ffi.DISTILLATION, result, vmap_method="sequential")
    values, gradient = call(
        s,
        t,
        temperature=np.float64(spec.temperature),
        reduction=np.int32({"none": 0, "sum": 1, "mean": 2}[spec.reduction]),
    )
    return values, gradient


def _reference_pair(s: jax.Array, t: jax.Array, spec: Distillation) -> tuple[jax.Array, jax.Array]:
    values = spec.reference(s, t, spec.temperature, spec.double, spec.reduction)
    gradient = jnp.zeros((0,), s.dtype)
    if spec.need_gradient:
        gradient = (
            spec.gradient(
                s, t, jnp.ones(s.shape[:-1], s.dtype), spec.temperature, spec.double, "none"
            )
            * spec.temperature
        )
    return values, gradient


@_pair.defjvp
def _pair_jvp(spec: Distillation, primals: Sequence[Any], tangents: Sequence[Any]) -> Any:
    reference = functools.partial(_reference_pair, spec=spec)
    return _pair(*primals, spec), jax.jvp(reference, tuple(primals), tuple(tangents))[1]


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _loss(s: jax.Array, t: jax.Array, spec: Distillation) -> jax.Array:
    return _pair(s, t, spec)[0]


def _forward(s: jax.Array, t: jax.Array, spec: Distillation) -> Any:
    value, gradient = _pair(s, t, replace(spec, need_gradient=True))
    return value, gradient


def _backward(spec: Distillation, gradient: jax.Array, ct: jax.Array) -> Any:
    count = max(math.prod(gradient.shape[:-1]), 1) if spec.reduction == "mean" else 1
    up = ct[..., None] if spec.reduction == "none" else ct
    info = jnp.finfo(gradient.dtype)
    scale64 = (1.0 / spec.temperature) / count
    if float(info.tiny) <= scale64 <= float(info.max):
        scale = jnp.asarray(scale64, dtype=gradient.dtype)
        factor = up * scale
        return jnp.where(jnp.isfinite(factor), gradient * factor, (gradient * scale) * up), None
    inv_t = jnp.asarray(1.0 / spec.temperature, dtype=gradient.dtype)
    rows = up / count
    factor = rows * inv_t
    return (
        jnp.where(
            jnp.isfinite(factor) & (factor != 0), gradient * factor, (gradient * inv_t) * rows
        ),
        None,
    )


_loss.defvjp(_forward, _backward)


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def _stats_pair(s: jax.Array, t: jax.Array, spec: Distillation) -> tuple[jax.Array, jax.Array]:
    shape = s.shape[:-1] if spec.reduction == "none" else ()
    result = (
        jax.ShapeDtypeStruct(shape, jnp.float32),
        jax.ShapeDtypeStruct((*s.shape[:-1], 14) if spec.need_gradient else (0,), jnp.uint32),
    )
    call = jax.ffi.ffi_call(_jax_ffi.DISTILLATION_STATS, result, vmap_method="sequential")
    value, stats = call(s, t, **_stats_options(spec))
    return value, stats


def _stats_options(spec: Distillation) -> dict[str, Any]:
    return dict(
        temperature=np.float64(spec.temperature),
        work32=np.int32(not spec.double),
        reduction=np.int32({"none": 0, "sum": 1, "mean": 2}[spec.reduction]),
    )


@_stats_pair.defjvp
def _stats_pair_jvp(spec: Distillation, primals: Sequence[Any], tangents: Sequence[Any]) -> Any:
    value, stats = _stats_pair(*primals, spec)

    def reference(s: jax.Array, t: jax.Array) -> jax.Array:
        return spec.reference(s, t, spec.temperature, spec.double, spec.reduction)

    tangent = jax.jvp(reference, tuple(primals), tuple(tangents))[1]
    return (value, stats), (tangent, np.zeros(stats.shape, dtype=jax.dtypes.float0))


@functools.partial(jax.custom_jvp, nondiff_argnums=(4,))
def _stats_gradient(
    s: jax.Array, t: jax.Array, stats: jax.Array, ct: jax.Array, spec: Distillation
) -> jax.Array:
    call = jax.ffi.ffi_call(
        _jax_ffi.DISTILLATION_BACKWARD,
        jax.ShapeDtypeStruct(s.shape, s.dtype),
        vmap_method="sequential",
    )
    return call(s, t, stats, ct.astype(jnp.float32), **_stats_options(spec))


@_stats_gradient.defjvp
def _stats_gradient_jvp(spec: Distillation, primals: Sequence[Any], tangents: Sequence[Any]) -> Any:
    s, t, _, ct = primals

    def reference(a: jax.Array, b: jax.Array, u: jax.Array) -> jax.Array:
        return spec.gradient(a, b, u, spec.temperature, spec.double, spec.reduction).astype(a.dtype)

    tangent = jax.jvp(reference, (s, t, ct), (tangents[0], tangents[1], tangents[3]))[1]
    return _stats_gradient(*primals, spec), tangent


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _stats_loss(s: jax.Array, t: jax.Array, spec: Distillation) -> jax.Array:
    return _stats_pair(s, t, spec)[0]


def _stats_forward(s: jax.Array, t: jax.Array, spec: Distillation) -> Any:
    value, stats = _stats_pair(s, t, replace(spec, need_gradient=True))
    return value, (s, t, stats)


def _stats_backward(spec: Distillation, saved: Sequence[jax.Array], ct: jax.Array) -> Any:
    return _stats_gradient(*saved, ct, spec), None


_stats_loss.defvjp(_stats_forward, _stats_backward)


def distillation(
    student: jax.Array,
    teacher: jax.Array,
    temperature: float,
    double: bool,
    reduction: str,
    reference: Callable[..., jax.Array],
    gradient: Callable[..., jax.Array],
) -> jax.Array | None:
    """Shared CPU distillation, saving statistics for 16-bit or explicit-double work."""
    if _jax_cuda.on_gpu(student) or student.dtype not in (
        jnp.float16,
        jnp.bfloat16,
        jnp.float32,
        jnp.float64,
    ):
        return None
    tiny = float(jnp.finfo(jnp.float64 if student.dtype == jnp.float64 else jnp.float32).tiny)
    if not tiny <= temperature <= 1 / tiny or reduction not in ("none", "sum", "mean"):
        return None
    if not _jax_ffi.enabled():
        return None
    spec = Distillation(temperature, double, reduction, reference, gradient)
    if student.dtype in (jnp.float16, jnp.bfloat16) or (student.dtype == jnp.float32 and double):
        return _stats_loss(student, teacher, spec)
    return _loss(student, teacher, spec)


@dataclass(frozen=True)
class ProximalDistillation:
    temperature: float
    alpha: float
    double: bool
    reference: Callable[..., tuple[jax.Array, jax.Array]]


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def _proximal_pair(s: jax.Array, t: jax.Array, spec: ProximalDistillation) -> Any:
    dtype = jnp.result_type(s.dtype, jnp.float32)
    result = (
        jax.ShapeDtypeStruct(s.shape[:-1], dtype),
        jax.ShapeDtypeStruct(s.shape, dtype),
    )
    call = jax.ffi.ffi_call(_jax_ffi.PROXIMAL_DISTILLATION, result, vmap_method="broadcast_all")
    return call(s, t, temperature=np.float64(spec.temperature), alpha=np.float64(spec.alpha))


@_proximal_pair.defjvp
def _proximal_pair_jvp(
    spec: ProximalDistillation, primals: Sequence[Any], tangents: Sequence[Any]
) -> Any:
    def reference(s: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
        return spec.reference(s, t, spec.temperature, spec.alpha, spec.double)

    return _proximal_pair(*primals, spec), jax.jvp(reference, tuple(primals), tuple(tangents))[1]


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _proximal_rows(s: jax.Array, t: jax.Array, spec: ProximalDistillation) -> jax.Array:
    return _proximal_pair(s, t, spec)[0]


def _proximal_forward(s: jax.Array, t: jax.Array, spec: ProximalDistillation) -> Any:
    values, gradient = _proximal_pair(s, t, spec)
    return values, (gradient, jnp.zeros((0,), s.dtype))


def _proximal_backward(spec: ProximalDistillation, saved: Any, ct: jax.Array) -> Any:
    del spec
    gradient, like = saved
    return (gradient * ct[..., None]).astype(like.dtype), None


_proximal_rows.defvjp(_proximal_forward, _proximal_backward)


def proximal_distillation(
    student: jax.Array,
    teacher: jax.Array,
    temperature: float,
    alpha: float,
    double: bool,
    reference: Callable[..., tuple[jax.Array, jax.Array]],
) -> jax.Array | None:
    """Per-row native proximal loss on CPU; this solver currently uses double work."""
    if _jax_cuda.on_gpu(student) or not _jax_ffi.enabled():
        return None
    if double:
        # Preserve double values through reduction and double gradient entries
        # through upstream scaling, as the explicit-double array implementation.
        student, teacher = student.astype(jnp.float64), teacher.astype(jnp.float64)
    return _proximal_rows(
        student, teacher, ProximalDistillation(temperature, alpha, double, reference)
    )
