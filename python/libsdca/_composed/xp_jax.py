"""The composed-loss namespace (_xp.Namespace) over JAX arrays."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import lax

from ._xp import FIRST_ORDER, cpu_labels

State = TypeVar("State")
Carry = tuple[jax.Array, ...]

name = "jax"
float32 = jnp.float32
float64 = jnp.float64
int_ = jnp.int32


def finfo(dtype: Any) -> Any:
    return jnp.finfo(dtype)


def work(x: jax.Array) -> jax.Array:
    return x.astype(jnp.float32) if jnp.dtype(x.dtype).itemsize <= 2 else x


def wide() -> Any:
    return jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


def astype(x: jax.Array, dtype: Any) -> jax.Array:
    if x.dtype == jnp.bfloat16 and dtype == jnp.float64 and jax.config.x64_enabled:
        from ._jax_bfloat import cast

        return cast(x)
    return x.astype(dtype)


def result_type(a: jax.Array, b: jax.Array) -> Any:
    return jnp.result_type(a, b)


def full_like(x: jax.Array, value: float, dtype: Any = None) -> jax.Array:
    return jnp.full_like(x, value, dtype=dtype)


def eager_labels(label: jax.Array, scores: jax.Array) -> jax.Array | None:
    if isinstance(scores, jax.core.Tracer) or isinstance(label, jax.core.Tracer):
        return None
    devices = scores.devices()
    if len(devices) != 1 or next(iter(devices)).platform != "cpu":
        return None
    return jax.device_put(cpu_labels(np.asarray(label), scores.shape[-1]), next(iter(devices)))


def arange(n: int, like: jax.Array) -> jax.Array:
    del like
    return jnp.arange(n, dtype=int_)


where = jnp.where
exp = jnp.exp
expm1 = jnp.expm1
log = jnp.log
log1p = jnp.log1p
abs = jnp.abs  # noqa: A001
isfinite = jnp.isfinite
isnan = jnp.isnan
maximum = jnp.maximum
minimum = jnp.minimum
nextafter = jnp.nextafter


def bits(x: jax.Array) -> jax.Array:
    return lax.bitcast_convert_type(x, jnp.int32 if x.dtype == jnp.float32 else jnp.int64)


def from_bits(bits: jax.Array, like: jax.Array) -> jax.Array:
    return lax.bitcast_convert_type(bits, like.dtype)


def clip(x: jax.Array, lo: Any, hi: Any) -> jax.Array:
    return jnp.minimum(jnp.maximum(x, lo), hi)


def sum(x: jax.Array) -> jax.Array:  # noqa: A001
    return x.sum(-1, keepdims=True)


def max(x: jax.Array) -> jax.Array:  # noqa: A001
    return x.max(-1, keepdims=True)


def min(x: jax.Array) -> jax.Array:  # noqa: A001
    return x.min(-1, keepdims=True)


def argmax(x: jax.Array) -> jax.Array:
    return jnp.argmax(x, -1, keepdims=True).astype(int_)


def argmin(x: jax.Array) -> jax.Array:
    return jnp.argmin(x, -1, keepdims=True).astype(int_)


def count(mask: jax.Array) -> jax.Array:
    return mask.sum(-1, keepdims=True, dtype=int_)


def all(mask: jax.Array) -> jax.Array:  # noqa: A001
    return mask.all(-1, keepdims=True)


@jax.custom_jvp
def logsumexp(x: jax.Array) -> jax.Array:
    top = lax.stop_gradient(x.max(-1, keepdims=True))
    top = jnp.where(jnp.isfinite(top), top, 0)
    return top + jnp.log(jnp.exp(x - top).sum(-1, keepdims=True))


@logsumexp.defjvp
def _logsumexp_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    (x,), (dx,) = primals, tangents
    y = logsumexp(x)
    return y, (jnp.exp(x - y) * dx).sum(-1, keepdims=True)


def softmax(x: jax.Array) -> jax.Array:
    # As the derivative of logsumexp computes it.
    return jnp.exp(x - logsumexp(x))


def log_softmax(x: jax.Array) -> jax.Array:
    return jax.nn.log_softmax(x, axis=-1)


def sort(x: jax.Array) -> jax.Array:
    return -jnp.sort(-x, axis=-1)


# Dispatch the compensated reduction as one operation in an eager JAX call.
@jax.jit
def mean_margin(best: jax.Array, label: jax.Array) -> tuple[jax.Array, jax.Array]:
    from . import _losses, xp_jax

    return _losses.mean_margin(xp_jax, best, label)


def topk(x: jax.Array, k: int, *, stable: bool = True) -> tuple[jax.Array, jax.Array]:
    del stable  # lax.top_k already prefers the lower index.
    if k == 1:
        index = argmax(x)
        return take(x, index), index
    values, index = lax.top_k(x, k)
    return values, index.astype(int_)


def take(x: jax.Array, index: jax.Array) -> jax.Array:
    rows = jnp.broadcast_shapes(x.shape[:-1], index.shape[:-1])
    x = jnp.broadcast_to(x, (*rows, x.shape[-1]))
    index = jnp.broadcast_to(index, (*rows, index.shape[-1]))
    if len(rows) == 1 and index.shape[-1] == 1:
        # Each (row, column) is distinct. This lets the CUDA transpose write
        # the cotangent directly and preserves subnormals lost by atomic add.
        return x.at[jnp.arange(rows[0])[:, None], index].get(mode="fill", unique_indices=True)
    return jnp.take_along_axis(x, index, axis=-1)


def put(x: jax.Array, index: jax.Array, values: Any) -> jax.Array:
    rows = jnp.arange(x.shape[0])[:, None]
    return x.at[rows, index].set(values)


def concat(xs: Sequence[jax.Array]) -> jax.Array:
    return jnp.concatenate(list(xs), -1)


def pad(x: jax.Array, size: int) -> jax.Array:
    return jnp.pad(x, ((0, 0), (0, size - x.shape[-1])))


stop = lax.stop_gradient


@jax.custom_jvp
def _guard(gradient: jax.Array, z: jax.Array) -> jax.Array:
    """gradient, raising where it is differentiated (in z)."""
    return gradient


@_guard.defjvp
def _guard_jvp(primals: tuple[jax.Array, ...], tangents: tuple[jax.Array, ...]) -> Any:
    raise RuntimeError(FIRST_ORDER)


@jax.custom_jvp
def attach(value: jax.Array, z: jax.Array, gradient: jax.Array) -> jax.Array:
    return value


@attach.defjvp
def _attach_jvp(
    primals: tuple[jax.Array, ...], tangents: tuple[jax.Array, ...]
) -> tuple[jax.Array, jax.Array]:
    value, z, gradient = primals
    g = _guard(gradient, z)
    return value, jnp.where(g == 0, 0.0, g * tangents[1]).sum(-1)


@jax.custom_jvp
def attach_smooth(value: jax.Array, z: jax.Array, gradient: jax.Array) -> jax.Array:
    return value


@attach_smooth.defjvp
def _attach_smooth_jvp(
    primals: tuple[jax.Array, ...], tangents: tuple[jax.Array, ...]
) -> tuple[jax.Array, jax.Array]:
    value, _, gradient = primals
    # A zero slope can have a nonzero derivative. Preserve that dependence while
    # transposing exact zero entries to positive zero for either upstream sign.
    direction = tangents[1]
    slope = jax.lax.stop_gradient(gradient + 0.0)
    fixed = jnp.where(slope == 0, 0.0, slope * direction)
    residual = jnp.where(jnp.isfinite(slope), gradient - slope, 0.0)
    return value, (fixed + residual * direction).sum(-1)


@functools.partial(jax.custom_jvp, nondiff_argnums=(3,))
def attach_teacher(
    value: jax.Array, student: jax.Array, teacher: jax.Array, temperature: float
) -> jax.Array:
    """Attach the teacher differential of regular distillation, including its mixed derivatives."""
    return value


@functools.partial(jax.jit, static_argnums=(3,), inline=True)
def _teacher_direction(
    student: jax.Array, teacher: jax.Array, direction: jax.Array, temperature: float
) -> jax.Array:
    from . import xp_jax
    from ._targets import teacher_gradient

    gradient = teacher_gradient(xp_jax, student, teacher, temperature)
    return (gradient * direction).sum(-1)


@functools.partial(attach_teacher.defjvp, symbolic_zeros=True)
def _attach_teacher_jvp(
    temperature: float, primals: tuple[jax.Array, ...], tangents: tuple[Any, ...]
) -> tuple[jax.Array, Any]:
    value, student, teacher = primals
    dvalue, _, dteacher = tangents
    if isinstance(dteacher, jax.custom_derivatives.SymbolicZero):
        return value, dvalue
    contribution = _teacher_direction(student, teacher, dteacher, temperature)
    if isinstance(dvalue, jax.custom_derivatives.SymbolicZero):
        return value, contribution
    return value, dvalue + contribution


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def target_gradient(
    value: jax.Array, target: jax.Array, gradient: Any, scale: float, *operands: Any
) -> jax.Array:
    return value


def _target_gradient_forward(
    value: Any, target: Any, gradient: Any, scale: float, *operands: Any
) -> tuple[jax.Array, Any]:
    plain = jax.tree_util.tree_map(lambda primal: primal.value, operands)
    return value.value, (target.value, target.perturbed, plain)


def _target_gradient_backward(
    gradient: Any, scale: float, residuals: Any, upstream: jax.Array
) -> Any:
    from . import xp_jax
    from ._targets import cotangent

    target, perturbed, operands = residuals
    unused = (None,) * len(operands)
    if not perturbed:
        return (upstream, None, *unused)
    g = gradient(xp_jax, *operands)
    return (upstream, cotangent(xp_jax, g, upstream, scale, target.dtype), *unused)


target_gradient.defvjp(_target_gradient_forward, _target_gradient_backward, symbolic_zeros=True)


def grad(f: Callable[[jax.Array], jax.Array], x: jax.Array) -> jax.Array:
    return jax.grad(lambda x: f(x).sum())(lax.stop_gradient(x))


def fori(n: int, body: Callable[[int, State], State], state: State) -> State:
    return lax.fori_loop(0, n, body, state)


def while_loop(
    more: Callable[[Carry], jax.Array],
    body: Callable[[Carry], Carry],
    state: Carry,
    steps: int,
) -> Carry:
    def going(carry: tuple[jax.Array, Carry]) -> jax.Array:
        return (carry[0] < steps) & more(carry[1])

    def step(carry: tuple[jax.Array, Carry]) -> tuple[jax.Array, Carry]:
        return carry[0] + 1, body(carry[1])

    return lax.while_loop(going, step, (jnp.zeros((), int_), state))[1]
