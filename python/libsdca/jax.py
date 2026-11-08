"""JAX losses: sdca_loss (the LOSSES at the input scores), distillation_loss
(cross-entropy to a teacher's softmax) and proximal_distillation_loss, under jit, grad
and vmap. The native backend calls the C ABI through an XLA FFI call on the
CPU (_jax_ffi) and through jax.pure_callback elsewhere or where the FFI handlers
do not build; the jax backend computes every loss with jax.numpy on any device
(entropy to any order, the others first-order reverse mode).
"""

from __future__ import annotations

import functools
import math
import os
import re
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.typing import ArrayLike
from numpy.typing import DTypeLike

from . import _jax_ffi, _jax_losses
from ._core import (
    _IDS,
    _PRECISION_CODES,
    TOPK,
    _canonical_precision,
    _check_options,
    _loss_of_kind,
    _tensor_kind,
    _topk_hinge_name,
    loss_and_grad,
)

__all__ = [
    "cross_entropy",
    "distillation_loss",
    "multiclass_hinge_loss",
    "multilabel_cross_entropy",
    "multilabel_hinge_loss",
    "proximal_cross_entropy",
    "proximal_distillation_loss",
    "sdca_loss",
    "topk_cross_entropy",
    "topk_hinge_alpha_loss",
    "topk_hinge_beta_loss",
    "topk_hinge_loss",
]


def _reduce(
    loss: jax.Array, reduction: str, dtype: jax.typing.DTypeLike | None = None
) -> jax.Array:
    """The reduction of per-example losses, summed in their own (working) dtype and rounded
    once to dtype."""
    if reduction == "none":
        reduced = loss
    elif reduction == "sum":
        reduced = loss.sum()
    elif reduction == "mean":
        reduced = loss.mean()
    else:
        raise ValueError("reduction must be none, sum, or mean")
    return reduced if dtype is None else reduced.astype(dtype)


def _empty_loss(z: jax.Array, reduction: str) -> jax.Array:
    """The loss of no examples, as libsdca.torch and torch.nn.functional: empty per-example
    losses, a zero sum and a NaN mean, with a zero gradient for the (class-last) logits z."""
    return _reduce(z.sum(-1).astype(_work(z).dtype), reduction)


def _work(z: jax.Array, double: bool = False) -> jax.Array:
    if double:
        return z.astype(jnp.float64)
    return z.astype(jnp.float32) if z.dtype in (jnp.float16, jnp.bfloat16) else z


def _check_precision(x: jax.Array, canonical: str | None, tensor: bool) -> None:
    """As libsdca.torch; "float64" in jax arithmetic (tensor) needs jax_enable_x64."""
    if x.dtype == jnp.float64 and canonical == "float32":
        raise ValueError("float64 logits require precision='float64' or None")
    if tensor and canonical == "float64" and x.dtype != jnp.float64 and not _x64():
        raise ValueError("precision='float64' needs jax_enable_x64")


def _x64() -> bool:
    return bool(jax.dtypes.canonicalize_dtype(jnp.float64) == jnp.float64)


def _concrete(*arrays: ArrayLike) -> bool:
    return not any(isinstance(a, jax.core.Tracer) for a in arrays)


def _float32_holds(*scales: float) -> bool:
    tiny = float(np.finfo(np.float32).tiny)
    return all(tiny <= s <= 1 / tiny for s in scales)


def _accelerator(x: jax.Array) -> bool:
    """Whether x lives (traced: the default backend computes) off the CPU."""
    if _concrete(x):
        return any(d.platform != "cpu" for d in x.devices())
    return jax.default_backend() != "cpu"


# Calls of at most this many scores take the jax backend on the CPU: a native call costs about
# 0.03 ms as an FFI call (0.35 to 0.65 ms as a host callback), more than these losses in jax.
_SMALL = 1 << 14
# jaxlib 0.6's XLA:CPU miscompiled loss_and_gradient called with a constant boolean mask
# whose class axis was moved (a NumPy mask under jit, axis=0): the smooth hinges gave wrong
# losses. Before jaxlib 0.12 a fence keeps the mask an operand; 0.12 computes them right
# without it, which saves up to a third of the time of the nonsmooth multilabel hinge.
_FENCE_MASKS = tuple(int(part) for part in jax.__version__.split(".")[:2]) < (0, 12)


def _cpu_jax(objective: str, k: int, size: int, classes: int = 1024) -> bool:
    """Whether "auto" computes the loss in jax.numpy on the CPU (where its arithmetic is
    available): calls of at most _SMALL scores but proximal_cross_entropy, and without the FFI
    handlers (or on one thread for narrow rows) also the nonsmooth multiclass and multilabel
    hinges. With the FFI handlers, the larger calls run in the native library."""
    if objective == "proximal_cross_entropy":
        return False
    if size <= _SMALL:
        return True
    nonsmooth = objective == "multilabel_hinge" or (
        objective in ("topk_hinge_alpha", "topk_hinge_beta") and k == 1
    )
    if _jax_ffi.enabled():
        return nonsmooth and classes <= 32 and _single_thread_cpu()
    return nonsmooth


def _jax_backend(
    x: jax.Array,
    objective: str,
    backend: str,
    precision: str | None,
    gamma: float,
    alpha: float,
    k: int,
    classes: int = 1024,
) -> bool:
    """Whether jax.numpy computes the loss: "auto" takes it for entropy, and for the other
    losses where their arithmetic is available (float64 unless precision="float32"): off the
    CPU, and on it where _cpu_jax holds."""
    if backend == "native":
        return False
    if objective == "cross_entropy":
        return True
    double = precision != "float32" or not _float32_holds(gamma, alpha)
    if backend == "jax" and double and not _x64():
        raise ValueError("backend='jax' solves this loss in float64; enable jax_enable_x64")
    if backend == "jax":
        return True
    available = _x64() or not double
    return available and (_accelerator(x) or _cpu_jax(objective, k, int(np.size(x)), classes))


def _host(
    z: np.ndarray,
    positive: np.ndarray,
    *,
    loss: str,
    k: int,
    gamma: float,
    alpha: float,
    precision: str | None,
    values_dtype: DTypeLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Native losses (in values_dtype, declared at trace time: XLA's callback thread may see
    another x64 setting) and gradients of any leading shape (vmap broadcasts them), of integer
    labels (the leading shape) or a label mask."""
    z, positive = np.asarray(z), np.asarray(positive)
    classes = z.shape[-1]
    name, nonsmooth = _loss_of_kind(loss)
    if positive.ndim < z.ndim:
        # A scatter into zeros, not a comparison of every class; out of range, a label marks
        # no class (the native call rejects the row).
        labels = positive.reshape(-1).astype(np.int64)
        target = np.zeros((labels.size, classes), np.bool_)
        rows = np.flatnonzero((labels >= 0) & (labels < classes))
        target[rows, labels[rows]] = True
    else:
        target = positive.reshape(-1, classes)
    values, gradients = loss_and_grad(
        z.reshape(-1, classes),
        target,
        name,
        k=k,
        gamma=gamma if nonsmooth is None else nonsmooth,
        alpha=alpha,
        # The available CPUs, as XLA's own CPU operations, unless SDCA_NUM_THREADS is set.
        num_threads=None if "SDCA_NUM_THREADS" in os.environ else 0,
        precision=precision,
    )
    return values.astype(values_dtype).reshape(z.shape[:-1]), gradients.reshape(z.shape)


def _values_dtype(dtype: DTypeLike) -> DTypeLike:
    """float64 values (as computed) where JAX has float64: the reduction rounds them once."""
    return np.float64 if _x64() else dtype


# The static options of a loss: (on the host, loss, k, gamma, alpha, precision, label width).
_Options = tuple[bool, str, int, float, float, str | None, int]


def _host_of(
    options: _Options, values_dtype: DTypeLike
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """_host of the static options, returning values_dtype values."""
    _, loss, k, gamma, alpha, precision, _ = options
    return functools.partial(
        _host,
        loss=loss,
        k=k,
        gamma=gamma,
        alpha=alpha,
        precision=precision,
        values_dtype=values_dtype,
    )


def _ffi(z: jax.Array) -> bool:
    """Whether the native loss of z is an XLA FFI call (on the CPU, where the handlers load:
    _jax_ffi), else a host callback."""
    return not _accelerator(z) and _jax_ffi.enabled()


def _attributes(options: _Options, dtype: DTypeLike) -> dict[str, np.generic]:
    """The options of the FFI handlers, as _host passes them to loss_and_grad."""
    _, loss, k, gamma, alpha, precision, _ = options
    name, nonsmooth = _loss_of_kind(loss)
    work = "float64" if precision is None else precision
    storage = np.float32 if np.dtype(dtype) == np.float32 else np.float64
    return {
        "loss": np.int32(_IDS[name]),
        "k": np.int64(k),
        "gamma": np.float64(gamma if nonsmooth is None else nonsmooth),
        "alpha": np.float64(alpha),
        "precision": np.int32(_PRECISION_CODES[(work, storage)]),
    }


def _native_call(
    z: jax.Array, positive: jax.Array, options: _Options
) -> tuple[jax.Array, jax.Array]:
    values = jax.ShapeDtypeStruct(z.shape[:-1], _values_dtype(z.dtype))
    shapes = (values, jax.ShapeDtypeStruct(z.shape, z.dtype))
    if _ffi(z):
        call = jax.ffi.ffi_call(_jax_ffi.GRADIENT, shapes, vmap_method="broadcast_all")
        values, gradients = call(z, positive, **_attributes(options, z.dtype))
        return values, gradients
    host = _host_of(options, values.dtype)
    values, gradients = jax.pure_callback(host, shapes, z, positive, vmap_method="broadcast_all")
    return values, gradients


def _native_values(z: jax.Array, positive: jax.Array, options: _Options) -> jax.Array:
    """The native losses alone: the handler (or host) returns no gradient."""
    shape = jax.ShapeDtypeStruct(z.shape[:-1], _values_dtype(z.dtype))
    if _ffi(z):
        call = jax.ffi.ffi_call(_jax_ffi.VALUES, shape, vmap_method="broadcast_all")
        return call(z, positive, **_attributes(options, z.dtype))
    host = _host_of(options, shape.dtype)

    def values(z: np.ndarray, positive: np.ndarray) -> np.ndarray:
        return host(z, positive)[0]

    return jax.pure_callback(values, shape, z, positive, vmap_method="broadcast_all")


def _tensor_call(
    z: jax.Array, positive: jax.Array, options: _Options, values_only: bool = False
) -> tuple[jax.Array, jax.Array]:
    """jax.numpy losses as libsdca.torch's tensor backend: float64, or float32 where
    precision="float32" and float32 holds gamma, alpha and the spread of every row;
    values_only: computed for the losses alone (the gradient may be skipped)."""
    _, loss, k, gamma, alpha, precision, width = options
    flat = z.reshape(-1, z.shape[-1])
    mask = positive.reshape(-1) if positive.ndim < z.ndim else positive.reshape(flat.shape)
    if _FENCE_MASKS and positive.ndim == z.ndim:
        mask = lax.optimization_barrier(mask)

    def solve(work: DTypeLike) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        # Rounded once from double, as the scalars of the tensor backend.
        scalars: np.ndarray = np.array([gamma, gamma / k, alpha, math.log(alpha)], work)

        def call(flat: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
            values, gradients = _jax_losses.loss_and_gradient(
                flat, mask, scalars, objective=loss, k=k, width=width, values_only=values_only
            )
            # Losses as computed, in float64 where JAX has it: the reduction rounds them once.
            return values.astype(_values_dtype(values.dtype)), gradients

        return call

    if precision != "float32" or not _float32_holds(gamma, alpha):
        values, gradients = solve(jnp.float64)(flat, mask)
    elif _x64():
        spread = flat.max(-1) - flat.min(-1)
        values, gradients = lax.cond(
            jnp.isfinite(spread).all(), solve(jnp.float32), solve(jnp.float64), flat, mask
        )
    else:
        values, gradients = solve(jnp.float32)(flat, mask)
    return values.reshape(z.shape[:-1]), gradients.reshape(z.shape)


@jax.custom_jvp
def _first_order(gradients: jax.Array) -> jax.Array:
    """The saved gradient (or the scores of a native call), which forward mode (jvp, hessian)
    must not differentiate."""
    return gradients


@_first_order.defjvp
def _first_order_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    raise TypeError("sdca_loss has no JVP: its derivatives are first-order reverse mode")


def _call(z: jax.Array, positive: jax.Array, options: _Options) -> tuple[jax.Array, jax.Array]:
    if options[0]:
        # Forward mode over this rule (hessian) raises here, not in the FFI call or callback.
        return _native_call(_first_order(z), positive, options)
    values, gradients = _tensor_call(z, positive, options)
    return values, _first_order(gradients)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _loss(z: jax.Array, positive: jax.Array, options: _Options) -> jax.Array:
    if options[0]:
        return _native_values(z, positive, options)
    return _tensor_call(z, positive, options, values_only=True)[0]


def _saved_bwd(*args: Any) -> tuple[jax.Array, None]:
    """The saved gradient (the residual after the static options) times the cotangent; the
    second argument (target or teacher) gets none."""
    gradient, ct = args[-2:]
    return (gradient * ct[..., None]).astype(gradient.dtype), None


_loss.defvjp(_call, _saved_bwd)


def _rounded_bwd(*args: Any) -> tuple[jax.Array, None]:
    """As _saved_bwd for a working-precision gradient saved with an empty array of the logits'
    dtype: the product with the cotangent is rounded once, so loss scaling keeps the tails."""
    (gradient, like), ct = args[-2:]
    return (gradient * ct[..., None]).astype(like.dtype), None


def _pinned(e: jax.Array, at: jax.Array | bool) -> jax.Array:
    """e, but exactly 0 where at holds, keeping the derivatives of e there: a tied maximum
    takes only a share of the row maximum's derivative, which higher derivatives need."""
    return jnp.where(at, e - lax.stop_gradient(e), e)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _fenced(x: jax.Array, step: int) -> jax.Array:
    return lax.optimization_barrier((x, jnp.int32(step)))[0]


@_fenced.defjvp
def _fenced_jvp(
    step: int, primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    return _fenced(primals[0], step), _fenced(tangents[0], step)


def _pass(x: jax.Array, step: int) -> jax.Array:
    """A view of x for pass step on the CPU above _SMALL, so XLA:CPU fuses each pass instead
    of materializing shared intermediates on four threads."""
    return _fenced(x, step) if step and _jax_losses._cpu() and x.size > _SMALL else x


def _rest(
    z: jax.Array,
    maximum: jax.Array,
    finite: bool = False,
    terms: tuple[jax.Array, ...] = (),
    masks: tuple[jax.Array, ...] = (),
    excluded: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """The exponentials of z - maximum and the mass of the classes other than one maximum: in
    one pass, accurate below eps for log1p (a tied maximum adds
    exactly one). finite: rows with a nonfinite score give nan. The row sums of further terms
    and counts of masks share the pass, after the mass. excluded: the index of one maximum,
    if already available, avoiding a separate count of tied maxima."""
    top = (
        z == maximum
        if excluded is None
        else jnp.arange(z.shape[-1], dtype=excluded.dtype) == excluded
    )
    e = jnp.exp(z - maximum)
    # _pinned keeps the derivatives at the excluded maxima.
    rest = _pinned(e, top)
    if finite:
        # z * 0 is nan exactly where a score is not finite (a difference to the maximum may
        # overflow to -inf).
        rest = rest + z * 0
    if excluded is not None:
        return (e, *_jax_losses.row_sums((rest, *terms), masks))
    mass, *sums, ties = _jax_losses.row_sums((rest, *terms), (*masks, top))
    return e, mass + (ties - 1), *sums


def _label_entropy(z: jax.Array, labels: jax.Array, double: bool) -> tuple[jax.Array, jax.Array]:
    """Entropy loss and gradient of integer labels in two passes over the scores, as a
    composed log-softmax: the maximum, and the mass of the classes but one maximum.

    The mass leaves out a maximum, so log1p keeps tiny losses, and the target gradient is minus
    the mass of the other classes, which keeps it accurate on confident rows. Autodiff of these
    formulas gives the higher derivatives.
    """
    classes = jnp.arange(z.shape[-1])
    maximum = _work(z.max(-1, keepdims=True), double)
    e, rest = _rest(_work(_pass(z, 1), double), maximum, finite=True)
    # A label out of range (no wrapping of negative ones) picks nan, which spreads to the row.
    index = labels[..., None].astype(jnp.promote_types(labels.dtype, jnp.int32))
    index = jnp.where((index >= 0) & (index < z.shape[-1]), index, z.shape[-1])
    picked = _work(jnp.take_along_axis(z, index, -1, mode="fill", fill_value=jnp.nan), double)
    total = (1 + rest) + picked * 0
    target = jnp.exp(picked - maximum)
    others = jnp.where(picked == maximum, rest - _pinned(target, True), 1 + (rest - target))
    value = jnp.log1p(rest[..., 0]) + (maximum - picked)[..., 0]
    if _jax_losses._cpu() and z.size > _SMALL:
        e = jnp.exp(_work(_pass(z, 2), double) - maximum)
    # Times the reciprocal of the row's total, as a composed softmax.
    probability = e * (1 / total)
    return value, jnp.where(classes == index, -others / total, probability)


def _mask_entropy(z: jax.Array, positive: jax.Array, double: bool) -> tuple[jax.Array, jax.Array]:
    """Entropy loss and gradient of a label mask (uniform target), as torch._TensorEntropy:
    small logit differences stay exact in the gradient. Three passes over the scores for the
    value (the maximum; the mass with the counts; the label gaps), three more for the
    gradient (a positive anchor, the correction and the gradient)."""
    maximum = _work(z.max(-1, keepdims=True), double)
    if jax.default_backend() == "gpu":
        return _mask_entropy_gpu(z, positive, maximum, double)
    difference = maximum - _work(_pass(z, 2), double)
    # The mass but one maximum (log1p keeps tiny losses; nan for a nonfinite score), the sum
    # of the label gaps and the label count.
    terms = (jnp.where(positive, difference, 0.0),)
    e, rest, gaps, count = _rest(_work(_pass(z, 1), double), maximum, True, terms, (positive,))

    def rescued() -> jax.Array:
        scaled = _jax_losses.row_sums((_scaled_gaps(z, positive, maximum, double, 3),), ())[0]
        return _mean_gap(gaps, scaled, count, z.shape[-1])

    # On the CPU its pass only where a sum overflows (elsewhere the predicate would wait;
    # small calls take the pass rather than the conditional).
    if _jax_losses._cpu() and z.size > _SMALL:
        gaps = lax.cond(jnp.isfinite(gaps).all(), lambda: gaps / count, rescued)
    else:
        gaps = rescued()
    value = gaps[..., 0] + jnp.log1p(rest[..., 0])
    den = 1 + rest
    z_anchor = _mask_anchor(z, positive, maximum, double)
    e_anchor = jnp.exp(z_anchor - maximum)

    def raw(step: int) -> jax.Array:
        zw = _work(_pass(z, step), double)
        ew = jnp.exp(zw - maximum) if _jax_losses._cpu() and z.size > _SMALL else e
        differences = _anchored(zw, ew, z_anchor, e_anchor) / den
        return jnp.where(positive, differences, ew / den)

    correction = _jax_losses.row_sums((raw(4),), ())[0] / count
    entries = raw(5)
    gradient = jnp.where(positive, entries - correction, entries)
    valid = count > 0
    return jnp.where(valid[..., 0], value, jnp.nan), jnp.where(valid, gradient, jnp.nan)


def _mask_entropy_gpu(
    z: jax.Array, positive: jax.Array, maximum: jax.Array, double: bool
) -> tuple[jax.Array, jax.Array]:
    """_mask_entropy on the GPU in three passes over the scores: the maximum with the
    positive anchor; the mass, counts, label gaps (plain and scaled) and the positives'
    mass; the gradient. A fourth pass sums the numerator of the correction only if a row's
    positives hold more than half of its mass."""
    z_anchor = _mask_anchor(z, positive, maximum, double)
    e_anchor = jnp.exp(z_anchor - maximum)
    zw = _work(z, double)
    terms = (
        jnp.where(positive, maximum - zw, 0.0),
        _scaled_gaps(z, positive, maximum, double, 0),
        jnp.where(positive, jnp.exp(zw - maximum), 0.0),
    )
    e, rest, gaps, scaled, held, count = _rest(zw, maximum, True, terms, (positive,))
    value = _mean_gap(gaps, scaled, count, z.shape[-1])[..., 0] + jnp.log1p(rest[..., 0])
    den = 1 + rest
    # The numerator of the correction, the sum of the negatives' mass and the positives'
    # differences to the anchor, is den - count e_anchor. Where the positives hold at most
    # half the mass, that difference is within a small factor of the sum's own rounding.
    # Elsewhere it is summed from the differences, in a pass of its own.
    ordinary = held <= den / 2

    def summed() -> jax.Array:
        # From the scores anew: a conditional's operands are stored.
        zw = _work(z, double)
        e = jnp.exp(zw - maximum)
        differences = jnp.where(positive, _anchored(zw, e, z_anchor, e_anchor), e)
        return _jax_losses.row_sums((differences,), ())[0]

    numerator = jnp.where(ordinary, den - count * e_anchor, _only_if(~ordinary.all(), summed, den))
    correction = numerator / den / count
    entries = jnp.where(positive, _anchored(zw, e, z_anchor, e_anchor), e) / den
    gradient = jnp.where(positive, entries - correction, entries)
    valid = count > 0
    return jnp.where(valid[..., 0], value, jnp.nan), jnp.where(valid, gradient, jnp.nan)


def _only_if(pred: jax.Array, fn: Callable[[], jax.Array], default: jax.Array) -> jax.Array:
    """fn() where pred holds, else default, computing fn only then. XLA flattens a
    conditional of elementwise operations and reductions into a select of both branches;
    the barrier keeps the branch a conditional."""
    return lax.cond(pred, lambda: lax.optimization_barrier(fn()), lambda: default)


def _mask_anchor(z: jax.Array, positive: jax.Array, maximum: jax.Array, double: bool) -> jax.Array:
    """A positive score of every row, which anchors the differences of the positives."""
    if jax.default_backend() == "gpu" and maximum.dtype == jnp.float32:
        # Any positive score anchors the differences; this avoids a GPU index and gather.
        return _work(jnp.where(positive, z, -jnp.inf).max(-1, keepdims=True), double)
    # Keep the first-positive gather for CPU and FP64 work, where the maximum is slower.
    anchor = jnp.argmax(positive, -1, keepdims=True)
    return _work(jnp.take_along_axis(z, anchor, -1), double)


def _anchored(zw: jax.Array, ew: jax.Array, z_anchor: jax.Array, e_anchor: jax.Array) -> jax.Array:
    """ew - e_anchor from one expm1 of minus the distance (no overflow), whose selection takes
    the branch delta >= 0 for the higher derivatives at equal positive logits."""
    delta = zw - z_anchor
    distance = jnp.expm1(jnp.where(delta >= 0, -delta, delta))
    return jnp.where(delta >= 0, -ew * distance, e_anchor * distance)


def _scaled_gaps(
    z: jax.Array, positive: jax.Array, maximum: jax.Array, double: bool, step: int
) -> jax.Array:
    """The label gaps scaled by a power of two (exact), whose sum is finite where the plain
    gaps or their sum overflow."""
    scale = 2.0 ** (z.shape[-1].bit_length() + 1)
    zw = _work(_pass(z, step), double)
    return jnp.where(positive, maximum * (1 / scale) - zw * (1 / scale), 0.0)


def _mean_gap(gaps: jax.Array, scaled: jax.Array, count: jax.Array, classes: int) -> jax.Array:
    """The mean label gap: a representable mean survives an overflowing difference or sum
    from the scaled gaps (taken only where the plain sum is not finite, so no tiny term then
    matters)."""
    scale = 2.0 ** (classes.bit_length() + 1)
    return jnp.where(jnp.isfinite(gaps), gaps / count, scaled / count * scale)


def _entropy_parts(z: jax.Array, target: jax.Array, double: bool) -> tuple[jax.Array, jax.Array]:
    if target.dtype == jnp.bool_:
        return _mask_entropy(z, target, double)
    return _label_entropy(z, target, double)


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def _entropy(z: jax.Array, target: jax.Array, double: bool) -> jax.Array:
    """Entropy loss of floating-point scores and integer labels or a label mask."""
    return _entropy_parts(z, target, double)[0]


@_entropy.defjvp
def _entropy_jvp(
    double: bool, primals: tuple[jax.Array, jax.Array], tangents: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    value, gradient = _entropy_parts(*primals, double)
    return value, (gradient * _work(tangents[0], double)).sum(-1)


def _targets(
    target: ArrayLike, logits: jax.Array, axis: int, objective: str
) -> tuple[jax.Array, int]:
    """Integer labels (...) or a boolean label mask (..., classes), and a bound on the label
    count of every example.

    Concrete targets are checked here; traced ones only by the losses (native: on the host).
    """
    shape, classes = np.shape(logits), logits.shape[axis]
    t = jnp.asarray(target)
    if t.dtype == jnp.bool_ and t.shape == shape:
        if not _concrete(target):
            return jnp.moveaxis(t, axis, -1), classes
        count = np.moveaxis(np.asarray(target), axis, -1).sum(-1)
        if (count == 0).any():
            raise ValueError("Every example needs a positive label")
        if objective.removesuffix("_smooth") in TOPK and (count != 1).any():
            raise ValueError("This loss needs exactly one target")
        if objective != "cross_entropy" and (count == classes).any():
            raise ValueError("This loss needs a negative label")
        return jnp.moveaxis(t, axis, -1), max(1, int(count.max()))
    if jnp.issubdtype(t.dtype, jnp.integer) and t.shape == shape[:axis] + shape[axis + 1 :]:
        if _concrete(target):
            labels = np.asarray(target)
            if ((labels < 0) | (labels >= classes)).any():
                raise ValueError("Target index out of range")
        return t, 1
    raise ValueError("target must contain integer class indices or be a boolean mask")


def sdca_loss(
    logits: ArrayLike,
    target: ArrayLike,
    loss: str = "cross_entropy",
    *,
    backend: str = "auto",
    k: int = 1,
    gamma: float = 0.0,
    alpha: float = 1.0,
    reduction: str = "mean",
    axis: int = -1,
    precision: object = None,
) -> jax.Array:
    """A loss of the scores (not a differentiable call to libsdca.fit).

    Gradients are first-order reverse mode (grad, vjp; not jvp or hessian),
    except that the jax entropy has derivatives of any order.

    Args:
        logits: Finite floating-point scores with the class axis
            (at least two classes and one example).
        target: Integer class indices of logits.shape without axis, or a
            boolean mask of logits.shape, as in libsdca.torch.sdca_loss.
        loss: One of libsdca.LOSSES.
        backend: "auto" (jax for entropy, and where float64 is enabled or
            precision="float32": off the CPU, and on it for small calls and,
            without the FFI handlers, the nonsmooth multiclass and multilabel
            hinges; else native),
            "native" (C++ on the host) or "jax" (jax.numpy on any device).
        k: Top-k parameter in [1, classes-1]; only the topk losses use it.
        gamma: Hinge smoothing, finite and >= 0 (0: nonsmooth).
        alpha: proximal_cross_entropy step, finite and positive.
        reduction: "none", "sum" or "mean".
        axis: Class axis of logits.
        precision: As in libsdca.torch.sdca_loss; float64 arithmetic in
            the jax backend needs jax_enable_x64.

    Returns:
        Per-example losses (logits.shape without axis) or their reduction.

    Raises:
        ValueError: For invalid inputs or options.
        JaxRuntimeError: If the native loss fails on the host: traced
            inputs are invalid or a proximal_cross_entropy normalization does not
            converge. The jax backend gives nan for those examples instead. On the
            CPU with float64 (or precision="float32"), "auto" picks the backend by
            objective and size: an invalid traced target of a loss picked by size
            raises above 16,384 scores and gives nan at or below; under vmap the
            size of one example decides.
    """
    # The private kind of the tensor backend; gamma is 1 for a nonsmooth hinge.
    objective, kind_k, gamma = _tensor_kind(loss, k, gamma)
    if backend not in ("auto", "native", "jax"):
        raise ValueError("Unknown backend")
    x = jnp.asarray(logits)
    if x.ndim < 1 or not jnp.issubdtype(x.dtype, jnp.floating):
        raise ValueError("logits must be finite floating-point values")
    if not -x.ndim <= axis < x.ndim:
        raise ValueError("axis is out of range")
    axis = axis % x.ndim
    z = jnp.moveaxis(x, axis, -1)
    classes = z.shape[-1]
    if classes < 2:
        raise ValueError("At least two classes are required")
    _check_options(k, gamma, alpha, classes)
    k = kind_k
    if z.size == 0:
        return _empty_loss(z, reduction)
    # Top-k entropy at k = 1 is the entropy of its one target.
    entropy = objective == "cross_entropy" or (objective == "topk_cross_entropy" and k == 1)
    canonical = _canonical_precision(precision)
    tensor = _jax_backend(
        x, "cross_entropy" if entropy else objective, backend, canonical, gamma, alpha, k, classes
    )
    _check_precision(x, canonical, tensor)
    if _concrete(x) and not bool(jnp.isfinite(x).all()):
        raise ValueError("logits must be finite floating-point values")
    target, width = _targets(target, x, axis, objective)
    if tensor and entropy:
        if objective != "cross_entropy" and target.dtype == jnp.bool_:
            # Rows without exactly one target (of a traced mask) give nan.
            target = jnp.where(target.sum(-1) == 1, target.argmax(-1), -1)
        values = _entropy(z, target, canonical == "float64")
    else:
        dtype = z.dtype if z.dtype in (jnp.float32, jnp.float64) else jnp.float32
        options = (not tensor, objective, int(k), float(gamma), float(alpha), canonical, width)
        values = _loss(z.astype(dtype), target, options)
    # Loss values are promote(logits dtype, float32), as libsdca.torch.
    return _reduce(values, reduction, _work(z).dtype)


def _single_thread_cpu() -> bool:
    if os.environ.get("SDCA_NUM_THREADS") == "1" or os.environ.get("OMP_NUM_THREADS") == "1":
        return True
    threads = re.search(r"intra_op_parallelism_threads=(\d+)", os.environ.get("XLA_FLAGS", ""))
    return threads is not None and int(threads.group(1)) == 1


def _distillation_max(
    student: jax.Array, teacher: jax.Array, temperature: float, double: bool
) -> tuple[jax.Array, jax.Array]:
    """Row maxima of student and teacher in the working dtype, reduced before widening."""
    return (
        _distillation_work(student.max(-1, keepdims=True), temperature, double),
        _distillation_work(teacher.max(-1, keepdims=True), temperature, double),
    )


def _distillation_pass(
    student: jax.Array,
    teacher: jax.Array,
    student_max: jax.Array,
    teacher_max: jax.Array,
    offset: tuple[jax.Array, jax.Array],
    temperature: float,
    double: bool,
    step: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Centered scaled logits s, t and offset-corrected scaled differences d for pass step."""
    step = 0 if _single_thread_cpu() else step
    cs = _distillation_work(student, temperature, double, step)
    ct = _distillation_work(teacher, temperature, double, step)
    # The represented differences less their common offset, in double-word arithmetic: the
    # differences of nearly equal logits stay exact, and an offset between student and teacher
    # does not round them (both are rounded by centering cs and ct first).
    delta = _jax_losses.difference(cs, ct)
    delta = (delta[0] - offset[0]) + (delta[1] - offset[1])
    # Plus z * 0: nan exactly where a logit is not finite, so the row's sums mark it invalid
    # without passes of their own.
    s = _scaled_difference(cs, student_max, temperature) + cs * 0
    t = _scaled_difference(ct, teacher_max, temperature) + ct * 0
    # The scaled differences, from the centered scaled logits where the raw difference
    # overflows (T > 1).
    d = _jax_losses.over(delta, temperature)
    if temperature > 1:
        d = jnp.where(jnp.isfinite(delta), d, s - t)
    return s, t, d


def _normalizer_index(z: jax.Array) -> jax.Array | None:
    """One maximum for the normalizer, or count its ties for an unbounded class dimension."""
    # GPU reductions benefit from excluding an index; CPU tie counting is faster.
    classes = z.shape[-1]
    if jax.default_backend() != "gpu" or not isinstance(classes, int) or classes >= 2**31:
        return None
    return lax.argmax(z, axis=z.ndim - 1, index_dtype=jnp.int32)[..., None]


def _distillation_value(
    student: jax.Array, teacher: jax.Array, temperature: float, double: bool
) -> jax.Array:
    """distillation_loss per row for the losses alone (the primal of _distillation, as
    _loss's values_only): the value of _distillation_stats without the double-word
    differences, the log-ratio correction and the gradient, which only a backward needs."""
    student_max, teacher_max = _distillation_max(student, teacher, temperature, double)
    first = _normalizer_index(student)
    cs = _distillation_work(student, temperature, double, 1)
    ct = _distillation_work(teacher, temperature, double, 1)
    s = _scaled_difference(cs, student_max, temperature) + cs * 0
    t = _scaled_difference(ct, teacher_max, temperature) + ct * 0
    zero = jnp.zeros_like(student_max)
    et = jnp.exp(t)
    _, rest, mass = _rest(s, zero, False, (et,), excluded=first)
    lognorm = jnp.log1p(rest)
    if _jax_losses._cpu() and student.size > _SMALL:
        s = _scaled_difference(
            _distillation_work(student, temperature, double, 2), student_max, temperature
        )
        et = jnp.exp(
            _scaled_difference(
                _distillation_work(teacher, temperature, double, 2), teacher_max, temperature
            )
        )
    p = et / mass
    value = _jax_losses.row_sums((jnp.where(p > 0, p * (lognorm - s), 0.0),), ())[0][..., 0]
    valid = (jnp.isfinite(rest) & jnp.isfinite(mass))[..., 0]
    return jnp.where(valid, value, jnp.nan)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _distillation(
    student: jax.Array, teacher: jax.Array, temperature: float, double: bool
) -> jax.Array:
    return _distillation_value(student, teacher, temperature, double)


def _distillation_stats(
    student: jax.Array, teacher: jax.Array, temperature: float, double: bool, exact_log_p: bool
) -> tuple[jax.Array, ...]:
    """Value, log p, log q, p, q, the log-ratio log(q / p), q - p, the row sum of u and the
    valid rows, as libsdca.torch. log p is accurate below eps only if exact_log_p (proximal
    distillation), which costs a pass for the teacher's ties."""
    # Center before scaling: a small temperature would overflow the logits.
    student_max, teacher_max = _distillation_max(student, teacher, temperature, double)
    first = _normalizer_index(student)
    offset = _jax_losses.difference(student_max, teacher_max)
    # The largest centered logits are exactly 0, and their exponentials 1.
    zero = jnp.zeros_like(student_max)
    s1, t1, _ = _distillation_pass(
        student, teacher, student_max, teacher_max, offset, temperature, double, 1
    )
    _, t2, d2 = _distillation_pass(
        student, teacher, student_max, teacher_max, offset, temperature, double, 2
    )
    et1, et2, teacher_top = jnp.exp(t1), jnp.exp(t2), t1 == 0
    # The teacher's mass (but one maximum for an exact log p) and the center's sum share
    # the student's pass.
    center = jnp.where(et2 > 0, et2 * d2, 0.0)
    if exact_log_p:
        teacher_term = jnp.where(teacher_top, 0.0, et1)
        _, rest, mass, center, ties = _rest(
            s1, zero, False, (teacher_term, center), (teacher_top,), excluded=first
        )
        teacher_lognorm = jnp.log1p(mass + (ties - 1))
        mass = mass + ties
    else:
        _, rest, mass, center = _rest(s1, zero, False, (et1, center), excluded=first)
        # The softmax of the teacher, as torch.softmax.
        teacher_lognorm = jnp.log(mass)
    lognorm = jnp.log1p(rest)
    valid = jnp.isfinite(rest) & jnp.isfinite(mass)
    center = center / mass
    shift = -center - teacher_lognorm
    s3, t3, _ = _distillation_pass(
        student, teacher, student_max, teacher_max, offset, temperature, double, 3
    )
    p3 = jnp.exp(t3) / mass
    val_term = jnp.where(p3 > 0, p3 * (lognorm - s3), 0.0)
    # Select Barron's fallback per class, so a distant negligible class does
    # not erase small probability differences among the dominant classes.
    # exp(log(p) + d) = exp(s + shift), using student logits to prevent
    # cancellation between a huge teacher tail and d, or expm1(d) overflow.
    s4, t4, d4 = _distillation_pass(
        student, teacher, student_max, teacher_max, offset, temperature, double, 4
    )
    p4, dn4 = jnp.exp(t4) / mass, d4 - center
    correction = jnp.where(
        jnp.abs(dn4) < 1,
        p4 * jnp.expm1(jnp.clip(dn4, -1, 1)),
        jnp.exp(s4 + shift) - p4,
    )
    value, corr_sum = _jax_losses.row_sums((val_term, correction), ())
    log1p_corr = jnp.log1p(corr_sum)

    def terms_of(step: int) -> tuple[jax.Array, ...]:
        s, t, d = _distillation_pass(
            student, teacher, student_max, teacher_max, offset, temperature, double, step
        )
        p, q = jnp.exp(t) / mass, jnp.exp(s) / (1 + rest)
        logratio = (d - center) - log1p_corr
        u = jnp.where(jnp.abs(logratio) < 1, p * jnp.expm1(jnp.clip(logratio, -1, 1)), q - p)
        return t - teacher_lognorm, s - lognorm, p, q, logratio, u

    log_p, log_q, p, q, logratio, u = terms_of(5)
    u_sum = u if exact_log_p else _jax_losses.row_sums((u,), ())[0]
    if (
        not exact_log_p
        and _jax_losses._cpu()
        and not _single_thread_cpu()
        and student.size > _SMALL
    ):
        _, _, p, q, logratio, u = terms_of(6)
    return value[..., 0], log_p, log_q, p, q, logratio, u, valid[..., 0], u_sum


def _distillation_gpu(
    student: jax.Array, teacher: jax.Array, temperature: float, double: bool
) -> tuple[jax.Array, ...]:
    """Plain distillation on the GPU: raw difference bounds share the maxima pass (fenced from
    the normalizer pass), so rows with |d - center| <= 2 skip the correction pass via _only_if
    and fuse value with u_sum."""
    student_max, teacher_max = _distillation_max(student, teacher, temperature, double)
    first = _normalizer_index(student)
    raw = _distillation_work(lax.stop_gradient(student), temperature, double) - _distillation_work(
        lax.stop_gradient(teacher), temperature, double
    )
    raw_min = cast(jax.Array, lax.optimization_barrier(raw.min(-1, keepdims=True)))
    raw_max = cast(jax.Array, lax.optimization_barrier(raw.max(-1, keepdims=True)))
    s_fenced = cast(jax.Array, lax.optimization_barrier(student))
    t_fenced = cast(jax.Array, lax.optimization_barrier(teacher))
    offset = _jax_losses.difference(student_max, teacher_max)
    zero = jnp.zeros_like(student_max)
    s, t, d = _distillation_pass(
        s_fenced, t_fenced, student_max, teacher_max, offset, temperature, double, 1
    )
    et = jnp.exp(t)
    terms = (et, jnp.where(et > 0, et * d, 0.0))
    _, rest, mass, center = _rest(s, zero, False, terms, excluded=first)
    center = center / mass
    valid = jnp.isfinite(rest) & jnp.isfinite(mass)
    shift_offset = offset[0] + offset[1]
    low = _jax_losses.over(raw_min - shift_offset, temperature) - center
    high = _jax_losses.over(raw_max - shift_offset, temperature) - center
    moderate = (jnp.abs(student_max) <= 1e4) & (jnp.abs(teacher_max) <= 1e4)
    bounded = valid & moderate & (low >= -1.95) & (high <= 1.95)
    p, q = et / mass, jnp.exp(s) / (1 + rest)
    dn = d - center

    def corrected() -> jax.Array:
        s, t, d = _distillation_pass(
            student, teacher, student_max, teacher_max, offset, temperature, double, 2
        )
        p, dn, shift = jnp.exp(t) / mass, d - center, -center - jnp.log(mass)
        correction = jnp.where(
            jnp.abs(dn) < 1, p * jnp.expm1(jnp.clip(dn, -1, 1)), jnp.exp(s + shift) - p
        )
        return jnp.log1p(_jax_losses.row_sums((correction,), ())[0])

    logratio = dn - jnp.where(bounded, 0.0, _only_if(~bounded.all(), corrected, zero))
    near = bounded | (jnp.abs(logratio) < 1)
    u = jnp.where(near, p * jnp.expm1(jnp.clip(logratio, -4, 4)), q - p)
    val_term = jnp.where(p > 0, p * (jnp.log1p(rest) - s), 0.0)
    value, u_sum = _jax_losses.row_sums((val_term, u), ())
    return value[..., 0], p, q, u, valid[..., 0], u_sum


def _first_maximum(x: jax.Array) -> jax.Array:
    """Where the first maximum of each row is (argmax)."""
    return jnp.arange(x.shape[-1]) == _jax_losses.row_first_max(x)[1]


def _distillation_fwd(
    student: jax.Array, teacher: jax.Array, temperature: float, double: bool
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    if jax.default_backend() == "gpu":
        value, p, q, u, valid, u_sum = _distillation_gpu(student, teacher, temperature, double)
    else:
        value, _, _, p, q, _, u, valid, u_sum = _distillation_stats(
            student, teacher, temperature, double, False
        )
    gradient = u - q * u_sum
    if student.shape[-1] <= 16:
        pivot = _first_maximum(p + q)
        others = jnp.where(pivot, 0.0, gradient)
        gradient = jnp.where(pivot, -_jax_losses.row_sums((others,), ())[0], others)
    gradient = _jax_losses.over(gradient, temperature)
    value = jnp.where(valid, value, jnp.nan)
    gradient = jnp.where(valid[..., None], gradient, jnp.nan)
    # float32 values for 16-bit logits, as libsdca.torch.
    return value, (gradient, jnp.zeros((0,), student.dtype))


def _scaled_difference(z: jax.Array, origin: jax.Array, temperature: float) -> jax.Array:
    """Center before division, rescuing overflowing differences when T > 1."""
    difference = z - origin
    scaled = _jax_losses.over(difference, temperature)
    if temperature > 1:
        divided = _jax_losses.over(z, temperature) - _jax_losses.over(origin, temperature)
        scaled = jnp.where(jnp.isfinite(difference), scaled, divided)
    return scaled


def _distillation_work(z: jax.Array, temperature: float, double: bool, step: int = 0) -> jax.Array:
    """The working logits: float64 if double or where the dtype cannot hold T and 1/T."""
    work = _work(_pass(z, step), double)
    tiny = float(jnp.finfo(work.dtype).tiny)
    if tiny <= temperature <= 1 / tiny:
        return work
    if not _x64():
        raise ValueError("temperature is outside the float32 range; enable jax_enable_x64")
    return work.astype(jnp.float64)


_distillation.defvjp(_distillation_fwd, _rounded_bwd)


def distillation_loss(
    student: ArrayLike,
    teacher: ArrayLike,
    temperature: float = 1.0,
    *,
    reduction: str = "mean",
    axis: int = -1,
    precision: object = None,
) -> jax.Array:
    """Cross-entropy with a fixed teacher distribution softmax(teacher/T).

    Both arguments are LOGITS. The teacher gets no gradient. No T² multiplier
    is applied. The first-order gradient is exactly zero at equal represented
    logits; near equality it avoids probability subtraction.

    Args:
        student: Finite floating-point logits with the class axis
            (at least two classes and one example).
        teacher: Finite logits of the same shape and dtype.
        temperature: T, finite and positive.
        reduction: "none", "sum" or "mean".
        axis: Class axis of the logits.
        precision: As in libsdca.torch.distillation_loss; float64 arithmetic needs
            jax_enable_x64.

    Returns:
        Per-example losses (student.shape without axis) or their reduction.

    Raises:
        ValueError: For invalid inputs or options.
    """
    s, t, double = _distillation_pair(student, teacher, temperature, axis, precision)
    if s.size == 0:
        return _empty_loss(s, reduction)
    values = _distillation(s, t, float(temperature), double)
    return _reduce(values, reduction, _work(s).dtype)


def _distillation_pair(
    student: ArrayLike, teacher: ArrayLike, temperature: float, axis: int, precision: object
) -> tuple[jax.Array, jax.Array, bool]:
    """Checked logits with the class axis last (teacher without gradient), and float64."""
    s, t = jnp.asarray(student), jnp.asarray(teacher)
    if s.shape != t.shape or s.dtype != t.dtype or s.ndim < 1:
        raise ValueError("Student and teacher must have matching shape and dtype")
    if not jnp.issubdtype(s.dtype, jnp.floating) or (
        _concrete(s, t) and not bool(jnp.isfinite(s).all() & jnp.isfinite(t).all())
    ):
        raise ValueError("Logits must be finite floating-point arrays")
    canonical = _canonical_precision(precision)
    _check_precision(s, canonical, True)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not -s.ndim <= axis < s.ndim:
        raise ValueError("axis is out of range")
    s, t = jnp.moveaxis(s, axis, -1), jnp.moveaxis(jax.lax.stop_gradient(t), axis, -1)
    if s.shape[-1] < 2:
        raise ValueError("At least two classes are required")
    return s, t, canonical == "float64"


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _proximal_distillation(
    student: jax.Array, teacher: jax.Array, temperature: float, alpha: float, double: bool
) -> jax.Array:
    return _proximal_distillation_fwd(student, teacher, temperature, alpha, double)[0]


def _proximal_distillation_fwd(
    student: jax.Array, teacher: jax.Array, temperature: float, alpha: float, double: bool
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    *stats, valid, _ = _distillation_stats(student, teacher, temperature, double, True)[1:]
    if _accelerator(student):
        # XLA:GPU compiles for minutes when it fuses these statistics into the solve.
        stats = lax.optimization_barrier(stats)
    log_p, log_q, p, q, logratio, u = stats
    value, gradient = _jax_losses.proximal_distillation(
        log_p, log_q, p, q, logratio, u, alpha, ~valid[..., None]
    )
    value = jnp.where(valid, value, jnp.nan)
    gradient = jnp.where(valid[..., None], _jax_losses.over(gradient, temperature), jnp.nan)
    return value, (gradient, jnp.zeros((0,), student.dtype))


_proximal_distillation.defvjp(_proximal_distillation_fwd, _rounded_bwd)


def proximal_distillation_loss(
    student: ArrayLike,
    teacher: ArrayLike,
    temperature: float = 1.0,
    alpha: float = 1.0,
    *,
    reduction: str = "mean",
    axis: int = -1,
    precision: object = None,
) -> jax.Array:
    """Moreau envelope of distillation_loss with step alpha, as in libsdca.torch.

    min_u { H(softmax(teacher/T), softmax(u)) + |u - student/T|^2 / (2 alpha) }; first-order
    reverse mode, exactly zero gradient at equal represented logits. alpha outside the float32
    range computes in float64 (needs jax_enable_x64). alpha is finite and positive; other
    arguments, the result and errors as in distillation_loss.
    """
    s, t, double = _distillation_pair(student, teacher, temperature, axis, precision)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not _float32_holds(alpha):
        if not _x64():
            raise ValueError("alpha is outside the float32 range; enable jax_enable_x64")
        double = True
    if s.size == 0:
        return _empty_loss(s, reduction)
    values = _proximal_distillation(s, t, float(temperature), float(alpha), double)
    return _reduce(values, reduction, _work(s).dtype)


def cross_entropy(
    logits: ArrayLike,
    target: ArrayLike,
    *,
    reduction: str = "mean",
    axis: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> jax.Array:
    """Softmax cross-entropy loss of integer labels or a boolean mask.

    Other arguments, the result and errors as in sdca_loss.
    """
    return sdca_loss(
        logits,
        target,
        "cross_entropy",
        backend=backend,
        reduction=reduction,
        axis=axis,
        precision=precision,
    )


def topk_cross_entropy(
    logits: ArrayLike,
    target: ArrayLike,
    k: int = 1,
    *,
    reduction: str = "mean",
    axis: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> jax.Array:
    """Top-k cross-entropy (the top-k entropy loss); cross_entropy at k = 1.

    Other arguments, the result and errors as in sdca_loss.
    """
    return sdca_loss(
        logits,
        target,
        "topk_cross_entropy",
        k=k,
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )


def multiclass_hinge_loss(
    logits: ArrayLike,
    target: ArrayLike,
    gamma: float = 0.0,
    *,
    reduction: str = "mean",
    axis: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> jax.Array:
    """Multiclass hinge loss of Crammer and Singer, smoothed by gamma > 0.

    Other arguments, the result and errors as in sdca_loss.
    """
    return sdca_loss(
        logits,
        target,
        "multiclass_hinge",
        gamma=gamma,
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )


def topk_hinge_loss(
    logits: ArrayLike,
    target: ArrayLike,
    k: int = 1,
    *,
    variant: str = "beta",
    gamma: float = 0.0,
    reduction: str = "mean",
    axis: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> jax.Array:
    """Top-k hinge loss, smoothed by gamma > 0. Variant "beta" averages the k
    largest hinges; "alpha" is the hinge of the average of the k largest margins.

    variant is "alpha" or "beta"; other arguments, the result and errors as in sdca_loss.
    """
    name = _topk_hinge_name(variant)
    return sdca_loss(
        logits,
        target,
        name,
        k=k,
        gamma=gamma,
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )


def topk_hinge_alpha_loss(
    logits: ArrayLike,
    target: ArrayLike,
    k: int = 1,
    *,
    gamma: float = 0.0,
    reduction: str = "mean",
    axis: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> jax.Array:
    """topk_hinge_loss with variant="alpha".

    Other arguments, the result and errors as in sdca_loss.
    """
    return sdca_loss(
        logits,
        target,
        "topk_hinge_alpha",
        k=k,
        gamma=gamma,
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )


def topk_hinge_beta_loss(
    logits: ArrayLike,
    target: ArrayLike,
    k: int = 1,
    *,
    gamma: float = 0.0,
    reduction: str = "mean",
    axis: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> jax.Array:
    """topk_hinge_loss with variant="beta".

    Other arguments, the result and errors as in sdca_loss.
    """
    return sdca_loss(
        logits,
        target,
        "topk_hinge_beta",
        k=k,
        gamma=gamma,
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )


def multilabel_cross_entropy(
    logits: ArrayLike,
    target: ArrayLike,
    *,
    reduction: str = "mean",
    axis: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> jax.Array:
    """Multilabel cross-entropy: uniform target probabilities on the positive labels.

    Other arguments, the result and errors as in sdca_loss.
    """
    return sdca_loss(
        logits,
        target,
        "multilabel_cross_entropy",
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )


def multilabel_hinge_loss(
    logits: ArrayLike,
    target: ArrayLike,
    gamma: float = 0.0,
    *,
    reduction: str = "mean",
    axis: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> jax.Array:
    """Multilabel hinge loss, smoothed by gamma > 0.

    Other arguments, the result and errors as in sdca_loss.
    """
    return sdca_loss(
        logits,
        target,
        "multilabel_hinge",
        gamma=gamma,
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )


def proximal_cross_entropy(
    logits: ArrayLike,
    target: ArrayLike,
    alpha: float = 1.0,
    *,
    reduction: str = "mean",
    axis: int = -1,
    backend: str = "auto",
    precision: object = None,
) -> jax.Array:
    """Moreau envelope of the cross-entropy with step alpha.

    Other arguments, the result and errors as in sdca_loss.
    """
    return sdca_loss(
        logits,
        target,
        "proximal_cross_entropy",
        alpha=alpha,
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )
