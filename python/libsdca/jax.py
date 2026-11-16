"""JAX losses under jit, grad and vmap: the cross-entropies, top-k and hinge losses of class
labels, the multilabel losses of boolean masks, and distillation_loss and
proximal_distillation_loss of teacher logits. The native backend calls the C ABI through an
XLA FFI call on the CPU (_jax_ffi) and through jax.pure_callback elsewhere or where the FFI
handlers do not build; the jax backend computes every loss with jax.numpy on any device
(entropy to any order, the others first-order reverse mode).
"""

from __future__ import annotations

import functools
import math
import os
import re
import sys
from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.typing import ArrayLike
from numpy.typing import DTypeLike

from . import _jax_cpu, _jax_cuda, _jax_ffi, _jax_fusion, _jax_losses
from . import _loss_options as _options
from ._core import _IDS, _PRECISION_CODES, _host_loss_and_grad
from ._jax_compat import optimization_barrier as _optimization_barrier

__all__ = [
    "cross_entropy",
    "distillation_loss",
    "multiclass_hinge_loss",
    "multilabel_cross_entropy",
    "multilabel_hinge_loss",
    "proximal_cross_entropy",
    "proximal_distillation_loss",
    "topk_cross_entropy",
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


def _default_precision(
    precision: str | None, objective: str, dtype: Any, backend: str, k: int, classes: int, size: int
) -> str | None:
    """Measured fp32 defaults, preserving faster double work in the Mac serial region."""
    if precision is not None or dtype == jnp.float64:
        return precision
    # The native scheduler can leave short fp32 solves serial while the more
    # expensive double solve crosses its parallel-work threshold on macOS.
    narrow_serial = (
        sys.platform == "darwin"
        and backend != "jax"
        and classes < 32
        and size > _SMALL
        and not _single_thread_cpu()
    )
    fast32 = objective == "proximal_cross_entropy" or (
        objective == "topk_cross_entropy" and 1 < k and classes <= 64 and not narrow_serial
    )
    return "float32" if fast32 else None


def _work_precision(objective: str, canonical: str | None, x: jax.Array) -> str | None:
    """The precision of the jax solve of a non-entropy loss. "float32" takes float64 for the
    smoothed hinges, which form their entries in float64 (docs/API.md#precision): the jax
    backend with x64, else native. None stays float64 without x64 too: the float32 solve
    loses subnormal gaps and outputs where XLA flushes them (XLA:CPU)."""
    if objective.endswith("_smooth") and canonical == "float32" and x.dtype != jnp.float64:
        return None
    return canonical


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
    labels (the leading shape) or a label mask. Rows the native library rejects (traced
    invalid inputs) give NaN: the call is repeated row by row only after a failure."""
    z, positive = np.asarray(z), np.asarray(positive)
    classes = z.shape[-1]
    name, nonsmooth = _native_name(loss, positive.dtype == np.bool_)
    flat = z.reshape(-1, classes)
    if positive.ndim < z.ndim:
        # A scatter into zeros, not a comparison of every class; out of range, a label marks
        # no class (the native call rejects the row).
        labels = positive.reshape(-1).astype(np.int64)
        target = np.zeros((labels.size, classes), np.bool_)
        rows = np.flatnonzero((labels >= 0) & (labels < classes))
        target[rows, labels[rows]] = True
    else:
        target = positive.reshape(-1, classes)

    def call(rows: slice) -> tuple[np.ndarray, np.ndarray]:
        return _host_loss_and_grad(
            flat[rows],
            target[rows],
            name,
            k=k,
            gamma=gamma if nonsmooth is None else nonsmooth,
            alpha=alpha,
            # SDCA_NUM_THREADS, else the available CPUs, as XLA's own CPU operations.
            num_threads=None,
            # Match _attributes: the native JAX adapter owns its precision policy.
            # Do not inherit a change in the NumPy convenience API's default.
            precision="float64" if precision is None else precision,
        )

    try:
        values, gradients = call(slice(None))
    except (ValueError, RuntimeError):
        values = np.full(flat.shape[0], np.nan)
        gradients = np.full(flat.shape, np.nan, flat.dtype)
        for row in range(flat.shape[0]):
            try:
                values[row], gradients[row] = (a[0] for a in call(slice(row, row + 1)))
            except (ValueError, RuntimeError):
                pass
    return values.astype(values_dtype).reshape(z.shape[:-1]), gradients.reshape(z.shape)


def _native_name(kind: str, mask: bool) -> tuple[str, float | None]:
    """The native loss of a tensor kind and target (the entropy of a mask is
    multilabel_cross_entropy), and gamma 0 for a nonsmooth kind (None: the given gamma)."""
    name, nonsmooth = _options.loss_of_kind(kind)
    return ("multilabel_cross_entropy" if mask and name == "cross_entropy" else name), nonsmooth


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


def _attributes(options: _Options, dtype: DTypeLike, mask: bool) -> dict[str, np.generic]:
    """The options of the FFI handlers, as _host passes them to _host_loss_and_grad."""
    _, loss, k, gamma, alpha, precision, _ = options
    name, nonsmooth = _native_name(loss, mask)
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
        values, gradients = call(
            z, positive, **_attributes(options, z.dtype, positive.dtype == jnp.bool_)
        )
        return values, gradients
    host = _host_of(options, values.dtype)
    values, gradients = jax.pure_callback(host, shapes, z, positive, vmap_method="broadcast_all")
    return values, gradients


def _native_values(z: jax.Array, positive: jax.Array, options: _Options) -> jax.Array:
    """The native losses alone: the handler (or host) returns no gradient."""
    shape = jax.ShapeDtypeStruct(z.shape[:-1], _values_dtype(z.dtype))
    if _ffi(z):
        call = jax.ffi.ffi_call(_jax_ffi.VALUES, shape, vmap_method="broadcast_all")
        return call(z, positive, **_attributes(options, z.dtype, positive.dtype == jnp.bool_))
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
        mask = _optimization_barrier(mask)

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
        if loss == "proximal_cross_entropy":
            # Without x64 its rows whose spread float32 cannot hold have no float64 to fall
            # back to (the hinges and top-k entropy round those rows correctly in float32).
            held = jnp.isfinite(flat.max(-1) - flat.min(-1))
            values = jnp.where(held, values, jnp.nan)
            gradients = jnp.where(held[:, None], gradients, jnp.nan)
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
    raise TypeError("This loss has no JVP: its derivatives are first-order reverse mode")


def _call(z: jax.Array, positive: jax.Array, options: _Options) -> tuple[jax.Array, jax.Array]:
    if options[0]:
        # Forward mode over this rule (hessian) raises here, not in the FFI call or callback.
        return _native_call(_first_order(z), positive, options)
    values, gradients = _tensor_call(z, positive, options)
    return values, _first_order(gradients)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _rows(z: jax.Array, positive: jax.Array, options: _Options) -> jax.Array:
    if options[0]:
        return _native_values(z, positive, options)
    return _tensor_call(z, positive, options, values_only=True)[0]


def _saved_bwd(*args: Any) -> tuple[jax.Array, None]:
    """The saved gradient (the residual after the static options) times the cotangent; the
    second argument (target or teacher) gets none."""
    gradient, ct = args[-2:]
    return (gradient * ct[..., None]).astype(gradient.dtype), None


_rows.defvjp(_call, _saved_bwd)


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
    return _optimization_barrier((x, jnp.int32(step)))[0]


@_fenced.defjvp
def _fenced_jvp(
    step: int, primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    return _fenced(primals[0], step), _fenced(tangents[0], step)


def _pass(x: jax.Array, step: int, fenced: bool = True) -> jax.Array:
    """A view of x for pass step on the CPU above _SMALL where fenced (_fences), so XLA:CPU
    fuses each pass instead of materializing shared intermediates on four threads."""
    return _fenced(x, step) if fenced and step and _jax_losses._cpu() and x.size > _SMALL else x


def _fences(z: jax.Array, family: str) -> bool:
    """Whether the passes of a loss family over z, traced for the CPU above _SMALL scores, each
    take a view of their own (_pass) and recompute their exponentials: SDCA_JAX_CPU_FUSION,
    read at trace time (_jax_fusion)."""
    if not (_jax_losses._cpu() and z.size > _SMALL):
        return False
    return _jax_fusion.fenced(family, z.dtype, _single_thread_cpu())


def _rest(
    z: jax.Array,
    maximum: jax.Array,
    finite: bool = False,
    terms: tuple[jax.Array, ...] = (),
    masks: tuple[jax.Array, ...] = (),
    excluded: jax.Array | None = None,
    far: jax.typing.DTypeLike | None = None,
) -> tuple[jax.Array, ...]:
    """The exponentials of z - maximum and the mass of the classes other than one maximum: in
    one pass, accurate below eps for log1p (a tied maximum adds
    exactly one). finite: rows with a nonfinite score give nan. The row sums of further terms
    and counts of masks share the pass, after the mass. excluded: the index of one maximum,
    if already available, avoiding a separate count of tied maxima. far: the dtype of the
    mass with far gaps rescaled (_rescanned), returned last; the mass is then its rounding."""
    top = (
        z == maximum
        if excluded is None
        else jnp.arange(z.shape[-1], dtype=excluded.dtype) == excluded
    )
    e = jnp.exp(z - maximum)
    # _pinned keeps the derivatives at the excluded maxima.
    rest = _pinned(e, top)
    if far is not None:
        shifted, flags = _shifted(z - maximum)
        rest = _pinned(shifted, top)
        terms = (jnp.where(flags, rest, 0.0), *terms)
        rest = jnp.where(flags, 0.0, rest)
    if finite:
        # z * 0 is nan exactly where a score is not finite (a difference to the maximum may
        # overflow to -inf).
        rest = rest + z * 0
    if far is None:
        if excluded is not None:
            return (e, *_jax_losses.row_sums((rest, *terms), masks))
        mass, *sums, ties = _jax_losses.row_sums((rest, *terms), (*masks, top))
        return e, mass + (ties - 1), *sums
    if excluded is not None:
        near, tail, *sums = _jax_losses.row_sums((rest, *terms), masks)
        ties = jnp.ones_like(near)
    else:
        near, tail, *sums, ties = _jax_losses.row_sums((rest, *terms), (*masks, top))
    mass = near.astype(far) + tail.astype(far) * np.asarray(_FAR_UNIT, far)
    return e, mass.astype(near.dtype) + (ties - 1), *sums, mass


def _label_entropy(z: jax.Array, labels: jax.Array, double: bool) -> tuple[jax.Array, jax.Array]:
    """Entropy loss and gradient of integer labels in two passes over the scores, as a
    composed log-softmax: the maximum, and the mass of the classes but one maximum.

    The mass leaves out a maximum, so log1p keeps tiny losses, and the target gradient is minus
    the mass of the other classes, which keeps it accurate on confident rows. Autodiff of these
    formulas gives the higher derivatives.
    """
    classes = jnp.arange(z.shape[-1])
    maximum = _work(z.max(-1, keepdims=True), double)
    fenced = _fences(z, "cross_entropy")
    zw = _work(_pass(z, 1, fenced), double)
    dtype = _far_dtype(maximum.dtype)
    mass = None
    if dtype is not None and _FAR_TAIL == "inline":
        e, rest, mass = _rest(zw, maximum, finite=True, far=dtype)
    else:
        e, rest = _rest(zw, maximum, finite=True)
    # A label out of range (no wrapping of negative ones) picks nan, which spreads to the row.
    index = labels[..., None].astype(jnp.promote_types(labels.dtype, jnp.int32))
    index = jnp.where((index >= 0) & (index < z.shape[-1]), index, z.shape[-1])
    picked = _work(jnp.take_along_axis(z, index, -1, mode="fill", fill_value=jnp.nan), double)
    total = (1 + rest) + picked * 0
    target = jnp.exp(picked - maximum)
    others = jnp.where(picked == maximum, rest - _pinned(target, True), 1 + (rest - target))
    value = jnp.log1p(rest[..., 0]) + (maximum - picked)[..., 0]
    entry = -others / total
    if dtype is not None:
        # Only a maximum label's loss and entry are the mass alone.
        far = (rest < z.shape[-1] * _FAR_SUM) & (picked == maximum)

        def rescan() -> jax.Array:
            zw = _work(z, double)
            e, flags = _shifted(zw - maximum)
            return _rescanned(((_pinned(e, zw == maximum), flags),), dtype)[0]

        if mass is None:
            mass = _only_if(far.any(), rescan, rest.astype(dtype))
        gap = (maximum - picked).astype(dtype)
        value = jnp.where(far[..., 0], (mass + gap)[..., 0].astype(value.dtype), value)
        tail = -(mass - _pinned(target, True).astype(dtype))
        entry = jnp.where(far, tail.astype(entry.dtype), entry)
    if fenced:
        e = jnp.exp(_work(_pass(z, 2), double) - maximum)
    # Times the reciprocal of the row's total, as a composed softmax.
    probability = e * (1 / total)
    return value, jnp.where(classes == index, entry, probability)


def _mask_entropy(z: jax.Array, positive: jax.Array, double: bool) -> tuple[jax.Array, jax.Array]:
    """Entropy loss and gradient of a label mask (uniform target), as torch._TensorEntropy:
    small logit differences stay exact in the gradient. Three passes over the scores for the
    value (the maximum; the mass with the counts; the label gaps), three more for the
    gradient (a positive anchor, the correction and the gradient)."""
    maximum = _work(z.max(-1, keepdims=True), double)
    if jax.default_backend() == "gpu":
        return _mask_entropy_gpu(z, positive, maximum, double)
    fenced = _fences(z, "multilabel_cross_entropy")
    difference = maximum - _work(_pass(z, 2, fenced), double)
    # The mass but one maximum (log1p keeps tiny losses; nan for a nonfinite score), the sum
    # of the label gaps and the label count.
    terms = (jnp.where(positive, difference, 0.0),)
    scores = _work(_pass(z, 1, fenced), double)
    e, rest, gaps, count = _rest(scores, maximum, True, terms, (positive,))

    def rescued() -> jax.Array:
        scaled = _scaled_gaps(z, positive, maximum, double, 3 if fenced else 0)
        return _mean_gap(gaps, _jax_losses.row_sums((scaled,), ())[0], count, z.shape[-1])

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
        zw = _work(_pass(z, step, fenced), double)
        ew = jnp.exp(zw - maximum) if fenced else e
        differences = _anchored(zw, ew, z_anchor, e_anchor) / den
        return jnp.where(positive, differences, ew / den)

    correction = _jax_losses.row_sums((raw(4),), ())[0] / count
    dtype = _far_dtype(maximum.dtype)
    if dtype is not None:
        value, correction = _mask_far_rows(
            z, positive, maximum, double, rest, gaps, count, value, correction, z_anchor, dtype
        )
    entries = raw(5)
    gradient = jnp.where(positive, entries - correction, entries)
    valid = count > 0
    return jnp.where(valid[..., 0], value, jnp.nan), jnp.where(valid, gradient, jnp.nan)


def _mask_far_rows(
    z: jax.Array,
    positive: jax.Array,
    maximum: jax.Array,
    double: bool,
    rest: jax.Array,
    gap: jax.Array,
    count: jax.Array,
    value: jax.Array,
    correction: jax.Array,
    z_anchor: jax.Array,
    dtype: jax.typing.DTypeLike,
) -> tuple[jax.Array, jax.Array]:
    """_mask_entropy's loss and correction of float32 rows with a far mass: the loss of a row
    whose one positive holds the maximum is its mass, and a small correction is the sum of
    the positives' differences and the negatives' mass, far ones included (_rescanned)."""
    bound = z.shape[-1] * _FAR_SUM
    light = (rest < bound) & (gap == 0)
    weak = jnp.abs(correction) < bound
    den = 1 + rest

    def rescan(numerator: bool) -> jax.Array:
        zw = _work(z, double)
        e = jnp.exp(zw - maximum)
        shifted, flags = _shifted(zw - maximum)
        if not numerator:
            return _rescanned(((_pinned(shifted, zw == maximum), flags),), dtype)[0]
        negative = flags & ~positive
        differences = jnp.where(
            positive,
            _anchored(zw, e, z_anchor, jnp.exp(z_anchor - maximum)),
            jnp.where(negative, shifted, e),
        )
        return _rescanned(((differences, negative),), dtype)[0]

    # Separate passes: the loss alone needs no correction.
    mass = _only_if(light.any(), lambda: rescan(False), rest.astype(dtype))
    value = jnp.where(light[..., 0], mass[..., 0].astype(value.dtype), value)
    default = (correction * count * den).astype(dtype)
    numerator = _only_if(weak.any(), lambda: rescan(True), default)
    # Rounded once.
    tail = numerator / den.astype(dtype) / count.astype(dtype)
    return value, jnp.where(weak, tail.astype(correction.dtype), correction)


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
    dtype = _far_dtype(maximum.dtype)
    inline = dtype is not None and _FAR_TAIL == "inline"
    e, rest, gaps, scaled, held, count, *mass = _rest(
        zw, maximum, True, terms, (positive,), far=dtype if inline else None
    )
    mean = _mean_gap(gaps, scaled, count, z.shape[-1])
    value = mean[..., 0] + jnp.log1p(rest[..., 0])
    den = 1 + rest
    bound = z.shape[-1] * _FAR_SUM
    if dtype is not None:
        # The loss of a row whose positives all hold the maximum is its mass.
        light = (rest < bound) & (mean == 0)

        def rescan() -> jax.Array:
            zw = _work(z, double)
            e, flags = _shifted(zw - maximum)
            return _rescanned(((_pinned(e, zw == maximum), flags),), dtype)[0]

        tail = mass[0] if inline else _only_if(light.any(), rescan, rest.astype(dtype))
        value = jnp.where(light[..., 0], tail[..., 0].astype(value.dtype), value)
    # The numerator of the correction, the sum of the negatives' mass and the positives'
    # differences to the anchor, is den - count e_anchor. Where the positives hold at most
    # half the mass, that difference is within a small factor of the sum's own rounding.
    # Elsewhere it is summed from the differences, in a pass of its own.
    ordinary = held <= den / 2

    def summed() -> tuple[jax.Array, ...]:
        # From the scores anew: a conditional's operands are stored.
        zw = _work(z, double)
        if dtype is None:
            e = jnp.exp(zw - maximum)
            differences = jnp.where(positive, _anchored(zw, e, z_anchor, e_anchor), e)
            return _jax_losses.row_sums((differences,), ())
        # The negatives' far mass (_rescanned) apart, which may be a row's whole numerator.
        # _anchored takes e only at positives from the anchor up, which are near the maximum
        # where the positives hold most of the mass (the rows this pass is for).
        e, flags = _shifted(zw - maximum)
        differences = jnp.where(
            positive, _anchored(zw, e, z_anchor, e_anchor), jnp.where(flags, 0.0, e)
        )
        return _jax_losses.row_sums((differences, jnp.where(positive | ~flags, 0.0, e)), ())

    sums = _only_if(~ordinary.all(), summed, (den,) if dtype is None else (den, den))
    tail = sums[0]
    if dtype is not None:
        tail = tail.astype(dtype) + sums[1].astype(dtype) * np.asarray(_FAR_UNIT, dtype)
    numerator = jnp.where(ordinary, den - count * e_anchor, tail.astype(den.dtype))
    correction = numerator / den / count
    if dtype is not None and dtype != den.dtype:
        # Rounded once.
        tail = tail / den.astype(dtype) / count.astype(dtype)
        correction = jnp.where(ordinary, correction, tail.astype(correction.dtype))
    entries = jnp.where(positive, _anchored(zw, e, z_anchor, e_anchor), e) / den
    gradient = jnp.where(positive, entries - correction, entries)
    valid = count > 0
    return jnp.where(valid[..., 0], value, jnp.nan), jnp.where(valid, gradient, jnp.nan)


def _only_if(pred: jax.Array, fn: Callable[[], Any], default: Any) -> Any:
    """fn() where pred holds, else default, computing fn only then. XLA flattens a
    conditional of elementwise operations and reductions into a select of both branches;
    the barrier keeps the branch a conditional."""
    return lax.cond(pred, lambda: _optimization_barrier(fn()), lambda: default)


# Float32 exponentials below exp(-87.3) = 2^-126 are subnormal (flushed by XLA:CPU, rounded to
# zero below exp(-103.97) on the GPU), losing under C * 2^-126 of a row's sum: under 2^-36 of
# a sum of C * 2^-90 or more. Float32 rows whose mass falls below that are summed again
# (_rescanned) with far gaps (below -80) taken at gap + 64 and those sums scaled by exp(-64),
# in a pass that batches without such a row skip (_only_if); the masks' correction sums its
# far negatives in its own pass. In those rows log1p(mass) is the mass and 1 + mass is 1.
# exp(gap + 64) is normal down to a gap of -151.3, so XLA:CPU keeps every term of a sum
# that float32 can hold.
_FAR_SUM = 2.0**-90
_FAR_GAP = -80.0
_FAR_UNIT = 1.603810890548638e-28  # exp(-64)
# The dtype of the rescaled sums, read at trace time: "float32", "float64" (rounding each
# row's sums once; float32 without jax_enable_x64), "inline" (float32; the entropy of labels
# and masks rescales the mass in its pass over the scores instead of a rescan) or "off".
_FAR_TAIL = "float32"


def _far_dtype(work: jax.typing.DTypeLike) -> jax.typing.DTypeLike | None:
    """The dtype of the far-tail rescans of work (float32 only), None if none."""
    if _FAR_TAIL == "off" or work != jnp.float32:
        return None
    return jnp.float64 if _FAR_TAIL == "float64" and _x64() else jnp.float32


def _shifted(gap: jax.Array) -> tuple[jax.Array, jax.Array]:
    """exp(gap) with far gaps (below _FAR_GAP) taken at gap + 64, and the far flags.
    exp(gap + 64) is itself subnormal below a gap of about -151 and zero below -168: under
    C * 2^-218 of a sum, below the float32 range for any row of the rescan."""
    far = gap < _FAR_GAP
    return jnp.exp(jnp.where(far, gap + 64, gap)), far


def _rescanned(
    pairs: tuple[tuple[jax.Array, jax.Array], ...], dtype: jax.typing.DTypeLike
) -> tuple[jax.Array, ...]:
    """Row sums (keepdims, in dtype) of terms from _shifted, each with its far flags: the
    near terms plus exp(-64) times the far ones, in one pass."""
    terms = [
        x for term, far in pairs for x in (jnp.where(far, 0.0, term), jnp.where(far, term, 0.0))
    ]
    sums = _jax_losses.row_sums(tuple(terms), ())
    unit = np.asarray(_FAR_UNIT, dtype)
    return tuple(
        n.astype(dtype) + f.astype(dtype) * unit for n, f in zip(sums[::2], sums[1::2], strict=True)
    )


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


def _reduced_entropy(z: jax.Array, target: jax.Array, double: bool, reduction: str) -> jax.Array:
    """The reduced entropy whose derivatives are the fused kernels' higher ones (_jax_cuda)."""
    return _reduce(_entropy(z, target, double), reduction, _work(z).dtype)


def _label_array(
    target: ArrayLike, shape: tuple[int, ...], axis: int, spec: _options.Spec
) -> jax.Array:
    """The integer labels of a single-label loss (logits.shape without axis), values unchecked."""
    t = jnp.asarray(target)
    if t.dtype == jnp.bool_ or not jnp.issubdtype(t.dtype, jnp.integer):
        hint = (
            "; boolean masks are the targets of multilabel_cross_entropy and multilabel_hinge_loss"
            if t.dtype == jnp.bool_
            else ""
        )
        raise TypeError(f"{spec.name} takes integer class labels, not {t.dtype} targets{hint}")
    expected = shape[:axis] + shape[axis + 1 :]
    if t.shape != expected:
        raise ValueError(f"labels must have the logits' shape without the class axis, {expected}")
    return t


def _targets(
    target: ArrayLike, logits: jax.Array, axis: int, spec: _options.Spec
) -> tuple[jax.Array, int]:
    """Integer labels (...) of a single-label loss or the boolean mask (..., classes) of a
    multilabel one, and a bound on the label count of every example.

    Concrete targets are checked here; traced ones only by the losses (invalid rows give NaN).
    """
    shape, classes = tuple(np.shape(logits)), logits.shape[axis]
    if not spec.multilabel:
        t = _label_array(target, shape, axis, spec)
        if _concrete(target):
            labels = np.asarray(target)
            if ((labels < 0) | (labels >= classes)).any():
                raise ValueError("Target index out of range")
        return t, 1
    t = jnp.asarray(target)
    if t.dtype != jnp.bool_:
        raise TypeError(f"{spec.name} takes a boolean mask of positive labels, not {t.dtype}")
    if t.shape != shape:
        raise ValueError(f"the mask must have the logits' shape {shape}")
    mask = jnp.moveaxis(t, axis, -1)
    if not _concrete(target):
        return mask, classes
    count = np.moveaxis(np.asarray(target), axis, -1).sum(-1)
    if (count == 0).any():
        raise ValueError("Every example needs a positive label")
    if spec.objective != "cross_entropy" and (count == classes).any():
        raise ValueError("This loss needs a negative label")
    return mask, max(1, int(count.max()))


def _class_last(logits: ArrayLike, axis: int) -> tuple[jax.Array, jax.Array, int]:
    """Validate the class axis and floating dtype, preserving the original array too."""
    x = jnp.asarray(logits)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError(f"logits must be a floating-point array, not {x.dtype}")
    if x.ndim < 1:
        raise ValueError("logits need a class axis")
    if not -x.ndim <= axis < x.ndim:
        raise ValueError("axis is out of range")
    axis %= x.ndim
    z = jnp.moveaxis(x, axis, -1)
    if z.shape[-1] < 2:
        raise ValueError("At least two classes are required")
    return x, z, axis


_BACKENDS = ("auto", "native", "jax")
Reduction = Literal["none", "sum", "mean"]
Backend = Literal["auto", "native", "jax"]
_PRECISIONS: dict[object, str] = {
    np.float32: "float32",
    np.float64: "float64",
    np.dtype(np.float32): "float32",
    np.dtype(np.float64): "float64",
    jnp.float32: "float32",
    jnp.float64: "float64",
}


def _static(value: object, name: str) -> object:
    """A hyperparameter that selects kernels and arithmetic: a Python number or a concrete 0-d
    array; a traced value raises TypeError."""
    if isinstance(value, jax.core.Tracer):
        raise TypeError(
            f"{name} must be a static Python number, not a traced value; under jax.jit pass it "
            "as a static argument or close over it"
        )
    if isinstance(value, jax.Array | np.ndarray | np.generic) and np.ndim(value) == 0:
        return np.asarray(value).item()
    return value


class _Common(NamedTuple):
    """The checked options every loss takes."""

    reduction: str
    axis: int
    backend: str
    precision: str | None


def _common(reduction: object, axis: object, backend: object, precision: object) -> _Common:
    canonical = _options.check_precision_name(precision, _PRECISIONS, "jnp")
    if canonical == "float64" and not _x64():
        raise ValueError("precision='float64' needs jax_enable_x64")
    return _Common(
        _options.check_reduction(reduction),
        _options.check_axis(axis, "axis"),
        _options.check_backend(backend, _BACKENDS),
        canonical,
    )


def _loss(spec: _options.Spec, logits: ArrayLike, target: ArrayLike, common: _Common) -> jax.Array:
    """A loss of checked options: the paths of every single-label and multilabel loss."""
    backend, reduction = common.backend, common.reduction
    k, gamma, alpha = spec.k, spec.gamma, spec.alpha
    objective = spec.objective
    x, z, axis = _class_last(logits, common.axis)
    classes = z.shape[-1]
    _options.check_classes(k, classes)
    if z.size == 0:
        _targets(target, x, axis, spec)
        return _empty_loss(z, reduction)
    # Top-k entropy at k = 1 is the entropy of its one target.
    entropy = objective == "cross_entropy" or (objective == "topk_cross_entropy" and k == 1)
    canonical = common.precision
    _check_precision(x, canonical, True)
    canonical = _default_precision(canonical, objective, x.dtype, backend, k, classes, z.size)
    work = canonical if entropy else _work_precision(objective, canonical, x)
    tensor = _jax_backend(
        x, "cross_entropy" if entropy else objective, backend, work, gamma, alpha, k, classes
    )
    if _concrete(x) and not bool(jnp.isfinite(x).all()):
        raise ValueError("logits must be finite floating-point values")
    target, width = _targets(target, x, axis, spec)
    if backend == "auto":
        fused = _jax_cuda.loss(
            z,
            target,
            objective,
            int(k),
            float(gamma),
            float(alpha),
            reduction,
            canonical,
            _reduced_entropy,
            (canonical == "float64", reduction),
        )
        if fused is not None:
            return fused
    if tensor and entropy:
        values = _entropy(z, target, canonical == "float64")
    else:
        dtype = z.dtype if z.dtype in (jnp.float32, jnp.float64) else jnp.float32
        # The native library takes the given precision (float64 for None, as fast).
        work = work if tensor else canonical
        options = (not tensor, objective, int(k), float(gamma), float(alpha), work, width)
        values = _rows(z.astype(dtype), target, options)
    # Loss values are promote(logits dtype, float32), as libsdca.torch.
    return _reduce(values, reduction, _work(z).dtype)


def _weight_array(weight: ArrayLike, classes: int) -> tuple[jax.Array, jax.Array | None]:
    """Checked class weights (classes,) and, for traced ones, whether they are valid (finite
    and nonnegative; invalid traced weights give NaN losses)."""
    w = jnp.asarray(weight)
    if not jnp.issubdtype(w.dtype, jnp.floating):
        raise TypeError(f"weight must be a floating-point array, not {w.dtype}")
    if w.shape != (classes,):
        raise ValueError(f"weight must have shape (classes,) = ({classes},)")
    if _concrete(weight):
        values = np.asarray(weight)
        if not (np.isfinite(values) & (values >= 0)).all():
            raise ValueError("weight must be finite and nonnegative")
        return w, None
    return w, (jnp.isfinite(w) & (w >= 0)).all()


def _smoothing(label_smoothing: object) -> float | jax.Array:
    """label_smoothing in [0, 1]: a checked Python number, or a traced scalar (unchecked)."""
    if isinstance(label_smoothing, jax.core.Tracer):
        if np.ndim(label_smoothing) != 0 or not jnp.issubdtype(label_smoothing.dtype, jnp.floating):
            raise TypeError("label_smoothing must be a real scalar")
        return label_smoothing
    return _options.check_label_smoothing(_static(label_smoothing, "label_smoothing"))


def _smoothed(label_smoothing: float | jax.Array) -> bool:
    return not (isinstance(label_smoothing, float) and label_smoothing == 0.0)


def _composed(
    spec: _options.Spec,
    logits: ArrayLike,
    target: ArrayLike,
    weight: ArrayLike | None,
    ignore_index: int | None,
    label_smoothing: float | jax.Array,
    common: _Common,
) -> jax.Array:
    """A single-label loss with class weights, an ignored label or (cross_entropy) label
    smoothing, as torch.nn.functional.cross_entropy: per-example losses times the weight of
    their label (0 where ignored), plus for smoothing eps / C times sum_j w_j (lse - z_j); the
    mean divides by the sum of the label weights."""
    x, z, axis = _class_last(logits, common.axis)
    classes = z.shape[-1]
    labels = _label_array(target, tuple(np.shape(x)), axis, spec)
    w, valid = (None, None) if weight is None else _weight_array(weight, classes)
    ignored = None if ignore_index is None else labels == ignore_index
    if ignored is not None:
        # Ignored rows may hold any values: they enter as zero logits of label 0.
        z = jnp.where(ignored[..., None], jnp.zeros((), z.dtype), z)
        labels = jnp.where(ignored, jnp.zeros((), labels.dtype), labels)
    values = _loss(spec, z, labels, common._replace(reduction="none", axis=-1))
    dtype = values.dtype
    if w is None:
        rows = jnp.ones(labels.shape, dtype)
    else:
        rows = w.astype(dtype)[jnp.clip(labels, 0, classes - 1)]
    if ignored is not None:
        rows = jnp.where(ignored, jnp.zeros((), dtype), rows)
    loss = values * rows
    if _smoothed(label_smoothing):
        eps = jnp.asarray(label_smoothing, dtype)
        log_p = jax.nn.log_softmax(_work(z, common.precision == "float64"), -1)
        terms = -(log_p if w is None else log_p * w.astype(log_p.dtype)).sum(-1).astype(dtype)
        if ignored is not None:
            terms = jnp.where(ignored, jnp.zeros((), dtype), terms)
        loss = (1 - eps) * loss + eps / classes * terms
    if valid is not None:
        loss = jnp.where(valid, loss, jnp.nan)
    if common.reduction == "none":
        return loss
    if common.reduction == "sum":
        return loss.sum()
    return loss.sum() / rows.sum()


def _probabilities(
    logits: ArrayLike,
    target: jax.Array,
    weight: ArrayLike | None,
    ignore_index: int | None,
    label_smoothing: float | jax.Array,
    common: _Common,
) -> jax.Array:
    """cross_entropy of class probabilities, as torch.nn.functional.cross_entropy:
    -sum_j w_j t_j log softmax(z)_j per example, t smoothed to (1 - eps) t + eps / C; the mean
    divides by the number of examples."""
    if ignore_index is not None:
        raise ValueError("ignore_index is not supported with probability targets")
    if common.backend == "native":
        raise NotImplementedError(
            "probability targets are computed with jax.numpy; use backend='auto' or 'jax'"
        )
    x, z, axis = _class_last(logits, common.axis)
    if target.shape != x.shape:
        raise ValueError(f"probability targets must have the logits' shape {x.shape}")
    _check_precision(x, common.precision, True)
    if _concrete(x) and not bool(jnp.isfinite(x).all()):
        raise ValueError("logits must be finite floating-point values")
    classes = z.shape[-1]
    log_p = jax.nn.log_softmax(_work(z, common.precision == "float64"), -1)
    t = jnp.moveaxis(target, axis, -1).astype(log_p.dtype)
    if _smoothed(label_smoothing):
        eps = jnp.asarray(label_smoothing, log_p.dtype)
        t = t * (1 - eps) + eps / classes
    terms = log_p * t
    if weight is not None:
        w, valid = _weight_array(weight, classes)
        terms = terms * w.astype(log_p.dtype)
        if valid is not None:
            terms = jnp.where(valid, terms, jnp.nan)
    return _reduce(-terms.sum(-1), common.reduction, _work(z).dtype)


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
    fenced: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Centered scaled logits s, t and offset-corrected scaled differences d for pass step
    (fenced: _fences)."""
    step = step if fenced else 0
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
    fenced = _fences(student, "distillation_value")
    cs = _distillation_work(student, temperature, double, 1 if fenced else 0)
    ct = _distillation_work(teacher, temperature, double, 1 if fenced else 0)
    s = _scaled_difference(cs, student_max, temperature) + cs * 0
    t = _scaled_difference(ct, teacher_max, temperature) + ct * 0
    zero = jnp.zeros_like(student_max)
    et = jnp.exp(t)
    _, rest, mass = _rest(s, zero, False, (et,), excluded=first)
    lognorm = jnp.log1p(rest)
    if fenced:
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
    dtype = _far_dtype(s.dtype)
    if dtype is not None:
        far = rest < student.shape[-1] * _FAR_SUM

        def rescan() -> jax.Array:
            cs = _distillation_work(student, temperature, double)
            ct = _distillation_work(teacher, temperature, double)
            s = _scaled_difference(cs, student_max, temperature)
            t = _scaled_difference(ct, teacher_max, temperature)
            return _distillation_rescan(s, t, _top(s, first), mass, dtype)[1]

        tail = _only_if(far.any(), rescan, zero.astype(dtype))
        value = jnp.where(far[..., 0], tail[..., 0].astype(value.dtype), value)
    valid = (jnp.isfinite(rest) & jnp.isfinite(mass))[..., 0]
    return jnp.where(valid, value, jnp.nan)


def _top(s: jax.Array, first: jax.Array | None) -> jax.Array:
    """The student's maximum that its mass leaves out (of the normalizer index, or every tied
    one)."""
    if first is None:
        return s == 0
    return jnp.arange(s.shape[-1], dtype=first.dtype) == first


def _distillation_rescan(
    s: jax.Array,
    t: jax.Array,
    top: jax.Array,
    mass: jax.Array,
    dtype: jax.typing.DTypeLike,
    logratio: jax.Array | None = None,
    near: jax.Array | None = None,
    den: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """Row sums of a far row (_rescanned), from centered scaled logits s and t and the
    teacher's mass: the student's mass but top, the loss (sum p (log1p(rest) - s), log1p(rest)
    being rest) and, given the log-ratios with the rows' near flags and student normalizer
    den, the sum of u but top, each from the exponentials of its own gaps."""
    es, far_s = _shifted(s)
    et, far_t = _shifted(t)
    p = et / mass
    pairs = [(jnp.where(top, 0.0, es), far_s), (jnp.where(p > 0, p * -s, 0.0), far_t)]
    if logratio is not None and near is not None and den is not None:
        u = jnp.where(top | ~near, 0.0, p * jnp.expm1(jnp.clip(logratio, -4, 4)))
        other = top | near
        pairs += [
            (u, far_t),
            (jnp.where(other, 0.0, es / den), far_s),
            (jnp.where(other, 0.0, p), far_t),
        ]
    sums = _rescanned(tuple(pairs), dtype)
    rest, value = sums[0], sums[0] + sums[1]
    if len(sums) == 2:
        return rest, value
    return rest, value, sums[2] + sums[3] - sums[4]


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _distillation(
    student: jax.Array, teacher: jax.Array, temperature: float, double: bool
) -> jax.Array:
    return _distillation_value(student, teacher, temperature, double)


def _distillation_stats(
    student: jax.Array, teacher: jax.Array, temperature: float, double: bool, exact_log_p: bool
) -> tuple[Any, ...]:
    """Value, log p, log q, p, q, the log-ratio log(q / p), q - p, the row sum of u (u itself
    if exact_log_p), the valid rows, as libsdca.torch, and the far rows of float32 work
    (_far_dtype; else None) with their student maxima, those maxima's gradient entries times T
    and their row sums of u. log p is accurate below eps only if exact_log_p (proximal
    distillation), which costs a pass for the teacher's ties."""
    # Center before scaling: a small temperature would overflow the logits.
    student_max, teacher_max = _distillation_max(student, teacher, temperature, double)
    first = _normalizer_index(student)
    offset = _jax_losses.difference(student_max, teacher_max)
    # The largest centered logits are exactly 0, and their exponentials 1.
    zero = jnp.zeros_like(student_max)
    fenced = _fences(student, "proximal_distillation" if exact_log_p else "distillation")
    s1, t1, _ = _distillation_pass(
        student, teacher, student_max, teacher_max, offset, temperature, double, 1, fenced
    )
    _, t2, d2 = _distillation_pass(
        student, teacher, student_max, teacher_max, offset, temperature, double, 2, fenced
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
        student, teacher, student_max, teacher_max, offset, temperature, double, 3, fenced
    )
    p3 = jnp.exp(t3) / mass
    val_term = jnp.where(p3 > 0, p3 * (lognorm - s3), 0.0)
    # Select Barron's fallback per class, so a distant negligible class does
    # not erase small probability differences among the dominant classes.
    # exp(log(p) + d) = exp(s + shift), using student logits to prevent
    # cancellation between a huge teacher tail and d, or expm1(d) overflow.
    s4, t4, d4 = _distillation_pass(
        student, teacher, student_max, teacher_max, offset, temperature, double, 4, fenced
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
            student, teacher, student_max, teacher_max, offset, temperature, double, step, fenced
        )
        p, q = jnp.exp(t) / mass, jnp.exp(s) / (1 + rest)
        logratio = (d - center) - log1p_corr
        u = jnp.where(jnp.abs(logratio) < 1, p * jnp.expm1(jnp.clip(logratio, -1, 1)), q - p)
        return t - teacher_lognorm, s - lognorm, p, q, logratio, u

    log_p, log_q, p, q, logratio, u = terms_of(5)
    u_sum = u if exact_log_p else _jax_losses.row_sums((u,), ())[0]
    far_rows = None
    dtype = _far_dtype(s1.dtype)
    if dtype is not None:
        # As _distillation_gpu: a far row's loss, and its maximum's entry from the others'.
        far = valid & (rest < student.shape[-1] * _FAR_SUM)
        top = _top(s1, first)

        def rescan() -> tuple[jax.Array, ...]:
            s, t, d = _distillation_pass(
                student, teacher, student_max, teacher_max, offset, temperature, double, 0
            )
            logratio = (d - center) - log1p_corr
            near = jnp.abs(logratio) < 1
            sums = _distillation_rescan(s, t, _top(s, first), mass, dtype, logratio, near, 1 + rest)
            row_u = _jax_losses.row_sums((u,), ())[0] if exact_log_p else u_sum
            return *sums, row_u

        row_u = jnp.zeros_like(rest) if exact_log_p else u_sum
        default = (rest.astype(dtype), value.astype(dtype), value.astype(dtype), row_u)
        tail, total, others, row_u = _only_if(far.any(), rescan, default)
        value = jnp.where(far, total.astype(value.dtype), value)
        far_rows = (far, top, row_u.astype(dtype) * tail - others, row_u)
    if fenced and not exact_log_p:
        _, _, p, q, logratio, u = terms_of(6)
    return value[..., 0], log_p, log_q, p, q, logratio, u, valid[..., 0], u_sum, far_rows


def _distillation_gpu(
    student: jax.Array, teacher: jax.Array, temperature: float, double: bool
) -> tuple[Any, ...]:
    """Plain distillation on the GPU: raw difference bounds share the maxima pass (fenced from
    the normalizer pass), so rows with |d - center| <= 2 skip the correction pass via _only_if
    and fuse value with u_sum."""
    student_max, teacher_max = _distillation_max(student, teacher, temperature, double)
    first = _normalizer_index(student)
    raw = _distillation_work(lax.stop_gradient(student), temperature, double) - _distillation_work(
        lax.stop_gradient(teacher), temperature, double
    )
    raw_min = _optimization_barrier(raw.min(-1, keepdims=True))
    raw_max = _optimization_barrier(raw.max(-1, keepdims=True))
    s_fenced = _optimization_barrier(student)
    t_fenced = _optimization_barrier(teacher)
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

    shift = jnp.where(bounded, 0.0, _only_if(~bounded.all(), corrected, zero))
    logratio = dn - shift
    near = bounded | (jnp.abs(logratio) < 1)
    u = jnp.where(near, p * jnp.expm1(jnp.clip(logratio, -4, 4)), q - p)
    val_term = jnp.where(p > 0, p * (jnp.log1p(rest) - s), 0.0)
    value, u_sum = _jax_losses.row_sums((val_term, u), ())
    dtype = _far_dtype(s.dtype)
    if dtype is None:
        return value[..., 0], p, q, u, valid[..., 0], u_sum, None
    # A far row's loss, and its maximum's entry from the others' (which sum to zero):
    # -sum(u but top) + u_sum (1 - q_top), where 1 - q_top = rest / (1 + rest) is rest.
    far = valid & (rest < student.shape[-1] * _FAR_SUM)

    def rescan() -> tuple[jax.Array, ...]:
        s, t, d = _distillation_pass(
            student, teacher, student_max, teacher_max, offset, temperature, double, 3
        )
        logratio = (d - center) - shift
        near = bounded | (jnp.abs(logratio) < 1)
        return _distillation_rescan(s, t, _top(s, first), mass, dtype, logratio, near, 1 + rest)

    default = (rest.astype(dtype), value.astype(dtype), u_sum.astype(dtype))
    tail, total, others = _only_if(far.any(), rescan, default)
    value = jnp.where(far, total.astype(value.dtype), value)
    pivot = u_sum.astype(dtype) * tail - others
    return value[..., 0], p, q, u, valid[..., 0], u_sum, (far, _top(s, first), pivot)


def _first_maximum(x: jax.Array) -> jax.Array:
    """Where the first maximum of each row is (argmax)."""
    return jnp.arange(x.shape[-1]) == _jax_losses.row_first_max(x)[1]


def _distillation_fwd(
    student: jax.Array, teacher: jax.Array, temperature: float, double: bool
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    tail = None
    if jax.default_backend() == "gpu":
        value, p, q, u, valid, u_sum, tail = _distillation_gpu(
            student, teacher, temperature, double
        )
    else:
        value, _, _, p, q, _, u, valid, u_sum, tail = _distillation_stats(
            student, teacher, temperature, double, False
        )
    gradient = u - q * u_sum
    if student.shape[-1] <= 16:
        pivot = _first_maximum(p + q)
        others = jnp.where(pivot, 0.0, gradient)
        gradient = jnp.where(pivot, -_jax_losses.row_sums((others,), ())[0], others)
    gradient = _jax_losses.over(gradient, temperature)
    if tail is not None:
        far, top, entry = tail[:3]
        entry = _jax_losses.over(entry, temperature).astype(gradient.dtype)
        gradient = jnp.where(far & top, entry, gradient)
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
    *,
    temperature: float = 1.0,
    reduction: Reduction = "mean",
    axis: int = -1,
    backend: Backend = "auto",
    precision: object = None,
) -> jax.Array:
    """Cross-entropy with a fixed teacher distribution softmax(teacher/T).

    Both arguments are LOGITS. The teacher gets no gradient. No T² multiplier
    is applied. The first-order gradient is exactly zero at equal represented
    logits; near equality it avoids probability subtraction.

    Args:
        student: Finite floating-point logits with the class axis
            (at least two classes).
        teacher: Finite floating-point logits of the same shape; a different dtype is
            promoted with the student's.
        temperature: T, a static number, finite and positive.
        reduction: "none", "sum" or "mean".
        axis: Class axis of the logits.
        backend: "auto" (the fused CUDA kernels on NVIDIA GPUs, the native CPU kernels
            through the FFI handlers, else jax.numpy), "native" (the native CPU kernels;
            NotImplementedError where they do not run) or "jax" (jax.numpy).
        precision: None, "float32" or "float64" (or jnp.float32/float64); float64
            needs jax_enable_x64.

    Returns:
        Per-example losses (student.shape without axis) or their reduction.

    Raises:
        TypeError, ValueError: For invalid inputs or options.
    """
    temperature = _options.check_positive(_static(temperature, "temperature"), "temperature")
    common = _common(reduction, axis, backend, precision)
    s, t, double = _distillation_pair(student, teacher, temperature, common)
    if s.size == 0:
        return _empty_loss(s, common.reduction)
    if common.backend == "auto":
        fused = _jax_cuda.distillation(
            s,
            t,
            temperature,
            double,
            common.reduction,
            _distillation_reference,
            _distillation_reference_gradient,
        )
        if fused is not None:
            return fused
    if common.backend != "jax":
        fused = _jax_cpu.distillation(
            s,
            t,
            temperature,
            double,
            common.reduction,
            _distillation_reference,
            _distillation_reference_gradient,
        )
        if fused is not None:
            return fused
        if common.backend == "native":
            raise NotImplementedError(_native_distillation_absent(s))
    values = _distillation(s, t, temperature, double)
    return _reduce(values, common.reduction, _work(s).dtype)


def _native_distillation_absent(s: jax.Array) -> str:
    """Why backend="native" distillation does not run."""
    if _jax_cuda.on_gpu(s):
        why = "the logits are on a GPU"
    elif not _jax_ffi.enabled():
        why = f"the JAX FFI handlers are unavailable ({_jax_ffi.why_not()})"
    else:
        why = "the temperature or dtype is outside the native kernels' range"
    return f"native distillation runs on the CPU through the FFI handlers, but {why}"


def _distillation_reference(
    student: jax.Array, teacher: jax.Array, temperature: float, double: bool, reduction: str
) -> jax.Array:
    """The reduced loss of _distillation_fwd, whose derivatives are the fused kernels' higher
    ones (_jax_cuda; without the custom VJP, which forward mode cannot differentiate)."""
    value = _distillation_fwd(student, teacher, temperature, double)[0]
    return _reduce(value, reduction, _work(student).dtype)


def _distillation_reference_gradient(
    student: jax.Array,
    teacher: jax.Array,
    ct: jax.Array,
    temperature: float,
    double: bool,
    reduction: str,
) -> jax.Array:
    """_rounded_bwd of _distillation_fwd's gradient for the reduced loss's cotangent ct."""
    _, residuals = _distillation_fwd(student, teacher, temperature, double)
    rows = jnp.zeros(student.shape[:-1], residuals[0].dtype)
    upstream = jax.vjp(lambda v: _reduce(v, reduction, _work(student).dtype), rows)[1](ct)[0]
    return _rounded_bwd(residuals, upstream)[0]


def _distillation_pair(
    student: ArrayLike, teacher: ArrayLike, temperature: float, common: _Common
) -> tuple[jax.Array, jax.Array, bool]:
    """Checked logits with the class axis last (teacher without gradient; a different dtype
    is promoted), and whether the work is float64."""
    s, t = jnp.asarray(student), jnp.asarray(teacher)
    for name, x in (("student", s), ("teacher", t)):
        if not jnp.issubdtype(x.dtype, jnp.floating):
            raise TypeError(f"{name} logits must be a floating-point array, not {x.dtype}")
    if s.shape != t.shape or s.ndim < 1:
        raise ValueError("Student and teacher must have matching shapes with a class axis")
    if s.dtype != t.dtype:
        dtype = jnp.promote_types(s.dtype, t.dtype)
        s, t = s.astype(dtype), t.astype(dtype)
    if _concrete(s, t) and not bool(jnp.isfinite(s).all() & jnp.isfinite(t).all()):
        raise ValueError("Logits must be finite floating-point arrays")
    _check_precision(s, common.precision, True)
    _, s, axis = _class_last(s, common.axis)
    t = jnp.moveaxis(jax.lax.stop_gradient(t), axis, -1)
    return s, t, common.precision == "float64"


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _proximal_distillation(
    student: jax.Array, teacher: jax.Array, temperature: float, alpha: float, double: bool
) -> jax.Array:
    return _proximal_distillation_fwd(student, teacher, temperature, alpha, double)[0]


def _proximal_distillation_fwd(
    student: jax.Array, teacher: jax.Array, temperature: float, alpha: float, double: bool
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    distilled, *stats, valid, _, far_rows = _distillation_stats(
        student, teacher, temperature, double, True
    )
    if _accelerator(student):
        # XLA:GPU compiles for minutes when it fuses these statistics into the solve.
        stats = _optimization_barrier(stats)
    log_p, log_q, p, q, logratio, u = stats
    value, gradient = _jax_losses.proximal_distillation(
        log_p, log_q, p, q, logratio, u, alpha, ~valid[..., None]
    )
    # A concentrated student alone does not bound the proximal correction: the teacher
    # may prefer another class. Require the cross-entropy itself to be tiny as well,
    # which bounds the teacher mass away from the student maximum.
    if far_rows is not None and alpha * student.shape[-1] * _FAR_SUM <= 2.0**-30:
        far, top, entry, u_sum = far_rows
        far = far & (distilled[..., None] < student.shape[-1] * _FAR_SUM)
        distilled_gradient = jnp.where(top, entry.astype(u.dtype), u - q * u_sum.astype(u.dtype))
        value = jnp.where(far[..., 0], distilled, value)
        gradient = jnp.where(far, distilled_gradient, gradient)
    value = jnp.where(valid, value, jnp.nan)
    gradient = jnp.where(valid[..., None], _jax_losses.over(gradient, temperature), jnp.nan)
    return value, (gradient, jnp.zeros((0,), student.dtype))


_proximal_distillation.defvjp(_proximal_distillation_fwd, _rounded_bwd)


def proximal_distillation_loss(
    student: ArrayLike,
    teacher: ArrayLike,
    *,
    temperature: float = 1.0,
    alpha: float = 1.0,
    reduction: Reduction = "mean",
    axis: int = -1,
    backend: Backend = "auto",
    precision: object = None,
) -> jax.Array:
    """Moreau envelope of distillation_loss with step alpha, as in libsdca.torch.

    min_u { H(softmax(teacher/T), softmax(u)) + |u - student/T|^2 / (2 alpha) }; reverse
    mode and jax.hessian, exactly zero gradient at equal represented logits. alpha (a static
    number, finite and positive) outside the float32 range computes in float64 (needs
    jax_enable_x64). Other arguments, the result and errors as in distillation_loss.
    """
    temperature = _options.check_positive(_static(temperature, "temperature"), "temperature")
    alpha = _options.check_positive(_static(alpha, "alpha"), "alpha")
    common = _common(reduction, axis, backend, precision)
    s, t, double = _distillation_pair(student, teacher, temperature, common)
    if not _float32_holds(alpha):
        if not _x64():
            raise ValueError("alpha is outside the float32 range; enable jax_enable_x64")
        double = True
    if s.size == 0:
        return _empty_loss(s, common.reduction)
    if common.backend == "auto":
        fused = _jax_cuda.proximal_distillation(
            s, t, temperature, alpha, double, common.reduction, _proximal_distillation_rows
        )
        if fused is not None:
            return fused
    # Keep explicit fp32 on the array implementation until the native solver
    # exposes selectable work precision. The native default is measured separately.
    if common.backend == "native" or (
        common.backend == "auto" and (common.precision != "float32" or double)
    ):
        values = _jax_cpu.proximal_distillation(
            s, t, temperature, alpha, double, _proximal_distillation_rows
        )
        if values is not None:
            return _reduce(values, common.reduction, _work(s).dtype)
        if common.backend == "native":
            raise NotImplementedError(_native_distillation_absent(s))
    values = _proximal_distillation(s, t, temperature, alpha, double)
    return _reduce(values, common.reduction, _work(s).dtype)


def _proximal_distillation_rows(
    student: jax.Array, teacher: jax.Array, temperature: float, alpha: float, double: bool
) -> tuple[jax.Array, jax.Array]:
    """Values and gradient of _proximal_distillation, whose derivatives are the fused kernel's
    higher ones (_jax_cuda)."""
    value, (gradient, _) = _proximal_distillation_fwd(student, teacher, temperature, alpha, double)
    return value, gradient


_CROSS_ENTROPY = _options.spec("cross_entropy")
_MULTILABEL_CROSS_ENTROPY = _options.spec("multilabel_cross_entropy")


def _single_label(
    spec: _options.Spec,
    logits: ArrayLike,
    target: ArrayLike,
    weight: ArrayLike | None,
    ignore_index: object,
    common: _Common,
) -> jax.Array:
    """A single-label loss: the loss itself, or composed with class weights or an ignored
    label."""
    ignore = _options.check_ignore_index(ignore_index)
    if weight is None and ignore is None:
        return _loss(spec, logits, target, common)
    return _composed(spec, logits, target, weight, ignore, 0.0, common)


def cross_entropy(
    logits: ArrayLike,
    target: ArrayLike,
    *,
    weight: ArrayLike | None = None,
    ignore_index: int | None = None,
    label_smoothing: float = 0.0,
    reduction: Reduction = "mean",
    axis: int = -1,
    backend: Backend = "auto",
    precision: object = None,
) -> jax.Array:
    """Softmax cross-entropy of class labels or class probabilities, as
    torch.nn.functional.cross_entropy (with the class axis axis).

    Args:
        logits: Finite floating-point scores with the class axis (at least two classes).
        target: Integer class labels of logits.shape without axis (any integer dtype), or
            floating-point class probabilities of logits.shape. Boolean masks are the
            targets of multilabel_cross_entropy (TypeError here).
        weight: Class weights (classes,), finite and nonnegative, or None. Each example's
            loss is multiplied by its label's weight (probabilities: sum_j w_j t_j ...);
            the mean of labels divides by the sum of those weights, as PyTorch.
        ignore_index: A label whose examples add nothing (zero loss and gradient; excluded
            from the mean), or None. Not with probability targets (ValueError).
        label_smoothing: eps in [0, 1]: the target mixes (1 - eps) of the label (or the
            probabilities) with eps of the uniform distribution. May be traced.
        reduction: "none", "sum" or "mean".
        axis: Class axis of logits.
        backend: "auto" (jax.numpy, or the fused CUDA kernels on NVIDIA GPUs), "native"
            (the native library on the host; labels only) or "jax" (jax.numpy).
        precision: None, "float32" or "float64" (or jnp.float32/float64); float64 needs
            jax_enable_x64.

    Returns:
        Per-example losses (logits.shape without axis) or their reduction, in float32 for
        16-bit logits.

    Raises:
        TypeError, ValueError: For invalid inputs or options, and for concrete invalid
            values (nonfinite logits, labels out of range). Traced invalid values give NaN.
        NotImplementedError: For probability targets with backend="native".
    """
    common = _common(reduction, axis, backend, precision)
    ignore = _options.check_ignore_index(ignore_index)
    eps = _smoothing(label_smoothing)
    dtype = getattr(target, "dtype", None)
    if dtype is None:
        target = jnp.asarray(target)
        dtype = target.dtype
    if jnp.issubdtype(dtype, jnp.floating):
        return _probabilities(logits, jnp.asarray(target), weight, ignore, eps, common)
    if weight is None and ignore is None and not _smoothed(eps):
        return _loss(_CROSS_ENTROPY, logits, target, common)
    return _composed(_CROSS_ENTROPY, logits, target, weight, ignore, eps, common)


def topk_cross_entropy(
    logits: ArrayLike,
    target: ArrayLike,
    *,
    k: int,
    weight: ArrayLike | None = None,
    ignore_index: int | None = None,
    reduction: Reduction = "mean",
    axis: int = -1,
    backend: Backend = "auto",
    precision: object = None,
) -> jax.Array:
    """Top-k cross-entropy (the top-k entropy loss) of class labels; cross_entropy at k = 1.

    k: a static integer in [1, classes - 1]. weight and ignore_index as in cross_entropy;
    other arguments, the result and errors as in cross_entropy (labels only).
    """
    k = _options.check_k(_static(k, "k"))
    spec = _options.spec("topk_cross_entropy", k=k)
    common = _common(reduction, axis, backend, precision)
    return _single_label(spec, logits, target, weight, ignore_index, common)


def multiclass_hinge_loss(
    logits: ArrayLike,
    target: ArrayLike,
    *,
    gamma: float = 0.0,
    weight: ArrayLike | None = None,
    ignore_index: int | None = None,
    reduction: Reduction = "mean",
    axis: int = -1,
    backend: Backend = "auto",
    precision: object = None,
) -> jax.Array:
    """Multiclass hinge loss of Crammer and Singer, smoothed by gamma > 0 (static, finite).

    Other arguments, the result and errors as in topk_cross_entropy.
    """
    gamma = _options.check_gamma(_static(gamma, "gamma"))
    spec = _options.spec("multiclass_hinge", gamma=gamma)
    common = _common(reduction, axis, backend, precision)
    return _single_label(spec, logits, target, weight, ignore_index, common)


def topk_hinge_loss(
    logits: ArrayLike,
    target: ArrayLike,
    *,
    k: int,
    variant: Literal["alpha", "beta"] = "beta",
    gamma: float = 0.0,
    weight: ArrayLike | None = None,
    ignore_index: int | None = None,
    reduction: Reduction = "mean",
    axis: int = -1,
    backend: Backend = "auto",
    precision: object = None,
) -> jax.Array:
    """Top-k hinge loss, smoothed by gamma > 0 (static, finite). Variant "beta" averages the
    k largest hinges; "alpha" is the hinge of the average of the k largest margins.

    k: a static integer in [1, classes - 1]. Other arguments, the result and errors as in
    topk_cross_entropy.
    """
    k = _options.check_k(_static(k, "k"))
    name = "topk_hinge_" + _options.check_variant(variant)
    gamma = _options.check_gamma(_static(gamma, "gamma"))
    spec = _options.spec(name, k=k, gamma=gamma)
    common = _common(reduction, axis, backend, precision)
    return _single_label(spec, logits, target, weight, ignore_index, common)


def proximal_cross_entropy(
    logits: ArrayLike,
    target: ArrayLike,
    *,
    alpha: float = 1.0,
    weight: ArrayLike | None = None,
    ignore_index: int | None = None,
    reduction: Reduction = "mean",
    axis: int = -1,
    backend: Backend = "auto",
    precision: object = None,
) -> jax.Array:
    """Moreau envelope of the cross-entropy with step alpha (static, finite, positive).

    Other arguments, the result and errors as in topk_cross_entropy.
    """
    alpha = _options.check_positive(_static(alpha, "alpha"), "alpha")
    spec = _options.spec("proximal_cross_entropy", alpha=alpha)
    common = _common(reduction, axis, backend, precision)
    return _single_label(spec, logits, target, weight, ignore_index, common)


def multilabel_cross_entropy(
    logits: ArrayLike,
    target: ArrayLike,
    *,
    reduction: Reduction = "mean",
    axis: int = -1,
    backend: Backend = "auto",
    precision: object = None,
) -> jax.Array:
    """Multilabel cross-entropy: uniform target probabilities on the positive labels.

    target: a boolean mask of logits.shape with a positive label per example. Other
    arguments, the result and errors as in cross_entropy.
    """
    common = _common(reduction, axis, backend, precision)
    return _loss(_MULTILABEL_CROSS_ENTROPY, logits, target, common)


def multilabel_hinge_loss(
    logits: ArrayLike,
    target: ArrayLike,
    *,
    gamma: float = 0.0,
    reduction: Reduction = "mean",
    axis: int = -1,
    backend: Backend = "auto",
    precision: object = None,
) -> jax.Array:
    """Multilabel hinge loss, smoothed by gamma > 0 (static, finite).

    target: a boolean mask of logits.shape with a positive and a negative label per
    example. Other arguments, the result and errors as in cross_entropy.
    """
    gamma = _options.check_gamma(_static(gamma, "gamma"))
    spec = _options.spec("multilabel_hinge", gamma=gamma)
    common = _common(reduction, axis, backend, precision)
    return _loss(spec, logits, target, common)
