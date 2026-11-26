"""JAX losses under jit, grad and vmap: the cross-entropies, top-k and hinge losses of class
labels, the multilabel losses of boolean masks, and distillation_loss and
proximal_distillation_loss of teacher logits.

backend="auto" (and its synonym "kernel") runs libsdca's kernels through one XLA FFI target
(the CPU handler built on first use, the Triton kernels on CUDA; without the CPU handler the
same kernels of the native library through a host callback, _jax_native) in a custom_vjp
(_jax_ops), or raises NotImplementedError naming backend="composed", which runs the
composed jax.numpy losses (_composed) on any device. Invalid rows give NaN losses and
gradients, without an error.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from . import _frontend as _f
from . import _jax_ops as _ops
from ._composed import loss as _composed_loss
from ._composed import xp_jax
from ._composed._losses import WIDE_DISTILLATION_CLASSES

__all__ = [
    "cross_entropy",
    "distillation_loss",
    "explain_backend",
    "multiclass_hinge_loss",
    "multilabel_cross_entropy",
    "multilabel_hinge_loss",
    "proximal_cross_entropy",
    "proximal_distillation_loss",
    "topk_cross_entropy",
    "topk_hinge_loss",
]

Reduction = Literal["none", "sum", "mean"]
Backend = Literal["auto", "kernel", "composed"]


def _static(value: object, name: str) -> object:
    """A hyperparameter as a Python number: concrete scalar arrays convert, traced ones raise."""
    if isinstance(value, jax.core.Tracer):
        raise TypeError(
            f"{name} must be a static number (close over it or mark it static under jit), "
            "not a traced value"
        )
    if isinstance(value, np.ndarray | np.generic | jax.Array) and np.ndim(value) == 0:
        return np.asarray(value).item()
    return value


_PRECISIONS: dict[Any, str] = {
    jnp.float32: "float32",
    jnp.float64: "float64",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
}
JAX = _f.Framework("axis", "axis", _PRECISIONS, "jnp", _static)

# explain_backend's record of the routes a call takes.
_TAKEN: contextvars.ContextVar[list[tuple[str, str]] | None] = contextvars.ContextVar(
    "libsdca_jax_taken", default=None
)


def _kind(dtype: Any) -> str:
    if dtype == jnp.bool_:
        return "b"
    if jnp.issubdtype(dtype, jnp.floating):
        return "f"
    return "c" if jnp.issubdtype(dtype, jnp.complexfloating) else "i"


def _x64() -> bool:
    return bool(jax.dtypes.canonicalize_dtype(jnp.float64) == jnp.float64)


def _asarray(value: ArrayLike) -> jax.Array:
    """Keep a JAX array's buffer when its dtype already obeys the active x64 setting."""
    if (
        isinstance(value, jax.Array)
        and jax.dtypes.canonicalize_dtype(value.dtype, allow_extended_dtype=True) == value.dtype
    ):
        return value
    return jnp.asarray(value)


def _prepare(
    call: _f.Call, logits: ArrayLike, target: ArrayLike, weight: ArrayLike | None
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Checked class-last arrays; TypeError for wrong types, ValueError for bad shapes."""
    z, t = _asarray(logits), _asarray(target)
    _f.check_logits(call, _kind(z.dtype), z.dtype)
    axis = _f.class_axis(JAX, z.ndim, call.axis)
    role = _f.target_role(JAX, call, _kind(t.dtype), t.dtype, t.shape, z.shape, axis)
    if role == "labels":
        t = _ops.labels(t)
    else:
        if role == "teacher" and t.dtype != z.dtype:
            common = jnp.promote_types(z.dtype, t.dtype)
            z, t = z.astype(common), t.astype(common)
        elif role == "probabilities" and _f.probability_dtype(t.dtype, z.dtype):
            t = t.astype(jnp.float32)
        t = jnp.moveaxis(t, axis, -1)
    z = jnp.moveaxis(z, axis, -1)
    _f.check_classes(call.k, z.shape[-1])
    _f.check_precision(z.dtype, call.precision)
    w = None
    if weight is not None:
        w = jnp.asarray(weight)
        _f.check_weight(_kind(w.dtype), w.dtype, w.shape, z.shape[-1])
        w = _f.nan_weights(xp_jax, w)
    return z, t, w


def _reduce(values: jax.Array, reduction: str, total: Any) -> jax.Array:
    if reduction == "none":
        return values
    value = values.sum()
    return value / total if reduction == "mean" else value


# Keep a single dispatch for the wide float64 composed calculation in eager mode.
# JAX can inline the same expression under jit, vmap and higher-order transforms.
_compiled_composed = jax.jit(
    _composed_loss, static_argnames=("name", "temperature", "reduction", "precision"), inline=True
)


def _stage_composed(call: _f.Call, z: jax.Array, work: str) -> bool:
    if work != "float64":
        return False
    if call.name == "distillation":
        return z.shape[-1] >= WIDE_DISTILLATION_CLASSES
    return (
        call.name == "multilabel_cross_entropy"
        and z.dtype == jnp.bfloat16
        and _ops.platform(z) == "cpu"
    )


def _evaluate(
    call: _f.Call,
    logits: ArrayLike,
    target: ArrayLike,
    weight: ArrayLike | None = None,
    label_smoothing: Any = 0.0,
) -> jax.Array:
    z, t, w = _prepare(call, logits, target, weight)
    taken = _TAKEN.get()
    if call.composed:
        work = _f.work(z.dtype, call.precision)
        if work == "float64" and not _x64():
            raise ValueError("precision='float64' needs jax_enable_x64")
        if taken is not None:
            taken.append(_f.explanation(True, "", work, "jax.numpy"))
        if _stage_composed(call, z, work):
            return _compiled_composed(
                call.name,
                z,
                t,
                temperature=call.temperature,
                reduction=call.reduction,
                precision=call.precision,
            )
        return _composed_loss(call.name, z, t, weight=w, **call.options(label_smoothing))
    n, classes = int(np.prod(z.shape[:-1])), z.shape[-1]
    reason = _ops.kernels(z, n, classes, call.kind, call.k)
    if reason is not None:
        raise _f.no_kernel(reason)
    if taken is not None:
        device = _ops.platform(z)
        taken.append(
            _f.explanation(
                False, device, _f.work(z.dtype, call.precision), "jax.numpy", *_ops.route(device)
            )
        )
    floating = jnp.issubdtype(t.dtype, jnp.floating)
    dtype = _ops.value_dtype(z.dtype, t.dtype if floating else None)
    rows = call.rows()
    if n == 0:
        return _reduce(z.sum(-1).astype(dtype), call.reduction, n)
    terms = None
    if call.reduction == "mean":
        labels = _kind(t.dtype) == "i"
        targets = t.reshape(-1) if labels else t.reshape(n, -1)
        terms = _f.mean_terms(xp_jax, targets, w, call.ignore_index, labels)
    total = n if terms is None else terms
    return _ops.evaluate(z, t, w, label_smoothing, rows, call.reduction, total)


_eager_loss = jax.jit(_evaluate, static_argnums=(0, 4))


def _loss(
    call: _f.Call,
    logits: ArrayLike,
    target: ArrayLike,
    weight: ArrayLike | None = None,
    label_smoothing: Any = 0.0,
) -> jax.Array:
    # Eager differentiation of short matrices otherwise traces array conversions,
    # layout, and divisor construction on each call. Outer jit owns its own graph.
    if (
        isinstance(logits, jax.core.Tracer)
        and logits.ndim == 2
        and logits.shape[0] <= 64
        and call.axis in (-1, 1)
        and not call.composed
        and call.reduction != "none"
        and isinstance(target, jax.Array)
        and (weight is None or isinstance(weight, jax.Array))
        and isinstance(label_smoothing, float)
        and _TAKEN.get() is None
        and _ops._concrete_primal(logits)
    ):
        try:
            return _eager_loss(call, logits, target, weight, label_smoothing)
        except RuntimeError as error:
            _f.raise_label_error(error)
            raise
    return _evaluate(call, logits, target, weight, label_smoothing)


def _bind(name: str, **options: Any) -> _f.Call:
    return _f.bind(JAX, name, **options)


def _smoothing(label_smoothing: object) -> Any:
    """A static label_smoothing as a float (checked), a traced one as is."""
    if isinstance(label_smoothing, jax.core.Tracer):
        return label_smoothing
    return _f.check_label_smoothing(_static(label_smoothing, "label_smoothing"))


@_f.loss_function
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
        logits: Floating-point scores with the class axis (at least two classes); a score
            of -inf masks its class (docs/API.md, Masked classes).
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
        backend: "auto" or "kernel" (libsdca's kernels; NotImplementedError where none
            computes the call) or "composed" (jax.numpy).
        precision: None, "float32" or "float64" (or jnp.float32/float64).

    Returns:
        Per-example losses (logits.shape without axis) or their reduction, in float32 for
        16-bit logits.

    Raises:
        TypeError, ValueError: For invalid inputs or options. Invalid logit rows give NaN.
        IndexError: For labels outside the classes in synchronous CPU calls.
        NotImplementedError: For backend "auto" or "kernel" where no kernel computes the call.
    """
    call = _bind(
        "cross_entropy",
        ignore_index=ignore_index,
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )
    return _loss(call, logits, target, weight, _smoothing(label_smoothing))


@_f.loss_function
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
    call = _bind(
        "topk_cross_entropy",
        k=k,
        ignore_index=ignore_index,
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )
    return _loss(call, logits, target, weight)


@_f.loss_function
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
    call = _bind(
        "multiclass_hinge",
        gamma=gamma,
        ignore_index=ignore_index,
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )
    return _loss(call, logits, target, weight)


@_f.loss_function
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
    call = _bind(
        "topk_hinge",
        k=k,
        variant=variant,
        gamma=gamma,
        ignore_index=ignore_index,
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )
    return _loss(call, logits, target, weight)


@_f.loss_function
def proximal_cross_entropy(
    logits: ArrayLike,
    target: ArrayLike,
    *,
    gamma: float = 1.0,
    weight: ArrayLike | None = None,
    ignore_index: int | None = None,
    reduction: Reduction = "mean",
    axis: int = -1,
    backend: Backend = "auto",
    precision: object = None,
) -> jax.Array:
    """Moreau envelope of the cross-entropy with step gamma (static, finite, positive).

    Other arguments, the result and errors as in topk_cross_entropy.
    """
    call = _bind(
        "proximal_cross_entropy",
        gamma=gamma,
        ignore_index=ignore_index,
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )
    return _loss(call, logits, target, weight)


@_f.loss_function
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
    call = _bind(
        "multilabel_cross_entropy",
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )
    return _loss(call, logits, target)


@_f.loss_function
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
    call = _bind(
        "multilabel_hinge",
        gamma=gamma,
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )
    return _loss(call, logits, target)


@_f.loss_function
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

    Both arguments are LOGITS. No T² multiplier is applied. The student's gradient is exactly
    zero at equal represented logits. Differentiated in the teacher, the loss gives the
    gradient of cross_entropy(student / T, softmax(teacher / T)): p (-log q - loss) / T per
    unit of loss, with p and q the teacher's and the student's softmax at T.

    Args:
        student: Floating-point logits with the class axis (at least two classes).
        teacher: Floating-point logits of the same shape; a different dtype is promoted
            with the student's. A logit of -inf in both masks the class, one in the
            teacher alone is teacher probability zero and one in the student alone gives
            the loss +inf (docs/API.md, Masked classes).
        temperature: T, a static number, finite and positive.
        reduction: "none", "sum" or "mean".
        axis: Class axis of the logits.
        backend: "auto" or "kernel" (libsdca's kernels; NotImplementedError where none
            computes the call) or "composed" (jax.numpy).
        precision: None, "float32" or "float64" (or jnp.float32/float64).

    Returns:
        Per-example losses (student.shape without axis) or their reduction.

    Raises:
        TypeError, ValueError: For invalid inputs or options.
    """
    call = _bind(
        "distillation",
        temperature=temperature,
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )
    return _loss(call, student, teacher)


@_f.loss_function
def proximal_distillation_loss(
    student: ArrayLike,
    teacher: ArrayLike,
    *,
    temperature: float = 1.0,
    gamma: float = 1.0,
    reduction: Reduction = "mean",
    axis: int = -1,
    backend: Backend = "auto",
    precision: object = None,
) -> jax.Array:
    """Moreau envelope of distillation_loss with step gamma, as in libsdca.torch.

    min_u { H(softmax(teacher/T), softmax(u)) + |u - student/T|^2 / (2 gamma) }; gamma is
    a static number, finite and positive. Other arguments, the result and errors as in
    distillation_loss.
    """
    call = _bind(
        "proximal_distillation",
        temperature=temperature,
        gamma=gamma,
        reduction=reduction,
        axis=axis,
        backend=backend,
        precision=precision,
    )
    return _loss(call, student, teacher)


_LOSSES = (
    cross_entropy,
    topk_cross_entropy,
    multiclass_hinge_loss,
    topk_hinge_loss,
    proximal_cross_entropy,
    multilabel_cross_entropy,
    multilabel_hinge_loss,
    distillation_loss,
    proximal_distillation_loss,
)


def explain_backend(
    loss: Callable[..., jax.Array], /, *args: Any, **kwargs: Any
) -> tuple[str, str]:
    """(backend, reason): the path that loss(*args, **kwargs) takes, as libsdca.torch's.

    loss is a loss function of libsdca.jax, called with these arguments. The backend is the
    value of the backend option that takes this path: "kernel" (libsdca's kernels through
    the FFI target; the reason says which) or "composed" (jax.numpy). The call runs as given
    (eagerly for concrete arrays, traced under jax.jit).

    Raises:
        TypeError: If loss is not a libsdca.jax loss, or for arguments the call rejects.
        ValueError, NotImplementedError: For calls the loss rejects.
    """
    if loss not in _LOSSES:
        raise TypeError(f"{loss!r} is not a libsdca.jax loss function")
    taken: list[tuple[str, str]] = []
    token = _TAKEN.set(taken)
    try:
        loss(*args, **kwargs)
    finally:
        _TAKEN.reset(token)
    return taken[0]
