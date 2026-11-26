"""The losses of libsdca and their arguments, checked in one place for NumPy, PyTorch and JAX:
the losses by their shared names with their own parameters and defaults (PARAMETERS), the
options of a call (parameters, bind and the check_* functions on values), and the dtypes and
shapes of the tensor frameworks' arrays (the check_* functions on shapes and dtype kinds).
Wrong types raise TypeError, bad values ValueError, before any computation; messages differ
only in the framework's words.
"""

from __future__ import annotations

import functools
import inspect
import math
import re
import warnings
from collections.abc import Callable, Collection, Container, Iterable, Mapping, Sequence
from numbers import Integral, Real
from typing import Any, NamedTuple, TypeVar, cast

REDUCTIONS = ("none", "sum", "mean")
BACKENDS = ("auto", "kernel", "composed")
VARIANTS = ("alpha", "beta")


class _Required:
    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED: Any = _Required()

# The losses by their shared names: their own parameters and defaults (REQUIRED: none), in the
# order of the positional arguments of the NumPy loss objects. PyTorch and JAX take them as
# keywords.
PARAMETERS: Mapping[str, Mapping[str, Any]] = {
    "cross_entropy": {},
    "topk_cross_entropy": {"k": REQUIRED},
    "multiclass_hinge": {"gamma": 0.0},
    "topk_hinge": {"k": REQUIRED, "variant": "beta", "gamma": 0.0},
    "multilabel_hinge": {"gamma": 0.0},
    "multilabel_cross_entropy": {},
    "proximal_cross_entropy": {"gamma": 1.0},
    "distillation": {"temperature": 1.0},
    "proximal_distillation": {"temperature": 1.0, "gamma": 1.0},
}
# The losses that fit and coordinate_update train (sdca_model_fit and sdca_dual_update).
OBJECTIVES = tuple(PARAMETERS)[:6]
MULTILABEL = frozenset(("multilabel_cross_entropy", "multilabel_hinge"))
DISTILLATION = frozenset(("distillation", "proximal_distillation"))
TOPK = frozenset(name for name, own in PARAMETERS.items() if "k" in own)
HINGES = frozenset(("multiclass_hinge", "topk_hinge", "multilabel_hinge"))
# gamma: the smoothing of a hinge (>= 0) or the Moreau step of a proximal loss (> 0).
PROXIMAL = frozenset(("proximal_cross_entropy", "proximal_distillation"))
SINGLE_LABEL = frozenset(name for name in PARAMETERS if name not in MULTILABEL | DISTILLATION)
# The options of a loss evaluation beside a loss's parameters, PyTorch's cross_entropy's, by
# the losses that take them in every interface (the C ABI's struct sdca_loss checks the same).
OPTIONS: Mapping[str, frozenset[str]] = {
    "weight": SINGLE_LABEL,
    "label_smoothing": frozenset(("cross_entropy",)),
    "ignore_index": SINGLE_LABEL,
}

# The losses by their C kinds (topk_hinge by its variant), with the C ABI's SDCA_LOSS_* ids.
KINDS = {
    "cross_entropy": 0,
    "topk_cross_entropy": 1,
    "multiclass_hinge": 2,
    "topk_hinge_alpha": 3,
    "topk_hinge_beta": 4,
    "multilabel_hinge": 5,
    "multilabel_cross_entropy": 6,
    "proximal_cross_entropy": 7,
    "distillation": 8,
    "proximal_distillation": 9,
}
NAMES = {kind: name for name, kind in KINDS.items()}
# At k = 1 the top-k losses are the dedicated ones: topk_cross_entropy is cross_entropy, and
# both top-k hinges are multiclass_hinge with the same gamma (both feasible sets are the
# simplex {p >= 0, sum p <= 1}); include/sdca/kernel.h maps the C kinds the same way.
_AT_K1 = {
    KINDS["topk_cross_entropy"]: KINDS["cross_entropy"],
    KINDS["topk_hinge_alpha"]: KINDS["multiclass_hinge"],
    KINDS["topk_hinge_beta"]: KINDS["multiclass_hinge"],
}


def kernel_kind(kind: int, k: int) -> int:
    """The SDCA_LOSS_* id whose kernels compute loss kind with parameter k."""
    return _AT_K1.get(kind, kind) if k == 1 else kind


# Named options by their C ids: sdca_precision (the work of the storage type, float32,
# float64) and sdca_shrinking.
PRECISIONS = {None: 0, "float32": 1, "float64": 2}
SHRINKING = {"auto": 0, "off": 1, "heuristic": 2, "safe": 3, "both": 4}

FLOAT64_PRECISION = "float64 logits require precision='float64' or None"
IGNORED_PROBABILITIES = "ignore_index is not supported with class probabilities"
# The kernels' first-order message, also compiled into _torch_cpu.cpp (_torch_ops._cpu_build).
FIRST_ORDER = "libsdca kernels compute first derivatives only; use backend='composed' for more"
# The storage dtypes of the kernels' scores.
KERNEL_DTYPES = ("float32", "float64", "float16", "bfloat16")
_MASK_HINT = "; boolean masks are the targets of multilabel_cross_entropy and multilabel_hinge_loss"


class Framework(NamedTuple):
    """The words of a framework: its class-axis argument ("dim" or "axis") and that axis in
    prose, its precision dtypes (to their names) and module name, and the check that makes a
    hyperparameter a static number (JAX: concrete, not traced)."""

    axis: str
    axis_word: str
    precisions: Mapping[Any, str]
    module: str
    static: Callable[[object, str], object] | None = None


class Rows(NamedTuple):
    """The options of a row kernel (libsdca::loss, libsdca_loss): the loss's SDCA_LOSS_* id
    and parameters, the ignored label and the sdca_precision."""

    kind: int
    k: int = 1
    gamma: float = 0.0
    temperature: float = 1.0
    label_smoothing: float = 0.0
    ignore_index: int = -100
    ignore: bool = False
    precision: int = 0


class Call(NamedTuple):
    """A loss with checked options: what a module holds and a function checks per call. name
    is the C kind (KINDS)."""

    name: str
    k: int = 1
    gamma: float = 0.0
    temperature: float = 1.0
    ignore_index: int | None = None
    reduction: str = "mean"
    axis: int = -1
    backend: str = "auto"
    precision: str | None = None
    reuse_logits: bool = True

    @property
    def kind(self) -> int:
        return KINDS[self.name]

    @property
    def multilabel(self) -> bool:
        return self.name in MULTILABEL

    @property
    def composed(self) -> bool:
        return self.backend == "composed"

    def options(self, label_smoothing: Any = 0.0) -> dict[str, Any]:
        """The keyword arguments of _composed.loss."""
        return dict(
            k=self.k,
            gamma=self.gamma,
            temperature=self.temperature,
            label_smoothing=label_smoothing,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            precision=self.precision,
        )

    def rows(self, label_smoothing: float = 0.0) -> Rows:
        """The options of the row kernels, by the loss's mathematics rather than its name:
        at k = 1 the top-k losses take the dedicated kernels (kernel_kind)."""
        ignore = self.ignore_index is not None
        return Rows(
            kernel_kind(self.kind, self.k),
            self.k,
            self.gamma,
            self.temperature,
            label_smoothing,
            self.ignore_index if self.ignore_index is not None else 0,
            ignore,
            PRECISIONS[self.precision],
        )


def check_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    return float(value)


def check_integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return int(value)


def check_k(k: object) -> int:
    """A top-k parameter: an integer >= 1 (k < classes is checked with the logits)."""
    return check_integer(k, "k", 1)


def check_classes(k: int, classes: int) -> None:
    """At least two classes, and k in [1, classes - 1]."""
    if classes < 2:
        raise ValueError("At least two classes are required")
    if k >= classes:
        raise ValueError(f"k must be in [1, classes - 1] = [1, {classes - 1}], got {k}")


def check_gamma(gamma: object) -> float:
    value = check_real(gamma, "gamma")
    if not (math.isfinite(value) and value >= 0):
        raise ValueError(f"gamma must be finite and nonnegative, got {gamma}")
    return value


def check_positive(value: object, name: str) -> float:
    """A proximal step (gamma) or a temperature: finite and positive."""
    out = check_real(value, name)
    if not (math.isfinite(out) and out > 0):
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return out


def check_label_smoothing(label_smoothing: object) -> float:
    value = check_real(label_smoothing, "label_smoothing")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1], got {label_smoothing}")
    return value


def check_ignore_index(ignore_index: object, optional: bool = True) -> int | None:
    """None (no ignored label, where optional) or an int64 label."""
    if ignore_index is None and optional:
        return None
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral):
        allowed = "None or an integer" if optional else "an integer"
        raise TypeError(f"ignore_index must be {allowed}, not {type(ignore_index).__name__}")
    value = int(ignore_index)
    if not -(2**63) <= value < 2**63:
        raise ValueError("ignore_index must fit in int64")
    return value


def check_choice(value: object, name: str, allowed: Collection[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, not {type(value).__name__}")
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(map(repr, allowed))}, got {value!r}")
    return value


def check_reduction(reduction: object) -> str:
    return check_choice(reduction, "reduction", REDUCTIONS)


def check_variant(variant: object) -> str:
    return check_choice(variant, "variant", VARIANTS)


def check_backend(backend: object) -> str:
    return check_choice(backend, "backend", BACKENDS)


def check_flag(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be True or False, not {type(value).__name__}")
    return value


def check_axis(axis: object, name: str) -> int:
    """The class dimension dim (PyTorch) or axis (JAX): an integer."""
    if isinstance(axis, bool) or not isinstance(axis, Integral):
        raise TypeError(f"{name} must be an integer, not {type(axis).__name__}")
    return int(axis)


def check_precision_name(precision: object, framework: Framework) -> str | None:
    """None, "float32", "float64" or a framework dtype of its precisions (to its name)."""
    if precision is None:
        return None
    names = f"None, 'float32', 'float64', {framework.module}.float32 or {framework.module}.float64"
    if isinstance(precision, str):
        if precision in ("float32", "float64"):
            return precision
        raise ValueError(f"precision must be {names}, got {precision!r}")
    try:
        name = framework.precisions.get(precision)
    except TypeError:  # unhashable
        name = None
    if name is not None:
        return name
    if any(type(precision) is type(key) for key in framework.precisions):
        raise ValueError(f"precision must be {names}, got {precision!r}")
    raise TypeError(f"precision must be {names}, not {precision!r}")


# ---------------------------------------------------------------------------
# Losses and their parameters.

_CHECKS: Mapping[str, Callable[[str, object], object]] = {
    "k": lambda loss, value: check_k(value),
    "variant": lambda loss, value: check_variant(value),
    "gamma": lambda loss, value: (
        check_positive(value, "gamma") if loss in PROXIMAL else check_gamma(value)
    ),
    "temperature": lambda loss, value: check_positive(value, "temperature"),
}


def check_keywords(name: str, given: Iterable[str], allowed: Container[str]) -> None:
    """TypeError for a keyword that loss name does not take, such as another loss's parameter."""
    for key in given:
        if key not in allowed:
            raise TypeError(f"{name} takes no parameter {key}")


def check_options(name: str, given: Iterable[str]) -> None:
    """TypeError for an option of OPTIONS that loss name does not take."""
    for key in given:
        if name not in OPTIONS[key]:
            raise TypeError(f"{name} takes no parameter {key}")


def parameters(name: str, given: Mapping[str, object]) -> dict[str, Any]:
    """The checked parameters of loss name (a key of PARAMETERS): those given and the defaults
    of the others. TypeError for a parameter it does not take or a missing one."""
    own = PARAMETERS[name]
    check_keywords(name, given, own)
    out: dict[str, Any] = {}
    for key, default in own.items():
        value = given.get(key, default)
        if value is REQUIRED:
            raise TypeError(f"{name} needs the parameter {key}")
        out[key] = _CHECKS[key](name, value)
    return out


def kind(name: str, checked: Mapping[str, Any]) -> str:
    """The C kind (KINDS) of loss name with its checked parameters."""
    return f"topk_hinge_{checked['variant']}" if name == "topk_hinge" else name


def loss_name(function: Callable[..., Any]) -> str:
    """The shared name of a loss function of libsdca.torch or libsdca.jax."""
    return function.__name__.removesuffix("_loss")


F = TypeVar("F", bound=Callable[..., Any])


def loss_function(function: F) -> F:
    """A loss function of libsdca.torch or libsdca.jax whose unknown keywords raise the
    TypeError of check_keywords, as the NumPy loss objects' do."""
    name, allowed = loss_name(function), frozenset(inspect.signature(function).parameters)

    @functools.wraps(function)
    def checked(*args: Any, **kwargs: Any) -> Any:
        if kwargs:
            check_keywords(name, kwargs, allowed)
        return function(*args, **kwargs)

    return cast(F, checked)


_CALLS: dict[tuple[object, ...], Call] = {}


def bind(
    framework: Framework,
    name: str,
    *,
    ignore_index: object = None,
    reduction: object = "mean",
    axis: object = -1,
    backend: object = "auto",
    precision: object = None,
    reuse_logits: object = True,
    memo: bool = True,
    **given: object,
) -> Call:
    """The checked Call of loss name (a key of PARAMETERS) with its parameters given:
    TypeError for wrong types, ValueError for bad values. reuse_logits is PyTorch's. memo
    False neither reads nor fills the cache of Calls (compiled graphs would guard on it)."""
    static = framework.static
    if static is not None:
        given = {key: static(value, key) for key, value in given.items()}
    # Types tell apart equal values of which only one is valid (-100 and -100.0, 1 and True).
    key: tuple[object, ...] | None = (
        framework.module,
        name,
        ignore_index,
        type(ignore_index),
        axis,
        type(axis),
        reduction,
        backend,
        precision,
        reuse_logits,
        type(reuse_logits),
        *given.items(),
        *map(type, given.values()),
    )
    try:
        call = _CALLS.get(key) if memo else None  # type: ignore[arg-type]
    except TypeError:  # an unhashable option: checked below
        call, key = None, None
    if call is not None:
        return call
    checked = parameters(name, given)
    call = Call(
        kind(name, checked),
        checked.get("k", 1),
        checked.get("gamma", 0.0),
        checked.get("temperature", 1.0),
        None if name in DISTILLATION else check_ignore_index(ignore_index),
        check_reduction(reduction),
        check_axis(axis, framework.axis),
        check_backend(backend),
        check_precision_name(precision, framework),
        check_flag(reuse_logits, "reuse_logits"),
    )
    if memo and key is not None and len(_CALLS) < 4096:
        _CALLS[key] = call
    return call


# Shapes are sequences of ints (tuples or torch.Size, printed as tuples); a dtype kind is "f"
# (floating-point, 16-bit included), "i" (integer), "b" (boolean) or "c" (complex).
Shape = Sequence[int]


def _name(dtype: object) -> str:
    return str(dtype).removeprefix("torch.")


def class_axis(framework: Framework, ndim: int, axis: int) -> int:
    """The class axis of ndim-D logits, in [0, ndim)."""
    if ndim < 1:
        raise ValueError(f"logits must have a class {framework.axis_word}")
    if not -ndim <= axis < ndim:
        raise ValueError(f"{framework.axis} {axis} is out of range for {ndim}-D logits")
    return axis % ndim


def check_floating(kind: str, dtype: object, name: str = "logits") -> None:
    if kind != "f":
        raise TypeError(f"{name} must be floating-point, not {_name(dtype)}")


def check_precision(dtype: object, precision: str | None) -> None:
    """float64 logits take float64 (or None); other dtypes take float32 or float64 (or None)."""
    if precision == "float32" and _name(dtype) == "float64":
        raise ValueError(FLOAT64_PRECISION)


def without(shape: Shape, axis: int) -> tuple[int, ...]:
    return tuple(shape[:axis]) + tuple(shape[axis + 1 :])


def raise_label_error(error: Exception) -> None:
    """Map the framework dispatcher error from a CPU label kernel back to IndexError."""
    match = re.search(r"(Target -?\d+ is out of bounds\.)", str(error))
    if match:
        raise IndexError(match.group(1)) from error


def check_labels(
    framework: Framework,
    name: str,
    kind: str,
    dtype: object,
    shape: Shape,
    logits: Shape,
    axis: int,
) -> None:
    """Integer class labels of the logits' shape without the (checked) class axis."""
    if kind != "i":
        hint = _MASK_HINT if kind == "b" else ""
        what = "integer class labels"
        if name == "cross_entropy":
            what += " or floating-point class probabilities"
        raise TypeError(f"{name} takes {what}, not {_name(dtype)}{hint}")
    expected = without(logits, axis)
    if shape != expected:
        raise ValueError(
            f"target must have shape {expected} (the logits' shape without the class "
            f"{framework.axis_word}), not {tuple(shape)}"
        )


def check_mask(name: str, kind: str, dtype: object, shape: Shape, logits: Shape):
    """A boolean mask of the logits' shape (the positive labels)."""
    if kind != "b":
        raise TypeError(
            f"{name} takes a boolean mask of the logits' shape, not {_name(dtype)} targets"
        )
    if shape != logits:
        raise ValueError(f"target must have the logits' shape {tuple(logits)}, not {tuple(shape)}")


def check_probabilities(shape: Shape, logits: Shape) -> None:
    if shape != logits:
        raise ValueError(
            f"class probabilities must have the logits' shape {tuple(logits)}, not {tuple(shape)}"
        )


def check_pair(student: Shape, teacher: Shape) -> None:
    """The student and teacher logits of a distillation loss: one shape."""
    if student != teacher:
        raise ValueError(
            f"student and teacher must have one shape, not {tuple(student)} and {tuple(teacher)}"
        )


def check_logits(call: Call, kind: str, dtype: object) -> None:
    """The logits of a call (the student's of distillation): floating-point."""
    check_floating(kind, dtype, "student logits" if call.name in DISTILLATION else "logits")


def target_role(
    framework: Framework,
    call: Call,
    kind: str,
    dtype: object,
    shape: Shape,
    logits: Shape,
    axis: int,
) -> str:
    """The role of a call's target, checked against the logits' shape (class axis axis):
    "teacher" logits (distillation), a label "mask" (multilabel), class "probabilities"
    (floating-point targets of cross_entropy; with ignore_index negative, as PyTorch) or class
    "labels"."""
    name = call.name
    if name in DISTILLATION:
        check_floating(kind, dtype, "teacher logits")
        check_pair(logits, shape)
        return "teacher"
    if name in MULTILABEL:
        check_mask(name, kind, dtype, shape, logits)
        return "mask"
    if kind == "f" and name == "cross_entropy":
        check_probabilities(shape, logits)
        if call.ignore_index is not None and call.ignore_index >= 0:
            raise ValueError(IGNORED_PROBABILITIES)
        return "probabilities"
    check_labels(framework, name, kind, dtype, shape, logits, axis)
    return "labels"


def probability_dtype(target: object, logits: object) -> str | None:
    """The dtype that class probabilities of dtype target take with logits: "float32" where
    the kernels take neither target (the logits' dtype, float32 or float64), else None."""
    name = _name(target)
    return None if name in (_name(logits), "float32", "float64") else "float32"


# The rules of the values, on the arrays of a framework through its composed-loss namespace
# xp (_composed/xp_torch.py, xp_jax.py).


def value_dtype(xp: Any, dtype: object, target: object = None) -> Any:
    """The dtype of the losses of scores of dtype (and class probabilities of dtype target):
    float64 where either is float64, as PyTorch and optax promote, else float32."""
    return xp.float64 if dtype == xp.float64 or target == xp.float64 else xp.float32


def nan_weights(xp: Any, weight: Any) -> Any:
    """Class weights (classes,), all NaN unless every weight is finite and nonnegative."""
    valid = xp.all((weight >= 0) & (weight < float("inf")))
    return xp.where(valid, weight, float("nan"))


def mean_divisor(
    xp: Any, target: Any, weight: Any | None, ignore_index: int | None, labels: bool
) -> Any:
    """The divisor of a mean reduction, as PyTorch's cross-entropy: the number of rows
    (target: rows first), or for class labels (rows,) with class weights or ignore_index the
    sum of the weights of the labels (clamped into the classes) of the rows not ignored, in
    the framework's widest float (float64 where it has one), as a 0-d array."""
    terms = mean_terms(xp, target, weight, ignore_index, labels)
    return target.shape[0] if terms is None else xp.sum(terms).reshape(())


def mean_terms(
    xp: Any, target: Any, weight: Any | None, ignore_index: int | None, labels: bool
) -> Any | None:
    """The terms of mean_divisor's sum, one per row (rows,), or None where the divisor is the
    number of rows."""
    if not labels or (weight is None and ignore_index is None):
        return None
    wide = xp.wide()
    t = target.reshape(1, -1)
    if weight is None:
        picked = xp.full_like(t, 1.0, dtype=wide)
    else:
        w = xp.astype(weight, wide)[None, :]
        picked = xp.take(w, xp.clip(t, 0, weight.shape[0] - 1))
    if ignore_index is not None:
        picked = xp.where(t == ignore_index, 0.0, picked)
    return picked.reshape(-1)


def mean_scale(count: int) -> float:
    """2**-k with 2**k > 2 * count, for a mean of count rows whose totals leave the range: the
    sums of the rows and of mean_terms (or count) each times it, exactly before their rounding,
    stay finite where the rows and terms are, and so does their quotient where the mean is. A
    mean takes this quotient only where the plain one is not finite (as the PyTorch
    extension's finite_mean); with rows or terms that are not finite both are equal."""
    return 2.0 ** -(count.bit_length() + 1)


def check_weight(kind: str, dtype: object, shape: Shape, classes: int | None) -> None:
    """Class weights (classes,), floating-point (classes None: not known yet, any length);
    invalid values give NaN losses."""
    if kind != "f":
        raise TypeError(f"weight must be floating-point, not {_name(dtype)}")
    if len(shape) != 1 or (classes is not None and shape[0] != classes):
        raise ValueError(f"weight must have shape ({classes or 'classes'},), not {tuple(shape)}")


def missing_kernels(kinds: int, kind: int) -> str | None:
    """None where the bit mask of loss kinds of a CPU build has kind, else why not."""
    if kinds >> kind & 1:
        return None
    return f"the kernels of {NAMES[kind]} are not in this build of sdca/loss"


CUDA_MAX_ROWS = 2**28


def triton_reason(
    launch: str, kind: int, k: int, n: int, classes: int, dtype: object
) -> str | None:
    """None where the Triton kernels of a launch layer (triton/launch_torch.py or
    launch_jax.py) compute a call, else why not."""
    import importlib

    try:
        module = importlib.import_module(f".triton.{launch}", __package__)
    except ImportError as error:
        return f"the Triton launch layer does not load ({error})"
    return module.available() or module.covers(kind, k, n, classes, dtype)


def kernel_reason(
    dtype: object, platform: str, cpu: Callable[[], str | None], cuda: Callable[[], str | None]
) -> str | None:
    """None where a row kernel computes scores of dtype on platform ("cpu", "cuda" or "gpu"
    for CUDA, others), else why not; cpu and cuda give the reasons of those platforms."""
    if _name(dtype) not in KERNEL_DTYPES:
        return f"no kernel takes {_name(dtype)} scores"
    if platform == "cpu":
        return cpu()
    if platform in ("cuda", "gpu"):
        return cuda()
    return f"no kernel runs on {platform}"


def work(dtype: object, precision: str | None) -> str:
    """The working precision of scores of dtype: float64 for float64 scores or precision
    "float64", else float32."""
    return "float64" if precision == "float64" or _name(dtype) == "float64" else "float32"


def cpu_route(
    extension: str | None, native: Callable[[], str | None], env: str, guide: str
) -> str | None:
    """The rule of the CPU kernels of PyTorch and JAX: those of the framework's extension
    (extension: None where it loads, else why not), else the same kernels in the native
    library (native(): None where it computes them, else why not), with a warning once
    unless the extension's switch env is off. None where either computes a call, else why
    neither does."""
    if extension is None:
        return None
    reason = native()
    if reason is not None:
        return f"{extension}; the native library: {reason}"
    if extension != f"{env}=off":
        _warn_native(extension, env, guide)
    return None


@functools.cache
def _warn_native(extension: str, env: str, guide: str) -> None:
    warnings.warn(
        f"libsdca: {extension}. CPU losses run the same kernels in the native library "
        "(libsdca_c) instead, with more overhead per call. Install a C++ compiler and the "
        f"Python headers to build the extension ({guide}), or set {env}=off to take the "
        "native library without this warning.",
        RuntimeWarning,
        stacklevel=2,
    )


@functools.cache
def warn_python_launcher(extension: str, env: str, guide: str) -> None:
    """Once per process: the CUDA losses of libsdca.torch launch their kernels through Python
    because the C++ extension does not load."""
    warnings.warn(
        f"libsdca: {extension}. CUDA losses launch the same kernels through Python instead, "
        "with much more overhead per call than through the extension. Install a C++ compiler "
        f"and the Python headers to build the extension ({guide}), or set {env}=off to launch "
        "through Python without this warning.",
        RuntimeWarning,
        stacklevel=3,
    )


def explanation(
    composed: bool,
    platform: str,
    work: str,
    operations: str,
    route: str = "",
    fallback: str | None = None,
) -> tuple[str, str]:
    """(backend, reason) of explain_backend: the framework's composed operations, or the
    kernels of the platform by route; fallback: why the framework's extension does not load,
    where the native library computes a CPU call (cpu_route) or Python launches a CUDA call
    (warn_python_launcher)."""
    if composed:
        return "composed", f"{operations} in {work} (backend='composed')"
    where = "the CPU kernels" if platform == "cpu" else "the Triton kernels"
    if fallback is not None and platform != "cpu":
        return "kernel", f"{where} through the Python launcher ({work} work; {fallback})"
    if fallback is not None:
        return "kernel", f"{where} of the native library{route} ({work} work; {fallback})"
    return "kernel", f"{where}{route} ({work} work)"


def no_kernel(reason: str) -> NotImplementedError:
    """The error of backend="auto" or "kernel" where no kernel computes a call."""
    return NotImplementedError(
        f"no libsdca kernel computes this call ({reason}); pass backend='composed' for the "
        "composed implementation"
    )
