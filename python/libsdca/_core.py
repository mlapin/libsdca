"""The native library (through ctypes): its C ABI (include/sdca/c_api.h), its internal
entry points (src/c_api_internal.h) and the shared names and checks of the Python layers.

The NumPy API is in _numpy.py; the PyTorch and JAX layers call the raw-address helpers below.
"""

from __future__ import annotations

import ctypes as ct
import math
import os
import sys
import warnings
from functools import cache
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

ABI_VERSION = 6  # SDCA_ABI_VERSION of include/sdca/c_api.h
_LIBRARY_NAME = "libsdca_c.so"  # python/build.py output in build/python or SDCA_LIBRARY_DIR

# Working precisions of the precision options.
PRECISIONS: tuple[str, ...] = ("float32", "float64")
# The internal codes of (work, storage) pairs, as the JAX FFI attribute "precision" takes them:
# 0 = float work on float32 storage, 1 = double work on float32, 2 = double on float64.
_PRECISION_CODES: dict[tuple[str, type[np.floating]], int] = {
    ("float32", np.float32): 0,
    ("float64", np.float32): 1,
    ("float64", np.float64): 2,
}

# The shared loss names in the order of the C ids (enum sdca_loss_kind).
LOSS_NAMES: tuple[str, ...] = (
    "cross_entropy",
    "topk_cross_entropy",
    "multiclass_hinge",
    "topk_hinge_alpha",
    "topk_hinge_beta",
    "multilabel_hinge",
    "multilabel_cross_entropy",
    "proximal_cross_entropy",
    "distillation",
    "proximal_distillation",
)
# The losses of label targets (ids 0 to 7): the kinds of the tensor layers.
_LABEL_LOSSES = LOSS_NAMES[:8]
_IDS = {name: i for i, name in enumerate(_LABEL_LOSSES)}
# The training objectives (sdca_model_fit and sdca_dual_update).
OBJECTIVES: tuple[str, ...] = LOSS_NAMES[:7]
MULTILABEL = frozenset(("multilabel_hinge", "multilabel_cross_entropy"))
TOPK = frozenset(("topk_cross_entropy", "topk_hinge_alpha", "topk_hinge_beta"))
HINGES = frozenset(("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge"))
# Columns of a fit's history (enum sdca_history_field); accuracy is the rank loss of the
# multilabel losses.
HISTORY: tuple[str, ...] = (
    "epoch",
    "primal",
    "dual",
    "primal_loss",
    "dual_loss",
    "regularizer",
    "relative_gap",
    "accuracy",
    "cpu_time",
    "wall_time",
)
# Fit status (enum sdca_status); only "solved" guarantees the duality gap.
STATUS: tuple[str, ...] = ("solved", "no_progress", "max_epoch", "failed")

_INVALID_ARGUMENT = -2  # SDCA_INVALID_ARGUMENT
# Storage codes (enum sdca_dtype, enum sdca_internal_dtype); 16-bit values as raw bits.
DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_FLOAT16, DTYPE_BFLOAT16 = 0, 1, 2, 3
# enum sdca_precision.
_WORK = {None: 0, "float32": 1, "float64": 2}
# enum sdca_projection_precision.
_PROJECTIONS = {"float64": 0, "extended": 1}
# enum sdca_shrinking.
_SHRINKING = {"auto": 0, "off": 1, "heuristic": 2, "safe": 3, "both": 4}
# enum sdca_target_kind and enum sdca_input_kind.
TARGET_LABELS, TARGET_LABEL_SETS, TARGET_MASK, TARGET_PROBABILITIES, TARGET_LOGITS = range(5)
INPUT_FEATURES, INPUT_KERNEL = 0, 1


_s, _i, _d, _p = ct.c_size_t, ct.c_int, ct.c_double, ct.c_void_p


class _Loss(ct.Structure):
    """struct sdca_loss."""

    _fields_ = [
        ("size", _s),
        ("kind", _i),
        ("k", _s),
        ("gamma", _d),
        ("alpha", _d),
        ("temperature", _d),
        ("projection_precision", _i),
    ]


class _Execution(ct.Structure):
    """struct sdca_execution."""

    _fields_ = [("size", _s), ("precision", _i), ("num_threads", _s)]


class _Targets(ct.Structure):
    """struct sdca_targets."""

    _fields_ = [
        ("size", _s),
        ("kind", _i),
        ("labels", _p),
        ("offsets", _p),
        ("mask", _p),
        ("probabilities", _p),
        ("logits", _p),
    ]


class _Input(ct.Structure):
    """struct sdca_input."""

    _fields_ = [("size", _s), ("kind", _i), ("dtype", _i), ("n", _s), ("d", _s), ("x", _p)]


class _FitOptions(ct.Structure):
    """struct sdca_fit_options."""

    _fields_ = [
        ("size", _s),
        ("c", _d),
        ("tol", _d),
        ("max_epoch", _s),
        ("eval_epoch", _s),
        ("eval_on_start", _i),
        ("recompute_primal", _i),
        ("shrinking", _i),
        ("seed", ct.c_uint64),
    ]


class _FitResult(ct.Structure):
    """struct sdca_fit_result."""

    _fields_ = [("size", _s), ("status", _i), ("epochs", _s), ("history_rows", _s)]


class _InternalOptions(ct.Structure):
    """struct sdca_internal_options of src/c_api_internal.h."""

    _fields_ = [
        ("size", _s),
        ("float_work", _i),
        ("loss", _i),
        ("k", _s),
        ("gamma", _d),
        ("alpha", _d),
        ("projection_precision", _i),
        ("num_threads", _s),
    ]


# The SDCA_*_INIT values of the header (test_abi_compatibility compares them).
_DEFAULTS: dict[type[ct.Structure], dict[str, Any]] = {
    _Loss: dict(kind=0, k=0, gamma=0.0, alpha=1.0, temperature=1.0, projection_precision=0),
    _Execution: dict(precision=0, num_threads=0),
    _Targets: dict(kind=TARGET_LABELS),
    _Input: dict(kind=INPUT_FEATURES, dtype=DTYPE_FLOAT64, n=0, d=0),
    _FitOptions: dict(
        c=1.0,
        tol=1e-3,
        max_epoch=1000,
        eval_epoch=10,
        eval_on_start=0,
        recompute_primal=1,
        shrinking=0,
        seed=0,
    ),  # fmt: skip
    _FitResult: dict(status=2, epochs=0, history_rows=0),
}


def _make(struct: type[ct.Structure], **fields: Any) -> Any:
    """A struct of the header: its SDCA_*_INIT values with the given fields."""
    return struct(size=ct.sizeof(struct), **{**_DEFAULTS.get(struct, {}), **fields})


def _library_path(library: str | os.PathLike[str] | None) -> str | os.PathLike[str]:
    """The explicit library, else build/python (or SDCA_LIBRARY_DIR), else the installed one."""
    if library:
        return library
    folder = Path(
        os.environ.get("SDCA_LIBRARY_DIR", Path(__file__).resolve().parents[2] / "build/python")
    )
    path = folder / _LIBRARY_NAME
    if library is None and "SDCA_LIBRARY_DIR" not in os.environ:
        if not path.exists():
            installed = sorted(Path(__file__).resolve().parent.glob("_libsdca_native*.so"))
            if installed:
                path = installed[0]
        if path.exists():
            _warn_stale_build(path)
    return path


def stale_build(path: Path) -> str | None:
    """Why a library built in a source checkout (build/python, or the editable install's
    python/libsdca/_libsdca_native*.so) is older than its C++ sources, or None if it is
    current or not in a source checkout."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "include" / "sdca").is_dir() or not path.resolve().is_relative_to(root):
        return None
    built = path.stat().st_mtime
    sources = [root / "src" / "c_api.cpp", root / "VERSION"]
    sources += [s for s in (root / "src").glob("c_api*.h") if s.is_file()]
    sources += [s for s in (root / "include").rglob("*") if s.is_file()]
    newer = sorted(s for s in sources if s.is_file() and s.stat().st_mtime > built)
    if not newer:
        return None
    return f"{newer[0].relative_to(root)} and {len(newer) - 1} other sources are newer"


@cache
def _warn_stale_build(path: Path) -> None:
    reason = stale_build(path)
    if reason:
        warnings.warn(
            f"Loading {path}, built before its sources changed ({reason}). Rebuild it "
            "('python python/build.py' for build/python, 'pip install -e .' for the "
            "editable install).",
            RuntimeWarning,
            stacklevel=4,
        )


def _declare(lib: ct.CDLL) -> None:
    """Set the ctypes signatures of the native entry points."""
    i, s, p, f, i64 = ct.c_int, ct.c_size_t, ct.c_void_p, ct.c_double, ct.c_int64
    lib.sdca_last_error.restype = ct.c_char_p
    lib.sdca_version.restype = ct.c_char_p
    signatures: dict[str, tuple[Any, list[Any]]] = {
        # loss, execution, dtype, n, classes, scores, targets, values, gradients
        "sdca_loss_evaluate": (i, [p, p, i, s, s, p, p, p, p]),
        # loss, c, execution, dtype, n, classes, scores, targets, previous, norm2, out
        "sdca_dual_update": (i, [p, f, p, i, s, s, p, p, p, p, p]),
        "sdca_history_rows": (s, [p]),
        "sdca_max_extent": (s, []),
        # loss, options, execution, input, classes, targets, center, initial, W, A, history,
        # rows, result
        "sdca_model_fit": (i, [p, p, p, p, s, p, p, p, p, p, p, s, p]),
        # input, classes, coefficients, scores
        "sdca_model_scores": (i, [p, s, p, p]),
        # options, dtype, n, classes, logits, labels, positive, values, gradients
        "sdca_internal_loss_rows": (i, [p, i, s, s, p, p, p, p, p]),
        # options, dtype, n, classes, logits, labels, ignore_index, weights, values, stats, totals
        "sdca_internal_cross_entropy_forward": (i, [p, i, s, s, p, p, i64, p, p, p, p]),
        # options, dtype, n, classes, logits, labels, ignore_index, weights, stats, scale,
        # stride, gradients
        "sdca_internal_cross_entropy_backward": (i, [p, i, s, s, p, p, i64, p, p, p, s, p]),
        # options, dtype, n, classes, student, teacher, temperature, values, stats
        "sdca_internal_distillation_forward": (i, [p, i, s, s, p, p, f, p, p]),
        # options, dtype, n, classes, student, teacher, temperature, stats, scale, stride,
        # gradients
        "sdca_internal_distillation_backward": (i, [p, i, s, s, p, p, f, p, p, s, p]),
        # options, dtype, n, classes, student, teacher, temperature, alpha, values, gradients
        "sdca_internal_proximal_distillation_rows": (i, [p, i, s, s, p, p, f, f, p, p]),
        # rows, work per row, threads, callback, context
        "sdca_internal_run_rows": (i, [s, s, s, p, p]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)
        function.restype, function.argtypes = restype, argtypes


class _Native:
    """A loaded native library of this ABI."""

    def __init__(self, library: str | os.PathLike[str] | None = None) -> None:
        path = _library_path(library)
        try:
            self.lib = ct.CDLL(str(path))
        except OSError as error:
            raise OSError(
                f"Cannot load the native library {path} ({error}). Build it with "
                "'python python/build.py', install the package, or set SDCA_LIBRARY_DIR."
            ) from error
        if not hasattr(self.lib, "sdca_abi_version") or self.lib.sdca_abi_version() != ABI_VERSION:
            raise RuntimeError("Library and wrapper disagree on the C ABI; rebuild the library")
        _declare(self.lib)
        self.version: str = self.lib.sdca_version().decode()

    def call(self, name: str, *args: object) -> None:
        """Calls an entry point; ValueError for SDCA_INVALID_ARGUMENT, else RuntimeError."""
        code = getattr(self.lib, name)(*args)
        if code:
            _fail(self.lib, code)


@cache
def _native() -> _Native:
    """The default native library, loaded once."""
    return _Native()


def _fail(lib: ct.CDLL, code: int) -> None:
    message = lib.sdca_last_error().decode()
    raise ValueError(message) if code == _INVALID_ARGUMENT else RuntimeError(message)


def _loss_of_kind(kind: str) -> tuple[str, float | None]:
    """The loss name of a tensor-backend kind, and gamma 0 if the kind is nonsmooth."""
    return kind.removesuffix("_smooth"), None if kind.endswith("_smooth") else 0.0


_BASE_KINDS = {"multiclass_hinge": "topk_hinge_alpha", "multilabel_cross_entropy": "cross_entropy"}


def _kind(name: str, gamma: float) -> str:
    """The tensor backends' kind of a loss: the top-k hinge of multiclass_hinge,
    cross_entropy for multilabel_cross_entropy, and a _smooth suffix if gamma > 0."""
    base = _BASE_KINDS.get(name, name)
    return base + "_smooth" if gamma > 0 and name in HINGES else base


# Each label loss, bound once: (kind, kind for gamma > 0 or None if not a hinge, whether k is 1).
_Entry = tuple[str, "str | None", bool]
_KINDS: dict[str, _Entry] = {
    name: (
        _kind(name, 0.0),
        _kind(name, 1.0) if name in HINGES else None,
        name == "multiclass_hinge",
    )
    for name in _LABEL_LOSSES
}


def _check_name(name: object, names: tuple[str, ...], what: str) -> str:
    """name if it is one of names, else TypeError (not a string) or ValueError."""
    if not isinstance(name, str):
        raise TypeError(f"{what} must be a string, not {type(name).__name__}")
    if name in names:
        return name
    hint = "; the hinges are smoothed by gamma > 0" if name.endswith("_smooth") else ""
    raise ValueError(f"Unknown {what}: {name!r}{hint}")


def _entry(loss: object) -> _Entry:
    """The bound kinds of a label loss name: one table lookup, else the name error."""
    entry = _KINDS.get(loss) if type(loss) is str else None
    return entry if entry is not None else _KINDS[_check_name(loss, _LABEL_LOSSES, "loss")]


def _kind_of(entry: _Entry, k: int, gamma: float) -> tuple[str, int, float]:
    """The tensor-backend kind of a bound loss, its k (1 for multiclass_hinge) and its
    gamma (1 for a nonsmooth hinge). Callers check the given k with _check_options: the
    k of multiclass_hinge is 1 whatever the given k."""
    exact = type(gamma) is float or type(gamma) is int
    if not (exact and 0 <= gamma < math.inf) and (
        isinstance(gamma, bool | np.bool_)
        or not isinstance(gamma, Real)
        or not math.isfinite(gamma)
        or not gamma >= 0
    ):
        raise ValueError("gamma must be finite and nonnegative")
    kind, smooth, multiclass = entry
    if multiclass:
        k = 1
    return (smooth, k, float(gamma)) if smooth is not None and gamma > 0 else (kind, k, 1.0)


def _tensor_kind(loss: str, k: int, gamma: float) -> tuple[str, int, float]:
    """_kind_of the loss named loss."""
    return _kind_of(_entry(loss), k, gamma)


def _canonical_precision(precision: object) -> str | None:
    """None, or 'float32' / 'float64' of a precision string or floating-point dtype object.

    Raises:
        TypeError: For an object that is neither None, a string nor a float32/float64 dtype.
        ValueError: For a string other than "float32" and "float64".
    """
    if precision is None:
        return None
    if isinstance(precision, str):
        if precision in PRECISIONS:
            return precision
        raise ValueError(f"precision must be None or one of {', '.join(PRECISIONS)}")
    if precision is float or precision in (np.float64, np.dtype(np.float64)):
        return "float64"
    if precision in (np.float32, np.dtype(np.float32)):
        return "float32"
    if type(precision).__module__ == "torch" and str(precision) in (
        "torch.float32",
        "torch.float64",
    ):
        return str(precision).split(".")[1]
    raise TypeError(
        f"precision must be None, 'float32' or 'float64', not {type(precision).__name__}"
    )


def _topk_hinge_name(variant: str) -> str:
    """The top-k hinge loss of variant 'alpha' or 'beta'."""
    if variant in ("alpha", "beta"):
        return "topk_hinge_" + variant
    raise ValueError("variant must be 'alpha' or 'beta'")


def _check_options(k: int, gamma: float, alpha: float, classes: int) -> None:
    """The loss options of the tensor layers, before any computation."""
    fast = type(k) is int and 1 <= k < classes and type(gamma) is float and type(alpha) is float
    if fast and 0.0 <= gamma < math.inf and 0.0 < alpha < math.inf:
        return
    if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= int(k) < classes:
        raise ValueError("k must be an integer in [1, classes-1]")
    if isinstance(gamma, bool | np.bool_) or not math.isfinite(gamma) or not gamma >= 0:
        raise ValueError("gamma must be finite and nonnegative")
    if isinstance(alpha, bool | np.bool_) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")


def _is_int(value: object) -> bool:
    return not isinstance(value, bool | np.bool_) and isinstance(value, int | np.integer)


def _threads(num_threads: int | None) -> int:
    """The native num_threads of the internal helpers: 0 (the default: SDCA_NUM_THREADS,
    else the available CPUs; in a PyTorch DataLoader worker PyTorch's count) for None or 0,
    else the count."""
    if num_threads is None or num_threads == 0:
        data = sys.modules.get("torch.utils.data")
        if data is None or "SDCA_NUM_THREADS" in os.environ or data.get_worker_info() is None:
            return 0
        return int(sys.modules["torch"].get_num_threads())
    if not _is_int(num_threads) or num_threads < 0:
        raise ValueError("num_threads must be None or a nonnegative integer")
    return int(num_threads)


_Address = int | None  # a buffer's address (e.g. Tensor.data_ptr()); None or 0 is NULL

# The internal entry points under their former names (the callers' vocabulary).
_INTERNAL = {
    "sdca_loss_gradient_rows": "sdca_internal_loss_rows",
    "sdca_cross_entropy_forward": "sdca_internal_cross_entropy_forward",
    "sdca_cross_entropy_backward": "sdca_internal_cross_entropy_backward",
    "sdca_distillation_forward": "sdca_internal_distillation_forward",
    "sdca_distillation_backward": "sdca_internal_distillation_backward",
    "sdca_proximal_distillation_rows": "sdca_internal_proximal_distillation_rows",
    "sdca_run_rows": "sdca_internal_run_rows",
}


@cache
def _tensor_entry(name: str, lib: ct.CDLL | None = None) -> tuple[ct.CDLL, Any]:
    """The library (by default the shared one) and its internal entry point name
    (an sdca_internal_* name, or the former public name of one)."""
    if lib is None:
        lib = _native().lib
    return lib, getattr(lib, _INTERNAL.get(name, name))


@cache
def _tensor_options(
    fp32: bool,
    num_threads: int | None,
    loss: str = "cross_entropy",
    k: int = 1,
    gamma: float = 0.0,
    alpha: float = 1.0,
    projection_precision: str = "float64",
) -> tuple[_InternalOptions, int]:
    """Options of the internal entry points (kept alive by the cache) and their address."""
    if projection_precision not in _PROJECTIONS:
        raise ValueError("projection_precision must be 'float64' or 'extended'")
    options = _InternalOptions(
        size=ct.sizeof(_InternalOptions),
        float_work=int(bool(fp32)),
        loss=_IDS[_check_name(loss, _LABEL_LOSSES, "loss")],
        k=k,
        gamma=gamma,
        alpha=alpha,
        projection_precision=_PROJECTIONS[projection_precision],
        num_threads=_threads(num_threads),
    )
    return options, ct.addressof(options)


def _cross_entropy_forward(
    z: int,
    dtype: int,
    n: int,
    classes: int,
    labels: int,
    ignore_index: int,
    weights: _Address,
    values: int,
    stats: _Address = None,
    totals: _Address = None,
    num_threads: int | None = None,
    fp32: bool = False,
) -> None:
    """sdca_internal_cross_entropy_forward on raw addresses: contiguous logits (n, classes)
    of dtype (DTYPE_*), int64 labels, float64 class weights (or None), values (n), stats (n, 5)
    and totals (2). fp32 takes float exponentials (not for DTYPE_FLOAT64).

    Raises ValueError for invalid arguments (the native message), else RuntimeError.
    """
    lib, function = _tensor_entry("sdca_internal_cross_entropy_forward")
    address = _tensor_options(fp32, num_threads)[1]
    code = function(
        address, dtype, n, classes, z, labels, ignore_index, weights, values, stats, totals
    )
    if code:
        _fail(lib, code)


def _cross_entropy_backward(
    z: int,
    dtype: int,
    n: int,
    classes: int,
    labels: int,
    ignore_index: int,
    weights: _Address,
    stats: int,
    scale: int,
    scale_stride: int,
    gradients: int,
    num_threads: int | None = None,
    fp32: bool = False,
) -> None:
    """sdca_internal_cross_entropy_backward on raw addresses: gradients (n, classes) of dtype
    from the stats of _cross_entropy_forward, times float64 scale[i * scale_stride]."""
    lib, function = _tensor_entry("sdca_internal_cross_entropy_backward")
    address = _tensor_options(fp32, num_threads)[1]
    code = function(
        address,
        dtype,
        n,
        classes,
        z,
        labels,
        ignore_index,
        weights,
        stats,
        scale,
        scale_stride,
        gradients,
    )
    if code:
        _fail(lib, code)


def _loss_gradient_rows(
    z: int,
    dtype: int,
    n: int,
    classes: int,
    labels: _Address,
    positive: _Address,
    values: int,
    gradients: int,
    loss: str = "cross_entropy",
    k: int = 1,
    gamma: float = 0.0,
    alpha: float = 1.0,
    projection_precision: str = "float64",
    num_threads: int | None = None,
    fp32: bool = False,
) -> None:
    """sdca_internal_loss_rows on raw addresses: int64 labels (n) or a uint8 mask
    (n, classes), values (n) float64 and gradients (n, classes) in float32 for 16-bit
    dtypes, else dtype (0 or None: the values alone)."""
    lib, function = _tensor_entry("sdca_internal_loss_rows")
    options = _tensor_options(fp32, num_threads, loss, k, gamma, alpha, projection_precision)
    code = function(options[1], dtype, n, classes, z, labels, positive, values, gradients)
    if code:
        _fail(lib, code)


def _distillation_forward(
    student: int,
    teacher: int,
    dtype: int,
    n: int,
    classes: int,
    temperature: float,
    values: int,
    stats: _Address = None,
    num_threads: int | None = None,
    fp32: bool = False,
    library: ct.CDLL | None = None,
) -> None:
    """sdca_internal_distillation_forward on raw addresses: contiguous student and teacher
    logits (n, classes) of dtype, values (n) and stats (n, 7) float64 (or None)."""
    lib, function = _tensor_entry("sdca_internal_distillation_forward", library)
    address = _tensor_options(fp32, num_threads)[1]
    code = function(address, dtype, n, classes, student, teacher, temperature, values, stats)
    if code:
        _fail(lib, code)


def _distillation_backward(
    student: int,
    teacher: int,
    dtype: int,
    n: int,
    classes: int,
    temperature: float,
    stats: int,
    scale: int,
    scale_stride: int,
    gradients: int,
    num_threads: int | None = None,
    fp32: bool = False,
    library: ct.CDLL | None = None,
) -> None:
    """sdca_internal_distillation_backward on raw addresses: gradients (n, classes) of dtype
    from the stats of _distillation_forward, times float64 scale[i * scale_stride]."""
    lib, function = _tensor_entry("sdca_internal_distillation_backward", library)
    address = _tensor_options(fp32, num_threads)[1]
    code = function(
        address,
        dtype,
        n,
        classes,
        student,
        teacher,
        temperature,
        stats,
        scale,
        scale_stride,
        gradients,
    )
    if code:
        _fail(lib, code)


def _proximal_distillation_rows(
    student: int,
    teacher: int,
    dtype: int,
    n: int,
    classes: int,
    temperature: float,
    alpha: float,
    values: int,
    gradients: int,
    num_threads: int | None = None,
    library: ct.CDLL | None = None,
) -> None:
    """sdca_internal_proximal_distillation_rows on raw addresses: values (n) float64 and
    gradients (n, classes) in float32 for 16-bit and float32 dtypes, else float64."""
    lib, function = _tensor_entry("sdca_internal_proximal_distillation_rows", library)
    address = _tensor_options(False, num_threads)[1]
    code = function(
        address, dtype, n, classes, student, teacher, temperature, alpha, values, gradients
    )
    if code:
        _fail(lib, code)


def _row_message(message: str, loss: str) -> str:
    """The message of a row check of sdca_internal_loss_rows, in the wording of the losses."""
    if message == "This loss takes one positive label per row":
        return "This loss needs exactly one target"
    if message == "This loss needs a negative label per row":
        if loss == "proximal_cross_entropy":
            return "Proximal entropy needs a non-target class"
        return "Multilabel hinge needs a negative label"
    return message


def _host_loss_and_grad(
    logits: Any,
    positive: Any,
    loss: str,
    *,
    k: int = 1,
    gamma: float = 0.0,
    alpha: float = 1.0,
    num_threads: int | None = None,
    precision: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The label losses of the tensor layers' host paths: per-row values (float64) and
    gradients (the storage dtype) of float32/float64 logits (n, classes) and a boolean
    positive-label mask (n, classes), as the former public loss_and_grad of a mask (k, gamma
    and alpha are read only by the losses that take them; cross_entropy of several labels
    has the uniform target on them). precision is the working precision (None: the
    storage's); num_threads None or 0 is the default, else a count."""
    loss = _check_name(loss, _LABEL_LOSSES, "loss")
    z = np.asarray(logits)
    if z.dtype not in (np.float32, np.float64):
        z = z.astype(np.float64)
    z = np.ascontiguousarray(z)
    work = _canonical_precision(precision) or z.dtype.name
    if work == "float32" and z.dtype == np.float64:
        raise ValueError("float64 inputs require precision='float64' or None")
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] < 2:
        raise ValueError("logits must have shape (n>=1,classes>=2)")
    if not np.isfinite(z).all():
        raise ValueError("Inputs must be finite")
    if not _is_int(k) or not 1 <= k < z.shape[1]:
        raise ValueError("k must be an integer in [1, classes-1]")
    if not (math.isfinite(gamma) and gamma >= 0):
        raise ValueError("gamma must be finite and nonnegative")
    if not (math.isfinite(alpha) and alpha > 0):
        raise ValueError("alpha must be finite and positive")
    mask = np.ascontiguousarray(np.asarray(positive, dtype=bool))
    if mask.shape != z.shape:
        raise ValueError("A label mask must have shape (n, num_classes)")
    if not mask.any(axis=1).all():
        raise ValueError("Every example must have at least one positive label")
    values = np.empty(len(z), np.float64)
    gradients = np.empty_like(z)
    try:
        n, classes = z.shape
        _loss_gradient_rows(
            z.ctypes.data,
            DTYPE_FLOAT32 if z.dtype == np.float32 else DTYPE_FLOAT64,
            n,
            classes,
            None,
            mask.ctypes.data,
            values.ctypes.data,
            gradients.ctypes.data,
            loss=loss,
            k=int(k),
            gamma=float(gamma),
            alpha=float(alpha),
            num_threads=num_threads,
            fp32=work == "float32",
        )
    except ValueError as error:
        raise ValueError(_row_message(str(error), loss)) from None
    return values, gradients
