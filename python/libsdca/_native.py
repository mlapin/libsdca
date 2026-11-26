"""The NumPy interface: a ctypes binding of the C ABI (include/sdca/c_api.h).

Arrays are C-contiguous and shared with the native library without copies: X (n, d), the
weights W (m, d), one row per class, and the duals A (n, m) for m classes. The objective
is 0.5 * ||W - center||^2 + C * sum_i loss(W @ X[i], y[i]); there is no implicit intercept
(append a constant feature for a regularized one).
"""

from __future__ import annotations

import ctypes as ct
import importlib.util
import inspect
import json
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from . import _frontend as _f

ABI_VERSION = 7  # SDCA_ABI_VERSION
_LIBRARY_NAME = "libsdca_c.so"  # python/build.py output in build/python or SDCA_LIBRARY_DIR

# Working precisions of the precision options.
PRECISIONS: tuple[str, ...] = tuple(name for name in _f.PRECISIONS if name is not None)
# The shared names of the training objectives (sdca_model_fit and sdca_dual_update).
OBJECTIVES: tuple[str, ...] = _f.OBJECTIVES
MULTILABEL = _f.MULTILABEL
TOPK = _f.TOPK
HINGES = _f.HINGES
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

_INVALID_ROWS, _INVALID_ARGUMENT = 1, -2  # SDCA_INVALID_ROWS, SDCA_INVALID_ARGUMENT
_DTYPE_FLOAT32, _DTYPE_FLOAT64, _DTYPE_FLOAT16, _DTYPE_BFLOAT16 = range(4)  # enum sdca_dtype
# enum sdca_target_kind and enum sdca_input_kind.
_LABELS, _LABEL_SETS, _MASK, _PROBABILITIES, _LOGITS = range(5)
_FEATURES, _KERNEL = 0, 1

_s, _i, _d, _p = ct.c_size_t, ct.c_int, ct.c_double, ct.c_void_p


class _Loss(ct.Structure):
    _fields_ = [
        ("size", _s),
        ("kind", _i),
        ("k", _s),
        ("gamma", _d),
        ("temperature", _d),
        ("weight", _p),
        ("label_smoothing", _d),
        ("ignore", _i),
        ("ignore_index", ct.c_int64),
    ]


class _LossReport(ct.Structure):
    _fields_ = [("size", _s), ("invalid_rows", _s), ("first_invalid_row", _s)]


class _Execution(ct.Structure):
    _fields_ = [
        ("size", _s),
        ("precision", _i),
        ("num_threads", _s),
        ("check_rows", _i),
        ("report", ct.POINTER(_LossReport)),
    ]


class _Targets(ct.Structure):
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
    _fields_ = [("size", _s), ("kind", _i), ("dtype", _i), ("n", _s), ("d", _s), ("x", _p)]


class _FitOptions(ct.Structure):
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
    _fields_ = [("size", _s), ("status", _i), ("epochs", _s), ("history_rows", _s)]


class _RowOptions(ct.Structure):
    _fields_ = [
        ("size", _s),
        ("probability_dtype", _i),
        ("upstream", _d),
        ("row_upstream", _p),
    ]


# The SDCA_*_INIT values of the header (a Loss sets every parameter of its kind).
_DEFAULTS: dict[type[ct.Structure], dict[str, Any]] = {
    _Loss: dict(kind=0, k=0, gamma=0.0, temperature=1.0, label_smoothing=0.0, ignore_index=-100),
    _Execution: dict(precision=0, num_threads=0, check_rows=0),
    _Targets: dict(kind=_LABELS),
    _Input: dict(kind=_FEATURES, dtype=_DTYPE_FLOAT64, n=0, d=0),
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
    _RowOptions: dict(probability_dtype=_DTYPE_FLOAT64, upstream=1.0),
}


def _make(struct: type[ct.Structure], **fields: Any) -> Any:
    """A struct of the header: its SDCA_*_INIT values with the given fields."""
    return struct(size=ct.sizeof(struct), **{**_DEFAULTS.get(struct, {}), **fields})


class InvalidRowsWarning(RuntimeWarning):
    """loss_and_grad computed rows it could not compute: their values and gradients are NaN
    (scores of NaN or +inf, invalid rows of masked classes). Pass
    check_rows=True to raise ValueError instead."""


# ---------------------------------------------------------------------------
# The library.


def _library_path() -> Path:
    """SDCA_LIBRARY_DIR, else build/python of a source checkout, else the installed one."""
    folder = os.environ.get("SDCA_LIBRARY_DIR")
    if folder:
        return Path(folder) / _LIBRARY_NAME
    path = Path(__file__).resolve().parents[2] / "build" / "python" / _LIBRARY_NAME
    if not path.exists():
        installed = sorted(Path(__file__).resolve().parent.glob("_libsdca_native*.so"))
        if installed:
            path = installed[0]
    if path.exists():
        _warn_stale_build(path)
    return path


def stale_build(path: Path) -> str | None:
    """Why a library built in a source checkout is older than its C++ sources, or None if
    it is current or not in a source checkout."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "include" / "sdca").is_dir() or not path.resolve().is_relative_to(root):
        return None
    built = path.stat().st_mtime
    sources = [root / "VERSION", *(root / "src").glob("*.[ch]*"), *(root / "include").rglob("*")]
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


def _load_blas_package(path: Path) -> ct.CDLL | None:
    """Load the library of the BLAS package that the build of path linked (SDCA_BLAS_BACKEND=
    PACKAGE, recorded in path.with_suffix(".json")) from wherever the package is installed; the
    loader then resolves path's dependency on it by its soname. RTLD_LOCAL, without importing
    the package (which loads it RTLD_GLOBAL): SciPy bundles the same OpenBLAS symbols."""
    record = path.with_suffix(".json")
    package = json.loads(record.read_text()).get("blas_package") if record.is_file() else None
    if package is None:
        return None
    spec = importlib.util.find_spec(package["module"])
    if spec is None or spec.origin is None:
        raise OSError(
            f"The native library {path} needs the package {package['distribution']}: "
            f"pip install {package['distribution']}"
        )
    return ct.CDLL(str(Path(spec.origin).parent / package["library"]))


class _Native:
    """The loaded native library of this ABI."""

    def __init__(self) -> None:
        path = _library_path()
        self.blas_library = _load_blas_package(path)
        try:
            self.lib = ct.CDLL(str(path))
        except OSError as error:
            raise OSError(
                f"Cannot load the native library {path} ({error}). Build it with "
                "'python python/build.py', install the package, or set SDCA_LIBRARY_DIR."
            ) from error
        if not hasattr(self.lib, "sdca_abi_version") or self.lib.sdca_abi_version() != ABI_VERSION:
            raise RuntimeError("Library and wrapper disagree on the C ABI; rebuild the library")
        i, s, p, f = ct.c_int, ct.c_size_t, ct.c_void_p, ct.c_double
        self.lib.sdca_last_error.restype = ct.c_char_p
        self.lib.sdca_version.restype = ct.c_char_p
        self.lib.sdca_blas_backend.restype = ct.c_char_p
        signatures: dict[str, tuple[Any, list[Any]]] = {
            # loss, execution, dtype, n, classes, scores, targets, values, gradients
            "sdca_loss_evaluate": (i, [p, p, i, s, s, p, p, p, p]),
            # loss, c, execution, dtype, n, classes, scores, targets, previous, norm2, out
            "sdca_dual_update": (i, [p, f, p, i, s, s, p, p, p, p, p]),
            "sdca_history_rows": (s, [p]),
            "sdca_max_extent": (s, []),
            # loss, options, execution, input, classes, targets, center, initial, W, A,
            # history, rows, result
            "sdca_model_fit": (i, [p, p, p, p, s, p, p, p, p, p, p, s, p]),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = restype, argtypes
        self.version: str = self.lib.sdca_version().decode()
        self.blas: str = self.lib.sdca_blas_backend().decode()
        self._blas_warned = False

    def warn_without_blas(self) -> None:
        """A RuntimeWarning, once, before the first BLAS call of a library built without an
        external BLAS because none was found (sdca_blas_backend "fallback")."""
        if self.blas != "fallback" or self._blas_warned:
            return
        self._blas_warned = True
        warnings.warn(
            "libsdca was built without a BLAS (none was found) and uses its built-in "
            "products; install OpenBLAS (or another CBLAS) and reinstall libsdca from source, "
            "or set SDCA_BLAS to its linker flags (docs/API.md#blas)",
            RuntimeWarning,
            stacklevel=3,
        )

    def call(self, name: str, *args: object) -> str | None:
        """Calls an entry point: the message of SDCA_INVALID_ROWS, None for SDCA_OK;
        ValueError for SDCA_INVALID_ARGUMENT, RuntimeError for the other errors."""
        code = getattr(self.lib, name)(*args)
        if code == 0:
            return None
        message = self.lib.sdca_last_error().decode()
        if code == _INVALID_ROWS:
            return message
        raise ValueError(message) if code == _INVALID_ARGUMENT else RuntimeError(message)


@cache
def _native() -> _Native:
    return _Native()


# ---------------------------------------------------------------------------
# Arguments: TypeError for a wrong type, ValueError for a bad value (_frontend's checks).

FloatArray = npt.NDArray[np.floating]
# A working precision: None (the storage's), "float32" or "float64" (or their dtypes).
Precision = str | type[float] | type[np.floating] | np.dtype[Any] | None
IndexArray = npt.NDArray[np.int64]

NUMPY = _f.Framework(
    "axis",
    "axis",
    {
        float: "float64",
        np.float64: "float64",
        np.dtype(np.float64): "float64",
        np.float32: "float32",
        np.dtype(np.float32): "float32",
    },
    "np",
)


def _flag(value: object, name: str) -> int:
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a bool, not {type(value).__name__}")
    return int(bool(value))


def _precision(precision: object) -> str | None:
    """None, or 'float32' / 'float64' of a precision string or floating-point dtype."""
    return _f.check_precision_name(precision, NUMPY)


# ---------------------------------------------------------------------------
# Losses.


class Loss:
    """A loss of the shared vocabulary (docs/API.md) with its own parameters, as positional or
    keyword arguments.

    Attributes:
        name: The shared name, a key of LOSSES.
    """

    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        own = _f.PARAMETERS[cls.name].items()
        empty, positional = inspect.Parameter.empty, inspect.Parameter.POSITIONAL_OR_KEYWORD
        cls.__signature__ = inspect.Signature(
            [
                inspect.Parameter(key, positional, default=empty if d is _f.REQUIRED else d)
                for key, d in own
            ]
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        own = tuple(_f.PARAMETERS[self.name])
        if len(args) > len(own):
            raise TypeError(
                f"{self.name} takes the parameters ({', '.join(own)}), not {len(args)} values"
            )
        given = dict(zip(own, args, strict=False))
        for key in kwargs:
            if key in given:
                raise TypeError(f"{self.name} got two values of parameter {key}")
        given.update(kwargs)
        for key, value in _f.parameters(self.name, given).items():
            object.__setattr__(self, key, value)

    def _struct(self) -> _Loss:
        """struct sdca_loss of this loss."""
        params = {key: getattr(self, key) for key in _f.PARAMETERS[self.name]}
        fields = {key: params[key] for key in ("k", "gamma", "temperature") if key in params}
        return _make(_Loss, kind=_f.KINDS[_f.kind(self.name, params)], **fields)


# The fields of each loss are _frontend.PARAMETERS[name] in its order, which holds their
# defaults and checks.


@dataclass(frozen=True, init=False)
class CrossEntropy(Loss):
    """cross_entropy: log sum_j exp(z_j) - z_y of one label (a training objective; also on
    Probabilities targets)."""

    name: ClassVar[str] = "cross_entropy"


@dataclass(frozen=True, init=False)
class TopKCrossEntropy(Loss):
    """topk_cross_entropy of one label; k is an integer in [1, num_classes - 1]."""

    name: ClassVar[str] = "topk_cross_entropy"
    k: int


@dataclass(frozen=True, init=False)
class MulticlassHinge(Loss):
    """multiclass_hinge (Crammer-Singer) of one label; gamma > 0 smooths it."""

    name: ClassVar[str] = "multiclass_hinge"
    gamma: float


@dataclass(frozen=True, init=False)
class TopKHinge(Loss):
    """topk_hinge of one label; k is an integer in [1, num_classes - 1]. variant "beta" (the
    default) averages the k largest hinges, "alpha" is the hinge of the average of the k
    largest margins; gamma > 0 smooths it."""

    name: ClassVar[str] = "topk_hinge"
    k: int
    variant: str
    gamma: float


@dataclass(frozen=True, init=False)
class MultilabelHinge(Loss):
    """multilabel_hinge of label sets (each with a negative class); gamma > 0 smooths it."""

    name: ClassVar[str] = "multilabel_hinge"
    gamma: float


@dataclass(frozen=True, init=False)
class MultilabelCrossEntropy(Loss):
    """multilabel_cross_entropy: cross-entropy to the uniform target on a label set."""

    name: ClassVar[str] = "multilabel_cross_entropy"


@dataclass(frozen=True, init=False)
class ProximalCrossEntropy(Loss):
    """proximal_cross_entropy: the Moreau envelope of cross-entropy with step gamma (finite
    and positive; no fit)."""

    name: ClassVar[str] = "proximal_cross_entropy"
    gamma: float


@dataclass(frozen=True, init=False)
class Distillation(Loss):
    """distillation: H(softmax(t / T), softmax(z / T)) of teacher logits t (no fit)."""

    name: ClassVar[str] = "distillation"
    temperature: float


@dataclass(frozen=True, init=False)
class ProximalDistillation(Loss):
    """proximal_distillation: the Moreau envelope (step gamma, finite and positive) of
    distillation in z / T."""

    name: ClassVar[str] = "proximal_distillation"
    temperature: float
    gamma: float


_CROSS_ENTROPY = CrossEntropy()

# The losses by their shared names, in the order of the C ids.
LOSSES: Mapping[str, type[Loss]] = {
    cls.name: cls
    for cls in (
        CrossEntropy,
        TopKCrossEntropy,
        MulticlassHinge,
        TopKHinge,
        MultilabelHinge,
        MultilabelCrossEntropy,
        ProximalCrossEntropy,
        Distillation,
        ProximalDistillation,
    )
}
_PROBABILISTIC = frozenset(("cross_entropy", "multilabel_cross_entropy"))


# ---------------------------------------------------------------------------
# Targets and arrays.


@dataclass(frozen=True, eq=False)
class Probabilities:
    """Target probabilities (n, num_classes) for CrossEntropy: the distillation objective
    H(p_i, softmax(x_i W)). Rows must be finite and nonnegative; in fit and
    coordinate_update they must also sum to one within gamma_{m+1} of the dtype of
    X (or the scores), and are renormalized in float64. loss_and_grad takes them as given (as
    torch.nn.functional.cross_entropy: the loss scales with the row sum)."""

    probabilities: npt.ArrayLike


@dataclass(frozen=True, eq=False)
class Kernel:
    """A Gram matrix K (n, n) as the input of fit (in place of features X); it must be
    symmetric positive semidefinite (docs/API.md#options)."""

    gram: npt.ArrayLike


Targets = npt.ArrayLike | Sequence[Sequence[int]] | Probabilities


_warned_device_copy = False


def _numpy(x: Any) -> Any:
    """A torch tensor (any device) as a NumPy array, without importing torch: bfloat16,
    which NumPy lacks, converts exactly to float32."""
    global _warned_device_copy
    if not type(x).__module__.startswith("torch") or not hasattr(x, "detach"):
        return x
    device = str(x.device.type)
    if device != "cpu" and not _warned_device_copy:
        _warned_device_copy = True
        warnings.warn(
            f"libsdca copies {device} tensors to the host: the NumPy API runs on the CPU; "
            "use libsdca.torch to keep the loss on the device (warned once per process)",
            RuntimeWarning,
            stacklevel=3,
        )
    x = x.detach().cpu()
    return (x.float() if str(x.dtype) == "torch.bfloat16" else x).numpy()


def _dense(x: Any, name: str) -> np.ndarray:
    """x as a NumPy array of real numbers (TypeError for sparse matrices and other types)."""
    x = _numpy(x)
    if type(x).__module__.startswith("scipy.sparse"):
        raise TypeError(
            f"{name} is a sparse matrix, which libsdca does not take: pass {name}.toarray() "
            "(a dense copy of n * d values)"
        )
    try:
        a = np.asarray(x)
    except ValueError:
        raise TypeError(f"{name} must be a rectangular array of real numbers") from None
    if a.dtype.kind not in "fiu" and a.dtype.name != "bfloat16":
        raise TypeError(f"{name} must hold real numbers, not {a.dtype}")
    return a


def _storage(a: np.ndarray, precision: object) -> tuple[type[np.floating], int]:
    """The storage dtype of an input (float32 and float64 kept, 16-bit floats cast to
    float32, integers to float64; precision="float32" casts float64 to float32) and its
    working precision (enum sdca_precision)."""
    work = _precision(precision)
    narrow = a.dtype.kind == "f" and a.dtype.itemsize <= 4 or a.dtype.name == "bfloat16"
    storage: type[np.floating] = np.float32 if narrow or work == "float32" else np.float64
    return storage, _f.PRECISIONS[work]


def _cast(a: np.ndarray, storage: type[np.floating]) -> FloatArray:
    return np.require(a, dtype=storage, requirements=["C", "A"])


def _dtype(a: np.ndarray) -> int:
    return _DTYPE_FLOAT32 if a.dtype == np.float32 else _DTYPE_FLOAT64


def _scores(logits: Any, precision: object, name: str = "logits") -> tuple[FloatArray, int]:
    """Scores (n >= 0, classes >= 2) in their storage dtype, and the working precision."""
    a = _dense(logits, name)
    storage, work = _storage(a, precision)
    if a.ndim != 2:
        raise ValueError(f"{name} must have shape (n, classes)")
    if a.shape[1] < 2:
        raise ValueError("At least two classes are required")
    return _cast(a, storage), work


def _execution(work: int, num_threads: object, **fields: Any) -> _Execution:
    threads = 0 if num_threads is None else _f.check_integer(num_threads, "num_threads", 1)
    return _make(_Execution, precision=work, num_threads=threads, **fields)


def _check_loss(loss: object, objective: bool = False) -> Loss:
    if not isinstance(loss, Loss):
        raise TypeError(
            f"loss must be a libsdca loss such as CrossEntropy(), not {type(loss).__name__}"
        )
    if objective and loss.name not in OBJECTIVES:
        raise ValueError(f"{loss.name} is not a training objective (one of {OBJECTIVES})")
    return loss


def _labels(y: Any, n: int, loss: str | None = None) -> IndexArray:
    """One integer class label per example as a contiguous int64 array (n,); the texts of
    loss_and_grad name its loss (docs/API.md, Errors: one text per error)."""
    y = _numpy(y)
    try:
        labels = np.asarray(y)
    except ValueError:
        raise TypeError("y must be an integer label array of shape (n,)") from None
    if labels.dtype.kind not in "iu":
        if loss is not None:
            raise TypeError(f"{loss} takes integer class labels, not {labels.dtype}")
        raise TypeError(f"y must hold integer class labels, not {labels.dtype}")
    if labels.ndim == 2 and labels.shape[1] == 1:
        raise ValueError("y of shape (n, 1): pass y.ravel() for one label per example")
    if labels.shape != (n,):
        if loss is not None:
            raise ValueError(
                f"target must have shape ({n},) (the logits' shape without the class "
                f"dimension), not {labels.shape}"
            )
        raise ValueError(f"y must have shape ({n},): one label per example")
    return np.require(labels, dtype=np.int64, requirements=["C", "A"])


def _nested_mask(y: Any) -> Any:
    """y, or the boolean array of y if it is a mask as nested lists."""
    if isinstance(y, Sequence) and not isinstance(y, str | bytes):
        try:
            rectangular = np.asarray(y)
        except ValueError:  # ragged label sets
            rectangular = None
        if rectangular is not None and rectangular.dtype == np.bool_:
            return rectangular
    return y


def _label_set(row: Any) -> IndexArray:
    """The sorted int64 labels of one label set."""
    row = _numpy(row)
    if isinstance(row, str | bytes) or np.ndim(row) != 1:
        raise TypeError("each label set must be a sequence of integer labels")
    labels = np.asarray(row)
    if labels.dtype == np.bool_ or (labels.size and labels.dtype.kind not in "iu"):
        raise TypeError(f"label sets must hold integer labels, not {labels.dtype}")
    labels = np.sort(labels.astype(np.int64))
    if labels.size > 1 and np.any(labels[1:] == labels[:-1]):
        raise ValueError("Label sets must not contain duplicates")
    return labels


def _label_sets(
    y: Any, n: int
) -> tuple[npt.NDArray[np.uint8], None] | tuple[IndexArray, IndexArray]:
    """Label sets: a (n, m) boolean mask as uint8 (and None), or the sorted labels of every
    set concatenated with their offsets (n + 1)."""
    y = _nested_mask(_numpy(y))
    if isinstance(y, np.ndarray) and y.dtype == np.bool_:
        if y.ndim != 2 or len(y) != n:
            raise ValueError(f"A label mask must have shape ({n}, num_classes)")
        return np.require(y, requirements=["C", "A"]).view(np.uint8), None
    if isinstance(y, np.ndarray) and y.ndim == 1 and y.dtype.kind in "iu":
        raise TypeError("label sets must be sequences of labels: pass [[label] for label in y]")
    if isinstance(y, str | bytes) or not isinstance(y, Sequence | np.ndarray):
        raise TypeError("y must be a boolean mask (n, num_classes) or a sequence of label sets")
    if len(y) != n:
        raise ValueError(f"y must have {n} label sets, one per example")
    rows = [_label_set(row) for row in y]
    offsets = np.zeros(n + 1, np.int64)
    np.cumsum([len(row) for row in rows], out=offsets[1:])
    flat = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    return np.ascontiguousarray(flat, dtype=np.int64), offsets


def _probabilities(p: Probabilities, n: int) -> npt.NDArray[np.float64]:
    """Target probabilities (n, classes) in float64 (the library checks the values and,
    for training, the row sums, and renormalizes them)."""
    a = _dense(p.probabilities, "probabilities")
    if a.ndim != 2 or len(a) != n or a.shape[1] < 2:
        raise ValueError(f"probabilities must have shape ({n}, num_classes>=2)")
    return np.ascontiguousarray(a, dtype=np.float64)


class _TargetArgs:
    """struct sdca_targets of the targets of a loss, with the arrays it points to."""

    def __init__(self, kind: int, **arrays: np.ndarray | None) -> None:
        self.arrays = arrays
        address = {key: None if a is None else a.ctypes.data for key, a in arrays.items()}
        self.struct = _make(_Targets, kind=kind, **address)
        self.kind = kind

    @property
    def ref(self) -> Any:
        return ct.byref(self.struct)

    def num_classes(self) -> int | None:
        """The classes the targets imply: a mask's or matrix's width, else max label + 1."""
        if self.kind in (_MASK, _PROBABILITIES, _LOGITS):
            array = next(a for a in self.arrays.values() if a is not None)
            return int(array.shape[1])
        labels = self.arrays["labels"]
        return int(labels.max()) + 1 if labels is not None and labels.size else None


def _targets(
    loss: Loss, y: Any, n: int, *, scores: FloatArray | None = None, evaluation: bool = False
) -> _TargetArgs:
    """The targets a loss takes (no guessing from y): labels, label sets, Probabilities
    (CrossEntropy) or teacher logits (the distillation losses); matrices of the shape of
    the scores where they are given. evaluation: the label texts of loss_and_grad."""
    if isinstance(y, Probabilities):
        if loss.name != "cross_entropy":
            raise TypeError("Probabilities are targets of CrossEntropy only")
        q = _probabilities(y, n)
        if scores is not None:
            _f.check_probabilities(q.shape, scores.shape)
        return _TargetArgs(_PROBABILITIES, probabilities=q)
    if loss.name in MULTILABEL:
        first, second = _label_sets(y, n)
        if second is not None:
            return _TargetArgs(_LABEL_SETS, labels=first, offsets=second)
        if scores is not None:
            _f.check_mask(loss.name, "b", np.bool_, first.shape, scores.shape)
        return _TargetArgs(_MASK, mask=first)
    if loss.name not in _f.DISTILLATION:
        return _TargetArgs(_LABELS, labels=_labels(y, n, loss.name if evaluation else None))
    assert scores is not None
    teacher = _dense(y, "teacher logits")
    _f.check_pair(scores.shape, teacher.shape)
    return _TargetArgs(_LOGITS, logits=_cast(teacher, scores.dtype.type))


# ---------------------------------------------------------------------------
# The entry points.


def loss_and_grad(
    logits: npt.ArrayLike,
    target: Targets,
    loss: Loss = _CROSS_ENTROPY,
    *,
    precision: Precision = None,
    num_threads: int | None = None,
    check_rows: bool = False,
    weight: npt.ArrayLike | None = None,
    label_smoothing: float | None = None,
    ignore_index: int | None = None,
) -> tuple[npt.NDArray[np.float64], FloatArray]:
    """Per-row unregularized losses and their gradients with respect to the logits.

    Nonsmooth hinge gradients are subgradients at the kinks. ProximalCrossEntropy and
    ProximalDistillation are Moreau envelopes, not the plain losses.

    Args:
        logits: Scores (n, classes), classes >= 2; a score of -inf masks its class
            (docs/API.md, Masked classes).
        target: What the loss takes: integer labels (n,) for the single-label losses and
            ProximalCrossEntropy; a boolean mask (n, classes) or a sequence of integer label
            sets for the multilabel losses; teacher logits (n, classes) for Distillation and
            ProximalDistillation; Probabilities (n, classes) for CrossEntropy (finite and
            nonnegative; a score of -inf at a class of positive probability gives +inf).
        loss: A loss of LOSSES, such as CrossEntropy() or TopKHinge(k=2).
        precision: Working precision: None (the storage's: 16-bit inputs are cast to float32
            and integers to float64), "float32" (float64 inputs are cast to float32) or
            "float64".
        num_threads: Threads for the rows: None takes SDCA_NUM_THREADS, else the available
            CPUs; a positive integer caps them. The results do not depend on it.
        check_rows: Raise ValueError for an invalid row instead of returning NaN rows.
        weight: Class weights (classes,), finite and nonnegative, of the single-label losses
            (docs/API.md, Options): a row's loss is w_y times its loss, and CrossEntropy's
            targets (labels or Probabilities) take the target weights w_j t_j, as PyTorch's
            cross_entropy (None: every weight 1).
        label_smoothing: CrossEntropy's eps in [0, 1]: the targets (1 - eps) t + eps / classes
            (None: 0).
        ignore_index: A label of the single-label losses whose rows have the loss 0 and a
            gradient of +0.0, whatever their scores (None: none; not with Probabilities if
            >= 0).

    Returns:
        Losses (n,) in float64 and gradients (n, classes) in the storage dtype. An invalid
        row (a score of NaN or +inf, an empty label set, an invalid row of masked classes)
        has a NaN loss and NaN gradients, with one InvalidRowsWarning per call; the other
        rows are as computed alone. A label out of range raises ValueError, as in PyTorch.

    Raises:
        TypeError: For arguments of a wrong type (such as float labels or a sparse matrix), and
            for an option the loss does not take.
        ValueError: For invalid values, and with check_rows for an invalid row.
        RuntimeError: If the native computation fails.
    """
    loss = _check_loss(loss)
    options = dict(weight=weight, label_smoothing=label_smoothing, ignore_index=ignore_index)
    _f.check_options(loss.name, [key for key, value in options.items() if value is not None])
    z, work = _scores(logits, precision)
    _f.check_classes(getattr(loss, "k", 1), z.shape[1])
    struct = loss._struct()
    weights = None
    if weight is not None:
        weights = _dense(weight, "weight")
        _f.check_weight(weights.dtype.kind, weights.dtype, weights.shape, z.shape[1])
        weights = np.ascontiguousarray(weights, np.float64)
        struct.weight = weights.ctypes.data
    if label_smoothing is not None:
        struct.label_smoothing = _f.check_label_smoothing(label_smoothing)
    ignored = _f.check_ignore_index(ignore_index)
    if ignored is not None:
        struct.ignore, struct.ignore_index = 1, ignored
    targets = _targets(loss, target, len(z), scores=z, evaluation=True)
    execution = _execution(work, num_threads, check_rows=_flag(check_rows, "check_rows"))
    values = np.empty(len(z), np.float64)
    gradients = np.empty_like(z)
    invalid_rows = _native().call(
        "sdca_loss_evaluate",
        ct.byref(struct),
        ct.byref(execution),
        _dtype(z),
        *z.shape,
        z.ctypes.data,
        targets.ref,
        values.ctypes.data,
        gradients.ctypes.data,
    )
    if invalid_rows is not None:
        warnings.warn(invalid_rows, InvalidRowsWarning, stacklevel=2)
    return values, gradients


# The rows of libsdca.torch and libsdca.jax where their CPU extensions are not built
# (sdca_loss_rows): the kernels of those extensions on the frameworks' buffers.

# enum sdca_dtype of the storage types by their names.
ROW_DTYPES = {
    "float32": _DTYPE_FLOAT32,
    "float64": _DTYPE_FLOAT64,
    "float16": _DTYPE_FLOAT16,
    "bfloat16": _DTYPE_BFLOAT16,
}
# enum sdca_target_kind of the frameworks' target roles (_frontend.target_role).
ROW_TARGETS = {
    "labels": _LABELS,
    "mask": _MASK,
    "probabilities": _PROBABILITIES,
    "teacher": _LOGITS,
}
_TARGET_FIELDS = {
    _LABELS: "labels",
    _MASK: "mask",
    _PROBABILITIES: "probabilities",
    _LOGITS: "logits",
}


@cache
def _reduce_rows() -> Any:
    function = _native().lib.sdca_reduce_rows
    function.restype, function.argtypes = ct.c_int, [_i, _i, _s, _p, _p, _p]
    return function


def reduce_rows(
    dtype: str, reduction: int, n: int, values: int, terms: int | None, out: int
) -> None:
    """Finalize double row values on the framework's own buffers, with IEEE subnormals."""
    _reduce_rows()
    _native().call("sdca_reduce_rows", ROW_DTYPES[dtype], reduction, n, values, terms, out)


@cache
def _grouped_reducer() -> Any:
    function = _native().lib.sdca_reduce_row_groups
    function.restype, function.argtypes = ct.c_int, [_i, _i, _s, _s, _p, _p, _p]
    return function


def reduce_row_groups(
    dtype: str, reduction: int, groups: int, rows: int, values: int, terms: int | None, out: int
) -> None:
    try:
        _grouped_reducer()
    except AttributeError:
        # A caller may supply an older libsdca_c with only the scalar reducer.
        itemsize = ct.sizeof(ct.c_double if dtype == "float64" else ct.c_float)
        for i in range(groups):
            offset = i * rows * ct.sizeof(ct.c_double)
            reduce_rows(
                dtype,
                reduction,
                rows,
                values + offset,
                terms + offset if terms else None,
                out + i * itemsize,
            )
        return
    _native().call(
        "sdca_reduce_row_groups", ROW_DTYPES[dtype], reduction, groups, rows, values, terms, out
    )


@cache
def _loss_rows() -> Any:
    function = _native().lib.sdca_loss_rows
    # loss, options, execution, dtype, n, classes, scores, targets, values, gradients
    function.restype, function.argtypes = ct.c_int, [_p, _p, _p, _i, _s, _s, _p, _p, _p, _p]
    return function


def rows_reason() -> str | None:
    """None where the native library computes the rows of the frameworks' losses, else why
    not."""
    try:
        _loss_rows()
        _reduce_rows()
    except AttributeError:
        return f"the native library {_library_path()} lacks the row kernels or reducer (rebuild it)"
    except (OSError, RuntimeError) as error:
        return str(error)
    return None


@cache
def _row_execution(precision: int, threads: int) -> _Execution:
    return _make(_Execution, precision=precision, num_threads=threads)


def loss_rows(
    rows: _f.Rows,
    threads: int,
    dtype: str,
    n: int,
    classes: int,
    scores: int,
    target: int,
    role: str,
    probability_dtype: str,
    weight: int | None,
    upstream: float,
    row_upstream: int | None,
    values: int | None,
    gradients: int | None,
) -> None:
    """sdca_loss_rows on buffers at these addresses: n rows of scores of dtype, the target of
    a role (ROW_TARGETS; probabilities of probability_dtype), class weights (float64, or
    None), the gradients' factors upstream and row_upstream (n float64, or None), and the
    outputs values (n float64) and gradients (the scores' dtype; may be scores), either None.
    Invalid rows give NaN rows; ValueError or RuntimeError for the errors of the library."""
    kind = ROW_TARGETS[role]
    loss = _make(
        _Loss,
        kind=rows.kind,
        k=rows.k if _f.NAMES[rows.kind].startswith("topk_") else 0,
        gamma=rows.gamma,
        temperature=rows.temperature,
        label_smoothing=rows.label_smoothing,
        weight=weight,
        ignore=int(rows.ignore),
        ignore_index=rows.ignore_index if rows.ignore else -100,
    )
    options = _make(
        _RowOptions,
        probability_dtype=ROW_DTYPES[probability_dtype],
        upstream=upstream,
        row_upstream=row_upstream,
    )
    targets = _make(_Targets, kind=kind, **{_TARGET_FIELDS[kind]: target})
    _loss_rows()
    invalid_rows = _native().call(
        "sdca_loss_rows",
        ct.byref(loss),
        ct.byref(options),
        ct.byref(_row_execution(rows.precision, threads)),
        ROW_DTYPES[dtype],
        n,
        classes,
        scores,
        ct.byref(targets),
        values,
        gradients,
    )
    if invalid_rows and kind == _LABELS:
        for label in (ct.c_int64 * n).from_address(target):
            if (label < 0 or label >= classes) and not (rows.ignore and label == rows.ignore_index):
                raise IndexError(f"Target {label} is out of bounds.")


def coordinate_update(
    logits: npt.ArrayLike,
    target: Targets,
    loss: Loss = _CROSS_ENTROPY,
    *,
    previous: npt.ArrayLike | None = None,
    C: float = 1.0,
    norm2: float | npt.ArrayLike = 1.0,
    precision: Precision = None,
    num_threads: int | None = None,
) -> FloatArray:
    """New dual blocks of one coordinate step of a training objective (not gradients).

    For CrossEntropy, A_new / C = one_hot(y) - softmax(logits + norm2 * (A_new - previous))
    at the block optimum; the other OBJECTIVES return their own blocks.

    Args:
        logits: Current scores (n, classes), classes >= 2, finite.
        target: Integer labels (n,), label sets for the multilabel objectives, or
            Probabilities for CrossEntropy (the distillation objective).
        loss: A training objective (a loss named in OBJECTIVES).
        previous: Current dual blocks (n, classes), feasible for the objective (as a fit's
            duals are; checked); zeros if None.
        C: Loss weight, finite and positive.
        norm2: Squared norm of each example: a scalar or an array (n,), finite and positive.
        precision: Working precision, as in loss_and_grad.
        num_threads: Threads for the rows, as in loss_and_grad.

    Returns:
        New dual blocks (n, classes) in the storage dtype.

    Raises:
        TypeError, ValueError, RuntimeError: As loss_and_grad.
    """
    loss = _check_loss(loss, objective=True)
    z, work = _scores(logits, precision)
    n = len(z)
    _f.check_classes(getattr(loss, "k", 1), z.shape[1])
    targets = _targets(loss, target, n, scores=z)
    c = _f.check_real(C, "C")
    norms = _dense(norm2, "norm2").astype(np.float64)
    if norms.ndim == 0:
        norms = np.full(n, float(norms))
    if norms.shape != (n,):
        raise ValueError(f"norm2 must be a scalar or an array of shape ({n},)")
    norms = np.ascontiguousarray(norms)
    a = None
    if previous is not None:
        a = _cast(_dense(previous, "previous"), z.dtype.type)
        if a.shape != z.shape:
            raise ValueError("previous must have the shape of logits")
    execution = _execution(work, num_threads)
    out = np.empty_like(z)
    struct = loss._struct()
    _native().call(
        "sdca_dual_update",
        ct.byref(struct),
        c,
        ct.byref(execution),
        _dtype(z),
        *z.shape,
        z.ctypes.data,
        targets.ref,
        None if a is None else a.ctypes.data,
        norms.ctypes.data,
        out.ctypes.data,
    )
    return out


def _log_softmax(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Row log-softmax with the largest score left out of the sum (log1p): the largest
    class keeps a log-probability below zero where the others are representable."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    others = np.exp(shifted)
    others[np.arange(len(scores)), scores.argmax(axis=1)] = 0.0
    return shifted - np.log1p(others.sum(axis=1, keepdims=True))


@dataclass(frozen=True, eq=False)
class Model:
    """The result of fit.

    Attributes:
        loss: The loss trained.
        C: The loss weight.
        num_classes: The number of classes m.
        kernel: Whether the model was trained on a Gram matrix (Kernel input).
        weights: Primal weights W (m, d) in the storage dtype, one row per class; None
            for a kernel.
        dual: Dual variables A (n, m) in the storage dtype.
        status: One of STATUS; only "solved" guarantees the duality gap.
        epochs: Number of epochs run.
        history: One array per HISTORY field, one entry per evaluation (accuracy is the
            rank loss for the multilabel losses).
    """

    loss: Loss
    C: float
    num_classes: int
    kernel: bool
    weights: FloatArray | None
    dual: FloatArray
    status: str
    epochs: int
    history: dict[str, npt.NDArray[np.float64]]

    def decision_function(self, x: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Scores (n_test, m) of features x (n_test, d), or of the cross-kernel
        K(test, train) (n_test, n_train) for a kernel model."""
        a = _dense(x, "x")
        coefficients = self.dual if self.weights is None else self.weights.T
        columns = coefficients.shape[0]
        if a.ndim != 2 or a.shape[1] != columns:
            raise ValueError(f"x must have shape (n_test, {columns})")
        return a @ coefficients

    def predict(self, x: npt.ArrayLike) -> npt.NDArray[np.intp]:
        """Predicted labels (n_test,): the argmax of decision_function(x)."""
        return self.decision_function(x).argmax(axis=1)

    def score(self, x: npt.ArrayLike, y: Targets) -> float:
        """Top-1 accuracy: the fraction of examples whose predicted class is their label
        (integer labels y), or lies in their label set (multilabel losses: a boolean mask or
        label sets)."""
        predicted = self.predict(x)
        n = len(predicted)
        if self.loss.name not in MULTILABEL:
            return float(np.mean(predicted == _labels(y, n))) if n else 0.0
        first, offsets = _label_sets(y, n)
        if offsets is None:
            if first.shape[1] != self.num_classes:
                raise ValueError(f"A label mask must have {self.num_classes} columns")
            hits = first[np.arange(n), predicted].astype(bool)
        else:
            hits = np.array(
                [predicted[i] in first[offsets[i] : offsets[i + 1]] for i in range(n)], bool
            )
        return float(np.mean(hits)) if n else 0.0

    def predict_log_proba(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Log-probabilities (n_test, m): the log-softmax of decision_function(x) (the
        cross-entropy losses only)."""
        if self.loss.name not in _PROBABILISTIC:
            raise ValueError("predict_log_proba needs a cross-entropy loss")
        return _log_softmax(np.asarray(self.decision_function(x), dtype=np.float64))

    def predict_proba(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Class probabilities (n_test, m): the softmax of decision_function(x) (the
        cross-entropy losses only)."""
        return np.exp(self.predict_log_proba(x))


def fit(
    X: npt.ArrayLike | Kernel,
    y: Targets,
    loss: Loss = _CROSS_ENTROPY,
    *,
    C: float = 1.0,
    tol: float = 1e-3,
    max_epoch: int = 1000,
    eval_epoch: int = 10,
    eval_on_start: bool = False,
    recompute_primal: bool = True,
    shrinking: str = "auto",
    seed: int = 0,
    num_classes: int | None = None,
    center: npt.ArrayLike | None = None,
    initial_dual: npt.ArrayLike | None = None,
    precision: Precision = None,
    num_threads: int | None = None,
) -> Model:
    """Train a linear (or kernel) classifier by SDCA on one of the OBJECTIVES.

    Minimizes 0.5 * ||W - center||^2 + C * sum_i loss(X[i] @ W, y[i]) (no intercept).

    Args:
        X: Features (n, d), or Kernel(K) for a Gram matrix (n, n).
        y: Integer labels (n,) for the single-label losses; a boolean mask (n, m) or a
            sequence of integer label sets for the multilabel losses; Probabilities (n, m)
            for CrossEntropy (the distillation objective).
        loss: A training objective, such as CrossEntropy() or TopKHinge(k=2).
        C: Loss weight, finite and positive.
        tol: The relative duality gap of status "solved", finite and >= 0 (0: never).
        max_epoch: Epoch limit, an integer >= 0.
        eval_epoch: Epochs between evaluations (0: evaluate only at the end); up to two more
            fall between where the gap's trend reaches tol.
        eval_on_start: Also evaluate before the first epoch.
        recompute_primal: Rebuild W = center + A.T @ X at each evaluation (features).
        shrinking: What the epochs of the hinge losses leave out: "auto" (the default,
            "both"), "off", "heuristic", "safe" or "both" (docs/API.md#shrinking). The
            other losses update every class of every example and take "auto" only.
        seed: The order of the examples in the epochs, an integer in [0, 2**64).
        num_classes: Number of classes (>= 2); None infers it from y.
        center: Center (m, d) of the regularizer (features only); None is zero.
        initial_dual: Warm start (n, m), usually model.dual of an earlier fit; the library
            checks it against the objective's feasible set and builds W from it.
        precision: Working precision, as in loss_and_grad; with float32 storage, "float64"
            keeps the iterate in double and returns it rounded once.
        num_threads: Threads of the evaluations and checks (the epochs are sequential).

    Returns:
        The trained Model.

    Raises:
        TypeError: For arguments of a wrong type.
        ValueError: For invalid values.
        RuntimeError: If the native solver fails.
    """
    loss = _check_loss(loss, objective=True)
    kernel = isinstance(X, Kernel)
    x = _dense(X.gram if isinstance(X, Kernel) else X, "X")
    storage, work = _storage(x, precision)
    x = _cast(x, storage)
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("X must be a nonempty matrix")
    if kernel and x.shape[0] != x.shape[1]:
        raise ValueError("A Gram matrix must be square")
    n, d = x.shape
    targets = _targets(loss, y, n)
    classes = _fit_classes(targets, num_classes)
    _f.check_classes(getattr(loss, "k", 1), classes)
    options = _make(
        _FitOptions,
        c=_f.check_real(C, "C"),
        tol=_f.check_real(tol, "tol"),
        max_epoch=_f.check_integer(max_epoch, "max_epoch", 0),
        eval_epoch=_f.check_integer(eval_epoch, "eval_epoch", 0),
        eval_on_start=_flag(eval_on_start, "eval_on_start"),
        recompute_primal=_flag(recompute_primal, "recompute_primal"),
        shrinking=_f.SHRINKING[_f.check_choice(shrinking, "shrinking", _f.SHRINKING)],
        seed=_f.check_integer(seed, "seed", 0),
    )
    if options.seed != seed:
        raise ValueError("seed must be below 2**64")
    c0, a0 = _warm_start(center, initial_dual, kernel, storage, n, d, classes)
    weights, dual, history, result = _solve_fit(
        x, targets, loss, options, classes, kernel, c0, a0, work, num_threads
    )
    columns = history[: result.history_rows].T.copy()
    return Model(
        loss=loss,
        C=float(C),
        num_classes=classes,
        kernel=kernel,
        weights=weights,
        dual=dual,
        status=STATUS[result.status],
        epochs=int(result.epochs),
        history={name: columns[j] for j, name in enumerate(HISTORY)},
    )


def _warm_start(
    center: npt.ArrayLike | None,
    initial_dual: npt.ArrayLike | None,
    kernel: bool,
    storage: type[np.floating],
    n: int,
    d: int,
    classes: int,
) -> tuple[FloatArray | None, FloatArray | None]:
    c0 = None
    if center is not None:
        if kernel:
            raise ValueError("center applies to feature input only")
        c0 = _cast(_dense(center, "center"), storage)
        if c0.shape != (classes, d):
            raise ValueError(f"center must have shape ({classes}, {d})")
    a0 = None
    if initial_dual is not None:
        a0 = _cast(_dense(initial_dual, "initial_dual"), storage)
        if a0.shape != (n, classes):
            raise ValueError(f"initial_dual must have shape ({n}, {classes})")
    return c0, a0


def _solve_fit(
    x: FloatArray,
    targets: _TargetArgs,
    loss: Loss,
    options: _FitOptions,
    classes: int,
    kernel: bool,
    c0: FloatArray | None,
    a0: FloatArray | None,
    work: int,
    num_threads: int | None,
) -> tuple[FloatArray | None, FloatArray, np.ndarray, _FitResult]:
    """Allocate the native training outputs and run the validated problem."""
    n, d = x.shape
    storage = x.dtype
    native = _native()
    limit = native.lib.sdca_max_extent()
    if max(n, d, classes, n * classes, d * classes) > limit:
        raise ValueError("Dimensions exceed the native BLAS integer limit")
    rows = native.lib.sdca_history_rows(ct.byref(options))
    if rows == 0:
        raise ValueError(native.lib.sdca_last_error().decode())
    try:
        history = np.empty((rows, len(HISTORY)), np.float64)
    except (MemoryError, ValueError) as error:
        raise ValueError("max_epoch / eval_epoch is too large for the history") from error
    weights = None if kernel else np.empty((classes, d), storage)
    dual = np.empty((n, classes), storage)
    data = _make(
        _Input, kind=_KERNEL if kernel else _FEATURES, dtype=_dtype(x), n=n, d=d, x=x.ctypes.data
    )
    struct = loss._struct()
    execution = _execution(work, num_threads)
    result = _make(_FitResult)
    native.warn_without_blas()
    native.call(
        "sdca_model_fit",
        ct.byref(struct),
        ct.byref(options),
        ct.byref(execution),
        ct.byref(data),
        classes,
        targets.ref,
        None if c0 is None else c0.ctypes.data,
        None if a0 is None else a0.ctypes.data,
        None if weights is None else weights.ctypes.data,
        dual.ctypes.data,
        history.ctypes.data,
        rows,
        ct.byref(result),
    )
    return weights, dual, history, result


def _fit_classes(targets: _TargetArgs, num_classes: int | None) -> int:
    """The class count of fit: num_classes, checked against the targets, or inferred."""
    if num_classes is None:
        classes = targets.num_classes()
        if classes is None:
            raise ValueError("num_classes cannot be inferred from empty label sets")
    else:
        classes = _f.check_integer(num_classes, "num_classes", 2)
        implied = targets.num_classes() if targets.kind != _LABELS else None
        if targets.kind != _LABEL_SETS and implied not in (None, classes):
            raise ValueError(f"The targets have {implied} classes, not num_classes={classes}")
    return classes
