"""NumPy interface to the native solvers and losses (through ctypes).

fit, loss_and_grad and coordinate_update use a shared Solver of the precision
implied by the input dtype; a Solver binds one precision and native library.

Arrays use conventional Python layouts: X (n,d), W (d,m), A (n,m) for m classes.
Objective = 0.5*||W||^2 + C*sum_i loss(X[i]@W, y[i]).
No implicit intercept; append a constant feature for a regularized intercept.
"""

from __future__ import annotations

import ctypes as ct
import math
import os
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from numbers import Integral, Real
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]
IndexArray = npt.NDArray[np.int64]
# Integer labels (n,), a boolean mask (n,m), or one label list per example.
Targets = npt.ArrayLike | Sequence[npt.ArrayLike]
_T = TypeVar("_T")

# Compute precisions: "float32" (single arithmetic) or "float64" (double arithmetic).
# Native C ABI codes: 0 = <float, float>, 1 = <float, double>, 2 = <double, double>.
PRECISIONS: tuple[str, ...] = ("float32", "float64")
_PRECISION_CODES: dict[tuple[str, type[np.floating]], int] = {
    ("float32", np.float32): 0,
    ("float64", np.float32): 1,
    ("float64", np.float64): 2,
}
_STR_PRECISIONS: dict[str, str] = {
    "float32": "float32",
    "fp32": "float32",
    "single": "float32",
    "float64": "float64",
    "fp64": "float64",
    "double": "float64",
}
ABI_VERSION = 5  # SDCA_ABI_VERSION of include/sdca/c_api.h
_LIBRARY_NAME = "libsdca_c.so"  # python/build.py output in build/python or SDCA_LIBRARY_DIR
HISTORY = (
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
STATUS = (
    "none",
    "solving",
    "solved",
    "no_progress",
    "max_epoch",
    "max_cpu_time",
    "max_wall_time",
    "failed",
)
# Loss and objective names in the order of the C ABI ids (sdca_loss); every
# name but the last is a training objective. The hinges are smoothed by gamma > 0.
LOSSES = (
    "cross_entropy",
    "topk_cross_entropy",
    "multiclass_hinge",
    "topk_hinge_alpha",
    "topk_hinge_beta",
    "multilabel_hinge",
    "multilabel_cross_entropy",
    "proximal_cross_entropy",
)
OBJECTIVES = LOSSES[:-1]
_IDS = {name: i for i, name in enumerate(LOSSES)}
MULTILABEL = frozenset(("multilabel_hinge", "multilabel_cross_entropy"))
TOPK = frozenset(("topk_cross_entropy", "topk_hinge_alpha", "topk_hinge_beta"))
HINGES = frozenset(("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge"))
_INVALID_ARGUMENT = -2  # SDCA_INVALID_ARGUMENT
# projection_precision names and their SDCA_PROJECTION_* codes.
_PROJECTIONS = {"float64": 0, "extended": 1}
# shrinking names and their SDCA_SHRINKING_* codes.
_SHRINKING = {"auto": 0, "off": 1, "heuristic": 2, "safe": 3, "both": 4}
_THREADS_DEFAULT = ct.c_size_t(-1).value  # SDCA_THREADS_DEFAULT


class _Options(ct.Structure):
    """struct sdca_options of include/sdca/c_api.h."""

    _fields_ = [
        ("size", ct.c_size_t),
        ("precision", ct.c_int),
        ("loss", ct.c_int),
        ("c", ct.c_double),
        ("k", ct.c_size_t),
        ("gamma", ct.c_double),
        ("alpha", ct.c_double),
        ("projection_precision", ct.c_int),
        ("num_threads", ct.c_size_t),
        ("max_epoch", ct.c_size_t),
        ("eval_epoch", ct.c_size_t),
        ("tol", ct.c_double),
        ("eval_on_start", ct.c_int),
        ("recompute_primal", ct.c_int),
        ("kernel", ct.c_int),
        ("warm", ct.c_int),
        ("shrinking", ct.c_int),
    ]


class _FitResult(ct.Structure):
    """struct sdca_fit_result of include/sdca/c_api.h."""

    _fields_ = [
        ("size", ct.c_size_t),
        ("status", ct.c_int),
        ("epochs", ct.c_size_t),
        ("history_used", ct.c_size_t),
    ]


# SDCA_OPTIONS_DEFAULT of include/sdca/c_api.h.
_DEFAULT_OPTIONS: dict[str, Any] = dict(
    precision=2,
    loss=0,
    c=1.0,
    k=1,
    gamma=0.0,
    alpha=1.0,
    projection_precision=0,
    num_threads=_THREADS_DEFAULT,
    max_epoch=1000,
    eval_epoch=10,
    tol=1e-3,
    eval_on_start=0,
    recompute_primal=1,
    kernel=0,
    warm=0,
    shrinking=0,
)


def _options(**fields: Any) -> _Options:
    """struct sdca_options: SDCA_OPTIONS_DEFAULT with the given fields."""
    return _Options(size=ct.sizeof(_Options), **{**_DEFAULT_OPTIONS, **fields})


def _fit_result() -> _FitResult:
    return _FitResult(size=ct.sizeof(_FitResult))


def _check_name(name: object, names: tuple[str, ...], what: str) -> str:
    """name if it is one of names ('topk_hinge' is 'topk_hinge_beta', 'distillation' is a fit/update objective), else ValueError."""
    if name == "topk_hinge":
        name = "topk_hinge_beta"
    if isinstance(name, str) and (
        name in names or (what == "objective" and name == "distillation")
    ):
        return name
    hint = "; the hinges are smoothed by gamma > 0" if str(name).endswith("_smooth") else ""
    raise ValueError(f"Unknown {what}: {name!r}{hint}")


_BASE_KINDS = {"multiclass_hinge": "topk_hinge_alpha", "multilabel_cross_entropy": "cross_entropy"}


def _kind(name: str, gamma: float) -> str:
    """The tensor backends' kind of a loss: the top-k hinge of multiclass_hinge,
    cross_entropy for multilabel_cross_entropy, and a _smooth suffix if gamma > 0."""
    base = _BASE_KINDS.get(name, name)
    return base + "_smooth" if gamma > 0 and name in HINGES else base


def _loss_of_kind(kind: str) -> tuple[str, float | None]:
    """The loss name of a tensor-backend kind, and gamma 0 if the kind is nonsmooth."""
    return kind.removesuffix("_smooth"), None if kind.endswith("_smooth") else 0.0


# Each loss, bound once: (kind, kind for gamma > 0 or None if not a hinge, whether k is 1).
_Entry = tuple[str, "str | None", bool]
_KINDS: dict[str, _Entry] = {
    name: (
        _kind(name, 0.0),
        _kind(name, 1.0) if name in HINGES else None,
        name == "multiclass_hinge",
    )
    for name in LOSSES
}
_KINDS["topk_hinge"] = _KINDS["topk_hinge_beta"]


def _entry(loss: object) -> _Entry:
    """The bound kinds of a loss name: one table lookup, else the name error."""
    entry = _KINDS.get(loss) if type(loss) is str else None
    return entry if entry is not None else _KINDS[_check_name(loss, LOSSES, "loss")]


def _kind_of(entry: _Entry, k: int, gamma: float) -> tuple[str, int, float]:
    """The tensor-backend kind of a bound loss, its k (1 for multiclass_hinge) and its
    gamma (1 for a nonsmooth hinge). Callers check the given k with _check_options: the
    k of multiclass_hinge is 1 whatever the given k."""
    exact = type(gamma) is float or type(gamma) is int
    if not (exact and 0 <= gamma < math.inf) and (
        not isinstance(gamma, Real) or not math.isfinite(gamma) or not gamma >= 0
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
    """Normalize a precision string or floating-point dtype object to 'float32' or 'float64'."""
    if precision is None:
        return None
    out: str | None = None
    if isinstance(precision, str):
        out = _STR_PRECISIONS.get(precision)
    elif precision is float or precision in (np.float64, np.dtype(np.float64)):
        out = "float64"
    elif precision in (np.float32, np.dtype(np.float32)):
        out = "float32"
    elif type(precision).__module__ == "torch" and str(precision) in (
        "torch.float32",
        "torch.float64",
    ):
        out = str(precision).split(".")[1]
    else:
        dt = getattr(precision, "dtype", None)
        if isinstance(dt, np.dtype) and dt.name in PRECISIONS:
            out = dt.name
    if out is None:
        raise ValueError(f"precision must be None or one of {', '.join(PRECISIONS)}")
    return out


def _topk_hinge_name(variant: str) -> str:
    """The top-k hinge loss of variant 'alpha' or 'beta'."""
    if variant in ("alpha", "beta"):
        return "topk_hinge_" + variant
    raise ValueError("variant must be 'alpha' or 'beta'")


def _check_options(k: int, gamma: float, alpha: float, classes: int) -> None:
    """The loss options of the tensor layers, before any computation."""
    fast = type(k) is int and 1 <= k < classes and type(gamma) is float
    if fast and 0.0 <= gamma < math.inf and 0.0 < alpha < math.inf:
        return
    if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= int(k) < classes:
        raise ValueError("k must be an integer in [1, classes-1]")
    if not math.isfinite(gamma) or not gamma >= 0:
        raise ValueError("gamma must be finite and nonnegative")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")


def _ptr(a: npt.NDArray[np.generic] | None) -> ct.c_void_p | None:
    """The address of a, which the caller must keep alive through the native call."""
    return None if a is None else ct.c_void_p(a.ctypes.data)


def _numpy(x: _T) -> _T | npt.NDArray[np.generic]:
    """Detach a torch tensor (any device) without importing torch."""
    detach = getattr(x, "detach", None) if type(x).__module__.startswith("torch") else None
    return x if detach is None else detach().cpu().numpy()


def _rows(y: Targets) -> Sequence[npt.ArrayLike]:
    """View label-set input as per-example rows (len() rejects scalars)."""
    return cast("Sequence[npt.ArrayLike]", y)


def _check_native_storage(x: object) -> None:
    """Reject unsupported half storage before conversion hides the input dtype."""
    if str(getattr(x, "dtype", "")).endswith(("float16", "bfloat16")):
        raise ValueError(
            "Native APIs require float32 or float64 storage; "
            "convert float16/bfloat16 inputs explicitly"
        )


def _array(x: npt.ArrayLike, dtype: type[np.floating]) -> FloatArray:
    _check_native_storage(x)
    a = np.require(_numpy(x), dtype=dtype, requirements=["C", "A"])
    if not np.isfinite(a).all():
        raise ValueError("Inputs must be finite")
    return a


def _labels(y: Targets, n: int, k: int) -> IndexArray:
    """One label in [0, k) per example as a contiguous int64 array (n,)."""
    labels = np.asarray(_numpy(y))
    if labels.shape != (n,) or labels.dtype.kind not in "iu":
        raise ValueError("y must be a one-dimensional integer label array")
    if np.any(labels < 0) or np.any(labels >= k):
        raise ValueError("Labels must lie in [0, num_classes)")
    return np.require(labels, dtype=np.int64, requirements=["C", "A"])


def _bool_rows(y: Targets, k: int | None = None) -> bool:
    """True for a nonempty non-array list of boolean rows (of length k if given)."""
    rows = _rows(y)
    try:
        return (
            not isinstance(y, np.ndarray)
            and len(rows) > 0
            and all(
                (k is None or len(_rows(row)) == k)
                and all(isinstance(x, bool | np.bool_) for x in _rows(row))
                for row in rows
            )
        )
    except TypeError:  # scalar rows: labels, not a mask
        return False


def _label_mask(y: Targets, k: int) -> npt.NDArray[np.bool_] | None:
    """y as a boolean mask, or None if y holds label lists."""
    mask = np.asarray(y) if _bool_rows(y, k) else y
    if isinstance(mask, np.ndarray) and mask.dtype == np.bool_:
        return mask
    return None


def _label_row(row: npt.ArrayLike, k: int) -> npt.NDArray[np.integer]:
    """One sorted label list in [0, k) without duplicates."""
    labels = np.atleast_1d(np.asarray(_numpy(row)))
    if labels.size and (labels.dtype.kind not in "iu" or labels.min() < 0 or labels.max() >= k):
        raise ValueError("Labels must lie in [0, num_classes)")
    if len(np.unique(labels)) != labels.size:
        raise ValueError("Label sets must not contain duplicates")
    return np.sort(labels)


def _label_rows(y: Targets, n: int, k: int) -> list[npt.NDArray[np.integer]]:
    """Per-example label lists from a (n,m) mask or a list of label lists."""
    y = _numpy(y)
    mask = _label_mask(y, k)
    if mask is not None:
        if mask.shape != (n, k):
            raise ValueError("A label mask must have shape (n, num_classes)")
        return [np.flatnonzero(row) for row in mask]
    if len(_rows(y)) != n:
        raise ValueError("y must have one label set per example")
    return [_label_row(row, k) for row in _rows(y)]


def _label_sets(
    y: Targets, n: int, k: int, largest: int | None = None
) -> tuple[IndexArray, IndexArray]:
    """Concatenated sorted label lists (sum of sizes,) and their offsets (n+1,).

    Each list has between 1 and largest (default k-1, training) labels.
    """
    y = _numpy(y)
    mask = _label_mask(y, k)
    if mask is not None and mask.shape == (n, k):
        # Row-major nonzeros: each row's labels in increasing order, no Python loop.
        sizes = mask.sum(1, dtype=np.int64)
        labels = np.nonzero(mask)[1]
    else:
        rows = _label_rows(y, n, k)
        sizes = np.array([len(r) for r in rows], dtype=np.int64)
        labels = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    largest = k - 1 if largest is None else largest
    if np.any(sizes < 1) or np.any(sizes > largest):
        raise ValueError(f"Each example needs between 1 and {largest} labels")
    offsets = np.zeros(n + 1, np.int64)
    np.cumsum(sizes, out=offsets[1:])
    return np.ascontiguousarray(labels, dtype=np.int64), offsets


def _positive_mask(
    labels: IndexArray, offsets: IndexArray | None, n: int, classes: int
) -> npt.NDArray[np.bool_]:
    """Boolean (n,classes) mask of the labels; one per row if offsets is None."""
    positive = np.zeros((n, classes), bool)
    rows = np.arange(n) if offsets is None else np.repeat(np.arange(n), np.diff(offsets))
    positive[rows, labels] = True
    return positive


def _row_message(message: str, loss: str) -> str:
    """The message of sdca_loss_gradient for one of sdca_loss_gradient_rows."""
    if message == "This loss takes one positive label per row":
        return "This loss needs exactly one target"
    if message == "This loss needs a negative label per row":
        if loss == "proximal_cross_entropy":
            return "Proximal entropy needs a non-target class"
        return "Multilabel hinge needs a negative label"
    return message


def _infer_classes(y: Targets, multilabel: bool) -> int:
    """Number of classes implied by the targets: mask width or largest label + 1."""
    y = _numpy(y)
    if multilabel and isinstance(y, np.ndarray) and y.dtype == np.bool_ and y.ndim == 2:
        return y.shape[1]
    if multilabel and _bool_rows(y):
        return len(_rows(_rows(y)[0]))
    try:
        flat = (
            np.concatenate([np.atleast_1d(np.asarray(r)) for r in _rows(y)])
            if multilabel
            else np.asarray(y)
        )
        return int(np.max(flat)) + 1
    except (TypeError, ValueError):
        raise ValueError("y must be integer labels, a (n,m) boolean mask, or label lists") from None


def _is_int(value: object) -> bool:
    return not isinstance(value, bool | np.bool_) and isinstance(value, int | np.integer)


def _num_classes(y: Targets, num_classes: int | None, multilabel: bool) -> int:
    if num_classes is not None and not _is_int(num_classes):
        raise ValueError("num_classes must be an integer")
    classes = _infer_classes(y, multilabel) if num_classes is None else int(num_classes)
    if classes < 2:
        raise ValueError("num_classes must be >=2")
    return classes


def _label_vector(y: Targets) -> bool:
    """True for one integer label per example (anything else is a label set)."""
    try:
        labels = np.asarray(_numpy(y))
    except ValueError:
        return False
    return labels.ndim == 1 and labels.dtype.kind in "iu"


def _check_label_array(y: Targets, objective: str) -> None:
    """Reject float labels, and a (n,1) integer array where cross_entropy would
    have to guess between labels and label sets."""
    if objective == "distillation":
        return
    labels = _numpy(y)
    if not isinstance(labels, np.ndarray):  # lists may hold label sets
        try:
            labels = np.asarray(labels)
        except ValueError:  # ragged label lists
            return
        if labels.dtype.kind in "iub":
            return
    if labels.dtype.kind in "fc":
        raise ValueError("y must hold integer labels, not floating-point values")
    if objective == "cross_entropy" and labels.ndim == 2 and labels.shape[1] == 1:
        raise ValueError(
            "y of shape (n, 1) is ambiguous: pass y.ravel() for one label per example, "
            "or a boolean mask or label lists for label sets"
        )


def _distribution(y: Targets, n: int, classes: int | None) -> tuple[npt.NDArray[np.float64], int]:
    """Validated soft-target probability matrix (n, classes) in float64."""
    _check_native_storage(y)
    q = np.require(_numpy(y), dtype=np.float64, requirements=["C", "A"])
    if q.ndim != 2 or len(q) != n or q.shape[1] < 2:
        raise ValueError("distillation targets must have shape (n, num_classes>=2)")
    if classes is not None and q.shape[1] != classes:
        raise ValueError(f"distillation targets must have {classes} columns")
    if not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("Target probabilities must be finite and nonnegative")
    if not np.allclose(
        q.sum(axis=1), 1.0, rtol=0.0, atol=64 * np.finfo(np.float64).eps * q.shape[1]
    ):
        raise ValueError("Target probability rows must sum to one")
    return q, int(q.shape[1])


def _resolve_objective(objective: str, y: Targets) -> str:
    """Cross-entropy on label sets uses uniform targets: multilabel_cross_entropy."""
    objective = _check_name(objective, OBJECTIVES, "objective")
    if objective == "distillation":
        return objective
    _check_label_array(y, objective)
    if objective == "cross_entropy" and not _label_vector(y):
        return "multilabel_cross_entropy"
    return objective


def _positive(value: float, name: str) -> None:
    # By the bits: a thread that reads subnormals as zero (an XLA:CPU host callback) would
    # find 1e-310 <= 0.
    try:
        bits = struct.unpack("<q", struct.pack("<d", value))[0]
    except (struct.error, TypeError):
        bits = 0
    if not 0 < bits < 0x7FF0000000000000:
        raise ValueError(f"{name} must be finite and positive")


def _nonnegative(value: float, name: str) -> None:
    if value != 0:
        try:
            _positive(value, name)
        except ValueError:
            raise ValueError(f"{name} must be finite and nonnegative") from None


def _check_k(k: int, classes: int) -> None:
    if not _is_int(k) or not 1 <= k < classes:
        raise ValueError("k must be an integer in [1, classes-1]")


def _threads(num_threads: int | None) -> int:
    """The native num_threads: SDCA_THREADS_DEFAULT for None (in a PyTorch
    DataLoader worker PyTorch's count, 1 there), else a count >= 0."""
    if num_threads is None:
        data = sys.modules.get("torch.utils.data")
        if data is None or "SDCA_NUM_THREADS" in os.environ or data.get_worker_info() is None:
            return _THREADS_DEFAULT
        return int(sys.modules["torch"].get_num_threads())
    if not _is_int(num_threads) or num_threads < 0:
        raise ValueError("num_threads must be None or a nonnegative integer")
    return int(num_threads)


def _projection(projection_precision: object) -> int:
    """The SDCA_PROJECTION_* code of a projection_precision name."""
    code = _PROJECTIONS.get(projection_precision) if isinstance(projection_precision, str) else None
    if code is None:
        raise ValueError("projection_precision must be 'float64' or 'extended'")
    return code


def _shrinking(shrinking: object) -> int:
    """The SDCA_SHRINKING_* code of a shrinking name."""
    code = _SHRINKING.get(shrinking) if isinstance(shrinking, str) else None
    if code is None:
        raise ValueError("shrinking must be one of " + ", ".join(map(repr, _SHRINKING)))
    return code


def _check_schedule(max_epoch: int, eval_epoch: int, tol: float) -> None:
    if not _is_int(max_epoch) or max_epoch < 0 or not _is_int(eval_epoch) or eval_epoch < 1:
        raise ValueError("max_epoch must be >=0 and eval_epoch >=1 integers")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")


def _check_native_size(n: int, d: int, classes: int) -> None:
    if max(n, d, classes, d * classes, n * classes) > np.iinfo(np.int32).max:
        raise ValueError("Dimensions exceed the native BLAS integer limit")


def _validate_initial(
    a: FloatArray,
    labels: IndexArray,
    offsets: IndexArray | None,
    C: float,
    objective: str,
    k: int,
    targets: npt.NDArray[np.float64] | None = None,
) -> None:
    """Reject dual blocks outside the objective's feasible set (to rounding)."""
    v = np.asarray(a, dtype=np.float64)
    n, classes = v.shape
    eps, slack = 32 * np.finfo(a.dtype).eps, 32 * np.finfo(a.dtype).eps * C
    # Relative to each row's largest entry: near the float64 maximum, sum |v| overflows.
    largest = np.abs(v).max(1, keepdims=True)
    u = v / np.where(largest > 0, largest, 1.0)
    if np.any(np.abs(u.sum(1)) > eps * np.abs(u).sum(1)):
        raise ValueError("initial_dual rows must sum to zero to storage precision")
    if objective == "distillation" and targets is not None:
        if np.any(v > C * targets + slack) or np.any(v < C * (targets - 1.0) - slack):
            raise ValueError("initial_dual is outside the distillation feasible set")
        return
    positive = _positive_mask(labels, offsets, n, classes)
    counts = positive.sum(1)
    # Exact: a positive non-label entry is outside the domain of every dual.
    if np.any(v[~positive] > 0):
        raise ValueError("initial_dual: non-label entries must be nonpositive")
    pos = np.where(positive, v, 0.0)
    if objective in ("cross_entropy", "multilabel_cross_entropy"):
        if np.any(pos > C / counts[:, None] + slack):
            raise ValueError("initial_dual is outside the entropy feasible set")
    else:
        if np.any(pos < -slack) or np.any(pos.sum(1) > C + slack):
            raise ValueError("initial_dual is outside the feasible set")
        if objective in ("topk_cross_entropy", "multiclass_hinge", "topk_hinge_alpha"):
            cone_slack = max(1, classes // k) * 64 * np.finfo(a.dtype).eps * C
            if np.any(-v > pos.sum(1)[:, None] / k + cone_slack):
                raise ValueError("initial_dual violates the top-k cone")
        elif objective == "topk_hinge_beta" and np.any(-v > C / k + slack):
            raise ValueError("initial_dual violates the top-k cap")


def _warm_primal(w: FloatArray, x: FloatArray, a: FloatArray, center: FloatArray | None) -> None:
    """Set w = center + x.T @ a, accumulated in float64."""
    w64 = np.asarray(x, dtype=np.float64).T @ np.asarray(a, dtype=np.float64)
    if center is not None:
        w64 += center
    w[...] = w64


def _library_path(library: str | os.PathLike[str] | None) -> str | os.PathLike[str]:
    """The explicit library, else build/python (or SDCA_LIBRARY_DIR), else the installed one."""
    if library:
        return library
    folder = Path(
        os.environ.get("SDCA_LIBRARY_DIR", Path(__file__).resolve().parents[2] / "build/python")
    )
    path = folder / _LIBRARY_NAME
    if library is None and "SDCA_LIBRARY_DIR" not in os.environ and not path.exists():
        installed = sorted(Path(__file__).resolve().parent.glob("_libsdca_native*.so"))
        if installed:
            return installed[0]
    return path


def _load(path: str | os.PathLike[str]) -> ct.CDLL:
    try:
        return ct.CDLL(str(path))
    except OSError as error:
        raise OSError(
            f"Cannot load the native library {path} ({error}). Build it with "
            "'python python/build.py', install the package, or pass library=."
        ) from error


def _declare(lib: ct.CDLL) -> None:
    """Set the ctypes signatures of the native entry points."""
    i, s, p, f = ct.c_int, ct.c_size_t, ct.c_void_p, ct.c_double
    lib.sdca_last_error.restype = ct.c_char_p
    lib.sdca_history_fields.restype = s
    lib.sdca_abi_version.restype = i
    if hasattr(lib, "sdca_version"):
        lib.sdca_version.restype = ct.c_char_p
    options, result = ct.POINTER(_Options), ct.POINTER(_FitResult)
    # options, n, d, classes, X, labels, offsets, center, W, A, history, rows, result
    lib.sdca_fit.argtypes = [options, s, s, s, p, p, p, p, p, p, p, s, result]
    lib.sdca_fit.restype = i
    # options, n, classes, logits, positive, values, gradients
    lib.sdca_loss_gradient.argtypes = [options, s, s, p, p, p, p]
    lib.sdca_loss_gradient.restype = i
    # options, n, classes, logits, labels, values, stats
    lib.sdca_entropy_forward.argtypes = [options, s, s, p, p, p, p]
    lib.sdca_entropy_forward.restype = i
    # options, n, classes, logits, labels, stats, scale, gradients
    lib.sdca_entropy_gradient.argtypes = [options, s, s, p, p, p, p, p]
    lib.sdca_entropy_gradient.restype = i
    # options, n, classes, logits, labels, offsets, previous, norm2, out
    lib.sdca_update_objective.argtypes = [options, s, s, p, p, p, p, f, p]
    lib.sdca_update_objective.restype = i
    if hasattr(lib, "sdca_update_distillation"):
        # options, n, classes, logits, targets, previous, norm2, out
        lib.sdca_update_distillation.argtypes = [options, s, s, p, p, p, f, p]
        lib.sdca_update_distillation.restype = i
        # options, n, d, classes, X, targets, center, W, A, history, rows, result
        lib.sdca_fit_distillation.argtypes = [options, s, s, s, p, p, p, p, p, p, s, result]
        lib.sdca_fit_distillation.restype = i
    # The tensor entry points take the options by address.
    # options, dtype, n, classes, logits, labels, ignore_index, weights, values, stats, totals
    lib.sdca_cross_entropy_forward.argtypes = [p, i, s, s, p, p, ct.c_int64, p, p, p, p]
    lib.sdca_cross_entropy_forward.restype = i
    # options, dtype, n, classes, logits, labels, ignore_index, weights, stats, scale, stride,
    # gradients
    lib.sdca_cross_entropy_backward.argtypes = [p, i, s, s, p, p, ct.c_int64, p, p, p, s, p]
    lib.sdca_cross_entropy_backward.restype = i
    # options, dtype, n, classes, logits, labels, positive, values, gradients
    lib.sdca_loss_gradient_rows.argtypes = [p, i, s, s, p, p, p, p, p]
    lib.sdca_loss_gradient_rows.restype = i
    # options, dtype, n, classes, student, teacher, temperature, values, stats
    lib.sdca_distillation_forward.argtypes = [p, i, s, s, p, p, f, p, p]
    lib.sdca_distillation_forward.restype = i
    # options, dtype, n, classes, student, teacher, temperature, stats, scale, stride, gradients
    lib.sdca_distillation_backward.argtypes = [p, i, s, s, p, p, f, p, p, s, p]
    lib.sdca_distillation_backward.restype = i
    # options, dtype, n, classes, student, teacher, temperature, alpha, values, gradients
    lib.sdca_proximal_distillation_rows.argtypes = [p, i, s, s, p, p, f, f, p, p]
    lib.sdca_proximal_distillation_rows.restype = i


@dataclass(frozen=True, eq=False)
class Model:
    """Result of Solver.fit.

    Attributes:
        weights: Primal weights (d,m) in the data dtype; None for kernel input.
        dual: Dual variables A (n,m) in the data dtype.
        history: One array per HISTORY field (accuracy is rank_loss for the
            multilabel objectives), one entry per evaluation.
        status: One of STATUS; only "solved" certifies the duality gap.
        epochs: Number of epochs run.
        kernel: Whether the model was trained on a Gram matrix.
        objective: The objective trained (cross_entropy on label sets becomes
            multilabel_cross_entropy).
        params: C, k, gamma and num_classes of the fit.
    """

    weights: FloatArray | None
    dual: FloatArray
    history: dict[str, npt.NDArray[np.float64]]
    status: str
    epochs: int
    kernel: bool
    objective: str = "cross_entropy"
    params: dict[str, float] = field(default_factory=dict)

    def decision_function(self, x: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Scores (n_test,k) of x (n_test,d), or of the cross-kernel (n_test,n_train).

        Raises:
            ValueError: If a feature model has no weights, or x has the wrong
                number of columns.
        """
        x = np.asarray(_numpy(x))
        if self.kernel:
            columns = self.dual.shape[0]
        elif self.weights is None:
            raise ValueError("A feature model needs weights")
        else:
            columns = self.weights.shape[0]
        if x.ndim != 2 or x.shape[1] != columns:
            raise ValueError(f"x must have {columns} columns")
        return x @ (self.dual if self.kernel else self.weights)

    def predict(self, x: npt.ArrayLike) -> npt.NDArray[np.intp]:
        """Predicted labels (n_test,): the argmax of decision_function(x)."""
        return self.decision_function(x).argmax(axis=1)

    @property
    def multilabel(self) -> bool:
        """Whether the model was trained with a multilabel objective."""
        return self.objective in MULTILABEL

    def score(self, x: npt.ArrayLike, y: Targets) -> float:
        """Mean accuracy of predict(x) on integer labels y (n_test,)."""
        scores = self.decision_function(x)
        labels = _labels(y, len(scores), scores.shape[1])
        return float(np.mean(scores.argmax(axis=1) == labels))

    def predict_log_proba(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Log-probabilities (n_test,k): log-softmax of decision_function(x)."""
        if self.objective not in ("cross_entropy", "multilabel_cross_entropy", "distillation"):
            raise ValueError("predict_log_proba is only available for cross-entropy objectives")
        return _log_softmax(np.asarray(self.decision_function(x), dtype=np.float64))

    def predict_proba(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Class probabilities (n_test,k): softmax of decision_function(x)."""
        if self.objective not in ("cross_entropy", "multilabel_cross_entropy", "distillation"):
            raise ValueError("predict_proba is only available for cross-entropy objectives")
        return np.exp(self.predict_log_proba(x))


def _log_softmax(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Row log-softmax with the largest score left out of the sum (log1p): the largest
    class keeps a log-probability below zero where the others are representable."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    others = np.exp(shifted)
    others[np.arange(len(scores)), scores.argmax(axis=1)] = 0.0
    return shifted - np.log1p(others.sum(axis=1, keepdims=True))


@dataclass(frozen=True)
class _Problem:
    """Validated training data in the layout of the native solver."""

    x: FloatArray  # (n,d) features or (n,n) Gram matrix
    labels: IndexArray  # (n,) labels or concatenated label sets
    offsets: IndexArray | None  # (n+1,) label-set offsets
    objective: str
    classes: int
    kernel: bool
    targets: npt.NDArray[np.float64] | None = None  # (n,classes) soft-target distribution

    @property
    def multilabel(self) -> bool:
        return self.objective in MULTILABEL


def _model(
    problem: _Problem,
    w: FloatArray | None,
    a: FloatArray,
    hist: npt.NDArray[np.float64],
    result: _FitResult,
    params: dict[str, float],
) -> Model:
    """Model from the outputs of sdca_fit."""
    names = tuple("rank_loss" if h == "accuracy" and problem.multilabel else h for h in HISTORY)
    columns = hist[: result.history_used].T.copy()
    history = {name: columns[j] for j, name in enumerate(names)}
    status, epochs = STATUS[result.status], int(result.epochs)
    return Model(w, a, history, status, epochs, problem.kernel, problem.objective, params)


class Solver:
    """Native solver, losses and coordinate steps of one precision and library.

    The module functions fit, loss_and_grad and coordinate_update use a shared
    Solver of the default library; create one to choose another precision or library.

    Args:
        precision: Working precision, one of PRECISIONS ("float32" or
            "float64", or a matching dtype object). float32 losses compute in
            double where float cannot hold gamma, alpha or a row's score spread.
        dtype: Storage dtype (np.float32 or np.float64); defaults to the dtype
            matching precision.
        library: Path of a native library to load instead.

    Raises:
        ValueError: For an unknown precision or dtype (before loading).
        OSError: If the library cannot be loaded.
        RuntimeError: If the library and this wrapper disagree on the C ABI
            (SDCA_ABI_VERSION) or the history layout.
    """

    def __init__(
        self,
        precision: object = "float64",
        *,
        dtype: npt.DTypeLike | None = None,
        library: str | os.PathLike[str] | None = None,
    ) -> None:
        try:
            prec = _canonical_precision(precision)
        except ValueError as error:
            raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}") from error
        if prec is None:
            raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}")
        if dtype is None:
            storage: type[np.floating] = np.float32 if prec == "float32" else np.float64
        else:
            try:
                candidate = np.dtype(dtype).type
            except TypeError as error:
                raise ValueError("dtype must be float32 or float64") from error
            if candidate not in (np.float32, np.float64):
                raise ValueError("dtype must be float32 or float64")
            storage = cast("type[np.floating]", candidate)
            if (prec, storage) not in _PRECISION_CODES:
                raise ValueError("float64 storage requires precision='float64'")
        self.code, self.dtype = _PRECISION_CODES[(prec, storage)], storage
        self.precision = prec
        self.lib = _load(_library_path(library))
        if not hasattr(self.lib, "sdca_abi_version") or self.lib.sdca_abi_version() != ABI_VERSION:
            raise RuntimeError("Library and wrapper disagree on the C ABI; rebuild the library")
        _declare(self.lib)
        if self.lib.sdca_history_fields() != len(HISTORY):
            raise RuntimeError("Library and wrapper disagree on the history layout; rebuild")
        self.version = (
            self.lib.sdca_version().decode() if hasattr(self.lib, "sdca_version") else "unknown"
        )

    def _call(self, name: str, *args: object) -> None:
        code = getattr(self.lib, name)(*args)
        if code:
            message = self.lib.sdca_last_error().decode()
            raise ValueError(message) if code == _INVALID_ARGUMENT else RuntimeError(message)

    def _logits(self, logits: npt.ArrayLike, name: str = "classes") -> FloatArray:
        """Finite scores (n>=1,k>=2) in the data dtype; name labels k in the error."""
        z = _array(logits, self.dtype)
        if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] < 2:
            raise ValueError(f"logits must have shape (n>=1,{name}>=2)")
        return z

    def _positive_labels(self, target: Targets, n: int, k: int) -> npt.NDArray[np.uint8]:
        """Positive-label mask (n,m) as uint8 from labels, a mask or label lists."""
        target = _numpy(target)
        if isinstance(target, np.ndarray) and target.dtype == np.bool_ and target.shape == (n, k):
            positive = np.require(target, requirements=["C", "A"]).view(np.uint8)
        elif _label_vector(target):
            positive = _positive_mask(_labels(target, n, k), None, n, k).view(np.uint8)
        else:
            # A loss may mark every class (cross-entropy with a uniform target).
            labels, offsets = _label_sets(target, n, k, largest=k)
            positive = _positive_mask(labels, offsets, n, k).view(np.uint8)
        if not positive.any(axis=1).all():
            raise ValueError("Every example must have at least one positive label")
        return positive

    def loss_and_grad(
        self,
        logits: npt.ArrayLike,
        target: Targets,
        loss: str = "cross_entropy",
        *,
        k: int = 1,
        gamma: float = 0.0,
        alpha: float = 1.0,
        num_threads: int | None = None,
        projection_precision: str = "float64",
    ) -> tuple[npt.NDArray[np.float64], FloatArray]:
        """As libsdca.loss_and_grad, at the precision of this solver."""
        loss = _check_name(loss, LOSSES, "loss")
        z = self._logits(logits)
        _check_k(k, z.shape[1])
        _nonnegative(gamma, "gamma")
        _positive(alpha, "alpha")
        projection = _projection(projection_precision)
        threads = _threads(num_threads)
        positive = self._positive_labels(target, *z.shape)
        values = np.empty(len(z), np.float64)
        gradients = np.empty_like(z)
        options = _options(
            precision=self.code,
            loss=_IDS[loss],
            k=k,
            gamma=gamma,
            alpha=alpha,
            projection_precision=projection,
            num_threads=threads,
        )
        # The rows entry point writes into the outputs; sdca_loss_gradient computes into copies
        # first, to leave a failed call's outputs as given.
        try:
            self._call(
                "sdca_loss_gradient_rows",
                ct.byref(options),
                DTYPE_FLOAT32 if z.dtype == np.float32 else DTYPE_FLOAT64,
                *z.shape,
                _ptr(z),
                None,
                _ptr(positive),
                _ptr(values),
                _ptr(gradients),
            )
        except ValueError as error:
            raise ValueError(_row_message(str(error), loss)) from None
        return values, gradients

    def distillation_loss_and_grad(
        self,
        student: npt.ArrayLike,
        teacher: npt.ArrayLike,
        *,
        temperature: float = 1.0,
        alpha: float = 0.0,
        num_threads: int | None = None,
    ) -> tuple[npt.NDArray[np.float64], FloatArray]:
        """Per-row distillation loss (or proximal distillation if alpha > 0) and gradient w.r.t. student."""
        zs = self._logits(student)
        zt = self._logits(teacher)
        if zt.shape != zs.shape:
            raise ValueError("teacher must have the same shape as student")
        _positive(temperature, "temperature")
        _nonnegative(alpha, "alpha")
        threads = _threads(num_threads)
        n, classes = zs.shape
        dtype = DTYPE_FLOAT32 if zs.dtype == np.float32 else DTYPE_FLOAT64
        values = np.empty(n, np.float64)
        gradients = np.empty_like(zs)
        if alpha > 0:
            _proximal_distillation_rows(
                zs.ctypes.data,
                zt.ctypes.data,
                dtype,
                n,
                classes,
                float(temperature),
                float(alpha),
                values.ctypes.data,
                gradients.ctypes.data,
                threads,
                self.lib,
            )
            return values, gradients
        fp32 = self.code == 0
        stats = np.empty((n, 7), np.float64)
        scale = np.ones(1, np.float64)
        _distillation_forward(
            zs.ctypes.data,
            zt.ctypes.data,
            dtype,
            n,
            classes,
            float(temperature),
            values.ctypes.data,
            stats.ctypes.data,
            threads,
            fp32,
            self.lib,
        )
        _distillation_backward(
            zs.ctypes.data,
            zt.ctypes.data,
            dtype,
            n,
            classes,
            float(temperature),
            stats.ctypes.data,
            scale.ctypes.data,
            0,
            gradients.ctypes.data,
            threads,
            fp32,
            self.lib,
        )
        return values, gradients

    def _targets(
        self, y: Targets, n: int, classes: int, objective: str
    ) -> tuple[IndexArray, IndexArray | None]:
        """Label array (and offsets for the multilabel objectives)."""
        if objective in MULTILABEL:
            return _label_sets(y, n, classes)
        return _labels(y, n, classes), None

    def coordinate_update(
        self,
        logits: npt.ArrayLike,
        y: Targets,
        objective: str = "cross_entropy",
        *,
        previous: npt.ArrayLike | None = None,
        C: float = 1.0,
        norm2: float = 1.0,
        k: int = 1,
        gamma: float = 0.0,
        num_threads: int | None = None,
        projection_precision: str = "float64",
    ) -> FloatArray:
        """As libsdca.coordinate_update, at the precision of this solver."""
        z = self._logits(logits)
        objective = _resolve_objective(objective, y)
        n, classes = z.shape
        q = _distribution(y, n, classes)[0] if objective == "distillation" else None
        labels, offsets = (
            (np.zeros(0, np.int64), None)
            if q is not None
            else self._targets(y, n, classes, objective)
        )
        _positive(C, "C")
        _positive(norm2, "norm2")
        _nonnegative(gamma, "gamma")
        _check_k(k, classes)
        k = (
            1 if objective in ("multiclass_hinge", "distillation") else k
        )  # no top-k, as cross_entropy
        projection = _projection(projection_precision)
        threads = _threads(num_threads)
        self._check_scale(C, norm2)
        a = np.zeros_like(z) if previous is None else _array(previous, self.dtype)
        if a.shape != z.shape:
            raise ValueError("previous must match logits.shape")
        out = np.empty_like(a)
        options = _options(
            precision=self.code,
            loss=0 if q is not None else _IDS[objective],
            c=C,
            k=k,
            gamma=gamma,
            projection_precision=projection,
            num_threads=threads,
        )
        if q is not None:
            self._call(
                "sdca_update_distillation",
                ct.byref(options),
                n,
                classes,
                _ptr(z),
                _ptr(q),
                _ptr(a),
                norm2,
                _ptr(out),
            )
            return out
        self._call(
            "sdca_update_objective",
            ct.byref(options),
            n,
            classes,
            _ptr(z),
            _ptr(labels),
            _ptr(offsets),
            _ptr(a),
            norm2,
            _ptr(out),
        )
        return out

    def _check_scale(self, C: float, norm2: float) -> None:
        small = np.finfo(self.dtype).tiny
        large = np.finfo(self.dtype).max
        if (
            not small <= C <= large
            or not small <= norm2 <= large
            or not small <= C * norm2 <= large
        ):
            raise ValueError("C, norm2, and C*norm2 must be normal in the data dtype")

    def _training_data(self, X: npt.ArrayLike, kernel: bool) -> FloatArray:
        # Finiteness, and a kernel's symmetry and Cauchy-Schwarz bounds, are checked
        # natively without copies of X.
        _check_native_storage(X)
        x = np.require(_numpy(X), dtype=self.dtype, requirements=["C", "A"])
        if x.ndim != 2 or min(x.shape) < 1:
            raise ValueError("X must be a nonempty matrix")
        if kernel and x.shape[0] != x.shape[1]:
            raise ValueError("Training kernel must be square")
        return x

    def _problem(
        self, X: npt.ArrayLike, y: Targets, objective: str, num_classes: int | None, kernel: bool
    ) -> _Problem:
        x = self._training_data(X, kernel)
        objective = _resolve_objective(objective, y)
        if objective == "distillation":
            if num_classes is not None and (not _is_int(num_classes) or int(num_classes) < 2):
                raise ValueError("num_classes must be an integer >=2")
            q, classes = _distribution(y, len(x), None if num_classes is None else int(num_classes))
            _check_native_size(len(x), x.shape[1], classes)
            return _Problem(x, np.zeros(0, np.int64), None, objective, classes, kernel, q)
        classes = _num_classes(y, num_classes, objective in MULTILABEL)
        _check_native_size(len(x), x.shape[1], classes)
        labels, offsets = self._targets(y, len(x), classes, objective)
        return _Problem(x, labels, offsets, objective, classes, kernel)

    def _check_norms(self, problem: _Problem, C: float) -> None:
        """Nonzero squared norms must keep C*norm2 normal; zero rows have zero scores."""
        x = problem.x
        norms = np.diagonal(x) if problem.kernel else np.einsum("ij,ij->i", x, x, dtype=np.float64)
        if np.any(norms < 0):
            raise ValueError("A kernel diagonal must be nonnegative")
        # Zero rows take a block of size C, so C itself must fit the dtype.
        info = np.finfo(self.dtype)
        if not float(info.tiny) <= C <= float(info.max):
            raise ValueError("C, norm2, and C*norm2 must be normal in the data dtype")
        positive = norms[norms > 0]
        if len(positive):
            for norm in (positive.min(), positive.max()):
                self._check_scale(C, float(norm))

    def _primal(
        self, problem: _Problem, center: npt.ArrayLike | None
    ) -> tuple[FloatArray | None, FloatArray | None]:
        """Output weights (d,m) in Fortran order and the validated center (none for kernels)."""
        if problem.kernel:
            if center is not None:
                raise ValueError("center is only supported for feature input")
            return None, None
        w = np.empty((problem.x.shape[1], problem.classes), dtype=self.dtype, order="F")
        if center is None:
            return w, None
        c0 = np.require(_numpy(center), dtype=self.dtype, requirements=["F", "A"])
        if c0.shape != w.shape or not np.isfinite(c0).all():
            raise ValueError("center must be finite with shape (d,m)")
        return w, c0

    def _dual(
        self, problem: _Problem, initial_dual: npt.ArrayLike | None, C: float, k: int
    ) -> FloatArray:
        """Output dual variables (n,m): a validated copy of initial_dual if given."""
        n, labels = len(problem.x), problem.labels
        if initial_dual is None:
            return np.empty((n, problem.classes), dtype=self.dtype)
        a = _array(initial_dual, self.dtype).copy()
        if a.shape != (n, problem.classes):
            raise ValueError("initial_dual must have shape (n, num_classes)")
        _validate_initial(a, labels, problem.offsets, C, problem.objective, k, problem.targets)
        if not problem.multilabel and problem.targets is None:
            # Rows must sum to zero exactly; rebuild the label entry.
            rows = np.arange(n)
            a[rows, labels] = -(a.sum(axis=1, dtype=np.float64) - a[rows, labels])
        return a

    def _native_fit(
        self,
        problem: _Problem,
        w: FloatArray | None,
        a: FloatArray,
        center: FloatArray | None,
        *,
        C: float,
        k: int,
        gamma: float,
        max_epoch: int,
        eval_epoch: int,
        tol: float,
        eval_on_start: bool,
        recompute_primal: bool,
        projection: int,
        warm: bool,
        shrinking: int = 0,
    ) -> tuple[npt.NDArray[np.float64], _FitResult]:
        """Run sdca_fit, updating w and a in place.

        Returns:
            The history buffer (rows, len(HISTORY)) and the fit result.
        """
        n, d = problem.x.shape
        try:
            hist = np.empty((max_epoch // eval_epoch + 2, len(HISTORY)), np.float64)
        except (MemoryError, OverflowError, ValueError) as error:
            raise ValueError(
                "max_epoch / eval_epoch is too large for the history buffer"
            ) from error
        options = _options(
            precision=self.code,
            loss=0 if problem.targets is not None else _IDS[problem.objective],
            c=C,
            k=k,
            gamma=gamma,
            projection_precision=projection,
            max_epoch=max_epoch,
            eval_epoch=eval_epoch,
            tol=tol,
            eval_on_start=eval_on_start,
            recompute_primal=recompute_primal,
            kernel=problem.kernel,
            warm=warm,
            shrinking=shrinking,
        )
        result = _fit_result()
        if problem.targets is not None:
            self._call(
                "sdca_fit_distillation",
                ct.byref(options),
                n,
                d,
                problem.classes,
                _ptr(problem.x),
                _ptr(problem.targets),
                _ptr(center),
                _ptr(w),
                _ptr(a),
                _ptr(hist),
                len(hist),
                ct.byref(result),
            )
            return hist, result
        self._call(
            "sdca_fit",
            ct.byref(options),
            n,
            d,
            problem.classes,
            _ptr(problem.x),
            _ptr(problem.labels),
            _ptr(problem.offsets),
            _ptr(center),
            _ptr(w),
            _ptr(a),
            _ptr(hist),
            len(hist),
            ct.byref(result),
        )
        return hist, result

    def fit(
        self,
        X: npt.ArrayLike,
        y: Targets,
        objective: str = "cross_entropy",
        *,
        C: float = 1.0,
        k: int = 1,
        gamma: float = 0.0,
        num_classes: int | None = None,
        max_epoch: int = 1000,
        eval_epoch: int = 10,
        tol: float = 1e-3,
        eval_on_start: bool = False,
        recompute_primal: bool = True,
        projection_precision: str = "float64",
        kernel: bool = False,
        center: npt.ArrayLike | None = None,
        initial_dual: npt.ArrayLike | None = None,
        shrinking: str = "auto",
    ) -> Model:
        """As libsdca.fit, at the precision of this solver."""
        problem = self._problem(X, y, objective, num_classes, kernel)
        _positive(C, "C")
        _nonnegative(gamma, "gamma")
        _check_k(k, problem.classes)
        k = 1 if problem.objective in ("multiclass_hinge", "distillation") else k  # no top-k
        self._check_norms(problem, C)
        _check_schedule(max_epoch, eval_epoch, tol)
        projection = _projection(projection_precision)
        shrink = _shrinking(shrinking)
        w, c0 = self._primal(problem, center)
        a = self._dual(problem, initial_dual, C, k)
        warm = initial_dual is not None
        if warm and w is not None:
            _warm_primal(w, problem.x, a, c0)
        hist, result = self._native_fit(
            problem,
            w,
            a,
            c0,
            C=C,
            k=k,
            gamma=gamma,
            max_epoch=max_epoch,
            eval_epoch=eval_epoch,
            tol=tol,
            eval_on_start=eval_on_start,
            recompute_primal=recompute_primal,
            projection=projection,
            warm=warm,
            shrinking=shrink,
        )
        params = dict(C=C, k=k, gamma=gamma, num_classes=problem.classes)
        return _model(problem, w, a, hist, result, params)


_FLOAT_STORAGE = {np.dtype(np.float32): True, np.dtype(np.float64): False}


def _solver(x: object, precision: object) -> Solver:
    """Shared Solver for input x (preserving float32/float64 storage) and working precision."""
    if precision is None and type(x) is np.ndarray:
        fp32 = _FLOAT_STORAGE.get(x.dtype)
        if fp32 is not None:
            return _default("float64", fp32)
    prec = _canonical_precision(precision)
    _check_native_storage(x)
    dtype_name = str(getattr(x, "dtype", ""))
    if dtype_name.endswith("float32"):
        storage: type[np.floating] = np.float32
    elif dtype_name.endswith("float64"):
        storage = np.float64
    else:
        storage = np.float32 if prec == "float32" else np.float64
    work = "float64" if prec is None else prec
    if (work, storage) not in _PRECISION_CODES:
        raise ValueError("float64 inputs require precision='float64' or None")
    return _default(work, storage is np.float32)


@cache
def _default(precision: str, fp32_storage: bool = False) -> Solver:
    return Solver(precision, dtype=np.float32 if fp32_storage else np.float64)


def fit(
    X: npt.ArrayLike,
    y: Targets,
    objective: str = "cross_entropy",
    *,
    C: float = 1.0,
    k: int = 1,
    gamma: float = 0.0,
    num_classes: int | None = None,
    max_epoch: int = 1000,
    eval_epoch: int = 10,
    tol: float = 1e-3,
    eval_on_start: bool = False,
    recompute_primal: bool = True,
    projection_precision: str = "float64",
    kernel: bool = False,
    center: npt.ArrayLike | None = None,
    initial_dual: npt.ArrayLike | None = None,
    shrinking: str = "auto",
    precision: object = None,
) -> Model:
    """Train one of the OBJECTIVES on features or a Gram matrix.

    Tensors are detached and copied; results are NumPy arrays.

    Args:
        X: Features (n,d), or the Gram matrix (n,n) if kernel is True.
        y: Integer labels (n,); for the multilabel objectives (or cross_entropy
            with uniform targets) a boolean mask (n,num_classes) or one
            list of labels per example.
        objective: One of OBJECTIVES.
        C: Loss weight, finite and positive.
        k: Top-k parameter in [1, num_classes-1].
        gamma: Hinge smoothing, finite and >= 0 (0: nonsmooth).
        num_classes: Number of classes (>= 2); inferred from y if None.
        max_epoch: Epoch limit, an integer >= 0.
        eval_epoch: Epochs between evaluations, an integer >= 1; up to two
            more fall between where the gap's trend reaches tol.
        tol: Relative duality gap of success; 0 disables gap-based
            success (failure and progress checks remain).
        eval_on_start: Evaluate the objective before the first epoch.
        recompute_primal: Rebuild W = center + X.T @ A at each evaluation
            (feature input); False keeps the W maintained by the updates.
        projection_precision: "float64" solves the multilabel_hinge projections
            in double and re-solves with the extended-precision solver where a
            per-call check fails; "extended" always runs that solver (see
            "Projection precision" in docs/API.md).
        kernel: Whether X is a Gram matrix.
        center: (d,num_classes) center of the regularizer
            0.5*||W-center||^2 (feature input only). History and the stopping
            certificate use the full centered objective, evaluated directly
            from W-center without subtracting large squared norms.
        initial_dual: Warm start (n,num_classes), usually model.dual of an
            earlier fit. It is checked against the objective's feasible
            set, the label entry of single-label rows is rebuilt so that
            rows sum to zero exactly, and W = center + X.T @ initial_dual.
        shrinking: What the epochs of the hinge objectives leave out:
            "heuristic" skips, between full passes every tenth epoch, the
            examples whose last step left their duals unchanged and steps an
            example on its classes of nonzero dual (liblinear-style
            shrinking); "safe" leaves out for good the zero duals that the
            certified gap proves zero at the optimum (gap-safe screening,
            single-label hinges on features); "both"; "off". "auto", the
            default, is "both". The entropy objectives update everything
            whatever the value, and every evaluation, so the "solved"
            certificate, covers every example.
        precision: Working precision, one of PRECISIONS ("float32" or
            "float64", or a matching dtype); defaults to "float64" while
            keeping the storage dtype of X.

    Returns:
        The trained Model.

    Raises:
        ValueError: For invalid inputs or options.
        RuntimeError: If the native solver fails.
    """
    return _solver(X, precision).fit(
        X,
        y,
        objective,
        C=C,
        k=k,
        gamma=gamma,
        num_classes=num_classes,
        max_epoch=max_epoch,
        eval_epoch=eval_epoch,
        tol=tol,
        eval_on_start=eval_on_start,
        recompute_primal=recompute_primal,
        projection_precision=projection_precision,
        kernel=kernel,
        center=center,
        initial_dual=initial_dual,
        shrinking=shrinking,
    )


def loss_and_grad(
    logits: npt.ArrayLike,
    target: Targets,
    loss: str = "cross_entropy",
    *,
    k: int = 1,
    gamma: float = 0.0,
    alpha: float = 1.0,
    num_threads: int | None = None,
    projection_precision: str = "float64",
    precision: object = None,
) -> tuple[npt.NDArray[np.float64], FloatArray]:
    """Unregularized per-row loss and gradient w.r.t. the INPUT logits.

    Cross-entropy uses a uniform target over the marked classes (as
    multilabel_cross_entropy). Nonsmooth hinge gradients are selected
    subgradients at kinks. proximal_cross_entropy is the Moreau envelope
    min_u CE(u,target)+||u-logits||²/(2*alpha), not ordinary CE.

    Args:
        logits: Scores (n,classes), n >= 1, classes >= 2, finite.
        target: Integer labels (n,), a boolean positive-label mask
            (n,classes) or one list of labels per example; cross-entropy
            allows every class to be positive.
        loss: One of LOSSES.
        k: Top-k parameter in [1, classes-1].
        gamma: Hinge smoothing, finite and >= 0 (0: nonsmooth).
        alpha: proximal_cross_entropy step, finite and positive.
        num_threads: Threads for the rows: None takes SDCA_NUM_THREADS, else 0:
            automatic (the available CPUs as the work allows; PyTorch's count
            in a DataLoader worker); 1 is serial. Results do not depend on it.
        projection_precision: "float64" or "extended", as in fit; it affects
            only the smooth multilabel_hinge values.
        precision: Working precision, one of PRECISIONS ("float32" or
            "float64", or a matching dtype); defaults to "float64" while
            keeping the storage dtype of logits.

    Returns:
        Losses (n,) in float64 and gradients (n,classes) in the data dtype.

    Raises:
        ValueError: For invalid inputs or options.
        RuntimeError: If the native call fails.
    """
    solver = _solver(logits, precision)
    return solver.loss_and_grad(
        logits,
        target,
        loss,
        k=k,
        gamma=gamma,
        alpha=alpha,
        num_threads=num_threads,
        projection_precision=projection_precision,
    )


def distillation_loss_and_grad(
    student: npt.ArrayLike,
    teacher: npt.ArrayLike,
    *,
    temperature: float = 1.0,
    alpha: float = 0.0,
    num_threads: int | None = None,
    precision: object = None,
) -> tuple[npt.NDArray[np.float64], FloatArray]:
    """Unregularized per-row distillation loss and gradient w.r.t. the student logits.

    Computes H(softmax(teacher / temperature), softmax(student / temperature)) and its
    gradient (q - p) / temperature with respect to student. When alpha > 0, computes the
    Moreau envelope (proximal distillation loss)
    min_u H(softmax(teacher / temperature), softmax(u)) + ||u - student / temperature||^2 / (2 * alpha)
    and its gradient (pi - p) / temperature.

    Args:
        student: Student logits (n,classes), n >= 1, classes >= 2, finite.
        teacher: Teacher logits (n,classes) of the same shape, finite.
        temperature: Positive temperature T.
        alpha: Proximal step >= 0 (0: plain distillation, > 0: proximal distillation).
        num_threads: Threads for the rows, as in loss_and_grad.
        precision: Working precision, one of PRECISIONS ("float32" or "float64",
            or a matching dtype); defaults to "float64" while keeping the storage
            dtype of student.

    Returns:
        Losses (n,) in float64 and gradients (n,classes) in the student data dtype.
    """
    return _solver(student, precision).distillation_loss_and_grad(
        student,
        teacher,
        temperature=temperature,
        alpha=alpha,
        num_threads=num_threads,
    )


def proximal_distillation_loss_and_grad(
    student: npt.ArrayLike,
    teacher: npt.ArrayLike,
    *,
    temperature: float = 1.0,
    alpha: float = 1.0,
    num_threads: int | None = None,
    precision: object = None,
) -> tuple[npt.NDArray[np.float64], FloatArray]:
    """Per-row proximal distillation loss and gradient w.r.t. the student logits (alpha > 0)."""
    _positive(alpha, "alpha")
    return distillation_loss_and_grad(
        student,
        teacher,
        temperature=temperature,
        alpha=alpha,
        num_threads=num_threads,
        precision=precision,
    )


# Columns of the row statistics of sdca_entropy_forward; the last flags tiny derivatives.
ENTROPY_STATS = 5


def _entropy_forward(
    logits: FloatArray, labels: IndexArray, num_threads: int | None, stats: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    """Cross-entropy of contiguous logits and int64 labels, in float64 arithmetic.

    The native call checks finiteness and labels; the row statistics only if asked.
    """
    solver = _solver(logits, None)
    values = np.empty(len(logits), np.float64)
    statistics = np.empty((len(logits), ENTROPY_STATS), np.float64) if stats else None
    options = _options(precision=solver.code, num_threads=_threads(num_threads))
    solver._call(
        "sdca_entropy_forward",
        ct.byref(options),
        *logits.shape,
        _ptr(logits),
        _ptr(labels),
        _ptr(values),
        _ptr(statistics),
    )
    return values, statistics


def _entropy_gradient(
    logits: FloatArray,
    labels: IndexArray,
    stats: npt.NDArray[np.float64],
    scale: npt.NDArray[np.float64],
    num_threads: int | None,
    out: FloatArray,
) -> None:
    """out = scale[:, None] * the gradient of _entropy_forward, rounded once."""
    solver = _solver(logits, None)
    options = _options(precision=solver.code, num_threads=_threads(num_threads))
    solver._call(
        "sdca_entropy_gradient",
        ct.byref(options),
        *logits.shape,
        _ptr(logits),
        _ptr(labels),
        _ptr(stats),
        _ptr(scale),
        _ptr(out),
    )


# Storage codes of the tensor entry points (enum sdca_dtype); 16-bit values as raw bits.
DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_FLOAT16, DTYPE_BFLOAT16 = 0, 1, 2, 3
_Address = int | None  # a buffer's address (e.g. Tensor.data_ptr()); None or 0 is NULL


@cache
def _tensor_entry(name: str, lib: ct.CDLL | None = None) -> tuple[ct.CDLL, Any]:
    """The library (by default that of the shared float64 Solver) and its entry point name."""
    if lib is None:
        lib = _default("float64").lib
    return lib, getattr(lib, name)


@cache
def _tensor_options(
    fp32: bool,
    num_threads: int | None,
    loss: str = "cross_entropy",
    k: int = 1,
    gamma: float = 0.0,
    alpha: float = 1.0,
    projection_precision: str = "float64",
) -> tuple[_Options, int]:
    """Options of the tensor entry points (kept alive by the cache) and their address."""
    options = _options(
        precision=0 if fp32 else 2,  # SDCA_FLOAT32 or SDCA_FLOAT64: the working arithmetic
        loss=_IDS[_check_name(loss, LOSSES, "loss")],
        k=k,
        gamma=gamma,
        alpha=alpha,
        projection_precision=_projection(projection_precision),
        num_threads=_threads(num_threads),
    )
    return options, ct.addressof(options)


def _fail(lib: ct.CDLL, code: int) -> None:
    message = lib.sdca_last_error().decode()
    raise ValueError(message) if code == _INVALID_ARGUMENT else RuntimeError(message)


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
    """sdca_cross_entropy_forward on raw addresses: contiguous logits (n, classes) of dtype
    (DTYPE_*), int64 labels, float64 class weights (or None), values (n), stats (n, 5) and
    totals (2). fp32 takes float exponentials (not for DTYPE_FLOAT64).

    Raises ValueError for invalid arguments (the native message), else RuntimeError.
    """
    lib, function = _tensor_entry("sdca_cross_entropy_forward")
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
    """sdca_cross_entropy_backward on raw addresses: gradients (n, classes) of dtype from the
    stats of _cross_entropy_forward, times float64 scale[i * scale_stride] (stride 0 or 1)."""
    lib, function = _tensor_entry("sdca_cross_entropy_backward")
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
    """sdca_loss_gradient_rows on raw addresses: int64 labels (n) or a uint8 mask (n, classes),
    values (n) float64 and gradients (n, classes) in float32 for 16-bit dtypes, else dtype."""
    lib, function = _tensor_entry("sdca_loss_gradient_rows")
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
    """sdca_distillation_forward on raw addresses: contiguous student and teacher logits
    (n, classes) of dtype, values (n) and stats (n, 7) float64 (or None)."""
    lib, function = _tensor_entry("sdca_distillation_forward", library)
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
    """sdca_distillation_backward on raw addresses: gradients (n, classes) of dtype from the
    stats of _distillation_forward, times float64 scale[i * scale_stride] (stride 0 or 1)."""
    lib, function = _tensor_entry("sdca_distillation_backward", library)
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
    """sdca_proximal_distillation_rows on raw addresses: values (n) float64 and gradients
    (n, classes) in float32 for 16-bit and float32 dtypes, else float64."""
    lib, function = _tensor_entry("sdca_proximal_distillation_rows", library)
    address = _tensor_options(False, num_threads)[1]
    code = function(
        address, dtype, n, classes, student, teacher, temperature, alpha, values, gradients
    )
    if code:
        _fail(lib, code)


def coordinate_update(
    logits: npt.ArrayLike,
    y: Targets,
    objective: str = "cross_entropy",
    *,
    previous: npt.ArrayLike | None = None,
    C: float = 1.0,
    norm2: float = 1.0,
    k: int = 1,
    gamma: float = 0.0,
    num_threads: int | None = None,
    projection_precision: str = "float64",
    precision: object = None,
) -> FloatArray:
    """Return new dual blocks of one coordinate step, not gradients.

    For cross_entropy, A_new/C = one_hot(y)-softmax(logits+norm2*(A_new-A_old))
    at the block optimum; the other OBJECTIVES return their own blocks.

    Args:
        logits: Current scores (n,classes), n >= 1, classes >= 2, finite.
        y: Integer labels (n,); for the multilabel objectives (or cross_entropy
            with uniform targets) a boolean mask (n,classes) or label lists.
        objective: One of OBJECTIVES.
        previous: Current dual blocks (n,classes); zeros if None.
        C: Loss weight, finite and positive.
        norm2: Squared norm of the example (the same for every row).
        k: Top-k parameter in [1, classes-1].
        gamma: Hinge smoothing, finite and >= 0 (0: nonsmooth).
        num_threads: Threads for the rows, as in loss_and_grad.
        projection_precision: "float64" or "extended", as in fit; it affects
            only the multilabel_hinge steps.
        precision: Working precision, one of PRECISIONS ("float32" or
            "float64", or a matching dtype); defaults to "float64" while
            keeping the storage dtype of logits.

    Returns:
        New dual blocks (n,classes) in the data dtype.

    Raises:
        ValueError: For invalid inputs or options.
        RuntimeError: If the native call fails.
    """
    return _solver(logits, precision).coordinate_update(
        logits,
        y,
        objective,
        previous=previous,
        C=C,
        norm2=norm2,
        k=k,
        gamma=gamma,
        num_threads=num_threads,
        projection_precision=projection_precision,
    )
