"""The NumPy interface of the native library: losses, coordinate steps and training.

Arrays use conventional Python layouts: X (n, d), W (d, m) and the duals A (n, m) for m
classes. The objective is 0.5 * ||W - center||^2 + C * sum_i loss(X[i] @ W, y[i]); there is
no implicit intercept (append a constant feature for a regularized one).
"""

from __future__ import annotations

import ctypes as ct
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ._core import (
    _PROJECTIONS,
    _SHRINKING,
    _WORK,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    HISTORY,
    INPUT_FEATURES,
    INPUT_KERNEL,
    LOSS_NAMES,
    MULTILABEL,
    OBJECTIVES,
    STATUS,
    TARGET_LABEL_SETS,
    TARGET_LABELS,
    TARGET_LOGITS,
    TARGET_MASK,
    TARGET_PROBABILITIES,
    _canonical_precision,
    _Execution,
    _FitOptions,
    _FitResult,
    _Input,
    _is_int,
    _Loss,
    _make,
    _native,
    _Targets,
)

FloatArray = npt.NDArray[np.floating]
# A working precision: None (the storage's), "float32" or "float64" (or their dtypes).
Precision = str | type[float] | type[np.floating] | np.dtype[Any] | None
IndexArray = npt.NDArray[np.int64]


# Checks of scalar arguments: TypeError for a wrong type, ValueError for a bad value.
def _real(value: object, name: str) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    return float(value)


def _positive(value: object, name: str) -> float:
    number = _real(value, name)
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"{name} must be finite and positive")
    return number


def _nonnegative(value: object, name: str) -> float:
    number = _real(value, name)
    if not (math.isfinite(number) and number >= 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _integer(value: object, name: str, minimum: int) -> int:
    if not _is_int(value):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    number = int(value)  # pyright: ignore[reportArgumentType]
    if number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return number


def _flag(value: object, name: str) -> int:
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a bool, not {type(value).__name__}")
    return int(bool(value))


def _choice(value: object, name: str, choices: Mapping[str, int]) -> int:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, not {type(value).__name__}")
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(map(repr, choices))}")
    return choices[value]


class Loss:
    """A loss of the shared vocabulary (docs/API.md) with its own parameters.

    Attributes:
        name: The shared name, a key of LOSSES.
    """

    name: ClassVar[str]

    def _struct(self) -> _Loss:
        """struct sdca_loss of this loss."""
        keys = ("k", "gamma", "alpha", "temperature")
        fields = {key: getattr(self, key) for key in keys if hasattr(self, key)}
        projection = getattr(self, "projection_precision", "float64")
        return _make(
            _Loss,
            kind=LOSS_NAMES.index(self.name),
            projection_precision=_PROJECTIONS[projection],
            **fields,
        )

    def _check(self) -> None:
        """Normalizes and checks the parameters (in __post_init__)."""
        for key, check in (
            ("k", lambda v: _integer(v, "k", 1)),
            ("gamma", lambda v: _nonnegative(v, "gamma")),
            ("alpha", lambda v: _positive(v, "alpha")),
            ("temperature", lambda v: _positive(v, "temperature")),
        ):
            if hasattr(self, key):
                object.__setattr__(self, key, check(getattr(self, key)))
        if hasattr(self, "projection_precision"):
            _choice(self.projection_precision, "projection_precision", _PROJECTIONS)  # pyright: ignore[reportAttributeAccessIssue]

    def __post_init__(self) -> None:
        self._check()


@dataclass(frozen=True)
class CrossEntropy(Loss):
    """cross_entropy: log sum_j exp(z_j) - z_y of one label (a training objective; also on
    Probabilities targets in fit and coordinate_update)."""

    name: ClassVar[str] = "cross_entropy"


@dataclass(frozen=True)
class TopKCrossEntropy(Loss):
    """topk_cross_entropy of one label; k is an integer in [1, num_classes - 1]."""

    name: ClassVar[str] = "topk_cross_entropy"
    k: int


@dataclass(frozen=True)
class MulticlassHinge(Loss):
    """multiclass_hinge (Crammer-Singer) of one label; gamma > 0 smooths it."""

    name: ClassVar[str] = "multiclass_hinge"
    gamma: float = 0.0


@dataclass(frozen=True)
class TopKHingeAlpha(Loss):
    """topk_hinge_alpha of one label: the hinge of the average of the k largest margins."""

    name: ClassVar[str] = "topk_hinge_alpha"
    k: int
    gamma: float = 0.0


@dataclass(frozen=True)
class TopKHingeBeta(Loss):
    """topk_hinge_beta of one label: the average hinge of the k largest margins."""

    name: ClassVar[str] = "topk_hinge_beta"
    k: int
    gamma: float = 0.0


@dataclass(frozen=True)
class MultilabelHinge(Loss):
    """multilabel_hinge of label sets (each with a negative class); gamma > 0 smooths it.
    projection_precision: "float64" or "extended" (docs/API.md#projection-precision)."""

    name: ClassVar[str] = "multilabel_hinge"
    gamma: float = 0.0
    projection_precision: str = "float64"


@dataclass(frozen=True)
class MultilabelCrossEntropy(Loss):
    """multilabel_cross_entropy: cross-entropy to the uniform target on a label set."""

    name: ClassVar[str] = "multilabel_cross_entropy"


@dataclass(frozen=True)
class ProximalCrossEntropy(Loss):
    """proximal_cross_entropy: the Moreau envelope of cross-entropy with step alpha (no fit)."""

    name: ClassVar[str] = "proximal_cross_entropy"
    alpha: float = 1.0


@dataclass(frozen=True)
class Distillation(Loss):
    """distillation: H(softmax(t / T), softmax(z / T)) of teacher logits t (no fit)."""

    name: ClassVar[str] = "distillation"
    temperature: float = 1.0


@dataclass(frozen=True)
class ProximalDistillation(Loss):
    """proximal_distillation: the Moreau envelope (step alpha) of distillation in z / T."""

    name: ClassVar[str] = "proximal_distillation"
    temperature: float = 1.0
    alpha: float = 1.0


# The default loss (losses are immutable).
_CROSS_ENTROPY = CrossEntropy()

# The losses by their shared names, in the order of the C ids.
LOSSES: Mapping[str, type[Loss]] = {
    cls.name: cls
    for cls in (
        CrossEntropy,
        TopKCrossEntropy,
        MulticlassHinge,
        TopKHingeAlpha,
        TopKHingeBeta,
        MultilabelHinge,
        MultilabelCrossEntropy,
        ProximalCrossEntropy,
        Distillation,
        ProximalDistillation,
    )
}
assert tuple(LOSSES) == LOSS_NAMES
_LABELED = frozenset(
    (
        "cross_entropy",
        "topk_cross_entropy",
        "multiclass_hinge",
        "topk_hinge_alpha",
        "topk_hinge_beta",
        "proximal_cross_entropy",
    )
)
_DISTILLATION = frozenset(("distillation", "proximal_distillation"))
_PROBABILISTIC = frozenset(("cross_entropy", "multilabel_cross_entropy"))


@dataclass(frozen=True, eq=False)
class Probabilities:
    """Target probabilities (n, num_classes) for CrossEntropy in fit and coordinate_update:
    the distillation objective H(p_i, softmax(x_i W)). Rows must be finite, nonnegative and
    sum to one within 64 * num_classes eps of their dtype; they are renormalized in float64."""

    probabilities: npt.ArrayLike


@dataclass(frozen=True, eq=False)
class Kernel:
    """A Gram matrix K (n, n) as the input of fit (in place of features X); it must be
    symmetric positive semidefinite (docs/API.md#options)."""

    gram: npt.ArrayLike


Targets = npt.ArrayLike | Sequence[Sequence[int]] | Probabilities


def _numpy(x: Any) -> Any:
    """A torch tensor (any device) as a NumPy array, without importing torch: bfloat16,
    which NumPy lacks, converts exactly to float32."""
    if not type(x).__module__.startswith("torch") or not hasattr(x, "detach"):
        return x
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
    work = _canonical_precision(precision)
    narrow = a.dtype.kind == "f" and a.dtype.itemsize <= 4 or a.dtype.name == "bfloat16"
    storage: type[np.floating] = np.float32 if narrow or work == "float32" else np.float64
    return storage, _WORK[work]


def _cast(a: np.ndarray, storage: type[np.floating], fortran: bool = False) -> FloatArray:
    return np.require(a, dtype=storage, requirements=["F" if fortran else "C", "A"])


def _dtype(a: np.ndarray) -> int:
    return DTYPE_FLOAT32 if a.dtype == np.float32 else DTYPE_FLOAT64


def _scores(logits: Any, precision: object, name: str = "logits") -> tuple[FloatArray, int]:
    """Scores (n >= 0, classes >= 2) in their storage dtype, and the working precision."""
    a = _dense(logits, name)
    storage, work = _storage(a, precision)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError(f"{name} must have shape (n, classes>=2)")
    return _cast(a, storage), work


def _execution(work: int, num_threads: object) -> _Execution:
    threads = 0 if num_threads is None else _integer(num_threads, "num_threads", 1)
    return _make(_Execution, precision=work, num_threads=threads)


def _check_loss(loss: object, objective: bool = False) -> Loss:
    if not isinstance(loss, Loss):
        raise TypeError(
            f"loss must be a libsdca loss such as CrossEntropy(), not {type(loss).__name__}"
        )
    if objective and loss.name not in OBJECTIVES:
        raise ValueError(f"{loss.name} is not a training objective (one of {OBJECTIVES})")
    return loss


def _labels(y: Any, n: int) -> IndexArray:
    """One integer class label per example as a contiguous int64 array (n,)."""
    y = _numpy(y)
    try:
        labels = np.asarray(y)
    except ValueError:
        raise TypeError("y must be an integer label array of shape (n,)") from None
    if labels.dtype.kind not in "iu":
        raise TypeError(f"y must hold integer class labels, not {labels.dtype}")
    if labels.ndim == 2 and labels.shape[1] == 1:
        raise ValueError("y of shape (n, 1): pass y.ravel() for one label per example")
    if labels.shape != (n,):
        raise ValueError(f"y must have shape ({n},): one label per example")
    return np.require(labels, dtype=np.int64, requirements=["C", "A"])


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
    sizes = np.array([len(row) for row in rows], dtype=np.int64)
    offsets = np.zeros(n + 1, np.int64)
    np.cumsum(sizes, out=offsets[1:])
    flat = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    return np.ascontiguousarray(flat, dtype=np.int64), offsets


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


def _probabilities(p: Probabilities, n: int) -> npt.NDArray[np.float64]:
    """Validated target probabilities (n, classes) in float64, renormalized."""
    a = _dense(p.probabilities, "probabilities")
    if a.ndim != 2 or len(a) != n or a.shape[1] < 2:
        raise ValueError(f"probabilities must have shape ({n}, num_classes>=2)")
    q = np.array(a, dtype=np.float64)
    if not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("Target probabilities must be finite and nonnegative")
    eps = np.finfo(a.dtype).eps if a.dtype.kind == "f" else np.finfo(np.float64).eps
    sums = q.sum(axis=1)
    if np.any(np.abs(sums - 1) > 64 * float(eps) * q.shape[1]):
        raise ValueError("Target probability rows must sum to one")
    q /= sums[:, None]
    return q


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
        if self.kind in (TARGET_MASK, TARGET_PROBABILITIES, TARGET_LOGITS):
            array = next(a for a in self.arrays.values() if a is not None)
            return int(array.shape[1])
        labels = self.arrays["labels"]
        return int(labels.max()) + 1 if labels is not None and labels.size else None


def _targets(
    loss: Loss, y: Any, n: int, *, training: bool, scores: FloatArray | None = None
) -> _TargetArgs:
    """The targets a loss takes (no guessing from y): labels, label sets, Probabilities
    (CrossEntropy when training) or teacher logits (the distillation losses)."""
    if isinstance(y, Probabilities):
        if not (training and loss.name == "cross_entropy"):
            raise TypeError(
                "Probabilities are targets of CrossEntropy in fit and coordinate_update only"
            )
        return _TargetArgs(TARGET_PROBABILITIES, probabilities=_probabilities(y, n))
    if loss.name in _LABELED:
        return _TargetArgs(TARGET_LABELS, labels=_labels(y, n))
    if loss.name in MULTILABEL:
        first, second = _label_sets(y, n)
        if second is None:
            return _TargetArgs(TARGET_MASK, mask=first)
        return _TargetArgs(TARGET_LABEL_SETS, labels=first, offsets=second)
    assert scores is not None and loss.name in _DISTILLATION
    teacher = _dense(y, "teacher logits")
    if teacher.shape != scores.shape:
        raise ValueError("teacher logits must have the shape of the student logits")
    return _TargetArgs(TARGET_LOGITS, logits=_cast(teacher, scores.dtype.type))


def loss_and_grad(
    logits: npt.ArrayLike,
    target: Targets,
    loss: Loss = _CROSS_ENTROPY,
    *,
    precision: Precision = None,
    num_threads: int | None = None,
) -> tuple[npt.NDArray[np.float64], FloatArray]:
    """Per-row unregularized losses and their gradients with respect to the logits.

    Nonsmooth hinge gradients are subgradients at the kinks. ProximalCrossEntropy and
    ProximalDistillation are Moreau envelopes, not the plain losses.

    Args:
        logits: Scores (n, classes), classes >= 2, finite.
        target: What the loss takes: integer labels (n,) for the single-label losses and
            ProximalCrossEntropy; a boolean mask (n, classes) or a sequence of integer label
            sets for the multilabel losses; teacher logits (n, classes) for Distillation and
            ProximalDistillation.
        loss: A loss of LOSSES, such as CrossEntropy() or TopKHingeBeta(k=2).
        precision: Working precision: None (the storage's: 16-bit inputs are cast to float32
            and integers to float64), "float32" (float64 inputs are cast to float32) or
            "float64".
        num_threads: Threads for the rows: None takes SDCA_NUM_THREADS, else the available
            CPUs; a positive integer caps them. The results do not depend on it.

    Returns:
        Losses (n,) in float64 and gradients (n, classes) in the storage dtype.

    Raises:
        TypeError: For arguments of a wrong type (such as float labels or a sparse matrix).
        ValueError: For invalid values.
        RuntimeError: If the native computation fails.
    """
    loss = _check_loss(loss)
    z, work = _scores(logits, precision)
    targets = _targets(loss, target, len(z), training=False, scores=z)
    execution = _execution(work, num_threads)
    values = np.empty(len(z), np.float64)
    gradients = np.empty_like(z)
    struct = loss._struct()
    _native().call(
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
    return values, gradients


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
    targets = _targets(loss, target, n, training=True)
    c = _real(C, "C")
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
        weights: Primal weights W (d, m) in the storage dtype; None for a kernel.
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
        coefficients = self.dual if self.kernel else self.weights
        assert coefficients is not None
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
        loss: A training objective, such as CrossEntropy() or TopKHingeBeta(k=2).
        C: Loss weight, finite and positive.
        tol: The relative duality gap of status "solved", finite and >= 0 (0: never).
        max_epoch: Epoch limit, an integer >= 0.
        eval_epoch: Epochs between evaluations (0: evaluate only at the end); up to two more
            fall between where the gap's trend reaches tol.
        eval_on_start: Also evaluate before the first epoch.
        recompute_primal: Rebuild W = center + X.T @ A at each evaluation (features).
        shrinking: What the epochs of the hinge losses leave out: "auto" (the default,
            "both"), "off", "heuristic", "safe" or "both" (docs/API.md#shrinking). The
            other losses update every class of every example and take "auto" only.
        seed: The order of the examples in the epochs, an integer in [0, 2**64).
        num_classes: Number of classes (>= 2); None infers it from y.
        center: Center (d, m) of the regularizer (features only); None is zero.
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
    targets = _targets(loss, y, n, training=True)
    classes = _fit_classes(targets, num_classes)
    options = _make(
        _FitOptions,
        c=_real(C, "C"),
        tol=_real(tol, "tol"),
        max_epoch=_integer(max_epoch, "max_epoch", 0),
        eval_epoch=_integer(eval_epoch, "eval_epoch", 0),
        eval_on_start=_flag(eval_on_start, "eval_on_start"),
        recompute_primal=_flag(recompute_primal, "recompute_primal"),
        shrinking=_choice(shrinking, "shrinking", _SHRINKING),
        seed=_integer(seed, "seed", 0),
    )
    if options.seed != seed:
        raise ValueError("seed must be below 2**64")
    c0, a0 = _warm_start(center, initial_dual, kernel, storage, (n, d, classes))
    return _train(loss, C, x, kernel, storage, work, num_threads, targets, classes, options, c0, a0)


def _fit_classes(targets: _TargetArgs, num_classes: int | None) -> int:
    """The class count of fit: num_classes, checked against the targets, or inferred."""
    if num_classes is None:
        classes = targets.num_classes()
        if classes is None:
            raise ValueError("num_classes cannot be inferred from empty label sets")
    else:
        classes = _integer(num_classes, "num_classes", 2)
        implied = targets.num_classes() if targets.kind != TARGET_LABELS else None
        if targets.kind != TARGET_LABEL_SETS and implied not in (None, classes):
            raise ValueError(f"The targets have {implied} classes, not num_classes={classes}")
    if classes < 2:
        raise ValueError("Training needs num_classes >= 2")
    return classes


def _warm_start(
    center: npt.ArrayLike | None,
    initial_dual: npt.ArrayLike | None,
    kernel: bool,
    storage: type[np.floating],
    sizes: tuple[int, int, int],
) -> tuple[FloatArray | None, FloatArray | None]:
    """fit's center (d, m) and initial_dual (n, m) in the storage dtype (sizes: n, d, m),
    or None."""
    n, d, classes = sizes
    c0 = None
    if center is not None:
        if kernel:
            raise ValueError("center applies to feature input only")
        c0 = _cast(_dense(center, "center"), storage, fortran=True)
        if c0.shape != (d, classes):
            raise ValueError(f"center must have shape ({d}, {classes})")
    a0 = None
    if initial_dual is not None:
        a0 = _cast(_dense(initial_dual, "initial_dual"), storage)
        if a0.shape != (n, classes):
            raise ValueError(f"initial_dual must have shape ({n}, {classes})")
    return c0, a0


def _train(
    loss: Loss,
    C: float,
    x: FloatArray,
    kernel: bool,
    storage: type[np.floating],
    work: int,
    num_threads: int | None,
    targets: _TargetArgs,
    classes: int,
    options: _FitOptions,
    c0: FloatArray | None,
    a0: FloatArray | None,
) -> Model:
    """fit on its checked arguments: the native solver and the Model of its result."""
    n, d = x.shape
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
    weights = None if kernel else np.empty((d, classes), storage, order="F")
    dual = np.empty((n, classes), storage)
    data = _make(
        _Input,
        kind=INPUT_KERNEL if kernel else INPUT_FEATURES,
        dtype=_dtype(x),
        n=n,
        d=d,
        x=x.ctypes.data,
    )
    struct = loss._struct()
    execution = _execution(work, num_threads)
    result = _make(_FitResult)
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
