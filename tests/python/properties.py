"""Property-based tests of the loss gradients: the inputs, the properties and their
tolerances, shared by test_properties_torch.py (libsdca.torch and NumPy's loss_and_grad) and
test_properties_jax.py (libsdca.jax and libsdca.optax). It imports neither framework, so the
--group option of conftest.py keeps the two files in their groups.

The properties apply to the gradients of the losses with respect to the logits:

- P1, zero sum: every loss is invariant to adding a constant to the logits of a row, so the
  exact gradient of a row sums to zero. The returned row, summed exactly (math.fsum of its
  values, exact in float64), satisfies |sum g| <= zero_sum_bound, and a row with a single
  nonzero entry fails (where every other entry underflows, the target entry is 0 too).
- P1', no NaN: for finite logits (and -inf masks with at least one finite entry per row)
  the loss is not NaN or -inf (a masked label, or a row loss beyond the range, is +inf:
  docs/API.md, Masked classes and Known limits) and every gradient entry is finite.
- P3, nonnegative loss: every loss is nonnegative by definition; a row's loss is >= minus
  a rounding allowance of the scale of its terms (nonnegative_allowance), so a finite
  negative loss with a zero-sum gradient fails.
- P2, identical logits: distillation_loss(x, x) and proximal_distillation_loss(x, x) have an
  exactly zero gradient at every temperature. Their loss is H(p, p), the entropy, not a KL
  divergence, so it is not 0. cross_entropy(x, softmax(x)), with the softmax rounded to the
  logits' dtype, is the documented cancellation limit (docs/API.md, Known limits: within
  eps of the probabilities), checked with that bound (p2_floor_bound).

The inputs come from one set of generators (draw_row, draw_batch) that read their choices
from a Source: HypothesisSource (the strategies of the property tests, with shrinking,
targeting and explicit examples) or RngSource (a seeded numpy Generator, for the batched
sweep that runs on CUDA), so both test the same input distribution:

- general rows: lists of finite floats of the dtype (st.floats(width=...); bfloat16
  draws float32 values and keeps those finite after the cast), over the full range;
- margin rows: [m, 0, ..., 0] with the target at m, and moderate random rows plus a target
  margin m, with m up to the collapse (16.6 float32, 36.7 float64) and underflow (about 104
  float32, 745 float64) margins of the output and work dtypes;
- masks: -inf entries with at least one finite entry per row;
- rows of 2 to 64 classes, batches of 1 to 4 rows, reduction "none" mostly, "sum" and "mean"
  (an all-ones cotangent) for the frameworks.

Explicit examples (EXAMPLES, P2_EXAMPLES) are dtype-free Batches, cast to the dtype of a test
(saturating at its largest finite value), including rows of 1000 and 32000 classes, where the
CUDA kernels take another path.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol, TypeVar

import numpy as np
from interfaces import bfloat16_bits, from_bfloat16_bits
from storage_limits import LIMITS

T = TypeVar("T")

# The loss kinds (as interfaces.C_KINDS): topk_hinge by variant, cross-entropy of labels and
# of probabilities.
PROBABILITIES = "cross_entropy_probabilities"
FAMILIES = (
    "cross_entropy",
    PROBABILITIES,
    "topk_cross_entropy",
    "proximal_cross_entropy",
    "multilabel_cross_entropy",
    "multiclass_hinge",
    "topk_hinge_alpha",
    "topk_hinge_beta",
    "multilabel_hinge",
    "distillation",
    "proximal_distillation",
)
TOPK = ("topk_cross_entropy", "topk_hinge_alpha", "topk_hinge_beta")
HINGES = ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge")
MULTILABEL = ("multilabel_cross_entropy", "multilabel_hinge")
TEACHER = ("distillation", "proximal_distillation")
PROXIMAL = ("proximal_cross_entropy", "proximal_distillation")
# The losses of one label per row take class weights; cross-entropy takes label smoothing.
WEIGHTED = (
    "cross_entropy",
    PROBABILITIES,
    "topk_cross_entropy",
    "proximal_cross_entropy",
    "multiclass_hinge",
    "topk_hinge_alpha",
    "topk_hinge_beta",
)
SMOOTHED = ("cross_entropy", PROBABILITIES)
# P2: the losses of a target that the logits determine.
P2_FAMILIES = ("distillation", "proximal_distillation", PROBABILITIES)

# Dtype modes: name -> (storage dtype, precision option).
MODES: dict[str, tuple[str, str | None]] = {
    "float64": ("float64", None),
    "float32": ("float32", None),
    "float32-p64": ("float32", "float64"),
    "float16": ("float16", None),
    "bfloat16": ("bfloat16", None),
}
MAX_CLASSES = 64
MAX_ROWS = 4
REDUCTIONS = ("none", "none", "sum", "mean")

# P1 bounds: |sum g| <= relative * sum |g| + C eta_out (+ C tiny_out where XLA flushes).
# float64: the relative bound 1e-12. float32, float16 and bfloat16 gradients: C_LOW eps_out
# = 2 u_out. Each stored entry is one rounding of a work value (u_out of it), so the exact sum
# of the stored row is within u_out sum |g| of the work row's, which sums to zero within the
# work's rounding (at most u_work sum |g|, u_work <= u_out, for the target entry formed as
# minus the sum of the others).
REL_FLOAT64 = 1e-12
C_LOW = float(os.environ.get("SDCA_PROPERTY_C_LOW", "1.0"))  # the calibration run raises it
# P2 floor of cross_entropy(x, softmax(x)) (the documented cancellation of q - t): each entry
# within C_P2 eps_out of the larger of t_j and the work's T q_j, plus tiny_out.
C_P2 = 4.0
# The distillation losses (CPU kernels): C_DISTILL eps_out. Their entries (q - p) / T are each
# formed from expm1 of the log-ratio and rounded on their own, with no entry closing the sum,
# and docs/API.md (Known limits, Distillation) states that each row sums to zero "within a few
# eps of its 1-norm"; rows reach about 2 eps (float32, bfloat16 and float16 inputs; NumPy,
# PyTorch and JAX alike).
C_DISTILL = 4.0
# float64 work: the targets' allowance 2 eps_work T scaled by the documented c of the CPU
# kernels in float64 work (docs/API.md, Known limits, Row sums). Each probability carries one
# or two eps of the log-sum-exp's rounding, all of one sign, so a row whose gradient nearly
# cancels (targets near softmax(x)) sums them, above C_LOW.
C_FLOAT64 = 16.0
# CUDA kernels: P1 with C_CUDA eps_out instead of C_LOW, and two subnormal roundings per
# entry. C_CUDA is the documented float32 accuracy of the work (docs/API.md, Known limits:
# 64 eps of each entry's scale). Rows whose entries lie within 2^20 of float32's subnormal
# range take exp32's rescaled path and sum to more eps than elsewhere. Not a CPU constant.
C_CUDA = 64.0
# P3: a loss is >= -C_P3 eps_work (1 + max |score|) (nonnegative_allowance).
C_P3 = 4.0
SWEEP_SEED = 20261124
SWEEP_LENGTHS = (2, 3, 4, 5, 7, 8, 13, 16, 17, 31, 32, 33, 48, 63, 64)
SWEEP_ROWS = 10_000


def work_dtype(dtype: str, precision: str | None) -> str:
    return "float64" if dtype == "float64" or precision == "float64" else "float32"


def eta(dtype: str) -> float:
    """The smallest subnormal of dtype."""
    return LIMITS[dtype].tiny * LIMITS[dtype].eps


def represent(values: Any, dtype: str) -> np.ndarray:
    """float64 values rounded to dtype (to nearest even), finite values beyond its range
    saturated at its largest finite value."""
    x = np.asarray(values, np.float64)
    with np.errstate(over="ignore"):
        if dtype == "bfloat16":
            y = from_bfloat16_bits(bfloat16_bits(x.astype(np.float32)))
        else:
            y = x.astype(dtype).astype(np.float64)
    big = LIMITS[dtype].max
    return np.where(np.isfinite(x) & ~np.isfinite(y), np.copysign(big, x), y)


# --- sources ---------------------------------------------------------------------------------


class Source(Protocol):
    def integer(self, lo: int, hi: int) -> int: ...
    def sample(self, values: Sequence[T]) -> T: ...
    def uniform(self, lo: float, hi: float) -> float: ...
    def full_float(self, dtype: str) -> float: ...


@functools.cache
def _full_floats(dtype: str) -> Any:
    """st.floats of a dtype: finite, the full range."""
    from hypothesis import strategies as st

    if dtype == "bfloat16":
        cast = st.floats(width=32, allow_nan=False, allow_infinity=False)
        return cast.map(lambda v: float(represent(v, "bfloat16"))).filter(math.isfinite)
    if dtype == "float16":
        return st.floats(width=16, allow_nan=False, allow_infinity=False)
    if dtype == "float32":
        return st.floats(width=32, allow_nan=False, allow_infinity=False)
    return st.floats(allow_nan=False, allow_infinity=False)


class HypothesisSource:
    """The choices of a Hypothesis draw (st.composite)."""

    def __init__(self, draw: Callable[[Any], Any]) -> None:
        self.draw = draw

    def integer(self, lo: int, hi: int) -> int:
        from hypothesis import strategies as st

        return self.draw(st.integers(lo, hi))

    def sample(self, values: Sequence[T]) -> T:
        from hypothesis import strategies as st

        return self.draw(st.sampled_from(values))

    def uniform(self, lo: float, hi: float) -> float:
        from hypothesis import strategies as st

        return self.draw(st.floats(lo, hi))

    def full_float(self, dtype: str) -> float:
        return self.draw(_full_floats(dtype))


@functools.cache
def special_values(dtype: str) -> tuple[float, ...]:
    """The values that Hypothesis prefers among the finite floats of a dtype: zeros, small
    integers, the largest finite value, the smallest normal and subnormal, and halves."""
    lim = LIMITS[dtype]
    base = [0.0, 1.0, 0.5, 2.0, 37.0, lim.max, lim.max / 2, lim.tiny, eta(dtype), 1e-8, 1e30]
    values = represent(base + [-v for v in base], dtype)
    return tuple(float(v) for v in values if v <= lim.max)


class RngSource:
    """The choices of a seeded numpy Generator: full_float draws a uniform bit pattern of the
    dtype (or, a quarter of the time, one of special_values), as Hypothesis mixes both."""

    BITS = {"float64": np.uint64, "float32": np.uint32, "float16": np.uint16}

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def integer(self, lo: int, hi: int) -> int:
        return int(self.rng.integers(lo, hi + 1))

    def sample(self, values: Sequence[T]) -> T:
        return values[int(self.rng.integers(len(values)))]

    def uniform(self, lo: float, hi: float) -> float:
        return float(self.rng.uniform(lo, hi))

    def full_float(self, dtype: str) -> float:
        if self.rng.random() < 0.25:
            return self.sample(special_values(dtype))
        while True:
            if dtype == "bfloat16":
                bits = self.rng.integers(0, 1 << 16, dtype=np.uint16)
                value = float(from_bfloat16_bits(np.array([bits]))[0])
            else:
                kind = self.BITS[dtype]
                width = 8 * np.dtype(kind).itemsize
                bits = self.rng.integers(0, 1 << width, dtype=kind, endpoint=False)
                value = float(np.array([bits], kind).view(dtype)[0])
            if math.isfinite(value):
                return value


# --- generators ------------------------------------------------------------------------------


@functools.cache
def special_margins(dtype: str, work: str) -> tuple[float, ...]:
    """The collapse margins p ln 2 and the underflow margins -ln(tiny) and -ln(eta) of the
    output and work dtypes, and points around them."""
    out = {0.0, 0.5, 1.0, 2.0}
    for d in sorted({dtype, work}):
        lim = LIMITS[d]
        for m in (lim.digits * math.log(2), -math.log(lim.tiny), -math.log(eta(d))):
            out.update(m * f for f in (0.95, 1.0, 1.05))
    return tuple(sorted(out))


def draw_margin(src: Source, dtype: str, work: str) -> float:
    """A target margin: a special margin or uniform up to 1.1 times the work's underflow."""
    if src.sample(("special", "uniform")) == "special":
        m = src.sample(special_margins(dtype, work))
    else:
        m = src.uniform(0.0, -1.1 * math.log(eta(work)))
    return m * src.sample((1.0, -1.0))


ROW_KINDS = ("general", "flat", "offset")


def draw_row(src: Source, dtype: str, work: str, classes: int, target: int) -> list[float]:
    """A row of logits (float64 values of dtype): general, flat ([m, 0, ...]) or offset (a
    moderate random row plus the margin m at the target), with -inf masks half the time."""
    kind = src.sample(ROW_KINDS)
    if kind == "general":
        row = [src.full_float(dtype) for _ in range(classes)]
    else:
        m = draw_margin(src, dtype, work)
        if kind == "flat":
            row = [0.0] * classes
        else:
            shift = src.sample((0.0, 100.0, -3e4))
            row = [shift + src.uniform(-8.0, 8.0) for _ in range(classes)]
        row[target] += m
    if src.sample((False, True)):
        row = [-math.inf if src.integer(0, 3) == 0 else v for v in row]
        if all(v == -math.inf for v in row):
            row[src.integer(0, classes - 1)] = 0.0
    return [float(v) for v in represent(row, dtype)]


def draw_positives(src: Source, classes: int, label: int) -> list[bool]:
    """A multilabel target: the label and, half the time, each other class with chance 1/3."""
    extra = src.sample((False, True))
    return [j == label or (extra and src.integer(0, 2) == 0) for j in range(classes)]


def softmax(row: Any) -> np.ndarray:
    z = np.asarray(row, np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(z - z[np.isfinite(z)].max())
    return e / e.sum()


def draw_probabilities(src: Source, dtype: str, work: str, classes: int, label: int) -> list[float]:
    """A probability target: one-hot, uniform draws (not normalized) or the softmax of a
    drawn row, rounded to dtype."""
    kind = src.sample(("onehot", "uniform", "softmax"))
    if kind == "onehot":
        row = np.eye(classes)[label]
    elif kind == "uniform":
        row = np.array([src.uniform(0.0, 1.0) for _ in range(classes)])
    else:
        row = softmax(draw_row(src, dtype, work, classes, label))
    return [float(v) for v in represent(row, dtype)]


def _fmt(value: float) -> str:
    if math.isinf(value):
        return "math.inf" if value > 0 else "-math.inf"
    return repr(float(value))


@dataclasses.dataclass(frozen=True, eq=False)
class Batch:
    """Rows of logits of one width and every kind of target; a loss reads what it takes.
    Values are float64 numbers of the storage dtype (cast rounds and saturates). Defaults: the
    one-hot positives and probabilities of the labels, the reversed logits as the teacher."""

    logits: np.ndarray  # (n, classes) float64
    labels: np.ndarray  # (n,) int64
    positives: np.ndarray | None = None  # (n, classes) bool
    probabilities: np.ndarray | None = None  # (n, classes) float64
    teacher: np.ndarray | None = None  # (n, classes) float64
    k: int = 2
    gamma: float = 0.0
    alpha: float = 1.0
    temperature: float = 1.0
    weight: np.ndarray | None = None  # (classes,) float64
    smoothing: float = 0.0
    reduction: str = "none"
    name: str = ""

    @staticmethod
    def of(logits: Any, labels: Any = None, **fields: Any) -> Batch:
        z = np.asarray(logits, np.float64)
        if z.ndim == 1:
            z = z[None]
        y = np.zeros(len(z), np.int64) if labels is None else np.asarray(labels, np.int64)
        arrays = {}
        for key in ("positives", "probabilities", "teacher", "weight"):
            if fields.get(key) is not None:
                kind = bool if key == "positives" else np.float64
                value = np.asarray(fields.pop(key), kind)
                arrays[key] = value[None] if value.ndim == 1 and key != "weight" else value
        return Batch(z, y, **arrays, **fields)

    @property
    def rows(self) -> int:
        return self.logits.shape[0]

    @property
    def classes(self) -> int:
        return self.logits.shape[1]

    def cast(self, dtype: str) -> Batch:
        def rounded(x: np.ndarray | None) -> np.ndarray | None:
            return None if x is None else represent(x, dtype)

        return dataclasses.replace(
            self,
            logits=represent(self.logits, dtype),
            probabilities=rounded(self.probabilities),
            teacher=rounded(self.teacher),
            weight=rounded(self.weight),
        )

    def plain(self) -> Batch:
        """Without the options that optax's losses lack."""
        return dataclasses.replace(self, gamma=0.0, weight=None, smoothing=0.0, reduction="none")

    def __repr__(self) -> str:
        if self.name:
            return f"<example {self.name}: {self.rows} x {self.classes}>"
        parts = ["Batch.of(", self._matrix(self.logits), f", {self.labels.tolist()}"]
        for key in ("positives", "probabilities", "teacher"):
            value = getattr(self, key)
            if value is not None:
                text = repr(value.tolist()) if key == "positives" else self._matrix(value)
                parts.append(f", {key}={text}")
        if self.weight is not None:
            parts.append(", weight=[" + ", ".join(map(_fmt, self.weight)) + "]")
        for key, default in (("k", 2), ("gamma", 0.0), ("alpha", 1.0), ("temperature", 1.0)):
            if getattr(self, key) != default:
                parts.append(f", {key}={getattr(self, key)!r}")
        if self.smoothing:
            parts.append(f", smoothing={self.smoothing!r}")
        if self.reduction != "none":
            parts.append(f", reduction={self.reduction!r}")
        return "".join(parts) + ")"

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        """Hypothesis prints falsifying examples with this (Batch.of(...), to paste)."""
        printer.text(repr(self))

    @staticmethod
    def _matrix(x: np.ndarray) -> str:
        return "[" + ", ".join("[" + ", ".join(map(_fmt, row)) + "]" for row in x) + "]"


def draw_batch(
    src: Source,
    family: str,
    mode: str,
    *,
    classes: int | None = None,
    rows: int | None = None,
    options: bool = True,
    reductions: bool = True,
) -> Batch:
    """A batch for a loss in a dtype mode: rows of the width classes (drawn from 2 to 64),
    the targets the loss takes and its parameters; options: class weights and label smoothing
    where the loss takes them, and smoothed hinges; reductions: sum and mean besides none."""
    dtype, precision = MODES[mode]
    work = work_dtype(dtype, precision)
    c = classes if classes is not None else src.integer(2, MAX_CLASSES)
    n = rows if rows is not None else src.integer(1, MAX_ROWS)
    labels = [src.integer(0, c - 1) for _ in range(n)]
    logits = [draw_row(src, dtype, work, c, y) for y in labels]
    fields: dict[str, Any] = {}
    if family in MULTILABEL:
        fields["positives"] = [draw_positives(src, c, y) for y in labels]
    if family == PROBABILITIES:
        fields["probabilities"] = [draw_probabilities(src, dtype, work, c, y) for y in labels]
    if family in TEACHER:
        fields["teacher"] = [draw_row(src, dtype, work, c, y) for y in labels]
        fields["temperature"] = src.sample((1.0, 4.0))
    if family in TOPK:
        fields["k"] = min(src.sample((1, 2, 3, 5, c - 1)), c - 1)
    if family in HINGES and options:
        fields["gamma"] = src.sample((0.0, 0.5, 2.0))
    if family in PROXIMAL:
        fields["alpha"] = src.sample((1.0, 0.1, 10.0))
    if family in WEIGHTED and options and src.sample((False, True)):
        fields["weight"] = [src.sample((1.0, 0.25, 0.5, 2.0, 4.0)) for _ in range(c)]
    if family in SMOOTHED and options:
        fields["smoothing"] = src.sample((0.0, 0.1))
    if reductions:
        fields["reduction"] = src.sample(REDUCTIONS)
    return Batch.of(logits, labels, **fields).cast(dtype)


def batches(family: str, mode: str, **kwargs: Any) -> Any:
    """The Hypothesis strategy of draw_batch."""
    from hypothesis import strategies as st

    @st.composite
    def strategy(draw: Callable[[Any], Any]) -> Batch:
        return draw_batch(HypothesisSource(draw), family, mode, **kwargs)

    return strategy()


def p2_batch(batch: Batch, family: str) -> Batch:
    """The batch of P2: the teacher equal to the logits, or the probabilities softmax(logits)
    rounded to the logits' dtype (the batch is cast; its dtype is that of the logits)."""
    if family in TEACHER:
        return dataclasses.replace(batch, teacher=batch.logits.copy(), reduction="none")
    probabilities = np.array([softmax(row) for row in batch.logits])
    return dataclasses.replace(
        batch, probabilities=probabilities, weight=None, smoothing=0.0, reduction="none"
    )


def sweep_batches(family: str, mode: str, *, options: bool = True) -> Iterator[Batch]:
    """The batched sweep (SWEEP_SEED): SWEEP_ROWS rows of the widths SWEEP_LENGTHS, four
    batches of one width each (each with its own parameters), then 64 rows of 1000 classes
    and 8 of 32000, all with reduction "none"; the same generators as the strategies."""
    seed = [SWEEP_SEED, FAMILIES.index(family), list(MODES).index(mode)]
    src = RngSource(np.random.default_rng(seed))
    per = SWEEP_ROWS // (4 * len(SWEEP_LENGTHS))
    shapes = [(c, per) for c in SWEEP_LENGTHS for _ in range(4)] + [(1000, 64), (32000, 8)]
    for c, n in shapes:
        yield draw_batch(src, family, mode, classes=c, rows=n, options=options, reductions=False)


# --- resolution of a batch for a loss --------------------------------------------------------


def target(family: str, batch: Batch) -> np.ndarray:
    """The target of the loss: labels, a mask (with a negative for multilabel_hinge),
    probabilities or teacher logits."""
    onehot = np.eye(batch.classes, dtype=bool)[batch.labels]
    if family in MULTILABEL:
        mask = (batch.positives if batch.positives is not None else onehot).copy()
        if family == "multilabel_hinge":
            for i in np.flatnonzero(mask.all(1)):
                mask[i, (batch.labels[i] + 1) % batch.classes] = False
        return mask
    if family == PROBABILITIES:
        return batch.probabilities if batch.probabilities is not None else onehot * 1.0
    if family in TEACHER:
        return batch.teacher if batch.teacher is not None else batch.logits[:, ::-1].copy()
    return batch.labels


def parameters(family: str, batch: Batch) -> dict[str, float]:
    """The parameters of interfaces.framework_call and interfaces.numpy_loss: k (within
    [1, classes)), gamma (the hinges), alpha (the step of the proximal losses), temperature."""
    k = max(1, min(batch.k, batch.classes - 1))
    gamma = batch.gamma if family in HINGES else 0.0
    return {"k": k, "gamma": gamma, "alpha": batch.alpha, "temperature": batch.temperature}


def options(family: str, batch: Batch) -> dict[str, Any]:
    """The class weights and label smoothing that the loss takes (weight as float64)."""
    out: dict[str, Any] = {}
    if family in WEIGHTED and batch.weight is not None:
        out["weight"] = batch.weight
    if family in SMOOTHED and batch.smoothing:
        out["label_smoothing"] = batch.smoothing
    return out


# --- checks ----------------------------------------------------------------------------------


@dataclasses.dataclass
class Outcome:
    """The failures of a call (one line each), the largest observed P1 constant c (|sum g|
    beyond the absolute allowance, in eps_out sum |g|) and the rows inside documented limits."""

    failures: list[str] = dataclasses.field(default_factory=list)
    c: float = 0.0
    documented: list[str] = dataclasses.field(default_factory=list)


def _spread_beyond(high: float, low: float, work: str) -> bool:
    return math.isfinite(high) and math.isfinite(low) and high / 2 - low / 2 > LIMITS[work].max / 2


def outside_documented_range(
    family: str, batch: Batch, i: int, work: str, framework: str, device: str = "cpu"
) -> str | None:
    """Why row i is outside the documented range of the loss (docs/API.md, Known limits),
    or None. Such rows are left out of the assertions (the strategies still generate them;
    test_documented_limits pins the limits themselves). finfo and eps are of the work dtype,
    spreads are over the finite scores:

    - Extreme spreads: a competitor more than finfo.max above the label (or a masked label,
      the remaining scores more than finfo.max apart) for the smoothed hinges,
      proximal_cross_entropy and multilabel_cross_entropy; scores more than finfo.max apart
      for the distillation losses, but plain distillation in NumPy's native CPU kernel.
    - Sums near overflow: scores spanning more than finfo.max / classes, for
      topk_cross_entropy with k >= 3, the smoothed multilabel_hinge and topk_hinge (k >= 2),
      multilabel_cross_entropy and proximal_distillation.
    - Top-k cross-entropy at extreme spreads: k >= 2 and scores spanning more than 2^63, on
      CUDA (the CPU kernels take the capped mean score exactly).
    """
    z = batch.logits[i]
    finite = z[np.isfinite(z)]
    if len(finite) < 2:
        return None
    lim = LIMITS[work]
    half = finite.max() / 2 - finite.min() / 2  # half the spread: no overflow
    k = parameters(family, batch)["k"]
    smoothed = family in HINGES and parameters(family, batch)["gamma"] > 0
    near = {
        "topk_cross_entropy": k >= 3,
        "multilabel_hinge": smoothed,
        "topk_hinge_alpha": smoothed and k >= 2,
        "topk_hinge_beta": smoothed and k >= 2,
        "proximal_distillation": True,
        "multilabel_cross_entropy": True,
    }
    if near.get(family) and half > lim.max / 2 / batch.classes:
        return "sums near overflow"
    if family == "topk_cross_entropy" and k >= 2 and half > 2.0**62 and device == "cuda":
        return "top-k cross-entropy at extreme spreads"
    if family in TEACHER:
        t = target(family, batch)[i]
        rows = [finite, t[np.isfinite(t)]]
        native = framework == "numpy" and family == "distillation"
        if not native and any(len(r) and _spread_beyond(r.max(), r.min(), work) for r in rows):
            return "extreme spreads (distillation)"
        return None
    if not (smoothed or family in ("proximal_cross_entropy", "multilabel_cross_entropy")):
        return None
    if family in MULTILABEL:
        mask = target(family, batch)[i]
        labels, others = z[mask], z[~mask]
    else:
        y = int(batch.labels[i])
        labels, others = z[y : y + 1], np.delete(z, y)
    labels, others = labels[np.isfinite(labels)], others[np.isfinite(others)]
    if len(labels) and len(others) and _spread_beyond(others.max(), labels.min(), work):
        return "extreme spreads (competitor above the label)"
    if not len(labels) and len(others) and _spread_beyond(others.max(), others.min(), work):
        return "extreme spreads (masked label)"
    return None


def rounded_targets(family: str, batch: Batch, i: int, work: str) -> float:
    """The allowance of row i for targets that the work rounds (probabilities, label
    smoothing, multilabel_cross_entropy's 1 / |Y|): eps_work sum_j (u_j + T q_j) = 2 eps_work T,
    u the weighted smoothed targets and T their sum. The gradient T q - u then cancels to the
    rounding of u (docs/API.md, Known limits, Cancellation and Tiny probability targets); 0
    for other rows."""
    if family == "multilabel_cross_entropy":
        return 2 * LIMITS[work].eps  # the targets 1 / |Y| at the positives
    if not (family == PROBABILITIES or (family in SMOOTHED and batch.smoothing)):
        return 0.0
    t = np.asarray(target(family, batch)[i], np.float64)
    if t.ndim == 0:
        t = np.eye(batch.classes)[int(t)]
    u = (1 - batch.smoothing) * t + batch.smoothing / batch.classes
    if family in WEIGHTED and batch.weight is not None:
        u = u * batch.weight
    return 2 * LIMITS[work].eps * math.fsum(u.tolist())


def zero_sum_bound(
    g: np.ndarray,
    out: str,
    flush: bool,
    targets: float = 0.0,
    cuda: bool = False,
    teacher: bool = False,
) -> tuple[float, float, float]:
    """(|exact sum|, the P1 bound, the observed c) of a gradient row in dtype out; flush: XLA
    may flush subnormal entries (docs/API.md, Known limits, JAX on the CPU); targets: the
    allowance of rounded targets (rounded_targets), in units of C_LOW; cuda: the CUDA kernels'
    constant C_CUDA and two subnormal roundings per entry; teacher: the distillation losses'
    constant C_DISTILL."""
    lim = LIMITS[out]
    total = abs(math.fsum(g.tolist()))
    norm = math.fsum(np.abs(g).tolist())
    absolute = len(g) * eta(out) * (2 if cuda else 1) + (len(g) * lim.tiny if flush else 0.0)
    scale = lim.eps * norm + targets
    low = C_CUDA if cuda else C_LOW
    if teacher and not cuda:
        low = max(low, C_DISTILL)
    if out == "float64":
        relative = REL_FLOAT64 * norm + max(low, C_FLOAT64) * targets
    else:
        relative = low * scale
    c = max(0.0, total - absolute) / scale if scale else 0.0
    return total, relative + absolute, c


def check_p1(
    family: str,
    batch: Batch,
    values: np.ndarray,
    gradient: np.ndarray,
    *,
    out: str,
    work: str,
    framework: str,
    exempt: bool = True,
    device: str = "cpu",
) -> Outcome:
    """P1 and P1' of a call: values (n,) per row or one reduced value, gradient (n, classes)
    as float64 of the output dtype out; exempt: leave out the rows outside the documented
    range (outside_documented_range), and a reduction over them; device: cpu or cuda."""
    result = Outcome()
    outside: list[int] = []
    flush = framework in ("jax", "optax")
    values = np.atleast_1d(np.asarray(values, np.float64))
    reduced = len(values) != batch.rows or batch.reduction != "none"
    for i in range(batch.rows):
        g = gradient[i]
        value = 0.0 if reduced else float(values[i])
        reason = exempt and outside_documented_range(family, batch, i, work, framework, device)
        if reason:
            outside.append(i)
            result.documented.append(f"row {i}: {reason}")
            continue
        problems, c = [], 0.0
        if math.isnan(value) or value == -math.inf:
            problems.append(f"loss {value}")
        elif value < -nonnegative_allowance(family, batch, i, work):
            problems.append(f"negative loss {value!r} (P3)")
        if not np.isfinite(g).all():
            problems.append("gradient not finite")
        else:
            targets = rounded_targets(family, batch, i, work)
            total, bound, c = zero_sum_bound(
                g, out, flush, targets, cuda=device == "cuda", teacher=family in TEACHER
            )
            if total <= bound and np.count_nonzero(g) == 1:
                # The spec's underflow rule (the target entry is 0 where every other entry
                # underflows) holds to the absolute allowance: libsdca keeps a target entry
                # that the output dtype represents (docs/API.md, Known limits,
                # Representation). A lone entry beyond it fails the bound.
                result.documented.append(f"row {i}: lone entry within the allowance")
            if total > bound:
                problems.append(f"|sum g| = {total:.3g} > {bound:.3g} (c = {c:.3g})")
        if problems:
            result.failures.append(f"row {i}: {'; '.join(problems)}; {_row(family, batch, i, g)}")
        else:
            result.c = max(result.c, c)
    if reduced and not outside:
        value = float(values[0])
        allowance = sum(nonnegative_allowance(family, batch, i, work) for i in range(batch.rows))
        if math.isnan(value) or value == -math.inf or value < -allowance:
            result.failures.append(f"{batch.reduction} of the rows: loss {value!r}")
    return result


def nonnegative_allowance(family: str, batch: Batch, i: int, work: str) -> float:
    """P3: every loss is nonnegative by definition (cross-entropies of distributions or
    targets in [0, 1] scaled by nonnegative weights, hinges and their Moreau envelopes, which
    take the value 0 at p = 0 or u = z), so a row's loss is >= minus this allowance: C_P3
    eps_work of the scale of its terms, 1 + the largest finite |score| (over T, and with the
    teacher's, for distillation) times the row's target mass and weight. Only a loss whose
    exact value is within the allowance of 0 can round below 0."""
    z = batch.logits[i]
    scale = 1.0 + float(np.abs(z[np.isfinite(z)]).max(initial=0.0))
    if family in TEACHER:
        t = target(family, batch)[i]
        scale = (
            1.0 + (scale + float(np.abs(t[np.isfinite(t)]).max(initial=0.0))) / batch.temperature
        )
    mass = 1.0
    if family == PROBABILITIES:
        mass = max(1.0, float(np.sum(target(family, batch)[i])))
    if family in WEIGHTED and batch.weight is not None:
        mass *= float(batch.weight.max())
    return C_P3 * LIMITS[work].eps * scale * mass


def _entries(x: np.ndarray, limit: int = 16) -> str:
    return "[" + ", ".join(map(_fmt, x[:limit])) + (", ..." if len(x) > limit else "") + "]"


def _row(family: str, batch: Batch, i: int, g: np.ndarray) -> str:
    """Row i's logits, target and gradient (the first entries of wide rows)."""
    t = target(family, batch)[i]
    shown = _entries(t) if t.ndim else str(int(t))
    params = {k: v for k, v in parameters(family, batch).items() if k in _TAKES.get(family, ())}
    return f"{params} logits {_entries(batch.logits[i])} target {shown} gradient {_entries(g)}"


_TAKES = {
    **{name: ("k",) for name in TOPK},
    **{name: ("gamma",) for name in HINGES},
    "topk_hinge_alpha": ("k", "gamma"),
    "topk_hinge_beta": ("k", "gamma"),
    "proximal_cross_entropy": ("alpha",),
    "distillation": ("temperature",),
    "proximal_distillation": ("alpha", "temperature"),
}


def check_p2(
    family: str,
    batch: Batch,
    values: np.ndarray,
    gradient: np.ndarray,
    *,
    out: str,
    storage: str,
) -> Outcome:
    """P2: an exactly zero gradient of the distillation losses at identical logits; the
    documented floor of cross_entropy(x, softmax(x)) (p2_floor); no NaN loss."""
    result = Outcome()
    values = np.asarray(values, np.float64)
    for i in range(batch.rows):
        g = gradient[i]
        problems = []
        if np.isnan(values[i]):
            problems.append("loss NaN")
        if family in TEACHER:
            if not (g == 0).all():
                problems.append(f"nonzero gradient, largest {np.abs(g).max()!r}")
        elif not np.isfinite(g).all():
            # Checked first: a NaN entry makes every comparison with the floor false.
            problems.append("gradient not finite")
        else:
            t = target(family, batch)[i]
            excess, c = p2_floor(g, t, batch.logits[i], storage, out)
            result.c = max(result.c, c)
            if not excess <= 0:
                problems.append(f"entry beyond the floor by {excess!r}")
        if problems:
            result.failures.append(f"row {i}: {'; '.join(problems)}; {_row(family, batch, i, g)}")
    return result


def p2_floor(
    g: np.ndarray, t: np.ndarray, z: np.ndarray, storage: str, out: str
) -> tuple[float, float]:
    """(the largest excess over the floor, the observed constant) of the gradient T q - t of
    cross_entropy(z, t), t = softmax(z) rounded to the storage dtype and q = softmax(z) of the
    work: each entry within (|T - 1| + C_P2 eps (1 + |z_j - lse(z)|)) max(t_j, q_j) + tiny,
    T = sum t, with the larger eps and tiny of the storage and output dtypes: the rounding of
    t and of the exponent of q (docs/API.md, Known limits, Cancellation). An entry with
    t_j = q_j = 0 has the floor tiny alone (its condition number can overflow, and 0 times
    inf would hide the other entries); a NaN excess counts as infinite."""
    eps = max(LIMITS[storage].eps, LIMITS[out].eps)
    tiny = max(LIMITS[storage].tiny, LIMITS[out].tiny)
    t = np.asarray(t, np.float64)
    q = softmax(z)
    finite = np.isfinite(z)
    top = z[finite].max()
    # Rows wider than finfo.max overflow the differences: their floor is then infinite.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lse = top + math.log(math.fsum(np.exp(z[finite] - top).tolist()))
        condition = np.where(finite, 1 + np.abs(np.where(finite, z, 0.0) - lse), 0.0)
        m = np.maximum(t, q)
        drift = abs(math.fsum(t.tolist()) - 1) * m + tiny
        scale = np.where(m > 0, eps * m * condition, 0.0)
        over = np.abs(g) - drift - C_P2 * scale
        excess = float(np.where(np.isnan(over), np.inf, over).max())
        c = np.where((scale > 0) & np.isfinite(scale), (np.abs(g) - drift) / scale, 0.0)
    return excess, float(np.nanmax(np.maximum(c, 0.0)))


_RECORDED: dict[str, float] = {}
_REASONS: dict[str, set[str]] = {}


def record(test: str, outcome: Outcome) -> None:
    """Append a new largest observed c of a test, and rows inside documented limits, to the
    JSON lines file that SDCA_PROPERTY_RECORD names (the record of a run), if set."""
    path = os.environ.get("SDCA_PROPERTY_RECORD")
    larger = outcome.c > _RECORDED.get(test, 0.0)
    reasons = sorted({d.split(": ", 1)[1] for d in outcome.documented} - _REASONS.get(test, set()))
    if not path or not (larger or reasons):
        return
    _RECORDED[test] = max(outcome.c, _RECORDED.get(test, 0.0))
    _REASONS.setdefault(test, set()).update(reasons)
    line = {"test": test, "c": outcome.c, "documented": reasons}
    with open(path, "a") as f:
        f.write(json.dumps(line) + "\n")


def report(outcome: Outcome, label: str, batch: Batch) -> str:
    lines = [f"{label}: {batch!r}", *outcome.failures[:20]]
    if len(outcome.failures) > 20:
        lines.append(f"... {len(outcome.failures) - 20} more")
    return "\n".join(lines)


# --- explicit examples -----------------------------------------------------------------------

F64_MAX = 1.7976931348623157e308
F32_MAX = 3.4028234663852886e38
INF = math.inf


def _wide(classes: int, rows: int, seed: int) -> Batch:
    """Rows of a wide vocabulary: Gaussian logits (scale 3), a flat margin of 37 at the
    label, half the classes masked (the label too in the last row), and a spread of 2e38."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((rows, classes)) * 3
    labels = rng.integers(0, classes, rows)
    if rows > 1:
        z[1] = 0.0
        z[1, labels[1]] = 37.0
    if rows > 2:
        z[2, 1::2] = -INF
        labels[2] = 0
    if rows > 3:
        z[3, ::2] = -INF
        labels[3] = 0
    teacher = rng.standard_normal((rows, classes)) * 3
    probabilities = np.array([softmax(row) for row in teacher])
    name = f"{rows}x{classes}"
    return Batch.of(z, labels, teacher=teacher, probabilities=probabilities, name=name)


def _flat(classes: int, value: float, top: float = 0.0, name: str = "") -> Batch:
    z = np.full(classes, value)
    z[0] = top
    return Batch.of(z, [0], name=name or f"1x{classes} [{top}, {value}, ...]")


EXAMPLES: list[Batch] = [
    # Fails standard PyTorch (P1), with either target.
    Batch.of([0.0, 37.0], [0]),
    Batch.of([0.0, 37.0], [1]),
    # NaN Jacobian of a softmax that shifts by the largest logit (the shift overflows).
    Batch.of([F64_MAX, -9.9792015476736e291], [0]),
    Batch.of([F64_MAX, -9.9792015476736e291], [1]),
    Batch.of([1e308, -1e308], [1]),
    # An identical-logits (P2) example, here as P1.
    Batch.of([0.0, 1.0], [0]),
    # A masked row (causal attention masking).
    Batch.of([5.0, -INF, -INF], [0]),
    Batch.of([5.0, -INF, -INF], [1]),
    # The collapse example (float32).
    Batch.of([0.0, -20.0], [0]),
    # Subnormal results, near the underflow margins.
    Batch.of([[0.0, -103.97], [0.0, -745.13]], [0, 0]),
    Batch.of([[0.0, -40.0, -40.0, -41.0], [0.0, -18.0, -55.0, -20.0]], [0, 0]),
    _flat(1000, -106.0, name="tail underflow 1x1000 [0, -106, ...]"),
    _flat(1000, -750.0, name="float64 terms underflow 1x1000 [0, -750, ...]"),
    # Margins that cancel (top-2 alpha hinge), top-k at extreme spreads.
    Batch.of([F32_MAX, F32_MAX / 2, -F32_MAX / 2, 0.0], [1], k=2),
    Batch.of([F64_MAX, F64_MAX / 2, -F64_MAX / 2, 0.0], [1], k=2),
    Batch.of([[0.0, 1e16, -1e16, -1e16], [0.0, 1e300, -1e300, -1e300]], [0, 0], k=2),
    # Cancellation in the hinge (float32), nonsmooth and smoothed.
    Batch.of([1.0, 1e-8], [0], gamma=0.0),
    Batch.of([1.0, 1e-8], [0], gamma=0.5),
    # Wide rows: the CUDA kernels take other paths above 4096 classes.
    _wide(1000, 4, 1000),
    _wide(32000, 2, 32000),
    # Sums near overflow (documented limits):
    Batch.of([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 8.98846567431158e307], [0], k=5),
    Batch.of([0.0, 8.98846567431158e307, 0.0, 0.0], [0], positives=[1, 0, 0, 0], gamma=0.5),
    Batch.of(
        [0.0, 0.0, 2.9937604643020797e292, 1.7976931348623155e308],
        [0],
        positives=[1, 0, 0, 0],
        gamma=0.5,
    ),
    Batch.of(
        [0.0] * 17 + [-1.9974368165136835e307, 0.0],
        [0],
        positives=[1] * 17 + [0, 1],
        gamma=0.5,
    ),
    Batch.of([0.0, -8.988465674311579e307, -8.98846567431158e307, 0.0, 0.0, 0.0], [0], gamma=0.5),
    # Masked labels at extreme spreads (smoothed hinges: documented NaN).
    Batch.of([-INF, -F64_MAX, 9.9792015476736e291], [0], gamma=0.5),
    Batch.of([-INF, F64_MAX, -9.9792015476736e291], [0], gamma=0.5),
    # Wide top-k cross-entropy rows (the CPU kernels take the capped mean score exactly),
    # proximal distillation at finfo.max (a documented limit).
    Batch.of([1e10, 1e20, 0.0, 0.0], [1], k=2),
    Batch.of([1.0368059999386634e275, 2.0998790532060954e282, 0.0, 0.0], [1], k=2),
    Batch.of([0.0, F64_MAX, 0.0], [0], teacher=[-30005.57073612184, -INF, -30006.076200380194]),
    Batch.of([0.0, 0.0, 0.0, F64_MAX], [0], teacher=[-0.5, 0.0, 0.0, -INF], alpha=10.0),
    # Rounded probability targets and label smoothing (cancellation of T q - u).
    Batch.of([0.0] * 13, [0], probabilities=[1 / 13] * 13),
    Batch.of([0.0, 0.0], [0], probabilities=[0.5, 0.5781669020652771]),
    Batch.of([0.0, 0.0, 0.0], [0], probabilities=[0.0, 0.0, 0.0], smoothing=0.1),
    # Student classes masked where the teacher has probability (composed: cancellation).
    Batch.of([0.0] + [-INF] * 6, [0], teacher=[15.803755760192871] + [0.0] * 6),
    Batch.of([-INF, -INF, -INF, 0.0, -INF, -INF], [3], teacher=[0.0, 0.0, 0.0, 12.0, 0.0, 0.0]),
    # Distillation rows summing to more than 1 eps of their 1-norm (C_DISTILL).
    Batch.of([9.703125] + [0.0] * 31, [0], teacher=[-INF] * 15 + [0.0] + [-INF] * 16),
    Batch.of(
        [[13.5] + [0.0] * 32, [0.0] * 33],
        [0, 0],
        teacher=[[0.0] * 15 + [16.0] + [0.0] * 17, [0.0] * 33],
    ),
    Batch.of(
        [[5.5] + [0.0] * 33, [0.0] * 34],
        [0, 0],
        teacher=[[0.0] * 15 + [19.0] + [0.0] * 18, [0.0] * 34],
    ),
    # Rounding of float32 gradients: smoothed multilabel hinge, multilabel cross-entropy.
    Batch.of([0.0] * 10, [0], positives=[1] + [0] * 9, gamma=2.0),
    Batch.of([0.0] * 44, [0], positives=[1] * 44),
]

P2_EXAMPLES: list[Batch] = [
    Batch.of([0.0, 1.0], [0]),
    Batch.of([0.0, 1.0], [0], temperature=4.0),
    Batch.of([0.0, 37.0], [0]),
    Batch.of([F64_MAX, -9.9792015476736e291], [0]),
    Batch.of([1e308, -1e308], [0], temperature=4.0),
    Batch.of([5.0, -INF, -INF], [0]),
    _wide(1000, 64, 64),
    dataclasses.replace(_wide(1000, 64, 64), temperature=4.0, name="64x1000 T=4"),
    _wide(32000, 4, 4),
]


def with_examples(examples: Sequence[Batch]) -> Callable[[Any], Any]:
    """Hypothesis @example(batch=...) of each example."""
    from hypothesis import example

    def decorate(test: Any) -> Any:
        for batch in reversed(examples):
            test = example(batch=batch)(test)
        return test

    return decorate


# --- the tests -------------------------------------------------------------------------------


def _composed_marks() -> list[Any]:
    """The marks of the composed cells: xfail (their failures are reported separately;
    --runxfail reports them in full) and long (CI deselects them: compiling the composed
    programs for a few hundred shapes takes tens of minutes on 4 cores, for cells that run in
    the release-time search)."""
    import pytest

    reason = (
        'backend="composed" runs the framework\'s operations: its failures are reported '
        "separately (docs/API.md, Known limits); --runxfail reports them in full"
    )
    return [pytest.mark.xfail(reason=reason, strict=False), pytest.mark.long]


COMPOSED = _composed_marks()

# call(backend, family, mode, batch) -> (values, gradient as float64, gradient dtype name),
# backend = (framework, device, option).
Call = Callable[[tuple[str, str, str], str, str, Batch], tuple[np.ndarray, np.ndarray, str]]


def note(outcome: Outcome) -> None:
    """Hypothesis statistics: the rows outside the documented range and other documented
    notes; the largest c is maximized (target)."""
    from hypothesis import currently_in_test_context, event
    from hypothesis import target as maximize

    if currently_in_test_context():
        for reason in sorted({d.split(": ", 1)[1] for d in outcome.documented}):
            event(f"documented: {reason}")
        maximize(outcome.c, label="c")


def outside_kernel_coverage(family: str, batch: Batch, backend: tuple[str, str, str]) -> str | None:
    """Why backend="kernel" on CUDA computes no such call (docs/TORCH.md, Backends: the
    smooth top-k hinges take k <= 32, the other top-k losses k <= 256 or, for top-k
    cross-entropy, k = C - 1), or None."""
    if backend[1] != "cuda" or backend[2] != "kernel" or family not in TOPK:
        return None
    params = parameters(family, batch)
    k = params["k"]
    if family == "topk_cross_entropy":
        covered = k <= 256 or k == batch.classes - 1
    else:
        covered = k <= (32 if params["gamma"] > 0 else 256)
    return None if covered else f"outside the CUDA kernels' coverage (k = {k})"


def run_p1(
    call: Call,
    backend: tuple[str, str, str],
    family: str,
    mode: str,
    batch: Batch,
    exempt: bool = True,
) -> Outcome:
    dtype, precision = MODES[mode]
    batch = batch.cast(dtype)
    if backend[0] in ("numpy", "optax"):
        batch = dataclasses.replace(batch, reduction="none")
    if backend[0] == "optax":
        batch = batch.plain()
    work = work_dtype(dtype, precision)
    reason = outside_kernel_coverage(family, batch, backend)
    if reason:  # docs/TORCH.md, Backends: backend="kernel" raises NotImplementedError
        outcome = Outcome(documented=[f"row 0: {reason}"])
        record(f"p1 {family} {mode} {'-'.join(backend)}", outcome)
        return outcome
    values, gradient, out = call(backend, family, mode, batch)
    outcome = check_p1(
        family,
        batch,
        values,
        gradient,
        out=out,
        work=work,
        framework=backend[0],
        exempt=exempt,
        device=backend[1],
    )
    record(f"p1 {family} {mode} {'-'.join(backend)}", outcome)
    return outcome


def run_p2(
    call: Call, backend: tuple[str, str, str], family: str, mode: str, batch: Batch
) -> Outcome:
    dtype, _ = MODES[mode]
    batch = p2_batch(batch.cast(dtype), family).cast(dtype)
    values, gradient, out = call(backend, family, mode, batch)
    outcome = check_p2(family, batch, values, gradient, out=out, storage=dtype)
    record(f"p2 {family} {mode} {'-'.join(backend)}", outcome)
    return outcome


def _test(call: Call, kind: str, family: str, mode: str, backends: list[Any], name: str) -> Any:
    import pytest
    from hypothesis import given

    run = run_p1 if kind == "p1" else run_p2

    def test(backend: tuple[str, str, str], batch: Batch) -> None:
        outcome = run(call, backend, family, mode, batch)
        note(outcome)
        assert not outcome.failures, report(outcome, f"{family} {mode}", batch)

    test.__name__ = test.__qualname__ = name
    plain = bool(backends) and all(b.values[0][0] == "optax" for b in backends)
    test = with_examples(EXAMPLES if kind == "p1" else P2_EXAMPLES)(test)
    draw = "cross_entropy" if kind == "p2" and family == PROBABILITIES else family
    rich = kind == "p1" and not plain
    test = given(batch=batches(draw, mode, options=rich, reductions=rich))(test)
    return pytest.mark.slow(pytest.mark.parametrize("backend", backends)(test))


def install(
    namespace: dict[str, Any],
    call: Call,
    backends: list[Any],
    *,
    families: Sequence[str] = FAMILIES,
    p2_families: Sequence[str] = P2_FAMILIES,
    modes: Sequence[str] = tuple(MODES),
    prefix: str = "",
) -> None:
    """Define test_<prefix>p1_<family>_<mode> and test_<prefix>p2_<family>_<mode> in a test
    module: Hypothesis tests of one loss and dtype mode over backends (pytest params)."""
    for kind, names in (("p1", families), ("p2", p2_families)):
        for family in names:
            for mode in modes:
                name = f"test_{prefix}{kind}_{family}_{mode.replace('-', '_')}"
                namespace[name] = _test(call, kind, family, mode, backends, name)


def sweep(call: Call, kind: str, family: str, mode: str, backend: tuple[str, str, str]) -> None:
    """The batched sweep of a loss and mode on a backend: every failing row, one line each."""
    failures = []
    draw = "cross_entropy" if kind == "p2" and family == PROBABILITIES else family
    run = run_p1 if kind == "p1" else run_p2
    for batch in sweep_batches(draw, mode, options=kind == "p1"):
        outcome = run(call, backend, family, mode, batch)
        label = f"{family} {mode} {batch.rows} x {batch.classes}"
        failures += [f"{label} {line}" for line in outcome.failures]
    assert not failures, "\n".join(failures[:40] + [f"{len(failures)} rows failed"])


# The documented limits (docs/API.md, Known limits), one input each: (id, loss, dtype mode,
# batch, the reason of outside_documented_range, the exact loss or None). test_documented_limits
# asserts P1, P1' and the loss (relative 1e-6) on the CPU kernels, which each case fails
# (strict xfail: a fix makes it pass, and the docs and outside_documented_range must change
# with it; not strict for the smoothed hinges, which fail on macOS arm64 only);
# test_documented_limit_rows that the helper leaves each out.
TWO_1023 = 2.0**1023
NEAR = "sums near overflow"
DOCUMENTED_LIMITS: list[tuple[str, str, str, Batch, str, float | None]] = [
    *(
        (
            f"topk_ce-k{k}-near-overflow",
            "topk_cross_entropy",
            "float64",
            Batch.of([F64_MAX, -F64_MAX] + [0.0] * 14, [1], k=k),
            NEAR,
            math.inf,  # about (1 + 1/k) finfo.max: nan instead of inf
        )
        for k in (3, 4, 5)
    ),
    (
        "multilabel_hinge-near-overflow",
        "multilabel_hinge",
        "float64",
        Batch.of([0.0, TWO_1023, 0.0, 0.0], [0], positives=[1, 0, 0, 0], gamma=0.5),
        NEAR,
        None,
    ),
    (
        "multilabel_hinge-18-positives",
        "multilabel_hinge",
        "float64",
        Batch.of([0.0] * 17 + [-2e307, 0.0], [0], positives=[1] * 17 + [0, 1], gamma=0.5),
        NEAR,
        0.0,
    ),
    *(
        (
            f"{name}-near-overflow",
            name,
            "float64",
            Batch.of([0.0, -TWO_1023, -TWO_1023, 0.0, 0.0, 0.0], [0], k=2, gamma=0.5),
            NEAR,
            11 / 12,  # as without the two classes at -2^1023 (both variants)
        )
        for name in ("topk_hinge_alpha", "topk_hinge_beta")
    ),
    (
        "multilabel_ce-beyond-max",
        "multilabel_cross_entropy",
        "float64",
        Batch.of([1e308, -1e308], [1], positives=[0, 1]),
        NEAR,
        math.inf,
    ),
    (
        "multilabel_ce-near-overflow",
        "multilabel_cross_entropy",
        "float64",
        Batch.of([0.0] * 10 + [1.7976931348623163e307], [0], positives=[1] * 11),
        NEAR,
        None,
    ),
    (
        "proximal_distillation-at-max",
        "proximal_distillation",
        "float64",
        Batch.of([0.0, F64_MAX, 0.0], [0], teacher=[-30005.57073612184, -INF, -30006.076200380194]),
        NEAR,
        None,
    ),
]


def install_documented(namespace: dict[str, Any], call: Call, backends: list[Any]) -> None:
    """Define test_documented_limits and test_documented_limit_rows over the CPU backends."""
    import pytest

    limit = "documented limit, docs/API.md#known-limits"
    # The smoothed hinges' near-overflow rows fail on macOS arm64 and pass on some x86-64
    # kernels: not strict for them.
    cases = [
        pytest.param(
            *case[1:],
            id=case[0],
            marks=pytest.mark.xfail(strict=case[1] not in HINGES, reason=limit),
        )
        for case in DOCUMENTED_LIMITS
    ]
    rows = [pytest.param(*case[1:], id=case[0]) for case in DOCUMENTED_LIMITS]
    names = ("family", "mode", "batch", "reason", "expected")

    @pytest.mark.parametrize("backend", backends)
    @pytest.mark.parametrize(names, cases)
    def test_documented_limits(
        family: str,
        mode: str,
        batch: Batch,
        reason: str,
        expected: float | None,
        backend: tuple[str, str, str],
    ) -> None:
        outcome = run_p1(call, backend, family, mode, batch, exempt=False)
        assert not outcome.failures, report(outcome, f"{family} {mode}", batch)
        if expected is not None:
            values = call(backend, family, mode, batch.cast(MODES[mode][0]))[0]
            assert float(values[0]) == pytest.approx(expected, rel=1e-6, abs=1e-12)

    @pytest.mark.parametrize("backend", backends)
    @pytest.mark.parametrize(names, rows)
    def test_documented_limit_rows(
        family: str,
        mode: str,
        batch: Batch,
        reason: str,
        expected: float | None,
        backend: tuple[str, str, str],
    ) -> None:
        dtype, precision = MODES[mode]
        work = work_dtype(dtype, precision)
        found = outside_documented_range(family, batch.cast(dtype), 0, work, backend[0])
        assert found == reason

    namespace["test_documented_limits"] = test_documented_limits
    namespace["test_documented_limit_rows"] = test_documented_limit_rows
