"""Kernel sharing across the CPU interfaces: the same inputs give byte-identical losses and
gradients through every interface.

The interfaces:

- capi: the C ABI (sdca_loss_evaluate) through ctypes, with masks and label sets;
  capi_values: the same without gradients (values only, as the frameworks' forward);
- numpy: libsdca.loss_and_grad;
- rust: Loss::loss_and_grad_with (rust/tests/cross_interface.rs, run by cargo);
- cpp_kernel: sdca::kernel::evaluate on the batch, with the class weights, ignore_index and
  label smoothing of the frameworks; cpp_loss: sdca/loss.h, one row at a time
  (tests/cpp/cross_interface.cpp, compiled once into the libsdca cache);
- torch: libsdca.torch, backend="kernel" (the CPU extension, or with SDCA_CPU_EXTENSION=off
  the native library: test_native_route_torch.py);
- jax: libsdca.jax, backend="kernel" (the CPU FFI handler, or with SDCA_JAX_FFI=off the
  native library through a host callback: test_native_route_jax.py).

The inputs are written once per session (cases.txt and raw buffers in a temporary directory,
shared by the xdist workers; interfaces.py has the format); Rust and C++ read the files, the Python interfaces the same
arrays. Each case compares an interface with the reference of its storage dtype (capi for
float32 and float64, cpp_kernel for float16, bfloat16 and the framework options) to the
byte: values, gradients, signs of zero and NaN positions. Every loss family runs on float32
and float64 (and the 16-bit types where an interface takes them), on 1-D, 2-D and (N, C, H,
W) logits (classes on dim 1), on one thread and the default threads (rows enough for
several). The rows: finite, ties, large margins of either sign, all zeros, and scores of -inf
at classes without target mass; the subnormal cases (SUBNORMAL_KINDS) put losses and gradient
entries in the output dtype's subnormal range, with negative entries whose sign of zero
matters.

Where interfaces differ by design, a case compares what they share; EXCEPTIONS names each such
difference and why. FINDINGS are the disagreements found: their cases xfail (not strict: a
finding need not show in every case) and each has a strict minimal repro below. Rows of
masked classes with target mass give the +inf limit: pending.MASKED_LIMIT_ROW_KINDS lists
them, generated with pending.MASKED_LIMITS.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib
import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

import interfaces
import libsdca
import numpy as np
import pending
import pytest
from interfaces import NATIVE_DTYPES, Session, represent, storage

# What the cases compare differently, by design, and why.
EXCEPTIONS = {
    "framework_value_dtype": (
        "PyTorch and JAX return per-example losses in the logits' dtype promoted to float32 "
        "(float64 for float64 logits), the C ABI, NumPy, Rust and C++ in float64: the "
        "framework's values are compared with the reference's rounded once to float32."
    ),
    "framework_reductions": (
        "reductions are done by the frameworks (PyTorch sums float64 values, JAX the values "
        "in their dtype) and the native interfaces have none: every case is "
        "reduction='none', the per-example outputs."
    ),
    "loss_h_options": (
        "sdca/loss.h takes no class weights, ignore_index or label smoothing (every other "
        "interface does): its rows have the losses' own parameters only."
    ),
    "rust_ignore": "Rust labels are usize: it has no label for an ignore_index of -100.",
    "native_16bit": (
        "the C ABI and Rust store float32 and float64 only, and sdca/loss.h takes the 16-bit "
        "types in no public function: 16-bit cases compare cpp_kernel, torch, jax and numpy."
    ),
    "numpy_16bit": (
        "NumPy casts 16-bit logits to float32 storage (docs/API.md#precision), so its "
        "gradient is float32: compared after rounding it once to the 16-bit type."
    ),
    "loss_h_scope": (
        "sdca/loss.h computes one row in the storage's precision, of labels, label sets or "
        "teacher logits: no probability targets and no precision option."
    ),
    "framework_masks": (
        "PyTorch and JAX take multilabel targets as masks only: the mask cases cover them, "
        "the label-set cases the native interfaces."
    ),
    "probability_storage": (
        "probability targets reach the kernels as float64 from the C ABI, NumPy, Rust and "
        "C++, and in the logits' dtype from PyTorch and JAX; the test's probabilities are "
        "multiples of 1/16, equal in every dtype."
    ),
    "jax_threads": (
        "the JAX CPU handler takes no per-call thread count (SDCA_NUM_THREADS is read once "
        "per process): it runs the default-thread cases only."
    ),
    "jax_time_budget": (
        "each JAX case compiles its XLA programs (about 0.15 s): float64 runs JAX on 2-D "
        "logits, the 1-D, 4-D and big layouts run it in float32."
    ),
    "external_blas": "the losses call no BLAS, so the BLAS backend cannot make them differ.",
}


DTYPES = ("float32", "float64", "float16", "bfloat16")
INTERFACES = ("capi", "capi_values", "numpy", "rust", "cpp_kernel", "cpp_loss", "torch", "jax")
LABEL_SETS = ("rust", "cpp_loss")  # interfaces that take multilabel targets as label sets
IGNORE = -100
# float16 and bfloat16 hold the class weights exactly.
WEIGHTS = (0.5, 1.0, 2.0, 0.75, 1.5, 0.0, 3.0)
CLASSES = 7
N = 14  # 2-D rows: every row kind twice
SHAPE_4D = (2, 2, 3)  # (N, H, W) of the (N, C, H, W) logits
BIG = (2600, 101)  # rows and classes enough for several threads
DISTINCT = 182  # 26 rows of each row kind; the big inputs repeat them


@dataclasses.dataclass(frozen=True)
class Family:
    """A loss: its SDCA_LOSS_* kind, parameters and target."""

    name: str
    kind: int
    target: str  # labels, mask, probs or teacher
    k: int = 1
    gamma: float = 0.0
    temperature: float = 1.0

    @property
    def base(self) -> str:
        """The name without the variant suffixes _smooth, _probs and _t1."""
        for suffix in ("_smooth", "_probs", "_t1"):
            if self.name.endswith(suffix):
                return self.name.removesuffix(suffix)
        return self.name


FAMILIES = (
    Family("cross_entropy", 0, "labels"),
    Family("cross_entropy_probs", 0, "probs"),
    Family("topk_cross_entropy", 1, "labels", k=3),
    Family("multiclass_hinge", 2, "labels"),
    Family("multiclass_hinge_smooth", 2, "labels", gamma=0.5),
    Family("topk_hinge_alpha", 3, "labels", k=2),
    Family("topk_hinge_alpha_smooth", 3, "labels", k=2, gamma=1.0),
    Family("topk_hinge_beta", 4, "labels", k=2),
    Family("topk_hinge_beta_smooth", 4, "labels", k=2, gamma=1.0),
    Family("multilabel_hinge", 5, "mask"),
    Family("multilabel_hinge_smooth", 5, "mask", gamma=0.5),
    Family("multilabel_cross_entropy", 6, "mask"),
    Family("proximal_cross_entropy", 7, "labels", gamma=0.7),
    Family("distillation", 8, "teacher", temperature=2.0),
    Family("distillation_t1", 8, "teacher"),
    Family("proximal_distillation", 9, "teacher", temperature=2.0, gamma=0.7),
)
BY_NAME = {f.name: f for f in FAMILIES}


@dataclasses.dataclass(frozen=True)
class Case:
    family: Family
    dtype: str
    shape: str  # 1d, 2d, 4d or big
    threads: int = 0  # 0: the default; 1: one thread
    weight: bool = False
    ignore: bool = False
    label_smoothing: float = 0.0
    precision: int = 0  # sdca_precision: 0 the storage's, 2 float64
    sets: bool = False  # multilabel targets as label sets (capi, numpy, cpp_kernel)
    subnormal: bool = False  # the rows of SUBNORMAL_KINDS

    @property
    def id(self) -> str:
        parts = [self.family.name, self.dtype, self.shape, f"t{self.threads}"]
        parts += ["weight"] * self.weight + ["ignore"] * self.ignore
        parts += [f"ls{self.label_smoothing}"] * bool(self.label_smoothing)
        parts += ["p64"] * bool(self.precision) + ["sets"] * self.sets
        parts += ["subnormal"] * self.subnormal
        return "-".join(parts)

    @property
    def key(self) -> str:
        """The name of the case's input files, shared by the cases of equal inputs."""
        parts = [self.family.name, self.dtype, self.shape]
        return "-".join(parts + ["ignore"] * self.ignore + ["subnormal"] * self.subnormal)

    @property
    def options(self) -> bool:
        """Whether the case takes options only the frameworks and the C++ kernels take."""
        return self.weight or self.ignore or bool(self.label_smoothing)

    @property
    def target(self) -> str:
        return "sets" if self.sets else self.family.target

    @property
    def rows(self) -> tuple[int, int]:
        if self.shape == "big":
            return BIG
        n = {"1d": 1, "2d": N, "4d": int(np.prod(SHAPE_4D))}[self.shape]
        return n, CLASSES


def _cases() -> list[Case]:
    cases = []
    for f in FAMILIES:
        for dtype in DTYPES:
            cases += [Case(f, dtype, "2d", threads) for threads in (1, 0)]
        for dtype in NATIVE_DTYPES:
            cases += [Case(f, dtype, "1d"), Case(f, dtype, "4d")]
            cases += [Case(f, dtype, "big", threads) for threads in (1, 0)]
        cases.append(Case(f, "float32", "2d", precision=2))
        if f.target == "mask":
            cases += [Case(f, dtype, "2d", sets=True) for dtype in NATIVE_DTYPES]
        if f.target == "labels":
            cases += [Case(f, dtype, "2d", weight=True, ignore=True) for dtype in DTYPES]
    ce, probs = BY_NAME["cross_entropy"], BY_NAME["cross_entropy_probs"]
    for dtype in DTYPES:
        cases += [
            Case(ce, dtype, "2d", weight=True),
            Case(ce, dtype, "2d", ignore=True),
            Case(ce, dtype, "2d", label_smoothing=0.1),
            Case(ce, dtype, "2d", label_smoothing=0.25),
            Case(ce, dtype, "2d", weight=True, ignore=True, label_smoothing=0.1),
            Case(probs, dtype, "2d", weight=True),
            Case(probs, dtype, "2d", label_smoothing=0.1),
            Case(probs, dtype, "2d", label_smoothing=0.25),
        ]
    cases.append(Case(ce, "float32", "4d", weight=True, ignore=True, label_smoothing=0.25))
    cases += [Case(f, dtype, "2d", subnormal=True) for f in FAMILIES for dtype in DTYPES]
    return cases


CASES = _cases()


@dataclasses.dataclass(frozen=True)
class Finding:
    """A disagreement found: the interfaces and cases where it shows, and a repro test."""

    description: str
    applies: Callable[[Case, str], bool]
    repro: str


FINDINGS = {
    "distillation_values_depend_on_gradients": Finding(
        "distillation (both temperatures) returns values one float64 ulp apart with and "
        "without gradients requested (sdca_loss_evaluate with gradients NULL, and the forward "
        "of PyTorch and JAX, which computes values only): in sdca/loss/distill.h the blocked "
        "pass takes the value_entry formula when !b.grad and teacher_entry in float64 work "
        "with gradients.",
        lambda c, i: c.family.kind == 8 and i in ("capi_values", "torch", "jax"),
        "test_finding_distillation_values_depend_on_gradients",
    ),
    "distillation_float64_shift_overflow": Finding(
        "distillation in float64 work of float32 scores reports a row invalid (NaN) whose "
        "correction exp(shift) times the far students' sum exceeds double although the loss is "
        "finite (shift 722 above log(max) 709.8; the float32 work gives 722.63): in "
        "sdca/loss/distill.h the blocked pass forms exp(shift) apart, which the exponential's "
        "former clamp at 709 turned into a wrong finite loss (709.66, gradients -1e9). The "
        "distillation rewrite's single log-space pass fixes it.",
        lambda c, i: c.family.name == "distillation_t1" and c.precision == 2,
        "test_finding_distillation_float64_shift_overflow",
    ),
}


# --- inputs ----------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Inputs:
    scores: np.ndarray  # (n, classes) float64, exact in the dtype
    labels: np.ndarray  # (n,) int64
    mask: np.ndarray  # (n, classes) bool
    probs: np.ndarray  # (n, classes) float64 multiples of 1/16
    teacher: np.ndarray  # (n, classes) float64, exact in the dtype
    weight: np.ndarray  # (classes,) float64


ROW_KINDS = ("finite", "ties", "margin", "negative_margin", "masked", "tied_margin", "zeros")


def _row(rng: np.random.Generator, kind: str, classes: int, k: int) -> dict[str, Any]:
    """One row of every target: scores, a label, positives, probabilities and a teacher;
    scores of -inf only at classes without target mass."""
    z = rng.normal(0, 3, classes)
    label = int(rng.integers(classes))
    others = [j for j in range(classes) if j != label]
    positives = {label, *rng.choice(others, int(rng.integers(0, 3)), replace=False).tolist()}
    teacher = rng.normal(0, 3, classes)
    masked: list[int] = []
    if kind == "ties":
        z = rng.integers(-2, 3, classes).astype(np.float64)
        teacher = z.copy()  # an exact zero gradient for distillation at T = 1
    elif kind in ("margin", "tied_margin"):
        z[sorted(positives)] += 1000
        teacher[label] += 1000
        if kind == "tied_margin":
            z[others[: k + 1]] = z[label]
    elif kind == "negative_margin":
        z[sorted(positives)] -= 1000
        teacher[label] -= 1000
    elif kind == "zeros":
        z[:] = 0
        teacher[:] = 0
    elif kind == "masked":
        negatives = [j for j in range(classes) if j not in positives]
        # Keep k + 1 finite classes beside the label and a finite negative.
        room = min(len(negatives) - 1, classes - 1 - (k + 1))
        masked = rng.choice(negatives, max(0, min(2, room)), replace=False).tolist()
        z[masked] = -np.inf
        teacher[masked] = -np.inf
        free = [j for j in range(classes) if j not in masked and j != label]
        teacher[free[0]] = -np.inf  # a teacher probability of 0 at a finite student class
    counts = np.zeros(classes)
    if kind in ("margin", "zeros"):
        counts[label] = 16
    else:
        np.add.at(counts, rng.choice([j for j in range(classes) if j not in masked], 16), 1)
    mask = np.zeros(classes, bool)
    mask[sorted(positives)] = True
    return dict(z=z, label=label, mask=mask, probs=counts / 16, teacher=teacher)


# Rows whose losses and gradient entries lie in the subnormal range of the output dtype:
# tail: competitors at gaps whose exponentials are subnormal, so that the label's entry is a
# negative subnormal (a flushed one is -0.0); near: one competitor near the label, the tail
# entries subnormal and the label's normal; deep (16-bit): entries that round to zero only
# in the storage, the label's to -0.0; teacher_tail: student and teacher tails apart, entries
# (q - p) / T subnormal of either sign; inputs: subnormal scores beside zeros.
SUBNORMAL_KINDS = ("tail", "near", "deep", "teacher_tail", "inputs")
# The gaps of the tails: exp(-gap) is subnormal in the output dtype (16-bit: its gradient).
SUBNORMAL_GAPS = {
    "float32": (88.0, 103.0),
    "bfloat16": (88.0, 103.0),
    "float64": (709.0, 744.0),
    "float16": (10.0, 16.5),
}
DEEP_GAPS = {"float16": (18.0, 24.0), "bfloat16": (104.0, 110.0)}
TINY = {"float32": 2.0**-140, "bfloat16": 2.0**-130, "float64": 2.0**-1060, "float16": 2.0**-20}


def _subnormal_row(rng: np.random.Generator, kind: str, classes: int, dtype: str) -> dict[str, Any]:
    label = int(rng.integers(classes))
    others = [j for j in range(classes) if j != label]
    lo, hi = (
        DEEP_GAPS.get(dtype, SUBNORMAL_GAPS[dtype]) if kind == "deep" else SUBNORMAL_GAPS[dtype]
    )
    z = np.zeros(classes)
    z[others] = -rng.uniform(lo, hi, len(others))
    teacher = z.copy()
    if kind == "near":
        z[others[0]] = -0.5
    elif kind == "teacher_tail":
        teacher[others] -= rng.uniform(-2, 2, len(others))
    elif kind == "inputs":
        z = np.zeros(classes)
        z[others[:3]] = np.array([1.0, -1.0, 3.0]) * TINY[dtype]
        teacher = np.zeros(classes)
    mask = np.zeros(classes, bool)
    mask[label] = True
    probs = mask.astype(np.float64)
    return dict(z=z, label=label, mask=mask, probs=probs, teacher=teacher)


def _pending_row(rng: np.random.Generator, kind: str, classes: int, k: int) -> dict[str, Any]:
    """The rows of pending.MASKED_LIMIT_ROW_KINDS."""
    row = _row(rng, "finite", classes, k)
    # A class weight of zero at a masked label makes the row invalid (0 * inf).
    while np.resize(WEIGHTS, classes)[row["label"]] == 0:
        row = _row(rng, "finite", classes, k)
    label = row["label"]
    if kind == "masked_label":
        row["z"][label] = -np.inf
    elif kind == "masked_positive":
        row["z"][np.flatnonzero(row["mask"])[0]] = -np.inf
    elif kind == "masked_target_mass":
        j = int(np.flatnonzero(row["probs"])[0])
        row["z"][j] = -np.inf
        row["teacher"][j] = 0.0
    elif kind == "topk_few_finite":
        row["z"][[j for j in range(classes) if j != label][: classes - k]] = -np.inf
    return row


@functools.cache
def _inputs(
    family: Family, dtype: str, shape: str, ignore: bool, subnormal: bool, n: int, classes: int
) -> Inputs:
    seed = hashlib.sha256(f"{family.name}/{dtype}/{shape}/{subnormal}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(seed[:8], "little"))
    kinds = list(ROW_KINDS) + list(pending.MASKED_LIMIT_ROW_KINDS) * pending.MASKED_LIMITS
    start = kinds.index("masked") if shape == "1d" else 0
    rows = []
    for i in range(min(n, DISTINCT)):
        if subnormal:
            kind = SUBNORMAL_KINDS[i % len(SUBNORMAL_KINDS)]
            rows.append(_subnormal_row(rng, kind, classes, dtype))
            continue
        kind = kinds[(start + i) % len(kinds)]
        make = _pending_row if kind in pending.MASKED_LIMIT_ROW_KINDS else _row
        rows.append(make(rng, kind, classes, family.k))
    rows = [rows[i % len(rows)] for i in range(n)]
    labels = np.array([r["label"] for r in rows], np.int64)
    if ignore:
        labels[1::4] = IGNORE
    arrays = Inputs(
        scores=represent(np.array([r["z"] for r in rows]), dtype),
        labels=labels,
        mask=np.array([r["mask"] for r in rows]),
        probs=np.array([r["probs"] for r in rows]),
        teacher=represent(np.array([r["teacher"] for r in rows]), dtype),
        weight=np.resize(np.array(WEIGHTS), classes),
    )
    for field in dataclasses.fields(arrays):
        getattr(arrays, field.name).flags.writeable = False
    return arrays


def make_inputs(case: Case) -> Inputs:
    """The inputs of a case: deterministic in its family, dtype, shape and ignore."""
    return _inputs(case.family, case.dtype, case.shape, case.ignore, case.subnormal, *case.rows)


def write_inputs(directory: Path, case: Case) -> None:
    """The input files of a case: <key>.scores, .labels, .weight and those of its target."""
    x, target = make_inputs(case), case.family.target
    interfaces.write_inputs(
        directory,
        case.key,
        case.dtype,
        x.scores,
        labels=x.labels,
        weight=x.weight,
        mask=x.mask if target == "mask" else None,
        probs=x.probs if target == "probs" else None,
        teacher=x.teacher if target == "teacher" else None,
    )


def manifest_line(case: Case) -> str:
    """The line of a case in cases.txt (interfaces.manifest_line)."""
    f, (n, classes) = case.family, case.rows
    return interfaces.manifest_line(
        case.id,
        case.key,
        f.kind,
        case.dtype,
        n,
        classes,
        case.target,
        k=f.k,
        gamma=f.gamma,
        temperature=f.temperature,
        label_smoothing=case.label_smoothing,
        ignore_index=IGNORE if case.ignore else None,
        weight=case.weight,
        threads=case.threads,
        precision=case.precision,
    )


def _prepare(directory: Path) -> None:
    written: set[str] = set()
    for case in CASES:
        if case.key not in written:
            write_inputs(directory, case)
            written.add(case.key)
    (directory / "cases.txt").write_text("".join(manifest_line(c) + "\n" for c in CASES))


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory) -> Session:
    """The inputs and the outputs of the compiled interfaces, once for all xdist workers."""
    return interfaces.shared_session(tmp_path_factory, "cross_interface", _prepare)


# --- the Python interfaces -------------------------------------------------------------------


Outputs = tuple[bytes, bytes]  # values, gradients


def sdca_loss_evaluate(
    family: Family,
    dtype: str,
    scores: np.ndarray,
    kind: str,
    target: np.ndarray,
    *,
    threads: int = 1,
    precision: int = 0,
    gradients: bool = True,
    **options: Any,
) -> tuple[np.ndarray, np.ndarray | None]:
    """interfaces.c_loss_evaluate of a family: values and (unless not gradients) gradients.
    kind: labels, sets or mask (of a mask target), probs or teacher; options: weight,
    label_smoothing and ignore_index."""
    # A loss takes only its own fields (SDCA_LOSS_INIT): k of the top-k losses, gamma of the
    # hinges and proximal losses, temperature of the distillation losses.
    result = interfaces.c_loss_evaluate(
        family.kind,
        dtype,
        scores,
        kind,
        target,
        k=family.k if family.kind in (1, 3, 4) else 0,
        gamma=family.gamma if family.kind in (2, 3, 4, 5, 7, 9) else 0.0,
        temperature=family.temperature if family.kind in (8, 9) else 1.0,
        threads=threads,
        precision=precision,
        gradients=gradients,
        **options,
    )
    assert result.code == 0 and result.invalid_rows == 0, (result.code, result.message)
    return result.values, result.grads


def run_capi(case: Case, x: Inputs, gradients: bool = True) -> Outputs:
    raw, _ = _framework_target(case, x)
    values, grads = sdca_loss_evaluate(
        case.family,
        case.dtype,
        x.scores,
        case.target,
        raw,
        threads=case.threads,
        precision=case.precision,
        gradients=gradients,
        **_native_options(case, x),
    )
    return values.tobytes(), b"" if grads is None else grads.tobytes()


def _native_options(case: Case, x: Inputs) -> dict[str, Any]:
    """The weight, label_smoothing and ignore_index of a case, as the C ABI and NumPy take
    them."""
    options: dict[str, Any] = {}
    if case.weight:
        options["weight"] = x.weight
    if case.label_smoothing:
        options["label_smoothing"] = case.label_smoothing
    if case.ignore:
        options["ignore_index"] = IGNORE
    return options


def _numpy_loss(f: Family) -> Any:
    base = f.base
    if base.startswith("topk_hinge_"):
        return libsdca.TopKHinge(k=f.k, variant=base.removeprefix("topk_hinge_"), gamma=f.gamma)
    cls = libsdca.LOSSES[base]
    taken = {field.name for field in dataclasses.fields(cls)}  # pyright: ignore[reportArgumentType]
    params = dict(k=f.k, gamma=f.gamma, temperature=f.temperature)
    return cls(**{name: value for name, value in params.items() if name in taken})


def run_numpy(case: Case, x: Inputs) -> Outputs:
    stored = case.dtype if case.dtype in NATIVE_DTYPES else "float32"
    if case.target == "labels":
        target: Any = x.labels
    elif case.target == "sets":
        target = [np.flatnonzero(row) for row in x.mask]
    elif case.target == "mask":
        target = x.mask
    elif case.target == "probs":
        target = libsdca.Probabilities(x.probs)
    else:
        target = x.teacher.astype(stored)
    values, grads = libsdca.loss_and_grad(
        x.scores.astype(stored),
        target,
        _numpy_loss(case.family),
        precision="float64" if case.precision == 2 else None,
        num_threads=case.threads or None,
        **_native_options(case, x),
    )
    assert grads.dtype == np.dtype(stored)
    # EXCEPTIONS["numpy_16bit"]: a float32 gradient for 16-bit cases, rounded once.
    return np.asarray(values, np.float64).tobytes(), storage(grads, case.dtype).tobytes()


def _framework_layout(case: Case, a: np.ndarray, per_class: bool) -> np.ndarray:
    """(n, ...) rows in the case's logits layout: 1-D one row, 4-D (N, C, H, W) with the
    rows in (N, H, W) order."""
    if case.shape == "1d":
        return a[0]
    if case.shape == "4d":
        a = a.reshape(*SHAPE_4D, *a.shape[1:])
        return np.ascontiguousarray(np.moveaxis(a, -1, 1)) if per_class else a
    return a


def _rows_of(case: Case, a: np.ndarray) -> np.ndarray:
    """The (n, classes) rows of a gradient in the case's logits layout."""
    if case.shape == "4d":
        a = np.moveaxis(a, 1, -1)
    return np.ascontiguousarray(a).reshape(case.rows)


def _framework_target(case: Case, x: Inputs) -> tuple[np.ndarray, bool]:
    """The framework target of a case and whether it has a class axis."""
    target = case.family.target
    if target == "labels":
        return x.labels, False
    return {"mask": x.mask, "probs": x.probs, "teacher": x.teacher}[target], True


def _framework_call(case: Case, module: Any, z: Any, target: Any, extra: dict[str, Any]) -> Any:
    f, base = case.family, case.family.base
    torch = module.__name__ == "libsdca.torch"
    axis = {("dim" if torch else "axis"): 1} if case.shape == "4d" else {}
    precision = "float64" if case.precision == 2 else None
    common: dict[str, Any] = dict(reduction="none", backend="kernel", precision=precision, **extra)
    if base == "cross_entropy":
        # PyTorch's cross_entropy takes the classes on dim 1 (0 of 1-D logits).
        return module.cross_entropy(z, target, **({} if torch else axis), **common)
    common.update(axis)
    if base == "topk_cross_entropy":
        return module.topk_cross_entropy(z, target, k=f.k, **common)
    if base == "multiclass_hinge":
        return module.multiclass_hinge_loss(z, target, gamma=f.gamma, **common)
    if base.startswith("topk_hinge_"):
        variant = base.removeprefix("topk_hinge_")
        return module.topk_hinge_loss(z, target, k=f.k, variant=variant, gamma=f.gamma, **common)
    if base == "multilabel_hinge":
        return module.multilabel_hinge_loss(z, target, gamma=f.gamma, **common)
    if base == "multilabel_cross_entropy":
        return module.multilabel_cross_entropy(z, target, **common)
    if base == "proximal_cross_entropy":
        return module.proximal_cross_entropy(z, target, gamma=f.gamma, **common)
    if base == "distillation":
        return module.distillation_loss(z, target, temperature=f.temperature, **common)
    return module.proximal_distillation_loss(
        z, target, temperature=f.temperature, gamma=f.gamma, **common
    )


def _options(case: Case) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if case.ignore:
        extra["ignore_index"] = IGNORE
    if case.label_smoothing:
        extra["label_smoothing"] = case.label_smoothing
    return extra


def run_torch(case: Case, x: Inputs) -> Outputs:
    torch = importlib.import_module("torch")
    L = importlib.import_module("libsdca.torch")
    kind = getattr(torch, case.dtype)
    z = torch.tensor(_framework_layout(case, x.scores, True)).to(kind).requires_grad_()
    raw, per_class = _framework_target(case, x)
    t = torch.tensor(_framework_layout(case, raw, per_class))
    t = t.to(kind) if t.is_floating_point() else t
    extra = _options(case)
    if case.weight:
        extra["weight"] = torch.tensor(x.weight).to(kind)
    # backend="kernel" raises NotImplementedError where the CPU extension does not compute it.
    with interfaces.torch_threads(torch, case.threads):
        value = _framework_call(case, L, z, t, extra)
        value.backward(torch.ones_like(value))
    assert z.grad is not None and z.grad.dtype == kind
    grad = z.grad.view(torch.int16) if case.dtype in ("float16", "bfloat16") else z.grad
    values = value.detach().numpy().reshape(-1)
    return values.tobytes(), _rows_of(case, grad.numpy()).tobytes()


def run_jax(case: Case, x: Inputs) -> Outputs:
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    J = importlib.import_module("libsdca.jax")
    x64 = case.dtype == "float64" or case.precision == 2
    with interfaces.x64(jax, x64), jax.default_device(jax.devices("cpu")[0]):
        kind = jnp.dtype(case.dtype)

        def floats(a: np.ndarray) -> Any:  # exact: the values are those of the dtype
            return jnp.asarray(a.astype(np.float64 if x64 else np.float32)).astype(kind)

        z = floats(_framework_layout(case, x.scores, True))
        raw, per_class = _framework_target(case, x)
        raw = _framework_layout(case, raw, per_class)
        if raw.dtype == np.float64:
            t = floats(raw)
        else:
            t = jnp.asarray(raw if raw.dtype == bool or x64 else raw.astype(np.int32))
        extra = _options(case)
        if case.weight:
            extra["weight"] = floats(x.weight)
        value, pullback = jax.vjp(lambda s: _framework_call(case, J, s, t, extra), z)
        (grad,) = pullback(jnp.ones_like(value))
    assert grad.dtype == kind
    values = np.asarray(value).reshape(-1)
    return values.tobytes(), _rows_of(case, np.asarray(grad)).tobytes()


# --- comparisons -----------------------------------------------------------------------------


def _read(session: Session, case: Case, interface: str) -> Outputs:
    """The outputs of a compiled interface (a call without invalid rows), removed unless the
    case's reference (each other output has one reader: its test)."""
    status = {"cpp_kernel": "status", "rust": "report"}.get(interface)
    paths = [session.path(case.id, interface, part) for part in ("values", "grads")]
    if status is not None:
        paths.append(session.path(case.id, interface, status))
        # cpp_kernel: status_ok and no invalid rows; rust: no invalid rows
        fields = paths[2].read_text().split()
        assert fields[: 2 if interface == "cpp_kernel" else 1] in (["0", "0"], ["0"]), fields
    error = session.path(case.id, interface, "error")
    assert not error.exists(), error.read_text()
    outputs = paths[0].read_bytes(), paths[1].read_bytes()
    if interface != reference(case):
        for path in paths:
            path.unlink()
    return outputs


def why_not(case: Case, interface: str) -> str | None:
    """The EXCEPTIONS entry that keeps an interface out of a case, or None."""
    if interface == "cpp_loss" and case.options:
        return "loss_h_options"
    if interface == "rust" and case.ignore:
        return "rust_ignore"
    if interface in ("capi", "capi_values", "rust", "cpp_loss") and case.dtype not in NATIVE_DTYPES:
        return "native_16bit"
    if interface == "cpp_loss" and (case.target == "probs" or case.precision):
        return "loss_h_scope"
    if interface in ("torch", "jax") and case.sets:
        return "framework_masks"
    if interface == "jax" and case.threads == 1:
        return "jax_threads"
    if interface == "jax" and case.dtype == "float64" and case.shape != "2d":
        return "jax_time_budget"
    return None


def reference(case: Case) -> str:
    """The interface every other one is compared with: the C ABI where it takes the case."""
    return "capi" if case.dtype in NATIVE_DTYPES and not case.options else "cpp_kernel"


def pairs() -> list[Any]:
    """The (case, interface) pairs to compare: every pair but the reference and why_not's."""
    return [
        pytest.param(case, interface, id=f"{case.id}-{interface}")
        for case in CASES
        for interface in INTERFACES
        if interface != reference(case) and why_not(case, interface) is None
    ]


def _first_difference(got: bytes, want: bytes, dtype: np.dtype, classes: int) -> str:
    a, b = np.frombuffer(got, dtype), np.frombuffer(want, dtype)
    if a.size != b.size:
        return f"{a.size} elements, expected {b.size}"
    raw = np.frombuffer(got, np.uint8).reshape(a.size, -1)
    ref = np.frombuffer(want, np.uint8).reshape(b.size, -1)
    differ = np.flatnonzero((raw != ref).any(1))
    i = int(differ[0])
    where = f"element {i} (row {i // classes}, class {i % classes})" if classes else f"row {i}"
    return (
        f"{differ.size} of {a.size} differ; first at {where}: "
        f"0x{raw[i][::-1].tobytes().hex()} ({a[i]!r}), "
        f"expected 0x{ref[i][::-1].tobytes().hex()} ({b[i]!r})"
    )


def compare(got: Outputs, want: Outputs, value_dtype: np.dtype, case: Case) -> list[str]:
    """The differences of values and gradients (none if want has no gradients)."""
    errors = []
    if got[0] != want[0]:
        errors.append(f"values: {_first_difference(got[0], want[0], value_dtype, 0)}")
    grad_dtype = np.dtype(np.uint16) if case.dtype == "bfloat16" else np.dtype(case.dtype)
    if got[1] and got[1] != want[1]:
        errors.append(f"gradients: {_first_difference(got[1], want[1], grad_dtype, case.rows[1])}")
    return errors


def interface_outputs(session: Session, case: Case, interface: str, x: Inputs) -> Outputs:
    if interface in ("rust", "cpp_kernel", "cpp_loss"):
        return _read(session, case, interface)
    if interface == "capi_values":
        return run_capi(case, x, gradients=False)
    runs = {"capi": run_capi, "numpy": run_numpy, "torch": run_torch, "jax": run_jax}
    return runs[interface](case, x)


@pytest.mark.parametrize(("case", "interface"), pairs())
def test_cross_interface(session: Session, case: Case, interface: str) -> None:
    framework = interface in ("torch", "jax")
    if framework and importlib.util.find_spec(interface) is None:
        pytest.skip(f"{interface} is not installed")
    want_from = reference(case)
    for name in {interface, want_from} & {"rust", "cpp_kernel", "cpp_loss"}:
        session.require(name)
    x = make_inputs(case)
    findings = [name for name, f in FINDINGS.items() if f.applies(case, interface)]
    shown = f"{', '.join(findings)} ({FINDINGS[findings[0]].repro})" if findings else ""
    try:
        want = interface_outputs(session, case, want_from, x)
        got = interface_outputs(session, case, interface, x)
    except AssertionError as error:
        if findings:
            pytest.xfail(f"{shown}: {interface} or {want_from} on {case.id}: {error}")
        raise
    value_dtype = np.dtype(np.float64)
    if framework and case.dtype != "float64":
        # EXCEPTIONS["framework_value_dtype"]
        value_dtype = np.dtype(np.float32)
        want = (np.frombuffer(want[0]).astype(np.float32).tobytes(), want[1])
    errors = compare(got, want, value_dtype, case)
    message = f"{interface} vs {want_from} on {case.id}:\n" + "\n".join(errors)
    if errors and findings:
        pytest.xfail(f"{shown}: {message}")
    assert not errors, message


def test_left_out_pairs_are_named() -> None:
    """Every (case, interface) pair left out names an EXCEPTIONS entry, and every case runs
    on at least three interfaces."""
    for case in CASES:
        compared = [i for i in INTERFACES if why_not(case, i) is None]
        assert reference(case) in compared and len(compared) >= 3, (case.id, compared)
        for interface in INTERFACES:
            reason = why_not(case, interface)
            assert reason is None or reason in EXCEPTIONS


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("classes", [3, 17, 40])
def test_multilabel_ce_mask_and_label_sets_agree(dtype: type, classes: int) -> None:
    """The same label sets as a 0/1 mask and as label sets give the same bytes, in the blocked
    pass (up to 32 classes) and the row pass, with tied positives and an empty set (NaN)."""
    rng = np.random.default_rng(classes)
    z = np.round(rng.standard_normal((13, classes)) * 4).astype(dtype)
    z[0] = [0.5, 0.25, -1.75] + [-4.0] * (classes - 3)
    mask = rng.random(z.shape) < 0.3
    mask[0] = np.arange(classes) == 0
    mask[1, :] = False
    loss = libsdca.MultilabelCrossEntropy()
    v_mask, g_mask = libsdca.loss_and_grad(z, mask, loss)
    v_sets, g_sets = libsdca.loss_and_grad(z, [np.flatnonzero(m).tolist() for m in mask], loss)
    assert np.isnan(v_mask[1]) and np.isnan(v_sets[1])
    assert v_mask.tobytes() == v_sets.tobytes(), v_mask - v_sets
    assert g_mask.tobytes() == g_sets.tobytes()


# --- minimal repros of FINDINGS: strict xfail, they pass once fixed --------------------------


@pytest.mark.xfail(
    strict=True, reason=FINDINGS["distillation_values_depend_on_gradients"].description
)
def test_finding_distillation_values_depend_on_gradients() -> None:
    family = BY_NAME["distillation_t1"]
    z, teacher = np.array([[1.0, 2.0]]), np.array([[-1.75, 1.0]])
    with_gradients, _ = sdca_loss_evaluate(family, "float64", z, "teacher", teacher)
    without, _ = sdca_loss_evaluate(family, "float64", z, "teacher", teacher, gradients=False)
    # 0x1.7e4f06d19dcacp-2 and 0x1.7e4f06d19dcadp-2 on macOS arm64
    assert with_gradients.tobytes() == without.tobytes(), (
        with_gradients[0].hex(),
        without[0].hex(),
    )


def test_distillation_float64_shift_overflow() -> None:
    """float64 work of a row whose shift exceeds log(max) gives the float32 work's loss."""
    case = next(c for c in CASES if c.family.name == "distillation_t1" and c.precision == 2)
    x = make_inputs(case)
    z, teacher = x.scores[3:4].astype(np.float32), x.teacher[3:4].astype(np.float32)
    loss = libsdca.Distillation(temperature=1.0)
    want, want_grad = libsdca.loss_and_grad(z, teacher, loss)
    value, grad = libsdca.loss_and_grad(z, teacher, loss, precision="float64")
    # float32 work: 722.6347 and entries within 0.006; float64 work: NaN
    np.testing.assert_allclose(value, want, rtol=1e-6)
    np.testing.assert_allclose(grad, want_grad, rtol=1e-5, atol=1e-7)


def test_jax_label_smoothing_in_float64() -> None:
    """libsdca.jax passes a static label_smoothing to the CPU kernels in float64, as PyTorch
    and the C++ kernels take it: float64 work of float32 logits sees 0.1, not float32(0.1).
    Both calls run on the CPU (JAX's default device on a GPU host is the GPU, whose kernels
    read the smoothing from an array of the logits' type)."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    L = importlib.import_module("libsdca.torch")
    J = importlib.import_module("libsdca.jax")
    z = np.array([[-2.0, -1.0, -1000.0, 1.0, 5.0, 3.0, -5.0]], np.float32)
    options = dict(label_smoothing=0.1, reduction="none", backend="kernel", precision="float64")
    leaf = torch.tensor(z, requires_grad=True)
    value = L.cross_entropy(leaf, torch.tensor([2]), **options)
    value.sum().backward()
    assert leaf.grad is not None
    with jax.default_device(jax.devices("cpu")[0]):
        got, pullback = jax.vjp(lambda s: J.cross_entropy(s, jax.numpy.array([2]), **options), z)
        (grad,) = pullback(jax.numpy.ones_like(got))
    assert np.asarray(got).tobytes() == value.detach().numpy().tobytes(), (got, value)
    assert np.asarray(grad).tobytes() == leaf.grad.numpy().tobytes(), (grad, leaf.grad)


def test_jax_keeps_subnormals() -> None:
    """The CPU kernels of libsdca.jax run with XLA's flush-to-zero and denormals-are-zero
    cleared, as the C ABI's."""
    jax = pytest.importorskip("jax")
    J = importlib.import_module("libsdca.jax")
    if importlib.import_module("libsdca._jax_ops")._handler()[0] is None:
        pytest.xfail("the native route keeps subnormals: the C ABI clears XLA's modes")
    z = np.array([[0.0, -95.0]], np.float32)
    want_value, want_grad = libsdca.loss_and_grad(z, np.array([0]), libsdca.CrossEntropy())
    value, pullback = jax.vjp(
        lambda s: J.cross_entropy(s, jax.numpy.array([0]), reduction="none"), z
    )
    (grad,) = pullback(jax.numpy.ones_like(value))
    # 5.52e-42 and [-5.521e-42, 5.521e-42]
    assert np.asarray(value).tobytes() == want_value.astype(np.float32).tobytes(), value
    assert np.asarray(grad).tobytes() == want_grad.tobytes(), grad
