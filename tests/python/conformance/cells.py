"""The conformance matrix: a cell is a loss/backend/dtype/precision/reduction/input family.

Every value and gradient entry is compared with the independent references. Only the exact
row/entry records in known_failures can be expected failures; fixes and regressions both
fail verification (known.py). The older broad predicates below annotate diagnostic records
for review; they never exempt an entry. Their reasons state the limits of docs/API.md,
Known limits, with the recorded magnitudes.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import math
import os
import platform
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from mpmath_types import MPReal

from conformance import backends
from conformance.backends import Backend
from conformance.contract import Failure, Setting, check, describe, ratio, terms
from conformance.inputs import FINFO, Case, cases
from conformance.known import verify
from conformance.references import (
    DISTILLATION,
    HINGES,
    PROXIMAL,
    Loss,
    Reference,
    Row,
    reference,
)

LOSSES = (
    Loss("cross_entropy"),
    Loss("multilabel_cross_entropy"),
    Loss("topk_cross_entropy", k=1),
    Loss("topk_cross_entropy", k=2),
    Loss("topk_cross_entropy", k=5),
    Loss("topk_cross_entropy", k=100),
    Loss("topk_cross_entropy", k=-1),  # k = C - 1
    Loss("multiclass_hinge"),
    Loss("multiclass_hinge", gamma=0.5),
    Loss("topk_hinge_alpha", k=2),
    Loss("topk_hinge_alpha", k=2, gamma=0.5),
    Loss("topk_hinge_alpha", k=100),
    Loss("topk_hinge_alpha", k=-1),
    Loss("topk_hinge_alpha", k=-1, gamma=0.5),
    Loss("topk_hinge_beta", k=2),
    Loss("topk_hinge_beta", k=2, gamma=0.5),
    Loss("topk_hinge_beta", k=100),
    Loss("topk_hinge_beta", k=-1),
    Loss("topk_hinge_beta", k=-1, gamma=0.5),
    Loss("multilabel_hinge"),
    Loss("multilabel_hinge", gamma=0.5),
    Loss("proximal_cross_entropy", alpha=0.1),
    Loss("proximal_cross_entropy", alpha=10.0),
    Loss("distillation", temperature=0.5),
    Loss("distillation"),
    Loss("distillation", temperature=4.0),
    Loss("proximal_distillation", temperature=2.0, alpha=0.01),
    Loss("proximal_distillation", alpha=1.0),
    Loss("linear_cross_entropy"),
    Loss("linear_cross_entropy", recompute=True),
)

_EXT_ON, _EXT_OFF = (("SDCA_CPU_EXTENSION", "on"),), (("SDCA_CPU_EXTENSION", "off"),)
_WIDE = ("float32", "float64")
# Distillation has no backend option: explicitly disable the fused route in tensor cells.
_JAX_CUDA_OFF = (("SDCA_JAX_CUDA", "off"),)
BACKENDS = (
    Backend("numpy", "numpy", "cpu"),
    Backend("torch-cpu-extension", "torch", "cpu", "auto", _EXT_ON),
    Backend("torch-cpu-auto", "torch", "cpu", "auto", _EXT_OFF),
    Backend("torch-cpu-native", "torch", "cpu", "native", _EXT_OFF),
    Backend("torch-cpu-tensor", "torch", "cpu", "torch", _EXT_OFF),
    Backend("torch-cuda-launcher", "torch", "cuda", "auto", (("SDCA_CUDA_LAUNCHER", "on"),)),
    Backend("torch-cuda-triton", "torch", "cuda", "auto", (("SDCA_CUDA_LAUNCHER", "off"),)),
    Backend("torch-cuda-tensor", "torch", "cuda", "torch"),
    Backend("jax-cpu-jax", "jax", "cpu", "jax", (("SDCA_JAX_FFI", "off"),)),
    Backend("jax-cpu-auto", "jax", "cpu", "auto", (("SDCA_JAX_FFI", "on"),)),
    Backend("jax-cpu-ffi", "jax", "cpu", "native", (("SDCA_JAX_FFI", "on"),)),
    # The host callback computes what the FFI handlers do; 16-bit storage casts alike.
    Backend("jax-cpu-callback", "jax", "cpu", "native", (("SDCA_JAX_FFI", "off"),), _WIDE),
    Backend("jax-gpu-jax", "jax", "gpu", "jax", _JAX_CUDA_OFF),
    Backend("jax-gpu-triton", "jax", "gpu", "auto", (("SDCA_JAX_CUDA", "on"),)),
    Backend("jax-gpu-callback", "jax", "gpu", "native", _JAX_CUDA_OFF),
)
DTYPES = {
    "numpy": ("float32", "float64"),
    "torch": ("float16", "bfloat16", "float32", "float64"),
    "jax": ("float16", "bfloat16", "float32", "float64"),
}
# The 10-class random rows also run the reductions.
REDUCED = ("ordinary", 10)


@dataclass(frozen=True)
class Cell:
    loss: Loss
    backend: Backend
    dtype: str
    precision: str | None
    reduction: str
    case: Case

    @property
    def id(self) -> str:
        precision = self.precision or "default"
        family = f"{self.case.family}{self.case.classes}"
        return f"{self.loss.label}-{self.backend.name}-{self.dtype}-{precision}-{family}-{self.reduction}"

    @property
    def setting(self) -> Setting:
        return Setting(
            self.loss,
            self.dtype,
            self.precision,
            self.reduction,
            self.backend.framework,
            self.backend.native,
            self.backend.cuda,
            self.case.classes,
        )


Predicate = Callable[[Cell, Row, Reference], bool]


@dataclass(frozen=True)
class Limitation:
    reason: str
    applies: Predicate


class DocumentedLimitation(Exception):
    """The failures of a cell are those of its documented limitations."""


ENTROPY = ("cross_entropy", "multilabel_cross_entropy", "topk_cross_entropy", "distillation")
JAX_CPU = ("jax-cpu-jax", "jax-cpu-auto", "jax-cpu-ffi", "jax-cpu-callback")
CUDA_KERNELS = ("torch-cuda-launcher", "torch-cuda-triton")
NATIVE_CPU = ("numpy", "torch-cpu-auto", "torch-cpu-native")


def gaps(cell: Cell, row: Row) -> list[float]:
    """The distances of the logits (and teacher logits) below their row's maximum, over T."""
    inputs = [[b.value for b in row]]
    if cell.loss.name in DISTILLATION:
        inputs.append([b.teacher for b in row])
    with np.errstate(over="ignore"):
        return [(max(v) - x) / cell.loss.temperature for v in inputs for x in v]


def tail(cell: Cell, row: Row, low: float, high: float) -> bool:
    return any(low < g <= high for g in gaps(cell, row))


def float32_work(cell: Cell) -> bool:
    return cell.setting.work == "float32"


def float64_values(cell: Cell) -> bool:
    """Loss values returned in float64 (whose eps the bound takes)."""
    return cell.setting.value_dtypes[0] == "float64"


def outputs(cell: Cell, ref: Reference) -> list[tuple[MPReal, str]]:
    """The reference value and gradient entries with their output dtypes."""
    return [(ref.value, cell.setting.value_dtypes[1])] + [(g, cell.dtype) for g in ref.gradient]


def zero_fails(cell: Cell, x: MPReal, dtype: str) -> bool:
    """Whether a returned 0 is out of the bound of reference x."""
    delta, floor = terms(cell.setting, cell.case.classes)
    return ratio(0.0, x, (dtype, dtype), delta, floor) > 1


def jax_cpu_flush(cell: Cell, row: Row, ref: Reference) -> bool:
    """XLA flushes to zero the subnormal results of every backend and, in jax.numpy, the
    exponentials below the normal range of the work (87.3 in float32, 708.4 in float64; 16x
    margins for the intermediates): a row fails where they carry an entry its output dtype
    represents (the native library computes with subnormals)."""
    if cell.backend.name not in JAX_CPU or cell.loss.name not in (*ENTROPY, *PROXIMAL):
        return False
    if float32_work(cell):
        low, high, small = 84.5, 120.0, 2.0**-100
    else:
        low, high, small = 705.6, 760.0, 2.0**-1000
    entries = [(x, dtype) for x, dtype in outputs(cell, ref) if zero_fails(cell, x, dtype)]
    tensor_distillation = (
        cell.backend.name == "jax-cpu-auto" and cell.loss.name == "proximal_distillation"
    )
    flushed = (cell.backend.option == "jax" or tensor_distillation) and tail(cell, row, low, high)
    if flushed and any(abs(x) < small for x, _ in entries):
        return True
    # float16's subnormals are normal in its float32 work.
    return any(dtype != "float16" and abs(x) < FINFO[dtype][1] for x, dtype in entries)


def cuda_distillation(cell: Cell, row: Row, ref: Reference) -> bool:
    return (
        cell.backend.name in CUDA_KERNELS
        and cell.loss.name == "distillation"
        and float32_work(cell)
        and cell.case.classes >= 100
        and tail(cell, row, 87.0, 120.0)
    )


def jax_gpu(cell: Cell, row: Row, ref: Reference) -> bool:
    return (
        cell.backend.name == "jax-gpu-jax"
        and cell.loss.name in ENTROPY
        and float32_work(cell)
        and tail(cell, row, 87.0, 120.0)
    )


def topk_spread(cell: Cell, row: Row, ref: Reference) -> bool:
    return cell.case.family == "topk_spread"


def all_capped(cell: Cell, row: Row, ref: Reference) -> bool:
    return cell.case.family == "all_capped"


def max_half_row(cell: Cell, row: Row, ref: Reference) -> bool:
    """Precision-policy limit [max, max/2, -max/2, 0] (float32 max) with label 1, top-k alpha at k = 2."""
    alpha = cell.loss.name == "topk_hinge_alpha" and cell.loss.k == 2
    return (
        cell.case.family == "limits"
        and alpha
        and row[0].value == FINFO["float32"][2]
        and row[1].positive
    )


def big_row(cell: Cell, row: Row, ref: Reference) -> bool:
    """Precision-policy limit [1e30, -1e30, 0, 1] with label (positives) second: top-k hinges at
    k = 2 and the smoothed multilabel hinge."""
    loss = cell.loss
    topk = loss.name.startswith("topk_hinge") and loss.k == 2
    hinge = topk or loss.name == "multilabel_hinge" and loss.gamma > 0
    return cell.case.family == "limits" and hinge and 1e29 < row[0].value < 1e31 and row[1].positive


def tensor_mask_entropy(cell: Cell, row: Row, ref: Reference) -> bool:
    return (
        cell.case.family == "gaussian"
        and cell.loss.name == "multilabel_cross_entropy"
        and cell.dtype == "float32"
        and float32_work(cell)
        and cell.backend.name
        in (
            "torch-cpu-tensor",
            "jax-cpu-jax",
            "jax-cpu-auto",
            "jax-gpu-jax",
            "torch-cuda-tensor",
            "torch-cuda-launcher",
            "torch-cuda-triton",
            "jax-gpu-triton",
        )
    )


def extreme64(cell: Cell, row: Row, ref: Reference) -> bool:
    if cell.dtype != "float64" or cell.case.family != "extreme":
        return False
    loss = cell.loss
    if loss.name in (*HINGES, "proximal_cross_entropy"):
        return True
    native = cell.backend.framework != "jax" or cell.backend.name == "jax-cpu-auto"
    return native and loss.name in DISTILLATION and loss.temperature <= 1 and loss.alpha >= 1


def gpu_float64_tail(cell: Cell, row: Row, ref: Reference) -> bool:
    """Individual double exponentials become subnormal before accumulation on CUDA."""
    return (
        cell.backend.cuda
        and cell.dtype == "float64"
        and cell.loss.name in (*ENTROPY, *PROXIMAL)
        and cell.case.family == "far_tail"
        and tail(cell, row, 700.0, 760.0)
    )


def native_subnormal_distillation(cell: Cell, row: Row, ref: Reference) -> bool:
    return (
        cell.loss.name in DISTILLATION
        and cell.dtype == "float64"
        and cell.backend.name in (*NATIVE_CPU, "torch-cpu-extension", "jax-cpu-auto")
        and tail(cell, row, 739.0, 760.0)
    )


def all_capped_odds(cell: Cell, row: Row, ref: Reference) -> bool:
    return (
        cell.loss.name == "topk_cross_entropy"
        and cell.loss.k == cell.case.classes - 1
        and cell.case.family == "capped"
        and cell.case.classes == 4096
        and float64_values(cell)
    )


def gaussian64(name: str) -> Predicate:
    def applies(cell: Cell, row: Row, ref: Reference) -> bool:
        return cell.loss.name == name and cell.case.family == "gaussian" and cell.dtype == "float64"

    return applies


def jax_offset_distillation(cell: Cell, row: Row, ref: Reference) -> bool:
    return (
        cell.backend.name == "jax-cpu-jax"
        and cell.loss == Loss("distillation")
        and cell.dtype == "float64"
        and cell.case.family == "offset"
    )


REASONS = {
    "spread": "Top-k cross-entropy at extreme score spreads: partially capped "
    "rows ([0, S, -S, -S], k=2) lose digits with the spread and return log 3 at 1e16, 0 at "
    "finfo.max",
    "all-capped": "Top-k cross-entropy at extreme score spreads: all-capped "
    "rows spanning > 2^100 lose their smallest term ([M/4, H, M, 1, -H] returns log 5)",
    "jax-cpu": "JAX CPU, default float32 work: XLA on the CPU flushes every "
    "subnormal exponential to zero (and JAX.md: subnormal inputs and outputs, across all "
    "backends; exp(-708.4) in float64)",
    "cuda-distillation": "CUDA distillation, default float32 work: competitor "
    "terms more than about 87 below the leading logit round to subnormals or zero before they "
    "are summed (from 100 classes on)",
    "jax-gpu": "JAX GPU, default float32 work: individually underflowed "
    "float32 exponentials vanish before summation",
    "max-half": "top-k alpha at [max, max/2, -max/2, 0] returns 0 in every precision (margins 1 +- max/2 "
    "cancel; exact loss 1, smooth 0.875 at gamma=0.5)",
    "1e30": "[1e30, -1e30, 0, 1] cannot separate 1e30 + 1 from 1e30 + 2 (top-k hinges at k=2 and the "
    "smoothed multilabel hinge select the wrong competitor)",
    "mask-entropy": "Float32 mask entropy "
    "gradients near probability cancellation exceed the componentwise bound on wide "
    "Gaussian rows, in tensor and CUDA implementations; explicit float64 work reduces "
    "the cancellation error",
    "gpu-float64-tail": "CUDA float64 "
    "entropy and distillation round or underflow individual exponentials before summing "
    "their aggregate tail (score gaps 700-760)",
    "extreme64": "Float64 rows at +-finfo.max: hinges, proximal_cross_entropy and "
    "(where the loss overflows, T <= 1) distillation return NaN, inf for a finite loss, a "
    "wrong finite value, or raise ValueError ('... scores must be finite') for finite inputs",
    "subnormal-distillation": "Float64 distillation on the native library and the CPU "
    "extension sums competitor terms rounded to float64 subnormals (tails 740-750 below the "
    "maximum)",
    "all-capped-odds": "Top-k entropy with k = C - 1 rounds its log-odds to "
    "float64 before exp: at log-odds near -350 float64 losses and gradients are off by "
    "|odds| eps / 2 (128 eps)",
    "near-distillation": "Float64 distillation gradient entries where student and teacher "
    "probabilities nearly agree (4097-class Gaussian rows): within 5e-17 absolute, relative "
    "error up to 4e-11 (2500x the bound on the CPU routes, 23x on CUDA)",
    "near-mask": "Float64 multilabel_cross_entropy gradients p - 1/|Y| that nearly cancel "
    "(4097-class Gaussian rows): within 5e-17 absolute, relative error up to 4e-12 (74x the "
    "bound)",
    "jax-offset": "JAX distillation of float64 students at offset 2/eps (teacher "
    "unshifted): a gradient entry 70 eps off (bound 64)",
}

# One line per limitation: a fix deletes (or narrows) its line, a new route's limit adds one.
LIMITATIONS = (
    Limitation(REASONS["spread"], topk_spread),
    Limitation(REASONS["all-capped"], all_capped),
    Limitation(REASONS["jax-cpu"], jax_cpu_flush),
    Limitation(REASONS["cuda-distillation"], cuda_distillation),
    Limitation(REASONS["jax-gpu"], jax_gpu),
    Limitation(REASONS["max-half"], max_half_row),
    Limitation(REASONS["1e30"], big_row),
    Limitation(REASONS["mask-entropy"], tensor_mask_entropy),
    Limitation(REASONS["gpu-float64-tail"], gpu_float64_tail),
    Limitation(REASONS["extreme64"], extreme64),
    Limitation(REASONS["subnormal-distillation"], native_subnormal_distillation),
    Limitation(REASONS["all-capped-odds"], all_capped_odds),
    Limitation(REASONS["near-distillation"], gaussian64("distillation")),
    Limitation(REASONS["near-mask"], gaussian64("multilabel_cross_entropy")),
    Limitation(REASONS["jax-offset"], jax_offset_distillation),
)


def precisions(dtype: str) -> tuple[str | None, ...]:
    return (None, "float64") if dtype == "float64" else (None, "float32", "float64")


def cells(framework: str) -> Iterator[Cell]:
    for loss in LOSSES:
        for backend in BACKENDS:
            if backend.framework != framework or not backend.runs(loss):
                continue
            for dtype in backend.dtypes or DTYPES[framework]:
                for resolved, case in cases(loss, dtype):
                    for precision in precisions(dtype):
                        reductions = ("none",)
                        if framework != "numpy" and (case.family, case.classes) == REDUCED:
                            reductions += ("sum", "mean")
                        for reduction in reductions:
                            yield Cell(resolved, backend, dtype, precision, reduction, case)


def params(framework: str) -> list[object]:
    """One test per cell (k = C - 1 can resolve to a fixed k's cell: once)."""
    unique = {cell.id: cell for cell in cells(framework)}
    return [
        pytest.param(cell, id=key, marks=[pytest.mark.slow] if slow(cell) else [])
        for key, cell in unique.items()
    ]


# The CI profile of the matrix (pytest --conformance ci): projections that keep every loss,
# backend, dtype, precision, reduction and input family without their product.
DEFAULT_BACKENDS = (
    "numpy",
    "torch-cpu-auto",
    "torch-cuda-launcher",
    "jax-cpu-auto",
    "jax-gpu-triton",
)
STANDARD_LOSSES = ("cross_entropy", "multiclass_hinge[gamma=0]")


@functools.cache
def ci_profile(framework: str) -> frozenset[str]:
    """The ids of the framework's cells in the CI profile: every loss on every input family
    (default backend, float32), the standard losses on every backend and dtype and in every
    precision, every loss in every dtype on the 10-class rows, and the cross-entropy
    reductions. A (loss, family) pair without float32 rows runs in its first dtype."""
    selected: dict[str, Cell] = {}
    every = list({cell.id: cell for cell in cells(framework)}.values())
    for cell in every:
        default = cell.backend.name in DEFAULT_BACKENDS
        plain = cell.precision is None and cell.reduction == "none"
        standard = cell.loss.label in STANDARD_LOSSES
        family = (cell.case.family, cell.case.classes)
        if (
            (default and plain and cell.dtype == "float32")
            or (standard and plain)
            or (standard and default and cell.reduction == "none")
            or (default and plain and family == REDUCED)
            or (default and cell.precision is None and cell.loss.name == "cross_entropy")
        ):
            selected[cell.id] = cell
    covered = {(c.loss.label, c.case.family, c.case.classes) for c in selected.values()}
    for cell in every:
        key = (cell.loss.label, cell.case.family, cell.case.classes)
        plain = cell.precision is None and cell.reduction == "none"
        if cell.backend.name in DEFAULT_BACKENDS and plain and key not in covered:
            selected[cell.id] = cell
            covered.add(key)
    return frozenset(selected)


def slow(cell: Cell) -> bool:
    """The widest rows: started first under -n."""
    return cell.case.classes >= 32768 or cell.case.family == "gaussian"


def nonfinite_params(framework: str) -> list[object]:
    """(backend, loss) pairs whose nonfinite-logit behaviour a document states: ValueError on
    the CPU (API.md, Errors; TORCH.md, Argument parity) and from concrete JAX calls, NaN from
    CUDA cross-entropies (TORCH.md) and from every traced JAX loss (JAX.md, Errors)."""
    cuda_nan = ("cross_entropy", "linear_cross_entropy")
    pairs = []
    for loss in LOSSES:
        for backend in BACKENDS:
            if backend.framework != framework or not backend.runs(loss):
                continue
            if backend.device == "cuda" and loss.name not in cuda_nan:
                continue
            pairs.append(pytest.param(backend, loss, id=f"{loss.label}-{backend.name}"))
    return pairs


def check_nonfinite(backend: Backend, loss: Loss, traced: bool = True) -> None:
    """A NaN logit in the first row of the loss's first float32 case, as documented."""
    reason = backends.unavailable(backend)
    if reason:
        pytest.skip(reason)
    resolved, case = next(cases(loss, "float32"))
    first = list(case.rows[0])
    first[1] = replace(first[1], value=math.nan)
    case = replace(case, rows=(tuple(first), *case.rows[1:]))
    if backend.framework == "jax" and not traced:
        with pytest.raises(ValueError):
            backends.concrete_value(backend, resolved, case)
        return
    expected: type[Exception] | None = ValueError
    if (backend.cuda and not backend.native) or backend.framework == "jax":
        expected = None
    if expected is not None:
        with pytest.raises(expected):
            backends.run(backend, resolved, "float32", None, "none", case)
        return
    values, _, _ = backends.run(backend, resolved, "float32", None, "none", case)
    assert math.isnan(values[0]), values


def _covering(cell: Cell, entries: tuple[Limitation, ...]) -> dict[int, list[Limitation]]:
    """The entries that apply to each row of a cell."""
    found: dict[int, list[Limitation]] = {}
    for i, row in enumerate(cell.case.rows):
        ref = reference(cell.loss, row)
        for entry in entries:
            if entry.applies(cell, row, ref):
                found.setdefault(i, []).append(entry)
    return found


def _evaluate(cell: Cell) -> tuple[list[Failure], str]:
    """Only the named finite-input overflow may be an expected exception; isolate its
    rows so an exception cannot hide correct or newly broken rows in the same batch."""
    try:
        values, gradients, route = backends.run(
            cell.backend, cell.loss, cell.dtype, cell.precision, cell.reduction, cell.case
        )
        failures, _ = check(cell.setting, cell.case, values, gradients)
    except (ValueError, RuntimeError) as error:
        # Only the documented overflow in finite float64 extreme rows may become
        # a numerical limitation. Compiler failures and arbitrary backend errors fail.
        if not (
            cell.dtype == "float64"
            and cell.case.family == "extreme"
            and any(
                message in str(error)
                for message in (
                    "scores must be finite",
                    "Logits must be finite floating-point tensors",
                    "proximal_cross_entropy needs finite scores, labels and a non-target class",
                )
            )
            and all(extreme64(cell, row, reference(cell.loss, row)) for row in cell.case.rows)
        ):
            raise
        if len(cell.case.rows) > 1:
            failures = []
            for i, row in enumerate(cell.case.rows):
                part = replace(cell.case, rows=(row,), tags=(cell.case.tags[i],))
                found, _ = _evaluate(replace(cell, case=part))
                failures.extend(replace(f, row=i) for f in found)
        else:
            failures = [
                Failure(
                    0,
                    "call:finite-overflow",
                    math.nan,
                    "scores must be finite after float64 margin overflow",
                    math.inf,
                )
            ]
        route = cell.backend.name
    return failures, route


def check_cell(cell: Cell, request: pytest.FixtureRequest) -> None:
    reason = backends.unavailable(cell.backend)
    if cell.loss.name in HINGES and importlib.util.find_spec("torch") is None:
        reason = "the hinge references (test_topk_kernels, test_multilabel_kernels) need torch"
    if reason:
        pytest.skip(reason)
    covered = _covering(cell, LIMITATIONS)
    failures, route = _evaluate(cell)
    record = os.environ.get("SDCA_CONFORMANCE_RECORD")
    if record and (covered or failures):
        directory = Path(record)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"worker-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "cell": cell.id,
                        "platform": f"{platform.system()}-{platform.machine()}",
                        "route": route,
                        "failures": [asdict(f) for f in failures],
                        "covered": {
                            str(i): [e.reason for e in entries] for i, entries in covered.items()
                        },
                    }
                )
                + "\n"
            )
    reasons = verify(cell.id, failures)
    if failures:
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                raises=DocumentedLimitation,
                reason="known limit: " + ", ".join(reasons),
            )
        )
        raise DocumentedLimitation(f"route {route}:\n{describe(failures, cell.case.tags)}")
