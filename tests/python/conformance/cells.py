"""The conformance matrix: a cell is a loss/backend/dtype/precision/reduction/input family.

Every value and gradient entry is compared with the independent references. Only the exact
row/entry records in known_failures can be expected failures; fixes and regressions both
fail verification (known.py). The predicates below (LIMITATIONS) name the documented limit
of each recorded entry (manifest.py writes the records from a recorded run); they never
exempt an entry by themselves.
"""

from __future__ import annotations

import functools
import json
import math
import os
import platform
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

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

BACKENDS = (
    Backend("numpy", "numpy", "cpu"),
    Backend("torch-cpu-kernel", "torch", "cpu", "kernel"),
    Backend("torch-cpu-composed", "torch", "cpu", "composed"),
    Backend("torch-cuda-kernel", "torch", "cuda", "kernel"),
    Backend("torch-cuda-composed", "torch", "cuda", "composed"),
    Backend("jax-cpu-kernel", "jax", "cpu", "kernel"),
    Backend("jax-cpu-composed", "jax", "cpu", "composed"),
    Backend("jax-gpu-kernel", "jax", "gpu", "kernel"),
    Backend("jax-gpu-composed", "jax", "gpu", "composed"),
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
    key: str
    applies: Predicate

    @property
    def reason(self) -> str:
        return REASONS[self.key]


class DocumentedLimitation(Exception):
    """The failures of a cell are those of its documented limitations."""


ENTROPY = ("cross_entropy", "multilabel_cross_entropy", "topk_cross_entropy", "distillation")
JAX_CPU = ("jax-cpu-kernel", "jax-cpu-composed")


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


def outputs(cell: Cell, ref: Reference) -> list[tuple[MPReal, str]]:
    """The reference value and gradient entries with their output dtypes."""
    return [(ref.value, cell.setting.value_dtypes[1])] + [(g, cell.dtype) for g in ref.gradient]


def zero_fails(cell: Cell, x: MPReal, dtype: str) -> bool:
    """Whether a returned 0 is out of the bound of reference x."""
    delta, floor = terms(cell.setting, cell.case.classes)
    return ratio(0.0, x, (dtype, dtype), delta, floor) > 1


def kernel(cell: Cell) -> bool:
    return cell.backend.native


def composed(cell: Cell, row: Row, ref: Reference) -> bool:
    return not cell.backend.native and cell.case.family not in ("ordinary", "ties")


def jax_cpu_flush(cell: Cell, row: Row, ref: Reference) -> bool:
    """XLA flushes to zero the subnormal results of its own operations (the composed losses
    and the kernels' reductions; the FFI kernels run in IEEE modes): a row fails where they
    carry an entry its output dtype represents."""
    if cell.backend.name not in JAX_CPU or cell.loss.name not in (*ENTROPY, *PROXIMAL):
        return False
    if cell.backend.native and cell.reduction == "none":
        return False
    entries = [(x, dtype) for x, dtype in outputs(cell, ref) if zero_fails(cell, x, dtype)]
    # float16's subnormals are normal in its float32 work.
    return any(dtype != "float16" and abs(x) < FINFO[dtype][1] for x, dtype in entries)


def extreme(cell: Cell, row: Row, ref: Reference) -> bool:
    """[max, -max, ...] rows whose spread (twice the storage max) overflows the work type."""
    big = FINFO[cell.dtype][2]
    return cell.case.family == "extreme" and 2 * big > FINFO[cell.setting.work][2]


def max_half_row(cell: Cell, row: Row, ref: Reference) -> bool:
    """[max, max/2, -max/2, 0] (float32 max) with label 1, top-k alpha at k = 2."""
    alpha = cell.loss.name == "topk_hinge_alpha" and cell.loss.k == 2
    return (
        cell.case.family == "limits"
        and alpha
        and row[0].value == FINFO["float32"][2]
        and row[1].positive
    )


def big_row(cell: Cell, row: Row, ref: Reference) -> bool:
    """[1e30, -1e30, 0, 1] with the label (positives) second: top-k hinges at k = 2 and the
    smoothed multilabel hinge."""
    loss = cell.loss
    topk = loss.name.startswith("topk_hinge") and loss.k == 2
    hinge = topk or loss.name == "multilabel_hinge" and loss.gamma > 0
    return cell.case.family == "limits" and hinge and 1e29 < row[0].value < 1e31 and row[1].positive


def proximal_tail(cell: Cell, row: Row, ref: Reference) -> bool:
    return (
        kernel(cell)
        and cell.loss.name == "proximal_cross_entropy"
        and cell.case.family == "far_tail"
    )


def distillation_tail(cell: Cell, row: Row, ref: Reference) -> bool:
    """float64 proximal distillation terms more than about 700 (in units of T) below the
    leading logit, of the student or the teacher."""
    return (
        kernel(cell)
        and cell.loss.name == "proximal_distillation"
        and cell.setting.work == "float64"
        and tail(cell, row, 700.0, 760.0)
    )


def subnormal_sum(cell: Cell, row: Row, ref: Reference) -> bool:
    """float64 multilabel rows whose negatives lie beyond exp(-708): each negative's term
    rounds to a subnormal before the sum that the positives' entries are of."""
    return (
        kernel(cell)
        and cell.loss.name == "multilabel_cross_entropy"
        and cell.setting.work == "float64"
        and cell.case.family == "far_tail"
        and tail(cell, row, 708.0, 760.0)
    )


def capped_odds(cell: Cell, row: Row, ref: Reference) -> bool:
    return (
        kernel(cell)
        and cell.loss.name == "topk_cross_entropy"
        and cell.case.family == "capped"
        and cell.setting.work == "float64"
    )


def topk_spread(cell: Cell, row: Row, ref: Reference) -> bool:
    return cell.case.family == "topk_spread"


def all_capped(cell: Cell, row: Row, ref: Reference) -> bool:
    return cell.case.family == "all_capped"


def near_distillation(cell: Cell, row: Row, ref: Reference) -> bool:
    return (
        cell.loss.name in DISTILLATION
        and cell.case.family in ("gaussian", "offset")
        and cell.setting.work == "float64"
    )


def near_mask(cell: Cell, row: Row, ref: Reference) -> bool:
    return (
        cell.loss.name == "multilabel_cross_entropy"
        and cell.case.family == "gaussian"
        and cell.dtype == "float64"
    )


REASONS = {
    "jax-cpu": "JAX CPU: XLA's own operations flush subnormal results to zero (docs/API.md, "
    "Known limits: JAX on the CPU)",
    "extreme": "Rows at +-finfo.max whose spread overflows the work type: NaN, inf or a wrong "
    "value (docs/API.md, Known limits: Extreme spreads)",
    "max-half": "top-k alpha at [max, max/2, -max/2, 0] returns 0 in every precision (margins "
    "1 +- max/2 cancel; exact loss 1, smooth 0.875 at gamma=0.5)",
    "1e30": "[1e30, -1e30, 0, 1] cannot separate 1e30 + 1 from 1e30 + 2 (top-k hinges at k=2 "
    "and the smoothed multilabel hinge select the wrong competitor)",
    "proximal-tail": "proximal_cross_entropy rows whose competitors lie beyond the work type's "
    "exponential range lose the relative accuracy of their tiny loss",
    "distillation-tail": "float64 proximal_distillation terms more than ~708 (in units of T) "
    "below the leading logit are rounded to subnormals or zero by the proximal step",
    "subnormal-sum": "float64 multilabel_cross_entropy rows whose negatives lie beyond "
    "exp(-708): each term rounds to a subnormal before the sum, so the positives' entries err "
    "by up to C / 2 units of the least subnormal (docs/API.md, Known limits: Subnormal results)",
    "capped-odds": "Top-k entropy rounds its log-odds to float64 before exp: losses and "
    "gradients near exp(-700) are off by |odds| eps / 2 (up to 350 eps)",
    "spread": "Top-k cross-entropy at extreme score spreads ([0, S, -S, -S], k=2) loses "
    "digits with the spread (docs/API.md, Known limits: Extreme spreads)",
    "all-capped": "All-capped top-k cross-entropy rows spanning > 2^100 lose their smallest "
    "term ([M/4, H, M, 1, -H] returns log 5)",
    "near-distillation": "float64 distillation gradient entries where student and teacher "
    "probabilities nearly agree: within 5e-17 absolute, above the componentwise relative bound "
    "(docs/API.md, Known limits: Cancellation)",
    "near-mask": "float64 multilabel_cross_entropy gradients p - 1/|Y| that nearly cancel "
    "(4097-class Gaussian rows): within 5e-17 absolute (docs/API.md, Known limits: "
    "Cancellation)",
    "composed": "backend='composed' computes with the framework's tensor operations, as "
    "PyTorch and JAX do: float32 multilabel gradients that nearly cancel on wide rows, results "
    "below the normal range (XLA flushes them), the framework's rounding of wide sums, and "
    "hinge spreads at +-finfo.max that overflow",
}

# One line per limitation: a fix deletes (or narrows) its line, a new route's limit adds one.
LIMITATIONS = (
    Limitation("jax-cpu", jax_cpu_flush),
    Limitation("extreme", extreme),
    Limitation("max-half", max_half_row),
    Limitation("1e30", big_row),
    Limitation("proximal-tail", proximal_tail),
    Limitation("distillation-tail", distillation_tail),
    Limitation("subnormal-sum", subnormal_sum),
    Limitation("capped-odds", capped_odds),
    Limitation("spread", topk_spread),
    Limitation("all-capped", all_capped),
    Limitation("near-distillation", near_distillation),
    Limitation("near-mask", near_mask),
    Limitation("composed", composed),
)


def precisions(dtype: str) -> tuple[str | None, ...]:
    return (None, "float64") if dtype == "float64" else (None, "float32", "float64")


def cells(framework: str) -> Iterator[Cell]:
    for loss in LOSSES:
        for backend in BACKENDS:
            if backend.framework != framework or not backend.runs(loss):
                continue
            for dtype in DTYPES[framework]:
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
    out = []
    for key, cell in unique.items():
        marks: list[Any] = [pytest.mark.slow] if slow(cell) else []
        out.append(pytest.param(cell, id=key, marks=marks))
    return out


# The CI profile of the matrix (pytest --conformance ci): projections that keep every loss,
# backend, dtype, precision, reduction and input family without their product.
DEFAULT_BACKENDS = (
    "numpy",
    "torch-cpu-kernel",
    "torch-cuda-kernel",
    "jax-cpu-kernel",
    "jax-gpu-kernel",
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
    """(backend, loss) pairs of the framework: a NaN logit makes its row NaN on every backend
    (docs/API.md, Invalid rows). linear_cross_entropy has none: its rows share the weights,
    so one NaN weight reaches every row."""
    return [
        pytest.param(backend, loss, id=f"{loss.label}-{backend.name}")
        for loss in LOSSES
        for backend in BACKENDS
        if backend.framework == framework
        and backend.runs(loss)
        and loss.name != "linear_cross_entropy"
    ]


def check_nonfinite(backend: Backend, loss: Loss) -> None:
    """A NaN logit in the first row of the loss's first float32 case: that row's value is NaN,
    the others keep theirs; NumPy warns once (InvalidRowsWarning)."""
    reason = backends.unavailable(backend)
    if reason:
        pytest.skip(reason)
    resolved, case = next(cases(loss, "float32"))
    first = list(case.rows[0])
    first[1] = replace(first[1], value=math.nan)
    case = replace(case, rows=(tuple(first), *case.rows[1:]))
    if backend.framework == "numpy":
        import libsdca

        with pytest.warns(libsdca.InvalidRowsWarning):
            values, gradients, _ = backends.run(backend, resolved, "float32", None, "none", case)
    else:
        values, gradients, _ = backends.run(backend, resolved, "float32", None, "none", case)
    assert math.isnan(values[0]), values
    assert np.isfinite(values[1:]).all(), values
    assert np.isfinite(gradients[1:]).all()


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
    """The failures of a cell's call (finite inputs raise nothing) and its route."""
    values, gradients, route = backends.run(
        cell.backend, cell.loss, cell.dtype, cell.precision, cell.reduction, cell.case
    )
    failures, _ = check(cell.setting, cell.case, values, gradients)
    return failures, route


def check_cell(cell: Cell, request: pytest.FixtureRequest) -> None:
    reason = backends.unavailable(cell.backend)
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
                            str(i): [e.key for e in entries] for i, entries in covered.items()
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
