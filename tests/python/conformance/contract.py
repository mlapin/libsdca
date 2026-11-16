"""The documented error bound of every returned value and gradient entry.

A value x^ with reference x passes when, in the eps and smallest normal x_min of its output
dtype (promote(logits, float32) for loss values, the logits' dtype for gradients; the native
NumPy losses return float64 values, computed in their working precision),

    |x^ - x| <= (64 eps + delta) |x| + eps x_min + allowance,

the error bound of the report (check_number of release_numerics.py),
and a reference that rounds beyond the dtype's max needs an infinity of its sign. delta and
the allowances are the documented terms of their regimes:

- delta = C eps_w (working eps) for top-k entropy with float32 work, and for distillation and
  proximal distillation on CUDA, plus C eps_w x_min / 2 for the native float32 top-k rows
  (report);
- distillation with float32 work: 64 float32 eps of (p + q) / T (docs/API.md, Known limits);
- proximal_cross_entropy: 64 working eps of the row's gradient 1-norm (a 36-digit
  proximal oracle);
- hinges: their margins 1 + z_j - z_y are formed in working precision (report),
  so on random and offset rows values add 64 working eps of 1 + the row's spread, and
  smoothed hinges' gradients 64 working eps of sqrt(C) (1 + spread) / gamma (the
  1/gamma-Lipschitz projection of test_tensor_losses and common_inputs); where margins lose
  the row's information (finfo.max, 1e30 rows) the limit is a table entry, not a bound;
- the random rows of the ordinary sweeps (families ordinary and ties;
  16-bit & ordinary sweeps, whose tolerances are those of common_inputs.py): 64 working eps
  of |loss| + 1 + max|z| (scores over T for distillation), and of the row's largest gradient
  entry, times sqrt(C / 1000) from 1000 classes; smoothed hinges add 64 working eps of
  sqrt(C) (1 + max|z|) / gamma. 16-bit logits with float32 work add 100 float32 eps of
  max(|loss|, 1) and one 16-bit unit plus 2000 float32 eps of the row's largest gradient
  entry (docs/API.md, Known limits).

The other families (offsets, finfo.max, far tails, capped and documented rows) are those of
the release corpus, where the bound above holds entry by entry. A nonsmooth hinge's gradient is
compared only away from kinks (several subgradients).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import mpmath as mp
import numpy as np
from mpmath_types import MPReal

from conformance.inputs import FINFO, Case
from conformance.references import (
    DISTILLATION,
    HINGES,
    Loss,
    Reference,
    Row,
    reduced,
    reference,
)

BUDGET = 64
EPS32 = FINFO["float32"][0]
SWEEPS = ("ordinary", "ties")


@dataclass(frozen=True)
class Setting:
    """What decides the bound of a call: where it runs and in which arithmetic."""

    loss: Loss
    dtype: str
    precision: str | None
    reduction: str
    framework: str
    native: bool  # the native library computes the loss (numpy, backend="native")
    cuda: bool
    classes: int = 0

    @property
    def work(self) -> str:
        """NumPy defaults follow storage. Tensor native fallbacks still select double
        unless float32 work is requested; other tensor defaults permit float32."""
        if self.dtype == "float64" or self.precision == "float64":
            return "float64"
        return (
            "float64"
            if self.native
            and self.framework != "numpy"
            and not (
                self.framework == "jax"
                and (
                    self.loss.name == "proximal_cross_entropy"
                    or (self.loss.name == "topk_cross_entropy" and 0 < self.classes <= 64)
                )
            )
            and self.precision is None
            else "float32"
        )

    @property
    def value_dtypes(self) -> tuple[str, str]:
        """The dtypes of a loss value's eps and of its range."""
        if self.framework == "numpy":
            return self.work, "float64"
        promoted = "float64" if self.dtype == "float64" else "float32"
        return promoted, promoted


@dataclass(frozen=True)
class Failure:
    row: int | None  # None: a reduced value
    entry: str
    actual: float
    expected: str
    ratio: float

    def __str__(self) -> str:
        where = "reduced" if self.row is None else f"row {self.row}"
        return (
            f"{where} {self.entry}: {self.actual!r} vs {self.expected} ({self.ratio:.3g} x bound)"
        )


def overflows(expected: MPReal, dtype: str) -> bool:
    """Whether expected rounds to an infinity in dtype: beyond max by half an ulp."""
    eps, _, big = FINFO[dtype]
    return abs(expected) >= mp.mpf(big) + eps * 2 ** (math.frexp(big)[1] - 2)


def ratio(
    actual: float, expected: MPReal, dtypes: tuple[str, str], delta: float, allowance: MPReal
) -> float:
    """Error over its bound (<= 1 passes); inf for a wrong infinity or a NaN."""
    eps, tiny, _ = FINFO[dtypes[0]]
    if overflows(expected, dtypes[1]):
        return 0.0 if math.isinf(actual) and (actual > 0) == (expected > 0) else math.inf
    if not math.isfinite(actual):
        return math.inf
    error = abs(mp.mpf(actual) - expected)
    bound = (BUDGET * eps + delta) * abs(expected) + eps * tiny + allowance
    return float(error / bound)


def terms(setting: Setting, classes: int) -> tuple[float, float]:
    """The relative delta and the absolute floor of the report's bound."""
    eps_w = FINFO[setting.work][0]
    name = setting.loss.name
    if name == "topk_cross_entropy" and setting.loss.k > 1 and setting.work == "float32":
        floor = classes * eps_w * FINFO["float32"][1] / 2 if setting.native else 0.0
        return classes * eps_w, floor
    if name in DISTILLATION and setting.cuda:
        return classes * eps_w, 0.0
    return 0.0, 0.0


def _scale(loss: Loss, row: Row) -> float:
    """1 + the row's largest score (over T, with the teacher's, for distillation)."""
    if loss.name in DISTILLATION:
        return 1 + max(max(abs(b.value), abs(b.teacher)) for b in row) / loss.temperature
    return 1 + max(abs(b.value) for b in row)


def _row_allowances(
    setting: Setting, family: str, row: Row, ref: Reference, factor: MPReal
) -> tuple[MPReal, list[MPReal]]:
    """The documented allowances of a row's value and of each block's gradient entries."""
    loss, work = setting.loss, setting.work
    eps_w = FINFO[work][0]
    value, gradient = mp.mpf(0), [mp.mpf(0)] * len(ref.gradient)
    if loss.name == "distillation" and work == "float32":
        gradient = [BUDGET * EPS32 * s * factor for s in ref.scale]
    if loss.name == "proximal_cross_entropy":
        norm = mp.fsum(b.count * abs(g) for b, g in zip(row, ref.gradient, strict=True))
        gradient = [BUDGET * eps_w * norm * factor] * len(ref.gradient)
    classes = sum(b.count for b in row)
    if loss.name in HINGES:
        spread = 1 + mp.mpf(max(b.value for b in row)) - min(b.value for b in row)
        if family not in (*SWEEPS, "offset"):
            return value, gradient
        value += BUDGET * eps_w * spread * factor
        if loss.gamma > 0:
            margin = BUDGET * eps_w * math.sqrt(classes) * spread / loss.gamma * factor
            gradient = [g + margin for g in gradient]
    if family not in SWEEPS:
        return value, gradient
    wide = BUDGET * eps_w * max(1.0, math.sqrt(classes / 1000))
    scale = _scale(loss, row)
    top = max(abs(g) for g in ref.gradient) * factor
    value += wide * (abs(ref.value) + scale) * factor
    extra = wide * top
    if loss.name in HINGES and loss.gamma > 0:
        extra += BUDGET * eps_w * math.sqrt(classes) * scale / loss.gamma * factor
    if setting.dtype in ("float16", "bfloat16") and work == "float32":
        value += 100 * EPS32 * max(abs(ref.value), 1) * factor
        bits = 10 if setting.dtype == "float16" else 7
        extra += 2000 * EPS32 * top + mp.mpf(2) ** (math.frexp(float(top) or 1.0)[1] - 1 - bits)
    return value, [g + extra for g in gradient]


def kinked(loss: Loss, ref: Reference, row: Row, family: str) -> bool:
    """A nonsmooth hinge row at a kink, where its subgradient is not unique: on random and
    offset rows within rounding of one, elsewhere exactly (a kink the margins only reach by
    rounding is a limit of its own)."""
    if loss.name not in HINGES or loss.gamma > 0:
        return False
    if family in SWEEPS:
        return ref.kink <= BUDGET * EPS32 * _scale(loss, row)
    return ref.kink == 0


def check(
    setting: Setting, case: Case, values: np.ndarray, gradients: np.ndarray
) -> tuple[list[Failure], float]:
    """Every value and gradient entry of a call against the references: the failures and the
    largest error over its bound."""
    loss, rows = setting.loss, case.rows
    refs = [reference(loss, row) for row in rows]
    total, factor = reduced(loss, rows, setting.reduction)
    delta, floor = terms(setting, case.classes)
    failures, worst = [], 0.0

    def compare(
        row: int | None,
        entry: str,
        actual: float,
        expected: MPReal,
        dtypes: tuple[str, str],
        allowance: MPReal,
    ) -> None:
        nonlocal worst
        r = ratio(actual, expected, dtypes, delta, allowance + floor)
        worst = max(worst, r)
        if r > 1:
            failures.append(Failure(row, entry, actual, str(mp.nstr(expected, 12)), r))

    allowances = [
        _row_allowances(setting, case.family, row, ref, factor)
        for row, ref in zip(rows, refs, strict=True)
    ]
    if setting.reduction != "none":
        allowance = mp.fsum(a[0] for a in allowances)
        compare(None, "value", float(values.reshape(())), total, setting.value_dtypes, allowance)
    for i, (row, ref) in enumerate(zip(rows, refs, strict=True)):
        if setting.reduction == "none":
            compare(i, "value", float(values[i]), ref.value, setting.value_dtypes, allowances[i][0])
        if kinked(loss, ref, row, case.family):
            continue
        clear, ratios = _clear(setting, row, ref, factor, delta, allowances[i][1], gradients[i])
        worst = max(worst, ratios)
        start = 0
        for j, (block, expected) in enumerate(zip(row, ref.gradient, strict=True)):
            stop = start + block.count
            dtypes = (setting.dtype, setting.dtype)
            for actual in np.unique(gradients[i, start:stop][~clear[start:stop]]):
                compare(
                    i,
                    f"gradient {j}",
                    float(actual),
                    expected * factor,
                    dtypes,
                    allowances[i][1][j],
                )
            start = stop
    return failures, worst


def _clear(
    setting: Setting,
    row: Row,
    ref: Reference,
    factor: MPReal,
    delta: float,
    allowances: Sequence[MPReal],
    gradient: np.ndarray,
) -> tuple[np.ndarray, float]:
    """The gradient entries within a quarter of their bound in float64 arithmetic (with the
    float64 rounding of the reference, within the bound) and their largest ratio; the others
    take the exact comparison."""
    eps, tiny, big = FINFO[setting.dtype]
    _, floor = terms(setting, sum(b.count for b in row))
    counts = [b.count for b in row]
    expected = np.repeat([float(g * factor) for g in ref.gradient], counts)
    extra = np.repeat([float(a + floor) for a in allowances], counts)
    with np.errstate(over="ignore", invalid="ignore"):
        bound = (BUDGET * eps + delta) * np.abs(expected) + eps * tiny + extra
        ratios = np.abs(gradient - expected) / bound
    clear = np.isfinite(ratios) & (ratios <= 0.25) & (np.abs(expected) < big / 2)
    return clear, float(ratios[clear].max(initial=0.0))


def describe(
    failures: Sequence[Failure], tags: Sequence[dict[str, float]] = (), limit: int = 12
) -> str:
    """The failures, with the tags of their rows (gaps, offsets) where given."""
    lines = [
        f"{f} {tags[f.row]}" if f.row is not None and f.row < len(tags) and tags[f.row] else str(f)
        for f in failures[:limit]
    ]
    if len(failures) > limit:
        lines.append(f"... {len(failures) - limit} more")
    return "\n".join(lines)
