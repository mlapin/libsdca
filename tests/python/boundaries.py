"""Boundary scan of the libsdca losses: where each loss, format and backend stops being correct.

    python tests/python/boundaries.py --device cpu --out DIR [--quick] [--jobs N]
    python tests/python/boundaries.py --device cuda --frameworks torch,jax --out DIR

The script puts the source tree's python/ first on sys.path (--installed tests the libsdca that
Python finds instead). It writes DIR/boundaries.jsonl (one line per evaluated point),
DIR/series.jsonl (one line per scanned series) and DIR/boundaries.md (the tables);
--report-only rebuilds the Markdown from DIR/series.jsonl, and --resume keeps the finished
(loss, family) parts of an interrupted run with matching scan options (DIR/parts). Use a
fresh directory when changing the scanner revision or software environment.

A series is a (loss, format, backend, family, variant): a parametric row family scanned along
one scalar parameter on a log grid, from benign values to the format's extreme (finfo.max for
scores and weights, the smallest subnormal for target masses), with 4 points per decade around
the landmarks of the format (collapse and underflow margins, 1/eps, sqrt(max), 2^63,
finfo.max / C, finfo.max / 2, finfo.max), coarser elsewhere, and bisection of the first
failure (and of the first undocumented failure). Families:

- margin: [g, 0, ..., 0] of 2 and 10 classes, the label at the top (g) or at the bottom;
- spread: [S, -S, 0, ...] of 3 and 16 classes, the label at +S, -S or 0;
- offset: O + a small pattern (O of either sign), the label at the top or second;
- ties: k tied entries at V (either sign), the rest at 0 or V - 1, the label (probability
  targets 1/k, multilabel positives) on the ties or on the rest; 8 and 600 classes;
- width: one entry L in rows of 2 to 4096 classes (--wide: 32000, 128256), label on it or not;
- masked: -inf entries beside [M, -M, 0, 1, -1, 2], and rows masked but for one entry;
- targets: probability targets of tiny total mass or tiny entries beside 1, label smoothing
  down to subnormal values, class weights from the subnormal range to finfo.max;
- upstream: the cotangent from 1 to 1e30 and down to 1e-30, on an ordinary row and on a row
  whose gradient lies in the work type's subnormal range (reduction none and sum);
- distillation: student minus teacher Delta from 0 to finfo.max (one entry, a pair, identical
  rows at spread Delta, a teacher at spread Delta, 1000 classes) and temperatures 1e-300..1e300.

Verdicts of a point (conformance/contract.py, properties.py): nan_loss (the loss is NaN or
-inf), nonfinite_grad (a gradient entry is not finite where the exact one rounds finite),
bound (a value or gradient entry outside contract.py's bound against the mpmath reference of
conformance/references.py), row_sum (properties.zero_sum_bound), error (the call raised; a
NotImplementedError is recorded as unsupported, not as a failure). Each failure is matched
against the documented limits of docs/API.md (LIMITS below); a failure that matches none is
NEW.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if "--installed" not in sys.argv and (ROOT / "python" / "libsdca").is_dir():
    if str(ROOT / "python") not in sys.path:
        sys.path.insert(0, str(ROOT / "python"))

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
import properties as P  # noqa: E402
from conformance import contract  # noqa: E402
from conformance.references import (  # noqa: E402
    HINGES,
    Block,
    CutBlockActive,
    Loss,
    Reference,
    Row,
    reference,
)
from storage_limits import LIMITS  # noqa: E402

INF = math.inf
MODES = P.MODES
BUDGET = contract.BUDGET

# --- losses ----------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Spec:
    """A loss configuration: its label, the reference Loss and the kind of properties.py
    (named_losses names, properties.PROBABILITIES for probability targets)."""

    label: str
    loss: Loss
    kind: str

    @property
    def target(self) -> str:
        if self.kind == P.PROBABILITIES:
            return "probabilities"
        if self.kind in P.MULTILABEL:
            return "mask"
        if self.kind in P.TEACHER:
            return "teacher"
        return "labels"

    @property
    def topk(self) -> bool:
        return self.kind in P.TOPK


def _spec(loss: Loss, kind: str | None = None) -> Spec:
    kind = kind or loss.name
    label = loss.label + ("[probabilities]" if kind == P.PROBABILITIES else "")
    return Spec(label, loss, kind)


SPECS: tuple[Spec, ...] = (
    _spec(Loss("cross_entropy")),
    _spec(Loss("cross_entropy"), P.PROBABILITIES),
    _spec(Loss("topk_cross_entropy", k=2)),
    _spec(Loss("topk_cross_entropy", k=3)),
    _spec(Loss("topk_cross_entropy", k=5)),
    _spec(Loss("multiclass_hinge")),
    _spec(Loss("multiclass_hinge", gamma=0.5)),
    _spec(Loss("topk_hinge_alpha", k=2)),
    _spec(Loss("topk_hinge_alpha", k=2, gamma=0.5)),
    _spec(Loss("topk_hinge_beta", k=2)),
    _spec(Loss("topk_hinge_beta", k=2, gamma=0.5)),
    _spec(Loss("multilabel_hinge")),
    _spec(Loss("multilabel_hinge", gamma=0.5)),
    _spec(Loss("multilabel_cross_entropy")),
    _spec(Loss("proximal_cross_entropy", alpha=1.0)),
    _spec(Loss("distillation")),
    _spec(Loss("proximal_distillation")),
)
SPEC = {s.label: s for s in SPECS}


def select_specs(names: Sequence[str] | None) -> list[Spec]:
    """The specs named exactly, or whose label starts with a name (topk_cross_entropy: every
    k)."""
    if not names:
        return list(SPECS)
    out = [s for s in SPECS if any(s.label == n or s.label.startswith(n) for n in names)]
    if not out:
        raise SystemExit(f"no loss matches {names}; losses: {', '.join(SPEC)}")
    return out


# --- points ----------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Point:
    """One row of a call: blocks of equal entries (logit, positive flag, teacher logit, target
    mass), per-block class weights, label smoothing, the upstream factor (the cotangent of each
    row value, or of the sum), the reduction and a temperature replacing the loss's."""

    row: Row
    weights: tuple[float, ...] | None = None
    smoothing: float = 0.0
    upstream: float = 1.0
    reduction: str = "none"
    temperature: float | None = None

    @property
    def classes(self) -> int:
        return sum(b.count for b in self.row)

    def loss(self, spec: Spec) -> Loss:
        if self.temperature is None:
            return spec.loss
        return dataclasses.replace(spec.loss, temperature=self.temperature)


@dataclasses.dataclass
class Arrays:
    logits: np.ndarray
    positives: np.ndarray
    teacher: np.ndarray
    mass: np.ndarray
    weights: np.ndarray | None

    @property
    def label(self) -> int:
        return int(np.argmax(self.positives))


def expand(point: Point) -> Arrays:
    row = point.row
    counts = [b.count for b in row]

    def rep(values: Iterable[Any], kind: Any) -> np.ndarray:
        return np.repeat(np.array(list(values), kind), counts)

    weights = None if point.weights is None else rep(point.weights, np.float64)
    return Arrays(
        logits=rep((b.value for b in row), np.float64),
        positives=rep((b.positive for b in row), bool),
        teacher=rep((b.teacher for b in row), np.float64),
        mass=rep((float(b.mass or 0.0) for b in row), np.float64),
        weights=weights,
    )


def batch_of(spec: Spec, point: Point, arrays: Arrays) -> P.Batch:
    """The properties.Batch of a point (for outside_documented_range and rounded_targets)."""
    loss = point.loss(spec)
    fields: dict[str, Any] = dict(
        k=loss.k if spec.topk else 2,
        gamma=loss.gamma,
        alpha=loss.alpha,
        temperature=loss.temperature,
        smoothing=point.smoothing,
        reduction=point.reduction,
    )
    if spec.target == "mask":
        fields["positives"] = arrays.positives
    elif spec.target == "probabilities":
        fields["probabilities"] = arrays.mass
    elif spec.target == "teacher":
        fields["teacher"] = arrays.teacher
    if arrays.weights is not None:
        fields["weight"] = arrays.weights
    return P.Batch.of(arrays.logits, [arrays.label], **fields)


def rep1(x: float, dtype: str) -> float:
    """x rounded to dtype (saturating at its max; -inf stays)."""
    if x == -INF:
        return -INF
    return float(P.represent([x], dtype)[0])


Entry = tuple[float, int, bool, float]  # value, count, positive, teacher


def make_row(spec: Spec, dtype: str, entries: Sequence[Entry]) -> Row | None:
    """The row of entries (values and teachers rounded to dtype) for the spec's target: one
    label (a positive block of several entries is split), probability masses 1/|positives|,
    multilabel masks (a negative for multilabel_hinge); None where the spec takes no such row
    (k >= classes for the top-k losses)."""
    blocks: list[Block] = []
    for value, count, positive, teacher in entries:
        if count <= 0:
            continue
        v, t = rep1(value, dtype), rep1(teacher, dtype)
        if positive and spec.target == "labels" and any(b.positive for b in blocks):
            positive = False
        if positive and spec.target == "labels" and count > 1:
            blocks += [Block(v, 1, True, t), Block(v, count - 1, False, t)]
            continue
        blocks.append(Block(v, count, positive, t))
    classes = sum(b.count for b in blocks)
    positives = sum(b.count for b in blocks if b.positive)
    if classes < 2 or (spec.target != "teacher" and positives == 0):
        return None
    if spec.topk and spec.loss.k >= classes:
        return None
    if spec.kind == "multilabel_hinge" and positives == classes:
        return None
    if spec.target == "probabilities":
        mass = rep1(1.0 / positives, dtype)
        blocks = [dataclasses.replace(b, mass=mass if b.positive else 0.0) for b in blocks]
    return tuple(blocks)


# --- families --------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Options:
    quick: bool = False
    wide: bool = False


@dataclasses.dataclass(frozen=True)
class Domain:
    """The scanned parameter: from start (benign) to stop (extreme) on a log grid, 0 first
    when zero; landmarks are dense points (none: the score landmarks of the format)."""

    start: float
    stop: float
    zero: bool = False
    kind: str = "scores"  # scores (landmarks of score magnitudes), small, upstream, temperature


T0 = (1.0, 0.0, -1.0, 0.5, -0.5, 2.0)  # a teacher row
S0 = (2.0, 0.0, -1.0, 1.0, -3.0, 0.5)  # a student row
BASE = (1.0, 0.0, -1.0, 2.0)  # targets and upstream: the label (0) below the top
OFFSET_PATTERN = (0.0, -1.0, -2.0, -0.5, -3.0, -1.5)
OFFSET_TEACHER = (0.5, -1.0, 0.0, -2.0, 1.0, -0.5)


def _plain(
    spec: Spec, values: Sequence[float], label: int, teacher: Sequence[float] = ()
) -> list[Entry]:
    t = list(teacher) + [0.0] * (len(values) - len(teacher))
    return [(v, 1, j == label, t[j]) for j, v in enumerate(values)]


class Family:
    name = ""
    frameworks = ("numpy", "torch", "jax")

    def applies(self, spec: Spec) -> bool:
        return spec.target != "teacher"

    def variants(self, spec: Spec, opts: Options) -> list[str]:
        raise NotImplementedError

    def domain(self, variant: str, dtype: str) -> Domain:
        return Domain(1.0, LIMITS[dtype].max)

    def build(self, spec: Spec, dtype: str, work: str, variant: str, p: float) -> Point | None:
        raise NotImplementedError


class Margin(Family):
    name = "margin"

    def variants(self, spec: Spec, opts: Options) -> list[str]:
        return [f"{at}-C{c}" for c in (2, 10) for at in ("top", "bottom")]

    def domain(self, variant: str, dtype: str) -> Domain:
        return Domain(1e-2, LIMITS[dtype].max, zero=True)

    def build(self, spec: Spec, dtype: str, work: str, variant: str, p: float) -> Point | None:
        at, c = variant.split("-C")
        c = int(c)
        if at == "top":
            entries = [(p, 1, True, 0.0), (0.0, c - 1, False, 0.0)]
        else:
            entries = [(p, 1, False, 0.0), (0.0, 1, True, 0.0), (0.0, c - 2, False, 0.0)]
        row = make_row(spec, dtype, entries)
        return None if row is None else Point(row)


class Spread(Family):
    name = "spread"

    def variants(self, spec: Spec, opts: Options) -> list[str]:
        if opts.quick:
            return ["plus-C3", "minus-C3", "zero-C3", "minus-C16"]
        return [f"{at}-C{c}" for c in (3, 16) for at in ("plus", "minus", "zero")]

    def build(self, spec: Spec, dtype: str, work: str, variant: str, p: float) -> Point | None:
        at, c = variant.split("-C")
        entries = [
            (p, 1, at == "plus", 0.0),
            (-p, 1, at == "minus", 0.0),
            (0.0, 1, at == "zero", 0.0),
            (0.0, int(c) - 3, False, 0.0),
        ]
        row = make_row(spec, dtype, entries)
        return None if row is None else Point(row)


class Offset(Family):
    name = "offset"

    def applies(self, spec: Spec) -> bool:
        return True

    def variants(self, spec: Spec, opts: Options) -> list[str]:
        if spec.target == "teacher":
            return ["pos", "neg"]
        return [f"{s}-{at}" for s in ("pos", "neg") for at in ("top", "second")]

    def build(self, spec: Spec, dtype: str, work: str, variant: str, p: float) -> Point | None:
        sign = -1.0 if variant.startswith("neg") else 1.0
        label = 1 if variant.endswith("second") else 0
        values = [sign * p + x for x in OFFSET_PATTERN]
        row = make_row(spec, dtype, _plain(spec, values, label, OFFSET_TEACHER))
        return None if row is None else Point(row)


class Ties(Family):
    name = "ties"

    def variants(self, spec: Spec, opts: Options) -> list[str]:
        if opts.quick:
            return ["k2-zero-pos-tie-C8", "k2-zero-neg-tie-C8", "k2-minus1-pos-tie-C8"] + [
                "k2-zero-pos-tie-C600"
            ]
        out = [
            f"k{k}-{rest}-{s}-tie-C8"
            for k in (2, 3)
            for rest in ("zero", "minus1")
            for s in ("pos", "neg")
        ]
        out += [f"k2-{rest}-{s}-rest-C8" for rest in ("zero", "minus1") for s in ("pos", "neg")]
        out += [f"k2-zero-{s}-tie-C600" for s in ("pos", "neg")]
        return out

    def build(self, spec: Spec, dtype: str, work: str, variant: str, p: float) -> Point | None:
        k, rest, sign, label, c = variant.split("-")
        k, c = int(k[1:]), int(c[1:])
        v = rep1(p if sign == "pos" else -p, dtype)
        other = 0.0 if rest == "zero" else v - 1.0
        if label == "tie":
            entries = [(v, k, True, 0.0), (other, c - k, False, 0.0)]
        else:
            entries = [(v, k, False, 0.0), (other, 1, True, 0.0), (other, c - k - 1, False, 0.0)]
        row = make_row(spec, dtype, entries)
        return None if row is None else Point(row)


class Width(Family):
    name = "width"

    def variants(self, spec: Spec, opts: Options) -> list[str]:
        widths = (2, 1000) if opts.quick else (2, 64, 1000, 4096)
        widths += (32000, 128256) if opts.wide else ()
        signs = ("pos",) if opts.quick else ("pos", "neg")
        return [f"{at}-{s}-C{c}" for c in widths for s in signs for at in ("big", "other")]

    def build(self, spec: Spec, dtype: str, work: str, variant: str, p: float) -> Point | None:
        at, sign, c = variant.split("-")
        c = int(c[1:])
        big = p if sign == "pos" else -p
        entries = [(big, 1, at == "big", 0.0), (0.0, 1, at == "other", 0.0)]
        entries.append((0.0, c - 2, False, 0.0))
        row = make_row(spec, dtype, entries)
        return None if row is None else Point(row)


class Masked(Family):
    name = "masked"

    def applies(self, spec: Spec) -> bool:
        return True

    def variants(self, spec: Spec, opts: Options) -> list[str]:
        if spec.target == "teacher":
            return ["three", "one-pos", "one-neg"]
        if opts.quick:
            return ["three-top", "three-bottom", "one-pos"]
        return ["three-top", "three-bottom", "three-mid", "one-pos", "one-neg"]

    def build(self, spec: Spec, dtype: str, work: str, variant: str, p: float) -> Point | None:
        if variant.startswith("one"):
            x = p if variant == "one-pos" else -p
            values = [x] + [-INF] * 7
            teacher = [0.0] + [-INF] * 7
            label = 0
        else:
            values = [p, -p, 0.0, 1.0, -1.0, 2.0, -INF, -INF]
            teacher = [0.5, -1.0, 0.0, -2.0, 1.0, -0.5, -INF, -INF]
            label = {"three": 0, "three-top": 0, "three-bottom": 1, "three-mid": 2}[variant]
        row = make_row(spec, dtype, _plain(spec, values, label, teacher))
        return None if row is None else Point(row)


class Targets(Family):
    name = "targets"

    def applies(self, spec: Spec) -> bool:
        return spec.kind in P.WEIGHTED

    def variants(self, spec: Spec, opts: Options) -> list[str]:
        out = ["weight-up", "weight-down"]
        if spec.kind in P.SMOOTHED:
            out = ["smooth", *out]
        if spec.target == "probabilities":
            out = ["mass-all", "mass-tail", *out]
        return out

    def domain(self, variant: str, dtype: str) -> Domain:
        if variant == "weight-up":
            return Domain(1.0, LIMITS[dtype].max)
        return Domain(1.0, P.eta(dtype), kind="small")

    def build(self, spec: Spec, dtype: str, work: str, variant: str, p: float) -> Point | None:
        row = make_row(spec, dtype, _plain(spec, BASE, 0))
        if row is None:
            return None
        if variant.startswith("mass"):
            m = rep1(p, dtype)
            masses = [m / 2, m / 4, m / 4, 0.0] if variant == "mass-all" else [1.0, m, m, 0.0]
            masses = [rep1(x, dtype) for x in masses]
            row = tuple(dataclasses.replace(b, mass=x) for b, x in zip(row, masses, strict=True))
            return Point(row)
        if variant == "smooth":
            return Point(row, smoothing=p)
        return Point(row, weights=(rep1(p, dtype), 1.0, 1.0, 1.0))


class Upstream(Family):
    name = "upstream"
    frameworks = ("torch", "jax")

    def applies(self, spec: Spec) -> bool:
        return True

    def variants(self, spec: Spec, opts: Options) -> list[str]:
        if opts.quick:
            return ["plain-up", "plain-down", "tail-up"]
        return [f"{kind}-{d}" for kind in ("plain", "tail", "sum") for d in ("up", "down")]

    def domain(self, variant: str, dtype: str) -> Domain:
        if variant.endswith("up"):
            return Domain(1.0, 1e30, kind="upstream")
        return Domain(1.0, 1e-30, kind="upstream")

    def build(self, spec: Spec, dtype: str, work: str, variant: str, p: float) -> Point | None:
        if variant.startswith("tail"):
            lim = LIMITS[work]
            gap = -(lim.log_min + lim.log_denorm_min) / 2  # mid subnormal range of the work
            values = [0.0, -gap, -gap, -gap]
            teacher = [0.0, -gap / 2, -gap / 2, -gap / 2]
        else:
            values, teacher = list(BASE), list(T0[:4])
        row = make_row(spec, dtype, _plain(spec, values, 0, teacher))
        if row is None:
            return None
        value_dtype = "float64" if dtype == "float64" else "float32"
        reduction = "sum" if variant.startswith("sum") else "none"
        return Point(row, upstream=rep1(p, value_dtype), reduction=reduction)


class Distillation(Family):
    name = "distillation"

    def applies(self, spec: Spec) -> bool:
        return spec.target == "teacher"

    def variants(self, spec: Spec, opts: Options) -> list[str]:
        if opts.quick:
            return ["shift-one", "shift-pair", "identical", "T-up", "T-down"]
        return [
            "shift-one",
            "shift-one-neg",
            "shift-pair",
            "identical",
            "teacher-spread",
            "wide-shift",
            "T-up",
            "T-down",
        ]

    def domain(self, variant: str, dtype: str) -> Domain:
        if variant == "T-up":
            return Domain(1.0, 1e300, kind="temperature")
        if variant == "T-down":
            return Domain(1.0, 1e-300, kind="temperature")
        return Domain(1e-3, LIMITS[dtype].max, zero=True)

    def build(self, spec: Spec, dtype: str, work: str, variant: str, p: float) -> Point | None:
        teacher, student = list(T0), list(T0)
        temperature = None
        if variant in ("shift-one", "shift-one-neg"):
            student[0] += p if variant == "shift-one" else -p
        elif variant == "shift-pair":
            student[0] += p
            student[1] -= p
        elif variant == "identical":
            student = teacher = [p, -p, 0.0, 1.0, -1.0, 0.5]
        elif variant == "teacher-spread":
            teacher = [p, -p, 0.0, 1.0, -1.0, 0.5]
        elif variant == "wide-shift":
            entries = [(1.0 + p, 1, False, 1.0), (0.0, 999, False, 0.0)]
            row = make_row(spec, dtype, entries)
            return None if row is None else Point(row)
        else:
            student, temperature = list(S0), p
        row = make_row(spec, dtype, _plain(spec, student, -1, teacher))
        return None if row is None else Point(row, temperature=temperature)


FAMILIES: dict[str, Family] = {
    f.name: f
    for f in (
        Margin(),
        Spread(),
        Offset(),
        Ties(),
        Width(),
        Masked(),
        Targets(),
        Upstream(),
        Distillation(),
    )
}

# --- grids -----------------------------------------------------------------------------------


def landmarks(domain: Domain, dtype: str, work: str, classes: int) -> list[float]:
    """The points around which the grid is dense: where formats collapse, underflow, lose
    their unit, overflow sums of C entries or overflow."""
    kinds = sorted({dtype, work})
    out: list[float] = []
    if domain.kind == "scores":
        for d in kinds:
            lim = LIMITS[d]
            out += [lim.digits * math.log(2), -lim.log_min, -lim.log_denorm_min]
            out += [lim.inv_eps, 2 * lim.inv_eps, math.sqrt(lim.max)]
            out += [lim.max / (2 * classes), lim.max / classes, lim.max / 2, lim.max]
        out += [2.0**62, 2.0**63, 1e16]
    elif domain.kind == "small":
        for d in kinds:
            lim = LIMITS[d]
            out += [lim.eps, lim.tiny, P.eta(d), math.sqrt(lim.tiny), classes * lim.tiny]
        out += [classes * LIMITS["float32"].tiny, 1e-30]
    elif domain.kind == "upstream":
        out += [2.0**24, 1e8, 2.0**-24, 1e-8, 1e20, 1e-20]
    else:
        f32 = LIMITS["float32"]
        out += [f32.max, f32.tiny, f32.inv_eps, f32.eps, 1e30, 1e-30]
    return out


def _pow10(e: float) -> float:
    try:
        return 10.0**e
    except OverflowError:
        return INF


def grid(domain: Domain, marks: Sequence[float], quick: bool) -> list[float]:
    """The parameters in scan order (benign to extreme)."""
    a, b = math.log10(domain.start), math.log10(domain.stop)
    up = b > a
    decades = abs(b - a)
    step = max(0.25, decades / (12 if quick else 32))
    n = int(decades / step)
    exponents = [a + (step if up else -step) * i for i in range(n + 1)]
    xs = [_pow10(e) for e in exponents] + [domain.stop]
    low, high = min(domain.start, domain.stop), max(domain.start, domain.stop)
    inside = [m for m in marks if low <= m <= high]
    xs += inside
    if not quick:
        with np.errstate(over="ignore"):
            xs += [float(np.float64(m) * 10**d) for m in inside for d in (-0.25, 0.25)]
    xs = sorted({min(max(x, low), high) for x in xs if math.isfinite(x)}, reverse=not up)
    return ([0.0] if domain.zero else []) + xs


def midpoint(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return (a + b) / 2 if a < 0 or b < 0 else max(a, b) / 4
    return math.exp((math.log(a) + math.log(b)) / 2)


# --- backends --------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Backend:
    framework: str  # numpy, torch or jax
    device: str  # cpu or cuda
    option: str  # kernel, auto or composed

    @property
    def name(self) -> str:
        if self.framework == "numpy":
            return "numpy"
        return f"{self.framework}-{self.device}-{self.option}"

    @property
    def native(self) -> bool:
        return self.option != "composed"

    @property
    def cuda(self) -> bool:
        return self.device == "cuda"


def backends(device: str, frameworks: Sequence[str], options: Sequence[str]) -> list[Backend]:
    out = []
    if "numpy" in frameworks and device == "cpu":
        out.append(Backend("numpy", "cpu", "kernel"))
    for framework in ("torch", "jax"):
        if framework in frameworks:
            out += [Backend(framework, device, o) for o in options]
    return out


@functools.cache
def available_modes(modes: Sequence[str]) -> tuple[str, ...]:
    """The modes this host can store: bfloat16 needs ml_dtypes (installed with JAX)."""
    import importlib.util

    if importlib.util.find_spec("ml_dtypes") is not None:
        return tuple(modes)
    return tuple(m for m in modes if MODES[m][0] != "bfloat16")


def unavailable(backend: Backend) -> str | None:
    import importlib.util

    if backend.framework != "numpy" and importlib.util.find_spec(backend.framework) is None:
        return f"{backend.framework} is not installed"
    if backend.framework == "torch" and backend.cuda:
        import torch

        if not torch.cuda.is_available():
            return "CUDA is unavailable"
    if backend.framework == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
        if backend.cuda and not any(d.platform == "gpu" for d in jax.devices()):
            return "no JAX GPU"
    return None


def _params(spec: Spec, point: Point) -> dict[str, float]:
    loss = point.loss(spec)
    return {"k": loss.k, "gamma": loss.gamma, "alpha": loss.alpha, "temperature": loss.temperature}


def _function(module: Any, spec: Spec, point: Point, common: dict[str, Any]):
    """The public function of the spec and its keyword arguments."""
    import named_losses

    loss = point.loss(spec)
    if spec.kind == "distillation":
        return module.distillation_loss, dict(temperature=loss.temperature, **common)
    if spec.kind == "proximal_distillation":
        kwargs = dict(temperature=loss.temperature, gamma=loss.alpha, **common)
        return module.proximal_distillation_loss, kwargs
    name = "cross_entropy" if spec.kind == P.PROBABILITIES else spec.kind
    kwargs = dict(**named_losses.parameters(name, loss.k, loss.gamma, loss.alpha), **common)
    return named_losses.function(module, name), kwargs


Result = tuple[np.ndarray, np.ndarray, str, str]  # values, gradient, gradient dtype, route


def _options(point: Point, arrays: Arrays) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if arrays.weights is not None:
        out["weight"] = arrays.weights
    if point.smoothing:
        out["label_smoothing"] = point.smoothing
    return out


def _call_numpy(spec: Spec, mode: str, point: Point, arrays: Arrays) -> Result:
    import interfaces
    import libsdca

    dtype, precision = MODES[mode]
    if dtype == "bfloat16":
        import ml_dtypes

        storage: Any = ml_dtypes.bfloat16
    else:
        storage = np.dtype(dtype)
    z = arrays.logits[None].astype(storage)
    if spec.target == "probabilities":
        target: Any = libsdca.Probabilities(arrays.mass[None])
    elif spec.target == "teacher":
        target = arrays.teacher[None].astype(storage)
    elif spec.target == "mask":
        target = arrays.positives[None]
    else:
        target = np.array([arrays.label])
    loss = interfaces.numpy_loss(spec.kind, _params(spec, point))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values, gradient = libsdca.loss_and_grad(
            z, target, loss, precision=precision, **_options(point, arrays)
        )
    g = np.asarray(gradient, np.float64)
    return np.asarray(values, np.float64).reshape(-1), g, str(gradient.dtype), "kernel"


def _call_torch(backend: Backend, spec: Spec, mode: str, point: Point, arrays: Arrays) -> Result:
    import libsdca.torch as L
    import torch

    dtype, precision = MODES[mode]
    device = "cuda" if backend.cuda else "cpu"
    kind = getattr(torch, dtype)

    def tensor(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.asarray(x, np.float64), device=device).to(kind)

    z = tensor(arrays.logits[None]).requires_grad_()
    if spec.target == "probabilities":
        target = tensor(arrays.mass[None])
    elif spec.target == "teacher":
        target = tensor(arrays.teacher[None])
    elif spec.target == "mask":
        target = torch.tensor(arrays.positives[None], device=device)
    else:
        target = torch.tensor([arrays.label], device=device)
    opts = _options(point, arrays)
    if "weight" in opts:
        opts["weight"] = tensor(opts["weight"])
    common = dict(reduction=point.reduction, backend=backend.option, precision=precision, **opts)
    fn, kwargs = _function(L, spec, point, common)
    route = _route(backend, spec, mode, point, lambda: L.explain_backend(fn, z, target, **kwargs))
    value = fn(z, target, **kwargs)
    value.backward(torch.full_like(value, point.upstream))
    assert z.grad is not None
    out = str(z.grad.dtype).removeprefix("torch.")
    values = value.detach().double().cpu().numpy().reshape(-1)
    return values, z.grad.double().cpu().numpy(), out, route


def _call_jax(backend: Backend, spec: Spec, mode: str, point: Point, arrays: Arrays) -> Result:
    import jax
    import jax.numpy as jnp
    from libsdca import jax as J

    dtype, precision = MODES[mode]
    device = jax.devices("gpu" if backend.cuda else "cpu")[0]
    kind = jnp.dtype(dtype)

    def put(x: np.ndarray, as_dtype: Any = None) -> Any:
        return jax.device_put(np.asarray(x, dtype=as_dtype or kind), device)

    if spec.target == "probabilities":
        target = put(arrays.mass[None])
    elif spec.target == "teacher":
        target = put(arrays.teacher[None])
    elif spec.target == "mask":
        target = put(arrays.positives[None], bool)
    else:
        target = put(np.array([arrays.label]), np.int32)
    opts = _options(point, arrays)
    if "weight" in opts:
        opts["weight"] = put(opts["weight"])
    common = dict(reduction=point.reduction, backend=backend.option, precision=precision, **opts)
    with jax.default_device(device):
        fn, kwargs = _function(J, spec, point, common)
        z = put(arrays.logits[None])
        route = _route(
            backend, spec, mode, point, lambda: J.explain_backend(fn, z, target, **kwargs)
        )
        value, pullback = jax.vjp(lambda x: fn(x, target, **kwargs), z)
        (gradient,) = pullback(jnp.full_like(value, point.upstream))
    out = str(gradient.dtype)
    return np.asarray(value, np.float64).reshape(-1), np.asarray(gradient, np.float64), out, route


_ROUTES: dict[tuple, str] = {}


def _route(backend: Backend, spec: Spec, mode: str, point: Point, explain: Callable) -> str:
    """The backend that the option resolves to, once per shape and options (explain_backend
    runs the call in JAX)."""
    if backend.option != "auto":
        return backend.option
    key = (backend, spec.label, mode, point.classes, point.reduction, bool(point.weights))
    if key not in _ROUTES:
        try:
            _ROUTES[key] = explain()[0]
        except Exception as e:  # noqa: BLE001 - the call itself reports it
            _ROUTES[key] = f"unexplained ({type(e).__name__})"
    return _ROUTES[key]


def call(backend: Backend, spec: Spec, mode: str, point: Point) -> Result:
    arrays = expand(point)
    if backend.framework == "numpy":
        return _call_numpy(spec, mode, point, arrays)
    if backend.framework == "torch":
        return _call_torch(backend, spec, mode, point, arrays)
    return _call_jax(backend, spec, mode, point, arrays)


# --- references ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Exact:
    """The exact loss of a point (times the class weight of the label, for the losses other
    than cross-entropy) and its gradient per point block (before the upstream factor); row and
    ref are the reference row (masked blocks removed, soft-target masses for weighted or
    smoothed cross-entropy) and its Reference; keep maps point blocks to its blocks; scale is
    the probability scale of each entry where the docs bound its cancellation (soft targets,
    multilabel_cross_entropy, distillation: T q_j + u_j, (p_j + q_j) / T), else None."""

    value: Any
    gradient: tuple[Any, ...]
    row: Row
    ref: Reference
    keep: tuple[int | None, ...]
    weight: Any
    scale: tuple[Any, ...] | None


def soft(spec: Spec, point: Point) -> bool:
    """Whether the reference takes target masses: probability targets, or cross-entropy of
    labels with class weights or label smoothing."""
    if spec.target == "probabilities":
        return True
    return spec.kind == "cross_entropy" and (point.weights is not None or point.smoothing > 0)


def _masses(spec: Spec, point: Point, kept: Sequence[Block], weights: Sequence[float]) -> Row:
    """Blocks of the target masses u_j = w_j ((1 - s) t_j + s / C), exact to 80 digits."""
    with mp.workdps(80):
        s, c = mp.mpf(point.smoothing), point.classes
        out = []
        for b, w in zip(kept, weights, strict=True):
            t = mp.mpf(b.mass or 0.0) if spec.target == "probabilities" else mp.mpf(int(b.positive))
            u = mp.mpf(w) * ((1 - s) * t + s / c)
            out.append(Block(b.value, b.count, False, 0.0, u))
    return tuple(out)


def _exact(spec: Spec, point: Point) -> Exact:
    loss = point.loss(spec)
    keep: list[int | None] = []
    kept: list[Block] = []
    weights: list[float] = []
    for i, b in enumerate(point.row):
        if b.value == -INF:
            keep.append(None)
            continue
        keep.append(len(kept))
        kept.append(b)
        weights.append(1.0 if point.weights is None else point.weights[i])
    weight = mp.mpf(1)
    if sum(b.count for b in kept) == 1:  # every class but the label masked: loss 0
        row: Row = tuple(kept)
        ref = Reference(mp.mpf(0), (mp.mpf(0),))
    elif soft(spec, point):
        row = _masses(spec, point, kept, weights)
        ref = reference(loss, row)
    elif spec.target == "teacher" and loss.temperature < 1:
        row = tuple(kept)
        ref = _cooled(loss, row)
    else:
        row = tuple(kept)
        ref = reference(loss, row)
        if point.weights is not None:
            label = next(i for i, b in enumerate(point.row) if b.positive)
            weight = mp.mpf(point.weights[label])
    gradient = tuple(mp.mpf(0) if k is None else ref.gradient[k] * weight for k in keep)
    scale = None
    if soft(spec, point) and len(ref.gradient) == len(row):
        # T q_j + u_j = g_j + 2 u_j.
        per = [abs(g) + 2 * mp.mpf(b.mass or 0) for g, b in zip(ref.gradient, row, strict=True)]
        scale = tuple(mp.mpf(0) if k is None else per[k] for k in keep)
    elif spec.kind in ("multilabel_cross_entropy", "distillation") and ref.scale:
        scale = tuple(mp.mpf(0) if k is None else ref.scale[k] * weight for k in keep)
    return Exact(ref.value * weight, gradient, row, ref, tuple(keep), weight, scale)


def _cooled(loss: Loss, row: Row) -> Reference:
    """The distillation losses at T < 1 as at T = 1 of z / T and t / T (exact to 400 digits),
    the gradient and scale over T: the references take the digits of the largest input, not
    of the input over T, so that log q - log p would cancel."""
    with mp.workdps(400):
        t = mp.mpf(loss.temperature)
        scaled = tuple(
            dataclasses.replace(
                b,
                value=mp.mpf(b.value) / t,
                teacher=b.teacher if b.teacher == -INF else mp.mpf(b.teacher) / t,
            )
            for b in row
        )
    ref = reference(dataclasses.replace(loss, temperature=1.0), scaled)
    with mp.workdps(60):
        t = mp.mpf(loss.temperature)
        return Reference(
            ref.value, tuple(g / t for g in ref.gradient), tuple(x / t for x in ref.scale)
        )


@functools.cache
def exact(spec: Spec, point: Point) -> Exact | str:
    """The Exact of a point, or why there is none."""
    try:
        return _exact(spec, point)
    except CutBlockActive:
        return "no reference: a cut block of a wide smoothed-hinge row is active"
    except (ArithmeticError, ValueError, ZeroDivisionError) as e:
        return f"no reference: {type(e).__name__}: {e}"


# --- verdicts --------------------------------------------------------------------------------


@dataclasses.dataclass
class Failure:
    kind: str  # nan_loss, nonfinite_grad, bound, row_sum, error
    entry: str  # value, gradient <block>, row
    actual: float
    expected: float | None = None  # float64 rounding of the reference
    ratio: float | None = None
    error: float | None = None  # |actual - exact| (float64)
    scale: float | None = None  # bound terms the documented-limit checks use
    keys: list[str] = dataclasses.field(default_factory=list)

    def line(self) -> str:
        parts = [f"{self.kind} {self.entry}: {self.actual!r}"]
        if self.expected is not None:
            parts.append(f"exact {self.expected!r}")
        if self.ratio is not None:
            parts.append(f"{self.ratio:.3g} x bound")
        return " ".join(parts)


@dataclasses.dataclass
class Context:
    spec: Spec
    mode: str
    backend: Backend
    family: str
    variant: str
    point: Point

    @functools.cached_property
    def dtype(self) -> str:
        return MODES[self.mode][0]

    @functools.cached_property
    def precision(self) -> str | None:
        return MODES[self.mode][1]

    @functools.cached_property
    def work(self) -> str:
        return P.work_dtype(self.dtype, self.precision)

    @functools.cached_property
    def arrays(self) -> Arrays:
        return expand(self.point)

    @functools.cached_property
    def batch(self) -> P.Batch:
        return batch_of(self.spec, self.point, self.arrays)

    @functools.cached_property
    def setting(self) -> contract.Setting:
        return contract.Setting(
            self.point.loss(self.spec),
            self.dtype,
            self.precision,
            "none",
            self.backend.framework,
            self.backend.native,
            self.backend.cuda,
            self.point.classes,
        )

    @functools.cached_property
    def outside(self) -> str | None:
        """properties.outside_documented_range of the row."""
        return P.outside_documented_range(
            self.spec.kind, self.batch, 0, self.work, self.backend.framework, self.backend.device
        )

    def spread(self, teacher: bool = False) -> float:
        """Half the spread of the finite scores (or teacher scores): no overflow."""
        z = self.arrays.teacher if teacher else self.arrays.logits
        z = z[np.isfinite(z)]
        return float(z.max() / 2 - z.min() / 2) if len(z) else 0.0

    @functools.cached_property
    def biggest(self) -> float:
        z = self.arrays.logits
        return float(np.abs(z[np.isfinite(z)]).max(initial=0.0))


def judge(ctx: Context, values: np.ndarray, gradient: np.ndarray, out: str) -> list[Failure]:
    """The failures of a call: nan_loss, nonfinite_grad, bound and row_sum."""
    failures: list[Failure] = []
    value = float(values[0])
    ex = exact(ctx.spec, ctx.point)
    invalid = math.isnan(value) or value == -INF
    if invalid:
        failures.append(Failure("nan_loss", "value", value))
    g = gradient[0]
    failures += _nonfinite(ctx, ex, g, out)
    if isinstance(ex, Exact):
        failures += _bounds(ctx, ex, value, g, out, skip_value=invalid)
    if np.isfinite(g).all():
        failures += _row_sum(ctx, g, out)
    return failures


def _blocks(point: Point) -> Iterator[tuple[int, int, int]]:
    start = 0
    for i, b in enumerate(point.row):
        yield i, start, start + b.count
        start += b.count


def _nonfinite(ctx: Context, ex: Exact | str, g: np.ndarray, out: str) -> list[Failure]:
    up = mp.mpf(ctx.point.upstream)
    failures = []
    for i, start, stop in _blocks(ctx.point):
        bad = g[start:stop][~np.isfinite(g[start:stop])]
        if not len(bad):
            continue
        if isinstance(ex, Exact):
            if contract.overflows(ex.gradient[i] * up, out):
                continue
        elif abs(ctx.point.upstream) * max(ctx.point.weights or (1.0,)) > 1:
            continue  # without a reference, a scaled gradient may round to an infinity
        failures.append(Failure("nonfinite_grad", f"gradient {i}", float(bad[0])))
    return failures


def _bounds(
    ctx: Context, ex: Exact, value: float, g: np.ndarray, out: str, skip_value: bool
) -> list[Failure]:
    setting = ctx.setting
    delta, floor = contract.terms(setting, ctx.point.classes)
    family = "offset" if ctx.family == "offset" else "boundary"
    up = mp.mpf(ctx.point.upstream)
    v_allow, _ = contract._row_allowances(setting, family, ex.row, ex.ref, ex.weight)
    _, g_allow = contract._row_allowances(setting, family, ex.row, ex.ref, ex.weight * up)
    failures = []
    if not skip_value:
        r = contract.ratio(value, ex.value, setting.value_dtypes, delta, v_allow + floor)
        if r > 1:
            failures.append(_bound_failure("value", value, ex.value, r))
    if contract.kinked(setting.loss, ex.ref, ex.row, family):
        return failures
    for i, start, stop in _blocks(ctx.point):
        expected = ex.gradient[i] * up
        k = ex.keep[i]
        allowance = (g_allow[k] if k is not None and k < len(g_allow) else 0) + floor
        entries = g[start:stop]
        for actual in np.unique(entries[np.isfinite(entries)]):
            r = contract.ratio(float(actual), expected, (out, out), delta, allowance)
            if r > 1:
                failure = _bound_failure(f"gradient {i}", float(actual), expected, r)
                if ex.scale is not None:
                    failure.scale = float(ex.scale[i] * abs(up))
                failures.append(failure)
                break
    return failures


def _bound_failure(entry: str, actual: float, expected: Any, r: float) -> Failure:
    with np.errstate(over="ignore"):
        error = float(abs(mp.mpf(actual) - expected)) if math.isfinite(actual) else INF
    return Failure("bound", entry, actual, float(expected), r, error)


def _row_sum(ctx: Context, g: np.ndarray, out: str) -> list[Failure]:
    spec = ctx.spec
    targets = P.rounded_targets(spec.kind, ctx.batch, 0, ctx.work) * abs(ctx.point.upstream)
    top = float(np.abs(g).max(initial=0.0))
    shift = max(0, math.frexp(top * len(g))[1] - 1000) if math.isfinite(top * len(g)) else 64
    if shift:  # sums that overflow math.fsum: an exact power-of-two scaling (the absolute
        g = np.ldexp(g, -shift)  # allowance, n subnormals, is then negligible)
        targets = math.ldexp(targets, -shift)
    total, bound, c = P.zero_sum_bound(
        g,
        out,
        flush=ctx.backend.framework == "jax",
        targets=targets,
        cuda=ctx.backend.cuda,
        teacher=spec.kind in P.TEACHER,
    )
    if total <= bound:
        return []
    norm = math.fsum(np.abs(g).tolist()) + targets / LIMITS[ctx.work].eps  # + 2T
    failure = Failure("row_sum", "row", total, 0.0, total / bound if bound else INF, total, norm)
    failure.entry = f"row (c = {c:.3g})"
    return [failure]


# --- documented limits -----------------------------------------------------------------------

EXTREME = (*HINGES, "proximal_cross_entropy", "multilabel_cross_entropy", *P.TEACHER)


def _extreme(ctx: Context, f: Failure) -> bool:
    """Extreme spreads: finite scores more than finfo.max (of the work) apart."""
    if ctx.outside and ctx.outside.startswith("extreme spreads"):
        return True
    half = LIMITS[ctx.work].max / 2
    beyond = ctx.spread() > half or (ctx.spec.target == "teacher" and ctx.spread(True) > half)
    if not beyond:
        return False
    if ctx.spec.kind == "distillation" and ctx.backend.framework == "numpy":
        return False  # the native CPU distillation keeps these spreads
    if ctx.spec.kind == P.PROBABILITIES:
        return ctx.backend.cuda
    return ctx.spec.kind in EXTREME


def _near_overflow(ctx: Context, f: Failure) -> bool:
    return ctx.outside == "sums near overflow"


def _topk_spread(ctx: Context, f: Failure) -> bool:
    """Top-k cross-entropy (k >= 2): CUDA across 2^63 (NaN) and, composed, from 1e16 (digits).
    The CPU kernels take the capped mean score exactly."""
    if not (ctx.spec.topk and ctx.spec.kind == "topk_cross_entropy" and ctx.spec.loss.k >= 2):
        return False
    if ctx.backend.native and not ctx.backend.cuda:
        return False
    return 2 * ctx.spread() > (1e16 if not ctx.backend.native else 2.0**63)


def _hinge_margins(ctx: Context, f: Failure) -> bool:
    """Hinge margins 1 + z_j - z_y formed in working precision: scores past 1/eps lose the
    unit (max-half, 1e30 rows), and kinks reached only by rounding."""
    if ctx.spec.kind not in HINGES:
        return False
    eps = LIMITS[ctx.work].eps
    if ctx.biggest * eps >= 0.5:
        return True
    ex = exact(ctx.spec, ctx.point)
    if ctx.spec.loss.gamma == 0 and isinstance(ex, Exact):
        return ex.ref.kink <= BUDGET * eps * (1 + ctx.biggest)
    return False


def _float32_work(ctx: Context, f: Failure) -> bool:
    """float32 work: 64 float32 eps of the value and of the row's largest gradient entry;
    16-bit storage adds 100 eps of max(|loss|, 1) and one 16-bit unit plus 2000 eps of the
    largest entry; row sums within the 64 eps."""
    if ctx.work != "float32" or f.kind not in ("bound", "row_sum") or f.error is None:
        return False
    eps = LIMITS["float32"].eps
    sixteen = ctx.dtype in ("float16", "bfloat16")
    if f.kind == "row_sum":
        return f.error <= BUDGET * eps * (f.scale or 0.0) + ctx.point.classes * P.eta(ctx.dtype)
    if f.entry == "value":
        x = abs(f.expected or 0.0)
        return f.error <= BUDGET * eps * x + (100 * eps * max(x, 1.0) if sixteen else 0.0)
    ex = exact(ctx.spec, ctx.point)
    if not isinstance(ex, Exact):
        return False
    with np.errstate(over="ignore"):
        top = float(max(abs(x) for x in ex.gradient) * abs(mp.mpf(ctx.point.upstream)))
    allowance = BUDGET * eps * top + (BUDGET * eps * LIMITS["float32"].tiny)
    if sixteen:
        bits = 10 if ctx.dtype == "float16" else 7
        allowance += 2000 * eps * top + 2.0 ** (math.frexp(top or 1.0)[1] - 1 - bits)
    return f.error <= allowance


def _row_sums(ctx: Context, f: Failure) -> bool:
    """Row sums: |sum g| <= c eps (||g||_1 + 2T) + n eta with the largest c the docs state:
    11 for the CPU kernels in float64 work (4 in float32 work: distillation 1.8), 16 and 128
    for the CUDA kernels in float32 and float64 work (2 eta per entry)."""
    if f.kind != "row_sum" or f.error is None:
        return False
    double = ctx.work == "float64"
    c = (128.0 if double else 16.0) if ctx.backend.cuda else (11.0 if double else 4.0)
    eta = P.eta(ctx.dtype if ctx.backend.framework != "numpy" else ctx.work)
    out_eps = LIMITS["float32" if ctx.dtype != "float64" else "float64"].eps
    absolute = ctx.point.classes * eta * (2 if ctx.backend.cuda else 1)
    return f.error <= c * max(out_eps, LIMITS[ctx.work].eps) * (f.scale or 0.0) + absolute


def _unscaled(ctx: Context, f: Failure) -> float | None:
    """|exact| of a failure before the upstream factor (None: not an entry)."""
    if f.expected is None:
        return None
    if f.entry == "value":
        return abs(f.expected)
    return abs(f.expected / ctx.point.upstream)


def _proximal_underflow(ctx: Context, f: Failure) -> bool:
    """Proximal rows whose terms underflow the work (Representation; distillation-tail).
    Under an upstream factor: _upstream_tail."""
    x = _unscaled(ctx, f)
    if ctx.spec.kind not in P.PROXIMAL or x is None:
        return False
    if f.entry != "value" and ctx.point.upstream != 1:
        return False
    return x < 4 * LIMITS[ctx.work].tiny


def _subnormal_sum(ctx: Context, f: Failure) -> bool:
    if ctx.spec.kind != "multilabel_cross_entropy" or ctx.work != "float64" or f.error is None:
        return False
    return ctx.point.upstream == 1 and f.error <= ctx.point.classes * P.eta("float64")


def _jax_flush(ctx: Context, f: Failure) -> bool:
    """JAX on the CPU: XLA's own operations flush subnormal results (composed, reductions)."""
    x = _unscaled(ctx, f)
    if ctx.backend.framework != "jax" or ctx.backend.cuda or x is None:
        return False
    if ctx.backend.native and ctx.point.reduction == "none":
        return False
    dtype = "float64" if ctx.dtype == "float64" else "float32"
    return x < LIMITS[dtype].tiny * 2**24


def _tiny_targets(ctx: Context, f: Failure) -> bool:
    if ctx.work != "float32" or not soft(ctx.spec, ctx.point):
        return False
    a = ctx.arrays
    t = a.mass if ctx.spec.target == "probabilities" else a.positives.astype(float)
    u = (1 - ctx.point.smoothing) * t + ctx.point.smoothing / len(t)
    u = u * (a.weights if a.weights is not None else 1.0)
    return bool((u[u > 0] < 2**24 * LIMITS["float32"].tiny).any())


def _cancellation(ctx: Context, f: Failure) -> bool:
    """Entries that are differences of nearly equal probabilities: 64 eps of T q_j + u_j
    (soft targets, multilabel_cross_entropy) or of (p_j + q_j) / T (distillation)."""
    if f.kind != "bound" or f.scale is None or f.error is None:
        return False
    eps = LIMITS[ctx.work].eps
    return f.error <= BUDGET * eps * (abs(f.expected or 0.0) + f.scale)


def _cuda_upstream(ctx: Context, f: Failure) -> bool:
    if not ctx.backend.cuda or ctx.work != "float32" or abs(ctx.point.upstream) <= 1:
        return False
    if ctx.spec.kind not in ("cross_entropy", P.PROBABILITIES) or f.expected is None:
        return False
    return abs(f.expected / ctx.point.upstream) < 2**24 * LIMITS["float32"].tiny


def _upstream_tail(ctx: Context, f: Failure) -> bool:
    """Gradient entries computed in the work type's subnormal range, then scaled by the
    upstream factor: a few subnormal steps of absolute error, scaled with the entry."""
    if f.kind != "bound" or f.error is None or abs(ctx.point.upstream) <= 1:
        return False
    step = float(np.finfo(ctx.work).smallest_subnormal)
    return f.error <= 2 * ctx.point.classes * abs(ctx.point.upstream) * step


def _jax_cuda_smoothing(ctx: Context, f: Failure) -> bool:
    return (
        ctx.backend.framework == "jax"
        and ctx.backend.cuda
        and ctx.point.smoothing > 0
        and ctx.dtype != "float64"
    )


def _capped_odds(ctx: Context, f: Failure) -> bool:
    """Top-k entropy rounds its log-odds to float64 before exp: an entry x is off by up to
    |ln x| eps / 2 relative (350 eps near exp(-700)); the CPU kernels keep the low part of the
    capped mean score."""
    if ctx.spec.kind != "topk_cross_entropy" or ctx.work != "float64" or f.kind != "bound":
        return False
    if ctx.backend.native and not ctx.backend.cuda:
        return False
    x, scaled = _unscaled(ctx, f), abs(f.expected or 0.0)
    if not x or f.error is None:
        return False
    return f.error <= (BUDGET + abs(math.log(x)) / 2) * LIMITS["float64"].eps * scaled


def _composed(ctx: Context, f: Failure) -> bool:
    return not ctx.backend.native


@dataclasses.dataclass(frozen=True)
class Limit:
    key: str
    docs: str  # the docs/API.md Known limits entry
    applies: Callable[[Context, Failure], bool]


LIMITS_DOCUMENTED: tuple[Limit, ...] = (
    Limit("extreme-spreads", "Extreme spreads", _extreme),
    Limit("sums-near-overflow", "Sums near overflow", _near_overflow),
    Limit("topk-spread", "Top-k cross-entropy at extreme spreads", _topk_spread),
    Limit("hinge-margins", "Extreme spreads (margins that cancel); contract.py", _hinge_margins),
    Limit("proximal-underflow", "Representation (proximal rows)", _proximal_underflow),
    Limit("subnormal-sum", "Representation (float64 multilabel)", _subnormal_sum),
    Limit("jax-cpu-flush", "JAX on the CPU", _jax_flush),
    Limit("tiny-targets", "Tiny probability targets", _tiny_targets),
    Limit("cancellation", "Cancellation; Distillation", _cancellation),
    Limit("cuda-upstream", "Gradients near the subnormal range (CUDA)", _cuda_upstream),
    Limit("upstream-tail", "Gradients near the subnormal range", _upstream_tail),
    Limit("jax-cuda-smoothing", "JAX label smoothing on CUDA", _jax_cuda_smoothing),
    Limit("capped-odds", "Top-k cross-entropy (capped rows near exp(-700))", _capped_odds),
    Limit("row-sums", "Row sums (the largest c observed)", _row_sums),
    Limit("float32-work", "float32 work", _float32_work),
    Limit("composed", 'backend="composed"', _composed),
)
DOCS = {limit.key: limit.docs for limit in LIMITS_DOCUMENTED}


def classify(ctx: Context, failures: Sequence[Failure]) -> None:
    """The documented limits of each failure (none: NEW)."""
    for f in failures:
        f.keys = [limit.key for limit in LIMITS_DOCUMENTED if limit.applies(ctx, f)]


# --- evaluation ------------------------------------------------------------------------------


def describe(spec: Spec, point: Point) -> dict[str, Any]:
    """A point as a compact input: blocks [value, count] and the spec's target per block."""
    blocks = []
    for b in point.row:
        entry: list[Any] = [b.value, b.count]
        if spec.target == "probabilities":
            entry.append(float(b.mass or 0.0))
        elif spec.target == "teacher":
            entry.append(b.teacher)
        else:
            entry.append(bool(b.positive))
        blocks.append(entry)
    out: dict[str, Any] = {"blocks": blocks}
    if point.weights is not None:
        out["weights"] = list(point.weights)
    for key in ("smoothing", "upstream", "reduction", "temperature"):
        value = getattr(point, key)
        default = {"smoothing": 0.0, "upstream": 1.0, "reduction": "none", "temperature": None}
        if value != default[key]:
            out[key] = value
    return out


def _json_float(x: float | None) -> Any:
    if x is None or math.isfinite(x):
        return x
    return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")


def evaluate(ctx: Context) -> dict[str, Any]:
    """The verdict of one point as a JSON record."""
    record: dict[str, Any] = {}
    try:
        values, gradient, out, route = call(ctx.backend, ctx.spec, ctx.mode, ctx.point)
    except NotImplementedError as e:
        return {"status": "unsupported", "message": str(e)[:200]}
    except Exception as e:  # noqa: BLE001 - every raised error is a verdict
        failure = Failure("error", f"{type(e).__name__}: {str(e)[:160]}", math.nan)
        record.update(status="fail", failures=[failure], route=None, value=None)
        failure.keys = ["composed"] if not ctx.backend.native else []
        return record
    failures = judge(ctx, values, gradient, out)
    classify(ctx, failures)
    ex = exact(ctx.spec, ctx.point)
    record.update(
        status="fail" if failures else "pass",
        failures=failures,
        route=route,
        value=float(values[0]),
        exact=float(ex.value) if isinstance(ex, Exact) else None,
        exact_zero=isinstance(ex, Exact) and zero_passes(ctx, ex),
    )
    if not isinstance(ex, Exact):
        record["reference"] = ex
    return record


def to_json(ctx: Context, param: float, refine: bool, record: dict[str, Any]) -> dict[str, Any]:
    failures: list[Failure] = record.get("failures", [])
    line: dict[str, Any] = {
        "loss": ctx.spec.label,
        "mode": ctx.mode,
        "backend": ctx.backend.name,
        "family": ctx.family,
        "variant": ctx.variant,
        "classes": ctx.point.classes,
        "param": param,
        "refine": refine,
        "status": record["status"],
    }
    if record.get("route") not in (None, ctx.backend.option):
        line["route"] = record["route"]
    if "value" in record:
        line["value"] = _json_float(record["value"])
        line["exact"] = _json_float(record.get("exact"))
        if record.get("exact_zero"):
            line["exact_zero"] = True
    if record.get("reference"):
        line["reference"] = record["reference"]
    if record.get("message"):
        line["message"] = record["message"]
    if failures:
        line["new"] = any(not f.keys for f in failures)
        line["outside"] = ctx.outside
        line["failures"] = [
            {
                "kind": f.kind,
                "entry": f.entry,
                "actual": _json_float(f.actual),
                "exact": _json_float(f.expected),
                "ratio": _json_float(f.ratio),
                "keys": f.keys,
            }
            for f in failures[:6]
        ]
        line["input"] = describe(ctx.spec, ctx.point)
    return line


# --- series ----------------------------------------------------------------------------------


@dataclasses.dataclass
class Scan:
    """The evaluated points of a series in scan order."""

    params: list[float] = dataclasses.field(default_factory=list)
    lines: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    points: list[Point] = dataclasses.field(default_factory=list)


LineTest = Callable[[dict[str, Any]], bool]
Visit = Callable[[float, bool], tuple[Point | None, dict[str, Any] | None]]
Bounds = tuple[float | None, float]  # (last passing, first failing) params; None: the first fails


def _failed(line: dict[str, Any]) -> bool:
    return line["status"] == "fail"


def _new(line: dict[str, Any]) -> bool:
    return _failed(line) and line.get("new", False)


def _zero(line: dict[str, Any]) -> bool:
    return line.get("value") == 0.0


def _exact_zero(line: dict[str, Any]) -> bool:
    return line.get("exact_zero", False)


def zero_passes(ctx: Context, ex: Exact) -> bool:
    """Whether a returned loss of 0 is within the bound (the exact loss is below the absolute
    floor of the output)."""
    delta, floor = contract.terms(ctx.setting, ctx.point.classes)
    return contract.ratio(0.0, ex.value, ctx.setting.value_dtypes, delta, floor) <= 1


def run_series(
    spec: Spec,
    family: Family,
    variant: str,
    mode: str,
    backend: Backend,
    opts: Options,
    refine: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The points of a series (grid, then bisection) and its summary."""
    dtype, precision = MODES[mode]
    work = P.work_dtype(dtype, precision)
    domain = family.domain(variant, dtype)
    probe = family.build(spec, dtype, work, variant, domain.start)
    if probe is None:
        return [], {}
    marks = landmarks(domain, dtype, work, probe.classes)
    scan = Scan()

    def visit(param: float, refined: bool) -> tuple[Point | None, dict[str, Any] | None]:
        point = family.build(spec, dtype, work, variant, param)
        if point is None:
            return None, None
        ctx = Context(spec, mode, backend, family.name, variant, point)
        return point, to_json(ctx, param, refined, evaluate(ctx))

    for param in grid(domain, marks, opts.quick):
        point = family.build(spec, dtype, work, variant, param)
        if point is None or (scan.points and point == scan.points[-1]):
            continue
        _, line = visit(param, False)
        assert line is not None
        scan.params.append(param)
        scan.points.append(point)
        scan.lines.append(line)
    extra: list[dict[str, Any]] = []
    bounds: dict[str, Bounds] = {}
    tests: list[tuple[str, LineTest]] = [("first", _failed), ("new", _new)]
    if family.name == "margin":
        tests += [("zero", _zero), ("exact_zero", _exact_zero)]
    for name, test in tests:
        found = _bisect(scan, test, visit, refine, extra)
        if found:
            bounds[name] = found
    return scan.lines + extra, summarize(spec, family, variant, mode, backend, scan, bounds)


def _bisect(
    scan: Scan, test: LineTest, visit: Visit, steps: int, extra: list[dict[str, Any]]
) -> Bounds | None:
    """(last passing, first failing) params of the first failure of test, refined."""
    index = next((i for i, line in enumerate(scan.lines) if test(line)), None)
    if index is None:
        return None
    if index == 0:
        return (None, scan.params[0])
    a, b = scan.params[index - 1], scan.params[index]
    pa, pb = scan.points[index - 1], scan.points[index]
    for _ in range(steps):
        mid = midpoint(a, b)
        point, line = visit(mid, True)
        if point is None or line is None or point in (pa, pb):
            break
        extra.append(line)
        if test(line):
            b, pb = mid, point
        else:
            a, pa = mid, point
    return (a, b)


def _pattern(fails: Sequence[bool]) -> str:
    if not any(fails):
        return "none"
    first = fails.index(True)
    if all(fails[first:]):
        return "continues" if first else "everywhere"
    return "isolated" if sum(fails) == 1 else f"intermittent ({sum(fails)}/{len(fails)})"


def summarize(
    spec: Spec,
    family: Family,
    variant: str,
    mode: str,
    backend: Backend,
    scan: Scan,
    bounds: dict[str, Bounds],
) -> dict[str, Any]:
    lines = scan.lines
    fails = [_failed(line) for line in lines]
    news = [_new(line) for line in lines]
    summary: dict[str, Any] = {
        "loss": spec.label,
        "mode": mode,
        "backend": backend.name,
        "family": family.name,
        "variant": variant,
        "classes": lines[0]["classes"] if lines else None,
        "points": len(lines),
        "failing": sum(fails),
        "unsupported": sum(line["status"] == "unsupported" for line in lines),
        "pattern": _pattern(fails),
        "routes": sorted({line["route"] for line in lines if "route" in line}),
        "params": [scan.params[0], scan.params[-1]] if lines else None,
    }
    if "first" in bounds:
        first = next(line for line in lines if _failed(line))
        summary["first"] = list(bounds["first"])
        summary["kinds"] = sorted({f["kind"] for f in first["failures"]})
        summary["keys"] = sorted({k for f in first["failures"] for k in f["keys"]})
        summary["documented"] = sorted(
            {k for line in lines if _failed(line) for f in line["failures"] for k in f["keys"]}
        )
    if "new" in bounds:
        new = next(line for line in lines if _new(line))
        summary["new"] = list(bounds["new"])
        summary["new_pattern"] = _pattern(news)
        summary["new_kinds"] = sorted({f["kind"] for f in new["failures"] if not f["keys"]})
        summary["repro"] = {
            "param": new["param"],
            "input": new["input"],
            "failures": [f for f in new["failures"] if not f["keys"]][:3],
        }
    if family.name == "margin":
        summary["zero"] = {"actual": bounds.get("zero"), "exact": bounds.get("exact_zero")}
    return summary


# --- tasks -----------------------------------------------------------------------------------


@functools.cache
def build_digest() -> str:
    """Fingerprint package sources, shared headers and the selected native binary."""
    import libsdca
    from libsdca._native import _library_path

    package = Path(libsdca.__file__).resolve().parent
    members = {
        path.relative_to(package).as_posix(): path
        for path in package.rglob("*")
        if path.suffix in {".py", ".cpp", ".h"}
    }
    # Editable builds keep the shared core headers outside the Python package.
    headers = package.parents[1] / "include" / "sdca"
    if package.parent.name == "python" and headers.is_dir():
        members.update(
            ("include/sdca/" + path.relative_to(headers).as_posix(), path)
            for path in headers.rglob("*.h")
        )
    native = _library_path()
    if native.is_file():
        members["native-binary"] = native
    digest = hashlib.sha256()
    for name, path in sorted(members.items()):
        digest.update(name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class Task:
    spec: str
    family: str
    modes: tuple[str, ...]
    backends: tuple[Backend, ...]
    opts: Options
    refine: int
    build: str = dataclasses.field(default_factory=build_digest)

    @property
    def name(self) -> str:
        safe = self.spec.replace("[", "_").replace("]", "").replace(",", "_").replace("=", "")
        return f"{safe}-{self.family}"


def run_task(task: Task, directory: Path | None = None) -> tuple[list[dict], list[dict], float]:
    """The point lines and series summaries of a (loss, family); with directory, also written
    to directory/parts/<task>.json."""
    start = time.time()
    spec, family = SPEC[task.spec], FAMILIES[task.family]
    lines: list[dict] = []
    series: list[dict] = []
    for variant in family.variants(spec, task.opts):
        for mode in task.modes:
            for backend in task.backends:
                if backend.framework not in family.frameworks or unavailable(backend):
                    continue
                points, summary = run_series(
                    spec, family, variant, mode, backend, task.opts, task.refine
                )
                lines += points
                if summary:
                    series.append(summary)
    exact.cache_clear()
    elapsed = time.time() - start
    if directory is not None:
        parts = directory / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        tmp = parts / f"{task.name}.json.tmp"
        tmp.write_text(
            json.dumps(
                {
                    "task": dataclasses.asdict(task),
                    "lines": lines,
                    "series": series,
                    "elapsed": elapsed,
                }
            )
        )
        tmp.replace(parts / f"{task.name}.json")
    return lines, series, elapsed


def tasks(args: argparse.Namespace) -> list[Task]:
    specs = select_specs(args.losses)
    families = args.families or list(FAMILIES)
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise SystemExit(f"unknown families {sorted(unknown)}; families: {', '.join(FAMILIES)}")
    chosen = tuple(backends(args.device, args.frameworks, args.backends))
    if not chosen:
        raise SystemExit("no backend selected for this device")
    opts = Options(quick=args.quick, wide=args.wide)
    refine = args.refine if args.refine is not None else (3 if args.quick else 8)
    return [
        Task(s.label, f, available_modes(tuple(args.modes)), chosen, opts, refine)
        for s in specs
        for f in families
        if FAMILIES[f].applies(s)
    ]


def _worker_init(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass


def _part(directory: Path, task: Task) -> tuple[list, list, float] | None:
    path = directory / "parts" / f"{task.name}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("task") != json.loads(json.dumps(dataclasses.asdict(task))):
        return None
    return data["lines"], data["series"], data["elapsed"]


def _results(args: argparse.Namespace, todo: list[Task], directory: Path) -> Iterator:
    """(task, (lines, series, seconds)) in the order of todo, finished parts first kept
    (--resume); --jobs runs the others in spawned processes."""
    kept = {t.name: _part(directory, t) for t in todo} if args.resume else {}
    pending = [t for t in todo if not kept.get(t.name)]
    if args.jobs <= 1:
        for task in todo:
            yield task, kept.get(task.name) or run_task(task, directory)
        return
    import multiprocessing

    threads = max(1, (os.cpu_count() or 1) // args.jobs)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        args.jobs, mp_context=context, initializer=_worker_init, initargs=(threads,)
    ) as pool:
        futures = {t.name: pool.submit(run_task, t, directory) for t in pending}
        for task in todo:
            yield task, kept.get(task.name) or futures[task.name].result()


def run(args: argparse.Namespace) -> None:
    """Every task; the point lines streamed to boundaries.jsonl as their tasks finish."""
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    todo = tasks(args)
    start = time.time()
    series: list[dict] = []
    points, seconds = 0, 0.0
    with (directory / "boundaries.jsonl").open("w") as stream:
        for task, (lines, found, elapsed) in _results(args, todo, directory):
            for line in lines:
                stream.write(json.dumps(line) + "\n")
            points += len(lines)
            series += found
            seconds += elapsed
            print(f"{task.name}: {len(lines)} points, {elapsed:.1f} s", flush=True)
    elapsed = time.time() - start
    with (directory / "series.jsonl").open("w") as stream:
        for s in series:
            stream.write(json.dumps(s) + "\n")
    meta = metadata(args, elapsed, seconds, points)
    (directory / "meta.json").write_text(json.dumps(meta, indent=1))
    (directory / "boundaries.md").write_text(report(series, meta))
    print(f"{points} points, {len(series)} series in {elapsed:.1f} s -> {directory}")
    if not any(s["points"] > s["unsupported"] for s in series):
        raise SystemExit("no supported points evaluated")


def metadata(args: argparse.Namespace, elapsed: float, cpu: float, points: int) -> dict:
    import libsdca

    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        commit = ""
    versions = {"numpy": np.__version__, "mpmath": mp.__version__}
    for name in ("torch", "jax"):
        if name in args.frameworks:
            try:
                versions[name] = __import__(name).__version__
            except ImportError:
                versions[name] = None
    return {
        "settings": {
            key: getattr(args, key)
            for key in (
                "device",
                "modes",
                "backends",
                "frameworks",
                "families",
                "losses",
                "quick",
                "wide",
                "refine",
                "installed",
            )
        },
        "commit": commit,
        "libsdca_version": libsdca.__version__,
        "build_sha256": build_digest(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "processor": platform.processor(),
        "versions": versions,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed": round(elapsed, 1),
        "task_seconds": round(cpu, 1),
        "points": points,
    }


# --- report ----------------------------------------------------------------------------------


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def _interval(pair: Sequence[Any] | None) -> str:
    if not pair:
        return "-"
    a, b = pair
    return f"{_fmt(b)} (from the first point)" if a is None else f"({_fmt(a)}, {_fmt(b)}]"


def _merge(rows: Iterable[tuple[tuple, str]]) -> list[tuple[tuple, list[str]]]:
    """Rows that differ only by backend, merged (backends listed)."""
    merged: dict[tuple, list[str]] = {}
    for key, backend in rows:
        merged.setdefault(key, []).append(backend)
    return list(merged.items())


def report(series: Sequence[dict], meta: dict) -> str:
    failing = [s for s in series if s.get("first")]
    new = [s for s in series if s.get("new")]
    out = [
        "# Boundary scan",
        "",
        f"{meta['date']}, scanner commit {meta['commit']}, {meta['platform']} ({meta['processor']}), "
        f"versions {meta['versions']}; {meta['points']} points in "
        f"{meta['elapsed']} s.",
        "",
        f"{len(series)} series, {len(failing)} with failures, {len(new)} with failures that "
        "match no documented limit (NEW). Parameters: the first breaking parameter as "
        "(last passing, first failing] after bisection. Kinds: nan_loss, nonfinite_grad, bound "
        "(contract.py), row_sum (properties.zero_sum_bound), error (the call raised).",
        "",
        "## NEW failures",
        "",
    ]
    if settings := meta.get("settings"):
        out[4:4] = [f"Scan options: `{json.dumps(settings, sort_keys=True)}`.", ""]
    if build := meta.get("build_sha256"):
        out[4:4] = [f"Tested build: `{build}` (SHA-256).", ""]
    if not any(s["points"] > s["unsupported"] for s in series):
        out.append("No supported points were evaluated.")
    elif new:
        out += [
            "| loss | format | family:variant | first NEW | kinds | pattern | backends | input |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        rows = (
            (
                (
                    s["loss"],
                    s["mode"],
                    f"{s['family']}:{s['variant']}",
                    _interval(s["new"]),
                    ", ".join(s["new_kinds"]),
                    s["new_pattern"],
                    _repro(s["repro"]),
                ),
                s["backend"],
            )
            for s in new
        )
        for key, names in _merge(rows):
            cells = [*key[:6], ", ".join(names), key[6]]
            out.append("| " + " | ".join(cells) + " |")
    else:
        out.append("None.")
    composed = [s for s in failing if s["backend"].endswith("-composed")]
    kernels = [s for s in failing if not s["backend"].endswith("-composed")]
    out += _failing("Failing series: libsdca's kernels (numpy, kernel, auto)", kernels)
    out += _failing('Failing series: backend="composed"', composed)
    out += _zeros(series) + _limits_seen(failing)
    return "\n".join(out) + "\n"


def _failing(title: str, failing: Sequence[dict]) -> list[str]:
    out = [
        "",
        f"## {title}",
        "",
        "| loss | format | family:variant | first break | kinds | pattern | documented | backends |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = (
        (
            (
                s["loss"],
                s["mode"],
                f"{s['family']}:{s['variant']}",
                _interval(s["first"]),
                ", ".join(s["kinds"]),
                s["pattern"],
                ", ".join(s["keys"]) or "NEW",
            ),
            s["backend"],
        )
        for s in failing
    )
    for key, names in _merge(rows):
        out.append("| " + " | ".join([*key, ", ".join(names)]) + " |")
    return out if failing else out[:3] + ["None."]


def _repro(repro: dict) -> str:
    blocks = repro["input"]["blocks"]
    shown = ", ".join(f"{_fmt(v)}x{n}:{t}" if n > 1 else f"{_fmt(v)}:{t}" for v, n, t in blocks)
    extra = {k: v for k, v in repro["input"].items() if k != "blocks"}
    first = repro["failures"][0] if repro["failures"] else {}
    what = f"{first.get('kind')} {first.get('entry')} {first.get('actual')} vs {first.get('exact')}"
    return f"`[{shown}]` {extra or ''} {what}".strip()


def _zeros(series: Sequence[dict]) -> list[str]:
    rows = [s for s in series if s.get("zero")]
    if not rows:
        return []
    out = [
        "",
        "## First zeros of the margin family",
        "",
        "The first margin g where the returned loss is 0, and where a returned 0 is within the "
        "bound (the exact loss below the output's absolute floor), as (last nonzero, first "
        "zero] after bisection.",
        "",
        "| loss | format | variant | loss 0 from | exact 0 from | backends |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    keyed = (
        (
            (
                s["loss"],
                s["mode"],
                s["variant"],
                _interval(s["zero"]["actual"]),
                _interval(s["zero"]["exact"]),
            ),
            s["backend"],
        )
        for s in rows
        if s["zero"]["actual"] is not None or s["zero"]["exact"] is not None
    )
    for key, names in _merge(keyed):
        out.append("| " + " | ".join([*key, ", ".join(names)]) + " |")
    return out


def _limits_seen(failing: Sequence[dict]) -> list[str]:
    seen: dict[str, int] = {}
    for s in failing:
        for key in s.get("documented", []):
            seen[key] = seen.get(key, 0) + 1
    out = [
        "",
        "## Documented limits",
        "",
        "| limit | docs/API.md, Known limits | series |",
        "| --- | --- | --- |",
    ]
    out += [f"| {k} | {DOCS[k]} | {seen.get(k, 0)} |" for k in DOCS]
    return out


def report_only(directory: Path) -> None:
    series = [json.loads(line) for line in (directory / "series.jsonl").open()]
    meta = json.loads((directory / "meta.json").read_text())
    (directory / "boundaries.md").write_text(report(series, meta))


# --- CLI -------------------------------------------------------------------------------------


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])

    def items(text: str) -> list[str]:
        return [x for x in text.split(",") if x]

    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--backends", type=items, default=["kernel", "auto", "composed"])
    parser.add_argument("--frameworks", type=items, default=["numpy", "torch", "jax"])
    parser.add_argument("--modes", type=items, default=list(MODES))
    parser.add_argument("--families", type=items, default=None)
    parser.add_argument("--losses", type=items, default=None, help="labels or label prefixes")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quick", action="store_true", help="coarse grids, fewer variants")
    parser.add_argument("--wide", action="store_true", help="also 32000 and 128256 classes")
    parser.add_argument("--refine", type=int, default=None, help="bisection steps (8; quick 3)")
    parser.add_argument("--jobs", type=int, default=1, help="processes over (loss, family)")
    parser.add_argument("--resume", action="store_true", help="keep finished parts of DIR")
    parser.add_argument("--report-only", action="store_true", help="rebuild boundaries.md")
    parser.add_argument("--installed", action="store_true", help="the libsdca Python finds")
    parser.add_argument("--list", action="store_true", help="print the tasks and exit")
    args = parser.parse_args(argv)
    bad = set(args.modes) - set(MODES)
    if bad:
        parser.error(f"unknown modes {sorted(bad)}; modes: {', '.join(MODES)}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse(argv)
    if args.report_only:
        report_only(Path(args.out))
        return
    if args.list:
        for task in tasks(args):
            print(task.name)
        return
    run(args)


if __name__ == "__main__":
    main()
