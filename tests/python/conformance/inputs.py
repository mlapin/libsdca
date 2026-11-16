"""Input rows of the conformance cells, as represented in each storage dtype.

A family gives the rows of one call (equal widths). Each Case carries its rows and tags that
the limitation tables match (the gap of a far-tail row, the teacher's gap, the width).
Families:

- ordinary: random normal rows (scale 3), random labels, positives and teachers (one teacher
  equal to its student: an exact zero gradient);
- ties: integer rows in [-2, 2];
- offset: random rows plus a common offset of 1/eps or 2/eps, and [o, o - 1, ...] rows, where
  rounding of the inputs destroys a unit gap (release_numerics.py);
- extreme: [max, -max, ...] and [-max, max, ...] at finfo.max;
- far_tail: the target at 0 and competitors at -gap (gaps 80-110, and 700-750 in float64)
  at 10, 1000, 4096 and 32768 classes; distillation rows scale the student and teacher gaps
  by T, so that the tails are those of the softmax;
- capped: top-k rows [0, -m, -m - 40, ...] whose cap set is known (topk_rows);
- documented: documented limit examples that are not in the grids.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

import numpy as np
from topk_rows import CAPPED_GAPS

from conformance.references import (
    DISTILLATION,
    HINGES,
    PROXIMAL,
    Block,
    Loss,
    Row,
    has_reference,
)

DTYPES = ("float16", "bfloat16", "float32", "float64")
# eps, tiny (smallest normal) and max of each storage dtype.
FINFO = {
    "float16": (2.0**-10, 2.0**-14, 65504.0),
    "bfloat16": (2.0**-7, 2.0**-126, float.fromhex("0x1.fep127")),
    "float32": (2.0**-23, 2.0**-126, float.fromhex("0x1.fffffep127")),
    "float64": (2.0**-52, 2.0**-1022, float.fromhex("0x1.fffffffffffffp1023")),
}
WIDE = (10, 1000, 4096, 32768)
_GAPS32 = (80, 87, 90, 95, 100, 104, 106, 110)
GAPS: dict[str, tuple[int, ...]] = {
    "float16": _GAPS32,
    "bfloat16": _GAPS32,
    "float32": _GAPS32,
    "float64": (80, 104, 700, 708, 720, 745, 750),
}
# (student gap, teacher gap) of distillation far-tail rows, in units of T.
_PAIRS32 = ((80, 90), (90, 80), (100, 100), (104, 104), (100, 110), (110, 100), (110, 111))
PAIRS: dict[str, tuple[tuple[int, int], ...]] = {
    "float16": _PAIRS32,
    "bfloat16": _PAIRS32,
    "float32": _PAIRS32,
    "float64": ((700, 710), (710, 700), (745, 745), (740, 750), (750, 740)),
}


@dataclass(frozen=True)
class Case:
    """The rows of one call and their tags (one dict per row)."""

    family: str
    rows: tuple[Row, ...]
    tags: tuple[dict[str, float], ...] = field(compare=False)

    @property
    def classes(self) -> int:
        return sum(b.count for b in self.rows[0])


def represent(values: np.ndarray, dtype: str) -> np.ndarray:
    """float64 values rounded to dtype (to nearest even; bfloat16 through float32)."""
    x = np.asarray(values, np.float64)
    if dtype == "float64":
        return x.copy()
    if dtype != "bfloat16":
        return x.astype(dtype).astype(np.float64)
    bits = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32).astype(np.float64)


def _row(values: np.ndarray, positive: np.ndarray, teacher: np.ndarray | None = None) -> Row:
    """One block per entry."""
    t = np.zeros_like(values) if teacher is None else teacher
    return tuple(
        Block(float(v), 1, bool(p), float(s)) for v, p, s in zip(values, positive, t, strict=True)
    )


def _random_rows(
    loss: Loss, dtype: str, z: np.ndarray, rng: np.random.Generator, teacher: np.ndarray
) -> list[Row]:
    """Rows of z with random labels or positive sets (target kind of loss)."""
    n, c = z.shape
    rows = []
    for i in range(n):
        positive = np.zeros(c, bool)
        if loss.name in ("multilabel_cross_entropy", "multilabel_hinge"):
            positive[rng.permutation(c)[: rng.integers(1, c)]] = True
        else:
            positive[rng.integers(c)] = True
        rows.append(_row(represent(z[i], dtype), positive, represent(teacher[i], dtype)))
    return rows


def ordinary(loss: Loss, dtype: str, classes: int) -> Case:
    rng = np.random.default_rng(classes)
    z = 3 * rng.standard_normal((6, classes))
    teacher = 3 * rng.standard_normal((6, classes))
    teacher[0] = z[0]
    return Case("ordinary", tuple(_random_rows(loss, dtype, z, rng, teacher)), ({},) * 6)


def ties(loss: Loss, dtype: str) -> Case:
    rng = np.random.default_rng(6)
    z = rng.integers(-2, 3, (6, 6)).astype(np.float64)
    teacher = rng.integers(-2, 3, (6, 6)).astype(np.float64)
    return Case("ties", tuple(_random_rows(loss, dtype, z, rng, teacher)), ({},) * 6)


def offset(loss: Loss, dtype: str) -> Case:
    """Random rows at offsets 1/eps and 2/eps (the teacher unshifted), and two-group rows
    [o, o - 1, ...] with the label first and second."""
    eps = FINFO[dtype][0]
    rng = np.random.default_rng(16)
    rows: list[Row] = []
    tags: list[dict[str, float]] = []
    for o in (1 / eps, 2 / eps):
        z = 3 * rng.standard_normal((2, 10)) + o
        rows += _random_rows(loss, dtype, z, rng, 3 * rng.standard_normal((2, 10)))
        tags += [{"offset": o}] * 2
        if loss.name in ("multilabel_cross_entropy", "multilabel_hinge"):
            continue
        first, other = represent(np.array([o, o - 1]), dtype)
        for label in (0, 1):
            rows.append(
                (Block(first, 1, label == 0, 0.0), Block(other, 1, label == 1, 0.0))
                + (Block(other, 8, False, -1.0),)
            )
            tags.append({"offset": o, "two_group": 1})
    return Case("offset", tuple(rows), tuple(tags))


def extreme(loss: Loss, dtype: str) -> Case:
    """[s max, -s max, -s max] rows: the label first or second; teachers as students."""
    big = FINFO[dtype][2]
    rows = []
    for sign in (1, -1):
        for label in (0, 1):
            first, other = sign * big, -sign * big
            if loss.name in ("multilabel_cross_entropy", "multilabel_hinge"):
                positive = (True, label == 1, False)
            else:
                positive = (label == 0, label == 1, False)
            teacher = (first, other, other) if label == 0 else (other, first, first)
            rows.append(
                tuple(
                    Block(v, 1, p, t)
                    for v, p, t in zip((first, other, other), positive, teacher, strict=True)
                )
            )
    return Case("extreme", tuple(rows), ({"max": big},) * 4)


def far_tail(loss: Loss, dtype: str, classes: int) -> Case:
    """The target (two positives for masks) at 0, the other classes at -gap; distillation:
    student [0, -T gs, ...] and teacher [0, -T gt, ...]."""
    rows: list[Row] = []
    tags: list[dict[str, float]] = []
    if loss.name in ("distillation", "proximal_distillation"):
        t = loss.temperature
        for gs, gt in PAIRS[dtype]:
            s, u = represent(np.array([-t * gs, -t * gt]), dtype)
            rows.append((Block(0.0, 1, False, 0.0), Block(s, classes - 1, False, u)))
            tags.append({"gap": max(gs, gt), "student_gap": gs, "teacher_gap": gt})
        return Case("far_tail", tuple(rows), tuple(tags))
    masks = loss.name in ("multilabel_cross_entropy", "multilabel_hinge")
    lead = 2 if masks else 1
    for gap in GAPS[dtype]:
        (g,) = represent(np.array([-gap]), dtype)
        rows.append((Block(0.0, lead, True), Block(g, classes - lead)))
        tags.append({"gap": gap})
        if loss.name in HINGES:
            # Keep the zero-loss row as well as a violated margin. A target below the
            # leaders exercises the loss and gradient instead of accepting all zeros.
            leaders = max(1, loss.k if loss.k > 0 else classes - lead)
            rest = classes - lead - leaders
            row = (Block(g / 2, lead, True), Block(0.0, leaders))
            rows.append(row + ((Block(g, rest),) if rest else ()))
            tags.append({"gap": gap, "violated": 1})
    return Case("far_tail", tuple(rows), tuple(tags))


def wide(loss: Loss, dtype: str, classes: int) -> Case:
    """Two-group rows at the release widths around vector and tile boundaries: the label at 0
    (or two positives) and the rest at -gap, or the label among them (positives 0 and 1);
    distillation pairs of student and teacher gaps (in units of T)."""
    rows: list[Row] = []
    tags: list[dict[str, float]] = []
    if loss.name in ("distillation", "proximal_distillation"):
        t = loss.temperature
        for gs, gt in ((0, 0), (1, 0), (0, 1), (20, 1), (1, 20)):
            s, u = represent(np.array([-t * gs, -t * gt]), dtype)
            rows.append((Block(0.0, 1, False, 0.0), Block(s, classes - 1, False, u)))
            tags.append({"student_gap": gs, "teacher_gap": gt})
        return Case("wide", tuple(rows), tuple(tags))
    masks = loss.name in ("multilabel_cross_entropy", "multilabel_hinge")
    for gap in (0.0, 1.0, 20.0):
        for label in (0, 1):
            first = Block(0.0, 1, label == 0 or masks)
            rows.append((first, Block(-gap, 1, label == 1), Block(-gap, classes - 2)))
            tags.append({"gap": gap, "label": label})
    return Case("wide", tuple(rows), tuple(tags))


def gaussian(loss: Loss, dtype: str) -> Case:
    """Wide Gaussian row (4097 classes, scale 1), with a random label or positive set, under
    the report's bound entry by entry."""
    rng = np.random.default_rng(4097)
    z = rng.standard_normal((2, 4097))
    teacher = rng.standard_normal((2, 4097))
    return Case("gaussian", tuple(_random_rows(loss, dtype, z, rng, teacher)), ({},) * 2)


def limits(loss: Loss, dtype: str) -> Case:
    """Extreme-spread rows (docs/API.md, Known limits) whose margins exceed working precision:
    [max, max/2, -max/2, 0] (float32 max) and [1e30, -1e30, 0, 1], with the label (or the
    positives) first, then second."""
    big = FINFO["float32"][2]
    rows: list[Row] = []
    tags: list[dict[str, float]] = []
    masks = loss.name in ("multilabel_cross_entropy", "multilabel_hinge")
    for values in ([big, big / 2, -big / 2, 0.0], [1e30, -1e30, 0.0, 1.0]):
        z = represent(np.array(values), dtype)
        for label in (0, 1):
            positive = [j <= label if masks else j == label for j in range(4)]
            rows.append(tuple(Block(float(v), 1, p) for v, p in zip(z, positive, strict=True)))
            tags.append({"first": values[0], "label": label})
    return Case("limits", tuple(rows), tuple(tags))


def capped(dtype: str, classes: int) -> Case:
    """topk_rows.capped_rows: [0, -m, -m - 40, ...] of label 0 at every gap m."""
    rows = []
    for gap in CAPPED_GAPS:
        first, rest = represent(np.array([-gap, -gap - 40]), dtype)
        rows.append((Block(0.0, 1, True), Block(first), Block(rest, classes - 2)))
    return Case("capped", tuple(rows), tuple({"gap": g} for g in CAPPED_GAPS))


def topk_spread(dtype: str) -> Case:
    """Top-k cross-entropy at extreme score spreads: [0, S, -S, -S] (k = 2; loss
    log(1 + 2 sqrt 2) at every S)."""
    rows: list[Row] = []
    tags: list[dict[str, float]] = []
    for s in (1e4, 1e8, 1e12, 1e16, FINFO[dtype][2]):
        (s,) = represent(np.array([s]), dtype)
        rows.append((Block(0.0, 1, True), Block(s), Block(-s, 2)))
        tags.append({"spread": s})
    return Case("topk_spread", tuple(rows), tuple(tags))


def all_capped(dtype: str) -> Case:
    """All-capped top-k cross-entropy: [M/4, H, M, 1, -H], H = 2^200, M = 2^100, k = 4 (float64)."""
    h, m = 2.0**200, 2.0**100
    row = (Block(m / 4, 1, True), Block(h), Block(m), Block(1.0), Block(-h))
    return Case("all_capped", (row,), ({"spread": h},))


def distillation_example(dtype: str) -> Case:
    """The report's CUDA example: 8192 classes, student [0, -110, ...], teacher [0, -220, ...]."""
    s, t = represent(np.array([-110.0, -220.0]), dtype)
    row = (Block(0.0, 1, False, 0.0), Block(s, 8191, False, t))
    return Case("documented", (row,), ({"gap": 220, "student_gap": 110, "teacher_gap": 220},))


def _families(loss: Loss, dtype: str) -> Iterator[Case]:
    yield ordinary(loss, dtype, 2)
    yield ordinary(loss, dtype, 10)
    yield ties(loss, dtype)
    yield offset(loss, dtype)
    yield extreme(loss, dtype)
    for classes in WIDE:
        yield far_tail(loss, dtype, classes)
    for classes in (4097, 32769):
        yield wide(loss, dtype, classes)
    if dtype in ("float32", "float64"):
        yield gaussian(loss, dtype)
    if dtype in ("float32", "float64") and loss.name not in DISTILLATION:
        yield limits(loss, dtype)
    if loss.name == "topk_cross_entropy":
        yield capped(dtype, 10)
        yield capped(dtype, 4096)
        if dtype != "float16":  # spreads to 1e16
            yield topk_spread(dtype)
        if dtype == "float64":
            yield all_capped(dtype)
    if loss.name == "distillation" and loss.temperature == 1:
        yield distillation_example(dtype)


def cases(loss: Loss, dtype: str) -> Iterator[tuple[Loss, Case]]:
    """The families of a loss in dtype with the loss's k resolved: k = -1 is C - 1, which the
    hinge references take only on narrow rows. Top-k families need 1 < k < C (k = 1 is
    cross_entropy); the spread rows are the documented k = 2 and k = 4 examples."""
    for case in _families(loss, dtype):
        if loss.k == 100 and (case.family, case.classes) != ("wide", 4097):
            continue
        k = case.classes - 1 if loss.k == -1 else loss.k
        if loss.name.startswith("topk") and not 1 <= k < case.classes:
            continue
        if loss.k == -1 and (k == 1 or loss.name in HINGES and case.classes > 64):
            continue
        if case.family == "topk_spread" and k != 2 or case.family == "all_capped" and k != 4:
            continue
        if case.family in ("capped", "topk_spread", "all_capped") and k == 1:
            continue
        resolved = replace(loss, k=k)
        smooth = loss.name in HINGES and loss.gamma > 0
        if case.family == "gaussian" and (smooth or loss.name in PROXIMAL):
            continue  # references quadratic in C; proximal losses are covered by their oracle
        if case.family == "wide" and smooth:
            case = _referenced(resolved, case)
        yield resolved, case


def _referenced(loss: Loss, case: Case) -> Case:
    """The rows of a wide smoothed-hinge case whose cut rows give the reference (the cut
    blocks are inactive); without torch the hinge cells skip."""
    if importlib.util.find_spec("torch") is None:
        return case
    kept = [
        (row, tag)
        for row, tag in zip(case.rows, case.tags, strict=True)
        if has_reference(loss, row)
    ]
    return Case(case.family, tuple(r for r, _ in kept), tuple(t for _, t in kept))
