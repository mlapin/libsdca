"""The rule table: one table of rows for the tests of every interface and backend.

The rule for masked classes (docs/API.md, Masked classes): a logit of -inf is the limit of
the loss as that logit goes to -inf.

1. -inf at a class that is not a label removes the class: its gradient is +0.0 and the loss is
   that of the remaining classes. A top-k row left with fewer than k other classes takes the
   limit too: zero for topk_cross_entropy and topk_hinge_alpha; for topk_hinge_beta the
   removed classes add zero margins.
2. Every other class at -inf: loss 0 and a gradient of +0.0.
3. -inf at a label, at a positive class or in the student where the teacher has probability,
   with a class left: loss +inf and the limit of the gradient as those logits go to -inf
   (the other masked classes removed first). cross_entropy: -1 at the label and the softmax
   of the other classes; topk_cross_entropy (k > 1): -1 and the capped softmax
   min(1 / k, exp(z_j - tau)) of the others; the hinges: 1 / k at the k largest others (the
   k-th place split evenly among ties; beta with r < k others: 1 / k at each), smoothed:
   a / gamma for a the projection of the others onto {0 <= a <= gamma / k, sum a = gamma},
   and minus their sum at the label; multilabel_hinge: 1 split among the largest negatives
   and -1 among the masked positives, smoothed: the projection of the negatives onto
   {y >= 0, sum y = gamma} over gamma and -1 / (masked positives) at each; multilabel_cross_entropy: softmax - 1 / |Y|;
   distillation: (q - p) / T; the proximal losses: (pi - p) / T for the limit step on the
   classes left (alpha pi + log pi = s + alpha p - t, sum pi = 1) and -p / T at the masked.
4. The limits that stay finite: zero for a masked label among fewer than k others
   (topk_cross_entropy, topk_hinge_alpha) and for masked positives without a negative left
   (multilabel_hinge).
5. A row of -inf only, NaN and +inf are invalid.
6. Distillation: -inf in the teacher alone is teacher probability zero; -inf in both removes
   the class; a teacher row of -inf only, and NaN or +inf teacher logits, are invalid.
7. Finite zero rows: hinges whose margins are all met, and rows 1024 above the other classes
   (whose exponentials are zero), have loss +0.0 and a gradient of +0.0.
10. Kinks of the nonsmooth hinges (finite rows): ties at the k-th largest margin split its
   place evenly, and a margin (alpha: the mean) of exactly zero counts half, the mean of the
   one-sided subgradients (as optax's multiclass hinge and torch.amax).
8. cross_entropy of probability targets (PROBABILITIES): -inf at a class of target weight
   u_j = w_j ((1 - eps) t_j + eps / C) zero removes the class, its gradient +0.0. Targets
   need not sum to one; the gradient is T softmax - u with T = sum_j u_j.
9. Probability targets with -inf at a class of positive target weight (label smoothing puts
   weight on every class): loss +inf, gradient -u_j there and T softmax elsewhere.

Signed zeros (docs/API.md): every zero of a loss or gradient is +0.0, for either sign of the
upstream derivative, as in PyTorch and JAX. The tests compare the exact entries of a row (its
zeros and infinities) by bytes and the others to a tolerance (mismatches()).

CASES holds the rows: every value is exact in float16 and bfloat16, so one row serves every
dtype. expected(case) gives the float64 loss and gradient of a valid row from references on
its remaining classes: libsdca.loss_and_grad of the finite classes (the kernels of finite
rows), closed forms for cross-entropy and distillation, and the mpmath envelope of
conformance.references for proximal distillation. batches() groups the valid rows of one loss
into arrays; invalid() lists the invalid rows (NaN rows; rule_table.py and test_rule_table.py
say how each interface reports them).

Run as a script, it writes the same rows for the other languages: tests/cpp/masked_cases.h
(C and C++) and rust/tests/masked_cases.txt (test_rule_table.py checks that both are
current).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

INF = math.inf
NAN = math.nan
SINGLE_LABEL = (
    "cross_entropy",
    "topk_cross_entropy",
    "multiclass_hinge",
    "topk_hinge_alpha",
    "topk_hinge_beta",
    "proximal_cross_entropy",
)
MULTILABEL = ("multilabel_cross_entropy", "multilabel_hinge")
DISTILLATION = ("distillation", "proximal_distillation")
HINGES = ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge")
# cross_entropy of probability targets (rules 8, 9): not a loss of libsdca.LOSSES, so valid(),
# invalid() and batches() list its rows only when asked for it by name. Its options are
# label_smoothing and weighted (1: the class weights WEIGHTS), which C and NumPy do not take.
PROBABILITIES = "cross_entropy_probabilities"
WEIGHTS = (1.0, 0.5, 2.0, 0.25, 1.5, 0.0)


class Case(NamedTuple):
    """One row: a loss with its parameters (k, gamma, alpha, temperature: those it takes),
    the logits, the target (a label, the sorted positive classes of a multilabel loss, or
    the teacher's logits) and the rule above that it exercises. valid: rules 1, 2, 3, 4, 6, 7, 10;
    error: "" for an invalid row (NaN outputs; docs/API.md, Errors: the C ABI reports the
    count and the first invalid row), else the message of the argument error of the C ABI."""

    name: str
    loss: str
    params: tuple[tuple[str, float], ...]
    logits: tuple[float, ...]
    target: Any
    rule: int
    valid: bool = True
    error: str = ""

    @property
    def options(self) -> dict[str, Any]:
        return dict(self.params)

    @property
    def classes(self) -> int:
        return len(self.logits)

    @property
    def masked(self) -> tuple[bool, ...]:
        """The classes removed from the row: its -inf logits but those of rule 3 that make
        the loss infinite (a label, a positive class, a student class the teacher keeps)."""
        if self.rule != 3:
            return tuple(v == -INF for v in self.logits)
        if self.loss in DISTILLATION:
            keep = tuple(t > -INF for t in self.target)
        elif self.loss in MULTILABEL:
            keep = tuple(j in self.target for j in range(self.classes))
        else:
            keep = tuple(j == self.target for j in range(self.classes))
        return tuple(v == -INF and not k for v, k in zip(self.logits, keep, strict=True))


def _mask(row: tuple[float, ...], *classes: int) -> tuple[float, ...]:
    return tuple(-INF if j in classes else v for j, v in enumerate(row))


def _wide(classes: int, label: int) -> tuple[float, ...]:
    """A row of distinct eighths in [-4, 4) (at most 64 classes) whose every third class is
    masked (the label never is)."""
    row = [((37 * j) % 64 - 32) / 8 for j in range(classes)]
    return tuple(-INF if j % 3 == 0 and j != label else v for j, v in enumerate(row))


# Six classes with label 2 (positives 2 and 4): distinct margins, none at a kink.
_A = (1.5, -0.75, 0.75, -2.0, 0.5, 2.25)
_B = (0.375, 1.25, 1.5, -0.75, 1.0, 0.25)
_TEACHER = (0.5, 1.0, -0.25, 0.0, 2.0, -1.5)
# Rule 7 with label 2 (positives 2 and 4): every hinge margin met; the label (and the
# teacher's class) 2048 above the other classes.
_MET = (0.5, -7.5, 7.5, -6.0, 5.0, 0.25)
_MET_SETS = (0.5, -7.5, 7.5, -6.0, 7.0, 0.25)
FAR = 2048.0
# Rule 10 with label 2 (positives 2 and 4): three margins tied at the largest (2), and margins
# of exactly zero tied at the largest; for the sets, ties at both ends with a positive margin
# and with a zero one.
_KINK_TIES = (1.0, 1.0, 0.0, 1.0, -2.0, 0.25)
_KINK_ZERO = (-1.0, -2.0, 0.0, -1.0, -3.0, -1.5)
_KINK_SET_TIES = (1.5, 1.5, 2.0, -1.0, 2.0, 0.5)
_KINK_SET_ZERO = (1.0, 1.0, 2.0, -1.0, 2.0, 0.5)
_CONFIDENT = (-FAR, -FAR, 0.0, -FAR, -FAR, -FAR)
# Masked competitors of a row with label 2: what remains of 0, 1, 3, 4, 5.
_MASKS = {
    "one": (3,),
    "two": (0, 5),
    "three": (0, 3, 5),  # two competitors left
    "four": (0, 1, 3, 4),  # one competitor left
}


def _single_label(loss: str, tag: str, **params: float) -> Iterator[Case]:
    options = tuple(sorted(params.items()))
    k = int(params.get("k", 1))
    for base, row in (("a", _A), ("b", _B)):
        for name, classes in _MASKS.items():
            yield Case(f"{tag}-{base}-{name}", loss, options, _mask(row, *classes), 2, 1)
        yield Case(f"{tag}-{base}-alone", loss, options, _mask(row, 0, 1, 3, 4, 5), 2, 2)
    yield Case(f"{tag}-wide", loss, options, _wide(40, 7), 7, 1)
    yield Case(f"{tag}-zero-row", loss, options, _MET if loss in HINGES else _CONFIDENT, 2, 7)
    if loss in HINGES and not params.get("gamma"):
        yield Case(f"{tag}-kink-ties", loss, options, _KINK_TIES, 2, 10)
        yield Case(f"{tag}-kink-zero", loss, options, _KINK_ZERO, 2, 10)
    yield Case(f"{tag}-label", loss, options, _mask(_A, 2), 2, 3)
    yield Case(f"{tag}-label-one", loss, options, _mask(_A, 2, 3), 2, 3)
    yield Case(f"{tag}-label-ties", loss, options, (0.5, 1.0, -INF, 1.0, -INF, 0.5), 2, 3)
    # Two others left: fewer than k = 3.
    few = 4 if k > 2 and loss in ("topk_cross_entropy", "topk_hinge_alpha") else 3
    yield Case(f"{tag}-label-few", loss, options, _mask(_A, 0, 1, 2, 3), 2, few)
    yield Case(f"{tag}-all", loss, options, (-INF,) * 6, 2, 5, False)
    yield Case(f"{tag}-nan", loss, options, (1.5, NAN, 0.75, -INF, 0.5, 2.25), 2, 5, False)
    yield Case(f"{tag}-inf", loss, options, (1.5, INF, 0.75, -INF, 0.5, 2.25), 2, 5, False)


def _multilabel(loss: str, tag: str, **params: float) -> Iterator[Case]:
    options = tuple(sorted(params.items()))
    positive = (2, 4)
    for name, classes in (("one", (3,)), ("top", (5,)), ("three", (0, 3, 5))):
        yield Case(f"{tag}-{name}", loss, options, _mask(_A, *classes), positive, 1)
    yield Case(f"{tag}-b-two", loss, options, _mask(_B, 0, 3), (1, 2, 4), 1)
    # No class outside the set is left: the hinge is zero, the entropy that of the set.
    alone = 2 if loss == "multilabel_hinge" else 1
    yield Case(f"{tag}-alone", loss, options, _mask(_A, 0, 1, 3, 5), positive, alone)
    yield Case(f"{tag}-single", loss, options, _mask(_A, 0, 1, 3, 4, 5), (2,), 2)
    if loss in HINGES:
        yield Case(f"{tag}-zero-row", loss, options, _MET_SETS, positive, 7)
        if not params.get("gamma"):
            yield Case(f"{tag}-kink-ties", loss, options, _KINK_SET_TIES, positive, 10)
            yield Case(f"{tag}-kink-zero", loss, options, _KINK_SET_ZERO, positive, 10)
    else:
        yield Case(f"{tag}-zero-row", loss, options, _CONFIDENT, (2,), 7)
    yield Case(f"{tag}-positive", loss, options, _mask(_A, 3, 4), positive, 3)
    yield Case(f"{tag}-positives", loss, options, _mask(_A, 2, 4, 5), positive, 3)
    # A masked positive without a negative left: zero for the hinge.
    alone = 4 if loss == "multilabel_hinge" else 3
    yield Case(f"{tag}-positive-alone", loss, options, _mask(_A, 0, 1, 3, 4, 5), positive, alone)
    yield Case(f"{tag}-all", loss, options, (-INF,) * 6, positive, 5, False)
    nan = (1.5, NAN, 0.75, -INF, 0.5, 2.25)
    yield Case(f"{tag}-nan", loss, options, nan, positive, 5, False)


def _distillation(loss: str, tag: str, **params: float) -> Iterator[Case]:
    options = tuple(sorted(params.items()))

    def case(name: str, both: tuple[int, ...], teacher: tuple[int, ...], rule: int = 6) -> Case:
        return Case(
            f"{tag}-{name}", loss, options, _mask(_A, *both), _mask(_TEACHER, *both, *teacher), rule
        )

    yield case("both-one", (3,), ())
    yield case("both-three", (0, 3, 5), ())
    yield case("teacher-one", (), (1,))
    yield case("teacher-two", (), (0, 4))
    yield case("mixed", (3,), (1, 5))
    yield case("pair", (0, 1, 3, 5), ())
    yield case("alone", (0, 1, 3, 4, 5), (), 2)
    yield Case(f"{tag}-zero-row", loss, options, _CONFIDENT, _CONFIDENT, 7)
    student = _mask(_A, 3)
    yield Case(f"{tag}-student", loss, options, student, _TEACHER, 3)
    yield Case(f"{tag}-student-two", loss, options, _mask(_A, 0, 3, 5), _mask(_TEACHER, 5), 3)
    yield Case(f"{tag}-teacher-all", loss, options, _A, (-INF,) * 6, 6, False)
    yield Case(f"{tag}-all", loss, options, (-INF,) * 6, (-INF,) * 6, 5, False)
    nan = (1.5, NAN, 0.75, -INF, 0.5, 2.25)
    yield Case(f"{tag}-nan", loss, options, nan, _TEACHER, 5, False)
    infinite = _mask(_TEACHER, 3)[:5] + (INF,)
    yield Case(f"{tag}-teacher-inf", loss, options, student, infinite, 5, False)


# Target probabilities of six classes, zero at classes 3 and 5; and a row that sums to two.
_P = (0.125, 0.25, 0.375, 0.0, 0.25, 0.0)
_Q = (0.5, 0.25, 0.0, 0.0, 1.0, 0.25)
_ONE = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def _probabilities(tag: str, **params: float) -> Iterator[Case]:
    options = tuple(sorted(params.items()))
    smooth = params.get("label_smoothing", 0.0) > 0
    loss = PROBABILITIES

    def masked(rule: int) -> int:
        return 9 if smooth else rule

    yield Case(f"{tag}-a", loss, options, _A, _P, 8)
    yield Case(f"{tag}-b", loss, options, _B, _Q, 8)
    yield Case(f"{tag}-zero-masked", loss, options, _mask(_A, 3, 5), _P, masked(8))
    yield Case(f"{tag}-zero-masked-one", loss, options, _mask(_B, 3), _Q, masked(8))
    yield Case(f"{tag}-positive-masked", loss, options, _mask(_A, 2), _P, 9)
    yield Case(f"{tag}-positive-masked-two", loss, options, _mask(_B, 0, 4), _Q, 9)
    yield Case(f"{tag}-alone", loss, options, _mask(_A, 0, 1, 3, 4, 5), _ONE, masked(2))
    yield Case(f"{tag}-wide", loss, options, _wide(40, 7), tuple(
        (j % 3 != 0 or j == 7) * ((j % 5) + 1) / 16 for j in range(40)), masked(8))  # fmt: skip
    if params.get("weighted"):
        # Weight zero at class 5: its target weight is zero although its target is not.
        yield Case(f"{tag}-weight-zero-masked", loss, options, _mask(_B, 5), _Q, masked(8))
    if not smooth:
        yield Case(f"{tag}-zero-row", loss, options, _CONFIDENT, _ONE, 7)
        yield Case(f"{tag}-zero-targets", loss, options, _mask(_A, 3), (0.0,) * 6, 7)
    if params:
        return
    yield Case(f"{tag}-all", loss, options, (-INF,) * 6, _P, 5, False)
    yield Case(f"{tag}-nan", loss, options, (1.5, NAN, 0.75, -INF, 0.5, 2.25), _P, 5, False)
    yield Case(f"{tag}-inf", loss, options, (1.5, INF, 0.75, -INF, 0.5, 2.25), _P, 5, False)
    # Invalid target probabilities are invalid rows too (loss evaluations take them as given).
    negative = (0.5, -0.25, 0.75, 0.0, 0.0, 0.0)
    yield Case(f"{tag}-negative", loss, options, _A, negative, 5, False)
    yield Case(f"{tag}-nan-target", loss, options, _A, (0.5, NAN, 0.5, 0, 0, 0), 5, False)


def _cases() -> Iterator[Case]:
    yield from _single_label("cross_entropy", "ce")
    # Rows of two and three classes.
    for name, row, label, rule in (
        ("ce-zero", (0.0, -INF, -INF), 0, 2),
        ("ce-small", (0.0, -5.0, -INF), 0, 1),
        ("ce-large", (0.0, 2.0, -INF), 0, 1),
        ("ce-masked-label", (-INF, 0.0, 1.0), 0, 3),
        ("ce-two-classes", (0.25, -INF), 0, 2),
        ("ce-two-label", (-INF, 0.25), 0, 3),
    ):
        yield Case(name, "cross_entropy", (), row, label, rule)
    yield from _single_label("topk_cross_entropy", "top1", k=1)
    yield from _single_label("topk_cross_entropy", "top2", k=2)
    yield from _single_label("topk_cross_entropy", "top3", k=3)
    yield from _single_label("multiclass_hinge", "hinge", gamma=0.0)
    yield from _single_label("multiclass_hinge", "hinge-smooth", gamma=0.5)
    for loss, tag in (("topk_hinge_alpha", "alpha"), ("topk_hinge_beta", "beta")):
        yield from _single_label(loss, f"{tag}2", k=2, gamma=0.0)
        yield from _single_label(loss, f"{tag}2-smooth", k=2, gamma=0.5)
        yield from _single_label(loss, f"{tag}3-smooth", k=3, gamma=2.0)
    yield from _single_label("proximal_cross_entropy", "prox", alpha=0.5)
    yield from _single_label("proximal_cross_entropy", "prox-wide-step", alpha=32.0)
    yield from _multilabel("multilabel_cross_entropy", "sets")
    yield from _multilabel("multilabel_hinge", "sets-hinge", gamma=0.0)
    yield from _multilabel("multilabel_hinge", "sets-hinge-smooth", gamma=0.5)
    yield from _distillation("distillation", "distill", temperature=1.0)
    yield from _distillation("distillation", "distill-warm", temperature=2.0)
    yield from _distillation("proximal_distillation", "prox-distill", alpha=0.5, temperature=2.0)
    yield from _probabilities("probs")
    yield from _probabilities("probs-smooth", label_smoothing=0.125)
    yield from _probabilities("probs-weighted", weighted=1.0)
    yield from _probabilities("probs-weighted-smooth", label_smoothing=0.25, weighted=1.0)


CASES: tuple[Case, ...] = tuple(_cases())


def _of(case: Case, loss: str | None) -> bool:
    """Whether a row is of loss: of every loss but PROBABILITIES for None."""
    return case.loss == loss if loss else case.loss != PROBABILITIES


def valid(loss: str | None = None) -> list[Case]:
    return [c for c in CASES if c.valid and _of(c, loss)]


def invalid(loss: str | None = None) -> list[Case]:
    return [c for c in CASES if not c.valid and _of(c, loss)]


def plain(case: Case) -> bool:
    """Whether every interface takes the row: no label smoothing or class weights."""
    return case.loss != PROBABILITIES or not case.params


def positive_mask(case: Case) -> np.ndarray:
    """The boolean label mask (classes,) of a label or multilabel row (none for the other rows)."""
    mask = np.zeros(case.classes, bool)
    if case.loss in DISTILLATION or case.loss == PROBABILITIES:
        return mask
    mask[list(case.target) if case.loss in MULTILABEL else case.target] = True
    return mask


def target_weights(case: Case) -> np.ndarray:
    """u = w ((1 - eps) t + eps / C) of a probability row, in float64."""
    u = np.array(case.target, np.float64)
    eps = case.options.get("label_smoothing", 0.0)
    if eps:
        u = u * (1 - eps) + eps / case.classes
    if case.options.get("weighted"):
        u = u * np.array(WEIGHTS[: case.classes] + (1.0,) * (case.classes - len(WEIGHTS)))
    return u


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _infinite(case: Case) -> np.ndarray:
    """The masked classes of rule 3 that make the loss infinite: a label, the positive
    classes, the classes the student masks where the teacher keeps them."""
    z = np.array(case.logits, np.float64)
    if case.rule != 3:
        return np.zeros(len(z), bool)
    if case.loss in DISTILLATION:
        keep = np.isfinite(np.array(case.target, np.float64))
    else:
        keep = positive_mask(case)
    return ~np.isfinite(z) & keep


def _reference_row(case: Case) -> tuple[float, np.ndarray]:
    """conformance.references (mpmath) of the row with finite stand-ins for its -inf scores,
    deep enough that the result is the limit's to float64: the classes that make the loss
    infinite gap below the lowest finite score, the removed ones (masked first) 2 gap below,
    and a teacher's -inf gap below its lowest. Hinge margins then pass every bound, and
    exponentials of -gap / T (less 2 gamma) are below float64's subnormals."""
    from conformance import references
    from libsdca._limits import limits as _limits

    z, options = np.array(case.logits, np.float64), case.options
    finite, infinite = np.isfinite(z), _infinite(case)
    temperature = options.get("temperature", 1.0)
    step = options.get("gamma", 0.0) + options.get("alpha", 0.0)
    gap = (1 - _limits("float64").log_denorm_min + 2 * step) * max(temperature, 1.0)
    low = z[finite].min()
    zs = np.where(finite, z, np.where(infinite, low - gap, low - 2 * gap))
    teacher = target = None
    if case.loss in DISTILLATION:
        t = np.array(case.target, np.float64)
        teacher = np.where(np.isfinite(t), t, t[np.isfinite(t)].min() - gap)[None]
    elif case.loss in MULTILABEL:
        target = positive_mask(case)[None]
    else:
        target = np.array([case.target])
    loss = references.named(
        case.loss,
        k=int(options.get("k", 1)),
        gamma=options.get("gamma", 0.0),
        temperature=temperature,
        alpha=options.get("alpha", 1.0),
    )
    value, gradient = references.evaluate(loss, zs[None], target, teacher)
    gradient = np.where(finite | infinite, gradient[0], 0.0)
    return (INF if infinite.any() else float(value[0])), gradient


def _probability_row(case: Case, z: np.ndarray) -> tuple[float, np.ndarray]:
    u = target_weights(case)
    keep = np.isfinite(z)
    top = z[keep].max()
    lse = top + math.log(math.fsum(np.exp(z[keep] - top)))
    p = np.zeros(len(z))
    p[keep] = np.exp(z[keep] - lse)
    gradient = math.fsum(u) * p - u
    if (u[~keep] > 0).any():
        return INF, gradient
    return math.fsum(u[keep] * (lse - z[keep])), gradient


@functools.cache
def expected(case: Case) -> tuple[float, np.ndarray]:
    """The float64 loss and gradient (classes,) of a valid row."""
    if not case.valid:
        raise ValueError(f"{case.name} is an invalid row")
    if case.loss == PROBABILITIES:
        value, gradient = _probability_row(case, np.array(case.logits, np.float64))
    elif case.rule == 4:  # the limit is zero for every finite masked score
        value, gradient = 0.0, np.zeros(case.classes)
    else:
        value, gradient = _reference_row(case)
    gradient = gradient + 0.0  # +0.0 in the entries of removed classes
    gradient.setflags(write=False)
    return value + 0.0, gradient


def differs(actual: np.ndarray, want: np.ndarray, tol: float) -> np.ndarray:
    """Where actual is not want: by bytes where want is exact (a zero, with its sign, or an
    infinity), else beyond tol relative to 1 + its size (NaN always)."""
    exact = (want == 0) | np.isinf(want)
    same = (actual == want) & (np.signbit(actual) == np.signbit(want))
    with np.errstate(invalid="ignore"):
        near = np.abs(actual - want) <= tol * (1 + np.abs(want))
    return ~np.where(exact, same, near)


def mismatches(
    case: Case, value: Any, gradient: Any, tol: float, upstream: float = 1.0
) -> list[str]:
    """What differs between a computed loss (None: not checked) and gradient, of any float
    dtype, and the row's expected ones, the gradient times the upstream derivative: the exact
    entries by bytes (every zero +0.0), the others within tol (relative to 1 + their size)."""
    want_value, want = expected(case)
    want = want * upstream + 0.0
    g = np.asarray(gradient, np.float64)
    found = []
    v = np.asarray(value if value is not None else want_value, np.float64).reshape(1)
    if differs(v, np.array([want_value]), tol).any():
        found.append(f"{case.name}: loss {v[0]!r}, not {want_value!r}")
    bad = np.flatnonzero(differs(g, want, tol))
    if len(bad):
        entries = ", ".join(f"{j}: {g[j]!r} not {want[j]!r}" for j in bad[:4])
        found.append(f"{case.name} (upstream {upstream}): gradient {entries}")
    return found


class Batch(NamedTuple):
    """The valid rows of one loss, parameter set and width as float64 arrays: logits (n,
    classes), the target (labels (n,), a boolean mask, teacher logits or target probabilities
    (n, classes)), the
    expected values (n,) and gradients (n, classes), and the removed classes."""

    loss: str
    options: dict[str, Any]
    cases: tuple[Case, ...]
    logits: np.ndarray
    target: np.ndarray
    values: np.ndarray
    gradients: np.ndarray
    masked: np.ndarray


def batches(loss: str | None = None, *, limits: bool = True) -> list[Batch]:
    """The valid rows by loss, parameters and width. limits=False leaves out the rows of
    rules 3 and 9 (an infinite loss), which a mean or sum would make infinite."""
    groups: dict[tuple[Any, ...], list[Case]] = {}
    for case in valid(loss):
        if limits or case.rule not in (3, 9):
            groups.setdefault((case.loss, case.params, case.classes), []).append(case)
    found = []
    for (name, _, _), rows in groups.items():
        logits = np.array([c.logits for c in rows], np.float64)
        if name in DISTILLATION or name == PROBABILITIES:
            target = np.array([c.target for c in rows], np.float64)
        elif name in MULTILABEL:
            target = np.array([positive_mask(c) for c in rows])
        else:
            target = np.array([c.target for c in rows], np.int64)
        results = [expected(c) for c in rows]
        found.append(
            Batch(
                name,
                rows[0].options,
                tuple(rows),
                logits,
                target,
                np.array([v for v, _ in results]),
                np.array([g for _, g in results]),
                np.array([c.masked for c in rows]),
            )
        )
    return found


# enum sdca_loss_kind (include/sdca/c_api.h), which is sdca::loss_kind.
_C_LOSS = {
    "cross_entropy": 0,
    "topk_cross_entropy": 1,
    "multiclass_hinge": 2,
    "topk_hinge_alpha": 3,
    "topk_hinge_beta": 4,
    "multilabel_hinge": 5,
    "multilabel_cross_entropy": 6,
    "proximal_cross_entropy": 7,
    "distillation": 8,
    "proximal_distillation": 9,
    PROBABILITIES: 0,
}
HEADER = Path(__file__).resolve().parents[1] / "cpp" / "masked_cases.h"


def _c_number(value: float) -> str:
    if math.isnan(value):
        return "SDCA_MASKED_NAN"
    if math.isinf(value):
        return "HUGE_VAL" if value > 0 else "-HUGE_VAL"
    return repr(float(value))


def _c_row(values: Any) -> str:
    return "{" + ", ".join(_c_number(float(v)) for v in values) + "}"


def header() -> str:
    """tests/cpp/masked_cases.h: CASES for C and C++ (rows padded to the widest), the
    probability rows without options among them."""
    rows = [c for c in CASES if plain(c)]
    width = max(c.classes for c in rows)
    lines = [
        "/* Masked classes (logits of -inf): the rows of tests/python/masked_cases.py with their",
        " * expected float64 losses and gradients. Written by that script; do not edit. */",
        "#ifndef SDCA_TESTS_MASKED_CASES_H",
        "#define SDCA_TESTS_MASKED_CASES_H",
        "",
        "#include <math.h>",
        "",
        "/* clang-format off */",
        f"#define SDCA_MASKED_WIDTH {width}",
        "/* Infinity is HUGE_VAL, a double. */",
        "#if defined(__GNUC__)",
        '#define SDCA_MASKED_NAN __builtin_nan("")',
        "#else",
        "#define SDCA_MASKED_NAN (HUGE_VAL - HUGE_VAL)",
        "#endif",
        "",
        "/* loss: enum sdca_loss_kind. target: the label, the 0/1 mask of a multilabel loss, or",
        " * the teacher's logits; probabilities: cross_entropy of the target probabilities in",
        " * teacher (the C ABI's; not the C++ loss functions'). A row that is not valid is an",
        ' * invalid row (error "": NaN outputs) or an argument error of the C ABI (error). */',
        "typedef struct sdca_masked_case {",
        "  const char* name;",
        "  int loss, rule, valid, classes, label, probabilities;",
        "  long k;",
        "  double gamma, alpha, temperature;",
        "  double logits[SDCA_MASKED_WIDTH];",
        "  unsigned char positive[SDCA_MASKED_WIDTH];",
        "  double teacher[SDCA_MASKED_WIDTH];",
        "  double value;",
        "  double gradient[SDCA_MASKED_WIDTH];",
        "  const char* error;",
        "} sdca_masked_case;",
        "",
        "static const sdca_masked_case sdca_masked_cases[] = {",
    ]
    for case in rows:
        options = case.options
        zeros = (0.0,) * case.classes
        value, gradient = expected(case) if case.valid else (0.0, zeros)
        label = case.target if case.loss in SINGLE_LABEL else -1
        positive = positive_mask(case)
        soft = case.loss == PROBABILITIES
        teacher = case.target if case.loss in DISTILLATION or soft else zeros
        lines.append(
            f'    {{"{case.name}", {_C_LOSS[case.loss]}, {case.rule}, {int(case.valid)}, '
            f"{case.classes}, {label}, {int(soft)}, {int(options.get('k', 0))}, "
            f"{_c_number(options.get('gamma', 0.0))}, {_c_number(options.get('alpha', 1.0))}, "
            f"{_c_number(options.get('temperature', 1.0))},"
        )
        lines.append(f"     {_c_row(case.logits)},")
        lines.append("     {" + ", ".join(str(int(v)) for v in positive) + "},")
        lines.append(f"     {_c_row(teacher)},")
        lines.append(f"     {_c_number(value)},")
        lines.append(f"     {_c_row(gradient)},")
        lines.append(f'     "{case.error}"}},')
    lines += [
        "};",
        "",
        "#define SDCA_MASKED_CASES (sizeof sdca_masked_cases / sizeof sdca_masked_cases[0])",
        "/* clang-format on */",
        "",
        "#endif",
        "",
    ]
    return "\n".join(lines)


RUST = Path(__file__).resolve().parents[2] / "rust" / "tests" / "masked_cases.txt"


def _words(values: Any) -> str:
    return " ".join(repr(float(v)) for v in values)


def text() -> str:
    """rust/tests/masked_cases.txt: one row of CASES per line, fields separated by " | ":
    name, loss (enum sdca_loss_kind), rule, valid, label (-1: none), k, gamma, alpha,
    temperature; the logits; the 0/1 positive mask; the teacher's logits; the expected loss;
    the expected gradient; the C ABI's argument error of an invalid row ("" for a NaN row).
    Numbers are float64 (inf, -inf, nan).
    The Rust crate evaluates no probability targets: their rows are left out."""
    lines = []
    for case in CASES:
        if case.loss == PROBABILITIES:
            continue
        options = case.options
        zeros = (0.0,) * case.classes
        value, gradient = expected(case) if case.valid else (0.0, zeros)
        label = case.target if case.loss in SINGLE_LABEL else -1
        positive = positive_mask(case)
        teacher = case.target if case.loss in DISTILLATION else zeros
        head = (
            f"{case.name} {_C_LOSS[case.loss]} {case.rule} {int(case.valid)} {label} "
            f"{int(options.get('k', 0))} {options.get('gamma', 0.0)!r} "
            f"{options.get('alpha', 1.0)!r} {options.get('temperature', 1.0)!r}"
        )
        fields = [head, _words(case.logits), " ".join(str(int(v)) for v in positive)]
        fields += [_words(teacher), repr(float(value)), _words(gradient), case.error]
        lines.append(" | ".join(fields))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    HEADER.write_text(header())
    RUST.write_text(text())
    print(f"{HEADER} and {RUST}: {len(CASES)} rows")


def bfloat_multilabel_positive_residual() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two rows straddle a zero gradient, another masks a positive, and two have a dominant score."""
    classes, zeros, positives = 16401, 1339, 6880
    z = np.full((5, classes), -1.0, np.float32)
    z[:, :zeros] = 0
    mask = np.zeros_like(z, dtype=bool)
    mask[:, zeros : zeros + positives] = True
    mask[1, 0], mask[1, zeros] = True, False
    z[2, zeros] = -math.inf
    z[3:] = 0
    z[3:, 0] = 15
    mask[3:] = False
    mask[3, :2], mask[4, 0] = True, True
    # The positive residual in row 1 is smaller than 3e-12 before applying the cotangent.
    part = zeros + (classes - zeros) / math.e
    peak = 1.0 / (1.0 + (classes - 1) * math.exp(-15.0))
    reference = np.array(
        [1 / part, 1 / part - 1 / positives, 1 / (part - 1 / math.e), peak - 0.5, peak - 1]
    )
    return z, mask, np.array([1e20, -1e20, 1.0, 1e20, -1e20], np.float32), reference


def amplified_multilabel_tail(
    classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """BF16-exact logits with representable scaled gradients below the FP32 probability floor."""
    gaps = np.tile([100, 126, 128, 130, 175, 190, 64, 0], 8)
    scores = np.zeros((len(gaps), classes), dtype=np.float32)
    scores[:, 0], scores[:, 1] = gaps / 2, -gaps / 2
    scores[7::8, 1] = -np.inf
    target = np.zeros_like(scores, dtype=bool)
    target[:, 2] = True
    upstream = np.tile(
        np.asarray([1e20, -1e20, 1e20, -1e20, 3e37, -3e37, 1.0, -1e20], dtype=np.float32), 8
    )
    # This gap-175 entry underflows even FP32, unlike the gap-190 entry at upstream -3e37.
    upstream[-4] = -(2.0**56)
    high = scores[:, 0].astype(np.float64)
    tail = np.exp(scores[:, 1].astype(np.float64) - high)
    partition = 1 + tail + (classes - 2) * np.exp(-high)
    return scores, target, upstream, (upstream.astype(np.float64) * tail) / partition
