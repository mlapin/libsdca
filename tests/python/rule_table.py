"""The rule table: one table of edge-case rows with their expected outputs, which every backend
is tested against (test_rule_table.py).

Each rule is written once, from the decided behaviour (docs/API.md: Masked classes, Errors,
Parameter domains; the docstrings of libsdca.torch and libsdca.jax). ROWS holds the rows of a
loss evaluation, ERRORS the calls every interface must reject with one message.

CANONICAL holds the one text of each error (PyTorch's wording where PyTorch has a message for
the condition, else the C ABI's and Rust's).

A Row expects a loss and a gradient: exact where the rule fixes them (0 with its sign, +inf,
NaN, the value of a hinge at all-zero scores), else references within a tolerance
(conformance.references, and for the masked-class rules masked_cases.expected: the loss on the
remaining classes). Rules (Row.rule):

- masked_removed, masked_alone, masked_label_limit, masked_finite_limit, teacher_masked,
  probabilities_masked, zero_loss: the rules of masked_cases.py (its rows, reused), -inf at
  classes without target mass: +0.0 gradients there, the loss as if the class were removed;
  at a label, a positive or a student class with teacher mass: the limit (+inf, or zero
  where no other class competes); hinge_kinks: ties and zero margins of the nonsmooth
  hinges (the mean of the one-sided subgradients);
- zero_logits: rows of zero scores;
- signed_zero_inputs: scores of -0.0 give the bytes of the same row of +0.0 (Row.same_as);
- negative_upstream: every zero of a gradient is +0.0 for either sign of the upstream factor;
- storage_negative_zero: a negative entry that rounds to zero only in a 16-bit storage stays
  -0.0; work_underflow: entries that underflow in the work dtype are +0.0;
- invalid: NaN or +inf scores, every class masked, an empty label set, a multilabel hinge row
  of every class: NaN rows (C: SDCA_INVALID_ROWS and the count; NumPy: one InvalidRowsWarning
  per call; Rust: Ok and a report; PyTorch and JAX: NaN rows silently; check_rows raises before
  writing). Invalid target probabilities are NaN rows in every interface;
- ignore_index: ignored rows have loss and gradient +0.0, whatever their scores and weights;
- zero_weight: a class weight of 0 at the label gives +0.0; at a masked label NaN;
- invalid_weight: a NaN or negative class weight makes every row not ignored NaN in PyTorch and
  JAX, and is an argument error of the C ABI, NumPy and Rust (Row.native_error);
- label_smoothing: eps = 1 (uniform targets), tiny eps, and eps > 0 at a masked class (+inf).

The rows of FLOAT32_RULES run in float32 storage too; the others in float64 (and the storage
rules in their 16-bit and float32 dtypes). Rows of rules that change in open work are
pending (Row.pending, Error.pending) and left out unless their flag in pending.py is set:
pending.LABEL_ERRORS (labels out of range: errors, not NaN rows).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import masked_cases as M
import numpy as np
from conformance import references as R
from pending import ENABLED

INF, NAN = math.inf, math.nan
IGNORE = -100


@dataclasses.dataclass(frozen=True)
class Row:
    """One row of a loss evaluation and its expected loss and gradient."""

    name: str
    rule: str
    loss: str  # masked_cases names: topk_hinge_alpha/beta, cross_entropy_probabilities
    params: tuple[tuple[str, float], ...]  # k, gamma (hinges), alpha (proximal), temperature
    logits: tuple[float, ...]
    target: Any  # label, sorted positives, teacher logits or target probabilities
    value: float
    gradient: tuple[float, ...] | None
    weight: tuple[float, ...] | None = None
    ignore_index: int | None = None
    label_smoothing: float = 0.0
    dtype: str = "float64"
    upstream: float = 1.0
    native_error: str = ""  # the message of the C ABI, NumPy and Rust, which reject the call
    same_as: str = ""  # a row of the same batch whose outputs these must equal, by bytes
    pending: str = ""  # a key of pending.ENABLED: the row runs once it is set
    until: str = ""  # a key of pending.ENABLED: the row runs until it is set
    # The gradient entries the rule fixes, compared by bytes (None: its zeros, infinities and
    # NaN); the others are references, within a tolerance.
    exact: tuple[bool, ...] | None = None

    @property
    def options(self) -> dict[str, float]:
        return dict(self.params)

    @property
    def classes(self) -> int:
        return len(self.logits)

    @property
    def valid(self) -> bool:
        return not math.isnan(self.value)

    @property
    def has_options(self) -> bool:
        """Class weights, an ignored label or label smoothing (sdca/loss.h takes none)."""
        smooth = self.label_smoothing != 0
        return self.weight is not None or self.ignore_index is not None or smooth

    def options_of(self) -> dict[str, Any]:
        """The weight, label_smoothing and ignore_index given, as the C ABI and NumPy take
        them."""
        options: dict[str, Any] = {}
        if self.weight is not None:
            options["weight"] = np.array(self.weight)
        if self.label_smoothing:
            options["label_smoothing"] = self.label_smoothing
        if self.ignore_index is not None:
            options["ignore_index"] = self.ignore_index
        return options


@dataclasses.dataclass(frozen=True)
class Error:
    """A call every interface that can make it rejects: the loss, its parameters, the shape of
    the scores and a change of the targets or options."""

    name: str
    loss: str
    params: tuple[tuple[str, float], ...] = ()
    shape: tuple[int, ...] = (2, 4)
    target: str = ""  # labels_length, float_labels, teacher_shape, mask_shape, probs_shape, ...
    options: tuple[tuple[str, Any], ...] = ()  # weight, label_smoothing (frameworks)
    pending: str = ""


# --- references ------------------------------------------------------------------------------


def _reference(
    loss: str,
    logits: tuple[float, ...],
    target: Any = None,
    teacher: Any = None,
    temperature: float = 1.0,
    k: int = 1,
) -> tuple[float, tuple[float, ...]]:
    """conformance.references of one row: its loss and gradient, each rounded once to float64.
    target: a label, a positive mask or target masses of cross-entropy (as a row)."""
    z = np.array([logits], np.float64)
    t = None if teacher is None else np.array([teacher], np.float64)
    values, gradients = R.evaluate(
        R.named(loss, temperature=temperature, k=k), z, target, teacher=t
    )
    return float(values[0]), tuple(float(g) for g in gradients[0])


def _targets(row_target: Any, classes: int, weight: Any, eps: float) -> np.ndarray:
    """u = w ((1 - eps) t + eps / C) of a label (one-hot t) or target probabilities."""
    if isinstance(row_target, int):
        t = np.zeros(classes)
        t[row_target] = 1.0
    else:
        t = np.array(row_target, np.float64)
    u = t * (1 - eps) + eps / classes if eps else t
    return u * np.array(weight) if weight is not None else u


def _expected_soft(
    loss: str, logits: tuple[float, ...], target: Any, weight: Any = None, eps: float = 0.0
) -> tuple[float, tuple[float, ...]]:
    """The cross-entropy of a label or of target probabilities as target masses (PyTorch's
    weights and label smoothing), and the multilabel cross-entropy of a label set."""
    if loss == "multilabel_cross_entropy":
        mask = np.zeros((1, len(logits)), bool)
        mask[0, list(target)] = True
        return _reference(loss, logits, mask)
    return _reference("cross_entropy", logits, _targets(target, len(logits), weight, eps)[None])


def _distillation(student: tuple[float, ...], teacher: tuple[float, ...], temperature: float):
    return _reference("distillation", student, teacher=teacher, temperature=temperature)


# --- rows ------------------------------------------------------------------------------------

_MASKED_RULES = {
    1: "masked_removed",
    2: "masked_alone",
    3: "masked_label_limit",
    4: "masked_finite_limit",
    6: "teacher_masked",
    7: "zero_loss",
    8: "probabilities_masked",
    9: "probabilities_masked",
    10: "hinge_kinks",
}


def _from_masked(case: M.Case) -> Row:
    """A row of masked_cases.py: valid rows with their references and NaN outputs for invalid rows."""
    weighted = case.loss == M.PROBABILITIES and case.options.get("weighted")
    params = tuple((k, v) for k, v in case.params if k not in ("weighted", "label_smoothing"))
    common: dict[str, Any] = dict(
        params=params,
        logits=case.logits,
        target=case.target,
        # masked_cases.target_weights: WEIGHTS, then 1 for the classes beyond them.
        weight=M.WEIGHTS[: case.classes] + (1.0,) * (case.classes - len(M.WEIGHTS))
        if weighted
        else None,
        label_smoothing=case.options.get("label_smoothing", 0.0),
    )
    if case.valid:
        value, gradient = M.expected(case)
        rule = _MASKED_RULES[case.rule]
        # Fixed: the removed classes, every entry of rules 2, 4 and 7, the infinite limits.
        exact = tuple(
            bool(m or case.rule in (2, 4, 7) or not math.isfinite(g))
            for m, g in zip(case.masked, gradient, strict=True)
        )
        return Row(
            case.name, rule, case.loss, value=value, gradient=tuple(gradient), exact=exact,
            **common,
        )  # fmt: skip
    # Invalid target probabilities and NaN or +inf teacher logits are invalid rows, as NaN or
    # +inf scores (loss evaluations take probability targets as given).
    nan = (NAN,) * case.classes
    return Row(case.name, "invalid", case.loss, value=NAN, gradient=nan, **common)  # fmt: skip


_Z0 = (0.0, 0.0, 0.0, 0.0)
_NEG0 = (-0.0, 0.0, -0.0, -0.0)
_POS = (1, 3)


def _zero_rows() -> list[Row]:
    rows = []
    for loss, params, target in (
        ("cross_entropy", (), 1),
        ("multilabel_cross_entropy", (), _POS),
        (M.PROBABILITIES, (), (0.25, 0.5, 0.0, 0.25)),
    ):
        value, gradient = _expected_soft(loss, _Z0, target)
        rows.append(Row(f"zero-{loss}", "zero_logits", loss, params, _Z0, target, value, gradient))
        rows.append(
            Row(f"negzero-{loss}", "signed_zero_inputs", loss, params, _NEG0, target, value,
                gradient, same_as=f"zero-{loss}")
        )  # fmt: skip
    # Nonsmooth hinges at zero scores: every margin is 1, tied (the place splits evenly).
    for loss, params, target in (
        ("multiclass_hinge", (("gamma", 0.0),), 1),
        ("topk_hinge_alpha", (("gamma", 0.0), ("k", 2)), 1),
        ("topk_hinge_beta", (("gamma", 0.0), ("k", 2)), 1),
        ("multilabel_hinge", (("gamma", 0.0),), _POS),
    ):
        k = int(dict(params).get("k", 1))
        if isinstance(target, tuple):
            labels = np.zeros((1, len(_Z0)), bool)
            labels[0, list(target)] = True
        else:
            labels = np.array([target])
        value, gradient = _reference(loss, _Z0, labels, k=k)
        rows.append(Row(f"zero-{loss}", "zero_logits", loss, params, _Z0, target, value,
                        gradient))  # fmt: skip
        rows.append(
            Row(f"negzero-{loss}", "signed_zero_inputs", loss, params, _NEG0, target, value,
                gradient, same_as=f"zero-{loss}")
        )  # fmt: skip
    # Equal student and teacher scores: an exact zero gradient.
    for loss, params in (
        ("distillation", (("temperature", 1.0),)),
        ("distillation", (("temperature", 2.0),)),
    ):
        value, _ = _distillation(_Z0, _Z0, params[0][1])
        tag = f"{loss}-t{params[0][1]:g}"
        rows.append(Row(f"zero-{tag}", "zero_logits", loss, params, _Z0, _Z0, value, (0.0,) * 4))
        rows.append(
            Row(f"negzero-{tag}", "signed_zero_inputs", loss, params, _NEG0, _Z0, value,
                (0.0,) * 4, same_as=f"zero-{tag}")
        )  # fmt: skip
    return rows


def _upstream_rows() -> list[Row]:
    """Masked-class and zero-loss rows of masked_cases at upstream -1: the gradient negated,
    its zeros +0.0."""
    rows = []
    for case in M.valid():
        if (
            case.rule not in (1, 7)
            or case.loss in M.DISTILLATION[1:]
            or case.loss == M.PROBABILITIES
        ):
            continue
        if not case.name.endswith(("-a-one", "-a-three", "-zero-row", "sets-one")):
            continue
        row = _from_masked(case)
        gradient = tuple(-g + 0.0 for g in row.gradient or ())
        rows.append(
            dataclasses.replace(
                row, name=f"{row.name}-up", rule="negative_upstream", upstream=-1.0,
                gradient=gradient,
            )
        )  # fmt: skip
    return rows


def _storage_rows() -> list[Row]:
    rows = []
    # float32 work: p_y - 1 is a negative number that rounds to zero only in the storage.
    for dtype, gap in (("float16", 20.0), ("bfloat16", 100.0)):
        z = (0.0, -gap, -gap)
        value, _ = _expected_soft("cross_entropy", z, 0)
        rows.append(
            Row(f"storage-zero-{dtype}", "storage_negative_zero", "cross_entropy", (), z, 0,
                value, (-0.0, 0.0, 0.0), dtype=dtype)
        )  # fmt: skip
    # Entries that underflow in the work dtype: +0.0 (the loss is a reference: its float64
    # row sum may keep it).
    for dtype, gap in (("float32", 200.0), ("float64", 800.0)):
        z = (0.0, -gap, -gap)
        value, _ = _expected_soft("cross_entropy", z, 0)
        rows.append(
            Row(f"underflow-{dtype}", "work_underflow", "cross_entropy", (), z, 0, value,
                (0.0, 0.0, 0.0), dtype=dtype)
        )  # fmt: skip
    return rows


def _invalid_rows() -> list[Row]:
    nan4 = (NAN,) * 4
    rows = [
        Row("empty-set-ce", "invalid", "multilabel_cross_entropy", (), (0.5, 1.0, 0.0, 2.0), (),
            NAN, nan4),
        Row("empty-set-hinge", "invalid", "multilabel_hinge", (("gamma", 0.0),),
            (0.5, 1.0, 0.0, 2.0), (), NAN, nan4),
        Row("every-class-hinge", "invalid", "multilabel_hinge", (("gamma", 0.5),),
            (0.5, 1.0, 0.0, 2.0), (0, 1, 2, 3), NAN, nan4),
    ]  # fmt: skip
    # Labels out of range: NaN rows until pending.LABEL_ERRORS, then errors (ERRORS).
    for label in (4, -1):
        rows.append(
            Row(f"label-{label}", "invalid", "cross_entropy", (), (0.5, 1.0, 0.0, 2.0), label,
                NAN, nan4, until="label_errors")
        )  # fmt: skip
    return rows


INVALID_WEIGHT = "weight must be finite and nonnegative"
_Z = (1.5, -0.75, 0.75, 2.25)
_W = (0.5, 2.0, 0.0, 1.5)


def _option_rows() -> list[Row]:
    rows = []
    zero4 = (0.0,) * 4
    # ignore_index: ignored rows are +0.0 whatever their scores and class weights.
    for loss, params in (
        ("cross_entropy", ()),
        ("topk_hinge_beta", (("gamma", 0.5), ("k", 2))),
        ("proximal_cross_entropy", (("alpha", 0.5),)),
    ):
        for weight in (None, _W):
            tag = f"{loss}-{'w' if weight else 'plain'}"
            for name, z in (("finite", _Z), ("nan", (NAN, 0.0, 1.0, 2.0)), ("inf", (INF,) * 4),
                            ("masked", (-INF,) * 4)):  # fmt: skip
                rows.append(
                    Row(f"ignored-{tag}-{name}", "ignore_index", loss, params, z, IGNORE, 0.0,
                        zero4, weight=weight, ignore_index=IGNORE)
                )  # fmt: skip
            if loss == "cross_entropy":
                value, gradient = _expected_soft(loss, _Z, 0, weight)
                rows.append(
                    Row(f"kept-{tag}", "ignore_index", loss, params, _Z, 0, value, gradient,
                        weight=weight, ignore_index=IGNORE)
                )  # fmt: skip
    # An ignore_index that is a class.
    rows.append(
        Row("ignored-class", "ignore_index", "cross_entropy", (), _Z, 2, 0.0, zero4,
            ignore_index=2)
    )  # fmt: skip
    value, gradient = _expected_soft("cross_entropy", _Z, 1)
    rows.append(
        Row("kept-class", "ignore_index", "cross_entropy", (), _Z, 1, value, gradient,
            ignore_index=2)
    )  # fmt: skip
    # Class weights: 0 at the label gives +0.0; other weights scale the row's targets.
    for loss, params in (("cross_entropy", ()), ("multiclass_hinge", (("gamma", 0.0),))):
        rows.append(
            Row(f"weight-zero-{loss}", "zero_weight", loss, params, _Z, 2, 0.0, zero4, weight=_W)
        )  # fmt: skip
    value, gradient = _expected_soft("cross_entropy", _Z, 1, _W)
    rows.append(Row("weight-two", "zero_weight", "cross_entropy", (), _Z, 1, value, gradient,
                    weight=_W))  # fmt: skip
    rows.append(
        Row("weight-zero-masked-label", "zero_weight", "cross_entropy", (), (-INF, 1.0, 0.5, 2.0),
            2 * 0, NAN, (NAN,) * 4, weight=(0.0, 1.0, 1.0, 1.0))
    )  # fmt: skip
    # Invalid class weights: every row not ignored is NaN in PyTorch and JAX (which check no
    # traced values); an argument error of the C ABI, NumPy and Rust.
    for name, weight in (("nan", (1.0, NAN, 1.0, 1.0)), ("negative", (1.0, -0.5, 1.0, 1.0))):
        for loss, params in (
            ("cross_entropy", ()),
            ("topk_hinge_alpha", (("gamma", 0.0), ("k", 2))),
        ):
            rows.append(
                Row(f"weight-{name}-{loss}", "invalid_weight", loss, params, _Z, 0, NAN,
                    (NAN,) * 4, weight=weight, native_error=INVALID_WEIGHT)
            )  # fmt: skip
            rows.append(
                Row(f"weight-{name}-{loss}-ignored", "invalid_weight", loss, params, _Z, IGNORE,
                    0.0, zero4, weight=weight, ignore_index=IGNORE, native_error=INVALID_WEIGHT)
            )  # fmt: skip
    # Label smoothing.
    for eps, z, target in (
        (1.0, _Z, 0),
        (2.0**-40, _Z, 0),
        (0.5, (1.5, -INF, 0.75, 2.25), 0),  # target weight on the masked class: +inf
        (1.0, _Z, (0.25, 0.25, 0.5, 0.0)),
        (0.5, (1.5, -INF, 0.75, 2.25), (0.25, 0.0, 0.5, 0.25)),
    ):
        loss = "cross_entropy" if isinstance(target, int) else M.PROBABILITIES
        for weight in (None, _W):
            value, gradient = _expected_soft(loss, z, target, weight, eps)
            masked = "-masked" if -INF in z else ""
            tag = f"{'labels' if isinstance(target, int) else 'probs'}-{eps:g}{masked}"
            rows.append(
                Row(f"smoothing-{tag}-{'w' if weight else 'plain'}", "label_smoothing", loss, (),
                    z, target, value, gradient, weight=weight, label_smoothing=eps)
            )  # fmt: skip
    return rows


# The rules whose rows run in float32 storage too (the values are exact there).
FLOAT32_RULES = (
    "zero_logits",
    "signed_zero_inputs",
    "invalid",
    "ignore_index",
    "zero_weight",
    "invalid_weight",
    "label_smoothing",
)


def _float32(row: Row) -> Row:
    same_as = f"{row.same_as}-f32" if row.same_as else ""
    return dataclasses.replace(row, name=f"{row.name}-f32", dtype="float32", same_as=same_as)


def _rows() -> list[Row]:
    rows = [_from_masked(case) for case in M.CASES]
    new = _zero_rows() + _invalid_rows() + _option_rows()
    rows += new + _upstream_rows() + _storage_rows()
    rows += [_float32(row) for row in new if row.rule in FLOAT32_RULES]
    names = [row.name for row in rows]
    assert len(names) == len(set(names)), "row names must be unique"
    return rows


ALL_ROWS: tuple[Row, ...] = tuple(_rows())
ROWS: tuple[Row, ...] = tuple(
    r
    for r in ALL_ROWS
    if (not r.pending or ENABLED[r.pending]) and not (r.until and ENABLED[r.until])
)


# --- errors ----------------------------------------------------------------------------------

_K2 = (("k", 2),)
ERRORS: tuple[Error, ...] = (
    Error("k-zero", "topk_cross_entropy", (("k", 0),)),
    Error("k-classes", "topk_cross_entropy", (("k", 4),)),
    Error("k-classes-hinge", "topk_hinge_beta", (("k", 4), ("gamma", 0.0))),
    Error("gamma-negative", "multiclass_hinge", (("gamma", -1.0),)),
    Error("gamma-nan", "multilabel_hinge", (("gamma", NAN),)),
    Error("gamma-inf", "topk_hinge_alpha", (("k", 2), ("gamma", INF))),
    Error("step-zero", "proximal_cross_entropy", (("alpha", 0.0),)),
    Error("step-negative", "proximal_distillation", (("alpha", -1.0), ("temperature", 1.0))),
    Error("temperature-zero", "distillation", (("temperature", 0.0),)),
    Error("temperature-inf", "distillation", (("temperature", INF),)),
    Error("one-class", "cross_entropy", shape=(2, 1)),
    Error("one-class-distillation", "distillation", (("temperature", 1.0),), shape=(2, 1)),
    Error("labels-length", "cross_entropy", target="labels_length"),
    Error("float-labels", "multiclass_hinge", (("gamma", 0.0),), target="float_labels"),
    Error("teacher-shape", "distillation", (("temperature", 1.0),), target="teacher_shape"),
    Error("mask-shape", "multilabel_cross_entropy", target="mask_shape"),
    Error("probabilities-shape", M.PROBABILITIES, target="probs_shape"),
    Error("weight-shape", "cross_entropy", options=(("weight", (1.0, 1.0, 1.0)),)),
    Error("smoothing-above", "cross_entropy", options=(("label_smoothing", 1.5),)),
    Error("smoothing-negative", "cross_entropy", options=(("label_smoothing", -0.1),)),
    Error("smoothing-nan", "cross_entropy", options=(("label_smoothing", NAN),)),
    Error(
        "smoothing-hinge",
        "multiclass_hinge",
        (("gamma", 0.0),),
        options=(("label_smoothing", 0.1),),
    ),
    Error(
        "weight-multilabel",
        "multilabel_hinge",
        (("gamma", 0.0),),
        options=(("weight", (1.0,) * 4),),
    ),
    Error(
        "ignore-distillation",
        "distillation",
        (("temperature", 1.0),),
        options=(("ignore_index", 1),),
    ),
    Error("ignore-probabilities", M.PROBABILITIES, options=(("ignore_index", 0),)),
    Error("weight-nan", "cross_entropy", options=(("weight", (1.0, NAN, 1.0, 1.0)),)),
    Error("label-above", "cross_entropy", target="label_above", pending="label_errors"),
    Error("label-negative", "cross_entropy", target="label_negative", pending="label_errors"),
)
ACTIVE_ERRORS = tuple(e for e in ERRORS if not e.pending or ENABLED[e.pending])

# One text per error in every interface (docs/API.md, Errors): PyTorch's wording where PyTorch
# has a message for the condition, else the C ABI's and Rust's. JAX says "axis" for the class
# dimension, as its arguments do; the dtype of float labels is the one the interface sees.
CANONICAL: dict[str, str] = {
    "k-zero": "k must be at least 1, got 0",
    "k-classes": "k must be in [1, classes - 1] = [1, 3], got 4",
    "k-classes-hinge": "k must be in [1, classes - 1] = [1, 3], got 4",
    "gamma-negative": "gamma must be finite and nonnegative, got -1.0",
    "gamma-nan": "gamma must be finite and nonnegative, got nan",
    "gamma-inf": "gamma must be finite and nonnegative, got inf",
    "step-zero": "gamma must be finite and positive, got 0.0",
    "step-negative": "gamma must be finite and positive, got -1.0",
    "temperature-zero": "temperature must be finite and positive, got 0.0",
    "temperature-inf": "temperature must be finite and positive, got inf",
    "one-class": "At least two classes are required",
    "one-class-distillation": "At least two classes are required",
    "labels-length": (
        "target must have shape (2,) (the logits' shape without the class dimension), not (3,)"
    ),
    "float-labels": "multiclass_hinge takes integer class labels, not {dtype}",
    "teacher-shape": "student and teacher must have one shape, not (2, 4) and (2, 3)",
    "mask-shape": "target must have the logits' shape (2, 4), not (2, 3)",
    "probabilities-shape": "class probabilities must have the logits' shape (2, 4), not (2, 3)",
    "weight-shape": "weight must have shape (4,), not (3,)",
    "smoothing-above": "label_smoothing must be in [0, 1], got 1.5",
    "smoothing-negative": "label_smoothing must be in [0, 1], got -0.1",
    "smoothing-nan": "label_smoothing must be in [0, 1], got nan",
    "smoothing-hinge": "multiclass_hinge takes no parameter label_smoothing",
    "weight-multilabel": "multilabel_hinge takes no parameter weight",
    "ignore-distillation": "distillation takes no parameter ignore_index",
    "ignore-probabilities": "ignore_index is not supported with class probabilities",
    "weight-nan": INVALID_WEIGHT,
    "label-above": "Target 4 is out of bounds.",
    "label-negative": "Target -1 is out of bounds.",
}


def canonical(error: Error, interface: str, dtype: str = "float64") -> str:
    """The text of an error in an interface (capi, numpy, rust, cpp_loss, torch, jax, ...)."""
    text = CANONICAL[error.name].replace("{dtype}", dtype)
    if interface.startswith("jax"):
        text = text.replace("class dimension", "class axis")
    return text
