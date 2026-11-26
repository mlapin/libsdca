"""Cross-entropy rows at the limits of the work type's exponentials, entry by entry against
the mpmath references (conformance.references) of the represented inputs:

- rows of three or more classes near the precision cliff: the loss and the mass of the label
  from exp(gap) times the competitors' sum, not from exp(gap + log(sum));
- float64 probability targets, multilabel masks and proximal rows whose exponentials are
  subnormal or near the smallest normal: the vector exponentials round subnormal results
  once instead of flushing them;
- the first largest class's entry of probability targets: accurate relative to itself in
  float work, and at g = 0 (targets the softmax of the logits) within a few ulps of the larger
  of T q_j and u_j, where others - T tail / (1 + tail) cancels.

Each entry x^ of an exact x is within ULPS (eps |x| + work_eps a) + eta of it, eta the least
subnormal of the output and a an entry's allowance (0 unless stated).
"""

from __future__ import annotations

import libsdca
import named_losses
import numpy as np
import pytest
from conformance.references import Loss, evaluate
from storage_limits import LIMITS

ULPS = 4


def call(name: str, z: np.ndarray, target: object, **params: float) -> tuple[np.ndarray, ...]:
    values, gradient = libsdca.loss_and_grad(z, target, named_losses.numpy_loss(name, **params))
    return np.asarray(values, np.float64), np.asarray(gradient, np.float64)


def check(
    got: np.ndarray, want: np.ndarray, dtype: type, allowance: np.ndarray | float = 0.0
) -> None:
    lim = LIMITS[np.dtype(dtype).name]
    bound = ULPS * (lim.eps * np.abs(want) + lim.eps * allowance) + lim.tiny * lim.eps
    bad = np.abs(got - want) > bound
    ulps = np.abs(got - want) / (lim.eps * np.abs(want))
    assert not bad.any(), (got[bad], want[bad], ulps[bad])


def cliff_rows(dtype: type) -> np.ndarray:
    """The label at 0 above a tie at -m and a competitor at -(m + 1), for margins about the
    float64 cliff 53 ln 2 (36.7) and on to where the loss nears the smallest normal."""
    margins = [30.5, 33.0631205, 36.7368006, 40.0, 100.0, 700.0, 707.0]
    return np.array([[0.0, -m, -m, -(m + 1)] for m in margins], dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("name", ["cross_entropy", "topk_cross_entropy"])
def test_cliff_rows(name: str, dtype: type) -> None:
    z = cliff_rows(dtype)
    if dtype == np.float32:
        z = z[np.abs(z).max(-1) < 100]
    labels = np.zeros(len(z), np.int64)
    k = 2 if name == "topk_cross_entropy" else 1
    value, gradient = call(name, z, labels, **({"k": k} if k > 1 else {}))
    want_value, want_gradient = evaluate(Loss(name, k=k), z.astype(np.float64), labels)
    check(value, want_value, np.float64 if dtype == np.float64 else np.float32)
    check(gradient, want_gradient, dtype)


# Rows whose competitors' exponentials are at the smallest normal, below it, and down to the
# least subnormal of double.
SUBNORMAL_ROWS = [
    [0.0, -708.396419],
    [0.0, -713.396419],
    [0.0, -744.440072],
    [0.0, -707.218068, -707.218068, -708.218068],
    [0.0, -708.396419, -708.396419, -709.396419],
]


@pytest.mark.parametrize("row", SUBNORMAL_ROWS, ids=lambda r: f"{len(r)}x{r[-1]:g}")
def test_float64_subnormal_probabilities(row: list[float]) -> None:
    z = np.array([row])
    p = np.zeros_like(z)
    p[0, 0] = 1
    label_value, label_gradient = call("cross_entropy", z, np.array([0]))
    value, gradient = call("cross_entropy", z, libsdca.Probabilities(p))
    want_value, want_gradient = evaluate(Loss("cross_entropy"), z, p)
    assert want_value[0] > 0
    check(value, want_value, np.float64)
    check(gradient, want_gradient, np.float64)
    check(label_value, want_value, np.float64)
    check(label_gradient, want_gradient, np.float64)


@pytest.mark.parametrize("row", SUBNORMAL_ROWS, ids=lambda r: f"{len(r)}x{r[-1]:g}")
def test_float64_subnormal_multilabel(row: list[float]) -> None:
    """One positive on top, and two tied there (the anchored entries of the positives)."""
    for z, positives in ((np.array([row]), [0]), (np.array([[0.0, *row]]), [0, 1])):
        mask = np.zeros(z.shape, bool)
        mask[0, positives] = True
        want_value, want_gradient = evaluate(Loss("multilabel_cross_entropy"), z, mask)
        for target in (mask, [positives]):
            value, gradient = call("multilabel_cross_entropy", z, target)
            check(value, want_value, np.float64)
            check(gradient, want_gradient, np.float64)


# The gaps at the smallest normal of the work (log_min), and a tie with a competitor one below.
F32, F64 = LIMITS["float32"].log_min, LIMITS["float64"].log_min
TIE64 = F64 + 1.1783505  # the step's tie just above log_min, its competitor just below


@pytest.mark.parametrize(
    "dtype,row",
    [
        (np.float32, [0.0, F32]),
        (np.float32, [0.0, F32, F32, F32 - 1]),
        (np.float32, [0.0, -90.0, -95.0]),
        (np.float64, [0.0, F64]),
        (np.float64, [0.0, TIE64, TIE64, TIE64 - 1]),
        (np.float64, [0.0, F64 - 5, F64 - 5]),
    ],
)
def test_proximal_flush_threshold(dtype: type, row: list[float]) -> None:
    """The step's exponentials at and below the smallest normal of the work: the loss and the
    entries within the proximal allowance, the gradient's 1-norm."""
    z = np.array([row], dtype)
    labels = np.array([0])
    value, gradient = call("proximal_cross_entropy", z, labels, gamma=0.1)
    want_value, want_gradient = evaluate(
        Loss("proximal_cross_entropy", alpha=0.1), z.astype(np.float64), labels
    )
    assert want_value[0] > 0 and value[0] > 0
    check(value, want_value, dtype)
    check(gradient, want_gradient, dtype, np.abs(want_gradient).sum())


@pytest.mark.parametrize("classes", [8, 40])
def test_float32_first_class_entry(classes: int) -> None:
    """A confident first class of soft targets (q_0 = 1 - 2.3e-3, u_0 = 1 - 1e-5): its entry
    -2.3e-3 within ULPS of itself, not of u_0 (T less u_0 in float sums)."""
    rows = 16
    z = np.zeros((rows, classes), np.float32)
    z[:, 0] = np.float32(np.log(classes - 1) + 6) + np.arange(rows, dtype=np.float32) / 8
    p = np.full((rows, classes), 1e-5 / (classes - 1), np.float32)
    p[:, 0] = 1 - 1e-5
    value, gradient = call("cross_entropy", z, libsdca.Probabilities(p))
    _, want = evaluate(Loss("cross_entropy"), z.astype(np.float64), p.astype(np.float64))
    check(gradient, want, np.float32)


def test_float32_underflow_is_positive_zero() -> None:
    """Entries that underflow float work are +0.0 (docs/API.md, Signed zeros), the label's and
    the first class's formed in double included."""
    z = np.array([[0.0, -200.0, -200.0]], np.float32)
    targets = (np.array([0]), libsdca.Probabilities(np.array([[1.0, 0.0, 0.0]], np.float32)))
    for target in targets:
        _, gradient = call("cross_entropy", z, target)
        assert (gradient == 0).all() and not np.signbit(gradient).any(), gradient
    for target in (np.array([[True, False, False]]), [[0]]):
        _, gradient = call("multilabel_cross_entropy", z, target)
        assert (gradient == 0).all() and not np.signbit(gradient).any(), gradient


def test_float64_probabilities_at_the_softmax() -> None:
    """Targets the softmax of the logits (Barron et al., Fig. 1b at g = 0): every entry within
    ULPS of the larger of T q_j and u_j."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((32, 53))
    e = np.exp(z - z.max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)
    _, gradient = call("cross_entropy", z, libsdca.Probabilities(p))
    _, want = evaluate(Loss("cross_entropy"), z, p)
    q = p.sum(-1, keepdims=True) * e / e.sum(-1, keepdims=True)
    check(gradient, want, np.float64, np.maximum(q, p))
