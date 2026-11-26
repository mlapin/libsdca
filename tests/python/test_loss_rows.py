"""sdca_loss_rows, the C entry point of the native route of libsdca.torch and libsdca.jax: at
its default options it gives the bytes of sdca_loss_evaluate (NumPy's loss_and_grad), an out-of-range
label raises IndexError, the gradient factors apply, and the
options a call cannot take are invalid arguments. test_native_route_torch.py and
test_native_route_jax.py compare its options and 16-bit storage with the CPU extensions."""

from __future__ import annotations

import re
from typing import Any

import libsdca
import numpy as np
import pytest
from libsdca import _frontend as _f
from libsdca import _native

N, C = 9, 6
RNG = np.random.default_rng(0)
Z = RNG.standard_normal((N, C))
LABELS = RNG.integers(0, C, N)
MASK = RNG.random((N, C)) < 0.4
MASK[:, 0], MASK[:, 1] = True, False
TEACHER = RNG.standard_normal((N, C))
P = RNG.random((N, C))
P /= P.sum(1, keepdims=True)
LOSSES = [
    (libsdca.CrossEntropy(), "labels"),
    (libsdca.CrossEntropy(), "probabilities"),
    (libsdca.TopKCrossEntropy(k=2), "labels"),
    (libsdca.MulticlassHinge(gamma=0.5), "labels"),
    (libsdca.TopKHinge(k=2, variant="alpha"), "labels"),
    (libsdca.TopKHinge(k=3), "labels"),
    (libsdca.MultilabelHinge(), "mask"),
    (libsdca.MultilabelCrossEntropy(), "mask"),
    (libsdca.ProximalCrossEntropy(gamma=0.7), "labels"),
    (libsdca.Distillation(temperature=2.0), "teacher"),
    (libsdca.ProximalDistillation(temperature=2.0, gamma=0.5), "teacher"),
]


def rows_of(loss: libsdca.Loss, **options: Any) -> _f.Rows:
    struct = loss._struct()
    return _f.Rows(struct.kind, max(struct.k, 1), struct.gamma, struct.temperature, **options)


def target_of(role: str, dtype: Any) -> np.ndarray:
    return {
        "labels": LABELS,
        "mask": MASK.view(np.uint8),
        "teacher": TEACHER.astype(dtype),
        "probabilities": P,
    }[role]


def call(
    rows: _f.Rows,
    z: np.ndarray,
    target: np.ndarray,
    role: str,
    weight: np.ndarray | None = None,
    upstream: float = 1.0,
    row_upstream: np.ndarray | None = None,
    probability_dtype: str = "float64",
) -> tuple[np.ndarray, np.ndarray]:
    values, grads = np.empty(len(z)), np.empty_like(z)
    _native.loss_rows(
        rows, 1, z.dtype.name, len(z), C, z.ctypes.data, target.ctypes.data, role,
        probability_dtype, None if weight is None else weight.ctypes.data, upstream,
        None if row_upstream is None else row_upstream.ctypes.data,
        values.ctypes.data, grads.ctypes.data,
    )  # fmt: skip
    return values, grads


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(("loss", "role"), LOSSES, ids=lambda x: str(getattr(x, "name", x)))
def test_default_options_give_the_bytes_of_sdca_loss_evaluate(
    loss: libsdca.Loss, role: str, dtype: str
) -> None:
    z = Z.astype(dtype)
    target = target_of(role, dtype)
    given = {"labels": LABELS, "mask": MASK, "teacher": target, "probabilities": None}[role]
    want = libsdca.loss_and_grad(z, libsdca.Probabilities(P) if given is None else given, loss)
    got = call(rows_of(loss), z, target, role)
    assert got[0].tobytes() == want[0].tobytes() and got[1].tobytes() == want[1].tobytes()


def test_labels_out_of_range_raise() -> None:
    labels = LABELS.copy()
    labels[2] = C
    with pytest.raises(IndexError, match=rf"^Target {C} is out of bounds\.$"):
        call(rows_of(libsdca.CrossEntropy()), Z, labels, "labels")


def test_the_factors_multiply_the_gradients() -> None:
    rows = rows_of(libsdca.CrossEntropy())
    values, unit = call(rows, Z, LABELS, "labels")
    factors = np.arange(N, dtype=np.float64)
    scaled_values, scaled = call(rows, Z, LABELS, "labels", upstream=0.5, row_upstream=factors)
    assert scaled_values.tobytes() == values.tobytes()
    np.testing.assert_allclose(scaled, unit * (0.5 * factors)[:, None], rtol=1e-15, atol=0)


def test_ignored_rows_and_class_weights() -> None:
    rows = rows_of(libsdca.CrossEntropy(), ignore_index=int(LABELS[0]), ignore=True)
    weight = RNG.uniform(0.5, 2.0, C)
    values, grads = call(rows, Z, LABELS, "labels", weight=weight)
    plain, plain_grads = call(rows_of(libsdca.CrossEntropy()), Z, LABELS, "labels")
    ignored = LABELS == LABELS[0]
    assert not values[ignored].any() and not grads[ignored].any()
    np.testing.assert_allclose(values[~ignored], (plain * weight[LABELS])[~ignored], rtol=1e-15)


@pytest.mark.parametrize(
    ("loss", "options", "message"),
    [
        (libsdca.MulticlassHinge(), {"label_smoothing": 0.1}, "multiclass_hinge takes no parameter label_smoothing"),
        (libsdca.CrossEntropy(), {"label_smoothing": 1.5}, "label_smoothing must be in [0, 1], got 1.5"),
        (libsdca.CrossEntropy(), {"ignore": 2}, "ignore must be 0 or 1"),
    ],
)  # fmt: skip
def test_options_a_loss_does_not_take_are_invalid(
    loss: libsdca.Loss, options: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        call(rows_of(loss, **options), Z, LABELS, "labels")


def test_targets_a_call_cannot_take_are_invalid() -> None:
    rows = rows_of(libsdca.MultilabelHinge())
    with pytest.raises(ValueError, match="^multilabel_hinge takes no parameter weight$"):
        call(rows, Z, MASK.view(np.uint8), "mask", weight=np.ones(C))
    with pytest.raises(ValueError, match="^Probabilities must be float32, float64 or of the"):
        call(rows_of(libsdca.CrossEntropy()), Z, P, "probabilities", probability_dtype="float16")
