"""Solver work as a runtime regression check that needs no timer: epochs to a 1e-6 duality gap
on seeded problems, within 10% of the counts recorded on macOS arm64 and Linux x86_64 (their
std::shuffle orders differ). Update the table when a deliberate change moves a count."""

import math

import libsdca
import numpy as np
import pytest

# objective: (options, fewest and most epochs recorded)
EPOCHS = {
    "cross_entropy": ({}, 59, 59),
    "topk_cross_entropy": ({"k": 3}, 33, 33),
    "multiclass_hinge": ({"gamma": 1.0}, 284, 287),
    "topk_hinge_alpha": ({"k": 3, "gamma": 1.0}, 234, 237),
    "topk_hinge_beta": ({"k": 3, "gamma": 1.0}, 228, 233),
    "multilabel_hinge": ({"gamma": 1.0}, 165, 166),
    "multilabel_cross_entropy": ({}, 56, 56),
}


def _problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n, d, m = 400, 30, 8
    X = rng.standard_normal((n, d))
    W = rng.standard_normal((d, m))
    y = (X @ W + rng.standard_normal((n, m))).argmax(1)
    mask = (X @ W + rng.standard_normal((n, m))) > 1.0
    mask[np.arange(n), y] = True
    return X, y, mask


def _check(model: libsdca.Model, low: int, high: int) -> None:
    assert model.status == "solved"
    assert math.floor(0.9 * low) <= model.epochs <= math.ceil(1.1 * high)


@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("objective", list(EPOCHS))
def test_epochs_to_tolerance(objective: str, precision: str):
    X, y, mask = _problem()
    options, low, high = EPOCHS[objective]
    target = mask if objective.startswith("multilabel") else y
    model = libsdca.fit(
        X.astype(precision),
        target,
        objective,
        tol=1e-6,
        eval_epoch=1,
        max_epoch=2000,
        precision=precision,
        **options,
    )
    _check(model, low, high)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_kernel_epochs_to_tolerance(precision: str):
    X, y, _ = _problem()
    model = libsdca.fit(
        (X @ X.T).astype(precision),
        y,
        kernel=True,
        tol=1e-6,
        eval_epoch=1,
        max_epoch=2000,
        precision=precision,
    )
    _check(model, 59, 59)
