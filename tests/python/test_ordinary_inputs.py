"""The NumPy API on ordinary inputs against independent float64 references.

loss_and_grad of every hinge (nonsmooth and smoothed) and cross-entropy against
loss_references.py, over shapes, storage, precision, target encodings and layouts; every
objective's "solved" fits within tol of the optimum, judged by an independent primal.
"""

from typing import Any

import libsdca
import numpy as np
import pytest
from libsdca import OBJECTIVES, loss_and_grad
from loss_references import REFERENCED, nonsmooth_gap, primal, reference


def targets(loss: str, n: int, m: int, rng: np.random.Generator) -> tuple[np.ndarray, Any]:
    """Labels and the loss's target: labels, or a mask with a negative label (multilabel)."""
    labels = rng.integers(0, m, n)
    if not loss.startswith("multilabel"):
        return labels, labels
    mask = np.eye(m, dtype=bool)[labels] | (rng.random((n, m)) < 0.3)
    mask[np.arange(n), (labels + 1) % m] = False
    return labels, mask


class OrdinaryInputTests:
    @pytest.mark.parametrize("loss", REFERENCED)
    def test_losses_match_independent_references(self, loss: str):
        rng = np.random.default_rng(len(loss))
        for m in (2, 3, 10, 37):
            for n in (1, 7):
                labels, target = targets(loss, n, m, rng)
                multilabel = loss.startswith("multilabel")
                positive = np.asarray(target) if multilabel else np.eye(m, dtype=bool)[labels]
                z = 3 * rng.normal(size=(n, m))
                ks = sorted({1, max(1, m // 2), m - 1}) if "topk" in loss else [1]
                gammas = (0.0, 0.1, 1.0, 30.0) if "hinge" in loss else (0.0,)
                for k in ks:
                    for gamma in gammas:
                        rv, rg = reference(z, positive, loss, k, gamma)
                        ok = np.ones(n, bool)
                        if gamma == 0 and "hinge" in loss:
                            ok = nonsmooth_gap(z, positive, loss, k) > 1e-6
                        for dtype, precision, rtol in [
                            (np.float64, "float64", 1e-11),
                            (np.float32, "float64", 1e-11),
                            (np.float32, "float32", 3e-5),
                        ]:
                            zz = z.astype(dtype)
                            if dtype == np.float32 and precision == "float64":
                                rv, rg = reference(zz, positive, loss, k, gamma)
                            # Fortran order, and the target as labels, a mask or label lists.
                            encodings = [
                                target,
                                positive,
                                [list(np.flatnonzero(r)) for r in positive],
                            ]
                            for t in encodings:
                                v, g = loss_and_grad(
                                    np.asfortranarray(zz),
                                    t,
                                    loss,
                                    k=k,
                                    gamma=gamma,
                                    precision=precision,
                                )
                                message = (
                                    f"{loss} m={m} n={n} k={k} gamma={gamma} {dtype} {precision}"
                                )
                                scale = 1 + np.abs(z).max(-1)
                                assert g.dtype == dtype, message
                                np.testing.assert_array_less(
                                    np.abs(v - rv), rtol * (np.abs(rv) + scale), err_msg=message
                                )
                                assert np.all(
                                    np.abs(g - rg)[ok]
                                    <= rtol * scale[ok, None] + 4 * np.finfo(dtype).eps
                                ), message

    def test_losses_do_not_depend_on_layout_or_label_dtype(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=(40, 6))
        y = rng.integers(0, 6, 40)
        for loss in ("cross_entropy", "topk_hinge_beta", "multiclass_hinge"):
            expected = loss_and_grad(z, y, loss, k=2, gamma=0.5, num_threads=1)
            wide = np.zeros((40, 12))
            wide[:, ::2] = z
            for zz in (np.asfortranarray(z), wide[:, ::2]):
                for dtype in (np.int8, np.uint8, np.int16, np.int32, np.uint64):
                    actual = loss_and_grad(zz, y.astype(dtype), loss, k=2, gamma=0.5)
                    np.testing.assert_array_equal(actual[0], expected[0])
                    np.testing.assert_array_equal(actual[1], expected[1])

    @pytest.mark.parametrize("objective", OBJECTIVES)
    def test_solved_fits_are_within_tol_of_the_optimum(self, objective: str):
        rng = np.random.default_rng(5)
        n, d, m = 80, 5, 4
        labels, y = targets(objective, n, m, rng)
        # Unbalanced classes: class 3 has two examples.
        labels[labels == 3] = 2
        labels[:2] = 3
        if not objective.startswith("multilabel"):
            y = labels
        x = rng.normal(size=(n, d)) + 2 * np.eye(m, d)[labels]
        options: dict[str, Any] = dict(k=2, gamma=0.5 if "hinge" in objective else 0.0)
        tight = libsdca.fit(x, y, objective, tol=1e-10, max_epoch=20000, **options)
        assert tight.status == "solved" and tight.weights is not None
        best = primal(x, tight.weights, y, objective, 1.0, **options)
        for dtype in (np.float64, np.float32):
            for order in ("C", "F"):
                data = np.asarray(x, dtype, order=order)
                for tol in (1e-2, 1e-4):
                    model = libsdca.fit(data, y, objective, tol=tol, **options)
                    message = f"{objective} {dtype} {order} tol={tol}"
                    assert model.status == "solved", message
                    assert model.weights is not None and model.weights.dtype == dtype
                    value = primal(data, model.weights, y, objective, 1.0, **options)
                    # The certificate: P(W) - P* <= tol max(|P|, |D|) (here P >= D > 0).
                    assert value - best <= tol * value * (1 + 1e-9), message
                    reported = model.history["primal"][-1]
                    assert abs(reported - value) <= 1e-9 * value, message
                    if order == "F":
                        again = libsdca.fit(
                            np.ascontiguousarray(data), y, objective, tol=tol, **options
                        )
                        np.testing.assert_array_equal(again.dual, model.dual, err_msg=message)
        # A Gram matrix trains the same scores.
        gram = libsdca.fit(
            x @ x.T, y, objective, tol=1e-10, kernel=True, max_epoch=20000, **options
        )
        # (||W - W*||^2 <= 2 gap bounds the scores.)
        np.testing.assert_allclose(
            gram.decision_function(x @ x.T), tight.decision_function(x), atol=1e-3
        )
