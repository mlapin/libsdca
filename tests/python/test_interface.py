"""Numerical and memory-layout checks of the NumPy API; run with pytest from the repo root."""

import json
import warnings
from pathlib import Path

import libsdca
import numpy as np
import pytest
from libsdca import (
    CrossEntropy,
    Kernel,
    MulticlassHinge,
    MultilabelCrossEntropy,
    MultilabelHinge,
    TopKHinge,
)
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


class InterfaceTests:
    def test_loss_and_noncontiguous_input(self):
        rng = np.random.default_rng(5)
        z = rng.normal(size=(12, 6))[::2, ::2]
        y = np.arange(6) % 3
        original = z.copy()
        for precision, dtype, atol in [
            ("float64", np.float64, 2e-15),
            ("float64", np.float32, 2e-7),
            ("float32", np.float32, 4e-7),
        ]:
            got = libsdca.loss_and_grad(z.astype(dtype), y, precision=precision)[0]
            expected = np.asarray(logsumexp(z, axis=1)) - z[np.arange(6), y]
            np.testing.assert_allclose(got, expected, atol=atol)
        np.testing.assert_array_equal(z, original)

    def test_coordinate_kkt_and_label_permutation(self):
        z = np.array([[2.0, 0.2, -1.0], [-0.5, 1.2, 2.0]])
        y = np.array([2, 1])
        c, norm = 0.7, 1.3
        old = libsdca.coordinate_update(z, y, C=c, norm2=norm)
        a = libsdca.coordinate_update(z, y, previous=old, C=c, norm2=norm)
        new_scores = z + norm * (a - old)
        expected = c * (np.eye(3)[y] - softmax(new_scores, axis=1))
        np.testing.assert_allclose(a, expected, atol=2e-14)
        np.testing.assert_allclose(a.sum(axis=1), 0, atol=2e-14)

    def test_per_row_norms_equal_separate_steps(self):
        z = np.array([[2.0, 0.2, -1.0], [-0.5, 1.2, 2.0]])
        y = np.array([2, 1])
        norms = np.array([0.5, 3.0])
        both = libsdca.coordinate_update(z, y, C=0.7, norm2=norms)
        for i in range(2):
            one = libsdca.coordinate_update(z[i : i + 1], y[i : i + 1], C=0.7, norm2=norms[i])
            np.testing.assert_array_equal(both[i : i + 1], one)
        with pytest.raises(ValueError, match="norm2"):
            libsdca.coordinate_update(z, y, norm2=[1.0, 2.0, 3.0])

    def test_feature_kernel_and_independent_optimizer(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(24, 4))
        y = rng.integers(0, 3, size=24)
        c = 0.4

        def fun(flat: np.ndarray) -> tuple[float, np.ndarray]:
            w = flat.reshape(4, 3)
            z = x @ w
            p = softmax(z, axis=1)
            p[np.arange(len(y)), y] -= 1
            losses = np.asarray(logsumexp(z, axis=1)) - z[np.arange(len(y)), y]
            return 0.5 * np.sum(w * w) + c * np.sum(losses), (w + c * x.T @ p).ravel()

        ref = minimize(fun, np.zeros(12), jac=True, method="BFGS", options={"gtol": 1e-10})
        for recompute in (True, False):
            m = libsdca.fit(x, y, C=c, tol=1e-11, recompute_primal=recompute)
            ker = libsdca.fit(Kernel(x @ x.T), y, C=c, tol=1e-11)
            assert m.status == "solved"
            assert m.weights is not None and ker.weights is None and ker.kernel
            np.testing.assert_allclose(m.weights.T, ref.x.reshape(4, 3), atol=5e-6)
            np.testing.assert_allclose(
                m.decision_function(x), ker.decision_function(x @ x.T), atol=2e-5
            )
            np.testing.assert_allclose(m.weights, m.dual.T @ x, atol=2e-14)
            assert abs(fun(m.weights.T.ravel())[0] - ref.fun) < 1e-9

    def test_collapse_is_fixed(self):
        for precision, dtype, gap in [
            ("float32", np.float32, 20),
            ("float64", np.float32, 20),
            ("float64", np.float64, 50),
        ]:
            z = np.array([[gap / 2, -gap / 2, -gap / 2]], dtype)
            a = libsdca.coordinate_update(z, [0], precision=precision)
            assert a[0, 0] > 0
            assert np.all(a[0, 1:] < 0)
            np.testing.assert_allclose(a[0, 0], -np.sum(a[0, 1:]), rtol=1e-7)
            assert libsdca.loss_and_grad(z, [0], precision=precision)[0][0] > 0
        # This block is reachable from a feasible ordinary SDCA state:
        # X=[1,32], A=[[0,0,0],[.375,-.1875,-.1875]], W=X.T@A.
        for precision in ["float32", "float64"]:
            z = np.array([[12.0, -6.0, -6.0]], np.float32)
            a = libsdca.coordinate_update(z, [0], precision=precision)
            assert a[0, 0] > 0
            assert float(a.sum()) == 0

    def test_invalid_input(self):
        # A label out of range is an error, as in PyTorch.
        for y in [[-1, 0], [3, 0]]:
            with pytest.raises(ValueError, match=f"Target {y[0]} is out of bounds."):
                libsdca.loss_and_grad([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], y)
        for y in [[0.1], [[1.0, 0.0]]]:
            with pytest.raises(TypeError):
                libsdca.loss_and_grad([[1.0, 2.0, 3.0]], y)
        with pytest.raises(ValueError, match="nonnegative"):
            libsdca.fit(Kernel(-np.eye(2)), [0, 1])
        # All-zero rows take a block of size C, which must fit float32 storage;
        # the check compares C exactly, without an overflowing cast.
        for precision in ["float32", "float64"]:
            for C in [1e100, 1e-100]:
                for x in [np.zeros((1, 1), np.float32), Kernel(np.zeros((1, 1), np.float32))]:
                    with pytest.raises(ValueError, match="normal"):
                        with warnings.catch_warnings():
                            warnings.simplefilter("error")
                            libsdca.fit(x, [0], num_classes=2, C=C, precision=precision)
        with pytest.raises(ValueError):
            libsdca.coordinate_update([[1.0, float("nan")]], [0])
        with pytest.raises(TypeError):
            libsdca.fit(np.eye(2), [0, 1], max_epoch=1.5)  # pyright: ignore[reportArgumentType]

    def test_labels_and_kernels_are_checked_without_guessing(self):
        x = np.eye(3)
        # Float labels, and a (n,1) integer column, are rejected.
        with pytest.raises(TypeError, match="integer"):
            libsdca.fit(x, np.array([0.0, 1.0, 2.0]))
        with pytest.raises(ValueError, match="ravel"):
            libsdca.fit(x, np.array([[0], [1], [2]]))
        # cross_entropy takes labels only: label sets are the multilabel losses' targets.
        with pytest.raises(TypeError):
            libsdca.fit(x, [[0, 1], [1], [2]], max_epoch=1)
        assert libsdca.fit(x, np.array([0, 1, 2]), max_epoch=1).loss == CrossEntropy()
        model = libsdca.fit(x, [[0], [1], [2]], MultilabelCrossEntropy(), max_epoch=1)
        assert model.loss == MultilabelCrossEntropy()
        # The library checks a training kernel's symmetry and |K_ij| <= sqrt(K_ii K_jj).
        for gram, message in [
            ([[1.0, 0.5], [0.4, 1.0]], "symmetric"),
            ([[1.0, 2.0], [2.0, 1.0]], "positive semidefinite"),
            ([[1.0, np.nan], [np.nan, 1.0]], "finite"),
            (np.ones((2, 3)), "square"),
        ]:
            with pytest.raises(ValueError, match=message):
                libsdca.fit(Kernel(np.array(gram)), [0, 1])
        # A Gram matrix rounded asymmetrically (float32 products) is accepted.
        rng = np.random.default_rng(3)
        f = rng.normal(size=(40, 200)).astype(np.float32)
        gram = np.empty((40, 40), np.float32)
        for i in range(40):
            gram[i] = f @ f[i]
        model = libsdca.fit(Kernel(gram), np.arange(40) % 3, tol=1e-4)
        assert model.status == "solved"

    def test_label_sets_from_masks_and_lists(self):
        mask = np.array([[1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0]], bool)
        lists = [[2, 0], [3], [1, 0]]
        z = np.array([[0.5, -1.0, 2.0, 0.0], [1.0, 0.0, -2.0, 3.0], [0.0, 1.0, 0.5, -0.5]])
        reference = libsdca.loss_and_grad(z, mask, MultilabelHinge(gamma=0.5))
        for y in [np.asfortranarray(mask), lists, [np.array(row) for row in lists]]:
            got = libsdca.loss_and_grad(z, y, MultilabelHinge(gamma=0.5))
            np.testing.assert_array_equal(got[0], reference[0])
            np.testing.assert_array_equal(got[1], reference[1])
        for y, error in [
            (mask[:2], ValueError),
            ([[0, 0], [3], [1]], ValueError),
            ([0, 3, 1], TypeError),
            (np.array([0, 3, 1]), TypeError),
            ([[0.5], [3], [1]], TypeError),
        ]:
            with pytest.raises(error):
                libsdca.loss_and_grad(z, y, MultilabelHinge())

    def test_center_and_history_capacity(self):
        x = np.eye(3)
        center = np.array([[2.0, 0.0, -1.0], [1.0, 0.0, 2.0], [0.0, 2.0, -1.0]])
        for epochs in [0, 1, 2, 3, 7]:
            for eval_epoch in [0, 3]:
                m = libsdca.fit(
                    x, [0, 2, 1], center=center, max_epoch=epochs, eval_epoch=eval_epoch, tol=0
                )
                assert m.history["epoch"][-1] == m.epochs
                assert m.weights is not None
                np.testing.assert_allclose(m.weights, center + m.dual.T @ x, atol=1e-15)
                if eval_epoch == 0:
                    assert len(m.history["epoch"]) == 1

    def test_names_and_model_ergonomics(self):
        x = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, -2.0], [1.5, 0.5]])
        y = np.array([0, 1, 2, 0])
        mask = np.array(
            [[True, True, False], [False, True, False], [False, False, True], [True, False, True]]
        )
        # Losses are objects: no names, no aliases.
        for name in ("cross_entropy", "softmax", "topk_hinge"):
            with pytest.raises(TypeError, match="loss"):
                libsdca.loss_and_grad(x @ np.ones((2, 3)), y, name)  # pyright: ignore[reportArgumentType]
            with pytest.raises(TypeError, match="loss"):
                libsdca.fit(x, y, name)  # pyright: ignore[reportArgumentType]
        # multiclass_hinge is topk_hinge of either variant at k = 1.
        z = x @ np.array([[1.0, -1.0, 0.5], [0.0, 2.0, -1.0]])
        for gamma in (0.0, 0.5):
            for other in (TopKHinge(1, "alpha", gamma), TopKHinge(1, "beta", gamma)):
                np.testing.assert_array_equal(
                    libsdca.loss_and_grad(z, y, MulticlassHinge(gamma))[1],
                    libsdca.loss_and_grad(z, y, other)[1],
                )
        model = libsdca.fit(x, y)
        assert model.loss == CrossEntropy() and model.C == 1.0 and model.num_classes == 3
        assert model.score(x, y) == float(np.mean(model.predict(x) == y))
        proba = model.predict_proba(x)
        log_proba = model.predict_log_proba(x)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-15)
        np.testing.assert_allclose(
            log_proba,
            log_proba - np.asarray(logsumexp(log_proba, axis=1, keepdims=True)),
            atol=1e-15,
        )
        np.testing.assert_allclose(np.exp(log_proba), proba, atol=1e-15)
        ml_model = libsdca.fit(x, mask, MultilabelCrossEntropy(), max_epoch=10)
        np.testing.assert_allclose(ml_model.predict_proba(x).sum(axis=1), 1.0, atol=1e-15)
        # Multilabel scores: the top class within the label set; masks and lists alike.
        top = ml_model.predict(x)
        expected = float(np.mean(mask[np.arange(4), top]))
        assert ml_model.score(x, mask) == expected
        assert ml_model.score(x, [np.flatnonzero(row).tolist() for row in mask]) == expected
        svm_model = libsdca.fit(x, y, MulticlassHinge(), max_epoch=10)
        with pytest.raises(ValueError, match="cross-entropy"):
            svm_model.predict_proba(x)
        with pytest.raises(ValueError, match="cross-entropy"):
            svm_model.predict_log_proba(x)


def test_projection_precision_is_not_an_option():
    with pytest.raises(TypeError, match="projection_precision"):
        MultilabelHinge(projection_precision="extended")


def test_blas_package_record(tmp_path: Path):
    from libsdca._native import _load_blas_package

    library = tmp_path / "libsdca_c.so"
    assert _load_blas_package(library) is None
    record = library.with_suffix(".json")
    record.write_text(json.dumps({"backend": "EXTERNAL"}))
    assert _load_blas_package(library) is None
    package = {"distribution": "no-such-blas", "module": "no_such_blas", "library": "lib/x.so"}
    record.write_text(json.dumps({"blas_package": package}))
    with pytest.raises(OSError, match="pip install no-such-blas"):
        _load_blas_package(library)


def test_blas_package_threads_follow_threadpoolctl():
    threadpoolctl = pytest.importorskip("threadpoolctl")
    from libsdca._native import _native

    library = _native().blas_library
    if library is None:
        pytest.skip("the library links no BLAS package")
    path = Path(library._name).resolve()

    def threads() -> list[int]:
        pools = threadpoolctl.threadpool_info()
        return [p["num_threads"] for p in pools if Path(p["filepath"]).resolve() == path]

    assert threads(), f"threadpoolctl does not list {path}"
    with threadpoolctl.threadpool_limits(1):
        assert threads() == [1]
