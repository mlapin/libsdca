"""Numerical and memory-layout checks; run with pytest from the repo root."""

import ctypes as ct
import warnings
from typing import Any

import numpy as np
import pytest
from libsdca._core import HISTORY, Solver, _fit_result, _label_sets, _options, _ptr
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
            got = Solver(precision, dtype=dtype).loss_and_grad(z, y)[0]
            expected = np.asarray(logsumexp(z, axis=1)) - z[np.arange(6), y]
            np.testing.assert_allclose(got, expected, atol=atol)
        np.testing.assert_array_equal(z, original)

    def test_coordinate_kkt_and_label_permutation(self):
        z = np.array([[2.0, 0.2, -1.0], [-0.5, 1.2, 2.0]])
        y = np.array([2, 1])
        c, norm = 0.7, 1.3
        solver = Solver()
        old = solver.coordinate_update(z, y, C=c, norm2=norm)
        a = solver.coordinate_update(z, y, previous=old, C=c, norm2=norm)
        new_scores = z + norm * (a - old)
        expected = c * (np.eye(3)[y] - softmax(new_scores, axis=1))
        np.testing.assert_allclose(a, expected, atol=2e-14)
        np.testing.assert_allclose(a.sum(axis=1), 0, atol=2e-14)

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
        solver = Solver()
        for recompute in (True, False):
            m = solver.fit(x, y, C=c, tol=1e-11, recompute_primal=recompute)
            ker = solver.fit(x @ x.T, y, C=c, tol=1e-11, kernel=True)
            assert m.status == "solved"
            assert m.weights is not None
            np.testing.assert_allclose(m.weights, ref.x.reshape(4, 3), atol=5e-6)
            np.testing.assert_allclose(
                m.decision_function(x), ker.decision_function(x @ x.T), atol=2e-5
            )
            np.testing.assert_allclose(m.weights, x.T @ m.dual, atol=2e-14)
            assert abs(fun(m.weights.ravel())[0] - ref.fun) < 1e-9

    def test_collapse_is_fixed(self):
        for precision, dtype, gap in [
            ("float32", np.float32, 20),
            ("float64", np.float32, 20),
            ("float64", np.float64, 50),
        ]:
            s = Solver(precision, dtype=dtype)
            z = np.array([[gap / 2, -gap / 2, -gap / 2]])
            a = s.coordinate_update(z, [0])
            assert a[0, 0] > 0
            assert np.all(a[0, 1:] < 0)
            np.testing.assert_allclose(a[0, 0], -np.sum(a[0, 1:]), rtol=1e-7)
            assert s.loss_and_grad(z, [0])[0][0] > 0
        # This block is reachable from a feasible ordinary SDCA state:
        # X=[1,32], A=[[0,0,0],[.375,-.1875,-.1875]], W=X.T@A.
        for precision in ["float32", "float64"]:
            s = Solver(precision, dtype=np.float32)
            a = s.coordinate_update([[12.0, -6.0, -6.0]], [0])
            assert a[0, 0] > 0
            assert float(a.sum()) == 0

    def test_invalid_input(self):
        s = Solver()
        for y in [[0.1], [[1.0, 0.0]], [-1], [3]]:
            with pytest.raises(ValueError):
                s.loss_and_grad([[1, 2, 3]], y)
        with pytest.raises(ValueError, match="nonnegative"):
            s.fit(-np.eye(2), [0, 1], kernel=True)
        # All-zero rows take a block of size C, which must fit float32 storage;
        # the check compares C exactly, without an overflowing cast.
        for precision in ["float32", "float64"]:
            for C in [1e100, 1e-100]:
                for x, kernel in [(np.zeros((1, 1)), False), (np.zeros((1, 1)), True)]:
                    with pytest.raises(ValueError, match="normal in the data dtype"):
                        with warnings.catch_warnings():
                            warnings.simplefilter("error")
                            Solver(precision, dtype=np.float32).fit(
                                x, [0], num_classes=2, C=C, kernel=kernel
                            )
        with pytest.raises(ValueError):
            s.coordinate_update([[1.0, float("nan")]], [0])
        with pytest.raises(ValueError):
            s.fit(np.eye(2), [0, 1], max_epoch=1.5)  # pyright: ignore[reportArgumentType]

    def test_labels_and_kernels_are_checked_without_guessing(self):
        s, x = Solver(), np.eye(3)
        # Float labels, and a (n,1) integer column for cross_entropy, are rejected.
        with pytest.raises(ValueError, match="integer labels"):
            s.fit(x, np.array([0.0, 1.0, 2.0]))
        with pytest.raises(ValueError, match="ambiguous"):
            s.fit(x, np.array([[0], [1], [2]]))
        assert s.fit(x, np.array([0, 1, 2]), max_epoch=1).objective == "cross_entropy"
        assert s.fit(x, [[0], [1], [2]], max_epoch=1).objective == "multilabel_cross_entropy"
        # The library checks a training kernel's symmetry and |K_ij| <= sqrt(K_ii K_jj).
        for gram, message in [
            ([[1.0, 0.5], [0.4, 1.0]], "symmetric"),
            ([[1.0, 2.0], [2.0, 1.0]], "positive semidefinite"),
            ([[1.0, np.nan], [np.nan, 1.0]], "finite"),
            (np.ones((2, 3)), "square"),
        ]:
            with pytest.raises(ValueError, match=message):
                s.fit(np.array(gram), [0, 1], kernel=True)
        # A Gram matrix rounded asymmetrically (float32 products) is accepted.
        rng = np.random.default_rng(3)
        f = rng.normal(size=(40, 200)).astype(np.float32)
        gram = np.empty((40, 40), np.float32)
        for i in range(40):
            gram[i] = f @ f[i]
        model = Solver(dtype=np.float32).fit(gram, np.arange(40) % 3, kernel=True, tol=1e-4)
        assert model.status == "solved"

    def test_label_sets_from_masks_and_lists(self):
        mask = np.array([[1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0]], bool)
        lists = [[2, 0], [3], [1, 0]]
        for y in [mask, np.asfortranarray(mask), mask.tolist(), lists]:
            labels, offsets = _label_sets(y, 3, 4)
            np.testing.assert_array_equal(labels, [0, 2, 3, 0, 1])
            np.testing.assert_array_equal(offsets, [0, 2, 3, 5])
            assert (labels.dtype, offsets.dtype) == (np.int64, np.int64)
        for y, message in [(mask[:2], "shape"), (mask & False, "between 1 and 3")]:
            with pytest.raises(ValueError, match=message):
                _label_sets(y, 3, 4)

    def test_center_and_history_capacity(self):
        s = Solver()
        x = np.eye(3)
        center = np.array([[2.0, 0.0, -1.0], [1.0, 0.0, 2.0], [0.0, 2.0, -1.0]])
        for epochs in [0, 1, 2, 3, 7]:
            m = s.fit(x, [0, 2, 1], center=center, max_epoch=epochs, eval_epoch=3, tol=0)
            assert m.history["epoch"][-1] == m.epochs
            assert m.weights is not None
            np.testing.assert_allclose(m.weights, center + x.T @ m.dual, atol=1e-15)

    def test_names_and_model_ergonomics(self):
        import libsdca

        x = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, -2.0], [1.5, 0.5]])
        y = np.array([0, 1, 2, 0])
        mask = np.array(
            [[True, True, False], [False, True, False], [False, False, True], [True, False, True]]
        )
        # Canonical names only; smoothing is gamma, not a name.
        for name in ("ce", "softmax", "svm", "topk_hinge_alpha_smooth"):
            with pytest.raises(ValueError, match="Unknown"):
                libsdca.loss_and_grad(x @ np.ones((2, 3)), y, name)
            with pytest.raises(ValueError, match="Unknown"):
                libsdca.fit(x, y, name)
        # multiclass_hinge is topk_hinge_alpha (and beta) at k = 1, whatever k says.
        z = x @ np.array([[1.0, -1.0, 0.5], [0.0, 2.0, -1.0]])
        for gamma in (0.0, 0.5):
            for other in ("topk_hinge_alpha", "topk_hinge_beta"):
                np.testing.assert_array_equal(
                    libsdca.loss_and_grad(z, y, "multiclass_hinge", k=2, gamma=gamma)[1],
                    libsdca.loss_and_grad(z, y, other, gamma=gamma)[1],
                )
        model = libsdca.fit(x, y, "cross_entropy")
        assert model.objective == "cross_entropy"
        assert not model.multilabel
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
        ml_model = libsdca.fit(x, mask, "multilabel_cross_entropy", max_epoch=10)
        assert ml_model.multilabel
        np.testing.assert_allclose(ml_model.predict_proba(x).sum(axis=1), 1.0, atol=1e-15)
        svm_model = libsdca.fit(x, y, "multiclass_hinge", max_epoch=10)
        with pytest.raises(ValueError, match="cross-entropy"):
            svm_model.predict_proba(x)
        with pytest.raises(ValueError, match="cross-entropy"):
            svm_model.predict_log_proba(x)


class ProjectionPrecisionTests:
    """projection_precision: float64 by default, extended agrees to rounding. Shrinking is off:
    its keep-or-drop tests would turn rounding differences (long double is double on Apple
    arm64) into different paths to the optimum."""

    def setup_method(self):
        rng = np.random.default_rng(23)
        self.x = rng.normal(size=(30, 5))
        self.labels = [sorted({i % 4, (i * 7 + 1) % 4}) for i in range(30)]

    def test_default_and_extended_agree_on_a_multilabel_hinge_fit(self):
        solver = Solver()
        fits = {
            mode: solver.fit(
                self.x, self.labels, "multilabel_hinge", gamma=0.5, tol=1e-10, shrinking="off", **kw
            )
            for mode, kw in [
                ("default", {}),
                ("float64", {"projection_precision": "float64"}),
                ("extended", {"projection_precision": "extended"}),
            ]
        }
        np.testing.assert_array_equal(fits["default"].weights, fits["float64"].weights)
        np.testing.assert_array_equal(fits["default"].dual, fits["float64"].dual)
        extended, float64 = fits["extended"].weights, fits["float64"].weights
        assert extended is not None and float64 is not None
        np.testing.assert_allclose(extended, float64, atol=1e-12)
        np.testing.assert_allclose(fits["extended"].dual, fits["float64"].dual, atol=1e-12)

    def test_losses_and_steps_agree(self):
        z = self.x[:, :4]
        mask = np.zeros((30, 4), bool)
        for i, row in enumerate(self.labels):
            mask[i, row] = True
        solver = Solver()
        loss = {
            mode: solver.loss_and_grad(
                z, mask, "multilabel_hinge", gamma=0.5, projection_precision=mode
            )
            for mode in ("float64", "extended")
        }
        default = solver.loss_and_grad(z, mask, "multilabel_hinge", gamma=0.5)
        np.testing.assert_array_equal(default[0], loss["float64"][0])
        for got, ref in zip(loss["extended"], loss["float64"], strict=True):
            np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-15)
        step = {
            mode: solver.coordinate_update(
                z, mask, "multilabel_hinge", gamma=0.5, projection_precision=mode
            )
            for mode in ("float64", "extended")
        }
        np.testing.assert_allclose(step["extended"], step["float64"], rtol=1e-13, atol=1e-15)

    def test_invalid_names_are_rejected(self, subtests: pytest.Subtests):
        solver, z, y = Solver(), self.x[:, :3], np.arange(30) % 3
        message = "projection_precision must be 'float64' or 'extended'"
        invalid: list[Any] = ["double", "Float64", "fp64", "long double", None, 0]
        for mode in invalid:
            with subtests.test(mode=mode):
                with pytest.raises(ValueError, match=message):
                    solver.fit(self.x, y, projection_precision=mode)
                with pytest.raises(ValueError, match=message):
                    solver.loss_and_grad(z, y, projection_precision=mode)
                with pytest.raises(ValueError, match=message):
                    solver.coordinate_update(z, y, projection_precision=mode)

    def test_the_abi_rejects_unknown_codes(self):
        solver, x, y = Solver(), np.ones((1, 1)), np.zeros(1, np.int64)
        w, a = np.zeros((1, 2)), np.zeros((1, 2))
        hist = np.zeros((2, len(HISTORY)))
        for code in (-1, 2):
            options = _options(gamma=1.0, max_epoch=0, eval_epoch=1, projection_precision=code)
            args = (ct.byref(options), 1, 1, 2, _ptr(x), _ptr(y), None, None, _ptr(w), _ptr(a))
            with pytest.raises(ValueError, match="Invalid projection precision"):
                solver._call("sdca_fit", *args, _ptr(hist), 2, ct.byref(_fit_result()))
