"""The module functions, the package layout and the scikit-learn estimator."""

import importlib.util
import warnings
from typing import Any

import libsdca
import numpy as np
import pytest
from child_process import run_python
from libsdca._core import Solver
from loss_references import primal

HAVE_SKLEARN = importlib.util.find_spec("sklearn") is not None


def blobs(n: int = 90, classes: int = 3, seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.arange(n) % classes
    x = rng.normal(size=(n, 4)) + 3 * np.eye(classes, 4)[y]
    return x, y


class ModuleFunctionTests:
    def test_precision_follows_the_dtype(self, subtests: pytest.Subtests):
        x, y = blobs()
        for dtype in (np.float64, np.float32):
            with subtests.test(dtype=dtype):
                model = libsdca.fit(x.astype(dtype), y, tol=1e-6)
                ref = Solver("float64", dtype=dtype).fit(x.astype(dtype), y, tol=1e-6)
                assert model.weights is not None
                assert model.weights.dtype == dtype
                np.testing.assert_array_equal(model.weights, ref.weights)
                z = x.astype(dtype) @ model.weights
                values, grad = libsdca.loss_and_grad(z, y, "topk_cross_entropy", k=2)
                expected = Solver("float64", dtype=dtype).loss_and_grad(
                    z, y, "topk_cross_entropy", k=2
                )
                np.testing.assert_array_equal(values, expected[0])
                assert grad.dtype == dtype
                assert libsdca.coordinate_update(z, y, C=0.5).dtype == dtype
        for p in ("float32", "fp32", "single", np.float32):
            fast = libsdca.fit(x.astype(np.float32), y, precision=p, tol=1e-4)
            assert fast.status == "solved"
            assert fast.weights is not None
            assert fast.weights.dtype == np.float32
        for p in ("float64", "fp64", "double", float, np.float64):
            exact32 = libsdca.fit(x.astype(np.float32), y, precision=p, tol=1e-6)
            assert exact32.status == "solved"
            assert exact32.weights is not None
            assert exact32.weights.dtype == np.float32
        with pytest.raises(ValueError, match="precision"):
            libsdca.fit(x, y, precision="half")
        with pytest.raises(ValueError, match="precision"):
            libsdca.fit(x, y, precision="mixed")
        with pytest.raises(ValueError, match="precision"):
            libsdca.fit(x.astype(np.float64), y, precision="float32")

    def test_options_are_keyword_only_and_results_frozen(self):
        x, y = blobs()
        # Deliberately invalid calls and assignments.
        with pytest.raises(TypeError):
            libsdca.fit(x, y, "cross_entropy", 1.0)  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError):
            libsdca.loss_and_grad(x[:, :3], y, "topk_cross_entropy", 2)  # pyright: ignore[reportCallIssue]
        model = libsdca.fit(x, y)
        with pytest.raises(AttributeError):
            model.status = "solved"  # pyright: ignore[reportAttributeAccessIssue]
        assert isinstance(libsdca.LOSSES, tuple)
        assert isinstance(libsdca.MULTILABEL, frozenset)

    def test_native_invalid_arguments_raise_value_error(self):
        with pytest.raises(ValueError, match="exactly one target"):
            libsdca.loss_and_grad([[0.0, 1.0, 2.0]], [[True, True, False]], "topk_hinge_alpha")

    def test_import_loads_no_framework(self):
        code = (
            "import sys, libsdca; from libsdca import *; libsdca.fit; "
            "assert 'torch' not in sys.modules and 'sklearn' not in sys.modules"
        )
        run_python("-c", code, check=True)
        with pytest.raises(AttributeError):
            libsdca.not_a_name  # noqa: B018

    def test_threads_run_in_parallel_with_identical_results(self):
        from concurrent.futures import ThreadPoolExecutor

        x, y = blobs(n=300)
        expected = libsdca.fit(x, y, "topk_cross_entropy", k=2, tol=1e-8).weights
        with ThreadPoolExecutor(4) as pool:
            models = list(
                pool.map(lambda _: libsdca.fit(x, y, "topk_cross_entropy", k=2, tol=1e-8), range(8))
            )
        for model in models:
            np.testing.assert_array_equal(model.weights, expected)


@pytest.mark.skipif(not HAVE_SKLEARN, reason="scikit-learn is not installed")
class ClassifierTests:
    def setup_method(self):
        from libsdca.sklearn import ConvergenceWarning, SDCAClassifier

        self.Classifier, self.Warning = SDCAClassifier, ConvergenceWarning

    def test_check_estimator(self, subtests: pytest.Subtests):
        from sklearn.utils.estimator_checks import check_estimator

        for estimator in [self.Classifier(), self.Classifier("topk_hinge_alpha", gamma=0.5)]:
            with subtests.test(estimator=estimator), warnings.catch_warnings():
                warnings.simplefilter("ignore", self.Warning)
                check_estimator(estimator)

    def test_labels_scores_and_probabilities(self):
        x, y = blobs()
        names = np.array(["cat", "dog", "emu"])[y]
        clf = libsdca.SDCAClassifier(C=10.0, tol=1e-8).fit(x, names)
        np.testing.assert_array_equal(clf.classes_, ["cat", "dog", "emu"])
        assert clf.status_ == "solved"
        assert clf.coef_ is not None
        assert clf.coef_.shape == (3, 4)
        assert clf.dual_coef_.shape == (3, len(x))
        np.testing.assert_allclose(clf.decision_function(x), x @ clf.coef_.T, rtol=1e-12)
        assert clf.score(x, names) > 0.9
        proba = clf.predict_proba(x)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        np.testing.assert_allclose(np.exp(clf.predict_log_proba(x)), proba, rtol=1e-12)
        np.testing.assert_array_equal(clf.classes_[proba.argmax(axis=1)], clf.predict(x))
        assert not hasattr(self.Classifier("topk_cross_entropy", k=2), "predict_proba")

    def test_log_proba_keeps_the_confident_class_below_zero(self):
        clf = libsdca.SDCAClassifier(tol=1e-8).fit([[-1.0], [1.0]], [0, 1])
        assert clf.coef_ is not None
        x = [[50 / (clf.coef_[1, 0] - clf.coef_[0, 0])]]
        log_proba = clf.predict_log_proba(x)
        np.testing.assert_allclose(log_proba[0, 0], -50, rtol=1e-12)
        np.testing.assert_allclose(log_proba[0, 1], -np.exp(-50), rtol=1e-12)
        np.testing.assert_array_equal(log_proba, clf.model_.predict_log_proba(x))

    def test_zero_rows_binary_kernel_and_warm_start(self):
        x, y = blobs(classes=2)
        x[:5] = 0
        labels = np.where(y == 1, 1, -1)
        gram = x @ x.T
        linear = self.Classifier("multiclass_hinge", gamma=1.0, tol=1e-9).fit(x, labels)
        assert linear.status_ == "solved"
        # A zero row's block maximizes the dual alone: +1 at the label, -1 elsewhere.
        own = (linear.classes_[:, None] == labels[None, :5]).astype(float)
        np.testing.assert_allclose(linear.dual_coef_[:, :5], 2 * own - 1, rtol=1e-12)
        kernel = self.Classifier("multiclass_hinge", gamma=1.0, tol=1e-9, kernel="precomputed")
        kernel.fit(gram, labels)
        assert kernel.coef_ is None
        assert linear.decision_function(x).shape == (len(x),)
        np.testing.assert_allclose(
            kernel.decision_function(gram), linear.decision_function(x), atol=1e-3
        )
        warm = self.Classifier(
            "multiclass_hinge", gamma=1.0, tol=1e-9, eval_epoch=1, warm_start=True
        )
        warm.fit(x, labels)
        epochs = warm.n_iter_
        warm.fit(x, labels)
        assert warm.n_iter_ < epochs

    def test_warm_start_follows_c_down_and_up(self, subtests: pytest.Subtests):
        x, y = blobs()
        for objective in ["cross_entropy", "multiclass_hinge", "topk_hinge_beta"]:
            with subtests.test(objective=objective):
                options: dict[str, Any] = dict(k=1, gamma=0.1, tol=1e-6, max_epoch=5000)
                clf = self.Classifier(objective, warm_start=True, **options)
                cold = self.Classifier(objective, **options)
                for C in [1.0, 0.1, 0.01, 1.0]:
                    clf.set_params(C=C).fit(x, y)
                    assert clf.status_ == "solved"
                    reference = cold.set_params(C=C).fit(x, y)
                    np.testing.assert_allclose(
                        clf.decision_function(x), reference.decision_function(x), atol=1e-2
                    )
        # Another k starts cold instead of from an infeasible block.
        clf = self.Classifier("topk_hinge_beta", k=1, gamma=0.1, warm_start=True).fit(x, y)
        assert clf.set_params(k=2).fit(x, y).status_ == "solved"

    def test_validation_and_warnings(self):
        from sklearn.exceptions import NotFittedError

        x, y = blobs()
        with pytest.raises(ValueError, match="single-label"):
            self.Classifier("multilabel_hinge").fit(x, y)
        with pytest.raises(ValueError, match="kernel"):
            self.Classifier(kernel="rbf").fit(x, y)
        with pytest.raises(NotFittedError):
            self.Classifier().predict(x)
        with pytest.warns(self.Warning):
            self.Classifier(max_epoch=1, eval_epoch=1, tol=1e-12).fit(x, y)
        # A precomputed kernel is square with a nonnegative diagonal, checked
        # in full before any sample is used.
        kernel = self.Classifier(kernel="precomputed")
        with pytest.raises(ValueError, match="square"):
            kernel.fit(np.c_[np.eye(3), np.ones(3)], [0, 1, 0])
        with pytest.raises(ValueError, match="nonnegative"):
            kernel.fit(np.diag([1.0, -1.0, 1.0]), [0, 1, 0])
        warm = self.Classifier(warm_start=True).fit(x, y)
        epochs = warm.fit(x[:, :3], y).n_iter_
        assert epochs == self.Classifier().fit(x[:, :3], y).n_iter_
        # recompute_primal (default on) rebuilds W from the duals; the duals agree.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", self.Warning)
            rebuilt = self.Classifier(max_epoch=5, eval_epoch=5).fit(x, y)
            kept = self.Classifier(max_epoch=5, eval_epoch=5, recompute_primal=False).fit(x, y)
        assert rebuilt.get_params()["recompute_primal"]
        np.testing.assert_array_equal(rebuilt.dual_coef_, kept.dual_coef_)
        np.testing.assert_array_equal(rebuilt.predict(x), kept.predict(x))
        with pytest.raises(TypeError):
            self.Classifier(mode="strict")  # pyright: ignore[reportCallIssue]  # deliberately invalid
        # projection_precision: "float64" by default, "extended" accepted, others rejected.
        assert self.Classifier().get_params()["projection_precision"] == "float64"
        hinge: dict[str, Any] = dict(
            objective="multiclass_hinge", gamma=0.5, max_epoch=20, tol=1e-9
        )
        default = self.Classifier(**hinge).fit(x, y)
        extended = self.Classifier(projection_precision="extended", **hinge).fit(x, y)
        assert extended.coef_ is not None and default.coef_ is not None
        np.testing.assert_allclose(extended.coef_, default.coef_, atol=1e-12)
        with pytest.raises(ValueError, match="projection_precision"):
            self.Classifier(projection_precision="double").fit(x, y)

    def test_pipelines_and_search(self):
        from sklearn.model_selection import GridSearchCV, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        x, y = blobs(n=120)
        with warnings.catch_warnings():
            warnings.simplefilter("error", self.Warning)
            pipeline = make_pipeline(StandardScaler(), self.Classifier("topk_cross_entropy", k=2))
            scores = cross_val_score(pipeline, x, y, cv=3)
            search = GridSearchCV(self.Classifier(), {"C": [0.1, 10.0]}, cv=3).fit(x, y)
        assert scores.mean() > 0.8
        assert search.best_params_["C"] in (0.1, 10.0)

    def test_objectives_match_scikit_learn_solvers(self):
        # LogisticRegression and LinearSVC (Crammer-Singer) without intercepts minimize
        # 0.5 ||W||^2 + C sum loss of cross_entropy and multiclass_hinge.
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC

        x, y = load_iris(return_X_y=True)
        x = (x - x.mean(0)) / x.std(0)
        references: list[tuple[str, Any]] = [
            ("cross_entropy", LogisticRegression(fit_intercept=False, tol=1e-10, max_iter=10000)),
            (
                "multiclass_hinge",
                LinearSVC(
                    multi_class="crammer_singer", fit_intercept=False, tol=1e-8, max_iter=100000
                ),
            ),
        ]
        for objective, reference in references:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reference.fit(x, y)
            ours = self.Classifier(objective, tol=1e-10, max_epoch=100000).fit(x, y)
            assert ours.status_ == "solved" and ours.coef_ is not None
            mine = primal(x, ours.coef_.T, y, objective, 1.0)
            theirs = primal(x, reference.coef_.T, y, objective, 1.0)
            # The certified optimum is no higher than any other solver's, and close to it.
            assert mine <= theirs * (1 + 1e-9), objective
            assert theirs - mine <= 1e-4 * mine, objective
            assert np.mean(ours.predict(x) == reference.predict(x)) >= 0.98, objective

    def test_non_contiguous_integer_labels(self):
        x, y = blobs()
        expected = libsdca.fit(x, y, "topk_hinge_beta", k=2, gamma=0.5, tol=1e-8)
        for classes in (np.array([-5, 2, 9]), np.array([2**40, 3, 7], np.int64)):
            clf = self.Classifier("topk_hinge_beta", k=2, gamma=0.5, tol=1e-8)
            clf.fit(np.asfortranarray(x), classes[y])
            np.testing.assert_array_equal(clf.classes_, np.sort(classes))
            # classes_ is sorted: the score columns permute.
            order = np.argsort(classes)
            scores = expected.decision_function(x)[:, order]
            np.testing.assert_allclose(clf.decision_function(x), scores, atol=1e-3)
            np.testing.assert_array_equal(clf.predict(x), classes[expected.predict(x)])
            if np.array_equal(order, np.arange(3)):
                np.testing.assert_array_equal(clf.dual_coef_, expected.dual.T)

    def test_objective_names(self):
        from sklearn.utils.estimator_checks import check_estimator

        x, y = blobs()
        clf = self.Classifier("cross_entropy", tol=1e-7).fit(x, y)
        assert clf.model_.objective == "cross_entropy"
        np.testing.assert_allclose(clf.predict_proba(x).sum(axis=1), 1.0)
        np.testing.assert_allclose(
            np.exp(clf.predict_log_proba(x)), clf.predict_proba(x), rtol=1e-12
        )
        for name in ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta"):
            for gamma in (0.0, 0.5):
                c1 = self.Classifier(name, k=2, gamma=gamma, tol=0, max_epoch=20).fit(x, y)
                assert c1.model_.objective == name
                assert not hasattr(c1, "predict_proba")
        # No aliases; smoothing is gamma.
        for name in ("ce", "softmax", "svm", "topk_hinge_beta_smooth", "multilabel_hinge"):
            with pytest.raises(ValueError, match="single-label objective"):
                self.Classifier(name).fit(x, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", self.Warning)
            check_estimator(self.Classifier("cross_entropy"))
