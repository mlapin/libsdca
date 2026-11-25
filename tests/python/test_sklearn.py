"""The module functions, the package layout and the scikit-learn estimator."""

import importlib.util
import warnings
from typing import Any

import libsdca
import numpy as np
import pytest
from child_process import run_python
from conformance import references as oracle
from libsdca import CrossEntropy, TopKCrossEntropy, TopKHinge

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
                ref = libsdca.fit(x.astype(dtype), y, tol=1e-6, precision=np.dtype(dtype).name)
                assert model.weights is not None
                assert model.weights.dtype == dtype
                np.testing.assert_array_equal(model.weights, ref.weights)
                z = x.astype(dtype) @ model.weights.T
                values, grad = libsdca.loss_and_grad(z, y, TopKCrossEntropy(2))
                expected = libsdca.loss_and_grad(
                    z, y, TopKCrossEntropy(2), precision=np.dtype(dtype).name
                )
                np.testing.assert_array_equal(values, expected[0])
                assert grad.dtype == dtype
                assert libsdca.coordinate_update(z, y, C=0.5).dtype == dtype
        for p in ("float32", np.float32):
            fast = libsdca.fit(x.astype(np.float32), y, precision=p, tol=1e-4)
            assert fast.status == "solved"
            assert fast.weights is not None
            assert fast.weights.dtype == np.float32
        for p in ("float64", float, np.float64):
            exact32 = libsdca.fit(x.astype(np.float32), y, precision=p, tol=1e-6)
            assert exact32.status == "solved"
            assert exact32.weights is not None
            assert exact32.weights.dtype == np.float32
        # float64 data with float32 work is cast to float32 storage.
        cast = libsdca.fit(x, y, precision="float32", tol=1e-4)
        assert cast.weights is not None and cast.weights.dtype == np.float32
        np.testing.assert_array_equal(
            cast.weights, libsdca.fit(x.astype(np.float32), y, tol=1e-4).weights
        )
        # 16-bit floats are cast to float32, integers to float64.
        half = libsdca.fit(x.astype(np.float16), y, max_epoch=2)
        assert half.dual.dtype == np.float32
        assert libsdca.fit(np.round(x).astype(int), y, max_epoch=2).dual.dtype == np.float64
        for spelling in ("half", "mixed", "fp32", "single", "double"):
            with pytest.raises(ValueError, match="precision"):
                libsdca.fit(x, y, precision=spelling)
        with pytest.raises(TypeError, match="precision"):
            libsdca.fit(x, y, precision=32)  # pyright: ignore[reportArgumentType]

    def test_options_are_keyword_only_and_results_frozen(self):
        x, y = blobs()
        # Deliberately invalid calls and assignments.
        with pytest.raises(TypeError):
            libsdca.fit(x, y, CrossEntropy(), 1.0)  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError):
            libsdca.loss_and_grad(x[:, :3], y, TopKCrossEntropy(2), "float64")  # pyright: ignore[reportCallIssue]
        model = libsdca.fit(x, y)
        with pytest.raises(AttributeError):
            model.status = "solved"  # pyright: ignore[reportAttributeAccessIssue]
        with pytest.raises(AttributeError):
            TopKCrossEntropy(2).k = 3  # pyright: ignore[reportAttributeAccessIssue]
        assert list(libsdca.LOSSES)[0] == "cross_entropy"
        assert isinstance(libsdca.MULTILABEL, frozenset)

    def test_native_invalid_arguments_raise_value_error(self):
        with pytest.raises(ValueError, match="k must"):
            libsdca.loss_and_grad([[0.0, 1.0, 2.0]], [0], TopKHinge(3))

    def test_import_loads_no_framework(self):
        code = (
            "import sys, libsdca; from libsdca import *; libsdca.fit; "
            "assert 'torch' not in sys.modules and 'sklearn' not in sys.modules "
            "and 'jax' not in sys.modules"
        )
        run_python("-c", code, check=True)
        with pytest.raises(AttributeError):
            libsdca.not_a_name  # noqa: B018  # pyright: ignore[reportAttributeAccessIssue]
        for name in ("CrossEntropyLoss", "SDCAClassifier", "Solver", "distillation_loss_and_grad"):
            assert not hasattr(libsdca, name), name

    def test_threads_run_in_parallel_with_identical_results(self):
        from concurrent.futures import ThreadPoolExecutor

        x, y = blobs(n=300)
        loss = TopKCrossEntropy(2)
        expected = libsdca.fit(x, y, loss, tol=1e-8).weights
        with ThreadPoolExecutor(4) as pool:
            models = list(pool.map(lambda _: libsdca.fit(x, y, loss, tol=1e-8), range(8)))
        for model in models:
            np.testing.assert_array_equal(model.weights, expected)


@pytest.mark.skipif(not HAVE_SKLEARN, reason="scikit-learn is not installed")
class ClassifierTests:
    def setup_method(self):
        from libsdca.sklearn import ConvergenceWarning, SDCAClassifier

        self.Classifier, self.Warning = SDCAClassifier, ConvergenceWarning

    def test_failed_refit_preserves_feature_count(self):
        x, y = blobs()
        estimator = self.Classifier().fit(x, y)
        expected = estimator.predict(x)
        with pytest.raises(ValueError, match="at least two classes"):
            estimator.fit(x[:, :1], np.zeros(len(y)))
        np.testing.assert_array_equal(estimator.predict(x), expected)

    def test_check_estimator(self, subtests: pytest.Subtests):
        from sklearn.utils.estimator_checks import check_estimator

        for estimator in [
            self.Classifier(),
            self.Classifier("topk_hinge", k=1, variant="alpha", gamma=0.5),
        ]:
            with subtests.test(estimator=estimator), warnings.catch_warnings():
                warnings.simplefilter("ignore", self.Warning)
                check_estimator(estimator)

    def test_check_estimator_precomputed(self, subtests: pytest.Subtests):
        from sklearn.exceptions import SkipTestWarning
        from sklearn.utils.estimator_checks import check_estimator

        # These checks pass matrices that are no Gram matrix, which fit rejects.
        invalid = {
            "check_positive_only_tag_during_fit": "a mean-shifted kernel is not PSD",
            "check_estimators_dtypes": "an integer-truncated kernel is not PSD",
        }
        for loss in ("cross_entropy", "multiclass_hinge"):
            expected = dict(invalid)
            if loss == "cross_entropy":
                expected["check_decision_proba_consistency"] = "fits a non-square matrix"
            estimator = self.Classifier(loss, kernel="precomputed")
            with subtests.test(loss=loss), warnings.catch_warnings():
                warnings.simplefilter("ignore", self.Warning)
                warnings.simplefilter("ignore", SkipTestWarning)
                check_estimator(estimator, expected_failed_checks=expected)

    def test_labels_scores_and_probabilities(self):
        x, y = blobs()
        names = np.array(["cat", "dog", "emu"])[y]
        clf = self.Classifier(C=10.0, tol=1e-8).fit(x, names)
        np.testing.assert_array_equal(clf.classes_, ["cat", "dog", "emu"])
        assert clf.status_ == "solved"
        assert clf.coef_.shape == (3, 4)
        assert clf.dual_coef_.shape == (3, len(x))
        np.testing.assert_allclose(clf.decision_function(x), x @ clf.coef_.T, rtol=1e-12)
        assert clf.score(x, names) > 0.9
        proba = clf.predict_proba(x)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        np.testing.assert_allclose(np.exp(clf.predict_log_proba(x)), proba, rtol=1e-12)
        np.testing.assert_array_equal(clf.classes_[proba.argmax(axis=1)], clf.predict(x))
        assert not hasattr(self.Classifier("topk_cross_entropy", k=2), "predict_proba")

    def test_binary_coef_matches_the_decision_function(self):
        x, y = blobs(classes=2)
        clf = self.Classifier(tol=1e-8).fit(x, y)
        assert clf.coef_.shape == (1, 4)
        np.testing.assert_allclose(clf.decision_function(x), x @ clf.coef_[0], rtol=1e-12)
        weights = clf.model_.weights
        assert weights is not None
        np.testing.assert_array_equal(clf.coef_[0], weights[1] - weights[0])

    def test_log_proba_keeps_the_confident_class_below_zero(self):
        clf = self.Classifier(tol=1e-8).fit([[-1.0], [1.0]], [0, 1])
        x = [[50 / clf.coef_[0, 0]]]
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
        assert not hasattr(kernel, "coef_")
        with pytest.raises(AttributeError):
            kernel.coef_  # noqa: B018
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
        for loss, params in [
            ("cross_entropy", {}),
            ("multiclass_hinge", {"gamma": 0.1}),
            ("topk_hinge", {"k": 1, "gamma": 0.1}),
        ]:
            with subtests.test(loss=loss):
                options: dict[str, Any] = dict(tol=1e-6, max_iter=5000, **params)
                clf = self.Classifier(loss, warm_start=True, **options)
                cold = self.Classifier(loss, **options)
                for C in [1.0, 0.1, 0.01, 1.0]:
                    clf.set_params(C=C).fit(x, y)
                    assert clf.status_ == "solved"
                    reference = cold.set_params(C=C).fit(x, y)
                    np.testing.assert_allclose(
                        clf.decision_function(x), reference.decision_function(x), atol=1e-2
                    )
        # Another k starts cold instead of from an infeasible block.
        clf = self.Classifier("topk_hinge", k=1, gamma=0.1, warm_start=True).fit(x, y)
        assert clf.set_params(k=2).fit(x, y).status_ == "solved"

    def test_warm_start_with_changed_training_labels(self):
        x, y = blobs()
        warm = self.Classifier(warm_start=True, random_state=0).fit(x, y)
        warm.fit(x, y[::-1])
        cold = self.Classifier(random_state=0).fit(x, y[::-1])
        np.testing.assert_array_equal(warm.dual_coef_, cold.dual_coef_)

    def test_validation_and_warnings(self):
        from sklearn.exceptions import NotFittedError

        x, y = blobs()
        with pytest.raises(ValueError, match="loss must be one of"):
            self.Classifier("multilabel_hinge").fit(x, y)
        with pytest.raises(ValueError, match="kernel"):
            self.Classifier(kernel="rbf").fit(x, y)
        with pytest.raises(NotFittedError):
            self.Classifier().predict(x)
        with pytest.warns(self.Warning):
            self.Classifier(max_iter=1, eval_epoch=1, tol=1e-12).fit(x, y)
        # Parameters a loss does not take are errors, not ignored.
        for loss, params in [
            ("cross_entropy", {"k": 2}),
            ("cross_entropy", {"gamma": 0.5}),
            ("topk_cross_entropy", {"k": 2, "gamma": 0.5}),
            ("multiclass_hinge", {"k": 2}),
            ("multiclass_hinge", {"variant": "alpha"}),
            ("topk_hinge", {}),
        ]:
            with pytest.raises(TypeError, match="k|gamma|variant"):
                self.Classifier(loss, **params).fit(x, y)
        with pytest.raises(TypeError):
            self.Classifier("topk_cross_entropy", k=2.0).fit(x, y)  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValueError, match="n_jobs"):
            self.Classifier(n_jobs=0).fit(x, y)
        # A precomputed kernel is square with a nonnegative diagonal, checked
        # in full before any sample is used.
        kernel = self.Classifier(kernel="precomputed")
        with pytest.raises(ValueError, match="square"):
            kernel.fit(np.c_[np.eye(3), np.ones(3)], [0, 1, 0])
        with pytest.raises(ValueError, match="nonnegative"):
            kernel.fit(np.diag([1.0, -1.0, 1.0]), [0, 1, 0])
        # Another feature count starts cold (random_state fixes the order of the epochs).
        warm = self.Classifier(warm_start=True, random_state=0).fit(x, y)
        epochs = warm.fit(x[:, :3], y).n_iter_
        assert epochs == self.Classifier(random_state=0).fit(x[:, :3], y).n_iter_
        with pytest.raises(TypeError):
            self.Classifier(mode="strict")  # pyright: ignore[reportCallIssue]  # deliberately invalid

    def test_random_state_and_n_jobs(self):
        x, y = blobs()
        options: dict[str, Any] = dict(loss="topk_hinge", k=2, tol=0, max_iter=3)
        a = self.Classifier(random_state=1, **options).fit(x, y)
        b = self.Classifier(random_state=1, n_jobs=-1, **options).fit(x, y)
        c = self.Classifier(random_state=2, n_jobs=2, **options).fit(x, y)
        np.testing.assert_array_equal(a.dual_coef_, b.dual_coef_)
        assert not np.array_equal(a.dual_coef_, c.dual_coef_)

    def test_pipelines_and_search(self):
        from sklearn.model_selection import GridSearchCV, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        x, y = blobs(n=120)
        with warnings.catch_warnings():
            warnings.simplefilter("error", self.Warning)
            pipeline = make_pipeline(StandardScaler(), self.Classifier("topk_cross_entropy", k=2))
            scores = cross_val_score(pipeline, x, y, cv=3)
            grid = {"loss": ["multiclass_hinge", "topk_hinge"], "k": [None, 1]}
            search = GridSearchCV(self.Classifier(), {"C": [0.1, 10.0]}, cv=3).fit(x, y)
            # The grid holds combinations a loss does not take: they fail to fit (nan score).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hinges = GridSearchCV(self.Classifier(gamma=0.1), grid, cv=3).fit(x, y)
        assert scores.mean() > 0.8
        assert search.best_params_["C"] in (0.1, 10.0)
        assert hinges.best_params_ in (
            {"loss": "multiclass_hinge", "k": None},
            {"loss": "topk_hinge", "k": 1},
        )

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
        for loss, reference in references:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reference.fit(x, y)
            ours = self.Classifier(loss, tol=1e-10, max_iter=100000).fit(x, y)
            assert ours.status_ == "solved"
            mine = float(oracle.primal(oracle.named(loss), x, ours.coef_.T, y, 1.0))
            theirs = float(oracle.primal(oracle.named(loss), x, reference.coef_.T, y, 1.0))
            # The guaranteed optimum is no higher than any other solver's, and close to it.
            assert mine <= theirs * (1 + 1e-9), loss
            assert theirs - mine <= 1e-4 * mine, loss
            assert np.mean(ours.predict(x) == reference.predict(x)) >= 0.98, loss

    def test_non_contiguous_integer_labels(self):
        x, y = blobs()
        expected = libsdca.fit(x, y, TopKHinge(2, gamma=0.5), tol=1e-8)
        for classes in (np.array([-5, 2, 9]), np.array([2**40, 3, 7], np.int64)):
            clf = self.Classifier("topk_hinge", k=2, gamma=0.5, tol=1e-8, random_state=0)
            clf.fit(np.asfortranarray(x), classes[y])
            np.testing.assert_array_equal(clf.classes_, np.sort(classes))
            # classes_ is sorted: the score columns permute.
            order = np.argsort(classes)
            scores = expected.decision_function(x)[:, order]
            np.testing.assert_allclose(clf.decision_function(x), scores, atol=1e-3)
            np.testing.assert_array_equal(clf.predict(x), classes[expected.predict(x)])

    def test_loss_names(self):
        x, y = blobs()
        clf = self.Classifier("cross_entropy", tol=1e-7).fit(x, y)
        assert clf.model_.loss == CrossEntropy()
        np.testing.assert_allclose(clf.predict_proba(x).sum(axis=1), 1.0)
        for name, variant in (
            ("multiclass_hinge", None),
            ("topk_hinge", "alpha"),
            ("topk_hinge", None),
        ):
            for gamma in (0.0, 0.5):
                k = None if name == "multiclass_hinge" else 2
                c1 = self.Classifier(name, k=k, variant=variant, gamma=gamma, tol=0, max_iter=20)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", self.Warning)
                    c1.fit(x, y)
                assert c1.model_.loss.name == name
                expected = variant or ("beta" if name == "topk_hinge" else None)
                assert getattr(c1.model_.loss, "variant", None) == expected
                assert not hasattr(c1, "predict_proba")
        # No aliases; the variant and the smoothing are parameters.
        for name in ("ce", "softmax", "svm", "topk_hinge_alpha", "topk_hinge_beta_smooth"):
            with pytest.raises(ValueError, match="loss must be one of"):
                self.Classifier(name).fit(x, y)
