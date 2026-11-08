"""A scikit-learn classifier trained by libsdca.fit (needs scikit-learn)."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data as _validate_data

from ._core import (
    MULTILABEL,
    OBJECTIVES,
    FloatArray,
    Model,
    _log_softmax,
    _solver,
)

__all__ = ["ConvergenceWarning", "SDCAClassifier"]

# Untyped in scikit-learn; inference would type X and y as str from their defaults.
validate_data: Callable[..., Any] = _validate_data


class SDCAClassifier(ClassifierMixin, BaseEstimator):
    """Linear or kernel classifier trained by SDCA.

    Minimizes 0.5*||W||^2 + C*sum_i loss(W.T x_i, y_i) for a single-label
    objective; the multilabel objectives take label sets through libsdca.fit.
    There is no sample_weight or class_weight.

    Args:
        objective: A single-label name of libsdca.OBJECTIVES.
        C: Loss weight, finite and positive.
        k: Top-k parameter in [1, n_classes-1].
        gamma: Hinge smoothing, finite and >= 0 (0: nonsmooth).
        tol: Relative duality gap to stop at.
        max_epoch: Epoch limit.
        eval_epoch: Epochs between evaluations of the gap (up to two more
            fall between where its trend reaches tol).
        recompute_primal: Rebuild W = X.T @ A at each evaluation (features).
        projection_precision: "float64" or "extended", as in libsdca.fit; the
            single-label objectives currently solve alike in both.
        shrinking: "auto", "off", "heuristic", "safe" or "both", as in
            libsdca.fit.
        kernel: "linear" for features, or "precomputed" for Gram matrices
            (fit takes K(train, train), predict K(test, train)).
        precision: One of libsdca.PRECISIONS, or None for the default of the
            data dtype.
        warm_start: Start a refit from the previous solution, its duals scaled
            by C / previous C (the feasible sets scale with C); the data must be
            the same (only the shapes and classes are checked). A changed
            objective or k starts cold.

    Attributes:
        classes_: The class labels (n_classes,).
        coef_: Weights (n_classes, n_features); None for precomputed kernels.
        dual_coef_: Dual variables (n_classes, n_samples).
        n_features_in_: Number of features (training samples for kernels).
        n_iter_: Epochs run.
        status_: Solver status; "solved" certifies the duality gap.
        history_: The evaluation history (see libsdca.HISTORY).
        model_: The libsdca.Model of the last fit.
    """

    def __init__(
        self,
        objective: str = "cross_entropy",
        *,
        C: float = 1.0,
        k: int = 1,
        gamma: float = 0.0,
        tol: float = 1e-3,
        max_epoch: int = 1000,
        eval_epoch: int = 10,
        recompute_primal: bool = True,
        projection_precision: str = "float64",
        kernel: str = "linear",
        precision: object = None,
        warm_start: bool = False,
        shrinking: str = "auto",
    ) -> None:
        self.objective = objective
        self.C = C
        self.k = k
        self.gamma = gamma
        self.tol = tol
        self.max_epoch = max_epoch
        self.eval_epoch = eval_epoch
        self.recompute_primal = recompute_primal
        self.projection_precision = projection_precision
        self.kernel = kernel
        self.precision = precision
        self.warm_start = warm_start
        self.shrinking = shrinking

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.kernel == "precomputed"
        return tags

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> SDCAClassifier:
        """Train on X (n_samples, n_features), or a Gram matrix, and labels y."""
        objective = "topk_hinge_beta" if self.objective == "topk_hinge" else self.objective
        if objective not in OBJECTIVES or objective in MULTILABEL:
            raise ValueError(f"objective must be a single-label objective, not {self.objective!r}")
        if self.kernel not in ("linear", "precomputed"):
            raise ValueError("kernel must be 'linear' or 'precomputed'")
        features = getattr(self, "n_features_in_", None)
        x, labels = validate_data(self, X, y, dtype=[np.float64, np.float32])
        check_classification_targets(labels)
        classes, encoded = np.unique(labels, return_inverse=True)
        if len(classes) < 2:
            raise ValueError(f"Training needs at least two classes; got {len(classes)} class")
        precomputed = self.kernel == "precomputed"
        if precomputed and x.shape[0] != x.shape[1]:
            raise ValueError("A precomputed training kernel must be square")
        initial = None
        previous: Model | None = getattr(self, "model_", None)
        known = np.asarray(getattr(self, "classes_", ()))
        k = 1 if objective == "multiclass_hinge" else self.k
        if (
            self.warm_start
            and previous is not None
            and np.array_equal(known, classes)
            and previous.objective == objective
            and previous.params["k"] == k
            and features == x.shape[1]
            and previous.dual.shape == (len(x), len(classes))
        ):
            # The feasible set of every objective is C times a fixed set.
            initial = previous.dual * (float(self.C) / float(previous.params["C"]))
        model = _solver(x, self.precision).fit(
            x,
            encoded,
            objective,
            C=self.C,
            k=self.k,
            gamma=self.gamma,
            num_classes=len(classes),
            max_epoch=self.max_epoch,
            eval_epoch=self.eval_epoch,
            tol=self.tol,
            recompute_primal=self.recompute_primal,
            projection_precision=self.projection_precision,
            kernel=precomputed,
            initial_dual=initial,
            shrinking=self.shrinking,
        )
        self.model_ = model
        self.classes_ = classes
        self.coef_ = None if model.weights is None else model.weights.T
        self.dual_coef_ = model.dual.T
        self.n_iter_ = model.epochs
        self.status_ = model.status
        self.history_ = model.history
        if model.status != "solved":
            warnings.warn(
                f"Training stopped with status {model.status!r} before reaching tol; "
                "increase max_epoch or tol",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    def _scores(self, X: npt.ArrayLike) -> FloatArray:
        check_is_fitted(self)
        x = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        if self.coef_ is None:
            return x @ self.dual_coef_.T
        return x @ self.coef_.T

    def decision_function(self, X: npt.ArrayLike) -> FloatArray:
        """Class scores (n_samples, n_classes); for two classes the margins
        scores[:, 1] - scores[:, 0] of shape (n_samples,)."""
        scores = self._scores(X)
        return scores[:, 1] - scores[:, 0] if len(self.classes_) == 2 else scores

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """The class with the largest score."""
        scores = self._scores(X)
        return self.classes_[scores.argmax(axis=1)]

    def _has_proba(self) -> bool:
        return self.objective == "cross_entropy"

    @available_if(_has_proba)
    def predict_proba(self, X: npt.ArrayLike) -> FloatArray:
        """Class probabilities, the softmax of the scores (objective="cross_entropy")."""
        scores = self._scores(X).astype(np.float64)
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    @available_if(_has_proba)
    def predict_log_proba(self, X: npt.ArrayLike) -> FloatArray:
        """Logarithms of predict_proba, without rounding the largest class to 0."""
        return _log_softmax(self._scores(X).astype(np.float64))
