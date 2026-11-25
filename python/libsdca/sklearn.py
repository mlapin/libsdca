"""A scikit-learn classifier trained by libsdca.fit (needs scikit-learn)."""

from __future__ import annotations

import hashlib
import os
import warnings
from collections.abc import Callable
from copy import copy
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data as _validate_data

from . import _frontend as _f
from ._native import (
    LOSSES,
    MULTILABEL,
    OBJECTIVES,
    FloatArray,
    Kernel,
    Loss,
    Model,
    _log_softmax,
    fit,
)

__all__ = ["ConvergenceWarning", "SDCAClassifier"]

# Untyped in scikit-learn; inference would type X and y as str from their defaults.
validate_data: Callable[..., Any] = _validate_data

# The single-label training objectives the classifier takes.
_SINGLE_LABEL = tuple(name for name in OBJECTIVES if name not in MULTILABEL)


class SDCAClassifier(ClassifierMixin, BaseEstimator):
    """Linear or kernel classifier trained by SDCA (libsdca.fit).

    Minimizes 0.5 * ||W||^2 + C * sum_i loss(W.T x_i, y_i) over W for a single-label loss.
    There is no intercept: append a constant feature to X for a regularized one. There is no
    sample_weight or class_weight; the multilabel losses take label sets through libsdca.fit.

    Args:
        loss: A single-label loss name: "cross_entropy", "topk_cross_entropy",
            "multiclass_hinge" or "topk_hinge".
        C: Loss weight, finite and positive.
        k: The top-k parameter of the top-k losses, an integer in [1, n_classes - 1]; None
            for the other losses (giving it to one of them raises ValueError).
        variant: The variant of topk_hinge, "alpha" or "beta" (None: "beta"); None for the
            other losses.
        gamma: The smoothing of the hinges, finite and >= 0 (None: 0, nonsmooth); None for
            the other losses.
        tol: Relative duality gap of status "solved".
        max_iter: Epoch limit (n_iter_ gives the epochs run).
        eval_epoch: Epochs between evaluations of the gap (0: only at the end).
        shrinking: "auto", "off", "heuristic", "safe" or "both", as in libsdca.fit (the
            hinge losses; the entropy losses take "auto" only).
        kernel: "linear" for features, or "precomputed" for Gram matrices (fit takes
            K(train, train), predict K(test, train)).
        precision: None, "float32" or "float64", as in libsdca.fit.
        n_jobs: Threads of the evaluations: None for the default (SDCA_NUM_THREADS, else
            the available CPUs), a positive count, or -1 for all CPUs (-2 all but one, ...).
        random_state: The order of the examples in the epochs (an int, a RandomState or
            None, as in scikit-learn).
        warm_start: Start a refit from the previous solution, its duals scaled by C /
            previous C (the feasible sets scale with C); the data must be the same (only
            the shapes, classes and loss are checked).

    Attributes:
        classes_: The class labels (n_classes,).
        coef_: Weights (n_classes, n_features), or (1, n_features) for two classes (the
            weights of the margin of decision_function); absent for precomputed kernels.
        dual_coef_: Dual variables (n_classes, n_samples).
        n_features_in_: Number of features (training samples for kernels).
        n_iter_: Epochs run.
        status_: Solver status; only "solved" guarantees the duality gap.
        history_: The evaluation history (libsdca.HISTORY keys).
        model_: The libsdca.Model of the last fit.
    """

    def __init__(
        self,
        loss: str = "cross_entropy",
        *,
        C: float = 1.0,
        k: int | None = None,
        variant: str | None = None,
        gamma: float | None = None,
        tol: float = 1e-3,
        max_iter: int = 1000,
        eval_epoch: int = 10,
        shrinking: str = "auto",
        kernel: str = "linear",
        precision: str | None = None,
        n_jobs: int | None = None,
        random_state: Any = None,
        warm_start: bool = False,
    ) -> None:
        self.loss = loss
        self.C = C
        self.k = k
        self.variant = variant
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.eval_epoch = eval_epoch
        self.shrinking = shrinking
        self.kernel = kernel
        self.precision = precision
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.warm_start = warm_start

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.kernel == "precomputed"
        return tags

    def _loss(self) -> Loss:
        """The loss object of the parameters loss, k, variant and gamma."""
        loss = _f.check_choice(self.loss, "loss", _SINGLE_LABEL)
        names = ("k", "variant", "gamma")
        params = {name: getattr(self, name) for name in names if getattr(self, name) is not None}
        return LOSSES[loss](**params)

    def _threads(self) -> int | None:
        """num_threads of n_jobs, as in joblib: -1 all CPUs, -2 all but one, ..."""
        jobs = self.n_jobs
        if jobs is None:
            return None
        jobs = _f.check_integer(jobs, "n_jobs", -(2**63))
        if jobs == 0:
            raise ValueError("n_jobs must not be 0")
        if jobs > 0:
            return int(jobs)
        affinity = getattr(os, "sched_getaffinity", None)
        cpus = len(affinity(0)) if affinity else os.cpu_count() or 1
        return max(1, cpus + 1 + int(jobs))

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> SDCAClassifier:
        """Train on X (n_samples, n_features), or a Gram matrix, and labels y."""
        loss = self._loss()
        if self.kernel not in ("linear", "precomputed"):
            raise ValueError("kernel must be 'linear' or 'precomputed'")
        features = getattr(self, "n_features_in_", None)
        # validate_data updates feature metadata before the solver can reject a fit.
        validated = copy(self)
        x, labels = validate_data(validated, X, y, dtype=[np.float64, np.float32])
        check_classification_targets(labels)
        classes, encoded = np.unique(labels, return_inverse=True)
        label_digest = (encoded.dtype.str, encoded.shape, hashlib.sha256(encoded).digest())
        if len(classes) < 2:
            raise ValueError(f"Training needs at least two classes; got {len(classes)} class")
        precomputed = self.kernel == "precomputed"
        if precomputed and x.shape[0] != x.shape[1]:
            raise ValueError("A precomputed training kernel must be square")
        initial = None
        previous: Model | None = getattr(self, "model_", None)
        known = np.asarray(getattr(self, "classes_", ()))
        if (
            self.warm_start
            and previous is not None
            and np.array_equal(known, classes)
            and getattr(self, "_fit_label_digest", None) == label_digest
            and previous.loss == loss
            and previous.kernel == precomputed
            and features == x.shape[1]
            and previous.dual.shape == (len(x), len(classes))
        ):
            # The feasible set of every objective is C times a fixed set.
            initial = previous.dual * (float(self.C) / previous.C)
        seed = int(check_random_state(self.random_state).randint(np.iinfo(np.int32).max))
        model = fit(
            Kernel(x) if precomputed else x,
            encoded,
            loss,
            C=self.C,
            tol=self.tol,
            max_epoch=self.max_iter,
            eval_epoch=self.eval_epoch,
            shrinking=self.shrinking,
            seed=seed,
            num_classes=len(classes),
            initial_dual=initial,
            precision=self.precision,
            num_threads=self._threads(),
        )
        self.n_features_in_ = validated.n_features_in_
        if hasattr(validated, "feature_names_in_"):
            self.feature_names_in_ = validated.feature_names_in_
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self._fit_label_digest = label_digest
        self.model_ = model
        self.classes_ = classes
        if model.weights is None:
            if hasattr(self, "coef_"):
                del self.coef_
        else:
            weights = model.weights
            self.coef_ = weights[1:] - weights[:1] if len(classes) == 2 else weights
        self.dual_coef_ = model.dual.T
        self.n_iter_ = model.epochs
        self.status_ = model.status
        self.history_ = model.history
        if model.status != "solved":
            warnings.warn(
                f"Training stopped with status {model.status!r} before reaching tol; "
                "increase max_iter or tol",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    def _scores(self, X: npt.ArrayLike) -> FloatArray:
        check_is_fitted(self)
        x = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        return self.model_.decision_function(x)

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
        return self.loss == "cross_entropy"

    @available_if(_has_proba)
    def predict_log_proba(self, X: npt.ArrayLike) -> FloatArray:
        """Log-probabilities: the log-softmax of the scores (loss="cross_entropy")."""
        return _log_softmax(self._scores(X).astype(np.float64))

    @available_if(_has_proba)
    def predict_proba(self, X: npt.ArrayLike) -> FloatArray:
        """Class probabilities: the softmax of the scores (loss="cross_entropy")."""
        return np.exp(self.predict_log_proba(X))
