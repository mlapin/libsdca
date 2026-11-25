"""Accurate and fast classification losses for SDCA and neural networks.

NumPy (this module):
    fit, loss_and_grad, coordinate_update and Model; the losses CrossEntropy,
    TopKCrossEntropy, MulticlassHinge, TopKHinge, MultilabelHinge, MultilabelCrossEntropy,
    ProximalCrossEntropy, Distillation and ProximalDistillation;
    the target and input markers Probabilities and Kernel; InvalidRowsWarning, which
    loss_and_grad issues once per call with NaN rows (check_rows=True raises instead).
scikit-learn (import libsdca.sklearn; needs scikit-learn):
    SDCAClassifier.
PyTorch (import libsdca.torch; needs torch) and JAX (import libsdca.jax; needs jax):
    the losses as framework modules and functions; libsdca.optax has optax's names and
    signatures for the cross-entropies and the multiclass hinge (needs jax, not optax).
    `import libsdca` imports neither.
Names: LOSSES (shared name -> loss class), OBJECTIVES (the names fit trains), their
    subsets TOPK, HINGES and MULTILABEL, PRECISIONS, HISTORY (Model.history keys) and
    STATUS (Model.status values).

Arrays are C-contiguous: X (n, d), the weights W (m, d) with one row per class, and the
duals A (n, m) for m classes. Objective = 0.5 * ||W - center||^2 + C * sum_i loss(W @ X[i],
y[i]); there is no implicit intercept (append a constant feature for a regularized
intercept).
"""

from __future__ import annotations

from ._native import (
    HINGES,
    HISTORY,
    LOSSES,
    MULTILABEL,
    OBJECTIVES,
    PRECISIONS,
    STATUS,
    TOPK,
    CrossEntropy,
    Distillation,
    FloatArray,
    IndexArray,
    InvalidRowsWarning,
    Kernel,
    Loss,
    Model,
    MulticlassHinge,
    MultilabelCrossEntropy,
    MultilabelHinge,
    Probabilities,
    ProximalCrossEntropy,
    ProximalDistillation,
    Targets,
    TopKCrossEntropy,
    TopKHinge,
    coordinate_update,
    fit,
    loss_and_grad,
)

__version__ = "0.4.0"

__all__ = [
    "HINGES",
    "HISTORY",
    "LOSSES",
    "MULTILABEL",
    "OBJECTIVES",
    "PRECISIONS",
    "STATUS",
    "TOPK",
    "CrossEntropy",
    "Distillation",
    "FloatArray",
    "IndexArray",
    "InvalidRowsWarning",
    "Kernel",
    "Loss",
    "Model",
    "MulticlassHinge",
    "MultilabelCrossEntropy",
    "MultilabelHinge",
    "Probabilities",
    "ProximalCrossEntropy",
    "ProximalDistillation",
    "Targets",
    "TopKCrossEntropy",
    "TopKHinge",
    "coordinate_update",
    "fit",
    "loss_and_grad",
]
