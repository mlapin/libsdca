"""Accurate and fast classification losses for SDCA and neural networks.

NumPy (this module):
    fit, loss_and_grad, coordinate_update and Model; the losses CrossEntropy,
    TopKCrossEntropy, MulticlassHinge, TopKHingeAlpha, TopKHingeBeta, MultilabelHinge,
    MultilabelCrossEntropy, ProximalCrossEntropy, Distillation and ProximalDistillation;
    the target and input markers Probabilities and Kernel.
scikit-learn (import libsdca.sklearn; needs scikit-learn):
    SDCAClassifier.
PyTorch (import libsdca.torch; needs torch) and JAX (import libsdca.jax; needs jax):
    the losses as framework modules and functions. `import libsdca` imports neither.
Names: LOSSES (shared name -> loss class), OBJECTIVES (the names fit trains), their
    subsets TOPK, HINGES and MULTILABEL, PRECISIONS, HISTORY (Model.history keys) and
    STATUS (Model.status values).

Arrays use conventional Python layouts: X (n, d), W (d, m), A (n, m) for m classes.
Objective = 0.5 * ||W - center||^2 + C * sum_i loss(X[i] @ W, y[i]); there is no implicit
intercept (append a constant feature for a regularized intercept).
"""

from __future__ import annotations

from ._core import HINGES, HISTORY, MULTILABEL, OBJECTIVES, PRECISIONS, STATUS, TOPK
from ._numpy import (
    LOSSES,
    CrossEntropy,
    Distillation,
    FloatArray,
    IndexArray,
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
    TopKHingeAlpha,
    TopKHingeBeta,
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
    "TopKHingeAlpha",
    "TopKHingeBeta",
    "coordinate_update",
    "fit",
    "loss_and_grad",
]
