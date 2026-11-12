"""Accurate and fast classification losses for SDCA and neural networks.

NumPy:
    fit, loss_and_grad, coordinate_update, Model; Solver for another
    precision or library.
scikit-learn (imported on first use; needs scikit-learn):
    SDCAClassifier (libsdca.sklearn).
PyTorch (imported on first use; needs torch):
    CrossEntropyLoss, TopKCrossEntropyLoss, MulticlassHingeLoss, TopKHingeLoss
    (TopKHingeAlphaLoss, TopKHingeBetaLoss), MultilabelCrossEntropyLoss,
    MultilabelHingeLoss, ProximalCrossEntropyLoss, DistillationLoss,
    ProximalDistillationLoss, SDCALoss, and their functions (cross_entropy, ...).
JAX (import libsdca.jax; needs jax):
    sdca_loss, distillation_loss and the loss functions above.
Names: LOSSES, OBJECTIVES (the trainable LOSSES), their subsets TOPK, HINGES
    and MULTILABEL, PRECISIONS, HISTORY (Model.history fields) and STATUS
    (Model.status values); array types FloatArray, IndexArray and Targets.

Arrays use conventional Python layouts: X (n,d), W (d,m), A (n,m) for m classes.
Objective = 0.5*||W||^2 + C*sum_i loss(X[i]@W, y[i]).
No implicit intercept; append a constant feature for a regularized intercept.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ._core import (
    HINGES,
    HISTORY,
    LOSSES,
    MULTILABEL,
    OBJECTIVES,
    PRECISIONS,
    STATUS,
    TOPK,
    FloatArray,
    IndexArray,
    Model,
    Solver,
    Targets,
    coordinate_update,
    distillation_loss_and_grad,
    fit,
    loss_and_grad,
    proximal_distillation_loss_and_grad,
)

if TYPE_CHECKING:
    from .sklearn import SDCAClassifier as SDCAClassifier
    from .torch import CrossEntropyLoss as CrossEntropyLoss
    from .torch import DistillationLoss as DistillationLoss
    from .torch import LinearCrossEntropyLoss as LinearCrossEntropyLoss
    from .torch import MulticlassHingeLoss as MulticlassHingeLoss
    from .torch import MultilabelCrossEntropyLoss as MultilabelCrossEntropyLoss
    from .torch import MultilabelHingeLoss as MultilabelHingeLoss
    from .torch import ProximalCrossEntropyLoss as ProximalCrossEntropyLoss
    from .torch import ProximalDistillationLoss as ProximalDistillationLoss
    from .torch import SDCALoss as SDCALoss
    from .torch import TopKCrossEntropyLoss as TopKCrossEntropyLoss
    from .torch import TopKHingeAlphaLoss as TopKHingeAlphaLoss
    from .torch import TopKHingeBetaLoss as TopKHingeBetaLoss
    from .torch import TopKHingeLoss as TopKHingeLoss
    from .torch import cross_entropy as cross_entropy
    from .torch import distillation_loss as distillation_loss
    from .torch import linear_cross_entropy as linear_cross_entropy
    from .torch import multiclass_hinge_loss as multiclass_hinge_loss
    from .torch import multilabel_cross_entropy as multilabel_cross_entropy
    from .torch import multilabel_hinge_loss as multilabel_hinge_loss
    from .torch import proximal_cross_entropy as proximal_cross_entropy
    from .torch import proximal_distillation_loss as proximal_distillation_loss
    from .torch import sdca_loss as sdca_loss
    from .torch import topk_cross_entropy as topk_cross_entropy
    from .torch import topk_hinge_alpha_loss as topk_hinge_alpha_loss
    from .torch import topk_hinge_beta_loss as topk_hinge_beta_loss
    from .torch import topk_hinge_loss as topk_hinge_loss

__version__ = "0.3.1"
# Names of the optional submodules, imported on first use.
_LAZY = {
    "SDCAClassifier": ".sklearn",
    "CrossEntropyLoss": ".torch",
    "DistillationLoss": ".torch",
    "LinearCrossEntropyLoss": ".torch",
    "MulticlassHingeLoss": ".torch",
    "MultilabelCrossEntropyLoss": ".torch",
    "MultilabelHingeLoss": ".torch",
    "ProximalCrossEntropyLoss": ".torch",
    "ProximalDistillationLoss": ".torch",
    "SDCALoss": ".torch",
    "TopKCrossEntropyLoss": ".torch",
    "TopKHingeAlphaLoss": ".torch",
    "TopKHingeBetaLoss": ".torch",
    "TopKHingeLoss": ".torch",
    "cross_entropy": ".torch",
    "distillation_loss": ".torch",
    "linear_cross_entropy": ".torch",
    "multiclass_hinge_loss": ".torch",
    "multilabel_cross_entropy": ".torch",
    "multilabel_hinge_loss": ".torch",
    "proximal_cross_entropy": ".torch",
    "proximal_distillation_loss": ".torch",
    "sdca_loss": ".torch",
    "topk_cross_entropy": ".torch",
    "topk_hinge_alpha_loss": ".torch",
    "topk_hinge_beta_loss": ".torch",
    "topk_hinge_loss": ".torch",
}

# The lazy names stay out of __all__: a star import must not need torch.
__all__ = [
    "HINGES",
    "HISTORY",
    "LOSSES",
    "MULTILABEL",
    "OBJECTIVES",
    "PRECISIONS",
    "STATUS",
    "TOPK",
    "FloatArray",
    "IndexArray",
    "Model",
    "Solver",
    "Targets",
    "coordinate_update",
    "distillation_loss_and_grad",
    "fit",
    "loss_and_grad",
    "proximal_distillation_loss_and_grad",
]


def __getattr__(name: str) -> Any:
    # The scikit-learn and PyTorch layers import their framework only when used.
    if name in _LAZY:
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
