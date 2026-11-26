"""The composed losses: one implementation of every loss over an array namespace xp
(xp_torch for PyTorch, xp_jax for JAX), used by backend="composed" and as the test reference.

loss(name, scores, target, **options) is the frontends' entry point. Behind it, each loss is
a function (xp, logits, target, **params) -> per-row losses (rows,) of (rows, classes)
logits, differentiable in the logits by the framework: classes last, 16-bit logits computed
in float32 (pass float64 logits for float64 work). Masked classes (-inf) follow
docs/API.md in the same batched pass; invalid rows are NaN, with NaN gradients.

Cross-entropy (labels and probabilities), multilabel cross-entropy, distillation and the
nonsmooth hinges differentiate at any order and in forward mode. The smoothed hinges, top-k
cross-entropy with k > 1 and the proximal losses attach the gradient of their solution
(xp.attach): exact first derivatives in reverse and forward mode; differentiating them again
raises RuntimeError("... first derivatives only ...").
"""

from ._api import NAMES, loss, namespace
from ._losses import (
    cross_entropy,
    distillation,
    label_weights,
    multiclass_hinge,
    multilabel_cross_entropy,
    multilabel_hinge,
    proximal_cross_entropy,
    proximal_distillation,
    topk_cross_entropy,
    topk_hinge,
)
from ._xp import Namespace

__all__ = [
    "NAMES",
    "Namespace",
    "loss",
    "namespace",
    "cross_entropy",
    "distillation",
    "label_weights",
    "multiclass_hinge",
    "multilabel_cross_entropy",
    "multilabel_hinge",
    "proximal_cross_entropy",
    "proximal_distillation",
    "topk_cross_entropy",
    "topk_hinge",
]
