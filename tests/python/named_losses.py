"""The PyTorch and JAX losses by their shared names (libsdca.LOSSES), for tests that sweep them.

loss(framework_module, name, logits, target, k=..., gamma=..., alpha=..., **options) calls the
named public function with the parameters that loss takes: k for the top-k losses, gamma for
the hinges, alpha for proximal_cross_entropy; the others are ignored, so a sweep can pass one
set. Options (reduction, dim/axis, backend, precision, weight, ...) pass through.
"""

from __future__ import annotations

from typing import Any

TOPK = ("topk_cross_entropy", "topk_hinge_alpha", "topk_hinge_beta")
HINGES = ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge")
MULTILABEL = ("multilabel_cross_entropy", "multilabel_hinge")
SINGLE_LABEL = (
    "cross_entropy",
    "topk_cross_entropy",
    "multiclass_hinge",
    "topk_hinge_alpha",
    "topk_hinge_beta",
    "proximal_cross_entropy",
)


def parameters(name: str, k: int = 1, gamma: float = 0.0, alpha: float = 1.0) -> dict[str, Any]:
    """The keyword arguments of the named loss's function for these parameters."""
    out: dict[str, Any] = {}
    if name in TOPK:
        out["k"] = k
    if name in HINGES:
        out["gamma"] = gamma
    if name.startswith("topk_hinge_"):
        out["variant"] = name.removeprefix("topk_hinge_")
    if name == "proximal_cross_entropy":
        out["alpha"] = alpha
    return out


def function(module: Any, name: str) -> Any:
    """The public function of a loss name in libsdca.torch or libsdca.jax."""
    if name.startswith("topk_hinge_"):
        return module.topk_hinge_loss
    if name in ("multiclass_hinge", "multilabel_hinge"):
        return getattr(module, name + "_loss")
    return getattr(module, name)


def loss(
    module: Any,
    name: str,
    logits: Any,
    target: Any,
    *,
    k: int = 1,
    gamma: float = 0.0,
    alpha: float = 1.0,
    **options: Any,
) -> Any:
    """The named loss of libsdca.torch or libsdca.jax. torch's cross_entropy takes the class
    dimension 1; a dim option other than 1 (or -1 of 2-D logits) moves it there."""
    fn = function(module, name)
    if name == "cross_entropy" and module.__name__ == "libsdca.torch":
        dim = options.pop("dim", -1)
        if logits.ndim > 1 and dim % logits.ndim != 1:
            logits = logits.movedim(dim, 1)
    return fn(logits, target, **parameters(name, k, gamma, alpha), **options)


def torch_call(
    logits: Any,
    target: Any,
    name: str = "cross_entropy",
    *,
    k: int = 1,
    gamma: float = 0.0,
    alpha: float = 1.0,
    **options: Any,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """(function, logits, target, keyword arguments) of a libsdca.torch loss by its shared
    name, for sweeps over LOSSES with one target per example: "cross_entropy" of integer
    labels is the shared integer-entropy path (topk_cross_entropy at k = 1, with dim), of a
    boolean mask multilabel_cross_entropy; a multilabel loss of integer labels takes their
    one-hot mask, and another single-label loss of a (one-hot) mask its first positive."""
    import libsdca.torch as L
    import torch

    dim = options.get("dim", -1)
    if not isinstance(target, torch.Tensor):  # passed through for the losses' own errors
        return function(L, name), logits, target, {**parameters(name, k, gamma, alpha), **options}
    if name == "cross_entropy":
        if target.dtype == torch.bool:
            return L.multilabel_cross_entropy, logits, target, options
        return L.topk_cross_entropy, logits, target, {"k": 1, **options}
    if name in MULTILABEL and target.dtype != torch.bool:
        classes = logits.shape[dim]
        mask = torch.nn.functional.one_hot(target.long().clamp(0, classes - 1), classes).bool()
        target = mask.movedim(-1, dim % logits.ndim)
    elif name not in MULTILABEL and target.dtype == torch.bool:
        rows = target.movedim(dim, -1)
        # Only a one-hot mask has a class label: a test of other masks needs another loss.
        assert torch.compiler.is_compiling() or bool((rows.sum(-1) == 1).all()), name
        target = rows.to(torch.uint8).argmax(-1)
    return function(L, name), logits, target, {**parameters(name, k, gamma, alpha), **options}


def torch_loss(logits: Any, target: Any, name: str = "cross_entropy", **options: Any) -> Any:
    """The libsdca.torch loss of a shared name (torch_call)."""
    fn, logits, target, kwargs = torch_call(logits, target, name, **options)
    return fn(logits, target, **kwargs)


def torch_explain(logits: Any, target: Any, name: str = "cross_entropy", **options: Any) -> Any:
    """libsdca.torch.explain_backend of torch_loss's call."""
    import libsdca.torch as L

    fn, logits, target, kwargs = torch_call(logits, target, name, **options)
    return L.explain_backend(fn, logits, target, **kwargs)


MODULES = {
    "cross_entropy": "TopKCrossEntropyLoss",
    "topk_cross_entropy": "TopKCrossEntropyLoss",
    "multiclass_hinge": "MulticlassHingeLoss",
    "topk_hinge_alpha": "TopKHingeLoss",
    "topk_hinge_beta": "TopKHingeLoss",
    "multilabel_cross_entropy": "MultilabelCrossEntropyLoss",
    "multilabel_hinge": "MultilabelHingeLoss",
    "proximal_cross_entropy": "ProximalCrossEntropyLoss",
}


def torch_module(
    name: str = "cross_entropy",
    *,
    k: int = 1,
    gamma: float = 0.0,
    alpha: float = 1.0,
    **options: Any,
) -> Any:
    """The libsdca.torch loss module of a shared name ("cross_entropy": of integer labels,
    TopKCrossEntropyLoss at k = 1, as torch_call)."""
    import libsdca.torch as L

    parameters_ = parameters(name, k, gamma, alpha)
    if name == "cross_entropy":
        parameters_ = {"k": 1}
    return getattr(L, MODULES[name])(**parameters_, **options)


def proximal_mask_loss(
    logits: Any,
    mask: Any,
    *,
    alpha: float = 1.0,
    backend: str = "auto",
    reduction: str = "mean",
    dim: int = -1,
    precision: str | None = None,
) -> Any:
    """proximal_cross_entropy of uniform target probabilities on a positive mask: the kernels,
    tensor operations and native library compute it, the public PyTorch API (class labels)
    does not take it; the internal loss of the same backends."""
    import libsdca.torch as L

    objective = "proximal_cross_entropy"
    return L._resolved_loss(
        logits, mask, objective, backend, 1, 1.0, alpha, reduction, dim, precision, 1
    )


def route(explained: tuple[str, str]) -> str:
    """The route of an explain_backend result: "cpu-kernel" (the CPU extension),
    "cuda-launcher" (fused kernels launched from C++), "triton" (launched by Triton or a
    Triton kernel), else its backend ("native" or "torch")."""
    backend, reason = explained
    if backend != "kernel":
        return backend
    if "CPU extension" in reason:
        return "cpu-kernel"
    if "launched from C++ autograd" in reason:
        return "cuda-launcher"
    return "triton"


def explain_route(logits: Any, target: Any, name: str = "cross_entropy", **options: Any) -> Any:
    """(route, reason) of torch_loss's call: route(torch_explain(...)) and its reason."""
    explained = torch_explain(logits, target, name, **options)
    return route(explained), explained[1]


def jax_loss(logits: Any, target: Any, name: str = "cross_entropy", **options: Any) -> Any:
    """loss(libsdca.jax, name, logits, target, **options): the JAX loss of a shared name."""
    import libsdca.jax

    return loss(libsdca.jax, name, logits, target, **options)
