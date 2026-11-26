"""The PyTorch and JAX losses by their kinds, for tests that sweep them.

The kinds are the shared names (libsdca.LOSSES), but for topk_hinge_alpha and topk_hinge_beta:
topk_hinge of variant "alpha" and "beta". loss(framework_module, name, logits, target, k=...,
gamma=..., alpha=..., **options) calls the named public function with the parameters that loss
takes: k for the top-k losses, gamma for the hinges, and alpha, the proximal step, as the gamma
of proximal_cross_entropy; the others are ignored, so a sweep can pass one set. Options
(reduction, dim/axis, backend, precision, weight, ...) pass through.
"""

from __future__ import annotations

from typing import Any

TOPK = ("topk_cross_entropy", "topk_hinge_alpha", "topk_hinge_beta")
HINGES = ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge")
MULTILABEL = ("multilabel_cross_entropy", "multilabel_hinge")
# The training objectives, in the order of the C ids.
OBJECTIVES = (
    "cross_entropy",
    "topk_cross_entropy",
    "multiclass_hinge",
    "topk_hinge_alpha",
    "topk_hinge_beta",
    "multilabel_hinge",
    "multilabel_cross_entropy",
)
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
        out["gamma"] = alpha
    return out


def options_of(name: str, options: dict[str, Any]) -> dict[str, Any]:
    """The keyword arguments of a kind's PyTorch or JAX function for the parameters of a row
    of the rule table (masked_cases: k, gamma, alpha, temperature)."""
    if name in ("distillation", "proximal_distillation"):
        out = dict(options)
        if "alpha" in out:
            out["gamma"] = out.pop("alpha")
        return out
    k, gamma, alpha = int(options.get("k", 1)), options.get("gamma", 0.0), options.get("alpha", 1.0)
    return parameters(name, k, gamma, alpha)


def numpy_loss(name: str, **options: Any) -> Any:
    """The NumPy loss object (libsdca.LOSSES) of a kind with the options it takes, of the
    names of parameters: k, gamma, alpha (the proximal step, its gamma) and temperature; the
    others are ignored."""
    import dataclasses

    import libsdca

    base = "topk_hinge" if name.startswith("topk_hinge_") else name
    cls = libsdca.LOSSES[base]
    taken = {field.name for field in dataclasses.fields(cls)}  # pyright: ignore[reportArgumentType]
    if "alpha" in options and base in ("proximal_cross_entropy", "proximal_distillation"):
        options = {**options, "gamma": options["alpha"]}
    if name != base:
        options = {**options, "variant": name.removeprefix("topk_hinge_")}
    return cls(**{key: value for key, value in options.items() if key in taken})


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


def route(explained: tuple[str, str]) -> str:
    """The route of an explain_backend result: "cpu-kernel" (the CPU kernels), "triton" (the
    Triton kernels) or "composed" (tensor operations)."""
    backend, reason = explained
    if backend == "composed":
        return "composed"
    if reason.startswith("the CPU kernels"):
        return "cpu-kernel"
    if reason.startswith("the Triton kernels"):
        return "triton"
    raise ValueError(f"unknown route of {explained}")


def explain_route(logits: Any, target: Any, name: str = "cross_entropy", **options: Any) -> Any:
    """(route, reason) of torch_loss's call: route(torch_explain(...)) and its reason."""
    explained = torch_explain(logits, target, name, **options)
    return route(explained), explained[1]


def jax_loss(logits: Any, target: Any, name: str = "cross_entropy", **options: Any) -> Any:
    """loss(libsdca.jax, name, logits, target, **options): the JAX loss of a shared name."""
    import libsdca.jax

    return loss(libsdca.jax, name, logits, target, **options)
