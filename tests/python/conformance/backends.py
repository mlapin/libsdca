"""One call of a loss on a backend: the values and the logits' gradient as float64 arrays.

A Backend is a framework (numpy: libsdca.loss_and_grad; torch: libsdca.torch; jax:
libsdca.jax), a device and the backend= option of the call (None for NumPy, which has only
the kernels): "kernel" (libsdca's kernels) or "composed" (the composed tensor operations).
The frameworks are imported on first use. JAX losses run eagerly. Reductions take an upstream
derivative of one (all ones for "none").
"""

from __future__ import annotations

import functools
import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

from conformance.inputs import Case
from conformance.references import DISTILLATION, Loss

MASKS = ("multilabel_cross_entropy", "multilabel_hinge")


@dataclass(frozen=True)
class Backend:
    name: str
    framework: str  # numpy, torch or jax
    device: str  # cpu, cuda (torch) or gpu (jax)
    option: str | None = None

    @property
    def native(self) -> bool:
        """Whether libsdca's kernels compute the loss."""
        return self.option != "composed"

    @property
    def cuda(self) -> bool:
        return self.device in ("cuda", "gpu")

    def runs(self, loss: Loss) -> bool:
        """linear_cross_entropy is PyTorch's alone."""
        return loss.name != "linear_cross_entropy" or self.framework == "torch"


def arrays(case: Case) -> dict[str, np.ndarray]:
    """Logits, teacher logits, positive mask and labels (the first positive) of a case."""

    def repeat(field: str) -> np.ndarray:
        return np.array(
            [[getattr(b, field) for b in row for _ in range(b.count)] for row in case.rows]
        )

    mask = repeat("positive").astype(bool)
    return dict(
        logits=repeat("value").astype(np.float64),
        teacher=repeat("teacher").astype(np.float64),
        mask=mask,
        labels=mask.argmax(-1).astype(np.int64),
    )


@functools.cache
def unavailable(backend: Backend) -> str | None:
    """Why a backend cannot run here, or None."""
    if backend.framework != "numpy" and importlib.util.find_spec(backend.framework) is None:
        return f"{backend.framework} is not installed"
    if backend.framework == "torch" and backend.cuda:
        import torch

        if not torch.cuda.is_available():
            return "CUDA is unavailable"
    if backend.framework == "jax" and backend.cuda:
        import jax

        if not any(d.platform == "gpu" for d in jax.devices()):
            return "no JAX GPU"
    return None


def run(
    backend: Backend, loss: Loss, dtype: str, precision: str | None, reduction: str, case: Case
) -> tuple[np.ndarray, np.ndarray, str]:
    """Values (per row, or the reduction), the logits' gradient and the route taken."""
    data = arrays(case)
    if backend.framework == "numpy":
        return _numpy(loss, dtype, precision, data)
    if backend.framework == "torch":
        return _torch(backend, loss, dtype, precision, reduction, data)
    return _jax(backend, loss, dtype, precision, reduction, data)


def _numpy(
    loss: Loss, dtype: str, precision: str | None, data: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, str]:
    import libsdca
    import named_losses

    z = data["logits"].astype(dtype)
    if loss.name in DISTILLATION:
        target = data["teacher"].astype(dtype)
        params: dict[str, Any] = dict(temperature=loss.temperature)
        if loss.name == "proximal_distillation":
            params["alpha"] = loss.alpha
    else:
        target = data["mask"] if loss.name in MASKS else data["labels"]
        params = dict(k=loss.k, gamma=loss.gamma, alpha=loss.alpha)
    instance = named_losses.numpy_loss(loss.name, **params)
    values, gradient = libsdca.loss_and_grad(z, target, instance, precision=precision)
    return np.asarray(values, np.float64), np.asarray(gradient, np.float64), "kernel"


def _torch(
    backend: Backend,
    loss: Loss,
    dtype: str,
    precision: str | None,
    reduction: str,
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, str]:
    import libsdca.torch as L
    import torch

    device = "cuda" if backend.cuda else "cpu"
    kind = getattr(torch, dtype)

    def tensor(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, device=device).to(kind)

    option = backend.option or "kernel"
    common: dict[str, Any] = dict(reduction=reduction, precision=precision, backend=option)
    if loss.name == "linear_cross_entropy":
        n = data["logits"].shape[0]
        weight = tensor(data["logits"].T.copy()).requires_grad_()
        hidden = torch.eye(n, device=device).to(kind)
        labels = torch.tensor(data["labels"], device=device)
        value = L.linear_cross_entropy(hidden, weight, labels, recompute=loss.recompute, **common)
        leaf = weight
        route = option
    else:
        from named_losses import function, parameters

        leaf = tensor(data["logits"]).requires_grad_()
        if loss.name in DISTILLATION:
            teacher = tensor(data["teacher"])
            options: dict[str, Any] = dict(temperature=loss.temperature, **common)
            if loss.name == "proximal_distillation":
                options["gamma"] = loss.alpha
            target: Any = teacher
            fn = getattr(L, loss.name + "_loss")
        else:
            target = torch.tensor(
                data["mask"] if loss.name in MASKS else data["labels"], device=device
            )
            options = dict(**parameters(loss.name, loss.k, loss.gamma, loss.alpha), **common)
            fn = function(L, loss.name)
        route = L.explain_backend(fn, leaf, target, **options)[0]
        value = fn(leaf, target, **options)
    value.backward(torch.ones_like(value))
    assert leaf.grad is not None
    gradient = leaf.grad.T if loss.name == "linear_cross_entropy" else leaf.grad
    return (
        value.detach().double().cpu().numpy(),
        gradient.double().cpu().numpy(),
        route,
    )


def _jax(
    backend: Backend,
    loss: Loss,
    dtype: str,
    precision: str | None,
    reduction: str,
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, str]:
    import jax
    import jax.numpy as jnp
    import named_losses
    from libsdca import jax as J

    device = next(d for d in jax.devices("gpu" if backend.cuda else "cpu"))
    kind = jnp.dtype(dtype)

    def put(x: np.ndarray, as_dtype: Any = None) -> jax.Array:
        return jax.device_put(np.asarray(x, dtype=as_dtype or kind), device)

    option = backend.option or "kernel"
    common: dict[str, Any] = dict(reduction=reduction, precision=precision, backend=option)
    with jax.default_device(device):
        if loss.name in DISTILLATION:
            teacher = put(data["teacher"])
            options: dict[str, Any] = dict(temperature=loss.temperature, **common)
            if loss.name == "proximal_distillation":
                options["gamma"] = loss.alpha
            fn = getattr(J, loss.name + "_loss")

            def f(z: jax.Array) -> jax.Array:
                return fn(z, teacher, **options)

        else:
            target = (
                put(data["mask"], bool) if loss.name in MASKS else put(data["labels"], np.int32)
            )

            def f(z: jax.Array) -> jax.Array:
                return named_losses.loss(
                    J,
                    loss.name,
                    z,
                    target,
                    k=loss.k,
                    gamma=loss.gamma,
                    alpha=loss.alpha,
                    **common,
                )

        value, pullback = jax.vjp(f, put(data["logits"]))
        (gradient,) = pullback(jnp.ones_like(value))
    return np.asarray(value, np.float64), np.asarray(gradient, np.float64), option
