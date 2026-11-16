"""One call of a loss on a backend: the values and the logits' gradient as float64 arrays.

A Backend is a framework (numpy: libsdca.loss_and_grad; torch: libsdca.torch; jax:
libsdca.jax), a device, the backend= option of the call (None: the function has none) and
the extension switches it sets (SDCA_CPU_EXTENSION, SDCA_CUDA_LAUNCHER, SDCA_JAX_FFI,
SDCA_JAX_CUDA, read at each call). The frameworks are imported on first use. JAX losses run
eagerly. Reductions take an upstream derivative of one (all ones for "none").
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from unittest import mock

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
    env: tuple[tuple[str, str], ...] = ()
    dtypes: tuple[str, ...] = ()  # the framework's storage dtypes if empty

    @property
    def native(self) -> bool:
        return self.framework == "numpy" or self.option == "native"

    @property
    def cuda(self) -> bool:
        return self.device in ("cuda", "gpu")

    def runs(self, loss: Loss) -> bool:
        """Distillation has no backend option: its path is the auto (or jax) backend's."""
        if loss.name in DISTILLATION:
            return self.option in (None, "auto", "jax")
        if loss.name == "linear_cross_entropy":
            return self.framework == "torch" and not (self.cuda and self.option == "native")
        return True


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
    env = dict(backend.env)
    if backend.framework == "torch":
        import torch

        if backend.cuda and not torch.cuda.is_available():
            return "CUDA is unavailable"
        try:
            if env.get("SDCA_CPU_EXTENSION") == "on":
                from libsdca import _cpu_entropy

                _cpu_entropy.operators()
            if env.get("SDCA_CUDA_LAUNCHER") == "on":
                from libsdca import _cuda_launch

                with switches(backend):
                    _cuda_launch.status()
        except RuntimeError as error:
            return f"{backend.name} does not build: {error}"
    if backend.framework == "jax":
        import jax

        if backend.cuda and not any(d.platform == "gpu" for d in jax.devices()):
            return "no JAX GPU"
        if backend.name == "jax-cpu-auto" and jax.default_backend() != "cpu":
            return "traced auto calls take the default backend's route, not the CPU's"
        if env.get("SDCA_JAX_FFI") == "on":
            from libsdca import _jax_ffi

            if _jax_ffi.missing_tools() is not None:
                return f"no JAX FFI build: {_jax_ffi.missing_tools()}"
    return None


@contextlib.contextmanager
def switches(backend: Backend) -> Iterator[None]:
    """The backend's environment; the CUDA launcher caches its status, cleared around it."""
    with mock.patch.dict(os.environ, dict(backend.env)):
        if "SDCA_CUDA_LAUNCHER" not in dict(backend.env):
            yield
            return
        from libsdca import _cuda_launch

        _cuda_launch.status.cache_clear()
        try:
            yield
        finally:
            _cuda_launch.status.cache_clear()


def run(
    backend: Backend, loss: Loss, dtype: str, precision: str | None, reduction: str, case: Case
) -> tuple[np.ndarray, np.ndarray, str]:
    """Values (per row, or the reduction), the logits' gradient and the route taken."""
    data = arrays(case)
    with switches(backend):
        if backend.framework == "numpy":
            return _numpy(loss, dtype, precision, data)
        if backend.framework == "torch":
            return _torch(backend, loss, dtype, precision, reduction, data)
        return _jax(backend, loss, dtype, precision, reduction, data)


def concrete_value(backend: Backend, loss: Loss, case: Case) -> np.ndarray:
    """A JAX loss of concrete (untraced) float32 arrays: the call that validates its inputs."""
    import jax
    from libsdca import jax as J

    data = arrays(case)
    device = next(d for d in jax.devices("gpu" if backend.cuda else "cpu"))
    z = jax.device_put(data["logits"].astype(np.float32), device)
    with switches(backend):
        if loss.name in DISTILLATION:
            teacher = jax.device_put(data["teacher"].astype(np.float32), device)
            if loss.name == "distillation":
                return np.asarray(J.distillation_loss(z, teacher, loss.temperature))
            return np.asarray(
                J.proximal_distillation_loss(z, teacher, loss.temperature, loss.alpha)
            )
        target = jax.device_put(data["mask"] if loss.name in MASKS else data["labels"], device)
        value = J.sdca_loss(
            z,
            target,
            loss.name,
            backend=backend.option or "auto",
            k=loss.k,
            gamma=loss.gamma,
            alpha=loss.alpha,
        )
        return np.asarray(value)


def _numpy(
    loss: Loss, dtype: str, precision: str | None, data: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, str]:
    import libsdca

    z = data["logits"].astype(dtype)
    if loss.name in DISTILLATION:
        alpha = loss.alpha if loss.name == "proximal_distillation" else 0.0
        teacher = data["teacher"].astype(dtype)
        values, gradient = libsdca.distillation_loss_and_grad(
            z, teacher, temperature=loss.temperature, alpha=alpha, precision=precision
        )
    else:
        target = data["mask"] if loss.name in MASKS else data["labels"]
        values, gradient = libsdca.loss_and_grad(
            z, target, loss.name, k=loss.k, gamma=loss.gamma, alpha=loss.alpha, precision=precision
        )
    return np.asarray(values, np.float64), np.asarray(gradient, np.float64), "native"


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

    common: dict[str, Any] = dict(reduction=reduction, precision=precision)
    route = backend.option or "auto"
    if loss.name == "linear_cross_entropy":
        n = data["logits"].shape[0]
        weight = tensor(data["logits"].T.copy()).requires_grad_()
        # torch 2.1 cannot construct a bfloat16 identity directly.
        hidden = torch.eye(n, device=device).to(kind)
        labels = torch.tensor(data["labels"], device=device)
        value = L.linear_cross_entropy(
            hidden, weight, labels, backend=route, recompute=loss.recompute, **common
        )
        leaf = weight
    else:
        leaf = tensor(data["logits"]).requires_grad_()
        if loss.name in DISTILLATION:
            teacher = tensor(data["teacher"])
            if loss.name == "distillation":
                value = L.distillation_loss(leaf, teacher, loss.temperature, **common)
            else:
                value = L.proximal_distillation_loss(
                    leaf, teacher, loss.temperature, loss.alpha, **common
                )
        else:
            target = torch.tensor(
                data["mask"] if loss.name in MASKS else data["labels"], device=device
            )
            options: dict[str, Any] = dict(k=loss.k, gamma=loss.gamma, **common)
            route = L.explain_backend(leaf, target, loss.name, backend=route, **options)[0]
            value = L.sdca_loss(
                leaf,
                target,
                loss.name,
                backend=backend.option or "auto",
                alpha=loss.alpha,
                **options,
            )
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
    from libsdca import jax as J

    device = next(d for d in jax.devices("gpu" if backend.cuda else "cpu"))
    kind = jnp.dtype(dtype)

    def put(x: np.ndarray, as_dtype: Any = None) -> jax.Array:
        return jax.device_put(np.asarray(x, dtype=as_dtype or kind), device)

    common: dict[str, Any] = dict(reduction=reduction, precision=precision)
    with jax.default_device(device):
        if loss.name in DISTILLATION:
            teacher = put(data["teacher"])
            if loss.name == "distillation":

                def f(z: jax.Array) -> jax.Array:
                    return J.distillation_loss(z, teacher, loss.temperature, **common)
            else:

                def f(z: jax.Array) -> jax.Array:
                    return J.proximal_distillation_loss(
                        z, teacher, loss.temperature, loss.alpha, **common
                    )
        else:
            target = (
                put(data["mask"], bool) if loss.name in MASKS else put(data["labels"], np.int64)
            )

            def f(z: jax.Array) -> jax.Array:
                return J.sdca_loss(
                    z,
                    target,
                    loss.name,
                    backend=backend.option or "auto",
                    k=loss.k,
                    gamma=loss.gamma,
                    alpha=loss.alpha,
                    **common,
                )

        value, pullback = jax.vjp(f, put(data["logits"]))
        (gradient,) = pullback(jnp.ones_like(value))
    route = backend.option or "auto"
    return np.asarray(value, np.float64), np.asarray(gradient, np.float64), route
