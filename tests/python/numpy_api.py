"""Test helpers over the NumPy API: loss objects by shared name, and the API at one storage
dtype and working precision (for numerical tests parametrized over names and precisions;
the API contracts themselves are tested on libsdca directly)."""

from __future__ import annotations

import dataclasses
from typing import Any

import libsdca
import numpy as np
from libsdca._core import _canonical_precision


def make_loss(
    name: str,
    *,
    k: int = 1,
    gamma: float = 0.0,
    alpha: float = 1.0,
    temperature: float = 1.0,
    projection_precision: str = "float64",
) -> libsdca.Loss:
    """The loss named name with the parameters it takes (the others are dropped)."""
    cls = libsdca.LOSSES[name]
    given = dict(
        k=k,
        gamma=gamma,
        alpha=alpha,
        temperature=temperature,
        projection_precision=projection_precision,
    )
    fields = {field.name for field in dataclasses.fields(cls)}  # pyright: ignore[reportArgumentType]
    return cls(**{key: value for key, value in given.items() if key in fields})


class Solver:
    """The NumPy API with inputs cast to one storage dtype, at one working precision.

    precision: "float32" or "float64" (None: float64, or that of dtype); dtype: the storage
    (None: that of precision). Losses are given by name, with the parameters of make_loss.
    """

    def __init__(self, precision: Any = None, *, dtype: Any = None) -> None:
        work = _canonical_precision(precision)
        if work is None:
            work = "float64" if dtype is None else np.dtype(dtype).name
        self.precision: str = work
        self.dtype: type[np.floating] = (
            np.dtype(dtype).type if dtype is not None else getattr(np, work)
        )

    def _array(self, a: Any) -> np.ndarray:
        if type(a).__module__.startswith("torch"):
            a = a.detach().cpu().numpy()
        return np.asarray(a, dtype=self.dtype)

    def loss_and_grad(
        self, logits: Any, target: Any, loss: str = "cross_entropy", **params: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        threads = params.pop("num_threads", None)
        return libsdca.loss_and_grad(
            self._array(logits),
            target,
            make_loss(loss, **params),
            precision=self.precision,
            num_threads=threads,
        )

    def coordinate_update(
        self,
        logits: Any,
        target: Any,
        objective: str = "cross_entropy",
        *,
        previous: Any = None,
        C: float = 1.0,
        norm2: Any = 1.0,
        num_threads: int | None = None,
        **params: Any,
    ) -> np.ndarray:
        return libsdca.coordinate_update(
            self._array(logits),
            target,
            make_loss(objective, **params),
            previous=None if previous is None else self._array(previous),
            C=C,
            norm2=norm2,
            precision=self.precision,
            num_threads=num_threads,
        )

    def fit(
        self,
        X: Any,
        y: Any,
        objective: str = "cross_entropy",
        *,
        kernel: bool = False,
        k: int = 1,
        gamma: float = 0.0,
        projection_precision: str = "float64",
        **options: Any,
    ) -> libsdca.Model:
        x = self._array(X)
        loss = make_loss(objective, k=k, gamma=gamma, projection_precision=projection_precision)
        for key in ("center", "initial_dual"):
            if options.get(key) is not None:
                options[key] = self._array(options[key])
        return libsdca.fit(
            libsdca.Kernel(x) if kernel else x, y, loss, precision=self.precision, **options
        )


def loss_and_grad(
    logits: Any,
    target: Any,
    loss: str = "cross_entropy",
    *,
    precision: Any = None,
    num_threads: int | None = None,
    **params: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """libsdca.loss_and_grad of the loss named loss (with the parameters it takes)."""
    return libsdca.loss_and_grad(
        logits, target, make_loss(loss, **params), precision=precision, num_threads=num_threads
    )


def coordinate_update(
    logits: Any, target: Any, loss: str = "cross_entropy", **options: Any
) -> np.ndarray:
    """libsdca.coordinate_update of the loss named loss (with the parameters it takes)."""
    params = {key: options.pop(key) for key in PARAMETERS if key in options}
    return libsdca.coordinate_update(logits, target, make_loss(loss, **params), **options)


def fit(X: Any, y: Any, loss: str = "cross_entropy", *, kernel: bool = False, **options: Any):
    """libsdca.fit of the loss named loss (with the parameters it takes); kernel marks a
    Gram matrix."""
    params = {key: options.pop(key) for key in PARAMETERS if key in options}
    data = libsdca.Kernel(X) if kernel else X
    return libsdca.fit(data, y, make_loss(loss, **params), **options)


PARAMETERS = ("k", "gamma", "alpha", "temperature", "projection_precision")


def distillation_loss_and_grad(
    student: Any,
    teacher: Any,
    *,
    temperature: float = 1.0,
    alpha: float = 0.0,
    num_threads: int | None = None,
    precision: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """libsdca.loss_and_grad of Distillation, or of ProximalDistillation for alpha > 0."""
    loss: libsdca.Loss = (
        libsdca.ProximalDistillation(temperature, alpha)
        if alpha > 0
        else libsdca.Distillation(temperature)
    )
    return libsdca.loss_and_grad(
        student, teacher, loss, precision=precision, num_threads=num_threads
    )


def proximal_distillation_loss_and_grad(
    student: Any, teacher: Any, *, temperature: float = 1.0, alpha: float = 1.0, **options: Any
) -> tuple[np.ndarray, np.ndarray]:
    """libsdca.loss_and_grad of ProximalDistillation."""
    return libsdca.loss_and_grad(
        student, teacher, libsdca.ProximalDistillation(temperature, alpha), **options
    )


def mask_loss_and_grad(
    logits: Any,
    target: Any,
    loss: str = "cross_entropy",
    *,
    k: int = 1,
    gamma: float = 0.0,
    alpha: float = 1.0,
    precision: Any = None,
    num_threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The internal rows of the tensor layers' host paths (_core._host_loss_and_grad) of
    integer labels or a label mask: the references of the PyTorch and JAX tests, where
    cross_entropy of a mask has the uniform target on its labels."""
    from libsdca._core import _host_loss_and_grad

    z = np.asarray(logits)
    labels = np.asarray(target)
    mask = labels if labels.dtype == np.bool_ else np.eye(z.shape[-1], dtype=bool)[labels]
    return _host_loss_and_grad(
        z,
        mask,
        loss,
        k=k,
        gamma=gamma,
        alpha=alpha,
        precision=precision,
        num_threads=num_threads,
    )
