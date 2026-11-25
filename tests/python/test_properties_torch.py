"""Property-based tests of libsdca.torch and NumPy's
loss_and_grad: P1 (each gradient row sums to zero), P1' (no NaN) and P2 (a zero gradient at
identical logits). properties.py defines the properties, the inputs and the tolerances.

- test_p1_<loss>_<mode> and test_p2_<loss>_<mode>: Hypothesis tests of one loss and dtype
  mode, each over NumPy, the CPU kernels (backend "kernel" and "auto"), the composed
  operations (reported separately: they run PyTorch's operations) and the CUDA kernels.
  Profiles (conftest.py): ci by default (the explicit examples and a few hundred derandomized
  ones), search (--hypothesis-profile=search: 10,000 examples a test, a seed given with
  --hypothesis-seed), replay (the explicit examples only, for the CUDA run).
- test_sweep_p1 and test_sweep_p2 (markers sweep and long): the batched sweep, SWEEP_ROWS rows
  per loss and mode from the same generators with the seed SWEEP_SEED, one call per batch,
  on the CPU and CUDA kernels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("hypothesis")

import interfaces  # noqa: E402
import libsdca  # noqa: E402
import libsdca.torch as L  # noqa: E402
import properties as P  # noqa: E402
import torch  # noqa: E402

CUDA = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
]
# (framework, device, backend option)
Backend = tuple[str, str, str]
BACKENDS = [
    pytest.param(("numpy", "cpu", "kernel"), id="numpy"),
    pytest.param(("torch", "cpu", "kernel"), id="torch-cpu-kernel"),
    pytest.param(("torch", "cpu", "auto"), id="torch-cpu-auto"),
    pytest.param(("torch", "cpu", "composed"), id="torch-cpu-composed", marks=P.COMPOSED),
    # CUDA: the kernels only ("auto" runs the same kernels there).
    pytest.param(("torch", "cuda", "kernel"), id="torch-cuda-kernel", marks=CUDA),
]
SWEEP_BACKENDS = [
    pytest.param(("torch", "cpu", "kernel"), id="torch-cpu-kernel"),
    pytest.param(("torch", "cuda", "kernel"), id="torch-cuda-kernel", marks=CUDA),
]


def _numpy_dtype(dtype: str) -> Any:
    if dtype == "bfloat16":
        return pytest.importorskip("ml_dtypes").bfloat16
    return np.dtype(dtype)


def call_numpy(family: str, mode: str, batch: P.Batch) -> tuple[np.ndarray, np.ndarray, str]:
    """(values, gradient as float64, gradient dtype) of libsdca.loss_and_grad."""
    dtype, precision = P.MODES[mode]
    storage = _numpy_dtype(dtype)
    t = P.target(family, batch)
    if family == P.PROBABILITIES:
        target: Any = libsdca.Probabilities(t)
    elif family in P.TEACHER:
        target = t.astype(storage)
    else:
        target = t
    loss = interfaces.numpy_loss(family, P.parameters(family, batch))
    z = batch.logits.astype(storage)
    values, gradient = libsdca.loss_and_grad(
        z, target, loss, precision=precision, **P.options(family, batch)
    )
    return np.asarray(values), np.asarray(gradient, np.float64), str(gradient.dtype)


def call_torch(
    family: str, mode: str, batch: P.Batch, option: str, device: str
) -> tuple[np.ndarray, np.ndarray, str]:
    """(values, gradient as float64, gradient dtype) of the libsdca.torch loss, the
    gradient of the values with an all-ones cotangent."""
    dtype, precision = P.MODES[mode]
    kind = getattr(torch, dtype)

    def tensor(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.asarray(x, np.float64), device=device).to(kind)

    z = tensor(batch.logits).requires_grad_()
    t = P.target(family, batch)
    target = tensor(t) if t.dtype.kind == "f" else torch.tensor(t, device=device)
    opts = P.options(family, batch)
    if "weight" in opts:
        opts["weight"] = tensor(opts["weight"])
    value = interfaces.framework_call(
        L,
        family,
        z,
        target,
        P.parameters(family, batch),
        reduction=batch.reduction,
        backend=option,
        precision=precision,
        **opts,
    )
    value.backward(torch.ones_like(value))
    assert z.grad is not None
    out = str(z.grad.dtype).removeprefix("torch.")
    return value.detach().double().cpu().numpy(), z.grad.double().cpu().numpy(), out


def call(backend: Backend, family: str, mode: str, batch: P.Batch) -> tuple[Any, ...]:
    framework, device, option = backend
    if framework == "numpy":
        return call_numpy(family, mode, batch)
    return call_torch(family, mode, batch, option, device)


P.install(globals(), call, BACKENDS)

P.install_documented(
    globals(),
    call,
    [
        pytest.param(("numpy", "cpu", "kernel"), id="numpy"),
        pytest.param(("torch", "cpu", "kernel"), id="torch-cpu-kernel"),
    ],
)


@pytest.mark.sweep
@pytest.mark.long
@pytest.mark.parametrize("backend", SWEEP_BACKENDS)
@pytest.mark.parametrize("mode", list(P.MODES))
@pytest.mark.parametrize("family", P.FAMILIES)
def test_sweep_p1(family: str, mode: str, backend: Backend) -> None:
    P.sweep(call, "p1", family, mode, backend)


@pytest.mark.sweep
@pytest.mark.long
@pytest.mark.parametrize("backend", SWEEP_BACKENDS)
@pytest.mark.parametrize("mode", list(P.MODES))
@pytest.mark.parametrize("family", P.P2_FAMILIES)
def test_sweep_p2(family: str, mode: str, backend: Backend) -> None:
    P.sweep(call, "p2", family, mode, backend)
