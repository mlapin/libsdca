"""The precision policy of the PyTorch and native losses (docs/API.md#precision).

precision=None and "float32" promise float32 work for 16- and 32-bit logits and take the same
route; "float64" keeps float64 throughout; float64 logits reject "float32".
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch
from libsdca import _cpu_entropy, loss_and_grad
from libsdca import torch as sdca_torch
from libsdca.torch import explain_backend, sdca_loss

# (loss, k, gamma, mask target)
LOSSES = [
    ("cross_entropy", 1, 0.0, False),
    ("cross_entropy", 1, 0.0, True),
    ("topk_cross_entropy", 3, 0.0, False),
    ("topk_cross_entropy", 70, 0.0, False),
    ("multiclass_hinge", 1, 0.0, False),
    ("multiclass_hinge", 1, 0.5, False),
    ("topk_hinge_alpha", 3, 0.0, False),
    ("topk_hinge_alpha", 3, 0.5, False),
    ("topk_hinge_beta", 3, 0.5, False),
    ("multilabel_hinge", 1, 0.0, True),
    ("multilabel_hinge", 1, 0.5, True),
    ("proximal_cross_entropy", 1, 0.0, False),
]
SMOOTH = [(loss, k, gamma, mask) for loss, k, gamma, mask in LOSSES if gamma > 0]


def _problem(dtype: torch.dtype, mask: bool, device: str = "cpu", classes: int = 80):
    generator = torch.Generator().manual_seed(7)
    z = (3 * torch.randn(16, classes, generator=generator)).to(dtype).to(device)
    labels = torch.randint(0, classes, (16,), generator=generator).to(device)
    if not mask:
        return z, labels
    target = torch.nn.functional.one_hot(labels, classes).bool()
    target[:, :2] = True
    return z, target


def _value_and_gradient(z: torch.Tensor, target: torch.Tensor, loss: str, **options: Any):
    x = z.detach().clone().requires_grad_()
    value = sdca_loss(x, target, loss, reduction="none", **options)
    (gradient,) = torch.autograd.grad(value.sum(), x)
    return value.detach(), gradient


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("loss,k,gamma,mask", LOSSES)
def test_float32_is_the_default_route(
    loss: str, k: int, gamma: float, mask: bool, dtype: torch.dtype
):
    z, target = _problem(dtype, mask)
    for backend in ("auto", "native", "torch"):
        if backend == "native" and dtype != torch.float32:
            continue
        options: dict[str, Any] = dict(k=k, gamma=gamma, backend=backend)
        assert explain_backend(z, target, loss, precision="float32", **options) == (
            explain_backend(z, target, loss, **options)
        )
        default = _value_and_gradient(z, target, loss, **options)
        single = _value_and_gradient(z, target, loss, precision="float32", **options)
        for a, b in zip(default, single, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=f"{loss} {backend}")


@pytest.mark.parametrize("loss,k,gamma,mask", LOSSES)
def test_float64_option_is_float64(loss: str, k: int, gamma: float, mask: bool):
    z, target = _problem(torch.float32, mask)
    _, reason = explain_backend(z, target, loss, k=k, gamma=gamma, precision="float64")
    assert "float64" in reason and "float32" not in reason, reason
    with pytest.raises(ValueError, match="float64 logits require"):
        sdca_loss(z.double(), target, loss, k=k, gamma=gamma, precision="float32")


@pytest.mark.skipif(not _cpu_entropy.enabled(), reason="needs the CPU extension")
def test_cpu_routes_of_float32_work():
    """The CPU extension's kernels for every loss (the hinges in float64, as fast), and the
    float32 proximal kernel for float32 logits as for 16-bit ones."""
    for loss, k, gamma, mask in LOSSES:
        z, target = _problem(torch.float32, mask)
        name, reason = explain_backend(z, target, loss, k=k, gamma=gamma, precision="float32")
        if "hinge" in loss:
            assert name == "cpu-kernel" and "(float64)" in reason, (loss, reason)
        if loss == "proximal_cross_entropy":
            assert "kernel (float32" in reason, reason
            half = explain_backend(z.bfloat16(), target, loss, precision=None)
            assert half == (name, reason)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_float32_takes_the_fused_kernels(dtype: torch.dtype):
    """precision="float32" on CUDA takes the kernels of None, not float64 tensor operations."""
    for loss, k, gamma, mask in LOSSES:
        if loss == "topk_cross_entropy" and k > 32:
            continue
        z, target = _problem(dtype, mask, "cuda")
        options: dict[str, Any] = dict(k=k, gamma=gamma)
        route = explain_backend(z, target, loss, precision="float32", **options)
        assert route == explain_backend(z, target, loss, **options), loss
        assert route[0] != "torch", (loss, route)


def test_proximal_kernel_takes_float32_work_for_float32_logits():
    """The proximal kernel (CUDA, mocked here) takes float32 for float32 logits under None, as
    for 16-bit ones; the tensor operations, "float64" and float64 logits keep float64."""
    z = torch.randn(4, 6)
    work = sdca_torch._device_working
    loss = "proximal_cross_entropy"
    with mock.patch.object(sdca_torch, "_prox_kernel", lambda flat: object()):
        assert work(z, z, None, loss, True) == "float32"
        assert work(z.bfloat16(), z, None, loss, True) == "float32"
        assert work(z, z, None, loss, False) is None
        assert work(z, z, "float64", loss, True) == "float64"
        assert work(z.double(), z, None, loss, True) is None
        assert work(z, z, None, "multiclass_hinge", True) is None
    with mock.patch.object(sdca_torch, "_prox_kernel", lambda flat: None):
        assert work(z, z, None, loss, True) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_proximal_default_is_float32_work():
    z, y = _problem(torch.float32, False, "cuda")
    name, reason = explain_backend(z, y, "proximal_cross_entropy")
    assert name == "triton" and "float32" in reason, reason
    name, reason = explain_backend(z, y, "proximal_cross_entropy", precision="float64")
    assert name == "triton" and "float32" not in reason, reason


def _native_problem(mask: bool, classes: int = 80):
    rng = np.random.default_rng(11)
    z = (2 * rng.standard_normal((64, classes))).astype(np.float32)
    labels = rng.integers(0, classes, 64)
    if not mask:
        return z, labels
    target = np.eye(classes, dtype=bool)[labels]
    target[:, :3] = True
    return z, target


@pytest.mark.parametrize("loss,k,gamma,mask", LOSSES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_default_follows_storage(
    loss: str, k: int, gamma: float, mask: bool, dtype: type[np.float32] | type[np.float64]
):
    z, target = _native_problem(mask)
    z = z.astype(dtype)
    default = loss_and_grad(z, target, loss, k=k, gamma=gamma)
    explicit = loss_and_grad(z, target, loss, k=k, gamma=gamma, precision=dtype)
    for a, b in zip(default, explicit, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("loss,k,gamma,mask", SMOOTH)
def test_native_float32_smoothed_hinges_form_entries_in_float64(
    loss: str, k: int, gamma: float, mask: bool
):
    z, target = _native_problem(mask)
    default = loss_and_grad(z, target, loss, k=k, gamma=gamma)
    single = loss_and_grad(z, target, loss, k=k, gamma=gamma, precision="float32")
    for a, b in zip(default, single, strict=True):
        np.testing.assert_array_equal(a, b)


def test_native_float32_proximal_sums_its_loss_in_double():
    # 4,096 equal competitors 20 below the target: alpha |out|^2 summed in float drifted by
    # about a thousand float32 eps.
    z = np.full((1, 4097), -20.0, np.float32)
    z[0, 0] = 0
    exact = loss_and_grad(z.astype(np.float64), [0], "proximal_cross_entropy", alpha=1e3)[0]
    single = loss_and_grad(z, [0], "proximal_cross_entropy", alpha=1e3, precision="float32")[0]
    assert abs(single[0] - exact[0]) <= 64 * np.finfo(np.float32).eps * abs(exact[0])
