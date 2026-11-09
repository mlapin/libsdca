"""The routing of backend="torch" and "native" off the CPU, checked on the CPU.

Every fused-kernel probe is patched to accept the call and every kernel to raise, so a call
that reaches a kernel fails. backend="auto" must reach them; backend="torch" must not, and
must give the tensor operations' result.
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import libsdca
import pytest
import torch
from libsdca import _entropy, _multilabel
from libsdca import torch as sdca_torch
from libsdca.torch import explain_backend, sdca_loss

CLASSES = 6


class KernelCalled(Exception):
    pass


def _raise(*args: object, **kwargs: object) -> Any:
    raise KernelCalled


def cuda_like(shape: torch.Size) -> torch.Tensor:
    """A stand-in for CUDA logits of this shape, for the shape and device checks alone."""
    return cast(torch.Tensor, SimpleNamespace(is_cuda=True, shape=shape, numel=shape.numel))


def problem(loss: str) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(3)
    z = torch.randn(5, CLASSES, dtype=torch.float64)
    y = torch.arange(5) % CLASSES
    if "multilabel" in loss:
        return z, torch.nn.functional.one_hot(y, CLASSES).bool()
    return z, y


def cases() -> list[tuple[str, int, float]]:
    out = []
    for loss in libsdca.LOSSES:
        ks = (1, 2) if "topk" in loss else (1,)
        gammas = (0.0, 0.5) if loss in libsdca.HINGES else (0.0,)
        out += [(loss, k, gamma) for k in ks for gamma in gammas]
    return out


def kernels_everywhere() -> list[Any]:
    """Patch every fused-kernel probe to accept and every kernel to raise KernelCalled."""
    real_entropy = sdca_torch._indexed_entropy
    real_applies = _multilabel.applies

    def entropy(
        logits: torch.Tensor,
        target: torch.Tensor,
        reduction: str,
        fp32: bool = False,
        is_fused: bool | None = None,
    ) -> torch.Tensor:
        if is_fused:
            raise KernelCalled
        return real_entropy(logits, target, reduction, fp32=fp32, is_fused=is_fused)

    def applies(
        objective: str,
        logits: torch.Tensor,
        target: torch.Tensor,
        classes: int,
        backend: str,
        precision: str | None,
        reduction: str,
    ) -> bool:
        # The real decision on a CUDA stand-in with a Triton kernel available.
        cuda = cuda_like(logits.shape)
        with mock.patch.object(_multilabel, "_kernel", lambda: SimpleNamespace(covers=bool)):
            return real_applies(objective, cuda, target, classes, backend, precision, reduction)

    return [
        mock.patch.object(sdca_torch, "_fused_entropy", lambda flat: True),
        mock.patch.object(sdca_torch, "_indexed_entropy", entropy),
        mock.patch.object(sdca_torch._hinge, "covers", lambda flat: True),
        mock.patch.object(sdca_torch._hinge, "hinge", _raise),
        mock.patch.object(sdca_torch._multilabel, "applies", applies),
        mock.patch.object(sdca_torch._multilabel, "loss", _raise),
        mock.patch.object(sdca_torch._topk, "covers", lambda flat, objective, k: True),
        mock.patch.object(sdca_torch._topk, "topk", _raise),
        mock.patch.object(sdca_torch, "_prox_kernel", lambda logits: _raise),
    ]


def device_loss(
    z: torch.Tensor, target: torch.Tensor, loss: str, k: int, gamma: float, backend: str
) -> torch.Tensor:
    """The device path of sdca_loss (the CUDA dispatch, on CPU tensors)."""
    objective, kind_k, kind_gamma = libsdca._core._tensor_kind(loss, k, gamma)
    return sdca_torch._device_loss(
        z, z, target, objective, kind_k, kind_gamma, 1.0, "none", -1, None, backend
    )


def test_torch_backend_reaches_no_kernel(subtests: pytest.Subtests):
    expected = {}
    for loss, k, gamma in cases():
        z, target = problem(loss)
        expected[loss, k, gamma] = sdca_loss(
            z, target, loss, k=k, gamma=gamma, backend="torch", reduction="none"
        )
    patches = kernels_everywhere()
    for patch in patches:
        patch.start()
    try:
        for loss, k, gamma in cases():
            z, target = problem(loss)
            with subtests.test(loss=loss, k=k, gamma=gamma):
                value = device_loss(z, target, loss, k, gamma, "torch")
                torch.testing.assert_close(value, expected[loss, k, gamma], rtol=0, atol=0)
                name, reason = explain_backend(z, target, loss, k=k, gamma=gamma, backend="torch")
                assert name == "torch" and "backend='torch'" in reason
                # The same call with backend="auto" reaches a kernel.
                with pytest.raises(KernelCalled):
                    device_loss(z, target, loss, k, gamma, "auto")
    finally:
        for patch in patches:
            patch.stop()


def test_torch_backend_with_float32_precision():
    # precision="float32" keeps integer entropy on its kernel for "auto", never for "torch".
    z, y = problem("cross_entropy")
    expected = sdca_loss(z.float(), y, backend="torch", precision="float32", reduction="none")
    with mock.patch.object(sdca_torch._entropy, "_kernel", lambda: object()):
        with mock.patch.object(sdca_torch, "_fused_entropy", lambda flat: True):
            assert (
                sdca_torch._indexed_entropy_path(z, y, "cross_entropy", 1, True, fused=False)
                is False
            )
            value = sdca_torch._device_loss(
                z.float(),
                z.float(),
                y,
                "cross_entropy",
                1,
                1.0,
                1.0,
                "none",
                -1,
                "float32",
                "torch",
            )
    torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_entropy_without_kernel_never_launches():
    z, y = problem("cross_entropy")
    z.requires_grad_(True)
    kernel = SimpleNamespace(forward=_raise, backward=_raise, apply=_raise)
    with mock.patch.object(_entropy, "fused", lambda logits: True):
        with mock.patch.object(_entropy, "_kernel", lambda: kernel):
            value = _entropy.entropy(z, y, "sum", is_fused=False)
            (gradient,) = torch.autograd.grad(value, z)
    reference = torch.nn.functional.cross_entropy(z.detach(), y, reduction="sum")
    torch.testing.assert_close(value.detach(), reference)
    assert gradient.isfinite().all()


def test_multilabel_kernels_take_auto_only(subtests: pytest.Subtests):
    cuda = cuda_like(torch.Size([4, 5]))
    mask = torch.ones(4, 5, dtype=torch.bool)
    with mock.patch.object(_multilabel, "_kernel", lambda: SimpleNamespace(covers=bool)):
        for backend, applies in (("auto", True), ("torch", False), ("native", False)):
            with subtests.test(backend=backend):
                got = _multilabel.applies("multilabel_hinge", cuda, mask, 5, backend, None, "mean")
                assert got is applies


def test_native_backend_rejects_other_devices(subtests: pytest.Subtests):
    for loss, k, gamma in cases():
        z, target = problem(loss)
        z, target = z.to("meta"), target.to("meta")
        with subtests.test(loss=loss, k=k, gamma=gamma):
            with pytest.raises(ValueError, match="Native losses require CPU tensors"):
                sdca_loss(z, target, loss, k=k, gamma=gamma, backend="native")
            with pytest.raises(ValueError, match="Native losses require CPU tensors"):
                explain_backend(z, target, loss, k=k, gamma=gamma, backend="native")
