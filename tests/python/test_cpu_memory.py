"""CPU calls without gradient: the gradient kernels run on row blocks, with the whole call's
values bit for bit."""

from __future__ import annotations

import os
from collections.abc import Callable
from unittest import mock

import libsdca.torch as sdca
import pytest
import torch
from libsdca import _cpu, _cpu_entropy

DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.float64]


def _settings() -> list[str]:
    """The SDCA_CPU_EXTENSION settings to test: the extension unless it is off or unavailable."""
    if _cpu_entropy.mode() == "off" or not _cpu_entropy.enabled():
        return ["off"]
    return ["off", "on"]


def _losses(
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[Callable[[torch.Tensor], torch.Tensor]]]:
    """Logits and the gradient kernels' losses of them (native and extension), per row."""
    g = torch.Generator().manual_seed(0)
    n, c = 203, 37
    z = (3 * torch.randn(n, c, generator=g)).to(dtype)
    teacher = (3 * torch.randn(n, c, generator=g)).to(dtype)
    y = torch.randint(0, c, (n,), generator=g)
    mask = torch.rand(n, c, generator=g) < 0.2
    mask[:, 0], mask[:, 1] = True, False
    return z, [
        lambda x: sdca.sdca_loss(
            x, y, "topk_cross_entropy", k=3, backend="native", reduction="none"
        ),
        lambda x: sdca.sdca_loss(
            x, y, "topk_hinge_beta", k=3, gamma=0.5, backend="native", reduction="none"
        ),
        lambda x: sdca.sdca_loss(x, mask, "multilabel_hinge", backend="native", reduction="none"),
        lambda x: sdca.proximal_cross_entropy(x, y, 0.7, reduction="none"),
        lambda x: sdca.proximal_distillation_loss(x, teacher, 2.0, 0.7, reduction="none"),
    ]


@pytest.mark.parametrize("setting", _settings())
@pytest.mark.parametrize("dtype", DTYPES)
def test_blocks_without_gradient_give_the_values_of_one_call(
    setting: str, dtype: torch.dtype
) -> None:
    z, losses = _losses(dtype)
    with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": setting}):
        for index, loss in enumerate(losses):
            x = z.clone().requires_grad_(True)
            whole = loss(x).detach()
            # Blocks of four rows; the kernels keep their threads.
            with (
                mock.patch.object(_cpu, "_BLOCKS", 64),
                mock.patch.object(_cpu, "_BLOCK_BYTES", 1),
                mock.patch.object(_cpu, "_THREAD_ROWS", 0),
            ):
                for blocked in (loss(z), _no_grad(loss, x)):
                    torch.testing.assert_close(blocked, whole, rtol=0, atol=0, msg=f"loss {index}")
            assert loss(z).equal(whole), index


@pytest.mark.parametrize("setting", _settings())
def test_values_without_gradient_do_not_depend_on_the_threads(setting: str) -> None:
    z, losses = _losses(torch.float32)
    threads = torch.get_num_threads()
    with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": setting}):
        expected = [loss(z) for loss in losses]
        try:
            torch.set_num_threads(1)
            for index, loss in enumerate(losses):
                assert loss(z).equal(expected[index]), index
        finally:
            torch.set_num_threads(threads)


def _no_grad(loss: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return loss(x)
