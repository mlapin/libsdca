"""CUDA inference omits derivative statistics without changing values or training."""

import contextlib

import pytest
import torch
from libsdca.torch import distillation_loss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_distillation_inference_and_training_routes(dtype: torch.dtype, reduction: str):
    pytest.importorskip("triton")
    from libsdca import _triton_distillation as kernels

    # Each parameter checks several grad-mode guards; isolate its Dynamo cache budget.
    torch._dynamo.reset()
    torch.manual_seed(1729)
    student = torch.randn(3, 4097, device="cuda").to(dtype)
    teacher = (student.float() + 0.5 * torch.randn_like(student.float())).to(dtype)
    compiled = torch.compile(distillation_loss, fullgraph=True)
    expected, stats = kernels.forward(student, teacher, 2.0, False, reduction)
    for mode in ("detached", "no_grad", "inference"):
        x = student.detach().requires_grad_(mode != "detached")
        t = teacher.detach().requires_grad_()
        context = {
            "detached": contextlib.nullcontext,
            "no_grad": torch.no_grad,
            "inference": torch.inference_mode,
        }[mode]
        with context():
            for fn in (distillation_loss, compiled):
                value = fn(x, t, 2.0, reduction=reduction)
                assert not value.requires_grad
                torch.testing.assert_close(
                    value, expected, rtol=8 * torch.finfo(value.dtype).eps, atol=0
                )
    # A reused compiled callable must guard grad mode and requires_grad correctly.
    upstream = torch.ones_like(expected)
    expected_grad = kernels.backward(student, teacher, stats, upstream, 2.0, False, reduction)
    for fn in (distillation_loss, compiled):
        x = student.detach().requires_grad_()
        t = teacher.detach().requires_grad_()
        value = fn(x, t, 2.0, reduction=reduction)
        gradient, teacher_gradient = torch.autograd.grad(value, (x, t), upstream, allow_unused=True)
        torch.testing.assert_close(gradient, expected_grad, rtol=0, atol=0)
        assert teacher_gradient is None
