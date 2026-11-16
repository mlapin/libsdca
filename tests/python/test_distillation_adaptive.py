"""Mixed ordinary and sensitive rows exercise both paths of adaptive CUDA distillation."""

import pytest
import torch
from libsdca.torch import distillation_loss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def mixed_inputs(classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(483)
    s = 3 * torch.randn(8, classes, device="cuda", generator=generator)
    t = 3 * torch.randn(8, classes, device="cuda", generator=generator)
    t[1] = s[1]
    s[2].fill_(-90)
    s[2, 0] = 0
    t[2] = 2 * s[2]
    s[3] += 10000
    s[4].zero_()
    s[4, 0] = 25
    s[5, 7] = -120
    s[6].zero_()
    t[6].zero_()
    s[6, 7], t[6, 7] = -158, -160
    return s.repeat(128, 1), t.repeat(128, 1)


@pytest.mark.parametrize("classes", [300, 1000, 4096, 16384, 32000, 32768])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("launcher", ["on", "off"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mixed_adaptive_rows(
    monkeypatch: pytest.MonkeyPatch, classes: int, reduction: str, launcher: str, dtype: torch.dtype
) -> None:
    monkeypatch.setenv("SDCA_CUDA_LAUNCHER", launcher)
    s, t = (x.to(dtype) for x in mixed_inputs(classes))
    temperature = 2.0
    s.requires_grad_()
    actual = distillation_loss(s, t, temperature, reduction=reduction)
    exact = distillation_loss(s, t, temperature, reduction=reduction, precision="float64")
    weight = (
        torch.linspace(0.25, 2, len(s), device="cuda")
        if reduction == "none"
        else torch.ones_like(actual) * 1.7
    )
    (g,) = torch.autograd.grad(actual, s, weight)
    (reference,) = torch.autograd.grad(exact, s, weight)
    eps = torch.finfo(torch.float32).eps
    assert torch.all((actual - exact).abs() <= 64 * eps * exact.abs() + 2**-149)
    p, q = torch.softmax(t.double() / temperature, -1), torch.softmax(s.double() / temperature, -1)
    scale = (
        weight[:, None] if reduction == "none" else weight / (len(s) if reduction == "mean" else 1)
    )
    bound = 64 * eps * (p + q) / temperature * scale
    output = torch.finfo(dtype)
    bound += output.eps * reference.double().abs() + output.tiny * output.eps
    assert torch.all((g.double() - reference.double()).abs() <= bound)
    assert not torch.count_nonzero(g[1::8])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_adaptive_invalid_rows_remain_invalid(bad: float) -> None:
    s, t = mixed_inputs(1000)
    s[0, 50] = bad
    s.requires_grad_()
    loss = distillation_loss(s, t, reduction="none")
    (gradient,) = torch.autograd.grad(loss.sum(), s)
    assert torch.isnan(loss[0])
    assert torch.isnan(gradient[0]).all()
    assert torch.isfinite(loss[1:]).all()
    assert torch.isfinite(gradient[1:]).all()
