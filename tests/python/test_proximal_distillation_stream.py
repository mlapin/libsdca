"""Streamed float32-work proximal_distillation rows (more classes than a register row) on CUDA:
the chunk passes against float64 work, the general path's rows (masked, non-finite,
confident), identical rows' exact zero gradient and large steps."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA", allow_module_level=True)

from libsdca.torch import proximal_distillation_loss  # noqa: E402

pytestmark = pytest.mark.cuda


def _run(
    z: torch.Tensor,
    t: torch.Tensor,
    gamma: float,
    temperature: float,
    precision: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = z.detach().clone().requires_grad_()
    options: dict[str, Any] = {"precision": precision} if precision else {}
    v = proximal_distillation_loss(
        z, t, gamma=gamma, temperature=temperature, reduction="none", **options
    )
    up = torch.linspace(0.5, 1.5, v.numel(), device=v.device, dtype=v.dtype)
    (g,) = torch.autograd.grad(v, z, up)
    return v.detach().double(), g.detach().double()


def _rows(c: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(c)
    z = torch.randn(12, c, device="cuda", generator=gen) * 2
    t = torch.randn(12, c, device="cuda", generator=gen) * 2
    t[1] = z[1] + 1e-3 * torch.randn(c, device="cuda", generator=gen)  # near rows
    t[2] = z[2]  # identical
    z[3, 7] += 15  # confident student and teacher on one class
    t[3, 7] += 14
    t[4, 3] += 25  # confident teacher alone
    z[5, 5] += 25  # confident student alone
    z[6, 11] = -torch.inf  # masked student class: the general path
    t[7, 13] = -torch.inf  # masked teacher class
    z[8, 1:] = z[8, 0]  # ties
    t[9] = t[9] * 4
    z[10] = z[10] * 4
    return z.to(dtype), t.to(dtype)


@pytest.mark.parametrize("c", [4097, 50257])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gamma, temperature", [(1.0, 2.0), (0.1, 1.0), (10.0, 1.0)])
def test_stream_matches_float64(
    c: int, dtype: torch.dtype, gamma: float, temperature: float
) -> None:
    z, t = _rows(c, dtype)
    v, g = _run(z, t, gamma, temperature)
    v64, g64 = _run(z, t, gamma, temperature, "float64")
    torch.testing.assert_close(v, v64, rtol=2e-5, atol=0)
    # Gradients relative to each row's largest entry; BF16 storage rounds to 2^-9.
    scale = g64.abs().amax(1, keepdim=True)
    tol = 5e-3 if dtype == torch.bfloat16 else 2e-4
    assert ((g - g64).abs() <= tol * scale).all()
    assert torch.all(g[2] == 0)
    assert not torch.any((g == 0) & torch.signbit(g))


@pytest.mark.parametrize("c", [4097, 32767, 32769])
@pytest.mark.parametrize("gamma", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("gaps", [(1.0, 1.015625), (0.0, 1.0), (1.0, 0.0)])
def test_stream_near_rows_against_exact(c: int, gamma: float, gaps: tuple[float, float]) -> None:
    """Release numerics' near rows (student [0, -a, ...], teacher [0, -b, ...], T = 2): the
    others' gradient (1e-6 of p down to 2e-7 of it) within (64 + C) eps of the 100-digit
    reference. The solve's tail sum must not be the float32 moments' (about 1e-7 of the tail
    mass) less the tail's p mass, which would move omega by as much; and the bracket of
    -gamma max(q - p) must contain the root despite the normalizers' log ratio error."""
    import mpmath as mp
    from conformance.references import Block, Loss, reference

    student, teacher = gaps
    z = torch.full((1, c), -student, device="cuda")
    t = torch.full((1, c), -teacher, device="cuda")
    z[0, 0] = t[0, 0] = 0
    x = z.clone().requires_grad_()
    v = proximal_distillation_loss(x, t, gamma=gamma, temperature=2.0, reduction="none")
    v.sum().backward()
    assert x.grad is not None
    with mp.workdps(60):
        loss = Loss("proximal_distillation", temperature=2.0, alpha=gamma)
        ref = reference(
            loss, (Block(0.0, 1, teacher=0.0), Block(-student, c - 1, teacher=-teacher))
        )
        eps = torch.finfo(torch.float32).eps
        budget = (64 + c) * eps
        checks = [(v[0].item(), ref.value), (x.grad[0, 0].item(), ref.gradient[0])]
        checks += [(g, ref.gradient[1]) for g in torch.unique(x.grad[0, 1:]).tolist()]
        for actual, exact in checks:
            assert abs(mp.mpf(actual) - exact) <= budget * abs(exact) + 2.0**-149, (actual, exact)
