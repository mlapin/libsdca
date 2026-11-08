"""Wide float32-exponential mask entropy takes one scan per row: rows with a negative
maximum finish in it, the others continue with the residual scan."""

import math

import pytest
import torch
from libsdca.torch import multilabel_cross_entropy

try:
    from libsdca import _triton_multilabel
except ImportError:
    pytest.skip("Triton required", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
EPS = torch.finfo(torch.float32).eps
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def reference(z: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Float64 losses and gradients of the represented scores."""
    z, m = z.double(), mask.double()
    p = m.sum(1)
    value = torch.logsumexp(z, 1) - (z * m).sum(1) / p
    return value, torch.softmax(z, 1) - m / p[:, None]


def rows(classes: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Random, dense, offset, rising, falling and well-fit rows, and (when representable)
    gaps whose float32 sums overflow."""
    generator = torch.Generator().manual_seed(classes)
    noise = torch.randn(4, classes, generator=generator, dtype=torch.float64)
    ramp = torch.linspace(-20, 20, classes, dtype=torch.float64)
    sparse = torch.rand(4, classes, generator=generator) < 0.01
    dense = torch.rand(4, classes, generator=generator) < 0.5
    blocks = [
        (3 * noise, sparse),
        (30 * noise, dense),
        (1e4 + 3 * noise, sparse),
        (ramp + 0.1 * noise, sparse),
        (ramp.flip(0) + 0.1 * noise, dense),
        (torch.where(sparse, 3 * noise + 12, 3 * noise), sparse),
    ]
    if torch.finfo(dtype).max > 1e38:
        huge = torch.zeros(4, classes, dtype=torch.float64)
        huge[:, 0] = 1e38
        huge[:, 8:16] = -1e38
        flags = torch.zeros(4, classes, dtype=torch.bool)
        flags[:, 8:16] = True
        blocks.append((huge, flags))
    z = torch.cat([b[0] for b in blocks]).to(dtype)
    mask = torch.cat([b[1] for b in blocks])
    mask[:, 1] = True
    return z, mask


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("classes", [16384, 16385, 33000])
def test_online_rows_against_float64(dtype: torch.dtype, classes: int):
    assert _triton_multilabel._online(classes, "cross_entropy", True)
    z, mask = rows(classes, dtype)
    x = z.cuda().requires_grad_()
    value = multilabel_cross_entropy(x, mask.cuda(), reduction="none")
    (gradient,) = torch.autograd.grad(value.sum(), x)
    expected, ref = reference(z, mask)
    error = (value.detach().double().cpu() - expected).abs()
    tiny = torch.finfo(torch.float32).tiny * EPS
    assert torch.all(error <= 8 * EPS * expected.abs() + tiny), error.max()
    info = torch.finfo(dtype)
    budget = (info.eps + 8 * EPS) * ref.abs().amax(1, keepdim=True) + info.tiny * info.eps
    assert torch.all((gradient.double().cpu() - ref).abs() <= budget)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("column", [5, 16384])
@pytest.mark.parametrize("positive", [False, True])
def test_online_nonfinite_rows(bad: float, column: int, positive: bool):
    z, mask = rows(16385, torch.float32)
    z[2, column] = bad
    mask[2, column] = positive
    value = multilabel_cross_entropy(z.cuda(), mask.cuda(), reduction="none").cpu()
    assert value[2].isnan()
    keep = torch.arange(len(z)) != 2
    expected, _ = reference(z[keep], mask[keep])
    torch.testing.assert_close(value[keep].double(), expected, rtol=8 * EPS, atol=0)
