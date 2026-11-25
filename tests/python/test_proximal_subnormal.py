"""A confident proximal loss uses the recovered mass, including subnormal values.

For these represented logits and the selected alphas the proximal correction to cross-entropy
is O(alpha * exp(-2 * gap)), many orders below an output subnormal quantum. Thus the
100-digit cross-entropy value is also an oracle for the rounded proximal value.
"""

import mpmath as mp
import pytest
import torch
from libsdca.torch import Backend, proximal_cross_entropy

# The CPU kernels lose the relative accuracy of a proximal loss whose competitors' terms each
# underflow the work type, though their sum does not (docs/API.md, Known limits).
LIMIT = pytest.mark.xfail(strict=True, reason="known limit: proximal tails beyond exp's range")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["auto", "composed"])
@pytest.mark.parametrize(
    "dtype,gap",
    [
        (torch.float16, 102.0),
        (torch.bfloat16, 102.0),
        (torch.float32, 102.0),
        (torch.float64, 744.0),
    ],
)
@pytest.mark.parametrize("classes", [2, 4097])
def test_one_tiny_competitor(
    device: str, backend: Backend, dtype: torch.dtype, gap: float, classes: int
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # A non-leading label exercises pivot indexing; only one tail term is representable.
    z = torch.full((1, classes), -2000.0, dtype=dtype, device=device)
    z[0, 1], z[0, 0] = 0.0, -gap
    z.requires_grad_()
    y = torch.ones(1, dtype=torch.long, device=device)
    with mp.workdps(100):
        expected = float(mp.log1p(mp.exp(-gap)))
    value = proximal_cross_entropy(z, y, gamma=1.0, backend=backend)
    (gradient,) = torch.autograd.grad(value, z)
    rounded = torch.tensor(expected, dtype=value.dtype).item()
    quantum = torch.finfo(value.dtype).tiny * torch.finfo(value.dtype).eps
    assert value.item() > 0, (device, backend, dtype, classes, rounded)
    assert abs(value.item() - rounded) <= quantum
    target = torch.tensor(-expected, dtype=dtype).item()
    grad_quantum = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
    assert abs(gradient[0, 1].item() - target) <= grad_quantum


@pytest.mark.parametrize(
    "device,backend",
    [
        pytest.param("cpu", "auto", marks=LIMIT),
        ("cpu", "composed"),
        ("cuda", "auto"),
        ("cuda", "composed"),
    ],
)
@pytest.mark.parametrize(
    "dtype,gap",
    [
        (torch.float16, 110.0),
        (torch.bfloat16, 110.0),
        (torch.float32, 110.0),
        (torch.float64, 750.0),
    ],
)
@pytest.mark.parametrize("classes", [1000, 4097])
@pytest.mark.parametrize("alpha", [1.0, 1e10])
def test_many_underflowed_competitors(
    device: str,
    backend: Backend,
    dtype: torch.dtype,
    gap: float,
    classes: int,
    alpha: float,
    precision: str | None = None,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    z = torch.full((1, classes), -gap, dtype=dtype, device=device)
    z[0, 1] = 0
    z.requires_grad_()
    y = torch.ones(1, dtype=torch.long, device=device)
    with mp.workdps(100):
        expected = float(mp.log1p((classes - 1) * mp.exp(-gap)))
    value = proximal_cross_entropy(z, y, gamma=alpha, backend=backend, precision=precision)
    (gradient,) = torch.autograd.grad(value, z)
    rounded = torch.tensor(expected, dtype=value.dtype).item()
    quantum = torch.finfo(value.dtype).tiny * torch.finfo(value.dtype).eps
    assert value.item() > 0, (device, backend, dtype, classes, alpha, rounded)
    assert abs(value.item() - rounded) <= quantum
    target = torch.tensor(-expected, dtype=dtype).item()
    grad_quantum = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
    assert abs(gradient[0, 1].item() - target) <= grad_quantum
    gradient[0, 1] = 0
    assert torch.count_nonzero(gradient).item() == 0


@pytest.mark.parametrize(
    "device,backend",
    [
        pytest.param("cpu", "auto", marks=LIMIT),
        ("cpu", "composed"),
        ("cuda", "auto"),
        ("cuda", "composed"),
    ],
)
@pytest.mark.parametrize("classes", [1000, 4097])
def test_explicit_float32_aggregate(device: str, backend: Backend, classes: int):
    test_many_underflowed_competitors(
        device, backend, torch.float32, 110.0, classes, 1.0, "float32"
    )


@pytest.mark.parametrize(
    "device,backend",
    [
        pytest.param("cpu", "auto", marks=LIMIT),
        ("cpu", "composed"),
        ("cuda", "auto"),
        ("cuda", "composed"),
    ],
)
@pytest.mark.parametrize(
    "dtype,gap,alpha,precision",
    [(torch.float64, 750.0, 1e308, None), (torch.float32, 110.0, 1e37, "float32")],
)
def test_subnormal_quantum_bounds_proximal_correction(
    device: str,
    backend: Backend,
    dtype: torch.dtype,
    gap: float,
    alpha: float,
    precision: str | None,
):
    test_many_underflowed_competitors(device, backend, dtype, gap, 4097, alpha, precision)


@pytest.mark.cuda
def test_masked_competitor_with_step_above_float32_range():
    """The two supported classes have a positive envelope and residual below float32 range."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    z = torch.tensor(
        [[0.0, 1.0, -torch.inf]], device="cuda", dtype=torch.float32, requires_grad=True
    )
    value = proximal_cross_entropy(z, torch.tensor([0], device="cuda"), gamma=1e60)
    (gradient,) = torch.autograd.grad(value, z)
    assert value.item() == 0 and not torch.signbit(value).item()
    assert torch.equal(gradient, torch.zeros_like(gradient))
    assert torch.signbit(gradient).tolist() == [[True, False, False]]
