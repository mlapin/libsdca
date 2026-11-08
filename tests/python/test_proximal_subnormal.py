"""A confident proximal loss uses the recovered mass, including subnormal values.

For these represented logits and the selected alphas the proximal correction to cross-entropy
is O(alpha * exp(-2 * gap)), many orders below an output subnormal quantum. Thus the
100-digit cross-entropy value is also an oracle for the rounded proximal value.
"""

import os
from unittest import mock

import mpmath as mp
import pytest
import torch
from libsdca.torch import proximal_cross_entropy


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["auto", "torch"])
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
    device: str, backend: str, dtype: torch.dtype, gap: float, classes: int
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
    with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": "off"}):
        value = proximal_cross_entropy(z, y, alpha=1.0, backend=backend)
        (gradient,) = torch.autograd.grad(value, z)
    rounded = torch.tensor(expected, dtype=value.dtype).item()
    quantum = torch.finfo(value.dtype).tiny * torch.finfo(value.dtype).eps
    assert value.item() > 0, (device, backend, dtype, classes, rounded)
    assert abs(value.item() - rounded) <= quantum
    target = torch.tensor(-expected, dtype=dtype).item()
    grad_quantum = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
    assert abs(gradient[0, 1].item() - target) <= grad_quantum


@pytest.mark.parametrize(
    "device,backend,extension",
    [
        ("cpu", "auto", "off"),
        ("cpu", "auto", "on"),
        ("cpu", "torch", "off"),
        ("cuda", "auto", "off"),
        ("cuda", "torch", "off"),
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
    backend: str,
    extension: str,
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
    with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": extension}):
        value = proximal_cross_entropy(z, y, alpha=alpha, backend=backend, precision=precision)
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
    "device,backend,extension",
    [
        ("cpu", "auto", "off"),
        ("cpu", "auto", "on"),
        ("cpu", "torch", "off"),
        ("cuda", "auto", "off"),
        ("cuda", "torch", "off"),
    ],
)
@pytest.mark.parametrize("classes", [1000, 4097])
def test_explicit_float32_aggregate(device: str, backend: str, extension: str, classes: int):
    test_many_underflowed_competitors(
        device, backend, extension, torch.float32, 110.0, classes, 1.0, "float32"
    )


@pytest.mark.parametrize(
    "device,backend", [("cpu", "auto"), ("cpu", "torch"), ("cuda", "auto"), ("cuda", "torch")]
)
@pytest.mark.parametrize(
    "dtype,gap,alpha,precision",
    [(torch.float64, 750.0, 1e308, None), (torch.float32, 110.0, 1e37, "float32")],
)
def test_subnormal_quantum_bounds_proximal_correction(
    device: str, backend: str, dtype: torch.dtype, gap: float, alpha: float, precision: str | None
):
    test_many_underflowed_competitors(device, backend, "on", dtype, gap, 4097, alpha, precision)
