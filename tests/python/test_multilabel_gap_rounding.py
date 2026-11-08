"""Small mask gradients retain represented score differences before exponentiation."""

import functools
import os
from unittest import mock

import mpmath as mp
import pytest
import torch
from libsdca import _cpu_entropy
from libsdca.torch import sdca_loss

ROWS = [
    [40.1234567, 40.1234567, -29.3456789, -29.7654321],
    [20.1234567, 20.1234567, -20.3456789, -20.7654321],
    [50.1234567, 50.1234567, -30.3456789, -30.7654321],
]


@functools.cache
def reference(row: tuple[float, ...]) -> list[float]:
    with mp.workdps(80):
        terms = [mp.exp(mp.mpf(v) - row[0]) for v in row]
        denominator = sum(terms)
        return [
            float(v / denominator - (mp.mpf("0.5") if j < 2 else 0)) for j, v in enumerate(terms)
        ]


@pytest.mark.parametrize("device", ["cpu", *(["cuda"] if torch.cuda.is_available() else [])])
@pytest.mark.parametrize("extension", ["off", *(["on"] if _cpu_entropy.buildable() else [])])
@pytest.mark.parametrize("backend", ["auto", "torch"])
@pytest.mark.parametrize("precision", [None, "float64"])
@pytest.mark.parametrize("classes", [4, 1000])
@pytest.mark.parametrize("tied", [False, True])
def test_tied_positives_large_nonintegral_gap(
    device: str, extension: str, backend: str, precision: str | None, classes: int, tied: bool
):
    base = torch.tensor(
        ROWS + [[10.1234567, 10.1234567, -10.3456789, -10.7654321]],
        dtype=torch.float32,
        device=device,
    )
    z = base[:, 2:].repeat(1, classes // 2)
    z[:, :2] = base[:, :2]
    if not tied:
        z[:, 0] = torch.nextafter(z[:, 0], torch.full_like(z[:, 0], -torch.inf))
    z.requires_grad_()
    mask = torch.zeros_like(z, dtype=torch.bool)
    mask[:, :2] = True
    with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": extension}):
        value = sdca_loss(
            z,
            mask,
            "multilabel_cross_entropy",
            backend=backend,
            precision=precision,
            reduction="none",
        )
        gradient = torch.autograd.grad(value.sum(), z)[0]
    expected = torch.tensor(
        [reference(tuple(row)) for row in z.detach().cpu().tolist()], dtype=torch.float64
    )
    scale = expected.abs().amax(-1, keepdim=True)
    error = ((gradient.cpu().double() - expected).abs() / scale).max().item()
    # Mixed-precision kernels have tighter budgets than the all-float32
    # tensor composition (the latter uses the release oracle's 64 eps).
    budget = 64 if backend == "torch" and precision is None else 8 if device == "cuda" else 2.5
    assert error <= budget * torch.finfo(torch.float32).eps, (
        device,
        extension,
        backend,
        precision,
        error,
    )
