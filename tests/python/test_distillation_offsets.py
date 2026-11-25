"""Independent common offsets must not erase class differences in distillation."""

from typing import Any

import numpy as np
import pytest
import torch
from conformance.references import Loss, evaluate
from libsdca.torch import distillation_loss

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def offset_rows(dtype: type[np.floating[Any]], classes: int):
    huge = 1e30 if dtype == np.float32 else 1e300
    uniform = np.full(classes, huge, dtype=dtype)
    varying = (np.arange(classes) % 2 + 1).astype(dtype)
    # More than one SIMD block exercises the small-class CPU batching path too.
    student = np.stack([uniform, varying, -uniform] * 3)
    teacher = np.stack([varying, uniform, varying] * 3)
    return student, teacher


@pytest.mark.parametrize("backend", ["kernel", "composed"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("classes", [2, 17, 4097])
def test_independent_offsets(
    backend: str, device: str, dtype: type[np.floating[Any]], classes: int
):
    student, teacher = offset_rows(dtype, classes)
    expected_value, expected_gradient = evaluate(Loss("distillation"), student, teacher=teacher)
    eps = float(np.finfo(dtype).eps)
    x = torch.tensor(student, device=device, requires_grad=True)
    t = torch.tensor(teacher, device=device)
    value = distillation_loss(x, t, reduction="none", backend=backend)
    (gradient,) = torch.autograd.grad(value.sum(), x)
    np.testing.assert_allclose(value.detach().cpu(), expected_value, rtol=64 * eps, atol=0)
    np.testing.assert_allclose(gradient.cpu(), expected_gradient, rtol=64 * eps, atol=0)
