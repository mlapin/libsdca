"""Boolean-mask cross-entropy of well-fit rows: two positives at 5, C - 2 negatives at -20.

The C - 2 equal small terms exp(-25) beside the other positive's 1 lost up to 772 float64
eps of the loss in plain sums (native rows, before 0.3.0). Every path (the CPU extension,
the native rows of PyTorch and NumPy, the C API sdca_loss_gradient) stays within 64 output
eps of the exact values, the positives first or last."""

import os
from unittest import mock

import mpmath as mp
import numpy as np
import pytest
import torch
from libsdca import _cpu_entropy, loss_and_grad
from libsdca.torch import multilabel_cross_entropy
from test_native_tensor_losses import loss_gradient

EPS = 64


def exact(classes: int, dtype: torch.dtype) -> tuple[float, float, float]:
    """Loss, positive and negative gradient entries of the represented row."""
    with mp.workdps(50):
        e = mp.exp(mp.mpf(torch.tensor(-20.0, dtype=dtype).item()) - 5)
        den = 2 + (classes - 2) * e
        return mp.log(den), -(classes - 2) * e / (2 * den), e / den


def row(classes: int, last: bool, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.full((1, classes), -20.0, dtype=dtype)
    second = classes - 1 if last else 1
    z[0, [0, second]] = 5.0
    mask = torch.zeros(1, classes, dtype=torch.bool)
    mask[0, [0, second]] = True
    return z, mask


def check(value: float, grad: np.ndarray, mask: np.ndarray, dtype: torch.dtype, what: str):
    eps = torch.finfo(dtype).eps
    loss, positive, negative = exact(grad.shape[-1], dtype)
    assert abs(value - loss) <= EPS * eps * loss, f"{what}: loss {value} vs {loss}"
    g = grad.reshape(-1)
    m = mask.reshape(-1)
    for j in (np.flatnonzero(m)[0], np.flatnonzero(m)[-1], np.flatnonzero(~m)[0]):
        expected = positive if m[j] else negative
        assert abs(g[j] - expected) <= EPS * eps * abs(expected), (
            f"{what}: entry {j} {g[j]} vs {expected}"
        )


@pytest.mark.parametrize("classes", [1000, 4097])
@pytest.mark.parametrize("last", [False, True], ids=["second", "last"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
class WellFitMultilabelTests:
    def test_torch_paths(self, classes: int, last: bool, dtype: torch.dtype):
        paths = ["off", *(["on"] if _cpu_entropy.buildable() else [])]
        for flag in paths:
            for precision in (None, "float64"):
                z, mask = row(classes, last, dtype)
                x = z.requires_grad_()
                with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": flag}):
                    value = multilabel_cross_entropy(x, mask, reduction="sum", precision=precision)
                    (grad,) = torch.autograd.grad(value, x)
                what = f"SDCA_CPU_EXTENSION={flag} precision={precision}"
                check(value.item(), grad.double().numpy(), mask.numpy(), dtype, what)

    def test_numpy_and_c_api(self, classes: int, last: bool, dtype: torch.dtype):
        z, mask = row(classes, last, dtype)
        for precision in ("float32", "float64") if dtype == torch.float32 else ("float64",):
            value, grad = loss_and_grad(
                z.numpy(), mask.numpy(), "multilabel_cross_entropy", precision=precision
            )
            check(float(np.sum(value)), grad, mask.numpy(), dtype, f"loss_and_grad {precision}")
            values, gradients = loss_gradient(
                z.numpy(), mask.numpy(), "multilabel_cross_entropy", precision
            )
            check(float(values[0]), gradients, mask.numpy(), dtype, f"C API {precision}")
