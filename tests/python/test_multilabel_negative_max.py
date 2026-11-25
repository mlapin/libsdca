"""A negative maximum bounds the error of direct positive residuals by the row norm."""

import numpy as np
import pytest
import torch
from conformance.references import Loss, evaluate
from libsdca.torch import multilabel_cross_entropy


def exact(z: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
    values, gradients = evaluate(Loss("multilabel_cross_entropy"), z[None], mask[None])
    return float(values[0]), gradients[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("classes", [3, 17, 513, 4097])
@pytest.mark.parametrize("scale", [0.0001, 0.1, 1.0, 3.0, 20.0])
def test_negative_maximum(classes: int, scale: float):
    rng = np.random.default_rng(7193 + classes)
    z = (rng.normal(size=(4, classes)) * scale).astype(np.float32)
    mask = rng.random(z.shape) < 0.25
    for row in range(4):
        mask[row, np.argmax(z[row])] = False
        mask[row, np.argmin(z[row])] = True
    x = torch.tensor(z, device="cuda", requires_grad=True)
    y = torch.tensor(mask, device="cuda")
    value = multilabel_cross_entropy(x, y, reduction="none")
    (gradient,) = torch.autograd.grad(value.sum(), x)
    actual_value = value.detach().cpu().numpy()
    actual_gradient = gradient.cpu().numpy()
    eps = np.finfo(np.float32).eps
    for row in range(4):
        loss, expected = exact(z[row], mask[row])
        assert abs(float(actual_value[row]) - loss) <= 8 * eps * abs(loss)
        assert np.max(np.abs(actual_gradient[row] - expected)) <= 8 * eps * np.max(np.abs(expected))
