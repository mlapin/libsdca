"""MulticlassHingeLoss() with its defaults gives libsdca 0.3.0's values and gradients.

The expected numbers were printed by the PyPI wheel of 0.3.0 (gamma 0, mean over rows,
classes on dim -1): a zero-loss row, a label tied with the largest competitor (margin 0),
competitors 2**-10 apart (the larger one takes the step) and a hinge of 0.25. Rows with
tied competitors or a hinge of exactly 0 are left out: their subgradient is a choice.
"""

import libsdca.torch as L
import pytest
import torch

Z = [
    [0.5, 0.25, -1.0, 2.0],
    [1.0, 0.75, 0.0, 0.25],
    [0.5, 0.5, -1.0, 0.0],
    [0.25, 0.25 + 2**-10, 0.0, -0.5],
    [-0.25, -1.0, 0.5, -2.0],
]
LABELS = [3, 2, 1, 3, 2]
VALUES = [0.0, 2.0, 1.0, 1.7509765625, 0.25]
GRADIENT = [
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, -1.0, 0.0],
    [1.0, -1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0, 0.0],
]


@pytest.mark.parametrize("backend", ["auto", "kernel", "composed"])
@pytest.mark.parametrize(
    "kwargs,scale", [({}, 1 / 5), ({"reduction": "mean"}, 1 / 5), ({"reduction": "sum"}, 1.0)]
)
def test_defaults_match_0_3_0(backend: str, kwargs: dict, scale: float):
    x = torch.tensor(Z, requires_grad=True)
    value = L.MulticlassHingeLoss(backend=backend, **kwargs)(x, torch.tensor(LABELS))
    (gradient,) = torch.autograd.grad(value, x)
    torch.testing.assert_close(value, torch.tensor(sum(VALUES) * scale), rtol=0, atol=0)
    torch.testing.assert_close(gradient, torch.tensor(GRADIENT) * scale, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["auto", "kernel", "composed"])
def test_rows_and_class_axis_match_0_3_0(backend: str):
    x = torch.tensor(Z).t().reshape(1, 4, 5).requires_grad_()
    loss = L.MulticlassHingeLoss(dim=1, reduction="none", backend=backend)
    value = loss(x, torch.tensor(LABELS).reshape(1, 5))
    (gradient,) = torch.autograd.grad(value.sum(), x)
    torch.testing.assert_close(value, torch.tensor([VALUES]), rtol=0, atol=0)
    torch.testing.assert_close(gradient[0].t(), torch.tensor(GRADIENT), rtol=0, atol=0)
