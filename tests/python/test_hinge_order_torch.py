"""Fused and tensor PyTorch top-1 hinges preserve the ordering of represented scores."""

import pytest
import torch
from hinge_order_cases import rows
from named_losses import torch_loss


@pytest.mark.parametrize("classes", [3, 17, 1000])
@pytest.mark.parametrize("power", [-29, -60])
@pytest.mark.parametrize(
    "dtype,precision",
    [
        (dtype, precision)
        for dtype in [torch.float32, torch.float64, torch.bfloat16]
        for precision in (None, "float32", "float64")
        if not (dtype == torch.float64 and precision == "float32")
    ],
)
@pytest.mark.parametrize("backend", ["auto", "native", "torch"])
def test_distinct_hinge_scores(
    classes: int, power: int, dtype: torch.dtype, precision: str | None, backend: str
):
    z, labels, expected = rows(classes, power)
    x = torch.tensor(z, dtype=dtype, requires_grad=True)
    for loss in ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta"):
        value = torch_loss(x, torch.tensor(labels), loss, precision=precision, backend=backend)
        gradient = torch.autograd.grad(value, x)[0]
        torch.testing.assert_close(
            gradient, torch.tensor(expected / len(labels), dtype=dtype), rtol=0, atol=0
        )
