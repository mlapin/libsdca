"""Native top-1 hinges select scores before rounding their margins."""

from typing import Any

import numpy as np
import pytest
from hinge_order_cases import rows
from numpy_api import loss_and_grad

SETTINGS = [
    (dtype, precision)
    for dtype in [np.float32, np.float64]
    for precision in (None, "float32", "float64")
    if not (dtype == np.float64 and precision == "float32")
]


def params() -> list[Any]:
    out = []
    for power in (-29, -60):
        for dtype, precision in SETTINGS:
            key = f"{np.dtype(dtype).name}-{precision}-{power}"
            out.append(pytest.param(dtype, precision, power, id=key))
    return out


@pytest.mark.parametrize("classes", [3, 17, 1000])
@pytest.mark.parametrize("dtype,precision,power", params())
def test_distinct_hinge_scores(
    classes: int, power: int, dtype: type[np.float32] | type[np.float64], precision: str | None
):
    z, labels, expected = rows(classes, power)
    for loss in ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta"):
        _, gradient = loss_and_grad(z.astype(dtype), labels, loss, precision=precision)
        np.testing.assert_array_equal(gradient, expected)


@pytest.mark.parametrize("classes", [10, 4096])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multilabel_hinge_splits_only_the_extremal_classes(
    classes: int, dtype: type[np.float32] | type[np.float64]
) -> None:
    # Tied minima and maxima at zero margin, then a unique winner beside a higher positive.
    z = np.full((2, classes), -4, dtype=dtype)
    mask = np.zeros_like(z, dtype=bool)
    mask[:, :3] = True
    z[0, :5] = [1, 1, 9, 0, 0]
    z[1, :5] = [-2, 2, 9, 0, 1]
    values, gradient = loss_and_grad(z, mask, "multilabel_hinge")
    expected = np.zeros_like(z)
    expected[0, :5] = [-0.25, -0.25, 0, 0.25, 0.25]
    expected[1, [0, 4]] = [-1, 1]
    np.testing.assert_array_equal(values, [0, 4])
    np.testing.assert_array_equal(gradient, expected)
