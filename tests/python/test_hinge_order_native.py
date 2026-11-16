"""Native top-1 hinges select scores before rounding their margins."""

import numpy as np
import pytest
from hinge_order_cases import rows
from libsdca import loss_and_grad


@pytest.mark.parametrize("classes", [3, 17, 1000])
@pytest.mark.parametrize("power", [-29, -60])
@pytest.mark.parametrize(
    "dtype,precision",
    [
        (dtype, precision)
        for dtype in [np.float32, np.float64]
        for precision in (None, "float32", "float64")
        if not (dtype == np.float64 and precision == "float32")
    ],
)
def test_distinct_hinge_scores(
    classes: int, power: int, dtype: type[np.float32] | type[np.float64], precision: str | None
):
    z, labels, expected = rows(classes, power)
    for loss in ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta"):
        _, gradient = loss_and_grad(z.astype(dtype), labels, loss, precision=precision)
        np.testing.assert_array_equal(gradient, expected)
