"""JAX fused, callback and array top-1 hinges preserve score order."""

from typing import Any

import jax
import jax.numpy as jnp
import named_losses
import numpy as np
import pytest
from hinge_order_cases import rows
from libsdca import jax as J
from libsdca._jax_fusion import _precision_scope


@pytest.mark.parametrize("classes", [3, 17, 1000])
@pytest.mark.parametrize("power", [-29, -60])
@pytest.mark.parametrize(
    "dtype,precision",
    [
        (dtype, precision)
        for dtype in [jnp.float32, jnp.float64, jnp.bfloat16]
        for precision in (None, "float32", "float64")
        if not (dtype == jnp.float64 and precision == "float32")
    ],
)
@pytest.mark.parametrize("backend", ["auto", "native", "jax"])
def test_distinct_hinge_scores(
    classes: int, power: int, dtype: Any, precision: str | None, backend: str
):
    z, labels, expected = rows(classes, power)
    with _precision_scope(True):
        x, y = jnp.asarray(z, dtype=dtype), jnp.asarray(labels)
        for loss in ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta"):

            def f(v: jax.Array):
                return named_losses.loss(J, loss, v, y, precision=precision, backend=backend)

            gradient = jax.jit(jax.grad(f))(x)
            np.testing.assert_array_equal(gradient, expected / len(labels))
    jax.clear_caches()
