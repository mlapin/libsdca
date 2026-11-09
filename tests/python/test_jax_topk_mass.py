"""XLA retains a normal target mass independently of subnormal competitors."""

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from libsdca.jax import topk_cross_entropy
from test_jax import x64


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("classes", [3, 65])
def test_topk_normal_mass_above_flushed_competitors(dtype: Any, classes: int):
    with x64(True):
        info = np.finfo(dtype)
        gap = math.log(classes - 1) - math.log(float(info.tiny) * 1.5)
        z = np.zeros((1, classes), dtype=dtype)
        z[0, 0] = gap
        x, y = jnp.asarray(z), jnp.array([0], dtype=jnp.int32)

        def f(x: jax.Array):
            return topk_cross_entropy(x, y, k=2, backend="jax")

        value, gradient = jax.value_and_grad(f)(x)
        tail = math.exp(-float(z[0, 0]) + math.log(classes - 1))
        tolerance = 1e-12 if dtype == np.float64 else 1e-5
        assert float(value) > 0
        assert float(gradient[0, 0]) < 0
        assert abs(float(value) / tail - 1) < tolerance
        assert abs(float(gradient[0, 0]) / -tail - 1) < tolerance
