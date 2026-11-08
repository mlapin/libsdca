"""Represented nonintegral score gaps on the JAX mask-entropy paths."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from libsdca.jax import multilabel_cross_entropy

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("precision", [None, "float64"])
@pytest.mark.parametrize("classes", [4, 1000])
@pytest.mark.parametrize("tied", [False, True])
def test_mask_score_gap_rounding(precision: str | None, classes: int, tied: bool):
    base = np.array(
        [
            [40.1234567, 40.1234567, -29.3456789, -29.7654321],
            [20.1234567, 20.1234567, -20.3456789, -20.7654321],
            [50.1234567, 50.1234567, -30.3456789, -30.7654321],
        ],
        dtype=np.float32,
    )
    z = np.tile(base[:, 2:], (1, classes // 2))
    z[:, :2] = base[:, :2]
    if not tied:
        z[:, 0] = np.nextafter(z[:, 0], np.float32(-np.inf))
    mask = np.zeros(z.shape, dtype=bool)
    mask[:, :2] = True
    expected = []
    with mp.workdps(80):
        for row in z.tolist():
            terms = [mp.exp(mp.mpf(x) - row[0]) for x in row]
            denominator = sum(terms)
            expected.append(
                [
                    float(v / denominator - (mp.mpf("0.5") if j < 2 else 0))
                    for j, v in enumerate(terms)
                ]
            )
    reference = np.asarray(expected)
    fn = jax.jit(
        jax.grad(lambda x, m: multilabel_cross_entropy(x, m, precision=precision, reduction="sum"))
    )
    actual = np.asarray(fn(jnp.asarray(z), jnp.asarray(mask)), dtype=np.float64)
    error = np.max(np.abs(actual - reference) / np.max(np.abs(reference), axis=1, keepdims=True))
    budget = 64 if precision is None else 1
    assert error <= budget * np.finfo(np.float32).eps
