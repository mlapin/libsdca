"""float64 cross-entropy reductions beyond double's range: a mean of finite row losses (or of
finite class weights) whose sums overflow stays finite, in the kernels of PyTorch and JAX with
the same bytes; a sum that overflows is +inf (mpmath references)."""

import math
from typing import Literal

import libsdca.torch as lt
import mpmath as mp
import numpy as np
import pytest
import torch
from conformance.references import Loss, rows
from conformance.references import reference as shared_reference


def reference(z: np.ndarray, y: np.ndarray, w: np.ndarray | None, reduction: str) -> float:
    """The weighted mean or sum of the row losses (conformance.references), in mpmath."""
    with mp.workdps(60):
        losses, weights = [], []
        for (row, _), label in zip(rows(z, y), y, strict=True):
            weight = mp.mpf(1) if w is None else mp.mpf(float(w[label]))
            losses.append(weight * shared_reference(Loss("cross_entropy"), row).value)
            weights.append(weight)
        total = mp.fsum(losses)
        value = total / mp.fsum(weights) if reduction == "mean" else total
        return float(value) if abs(value) < mp.mpf(np.finfo(np.float64).max) else math.inf


CASES = {
    "losses near 1e308": (np.array([[0.0, 1e308], [0.0, 1.5e308], [1.0, 2.0]]), [0, 0, 1], None),
    "weights near 1e308": (
        np.array([[0.3, -1.0], [2.0, 1.5], [-0.7, 0.1]]),
        [0, 1, 0],
        [1e308, 9e307],  # weighted rows within double's range, their total beyond it
    ),
}


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_reductions_beyond_double(name: str, reduction: Literal["mean", "sum"]):
    z, y, w = CASES[name]
    labels = np.array(y)
    weights = None if w is None else np.array(w)
    expected = reference(z, labels, weights, reduction)
    tw = None if weights is None else torch.tensor(weights)
    value = lt.cross_entropy(torch.tensor(z), torch.tensor(labels), tw, reduction=reduction)
    if math.isinf(expected):
        assert value.item() == math.inf
    else:
        assert math.isclose(value.item(), expected, rel_tol=8 * np.finfo(np.float64).eps)
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)  # as test_masked_classes_jax
    import jax.numpy as jnp
    import libsdca.jax as lj

    jw = None if weights is None else jnp.asarray(weights)
    got = lj.cross_entropy(jnp.asarray(z), jnp.asarray(labels), weight=jw, reduction=reduction)
    assert np.asarray(got).tobytes() == value.numpy().tobytes()


@pytest.mark.parametrize("name", list(CASES))
def test_transformed_means_beyond_double(name: str):
    """torch.func, torch.compile and vmap (class weights cannot be mapped) take the
    extension's mean."""
    z, y, w = CASES[name]
    scores, labels = torch.tensor(z), torch.tensor(y)
    weight = None if w is None else torch.tensor(np.array(w))
    loss = lt.cross_entropy(scores, labels, weight)
    assert math.isfinite(loss.item())
    expected = loss.numpy().tobytes()

    def mean(s: torch.Tensor) -> torch.Tensor:
        return lt.cross_entropy(s, labels, weight)

    _, value = torch.func.grad_and_value(mean)(scores)
    compiled = torch.compile(mean, backend="aot_eager", fullgraph=True)(scores)
    assert value.numpy().tobytes() == expected
    assert compiled.numpy().tobytes() == expected
    if weight is None:
        mapped = torch.func.vmap(mean)(torch.stack([scores, scores]))
        assert mapped.numpy().tobytes() == expected * 2
