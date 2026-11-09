"""Every JAX entry point checks 1 <= k < classes for every loss and backend, also the
losses that do not use k (docs/API.md, parameter domains)."""

import jax
import jax.numpy as jnp
import libsdca
import numpy as np
import pytest
from libsdca import jax as sdca_jax

jax.config.update("jax_enable_x64", True)

CLASSES = 5
OUT_OF_RANGE = [0, CLASSES, 2 * CLASSES]
VALID = [1, CLASSES - 1]
BACKENDS = ["auto", "native", "jax"]
FUNCTIONS = {
    "topk_cross_entropy": sdca_jax.topk_cross_entropy,
    "topk_hinge_alpha": sdca_jax.topk_hinge_alpha_loss,
    "topk_hinge_beta": sdca_jax.topk_hinge_beta_loss,
}


def problem(loss: str) -> tuple[jax.Array, jax.Array]:
    z = jnp.asarray(np.random.default_rng(2).normal(size=(6, CLASSES)))
    y = np.arange(6) % CLASSES
    target = np.eye(CLASSES, dtype=bool)[y] if "multilabel" in loss else y
    return z, jnp.asarray(target)


def test_sdca_loss(subtests: pytest.Subtests):
    for loss in libsdca.LOSSES:
        z, target = problem(loss)
        for backend in BACKENDS:
            for k in OUT_OF_RANGE:
                with subtests.test(loss=loss, backend=backend, k=k):
                    with pytest.raises(ValueError, match="k must be"):
                        sdca_jax.sdca_loss(z, target, loss, backend=backend, k=k)
            for k in VALID:
                with subtests.test(loss=loss, backend=backend, k=k):
                    value = sdca_jax.sdca_loss(z, target, loss, backend=backend, k=k)
                    assert bool(jnp.isfinite(value))


def test_functions_that_take_k(subtests: pytest.Subtests):
    for loss, function in FUNCTIONS.items():
        z, target = problem(loss)
        for backend in BACKENDS:
            for k in OUT_OF_RANGE:
                with subtests.test(loss=loss, backend=backend, k=k):
                    with pytest.raises(ValueError, match="k must be"):
                        function(z, target, k, backend=backend)
            with subtests.test(loss=loss, backend=backend, k=CLASSES - 1):
                expected = sdca_jax.sdca_loss(z, target, loss, backend=backend, k=CLASSES - 1)
                np.testing.assert_array_equal(
                    function(z, target, CLASSES - 1, backend=backend), expected
                )
    z, target = problem("topk_hinge_beta")
    for k in OUT_OF_RANGE:
        with subtests.test(loss="topk_hinge_loss", k=k):
            with pytest.raises(ValueError, match="k must be"):
                sdca_jax.topk_hinge_loss(z, target, k, variant="alpha")


def test_multiclass_hinge_ignores_a_valid_k(subtests: pytest.Subtests):
    z, target = problem("multiclass_hinge")
    for backend in BACKENDS:
        for gamma in (0.0, 0.5):
            with subtests.test(backend=backend, gamma=gamma):
                options = {"backend": backend, "gamma": gamma}
                expected = sdca_jax.sdca_loss(z, target, "multiclass_hinge", **options)
                value = sdca_jax.sdca_loss(z, target, "multiclass_hinge", k=CLASSES - 1, **options)
                np.testing.assert_array_equal(value, expected)
