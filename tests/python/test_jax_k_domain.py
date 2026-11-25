"""The JAX top-k losses check 1 <= k < classes on every backend; k is a required keyword
of the top-k losses only (docs/API.md, parameter domains)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from libsdca import jax as sdca_jax

jax.config.update("jax_enable_x64", True)

CLASSES = 5
OUT_OF_RANGE = [CLASSES, 2 * CLASSES]
BACKENDS = ["auto", "kernel", "composed"]
TOPK = {
    "topk_cross_entropy": sdca_jax.topk_cross_entropy,
    "topk_hinge_alpha": lambda *a, **o: sdca_jax.topk_hinge_loss(*a, variant="alpha", **o),
    "topk_hinge_beta": sdca_jax.topk_hinge_loss,
}


def problem() -> tuple[jax.Array, jax.Array]:
    z = jnp.asarray(np.random.default_rng(2).normal(size=(6, CLASSES)))
    return z, jnp.asarray(np.arange(6) % CLASSES)


def test_topk_losses_check_k(subtests: pytest.Subtests):
    z, target = problem()
    for loss, function in TOPK.items():
        for backend in BACKENDS:
            for k in OUT_OF_RANGE:
                with subtests.test(loss=loss, backend=backend, k=k):
                    with pytest.raises(ValueError, match="k must be in"):
                        function(z, target, k=k, backend=backend)
            for k in (0, -1):
                with subtests.test(loss=loss, backend=backend, k=k):
                    with pytest.raises(ValueError, match="k must be at least 1"):
                        function(z, target, k=k, backend=backend)
            for k in (1, CLASSES - 1):
                with subtests.test(loss=loss, backend=backend, k=k):
                    assert bool(jnp.isfinite(function(z, target, k=k, backend=backend)))
        for k in (2.0, True, "2", None):
            with subtests.test(loss=loss, k=k):
                with pytest.raises(TypeError, match="k must be an integer"):
                    function(z, target, k=k)
        with subtests.test(loss=loss, k="missing"):
            with pytest.raises(TypeError, match="k"):
                function(z, target)
        with subtests.test(loss=loss, k="positional"):
            with pytest.raises(TypeError):
                function(z, target, 2)


def test_k_is_static():
    z, target = problem()
    with pytest.raises(TypeError, match="k must be a static number"):
        jax.jit(lambda k: sdca_jax.topk_cross_entropy(z, target, k=k))(2)
    # A concrete 0-d array is a number.
    np.testing.assert_array_equal(
        sdca_jax.topk_hinge_loss(z, target, k=jnp.asarray(2)),  # pyright: ignore[reportArgumentType]
        sdca_jax.topk_hinge_loss(z, target, k=2),
    )


def test_losses_without_k_reject_it(subtests: pytest.Subtests):
    z, target = problem()
    mask = jnp.asarray(np.eye(CLASSES, dtype=bool)[np.arange(6) % CLASSES])
    for function, t in (
        (sdca_jax.cross_entropy, target),
        (sdca_jax.multiclass_hinge_loss, target),
        (sdca_jax.proximal_cross_entropy, target),
        (sdca_jax.multilabel_cross_entropy, mask),
        (sdca_jax.multilabel_hinge_loss, mask),
    ):
        with subtests.test(loss=function.__name__):
            with pytest.raises(TypeError, match="k"):
                function(z, t, k=2)  # pyright: ignore[reportCallIssue]
