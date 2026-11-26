"""The CPU extension of libsdca.jax loads once per process: later calls, eager or traced, find
it without hashing the sources, probing the compiler or registering the target again."""

from __future__ import annotations

from unittest import mock

import jax
import jax.numpy as jnp
import libsdca.jax as J
import pytest
from libsdca import _extensions, _jax_ops

if _jax_ops.cpu_kernels(0) is not None:
    pytest.skip(f"no FFI handlers: {_jax_ops.cpu_kernels(0)}", allow_module_level=True)

SCORES = jax.random.normal(jax.random.PRNGKey(0), (8, 5))
LABELS = jnp.arange(8) % 5


def test_calls_do_not_load_the_extension_again():
    J.cross_entropy(SCORES, LABELS)
    loads = _extensions.load.cache_info().currsize
    with (
        mock.patch.object(_extensions, "missing_compiler", side_effect=AssertionError),
        mock.patch.object(_extensions, "directory", side_effect=AssertionError),
    ):
        J.cross_entropy(SCORES, LABELS)
        jax.value_and_grad(J.cross_entropy)(SCORES, LABELS)
        J.topk_hinge_loss(SCORES, LABELS, k=2)
    assert _extensions.load.cache_info().currsize == loads
