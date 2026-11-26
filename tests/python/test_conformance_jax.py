"""Conformance of the JAX losses on every CPU and GPU backend to the shared mpmath references
(tests/python/conformance); the GPU cells skip without a GPU."""

import jax
import pytest
from conformance.backends import Backend
from conformance.cells import Cell, check_cell, check_nonfinite, nonfinite_params, params
from conformance.references import Loss

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("cell", params("jax"))
def test_conformance(cell: Cell, request: pytest.FixtureRequest):
    check_cell(cell, request)


@pytest.mark.parametrize("backend,loss", nonfinite_params("jax"))
def test_nonfinite_logits(backend: Backend, loss: Loss):
    check_nonfinite(backend, loss)
