"""Conformance of the PyTorch losses on every CPU and CUDA backend to the shared mpmath
references (tests/python/conformance); the CUDA cells skip without CUDA."""

import pytest
import torch  # noqa: F401  # the torch group
from conformance.backends import Backend
from conformance.cells import Cell, check_cell, check_nonfinite, nonfinite_params, params
from conformance.references import Loss


@pytest.mark.parametrize("cell", params("torch"))
def test_conformance(cell: Cell, request: pytest.FixtureRequest):
    check_cell(cell, request)


@pytest.mark.parametrize("backend,loss", nonfinite_params("torch"))
def test_nonfinite_logits(backend: Backend, loss: Loss):
    check_nonfinite(backend, loss)
