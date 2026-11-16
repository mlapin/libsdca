"""Conformance of the native NumPy losses (loss_and_grad, distillation_loss_and_grad) to the
shared mpmath references (tests/python/conformance)."""

import pytest
from conformance.backends import Backend
from conformance.cells import Cell, check_cell, check_nonfinite, nonfinite_params, params
from conformance.references import Loss


@pytest.mark.parametrize("cell", params("numpy"))
def test_conformance(cell: Cell, request: pytest.FixtureRequest):
    check_cell(cell, request)


@pytest.mark.parametrize("backend,loss", nonfinite_params("numpy"))
def test_nonfinite_logits(backend: Backend, loss: Loss):
    check_nonfinite(backend, loss)
