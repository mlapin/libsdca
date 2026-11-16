"""Conformance of the JAX losses on CPU (jax.numpy, the FFI handlers and the host callback)
and GPU to the shared mpmath references (tests/python/conformance)."""

import gc
import itertools
from dataclasses import replace
from typing import Any

import jax
import pytest
from conformance import backends
from conformance.backends import Backend
from conformance.cells import Cell, cells, check_cell, check_nonfinite, nonfinite_params, params
from conformance.references import Loss
from libsdca import _jax_cpu, _jax_cuda
from libsdca._jax_fusion import _precision_scope

_COMPLETED = itertools.count(1)


@pytest.fixture(autouse=True)
def x64_scope():
    # Do not change process-wide JAX defaults during collection of unrelated tests.
    with _precision_scope(True):
        yield
    # Thousands of distinct shape/precision cells otherwise retain compiled
    # executables for the entire worker lifetime. This is test-harness state.
    if next(_COMPLETED) % 64 == 0:
        jax.clear_caches()
        gc.collect()


_DEFAULT_CELLS = [
    replace(
        cell,
        backend=replace(
            cell.backend, name="jax-cpu-default", option="auto", env=(("SDCA_JAX_FFI", "on"),)
        ),
    )
    for cell in cells("jax")
    if cell.backend.name == "jax-cpu-jax"
    and cell.dtype == "float32"
    and cell.precision is None
    and cell.case.family == "ordinary"
]


@pytest.mark.parametrize("cell", _DEFAULT_CELLS, ids=lambda cell: cell.id)
def test_default_without_x64(cell: Cell, request: pytest.FixtureRequest):
    with _precision_scope(False):
        assert not jax.config.x64_enabled
        check_cell(cell, request)


@pytest.mark.parametrize("cell", params("jax"))
def test_conformance(cell: Cell, request: pytest.FixtureRequest):
    check_cell(cell, request)


@pytest.mark.parametrize("backend,loss", nonfinite_params("jax"))
def test_nonfinite_logits(backend: Backend, loss: Loss):
    check_nonfinite(backend, loss)


@pytest.mark.parametrize("backend,loss", nonfinite_params("jax"))
def test_nonfinite_logits_of_concrete_calls(backend: Backend, loss: Loss):
    check_nonfinite(backend, loss, traced=False)


def test_gpu_triton_backend_reaches_shared_kernels(monkeypatch: pytest.MonkeyPatch):
    """A backend label must exercise its implementation, including generic losses."""
    cell = next(
        cell
        for cell in cells("jax")
        if cell.backend.name == "jax-gpu-triton"
        and cell.loss == Loss("cross_entropy")
        and cell.dtype == "float32"
        and cell.precision is None
        and cell.case.family == "ordinary"
        and cell.case.classes == 10
    )
    reason = backends.unavailable(cell.backend)
    if reason:
        pytest.skip(reason)
    accepted: list[tuple[str, bool]] = []
    ready = _jax_cuda._ready

    def observed(spec: _jax_cuda.Spec, x: jax.Array) -> bool:
        result = ready(spec, x)
        accepted.append((spec.family, result))
        return result

    monkeypatch.setattr(_jax_cuda, "_ready", observed)
    backends.run(cell.backend, cell.loss, cell.dtype, cell.precision, cell.reduction, cell.case)
    assert ("entropy", True) in accepted, accepted


def test_cpu_auto_distillation_reaches_shared_kernels(monkeypatch: pytest.MonkeyPatch):
    cell = next(
        cell
        for cell in cells("jax")
        if cell.backend.name == "jax-cpu-auto"
        and cell.loss == Loss("distillation")
        and cell.dtype == "float32"
        and cell.precision is None
        and cell.case.family == "ordinary"
        and cell.case.classes == 10
    )
    reason = backends.unavailable(cell.backend)
    if reason:
        pytest.skip(reason)
    accepted: list[bool] = []
    distillation = _jax_cpu.distillation

    def observed(*args: Any, **kwargs: Any) -> Any:
        result = distillation(*args, **kwargs)
        accepted.append(result is not None)
        return result

    monkeypatch.setattr(_jax_cpu, "distillation", observed)
    backends.run(cell.backend, cell.loss, cell.dtype, cell.precision, cell.reduction, cell.case)
    assert True in accepted, accepted
