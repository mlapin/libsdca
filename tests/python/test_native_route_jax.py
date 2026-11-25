"""The native route of libsdca.jax: without its FFI handler (SDCA_JAX_FFI=off, or no
compiler) CPU calls run the same kernels in the native library (sdca_loss_rows) through
jax.experimental.buffer_callback (else jax.pure_callback), under jit, grad and vmap. A child without the handler gives the handler's
bytes for every loss, option and dtype of native_route_cases (upstream derivatives other
than 1, and vmap of the logits, the label smoothing and the class weights, included); a
missing compiler warns once, and explain_backend names the native library and why.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import libsdca
import native_route_cases as cases
import numpy as np
import pytest
from child_process import run_python
from libsdca import _jax_native, _jax_ops
from libsdca._frontend import KINDS, Rows

NO_COMPILER = "the CPU kernels of libsdca.jax cannot be built here: no C++ compiler"


def child(directory: Path, outputs: str = "outputs", **env: str) -> dict[str, Any]:
    path = directory / "jax.pkl"
    result = run_python(
        "-m", "native_route_cases", "jax", str(path), outputs,
        env={**os.environ, **env, "JAX_PLATFORMS": "cpu"}, capture_output=True, text=True,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr[-4000:]
    return pickle.loads(path.read_bytes())


@pytest.mark.slow
def test_the_native_route_gives_the_handlers_bytes(tmp_path: Path) -> None:
    library, reason = _jax_ops._handler()
    if library is None:
        pytest.skip(f"no FFI handler to compare with: {reason}")
    native = child(tmp_path, SDCA_JAX_FFI="off")
    with jax.default_device(jax.devices("cpu")[0]):
        want = cases.jax_outputs()
    assert native["outputs"].keys() == want.keys()
    assert not [key for key, bytes_ in want.items() if native["outputs"][key] != bytes_]
    assert native["explain"] == (
        "kernel",
        f"the CPU kernels of the native library{_jax_native.ROUTE} (float32 work; "
        "SDCA_JAX_FFI=off)",
    )
    assert native["warnings"] == []


def test_a_static_label_smoothing_reaches_the_kernels_in_float64() -> None:
    """A static label_smoothing (rows.label_smoothing) wins over the eps array, which holds it
    in the logits' work type: float64 work of float32 logits sees 0.1, not float32(0.1), as
    through the FFI handler. eps is read only for a traced smoothing (rows: NaN)."""
    z = np.array([[-2.0, -1.0, -1000.0, 1.0, 5.0, 3.0, -5.0]], np.float32)
    y = np.array([2])
    none = jnp.zeros((0,), jnp.float32)
    eps = jnp.asarray(0.1, jnp.float32)
    want = libsdca.loss_and_grad(
        z, y, libsdca.CrossEntropy(), precision="float64", label_smoothing=0.1
    )
    for smoothing in (0.1, float("nan")):
        rows = Rows(KINDS["cross_entropy"], label_smoothing=smoothing, precision=2)
        shape = jax.ShapeDtypeStruct(z.shape, z.dtype)
        with jax.default_device(jax.devices("cpu")[0]):
            grad = _jax_native.call(
                shape, jnp.asarray(z), jnp.asarray(y), none, eps, jnp.ones((1,), jnp.float32),
                rows, True,
            )  # fmt: skip
        if smoothing == smoothing:
            assert np.asarray(grad).tobytes() == want[1].tobytes(), (grad, want[1])
        else:  # traced: the eps array's float32(0.1)
            assert np.asarray(grad).tobytes() != want[1].tobytes()


@pytest.mark.parametrize("zero_copy", [True, False], ids=["buffer_callback", "pure_callback"])
def test_both_callbacks_give_the_native_librarys_bytes(
    monkeypatch: pytest.MonkeyPatch, zero_copy: bool
) -> None:
    """The host call of each callback, under jit and vmap, against NumPy's loss_and_grad
    (sdca_loss_evaluate)."""
    if not zero_copy:
        monkeypatch.setattr(_jax_native, "buffer_callback", None)
    elif _jax_native.buffer_callback is None:
        pytest.skip("this JAX has no buffer_callback")
    rng = np.random.default_rng(0)
    z = rng.standard_normal((2, 6, 5)).astype(np.float32)
    y = rng.integers(0, 5, 6)
    rows = Rows(KINDS["cross_entropy"])
    none, eps = jnp.zeros((0,), jnp.float32), jnp.zeros((), jnp.float32)

    def values(s: jax.Array) -> jax.Array:
        shape = jax.ShapeDtypeStruct(s.shape[:-1], jnp.float32)
        return _jax_native.call(shape, s, jnp.asarray(y), none, eps, none, rows, False)

    def gradient(s: jax.Array) -> jax.Array:
        upstream = jnp.ones(s.shape[:-1], jnp.float32)
        shape = jax.ShapeDtypeStruct(s.shape, s.dtype)
        return _jax_native.call(shape, s, jnp.asarray(y), none, eps, upstream, rows, True)

    def fused(s: jax.Array) -> tuple[jax.Array, jax.Array]:
        shape = jax.ShapeDtypeStruct(s.shape[:-1], jnp.float32)
        upstream = jnp.ones(s.shape[:-1], jnp.float32)
        return _jax_native.call(
            shape, s, jnp.asarray(y), none, eps, upstream, rows, True, both=True
        )

    with jax.default_device(jax.devices("cpu")[0]):
        got = [np.asarray(jax.jit(f)(z[0])) for f in (values, gradient)]
        mapped = [np.asarray(jax.jit(jax.vmap(f))(z)) for f in (values, gradient)]
        together = jax.jit(jax.vmap(fused))(z)
    for i, rows_of in enumerate(z):
        want = libsdca.loss_and_grad(rows_of, y, libsdca.CrossEntropy())
        if i == 0:
            assert got[0].tobytes() == want[0].astype(np.float32).tobytes()
            assert got[1].tobytes() == want[1].tobytes()
        assert mapped[0][i].tobytes() == want[0].astype(np.float32).tobytes()
        assert mapped[1][i].tobytes() == want[1].tobytes()
        for combined, separate in zip(together, mapped, strict=True):
            assert np.asarray(combined[i]).tobytes() == separate[i].tobytes()


def test_without_a_compiler_the_native_route_warns_once(tmp_path: Path) -> None:
    found = child(tmp_path, "none", CXX="/nonexistent/c++", SDCA_JAX_FFI="auto")
    (message,) = found["warnings"]
    assert message.startswith(f"libsdca: {NO_COMPILER}")
    assert "native library (libsdca_c)" in message and "SDCA_JAX_FFI=off" in message
    backend, reason = found["explain"]
    assert backend == "kernel"
    assert reason.startswith(
        f"the CPU kernels of the native library{_jax_native.ROUTE} (float32 work; {NO_COMPILER}"
    )
