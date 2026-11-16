"""The XLA FFI calls of the native JAX losses (_jax_ffi) against the host callback they replace:
the same bits, dtypes and messages under jit, grad and vmap, and the build rules."""

import os
import subprocess
import textwrap
import types
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from child_process import run_python
from jax.typing import ArrayLike
from libsdca import _extensions, _jax_ffi
from libsdca._core import _LABEL_LOSSES as LOSSES
from named_losses import jax_loss as named_loss

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(
    _jax_ffi.missing_tools() is not None, reason=f"no FFI build: {_jax_ffi.missing_tools()}"
)
# Traced calls take the FFI only where the default backend is the CPU.
cpu_backend = pytest.mark.skipif(jax.default_backend() != "cpu", reason="default backend not CPU")


@contextmanager
def ffi(setting: str) -> Iterator[None]:
    with mock.patch.dict(os.environ, {_jax_ffi.ENV: setting}):
        yield


def x64(enabled: bool) -> Any:
    """A thread-local x64 scope: jax.enable_x64 in newer releases, else jax.experimental's."""
    import jax.experimental

    scope = getattr(jax, "enable_x64", None)
    if scope is None:
        scope = getattr(jax.experimental, "enable_x64", None)
    assert scope is not None
    return scope(enabled)


def native(x: ArrayLike, t: ArrayLike, loss: str, reduction: str = "none") -> jax.Array:
    return named_loss(x, t, loss, backend="native", reduction=reduction, k=2, gamma=0.6, alpha=0.9)


def problem(dtype: Any, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores (with a confident row whose float32 entropy is subnormal), labels and a mask with
    a positive and a negative label per row."""
    rng = np.random.default_rng(seed)
    z = (3 * rng.normal(size=(6, 7))).astype(dtype)
    z[0] = [100, 0, 0, 0, 0, 0, 0]
    labels = rng.integers(0, 7, 6)
    labels[0] = 0
    mask = rng.random((6, 7)) < 0.4
    mask[np.arange(6), labels] = True
    mask[np.arange(6), (labels + 1) % 7] = False
    return z, labels, mask


def both(f: Callable[[], Any]) -> tuple[Any, Any]:
    """f() through the FFI handlers and through the host callback, each traced afresh."""
    jax.clear_caches()
    with ffi("on"):
        handlers = jax.tree.map(np.asarray, f())
    jax.clear_caches()
    with ffi("off"):
        callback = jax.tree.map(np.asarray, f())
    return handlers, callback


def assert_same_bits(actual: Any, expected: Any, message: str = "") -> None:
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert a.dtype == e.dtype, message
        np.testing.assert_array_equal(a.view(f"u{a.itemsize}"), e.view(f"u{e.itemsize}"), message)


class JaxFfiTests:
    def test_cpu_isa_invalidates_only_cpu_builds(self):
        """A shared cache cannot reuse host-native instructions on another CPU ISA."""
        cuda = replace(_jax_ffi.CPU, source=_jax_ffi.SOURCE.with_name("_jax_cuda.cpp"))
        with mock.patch.object(_jax_ffi, "_cpu_target", return_value=(("-march=native",), "avx2")):
            first_cpu = _jax_ffi.build_directory()
            first_cuda = _jax_ffi.build_directory(cuda)
        with mock.patch.object(
            _jax_ffi, "_cpu_target", return_value=(("-march=native",), "avx512")
        ):
            assert first_cpu != _jax_ffi.build_directory()
            assert first_cuda == _jax_ffi.build_directory(cuda)

    def test_cpu_target_probe_failures_give_the_portable_build(self):
        failures = (
            OSError("no compiler"),
            subprocess.CalledProcessError(1, "c++"),
            subprocess.TimeoutExpired("c++", 30),
        )
        for failure in failures:
            _jax_ffi._cpu_target.cache_clear()
            with mock.patch.object(_jax_ffi.subprocess, "run", side_effect=failure):
                assert _jax_ffi._cpu_target("c++", "x86_64") == ((), "")
        _jax_ffi._cpu_target.cache_clear()

    @cpu_backend
    def test_lowers_to_the_ffi_targets_on_the_cpu(self):
        z, labels, _ = problem(np.float64)

        def loss(x: jax.Array) -> jax.Array:
            return native(x, labels, "topk_hinge_beta", "mean")

        jax.clear_caches()
        with ffi("on"):
            assert "@libsdca_loss_gradient(" in jax.jit(jax.grad(loss)).lower(z).as_text()
            assert "@libsdca_loss(" in jax.jit(loss).lower(z).as_text()
        jax.clear_caches()
        with ffi("off"):
            text = jax.jit(jax.grad(loss)).lower(z).as_text()
            assert "@libsdca_loss" not in text and "callback" in text

    @cpu_backend
    def test_export_rejects_the_ffi_call(self):
        from jax import export

        z, labels, _ = problem(np.float64)
        jax.clear_caches()
        f = jax.jit(lambda x: native(x, labels, "topk_hinge_beta", "mean"))
        with ffi("on"), pytest.raises(ValueError, match="custom call"):
            export.export(f)(jax.ShapeDtypeStruct(z.shape, z.dtype))

    @pytest.mark.long
    def test_every_loss_matches_the_host_callback_bitwise(self, subtests: pytest.Subtests):
        """Losses and gradients of every loss, labels of several integer types or masks,
        float32 and float64 scores, with and without x64 (float32 losses rounded from
        double), forward alone and with the gradient, jit and vmap."""
        for enabled in (True, False):
            for dtype in (np.float32, np.float64) if enabled else (np.float32,):
                z, labels, mask = problem(dtype)
                weights = np.linspace(0.5, 2.0, len(z)).astype(dtype)
                for loss in LOSSES:
                    integers = [labels.astype(np.int32 if enabled else np.int64), labels]
                    integers.append(labels.astype(np.uint8))
                    targets = [mask] if loss.startswith("multilabel") else integers
                    for target in targets:

                        def per(x: jax.Array, t: jax.Array, loss: str = loss) -> jax.Array:
                            return native(x, t, loss)

                        def weighted(x: jax.Array, t: jax.Array) -> jax.Array:
                            return (per(x, t) * weights).sum()

                        def run(target: np.ndarray = target) -> Any:
                            grad = jax.jit(jax.value_and_grad(weighted))(z, target)
                            batch = (np.stack([z, z[::-1]]), np.stack([target, target[::-1]]))
                            return (
                                jax.jit(per)(z, target),
                                grad,
                                jax.jit(jax.vmap(per))(*batch),
                                jax.jit(jax.vmap(per, (0, None)))(batch[0], target),
                            )

                        name = f"{loss} {np.dtype(dtype)} {target.dtype} x64={enabled}"
                        with subtests.test(name), x64(enabled):
                            handlers, callback = both(run)
                            assert_same_bits(handlers, callback, name)

    def test_traced_invalid_rows_give_nan(self, subtests: pytest.Subtests):
        """Rows the native library rejects give NaN losses and gradients through the FFI
        handlers and the host callback alike (each repeats a failed call row by row); the
        other rows keep their values."""
        z, labels, mask = problem(np.float64)
        none, every = mask.copy(), mask.copy()
        none[1], every[1] = False, True
        cases = [
            ("topk_hinge_beta", z, np.where(np.arange(6) == 1, 7, labels)),
            ("multiclass_hinge", z, np.where(np.arange(6) == 1, -9, labels)),
            ("proximal_cross_entropy", z, np.where(np.arange(6) == 1, 7, labels)),
            ("multilabel_hinge", z, none),
            ("multilabel_hinge", z, every),
            ("multilabel_cross_entropy", z, none),
            ("topk_hinge_alpha", np.where(np.arange(6)[:, None] == 1, np.inf, z), labels),
            ("multilabel_hinge", np.where(np.arange(6)[:, None] == 1, np.nan, z), mask),
        ]
        for loss, scores, target in cases:
            with subtests.test(loss=loss):

                def f(x: jax.Array, t: jax.Array, loss: str = loss) -> jax.Array:
                    return native(x, t, loss, "none")

                def run(scores: np.ndarray = scores, target: np.ndarray = target) -> Any:
                    return (
                        jax.jit(f)(scores, target),
                        jax.jit(jax.grad(lambda x, t: f(x, t).sum()))(scores, target),
                    )

                handlers, callback = both(run)
                assert_same_bits(handlers, callback, loss)
                values, gradient = handlers
                assert np.isnan(values).tolist() == [i == 1 for i in range(6)], loss
                assert np.isnan(gradient[1]).all() and np.isfinite(gradient[[0, 2, 3, 4, 5]]).all()

    def test_results_do_not_depend_on_the_threads(self):
        rng = np.random.default_rng(3)
        z, labels = rng.normal(size=(4096, 300)), rng.integers(0, 300, 4096)

        def run() -> Any:
            loss = jax.jit(
                jax.value_and_grad(lambda x: native(x, labels, "topk_hinge_alpha", "sum"))
            )
            return jax.tree.map(np.asarray, loss(z))

        jax.clear_caches()
        with ffi("on"):
            automatic = run()
            with mock.patch.dict(os.environ, {"SDCA_NUM_THREADS": "1"}):
                serial = run()
        assert_same_bits(automatic, serial)

    def test_half_scores_compute_in_float32(self):
        z, labels, _ = problem(np.float32)
        half = jnp.asarray(z, jnp.bfloat16)

        def run() -> Any:
            return jax.jit(
                jax.value_and_grad(lambda x: native(x, labels, "topk_hinge_beta", "sum"))
            )(half)

        handlers, callback = both(run)
        assert_same_bits(handlers, callback)

    def test_build_rules_in_child_processes(self, tmp_path: Path):
        """auto builds once into SDCA_CACHE_DIR and later processes load it; without a compiler
        auto warns once and keeps the callback, on raises and off builds nothing."""
        code = textwrap.dedent(
            """
            import warnings, jax, numpy as np
            from libsdca import _jax_ffi
            from named_losses import jax_loss as named_loss
            z = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
            labels = np.array([0, 1, 2])
            def run(loss, **options):
                f = jax.jit(lambda x: named_loss(x, labels, loss, backend="native", **options))
                try:
                    print("value", float(f(z)))
                except RuntimeError as error:
                    print("raised", error)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                run("topk_hinge_beta", k=2)
                run("multiclass_hinge", gamma=0.5)
            print("warnings", sum(w.category is RuntimeWarning for w in caught))
            print("why", _jax_ffi.why_not() if _jax_ffi.mode() != "on" else "")
            """
        )

        def child(**env: str) -> str:
            environment = {**os.environ, "SDCA_CACHE_DIR": str(tmp_path), "JAX_PLATFORMS": "cpu"}
            environment.update(env)
            result = run_python("-c", code, capture_output=True, text=True, env=environment)
            assert result.returncode == 0, result.stderr
            return result.stdout + result.stderr

        def values(output: str) -> list[str]:
            return [line for line in output.splitlines() if line.startswith("value")]

        first = child(SDCA_JAX_FFI="auto")
        assert "building the JAX FFI handlers once" in first and "warnings 0" in first
        builds = [p for p in tmp_path.iterdir() if p.name.startswith("jax_ffi_")]
        assert len(builds) == 1 and (builds[0] / _jax_ffi.LIBRARY).is_file()
        second = child(SDCA_JAX_FFI="auto")
        assert "building" not in second and "warnings 0" in second
        missing = child(SDCA_JAX_FFI="auto", CXX="/nonexistent/c++")
        assert "warnings 1" in missing and "no C++ compiler" in missing
        off = child(SDCA_JAX_FFI="off", SDCA_CACHE_DIR=str(tmp_path / "off"))
        assert "why SDCA_JAX_FFI=off" in off and not (tmp_path / "off").exists()
        assert len(values(first)) == 2
        assert values(first) == values(second) == values(missing) == values(off)
        raised = child(SDCA_JAX_FFI="on", CXX="/nonexistent/c++")
        assert raised.count("raised libsdca: the JAX FFI handlers cannot be built here") == 2

    def test_failed_build_is_recorded(self, tmp_path: Path):
        error = _extensions.BuildFailure("bad headers")
        failing = mock.patch.object(_jax_ffi, "_build", side_effect=error)
        cache = mock.patch.dict(os.environ, {"SDCA_CACHE_DIR": str(tmp_path)})
        with failing, cache, ffi("auto"):
            _jax_ffi.register.cache_clear()
            _jax_ffi.status.cache_clear()
            try:
                with pytest.warns(RuntimeWarning, match="did not build") as caught:
                    available, reason = _jax_ffi.status()
                assert not available and "bad headers" in reason
                assert _jax_ffi.failure_record().is_file()
                assert "delete it or set SDCA_JAX_FFI=on to retry now" in str(caught[0].message)
                _jax_ffi.status.cache_clear()
                available, reason = _jax_ffi.status()
                assert not available and reason.startswith("an earlier build failed")
            finally:
                _jax_ffi.register.cache_clear()
                _jax_ffi.status.cache_clear()

    def test_failure_outside_the_compiler_is_not_recorded(self, tmp_path: Path):
        """A native library that does not load (a source tree without its build) warns in this
        process and leaves no record for later ones."""
        missing = OSError("Cannot load the native library build/python/libsdca_c.so")
        building = mock.patch.object(_jax_ffi, "_build")
        loading = mock.patch.object(_jax_ffi, "_register", side_effect=missing)
        cache = mock.patch.dict(os.environ, {"SDCA_CACHE_DIR": str(tmp_path)})
        with building, loading, cache, ffi("auto"):
            try:
                for _ in range(2):
                    _jax_ffi.register.cache_clear()
                    _jax_ffi.status.cache_clear()
                    with pytest.warns(RuntimeWarning, match="host callback in this process"):
                        available, reason = _jax_ffi.status()
                    assert not available and "Cannot load the native library" in reason
                    assert "could not be loaded" in reason and "did not build" not in reason
                    assert not _jax_ffi.failure_record().exists()
            finally:
                _jax_ffi.register.cache_clear()
                _jax_ffi.status.cache_clear()

    def test_corrupt_handler_names_rebuild_directory(self, tmp_path: Path):
        library = tmp_path / _jax_ffi.LIBRARY
        library.write_bytes(b"")
        with pytest.raises(OSError, match="delete .* to build the handlers again"):
            _jax_ffi._register(library)

    def test_stale_native_library_without_run_rows_falls_back(self, tmp_path: Path):
        fake_handlers = types.SimpleNamespace(
            sdca_jax_bind=lambda *_: 0,
            sdca_jax_bind_shared=mock.Mock(),
        )
        fake_native = types.SimpleNamespace(sdca_last_error=1)  # lacks sdca_internal_run_rows
        with (
            mock.patch.object(_jax_ffi.ctypes, "CDLL", return_value=fake_handlers),
            mock.patch.object(_jax_ffi.ctypes, "cast", side_effect=lambda x, _: x),
            mock.patch.object(_jax_ffi._core, "_tensor_entry", return_value=(fake_native, 2)),
            mock.patch.dict(os.environ, {"SDCA_CACHE_DIR": str(tmp_path)}),
            ffi("auto"),
        ):
            try:
                _jax_ffi.register.cache_clear()
                _jax_ffi.status.cache_clear()
                with pytest.warns(RuntimeWarning, match="sdca_internal_run_rows"):
                    available, reason = _jax_ffi.status()
                assert not available and "sdca_internal_run_rows" in reason
                assert not _jax_ffi.failure_record().exists()
            finally:
                _jax_ffi.register.cache_clear()
                _jax_ffi.status.cache_clear()

    @pytest.mark.parametrize(
        ("returncode", "stderr", "first"),
        [
            (1, "fatal error: bad headers\n", "(BuildFailure: fatal error: bad headers)"),
            (-9, "", "(RuntimeError: exit status -9)"),
        ],
        ids=["error", "signal"],
    )
    def test_compiler_exit_status(self, tmp_path: Path, returncode: int, stderr: str, first: str):
        """A compiler that exits with an error leaves a record that a later process skips; one
        killed by a signal (a negative status) may succeed next time, so every process warns
        and builds again."""
        run = mock.Mock(return_value=subprocess.CompletedProcess([], returncode, "", stderr))
        compiler = mock.patch.object(_jax_ffi, "subprocess", types.SimpleNamespace(run=run))
        cache = mock.patch.dict(os.environ, {"SDCA_CACHE_DIR": str(tmp_path)})
        recorded = returncode > 0
        with compiler, cache, ffi("auto"):
            try:
                for process in range(2):
                    _jax_ffi.register.cache_clear()
                    _jax_ffi.status.cache_clear()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        available, reason = _jax_ffi.status()
                    messages = [str(w.message) for w in caught if w.category is RuntimeWarning]
                    assert not available and f"did not build {first}" in reason
                    assert _jax_ffi.failure_record().is_file() == recorded
                    if recorded and process:
                        assert reason.startswith("an earlier build failed") and not messages
                    else:
                        assert len(messages) == 1 and "did not build" in messages[0]
                        assert ("host callback in this process" in messages[0]) != recorded
                assert run.call_count == (1 if recorded else 2)
                assert not [path for path in tmp_path.iterdir() if path.is_dir()]
            finally:
                _jax_ffi.register.cache_clear()
                _jax_ffi.status.cache_clear()
