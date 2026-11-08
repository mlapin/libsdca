"""The XLA FFI calls of the native JAX losses (_jax_ffi) against the host callback they replace:
the same bits, dtypes and messages under jit, grad and vmap, and the build rules."""

import os
import textwrap
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from child_process import run_python
from jax.typing import ArrayLike
from libsdca import LOSSES, _jax_ffi
from libsdca.jax import sdca_loss

jax.config.update("jax_enable_x64", True)

SINGLE = ("topk_cross_entropy", "multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta")
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
    return sdca_loss(x, t, loss, backend="native", reduction=reduction, k=2, gamma=0.6, alpha=0.9)


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

    def test_every_loss_matches_the_host_callback_bitwise(self, subtests: pytest.Subtests):
        """Losses and gradients of every loss, labels of several integer types and masks,
        float32 and float64 scores, with and without x64 (float32 losses rounded from
        double), forward alone and with the gradient, jit and vmap."""
        for enabled in (True, False):
            for dtype in (np.float32, np.float64) if enabled else (np.float32,):
                z, labels, mask = problem(dtype)
                weights = np.linspace(0.5, 2.0, len(z)).astype(dtype)
                for loss in LOSSES:
                    one = np.eye(7, dtype=bool)[labels] if loss in SINGLE else mask
                    integers = [labels.astype(np.int32 if enabled else np.int64), labels]
                    integers.append(labels.astype(np.uint8))
                    targets = [one] if loss.startswith("multilabel") else [*integers, one]
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

    def test_traced_invalid_inputs_raise_the_host_callback_messages(
        self, subtests: pytest.Subtests
    ):
        z, labels, mask = problem(np.float64)
        none, every, two = np.zeros_like(mask), np.ones_like(mask), mask.copy()
        two[:, :2] = True
        first = "Every example must have at least one positive label"
        cases = [
            ("topk_hinge_beta", z, labels + 7, first),
            ("cross_entropy", z, labels - 9, first),
            ("multilabel_hinge", z, none, first),
            ("topk_hinge_alpha", z, two, "This loss needs exactly one target"),
            ("multilabel_hinge", z, every, "Multilabel hinge needs a negative label"),
            ("proximal_cross_entropy", z, every, "Proximal entropy needs a non-target class"),
            ("topk_hinge_beta", np.where(mask, np.inf, z), labels + 7, "Inputs must be finite"),
            ("multilabel_hinge", np.where(mask, np.nan, z), none, "Inputs must be finite"),
        ]
        for loss, scores, target, message in cases:
            for setting in ("on", "off"):
                with subtests.test(loss=loss, message=message, setting=setting):

                    def f(x: jax.Array, t: jax.Array, loss: str = loss) -> jax.Array:
                        return native(x, t, loss, "mean")

                    jax.clear_caches()
                    with ffi(setting), pytest.raises(Exception, match=message):
                        jax.jit(jax.grad(f))(scores, target).block_until_ready()

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
            from libsdca.jax import sdca_loss
            z = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
            labels = np.array([0, 1, 2])
            def run(loss, **options):
                f = jax.jit(lambda x: sdca_loss(x, labels, loss, backend="native", **options))
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
        failing = mock.patch.object(_jax_ffi, "_build", side_effect=RuntimeError("bad headers"))
        cache = mock.patch.dict(os.environ, {"SDCA_CACHE_DIR": str(tmp_path)})
        with failing, cache, ffi("auto"):
            _jax_ffi.register.cache_clear()
            _jax_ffi.status.cache_clear()
            try:
                with pytest.warns(RuntimeWarning, match="did not build"):
                    available, reason = _jax_ffi.status()
                assert not available and "bad headers" in reason
                assert _jax_ffi.failure_record().is_file()
                _jax_ffi.status.cache_clear()
                available, reason = _jax_ffi.status()
                assert not available and reason.startswith("an earlier build failed")
            finally:
                _jax_ffi.register.cache_clear()
                _jax_ffi.status.cache_clear()
