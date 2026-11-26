"""Property-based tests of libsdca.jax and libsdca.optax: P1
(each gradient row sums to zero), P1' (no NaN) and P2 (a zero gradient at identical logits).
properties.py defines the properties, the inputs, the tolerances and the tests.

- test_p1_<loss>_<mode> and test_p2_<loss>_<mode>: Hypothesis tests of one loss and dtype
  mode over the CPU kernels (backend "kernel" and "auto"), the composed operations (reported
  separately: they run XLA's operations) and the CUDA kernels (jax-cuda: the FFI and Triton
  paths). Gradients are jax.vjp of the values with an all-ones cotangent, under jit.
- test_optax_p1_<loss>_<mode> and test_optax_p2_...: libsdca.optax's
  softmax_cross_entropy_with_integer_labels (cross_entropy), softmax_cross_entropy and
  safe_softmax_cross_entropy (cross_entropy_probabilities) and multiclass_hinge_loss, per
  example (no options, no precision).
- test_sweep_p1 and test_sweep_p2 (markers sweep and long): the batched sweep of
  properties.sweep_batches on the CPU and CUDA kernels.

Profiles: conftest.py (ci by default, search, replay). XLA flushes subnormal results of its
own operations on the CPU (docs/API.md, Known limits): the P1 bound adds C tiny_out for JAX.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import pytest

pytest.importorskip("hypothesis")

import interfaces  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import libsdca.jax as J  # noqa: E402
import libsdca.optax as O  # noqa: E402
import properties as P  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _gpu() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except RuntimeError:
        return False


CUDA = [pytest.mark.cuda, pytest.mark.skipif(not _gpu(), reason="needs a CUDA device")]
Backend = tuple[str, str, str]
BACKENDS = [
    pytest.param(("jax", "cpu", "kernel"), id="jax-cpu-kernel"),
    pytest.param(("jax", "cpu", "auto"), id="jax-cpu-auto"),
    pytest.param(("jax", "cpu", "composed"), id="jax-cpu-composed", marks=P.COMPOSED),
    # CUDA: the kernels only ("auto" runs the same kernels there).
    pytest.param(("jax", "cuda", "kernel"), id="jax-cuda-kernel", marks=CUDA),
]
SWEEP_BACKENDS = [
    pytest.param(("jax", "cpu", "kernel"), id="jax-cpu-kernel"),
    pytest.param(("jax", "cuda", "kernel"), id="jax-cuda-kernel", marks=CUDA),
]
OPTAX = {
    "softmax_cross_entropy_with_integer_labels": "cross_entropy",
    "softmax_cross_entropy": P.PROBABILITIES,
    "safe_softmax_cross_entropy": P.PROBABILITIES,
    "multiclass_hinge_loss": "multiclass_hinge",
}


def _device(name: str) -> Any:
    return jax.devices("gpu" if name == "cuda" else "cpu")[0]


def _array(x: np.ndarray, dtype: str, device: Any) -> Any:
    kind = jnp.bfloat16 if dtype == "bfloat16" else np.dtype(dtype)
    return jax.device_put(np.asarray(x, np.float64).astype(kind), device)


@functools.cache
def _jitted(family: str, parameters: tuple, options: tuple, precision: str | None) -> Any:
    """The jitted values and vjp (all-ones cotangent) of a libsdca.jax loss."""
    params, opts = dict(parameters), dict(options)

    def run(z: Any, target: Any, weight: Any) -> tuple[Any, Any]:
        extra = {} if weight is None else {"weight": weight}

        def loss(z: Any) -> Any:
            return interfaces.framework_call(
                J, family, z, target, params, precision=precision, **opts, **extra
            )

        value, pullback = jax.vjp(loss, z)
        (gradient,) = pullback(jnp.ones_like(value))
        return value, gradient

    return jax.jit(run)


@functools.cache
def _optax_jitted(name: str) -> Any:
    fn = getattr(O, name)

    def run(z: Any, target: Any) -> tuple[Any, Any]:
        value, pullback = jax.vjp(lambda z: fn(z, target), z)
        (gradient,) = pullback(jnp.ones_like(value))
        return value, gradient

    return jax.jit(run)


def _target(family: str, batch: P.Batch, dtype: str, device: Any) -> Any:
    t = P.target(family, batch)
    if t.dtype.kind == "f":
        return _array(t, dtype, device)
    return jax.device_put(t, device)


# Each compiled executable holds memory maps; a search compiles thousands per process (one
# per shape and set of options), and Linux allows 65,530 maps per process (vm.max_map_count):
# LLVM then aborts ("Unable to allocate section memory"). Clear the compiled functions after
# COMPILED_LIMIT distinct calls. The composed functions compile far larger programs than the
# kernels (a single-process search of composed proximal cross-entropy can approach the limit),
# hence 64.
COMPILED_LIMIT = 64
_compiled: set[tuple] = set()


def _bound_compiled(key: tuple) -> None:
    if key in _compiled:
        return
    if len(_compiled) >= COMPILED_LIMIT:
        _compiled.clear()
        _jitted.cache_clear()
        _optax_jitted.cache_clear()
        jax.clear_caches()
    _compiled.add(key)


def call(backend: Backend, family: str, mode: str, batch: P.Batch) -> tuple[Any, ...]:
    """(values, gradient as float64, gradient dtype) of the libsdca.jax or libsdca.optax
    loss (optax: backend[2] names the function)."""
    framework, device_name, option = backend
    dtype, precision = P.MODES[mode]
    device = _device(device_name)
    z = _array(batch.logits, dtype, device)
    target = _target(family, batch, dtype, device)
    with jax.default_device(device):
        if framework == "optax":
            _bound_compiled((option, z.shape, z.dtype, target.shape, target.dtype))
            value, gradient = _optax_jitted(option)(z, target)
        else:
            opts = P.options(family, batch)
            weight = opts.pop("weight", None)
            if weight is not None:
                weight = _array(weight, dtype, device)
            options = (*sorted(opts.items()), ("reduction", batch.reduction))
            options += (("backend", option),)
            params = tuple(sorted(P.parameters(family, batch).items()))
            shapes = (z.shape, z.dtype, target.shape, target.dtype, weight is None)
            _bound_compiled((family, params, options, precision, device_name, *shapes))
            value, gradient = _jitted(family, params, options, precision)(z, target, weight)
    out = str(gradient.dtype)
    return np.asarray(value, np.float64), np.asarray(gradient, np.float64), out


P.install(globals(), call, BACKENDS)
for _name, _family in OPTAX.items():
    _backends = [pytest.param(("optax", "cpu", _name), id=f"optax-{_name}")]
    _p2 = (_family,) if _family in P.P2_FAMILIES else ()
    _modes = tuple(m for m in P.MODES if m != "float32-p64")
    P.install(
        globals(),
        call,
        _backends,
        families=(_family,),
        p2_families=_p2,
        modes=_modes,
        prefix=f"optax_{_name}_",
    )

P.install_documented(globals(), call, [pytest.param(("jax", "cpu", "kernel"), id="jax-cpu-kernel")])


@pytest.mark.sweep
@pytest.mark.long
@pytest.mark.parametrize("backend", SWEEP_BACKENDS)
@pytest.mark.parametrize("mode", list(P.MODES))
@pytest.mark.parametrize("family", P.FAMILIES)
def test_sweep_p1(family: str, mode: str, backend: Backend) -> None:
    P.sweep(call, "p1", family, mode, backend)


@pytest.mark.sweep
@pytest.mark.long
@pytest.mark.parametrize("backend", SWEEP_BACKENDS)
@pytest.mark.parametrize("mode", list(P.MODES))
@pytest.mark.parametrize("family", P.P2_FAMILIES)
def test_sweep_p2(family: str, mode: str, backend: Backend) -> None:
    P.sweep(call, "p2", family, mode, backend)
