"""The fused Triton kernels of libsdca.jax on CUDA GPUs (_jax_cuda).

Without a GPU: the routing (the fused paths libsdca.torch takes), the SDCA_JAX_CUDA rules, the
label and saved-gradient conversions, the handler's build, the launch plans against the
launches of the PyTorch path's Triton launch (recorded without running them), compiling the
kernels for sm_80, and jit, grad and vmap of the calls with the kernels emulated by their
jax.numpy references. On a CUDA GPU: the bits of the PyTorch
CUDA path, float64 references, derivatives, vmap, export and the fallback.
"""

import functools
import importlib.util
import json
import math
import os
import types
import warnings
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import named_losses
import numpy as np
import pytest
from jax.interpreters import mlir
from libsdca import _cuda_launch, _extensions, _jax_cuda, _jax_ffi
from libsdca import jax as sdca_jax
from libsdca.jax import Reduction

TRITON = importlib.util.find_spec("triton") is not None
TORCH = importlib.util.find_spec("torch") is not None
needs_triton = pytest.mark.skipif(not TRITON, reason="Triton is not installed")
needs_torch_triton = pytest.mark.skipif(not (TRITON and TORCH), reason="needs torch and Triton")


def _gpu() -> bool:
    try:
        return _jax_cuda.architecture() is not None and _jax_cuda.status()[0]
    except Exception:  # noqa: BLE001 - any failure: no GPU tests
        return False


needs_gpu = pytest.mark.skipif(not _gpu(), reason="no CUDA GPU with the fused JAX kernels")


@needs_gpu
@needs_torch_triton
@pytest.mark.parametrize("classes", [300, 1000, 4096, 32000])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_adaptive_distillation_matches_torch(
    classes: int, reduction: Reduction, dtype: str
) -> None:
    import torch
    from libsdca.torch import distillation_loss
    from test_distillation_adaptive import mixed_inputs

    s, t = (x.to(getattr(torch, dtype)) for x in mixed_inputs(classes))
    s.requires_grad_()
    expected = distillation_loss(s, t, temperature=2.0, reduction=reduction)
    weight = torch.full_like(expected, 1.7)
    (gradient,) = torch.autograd.grad(expected, s, weight)
    z, teacher = (
        jnp.asarray(x.detach().float().cpu().numpy(), dtype=getattr(jnp, dtype)) for x in (s, t)
    )
    upstream = jnp.asarray(weight.cpu().numpy())

    @jax.jit
    def run(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        value, backward = jax.vjp(
            lambda a: sdca_jax.distillation_loss(a, teacher, temperature=2.0, reduction=reduction),
            x,
        )
        return value, backward(upstream)[0]

    with setting("on"):
        actual, grad = run(z)
    np.testing.assert_array_equal(np.asarray(actual), expected.detach().cpu().numpy())
    np.testing.assert_array_equal(np.asarray(grad, np.float32), gradient.float().cpu().numpy())


@contextmanager
def setting(value: str | None) -> Iterator[None]:
    environment = {} if value is None else {_jax_cuda.ENV: value}
    with mock.patch.dict(os.environ, environment):
        if value is None:
            os.environ.pop(_jax_cuda.ENV, None)
        _jax_cuda.decide.cache_clear()
        try:
            yield
        finally:
            _jax_cuda.decide.cache_clear()


# Routing: the fused paths of libsdca.torch's _device_loss and distillation (backend "auto").

ROUTES = [
    # (loss, k, gamma, labels, dtype, precision, family)
    ("cross_entropy", 1, 0.0, True, "float32", None, "entropy"),
    ("cross_entropy", 1, 0.0, True, "bfloat16", "float32", "entropy"),
    ("cross_entropy", 1, 0.0, True, "float32", "float64", "entropy"),
    ("topk_cross_entropy", 1, 0.0, True, "float16", None, "entropy"),
    ("cross_entropy", 1, 0.0, False, "float32", None, "multilabel"),
    ("multilabel_cross_entropy", 1, 0.0, False, "bfloat16", None, "multilabel"),
    ("cross_entropy", 1, 0.0, False, "float32", "float32", "multilabel"),
    ("multiclass_hinge", 1, 0.0, True, "float32", None, "hinge"),
    ("multiclass_hinge", 1, 0.5, True, "float32", "float64", "hinge"),
    ("multiclass_hinge", 1, 0.0, True, "float32", "float32", "hinge"),
    ("topk_hinge_alpha", 1, 0.5, True, "bfloat16", None, "hinge"),
    ("topk_hinge_beta", 3, 0.0, True, "float32", None, "topk"),
    ("topk_hinge_beta", 3, 0.5, True, "float32", None, "topk"),
    ("topk_hinge_alpha", 33, 0.0, True, "float32", None, None),
    ("topk_cross_entropy", 5, 0.0, True, "float32", None, "topk"),
    ("topk_cross_entropy", 5, 0.0, True, "float32", "float32", "topk"),
    ("topk_cross_entropy", 1, 0.0, False, "float32", None, None),
    ("topk_hinge_beta", 3, 0.0, False, "float32", None, None),
    ("multilabel_hinge", 1, 0.0, False, "float32", None, "multilabel"),
    ("multilabel_hinge", 1, 0.5, False, "float16", None, "multilabel"),
    ("multilabel_hinge", 1, 0.0, False, "float32", "float32", "multilabel"),
    ("proximal_cross_entropy", 1, 0.0, True, "float32", None, "prox"),
    ("proximal_cross_entropy", 1, 0.0, False, "bfloat16", None, "prox"),
    ("proximal_cross_entropy", 1, 0.0, True, "float32", "float32", "prox"),
]


@pytest.mark.parametrize("case", ROUTES, ids=lambda c: "-".join(map(str, c[:6])))
def test_routes_follow_the_torch_paths(case: tuple[Any, ...]):
    from libsdca._core import _tensor_kind

    loss, k, gamma, labels, dtype, precision, family = case
    objective, kind_k, kind_gamma = _tensor_kind(loss, k, gamma)
    shape = (64, 1000)
    route = _jax_cuda._route(objective, kind_k, labels, dtype, precision, shape, kind_gamma, 1.0)
    assert (None if route is None else route[0]) == family


def test_route_options():
    """The fp32 flags of the families, as libsdca.torch passes them."""

    def route(*args: Any) -> Any:
        found = _jax_cuda._route(*args)
        return (None, (None,) * 4) if found is None else found

    assert route("cross_entropy", 1, True, "float32", None, (4, 10), 1.0, 1.0) == (
        "entropy",
        (True,),
    )
    assert route("cross_entropy", 1, True, "float64", None, (4, 10), 1.0, 1.0) == (
        "entropy",
        (False,),
    )
    assert route("cross_entropy", 1, True, "float32", "float64", (4, 10), 1.0, 1.0)[1] == (False,)
    assert route("topk_hinge_beta", 3, True, "float32", None, (4, 10), 1.0, 1.0)[1][3] is False
    assert route("topk_cross_entropy", 3, True, "float32", None, (4, 10), 1.0, 1.0)[1][3] is True
    assert route("multilabel_hinge", 1, False, "float32", None, (4, 10), 1.0, 1.0)[1][1] is False
    assert route("cross_entropy", 1, False, "float16", None, (4, 10), 1.0, 1.0)[1][1] is True
    for gamma in (0.0, 0.5, 1e-100):
        options = route("topk_hinge_alpha", 1, True, "float32", None, (4, 10), gamma, 1.0)[1]
        assert options[-1] is (gamma != 1e-100)
        assert (
            route("topk_hinge_alpha", 1, True, "float32", "float64", (4, 10), gamma, 1.0)[1][-1]
            is False
        )
    # proximal_cross_entropy: float32 for 16-bit logits (or precision="float32") where alpha
    # holds; else float64.
    assert route("proximal_cross_entropy", 1, True, "bfloat16", None, (4, 10), 1.0, 0.5)[1] == (
        0.5,
        True,
    )
    assert route("proximal_cross_entropy", 1, True, "float32", None, (4, 10), 1.0, 0.5)[1] == (
        0.5,
        True,
    )
    assert (
        route("proximal_cross_entropy", 1, True, "bfloat16", None, (4, 10), 1.0, 1e-39)[1][1]
        is False
    )
    # Shape limits: rows, classes of the entropy kernels, top-k's k and k < classes - 1.
    assert route("cross_entropy", 1, True, "float32", None, (2**28 + 1, 10), 1.0, 1.0)[0] is None
    assert route("cross_entropy", 1, True, "float32", None, (4, 2**25 + 1), 1.0, 1.0)[0] is None
    assert route("topk_cross_entropy", 9, True, "float32", None, (4, 10), 1.0, 1.0)[0] is None
    assert route("topk_cross_entropy", 8, True, "float32", None, (4, 10), 1.0, 1.0)[0] == "topk"


# SDCA_JAX_CUDA.


def test_off_keeps_jax_numpy_silently():
    with setting("off"), warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _jax_cuda.status() == (False, "SDCA_JAX_CUDA=off")


def test_missing_tools_warn_once_in_auto_and_raise_with_on():
    reason = "Triton is not installed (test)"
    with mock.patch.object(_jax_cuda, "missing", return_value=reason):
        for value in (None, "auto"):
            with setting(value):
                with pytest.warns(RuntimeWarning, match="use jax.numpy"):
                    assert _jax_cuda.status() == (False, reason)
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    assert _jax_cuda.status() == (False, reason)
        with setting("on"), pytest.raises(RuntimeError, match="cannot run here"):
            _jax_cuda.status()
    with setting("maybe"), pytest.raises(ValueError, match="SDCA_JAX_CUDA"):
        _jax_cuda.status()


def test_unverified_triton_is_skipped_silently_in_auto():
    fake = mock.Mock(__version__="3.99.0")
    with (
        mock.patch.dict("sys.modules", {"triton": fake}),
        mock.patch.object(_jax_cuda, "missing", return_value=None),
        setting("auto"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")
        available, reason = _jax_cuda.status()
    assert not available and "3.99.0 is outside the verified 3.3-3.9" in reason
    assert "SDCA_JAX_CUDA=on tries it" in reason


@pytest.mark.skipif(not hasattr(jax, "ffi"), reason="requires JAX FFI")
def test_failed_handler_build_is_recorded(tmp_path: Path):
    fake = mock.Mock(__version__="3.9.0")
    with (
        mock.patch.dict("sys.modules", {"triton": fake}),
        mock.patch.object(_jax_cuda, "missing", return_value=None),
        mock.patch.object(_jax_ffi, "missing_tools", return_value=None),
        mock.patch.object(_jax_ffi, "_build", side_effect=_extensions.BuildFailure("bad headers")),
        mock.patch.dict(os.environ, {"SDCA_CACHE_DIR": str(tmp_path)}),
        setting("auto"),
    ):
        _jax_cuda.register.cache_clear()
        try:
            with pytest.warns(RuntimeWarning, match="did not build"):
                available, reason = _jax_cuda.status()
            assert not available and "bad headers" in reason
            assert _jax_ffi.failure_record(_jax_cuda.HANDLERS).is_file()
            _jax_cuda.decide.cache_clear()
            assert _jax_cuda.status()[1].startswith("an earlier build failed")
        finally:
            _jax_cuda.register.cache_clear()


@pytest.mark.skipif(not hasattr(jax, "ffi"), reason="requires JAX FFI")
def test_failed_builds_of_one_handler_leave_the_other_alone(tmp_path: Path):
    with (
        mock.patch.object(_jax_ffi, "missing_tools", return_value=None),
        mock.patch.dict(os.environ, {"SDCA_CACHE_DIR": str(tmp_path)}),
    ):
        cpu, cuda = (
            _jax_ffi.failure_record(_jax_ffi.CPU),
            _jax_ffi.failure_record(_jax_cuda.HANDLERS),
        )
        assert cpu != cuda
        assert _jax_ffi.build_directory(_jax_ffi.CPU) != _jax_ffi.build_directory(
            _jax_cuda.HANDLERS
        )
        for failed, other in (
            (_jax_ffi.CPU, _jax_cuda.HANDLERS),
            (_jax_cuda.HANDLERS, _jax_ffi.CPU),
        ):
            with pytest.warns(RuntimeWarning, match="bad headers"):
                _jax_ffi.automatic(
                    failed, mock.Mock(side_effect=_extensions.BuildFailure("bad headers"))
                )
            assert _jax_ffi.failure_record(failed).is_file()
            assert not _jax_ffi.failure_record(other).exists()
            assert _jax_ffi.automatic(other, mock.Mock()) == (True, "")
            assert _jax_ffi.automatic(failed, mock.Mock())[1].startswith("an earlier build failed")
            _jax_ffi.failure_record(failed).unlink()


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="default backend not CPU")
def test_cpu_calls_never_ask_for_the_kernels():
    z = jnp.asarray(np.random.default_rng(0).normal(size=(5, 7)), jnp.float32)
    labels, mask = jnp.arange(5) % 7, z > 0.5
    teacher = z[::-1]
    with mock.patch.object(_jax_cuda, "status", side_effect=AssertionError("asked")):
        jax.jit(lambda z: sdca_jax.cross_entropy(z, labels))(z)
        sdca_jax.multilabel_cross_entropy(z, mask | (jnp.arange(7) == 0))
        sdca_jax.distillation_loss(z, teacher, temperature=2.0)
        sdca_jax.proximal_distillation_loss(z, teacher, temperature=2.0, alpha=0.5)
        sdca_jax.topk_hinge_loss(z, labels, k=2, precision="float32", backend="auto")
    assert not _jax_cuda.on_gpu(z)


# Conversions.


def x64() -> Any:
    """A thread-local x64 scope: jax.enable_x64 in newer releases, else jax.experimental's."""
    import jax.experimental

    scope = getattr(jax, "enable_x64", None)
    if scope is None:
        scope = getattr(jax.experimental, "enable_x64", None)
    assert scope is not None
    return scope(True)


def test_labels_are_int64_words():
    words = _jax_cuda._labels(jnp.asarray([0, 3, -1, 2**31 - 1, -(2**31)], jnp.int32))
    expected = np.asarray([0, 3, -1, 2**31 - 1, -(2**31)], np.int64).view(np.uint32).reshape(-1, 2)
    np.testing.assert_array_equal(np.asarray(words), expected)
    unsigned = _jax_cuda._labels(jnp.asarray([0, 255], jnp.uint8))
    np.testing.assert_array_equal(np.asarray(unsigned), [[0, 0], [255, 0]])
    spec = _jax_cuda.Spec("entropy", (True,), "forward", 5, 7, "float32", "none")
    back = np.asarray(_jax_cuda._reference_target(words, spec))
    np.testing.assert_array_equal(back, [0, 3, -1, 2**31 - 1, -(2**31)])


def test_padded_elements_are_16_bytes():
    assert _jax_cuda._padded(1, np.float32) == 4
    assert _jax_cuda._padded(5, np.float64) == 6
    assert _jax_cuda._padded(3, np.bool_) == 16
    assert _jax_cuda._padded(8, jnp.bfloat16) == 8
    assert _jax_cuda._padded(0, np.float32) == 4


@pytest.mark.skipif(not TORCH, reason="needs torch")
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("work", [np.float32, np.float64])
def test_saved_gradient_backward_is_torchs(reduction: Reduction, work: Any):
    """The rows stage's reduction and backward (proximal losses) as libsdca.torch's
    _SavedGradient: the reduced values' cotangent per row in the values' dtype, cast to the
    gradient's dtype, times the gradient, rounded once into the logits' dtype."""
    import torch
    from libsdca import _cpu

    with x64():
        rng = np.random.default_rng(1)
        values = rng.random(7).astype(work)
        gradient = rng.normal(size=(7, 5)).astype(np.float32) * 1e-3
        upstream = np.float32(0.37) if reduction != "none" else rng.random(7).astype(np.float32)
        spec = _jax_cuda.Spec("prox", (1.0, False), "rows", 7, 5, "bfloat16", reduction)
        like = jnp.zeros((0,), jnp.bfloat16)
        residuals = (jnp.asarray(values), jnp.asarray(gradient), like)
        out = _jax_cuda._reduced(jnp.asarray(values), spec)
        ct = jnp.asarray(upstream, out.dtype)
        jax_gradient = _jax_cuda._rows_bwd(spec, residuals, ct)[0]
        t_values = torch.tensor(values, requires_grad=True)
        reduced = {"none": t_values, "mean": t_values.mean(), "sum": t_values.sum()}[reduction]
        reduced = reduced.to(torch.float32)
        (row_upstream,) = torch.autograd.grad(reduced, t_values, torch.tensor(upstream))
        t_gradient = _cpu.scaled(torch.tensor(gradient), row_upstream, torch.bfloat16, False)
        np.testing.assert_allclose(np.asarray(out), reduced.detach().numpy(), rtol=1e-6)
        expected = t_gradient.float().numpy()
        np.testing.assert_array_equal(np.asarray(jax_gradient.astype(jnp.float32)), expected)


# The handler.


@pytest.mark.skipif(_jax_ffi.missing_tools() is not None, reason="no FFI build tools")
def test_handler_builds_and_exports_its_stages(tmp_path: Path):
    import ctypes

    directory = tmp_path / "handler"
    _jax_ffi._build(directory, _jax_cuda.HANDLERS)
    library = ctypes.CDLL(str(directory / _jax_cuda.LIBRARY))
    assert library.sdca_jax_cuda_execute and library.sdca_jax_cuda_initialize
    assert library.sdca_jax_cuda_preload


# The handler's launch program against a fake driver: what cuLaunchKernel receives.

FAKE_DRIVER = r"""
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static FILE* out() {
  static FILE* file = std::fopen(std::getenv("FAKE_CUDA_LOG"), "w");
  return file;
}
// A function's handle is its parameter count + 1, from its name "k<count>".
extern "C" {
int cuModuleLoadData(void** module, const void* image) {
  std::fprintf(out(), "load %s\n", static_cast<const char*>(image));
  *module = reinterpret_cast<void*>(16);
  return 0;
}
int cuModuleGetFunction(void** function, void*, const char* name) {
  *function = reinterpret_cast<void*>(static_cast<std::intptr_t>(std::atoi(name + 1) + 1));
  return 0;
}
int cuLaunchKernel(void* function, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned,
                   unsigned, unsigned shared, void*, void** params, void**) {
  const int count = static_cast<int>(reinterpret_cast<std::intptr_t>(function)) - 1;
  std::fprintf(out(), "launch %u %u %u %u %u", gx, gy, gz, bx, shared);
  for (int i = 0; i < count; ++i) {
    std::uint64_t value = 0;
    std::memcpy(&value, params[i], sizeof(value));
    std::fprintf(out(), " %llu", static_cast<unsigned long long>(value));
  }
  std::fprintf(out(), "\n");
  std::fflush(out());
  return 0;
}
int cuStreamGetCtx(void*, void** context) { *context = reinterpret_cast<void*>(1); return 0; }
int cuCtxGetCurrent(void** context) { *context = reinterpret_cast<void*>(1); return 0; }
int cuCtxPushCurrent_v2(void*) { return 0; }
int cuCtxPopCurrent_v2(void**) { return 0; }
int cuCtxGetDevice(int* device) { *device = 0; return 0; }
int cuDeviceGetAttribute(int* value, int, int) { *value = 101376; return 0; }
int cuFuncGetAttribute(int* value, int, void*) { *value = 0; return 0; }
int cuFuncSetAttribute(void*, int, int) { return 0; }
int cuFuncSetCacheConfig(void*, int) { return 0; }
int cuStreamIsCapturing(void*, int* capturing) { *capturing = 0; return 0; }
int cuGetErrorString(int, const char** text) { *text = "fake"; return 0; }
}
"""

HARNESS = r"""
#include "_jax_cuda.cpp"
// The handler's interpreter on host buffers, launching through the fake driver.
extern "C" int harness(const std::int64_t* kernels, std::size_t kernel_size, const char* binaries,
                       std::size_t binary_size, const std::int64_t* program,
                       std::size_t program_size, char** buffers, std::size_t buffer_count,
                       const std::int64_t* strides, std::int64_t batch) {
  ContextScope scope;
  if (scope.enter(nullptr).failure()) return 1;
  std::vector<CUfunction> loaded_kernels;
  ffi::Span<const std::int64_t> table(kernels, kernel_size);
  if (functions(scope.context, nullptr, table, std::string_view(binaries, binary_size),
                loaded_kernels).failure())
    return 2;
  std::vector<char*> addresses(buffers, buffers + buffer_count);
  for (std::int64_t element = 0; element < batch; ++element) {
    ffi::Error error = run(nullptr, loaded_kernels, table,
                           ffi::Span<const std::int64_t>(program, program_size), addresses,
                           ffi::Span<const std::int64_t>(strides, buffer_count), element);
    if (error.failure()) {
      std::fprintf(stderr, "%s\n", error.message().c_str());
      return 3;
    }
  }
  return 0;
}
"""

HARNESS_RUN = """
import ctypes, json, sys
import numpy as np
from unittest import mock
from libsdca import _jax_cuda

harness = ctypes.CDLL(sys.argv[1]).harness
i64 = ctypes.POINTER(ctypes.c_int64)
size = ctypes.c_size_t
harness.argtypes = [i64, size, ctypes.c_char_p, size, i64, size, ctypes.POINTER(ctypes.c_char_p),
                    size, i64, ctypes.c_int64]
k3 = _jax_cuda.Kernel(b"cubin-k5", "k5", 4, 0, ("*fp32", "*i64", "i32"))
k4 = _jax_cuda.Kernel(b"cubin-k6", "k6", 8, 512, ("*bf16", "*fp64", "fp32", "i64"))
one = _jax_cuda.Launch("first", (3, 2), (_jax_cuda.Ref(0, "float32"), _jax_cuda.Ref(2, "int64"), 7), 4)
two = _jax_cuda.Launch("second", (5,), (_jax_cuda.Ref(1, "bfloat16"), _jax_cuda.Ref(3, "float64"), 0.5, 2**40), 8)
compiled = {one: (k3, one.args), two: (k4, two.args)}
with mock.patch.object(_jax_cuda, "compile_launch", lambda launch, arch: compiled[launch]), \\
        mock.patch.object(_jax_cuda, "_loaded_sizes", return_value=None):
    program = _jax_cuda.encode([one, two, one], 80)
buffers = [np.zeros(64, np.uint8) for _ in range(4)]
pointers = (ctypes.c_char_p * 4)(*[ctypes.c_char_p(b.ctypes.data) for b in buffers])
batch = int(sys.argv[2])
strides = np.asarray([16, 32, 0, 48], np.int64) if batch > 1 else np.zeros(4, np.int64)
binaries = program.binaries.encode()
code = harness(
    program.kernels.ctypes.data_as(i64), program.kernels.size, binaries, len(binaries),
    program.program.ctypes.data_as(i64), program.program.size, pointers, 4,
    strides.ctypes.data_as(i64), batch,
)
print(json.dumps({"code": code, "addresses": [b.ctypes.data for b in buffers],
                  "strides": strides.tolist()}))
"""


@pytest.mark.skipif(_jax_ffi.missing_tools() is not None, reason="no FFI build tools")
@pytest.mark.parametrize("batch", [1, 3])
def test_launch_program_reaches_cuLaunchKernel(tmp_path: Path, batch: int):
    """Encoded launches (kernel table, base64 binaries, buffer and scalar arguments, Triton's
    null scratch pointers, per-element strides) as the handler passes them to the driver."""
    import subprocess
    import sys

    from child_process import run_python

    compiler = os.environ.get("CXX", "c++").split()
    fake = tmp_path / "libcuda.so.1"
    (tmp_path / "fake.cpp").write_text(FAKE_DRIVER)
    name = (
        ["-Wl,-install_name,libcuda.so.1"]
        if sys.platform == "darwin"
        else ["-Wl,-soname,libcuda.so.1"]
    )
    subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            *name,
            str(tmp_path / "fake.cpp"),
            "-o",
            str(fake),
        ],
        check=True,
    )
    (tmp_path / "harness.cpp").write_text(HARNESS)
    source = Path(_jax_cuda.__file__).parent
    harness = tmp_path / "harness.so"
    subprocess.run(
        [*compiler, "-std=c++17", "-shared", "-fPIC", "-isystem", jax.ffi.include_dir(), "-I", str(source),
         str(tmp_path / "harness.cpp"), "-o", str(harness)],
        check=True,
    )  # fmt: skip
    log = tmp_path / "launches.log"
    path = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    env = {**os.environ, path: str(tmp_path), "FAKE_CUDA_LOG": str(log), "JAX_PLATFORMS": "cpu"}
    result = run_python(
        "-c", HARNESS_RUN, str(harness), str(batch), capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["code"] == 0
    lines = log.read_text().splitlines()
    assert lines[:2] == ["load cubin-k5", "load cubin-k6"]
    a, s = report["addresses"], report["strides"]
    expected = []
    for element in range(batch):
        at = [a[i] + element * s[i] for i in range(4)]
        first = f"launch 3 2 1 128 0 {at[0]} {at[2]} 7 0 0"
        half = int.from_bytes(np.float32(0.5).tobytes(), "little")
        second = f"launch 5 1 1 256 512 {at[1]} {at[3]} {half} {2**40} 0 0"
        expected += [first, second, first]
    assert lines[2:] == expected


def _spec(
    family: str,
    options: tuple[Any, ...],
    stage: str,
    n: int,
    c: int,
    dtype: str,
    reduction: str,
) -> Any:
    return _jax_cuda.Spec(family, options, stage, n, c, dtype, reduction)


@contextmanager
def _recorded() -> Iterator[list[tuple[Any, tuple[int, ...], tuple[Any, ...], dict[str, Any]]]]:
    """The launches of Triton kernels, recorded instead of run; CPU tensors start zeroed (as
    the CPU checks of the Triton path read them) and the C++ launcher is off."""
    import torch
    from triton.runtime.jit import JITFunction

    launches: list[tuple[Any, tuple[int, ...], tuple[Any, ...], dict[str, Any]]] = []

    def run(self: Any, *args: Any, grid: Any, warmup: bool, **kwargs: Any) -> None:
        launches.append((self, tuple(grid), args, kwargs))

    with (
        mock.patch.object(JITFunction, "run", run),
        mock.patch.object(torch, "empty", torch.zeros),
        mock.patch.object(_cuda_launch, "status", return_value=(None, "recording")),
    ):
        yield launches


def _roles(spec: Any, tensors: dict[str, Any]) -> dict[int, str]:
    """The roles of the call's operand and result indices."""
    params = len(_jax_cuda.plan(spec).params)
    if spec.stage == "backward":
        operands = ["logits", "target", "stats", "upstream"] + ["param"] * params
        results = ["gradient"]
    elif spec.stage == "rows":
        operands = ["logits", "target"] + ["param"] * params
        results = ["loss", "gradient", "status"][: len(_jax_cuda.plan(spec).results)]
    else:
        operands = ["logits", "target"] + ["param"] * params
        results = ["out", "stats"]
    return dict(enumerate(operands + results))


def _role_of(tensor: Any, tensors: dict[str, Any]) -> str:
    for name, t in tensors.items():
        if t is not None and t.data_ptr() == tensor.data_ptr():
            return name
    return "param"


def _assert_same_launches(spec: Any, recorded: list[Any], tensors: dict[str, Any]) -> None:
    import torch

    plan = _jax_cuda.plan(spec)
    roles = _roles(spec, tensors)
    assert len(recorded) == len(plan.launches), (spec, [r[0] for r in recorded])
    for (kernel, grid, args, kwargs), launch in zip(recorded, plan.launches, strict=True):
        assert kernel is launch.kernel
        assert grid == launch.grid
        assert len(args) == len(launch.args)
        for got, want in zip(args, launch.args, strict=True):
            if isinstance(got, torch.Tensor):
                assert isinstance(want, _jax_cuda.Ref), (kernel, got, want)
                # proximal_entropy keeps its status buffer to itself.
                role = _role_of(got, tensors)
                assert roles[want.index] in (role, "status" if role == "param" else role), want
                assert str(got.dtype).split(".")[-1] == want.dtype, (kernel, got.dtype, want)
            else:
                assert not isinstance(want, _jax_cuda.Ref) and got == want, (kernel, got, want)
        options = dict(launch.options)
        assert kwargs.pop("num_warps") == launch.warps
        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                assert isinstance(options[name], _jax_cuda.Ref)
            else:
                assert options[name] == value, (kernel, name)
        assert set(kwargs) == set(options)


def _params_match(spec: Any, scalars: list[Any]) -> None:
    import torch

    words = [p for p in _jax_cuda.plan(spec).params]
    assert len(words) == len(scalars)
    for w, t in zip(words, scalars, strict=True):
        assert isinstance(t, torch.Tensor)
        assert w.tobytes() == t.contiguous().cpu().numpy().tobytes()


SHAPES = [(3, 10), (37, 100), (5, 1000), (2, 5000), (20000, 7), (3, 40000)]
DTYPES = ["float32", "bfloat16", "float64"]


def _torch_dtype(name: str) -> Any:
    import torch

    return getattr(torch, name)


def _inputs(n: int, c: int, dtype: str) -> tuple[Any, Any, Any, Any]:
    import torch

    generator = torch.Generator().manual_seed(n * 1000 + c)
    z = torch.randn(n, c, generator=generator).to(_torch_dtype(dtype))
    labels = torch.randint(0, c, (n,), generator=generator)
    mask = torch.rand(n, c, generator=generator) < 0.3
    mask[:, 0] = True
    mask[:, 1] = False
    return z, labels, mask, torch.randn(n, c, generator=generator).to(_torch_dtype(dtype))


def _upstream(spec: Any) -> Any:
    import torch

    dtype = torch.float64 if spec.dtype == "float64" else torch.float32
    shape = (spec.rows,) if spec.reduction == "none" else ()
    return torch.ones(shape, dtype=dtype)


@needs_torch_triton
@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES)
def test_entropy_and_hinge_plans_are_the_triton_launches(shape: tuple[int, int], dtype: str):
    from libsdca import _triton_entropy, _triton_hinge

    n, c = shape
    z, labels, _, _ = _inputs(n, c, dtype)
    for reduction in ("none", "mean", "sum"):
        for fp32 in {False} if dtype == "float64" else {True, False}:
            spec = _spec("entropy", (fp32,), "forward", n, c, dtype, reduction)
            with _recorded() as launches:
                out, stats = _triton_entropy.forward(z, labels, reduction, fp32)
            tensors = {"logits": z, "target": labels, "out": out, "stats": stats}
            _assert_same_launches(spec, launches, tensors)
            u = _upstream(spec)
            with _recorded() as launches:
                grad = _triton_entropy.backward(z, labels, stats, u, reduction, fp32)
            tensors = {
                "logits": z,
                "target": labels,
                "stats": stats,
                "upstream": u,
                "gradient": grad,
            }
            _assert_same_launches(spec.at("backward"), launches, tensors)
        for smooth, gamma, fp32 in (
            (False, 1.0, False),
            (True, 0.5, False),
            (False, 1.0, True),
            (True, 0.5, True),
        ):
            spec = _spec("hinge", (smooth, gamma, fp32), "forward", n, c, dtype, reduction)
            with _recorded() as launches:
                out, stats = _triton_hinge.forward(z, labels, reduction, smooth, gamma, fp32)
            tensors = {"logits": z, "target": labels, "out": out, "stats": stats}
            _assert_same_launches(spec, launches, tensors)
            _params_match(spec, [launches[0][2][3]])
            u = _upstream(spec)
            with _recorded() as launches:
                grad = _triton_hinge.backward(z, labels, stats, u, reduction, smooth, gamma, fp32)
            tensors = {
                "logits": z,
                "target": labels,
                "stats": stats,
                "upstream": u,
                "gradient": grad,
            }
            _assert_same_launches(spec.at("backward"), launches, tensors)


@needs_torch_triton
@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES)
def test_multilabel_and_topk_plans_are_the_triton_launches(shape: tuple[int, int], dtype: str):
    from libsdca import _triton_multilabel, _triton_topk

    n, c = shape
    z, labels, mask, _ = _inputs(n, c, dtype)
    for reduction in ("none", "mean"):
        for objective, gamma in (
            ("cross_entropy", 1.0),
            ("multilabel_hinge", 1.0),
            ("multilabel_hinge_smooth", 0.5),
        ):
            fp32 = objective == "cross_entropy" and dtype != "float64"
            spec = _spec("multilabel", (objective, fp32, gamma), "forward", n, c, dtype, reduction)
            with _recorded() as launches:
                out, stats = _triton_multilabel.forward(z, mask, objective, reduction, gamma, fp32)
            tensors = {"logits": z, "target": mask, "out": out, "stats": stats}
            _assert_same_launches(spec, launches, tensors)
            u = _upstream(spec)
            with _recorded() as launches:
                grad = _triton_multilabel.backward(
                    z, mask, stats, u, objective, reduction, gamma, fp32
                )
            tensors = {"logits": z, "target": mask, "stats": stats, "upstream": u, "gradient": grad}
            _assert_same_launches(spec.at("backward"), launches, tensors)
        if c < 10:
            continue
        for objective, gamma in (
            ("topk_cross_entropy", 1.0),
            ("topk_hinge_alpha", 1.0),
            ("topk_hinge_beta_smooth", 0.5),
        ):
            fp32 = objective == "topk_cross_entropy" and dtype != "float64"
            spec = _spec("topk", (objective, 5, gamma, fp32), "forward", n, c, dtype, reduction)
            with _recorded() as launches:
                out, stats = _triton_topk.forward(z, labels, reduction, objective, 5, gamma, fp32)
            tensors = {"logits": z, "target": labels, "out": out, "stats": stats}
            _assert_same_launches(spec, launches, tensors)
            u = _upstream(spec)
            with _recorded() as launches:
                grad = _triton_topk.backward(
                    z, labels, stats, u, reduction, objective, 5, gamma, fp32
                )
            tensors = {
                "logits": z,
                "target": labels,
                "stats": stats,
                "upstream": u,
                "gradient": grad,
            }
            _assert_same_launches(spec.at("backward"), launches, tensors)


@needs_torch_triton
@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES)
def test_distillation_and_proximal_plans_are_the_triton_launches(
    shape: tuple[int, int], dtype: str
):
    from libsdca import _triton_distillation, _triton_prox

    n, c = shape
    z, labels, mask, teacher = _inputs(n, c, dtype)
    for temperature, double in ((2.0, False), (0.5, True), (1e-39, False)):
        if dtype == "float64" or double:
            work = "float64"
        else:
            work = "float32" if float(np.finfo(np.float32).tiny) <= temperature else "float64"
        for reduction, values_only in (("none", False), ("mean", True), ("sum", False)):
            stage = "values" if values_only else "forward"
            spec = _spec("distillation", (temperature, work), stage, n, c, dtype, reduction)
            with _recorded() as launches:
                out, stats = _triton_distillation.forward(
                    z, teacher, temperature, double, reduction, values_only
                )
            tensors = {"logits": z, "target": teacher, "out": out, "stats": stats}
            _assert_same_launches(spec, launches, tensors)
            _params_match(spec, [launches[0][2][2]])
            u = _upstream(spec)
            with _recorded() as launches:
                grad = _triton_distillation.backward(
                    z, teacher, stats, u, temperature, double, reduction
                )
            tensors = {
                "logits": z,
                "target": teacher,
                "stats": stats,
                "upstream": u,
                "gradient": grad,
            }
            _assert_same_launches(spec.at("backward"), launches, tensors)
        alpha = 0.7
        tiny = float(np.finfo(np.float32).tiny)
        fp32 = not double and dtype != "float64" and tiny <= temperature <= 1 / tiny
        spec = _spec("prox_distillation", (temperature, alpha, fp32), "rows", n, c, dtype, "none")
        with _recorded() as launches:
            loss, grad = _triton_distillation.proximal_distillation(
                z, teacher, temperature, alpha, double
            )
        tensors = {"logits": z, "target": teacher, "loss": loss, "gradient": grad}
        _assert_same_launches(spec, launches, tensors)
        _params_match(spec, [launches[0][2][2], launches[0][2][3]])
    for alpha, fp32 in ((0.7, False), (0.7, True)):
        spec = _spec("prox", (alpha, fp32), "rows", n, c, dtype, "none")
        with _recorded() as launches:
            loss, grad = _triton_prox.proximal_entropy(z, mask, alpha, fp32)
        tensors = {"logits": z, "target": mask, "loss": loss, "gradient": grad}
        _assert_same_launches(spec, launches, tensors)
        _params_match(spec, [launches[0][2][2]])


# Compiling for sm_80 without a GPU, as Triton does for the PyTorch path.


@needs_triton
@pytest.mark.slow
def test_kernels_compile_for_sm80_without_a_gpu():
    specs = [
        _spec("entropy", (True,), "forward", 20000, 1000, "bfloat16", "mean"),
        _spec("entropy", (False,), "forward", 8, 40000, "float32", "sum"),
        _spec("hinge", (True, 0.5, True), "forward", 8, 100, "float32", "none"),
        _spec("multilabel", ("cross_entropy", True, 1.0), "forward", 8, 32000, "bfloat16", "mean"),
        _spec(
            "multilabel",
            ("multilabel_hinge_smooth", False, 0.5),
            "forward",
            8,
            100,
            "float32",
            "mean",
        ),
        _spec("topk", ("topk_cross_entropy", 5, 1.0, True), "forward", 8, 1000, "float32", "mean"),
        _spec("distillation", (2.0, "float32"), "forward", 8, 32000, "float32", "mean"),
        _spec("prox", (0.5, True), "rows", 8, 100, "float32", "none"),
        _spec("prox_distillation", (2.0, 0.5, True), "rows", 8, 5000, "bfloat16", "none"),
    ]
    for spec in specs:
        stages = ("rows",) if spec.stage == "rows" else ("forward", "backward")
        for stage in stages:
            program = _jax_cuda.program(spec.at(stage), 80)
            kernels = program.kernels.reshape(-1, 7)
            assert len(program.binaries) == int(kernels[-1, 3] + kernels[-1, 4])
            plan = _jax_cuda.plan(spec.at(stage))
            # Per launch: kernel, grid, argument count, then (kind, a, b) per argument.
            at, launches = 0, 0
            while at < len(program.program):
                count = int(program.program[at + 4])
                kinds = program.program[at + 5 : at + 5 + 3 * count : 3]
                assert list(kinds[-2:]) == [_jax_cuda._NULL] * 2
                at += 5 + 3 * count
                launches += 1
            assert launches == len(plan.launches)


@needs_torch_triton
@pytest.mark.slow
def test_compiled_kernels_are_the_torch_kernels():
    """The cubin of a launch is the one Triton compiles from torch tensors' dtypes."""
    import torch
    from triton.runtime.jit import MockTensor

    spec = _spec("entropy", (True,), "forward", 64, 1000, "bfloat16", "mean")
    for launch in _jax_cuda.plan(spec).launches:
        kernel, _ = _jax_cuda.compile_launch(launch, 80)
        args = [
            MockTensor(getattr(torch, a.dtype)) if isinstance(a, _jax_cuda.Ref) else a
            for a in launch.args
        ]
        with _jax_cuda._compiling_for(80):
            compiled = launch.kernel.warmup(
                *args, grid=(1,), num_warps=launch.warps, **dict(launch.options)
            )
        assert compiled.asm["cubin"] == kernel.binary


# jit, grad and vmap of the calls, with the kernels emulated by their references.


def _emulated(*args: Any, spec: Any, batch: int) -> list[jax.Array]:
    """The results of a call in the kernels' layout from its reference (zero statistics)."""
    shared = _jax_cuda.arrays(spec)
    reference = spec.reference

    def one(*a: Any) -> list[jax.Array]:
        target = _jax_cuda._reference_target(a[1], spec)
        if spec.stage == "backward":

            def value(z: jax.Array) -> jax.Array:
                return reference(z, target, *spec.reference_args)

            ct = a[3].astype(_jax_cuda._out_dtype(spec.dtype))
            if spec.gradient is not None:
                return [spec.gradient(a[0], target, ct, *spec.reference_args)]
            return [jax.vjp(value, a[0])[1](ct)[0]]
        if spec.stage == "rows":
            values, gradient = reference(a[0], target, *spec.reference_args)
            dtypes = [d for _, d in _jax_cuda.plan(spec).results]
            return [values.astype(dtypes[0]), gradient.astype(dtypes[1])]
        out = reference(a[0], target, *spec.reference_args)
        shape, dtype = _jax_cuda.plan(spec).results[1]
        return [out, jnp.zeros(shape, dtype)]

    if batch == 1:
        return one(*args)
    unbatched = _jax_cuda._unbatched(args, spec, batch)
    axes = [0] * shared + [None] * (len(args) - shared)
    return _jax_cuda._flattened(jax.vmap(one, in_axes=axes)(*unbatched), spec, batch)


# Without Triton: the constants of the kernel modules that the result shapes read.
FAKE_MODULES = {
    "reduce": types.SimpleNamespace(BLOCK=2048),
    "entropy": types.SimpleNamespace(_LAYOUT=(3, 4, 5, 6)),
    "hinge": types.SimpleNamespace(_LAYOUT=(0, -1, 3, 4)),
    "topk": types.SimpleNamespace(_LAYOUT=(0, -1, 3, 4)),
    "multilabel": types.SimpleNamespace(
        _LAYOUTS={
            "cross_entropy": (0, -1, 5, 6),
            "multilabel_hinge": (0, -1, 4, 5),
            "multilabel_hinge_smooth": (0, -1, 8, 9),
        }
    ),
    "distillation": types.SimpleNamespace(_LAYOUT=(0, -1, 1, 13)),
}


def _fake_plan(spec: Any) -> Any:
    if spec.stage == "rows":
        out = _jax_cuda._out_dtype(spec.dtype)
        return _jax_cuda.Plan((), (), (((spec.rows,), out), ((spec.rows, spec.classes), out)))
    return _jax_cuda.Plan((), (), _jax_cuda._results(spec))


@pytest.fixture
def emulated_kernels() -> Iterator[None]:
    """The kernels' lowering on the CPU emulated by their references (CPU backend only: on a
    GPU the tests run the kernels)."""
    if jax.default_backend() != "cpu":
        pytest.skip("emulates the kernels on the CPU backend")
    rule = mlir.lower_fun(_emulated, multiple_results=True)
    mlir.register_lowering(_jax_cuda.cuda_p, rule, platform="cpu")
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(_jax_cuda, "_ready", return_value=True))
        stack.enter_context(mock.patch.object(_jax_cuda, "on_gpu", return_value=True))
        if not TRITON:
            stack.enter_context(mock.patch.object(_jax_cuda, "_modules", return_value=FAKE_MODULES))
            stack.enter_context(mock.patch.object(_jax_cuda, "plan", side_effect=_fake_plan))
        yield
    mlir.register_lowering(_jax_cuda.cuda_p, _jax_cuda._elsewhere, platform="cpu")


def _close(a: Any, b: Any) -> None:
    np.testing.assert_allclose(
        np.asarray(a, np.float64), np.asarray(b, np.float64), rtol=2e-6, atol=1e-7
    )


EMULATED = [
    ("cross_entropy", "labels"),
    ("multilabel_cross_entropy", "mask"),
    ("distillation", "teacher"),
    ("proximal_distillation", "teacher"),
]


def _loss(name: str, reduction: Reduction) -> Callable[[jax.Array, jax.Array], jax.Array]:
    if name == "distillation":
        return lambda z, t: sdca_jax.distillation_loss(z, t, temperature=2.0, reduction=reduction)
    if name == "proximal_distillation":
        return lambda z, t: sdca_jax.proximal_distillation_loss(
            z, t, temperature=2.0, alpha=0.5, reduction=reduction
        )
    return lambda z, t: named_losses.loss(sdca_jax, name, z, t, reduction=reduction)


def _targets(kind: str, z: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(3)
    if kind == "labels":
        return rng.integers(0, z.shape[-1], z.shape[:-1]).astype(np.int32)
    if kind == "mask":
        mask = rng.random(z.shape) < 0.3
        mask[..., 0] = True
        return mask
    return (2 * rng.normal(size=z.shape)).astype(z.dtype)


@pytest.mark.usefixtures("emulated_kernels")
@pytest.mark.parametrize("name, kind", EMULATED)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_calls_jit_grad_and_vmap_as_their_references(name: str, kind: str, reduction: Reduction):
    """With the kernels emulated by their references, values and gradients under jit, grad and
    vmap (folded or looped) are the references' (_jax_cuda wires the calls, not the math)."""
    rng = np.random.default_rng(2)
    z = (3 * rng.normal(size=(4, 5, 9))).astype(np.float32)
    t = _targets(kind, z)
    fused = _loss(name, reduction)

    def ordinary(z: Any, t: Any) -> jax.Array:
        with mock.patch.object(_jax_cuda, "on_gpu", return_value=False):
            return fused(z, t)

    for batched in (False, True):
        f = jax.vmap(fused) if batched else fused
        g = jax.vmap(ordinary) if batched else ordinary

        def total(fn: Callable[..., jax.Array], z: jax.Array) -> jax.Array:
            return (fn(z, t) * jnp.arange(1, 1 + fn(z, t).size).reshape(fn(z, t).shape)).sum()

        _close(jax.jit(f)(z, t), g(z, t))
        _close(
            jax.jit(jax.grad(functools.partial(total, f)))(z),
            jax.grad(functools.partial(total, g))(z),
        )


@pytest.mark.usefixtures("emulated_kernels")
def test_entropy_keeps_higher_derivatives():
    rng = np.random.default_rng(4)
    z = (2 * rng.normal(size=(3, 6))).astype(np.float32)
    labels = np.asarray([0, 2, 5], np.int32)
    fused = functools.partial(sdca_jax.cross_entropy, target=labels)

    def ordinary(z: jax.Array) -> jax.Array:
        with mock.patch.object(_jax_cuda, "on_gpu", return_value=False):
            return sdca_jax.cross_entropy(z, labels)

    _close(jax.hessian(fused)(z), jax.hessian(ordinary)(z))
    tangent = jnp.ones_like(z)
    _close(jax.jvp(fused, (z,), (tangent,))[1], jax.jvp(ordinary, (z,), (tangent,))[1])
    _close(
        jax.grad(lambda z: jax.grad(fused)(z).sum())(z),
        jax.grad(lambda z: jax.grad(ordinary)(z).sum())(z),
    )


@pytest.mark.usefixtures("emulated_kernels")
@pytest.mark.parametrize("reduction", ["none", "mean"])
def test_nested_partial_and_single_vmaps(reduction: Reduction):
    """vmap of vmap (folded twice, or looped twice), over the target alone, and of size 1."""
    rng = np.random.default_rng(6)
    z = (2 * rng.normal(size=(3, 2, 4, 7))).astype(np.float32)
    labels = rng.integers(0, 7, (3, 2, 4)).astype(np.int32)
    fused = functools.partial(sdca_jax.cross_entropy, reduction=reduction)

    def ordinary(z: Any, t: Any) -> jax.Array:
        with mock.patch.object(_jax_cuda, "on_gpu", return_value=False):
            return fused(z, t)

    def both(transform: Callable[..., Any], *args: Any) -> None:
        _close(jax.jit(transform(fused))(*args), transform(ordinary)(*args))
        grad = jax.grad(lambda z: transform(fused)(z, *args[1:]).sum())
        expected = jax.grad(lambda z: transform(ordinary)(z, *args[1:]).sum())
        _close(jax.jit(grad)(args[0]), expected(args[0]))

    both(lambda f: jax.vmap(jax.vmap(f)), z, labels)
    both(lambda f: jax.vmap(f, in_axes=(None, 0)), z[0, 0], labels[:, 0])
    both(lambda f: jax.vmap(f), z[:1, 0], labels[:1, 0])


@pytest.mark.usefixtures("emulated_kernels")
@pytest.mark.parametrize("name", ["distillation", "proximal_distillation"])
def test_distillation_keeps_second_derivatives(name: str):
    rng = np.random.default_rng(8)
    z = (2 * rng.normal(size=(3, 6))).astype(np.float32)
    t = (2 * rng.normal(size=(3, 6))).astype(np.float32)
    loss = _loss(name, "sum")

    def fused(z: Any) -> jax.Array:
        return loss(z, jnp.asarray(t))

    def ordinary(z: Any) -> jax.Array:
        with mock.patch.object(_jax_cuda, "on_gpu", return_value=False):
            return fused(z)

    _close(jax.hessian(fused)(z), jax.hessian(ordinary)(z))
    if name == "distillation":
        # The jax.numpy proximal solve is a while loop, which reverse mode cannot transpose.
        _close(
            jax.grad(lambda z: jax.grad(fused)(z).sum())(z),
            jax.grad(lambda z: jax.grad(ordinary)(z).sum())(z),
        )
    with pytest.raises(TypeError):
        jax.jvp(fused, (z,), (z,))


# On a CUDA GPU: the bits of the PyTorch CUDA path.


def _gpu_cases() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("cross_entropy", {}),
        ("cross_entropy", {"precision": "float64"}),
        ("multilabel_cross_entropy", {}),
        ("topk_cross_entropy", {"k": 5}),
        ("multiclass_hinge", {}),
        ("multiclass_hinge", {"gamma": 0.5}),
        ("topk_hinge_alpha", {"k": 3}),
        ("topk_hinge_beta", {"k": 3, "gamma": 0.5}),
        ("multilabel_hinge", {}),
        ("multilabel_hinge", {"gamma": 0.5}),
        ("proximal_cross_entropy", {"alpha": 0.5}),
        ("distillation", {"temperature": 2.0}),
        ("proximal_distillation", {"temperature": 2.0, "alpha": 0.5}),
    ]


@pytest.fixture
def gpu_work_precision(request: pytest.FixtureRequest):
    # Explicit float64 follows the public JAX contract; leave default cases at x64 off.
    with ExitStack() as scopes:
        if request.node.callspec.params["case"][1].get("precision") == "float64":
            scopes.enter_context(x64())
        yield


def _gpu_inputs(
    name: str, n: int, c: int, dtype: str, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = (3 * rng.normal(size=(n, c))).astype(np.float32)
    z[0, : min(c, 4)] = [100.0, 0.0, -50.0, 3.0][: min(c, 4)]
    if name.startswith("multilabel"):
        mask = rng.random((n, c)) < 0.05
        mask[:, 0], mask[:, 1] = True, False
        return z, mask
    if "distillation" in name:
        return z, (z + rng.normal(size=(n, c)).astype(np.float32) * 0.1)
    return z, rng.integers(0, c, n).astype(np.int64)


def _torch_loss(name: str, z: Any, t: Any, reduction: Reduction, options: dict[str, Any]) -> Any:
    import libsdca.torch as sdca_torch

    if name == "distillation":
        return sdca_torch.distillation_loss(
            z, t, temperature=options["temperature"], reduction=reduction
        )
    if name == "proximal_distillation":
        return sdca_torch.proximal_distillation_loss(
            z, t, temperature=options["temperature"], alpha=options["alpha"], reduction=reduction
        )
    return named_losses.loss(sdca_torch, name, z, t, reduction=reduction, **options)


def _jax_loss(name: str, z: Any, t: Any, reduction: Reduction, options: dict[str, Any]) -> Any:
    if name == "distillation":
        return sdca_jax.distillation_loss(
            z, t, temperature=options["temperature"], reduction=reduction
        )
    if name == "proximal_distillation":
        return sdca_jax.proximal_distillation_loss(
            z, t, temperature=options["temperature"], alpha=options["alpha"], reduction=reduction
        )
    return named_losses.loss(sdca_jax, name, z, t, reduction=reduction, **options)


def _bits(x: Any) -> bytes:
    return np.ascontiguousarray(np.asarray(x)).tobytes()


@pytest.mark.usefixtures("gpu_work_precision")
@needs_gpu
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("shape", [(64, 10), (300, 1000), (17, 40000), (20000, 50)], ids=str)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("case", _gpu_cases(), ids=lambda c: c[0] + str(sorted(c[1].items())))
def test_values_and_gradients_are_the_torch_cuda_bits(
    case: tuple[str, dict[str, Any]], reduction: Reduction, shape: tuple[int, int], dtype: str
):
    torch = pytest.importorskip("torch")
    name, options = case
    n, c = shape
    z, t = _gpu_inputs(name, n, c, dtype)
    if name.startswith("topk") and options.get("k", 1) >= c - 1:
        pytest.skip("k >= classes - 1")
    tz = torch.tensor(z, device="cuda").to(getattr(torch, dtype)).requires_grad_(True)
    tt = torch.tensor(t, device="cuda")
    if "distillation" in name:
        tt = tt.to(getattr(torch, dtype))
    value = _torch_loss(name, tz, tt, reduction, options)
    upstream = (
        torch.linspace(0.5, 1.5, value.numel(), device="cuda").reshape(value.shape).to(value.dtype)
    )
    (grad,) = torch.autograd.grad(value, tz, upstream)
    jz = jnp.asarray(tz.detach().float().cpu().numpy()).astype(dtype)
    jt = jnp.asarray(t).astype(dtype) if "distillation" in name else jnp.asarray(t)
    f = jax.jit(lambda z: _jax_loss(name, z, jt, reduction, options))
    j_value, vjp = jax.vjp(f, jz)
    (j_grad,) = vjp(jnp.asarray(upstream.cpu().numpy()))
    exact = name not in ("proximal_cross_entropy", "proximal_distillation") or reduction == "none"
    if exact:
        assert _bits(j_value) == _bits(value.detach().cpu().numpy())
    else:
        np.testing.assert_allclose(np.asarray(j_value), value.detach().cpu().numpy(), rtol=1e-6)
    assert _bits(j_grad.astype(jnp.float32)) == _bits(grad.float().cpu().numpy())


@pytest.mark.usefixtures("gpu_work_precision")
@needs_gpu
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("case", _gpu_cases(), ids=lambda c: c[0] + str(sorted(c[1].items())))
def test_vmap_is_the_unbatched_bits(case: tuple[str, dict[str, Any]], reduction: Reduction):
    name, options = case
    z, t = _gpu_inputs(name, 6 * 7, 33, "float32", seed=5)
    z = z.reshape(6, 7, 33)
    t = t.reshape(6, 7, *t.shape[1:])
    f = functools.partial(_jax_loss, name, reduction=reduction, options=options)
    values = jax.jit(jax.vmap(f))(z, t)
    grads = jax.jit(jax.vmap(jax.grad(lambda z, t: f(z, t).sum())))(z, t)
    for b in range(6):
        expected = f(z[b], t[b])
        if name.startswith("proximal_") and reduction != "none":
            # Row kernels are identical. XLA may reassociate their outer reduction
            # under jit/vmap, just as PyTorch uses a different reduction tree.
            np.testing.assert_allclose(
                values[b], expected, rtol=2 * np.finfo(np.float32).eps, atol=0
            )
        else:
            assert _bits(values[b]) == _bits(expected)
        assert _bits(grads[b]) == _bits(jax.grad(lambda z: f(z, t[b]).sum())(z[b]))


@needs_gpu
def test_export_round_trip():
    pytest.importorskip("flatbuffers", reason="jax.export serialization needs flatbuffers")
    from jax import export

    z, t = _gpu_inputs("cross_entropy", 64, 1000, "float32")
    f = jax.jit(lambda z: sdca_jax.cross_entropy(z, t))
    exported = export.export(
        f, disabled_checks=[export.DisabledSafetyCheck.custom_call(_jax_cuda.TARGET)]
    )(z)
    restored = export.deserialize(exported.serialize())
    assert _bits(restored.call(z)) == _bits(f(z))
    assert _jax_cuda.TARGET in exported.mlir_module()


@needs_gpu
def test_off_is_the_jax_numpy_path():
    from child_process import run_python

    code = (
        "import numpy as np, jax.numpy as jnp\n"
        "from libsdca import _jax_cuda\n"
        "from libsdca.jax import cross_entropy\n"
        "z = jnp.asarray(np.random.default_rng(0).normal(size=(8, 100)), jnp.float32)\n"
        "print(float(cross_entropy(z, jnp.arange(8))), _jax_cuda.why_not() or 'kernels')\n"
    )
    outputs = {}
    for value in ("on", "off"):
        result = run_python(
            "-c", code, capture_output=True, text=True, env={**os.environ, _jax_cuda.ENV: value}
        )
        assert result.returncode == 0, result.stderr
        outputs[value] = result.stdout.split()
    assert outputs["on"][1] == "kernels" and outputs["off"][1] == "SDCA_JAX_CUDA=off"
    assert math.isclose(float(outputs["on"][0]), float(outputs["off"][0]), rel_tol=1e-6)


@pytest.mark.usefixtures("emulated_kernels")
def test_eager_calls_reuse_the_compiled_primitive():
    jax.clear_caches()
    _jax_cuda._bound.cache_clear()
    z = jnp.asarray([[1.0, 2.0, -1.0]], dtype=jnp.float32)
    labels = jnp.asarray([0], dtype=jnp.int32)
    loss = functools.partial(sdca_jax.cross_entropy, target=labels)
    jax.value_and_grad(loss)(z)
    before = _jax_cuda._bound.cache_info()
    jax.value_and_grad(loss)(z)
    after = _jax_cuda._bound.cache_info()
    assert after.misses == before.misses
    assert after.hits > before.hits


def test_explicit_cpu_default_device_keeps_the_cpu_path():
    with (
        jax.default_device(jax.devices("cpu")[0]),
        mock.patch.object(jax, "default_backend", return_value="gpu"),
        mock.patch.object(_jax_cuda, "architecture", return_value=80),
    ):
        z = jnp.ones((2, 3), dtype=jnp.float32)
        assert not _jax_cuda.on_gpu(z)
        assert not bool(jax.jit(lambda x: jnp.asarray(_jax_cuda.on_gpu(x)))(z))


@pytest.mark.parametrize("case", _gpu_cases(), ids=lambda c: c[0] + str(sorted(c[1].items())))
def test_cuda_trace_can_lower_for_committed_cpu_arguments(case: tuple[str, dict[str, Any]]):
    name, options = case
    cpu = jax.devices("cpu")[0]
    source, target = _gpu_inputs(name, 7, 33, "float32", seed=5)
    z, t = jax.device_put(source, cpu), jax.device_put(target, cpu)
    f = functools.partial(_jax_loss, name, reduction="mean", options=options)
    with x64(), ExitStack() as patches:
        patches.enter_context(mock.patch.object(_jax_cuda, "_ready", return_value=True))
        patches.enter_context(mock.patch.object(_jax_cuda, "plan", side_effect=_fake_plan))
        patches.enter_context(mock.patch.object(_jax_cuda, "_modules", return_value=FAKE_MODULES))
        with mock.patch.object(_jax_cuda, "on_gpu", return_value=False):
            expected = jax.jit(jax.value_and_grad(f))(z, t)
        with mock.patch.object(_jax_cuda, "on_gpu", return_value=True):
            actual = jax.jit(jax.value_and_grad(lambda z, t: f(z, t)))(z, t)
        for a, b in zip(actual, expected, strict=True):
            _close(a, b)
