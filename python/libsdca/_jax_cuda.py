"""The fused Triton kernels of libsdca.torch for libsdca.jax on CUDA GPUs.

A loss lowers to one XLA FFI call for its forward pass (the kernel and the deterministic
reduction of _triton_reduce) and one for its backward pass, as the PyTorch CUDA path launches
them through Triton: the same kernels, compiled by Triton for the device's architecture with
the same specialization, grids, warps and buffers. The outer sum/mean of proximal losses
uses each framework's reduction and can differ in its final bits.
The handler (_jax_cuda.cpp) interprets a launch program in the call's attributes, whose kernel
binaries it loads once per context; it calls cuLaunchKernel on XLA's stream with the kernels'
own arguments and null scratch pointers, which _cuda_launch.kernel_gate checks suffices.

SDCA_JAX_CUDA selects the kernels: "auto" (or unset) runs them within the Triton releases the
PyTorch launcher is verified with (_cuda_launch.VERIFIED) where Triton, libcuda.so.1, the
jax.ffi headers and a C++ compiler are available, and otherwise keeps jax.numpy with a
one-time warning (silently outside the verified releases); a failed handler build is
recorded and retried as the other extensions' (_jax_ffi.automatic). "on" (or "1") skips the
version gate and raises where they cannot run; "off" (or "0") keeps jax.numpy. A kernel the
gate rejects keeps jax.numpy for that call, with a one-time warning.

Derivatives: entropy (integer labels and masks) is a custom JVP whose tangent's transpose is
the backward kernel, so reverse mode runs the kernels and higher derivatives differentiate
the jax.numpy reference; distillation runs the kernels in its custom VJP, whose second
derivatives differentiate the reference; the other losses are first-order. Under vmap,
per-example losses fold the batch into rows, and reduced ones run the launch program once per
batch element in the same call.
"""

from __future__ import annotations

import base64
import ctypes
import functools
import hashlib
import math
import struct
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

from . import _cuda_launch, _extensions, _jax_ffi

ENV = "SDCA_JAX_CUDA"
TARGET = "libsdca_cuda_kernels"
LIBRARY = "libsdca_jax_cuda.so"
REDUCTIONS = {"none": 0, "mean": 1, "sum": 2}
FIRST_ORDER = "sdca_loss has no JVP: its derivatives are first-order reverse mode"
# The handler's argument kinds (_jax_cuda.cpp) and Triton's scratch pointers after them.
_BUFFER, _INT32, _INT64, _NULL = 0, 1, 2, 3
_SCRATCH = 2
_ALIGN = 16
_lock = threading.RLock()


def _register(library: Path) -> ctypes.CDLL:
    from jaxlib import xla_client

    handlers = ctypes.CDLL(str(library))
    stages = {
        "initialize": jax.ffi.pycapsule(handlers.sdca_jax_cuda_initialize),
        "execute": jax.ffi.pycapsule(handlers.sdca_jax_cuda_execute),
    }
    traits = xla_client.CustomCallTargetTraits.COMMAND_BUFFER_COMPATIBLE
    jax.ffi.register_ffi_target(TARGET, stages, platform="CUDA", traits=traits)
    return handlers


HANDLERS = _jax_ffi.Handlers(
    env=ENV,
    source=Path(__file__).with_name("_jax_cuda.cpp"),
    library=LIBRARY,
    prefix="jax_cuda_",
    what="the JAX CUDA kernel handler",
    fallback="JAX GPU losses use jax.numpy instead of the fused kernels",
    register=_register,
    prebuilt="_jax_cuda_native",
)


@functools.cache
def register() -> ctypes.CDLL:
    """Load the cached handler (building it first if needed) and register its target, once
    per process; raises RuntimeError naming the missing tool or why the build failed."""
    return _jax_ffi.load(HANDLERS)


def mode() -> str:
    """SDCA_JAX_CUDA as "auto", "on" or "off"; ValueError for other values."""
    return _extensions.mode(ENV)


def missing() -> str | None:
    """What the kernels need here besides the handler's build tools, or None."""
    try:
        import triton
    except ImportError:
        return "Triton is not installed (pip install 'libsdca[jax-cuda]' on Linux, or triton)"
    if not _cuda_launch.kernels_supported(triton.__version__):
        return f"Triton {triton.__version__} predates 3.3, which the fused kernels need"
    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError:
        return "libcuda.so.1 (the NVIDIA driver library) is not found"
    return None


def status() -> tuple[bool, str]:
    """(True, "") where the kernels run, else (False, why not); see the module docstring.
    Decides once per process and setting of SDCA_JAX_CUDA (decide.cache_clear() decides
    again), so a test or benchmark may switch it between calls; raises RuntimeError only for
    SDCA_JAX_CUDA=on."""
    return decide(mode())


@functools.cache
def decide(selected: str) -> tuple[bool, str]:
    """status() for a setting of SDCA_JAX_CUDA."""
    if selected == "off":
        return False, f"{ENV}=off"
    reason = missing()
    if selected == "on":
        if reason is not None:
            raise RuntimeError(f"libsdca: the fused JAX GPU kernels cannot run here: {reason}")
        register()
        return True, ""
    if reason is not None:
        _extensions.warn(
            f"libsdca: {HANDLERS.fallback}: {reason}. Install it, or set {ENV}=off to silence this."
        )
        return False, reason
    import triton

    if _cuda_launch.version_gate(triton.__version__) is not None:
        low, high = (".".join(map(str, v)) for v in _cuda_launch.VERIFIED)
        return (
            False,
            f"Triton {triton.__version__} is outside the verified {low}-{high} ({ENV}=on tries it)",
        )
    return _jax_ffi.automatic(HANDLERS, register)


def why_not() -> str:
    """Why JAX GPU losses keep jax.numpy ("" if the kernels run)."""
    return status()[1]


# Devices.


def _capability(device: Any) -> int | None:
    """The CUDA architecture of a device as Triton numbers it (80 for 8.0), else None."""
    value = getattr(device, "compute_capability", None)
    try:
        major, minor = str(value).split(".")[:2]
        return int(major) * 10 + int(minor)
    except ValueError:
        return None


@functools.cache
def architecture() -> int | None:
    """The architecture of the local CUDA GPUs (one for all), else None."""
    try:
        devices = jax.local_devices(backend="gpu")
    except RuntimeError:
        return None
    found = {_capability(d) for d in devices}
    return found.pop() if len(found) == 1 and None not in found else None


def on_gpu(x: jax.Array) -> bool:
    """Whether x lives (traced: the default backend computes) on the local CUDA GPUs."""
    if isinstance(x, jax.core.Tracer):
        device = jax.config.jax_default_device
        if device is not None:
            return _capability(device) is not None and architecture() is not None
        return jax.default_backend() == "gpu" and architecture() is not None
    devices = x.devices()
    return architecture() is not None and all(_capability(d) is not None for d in devices)


# Compiling the kernels: Triton's warmup with stand-ins for the tensors, for a target.


class _Pointer:
    """A pointer argument as Triton specializes it (as torch tensors of the PyTorch path):
    its element dtype, 16-byte aligned."""

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    @staticmethod
    def data_ptr() -> int:
        return 0


class _Target:
    """What Triton's warmup asks its active driver, for an architecture without a device."""

    def __init__(self, architecture: int) -> None:
        from triton.backends.compiler import GPUTarget

        self.target = GPUTarget("cuda", architecture, 32)
        self.device = -architecture  # a cache key no CUDA device takes

    def get_current_device(self) -> int:
        return self.device

    def get_current_stream(self, device: int | None = None) -> int:
        return 0

    def get_current_target(self) -> Any:
        return self.target


@contextmanager
def _compiling_for(architecture: int) -> Any:
    """Triton's driver answering for the architecture (warmup compiles, never launches)."""
    from triton.runtime.driver import driver as config

    slot = "_active" if hasattr(config, "_active") else "active"
    with _lock:
        previous = getattr(config, slot)
        config.set_active(_Target(architecture))
        try:
            yield
        finally:
            setattr(config, slot, previous)


@dataclass(frozen=True)
class Ref:
    """A buffer argument: the call's operand (then result) index and the pointer's dtype."""

    index: int
    dtype: str


@dataclass(frozen=True)
class Launch:
    """One launch: the jit function, its grid, its arguments in order (Ref for buffers,
    constexprs included) and its options (warps; constexprs and compiler options by name)."""

    kernel: Any
    grid: tuple[int, ...]
    args: tuple[Any, ...]
    warps: int
    options: tuple[tuple[str, Any], ...] = ()


class Kernel(NamedTuple):
    """A compiled kernel: its cubin, name, warps, shared bytes and runtime parameter types."""

    binary: bytes
    name: str
    warps: int
    shared: int
    params: tuple[str, ...]


class Unsupported(Exception):
    """A kernel the launch gate rejects (its call keeps jax.numpy)."""


_SIZES = {"i32": 4, "u32": 4, "fp32": 4, "i64": 8, "u64": 8, "fp64": 8}


def _word(value: Any, kind: str) -> int:
    """A scalar argument as the integer of its bits (floats) or its value."""
    if kind == "fp32":
        return struct.unpack("<i", struct.pack("<f", value))[0]
    if kind == "fp64":
        return struct.unpack("<q", struct.pack("<d", value))[0]
    return int(value)


def _size(kind: str) -> int:
    if kind.startswith("*"):
        return _cuda_launch.POINTER
    if kind not in _SIZES:
        raise Unsupported(f"a kernel takes a {kind} argument")
    return _SIZES[kind]


@functools.lru_cache(maxsize=256)
def compile_launch(launch: Launch, architecture: int) -> tuple[Kernel, tuple[Any, ...]]:
    """The compiled kernel of a launch and its runtime arguments (the arguments that are
    not constexpr), checked against the launch gate."""
    args = [_Pointer(a.dtype) if isinstance(a, Ref) else a for a in launch.args]
    options = dict(launch.options)
    with _compiling_for(architecture):
        compiled = launch.kernel.warmup(*args, grid=(1,), num_warps=launch.warps, **options)
    reason = _cuda_launch.kernel_gate(compiled.metadata, None, ())
    if reason is not None:
        raise Unsupported(reason)
    positional = [p.name for p in launch.kernel.params if p.name not in options]
    bound = dict(zip(positional, launch.args, strict=True))
    bound.update(options)
    runtime = [(name, kind) for name, kind in compiled.src.signature.items() if kind != "constexpr"]
    values = tuple(bound[name] for name, _ in runtime)
    for value, (name, kind) in zip(values, runtime, strict=True):
        pointer = kind.startswith("*")
        if pointer != isinstance(value, Ref) or (not pointer and _size(kind) not in (4, 8)):
            raise Unsupported(f"{compiled.metadata.name} takes {name} as {kind}")
    binary = compiled.asm["cubin"]
    kernel = Kernel(
        binary,
        compiled.metadata.name,
        compiled.metadata.num_warps,
        compiled.metadata.shared,
        tuple(kind for _, kind in runtime),
    )
    return kernel, values


def _identity(kernel: Kernel) -> int:
    digest = hashlib.sha256(kernel.binary + kernel.name.encode()).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


@functools.cache
def _loaded_sizes(identity: int, binary: bytes, name: str) -> list[int] | None:
    """The parameter sizes the driver reports for a kernel, loaded in the primary context of
    the first GPU (None without a GPU, or when the driver cannot report them)."""
    try:
        devices = [d for d in jax.local_devices(backend="gpu") if _capability(d) is not None]
    except RuntimeError:
        devices = []
    if not devices:
        return None
    cuda = ctypes.CDLL("libcuda.so.1")
    _cuda_launch.ensure_context(devices[0].local_hardware_id)
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    if cuda.cuModuleLoadData(ctypes.byref(module), ctypes.c_char_p(binary)) != 0:
        raise Unsupported(f"{name} does not load")
    try:
        if cuda.cuModuleGetFunction(ctypes.byref(function), module, name.encode()) != 0:
            raise Unsupported(f"{name} is not in its binary")
        return _cuda_launch.param_sizes(int(function.value or 0))
    finally:
        cuda.cuModuleUnload(module)


class Program(NamedTuple):
    """The handler's attributes of a plan: kernel table, binaries and launch program."""

    kernels: np.ndarray
    binaries: str
    program: np.ndarray


def encode(launches: Sequence[Launch], architecture: int) -> Program:
    """Compile and encode launches (Unsupported where the gate rejects a kernel)."""
    table: list[int] = []
    blobs: list[bytes] = []
    indices: dict[int, int] = {}
    program: list[int] = []
    offset = 0
    for launch in launches:
        kernel, values = compile_launch(launch, architecture)
        identity = _identity(kernel)
        if identity not in indices:
            sizes = tuple(_size(kind) for kind in kernel.params)
            reason = _cuda_launch.kernel_gate(
                _Metadata(kernel.name), _loaded_sizes(identity, kernel.binary, kernel.name), sizes
            )
            if reason is not None:
                raise Unsupported(reason)
            binary = base64.b64encode(kernel.binary)
            name = kernel.name.encode()
            indices[identity] = len(indices)
            table += [identity, offset, len(binary), offset + len(binary), len(name)]
            table += [kernel.warps, kernel.shared]
            blobs += [binary, name]
            offset += len(binary) + len(name)
        grid = (*launch.grid, 1, 1)[:3]
        program += [indices[identity], *grid, len(values) + _SCRATCH]
        for value, kind in zip(values, kernel.params, strict=True):
            if isinstance(value, Ref):
                program += [_BUFFER, value.index, 0]
            else:
                program += [_INT64 if _size(kind) == 8 else _INT32, _word(value, kind), 0]
        program += [_NULL, 0, 0] * _SCRATCH
    return Program(
        np.asarray(table, np.int64), b"".join(blobs).decode("ascii"), np.asarray(program, np.int64)
    )


class _Metadata(NamedTuple):
    name: str


# Plans: the launches of the PyTorch path's Triton launch, for JAX's buffers.


@dataclass(frozen=True)
class Spec:
    """A fused call: the loss family and its options, the stage, the rows and classes of the
    logits (and their dtype), the reduction and the row statistics' capacity in doubles.

    reference(z, target, *reference_args) computes the same loss (reduced) in jax.numpy, or
    for the "rows" stage its per-row values and gradient; its derivatives are the kernels'
    higher derivatives (None: first-order). gradient(z, target, ct, *reference_args), if
    given, is the reference's gradient scaled by ct, whose derivatives the backward kernel's
    are (else those of the reference's VJP)."""

    family: str
    options: tuple[Any, ...]
    stage: str
    rows: int
    classes: int
    dtype: str
    reduction: str
    stats: int = 0
    reference: Callable[..., Any] | None = None
    reference_args: tuple[Any, ...] = ()
    gradient: Callable[..., Any] | None = None

    def at(self, stage: str) -> Spec:
        return replace(self, stage=stage)


class Plan(NamedTuple):
    """The launches of a stage, the constant operands after the call's arrays (uint32 words)
    and the results (shape, dtype)."""

    launches: tuple[Launch, ...]
    params: tuple[np.ndarray, ...]
    results: tuple[tuple[tuple[int, ...], Any], ...]


def _words(values: Sequence[float], dtype: Any) -> np.ndarray:
    """Device scalars as torch fills them (rounded once into dtype), as uint32 words."""
    return np.asarray(values, dtype).view(np.uint32)


def _out_dtype(dtype: str) -> Any:
    return np.float64 if dtype == "float64" else np.float32


def _modules() -> Any:
    from . import (
        _triton_distillation,
        _triton_entropy,
        _triton_hinge,
        _triton_multilabel,
        _triton_prox,
        _triton_reduce,
        _triton_topk,
    )

    return {
        "entropy": _triton_entropy,
        "hinge": _triton_hinge,
        "multilabel": _triton_multilabel,
        "topk": _triton_topk,
        "prox": _triton_prox,
        "distillation": _triton_distillation,
        "prox_distillation": _triton_distillation,
        "reduce": _triton_reduce,
    }


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _slots(spec: Spec) -> int:
    """The status slot of a family's row statistics (_triton_reduce's layout)."""
    m = _modules()
    if spec.family == "multilabel":
        return m["multilabel"]._LAYOUTS[spec.options[0]][3]
    return m[spec.family]._LAYOUT[3]


def capacity(spec: Spec) -> int:
    """Doubles of the row statistics, with room for every reduction (even, for alignment)."""
    block = _modules()["reduce"].BLOCK
    size = _slots(spec) * spec.rows + 1 + 4 * _cdiv(spec.rows, block)
    return max(spec.stats, size + size % 2)


def _reduce(spec: Spec, s: Ref, out: Ref) -> list[Launch]:
    """_triton_reduce.launch."""
    r = _modules()["reduce"]
    n, code = spec.rows, REDUCTIONS[spec.reduction]
    layout = tuple(_layout(spec))
    if n > r.DIRECT_ROWS:
        return [
            Launch(r.partials, (_cdiv(n, r.BLOCK),), (s, out, n, *layout, code, r.BLOCK), r.WARPS),
            Launch(r.finish, (1,), (s, out, n, *layout, code, r.BLOCK, False), r.WARPS),
        ]
    args = (s, out, n, *layout, code, r.DIRECT_BLOCK, True)
    return [Launch(r.finish, (1,), args, r.DIRECT_WARPS)]


def _layout(spec: Spec) -> tuple[int, ...]:
    m = _modules()
    if spec.family == "multilabel":
        return m["multilabel"]._LAYOUTS[spec.options[0]]
    return m[spec.family]._LAYOUT


def _results(spec: Spec) -> tuple[tuple[tuple[int, ...], Any], ...]:
    n, c = spec.rows, spec.classes
    if spec.stage == "backward":
        return (((n, c), np.dtype(spec.dtype)),)
    out = ((n,) if spec.reduction == "none" else (), _out_dtype(spec.dtype))
    return (out, ((2 * capacity(spec),), np.uint32))


def _entropy(spec: Spec) -> Plan:
    """_triton_entropy.forward and launch_backward."""
    e = _modules()["entropy"]
    n, c, (fp32,) = spec.rows, spec.classes, spec.options
    code, rows = REDUCTIONS[spec.reduction], e.rows_per_cta(c)
    z, y, s = Ref(0, spec.dtype), Ref(1, "int64"), Ref(2, "float64")
    if spec.stage == "backward":
        u, g = Ref(3, np.dtype(_out_dtype(spec.dtype)).name), Ref(4, spec.dtype)
        block, warps, tiles = e._bwd_shape(c, fp32)
        args = (z, y, s, u, g, n, c, 0, code, block, False, False, fp32, tiles, rows)
        grid = (_cdiv(n, rows), _cdiv(c, block * tiles))
        return Plan((Launch(e._backward, grid, args, warps),), (), _results(spec))
    out, s = Ref(2, np.dtype(_out_dtype(spec.dtype)).name), Ref(3, "float64")
    block, warps = e._shape(c, fp32)
    forward = Launch(e._forward, (_cdiv(n, rows),), (z, y, s, n, c, block, fp32, rows), warps)
    return Plan((forward, *_reduce(spec, s, out)), (), _results(spec))


def _hinge(spec: Spec) -> Plan:
    """_triton_hinge.forward and backward."""
    h = _modules()["hinge"]
    n, c, (smooth, gamma, fp32) = spec.rows, spec.classes, spec.options
    rows = h.rows_per_cta(c)
    block, warps = h._shape(c, smooth)
    z, y, s = Ref(0, spec.dtype), Ref(1, "int64"), Ref(2, "float64")
    value = np.dtype(_out_dtype(spec.dtype)).name
    params = (_words([gamma], np.float64),)
    if spec.stage == "backward":
        g, u, grad = Ref(4, "float64"), Ref(3, value), Ref(5, spec.dtype)
        code = REDUCTIONS[spec.reduction]
        args = (z, y, s, u, grad, g, n, c, block, smooth, code, rows, fp32)
        return Plan((Launch(h._backward, (_cdiv(n, rows),), args, warps),), params, _results(spec))
    g, out, s = Ref(2, "float64"), Ref(3, value), Ref(4, "float64")
    args = (z, y, s, g, n, c, block, smooth, rows, fp32)
    forward = Launch(h._forward, (_cdiv(n, rows),), args, warps)
    return Plan((forward, *_reduce(spec, s, out)), params, _results(spec))


def _multilabel(spec: Spec) -> Plan:
    """_triton_multilabel.forward and backward (the mask as uint8 flags)."""
    m = _modules()["multilabel"]
    n, c, (objective, fp32, gamma) = spec.rows, spec.classes, spec.options
    smooth = objective == "multilabel_hinge_smooth"
    params = (_words([gamma], np.float64),) if smooth else ()
    z, f, s = Ref(0, spec.dtype), Ref(1, "uint8"), Ref(2, "float64")
    value = np.dtype(_out_dtype(spec.dtype)).name
    rows = m._rows_per_cta(c, objective)
    if spec.stage == "backward":
        u, gamma_ref = Ref(3, value), Ref(4, "float64")
        grad = Ref(4 + len(params), spec.dtype)
        block, code = m._backward_block(c, objective), REDUCTIONS[spec.reduction]
        grid = (_cdiv(n, rows), _cdiv(c, block))
        if objective == "cross_entropy":
            args: tuple[Any, ...] = (z, f, s, u, grad, n, c, block, code, fp32, rows)
            launch = Launch(m._entropy_backward, grid, args, 4)
        elif objective == "multilabel_hinge":
            launch = Launch(m._hinge_backward, grid, (s, u, grad, n, c, block, code), 4)
        else:
            args = (z, f, s, u, grad, gamma_ref, n, c, block, code)
            launch = Launch(m._smooth_backward, grid, args, 4)
        return Plan((launch,), params, _results(spec))
    gamma_ref = Ref(2, "float64")
    out, s = Ref(2 + len(params), value), Ref(3 + len(params), "float64")
    block, warps = m._shape(c, objective)
    if m._online(c, objective, fp32):
        block, warps = m._ONLINE
        forward = Launch(m._entropy_online, (n,), (z, f, s, n, c, block), warps)
    elif objective == "cross_entropy":
        args = (z, f, s, n, c, block, fp32, rows)
        forward = Launch(m._entropy_forward, (_cdiv(n, rows),), args, warps)
    elif objective == "multilabel_hinge":
        forward = Launch(m._hinge_forward, (n,), (z, f, s, n, c, block), warps)
    else:
        eps = float(np.finfo(np.float64).eps)
        args = (z, f, s, gamma_ref, n, c, block, eps)
        forward = Launch(m._smooth_forward, (n,), args, warps)
    return Plan((forward, *_reduce(spec, s, out)), params, _results(spec))


def _topk(spec: Spec) -> Plan:
    """_triton_topk.forward and backward."""
    t = _modules()["topk"]
    n, c, (objective, k, gamma, fp32) = spec.rows, spec.classes, spec.options
    loss = t.LOSSES[objective]
    block, warps, kb = t._shape(c, k, loss)
    rows = t._rows_per_cta(c, loss, k)
    params = (_words([gamma], np.float64),)
    z, y, s = Ref(0, spec.dtype), Ref(1, "int64"), Ref(2, "float64")
    value = np.dtype(_out_dtype(spec.dtype)).name
    if spec.stage == "backward":
        u, g, grad = Ref(3, value), Ref(4, "float64"), Ref(5, spec.dtype)
        code = REDUCTIONS[spec.reduction]
        args = (z, y, s, u, grad, g, n, c, block, loss, k, code, fp32, rows)
        return Plan((Launch(t._backward, (_cdiv(n, rows),), args, warps),), params, _results(spec))
    g, out, s = Ref(2, "float64"), Ref(3, value), Ref(4, "float64")
    args = (z, y, s, g, n, c, block, loss, k, kb, fp32, rows)
    forward = Launch(t._forward, (_cdiv(n, rows),), args, warps)
    return Plan((forward, *_reduce(spec, s, out)), params, _results(spec))


def _distillation(spec: Spec) -> Plan:
    """_triton_distillation.forward and backward (value_only for the "values" stage)."""
    d = _modules()["distillation"]
    n, c, (temperature, work) = spec.rows, spec.classes, spec.options
    (block, warps), (b_block, b_warps) = d._shapes(c)
    rows, fp32, large = d._rows_per_cta(c), work == "float32", temperature > 1.0
    params = (_words([temperature], work),)
    z, t = Ref(0, spec.dtype), Ref(1, spec.dtype)
    value = np.dtype(_out_dtype(spec.dtype)).name
    options = (("enable_reflect_ftz", False),)
    if d.adaptive(n, c, spec.dtype, fp32, spec.stage == "values"):
        return _distillation_adaptive(spec)
    if spec.stage == "backward":
        temp, s, u, grad = Ref(4, work), Ref(2, "float64"), Ref(3, value), Ref(5, spec.dtype)
        code = REDUCTIONS[spec.reduction]
        args = (z, t, temp, s, u, grad, n, c, b_block, fp32, large, code, rows)
        launch = Launch(d._backward, (_cdiv(n, rows),), args, b_warps, options)
        return Plan((launch,), params, _results(spec))
    temp, out, s = Ref(2, work), Ref(3, value), Ref(4, "float64")
    eps = float(np.finfo(work).eps)
    values_only = spec.stage == "values"
    args = (z, t, temp, s, n, c, block, fp32, large, eps, values_only, rows)
    forward = Launch(d._forward, (_cdiv(n, rows),), args, warps, options)
    return Plan((forward, *_reduce(spec, s, out)), params, _results(spec))


def _distillation_adaptive(spec: Spec) -> Plan:
    """The same ordinary-row and rescue launches used by the Torch CUDA API."""
    d = _modules()["distillation"]
    f = d._fast()
    n, c, (temperature, work) = spec.rows, spec.classes, spec.options
    (block, warps), (b_block, b_warps) = d._shapes(c)
    z, t = Ref(0, spec.dtype), Ref(1, spec.dtype)
    value = np.dtype(_out_dtype(spec.dtype)).name
    width, fast_warps = d.adaptive_shape(c)
    options = (("enable_reflect_ftz", False),)
    params = (_words([temperature], work),)
    if spec.stage == "backward":
        temp, s, u, g = Ref(4, work), Ref(2, "float64"), Ref(3, value), Ref(5, spec.dtype)
        code = REDUCTIONS[spec.reduction]
        fast = Launch(
            f._backward, (n,), (z, t, temp, s, u, g, n, c, code, width), fast_warps, options
        )
        rescue = Launch(
            f._rescue_backward,
            (n,),
            (z, t, temp, s, u, g, n, c, b_block, temperature > 1.0, code),
            b_warps,
            options,
        )
        return Plan((fast, rescue), params, _results(spec))
    temp, out, s = Ref(2, work), Ref(3, value), Ref(4, "float64")
    fast = Launch(f._forward, (n,), (z, t, temp, s, n, c, width), fast_warps, options)
    rescue = Launch(
        f._rescue_forward, (n,), (z, t, temp, s, n, c, block, temperature > 1.0), warps, options
    )
    return Plan((fast, rescue, *_reduce(spec, s, out)), params, _results(spec))


def _prox(spec: Spec) -> Plan:
    """_triton_prox.proximal_entropy: per-row losses (working dtype), gradient and status."""
    p = _modules()["prox"]
    n, c, (alpha, fp32) = spec.rows, spec.classes, spec.options
    work = np.float32 if fp32 else np.float64
    cut, large, steps, res_steps = (-18.0, 1e7, 2, 4) if fp32 else (-36.0, 1e16, 3, 6)
    stream = c > p.REGISTER_CLASSES
    block = p._STREAM_BLOCK if stream else max(16, 1 << (c - 1).bit_length())
    warps = p._STREAM_WARPS if stream else max(1, min(16, block // 64))
    gradient = np.dtype(_out_dtype(spec.dtype))
    loss, grad, status = Ref(3, np.dtype(work).name), Ref(4, gradient.name), Ref(5, "int8")
    args = (Ref(0, spec.dtype), Ref(1, "uint8"), Ref(2, "float64"), loss, grad, status, c)
    options = (
        ("BLOCK", block),
        ("EPS", float(np.finfo(work).eps)),
        ("TINY", float(np.finfo(work).tiny)),
        ("CUT", cut),
        ("LARGE", large),
        ("STEPS", steps),
        ("RES_STEPS", res_steps),
        ("FP32", fp32),
        ("STREAM", stream),
        ("enable_reflect_ftz", False),
    )
    results = (((n,), work), ((n, c), gradient), ((n,), np.int8))
    params = (_words([alpha, math.log(alpha)], np.float64),)
    return Plan((Launch(p._kernel, (n,), args, warps, options),), params, results)


def _prox_distillation(spec: Spec) -> Plan:
    """_triton_distillation.proximal_distillation: per-row losses and gradient."""
    d = _modules()["prox_distillation"]
    n, c, (temperature, alpha, fp32) = spec.rows, spec.classes, spec.options
    work = np.float32 if fp32 else np.float64
    cut, large, steps, res_steps = (-18.0, 1e7, 2, 4) if fp32 else (-36.0, 1e16, 3, 6)
    stream = c > d._CHUNK
    block = d._STREAM_BLOCK if stream else max(16, 1 << (c - 1).bit_length())
    warps = d._STREAM_WARPS if stream else max(1, min(16, block // 64))
    out = np.dtype(_out_dtype(spec.dtype))
    kernel = d._prox_stream_kernel if stream else d._prox_kernel
    args = (
        Ref(0, spec.dtype),
        Ref(1, spec.dtype),
        Ref(2, np.dtype(work).name),
        Ref(3, "float64"),
        Ref(4, out.name),
        Ref(5, out.name),
        n,
        c,
    )
    options = (
        ("B", block),
        ("FP32", fp32),
        ("LARGE_T", temperature > 1.0),
        ("EPS", float(np.finfo(work).eps)),
        ("TINY", float(np.finfo(work).tiny)),
        ("CUT", cut),
        ("LARGE", large),
        ("STEPS", steps),
        ("RES_STEPS", res_steps),
        ("enable_reflect_ftz", False),
    )
    params = (
        _words([temperature, alpha, math.log(alpha)], work),
        _words([temperature, alpha, math.log(alpha)], np.float64),
    )
    results = (((n,), out), ((n, c), out))
    return Plan((Launch(kernel, (n,), args, warps, options),), params, results)


_PLANS: dict[str, Callable[[Spec], Plan]] = {
    "entropy": _entropy,
    "hinge": _hinge,
    "multilabel": _multilabel,
    "topk": _topk,
    "distillation": _distillation,
    "prox": _prox,
    "prox_distillation": _prox_distillation,
}


@functools.lru_cache(maxsize=1024)
def plan(spec: Spec) -> Plan:
    """The launches, constant operands and results of a call."""
    return _PLANS[spec.family](spec)


@functools.lru_cache(maxsize=1024)
def program(spec: Spec, architecture: int) -> Program:
    """The handler's attributes of a call (Unsupported where the gate rejects a kernel)."""
    return encode(plan(spec).launches, architecture)


def arrays(spec: Spec) -> int:
    """The call's array operands before its constant ones: the logits and the target (or
    teacher), then for backward the row statistics and the upstream derivative."""
    return 4 if spec.stage == "backward" else 2


# The primitive: one FFI call. With batch > 1 its arrays and results are (batch, words) padded
# to 16 bytes per element, and the handler runs the launch program once per element.

cuda_p = Primitive("sdca_cuda_kernels")
cuda_p.multiple_results = True


def _padded(count: int, dtype: Any) -> int:
    per = _ALIGN // math.gcd(_ALIGN, np.dtype(dtype).itemsize)
    return _cdiv(max(count, 1), per) * per


def _shapes(spec: Spec, batch: int) -> list[tuple[tuple[int, ...], Any]]:
    results = plan(spec).results
    if batch == 1:
        return list(results)
    return [((batch, _padded(math.prod(shape), dtype)), dtype) for shape, dtype in results]


def _abstract(*args: Any, spec: Spec, batch: int) -> list[jax.core.ShapedArray]:
    return [jax.core.ShapedArray(shape, dtype) for shape, dtype in _shapes(spec, batch)]


@functools.lru_cache(maxsize=1024)
def _bound(spec: Spec, batch: int) -> Any:
    """Reuse the callable identity so eager execution shares compiled executables."""
    return jax.jit(functools.partial(cuda_p.bind, spec=spec, batch=batch))


def _impl(*args: Any, spec: Spec, batch: int) -> Any:
    return _bound(spec, batch)(*args)


_preloaded: set[tuple[int, ...]] = set()


def _preload(attributes: Program) -> None:
    """Load the kernels in each local GPU's primary context, which XLA's streams use, so that
    no call loads them, nor any CUDA graph that captures a call (the handler's initialize
    stage would otherwise load them before XLA records command buffers)."""
    key = tuple(int(i) for i in attributes.kernels[::7])
    if key in _preloaded:
        return
    library = register()
    preload = library.sdca_jax_cuda_preload
    preload.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    cuda = ctypes.CDLL("libcuda.so.1")
    binaries = attributes.binaries.encode()
    kernels = np.ascontiguousarray(attributes.kernels, np.int64)
    error = ctypes.create_string_buffer(512)
    for device in jax.local_devices(backend="gpu"):
        context = _cuda_launch._primary(device.local_hardware_id)
        if cuda.cuCtxPushCurrent_v2(ctypes.c_void_p(context)) != 0:
            raise RuntimeError("libsdca: cuCtxPushCurrent failed")
        try:
            pointer = kernels.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
            code = preload(binaries, len(binaries), pointer, kernels.size, error, len(error))
        finally:
            cuda.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
        if code != 0:
            raise RuntimeError(error.value.decode())
    _preloaded.add(key)


def _lowering(ctx: Any, *operands: Any, spec: Spec, batch: int) -> Any:
    target = architecture()
    if target is None:
        raise NotImplementedError("libsdca's fused kernels lower for one CUDA architecture")
    attributes = program(spec, target)
    _preload(attributes)
    # Per batch element: the arrays and results advance, the constant operands do not.
    constant = range(arrays(spec), len(ctx.avals_in))
    strides = [
        _stride(aval) if batch > 1 and i not in constant else 0
        for i, aval in enumerate([*ctx.avals_in, *ctx.avals_out])
    ]
    rule = jax.ffi.ffi_lowering(TARGET)
    return rule(
        ctx,
        *operands,
        kernels=attributes.kernels,
        binaries=attributes.binaries,
        program=attributes.program,
        batch=np.int64(batch),
        strides=np.asarray(strides, np.int64),
    )


def _stride(aval: Any) -> int:
    return math.prod(aval.shape[1:]) * np.dtype(aval.dtype).itemsize


def _unbatched(args: Sequence[jax.Array], spec: Spec, batch: int) -> list[jax.Array]:
    """The arrays of a batched call as (batch, *unbatched shape)."""
    shapes = _operand_shapes(spec)
    out = []
    for i, a in enumerate(args):
        if i >= arrays(spec) or batch == 1:
            out.append(a)
        else:
            shape, _ = shapes[i]
            out.append(a[:, : math.prod(shape)].reshape(batch, *shape))
    return out


def _operand_shapes(spec: Spec) -> list[tuple[tuple[int, ...], Any]]:
    n, c, dtype = spec.rows, spec.classes, np.dtype(spec.dtype)
    target = _TARGETS[spec.family](n, c, dtype)
    shapes = [((n, c), dtype), target]
    if spec.stage == "backward":
        out = ((n,) if spec.reduction == "none" else (), _out_dtype(spec.dtype))
        shapes += [((2 * capacity(spec),), np.uint32), out]
    return shapes


_TARGETS: dict[str, Callable[[int, int, Any], tuple[tuple[int, ...], Any]]] = {
    "entropy": lambda n, c, dtype: ((n, 2), np.uint32),
    "hinge": lambda n, c, dtype: ((n, 2), np.uint32),
    "topk": lambda n, c, dtype: ((n, 2), np.uint32),
    "multilabel": lambda n, c, dtype: ((n, c), np.bool_),
    "prox": lambda n, c, dtype: ((n, c), np.bool_),
    "distillation": lambda n, c, dtype: ((n, c), dtype),
    "prox_distillation": lambda n, c, dtype: ((n, c), dtype),
}


def _flattened(results: Sequence[jax.Array], spec: Spec, batch: int) -> list[jax.Array]:
    """Unbatched-shape results of a batched call in its (batch, padded words) layout."""
    if batch == 1:
        return list(results)
    out = []
    for r, (shape, dtype) in zip(results, _shapes(spec, batch), strict=False):
        flat = jnp.asarray(r, dtype).reshape(batch, -1)
        out.append(jnp.pad(flat, ((0, 0), (0, shape[1] - flat.shape[1]))))
    return out


def _jvp(primals: Sequence[Any], tangents: Sequence[Any], *, spec: Spec, batch: int) -> Any:
    """The kernels' results with the reference's tangents: of the loss (forward stages; the
    statistics get none), of the gradient (backward: second derivatives) and of the rows
    stage's values and gradient. Losses without a reference are first-order."""
    function = spec.reference
    if function is None:
        raise TypeError(FIRST_ORDER)
    outs = cuda_p.bind(*primals, spec=spec, batch=batch)
    args = _unbatched(primals, spec, batch)
    z_dot = _unbatched([ad.instantiate_zeros(tangents[0])], spec, batch)[0]
    dtypes = [dtype for _, dtype in plan(spec).results]

    def reference(z: jax.Array, ct: jax.Array, words: jax.Array) -> list[jax.Array]:
        target = _reference_target(words, spec)
        if spec.stage == "rows":
            values, gradient = function(z, target, *spec.reference_args)
            return [values.astype(dtypes[0]), gradient.astype(dtypes[1])]
        if spec.stage == "backward":
            ct = ct.astype(_out_dtype(spec.dtype))
            if spec.gradient is not None:
                return [spec.gradient(z, target, ct, *spec.reference_args)]
            vjp = jax.vjp(lambda z: function(z, target, *spec.reference_args), z)[1]
            return [vjp(ct)[0]]
        return [function(z, target, *spec.reference_args).astype(dtypes[0])]

    if spec.stage == "backward":
        ct, ct_dot = args[3], ad.instantiate_zeros(tangents[3])
        if batch > 1:
            ct_dot = ct_dot[:, : math.prod(ct.shape[1:])].reshape(ct.shape)
    else:
        ct, ct_dot = args[0], jnp.zeros_like(args[0])

    def tangent(
        z: jax.Array, ct: jax.Array, words: jax.Array, z_dot: jax.Array, ct_dot: jax.Array
    ) -> list[jax.Array]:
        return jax.jvp(lambda z, ct: reference(z, ct, words), (z, ct), (z_dot, ct_dot))[1]

    if batch == 1:
        dots = tangent(args[0], ct, args[1], z_dot, ct_dot)
    else:
        dots = jax.vmap(tangent)(args[0], ct, args[1], z_dot, ct_dot)
    flat: list[Any] = _flattened(dots, spec, batch)
    if spec.stage in ("forward", "values"):
        flat.append(ad.Zero(jax.core.ShapedArray(outs[1].shape, jax.dtypes.float0)))
    return outs, flat


def _reference_target(target: jax.Array, spec: Spec) -> jax.Array:
    """The reference's target: labels from their int64 words."""
    if _TARGETS[spec.family](1, 1, np.float32)[1] != np.uint32:
        return target
    low, high = target[..., 0], target[..., 1]
    if jax.dtypes.canonicalize_dtype(jnp.int64) == jnp.int64:
        return lax.bitcast_convert_type(jnp.stack([low, high], -1), jnp.int64)
    signed = lax.bitcast_convert_type(low, jnp.int32)
    valid = high == jnp.where(signed < 0, jnp.uint32(0xFFFFFFFF), jnp.uint32(0))
    return jnp.where(valid, signed, -1)


def _batch(args: Sequence[Any], dims: Sequence[Any], *, spec: Spec, batch: int) -> Any:
    """vmap: per-example calls fold the batch into rows; others loop over it in the call."""
    size = next(a.shape[d] for a, d in zip(args, dims, strict=True) if d is not None)
    shared = arrays(spec)
    if any(d is not None for d in dims[shared:]):
        raise NotImplementedError("libsdca's fused kernels batch over arrays, not options")
    moved = [
        batching.bdim_at_front(a, d, size) if i < shared else a
        for i, (a, d) in enumerate(zip(args, dims, strict=True))
    ]
    per_row = spec.reduction == "none" or spec.stage == "rows"
    if size == 1:
        flat = [m[0] if i < shared else m for i, m in enumerate(moved)]
        results = [r[None] for r in cuda_p.bind(*flat, spec=spec, batch=batch)]
    elif batch == 1 and per_row and size * spec.rows <= _cuda_launch.ROWS:
        results = _fold(moved, spec, size)
    elif batch == 1:
        results = _loop(moved, spec, size)
    else:
        flat = [
            m.reshape(size * batch, *m.shape[2:]) if i < shared else m for i, m in enumerate(moved)
        ]
        results = cuda_p.bind(*flat, spec=spec, batch=size * batch)
        results = [r.reshape(size, batch, *r.shape[1:]) for r in results]
    return results, [0] * len(results)


def _fold(moved: Sequence[jax.Array], spec: Spec, size: int) -> list[jax.Array]:
    """One call of size times the rows; its statistics hold each example's, as the results of
    unfolded calls would."""
    shared = arrays(spec)
    stats = 0 if spec.stage == "rows" else size * capacity(spec)
    folded = replace(spec, rows=size * spec.rows, stats=stats)
    flat = [m.reshape(-1, *m.shape[2:]) if i < shared else m for i, m in enumerate(moved)]
    results = cuda_p.bind(*flat, spec=folded, batch=1)
    return [r.reshape(size, -1, *r.shape[1:]) for r in results]


def _loop(moved: Sequence[jax.Array], spec: Spec, size: int) -> list[jax.Array]:
    """One call that runs the launch program per example, on 16-byte aligned elements."""
    flat = list(moved)
    for i, m in enumerate(moved[: arrays(spec)]):
        words = m.reshape(size, -1)
        flat[i] = jnp.pad(words, ((0, 0), (0, _padded(words.shape[1], m.dtype) - words.shape[1])))
    results = cuda_p.bind(*flat, spec=spec, batch=size)
    return [
        r[:, : math.prod(shape)].reshape(size, *shape)
        for r, (shape, _) in zip(results, plan(spec).results, strict=True)
    ]


def _elsewhere(ctx: Any, *operands: Any, spec: Spec, batch: int) -> Any:
    # A trace does not necessarily know its eventual device. In particular, committed
    # CPU arguments can select CPU lowering on a machine whose default backend is CUDA.
    return mlir.lower_fun(
        functools.partial(_fallback, spec=spec, batch=batch), multiple_results=True
    )(ctx, *operands)


def _fallback(*args: Any, spec: Spec, batch: int) -> list[jax.Array]:
    """Reuse existing loss implementations if a CUDA trace ultimately lowers elsewhere."""
    from . import jax as losses

    def value(z: jax.Array, target: jax.Array) -> jax.Array:
        if spec.reference is not None:
            return spec.reference(z, target, *spec.reference_args)
        options: dict[str, Any] = {"backend": "native", "reduction": spec.reduction}
        if spec.family == "hinge":
            smooth, gamma, _ = spec.options
            objective = "multiclass_hinge"
            options["gamma"] = gamma if smooth else 0.0
        elif spec.family == "topk":
            objective, k, gamma, _ = spec.options
            options.update(k=k, gamma=gamma if objective.endswith("_smooth") else 0.0)
        elif spec.family == "multilabel":
            objective, _, gamma = spec.options
            options["gamma"] = gamma if objective.endswith("_smooth") else 0.0
        else:
            assert spec.family == "prox"
            alpha, fp32 = spec.options
            objective = "proximal_cross_entropy"
            options.update(
                alpha=alpha, precision="float32" if fp32 else "float64", reduction="none"
            )
        return losses.sdca_loss(z, target, objective.removesuffix("_smooth"), **options)

    def one(*operands: Any) -> list[jax.Array]:
        z, target = operands[0], _reference_target(operands[1], spec)
        if spec.stage == "backward":
            upstream = operands[3].astype(_out_dtype(spec.dtype))
            gradient = (
                spec.gradient(z, target, upstream, *spec.reference_args)
                if spec.gradient is not None
                else jax.vjp(lambda x: value(x, target), z)[1](upstream)[0]
            )
            results = [gradient]
        elif spec.stage == "rows":
            if spec.reference is not None:
                results = list(value(z, target))
            else:
                values, vjp = jax.vjp(lambda x: value(x, target), z)
                results = [values, vjp(jnp.ones_like(values))[0]]
        else:
            results = [value(z, target)]
        layouts = plan(spec).results
        results = [r.astype(dtype) for r, (_, dtype) in zip(results, layouts, strict=False)]
        results.extend(jnp.zeros(shape, dtype) for shape, dtype in layouts[len(results) :])
        return results

    if batch == 1:
        return one(*args)
    unbatched = _unbatched(args, spec, batch)
    shared = arrays(spec)
    axes = [0] * shared + [None] * (len(args) - shared)
    return _flattened(jax.vmap(one, in_axes=axes)(*unbatched), spec, batch)


cuda_p.def_impl(_impl)
cuda_p.def_abstract_eval(_abstract)
ad.primitive_jvps[cuda_p] = _jvp
batching.primitive_batchers[cuda_p] = _batch
mlir.register_lowering(cuda_p, _elsewhere)
mlir.register_lowering(cuda_p, _lowering, platform="cuda")


# The tangent of entropy: forward mode evaluates the reference's; its transpose (reverse
# mode) is the backward kernel.

tangent_p = Primitive("sdca_cuda_tangent")


def _tangent_reference(z: Any, target: Any, stats: Any, dz: Any, *, spec: Spec) -> Any:
    labels = _reference_target(target, spec)

    reference = spec.reference
    assert reference is not None

    def value(z: jax.Array) -> jax.Array:
        return reference(z, labels, *spec.reference_args)

    return jax.jvp(value, (z,), (dz,))[1]


def _tangent_abstract(z: Any, target: Any, stats: Any, dz: Any, *, spec: Spec) -> Any:
    shape, dtype = plan(spec.at("forward")).results[0]
    return jax.core.ShapedArray(shape, dtype)


def _tangent_transpose(ct: Any, z: Any, target: Any, stats: Any, dz: Any, *, spec: Spec) -> Any:
    assert ad.is_undefined_primal(dz) and not ad.is_undefined_primal(z)
    if type(ct) is ad.Zero:
        return [None, None, None, ad.Zero(dz.aval)]
    return [None, None, None, backward(z, target, stats, ct, spec)]


def _tangent_jvp(primals: Sequence[Any], tangents: Sequence[Any], *, spec: Spec) -> Any:
    z, target, stats, dz = primals
    z_dot, dz_dot = ad.instantiate_zeros(tangents[0]), ad.instantiate_zeros(tangents[3])

    def tangent(z: jax.Array, dz: jax.Array) -> jax.Array:
        return _tangent_reference(z, target, stats, dz, spec=spec)

    return jax.jvp(tangent, (z, dz), (z_dot, dz_dot))


def _tangent_batch(args: Sequence[Any], dims: Sequence[Any], *, spec: Spec) -> Any:
    rule = functools.partial(_tangent_reference, spec=spec)
    return jax.vmap(rule, in_axes=tuple(dims))(*args), 0


tangent_p.def_impl(lambda *args, spec: _tangent_reference(*args, spec=spec))
tangent_p.def_abstract_eval(_tangent_abstract)
ad.primitive_jvps[tangent_p] = _tangent_jvp
ad.primitive_transposes[tangent_p] = _tangent_transpose
batching.primitive_batchers[tangent_p] = _tangent_batch
mlir.register_lowering(tangent_p, mlir.lower_fun(_tangent_reference, multiple_results=False))


# The calls.


def _call(spec: Spec, *operands: jax.Array) -> list[jax.Array]:
    params = [jnp.asarray(p) for p in plan(spec).params]
    return cuda_p.bind(*operands, *params, spec=spec, batch=1)


def forward(z: jax.Array, target: jax.Array, spec: Spec) -> tuple[jax.Array, jax.Array]:
    """The loss (reduced) and row statistics."""
    out, stats = _call(spec, z, target)
    return out, stats


def backward(
    z: jax.Array, target: jax.Array, stats: jax.Array, ct: jax.Array, spec: Spec
) -> jax.Array:
    """The gradient, scaled by the cotangent ct before its only rounding."""
    upstream = jnp.asarray(ct, _out_dtype(spec.dtype))
    return _call(spec.at("backward"), z, target, stats, upstream)[0]


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def _entropy_loss(z: jax.Array, target: jax.Array, spec: Spec) -> jax.Array:
    return forward(z, target, spec)[0]


@_entropy_loss.defjvp
def _entropy_jvp(spec: Spec, primals: Sequence[Any], tangents: Sequence[Any]) -> Any:
    z, target = primals
    out, stats = forward(z, target, spec)
    dz = ad.instantiate_zeros(tangents[0])
    return out, tangent_p.bind(z, target, stats, dz, spec=spec)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _first_order_loss(z: jax.Array, target: jax.Array, spec: Spec) -> jax.Array:
    stage = "values" if spec.family == "distillation" else "forward"
    return forward(z, target, spec.at(stage))[0]


def _first_order_fwd(z: jax.Array, target: jax.Array, spec: Spec) -> Any:
    out, stats = forward(z, target, spec)
    return out, (z, target, stats)


def _first_order_bwd(spec: Spec, residuals: Any, ct: jax.Array) -> Any:
    z, target, stats = residuals
    return backward(z, target, stats, ct, spec), None


_first_order_loss.defvjp(_first_order_fwd, _first_order_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _rows_loss(z: jax.Array, target: jax.Array, spec: Spec) -> jax.Array:
    return _rows_fwd(z, target, spec)[0]


def _rows_fwd(z: jax.Array, target: jax.Array, spec: Spec) -> Any:
    values, gradient = _call(spec, z, target)[:2]
    return _reduced(values, spec), (values, gradient, jnp.zeros((0,), z.dtype))


def _reduced(values: jax.Array, spec: Spec) -> jax.Array:
    """As libsdca.torch: the values reduced in their dtype, then the value dtype."""
    if spec.reduction == "sum":
        values = values.sum()
    elif spec.reduction == "mean":
        values = values.mean()
    return values.astype(_out_dtype(spec.dtype))


def _rows_bwd(spec: Spec, residuals: Any, ct: jax.Array) -> Any:
    """_cpu.scaled of the saved gradient by the upstream of each row, as torch's reduction
    backward forms it."""
    values, gradient, like = residuals
    upstream = ct.astype(values.dtype)
    if spec.reduction != "none":
        upstream = jnp.broadcast_to(upstream, values.shape)
        if spec.reduction == "mean":
            upstream = upstream / values.shape[0]
    scale = upstream.astype(gradient.dtype)
    return (gradient * scale[:, None]).astype(like.dtype), None


_rows_loss.defvjp(_rows_fwd, _rows_bwd)


# Entry points of libsdca.jax.


def _labels(labels: jax.Array) -> jax.Array:
    """Integer labels as int64 words (as uint32 pairs: also without jax_enable_x64)."""
    if labels.dtype.itemsize == 8:
        return lax.bitcast_convert_type(labels, jnp.uint32)
    signed = jnp.issubdtype(labels.dtype, jnp.signedinteger)
    low = lax.bitcast_convert_type(labels.astype(jnp.int32 if signed else jnp.uint32), jnp.uint32)
    negative = labels < 0 if signed else jnp.zeros(labels.shape, bool)
    high = jnp.where(negative, jnp.uint32(0xFFFFFFFF), jnp.uint32(0))
    return jnp.stack([low, high], -1)


def _ready(spec: Spec, x: jax.Array) -> bool:
    """Whether the kernels run this call: on the local CUDA GPUs where status() holds and the
    gate accepts every kernel of both stages (a rejected kernel warns once)."""
    if not on_gpu(x) or not status()[0]:
        return False
    target = architecture()
    assert target is not None
    stages = ("rows",) if spec.stage == "rows" else ("forward", "backward")
    if spec.family == "distillation":
        stages += ("values",)
    try:
        for stage in stages:
            program(spec.at(stage), target)
    except Unsupported as error:
        _warn_once(f"{error}; this loss keeps jax.numpy")
        return False
    return True


@functools.cache
def _warn_once(message: str) -> None:
    _extensions.warn(f"libsdca: {message}")


_HALF = ("float16", "bfloat16")


def _float32_holds(*scales: float) -> bool:
    tiny = float(np.finfo(np.float32).tiny)
    return all(tiny <= s <= 1 / tiny for s in scales)


def _route(
    objective: str,
    k: int,
    labels: bool,
    dtype: str,
    canonical: str | None,
    shape: tuple[int, int],
    gamma: float,
    alpha: float,
) -> tuple[str, tuple[Any, ...]] | None:
    """The family and options of the fused path libsdca.torch takes for a CUDA call
    (backend "auto"), or None where it composes tensor operations."""
    n, c = shape
    wide = canonical != "float64" and dtype != "float64"
    if objective == "proximal_cross_entropy":
        work = "float32" if canonical != "float64" and dtype != "float64" else canonical
        return ("prox", (alpha, work == "float32" and _float32_holds(alpha))) if n < 2**31 else None
    if n > _cuda_launch.ROWS:
        return None
    if labels and (objective == "cross_entropy" or (objective == "topk_cross_entropy" and k == 1)):
        return ("entropy", (wide,)) if c <= 2**25 else None
    if not labels:
        multilabel = objective in ("cross_entropy", "multilabel_hinge", "multilabel_hinge_smooth")
        if not multilabel or _cdiv(c, 1024) >= 65536:
            return None
        return "multilabel", (objective, objective == "cross_entropy" and wide, gamma)
    return _label_route(objective, k, wide, shape, gamma)


def _label_route(
    objective: str, k: int, wide: bool, shape: tuple[int, int], gamma: float
) -> tuple[str, tuple[Any, ...]] | None:
    """_route of the integer-label hinges and top-k losses."""
    n, c = shape
    kind = objective.removesuffix("_smooth")
    if k == 1 and kind in ("topk_hinge_alpha", "topk_hinge_beta"):
        return "hinge", (
            objective.endswith("_smooth"),
            gamma,
            wide and (gamma == 0 or _float32_holds(gamma, gamma * c)),
        )
    entropy = kind == "topk_cross_entropy"
    topk = entropy or kind in ("topk_hinge_alpha", "topk_hinge_beta")
    if topk and 1 < k <= 32 and (not entropy or k < c - 1) and n * c < 2**31:
        return "topk", (objective, k, gamma, wide and entropy)
    return None


def loss(
    z: jax.Array,
    target: jax.Array,
    objective: str,
    k: int,
    gamma: float,
    alpha: float,
    reduction: str,
    canonical: str | None,
    reference: Callable[..., jax.Array] | None,
    reference_args: tuple[Any, ...] = (),
) -> jax.Array | None:
    """sdca_loss of class-last z on the fused kernels, or None where jax.numpy computes it.

    target: integer labels (z.shape[:-1]) or a boolean mask (z.shape); reference: the
    jax.numpy entropy (for its higher derivatives)."""
    if reduction not in REDUCTIONS or not on_gpu(z):
        return None
    shape, classes = z.shape[:-1], z.shape[-1]
    rows = math.prod(shape)
    labels = target.dtype != jnp.bool_
    dtype = np.dtype(z.dtype).name
    route = _route(objective, k, labels, dtype, canonical, (rows, classes), gamma, alpha)
    if route is None:
        return None
    family, options = route
    x64 = jax.dtypes.canonicalize_dtype(jnp.float64) == jnp.float64
    if family == "prox" and not options[1] and dtype != "float64" and not x64:
        return None
    entropy = family == "entropy" or (family == "multilabel" and options[0] == "cross_entropy")
    stage = "rows" if family == "prox" else "forward"
    spec = Spec(family, options, stage, rows, classes, dtype, reduction)
    if entropy:
        spec = replace(spec, reference=reference, reference_args=reference_args)
    if not _ready(spec, z):
        return None
    flat = z.reshape(rows, classes)
    t = _labels(target.reshape(rows)) if labels else target.reshape(rows, classes)
    if family == "prox":
        if labels:
            index = target.reshape(rows, 1)
            t = (jnp.arange(classes) == index) & (index >= 0) & (index < classes)
        values = _rows_loss(flat, t, spec)
    elif entropy:
        values = _entropy_loss(flat, t, spec)
    else:
        values = _first_order_loss(flat, t, spec)
    return values.reshape(shape) if reduction == "none" else values


def distillation(
    student: jax.Array,
    teacher: jax.Array,
    temperature: float,
    double: bool,
    reduction: str,
    reference: Callable[..., jax.Array],
    gradient: Callable[..., jax.Array],
) -> jax.Array | None:
    """distillation_loss of class-last logits on the fused kernels, or None. reference(s, t,
    temperature, double, reduction): the jax.numpy loss; gradient(s, t, ct, temperature,
    double, reduction): its gradient scaled by ct."""
    if not on_gpu(student):
        return None
    shape, classes = student.shape[:-1], student.shape[-1]
    rows = math.prod(shape)
    dtype = np.dtype(student.dtype).name
    if rows > _cuda_launch.ROWS or reduction not in REDUCTIONS:
        return None
    work = np.float64 if double or dtype == "float64" else np.float32
    tiny = float(np.finfo(work).tiny)
    work = work if tiny <= temperature <= 1 / tiny else np.float64
    options = (float(temperature), np.dtype(work).name)
    args = (float(temperature), double, reduction)
    spec = Spec("distillation", options, "forward", rows, classes, dtype, reduction)
    spec = replace(spec, reference=reference, reference_args=args, gradient=gradient)
    if not _ready(spec, student):
        return None
    flat_s, flat_t = student.reshape(rows, classes), teacher.reshape(rows, classes)
    values = _first_order_loss(flat_s, flat_t, spec)
    return values.reshape(shape) if reduction == "none" else values


def proximal_distillation(
    student: jax.Array,
    teacher: jax.Array,
    temperature: float,
    alpha: float,
    double: bool,
    reduction: str,
    reference: Callable[..., tuple[jax.Array, jax.Array]],
) -> jax.Array | None:
    """proximal_distillation_loss of class-last logits on the fused kernel, or None.
    reference(s, t, temperature, alpha, double): the jax.numpy values and gradient."""
    if not on_gpu(student):
        return None
    shape, classes = student.shape[:-1], student.shape[-1]
    rows = math.prod(shape)
    dtype = np.dtype(student.dtype).name
    if rows >= 2**31 or reduction not in REDUCTIONS:
        return None
    fp32 = not double and dtype != "float64" and _float32_holds(temperature, alpha)
    options = (float(temperature), float(alpha), fp32)
    spec = Spec("prox_distillation", options, "rows", rows, classes, dtype, reduction)
    spec = replace(
        spec, reference=reference, reference_args=(float(temperature), float(alpha), double)
    )
    if not _ready(spec, student):
        return None
    flat_s, flat_t = student.reshape(rows, classes), teacher.reshape(rows, classes)
    values = _rows_loss(flat_s, flat_t, spec)
    return values.reshape(shape) if reduction == "none" else values
