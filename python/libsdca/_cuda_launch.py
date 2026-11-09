"""Gate, build and check the optional C++ launcher of the fused Triton kernels.

The launcher calls cuLaunchKernel on Triton's compiled functions with the kernel's
non-constexpr arguments and null scratch pointers, which follows Triton's launch ABI.
SDCA_CUDA_LAUNCHER selects it: "auto" (or unset) builds it within the verified Triton
releases where ninja, a C++ compiler, the Python and CUDA headers and libcuda.so.1 are
available; otherwise, or if that build fails (recorded as for the CPU extension, retried
after 24 h), the same kernels launch through Triton, with a one-time warning. "on" (or
"1") skips the version gate and raises if the tools are missing or the launcher does
not build; "off" (or "0") always launches through Triton (_extensions holds these rules).
A kernel that needs more than the launcher passes launches through Triton in every mode,
with a one-time warning (kernel_gate).
"""

from __future__ import annotations

import ctypes
import functools
import math
import re
import warnings
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from . import _extensions

ENV = "SDCA_CUDA_LAUNCHER"
# The fused kernels need Triton 3.3: 3.2 aborts compiling float64 to bfloat16 stores.
KERNELS = (3, 3)
# Triton releases whose CUDA driver passes the non-constexpr arguments, then at most a
# global and a profile scratch pointer (3.3-3.4: one, 3.5-3.9: both).
VERIFIED = ((3, 3), (3, 9))
# Bound on cached kernel handle sets: one per device, dtype, class count and options.
CACHE = 64
POINTER, INT32 = 8, 4
# Rows of the fused kernels: their int32 offsets reach 7 * N (wider ones are int64).
ROWS = 2**28


def device_guard(tensor: Any) -> AbstractContextManager[None]:
    """Select a CUDA tensor's device for a direct launch, preserving the caller's.

    Triton's launcher uses the current device, not the pointer's device. CPU
    tensors use a no-op context so TRITON_INTERPRET=1 tests still run on CPU.
    """
    import torch

    return torch.cuda.device(tensor.device) if tensor.is_cuda else nullcontext()


def _release(version: str) -> tuple[int, int] | None:
    match = re.match(r"(\d+)\.(\d+)", version)
    return (int(match[1]), int(match[2])) if match else None


def kernels_supported(version: str) -> bool:
    """Whether this Triton version compiles the fused kernels."""
    release = _release(version)
    return release is not None and release >= KERNELS


def version_gate(version: str) -> str | None:
    """None when the launcher is verified with this Triton version, else why not."""
    release = _release(version)
    if release is None or not VERIFIED[0] <= release <= VERIFIED[1]:
        low, high = (".".join(map(str, v)) for v in VERIFIED)
        return (
            f"Triton {version} is outside the launcher's verified {low}-{high} ({ENV}=on tries it)"
        )
    return None


def kernel_gate(metadata: Any, params: list[int] | None, expected: tuple[int, ...]) -> str | None:
    """None when cuLaunchKernel with null scratch pointers runs this compiled kernel, else why not.

    params are the parameter sizes the driver reports (None when it cannot), expected the
    sizes of the kernel's own arguments.
    """
    name = getattr(metadata, "name", "kernel")
    clusters = math.prod(getattr(metadata, "cluster_dims", None) or (1,))
    if getattr(metadata, "num_ctas", 1) != 1 or clusters != 1:
        return f"{name} uses thread block clusters"
    for field in ("global_scratch_size", "profile_scratch_size"):
        if getattr(metadata, field, 0):
            return f"{name} needs {field.replace('_size', '')} memory"
    for field in ("launch_cooperative_grid", "launch_pdl", "instrumentation_mode"):
        if getattr(metadata, field, None):
            return f"{name} sets {field}"
    if params is not None:
        own, extra = params[: len(expected)], params[len(expected) :]
        if tuple(own) != expected or len(extra) > 2 or any(size != POINTER for size in extra):
            return f"{name} takes parameters of sizes {params}, not {list(expected)} and scratch"
    return None


@functools.cache
def _param_info() -> Any:
    """cuFuncGetParamInfo (CUDA 12.4 drivers), or None."""
    try:
        function = ctypes.CDLL("libcuda.so.1").cuFuncGetParamInfo
    except (OSError, AttributeError):
        return None
    size_p = ctypes.POINTER(ctypes.c_size_t)
    function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, size_p, size_p]
    function.restype = ctypes.c_int
    return function


@functools.cache
def _driver() -> tuple[Any, Any]:
    """cuCtxGetCurrent and cuCtxSetCurrent."""
    cuda = ctypes.CDLL("libcuda.so.1")
    get, set_ = cuda.cuCtxGetCurrent, cuda.cuCtxSetCurrent
    get.argtypes, set_.argtypes = [ctypes.POINTER(ctypes.c_void_p)], [ctypes.c_void_p]
    return get, set_


@functools.cache
def _primary(device: int) -> int:
    """The retained primary context of a device, the one torch and Triton use."""
    cuda = ctypes.CDLL("libcuda.so.1")
    handle, context = ctypes.c_int(), ctypes.c_void_p()
    if cuda.cuDeviceGet(ctypes.byref(handle), device) != 0:
        raise RuntimeError("cuDeviceGet failed")
    if cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), handle) != 0:
        raise RuntimeError("cuDevicePrimaryCtxRetain failed")
    return int(context.value or 0)


def ensure_context(device: int) -> int:
    """The device's primary context, made current on a thread without one (autograd workers).

    As Triton's launcher: a thread with a current context keeps it.
    """
    get, set_ = _driver()
    primary, current = _primary(device), ctypes.c_void_p()
    if get(ctypes.byref(current)) == 0 and not current.value and set_(primary) != 0:
        raise RuntimeError("cuCtxSetCurrent failed")
    return primary


def param_sizes(function: int) -> list[int] | None:
    """Parameter sizes of a loaded CUfunction, or None when the driver cannot report them."""
    info = _param_info()
    if info is None:
        return None
    offset, size = ctypes.c_size_t(), ctypes.c_size_t()
    sizes: list[int] = []
    while len(sizes) < 64:
        if info(function, len(sizes), ctypes.byref(offset), ctypes.byref(size)) != 0:
            break
        sizes.append(size.value)
    return sizes or None


def handles(kernels: Sequence[tuple[Any, tuple[int, ...]]], *extra: int) -> Any:
    """CPU int64 tensor of (CUfunction, warps, shared bytes) per kernel, then extra; or why not.

    A tensor reaches the C++ launcher without per-call conversion.
    """
    import torch

    out: list[int] = []
    for kernel, expected in kernels:
        kernel._init_handles()
        reason = kernel_gate(kernel.metadata, param_sizes(kernel.function), expected)
        if reason is not None:
            _warn(f"{reason}; its kernels launch through Triton")
            return reason
        out += [kernel.function, kernel.metadata.num_warps, kernel.metadata.shared]
    # The launcher reads them on the host, whatever torch's default device.
    with torch.inference_mode(False):
        return torch.tensor([*out, *extra], dtype=torch.int64, device="cpu")


def _warn(message: str) -> None:
    warnings.warn(f"libsdca: {message}", RuntimeWarning, stacklevel=3)


@functools.cache
def _extension(triton_version: str) -> _extensions.Extension:
    """The launcher build: the C++ extension compiled with SDCA_CUDA against the CUDA
    toolkit (its version, and Triton's, in the cache key)."""
    import torch

    cuda_home = _extensions.cuda_home()
    return _extensions.Extension(
        env=ENV,
        what="the C++ launcher",
        fallback="fused CUDA kernels launch through Triton",
        prefix="libsdca_entropy_cuda_",
        cflags=(
            "-O3",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-DSDCA_CUDA",
            f"-I{cuda_home}/include",
        ),
        ldflags=(f"-L{cuda_home}/lib64", f"-L{cuda_home}/lib64/stubs", "-lcuda", "-lcudart"),
        key=(f"triton {triton_version}", f"cuda {_extensions.cuda_version()} {torch.version.cuda}"),
        cuda=True,
    )


def failure_record(triton_version: str) -> Path:
    """The record of a failed automatic build, next to the build directory."""
    return _extensions.failure_record(_extension(triton_version))


@functools.cache
def status() -> tuple[Any, str]:
    """(extension module, "") when the launcher runs, else (None, why not); raises with
    SDCA_CUDA_LAUNCHER=on where it does not build (see the module docstring).

    Reads ENV and Triton's version once per process (status.cache_clear() rereads them).
    """
    selected = _extensions.mode(ENV)
    if selected == "off":
        return None, f"{ENV}=off"
    import triton

    version = triton.__version__
    if selected == "on":
        return _extensions.load(_extension(version)), ""
    reason = version_gate(version)
    if reason is not None:
        return None, reason
    return _extensions.automatic(_extension(version))


def host() -> Any:
    """The C++ launcher extension, or None."""
    return status()[0]
