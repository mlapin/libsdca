"""The CPU extension: losses on ATen's SIMD and thread pool, built on first use.

SDCA_CPU_EXTENSION selects it: "auto" (or unset) builds and caches it on the first CPU loss
call where a C++ compiler with OpenMP, ninja and the Python headers are available, and
otherwise (or if that build fails) keeps the prebuilt native library, with a one-time
warning that names what is missing; "on" (or "1") requires it (missing tools or a failed
build raise); "off" (or "0") never builds or loads it. Importing libsdca never builds code.
_extensions holds these rules.
"""

from __future__ import annotations

import ctypes
import functools
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from . import _extensions

ENV = "SDCA_CPU_EXTENSION"
FAILED, RETRY = _extensions.FAILED, _extensions.RETRY


def _constant(function: Any) -> Any:
    """torch.compile calls function at trace time and keeps its result: no build traced."""
    mark = getattr(getattr(torch, "compiler", None), "assume_constant_result", None)
    return function if mark is None else mark(function)


def mode() -> str:
    """SDCA_CPU_EXTENSION as "auto", "on" or "off"; ValueError for other values."""
    return _extensions.mode(ENV)


@_constant
def enabled() -> bool:
    """Whether CPU losses take the extension (SDCA_CPU_EXTENSION, see the module docstring)."""
    selected = mode()
    if selected != "auto":
        return selected == "on"
    return status()[0] is not None


def why_not() -> str:
    """Why CPU losses do not take the extension ("" if they do)."""
    selected = mode()
    if selected != "auto":
        return f"{ENV}=off" if selected == "off" else ""
    return status()[1]


def _openmp() -> tuple[list[str], list[str]]:
    """Compile and link flags of OpenMP, the threads of ATen's inline parallel_for."""
    if sys.platform != "darwin":
        return ["-fopenmp"], ["-fopenmp"]
    # Apple clang has no -fopenmp driver flag. PyTorch ships omp.h, libtorch_cpu loads its
    # libomp, and extensions link with -undefined dynamic_lookup: the kernels take that
    # runtime (a second libomp would not share its threads). Without it they run serially.
    if torch._C.has_openmp and (Path(torch.__file__).parent / "include/omp.h").is_file():
        return ["-Xpreprocessor", "-fopenmp"], []
    return [], []


# The instruction-set flags of each ATen CPU capability the extension builds for; any other
# capability (e.g. DEFAULT on x86 without AVX2, arm64) takes none: the generic build.
ISA_FLAGS = {
    "AVX512": [
        "-mavx512f", "-mavx512dq", "-mavx512bw", "-mavx512vl", "-mfma", "-mf16c",
        "-DCPU_CAPABILITY_AVX512",
    ],
    "AVX2": ["-mavx2", "-mfma", "-mf16c", "-DCPU_CAPABILITY_AVX2"],
}  # fmt: skip


def _flags(capability: str | None = None) -> list[str]:
    """Compiler flags: IEEE arithmetic, OpenMP and the ISA of ATen's kernels on this CPU (or of
    capability); the C++ standard is PyTorch's own."""
    flags = ["-O3", "-fno-fast-math", "-ffp-contract=off", *_openmp()[0]]
    # ATEN_CPU_CAPABILITY can lower the capability.
    if capability is None:
        capability = torch.backends.cpu.get_cpu_capability()
    flags += ISA_FLAGS.get(capability, [])
    if _sleef():
        # ATen's arm64 vectors call Sleef only where the build defines this; otherwise their
        # exp, expm1 and log run element by element through libm.
        flags += ["-DAT_BUILD_ARM_VEC256_WITH_SLEEF"]
    return flags


def _sleef() -> bool:
    """Whether ATen's arm64 vectors can call the Sleef functions of PyTorch's own library:
    its sleef.h ships and libtorch_cpu exports them (as the macOS and Linux arm64 wheels)."""
    if platform.machine().lower() not in ("arm64", "aarch64"):
        return False
    root = Path(torch.__file__).parent
    if not (root / "include/sleef.h").is_file():
        return False
    for name in ("libtorch_cpu.dylib", "libtorch_cpu.so"):
        if (root / "lib" / name).is_file():
            try:
                library = ctypes.CDLL(str(root / "lib" / name))
            except OSError:
                return False
            return all(
                hasattr(library, f"Sleef_{f}d2_u10") and hasattr(library, f"Sleef_{f}f4_u10")
                for f in ("exp", "expm1", "log", "log1p")
            )
    return False


@functools.cache
def _extension() -> _extensions.Extension:
    return _extensions.Extension(
        env=ENV,
        what="the CPU extension",
        fallback="CPU losses use the slower native library",
        prefix="libsdca_entropy_",
        cflags=tuple(_flags()),
        ldflags=tuple(_openmp()[1]),
    )


@functools.cache
def buildable() -> bool:
    """Whether this machine has the tools to build the extension (checked without compiling)."""
    return _extensions.missing_tools() is None


@functools.cache
def operators() -> Any:
    """Load the cached extension module, building it first if needed; raises RuntimeError if
    the tools are missing or it does not build."""
    return _extensions.load(_extension())


@functools.cache
def status() -> tuple[Any, str]:
    """(extension module, "") where SDCA_CPU_EXTENSION is "auto" and the extension loads,
    else (None, why not). Tries once per process (status.cache_clear() tries again) and
    never raises: missing tools or a failed build warn once (a failed build leaves a record
    that later processes skip for RETRY seconds)."""
    return _extensions.automatic(_extension())
