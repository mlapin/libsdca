"""The XLA FFI handlers of the native JAX losses on the CPU (_jax_ffi.cpp), built on first use.

SDCA_JAX_FFI selects them: "auto" (or unset) builds and caches them on the first native JAX
loss traced for the CPU where jax.ffi, its headers and a C++ compiler are available, and
otherwise (or if that build fails) keeps jax.pure_callback, the same losses through a host
round trip, with a one-time warning that names what is missing; "on" (or "1") requires them
(missing tools or a failed build raise); "off" (or "0") never builds or loads them. The
handlers are compiled against the running jaxlib's headers, so their FFI version is the
runtime's; they call the C ABI of the loaded native library, whose kernels and threads they
share. The build cache is SDCA_CACHE_DIR (default $XDG_CACHE_HOME/libsdca, else
~/.cache/libsdca), one directory per source, jaxlib, platform, compiler and flags.
_extensions holds the rules shared with the PyTorch extensions; the same rules build the CUDA
handler of _jax_cuda (Handlers).
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import importlib.metadata
import logging
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import _core, _extensions

ENV = "SDCA_JAX_FFI"
SOURCE = Path(__file__).with_name("_jax_ffi.cpp")
LIBRARY = "libsdca_jax_ffi.so"
WHAT = "the JAX FFI handlers"
FALLBACK = "native JAX losses use the slower host callback"
# The FFI targets of the losses and gradients, and of the losses alone.
GRADIENT, VALUES = "libsdca_loss_gradient", "libsdca_loss"
DISTILLATION = "libsdca_distillation_cpu"
PROXIMAL_DISTILLATION = "libsdca_proximal_distillation_cpu"
DISTILLATION_STATS = "libsdca_distillation_stats_cpu"
DISTILLATION_BACKWARD = "libsdca_distillation_backward_cpu"
# IEEE arithmetic, as the native library; the handlers only convert and check.
FLAGS = ("-std=c++17", "-O2", "-fPIC", "-shared", "-fno-fast-math", "-ffp-contract=off")
_lock = threading.RLock()
_log = logging.getLogger("libsdca")


@dataclass(frozen=True)
class Handlers:
    """One library of FFI handlers built from a source, and the switch that selects it."""

    env: str  # the auto|on|off switch
    source: Path
    library: str
    prefix: str  # of its build directory's name
    what: str  # its name in messages
    fallback: str  # what runs without it
    register: Callable[[Path], ctypes.CDLL]  # loads a build and registers its targets
    prebuilt: str | None = None  # optional platform wheel library, loaded without a compiler


def mode() -> str:
    """SDCA_JAX_FFI as "auto", "on" or "off"; ValueError for other values."""
    return _extensions.mode(ENV)


def enabled() -> bool:
    """Whether native JAX losses on the CPU call the FFI handlers (SDCA_JAX_FFI, see the
    module docstring); "on" registers them first or raises RuntimeError."""
    selected = mode()
    if selected == "on":
        register()
    if selected != "auto":
        return selected == "on"
    return status()[0]


def why_not() -> str:
    """Why native JAX losses on the CPU take the host callback ("" if they do not)."""
    selected = mode()
    if selected != "auto":
        return f"{ENV}=off" if selected == "off" else ""
    return status()[1]


def missing_tools() -> str | None:
    """What the handlers need to build here and is missing (checked without compiling)."""
    import jax

    ffi = getattr(jax, "ffi", None)
    if ffi is None or not hasattr(ffi, "include_dir"):
        return f"jax {jax.__version__} has no jax.ffi (upgrade jax)"
    header = Path(ffi.include_dir()) / "xla/ffi/api/ffi.h"
    if not header.is_file():
        return f"jaxlib ships no XLA FFI headers ({header})"
    return _extensions.missing_compiler()


def cache_root() -> Path:
    """SDCA_CACHE_DIR, else $XDG_CACHE_HOME/libsdca, else ~/.cache/libsdca."""
    root = os.environ.get("SDCA_CACHE_DIR")
    if root:
        return Path(root)
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "libsdca"


@functools.cache
def _cpu_target(cxx: str, machine: str) -> tuple[tuple[str, ...], str]:
    """Native x86 ISA and its compiler feature fingerprint, also on shared caches."""
    if machine.lower() not in ("x86_64", "amd64"):
        return (), ""
    flags = ("-march=native",)
    try:
        result = subprocess.run(
            [*shlex.split(cxx), *flags, "-dM", "-E", "-x", "c++", "-"],
            input="",
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        # The portable build: a compiler without -march=native, or none (the build reports it).
        return (), ""
    return flags, hashlib.sha256(result.stdout.encode()).hexdigest()


def _compile_flags(handlers: Handlers) -> tuple[str, ...]:
    if handlers.source != SOURCE:
        return FLAGS
    native, _ = _cpu_target(os.environ.get("CXX", "c++"), platform.machine())
    return tuple("-O3" if flag == "-O2" else flag for flag in FLAGS) + native


def build_directory(handlers: Handlers | None = None) -> Path:
    """Where the handlers are cached: one directory per source, jaxlib (its version and FFI
    headers), platform, compiler (command and version) and flags. It runs the compiler (once
    per CXX): check the tools first."""
    import jax

    handlers = handlers or CPU
    headers = sorted((Path(jax.ffi.include_dir()) / "xla/ffi/api").glob("*.h"))
    if handlers.source == SOURCE:
        headers += _extensions.cpu_headers()
    parts = [
        _jaxlib_version(),
        sys.platform,
        platform.machine(),
        _extensions.compiler(),
        *_compile_flags(handlers),
    ]
    if handlers.source == SOURCE:
        parts.append(_cpu_target(os.environ.get("CXX", "c++"), platform.machine())[1])
    identity = b"\0".join(
        [
            handlers.source.read_bytes(),
            *(h.read_bytes() for h in headers),
            *(p.encode() for p in parts),
        ]
    )
    return cache_root() / (handlers.prefix + hashlib.sha256(identity).hexdigest()[:16])


def _jaxlib_version() -> str:
    try:
        return importlib.metadata.version("jaxlib")
    except importlib.metadata.PackageNotFoundError:
        return ""


def failure_record(handlers: Handlers | None = None) -> Path:
    """The record of a failed automatic build, next to the build directory."""
    directory = build_directory(handlers)
    return directory.with_name(f"{directory.name}.{_extensions.FAILED}")


def _build(directory: Path, handlers: Handlers | None = None) -> None:
    """Compile into a private sibling directory, then rename it into place (unless another
    process got there first: then its complete build stays)."""
    import jax

    handlers = handlers or CPU
    directory.parent.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix=directory.name + ".", dir=directory.parent))
    try:
        command = [*shlex.split(os.environ.get("CXX", "c++")), *_compile_flags(handlers)]
        command += ["-I", jax.ffi.include_dir(), str(handlers.source)]
        command += ["-o", str(private / handlers.library)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            lines = result.stderr.splitlines() or [f"exit status {result.returncode}"]
            first = next((line for line in lines if "error" in line), lines[-1])
            # A compiler killed by a signal (negative status) may succeed next time.
            failed = _extensions.BuildFailure if result.returncode > 0 else RuntimeError
            raise failed(f"{first}\n\n{shlex.join(command)}\n{result.stderr}")
        try:
            private.rename(directory)
        except OSError:
            pass
    finally:
        if private.exists():
            shutil.rmtree(private, ignore_errors=True)


def _register(library: Path) -> ctypes.CDLL:
    """Load the handlers, bind them to the native library and register their targets."""
    import jax

    try:
        handlers = ctypes.CDLL(str(library))
    except OSError as error:
        raise OSError(f"{error}; delete {library.parent} to build the handlers again") from error
    native, rows = _core._tensor_entry("sdca_loss_gradient_rows")
    handlers.sdca_jax_bind.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    bound = handlers.sdca_jax_bind(
        ctypes.cast(rows, ctypes.c_void_p),
        ctypes.cast(native.sdca_last_error, ctypes.c_void_p),
        ctypes.sizeof(_core._Options),
    )
    if bound != 0:
        raise RuntimeError("the handlers' struct sdca_options is not the native library's")
    if not hasattr(native, "sdca_run_rows"):
        raise RuntimeError("the native library lacks sdca_run_rows; rebuild the native library")
    handlers.sdca_jax_bind_shared.argtypes = [ctypes.c_void_p] * 3
    handlers.sdca_jax_bind_shared.restype = None
    handlers.sdca_jax_bind_shared(
        ctypes.cast(native.sdca_run_rows, ctypes.c_void_p),
        ctypes.cast(native.sdca_distillation_forward, ctypes.c_void_p),
        ctypes.cast(native.sdca_distillation_backward, ctypes.c_void_p),
    )
    handlers.sdca_jax_bind_prox.argtypes = [ctypes.c_void_p]
    handlers.sdca_jax_bind_prox.restype = None
    handlers.sdca_jax_bind_prox(
        ctypes.cast(native.sdca_proximal_distillation_rows, ctypes.c_void_p)
    )
    targets = {
        PROXIMAL_DISTILLATION: handlers.sdca_jax_proximal_distillation,
        GRADIENT: handlers.sdca_jax_loss_gradient,
        VALUES: handlers.sdca_jax_loss,
        DISTILLATION: handlers.sdca_jax_distillation,
        DISTILLATION_STATS: handlers.sdca_jax_distillation_stats_forward,
        DISTILLATION_BACKWARD: handlers.sdca_jax_distillation_stats_backward,
    }
    for name, handler in targets.items():
        jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(handler), platform="cpu")
    return handlers


CPU = Handlers(ENV, SOURCE, LIBRARY, "jax_ffi_", WHAT, FALLBACK, lambda library: _register(library))


def bundled_library(handlers: Handlers) -> Path | None:
    """The ctypes library shipped in a platform wheel, independent of Python's ABI tag."""
    if handlers.prebuilt is None:
        return None
    libraries = sorted(handlers.source.parent.glob(handlers.prebuilt + "*.so"))
    if len(libraries) > 1:
        raise RuntimeError(f"multiple bundled libraries for {handlers.what}")
    return libraries[0] if libraries else None


def load(handlers: Handlers) -> ctypes.CDLL:
    """Load the cached handlers (building them first if needed) and register their targets;
    raises RuntimeError naming the missing tool or why the build failed. Callers cache it."""
    with _lock:
        bundled = bundled_library(handlers)
        if bundled is not None:
            return handlers.register(bundled)
        reason = missing_tools()
        if reason is not None:
            raise RuntimeError(f"libsdca: {handlers.what} cannot be built here: {reason}")
        directory = build_directory(handlers)
        library = directory / handlers.library
        if not library.is_file():
            _log.warning(
                "libsdca: building %s once in %s (a few seconds; %s=off skips it)",
                handlers.what,
                directory,
                handlers.env,
            )
            try:
                _build(directory, handlers)
            except Exception as error:
                failed = (
                    _extensions.BuildFailure
                    if isinstance(error, _extensions.BuildFailure)
                    else RuntimeError
                )
                raise failed(
                    f"libsdca: {_extensions.failure_reason(error, handlers.what)}"
                ) from error
        try:
            loaded = handlers.register(library)
        except Exception as error:
            reason = _extensions.failure_reason(error, handlers.what, "could not be loaded")
            raise RuntimeError(f"libsdca: {reason}") from error
        failure_record(handlers).unlink(missing_ok=True)
        return loaded


def automatic(handlers: Handlers, register: Callable[[], object]) -> tuple[bool, str]:
    """(True, "") where register() loads the handlers, else (False, why not); never raises:
    missing tools warn once, a failed build warns and leaves a record that later processes skip
    for RETRY seconds. Callers cache it: once per process."""
    if bundled_library(handlers) is not None:
        try:
            register()
            return True, ""
        except Exception as error:  # noqa: BLE001 - preserve the automatic fallback
            reason = f"bundled {handlers.what} could not be loaded: {error}"
            _extensions.warn(f"libsdca: {reason}; {handlers.fallback}")
            return False, reason
    reason = missing_tools()
    if reason is not None:
        _extensions.warn(
            f"libsdca: {handlers.fallback}: {reason}. Install it to build {handlers.what}, or "
            f"set {handlers.env}=off to silence this."
        )
        return False, reason
    record = failure_record(handlers)
    recorded = _extensions.recorded_failure(record, handlers.env)
    if recorded is not None:
        return False, recorded
    try:
        register()
        return True, ""
    except Exception as error:  # noqa: BLE001 - any failure keeps the fallback
        reason = str(error).removeprefix("libsdca: ")
        if not isinstance(error, _extensions.BuildFailure):
            _extensions.warn(
                f"libsdca: {reason}; {handlers.fallback} in this process "
                f"({handlers.env}=off silences this)"
            )
            return False, reason
        _extensions.record_failure(record, reason, error.__cause__ or error)
        _extensions.warn(
            f"libsdca: {reason}; {handlers.fallback}. {_extensions.retry_hint(record, handlers.env)}"
        )
        return False, f"{reason}; record {record}"


@functools.cache
def register() -> ctypes.CDLL:
    """Load the cached CPU handlers (building them first if needed) and register their
    targets, once per process; raises RuntimeError naming the missing tool or why the build
    failed."""
    return load(CPU)


@functools.cache
def status() -> tuple[bool, str]:
    """(True, "") where SDCA_JAX_FFI is "auto" and the handlers register, else (False, why
    not). Tries once per process (status.cache_clear() tries again) and never raises: missing
    tools warn once, a failed build warns and leaves a record that later processes skip for
    RETRY seconds."""
    return automatic(CPU, register)
