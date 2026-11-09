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
_extensions holds the rules shared with the PyTorch extensions.
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
import warnings
from pathlib import Path

from . import _core, _extensions

ENV = "SDCA_JAX_FFI"
SOURCE = Path(__file__).with_name("_jax_ffi.cpp")
LIBRARY = "libsdca_jax_ffi.so"
WHAT = "the JAX FFI handlers"
FALLBACK = "native JAX losses use the slower host callback"
# The FFI targets of the losses and gradients, and of the losses alone.
GRADIENT, VALUES = "libsdca_loss_gradient", "libsdca_loss"
# IEEE arithmetic, as the native library; the handlers only convert and check.
FLAGS = ("-std=c++17", "-O2", "-fPIC", "-shared", "-fno-fast-math", "-ffp-contract=off")
_lock = threading.Lock()
_log = logging.getLogger("libsdca")


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


def build_directory() -> Path:
    """Where the handlers are cached: one directory per source, jaxlib (its version and FFI
    headers), platform, compiler (command and version) and flags. It runs the compiler (once
    per CXX): check the tools first."""
    import jax

    headers = sorted((Path(jax.ffi.include_dir()) / "xla/ffi/api").glob("*.h"))
    parts = [_jaxlib_version(), sys.platform, platform.machine(), _extensions.compiler(), *FLAGS]
    identity = b"\0".join(
        [SOURCE.read_bytes(), *(h.read_bytes() for h in headers), *(p.encode() for p in parts)]
    )
    return cache_root() / ("jax_ffi_" + hashlib.sha256(identity).hexdigest()[:16])


def _jaxlib_version() -> str:
    try:
        return importlib.metadata.version("jaxlib")
    except importlib.metadata.PackageNotFoundError:
        return ""


def failure_record() -> Path:
    """The record of a failed automatic build, next to the build directory."""
    directory = build_directory()
    return directory.with_name(f"{directory.name}.{_extensions.FAILED}")


def _build(directory: Path) -> None:
    """Compile into a private sibling directory, then rename it into place (unless another
    process got there first: then its complete build stays)."""
    import jax

    directory.parent.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix=directory.name + ".", dir=directory.parent))
    try:
        command = [*shlex.split(os.environ.get("CXX", "c++")), *FLAGS]
        command += ["-I", jax.ffi.include_dir(), str(SOURCE), "-o", str(private / LIBRARY)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            lines = result.stderr.splitlines() or [f"exit status {result.returncode}"]
            first = next((line for line in lines if "error" in line), lines[-1])
            raise RuntimeError(f"{first}\n\n{shlex.join(command)}\n{result.stderr}")
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

    handlers = ctypes.CDLL(str(library))
    native, rows = _core._tensor_entry("sdca_loss_gradient_rows")
    handlers.sdca_jax_bind.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    bound = handlers.sdca_jax_bind(
        ctypes.cast(rows, ctypes.c_void_p),
        ctypes.cast(native.sdca_last_error, ctypes.c_void_p),
        ctypes.sizeof(_core._Options),
    )
    if bound != 0:
        raise RuntimeError("the handlers' struct sdca_options is not the native library's")
    targets = {GRADIENT: handlers.sdca_jax_loss_gradient, VALUES: handlers.sdca_jax_loss}
    for name, handler in targets.items():
        jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(handler), platform="cpu")
    return handlers


@functools.cache
def register() -> ctypes.CDLL:
    """Load the cached handlers (building them first if needed) and register their targets,
    once per process; raises RuntimeError naming the missing tool or why the build failed."""
    with _lock:
        reason = missing_tools()
        if reason is not None:
            raise RuntimeError(f"libsdca: {WHAT} cannot be built here: {reason}")
        directory = build_directory()
        library = directory / LIBRARY
        try:
            if not library.is_file():
                _log.warning(
                    "libsdca: building %s once in %s (a few seconds; %s=off skips it)",
                    WHAT,
                    directory,
                    ENV,
                )
                _build(directory)
            handlers = _register(library)
        except Exception as error:
            raise RuntimeError(f"libsdca: {_extensions.failure_reason(error, WHAT)}") from error
        failure_record().unlink(missing_ok=True)
        return handlers


@functools.cache
def status() -> tuple[bool, str]:
    """(True, "") where SDCA_JAX_FFI is "auto" and the handlers register, else (False, why
    not). Tries once per process (status.cache_clear() tries again) and never raises: missing
    tools warn once, a failed build warns and leaves a record that later processes skip for
    RETRY seconds."""
    reason = missing_tools()
    if reason is not None:
        warnings.warn(
            f"libsdca: {FALLBACK}: {reason}. Install it to build {WHAT}, or set {ENV}=off "
            "to silence this.",
            RuntimeWarning,
            stacklevel=4,
        )
        return False, reason
    record = failure_record()
    recorded = _extensions.recorded_failure(record, ENV)
    if recorded is not None:
        return False, recorded
    try:
        register()
        return True, ""
    except Exception as error:  # noqa: BLE001 - any failure keeps the host callback
        reason = str(error).removeprefix("libsdca: ")
        _extensions.record_failure(record, reason, error.__cause__ or error)
        warnings.warn(
            f"libsdca: {reason}; {FALLBACK}. Details: {record} (retried after 24 h; "
            f"{ENV}=off silences this)",
            RuntimeWarning,
            stacklevel=4,
        )
        return False, f"{reason}; record {record}"
