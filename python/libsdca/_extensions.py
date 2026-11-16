"""The rules shared by libsdca's optional compiled extensions (the CPU extension and the CUDA
launcher): an auto|on|off environment variable, the tools check, the build cache, the one-time
hint when a slower path stands in, and the record of a failed automatic build that later
processes skip until it is RETRY seconds old.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import importlib.util
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODES = {"": "auto", "auto": "auto", "on": "on", "1": "on", "off": "off", "0": "off"}
# Seconds after which a recorded build failure is retried.
RETRY = 24 * 3600
# The record of a failed automatic build, next to its build directory.
FAILED = "build-failed.txt"
SOURCE = Path(__file__).with_name("_torch_entropy.cpp")
# One build or load at a time in this process; each build directory loads once.
_lock = threading.RLock()
_modules: dict[Path, Any] = {}
_log = logging.getLogger("libsdca")


class BuildFailure(RuntimeError):
    """A new build failed in a way taken to repeat with the same build key, so automatic
    records it: the compiler exited with an error (also one of the environment that it reports
    itself, such as no space left on device), or the new module does not import. Not
    recorded: an OSError in this process (the file system), an existing build or native
    library that does not load, and, for the JAX handlers only, a compiler killed by a signal
    (PyTorch's cpp_extension reports every failed compile as the same RuntimeError)."""


@dataclass(frozen=True)
class Extension:
    """One build of SOURCE and the switch that selects it."""

    env: str  # the auto|on|off switch
    what: str  # its name in messages, e.g. "the CPU extension"
    fallback: str  # what runs without it, e.g. "CPU losses use the slower native library"
    prefix: str  # of its build directory's name
    cflags: tuple[str, ...]
    ldflags: tuple[str, ...] = ()
    key: tuple[str, ...] = ()  # what else changes the binary (CUDA and Triton versions)
    cuda: bool = False  # it also needs the CUDA headers and libcuda.so.1


def mode(env: str) -> str:
    """The setting of env as "auto" (also unset), "on" ("1") or "off" ("0"); ValueError for
    other values."""
    value = os.environ.get(env, "").strip().lower()
    if value not in MODES:
        raise ValueError(f"{env} must be auto, on or off (or 1, 0), not {value!r}")
    return MODES[value]


def ensure_ninja() -> None:
    """Put the interpreter's ninja on PATH when PATH has none: pip installs it there with
    libsdca[torch], also in a virtual environment that is not activated."""
    if shutil.which("ninja") is None:
        scripts = Path(sysconfig.get_path("scripts"))
        if shutil.which("ninja", path=str(scripts)):
            os.environ["PATH"] = f"{scripts}{os.pathsep}{os.environ.get('PATH', '')}"


def cuda_home() -> Path:
    """The CUDA toolkit of the launcher: CUDA_HOME (as PyTorch detects it), else
    /usr/local/cuda."""
    from torch.utils.cpp_extension import CUDA_HOME

    return Path(CUDA_HOME or "/usr/local/cuda")


def cuda_version() -> str:
    """CUDA_VERSION of the toolkit's cuda.h ("" without it)."""
    try:
        header = (cuda_home() / "include/cuda.h").read_text(errors="replace")
    except OSError:
        return ""
    match = re.search(r"#define\s+CUDA_VERSION\s+(\d+)", header)
    return match[1] if match else ""


def missing_tools(cuda: bool = False) -> str | None:
    """What a PyTorch C++ extension (cuda: the CUDA launcher) needs to build here and is
    missing (checked without compiling), or None."""
    ensure_ninja()
    if shutil.which("ninja") is None:
        return "ninja is not installed (pip install ninja, or libsdca[torch])"
    reason = missing_compiler(" with OpenMP")
    if reason is not None:
        return reason
    try:  # older PyTorch loads setuptools (2.1 also its pkg_resources) for these tools
        import torch.utils.cpp_extension  # noqa: F401
    except ImportError as error:
        return f"PyTorch's C++ extension tools do not load ({error}; pip install 'libsdca[torch]')"
    if not (Path(sysconfig.get_paths()["include"]) / "Python.h").is_file():
        return (
            "the Python development headers (Python.h) are not installed (Debian/Ubuntu: "
            "python3-dev)"
        )
    if cuda:
        headers = cuda_home() / "include/cuda.h"
        if not headers.is_file():
            return f"the CUDA headers are not installed ({headers}; CUDA_HOME selects the toolkit)"
        try:
            ctypes.CDLL("libcuda.so.1")
        except OSError:
            return "libcuda.so.1 (the NVIDIA driver library) is not found"
    return None


def missing_compiler(needs: str = "") -> str | None:
    """Why the C++ compiler (CXX, else c++) cannot run here, or None; needs qualifies g++."""
    command = shlex.split(os.environ.get("CXX", "c++"))
    compiler = shutil.which(command[0]) if command else None
    if compiler is None:
        return (
            f"no C++ compiler (c++ or CXX) on PATH (Linux: g++{needs}, e.g. the "
            "build-essential or gcc-c++ package; macOS: xcode-select --install)"
        )
    # Apple's /usr/bin shims offer to install the command line tools instead of compiling.
    if sys.platform == "darwin" and compiler.startswith("/usr/bin/"):
        tools = subprocess.run(["xcode-select", "-p"], capture_output=True, check=False)
        if tools.returncode != 0:
            return "the Xcode Command Line Tools are not installed (xcode-select --install)"
    return None


def compiler() -> str:
    """The C++ compiler command (CXX, else c++) and its version line, or why it cannot run."""
    return _compiler(os.environ.get("CXX", "c++"))


@functools.cache
def _compiler(cxx: str) -> str:
    command = shlex.split(cxx)
    try:
        version = subprocess.run(
            [*command, "--version"], capture_output=True, text=True, check=False, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError, IndexError) as error:
        version = repr(error)
    return " ".join(command) + "\n" + (version.splitlines() or [""])[0]


def cpu_headers() -> list[Path]:
    """Portable headers, including the canonical storage header in source/editable builds."""
    headers = sorted(SOURCE.parent.glob("_cpu_*.h"))
    if not SOURCE.with_name("_cpu_storage.h").is_file():
        canonical = SOURCE.parents[2] / "include/sdca/math/storage.h"
        if canonical.is_file():
            headers.append(canonical)
    return headers


def build_directory(extension: Extension) -> Path:
    """Where an extension is cached: one directory per source, PyTorch build, Python, platform,
    compiler (command and version), flags and key, under TORCH_EXTENSIONS_DIR, else
    PyTorch's default extension root. It runs the compiler (once per CXX): check the tools
    first."""
    import torch
    from torch.utils.cpp_extension import get_default_build_root

    # Builds of one torch version (pip, conda, CUDA variants) differ in ABI.
    build = (torch.version.git_version, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)
    parts = [torch.__version__, *map(str, build), sys.implementation.cache_tag, sys.platform]
    parts.append(compiler())
    parts += [*extension.cflags, *extension.ldflags, *extension.key]
    headers = sorted([*SOURCE.parent.glob("_torch_*.h"), *cpu_headers()])
    identity = b"\0".join(
        [
            SOURCE.read_bytes(),
            *(h.read_bytes() for h in headers),
            *(part.encode() for part in parts),
        ]
    )
    name = extension.prefix + hashlib.sha256(identity).hexdigest()[:16]
    return Path(os.environ.get("TORCH_EXTENSIONS_DIR") or get_default_build_root()) / name


def failure_record(extension: Extension) -> Path:
    """The record of a failed automatic build, next to the build directory."""
    directory = build_directory(extension)
    return directory.with_name(f"{directory.name}.{FAILED}")


def _import(directory: Path) -> Any:
    """The extension module of a complete build."""
    library = directory / f"{directory.name}.so"
    spec = importlib.util.spec_from_file_location(directory.name, library)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(extension: Extension, directory: Path) -> Any:
    """Build into a private sibling directory, load it, then rename it into place (unless
    another process got there first: then its complete build stays). A process never loads
    (or overwrites) another process's half-written library."""
    from torch.utils import cpp_extension

    ensure_ninja()
    directory.parent.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix=directory.name + ".", dir=directory.parent))
    # PyTorch names a second build of a name in another directory name_v1: a retry after a
    # failed build keeps the name that later processes import.
    cpp_extension.JIT_EXTENSION_VERSIONER.entries.pop(directory.name, None)
    try:
        module = cpp_extension.load(
            name=directory.name,
            sources=[str(SOURCE)],
            extra_cflags=list(extension.cflags),
            extra_ldflags=list(extension.ldflags),
            build_directory=str(private),
            is_python_module=True,
            with_cuda=False,
        )
        try:
            private.rename(directory)
        except OSError:
            # Another process renamed its build first; this process keeps the library it
            # loaded (unlinking a loaded library leaves its mapping intact).
            pass
    finally:
        if private.exists():
            shutil.rmtree(private, ignore_errors=True)
    return module


def load(extension: Extension) -> Any:
    """The extension module, loaded from its cache or built first (once per process). Raises
    RuntimeError naming the missing tool or why the build failed."""
    with _lock:
        reason = missing_tools(extension.cuda)
        if reason is not None:
            raise RuntimeError(f"libsdca: {extension.what} cannot be built here: {reason}")
        directory = build_directory(extension)
        if directory in _modules:
            return _modules[directory]
        built = (directory / f"{directory.name}.so").is_file()
        if not built:
            _log.warning(
                "libsdca: building %s once in %s (about a minute; %s=off skips it)",
                extension.what,
                directory,
                extension.env,
            )
        try:
            module = _import(directory) if built else _build(extension, directory)
        except Exception as error:
            if built:
                reason = failure_reason(error, extension.what, "could not be loaded")
                raise RuntimeError(
                    f"libsdca: {reason}; delete {directory} to build it again"
                ) from error
            failed = RuntimeError if isinstance(error, OSError) else BuildFailure
            raise failed(f"libsdca: {failure_reason(error, extension.what)}") from error
        failure_record(extension).unlink(missing_ok=True)
        _modules[directory] = module
        return module


_INTERNAL = frozenset(("libsdca", "torch", "jax", "jaxlib"))


def warn(message: str) -> None:
    """A RuntimeWarning attributed to the first caller outside libsdca, torch and jax."""
    frame, level = sys._getframe(1), 2
    while (
        frame.f_back is not None
        and frame.f_globals.get("__name__", "").partition(".")[0] in _INTERNAL
    ):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, RuntimeWarning, stacklevel=level)


def automatic(extension: Extension) -> tuple[Any, str]:
    """(module, "") where the extension loads, else (None, why not); never raises. Missing
    tools warn with a hint, any other failure warns, and a BuildFailure leaves a record that
    later processes skip for RETRY seconds. Callers cache the result: once per process."""
    with _lock:
        reason = missing_tools(extension.cuda)
        if reason is not None:
            warn(
                f"libsdca: {extension.fallback}: {reason}. Install it to build "
                f"{extension.what}, or set {extension.env}=off to silence this."
            )
            return None, reason
        record = failure_record(extension)
        recorded = recorded_failure(record, extension.env)
        if recorded is not None:
            return None, recorded
        try:
            return load(extension), ""
        except Exception as error:  # noqa: BLE001 - any failure keeps the fallback
            reason = str(error).removeprefix("libsdca: ")
            if not isinstance(error, BuildFailure):
                warn(
                    f"libsdca: {reason}; {extension.fallback} in this process "
                    f"({extension.env}=off silences this)"
                )
                return None, reason
            record_failure(record, reason, error.__cause__ or error)
            warn(f"libsdca: {reason}; {extension.fallback}. {retry_hint(record, extension.env)}")
            return None, f"{reason}; record {record}"


def recorded_failure(record: Path, env: str) -> str | None:
    """The reason of a failed build recorded less than RETRY seconds ago, with the record's
    path; else None (an older record is removed: the caller builds again)."""
    try:
        age = time.time() - record.stat().st_mtime
        if age >= RETRY:
            record.unlink(missing_ok=True)
            return None
        reason = (record.read_text().splitlines() or [""])[0]
    except OSError:
        return None
    hours = max(1, round((RETRY - age) / 3600))
    return (
        f"an earlier build failed: {reason}; record {record} (retried in about {hours} h; "
        f"delete it or set {env}=on to retry now)"
    )


def retry_hint(record: Path, env: str) -> str:
    """Where a new failure record is and how to retry before it expires."""
    return (
        f"Details: {record}; processes sharing this cache skip the build for 24 h (delete "
        f"it or set {env}=on to retry now; {env}=off silences this)"
    )


def record_failure(record: Path, reason: str, error: BaseException) -> None:
    """Record a failed build (its reason on the first line); best effort."""
    try:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(f"{reason}\n\n{error}\n")
    except OSError:
        pass


def failure_reason(error: BaseException, what: str, failed: str = "did not build") -> str:
    """The one-line reason why what failed ("did not build", "could not be loaded")."""
    first = (str(error).strip().splitlines() or [""])[0]
    return f"{what} {failed} ({type(error).__name__}: {first})"
