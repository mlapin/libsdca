"""First-use builds of the CPU kernels of libsdca.torch (_torch_cpu.cpp, torch.library CPU
kernels) and libsdca.jax (_jax_cpu.cpp, an XLA FFI handler): one translation unit each over
the header-only kernels (include/sdca), compiled against the running framework.

Each build has an auto|on|off switch (SDCA_CPU_EXTENSION, SDCA_JAX_FFI): "auto" (or unset)
builds on first use, "on" also retries a recorded failure, "off" never builds (CPU calls
then take the same kernels in the native library: _frontend.cpu_route). Builds are cached
in SDCA_CACHE_DIR (default $XDG_CACHE_HOME/libsdca, else ~/.cache/libsdca), one directory
per source, headers, framework build, compiler and flags; a failed build is recorded there
and skipped by later processes for RETRY seconds. Processes that need the same build at once
(ranks, workers) take turns on a lock file: one compiles, the others load its library.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

MODES = {"": "auto", "auto": "auto", "on": "on", "1": "on", "off": "off", "0": "off"}
RETRY = 24 * 3600
FAILED = "build-failed.txt"
HERE = Path(__file__).parent
# IEEE arithmetic, as the native library: no contraction, no fast math.
FLAGS = ("-O3", "-DNDEBUG", "-fPIC", "-shared", "-fno-fast-math", "-ffp-contract=off", "-g0")
_lock = threading.RLock()
_log = logging.getLogger("libsdca")


@dataclass(frozen=True)
class Build:
    """One shared library of a source and the switch that selects it."""

    env: str
    source: str  # a file next to this module
    what: str  # its name in messages
    cflags: Callable[[], list[str]]  # the framework's standard, includes and macros
    link: Callable[[], list[str]]  # its linker flags
    key: Callable[[], list[str]]  # what else changes the binary (framework build)
    missing: Callable[[], str | None]  # what the framework lacks to build it, or None


def mode(env: str) -> str:
    """The setting of env: "auto" (also unset), "on" ("1") or "off" ("0")."""
    value = os.environ.get(env, "").strip().lower()
    if value not in MODES:
        raise ValueError(f"{env} must be auto, on or off (or 1, 0), not {value!r}")
    return MODES[value]


def cache_root() -> Path:
    root = os.environ.get("SDCA_CACHE_DIR")
    if root:
        return Path(root)
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "libsdca"


def include_dir() -> Path:
    """The kernel headers: the copy in a wheel (libsdca/include), else the source tree's."""
    bundled = HERE / "include"
    return bundled if (bundled / "sdca").is_dir() else HERE.parents[1] / "include"


def headers() -> list[Path]:
    root = include_dir() / "sdca"
    return sorted(p for p in root.rglob("*.h")) + [HERE / "_frontend_cpu.h"]


def compiler_command() -> list[str]:
    return shlex.split(os.environ.get("CXX", "c++"))


def missing_compiler() -> str | None:
    """Why the C++ compiler (CXX, else c++) cannot run here, or None."""
    command = compiler_command()
    found = shutil.which(command[0]) if command else None
    if found is None:
        return (
            "no C++ compiler (c++ or CXX) on PATH (Linux: g++, e.g. the build-essential "
            "package; macOS: xcode-select --install)"
        )
    if sys.platform == "darwin" and found.startswith("/usr/bin/"):
        tools = subprocess.run(["xcode-select", "-p"], capture_output=True, check=False)
        if tools.returncode != 0:
            return "the Xcode Command Line Tools are not installed (xcode-select --install)"
    return None


@functools.cache
def _compiler_version(command: tuple[str, ...]) -> str:
    try:
        out = subprocess.run(
            [*command, "--version"], capture_output=True, text=True, check=False, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        out = repr(error)
    return " ".join(command) + "\n" + (out.splitlines() or [""])[0]


def directory(build: Build) -> Path:
    """The cache directory of a build: one per source, headers, framework, compiler, flags."""
    source = HERE / build.source
    inputs = headers()
    if build.source == "_torch_cpu.cpp":
        inputs.append(HERE / "_torch_cuda_plan.h")
    parts = [
        source.read_bytes(),
        *(h.read_bytes() for h in inputs),
        *(s.encode() for s in (*FLAGS, *build.cflags(), *build.key(), sys.platform)),
        _compiler_version(tuple(compiler_command())).encode(),
    ]
    digest = hashlib.sha256(b"\0".join(parts)).hexdigest()[:16]
    return cache_root() / f"{Path(build.source).stem}-{digest}"


def library(build: Build) -> Path:
    return directory(build) / f"{Path(build.source).stem}.so"


def _compile(build: Build, target: Path) -> None:
    """Compile into a private sibling directory, then rename it into place (another
    process's complete build wins)."""
    target.parent.parent.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix=target.parent.name + ".", dir=target.parent.parent))
    try:
        out = private / target.name
        command = [
            *compiler_command(),
            *FLAGS,
            f"-I{include_dir()}",
            f"-I{HERE}",
            *build.cflags(),
            str(HERE / build.source),
            "-o",
            str(out),
            *build.link(),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            errors = [line for line in result.stderr.splitlines() if "error" in line]
            first = errors[0].strip() if errors else f"exit status {result.returncode}"
            raise BuildFailure(
                f"{build.what} did not build ({first})\n{shlex.join(command)}\n"
                f"{result.stderr[-8000:]}"
            )
        try:
            private.rename(target.parent)
        except OSError:
            pass
    finally:
        if private.exists():
            shutil.rmtree(private, ignore_errors=True)


@contextlib.contextmanager
def _exclusive(directory: Path) -> Iterator[None]:
    """Hold the lock file of a build directory, so that one process compiles it at a time
    (each compile of the framework's headers takes gigabytes of memory). Without fcntl
    (Windows) processes compile at once and the first complete build wins."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    with open(directory.with_name(directory.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class BuildFailure(RuntimeError):
    """The compiler failed: recorded, as the same build would fail again."""


def _recorded(record: Path, env: str) -> str | None:
    try:
        age = time.time() - record.stat().st_mtime
        if age >= RETRY or mode(env) == "on":
            record.unlink(missing_ok=True)
            return None
        reason = (record.read_text().splitlines() or [""])[0]
    except OSError:
        return None
    return f"an earlier build failed ({reason}); see {record}, or set {env}=on to retry"


def _build_once(build: Build, path: Path, record: Path) -> str | None:
    """Compile path unless another process built it (or recorded its failure) while this
    one waited for the lock; None, else why it is not built."""
    if path.is_file():
        return None
    recorded = _recorded(record, build.env)
    if recorded is not None:
        return recorded
    _log.warning("libsdca: building %s once in %s", build.what, path.parent)
    try:
        _compile(build, path)
    except BuildFailure as error:
        try:
            record.write_text(f"{str(error).splitlines()[0]}\n\n{error}\n")
        except OSError:
            pass
        return str(error).splitlines()[0]
    return None


@functools.cache
def load(build: Build, loader: Callable[[Path], object]) -> tuple[object | None, str]:
    """(loaded library, "") of a build, compiled first where needed, or (None, why not).
    Never raises: callers turn the reason into NotImplementedError."""
    with _lock:
        if mode(build.env) == "off":
            return None, f"{build.env}=off"
        reason = build.missing() or missing_compiler()
        if reason is not None:
            return None, f"{build.what} cannot be built here: {reason}"
        path = library(build)
        record = path.parent.with_name(path.parent.name + "." + FAILED)
        if not path.is_file():
            try:
                with _exclusive(path.parent):
                    reason = _build_once(build, path, record)
            except OSError as error:
                return None, f"{build.what} did not build ({error})"
            if reason is not None:
                return None, reason
        try:
            return loader(path), ""
        except Exception as error:  # noqa: BLE001 - reported as the reason
            return None, f"{build.what} did not load ({type(error).__name__}: {error})"


def python_include() -> str:
    return sysconfig.get_paths()["include"]
