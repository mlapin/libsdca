"""Shared, standard-library-only configuration for the two Python builders."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import shlex
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def version() -> str:
    """The release version, from the VERSION file next to this script."""
    return (ROOT / "VERSION").read_text().strip()


def enabled(name: str, default: bool = True) -> bool:
    """Read an ON/OFF (1/0, TRUE/FALSE) environment switch."""
    value = os.environ.get(name, "ON" if default else "OFF").upper()
    if value not in {"ON", "OFF", "1", "0", "TRUE", "FALSE"}:
        raise ValueError(name + " must be ON or OFF")
    return value in {"ON", "1", "TRUE"}


# The driver roles of a distutils Unix compiler; names ending in "_cxx" run the C++ driver.
_DRIVERS = (
    "preprocessor",
    "compiler",
    "compiler_so",
    "compiler_cxx",
    "compiler_so_cxx",
    "linker_so",
    "linker_so_cxx",
    "linker_exe",
    "linker_exe_cxx",
)


def use_available_compilers(compiler: Any) -> list[str]:
    """Replace a configured compiler that is not on PATH by one that is; return what changed.

    A relocatable Python (uv, python-build-standalone) names the compiler of its own build,
    such as clang++, which the host may not have. CC and CXX still take precedence.
    """
    # Prefer one family for C and C++ (cc/c++, gcc/g++, clang/clang++); if no family has both
    # drivers on PATH, choose each driver on its own.
    families = (("cc", "c++"), ("gcc", "g++"), ("clang", "clang++"))
    pair = next((f for f in families if all(shutil.which(name) for name in f)), None)
    candidates = {
        cxx: (pair[cxx],) if pair else tuple(f[cxx] for f in families) for cxx in (False, True)
    }
    changes: list[str] = []
    for role in _DRIVERS:
        command = getattr(compiler, role, None)
        cxx = role.endswith("_cxx")
        if not command or shutil.which(command[0]) or os.environ.get("CXX" if cxx else "CC"):
            continue
        found = next((name for name in candidates[cxx] if shutil.which(name)), None)
        if found is None:
            continue
        change = (
            f"{command[0]} is not on PATH; building with {found}"
            " (set CC and CXX to choose, e.g. CC=gcc CXX=g++)"
        )
        if change not in changes:
            changes.append(change)
        compiler.set_executable(role, [found, *command[1:]])
    return changes


def settings(key: str) -> list[str]:
    """The entries of key in src/native_build.txt, the build settings of every build."""
    lines = (ROOT / "src" / "native_build.txt").read_text().splitlines()
    return [line[len(key) + 1 :].strip() for line in lines if line.startswith(key + " ")]


# The compile flags of the C ABI library.
COMPILE_FLAGS = settings("flags")
# The warning of a build that found no BLAS (AUTO).
FALLBACK_WARNING = (
    "libsdca found no BLAS (tried {tried}) and uses its built-in products: install OpenBLAS"
    " (or another CBLAS) or set SDCA_BLAS to its linker flags, then rebuild"
)


def _cxx() -> list[str]:
    """The C++ compiler: CXX, else the first of c++, g++ and clang++ on PATH."""
    if os.environ.get("CXX"):
        return shlex.split(os.environ["CXX"])
    return [next((name for name in ("c++", "g++", "clang++") if shutil.which(name)), "c++")]


def links(cxx: list[str], flags: list[str]) -> bool:
    """Whether src/blas_probe.cpp, which calls the CBLAS functions of libsdca, links with
    the linker flags."""
    with tempfile.TemporaryDirectory() as tmp:
        command = [*cxx, "-std=c++11", "-I", str(ROOT / "include")]
        command += [str(ROOT / "src" / "blas_probe.cpp"), "-o", str(Path(tmp) / "probe"), *flags]
        try:
            done = subprocess.run(command, capture_output=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return done.returncode == 0


def blas_package() -> dict[str, Any]:
    """The BLAS package of src/native_build.txt, installed in this environment: its
    distribution, requirement (at least this version), module, library (its path in the
    package), linker flags and macros."""
    distribution, prefix, name = settings("package")[0].split(maxsplit=2)
    module_name = distribution.replace("-", "_")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(
            f"SDCA_BLAS_BACKEND=PACKAGE links the library of {distribution}; install it in the"
            " build environment"
        ) from error
    folder = Path(module.get_lib_dir()).resolve()
    return {
        "distribution": distribution,
        "requirement": f"{distribution}>={importlib.metadata.version(distribution)}",
        "module": module_name,
        "library": (folder / module.get_library(fullname=True))
        .relative_to(Path(str(module.__file__)).resolve().parent)
        .as_posix(),
        "flags": [f"-L{folder}", f"-l{module.get_library()}"],
        "macros": [f"SDCA_CBLAS_PREFIX={prefix}", f'SDCA_BLAS_NAME="{name} ({distribution})"'],
    }


def blas_config(
    backend: str | None = None, cxx: list[str] | None = None
) -> tuple[str, list[str], list[str]]:
    """Select the BLAS (src/native_build.txt): (EXTERNAL, PACKAGE, BUILTIN or FALLBACK, the
    macros, the linker flags). SDCA_BLAS_BACKEND (or backend) is AUTO (the default), EXTERNAL,
    PACKAGE or BUILTIN; SDCA_BLAS replaces the candidates by its own linker flags."""
    requested = (backend or os.environ.get("SDCA_BLAS_BACKEND", "AUTO")).upper()
    if requested not in {"AUTO", "EXTERNAL", "PACKAGE", "BUILTIN"}:
        raise ValueError("SDCA_BLAS_BACKEND must be AUTO, EXTERNAL, PACKAGE, or BUILTIN")
    if requested == "BUILTIN":
        return "BUILTIN", ["BLAS_BUILTIN"], []
    compiler = cxx or _cxx()
    if requested == "PACKAGE":
        package = blas_package()
        macros, flags = package["macros"], package["flags"]
        if not links(compiler, [*("-D" + macro for macro in macros), *flags]):
            raise RuntimeError(f"No CBLAS of {package['distribution']} links with {flags}")
        return "PACKAGE", ["BLAS_DEFAULT_LOCAL_HEADER", *macros], flags
    override = os.environ.get("SDCA_BLAS")
    candidates = [override] if override else settings("blas")
    for candidate in candidates:
        flags = shlex.split(candidate)
        if links(compiler, flags):
            return "EXTERNAL", ["BLAS_DEFAULT_LOCAL_HEADER"], flags
    if requested == "EXTERNAL" or override:
        raise RuntimeError(
            f"No CBLAS links with {' or '.join(candidates)}; set SDCA_BLAS to the linker flags"
            " of one, or choose SDCA_BLAS_BACKEND=BUILTIN"
        )
    warnings.warn(FALLBACK_WARNING.format(tried=", ".join(candidates)), stacklevel=2)
    return "FALLBACK", ["BLAS_BUILTIN", "SDCA_BLAS_FALLBACK"], []
