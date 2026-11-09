"""Shared, standard-library-only configuration for the two Python builders."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import shlex


def enabled(name: str, default: bool = True) -> bool:
    """Read an ON/OFF (1/0, TRUE/FALSE) environment switch."""
    value = os.environ.get(name, "ON" if default else "OFF").upper()
    if value not in {"ON", "OFF", "1", "0", "TRUE", "FALSE"}:
        raise ValueError(name + " must be ON or OFF")
    return value in {"ON", "1", "TRUE"}


def blas_config(backend: str | None = None) -> tuple[str, str, list[str]]:
    """Select the BLAS backend: (EXTERNAL or BUILTIN, its macro, linker arguments)."""
    default = os.environ.get("SDCA_BLAS_BACKEND", "AUTO")
    requested = (backend or default).upper()
    if requested not in {"AUTO", "EXTERNAL", "BUILTIN"}:
        raise ValueError("SDCA_BLAS_BACKEND must be AUTO, EXTERNAL, or BUILTIN")
    if requested != "BUILTIN":
        override = os.environ.get("SDCA_BLAS")
        if override:
            return "EXTERNAL", "BLAS_DEFAULT_LOCAL_HEADER", shlex.split(override)
        for candidate in ("blas", "openblas"):
            name = ctypes.util.find_library(candidate)
            if not name:
                continue
            try:
                lib = ctypes.CDLL(name)
                for suffix in [
                    "scal",
                    "copy",
                    "swap",
                    "axpy",
                    "dot",
                    "asum",
                    "nrm2",
                    "gemv",
                    "ger",
                    "gemm",
                ]:
                    for precision in ["s", "d"]:
                        getattr(lib, "cblas_" + precision + suffix)
            except (OSError, AttributeError):
                continue
            if "/" not in name:
                link = ["-l:" + name]
            elif os.path.exists(name):
                link = [name]
            else:
                # macOS 11+: system libraries load from the dyld shared cache, not a file.
                link = ["-l" + os.path.basename(name).split(".")[0].removeprefix("lib")]
            return "EXTERNAL", "BLAS_DEFAULT_LOCAL_HEADER", link
        if requested == "EXTERNAL":
            raise RuntimeError(
                "CBLAS is unavailable; set SDCA_BLAS to linker flags or choose BUILTIN"
            )
    return "BUILTIN", "BLAS_BUILTIN", []
