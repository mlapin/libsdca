"""Build the NumPy/ctypes library without Python headers or pybind11."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sdca_build import COMPILE_FLAGS, blas_config, blas_package, version


def build(backend: str | None = None, output: str | Path | None = None) -> Path:
    """Compile src/c_api.cpp into <output>/libsdca_c.so and return its path."""
    out = Path(output) if output else ROOT / "build" / "python"
    out.mkdir(parents=True, exist_ok=True)
    lib = out / "libsdca_c.so"
    cxx = shlex.split(os.environ.get("CXX", "g++"))
    selected, macros, link = blas_config(backend, cxx)
    cmd = cxx + [
        *COMPILE_FLAGS,
        "-shared",
        "-fPIC",
        *("-D" + macro for macro in macros),
        "-DSDCA_USE_CHRONO",
        f'-DSDCA_VERSION="{version()}"',
        "-I",
        str(ROOT / "include"),
        str(ROOT / "src/c_api.cpp"),
        "-o",
        str(lib),
        *link,
    ]
    subprocess.run(cmd, check=True)
    metadata = {
        "command": cmd,
        "backend": selected,
        "compiler": subprocess.check_output([cmd[0], "--version"], text=True).splitlines()[0],
    }
    if selected == "PACKAGE":
        # The BLAS package to load first (python/libsdca/_native.py).
        package = blas_package()
        metadata["blas_package"] = {k: package[k] for k in ("distribution", "module", "library")}
    lib.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return lib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blas-backend", choices=["AUTO", "EXTERNAL", "PACKAGE", "BUILTIN"])
    parser.add_argument("--output", type=Path, help="Directory for the native libraries")
    args = parser.parse_args()
    print(build(args.blas_backend, args.output))
