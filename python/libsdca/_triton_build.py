"""Provide Triton's native utility compiler on Linux hosts without a C toolchain.

The Linux torch extra includes Zig's standalone compiler. Use Triton's supported
build hook only when no host compiler or user hook was selected; never change CC,
CXX, PATH, or the user's compiler choices. CUDA kernel compilation remains Triton's.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


@functools.cache
def ensure_compiler() -> None:
    """Install the compiler hook once, only for a compiler-less Linux process."""
    if sys.platform != "linux" or os.environ.get("CC") or os.environ.get("CXX"):
        return
    if shutil.which("gcc") or shutil.which("clang"):
        return
    import triton

    knobs = getattr(triton, "knobs", None)
    if knobs is None or knobs.build.impl is not None:
        return
    if importlib.util.find_spec("ziglang") is None:
        return
    knobs.build.impl = _build


def _build(
    name: str,
    source: str,
    directory: str,
    library_dirs: list[str],
    include_dirs: list[str],
    libraries: list[str],
) -> str:
    """Triton's build.impl protocol, compiling its native driver/launch utilities."""
    import triton

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    output = str(Path(directory) / f"{name}{suffix}")
    cpp = Path(source).suffix in (".cc", ".cpp", ".cxx")
    command = [
        sys.executable,
        "-m",
        "ziglang",
        "c++" if cpp else "cc",
        source,
        "-O3",
        "-shared",
        "-fPIC",
        "-o",
        output,
    ]
    if cpp:
        command += ["-std=c++17"]
    scheme = sysconfig.get_default_scheme()
    if scheme == "posix_local":
        scheme = "posix_prefix"
    includes = [
        *include_dirs,
        directory,
        sysconfig.get_paths(scheme=scheme)["include"],
        *triton.knobs.build.backend_dirs,
    ]
    command += [f"-I{path}" for path in includes if path is not None]
    command += [f"-L{path}" for path in library_dirs]
    for library in libraries:
        exact = re.search(r"\.so(?:\.\d+)*$", library) or library.endswith(".a")
        command.append(f"-l:{library}" if exact else f"-l{library}")
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[0] if result.stderr.strip() else ""
        raise RuntimeError(detail or f"zig exited with status {result.returncode}")
    return output
