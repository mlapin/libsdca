"""Syntax-compile the PyTorch CPU extension with each compiler and instruction set it builds for.

    python tools/check_extension_builds.py [--compiler CXX ...] [--jobs N]

Compiles python/libsdca/_torch_entropy.cpp with -fsyntax-only under the flags of
libsdca._cpu_entropy for every ISA flag set (ISA_FLAGS: AVX-512 and AVX2 on x86-64) and
the generic set that other CPUs (arm64, x86 without AVX2) take, with PyTorch's headers, by
each compiler (default: g++). Needs PyTorch in this interpreter; exits
nonzero on any error. Nothing is built or loaded.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "python/libsdca/_torch_entropy.cpp"


def capabilities() -> list[str]:
    """The ATen capabilities whose flags apply on this machine: the x86 sets, and generic."""
    from libsdca import _cpu_entropy

    x86 = platform.machine().lower() in ("x86_64", "amd64")
    return [*(_cpu_entropy.ISA_FLAGS if x86 else []), "DEFAULT"]


def common_flags(compiler: str) -> list[str]:
    """PyTorch's extension flags (its headers, C++ standard, ABI and macros), with GCC's OpenMP header
    after the system headers for a compiler without one (e.g. a bare Clang)."""
    import torch
    from torch.utils import cpp_extension

    includes = [*cpp_extension.include_paths(), sysconfig.get_paths()["include"]]
    # PyTorch's standard: C++17 before 2.12, C++20 from 2.12.
    major, minor = (int(part) for part in torch.__version__.split(".")[:2])
    std = "c++20" if (major, minor) >= (2, 12) else "c++17"
    flags = [f"-std={std}", "-fsyntax-only", "-DTORCH_API_INCLUDE_EXTENSION_H"]
    flags += ["-DTORCH_EXTENSION_NAME=libsdca_entropy_check", *(f"-I{p}" for p in includes)]
    abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None)
    if abi is not None:
        flags.append(f"-D_GLIBCXX_USE_CXX11_ABI={int(abi)}")
    gcc = shutil.which("gcc")
    if "clang" in Path(compiler).name and gcc:
        include = subprocess.run(
            [gcc, "-print-file-name=include"], capture_output=True, text=True, check=False
        ).stdout.strip()
        if (Path(include) / "omp.h").is_file():
            flags += ["-idirafter", include]
    return flags


def compile_once(compiler: str, capability: str) -> tuple[str, int, str]:
    """(label, exit status, compiler output) of one syntax-only compile."""
    from libsdca import _cpu_entropy

    command = [compiler, *common_flags(compiler), *_cpu_entropy._flags(capability), str(SOURCE)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    label = f"{compiler} {capability.lower() if capability != 'DEFAULT' else 'generic'}"
    return label, result.returncode, result.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--compiler", action="append", help="a C++ compiler (repeatable)")
    parser.add_argument("--jobs", type=int, default=0, help="compiles at once (0: all)")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "python"))
    compilers = args.compiler or ["g++"]
    missing = [c for c in compilers if shutil.which(c) is None]
    if missing:
        print(f"compilers not found: {missing}")
        return 1
    jobs = [(c, capability) for c in compilers for capability in capabilities()]
    with ThreadPoolExecutor(args.jobs or len(jobs)) as pool:
        results = list(pool.map(lambda job: compile_once(*job), jobs))
    failed = 0
    for label, status, output in results:
        print(f"{'PASS' if status == 0 else 'FAIL'} {label}", flush=True)
        if status:
            failed += 1
            print(output)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
