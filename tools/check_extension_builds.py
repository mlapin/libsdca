"""Syntax-compile the PyTorch CPU extension with each compiler.

    python tools/check_extension_builds.py [--compiler CXX ...] [--jobs N]

Compiles python/libsdca/_torch_cpu.cpp with -fsyntax-only under the flags of its first-use
build (libsdca._torch_ops), with PyTorch's headers, by each compiler (default: g++). One
translation unit holds every instruction set, so one compile per compiler covers them. Needs
PyTorch in this interpreter; exits nonzero on any error. Nothing is built or loaded.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compile_once(compiler: str) -> tuple[str, int, str]:
    """(compiler, exit status, compiler output) of one syntax-only compile."""
    from libsdca import _extensions, _torch_ops

    build = _torch_ops._cpu_build()
    flags = [f for f in _extensions.FLAGS if f != "-shared"]
    command = [
        compiler,
        *flags,
        "-fsyntax-only",
        f"-I{ROOT / 'include'}",
        f"-I{_extensions.HERE}",
        *build.cflags(),
        str(_extensions.HERE / build.source),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return compiler, result.returncode, result.stderr


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
    with ThreadPoolExecutor(args.jobs or len(compilers)) as pool:
        results = list(pool.map(compile_once, compilers))
    failed = 0
    for label, status, output in results:
        print(f"{'PASS' if status == 0 else 'FAIL'} {label}", flush=True)
        if status:
            failed += 1
            print(output)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
