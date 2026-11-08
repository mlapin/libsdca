"""Run the lint, format, type and complexity checks of CONTRIBUTING.md.

    python tools/quality.py [--fix] [--skip-tidy]

Needs the tools of requirements-dev.txt on PATH or next to this interpreter.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_COMPLEXITY = 20
MAX_LINES = 100
COMPLEXITY_BASELINE = "tools/complexity_baseline.txt"


def run(*command: str) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def cpp_sources() -> list[str]:
    """The maintained C++ sources, without vendored code."""
    paths: list[Path] = [
        p for p in (ROOT / "python/libsdca").iterdir() if p.suffix in {".h", ".cpp"}
    ]
    for directory in ("include/sdca", "src", "tests", "tools"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.suffix in {".h", ".cpp"})
    vendored = {"mex.h"}
    return sorted(str(p.relative_to(ROOT)) for p in paths if p.name not in vendored)


def check_complexity() -> None:
    """Fail on a library function above the limits, unless baselined, and on stale entries."""
    import lizard

    # Library code only: tests are exempt.
    sources = [p for p in cpp_sources() if p.startswith(("include/sdca/", "src/", "python/"))]
    sources += sorted(str(p.relative_to(ROOT)) for p in (ROOT / "python/libsdca").glob("*.py"))
    found: dict[str, str] = {}
    for path in sources:
        for function in lizard.analyze_file(str(ROOT / path)).function_list:
            if function.cyclomatic_complexity > MAX_COMPLEXITY or function.nloc > MAX_LINES:
                found[f"{path} {function.name}"] = (
                    f"{path}:{function.start_line} {function.name}: complexity "
                    f"{function.cyclomatic_complexity}, {function.nloc} lines"
                )
    lines = (ROOT / COMPLEXITY_BASELINE).read_text().splitlines()
    baseline = {line for line in lines if line and not line.startswith("#")}
    failures = [found[key] for key in sorted(found.keys() - baseline)]
    failures += [
        f"{key}: not above the limits, remove it from {COMPLEXITY_BASELINE}"
        for key in sorted(baseline - found.keys())
    ]
    if failures:
        raise SystemExit("Complexity check failed:\n" + "\n".join(failures))


def gcc_toolchain() -> list[str]:
    """Point clang-tidy at the active GCC's standard library."""
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    crt = subprocess.check_output([*compiler, "-print-file-name=crtbegin.o"], text=True).strip()
    directory = Path(crt).parent
    return [f"--gcc-install-dir={directory}"] if (directory / "libstdc++.so").exists() else []


def check_tidy() -> None:
    toolchain = gcc_toolchain()
    for backend in ("BLAS_BUILTIN", "BLAS_DEFAULT_LOCAL_HEADER"):
        for source in ("src/c_api.cpp", "tests/cpp/regression.cpp"):
            run(
                "clang-tidy",
                "--quiet",
                source,
                "--",
                "-std=c++11",
                f"-D{backend}",
                '-DSDCA_VERSION="quality"',
                "-DSDCA_USE_CHRONO",
                "-Iinclude",
                "-Itests/cpp",
                "-isystem",
                "vendor/googletest/googletest/include",
                *toolchain,
            )


def _version(pattern: str, path: str) -> str:
    match = re.search(pattern, (ROOT / path).read_text(), re.M)
    if match is None:
        raise SystemExit(f"{path}: no version")
    return match.group(1)


def check_versions() -> None:
    """One version in CMake, the packages and builders, the C header and the changelog."""
    heading = r"^## (\S+) \((\d{4}-\d{2}-\d{2})\)$"
    changelog = re.search(heading, (ROOT / "CHANGELOG.md").read_text(), re.M)
    if changelog is None:
        raise SystemExit("CHANGELOG.md: no '## <version> (YYYY-MM-DD)' heading")
    date.fromisoformat(changelog.group(2))
    found = {
        "CHANGELOG.md": changelog.group(1),
        "include/sdca/c_api.h": _version(r'version, e\.g\. "([^"]+)"', "include/sdca/c_api.h"),
        "docs/C.md": _version(r"SDCA_VERSION=\"([^\"]+)\"", "docs/C.md"),
        "CMakeLists.txt": ".".join(
            _version(rf"libsdca_{part}_VERSION (\d+)", "CMakeLists.txt")
            for part in ("MAJOR", "MINOR", "PATCH")
        ),
        "pyproject.toml": _version(r'^version = "([^"]+)"', "pyproject.toml"),
        "rust/Cargo.toml": _version(r'^version = "([^"]+)"', "rust/Cargo.toml"),
        "setup.py": _version(r"SDCA_VERSION[^0-9]*([0-9.]+)", "setup.py"),
        "python/build.py": _version(r"SDCA_VERSION[^0-9]*([0-9.]+)", "python/build.py"),
        "matlab/make.m": _version(r"LIBSDCA_VERSION = '([^']+)'", "matlab/make.m"),
        "python/libsdca/__init__.py": _version(
            r'^__version__ = "([^"]+)"', "python/libsdca/__init__.py"
        ),
    }
    if len(set(found.values())) != 1:
        raise SystemExit(f"Versions differ: {found}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--fix", action="store_true", help="apply the formatters first")
    parser.add_argument("--skip-tidy", action="store_true", help="skip clang-tidy (slow)")
    args = parser.parse_args()
    # Prefer the tools installed next to this interpreter.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    if args.fix:
        run("ruff", "check", "--fix")
        run("ruff", "format")
        run("clang-format", "-i", *cpp_sources())
    run("ruff", "check")
    run("ruff", "format", "--check")
    run("clang-format", "--dry-run", "--Werror", *cpp_sources())
    run("basedpyright", "--pythonpath", sys.executable)
    check_versions()
    check_complexity()
    if not args.skip_tidy:
        check_tidy()


if __name__ == "__main__":
    main()
