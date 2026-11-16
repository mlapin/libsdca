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


def compiler_toolchain() -> list[str]:
    """Point standalone clang-tidy at the platform's C++ standard library."""
    if sys.platform == "darwin":
        sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
        return ["-isysroot", sdk, "-isystem", str(Path(sdk) / "usr/include/c++/v1")]
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    crt = subprocess.check_output([*compiler, "-print-file-name=crtbegin.o"], text=True).strip()
    directory = Path(crt).parent
    return [f"--gcc-install-dir={directory}"] if (directory / "libstdc++.so").exists() else []


def check_tidy() -> None:
    toolchain = compiler_toolchain()
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
    """The copies of VERSION that cannot read it (tools/bump_version.py rewrites them): the
    changelog, the crate, the Python package, the citation and the package-page links; and
    one SDCA_ABI_VERSION in the header, the shared library, Python and Rust."""
    version = (ROOT / "VERSION").read_text().strip()
    heading = r"^## (\S+) \((\d{4}-\d{2}-\d{2})\)$"
    changelog = re.search(heading, (ROOT / "CHANGELOG.md").read_text(), re.M)
    if changelog is None:
        raise SystemExit("CHANGELOG.md: no '## <version> (YYYY-MM-DD)' heading")
    date.fromisoformat(changelog.group(2))
    found = {
        "CHANGELOG.md": changelog.group(1),
        "rust/Cargo.toml": _version(r'^version = "([^"]+)"', "rust/Cargo.toml"),
        "python/libsdca/__init__.py": _version(
            r'^__version__ = "([^"]+)"', "python/libsdca/__init__.py"
        ),
        "CITATION.cff": _version(r"^version: (\S+)$", "CITATION.cff"),
    }
    stale = {path: found for path, found in found.items() if found != version}
    if stale:
        raise SystemExit(f"VERSION is {version}, but {stale}")
    abi = {
        "include/sdca/c_api.h": _version(
            r"^#define SDCA_ABI_VERSION (\d+)$", "include/sdca/c_api.h"
        ),
        "CMakeLists.txt": _version(
            r"OUTPUT_NAME sdca_c VERSION (\d+) SOVERSION \1\b", "CMakeLists.txt"
        ),
        "python/libsdca/_core.py": _version(r"^ABI_VERSION = (\d+)", "python/libsdca/_core.py"),
        "rust/src/ffi.rs": _version(r"^pub const ABI_VERSION: c_int = (\d+);", "rust/src/ffi.rs"),
    }
    if len(set(abi.values())) != 1:
        raise SystemExit(f"SDCA_ABI_VERSION differs: {abi}")
    if _version(r"^date-released: (\S+)$", "CITATION.cff") != changelog.group(2):
        raise SystemExit("CITATION.cff: date-released differs from the CHANGELOG.md date")
    # PyPI and crates.io render these pages once per release, so their links pin its tag.
    for path in ("pyproject.toml", "docs/PYTHON.md", "docs/RUST.md"):
        text = (ROOT / path).read_text()
        links = set(re.findall(r"mlapin/libsdca(?:/blob/| --tag )v(\d[\w.-]*\w)", text))
        links.discard(version)
        if links:
            raise SystemExit(f"{path}: links to v{', v'.join(sorted(links))}, not v{version}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--fix", action="store_true", help="apply the formatters first")
    parser.add_argument("--skip-tidy", action="store_true", help="skip clang-tidy (slow)")
    args = parser.parse_args()
    run(sys.executable, "tools/check_docs.py")
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
