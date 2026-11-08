"""Line coverage of the C++ library and the Python package, with a floor.

Builds the C++ tests and the C ABI with --coverage, runs the C++ and Python
suites, and reports gcovr (include/sdca, src/c_api.cpp) and coverage.py
(python/libsdca) line coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP_FLOOR = 90
PYTHON_FLOOR = 90


def _python_sources() -> dict[str, str]:
    package = ROOT / "python/libsdca"
    return {
        path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }


def write_python_manifest(data: Path) -> None:
    """Record every Python package source before collecting coverage on a frozen checkout."""
    path = Path(str(data) + ".sources.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema": 1, "files": _python_sources()}, indent=2, sort_keys=True) + "\n"
    )


def _verify_python_manifest(data: Path) -> None:
    path = Path(str(data) + ".sources.json")
    if not path.is_file():
        raise ValueError(f"Missing source manifest: {path}; capture it before collecting coverage")
    manifest = json.loads(path.read_text())
    saved = manifest.get("files")
    if manifest.get("schema") != 1 or not isinstance(saved, dict):
        raise ValueError(f"Invalid coverage source manifest: {path}")
    current = _python_sources()
    different = sorted(
        name for name in saved.keys() | current.keys() if saved.get(name) != current.get(name)
    )
    if different:
        raise ValueError(f"Incompatible coverage sources in {path}: {', '.join(different)}")


def run(*command: str | Path, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True)


def _default_jobs() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return os.cpu_count() or 2


def cpp_tests(build: Path, jobs: int) -> None:
    """The C++ suites built with the built-in BLAS, so that it is covered too."""
    run(
        "cmake",
        "-S",
        ROOT,
        "-B",
        build / "cpp",
        "-DBUILD_TESTING=ON",
        "-DSDCA_BUILD_C_API=ON",
        "-DSDCA_BLAS_BACKEND=BUILTIN",
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DCMAKE_CXX_FLAGS=--coverage -O1 -fprofile-update=atomic",
        "-DCMAKE_C_FLAGS=--coverage -O1 -fprofile-update=atomic",
    )
    run("cmake", "--build", build / "cpp", "--parallel", str(jobs))
    run("ctest", "--test-dir", build / "cpp", "--output-on-failure", "--parallel", str(jobs))


def python_tests(build: Path) -> None:
    """The Python suites against C ABI libraries built with --coverage."""
    # Parallel ABI calls need atomic counters to keep branch counts consistent.
    env = dict(os.environ, CXX=os.environ.get("CXX", "g++") + " --coverage -fprofile-update=atomic")
    run(sys.executable, "python/build.py", "--output", build / "python", env=env)
    env = dict(
        os.environ,
        SDCA_LIBRARY_DIR=str(build / "python"),
        PYTHONPATH=str(ROOT / "python"),
        COVERAGE_FILE=str(build / ".coverage"),
        OPENBLAS_NUM_THREADS="1",
        # The paths without the CPU extension; the torch group runs again with it below.
        SDCA_CPU_EXTENSION="off",
    )
    source = "--source=libsdca"
    write_python_manifest(build / ".coverage")
    run(
        sys.executable,
        "-m",
        "coverage",
        "run",
        source,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "tests/python",
        "--group",
        "all",
        env=env,
    )
    env.update(SDCA_CPU_EXTENSION="on", TORCH_EXTENSIONS_DIR=str(build / "torch-extensions"))
    run(
        sys.executable, "-m", "coverage", "run", "--append", source,
        "-m", "pytest", "-p", "no:cacheprovider", "tests/python", "--group", "torch", env=env,
    )  # fmt: skip
    _verify_python_manifest(build / ".coverage")


def python_report(build: Path, additional: list[Path], *, collect_only: bool = False) -> float:
    """Combine CPU/GPU host coverage without changing the input data files."""
    inputs = [build / ".coverage", *additional]
    output = build / ".coverage-host"
    if any(path.resolve() == output.resolve() for path in inputs):
        raise ValueError("Input coverage data must differ from the combined output")
    for path in inputs:
        _verify_python_manifest(path)
    output.unlink(missing_ok=True)
    env = dict(os.environ, COVERAGE_FILE=str(output), COVERAGE_RCFILE=str(ROOT / "pyproject.toml"))
    run(sys.executable, "-m", "coverage", "combine", "--keep", *inputs, env=env)
    include = f"--include={ROOT / 'python/libsdca/*'}"
    report_file = build / ("python-provisional.json" if collect_only else "python.json")
    run(sys.executable, "-m", "coverage", "json", include, "-o", report_file, env=env)
    if collect_only:
        print(
            "PROVISIONAL Python coverage: collected data; the 90% release floor is not evaluated.",
            flush=True,
        )
    run(
        sys.executable, "-m", "coverage", "report", include, "--show-missing",
        *([] if collect_only else [f"--fail-under={PYTHON_FLOOR}"]), env=env,
    )  # fmt: skip
    return float(json.loads(report_file.read_text())["totals"]["percent_covered"])


def report(
    build: Path, additional: list[Path], *, collect_python: bool = False, cpp_only: bool = False
) -> None:
    run(
        sys.executable,
        "-m",
        "gcovr",
        "--root",
        ROOT,
        "--filter",
        "include/sdca/",
        "--filter",
        "src/c_api.cpp",
        # The inner loops of the tests run more than gcov's suspicious count.
        "--gcov-ignore-parse-errors=suspicious_hits.warn_once_per_file",
        "--merge-lines",
        "--print-summary",
        "--txt",
        build / "cpp.txt",
        "--fail-under-line",
        str(CPP_FLOOR),
        build,
    )
    if not cpp_only:
        python_report(build, additional, collect_only=collect_python)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/coverage")
    parser.add_argument("--jobs", type=int, default=_default_jobs(), help="Parallel build jobs")
    parser.add_argument("--report-only", action="store_true", help="Report an earlier run")
    parser.add_argument(
        "--collect-python",
        action="store_true",
        help="Enforce C++ coverage and collect provisional Python data without the combined Python floor",
    )
    parser.add_argument(
        "--cpp-only", action="store_true", help="Only the C++ suites and their coverage floor"
    )
    parser.add_argument(
        "--write-python-manifest",
        type=Path,
        help="Capture a source manifest for DATA_PATH before collecting external coverage; then exit",
    )
    parser.add_argument(
        "--python-data",
        type=Path,
        action="append",
        default=[],
        help="Additional Python coverage data from the same sources (repeatable, kept unchanged)",
    )
    args = parser.parse_args()
    if args.write_python_manifest:
        write_python_manifest(args.write_python_manifest)
        return
    if not args.report_only:
        cpp_tests(args.build, args.jobs)
        if not args.cpp_only:
            python_tests(args.build)
    report(args.build, args.python_data, collect_python=args.collect_python, cpp_only=args.cpp_only)


if __name__ == "__main__":
    main()
