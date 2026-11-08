"""Options of the Python suites.

--group selects test files by their optional imports, without importing them: torch or jax
where a file imports either (or libsdca.torch / libsdca.jax), core otherwise. By default
every group whose package is installed runs. --installed tests a libsdca outside the source
tree and skips the tests of source-tree helpers.

Under pytest-xdist (-n) each worker gets an equal share of the CPUs as OMP_NUM_THREADS
(unless set), the CPU extension is built once before the workers start, tests marked slow
or long start first (one per worker under --dist worksteal), and tests marked serial are
deselected: run them in a second pass, -m serial without -n. CI passes -m "not long".
"""

from __future__ import annotations

import ast
import importlib.util
import itertools
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "python"
GROUPS = ("core", "torch", "jax")
# Tests of source-tree helpers, not of the installed package.
SOURCE_ONLY = {"test_build_config.py"}
SERIAL = "serial: needs the machine to itself (timing); deselected under -n, run with -m serial"
SLOW = "slow: takes about a minute on a loaded machine; started first under -n"
LONG = "long: over a minute on a 4-CPU runner; started first under -n, CI deselects it"
_LOCATION = pytest.StashKey[Path]()
# Builds (or loads) the CPU extension as the first CPU loss call of a test would.
PREBUILD = """
import libsdca._cpu_entropy as ext
ext.operators() if ext.mode() == "on" else ext.status()
"""


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--group",
        choices=[*GROUPS, "all"],
        help="core, torch, jax or all (default: core and the installed optional groups)",
    )
    parser.addoption("--installed", action="store_true", help="test the installed package")


def group(path: Path) -> str:
    """The group of a test file, from the modules it imports."""
    imports: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    found = "core"
    for name in ("torch", "jax"):
        if any(m == name or m.startswith(f"{name}.") or m == f"libsdca.{name}" for m in imports):
            found = name
    return found


def selected(config: pytest.Config) -> set[str]:
    option = config.getoption("group")
    if option == "all":
        return set(GROUPS)
    if option is not None:
        return {option}
    return {"core"} | {name for name in ("torch", "jax") if importlib.util.find_spec(name)}


def tested_libsdca(installed: bool) -> Path:
    """The libsdca that `import libsdca` finds; with installed, it must be outside python/."""
    import libsdca

    location = Path(libsdca.__file__).resolve()
    if installed and location.is_relative_to(SOURCE):
        raise pytest.UsageError(f"--installed imported libsdca from the source tree: {location}")
    return location


def available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1


def distributed(config: pytest.Config) -> bool:
    """Whether pytest-xdist runs the tests in workers (this process may be one of them)."""
    return bool(getattr(config.option, "numprocesses", None)) or hasattr(config, "workerinput")


def pytest_configure(config: pytest.Config) -> None:
    # Before NumPy loads: OpenBLAS and OpenMP size their pools when they start.
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    if hasattr(config, "workerinput") and "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(max(1, available_cpus() // workers))
    config.addinivalue_line("markers", SERIAL)
    config.addinivalue_line("markers", SLOW)
    config.addinivalue_line("markers", LONG)
    config.stash[_LOCATION] = tested_libsdca(bool(config.getoption("installed")))


def pytest_sessionstart(session: pytest.Session) -> None:
    """Under -n with the torch tests, build the CPU extension before xdist starts the workers
    (else each builds it)."""
    config = session.config
    if not distributed(config) or hasattr(config, "workerinput") or config.option.collectonly:
        return
    mode = os.environ.get("SDCA_CPU_EXTENSION", "").strip().lower()
    if mode in ("off", "0") or "torch" not in selected(config):
        return
    if importlib.util.find_spec("torch") is None:
        return
    from child_process import run_python

    built = run_python("-c", PREBUILD, capture_output=True, text=True)
    if built.returncode:
        print(
            f"conftest: the CPU extension did not build:\n{built.stderr[-2000:]}", file=sys.stderr
        )


def pytest_report_header(config: pytest.Config) -> list[str]:
    groups = ", ".join(name for name in GROUPS if name in selected(config))
    lines = [f"Testing {config.stash[_LOCATION]}", f"Test groups: {groups}"]
    if distributed(config):
        lines.append("Tests marked serial are deselected: run them with -m serial, without -n")
    return lines


def slow_first(items: list[pytest.Item], workers: int) -> list[pytest.Item]:
    """items with the slow ones at the start of the contiguous runs that --dist worksteal
    hands the workers (run k: the items left over divided by the workers left)."""

    def is_slow(item: pytest.Item) -> bool:
        return any(item.get_closest_marker(name) for name in ("slow", "long"))

    slow = [item for item in items if is_slow(item)]
    rest = iter([item for item in items if not is_slow(item)])
    ordered, left = [], len(items)
    for k in range(workers):
        size = left // (workers - k)
        left -= size
        first = slow[k::workers]
        ordered += first + list(itertools.islice(rest, max(0, size - len(first))))
    return ordered + list(rest)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not distributed(config):
        return
    serial = [item for item in items if item.get_closest_marker("serial")]
    if serial:
        config.hook.pytest_deselected(items=serial)
    kept = [item for item in items if not item.get_closest_marker("serial")]
    items[:] = slow_first(kept, int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")))


@pytest.fixture(autouse=True)
def _gradual_underflow() -> Iterator[None]:
    """Gradual underflow after each test: on PyTorch 2.5 the first CPU torch.compile turns
    flush-to-zero on for the whole process, which the later tests of a worker would inherit."""
    yield
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_flush_denormal(False)


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    collection_path = collection_path.resolve()
    if collection_path.parent != HERE or not collection_path.match("test_*.py"):
        return None
    if config.getoption("installed") and collection_path.name in SOURCE_ONLY:
        return True
    return True if group(collection_path) not in selected(config) else None
