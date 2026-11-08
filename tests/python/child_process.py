"""Run Python children that import the same libsdca as this test process."""

from __future__ import annotations

import os
import site
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def pythonpath(base: dict[str, str]) -> str:
    """The imported libsdca's parent (unless a site directory), these tests, then base's."""
    import libsdca

    package = Path(libsdca.__file__).resolve().parents[1]
    sites = {Path(p).resolve() for p in [*site.getsitepackages(), site.getusersitepackages()]}
    inherited = [str(Path(p).resolve()) for p in base.get("PYTHONPATH", "").split(os.pathsep) if p]
    first = [] if package in sites else [str(package)]
    return os.pathsep.join([*first, str(HERE), *inherited])


def run_python(
    *args: str, env: dict[str, str] | None = None, **kwargs: Any
) -> subprocess.CompletedProcess[Any]:
    """Run `python *args` in an empty directory: the caller's cwd cannot shadow libsdca."""
    base = dict(os.environ if env is None else env)
    base["PYTHONPATH"] = pythonpath(base)
    with tempfile.TemporaryDirectory() as cwd:
        return subprocess.run([sys.executable, *args], cwd=cwd, env=base, **kwargs)
