"""Processes that need the same first-use build at once compile it once.

Each compile of a framework extension takes gigabytes of memory; four processes compiling
both extensions at once starved a 16 GB CI runner. The others wait on the build's lock file
and load the library the first one built. A stand-in compiler logs each call and sleeps, so
that the processes overlap.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from child_process import pythonpath

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="no fcntl lock on Windows")

COMPILER = """#!/bin/sh
case "$1" in --version) echo "stand-in 1.0"; exit 0 ;; esac
echo compile >> "{log}"
sleep 2
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then : > "$2"; fi
  shift
done
"""

CHILD = """
from libsdca import _extensions as e
build = e.Build(
    env="SDCA_TEST_BUILD", source="_jax_cpu.cpp", what="the test build",
    cflags=list, link=list, key=lambda: ["lock-test"], missing=lambda: None,
)
library, reason = e.load(build, str)
print(library, reason)
"""


def test_concurrent_processes_compile_a_build_once(tmp_path: Path) -> None:
    log = tmp_path / "compiles.log"
    compiler = tmp_path / "cxx"
    compiler.write_text(COMPILER.format(log=log))
    compiler.chmod(0o755)
    env = dict(os.environ, CXX=str(compiler), SDCA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("SDCA_TEST_BUILD", None)
    env["PYTHONPATH"] = pythonpath(env)
    children = [
        subprocess.Popen(
            [sys.executable, "-c", CHILD],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    outputs = [child.communicate(timeout=120) for child in children]
    for child, (out, err) in zip(children, outputs, strict=True):
        assert child.returncode == 0, err
        library, *reason = out.split()
        assert not reason, out
        assert Path(library).is_file()
    assert log.read_text().splitlines() == ["compile"]
