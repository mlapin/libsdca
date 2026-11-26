"""Automatic threading (docs/API.md, Threads): with num_threads omitted and no
SDCA_NUM_THREADS or OMP_NUM_THREADS, a call with much work takes worker threads. The
timing-sensitive test is marked serial: CI runs it alone (-m serial, without -n)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest
from child_process import run_python

GROW = """
import numpy as np
from libsdca import TopKHinge, coordinate_update
def threads():
    with open("/proc/self/status") as status:
        return next(int(line.split()[1]) for line in status if line.startswith("Threads:"))
rng = np.random.default_rng(6)
z, y = rng.standard_normal((10000, 101)), rng.integers(0, 101, 10000)
before = threads()
coordinate_update(z, y, TopKHinge(k=3, variant="alpha"))
print(before, threads())
"""


@pytest.mark.serial
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_the_default_adds_threads_to_a_long_call() -> None:
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("one CPU")
    env = {k: v for k, v in os.environ.items() if k not in ("SDCA_NUM_THREADS", "OMP_NUM_THREADS")}
    out = run_python("-c", GROW, env=env, check=True, capture_output=True, text=True)
    before, after = map(int, out.stdout.split())
    assert after > before


def test_every_marker_ci_selects_marks_a_test() -> None:
    """A CI step that selects tests by one marker (pytest ... -m NAME) fails with pytest's exit
    code 5 when no test carries it: each such marker marks a test of tests/python."""
    workflows = Path(__file__).resolve().parents[2] / ".github" / "workflows"
    if not workflows.is_dir():
        pytest.skip("no workflows here (an sdist)")
    pattern = re.compile(r"pytest\b[^\n]*\s-m\s+(\w+)\b")
    selected = {m for p in workflows.glob("*.yml") for m in pattern.findall(p.read_text())}
    sources = "".join(p.read_text() for p in Path(__file__).parent.glob("test_*.py"))
    assert selected, "no marker selection found in the workflows"
    for marker in sorted(selected):
        assert f"mark.{marker}" in sources, f"CI selects -m {marker}; no test is marked {marker}"
