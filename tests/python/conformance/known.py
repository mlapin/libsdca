"""Exact per-row/per-entry expected failures, not blanket cell exemptions.

Each recorded exception has a documented reason, result kind and error ceiling. A fixed
entry, an additional failing entry, a changed nonfinite result or a worsened error fails
the test. The records are reviewed artifacts; test execution never updates them.
"""

from __future__ import annotations

import functools
import json
import math
import platform
from pathlib import Path
from typing import Any

from conformance.contract import Failure


def result_kind(failure: Failure) -> str:
    if failure.entry.startswith("call:"):
        return "exception"
    if math.isnan(failure.actual):
        return "nan"
    if math.isinf(failure.actual):
        return "+inf" if failure.actual > 0 else "-inf"
    return "finite"


@functools.cache
def _records() -> dict[str, Any]:
    path = (
        Path(__file__).with_name("known_failures")
        / f"{platform.system()}-{platform.machine()}.json"
    )
    return json.loads(path.read_text()) if path.is_file() else {}


def manifest() -> dict[str, Any]:
    return _records().get("cells", {})


def variants() -> dict[str, Any]:
    """Exact alternative entry sets: the profiles of XLA GPU compilation choices and of
    other CPUs (AVX-512 kernels round the near-cancelling float64 gradients of the 4097-class
    Gaussian rows differently from AVX2 ones, and some x86 PyTorch matrix kernels
    flush bfloat16 subnormal gradients in the composed linear head)."""
    return _records().get("variants", {})


def verify(cell_id: str, failures: list[Failure]) -> list[str]:
    expected = manifest().get(cell_id, [])
    keys = {(f.row, f.entry) for f in failures}
    # Match a whole reviewed profile; never accept arbitrary subsets of a limit.
    for alternative in variants().get(cell_id, []):
        if keys == {(r[0], r[1]) for r in alternative}:
            expected = alternative
            break
    keyed = {(row, entry): (limit, kind, reason) for row, entry, limit, kind, reason in expected}
    actual: dict[tuple[int | None, str], list[Failure]] = {}
    for failure in failures:
        actual.setdefault((failure.row, failure.entry), []).append(failure)
    missing = keyed.keys() - actual.keys()
    extra = actual.keys() - keyed.keys()
    assert not missing, (
        f"Fixed conformance entries; narrow the documented manifest: {sorted(missing)}"
    )
    assert not extra, f"New conformance failures require investigation: {sorted(extra)}"
    for key, entries in actual.items():
        limit, kind, reason = keyed[key]
        assert all(result_kind(f) == kind for f in entries), (key, reason, entries)
        if limit is not None:
            assert max(f.ratio for f in entries) <= limit, (key, reason, limit, entries)
    return sorted({reason for _, _, _, _, reason in expected})
