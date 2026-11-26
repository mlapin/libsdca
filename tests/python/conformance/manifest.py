"""Write known_failures/<platform>.json from a recorded run of the conformance cells.

    SDCA_CONFORMANCE_RECORD=DIR python -m pytest tests/python/test_conformance_*.py
    python tests/python/conformance/manifest.py [--variant] DIR

Every failing entry of the run that a documented limitation (cells.LIMITATIONS) covers becomes
a record of its row, entry, ratio ceiling (the recorded ratio, rounded up), result kind and
limitation; the uncovered entries are listed and left out, so their cells keep failing until
they are fixed or their limit is documented. The file's variants (known.variants) are kept.
With --variant, the run (of another CPU of the platform, such as AVX-512 under
SDCA_CPU_CAPABILITY or an emulator) adds each recorded cell's entries as a variant where they
differ from the cell's entries and variants, and leaves the cells as they are. Review the diff
before committing it.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformance.contract import Failure  # noqa: E402
from conformance.known import result_kind  # noqa: E402


def _ceiling(ratio: float) -> float | None:
    """The recorded ratio rounded up to two significant digits (None for NaN or inf)."""
    if not math.isfinite(ratio):
        return None
    scale = 10 ** (math.floor(math.log10(ratio)) - 1) if ratio > 0 else 1
    return math.ceil(ratio / scale) * scale


def _recorded(directory: str) -> tuple[dict[str, list[list[object]]], list[str]]:
    """The covered entries of each recorded cell (none for a cell that passes) and the
    uncovered failures."""
    cells: dict[str, list[list[object]]] = {}
    uncovered: list[str] = []
    for path in sorted(Path(directory).glob("**/*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            entries: dict[tuple[object, str], list[object]] = {}
            for raw in record["failures"]:
                failure = Failure(**raw)
                keys = record["covered"].get(str(failure.row), [])
                if not keys:
                    uncovered.append(f"{record['cell']}: {failure}")
                    continue
                key = (failure.row, failure.entry)
                limit = _ceiling(failure.ratio)
                previous = entries.get(key)
                if previous is not None and previous[2] is not None and limit is not None:
                    limit = max(limit, previous[2])  # pyright: ignore[reportArgumentType]
                entries[key] = [failure.row, failure.entry, limit, result_kind(failure), keys[0]]
            cells[record["cell"]] = sorted(entries.values(), key=lambda e: (e[0] or 0, e[1]))
    return cells, uncovered


def _keys(entries: list[list[object]]) -> set[tuple[object, object]]:
    return {(e[0], e[1]) for e in entries}


def main(directory: str, variant: bool = False) -> int:
    name = f"{platform.system()}-{platform.machine()}"
    target = Path(__file__).with_name("known_failures") / f"{name}.json"
    previous = json.loads(target.read_text()) if target.is_file() else {}
    variants: dict[str, list[list[list[object]]]] = previous.get("variants", {})
    recorded, uncovered = _recorded(directory)
    if variant:
        cells: dict[str, list[list[object]]] = previous.get("cells", {})
        for cell, entries in recorded.items():
            profiles = [cells.get(cell, []), *variants.get(cell, [])]
            if all(_keys(entries) != _keys(p) for p in profiles):
                variants.setdefault(cell, []).append(entries)
    else:
        cells = {cell: entries for cell, entries in recorded.items() if entries}

    def section(table: Mapping[str, object]) -> str:
        lines = [f" {json.dumps(k)}: {json.dumps(v)}" for k, v in sorted(table.items())]
        return ",\n".join(lines)

    text = f'{{"platform": {json.dumps(name)}, "cells": {{\n{section(cells)}\n}}'
    if variants:
        text += f', "variants": {{\n{section(variants)}\n}}'
    target.write_text(text + "}\n")
    print(
        f"{target}: {len(cells)} cells ({len(variants)} with variants); {len(uncovered)} uncovered"
    )
    for line in uncovered:
        print("  uncovered:", line)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(args[-1], variant="--variant" in args[:-1]))
