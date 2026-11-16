"""Observe native precision at the C ABI after a fresh README install.

This companion uses only NumPy and the standard library (plus sklearn when requested).
It leaves README examples untouched and calls the real native functions. Proximal
distillation's currently fixed fp64 work is recorded as an explicit known exception.
"""

from __future__ import annotations

import ctypes as ct
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import libsdca
import numpy as np
from libsdca import _core


@contextmanager
def observe() -> Iterator[list[dict[str, object]]]:
    """Record options reaching the native call, without replacing its computation."""
    calls: list[dict[str, object]] = []
    original_call, original_entry = _core.Solver._call, _core._tensor_entry

    def record(name: str, options: Any) -> None:
        code = ct.cast(options, ct.POINTER(_core._Options)).contents.precision
        calls.append({"entry_point": name, "precision_code": code})

    def call(solver: _core.Solver, name: str, *args: Any) -> None:
        record(name, args[0])
        original_call(solver, name, *args)

    def entry(name: str, library: Any = None) -> tuple[Any, Any]:
        lib, function = original_entry(name, library)

        def wrapped(*args: Any) -> Any:
            record(name, args[0])
            return function(*args)

        return lib, wrapped

    with (
        mock.patch.object(_core.Solver, "_call", call),
        mock.patch.object(_core, "_tensor_entry", entry),
    ):
        yield calls


def operations(dtype: Any, sklearn: bool) -> dict[str, Callable[[str | None], tuple[Any, ...]]]:
    rng = np.random.default_rng(17)
    z = rng.normal(size=(4, 3)).astype(dtype)
    teacher = rng.normal(size=z.shape).astype(dtype)
    labels = np.array([0, 1, 2, 1])
    mask = np.eye(3, dtype=bool)[labels]
    x = rng.normal(size=(12, 3)).astype(dtype)
    y = np.arange(12) % 3
    result: dict[str, Callable[[str | None], tuple[Any, ...]]] = {}
    for loss in libsdca.LOSSES:
        target = mask if loss in libsdca.MULTILABEL else labels
        result[loss] = lambda precision, loss=loss, target=target: libsdca.loss_and_grad(
            z, target, loss, k=2 if loss in libsdca.TOPK else 1, gamma=0.5, precision=precision
        )
    result["distillation"] = lambda precision: libsdca.distillation_loss_and_grad(
        z, teacher, precision=precision
    )
    result["proximal_distillation"] = lambda precision: libsdca.proximal_distillation_loss_and_grad(
        z, teacher, precision=precision
    )
    result["coordinate_update"] = lambda precision: (
        libsdca.coordinate_update(z, labels, precision=precision),
    )
    result["fit"] = lambda precision: (
        libsdca.fit(x, y, precision=precision, tol=0.1, max_epoch=100).weights,
    )
    result["Solver.loss_and_grad"] = lambda precision: libsdca.Solver(
        precision, dtype=dtype
    ).loss_and_grad(z, labels)
    if sklearn:
        from libsdca import SDCAClassifier

        result["SDCAClassifier.fit"] = lambda precision: (
            SDCAClassifier(precision=precision, tol=0.1, max_epoch=100).fit(x, y).coef_,
        )
    return result


def check(dtype: Any, sklearn: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    matching = "float32" if dtype == np.float32 else "float64"
    for name, operation in operations(dtype, sklearn).items():
        outputs = []
        for precision in (None, matching, *(["float64"] if dtype == np.float32 else [])):
            with observe() as calls:
                arrays = operation(precision)
            assert calls, f"{name} did not reach a native entry point"
            requested = matching if precision is None else precision
            observed = {"float32" if row["precision_code"] == 0 else "float64" for row in calls}
            exception = (
                name == "proximal_distillation"
                and requested == "float32"
                and observed == {"float64"}
            )
            assert observed == {requested} or exception, (name, precision, calls)
            assert all(array is not None and np.isfinite(array).all() for array in arrays), name
            assert arrays[-1].dtype == dtype, (name, arrays[-1].dtype, dtype)
            outputs.append(arrays)
            rows.append(
                {
                    "api": name,
                    "input_dtype": matching,
                    "requested_precision": precision,
                    "native_calls": calls,
                    "status": "KNOWN_FP64_EXCEPTION" if exception else "PASS",
                }
            )
        for automatic, explicit in zip(outputs[0], outputs[1], strict=True):
            np.testing.assert_array_equal(automatic, explicit)
    return rows


def main() -> None:
    rows = check(np.float32, "--sklearn" in sys.argv) + check(np.float64, "--sklearn" in sys.argv)
    Path("precision.json").write_text(json.dumps(rows, indent=2) + "\n")
    exceptions = [row for row in rows if row["status"] != "PASS"]
    print(
        "native precision:",
        len(rows) - len(exceptions),
        "PASS;",
        len(exceptions),
        "known fp64 exceptions",
    )
    for row in exceptions:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
