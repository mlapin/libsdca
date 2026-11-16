"""Observe the working precision reaching the C ABI after a fresh README install.

This companion uses only NumPy and the standard library (plus sklearn when requested).
It leaves README examples untouched and calls the real native functions. Proximal
distillation's fixed fp64 work is recorded as an explicit known exception.
"""

from __future__ import annotations

import dataclasses
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

# The argument index of struct sdca_execution in each public entry point.
_EXECUTION = {"sdca_loss_evaluate": 1, "sdca_dual_update": 2, "sdca_model_fit": 2}
# enum sdca_precision.
_WORK = {0: None, 1: "float32", 2: "float64"}


@contextmanager
def observe() -> Iterator[list[dict[str, object]]]:
    """Record the working precision reaching the native calls, without replacing them."""
    calls: list[dict[str, object]] = []
    native = _core._native()
    original = native.call

    def call(name: str, *args: Any) -> None:
        execution = args[_EXECUTION[name]]._obj
        calls.append({"entry_point": name, "precision": _WORK[execution.precision]})
        original(name, *args)

    with mock.patch.object(native, "call", call):
        yield calls


def operations(dtype: Any, sklearn: bool) -> dict[str, Callable[[str | None], tuple[Any, ...]]]:
    rng = np.random.default_rng(17)
    z = rng.normal(size=(4, 3)).astype(dtype)
    teacher = rng.normal(size=z.shape).astype(dtype)
    labels = np.array([0, 1, 2, 1])
    mask = np.eye(3, dtype=bool)[labels]
    x = rng.normal(size=(12, 3)).astype(dtype)
    y = np.arange(12) % 3
    params = {"k": 2, "gamma": 0.5, "alpha": 1.0, "temperature": 1.0}
    result: dict[str, Callable[[str | None], tuple[Any, ...]]] = {}
    for name, cls in libsdca.LOSSES.items():
        taken = {field.name for field in dataclasses.fields(cls)}  # pyright: ignore[reportArgumentType]
        loss = cls(**{key: value for key, value in params.items() if key in taken})
        target = (
            mask if name in libsdca.MULTILABEL else teacher if "distillation" in name else labels
        )
        result[name] = lambda precision, loss=loss, target=target: libsdca.loss_and_grad(
            z, target, loss, precision=precision
        )
    result["coordinate_update"] = lambda precision: (
        libsdca.coordinate_update(z, labels, precision=precision),
    )
    result["fit"] = lambda precision: (
        libsdca.fit(x, y, precision=precision, tol=0.1, max_epoch=100).weights,
    )
    if sklearn:
        from libsdca.sklearn import SDCAClassifier

        result["SDCAClassifier.fit"] = lambda precision: (
            SDCAClassifier(precision=precision, tol=0.1, max_iter=100, random_state=0)
            .fit(x, y)
            .coef_,
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
            observed = {row["precision"] or matching for row in calls}
            # Proximal distillation computes in double whatever the request (docs/API.md).
            exception = name == "proximal_distillation" and requested == "float32"
            assert observed == {requested}, (name, precision, calls)
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
