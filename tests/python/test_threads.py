"""Native rows on several threads: bitwise the results of one thread."""

import ctypes
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any
from unittest import mock

import libsdca
import numpy as np
import pytest
from child_process import run_python
from libsdca import MULTILABEL, OBJECTIVES, PRECISIONS, Distillation, ProximalDistillation
from libsdca._core import (
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    TARGET_MASK,
    _cross_entropy_backward,
    _cross_entropy_forward,
    _Execution,
    _Loss,
    _make,
    _native,
    _Targets,
)
from numpy_api import Solver

LABEL_LOSSES = OBJECTIVES + ("proximal_cross_entropy",)

# 2600 x 101 scores take milliseconds per call: the default (automatic) and the
# explicit counts add threads.
N, C, THREADS = 2600, 101, (None, 2, 3, 8)

# A fit's evaluations and kernel check, in a child with SDCA_NUM_THREADS set or not.
FIT = """
import sys
import numpy as np
from libsdca import Kernel, fit
rng = np.random.default_rng(5)
x = rng.standard_normal((50000, 100))
y = rng.integers(0, 10, 50000)
gram = x[:3000, :20] @ x[:3000, :20].T
arrays = {}
features = fit(x, y, max_epoch=2, eval_epoch=1, eval_on_start=True)
kernel = fit(Kernel(gram), y[:3000], max_epoch=2, eval_epoch=1)
for name, model in (("features", features), ("kernel", kernel)):
    arrays[name + "_a"] = model.dual
    for field in ("primal", "dual", "relative_gap"):
        arrays[name + "_" + field] = model.history[field]
arrays["features_w"] = features.weights
np.savez(sys.argv[1], **arrays)
"""


# A long coordinate step with the default count, in a fresh process: its threads before and after.
GROW = """
import numpy as np
from libsdca import TopKHingeAlpha, coordinate_update
def threads():
    with open("/proc/self/status") as status:
        return next(int(line.split()[1]) for line in status if line.startswith("Threads:"))
rng = np.random.default_rng(6)
z, y = rng.standard_normal((10000, 101)), rng.integers(0, 101, 10000)
before = threads()
coordinate_update(z, y, TopKHingeAlpha(3))
print(before, threads())
"""


def _problem(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores (a few rows far from zero), labels and a mask of 1 to C-1 labels per row."""
    rng = np.random.default_rng(seed)
    z = 3 * rng.standard_normal((N, C))
    z[::97] += 1e6
    labels = rng.integers(0, C, N)
    mask = rng.random((N, C)) < 0.1
    mask[:, 0], mask[:, 1] = True, False
    return z, labels, mask


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


class ThreadTests:
    def test_losses_do_not_depend_on_the_thread_count(self, subtests: pytest.Subtests):
        z, labels, mask = _problem()
        for precision in PRECISIONS:
            solver = Solver(precision)
            for loss in LABEL_LOSSES:
                for target in [mask] if loss in MULTILABEL else [labels]:
                    options: dict[str, Any] = dict(k=3, gamma=0.5, alpha=0.7)
                    v1, g1 = solver.loss_and_grad(z, target, loss, num_threads=1, **options)
                    for threads in THREADS:
                        with subtests.test(precision=precision, loss=loss, threads=threads):
                            v, g = solver.loss_and_grad(
                                z, target, loss, num_threads=threads, **options
                            )
                            assert _same(v, v1) and _same(g, g1)
            teacher = z[::-1].copy()
            for loss in (Distillation(2.0), ProximalDistillation(2.0, 0.5)):
                v1, g1 = libsdca.loss_and_grad(z, teacher, loss, num_threads=1)
                for threads in THREADS:
                    with subtests.test(precision=precision, loss=loss, threads=threads):
                        v, g = libsdca.loss_and_grad(z, teacher, loss, num_threads=threads)
                        assert _same(v, v1) and _same(g, g1)

    def test_coordinate_steps_do_not_depend_on_the_thread_count(self, subtests: pytest.Subtests):
        z, labels, mask = _problem(1)
        previous = np.zeros_like(z)
        for precision in PRECISIONS:
            solver = Solver(precision)
            for objective in OBJECTIVES:
                y = mask if objective in MULTILABEL else labels
                options: dict[str, Any] = dict(previous=previous, C=2.0, k=3, gamma=0.5)
                a1 = solver.coordinate_update(z, y, objective, num_threads=1, **options)
                for threads in THREADS:
                    with subtests.test(precision=precision, objective=objective, threads=threads):
                        a = solver.coordinate_update(
                            z, y, objective, num_threads=threads, **options
                        )
                        assert _same(a, a1)

    def test_concurrent_callers_get_the_serial_results(self):
        z, labels, _ = _problem(2)
        solver = Solver()
        expected = solver.loss_and_grad(z, labels, "topk_hinge_beta", k=2, num_threads=1)
        results: list[tuple[np.ndarray, np.ndarray]] = []

        def call() -> None:
            for _ in range(3):
                values = solver.loss_and_grad(z, labels, "topk_hinge_beta", k=2, num_threads=None)
                results.append(values)

        callers = [threading.Thread(target=call) for _ in range(4)]
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join()
        assert len(results) == 12
        for v, g in results:
            assert _same(v, expected[0]) and _same(g, expected[1])

    def test_fit_evaluations_do_not_depend_on_the_thread_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = []
            for setting in ("1", None):
                env = dict(os.environ)
                env.pop("SDCA_NUM_THREADS", None)
                if setting is not None:
                    env["SDCA_NUM_THREADS"] = setting
                path = str(Path(tmp) / f"fit{setting}.npz")
                run_python("-c", FIT, path, env=env, check=True)
                results.append(np.load(path))
            for name in results[0].files:
                assert _same(results[0][name], results[1][name]), name

    @pytest.mark.serial
    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
    def test_the_default_adds_threads_to_a_long_call(self):
        if len(os.sched_getaffinity(0)) < 2:
            pytest.skip("one CPU")
        env = {
            k: v for k, v in os.environ.items() if k not in ("SDCA_NUM_THREADS", "OMP_NUM_THREADS")
        }
        out = run_python("-c", GROW, env=env, check=True, capture_output=True, text=True)
        before, after = map(int, out.stdout.split())
        assert after > before

    def test_an_invalid_row_leaves_the_outputs_as_given(self):
        z, labels, _ = _problem(3)
        mask = np.zeros((N, C), np.uint8)
        mask[np.arange(N), labels] = 1
        mask[2000] = 1  # Every class: multilabel_hinge needs a negative one.
        lib = _native().lib
        for threads in (1, 8):
            values, grad = np.full(N, 7.0), np.full_like(z, 7.0)
            loss = _make(_Loss, kind=5)
            execution = _make(_Execution, num_threads=threads)
            targets = _make(_Targets, kind=TARGET_MASK, mask=mask.ctypes.data)
            code = lib.sdca_loss_evaluate(
                ctypes.byref(loss),
                ctypes.byref(execution),
                DTYPE_FLOAT64,
                N,
                C,
                z.ctypes.data,
                ctypes.byref(targets),
                values.ctypes.data,
                grad.ctypes.data,
            )
            assert code == -2 and b"class outside" in lib.sdca_last_error()
            assert np.all(values == 7.0) and np.all(grad == 7.0)

    def test_internal_cross_entropy_matches_the_loss_rows(self, subtests: pytest.Subtests):
        z, labels, _ = _problem(4)
        z[5, (labels[5] + 1) % C] -= 2000  # a derivative below double's normal range
        for dtype, code in ((np.float64, DTYPE_FLOAT64), (np.float32, DTYPE_FLOAT32)):
            x = z.astype(dtype)
            v1, g1 = Solver("float64", dtype=dtype).loss_and_grad(x, labels, num_threads=1)
            for threads in THREADS:
                with subtests.test(dtype=dtype, threads=threads):
                    values, stats = np.empty(N), np.empty((N, 5))
                    _cross_entropy_forward(
                        x.ctypes.data, code, N, C, labels.ctypes.data, -100, None,
                        values.ctypes.data, stats.ctypes.data, num_threads=threads,
                    )  # fmt: skip
                    out, scale = np.empty_like(x), np.ones(1)
                    _cross_entropy_backward(
                        x.ctypes.data, code, N, C, labels.ctypes.data, -100, None,
                        stats.ctypes.data, scale.ctypes.data, 0, out.ctypes.data,
                        num_threads=threads,
                    )  # fmt: skip
                    assert _same(values, v1) and _same(out, g1)
                    assert stats[:, -1].tolist() == [float(i == 5) for i in range(N)]

    def test_thread_options_reach_the_native_entry_points(self, subtests: pytest.Subtests):
        native = _native()
        z, labels = np.zeros((2, 3)), np.array([0, 1])
        for threads in (None, 1, 4):
            expected = 0 if threads is None else threads
            with subtests.test(num_threads=threads):
                with mock.patch.object(native, "call", wraps=native.call) as call:
                    libsdca.loss_and_grad(z, labels, num_threads=threads)
                    assert call.call_args.args[0] == "sdca_loss_evaluate"
                    assert call.call_args.args[2]._obj.num_threads == expected
                    libsdca.coordinate_update(z, labels, num_threads=threads)
                    assert call.call_args.args[0] == "sdca_dual_update"
                    assert call.call_args.args[3]._obj.num_threads == expected
                    libsdca.fit(np.eye(2), labels, num_threads=threads, max_epoch=1)
                    assert call.call_args.args[0] == "sdca_model_fit"
                    assert call.call_args.args[3]._obj.num_threads == expected

    def test_num_threads_must_be_none_or_a_positive_integer(self, subtests: pytest.Subtests):
        z, labels = np.zeros((2, 3)), np.array([0, 1])
        for threads, error in [
            (-1, ValueError),
            (0, ValueError),
            (1.5, TypeError),
            (True, TypeError),
            ("2", TypeError),
        ]:
            with subtests.test(threads=threads):
                with pytest.raises(error, match="num_threads"):
                    libsdca.loss_and_grad(z, labels, num_threads=threads)
                with pytest.raises(error, match="num_threads"):
                    libsdca.coordinate_update(z, labels, num_threads=threads)
