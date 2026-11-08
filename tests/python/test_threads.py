"""Native rows on several threads: bitwise the results of one thread."""

import ctypes
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
from child_process import run_python
from libsdca import LOSSES, OBJECTIVES, PRECISIONS
from libsdca._core import (
    _IDS,
    ENTROPY_STATS,
    MULTILABEL,
    Solver,
    _entropy_forward,
    _entropy_gradient,
    _options,
    _ptr,
)

# 2600 x 101 scores take milliseconds per call: the default (automatic) and the
# explicit counts add threads.
N, C, THREADS = 2600, 101, (None, 2, 3, 8, 0)

# A fit's evaluations and kernel check, in a child with SDCA_NUM_THREADS set or not.
FIT = """
import sys
import numpy as np
from libsdca import fit
rng = np.random.default_rng(5)
x = rng.standard_normal((50000, 100))
y = rng.integers(0, 10, 50000)
gram = x[:3000, :20] @ x[:3000, :20].T
arrays = {}
features = fit(x, y, max_epoch=2, eval_epoch=1, eval_on_start=True)
kernel = fit(gram, y[:3000], kernel=True, max_epoch=2, eval_epoch=1)
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
from libsdca import coordinate_update
def threads():
    with open("/proc/self/status") as status:
        return next(int(line.split()[1]) for line in status if line.startswith("Threads:"))
rng = np.random.default_rng(6)
z, y = rng.standard_normal((10000, 101)), rng.integers(0, 101, 10000)
before = threads()
coordinate_update(z, y, "topk_hinge_alpha", k=3)
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
            for loss in LOSSES:
                targets = [labels, mask] if loss == "cross_entropy" else [labels]
                if loss.startswith("multilabel") or loss == "proximal_cross_entropy":
                    targets = [mask]
                for target in targets:
                    options: dict[str, Any] = dict(k=3, gamma=0.5, alpha=0.7)
                    v1, g1 = solver.loss_and_grad(z, target, loss, num_threads=1, **options)
                    for threads in THREADS:
                        with subtests.test(precision=precision, loss=loss, threads=threads):
                            v, g = solver.loss_and_grad(
                                z, target, loss, num_threads=threads, **options
                            )
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
                values = solver.loss_and_grad(z, labels, "topk_hinge_beta", k=2, num_threads=0)
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

    def test_a_failing_row_leaves_the_outputs_as_given(self):
        z, labels, _ = _problem(3)
        mask = np.zeros((N, C), np.uint8)
        mask[np.arange(N), labels] = 1
        mask[2000, (labels[2000] + 1) % C] = 1  # Two targets: topk_hinge_alpha fails on this row.
        solver = Solver()
        for threads in (1, 8):
            values, grad = np.full(N, 7.0), np.full_like(z, 7.0)
            options = _options(loss=_IDS["topk_hinge_alpha"], num_threads=threads)
            args = (ctypes.byref(options), N, C, _ptr(z), _ptr(mask), _ptr(values), _ptr(grad))
            with pytest.raises(ValueError, match="exactly one target"):
                solver._call("sdca_loss_gradient", *args)
            assert np.all(values == 7.0) and np.all(grad == 7.0)

    def test_entropy_entry_points_match_the_loss_rows(self, subtests: pytest.Subtests):
        z, labels, _ = _problem(4)
        z[5, (labels[5] + 1) % C] -= 2000  # a derivative below double's normal range
        for dtype in (np.float64, np.float32):
            x = z.astype(dtype)
            v1, g1 = Solver(dtype=dtype).loss_and_grad(x, labels, num_threads=1)
            for threads in THREADS:
                with subtests.test(dtype=dtype, threads=threads):
                    values, stats = _entropy_forward(x, labels, threads, True)
                    assert stats is not None
                    out = np.empty_like(x)
                    _entropy_gradient(x, labels, stats, np.ones(N), threads, out)
                    assert _same(values, v1) and _same(out, g1)
                    assert _same(_entropy_forward(x, labels, threads, False)[0], v1)
                    assert stats[:, -1].tolist() == [float(i == 5) for i in range(N)]
            _, stats = _entropy_forward(x, labels, 8, True)
            assert stats is not None
            scale = np.linspace(-3.0, 3.0, N, dtype=np.float64)
            out = np.empty_like(x)
            _entropy_gradient(x, labels, stats, scale, 8, out)
            expected = (g1.astype(np.float64) * scale[:, None]).astype(dtype)
            assert np.abs(out - expected).max() <= 8 * np.finfo(dtype).eps

    def test_entropy_forward_fails_without_writing(self):
        z, labels, _ = _problem(5)
        bad = labels.copy()
        bad[2000] = C
        nan = z.copy()
        nan[2000, 3] = np.nan
        for x, y, message in ((z, bad, "Labels"), (nan, labels, "Logits must be finite")):
            values, stats = np.full(N, 7.0), np.full((N, ENTROPY_STATS), 7.0)
            options = _options(num_threads=8)
            args = (ctypes.byref(options), N, C, _ptr(x), _ptr(y), _ptr(values), _ptr(stats))
            with pytest.raises(ValueError, match=message):
                Solver()._call("sdca_entropy_forward", *args)
            assert np.all(values == 7.0) and np.all(stats == 7.0)

    def test_thread_options_reach_both_native_entry_points(self, subtests: pytest.Subtests):
        # The C++ barrier tests exercise actual workers and environment defaults;
        # here verify that the public bindings pass the requested ABI values.
        solver = Solver()
        z, labels = np.zeros((2, 3)), np.array([0, 1])
        for threads in (None, 0, 1, 4):
            expected = ctypes.c_size_t(-1).value if threads is None else threads
            with subtests.test(num_threads=threads):
                with mock.patch.object(solver, "_call", wraps=solver._call) as call:
                    solver.loss_and_grad(z, labels, num_threads=threads)
                    assert call.call_args is not None
                    assert call.call_args.args[0] == "sdca_loss_gradient_rows"
                    assert call.call_args.args[1]._obj.num_threads == expected
                    solver.coordinate_update(z, labels, num_threads=threads)
                    assert call.call_args.args[0] == "sdca_update_objective"
                    assert call.call_args.args[1]._obj.num_threads == expected

    def test_num_threads_must_be_none_or_a_nonnegative_integer(self, subtests: pytest.Subtests):
        z, labels = np.zeros((2, 3)), np.array([0, 1])
        invalid: list[Any] = [-1, 1.5, True, "2"]
        for threads in invalid:
            with subtests.test(threads=threads):
                with pytest.raises(ValueError, match="num_threads"):
                    Solver().loss_and_grad(z, labels, num_threads=threads)
                with pytest.raises(ValueError, match="num_threads"):
                    Solver().coordinate_update(z, labels, num_threads=threads)
