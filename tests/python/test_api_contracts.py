"""Contracts of the public Python API and the native C ABI behind it."""

import ctypes as ct
import inspect
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import libsdca
import numpy as np
import pytest
from child_process import run_python
from libsdca import HINGES, HISTORY, LOSSES, OBJECTIVES, PRECISIONS, STATUS, Model
from libsdca._core import ABI_VERSION, Solver, _fit_result, _Options, _options, _ptr

MISSING = "/does-not-exist/libsdca_missing.so"


class OptionContractTests:
    def test_options_are_validated_before_loading(self):
        with pytest.raises(ValueError, match="must be one"):
            Solver(library=MISSING, precision="float16")
        with pytest.raises(TypeError):  # one native build: no mode
            Solver(library=MISSING, mode="strict")  # pyright: ignore[reportCallIssue]

    def test_missing_library_error_names_the_fix(self):
        with pytest.raises(OSError, match="build.py"):
            Solver(library=MISSING)

    def test_explicit_library_serves_every_method(self):
        # Without a default library, a Solver given one must not look for another.
        code = (
            "import sys, numpy as np\n"
            "from libsdca._core import Solver\n"
            "solver = Solver(library=sys.argv[1])\n"
            "z, t = np.array([[0.0, -1.0, 2.0]]), np.array([[1.0, 0.0, 0.5]])\n"
            "solver.loss_and_grad(z, [0], 'cross_entropy')\n"
            "for alpha in (0.0, 0.5):\n"
            "    solver.distillation_loss_and_grad(z, t, alpha=alpha)\n"
        )
        env = dict(os.environ, SDCA_LIBRARY_DIR="/does-not-exist")
        library = Solver().lib._name
        run_python("-c", code, library, env=env, check=True)

    def test_import_loads_neither_torch_nor_the_library(self):
        code = (
            "import ctypes, sys\n"
            "def fail(*a, **k): raise AssertionError('library loaded on import')\n"
            "ctypes.CDLL = fail\n"
            "import libsdca\n"
            "assert 'torch' not in sys.modules\n"
        )
        run_python("-c", code, check=True)

    def test_child_processes_import_this_libsdca(self):
        # The source package next to the tests must not shadow the tested one.
        source = Path(__file__).resolve().parents[2] / "python"
        start = Path.cwd()
        for cwd in [start, source, Path(libsdca.__file__).resolve().parents[1]]:
            if not cwd.is_dir():
                continue
            os.chdir(cwd)
            try:
                code = "import libsdca; print(libsdca.__file__)"
                child = run_python("-c", code, check=True, capture_output=True, text=True)
            finally:
                os.chdir(start)
            assert Path(child.stdout.strip()).resolve() == Path(libsdca.__file__).resolve(), cwd


class VocabularyContractTests:
    """Names and ids shared with the C ABI, Matlab and Rust interfaces."""

    def test_objective_and_loss_ids(self):
        assert list(LOSSES) == [
            "cross_entropy",
            "topk_cross_entropy",
            "multiclass_hinge",
            "topk_hinge_alpha",
            "topk_hinge_beta",
            "multilabel_hinge",
            "multilabel_cross_entropy",
            "proximal_cross_entropy",
        ]
        assert OBJECTIVES == LOSSES[:7]
        assert HINGES == {
            "multiclass_hinge",
            "topk_hinge_alpha",
            "topk_hinge_beta",
            "multilabel_hinge",
        }
        x = np.array([[1.0, 0.2], [-0.3, 1.0], [0.7, -0.4], [-0.2, -0.8]])
        y = np.array([0, 1, 2, 0])
        assert libsdca.fit(x, y, "topk_hinge", k=2, max_epoch=2).objective == "topk_hinge_beta"
        v_alias, g_alias = libsdca.loss_and_grad(x @ x.T, y, "topk_hinge", k=2)
        v_beta, g_beta = libsdca.loss_and_grad(x @ x.T, y, "topk_hinge_beta", k=2)
        np.testing.assert_array_equal(v_alias, v_beta)
        np.testing.assert_array_equal(g_alias, g_beta)
        # Soft-target distillation fit, coordinate_update and loss wrappers.
        q = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6], [0.6, 0.3, 0.1]])
        distill_model = libsdca.fit(x, q, "distillation", C=2.0, tol=1e-6)
        assert distill_model.status == "solved" and distill_model.objective == "distillation"
        np.testing.assert_allclose(distill_model.predict_proba(x).sum(axis=1), 1.0, atol=1e-12)
        a_step = libsdca.coordinate_update(x @ distill_model.weights, q, "distillation", C=2.0)
        np.testing.assert_allclose(a_step.sum(axis=1), 0.0, atol=1e-14)
        dv, dg = libsdca.distillation_loss_and_grad(
            x @ distill_model.weights, x @ distill_model.weights
        )
        np.testing.assert_allclose(dg, 0.0, atol=1e-15)
        pv, pg = libsdca.proximal_distillation_loss_and_grad(
            x @ distill_model.weights, x @ distill_model.weights, temperature=1.0, alpha=0.5
        )
        np.testing.assert_allclose(pv, dv, atol=1e-14)
        np.testing.assert_allclose(pg, 0.0, atol=1e-15)

    def test_status_history_and_precision(self):
        assert STATUS == (
            "none",
            "solving",
            "solved",
            "no_progress",
            "max_epoch",
            "max_cpu_time",
            "max_wall_time",
            "failed",
        )
        assert Solver().lib.sdca_history_fields() == len(HISTORY)
        assert Solver().lib.sdca_abi_version() == ABI_VERSION
        assert HISTORY[:3] == ("epoch", "primal", "dual")
        assert list(PRECISIONS) == ["float32", "float64"]

    def test_signatures(self, subtests: pytest.Subtests):
        def names(f: Callable[..., object]) -> list[tuple[str, object]]:
            return [(p.name, p.default) for p in inspect.signature(f).parameters.values()][1:]

        empty = inspect.Parameter.empty
        assert names(Solver.fit) == [
            ("X", empty),
            ("y", empty),
            ("objective", "cross_entropy"),
            ("C", 1.0),
            ("k", 1),
            ("gamma", 0.0),
            ("num_classes", None),
            ("max_epoch", 1000),
            ("eval_epoch", 10),
            ("tol", 1e-3),
            ("eval_on_start", False),
            ("recompute_primal", True),
            ("projection_precision", "float64"),
            ("kernel", False),
            ("center", None),
            ("initial_dual", None),
            ("shrinking", "auto"),
        ]
        assert names(Solver.loss_and_grad) == [
            ("logits", empty),
            ("target", empty),
            ("loss", "cross_entropy"),
            ("k", 1),
            ("gamma", 0.0),
            ("alpha", 1.0),
            ("num_threads", None),
            ("projection_precision", "float64"),
        ]
        assert [n for n, _ in names(Solver.coordinate_update)] == [
            "logits",
            "y",
            "objective",
            "previous",
            "C",
            "norm2",
            "k",
            "gamma",
            "num_threads",
            "projection_precision",
        ]
        # The module functions take the method arguments plus precision.
        for name in (
            "fit",
            "loss_and_grad",
            "coordinate_update",
            "distillation_loss_and_grad",
        ):
            with subtests.test(name=name):
                method = getattr(Solver, name)
                function = inspect.signature(getattr(libsdca, name)).parameters.values()
                assert [(p.name, p.default) for p in function] == names(method) + [
                    ("precision", None)
                ]
        assert list(Model.__dataclass_fields__) == [
            "weights",
            "dual",
            "history",
            "status",
            "epochs",
            "kernel",
            "objective",
            "params",
        ]


class PrecisionContractTests:
    def test_output_dtypes(self, subtests: pytest.Subtests):
        x = np.array([[1.0, 0.2], [-0.3, 1.0], [0.7, -0.4]])
        y = np.array([0, 1, 2])
        for (precision, dtype_opt), dtype in (
            (("float32", None), np.float32),
            (("float64", np.float32), np.float32),
            (("float64", None), np.float64),
            ((np.float32, None), np.float32),
            ((np.float64, None), np.float64),
            (("fp32", None), np.float32),
            (("fp64", None), np.float64),
        ):
            solver = Solver(precision, dtype=dtype_opt)
            with subtests.test(precision=precision, dtype=dtype_opt):
                values, grad = solver.loss_and_grad(x @ x.T, y)
                assert (values.dtype, grad.dtype) == (np.float64, dtype)
                assert solver.coordinate_update(x @ x.T, y).dtype == dtype
                model = solver.fit(x, y, max_epoch=3)
                assert model.weights is not None
                assert (model.weights.dtype, model.dual.dtype) == (dtype, dtype)

    def test_float64_compute_solves_in_double_and_float32_in_float(self, subtests: pytest.Subtests):
        rng = np.random.default_rng(3)
        z = (rng.standard_normal((200, 12)) * 30).astype(np.float32)
        y = rng.integers(0, 12, 200)
        for objective, k in (("cross_entropy", 1), ("topk_cross_entropy", 3)):
            opts: dict[str, Any] = dict(C=10.0, objective=objective, k=k)
            exact = Solver("float64").coordinate_update(z.astype(np.float64), y, **opts)
            scale = np.abs(exact).max(axis=1, keepdims=True)
            ulp = np.finfo(np.float32).eps * scale
            double_on_fp32 = Solver("float64", dtype=np.float32).coordinate_update(z, y, **opts)
            single = Solver("float32").coordinate_update(z, y, **opts)
            with subtests.test(objective=objective):
                assert np.max(np.abs(double_on_fp32 - exact) / ulp) <= 8
                bound = 64 * 12 + 16 * np.abs(z).max()
                assert np.max(np.abs(single - exact) / ulp) <= bound


class BoundaryContractTests:
    def test_warm_start_signs_are_exact(self):
        with pytest.raises(ValueError, match="non-label entries"):
            Solver().fit([[1.0]], [0], num_classes=2, initial_dual=[[-1e-16, 1e-16]])

    def test_oversized_limits_and_boolean_options_are_rejected(self, subtests: pytest.Subtests):
        solver = Solver()
        with pytest.raises(ValueError, match="too large for the history buffer"):
            solver.fit([[1.0]], [0], num_classes=2, max_epoch=sys.maxsize, eval_epoch=1)
        with pytest.raises(ValueError, match="native BLAS integer limit"):
            solver.fit([[1.0, 2.0]], [0], num_classes=sys.maxsize)
        invalid: list[dict[str, Any]] = [
            {"num_classes": True},
            {"max_epoch": True},
            {"eval_epoch": False},
        ]
        for bad in invalid:
            with subtests.test(bad=bad), pytest.raises(ValueError):
                solver.fit([[1.0]], [0], **bad)

    def test_coordinate_update_output_is_valid_warm_start(self):
        rng = np.random.default_rng(2026)
        scores = rng.normal(0, 10, (200, 10)).astype(np.float32)
        y = np.zeros(200, dtype=int)
        for precision in ("float32", "float64"):
            solver = Solver(precision, dtype=np.float32)
            a = solver.coordinate_update(scores, y, C=10.0, objective="topk_cross_entropy", k=3)
            model = solver.fit(
                np.ones((200, 1), dtype=np.float32),
                y,
                C=10.0,
                num_classes=10,
                objective="topk_cross_entropy",
                k=3,
                initial_dual=a,
                max_epoch=0,
            )
            assert model.status == "max_epoch"

    def test_entropy_loss_takes_every_class_as_positive(self, subtests: pytest.Subtests):
        solver = Solver()
        for target in ([[0, 1]], [[True, True]], np.ones((1, 2), bool)):
            with subtests.test(target=target):
                value, grad = solver.loss_and_grad([[0.0, 0.0]], target)
                np.testing.assert_allclose(value, [np.log(2)], rtol=1e-15)
                np.testing.assert_allclose(grad, 0, atol=1e-15)
        with pytest.raises(ValueError, match="needs a negative label"):
            solver.loss_and_grad([[0.0, 0.0]], [[0, 1]], loss="multilabel_hinge")


class ModelContractTests:
    def test_prediction(self):
        weights = np.array([[1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
        x = np.array([[1.0, 0.0], [0.0, 2.0]])
        feature = Model(weights, np.zeros((2, 3)), {}, "solved", 1, False)
        np.testing.assert_array_equal(feature.predict(x), [0, 2])
        np.testing.assert_array_equal(feature.decision_function(x), x @ weights)
        kernel = Model(None, weights, {}, "max_epoch", 1, True)
        np.testing.assert_array_equal(kernel.decision_function(x), x @ weights)
        with pytest.raises(ValueError, match="needs weights"):
            Model(None, weights, {}, "none", 0, False).decision_function(x)


class NativeAbiContractTests:
    """Direct ABI callers: errors are returned, never thrown, and leave outputs as given."""

    def setup_method(self):
        self.solver = Solver()
        self.z = np.array([[1.0, 2.0, 3.0]])
        self.y = np.array([0], np.int64)

    def assert_error(self, name: str, message: str, *args: object) -> None:
        with pytest.raises(ValueError, match=re.escape(message)):
            self.solver._call(name, *args)
        assert self.solver.lib.sdca_last_error().decode() == message
        self.solver.loss_and_grad(self.z, self.y)
        assert self.solver.lib.sdca_last_error() == b""

    def calls(self, options: object, result: object = None) -> dict[str, tuple[object, ...]]:
        """One valid call of every computing export, with these options (and fit result)."""
        z, y = self.z, self.y
        mask = np.array([[1, 0, 0]], np.uint8)
        a, values = np.zeros_like(z), np.zeros(1)
        hist = np.zeros((2, len(HISTORY)))
        result = ct.byref(_fit_result()) if result is None else result
        x = (_ptr(z), _ptr(y), None, None, _ptr(a), _ptr(a), _ptr(hist), len(hist), result)
        return {
            "sdca_loss_gradient": (options, 1, 3, _ptr(z), _ptr(mask), _ptr(values), _ptr(a)),
            "sdca_update_objective": (options, 1, 3, _ptr(z), _ptr(y), None, _ptr(a), 1.0, _ptr(a)),
            "sdca_fit": (options, 1, 1, 3, *x),
        }

    def test_every_export_translates_exceptions(self, subtests: pytest.Subtests):
        options = _options(precision=9, max_epoch=1, eval_epoch=1)
        for name, args in self.calls(ct.byref(options)).items():
            with subtests.test(name=name):
                self.assert_error(name, "Invalid precision", *args)

    def test_structs_need_this_headers_size(self, subtests: pytest.Subtests):
        for size in (0, ct.sizeof(_Options) - 4, ct.sizeof(_Options) + 8):
            options = _options(max_epoch=1, eval_epoch=1)
            options.size = size
            for name, args in self.calls(ct.byref(options)).items():
                with subtests.test(name=name, size=size):
                    message = "options size must be sizeof(struct sdca_options) of this header"
                    self.assert_error(name, message, *args)
            result = _fit_result()
            result.size = size
            fit = self.calls(ct.byref(_options(max_epoch=1, eval_epoch=1)), ct.byref(result))
            message = "fit_result size must be sizeof(struct sdca_fit_result) of this header"
            self.assert_error("sdca_fit", message, *fit["sdca_fit"])
        for name, args in self.calls(None).items():
            with subtests.test(name=name):
                self.assert_error(name, "options must not be NULL", *args)
        fit = self.calls(ct.byref(_options(max_epoch=1, eval_epoch=1)), None)["sdca_fit"]
        self.assert_error("sdca_fit", "fit_result must not be NULL", *fit[:-1], None)

    def test_fit_checks_the_history_capacity(self):
        x, y = np.ones((1, 1)), np.zeros(1, np.int64)
        w, a = np.zeros((1, 2)), np.zeros((1, 2))
        hist, result = np.zeros((2, len(HISTORY))), _fit_result()
        args = (1, 1, 2, _ptr(x), _ptr(y), None, None, _ptr(w), _ptr(a), _ptr(hist), 2)
        options = _options(gamma=1.0, max_epoch=5, eval_epoch=1, tol=0.0, eval_on_start=1)
        with pytest.raises(ValueError, match="epochs / eval_epoch"):
            self.solver._call("sdca_fit", ct.byref(options), *args, ct.byref(result))
        options.max_epoch = 0
        self.solver._call("sdca_fit", ct.byref(options), *args, ct.byref(result))
        assert result.status == STATUS.index("max_epoch")
        assert result.epochs == 0
        assert result.history_used in (1, 2)

    def test_direct_calls_reject_nonfinite_values(self, subtests: pytest.Subtests):
        z, y, mask = self.z.copy(), self.y, np.array([[1, 0, 0]], np.uint8)
        z[0, 0] = np.nan
        values, grad = np.full(1, 7.0), np.full_like(z, 7.0)
        options = ct.byref(_options(gamma=1.0))
        loss = (options, 1, 3, _ptr(z), _ptr(mask), _ptr(values), _ptr(grad))
        self.assert_error("sdca_loss_gradient", "Logits must be finite", *loss)
        np.testing.assert_array_equal(values, 7.0)
        out, previous = np.full_like(z, 7.0), np.zeros_like(z)
        update = (options, 1, 3, _ptr(z), _ptr(y), None, _ptr(previous), 1.0, _ptr(out))
        self.assert_error("sdca_update_objective", "Logits must be finite", *update)
        np.testing.assert_array_equal(out, 7.0)
        x, w, a = np.ones((1, 1)), np.zeros((1, 2)), np.zeros((1, 2))
        hist = np.zeros((3, len(HISTORY)))
        for tol, data, message in [
            (np.nan, x, "tol must be finite and nonnegative"),
            (-1.0, x, "tol must be finite and nonnegative"),
            (0.0, np.full((1, 1), np.inf), "Inputs must be finite"),
        ]:
            with subtests.test(message=message):
                fit = ct.byref(_options(gamma=1.0, max_epoch=1, eval_epoch=1, tol=tol))
                args = (fit, 1, 1, 2, _ptr(data), _ptr(y), None, None, _ptr(w), _ptr(a))
                rest = (_ptr(hist), len(hist), ct.byref(_fit_result()))
                self.assert_error("sdca_fit", message, *args, *rest)

    def test_invalid_input_leaves_outputs_untouched(self, subtests: pytest.Subtests):
        z, previous = self.z, np.zeros((1, 3))
        labels = {"one": np.array([0], np.int64), "bad": np.array([3], np.int64)}
        sets = (np.array([0, 1], np.int64), np.array([0, 2], np.int64))
        cases = [
            (0, 1, 3, 0, labels["one"], None, 1.0, "Invalid objective dimensions"),
            (0, 1, 3, 3, labels["one"], None, 1.0, "Invalid objective dimensions"),
            (0, 1, 1, 1, labels["one"], None, 1.0, "Invalid objective dimensions"),
            (0, 1, 3, 1, labels["bad"], None, 1.0, "Labels must lie in [0, num_classes)"),
            (
                0,
                1,
                3,
                1,
                labels["one"],
                None,
                -1.0,
                "C must be finite and positive, gamma finite and nonnegative",
            ),
            (6, 1, 3, 1, sets[0], None, 1.0, "Multilabel objectives need label offsets"),
            (6, 1, 3, 1, sets[0][::-1].copy(), sets[1], 1.0, "sorted without duplicates"),
            (99, 1, 3, 1, labels["one"], None, 1.0, "Unknown objective"),
        ]
        for objective, n, classes, k, y, offsets, c, message in cases:
            out = np.full_like(z, 123.0)
            options = _options(loss=objective, c=c, k=k, gamma=1.0)
            with subtests.test(objective=objective, message=message):
                with pytest.raises(ValueError, match=re.escape(message)):
                    self.solver._call(
                        "sdca_update_objective",
                        ct.byref(options),
                        n,
                        classes,
                        _ptr(z),
                        _ptr(y),
                        _ptr(offsets),
                        _ptr(previous),
                        1.0,
                        _ptr(out),
                    )
                np.testing.assert_array_equal(out, 123.0)
        values, grad = np.full(1, 7.0), np.full_like(z, 7.0)
        mask = np.array([[1, 0, 0]], np.uint8)
        with pytest.raises(ValueError, match="Unknown loss"):
            self.solver._call(
                "sdca_loss_gradient",
                ct.byref(_options(loss=42, gamma=1.0)),
                1,
                3,
                _ptr(z),
                _ptr(mask),
                _ptr(values),
                _ptr(grad),
            )
        np.testing.assert_array_equal(values, 7.0)
        np.testing.assert_array_equal(grad, 7.0)


def test_initial_dual_balance_check_survives_huge_entries():
    """A row whose absolute sum overflows float64 (C near its maximum) is still balanced or not."""
    from libsdca._core import _validate_initial

    labels = np.zeros(1, dtype=np.int64)
    unbalanced = np.array([[1.6e308, -0.5e308, -0.5e308]])
    with pytest.raises(ValueError, match="sum to zero"):
        _validate_initial(unbalanced, labels, None, 1.6e308, "cross_entropy", 1)
    balanced = np.array([[1.6e308, -0.8e308, -0.8e308]])
    _validate_initial(balanced, labels, None, 1.6e308, "cross_entropy", 1)
