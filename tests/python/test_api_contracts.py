"""Contracts of the public Python API and the native C ABI behind it."""

import ctypes as ct
import dataclasses
import inspect
import os
import re
import sys
from pathlib import Path
from typing import Any

import libsdca
import numpy as np
import pytest
from child_process import run_python
from libsdca import (
    HINGES,
    HISTORY,
    LOSSES,
    OBJECTIVES,
    PRECISIONS,
    STATUS,
    CrossEntropy,
    Distillation,
    Kernel,
    Model,
    MulticlassHinge,
    MultilabelCrossEntropy,
    MultilabelHinge,
    Probabilities,
    ProximalCrossEntropy,
    ProximalDistillation,
    TopKCrossEntropy,
    TopKHingeAlpha,
    TopKHingeBeta,
)
from libsdca._core import (
    ABI_VERSION,
    TARGET_LABELS,
    _Execution,
    _FitOptions,
    _FitResult,
    _Input,
    _Loss,
    _make,
    _Native,
    _native,
    _Targets,
)

MISSING = "/does-not-exist/libsdca_missing.so"


class LibraryContractTests:
    def test_missing_library_error_names_the_fix(self):
        with pytest.raises(OSError, match="build.py"):
            _Native(MISSING)

    def test_import_loads_neither_frameworks_nor_the_library(self):
        code = (
            "import ctypes, sys\n"
            "def fail(*a, **k): raise AssertionError('library loaded on import')\n"
            "ctypes.CDLL = fail\n"
            "import libsdca\n"
            "assert 'torch' not in sys.modules and 'jax' not in sys.modules\n"
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

    def test_version_and_abi(self):
        native = _native()
        assert native.lib.sdca_abi_version() == ABI_VERSION
        # A stale native library with the same ABI would otherwise load silently.
        assert native.version == libsdca.__version__


class VocabularyContractTests:
    """Names shared with the C ABI, MATLAB and Rust interfaces."""

    def test_names_and_ids(self):
        assert list(LOSSES) == [
            "cross_entropy",
            "topk_cross_entropy",
            "multiclass_hinge",
            "topk_hinge_alpha",
            "topk_hinge_beta",
            "multilabel_hinge",
            "multilabel_cross_entropy",
            "proximal_cross_entropy",
            "distillation",
            "proximal_distillation",
        ]
        assert all(cls.name == name for name, cls in LOSSES.items())
        assert OBJECTIVES == tuple(LOSSES)[:7]
        assert HINGES == {
            "multiclass_hinge",
            "topk_hinge_alpha",
            "topk_hinge_beta",
            "multilabel_hinge",
        }
        assert STATUS == ("solved", "no_progress", "max_epoch", "failed")
        assert HISTORY == (
            "epoch",
            "primal",
            "dual",
            "primal_loss",
            "dual_loss",
            "regularizer",
            "relative_gap",
            "accuracy",
            "cpu_time",
            "wall_time",
        )
        assert list(PRECISIONS) == ["float32", "float64"]

    def test_losses_take_only_their_own_parameters(self, subtests: pytest.Subtests):
        params = {
            CrossEntropy: [],
            TopKCrossEntropy: ["k"],
            MulticlassHinge: ["gamma"],
            TopKHingeAlpha: ["k", "gamma"],
            TopKHingeBeta: ["k", "gamma"],
            MultilabelHinge: ["gamma", "projection_precision"],
            MultilabelCrossEntropy: [],
            ProximalCrossEntropy: ["alpha"],
            Distillation: ["temperature"],
            ProximalDistillation: ["temperature", "alpha"],
        }
        for cls, names in params.items():
            assert [f.name for f in dataclasses.fields(cls)] == names
        with pytest.raises(TypeError):
            CrossEntropy(k=2)  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError):
            TopKCrossEntropy(2, gamma=0.5)  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError):
            MulticlassHinge(k=2)  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError):
            TopKHingeBeta()  # pyright: ignore[reportCallIssue]  # k is required
        bad: list[tuple[Any, dict[str, Any], type[Exception]]] = [
            (TopKCrossEntropy, {"k": 0}, ValueError),
            (TopKCrossEntropy, {"k": True}, TypeError),
            (TopKCrossEntropy, {"k": 2.0}, TypeError),
            (MulticlassHinge, {"gamma": -1.0}, ValueError),
            (MulticlassHinge, {"gamma": float("nan")}, ValueError),
            (MulticlassHinge, {"gamma": True}, TypeError),
            (MulticlassHinge, {"gamma": "0.5"}, TypeError),
            (ProximalCrossEntropy, {"alpha": 0.0}, ValueError),
            (Distillation, {"temperature": float("inf")}, ValueError),
            (ProximalDistillation, {"alpha": None}, TypeError),
        ]
        for cls, kwargs, error in bad:
            with subtests.test(cls=cls.__name__, kwargs=kwargs), pytest.raises(error):
                cls(**kwargs)
        two: Any = np.int64(2)
        assert TopKHingeBeta(two).k == 2 and type(TopKHingeBeta(two).k) is int
        assert MulticlassHinge(1).gamma == 1.0 and type(MulticlassHinge(1).gamma) is float
        # k < classes is checked at the call.
        with pytest.raises(ValueError, match="k must"):
            libsdca.loss_and_grad([[0.0, 1.0]], [0], TopKCrossEntropy(2))

    def test_one_target_kind_per_loss(self):
        z = np.array([[0.5, -1.0, 2.0], [1.0, 0.0, -2.0]])
        labels, mask = np.array([0, 2]), np.array([[1, 0, 1], [0, 1, 0]], bool)
        with pytest.raises(TypeError):  # a mask is not a label vector
            libsdca.loss_and_grad(z, mask, CrossEntropy())
        with pytest.raises(TypeError):  # labels are not label sets
            libsdca.loss_and_grad(z, labels, MultilabelCrossEntropy())
        with pytest.raises(TypeError):
            libsdca.loss_and_grad(z, Probabilities(np.full((2, 3), 1 / 3)), CrossEntropy())
        with pytest.raises(TypeError):
            libsdca.fit(np.eye(2), Probabilities(np.full((2, 3), 1 / 3)), MulticlassHinge())
        with pytest.raises(ValueError, match="not a training objective"):
            libsdca.fit(np.eye(2), [0, 1], ProximalCrossEntropy())
        with pytest.raises(ValueError, match="not a training objective"):
            libsdca.coordinate_update(z, z, Distillation())
        # Label sets with every class are rejected (the solver's steps need a negative).
        full = np.ones((2, 3), bool)
        for loss in (MultilabelCrossEntropy(), MultilabelHinge()):
            with pytest.raises(ValueError, match="class outside"):
                libsdca.loss_and_grad(z, full, loss)
            with pytest.raises(ValueError, match="class outside"):
                libsdca.fit(np.eye(2), full, loss)

    def test_distillation(self):
        x = np.array([[1.0, 0.2], [-0.3, 1.0], [0.7, -0.4], [-0.2, -0.8]])
        q = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6], [0.6, 0.3, 0.1]])
        model = libsdca.fit(x, Probabilities(q), C=2.0, tol=1e-6)
        assert model.status == "solved" and model.loss == CrossEntropy()
        assert model.weights is not None
        np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1.0, atol=1e-12)
        z = x @ model.weights
        step = libsdca.coordinate_update(z, Probabilities(q), C=2.0)
        np.testing.assert_allclose(step.sum(axis=1), 0.0, atol=1e-14)
        dv, dg = libsdca.loss_and_grad(z, z, Distillation())
        np.testing.assert_allclose(dg, 0.0, atol=1e-15)
        pv, pg = libsdca.loss_and_grad(z, z, ProximalDistillation(alpha=0.5))
        np.testing.assert_allclose(pv, dv, atol=1e-14)
        np.testing.assert_allclose(pg, 0.0, atol=1e-15)
        with pytest.raises(ValueError, match="shape"):
            libsdca.loss_and_grad(z, z[:, :2], Distillation())

    def test_probabilities_are_checked_and_renormalized(self):
        x = np.array([[1.0, 0.2], [-0.3, 1.0]])
        q = np.array([[0.5, 0.5], [0.25, 0.75]])
        reference = libsdca.fit(x, Probabilities(q), tol=0, max_epoch=5)
        # Rounding of float32 targets is renormalized, not rejected.
        nudged = libsdca.fit(
            x, Probabilities((q * (1 + 1e-7)).astype(np.float32)), tol=0, max_epoch=5
        )
        np.testing.assert_allclose(nudged.dual, reference.dual, atol=1e-12)
        for bad in (q * 1.1, -q, np.full((2, 2), np.nan)):
            with pytest.raises(ValueError):
                libsdca.fit(x, Probabilities(bad))
        with pytest.raises(ValueError, match="shape"):
            libsdca.fit(x, Probabilities(q[:1]))
        with pytest.raises(ValueError, match="num_classes"):
            libsdca.fit(x, Probabilities(q), num_classes=3)

    def test_signatures(self):
        def names(f: Any) -> list[tuple[str, object]]:
            return [(p.name, p.default) for p in inspect.signature(f).parameters.values()]

        empty = inspect.Parameter.empty
        assert names(libsdca.fit) == [
            ("X", empty),
            ("y", empty),
            ("loss", CrossEntropy()),
            ("C", 1.0),
            ("tol", 1e-3),
            ("max_epoch", 1000),
            ("eval_epoch", 10),
            ("eval_on_start", False),
            ("recompute_primal", True),
            ("shrinking", "auto"),
            ("seed", 0),
            ("num_classes", None),
            ("center", None),
            ("initial_dual", None),
            ("precision", None),
            ("num_threads", None),
        ]
        assert names(libsdca.loss_and_grad) == [
            ("logits", empty),
            ("target", empty),
            ("loss", CrossEntropy()),
            ("precision", None),
            ("num_threads", None),
        ]
        assert names(libsdca.coordinate_update) == [
            ("logits", empty),
            ("target", empty),
            ("loss", CrossEntropy()),
            ("previous", None),
            ("C", 1.0),
            ("norm2", 1.0),
            ("precision", None),
            ("num_threads", None),
        ]
        assert [f.name for f in dataclasses.fields(Model)] == [
            "loss",
            "C",
            "num_classes",
            "kernel",
            "weights",
            "dual",
            "status",
            "epochs",
            "history",
        ]


class PrecisionContractTests:
    def test_output_dtypes(self, subtests: pytest.Subtests):
        x = np.array([[1.0, 0.2], [-0.3, 1.0], [0.7, -0.4]])
        y = np.array([0, 1, 2])
        for storage, precision, dtype in (
            (np.float32, None, np.float32),
            (np.float32, "float64", np.float32),
            (np.float64, None, np.float64),
            (np.float64, "float32", np.float32),
            (np.float16, None, np.float32),
            (np.int64, None, np.float64),
            (np.float64, np.float32, np.float32),
        ):
            with subtests.test(storage=storage, precision=precision):
                data = (x * 4).astype(storage)
                values, grad = libsdca.loss_and_grad(data @ data.T, y, precision=precision)
                assert (values.dtype, grad.dtype) == (np.float64, dtype)
                step = libsdca.coordinate_update(data @ data.T, y, precision=precision)
                assert step.dtype == dtype
                model = libsdca.fit(data, y, max_epoch=3, precision=precision)
                assert model.weights is not None
                assert (model.weights.dtype, model.dual.dtype) == (dtype, dtype)

    def test_float64_work_solves_in_double_and_float32_in_float(self, subtests: pytest.Subtests):
        rng = np.random.default_rng(3)
        z = (rng.standard_normal((200, 12)) * 30).astype(np.float32)
        y = rng.integers(0, 12, 200)
        for loss in (CrossEntropy(), TopKCrossEntropy(3)):
            exact = libsdca.coordinate_update(z.astype(np.float64), y, loss, C=10.0)
            scale = np.abs(exact).max(axis=1, keepdims=True)
            ulp = np.finfo(np.float32).eps * scale
            double_on_fp32 = libsdca.coordinate_update(z, y, loss, C=10.0, precision="float64")
            single = libsdca.coordinate_update(z, y, loss, C=10.0)
            with subtests.test(loss=loss):
                assert np.max(np.abs(double_on_fp32 - exact) / ulp) <= 8
                bound = 64 * 12 + 16 * np.abs(z).max()
                assert np.max(np.abs(single - exact) / ulp) <= bound


class BoundaryContractTests:
    def test_warm_start_is_checked_by_the_library(self):
        with pytest.raises(ValueError, match="nonpositive"):
            libsdca.fit([[1.0]], [0], num_classes=2, initial_dual=[[-1e-16, 1e-16]])
        x, y = np.eye(3), np.arange(3)
        model = libsdca.fit(x, y)
        for scale in (10.0, -1.0):
            with pytest.raises(ValueError, match="initial_dual"):
                libsdca.fit(x, y, initial_dual=scale * model.dual)
        with pytest.raises(ValueError, match="shape"):
            libsdca.fit(x, y, initial_dual=model.dual[:, :2])
        # The library builds W = center + X.T @ A from the warm start.
        center = np.ones((3, 3))
        warm = libsdca.fit(x, y, center=center, initial_dual=model.dual, max_epoch=0)
        assert warm.weights is not None
        np.testing.assert_allclose(warm.weights, center + x.T @ model.dual, atol=1e-15)

    def test_types_and_values(self, subtests: pytest.Subtests):
        x, y = [[1.0]], [0]
        wrong_type: list[dict[str, Any]] = [
            {"num_classes": True},
            {"max_epoch": True},
            {"max_epoch": 1.5},
            {"eval_epoch": False},
            {"C": True},
            {"C": "1"},
            {"seed": 1.0},
            {"eval_on_start": 1},
            {"shrinking": None},
            {"num_threads": 1.0},
            {"precision": 32},
        ]
        for bad in wrong_type:
            with subtests.test(bad=bad), pytest.raises(TypeError):
                options: dict[str, Any] = {"num_classes": 2} | bad
                libsdca.fit(x, y, **options)
        bad_value: list[dict[str, Any]] = [
            {"C": -1.0},
            {"C": float("nan")},
            {"tol": -1.0},
            {"max_epoch": -1},
            {"seed": -1},
            {"seed": 2**64},
            {"shrinking": "never"},
            {"num_threads": 0},
            {"precision": "double"},
        ]
        for bad in bad_value:
            with subtests.test(bad=bad), pytest.raises(ValueError):
                libsdca.fit(x, y, num_classes=2, **bad)
        with pytest.raises(TypeError, match="loss"):
            libsdca.fit(x, y, "cross_entropy")  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValueError, match="too large for the history"):
            libsdca.fit(x, y, num_classes=2, max_epoch=sys.maxsize, eval_epoch=1)
        with pytest.raises(ValueError, match="BLAS integer limit"):
            libsdca.fit([[1.0, 2.0]], [0], num_classes=2**40)

    def test_entropy_step_at_a_label_probability_of_one(self):
        # A cross-entropy step whose label probability is 1 to within exp(-218) has its
        # normalization threshold at the bracket's end 0, beyond the reach of bisection;
        # it failed the fit ("Entropy normalization did not converge") for some epoch orders
        # of this warm start (duals of C = 0.01 scaled to C = 1).
        rng = np.random.default_rng(4)
        y = np.arange(90) % 3
        x = rng.normal(size=(90, 4)) + 3 * np.eye(3, 4)[y]
        dual = libsdca.fit(x, y, C=1.0, tol=1e-6, seed=21).dual
        for previous, C in [(1.0, 0.1), (0.1, 0.01)]:
            start = dual * (C / previous)
            dual = libsdca.fit(x, y, C=C, tol=1e-6, seed=21, initial_dual=start).dual
        for seed in (21, 45, 62):  # the epoch orders whose first step failed
            model = libsdca.fit(x, y, C=1.0, tol=1e-6, seed=seed, initial_dual=dual * 100)
            assert model.status == "solved", seed

    def test_eval_epoch_zero_evaluates_at_the_end(self):
        # As in C++, MATLAB and Rust: one evaluation, after max_epoch; it decides the status.
        rng = np.random.default_rng(0)
        x, y = rng.normal(size=(50, 3)), np.arange(50) % 3
        short = libsdca.fit(x, y, eval_epoch=0, max_epoch=5, tol=1e-12)
        assert short.status == "max_epoch" and list(short.history["epoch"]) == [5]
        long = libsdca.fit(x, y, eval_epoch=0, max_epoch=300)
        assert long.status == "solved" and list(long.history["epoch"]) == [300]
        with pytest.raises(ValueError, match="eval_epoch"):
            libsdca.fit(x, y, eval_epoch=-1)

    def test_sparse_inputs_raise_type_error(self):
        sparse = pytest.importorskip("scipy.sparse")
        x = sparse.csr_matrix(np.eye(3))
        with pytest.raises(TypeError, match=r"sparse.*toarray"):
            libsdca.fit(x, [0, 1, 2])
        with pytest.raises(TypeError, match="toarray"):
            libsdca.loss_and_grad(x, [0, 1, 2])

    def test_coordinate_update_output_is_a_valid_warm_start(self):
        rng = np.random.default_rng(2026)
        scores = rng.normal(0, 10, (200, 10)).astype(np.float32)
        y = np.zeros(200, dtype=int)
        loss = TopKCrossEntropy(3)
        for precision in ("float32", "float64"):
            a = libsdca.coordinate_update(scores, y, loss, C=10.0, precision=precision)
            np.testing.assert_array_equal(
                libsdca.coordinate_update(
                    scores, y, loss, C=10.0, precision=precision, previous=a
                ).shape,
                a.shape,
            )
            model = libsdca.fit(
                np.ones((200, 1), dtype=np.float32),
                y,
                loss,
                C=10.0,
                num_classes=10,
                initial_dual=a,
                max_epoch=0,
                precision=precision,
            )
            assert model.status == "max_epoch"

    def test_seed_orders_the_epochs(self):
        rng = np.random.default_rng(5)
        x, y = rng.normal(size=(60, 4)), np.arange(60) % 3
        fits = [
            libsdca.fit(x, y, TopKHingeBeta(2), seed=seed, tol=0, max_epoch=3) for seed in (7, 7, 8)
        ]
        np.testing.assert_array_equal(fits[0].dual, fits[1].dual)
        assert not np.array_equal(fits[0].dual, fits[2].dual)
        for seed in (0, 2**64 - 1):
            assert libsdca.fit(x, y, seed=seed, tol=1e-6).status == "solved"

    def test_history_keys_are_fixed(self):
        x = np.array([[1.0, 0.2], [-0.3, 1.0], [0.7, -0.4]])
        for loss, y in [
            (CrossEntropy(), [0, 1, 2]),
            (MultilabelHinge(), [[0], [1, 2], [2]]),
            (MultilabelCrossEntropy(), [[0], [1, 2], [2]]),
        ]:
            model = libsdca.fit(x, y, loss, max_epoch=2)
            assert tuple(model.history) == HISTORY

    def test_kernel_marker(self):
        x = np.array([[1.0, 0.2], [-0.3, 1.0], [0.7, -0.4]])
        y = np.array([0, 1, 2])
        model = libsdca.fit(Kernel(x @ x.T), y, tol=1e-8)
        assert model.kernel and model.weights is None
        features = libsdca.fit(x, y, tol=1e-8)
        np.testing.assert_allclose(
            model.decision_function(x @ x.T), features.decision_function(x), atol=1e-6
        )
        with pytest.raises(ValueError, match="center"):
            libsdca.fit(Kernel(x @ x.T), y, center=np.zeros((3, 3)))
        with pytest.raises(ValueError, match="shape"):
            model.decision_function(x)


class ModelContractTests:
    def test_prediction(self):
        weights = np.array([[1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
        x = np.array([[1.0, 0.0], [0.0, 2.0]])
        common: dict[str, Any] = dict(
            loss=CrossEntropy(), C=1.0, num_classes=3, status="solved", epochs=1, history={}
        )
        feature = Model(kernel=False, weights=weights, dual=np.zeros((2, 3)), **common)
        np.testing.assert_array_equal(feature.predict(x), [0, 2])
        np.testing.assert_array_equal(feature.decision_function(x), x @ weights)
        kernel = Model(kernel=True, weights=None, dual=weights, **common)
        np.testing.assert_array_equal(kernel.decision_function(x), x @ weights)
        assert kernel.score(x, [0, 2]) == 1.0
        with pytest.raises(TypeError):
            Model(weights, weights)  # pyright: ignore[reportCallIssue]  # no defaults


class NativeAbiContractTests:
    """Direct ABI callers: errors are returned, never thrown, and leave outputs as given."""

    def setup_method(self):
        self.lib = _native().lib
        self.z = np.array([[1.0, 2.0, 3.0]])
        self.y = np.array([0], np.int64)

    def assert_error(self, name: str, message: str, *args: object) -> None:
        code = getattr(self.lib, name)(*args)
        assert code == -2, name
        assert re.search(re.escape(message), self.lib.sdca_last_error().decode())
        libsdca.loss_and_grad(self.z, self.y)
        assert self.lib.sdca_last_error() == b""

    def calls(self, loss: Any, execution: Any) -> dict[str, tuple[object, ...]]:
        """One valid call of every computing export with these structs, and its outputs."""
        z, y = self.z, self.y
        values, out = np.full(1, 7.0), np.full_like(z, 7.0)
        self.outputs = (values, out)
        targets = _make(_Targets, kind=TARGET_LABELS, labels=y.ctypes.data)
        self.keep = (targets, np.ones(1), np.zeros((2, len(HISTORY))), np.ones((1, 1)))
        norm2, history, x = self.keep[1:]
        data = _make(_Input, n=1, d=1, x=x.ctypes.data)
        options = _make(_FitOptions, max_epoch=1, eval_epoch=1)
        self.keep += (data, options)
        a = np.full((1, 3), 7.0)
        w = np.full((1, 3), 7.0)
        self.fit_outputs = (a, w)
        z_, t = z.ctypes.data, ct.byref(targets)
        return {
            "sdca_loss_evaluate": (loss, execution, 1, 1, 3, z_, t, values.ctypes.data, out.ctypes.data),
            "sdca_dual_update": (loss, 1.0, execution, 1, 1, 3, z_, t, None, norm2.ctypes.data, out.ctypes.data),
            "sdca_model_fit": (
                loss, ct.byref(options), execution, ct.byref(data), 3, t, None, None,
                w.ctypes.data, a.ctypes.data, history.ctypes.data, 2,
                ct.byref(_make(_FitResult)),
            ),
        }  # fmt: skip

    def test_every_export_checks_its_structs(self, subtests: pytest.Subtests):
        loss, execution = _make(_Loss), _make(_Execution)
        bad_loss = _make(_Loss, kind=42)
        bad_precision = _make(_Execution, precision=9)
        for what, args, message in [
            ("loss kind", (ct.byref(bad_loss), ct.byref(execution)), "Unknown loss kind"),
            ("precision", (ct.byref(loss), ct.byref(bad_precision)), "Invalid precision"),
            ("NULL loss", (None, ct.byref(execution)), "must not be NULL"),
            ("NULL execution", (ct.byref(loss), None), "must not be NULL"),
        ]:
            for name, call in self.calls(*args).items():
                with subtests.test(what=what, name=name):
                    self.assert_error(name, message, *call)
                    for output in self.outputs + self.fit_outputs:
                        np.testing.assert_array_equal(output, 7.0)
        for size in (0, ct.sizeof(_Loss) - 4, ct.sizeof(_Loss) + 8):
            sized = _make(_Loss)
            sized.size = size
            for name, call in self.calls(ct.byref(sized), ct.byref(execution)).items():
                with subtests.test(size=size, name=name):
                    self.assert_error(name, "size", *call)

    def test_parameters_of_other_losses_are_rejected(self):
        execution = _make(_Execution)
        for fields, message in [
            (dict(kind=0, k=2), "takes no parameter k"),
            (dict(kind=0, gamma=0.5), "takes no parameter gamma"),
            (dict(kind=2, alpha=2.0), "takes no parameter alpha"),
            (dict(kind=1, k=0), "k must"),
        ]:
            loss = _make(_Loss, **fields)
            call = self.calls(ct.byref(loss), ct.byref(execution))["sdca_loss_evaluate"]
            self.assert_error("sdca_loss_evaluate", message, *call)
            np.testing.assert_array_equal(self.outputs[0], 7.0)

    def test_history_rows(self):
        lib = self.lib
        for max_epoch, eval_epoch, rows in [(1000, 10, 102), (5, 0, 2), (0, 1, 2), (7, 3, 4)]:
            options = _make(_FitOptions, max_epoch=max_epoch, eval_epoch=eval_epoch)
            assert lib.sdca_history_rows(ct.byref(options)) == rows
        assert lib.sdca_history_rows(None) == 0
        assert lib.sdca_last_error() != b""
