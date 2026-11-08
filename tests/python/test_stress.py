"""Precision corner cases on the NumPy and C interfaces.

References are independent: closed forms, KKT solutions or mpmath, never
another backend or precision of this library.
"""

import ctypes as ct
import math
from fractions import Fraction

import libsdca
import numpy as np
import pytest
from libsdca import OBJECTIVES, STATUS
from libsdca._core import Solver, _fit_result, _options, _ptr

# (hinge, smoothed by gamma)
HINGES = [
    (name, smooth)
    for smooth in (False, True)
    for name in ("topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge")
]
CONFIGS = [
    ("float32", np.float32),
    ("float64", np.float32),
    ("float64", np.float64),
]
DTYPES = {"float32": np.float32, "mixed": np.float32, "float64": np.float64}


def _solvers():
    """(precision, solver) for every precision/dtype combination."""
    for name, (prec, dtype) in [
        ("float32", ("float32", np.float32)),
        ("mixed", ("float64", np.float32)),
        ("float64", ("float64", np.float64)),
    ]:
        yield name, Solver(prec, dtype=dtype)


def _width(objective: str, gamma: float) -> Fraction:
    """Two-class smoothing width: 0 nonsmooth (gamma = 0), gamma top-k, 2 gamma multilabel."""
    if gamma == 0:
        return Fraction(0)
    return Fraction(gamma) * (2 if objective.startswith("multilabel") else 1)


def _hinge(z0: float, z1: float, width: Fraction) -> tuple[float, float]:
    """Exact value and slope at h = 1 + z1 - z0: max(0, h), or max h v - width v^2/2 on [0, 1]."""
    h = 1 + Fraction(z1) - Fraction(z0)
    if width == 0:
        return float(max(h, 0)), float(h > 0)
    v = min(Fraction(1), max(Fraction(0), h / width))
    return float(h * v - width * v * v / 2), float(v)


class StressEntropyTests:
    def test_warm_step_keeps_the_residual_of_cancelling_scores(self):
        # Warm cancellation: old [A,-A/2,-A/2,0], scores old + [0,0,0,2], C = A = 2^80. Oracle:
        # 100-digit KKT solve at the exact residual [0,0,0,2].
        a = 2.0**80
        oracle = np.array(
            [40.1449881174924302, -12.7604368166896113, -12.7604368166896113, -14.6241144841132076]
        )
        for precision, solver in _solvers():
            dtype = DTYPES[precision]
            previous = np.array([[a, -a / 2, -a / 2, 0]], dtype)
            logits = np.array([[a, -a / 2, -a / 2, 2]], dtype)
            for objective, options in [("cross_entropy", {}), ("topk_cross_entropy", {"k": 2})]:
                out = solver.coordinate_update(
                    logits, [0], objective, previous=previous, C=a, norm2=1, **options
                )
                error = np.abs(out[0] - oracle).max() / np.abs(oracle).max()
                assert error / np.finfo(dtype).eps <= 32, f"{precision} {objective} {out}"

    def test_a1_multilabel_labels_share_the_underflowed_other_mass(self):
        # z = [400,400,-400], labels {0,1}, C = 1e200, norm2 = 1e-200.
        # C norm2 a is negligible: -a_2 = C e^-800 / (2 + e^-800), a_0 = a_1 = -a_2 / 2.
        other = math.exp(math.log(1e200) - 800 - math.log(2))
        out = Solver("float64").coordinate_update(
            [[400.0, 400.0, -400.0]], [[0, 1]], "multilabel_cross_entropy", C=1e200, norm2=1e-200
        )
        np.testing.assert_allclose(out[0], [other / 2, other / 2, -other], rtol=1e-12, atol=0)

    def test_proximal_entropy_values_at_extreme_scales(self):
        # Equal label scores reach log 2 at alpha = 1e50; at alpha = 1e300 the
        # value is d^2/(4 alpha) + log1p(e^-d), d + log d + log1p(e^-d) = log(2 alpha).
        solver = Solver("float64")
        for z, target, alpha, expected in [
            ([[1e16, 0.0, 0.0]], [[0, 1]], 1e50, math.log(2)),
            ([[0.0, 0.0]], [0], 1e300, 1.1762794618348152e-295),
        ]:
            value, _ = solver.loss_and_grad(z, target, "proximal_cross_entropy", alpha=alpha)
            assert abs(value[0] / expected - 1) <= 1e-13
        # 100-class equal-competitor row at alpha = 1e300 stays within 32 eps.
        z100 = [[0.0] + [1.0] * 99]
        v100, g100 = solver.loss_and_grad(z100, [0], "proximal_cross_entropy", alpha=1e300)
        ref_v100 = 2.3624605285888881e-295
        ref_g0 = -6.829457899398158e-298
        eps = np.finfo(np.float64).eps
        assert abs(v100[0] - ref_v100) / ref_v100 / eps <= 32
        assert abs(g100[0, 0] - ref_g0) / abs(ref_g0) / eps <= 32

    def test_topk_single_solve_around_split_threshold(self):
        # Scores right on the |v_i| = 64 split boundary match the 100-digit KKT root.
        expected = np.array(
            [1.0, -0.5, -0.1927026142304690, -0.1557295981036007, -0.1515677876659303]
        )
        for precision, solver in _solvers():
            dtype = DTYPES[precision]
            tol = 8 * float(np.finfo(dtype).eps)
            for gap in (63.875, 64.0, 64.125):
                prev = np.array([[32.0, -32.0, 0.0, 0.0, 0.0]], dtype)
                z = np.array([[32.0 - gap, gap - 32.0, 0.0, -0.25, -0.28125]], dtype)
                out = solver.coordinate_update(
                    z, [0], "topk_cross_entropy", previous=prev, C=1.0, k=2
                )
                np.testing.assert_allclose(
                    out[0], expected, rtol=tol, atol=0, err_msg=f"{precision} {gap}"
                )


class StressHingeTests:
    def test_losses_against_two_class_closed_forms(self):
        # Margins that cancel in 1 + z1 - z0, and h a - a^2/2 overflowing at gamma 1e300.
        cases = [(1.0, 1e-20, 1e-20), (1.0, 1e-20, 1.0), (1.0, 2.0**-26, 1.0), (0.0, 1e200, 1e300)]
        for precision, solver in _solvers():
            dtype = DTYPES[precision]
            for z0, z1, gamma in cases:
                z = np.array([[z0, z1]], dtype)
                if not np.isfinite(z).all():
                    continue
                for objective, smooth in HINGES:
                    g_used = gamma if smooth else 0.0
                    value, slope = _hinge(float(z[0, 0]), float(z[0, 1]), _width(objective, g_used))
                    target = [[0]] if objective.startswith("multilabel") else [0]
                    v, g = solver.loss_and_grad(z, target, objective, gamma=g_used)
                    message = f"{precision} {objective} {z[0]} {g_used}"
                    # Relative to the value and to the order-one slope; float32 computes in
                    # float, whose subnormals hold a value of 5e-41 to about 1e-45.
                    assert np.isfinite(v).all(), message
                    single = precision == "float32"
                    np.testing.assert_allclose(
                        v[0],
                        value,
                        rtol=4 * float(np.finfo(dtype).eps) if single else 1e-14,
                        atol=4 * float(np.finfo(np.float32).smallest_subnormal) if single else 0,
                        err_msg=message,
                    )
                    np.testing.assert_allclose(
                        g[0],
                        [-slope, slope],
                        rtol=2 * float(np.finfo(dtype).eps),
                        atol=0,
                        err_msg=message,
                    )

    def test_coordinate_steps_resolve_a_tiny_margin_and_huge_norm2(self):
        # Tiny margin: [1, 2^-26], C = norm2 = gamma = 1, zero duals: [s, -s], s = h / (2 + width),
        # to about an ulp of the scores (absolute).
        h = Fraction(2.0**-26)
        for precision, solver in _solvers():
            dtype = DTYPES[precision]
            for objective, smooth in HINGES:
                target = [[0]] if objective.startswith("multilabel") else [0]
                gamma = 1.0 if smooth else 0.0
                den = 2 + _width(objective, gamma)
                s = float(h / den)
                out = solver.coordinate_update(
                    np.array([[1.0, 2.0**-26]], dtype), target, objective, C=1, norm2=1, gamma=gamma
                )
                np.testing.assert_allclose(
                    out[0],
                    [s, -s],
                    rtol=4 * float(np.finfo(dtype).eps),
                    atol=4 * float(np.finfo(dtype).eps) * 2 / float(den),
                    err_msg=f"{precision} {objective}",
                )
            # Zero scores, 3 classes, C = 1, k = 1, huge norm2 -> [2, -1, -1] / (3 norm2).
            norm2 = 1e20 if dtype == np.float64 else 1e8
            s_norm = 1.0 / (3.0 * norm2)
            for objective, gamma in (
                ("topk_hinge_beta", 0.0),
                ("topk_hinge_beta", 1.0),
                ("topk_hinge_alpha", 0.0),
                ("topk_hinge_alpha", 1.0),
            ):
                out3 = solver.coordinate_update(
                    np.zeros((1, 3), dtype), [0], objective, C=1, norm2=norm2, k=1, gamma=gamma
                )
                np.testing.assert_allclose(
                    out3[0],
                    [2 * s_norm, -s_norm, -s_norm],
                    rtol=8 * float(np.finfo(dtype).eps),
                    atol=0,
                    err_msg=f"{precision} {objective} norm2={norm2}",
                )


class StressFitTests:
    def test_zero_rows_take_their_optimal_block(self):
        # X = [[0],[1]], labels [0,1], C = 1: the zero row's block is C (e_y - 1/m).
        x = np.array([[0.0], [1.0]])
        for precision, solver in _solvers():
            dtype = DTYPES[precision]
            for data, kernel in [(x, False), (x @ x.T, True)]:
                model = solver.fit(data.astype(dtype), [0, 1], C=1.0, kernel=kernel, tol=1e-6)
                message = f"{precision} kernel={kernel}"
                assert model.status == "solved", message
                np.testing.assert_allclose(
                    model.dual[0],
                    [0.5, -0.5],
                    rtol=2 * float(np.finfo(dtype).eps),
                    atol=0,
                    err_msg=message,
                )

    def test_a10_a11_exports_and_model_input_widths(self):
        # Solver and the array aliases are exported, and a
        # model rejects inputs of the wrong width instead of broadcasting them.
        for name in ["Solver", "FloatArray", "IndexArray", "Targets"]:
            assert hasattr(libsdca, name), name
        x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        model = Solver().fit(x, [0, 1, 0], max_epoch=5)
        for bad in [x[0], x[:, :1], np.c_[x, x]]:
            with pytest.raises(ValueError):
                model.decision_function(bad)


class StressNativeAbiTests:
    """Direct C callers: invalid input is rejected and leaves outputs as given."""

    def setup_method(self):
        self.solver = Solver()
        self.call = self.solver._call

    def test_zero_rows_with_c_beyond_float_storage(self):
        # c (e_y - 1/m) = [5e99, -5e99] overflows float storage; the fit must not
        # return infinite duals or NaN weights.
        entropy = OBJECTIVES.index("cross_entropy")
        for precision_code in [0, 1]:
            for kernel in [0, 1]:
                x = np.zeros((1, 1), np.float32)
                w, a = np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32)
                history, result = np.zeros((4, len(libsdca.HISTORY))), _fit_result()
                labels = np.zeros(1, np.int64)
                options = _options(precision=precision_code, loss=entropy, c=1e100, gamma=1.0)
                options.max_epoch, options.eval_epoch, options.kernel = 1, 1, kernel
                args = (ct.byref(options), 1, 1, 2, _ptr(x), _ptr(labels), None, None)
                message = f"code={precision_code} kernel={kernel}"
                try:
                    self.call(
                        "sdca_fit", *args, _ptr(w), _ptr(a), _ptr(history), 4, ct.byref(result)
                    )
                except ValueError:
                    pass
                else:
                    assert STATUS[result.status] != "solved", message
                assert np.isfinite(a).all() and np.isfinite(w).all(), message
                np.testing.assert_array_equal(a, 0, err_msg=message)

    def test_nonfinite_warm_state_and_extents_are_rejected(self):
        # Previous duals, center and warm duals must be finite, and
        # buffer extents must not overflow size_t.
        z, labels = np.array([[1.0, 2.0, 3.0]]), np.zeros(1, np.int64)
        previous, out = np.array([[0.0, 0.0, np.inf]]), np.full((1, 3), 7.0)
        options = ct.byref(_options(gamma=1.0))
        update = (options, 1, 3, _ptr(z), _ptr(labels), None, _ptr(previous), 1.0, _ptr(out))
        with pytest.raises(ValueError):
            self.call("sdca_update_objective", *update)
        np.testing.assert_array_equal(out, 7.0)
        x = np.ones((1, 1))
        history = np.zeros((2, len(libsdca.HISTORY)))
        for warm, center, dual in [
            (0, np.array([[np.inf, 0.0]]), np.full((1, 2), 7.0)),
            (1, None, np.array([[np.nan, 7.0]])),
        ]:
            w = np.full((1, 2), 7.0)
            fit = _options(gamma=1.0, max_epoch=0, eval_epoch=1, tol=0.0, warm=warm)
            args = (ct.byref(fit), 1, 1, 2, _ptr(x), _ptr(labels), None, _ptr(center))
            outputs = (_ptr(w), _ptr(dual), _ptr(history), 2, ct.byref(_fit_result()))
            with pytest.raises(ValueError):
                self.call("sdca_fit", *args, *outputs)
            np.testing.assert_array_equal(w, 7.0)
        mask, values, grad = np.array([[1, 0]], np.uint8), np.full(1, 7.0), np.full((1, 2), 7.0)
        loss = (options, 2**64 - 1, 2, _ptr(z), _ptr(mask), _ptr(values), _ptr(grad))
        with pytest.raises(ValueError):
            self.call("sdca_loss_gradient", *loss)
        np.testing.assert_array_equal(values, 7.0)
