from fractions import Fraction
from typing import Any

import numpy as np
from libsdca import LOSSES
from libsdca._core import Solver
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

# (hinge, smoothed by gamma)
HINGES = [
    (name, smooth)
    for smooth in (False, True)
    for name in ("topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge")
]
# {z0, z1, gamma}: margins 1 + z1 - z0 that cancel, h a overflowing or
# underflowing at huge or tiny gamma, and huge common offsets.
HINGE_CASES = [
    (1.0, 1e-20, 1e-20),
    (1.0, 2.0**-26, 1.0),
    (0.0, 1e200, 1e300),
    (1.0, 1e-200, 1e-300),
    (0.0, 0.0, 1e-300),
    (1e16, 1e16 + 2, 1.0),
    (1e16, 1e16 - 2, 1.0),
    (-1e16, -1e16 + 2, 1e16),
]


def _eps(dtype: type[np.floating[Any]]) -> float:
    return float(np.finfo(dtype).eps)


def width_of(objective: str, smooth: bool, gamma: float) -> float:
    """The smoothing width of a two-class hinge: 0 nonsmooth, gamma top-k, 2 gamma multilabel."""
    if not smooth:
        return 0
    return 2 * gamma if objective.startswith("multilabel") else gamma


def two_class_hinge(z0: float, z1: float, width: float) -> tuple[float, float]:
    """Exact value and slope v at h = 1 + z1 - z0: max(0, h), or max h v - width v^2/2 on [0, 1]."""
    h = 1 + Fraction(z1) - Fraction(z0)
    if width == 0:
        return float(max(h, 0)), float(h > 0)
    v = min(Fraction(1), max(Fraction(0), h / Fraction(width)))
    return float(h * v - Fraction(width) * v * v / 2), float(v)


class LossTests:
    def test_all_gradients_by_finite_differences(self):
        solver = Solver()
        z = np.random.default_rng(9).normal(size=(3, 5))
        for objective in LOSSES:
            target = np.array([0, 2, 4])
            if objective.startswith("multilabel"):
                target = np.array([[1, 0, 1, 0, 0], [0, 1, 0, 0, 1], [1, 1, 1, 0, 0]], bool)
            value, grad = solver.loss_and_grad(z, target, objective, k=2, gamma=0.7, alpha=1.3)
            numeric = np.empty_like(z)
            for i, j in np.ndindex(z.shape):
                plus, minus = z.copy(), z.copy()
                plus[i, j] += 1e-5
                minus[i, j] -= 1e-5
                vp = solver.loss_and_grad(plus, target, objective, k=2, gamma=0.7, alpha=1.3)[0]
                vm = solver.loss_and_grad(minus, target, objective, k=2, gamma=0.7, alpha=1.3)[0]
                numeric[i, j] = (vp[i] - vm[i]) / 2e-5
            np.testing.assert_allclose(grad, numeric, atol=2e-7, rtol=2e-5, err_msg=objective)
            np.testing.assert_allclose(grad.sum(1), 0, atol=1e-14, err_msg=objective)
            assert np.isfinite(value).all()

    def test_proximal_loss_against_independent_optimizer(self):
        z = np.array([[2.0, -1.0, 0.3]])
        alpha = 2.4

        def fun(u: np.ndarray) -> tuple[Any, np.ndarray]:
            g = softmax(u)
            g[1] -= 1
            return logsumexp(u) - u[1] + np.sum((u - z[0]) ** 2) / (2 * alpha), g + (
                u - z[0]
            ) / alpha

        ref = minimize(fun, z[0], jac=True, method="BFGS", options={"gtol": 1e-11})
        value, grad = Solver().loss_and_grad(z, [1], "proximal_cross_entropy", alpha=alpha)
        np.testing.assert_allclose(value[0], ref.fun, atol=2e-14)
        np.testing.assert_allclose(grad[0], (z[0] - ref.x) / alpha, atol=2e-8)

    def test_uniform_targets_and_near_zero_tail(self):
        s = Solver("float64", dtype=np.float32)
        z = np.array([[20.0, 20.0, -20.0]])
        target = np.array([[True, True, False]])
        _, g = s.loss_and_grad(z, target)
        assert g[0, 0] < 0
        np.testing.assert_array_equal(g[0, :2], [g[0, 0], g[0, 0]])
        np.testing.assert_allclose(g[0, 0], -np.exp(-40) / 4, rtol=2.4e-7, atol=0)
        z = np.array([[1.0, 2.0, 3.0]])
        target = np.ones(z.shape, bool)
        _, g = Solver().loss_and_grad(z, target)
        np.testing.assert_allclose(g, softmax(z, axis=1) - 1 / 3, atol=2e-16)

    def test_large_regularization_and_scale_invariance(self):
        for precision, dtype in [
            ("float32", np.float32),
            ("float64", np.float32),
            ("float64", np.float64),
        ]:
            s = Solver(precision, dtype=dtype)
            m = s.fit(np.eye(3), [0, 1, 2], C=1e18, max_epoch=5, tol=1e-5, eval_epoch=1)
            assert m.status == "solved"
            assert m.weights is not None
            np.testing.assert_allclose(m.weights.sum(1), 0, atol=1e-6)
            assert m.weights[0, 0] - m.weights[0, 1] > 38
            base = s.fit(np.eye(3), [0, 1, 2], max_epoch=1, tol=0)
            assert base.weights is not None
            for scale in [1e4, 1e8]:
                scaled = s.fit(scale * np.eye(3), [0, 1, 2], C=1 / scale**2, max_epoch=1, tol=0)
                assert scaled.weights is not None
                np.testing.assert_allclose(
                    scale * scaled.weights, base.weights, rtol=5e-7, atol=1e-8
                )

    def test_tiny_proximal_scale_does_not_underflow_probabilities(self):
        s = Solver()
        z = [[0.0, -50.0, -50.0]]
        value, grad = s.loss_and_grad(z, [0], "proximal_cross_entropy", alpha=1e-300)
        assert value[0] > 0
        np.testing.assert_allclose(grad[0, 0], -2 * np.exp(-50), rtol=1e-13, atol=0)
        a = s.coordinate_update(z, [0], C=1e-300)
        assert a[0, 0] > 0

    def test_hinge_margins_and_smoothing_against_two_class_closed_forms(self):
        for precision, dtype, rtol in [
            ("float32", np.float32, 1e-6),
            ("float64", np.float32, 2e-7),
            ("float64", np.float64, 1e-14),
        ]:
            solver = Solver(precision, dtype=dtype)
            for z0, z1, gamma in HINGE_CASES:
                z = np.array([[z0, z1]], dtype)
                if not np.isfinite(z).all():
                    continue
                for objective, smooth in HINGES:
                    width = width_of(objective, smooth, gamma)
                    first, second = z[0].tolist()
                    value, slope = two_class_hinge(first, second, width)
                    target = [[True, False]] if objective.startswith("multilabel") else [0]
                    g_used = gamma if smooth else 0.0
                    v, g = solver.loss_and_grad(z, target, objective, gamma=g_used)
                    message = f"{precision} {objective} {z[0]} {gamma}"
                    # Errors relative to the value and to the slope (order one at the kink);
                    # float32 arithmetic rounds the value too.
                    vtol = 1e-6 if precision == "float32" else 1e-14
                    np.testing.assert_allclose(v[0], value, rtol=vtol, atol=0, err_msg=message)
                    np.testing.assert_allclose(
                        g[0], [-slope, slope], rtol=rtol, atol=0, err_msg=message
                    )

    def test_hinge_steps_resolve_tiny_margins(self):
        # A cold two-class step is [s, -s] with s = h / (2 norm2 + width / C) below C;
        # the step forms h to about an ulp of the scores (absolute).
        C, gamma = 4.0, 0.5
        for precision, dtype in [
            ("float32", np.float32),
            ("float64", np.float32),
            ("float64", np.float64),
        ]:
            solver = Solver(precision, dtype=dtype)
            for z1 in [2.0**-26, 1e-20]:
                z = np.array([[1.0, z1]], dtype)
                h = 1 + Fraction(float(z[0, 1])) - 1
                for objective, smooth in HINGES:
                    target = [[0]] if objective.startswith("multilabel") else [0]
                    for norm2 in [1.0, 0.75]:
                        width = Fraction(width_of(objective, smooth, gamma))
                        den = 2 * Fraction(norm2) + width / Fraction(C)
                        s = float(h / den)
                        a = solver.coordinate_update(
                            z, target, objective, C=C, norm2=norm2, gamma=gamma if smooth else 0.0
                        )
                        np.testing.assert_allclose(
                            a[0],
                            [s, -s],
                            rtol=4 * _eps(dtype),
                            atol=4 * _eps(dtype) * 2 / float(den),
                            err_msg=f"{precision} {objective} {z1} {norm2}",
                        )
