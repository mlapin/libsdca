"""Solver.fit on every training objective; run with pytest from the repo root."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from libsdca import HINGES, MULTILABEL, OBJECTIVES
from libsdca._core import Solver
from scipy.optimize import minimize


def blobs(n: int = 120, d: int = 5, classes: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = 4 * rng.standard_normal((classes, d))
    y = np.arange(n) % classes
    x = centers[y] + rng.standard_normal((n, d))
    return np.hstack([x, np.ones((n, 1))]), y


def label_sets(y: np.ndarray, classes: int):
    mask = np.zeros((len(y), classes), bool)
    mask[np.arange(len(y)), y] = True
    mask[np.arange(len(y)), (y + 1) % classes] = True
    return mask


def _independent_loss_and_grad(
    z: npt.ArrayLike, target: npt.ArrayLike, objective: str, k: int = 2, gamma: float = 0.5
):
    """Independent pure-NumPy loss and gradient without calling libsdca.loss_and_grad."""
    from scipy.special import logsumexp, softmax

    z = np.asarray(z, dtype=np.float64)
    n, m = z.shape
    mask = np.zeros((n, m), dtype=bool)
    t_arr = np.asarray(target)
    if t_arr.dtype == bool and t_arr.shape == (n, m):
        mask = t_arr
    else:
        mask[np.arange(n), t_arr] = True
    values = np.zeros(n, dtype=np.float64)
    grads = np.zeros((n, m), dtype=np.float64)
    if objective in ("cross_entropy", "multilabel_cross_entropy"):
        counts = mask.sum(axis=1, keepdims=True)
        q = mask / counts
        values = logsumexp(z, axis=1) - (z * q).sum(axis=1)
        grads = softmax(z, axis=1) - q
        return values, grads
    for i in range(n):
        pos = np.flatnonzero(mask[i])
        neg = np.flatnonzero(~mask[i])
        if objective == "topk_cross_entropy":
            v = z[i, neg] - z[i, pos[0]]
            order = np.argsort(-v)
            vs = v[order]
            for u in range(k):
                lse = logsumexp(vs[u:])
                if vs[u] <= lse - np.log(k - u):
                    break
            else:
                u, lse = k, 0.0
            odds = (vs[:u].sum() + (k - u) * (lse - np.log(k - u) if u < k else 0.0)) / k + np.log(
                k
            )
            values[i] = np.logaddexp(0.0, odds)
            mass = 1.0 / (1.0 + np.exp(-odds))
            p = np.zeros(len(neg))
            p[order[:u]] = mass / k
            if u < k:
                p[order[u:]] = mass * (1.0 - u / k) * np.exp(vs[u:] - lse)
            grads[i, neg] = p
            grads[i, pos[0]] = -p.sum()
        elif objective == "topk_hinge_beta":
            h = 1.0 + z[i, neg] - z[i, pos[0]]
            cap = gamma / k
            a = np.clip(h, 0.0, cap)
            if a.sum() > gamma:
                lo, hi = h.min() - cap, h.max()
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if np.clip(h - mid, 0.0, cap).sum() > gamma:
                        lo = mid
                    else:
                        hi = mid
                a = np.clip(h - 0.5 * (lo + hi), 0.0, cap)
            values[i] = np.sum((h * a - 0.5 * a * a) / gamma)
            grads[i, neg] = a / gamma
            grads[i, pos[0]] = -grads[i, neg].sum()
        elif objective == "topk_hinge_alpha":
            h = np.sort(1.0 + z[i, neg] - z[i, pos[0]])[::-1]
            # Exact active-set search over (u capped at c, m free in (0, c)) on 0 <= a_j <= sum(a)/k.
            best_a = np.zeros_like(h)
            for u in range(k + 1):
                for m in range(0 if u == k else 1, len(h) - u + 1):
                    if u == k and m == 0:
                        c = h[:k].sum() / k
                        t = h[k - 1] - c
                    else:
                        d = (k - u) ** 2 + m * u
                        su = h[:u].sum()
                        sm = h[u : u + m].sum()
                        t = (u * sm - (k - u) * su) / d
                        c = (m * su + (k - u) * sm) / d
                    if c <= 0:
                        continue
                    ok_u = u == 0 or h[u - 1] >= t + c - 1e-12
                    ok_m = m == 0 or (h[u] <= t + c + 1e-12 and h[u + m - 1] >= t - 1e-12)
                    ok_l = u + m == len(h) or h[u + m] <= t + 1e-12
                    if ok_u and ok_m and ok_l:
                        best_a = np.clip(h - t, 0.0, c)
                        break
            if best_a.sum() > gamma:
                cap = gamma / k
                lo, hi = h.min() - cap, h.max()
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if np.clip(h - mid, 0.0, cap).sum() > gamma:
                        lo = mid
                    else:
                        hi = mid
                best_a = np.clip(h - 0.5 * (lo + hi), 0.0, cap)
            order = np.argsort(-(1.0 + z[i, neg] - z[i, pos[0]]))
            a = np.empty_like(h)
            a[order] = best_a
            values[i] = np.sum((a * (1.0 + z[i, neg] - z[i, pos[0]]) - 0.5 * a * a) / gamma)
            grads[i, neg] = a / gamma
            grads[i, pos[0]] = -grads[i, neg].sum()
        elif objective == "multilabel_hinge":
            hp = 0.5 - (z[i, pos] - z[i, pos[0]])
            hn = 0.5 + (z[i, neg] - z[i, pos[0]])
            lo, hi = -hn.max() - gamma, hp.max()
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if np.maximum(0.0, hp - mid).sum() > np.maximum(0.0, hn + mid).sum():
                    lo = mid
                else:
                    hi = mid
            t = 0.5 * (lo + hi)
            ap = np.maximum(0.0, hp - t)
            an = np.maximum(0.0, hn + t)
            if ap.sum() > gamma:
                for block, out in ((hp, ap), (hn, an)):
                    blo, bhi = block.min() - gamma, block.max()
                    for _ in range(80):
                        bmid = 0.5 * (blo + bhi)
                        if np.maximum(0.0, block - bmid).sum() > gamma:
                            blo = bmid
                        else:
                            bhi = bmid
                    out[:] = np.maximum(0.0, block - 0.5 * (blo + bhi))
            values[i] = (np.sum(hp * ap - 0.5 * ap * ap) + np.sum(hn * an - 0.5 * an * an)) / gamma
            grads[i, pos] = -ap / gamma
            grads[i, neg] = an / gamma
    return values, grads


# Every objective, and each hinge both nonsmooth and smoothed (gamma = 0.5).
CASES = [(o, 0.0) for o in OBJECTIVES] + [(o, 0.5) for o in OBJECTIVES if o in HINGES]
SMOOTH = (
    "cross_entropy",
    "topk_cross_entropy",
    "topk_hinge_alpha",
    "topk_hinge_beta",
    "multilabel_hinge",
    "multilabel_cross_entropy",
)


class ObjectiveTests:
    def setup_method(self):
        self.x, self.y = blobs()
        self.mask = label_sets(self.y, 4)
        self.solver = Solver("float64")
        self.k, self.gamma, self.C = 2, 0.5, 0.7

    def target(self, objective: str):
        return self.mask if objective in MULTILABEL else self.y

    def primal(self, objective: str, w: np.ndarray, gamma: float):
        values, _ = self.solver.loss_and_grad(
            self.x @ w, self.target(objective), objective, k=self.k, gamma=gamma
        )
        return 0.5 * np.sum(w**2) + self.C * values.sum()

    def test_every_objective_trains(self, subtests: pytest.Subtests):
        for objective, gamma in CASES:
            with subtests.test(objective=objective, gamma=gamma):
                smooth = gamma > 0 or "cross_entropy" in objective
                tol = 1e-6 if smooth else 1e-3
                model = self.solver.fit(
                    self.x,
                    self.target(objective),
                    C=self.C,
                    objective=objective,
                    k=self.k,
                    gamma=gamma,
                    tol=tol,
                    max_epoch=5000,
                    eval_epoch=5,
                )
                assert model.status == "solved", objective
                assert model.history["relative_gap"][-1] <= tol
                assert model.weights is not None
                assert abs(
                    model.history["primal"][-1] - self.primal(objective, model.weights, gamma)
                ) <= 1e-9 * abs(model.history["primal"][-1])
                metric = "rank_loss" if objective in MULTILABEL else "accuracy"
                assert metric in model.history
                if objective in MULTILABEL:
                    assert model.history["rank_loss"][-1] < 0.1
                else:
                    assert (model.predict(self.x) == self.y).mean() > 0.7

    @pytest.mark.parametrize("objective", SMOOTH)
    def test_smooth_objectives_match_independent_optimizer(self, objective: str):
        model = self.solver.fit(
            self.x,
            self.target(objective),
            C=self.C,
            objective=objective,
            k=self.k,
            gamma=self.gamma,
            tol=1e-9,
            max_epoch=5000,
            eval_epoch=5,
        )
        target = self.target(objective)

        def fun(flat: np.ndarray, objective: str = objective):
            w = flat.reshape(self.x.shape[1], 4)
            values, grad = _independent_loss_and_grad(
                self.x @ w, target, objective, k=self.k, gamma=self.gamma
            )
            return (
                0.5 * np.sum(w**2) + self.C * values.sum(),
                (w + self.C * self.x.T @ grad).ravel(),
            )

        reference = minimize(
            fun,
            np.zeros(self.x.shape[1] * 4),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10},
        )
        assert model.status == "solved", objective
        assert abs(model.history["primal"][-1] - reference.fun) <= 1e-6 * abs(reference.fun)

    def test_warm_start_continues_and_recognizes_solutions(self):
        # Every epoch evaluated, the warm start included.
        every: dict[str, Any] = dict(eval_epoch=1, eval_on_start=True)
        partial = self.solver.fit(self.x, self.y, C=self.C, max_epoch=3, tol=0, **every)
        assert partial.status == "max_epoch"
        resumed = self.solver.fit(
            self.x, self.y, C=self.C, initial_dual=partial.dual, tol=1e-8, max_epoch=1000, **every
        )
        assert resumed.status == "solved"
        assert round(abs(resumed.history["primal"][0] - partial.history["primal"][-1]), 7) == 0
        assert resumed.history["primal"][-1] < partial.history["primal"][-1]
        assert resumed.weights is not None
        np.testing.assert_allclose(resumed.weights, self.x.T @ resumed.dual, rtol=1e-10, atol=1e-12)
        again = self.solver.fit(
            self.x, self.y, C=self.C, initial_dual=resumed.dual, tol=1e-8, **every
        )
        assert (again.status, again.epochs) == ("solved", 0)
        with pytest.raises(ValueError):
            self.solver.fit(self.x, self.y, C=self.C, initial_dual=partial.dual[:, :3])

    def test_multilabel_inputs_and_validation(self):
        lists = [np.flatnonzero(row) for row in self.mask]
        a = self.solver.fit(self.x, self.mask, objective="multilabel_hinge", max_epoch=20, tol=0)
        b = self.solver.fit(self.x, lists, objective="multilabel_hinge", max_epoch=20, tol=0)
        np.testing.assert_array_equal(a.weights, b.weights)
        bad = self.mask.copy()
        bad[0] = True
        with pytest.raises(ValueError):
            self.solver.fit(self.x, bad, objective="multilabel_hinge")
        single = self.solver.fit(self.x, self.y, objective="multilabel_hinge", max_epoch=20, tol=0)
        one_hot = np.zeros_like(self.mask)
        one_hot[np.arange(len(self.y)), self.y] = True
        np.testing.assert_array_equal(
            single.weights,
            self.solver.fit(
                self.x, one_hot, objective="multilabel_hinge", max_epoch=20, tol=0
            ).weights,
        )
        with pytest.raises(ValueError):
            self.solver.fit(self.x, [[0, "a"]] * len(self.y), objective="multilabel_hinge")
        with pytest.raises(ValueError):
            self.solver.fit(self.x, self.y, objective="topk_cross_entropy", k=4)

    def test_precisions_and_kernel_agree(self):
        reference = self.solver.fit(
            self.x, self.y, C=self.C, objective="topk_cross_entropy", k=2, tol=1e-9, max_epoch=5000
        )
        for precision in ("float32", "float64"):
            model = Solver(precision, dtype=np.float32).fit(
                self.x,
                self.y,
                C=self.C,
                objective="topk_cross_entropy",
                k=2,
                tol=1e-5,
                max_epoch=5000,
            )
            assert model.weights is not None and reference.weights is not None
            np.testing.assert_allclose(model.weights, reference.weights, rtol=1e-2, atol=1e-3)
        kernel = self.solver.fit(
            self.x @ self.x.T,
            self.y,
            C=self.C,
            objective="topk_cross_entropy",
            k=2,
            tol=1e-9,
            max_epoch=5000,
            kernel=True,
        )
        np.testing.assert_allclose(
            kernel.decision_function(self.x @ self.x.T),
            reference.decision_function(self.x),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_kernel_and_warm_start_every_objective(self, subtests: pytest.Subtests):
        gram = self.x @ self.x.T
        for objective, gamma in CASES:
            with subtests.test(objective=objective, gamma=gamma):
                opts: dict[str, Any] = dict(
                    C=self.C,
                    objective=objective,
                    k=self.k,
                    gamma=gamma,
                    tol=1e-7,
                    max_epoch=5000,
                    eval_epoch=5,
                    eval_on_start=True,  # a warm start at the optimum stops at once
                )
                model = self.solver.fit(self.x, self.target(objective), **opts)
                kernel = self.solver.fit(gram, self.target(objective), kernel=True, **opts)
                assert (model.status, kernel.status) == ("solved", "solved")
                assert model.weights is not None
                np.testing.assert_allclose(model.weights, self.x.T @ model.dual, atol=1e-12)
                np.testing.assert_allclose(
                    self.x @ model.weights, kernel.decision_function(gram), rtol=1e-3, atol=1e-3
                )
                warm = self.solver.fit(
                    self.x, self.target(objective), initial_dual=model.dual, **opts
                )
                assert warm.status == "solved"
                assert warm.epochs <= 5

    def test_smooth_coordinate_kkt_with_feasible_old_blocks(self):
        rng = np.random.default_rng(96)
        for objective in (
            "cross_entropy",
            "topk_cross_entropy",
            "topk_hinge_alpha",
            "topk_hinge_beta",
            "multilabel_hinge",
            "multilabel_cross_entropy",
        ):
            for trial in range(20):
                z = 3 * rng.normal(size=(1, 5))
                y = np.array([[1, 0, 1, 0, 0]], bool) if objective in MULTILABEL else [0]
                c, norm = 10.0 ** (trial % 5 - 2), 10.0 ** (trial % 3 - 1)
                opts: dict[str, Any] = dict(C=c, norm2=norm, objective=objective, k=2, gamma=0.6)
                old = self.solver.coordinate_update(z, y, **opts)
                a = self.solver.coordinate_update(z, y, previous=old, **opts)
                new_scores = z + norm * (a - old)
                _, grad = _independent_loss_and_grad(new_scores, y, objective, k=2, gamma=0.6)
                np.testing.assert_allclose(
                    a, -c * grad, rtol=2e-7, atol=2e-8, err_msg=f"{objective} trial={trial}"
                )

    def test_hinge_coordinates_against_quadratic_program(self):
        rng = np.random.default_rng(6)
        for objective, smooth in (
            ("topk_hinge_alpha", 0.0),
            ("topk_hinge_beta", 0.0),
            ("topk_hinge_alpha", 0.6),
            ("topk_hinge_beta", 0.6),
        ):
            for k in (1, 2, 4):
                for _ in range(8):
                    scores = 3 * rng.normal(size=5)
                    C, norm, gamma = 0.7, 1.3, smooth

                    def dual(u: np.ndarray):
                        a = np.r_[u.sum(), -u]
                        value = (
                            -u.sum()
                            + smooth / (2 * C) * np.dot(u, u)
                            + scores @ a
                            + norm / 2 * np.dot(a, a)
                        )
                        grad = -1 + smooth / C * u + scores[0] - scores[1:] + norm * (u + u.sum())
                        return value, grad

                    constraints = [{"type": "ineq", "fun": lambda u: C - u.sum()}]
                    if objective.startswith("topk_hinge_alpha"):
                        constraints.append({"type": "ineq", "fun": lambda u: u.sum() / k - u})
                    bounds = [(0.0, C / k if objective.startswith("topk_hinge_beta") else C)] * 4
                    ref = minimize(
                        dual,
                        np.zeros(4),
                        jac=True,
                        bounds=bounds,
                        constraints=constraints,
                        method="SLSQP",
                        options={"ftol": 1e-12, "maxiter": 500},
                    )
                    a = self.solver.coordinate_update(
                        scores[None], [0], C=C, norm2=norm, objective=objective, k=k, gamma=gamma
                    )[0]
                    assert ref.success, ref.message
                    np.testing.assert_allclose(-a[1:], ref.x, atol=2e-6)

    def test_multilabel_coordinates_against_quadratic_program(self):
        rng = np.random.default_rng(806)
        target = np.array([[1, 1, 0, 0, 0]], bool)
        sign = np.array([1.0, 1.0, -1.0, -1.0, -1.0])
        for smooth in (False, True):
            name = "multilabel_hinge"
            c, norm, gamma = 0.7, 1.3, 0.6 if smooth else 0.0
            for _ in range(12):
                scores = 3 * rng.normal(size=5)

                def dual(u: np.ndarray):
                    linear = -0.5 + scores * sign
                    curvature = norm + gamma / c
                    return linear @ u + curvature / 2 * (u @ u), linear + curvature * u

                constraints = [
                    {"type": "eq", "fun": lambda u: sign @ u},
                    {"type": "ineq", "fun": lambda u: c - u[:2].sum()},
                ]
                ref = minimize(
                    dual,
                    np.zeros(5),
                    jac=True,
                    bounds=[(0.0, c)] * 5,
                    constraints=constraints,
                    method="SLSQP",
                    options={"ftol": 1e-12, "maxiter": 500},
                )
                a = self.solver.coordinate_update(
                    scores[None], target, C=c, norm2=norm, objective=name, gamma=gamma
                )[0]
                assert ref.success, ref.message
                np.testing.assert_allclose(a, sign * ref.x, atol=2e-6)

    def test_warm_start_validation_and_center(self):
        s = Solver("float64", dtype=np.float32)
        x, y = np.eye(3), np.arange(3)
        model = s.fit(x, y)
        center = np.ones((3, 3), np.float32)
        warm = s.fit(x, y, center=center, initial_dual=model.dual, max_epoch=0)
        assert warm.weights is not None
        np.testing.assert_allclose(warm.weights, center + x.T @ model.dual, atol=1e-7)
        for invalid in (
            lambda a: a.__setitem__((0, 0), a[0, 0] + 0.01),  # not zero-sum
            lambda a: a.__setitem__((0, 1), -a[0, 1]),  # wrong sign
            lambda a: a.__imul__(10),
        ):  # outside [0, C]
            a = model.dual.copy()
            invalid(a)
            with pytest.raises(ValueError):
                s.fit(x, y, initial_dual=a)
        with pytest.raises(ValueError):
            s.fit(
                x,
                y,
                objective="topk_hinge_alpha",
                k=2,
                initial_dual=np.array(
                    [[1.0, -0.2, -0.8], [-0.2, 1.0, -0.8], [-0.8, -0.2, 1.0]], np.float32
                ),
            )
        with pytest.raises(ValueError):
            s.fit(x, np.array([0.2, 1.0, 2.0]))
        with pytest.raises(ValueError):
            s.fit(x, y, objective="proximal_cross_entropy")

    def test_shifted_scores_and_multilabel_residuals(self):
        for precision, dtype in (
            ("float32", np.float32),
            ("float64", np.float32),
            ("float64", np.float64),
        ):
            s = Solver(precision, dtype=dtype)
            for objective, smooth in CASES:
                y = [[True, True, False]] if objective in MULTILABEL else [0]
                gamma = 0.7 if smooth else 0.0
                value, grad = s.loss_and_grad([[1e16] * 3], y, objective, k=2, gamma=gamma)
                reference = s.loss_and_grad([[0.0] * 3], y, objective, k=2, gamma=gamma)
                np.testing.assert_allclose(value, reference[0], rtol=1e-6)
                np.testing.assert_allclose(grad, reference[1], rtol=1e-6, atol=1e-7)
            # float32 solves in float: residuals hold to float relative precision.
            rtol = 1e-5 if precision == "float32" else 1e-6
            a = s.coordinate_update(
                [[20.0, 20.0, -20.0]], [[True, True, False]], objective="multilabel_cross_entropy"
            )
            np.testing.assert_allclose(a[0, :2], np.exp(-40) / 4, rtol=rtol, atol=0)
            a = s.coordinate_update(
                [[0.0, 1e-10, -40.0]], [[True, True, False]], objective="multilabel_cross_entropy"
            )
            np.testing.assert_allclose(
                a[0, :2], [1e-10 / 6, -1e-10 / 6], rtol=max(rtol, 2e-7), atol=0
            )

    def test_smooth_trainers_against_independent_optimizer(self):
        # L-BFGS training cross-check on the canonical API.
        rng = np.random.default_rng(2026)
        x = rng.normal(size=(40, 5))
        labels = np.arange(40) % 4
        mask = np.eye(4, dtype=bool)[labels] | np.eye(4, dtype=bool)[(labels + 1) % 4]
        solver, c = Solver(), 0.7
        for objective in [
            "cross_entropy",
            "topk_cross_entropy",
            "topk_hinge_alpha",
            "topk_hinge_beta",
            "multilabel_hinge",
            "multilabel_cross_entropy",
        ]:
            target = mask if objective.startswith("multilabel") else labels
            options: dict[str, Any] = dict(objective=objective, k=2, gamma=0.6)

            def fun(flat: np.ndarray, objective: str = objective):
                w = flat.reshape(5, 4)
                values, grad = _independent_loss_and_grad(x @ w, target, objective, k=2, gamma=0.6)
                return 0.5 * np.sum(w * w) + c * values.sum(), (w + c * x.T @ grad).ravel()

            ref = minimize(
                fun,
                np.zeros(20),
                jac=True,
                method="L-BFGS-B",
                options=dict(maxiter=3000, ftol=1e-15, gtol=1e-10),
            )
            model = solver.fit(x, target, C=c, tol=1e-9, max_epoch=3000, **options)
            assert model.status == "solved"
            assert model.weights is not None
            assert abs(fun(model.weights.ravel())[0] - ref.fun) <= 1e-7
