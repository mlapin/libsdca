"""The `solved` certificate against an independent high-precision primal and dual."""

import mpmath as mp
import numpy as np
import pytest
from conformance import references
from conformance.references import Loss
from named_losses import HINGES, MULTILABEL, OBJECTIVES
from numpy_api import Solver


class CertificateTests:
    def test_solved_meets_tol_in_high_precision(self, subtests: pytest.Subtests):
        # `solved` must hold in high precision despite negative float gaps,
        # float scores and drifting float weights.
        rng = np.random.default_rng(0)
        x = rng.normal(size=(64, 12)).astype(np.float32)
        y = rng.integers(0, 4, size=64)
        tol = 1e-10
        with mp.workdps(50):

            def primal(w: np.ndarray | None):
                assert w is not None
                return references.primal(Loss("cross_entropy"), x, w.T, y, 1.0)

            # Any feasible weights bound the optimum from above.
            reference = Solver("float64").fit(x, y, tol=1e-13, eval_epoch=10, max_epoch=1000)
            best = primal(reference.weights)
            for precision, dtype in [
                ("float32", np.float32),
                ("float64", np.float32),
                ("float64", np.float64),
            ]:
                for recompute in (True, False):
                    model = Solver(precision, dtype=dtype).fit(
                        x, y, tol=tol, eval_epoch=10, max_epoch=2000, recompute_primal=recompute
                    )
                    with subtests.test(
                        precision=precision, dtype=dtype, recompute=recompute, status=model.status
                    ):
                        if dtype == np.float64:
                            assert model.status == "solved"
                        if model.status == "solved":
                            value = primal(model.weights)
                            assert float((value - best) / value) <= tol
                            assert model.history["relative_gap"][-1] <= tol

    def test_kernel_primal_is_at_the_returned_float32_duals(self, subtests: pytest.Subtests):
        # The primal of a kernel fit is evaluated at the duals it returns, not at
        # their repaired copy, so a float32 Gram fit reports solved only within tol.
        def primal(K: np.ndarray, A: np.ndarray, y: np.ndarray, C: float, k: int) -> float:
            Z = K @ A
            values, _ = references.evaluate(Loss("topk_hinge_alpha", k=k), Z, y)
            return 0.5 * float(np.sum(A * Z)) + C * float(values.sum())

        rng = np.random.default_rng(8)
        for case in range(8):
            n, m, k, C = 80, 7, 2, float(rng.uniform(0.5, 2))
            f = rng.normal(size=(n, 10)) + 2 * rng.normal(size=(m, 10))[rng.integers(0, m, n)]
            gram = (f @ f.T / 10).astype(np.float32)
            y = rng.integers(0, m, n)
            y[:m] = np.arange(m)
            model = Solver("float64", dtype=np.float32).fit(
                gram,
                y,
                "topk_hinge_alpha",
                C=C,
                k=k,
                tol=1e-5,
                max_epoch=20000,
                eval_epoch=5,
                kernel=True,
            )
            with subtests.test(case=case, status=model.status):
                value = primal(gram.astype(np.float64), model.dual.astype(np.float64), y, C, k)
                assert abs(model.history["primal"][-1] - value) <= 1e-10 * value
                if model.status == "solved":
                    dual = model.history["dual"][-1]
                    assert value - dual <= 1e-5 * max(abs(value), abs(dual)) * (1 + 1e-9)

    def test_solved_certificate_across_objectives_centers_and_kernels(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(24, 5)).astype(np.float32)
        y = rng.integers(0, 4, size=24)
        mask = np.zeros((24, 4), dtype=bool)
        mask[np.arange(24), y] = True
        mask[np.arange(24), (y + 1) % 4] = True
        center = (0.3 * rng.normal(size=(5, 4))).astype(np.float32)
        c, k, tol = 0.8, 2, 1e-6
        with mp.workdps(50):
            cases = [(o, 0.0) for o in OBJECTIVES] + [(o, 0.6) for o in OBJECTIVES if o in HINGES]
            for objective, gamma in cases:
                target = mask if objective in MULTILABEL else y

                def exact_primal(w: np.ndarray | None, w0: np.ndarray | None = None):
                    assert w is not None
                    loss = references.named(objective, k, gamma)
                    return references.primal(loss, x, w.T, target, c, w0)

                ref = Solver("float64").fit(
                    x,
                    target,
                    objective,
                    C=c,
                    k=k,
                    gamma=gamma,
                    tol=1e-11,
                    max_epoch=4000,
                    eval_epoch=5,
                )
                best = exact_primal(ref.weights)
                for precision, dtype in [
                    ("float32", np.float32),
                    ("float64", np.float32),
                    ("float64", np.float64),
                ]:
                    model = Solver(precision, dtype=dtype).fit(
                        x,
                        target,
                        objective,
                        C=c,
                        k=k,
                        gamma=gamma,
                        tol=tol,
                        max_epoch=3000,
                        eval_epoch=5,
                    )
                    if model.status == "solved":
                        val = exact_primal(model.weights)
                        assert abs(float(val) - model.history["primal"][-1]) <= 1e-6 * abs(
                            float(val)
                        )
                        assert float((val - best) / abs(val)) <= tol * 1.05, (
                            f"{objective} {precision} {dtype}"
                        )
                if gamma == 0 and objective in (
                    "cross_entropy",
                    "topk_hinge_beta",
                    "multilabel_cross_entropy",
                ):
                    ref_c = Solver("float64").fit(
                        x,
                        target,
                        objective,
                        C=c,
                        k=k,
                        gamma=gamma,
                        center=center.T,
                        tol=1e-11,
                        max_epoch=4000,
                    )
                    best_c = exact_primal(ref_c.weights, center)
                    mod_c = Solver("float64", dtype=np.float32).fit(
                        x,
                        target,
                        objective,
                        C=c,
                        k=k,
                        gamma=gamma,
                        center=center.T,
                        tol=tol,
                        max_epoch=3000,
                    )
                    if mod_c.status == "solved":
                        val_c = exact_primal(mod_c.weights, center)
                        assert (
                            float((val_c - best_c) / max(abs(val_c), abs(best_c))) <= tol * 1.05
                        ), f"{objective} center"

    def test_topk_entropy_solved_with_free_scores_far_below_the_largest(self):
        # At C = 1e6 the largest competitors end up hundreds above the first free score,
        # where the coordinate step solves its block again relative to that score. The fit
        # still certifies `solved` against an independent 50-digit primal.
        rng = np.random.default_rng(7)
        x = rng.normal(size=(16, 3))
        y = rng.integers(0, 5, size=16)
        c, k, tol = 1e6, 3, 1e-9
        model = Solver("float64").fit(
            x, y, "topk_cross_entropy", C=c, k=k, tol=tol, eval_epoch=5, max_epoch=4000
        )
        assert model.status == "solved"
        with mp.workdps(50):
            assert model.weights is not None
            primal = references.primal(Loss("topk_cross_entropy", k=k), x, model.weights.T, y, c)
            w = mp.matrix(model.weights.T.tolist())
            z = mp.matrix(x.tolist()) * w
            rows = [[z[i, j] for j in range(5)] for i in range(16)]
            # The first free score of some rows lies far below the largest competitor.
            far = 0
            for i, row in enumerate(rows):
                v = sorted((row[j] - row[y[i]] for j in range(5) if j != y[i]), reverse=True)
                u = next(
                    u
                    for u in range(k)
                    if v[u]
                    <= v[u] + mp.log(mp.fsum(mp.exp(t - v[u]) for t in v[u:])) - mp.log(k - u)
                )
                far += v[0] - v[u] > 64
            assert far > 0
        dual = model.history["dual"][-1]
        assert float(primal - dual) <= tol * max(abs(float(primal)), abs(dual))
        assert round(abs(float(primal) / model.history["primal"][-1] - 1.0), 12) == 0
