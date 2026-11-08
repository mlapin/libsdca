"""The `solved` certificate against an independent high-precision primal and dual."""

from collections.abc import Iterable
from typing import Any

import mpmath as mp
import numpy as np
import pytest
from libsdca import HINGES, MULTILABEL, OBJECTIVES
from libsdca._core import Solver
from mpmath.matrices.matrices import _matrix
from mpmath_types import MPReal


def _rsum(terms: Iterable[Any]) -> MPReal | int | float:
    """mp.fsum of real terms."""
    total = mp.fsum(terms)
    assert isinstance(total, mp.mpf)
    return total


def _mp_primal_loss(
    row: list[MPReal], pos: list[int], objective: str, k: int, gamma: float
) -> MPReal | int | float:
    """50-digit per-example unregularized primal loss directly from the mathematical definition."""
    g = mp.mpf(gamma)
    if objective == "multiclass_hinge":
        objective, k = "topk_hinge_alpha", 1
    if objective in ("cross_entropy", "multilabel_cross_entropy"):
        top = max(row)
        lse = top + mp.log(mp.fsum(mp.exp(v - top) for v in row))
        return lse - mp.fsum(row[j] for j in pos) / len(pos)
    neg = [j for j in range(len(row)) if j not in pos]
    if objective == "topk_cross_entropy":
        z0 = row[pos[0]]
        v = sorted([row[j] - z0 for j in neg], reverse=True)
        for u in range(k):
            top = v[u]
            lse = top + mp.log(mp.fsum(mp.exp(x - top) for x in v[u:]))
            if v[u] <= lse - mp.log(k - u):
                break
        else:
            u, lse = k, mp.mpf(0)
        odds = (mp.fsum(v[:u]) + (k - u) * (lse - mp.log(k - u) if u < k else 0)) / k + mp.log(k)
        return odds + mp.log1p(mp.exp(-odds)) if odds > 0 else mp.log1p(mp.exp(odds))
    if objective in ("topk_hinge_alpha", "topk_hinge_beta") and gamma == 0:
        h = sorted([1 + row[j] - row[pos[0]] for j in neg], reverse=True)[:k]
        if objective == "topk_hinge_beta":
            return mp.fsum(max(mp.mpf(0), x) for x in h) / k
        return max(mp.mpf(0), mp.fsum(h) / k)
    if objective == "multilabel_hinge" and gamma == 0:
        return max(mp.mpf(0), 1 + max(row[j] for j in neg) - min(row[j] for j in pos))
    if objective == "topk_hinge_beta":
        h = [1 + row[j] - row[pos[0]] for j in neg]
        cap = g / k
        if mp.fsum(min(cap, max(mp.mpf(0), x)) for x in h) <= g:
            a = [min(cap, max(mp.mpf(0), x)) for x in h]
        else:
            lo, hi = min(h) - cap, max(h)
            for _ in range(220):
                mid = (lo + hi) / 2
                if mp.fsum(min(cap, max(mp.mpf(0), x - mid)) for x in h) > g:
                    lo = mid
                else:
                    hi = mid
            t = (lo + hi) / 2
            a = [min(cap, max(mp.mpf(0), x - t)) for x in h]
        return _rsum((h[j] - a[j] / 2) * (a[j] / g) for j in range(len(h)))
    if objective == "topk_hinge_alpha":
        h = sorted([1 + row[j] - row[pos[0]] for j in neg], reverse=True)
        best_a = [mp.mpf(0)] * len(h)
        for u in range(k + 1):
            for m in range(0 if u == k else 1, len(h) - u + 1):
                if u == k and m == 0:
                    c = mp.fsum(h[:k]) / k
                    t = h[k - 1] - c
                else:
                    d = mp.mpf((k - u) ** 2 + m * u)
                    su = mp.fsum(h[:u])
                    sm = mp.fsum(h[u : u + m])
                    t = (u * sm - (k - u) * su) / d
                    c = (m * su + (k - u) * sm) / d
                if c <= 0:
                    continue
                tol_c = mp.mpf("1e-35") * max(mp.mpf(1), abs(c))
                ok_u = u == 0 or h[u - 1] >= t + c - tol_c
                ok_m = m == 0 or (h[u] <= t + c + tol_c and h[u + m - 1] >= t - tol_c)
                ok_l = u + m == len(h) or h[u + m] <= t + tol_c
                if ok_u and ok_m and ok_l:
                    best_a = [min(c, max(mp.mpf(0), x - t)) for x in h]
                    break
        if mp.fsum(best_a) > g:
            cap = g / k
            lo, hi = min(h) - cap, max(h)
            for _ in range(200):
                mid = (lo + hi) / 2
                if mp.fsum(min(cap, max(mp.mpf(0), x - mid)) for x in h) > g:
                    lo = mid
                else:
                    hi = mid
            t = (lo + hi) / 2
            best_a = [min(cap, max(mp.mpf(0), x - t)) for x in h]
        return _rsum((h[j] - best_a[j] / 2) * (best_a[j] / g) for j in range(len(h)))
    # multilabel_hinge, gamma > 0
    hp = [mp.mpf("0.5") - (row[j] - row[pos[0]]) for j in pos]
    hn = [mp.mpf("0.5") + (row[j] - row[pos[0]]) for j in neg]
    lo, hi = -max(hn) - g, max(hp)
    for _ in range(220):
        mid = (lo + hi) / 2
        if _rsum(max(mp.mpf(0), x - mid) for x in hp) > _rsum(max(mp.mpf(0), x + mid) for x in hn):
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    s_eq = mp.fsum(max(mp.mpf(0), x - t) for x in hp)
    if s_eq <= g:
        ap = [max(mp.mpf(0), x - t) for x in hp]
        an = [max(mp.mpf(0), x + t) for x in hn]
    else:
        ap, an = [], []
        for block in (hp, hn):
            blo, bhi = min(block) - g, max(block)
            for _ in range(220):
                bmid = (blo + bhi) / 2
                if mp.fsum(max(mp.mpf(0), x - bmid) for x in block) > g:
                    blo = bmid
                else:
                    bhi = bmid
            bt = (blo + bhi) / 2
            (ap if block is hp else an).extend(max(mp.mpf(0), x - bt) for x in block)
    return mp.fsum((hp[i] - ap[i] / 2) * (ap[i] / g) for i in range(len(hp))) + mp.fsum(
        (hn[j] - an[j] / 2) * (an[j] / g) for j in range(len(hn))
    )


class CertificateTests:
    def test_solved_meets_tol_in_high_precision(self, subtests: pytest.Subtests):
        # These fits reported `solved` 2e-9 above the optimum, from negative
        # float gaps, float scores and drifting float weights.
        rng = np.random.default_rng(0)
        x = rng.normal(size=(64, 12)).astype(np.float32)
        y = rng.integers(0, 4, size=64)
        tol = 1e-10
        with mp.workdps(50):
            xx = mp.matrix(x.astype(np.float64).tolist())

            def primal(w: np.ndarray | None):
                assert w is not None
                ww = mp.matrix(np.asarray(w, dtype=np.float64).tolist())
                z = xx * ww
                loss = mp.mpf(0)
                for i in range(len(y)):
                    row = [z[i, j] for j in range(4)]
                    top = max(row)
                    loss += top - row[y[i]] + mp.log(mp.fsum(mp.exp(v - top) for v in row))
                return mp.fsum(v * v for v in ww) / 2 + loss

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
        # their repaired copy: a float32 Gram fit reported solved 1.75x above tol.
        def primal(K: np.ndarray, A: np.ndarray, y: np.ndarray, C: float, k: int) -> float:
            Z = K @ A
            margins = 1 + Z - Z[np.arange(len(y)), y][:, None]
            margins[np.arange(len(y)), y] = -np.inf
            top = -np.sort(-margins, axis=1)[:, :k]
            return 0.5 * float(np.sum(A * Z)) + C * float(np.maximum(top.mean(1), 0).sum())

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
            xx = mp.matrix(x.astype(np.float64).tolist())
            cc = mp.matrix(center.astype(np.float64).tolist())
            cases = [(o, 0.0) for o in OBJECTIVES] + [(o, 0.6) for o in OBJECTIVES if o in HINGES]
            for objective, gamma in cases:
                target = mask if objective in MULTILABEL else y
                pos_rows = [
                    np.flatnonzero(mask[i]).tolist() if objective in MULTILABEL else [int(y[i])]
                    for i in range(24)
                ]

                def exact_primal(w: np.ndarray | None, w0: _matrix | None = None):
                    assert w is not None
                    ww = mp.matrix(np.asarray(w, dtype=np.float64).tolist())
                    z = xx * ww
                    reg = mp.fsum(v * v for v in ww) / 2
                    if w0 is not None:
                        reg = (
                            mp.fsum(
                                (ww[r, col] - w0[r, col]) ** 2 for r in range(5) for col in range(4)
                            )
                            / 2
                        )
                    loss = mp.fsum(
                        _mp_primal_loss(
                            [z[i, j] for j in range(4)], pos_rows[i], objective, k, gamma
                        )
                        for i in range(24)
                    )
                    return reg + mp.mpf(c) * loss

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
                        center=center,
                        tol=1e-11,
                        max_epoch=4000,
                    )
                    best_c = exact_primal(ref_c.weights, cc)
                    mod_c = Solver("float64", dtype=np.float32).fit(
                        x,
                        target,
                        objective,
                        C=c,
                        k=k,
                        gamma=gamma,
                        center=center,
                        tol=tol,
                        max_epoch=3000,
                    )
                    if mod_c.status == "solved":
                        val_c = exact_primal(mod_c.weights, cc)
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
            w = mp.matrix(model.weights.tolist())
            z = mp.matrix(x.tolist()) * w
            rows = [[z[i, j] for j in range(5)] for i in range(16)]
            loss = mp.fsum(
                _mp_primal_loss(row, [int(y[i])], "topk_cross_entropy", k, 1.0)
                for i, row in enumerate(rows)
            )
            primal = mp.fsum(v * v for v in w) / 2 + c * loss
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
