"""Top-k entropy and smoothed top-k hinge steps against independent references."""

import mpmath as mp
import numpy as np
import pytest
from libsdca._core import Solver
from scipy.optimize import minimize

# Whether long double extends double (x87, binary128). Where it is double (Apple
# arm64), the extended retries have the range of double: beyond it a block raises
# (docs/numerics.md, platform ranges).
_LONG_DOUBLE = np.finfo(np.longdouble)
WIDE_LONG_DOUBLE = _LONG_DOUBLE.nmant >= 63 and _LONG_DOUBLE.maxexp > 1024


def all_capped_reference(scores: np.ndarray, previous: np.ndarray, c: float, norm2: float = 1.0):
    # Average exact represented inputs before forming any rounded differences.
    # Solve logit(s) + alpha*(1+1/k)*s = mean + log(k) by bisection.
    with mp.workdps(160):
        k = len(scores) - 1
        z, a = [mp.mpf(float(x)) for x in scores], [mp.mpf(float(x)) for x in previous]
        mean = sum(z[1:]) / k - z[0] - mp.mpf(norm2) * (sum(a[1:]) / k - a[0])
        coefficient = mp.mpf(c) * norm2 * (1 + mp.mpf(1) / k)
        rhs = mean + mp.log(k)
        lo, hi = rhs - coefficient, rhs
        # Include the bracket's exponent: 1e308 needs over 1024 halvings
        # merely to reach unit width, before resolving the actual root.
        for _ in range(max(600, int(mp.log(coefficient, 2)) + 600)):
            mid = (lo + hi) / 2
            if mid == lo or mid == hi:
                break
            mass = 1 / (1 + mp.exp(-mid))
            if mid + coefficient * mass < rhs:
                lo = mid
            else:
                hi = mid
            if hi == lo:
                break
        mass = mp.mpf(c) / (1 + mp.exp(-(lo + hi) / 2))
        return np.array([float(mass)] + [-float(mass / k)] * k)


def capped_softmax_dual(z: np.ndarray, a: np.ndarray, c: float, k: int):
    """C*[mass, -p] of the capped softmax at the updated scores z + a, in 90 digits."""
    with mp.workdps(90):
        v = [mp.mpf(float(z[j])) + mp.mpf(float(a[j])) for j in range(len(z))]
        v = [x - v[0] for x in v[1:]]
        order = sorted(range(len(v)), key=lambda j: -v[j])
        u, free, lse = 0, order, mp.mpf("nan")  # rebound by the loop: k >= 1
        for u in range(k):
            free = order[u:]
            lse = v[order[u]] + mp.log(sum(mp.exp(v[j] - v[order[u]]) for j in free))
            if v[order[u]] <= lse - mp.log(k - u):
                break
        odds = (sum(v[j] for j in order[:u]) + (k - u) * (lse - mp.log(k - u))) / k + mp.log(k)
        mass = 1 / (1 + mp.exp(-odds))
        p = [mp.mpf(0)] * len(v)
        for j in order[:u]:
            p[j] = mass / k
        for j in free:
            p[j] = mass * (1 - mp.mpf(u) / k) * mp.exp(v[j] - lse)
        return np.array([float(c * mass)] + [-float(c * x) for x in p])


def topk_entropy_reference(scores: np.ndarray, y: int, k: int):
    """Top-k entropy loss and gradient of one row in 60 digits, from its active set.

    Log-sum-exps are relative to the first free score: at 60 digits an absolute one
    of scores near 1e300 would carry errors near 1e240."""
    with mp.workdps(60):
        z = [mp.mpf(float(x)) for x in scores]
        others = [j for j in range(len(z)) if j != y]
        v = {j: z[j] - z[y] for j in others}
        order = sorted(others, key=lambda j: -v[j])
        u, top, lse = 0, mp.mpf("nan"), mp.mpf("nan")  # rebound by the loop: k >= 1
        for u in range(k):
            top = v[order[u]]
            lse = mp.log(sum(mp.exp(v[j] - top) for j in order[u:]))
            if lse - mp.log(k - u) >= 0:
                break
        odds = (sum(v[j] for j in order[:u]) + (k - u) * (top + lse - mp.log(k - u))) / k
        odds += mp.log(k)
        mass = 1 / (1 + mp.exp(-odds))
        gradient = [mp.mpf(0)] * len(z)
        for i, j in enumerate(order):
            share = mp.exp(v[j] - top - lse)
            gradient[j] = mass / k if i < u else mass * (k - u) / k * share
        gradient[y] = -sum(gradient, mp.mpf(0))
        return float(mp.log1p(mp.exp(odds))), np.array([float(g) for g in gradient])


class TopkReferenceTests:
    def test_alpha_hinge_excludes_the_target_zero_margin(self, subtests: pytest.Subtests):
        # Proposition 11's reduced non-target convention: top2([1, -3])
        # averages to -1, so alpha and its gamma=1 smoothing are both zero.
        # Including the target's zero margin would instead give .5 and .25.
        for storage, precision in (
            (np.float32, "float32"),
            (np.float32, "float64"),
            (np.float64, "float64"),
        ):
            solver = Solver(precision, dtype=storage)
            z = np.array([[0, 0, -4]], dtype=storage)
            for gamma in (0.0, 1.0):
                with subtests.test(storage=storage, precision=precision, gamma=gamma):
                    value, gradient = solver.loss_and_grad(
                        z, [0], "topk_hinge_alpha", k=2, gamma=gamma
                    )
                    np.testing.assert_array_equal(value, [0.0])
                    np.testing.assert_array_equal(gradient, np.zeros_like(z))

    def test_single_solve_split_boundary(self):
        # The exact-low-part path switches when a centered score exceeds 64.
        # Check adjacent represented inputs against the independent fixed point.
        for (precision, dtype), tolerance in (
            (("float32", np.float32), 2e-4),
            (("float64", np.float32), 2e-6),
            (("float64", np.float64), 2e-12),
        ):
            solver = Solver(precision, dtype=dtype)
            scalar = np.dtype(solver.dtype).type
            edge = scalar(64)
            gaps = [np.nextafter(edge, scalar(0)), edge, np.nextafter(edge, scalar(np.inf))]
            for gap in gaps:
                z = np.array([0, gap, 0, 0.25, 0.5], dtype=solver.dtype)
                for c in (0.1, 1.0, 10.0):
                    block = solver.coordinate_update(
                        [z], [0], C=c, objective="topk_cross_entropy", k=2
                    )[0]
                    expected = capped_softmax_dual(z, block, c, 2)
                    np.testing.assert_allclose(
                        block, expected, rtol=tolerance, atol=tolerance * np.abs(expected).max()
                    )

    def test_topk_entropy_free_scores_far_below_a_capped_competitor(self):
        # Scores relative to the maximum lost the free ones' differences beside a far
        # larger capped competitor (from a gap of about 1e4 on); relative to the first
        # free score they keep them. The label far below, two far competitors, two
        # tied ones, and the label far above.
        # float32 solvers compute in float32: tiny tails carry |odds| eps relative errors.
        for precision, rtol in (("float64", 1e-12), ("float32", 2e-5)):
            solver = Solver(precision)
            for gap in [50.0, 1e4, 1e8, 1e16, 1e30]:
                rows = np.array(
                    [
                        [-gap, 1.0, gap, -0.3, 0.2, -1.5],
                        [0.0, 1.0, gap, -0.3, 0.2, gap - 3.0],
                        [0.0, gap, gap, 1.0, -0.3, 0.2],
                        [gap, -gap, 2.0, 2.0, 0.5, 1.0],
                    ],
                    dtype=precision,
                )
                for k in [2, 3, 4]:
                    values, gradients = solver.loss_and_grad(
                        rows, [0] * 4, "topk_cross_entropy", k=k
                    )
                    for i, row in enumerate(rows):
                        value, gradient = topk_entropy_reference(row, 0, k)
                        message = f"{precision} gap={gap} k={k} row {i}"
                        np.testing.assert_allclose(values[i], value, rtol=rtol, err_msg=message)
                        np.testing.assert_allclose(
                            gradients[i],
                            gradient,
                            rtol=rtol,
                            atol=rtol * np.abs(gradient).max(),
                            err_msg=message,
                        )

    def test_topk_offset_coordinate_against_constrained_reference(self):
        # Once the target is this far below, mass=1 to double precision.
        # Solve the remaining strictly convex capped-simplex problem without
        # using the library's normalization root or loss implementation.
        base = np.array([0.0, 1.0, 2.0, -1.0])
        for k in [1, 2, 3, 4]:
            for c in [0.01, 1.0, 10.0]:

                def fun(x: np.ndarray):
                    return (
                        np.sum(x * np.log(x) - base * x + 0.5 * c * x * x),
                        np.log(x) + 1 - base + c * x,
                    )

                ref = minimize(
                    fun,
                    np.full(4, 0.25),
                    jac=True,
                    method="SLSQP",
                    bounds=[(1e-15, 1 / k)] * 4,
                    constraints=[
                        dict(type="eq", fun=lambda x: x.sum() - 1, jac=lambda x: np.ones(4))
                    ],
                    options=dict(ftol=1e-14, maxiter=1000),
                )
                assert ref.success, ref.message
                for gap in [700.0, 1e16, 1e30]:
                    z = np.r_[-gap, base]
                    for warm in [False, True]:
                        # A feasible prior block, with represented score changes.
                        old = c * np.array([0.4, -0.1, -0.1, -0.1, -0.1]) if warm else np.zeros(5)
                        scores = z + old
                        got = Solver().coordinate_update(
                            [scores], [0], C=c, objective="topk_cross_entropy", k=k, previous=[old]
                        )[0]
                        np.testing.assert_allclose(
                            got, c * np.r_[1.0, -ref.x], rtol=2e-6, atol=c * 2e-8
                        )
                        assert got[0] <= c * (1 + 2e-14)
                        assert max(-got[1:]) <= got[0] / k + c * 2e-14

    def test_topk_loss_gradient_extreme_offset_and_permutation(self):
        solver = Solver()
        for k in [1, 2, 3, 4]:
            # The unregularized capped-simplex maximizer is an independent
            # active-set softmax. Its proportions must survive the target gap.
            v = np.array([0.0, 1.0, 2.0, -1.0])
            order = np.argsort(-v)
            u, free = 0, np.full(len(v), np.nan)  # rebound by the loop: k >= 1
            for u in range(k):
                weights = np.exp(v[order[u:]] - v[order[u]])
                free = weights / weights.sum() * (1 - u / k)
                if free.max() <= 1 / k + 1e-15:
                    break
            p = np.empty(4)
            p[order[:u]] = 1 / k
            p[order[u:]] = free
            for gap in [700.0, 1e16, 1e30]:
                scores = np.r_[-gap, v][None]
                val, g = solver.loss_and_grad(scores, [0], loss="topk_cross_entropy", k=k)
                np.testing.assert_allclose(g[0], np.r_[-1.0, p], rtol=3e-14, atol=3e-15)
                assert abs(val[0] / gap - 1.0) <= 0.01
                permutation = np.array([3, 0, 4, 1, 2])
                _, permuted = solver.loss_and_grad(
                    scores[:, permutation], [1], loss="topk_cross_entropy", k=k
                )
                np.testing.assert_allclose(permuted, g[:, permutation], rtol=3e-14, atol=3e-15)

    def test_topk_rescaled_underflowed_tail(self):
        with mp.workdps(100):
            for k in [1, 2, 3, 4]:
                v = [mp.mpf(x) for x in [0, 1, 2, -1]]
                order = sorted(range(4), key=lambda j: -v[j])
                u, free, lse = 0, order, mp.mpf("nan")  # rebound by the loop: k >= 1
                for u in range(k):
                    free = order[u:]
                    lse = mp.log(sum(mp.exp(v[j]) for j in free))
                    if v[order[u]] <= lse - mp.log(k - u):
                        break
                odds = (
                    sum(v[j] for j in order[:u]) + (k - u) * (lse - mp.log(k - u))
                ) / k + mp.log(k)
                for margin in [710.0, 750.0, 1000.0]:
                    c = mp.mpf(1e200)
                    mass = c * mp.exp(odds - margin)
                    expected = [float(mass)] + [0.0] * 4
                    for j in order[:u]:
                        expected[j + 1] = -float(mass / k)
                    for j in free:
                        expected[j + 1] = -float(mass * (1 - mp.mpf(u) / k) * mp.exp(v[j] - lse))
                    got = Solver().coordinate_update(
                        [[margin, 0.0, 1.0, 2.0, -1.0]],
                        [0],
                        C=float(c),
                        objective="topk_cross_entropy",
                        k=k,
                    )[0]
                    np.testing.assert_allclose(got, expected, rtol=4e-13, atol=0)

    def test_scaled_topk_tails_against_high_precision_limit(self):
        # C*p < 1e-100, so the quadratic correction is much smaller than
        # the requested tolerance. Compute the capped-softmax limit in 100
        # digits; never subtract rounded probabilities from one.
        with mp.workdps(100):
            r = mp.log(2) + (2 + mp.log1p(mp.exp(1))) / 2
            for margin in [710.0, 740.0, 750.0]:
                c = mp.mpf(1e200)
                mass = c * mp.exp(-margin + r)
                free = mass / (2 * (1 + mp.exp(1)))
                expected = np.array(
                    [float(mass), -float(free), -float(free * mp.exp(1)), -float(mass / 2)]
                )
                try:
                    actual = Solver("float64").coordinate_update(
                        [[margin, 0.0, 1.0, 2.0]],
                        [0],
                        C=float(c),
                        objective="topk_cross_entropy",
                        k=2,
                    )[0]
                except RuntimeError:
                    # A subnormal total mass needs the exponent range of the wider retry.
                    assert not WIDE_LONG_DOUBLE, margin
                    continue
                np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=0)

    def test_topk_cap_ties_against_independent_log_odds(self):
        # k equal dominant competitors and an exponentially negligible tail.
        # The total mass solves a binary log-odds equation independently of
        # the production Lambert inversion and active-set search.
        with mp.workdps(90):
            for k, margin in [(2, 20.0), (3, 10.0), (4, 0.0)]:
                for c in [1.0, 100.0, 1e10, 1e18]:
                    C = mp.mpf(c)
                    lo = mp.mpf(-1000)
                    hi = mp.mpf(1000)
                    for _ in range(350):
                        t = (lo + hi) / 2
                        mass = 1 / (1 + mp.exp(-t))
                        if t + C * (1 + mp.mpf(1) / k) * mass < margin + mp.log(k):
                            lo = t
                        else:
                            hi = t
                    expected = float(C / (1 + mp.exp(-(lo + hi) / 2)))
                    for precision, dtype in [
                        ("float64", np.float64),
                        ("float64", np.float32),
                        ("float32", np.float32),
                    ]:
                        z = np.array([[0.0] + [margin] * k + [-100.0]])
                        a = Solver(precision, dtype=dtype).coordinate_update(
                            z, [0], C=c, objective="topk_cross_entropy", k=k
                        )[0]
                        # Float32 blocks retry in double, and then in long double.
                        tolerance = (
                            2e-12 if dtype == np.float64 else 2e-6 if WIDE_LONG_DOUBLE else 3e-6
                        )
                        np.testing.assert_allclose(a[0], expected, rtol=tolerance)
                        np.testing.assert_allclose(a[1 : k + 1], -expected / k, rtol=tolerance)
                        assert max(-a[1:]) <= a[0] / k + expected * tolerance

    def test_extreme_topk_never_returns_infeasible_cycle_exit(self):
        z = [[-1e16, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0]]
        # The centered root solves this large-alpha, large-offset block directly.
        a = Solver().coordinate_update(z, [0], C=1e18, objective="topk_cross_entropy", k=5)[0]
        assert max(-a[1:]) <= a[0] / 5 * (1 + 2e-13)
        assert a[0] <= 1e18 * (1 + 2e-13)
        np.testing.assert_allclose(a, [8.75e15] + [-1.25e15] * 7, rtol=2e-13)

    def test_clipped_topk_free_mass_is_not_rescaled_twice(self):
        a = Solver().coordinate_update(
            [[0.0, 1e12, 0.0, -1e12]], [0], objective="topk_cross_entropy", k=2
        )[0]
        np.testing.assert_allclose(a, [1.0, -0.5, -0.5, 0.0], rtol=1e-14, atol=0)

    def test_topk_large_alpha_and_far_caps_across_precisions(self):
        z = [[-1e16, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0]]
        for (precision, dtype), tol in [
            (("float64", np.float64), 1e-13),
            (("float64", np.float32), 1e-5),
            (("float32", np.float32), 1e-5),
        ]:
            a = Solver(precision, dtype=dtype).coordinate_update(
                z, [0], C=1e18, objective="topk_cross_entropy", k=5
            )[0]
            np.testing.assert_allclose(a[0], 8.750000000000006e15, rtol=tol)
            np.testing.assert_allclose(a[1:], -1.25e15, rtol=tol)
        z_caps = [[0.0, 1e15, 2e15, 3e15, 4e15, -1e15]]
        a_caps = Solver("float64").coordinate_update(
            z_caps, [0], C=1e-3, objective="topk_cross_entropy", k=4
        )[0]
        np.testing.assert_allclose(
            a_caps, [1e-3, -2.5e-4, -2.5e-4, -2.5e-4, -2.5e-4, 0.0], rtol=1e-14, atol=1e-17
        )

    def test_topk_balanced_offset_against_scalar_reference(self):
        # Tied competitors have equal mass for every k. The scalar equation
        # is independent of the production Lambert roots and active sets.
        with mp.workdps(90):
            for n in [3, 7]:
                for margin, c in [(1e16, 1e18), (1e8, 1e10), (10.0, 100.0), (-50.0, 1.0)]:
                    for precision, dtype in [
                        ("float64", np.float64),
                        ("float64", np.float32),
                        ("float32", np.float32),
                    ]:
                        z = np.r_[-margin, np.zeros(n)]
                        if dtype != np.float64:
                            z = z.astype(np.float32).astype(float)
                        m = mp.mpf(float(-z[0]))
                        C = mp.mpf(c)
                        if precision == "float32":
                            C = mp.mpf(float(np.float32(c)))
                        lo, hi = mp.mpf(-1000), mp.mpf(1000)
                        for _ in range(400):
                            r = (lo + hi) / 2
                            s = 1 / (1 + mp.exp(-r))
                            if r + C * (1 + mp.mpf(1) / n) * s < m + mp.log(n):
                                lo = r
                            else:
                                hi = r
                        mass = C / (1 + mp.exp(-(lo + hi) / 2))
                        expected = np.array([float(mass)] + [-float(mass / n)] * n)
                        for k in [1, 2, n]:
                            a = Solver(precision, dtype=dtype).coordinate_update(
                                [z], [0], C=c, objective="topk_cross_entropy", k=k
                            )[0]
                            np.testing.assert_allclose(
                                a, expected, rtol=3e-13 if dtype == np.float64 else 2e-6, atol=0
                            )

    def test_topk_separated_threshold_and_permutations(self):
        z = np.array([-1e16, 0, -1e15, -2e15, -3e15, -4e15, -5e15, -6e15])
        expected = np.array([0.001] + [-0.001 / 3] * 3 + [0.0] * 4)
        rng = np.random.default_rng(72)
        for _ in range(10):
            perm = rng.permutation(len(z))
            label = int(np.flatnonzero(perm == 0)[0])
            for precision, dtype in [
                ("float64", np.float64),
                ("float64", np.float32),
                ("float32", np.float32),
            ]:
                a = Solver(precision, dtype=dtype).coordinate_update(
                    [z[perm]], [label], C=0.001, objective="topk_cross_entropy", k=3
                )[0]
                np.testing.assert_allclose(
                    a, expected[perm], rtol=3e-13 if dtype == np.float64 else 2e-6, atol=0
                )

    def test_all_capped_offset_and_warm_start_reference(self):
        for precision, dtype in [
            ("float32", np.float32),
            ("float64", np.float32),
            ("float64", np.float64),
        ]:
            solver = Solver(precision, dtype=dtype)
            for target, warm in [(-1.0, False), (0.0, True)]:
                for huge in [1e16, 1e30]:
                    z = np.array([target, huge, 1.0, -huge], dtype=dtype)
                    old = np.array([3.0, -1.0, -1.0, -1.0] if warm else [0.0] * 4, dtype=dtype)
                    expected = all_capped_reference(z, old, 10.0)
                    actual = solver.coordinate_update(
                        z[None],
                        [0],
                        previous=old[None],
                        C=10.0,
                        objective="topk_cross_entropy",
                        k=3,
                    )[0]
                    np.testing.assert_allclose(
                        actual, expected, rtol=3e-7 if dtype == np.float32 else 3e-13, atol=0
                    )
                    value, gradient = solver.loss_and_grad(z[None], [0], "topk_cross_entropy", k=3)
                    mean = 1 / 3 - target
                    mass = 1 / (1 + np.exp(-mean) / 3)
                    np.testing.assert_allclose(
                        value,
                        [np.logaddexp(0, mean + np.log(3))],
                        rtol=3e-7 if dtype == np.float32 else 3e-14,
                    )
                    np.testing.assert_allclose(
                        gradient,
                        [[-mass, mass / 3, mass / 3, mass / 3]],
                        rtol=3e-7 if dtype == np.float32 else 3e-14,
                    )

    def test_all_capped_root_scales_and_scratch_sizes(self):
        solver = Solver("float64")
        for k in [2, 3, 64, 65, 100]:
            for target, c in [
                (0.0, 1.0),
                (-1e16, 1e18),
                (750.0, 1e200),
                (0.0, 1.5e308),
                (-1.0, 1.5e308),
                (1.0, 1.5e308),
            ]:
                z = np.r_[target, np.linspace(-1.0, 1.0, k)]
                old = np.zeros_like(z)
                if not WIDE_LONG_DOUBLE and c * (1 + 1 / k) > np.finfo(np.float64).max:
                    # The coefficient overflows double, the range of the retry.
                    with pytest.raises(RuntimeError, match="working range"):
                        solver.coordinate_update(
                            z[None], [0], C=c, objective="topk_cross_entropy", k=k
                        )
                    continue
                expected = all_capped_reference(z, old, c)
                actual = solver.coordinate_update(
                    z[None], [0], C=c, objective="topk_cross_entropy", k=k
                )[0]
                np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=0)

    def test_huge_c_blocks_satisfy_the_capped_softmax_fixed_point(self):
        # At C = 1e200 the optimum of a cold block is C times the capped
        # softmax at the updated scores (warm blocks make this ill-posed).
        rng = np.random.default_rng(200)
        solver, c = Solver("float64"), 1e200
        for _ in range(40):
            n = int(rng.integers(3, 71))
            k = n - 1 if rng.integers(3) == 0 else int(rng.integers(2, n))
            z = rng.choice([0.0, 1.0, 1e16]) + rng.choice([10.0, 100.0]) * rng.uniform(-1, 1, n)
            try:
                a = solver.coordinate_update([z], [0], C=c, objective="topk_cross_entropy", k=k)[0]
            except RuntimeError:
                # With the range of double the retry may fail loudly above C = 1e20.
                assert not WIDE_LONG_DOUBLE, (n, k)
                continue
            expected = capped_softmax_dual(z, a, c, k)
            np.testing.assert_allclose(a, expected, rtol=1e-9, atol=1e-9 * c)
            assert max(-a[1:]) <= a[0] / k * (1 + 1e-12)
            assert abs(a.sum()) <= 1e-14 * c

    def test_topk_coordinate_free_scores_far_below_a_capped_competitor(self):
        # The step centred the scores at the largest: free scores far below a capped
        # competitor lost their differences (dual drift 0.17 at a gap of 1e16). The
        # optimal block is C [mass, -p] of the capped softmax at the updated scores,
        # checked here in 90 digits, with its constraints.
        solver = Solver("float64")
        for gap in [50.0, 1e4, 1e8, 1e16, 1e30]:
            rows = [
                [-gap, 1.0, gap, -0.3, 0.2, -1.5],
                [0.0, 1.0, gap, -0.3, 0.2, gap - 3.0],
                [0.0, gap, gap, 1.0, -0.3, 0.2],
                [gap, -gap, 2.0, 2.0, 0.5, 1.0],
            ]
            for k in [2, 3, 4]:
                for c in [1e-3, 1.0, 1e3]:
                    for warm in [False, True]:
                        prior = c * np.array([0.4, -0.1, -0.1, -0.1, -0.1, 0.0]) * warm
                        for i, z in enumerate(rows):
                            a = solver.coordinate_update(
                                [z], [0], C=c, objective="topk_cross_entropy", k=k, previous=[prior]
                            )[0]
                            # With norm2 = 1 the updated scores are z + (a - prior).
                            expected = capped_softmax_dual(np.array(z) - prior, a, c, k)
                            message = f"gap={gap} k={k} C={c} warm={warm} row {i}"
                            np.testing.assert_allclose(
                                a, expected, rtol=1e-12, atol=1e-13 * c, err_msg=message
                            )
                            assert a[0] <= c * (1 + 1e-14), message
                            assert max(-a[1:]) <= a[0] / k + c * 1e-14, message
                            assert abs(a.sum()) <= 1e-14 * c, message

    def test_saturation_boundary_reference(self):
        solver = Solver("float64")
        for k in [2, 3, 10]:
            for c in [1e3, 1e10, 1e14, 1e16, 1e18, 1e30, 1e300]:
                coefficient = c * (1 + 1 / k)
                for factor in [0.49, 0.51, 0.99, 1.0, 1.01, 2.0]:
                    z = np.r_[0.0, np.full(k, coefficient * factor)]
                    expected = all_capped_reference(z, np.zeros_like(z), c)
                    actual = solver.coordinate_update(
                        z[None], [0], C=c, objective="topk_cross_entropy", k=k
                    )[0]
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=5e-13,
                        atol=0,
                        err_msg=f"k={k}, C={c}, factor={factor}",
                    )

    def test_uniform_and_nonuniform_warm_starts(self):
        for (precision, dtype), rtol in [
            (("float32", np.float32), 2e-6),
            (("float64", np.float32), 3e-7),
            (("float64", np.float64), 3e-13),
        ]:
            solver = Solver(precision, dtype=dtype)
            for k in [3, 64, 65, 100]:
                z = np.zeros(k + 1, dtype=dtype)
                z[1:4] = [1e30, 1.0, -1e30]
                for changed in [None, 1, k - 1]:
                    old = np.r_[k, np.full(k, -1.0)].astype(dtype)
                    if changed is not None:
                        old[changed] += 0.5
                        old[changed + 1] -= 0.5
                    expected = all_capped_reference(z, old, 100.0)
                    actual = solver.coordinate_update(
                        z[None],
                        [0],
                        C=100.0,
                        previous=old[None],
                        objective="topk_cross_entropy",
                        k=k,
                    )[0]
                    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0)

    def test_both_smoothed_hinges_with_rounded_caps(self):
        for precision, dtype in [
            ("float32", np.float32),
            ("float64", np.float32),
            ("float64", np.float64),
        ]:
            solver = Solver(precision, dtype=dtype)
            for objective in ["topk_hinge_alpha", "topk_hinge_beta"]:
                for gamma in [1.0, 1e-10, 1e-20, 1e-30]:
                    for k in [1, 2, 3]:
                        values, gradient = solver.loss_and_grad(
                            np.zeros((1, 4)), [0], objective, k=k, gamma=gamma
                        )
                        tolerance = 3e-15 if dtype == np.float64 else 3e-7
                        np.testing.assert_allclose(values, [1 - gamma / 6], rtol=tolerance)
                        np.testing.assert_allclose(
                            gradient, [[-1, 1 / 3, 1 / 3, 1 / 3]], rtol=tolerance, atol=0
                        )
