"""Centered feature objectives against references that retain exact represented inputs."""

import mpmath as mp
import numpy as np
import pytest
from libsdca import Solver


class CenteredCertificateTests:
    def test_common_center_does_not_certify_initial_unoptimized_weights(
        self, subtests: pytest.Subtests
    ):
        for precision, dtype in (
            ("float32", np.float32),
            ("float64", np.float32),
            ("float64", np.float64),
        ):
            with subtests.test(precision=precision, dtype=dtype):
                model = Solver(precision, dtype=dtype).fit(
                    [[1.0]],
                    [0],
                    num_classes=3,
                    C=1.0,
                    center=[[1e8] * 3],
                    eval_on_start=True,
                    max_epoch=0,
                    tol=1e-10,
                )
                assert model.status == "max_epoch"
                assert round(abs(model.history["primal"][-1] - np.log(3)), 14) == 0
                assert model.history["dual"][-1] == 0.0
                assert model.history["relative_gap"][-1] == 1.0

    def test_common_center_convergence_respects_weight_resolution(self, subtests: pytest.Subtests):
        with mp.workdps(80):
            a = mp.findroot(lambda a: a - 2 / (mp.exp(mp.mpf("1.5") * a) + 2), (0.3, 0.7))
            optimum = mp.mpf(".75") * a * a + mp.log1p(2 * mp.exp(-mp.mpf("1.5") * a))
            for precision, dtype in (
                ("float32", np.float32),
                ("float64", np.float32),
                ("float64", np.float64),
            ):
                for offset in (1e8, 1e16):
                    with subtests.test(precision=precision, dtype=dtype, offset=offset):
                        model = Solver(precision, dtype=dtype).fit(
                            [[1.0]],
                            [0],
                            num_classes=3,
                            C=1.0,
                            center=[[offset] * 3],
                            eval_on_start=True,
                            eval_epoch=1,
                            max_epoch=20,
                            tol=1e-10,
                        )
                        assert model.weights is not None
                        center = mp.mpf(float(np.array(offset, dtype=model.weights.dtype)))
                        w = [mp.mpf(float(x)) for x in model.weights[0]]
                        primal = sum((x - center) ** 2 for x in w) / 2 + mp.log(
                            sum(mp.exp(x - w[0]) for x in w)
                        )
                        assert round(abs(float(primal) - model.history["primal"][-1]), 13) == 0
                        if dtype == np.float64 and offset == 1e8:
                            assert model.status == "solved"
                            assert model.epochs >= 1
                            assert float((primal - optimum) / primal) < 1e-10
                        else:
                            assert model.status != "solved"
                            assert float((primal - optimum) / primal) > 1e-3

    def test_nonbinary_dual_balance_at_huge_common_center(self, subtests: pytest.Subtests):
        # The rounded target entry is not an exact-real sum of the other
        # represented entries. The certificate refers to a nearby EXACTLY
        # balanced feasible block, not that slightly unbalanced stored vector.
        with mp.workdps(100):
            negatives = np.array([0.07, 0.11, 0.03, 0.13, 0.09, 0.17, 0.019, 0.021])
            for precision, dtype in (
                ("float32", np.float32),
                ("float64", np.float32),
                ("float64", np.float64),
            ):
                neg = negatives.astype(dtype)
                dual = np.r_[neg.sum(dtype=np.float64), -neg].astype(dtype)[None, :]
                exact = [mp.mpf(float(x)) for x in neg]
                target = sum(exact)
                assert mp.mpf(float(dual[0, 0])) != target
                probabilities = [1 - target] + exact
                reference_dual = -sum(p * mp.log(p) for p in probabilities)
                reference_dual -= (target**2 + sum(p * p for p in exact)) / 2
                for offset in (1e8, 1e16):
                    with subtests.test(precision=precision, dtype=dtype, offset=offset):
                        model = Solver(precision, dtype=dtype).fit(
                            [[1.0]],
                            [0],
                            num_classes=9,
                            C=1.0,
                            center=[[offset] * 9],
                            initial_dual=dual,
                            eval_on_start=True,
                            max_epoch=0,
                            tol=1e-10,
                        )
                        assert model.status != "solved"
                        assert (
                            round(abs(model.history["dual"][-1] - float(reference_dual)), 13) == 0
                        )

    def test_entropy_boundary_and_nondyadic_label_distribution_duals(
        self, subtests: pytest.Subtests
    ):
        # Normalize represented probabilities in exact arithmetic to get a
        # feasible reference. Near-zero probabilities and non-dyadic 1/p label
        # masses exercise conjugate/repair rounding without a common-mode term.
        with mp.workdps(100):
            for precision, dtype in (
                ("float32", np.float32),
                ("float64", np.float32),
                ("float64", np.float64),
            ):
                c = 0.7
                exact_c = mp.mpf(float(dtype(c)) if precision == "float32" else c)
                for classes in (7, 23, 97):
                    for positive in (1, 3):
                        q = np.zeros(classes)
                        q[:positive] = 1 / positive
                        p = np.geomspace(1e-20, 1.0, classes)
                        p /= p.sum()
                        p[0] = 0.0  # a target probability on the entropy boundary
                        p /= p.sum()
                        initial = (float(exact_c) * (q - p)).astype(dtype)[None, :]
                        target = [0] if positive == 1 else np.array([q > 0])
                        objective = "cross_entropy" if positive == 1 else "multilabel_cross_entropy"
                        center = (1e16 + 32 * np.arange(classes)).astype(dtype)[None, :]
                        with subtests.test(
                            precision=precision, dtype=dtype, classes=classes, positive=positive
                        ):
                            model = Solver(precision, dtype=dtype).fit(
                                [[1.0]],
                                target,
                                objective,
                                num_classes=classes,
                                C=c,
                                center=center,
                                initial_dual=initial,
                                eval_on_start=True,
                                max_epoch=0,
                                tol=1e-10,
                            )
                            a = [mp.mpf(float(x)) for x in model.dual[0]]
                            qq = [
                                mp.mpf(1) / positive if j < positive else mp.mpf(0)
                                for j in range(classes)
                            ]
                            # Apply the public certificate's feasibility
                            # projection in exact arithmetic. Normalizing raw
                            # stored probabilities is a DIFFERENT projection:
                            # their float-storage imbalance may be 1e-7.
                            if positive == 1:
                                negatives = [min(max(-ai, 0), exact_c) for ai in a[1:]]
                                mass = sum(negatives)
                                if mass > exact_c:
                                    negatives = [bi * exact_c / mass for bi in negatives]
                                repaired = [sum(negatives)] + [-bi for bi in negatives]
                            else:
                                caps = [
                                    exact_c / positive if j < positive else mp.mpf(0)
                                    for j in range(classes)
                                ]
                                repaired = [min(ai, cap) for ai, cap in zip(a, caps, strict=True)]
                                mass = sum(repaired)
                                if mass > 0:
                                    repaired[0] -= mass
                                else:
                                    deficit = -mass
                                    for j in range(classes):
                                        room = min(caps[j] - repaired[j], deficit)
                                        repaired[j] += room
                                        deficit -= room
                            probabilities = [
                                max(mp.mpf(0), qi - ai / exact_c)
                                for qi, ai in zip(qq, repaired, strict=True)
                            ]
                            mass = sum(probabilities)
                            probabilities = [pj / mass for pj in probabilities]
                            exact_a = [
                                exact_c * (qi - pj)
                                for qi, pj in zip(qq, probabilities, strict=False)
                            ]
                            g0 = [mp.mpf(float(x)) - mp.mpf(float(center[0, 0])) for x in center[0]]
                            reference = -exact_c * sum(
                                pj * mp.log(pj) for pj in probabilities if pj
                            )
                            reference -= sum(ai * ai for ai in exact_a) / 2
                            reference -= sum(ai * zi for ai, zi in zip(exact_a, g0, strict=False))
                            scale = max(
                                1.0,
                                float(
                                    sum(abs(ai * zi) for ai, zi in zip(exact_a, g0, strict=False))
                                ),
                                abs(float(reference)),
                            )
                            # The reference starts from represented stored
                            # duals; evaluation/repair still uses double.
                            tolerance = 2e-12
                            assert (
                                abs(model.history["dual"][-1] - float(reference))
                                <= tolerance * scale
                            )
