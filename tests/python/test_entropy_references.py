"""Entropy coordinate steps against high-precision references."""

import mpmath as mp
import numpy as np
from libsdca._core import Solver
from mpmath_types import MPReal


class EntropyReferenceTests:
    def test_proximal_entropy_stops_at_resolved_threshold(self):
        # At alpha=1e300 a midpoint after an unrepresentable Newton step
        # increased this value's error from 39 to 90 double epsilons.
        scores = np.array(
            [
                -1.0483214263769076,
                0.3031678531969774,
                -0.13338986377238932,
                0.9581706912908616,
                0.3107827694553286,
                -1.262146206335577,
                -0.1722539931797783,
                -0.15317129642564592,
                -1.3094169073915682,
                -0.529477277357514,
                0.6259885924119122,
                -0.5161927919523119,
                0.04534127159789437,
                0.046934615112414146,
                0.7258639585277716,
                -0.7062915631562838,
                -1.7685844590153283,
                2.151434129044531,
                3.7807037145115574,
                0.8392764859773002,
                -0.4541729599157674,
                0.33227668336775124,
                -0.7623524735344774,
                0.7488920059082762,
                0.6573538699331856,
                -0.5172695862092952,
                0.01587688079631811,
                0.37529577501921696,
                -0.31917565161838946,
                0.21897102779001196,
                -1.1733768903209607,
                0.16727961803818514,
                0.3555842243691813,
                1.218202830728856,
                -0.7343055166886147,
                0.3137861438274786,
                1.493779663187635,
                -1.1665108557919397,
                -1.189786419318855,
                -1.3088668077993248,
                0.09603085245305117,
                -0.00015944680426722548,
                -0.7901806053475467,
                1.763374150084775,
                -1.5315613807251165,
                1.7511099051393872,
                0.9686640399836024,
                -0.4114861242093704,
                1.0818045243644152,
                1.9433235771852195,
                0.39237177471511636,
                -0.6025105941520223,
                -1.780560222345586,
                -0.3468332566420356,
                -0.2865316488337876,
                0.08399127629935922,
                -1.3254745543883975,
                0.41430794504824353,
                -0.3168656294839644,
                0.47347244651776615,
                1.4889481873544237,
                1.8821955442427563,
                0.39495148240825606,
                -0.48601385056578933,
                0.08919460657943189,
                -0.8922015668623301,
                -0.5652034141630178,
                0.4040840733841602,
                1.491379127701085,
                0.6798016853219903,
                0.15610454178133837,
                -0.33006451211576543,
                0.5500382858845679,
                0.6119575644506017,
                -2.382696526150933,
                -1.326529413169601,
                0.23742730579904545,
                1.2625637635072735,
                -1.0740747181321355,
                -2.151622260134446,
                -2.213646063723499,
                0.10486756023574018,
                -0.035243713493456806,
                -0.06783567685124253,
                1.0176933465558267,
                -1.2275813398734399,
                -1.273162826068457,
                -0.8560058224931847,
                0.5911537727946493,
                -0.5597643362950171,
                -2.1069186454657314,
                0.26085619920626407,
                -0.25309943396986356,
                -1.932160633224408,
                -0.8884348174683642,
                0.025609640443888283,
                0.17544495028507526,
                -2.0820323227191273,
                -0.17667802477994124,
                -0.29993811767801204,
            ]
        )
        target, alpha = 74, 1e300
        with mp.workdps(90):
            a = mp.mpf(alpha)
            competitors = [
                mp.mpf(float(z)) - mp.mpf(float(scores[target]))
                for j, z in enumerate(scores)
                if j != target
            ]

            def probabilities(t: MPReal):
                return [mp.lambertw(a * mp.exp(z - t)).real / a for z in competitors]

            def residual(t: MPReal):
                mass = mp.fsum(probabilities(t))
                return a * mass - mp.log1p(-mass) - t

            t = mp.findroot(residual, (mp.log(a) - 5, mp.log(a) + 5))
            other = probabilities(t)
            mass = mp.fsum(other)
            expected = -mp.log1p(-mass) + a * (mass * mass + mp.fsum(g * g for g in other)) / 2
            reference = other[:target] + [-mass] + other[target:]
            value, gradient = Solver("float64").loss_and_grad(
                scores[None], [target], "proximal_cross_entropy", alpha=alpha
            )
            eps = np.finfo(float).eps
            assert abs(mp.mpf(float(value[0])) - expected) / expected < 64 * eps
            error = max(
                abs(mp.mpf(float(g)) - r) for g, r in zip(gradient[0], reference, strict=True)
            )
            assert error / max(map(abs, reference)) < 64 * eps

    def test_entropy_extremes_against_high_precision_normalization(self):
        # Independent 90-digit probability normalization, including a clipped
        # plateau and a margin that centering at class 0 would lose.
        with mp.workdps(90):
            for scores, positives in [
                ([0.0, 60.0, -40.0], 2),
                ([-1e16, 0.0, 1.0], 1),
                ([20.0, 20.0, -20.0], 2),
            ]:
                for c in [1e-6, 1.0, 1e6]:
                    q, alpha = mp.mpf(1) / positives, mp.mpf(c)
                    z = list(map(mp.mpf, scores))
                    shift = max(z)
                    z = [v - shift for v in z]

                    def probabilities(t: MPReal):
                        return [
                            mp.lambertw(
                                mp.exp(
                                    mp.log(alpha * q) + v - t + (alpha * q if j < positives else 0)
                                )
                            )
                            / alpha
                            for j, v in enumerate(z)
                        ]

                    lo, hi = -alpha + mp.log(q), alpha * q + mp.log(q * len(z))
                    for _ in range(340):
                        mid = (lo + hi) / 2
                        if sum(probabilities(mid)) > 1:
                            lo = mid
                        else:
                            hi = mid
                    probs = probabilities((lo + hi) / 2)
                    expected = np.array(
                        [
                            float(alpha * ((q if j < positives else 0) - prob))
                            for j, prob in enumerate(probs)
                        ]
                    )
                    target = np.arange(len(z))[None] < positives
                    actual = Solver().coordinate_update([scores], target, C=c)[0]
                    np.testing.assert_allclose(
                        actual, expected, rtol=1.5e-14, err_msg=f"{scores} p={positives} C={c}"
                    )

    def test_rescaled_underflowed_entropy_tail(self):
        # exp(-margin) is subnormal or zero, but C*exp(-margin) is normal.
        # In these cases the alpha*r term is negligible; 100-digit exp is an
        # independent reference, also exercising the all-underflowed solve.
        with mp.workdps(100):
            for margin in [710.0, 745.0, 750.0, 1000.0]:
                for c in [1e100, 1e200]:
                    expected = float(mp.mpf(c) * mp.exp(-margin))
                    got = Solver().coordinate_update([[margin, 0.0]], [0], C=c)[0]
                    np.testing.assert_allclose(got, [expected, -expected], rtol=3e-13, atol=0)
