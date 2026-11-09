"""Independent regressions for the measured accuracy limits in docs/numerics.md."""

import mpmath as mp
import numpy as np
import pytest
from libsdca import loss_and_grad


class AccuracyEnvelopeTests:
    def test_equal_score_multilabel_proximal_gradient(self, subtests: pytest.Subtests):
        positives, negatives = 31, 34
        alpha = 1e-6
        # Equal scores give two probabilities a and b. With u = -g_positive,
        # a = 1/p - u, b = p*u/m, and p*a + m*b = 1. Solve their scalar KKT
        # equation by bisection, independently of the native Lambert-W solver.
        with mp.workdps(80):
            q, step = mp.mpf(1) / positives, mp.mpf(alpha)
            lower, upper = mp.mpf(0), q
            for _ in range(300):
                u = (lower + upper) / 2
                a, b = q - u, positives * u / negatives
                residual = mp.log(a / b) + step * (a - b) - step * q
                if residual > 0:
                    lower = u
                else:
                    upper = u
            u = (lower + upper) / 2
            a, b = q - u, positives * u / negatives
            residual = mp.log(a / b) + step * (a - b) - step * q
            assert abs(residual) < mp.mpf("1e-75")
            reference = np.array([-float(u)] * positives + [float(b)] * negatives)

        logits = np.full((1, positives + negatives), 1000, dtype=np.float32)
        target = np.arange(positives + negatives)[None, :] < positives
        scale = np.finfo(np.float32).eps * np.abs(reference).max()
        for precision, budget in (("float32", 400), ("float64", 1)):
            with subtests.test(precision=precision):
                _, gradient = loss_and_grad(
                    logits, target, "proximal_cross_entropy", alpha=alpha, precision=precision
                )
                assert gradient.dtype == np.float32
                error = np.abs(gradient[0].astype(np.float64) - reference).max() / scale
                assert float(error) <= budget
