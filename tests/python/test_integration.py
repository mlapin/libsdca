"""Feature and kernel fits of every objective, and regressions of shifted scores."""

from typing import Any

import numpy as np
import pytest
from libsdca import LOSSES
from libsdca._core import Solver


class IntegrationTests:
    def test_feature_kernel_and_warm_start_all_objectives(self, subtests: pytest.Subtests):
        rng = np.random.default_rng(402)
        x = rng.normal(size=(18, 4))
        labels = rng.integers(0, 4, len(x))
        mask = np.eye(4, dtype=bool)[labels]
        mask[np.arange(len(x)), (labels + 1) % 4] = True
        solver = Solver()
        for objective in list(LOSSES)[:-1]:
            for multi in (
                [False, True]
                if objective == "cross_entropy"
                else [objective.startswith("multilabel")]
            ):
                y = mask if multi else labels
                opts: dict[str, Any] = dict(
                    C=0.4, objective=objective, k=2, gamma=0.7, tol=1e-8, max_epoch=2000
                )
                with subtests.test(objective=objective, multi=multi):
                    model = solver.fit(x, y, **opts)
                    kernel = solver.fit(x @ x.T, y, kernel=True, **opts)
                    assert model.status == "solved"
                    assert kernel.status == "solved"
                    assert model.weights is not None
                    np.testing.assert_allclose(model.weights, x.T @ model.dual, atol=2e-13)
                    np.testing.assert_allclose(
                        x @ model.weights, kernel.decision_function(x @ x.T), atol=2e-4
                    )
                    value = solver.loss_and_grad(x @ model.weights, y, objective, k=2, gamma=0.7)[0]
                    primal = 0.5 * np.sum(model.weights**2) + 0.4 * value.sum()
                    assert round(abs(primal - model.history["primal"][-1]), 10) == 0
                    warm = solver.fit(x, y, initial_dual=model.dual, **opts)
                    assert warm.weights is not None
                    np.testing.assert_allclose(warm.weights, model.weights, atol=2e-4)
                    assert ("rank_loss" if multi else "accuracy") in model.history

    def test_shifted_loss_and_multilabel_residual_regressions(self):
        for precision, dtype in [
            ("float32", np.float32),
            ("float64", np.float32),
            ("float64", np.float64),
        ]:
            s = Solver(precision, dtype=dtype)
            for objective in list(LOSSES)[:-1]:
                y = [[True, True, False]] if objective.startswith("multilabel") else [0]
                value, grad = s.loss_and_grad([[1e16] * 3], y, objective, k=2, gamma=0.7)
                reference = s.loss_and_grad([[0.0] * 3], y, objective, k=2, gamma=0.7)
                np.testing.assert_allclose(value, reference[0], rtol=1e-6)
                np.testing.assert_allclose(grad, reference[1], rtol=1e-6, atol=1e-7)
            a = s.coordinate_update([[20.0, 20.0, -20.0]], [[True, True, False]])
            # float32 solves in float arithmetic: a few float ulps.
            rtol = 2e-6 if precision == "float32" else 1e-6
            np.testing.assert_allclose(a[0, :2], np.exp(-40) / 4, rtol=rtol, atol=0)
            a = s.coordinate_update([[0.0, 1e-10, -40.0]], [[True, True, False]])
            np.testing.assert_allclose(a[0, :2], [1e-10 / 6, -1e-10 / 6], rtol=2e-7, atol=0)
