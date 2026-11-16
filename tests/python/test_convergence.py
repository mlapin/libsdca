"""Independent binary root references and scratch-buffer boundary checks."""

import mpmath as mp
import numpy as np
import pytest
import torch
from numpy_api import Solver


class ConvergenceTests:
    def test_binary_coordinate_against_high_precision_log_odds(self):
        # If r is the non-target mass, logit(r)+2*alpha*r=-margin.
        # Bisect in log odds, independent of the production Lambert/Newton solve.
        with mp.workdps(90):
            for margin in [-100.0, -10.0, 0.0, 18.0, 40.0, 100.0]:
                for c in [1e-10, 0.001, 1.0, 1e10, 1e18]:
                    lo, hi = mp.mpf(-800), mp.mpf(800)
                    C = mp.mpf(c)
                    for _ in range(330):
                        mid = (lo + hi) / 2
                        r = 1 / (1 + mp.exp(-mid))
                        if mid + 2 * C * r < -margin:
                            lo = mid
                        else:
                            hi = mid
                    expected = float(C / (1 + mp.exp(-(lo + hi) / 2)))
                    for precision, dtype in [
                        ("float32", np.float32),
                        ("float64", np.float32),
                        ("float64", np.float64),
                    ]:
                        solver = Solver(precision, dtype=dtype)
                        a = solver.coordinate_update([[margin, 0.0]], [0], C=c)[0]
                        tolerance = 3e-6 if dtype == np.float32 else 3e-13
                        np.testing.assert_allclose(
                            a,
                            [expected, -expected],
                            rtol=tolerance,
                            atol=float(np.finfo(solver.dtype).smallest_subnormal),
                            err_msg=f"{precision} {dtype} margin={margin} C={c}",
                        )

    def test_multilabel_scratch_boundary_and_permutation(self):
        rng = np.random.default_rng(561)
        for classes in [64, 65, 200]:
            scores = rng.normal(size=(2, classes))
            target = np.zeros_like(scores, dtype=bool)
            target[:, [0, 3, 9]] = True
            permutation = rng.permutation(classes)
            for dtype in [np.float32, np.float64]:
                solver = Solver("float64", dtype=dtype)
                ml = "multilabel_cross_entropy"
                a = solver.coordinate_update(scores, target, ml, C=0.7)
                b = solver.coordinate_update(
                    scores[:, permutation], target[:, permutation], ml, C=0.7
                )
                np.testing.assert_allclose(a[:, permutation], b, rtol=2e-6, atol=2e-8)
                _, gradient = solver.loss_and_grad(scores + a, target, ml)
                np.testing.assert_allclose(a, -0.7 * gradient, rtol=3e-6, atol=2e-8)

    def test_cold_initialization_and_zero_epoch_center(self):
        rng = np.random.default_rng(52)
        x = rng.normal(size=(40, 8))
        y = np.arange(40) % 4
        center = rng.normal(size=(8, 4))
        for dtype in [np.float32, np.float64]:
            solver = Solver("float64", dtype=dtype)
            for c in [None, center]:
                model = solver.fit(x, y, center=c, max_epoch=0)
                np.testing.assert_array_equal(model.dual, 0.0)
                np.testing.assert_array_equal(
                    model.weights, np.zeros((8, 4)) if c is None else c.astype(solver.dtype)
                )

    def test_torch_backend_validation_parity(self, subtests: pytest.Subtests):
        import libsdca.torch as L

        scores = torch.zeros(2, 3)
        targets = torch.tensor([0, 1])
        nan = float("nan")
        for backend in ["native", "torch"]:
            for fn, options, error in [
                (L.topk_cross_entropy, dict(k=0), ValueError),
                (L.topk_cross_entropy, dict(k=3), ValueError),
                (L.topk_cross_entropy, dict(k=1.5), TypeError),
                (L.topk_cross_entropy, dict(k=True), TypeError),
                (L.proximal_cross_entropy, dict(alpha=0), ValueError),
                (L.multiclass_hinge_loss, dict(gamma=nan), ValueError),
            ]:
                with subtests.test(backend=backend, loss=fn.__name__, options=options):
                    with pytest.raises(error):
                        fn(scores, targets, backend=backend, **options)
            with pytest.raises(TypeError):
                # Deliberately not a tensor target.
                L.cross_entropy(scores, [0, 1], backend=backend)  # pyright: ignore[reportArgumentType]
