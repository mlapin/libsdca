"""Torch loss layers and tensor inputs at extreme scores."""

from typing import Any

import libsdca
import numpy as np
import pytest
import torch
from libsdca._core import Solver
from libsdca.torch import DistillationLoss, SDCALoss, distillation_loss, sdca_loss


class TorchReferenceTests:
    def test_top_level_names_and_module_options(self):
        assert libsdca.SDCALoss is SDCALoss
        assert libsdca.sdca_loss is sdca_loss
        assert libsdca.DistillationLoss is DistillationLoss
        criterion = SDCALoss("topk_hinge_beta", k=2, gamma=0.5, reduction="sum")
        z = torch.randn(4, 5, dtype=torch.float64)
        target = torch.tensor([0, 1, 4, 2])
        expected = sdca_loss(z, target, "topk_hinge_beta", k=2, gamma=0.5, reduction="sum")
        torch.testing.assert_close(criterion(z, target), expected)
        assert "k=2, gamma=0.5" in repr(criterion)
        assert "temperature=2.0" in repr(DistillationLoss(temperature=2.0))
        with pytest.raises(TypeError):  # options are keyword-only
            SDCALoss("cross_entropy", "native")  # pyright: ignore[reportCallIssue]

    def test_paper_collapse_and_cancellation_low_precision(self):
        # The tensor path promotes half/bfloat16 work, then stores gradients
        # in the input dtype. Check represented gradients, including bf16.
        for dtype, margin in [
            (torch.bfloat16, 12.0),
            (torch.float16, 12.0),
            (torch.float32, 20.0),
            (torch.float64, 50.0),
        ]:
            z = torch.tensor([[margin, 0.0, 0.0]], dtype=dtype, requires_grad=True)
            value = sdca_loss(z, torch.tensor([0]), backend="torch")
            value.backward()
            expected = 2 * np.exp(-margin) / (1 + 2 * np.exp(-margin))
            assert value.item() > 0
            assert z.grad is not None
            assert z.grad[0, 0].item() < 0
            tolerance = 0.02 if dtype in [torch.bfloat16, torch.float16] else 2e-6
            assert abs(-z.grad[0, 0].item() / expected - 1.0) <= tolerance
            teacher = torch.tensor([[2.0, 1.0, 0.0]], dtype=dtype)
            student = teacher.clone().requires_grad_()
            distillation_loss(student, teacher).backward()
            torch.testing.assert_close(student.grad, torch.zeros_like(student), rtol=0, atol=0)

    def test_all_capped_cancelling_scores_preserve_mean(self):
        for precision, dtype in [
            ("float64", np.float64),
            ("float64", np.float32),
            ("float32", np.float32),
        ]:
            solver = Solver(precision, dtype=dtype)
            baseline = np.array([[0.0, 0.0, 1.0, 0.0]])
            expected = solver.coordinate_update(baseline, [0], objective="topk_cross_entropy", k=3)
            value, gradient = solver.loss_and_grad(baseline, [0], loss="topk_cross_entropy", k=3)
            for scale in [1e16, 1e30]:
                z = np.array([[0.0, scale, 1.0, -scale]])
                got = solver.coordinate_update(z, [0], objective="topk_cross_entropy", k=3)
                actual, g = solver.loss_and_grad(z, [0], loss="topk_cross_entropy", k=3)
                np.testing.assert_allclose(got, expected, rtol=2e-6, atol=0)
                np.testing.assert_allclose(actual, value, rtol=2e-6, atol=0)
                np.testing.assert_allclose(g, gradient, rtol=2e-6, atol=0)
                t = torch.tensor(z, dtype=torch.float64, requires_grad=True)
                sdca_loss(t, torch.tensor([0]), loss="topk_cross_entropy", k=3).backward()
                assert t.grad is not None
                np.testing.assert_allclose(t.grad.numpy(), gradient, rtol=2e-6, atol=0)

    def test_smoothed_hinge_backward_with_tiny_gamma(self):
        for dtype in [torch.float32, torch.float64]:
            z = torch.zeros((1, 4), dtype=dtype, requires_grad=True)
            value = sdca_loss(z, torch.tensor([0]), "topk_hinge_alpha", k=3, gamma=1e-20)
            value.backward()
            torch.testing.assert_close(
                z.grad,
                torch.tensor([[-1.0, 1 / 3, 1 / 3, 1 / 3]], dtype=dtype),
                rtol=4 * torch.finfo(dtype).eps,
                atol=0,
            )

    def test_torch_backward_and_nonsymmetric_active_set(self):
        for dtype in [torch.float32, torch.float64]:
            for gamma in [1e-10, 1e-20, 1e-30]:
                # At k=2 the two leading positive margins share the mass.
                # The inactive example must keep zero loss and gradient.
                z = torch.tensor(
                    [[0.0, 2.0, 1.0, -2.0], [3.0, 0.0, 0.0, 0.0]], dtype=dtype, requires_grad=True
                )
                value = sdca_loss(
                    z, torch.tensor([0, 0]), "topk_hinge_beta", k=2, gamma=gamma, reduction="sum"
                )
                value.backward()
                tolerance = 3e-15 if dtype == torch.float64 else 2e-7
                torch.testing.assert_close(
                    value.detach(),
                    torch.tensor(2.5 - gamma / 4, dtype=dtype),
                    rtol=tolerance,
                    atol=0,
                )
                expected = torch.tensor([[-1.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=dtype)
                torch.testing.assert_close(z.grad, expected, rtol=tolerance, atol=0)

    def test_multilabel_entropy_alias_and_boolean_k_parity(self):
        z = torch.tensor(
            [[0.3, -0.2, 0.7], [0.1, 0.2, -0.4]], dtype=torch.float64, requires_grad=True
        )
        mask = torch.tensor([[True, False, True], [False, True, False]])
        for backend in ["native", "torch"]:
            actual = SDCALoss("multilabel_cross_entropy", backend=backend)(z, mask)
            expected = sdca_loss(z, mask, backend=backend)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            ga = torch.autograd.grad(actual, z, retain_graph=True)[0]
            ge = torch.autograd.grad(expected, z)[0]
            torch.testing.assert_close(ga, ge, rtol=0, atol=0)
        solver = Solver()
        for k in [True, np.bool_(True), False, np.bool_(False)]:
            with pytest.raises(ValueError):
                solver.loss_and_grad(z.detach().numpy(), mask, k=k)
            with pytest.raises(ValueError):
                solver.coordinate_update(z.detach().numpy(), mask, k=k)
            with pytest.raises(ValueError):
                solver.fit(np.eye(2), mask, k=k)

    def test_label_sets_and_tensors(self, subtests: pytest.Subtests):
        solver = Solver()
        x = np.eye(4)
        labels = [[0, 2], [1], [0, 1], [2]]
        mask = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], bool)
        for objective, gamma in [
            ("multilabel_cross_entropy", 0.0),
            ("multilabel_hinge", 0.0),
            ("multilabel_hinge", 1.0),
        ]:
            options: dict[str, Any] = dict(objective=objective, gamma=gamma, max_epoch=4, tol=0)
            ref = solver.fit(x, mask, **options)
            model = solver.fit(torch.tensor(x, requires_grad=True), labels, **options)
            np.testing.assert_array_equal(model.weights, ref.weights)
            np.testing.assert_array_equal(model.predict(torch.tensor(x)), ref.predict(x))
            values = solver.loss_and_grad(x[:, :3], labels, loss=objective)
            expected = solver.loss_and_grad(x[:, :3], mask.tolist(), loss=objective)
            for a, b in zip(values, expected, strict=False):
                np.testing.assert_array_equal(a, b)
            warm = solver.fit(
                x, labels, objective=objective, initial_dual=torch.tensor(model.dual), max_epoch=0
            )
            assert warm.weights is not None
            np.testing.assert_allclose(warm.weights, x.T @ model.dual, atol=1e-15)
        for bad in [
            [[0, 0], [1], [0, 1], [2]],
            [[], [1], [0, 1], [2]],
            [[0.0, 1.0], [1], [0, 1], [2]],
            [[-1], [1], [0, 1], [2]],
            [[0, "x"], [1], [0, 1], [2]],
        ]:
            with subtests.test(bad=bad), pytest.raises(ValueError):
                solver.fit(x, bad)
