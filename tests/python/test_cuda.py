"""CUDA numerical, autograd, layout, validation and training regressions.

These exercise the tensor entropy and distillation APIs; test_tensor_losses.py
and test_tensor_prox.py compare the other tensor losses with the native ones.
SDCA training remains a CPU API.
"""

import math

import pytest
import torch
from libsdca.torch import DistillationLoss, SDCALoss, distillation_loss, sdca_loss
from torch.autograd import gradcheck, gradgradcheck


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class CudaTests:
    def setup_method(self):
        torch.manual_seed(810)
        self.device = torch.device("cuda")

    def test_entropy_dtype_value_and_gradient_reference(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            z = torch.randn(4, 7, device=self.device, dtype=dtype, requires_grad=True)
            target = torch.tensor([0, 2, 4, 6], device=self.device)
            weights = torch.tensor([0.0, -2.0, 0.5, 3.0], device=self.device, dtype=dtype)
            actual = sdca_loss(z, target, backend="torch", reduction="none")
            actual.backward(weights)
            ref = z.detach().cpu().double().requires_grad_()
            expected = torch.nn.functional.cross_entropy(ref, target.cpu(), reduction="none")
            expected.backward(weights.cpu().double())
            tol = {
                torch.float16: 2e-3,
                torch.bfloat16: 1e-2,
                torch.float32: 2e-6,
                torch.float64: 2e-13,
            }[dtype]
            torch.testing.assert_close(actual.cpu().double(), expected, rtol=tol, atol=tol / 10)
            assert z.grad is not None
            torch.testing.assert_close(z.grad.cpu().double(), ref.grad, rtol=tol, atol=tol / 10)
            assert z.grad.dtype == dtype
            assert z.grad[0].abs().sum().item() == 0

    def test_collapse_across_dtypes_and_reductions(self):
        for dtype, margin in [
            (torch.float16, 12.0),
            (torch.bfloat16, 12.0),
            (torch.float32, 40.0),
            (torch.float64, 100.0),
        ]:
            for reduction in ["none", "sum", "mean"]:
                z = torch.tensor(
                    [[margin, 0.0, 0.0], [margin, 0.0, 0.0]],
                    device=self.device,
                    dtype=dtype,
                    requires_grad=True,
                )
                actual = sdca_loss(
                    z,
                    torch.tensor([0, 0], device=self.device),
                    backend="torch",
                    reduction=reduction,
                )
                actual.sum().backward()
                tail = 2 * torch.exp(torch.tensor(-margin, dtype=torch.float64))
                expected = tail / (1 + tail) / (2 if reduction == "mean" else 1)
                tol = 0.02 if dtype in [torch.float16, torch.bfloat16] else 2e-6
                assert (actual > 0).all().item()
                assert z.grad is not None
                assert (z.grad[:, 0] < 0).all().item()
                torch.testing.assert_close(
                    -z.grad[:, 0].cpu().double(), expected.expand(2), rtol=tol, atol=0
                )
                # Stored half gradients can round each entry by one subnormal
                # unit; zero-sum is required to storage precision.
                info = torch.finfo(dtype)
                torch.testing.assert_close(
                    z.grad[:, 0],
                    -z.grad[:, 1:].sum(-1),
                    rtol=4 * info.eps,
                    atol=2 * info.tiny * info.eps,
                )

    def test_multilabel_small_differences_and_tail(self):
        for dtype in [torch.float32, torch.float64]:
            z = torch.tensor(
                [[0.0, 1e-10, -1e-10], [20.0, 20.0, -20.0]],
                device=self.device,
                dtype=dtype,
                requires_grad=True,
            )
            mask = torch.tensor([[True, True, True], [True, True, False]], device=self.device)
            sdca_loss(z, mask, backend="torch", reduction="sum").backward()
            assert z.grad is not None
            torch.testing.assert_close(z.grad[0], z.detach()[0] / 3, rtol=2e-6, atol=1e-20)
            assert z.grad[1, 0].item() < 0
            assert z.grad[1, 0].item() == z.grad[1, 1].item()
            torch.testing.assert_close(2 * z.grad[1, 0], -z.grad[1, 2], rtol=2e-6, atol=0)

    def test_noncontiguous_layout_shift_and_permutation(self):
        z = (
            torch.randn(2, 3, 5, device=self.device, dtype=torch.float64)
            .transpose(1, 2)
            .requires_grad_()
        )
        assert not z.is_contiguous()
        labels = torch.tensor([[0, 1, 2], [3, 4, 0]], device=self.device)
        values = sdca_loss(z, labels, backend="torch", dim=1, reduction="none")
        values.sum().backward()
        reference = torch.nn.functional.cross_entropy(z.detach(), labels, reduction="none")
        torch.testing.assert_close(values, reference)
        permutation = torch.tensor([2, 4, 0, 1, 3], device=self.device)
        inverse = torch.argsort(permutation)
        shifted = (z.detach()[:, permutation, :] + 1024).requires_grad_()
        changed = sdca_loss(shifted, inverse[labels], backend="torch", dim=-2, reduction="sum")
        changed.backward()
        torch.testing.assert_close(changed, values.sum(), atol=1e-12, rtol=1e-12)
        assert z.grad is not None
        torch.testing.assert_close(shifted.grad, z.grad[:, permutation, :], atol=1e-13, rtol=1e-12)

    def test_entropy_first_and_second_derivatives_for_masks(self):
        z = torch.randn(2, 4, device=self.device, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor(
            [[True, True, False, False], [False, True, False, True]], device=self.device
        )

        def fn(x: torch.Tensor) -> torch.Tensor:
            return sdca_loss(x, mask, backend="torch")

        assert gradcheck(fn, (z,))
        assert gradgradcheck(fn, (z,))

    def test_distillation_equal_near_far_temperature_and_teacher(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            for temperature in [0.5, 1.0, 3.0]:
                teacher = torch.tensor(
                    [[2.0, 1.0, 0.0]], device=self.device, dtype=dtype, requires_grad=True
                )
                student = teacher.detach().clone().requires_grad_()
                DistillationLoss(temperature=temperature)(student, teacher).backward()
                torch.testing.assert_close(student.grad, torch.zeros_like(student), atol=0, rtol=0)
                assert teacher.grad is None
                eps = {
                    torch.float16: 0.01,
                    torch.bfloat16: 0.05,
                    torch.float32: 1e-5,
                    torch.float64: 1e-8,
                }[dtype]
                for perturbation in [eps, 10.0]:
                    student = (
                        teacher.detach()
                        + torch.tensor(
                            [[perturbation, -perturbation, 0.0]], device=self.device, dtype=dtype
                        )
                    ).requires_grad_()
                    value = distillation_loss(
                        student, teacher, temperature=temperature, reduction="sum"
                    )
                    value.backward()
                    # Work with the represented scaled inputs: low precision
                    # and float32 APIs divide in float32 before the softmax.
                    work = torch.float64 if dtype == torch.float64 else torch.float32
                    s = (student.detach().to(work) / temperature).cpu().double()
                    t = (teacher.detach().to(work) / temperature).cpu().double()
                    expected = (s.softmax(-1) - t.softmax(-1)) / temperature
                    tol = 0.04 if dtype in [torch.float16, torch.bfloat16] else 2e-5
                    assert student.grad is not None
                    scale = float(expected.abs().max())
                    torch.testing.assert_close(
                        student.grad.cpu().double(), expected, rtol=tol, atol=tol * scale
                    )
                    assert torch.isfinite(value).item()
                    assert teacher.grad is None
        student = torch.tensor([[-1000.0, 1000.0, 0.0]], device=self.device, requires_grad=True)
        teacher = -student.detach()
        distillation_loss(student, teacher).backward()
        torch.testing.assert_close(
            student.grad, torch.tensor([[-1.0, 1.0, 0.0]], device=self.device), rtol=0, atol=0
        )

    def test_distillation_at_subnormal_temperatures(self):
        # 1/T overflows below 5.6e-309: dividing by T must not become a multiply by inf.
        for dtype in [torch.float32, torch.float64]:
            for temperature in [1e-20, 1e-300, 1e-310, 5e-324]:
                x = torch.tensor(
                    [[1.0, 0.0], [3.0, 3.0]], device=self.device, dtype=dtype, requires_grad=True
                )
                value = distillation_loss(x, x.detach(), temperature=temperature)
                value.backward()
                message = f"{dtype} T={temperature}"
                assert abs(value.item() - math.log(2) / 2) <= 1e-6, message
                torch.testing.assert_close(x.grad, torch.zeros_like(x), atol=0, rtol=0)

    def test_distillation_layout_reductions_and_autograd_contract(self):
        teacher = torch.randn(2, 4, 3, device=self.device, dtype=torch.float64)
        z = (teacher + 0.03 * torch.randn_like(teacher)).requires_grad_()

        def fn(x: torch.Tensor) -> torch.Tensor:
            return distillation_loss(x, teacher, temperature=2.5, dim=1)

        assert gradcheck(fn, (z,))
        unreduced = distillation_loss(z, teacher, dim=1, reduction="none")
        torch.testing.assert_close(
            unreduced.sum(), distillation_loss(z, teacher, dim=1, reduction="sum")
        )
        torch.testing.assert_close(unreduced.mean(), distillation_loss(z, teacher, dim=1))
        gradient = torch.autograd.grad(fn(z), z, create_graph=True)[0]
        with pytest.raises(RuntimeError):
            torch.autograd.grad(gradient.sum(), z)

    def test_validation_and_unsupported_cuda_native_paths(self):
        z = torch.zeros(2, 3, device=self.device)
        y = torch.tensor([0, 1], device=self.device)
        for options in [
            dict(backend="bad"),
            dict(backend="torch", loss="bad"),
            dict(backend="native"),
            dict(backend="torch", k=True),
            dict(backend="torch", gamma=-1.0),
            dict(backend="torch", alpha=float("nan")),
            dict(backend="torch", reduction="bad"),
        ]:
            with pytest.raises(ValueError):
                sdca_loss(z, y, **options)
        for target in [
            y.cpu(),
            y.float(),
        ]:
            with pytest.raises(ValueError):
                sdca_loss(z, target, backend="torch")
        for target in [
            torch.tensor([0, 3], device=self.device),
            torch.zeros_like(z, dtype=torch.bool),
        ]:
            assert bool(sdca_loss(z, target, backend="torch").isnan().any())
        assert bool(sdca_loss(z.fill_(float("nan")), y, backend="torch").isnan().all())
        z.zero_()
        for logits, target in [(None, y), (z.to(torch.int64), y), (z[:, :1], y)]:
            with pytest.raises(ValueError):
                sdca_loss(logits, target, backend="torch")
        for teacher, options in [
            (z.cpu(), {}),
            (z.double(), {}),
            (z[:1], {}),
            (z, dict(temperature=0)),
            (z, dict(reduction="bad")),
        ]:
            with pytest.raises(ValueError):
                distillation_loss(z, teacher, **options)
        for student, teacher in [
            (z.to(torch.int64), z.to(torch.int64)),
            (z[:, :1], z[:, :1]),
        ]:
            with pytest.raises(ValueError):
                distillation_loss(student, teacher)
        assert bool(distillation_loss(z, z + float("inf")).isnan().all())
        for reduction in ("none", "sum", "mean"):
            empty_losses = (
                sdca_loss(z[:0], y[:0], backend="torch", reduction=reduction),
                distillation_loss(z[:0], z[:0], reduction=reduction),
            )
            for value in empty_losses:
                if reduction == "none":
                    assert value.shape == (0,)
                elif reduction == "sum":
                    assert value.item() == 0
                else:
                    assert value.isnan()

    def test_small_cuda_training_and_autocast(self):
        x = torch.eye(4, device=self.device).repeat(8, 1)
        y = torch.arange(4, device=self.device).repeat(8) % 3
        model = torch.nn.Linear(4, 3, device=self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.8)
        criterion = SDCALoss(backend="torch")
        first = criterion(model(x), y).item()
        for _ in range(30):
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value = criterion(model(x), y)
            value.backward()
            optimizer.step()
        assert criterion(model(x), y).item() < first / 3
        assert (model(x).argmax(-1) == y).all().item()
