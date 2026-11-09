"""Independent references for vanilla entropy's storage and reduction boundaries."""

import itertools
import math

import mpmath as mp
import pytest
import torch
from libsdca.torch import sdca_loss
from torch.autograd import gradcheck, gradgradcheck


class SoftmaxLimitTests:
    def devices(self):
        return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def test_float32_tensor_aggregate_tail(self):
        # Explicit float32 and masks take the portable tensor formula on CPU.
        # Summing rounded exp(-110) entries would erase a representable tail.
        for device in self.devices():
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                for gap in (100.0, 110.0):
                    x = torch.full((1, 4097), -gap, dtype=dtype, device=device, requires_grad=True)
                    with torch.no_grad():
                        x[0, 0] = 0
                    for mask in (False, True):
                        target = (
                            torch.zeros_like(x, dtype=torch.bool)
                            if mask
                            else torch.tensor([0], device=device)
                        )
                        if mask:
                            target[0, 0] = True
                        loss = sdca_loss(x, target, precision="float32", backend="torch")
                        (gradient,) = torch.autograd.grad(loss, x)
                        with mp.workdps(100):
                            tail = 4096 * mp.exp(-gap)
                            value = float(mp.log1p(tail))
                            derivative = float(-tail / (1 + tail))
                        quantum = torch.finfo(torch.float32).tiny * torch.finfo(torch.float32).eps
                        torch.testing.assert_close(
                            loss, torch.tensor(value, device=device), rtol=1e-6, atol=quantum
                        )
                        torch.testing.assert_close(
                            gradient[0, 0],
                            torch.tensor(derivative, dtype=dtype, device=device),
                            rtol=1e-6,
                            atol=torch.finfo(dtype).tiny * torch.finfo(dtype).eps,
                        )

    def test_float32_tensor_upstream_scaling(self):
        info = torch.finfo(torch.float32)
        for device in self.devices():
            for gap in (-20.0, 0.0, 20.0, 110.0):
                x = torch.tensor([[0.0, -gap]], device=device, requires_grad=True)
                mask = torch.tensor([[True, False]], device=device)
                value = sdca_loss(x, mask, precision="float32", backend="torch")
                for scale in (info.max, -info.max, info.tiny * info.eps):
                    (gradient,) = torch.autograd.grad(
                        value, x, torch.full_like(value, scale), retain_graph=True
                    )
                    with mp.workdps(100):
                        expected = float(mp.mpf(scale) / (1 + mp.exp(gap)))
                    reference = torch.tensor([[-expected, expected]], device=device)
                    torch.testing.assert_close(
                        gradient, reference, rtol=2e-6, atol=info.tiny * info.eps
                    )

    def test_total_underflow_is_zero_for_every_reduction(self):
        for device in self.devices():
            maximum = torch.finfo(torch.float64).max
            x = torch.tensor(
                [[maximum, -maximum]] * 4, dtype=torch.float64, device=device, requires_grad=True
            )
            y = torch.zeros(4, dtype=torch.long, device=device)
            for reduction in ("none", "mean", "sum"):
                loss = sdca_loss(x, y, reduction=reduction)
                assert (loss == 0).all()
                (gradient,) = torch.autograd.grad(loss.sum(), x)
                assert (gradient == 0).all()

    def test_aggregate_tail_survives_individual_underflow(self):
        for device in self.devices():
            for dtype, gap in [(torch.float32, 110.0), (torch.float64, 750.0)]:
                x = torch.full((1, 1000), -gap, dtype=dtype, device=device)
                x[0, 0] = 0
                x.requires_grad_()
                loss = sdca_loss(x, torch.tensor([0], device=device))
                loss.backward()
                with mp.workdps(100):
                    tail = 999 * mp.exp(-gap)
                    expected = float(mp.log1p(tail))
                rounded = torch.tensor(expected, dtype=loss.dtype, device=device)
                assert loss.item() > 0
                assert x.grad is not None
                assert x.grad[0, 0].item() < 0
                torch.testing.assert_close(
                    loss, rounded, rtol=0, atol=torch.finfo(dtype).tiny * torch.finfo(dtype).eps
                )

    def test_scaling_precedes_gradient_underflow(self):
        for device in self.devices():
            for dtype, gap in [
                (torch.float16, 20.0),
                (torch.bfloat16, 120.0),
                (torch.float32, 120.0),
                (torch.float64, 800.0),
            ]:
                x = torch.tensor([[0.0, -gap]], dtype=dtype, device=device, requires_grad=True)
                y = torch.tensor([0], device=device)
                loss = sdca_loss(x, y)
                scale = torch.finfo(dtype).max
                for sign in (-1, 1):
                    (g,) = torch.autograd.grad(
                        loss, x, torch.full_like(loss, sign * scale), retain_graph=True
                    )
                    with mp.workdps(100):
                        expected = float(mp.exp(-gap) / (1 + mp.exp(-gap)) * scale * sign)
                    reference = torch.tensor([[-expected, expected]], dtype=dtype, device=device)
                    assert g[0, 0].item() != 0
                    torch.testing.assert_close(
                        g,
                        reference,
                        rtol=1e-3
                        if dtype in (torch.float16, torch.bfloat16)
                        else 1e-12
                        if dtype == torch.float64
                        else 1e-6,
                        atol=0,
                    )

    def test_mean_does_not_overflow_a_finite_answer(self):
        for device in self.devices():
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                gap = torch.finfo(dtype).max / 4
                x = torch.tensor([[0, gap]] * 8, dtype=dtype, device=device, requires_grad=True)
                loss = sdca_loss(x, torch.zeros(8, dtype=torch.long, device=device))
                assert torch.isfinite(loss)
                assert round(abs(loss.item() / gap - 1.0), 6) == 0
                loss.backward()
                torch.testing.assert_close(
                    x.grad, torch.tensor([[-0.125, 0.125]] * 8, dtype=dtype, device=device)
                )

    def test_sum_of_individually_underflowing_losses(self):
        for device in self.devices():
            x = torch.tensor([[0.0, -750.0]] * 1000, dtype=torch.float64, device=device)
            y = torch.zeros(1000, dtype=torch.long, device=device)
            loss = sdca_loss(x, y, reduction="sum")
            with mp.workdps(100):
                expected = float(1000 * mp.log1p(mp.exp(-750)))
            assert loss.item() > 0
            assert loss.item() == expected

    def test_mean_of_unrepresentable_individual_losses(self):
        for device in self.devices():
            maximum = torch.finfo(torch.float64).max
            x = torch.tensor([[-maximum, maximum], [0, 0]], dtype=torch.float64, device=device)
            loss = sdca_loss(x, torch.zeros(2, dtype=torch.long, device=device))
            assert torch.isfinite(loss)
            assert abs(loss.item() / maximum - 1) < 1e-12

    def test_cuda_double_minimum_subnormal(self):
        for device in self.devices():
            x = torch.tensor(
                [[0.0, -745.0]], dtype=torch.float64, device=device, requires_grad=True
            )
            loss = sdca_loss(x, torch.tensor([0], device=device))
            loss.backward()
            minimum = math.ulp(0.0)
            assert loss.item() == minimum
            assert x.grad is not None
            assert x.grad[0, 0].item() == -minimum

    def test_index_entropy_higher_derivatives(self):
        for device in self.devices():
            x = torch.tensor(
                [[0.2, -0.4, 0.7], [1.0, 0.0, -0.1]],
                dtype=torch.float64,
                device=device,
                requires_grad=True,
            )
            y = torch.tensor([0, 2], device=device)

            def f(z: torch.Tensor):
                return sdca_loss(z, y, backend="torch")

            assert gradcheck(f, (x,))
            assert gradgradcheck(f, (x,))

    def test_misaligned_logits_keep_higher_derivatives(self):
        # A view one float64 into its storage is not 16-byte aligned; the
        # fused CUDA path must still differentiate through it twice. The CPU
        # native path is first order only, so select the tensor backend there.
        for device in self.devices():
            base = torch.randn(7, dtype=torch.float64, device=device, requires_grad=True)
            y = torch.tensor([0, 2], device=device)

            def f(b: torch.Tensor):
                return sdca_loss(b[1:].view(2, 3), y, backend="torch")

            assert gradcheck(f, (base,))
            assert gradgradcheck(f, (base,))

    def test_in_place_target_change_before_backward_raises(self):
        for device in self.devices():
            x = torch.randn(2, 3, device=device, requires_grad=True)
            y = torch.tensor([0, 2], device=device)
            loss = sdca_loss(x, y)
            y[0] = 1
            with pytest.raises(RuntimeError):
                loss.backward()

    def test_zero_upstream_and_extreme_second_derivatives(self):
        for device in self.devices():
            for gap in (0.0, 1000.0, -1000.0):
                x = torch.tensor(
                    [[0.0, -gap]], dtype=torch.float64, device=device, requires_grad=True
                )
                u = torch.tensor(0.0, dtype=torch.float64, device=device, requires_grad=True)
                loss = sdca_loss(x, torch.tensor([0], device=device), backend="torch")
                (gradient,) = torch.autograd.grad(loss, x, u, create_graph=True)
                second, weight = torch.autograd.grad(gradient[0, 0], (x, u))
                assert torch.isfinite(second).all()
                assert torch.isfinite(weight)
                assert (
                    round(abs(weight.item() - (-0.5 if gap == 0 else 0.0 if gap > 0 else -1.0)), 14)
                    == 0
                )

    def test_fused_validation(self):
        for device in self.devices():
            for x, y in [
                (torch.tensor([[float("nan"), 0]]), torch.tensor([0])),
                (torch.zeros(1, 2), torch.tensor([2])),
            ]:
                if device == "cpu":
                    with pytest.raises(ValueError):
                        sdca_loss(x.to(device), y.to(device))
                else:
                    assert bool(sdca_loss(x.to(device), y.to(device)).isnan().all())

    def test_mixed_rows_and_upstream_scales_against_reference(self):
        # Ordinary cached rows, the tail branch, and very small competitors
        # occur together. References never use an implementation's output.
        rows = [
            [0, 0, 0, 0],
            [0, 0.25, -0.125, 0.5],
            [0, -36, -37, -38],
            [0, -38, -39, -40],
            [0, 1, -599, -601],
            [0, 1, -800, -1400],
        ]
        for device in self.devices():
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                x = torch.tensor(rows, dtype=dtype, device=device, requires_grad=True)
                y = torch.zeros(len(rows), dtype=torch.long, device=device)
                minimum = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
                for reduction in ("none", "sum", "mean"):
                    loss = sdca_loss(x, y, reduction=reduction, backend="torch")
                    for scale in (
                        0,
                        1,
                        -1,
                        1e-20,
                        minimum,
                        -minimum,
                        torch.finfo(dtype).max,
                        -torch.finfo(dtype).max,
                    ):
                        u = torch.full_like(loss, scale)
                        if reduction == "none":
                            u[1::2] *= -1
                        (gradient,) = torch.autograd.grad(loss, x, u, retain_graph=True)
                        with mp.workdps(100):
                            expected = []
                            for i, row in enumerate(x.detach().cpu().double().tolist()):
                                anchor = max(row)
                                e = [mp.exp(mp.mpf(v) - anchor) for v in row]
                                denominator = mp.fsum(e)
                                g = [v / denominator for v in e]
                                g[0] = -mp.fsum(g[1:])
                                weight = mp.mpf(u[i].item() if reduction == "none" else u.item())
                                if reduction == "mean":
                                    weight /= len(rows)
                                expected.append([float(v * weight) for v in g])
                        reference = torch.tensor(expected, dtype=dtype, device=device)
                        torch.testing.assert_close(
                            gradient,
                            reference,
                            rtol=1e-12
                            if dtype == torch.float64
                            else 1e-6
                            if dtype == torch.float32
                            else 2e-3,
                            atol=minimum,
                        )

    def test_precision_table_of_the_numerical_notes(self, subtests: pytest.Subtests):
        # numerics.md, "PyTorch loss precision": logits [0, -gap], target 0, upstream 1 (a
        # GradScaler's 65504 for float16). The table lists the 100-digit loss and target
        # gradient rounded to the output dtypes; libsdca is within one unit of them with
        # float64 work and within 8 float32 eps more with float32 exponentials (default).
        table = [  # dtype, gap, upstream, documented loss and gradient with their digits
            (torch.bfloat16, 20.0, 1.0, (2.06115369e-9, 9), (-2.066e-9, 4)),
            (torch.float32, 20.0, 1.0, (2.06115369e-9, 9), (-2.06115369e-9, 9)),
            (torch.float64, 40.0, 1.0, (4.24835426e-18, 9), (-4.24835426e-18, 9)),
            (torch.float16, 20.0, 65504.0, None, (-1.35064e-4, 6)),
        ]
        eps32 = torch.finfo(torch.float32).eps
        for dtype, gap, upstream, loss_doc, grad_doc in table:
            with mp.workdps(100):
                loss_ref = float(mp.log1p(mp.exp(-gap)))
                grad_ref = float(-upstream * mp.exp(-gap) / (1 + mp.exp(-gap)))
            out = torch.float64 if dtype == torch.float64 else torch.float32
            for documented, ref, kind in ((loss_doc, loss_ref, out), (grad_doc, grad_ref, dtype)):
                if documented is not None:
                    value, digits = documented
                    rounded = torch.tensor(ref, dtype=torch.float64).to(kind).item()
                    assert float(f"{rounded:.{digits - 1}e}") == value
            for device in self.devices():
                backends = ["auto", "torch"] + (["native"] if device == "cpu" else [])
                for backend, precision in itertools.product(backends, (None, "float64")):
                    with subtests.test(dtype=dtype, device=device, backend=backend, p=precision):
                        x = torch.tensor([[0.0, -gap]], dtype=dtype, device=device)
                        x.requires_grad_()
                        y = torch.tensor([0], device=device)
                        loss = sdca_loss(x, y, backend=backend, precision=precision)
                        (g,) = torch.autograd.grad(loss, x, torch.full_like(loss, upstream))
                        fp32 = precision is None and dtype != torch.float64
                        extra = 8 * eps32 if fp32 else 0.0
                        checks = ((loss, loss_ref, out), (g[0, 0], grad_ref, dtype))
                        for result, ref, kind in checks:
                            unit = torch.finfo(kind).eps
                            assert abs(result.item() - ref) <= (unit + extra) * abs(ref)

    def test_hinge_keeps_a_tiny_margin(self, subtests: pytest.Subtests):
        # numerics.md: multiclass_hinge of [1, 1e-8] in float32 is 1e-8 (PyTorch's margin
        # 1 + 1e-8 - 1, formed in float32, is 0).
        tiny = torch.tensor(1e-8, dtype=torch.float32).item()
        eps = torch.finfo(torch.float32).eps
        for device in self.devices():
            backends = ["auto", "torch"] + (["native"] if device == "cpu" else [])
            for backend in backends:
                with subtests.test(device=device, backend=backend):
                    x = torch.tensor([[1.0, 1e-8]], device=device)
                    y = torch.tensor([0], device=device)
                    loss = sdca_loss(x, y, "multiclass_hinge", backend=backend)
                    assert abs(loss.item() - tiny) <= 2 * eps * tiny
