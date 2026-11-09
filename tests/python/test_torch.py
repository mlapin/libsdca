import functools
import itertools
import math
from collections.abc import Callable
from typing import Any

import libsdca
import mpmath
import pytest
import torch
from libsdca import LOSSES
from libsdca.torch import (
    CrossEntropyLoss,
    DistillationLoss,
    MulticlassHingeLoss,
    MultilabelCrossEntropyLoss,
    MultilabelHingeLoss,
    ProximalCrossEntropyLoss,
    ProximalDistillationLoss,
    SDCALoss,
    TopKCrossEntropyLoss,
    TopKHingeAlphaLoss,
    TopKHingeBetaLoss,
    TopKHingeLoss,
    cross_entropy,
    distillation_loss,
    multiclass_hinge_loss,
    multilabel_cross_entropy,
    multilabel_hinge_loss,
    proximal_cross_entropy,
    proximal_distillation_loss,
    sdca_loss,
    topk_cross_entropy,
    topk_hinge_alpha_loss,
    topk_hinge_beta_loss,
    topk_hinge_loss,
)
from mpmath_types import MPReal
from torch.autograd import gradcheck, gradgradcheck


class TorchTests:
    def test_native_objectives_gradcheck(self):
        torch.manual_seed(12)
        z = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
        for objective in LOSSES:
            target = torch.tensor([1, 3])
            if objective.startswith("multilabel"):
                target = torch.tensor([[1, 1, 0, 0, 0], [0, 1, 0, 1, 1]], dtype=torch.bool)
            assert gradcheck(
                lambda x: sdca_loss(x, target, objective, k=2, gamma=0.6, alpha=0.9), (z,)
            ), objective

    def test_tensor_entropy_first_and_second_derivatives(self):
        torch.manual_seed(23)
        z = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        for target in [
            torch.tensor([1, 2]),
            torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=torch.bool),
        ]:

            def f(x: torch.Tensor) -> torch.Tensor:
                return sdca_loss(x, target, backend="torch")

            assert gradcheck(f, (z,))
            assert gradgradcheck(f, (z,))
            torch.testing.assert_close(f(z), sdca_loss(z, target), rtol=1e-14, atol=1e-14)

    def test_uniform_target_retains_small_logit_differences(self):
        for backend in ["native", "torch"]:
            z = torch.tensor([[0.0, 1e-10, -1e-10]], dtype=torch.float32, requires_grad=True)
            sdca_loss(z, torch.ones_like(z, dtype=torch.bool), backend=backend).backward()
            torch.testing.assert_close(z.grad, z.detach() / 3, rtol=1e-6, atol=1e-18)
        z = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
        target = torch.ones_like(z, dtype=torch.bool)
        assert gradgradcheck(lambda x: sdca_loss(x, target, backend="torch"), (z,))

    def test_collapse_and_arbitrary_upstream_gradient(self):
        for dtype, gap in [(torch.float32, 18.0), (torch.float64, 50.0)]:
            for backend in ["native", "torch"]:
                z = torch.tensor(
                    [[gap / 2, -gap / 2, -gap / 2], [0.4, 0.2, -0.5]],
                    dtype=dtype,
                    requires_grad=True,
                )
                values = sdca_loss(z, torch.tensor([0, 2]), backend=backend, reduction="none")
                values.backward(torch.tensor([3.0, 0.0], dtype=dtype))
                assert values[0].item() > 0
                assert z.grad is not None
                assert z.grad[0, 0].item() < 0
                torch.testing.assert_close(z.grad[0, 0], -z.grad[0, 1:].sum(), rtol=1e-6, atol=0)
                assert z.grad[1].abs().sum().item() == 0

    def test_layout_and_standard_cross_entropy_agree(self):
        torch.manual_seed(8)
        z = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        target = torch.randint(0, 3, (2, 4))
        reference = torch.nn.functional.cross_entropy(z, target)
        for backend in ["native", "torch"]:
            actual = sdca_loss(z, target, dim=1, backend=backend)
            torch.testing.assert_close(actual, reference, rtol=1e-14, atol=1e-15)
            ga = torch.autograd.grad(actual, z)[0]
            gr = torch.autograd.grad(reference, z, retain_graph=True)[0]
            torch.testing.assert_close(ga, gr, rtol=1e-14, atol=1e-15)

    def test_distillation_equal_near_and_far(self):
        teacher = torch.tensor([[2.0, 1.0, 0.0], [1000.0, -1000.0, 0.0]], dtype=torch.float64)
        student = teacher.clone().requires_grad_()
        distillation_loss(student, teacher).backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student), atol=0, rtol=0)
        z = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True)
        t = z.detach() + torch.tensor([[0.03, -0.02, 0.01]])
        assert gradcheck(lambda x: distillation_loss(x, t, temperature=2.5), (z,))
        far = -teacher.clone().requires_grad_()
        value = distillation_loss(far, teacher)
        g = torch.autograd.grad(value, far)[0]
        assert torch.isfinite(value) and torch.isfinite(g).all()
        # A tiny temperature overflows the scaled logits, not the centered ones.
        x = torch.tensor([[1e30, -1e30]], dtype=torch.float32, requires_grad=True)
        value = distillation_loss(x, x.detach(), temperature=1e-20)
        value.backward()
        assert value.item() == 0.0
        torch.testing.assert_close(x.grad, torch.zeros_like(x), atol=0, rtol=0)
        # Compare a represented fp32 perturbation against an fp64 reference.
        t = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
        z = (t + torch.tensor([[-1e-7, 1e-7, 0.0]])).requires_grad_()
        distillation_loss(z, t).backward()
        exact = torch.softmax(z.detach().double(), -1) - torch.softmax(t.double(), -1)
        assert z.grad is not None
        torch.testing.assert_close(z.grad.double(), exact, rtol=2.4e-7, atol=1e-14)

    def test_distillation_keeps_represented_differences(self):
        """Gradients near equal distributions against 400-digit softmax(s/T) - softmax(t/T)."""
        mpmath.mp.dps = 400

        def softmax(row: list[float], temperature: float) -> list[MPReal]:
            e = [mpmath.exp(mpmath.mpf(v) / temperature) for v in row]
            return [v / sum(e) for v in e]

        cases = [
            (torch.float32, [20.0, 0.0, 0.0], [20.0, 1e-7, 0.0], 1.0),
            (torch.float32, [2.0, 0.0, 0.0], [2.0, 1e-7, 0.0], 1.0),
            (torch.float64, [50.0, 0.0, 0.0], [50.0, 1e-16, 0.0], 1.0),
            (torch.float64, [3e5, 1.0, 0.0], [3e5 + 1e-10, 1.0, 0.0], 1e3),
        ]
        for dtype, teacher, student, temperature in cases:
            t = torch.tensor([teacher], dtype=dtype)
            z = torch.tensor([student], dtype=dtype, requires_grad=True)
            distillation_loss(z, t, temperature=temperature, reduction="sum").backward()
            # The represented inputs, exactly.
            q = softmax(z.detach()[0].tolist(), temperature)
            p = softmax(t[0].tolist(), temperature)
            exact = [float((a - b) / temperature) for a, b in zip(q, p, strict=True)]
            scale = max(abs(v) for v in exact)
            assert scale > 0
            assert z.grad is not None
            error = max(abs(a - b) for a, b in zip(z.grad[0].tolist(), exact, strict=True))
            assert error / scale < 1e-5, f"{dtype} {teacher} {z.grad}"

    def test_distillation_at_temperatures_beyond_the_dtype(self):
        for dtype in [torch.float32, torch.float64]:
            for temperature in [1e-20, 1e-45, 1e-100, 1e-300, 1e-310, 5e-324]:
                x = torch.tensor([[1.0, 0.0], [3.0, 3.0]], dtype=dtype, requires_grad=True)
                value = distillation_loss(x, x.detach(), temperature=temperature)
                value.backward()
                message = f"{dtype} T={temperature}"
                assert value.dtype == dtype
                assert torch.isfinite(value), message
                # Equal logits: the row with a tie keeps entropy log 2, at any T.
                assert abs(value.item() - math.log(2) / 2) <= 1e-6, message
                torch.testing.assert_close(x.grad, torch.zeros_like(x), atol=0, rtol=0)

    def test_tensor_losses_at_the_edge_of_the_dtype(self):
        """Losses that round to zero stay zero, forward and backward."""
        for dtype, big in [(torch.float32, 3e38), (torch.float64, 1e308)]:
            z = torch.tensor([[big, -big]], dtype=dtype, requires_grad=True)
            value = sdca_loss(z, torch.tensor([0]), backend="torch")
            (gradient,) = torch.autograd.grad(value, z)
            assert value.item() == 0.0
            torch.testing.assert_close(gradient, torch.zeros_like(z), atol=0, rtol=0)
        z = torch.tensor([[1e308, 1e308, -1e308]], dtype=torch.float64, requires_grad=True)
        for backend in ["native", "torch"]:
            value = sdca_loss(z, torch.tensor([0]), "topk_cross_entropy", k=2, backend=backend)
            (gradient,) = torch.autograd.grad(value, z)
            assert value.item() == 0.0, backend
            assert bool(torch.isfinite(gradient).all()), backend

    def test_precision(self):
        g = torch.Generator().manual_seed(11)
        z = 2 * torch.randn(16, 6, generator=g)
        labels = torch.randint(0, 6, (16,), generator=g)
        mask = torch.rand(16, 6, generator=g) < 0.4
        mask[torch.arange(16), labels] = True
        mask[torch.arange(16), (labels + 1) % 6] = False
        for loss in LOSSES:
            target = mask if loss.startswith("multilabel") else labels
            for backend in ("native", "torch"):
                values, grads = {}, {}
                for precision in (None, "float64", "float32", torch.float64, torch.float32):
                    x = z.clone().requires_grad_()
                    value = sdca_loss(x, target, loss, backend=backend, k=2, precision=precision)
                    value.backward()
                    assert x.grad is not None
                    assert (value.dtype, x.grad.dtype) == (torch.float32, torch.float32)
                    values[precision], grads[precision] = value, x.grad
                message = f"{loss} {backend}"
                torch.testing.assert_close(values["float32"], values["float64"], msg=message)
                torch.testing.assert_close(grads["float32"], grads["float64"], msg=message)
                torch.testing.assert_close(values[torch.float32], values["float32"], rtol=0, atol=0)
                torch.testing.assert_close(values[torch.float64], values["float64"], rtol=0, atol=0)
        # float32 arithmetic falls back to float64 where float32 cannot hold alpha, gamma or the
        # spread.
        wide = torch.tensor([[3e38, -3e38, 0.0], [1.0, 2.0, 3.0]])
        # gamma only matters to the hinges.
        both = ("proximal_cross_entropy", "topk_hinge_alpha")
        for x, alpha, gamma, losses in (
            (z, 1e-40, 1.0, both),
            (z, 1.0, 1e-40, ("topk_hinge_alpha",)),
            (wide, 1.0, 1.0, both),
        ):
            target = torch.zeros(len(x), dtype=torch.int64)
            for loss, backend in itertools.product(losses, ("native", "torch")):
                options = {"backend": backend, "alpha": alpha, "gamma": gamma, "reduction": "none"}
                actual = sdca_loss(x, target, loss, precision="float32", **options)
                expected = sdca_loss(x, target, loss, precision="float64", **options)
                message = f"{loss} {backend}"
                if backend == "native" and x is wide:
                    # The native library falls back per row: the row float32 holds keeps float32
                    # work, within the proximal oracle's float32 limit (one rounding plus 60 eps).
                    alone = sdca_loss(x[1:], target[1:], loss, precision="float32", **options)
                    torch.testing.assert_close(actual[1:], alone, rtol=0, atol=0, msg=message)
                    eps = torch.finfo(torch.float32).eps
                    torch.testing.assert_close(
                        actual[1:], expected[1:], rtol=60.5 * eps, atol=0, msg=message
                    )
                    actual, expected = actual[:1], expected[:1]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=message)
        # precision="float64" computes tensor entropy and distillation in float64.
        x = (z + 1e-7).requires_grad_()
        teacher = torch.randn(16, 6, generator=g)
        options = {"temperature": 2.0, "reduction": "none"}
        f64 = distillation_loss(x, teacher, precision="float64", **options)
        reference = distillation_loss(x.double(), teacher.double(), **options)
        torch.testing.assert_close(f64, reference.float(), rtol=0, atol=0)
        entropy = sdca_loss(x, labels, backend="torch", precision="float64", reduction="none")
        double = sdca_loss(x.double(), labels, backend="torch", reduction="none")
        torch.testing.assert_close(entropy, double.float(), rtol=0, atol=0)
        assert "precision='float32'" in repr(SDCALoss("cross_entropy", precision="float32"))
        assert "precision='float64'" in repr(DistillationLoss(precision="float64"))
        for x, precision in (
            (z.double(), "float32"),
            (z.double(), torch.float32),
            (z.double(), "mixed"),
            (z, "mixed"),
        ):
            with pytest.raises(ValueError, match="precision"):
                sdca_loss(x, labels, precision=precision)
            with pytest.raises(ValueError, match="precision"):
                distillation_loss(x, x, precision=precision)
        with pytest.raises(ValueError, match="precision"):
            sdca_loss(z, labels, precision="half")

    def test_first_order_losses_reject_double_backward(self):
        """Differentiating a first-order gradient raises, alone or inside another term."""
        z = torch.randn(4, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
        labels = torch.tensor([0, 1, 2, 3])
        mask = torch.nn.functional.one_hot(labels, 5).bool()
        mask[:, 4] = True
        multilabel = (
            "cross_entropy",
            "multilabel_hinge",
            "multilabel_cross_entropy",
            "proximal_cross_entropy",
        )
        cases = [
            (objective, backend, target, k)
            for objective in LOSSES
            for backend in ("native", "torch", "auto")
            for target in (labels, mask)
            # topk_cross_entropy at k = 1 is cross_entropy; tensor cross-entropy and integer
            # entropy on every backend have higher derivatives.
            for k in ((2,) if objective == "topk_cross_entropy" else (1, 2))
            if objective not in ("cross_entropy", "multilabel_cross_entropy")
            or (backend == "native" and target is mask)
            if target is labels or objective in multilabel
        ]
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for dtype in (torch.float64, torch.float32):
                x = z.to(device, dtype).requires_grad_()
                teacher = (2 * z).to(device, dtype)
                losses: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = [
                    (
                        f"{objective} {backend} k={k} {target.dtype}",
                        functools.partial(
                            sdca_loss,
                            target=target.to(device),
                            loss=objective,
                            backend=backend,
                            k=k,
                            alpha=0.7,
                            reduction="sum",
                        ),
                    )
                    for objective, backend, target, k in cases
                    if device == "cpu" or backend != "native"
                ]
                losses.append(("distillation", lambda v: distillation_loss(v, teacher)))
                for name, f in losses:
                    message = f"{name} {device} {dtype}"
                    (g,) = torch.autograd.grad(f(x), x, create_graph=True)
                    assert g.requires_grad, message
                    # The message names the losses with higher derivatives, not entropy alone.
                    only = "first derivatives only.*topk_cross_entropy at k = 1"
                    for use in (g.sum(), (g * x).sum()):
                        with pytest.raises(RuntimeError, match=only):
                            torch.autograd.grad(use, x)
                    # Without create_graph the gradient is the same.
                    (plain,) = torch.autograd.grad(f(x), x)
                    torch.testing.assert_close(plain, g.detach(), rtol=0, atol=0, msg=message)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_tensor_entropy_and_distillation_on_cuda(self):
        # The tensor backend and the distillation loss are device-generic;
        # the native backend is CPU-only and must say so.
        torch.manual_seed(5)
        device = torch.device("cuda")
        for dtype in (torch.float32, torch.float64):
            z = torch.randn(4, 6, dtype=dtype)
            labels = torch.randint(0, 6, (4,))
            mask = torch.zeros(4, 6, dtype=torch.bool)
            mask[torch.arange(4), labels] = True
            mask[torch.arange(4), (labels + 1) % 6] = True
            for target in (labels, mask):
                cpu = z.clone().requires_grad_()
                gpu = z.to(device).requires_grad_()
                values = sdca_loss(cpu, target, backend="torch", reduction="none")
                device_values = sdca_loss(gpu, target.to(device), backend="torch", reduction="none")
                torch.testing.assert_close(device_values.cpu(), values)
                values.sum().backward()
                device_values.sum().backward()
                assert gpu.grad is not None
                torch.testing.assert_close(gpu.grad.cpu(), cpu.grad)
                with pytest.raises(ValueError):
                    sdca_loss(gpu, target.to(device), backend="native")
            confident = torch.tensor(
                [[20.0, -20.0, -20.0]], dtype=dtype, device=device, requires_grad=True
            )
            value = sdca_loss(confident, torch.tensor([0], device=device), backend="torch")
            value.backward()
            assert value.item() > 0
            assert confident.grad is not None
            assert confident.grad[0, 0].item() < 0
            torch.testing.assert_close(
                confident.grad[0, 0], -confident.grad[0, 1:].sum(), rtol=1e-6, atol=0
            )
            teacher = torch.randn(3, 5, dtype=dtype, device=device)
            student = (teacher + 0.01 * torch.randn_like(teacher)).requires_grad_()
            reference = (teacher.cpu() + (student.detach() - teacher).cpu()).requires_grad_()
            distillation_loss(student, teacher).backward()
            distillation_loss(reference, teacher.cpu()).backward()
            assert student.grad is not None
            torch.testing.assert_close(student.grad.cpu(), reference.grad)
        z = torch.randn(2, 4, dtype=torch.float64, device=device, requires_grad=True)
        target = torch.tensor([1, 2], device=device)
        assert gradcheck(lambda x: sdca_loss(x, target, backend="torch"), (z,))
        assert gradgradcheck(lambda x: sdca_loss(x, target, backend="torch"), (z,))
        half = torch.randn(2, 4, dtype=torch.bfloat16, device=device, requires_grad=True)
        sdca_loss(half, target, backend="torch").backward()
        assert half.grad is not None
        assert half.grad.dtype == torch.bfloat16

    def test_neural_network_trains_through_native_loss(self):
        torch.manual_seed(1)
        x = torch.randn(60, 3)
        y = x.argmax(-1)
        net = torch.nn.Sequential(torch.nn.Linear(3, 12), torch.nn.Tanh(), torch.nn.Linear(12, 3))
        optimizer = torch.optim.Adam(net.parameters(), lr=0.04)
        criterion = SDCALoss("proximal_cross_entropy", alpha=1.0)
        first = criterion(net(x), y).item()
        for _ in range(80):
            optimizer.zero_grad()
            criterion(net(x), y).backward()
            optimizer.step()
        assert criterion(net(x), y).item() < first / 3
        assert (net(x).argmax(-1) == y).float().mean().item() > 0.9

    def test_scalar_cache_after_inference_and_across_streams(self):
        from libsdca.torch import _over, _scalars

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            _scalars.clear()
            x = torch.tensor([2.0], dtype=torch.float64, device=device)
            with torch.inference_mode():
                _over(x, 0.7)
            x.requires_grad_()
            _over(x, 0.7).sum().backward()
            torch.testing.assert_close(x.grad, torch.full_like(x, 1 / 0.7))
        if torch.cuda.is_available():
            _scalars.clear()
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            results = []
            for stream in streams:
                with torch.cuda.stream(stream):
                    x = torch.tensor([2.0], dtype=torch.float64, device="cuda")
                    results.append(_over(x, 0.7))
            for stream in streams:
                stream.synchronize()
            for result in results:
                torch.testing.assert_close(result, torch.full_like(result, 2 / 0.7))
            assert len(_scalars) == 2

    def test_invalid_float_class_dimension(self):
        with pytest.raises(TypeError):
            # A float dim is invalid on purpose.
            sdca_loss(torch.tensor([[1.0, 2.0]]), torch.tensor([0]), dim=-1.0)  # pyright: ignore[reportArgumentType]

    def test_consolidated_hinge_signatures(self):
        z = torch.tensor([[0.3, -0.4, 0.8, 1.2]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([1])
        for variant, objective in (("alpha", "topk_hinge_alpha"), ("beta", "topk_hinge_beta")):
            for gamma in (0.0, 0.4):
                expected = sdca_loss(z, y, objective, k=2, gamma=gamma)
                actual = TopKHingeLoss(2, variant=variant, gamma=gamma)(z, y)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(
                    topk_hinge_loss(z, y, 2, variant=variant, gamma=gamma), expected
                )
                torch.testing.assert_close(
                    torch.autograd.grad(actual, z, retain_graph=True)[0],
                    torch.autograd.grad(expected, z, retain_graph=True)[0],
                )
        # variant is the only selector; the removed mode= alias is rejected.
        with pytest.raises(TypeError):
            TopKHingeLoss(2, mode="alpha")  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError):
            topk_hinge_loss(z, y, 2, mode="beta")  # pyright: ignore[reportCallIssue]
        with pytest.raises(ValueError, match="variant"):
            TopKHingeLoss(2, variant="invalid")

    def test_drop_in_modules_functions_and_aliases(self):
        torch.manual_seed(42)
        z = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0, 2, 1, 3, 0])
        mask = torch.tensor(
            [[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 1, 0]],
            dtype=torch.bool,
        )
        # Names only: no aliases, and smoothing is gamma.
        for bad in ("ce", "softmax", "svm", "topk_hinge_beta_smooth"):
            with pytest.raises(ValueError, match="Unknown loss"):
                sdca_loss(z, y, bad)
        torch.testing.assert_close(
            sdca_loss(z, y, "multiclass_hinge", k=2), sdca_loss(z, y, "topk_hinge_beta", k=1)
        )
        # Drop-in CrossEntropyLoss and cross_entropy.
        torch.testing.assert_close(CrossEntropyLoss()(z, y), sdca_loss(z, y, "cross_entropy"))
        torch.testing.assert_close(cross_entropy(z, y), sdca_loss(z, y, "cross_entropy"))
        assert "reduction='mean'" in repr(CrossEntropyLoss())
        unsupported: list[dict[str, Any]] = [{"label_smoothing": 0.1}]
        for options in unsupported:
            with pytest.raises(ValueError, match="not supported"):
                CrossEntropyLoss(**options)
            with pytest.raises(ValueError, match="not supported"):
                cross_entropy(z, y, **options)
        removed: list[dict[str, Any]] = [{"size_average": True}, {"reduce": False}]
        for options in removed:
            with pytest.raises(TypeError):
                CrossEntropyLoss(**options)
            with pytest.raises(TypeError):
                cross_entropy(z, y, **options)
        # Top-k cross-entropy, hinge, multilabel, proximal, and distillation helpers.
        torch.testing.assert_close(
            TopKCrossEntropyLoss(2)(z, y), sdca_loss(z, y, "topk_cross_entropy", k=2)
        )
        torch.testing.assert_close(
            topk_cross_entropy(z, y, 2), sdca_loss(z, y, "topk_cross_entropy", k=2)
        )
        torch.testing.assert_close(MulticlassHingeLoss()(z, y), sdca_loss(z, y, "multiclass_hinge"))
        torch.testing.assert_close(
            MulticlassHingeLoss(0.5)(z, y), sdca_loss(z, y, "topk_hinge_alpha", k=1, gamma=0.5)
        )
        torch.testing.assert_close(multiclass_hinge_loss(z, y, 0.5), MulticlassHingeLoss(0.5)(z, y))
        for variant in ("alpha", "beta"):
            base = f"topk_hinge_{variant}"
            for gamma in (0.0, 0.4):
                expected = sdca_loss(z, y, base, k=2, gamma=gamma)
                for actual in (
                    TopKHingeLoss(2, variant=variant, gamma=gamma)(z, y),
                    topk_hinge_loss(z, y, 2, variant=variant, gamma=gamma),
                    (TopKHingeAlphaLoss if variant == "alpha" else TopKHingeBetaLoss)(
                        2, gamma=gamma
                    )(z, y),
                    (topk_hinge_alpha_loss if variant == "alpha" else topk_hinge_beta_loss)(
                        z, y, 2, gamma=gamma
                    ),
                ):
                    torch.testing.assert_close(actual, expected)
        # The default top-k hinge is beta, nonsmooth.
        default_module, default_function = TopKHingeLoss, topk_hinge_loss
        torch.testing.assert_close(default_module(2)(z, y), sdca_loss(z, y, "topk_hinge_beta", k=2))
        torch.testing.assert_close(default_function(z, y, 2), TopKHingeBetaLoss(2)(z, y))
        with pytest.raises(ValueError, match="variant"):
            TopKHingeLoss(2, variant="invalid")
        with pytest.raises(ValueError, match="variant"):
            topk_hinge_loss(z, y, 2, variant="invalid")
        torch.testing.assert_close(
            MultilabelCrossEntropyLoss()(z, mask), sdca_loss(z, mask, "multilabel_cross_entropy")
        )
        torch.testing.assert_close(
            multilabel_cross_entropy(z, mask), sdca_loss(z, mask, "multilabel_cross_entropy")
        )
        torch.testing.assert_close(
            MultilabelHingeLoss()(z, mask), sdca_loss(z, mask, "multilabel_hinge")
        )
        torch.testing.assert_close(
            MultilabelHingeLoss(0.6)(z, mask),
            sdca_loss(z, mask, "multilabel_hinge", gamma=0.6),
        )
        torch.testing.assert_close(
            multilabel_hinge_loss(z, mask, 0.6), MultilabelHingeLoss(0.6)(z, mask)
        )
        torch.testing.assert_close(
            ProximalCrossEntropyLoss(0.7)(z, y),
            sdca_loss(z, y, "proximal_cross_entropy", alpha=0.7),
        )
        torch.testing.assert_close(
            proximal_cross_entropy(z, y, 0.7), ProximalCrossEntropyLoss(0.7)(z, y)
        )
        teacher = torch.randn_like(z)
        torch.testing.assert_close(
            DistillationLoss(4.0)(z, teacher),
            DistillationLoss(temperature=4.0)(z, teacher),
        )
        torch.testing.assert_close(
            distillation_loss(z, teacher, 4.0),
            distillation_loss(z, teacher, temperature=4.0),
        )
        torch.testing.assert_close(
            ProximalDistillationLoss(4.0, 0.5)(z, teacher),
            proximal_distillation_loss(z, teacher, temperature=4.0, alpha=0.5),
        )
        for mod in (
            TopKCrossEntropyLoss(2),
            MulticlassHingeLoss(0.5),
            TopKHingeLoss(2, gamma=0.4, variant="alpha"),
            MultilabelCrossEntropyLoss(),
            MultilabelHingeLoss(0.6),
            ProximalCrossEntropyLoss(0.7),
            ProximalDistillationLoss(2.0, 0.5),
        ):
            assert "reduction='mean'" in repr(mod)
        # Lazy top-level exports on libsdca.
        for name in (
            "CrossEntropyLoss",
            "TopKCrossEntropyLoss",
            "MulticlassHingeLoss",
            "TopKHingeLoss",
            "TopKHingeAlphaLoss",
            "TopKHingeBetaLoss",
            "MultilabelCrossEntropyLoss",
            "MultilabelHingeLoss",
            "ProximalCrossEntropyLoss",
            "ProximalDistillationLoss",
            "cross_entropy",
            "topk_cross_entropy",
            "multiclass_hinge_loss",
            "topk_hinge_loss",
            "topk_hinge_alpha_loss",
            "topk_hinge_beta_loss",
            "multilabel_cross_entropy",
            "multilabel_hinge_loss",
            "proximal_cross_entropy",
            "proximal_distillation_loss",
        ):
            assert getattr(libsdca, name) is getattr(libsdca.torch, name)

    def test_proximal_distillation_kkt_limits_and_zero_gradient(self):
        gen = torch.Generator().manual_seed(31)
        z = 3 * torch.randn(6, 9, dtype=torch.float64, generator=gen)
        t = 3 * torch.randn(6, 9, dtype=torch.float64, generator=gen)
        # Equal student and teacher logits give an identically zero gradient.
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            x = z.to(dtype).clone().requires_grad_()
            val = proximal_distillation_loss(x, x.detach(), temperature=4.0, alpha=0.5)
            val.backward()
            torch.testing.assert_close(x.grad, torch.zeros_like(x), rtol=0, atol=0)
        # KKT stationarity: softmax(z/T - alpha * T * grad) - softmax(t/T) == T * grad.
        for temp in (0.5, 1.0, 4.0):
            for alpha in (1e-4, 0.1, 1.0, 10.0):
                x = z.clone().requires_grad_()
                val = proximal_distillation_loss(
                    x, t, temperature=temp, alpha=alpha, reduction="sum"
                )
                val.backward()
                assert x.grad is not None
                u_star = z / temp - alpha * (x.grad * temp)
                torch.testing.assert_close(
                    torch.softmax(u_star, -1) - torch.softmax(t / temp, -1),
                    x.grad * temp,
                    rtol=1e-13,
                    atol=1e-14,
                )
        # As alpha -> 0+, proximal_distillation_loss converges to distillation_loss.
        x1 = z.clone().requires_grad_()
        v1 = proximal_distillation_loss(x1, t, temperature=2.0, alpha=1e-12, reduction="none")
        v1.sum().backward()
        x2 = z.clone().requires_grad_()
        v2 = distillation_loss(x2, t, temperature=2.0, reduction="none")
        v2.sum().backward()
        torch.testing.assert_close(v1.detach(), v2.detach(), rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-11, atol=1e-11)
        # gradcheck and rejected double backward.
        z_small = z[:2, :4].clone().requires_grad_()
        t_small = t[:2, :4]
        assert gradcheck(
            lambda s: proximal_distillation_loss(s, t_small, temperature=1.5, alpha=0.8),
            (z_small,),
        )
        (g,) = torch.autograd.grad(
            proximal_distillation_loss(z_small, t_small), z_small, create_graph=True
        )
        with pytest.raises(RuntimeError):
            torch.autograd.grad(g.sum(), z_small)
        with pytest.raises(ValueError):
            proximal_distillation_loss(z_small, t_small, alpha=0.0)
        with pytest.raises(ValueError):
            ProximalDistillationLoss(alpha=-1.0)


class PublicNamesTests:
    def test_star_import_exports_the_api_only(self):
        import libsdca.torch as module

        names: dict[str, Any] = {}
        exec("from libsdca.torch import *", names)
        assert set(names) - {"__builtins__"} == set(module.__all__)
        # Helpers from private modules stay private unless libsdca exports them too.
        leaked = {
            name
            for name, value in vars(module).items()
            if not name.startswith("_")
            and getattr(value, "__module__", "").startswith("libsdca")
            and name not in module.__all__
        }
        assert leaked <= set(libsdca.__all__)
