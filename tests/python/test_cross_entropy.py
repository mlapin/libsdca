"""PyTorch-compatible loss names, shapes, reductions and numerical tails."""

from types import SimpleNamespace
from unittest import mock

import mpmath as mp
import pytest
import torch
from libsdca import (
    CrossEntropyLoss,
    ProximalCrossEntropyLoss,
    TopKCrossEntropyLoss,
    TopKHingeLoss,
    _cpu_entropy,
    _cross_entropy,
)
from libsdca.torch import cross_entropy, sdca_loss


class CrossEntropyTests:
    def test_tensor_core_propagates_nonfinite_masked_rows(self, subtests: pytest.Subtests):
        # CPU public entry points reject these inputs. Exercise the tensor core
        # directly to check CUDA's asynchronous validation without requiring a GPU.
        for invalid in (float("nan"), float("inf"), -float("inf")):
            for ignored_row in (False, True):
                for reduction in ("none", "sum", "mean"):
                    for graph in (False, True):
                        with subtests.test(
                            invalid=invalid, ignored=ignored_row, reduction=reduction, graph=graph
                        ):
                            z = torch.tensor([[invalid, 0.0], [0.0, 0.0]], requires_grad=True)
                            y = torch.tensor([0, 1])
                            w = torch.tensor([0.0, 1.0], dtype=torch.float64)
                            ignored = torch.tensor([ignored_row, False])
                            value = _cross_entropy._WeightedEntropy.apply(
                                z, y, w, ignored, reduction
                            )
                            gradient = torch.autograd.grad(value.sum(), z, create_graph=graph)[0]
                            if reduction == "none":
                                assert bool(value[0].isnan())
                                assert bool(gradient[0].isnan().all())
                                torch.testing.assert_close(value[1], torch.tensor(2.0).log())
                                torch.testing.assert_close(gradient[1], torch.tensor([0.5, -0.5]))
                            else:
                                assert bool(value.isnan())
                                assert bool(gradient.isnan().all())

    def test_fused_higher_derivative_validates_unused_weights(self, subtests: pytest.Subtests):
        for invalid in (-1.0, float("nan"), float("inf")):
            for reduction in ("none", "sum", "mean"):
                with subtests.test(invalid=invalid, reduction=reduction):
                    z = torch.zeros((3, 3), requires_grad=True)
                    y = torch.tensor([0, 1, -100])
                    weight = torch.tensor([1.0, 1.0, invalid])
                    saved = torch.zeros(17, dtype=torch.float64)
                    saved[3:12:4] = torch.tensor([-3.0, 1.0, 0.0])
                    saved[15:] = torch.tensor([3.0, 1.0])
                    ctx = SimpleNamespace(
                        saved_tensors=(z, y, weight, saved),
                        has_weight=True,
                        ignore_index=-100,
                        reduction=reduction,
                    )
                    upstream = torch.ones(3) if reduction == "none" else torch.tensor(1.0)
                    # A stand-in for the autograd context.
                    backward = _cross_entropy._FusedWeightedEntropy.backward
                    gradient = backward(ctx, upstream)[0]  # pyright: ignore[reportArgumentType]
                    assert gradient.requires_grad
                    assert bool(gradient.isnan().all())

    def test_ignore_index_int64_limits(self):
        z = torch.zeros(2, 2, requires_grad=True)
        for sentinel in (-(2**63), 2**63 - 1):
            y = torch.tensor([sentinel, 1])
            actual = cross_entropy(z, y, ignore_index=sentinel, reduction="none")
            torch.testing.assert_close(actual, torch.tensor([1.0, 2.0]).log())
        for sentinel in (-(2**63) - 1, 2**63):
            with pytest.raises(ValueError, match="int64"):
                cross_entropy(z, torch.tensor([-(2**63), 1]), ignore_index=sentinel)

    def test_shapes_weights_and_ignores_match_torch(self, subtests: pytest.Subtests):
        for shape in [(5,), (4, 5), (2, 5, 3, 2)]:
            target_shape = () if len(shape) == 1 else shape[:1] + shape[2:]
            for weighted in [False, True]:
                for reduction in ["none", "sum", "mean"]:
                    with subtests.test(shape=shape, weighted=weighted, reduction=reduction):
                        torch.manual_seed(2)
                        z = torch.randn(shape, dtype=torch.double, requires_grad=True)
                        y = torch.randint(5, target_shape)
                        if y.numel() > 1:
                            y.reshape(-1)[0] = -100
                        weight = torch.arange(1, 6, dtype=torch.double) if weighted else None
                        got = cross_entropy(z, y, weight, reduction=reduction, backend="torch")
                        ref = torch.nn.functional.cross_entropy(z, y, weight, reduction=reduction)
                        torch.testing.assert_close(got, ref, rtol=1e-14, atol=1e-15)
                        g = torch.autograd.grad(got.sum(), z)[0]
                        r = torch.autograd.grad(ref.sum(), z)[0]
                        torch.testing.assert_close(g, r, rtol=1e-14, atol=1e-15)

    def test_noncontiguous_and_higher_derivatives(self):
        z = torch.randn(2, 3, 4, dtype=torch.double).transpose(1, 2).requires_grad_()
        y = torch.tensor([[0, -100, 1], [2, 0, 1]])
        w = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)

        def function(x: torch.Tensor):
            return cross_entropy(x, y, w, backend="torch")

        assert torch.autograd.gradcheck(function, (z,))
        assert torch.autograd.gradgradcheck(function, (z,))

    def test_empty_and_all_ignored(self):
        for empty in [False, True]:
            for reduction in ["none", "sum", "mean"]:
                z = torch.randn(0 if empty else 3, 5, requires_grad=True)
                y = torch.full((len(z),), -100)
                value = cross_entropy(z, y, reduction=reduction, backend="torch")
                reference = torch.nn.functional.cross_entropy(z, y, reduction=reduction)
                torch.testing.assert_close(value, reference, equal_nan=True)
                value.sum().backward()
                torch.testing.assert_close(z.grad, torch.zeros_like(z))

    def test_zero_weight_mean_matches_pytorch(self):
        z = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
        y = torch.tensor([0, -100])
        weight = torch.zeros(3, dtype=torch.double)
        for reduction in ["none", "sum", "mean"]:
            got = cross_entropy(z, y, weight, reduction=reduction)
            ref = torch.nn.functional.cross_entropy(z, y, weight, reduction=reduction)
            torch.testing.assert_close(got, ref, equal_nan=True)
            actual_grad = torch.autograd.grad(got.sum(), z)[0]
            expected_grad = torch.autograd.grad(ref.sum(), z)[0]
            torch.testing.assert_close(actual_grad, expected_grad, equal_nan=True)

    def test_module_weight_buffer_and_repr(self):
        criterion = CrossEntropyLoss(torch.ones(3), ignore_index=-1).double()
        assert criterion.weight is not None
        assert criterion.weight.dtype == torch.double
        assert "weight" in criterion.state_dict()
        assert "ignore_index=-1" in repr(criterion)
        assert criterion.ignore_index == -1

    def test_confident_target_and_weighted_tails(self):
        for dtype, gap in [(torch.float32, 40.0), (torch.float64, 750.0)]:
            z = torch.tensor([[0.0, -gap], [0.0, -gap]], dtype=dtype, requires_grad=True)
            y = torch.zeros(2, dtype=torch.long)
            weight = torch.tensor([1000.0, 1.0], dtype=dtype)
            loss = cross_entropy(z, y, weight, reduction="sum")
            expected = 2000 * torch.exp(torch.tensor(-gap / 2, dtype=torch.double)).square()
            # Evaluate the product before squaring when individual double tails underflow.
            if dtype == torch.float64:
                half = torch.exp(torch.tensor(-gap / 2, dtype=torch.double))
                expected = half * (2000 * half)
            torch.testing.assert_close(loss.double(), expected, rtol=1e-5, atol=0)
            loss.backward()
            assert z.grad is not None
            assert bool((z.grad[:, 0] < 0).all())
            torch.testing.assert_close(z.grad[:, 0], -z.grad[:, 1], rtol=1e-5, atol=0)

    def test_weighted_mean_does_not_overflow(self):
        z = torch.tensor([[0.0, -1e308], [0.0, -1e308]], dtype=torch.double, requires_grad=True)
        y = torch.ones(2, dtype=torch.long)
        weight = torch.tensor([1e308, 1e308], dtype=torch.double)
        loss = cross_entropy(z, y, weight)
        assert loss.item() == 1e308
        loss.backward()
        torch.testing.assert_close(
            z.grad, torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.double)
        )

    def test_weighted_mean_preserves_small_weighted_large_loss(self):
        z = torch.tensor([[0.0, -1e308], [0.0, -1e308]], dtype=torch.double)
        y = torch.tensor([0, 1])
        weight = torch.tensor([1e308, 1e-320], dtype=torch.double)
        value = cross_entropy(z, y, weight)
        torch.testing.assert_close(value, weight[1], rtol=1e-3, atol=0)

    def test_extreme_weight_and_upstream_scale_remain_factored(self):
        z = torch.zeros(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.tensor([0, 1])
        weight = torch.tensor([1e308, 1e-320], dtype=torch.double)
        (cross_entropy(z, y, weight) * 1e308).backward()
        assert z.grad is not None
        torch.testing.assert_close(z.grad[1, 0], weight[1] / 2, rtol=1e-3, atol=0)
        torch.testing.assert_close(z.grad[1, 1], -weight[1] / 2, rtol=1e-3, atol=0)
        z = torch.tensor([[0.0, -1000.0]], dtype=torch.double, requires_grad=True)
        weight = torch.tensor([1e308, 1.0], dtype=torch.double)
        (cross_entropy(z, torch.tensor([0]), weight, reduction="sum") * 1e308).backward()
        assert z.grad is not None
        with mp.workdps(80):
            tail = mp.exp(-1000)
            expected = float(mp.mpf(1e308) ** 2 * tail / (1 + tail))
        torch.testing.assert_close(z.grad[0, 0], z.new_tensor(-expected), rtol=6e-14, atol=0)
        torch.testing.assert_close(z.grad[0, 1], z.new_tensor(expected), rtol=6e-14, atol=0)

    def test_weighted_mean_keeps_partially_underflowed_terms(self):
        z = torch.tensor([[0.0, -1e308, -1e308]] * 3, dtype=torch.double, requires_grad=True)
        y = torch.tensor([0, 1, 2])
        weight = torch.tensor([1e308, 1e-16, 5e-16], dtype=torch.double)
        value = cross_entropy(z, y, weight)
        torch.testing.assert_close(value, weight[1:].sum(), rtol=2e-13, atol=0)
        (value * 1e308).backward()
        assert z.grad is not None
        torch.testing.assert_close(z.grad[2, 0], weight[2], rtol=2e-13, atol=0)
        torch.testing.assert_close(z.grad[2, 2], -weight[2], rtol=2e-13, atol=0)

    def test_float64_tails_near_the_subnormal_limit(self, subtests: pytest.Subtests):
        # Rows [0, -m, ..., -m] of label 0: log1p((C - 1) e^-m), rounded once into
        # subnormals; a log-domain tail would round its argument near -708 first.
        quantum, eps = 2.0**-1074, torch.finfo(torch.float64).eps
        for classes in (40, 1000):
            for gaps in ([715.5], [700.0, 712.5, 716.0], [709.75, 745.0, 751.0]):
                z = torch.tensor(gaps, dtype=torch.double)[:, None].expand(-1, classes).neg()
                z = z.contiguous()
                z[:, 0] = 0
                y = torch.zeros(len(gaps), dtype=torch.long)
                for scale in (None, 3.5):
                    weight = None if scale is None else torch.full((classes,), scale).double()
                    with mp.workdps(60):
                        rows = [
                            (scale or 1) * mp.log1p((classes - 1) * mp.exp(-mp.mpf(m)))
                            for m in gaps
                        ]
                        exact = {
                            "none": rows,
                            "sum": [mp.fsum(rows)],
                            "mean": [mp.fsum(rows) / ((scale or 1) * len(rows))],
                        }
                    for reduction, expected in exact.items():
                        for backend in ("auto", "native", "torch"):
                            with subtests.test(
                                classes=classes,
                                gaps=gaps,
                                weight=scale,
                                reduction=reduction,
                                backend=backend,
                            ):
                                value = cross_entropy(
                                    z, y, weight, reduction=reduction, backend=backend
                                )
                                rows = value.reshape(-1).tolist()
                                for got, ref in zip(rows, expected, strict=True):
                                    error = abs(mp.mpf(got) - ref)
                                    assert error <= quantum + 64 * eps * abs(ref), (got, ref)

    def test_float64_extension_rows_match_native(self, subtests: pytest.Subtests):
        # The extension's float64 rows (at most 32 classes) keep the native library's
        # statistics and fallbacks; their exponentials and logarithms are within a few ulps.
        if not _cpu_entropy.enabled():
            pytest.skip("the CPU extension is unavailable")
        gen = torch.Generator().manual_seed(5)
        for classes in (2, 3, 10, 32):
            for spread in (1.0, 30.0, 400.0):
                z = spread * torch.randn(37, classes, generator=gen, dtype=torch.double)
                y = torch.randint(0, classes, (37,), generator=gen)
                y[5] = -100
                w = 0.5 + torch.rand(classes, generator=gen, dtype=torch.double)
                for weight in (None, w):
                    for reduction in ("none", "sum", "mean"):
                        weighted = weight is not None
                        with subtests.test(
                            classes=classes, spread=spread, weighted=weighted, reduction=reduction
                        ):
                            results = []
                            for backend in ("auto", "native"):
                                x = z.clone().requires_grad_()
                                value = cross_entropy(
                                    x, y, weight, reduction=reduction, backend=backend
                                )
                                (value.sum() * 3.0).backward()
                                results.append((value.detach(), x.grad))
                            (value, grad), (expected, expected_grad) = results
                            torch.testing.assert_close(value, expected, rtol=1e-13, atol=0)
                            torch.testing.assert_close(grad, expected_grad, rtol=1e-13, atol=1e-300)

    def test_half_loss_is_float32(self):
        for dtype in [torch.float16, torch.bfloat16]:
            z = torch.randn(3, 5, dtype=dtype, requires_grad=True)
            value = cross_entropy(z, torch.tensor([0, 1, -100]), backend="torch")
            assert value.dtype == torch.float32
            value.backward()
            assert z.grad is not None
            assert z.grad.dtype == dtype

    def test_half_upstream_scale_precedes_gradient_rounding(self):
        for dtype in [torch.float16, torch.bfloat16]:
            z = torch.tensor([[0.0, -20.0]], dtype=dtype, requires_grad=True)
            weight = torch.tensor([2.0, 1.0], dtype=dtype)
            value = cross_entropy(z, torch.tensor([0]), weight, reduction="sum")
            (value * 1e4).backward()
            assert z.grad is not None
            tail = torch.exp(torch.tensor(-20.0, dtype=torch.double))
            expected = (2e4 * tail / (1 + tail)).to(dtype)
            torch.testing.assert_close(z.grad[0, 0], -expected, rtol=0, atol=0)
            torch.testing.assert_close(z.grad[0, 1], expected, rtol=0, atol=0)

    def test_autocast_accepts_float32_weights_for_half_logits(self):
        z = torch.randn(4, 3, dtype=torch.bfloat16, requires_grad=True)
        y = torch.tensor([0, 1, -100, 2])
        w = torch.tensor([1.0, 2.0, 3.0])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            value = cross_entropy(z, y, w)
            reference = torch.nn.functional.cross_entropy(z, y, w)
        torch.testing.assert_close(value, reference)
        actual_grad = torch.autograd.grad(value, z)[0]
        expected_grad = torch.autograd.grad(reference, z)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_unsupported_and_invalid_arguments(self, subtests: pytest.Subtests):
        z, y = torch.zeros(2, 3), torch.tensor([0, 1])
        for options in [
            dict(label_smoothing=0.1),
            dict(ignore_index=True),
            dict(reduction="bad"),
            dict(weight=torch.ones(2)),
            dict(weight=torch.tensor([1.0, -1.0, 1.0])),
            dict(weight=torch.ones(3, requires_grad=True)),
        ]:
            with subtests.test(options=options), pytest.raises(ValueError):
                cross_entropy(z, y, **options)
        with pytest.raises(ValueError, match="probability targets"):
            cross_entropy(z, torch.ones(2, 3) / 3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_dtypes_weights_ignores_and_reductions(self, subtests: pytest.Subtests):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            for weighted in [False, True]:
                for ignored in [False, True]:
                    for reduction in ["none", "sum", "mean"]:
                        with subtests.test(
                            dtype=dtype, weighted=weighted, ignored=ignored, reduction=reduction
                        ):
                            z = torch.randn(4, 5, dtype=dtype, device="cuda", requires_grad=True)
                            y = torch.tensor([0, 1, 2, -100 if ignored else 3], device="cuda")
                            w = torch.arange(1, 6, dtype=dtype, device="cuda") if weighted else None
                            ref_z = z.detach().double().requires_grad_()
                            got = cross_entropy(z, y, w, reduction=reduction)
                            ref = torch.nn.functional.cross_entropy(
                                ref_z, y, None if w is None else w.double(), reduction=reduction
                            )
                            tol = 64 * torch.finfo(got.dtype).eps
                            torch.testing.assert_close(got, ref.to(got.dtype), rtol=tol, atol=tol)
                            grad = torch.autograd.grad(got.sum(), z)[0]
                            ref_grad = torch.autograd.grad(ref.sum(), ref_z)[0].to(dtype)
                            tol = (
                                4 if dtype in (torch.float16, torch.bfloat16) else 64
                            ) * torch.finfo(dtype).eps
                            torch.testing.assert_close(grad, ref_grad, rtol=tol, atol=tol)

    def test_ignore_sentinels_preserve_validation_and_gradients(self, subtests: pytest.Subtests):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for ignore_index in [-100, 7, 1]:
                for reduction in ["none", "sum", "mean"]:
                    with subtests.test(
                        device=device, ignore_index=ignore_index, reduction=reduction
                    ):
                        z = torch.randn(3, 4, dtype=torch.double, device=device, requires_grad=True)
                        y = torch.tensor([0, ignore_index, 2], device=device)
                        got = cross_entropy(
                            z, y, ignore_index=ignore_index, reduction=reduction, backend="torch"
                        )
                        ref = torch.nn.functional.cross_entropy(
                            z, y, ignore_index=ignore_index, reduction=reduction
                        )
                        torch.testing.assert_close(got, ref)
                        torch.testing.assert_close(
                            torch.autograd.grad(got.sum(), z)[0],
                            torch.autograd.grad(ref.sum(), z)[0],
                        )
                # Bad labels or nonfinite logits raise on CPU and give NaN on CUDA.
                z = torch.zeros(3, 4, device=device)
                y = torch.tensor([0, ignore_index, 99], device=device)
                if device == "cpu":
                    with pytest.raises(ValueError, match="range"):
                        cross_entropy(z, y, ignore_index=ignore_index, backend="torch")
                else:
                    assert bool(
                        cross_entropy(z, y, ignore_index=ignore_index, backend="torch")
                        .isnan()
                        .all()
                    )
                y[-1] = 2
                for invalid in [float("nan"), float("inf")]:
                    z[0, 0] = invalid
                    if device == "cpu":
                        with pytest.raises(ValueError, match="finite"):
                            cross_entropy(z, y, ignore_index=ignore_index, backend="torch")
                    else:
                        assert bool(
                            cross_entropy(z, y, ignore_index=ignore_index, backend="torch")
                            .isnan()
                            .all()
                        )
                z[0, 0] = 0
                cross_entropy(z, y, ignore_index=ignore_index, backend="torch")

    def test_unrelated_errors_propagate(self):
        z, y = torch.zeros(2, 3), torch.tensor([0, 1])
        for backend in ("auto", "native"):
            for error in [ValueError("original failure"), RuntimeError("original failure")]:
                with mock.patch("libsdca._cpu.loss", side_effect=error):
                    with pytest.raises(type(error)) as caught:
                        cross_entropy(z, y, backend=backend)
                    assert caught.value is error

    def test_named_losses(self, subtests: pytest.Subtests):
        z = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
        y = torch.tensor([0, 1, 2])
        cases = [
            (TopKCrossEntropyLoss(2, backend="torch"), "topk_cross_entropy", dict(k=2)),
            (TopKHingeLoss(2, backend="torch", variant="alpha"), "topk_hinge_alpha", dict(k=2)),
            (
                TopKHingeLoss(2, variant="beta", gamma=0.3, backend="torch"),
                "topk_hinge_beta",
                dict(k=2, gamma=0.3),
            ),
            (
                ProximalCrossEntropyLoss(0.3, backend="torch"),
                "proximal_cross_entropy",
                dict(alpha=0.3),
            ),
        ]
        for module, objective, options in cases:
            with subtests.test(objective=objective):
                torch.testing.assert_close(
                    module(z, y), sdca_loss(z, y, objective, backend="torch", **options)
                )


def test_class_weights_stay_alive_across_threads():
    """The native route's float64 class weights are cached once; a thread holds its copy over
    the native call while others evict it with their own weights."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    from libsdca import _cpu
    from torch.nn import functional as F

    first, second = torch.rand(7) + 0.5, torch.rand(7) + 0.5
    held = _cpu._weights(first)
    _cpu._weights(second)
    assert held is not None and torch.equal(held, first.double())

    torch.manual_seed(5)
    logits = torch.randn(64, 7)
    target = torch.randint(7, (64,))
    # Float32 weights: each call converts them, so the cache holds the only float64 copy.
    weights = [torch.rand(7) + 0.5 for _ in range(8)]

    def run(index: int) -> bool:
        weight = weights[index % len(weights)]
        expected = F.cross_entropy(logits.double(), target, weight=weight.double()).float()
        return all(
            torch.allclose(cross_entropy(logits, target, weight), expected, rtol=1e-6)
            for _ in range(50)
        )

    with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": "off"}):
        with ThreadPoolExecutor(8) as pool:
            assert all(pool.map(run, range(32)))
