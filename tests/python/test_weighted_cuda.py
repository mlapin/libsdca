"""Bounded weighted CUDA entropy against represented-input high-precision references."""

import concurrent.futures
import contextlib
import ctypes
import os
from unittest import mock

import interpreter
import mpmath as mp
import pytest
import torch
from libsdca import _cross_entropy, _cuda_launch
from libsdca.torch import cross_entropy


def reference(
    z: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    reduction: str,
    upstream: torch.Tensor,
    ignore: int = -100,
):
    """Independent competitor-normalized entropy and analytic gradients in 100 digits."""
    z, y, weight = z.double().cpu(), y.cpu(), weight.double().cpu()
    u = upstream.double().cpu().reshape(-1)
    with mp.workdps(100):
        factors = [
            mp.mpf(0) if int(label) == ignore else mp.mpf(float(weight[label])) for label in y
        ]
        divisor = sum(factors) if reduction == "mean" else mp.mpf(1)
        values, gradients = [], []
        for i, row in enumerate(z.tolist()):
            label = int(y[i])
            if label == ignore:
                values.append(mp.mpf(0))
                gradients.append([mp.mpf(0)] * len(row))
                continue
            scores = [mp.mpf(v) for v in row]
            top = max(v for j, v in enumerate(scores) if j != label)
            exponentials = [
                mp.exp(v - top) if j != label else mp.mpf(0) for j, v in enumerate(scores)
            ]
            total = sum(exponentials)
            relative = top - scores[label] + mp.log(total)
            value = (
                relative + mp.log1p(mp.exp(-relative))
                if relative > 0
                else mp.log1p(mp.exp(relative))
            )
            mass = 1 / (1 + mp.exp(-relative))
            scale = factors[i] / divisor * mp.mpf(float(u[i] if reduction == "none" else u[0]))
            gradient = [e / total * mass * scale for e in exponentials]
            gradient[label] = -mass * scale
            values.append(value * factors[i])
            gradients.append(gradient)
        output = [float(v) for v in values] if reduction == "none" else float(sum(values) / divisor)
        return torch.tensor(output, dtype=torch.float64), torch.tensor(
            [[float(v) for v in row] for row in gradients], dtype=torch.float64
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class WeightedCudaTests:
    def test_invalid_inputs_with_higher_derivatives_on_both_launchers(
        self, subtests: pytest.Subtests
    ):
        from libsdca.torch import sdca_loss

        try:
            for setting in ("0", "1"):
                with mock.patch.dict(os.environ, {_cuda_launch.ENV: setting}):
                    _cuda_launch.status.cache_clear()
                    if setting == "1":
                        assert _cuda_launch.host() is not None
                    for invalid in (-1, 3, -(2**63), 2**63 - 1):
                        z = torch.zeros((2, 3), device="cuda", requires_grad=True)
                        y = torch.tensor([invalid, 1], device="cuda")
                        for reduction in ("none", "sum", "mean"):
                            with subtests.test(
                                launcher=setting, target=invalid, reduction=reduction
                            ):
                                value = sdca_loss(z, y, reduction=reduction)
                                grad = torch.autograd.grad(value.sum(), z, create_graph=True)[0]
                                assert bool(grad[0].isnan().all())
                                assert bool(grad[1].isfinite().all())
                                # A subsequent CUDA operation also exposes device assertions.
                                assert bool((z + 1).isfinite().all())
                    for problem in (
                        "target",
                        "unused_weight",
                        "ignored_logit",
                        "zero_weight_logit",
                    ):
                        for reduction in ("none", "sum", "mean"):
                            z = torch.zeros((3, 3), device="cuda")
                            y = torch.tensor([0, 1, -100], device="cuda")
                            w = torch.ones(3, device="cuda")
                            row = 0
                            if problem == "target":
                                y[0] = 3
                            elif problem == "unused_weight":
                                w[2] = -1
                            elif problem == "ignored_logit":
                                row = 2
                                z[row, 0] = torch.nan
                            else:
                                w[0] = 0
                                z[0, 0] = torch.inf
                            z.requires_grad_()
                            for backend in ("auto", "torch"):
                                with subtests.test(
                                    launcher=setting,
                                    problem=problem,
                                    reduction=reduction,
                                    backend=backend,
                                ):
                                    value = cross_entropy(
                                        z, y, w, reduction=reduction, backend=backend
                                    )
                                    ordinary = torch.autograd.grad(
                                        value.sum(), z, retain_graph=True
                                    )[0]
                                    higher = torch.autograd.grad(value.sum(), z, create_graph=True)[
                                        0
                                    ]
                                    torch.testing.assert_close(higher, ordinary, equal_nan=True)
                                    if reduction == "none" and problem != "unused_weight":
                                        assert bool(value[row].isnan())
                                        assert bool(higher[row].isnan().all())
                                    else:
                                        assert bool(value.isnan().any())
                                        assert bool(higher.isnan().all())
        finally:
            _cuda_launch.status.cache_clear()

    def assert_reference(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        reduction: str,
        upstream: torch.Tensor,
        ignore: int = -100,
    ) -> None:
        expected, gradient = reference(z, y, w, reduction, upstream, ignore)
        x = z.cuda().detach().requires_grad_()
        value = cross_entropy(x, y.cuda(), w.cuda(), reduction=reduction, ignore_index=ignore)
        actual = torch.autograd.grad(value, x, upstream.cuda().to(value.dtype))[0]
        eps = torch.finfo(z.dtype).eps
        quantum = torch.finfo(z.dtype).tiny * eps
        torch.testing.assert_close(
            value.cpu().double(),
            expected,
            rtol=8 * torch.finfo(torch.float32).eps,
            atol=torch.finfo(torch.float32).tiny * torch.finfo(torch.float32).eps,
        )
        torch.testing.assert_close(
            actual.cpu().double(), gradient.to(z.dtype).double(), rtol=8 * eps, atol=quantum
        )

    def test_weighted_tail_and_upstream_rescue(self, subtests: pytest.Subtests):
        for dtype, gap, large in [
            (torch.float16, 28.0, 60000.0),
            (torch.bfloat16, 170.0, 1e30),
            (torch.float32, 170.0, 1e30),
        ]:
            z = torch.tensor([[0.0, -gap], [0.0, -gap], [0.0, -gap]], dtype=dtype)
            y = torch.tensor([0, 0, -100])
            w = torch.tensor([large, 1.0], dtype=dtype)
            for reduction in ("none", "sum", "mean"):
                for scale in (1.0, 1e4, 1e20, 0.0, -3.0):
                    upstream = torch.full(
                        (3,) if reduction == "none" else (), scale, dtype=torch.float32
                    )
                    with subtests.test(dtype=dtype, reduction=reduction, scale=scale):
                        self.assert_reference(z, y, w, reduction, upstream)

    def test_mean_scales_before_rounding_and_extreme_scores(self, subtests: pytest.Subtests):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            limit = torch.finfo(dtype).max
            quantum = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
            z = torch.tensor([[-limit, limit], [-limit, limit], [limit, -limit]], dtype=dtype)
            y = torch.tensor([0, 1, -100])
            w = torch.tensor([4 * quantum, limit], dtype=dtype)
            for reduction in ("none", "sum", "mean"):
                for scale in (1.0, torch.finfo(torch.float32).max):
                    upstream = torch.full(
                        (3,) if reduction == "none" else (), scale, dtype=torch.float32
                    )
                    with subtests.test(dtype=dtype, reduction=reduction, scale=scale):
                        self.assert_reference(z, y, w, reduction, upstream)

    def test_aggregate_tail_and_resource_boundary(self, subtests: pytest.Subtests):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for classes in (4095, 4096, 4097, 8192):
                z = torch.full((2, classes), -110.0, dtype=dtype)
                z[:, 0] = 0
                y = torch.zeros(2, dtype=torch.int64)
                w = torch.ones(classes, dtype=dtype)
                with subtests.test(dtype=dtype, classes=classes):
                    self.assert_reference(z, y, w, "mean", torch.tensor(1e20))
            # Rows beyond CLASSES fall back to the tensor implementation.
            x = torch.zeros((2, 4097), device="cuda", dtype=dtype)
            w = torch.ones(4097, device="cuda", dtype=dtype)
            with (
                mock.patch.object(_cross_entropy._weighted_kernel(), "CLASSES", 4096),
                mock.patch.object(
                    _cross_entropy._FusedWeightedEntropy,
                    "apply",
                    side_effect=AssertionError("wide row entered kernel"),
                ),
            ):
                cross_entropy(x, torch.tensor([0, 1], device="cuda"), w)

    @pytest.mark.parametrize("precision", [None, "float64"])
    def test_reduction_partition_boundary(self, subtests: pytest.Subtests, precision: str | None):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for rows in (4095, 4096, 4097, 16383, 16384, 16385):
                torch.manual_seed(rows)
                x = torch.randn(rows, 3, device="cuda", dtype=dtype, requires_grad=True)
                y = torch.arange(rows, device="cuda") % 3
                y[::10] = -100
                w = torch.tensor([0.125, 1.0, 7.0], device="cuda", dtype=dtype)
                for reduction in ("none", "sum", "mean"):
                    with subtests.test(dtype=dtype, rows=rows, reduction=reduction):
                        value = cross_entropy(x, y, w, reduction=reduction, precision=precision)
                        rx = x.detach().double().requires_grad_()
                        ref = torch.nn.functional.cross_entropy(
                            rx, y, w.double(), reduction=reduction
                        )
                        actual = torch.autograd.grad(value.sum(), x)[0]
                        expected = torch.autograd.grad(ref.sum(), rx)[0]
                        torch.testing.assert_close(value.double(), ref, rtol=5e-7, atol=0)
                        # Default float32 exponentials have the same 8-eps budget as
                        # assert_reference above; full precision keeps the tighter check.
                        eps = torch.finfo(dtype).eps
                        tolerance = max(4 * eps, 8 * torch.finfo(torch.float32).eps)
                        torch.testing.assert_close(
                            actual.double(),
                            expected,
                            rtol=tolerance if precision is None else 4 * eps,
                            atol=torch.finfo(dtype).tiny * torch.finfo(dtype).eps,
                        )

    def test_strided_spatial_amp_and_higher_derivatives(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            z = torch.randn(2, 3, 5, 4, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
            y = torch.randint(5, (2, 4, 3), device="cuda").transpose(1, 2)
            y[0, 0, 0] = -100
            w = torch.linspace(0.25, 2, 5, device="cuda")
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                value = cross_entropy(z, y, w)
            assert value.dtype == torch.float32
            first_order = torch.autograd.grad(value, z, retain_graph=True)[0]
            gradient = torch.autograd.grad(value, z, create_graph=True)[0]
            direction = torch.randn_like(z)
            second = torch.autograd.grad((gradient * direction).sum(), z)[0]
            refz = z.detach().double().requires_grad_()
            ref = torch.nn.functional.cross_entropy(refz, y, w.double())
            refg = torch.autograd.grad(ref, refz, create_graph=True)[0]
            refh = torch.autograd.grad((refg * direction.double()).sum(), refz)[0]
            torch.testing.assert_close(value.double(), ref, rtol=5e-7, atol=0)
            torch.testing.assert_close(
                first_order.double(),
                refg,
                rtol=4 * torch.finfo(dtype).eps,
                atol=torch.finfo(dtype).tiny * torch.finfo(dtype).eps,
            )
            torch.testing.assert_close(
                gradient.double(),
                refg,
                rtol=4 * torch.finfo(dtype).eps,
                atol=torch.finfo(dtype).tiny * torch.finfo(dtype).eps,
            )
            torch.testing.assert_close(
                second.double(),
                refh,
                rtol=8 * torch.finfo(dtype).eps,
                atol=8 * torch.finfo(dtype).eps * float(refh.abs().max()),
            )

    def test_zero_weight_all_ignored_nan_and_validation(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for target in ([0, -100, 1], [-100, -100, -100]):
                x = torch.randn(3, 4, device="cuda", dtype=dtype, requires_grad=True)
                y = torch.tensor(target, device="cuda")
                w = torch.zeros(4, device="cuda", dtype=dtype)
                for reduction in ("none", "sum", "mean"):
                    value = cross_entropy(x, y, w, reduction=reduction)
                    rx = x.detach().double().requires_grad_()
                    ref = torch.nn.functional.cross_entropy(rx, y, w.double(), reduction=reduction)
                    torch.testing.assert_close(value.double(), ref, equal_nan=True)
                    actual = torch.autograd.grad(value.sum(), x)[0]
                    expected = torch.autograd.grad(ref.sum(), rx)[0]
                    torch.testing.assert_close(actual.double(), expected, equal_nan=True)
                with torch.no_grad():
                    x[1, 0] = torch.nan
                assert bool(cross_entropy(x, y, w).isnan().all())
            x = torch.zeros((2, 4), device="cuda", dtype=dtype)
            assert bool(
                cross_entropy(
                    x,
                    torch.tensor([-100, 9], device="cuda"),
                    torch.ones(4, device="cuda", dtype=dtype),
                )
                .isnan()
                .all()
            )

    def test_explicit_fallbacks_and_ordinary_dispatch_preserved(self):
        for dtype, backend in ((torch.float64, "torch"), (torch.float32, "torch")):
            z = torch.randn(3, 5, device="cuda", dtype=dtype, requires_grad=True)
            w = torch.ones(5, device="cuda", dtype=dtype)
            y = torch.tensor([0, 1, -100], device="cuda")
            with mock.patch.object(
                _cross_entropy._FusedWeightedEntropy,
                "apply",
                side_effect=AssertionError("explicit fallback entered kernel"),
            ):
                cross_entropy(z, y, w, backend=backend).backward()
        z = torch.randn(3, 5, device="cuda", requires_grad=True)
        y = torch.tensor([0, 1, 2], device="cuda")
        with mock.patch(
            "libsdca.torch.sdca_loss",
            side_effect=AssertionError("CUDA cross_entropy must run in one pass"),
        ):
            cross_entropy(z, y).backward()
        with pytest.raises(ValueError):
            cross_entropy(z, y, torch.ones(5, device="cuda"), backend="native")

    def test_nonfinite_upstream_keeps_ignored_and_zero_weights_zero(
        self, subtests: pytest.Subtests
    ):
        # Finite logits have strictly positive exact probabilities even when
        # exp(-2000) underflows internally. Infinite upstream therefore gives
        # signed infinite active gradients; masked rows must still be zero.
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for reduction in ("none", "sum", "mean"):
                for upstream in (float("inf"), -float("inf"), float("nan")):
                    with subtests.test(dtype=dtype, reduction=reduction, upstream=upstream):
                        z = torch.tensor(
                            [[0.0, -2000.0]] * 3, device="cuda", dtype=dtype, requires_grad=True
                        )
                        y = torch.tensor([0, -100, 1], device="cuda")
                        w = torch.tensor([1.0, 0.0], device="cuda", dtype=dtype)
                        loss = cross_entropy(z, y, w, reduction=reduction)
                        u = torch.full_like(loss, upstream)
                        gradient = torch.autograd.grad(loss, z, u)[0]
                        if upstream == upstream:
                            expected = torch.tensor(
                                [-upstream, upstream], device="cuda", dtype=dtype
                            )
                            torch.testing.assert_close(gradient[0], expected)
                        else:
                            assert bool(gradient[0].isnan().all())
                        torch.testing.assert_close(gradient[1:], torch.zeros_like(gradient[1:]))

    def test_custom_ignore_strided_weights_and_upstream(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            z = torch.tensor([[0, -40, 1], [-1, 2, 0], [0, -40, -40], [2, 1, 0]], dtype=dtype)
            y = torch.tensor([0, 1, 2, 0])
            w = torch.arange(1, 7, dtype=dtype)[::2]
            upstream = torch.tensor([0.5, 9.0, -2.0, 9.0, 1e20, 9.0, 0.0, 9.0])[::2]
            self.assert_reference(z, y, w, "none", upstream, ignore=1)
        for ignore in (-(2**63), 2**63 - 1):
            self.assert_reference(
                torch.tensor([[0.0, -40.0], [1.0, 2.0]]),
                torch.tensor([0, ignore]),
                torch.tensor([1.0, 3.0]),
                "mean",
                torch.tensor(1.0),
                ignore=ignore,
            )

    def test_weight_validation_covers_unused_and_ignored_classes(self):
        for targets in ([0, -100], [-100, -100], []):
            z = torch.zeros((len(targets), 4), device="cuda")
            y = torch.tensor(targets, dtype=torch.int64, device="cuda")
            for invalid in (-1.0, float("nan"), float("inf")):
                w = torch.ones(4, device="cuda")
                w[-1] = invalid
                assert bool(cross_entropy(z, y, w).isnan().all())
            if len(targets):
                z[0, 0] = torch.nan
                y[0] = 99
                assert bool(cross_entropy(z, y, w).isnan().all())

    def test_failed_ordinary_dispatch_still_validates_every_input(self):
        z = torch.zeros((3, 5), device="cuda", requires_grad=True)
        for targets in ([0, 1, 99], [0, -100, 99], [0, 1, -2]):
            y = torch.tensor(targets, device="cuda")
            loss = cross_entropy(z, y)
            assert bool(loss.isnan().all())
            (grad,) = torch.autograd.grad(loss, z)
            assert bool(grad.isnan().any())
        with torch.no_grad():
            z[0, 0] = torch.nan
        for targets in ([0, 1, 2], [-100, 1, 2], [-100, 1, 99]):
            y = torch.tensor(targets, device="cuda")
            assert bool(cross_entropy(z, y).isnan().all())

    def test_cuda_graph_capture_succeeds_without_host_sync(self):
        from libsdca.torch import sdca_loss

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            z = torch.randn(16, 10, device="cuda", requires_grad=True)
            y = torch.randint(0, 10, (16,), device="cuda")
            w = torch.ones(10, device="cuda")
            # Warmup compilation before graph capture.
            (sdca_loss(z, y) + cross_entropy(z, y, w)).backward()
            z.grad = None
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                loss = sdca_loss(z, y) + cross_entropy(z, y, w)
                loss.backward()
        graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        assert bool(torch.isfinite(loss).all())
        assert z.grad is not None
        assert bool(torch.isfinite(z.grad).all())

    def test_zero_weight_and_all_ignored_higher_derivatives(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for all_ignored in (False, True):
                z = torch.randn((3, 4), device="cuda", dtype=dtype, requires_grad=True)
                y = torch.tensor([-100] * 3 if all_ignored else [0, 1, -100], device="cuda")
                w = torch.tensor([0.0, 2.0, 1.0, 1.0], device="cuda", dtype=dtype)
                value = cross_entropy(z, y, w)
                grad = torch.autograd.grad(value, z, create_graph=True)[0]
                direction = torch.randn_like(z)
                second = torch.autograd.grad((grad * direction).sum(), z)[0]
                refz = z.detach().double().requires_grad_()
                ref = torch.nn.functional.cross_entropy(refz, y, w.double())
                refg = torch.autograd.grad(ref, refz, create_graph=True)[0]
                refh = torch.autograd.grad((refg * direction.double()).sum(), refz)[0]
                torch.testing.assert_close(
                    grad.double(), refg, rtol=4 * torch.finfo(dtype).eps, atol=0
                )
                torch.testing.assert_close(
                    second.double(),
                    refh,
                    rtol=8 * torch.finfo(dtype).eps,
                    atol=8 * torch.finfo(dtype).eps * float(refh.abs().max()),
                )

    def test_launcher_matches_triton_with_misaligned_contiguous_inputs(self):
        # Cached kernels assume 16-byte alignment; contiguous offset views do
        # not necessarily have it. Exercise every pointer, both reducers and
        # AMP float32 weights with the launcher enabled and disabled.
        try:
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                for n in (17, 4097):
                    c = 5
                    z = torch.randn(n * c + 1, dtype=dtype, device="cuda")[1:].view(n, c)
                    y = torch.arange(n + 1, device="cuda")[1:].remainder_(c)
                    y[::7] = -100
                    for weight_dtype in (None, dtype, torch.float32):
                        w = (
                            None
                            if weight_dtype is None
                            else torch.arange(c + 1, device="cuda", dtype=weight_dtype)[1:]
                        )
                        for reduction in ("none", "sum", "mean"):
                            results = []
                            for setting in ("0", "1"):
                                with mock.patch.dict(os.environ, {_cuda_launch.ENV: setting}):
                                    _cuda_launch.status.cache_clear()
                                    if setting == "1":
                                        assert _cuda_launch.host() is not None
                                    x = z.detach().requires_grad_()
                                    amp = (
                                        torch.autocast("cuda", dtype=dtype)
                                        if dtype != torch.float32 and weight_dtype == torch.float32
                                        else contextlib.nullcontext()
                                    )
                                    with amp:
                                        value = cross_entropy(x, y, w, reduction=reduction)
                                    u = torch.ones(value.numel() + 1, device="cuda")[1:]
                                    gradient = torch.autograd.grad(value, x, u.reshape_as(value))[0]
                                    results.append((value, gradient))
                            torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
        finally:
            _cuda_launch.status.cache_clear()

    def test_stream_worker_context_and_saved_weight_version(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            z = torch.randn(17, 5, device="cuda", requires_grad=True)
            y = torch.arange(17, device="cuda").remainder_(5)
            y[::7] = -100
            w = torch.arange(1, 6, dtype=torch.float32, device="cuda")
            loss = cross_entropy(z, y, w)
            gradient = torch.autograd.grad(loss, z)[0]
        torch.cuda.current_stream().wait_stream(stream)
        expected_z = z.detach().double().requires_grad_()
        expected = torch.nn.functional.cross_entropy(expected_z, y, w.double())
        expected_gradient = torch.autograd.grad(expected, expected_z)[0]
        torch.testing.assert_close(loss.double(), expected, rtol=4e-7, atol=0)
        torch.testing.assert_close(gradient.double(), expected_gradient, rtol=4e-7, atol=0)
        del loss
        # A fresh Python/autograd worker may have no current CUDA driver context.
        loss = cross_entropy(z, y, w)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            worker_gradient = executor.submit(torch.autograd.grad, loss, z).result()[0]
        torch.testing.assert_close(worker_gradient, gradient, rtol=0, atol=0)
        del loss
        loss = cross_entropy(z, y, w)
        w.add_(1)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            loss.backward()

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
    def test_noncurrent_device(self):
        original = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            z = torch.randn(17, 5, device="cuda:1", requires_grad=True)
            y = torch.arange(17, device="cuda:1").remainder_(5)
            w = torch.arange(1, 6, dtype=torch.float32, device="cuda:1")
            loss = cross_entropy(z, y, w)
            gradient = torch.autograd.grad(loss, z)[0]
            expected_z = z.detach().double().requires_grad_()
            expected = torch.nn.functional.cross_entropy(expected_z, y, w.double())
            expected_gradient = torch.autograd.grad(expected, expected_z)[0]
            torch.testing.assert_close(loss.double(), expected, rtol=4e-7, atol=0)
            torch.testing.assert_close(gradient.double(), expected_gradient, rtol=4e-7, atol=0)
            assert torch.cuda.current_device() == 0
            kernel = _cross_entropy._weighted_kernel()
            assert kernel is not None
            host, handles = kernel.launcher(z, w, 1, -100)
            if host is not None:
                driver = ctypes.CDLL("libcuda.so.1")

                def current_context():
                    context = ctypes.c_void_p()
                    assert driver.cuCtxGetCurrent(ctypes.byref(context)) == 0
                    return context.value

                previous = current_context()
                _, stats, context = kernel.forward(z, y, w, -100, "mean")
                u = torch.ones((), device=z.device)
                for valid_handles in (True, False):
                    selected = handles if valid_handles else torch.empty(1, dtype=torch.int64)
                    if valid_handles:
                        result = host.weighted_cuda_backward(z, y, stats, u, 1, selected, context)
                        torch.testing.assert_close(result, gradient, rtol=0, atol=0)
                    else:
                        with pytest.raises(RuntimeError, match="unexpected Triton kernel handles"):
                            host.weighted_cuda_backward(z, y, stats, u, 1, selected, context)
                    assert current_context() == previous
                    assert torch.cuda.current_device() == 0
        finally:
            torch.cuda.set_device(original)


def invalid_weight_check() -> None:
    """Every reduction="none" row is NaN for an invalid class weight (row 0 flags it), on
    the one-launch and block-partial reductions; valid weights match PyTorch."""
    from libsdca import _triton_weighted_entropy as kernels

    generator = torch.Generator().manual_seed(3)
    kernels.BLOCK, kernels.DIRECT_ROWS = 16, 8
    for dtype, fp32 in ((torch.float32, True), (torch.float32, False), (torch.float64, False)):
        for n in (5, 40):
            z = torch.randn(n, 6, generator=generator, dtype=dtype)
            y = torch.randint(0, 6, (n,), generator=generator)
            weight = torch.linspace(0.5, 2.0, 6, dtype=dtype)
            out = kernels._triton_forward(z, y, weight, -100, 0, fp32)[0]
            expected = torch.nn.functional.cross_entropy(
                z.double(), y, weight.double(), reduction="none"
            )
            torch.testing.assert_close(out.double(), expected, rtol=1e-6, atol=0)
            for bad in (float("nan"), -1.0, float("inf")):
                invalid = weight.clone()
                invalid[int(y[-1])] = bad  # not row 0's class
                out = kernels._triton_forward(z, y, invalid, -100, 0, fp32)[0]
                assert bool(out.isnan().all()), (dtype, fp32, n, bad, out)


class InterpreterTests:
    def test_invalid_weight_in_the_interpreter(self):
        interpreter.run("import test_weighted_cuda as t\nt.invalid_weight_check()\n")
