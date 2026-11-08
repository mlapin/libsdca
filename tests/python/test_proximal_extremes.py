"""Independent proximal limits for finite means and unsupported centered spreads."""

import contextlib
import itertools
from unittest import mock

import mpmath as mp
import pytest
import torch
from libsdca import torch as sdca_torch


class ProximalExtremeTests:
    def check_means(self, subtests: pytest.Subtests, device: str):
        with mp.workdps(80):
            p = mp.findroot(lambda t: mp.log(t / (1 - t)) - (mp.mpf(4) / 3 - 2 * t), 0.55)
            expected_gradient = [float(p - mp.mpf(1) / 3), -1 / 3, -1 / 3, float(1 - p)]
        for dtype, fallback, reduction in itertools.product(
            (torch.float16, torch.bfloat16, torch.float32, torch.float64),
            (False, True) if device == "cuda" else (False,),
            ("none", "sum", "mean"),
        ):
            for precision in (None,) if dtype == torch.float64 else (None, "float32"):
                with subtests.test(
                    dtype=dtype, fallback=fallback, reduction=reduction, precision=precision
                ):
                    limit = torch.finfo(dtype).max
                    x = torch.tensor([[0, -limit, -limit, 0]], device=device, dtype=dtype)
                    x = x.requires_grad_()
                    target = torch.tensor([[True, True, True, False]], device=device)
                    context = (
                        mock.patch.object(sdca_torch, "_prox_kernel", return_value=None)
                        if fallback
                        else contextlib.nullcontext()
                    )
                    with context:
                        value = sdca_torch.sdca_loss(
                            x,
                            target,
                            "proximal_cross_entropy",
                            backend="torch",
                            reduction=reduction,
                            precision=precision,
                        )
                    gradient = torch.autograd.grad(value, x, torch.full_like(value, -0.75))[0]
                    with mp.workdps(80):
                        expected_value = float(mp.mpf(limit) * 2 / 3)
                    tolerance = (
                        1e-2
                        if dtype == torch.bfloat16
                        else 2e-3
                        if dtype == torch.float16
                        else 2e-5
                        if dtype == torch.float32
                        else 2e-14
                    )
                    assert bool(value.isfinite().all())
                    torch.testing.assert_close(
                        value, torch.full_like(value, expected_value), rtol=tolerance, atol=0
                    )
                    expected = x.new_tensor([expected_gradient]) * -0.75
                    torch.testing.assert_close(
                        gradient, expected, rtol=tolerance, atol=tolerance * 0.1
                    )

    def test_cpu_finite_label_means(self, subtests: pytest.Subtests):
        self.check_means(subtests, "cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_finite_label_means(self, subtests: pytest.Subtests):
        self.check_means(subtests, "cuda")

    def test_empty_mean_remains_nan(self):
        assert bool(sdca_torch._reduce(torch.empty(0), "mean").isnan())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_tensor_certified_bracket_endpoint(self, subtests: pytest.Subtests):
        cases = [
            (
                [300.0] + [0.0] * 39,
                [True] * 20 + [False] * 20,
                1.0,
                284.525,
                [0.95] + [-0.05] * 19 + [0.0] * 20,
            ),
            (
                [60.0] + [0.0] * 9,
                [True] * 9 + [False],
                1.0,
                476 / 9,
                [8 / 9] + [-1 / 9] * 8 + [0.0],
            ),
            (
                [1e30, -1e30, 0.0, 1.0, 2.0],
                [True, True, False, False, False],
                1e30,
                0.75e30,
                [0.5, -0.5, 0.0, 0.0, 0.0],
            ),
            (
                [1e300, -1e300, 0.0, 1.0, 2.0],
                [True, True, False, False, False],
                1e30,
                1e300,
                [0.5, -0.5, 0.0, 0.0, 0.0],
            ),
        ]
        # In these limits p is the winning vertex to output precision, so
        # phi=(p-q).z-alpha*|p-q|^2/2 and gradient=p-q independently of the solve.
        for scores, mask, alpha, expected_loss, expected_gradient in cases:
            with subtests.test(scores=scores, alpha=alpha):
                x = torch.tensor([scores], device="cuda", dtype=torch.float64, requires_grad=True)
                target = torch.tensor([mask], device="cuda")
                with mock.patch.object(sdca_torch, "_prox_kernel", return_value=None):
                    value = sdca_torch.sdca_loss(
                        x, target, "proximal_cross_entropy", alpha=alpha, backend="torch"
                    )
                gradient = torch.autograd.grad(value, x)[0]
                torch.testing.assert_close(
                    value, value.new_tensor(expected_loss), rtol=2e-14, atol=0
                )
                torch.testing.assert_close(
                    gradient, x.new_tensor([expected_gradient]), rtol=2e-14, atol=1e-20
                )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_unsupported_spread_preserves_neighbor(self, subtests: pytest.Subtests):
        limit = torch.finfo(torch.float64).max
        with mp.workdps(80):
            r = mp.findroot(lambda p: mp.log(p / (1 - p)) + 2 * p, 0.34)
            expected_loss = float(-mp.log(1 - r) + r * r)
        for target_index, mask, fallback in itertools.product((0, 1), (False, True), (False, True)):
            with subtests.test(target=target_index, mask=mask, fallback=fallback):
                x = torch.tensor(
                    [[limit, -limit], [0, 0], [limit / 4, -limit / 4]],
                    device="cuda",
                    dtype=torch.float64,
                    requires_grad=True,
                )
                labels = torch.tensor([target_index, 0, 0], device="cuda")
                target = torch.nn.functional.one_hot(labels, 2).bool() if mask else labels
                context = (
                    mock.patch.object(sdca_torch, "_prox_kernel", return_value=None)
                    if fallback
                    else contextlib.nullcontext()
                )
                with context:
                    value = sdca_torch.sdca_loss(
                        x, target, "proximal_cross_entropy", backend="torch", reduction="none"
                    )
                gradient = torch.autograd.grad(value.sum(), x)[0]
                assert bool(value[0].isnan())
                assert bool(gradient[0].isnan().all())
                torch.testing.assert_close(
                    value[1], value.new_tensor(expected_loss), rtol=2e-14, atol=0
                )
                torch.testing.assert_close(
                    gradient[1], x.new_tensor([-float(r), float(r)]), rtol=2e-14, atol=0
                )
                assert value[2].item() == 0
                assert bool((gradient[2] == 0).all())

    def test_cpu_unsupported_spread_rejected(self):
        limit = torch.finfo(torch.float64).max
        x = torch.tensor([[limit, -limit]], dtype=torch.float64)
        with pytest.raises(ValueError, match="finite"):
            sdca_torch.sdca_loss(x, torch.tensor([0]), "proximal_cross_entropy", backend="torch")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_tensor_long_solve_graph_capture(self):
        from libsdca import _tensor_prox

        x = torch.zeros((1, 40), device="cuda", dtype=torch.float64)
        x[0, 0] = 300
        mask = (torch.arange(40, device="cuda") < 20)[None, :]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _tensor_prox.proximal_entropy(x, mask, 1.0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                value, gradient = _tensor_prox.proximal_entropy(x, mask, 1.0)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        torch.testing.assert_close(value, value.new_tensor([284.525]), rtol=2e-14, atol=0)
        expected = x.new_tensor([[0.95] + [-0.05] * 19 + [0.0] * 20])
        torch.testing.assert_close(gradient, expected, rtol=2e-14, atol=1e-20)
