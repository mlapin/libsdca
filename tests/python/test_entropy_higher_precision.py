"""Higher entropy derivatives preserve tails with bounded and float64 upstreams."""

import mpmath as mp
import pytest
import torch
from libsdca.torch import cross_entropy, sdca_loss


class EntropyHigherPrecisionTests:
    def check_later_derivatives(self, device: str) -> None:
        x = torch.tensor([[0.0, -780.0]], device=device, requires_grad=True)
        y = torch.tensor([0], device=device)
        loss = sdca_loss(x, y, backend="torch", reduction="sum")
        value = torch.autograd.grad(loss, x, create_graph=True)[0][0, 0]
        scale = x.new_tensor(1e38)
        # Each output keeps float32 storage, but its differentiable double
        # intermediates must survive repeated bounded upstream scaling.
        for _ in range(8):
            value = torch.autograd.grad(value, x, scale, create_graph=True)[0][0, 0]
        with mp.workdps(100):
            expected = float(-mp.exp(-780) * mp.mpf(float(scale)) ** 8)
        torch.testing.assert_close(value, value.new_tensor(expected), rtol=2e-6, atol=0)

    def test_cpu_later_derivatives_recover_underflowed_probability(self):
        self.check_later_derivatives("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_later_derivatives_recover_underflowed_probability(self):
        self.check_later_derivatives("cuda")

    def check_derivatives(self, subtests: pytest.Subtests, device: str) -> None:
        cases = ((20.0, 1.0, 1.0), (80.0, 1e30, 1e30), (170.0, 1e30, 1e30))
        for gap, weight, scale in cases:
            for weighted in (False, True):
                for backend in ("torch", "auto") if device == "cuda" else ("torch",):
                    with subtests.test(device=device, gap=gap, weighted=weighted, backend=backend):
                        x = torch.tensor([[0.0, -gap]], device=device, requires_grad=True)
                        y = torch.tensor([0], device=device)
                        w = torch.tensor([weight, 1.0], device=device)
                        value = (
                            cross_entropy(x, y, w, reduction="sum", backend=backend)
                            if weighted
                            else sdca_loss(x, y, reduction="sum", backend=backend)
                        )
                        upstream = value.new_tensor(scale)
                        gradient = torch.autograd.grad(value, x, upstream, create_graph=True)[0]
                        hessian_row = torch.autograd.grad(gradient[0, 0], x)[0]
                        with mp.workdps(100):
                            tail = mp.exp(-gap)
                            factor = mp.mpf(float(upstream)) * (float(w[0]) if weighted else 1)
                            first = float(tail / (1 + tail) * factor)
                            second = float(tail / (1 + tail) ** 2 * factor)
                        torch.testing.assert_close(
                            gradient, x.new_tensor([[-first, first]]), rtol=2e-6, atol=1e-45
                        )
                        torch.testing.assert_close(
                            hessian_row, x.new_tensor([[second, -second]]), rtol=2e-6, atol=1e-45
                        )

    def test_cpu(self, subtests: pytest.Subtests):
        self.check_derivatives(subtests, "cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda(self, subtests: pytest.Subtests):
        self.check_derivatives(subtests, "cuda")

    def test_float64_upstream_preserves_double_underflow_rescue(self):
        from libsdca import _entropy

        x = torch.tensor([[0.0, -780.0]], requires_grad=True)
        y = torch.tensor([0])
        stats = _entropy._statistics(x, y)
        upstream = torch.tensor(1e300, dtype=torch.float64)
        gradient = _entropy._gradient(x, y, stats, upstream, "none")
        with mp.workdps(100):
            expected = float(mp.exp(-780) * mp.mpf(float(upstream)))
        torch.testing.assert_close(
            gradient, x.new_tensor([[-expected, expected]]), rtol=2e-6, atol=1e-45
        )
