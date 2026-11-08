"""Precision corner cases on the Torch layers (CPU, and CUDA if present).

References are analytic or mpmath, never the other backend.
"""

import math

import mpmath
import pytest
import torch
from libsdca import _tensor_prox
from libsdca.torch import sdca_loss

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class StressTorchTests:
    def test_entropy_at_the_edge_of_the_dtype(self):
        # Target 0 at the largest logit of [big, -big]: the loss and gradient
        # round to zero. Forward first, then backward on its own.
        for dtype, big in [(torch.float32, 3e38), (torch.float64, 1e308)]:
            backends = [("native", "cpu")] + [("torch", device) for device in DEVICES]
            for backend, device in backends:
                z = torch.tensor([[big, -big]], dtype=dtype, device=device, requires_grad=True)
                value = sdca_loss(z, torch.tensor([0], device=device), backend=backend)
                message = f"{dtype} {backend} {device}"
                assert value.item() == 0.0, message
                value.backward()
                torch.testing.assert_close(z.grad, torch.zeros_like(z), atol=0, rtol=0, msg=message)

    def test_tensor_lambert_w_at_huge_arguments(self):
        # W(exp(x)) overflowed to NaN from x ~ 9e307. Reference: the
        # asymptotic series L1 - L2 + L2/L1 + L2(L2-2)/(2 L1^2), L1 = x, L2 = log x.
        def series(x: float) -> float:
            l1 = mpmath.mpf(x)
            l2 = mpmath.log(l1)
            return float(l1 - l2 + l2 / l1 + l2 * (l2 - 2) / (2 * l1 * l1))

        with mpmath.workdps(50):
            for dtype, values in [
                (torch.float64, [1e17, 1e100, 1e300, 9e307, 1e308, 1.7976931348623157e308]),
                (torch.float32, [1e8, 1e20, 3e38, 3.4028234663852886e38]),
            ]:
                x = torch.tensor(values, dtype=dtype)
                w = _tensor_prox._lambert_w_exp(x)
                expected = torch.tensor([series(v) for v in x.tolist()], dtype=torch.float64)
                torch.testing.assert_close(
                    w.double(), expected, rtol=2 * torch.finfo(dtype).eps, atol=0, msg=str(dtype)
                )

    def test_tensor_proximal_entropy_validates_its_own_inputs(self):
        # The tensor proximal_entropy itself rejects invalid input.
        z = torch.zeros(2, 3, dtype=torch.float64)
        labels = torch.tensor([[True, False, False], [False, True, False]])
        for scores, positive, alpha in [
            (z, labels, 0.0),
            (z, labels, math.nan),
            (z, labels, math.inf),
            (torch.tensor([[0.0, math.inf, 0.0], [0.0, 0.0, 0.0]]).double(), labels, 1.0),
            (z, torch.ones_like(labels), 1.0),
            (z, torch.zeros_like(labels), 1.0),
        ]:
            with pytest.raises(ValueError):
                _tensor_prox.proximal_entropy(scores, positive, alpha)

    def test_barron_collapse_and_extreme_wrong_margins(self):
        # Past -ln(eps/2), PyTorch cross_entropy collapses to 0 while sdca_loss stays
        # accurate and zero-sum; at [-max, max] the gradient stays exact [-1, 1].
        cases = [
            (torch.float16, 18.0, 1.5229979055675358e-8, 0.0),
            (torch.bfloat16, 18.0, 1.5229979055675358e-8, 1.525040715932846e-8),
            (torch.float32, 17.0, 4.139937459513021e-8, 4.139937459513021e-8),
            (torch.float64, 37.0, 8.533047625744065e-17, 8.533047625744065e-17),
        ]
        for device in DEVICES:
            y = torch.tensor([0], device=device)
            for dtype, margin, expected_loss, expected_g in cases:
                z_pt = torch.tensor([[margin, 0.0]], dtype=dtype, device=device, requires_grad=True)
                try:
                    l_pt = torch.nn.functional.cross_entropy(z_pt, y)
                except RuntimeError:  # PyTorch 2.1 has no CPU float16 log_softmax
                    l_pt = None
                if l_pt is not None:
                    l_pt.backward()
                    assert z_pt.grad is not None
                    assert l_pt.item() == 0.0
                    assert z_pt.grad[0, 0].item() == 0.0

                z = torch.tensor([[margin, 0.0]], dtype=dtype, device=device, requires_grad=True)
                val = sdca_loss(z, y)
                val.backward()
                assert z.grad is not None
                assert round(abs(val.item() / expected_loss - 1.0), 5) == 0
                if expected_g > 0:
                    assert round(abs(-z.grad[0, 0].item() / expected_g - 1.0), 2) == 0
                    assert round(abs(z.grad[0, 1].item() / expected_g - 1.0), 2) == 0
                assert (
                    abs(float(z.grad.double().sum().item())) <= torch.finfo(dtype).eps * expected_g
                )

                fmax = torch.finfo(dtype).max
                zw = torch.tensor([[-fmax, fmax]], dtype=dtype, device=device, requires_grad=True)
                sdca_loss(zw, y).backward()
                torch.testing.assert_close(
                    zw.grad, torch.tensor([[-1.0, 1.0]], dtype=dtype, device=device)
                )
