"""The tensor proximal_entropy against the native one."""

import math
import textwrap
import types

import interpreter
import pytest
import torch
from libsdca import _tensor_prox
from libsdca import torch as sdca_torch
from libsdca.torch import SDCALoss, sdca_loss
from torch.autograd import gradcheck

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
ALPHAS = [1e-300, 1e-18, 1e-6, 1e-2, 0.5, 1.0, 30.0, 1e4, 1e18, 1e300]


# Labels at the largest score, other classes whose probabilities underflow: the
# float64 rows did not converge or, at alpha = 1e300, gave a nonfinite loss.
UNDERFLOW_ROWS = [
    ([720.0, 720.0, 0.0, 1.0, 2.0], [0, 1]),
    ([745.0, 745.0, 0.0, 1.0, 2.0], [0, 1]),
    ([745.0, 0.0, 1.0, 2.0, -1.0], [0]),
]
# The same in float32 (scores 100 apart), which also gave wrong values at alpha = 1e4;
# the two-class row did not converge in the CUDA kernel only.
UNDERFLOW_ROWS_FLOAT = [
    ([100.0, 100.0, 0.0, 1.0, 2.0], [0, 1]),
    ([100.0, 0.0, 1.0, 2.0, -1.0], [0]),
    ([90.0, 90.0, 0.0, 1.0, 2.0], [0, 1]),
    ([-47.306270599365234, 52.37538528442383], [1]),
]


def underflow_cases(rows: list[tuple[list[float], list[int]]], dtype: torch.dtype = torch.float64):
    """(scores, mask) of rows of UNDERFLOW_ROWS."""
    for row, labels in rows:
        z = torch.tensor([row], dtype=dtype)
        mask = torch.zeros_like(z, dtype=torch.bool)
        mask[0, labels] = True
        yield z, mask


def labels_and_masks(examples: int, classes: int, generator: torch.Generator):
    """Class indices, and masks with a positive and a negative label."""
    labels = torch.randint(0, classes, (examples,), generator=generator)
    mask = torch.rand(examples, classes, generator=generator) < 0.4
    rows = torch.arange(examples)
    mask[rows, labels] = True
    mask[rows, (labels + 1) % classes] = False
    return [labels, mask]


class TensorProximalEntropyTests:
    def check(self, z: torch.Tensor, target: torch.Tensor, alpha: float, rtol: float, atol: float):
        """Values and weighted gradients of the tensor backend against the native backend."""
        reference = z.detach().clone().requires_grad_()
        weights = torch.linspace(0.5, 2.0, z.shape[0], dtype=z.dtype)
        expected = sdca_loss(
            reference,
            target,
            "proximal_cross_entropy",
            alpha=alpha,
            backend="native",
            reduction="none",
        )
        expected.backward(weights)
        for device in DEVICES:
            x = z.detach().to(device).requires_grad_()
            actual = sdca_loss(
                x,
                target.to(device),
                "proximal_cross_entropy",
                alpha=alpha,
                backend="torch",
                reduction="none",
            )
            actual.backward(weights.to(device))
            message = f"alpha={alpha} {device} {z.dtype} {target.dtype}"
            assert x.grad is not None
            torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol, msg=message)
            torch.testing.assert_close(
                x.grad.cpu(), reference.grad, rtol=rtol, atol=atol, msg=message
            )

    def test_random_scores_match_native(self):
        generator = torch.Generator().manual_seed(17)
        for dtype, rtol in [(torch.float64, 1e-12), (torch.float32, 3e-7)]:
            for classes in [2, 3, 7, 40, 300]:
                for scale in [1.0, 100.0]:
                    z = (scale * torch.randn(48, classes, generator=generator)).to(dtype)
                    for target in labels_and_masks(48, classes, generator):
                        for alpha in ALPHAS:
                            self.check(z, target, alpha, rtol, rtol * max(1.0, scale))

    def test_far_offsets_ties_and_confident_rows(self):
        rows = [
            [1e16] * 3,
            [1e16, 1e16, 1e16 + 2],
            [700.0, 0.0, 1.0, 700.0, -0.3, 0.2],
            [0.0, 20.0, 20.0, -100.0],
            [1000.0, 0.0, 0.0, -5.0],
            [0.0, 0.0, 0.0, 0.0],
            [40.0, 0.0, 0.0, 0.0],
            [-40.0, 0.0, 0.0, 0.0],
        ]
        for gap in [700.0, 1e16, 1e30]:
            rows += [[-gap, 0.0, 1.0, 2.0, -1.0], [gap, 0.0, 1.0, 2.0, -1.0]]
        for row in rows:
            z = torch.tensor([row], dtype=torch.float64)
            mask = torch.zeros_like(z, dtype=torch.bool)
            mask[0, ::2] = True
            for target in [torch.tensor([0]), torch.tensor([1]), mask]:
                for alpha in ALPHAS:
                    self.check(z, target, alpha, 1e-12, 1e-13)

    def test_values_against_independent_references(self):
        # Equal label scores reach log 2; u = v + alpha a would cancel.
        z = torch.tensor([[1e16, 0.0, 0.0]], dtype=torch.float64)
        mask = torch.tensor([[True, True, False]])
        # a^2 underflows; d + log d + log1p(exp(-d)) = log(2 alpha) gives
        # the value d^2 / (4 alpha) + log1p(exp(-d)) (80-digit solve).
        zero, label = torch.zeros(1, 2, dtype=torch.float64), torch.tensor([0])
        for backend in ["native", "torch"]:
            for logits, target, alpha, expected in [
                (z, mask, 1e50, math.log(2)),
                (zero, label, 1e300, 1.1762794618348152e-295),
            ]:
                value = sdca_loss(
                    logits, target, "proximal_cross_entropy", alpha=alpha, backend=backend
                ).item()
                assert abs(value / expected - 1.0) <= 1e-13, backend

    def test_stationarity_and_envelope(self):
        """g = softmax(z - alpha g) - q, and the value is at most H(q, z)."""
        generator = torch.Generator().manual_seed(9)
        z = 4 * torch.randn(32, 7, dtype=torch.float64, generator=generator)
        for target in labels_and_masks(32, 7, generator):
            mask = target if target.dtype == torch.bool else torch.nn.functional.one_hot(target, 7)
            q = mask.double() / mask.sum(-1, keepdim=True)
            for alpha in [1e-3, 0.5, 20.0]:
                x = z.clone().requires_grad_()
                value = sdca_loss(
                    x,
                    target,
                    "proximal_cross_entropy",
                    alpha=alpha,
                    backend="torch",
                    reduction="none",
                )
                (g,) = torch.autograd.grad(value.sum(), x)
                u = z - alpha * g
                torch.testing.assert_close(torch.softmax(u, -1) - q, g, rtol=0, atol=5e-14)
                cross = -(q * torch.log_softmax(z, -1)).sum(-1)
                assert bool((value <= cross + 1e-14).all())

    def test_gradcheck_layout_and_auto_backend(self):
        generator = torch.Generator().manual_seed(5)
        z = torch.randn(2, 5, 3, dtype=torch.float64, generator=generator)
        labels = torch.tensor([[0, 4, 2], [1, 1, 3]])
        expected = SDCALoss("proximal_cross_entropy", alpha=0.7, backend="native", dim=1)(z, labels)
        for device in DEVICES:
            x = z.to(device).requires_grad_()
            criterion = SDCALoss("proximal_cross_entropy", alpha=0.7, backend="torch", dim=1)
            torch.testing.assert_close(criterion(x, labels.to(device)).cpu(), expected)
            assert gradcheck(lambda v: criterion(v, labels.to(device)), (x,))
            if device == "cuda":
                auto = sdca_loss(x, labels.cuda(), "proximal_cross_entropy", alpha=0.7, dim=1)
                torch.testing.assert_close(auto.cpu(), expected)

    def test_wide_rows(self):
        """Rows past the register kernel: streamed on CUDA."""
        classes = 4097
        z = torch.randn(3, classes, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
        labels = torch.tensor([0, 5, classes - 1])
        self.check(z, labels, 0.5, 1e-12, 1e-12)

    def test_kernel_dispatch_bounds(self):
        """One program per row: 2**31 rows exceed the CUDA grid and take the tensor solve."""
        for n, c, kernel in ((2**31 - 1, 4096, True), (2**31, 2, False), (2, 2**20, True)):
            logits = types.SimpleNamespace(is_cuda=True, shape=(n, c))
            expected = sdca_torch._kernel_prox() if kernel else None
            # A stand-in: these shapes cannot be allocated.
            chosen = sdca_torch._prox_kernel(logits)  # pyright: ignore[reportArgumentType]
            assert chosen is expected, (n, c)

    def test_float32_solve_is_accurate_to_float32(self):
        """The float32 solve (fewer fixed steps) against float64 on the same scores."""
        generator = torch.Generator().manual_seed(29)
        for classes, scale in [(2, 1.0), (7, 10.0), (40, 1.0), (300, 100.0)]:
            z = (scale * torch.randn(16, classes, generator=generator)).float()
            for target in labels_and_masks(16, classes, generator):
                if target.dtype != torch.bool:
                    target = torch.nn.functional.one_hot(target, classes).bool()
                for alpha in [1e-30, 1e-3, 1.0, 1e3, 1e30]:
                    expected = _tensor_prox.proximal_entropy(z.double(), target, alpha)
                    actual = _tensor_prox.proximal_entropy(z, target, alpha)
                    for a, e in zip(actual, expected, strict=True):
                        assert a.dtype == torch.float32
                        torch.testing.assert_close(a.double(), e, rtol=2e-5, atol=2e-6)

    def test_underflowing_other_classes(self):
        for z, mask in underflow_cases(UNDERFLOW_ROWS):
            for alpha in ALPHAS:
                self.check(z, mask, alpha, 1e-12, 1e-300)
        # float32: the solve against native float32, and float16/bfloat16 logits.
        for z, mask in underflow_cases(UNDERFLOW_ROWS_FLOAT, torch.float32):
            for alpha in [1e-6, 1e-2, 1.0, 30.0, 1e4]:
                values = {}
                for dtype in [torch.float32, torch.bfloat16, torch.float16]:
                    for backend in ["native", "torch"] if dtype == torch.float32 else ["torch"]:
                        x = z.detach().to(dtype).clone().requires_grad_()
                        value = sdca_loss(
                            x,
                            mask,
                            "proximal_cross_entropy",
                            alpha=alpha,
                            backend=backend,
                            precision="float32",
                            reduction="sum",
                        )
                        value.backward()
                        assert x.grad is not None
                        assert bool(x.grad.isfinite().all()), f"{alpha} {dtype}"
                        values[backend, dtype] = value.detach().float(), x.grad.float()
                expected = values["native", torch.float32]
                for (backend, dtype), actual in values.items():
                    message = f"alpha={alpha} {backend} {dtype}"
                    # Values and gradients in the logits dtype; a subnormal float32
                    # value (one label) is resolved to a few subnormals.
                    single = dtype == torch.float32
                    info = torch.finfo(dtype)
                    rtol, atol = (3e-7, 2e-44) if single else (info.eps / 2, info.tiny)
                    torch.testing.assert_close(
                        actual[0], expected[0], rtol=rtol, atol=atol, msg=message
                    )
                    torch.testing.assert_close(
                        actual[1], expected[1], rtol=4e-3, atol=info.tiny, msg=message
                    )

    def test_validation(self):
        for device in DEVICES:
            z = torch.zeros(2, 3, device=device)
            if device == "cpu":
                with pytest.raises(ValueError, match="negative label"):
                    sdca_loss(
                        z,
                        torch.ones_like(z, dtype=torch.bool),
                        "proximal_cross_entropy",
                        backend="torch",
                    )
            else:
                assert bool(
                    sdca_loss(
                        z,
                        torch.ones_like(z, dtype=torch.bool),
                        "proximal_cross_entropy",
                        backend="torch",
                    )
                    .isnan()
                    .all()
                )
            with pytest.raises(ValueError):
                sdca_loss(
                    z, torch.tensor([0, 1], device=device), "proximal_cross_entropy", alpha=0.0
                )


# Rows that converge, tie, saturate or sit far from the others.
EDGE_ROWS = [
    [1e16] * 3,
    [700.0, 0.0, 1.0, 700.0, -0.3, 0.2],
    [0.0, 20.0, 20.0, -100.0],
    [40.0, 0.0, 0.0, 0.0],
    [-1e30, 0.0, 1.0, 2.0, -1.0],
]


def kernel_cases(generator: torch.Generator, examples: int = 8):
    """Scores, masks with a positive and a negative label, and alphas."""
    for classes in [2, 7, 40]:
        z = 5 * torch.randn(examples, classes, generator=generator, dtype=torch.float64)
        for target in labels_and_masks(examples, classes, generator):
            if target.dtype != torch.bool:
                target = torch.nn.functional.one_hot(target, classes).bool()
            yield z, target
    for row in EDGE_ROWS:
        z = torch.tensor([row], dtype=torch.float64)
        mask = torch.zeros_like(z, dtype=torch.bool)
        mask[0, ::2] = True
        yield z, mask
        yield z, torch.zeros_like(mask).index_fill_(1, torch.tensor([1]), True)
    yield from underflow_cases(UNDERFLOW_ROWS)


# The interpreted checks of TritonProximalEntropyTests.test_kernel_in_the_interpreter.
INTERPRETED = {
    "float64": """
        for z, target in t.kernel_cases(torch.Generator().manual_seed(23), 4):
            for alpha in [1e-300, 1e-6, 1.0, 1e4, 1e300]:
                expected = _tensor_prox.proximal_entropy(z, target, alpha)
                actual = _triton_prox.proximal_entropy(z, target, alpha)
                for a, e in zip(actual, expected, strict=True):
                    torch.testing.assert_close(a, e, rtol=1e-13, atol=1e-14)
            if z.abs().max() < 1e30:
                actual = _triton_prox.proximal_entropy(z.float(), target, 1.0)
                expected = _tensor_prox.proximal_entropy(z.float().double(), target, 1.0)
                for a, e in zip(actual, expected, strict=True):
                    torch.testing.assert_close(a.double(), e, rtol=2e-5, atol=2e-6)
    """,
    "float16": """
        # float16 scores read directly; float32 gradients, which backward scales before
        # their only rounding into float16.
        for z, target in t.kernel_cases(torch.Generator().manual_seed(29), 4):
            if z.abs().max() < 6e4:
                h = z.half()
                expected = _tensor_prox.proximal_entropy(h.double(), target, 1.0)
                loss, gradient = _triton_prox.proximal_entropy(h, target, 1.0, False)
                assert gradient.dtype == torch.float32
                torch.testing.assert_close(loss, expected[0], rtol=1e-13, atol=1e-14)
                torch.testing.assert_close(
                    gradient.double(), expected[1], rtol=2.4e-7, atol=1e-44
                )
    """,
    "underflow": """
        # The float32 rows whose other classes underflow: the kernel is the tensor solve.
        for z, target in t.underflow_cases(t.UNDERFLOW_ROWS_FLOAT, torch.float32):
            for alpha in [1e-6, 1.0, 1e4]:
                expected = _tensor_prox.proximal_entropy(z, target, alpha)
                actual = _triton_prox.proximal_entropy(z, target, alpha)
                for a, e in zip(actual, expected, strict=True):
                    torch.testing.assert_close(a, e, rtol=4e-7, atol=2e-44)
    """,
}


class TritonProximalEntropyTests:
    def test_kernel_matches_the_tensor_solve(self):
        kernel = sdca_torch._kernel_prox() if torch.cuda.is_available() else None
        if kernel is None:
            pytest.skip("needs Triton and an NVIDIA GPU")
        for z, target in kernel_cases(torch.Generator().manual_seed(23)):
            for alpha in ALPHAS:
                expected = _tensor_prox.proximal_entropy(z, target, alpha)
                actual = kernel(z.cuda(), target.cuda(), alpha)
                for a, e in zip(actual, expected, strict=True):
                    torch.testing.assert_close(a.cpu(), e, rtol=1e-13, atol=1e-14)
                # The float32 kernel is accurate to float32 where float32 holds the scores.
                if alpha in (1e-6, 1.0, 1e4) and z.abs().max() < 1e30:
                    actual = kernel(z.float().cuda(), target.cuda(), alpha)
                    expected = _tensor_prox.proximal_entropy(z.float().double(), target, alpha)
                    for a, e in zip(actual, expected, strict=True):
                        assert a.dtype == torch.float32
                        torch.testing.assert_close(a.cpu().double(), e, rtol=2e-5, atol=2e-6)
        # float32 rows whose other classes underflow, against the CPU tensor solve.
        for z, target in underflow_cases(UNDERFLOW_ROWS_FLOAT, torch.float32):
            for alpha in [1e-30, 1e-6, 1e-2, 1.0, 30.0, 1e4, 1e16, 1e30]:
                expected = _tensor_prox.proximal_entropy(z, target, alpha)
                actual = kernel(z.cuda(), target.cuda(), alpha)
                for a, e in zip(actual, expected, strict=True):
                    torch.testing.assert_close(a.cpu(), e, rtol=2e-6, atol=2e-44)
        # 16-bit scores, read directly: gradients stay float32 until upstream scaling.
        for z, target in kernel_cases(torch.Generator().manual_seed(29), 4):
            for dtype in (torch.float16, torch.bfloat16):
                if z.abs().max() >= torch.finfo(dtype).max:
                    continue
                h = z.to(dtype)
                expected = _tensor_prox.proximal_entropy(h.double(), target, 1.0)
                for fp32, tol in ((False, 1e-13), (True, 2e-5)):
                    loss, gradient = kernel(h.cuda(), target.cuda(), 1.0, fp32)
                    assert gradient.dtype == torch.float32
                    torch.testing.assert_close(loss.cpu().double(), expected[0], rtol=tol, atol=tol)
                    torch.testing.assert_close(
                        gradient.cpu().double(),
                        expected[1].to(gradient.dtype).double(),
                        rtol=tol + torch.finfo(gradient.dtype).eps,
                        atol=2e-6 if fp32 else 0.0,
                    )
        loss_bad, grad_bad = kernel(
            torch.zeros(1, 3, device="cuda", dtype=torch.float64),
            torch.ones(1, 3, device="cuda", dtype=torch.bool),
            1.0,
        )
        assert bool(loss_bad.isnan().all())
        assert bool(grad_bad.isnan().all())

    def test_streamed_rows_match_rows_in_registers(self):
        """Rows streamed in tiles of 16 columns against the same rows in registers."""
        from unittest import mock

        kernel = sdca_torch._kernel_prox() if torch.cuda.is_available() else None
        if kernel is None:
            pytest.skip("needs Triton and an NVIDIA GPU")
        from libsdca import _triton_prox

        generator = torch.Generator().manual_seed(31)
        cases = list(kernel_cases(generator, 4))
        z = 3 * torch.randn(3, 70, generator=generator, dtype=torch.float64)
        cases += [(z, mask) for mask in (z > 0, z >= z.amax(-1, keepdim=True))]
        # Labels far below: their residuals vanish and they share the others' mass.
        z, mask = torch.zeros(1, 40, dtype=torch.float64), torch.zeros(1, 40, dtype=torch.bool)
        z[0, :20], mask[0, :20] = -1e30, True
        cases.append((z, mask))
        for z, target in cases:
            for alpha in ALPHAS:
                for dtype in (torch.float64, torch.float32, torch.bfloat16):
                    if z.abs().max() >= torch.finfo(dtype).max:
                        continue
                    x, t = z.to("cuda", dtype), target.cuda()
                    for fp32 in (False, True) if dtype != torch.float64 else (False,):
                        rtol, atol = (2e-5, 2e-6) if fp32 else (1e-13, 1e-14)
                        expected = kernel(x, t, alpha, fp32)
                        with (
                            mock.patch.object(_triton_prox, "REGISTER_CLASSES", 0),
                            mock.patch.object(_triton_prox, "_STREAM_BLOCK", 16),
                        ):
                            actual = kernel(x, t, alpha, fp32)
                        message = f"{dtype} fp32={fp32} alpha={alpha} {z.shape}"
                        for a, e in zip(actual, expected, strict=True):
                            assert a.dtype == e.dtype
                            torch.testing.assert_close(
                                a.cpu().double(),
                                e.cpu().double(),
                                rtol=max(rtol, torch.finfo(a.dtype).eps),
                                atol=atol,
                                equal_nan=True,
                                msg=message,
                            )

    @pytest.mark.slow
    @pytest.mark.parametrize("part", list(INTERPRETED))
    def test_kernel_in_the_interpreter(self, part: str):
        """The kernel's control flow on CPU, with numpy in place of libdevice."""
        prelude = (
            "import torch, test_tensor_prox as t\nfrom libsdca import _tensor_prox, _triton_prox\n"
        )
        interpreter.run(prelude + textwrap.dedent(INTERPRETED[part]))
