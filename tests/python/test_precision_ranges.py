"""Storage extremes must not turn finite inputs into invalid working scores."""

import itertools

import numpy as np
import pytest
import torch
from libsdca import _core
from libsdca.torch import sdca_loss

OBJECTIVES = (
    "topk_hinge_alpha",
    "topk_hinge_beta",
    "multilabel_hinge",
    "proximal_cross_entropy",
)


def extreme_scores(dtype: torch.dtype, device: str, classes: int = 3):
    # Include an ordinary row in the same call: the proximal kernel must choose
    # its working precision independently for each row.
    large = 6e4 if dtype == torch.float16 else 3e38
    z = torch.zeros(5, classes, dtype=dtype, device=device)
    z[:, :3] = torch.tensor(
        [
            [large, -large, 0],
            [1, 2, 3],
            [0, large, -large],
            [large, large, -large],
            [-large, large, large],
        ],
        dtype=dtype,
        device=device,
    )
    return z


def value_and_gradient(z: torch.Tensor, objective: str, precision: str | None, backend: str):
    x = z.detach().clone().requires_grad_()
    labels = torch.zeros(len(x), device=x.device, dtype=torch.int64)
    value = sdca_loss(x, labels, objective, precision=precision, backend=backend, reduction="none")
    (gradient,) = torch.autograd.grad(value.sum(), x)
    return value.detach(), gradient


class PrecisionRangeTests:
    def check_multilabel_entropy_range(self, subtests: pytest.Subtests, device: str):
        for dtype in (torch.float32, torch.float64):
            large = torch.finfo(dtype).max
            cases = (
                ([0.0, -large, -large], large / 3 * 2, [2 / 3, -1 / 3, -1 / 3]),
                ([large, large, -large], large / 3 * 2, [1 / 6, 1 / 6, -1 / 3]),
                ([large, -large], large, [0.5, -0.5]),
            )
            for row, expected_value, expected_gradient in cases:
                precisions = (None, "float64") if dtype == torch.float64 else (None, "float32")
                for precision in precisions:
                    with subtests.test(device=device, dtype=dtype, row=row, precision=precision):
                        x = torch.tensor([row], dtype=dtype, device=device, requires_grad=True)
                        value = sdca_loss(
                            x,
                            torch.ones_like(x, dtype=torch.bool),
                            "multilabel_cross_entropy",
                            backend="torch",
                            precision=precision,
                            reduction="none",
                        )
                        (gradient,) = torch.autograd.grad(value.sum(), x)
                        # At these gaps the tails are below storage range. The
                        # exact limit is max(z)-mean(z), softmax(z)-uniform.
                        tol = 8 * torch.finfo(dtype).eps
                        torch.testing.assert_close(
                            value, value.new_tensor([expected_value]), rtol=tol, atol=0
                        )
                        torch.testing.assert_close(
                            gradient, x.new_tensor([expected_gradient]), rtol=tol, atol=0
                        )

    def test_cpu_multilabel_entropy_finite_average(self, subtests: pytest.Subtests):
        self.check_multilabel_entropy_range(subtests, "cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_cuda_multilabel_entropy_finite_average(self, subtests: pytest.Subtests):
        self.check_multilabel_entropy_range(subtests, "cuda")

    def test_native_half_storage_is_rejected_before_conversion(self, subtests: pytest.Subtests):
        inputs = (
            np.ones((2, 3), dtype=np.float16),
            torch.ones((2, 3), dtype=torch.float16),
            torch.ones((2, 3), dtype=torch.bfloat16),
        )
        for precision in (None, "float32", "float64"):
            for x in inputs:
                for operation in (_core.fit, _core.loss_and_grad, _core.coordinate_update):
                    with subtests.test(precision=precision, dtype=x.dtype, operation=operation):
                        with pytest.raises(
                            ValueError, match="Native APIs require float32 or float64"
                        ):
                            operation(x, [0, 1], precision=precision)
        # Explicit Solver instances must not silently cast half inputs either.
        solver = _core.Solver("float64", dtype=np.float32)
        for x, operation in itertools.product(
            inputs, (solver.fit, solver.loss_and_grad, solver.coordinate_update)
        ):
            with subtests.test(dtype=x.dtype, operation=operation):
                with pytest.raises(ValueError, match="Native APIs require float32 or float64"):
                    operation(x, [0, 1])

    def check_ranges(self, subtests: pytest.Subtests, device: str, classes: int = 3):
        for dtype, objective in itertools.product(
            (torch.float16, torch.bfloat16, torch.float32, torch.float64), OBJECTIVES
        ):
            z = extreme_scores(dtype, device, classes)
            reference = value_and_gradient(z.cpu().double(), objective, "float64", "torch")
            for precision, backend in itertools.product(
                (None, "float32", "float64"), ("auto", "torch")
            ):
                with subtests.test(
                    device=device,
                    dtype=dtype,
                    objective=objective,
                    precision=precision,
                    backend=backend,
                    classes=classes,
                ):
                    if dtype == torch.float64 and precision == "float32":
                        with pytest.raises(ValueError, match="float64 logits require"):
                            value_and_gradient(z, objective, precision, backend)
                        continue
                    value, gradient = value_and_gradient(z, objective, precision, backend)
                    assert gradient.dtype == dtype
                    # The winning target's true loss and gradient are both zero.
                    assert value[0].item() == 0
                    assert gradient[0].abs().sum().item() == 0
                    for actual, expected in zip((value, gradient), reference, strict=True):
                        torch.testing.assert_close(
                            actual.cpu(),
                            expected.to(actual.dtype),
                            rtol=2e-3 if dtype == torch.float16 else 1e-5,
                            atol=2e-3 if dtype == torch.float16 else 1e-6,
                        )

    def test_cpu_finite_storage_extremes(self, subtests: pytest.Subtests):
        self.check_ranges(subtests, "cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_cuda_finite_storage_extremes(self, subtests: pytest.Subtests):
        self.check_ranges(subtests, "cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_cuda_tensor_fallback_finite_storage_extremes(self, subtests: pytest.Subtests):
        # Above the proximal Triton kernel's class limit, exercise tensor solves.
        self.check_ranges(subtests, "cuda", classes=4097)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_cuda_graph_capture_with_overflowing_float32_spread(self, subtests: pytest.Subtests):
        for dtype, objective in itertools.product((torch.float32, torch.bfloat16), OBJECTIVES):
            with subtests.test(dtype=dtype, objective=objective):
                x = extreme_scores(dtype, "cuda").requires_grad_()
                labels = torch.zeros(len(x), device=x.device, dtype=torch.int64)

                def evaluate():
                    value = sdca_loss(
                        x,
                        labels,
                        objective,
                        precision="float32",
                        backend="torch",
                        reduction="none",
                    )
                    (gradient,) = torch.autograd.grad(value.sum(), x)
                    return value, gradient

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    evaluate()
                    evaluate()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    value, gradient = evaluate()
                graph.replay()
                assert value[0].item() == 0
                assert gradient[0].abs().sum().item() == 0
