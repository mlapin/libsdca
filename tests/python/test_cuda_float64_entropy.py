"""Full-range float64 named CUDA entropy, with forward validation reused by autograd.

Run this module under SDCA_CUDA_LAUNCHER=off and =on to cover both launchers.
The small extreme-input oracle uses represented inputs and 100-digit arithmetic;
large row/column boundaries use independent two-level-logit formulas.
"""

import math
import os
from dataclasses import dataclass
from unittest import mock

import mpmath as mp
import pytest
import torch
import torch.nn.functional as F
from libsdca import _cross_entropy, _cuda_launch
from libsdca.torch import cross_entropy

_QUANTUM = math.ulp(0.0)


@dataclass
class _WeightedContext:
    """The autograd context fields _FusedWeightedEntropy.backward reads."""

    saved_tensors: tuple[torch.Tensor, ...]
    has_weight: bool
    ignore_index: int
    reduction: str
    fp32: bool = False
    cuda_context: int | None = None

    def save_for_backward(self, *tensors: torch.Tensor) -> None:
        self.saved_tensors = tensors


@dataclass
class _EntropyContext:
    """The autograd context fields _Entropy.backward reads."""

    saved_tensors: tuple[torch.Tensor, ...]
    fused: bool
    cpu_fused: bool
    native: bool
    fp32: bool
    reduction: str

    def save_for_backward(self, *tensors: torch.Tensor) -> None:
        self.saved_tensors = tensors


def _reference(
    z: torch.Tensor,
    target: torch.Tensor,
    reduction: str,
    upstream: torch.Tensor,
    ignore: int = -100,
):
    values, gradients = [], []
    rows = z.detach().double().cpu().tolist()
    labels = target.cpu().tolist()
    u = upstream.double().cpu().reshape(-1).tolist()
    with mp.workdps(100):
        count = sum(label != ignore for label in labels)
        divisor = mp.mpf(count if reduction == "mean" else 1)
        for i, (row, label) in enumerate(zip(rows, labels, strict=True)):
            if label == ignore:
                values.append(mp.mpf(0))
                gradients.append([mp.mpf(0)] * len(row))
                continue
            scores = [mp.mpf(v) for v in row]
            maximum = max(score for j, score in enumerate(scores) if j != label)
            e = [
                mp.exp(score - maximum) if j != label else mp.mpf(0)
                for j, score in enumerate(scores)
            ]
            relative = maximum - scores[label] + mp.log(sum(e))
            values.append(
                relative + mp.log1p(mp.exp(-relative))
                if relative > 0
                else mp.log1p(mp.exp(relative))
            )
            mass = 1 / (1 + mp.exp(-relative))
            scale = mp.mpf(u[i] if reduction == "none" else u[0]) / divisor
            gradient = [entry / sum(e) * mass * scale for entry in e]
            gradient[label] = -mass * scale
            gradients.append(gradient)
        value = [float(v) for v in values] if reduction == "none" else float(sum(values) / divisor)
        return torch.tensor(value, dtype=torch.float64), torch.tensor(
            [[float(v) for v in row] for row in gradients], dtype=torch.float64
        )


class SavedFloat64ValidationTests:
    def test_python_higher_derivative_uses_full64_saved_status(self, subtests: pytest.Subtests):
        # This exercises the Python fallback without requiring CUDA. The full64
        # log-value tail follows the same status and denominator offsets.
        for reduction in ("none", "sum", "mean"):
            with subtests.test(reduction=reduction):
                z = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
                target = torch.tensor([99, 1, -100])
                stats = torch.zeros(6 * 3 + 2, dtype=torch.float64)
                stats[3:12:4] = torch.tensor([-1.0, 1.0, 0.0])
                stats[15:17] = torch.tensor([1.0, 1.0])
                stats[17:] = torch.tensor([math.nan, math.log(math.log(3)), -math.inf])
                ctx = _WeightedContext(
                    saved_tensors=(z, target, target, stats),
                    has_weight=False,
                    ignore_index=-100,
                    reduction=reduction,
                )
                upstream = (
                    torch.ones(3, dtype=torch.float64)
                    if reduction == "none"
                    else torch.tensor(1.0, dtype=torch.float64)
                )
                gradient = _cross_entropy._FusedWeightedEntropy.backward(ctx, upstream)[0]
                assert gradient.requires_grad
                if reduction == "none":
                    assert bool(gradient[0].isnan().all())
                    torch.testing.assert_close(gradient[1], z.new_tensor([1 / 3, -2 / 3, 1 / 3]))
                    torch.testing.assert_close(gradient[2], torch.zeros_like(gradient[2]))
                else:
                    assert bool(gradient.isnan().all())

    def test_entropy_python_higher_derivative_uses_saved_status(self):
        from libsdca import _entropy

        z = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
        target = torch.tensor([0, 1])
        stats = torch.zeros(6 * 2 + 1, dtype=torch.float64)
        stats[10:12] = torch.tensor([1.0, 0.0])
        ctx = _EntropyContext(
            saved_tensors=(z, target, stats),
            fused=True,
            cpu_fused=False,
            native=False,
            fp32=False,
            reduction="none",
        )
        gradient = _entropy._Entropy.backward(ctx, torch.ones(2, dtype=torch.float64))[0]
        assert gradient.requires_grad
        assert bool(gradient[0].isnan().all())
        torch.testing.assert_close(gradient[1], z.new_tensor([1 / 3, -2 / 3, 1 / 3]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class Float64CudaEntropyTests:
    @classmethod
    def setup_class(cls):
        if _cross_entropy._weighted_kernel() is None:
            pytest.skip("Triton entropy kernels required")
        if os.environ.get(_cuda_launch.ENV) == "1" and _cuda_launch.host() is None:
            raise AssertionError("Requested C++ launcher is unavailable")

    def assert_reference(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        reduction: str,
        upstream: torch.Tensor,
        *,
        graph: bool = False,
        ignore: int = -100,
    ) -> None:
        expected, expected_gradient = _reference(z, y, reduction, upstream, ignore)
        x = z.cuda().detach().requires_grad_()
        value = cross_entropy(x, y.cuda(), reduction=reduction, ignore_index=ignore)
        gradient = torch.autograd.grad(value, x, upstream.cuda(), create_graph=graph)[0]
        assert value.dtype == torch.float64
        assert gradient.dtype == torch.float64
        # Log-domain rescue incurs logarithm/exponential rounding at magnitudes
        # near exp(+/-700); subnormal errors remain bounded by one quantum.
        torch.testing.assert_close(value.cpu(), expected, rtol=2e-12, atol=_QUANTUM)
        torch.testing.assert_close(gradient.cpu(), expected_gradient, rtol=2e-12, atol=_QUANTUM)

    def test_fused_dispatch_layouts_and_hessian_products(self, subtests: pytest.Subtests):
        torch.manual_seed(1729)
        for shape in ((5,), (4, 5), (2, 3, 5, 2)):
            z = torch.randn(shape, device="cuda", dtype=torch.float64)
            if len(shape) == 4:
                z = z.transpose(1, 2)
            dim = 0 if z.ndim == 1 else 1
            target_shape = z.shape[:dim] + z.shape[dim + 1 :]
            y = torch.randint(z.shape[dim], target_shape, device="cuda")
            if y.numel() > 1:
                y.reshape(-1)[0] = -100
            for reduction in ("none", "sum", "mean"):
                with subtests.test(shape=shape, reduction=reduction):
                    x = z.detach().requires_grad_()
                    with mock.patch.object(
                        _cross_entropy._WeightedEntropy,
                        "apply",
                        side_effect=AssertionError("float64 unweighted call fell back to tensors"),
                    ):
                        actual = cross_entropy(x, y, reduction=reduction)
                    refx = z.detach().clone().requires_grad_()
                    reference = F.cross_entropy(refx, y, reduction=reduction)
                    grad = torch.autograd.grad(actual.sum(), x, create_graph=True)[0]
                    refgrad = torch.autograd.grad(reference.sum(), refx, create_graph=True)[0]
                    direction = torch.randn_like(x)
                    hessian = torch.autograd.grad((grad * direction).sum(), x)[0]
                    refhessian = torch.autograd.grad((refgrad * direction).sum(), refx)[0]
                    torch.testing.assert_close(actual, reference, rtol=2e-13, atol=2e-15)
                    torch.testing.assert_close(grad, refgrad, rtol=2e-12, atol=2e-15)
                    torch.testing.assert_close(hessian, refhessian, rtol=2e-11, atol=2e-14)

    def test_full_range_losses_and_finite_mean(self, subtests: pytest.Subtests):
        limit = torch.finfo(torch.float64).max
        z = torch.tensor([[-limit, limit], [limit, -limit], [0.0, 0.0]], dtype=torch.float64)
        y = torch.tensor([0, 0, -100])
        for reduction in ("none", "sum", "mean"):
            with subtests.test(reduction=reduction):
                upstream = (
                    torch.ones(3, dtype=torch.float64)
                    if reduction == "none"
                    else torch.tensor(1.0, dtype=torch.float64)
                )
                self.assert_reference(z, y, reduction, upstream)
                self.assert_reference(z, y, reduction, upstream, graph=True)
        # Every row's loss is finite, but the ordinary reduction sum overflows.
        z = torch.tensor([[0.0, 1e308]] * 4, dtype=torch.float64)
        self.assert_reference(
            z, torch.zeros(4, dtype=torch.int64), "mean", torch.tensor(1.0, dtype=torch.float64)
        )

    def test_subnormal_rows_and_upstream_rescue(self, subtests: pytest.Subtests):
        z = torch.tensor(
            [[0.0, -gap] for gap in (40, 710, 740, 745, 746, 750, 1000, 1400)], dtype=torch.float64
        )
        y = torch.zeros(len(z), dtype=torch.int64)
        y[-2] = -100
        for reduction in ("none", "sum", "mean"):
            for scale in (0.0, 1.0, -3.0, 1e300):
                for graph in (False, True):
                    with subtests.test(reduction=reduction, scale=scale, graph=graph):
                        upstream = torch.full(
                            (len(z),) if reduction == "none" else (), scale, dtype=torch.float64
                        )
                        self.assert_reference(z, y, reduction, upstream, graph=graph)

    def test_subnormal_aggregate_across_reduction_boundary(self, subtests: pytest.Subtests):
        for rows in (4095, 4096, 4097):
            z = (
                torch.tensor([[0.0, -750.0]], device="cuda", dtype=torch.float64)
                .expand(rows, -1)
                .contiguous()
                .requires_grad_()
            )
            y = torch.zeros(rows, device="cuda", dtype=torch.int64)
            y[::11] = -100
            active = rows - (rows + 10) // 11
            with mp.workdps(100):
                tail = mp.exp(-750)
                expected_sum = float(active * mp.log1p(tail))
                expected_gradient = float(tail / (1 + tail) * mp.mpf(1e300) / active)
            with subtests.test(rows=rows):
                per_row = cross_entropy(z, y, reduction="none")
                assert bool((per_row == 0).all())
                total = cross_entropy(z, y, reduction="sum")
                assert total.item() > 0.0
                torch.testing.assert_close(
                    total, total.new_tensor(expected_sum), rtol=2e-12, atol=_QUANTUM
                )
                mean = cross_entropy(z, y)
                grad = torch.autograd.grad(mean, z, mean.new_tensor(1e300))[0]
                expected = torch.full_like(grad, expected_gradient)
                expected[:, 0].neg_()
                expected[y == -100] = 0
                torch.testing.assert_close(grad, expected, rtol=2e-12, atol=_QUANTUM)

    def test_subnormal_class_sum_across_chunk_boundary(self, subtests: pytest.Subtests):
        for classes in (4095, 4096, 4097):
            z = torch.full((2, classes), -750.0, device="cuda", dtype=torch.float64)
            z[:, 0] = 0
            z.requires_grad_()
            y = torch.tensor([0, -100], device="cuda")
            with mp.workdps(100):
                e = mp.exp(-750)
                tail = (classes - 1) * e
                expected_value = float(mp.log1p(tail))
                competitor = float(e / (1 + tail) * mp.mpf(1e300))
                target = float(-tail / (1 + tail) * mp.mpf(1e300))
            for reduction in ("none", "sum", "mean"):
                with subtests.test(classes=classes, reduction=reduction):
                    value = cross_entropy(z, y, reduction=reduction)
                    expected = value.new_tensor(
                        [expected_value, 0.0] if reduction == "none" else expected_value
                    )
                    torch.testing.assert_close(value, expected, rtol=2e-12, atol=_QUANTUM)
                    gradient = torch.autograd.grad(value, z, torch.full_like(value, 1e300))[0]
                    expected_grad = torch.full_like(z, competitor)
                    expected_grad[0, 0] = target
                    expected_grad[1] = 0
                    torch.testing.assert_close(gradient, expected_grad, rtol=2e-12, atol=_QUANTUM)

    def test_partial_reduction_ordinary_and_all_ignored(self, subtests: pytest.Subtests):
        torch.manual_seed(1729)
        for rows in (4095, 4097):
            for ignore in (-100, -(2**63), 2**63 - 1):
                x = torch.randn(rows, 3, device="cuda", dtype=torch.float64, requires_grad=True)
                y = torch.arange(rows, device="cuda") % 3
                y[::7] = ignore
                for all_ignored in (False, True):
                    if all_ignored:
                        y.fill_(ignore)
                    for reduction in ("none", "sum", "mean"):
                        with subtests.test(
                            rows=rows, ignore=ignore, all_ignored=all_ignored, reduction=reduction
                        ):
                            value = cross_entropy(x, y, ignore_index=ignore, reduction=reduction)
                            refx = x.detach().clone().requires_grad_()
                            reference = F.cross_entropy(
                                refx, y, ignore_index=ignore, reduction=reduction
                            )
                            gradient = torch.autograd.grad(value.sum(), x)[0]
                            refgrad = torch.autograd.grad(reference.sum(), refx)[0]
                            torch.testing.assert_close(
                                value, reference, rtol=3e-13, atol=3e-15, equal_nan=True
                            )
                            torch.testing.assert_close(
                                gradient, refgrad, rtol=3e-12, atol=3e-15, equal_nan=True
                            )

    def test_class_weighted_float64_fuses_with_full_range_weights(self, subtests: pytest.Subtests):
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [-1.0, 2.0, 0.0]],
            device="cuda",
            dtype=torch.float64,
            requires_grad=True,
        )
        y = torch.tensor([0, 1], device="cuda")
        weight = torch.tensor([1e308, 1e-320, 1.0], device="cuda", dtype=torch.float64)
        for reduction in ("none", "sum", "mean"):
            with subtests.test(reduction=reduction):
                with mock.patch.object(
                    _cross_entropy._WeightedEntropy,
                    "apply",
                    side_effect=AssertionError("float64 class weights fell back to tensors"),
                ):
                    value = cross_entropy(x, y, weight, reduction=reduction)
                reference = cross_entropy(x, y, weight, reduction=reduction, backend="torch")
                torch.testing.assert_close(value, reference, rtol=2e-12, atol=_QUANTUM)
                gradient = torch.autograd.grad(value.sum(), x)[0]
                expected = torch.autograd.grad(reference.sum(), x)[0]
                torch.testing.assert_close(gradient, expected, rtol=2e-12, atol=_QUANTUM)

    def test_weighted_float64_extreme_values_and_upstream(self, subtests: pytest.Subtests):
        from test_weighted_cuda import reference

        limit = torch.finfo(torch.float64).max
        cases = (
            ([[0.0, -1000.0], [0.0, -750.0]], [0, 0], [1e308, 1.0]),
            ([[0.0, -1e308], [0.0, -1e308]], [0, 1], [1e308, 1e-320]),
            ([[0.0, 0.0]] * 3, [0, 1, -100], [1e308, 1e308]),
            ([[-limit, limit], [limit, -limit]], [0, 0], [0.5, 1.0]),
            ([[0.0, 0.0], [0.0, 0.0]], [0, 1], [1e-320, 1e-320]),
        )
        for scores, labels, weights in cases:
            z = torch.tensor(scores, dtype=torch.float64)
            y = torch.tensor(labels)
            w = torch.tensor(weights, dtype=torch.float64)
            for reduction in ("none", "sum", "mean"):
                for scale in (0.0, 1.0, -3.0, 1e308):
                    u = torch.full((len(z),) if reduction == "none" else (), scale, dtype=z.dtype)
                    expected, expected_grad = reference(z, y, w, reduction, u)
                    for higher in (False, True):
                        with subtests.test(
                            scores=scores,
                            weights=weights,
                            reduction=reduction,
                            scale=scale,
                            higher=higher,
                        ):
                            x = z.cuda().requires_grad_()
                            value = cross_entropy(x, y.cuda(), w.cuda(), reduction=reduction)
                            gradient = torch.autograd.grad(value, x, u.cuda(), create_graph=higher)[
                                0
                            ]
                            torch.testing.assert_close(
                                value.cpu(), expected, rtol=2e-12, atol=_QUANTUM
                            )
                            torch.testing.assert_close(
                                gradient.cpu(), expected_grad, rtol=2e-12, atol=_QUANTUM
                            )

    def test_weighted_float64_partial_reduction_extreme_denominators(
        self, subtests: pytest.Subtests
    ):
        for rows in (4095, 4096, 4097):
            for magnitude in (1e308, 1e-320):
                with subtests.test(rows=rows, magnitude=magnitude):
                    x = torch.zeros(rows, 2, device="cuda", dtype=torch.float64, requires_grad=True)
                    y = torch.arange(rows, device="cuda") % 2
                    y[::7] = -100
                    w = torch.full((2,), magnitude, device="cuda", dtype=torch.float64)
                    value = cross_entropy(x, y, w)
                    gradient = torch.autograd.grad(value, x)[0]
                    active = rows - (rows + 6) // 7
                    expected = torch.full_like(x, 0.5 / active)
                    expected.scatter_(1, y.clamp(min=0)[:, None], -0.5 / active)
                    expected[y == -100] = 0
                    torch.testing.assert_close(
                        value, value.new_tensor(math.log(2)), rtol=2e-12, atol=0
                    )
                    torch.testing.assert_close(gradient, expected, rtol=2e-12, atol=_QUANTUM)

    def test_weighted_float64_amplifies_subnormal_loss_before_rounding(
        self, subtests: pytest.Subtests
    ):
        for classes in (2, 4097):
            for gap in (740.0, 745.0, 750.0):
                x = torch.full((2, classes), -gap, device="cuda", dtype=torch.float64)
                x[:, 0] = 0
                x.requires_grad_()
                y = torch.tensor([0, -100], device="cuda")
                w = torch.ones(classes, device="cuda", dtype=torch.float64)
                w[0] = 1e308
                for reduction in ("none", "sum", "mean"):
                    with mp.workdps(100):
                        e = mp.exp(-gap)
                        tail = (classes - 1) * e
                        factor = mp.mpf(1) if reduction == "mean" else mp.mpf(1e308)
                        expected_value = float(mp.log1p(tail) * factor)
                        competitor = float(e / (1 + tail) * factor)
                        target = float(-tail / (1 + tail) * factor)
                    for higher in (False, True):
                        with subtests.test(
                            classes=classes, gap=gap, reduction=reduction, higher=higher
                        ):
                            value = cross_entropy(x, y, w, reduction=reduction)
                            gradient = torch.autograd.grad(value.sum(), x, create_graph=higher)[0]
                            expected = value.new_tensor(
                                [expected_value, 0] if reduction == "none" else expected_value
                            )
                            expected_gradient = torch.full_like(x, competitor)
                            expected_gradient[0, 0] = target
                            expected_gradient[1] = 0
                            torch.testing.assert_close(value, expected, rtol=2e-12, atol=_QUANTUM)
                            torch.testing.assert_close(
                                gradient, expected_gradient, rtol=2e-12, atol=_QUANTUM
                            )

    def test_invalid_full64_rows_match_first_and_higher_backward(self, subtests: pytest.Subtests):
        for invalid in ("target", "logit", "ignored_logit"):
            for reduction in ("none", "sum", "mean"):
                x = torch.zeros(3, 3, device="cuda", dtype=torch.float64)
                y = torch.tensor([0, 1, -100], device="cuda")
                row = 2 if invalid == "ignored_logit" else 0
                if invalid == "target":
                    y[row] = 2**63 - 1
                else:
                    x[row, 0] = torch.nan
                x.requires_grad_()
                with subtests.test(invalid=invalid, reduction=reduction):
                    value = cross_entropy(x, y, reduction=reduction)
                    first = torch.autograd.grad(value.sum(), x, retain_graph=True)[0]
                    higher = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
                    torch.testing.assert_close(first, higher, equal_nan=True)
                    assert bool(higher[row].isnan().all())
                    if reduction != "none":
                        assert bool(higher.isnan().all())

    def test_cuda_graph_capture_replays_changed_ignored_targets(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x = torch.randn(17, 5, device="cuda", dtype=torch.float64, requires_grad=True)
            y = torch.arange(17, device="cuda") % 5
            y[::3] = -100
            for _ in range(3):
                cross_entropy(x, y).backward()
                x.grad = None
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                value = cross_entropy(x, y)
                value.backward()
        torch.cuda.current_stream().wait_stream(stream)
        for all_ignored in (False, True):
            if all_ignored:
                y.fill_(-100)
            else:
                y[::3] = 0
            graph.replay()
            torch.cuda.synchronize()
            refx = x.detach().clone().requires_grad_()
            reference = F.cross_entropy(refx, y)
            refgrad = torch.autograd.grad(reference, refx)[0]
            torch.testing.assert_close(value, reference, rtol=2e-13, atol=2e-15, equal_nan=True)
            torch.testing.assert_close(x.grad, refgrad, rtol=2e-12, atol=2e-15, equal_nan=True)
