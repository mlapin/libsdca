"""CUDA numerical, autograd, layout, validation and training regressions.

These exercise the tensor entropy and distillation APIs; test_tensor_losses.py
and test_tensor_prox.py compare the other tensor losses with the native ones.
SDCA training remains a CPU API.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from libsdca.torch import (
    DistillationLoss,
    distillation_loss,
    multiclass_hinge_loss,
    proximal_cross_entropy,
    topk_cross_entropy,
)
from named_losses import torch_loss, torch_module
from torch.autograd import gradcheck, gradgradcheck


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class CudaTests:
    def setup_method(self):
        torch.manual_seed(810)
        self.device = torch.device("cuda")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_wide_half_label_ce_with_double_work(self, dtype: torch.dtype) -> None:
        from libsdca.torch import cross_entropy

        rows, classes = 256, 4097
        scores = torch.zeros(rows, classes, device=self.device, dtype=dtype, requires_grad=True)
        labels = torch.arange(rows, device=self.device)
        value = cross_entropy(scores, labels, precision="float64")
        (gradient,) = torch.autograd.grad(value, scores)
        expected = torch.full_like(scores, 1 / (rows * classes))
        expected[labels, labels] = (1 / classes - 1) / rows
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
        torch.testing.assert_close(value, value.new_tensor(math.log(classes)), rtol=0, atol=0)

    @pytest.mark.parametrize("classes", [2, 64, 1000, 32000])
    @pytest.mark.parametrize("rows", [1, 3, 8])
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("reduction", ["none", "mean"])
    def test_double_label_smoothing_every_tiling(
        self, classes: int, rows: int, weighted: bool, reduction: str
    ) -> None:
        """float64 cross-entropy with label smoothing compiles and matches the float64
        reference on every tiling, including the one-row path (ROWS == 1), where reshaping
        the smoothed loss's scalar operands to (ROWS,) is a Triton compile error."""
        from libsdca.torch import cross_entropy
        from torch.nn import functional as F

        options = {"dtype": torch.float64, "device": self.device}
        scores = (3 * torch.randn(rows, classes, **options)).requires_grad_()
        labels = torch.randint(0, classes, (rows,), device=self.device)
        weight = torch.rand(classes, **options) + 0.5 if weighted else None
        call = {"weight": weight, "reduction": reduction, "label_smoothing": 0.1}
        value = cross_entropy(scores, labels, backend="kernel", **call)
        expected = F.cross_entropy(scores, labels, **call)
        torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)
        (gradient,) = torch.autograd.grad(value.sum(), scores)
        (reference,) = torch.autograd.grad(expected.sum(), scores)
        torch.testing.assert_close(gradient, reference, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("classes", [512, 1000, 4096, 32000])
    @pytest.mark.parametrize("smoothing", [0.0, 0.1])
    def test_probabilities_16bit_large_scores(
        self, dtype: torch.dtype, classes: int, smoothing: float
    ) -> None:
        """16-bit probability targets on the chunked path (>= 512 classes) at large scores:
        the gradient folds the row's maximum into a float32 exponent, whose rounding must stay
        within exp2's range for |max| > 2**30 / log2(e), or every other class is NaN. Rows of
        k tied maxima at +-V with the rest at 0
        (V > 0) or below the ties (V < 0); the exact gradient is T q - t, 0 off the ties."""
        from libsdca.torch import cross_entropy

        top = torch.finfo(dtype).max
        values = [v for v in (1e4, 1e10, 5.6e9, 1e20, 1e30, 1e37, 2e38, top) if v <= top]
        # Negative ties need room below them for the other classes: 1.5 |v| <= max.
        for v in values + [-x for x in values if 1.5 * x <= top]:
            for k in (1, 3):
                z = torch.zeros(2, classes, dtype=torch.float64)
                z[:, :k] = v
                if v < 0:
                    z[:, k:] = 1.5 * v
                z = z.to(dtype)
                t = torch.zeros(2, classes, dtype=torch.float64)
                t[:, :k] = 1 / k
                x = z.to(self.device).requires_grad_()
                target = t.to(dtype).to(self.device)
                value = cross_entropy(x, target, reduction="sum", label_smoothing=smoothing,
                                      backend="kernel")  # fmt: skip
                (g,) = torch.autograd.grad(value, x)
                assert torch.isfinite(g).all(), (v, k)
                # The exact gradient off the ties is the smoothing's share of the targets (the
                # other classes' probabilities underflow): T q - u with q = 0 there.
                u = (1 - smoothing) * target[:, k:].double() + smoothing / classes
                torch.testing.assert_close(g[:, k:].double(), -u, rtol=1e-2, atol=1e-30)
                # Each row sums to zero up to the targets' rounding.
                assert g.double().sum(1).abs().max().item() <= 1e-2, (v, k)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("classes", [1000, 2049, 4096, 32000])
    @pytest.mark.parametrize("margin", [87.0, 88.0, 95.0, 103.25])
    def test_probabilities_tiny_loss(self, dtype: torch.dtype, classes: int, margin: float) -> None:
        """A confident probability-target row, [margin (target 1), 0, ..., 0]: its loss
        log1p((C - 1) exp(-margin)) reaches float32's subnormal range. A chunked path that sums
        each chunk's float32 exponentials before scaling them flushes every term below
        exp(-87.3) of the chunk's maximum to 0: the loss would be 0 at 1000 classes (6.0e-36 at
        margin 88) and lack one chunk of terms at 32000. The label kernel leaves the label out
        of its chunks."""
        from libsdca.torch import cross_entropy

        z = torch.zeros(3, classes, dtype=dtype, device=self.device)
        z[:, 5] = margin
        t = torch.zeros_like(z)
        t[:, 5] = 1
        x = float(z[0, 5])
        exact = torch.tensor(math.log1p((classes - 1) * math.exp(-x)), dtype=torch.float32)
        value = cross_entropy(z, t, reduction="none", backend="kernel")
        labels = cross_entropy(z, torch.full((3,), 5, device=self.device), reduction="none",
                               backend="kernel")  # fmt: skip
        # A few float32 ulps times the margin (the float32 exponent's rounding: exp(-x) is
        # x-conditioned), or a few least subnormals.
        eps = torch.finfo(torch.float32).eps
        tolerance = max((4 + margin) * eps * exact.item(), 8 * 2.0**-149)
        for got in (value, labels):
            torch.testing.assert_close(got.cpu(), exact.expand(3), rtol=0, atol=tolerance)

    @pytest.mark.parametrize("classes", [1000, 4096, 32000])
    @pytest.mark.parametrize("first", [True, False])
    @pytest.mark.parametrize("margin", [100.0, 140.0, 160.0])
    @pytest.mark.parametrize("power", [0, 60, 100])
    def test_probabilities_weighted_tiny_tail(
        self, classes: int, first: bool, margin: float, power: int
    ) -> None:
        """A class weight scales a confident row's tiny loss, weight log1p((C - 1)
        exp(-margin)), back into float32's range, so even competitors far beyond exp(-147) of
        the maximum count (float32 scaled exponentials of the streamed row would lose them). The
        maximum in the first or the last tile."""
        from libsdca.torch import cross_entropy

        top = 0 if first else classes - 1
        z = torch.zeros(1, classes, device=self.device)
        z[0, top] = margin
        t = torch.zeros_like(z)
        t[0, top] = 1
        w = torch.ones(classes, device=self.device)
        w[top] = 2.0**power
        exact = 2.0**power * math.log1p((classes - 1) * math.exp(-margin))
        expected = torch.tensor([exact], dtype=torch.float32)
        value = cross_entropy(z, t, weight=w, reduction="none", backend="kernel").cpu()
        eps = torch.finfo(torch.float32).eps
        tolerance = max((4 + margin) * eps * expected.item(), 8 * 2.0**-149)
        torch.testing.assert_close(value, expected, rtol=0, atol=tolerance)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("recompute", [False, True])
    @pytest.mark.parametrize("classes", [1001, 1024])
    def test_linear_padded_vocabulary(
        self, dtype: torch.dtype, reduction: str, recompute: bool, classes: int
    ) -> None:
        """Class counts that are not multiples of 8 run on zero-padded weight rows (aligned
        GEMMs); values and all three gradients match the unpadded float64 reference, with
        class weights, ignored rows, a bias and several chunks."""
        from libsdca.torch import linear_cross_entropy
        from torch.nn import functional as F

        rows, features = 300, 64
        hidden = torch.randn(rows, features, device=self.device, dtype=dtype, requires_grad=True)
        weight = (torch.randn(classes, features, device=self.device) * 0.1).to(dtype)
        weight.requires_grad_()
        bias = torch.randn(classes, device=self.device, dtype=dtype, requires_grad=True)
        target = torch.randint(0, classes, (rows,), device=self.device)
        target[::7] = -100
        cw = torch.rand(classes, device=self.device, dtype=torch.float32) + 0.5
        loss = linear_cross_entropy(
            hidden, weight, target, bias=bias, class_weight=cw, reduction=reduction,
            chunk_bytes=128 * classes * dtype.itemsize, recompute=recompute,
        )  # fmt: skip
        up = torch.rand_like(loss) if reduction == "none" else torch.ones_like(loss)
        grads = torch.autograd.grad(loss, (hidden, weight, bias), up)
        h, w, b = (t.detach().double().requires_grad_() for t in (hidden, weight, bias))
        ref = F.cross_entropy(h @ w.T + b, target, weight=cw.double(), reduction=reduction)
        expected = torch.autograd.grad(ref, (h, w, b), up.double())
        # BF16 storage rounds each result once (2^-8 of its scale), float32 to about 1e-6.
        unit = 2.0**-7 if dtype == torch.bfloat16 else 1e-5
        for got, want in ((loss, ref), *zip(grads, expected, strict=True)):
            scale = float(want.abs().max()) or 1.0
            torch.testing.assert_close(got.double(), want.detach(), rtol=0, atol=unit * scale)
        for got, src in zip(grads, (hidden, weight, bias), strict=True):
            assert got.shape == src.shape and got.dtype == src.dtype and got.is_contiguous()

    def test_linear_padded_vocabulary_compiled(self) -> None:
        from libsdca.torch import linear_cross_entropy

        hidden = torch.randn(64, 32, device=self.device, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(1001, 32, device=self.device, dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, 1001, (64,), device=self.device)
        eager = torch.autograd.grad(linear_cross_entropy(hidden, weight, target), (hidden, weight))
        step = torch.compile(lambda h, w: linear_cross_entropy(h, w, target), fullgraph=True)
        compiled = torch.autograd.grad(step(hidden, weight), (hidden, weight))
        for a, b in zip(eager, compiled, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    @pytest.mark.skipif(not hasattr(torch.ops.aten.mm, "dtype"), reason="needs FP32 GEMM output")
    def test_linear_hidden_pullback_rounds_after_scaling(self) -> None:
        from libsdca.torch import cross_entropy, linear_cross_entropy

        h = torch.randn(16, 32, device=self.device, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(257, 32, device=self.device, dtype=torch.bfloat16)
        y = torch.arange(16, device=self.device)
        scores = torch.nn.functional.linear(h, w).detach().requires_grad_()
        (g,) = torch.autograd.grad(cross_entropy(scores, y), scores)
        product = torch.mm(g, w, out_dtype=torch.float32)
        loss = linear_cross_entropy(h, w, y)
        for upstream in (1.0, -0.73, 0.03125):
            scale = loss.new_tensor(upstream)
            (actual,) = torch.autograd.grad(loss, h, scale, retain_graph=True)
            expected = (product * scale).to(h.dtype)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.skipif(not hasattr(torch.ops.aten.mm, "dtype"), reason="needs FP32 GEMM output")
    def test_large_linear_bfloat_weight_pullback_with_multiple_consumers(self) -> None:
        from libsdca.torch import cross_entropy, linear_cross_entropy

        h = torch.randn(5, 512, dtype=torch.bfloat16, device=self.device)
        w = torch.randn(32768, 512, dtype=torch.bfloat16, device=self.device, requires_grad=True)
        y = torch.arange(5, device=self.device)
        z = torch.nn.functional.linear(h, w.detach()).requires_grad_()
        (row_gradient,) = torch.autograd.grad(cross_entropy(z, y), z)
        product = torch.mm(row_gradient.T, h, out_dtype=torch.float32)
        loss = linear_cross_entropy(h, w, y)
        (branch,) = torch.autograd.grad(loss * 0.125 - loss * 1.5, w, retain_graph=True)
        torch.testing.assert_close(branch, (product * -1.375).to(w.dtype), rtol=0, atol=0)
        upstreams = loss.new_tensor([1.0, -0.75, 0.0])
        (batched,) = torch.autograd.grad(
            loss, w, upstreams, is_grads_batched=True, retain_graph=True
        )
        for up, actual in zip(upstreams, batched, strict=True):
            torch.testing.assert_close(actual, (product * up).to(w.dtype), rtol=0, atol=0)
        (repeat,) = torch.autograd.grad(loss, w)
        torch.testing.assert_close(repeat, product.to(w.dtype), rtol=0, atol=0)

    def test_large_linear_scale_native_bfloat_conversion(self) -> None:
        from libsdca import _torch_linear

        pytest.importorskip("triton")
        from libsdca.triton.common import native_bfloat_conversion

        if not native_bfloat_conversion(torch.cuda):
            pytest.skip("needs native CUDA BF16 conversion")
        capacity = getattr(torch.cuda.get_device_properties(self.device), "L2_cache_size", 0)
        if capacity <= 0:
            pytest.skip("CUDA does not report the L2 cache size")
        n = (capacity + 5) // 6 + 17
        scores = torch.randn(n, device=self.device)
        scores[:6] = scores.new_tensor([0.0, -0.0, float("inf"), -float("inf"), 2**-126, 2**-128])
        factor = scores.new_tensor(-0.75)
        with torch.no_grad():
            assert not _torch_linear._dense_bfloat_scale(scores[:1], torch.bfloat16)
            assert _torch_linear._dense_bfloat_scale(scores, torch.bfloat16)
            actual = _torch_linear._scale_saved(scores, factor, torch.bfloat16)
            expected = torch.mul(scores, factor).to(torch.bfloat16)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    def test_large_linear_scale_without_triton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from libsdca import _torch_linear

        monkeypatch.setitem(sys.modules, "libsdca.triton.launch_torch", None)
        g = torch.randn(1 << 24, device=self.device)
        factor = g.new_tensor(-0.73)
        with torch.no_grad():
            actual = _torch_linear._scale_saved(g, factor, torch.bfloat16)
        torch.testing.assert_close(actual, (g * factor).bfloat16(), rtol=0, atol=0)

    def test_linear_saved_gradient_batched_cotangents(self) -> None:
        from libsdca.torch import linear_cross_entropy

        hidden = torch.randn(7, 13, dtype=torch.bfloat16, device=self.device, requires_grad=True)
        weight = torch.randn(17, 13, dtype=torch.bfloat16, device=self.device, requires_grad=True)
        target = torch.arange(7, device=self.device)
        loss = linear_cross_entropy(hidden, weight, target, chunk_bytes=2 * 3 * 17)
        upstreams = loss.new_tensor([1.0, -0.875, 0.0])
        separate = [
            torch.autograd.grad(loss, (hidden, weight), up, retain_graph=True) for up in upstreams
        ]
        batched = torch.autograd.grad(loss, (hidden, weight), upstreams, is_grads_batched=True)
        for gradients, actual in zip(zip(*separate, strict=True), batched, strict=True):
            expected = torch.stack(gradients)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    @pytest.mark.parametrize(
        ("rows", "classes", "dtype"),
        [(131072, 10, torch.bfloat16), (64, 1000, torch.float32), (8, 113, torch.float32)],
    )
    def test_compiled_short_label_retained_divisor(
        self, rows: int, classes: int, dtype: torch.dtype
    ) -> None:
        from libsdca.torch import cross_entropy

        z = torch.randn(rows, classes, device=self.device, dtype=dtype)
        y = torch.randint(classes, (len(z),), device=self.device)
        y[::5] = -100
        z[1::5, -3:] = -torch.inf
        z[1::5, 0] = 80
        y[1::5] = 0
        eager = z.detach().requires_grad_()
        traced = z.detach().requires_grad_()
        expected = cross_entropy(eager, y)
        actual = torch.compile(cross_entropy, fullgraph=True, dynamic=False)(traced, y)
        torch.testing.assert_close(actual, expected)
        for scale in (0.0, -1.3, 0.03125):
            upstream = torch.full_like(actual, scale)
            (a,) = torch.autograd.grad(expected, eager, upstream, retain_graph=True)
            (b,) = torch.autograd.grad(actual, traced, upstream, retain_graph=True)
            torch.testing.assert_close(b, a, rtol=0, atol=0)

    @pytest.mark.parametrize("compiled", [False, True])
    @pytest.mark.parametrize("classes", [1000, 8193])
    @pytest.mark.parametrize("reduced", [False, True])
    def test_hinge_winners_and_retained_cotangents(
        self, compiled: bool, classes: int, reduced: bool
    ) -> None:
        mid = classes // 2
        z = torch.full((512, classes), -8.0, device=self.device)
        y = torch.zeros(len(z), dtype=torch.int64, device=self.device)
        z[:, 0] = 0
        z[0::8, -1] = 3
        z[1::8, mid - 1 : mid + 1] = 2
        z[2::8, [1, mid, -1]] = -1
        z[4::8, 0] = -torch.inf
        z[4::8, mid] = 2
        z[5::8, 1:] = -torch.inf
        z[6::8] = -torch.inf
        z[7::8, -1] = torch.nan
        expected = torch.tensor(
            [4, 3, 0, 0, torch.inf, 0, torch.nan, torch.nan], device=self.device
        )
        slope = torch.zeros_like(z)
        slope[0::8, 0], slope[0::8, -1] = -1, 1
        slope[1::8, 0], slope[1::8, mid - 1 : mid + 1] = -1, 0.5
        slope[2::8, 0], slope[2::8, [1, mid, -1]] = -0.75, 0.25
        slope[4::8, 0], slope[4::8, mid] = -1, 1
        slope[6::8], slope[7::8] = torch.nan, torch.nan

        def loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return multiclass_hinge_loss(scores, labels, reduction="mean" if reduced else "none")

        z.requires_grad_()
        op = torch.compile(loss, fullgraph=True, dynamic=False) if compiled else loss
        value = op(z, y)
        rows = expected.repeat(64)
        torch.testing.assert_close(
            value, rows.mean() if reduced else rows, rtol=0, atol=0, equal_nan=True
        )
        for factor in (1.0, -0.375, 0.0):
            up = torch.full_like(value, factor)
            if reduced:
                scaled = slope * (up / len(z))
            else:
                up[1::2] *= -2
                scaled = slope * up[:, None]
            (gradient,) = torch.autograd.grad(value, z, up, retain_graph=True)
            torch.testing.assert_close(gradient, scaled, rtol=0, atol=0, equal_nan=True)
            assert not torch.any((gradient == 0) & torch.signbit(gradient))

    @pytest.mark.parametrize("classes", [31999, 32001])
    def test_streamed_hinge_winner_at_last_column(self, classes: int) -> None:
        # Neighboring widths exposed a memory fault in the former tiled winner
        # plan. Use an odd row count, a partial final tile, and retained scores.
        z = torch.zeros((13, classes), device=self.device)
        z[:, -1] = 2.0
        z[1::2, 511] = 2.0
        z.requires_grad_()
        labels = torch.zeros(len(z), device=self.device, dtype=torch.int64)
        value = multiclass_hinge_loss(z, labels, reduction="none")
        torch.testing.assert_close(value, torch.full_like(value, 3.0), atol=0, rtol=0)
        for sign in (1.0, -1.0):
            up = torch.full_like(value, sign)
            (grad,) = torch.autograd.grad(value, z, up, retain_graph=True)
            expected = torch.zeros_like(z)
            expected[:, 0], expected[:, -1] = -sign, sign
            expected[1::2, -1], expected[1::2, 511] = 0.5 * sign, 0.5 * sign
            torch.testing.assert_close(grad, expected, atol=0, rtol=0)

    @pytest.mark.parametrize("classes", [3, 1000])
    def test_scaled_cross_entropy_tail(self, classes: int) -> None:
        from libsdca.torch import cross_entropy

        z = torch.zeros((3, classes), device=self.device)
        z[:, 0] = 120
        z.requires_grad_()
        y = torch.tensor([0, 0, -100], device=self.device)
        v = cross_entropy(z, y, reduction="none", backend="kernel")
        up = torch.tensor([2.0**60, -(2.0**60), 1.0], device=self.device)
        (g,) = torch.autograd.grad(v, z, up)
        tail = math.exp(-120) * 2**60
        expected = torch.full_like(g[0], tail)
        expected[0] = -(classes - 1) * tail
        torch.testing.assert_close(g[0], expected, rtol=4e-7, atol=0)
        torch.testing.assert_close(g[1], -g[0], rtol=0, atol=0)
        assert g[2].count_nonzero() == 0

    @pytest.mark.parametrize(("rows", "classes"), [(64, 16401), (512, 1000)])
    def test_multilabel_entropy_mixed_rows(self, rows: int, classes: int) -> None:
        from libsdca.torch import multilabel_cross_entropy

        positives = 32
        z = torch.zeros((rows, classes), device=self.device)
        mask = torch.zeros_like(z, dtype=torch.bool)
        mask[:, 1 : positives + 1] = True
        z[:, 0] = math.log(2 * classes / positives)
        z[:, -1] = -torch.inf
        # A negative maximum, with the first positive either far from or
        # almost exactly at probability 1 / positives.
        z[1::4, 1] = math.log((math.exp(z[0, 0].item()) + classes - 3) / (positives - 1))
        # A confident singleton retains its tiny top gradient. A masked
        # positive has infinite loss and still receives its negative target.
        z[2::4, 1] = 70
        mask[2::4, 2 : positives + 1] = False
        mask[3::4, -1] = True
        up = torch.tensor([1, 0.25, -2, 1024], device=self.device).repeat(rows // 4)
        z.requires_grad_()
        value = multilabel_cross_entropy(z, mask, reduction="none", backend="kernel")
        (gradient,) = torch.autograd.grad(value, z, up)

        x = z.detach().double()
        top, index = x.max(1, keepdim=True)
        p = (x - top).exp()
        tail = p.scatter(1, index, 0).sum(1, keepdim=True)
        count = mask.sum(1, keepdim=True)
        gap = torch.where(mask, top - x, 0).sum(1, keepdim=True) / count
        expected = p / (1 + tail) - mask.double() / count
        expected[2::4, 1:2] = -tail[2::4] / (1 + tail[2::4])
        torch.testing.assert_close(value, (gap + tail.log1p()).flatten().float(), rtol=2e-7, atol=0)
        expected = expected * up[:, None]
        # Rows 2::4 and 3::4 (a positive at the maximum, a masked positive) take float64 work.
        torch.testing.assert_close(gradient[2::4], expected[2::4].float(), rtol=4e-6, atol=0)
        torch.testing.assert_close(gradient[3::4], expected[3::4].float(), rtol=4e-6, atol=0)
        # Rows 0::4 and 1::4 have a negative class at the maximum: their float32 pass is final
        # (public's normwise rule, docs/API.md known limits: 64 eps of the row's largest
        # entry). Row 1's first positive sits within 1e-7 of its target 1 / 32: p - 1/32 =
        # 1.2e-9, below one float32 rounding of p (up to 9e-10). Each entry is held to 64 eps
        # of its terms |up| (p_j + t_j), at most the row's largest entry, not of itself.
        lean = torch.cat([torch.arange(0, rows, 4), torch.arange(1, rows, 4)]).to(z.device)
        error = (gradient[lean].double() - expected[lean]).abs()
        terms = up[lean, None].abs() * (p[lean] / (1 + tail[lean]) + mask[lean] / count[lean])
        largest = expected[lean].abs().amax(1, keepdim=True)
        bound = 64 * torch.finfo(torch.float32).eps * torch.minimum(terms, largest)
        assert (error <= bound).all(), (error / bound).max()

    def test_wide_multilabel_entropy_small_positive_residual(self) -> None:
        from libsdca.torch import multilabel_cross_entropy

        classes, count = 16401, 7
        z = torch.zeros((64, classes), device=self.device)
        z[:, ::3] = -4.75
        z[:, 0] = math.log(3 * classes / count)
        z[:, 1] = -torch.inf
        other = z[0].double().exp().sum()
        # Place the first positive on either side of its uniform target.
        # Round this constructed score to the stored dtype before the oracle.
        delta = torch.tensor([0.064, -0.064, 0.016, -0.016], device=self.device).double().repeat(16)
        z[:, 1] = ((1 + delta) * other / (count - 1 - delta)).log()
        mask = torch.zeros_like(z, dtype=torch.bool)
        mask[:, 1 : count + 1] = True
        z.requires_grad_()
        values = multilabel_cross_entropy(z, mask, reduction="none", backend="kernel")
        (gradient,) = torch.autograd.grad(values.sum(), z)
        p = z.detach().double().softmax(1)[:, 1]
        # The maximum is a negative class (z[:, 0]): the float32 pass is final, within 64 eps
        # of p (public's normwise rule, docs/API.md known limits: 64 eps of the row's largest
        # entry, here p_0 > p). |p - 1 / 7| is 2.3e-3 to 9.1e-3, and one float32 rounding of p
        # (up to 7.5e-9) alone exceeds 3e-7 of it.
        error = (gradient[:, 1].double() - (p - 1 / count)).abs()
        assert (error <= 64 * torch.finfo(torch.float32).eps * p).all()
        assert torch.equal(torch.sign(gradient[:, 1]), torch.sign(delta).float())

    @pytest.mark.parametrize("rows", [64, 1024])
    def test_wide_bfloat_multilabel_retained_gradients(self, rows: int) -> None:
        from libsdca.torch import multilabel_cross_entropy

        classes, count = 32000, 50
        z = torch.zeros((rows, classes), dtype=torch.bfloat16, device=self.device)
        mask = torch.zeros_like(z, dtype=torch.bool)
        z[::2, :10], z[::2, 10:30], z[::2, 30:100] = 4.0, -4.0, 7.0
        mask[::2, :8] = True
        # The positive gradients of these rows straddle their uniform target.
        boundary = math.log((classes - count + (count - 1) * math.exp(-0.75)) / (count - 1))
        x = torch.tensor(boundary, dtype=z.dtype, device=z.device)
        z[1::2, 1:count] = -0.75
        z[1::4, 0] = torch.nextafter(x, torch.full_like(x, math.inf))
        z[3::4, 0] = torch.nextafter(x, torch.full_like(x, -math.inf))
        mask[1::2, :count] = True
        original = z.clone()
        z.requires_grad_()
        value = multilabel_cross_entropy(z, mask, reduction="none", backend="kernel")
        scores = original[:4].double()
        target = mask[:4].double() / mask[:4].sum(1, keepdim=True)
        expected = scores.logsumexp(1) - (scores * target).sum(1)
        torch.testing.assert_close(
            value.reshape(-1, 4), expected.float().expand(rows // 4, -1), rtol=2e-7, atol=0
        )
        gradient = scores.softmax(1) - target
        up = torch.tensor([1.0, -0.375, 0.0, 2.0], device=self.device).repeat(rows // 4)
        for seed in (up, -up):
            (actual,) = torch.autograd.grad(value, z, seed, retain_graph=True)
            torch.testing.assert_close(
                actual.reshape(-1, 4, classes),
                (gradient * seed[:4, None]).to(z.dtype).expand(rows // 4, -1, -1),
                rtol=0,
                atol=0,
            )
        assert torch.equal(z, original)

    @pytest.mark.parametrize("repeats", [1, 820])
    def test_bfloat_multilabel_positive_residual(self, repeats: int) -> None:
        from libsdca.torch import multilabel_cross_entropy
        from masked_cases import bfloat_multilabel_positive_residual

        x, m, u, reference = bfloat_multilabel_positive_residual()
        x, m = np.tile(x, (repeats, 1)), np.tile(m, (repeats, 1))
        u, reference = np.tile(u, repeats), np.tile(reference, repeats)
        scores = torch.as_tensor(x, device=self.device, dtype=torch.bfloat16).requires_grad_()
        mask = torch.as_tensor(m, device=self.device)
        upstream = torch.as_tensor(u, device=self.device)
        values = multilabel_cross_entropy(scores, mask, reduction="none", backend="kernel")
        assert torch.isposinf(values[2])
        for sign in (1, -1):
            (gradient,) = torch.autograd.grad(values, scores, upstream * sign, retain_graph=True)
            expected = torch.as_tensor(reference * u * sign, device=self.device).bfloat16()
            torch.testing.assert_close(gradient[:, 0], expected, rtol=0, atol=0)
        torch.testing.assert_close(scores.float(), torch.as_tensor(x, device=self.device))

    @pytest.mark.parametrize("classes", [7, 32000])
    def test_bfloat_multilabel_amplified_tail(self, classes: int) -> None:
        from libsdca.torch import multilabel_cross_entropy
        from masked_cases import amplified_multilabel_tail

        x, target, upstream, reference = amplified_multilabel_tail(classes)
        z = torch.tensor(x, device=self.device, dtype=torch.bfloat16, requires_grad=True)
        mask = torch.tensor(target, device=self.device)
        u = torch.tensor(upstream, device=self.device)
        expected = torch.tensor(reference, device=self.device).to(z.dtype)
        v = multilabel_cross_entropy(z, mask, reduction="none", backend="kernel")
        for sign in (1, -1):
            (g,) = torch.autograd.grad(v, z, sign * u, retain_graph=True)
            torch.testing.assert_close(g[:, 1], sign * expected, atol=0, rtol=0)
            assert not torch.signbit(g[7::8, 1]).any()
            assert not torch.signbit(g[-4, 1])
            assert bool(torch.signbit(g[5, 1])) == (sign == 1)

    def test_entropy_dtype_value_and_gradient_reference(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            z = torch.randn(4, 7, device=self.device, dtype=dtype, requires_grad=True)
            target = torch.tensor([0, 2, 4, 6], device=self.device)
            weights = torch.tensor([0.0, -2.0, 0.5, 3.0], device=self.device, dtype=dtype)
            actual = torch_loss(z, target, backend="composed", reduction="none")
            actual.backward(weights)
            ref = z.detach().cpu().double().requires_grad_()
            expected = torch.nn.functional.cross_entropy(ref, target.cpu(), reduction="none")
            expected.backward(weights.cpu().double())
            tol = {
                torch.float16: 2e-3,
                torch.bfloat16: 1e-2,
                torch.float32: 2e-6,
                torch.float64: 2e-13,
            }[dtype]
            torch.testing.assert_close(actual.cpu().double(), expected, rtol=tol, atol=tol / 10)
            assert z.grad is not None
            torch.testing.assert_close(z.grad.cpu().double(), ref.grad, rtol=tol, atol=tol / 10)
            assert z.grad.dtype == dtype
            assert z.grad[0].abs().sum().item() == 0

    def test_collapse_across_dtypes_and_reductions(self):
        for dtype, margin in [
            (torch.float16, 12.0),
            (torch.bfloat16, 12.0),
            (torch.float32, 40.0),
            (torch.float64, 100.0),
        ]:
            for reduction in ["none", "sum", "mean"]:
                z = torch.tensor(
                    [[margin, 0.0, 0.0], [margin, 0.0, 0.0]],
                    device=self.device,
                    dtype=dtype,
                    requires_grad=True,
                )
                actual = torch_loss(
                    z,
                    torch.tensor([0, 0], device=self.device),
                    backend="composed",
                    reduction=reduction,
                )
                actual.sum().backward()
                tail = 2 * torch.exp(torch.tensor(-margin, dtype=torch.float64))
                expected = tail / (1 + tail) / (2 if reduction == "mean" else 1)
                tol = 0.02 if dtype in [torch.float16, torch.bfloat16] else 2e-6
                assert (actual > 0).all().item()
                assert z.grad is not None
                assert (z.grad[:, 0] < 0).all().item()
                torch.testing.assert_close(
                    -z.grad[:, 0].cpu().double(), expected.expand(2), rtol=tol, atol=0
                )
                # Stored half gradients can round each entry by one subnormal
                # unit; zero-sum is required to storage precision.
                info = torch.finfo(dtype)
                torch.testing.assert_close(
                    z.grad[:, 0],
                    -z.grad[:, 1:].sum(-1),
                    rtol=4 * info.eps,
                    atol=2 * info.tiny * info.eps,
                )

    def test_multilabel_small_differences_and_tail(self):
        for dtype in [torch.float32, torch.float64]:
            z = torch.tensor(
                [[0.0, 1e-10, -1e-10], [20.0, 20.0, -20.0]],
                device=self.device,
                dtype=dtype,
                requires_grad=True,
            )
            mask = torch.tensor([[True, True, True], [True, True, False]], device=self.device)
            torch_loss(z, mask, backend="composed", reduction="sum").backward()
            assert z.grad is not None
            torch.testing.assert_close(z.grad[0], z.detach()[0] / 3, rtol=2e-6, atol=1e-20)
            assert z.grad[1, 0].item() < 0
            assert z.grad[1, 0].item() == z.grad[1, 1].item()
            torch.testing.assert_close(2 * z.grad[1, 0], -z.grad[1, 2], rtol=2e-6, atol=0)

    def test_noncontiguous_layout_shift_and_permutation(self):
        z = (
            torch.randn(2, 3, 5, device=self.device, dtype=torch.float64)
            .transpose(1, 2)
            .requires_grad_()
        )
        assert not z.is_contiguous()
        labels = torch.tensor([[0, 1, 2], [3, 4, 0]], device=self.device)
        values = torch_loss(z, labels, backend="composed", dim=1, reduction="none")
        values.sum().backward()
        reference = torch.nn.functional.cross_entropy(z.detach(), labels, reduction="none")
        torch.testing.assert_close(values, reference)
        permutation = torch.tensor([2, 4, 0, 1, 3], device=self.device)
        inverse = torch.argsort(permutation)
        shifted = (z.detach()[:, permutation, :] + 1024).requires_grad_()
        changed = torch_loss(shifted, inverse[labels], backend="composed", dim=-2, reduction="sum")
        changed.backward()
        torch.testing.assert_close(changed, values.sum(), atol=1e-12, rtol=1e-12)
        assert z.grad is not None
        torch.testing.assert_close(shifted.grad, z.grad[:, permutation, :], atol=1e-13, rtol=1e-12)

    def test_entropy_first_and_second_derivatives_for_masks(self):
        z = torch.randn(2, 4, device=self.device, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor(
            [[True, True, False, False], [False, True, False, True]], device=self.device
        )

        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch_loss(x, mask, backend="composed")

        assert gradcheck(fn, (z,))
        assert gradgradcheck(fn, (z,))

    def test_distillation_equal_near_far_temperature_and_teacher(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            for temperature in [0.5, 1.0, 3.0]:
                teacher = torch.tensor(
                    [[2.0, 1.0, 0.0]], device=self.device, dtype=dtype, requires_grad=True
                )
                student = teacher.detach().clone().requires_grad_()
                DistillationLoss(temperature=temperature)(student, teacher).backward()
                torch.testing.assert_close(student.grad, torch.zeros_like(student), atol=0, rtol=0)
                t = teacher.detach().cpu().double() / temperature
                p = t.softmax(-1)
                teacher_grad = p * ((p * t).sum(-1, keepdim=True) - t) / temperature
                assert teacher.grad is not None
                torch.testing.assert_close(teacher.grad.cpu(), teacher_grad.to(dtype))
                eps = {
                    torch.float16: 0.01,
                    torch.bfloat16: 0.05,
                    torch.float32: 1e-5,
                    torch.float64: 1e-8,
                }[dtype]
                for perturbation in [eps, 10.0]:
                    teacher.grad = None
                    student = (
                        teacher.detach()
                        + torch.tensor(
                            [[perturbation, -perturbation, 0.0]], device=self.device, dtype=dtype
                        )
                    ).requires_grad_()
                    value = distillation_loss(
                        student, teacher, temperature=temperature, reduction="sum"
                    )
                    value.backward()
                    s = student.detach().cpu().double() / temperature
                    expected = (s.softmax(-1) - t.softmax(-1)) / temperature
                    tol = 0.04 if dtype in [torch.float16, torch.bfloat16] else 2e-5
                    assert student.grad is not None
                    scale = float(expected.abs().max())
                    torch.testing.assert_close(
                        student.grad.cpu().double(), expected, rtol=tol, atol=tol * scale
                    )
                    assert torch.isfinite(value).item()
                    teacher_grad = p * ((p * s).sum(-1, keepdim=True) - s) / temperature
                    assert teacher.grad is not None
                    torch.testing.assert_close(teacher.grad.cpu(), teacher_grad.to(dtype))
        student = torch.tensor([[-1000.0, 1000.0, 0.0]], device=self.device, requires_grad=True)
        teacher = -student.detach()
        distillation_loss(student, teacher).backward()
        torch.testing.assert_close(
            student.grad, torch.tensor([[-1.0, 1.0, 0.0]], device=self.device), rtol=0, atol=0
        )

    def test_short_distillation_probability_difference(self):
        import numpy as np

        rng = np.random.default_rng(1729)
        scores = 3 * rng.standard_normal((64, 1000), dtype=np.float32)
        teacher = scores + 0.5 * rng.standard_normal(scores.shape, dtype=np.float32)
        z = torch.tensor(scores, device=self.device, requires_grad=True)
        t = torch.tensor(teacher, device=self.device)
        value = distillation_loss(z, t, temperature=2.0, backend="kernel")
        (gradient,) = torch.autograd.grad(value, z)
        q = torch.softmax(z.detach().double() / 2, dim=1)
        p = torch.softmax(t.double() / 2, dim=1)
        denominator = 2 * z.shape[0]
        reference = (q - p) / denominator
        scale = (q + p).amax(dim=1) / denominator
        error = (gradient.double() - reference).abs().amax(dim=1)
        assert (error / scale).max().item() <= 0.75 * torch.finfo(z.dtype).eps

    def test_bfloat_distillation_reduced_rounding_and_scaled_gradient(self):
        import numpy as np

        rng = np.random.default_rng(0)
        scores = 3 * rng.standard_normal((8, 100), dtype=np.float32)
        teacher = scores + 0.5 * rng.standard_normal(scores.shape, dtype=np.float32)
        z = torch.tensor(scores, device=self.device, dtype=torch.bfloat16).repeat(64, 1)
        t = torch.tensor(teacher, device=self.device, dtype=z.dtype).repeat(64, 1)
        z.requires_grad_()
        value = distillation_loss(z, t, temperature=2.0, backend="kernel")
        scaled = torch.tensor(-0.75, device=self.device)
        (gradient,) = torch.autograd.grad(value, z, scaled)
        student = z[:8].detach().double() / 2
        p = (t[:8].double() / 2).softmax(dim=1)
        expected = -(p * student.log_softmax(dim=1)).sum(dim=1).mean()
        torch.testing.assert_close(value, expected.float(), atol=0, rtol=0)
        reference = (student.softmax(dim=1) - p) * (scaled.double() / (2 * len(z)))
        torch.testing.assert_close(gradient[:8], reference.to(z.dtype), atol=0, rtol=0)
        assert torch.equal(gradient, gradient[:8].repeat(64, 1))

    def test_short_bfloat_distillation_mixed_rows(self):
        # Batching ordinary, identical, sparse and masked rows must preserve each row's result.
        for classes in (9, 10):
            generator = torch.Generator().manual_seed(classes)
            z = torch.randn(19, classes, generator=generator).to(self.device, torch.bfloat16)
            t = torch.randn(19, classes, generator=generator).to(self.device, torch.bfloat16)
            t[1::4] = z[1::4]
            t[2::4, 1:] = z[2::4, 1:]
            z[3::4, 0] = t[3::4, 0] = -torch.inf
            upstream = torch.linspace(-1.5, 2.0, len(z), device=self.device)

            def evaluate(
                student: torch.Tensor, teacher: torch.Tensor, scale: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                student = student.detach().requires_grad_()
                values = distillation_loss(
                    student, teacher, temperature=2.0, reduction="none", backend="kernel"
                )
                (gradient,) = torch.autograd.grad(values, student, scale)
                return values.detach(), gradient

            values, gradients = evaluate(z, t, upstream)
            assert torch.count_nonzero(gradients[1::4]).item() == 0
            for i in range(len(z)):
                row = slice(i, i + 1)
                value, gradient = evaluate(z[row], t[row], upstream[row])
                torch.testing.assert_close(values[row], value, atol=0, rtol=0)
                torch.testing.assert_close(gradients[row], gradient, atol=0, rtol=0)

    def test_sparse_bfloat_distillation_permutation_and_quadratic_residual(self):
        import math

        n, c = 512, 96
        z = torch.zeros((n, c), device=self.device, dtype=torch.bfloat16)
        z[::2, 0], z[1::2, 0] = 1.0, 13.25
        t = z.clone()
        t[::2, 0], t[::2, 1] = 0.0, 1.0
        delta = 2.0**-23
        t[1::2, 1], t[1::2, 2] = delta, -delta
        z.requires_grad_()
        value = distillation_loss(z, t, temperature=2.0, reduction="none", backend="kernel")
        factor = 2.0**60
        (gradient,) = torch.autograd.grad(value, z, torch.full_like(value, factor))

        expected = torch.zeros((2, c), dtype=torch.float64)
        swap = math.expm1(0.5) * factor / (2 * (c + math.expm1(0.5)))
        expected[0, 0], expected[0, 1] = swap, -swap
        mass = math.exp(13.25 / 2)
        total = mass + c - 1
        # exp(d) + exp(-d) - 2, without subtracting nearly equal exponentials.
        change = 4 * math.sinh(delta / 4) ** 2
        scale = factor / (2 * total * (total + change))
        expected[1].fill_(scale * change)
        expected[1, 0] *= mass
        expected[1, 1] = scale * (change - total * math.expm1(delta / 2))
        expected[1, 2] = scale * (change - total * math.expm1(-delta / 2))
        torch.testing.assert_close(
            gradient.cpu(), expected.to(z.dtype).repeat(n // 2, 1), atol=0, rtol=0
        )

    def test_bounded_distillation_with_shared_and_one_sided_masks(self):
        # These dimensions use the bounded row statistics and the saved-gradient path.
        z = torch.randn(512, 1000, device=self.device)
        t = torch.randn_like(z)
        z[:, ::7] = t[:, ::7] = -torch.inf
        z[0, 1] = -torch.inf
        t[1, 2] = -torch.inf
        z.requires_grad_()
        value = distillation_loss(z, t, temperature=2.0, reduction="none", backend="kernel")
        (gradient,) = torch.autograd.grad(value, z, torch.ones_like(value))
        log_q = (z.detach().double() / 2).log_softmax(-1)
        p = (t.double() / 2).softmax(-1)
        expected_value = -torch.where(p > 0, p * log_q, 0.0).sum(-1)
        expected_gradient = (log_q.exp() - p) / 2
        torch.testing.assert_close(value.double(), expected_value, rtol=2e-6, atol=0)
        torch.testing.assert_close(gradient.double(), expected_gradient, rtol=2e-6, atol=3e-9)
        assert gradient[:, ::7].count_nonzero().item() == 0
        assert not torch.signbit(gradient[:, ::7]).any().item()

    def test_distillation_at_subnormal_temperatures(self):
        # 1/T overflows below 5.6e-309: dividing by T must not become a multiply by inf.
        for dtype in [torch.float32, torch.float64]:
            for temperature in [1e-20, 1e-300, 1e-310, 5e-324]:
                x = torch.tensor(
                    [[1.0, 0.0], [3.0, 3.0]], device=self.device, dtype=dtype, requires_grad=True
                )
                value = distillation_loss(x, x.detach(), temperature=temperature)
                value.backward()
                message = f"{dtype} T={temperature}"
                assert abs(value.item() - math.log(2) / 2) <= 1e-6, message
                torch.testing.assert_close(x.grad, torch.zeros_like(x), atol=0, rtol=0)

    def test_distillation_layout_reductions_and_autograd_contract(self):
        teacher = torch.randn(2, 4, 3, device=self.device, dtype=torch.float64)
        z = (teacher + 0.03 * torch.randn_like(teacher)).requires_grad_()

        def fn(x: torch.Tensor) -> torch.Tensor:
            return distillation_loss(x, teacher, temperature=2.5, dim=1)

        assert gradcheck(fn, (z,))
        unreduced = distillation_loss(z, teacher, dim=1, reduction="none")
        torch.testing.assert_close(
            unreduced.sum(), distillation_loss(z, teacher, dim=1, reduction="sum")
        )
        torch.testing.assert_close(unreduced.mean(), distillation_loss(z, teacher, dim=1))
        gradient = torch.autograd.grad(fn(z), z, create_graph=True)[0]
        with pytest.raises(RuntimeError):
            torch.autograd.grad(gradient.sum(), z)

    def test_cuda_validation(self):
        z = torch.zeros(2, 3, device=self.device)
        y = torch.tensor([0, 1], device=self.device)
        for fn, options, error in [
            (topk_cross_entropy, dict(k=1, backend="bad"), ValueError),
            (topk_cross_entropy, dict(k=True, backend="composed"), TypeError),
            (multiclass_hinge_loss, dict(backend="composed", gamma=-1.0), ValueError),
            (proximal_cross_entropy, dict(backend="composed", gamma=float("nan")), ValueError),
            (topk_cross_entropy, dict(k=1, backend="composed", reduction="bad"), ValueError),
        ]:
            with pytest.raises(error):
                fn(z, y, **options)
        # topk_cross_entropy at k = 1 is cross_entropy, also in the kernels.
        logits = torch.randn(2, 3, device=self.device)
        torch.testing.assert_close(
            topk_cross_entropy(logits, y, k=1, backend="kernel"),
            torch_loss(logits, y, backend="kernel"),
        )
        with pytest.raises(ValueError):
            torch_loss(z, y.cpu(), backend="composed")
        with pytest.raises(TypeError):
            torch_loss(z, y.float(), backend="composed")
        for target in [
            torch.tensor([0, 3], device=self.device),
            torch.zeros_like(z, dtype=torch.bool),
        ]:
            assert bool(torch_loss(z, target, backend="composed").isnan().any())
        assert bool(torch_loss(z.fill_(float("nan")), y, backend="composed").isnan().all())
        z.zero_()
        for logits, target, error in [
            (None, y, TypeError),
            (z.to(torch.int64), y, TypeError),
            (z[:, :1], y, ValueError),
        ]:
            with pytest.raises(error):
                torch_loss(logits, target, backend="composed")
        # Teachers of another floating-point dtype are promoted with the student.
        torch.testing.assert_close(
            distillation_loss(z, z.double()), distillation_loss(z.double(), z.double())
        )
        for teacher, options in [
            (z.cpu(), {}),
            (z[:1], {}),
            (z, dict(temperature=0)),
            (z, dict(reduction="bad")),
        ]:
            with pytest.raises(ValueError):
                distillation_loss(z, teacher, **options)
        with pytest.raises(TypeError):
            distillation_loss(z.to(torch.int64), z.to(torch.int64))
        with pytest.raises(ValueError):
            distillation_loss(z[:, :1], z[:, :1])
        assert bool(distillation_loss(z, z + float("inf")).isnan().all())
        for reduction in ("none", "sum", "mean"):
            empty_losses = (
                torch_loss(z[:0], y[:0], backend="composed", reduction=reduction),
                distillation_loss(z[:0], z[:0], reduction=reduction),
            )
            for value in empty_losses:
                if reduction == "none":
                    assert value.shape == (0,)
                elif reduction == "sum":
                    assert value.item() == 0
                else:
                    assert value.isnan()

    def test_small_cuda_training_and_autocast(self):
        x = torch.eye(4, device=self.device).repeat(8, 1)
        y = torch.arange(4, device=self.device).repeat(8) % 3
        model = torch.nn.Linear(4, 3, device=self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.8)
        criterion = torch_module(backend="composed")
        first = criterion(model(x), y).item()
        for _ in range(30):
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value = criterion(model(x), y)
            value.backward()
            optimizer.step()
        assert criterion(model(x), y).item() < first / 3
        assert (model(x).argmax(-1) == y).all().item()


_LAUNCHER_CHILD = """
import warnings, torch
import libsdca.torch as L
z = torch.randn(64, 1000, device="cuda")
y = torch.randint(0, 1000, (64,), device="cuda")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for _ in range(3):
        L.cross_entropy(z, y)
        L.CrossEntropyLoss()(z, y)
    torch.cuda.synchronize()
found = [str(w.message) for w in caught if "through Python" in str(w.message)]
print(len(found))
print(L.explain_backend(L.cross_entropy, z, y)[1])
"""


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("switch", ["auto", "off"])
def test_python_launcher_fallback_warns_once(tmp_path: Path, switch: str) -> None:
    """Where the C++ extension does not build, CUDA calls launch the kernels through Python
    (much more overhead per call): one RuntimeWarning per process, which
    SDCA_CPU_EXTENSION=off silences, and explain_backend names the launcher."""
    import os

    from child_process import run_python

    env = {
        **os.environ,
        "SDCA_CACHE_DIR": str(tmp_path),
        "CXX": "false",
        "SDCA_CPU_EXTENSION": switch,
    }
    result = run_python("-c", _LAUNCHER_CHILD, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    count, explanation = result.stdout.strip().splitlines()[-2:]
    assert count == ("1" if switch == "auto" else "0"), result.stdout
    assert "through the Python launcher" in explanation, explanation
