"""Fused CUDA integer entropy: column chunks, wide rows and precision="float32".

References are 60-digit mpmath values and gradients of the represented logits, or float64
composed PyTorch; no reference uses a kernel's output.
"""

import os
from unittest import mock

import interpreter
import mpmath as mp
import pytest
import torch
import torch.nn.functional as F
from libsdca import _cuda_launch
from libsdca.torch import explain_backend, sdca_loss

# The float32 contract, in float32 eps (losses) and eps of the logits dtype (gradients):
# relative loss and target-entry errors, competitor errors relative to the row's largest entry.
VALUE_EPS, TARGET_EPS, ROW_EPS = 8, 8, 8


def _kernels():
    if not torch.cuda.is_available():
        return None
    try:
        from libsdca import _triton_entropy
    except ImportError:
        return None
    return _triton_entropy


def _upstream(n: int, reduction: str, scale: float) -> torch.Tensor:
    if reduction == "none":
        u = torch.full((n,), scale, dtype=torch.float64)
        u[1::2] *= -1
        return u
    return torch.tensor(scale, dtype=torch.float64)


def _mp_reference(
    z: torch.Tensor, y: torch.Tensor, reduction: str, u: torch.Tensor
) -> tuple[list[float], list[list[float]]]:
    """Loss and gradient rows of the represented logits, 60 digits."""
    values, grads = [], []
    n = z.shape[0]
    with mp.workdps(60):
        for i, row in enumerate(z.double().tolist()):
            t = int(y[i])
            e = [mp.exp(mp.mpf(v) - row[t]) if j != t else mp.mpf(0) for j, v in enumerate(row)]
            tail = mp.fsum(e)
            values.append(mp.log1p(tail))
            weight = mp.mpf(float(u[i] if reduction == "none" else u)) / (
                n if reduction == "mean" else 1
            )
            grads.append([weight * v / (1 + tail) for v in e])
            grads[-1][t] = -weight * tail / (1 + tail)
        if reduction == "none":
            value = [float(v) for v in values]
        else:
            total = mp.fsum(values)
            assert isinstance(total, mp.mpf)
            value = [float(total / n if reduction == "mean" else total)]
    return value, [[float(v) for v in row] for row in grads]


def _call(
    z: torch.Tensor, y: torch.Tensor, reduction: str, u: torch.Tensor, precision: object
) -> tuple[torch.Tensor, torch.Tensor]:
    x = z.detach().clone().requires_grad_()
    value = sdca_loss(x, y, reduction=reduction, precision=precision)
    value.backward(u.to(device=value.device, dtype=value.dtype))
    assert x.grad is not None
    return value.detach(), x.grad


def interpreter_check(chunked: tuple[bool, ...] = (False, True)) -> None:
    """The float32 contract on the kernels (CPU tensors under the interpreter), in registers
    and in 4-column chunks: float32 exponentials forward and backward."""
    from libsdca import _triton_entropy as kernels

    def call(
        z: torch.Tensor, y: torch.Tensor, reduction: str, u: torch.Tensor, precision: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert precision == "float32"
        value, stats = kernels.forward(z, y, reduction, True)
        upstream = u.to(value.dtype).reshape(value.shape)
        return value, kernels.backward(z, y, stats, upstream, reduction, True)

    tests = FusedEntropyTests()
    tests.DEVICE = "cpu"
    globals()["_call"] = call  # assert_contract's kernel call
    gaps = (0.0, 1.0, 16.7, 36.5, 37.5, 60.0, 87.0, 88.0, 100.0, 103.9)
    two = torch.tensor([[0.0, -gap] for gap in gaps] + [[0.0, 30.0], [1e4, 1e4 - 0.5]])
    mixed = [[0, 0, 0, 0], [0, 0.25, -0.125, 0.5], [0, -38, -39, -40], [0, 1, -80, -90]]
    tiny = torch.finfo(torch.float32).tiny * torch.finfo(torch.float32).eps
    for chunks in chunked:
        if chunks:
            kernels._MAX_BLOCK, kernels._CHUNKS = 2, {True: (4, 4), False: (4, 4)}
        tests.setup_method()
        for c, scale in ((2, 3.0), (37, 3.0), (300, 0.01), (300, 10.0)):
            z = scale * torch.randn(4, c, generator=tests.gen)
            y = torch.randint(0, c, (4,), generator=tests.gen)
            for reduction in ("none", "mean"):
                tests.assert_contract(z, y, reduction, (0.75, -3e38), f"c={c} s={scale}")
        y = torch.zeros(len(two), dtype=torch.long)
        tests.assert_contract(two, y, "none", (1.0, -1e-30, 3e38), "two classes")
        z = torch.tensor(mixed)
        y = torch.zeros(len(mixed), dtype=torch.long)
        scales = (0.0, 1.0, 1e-20, tiny, torch.finfo(torch.float32).max)
        tests.assert_contract(z, y, "sum", scales, "mixed rows")
        z = torch.full((2, 300), -90.0)
        z[:, 0] = 0
        z[1, 1:] -= torch.linspace(0, 5, 299)
        tests.assert_contract(z, torch.zeros(2, dtype=torch.long), "sum", (1.0, 3e38), "300")


class InterpreterTests:
    @pytest.mark.slow
    @pytest.mark.parametrize("chunked", [False, True])
    def test_float32_contract_in_the_interpreter(self, chunked: bool):
        interpreter.run(f"import test_fused_entropy as t\nt.interpreter_check(({chunked},))\n")


@pytest.mark.skipif(not _kernels(), reason="CUDA with Triton is not available")
class FusedEntropyTests:
    """The default kernel configuration."""

    WIDE_COLS: tuple[int, ...] = (5000, 40000, 131072)
    ROUTING_COLS: tuple[int, int] = (40000, 100)
    DEVICE: str = "cuda"

    def setup_method(self):
        self.gen = torch.Generator().manual_seed(0)

    def assert_mixed(
        self, z: torch.Tensor, y: torch.Tensor, reduction: str, u: torch.Tensor, msg: str = ""
    ) -> None:
        """The default precision against mpmath, to float64 rounding."""
        ref_value, ref_grads = _mp_reference(z, y, reduction, u)
        value, grad = _call(z.cuda(), y.cuda(), reduction, u, None)
        expected = torch.tensor(ref_value, dtype=torch.float64).reshape(value.shape)
        torch.testing.assert_close(value.cpu(), expected, rtol=1e-12, atol=1e-300, msg=msg)
        expected = torch.tensor(ref_grads, dtype=torch.float64)
        torch.testing.assert_close(grad.cpu(), expected, rtol=1e-12, atol=1e-300, msg=msg)

    def assert_contract(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        reduction: str = "none",
        scales: tuple[float, ...] = (0.75,),
        msg: str = "",
    ) -> None:
        """precision="float32" against mpmath, within the stated float32 contract."""
        eps, tiny = torch.finfo(z.dtype).eps, torch.finfo(z.dtype).tiny
        eps32, tiny32 = torch.finfo(torch.float32).eps, torch.finfo(torch.float32).tiny
        for scale in scales:
            u = _upstream(z.shape[0], reduction, scale)
            ref_value, ref_grads = _mp_reference(z, y, reduction, u)
            value, grad = _call(z.to(self.DEVICE), y.to(self.DEVICE), reduction, u, "float32")
            message = f"{msg} {z.dtype} {reduction} u={scale:g}"
            for a, b in zip(value.double().flatten().tolist(), ref_value, strict=True):
                bound = max(VALUE_EPS * eps32 * abs(b), tiny32 * eps32)
                assert abs(a - b) <= bound, f"{message} value {a!r} vs {b!r}"
            for i, (row, ref) in enumerate(zip(grad.double().tolist(), ref_grads, strict=True)):
                big = max(abs(v) for v in ref)
                t = int(y[i])
                for j, (a, b) in enumerate(zip(row, ref, strict=True)):
                    if j == t:
                        bound = max(TARGET_EPS * eps * abs(b), tiny * eps)
                    else:
                        bound = max(ROW_EPS * eps * big, tiny * eps)
                    assert abs(a - b) <= bound, f"{message} row {i} entry {j}: {a!r} vs {b!r}"

    def test_float32_random_rows(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for c, scale in ((2, 3.0), (37, 3.0), (1000, 3.0), (1000, 0.01), (5000, 10.0)):
                z = (scale * torch.randn(4, c, generator=self.gen)).to(dtype)
                y = torch.randint(0, c, (4,), generator=self.gen)
                for reduction in ("none", "mean", "sum"):
                    self.assert_contract(z, y, reduction, msg=f"c={c} s={scale}")

    def test_float32_tails_and_storage_boundaries(self):
        # Correct predictions up to the float32 storage boundaries (103.97 for two classes,
        # 110.88 for 1000), wrong ones, offsets and ties; upstream scales to float32's maximum.
        gaps = (0.0, 1.0, 16.7, 36.5, 37.5, 60.0, 87.0, 88.0, 100.0, 103.9)
        rows = [[0.0, -gap] for gap in gaps]
        rows += [[0.0, 30.0], [0.0, 1e4], [1e4, 1e4 - 0.5], [-1e4, -1e4 + 3]]
        z = torch.tensor(rows)
        y = torch.zeros(len(rows), dtype=torch.long)
        scales = (1.0, -1e-30, 3e38)
        self.assert_contract(z, y, "none", scales, "two classes")
        self.assert_contract(z, y, "mean", scales, "two classes")
        for gap in (20.0, 40.0, 90.0, 110.8):
            z = torch.full((2, 1000), -gap)
            z[:, 0] = 0
            z[1, 1:] -= torch.linspace(0, 5, 999)
            self.assert_contract(z, torch.zeros(2, dtype=torch.long), "sum", (1.0, 3e38), "1000")
        z = torch.full((1, 1000), -110.8)
        z[0, 0] = 0
        x = z.cuda().requires_grad_()
        loss = sdca_loss(x, torch.zeros(1, dtype=torch.long, device="cuda"), precision="float32")
        loss.backward()
        assert loss.item() > 0
        assert x.grad is not None
        assert x.grad[0, 0].item() < 0

    def test_float32_mixed_rows_and_upstream_scales(self):
        rows = [
            [0, 0, 0, 0],
            [0, 0.25, -0.125, 0.5],
            [0, -36, -37, -38],
            [0, -38, -39, -40],
            [0, 1, -599, -601],
            [0, 1, -80, -90],
        ]
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            z = torch.tensor(rows, dtype=dtype)
            y = torch.zeros(len(rows), dtype=torch.long)
            minimum = torch.finfo(dtype).tiny * torch.finfo(dtype).eps
            scales = (0.0, 1.0, -1.0, 1e-20, minimum, torch.finfo(dtype).max)
            for reduction in ("none", "sum", "mean"):
                self.assert_contract(z, y, reduction, scales, "mixed rows")

    def test_wide_rows(self):
        # Rows past the register block and past the old 32768-class limit.
        for c in self.WIDE_COLS:
            z = 3 * torch.randn(3, c, generator=self.gen, dtype=torch.float64)
            y = torch.randint(0, c, (3,), generator=self.gen)
            for reduction in ("none", "mean"):
                u = _upstream(3, reduction, 0.75)
                ref = z.clone().requires_grad_()
                expected = F.cross_entropy(ref, y, reduction=reduction)
                expected.backward(u)
                assert ref.grad is not None
                value, grad = _call(z.cuda(), y.cuda(), reduction, u, None)
                message = f"c={c} {reduction}"
                torch.testing.assert_close(value.cpu(), expected.detach(), rtol=1e-12, atol=0)
                torch.testing.assert_close(grad.cpu(), ref.grad, rtol=1e-12, atol=1e-300)
                value, grad = _call(z.float().cuda(), y.cuda(), reduction, u, "float32")
                eps = torch.finfo(torch.float32).eps
                torch.testing.assert_close(
                    value.cpu().double(),
                    expected.detach(),
                    rtol=VALUE_EPS * eps,
                    atol=0,
                    msg=message,
                )
                error = (grad.cpu().double() - ref.grad).abs().amax(1)
                assert bool((error <= ROW_EPS * eps * ref.grad.abs().amax(1)).all())

    def test_routing_explain_and_launch_parity(self):
        wide_c, reg_c = self.ROUTING_COLS
        z = torch.randn(5, wide_c, device="cuda")
        y = torch.randint(0, wide_c, (5,), device="cuda")
        host, _ = _cuda_launch.status()
        name, reason = explain_backend(z, y, precision="float32")
        assert name == ("cuda-launcher" if host is not None else "triton"), reason
        assert "float32 exponentials" in reason
        assert "column chunks" in reason
        assert "rows in registers" in explain_backend(z[:, :reg_c], y % reg_c)[1]
        with pytest.raises(ValueError, match="precision"):
            explain_backend(z.double(), y, precision="float32")
        with pytest.raises(ValueError):
            sdca_loss(z, y, backend="native", precision="float32")
        for precision in (None, "float32", "float64", torch.float32, torch.float64):
            for reduction in ("none", "mean"):
                u = _upstream(5, reduction, 1.5)
                default = _call(z, y, reduction, u, precision)
                # The launcher status reads SDCA_CUDA_LAUNCHER once; clear it to reread.
                with mock.patch.dict(os.environ, {_cuda_launch.ENV: "off"}):
                    _cuda_launch.status.cache_clear()
                    try:
                        name, reason = explain_backend(z, y, precision=precision)
                        assert name == "triton", reason
                        forced = _call(z, y, reduction, u, precision)
                    finally:
                        _cuda_launch.status.cache_clear()
                message = f"{precision} {reduction}"
                assert torch.equal(default[0], forced[0]), message
                assert torch.equal(default[1], forced[1]), message

    def test_float32_higher_derivatives(self):
        # Double backward differentiates the float64 gradient, as for the default precision.
        z = torch.randn(3, 6, device="cuda", requires_grad=True)
        y = torch.tensor([0, 2, 5], device="cuda")
        second = []
        for precision in ("float64", "float32"):
            (g,) = torch.autograd.grad(sdca_loss(z, y, precision=precision), z, create_graph=True)
            second.append(torch.autograd.grad(g.square().sum(), z)[0])
        torch.testing.assert_close(second[1], second[0], rtol=1e-6, atol=1e-9)


@pytest.mark.skipif(not _kernels(), reason="CUDA with Triton is not available")
class ChunkedEntropyTests(FusedEntropyTests):
    """The same cases with rows wider than two columns in four-column chunks."""

    WIDE_COLS = (5, 13, 37)
    ROUTING_COLS = (13, 2)

    def setup_method(self):
        super().setup_method()
        kernels = _kernels()
        assert kernels is not None
        saved = kernels._MAX_BLOCK, kernels._CHUNKS

        def restore() -> None:
            kernels._MAX_BLOCK, kernels._CHUNKS = saved
            kernels._compiled.cache_clear()

        kernels._MAX_BLOCK, kernels._CHUNKS = 2, {True: (4, 4), False: (4, 4)}
        kernels._compiled.cache_clear()
        self.restore = restore

    def teardown_method(self):
        self.restore()

    def test_running_maximum_across_chunks(self):
        # Maxima rising by small and large steps, falling rows, ties, a chunk holding only the
        # target, and the maximum in the last (partial) chunk.
        j = torch.arange(13, dtype=torch.float64)
        z = torch.stack([0.1 * j, j, -0.1 * j, 0 * j, 2.0 * (j % 4 == 3), 40 * j, -40 * j, 800 * j])
        y = torch.tensor([12, 0, 5, 3, 12, 4, 0, 12])
        for reduction in ("none", "sum", "mean"):
            self.assert_mixed(z, y, reduction, _upstream(len(z), reduction, -0.5), reduction)
            self.assert_contract(z.float(), y, reduction, (1.0,), "rising")

    def test_validation_in_every_chunk(self):
        y = torch.tensor([0, 3, 12], device="cuda")
        for precision in (None, "float32", "float64"):
            for bad in (float("nan"), float("inf"), -float("inf")):
                for col in (1, 5, 12):
                    z = torch.zeros(3, 13, device="cuda", requires_grad=True)
                    with torch.no_grad():
                        z[1 if col != 12 else 2, col] = bad
                    loss = sdca_loss(z, y, precision=precision)
                    assert bool(loss.isnan().all()), f"{precision} {bad} {col}"
                    (grad,) = torch.autograd.grad(loss, z)
                    assert bool(grad.isnan().any()), f"{precision} {bad} {col}"
            for t in (-1, 13):
                target = torch.tensor([0, t, 1], device="cuda")
                z = torch.zeros(3, 13, device="cuda", requires_grad=True)
                loss = sdca_loss(z, target, precision=precision)
                assert bool(loss.isnan().all())
                (grad,) = torch.autograd.grad(loss, z)
                assert bool(grad[1].isnan().all())
