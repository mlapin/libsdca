"""float16/bfloat16 logits: float32 work by default, float64 with precision="float64".

Every loss against its float64 result on the upcast logits, the backend and arithmetic that
explain_backend reports, gradients in the logits dtype and (CUDA) peak memory of the fused paths.
"""

import os
from collections.abc import Callable
from unittest import mock

import pytest
import torch
from libsdca import _cpu_entropy
from libsdca.torch import (
    distillation_loss,
    explain_backend,
    proximal_distillation_loss,
    sdca_loss,
)

HALF = (torch.float16, torch.bfloat16)
EPS32 = torch.finfo(torch.float32).eps
# float32 work: values within 100 float32 eps of max(|loss|, 1), gradient entries within 2000
# eps of the row's largest entry (top-k entropy's float32 threshold solve), as numerics.md.
VALUE_EPS, ROW_EPS = 100, 2000
# (loss, target kind, k, gamma): integer targets ("index") or boolean masks ("mask").
CASES = [
    ("cross_entropy", "index", 1, 0.0),
    ("cross_entropy", "mask", 1, 0.0),
    ("topk_cross_entropy", "index", 1, 0.0),
    ("topk_cross_entropy", "index", 2, 0.0),
    ("topk_hinge_alpha", "index", 1, 0.0),
    ("topk_hinge_beta", "index", 2, 0.0),
    ("topk_hinge_alpha", "index", 1, 0.5),
    ("topk_hinge_beta", "index", 2, 0.5),
    ("multilabel_hinge", "mask", 1, 0.0),
    ("multilabel_hinge", "mask", 1, 0.5),
    ("proximal_cross_entropy", "index", 1, 0.0),
    ("proximal_cross_entropy", "mask", 1, 0.0),
]

# Teacher-logit losses; the CPU extension computes them in float64 (proximal) or float32.
DISTILLATION = {
    "distillation": lambda x, t, **kw: distillation_loss(x, t, 2.0, **kw),
    "proximal_distillation": lambda x, t, **kw: proximal_distillation_loss(x, t, 2.0, 0.7, **kw),
}


def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _targets(kind: str, n: int, c: int, gen: torch.Generator) -> torch.Tensor:
    labels = torch.randint(0, c, (n,), generator=gen)
    if kind == "index":
        return labels
    mask = torch.rand(n, c, generator=gen) < 0.3
    mask[torch.arange(n), labels] = True
    mask[torch.arange(n), (labels + 1) % c] = False
    return mask


def _cpu_kernel() -> bool:
    """Whether the CPU losses take the extension (built on first use unless disabled)."""
    return _cpu_entropy.enabled()


def _arithmetic(loss: str, kind: str, k: int, device: str) -> str:
    """The default arithmetic of float16/bfloat16 logits, as explain_backend names it."""
    if device == "cuda":
        # The fused entropy (integer targets and boolean masks), top-k included.
        if loss in ("cross_entropy", "topk_cross_entropy"):
            return "float32 exponentials"
        return "float32" if loss == "proximal_cross_entropy" else "float64"
    if loss in ("cross_entropy", "topk_cross_entropy") and kind == "index" and k == 1:
        return "float32 exponentials"  # as on CUDA, the extension or the native library
    if _cpu_kernel() and loss != "proximal_cross_entropy":
        return "float64"
    return "float32"


def _run(
    z: torch.Tensor,
    target: torch.Tensor,
    loss: str,
    k: int,
    gamma: float,
    reduction: str,
    precision: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = z.detach().clone().requires_grad_()
    value = sdca_loss(x, target, loss, k=k, gamma=gamma, reduction=reduction, precision=precision)
    weights = torch.linspace(0.5, 2.0, value.numel(), dtype=value.dtype, device=value.device)
    value.backward(weights.reshape(value.shape))
    assert x.grad is not None
    return value.detach(), x.grad


def _run_teacher(
    fn: Callable[..., torch.Tensor],
    z: torch.Tensor,
    teacher: torch.Tensor,
    reduction: str,
    precision: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = z.detach().clone().requires_grad_()
    value = fn(x, teacher, reduction=reduction, precision=precision)
    weights = torch.linspace(0.5, 2.0, value.numel(), dtype=value.dtype, device=value.device)
    value.backward(weights.reshape(value.shape))
    assert x.grad is not None
    return value.detach(), x.grad


class HalfPrecisionTests:
    def setup_method(self):
        self.gen = torch.Generator().manual_seed(0)

    def test_large_upstream_keeps_representable_hinge_gradients(self, subtests: pytest.Subtests):
        # These unconstrained smooth projections have analytic gradients.
        # The second row is strictly inside the zero-loss region. Converting
        # the upstream to fp16 first produces infinities and 0 * inf NaNs.
        for device in _devices():
            for dtype in HALF:
                for loss in ("topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge"):
                    for scale in (65536.0, -65536.0):
                        with subtests.test(device=device, dtype=dtype, loss=loss, scale=scale):
                            x = torch.tensor(
                                [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                                device=device,
                                dtype=dtype,
                                requires_grad=True,
                            )
                            value = sdca_loss(
                                x,
                                torch.zeros(2, device=device, dtype=torch.int64),
                                loss,
                                k=2,
                                gamma=1000.0,
                                backend="torch",
                                precision="float32",
                                reduction="none",
                            )
                            (gradient,) = torch.autograd.grad(
                                value, x, torch.full_like(value, scale)
                            )
                            divisor = 3000.0 if loss == "multilabel_hinge" else 1000.0
                            expected = x.new_tensor(
                                [
                                    [-2 * scale / divisor, scale / divisor, scale / divisor],
                                    [0.0, 0.0, 0.0],
                                ]
                            )
                            assert gradient.dtype == dtype
                            torch.testing.assert_close(
                                gradient,
                                expected,
                                rtol=2 * torch.finfo(dtype).eps,
                                atol=0,
                            )

    def assert_against_float64(
        self,
        z: torch.Tensor,
        target: torch.Tensor,
        loss: str,
        k: int,
        gamma: float,
        reduction: str,
        precision: str | None,
        message: str,
    ) -> None:
        value, grad = _run(z, target, loss, k, gamma, reduction, precision)
        ref_value, ref_grad = _run(z.double(), target, loss, k, gamma, reduction)
        self.assert_envelope(z, value, grad, ref_value, ref_grad, precision, message)

    def assert_envelope(
        self,
        z: torch.Tensor,
        value: torch.Tensor,
        grad: torch.Tensor,
        ref_value: torch.Tensor,
        ref_grad: torch.Tensor,
        precision: str | None,
        message: str,
    ) -> None:
        assert grad.dtype == z.dtype, message
        out = torch.finfo(value.dtype)
        dtype = torch.finfo(z.dtype)
        work = 0 if precision == "float64" else 1
        v, r = value.double().cpu(), ref_value.cpu()
        bound = out.eps * r.abs() + work * VALUE_EPS * EPS32 * r.abs().clamp_min(1) + out.tiny
        assert bool(((v - r).abs() <= bound).all()), f"{message} value {v} vs {r}"
        g, rg = grad.double().cpu(), ref_grad.cpu()
        big = rg.abs().amax(-1, keepdim=True)
        bound = dtype.eps * rg.abs() + work * ROW_EPS * EPS32 * big + dtype.tiny * dtype.eps
        error = ((g - rg).abs() - bound).max().item()
        assert error <= 0, f"{message} gradient"

    def test_every_loss_against_float64(self):
        for device in _devices():
            for dtype in HALF:
                for loss, kind, k, gamma in CASES:
                    z = (3 * torch.randn(6, 37, generator=self.gen)).to(device, dtype)
                    target = _targets(kind, 6, 37, self.gen).to(device)
                    for precision in (None, "float64"):
                        for reduction in ("none", "mean"):
                            message = (
                                f"{device} {dtype} {loss} {kind} k={k} gamma={gamma} {precision}"
                            )
                            self.assert_against_float64(
                                z, target, loss, k, gamma, reduction, precision, message
                            )

    def test_distillation_against_float64(self):
        for device in _devices():
            for kernel in (
                ("off", "on") if device == "cpu" and _cpu_entropy.buildable() else ("off",)
            ):
                for dtype in HALF:
                    for name, fn in DISTILLATION.items():
                        z = 3 * torch.randn(6, 37, generator=self.gen)
                        teacher = (z + torch.randn(6, 37, generator=self.gen)).to(device, dtype)
                        z = z.to(device, dtype)
                        for precision in (None, "float64"):
                            for reduction in ("none", "mean"):
                                message = f"{device} kernel={kernel} {dtype} {name} {precision}"
                                with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": kernel}):
                                    value, grad = _run_teacher(fn, z, teacher, reduction, precision)
                                    ref = _run_teacher(fn, z.double(), teacher.double(), reduction)
                                assert value.dtype == torch.float32, message
                                self.assert_envelope(z, value, grad, *ref, precision, message)

    def test_explain_backend_names_the_arithmetic(self):
        for device in _devices():
            for dtype in HALF:
                for loss, kind, k, gamma in CASES:
                    z = torch.randn(4, 9).to(device, dtype)
                    target = _targets(kind, 4, 9, self.gen).to(device)
                    message = f"{device} {dtype} {loss} {kind} k={k} gamma={gamma}"
                    name, reason = explain_backend(z, target, loss, k=k, gamma=gamma)
                    assert _arithmetic(loss, kind, k, device) in reason, message
                    if device == "cuda":
                        assert name in ("cuda-launcher", "triton"), message + reason
                    name, reason = explain_backend(
                        z, target, loss, k=k, gamma=gamma, precision="float64"
                    )
                    assert "float64" in reason, message
                    assert "float32 exponentials" not in reason, message
                    if loss == "proximal_cross_entropy" or kind == "mask":
                        assert "in float32" not in reason, message

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_fused_paths_keep_16_bit_storage(self):
        # Peak allocation of forward + backward beyond the logits: the gradient (one logits
        # size) and row statistics. proximal_cross_entropy saves a float32 gradient in forward
        # (two sizes) and rounds its scaled copy in backward (one more); its integer targets'
        # label mask is freed before that.
        n, c = 512, 4096
        for dtype in HALF:
            for loss, kind, k, gamma in CASES:
                limit = 3.15 if loss == "proximal_cross_entropy" else 1.15
                x = torch.randn(n, c, device="cuda").to(dtype).requires_grad_()
                target = _targets(kind, n, c, self.gen).cuda()
                message = f"{dtype} {loss} {kind} k={k} gamma={gamma}"
                base = 0
                for _ in range(2):  # the first call compiles
                    x.grad = None
                    torch.cuda.synchronize()
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    sdca_loss(x, target, loss, k=k, gamma=gamma).backward()
                    torch.cuda.synchronize()
                extra = torch.cuda.max_memory_allocated() - base
                assert x.grad is not None
                assert x.grad.dtype == dtype, message
                assert extra <= limit * x.nbytes + 2**20, (
                    f"{message}: {extra / x.nbytes:.2f} logits"
                )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_fused_distillation_keeps_16_bit_storage(self):
        # distillation_loss recomputes its gradient in backward (one size); the proximal kernel
        # saves a float32 gradient (two sizes) and backward rounds a scaled copy (one more).
        n, c = 512, 4096
        for dtype in HALF:
            for name, fn in DISTILLATION.items():
                x = torch.randn(n, c, device="cuda").to(dtype).requires_grad_()
                teacher = torch.randn(n, c, device="cuda").to(dtype)
                base = 0
                for _ in range(2):  # the first call compiles
                    x.grad = None
                    torch.cuda.synchronize()
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    fn(x, teacher).backward()
                    torch.cuda.synchronize()
                extra = torch.cuda.max_memory_allocated() - base
                assert x.grad is not None
                assert x.grad.dtype == dtype
                limit = 3.15 if name == "proximal_distillation" else 2.15
                assert extra <= limit * x.nbytes + 2**20, f"{dtype} {name}"


class NativeHalfEntropyTests:
    def test_native_rows_match_the_tensor_path(self, subtests: pytest.Subtests):
        """16-bit rows with float32 exponentials and float64 statistics (the native library or
        the extension) against the float64 tensor path: values within 8 float32 epsilons,
        gradients within one unit of the logits dtype."""
        torch.manual_seed(3)
        for dtype in HALF:
            for reduction in ("none", "sum", "mean"):
                z = (torch.randn(40, 9) * 6).to(dtype)
                z[1, 2] = -60  # a far competitor
                y = torch.randint(0, 9, (40,))
                results = []
                for backend in ("auto", "torch"):
                    x = z.clone().requires_grad_()
                    v = sdca_loss(x, y, reduction=reduction, backend=backend)
                    up = torch.linspace(0.5, 2.0, 40) if reduction == "none" else torch.tensor(3.0)
                    (g,) = torch.autograd.grad(v, x, up)
                    results.append((v, g))
                with subtests.test(dtype=dtype, reduction=reduction):
                    (value, grad), (value64, grad64) = results
                    eps32 = torch.finfo(torch.float32).eps
                    torch.testing.assert_close(value, value64, rtol=8 * eps32, atol=0)
                    unit = torch.finfo(dtype).eps * grad64.double().abs().amax(-1, keepdim=True)
                    assert bool(((grad.double() - grad64.double()).abs() <= unit).all())
                    expected = "cpu-kernel" if _cpu_entropy.enabled() else "native"
                    assert explain_backend(z, y)[0] == expected
                x = z.clone().requires_grad_()
                (g,) = torch.autograd.grad(sdca_loss(x, y), x, create_graph=True)
                (h,) = torch.autograd.grad(g.float().square().sum(), x)
                assert bool(h.isfinite().all())
