"""distillation_loss and proximal_distillation_loss on every path: the native library
(SDCA_CPU_EXTENSION=off), CPU tensor operations, the CPU extension, the fused CUDA kernels (rows up to 4096 classes in
registers, wider ones streamed) and the CUDA tensor fallback, against mpmath and float64
references."""

import contextlib
import math
import os
from collections.abc import Callable
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch
from libsdca import _cpu_entropy
from libsdca import torch as sdca_torch
from libsdca.torch import (
    DistillationLoss,
    ProximalDistillationLoss,
    distillation_loss,
    proximal_distillation_loss,
)
from test_distillation_references import proximal_reference, reference
from torch.autograd import gradcheck

DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _no_kernel(proximal: bool = False) -> None:
    return None


def _none(*args: object) -> None:
    return None


def paths():
    """(name, device, context factory) of every backend."""
    result = [
        ("cpu-native", "cpu", lambda: mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": "off"})),
        ("cpu-tensor", "cpu", lambda: mock.patch.object(sdca_torch, "_cpu_distillation", _none)),
    ]
    if _cpu_entropy.buildable():
        result.append(
            ("cpu-kernel", "cpu", lambda: mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": "on"}))
        )
    if torch.cuda.is_available():
        result += [
            ("cuda-kernel", "cuda", contextlib.nullcontext),
            (
                "cuda-tensor",
                "cuda",
                lambda: mock.patch.object(sdca_torch, "_kernel_distill", _no_kernel),
            ),
        ]
    return result


def loss_fn(proximal: bool, alpha: float) -> Callable[..., torch.Tensor]:
    if proximal:
        return lambda x, t, T, **kw: proximal_distillation_loss(x, t, T, alpha, **kw)
    return distillation_loss


def evaluate(
    fn: Callable[..., torch.Tensor],
    z: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
    **options: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unreduced values and the gradient of sum(w * values), divided by the row weights w."""
    x = z.detach().clone().requires_grad_()
    value = fn(x, teacher, temperature, reduction="none", **options)
    weights = torch.linspace(0.5, 2.0, value.shape[0], dtype=value.dtype, device=value.device)
    (gradient,) = torch.autograd.grad(value, x, weights)
    return value.detach(), gradient / weights[:, None].to(gradient.dtype)


def _eps(dtype: type[np.floating[Any]]) -> float:
    return float(np.finfo(dtype).eps)


def rows(
    dtype: type[np.floating[Any]], classes: int = 7, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Equal, near (1e-3) and far teachers for random students."""
    rng = np.random.default_rng(seed)
    student = rng.normal(size=(4, classes)) * 4
    spread = np.array([[0.0], [1e-3], [1.0], [5.0]])
    teacher = student + rng.normal(size=student.shape) * spread
    return student.astype(dtype), teacher.astype(dtype)


def _fused_rows(
    x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, temperature: float, double: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unreduced values and weighted gradients of the Triton-launched distillation_loss."""
    from libsdca import _triton_distillation as kernels

    value, stats = kernels.forward(x, y, temperature, double, "none")
    return value, kernels.backward(x, y, stats, weights, temperature, double, "none")


class DistillationPathTests:
    def assert_close(
        self,
        value: torch.Tensor,
        gradient: torch.Tensor,
        expected_value: np.ndarray,
        expected_gradient: np.ndarray,
        eps: float,
        message: str,
    ) -> None:
        v, g = value.double().cpu().numpy(), gradient.double().cpu().numpy()
        bound = 64 * eps * np.abs(expected_value) + 1e-300
        assert np.all(np.abs(v - expected_value) <= bound), f"{message} value {v}"
        # Relative to the row's largest entry; the oracle resolves pi - p to about 1e-70.
        scale = np.abs(expected_gradient).max(-1, keepdims=True)
        bound = 64 * eps * scale + 1e-60
        error = np.abs(g - expected_gradient)
        assert np.all(error <= bound), f"{message} gradient {error.max()}"

    def test_every_path_against_mpmath(self, subtests: pytest.Subtests):
        for name, device, context in paths():
            for dtype in (np.float32, np.float64):
                eps = _eps(dtype)
                student, teacher = rows(dtype)
                for temperature in (0.25, 1.0, 4.0):
                    for proximal, alpha in [(False, 0.0)] + [
                        (True, a) for a in (1e-6, 0.3, 10.0, 1e6)
                    ]:
                        message = f"{name} {dtype.__name__} T={temperature} alpha={alpha}"
                        with subtests.test(message), context():
                            z = torch.tensor(student, device=device)
                            t = torch.tensor(teacher, device=device)
                            fn = loss_fn(proximal, alpha)
                            value, gradient = evaluate(fn, z, t, temperature)
                            if proximal:
                                expected = proximal_reference(student, teacher, temperature, alpha)
                            else:
                                expected = reference(student, teacher, temperature)
                            self.assert_close(value, gradient, *expected, eps, message)

    def test_kernel_class_boundary(self, subtests: pytest.Subtests):
        # The fused kernels hold up to 4096 classes in registers and stream wider rows.
        calls = []
        kernel = sdca_torch._kernel_distill

        def spy(proximal: bool = False):
            inner = kernel(proximal)

            def call(*args: Any):
                assert inner is not None
                calls.append(args[0].shape[-1])
                return inner(*args)

            return None if inner is None else call

        for name, device, context in paths():
            for classes in (4095, 4096, 4097):
                for dtype in (torch.float32, torch.float64):
                    gen = torch.Generator().manual_seed(classes)
                    z = 3 * torch.randn(3, classes, generator=gen, dtype=torch.float64)
                    t = z + torch.randn(3, classes, generator=gen, dtype=torch.float64)
                    for proximal in (False, True):
                        fn = loss_fn(proximal, 0.7)
                        message = f"{name} C={classes} {dtype} proximal={proximal}"
                        with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": "off"}):
                            ref_v, ref_g = evaluate(fn, z, t, 2.0)
                        calls.clear()
                        with subtests.test(message), context():
                            if name == "cuda-kernel":
                                patch = mock.patch.object(sdca_torch, "_kernel_distill", spy)
                            else:
                                patch = contextlib.nullcontext()
                            with patch:
                                x, y = z.to(device, dtype), t.to(device, dtype)
                                value, gradient = evaluate(fn, x, y, 2.0)
                            if name == "cuda-kernel":
                                assert calls == [classes]
                            eps = 256 * torch.finfo(dtype).eps
                            # float32 inputs are rounded; compare with their float64 result.
                            if dtype == torch.float32:
                                with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": "off"}):
                                    ref_v, ref_g = evaluate(
                                        fn, z.float().double(), t.float().double(), 2.0
                                    )
                            torch.testing.assert_close(
                                value.double().cpu(), ref_v, rtol=eps, atol=0, msg=message
                            )
                            scale = ref_g.abs().amax(-1, keepdim=True)
                            error = (gradient.double().cpu() - ref_g).abs() / scale
                            assert error.max().item() <= eps, message

    def test_streamed_proximal_rows_match_rows_in_registers(self):
        """Proximal rows streamed in tiles of 16 columns against the same rows in registers."""
        kernel = sdca_torch._kernel_distill(True) if torch.cuda.is_available() else None
        if kernel is None:
            pytest.skip("needs Triton and an NVIDIA GPU")
        from libsdca import _triton_distillation

        gen = torch.Generator().manual_seed(11)
        z = 3 * torch.randn(4, 70, generator=gen, dtype=torch.float64)
        t = z + torch.randn(4, 70, generator=gen, dtype=torch.float64)
        t[1] = z[1]  # equal logits: a zero gradient
        z[2, :5] = 1e4  # tied largest logits, far from the others
        rows = [(z, t), (t[:, :37], z[:, :37]), (z.new_tensor([[0.0, 1e300, -1e300]]),) * 2]
        for z, t in rows:
            for temperature in (0.5, 1.0, 4.0):
                for alpha in (1e-3, 0.7, 100.0):
                    for dtype in (torch.float64, torch.float32, torch.bfloat16):
                        x, y = z.to("cuda", dtype), t.to("cuda", dtype)
                        for double in (False, True) if dtype != torch.float64 else (False,):
                            expected = kernel(x, y, temperature, alpha, double)
                            with mock.patch.multiple(
                                _triton_distillation, _CHUNK=0, _STREAM_BLOCK=16
                            ):
                                actual = kernel(x, y, temperature, alpha, double)
                            fp32 = dtype != torch.float64 and not double
                            rtol, atol = (2e-5, 2e-6) if fp32 else (1e-13, 1e-14)
                            message = f"{dtype} double={double} T={temperature} alpha={alpha}"
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

    def test_proximal_tiny_teacher_probabilities(self, subtests: pytest.Subtests):
        # pi far above a teacher probability near the underflow: log1p(-r / p) overflowed,
        # giving a -inf loss (float32: T=0.5, alpha=100, logits of scale 30).
        # Wide rows add classes whose probabilities are zero: streamed on CUDA.
        for name, device, context in paths():
            for dtype, gap in ((np.float32, 95.0), (np.float64, 740.0)):
                eps = _eps(dtype)
                for temperature in (0.5, 1.0):
                    student = np.array([[0.0, 10.0, 0.0], [0.0, 10.0, -5.0]]) * temperature
                    teacher = np.array([[0.0, -gap, 0.0], [0.0, -gap, 1.0]]) * temperature
                    student, teacher = student.astype(dtype), teacher.astype(dtype)
                    pad = np.full((2, 4100), -1e4 * temperature, dtype=dtype)
                    for alpha in (1.0, 100.0):
                        expected = proximal_reference(student, teacher, temperature, alpha)
                        for width in (3, 4103):
                            message = f"{name} {dtype.__name__} T={temperature} alpha={alpha}"
                            with subtests.test(message, width=width), context():
                                s, t = student, teacher
                                if width > 3:
                                    s, t = np.hstack([s, pad]), np.hstack([t, pad])
                                z = torch.tensor(s, device=device)
                                t = torch.tensor(t, device=device)
                                fn = loss_fn(True, alpha)
                                value, gradient = evaluate(fn, z, t, temperature)
                                assert not bool(gradient[:, 3:].any())
                                self.assert_close(value, gradient[:, :3], *expected, eps, message)

    def test_streamed_rows_match_rows_in_registers(self):
        """distillation_loss rows streamed in tiles of 16 columns against rows in registers."""
        if not torch.cuda.is_available() or sdca_torch._kernel_distill() is None:
            pytest.skip("needs Triton and an NVIDIA GPU")
        from libsdca import _triton_distillation as kernels

        gen = torch.Generator().manual_seed(12)
        z = 3 * torch.randn(4, 70, generator=gen, dtype=torch.float64)
        t = z + torch.randn(4, 70, generator=gen, dtype=torch.float64)
        t[1] = z[1]  # equal logits: a zero gradient
        z[2, :5] = 1e4  # tied largest logits, far from the others
        rows = [(z, t), (t[:, :37], z[:, :37]), (z.new_tensor([[0.0, 1e300, -1e300]]),) * 2]
        for z, t in rows:
            for temperature in (0.5, 1.0, 4.0):
                for dtype in (torch.float64, torch.float32, torch.bfloat16):
                    x, y = z.to("cuda", dtype), t.to("cuda", dtype)
                    weights = torch.linspace(0.5, 2.0, x.shape[0], device="cuda")
                    weights = weights.to(torch.promote_types(dtype, torch.float32))
                    for double in (False, True) if dtype != torch.float64 else (False,):
                        args = (x, y, weights, temperature, double)
                        expected = _fused_rows(*args)
                        with mock.patch.multiple(
                            kernels, _CHUNK=0, _STREAM=(16, 1), _BACKWARD=(16, 1)
                        ):
                            actual = _fused_rows(*args)
                        fp32 = dtype != torch.float64 and not double
                        rtol, atol = (2e-5, 2e-6) if fp32 else (1e-13, 1e-14)
                        message = f"{dtype} double={double} T={temperature}"
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

    def test_cpu_kernel_confident_rows(self, subtests: pytest.Subtests):
        # The CPU kernel's vectorized branch (C >= 4) must not cancel the leading term.
        for name, device, context in paths():
            if device != "cpu":
                continue
            for dtype in (np.float32, np.float64):
                eps = _eps(dtype)
                for gap in (20.0, 40.0, 60.0):
                    for classes in (3, 4, 5, 8, 37):
                        z = np.full((1, classes), -gap, dtype=dtype)
                        z[0, 0] = 0
                        expected, _ = reference(z, z, 1.0)
                        with subtests.test(name=name, dtype=dtype, gap=gap, classes=classes):
                            with context():
                                x = torch.tensor(z)
                                value = distillation_loss(x, x, reduction="none")
                            error = abs(value.double().item() - expected[0]) / expected[0]
                            assert error <= 8 * eps

    def test_wide_rows_every_entry_within_budget(self, subtests: pytest.Subtests):
        # 128256 classes: every gradient entry, the largest p + q one included, within 64 eps
        # of the row's largest entry (an entry formed as minus the sum of the others carried
        # their roundings: hundreds of eps). The float64 rows take a long double reference.
        extended = np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
        gen = torch.Generator().manual_seed(7)
        student = torch.randn(2, 128256, generator=gen, dtype=torch.float64)
        noise = torch.randn(2, 128256, generator=gen, dtype=torch.float64)
        for name, device, context in paths():
            if device != "cpu":
                continue
            for dtype in (torch.float32, torch.float64):
                if dtype == torch.float64 and (not extended or name == "cpu-tensor"):
                    continue
                # Student and teacher apart, and nearly equal (the hardest rows).
                for spread in (0.3, 1e-3):
                    s, t = student.to(dtype), (student + spread * noise).to(dtype)
                    ls, lt = (x.numpy().astype(np.longdouble) / np.longdouble(2.0) for x in (s, t))
                    ls = ls - ls.max(1, keepdims=True)
                    ls = ls - np.log(np.exp(ls).sum(1, keepdims=True))
                    lt = lt - lt.max(1, keepdims=True)
                    lt = lt - np.log(np.exp(lt).sum(1, keepdims=True))
                    ratio, p = ls - lt, np.exp(lt)
                    expected = np.where(
                        np.abs(ratio) < 1, p * np.expm1(ratio), np.exp(ls) - p
                    ) / np.longdouble(2.0)
                    x = s.clone().requires_grad_()
                    with subtests.test(name=name, dtype=dtype, spread=spread), context():
                        loss = distillation_loss(x, t, 2.0, reduction="sum")
                        (gradient,) = torch.autograd.grad(loss, x)
                        error = np.abs(gradient.numpy().astype(np.longdouble) - expected)
                        scale = np.abs(expected).max(1, keepdims=True)
                        assert (error <= 64 * torch.finfo(dtype).eps * scale).all()

    def test_equal_logits_give_zero_gradient(self, subtests: pytest.Subtests):
        for name, device, context in paths():
            for dtype in DTYPES:
                # Far classes with subnormal (float64: also float32) teacher probabilities.
                gaps = (720.0,) if dtype == torch.float64 else (88.0, 95.0)
                far = [[0.0, -g, -g / 2, 1.0, -g - 1] for g in gaps]
                gen = torch.Generator().manual_seed(1)
                z = torch.cat([torch.tensor(far), 3 * torch.randn(3, 5, generator=gen)])
                z = z.to(device, dtype)
                for temperature in (0.5, 1.0, 4.0):
                    for proximal, alpha in ((False, 0.0), (True, 1e-3), (True, 1.0), (True, 30.0)):
                        message = f"{name} {dtype} T={temperature} alpha={alpha}"
                        with subtests.test(message), context():
                            fn = loss_fn(proximal, alpha)
                            value, gradient = evaluate(fn, z, z.clone(), temperature)
                            assert gradient.dtype == dtype
                            assert int(gradient.count_nonzero()) == 0, message
                            assert bool(value.isfinite().all()), message

    def test_proximal_kkt_limits_and_envelope(self):
        for name, device, context in paths():
            for dtype, tol in ((torch.float32, 2e-6), (torch.float64, 1e-13)):
                gen = torch.Generator().manual_seed(5)
                z = (3 * torch.randn(6, 9, generator=gen, dtype=torch.float64)).to(device, dtype)
                t = (3 * torch.randn(6, 9, generator=gen, dtype=torch.float64)).to(device, dtype)
                for temperature in (0.5, 1.0, 4.0):
                    p = torch.softmax(t.double() / temperature, -1)
                    with context():
                        distill, distill_g = evaluate(distillation_loss, z, t, temperature)
                        tiny_alpha = evaluate(loss_fn(True, 1e-12), z, t, temperature)
                        for alpha in (1e-4, 0.1, 1.0, 10.0, 1e4):
                            message = f"{name} {dtype} T={temperature} alpha={alpha}"
                            value, g = evaluate(loss_fn(True, alpha), z, t, temperature)
                            # softmax(z/T - alpha T g) - softmax(t/T) == T g, and the value.
                            r = g.double() * temperature
                            u = z.double() / temperature - alpha * r
                            pi = torch.softmax(u, -1)
                            kkt = (pi - p - r).abs().max().item()
                            assert kkt <= tol, message
                            envelope = -(p * torch.log_softmax(u, -1)).sum(-1)
                            envelope = envelope + alpha * (r * r).sum(-1) / 2
                            torch.testing.assert_close(
                                value.double(), envelope, rtol=8 * tol, atol=0, msg=message
                            )
                            assert bool((value <= distill * (1 + 8 * tol)).all())
                    torch.testing.assert_close(tiny_alpha[0], distill, rtol=4 * tol, atol=0)
                    torch.testing.assert_close(
                        tiny_alpha[1],
                        distill_g,
                        rtol=4 * tol,
                        atol=float(4 * tol * distill_g.abs().max()),
                    )

    def test_layout_reductions_and_autograd_contract(self, subtests: pytest.Subtests):
        for name, device, context in paths():
            gen = torch.Generator().manual_seed(2)
            teacher = torch.randn(2, 4, 3, generator=gen, dtype=torch.float64).to(device)
            z = teacher + 0.3 * torch.randn(2, 4, 3, generator=gen, dtype=torch.float64).to(device)
            for proximal in (False, True):
                fn = loss_fn(proximal, 0.6)
                with subtests.test(name=name, proximal=proximal), context():
                    x = z.clone().requires_grad_()
                    unreduced = fn(x, teacher, 2.5, dim=1, reduction="none")
                    assert unreduced.shape == (2, 3)
                    torch.testing.assert_close(
                        unreduced.sum(), fn(x, teacher, 2.5, dim=1, reduction="sum")
                    )
                    torch.testing.assert_close(unreduced.mean(), fn(x, teacher, 2.5, dim=1))
                    moved = fn(x.movedim(1, -1), teacher.movedim(1, -1), 2.5, reduction="none")
                    torch.testing.assert_close(moved, unreduced)
                    assert gradcheck(lambda v: fn(v, teacher, 2.5, dim=1), (x,))
                    (g,) = torch.autograd.grad(fn(x, teacher, 2.5, dim=1), x, create_graph=True)
                    with pytest.raises(RuntimeError):
                        torch.autograd.grad(g.sum(), x)
                    learner = teacher.clone().requires_grad_()
                    fn(x, learner, 2.5).backward()
                    assert learner.grad is None
                    for dtype in (torch.float16, torch.bfloat16):
                        y = z.to(dtype).requires_grad_()
                        value = fn(y, teacher.to(dtype), 2.5, dim=1, reduction="none")
                        assert value.dtype == torch.float32
                        value.sum().backward()
                        assert y.grad is not None
                        assert y.grad.dtype == dtype

    def test_modules_and_validation(self, subtests: pytest.Subtests):
        z = torch.randn(3, 4)
        torch.testing.assert_close(
            ProximalDistillationLoss(2.0, 0.5, reduction="sum")(z, z.flip(-1)),
            proximal_distillation_loss(z, z.flip(-1), 2.0, 0.5, reduction="sum"),
        )
        torch.testing.assert_close(
            DistillationLoss(2.0, reduction="sum")(z, z.flip(-1)),
            distillation_loss(z, z.flip(-1), 2.0, reduction="sum"),
        )
        for name, device, context in paths():
            x = z.to(device)
            with subtests.test(name=name), context():
                for fn in (distillation_loss, loss_fn(True, 0.5)):
                    for teacher, options in [
                        (x.double(), {}),
                        (x[:1], {}),
                        (x, dict(temperature=0.0)),
                        (x, dict(temperature=math.inf)),
                        (x, dict(reduction="bad")),
                        (x, dict(precision="half")),
                    ]:
                        temperature = options.pop("temperature", 1.0)
                        with pytest.raises(ValueError):
                            fn(x, teacher, temperature, **options)
                    for bad in (x[:, :1], x.long()):
                        with pytest.raises(ValueError):
                            fn(bad, bad, 1.0)
                for alpha in (0.0, -1.0, math.nan):
                    with pytest.raises(ValueError):
                        proximal_distillation_loss(x, x, 1.0, alpha)
                # Nonfinite logits: an error from the CPU kernels, nan rows (only) from the
                # CUDA kernels and the tensor operations.
                y = x.clone()
                y[1, 2] = math.inf
                for fn in (distillation_loss, loss_fn(True, 0.5)):
                    if name in ("cpu-native", "cpu-kernel"):
                        with pytest.raises(ValueError):
                            fn(y, x, 1.0)
                    else:
                        value = fn(y, x, 1.0, reduction="none")
                        assert value.isnan().tolist() == [False, True, False]
