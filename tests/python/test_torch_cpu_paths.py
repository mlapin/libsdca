"""Regressions of the PyTorch CPU paths: gradient rounding, double backward, empty batches,
retained graphs and the distillation offset."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import libsdca.torch as sdca
import pytest
import torch
import torch.nn.functional as F
from child_process import pythonpath, run_python
from libsdca import _cpu, _cpu_entropy


def _default_backends() -> list[str]:
    """The SDCA_CPU_EXTENSION settings to test: the extension unless it is disabled or unavailable."""
    if _cpu_entropy.mode() == "off" or not _cpu_entropy.enabled():
        return ["off"]
    return ["off", "on"]


class HalfGradientRoundingTests:
    """16-bit gradients round once, after the upstream factor (an fp16 GradScaler's 65536)."""

    def check(self, loss: Callable[..., torch.Tensor], z: torch.Tensor, *args: object) -> None:
        x = z.detach().clone().requires_grad_(True)
        (loss(x, *args) * 65536).backward()
        x64 = z.detach().double().requires_grad_(True)
        (loss(x64, *args) * 65536).backward()
        assert x.grad is not None and x64.grad is not None
        expected = x64.grad.to(z.dtype).double()
        # Each entry within one unit of its own rounding of the float64 gradient: rounding
        # before the factor loses the subnormal float16 entries (over 100 units).
        info = torch.finfo(z.dtype)
        unit = info.eps * expected.abs() + info.smallest_normal * info.eps
        assert float(((x.grad.double() - expected).abs() / unit).max()) <= 1.0

    def test_tensor_and_native_losses(self, subtests: pytest.Subtests) -> None:
        g = torch.Generator().manual_seed(0)
        for flag in _default_backends():
            with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": flag}):
                for dtype in (torch.float16, torch.bfloat16):
                    z = (3 * torch.randn(64, 40, generator=g)).to(dtype)
                    y = torch.randint(0, 40, (64,), generator=g)
                    t = (3 * torch.randn(64, 40, generator=g)).to(dtype)
                    with subtests.test(flag=flag, dtype=dtype):
                        self.check(lambda x, y: sdca.topk_cross_entropy(x, y, 2), z, y)
                        self.check(lambda x, y: sdca.topk_hinge_loss(x, y, 3, gamma=0.5), z, y)
                        self.check(lambda x, t: sdca.distillation_loss(x, t.to(x.dtype), 2.0), z, t)
                        self.check(
                            lambda x, t: sdca.proximal_distillation_loss(x, t.to(x.dtype)), z, t
                        )
                        self.check(lambda x, y: sdca.proximal_cross_entropy(x, y), z, y)


class RetainedGraphTests:
    def test_backward_twice_with_retained_graph(self) -> None:
        g = torch.Generator().manual_seed(1)
        z = torch.randn(16, 9, generator=g, requires_grad=True)
        y = torch.randint(0, 9, (16,), generator=g)
        for loss in (
            lambda: sdca.topk_hinge_loss(z, y, 2),
            lambda: sdca.distillation_loss(z, z.detach().flip(0)),
            lambda: sdca.multilabel_hinge_loss(z, F.one_hot(y, 9).bool()),
        ):
            value = loss()
            first = torch.autograd.grad(value, z, retain_graph=True)[0]
            second = torch.autograd.grad(value, z)[0]
            assert torch.equal(first, second)


class DoubleBackwardTests:
    def test_default_cpu_entropy_has_second_derivatives(self) -> None:
        g = torch.Generator().manual_seed(2)
        y = torch.randint(0, 5, (8,), generator=g)
        for dtype in (torch.float32, torch.float64):
            for reduction in ("mean", "sum", "none"):
                x = torch.randn(8, 5, generator=g, dtype=dtype, requires_grad=True)
                ref = x.detach().double().requires_grad_(True)
                out = sdca.cross_entropy(x, y, reduction=reduction).sum()
                expected = F.cross_entropy(ref, y, reduction=reduction).sum()
                (grad,) = torch.autograd.grad(out, x, create_graph=True)
                (ref_grad,) = torch.autograd.grad(expected, ref, create_graph=True)
                (hvp,) = torch.autograd.grad(grad.square().sum(), x)
                (ref_hvp,) = torch.autograd.grad(ref_grad.square().sum(), ref)
                tol = 1e-12 if dtype == torch.float64 else 1e-6
                assert float((hvp.double() - ref_hvp).abs().max()) < tol


class EmptyBatchTests:
    def test_losses_follow_functional_semantics(self, subtests: pytest.Subtests) -> None:
        y = torch.zeros(0, dtype=torch.int64)
        for reduction in ("none", "sum", "mean"):
            reference = F.cross_entropy(torch.zeros(0, 5), y, reduction=reduction)
            calls = {
                "sdca_loss": lambda x: sdca.sdca_loss(x, y, reduction=reduction),
                "cross_entropy": lambda x: sdca.cross_entropy(x, y, reduction=reduction),
                "hinge": lambda x: sdca.multiclass_hinge_loss(x, y, reduction=reduction),
                "topk": lambda x: sdca.topk_cross_entropy(x, y, 2, reduction=reduction),
                "multilabel": lambda x: sdca.multilabel_hinge_loss(
                    x, torch.zeros(0, 5, dtype=torch.bool), reduction=reduction
                ),
                "distillation": lambda x: sdca.distillation_loss(
                    x, torch.zeros(0, 5, dtype=x.dtype), reduction=reduction
                ),
                "proximal_distillation": lambda x: sdca.proximal_distillation_loss(
                    x, torch.zeros(0, 5, dtype=x.dtype), reduction=reduction
                ),
            }
            for name, call in calls.items():
                for dtype in (torch.bfloat16, torch.float32):
                    with subtests.test(name=name, reduction=reduction, dtype=dtype):
                        x = torch.zeros(0, 5, dtype=dtype, requires_grad=True)
                        value = call(x)
                        assert value.shape == reference.shape
                        assert value.dtype == torch.float32
                        torch.testing.assert_close(value, reference, equal_nan=True)
                        value.sum().backward()
                        assert x.grad is not None
                        assert x.grad.shape == x.shape

    def test_one_class_still_raises(self) -> None:
        with pytest.raises(ValueError, match="two classes"):
            sdca.sdca_loss(torch.zeros(3, 1), torch.zeros(3, dtype=torch.int64))


class DistillationOffsetTests:
    def test_common_student_offset_keeps_the_gradient(self, subtests: pytest.Subtests) -> None:
        """Softmax is shift invariant: a large student offset must not round the gradient."""
        g = torch.Generator().manual_seed(2)
        base = torch.randn(64, 10, generator=g)
        teacher = base + 0.1 * torch.randn(64, 10, generator=g)
        for offset in (300.0, 1e4):
            for temperature in (1.0, 4.0):
                for loss in (sdca.distillation_loss, sdca.proximal_distillation_loss):
                    student = (base + offset).requires_grad_(True)
                    loss(student, teacher, temperature, reduction="sum").backward()
                    s64 = student.detach().double().requires_grad_(True)
                    loss(s64, teacher.double(), temperature, reduction="sum").backward()
                    assert student.grad is not None and s64.grad is not None
                    error = (student.grad.double() - s64.grad).abs().max()
                    scale = s64.grad.abs().max()
                    with subtests.test(offset=offset, temperature=temperature, loss=loss):
                        assert float(error / scale) < 1e-5


def _recorded(calls: list[str]) -> Callable[..., torch.Tensor]:
    """_cpu.loss, appending each call's kernel kind to calls."""
    loss = _cpu.loss

    def spy(kind: str, *args: Any) -> torch.Tensor:
        calls.append(kind)
        return loss(kind, *args)

    return spy


@pytest.mark.skipif(not _cpu_entropy.buildable(), reason="needs the CPU extension's build tools")
class DistillationGradientKernelTests:
    """A differentiable distillation_loss call on float32 or float64 logits on the extension
    saves its gradient in the forward (distillation_grad); forward-only calls, 16-bit logits,
    float64 work on narrower logits and temperatures whose reciprocal overflows keep the
    statistics kernel. Both give the same values and gradients to rounding."""

    def call(
        self,
        saving: bool,
        z: torch.Tensor,
        t: torch.Tensor,
        temperature: float,
        upstream: float,
        reduction: str = "none",
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """(values, gradient under an upstream of upstream * [0.5, 2] (reduced: upstream),
        kernels called)."""
        calls: list[str] = []
        chooser = _cpu.distillation_grad if saving else (lambda *_: False)
        x = z.detach().clone().requires_grad_(True)
        with (
            mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": "on"}),
            mock.patch.object(_cpu, "loss", _recorded(calls)),
            mock.patch.object(_cpu, "distillation_grad", chooser),
        ):
            value = sdca.distillation_loss(x, t, temperature, reduction=reduction)
            weights = torch.linspace(0.5, 2.0, value.numel(), dtype=value.dtype)
            if reduction != "none":
                weights = torch.ones((), dtype=value.dtype)
            (gradient,) = torch.autograd.grad(value, x, weights * upstream)
        return value.detach(), gradient, calls

    def test_kernels_agree(self, subtests: pytest.Subtests) -> None:
        g = torch.Generator().manual_seed(3)
        for c in (3, 10, 1000):
            z = 3 * torch.randn(17, c, generator=g)
            t = z + torch.randn(17, c, generator=g)
            for dtype in (torch.float32, torch.float64):
                with subtests.test(c=c, dtype=dtype):
                    zs, ts = z.to(dtype), t.to(dtype)
                    value, grad, calls = self.call(True, zs, ts, 2.0, 2.0**-12)
                    assert calls == ["distillation_grad"]
                    expected, expected_grad, calls = self.call(False, zs, ts, 2.0, 2.0**-12)
                    assert calls == ["distillation_stats"]
                    eps = torch.finfo(dtype).eps
                    assert float((value - expected).abs().max()) <= 8 * eps
                    # Relative to each row's largest entry.
                    largest = expected_grad.abs().amax(-1, keepdim=True)
                    error = ((grad - expected_grad).abs() / largest).max()
                    assert float(error) <= 64 * eps

    def test_other_calls_keep_the_statistics(self) -> None:
        z = torch.randn(5, 7, generator=torch.Generator().manual_seed(4))
        calls: list[str] = []
        with (
            mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": "on"}),
            mock.patch.object(_cpu, "loss", _recorded(calls)),
        ):
            with torch.no_grad():
                sdca.distillation_loss(z.requires_grad_(True), z.flip(1), 2.0)
            for dtype, temperature, precision in (
                (torch.float32, 2.0, "float64"),
                (torch.bfloat16, 2.0, None),
                (torch.float16, 2.0, None),
                (torch.float32, 1e-39, None),
                (torch.float64, 1e-310, None),
            ):
                x = z.detach().to(dtype).requires_grad_(True)
                # Equal logits: finite at any temperature.
                sdca.distillation_loss(x, x.detach(), temperature, precision=precision).backward()
        assert calls == ["distillation_stats"] * 6

    def test_overflowing_factor_scales_over_the_temperature_first(
        self, subtests: pytest.Subtests
    ) -> None:
        """upstream / T beyond float32 (a tiny temperature): the saved gradient over T, then
        times upstream, as the statistics kernel's."""
        g = torch.Generator().manual_seed(5)
        z = 3e-21 * torch.randn(4, 6, generator=g)
        t = z + 1e-22 * torch.randn(4, 6, generator=g)
        for reduction, upstream in (("none", 1e18), ("sum", 1e18), ("mean", 4e18)):
            with subtests.test(reduction=reduction):
                _, grad, calls = self.call(True, z, t, 1e-21, upstream, reduction)
                assert calls == ["distillation_grad"]
                _, expected, _ = self.call(False, z, t, 1e-21, upstream, reduction)
                assert bool(torch.isfinite(expected).all())
                largest = expected.abs().amax(-1, keepdim=True)
                error = ((grad - expected).abs() / largest).max()
                assert float(error) <= 64 * torch.finfo(torch.float32).eps


# A child that resolves the extension like the first CPU loss call: prints enabled() and the
# reason, after a call that must succeed either way.
STATUS_CHILD = """
import warnings, torch
import libsdca._cpu_entropy as ext
from libsdca.torch import cross_entropy
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    cross_entropy(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))
    print(ext.enabled(), repr(ext.why_not()), ext.operators.cache_info().currsize)
    print(sum("did not build" in str(w.message) for w in caught))
    print(sum("slower native library" in str(w.message) for w in caught))
"""


# A child that loads (or builds) the extension and calls one of its kernels.
BUILD_CHILD = """
import torch
import libsdca._cpu_entropy as ext
z, y = torch.randn(4, 3), torch.tensor([0, 1, 2, 0])
value, _ = ext.operators().hinge_forward(z, y, 2, False, 1.0)
print(bool(torch.isfinite(value)))
"""


class AutomaticExtensionTests:
    """SDCA_CPU_EXTENSION unset builds the extension on first use and never breaks a loss call."""

    def child(self, directory: str, **env: str) -> list[str]:
        base = {k: v for k, v in os.environ.items() if k != "SDCA_CPU_EXTENSION"}
        base.update(TORCH_EXTENSIONS_DIR=directory, **env)
        result = run_python("-c", STATUS_CHILD, env=base, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr[-3000:]
        return result.stdout.split("\n")

    @pytest.mark.slow
    def test_failed_build_falls_back_once_and_is_remembered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            # A compiler that fails every build (the ninja and header checks pass).
            compiler = Path(directory) / "failing-c++"
            compiler.write_text("#!/bin/sh\nexit 1\n")
            compiler.chmod(0o755)
            first = self.child(directory, CXX=str(compiler))
            if "ninja is not installed" in first[0] or "headers" in first[0]:
                pytest.skip(first[0])
            assert first[0].startswith("False") and "did not build" in first[0], first
            assert first[1] == "1"
            records = list(Path(directory).glob("*." + _cpu_entropy.FAILED))
            assert len(records) == 1
            # The next process skips the build without warning; explain_backend names the
            # record and its reason.
            second = self.child(directory, CXX=str(compiler))
            assert second[0].startswith("False ") and "an earlier build failed" in second[0]
            assert str(records[0]) in second[0]
            assert "did not build" in second[0]
            assert second[1] == "0"
            # A record older than RETRY seconds expires: the next process builds again.
            old = time.time() - _cpu_entropy.RETRY - 60
            os.utime(records[0], (old, old))
            third = self.child(directory, CXX=str(compiler))
            assert third[0].startswith("False") and "did not build" in third[0], third
            assert third[1] == "1"

    @pytest.mark.skipif(
        not _cpu_entropy.buildable(), reason="needs ninja, a C++ compiler and Python.h"
    )
    @pytest.mark.slow
    def test_concurrent_first_builds_share_one_complete_library(self) -> None:
        """Processes building at once each load a complete library; one build stays cached."""
        with tempfile.TemporaryDirectory() as directory:
            env = {k: v for k, v in os.environ.items() if k != "SDCA_CPU_EXTENSION"}
            env.update(TORCH_EXTENSIONS_DIR=directory, SDCA_CPU_EXTENSION="on")
            children = [
                subprocess.Popen(
                    [sys.executable, "-c", BUILD_CHILD],
                    env={**env, "PYTHONPATH": pythonpath(env)},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for _ in range(3)
            ]
            for child in children:
                out, err = child.communicate(timeout=900)
                assert child.returncode == 0, err[-3000:]
                assert out.startswith("True"), out
            builds = [p for p in Path(directory).iterdir() if p.is_dir()]
            assert len(builds) == 1, builds
            assert (builds[0] / f"{builds[0].name}.so").is_file()
            # A later process loads the cached build.
            later = run_python("-c", BUILD_CHILD, env=env, capture_output=True, text=True)
            assert later.stdout.strip() == "True", later.stderr[-3000:]

    def test_missing_tools_fall_back_without_building(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lines = self.child(directory, CXX="/nonexistent/c++")
            assert lines[0].startswith("False 'no C++ compiler (c++ or CXX) on PATH")
            assert lines[0].endswith(" 0"), lines
            assert lines[1] == "0"
            # One hint per process names the missing tool.
            assert lines[2] == "1"
            off = self.child(directory, CXX="/nonexistent/c++", SDCA_CPU_EXTENSION="off")
            assert off[0] == "False 'SDCA_CPU_EXTENSION=off' 0"
            assert off[2] == "0"
            # on raises, naming the missing tool.
            env = {**os.environ, "TORCH_EXTENSIONS_DIR": directory, "CXX": "/nonexistent/c++"}
            env["SDCA_CPU_EXTENSION"] = "on"
            on = run_python("-c", STATUS_CHILD, env=env, capture_output=True, text=True)
            assert on.returncode != 0
            message = "RuntimeError: libsdca: the CPU extension cannot be built here: no C++"
            assert message in on.stderr

    def test_modes(self) -> None:
        for value, expected in [("", "auto"), ("auto", "auto"), ("ON", "on"), ("1", "on")]:
            with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": value}):
                assert _cpu_entropy.mode() == expected
        for value in ("off", "0"):
            with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": value}):
                assert _cpu_entropy.mode() == "off"
                assert _cpu_entropy.why_not() == "SDCA_CPU_EXTENSION=off"
        with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": "yes"}):
            with pytest.raises(ValueError, match="auto, on or off"):
                _cpu_entropy.mode()

    def test_explain_backend_names_the_missing_tool(self) -> None:
        reason = "ninja is not installed (pip install ninja, or libsdca[torch])"
        with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": "auto"}):
            with mock.patch.object(_cpu_entropy, "status", return_value=(None, reason)):
                backend, why = sdca.explain_backend(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))
        assert backend == "native"
        assert reason in why

    def test_disabled_never_resolves(self) -> None:
        with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": "off"}):
            with mock.patch.object(_cpu_entropy, "status") as status:
                assert not _cpu_entropy.enabled()
                sdca.cross_entropy(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))
                status.assert_not_called()
