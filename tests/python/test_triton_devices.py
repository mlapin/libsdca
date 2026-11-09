"""Triton launches use their tensor's device and preserve the caller's state."""

import os
from unittest import mock

import pytest
import torch
from libsdca import _cuda_launch
from libsdca.torch import sdca_loss


class DeviceGuardTests:
    def test_cpu_interpreter_guard_never_enters_cuda(self):
        tensor = torch.zeros(1)
        with mock.patch("torch.cuda.device", side_effect=AssertionError("CPU entered CUDA")):
            with _cuda_launch.device_guard(tensor):
                assert not tensor.is_cuda


@pytest.mark.skipif(not (torch.cuda.device_count() >= 2), reason="two CUDA devices required")
class TritonDeviceTests:
    def setup_method(self):
        self.previous = torch.cuda.current_device()
        torch.cuda.set_device(0)
        self.setting = mock.patch.dict(os.environ, {_cuda_launch.ENV: "off"})
        self.setting.start()
        _cuda_launch.status.cache_clear()
        generator = torch.Generator().manual_seed(348)
        self.scores = torch.randn(7, 5, generator=generator, dtype=torch.float64)
        labels = torch.arange(7) % 5
        mask = torch.zeros(7, 5, dtype=torch.bool)
        mask[:, 0] = mask[:, 2] = True
        self.cases = [("cross_entropy", 1, labels), ("topk_hinge_beta", 1, labels)]
        self.cases += [
            (loss, 2, labels)
            for loss in (
                "topk_cross_entropy",
                "topk_hinge_alpha",
                "topk_hinge_beta",
            )
        ]
        self.cases += [
            (loss, 1, mask)
            for loss in (
                "cross_entropy",
                "multilabel_hinge",
                "proximal_cross_entropy",
            )
        ]

    def teardown_method(self):
        self.setting.stop()
        _cuda_launch.status.cache_clear()
        torch.cuda.set_device(self.previous)

    def test_public_forward_backward_match_cpu_and_preserve_device(self, subtests: pytest.Subtests):
        for setting in ("off", "on"):
            with mock.patch.dict(os.environ, {_cuda_launch.ENV: setting}):
                _cuda_launch.status.cache_clear()
                for dtype in (torch.float32, torch.float64):
                    for loss, k, target in self.cases:
                        with subtests.test(
                            launcher=setting,
                            dtype=dtype,
                            loss=loss,
                            k=k,
                            mask=target.dtype == torch.bool,
                        ):
                            reference = self.scores.to(dtype).requires_grad_()
                            expected = sdca_loss(
                                reference,
                                target,
                                loss,
                                k=k,
                                gamma=0.7,
                                alpha=0.4,
                                backend="native",
                                reduction="none",
                            )
                            upstream = torch.linspace(-0.8, 1.2, 7, dtype=dtype)
                            expected_gradient = torch.autograd.grad(expected, reference, upstream)[
                                0
                            ]
                            z = reference.detach().to("cuda:1").requires_grad_()
                            actual = sdca_loss(
                                z,
                                target.to(z.device),
                                loss,
                                k=k,
                                gamma=0.7,
                                alpha=0.4,
                                reduction="none",
                            )
                            assert torch.cuda.current_device() == 0
                            gradient = torch.autograd.grad(actual, z, upstream.to(z.device))[0]
                            assert torch.cuda.current_device() == 0
                            rtol, atol = (3e-6, 2e-7) if dtype == torch.float32 else (2e-11, 2e-12)
                            torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
                            torch.testing.assert_close(
                                gradient.cpu(), expected_gradient, rtol=rtol, atol=atol
                            )

    def test_validation_errors_restore_device(self, subtests: pytest.Subtests):
        for loss, k, target in self.cases:
            with subtests.test(loss=loss, k=k, mask=target.dtype == torch.bool):
                z = self.scores.to("cuda:1")
                z[0, 0] = torch.nan
                out = sdca_loss(z, target.to(z.device), loss, k=k, gamma=0.7, alpha=0.4)
                assert bool(out.isnan().all())
                assert torch.cuda.current_device() == 0

    def test_raw_backward_uses_input_device_stream_and_restores_caller(
        self, subtests: pytest.Subtests
    ):
        from libsdca import _triton_entropy, _triton_hinge, _triton_multilabel, _triton_topk

        stream = torch.cuda.Stream(device=1)
        with torch.cuda.device(1):
            previous_stream = torch.cuda.current_stream()
            torch.cuda.set_stream(stream)
            z = self.scores.to("cuda:1")
            y = (torch.arange(7) % 5).to(z.device)
            flags = torch.zeros_like(z, dtype=torch.bool)
            flags[:, 0] = flags[:, 2] = True
            u = torch.tensor(0.75, device=z.device, dtype=z.dtype)
        backends = [
            (
                "cross_entropy",
                lambda: _triton_entropy.forward(z, y, "mean"),
                lambda s: _triton_entropy.backward(z, y, s, u, "mean"),
            ),
            (
                "hinge",
                lambda: _triton_hinge.forward(z, y, "mean", True, 0.7),
                lambda s: _triton_hinge.backward(z, y, s, u, "mean", True, 0.7),
            ),
            (
                "topk",
                lambda: _triton_topk.forward(z, y, "mean", "topk_hinge_beta_smooth", 2, 0.7),
                lambda s: _triton_topk.backward(
                    z, y, s, u, "mean", "topk_hinge_beta_smooth", 2, 0.7
                ),
            ),
            (
                "multilabel",
                lambda: _triton_multilabel.forward(z, flags, "cross_entropy", "mean", 0.7),
                lambda s: _triton_multilabel.backward(z, flags, s, u, "cross_entropy", "mean", 0.7),
            ),
        ]
        try:
            for name, forward, backward in backends:
                with subtests.test(kernel=name):
                    with torch.cuda.device(1):
                        _, stats = forward()
                        expected = backward(stats)
                    actual = backward(stats)
                    assert torch.cuda.current_device() == 0
                    assert torch.cuda.current_stream(1) == stream
                    stream.synchronize()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            stream.synchronize()
            with torch.cuda.device(1):
                torch.cuda.set_stream(previous_stream)
