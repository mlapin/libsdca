"""One dtype rule for every PyTorch loss: values promote(logits, float32), gradients logits."""

import os
from collections.abc import Callable
from unittest import mock

import torch
from libsdca import _cpu_entropy
from libsdca.torch import (
    cross_entropy,
    distillation_loss,
    proximal_distillation_loss,
    sdca_loss,
)

DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
REDUCTIONS = ("none", "sum", "mean")
# (name, loss, target kind, keyword options): every loss, nonsmooth and smoothed.
CASES = [
    ("cross_entropy", "cross_entropy", "index", {}),
    ("cross_entropy mask", "cross_entropy", "mask", {}),
    ("topk_cross_entropy k=1", "topk_cross_entropy", "index", {"k": 1}),
    ("topk_cross_entropy k=2", "topk_cross_entropy", "index", {"k": 2}),
    ("multiclass_hinge", "multiclass_hinge", "index", {}),
    ("multiclass_hinge gamma", "multiclass_hinge", "index", {"gamma": 0.5}),
    ("topk_hinge_alpha", "topk_hinge_alpha", "index", {"k": 2}),
    ("topk_hinge_alpha gamma", "topk_hinge_alpha", "index", {"k": 2, "gamma": 0.5}),
    ("topk_hinge_beta", "topk_hinge_beta", "index", {"k": 2}),
    ("topk_hinge_beta gamma", "topk_hinge_beta", "index", {"k": 2, "gamma": 0.5}),
    ("multilabel_hinge", "multilabel_hinge", "mask", {}),
    ("multilabel_hinge gamma", "multilabel_hinge", "mask", {"gamma": 0.5}),
    ("multilabel_cross_entropy", "multilabel_cross_entropy", "mask", {}),
    ("proximal_cross_entropy", "proximal_cross_entropy", "index", {"alpha": 0.5}),
    ("proximal_cross_entropy mask", "proximal_cross_entropy", "mask", {"alpha": 0.5}),
]


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class LossDtypeTests:
    def setup_method(self):
        self.gen = torch.Generator().manual_seed(0)

    def inputs(self, kind: str, device: str, dtype: torch.dtype, n: int = 5, c: int = 7):
        z = torch.randn(n, c, generator=self.gen).to(device, dtype)
        if kind == "index":
            return z, torch.randint(c, (n,), generator=self.gen).to(device)
        mask = torch.rand(n, c, generator=self.gen) < 0.4
        mask[:, 0], mask[:, 1] = True, False
        return z, mask.to(device)

    def check(
        self,
        fn: Callable[[torch.Tensor, str], torch.Tensor],
        z: torch.Tensor,
        reduction: str,
        message: str,
    ):
        x = z.detach().clone().requires_grad_()
        value = fn(x, reduction)
        expected = torch.promote_types(z.dtype, torch.float32)
        assert value.dtype == expected, message
        assert value.shape == ((z.shape[0],) if reduction == "none" else ()), message
        (gradient,) = torch.autograd.grad(value.sum(), x)
        assert gradient.dtype == z.dtype, message
        assert bool(value.isfinite().all()), message

    def test_every_loss_dtype_and_reduction(self):
        for device in _devices():
            # 1: the CPU extension, where it builds.
            for kernel in (
                ("off", "on") if device == "cpu" and _cpu_entropy.buildable() else ("off",)
            ):
                with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": kernel}):
                    for dtype in DTYPES:
                        for name, loss, kind, options in CASES:
                            z, target = self.inputs(kind, device, dtype)
                            for reduction in REDUCTIONS:
                                message = f"{device} cpu={kernel} {dtype} {name} {reduction}"
                                self.check(
                                    lambda x, r, t=target, o=options, n=loss: sdca_loss(
                                        x, t, n, reduction=r, **o
                                    ),
                                    z,
                                    reduction,
                                    message,
                                )

    def test_weighted_and_teacher_losses(self):
        for device in _devices():
            for dtype in DTYPES:
                z, labels = self.inputs("index", device, dtype)
                teacher = torch.randn(z.shape, generator=self.gen).to(device, dtype)
                weight = torch.linspace(0.5, 2.0, z.shape[1]).to(device, dtype)
                cases = {
                    "weighted cross_entropy": lambda x, r: cross_entropy(
                        x, labels, weight, reduction=r
                    ),
                    "distillation": lambda x, r: distillation_loss(x, teacher, 2.0, reduction=r),
                    "proximal_distillation": lambda x, r: proximal_distillation_loss(
                        x, teacher, 2.0, 0.5, reduction=r
                    ),
                }
                for name, fn in cases.items():
                    for reduction in REDUCTIONS:
                        self.check(fn, z, reduction, f"{device} {dtype} {name} {reduction}")
