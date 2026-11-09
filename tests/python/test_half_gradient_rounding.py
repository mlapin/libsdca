"""16-bit gradients rounded once: every entry against a float64 reference, at batch sizes
where the scaled entries are float16 subnormals.

A mean over N rows scales each entry by 1 / N before its only rounding into the logits dtype.
At N = 4096 most float16 entries of a 1,000-class row are subnormal, and an entry derived
from other already-rounded entries (the CPU extension's top-k target entry was minus the sum
of the stored competitors) collects their rounding: 1.6% of the largest entry here, 16% at
32,000 classes. Every loss, top-k at k > 1, reductions mean, sum and none (with a 2^-12
upstream), both CPU paths (SDCA_CPU_EXTENSION on and off) and CUDA where available,
default and float64 precision.
The Triton kernels run the same checks in Triton's CPU interpreter (TRITON_INTERPRET=1) on a
few rows under a 2^-12 upstream; triton_check(n=4096) runs them on a CUDA device.

The envelope is that of test_half_precision.py: one rounding of each entry (one unit of the
logits dtype, plus one subnormal quantum), and with float32 work 2,000 float32 eps of the
row's largest entry.
"""

import os
from collections.abc import Callable
from typing import Any
from unittest import mock

import interpreter
import pytest
import torch
from libsdca import _cpu, _cpu_entropy
from libsdca.torch import distillation_loss, proximal_distillation_loss, sdca_loss

N, C = 4096, 1000
HALF = (torch.float16, torch.bfloat16)
REDUCTIONS = ("mean", "sum", "none")
EPS32 = torch.finfo(torch.float32).eps
ROW_EPS = 2000
PATHS = {"cpu-native": "off", **({"cpu-kernel": "on"} if _cpu_entropy.buildable() else {})}
# reduction="none" takes this upstream (times weights in [0.5, 2]), as a mean over 4096 rows.
SMALL_UPSTREAM = 2.0**-12

# name -> (loss, target kind, k, gamma)
LOSSES = {
    "cross_entropy": ("cross_entropy", "index", 1, 0.0),
    "cross_entropy mask": ("cross_entropy", "mask", 1, 0.0),
    "topk_cross_entropy k=5": ("topk_cross_entropy", "index", 5, 0.0),
    "multiclass_hinge": ("topk_hinge_alpha", "index", 1, 0.0),
    "multiclass_hinge smooth": ("topk_hinge_alpha", "index", 1, 0.5),
    "topk_hinge_alpha k=5": ("topk_hinge_alpha", "index", 5, 0.0),
    "topk_hinge_beta k=5": ("topk_hinge_beta", "index", 5, 0.0),
    "topk_hinge_alpha k=5 smooth": ("topk_hinge_alpha", "index", 5, 0.5),
    "topk_hinge_beta k=5 smooth": ("topk_hinge_beta", "index", 5, 0.5),
    "multilabel_cross_entropy": ("multilabel_cross_entropy", "mask", 1, 0.0),
    "multilabel_hinge": ("multilabel_hinge", "mask", 1, 0.0),
    "multilabel_hinge smooth": ("multilabel_hinge", "mask", 1, 0.5),
    "proximal_cross_entropy": ("proximal_cross_entropy", "index", 1, 0.0),
}
TEACHER: dict[str, Callable[..., torch.Tensor]] = {
    "distillation_loss": lambda x, t, **kw: distillation_loss(x, t, 2.0, **kw),
    "proximal_distillation_loss": lambda x, t, **kw: proximal_distillation_loss(
        x, t, 2.0, 0.7, **kw
    ),
}


def _data(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Logits (3 randn, peaked rows), labels, a positive mask and teacher logits."""
    gen = torch.Generator().manual_seed(0)
    z = 3 * torch.randn(N, C, generator=gen)
    labels = torch.randint(0, C, (N,), generator=gen)
    mask = torch.rand(N, C, generator=gen) < 0.01
    rows = torch.arange(N)
    mask[rows, labels], mask[rows, (labels + 1) % C] = True, False
    teacher = z + torch.randn(N, C, generator=gen)
    return z.to(dtype), labels, mask, teacher.to(dtype)


def _upstream(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        return torch.ones_like(value)
    weights = torch.linspace(0.5, 2.0, value.numel(), dtype=value.dtype, device=value.device)
    return SMALL_UPSTREAM * weights


def _gradient(call: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor) -> torch.Tensor:
    x = z.detach().clone().requires_grad_()
    value = call(x)
    (gradient,) = torch.autograd.grad(value, x, _upstream(value))
    return gradient


def _check(z: torch.Tensor, grad: torch.Tensor, ref: torch.Tensor, work: int, what: str) -> None:
    assert grad.dtype == z.dtype, what
    info = torch.finfo(z.dtype)
    g, r = grad.double(), ref.double()
    big = r.abs().amax(-1, keepdim=True)
    bound = info.eps * r.abs() + work * ROW_EPS * EPS32 * big + info.tiny * info.eps
    excess = (g - r).abs() - bound
    worst = int(excess.argmax())
    row, col = divmod(worst, z.shape[1])
    assert excess.max().item() <= 0, (
        f"{what}: entry ({row}, {col}) {g[row, col].item():.6g} vs {r[row, col].item():.6g}, "
        f"{(g - r)[row, col].abs().item() / big[row].item():.3g} of the row's largest"
    )


def _paths(z: torch.Tensor, call: Callable[..., torch.Tensor], what: str, reduction: str) -> None:
    """Every CPU path (and CUDA, where available) and precision against the native float64
    result of the represented logits; call(x, **options) moves its targets to x's device."""
    paths = {**PATHS, **({"cuda": "off"} if torch.cuda.is_available() else {})}
    with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": "off"}):
        ref = _gradient(lambda x: call(x, reduction=reduction), z.double())
    for name, flag in paths.items():
        x = z.cuda() if name == "cuda" else z
        for precision in (None, "float64"):
            with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": flag}):
                grad = _gradient(lambda v: call(v, reduction=reduction, precision=precision), x)
            work = 0 if precision == "float64" else 1
            message = f"{what} {z.dtype} {reduction} {name} precision={precision}"
            _check(z, grad.cpu(), ref, work, message)


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("dtype", HALF, ids=["float16", "bfloat16"])
class HalfGradientRoundingTests:
    @pytest.mark.parametrize("name", list(LOSSES))
    def test_losses(self, dtype: torch.dtype, name: str, reduction: str):
        loss, kind, k, gamma = LOSSES[name]
        z, labels, mask, _ = _data(dtype)
        target = labels if kind == "index" else mask

        def call(x: torch.Tensor, **options: Any) -> torch.Tensor:
            return sdca_loss(x, target.to(x.device), loss, k=k, gamma=gamma, **options)

        _paths(z, call, name, reduction)

    @pytest.mark.parametrize("name", list(TEACHER))
    def test_distillation(self, dtype: torch.dtype, name: str, reduction: str):
        z, _, _, teacher = _data(dtype)
        function = TEACHER[name]

        def call(x: torch.Tensor, **options: Any) -> torch.Tensor:
            t = teacher.double() if x.dtype == torch.float64 else teacher
            return function(x, t.to(x.device), **options)

        _paths(z, call, name, reduction)


def _rows(upstream: torch.Tensor, reduction: str, n: int) -> torch.Tensor:
    """The per-row upstream of a saved gradient, as autograd gives it after the reduction."""
    if reduction == "none":
        return upstream
    return upstream.expand(n) / (n if reduction == "mean" else 1)


def triton_check(n: int = 4, c: int = C, reductions: tuple[str, ...] = REDUCTIONS) -> None:
    """The Triton kernels' 16-bit gradients against the native float64 results, every
    reduction under a 2^-12 upstream (most float16 entries subnormal); run in the interpreter
    (float16) by HalfGradientKernelTests; call it with n = 4096 on CUDA for both dtypes."""
    from libsdca import (
        _triton_distillation,
        _triton_entropy,
        _triton_hinge,
        _triton_multilabel,
        _triton_prox,
        _triton_topk,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Few rows take a small upstream: their entries are as small as those of a mean of many.
    scale = SMALL_UPSTREAM if n < 1024 else 1.0
    gen = torch.Generator().manual_seed(0)
    z0 = 3 * torch.randn(n, c, generator=gen)
    labels = torch.randint(0, c, (n,), generator=gen)
    mask = torch.rand(n, c, generator=gen) < 0.01
    rows = torch.arange(n)
    mask[rows, labels], mask[rows, (labels + 1) % c] = True, False
    positive = torch.nn.functional.one_hot(labels, c).bool()
    teacher0 = z0 + torch.randn(n, c, generator=gen)
    worst: dict[str, float] = {}

    def reference(
        call: Callable[[torch.Tensor, str], torch.Tensor],
        z: torch.Tensor,
        reduction: str,
        upstream: torch.Tensor,
    ) -> torch.Tensor:
        with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": "off"}):
            x = z.double().cpu().requires_grad_()
            value = call(x, reduction)
            (gradient,) = torch.autograd.grad(value, x, upstream.double().cpu().expand_as(value))
        return gradient

    def check(
        name: str, z: torch.Tensor, gradient: torch.Tensor, ref: torch.Tensor, work: int
    ) -> None:
        what = f"{name} {z.dtype}"
        _check(z, gradient.cpu(), ref, work, what)
        g, r = gradient.double().cpu(), ref
        error = ((g - r).abs() / r.abs().amax(-1, keepdim=True)).max().item()
        worst[what] = max(worst.get(what, 0.0), error)

    # Triton's interpreter stores float64 values into bfloat16 as zeros: float16 only there.
    for dtype in HALF if device == "cuda" else (torch.float16,):
        z, teacher = z0.to(dtype).to(device), teacher0.to(dtype).to(device)
        y, m, p = labels.to(device), mask.to(device), positive.to(device)
        for reduction in reductions:
            # The upstream of a kernel call, float32 as its values.
            u = scale * (torch.linspace(0.5, 2.0, n) if reduction == "none" else torch.tensor(1.0))
            per_row = _rows(u, reduction, n)
            u = u.to(device)
            for kind, k, gamma in (
                ("topk_cross_entropy", 5, 0.0),
                ("topk_hinge_alpha", 5, 0.0),
                ("topk_hinge_beta", 5, 0.0),
                ("topk_hinge_alpha_smooth", 5, 0.5),
                ("topk_hinge_beta_smooth", 5, 0.5),
            ):
                name = kind.removesuffix("_smooth")
                ref = reference(
                    lambda x, r: sdca_loss(x, labels, name, k=k, gamma=gamma, reduction=r),
                    z,
                    reduction,
                    u,
                )
                for fp32 in (False, True) if kind == "topk_cross_entropy" else (False,):
                    _, stats = _triton_topk.forward(z, y, reduction, kind, k, gamma, fp32)
                    gradient = _triton_topk.backward(
                        z, y, stats, u, reduction, kind, k, gamma, fp32
                    )
                    check(f"triton top-k {kind} {reduction} fp32={fp32}", z, gradient, ref, fp32)
            for smooth in (False, True):
                gamma = 0.5 if smooth else 0.0
                ref = reference(
                    lambda x, r: sdca_loss(x, labels, "topk_hinge_alpha", gamma=gamma, reduction=r),
                    z,
                    reduction,
                    u,
                )
                _, stats = _triton_hinge.forward(z, y, reduction, smooth, gamma)
                gradient = _triton_hinge.backward(z, y, stats, u, reduction, smooth, gamma)
                check(f"triton hinge smooth={smooth} {reduction}", z, gradient, ref, 0)
            ref = reference(lambda x, r: sdca_loss(x, labels, reduction=r), z, reduction, u)
            for fp32 in (False, True):
                _, stats = _triton_entropy.forward(z, y, reduction, fp32)
                gradient = _triton_entropy.backward(z, y, stats, u, reduction, fp32)
                check(f"triton entropy {reduction} fp32={fp32}", z, gradient, ref, fp32)
            for objective, loss, gamma in (
                ("cross_entropy", "cross_entropy", 0.0),
                ("multilabel_hinge", "multilabel_hinge", 0.0),
                ("multilabel_hinge_smooth", "multilabel_hinge", 0.5),
            ):
                ref = reference(
                    lambda x, r: sdca_loss(x, mask, loss, gamma=gamma, reduction=r), z, reduction, u
                )
                for fp32 in (False, True) if objective == "cross_entropy" else (False,):
                    _, stats = _triton_multilabel.forward(z, m, objective, reduction, gamma, fp32)
                    gradient = _triton_multilabel.backward(
                        z, m, stats, u, objective, reduction, gamma, fp32
                    )
                    check(
                        f"triton {objective} mask {reduction} fp32={fp32}", z, gradient, ref, fp32
                    )
            for double in (False, True):
                ref = reference(
                    lambda x, r: distillation_loss(
                        x, teacher0.to(dtype).double(), 2.0, reduction=r
                    ),
                    z,
                    reduction,
                    u,
                )
                _, stats = _triton_distillation.forward(z, teacher, 2.0, double, reduction)
                gradient = _triton_distillation.backward(
                    z, teacher, stats, u, 2.0, double, reduction
                )
                check(
                    f"triton distillation {reduction} double={double}", z, gradient, ref, 1 - double
                )
            # The proximal kernels save float32 gradients; backward scales them, then rounds.
            ref = reference(
                lambda x, r: sdca_loss(x, labels, "proximal_cross_entropy", reduction=r),
                z,
                reduction,
                u,
            )
            for fp32 in (False, True):
                _, saved = _triton_prox.proximal_entropy(z, p, 1.0, fp32)
                assert saved.dtype == torch.float32
                gradient = _cpu.scaled(saved.cpu(), per_row, dtype, False)
                check(f"triton proximal_entropy {reduction} fp32={fp32}", z, gradient, ref, fp32)
            ref = reference(
                lambda x, r: proximal_distillation_loss(
                    x, teacher0.to(dtype).double(), 2.0, 0.7, reduction=r
                ),
                z,
                reduction,
                u,
            )
            for double in (False, True):
                _, saved = _triton_distillation.proximal_distillation(z, teacher, 2.0, 0.7, double)
                assert saved.dtype == torch.float32
                gradient = _cpu.scaled(saved.cpu(), per_row, dtype, False)
                check(
                    f"triton proximal_distillation {reduction} double={double}",
                    z,
                    gradient,
                    ref,
                    1 - double,
                )
    for what, error in sorted(worst.items()):
        print(f"{what}: {error:.3g} of the row's largest entry")


class HalfGradientKernelTests:
    @pytest.mark.slow
    @pytest.mark.parametrize("reduction", REDUCTIONS)
    def test_triton_kernels_in_the_interpreter(self, reduction: str):
        """The Triton kernels' control flow and rounding on CPU tensors (skips without
        Triton)."""
        call = f"t.triton_check(reductions=({reduction!r},))"
        interpreter.run(f"import test_half_gradient_rounding as t\n{call}\n")
