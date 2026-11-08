"""The tensor losses against the native ones, on the CPU and on CUDA when available."""

import os
from fractions import Fraction
from typing import Any
from unittest import mock

import mpmath as mp
import pytest
import torch
from libsdca import LOSSES, _cpu_entropy
from libsdca.torch import SDCALoss, sdca_loss
from topk_rows import CAPPED_GAPS, capped_reference, capped_rows
from torch.autograd import gradcheck

# test_tensor_prox covers proximal_entropy.
TENSOR = [objective for objective in LOSSES if objective != "proximal_cross_entropy"]
SMOOTH = ["multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge"]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def options_of(objective: str, classes: int) -> list[dict[str, Any]]:
    """The k and gamma values that matter for an objective."""
    ks = sorted({1, 2, classes // 2, classes - 1} & set(range(1, classes)))
    gammas = (0.0, 0.6, 1e-3, 30.0)
    if objective in ("cross_entropy", "multilabel_cross_entropy"):
        return [{}]
    if objective in ("multiclass_hinge", "multilabel_hinge"):
        return [dict(gamma=g) for g in gammas]
    if objective == "topk_cross_entropy":
        return [dict(k=k) for k in ks]
    return [dict(k=k, gamma=g) for k in ks for g in gammas]


def target_of(
    objective: str, examples: int, classes: int, generator: torch.Generator
) -> torch.Tensor:
    """Labels, or masks with a positive and a negative label for the multilabel hinges."""
    if not objective.startswith("multilabel"):
        return torch.randint(0, classes, (examples,), generator=generator)
    mask = torch.rand(examples, classes, generator=generator) < 0.4
    rows = torch.arange(examples)
    first = torch.randint(0, classes, (examples,), generator=generator)
    mask[rows, first] = True
    mask[rows, (first + 1) % classes] = False
    return mask


# {z0, z1, gamma}: margins 1 + z1 - z0 that cancel, h a overflowing or
# underflowing at huge or tiny gamma, and huge common offsets.
HINGE_CASES = [
    (1.0, 1e-20, 1e-20),
    (1.0, 2.0**-26, 1.0),
    (0.0, 1e200, 1e300),
    (1.0, 1e-200, 1e-300),
    (1e16, 1e16 + 2, 1.0),
    (1e16, 1e16 - 2, 1.0),
    (-1e16, -1e16 + 2, 1e16),
    (0.0, 0.0, 1e-310),
    (0.0, 0.5, 5e-324),
    (1.0, 0.0, 1e-310),
]


def two_class_hinge(z0: float, z1: float, objective: str, gamma: float):
    """Exact value and slope v at h = 1 + z1 - z0: max(0, h), or max h v - width v^2/2 on [0, 1]."""
    h = 1 + Fraction(z1) - Fraction(z0)
    if gamma == 0:
        return float(max(h, 0)), float(h > 0)
    width = Fraction(gamma) * (2 if objective.startswith("multilabel") else 1)
    v = min(Fraction(1), max(Fraction(0), h / width))
    return float(h * v - width * v * v / 2), float(v)


class TensorLossTests:
    def check(
        self,
        z: torch.Tensor,
        target: torch.Tensor,
        objective: str,
        rtol: float,
        atol: float,
        **options: Any,
    ):
        """Values and weighted gradients of the tensor backend against the native backend."""
        reference = z.detach().clone().requires_grad_()
        weights = torch.linspace(0.5, 2.0, z.shape[0], dtype=z.dtype)
        expected = sdca_loss(
            reference, target, objective, backend="native", reduction="none", **options
        )
        expected.backward(weights)
        for device in DEVICES:
            x = z.detach().to(device).requires_grad_()
            actual = sdca_loss(
                x, target.to(device), objective, backend="torch", reduction="none", **options
            )
            actual.backward(weights.to(device))
            message = f"{objective} {options} {device} {z.dtype}"
            assert x.grad is not None
            torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol, msg=message)
            torch.testing.assert_close(
                x.grad.cpu(), reference.grad, rtol=rtol, atol=atol, msg=message
            )

    def test_random_scores_match_native(self):
        generator = torch.Generator().manual_seed(31)
        for dtype, rtol in [(torch.float64, 1e-12), (torch.float32, 3e-7)]:
            for classes in [2, 3, 7, 40]:
                for scale in [1.0, 100.0]:
                    z = (scale * torch.randn(48, classes, generator=generator)).to(dtype)
                    for objective in TENSOR:
                        target = target_of(objective, 48, classes, generator)
                        for options in options_of(objective, classes):
                            atol = rtol * scale / (options.get("gamma") or 1.0)
                            self.check(z, target, objective, rtol, atol, **options)

    def test_ties_select_the_native_subgradients(self):
        # Integer scores tie often; kinks and degenerate breakpoints must agree.
        generator = torch.Generator().manual_seed(4)
        for classes in [3, 6]:
            z = torch.randint(-2, 3, (200, classes), generator=generator).double()
            for objective in TENSOR:
                target = target_of(objective, 200, classes, generator)
                for options in options_of(objective, classes):
                    rows = slice(None)
                    if objective == "topk_hinge_alpha":
                        # A mean margin of zero up to rounding admits either subgradient.
                        value = sdca_loss(z, target, objective, reduction="none", **options)
                        rows = value.abs() > 1e-12
                    self.check(z[rows], target[rows], objective, 1e-12, 1e-12, **options)

    def test_far_offsets_and_cancelling_scores(self):
        rows = [[1e16] * 3, [1e16, 1e16, 1e16 + 2], [700.0, 0.0, 1.0, 700.0, -0.3, 0.2]]
        for gap in [700.0, 1e16, 1e30]:
            rows += [[-gap, 0.0, 1.0, 2.0, -1.0], [gap, 0.0, 1.0, 2.0, -1.0]]
            rows.append([0.0, 1.0, gap, -0.3, 0.2])
        rows += [
            [0.0, 20.0, 20.0, -100.0],
            [0.0, 10.0, 10.0, 10.0, -100.0],
            [1000.0, 0.0, 0.0, -5.0],
        ]
        # Hinge values of cancelling scores are ill-conditioned; top-k entropy keeps the mean.
        cancelling = [[0.0, 1e16, 1.0, -1e16], [0.0, 1e30, 1.0, -1e30], [0.0, 1e12, 0.0, -1e12]]
        for row in rows + cancelling:
            z = torch.tensor([row], dtype=torch.float64)
            classes = z.shape[-1]
            objectives = TENSOR if row in rows else ["cross_entropy", "topk_cross_entropy"]
            if max(row) >= 1e16 and row[0] < max(row) and row[1] == 1.0:
                # The native top-k entropy loses these free scores; see the test below.
                objectives = [o for o in objectives if o != "topk_cross_entropy"]
            for objective in objectives:
                target = torch.tensor([0])
                if objective.startswith("multilabel"):
                    target = torch.zeros(1, classes, dtype=torch.bool)
                    target[0, ::2] = True
                settings = options_of(objective, classes)
                if objective in SMOOTH:
                    settings += [dict(s, gamma=g) for s in settings[::3] for g in (1e-20, 1e20)]
                for options in settings:
                    self.check(z, target, objective, 1e-12, 1e-12, **options)

    def test_hinge_margins_and_smoothing_against_two_class_closed_forms(self):
        hinges = [o for o in TENSOR if "hinge" in o]
        for dtype in [torch.float32, torch.float64]:
            for z0, z1, smoothing in HINGE_CASES:
                z = torch.tensor([[z0, z1]], dtype=dtype)
                if not z.isfinite().all():
                    continue
                for objective, gamma in [(o, g) for o in hinges for g in (0.0, smoothing)]:
                    value, slope = two_class_hinge(z[0, 0].item(), z[0, 1].item(), objective, gamma)
                    expected = torch.tensor(value, dtype=dtype)
                    # A mask takes the tensor path where indices take a fused kernel.
                    for target in [torch.tensor([0]), torch.tensor([[True, False]])]:
                        for backend, device in [("native", "cpu")] + [
                            ("torch", d) for d in DEVICES
                        ]:
                            x = z.to(device).requires_grad_()
                            v = sdca_loss(
                                x, target.to(device), objective, backend=backend, gamma=gamma
                            )
                            (g,) = torch.autograd.grad(v, x)
                            message = (
                                f"{objective} {backend} {device} {z.tolist()} {gamma} {target}"
                            )
                            # Errors relative to the value and to the slope (order one at the kink).
                            torch.testing.assert_close(
                                v.cpu(),
                                expected,
                                rtol=2 * torch.finfo(dtype).eps,
                                atol=0,
                                msg=message,
                            )
                            torch.testing.assert_close(
                                g.cpu()[0],
                                torch.tensor([-slope, slope], dtype=dtype),
                                rtol=2 * torch.finfo(dtype).eps,
                                atol=0,
                                msg=message,
                            )

    def test_topk_entropy_free_scores_below_a_far_competitor(self):
        # The label is far below and one competitor far above the others: the
        # gradient does not depend on the gap. The native loss keeps it to 700, up
        # to the rounding of its scores there.
        target = torch.tensor([0])
        for k in [2, 3]:
            near = torch.tensor([[-700.0, 1.0, 700.0, -0.3, 0.2]], dtype=torch.float64)
            near.requires_grad_()
            sdca_loss(near, target, "topk_cross_entropy", k=k, backend="native").backward()
            for gap in [1e16, 1e30]:
                z = torch.tensor([[-gap, 1.0, gap, -0.3, 0.2]], dtype=torch.float64)
                for device in DEVICES:
                    x = z.detach().to(device).requires_grad_()
                    sdca_loss(
                        x, target.to(device), "topk_cross_entropy", k=k, backend="torch"
                    ).backward()
                    assert x.grad is not None
                    torch.testing.assert_close(x.grad.cpu(), near.grad, rtol=3e-13, atol=0)

    def test_topk_entropy_confident_rows_within_64_eps(self):
        # Rows [0, -m, ..., -m] of label 0 cap no competitor: cross_entropy's loss
        # log1p((C - 1) e^-m) and entries, normal to m = 700. Log-domain odds rounded at the
        # magnitude of m: 154 eps at m = 261.5.
        eps = torch.finfo(torch.float64).eps
        gaps = [40.0 + 13.25 * i for i in range(50)]
        for classes, k in [(40, 2), (1000, 5)]:
            z = torch.tensor(gaps, dtype=torch.float64)[:, None].expand(-1, classes).neg()
            z = z.contiguous()
            z[:, 0] = 0
            for device in DEVICES:
                x = z.to(device).requires_grad_()
                target = torch.zeros(len(gaps), dtype=torch.long, device=device)
                value = sdca_loss(
                    x, target, "topk_cross_entropy", k=k, backend="torch", reduction="none"
                )
                (gradient,) = torch.autograd.grad(value.sum(), x)
                for i, gap in enumerate(gaps):
                    with mp.workdps(60):
                        e = mp.exp(-mp.mpf(gap))
                        tail = (classes - 1) * e
                        expected = (mp.log1p(tail), -tail / (1 + tail), e / (1 + tail))
                    actual = (value[i], gradient[i, 0], gradient[i, -1])
                    names = ("loss", "target", "other")
                    for name, got, ref in zip(names, actual, expected, strict=True):
                        error = abs(mp.mpf(got.item()) - ref) / abs(ref)
                        assert error <= 64 * eps, (name, classes, device, gap, error / eps)

    def test_topk_entropy_capped_confident_rows_within_64_eps(self, subtests: pytest.Subtests):
        # Log-domain odds and shifts put every CPU path up to 339 eps off at m = 649.5.
        eps = torch.finfo(torch.float64).eps
        paths = [("torch", device, "off") for device in DEVICES] + [("native", "cpu", "off")]
        if _cpu_entropy.buildable():
            paths.append(("auto", "cpu", "on"))
        for classes, k in [(40, 2), (1000, 5)]:
            z = torch.tensor(capped_rows(classes))
            expected = [capped_reference(gap, classes, k) for gap in CAPPED_GAPS]
            for backend, device, extension in paths:
                x = z.to(device).requires_grad_()
                target = torch.zeros(len(z), dtype=torch.long, device=device)
                with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": extension}):
                    value = sdca_loss(
                        x, target, "topk_cross_entropy", k=k, backend=backend, reduction="none"
                    )
                    (gradient,) = torch.autograd.grad(value.sum(), x)
                for i, gap in enumerate(CAPPED_GAPS):
                    actual = (value[i], gradient[i, 0], gradient[i, 1], gradient[i, -1])
                    names = ("loss", "target", "capped", "free")
                    with subtests.test(backend=backend, device=device, classes=classes, gap=gap):
                        for name, got, ref in zip(names, actual, expected[i], strict=True):
                            error = abs(mp.mpf(got.item()) - ref) / abs(ref)
                            assert error <= 64 * eps, (name, error / eps)

    def test_low_precision_inputs(self):
        generator = torch.Generator().manual_seed(2)
        z = 3 * torch.randn(16, 9, generator=generator)
        for dtype, tol in [(torch.float16, 4e-3), (torch.bfloat16, 3e-2)]:
            for objective in TENSOR + ["proximal_cross_entropy"]:
                target = target_of(objective, 16, 9, generator)
                for device in DEVICES:
                    x = z.to(device, dtype).requires_grad_()
                    reference = x.detach().float().cpu().requires_grad_()
                    values = sdca_loss(
                        x, target.to(device), objective, k=3, gamma=0.5, reduction="none"
                    )
                    expected = sdca_loss(
                        reference, target, objective, k=3, gamma=0.5, reduction="none"
                    )
                    values.sum().backward()
                    expected.sum().backward()
                    assert x.grad is not None
                    assert x.grad.dtype == dtype
                    torch.testing.assert_close(values.float().cpu(), expected, rtol=tol, atol=tol)
                    torch.testing.assert_close(
                        x.grad.float().cpu(), reference.grad, rtol=tol, atol=tol
                    )

    def test_layout_backends_and_validation(self):
        z = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[0, 4, 2], [1, 1, 3]])
        for device in DEVICES:
            x = z.detach().to(device).requires_grad_()
            for objective in TENSOR:
                options: dict[str, Any] = dict(k=2, gamma=0.7, dim=1)
                target = labels
                if objective.startswith("multilabel"):
                    target = torch.nn.functional.one_hot(labels, 5).movedim(-1, 1).bool()
                expected = SDCALoss(objective, backend="native", **options)(z, target)
                actual = SDCALoss(objective, **options)(x, target.to(device))
                torch.testing.assert_close(actual.cpu(), expected, rtol=1e-12, atol=1e-14)
                if objective in SMOOTH:
                    assert gradcheck(
                        lambda v: sdca_loss(
                            v, target.to(device), objective, backend="torch", **options
                        ),
                        (x,),
                    )
            mask = torch.tensor([[True, True, False], [False, True, False]], device=device)
            zeros = torch.zeros(2, 3, device=device)
            for objective, target in [
                ("topk_hinge_alpha", mask),
                ("multilabel_hinge", torch.ones_like(mask)),
                ("topk_hinge_alpha", torch.tensor([0, 3], device=device)),
            ]:
                if device == "cpu":
                    with pytest.raises(ValueError):
                        sdca_loss(zeros, target, objective, backend="torch")
                else:
                    assert bool(sdca_loss(zeros, target, objective, backend="torch").isnan().all())
            g = torch.autograd.grad(
                sdca_loss(x, labels.to(device), "topk_hinge_beta", dim=1), x, create_graph=True
            )[0]
            with pytest.raises(RuntimeError):
                torch.autograd.grad(g.sum(), x)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_auto_backend_runs_on_cuda(self):
        z = torch.randn(8, 6, device="cuda", requires_grad=True)
        labels = torch.randint(0, 6, (8,), device="cuda")
        for objective in ["cross_entropy", "topk_hinge_alpha"]:
            torch.testing.assert_close(
                sdca_loss(z, labels, objective, k=2),
                sdca_loss(z, labels, objective, k=2, backend="torch"),
            )
        with pytest.raises(ValueError):
            sdca_loss(z, labels, "cross_entropy", backend="native")
