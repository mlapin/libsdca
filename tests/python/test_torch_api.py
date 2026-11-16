"""The public surface of libsdca.torch: one function and one plain module per loss, keyword-only
hyperparameters checked up front, one target kind per loss, the precision spellings, class
weights and ignored labels on the single-label losses, label smoothing and class probabilities
as torch.nn.functional.cross_entropy, torch.func transforms, explain_backend and
backend="kernel"."""

from __future__ import annotations

import inspect
import shutil
from collections.abc import Callable
from typing import Any

import libsdca.torch as L
import pytest
import torch
import torch.nn.functional as F
from libsdca import _cpu, _cpu_entropy

FUNCTIONS = {
    "cross_entropy": L.cross_entropy,
    "topk_cross_entropy": L.topk_cross_entropy,
    "multiclass_hinge_loss": L.multiclass_hinge_loss,
    "topk_hinge_loss": L.topk_hinge_loss,
    "proximal_cross_entropy": L.proximal_cross_entropy,
    "multilabel_cross_entropy": L.multilabel_cross_entropy,
    "multilabel_hinge_loss": L.multilabel_hinge_loss,
    "distillation_loss": L.distillation_loss,
    "proximal_distillation_loss": L.proximal_distillation_loss,
    "linear_cross_entropy": L.linear_cross_entropy,
}
MODULES = {
    "CrossEntropyLoss": L.CrossEntropyLoss,
    "TopKCrossEntropyLoss": L.TopKCrossEntropyLoss,
    "MulticlassHingeLoss": L.MulticlassHingeLoss,
    "TopKHingeLoss": L.TopKHingeLoss,
    "ProximalCrossEntropyLoss": L.ProximalCrossEntropyLoss,
    "MultilabelCrossEntropyLoss": L.MultilabelCrossEntropyLoss,
    "MultilabelHingeLoss": L.MultilabelHingeLoss,
    "DistillationLoss": L.DistillationLoss,
    "ProximalDistillationLoss": L.ProximalDistillationLoss,
    "LinearCrossEntropyLoss": L.LinearCrossEntropyLoss,
}
# The single-label losses (besides cross_entropy) with their fixed hyperparameters.
SINGLE: dict[str, tuple[Callable[..., torch.Tensor], dict[str, Any]]] = {
    "topk_cross_entropy": (L.topk_cross_entropy, {"k": 2}),
    "multiclass_hinge": (L.multiclass_hinge_loss, {"gamma": 0.5}),
    "topk_hinge_alpha": (L.topk_hinge_loss, {"k": 2, "variant": "alpha"}),
    "topk_hinge_beta": (L.topk_hinge_loss, {"k": 2, "gamma": 0.3}),
    "proximal_cross_entropy": (L.proximal_cross_entropy, {"alpha": 0.7}),
}


def problem(
    n: int = 7, c: int = 5, dtype: torch.dtype = torch.float64, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Logits, labels and a mask with a positive and a negative label in every row."""
    g = torch.Generator().manual_seed(seed)
    z = (2 * torch.randn(n, c, generator=g)).to(dtype)
    y = torch.randint(0, c, (n,), generator=g)
    mask = F.one_hot(y, c).bool() | (torch.rand(n, c, generator=g) < 0.3)
    mask[torch.arange(n), (y + 1) % c] = False
    return z, y, mask


class SurfaceTests:
    def test_all_names(self):
        assert set(L.__all__) == {*FUNCTIONS, *MODULES, "explain_backend"}
        for removed in (
            "sdca_loss",
            "SDCALoss",
            "topk_hinge_alpha_loss",
            "topk_hinge_beta_loss",
            "TopKHingeAlphaLoss",
            "TopKHingeBetaLoss",
        ):
            assert not hasattr(L, removed), removed

    def test_hyperparameters_are_keyword_only(self):
        positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        # torch.nn's signatures: cross_entropy(input, target, weight), linear's bias.
        allowed = {
            "cross_entropy": {"input", "target", "weight"},
            "linear_cross_entropy": {"hidden", "weight", "target", "bias"},
            "distillation_loss": {"student", "teacher"},
            "proximal_distillation_loss": {"student", "teacher"},
        }
        for name, function in FUNCTIONS.items():
            parameters = inspect.signature(function).parameters.values()
            names = {p.name for p in parameters if p.kind in positional}
            assert names == allowed.get(name, {"logits", "target"}), name
        modules = {"CrossEntropyLoss": {"weight"}, "LinearCrossEntropyLoss": {"class_weight"}}
        for name, module in MODULES.items():
            parameters = list(inspect.signature(module).parameters.values())
            names = {p.name for p in parameters if p.kind in positional}
            assert names == modules.get(name, set()), name

    def test_k_is_required(self):
        assert inspect.signature(L.topk_cross_entropy).parameters["k"].default is inspect._empty
        assert inspect.signature(L.TopKHingeLoss).parameters["k"].default is inspect._empty
        z, y, _ = problem()
        with pytest.raises(TypeError):
            L.topk_hinge_loss(z, y)  # pyright: ignore[reportCallIssue]

    def test_modules_are_plain_and_hold_their_options(self):
        expected = {
            L.TopKCrossEntropyLoss(k=2): "k=2, ignore_index=None, reduction='mean', dim=-1, "
            "backend='auto', precision=None",
            L.TopKHingeLoss(k=3, gamma=0.5): "k=3, variant='beta', gamma=0.5, ignore_index=None, "
            "reduction='mean', dim=-1, backend='auto', precision=None",
            L.MultilabelHingeLoss(): "gamma=0.0, reduction='mean', dim=-1, backend='auto', "
            "precision=None",
            L.DistillationLoss(temperature=2.0): "temperature=2.0, reduction='mean', dim=-1, "
            "backend='auto', precision=None",
        }
        for module, options in expected.items():
            assert repr(module) == f"{type(module).__name__}({options})"
            assert type(module).__setattr__ is torch.nn.Module.__setattr__
        weighted = L.ProximalCrossEntropyLoss(weight=torch.ones(5))
        assert "weight" in weighted.state_dict()
        assert weighted.to(torch.float64).weight.dtype == torch.float64  # pyright: ignore[reportOptionalMemberAccess]

    def test_constructors_and_calls_check_options(self, subtests: pytest.Subtests):
        z, y, _ = problem()
        cases: list[tuple[Callable[[], Any], type[Exception], str]] = [
            (lambda: L.TopKHingeLoss(k=0), ValueError, "k must be"),
            (lambda: L.TopKHingeLoss(k=2.5), TypeError, "k must be an integer"),  # pyright: ignore[reportArgumentType]
            (lambda: L.TopKHingeLoss(k=2, gamma=-1.0), ValueError, "gamma"),
            (lambda: L.TopKHingeLoss(k=2, gamma="1"), TypeError, "gamma"),  # pyright: ignore[reportArgumentType]
            (lambda: L.TopKHingeLoss(k=2, variant="gamma"), ValueError, "variant"),  # pyright: ignore[reportArgumentType]
            (lambda: L.ProximalCrossEntropyLoss(alpha=0.0), ValueError, "alpha"),
            (lambda: L.DistillationLoss(temperature=float("inf")), ValueError, "temperature"),
            (lambda: L.MulticlassHingeLoss(reduction="avg"), ValueError, "reduction"),
            (lambda: L.MulticlassHingeLoss(backend="cuda"), ValueError, "backend"),
            (lambda: L.MulticlassHingeLoss(dim=None), TypeError, "dim"),  # pyright: ignore[reportArgumentType]
            (lambda: L.MulticlassHingeLoss(ignore_index="x"), TypeError, "ignore_index"),  # pyright: ignore[reportArgumentType]
            (lambda: L.MulticlassHingeLoss(weight=[1.0]), TypeError, "weight"),  # pyright: ignore[reportArgumentType]
            (lambda: L.CrossEntropyLoss(label_smoothing=2.0), ValueError, "label_smoothing"),
            (lambda: L.multiclass_hinge_loss(z, y, gamma=float("nan")), ValueError, "gamma"),
            (lambda: L.topk_cross_entropy(z, y, k=5), ValueError, "k must be"),
            (lambda: L.multiclass_hinge_loss(z.tolist(), y), TypeError, "logits"),  # pyright: ignore[reportArgumentType]
            (lambda: L.multiclass_hinge_loss(z, y.tolist()), TypeError, "target"),  # pyright: ignore[reportArgumentType]
            (lambda: L.multiclass_hinge_loss(z.long(), y), TypeError, "floating-point"),
            (lambda: L.multiclass_hinge_loss(z, y, dim=2), ValueError, "out of range"),
            (lambda: L.multiclass_hinge_loss(z, y, weight=torch.ones(4)), ValueError, "shape"),
            (
                lambda: L.multiclass_hinge_loss(z, y, weight=torch.ones(5).long()),
                TypeError,
                "weight",
            ),
            (lambda: L.distillation_loss(z, z[:2]), ValueError, "shape"),
            (lambda: L.distillation_loss(z, z.long()), TypeError, "floating-point"),
        ]
        for i, (call, error, match) in enumerate(cases):
            with subtests.test(case=i), pytest.raises(error, match=match):
                call()

    def test_one_target_kind_per_loss(self):
        z, y, mask = problem()
        for name, (function, options) in SINGLE.items():
            for bad in (mask, y.float()):
                with pytest.raises(TypeError, match="integer class labels"):
                    function(z, bad, **options)
            # Any integer dtype is cast.
            expected = function(z, y, **options)
            for dtype in (torch.int32, torch.int16, torch.uint8, torch.int8):
                torch.testing.assert_close(function(z, y.to(dtype), **options), expected, msg=name)
        for function in (L.multilabel_cross_entropy, L.multilabel_hinge_loss):
            for bad in (y, mask.float(), mask.long()):
                with pytest.raises(TypeError, match="boolean mask"):
                    function(z, bad)
        with pytest.raises(TypeError, match="multilabel_cross_entropy"):
            L.cross_entropy(z, mask)
        torch.testing.assert_close(L.cross_entropy(z, y.int()), F.cross_entropy(z, y))

    def test_precision_spellings(self):
        z, y, _ = problem(dtype=torch.float32)
        reference = {
            None: L.topk_cross_entropy(z, y, k=2),
            "float64": L.topk_cross_entropy(z, y, k=2, precision="float64"),
        }
        for spelling, same in (
            ("float32", None),
            (torch.float32, None),
            (torch.float64, "float64"),
        ):
            value = L.topk_cross_entropy(z, y, k=2, precision=spelling)
            torch.testing.assert_close(value, reference[same], rtol=0, atol=0)
        for bad, error in (
            ("fp32", ValueError),
            ("double", ValueError),
            ("single", ValueError),
            (torch.float16, ValueError),
            (float, TypeError),
            (1.0, TypeError),
        ):
            for call in (
                lambda: L.topk_cross_entropy(z, y, k=2, precision=bad),
                lambda: L.cross_entropy(z, y, precision=bad),
                lambda: L.distillation_loss(z, z, precision=bad),
                lambda: L.MultilabelHingeLoss(precision=bad),
            ):
                with pytest.raises(error, match="precision"):
                    call()


class WeightTests:
    @pytest.mark.parametrize("name", SINGLE)
    @pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
    def test_weight_and_ignore_index_follow_cross_entropy(self, name: str, reduction: str):
        function, options = SINGLE[name]
        z, y, _ = problem(n=9, c=6)
        y[2] = y[5] = 4
        w = torch.linspace(0.25, 2.0, 6, dtype=torch.float64)
        x = z.clone().requires_grad_()
        value = function(x, y, weight=w, ignore_index=4, reduction=reduction, **options)
        # The reference: per-example losses of the kept rows, weighted by their labels.
        keep = y != 4
        ref_x = z.clone().requires_grad_()
        rows = function(ref_x[keep], y[keep], reduction="none", **options) * w[y[keep]]
        if reduction == "none":
            expected = torch.zeros(9, dtype=torch.float64).masked_scatter(keep, rows)
        elif reduction == "sum":
            expected = rows.sum()
        else:
            expected = rows.sum() / w[y[keep]].sum()
        torch.testing.assert_close(value, expected, rtol=1e-13, atol=1e-15)
        value.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(x.grad, ref_x.grad, rtol=1e-12, atol=1e-15)
        assert x.grad is not None and bool((x.grad[~keep] == 0).all())

    def test_ignored_rows_are_not_read_and_modules_hold_weights(self):
        z, y, _ = problem()
        y[0] = -1
        z[0, 1] = float("nan")
        value = L.topk_hinge_loss(z, y, k=2, ignore_index=-1, reduction="none")
        assert value[0].item() == 0 and bool(value.isfinite().all())
        with pytest.raises(ValueError, match="finite"):
            L.topk_hinge_loss(z, y.clamp(min=0), k=2)
        w = torch.rand(5, dtype=torch.float64)
        module = L.TopKHingeLoss(k=2, weight=w, ignore_index=-1)
        torch.testing.assert_close(
            module(z, y), L.topk_hinge_loss(z, y, k=2, weight=w, ignore_index=-1)
        )
        with pytest.raises(ValueError, match="finite and nonnegative"):
            L.multiclass_hinge_loss(z.nan_to_num(), y.clamp(min=0), weight=-w)

    def test_all_ignored_mean_is_nan(self):
        z, y, _ = problem()
        value = L.multiclass_hinge_loss(z, torch.full_like(y, 3), ignore_index=3)
        assert value.isnan()


class CrossEntropyFeatureTests:
    @pytest.mark.parametrize("shape", [(5,), (6, 5), (3, 5, 4, 2)])
    def test_label_smoothing_matches_torch(self, shape: tuple[int, ...], subtests: pytest.Subtests):
        g = torch.Generator().manual_seed(1)
        z = torch.randn(shape, dtype=torch.float64, generator=g)
        target_shape = () if len(shape) == 1 else shape[:1] + shape[2:]
        y = torch.randint(0, 5, target_shape, generator=g)
        if y.numel() > 1:
            y.view(-1)[1] = -100
        w = torch.rand(5, dtype=torch.float64, generator=g) + 0.1
        for weight in (None, w):
            for reduction in ("none", "sum", "mean"):
                for eps in (0.1, 1.0):
                    with subtests.test(weight=weight is not None, reduction=reduction, eps=eps):
                        options: dict[str, Any] = dict(reduction=reduction, label_smoothing=eps)
                        x, r = z.clone().requires_grad_(), z.clone().requires_grad_()
                        a = L.cross_entropy(x, y, weight, **options)
                        b = F.cross_entropy(r, y, weight, **options)
                        torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-14)
                        a.sum().backward()
                        b.sum().backward()
                        torch.testing.assert_close(x.grad, r.grad, rtol=1e-12, atol=1e-14)
                        module = L.CrossEntropyLoss(weight, **options)
                        torch.testing.assert_close(module(z, y), b.detach(), rtol=1e-13, atol=1e-14)

    @pytest.mark.parametrize("shape", [(5,), (6, 5), (3, 5, 4)])
    def test_probability_targets_match_torch(
        self, shape: tuple[int, ...], subtests: pytest.Subtests
    ):
        g = torch.Generator().manual_seed(2)
        z = torch.randn(shape, dtype=torch.float64, generator=g)
        dim = 0 if len(shape) == 1 else 1
        t = torch.softmax(torch.randn(shape, dtype=torch.float64, generator=g), dim)
        w = torch.rand(5, dtype=torch.float64, generator=g) + 0.1
        for weight in (None, w):
            for reduction in ("none", "sum", "mean"):
                for eps in (0.0, 0.1, 1.0):
                    with subtests.test(weight=weight is not None, reduction=reduction, eps=eps):
                        options: dict[str, Any] = dict(reduction=reduction, label_smoothing=eps)
                        x, r = z.clone().requires_grad_(), z.clone().requires_grad_()
                        a = L.cross_entropy(x, t, weight, **options)
                        b = F.cross_entropy(r, t, weight, **options)
                        torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-14)
                        a.sum().backward()
                        b.sum().backward()
                        torch.testing.assert_close(x.grad, r.grad, rtol=1e-12, atol=1e-14)

    def test_half_logits_return_float32(self):
        z, y, _ = problem(dtype=torch.bfloat16)
        t = torch.softmax(z.float(), 1)
        for value in (L.cross_entropy(z, y, label_smoothing=0.2), L.cross_entropy(z, t)):
            assert value.dtype == torch.float32
        torch.testing.assert_close(
            L.cross_entropy(z, y, label_smoothing=0.2),
            F.cross_entropy(z.float(), y, label_smoothing=0.2),
            rtol=1e-5,
            atol=1e-6,
        )

    @pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
    def test_linear_cross_entropy_matches_torch(self, reduction: str, subtests: pytest.Subtests):
        g = torch.Generator().manual_seed(3)
        h = torch.randn(3, 4, 6, dtype=torch.float64, generator=g)
        weight = torch.randn(7, 6, dtype=torch.float64, generator=g)
        bias = torch.randn(7, dtype=torch.float64, generator=g)
        y = torch.randint(0, 7, (3, 4), generator=g)
        y[0, 1] = -100
        t = torch.softmax(torch.randn(3, 4, 7, dtype=torch.float64, generator=g), -1)
        cw = torch.rand(7, dtype=torch.float64, generator=g) + 0.1
        for class_weight in (None, cw):
            for eps in (0.0, 0.1, 1.0):
                for target in (y, t):
                    with subtests.test(weight=class_weight is not None, eps=eps, soft=target is t):
                        leaves = [p.clone().requires_grad_() for p in (h, weight, bias)]
                        refs = [p.clone().requires_grad_() for p in (h, weight, bias)]
                        a = L.linear_cross_entropy(
                            leaves[0],
                            leaves[1],
                            target,
                            leaves[2],
                            class_weight=class_weight,
                            reduction=reduction,
                            label_smoothing=eps,
                        )
                        logits = F.linear(refs[0], refs[1], refs[2]).movedim(-1, 1)
                        b = F.cross_entropy(
                            logits,
                            target.movedim(-1, 1) if target is t else target,
                            class_weight,
                            reduction=reduction,
                            label_smoothing=eps,
                        )
                        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-13)
                        a.sum().backward()
                        b.sum().backward()
                        for leaf, ref in zip(leaves, refs, strict=True):
                            torch.testing.assert_close(leaf.grad, ref.grad, rtol=1e-11, atol=1e-13)
        module = L.LinearCrossEntropyLoss(cw, label_smoothing=0.1, reduction=reduction)
        torch.testing.assert_close(
            module(h, weight, y, bias),
            L.linear_cross_entropy(
                h, weight, y, bias, class_weight=cw, label_smoothing=0.1, reduction=reduction
            ),
        )


class TransformTests:
    def test_grad_vmap_jacrev_for_every_loss(self, subtests: pytest.Subtests):
        z, y, mask = problem()
        teacher = torch.randn_like(z)
        calls: dict[str, tuple[torch.Tensor, Callable[..., torch.Tensor]]] = {
            name: (y, lambda t, o, f=f, opts=opts: f(t, o, **opts))
            for name, (f, opts) in SINGLE.items()
        }
        calls["cross_entropy"] = (y, L.cross_entropy)
        calls["multilabel_cross_entropy"] = (mask, L.multilabel_cross_entropy)
        calls["multilabel_hinge"] = (mask, lambda t, o: L.multilabel_hinge_loss(t, o, gamma=0.2))
        calls["distillation"] = (teacher, lambda t, o: L.distillation_loss(t, o, temperature=2.0))
        calls["proximal_distillation"] = (teacher, L.proximal_distillation_loss)
        for name, (other, loss) in calls.items():
            with subtests.test(loss=name):
                x = z.clone().requires_grad_()
                (expected,) = torch.autograd.grad(loss(x, other), x)
                torch.testing.assert_close(torch.func.grad(lambda t: loss(t, other))(z), expected)
                per_sample = torch.func.vmap(torch.func.grad(lambda t, o: loss(t[None], o[None])))(
                    z, other
                )
                (summed,) = torch.autograd.grad(
                    torch.stack(
                        [loss(x[i : i + 1], other[i : i + 1]) for i in range(len(z))]
                    ).sum(),
                    x,
                )
                torch.testing.assert_close(per_sample, summed)
                jacobian = torch.func.jacrev(lambda t: loss(t, other))(z)
                torch.testing.assert_close(jacobian, expected)
                if name != "cross_entropy":
                    with pytest.raises(RuntimeError, match="first derivatives only"):
                        torch.func.hessian(lambda t: loss(t, other))(z)


class ExplainTests:
    def test_functions_and_modules_agree(self):
        z, y, mask = problem(dtype=torch.float32)
        pairs = [
            ((L.topk_hinge_loss, z, y), {"k": 2}, L.TopKHingeLoss(k=2)),
            ((L.multiclass_hinge_loss, z, y), {"gamma": 0.5}, L.MulticlassHingeLoss(gamma=0.5)),
            ((L.multilabel_cross_entropy, z, mask), {}, L.MultilabelCrossEntropyLoss()),
            ((L.cross_entropy, z, y), {}, L.CrossEntropyLoss()),
            ((L.distillation_loss, z, z), {}, L.DistillationLoss()),
        ]
        for (function, *args), options, module in pairs:
            explained = L.explain_backend(function, *args, **options)
            assert explained[0] in ("kernel", "native", "torch"), explained
            assert L.explain_backend(module, *args) == explained
        with pytest.raises(TypeError):
            L.explain_backend(F.cross_entropy, z, y)
        with pytest.raises(TypeError):
            L.explain_backend(torch.nn.CrossEntropyLoss(), z, y)

    def test_class_dimension_and_backends(self):
        g = torch.Generator().manual_seed(4)
        z = torch.randn(2, 5, 3, 4, generator=g)
        y = torch.randint(0, 5, (2, 3, 4), generator=g)
        assert L.explain_backend(L.topk_hinge_loss, z, y, k=2, dim=1)[0] in ("kernel", "native")
        assert L.explain_backend(L.cross_entropy, z, y)[0] in ("kernel", "native")
        for backend in ("native", "torch"):
            assert L.explain_backend(L.topk_hinge_loss, z, y, k=2, dim=1, backend=backend)[0] == (
                backend
            )
            assert L.explain_backend(L.cross_entropy, z, y, backend=backend)[0] == backend
        name, reason = L.explain_backend(L.topk_hinge_loss, z, y, k=2, dim=1, weight=torch.ones(5))
        assert "class weights" in reason
        name, reason = L.explain_backend(L.cross_entropy, z, y, label_smoothing=0.1)
        assert "label smoothing" in reason
        t = torch.softmax(z, 1)
        assert L.explain_backend(L.cross_entropy, z, t)[0] == "torch"
        with pytest.raises(ValueError, match="k must be"):
            L.explain_backend(L.topk_hinge_loss, z, y, k=5, dim=1)

    @pytest.mark.skipif(not _cpu_entropy.enabled(), reason="needs the CPU extension")
    def test_backend_kernel(self):
        z, y, mask = problem(dtype=torch.float32)
        cases: list[tuple[Callable[..., torch.Tensor], torch.Tensor, dict[str, Any]]] = [
            (L.topk_hinge_loss, y, {"k": 2}),
            (L.multiclass_hinge_loss, y, {}),
            (L.cross_entropy, y, {}),
            (L.multilabel_hinge_loss, mask, {}),
            (L.distillation_loss, z, {}),
        ]
        for function, target, options in cases:
            expected = function(z, target, **options)
            assert L.explain_backend(function, z, target, **options)[0] == "kernel"
            torch.testing.assert_close(function(z, target, backend="kernel", **options), expected)
        # No fused kernel: float64 cross-entropy rows wider than the extension's, tensor-only
        # paths (class probabilities) and the tensor backend's route under transforms.
        wide = torch.randn(2, _cpu.DOUBLE_CLASSES + 1, dtype=torch.float64)
        label = torch.tensor([0, 1])
        assert L.explain_backend(L.cross_entropy, wide, label)[0] == "native"
        with pytest.raises(NotImplementedError, match="backend='kernel'"):
            L.cross_entropy(wide, label, backend="kernel")
        with pytest.raises(NotImplementedError, match="backend='kernel'"):
            L.topk_cross_entropy(wide, label, k=1, backend="kernel")
        assert L.explain_backend(L.cross_entropy, z, torch.softmax(z, 1), backend="kernel")[0] == (
            "torch"
        )


class DistillationTests:
    def test_teacher_dtype_is_promoted(self):
        z, _, _ = problem(dtype=torch.float32)
        teacher = torch.randn_like(z)
        for low in (torch.bfloat16, torch.float16):
            t = teacher.to(low)
            x = z.clone().requires_grad_()
            value = L.distillation_loss(x, t, temperature=2.0)
            expected = L.distillation_loss(z, t.float(), temperature=2.0)
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            value.backward()
            assert x.grad is not None and x.grad.dtype == torch.float32
            # A bfloat16 student with a float32 teacher computes in float32.
            s = z.to(low).requires_grad_()
            value = L.proximal_distillation_loss(s, teacher, alpha=0.5)
            value.backward()
            assert s.grad is not None and s.grad.dtype == low

    def test_backends(self):
        z, _, _ = problem(dtype=torch.float64)
        teacher = torch.randn_like(z)
        for function in (L.distillation_loss, L.proximal_distillation_loss):
            values = {
                backend: function(z, teacher, temperature=1.5, backend=backend, reduction="none")
                for backend in ("auto", "native", "torch")
            }
            for backend, value in values.items():
                torch.testing.assert_close(value, values["native"], rtol=1e-12, atol=1e-14)
                assert L.explain_backend(function, z, teacher, backend=backend)[0] in (
                    ("kernel", "native") if backend == "auto" else (backend,)
                )


@pytest.mark.skipif(
    not (_cpu.AVAILABLE and (shutil.which("g++") or shutil.which("clang++"))),
    reason="needs torch 2.4 and a C++ compiler for Inductor",
)
class CompileTests:
    def test_modules_compile_in_one_graph(self, subtests: pytest.Subtests):
        z, y, mask = problem(n=12, c=7, dtype=torch.float32)
        w = torch.rand(7) + 0.5
        calls: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            "topk_hinge": lambda x: L.TopKHingeLoss(k=2, gamma=0.5)(x, y),
            "weighted_topk_ce": lambda x: L.TopKCrossEntropyLoss(k=2, weight=w, ignore_index=3)(
                x, y
            ),
            "multilabel": lambda x: L.MultilabelHingeLoss()(x, mask),
            "smoothed_ce": lambda x: L.CrossEntropyLoss(label_smoothing=0.1)(x, y),
            "soft_ce": lambda x: L.cross_entropy(x, torch.softmax(z, 1)),
        }
        for name, loss in calls.items():
            with subtests.test(loss=name):
                torch._dynamo.reset()
                compiled = torch.compile(loss, fullgraph=True)
                eager_z, compiled_z = z.clone().requires_grad_(), z.clone().requires_grad_()
                expected, actual = loss(eager_z), compiled(compiled_z)
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                expected.backward()
                actual.backward()
                torch.testing.assert_close(compiled_z.grad, eager_z.grad, rtol=1e-5, atol=1e-7)
