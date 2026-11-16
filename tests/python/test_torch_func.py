"""cross_entropy under torch.func transforms, against torch.nn.functional.cross_entropy, and
the other losses (first order) against torch.autograd."""

from collections.abc import Callable

import libsdca.torch as L
import pytest
import torch
import torch.nn.functional as F
from libsdca.torch import CrossEntropyLoss, cross_entropy

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
Loss = Callable[..., torch.Tensor]


def inputs(device: str, dtype: torch.dtype = torch.float64):
    generator = torch.Generator().manual_seed(0)
    z = torch.randn(6, 5, generator=generator, dtype=dtype).to(device)
    y = torch.tensor([0, 1, 2, -100, 4, 3], device=device)
    w = torch.linspace(0.5, 2, 5, dtype=dtype, device=device)
    return z, y, w


@pytest.mark.parametrize("device", DEVICES)
class TorchFuncTests:
    def test_grad(self, device: str, subtests: pytest.Subtests):
        for dtype in (torch.float32, torch.float64):
            z, y, w = inputs(device, dtype)
            for weight in (None, w):
                for reduction in ("mean", "sum"):
                    with subtests.test(dtype=dtype, weight=weight is not None, reduction=reduction):

                        def grad(f: Loss):
                            return torch.func.grad(lambda t: f(t, y, weight, reduction=reduction))

                        torch.testing.assert_close(grad(cross_entropy)(z), grad(F.cross_entropy)(z))

    def test_per_sample_gradients(self, device: str):
        z, y, w = inputs(device)

        def per_sample(f: Loss):
            def loss(t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
                return f(t[None], label[None], w)

            return torch.func.vmap(torch.func.grad(loss))(z, y)

        actual, expected = per_sample(cross_entropy), per_sample(F.cross_entropy)
        torch.testing.assert_close(actual, expected, equal_nan=True)

    def test_per_sample_model_gradients(self, device: str):
        model = torch.nn.Linear(4, 3).double().to(device)
        params = {name: p.detach() for name, p in model.named_parameters()}
        x = torch.randn(8, 4, dtype=torch.float64, device=device)
        y = torch.randint(3, (8,), device=device)

        def per_sample(criterion: torch.nn.Module):
            def loss(
                p: dict[str, torch.Tensor], xi: torch.Tensor, yi: torch.Tensor
            ) -> torch.Tensor:
                return criterion(torch.func.functional_call(model, p, (xi[None],)), yi[None])

            return torch.func.vmap(torch.func.grad(loss), in_dims=(None, 0, 0))(params, x, y)

        actual, expected = per_sample(CrossEntropyLoss()), per_sample(torch.nn.CrossEntropyLoss())
        for name in params:
            torch.testing.assert_close(actual[name], expected[name])

    def test_vmap_jacrev_and_hessian(self, device: str, subtests: pytest.Subtests):
        z, y, w = inputs(device)
        with subtests.test("vmap"):
            zs, ys = torch.stack([z, 2 * z, -z]), torch.stack([y, y.roll(1), y.roll(2)])

            def batched(f: Loss):
                return torch.func.vmap(lambda t, label: f(t, label, w))(zs, ys)

            torch.testing.assert_close(batched(cross_entropy), batched(F.cross_entropy))
        with subtests.test("jacrev"):

            def jacobian(f: Loss):
                return torch.func.jacrev(lambda t: f(t, y, w, reduction="none"))(z)

            torch.testing.assert_close(jacobian(cross_entropy), jacobian(F.cross_entropy))
        with subtests.test("jvp"):
            tangent = torch.randn_like(z)
            for reduction in ("none", "sum", "mean"):

                def push(f: Loss):
                    return torch.func.jvp(
                        lambda t: f(t, y, w, reduction=reduction), (z,), (tangent,)
                    )

                torch.testing.assert_close(push(cross_entropy), push(F.cross_entropy))
        with subtests.test("hessian"):

            def hessian(f: Loss):
                return torch.func.hessian(lambda t: f(t, y, w))(z)

            torch.testing.assert_close(hessian(cross_entropy), hessian(F.cross_entropy))

    def test_invalid_rows_give_nan(self, device: str):
        z, y, w = inputs(device)
        y = y.clone()
        y[0] = 5
        gradient = torch.func.grad(lambda t: cross_entropy(t, y, w, reduction="sum"))(z)
        assert bool(gradient.isnan().any())

    def test_native_backend_raises(self, device: str):
        z, y, _ = inputs(device)
        with pytest.raises(ValueError, match="torch.func"):
            torch.func.grad(lambda t: cross_entropy(t, y, backend="native"))(z)


def _first_order_losses(
    z: torch.Tensor, y: torch.Tensor
) -> dict[str, tuple[torch.Tensor, Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor]]]:
    """Every first-order loss of logits like z: its target (labels y, a mask or teacher logits
    derived from them) and the call loss(logits, target, reduction)."""
    classes = z.shape[-1]
    mask = F.one_hot(y, classes).bool() | F.one_hot((y + 2) % classes, classes).bool()
    teacher = 3 * F.one_hot(y, classes).to(z.dtype)
    weight = torch.linspace(0.5, 2, classes, dtype=z.dtype)
    return {
        "topk_cross_entropy": (y, lambda t, o, r: L.topk_cross_entropy(t, o, k=2, reduction=r)),
        "multiclass_hinge": (
            y,
            lambda t, o, r: L.multiclass_hinge_loss(t, o, gamma=0.5, reduction=r),
        ),
        "topk_hinge_alpha": (
            y,
            lambda t, o, r: L.topk_hinge_loss(t, o, k=2, variant="alpha", reduction=r),
        ),
        "topk_hinge_beta": (y, lambda t, o, r: L.TopKHingeLoss(k=1, reduction=r)(t, o)),
        "proximal_cross_entropy": (y, lambda t, o, r: L.proximal_cross_entropy(t, o, reduction=r)),
        "weighted_hinge": (
            y,
            lambda t, o, r: L.topk_hinge_loss(t, o, k=2, weight=weight, reduction=r),
        ),
        "multilabel_cross_entropy": (
            mask,
            lambda t, m, r: L.multilabel_cross_entropy(t, m, reduction=r),
        ),
        "multilabel_hinge": (mask, lambda t, m, r: L.multilabel_hinge_loss(t, m, reduction=r)),
        "distillation_loss": (
            teacher,
            lambda t, s, r: L.distillation_loss(t, s, temperature=2.0, reduction=r),
        ),
        "proximal_distillation_loss": (
            teacher,
            lambda t, s, r: L.proximal_distillation_loss(t, s, reduction=r),
        ),
    }


def test_other_losses_support_transforms(subtests: pytest.Subtests):
    """grad, vmap (per-sample gradients, batched rows) and jacrev agree with torch.autograd;
    the losses are first order, so a hessian raises."""
    z, y, _ = inputs("cpu")
    y = y.clamp(min=0)
    for name, (target, loss) in _first_order_losses(z, y).items():
        with subtests.test(loss=name):
            x = z.clone().requires_grad_()
            for reduction in ("mean", "sum"):
                (expected,) = torch.autograd.grad(loss(x, target, reduction), x)
                actual = torch.func.grad(lambda t: loss(t, target, reduction))(z)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-15)
            rows = loss(z, target, "none")
            per_sample = torch.func.vmap(
                torch.func.grad(lambda t, other: loss(t[None], other[None], "sum"))
            )(z, target)
            (dense,) = torch.autograd.grad(loss(x, target, "sum"), x)
            torch.testing.assert_close(per_sample, dense, rtol=1e-12, atol=1e-15)
            other = target.roll(1, 0)
            batched = torch.func.vmap(lambda t, o: loss(t, o, "none"))(
                torch.stack([z, 2 * z]), torch.stack([target, other])
            )
            torch.testing.assert_close(batched[0], rows, rtol=0, atol=0)
            torch.testing.assert_close(batched[1], loss(2 * z, other, "none"), rtol=0, atol=0)
            jacobian = torch.func.jacrev(lambda t: loss(t, target, "none"))(z)
            for i in range(z.shape[0]):
                (row,) = torch.autograd.grad(loss(x, target, "none")[i], x)
                torch.testing.assert_close(jacobian[i], row, rtol=1e-12, atol=1e-15)
            with pytest.raises(RuntimeError, match="first derivatives only"):
                torch.func.hessian(lambda t: loss(t, target, "sum"))(z)


def test_transformed_cpu_losses_validate_labels():
    """Under transforms the CPU kernels validate as in eager mode (ValueError)."""
    z, y, _ = inputs("cpu")
    y = y.clamp(min=0)
    y[0] = 7
    with pytest.raises(ValueError, match="out of range"):
        torch.func.grad(lambda t: L.topk_cross_entropy(t, y, k=2))(z)


@pytest.mark.parametrize("device", DEVICES)
def test_transformed_power_normal_boundaries(device: str):
    from libsdca._entropy import _power

    exponents = torch.arange(-1022, 1024, dtype=torch.int64, device=device)
    expected = _power(exponents)
    actual = torch.func.vmap(_power)(exponents)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all() and (actual > 0).all()
