"""cross_entropy under torch.func transforms, against torch.nn.functional.cross_entropy."""

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from libsdca import CrossEntropyLoss
from libsdca.torch import cross_entropy

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
