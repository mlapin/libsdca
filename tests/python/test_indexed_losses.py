"""Integer-target fast paths at k = 1 against the boolean-mask (generic) path of the same loss."""

from typing import Any

import pytest
import torch
from libsdca.torch import sdca_loss

# (loss, gamma): each hinge nonsmooth and smoothed.
K1 = (
    ("topk_cross_entropy", 0.0),
    ("topk_hinge_alpha", 0.0),
    ("topk_hinge_beta", 0.0),
    ("topk_hinge_alpha", 0.5),
    ("topk_hinge_beta", 0.5),
)


def _case(loss: str | tuple[str, float], gamma: float | None = None) -> tuple[str, float]:
    """(name, gamma) of a K1 case or a name; a given gamma replaces a smoothed case's."""
    if isinstance(loss, str):
        return loss, 0.0 if gamma is None else gamma
    name, smooth = loss
    return name, 0.0 if smooth == 0 else (smooth if gamma is None else gamma)


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _pair(
    z: torch.Tensor,
    y: torch.Tensor,
    loss: str | tuple[str, float],
    reduction: str = "none",
    gamma: float | None = None,
    upstream: torch.Tensor | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Values and gradients from int64 targets and from the one-hot mask."""
    loss, gamma = _case(loss, gamma)
    out = []
    for target in (y, torch.nn.functional.one_hot(y, z.shape[1]).bool()):
        x = z.clone().requires_grad_()
        value = sdca_loss(x, target, loss, k=1, gamma=gamma, reduction=reduction)
        value.backward(torch.ones_like(value) if upstream is None else upstream)
        assert x.grad is not None
        out.append((value.detach(), x.grad))
    return out


class IndexedLossTests:
    def assert_pair(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        loss: str | tuple[str, float],
        rtol: float = 1e-13,
        atol: float = 0.0,
        **options: Any,
    ) -> None:
        (v, g), (rv, rg) = _pair(z, y, loss, **options)
        message = f"{loss} {z.device} {z.dtype} {options}"
        torch.testing.assert_close(v, rv, rtol=rtol, atol=atol, msg=message)
        torch.testing.assert_close(g, rg, rtol=rtol, atol=atol, msg=message)

    def test_random_rows_reductions_and_shapes(self):
        for device in _devices():
            gen = torch.Generator(device=device).manual_seed(0)
            for n, c in ((1, 2), (5, 3), (64, 10), (33, 1000), (8, 1025)):
                z = 3 * torch.randn(n, c, dtype=torch.float64, device=device, generator=gen)
                y = torch.randint(0, c, (n,), device=device, generator=gen)
                for loss in K1:
                    for reduction in ("none", "sum", "mean"):
                        self.assert_pair(z, y, loss, reduction=reduction)

    def test_wide_rows(self):
        # Rows wider than the register block, and wider than 32768 classes.
        for device in _devices():
            gen = torch.Generator(device=device).manual_seed(5)
            for c in (5000, 40000):
                z = 3 * torch.randn(3, c, dtype=torch.float64, device=device, generator=gen)
                y = torch.randint(0, c, (3,), device=device, generator=gen)
                for loss in K1[1:]:
                    self.assert_pair(z, y, loss, rtol=1e-12)

    def test_smooth_gamma_from_tiny_to_huge(self):
        # Tiny gamma keeps the projected mass at the top margin; a huge one keeps the box.
        for device in _devices():
            gen = torch.Generator(device=device).manual_seed(1)
            z = torch.randn(16, 7, dtype=torch.float64, device=device, generator=gen)
            z[0] = torch.tensor([0.0, 1.0, 1.0, 1.0, -1.0, 0.5, 1.0 - 1e-12])
            y = torch.randint(0, 7, (16,), device=device, generator=gen)
            y[0] = 0
            for gamma in (1e-300, 1e-20, 1e-8, 0.5, 1.0, 3.0, 1e8, 1e300):
                for loss in ("topk_hinge_alpha", "topk_hinge_beta"):
                    self.assert_pair(z, y, loss, gamma=gamma, rtol=1e-12, atol=1e-300)

    def test_tiny_margins_and_large_offsets(self):
        # z_j - z_y rounds to -1 and only TwoSum's low part keeps 1 + z_j - z_y = 1e-20.
        for device in _devices():
            z = torch.tensor([[1.0, 1e-20, -5.0]], dtype=torch.float64, device=device)
            y = torch.tensor([0], device=device)
            for loss in K1[1:]:
                self.assert_pair(z, y, loss, gamma=1e-30)
            (v, _), _ = _pair(z, y, "topk_hinge_alpha")
            assert v[0].item() == 1e-20
            for offset in (1e8, -1e15):
                z = torch.tensor(
                    [[offset, offset - 1, offset - 5], [offset - 2, offset, offset + 3]],
                    dtype=torch.float64,
                    device=device,
                )
                y = torch.tensor([0, 1], device=device)
                for loss in K1:
                    self.assert_pair(z, y, loss, gamma=1e-30)

    def test_ties_take_the_first_largest_margin(self):
        for device in _devices():
            z = torch.tensor([[0.0, 2.0, 2.0, 2.0], [5.0, 5.0, 0.0, 5.0]], device=device)
            y = torch.tensor([0, 1], device=device)
            for loss in ("topk_hinge_alpha", "topk_hinge_beta"):
                self.assert_pair(z, y, loss)
                (_, g), _ = _pair(z, y, loss)
                assert g[0].tolist() == [-1.0, 1.0, 0.0, 0.0]
                assert g[1].tolist() == [1.0, -1.0, 0.0, 0.0]

    def test_dtypes_and_upstream_scaling(self):
        for device in _devices():
            gen = torch.Generator(device=device).manual_seed(2)
            z64 = torch.randn(32, 9, dtype=torch.float64, device=device, generator=gen)
            y = torch.randint(0, 9, (32,), device=device, generator=gen)
            dtypes = [torch.float32, torch.float64] + (
                [torch.float16, torch.bfloat16] if device == "cuda" else []
            )
            for dtype in dtypes:
                z = z64.to(dtype)
                eps = max(torch.finfo(dtype).eps, 1e-14)
                for loss in K1:
                    rtol = (
                        8 * eps
                        if loss[0] == "topk_cross_entropy" and dtype == torch.float32
                        else 2 * eps
                    )
                    self.assert_pair(z, y, loss, rtol=rtol, atol=0.0)
                    for scale in (1e-30, 3.0):
                        u = torch.full((32,), scale, dtype=dtype, device=device)
                        if dtype in (torch.float16, torch.bfloat16) and scale < 1e-10:
                            continue
                        self.assert_pair(
                            z, y, loss, reduction="none", upstream=u, rtol=max(rtol, 4 * eps)
                        )

    def test_validation(self):
        for device in _devices():
            for loss in K1:
                for z, y in (
                    (torch.tensor([[float("nan"), 0.0, 1.0]]), torch.tensor([0])),
                    (torch.tensor([[0.0, float("inf"), 1.0]]), torch.tensor([0])),
                    (torch.zeros(1, 3), torch.tensor([3])),
                    (torch.zeros(1, 3), torch.tensor([-1])),
                ):
                    name, gamma = _case(loss)
                    if device == "cpu":
                        with pytest.raises(ValueError):
                            sdca_loss(z.to(device), y.to(device), name, k=1, gamma=gamma)
                    else:
                        value = sdca_loss(z.to(device), y.to(device), name, k=1, gamma=gamma)
                        assert bool(value.isnan().all()), f"{loss} {device} {z} {y}"

    def test_first_order_only_and_layout(self):
        for device in _devices():
            z = torch.randn(4, 3, 5, dtype=torch.float64, device=device)
            y = torch.randint(0, 3, (4, 5), device=device)
            for loss, gamma in K1[1:]:
                a = z.clone().requires_grad_()
                b = z.clone().requires_grad_()
                torch.testing.assert_close(
                    sdca_loss(a, y, loss, k=1, gamma=gamma, dim=1, reduction="none"),
                    sdca_loss(b.movedim(1, -1), y, loss, k=1, gamma=gamma, reduction="none"),
                )
                x = z.movedim(1, -1).clone().requires_grad_()
                value = sdca_loss(x, y, loss, k=1, gamma=gamma)
                (g,) = torch.autograd.grad(value, x, create_graph=True)
                with pytest.raises(RuntimeError):
                    torch.autograd.grad(g.sum(), x)


class ChunkedIndexedLossTests(IndexedLossTests):
    """The same cases with CUDA rows wider than two columns in four-column chunks."""

    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        from libsdca import _triton_hinge as kernels

        saved = kernels._MAX_BLOCK, kernels._CHUNKS

        def restore() -> None:
            kernels._MAX_BLOCK, kernels._CHUNKS = saved
            kernels._compiled.cache_clear()

        kernels._MAX_BLOCK, kernels._CHUNKS = 2, {False: (4, 4), True: (4, 4)}
        kernels._compiled.cache_clear()
        self.restore = restore

    def teardown_method(self):
        self.restore()

    def test_running_top_across_chunks(self):
        # Tops rising by less or more than gamma per chunk, ties, a chunk of only the target;
        # no gamma puts a margin exactly on the threshold.
        c = 13
        j = torch.arange(c, dtype=torch.float64)
        z = torch.stack([0.1 * j, j, -0.1 * j, 0 * j, 2.0 * (j % 4 == 3)]).cuda()
        y = torch.tensor([12, 0, 5, 3, 12], device="cuda")
        for gamma in (1e-8, 0.25, 0.55, 2.0, 30.0):
            for loss in K1[1:]:
                self.assert_pair(z, y, loss, gamma=gamma, rtol=1e-12, atol=1e-300)
        (_, g), _ = _pair(z, y, "topk_hinge_alpha")
        assert g[4].nonzero().flatten().tolist() == [3, 12]
        z[1, 9] = float("nan")
        for loss, gamma in K1[1:]:
            assert bool(sdca_loss(z, y, loss, k=1, gamma=gamma).isnan().all())
