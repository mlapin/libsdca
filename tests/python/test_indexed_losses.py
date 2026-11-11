"""Integer-target fast paths at k = 1 against the boolean-mask (generic) path of the same loss."""

from typing import Any

import interpreter
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


def check_packed_hinge(device: str) -> None:
    """Short-row packed k = 1 hinge kernels across narrow C, non-multiple N, scales, ties,
    in-place backward and mixed invalid rows within one CTA."""
    import triton
    from libsdca import _triton_hinge as H
    from libsdca import _triton_reduce as R

    assert [H.rows_per_cta(c) for c in (2, 10, 16, 31, 64, 100, 128, 129, 1000)] == [
        16,
        16,
        16,
        16,
        8,
        4,
        4,
        1,
        1,
    ]
    gen = torch.Generator().manual_seed(17)
    classes = (
        (2, 7, 10, 16, 31, 64, 100, 128, 129) if device == "cuda" else (2, 7, 10, 31, 100, 129)
    )
    batches = (1, 3, 15, 16, 17, 65) if device == "cuda" else (1, 3, 15, 16, 17)
    for c in classes:
        for n in batches:
            z0 = 3 * torch.randn(n, c, dtype=torch.float64, generator=gen)
            y0 = torch.randint(0, c, (n,), generator=gen)
            z, y = z0.to(device), y0.to(device)
            for smooth, gamma in ((False, 0.0), (True, 0.5), (True, 1e-20)):
                loss_name = "topk_hinge_alpha"
                for reduction in ("none", "mean", "sum"):
                    u = (
                        torch.linspace(-1.5, 2.5, n, dtype=torch.float64)
                        if reduction == "none"
                        else torch.tensor(0.75, dtype=torch.float64)
                    ).to(device)
                    val, stats = H.forward(z, y, reduction, smooth, gamma)
                    grad = H.backward(z, y, stats, u, reduction, smooth, gamma)
                    mask = torch.nn.functional.one_hot(y0, c).bool()
                    ref_x = z0.clone().requires_grad_()
                    ref_val = sdca_loss(
                        ref_x,
                        mask,
                        loss_name,
                        k=1,
                        gamma=gamma,
                        reduction=reduction,
                        backend="torch",
                    )
                    ref_val.backward(u.cpu())
                    assert ref_x.grad is not None
                    torch.testing.assert_close(val.cpu(), ref_val.detach(), rtol=1e-12, atol=1e-15)
                    torch.testing.assert_close(grad.cpu(), ref_x.grad, rtol=1e-12, atol=1e-15)
            # In-place backward (G == Z) on packed tiles.
            for smooth, gamma in ((False, 0.0), (True, 0.5)):
                _, stats = H.forward(z, y, "mean", smooth, gamma)
                u = torch.tensor(1.25, dtype=torch.float64, device=device)
                expected = H.backward(z, y, stats, u, "mean", smooth, gamma)
                inplace = z.clone()
                block, warps = H._shape(c, smooth)
                rows = H.rows_per_cta(c, smooth)
                H._backward[(triton.cdiv(n, rows),)](
                    inplace,
                    y,
                    stats,
                    u,
                    inplace,
                    H._gamma(z.device, gamma),
                    n,
                    c,
                    block,
                    smooth,
                    1,
                    rows,
                    num_warps=warps,
                )
                torch.testing.assert_close(inplace, expected, rtol=0, atol=0)
    # 100-digit scale sweep (1e-30 to 1e30) and leftmost ties on packed C = 10.
    base_z = torch.randn(19, 10, dtype=torch.float64, generator=gen)
    base_y = torch.randint(0, 10, (19,), generator=gen)
    mask10 = torch.nn.functional.one_hot(base_y, 10).bool()
    for exp in (-30, -10, 0, 10, 30):
        scale = 10.0**exp
        zs = (base_z * scale).to(device)
        ys = base_y.to(device)
        gamma = 0.5 * scale
        val, stats = H.forward(zs, ys, "none", True, gamma)
        u = torch.ones(19, dtype=torch.float64, device=device)
        grad = H.backward(zs, ys, stats, u, "none", True, gamma)
        rx = (base_z * scale).requires_grad_()
        rv = sdca_loss(
            rx, mask10, "topk_hinge_alpha", k=1, gamma=gamma, reduction="none", backend="torch"
        )
        rv.sum().backward()
        assert rx.grad is not None
        torch.testing.assert_close(val.cpu(), rv.detach(), rtol=1e-12, atol=1e-15 * scale)
        torch.testing.assert_close(grad.cpu(), rx.grad, rtol=1e-12, atol=1e-15)
    # Mixed valid and invalid rows in the same 16-row CTA (C = 10, N = 9).
    clean = torch.randn(9, 10, dtype=torch.float64, generator=gen)
    targets = torch.randint(0, 10, (9,), generator=gen)
    bad_rows = {1: ("nan", 0), 3: ("inf", 0), 5: ("ok", -1), 7: ("ok", 10)}
    mixed = clean.clone()
    mixed_y = targets.clone()
    for r, (kind, bad_y) in bad_rows.items():
        if kind == "nan":
            mixed[r, (int(targets[r]) + 1) % 10] = float("nan")
        elif kind == "inf":
            mixed[r, (int(targets[r]) + 1) % 10] = float("inf")
        else:
            mixed_y[r] = bad_y
    mz, my = mixed.to(device), mixed_y.to(device)
    cz, cy = clean.to(device), targets.to(device)
    u = torch.ones(9, dtype=torch.float64, device=device)
    for smooth, gamma in ((False, 0.0), (True, 0.5)):
        ref_v, ref_s = H.forward(cz, cy, "none", smooth, gamma)
        ref_g = H.backward(cz, cy, ref_s, u, "none", smooth, gamma)
        if device == "cpu":
            with pytest.raises(ValueError):
                H.forward(mz, my, "none", smooth, gamma)
        stats = torch.empty(R.size(9, 4), device=device, dtype=torch.float64)
        out = torch.empty(9, device=device, dtype=torch.float64)
        block, warps = H._shape(10, smooth)
        rows = H.rows_per_cta(10, smooth)
        H._forward[(triton.cdiv(9, rows),)](
            mz, my, stats, H._gamma(mz.device, gamma), 9, 10, block, smooth, rows, num_warps=warps
        )
        R.launch(stats, out, 9, 0, H._LAYOUT)
        grad = H.backward(mz, my, stats, u, "none", smooth, gamma)
        for r in range(9):
            if r in bad_rows:
                assert bool(out[r].isnan()) and bool(grad[r].isnan().all())
            else:
                torch.testing.assert_close(out[r], ref_v[r], rtol=0, atol=0)
                torch.testing.assert_close(grad[r], ref_g[r], rtol=0, atol=0)


class PackedHingeKernelTests:
    def test_packed_hinge_in_interpreter(self):
        interpreter.run("import test_indexed_losses as t\nt.check_packed_hinge('cpu')\n")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_packed_hinge_cuda_spills_and_graphs(self):
        pytest.importorskip("triton")
        from libsdca import _triton_hinge as H

        check_packed_hinge("cuda")
        z = torch.empty((1, 1), device="cuda", dtype=torch.float32)
        y = torch.empty(1, device="cuda", dtype=torch.int64)
        s = torch.empty(1, device="cuda", dtype=torch.float64)
        g = torch.empty(1, device="cuda", dtype=torch.float64)
        for c in (2, 7, 10, 16, 31, 64, 100, 128):
            for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
                zd = z.to(dtype)
                for smooth in (False, True):
                    block, warps = H._shape(c, smooth)
                    rows = H.rows_per_cta(c, smooth)
                    kf = H._forward.warmup(
                        zd, y, s, g, 2, c, block, smooth, rows, num_warps=warps, grid=(1,)
                    )
                    kb = H._backward.warmup(
                        zd, y, s, g, zd, g, 2, c, block, smooth, 1, rows, num_warps=warps, grid=(1,)
                    )
                    kf._init_handles()
                    kb._init_handles()
                    assert kf.n_spills == 0, (c, dtype, smooth, kf.n_spills)
                    assert kb.n_spills == 0, (c, dtype, smooth, kb.n_spills)
        # CUDA graph capture across packed shapes.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for c in (10, 100):
            x = torch.randn(257, c, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            yt = torch.randint(0, c, (257,), device="cuda")
            for gamma in (0.0, 0.5):
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        (warm,) = torch.autograd.grad(
                            sdca_loss(x, yt, "topk_hinge_alpha", k=1, gamma=gamma), x
                        )
                        del warm
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    loss = sdca_loss(x, yt, "topk_hinge_alpha", k=1, gamma=gamma)
                    (captured,) = torch.autograd.grad(loss, x)
                with torch.no_grad():
                    x.normal_()
                ref_loss = sdca_loss(x, yt, "topk_hinge_alpha", k=1, gamma=gamma)
                (expected,) = torch.autograd.grad(ref_loss, x)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(loss, ref_loss, rtol=0, atol=0)
                torch.testing.assert_close(captured, expected, rtol=0, atol=0)
