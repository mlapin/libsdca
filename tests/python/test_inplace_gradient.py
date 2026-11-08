"""The C++ launcher's backward writes the gradient over dead logits, and only then."""

import datetime
import tempfile
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from libsdca import _cuda_launch
from libsdca.torch import (
    cross_entropy,
    multiclass_hinge_loss,
    multilabel_cross_entropy,
    multilabel_hinge_loss,
    sdca_loss,
    topk_cross_entropy,
    topk_hinge_loss,
)
from torch.multiprocessing.spawn import spawn

Loss = Callable[[torch.Tensor], torch.Tensor]


def _launcher() -> bool:
    try:
        import triton

        return (
            _cuda_launch.kernels_supported(triton.__version__) and _cuda_launch.host() is not None
        )
    except ImportError:
        return False


def _losses(n: int, c: int, device: str) -> dict[str, Loss]:
    """Every loss with a C++ backward, on n x c logits."""
    gen = torch.Generator(device=device).manual_seed(1)
    y = torch.randint(0, c, (n,), device=device, generator=gen)
    ignored = y.clone()
    ignored[::3] = -100
    weight = torch.rand(c, device=device, generator=gen) + 0.5
    mask = torch.rand(n, c, device=device, generator=gen) < 0.05
    mask[torch.arange(n, device=device), y] = True
    return {
        "cross_entropy": lambda z: cross_entropy(z, y),
        "weighted": lambda z: cross_entropy(z, ignored, weight.to(z.dtype), reduction="sum"),
        "sdca_entropy": lambda z: sdca_loss(z, y, reduction="none").sum(),
        "hinge": lambda z: multiclass_hinge_loss(z, y),
        "smooth_hinge": lambda z: multiclass_hinge_loss(z, y, gamma=0.5),
        "topk_entropy": lambda z: topk_cross_entropy(z, y, k=3),
        "topk_hinge": lambda z: topk_hinge_loss(z, y, k=3, variant="alpha"),
        "multilabel_entropy": lambda z: multilabel_cross_entropy(z, mask),
        "multilabel_hinge": lambda z: multilabel_hinge_loss(z, mask),
        "multilabel_smooth": lambda z: multilabel_hinge_loss(z, mask, gamma=0.5),
    }


def _grad(t: torch.Tensor) -> torch.Tensor:
    assert t.grad is not None
    return t.grad


def _full(t: Any) -> torch.Tensor:
    """A DTensor's global value."""
    return t.full_tensor()


def _source(n: int, c: int, dtype: torch.dtype) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(0)
    return (3 * torch.randn(n, c, device="cuda", generator=gen)).to(dtype).requires_grad_()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and _launcher()), reason="the C++ CUDA launcher is off"
)
class InplaceGradientTests:
    # Wider than the backward column blocks and the hinge chunks.
    N, C = 33, 3000

    def _run(
        self,
        loss: Loss,
        x: torch.Tensor,
        keep: bool,
        shape: tuple[int, ...] | None = None,
        create_graph: bool = False,
    ) -> tuple[int, int]:
        """x.grad from loss(z) of z = x.clone() (or a view of it); (z's address, gradient's)."""
        seen: list[int] = []
        x.grad = None
        handle = x.register_hook(lambda g: seen.append(g.data_ptr()))
        z = x.clone() if shape is None else x.clone().reshape(shape).flatten(0, 1)
        address = z.data_ptr()
        kept = z if keep else None
        value = loss(z)
        del z
        value.backward(create_graph=create_graph, retain_graph=False)
        handle.remove()
        if kept is not None:
            torch.testing.assert_close(kept, x.detach(), rtol=0, atol=0)
        return address, seen[0]

    def test_dead_logits_take_the_gradient(self, subtests: pytest.Subtests):
        for dtype in (torch.float32, torch.bfloat16, torch.float64):
            for name, loss in _losses(self.N, self.C, "cuda").items():
                with subtests.test(dtype=dtype, loss=name):
                    x = _source(self.N, self.C, dtype)
                    address, gradient = self._run(loss, x, keep=True)
                    assert address != gradient
                    expected = _grad(x).clone()
                    address, gradient = self._run(loss, x, keep=False)
                    assert address == gradient
                    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0, equal_nan=True)

    def test_view_of_dead_base(self, subtests: pytest.Subtests):
        """(B, T, V) logits flattened to (B * T, V): in place once the base is dead."""
        loss = _losses(6 * 7, self.C, "cuda")["cross_entropy"]
        x = _source(6 * 7, self.C, torch.bfloat16)
        address, gradient = self._run(loss, x, keep=True, shape=(6, 7, self.C))
        expected = _grad(x).clone()
        address, gradient = self._run(loss, x, keep=False, shape=(6, 7, self.C))
        assert address == gradient
        torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
        # A live base, or another view of it, keeps the logits.
        for other in ("base", "view"):
            with subtests.test(other=other):
                base = x.detach().clone().reshape(6, 7, self.C).requires_grad_()
                copy = base.detach().clone()
                z = base.flatten(0, 1)
                keep = base if other == "base" else base[0]
                if other == "view":
                    base = None  # type: ignore[assignment]
                value = loss(z)
                del z
                value.backward()
                torch.testing.assert_close(
                    keep, copy if other == "base" else copy[0], rtol=0, atol=0
                )

    def test_retain_graph_and_saved_tensors_keep_logits(self, subtests: pytest.Subtests):
        for name, loss in _losses(self.N, self.C, "cuda").items():
            with subtests.test(loss=name):
                x = _source(self.N, self.C, torch.float32)
                self._run(loss, x, keep=True)
                expected = _grad(x).clone()
                x.grad = None
                value = loss(x.clone())
                value.backward(retain_graph=True)
                value.backward()
                torch.testing.assert_close(x.grad, 2 * expected, equal_nan=True)
                # Another node saves the logits: the sum of both gradients, either order.
                for first in (True, False):
                    x.grad = None
                    z = x.clone()
                    other = (z * z).sum()
                    value = other + loss(z) if first else loss(z) + other
                    del z
                    value.backward()
                    reference = expected + 2 * x.detach()
                    torch.testing.assert_close(x.grad, reference, equal_nan=True)

    def test_weakref_or_storage_keeps_logits(self):
        loss = _losses(self.N, self.C, "cuda")["cross_entropy"]
        x = _source(self.N, self.C, torch.float32)
        for holder in (weakref.ref, lambda z: z.untyped_storage()):
            seen: list[int] = []
            handle = x.register_hook(lambda g: seen.append(g.data_ptr()))
            z = x.clone()
            address, kept = z.data_ptr(), holder(z)
            value = loss(z)
            del z
            value.backward()
            handle.remove()
            assert address != seen[0]
            del kept

    def test_saved_tensor_hooks_keep_logits(self):
        loss = _losses(self.N, self.C, "cuda")["cross_entropy"]
        x = _source(self.N, self.C, torch.float32)
        self._run(loss, x, keep=True)
        expected = _grad(x).clone()
        packed: list[torch.Tensor] = []

        def pack(t: torch.Tensor) -> torch.Tensor:
            packed.append(t)
            return t

        x.grad = None
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            value = loss(x.clone())
        value.backward()
        torch.testing.assert_close(x.grad, expected)
        torch.testing.assert_close(packed[0], x.detach(), rtol=0, atol=0)
        x.grad = None
        with torch.autograd.graph.save_on_cpu():
            value = loss(x.clone())
        value.backward()
        torch.testing.assert_close(x.grad, expected)

    def test_create_graph_keeps_logits(self, subtests: pytest.Subtests):
        for name, loss in _losses(self.N, self.C, "cuda").items():
            with subtests.test(loss=name):
                x = _source(self.N, self.C, torch.float64)
                self._run(loss, x, keep=True)
                expected = _grad(x).clone()
                address, gradient = self._run(loss, x, keep=False, create_graph=True)
                assert address != gradient
                torch.testing.assert_close(x.grad, expected, equal_nan=True)

    def test_foreign_memory_is_kept(self):
        """DLPack-imported logits may still be read by their owner."""
        held = _source(self.N, self.C, torch.float32).detach()
        copy = held.clone()

        class Alias(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, *args: Any) -> torch.Tensor:
                return torch.from_dlpack(held)

            @staticmethod
            def backward(ctx: Any, *grads: Any) -> Any:
                return grads[0]

        x = torch.zeros_like(held, requires_grad=True)
        logits = cast(torch.Tensor, Alias.apply(x))
        _losses(self.N, self.C, "cuda")["cross_entropy"](logits).backward()
        torch.testing.assert_close(held, copy, rtol=0, atol=0)

    def test_compiled_step_matches_eager(self):
        torch.manual_seed(0)
        w = torch.randn(self.C, 64, device="cuda", requires_grad=True)
        h = torch.randn(self.N, 64, device="cuda")
        loss = _losses(self.N, self.C, "cuda")["cross_entropy"]

        def step(h: torch.Tensor) -> torch.Tensor:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return loss(h @ w.t())

        step(h).backward()
        expected = _grad(w).clone()
        w.grad = None
        torch.compile(step)(h).backward()
        torch.testing.assert_close(w.grad, expected)

    def test_peak_memory_drops_by_the_gradient(self):
        x = _source(4096, 8192, torch.bfloat16)
        loss = _losses(4096, 8192, "cuda")["cross_entropy"]
        peaks = []
        for keep in (True, False):
            z = x.detach().clone().requires_grad_()
            w = z * 1
            value = loss(w)
            kept = w if keep else None
            del w
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            value.backward()
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() - base)
            del kept, z, value
        logits = x.numel() * x.element_size()
        assert peaks[1] < peaks[0] - logits // 2


def dtensor_worker(rank: int, rendezvous: str) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=1,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        mesh = init_device_mesh("cuda", (1,))
        loss = _losses(16, 3000, "cuda")["cross_entropy"]
        x = _source(16, 3000, torch.float32)
        loss(x.clone()).backward()
        expected = _grad(x).clone()
        for placement in (Replicate(), Shard(1)):
            # A DTensor the caller keeps: its local values stay intact.
            x.grad = None
            logits = distribute_tensor(x.detach().clone(), mesh, [placement]).requires_grad_()
            copy = _full(logits).clone()
            _full(loss(logits)).backward()
            torch.testing.assert_close(_full(logits), copy, rtol=0, atol=0)
            torch.testing.assert_close(_full(logits.grad), expected)
            # Dead DTensor logits from a differentiable op: the same gradient.
            source = distribute_tensor(x.detach().clone(), mesh, [placement]).requires_grad_()
            _full(loss(source * 1)).backward()
            torch.testing.assert_close(_full(source.grad), expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and _launcher()), reason="the C++ CUDA launcher is off"
)
class InplaceDTensorTests:
    def test_dtensor_logits(self):
        with tempfile.TemporaryDirectory() as tmp:
            spawn(dtensor_worker, args=(str(Path(tmp) / "rendezvous"),), nprocs=1, join=True)
