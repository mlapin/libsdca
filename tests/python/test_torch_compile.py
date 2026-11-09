"""Tensor-path losses inside torch.compile (Inductor, CPU) match eager; the fused CUDA
losses compile without graph breaks through their torch.library operators."""

import shutil
from collections.abc import Callable
from typing import Any

import libsdca.torch as lt
import pytest
import torch
from libsdca import _cpu, _ops
from libsdca.torch import sdca_loss

LOSSES: list[tuple[str, dict[str, Any]]] = [
    ("multiclass_hinge", {}),
    ("topk_hinge_beta", {"k": 2}),
    ("topk_cross_entropy", {"k": 2}),
    ("proximal_cross_entropy", {}),
]


@pytest.mark.skipif(
    not (shutil.which("g++") or shutil.which("clang++")), reason="Inductor needs a C++ compiler"
)
class TestCompiledTensorLosses:
    @pytest.mark.slow
    @pytest.mark.parametrize(("loss", "options"), LOSSES, ids=[loss for loss, _ in LOSSES])
    def test_matches_eager(self, loss: str, options: dict[str, Any]):
        generator = torch.Generator().manual_seed(0)
        z = torch.randn(64, 10, generator=generator)
        y = torch.randint(0, 10, (64,), generator=generator)

        def f(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return sdca_loss(z, y, loss, backend="torch", reduction="none", **options)

        torch._dynamo.reset()
        torch.testing.assert_close(torch.compile(f)(z, y), f(z, y))


def _cuda_losses(
    n: int, c: int, device: str = "cuda"
) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Each fused loss of logits (n, c) on the device, with its fixed targets."""
    generator = torch.Generator(device=device).manual_seed(1)
    y = torch.randint(0, c, (n,), device=device, generator=generator)
    ignored = y.clone()
    ignored[::3] = -100
    mask = torch.rand(n, c, device=device, generator=generator) < 0.2
    mask[torch.arange(n), y] = True
    mask[torch.arange(n), (y + 1) % c] = False
    teacher = torch.randn(n, c, device=device, generator=generator)
    weight = torch.rand(c, device=device, generator=generator) + 0.5
    return {
        "cross_entropy": lambda z: lt.cross_entropy(z, y),
        "cross_entropy_none": lambda z: lt.cross_entropy(z, y, reduction="none").sum(),
        "weighted": lambda z: lt.cross_entropy(z, ignored, weight.to(z.dtype)),
        "sdca_loss": lambda z: sdca_loss(z, y, reduction="sum"),
        "hinge": lambda z: lt.multiclass_hinge_loss(z, y),
        "smooth_hinge": lambda z: lt.multiclass_hinge_loss(z, y, gamma=0.5),
        "topk_ce": lambda z: lt.topk_cross_entropy(z, y, k=3),
        "topk_hinge": lambda z: lt.topk_hinge_loss(z, y, k=3, variant="alpha"),
        "multilabel_ce": lambda z: lt.multilabel_cross_entropy(z, mask),
        "multilabel_hinge": lambda z: lt.multilabel_hinge_loss(z, mask, gamma=0.5),
        "distillation": lambda z: lt.distillation_loss(z, teacher.to(z.dtype), 2.0),
        "proximal_distillation": lambda z: lt.proximal_distillation_loss(
            z, teacher.to(z.dtype), 2.0
        ),
        "proximal_ce": lambda z: lt.proximal_cross_entropy(z, y),
    }


@pytest.mark.skipif(
    not (_cpu.AVAILABLE and (shutil.which("g++") or shutil.which("clang++"))),
    reason="needs torch 2.4 and a C++ compiler for Inductor",
)
class TestCompiledCpuLosses:
    def test_full_graph_matches_eager(self, subtests: pytest.Subtests):
        """The CPU kernels (extension or native) compile without graph breaks."""
        for dtype in (torch.float32, torch.bfloat16):
            z = torch.randn(12, 7, generator=torch.Generator().manual_seed(0)).to(dtype)
            for name, loss in _cuda_losses(12, 7, "cpu").items():
                with subtests.test(loss=name, dtype=dtype):
                    torch._dynamo.reset()
                    explained = torch._dynamo.explain(loss)(z.clone().requires_grad_(True))
                    assert explained.graph_break_count == 0, explained.break_reasons
                    compiled = torch.compile(loss, fullgraph=True)
                    eager_z = z.clone().requires_grad_(True)
                    compiled_z = z.clone().requires_grad_(True)
                    expected = loss(eager_z)
                    actual = compiled(compiled_z)
                    # A reduction in the graph may sum in another order.
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0)
                    expected.backward()
                    actual.backward()
                    eps = torch.finfo(dtype).eps
                    torch.testing.assert_close(compiled_z.grad, eager_z.grad, rtol=2 * eps, atol=0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_tensor_backend_cross_entropy_compiles_in_one_graph(
        self, subtests: pytest.Subtests, dtype: torch.dtype
    ):
        """backend="torch" has no data-dependent branch, in the forward or the backward:
        fullgraph with and without ignored rows and class weights (the eager backend: graph
        capture, not Inductor's code)."""
        z = torch.randn(4, 3, generator=torch.Generator().manual_seed(0), dtype=dtype)
        z[3, 1] = 45.0  # a far row: its target leads by more than 37
        weight = torch.tensor([1.0, 2.0, 0.5], dtype=dtype)
        cases = {
            "plain": (torch.tensor([0, 1, 2, 1]), None),
            "ignored": (torch.tensor([0, -100, 2, 1]), None),
            "weighted": (torch.tensor([0, -100, 2, 1]), weight),
        }
        for name, (y, w) in cases.items():
            with subtests.test(case=name):

                def loss(x: torch.Tensor, y: torch.Tensor = y, w: Any = w) -> torch.Tensor:
                    return lt.cross_entropy(x, y, weight=w, backend="torch")

                torch._dynamo.reset()
                compiled = torch.compile(loss, backend="eager", fullgraph=True)
                eager_z = z.clone().requires_grad_(True)
                compiled_z = z.clone().requires_grad_(True)
                expected, actual = loss(eager_z), compiled(compiled_z)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected.backward()
                actual.backward()
                torch.testing.assert_close(compiled_z.grad, eager_z.grad, rtol=0, atol=0)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and _ops.AVAILABLE), reason="needs CUDA and torch 2.4"
)
class TestCompiledCudaLosses:
    def test_full_graph_matches_eager(self, subtests: pytest.Subtests):
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            z0 = (torch.randn(40, 70, device="cuda") * 3).to(dtype)
            for name, loss in _cuda_losses(40, 70).items():
                with subtests.test(dtype=dtype, loss=name):
                    z = z0.clone().requires_grad_()
                    expected = loss(z)
                    (expected_grad,) = torch.autograd.grad(expected, z)
                    torch._dynamo.reset()
                    explained = torch._dynamo.explain(loss)(z)
                    assert explained.graph_break_count == 0
                    assert explained.break_reasons == []
                    compiled = torch.compile(loss, fullgraph=True)
                    torch.autograd.grad(compiled(z), z)
                    actual = compiled(z)
                    (grad,) = torch.autograd.grad(actual, z)
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0)
                    torch.testing.assert_close(grad, expected_grad, rtol=1e-6, atol=0)

    def test_backward_graph_calls_the_kernel(self, subtests: pytest.Subtests):
        # The compiled backward launches the backward kernel, not the create_graph rule.
        from torch._dynamo.backends.common import aot_autograd

        graphs: list[str] = []

        def keep(graph: torch.fx.GraphModule, _: Any) -> Any:
            graphs.append(graph.code)
            return graph.forward

        backend = aot_autograd(fw_compiler=keep, bw_compiler=keep)
        losses = _cuda_losses(8, 16)
        for name, op in (
            ("cross_entropy", "weighted_entropy_backward"),
            ("weighted", "weighted_entropy_backward"),
            ("sdca_loss", "entropy_backward"),
            ("hinge", "hinge_backward"),
            ("topk_hinge", "topk_backward"),
            ("multilabel_ce", "multilabel_backward"),
            ("distillation", "distillation_backward"),
        ):
            with subtests.test(loss=name):
                graphs.clear()
                torch._dynamo.reset()
                z = torch.randn(8, 16, device="cuda", requires_grad=True)
                torch.autograd.grad(torch.compile(losses[name], backend=backend)(z), z)
                assert len(graphs) == 2
                assert f"libsdca.{op}" in graphs[1]
                assert "float64" not in graphs[1]

    def test_operator_registrations(self, subtests: pytest.Subtests):
        z = torch.randn(6, 9, device="cuda", requires_grad=True)
        y = torch.randint(0, 9, (6,), device="cuda")
        mask = torch.rand(6, 9, device="cuda") < 0.3
        mask[:, 0] = True
        w = torch.rand(9, device="cuda") + 0.5
        ops = torch.ops.libsdca
        cases = [
            (ops.entropy, (z, y, "mean", True)),
            (ops.entropy, (z, y, "none", False)),
            (ops.weighted_entropy, (z, y, w, -100, "mean", True)),
            (ops.weighted_entropy, (z, y, None, 2, "sum", True)),
            (ops.hinge, (z, y, "mean", True, 0.5)),
            (ops.topk, (z, y, "none", "topk_hinge_beta", 2, 0.0, False)),
            (ops.topk, (z, y, "mean", "topk_cross_entropy", 2, 0.0, True)),
            (ops.multilabel, (z, mask, "cross_entropy", "mean", 0.0, True)),
            (ops.multilabel, (z, mask, "cross_entropy", "none", 0.0, False)),
            (ops.multilabel, (z, mask, "multilabel_hinge_smooth", "sum", 0.5, False)),
            (ops.distillation, (z.detach(), z.detach().flip(1), 2.0, False, "none")),
            (ops.distillation, (z.detach(), z.detach().flip(1), 0.5, False, "mean")),
            (ops.proximal_distillation, (z.detach(), z.detach().flip(1), 2.0, 1.0, False)),
            (ops.proximal_entropy, (z.detach(), torch.eye(9, device="cuda")[y] > 0, 1.0, True)),
        ]
        for op, args in cases:
            with subtests.test(op=str(op), args=args[2:]):
                torch.library.opcheck(op, args)

    def test_operator_higher_derivatives(self):
        # create_graph through an operator: the entropies' Hessian-vector products match
        # eager; first-order losses raise when their gradient is differentiated.
        z = torch.randn(5, 7, device="cuda", dtype=torch.float64, requires_grad=True)
        y = torch.randint(0, 7, (5,), device="cuda")
        v = torch.randn_like(z)
        weight = torch.rand(7, device="cuda", dtype=torch.float64) + 0.5
        pairs = [
            (_ops.entropy(z, y, "mean", False), torch.nn.functional.cross_entropy(z, y)),
            (
                _ops.weighted_entropy(z, y, weight, -100, "sum", False),
                torch.nn.functional.cross_entropy(z, y, weight, reduction="sum"),
            ),
        ]
        for actual, expected in pairs:
            (g,) = torch.autograd.grad(actual, z, create_graph=True)
            (h,) = torch.autograd.grad((g * v).sum(), z)
            (g0,) = torch.autograd.grad(expected, z, create_graph=True)
            (h0,) = torch.autograd.grad((g0 * v).sum(), z)
            torch.testing.assert_close(h, h0)
        (g,) = torch.autograd.grad(_ops.hinge(z, y, "mean", False, 0.0), z, create_graph=True)
        with pytest.raises(RuntimeError, match="first derivatives only"):
            torch.autograd.grad(g.sum(), z)
