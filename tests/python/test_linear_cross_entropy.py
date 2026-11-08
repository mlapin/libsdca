"""linear_cross_entropy: chunked rows against F.linear + cross_entropy, gradcheck, the CPU
paths, torch.compile and the Triton kernels in the interpreter."""

import itertools
import os
import shutil
from collections.abc import Callable, Sequence
from typing import Any
from unittest import mock

import interpreter
import pytest
import torch
import torch.nn.functional as F
from libsdca import LinearCrossEntropyLoss, _linear_cross_entropy, linear_cross_entropy
from libsdca.torch import cross_entropy

lce = _linear_cross_entropy
REDUCTIONS = ("none", "sum", "mean")


def inputs(
    n: int, h: int, v: int, dtype: torch.dtype = torch.float32, seed: int = 0, ignored: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """hidden (n, h), weight (v, h), bias, targets (every ignored-th -100) and class weights."""
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n, h, generator=gen).to(dtype)
    weight = (torch.randn(v, h, generator=gen) * 3 / h**0.5).to(dtype)
    bias = torch.randn(v, generator=gen).to(dtype)
    target = torch.randint(0, v, (n,), generator=gen)
    if ignored:
        target[::ignored] = -100
    class_weight = (torch.rand(v, generator=gen) + 0.5).to(dtype)
    return hidden, weight, bias, target, class_weight


def grads(
    loss: Callable[..., torch.Tensor], *tensors: torch.Tensor, upstream: torch.Tensor | None = None
) -> tuple[torch.Tensor, ...]:
    """The value and gradients of loss(*leaves) for leaf copies of the tensors."""
    leaves = [t.detach().clone().requires_grad_() for t in tensors]
    value = loss(*leaves)
    up = torch.ones_like(value) if upstream is None else upstream
    return (value.detach(), *torch.autograd.grad(value, leaves, up))


def reference(dtype: torch.dtype) -> Callable[..., torch.Tensor]:
    """cross_entropy of the logits linear_cross_entropy computes: on the CPU, float32 GEMMs of
    the operands rounded into dtype, rounded into dtype."""

    def call(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor, y: torch.Tensor, **kw: Any):
        z = F.linear(h.float(), w.float(), b.float()).to(dtype) if dtype in lce._HALF else None
        z = F.linear(h, w, b) if z is None else z
        return cross_entropy(z, y, **kw)

    return call


def close(
    actual: Sequence[torch.Tensor | None], expected: Sequence[torch.Tensor | None], tol: float
):
    """Values and gradients within tol of the largest entry of each."""
    for a, e in zip(actual, expected, strict=True):
        assert a is not None and e is not None
        assert a.shape == e.shape and a.dtype == e.dtype, (a.shape, e.shape, a.dtype, e.dtype)
        scale = e.double().abs().max().clamp(min=1e-30)
        assert ((a.double() - e.double()).abs().max() / scale).item() <= tol


@pytest.fixture(params=["on", "off"])
def extension(request: pytest.FixtureRequest) -> Any:
    """The CPU extension on and off (the native library)."""
    with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": request.param}):
        yield request.param


class LinearCrossEntropyTests:
    def test_gradcheck_float64(self, subtests: pytest.Subtests):
        # Chunks of two rows; ignored rows, class weights and every reduction.
        h, w, b, y, cw = inputs(7, 4, 5, torch.float64, ignored=3)
        for backend, reduction, weighted, recompute in itertools.product(
            ("auto", "native", "torch"), REDUCTIONS, (False, True), (False, True)
        ):
            with subtests.test(
                backend=backend, reduction=reduction, weighted=weighted, recompute=recompute
            ):

                def f(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    return linear_cross_entropy(
                        h,
                        w,
                        y,
                        b,
                        class_weight=cw if weighted else None,
                        reduction=reduction,
                        backend=backend,
                        chunk_bytes=2 * 5 * 8 * (5 if backend == "torch" else 1),
                        recompute=recompute,
                    )

                leaves = [t.clone().requires_grad_() for t in (h, w, b)]
                assert torch.autograd.gradcheck(f, leaves)

    def test_matches_linear_then_cross_entropy(self, extension: str, subtests: pytest.Subtests):
        # Chunks of 3 rows of 10 (the last one short), against the full logits.
        for dtype, tol in (
            (torch.float32, 1e-5),
            (torch.float64, 1e-12),
            (torch.bfloat16, 1e-2),
            (torch.float16, 2e-3),
        ):
            h, w, b, y, cw = inputs(10, 8, 37, dtype, ignored=4)
            for reduction in REDUCTIONS:
                for weighted in (False, True):
                    kw: dict[str, Any] = dict(reduction=reduction)
                    if weighted:
                        kw["weight"] = cw

                    def ours(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                        options = {k: v for k, v in kw.items() if k != "weight"}
                        return linear_cross_entropy(
                            h,
                            w,
                            y,
                            b,
                            class_weight=kw.get("weight"),
                            chunk_bytes=3 * 37 * 8,
                            **options,
                        )

                    def theirs(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                        return reference(dtype)(h, w, b, y, **kw)

                    with subtests.test(dtype=dtype, reduction=reduction, weighted=weighted):
                        close(grads(ours, h, w, b), grads(theirs, h, w, b), tol)

    def test_matches_pytorch_float64(self, subtests: pytest.Subtests):
        # Every dtype's result against F.cross_entropy of float64 logits of its operands.
        for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2), (torch.float16, 3e-3)):
            h, w, b, y, cw = inputs(64, 16, 300, dtype, seed=1, ignored=5)
            for reduction in REDUCTIONS:
                with subtests.test(dtype=dtype, reduction=reduction):
                    ours = grads(
                        lambda h, w, b: linear_cross_entropy(
                            h, w, y, b, class_weight=cw, reduction=reduction, chunk_bytes=4096
                        ),
                        h,
                        w,
                        b,
                    )
                    theirs = grads(
                        lambda h, w, b: F.cross_entropy(
                            F.linear(h, w, b), y, cw.double(), reduction=reduction
                        ),
                        h.double(),
                        w.double(),
                        b.double(),
                    )
                    rounded = tuple(t.to(o.dtype) for o, t in zip(ours, theirs, strict=True))
                    close(ours, rounded, tol)

    def test_backends_agree(self, subtests: pytest.Subtests):
        h, w, b, y, cw = inputs(33, 8, 129, torch.float32, ignored=4)
        results = {}
        for backend, setting in (
            ("auto", "on"),
            ("auto", "off"),
            ("native", "on"),
            ("torch", "on"),
        ):
            with mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": setting}):
                results[backend, setting] = grads(
                    lambda h, w, b: linear_cross_entropy(
                        h, w, y, b, class_weight=cw, backend=backend, chunk_bytes=5000
                    ),
                    h,
                    w,
                    b,
                )
        expected = results["native", "on"]
        for key, result in results.items():
            with subtests.test(backend=key):
                close(result, expected, 1e-6)

    def test_edge_shapes(self, extension: str, subtests: pytest.Subtests):
        h, w, b, y, _ = inputs(12, 6, 2, torch.float32)
        cases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            "one row": (h[:1], y[:1]),
            "one example (H,)": (h[0], y[0]),
            "(B, T, H)": (h.reshape(3, 4, 6), y.reshape(3, 4)),
            "two classes": (h, y),
        }
        for name, (hidden, target) in cases.items():
            for reduction in REDUCTIONS:
                with subtests.test(case=name, reduction=reduction):
                    ours = grads(
                        lambda h, w, b: linear_cross_entropy(
                            h, w, target, b, reduction=reduction, chunk_bytes=8
                        ),
                        hidden,
                        w,
                        b,
                    )
                    theirs = grads(
                        lambda h, w, b: cross_entropy(
                            F.linear(h, w, b).reshape(-1, 2),
                            target.reshape(-1),
                            reduction=reduction,
                        ).reshape(target.shape if reduction == "none" else ()),
                        hidden,
                        w,
                        b,
                    )
                    close(ours, theirs, 1e-6)

    def test_ignored_and_empty(self, extension: str, subtests: pytest.Subtests):
        h, w, b, y, _ = inputs(5, 3, 4)
        for name, (hidden, target) in {
            "all ignored": (h, torch.full_like(y, -100)),
            "no rows": (h[:0], y[:0]),
            "no rows (B, 0, H)": (h[:0].reshape(2, 0, 3), y[:0].reshape(2, 0)),
        }.items():
            for reduction in REDUCTIONS:
                with subtests.test(case=name, reduction=reduction):
                    value, gh, gw, gb = grads(
                        lambda h, w, b: linear_cross_entropy(h, w, target, b, reduction=reduction),
                        hidden,
                        w,
                        b,
                    )
                    if reduction == "mean":
                        assert bool(value.isnan())
                    else:
                        assert bool((value == 0).all())
                    for g in (gh, gw, gb):
                        assert bool((g == 0).all())

    def test_weightless_mean_is_undefined(self, extension: str, subtests: pytest.Subtests):
        """The mean over rows of total class weight 0 has NaN gradients, as F.linear +
        F.cross_entropy, on every backend; ignored rows keep zero gradients."""
        for dtype, backend, recompute in itertools.product(
            (torch.float32, torch.float64), ("auto", "torch"), (False, True)
        ):
            with subtests.test(dtype=dtype, backend=backend, recompute=recompute):
                h = torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=dtype)
                w = torch.tensor([[2.0, 1.0], [1.0, 1.0]], dtype=dtype)
                b = torch.tensor([0.5, -0.5], dtype=dtype)
                y = torch.tensor([0, -100])
                weight = torch.tensor([0.0, 1.0], dtype=dtype)

                def ours(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    return linear_cross_entropy(
                        h, w, y, b, class_weight=weight, backend=backend, recompute=recompute
                    )

                def theirs(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    return F.cross_entropy(F.linear(h, w, b), y, weight=weight)

                for got, expected in zip(grads(ours, h, w, b), grads(theirs, h, w, b), strict=True):
                    torch.testing.assert_close(got, expected, equal_nan=True)
                assert bool(grads(ours, h, w, b)[1][0].isnan().all())
                assert bool((grads(ours, h, w, b)[1][1] == 0).all())

    def test_custom_ignore_index_and_upstream(self):
        h, w, b, y, cw = inputs(9, 5, 11, torch.float64)
        y[1::3] = 7
        upstream = torch.linspace(-2, 3, 9, dtype=torch.float64)
        ours = grads(
            lambda h, w, b: linear_cross_entropy(
                h, w, y, b, class_weight=cw, ignore_index=7, reduction="none", chunk_bytes=200
            ),
            h,
            w,
            b,
            upstream=upstream,
        )
        theirs = grads(
            lambda h, w, b: F.cross_entropy(
                F.linear(h, w, b), y, cw, ignore_index=7, reduction="none"
            ),
            h,
            w,
            b,
            upstream=upstream,
        )
        close(ours, theirs, 1e-12)

    def test_far_ahead_rows_keep_their_tails(self, extension: str):
        # The correct class 20 ahead: libsdca's loss and gradient, where PyTorch's are 0.
        hidden = torch.tensor([[1.0, 0.0]], requires_grad=True)
        weight = torch.tensor([[0.0, 0.0], [-20.0, 0.0]], requires_grad=True)
        value = linear_cross_entropy(hidden, weight, torch.tensor([0]))
        value.backward()
        expected = torch.tensor(-20.0, dtype=torch.float64).exp().log1p()
        assert abs(value.item() / expected.item() - 1) < 1e-6
        assert hidden.grad is not None and weight.grad is not None
        assert abs(weight.grad[1, 0].item() / expected.item() - 1) < 1e-6

    def test_chunks_bound_the_logits(self, extension: str, subtests: pytest.Subtests):
        h, w, b, y, _ = inputs(50, 4, 100, ignored=7)
        cases = [(torch.float32, 4000), (torch.bfloat16, 4000), (torch.float32, 1)]
        for (dtype, budget), recompute in itertools.product(cases, (False, True)):
            rows: list[int] = []
            original = lce._logits

            def record(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
                z, hg = original(*args)
                rows.append(z.shape[0])
                return z, hg

            with (
                subtests.test(dtype=dtype, budget=budget, recompute=recompute),
                mock.patch.object(lce, "_logits", record),
            ):
                grads(
                    lambda h, w, b: linear_cross_entropy(
                        h, w, y, b, chunk_bytes=budget, recompute=recompute
                    ),
                    h.to(dtype),
                    w.to(dtype),
                    b.to(dtype),
                )
                kernel = lce._kernel(h.device, dtype, 100, "auto")
                limit = lce.chunk_rows(100, dtype, kernel, budget)
                assert limit == max(1, budget // (100 * dtype.itemsize))
                assert max(rows) == limit
                # The 42 rows not ignored: in the forward, and with recompute in the backward.
                assert sum(rows) == (2 if recompute else 1) * 42

    def test_weight_blocks(self, subtests: pytest.Subtests):
        # 16-bit CPU GEMMs convert the weight 5 classes at a time (37 classes: a short block).
        h, w, b, y, cw0 = inputs(10, 8, 37, ignored=4)
        for dtype, recompute in itertools.product((torch.bfloat16, torch.float16), (False, True)):
            cw = cw0.to(dtype)

            def ours(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return linear_cross_entropy(
                    h, w, y, b, class_weight=cw, chunk_bytes=3 * 37 * 2, recompute=recompute
                )

            def theirs(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return reference(dtype)(h, w, b, y, weight=cw)

            with (
                subtests.test(dtype=dtype, recompute=recompute),
                mock.patch.object(lce, "_BLOCK_BYTES", 4 * 8 * 5),
            ):
                args = (h.to(dtype), w.to(dtype), b.to(dtype))
                close(grads(ours, *args), grads(theirs, *args), 1e-2)

    def test_recompute_matches(self, extension: str, subtests: pytest.Subtests):
        # Gradients of the forward and of the backward's recomputation: equal at upstream
        # derivative 1, within a rounding otherwise; again after retain_graph.
        for dtype in (torch.float32, torch.bfloat16, torch.float64):
            h, w, b, y, cw = inputs(30, 8, 77, dtype, ignored=4)
            for reduction, factor in (("mean", 1.0), ("sum", 1.0), ("mean", -3.0)):
                results = []
                for recompute in (False, True):
                    leaves = [t.clone().requires_grad_() for t in (h, w, b)]
                    loss = factor * linear_cross_entropy(
                        leaves[0],
                        leaves[1],
                        y,
                        leaves[2],
                        class_weight=cw,
                        reduction=reduction,
                        chunk_bytes=1000,
                        recompute=recompute,
                    )
                    loss.backward(retain_graph=True)
                    first = [t.grad.clone() for t in leaves if t.grad is not None]
                    loss.backward()
                    second = [t.grad for t in leaves if t.grad is not None]
                    results.append((loss.detach(), *first))
                    for a, b2 in zip(first, second, strict=True):
                        torch.testing.assert_close(2 * a, b2, rtol=0, atol=0)
                with subtests.test(dtype=dtype, reduction=reduction, factor=factor):
                    if factor == 1.0:
                        for a, b2 in zip(*results, strict=True):
                            torch.testing.assert_close(a, b2, rtol=0, atol=0)
                    else:
                        close(results[0], results[1], 1e-6 if dtype == torch.float32 else 1e-2)

    def test_autocast(self, subtests: pytest.Subtests):
        # float32 parameters under bfloat16 autocast: bfloat16 logits, float32 loss and
        # gradients (accumulated in float32) against the autocast composition.
        h, w, b, y, _ = inputs(16, 8, 50, seed=3)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            ours = grads(lambda h, w, b: linear_cross_entropy(h, w, y, b), h, w, b)
            theirs = grads(lambda h, w, b: cross_entropy(F.linear(h, w, b), y), h, w, b)
            assert lce.explain(h, w, b)[1] == "bfloat16"
        assert all(t.dtype == torch.float32 for t in ours)
        close(ours, theirs, 2e-2)
        with pytest.raises(ValueError, match="share one floating-point dtype"):
            linear_cross_entropy(h.double(), w, y, b)

    def test_rejects_invalid_inputs(self, extension: str, subtests: pytest.Subtests):
        h, w, b, y, cw = inputs(4, 3, 5)
        bad_target = y.clone()
        bad_target[2] = 5
        nonfinite = h.clone()
        nonfinite[1, 0] = float("inf")
        negative = cw.clone()
        negative[0] = -1
        cases = {
            "Target index out of range": lambda: linear_cross_entropy(h, w, bad_target, b),
            "logits must be finite": lambda: linear_cross_entropy(nonfinite, w, y, b),
            "finite and nonnegative": lambda: linear_cross_entropy(
                h, w, y, b, class_weight=negative
            ),
            "share one floating-point dtype": lambda: linear_cross_entropy(h, w.double(), y, b),
            "hidden must be": lambda: linear_cross_entropy(h, w.T, y, b),
            "bias must have": lambda: linear_cross_entropy(h, w, y, b[:2]),
            "int64 class indices": lambda: linear_cross_entropy(h, w, y.int(), b),
            "label_smoothing": lambda: linear_cross_entropy(h, w, y, label_smoothing=0.1),
            "chunk_bytes": lambda: linear_cross_entropy(h, w, y, chunk_bytes=0),
            "float64 logits require": lambda: linear_cross_entropy(
                h.double(), w.double(), y, precision="float32"
            ),
            "Unknown backend": lambda: linear_cross_entropy(h, w, y, backend="cuda"),
        }
        for message, call in cases.items():
            with subtests.test(message=message), pytest.raises(ValueError, match=message):
                call()

    def test_module(self):
        h, w, b, y, cw = inputs(6, 4, 9, ignored=2)
        module = LinearCrossEntropyLoss(cw, reduction="sum", chunk_bytes=100)
        assert "class_weight" in module.state_dict()
        expected = linear_cross_entropy(h, w, y, b, class_weight=cw, reduction="sum")
        torch.testing.assert_close(module(h, w, y, b), expected)

    def test_double_backward_raises(self):
        h, w, b, y, _ = inputs(3, 2, 4, torch.float64)
        h.requires_grad_()
        (g,) = torch.autograd.grad(linear_cross_entropy(h, w, y, b), h, create_graph=True)
        with pytest.raises(RuntimeError, match="first derivatives only"):
            g.sum().backward()


@pytest.mark.skipif(
    not (lce.AVAILABLE and (shutil.which("g++") or shutil.which("clang++"))),
    reason="needs torch.library.custom_op and a C++ compiler for Inductor",
)
class TestCompiledLinearCrossEntropy:
    @pytest.mark.slow
    def test_full_graph_matches_eager(self, extension: str, subtests: pytest.Subtests):
        h, w, b, y, cw = inputs(20, 8, 33, ignored=3)
        for reduction, recompute in itertools.product(REDUCTIONS, (False, True)):
            with subtests.test(reduction=reduction, recompute=recompute):

                def f(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    loss = linear_cross_entropy(
                        torch.tanh(h),
                        w,
                        y,
                        b,
                        class_weight=cw,
                        reduction=reduction,
                        chunk_bytes=600,
                        recompute=recompute,
                    )
                    return 2 * loss.sum()

                torch._dynamo.reset()
                explained = torch._dynamo.explain(f)(h, w, b)
                assert explained.graph_break_count == 0, explained.break_reasons
                compiled = torch.compile(f, fullgraph=True)
                close(grads(compiled, h, w, b), grads(f, h, w, b), 1e-6)

    def test_operator_registrations(self, subtests: pytest.Subtests):
        h, w, b, y, cw = inputs(6, 4, 9, ignored=2)
        for (bias, weight), reduction, save in itertools.product(
            ((b, None), (None, cw)), (0, 1), (0, 7)
        ):
            ints = [-100, reduction, lce.NATIVE, 1, 2, 500, save if reduction else 0]
            leaves = (h.requires_grad_(), w.requires_grad_())
            args = (*leaves, None if bias is None else bias.requires_grad_(), y, weight, ints)
            with subtests.test(bias=bias is not None, reduction=reduction, save=save):
                torch.library.opcheck(torch.ops.libsdca.linear_cross_entropy, args)


def triton_check() -> None:
    """The Triton row kernels against the native library's, chunk by chunk (forward and
    backward of linear_cross_entropy's operators); run in the interpreter on CPU tensors."""
    for dtype in (torch.float32, torch.float16, torch.float64):
        h, w, b, y, cw = inputs(9, 4, 300, dtype, ignored=4)
        upstream = torch.linspace(0.5, 2, 9, dtype=torch.float64)
        for reduction in (0, 1, 2):
            for weight in (None, cw):
                results = []
                for kernel in (lce.TRITON, lce.NATIVE):
                    ints = [
                        -100,
                        reduction,
                        kernel,
                        int(dtype != torch.float64),
                        lce._DTYPES.index(dtype),
                        4 * 300 * 4,
                        0,
                    ]
                    loss, divisor, _ = lce._forward(h, w, b, y, weight, ints)
                    up = upstream if reduction == 0 else upstream[0]
                    gradients = lce._backward(
                        h, w, b, y, weight, divisor, up, ints, (True, True, True)
                    )
                    results.append((loss, *gradients))
                    if reduction:
                        # The forward's gradients, times the upstream derivative.
                        _, _, saved = lce._forward(h, w, b, y, weight, ints[:6] + [7])
                        scaled = lce._scaled(saved, up.float(), h, w, b, True)
                        close(scaled, gradients, tol(dtype))
                close(results[0], results[1], tol(dtype))
                print(dtype, reduction, weight is not None, "ok", flush=True)


def tol(dtype: torch.dtype) -> float:
    return 1e-12 if dtype == torch.float64 else 1e-5 if dtype == torch.float32 else 2e-3


class LinearCrossEntropyKernelTests:
    @pytest.mark.slow
    def test_triton_kernels_in_the_interpreter(self):
        """The Triton path's control flow and rounding on CPU tensors (skips without Triton)."""
        out = interpreter.run("import test_linear_cross_entropy as t\nt.triton_check()\n")
        assert out.count("ok") == 18
