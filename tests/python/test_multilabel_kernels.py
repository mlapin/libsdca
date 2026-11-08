"""The fused boolean-mask CUDA kernels (entropy, multilabel hinges) against the native losses
and exact references; the interpreter test runs the same kernels on the CPU."""

import os
from collections.abc import Callable, Iterator
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any, TypedDict
from unittest import mock

import interpreter
import pytest
import torch
from libsdca import _cuda_launch, _multilabel
from libsdca import torch as sdca_torch
from libsdca.torch import explain_backend, sdca_loss
from torch.autograd import gradcheck, gradgradcheck

EPS32 = torch.finfo(torch.float32).eps
# The kernels' kinds: the _smooth kind is multilabel_hinge smoothed by gamma.
SMOOTH_HINGE = "multilabel_hinge" + "_smooth"
OBJECTIVES = ("cross_entropy", "multilabel_hinge", SMOOTH_HINGE)
GAMMAS = (0.6, 1e-3, 30.0, 1e-20, 1e20)
# Rows with ties, common offsets of 1e16, scores 1e30 apart and cancelling margins.
EDGE_ROWS = [
    [1e16, 1e16, 1e16 + 2, 1e16, 1e16 - 2],
    [700.0, 0.0, 1.0, 700.0, -0.3],
    [-1e30, 0.0, 1.0, 2.0, -1.0],
    [1e30, 0.0, 1.0, 2.0, -1.0],
    [0.0, 1.0, 1e16, -0.3, 0.2],
    [0.0, 1e16, 1.0, -1e16, 3.0],
    [0.0, 20.0, 20.0, -100.0, 5.0],
    [1.0, 1e-20, -5.0, 0.3, 1e-20],
    [1.0, 2.0**-26, 0.5, 0.0, 1.0],
]
PATTERNS = [[1, 0, 1, 0, 1], [0, 1, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 0]]


def _public(kind: str, gamma: float = 1.0) -> tuple[str, float]:
    """The loss name and gamma of a kernel kind: gamma smooths only a _smooth kind."""
    return kind.removesuffix("_smooth"), gamma if kind.endswith("_smooth") else 0.0


def gammas_of(objective: str) -> tuple[float, ...]:
    return GAMMAS if objective.endswith("_smooth") else (1.0,)


def random_mask(
    examples: int, classes: int, positives: int, generator: torch.Generator
) -> torch.Tensor:
    """Exactly min(positives, classes - 1) random positives per row."""
    order = torch.rand(examples, classes, generator=generator).argsort(-1)
    return order < min(positives, classes - 1)


def cases(
    generator: torch.Generator, small: bool = False
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """(float64 scores, mask) pairs: random rows, integer ties and the edge rows."""
    shapes = ((4, 3), (4, 7)) if small else ((8, 2), (8, 3), (12, 7), (10, 13), (6, 40))
    for examples, classes in shapes:
        for scale in (1.0,) if small else (1.0, 100.0, 1e-3):
            z = scale * torch.randn(examples, classes, generator=generator, dtype=torch.float64)
            pos_counts = (1, classes // 2) if small else (1, 2, classes // 2, classes - 1)
            for positives in pos_counts:
                yield z, random_mask(examples, classes, positives, generator)
    z = torch.randint(-2, 3, (12 if small else 200, 6), generator=generator).double()
    yield z, random_mask(z.shape[0], 6, 3, generator)
    if small:
        groups = (
            [EDGE_ROWS[0], EDGE_ROWS[4], EDGE_ROWS[5]],
            [EDGE_ROWS[2], EDGE_ROWS[3]],
            [EDGE_ROWS[1], EDGE_ROWS[6]],
            [EDGE_ROWS[7], EDGE_ROWS[8]],
        )
        for group in groups:
            rows = [r for r in group for _ in PATTERNS[:2]]
            masks = PATTERNS[:2] * len(group)
            yield torch.tensor(rows, dtype=torch.float64), torch.tensor(masks, dtype=torch.bool)
    else:
        mask = torch.tensor(PATTERNS, dtype=torch.bool)
        for row in EDGE_ROWS:
            yield torch.tensor([row] * len(PATTERNS), dtype=torch.float64), mask


def exact(
    z: list[float], mask: list[bool], objective: str, gamma: float
) -> tuple[float, list[float]]:
    """Value and gradient of one row from exact rationals (hinges) or 80 digits (entropy)."""
    if objective == "cross_entropy":
        with localcontext() as context:
            context.prec = 80
            zs = [Decimal(v) for v in z]
            top = max(zs)
            e = [(v - top).exp() for v in zs]
            den = sum(e, Decimal(0))
            p = sum(mask)
            positive = sum((v for v, b in zip(zs, mask, strict=True) if b), Decimal(0))
            value = den.ln() + top - positive / p
            gradient = [
                ej / den - (Decimal(1) / p if b else 0) for ej, b in zip(e, mask, strict=True)
            ]
            return float(value), [float(g) for g in gradient]
    zs = [Fraction(v) for v in z]
    pos = [v for v, b in zip(zs, mask, strict=True) if b]
    neg = [v for v, b in zip(zs, mask, strict=True) if not b]
    if objective == "multilabel_hinge":
        value = max(Fraction(0), 1 + max(neg) - min(pos))
        gradient = [0.0] * len(z)
        if value > 0:
            gradient[next(i for i, v in enumerate(zs) if mask[i] and v == min(pos))] = -1.0
            gradient[next(i for i, v in enumerate(zs) if not mask[i] and v == max(neg))] = 1.0
        return float(value), gradient
    g = Fraction(gamma)
    z0 = pos[0]
    a = [Fraction(1, 2) - (v - z0) for v in pos]
    b = [Fraction(1, 2) + (v - z0) for v in neg]
    x, y = _two_simplex(a, b, g)
    value = sum(h * v - v * v / 2 for h, v in zip(a + b, x + y, strict=True)) / g
    xs, ys = iter(x), iter(y)
    return float(value), [float(-next(xs) / g if m else next(ys) / g) for m in mask]


def exact_rows(
    z: torch.Tensor, mask: torch.Tensor, objective: str, gamma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """exact() of every row, as float64 values and gradients."""
    rows = [exact(r, m, objective, gamma) for r, m in zip(z.tolist(), mask.tolist(), strict=True)]
    values = torch.tensor([v for v, _ in rows], dtype=torch.float64)
    return values, torch.tensor([g for _, g in rows], dtype=torch.float64)


def _simplex(a: list[Fraction], rhs: Fraction) -> Fraction:
    """Threshold t of the projection max(0, a - t) onto sum = rhs, by sorting."""
    s = sorted(a, reverse=True)
    total = Fraction(0)
    for i, v in enumerate(s):
        total += v
        t = (total - rhs) / (i + 1)
        if i + 1 == len(s) or s[i + 1] <= t:
            return t
    raise ValueError("empty vector")


def _two_simplex(
    a: list[Fraction], b: list[Fraction], rhs: Fraction
) -> tuple[list[Fraction], list[Fraction]]:
    """Projection onto x, y >= 0 with sum(x) = sum(y) <= rhs (KKT, exact breakpoints)."""
    zero = Fraction(0)

    def f(t: Fraction) -> Fraction:
        return sum((max(v - t, zero) for v in a), zero) - sum((max(v + t, zero) for v in b), zero)

    points = sorted(set(a) | {-v for v in b})
    values = [f(t) for t in points]
    if values[0] <= 0:
        # Left of every breakpoint only the positives are active.
        t = points[0] + values[0] / len(a)
    elif values[-1] >= 0:
        t = points[-1] + values[-1] / len(b)
    else:
        i = next(i for i in range(len(points) - 1) if values[i] >= 0 >= values[i + 1])
        lo, hi = points[i], points[i + 1]
        t = lo + values[i] / (values[i] - values[i + 1]) * (hi - lo)
    x = [max(v - t, zero) for v in a]
    if sum(x) <= rhs:
        return x, [max(v + t, zero) for v in b]
    ta, tb = _simplex(a, rhs), _simplex(b, rhs)
    return [max(v - ta, zero) for v in a], [max(v - tb, zero) for v in b]


def native(
    z: torch.Tensor,
    mask: torch.Tensor,
    objective: str,
    gamma: float,
    reduction: str,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU float64 native values and weighted gradients."""
    x = z.detach().cpu().double().requires_grad_()
    name, gamma = _public(objective, gamma)
    value = sdca_loss(x, mask.cpu(), name, backend="native", gamma=gamma, reduction=reduction)
    value.backward(weights.cpu().double() if reduction == "none" else torch.tensor(0.75).double())
    assert x.grad is not None
    return value.detach(), x.grad


class Tolerance(TypedDict):
    rtol: float
    atol: float


def tolerance(z: torch.Tensor, objective: str, gamma: float, rtol: float = 1e-12) -> Tolerance:
    """As test_tensor_losses: rtol, and rtol relative to the score scale over gamma."""
    scale = max(1.0, float(z.abs().max()))
    return Tolerance(rtol=rtol, atol=rtol * scale / (gamma if objective.endswith("_smooth") else 1))


def check_small_losses(entropy: Callable[[torch.Tensor, torch.Tensor, bool], Any]) -> None:
    """entropy(z, mask, fp32) -> unreduced values, gradient: float32 exponentials keep small
    float32 losses within 8 float eps (gradients: of the row's largest entry), float64 within 1."""
    # A tail of exp(-69) scaled from the runner-up.
    z = torch.tensor(
        [[40.1234567, -29.3456789, -29.7654321, -30.1111111], [2.5, 2.25, 3.125, -0.5]]
    )
    mask = torch.tensor([[True, False, False, False], [True, True, False, False]])
    expected = native(z.double(), mask, "cross_entropy", 1.0, "none", torch.ones(2))
    for fp32, rtol in [(True, 8 * EPS32), (False, EPS32)]:
        value, gradient = entropy(z, mask, fp32)
        torch.testing.assert_close(value.double(), expected[0], rtol=rtol, atol=0)
        big = expected[1].abs().amax(-1, keepdim=True)
        assert float(((gradient.double() - expected[1]).abs() / big).max()) <= rtol, fp32


def interpreter_check(small: bool = True, chunks: tuple[int, ...] = (16, 4)) -> None:
    """Kernels against native and exact rows, with whole and 4-column chunks (CPU tensors)."""
    from libsdca import _triton_multilabel as kernels
    from libsdca import _triton_reduce as reduce

    reduce.BLOCK, reduce.DIRECT_ROWS, reduce.DIRECT_BLOCK = 16, 64, 8
    kernels._BACKWARD = 8
    generator = torch.Generator().manual_seed(5)
    for chunk in (16, 4):
        kernels._CHUNK = chunk
        for z, mask in cases(generator, small):
            if chunk not in chunks:
                continue
            weights = torch.linspace(0.5, 2.0, z.shape[0], dtype=torch.float64)
            for objective in OBJECTIVES:
                if objective != "cross_entropy" and bool(mask.all(-1).any()):
                    continue
                gammas = gammas_of(objective)[:2] if small else gammas_of(objective)
                for gamma in gammas:
                    for reduction in ("none", "mean"):
                        out, stats = kernels.forward(z, mask, objective, reduction, gamma)
                        u = weights if reduction == "none" else torch.tensor(0.75).double()
                        grad = kernels.backward(z, mask, stats, u, objective, reduction, gamma)
                        expected = native(z, mask, objective, gamma, reduction, weights)
                        tol = tolerance(z, objective, gamma)
                        message = f"{objective} {gamma} {reduction} chunk {chunk} {z} {mask}"
                        torch.testing.assert_close(out, expected[0], **tol, msg=message)
                        torch.testing.assert_close(grad, expected[1], **tol, msg=message)

    def entropy(z: torch.Tensor, mask: torch.Tensor, fp32: bool) -> Any:
        out, stats = kernels.forward(z, mask, "cross_entropy", "none", 1.0, fp32)
        u = torch.ones(z.shape[0], dtype=out.dtype)
        return out, kernels.backward(z, mask, stats, u, "cross_entropy", "none", 1.0, fp32)

    check_small_losses(entropy)


def _kernel_available() -> bool:
    return torch.cuda.is_available() and _multilabel._kernel() is not None


@pytest.mark.skipif(not _kernel_available(), reason="needs CUDA and Triton 3.3")
class FusedMultilabelTests:
    def setup_method(self):
        self.generator = torch.Generator().manual_seed(17)

    def fused(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        objective: str,
        gamma: float,
        reduction: str,
        weights: torch.Tensor,
        **options: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = z.detach().cuda().requires_grad_()
        name, gamma = _public(objective, gamma)
        value = sdca_loss(x, mask.cuda(), name, gamma=gamma, reduction=reduction, **options)
        upstream = weights if reduction == "none" else torch.tensor(0.75, dtype=value.dtype)
        value.backward(upstream.to(value.dtype).cuda())
        assert x.grad is not None
        return value.detach().cpu(), x.grad.cpu()

    def test_backend(self):
        z = torch.zeros(3, 4, device="cuda")
        mask = torch.eye(3, 4, dtype=torch.bool, device="cuda")
        for objective in OBJECTIVES + ("multilabel_cross_entropy",):
            name, gamma = _public(objective)
            assert explain_backend(z, mask, name, gamma=gamma)[0] in ("cuda-launcher", "triton")
            assert explain_backend(z, mask, name, gamma=gamma, precision="float32")[0] == "torch"

    def test_float64_matches_native_and_exact(self):
        for z, mask in cases(self.generator):
            weights = torch.linspace(0.5, 2.0, z.shape[0], dtype=torch.float64)
            for objective in OBJECTIVES:
                if objective != "cross_entropy" and bool(mask.all(-1).any()):
                    continue
                for gamma in gammas_of(objective):
                    tol = tolerance(z, objective, gamma)
                    for reduction in ("none", "mean", "sum"):
                        actual = self.fused(z, mask, objective, gamma, reduction, weights)
                        expected = native(z, mask, objective, gamma, reduction, weights)
                        message = f"{objective} {gamma} {reduction} {z} {mask}"
                        for a, e in zip(actual, expected, strict=True):
                            torch.testing.assert_close(a, e, **tol, msg=message)
                    if z.shape[0] > 12:
                        continue
                    value, gradient = self.fused(
                        z, mask, objective, gamma, "none", torch.ones(z.shape[0]).double()
                    )
                    expected = exact_rows(z, mask, objective, gamma)
                    message = f"exact {objective} {gamma} {z} {mask}"
                    for a, e in zip((value, gradient), expected, strict=True):
                        torch.testing.assert_close(a, e, **tol, msg=message)

    def test_all_positive_entropy_and_invalid_masks(self):
        z = torch.tensor([[0.0, 1e-10, -1e-10], [20.0, 20.0, -20.0]], dtype=torch.float64)
        full = torch.ones_like(z, dtype=torch.bool)
        weights = torch.ones(2, dtype=torch.float64)
        actual = self.fused(z, full, "cross_entropy", 1.0, "none", weights)
        for a, e in zip(
            actual, native(z, full, "cross_entropy", 1.0, "none", weights), strict=True
        ):
            torch.testing.assert_close(a, e, rtol=1e-14, atol=1e-25)
        # The positives' differences of 1e-10 leave a second-order -1e-20 / 9 at the first.
        expected = exact_rows(z, full, "cross_entropy", 1.0)
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-12, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-30)
        empty = torch.zeros_like(full)
        nan = z.clone()
        nan[1, 2] = float("nan")
        for objective in OBJECTIVES:
            assert bool(
                sdca_loss(
                    z.cuda(), empty.cuda(), _public(objective)[0], gamma=_public(objective)[1]
                )
                .isnan()
                .all()
            )
            assert bool(
                sdca_loss(
                    nan.cuda(), full.cuda(), _public(objective)[0], gamma=_public(objective)[1]
                )
                .isnan()
                .all()
            )
            if objective != "cross_entropy":
                assert bool(
                    sdca_loss(
                        z.cuda(),
                        full.cuda(),
                        _public(objective)[0],
                        gamma=_public(objective)[1],
                    )
                    .isnan()
                    .all()
                )

    def test_dtypes(self):
        z = 3 * torch.randn(64, 37, generator=self.generator)
        mask = random_mask(64, 37, 5, self.generator)
        weights = torch.linspace(0.5, 2.0, 64, dtype=torch.float64)
        for dtype, rtol in [
            (torch.float32, 3e-7),
            (torch.float16, 2e-3),
            (torch.bfloat16, 1.6e-2),
        ]:
            x = z.to(dtype)
            for objective in OBJECTIVES:
                for gamma in gammas_of(objective)[:3]:
                    for reduction in ("none", "mean"):
                        # The float64 result of the represented logits, rounded once.
                        value, gradient = self.fused(x, mask, objective, gamma, reduction, weights)
                        expected = native(x.double(), mask, objective, gamma, reduction, weights)
                        out = torch.promote_types(dtype, torch.float32)
                        assert value.dtype == out
                        assert gradient.dtype == dtype
                        tol = tolerance(z, objective, gamma, rtol)
                        message = f"{dtype} {objective} {gamma} {reduction}"
                        torch.testing.assert_close(value.double(), expected[0], **tol, msg=message)
                        torch.testing.assert_close(
                            gradient.double(), expected[1], **tol, msg=message
                        )

    def test_float32_exponentials_keep_small_losses(self):
        # The default of float32 logits; precision="float64" keeps float64.
        ones = torch.ones(2, dtype=torch.float64)
        check_small_losses(
            lambda z, mask, fp32: self.fused(
                z, mask, "cross_entropy", 1.0, "none", ones, precision=None if fp32 else "float64"
            )
        )

    def test_large_batches_and_wide_rows(self):
        for examples, classes in [(5000, 7), (4, 5000), (3, 70000)]:
            z = torch.randn(examples, classes, generator=self.generator, dtype=torch.float64)
            mask = random_mask(examples, classes, 3, self.generator)
            weights = torch.rand(examples, generator=self.generator, dtype=torch.float64)
            for objective in OBJECTIVES:
                for gamma in gammas_of(objective)[:3]:
                    for reduction in ("none", "mean", "sum"):
                        actual = self.fused(z, mask, objective, gamma, reduction, weights)
                        expected = native(z, mask, objective, gamma, reduction, weights)
                        message = f"{examples}x{classes} {objective} {gamma} {reduction}"
                        for a, e in zip(actual, expected, strict=True):
                            torch.testing.assert_close(
                                a, e, **tolerance(z, objective, gamma, 1e-11), msg=message
                            )

    def test_layout_and_reductions(self):
        z = torch.randn(2, 5, 3, generator=self.generator, dtype=torch.float64)
        mask = random_mask(6, 5, 2, self.generator).reshape(2, 3, 5).transpose(1, 2)
        for objective in OBJECTIVES:
            x = z.cuda().requires_grad_()
            values = sdca_loss(
                x,
                mask.cuda(),
                _public(objective)[0],
                gamma=_public(objective)[1],
                dim=1,
                reduction="none",
            )
            assert values.shape == (2, 3)
            values.sum().backward()
            flat = z.movedim(1, -1).reshape(-1, 5)
            expected = native(
                flat,
                mask.movedim(1, -1).reshape(-1, 5),
                objective,
                1.0,
                "none",
                torch.ones(6).double(),
            )
            torch.testing.assert_close(values.detach().cpu().reshape(-1), expected[0])
            assert x.grad is not None
            torch.testing.assert_close(x.grad.cpu().movedim(1, -1).reshape(-1, 5), expected[1])
            for reduction, reduce in [("sum", torch.sum), ("mean", torch.mean)]:
                total = sdca_loss(
                    x,
                    mask.cuda(),
                    _public(objective)[0],
                    gamma=_public(objective)[1],
                    dim=1,
                    reduction=reduction,
                )
                torch.testing.assert_close(total, reduce(values.detach()), rtol=1e-14, atol=0)

    def test_derivatives(self):
        z = torch.randn(3, 5, device="cuda", dtype=torch.float64, requires_grad=True)
        mask = random_mask(3, 5, 2, self.generator).cuda()
        for reduction in ("none", "mean"):

            def entropy(x: torch.Tensor, reduction: str = reduction) -> torch.Tensor:
                return sdca_loss(x, mask, "multilabel_cross_entropy", reduction=reduction)

            assert gradcheck(entropy, (z,))
            assert gradgradcheck(entropy, (z,))
        for objective in OBJECTIVES[1:]:
            gradient = torch.autograd.grad(
                sdca_loss(z, mask, _public(objective)[0], gamma=_public(objective)[1]),
                z,
                create_graph=True,
            )[0]
            for use in (gradient.sum(), (gradient * z).sum()):
                with pytest.raises(RuntimeError, match="first derivatives only"):
                    torch.autograd.grad(use, z, retain_graph=True)

    @pytest.mark.parametrize("setting", ["0", "1"])
    def test_graph_gradient_at_large_gaps(self, setting: str):
        """The gradient with create_graph=True, through the C++ launcher where it is active, at
        gaps beyond float64's exponential range: a finite double backward at 710, and at 750 the
        tail that only a large upstream makes representable, as the tensor backend."""
        mask = torch.tensor([[True, False]], device="cuda")
        with mock.patch.dict(os.environ, {_cuda_launch.ENV: setting}):
            _cuda_launch.status.cache_clear()
            try:
                for gap, upstream in [(710.0, 1.0), (750.0, 1e100)]:
                    z = torch.tensor([[0.0, -gap]], device="cuda", dtype=torch.float64)
                    grads = []
                    for backend in ("auto", "torch"):
                        x = z.clone().requires_grad_()
                        loss = sdca_loss(x, mask, "multilabel_cross_entropy", backend=backend)
                        (g,) = torch.autograd.grad(
                            loss, x, loss.new_tensor(upstream), create_graph=True
                        )
                        (h,) = torch.autograd.grad(g.sum() + (g * x).sum(), x)
                        assert bool(h.isfinite().all())
                        grads.append(g.detach())
                    torch.testing.assert_close(grads[0], grads[1], rtol=1e-14, atol=0)
                expected = torch.tensor(
                    [[-1.9016849634750066e-226, 1.9016849634750066e-226]], dtype=torch.float64
                )
                torch.testing.assert_close(grads[0].cpu(), expected, rtol=1e-12, atol=0)
            finally:
                _cuda_launch.status.cache_clear()


@pytest.mark.skipif(
    not (os.environ.get("SDCA_LARGE_TESTS") == "1" and _kernel_available()),
    reason="needs SDCA_LARGE_TESTS=1 and CUDA with about 45 GB free",
)
class RowLimitTests:
    def test_smooth_hinge_at_the_row_limit(self):
        """ROWS rows, where 8 * N and STATUS * N = 9 * N exceed int32, against row chunks."""
        n, chunk, loss = _cuda_launch.ROWS, 2**24, "multilabel_hinge"
        gamma = 1.0
        generator = torch.Generator(device="cuda").manual_seed(0)
        z = torch.randn(n, 2, generator=generator, device="cuda", dtype=torch.float16)
        first = torch.rand(n, generator=generator, device="cuda") < 0.5
        mask = torch.stack([first, ~first], 1)
        mask[-1] = False  # One invalid row: a NaN loss.
        for setting in ("0", "1"):
            with mock.patch.dict(os.environ, {_cuda_launch.ENV: setting}):
                _cuda_launch.status.cache_clear()
                try:
                    expected = torch.cat(
                        [
                            sdca_loss(
                                z[s : s + chunk],
                                mask[s : s + chunk],
                                loss,
                                reduction="none",
                                gamma=gamma,
                            )
                            for s in range(0, n, chunk)
                        ]
                    )
                    x = z.detach().requires_grad_()
                    value = sdca_loss(x, mask, loss, reduction="none", gamma=gamma)
                    torch.testing.assert_close(
                        value.detach(), expected, rtol=0, atol=0, equal_nan=True
                    )
                    value.backward(torch.ones_like(value))
                    tail = z[-chunk:].detach().requires_grad_()
                    sdca_loss(
                        tail, mask[-chunk:], loss, reduction="none", gamma=gamma
                    ).sum().backward()
                    assert x.grad is not None
                    torch.testing.assert_close(
                        x.grad[-chunk:], tail.grad, rtol=0, atol=0, equal_nan=True
                    )
                    del value, x
                    mean = sdca_loss(z[:-1], mask[:-1], loss, gamma=gamma).double()
                    torch.testing.assert_close(
                        mean, expected[:-1].double().mean(), rtol=1e-3, atol=0
                    )
                finally:
                    _cuda_launch.status.cache_clear()


class InterpreterTests:
    @pytest.mark.slow
    @pytest.mark.parametrize("chunk", [16, 4])
    def test_kernels_in_the_interpreter(self, chunk: int):
        """The kernels' control flow on CPU tensors, with numpy in place of libdevice."""
        interpreter.run(
            f"import test_multilabel_kernels as t\nt.interpreter_check(chunks=({chunk},))\n"
        )


def test_dispatch_takes_float32_exponentials_by_default():
    """The CUDA dispatch (on CPU tensors): mask entropy of non-float64 logits takes float32
    exponentials unless precision="float64"; the hinges never do."""
    seen: list[bool] = []

    def loss(flat: torch.Tensor, mask: torch.Tensor, objective: str, *args: Any) -> torch.Tensor:
        seen.append(_multilabel.fp32(objective, flat, args[-1]))
        return flat.new_zeros((), dtype=torch.float32)

    mask = torch.eye(2, 3, dtype=torch.bool)
    with (
        mock.patch.object(_multilabel, "applies", lambda *args: True),
        mock.patch.object(_multilabel, "loss", loss),
    ):
        for objective in ("cross_entropy", "multilabel_hinge"):
            for dtype, precision in [
                (torch.bfloat16, None),
                (torch.float32, None),
                (torch.float32, "float64"),
                (torch.float64, None),
            ]:
                z = torch.zeros(2, 3, dtype=dtype)
                args = (objective, 1, 0.0, 1.0, "mean", -1, precision, "auto")
                sdca_torch._device_loss(z, z, mask, *args)
    assert seen == [True, True, False, False] + [False] * 4
