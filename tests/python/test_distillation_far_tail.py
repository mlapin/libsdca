"""distillation_loss and proximal_distillation_loss in float32 work on rows whose competitors
sit far below the leading logits: the fused CUDA kernels take such rows' forward sums again
from shifted exponentials (_far_row), and the proximal kernels take them again in float64
work (_prox_far_row). Against the same kernels in float64 work on the same float32 logits,
on CUDA and in Triton's CPU interpreter on every route (register tiles, rows in registers,
streamed rows)."""

from collections.abc import Callable
from typing import Any

import interpreter
import pytest
import torch
from libsdca import torch as sdca_torch
from libsdca.torch import distillation_loss, proximal_distillation_loss

EPS32 = torch.finfo(torch.float32).eps
TINY = 2.0**-149
# Competitor logits of the student and the teacher beside leading logits at 0.
CASES = (
    (-110.0, -220.0),  # zero float32 terms on both sides; a subnormal target gradient
    (-110.0, -111.0),  # a loss below the float32 normal range
    (-110.0, -100.0),  # the teacher nearer
    (-100.0, -110.0),  # the student nearer
    (-90.0, -180.0),
    (-90.0, -91.0),  # far terms of nearly equal student and teacher
    (-10.0, -12.0),  # no far term
)

Fused = Callable[[torch.Tensor, torch.Tensor, float, bool], tuple[torch.Tensor, torch.Tensor]]


# far_tail_rows that take far sums: every CASES row with a far term, and the mixed row, whose
# far terms are negligible: loss-only calls of streamed rows, which bound its loss by the
# row's spread, leave it out.
FAR_ROWS = [*range(len(CASES) - 1), len(CASES)]
MIXED = len(CASES)


def far_tail_rows(classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The CASES rows; a row of near and far competitors in one sum (alternating -70 and
    -110 in the student, -75 and -220 in the teacher); and an ordinary row with one far
    entry (-100 in both), whose loss is far above what its far term adds."""
    s = torch.tensor([[c[0]] for c in CASES] + [[-110.0], [0.0]]).expand(-1, classes).clone()
    t = torch.tensor([[c[1]] for c in CASES] + [[-220.0], [0.0]]).expand(-1, classes).clone()
    s[-2, 1::2], t[-2, 1::2] = -70.0, -75.0
    j = torch.arange(classes, dtype=torch.float32)
    s[-1], t[-1] = -(j % 5), -0.5 * (j % 3)
    s[-1, 1] = t[-1, 1] = -100.0
    s[:, 0] = t[:, 0] = 0.0
    return s, t


def check_far_tail(fused: Fused, classes: int, temperature: float) -> None:
    """fused(s, t, temperature, double) -> unreduced values and gradient: float32 work within
    64 float32 eps (and 2^-149 per entry, 4 * 2^-149 per value) of float64 work. The rows are
    scaled by the temperature, so every temperature has the same centered, scaled logits."""
    s, t = (x * temperature for x in far_tail_rows(classes))
    expected = fused(s, t, temperature, True)
    for name, actual, exact, tiny in zip(
        ("loss", "gradient"),
        fused(s, t, temperature, False),
        expected,
        (4 * TINY, TINY),
        strict=True,
    ):
        exact = exact.double().cpu()
        error = (actual.double().cpu() - exact).abs()
        bad = error > 64 * EPS32 * exact.abs() + tiny
        rows = bad.reshape(s.shape[0], -1).any(-1).nonzero().flatten().tolist()
        assert not rows, f"{name} C={classes} T={temperature}: rows {rows}"


def interpreter_far_tail_check() -> None:
    """check_far_tail (CPU tensors) of the Triton kernels in a register tile of short rows
    (C <= 16 and C > 16), a row in registers and streamed rows, also value-only."""
    from libsdca import _triton_distillation as kernels
    from libsdca import _triton_reduce as reduce

    reduce.BLOCK, reduce.DIRECT_ROWS, reduce.DIRECT_BLOCK = 16, 64, 8
    far_row, recomputed = kernels._far_row, []

    def counted_far_row(*args: Any):
        recomputed.append(int(args[5].handle.data.item()))
        return far_row(*args)

    kernels._far_row = counted_far_row

    def fused(
        s: torch.Tensor, t: torch.Tensor, temperature: float, double: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value, stats = kernels.forward(s, t, temperature, double, "none")
        only, _ = kernels.forward(s, t, temperature, double, "none", value_only=True)
        torch.testing.assert_close(only, value, rtol=4 * EPS32, atol=4 * TINY)
        weights = torch.ones(s.shape[0])
        return value, kernels.backward(s, t, stats, weights, temperature, double, "none")

    for chunk, classes, rows in ((4096, 16, 16), (4096, 40, 4), (4096, 300, 1), (0, 40, 1)):
        kernels._CHUNK = chunk
        kernels._STREAM, kernels._BACKWARD = (16, 1), (16, 1)
        assert kernels._rows_per_cta(classes) == rows
        assert (kernels._shapes(classes)[0] == kernels._STREAM) == (chunk == 0)
        for temperature in (1.0, 2.0):
            recomputed.clear()
            check_far_tail(fused, classes, temperature)
            # Two forwards (also value-only) in each of the two works; float64 takes none.
            expected = 2 * FAR_ROWS
            if chunk == 0:
                expected = FAR_ROWS + [r for r in FAR_ROWS if r != MIXED]
            assert recomputed == expected, (chunk, classes, temperature, recomputed)


def interpreter_prox_far_tail_check() -> None:
    """check_far_tail (CPU tensors) of the proximal kernels, a row in registers and streamed
    rows, at alpha 0.1 and 10: only the rows of FAR_ROWS take float64 work again."""
    from libsdca import _triton_distillation as kernels

    recomputed: list[int] = []
    for name in ("_prox_far_row", "_prox_stream_far_row"):
        rescan = getattr(kernels, name)

        def counted(*args: Any, rescan: Any = rescan) -> Any:
            recomputed.append(int(args[5].handle.data.item()))
            return rescan(*args)

        setattr(kernels, name, counted)
    kernels._STREAM_BLOCK = 16
    for chunk in (4096, 0):
        kernels._CHUNK = chunk
        for alpha in (0.1, 10.0):

            def fused(
                s: torch.Tensor, t: torch.Tensor, temperature: float, double: bool
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return kernels.proximal_distillation(s, t, temperature, alpha, double)

            for temperature in (1.0, 2.0):
                recomputed.clear()
                check_far_tail(fused, 40, temperature)
                assert recomputed == FAR_ROWS, (chunk, alpha, temperature, recomputed)


def _cuda_fused(
    s: torch.Tensor, t: torch.Tensor, temperature: float, double: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unreduced values and the gradient of their sum through the public loss on CUDA."""
    precision = "float64" if double else None
    x, y = s.cuda().requires_grad_(), t.cuda()
    value = distillation_loss(x, y, temperature=temperature, reduction="none", precision=precision)
    (gradient,) = torch.autograd.grad(value.sum(), x)
    with torch.no_grad():
        only = distillation_loss(
            x, y, temperature=temperature, reduction="none", precision=precision
        )
    torch.testing.assert_close(only, value.detach(), rtol=4 * EPS32, atol=4 * TINY)
    return value.detach(), gradient


@pytest.mark.skipif(
    not torch.cuda.is_available() or sdca_torch._kernel_distill() is None,
    reason="needs Triton and an NVIDIA GPU",
)
@pytest.mark.parametrize("classes", [16, 40, 300, 4096, 4097, 32768])
@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_float32_far_tail_rows(classes: int, temperature: float):
    check_far_tail(_cuda_fused, classes, temperature)


def _cuda_proximal(
    alpha: float,
) -> Fused:
    def fused(
        s: torch.Tensor, t: torch.Tensor, temperature: float, double: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        precision = "float64" if double else None
        x = s.cuda().requires_grad_()
        value = proximal_distillation_loss(
            x, t.cuda(), temperature=temperature, alpha=alpha, reduction="none", precision=precision
        )
        (gradient,) = torch.autograd.grad(value.sum(), x)
        return value.detach(), gradient

    return fused


@pytest.mark.skipif(
    not torch.cuda.is_available() or sdca_torch._kernel_distill(True) is None,
    reason="needs Triton and an NVIDIA GPU",
)
@pytest.mark.parametrize("classes", [16, 300, 4096, 4097, 32768])
@pytest.mark.parametrize("temperature", [1.0, 2.0])
@pytest.mark.parametrize("alpha", [0.1, 10.0])
def test_float32_far_tail_proximal_rows(classes: int, temperature: float, alpha: float):
    check_far_tail(_cuda_proximal(alpha), classes, temperature)


@pytest.mark.slow
def test_far_tail_in_the_interpreter():
    """Far-tail rows on CPU tensors on every route of the kernels."""
    interpreter.run("import test_distillation_far_tail as t\nt.interpreter_far_tail_check()\n")


@pytest.mark.long
def test_proximal_far_tail_in_the_interpreter():
    interpreter.run("import test_distillation_far_tail as t\nt.interpreter_prox_far_tail_check()\n")
