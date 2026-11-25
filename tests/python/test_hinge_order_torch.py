"""Fused and tensor PyTorch top-1 hinges preserve the ordering of represented scores."""

from typing import Any, Literal

import pytest
import torch
from hinge_order_cases import rows
from named_losses import torch_loss

SETTINGS = [
    (dtype, precision)
    for dtype in [torch.float32, torch.float64, torch.bfloat16]
    for precision in (None, "float32", "float64")
    if not (dtype == torch.float64 and precision == "float32")
]


def params() -> list[Any]:
    out = []
    for backend in ("auto", "kernel", "composed"):
        for power in (-29, -60):
            for i, (dtype, precision) in enumerate(SETTINGS):
                key = f"{backend}-dtype{i}-{precision}-{power}"
                out.append(pytest.param(backend, dtype, precision, power, id=key))
    return out


@pytest.mark.parametrize("classes", [3, 17, 1000])
@pytest.mark.parametrize("backend,dtype,precision,power", params())
def test_distinct_hinge_scores(
    classes: int, power: int, dtype: torch.dtype, precision: str | None, backend: str
):
    z, labels, expected = rows(classes, power)
    x = torch.tensor(z, dtype=dtype, requires_grad=True)
    for loss in ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta"):
        value = torch_loss(x, torch.tensor(labels), loss, precision=precision, backend=backend)
        gradient = torch.autograd.grad(value, x)[0]
        torch.testing.assert_close(
            gradient, torch.tensor(expected / len(labels), dtype=dtype), rtol=0, atol=0
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("backend", ["kernel", "composed"])
def test_hinge_ties_across_cuda_blocks(backend: str):
    # Cutoffs and multilabel extrema straddle several tiles.
    z = torch.full((1, 4099), -3.0, dtype=torch.float64, device="cuda")
    z[0, [1, 2048, 4097]] = 0
    z[0, 1024] = 1
    y = torch.tensor([1024], device=z.device)
    expected = torch.zeros_like(z)
    expected[0, [1, 2048, 4097]] = 0.25
    expected[0, 1024] = -0.75

    def check(name: str, t: torch.Tensor, loss_value: float, **kwargs: int) -> None:
        x = z.clone().requires_grad_()
        value = torch_loss(x, t, name, reduction="sum", backend=backend, **kwargs)
        assert value.item() == loss_value
        torch.testing.assert_close(torch.autograd.grad(value, x)[0], expected)

    check("multiclass_hinge", y, 0.0)
    z[0, 3255] = 2
    expected[0, [1, 2048, 4097]] = 1 / 6
    expected[0, 3255], expected[0, 1024] = 0.5, -1
    check("topk_hinge_alpha", y, 1.0, k=2)
    expected[0, [1, 2048, 4097]] = 1 / 12
    expected[0, 1024] = -0.75
    check("topk_hinge_beta", y, 1.0, k=2)

    pos = torch.zeros_like(z, dtype=torch.bool)
    pos[0, [1, 3255]] = True
    z[0, [1, 3255]] = 1
    z[0, 1024] = 0
    expected.zero_()
    expected[0, [1, 3255]] = -0.25
    expected[0, [1024, 2048, 4097]] = 1 / 6
    check("multilabel_hinge", pos, 0.0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_small_topk_mean_margin(gamma: float):
    # Each of the two selected margins is exactly 2**-60. Subtracting the label
    # before adding the offset rounds both to zero in double precision.
    tiny = 2.0**-60
    x = torch.tensor([[1, tiny, tiny, -3]], dtype=torch.float64, device="cuda", requires_grad=True)
    value = torch_loss(x, torch.tensor([0], device=x.device), "topk_hinge_alpha", k=2, gamma=gamma)
    (gradient,) = torch.autograd.grad(value, x)
    expected = tiny if gamma == 0 else tiny**2
    step = 0.5 if gamma == 0 else tiny
    assert value.item() == expected
    torch.testing.assert_close(gradient, x.new_tensor([[-2 * step, step, step, 0]]), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("classes", [10, 513, 1000])
def test_register_smooth_hinge_ranks_before_subtracting_label(classes: int) -> None:
    from libsdca.torch import multiclass_hinge_loss

    z = torch.full((2, classes), -128.0, dtype=torch.bfloat16, device="cuda")
    z[:, 0] = -(2.0**80)
    z[:, 1] = 0
    z[:, -1] = 0.25
    z[1, 1] = -torch.inf
    x = z.detach().clone().requires_grad_()
    y = torch.zeros(2, dtype=torch.long, device=z.device)
    seed = torch.tensor([-2.0, 0.5], device=z.device)
    values = multiclass_hinge_loss(x, y, gamma=0.5, reduction="none", backend="kernel")
    (grad,) = torch.autograd.grad(values, x, seed)
    expected = torch.zeros_like(x)
    expected[0, 0], expected[0, 1], expected[0, -1] = 2, -0.5, -1.5
    expected[1, 0], expected[1, -1] = -0.5, 0.5
    torch.testing.assert_close(values, torch.full_like(values, 2.0**80), atol=0, rtol=0)
    torch.testing.assert_close(grad, expected, atol=0, rtol=0)
    torch.testing.assert_close(x.detach(), z, atol=0, rtol=0)


@pytest.mark.parametrize(("classes", "gamma"), [(10, 0.5), (16, 3.0)])
def test_short_smooth_signed_weighted_replay(classes: int, gamma: float) -> None:
    from hinge_order_cases import short_smooth_rows
    from libsdca.torch import multiclass_hinge_loss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = 65537 if device == "cuda" else 19
    data = short_smooth_rows(classes, gamma, size)
    z, y, weight, seed, expected, derivative = [torch.as_tensor(a, device=device) for a in data]
    x = z.clone().requires_grad_()
    values = multiclass_hinge_loss(
        x, y, weight=weight, gamma=gamma, ignore_index=-100, reduction="none"
    )
    (gradient,) = torch.autograd.grad(values, x, seed)
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
    torch.testing.assert_close(gradient, derivative, rtol=0, atol=0)
    torch.testing.assert_close(x.detach(), z, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("variant", ["alpha", "beta"])
def test_streamed_topk_cutoff_multiplicity(device: str, variant: Literal["alpha", "beta"]) -> None:
    from libsdca.torch import topk_hinge_loss

    classes = 1541
    if device == "cuda":
        from libsdca.triton import common, topk

        kind = common.TOPK_HINGE_ALPHA if variant == "alpha" else common.TOPK_HINGE_BETA
        call = common.Call(kind, 2, classes, "float32", "int64", k=3, saved=common.SAVE, grad=False)
        assert dict(topk.plan(call).launches[0].meta)["B"] < classes
    scores = torch.full((2, classes), -4.0, device=device)
    scores[:, 0] = 0
    # A later tile raises the cutoff to 2: occurrences from the old retained
    # slots still count. Further ties need no selection slot but must share.
    ties = [12, 513, 514, 1026, 1200, 1539]
    scores[0, ties] = 2
    scores[0, 1538] = 5
    # A full set of cutoff ties, with negative and zero margins.
    scores[1, 1:] = -1
    z = scores.clone().requires_grad_()
    y = torch.zeros(2, device=device, dtype=torch.long)
    u = torch.tensor([-3.0, 2.0], device=device)
    v = topk_hinge_loss(z, y, k=3, variant=variant, reduction="none", backend="kernel")
    (g,) = torch.autograd.grad(v, z, u)
    expected = torch.zeros_like(g)
    expected[0, 0], expected[0, 1538] = 3.0, -1.0
    expected[0, ties] = -1.0 / 3.0
    expected[1, 0], expected[1, 1:] = -1.0, 1.0 / 1540
    torch.testing.assert_close(v, v.new_tensor([4.0, 0.0]), rtol=0, atol=0)
    torch.testing.assert_close(g, expected, rtol=0, atol=0)
    torch.testing.assert_close(z.detach(), scores, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_hinge_compile_reuses_graph_for_variable_batches() -> None:
    from libsdca.torch import multiclass_hinge_loss
    from torch._dynamo.testing import CompileCounterWithBackend

    def loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return multiclass_hinge_loss(scores, labels, reduction="mean", backend="kernel")

    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(loss, backend=counter, fullgraph=True, dynamic=None)
    counts = []
    for n in (9, 13, 17, 7):
        scores = torch.randn(n, 513, device="cuda", requires_grad=True)
        labels = torch.arange(n, device="cuda") % 513
        seed = scores.new_tensor(-1.75)
        actual, expected = compiled(scores, labels), loss(scores, labels)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        (gradient,) = torch.autograd.grad(actual, scores, seed)
        (reference,) = torch.autograd.grad(expected, scores, seed)
        torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
        counts.append(counter.frame_count)
    assert 1 <= counts[1] <= 2
    assert counts[2:] == [counts[1], counts[1]]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("classes", [100, 1000])
def test_projected_smooth_hinge_mass_with_signed_cotangents(classes: int) -> None:
    from libsdca.torch import multiclass_hinge_loss

    scores = torch.full((5, classes), -8.0, dtype=torch.bfloat16, device="cuda")
    scores[:, 0] = 0
    scores[0, 1:] = 1  # One equal share per competitor, exactly unit total mass.
    scores[1, -2:] = 3
    scores[2, -1] = -0.875  # Positive unprojected competitor, mass one quarter.
    scores[3, 1:] = -torch.inf
    labels = torch.zeros(5, dtype=torch.long, device=scores.device)
    z = scores.clone().requires_grad_()
    values = multiclass_hinge_loss(z, labels, gamma=0.5, reduction="none", backend="kernel")
    expected = values.new_tensor([2 - 0.25 / (classes - 1), 3.875, 0.015625, 0, 0])
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
    for sign in (1.0, -1.0):
        scales = sign * values.new_tensor([2.0**17, -3.0, 0.5, 1.0, 0])
        (gradient,) = torch.autograd.grad(values, z, scales, retain_graph=True)
        target = torch.zeros_like(z)
        target[0, 0], target[0, 1:] = -scales[0], scales[0] / (classes - 1)
        target[1, 0], target[1, -2:] = -scales[1], 0.5 * scales[1]
        target[2, 0], target[2, -1] = -0.25 * scales[2], 0.25 * scales[2]
        torch.testing.assert_close(gradient, target, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("active", "gamma", "nearest_word"),
    [(3, 0.6499995180785905, 1072768311), (7, 0.7833325862785803, 1073217875)],
)
def test_projected_hinge_keeps_correct_scalar_near_midpoint(
    active: int, gamma: float, nearest_word: int
) -> None:
    from libsdca.torch import multiclass_hinge_loss

    # The exact projected value, for active margins h, is
    # mean(h) - gamma/(2*k) + sum((h - mean(h))**2)/(2*gamma).
    z = torch.full((8, 1000), -5.0, device="cuda")
    z[:, 0] = 0
    z[:, 1 : active + 1] = z.new_tensor([1.0 - (j % 3) / 128 for j in range(active)])
    y = torch.zeros(8, device="cuda", dtype=torch.long)
    v = multiclass_hinge_loss(z, y, gamma=gamma, reduction="none", backend="kernel")
    assert torch.all(v.view(torch.int32) == nearest_word)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("family", ["hinge", "distillation"])
def test_cuda_compile_reuses_graph_for_variable_batches(family: str) -> None:
    from libsdca.torch import distillation_loss, multiclass_hinge_loss
    from torch._dynamo.testing import CompileCounterWithBackend

    def loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if family == "distillation":
            return distillation_loss(scores, labels, reduction="mean", backend="kernel")
        return multiclass_hinge_loss(scores, labels, reduction="mean", backend="kernel")

    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(loss, backend=counter, fullgraph=True, dynamic=None)
    counts = []
    for n in (9, 13, 17, 7):
        scores = torch.randn(n, 513, device="cuda", requires_grad=True)
        labels = (
            torch.randn_like(scores)
            if family == "distillation"
            else torch.arange(n, device="cuda") % 513
        )
        seed = scores.new_tensor(-1.75)
        actual, expected = compiled(scores, labels), loss(scores, labels)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        (gradient,) = torch.autograd.grad(actual, scores, seed)
        (reference,) = torch.autograd.grad(expected, scores, seed)
        torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
        counts.append(counter.frame_count)
    assert 1 <= counts[1] <= 2
    assert counts[2:] == [counts[1], counts[1]]


@pytest.mark.parametrize("classes", [4097, 32769])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_wide_smooth_retained_projected_tiles(classes: int, dtype: torch.dtype) -> None:
    from hinge_order_cases import wide_smooth_tiles
    from libsdca.torch import multiclass_hinge_loss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    z, y, seed, expected_values, expected = [
        torch.as_tensor(value, device=device) for value in wide_smooth_tiles(classes)
    ]
    z, expected = z.to(dtype), expected.to(dtype)
    x = z.clone().requires_grad_()
    values = multiclass_hinge_loss(x, y, gamma=0.5, reduction="none", backend="kernel")
    torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
    for sign in (1.0, -1.0):
        (gradient,) = torch.autograd.grad(values, x, sign * seed, retain_graph=sign > 0)
        torch.testing.assert_close(gradient, sign * expected, rtol=0, atol=0)
    torch.testing.assert_close(x.detach(), z, rtol=0, atol=0)


def test_wide_bfloat_smooth_midpoint_matches_streamed_pullback() -> None:
    from libsdca.torch import multiclass_hinge_loss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = 4097
    scores = torch.full((2, classes), -8.0, device=device, dtype=torch.bfloat16)
    scores[:, 0] = 0
    scores[:, [1, 1025, 2049]] = 3
    labels = torch.zeros(2, dtype=torch.long, device=device)
    upstream = torch.tensor([1.52734375, -1.52734375], device=device)
    weights = torch.ones(classes, device=device)
    scores.requires_grad_()

    def pullback(weight: torch.Tensor | None) -> torch.Tensor:
        values = multiclass_hinge_loss(
            scores, labels, weight=weight, gamma=0.5, reduction="none", backend="kernel"
        )
        return torch.autograd.grad(values, scores, upstream)[0]

    # Unit class weights route through the full-row kernel. A halfway target
    # exposes a one-word difference if the compact path skips signed summation.
    torch.testing.assert_close(pullback(None), pullback(weights), rtol=0, atol=0)


@pytest.mark.parametrize("gamma", [2.0**-100, 0.375, 0.5, 2.0**60])
def test_smooth_hinge_binary_scale_signed_pullback(gamma: float) -> None:
    """Two tied competitors split a capped unit mass or each take 1/gamma."""
    from libsdca.torch import multiclass_hinge_loss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scores = torch.full((3, 65), -torch.inf, dtype=torch.float32, device=device)
    scores[:, :3] = 0
    scores.requires_grad_()
    labels = torch.zeros(3, dtype=torch.int64, device=device)
    upstream = scores.new_tensor([2.0**-40, -1.75, 0.0])
    loss = multiclass_hinge_loss(scores, labels, gamma=gamma, reduction="none", backend="kernel")

    slope = min(0.5, 1.0 / gamma)
    expected = torch.zeros_like(scores)
    expected[:, 1:3] = (upstream.double() * slope).float()[:, None]
    expected[:, 0] = (-upstream.double() * (2 * slope)).float()
    value = 1.0 - gamma / 4.0 if gamma <= 2.0 else 1.0 / gamma
    torch.testing.assert_close(loss, torch.full_like(loss, value), rtol=0, atol=0)
    for sign in (-1.0, 1.0):
        (gradient,) = torch.autograd.grad(loss, scores, sign * upstream, retain_graph=True)
        torch.testing.assert_close(gradient, sign * expected, rtol=0, atol=0)
