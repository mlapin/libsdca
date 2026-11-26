"""Torch public transforms keep each mapped example's row total wide until its final cast."""

import libsdca.torch as L
import pytest
import torch


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_cpu_label_reduction_counts_ignored_rows_across_tiles(reduction: str) -> None:
    rng = torch.Generator().manual_seed(1729)
    scores = torch.randn(2053, 11, generator=rng)
    target = torch.randint(11, (len(scores),), generator=rng)
    target[::3] = -100
    target[1024:2048] = -100
    for labels in (target, torch.full_like(target, -100)):
        expected = torch.nn.functional.cross_entropy(scores.double(), labels, reduction=reduction)
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            serial = L.cross_entropy(scores, labels, reduction=reduction)
            torch.set_num_threads(min(4, threads))
            parallel = L.cross_entropy(scores, labels, reduction=reduction)
        finally:
            torch.set_num_threads(threads)
        torch.testing.assert_close(serial, expected.float(), equal_nan=True)
        torch.testing.assert_close(serial, parallel, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
        ),
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_hinge_rounds_after_reduction(device: str, reduction: L.Reduction) -> None:
    batch = torch.tensor(
        [[[0, 2**24], [0, 1]], [[1, 0], [0, 0]]], dtype=torch.float32, device=device
    )
    mask = torch.tensor([[True, False], [True, False]], device=device)

    def loss(scores: torch.Tensor) -> torch.Tensor:
        return L.multilabel_hinge_loss(scores, mask, reduction=reduction)

    divisor = 2 if reduction == "mean" else 1
    expected = torch.tensor(
        [(2**24 + 3) / divisor, 1 / divisor], dtype=torch.float32, device=device
    )
    expected_grad = torch.tensor(
        [[[-1.0, 1.0], [-1.0, 1.0]], [[-0.5, 0.5], [-1.0, 1.0]]], device=device
    )
    mapped = torch.vmap(loss)
    res = torch.func.vjp(mapped, batch)
    value, pullback = res[:2]
    scale = torch.tensor([-1.5, 0.25], device=device)
    torch.testing.assert_close(value, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        pullback(scale)[0], (expected_grad / divisor) * scale[:, None, None], rtol=0, atol=0
    )
    torch.testing.assert_close(torch.stack([loss(x) for x in batch]), expected, rtol=0, atol=0)
    compiled = torch.compile(loss, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(batch[0]), expected[0], rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nested_mapped_weighted_means(dtype: torch.dtype, device: str) -> None:
    """Each group uses its own mean divisor even when the weight vector is shared."""
    g = torch.Generator().manual_seed(712)
    z = torch.randn(2, 3, 5, 4, generator=g, dtype=dtype).to(device)
    y = torch.randint(4, (2, 3, 5), generator=g).to(device)
    y[1, 2, 0] = -1
    weight = torch.tensor([1.0, 5.0, 2.0, 3.0], dtype=dtype, device=device)
    seed = torch.tensor([[-1.5, 1.0, 2.25], [0.25, 0.0, -4.0]], device=device)

    def loss(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return L.cross_entropy(x, target, weight, ignore_index=-1, backend="kernel")

    values, back = torch.func.vjp(lambda x: torch.vmap(torch.vmap(loss))(x, y), z)[:2]
    (grads,) = back(seed)
    for i in range(2):
        for j in range(3):
            v, pb = torch.func.vjp(lambda x: loss(x, y[i, j]), z[i, j])[:2]
            torch.testing.assert_close(values[i, j], v, rtol=0, atol=0)
            torch.testing.assert_close(grads[i, j], pb(seed[i, j])[0], rtol=0, atol=0)


def test_mapped_pullback_runs_a_constant_number_of_row_calls() -> None:
    # Half storage re-applies the incoming cotangent before casting in backward.
    # libsdca::loss otherwise defaults to one dispatcher call per mapped example.
    def loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return L.multiclass_hinge_loss(x, y, backend="kernel")

    mapped = torch.vmap(torch.func.grad_and_value(loss))
    generator = torch.Generator().manual_seed(310)
    calls = []
    for size in (8, 32):
        scores = torch.randn(size, 5, 16, dtype=torch.bfloat16, generator=generator)
        targets = torch.randint(16, (size, 5), generator=generator)
        mapped(scores, targets)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            mapped(scores, targets)
        calls.append(
            sum(event.count for event in profile.key_averages() if event.key == "libsdca::loss")
        )
    assert calls[0] > 0 and calls[0] == calls[1], calls
