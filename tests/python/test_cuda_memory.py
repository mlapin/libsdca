"""Live CUDA allocations stabilize after warmup; allocator caches are not leaks."""

import gc
import json
from collections.abc import Callable

import pytest
import torch
from libsdca import torch as losses

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize(
    "family",
    ["cross_entropy", "distillation_loss", "proximal_cross_entropy", "proximal_distillation_loss"],
)
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_repeated_forward_backward_releases_device_buffers(
    family: str, precision: str, record_property: Callable
) -> None:
    torch.manual_seed(1729)
    student = torch.randn(256, 1000, device="cuda", requires_grad=True)
    target = (
        torch.randn_like(student)
        if "distillation" in family
        else torch.randint(1000, (256,), device="cuda")
    )
    loss = getattr(losses, family)

    def block(count: int) -> int:
        for _ in range(count):
            value = loss(student, target, precision=precision)
            value.backward()
            student.grad = None
            del value
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()

    block(32)
    samples = [block(20) for _ in range(11)]
    record_property("live_cuda_bytes", json.dumps(samples))
    # 64 KiB admits small bookkeeping allocations, but not even one 1 MiB
    # input-sized gradient retained by a completed call. Do not empty caches:
    # reserved bytes can legitimately remain at their high-water mark.
    assert max(samples[1:]) - samples[0] <= 64 * 1024, samples
