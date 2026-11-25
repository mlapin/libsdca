"""Top-k entropy for k above the insertion-selection size, against a 60-digit reference."""

import contextlib
import os

import numpy as np
import pytest
import torch
from child_process import run_python
from libsdca.torch import Backend, topk_cross_entropy
from test_topk_references import topk_entropy_reference

ROUTES = [("cpu", "auto"), ("cpu", "composed"), ("cuda", "auto"), ("cuda", "composed")]


def _route(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return contextlib.nullcontext()


def _rows(classes: int) -> np.ndarray:
    """Evenly spaced competitors at three spacings (none, about half, and all of the k
    largest capped at k = classes / 2), one with ties on a coarse grid, all permuted."""
    rng = np.random.default_rng(classes)
    steps = -np.arange(classes - 1, dtype=np.float64)
    rows = [steps * 0.002, steps * 0.02, steps * 2.0, np.floor(steps * 0.05) * 0.5]
    return np.stack([np.concatenate([[0.0], rng.permutation(row)]) for row in rows])


@pytest.mark.parametrize("device,backend", ROUTES)
@pytest.mark.parametrize("k", [16, 17, 64, 128, 255])
def test_large_k_against_reference(device: str, backend: Backend, k: int):
    route = _route(device)
    scores = _rows(257)
    x = torch.tensor(scores, device=device, requires_grad=True)
    y = torch.zeros(len(scores), device=device, dtype=torch.int64)
    with route:
        value = topk_cross_entropy(x, y, k=k, backend=backend, reduction="none")
        (gradient,) = torch.autograd.grad(value.sum(), (x,))
    for row, z in enumerate(scores):
        loss, expected = topk_entropy_reference(z, 0, k)
        assert value[row].item() == pytest.approx(loss, rel=1e-13)
        torch.testing.assert_close(
            gradient[row].cpu(), torch.from_numpy(expected), rtol=1e-12, atol=1e-300
        )


SCALING = """
import sys, time
import torch
from libsdca.torch import topk_cross_entropy
def seconds(classes):
    z = torch.linspace(-10, 10, classes, dtype=torch.float64).repeat(64, 1)
    y = torch.zeros(64, dtype=torch.int64)
    best = float("inf")
    for _ in range(5):
        start = time.process_time()
        topk_cross_entropy(z, y, k=classes // 2, backend=sys.argv[1])
        best = min(best, time.process_time() - start)
    return best
seconds(512)
print(seconds(8192) / seconds(1024))
"""


QUADRATIC = pytest.mark.xfail(
    strict=True, reason="[top-k] the CPU top-k CE kernel at k = n / 2 takes O(k^2): ratio 55"
)


@pytest.mark.parametrize("backend", [pytest.param("auto", marks=QUADRATIC), "composed"])
def test_large_k_scales_nearly_linearly(backend: Backend):
    """k = n / 2: eight times the classes take far less than the 64 times of O(k^2) work. On
    one thread, in a child, in CPU time: thread teams stall and other test workers preempt
    the child on a loaded machine."""
    env = {**os.environ, "SDCA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    out = run_python("-c", SCALING, backend, env=env, check=True, capture_output=True, text=True)
    ratio = float(out.stdout.split()[-1])
    assert ratio < 24, ratio
