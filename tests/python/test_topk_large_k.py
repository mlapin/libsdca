"""Top-k entropy for k above the insertion-selection size, against a 60-digit reference."""

import contextlib
import os
from unittest import mock

import numpy as np
import pytest
import torch
from child_process import run_python
from libsdca.torch import topk_cross_entropy
from test_topk_references import topk_entropy_reference

ROUTES = [
    ("cpu", "auto", "on"),
    ("cpu", "native", "off"),
    ("cpu", "torch", "off"),
    ("cuda", "auto", "on"),
    ("cuda", "torch", "off"),
]


def _route(device: str, extension: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if device == "cuda":
        return contextlib.nullcontext()
    return mock.patch.dict(os.environ, {"SDCA_CPU_EXTENSION": extension})


def _rows(classes: int) -> np.ndarray:
    """Evenly spaced competitors at three spacings (none, about half, and all of the k
    largest capped at k = classes / 2), one with ties on a coarse grid, all permuted."""
    rng = np.random.default_rng(classes)
    steps = -np.arange(classes - 1, dtype=np.float64)
    rows = [steps * 0.002, steps * 0.02, steps * 2.0, np.floor(steps * 0.05) * 0.5]
    return np.stack([np.concatenate([[0.0], rng.permutation(row)]) for row in rows])


@pytest.mark.parametrize("device,backend,extension", ROUTES)
@pytest.mark.parametrize("k", [16, 17, 64, 128, 255])
def test_large_k_against_reference(device: str, backend: str, extension: str, k: int):
    route = _route(device, extension)
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
        start = time.perf_counter()
        topk_cross_entropy(z, y, k=classes // 2, backend=sys.argv[1])
        best = min(best, time.perf_counter() - start)
    return best
seconds(512)
print(seconds(8192) / seconds(1024))
"""


@pytest.mark.parametrize("backend,extension", [route[1:] for route in ROUTES[:2]])
def test_large_k_scales_nearly_linearly(backend: str, extension: str):
    """k = n / 2: eight times the classes take far less than the 64 times of O(k^2) work. On
    one thread, in a child: thread teams stall on a loaded machine (other test workers)."""
    env = {**os.environ, "SDCA_CPU_EXTENSION": extension}
    env |= {"SDCA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    out = run_python("-c", SCALING, backend, env=env, check=True, capture_output=True, text=True)
    ratio = float(out.stdout.split()[-1])
    assert ratio < 24, ratio
