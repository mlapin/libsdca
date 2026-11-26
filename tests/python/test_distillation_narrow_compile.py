"""CUDA distillation of narrow rows compiles in seconds on a cold kernel cache.

The one-row-per-thread kernel unrolls its column loops, and its compile time grows steeply
with the class count: it once took rows below 32 classes, and a width in the twenties took
minutes, in every process through jax-triton (which keeps no kernel cache). Each case runs
in a child process with an empty Triton cache, so the time includes every compile of a
training step: the widest row that kernel takes and a width the general path takes.
Opt-in (-m cuda): a GPU session.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from child_process import run_python

pytestmark = pytest.mark.cuda

# Seconds for cold training steps (forward and backward) of every width and dtype below, with
# room for a busy machine: on an A100 both tests took 167 s together, and the unrolled kernel
# alone took 260 s for 31 float32 classes and 360 s for 31 bfloat16 classes.
BUDGET = 300

TORCH = """
import torch
from libsdca.torch import distillation_loss
for c in (11, 31):
    for dtype in (torch.float32, torch.bfloat16):
        z = (torch.randn(64, c, device="cuda") * 8).to(dtype).requires_grad_()
        t = (torch.randn(64, c, device="cuda") * 8).to(dtype)
        distillation_loss(z, t, temperature=2.0).backward()
torch.cuda.synchronize()
"""

JAX = """
import jax, jax.numpy as jnp
from libsdca.jax import distillation_loss
for c in (11, 31):
    for dtype in (jnp.float32, jnp.bfloat16):
        z = (jax.random.normal(jax.random.PRNGKey(0), (64, c)) * 8).astype(dtype)
        t = (jax.random.normal(jax.random.PRNGKey(1), (64, c)) * 8).astype(dtype)
        jax.block_until_ready(jax.grad(lambda a: distillation_loss(a, t, temperature=2.0))(z))
"""


def _cold(source: str, cache: Path) -> float:
    env = dict(os.environ, TRITON_CACHE_DIR=str(cache), XLA_PYTHON_CLIENT_PREALLOCATE="false")
    start = time.perf_counter()
    run_python("-c", source, env=env, check=True, capture_output=True, text=True)
    return time.perf_counter() - start


def test_torch_narrow_rows_compile_in_seconds(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    assert _cold(TORCH, tmp_path) < BUDGET


def test_jax_narrow_rows_compile_in_seconds(tmp_path: Path) -> None:
    jax = pytest.importorskip("jax")
    pytest.importorskip("jax_triton")
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA required")
    assert _cold(JAX, tmp_path) < BUDGET
