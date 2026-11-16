"""Repeated compiled CUDA calls release outputs without clearing compilation caches."""

import gc
import json
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from libsdca import _jax_cuda
from libsdca import jax as losses
from test_jax import x64

pytestmark = pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires CUDA")


@pytest.mark.parametrize(
    "family",
    ["cross_entropy", "distillation_loss", "proximal_cross_entropy", "proximal_distillation_loss"],
)
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_repeated_forward_backward_releases_device_buffers(
    family: str, precision: str, record_property: Callable
) -> None:
    with x64(precision == "float64"):
        rng = np.random.default_rng(1729)
        student = jnp.asarray(rng.normal(size=(256, 1000)).astype(np.float32))
        target = jnp.asarray(
            rng.normal(size=(256, 1000)).astype(np.float32)
            if "distillation" in family
            else rng.integers(1000, size=256, dtype=np.int32)
        )
        loss = getattr(losses, family)
        run = jax.jit(jax.value_and_grad(lambda s: loss(s, target, precision=precision)))
        assert _jax_cuda.TARGET in run.lower(student).as_text()
        device = next(iter(student.devices()))

        def block(count: int) -> int:
            for _ in range(count):
                result = run(student)
                jax.block_until_ready(result)
                del result
            gc.collect()
            jax.effects_barrier()
            stats = device.memory_stats()
            assert stats is not None and "bytes_in_use" in stats
            return stats["bytes_in_use"]

        block(32)
        samples = [block(20) for _ in range(11)]
        record_property("live_cuda_bytes", json.dumps(samples))
        assert max(samples[1:]) - samples[0] <= 64 * 1024, samples
