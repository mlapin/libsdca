"""Execute installed CUDA distillation through the shared Triton paths, not a benchmark."""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from unittest import mock


def torch_checks() -> list[dict[str, object]]:
    import libsdca.torch as sdca
    import torch
    from libsdca import _triton_distillation

    rows = []
    for precision in (None, "float64"):
        student = torch.linspace(-1, 1, 136, device="cuda").reshape(8, 17).requires_grad_()
        teacher = student.detach().flip(-1).contiguous()
        sdca._kernel_distill.cache_clear()
        with mock.patch.object(
            _triton_distillation,
            "distillation_loss",
            wraps=_triton_distillation.distillation_loss,
        ) as observed:
            value = sdca.distillation_loss(student, teacher, precision=precision)
            (gradient,) = torch.autograd.grad(value, student)
        sdca._kernel_distill.cache_clear()
        torch.cuda.synchronize()
        assert observed.call_count > 0, "distillation bypassed shared Triton kernels"
        assert value.isfinite().all() and gradient.isfinite().all()
        assert gradient.device.type == "cuda" and gradient.dtype == student.dtype
        rows.append(
            {
                "precision": precision,
                "entry": "_triton_distillation.distillation_loss",
                "calls": observed.call_count,
                "status": "PASS",
            }
        )
    return rows


def jax_checks() -> list[dict[str, object]]:
    import jax
    import jax.numpy as jnp
    import libsdca.jax as sdca
    from libsdca import _jax_cuda

    assert jax.default_backend() == "gpu"
    jax.config.update("jax_enable_x64", True)
    rows = []
    for precision in (None, "float64"):
        student = jnp.linspace(-1, 1, 136, dtype=jnp.float32).reshape(8, 17)
        teacher = jnp.flip(student, -1)

        loss = functools.partial(sdca.distillation_loss, teacher=teacher, precision=precision)
        operation = jax.jit(jax.value_and_grad(loss))
        assert _jax_cuda.TARGET in operation.lower(student).as_text(), (
            "distillation bypassed shared Triton FFI"
        )
        value, gradient = operation(student)
        gradient.block_until_ready()
        assert jnp.isfinite(value).all() and jnp.isfinite(gradient).all()
        assert gradient.dtype == student.dtype
        rows.append({"precision": precision, "entry": _jax_cuda.TARGET, "status": "PASS"})
    return rows


def main() -> None:
    rows = torch_checks() if sys.argv[1] == "torch" else jax_checks()
    Path("cuda-dispatch.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
