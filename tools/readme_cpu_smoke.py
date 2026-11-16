"""Confirm installed shared CPU distillation/proximal dispatch; not a benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any
from unittest import mock

import libsdca
import numpy as np


def torch_checks() -> list[dict[str, object]]:
    import torch
    from libsdca import _cpu, _cpu_entropy

    operators = _cpu_entropy.operators()
    rows = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        student = torch.linspace(-1, 1, 68, dtype=dtype).reshape(4, 17).requires_grad_()
        teacher = student.detach().flip(-1).contiguous()
        _cpu._applied.cache_clear()
        with mock.patch.object(operators, "apply", wraps=operators.apply) as observed:
            value = libsdca.distillation_loss(student, teacher)
            (gradient,) = torch.autograd.grad(value, student)
        _cpu._applied.cache_clear()
        assert observed.call_count > 0, "shared CPU distillation entry was not executed"
        kinds = [_cpu.APPLIED[call.args[0]] for call in observed.call_args_list]
        assert all(kind in ("distillation_stats", "distillation_grad") for kind in kinds)
        assert value.isfinite().all() and gradient.isfinite().all()
        assert gradient.dtype == dtype and gradient.device.type == "cpu"
        rows.append(
            {
                "dtype": str(dtype),
                "entry": "apply",
                "kinds": kinds,
                "calls": observed.call_count,
                "status": "PASS",
            }
        )
    return rows


def jax_checks() -> list[dict[str, object]]:
    import jax
    import jax.numpy as jnp
    import libsdca.jax as sjax
    from libsdca import _jax_ffi

    assert jax.default_backend() == "cpu"
    assert hasattr(libsdca.Solver().lib, "sdca_run_rows"), (
        "installed core lacks the shared row executor"
    )
    jax.config.update("jax_enable_x64", True)
    rows = []
    for dtype in (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64):
        student = jnp.linspace(-1, 1, 68, dtype=dtype).reshape(4, 17)
        teacher = jnp.flip(student, -1)
        operation = jax.jit(jax.value_and_grad(sjax.distillation_loss))
        lowered = operation.lower(student, teacher).as_text()
        entries = (
            [_jax_ffi.DISTILLATION_STATS, _jax_ffi.DISTILLATION_BACKWARD]
            if dtype in (jnp.float16, jnp.bfloat16)
            else [_jax_ffi.DISTILLATION]
        )
        assert all(entry in lowered for entry in entries), "shared CPU FFI target is absent"
        value, gradient = operation(student, teacher)
        gradient.block_until_ready()
        assert jnp.isfinite(value).all() and jnp.isfinite(gradient).all()
        assert gradient.dtype == dtype
        rows.append(
            {
                "api": "distillation",
                "dtype": str(gradient.dtype),
                "entries": entries,
                "status": "PASS",
            }
        )
    rows.extend(jax_proximal_checks())
    return rows


def proximal_call(
    student: Any, teacher: Any, precision: str | None, temperature: float = 2, alpha: float = 0.5
) -> tuple[Any, Any, str]:
    import jax
    import libsdca.jax as sjax

    operation = jax.jit(
        jax.value_and_grad(
            lambda s: sjax.proximal_distillation_loss(
                s, teacher, temperature, alpha, precision=precision, reduction="sum"
            )
        )
    )
    lowered = operation.lower(student).as_text()
    value, gradient = operation(student)
    gradient.block_until_ready()
    assert gradient.dtype == student.dtype
    return value, gradient, lowered


def proximal_native_case(student: Any, teacher: Any, precision: str | None) -> dict[str, object]:
    import jax.numpy as jnp
    from libsdca import _jax_ffi

    value, gradient, lowered = proximal_call(student, teacher, precision)
    assert _jax_ffi.PROXIMAL_DISTILLATION in lowered, "shared proximal CPU FFI target is absent"
    assert bool(jnp.isfinite(value)) and bool(jnp.isfinite(gradient).all())
    assert value.dtype == jnp.result_type(student.dtype, jnp.float32)
    np.testing.assert_array_equal(gradient[0], 0)
    storage = np.float64 if student.dtype == jnp.float64 or precision == "float64" else np.float32
    expected_value, expected_gradient = libsdca.proximal_distillation_loss_and_grad(
        np.asarray(student, storage), np.asarray(teacher, storage), temperature=2, alpha=0.5
    )
    expected_value = jnp.asarray(expected_value, storage).sum().astype(value.dtype)
    expected_gradient = jnp.asarray(expected_gradient).astype(student.dtype)
    for actual, expected in ((value, expected_value), (gradient, expected_gradient)):
        if student.dtype == jnp.float64:
            np.testing.assert_allclose(actual, expected, rtol=16 * np.finfo(np.float64).eps, atol=0)
        else:
            np.testing.assert_array_equal(actual, expected)
    return dict(
        api="proximal_distillation",
        dtype=str(student.dtype),
        precision=precision,
        entries=[_jax_ffi.PROXIMAL_DISTILLATION],
        work_precision="float64",
        work_policy="fixed-fp64 native solver",
        reference="native C ABI",
        status="PASS",
    )


def proximal_derivative_case() -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    import libsdca.jax as sjax

    student = jnp.asarray([[0.3, -0.1, 0.2]], jnp.float64)
    teacher = jnp.asarray([[-0.4, 0.1, 0.3]], jnp.float64)
    results = []
    for mode in ("on", "off"):
        with mock.patch.dict(os.environ, SDCA_JAX_FFI=mode):

            def loss(s: Any) -> Any:
                return sjax.proximal_distillation_loss(s, teacher, 2, 0.5)

            result = jax.jit(jax.hessian(loss))(student)
            result.block_until_ready()
            assert bool(jnp.isfinite(result).all())
            results.append(result)
    np.testing.assert_allclose(results[0], results[1], rtol=1e-10, atol=1e-12)
    return dict(
        api="proximal_distillation",
        dtype="float64",
        case="second_derivatives",
        derivatives=["hessian (forward-over-reverse)"],
        reference="XLA (FFI off)",
        status="PASS",
    )


def jax_proximal_checks() -> list[dict[str, object]]:
    import jax.numpy as jnp
    from libsdca import _jax_ffi

    rows = []
    for dtype in (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64):
        rng = np.random.default_rng(713)
        student = jnp.asarray(rng.normal(size=(3, 11)), dtype)
        teacher = jnp.asarray(rng.normal(size=(3, 11)), dtype).at[0].set(student[0])
        for precision in (None, "float64"):
            rows.append(proximal_native_case(student, teacher, precision))
        if dtype != jnp.float64:
            value, gradient, lowered = proximal_call(student, teacher, "float32")
            assert _jax_ffi.PROXIMAL_DISTILLATION not in lowered, "fp32 work entered fp64 solver"
            assert bool(jnp.isfinite(value)) and bool(jnp.isfinite(gradient).all())
            np.testing.assert_array_equal(gradient[0], 0)
            rows.append(
                dict(
                    api="proximal_distillation",
                    dtype=str(student.dtype),
                    precision="float32",
                    entries=[],
                    work_precision="float32",
                    work_policy="XLA array solve",
                    status="PASS",
                )
            )
    maximum = np.finfo(np.float64).max
    student = jnp.asarray([[-maximum, maximum, 0.0]], jnp.float64)
    value, gradient, lowered = proximal_call(student, -student, None, temperature=1, alpha=1)
    assert _jax_ffi.PROXIMAL_DISTILLATION in lowered
    assert bool(jnp.isposinf(value)), "overflow must remain +inf, not NaN"
    np.testing.assert_array_equal(gradient, [[-1.0, 1.0, 0.0]])
    rows.append(
        dict(
            api="proximal_distillation",
            dtype="float64",
            case="overflow",
            entries=[_jax_ffi.PROXIMAL_DISTILLATION],
            expected_loss="positive infinity",
            work_precision="float64",
            status="PASS",
        )
    )
    rows.append(proximal_derivative_case())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("torch", "jax"))
    parser.add_argument("--require-jax-only", action="store_true")
    args = parser.parse_args()
    if args.require_jax_only and args.profile != "jax":
        parser.error("--require-jax-only requires the JAX profile")
    torch_available = importlib.util.find_spec("torch") is not None
    if args.require_jax_only:
        assert not torch_available, "JAX-only environment unexpectedly contains PyTorch"
    rows = torch_checks() if args.profile == "torch" else jax_checks()
    if args.profile == "jax":
        for row in rows:
            row["jax_only_environment"] = not torch_available
    Path("cpu-dispatch.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
