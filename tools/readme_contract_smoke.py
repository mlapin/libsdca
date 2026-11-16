"""Installed-package numerical contracts; only README runtime dependencies are used.

The runner copies the canonical score-order fixture beside this standalone probe.
No source-tree imports, pytest, dependency repairs, or modified README examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast
from unittest import mock

import numpy as np
from hinge_order_cases import rows as hinge_rows

HINGES = ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta")


def hinge_checks(profile: str, device: str) -> list[dict[str, object]]:
    records = []
    for dtype in (
        ("float32", "float64")
        if profile in ("numpy", "sklearn")
        else ("float32", "float64", "bfloat16")
    ):
        for power in (-29, -60):
            z, labels, expected = hinge_rows(17, power)
            for loss in HINGES:
                for precision in (None, "float64"):
                    details = evaluate_hinge(profile, device, z, labels, dtype, loss, precision)
                    value, gradient, route = details
                    assert np.isfinite(value).all(), (loss, dtype, precision)
                    np.testing.assert_array_equal(gradient, expected)
                    records.append(
                        dict(
                            contract="hinge_score_order",
                            dtype=dtype,
                            power=power,
                            loss=loss,
                            precision=precision,
                            route=route,
                            status="PASS",
                        )
                    )
    return records


def evaluate_hinge(
    profile: str,
    device: str,
    z: np.ndarray,
    labels: np.ndarray,
    dtype: str,
    loss: str,
    precision: str | None,
) -> tuple[Any, np.ndarray, str]:
    if profile in ("numpy", "sklearn"):
        import libsdca

        cls: Any = libsdca.LOSSES[loss]
        instance = cls(k=1) if loss in libsdca.TOPK else cls()
        value, gradient = libsdca.loss_and_grad(
            z.astype(dtype), labels, instance, precision=precision
        )
        return value, gradient, "native C ABI"
    if profile == "torch":
        import libsdca.torch as api
        import torch
        from libsdca import _hinge

        x = torch.tensor(z, dtype=getattr(torch, dtype), device=device, requires_grad=True)
        y = torch.tensor(labels, device=device)
        options: dict[str, Any] = dict(precision=precision, reduction="sum")
        if loss != "multiclass_hinge":
            options.update(k=1, variant=loss.removeprefix("topk_hinge_"))
        function = api.multiclass_hinge_loss if loss == "multiclass_hinge" else api.topk_hinge_loss
        with mock.patch.object(_hinge, "hinge", wraps=_hinge.hinge) as fused:
            value = function(x, y, **options)
            (gradient,) = torch.autograd.grad(value, x)
        if device == "cuda":
            assert fused.call_count > 0, "hinge did not execute shared CUDA kernels"
        return (
            value.detach().double().cpu().numpy(),
            gradient.double().cpu().numpy(),
            type(value.grad_fn).__name__,
        )
    import jax
    import jax.numpy as jnp
    import libsdca.jax as api
    from libsdca import _jax_cuda

    x, y = jnp.asarray(z, dtype=getattr(jnp, dtype)), jnp.asarray(labels)
    hinge: dict[str, Any] = dict(precision=precision, reduction="sum")
    if loss != "multiclass_hinge":
        hinge.update(k=1, variant=loss.removeprefix("topk_hinge_"))
    loss_of = api.multiclass_hinge_loss if loss == "multiclass_hinge" else api.topk_hinge_loss
    operation = jax.jit(jax.value_and_grad(lambda a: loss_of(a, y, **hinge)))
    lowered = operation.lower(x).as_text()
    if device == "cuda":
        assert _jax_cuda.TARGET in lowered, "hinge did not lower to shared CUDA kernels"
    value, gradient = operation(x)
    gradient.block_until_ready()
    return (
        np.asarray(value),
        np.asarray(gradient),
        "shared CUDA FFI" if device == "cuda" else "auto",
    )


def proximal_case(dtype: str, backend: str, precision: str | None, device: str) -> tuple[Any, dict]:
    import jax
    import jax.numpy as jnp
    import libsdca.jax as api
    from libsdca import _jax_cuda, _jax_ffi

    x = jnp.asarray(np.random.default_rng(1729).normal(size=(7, 13)), dtype=getattr(jnp, dtype))
    y = jnp.arange(7)
    operation = jax.jit(
        jax.value_and_grad(
            lambda a: api.proximal_cross_entropy(
                a, y, backend=cast("Any", backend), precision=precision
            )
        )
    )
    # Wrap the real route decision: output dtype alone cannot identify working precision.
    with (
        mock.patch.object(api, "_jax_backend", wraps=api._jax_backend) as decision,
        mock.patch.object(api, "_native_call", wraps=api._native_call) as native_gradient,
        mock.patch.object(api, "_native_values", wraps=api._native_values) as native_value,
    ):
        lowered = operation.lower(x).as_text()
        outputs = operation(x)
        outputs[1].block_until_ready()
    resolved = precision or "float32"
    assert decision.call_args_list, "proximal CE route decision was not observed"
    assert all(call.args[3] == resolved for call in decision.call_args_list)
    native_calls = native_gradient.call_args_list + native_value.call_args_list
    assert all(call.args[2][5] == resolved for call in native_calls)
    native_entries = [entry for entry in (_jax_ffi.GRADIENT, _jax_ffi.VALUES) if entry in lowered]
    route = "array"
    if device == "cuda":
        assert _jax_cuda.TARGET in lowered, "proximal CE did not lower to shared CUDA kernels"
        route = "shared CUDA FFI"
    elif native_entries:
        # FFI attributes encode fp32 storage/work as 0 and fp32 storage/fp64 work as 1.
        code = 1 if resolved == "float64" else 0
        assert f"precision = {code} : i32" in lowered, "wrong native FFI work precision"
        route = "native FFI"
    elif "callback" in lowered:
        route = "native host callback"
    if backend == "native":
        assert route.startswith("native"), "explicit native backend was not executed"
        assert native_calls, "native proximal CE work precision was not observed"
    for array in outputs:
        assert np.isfinite(np.asarray(array, dtype=np.float64)).all()
    return outputs, dict(
        contract="jax_proximal_ce_work",
        dtype=dtype,
        backend=backend,
        precision=precision,
        resolved_precision=resolved,
        native_calls=len(native_calls),
        route=route,
        status="PASS",
    )


def proximal_checks(device: str) -> list[dict[str, object]]:
    records = []
    for dtype in ("float32", "bfloat16"):
        for backend in ("auto", "native") if device == "cpu" else ("auto",):
            outputs = []
            for precision in (None, "float32", "float64"):
                result, record = proximal_case(dtype, backend, precision, device)
                records.append(record)
                outputs.append(result)
            for default, narrow, wide in zip(*outputs, strict=True):
                np.testing.assert_array_equal(default, narrow)
                np.testing.assert_allclose(
                    np.asarray(default, dtype=np.float64),
                    np.asarray(wide, dtype=np.float64),
                    rtol=0.02 if dtype == "bfloat16" else 1e-5,
                    atol=0.002 if dtype == "bfloat16" else 1e-7,
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("numpy", "sklearn", "torch", "jax", "jax-cuda"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and args.profile not in ("torch", "jax-cuda"):
        parser.error("CUDA requires torch or jax-cuda")
    if args.profile == "jax-cuda" and args.device != "cuda":
        parser.error("jax-cuda requires CUDA")
    if args.profile in ("jax", "jax-cuda"):
        import jax

        assert jax.default_backend() == ("gpu" if args.device == "cuda" else "cpu")
        jax.config.update("jax_enable_x64", True)
    records = hinge_checks(args.profile, args.device)
    if args.profile in ("jax", "jax-cuda"):
        records.extend(proximal_checks(args.device))
    Path("contracts.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records))


if __name__ == "__main__":
    main()
