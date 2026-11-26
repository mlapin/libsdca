"""JAX CUDA losses whose kernels take a scalar argument equal to 1.

jax-triton (0.5) compiles every static kernel argument equal to 1 (or 1.0) as a constexpr,
whatever the kernel's do_not_specialize says, so inside the kernel it is a Python int: a
method call such as N.to(tl.float64) then fails to compile. One row (N), one row per vmapped
example (GROUP, ROWS), one partial sum of the reduction (N; at most a block of rows) and
ignore_index=1 (IGNORE) all pass 1. Each loss here runs its kernels on such shapes and matches
the composed backend, and the launches of each family compile with every scalar argument as 1.
The same cases also run through the CUDA FFI target, which compiles scalars as i32 or i64, in
a child process that blocks the jax_triton import.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import sys
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from child_process import run_python

# (loss, options, target): target "labels", "probabilities", "mask" or "teacher"; a True
# weight stands for class weights.
_CASES = [
    ("cross_entropy", {}, "labels"),
    ("cross_entropy", {"weight": True}, "labels"),
    ("cross_entropy", {"ignore_index": 1}, "labels"),
    ("cross_entropy", {"weight": True, "ignore_index": 1}, "labels"),
    ("cross_entropy", {"label_smoothing": 0.1}, "labels"),
    ("cross_entropy", {}, "probabilities"),
    ("cross_entropy", {"weight": True, "label_smoothing": 0.1}, "probabilities"),
    ("topk_cross_entropy", {"k": 1}, "labels"),
    ("topk_cross_entropy", {"k": 2, "ignore_index": 1}, "labels"),
    ("multiclass_hinge_loss", {}, "labels"),
    ("multiclass_hinge_loss", {"gamma": 1.0}, "labels"),
    ("multiclass_hinge_loss", {"weight": True, "ignore_index": 1}, "labels"),
    ("topk_hinge_loss", {"k": 1}, "labels"),
    ("topk_hinge_loss", {"k": 2}, "labels"),
    ("topk_hinge_loss", {"k": 2, "gamma": 1.0}, "labels"),
    ("topk_hinge_loss", {"k": 2, "variant": "alpha", "gamma": 1.0}, "labels"),
    ("proximal_cross_entropy", {"gamma": 1.0}, "labels"),
    ("proximal_cross_entropy", {"gamma": 1.0, "weight": True, "ignore_index": 1}, "labels"),
    ("multilabel_cross_entropy", {}, "mask"),
    ("multilabel_hinge_loss", {}, "mask"),
    ("multilabel_hinge_loss", {"gamma": 1.0}, "mask"),
    ("distillation_loss", {"temperature": 1.0}, "teacher"),
    ("proximal_distillation_loss", {"temperature": 1.0, "gamma": 1.0}, "teacher"),
]
_SHAPES = [(1, 10), (10, 10), (64, 10)]


def _cuda() -> None:
    pytest.importorskip("triton")
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("needs a CUDA device")
    from libsdca.triton import launch_jax

    reason = launch_jax.available()
    if reason is not None:
        pytest.skip(reason)


def _inputs(shape: tuple[int, ...], target: str, options: dict[str, Any]) -> tuple[Any, Any, dict]:
    key = jax.random.PRNGKey
    c = shape[-1]
    z = jax.random.normal(key(1), shape, jnp.float32) * 3
    if target == "labels":
        t = jax.random.randint(key(2), shape[:-1], 0, c)
        if "ignore_index" in options:
            # Some ignored rows, never all of them.
            t = t.at[..., 0].set(1).at[..., -1].set(0)
    elif target == "probabilities":
        t = jax.nn.softmax(jax.random.normal(key(2), shape), -1)
    elif target == "mask":
        t = (jax.random.uniform(key(2), shape) < 0.3).at[..., 0].set(True).at[..., 1].set(False)
    else:
        t = jax.random.normal(key(2), shape, jnp.float32) * 3
    options = dict(options)
    if options.get("weight"):
        options["weight"] = jnp.linspace(0.5, 2.0, c, dtype=jnp.float32)
    return z, t, options


def _check(f: Any, z: Any, t: Any) -> None:
    """Values and gradients of the kernel and composed backends of f(z, t, backend)."""

    def total(z: Any, backend: str) -> Any:
        value = f(z, t, backend)
        return value.sum() if value.ndim else value

    def run(backend: str) -> tuple[np.ndarray, np.ndarray]:
        value = jax.jit(lambda z: f(z, t, backend))(z)
        grad = jax.jit(jax.grad(lambda z: total(z, backend)))(z)
        return np.asarray(value), np.asarray(grad)

    expected = run("kernel")
    for got, want in zip(expected, run("composed"), strict=True):
        assert not np.isnan(got).any()
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize(("loss", "options", "target"), _CASES)
def test_scalar_arguments_equal_to_one(
    loss: str, options: dict[str, Any], target: str, shape: tuple[int, int], reduction: str
) -> None:
    _cuda()
    _single(loss, options, target, shape, reduction)


def _single(
    loss: str, options: dict[str, Any], target: str, shape: tuple[int, int], reduction: str
) -> None:
    import libsdca.jax as lj

    z, t, options = _inputs(shape, target, options)
    function = getattr(lj, loss)
    _check(lambda z, t, b: function(z, t, reduction=reduction, backend=b, **options), z, t)


@pytest.mark.cuda
@pytest.mark.parametrize(("loss", "options", "target"), _CASES)
def test_vmapped_single_rows(loss: str, options: dict[str, Any], target: str) -> None:
    """vmap over examples of one row: one row per group and per reduced result."""
    _cuda()
    _vmapped(loss, options, target)


def _vmapped(loss: str, options: dict[str, Any], target: str) -> None:
    import libsdca.jax as lj

    z, t, options = _inputs((4, 1, 10), target, options)
    function = getattr(lj, loss)

    def f(z: Any, t: Any, b: str) -> Any:
        return jax.vmap(lambda z, t: function(z, t, reduction="mean", backend=b, **options))(z, t)

    _check(f, z, t)


@pytest.mark.cuda
@pytest.mark.long
def test_scalar_arguments_without_jax_triton() -> None:
    """Every case above through the CUDA FFI target (a child process without jax_triton)."""
    _cuda()
    from libsdca.triton import launch_jax

    if launch_jax._jax_triton() is None:
        pytest.skip("the tests above already ran through the CUDA FFI target")
    result = run_python(
        "-m",
        "test_jax_cuda_scalars",
        text=True,
        capture_output=True,
        env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 failed" in result.stdout, result.stdout


def _without_jax_triton() -> int:
    """Run every case; print each failure and the counts. Nonzero exit on any failure."""
    from libsdca.triton import launch_jax

    reason = launch_jax.available()
    if reason is not None:
        print(f"CUDA FFI target unavailable: {reason}")
        return 2
    if launch_jax.route() != " through the CUDA FFI target":
        print(f"wrong route:{launch_jax.route()}")
        return 2
    cases: list[tuple[str, Callable[[], None]]] = [
        (
            f"{loss}{options}-{target}-{'x'.join(map(str, shape))}-{reduction}",
            functools.partial(_single, loss, options, target, shape, reduction),
        )
        for loss, options, target in _CASES
        for shape in _SHAPES
        for reduction in ("none", "sum", "mean")
    ] + [
        (f"vmap-{loss}{options}-{target}", functools.partial(_vmapped, loss, options, target))
        for loss, options, target in _CASES
    ]
    failed = 0
    for name, check in cases:
        try:
            check()
        except Exception as error:  # noqa: BLE001 - report every case
            failed += 1
            print(f"FAILED {name}: {type(error).__name__}: {str(error)[:400]}")
    print(f"{len(cases) - failed} passed, {failed} failed")
    return 1 if failed else 0


def _calls() -> list[Any]:
    """A call of every family and target, forward with each reduction and gradient."""
    from libsdca.triton import common as k
    from libsdca.triton.common import Call

    families = [
        (k.CROSS_ENTROPY, "int64", {"weighted": True, "ignore": True, "smoothed": True}),
        (k.CROSS_ENTROPY, "float32", {"smoothed": True}),
        (k.TOPK_CROSS_ENTROPY, "int64", {"k": 2}),
        (k.MULTICLASS_HINGE, "int64", {"gamma": 1.0}),
        (k.TOPK_HINGE_ALPHA, "int64", {"k": 2, "gamma": 1.0}),
        (k.TOPK_HINGE_BETA, "int64", {"k": 2}),
        (k.MULTILABEL_CROSS_ENTROPY, "bool", {}),
        (k.MULTILABEL_HINGE, "bool", {"gamma": 1.0}),
        (k.PROXIMAL_CROSS_ENTROPY, "int64", {"gamma": 1.0}),
        (k.DISTILLATION, "float32", {}),
        (k.PROXIMAL_DISTILLATION, "float32", {"gamma": 1.0}),
    ]
    calls = []
    for (kind, target, options), (n, c) in zip(
        families * 2, [(1, 10)] * len(families) + [(64, 5000)] * len(families), strict=True
    ):
        base = Call(kind, n, c, "float32", target, value_dtype="float32", **options)
        calls += [
            dataclasses.replace(base, grad=False, reduction=reduction)
            for reduction in (0, k.SUM, k.MEAN)
        ]
        calls.append(dataclasses.replace(base, values=False, rowwise=k.PER_ROW, inplace=True))
    return calls


@pytest.mark.cuda
@pytest.mark.long
def test_plans_compile_with_scalars_as_constexpr_one() -> None:
    """Every launch of these plans compiles with all its scalar arguments as the constexpr 1
    that jax-triton passes (also the float parameters, which travel as int64 bits)."""
    _cuda()
    import triton
    from libsdca._jax_ops import cuda_architecture
    from libsdca.triton import common
    from libsdca.triton.common import Ref
    from libsdca.triton.jax_ffi import _TYPES
    from triton.backends.compiler import GPUTarget

    target = GPUTarget("cuda", cuda_architecture(), 32)
    seen = set()
    for call in _calls():
        plan = common.plan(call)
        arrays = {a.name: a.dtype for a in plan.arrays}
        inputs = dict(Z=call.dtype, Y=call.target, W="float32", E="float32", R="float32")
        for launch in plan.launches:
            bound, meta = dict(launch.args), dict(launch.meta)
            signature, constexprs = {}, {}
            for param in launch.kernel.params:
                name = param.name
                if param.is_constexpr:
                    signature[name] = "constexpr"
                    constexprs[name] = meta.get(name, param.default)
                elif isinstance(value := bound[name], Ref):
                    names = arrays if value.name in arrays else inputs
                    signature[name] = "*" + _TYPES[names[value.name]]
                else:
                    signature[name] = "constexpr"
                    constexprs[name] = 1
            options = {k: v for k, v in meta.items() if k.islower()}
            key = (launch.kernel.fn, repr(signature), repr(constexprs), repr(options))
            if key in seen:
                continue
            seen.add(key)
            source = triton.compiler.ASTSource(launch.kernel, signature, constexprs, {})
            triton.compile(source, target=target, options=options)


if __name__ == "__main__":
    sys.modules["jax_triton"] = None  # pyright: ignore[reportArgumentType] - block the import
    sys.exit(_without_jax_triton())
