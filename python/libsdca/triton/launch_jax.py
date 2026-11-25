"""The CUDA kernels of libsdca.jax's row kernel: the operands, attributes and results of the
FFI target libsdca_loss (_jax_ops.py), launched through jax-triton or as one CUDA FFI call.

`call` takes class-last scores (rows: every other axis), the target, class weights ((C,),
(G, C) under vmap, or (0,)), the label smoothing (0-d, (G,) under vmap; traced or not) and the
per-row upstream factors, and returns the per-row values (float32 unless the scores are
float64) or the gradient (the scores' dtype) times the factors. A forward that saves also
returns the row statistics its gradient reads (`saved`), which then skips recomputing them.
Under vmap the batch folds into the rows; mapped weights and smoothing apply to consecutive
blocks of rows.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import common
from .common import Call, Ref


@functools.cache
def _jax_triton() -> Any:
    """Use the optional transport when its named array and mutable-ref contract is available."""
    try:
        module = importlib.import_module("jax_triton")
        from ._jax_transport import compatible
    except ImportError:
        return None
    return module if compatible(getattr(module, "triton_call", None)) else None


@functools.cache
def available() -> str | None:
    """None where the kernels run, else why not."""
    reason = common.triton_available()
    if reason is not None:
        return reason
    try:
        if not jax.devices("gpu"):
            return "no CUDA device"
    except RuntimeError:
        return "no CUDA device"
    if _jax_triton() is None:
        from . import jax_ffi

        return jax_ffi.available()
    return None


def route() -> str:
    return " through jax-triton" if _jax_triton() is not None else " through the CUDA FFI target"


def covers(kind: int, k: int, n: int, classes: int, dtype: Any) -> str | None:
    """None where the kernels compute a call, else why not."""
    return common.covers(kind, k, n, classes, jnp.dtype(dtype).name)


@functools.lru_cache(maxsize=256)
def _plan(call: Call) -> common.Plan:
    return common.plan(call)


def _struct(array: common.Array) -> jax.ShapeDtypeStruct:
    """A plan array; float64 statistics (internal or saved) travel as pairs of 32-bit words
    without x64."""
    statistics = array.internal or array.name == "S"
    if (
        statistics
        and array.name != "V"
        and array.dtype == "float64"
        and not jax.config.jax_enable_x64
    ):
        return jax.ShapeDtypeStruct((*array.shape[:-1], 2 * array.shape[-1]), jnp.uint32)
    return jax.ShapeDtypeStruct(array.shape, jnp.dtype(array.dtype))


def _triton_call(**kwargs: Any) -> Any:
    return _jax_triton().triton_call(out_type=kwargs.pop("out"), **kwargs)


def _aliased_gradient(
    launch: common.Launch, env: dict[str, Any], values: common.Array | None
) -> None:
    # A read-write reference lets the enclosing derivative remain pure. The fused
    # row kernel may also write a per-row value for the following reduction.
    from jax.experimental import pallas

    args = {
        k: (env["Y"] if v == Ref("Z") else env[v.name]) if isinstance(v, Ref) else v
        for k, v in launch.args
        if k != "G" and not (values is not None and k == "V")
    }
    meta = dict(launch.meta, ALIAS=True)
    options = {k: meta.pop(k) for k in list(meta) if k.islower() and k != "num_warps"}
    if options:
        meta["backend_options"] = options

    def write(buffers):
        outputs = {"V": buffers[1]} if values is not None else {}
        _triton_call(
            kernel=launch.kernel, out=[], grid=launch.grid, **args, **outputs, G=buffers[0], **meta
        )

    with jax.enable_x64(True):
        buffers = (
            (env["Z"],) if values is None else (env["Z"], jnp.empty(values.shape, values.dtype))
        )
        env["G"], *row_values = pallas.run_state(write)(buffers)
        if row_values:
            env["V"] = row_values[0]


def _aliased(call: Call, launch: common.Launch) -> bool:
    """Whether a launch of an in-place call may write the gradient over the scores: its G
    pointer is bound to Z and no GRAD=False says it leaves G alone. Such a G is a full-size
    output, whatever the launch lists in writes."""
    params = dict(common.parameters(launch.kernel))
    return (
        call.inplace
        and params.get("G") is False
        and dict(launch.args).get("G") == Ref("Z")
        and dict(launch.meta).get("GRAD", True) is not False
    )


def _partial_write(
    launch: common.Launch,
    env: dict[str, Any],
    arrays: dict[str, common.Array],
    aliased: bool = False,
) -> None:
    """Thread output refs between the ordinary and exceptional writers, including donated G:
    the launch's state, its writes and (aliased) the gradient over the scores, each at the
    full shape of its array."""
    from jax.experimental import pallas

    bound = dict(launch.args)
    extra = ("G",) if aliased else ()
    names = tuple(dict.fromkeys((*launch.state, *launch.writes, *extra)))
    alias = "G" in names and bound["G"] == Ref("Z")
    # A kernel with ALIAS reads its scores through G; one without reads Z as an input.
    swap = alias and "ALIAS" in dict(common.parameters(launch.kernel))
    args = {
        k: (env["Y"] if swap and v == Ref("Z") else env[v.name]) if isinstance(v, Ref) else v
        for k, v in launch.args
        if k not in names
    }
    meta = dict(launch.meta)
    if swap:
        meta["ALIAS"] = True
    options = {k: meta.pop(k) for k in list(meta) if k.islower() and k != "num_warps"}
    if options:
        meta["backend_options"] = options
    # Env keys are array names (a parameter can bind an array of another name).
    keys = ["G" if name == "G" and alias else bound[name].name for name in names]

    def write(refs):
        _triton_call(
            kernel=launch.kernel,
            out=[],
            grid=launch.grid,
            **args,
            **dict(zip(names, refs, strict=True)),
            **meta,
        )

    # New arrays take the caller's x64 mode (float64 statistics as 32-bit word pairs without
    # it), as the outputs of the positional binder.
    structs = {key: _struct(arrays[key]) for key in keys if key in arrays}
    with jax.enable_x64(True):
        buffers = []
        for name, key in zip(names, keys, strict=True):
            if key in env and env[key] is not None:
                buffers.append(env[key])
            elif name == "G" and alias:
                buffers.append(env["Z"])
            else:
                buffers.append(jnp.empty(structs[key].shape, structs[key].dtype))
        env.update(zip(keys, pallas.run_state(write)(tuple(buffers)), strict=True))


def _run(call: Call, env: dict[str, Any]) -> dict[str, Any]:
    """Execute a plan: each launch writes the arrays it creates.

    jax-triton binds every input pointer before every output pointer, so the unread pointers
    (Z bound to another parameter) after a launch's first write become 1-element outputs. A
    gradient over the scores (G bound to Z in an in-place call) is threaded as a full-size
    ref instead (_aliased).
    """
    plan = _plan(call)
    if _jax_triton() is None:
        from . import jax_ffi

        return jax_ffi.run(plan, env, call.inplace, _struct)
    arrays = {a.name: a for a in plan.arrays}
    for launch in plan.launches:
        bound = dict(launch.args)
        partial = launch.state and any(name not in launch.writes for name in launch.state)
        if (
            not partial
            and call.inplace
            and bound.get("G") == Ref("Z")
            and (not launch.writes or launch.writes == ("V",))
        ):
            _aliased_gradient(launch, env, arrays["V"] if launch.writes else None)
            continue
        aliased = _aliased(call, launch)
        if aliased or partial:
            # A gradient written over the scores (aliased) is a full-size output, never the
            # 1-element placeholder of an unread pointer below.
            _partial_write(launch, env, arrays, aliased)
            continue
        pointers = [
            name
            for name, constexpr in common.parameters(launch.kernel)
            if not constexpr and isinstance(bound.get(name), Ref)
        ]
        first = min(map(pointers.index, launch.writes), default=len(pointers))
        outputs = pointers[first:]
        if any(
            name not in launch.writes and (name == "Z" or bound[name].name != "Z")
            for name in outputs
        ):
            # An array read after the first written pointer cannot be a positional input:
            # bind the written arrays as named refs instead.
            _partial_write(launch, env, arrays)
            continue
        args = {
            k: env[v.name] if isinstance(v, Ref) else v for k, v in launch.args if k not in outputs
        }
        meta = dict(launch.meta)
        options = {k: meta.pop(k) for k in list(meta) if k.islower() and k != "num_warps"}
        out = [
            _struct(arrays[bound[name].name])
            if name in launch.writes
            else jax.ShapeDtypeStruct((1,), env["Z"].dtype)
            for name in outputs
        ]
        if options:
            meta["backend_options"] = options
        # Static scalars include float64 bit patterns, which overflow int32 without x64.
        with jax.enable_x64(True):
            results = _triton_call(kernel=launch.kernel, out=out, grid=launch.grid, **args, **meta)
        env.update(
            {
                bound[name].name: r
                for name, r in zip(outputs, results, strict=True)
                if name in launch.writes
            }
        )
    return env


def _call(scores, target, weight, attrs: tuple, group: int, saved: int) -> Call:
    *options, gradient = attrs
    c = scores.shape[-1]
    n = int(np.prod(scores.shape[:-1]))
    return Call.of(
        n,
        c,
        scores.dtype.name,
        target.dtype.name,
        *options,
        weighted=weight.size > 0,
        group=group,
        rowwise=common.PER_ROW if gradient else common.UNIFORM,
        saved=saved,
        values=not gradient,
        grad=gradient,
        value_dtype="float64" if jnp.float64 in (scores.dtype, target.dtype) else "float32",
        inplace=gradient,
    )


def saved(scores, target, weight, attrs: tuple) -> jax.ShapeDtypeStruct:
    """The row statistics a forward of attrs saves for its gradient (one row per score row;
    (0,) where it saves none)."""
    call = _call(scores, target, weight, attrs, 0, common.SAVE)
    arrays = {a.name: a for a in _plan(call).arrays}
    if "S" not in arrays or call.n == 0:
        return jax.ShapeDtypeStruct((0,), jnp.uint32)
    struct = _struct(arrays["S"])
    return jax.ShapeDtypeStruct((*scores.shape[:-1], struct.shape[-1]), struct.dtype)


def _launch(scores, target, weight, eps, upstream, stats, attrs: tuple, group: int, save: bool,
            reduction: int = 0, batch_dims: int = 0, inplace: bool = True, both: bool = False):  # fmt: skip
    """Values (no upstream factors; and the statistics a gradient reads, where save) or the
    gradient times them (from stats, where saved), of rows in groups of `group` rows per weight
    and smoothing (0: one group)."""
    gradient = attrs[-1]
    mode = common.SAVE if save else common.SAVED if gradient and stats.size else 0
    call = _call(scores, target, weight, attrs, group, mode)
    if gradient and not inplace:
        call = dataclasses.replace(call, inplace=False)
    if both:
        call = dataclasses.replace(call, values=True, inplace=inplace)
    n, c = call.n, call.c
    if gradient and upstream.ndim == 0:
        call = dataclasses.replace(call, rowwise=common.SCALAR)
    if reduction:
        call = dataclasses.replace(call, reduction=reduction, reduction_rows=group)
    y = target.reshape(n, c) if target.ndim == scores.ndim else target.reshape(n)
    env = {
        "Z": scores.reshape(n, c),
        "Y": y,
        "W": weight.reshape(-1),
        "E": eps.reshape(-1),
        "R": upstream.reshape(-1),
        "S": stats.reshape(n, -1) if mode == common.SAVED else None,
    }
    if gradient and not both:
        return _run(call, env)["G"].reshape(scores.shape) if n else jnp.zeros_like(scores)
    shape = scores.shape[:batch_dims] if reduction else scores.shape[:-1]
    if n:
        out = _run(call, env)
        values = out["L" if reduction else "V"].reshape(shape)
    else:
        out = {}
        values = jnp.full(shape, jnp.nan if reduction == common.MEAN else 0, call.value_dtype)
    if both:
        return values, out["G"].reshape(scores.shape) if n else jnp.zeros_like(scores)
    if not save:
        return values
    struct = saved(scores, target, weight, attrs)
    stats = out["S"].reshape(struct.shape) if struct.size else jnp.zeros(struct.shape, struct.dtype)
    return values, stats


@functools.cache
def _mapped(
    attrs: tuple,
    save: bool,
    reduction: int = 0,
    batch_dims: int = 0,
    inplace: bool = True,
    both: bool = False,
) -> Any:
    """The row kernel of attrs, whose vmap folds the batch into the rows (mapped weights and
    smoothing apply to blocks of rows, as the FFI's broadcast_all)."""

    @jax.custom_batching.custom_vmap
    def rows(scores, target, weight, eps, upstream, stats):
        group = int(np.prod(scores.shape[batch_dims:-1])) if batch_dims else 0
        return _launch(scores, target, weight, eps, upstream, stats, attrs, group, save,
                       reduction, batch_dims, inplace, both)  # fmt: skip

    @rows.def_vmap
    def _(size, batched, scores, target, weight, eps, upstream, stats):
        def front(x, mapped):
            return x if mapped else jnp.broadcast_to(x, (size, *x.shape))

        scores, target = front(scores, batched[0]), front(target, batched[1])
        if attrs[-1] and upstream.ndim == int(batched[4]):
            if batched[4]:
                # Each mapped scalar applies to every score row of that example.
                shape = (size,) + (1,) * (scores.ndim - 2)
                upstream = jnp.broadcast_to(upstream.reshape(shape), scores.shape[:-1])
        elif upstream.size:
            upstream = front(upstream, batched[4])
        if stats.size:
            stats = front(stats, batched[5])
        if reduction or batch_dims or batched[2] or batched[3]:
            weight, eps = front(weight, batched[2]), front(eps, batched[3])
            out = _mapped(attrs, save, reduction, batch_dims + 1, inplace, both)(
                scores, target, weight, eps, upstream, stats
            )
        else:
            out = rows(scores, target, weight, eps, upstream, stats)
        if both:
            return out, (True, True)
        if save:
            return out, (True, out[1].shape != (0,))
        return out, True

    return rows


def attributes(rows: Any, gradient: bool) -> tuple:
    """Call.of's options of rows (_frontend.Rows), and whether the call is a gradient."""
    return (
        int(rows.kind),
        int(rows.k),
        float(rows.gamma),
        float(rows.temperature),
        float(rows.label_smoothing),
        int(rows.ignore_index),
        bool(rows.ignore),
        int(rows.precision),
        bool(gradient),
    )


def call(
    scores: jax.Array,
    target: jax.Array,
    weight: jax.Array,
    eps: jax.Array,
    upstream: jax.Array,
    stats: jax.Array,
    rows: Any,
    gradient: bool,
    save: bool = False,
    reduction: str = "none",
    inplace: bool = True,
    both: bool = False,
) -> Any:
    """The FFI call libsdca_loss on CUDA: per-row values (value dtype; with the statistics
    its gradient reads, where save), or the gradient (scores' dtype) times the per-row
    upstream factors (from stats, unless (0,)); rows are every axis but the last. A nonzero
    rows.label_smoothing says eps can be nonzero."""
    attrs = attributes(rows, gradient)
    mode = {"none": 0, "sum": common.SUM, "mean": common.MEAN}[reduction]
    return _mapped(attrs, save, mode, inplace=inplace, both=both)(
        scores, target, weight, eps, upstream, stats
    )
