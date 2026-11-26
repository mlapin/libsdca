"""Choose a separate gradient buffer for a caller-owned JIT input at lowering time."""

from __future__ import annotations

import inspect
from typing import Any

import jax
import jax.core
import numpy as np
from jax.extend import core
from jax.interpreters import batching, mlir
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import func

_external = core.Primitive("libsdca_external_logits")
_external.def_impl(lambda _: np.bool_(True))
_external.def_abstract_eval(lambda _: jax.core.ShapedArray((), np.bool_))


def _external_input(value: Any) -> bool:
    """Only a public entry argument with no donor attribute must keep its original contents."""
    while isinstance(value, ir.OpResult):
        op = value.owner
        if op.name == "stablehlo.optimization_barrier":
            value = op.operands[value.result_number]
        elif op.name == "stablehlo.reshape":
            value = op.operands[0]
        else:
            return False
    if not isinstance(value, ir.BlockArgument):
        return False
    function = value.owner.owner
    if not isinstance(function, func.FuncOp) or function.sym_name.value != "main":
        return False
    attrs = function.attributes
    arg = attrs["arg_attrs"][value.arg_number] if "arg_attrs" in attrs else ()
    return "tf.aliasing_output" not in arg and "jax.buffer_donor" not in arg


def _lower(ctx: Any, value: Any) -> Any:
    return [mlir.ir_constant(np.bool_(_external_input(value)))]


# Recent JAX caches otherwise-identical primitive lowerings in a private function.
# The decision requires the actual argument at each call site.
if "cacheable" in inspect.signature(mlir.register_lowering).parameters:
    mlir.register_lowering(_external, _lower, cacheable=False)
else:
    mlir.register_lowering(_external, _lower)
batching.primitive_batchers[_external] = lambda xs, _: (_external.bind(xs[0]), None)


# JAX 0.4 has no vmap rule for its optimization barrier. Keep the ordering of
# the inputs, whose mapped axes can differ (a batch of logits and one scalar).
barrier = jax.lax.optimization_barrier
if (
    isinstance(batching.primitive_batchers, dict)
    and jax.lax.optimization_barrier_p not in batching.primitive_batchers
):
    barrier = jax.custom_batching.custom_vmap(jax.lax.optimization_barrier)

    @barrier.def_vmap
    def _batch_barrier(size: int, mapped: Any, value: Any) -> Any:
        return barrier(value), mapped[0]


def select(scores: jax.Array, call: Any) -> Any:
    if not isinstance(scores, jax.core.Tracer):
        return call(False)
    external = _external.bind(jax.lax.stop_gradient(scores))
    return jax.lax.cond(external, lambda: call(False), lambda: call(True))
