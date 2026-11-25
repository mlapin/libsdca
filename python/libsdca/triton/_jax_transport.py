"""Abstractly check the named pointer and mutable-reference contract of jax-triton."""

from typing import Any

import jax
import jax.numpy as jnp
import triton
import triton.language as tl


@triton.jit
def _copy(source, destination):
    tl.store(destination, tl.load(source))


def compatible(triton_call: Any) -> bool:
    """Trace on abstract inputs only. Ref writes must survive binding by kernel parameter name."""
    ref_api = getattr(jax, "ref", None)
    core_api = getattr(jax, "core", None)
    if ref_api is None or core_api is None:
        return False
    try:
        shape = jax.ShapeDtypeStruct((1,), jnp.int32)
        reference = ref_api.AbstractRef(core_api.ShapedArray(shape.shape, shape.dtype))

        def write(source, destination):
            destination[...] = source

        def named(source, destination):
            # Reverse keyword order to exercise the binding required by the shared launch plan.
            triton_call(
                kernel=_copy, out_type=[], grid=(1,), destination=destination, source=source
            )

        required = jax.make_jaxpr(write)(shape, reference).jaxpr.effects
        actual = jax.make_jaxpr(named)(shape, reference).jaxpr
    except (AttributeError, ImportError, NotImplementedError, TypeError, ValueError):
        return False
    effects = {
        type(effect)
        for equation in actual.eqns
        if tuple(equation.invars) == tuple(actual.invars)
        for effect in equation.effects
    }
    return bool(required) and all(type(effect) in effects for effect in required)
