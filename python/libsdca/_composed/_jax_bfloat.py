"""A differentiable BF16/FP64 cast that preserves representable subnormals on XLA."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core
from jax.interpreters import ad, batching, mlir


def _convert(x: jax.Array) -> jax.Array:
    if x.dtype == jnp.bfloat16:
        bits = lax.bitcast_convert_type(x, jnp.uint16)
        magnitude = bits & 0x7FFF
        tiny = magnitude < 128
        # Raise subnormals into the normal BF16 range before the native widening.
        # Subtract the implicit leading one in double and restore signed zero.
        normalized = magnitude | (tiny.astype(jnp.uint16) << 7)
        wide = lax.bitcast_convert_type(normalized, jnp.bfloat16).astype(jnp.float64)
        result = wide - jnp.where(tiny, 2.0**-126, 0.0)
        signed = lax.bitcast_convert_type(result, jnp.uint64) | (
            (bits & 0x8000).astype(jnp.uint64) << 48
        )
        return lax.bitcast_convert_type(signed, jnp.float64)
    # Drop 45 of the 52 fraction bits, rounding ties to the even BF16 mantissa.
    bits = lax.bitcast_convert_type(x, jnp.uint64)
    sign = (bits >> 48).astype(jnp.uint16) & 0x8000
    magnitude = bits & 0x7FFFFFFFFFFFFFFF
    rounded = (magnitude + ((1 << 44) - 1) + ((magnitude >> 45) & 1)) >> 45
    normal = jnp.minimum(rounded - (896 << 7), 0x7F80).astype(jnp.uint16)
    normal = jnp.where(jnp.isnan(x), jnp.uint16(0x7FC0), normal)
    # Around 2**-81, one binary64 ulp equals one BF16 subnormal. The addition
    # rounds its low bits to the required integer without a float-to-int conversion.
    tiny = lax.bitcast_convert_type(jnp.abs(x) + 2.0**-81, jnp.uint64).astype(jnp.uint16)
    result = sign | jnp.where(magnitude < (897 << 52), tiny, normal)
    return lax.bitcast_convert_type(result, jnp.bfloat16)


def _abstract(x: Any) -> Any:
    return x.update(dtype=jnp.dtype(jnp.float64 if x.dtype == jnp.bfloat16 else jnp.bfloat16))


_cast = core.Primitive("libsdca_bfloat16_float64_cast")
_cast.def_impl(jax.jit(lambda x: _cast.bind(x)))
_cast.def_abstract_eval(_abstract)


# Other platforms retain XLA's conversion. Select the CPU algorithm at lowering
# time, including when a traced program is compiled for a nondefault device.
def _native(x: jax.Array) -> jax.Array:
    dtype = jnp.float64 if x.dtype == jnp.bfloat16 else jnp.bfloat16
    return lax.convert_element_type(x, dtype)


mlir.register_lowering(_cast, mlir.lower_fun(_native, multiple_results=False))
mlir.register_lowering(_cast, mlir.lower_fun(_convert, multiple_results=False), platform="cpu")
batching.defvectorized(_cast)
ad.deflinear(_cast, lambda cotangent: (_cast.bind(cotangent),))


def cast(x: jax.Array) -> jax.Array:
    return _cast.bind(x)
