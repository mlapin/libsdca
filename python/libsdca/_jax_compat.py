"""Transformable optimization fences on every supported JAX version."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
from jax import lax, tree_util
from jax._src.interpreters import ad as internal_ad
from jax.extend import core
from jax.interpreters import ad, batching, mlir

_T = TypeVar("_T")
_NATIVE = lax.optimization_barrier_p


def _legacy_barrier() -> core.Primitive:
    # JAX 0.4.35 lowers the fence but has no differentiation/batching rules. Keep
    # those rules local to our primitive rather than modifying JAX's registry.
    primitive = core.Primitive("sdca_optimization_barrier")
    primitive.multiple_results = True
    primitive.def_impl(lambda *args: lax.optimization_barrier(args))
    primitive.def_abstract_eval(lambda *avals: avals)
    mlir.register_lowering(
        primitive,
        mlir.lower_fun(lambda *args: lax.optimization_barrier(args), multiple_results=True),
    )

    def nonzero(values: Any) -> list[Any]:
        present = [not isinstance(value, ad.Zero) for value in values]
        live = [value for value, exists in zip(values, present, strict=False) if exists]
        result = iter(primitive.bind(*live) if live else ())
        return [
            next(result) if exists else value
            for value, exists in zip(values, present, strict=False)
        ]

    def jvp(primals: Any, tangents: Any) -> tuple[Any, Any]:
        return primitive.bind(*primals), nonzero(tangents)

    def transpose(cotangents: Any, *args: Any) -> list[Any]:
        return nonzero(cotangents)

    ad.primitive_jvps[primitive] = jvp
    ad.primitive_transposes[primitive] = transpose
    batching.primitive_batchers[primitive] = lambda args, dims: (primitive.bind(*args), dims)
    return primitive


def _native_rules() -> bool:
    # Newer JAX moves transpose and batching rules into richer registries; its
    # public primitive_batchers proxy intentionally supports writes only.
    def contains(module: Any, name: str) -> bool:
        registry = getattr(module, name, {})
        return isinstance(registry, Mapping) and _NATIVE in registry

    return (
        contains(ad, "primitive_jvps")
        and (contains(ad, "primitive_transposes") or contains(internal_ad, "fancy_transposes"))
        and (
            contains(batching, "primitive_batchers")
            or contains(batching, "fancy_primitive_batchers")
        )
    )


_LEGACY = None if _native_rules() else _legacy_barrier()


def optimization_barrier(operand: _T) -> _T:
    """Preserve XLA fences, including through old JAX grad/vmap transforms."""
    if _LEGACY is None:
        return lax.optimization_barrier(operand)
    leaves, structure = tree_util.tree_flatten(operand)
    return tree_util.tree_unflatten(structure, _LEGACY.bind(*leaves) if leaves else [])


@jax.custom_jvp
def _legacy_nextafter(x: jax.Array, y: jax.Array | float) -> jax.Array:
    return jnp.nextafter(x, y)


@_legacy_nextafter.defjvp
def _nextafter_jvp(primals: Any, tangents: Any) -> tuple[jax.Array, jax.Array]:
    result = _legacy_nextafter(*primals)
    # Match newer JAX: locally the next representable value follows x, while
    # its direction argument selects the neighbour but contributes no tangent.
    return result, jnp.broadcast_to(tangents[0], result.shape)


def nextafter(x: jax.Array, y: jax.Array | float) -> jax.Array:
    """The native operation, with its modern derivative on older JAX."""
    if lax.nextafter_p in ad.primitive_jvps:
        return jnp.nextafter(x, y)
    return _legacy_nextafter(x, y)
