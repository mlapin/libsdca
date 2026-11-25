"""Use a joint row value and unit gradient only when the gradient remains live."""

from __future__ import annotations

from typing import Any

import jax
import jax.core
import numpy as np
from jax.extend import core
from jax.interpreters import batching, mlir, partial_eval

# The first and third operands are identical scalars. Their producers are the
# joint row kernel and the forward-only row kernel respectively.
_results = core.Primitive("libsdca_row_results")
_results.multiple_results = True
_results.def_impl(lambda *xs: xs[:2])
_results.def_abstract_eval(lambda *xs: xs[:2])
mlir.register_lowering(_results, lambda ctx, *xs: xs[:2])
batching.primitive_batchers[_results] = lambda xs, dims: (_results.bind(*xs), dims[:2])


def _prune(used: list[bool], *context: Any) -> tuple[list[bool], Any]:
    # JAX 0.12 inserts live reference inputs before the equation.
    equation = context[-1]
    n = len(equation.invars)
    if not any(used):
        return [False] * n, None
    indices = [i for i, live in enumerate(used) if live]
    if n == 3 and indices == [0]:
        indices = [2]
    inputs = [equation.invars[i] for i in indices]
    outputs = [v for v, live in zip(equation.outvars, used, strict=True) if live]
    return [i in indices for i in range(n)], equation.replace(invars=inputs, outvars=outputs)


_rules = getattr(partial_eval, "dce_rules", None)
available = isinstance(_rules, dict)
if available:
    _rules[_results] = _prune


def select(value: Any, gradient: Any, separate: Any) -> tuple[Any, Any]:
    first, second = _results.bind(value, gradient, separate)
    return first, second


def unit_literal(upstream: Any) -> bool:
    """A unit cotangent visible while tracing, without reading a device value."""
    get_const = getattr(upstream, "get_const", None)
    if get_const is None:
        return False
    value = get_const()
    return (
        isinstance(value, float | np.number | np.ndarray)
        and np.ndim(value) == 0
        and bool(value == 1)
    )


def outer_trace(scores: Any) -> bool:
    # Eager reverse mode wraps a concrete primal; an enclosing jit traces it.
    return isinstance(getattr(scores, "primal", scores), jax.core.Tracer)
