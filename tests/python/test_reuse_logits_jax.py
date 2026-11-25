"""The CPU kernels of libsdca.jax write the gradient into the logits' buffer where XLA can spare
it (input_output_aliases): values and gradients equal those without the alias, logits that are
also returned stay intact, and dead logits take no second buffer."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import libsdca.jax as J
import pytest
from libsdca import _jax_ops

if _jax_ops._handler()[0] is None:
    pytest.skip(f"no FFI handlers: {_jax_ops._handler()[1]}", allow_module_level=True)

N, D, C = 64, 32, 512
CPU = jax.devices("cpu")[0]
with jax.default_device(CPU):
    W = jax.random.normal(jax.random.PRNGKey(0), (D, C), dtype=jnp.float32)
    X = jax.random.normal(jax.random.PRNGKey(1), (N, D), dtype=jnp.float32)
    Y = jax.random.randint(jax.random.PRNGKey(2), (N,), 0, C)
    MASK = jax.random.bernoulli(jax.random.PRNGKey(3), 0.1, (N, C)).at[:, 0].set(True)
    TEACHER = jax.random.normal(jax.random.PRNGKey(4), (N, C), dtype=jnp.float32)

Loss = Callable[[jax.Array, jax.Array], jax.Array]
Results = dict[str, tuple[tuple[jax.Array, ...], int]]
LOSSES: dict[str, tuple[Loss, jax.Array]] = {
    "cross_entropy": (J.cross_entropy, Y),
    "cross_entropy_none": (lambda z, t: J.cross_entropy(z, t, reduction="none").sum(), Y),
    "topk_hinge": (lambda z, t: J.topk_hinge_loss(z, t, k=3), Y),
    "proximal_cross_entropy": (J.proximal_cross_entropy, Y),
    "multilabel_cross_entropy": (J.multilabel_cross_entropy, MASK),
    "distillation": (J.distillation_loss, TEACHER),
}


@pytest.fixture
def unaliased() -> Iterator[None]:
    ffi_call = jax.ffi.ffi_call

    def call(*args: Any, input_output_aliases: Any = None, **kwargs: Any) -> Any:
        return ffi_call(*args, **kwargs)

    jax.clear_caches()
    with mock.patch.object(jax.ffi, "ffi_call", call):
        yield
    jax.clear_caches()


def _results(loss: Loss, target: jax.Array) -> tuple[jax.Array, ...]:
    def dead(w: jax.Array) -> jax.Array:
        return loss(X @ w, target)

    def returned(w: jax.Array) -> tuple[jax.Array, jax.Array]:
        logits = X @ w
        return loss(logits, target), logits

    value, grad = jax.jit(jax.value_and_grad(dead))(W)
    (value2, logits), grad2 = jax.jit(jax.value_and_grad(returned, has_aux=True))(W)
    mapped = jax.jit(jax.vmap(jax.grad(loss)))(jnp.stack([X @ W, X @ -W]), jnp.stack([target] * 2))
    return value, grad, value2, logits, grad2, mapped


def _temp(loss: Loss, target: jax.Array) -> int:
    def dead(w: jax.Array) -> jax.Array:
        return loss(X @ w, target)

    memory = jax.jit(jax.grad(dead)).lower(W).compile().memory_analysis()
    assert memory is not None
    return memory.temp_size_in_bytes


@pytest.fixture(scope="module")
def aliased() -> Iterator[Results]:
    with jax.default_device(CPU):
        jax.clear_caches()
        yield {name: (_results(*case), _temp(*case)) for name, case in LOSSES.items()}


@pytest.mark.parametrize("name", list(LOSSES))
def test_alias_changes_no_result_and_spares_a_buffer(name: str, aliased: Results, unaliased: None):
    got, temp = aliased[name]
    want = _results(*LOSSES[name])
    for a, b in zip(got, want, strict=True):
        assert jnp.array_equal(a, b, equal_nan=True)
    assert jnp.array_equal(got[3], X @ W)
    assert temp + N * C * 4 <= _temp(*LOSSES[name])
