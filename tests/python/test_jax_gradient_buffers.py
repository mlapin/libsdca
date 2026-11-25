"""The JAX gradient kernel can use either a separate or a donated buffer."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from libsdca import _jax_alias
from libsdca._frontend import Rows
from libsdca.jax import cross_entropy


def test_entry_logits_ownership() -> None:
    def marker(z: jax.Array) -> jax.Array:
        return _jax_alias.select(z, lambda inplace: jnp.full_like(z, int(inplace)))

    x = jnp.zeros((4, 5))
    np.testing.assert_array_equal(jax.jit(marker)(x), 0)
    np.testing.assert_array_equal(jax.jit(marker, donate_argnums=(0,))(x.copy()), 1)
    np.testing.assert_array_equal(jax.jit(lambda z: marker(z * 2))(x), 1)
    np.testing.assert_array_equal(jax.jit(jax.vmap(marker))(x), 0)


def test_alias_preserves_values_and_gradients() -> None:
    x = jnp.arange(60, dtype=jnp.float32).reshape(2, 5, 6) / 4
    y = jnp.array([[0, 0, 4, 2, 5], [1, 4, 3, 2, 2]], jnp.int32)
    step = jax.vmap(jax.value_and_grad(cross_entropy))
    expected = jax.jit(step)(x, y)
    donated = jax.jit(step, donate_argnums=(0,))(x.copy(), y)
    nested = jax.jit(lambda z, t: jax.jit(step)(z, t))(x, y)
    for output in (donated, nested):
        for reference, actual in zip(expected, output, strict=True):
            np.testing.assert_array_equal(actual, reference)
    gradient = jax.jit(jax.vmap(jax.grad(cross_entropy)))(x, y)
    np.testing.assert_array_equal(gradient, expected[1])


def test_saved_gradient_uses_current_cotangent() -> None:
    # The unit gradient rounds below the float16 range. Applying the cotangent before
    # the final cast recovers the small gradient with its correct sign.
    x = jnp.full((8, 1000), -20, jnp.float16).at[:, 0].set(0)
    y = jnp.zeros(8, jnp.int32)
    cotangents = jnp.asarray([1, 8192, -4096, 0, 1], jnp.float32)

    def gradients(z: jax.Array, weights: jax.Array) -> jax.Array:
        _, back = jax.vjp(lambda v: cross_entropy(v, y), z)
        return jax.vmap(lambda w: back(w)[0])(weights)

    expected = gradients(x.astype(jnp.float32), cotangents).astype(x.dtype)
    assert float(expected[0, 0, 1]) == 0 and float(expected[1, 0, 1]) > 0
    np.testing.assert_array_equal(gradients(x, cotangents), expected)
    np.testing.assert_array_equal(jax.jit(gradients)(x, cotangents), expected)


def test_public_reduced_loss_sees_caller_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    # Expose the selected layout through the saved gradient. A cache placed between
    # the caller and the marker would instead choose the aliased kernel in both calls.
    from libsdca import _jax_ops

    dispatch = _jax_ops._dispatch

    def marked(
        rows: Rows, gradient: bool, save: bool, both: bool, reduction: str, inplace: bool
    ) -> Any:
        kernel = dispatch(rows, gradient, save, both, reduction, inplace)
        if not both:
            return kernel

        def run(*args: jax.Array) -> Any:
            value, saved = kernel(*args)
            return value, jnp.full_like(saved, int(inplace))

        return run

    monkeypatch.setattr(_jax_ops, "_dispatch", marked)
    jax.clear_caches()
    try:
        with jax.default_device(jax.devices("cpu")[0]):
            x = jnp.ones((5, 13), dtype=jnp.float32)
            y = jnp.arange(5, dtype=jnp.int32)
            step = jax.value_and_grad(cross_entropy)
            np.testing.assert_array_equal(jax.jit(step)(x, y)[1], 0)
            np.testing.assert_array_equal(jax.jit(step, donate_argnums=(0,))(x.copy(), y)[1], 1)
    finally:
        jax.clear_caches()


def test_short_eager_kernel_cache_matches_transformed_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    from libsdca import jax as api
    from libsdca.jax import distillation_loss, topk_cross_entropy

    scores = jnp.arange(56, dtype=jnp.float32).reshape(8, 7) / 17
    teacher = jnp.flip(scores, axis=1)
    labels = jnp.arange(8, dtype=jnp.int32) % 7
    weights = jnp.arange(1, 8, dtype=jnp.float32)

    def weighted(z: jax.Array) -> jax.Array:
        return cross_entropy(z, labels, weight=weights, ignore_index=2, label_smoothing=0.125)

    def topk(z: jax.Array) -> jax.Array:
        return topk_cross_entropy(z, labels, k=3, weight=weights)

    def distill(z: jax.Array) -> jax.Array:
        return distillation_loss(z, teacher, temperature=1.25)

    for loss in (weighted, topk, distill):
        eager_value, back = jax.vjp(loss, scores)
        with monkeypatch.context() as m:
            m.setattr(api, "_eager_loss", api._evaluate)
            reference, previous_back = jax.vjp(loss, scores)
        np.testing.assert_array_equal(eager_value, reference)
        _, staged_back = jax.vjp(jax.jit(loss), scores)
        for factor in (1.0, -0.75, 0.0):
            upstream = jnp.asarray(factor, scores.dtype)
            gradient = back(upstream)[0]
            np.testing.assert_array_equal(gradient, previous_back(upstream)[0])
            np.testing.assert_array_equal(gradient, staged_back(upstream)[0])


def test_cpu_retained_pullback_after_gradient_donation() -> None:
    with jax.default_device(jax.devices("cpu")[0]):
        x = jnp.arange(128 * 17, dtype=jnp.float32).reshape(128, 17) / 256
        y = jnp.arange(128) % 17
        unit, negative = jnp.float32(1), jnp.float32(-2)
        _, pullback = jax.vjp(lambda z: cross_entropy(z, y), x)
        (first,) = pullback(unit)
        (second,) = pullback(unit)
        reference = np.asarray(first).copy()
        jax.jit(lambda g: g + 1, donate_argnums=(0,))(first).block_until_ready()
        np.testing.assert_array_equal(second, reference)
        for factor in (negative, unit, jnp.float32(0), unit):
            (actual,) = pullback(factor)
            np.testing.assert_array_equal(actual, reference * float(factor))


def test_input_arrays_follow_the_active_x64_setting() -> None:
    from libsdca import jax as api

    original = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", True)
        doubles = jnp.asarray([1.5], dtype=jnp.float64)
        labels = jnp.asarray([1], dtype=jnp.int64)
        assert api._asarray(doubles) is doubles
        assert api._asarray(labels) is labels
        jax.config.update("jax_enable_x64", False)
        for value in (doubles, labels):
            expected = jnp.asarray(value)
            actual = api._asarray(value)
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
    finally:
        jax.config.update("jax_enable_x64", original)
