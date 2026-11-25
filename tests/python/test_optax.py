"""libsdca.optax against optax 0.2.9: the same values, output dtypes and gradients, the
hinge's kinks included, across dtypes, shapes and axes; jit, grad and vmap; and the masked
cases where libsdca gives the limit instead of optax's NaN (docs/JAX.md)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")
optax_losses = pytest.importorskip("optax.losses")
import jax.numpy as jnp  # noqa: E402
import libsdca.optax as S  # noqa: E402

ox: Any = optax_losses

DTYPES = (jnp.float32, jnp.float64, jnp.bfloat16)
# (shape, axis): the class axis or axes, and the batch the labels take.
LAYOUTS = (((2, 3), -1), ((4, 7), 1), ((3, 2, 5), -1), ((5, 3, 2), 0), ((2, 3, 4), (1, 2)))


@pytest.fixture(scope="module", autouse=True)
def enable_x64() -> Iterator[None]:
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _tolerance(dtype: Any) -> float:
    return {jnp.float64: 1e-12, jnp.float32: 2e-6}.get(dtype, 3e-2)


def _close(a: Any, b: Any, dtype: Any) -> None:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    tol = _tolerance(dtype)
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol)


def _batch(shape: tuple[int, ...], axis: Any) -> tuple[int, ...]:
    axes = {a % len(shape) for a in ((axis,) if isinstance(axis, int) else axis)}
    return tuple(n for i, n in enumerate(shape) if i not in axes)


def _classes(shape: tuple[int, ...], axis: Any) -> int:
    return int(np.prod(shape)) // int(np.prod(_batch(shape, axis)))


def _inputs(shape: tuple[int, ...], axis: Any, dtype: Any, seed: int = 0) -> tuple[Any, ...]:
    rng = np.random.default_rng(seed)
    z = jnp.asarray(2 * rng.normal(size=shape), dtype)
    y = jnp.asarray(rng.integers(0, _classes(shape, axis), _batch(shape, axis)))
    p = jax.nn.softmax(jnp.asarray(rng.normal(size=shape)), axis=axis).astype(dtype)
    return z, y, p


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: jnp.dtype(d).name)
@pytest.mark.parametrize(("shape", "axis"), LAYOUTS, ids=str)
def test_cross_entropies_match_optax(shape: tuple[int, ...], axis: Any, dtype: Any):
    z, y, p = _inputs(shape, axis, dtype)
    for ours, theirs, target in (
        (
            S.softmax_cross_entropy_with_integer_labels,
            ox.softmax_cross_entropy_with_integer_labels,
            y,
        ),
        (S.softmax_cross_entropy, ox.softmax_cross_entropy, p),
    ):
        a, b = ours(z, target, axis=axis), theirs(z, target, axis=axis)
        assert (a.shape, a.dtype) == (b.shape, b.dtype)
        _close(a, b, dtype)
        ga = jax.grad(lambda z: ours(z, target, axis=axis).astype(jnp.float32).sum())(z)
        gb = jax.grad(lambda z: theirs(z, target, axis=axis).astype(jnp.float32).sum())(z)
        assert ga.dtype == gb.dtype
        _close(ga, gb, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: jnp.dtype(d).name)
def test_soft_targets_match_optax(dtype: Any):
    z, _, p = _inputs((4, 6), -1, dtype, seed=1)
    p = p * 1.5  # optax does not normalize the targets
    for ours, theirs in (
        (S.softmax_cross_entropy, ox.softmax_cross_entropy),
        (S.safe_softmax_cross_entropy, ox.safe_softmax_cross_entropy),
    ):
        a, b = ours(z, p), theirs(z, p)
        assert a.dtype == b.dtype
        _close(a, b, dtype)
        # optax's labels are differentiable.
        ga = jax.grad(lambda z, p: ours(z, p).astype(jnp.float32).sum(), argnums=(0, 1))(z, p)
        gb = jax.grad(lambda z, p: theirs(z, p).astype(jnp.float32).sum(), argnums=(0, 1))(z, p)
        for x, w in zip(ga, gb, strict=True):
            assert x.dtype == w.dtype
            _close(x, w, dtype)
    # Mixed dtypes and integer (one-hot) labels promote as optax's.
    for labels in (p.astype(jnp.float32), jax.nn.one_hot(jnp.arange(4), 6, dtype=jnp.int32)):
        a, b = S.softmax_cross_entropy(z, labels), ox.softmax_cross_entropy(z, labels)
        assert a.dtype == b.dtype
        _close(a, b, dtype)


def test_every_axis():
    z, _, p = _inputs((3, 4), (0, 1), jnp.float64)
    a, b = S.softmax_cross_entropy(z, p, axis=None), ox.softmax_cross_entropy(z, p, axis=None)
    assert a.shape == b.shape == ()
    _close(a, b, jnp.float64)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: jnp.dtype(d).name)
@pytest.mark.parametrize("shape", ((1, 2), (16, 5), (64, 33)), ids=str)
def test_multiclass_hinge_matches_optax(shape: tuple[int, ...], dtype: Any):
    s, y, _ = _inputs(shape, -1, dtype, seed=2)
    a, b = S.multiclass_hinge_loss(s, y), ox.multiclass_hinge_loss(s, y)
    assert (a.shape, a.dtype) == (b.shape, b.dtype)
    _close(a, b, dtype)
    ga = jax.grad(lambda s: S.multiclass_hinge_loss(s, y).sum())(s)
    gb = jax.grad(lambda s: ox.multiclass_hinge_loss(s, y).sum())(s)
    _close(ga, gb, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: jnp.dtype(d).name)
def test_hinge_kinks_match_optax(dtype: Any):
    # Ties of the largest margin (two and three competitors), margins of exactly zero (alone,
    # tied with each other), and both: the mean of the one-sided subgradients, as optax.
    s = jnp.array(
        [
            [0.0, 0.5, 0.5, -3.0],
            [0.0, 2.0, 2.0, 2.0],
            [1.0, 0.0, -5.0, -5.0],
            [1.0, 0.0, 0.0, -5.0],
            [2.0, 1.0, 1.0, 1.0],
        ],
        dtype,
    )
    y = jnp.array([0, 0, 0, 0, 0])
    ga = jax.grad(lambda s: S.multiclass_hinge_loss(s, y).sum())(s)
    gb = jax.grad(lambda s: ox.multiclass_hinge_loss(s, y).sum())(s)
    _close(ga, gb, dtype)


def test_transformations():
    z, y, p = _inputs((6, 5), -1, jnp.float32, seed=3)
    batched = (jnp.stack([z, -z, 2 * z]), jnp.stack([y, y, y]), jnp.stack([p, p, p]))
    for ours, theirs, target, tb in (
        (
            S.softmax_cross_entropy_with_integer_labels,
            ox.softmax_cross_entropy_with_integer_labels,
            y,
            batched[1],
        ),
        (S.softmax_cross_entropy, ox.softmax_cross_entropy, p, batched[2]),
        (S.multiclass_hinge_loss, ox.multiclass_hinge_loss, y, batched[1]),
    ):
        _close(jax.jit(ours)(z, target), theirs(z, target), jnp.float32)
        _close(jax.vmap(ours)(batched[0], tb), jax.vmap(theirs)(batched[0], tb), jnp.float32)
        f = jax.jit(jax.grad(lambda z: ours(z, target).sum()))
        g = jax.grad(lambda z: theirs(z, target).sum())
        _close(f(z), g(z), jnp.float32)
    # jit with a traced where.
    where = jnp.arange(6) % 2 == 0
    f = jax.jit(S.softmax_cross_entropy_with_integer_labels)
    _close(
        f(z, y, where=where),
        ox.softmax_cross_entropy_with_integer_labels(z, y, where=where),
        jnp.float32,
    )


def test_example_where_matches_optax():
    z, y, p = _inputs((5, 4), -1, jnp.float64, seed=4)
    where = jnp.array([True, False, True, False, True])
    for ours, theirs, target in (
        (
            S.softmax_cross_entropy_with_integer_labels,
            ox.softmax_cross_entropy_with_integer_labels,
            y,
        ),
        (S.softmax_cross_entropy, ox.softmax_cross_entropy, p),
    ):
        a, b = ours(z, target, where=where), theirs(z, target, where=where)
        _close(a, b, jnp.float64)
        assert np.all(np.asarray(a)[~np.asarray(where)] == 0)
        ga = jax.grad(lambda z: ours(z, target, where=where).sum())(z)
        gb = jax.grad(lambda z: theirs(z, target, where=where).sum())(z)
        _close(ga, gb, jnp.float64)
    # Excluded examples may hold any labels.
    bad = y.at[1].set(-1)
    assert float(S.softmax_cross_entropy_with_integer_labels(z, bad, where=where)[1]) == 0


def test_class_where():
    z, y, p = _inputs((4, 5), -1, jnp.float64, seed=5)
    where = jnp.ones((4, 5), bool).at[:, 4].set(False).at[3].set(False)
    y = y.at[:].set(jnp.array([0, 1, 2, 4]))  # row 3: no class left
    p = p.at[:3, 4].set(0.0)
    # Labels and targets on the classes kept: optax's values and gradients.
    for ours, theirs, target in (
        (
            S.softmax_cross_entropy_with_integer_labels,
            ox.softmax_cross_entropy_with_integer_labels,
            y,
        ),
        (S.softmax_cross_entropy, ox.softmax_cross_entropy, p),
    ):
        a, b = ours(z, target, where=where), theirs(z, target, where=where)
        _close(a, b, jnp.float64)
        assert float(a[3]) == 0
        ga = jax.grad(lambda z: ours(z, target, where=where)[:3].sum())(z)
        gb = jax.grad(lambda z: theirs(z, target, where=where)[:3].sum())(z)
        _close(ga, gb, jnp.float64)
        assert np.all(np.asarray(ga)[:, 4] == 0)
    # A label or target mass on an excluded class: +inf (optax leaves it out of the sum).
    one = jnp.ones((1, 5), bool).at[0, 4].set(False)
    assert (
        float(S.softmax_cross_entropy_with_integer_labels(z[:1], jnp.array([4]), where=one)[0])
        == np.inf
    )
    mass = jnp.full((1, 5), 0.2)
    assert float(S.softmax_cross_entropy(z[:1], mass, where=one)[0]) == np.inf


def test_rows_without_classes_match_optax():
    """A row that where leaves no class: optax's value, signed zero included (0.0 for labels,
    -0.0 for probabilities), with a zero logits gradient; on a class axis other than the last
    too."""
    z, y, p = _inputs((3, 4, 5), 1, jnp.float64, seed=6)
    where = jnp.ones((3, 4, 5), bool).at[1, :, 2].set(False)
    for ours, theirs, target in (
        (
            S.softmax_cross_entropy_with_integer_labels,
            ox.softmax_cross_entropy_with_integer_labels,
            y,
        ),
        (S.softmax_cross_entropy, ox.softmax_cross_entropy, p),
    ):
        a, b = ours(z, target, axis=1, where=where), theirs(z, target, axis=1, where=where)
        _close(a, b, jnp.float64)
        empty, expected = np.asarray(a)[1, 2], np.asarray(b)[1, 2]
        assert empty == expected == 0 and np.signbit(empty) == np.signbit(expected)
        ga = jax.grad(lambda z: ours(z, target, axis=1, where=where).sum())(z)
        assert np.all(np.asarray(ga)[1, :, 2] == 0)
        _close(ga, jax.grad(lambda z: theirs(z, target, axis=1, where=where).sum())(z), jnp.float64)


def test_masked_logits_give_the_limit():
    z = jnp.array([[1.0, -jnp.inf, 0.5, -1.0], [0.3, 0.2, -jnp.inf, 2.0]], jnp.float64)
    kept = jnp.isfinite(z)
    y = jnp.array([0, 3])
    t = jnp.array([[0.7, 0.0, 0.3, 0.0], [0.1, 0.4, 0.0, 0.5]], jnp.float64)
    # Integer labels: as optax, finite with a +0.0 gradient at the masked class.
    a = S.softmax_cross_entropy_with_integer_labels(z, y)
    _close(a, ox.softmax_cross_entropy_with_integer_labels(z, y), jnp.float64)
    # No target mass at -inf: optax's softmax_cross_entropy gives NaN, libsdca the loss on
    # the other classes, which is optax's safe and where forms.
    assert np.all(np.isnan(np.asarray(ox.softmax_cross_entropy(z, t))))
    for ours in (S.softmax_cross_entropy, S.safe_softmax_cross_entropy):
        a = ours(z, t)
        _close(a, ox.safe_softmax_cross_entropy(z, t), jnp.float64)
        _close(a, ox.softmax_cross_entropy(jnp.where(kept, z, 0), t, where=kept), jnp.float64)
        g = np.asarray(jax.grad(lambda z: ours(z, t).sum())(z))
        assert np.all(g[~np.asarray(kept)] == 0) and not np.signbit(g[~np.asarray(kept)]).any()
        gb = jax.grad(lambda z: ox.safe_softmax_cross_entropy(z, t).sum())(z)
        _close(g, gb, jnp.float64)
    # Target mass at -inf: +inf, as optax's safe form.
    heavy = t.at[0, 1].set(0.2)
    assert float(S.softmax_cross_entropy(z, heavy)[0]) == np.inf
    # The hinge removes a masked class (optax: NaN).
    s = jnp.array([[0.2, -jnp.inf, 0.9, 0.1]], jnp.float64)
    assert np.isnan(float(ox.multiclass_hinge_loss(s, jnp.array([0]))[0]))
    a = S.multiclass_hinge_loss(s, jnp.array([0]))
    _close(a, ox.multiclass_hinge_loss(s[:, [0, 2, 3]], jnp.array([0])), jnp.float64)


def test_errors_are_libsdca_jax_errors():
    z = jnp.zeros((2, 3), jnp.int32)
    with pytest.raises(TypeError, match="logits must be floating-point"):
        S.softmax_cross_entropy_with_integer_labels(z, jnp.array([0, 1]))
    with pytest.raises(ValueError, match="axis 2 is out of range for 2-D logits"):
        S.softmax_cross_entropy(jnp.zeros((2, 3)), jnp.zeros((2, 3)), axis=2)
    with pytest.raises(ValueError, match="class probabilities must have the logits' shape"):
        S.softmax_cross_entropy(jnp.zeros((2, 3)), jnp.zeros((2, 4)))
    with pytest.raises(ValueError, match="repeats an axis"):
        S.softmax_cross_entropy(jnp.zeros((2, 3)), jnp.zeros((2, 3)), axis=(1, -1))


def test_empty_batch_with_multiple_class_axes():
    z = jnp.empty((0, 2, 3))
    p = S.softmax_cross_entropy(z, z, axis=(1, 2))
    y = jnp.empty((0,), dtype=jnp.int32)
    labels = S.softmax_cross_entropy_with_integer_labels(z, y, axis=(1, 2))
    assert p.shape == labels.shape == (0,)


def test_integer_labels_reject_probability_targets():
    z = jnp.zeros((2, 3))
    with pytest.raises(TypeError, match="integer dtype"):
        S.softmax_cross_entropy_with_integer_labels(z, jnp.full_like(z, 1 / 3))
    with pytest.raises(ValueError, match="labels must have shape"):
        S.softmax_cross_entropy_with_integer_labels(z, jnp.ones(z.shape, dtype=jnp.int32))
