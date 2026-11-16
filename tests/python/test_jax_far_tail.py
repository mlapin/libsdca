"""The GPU paths (forced) and the CPU paths of the JAX entropy, distillation and proximal
distillation keep a row's far mass: float32 rows [0, -gap, ..., -gap] whose competitors lie
beyond the float32 normal range (exp(-87.3)) while their sum over 1,000 or 32,000 classes is
normal.

XLA:CPU flushes those exponentials to zero, as the GPU rounds exponentials below exp(-103.97)
to zero, so without the far-tail rescan the loss and the target entry are zero there
(_FAR_TAIL = "off" checks that the cases discriminate where the device flushes). With each
other _FAR_TAIL they are within 64 float32 eps of 50-digit references; XLA:CPU flushes the
subnormal results of the GPU's failing gaps (104 at 1,000 classes), which only the GPU probe
can check.
"""

import functools
import os
from collections.abc import Callable, Iterator
from unittest import mock

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from jax.typing import DTypeLike
from libsdca import jax as sdca_jax
from libsdca.jax import (
    cross_entropy,
    distillation_loss,
    multilabel_cross_entropy,
    proximal_distillation_loss,
)
from mpmath_types import MPReal
from test_jax import x64

EPS = float(np.finfo(np.float32).eps)
QUANTUM = float(np.finfo(np.float32).smallest_subnormal)
NORMAL = float(np.finfo(np.float32).tiny)
# (rescaling dtype, jax_enable_x64)
VARIANTS = [("float32", False), ("float32", True), ("float64", True), ("inline", False)]


GAPS = (88.0, 90.0, 92.0, 94.0, 97.0)


@functools.cache
def _flushes() -> bool:
    """Whether the default device flushes float32 exponentials below the normal range."""
    return float(jax.jit(jnp.exp)(jnp.float32(-90.0))) == 0


@pytest.fixture(params=["gpu", "cpu"])
def path(request: pytest.FixtureRequest) -> Iterator[None]:
    """The GPU paths and the CPU paths, traced for the default device."""
    # These are XLA rescan tests; the shared native kernel has its own tail rescue.
    with (
        mock.patch.object(jax, "default_backend", lambda: request.param),
        mock.patch.dict(os.environ, {"SDCA_JAX_FFI": "off"}),
    ):
        yield


def _rows(gaps: list[float], classes: int, positives: int = 1) -> np.ndarray:
    z = np.zeros((len(gaps), classes), np.float32)
    z[:, positives:] = -np.asarray(gaps, np.float32)[:, None]
    return z


@functools.cache
def _entropy_reference(gap: float, classes: int, positives: int) -> tuple[MPReal, MPReal]:
    """Loss log(positives + t) and entry of a positive at the maximum, -t / (positives
    (positives + t)), t = (classes - positives) exp(-gap)."""
    with mp.workdps(50):
        tail = (classes - positives) * mp.exp(-mp.mpf(gap))
        value = mp.log1p(tail) if positives == 1 else mp.log(positives + tail)
        return value, -tail / (positives * (positives + tail))


@functools.cache
def _distillation_reference(gap: float, classes: int, temperature: float) -> tuple[MPReal, MPReal]:
    """Loss and student target entry of student [0, -T gap, ...], teacher [0, -T gap - 1, ...]:
    log1p(r) + gap p_tail and (1 / (1 + r) - 1 / (1 + p)) / T, r and p the tails."""
    with mp.workdps(50):
        rest = (classes - 1) * mp.exp(-mp.mpf(gap))
        teacher = (classes - 1) * mp.exp(-mp.mpf(gap) - 1 / mp.mpf(temperature))
        value = mp.log1p(rest) + gap * teacher / (1 + teacher)
        return value, (1 / (1 + rest) - 1 / (1 + teacher)) / temperature


def _within(actual: float, expected: MPReal, eps: float = 64 * EPS) -> bool:
    with mp.workdps(50):
        return bool(abs(mp.mpf(actual) - expected) <= eps * abs(expected) + QUANTUM)


def _value_and_target(
    loss: Callable[[jax.Array], jax.Array], z: np.ndarray, dtype: DTypeLike = jnp.float32
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row losses and the gradient's first column (the target) of jit(value_and_grad)."""

    def total(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        values = loss(x)
        return values.sum(), values

    (_, values), gradient = jax.jit(jax.value_and_grad(total, has_aux=True))(jnp.asarray(z, dtype))
    return np.asarray(values, np.float64), np.asarray(gradient[:, 0], np.float64)


def _cases() -> list[tuple[str, int]]:
    names = ["labels", "labels bfloat16", "mask", "tied mask", "distillation", "distillation T=2"]
    names += ["proximal distillation", "proximal distillation T=2"]
    return [(name, classes) for name in names for classes in (1000, 32000)]


def _evaluate(name: str, classes: int) -> tuple[list[float], np.ndarray, np.ndarray, list]:
    """Gaps beyond the normal range (whose target entries are normal), losses, target entries
    and their references."""
    temperature = 2.0 if name.endswith("T=2") else 1.0
    positives = 2 if name == "tied mask" else 1

    distillation = "distillation" in name

    def reference(gap: float) -> tuple[MPReal, MPReal]:
        # The proximal correction of these rows is below 2^-70 of their terms at alpha = 1.
        if distillation:
            return _distillation_reference(gap, classes, temperature)
        return _entropy_reference(gap, classes, positives)

    gaps = [g for g in GAPS if abs(reference(g)[1]) > 2 * NORMAL]
    references = [reference(g) for g in gaps]
    rows = len(gaps)
    if distillation:
        student = _rows([temperature * g for g in gaps], classes)
        teacher = _rows([temperature * g + 1 for g in gaps], classes)
        loss = proximal_distillation_loss if name.startswith("proximal") else distillation_loss

        def distill(x: jax.Array) -> jax.Array:
            return loss(x, teacher, temperature=temperature, reduction="none")

        values, targets = _value_and_target(distill, student)
        return gaps, values, targets, references
    z = _rows(gaps, classes, positives)
    if name.startswith("labels"):
        target = jnp.zeros(rows, jnp.int32)
    else:
        mask = np.zeros((rows, classes), bool)
        mask[:, :positives] = True
        target = jnp.asarray(mask)
    dtype = jnp.bfloat16 if name.endswith("bfloat16") else jnp.float32
    loss = cross_entropy if name.startswith("labels") else multilabel_cross_entropy

    def entropy(x: jax.Array) -> jax.Array:
        return loss(x, target, reduction="none")

    values, targets = _value_and_target(entropy, z, dtype)
    return gaps, values, targets, references


@pytest.mark.parametrize(("rescale", "enable_x64"), VARIANTS)
@pytest.mark.parametrize(("name", "classes"), _cases())
def test_far_mass_of_the_jnp_paths(
    path: None, name: str, classes: int, rescale: str, enable_x64: bool, subtests: pytest.Subtests
):
    with x64(enable_x64), mock.patch.object(sdca_jax, "_FAR_TAIL", rescale):
        gaps, values, targets, references = _evaluate(name, classes)
    # bfloat16 logits: a float32 loss and a bfloat16 gradient.
    entry_eps = 64 * EPS + (2.0**-8 if name.endswith("bfloat16") else 0)
    for gap, value, target, (expected, entry) in zip(
        gaps, values, targets, references, strict=True
    ):
        with subtests.test(gap=gap):
            assert _within(value, expected), (value, expected)
            assert _within(target, entry, entry_eps), (target, entry)


@pytest.mark.parametrize(("name", "classes"), _cases())
def test_far_mass_vanishes_without_the_rescan(path: None, name: str, classes: int):
    if not _flushes():
        pytest.skip("the device keeps subnormal float32 exponentials")
    with x64(False), mock.patch.object(sdca_jax, "_FAR_TAIL", "off"):
        _, values, targets, references = _evaluate(name, classes)
    # The tied mask's loss is log 2 either way.
    if name != "tied mask":
        assert not any(_within(v, r[0]) for v, r in zip(values, references, strict=True))
    assert (targets == 0).all()


@pytest.mark.parametrize(("rescale", "enable_x64"), VARIANTS)
def test_ordinary_rows_beside_far_rows_are_unchanged(path: None, rescale: str, enable_x64: bool):
    rng = np.random.default_rng(7)
    ordinary = (4 * rng.standard_normal((6, 1000))).astype(np.float32)
    z = jnp.asarray(np.concatenate([ordinary, _rows([90.0, 92.0], 1000)]))
    labels = jnp.asarray(rng.integers(0, 1000, 8), jnp.int32).at[6:].set(0)
    mask = jnp.arange(1000) == labels[:, None]
    noise = rng.standard_normal(ordinary.shape).astype(np.float32)
    teacher = np.concatenate([ordinary + noise, _rows([91.0, 93.0], 1000)])
    losses = {
        "labels": lambda x: cross_entropy(x, labels, reduction="sum"),
        "mask": lambda x: multilabel_cross_entropy(x, mask, reduction="sum"),
        "distillation": lambda x: distillation_loss(x, teacher, reduction="sum"),
    }
    for name, loss in losses.items():
        results = []
        for tail in (rescale, "off"):
            with x64(enable_x64), mock.patch.object(sdca_jax, "_FAR_TAIL", tail):
                gradient = np.asarray(jax.jit(jax.grad(loss))(z), np.float64)
                results.append((gradient[:6], gradient[6:, 0]))
        (fixed, far), (plain, vanished) = results
        assert (far < 0).all() and (vanished == 0).all() == _flushes(), name
        if name == "distillation" or (name == "labels" and rescale == "inline"):
            # The rescan's conditional changes XLA:CPU's fusions, and its vectorized
            # exponentials, and the inline mass its sum, by an ulp here and there.
            scale = np.abs(plain).max(-1, keepdims=True)
            assert (np.abs(fixed - plain) <= 4 * EPS * scale).all()
        else:
            np.testing.assert_array_equal(fixed, plain)


def test_far_mass_under_vmap_and_value_only(path: None):
    gaps = [90.0, 94.0]
    z, s, t = (
        _rows(gaps, 1000),
        _rows([2 * g for g in gaps], 1000),
        _rows([2 * g + 1 for g in gaps], 1000),
    )
    with x64(False):
        labels = jnp.zeros((2, 1), jnp.int32)

        def row(x: jax.Array, y: jax.Array) -> jax.Array:
            return cross_entropy(x[None], y, reduction="sum")

        mapped = jax.jit(jax.vmap(jax.value_and_grad(row)))(jnp.asarray(z), labels)
        # The value alone (the primal of distillation's custom VJP).
        distilled = jax.jit(lambda x: distillation_loss(x, t, temperature=2.0, reduction="none"))(s)
    for i, gap in enumerate(gaps):
        expected, entry = _entropy_reference(gap, 1000, 1)
        assert _within(float(mapped[0][i]), expected)
        assert _within(float(mapped[1][i, 0]), entry)
        assert _within(float(distilled[i]), _distillation_reference(gap, 1000, 2.0)[0])


@pytest.mark.parametrize("target", ["labels", "mask"])
def test_far_mass_in_second_derivatives(path: None, target: str):
    # The Hessian's target diagonal entry, t / (1 + t)^2, from autodiff of the far mass.
    z = jnp.asarray(_rows([90.0], 1000))
    y = jnp.zeros(1, jnp.int32) if target == "labels" else jnp.arange(1000)[None] == 0
    tangent = jnp.zeros_like(z).at[0, 0].set(1)
    loss = cross_entropy if target == "labels" else multilabel_cross_entropy
    with x64(False):
        column = jax.jit(
            lambda x: jax.jvp(jax.grad(lambda v: loss(v, y, reduction="sum")), (x,), (tangent,))[1]
        )(z)
    with mp.workdps(50):
        tail = 999 * mp.exp(-90)
        assert _within(float(column[0, 0]), tail / (1 + tail) ** 2)


@pytest.mark.parametrize("classes", [5, 1000])
@pytest.mark.parametrize("alpha", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_proximal_far_student_with_disagreeing_teacher(
    path: None, classes: int, alpha: float, temperature: float
):
    # A tiny student tail does not make the proximal envelope equal to cross-entropy.
    # Include a matching teacher in the same batch to exercise both sides of the gate.
    student = _rows([100.0 * temperature] * 2, classes)
    teacher = student.copy()
    teacher[0, :2] = teacher[0, :2][::-1]
    with x64(True):
        z, t = jnp.asarray(student), jnp.asarray(teacher)

        def evaluate(precision: str | None):
            return jax.jit(
                jax.value_and_grad(
                    lambda x: proximal_distillation_loss(
                        x,
                        t,
                        temperature=temperature,
                        alpha=alpha,
                        reduction="sum",
                        precision=precision,
                    )
                )
            )(z)

        value, gradient = evaluate(None)
        reference, expected_gradient = evaluate("float64")
    np.testing.assert_allclose(value, reference, rtol=64 * EPS, atol=0)
    np.testing.assert_allclose(gradient[0, :2], expected_gradient[0, :2], rtol=64 * EPS, atol=0)
