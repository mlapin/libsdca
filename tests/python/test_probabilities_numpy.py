"""cross_entropy of probability targets through the C ABI (libsdca.loss_and_grad with
Probabilities): accuracy against the mpmath reference, and the targets it takes."""

import warnings

import libsdca
import numpy as np
import pytest
from probability_references import Family, families, gradient_ulps, loss_ulps, reference

PLAIN = [f for f in families() if not f[3]]


@pytest.mark.parametrize(
    ("dtype", "precision"), [("float64", None), ("float32", None), ("float32", "float64")]
)
@pytest.mark.parametrize("family", PLAIN, ids=lambda f: f[0])
def test_accuracy_against_mpmath(family: Family, dtype: str, precision: str | None):
    """Losses within 8 ulps of the work (float32 work for float32 scores, as the precision
    takes it), gradients within 8 ulps of the largest term of their row, and of its largest
    entry where nothing cancels below the targets' rounding (confident rows included)."""
    name, z, t, _ = family
    z = z.astype(dtype).astype(np.float64)
    values, gradients = libsdca.loss_and_grad(
        z.astype(dtype), libsdca.Probabilities(t), libsdca.CrossEntropy(), precision=precision
    )
    want, want_gradient, terms = reference(z, t)
    work = "float32" if dtype == "float32" and precision is None else "float64"
    assert loss_ulps(values, want, work) <= 8
    g = gradients.astype(np.float64)
    assert gradient_ulps(g, want_gradient, terms, dtype) <= 8
    if name != "softmax":
        largest = np.abs(want_gradient).max(1, keepdims=True)
        assert gradient_ulps(g, want_gradient, largest, dtype) <= 8


def test_targets_need_not_sum_to_one():
    z = np.array([[0.5, -1.0, 2.0], [1.0, 1.0, -3.0]])
    p = np.array([[0.25, 0.25, 0.5], [0.0, 0.5, 0.0]])
    values, gradients = libsdca.loss_and_grad(z, libsdca.Probabilities(p), libsdca.CrossEntropy())
    twice, twice_gradients = libsdca.loss_and_grad(
        z, libsdca.Probabilities(2 * p), libsdca.CrossEntropy()
    )
    np.testing.assert_allclose(twice, 2 * values, rtol=1e-15)
    np.testing.assert_allclose(twice_gradients, 2 * gradients, rtol=1e-15, atol=1e-17)
    zero, zero_gradient = libsdca.loss_and_grad(
        z, libsdca.Probabilities(np.zeros((2, 3))), libsdca.CrossEntropy()
    )
    assert (zero == 0).all() and not np.signbit(zero).any()
    assert (zero_gradient == 0).all() and not np.signbit(zero_gradient).any()


# The cross-entropy kernels still mark rows of negative or infinite targets invalid (the
# target check of include/sdca/loss/ce.h).
_KERNEL_TARGET_CHECK = pytest.mark.xfail(
    strict=True, reason="the kernels take negative and infinite targets as invalid rows"
)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(-0.25, marks=_KERNEL_TARGET_CHECK),
        np.nan,
        pytest.param(np.inf, marks=_KERNEL_TARGET_CHECK),
    ],
)
def test_loss_evaluation_takes_targets_as_given(bad: float):
    # As PyTorch's cross_entropy: sum_j p_j (log sum_k exp(z_k) - z_j), log 3 sum_j p_j at
    # zero logits, NaN for a NaN target; only training requires distributions.
    p = np.array([[0.5, 0.5, 0.0]])
    p[0, 2] = bad
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", libsdca.InvalidRowsWarning)
        values, _ = libsdca.loss_and_grad(
            np.zeros((1, 3)), libsdca.Probabilities(p), libsdca.CrossEntropy()
        )
    np.testing.assert_array_equal(values, [p.sum() * np.log(3)])


def test_probabilities_belong_to_cross_entropy():
    p = libsdca.Probabilities(np.full((1, 3), 1 / 3))
    with pytest.raises(TypeError, match="Probabilities are targets of CrossEntropy"):
        libsdca.loss_and_grad(np.zeros((1, 3)), p, libsdca.TopKCrossEntropy(k=2))


def test_training_still_needs_distributions():
    x = np.eye(3)
    with pytest.raises(ValueError, match="Target probability rows must sum to one"):
        libsdca.fit(x, libsdca.Probabilities(np.full((3, 3), 0.5)), libsdca.CrossEntropy())
