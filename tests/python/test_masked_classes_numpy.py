"""Masked classes (scores of -inf; docs/API.md, Masked classes) in the NumPy interface beyond
the rule table (test_rule_table.py runs its rows): rows independent of their neighbours, and
temperatures near the top of the double range."""

import libsdca
import masked_cases as mc
import named_losses
import numpy as np
import pytest


def test_rows_are_independent():
    """A masked row beside finite ones: the shared CPU kernels refuse masked classes, and the
    call takes the native library's rows (as the PyTorch layer does), each row independent of
    the others; the finite rows agree with the kernels' alone."""
    rng = np.random.default_rng(5)
    for batch in mc.batches(limits=False):
        if batch.logits.shape[1] != 6:
            continue
        loss = named_losses.numpy_loss(batch.loss, **batch.options)
        finite = rng.standard_normal(batch.logits.shape)
        z = np.concatenate([finite, batch.logits, finite])
        target = np.concatenate([batch.target] * 3)
        values, gradients = libsdca.loss_and_grad(z, target, loss)
        n = len(batch.cases)
        # The native rows of the finite ones: beside the masked rows once.
        native, native_gradients = libsdca.loss_and_grad(z[: 2 * n], target[: 2 * n], loss)
        for rows in (slice(0, n), slice(2 * n, 3 * n)):
            assert (values[rows] == native[:n]).all()
            assert (gradients[rows] == native_gradients[:n]).all()
        alone, alone_gradients = libsdca.loss_and_grad(finite, batch.target, loss)
        np.testing.assert_allclose(native[:n], alone, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(native_gradients[:n], alone_gradients, rtol=1e-12, atol=1e-15)
        assert np.allclose(values[n : 2 * n], batch.values, rtol=1e-12, atol=1e-12)


# A masked row at temperatures near the top of the double range: below about 8.98e304 the
# loss is that of the classes left (log 2 here); above, 2000 T overflows, no finite logit
# stands for the masked class and the row is invalid (it once recursed without end).
HUGE_TEMPERATURES_REASON = (
    "[core] the limit of a masked distillation row where 2000 T overflows (now: loss log 2)"
)
TRACKED = pytest.mark.xfail(strict=True, reason=HUGE_TEMPERATURES_REASON)
HUGE_TEMPERATURES = [
    (8e304, True),
    (8.98e304, True),
    pytest.param(9e304, False, marks=TRACKED),
    pytest.param(1e305, False, marks=TRACKED),
    pytest.param(1e306, False, marks=TRACKED),
]


@pytest.mark.parametrize(("temperature", "valid"), HUGE_TEMPERATURES)
@pytest.mark.parametrize("name", ["Distillation", "ProximalDistillation"])
def test_huge_temperatures(name: str, temperature: float, valid: bool):
    z = np.array([[0.0, 1.0, -np.inf]])
    loss = getattr(libsdca, name)(temperature=temperature)
    if not valid:
        with pytest.raises(ValueError, match="must be finite"):
            libsdca.loss_and_grad(z, z, loss)
        return
    values, gradients = libsdca.loss_and_grad(z, z, loss)
    assert values[0] == pytest.approx(np.log(2), abs=1e-12)
    assert gradients[0, 2] == 0 and not np.signbit(gradients[0, 2])
