"""The rules of _frontend that every path follows: the divisor of a mean (the extension's C++
copy, kept for speed, gives the same), the NaN rule of class weights and the value dtype."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import libsdca.jax as J  # noqa: E402
import libsdca.torch as L  # noqa: E402
from libsdca import _frontend as _f  # noqa: E402
from libsdca._composed import xp_jax, xp_torch  # noqa: E402

jax.config.update("jax_enable_x64", True)

RNG = np.random.default_rng(0)
Z = RNG.normal(size=(7, 5))
W = RNG.uniform(0.1, 2.0, 5)
# Labels with an ignored row and a row out of range (clamped into the classes).
Y = np.array([0, 2, 4, 2, 9, 1, 3])
CASES = [{"weight": True}, {"ignore_index": 2}, {"weight": True, "ignore_index": 2}]


def _options(case: dict, convert: Any) -> dict:
    out: dict[str, Any] = {}
    if case.get("weight"):
        out["weight"] = convert(W)
    if "ignore_index" in case:
        out["ignore_index"] = case["ignore_index"]
    return out


@pytest.mark.parametrize("case", CASES, ids=str)
def test_the_extension_divides_as_the_rule(case: dict):
    options = _options(case, torch.tensor)
    z, y = torch.tensor(Z), torch.tensor(Y)
    with pytest.raises(IndexError, match="Target 9 is out of bounds"):
        L.cross_entropy(z, y, **options)
    weight = None if "weight" not in options else _f.nan_weights(xp_torch, options["weight"])
    divisor = _f.mean_divisor(xp_torch, y, weight, options.get("ignore_index"), True)
    # The divisor clamps a label outside the classes before the loss reports it.
    expected = W[[0, 2, 4, 2, 4, 1, 3]] if "weight" in options else np.ones(7)
    if "ignore_index" in options:
        expected = np.where(Y == 2, 0.0, expected)
    assert float(divisor) == expected.sum()
    valid = slice(0, 4)
    m = L.cross_entropy(z[valid], y[valid], **options)
    s = L.cross_entropy(z[valid], y[valid], reduction="sum", **options)
    d = _f.mean_divisor(xp_torch, y[valid], weight, options.get("ignore_index"), True)
    assert torch.equal(m, s / d)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_frameworks_and_backends_divide_alike(case: dict):
    valid = slice(0, 4)
    found = []
    for backend in ("auto", "composed"):
        found.append(float(L.cross_entropy(torch.tensor(Z[valid]), torch.tensor(Y[valid]),
                                           backend=backend, **_options(case, torch.tensor))))  # fmt: skip
        found.append(float(J.cross_entropy(jnp.asarray(Z[valid]), jnp.asarray(Y[valid]),
                                           backend=backend, **_options(case, jnp.asarray))))  # fmt: skip
    np.testing.assert_allclose(found, found[0], rtol=1e-15)


@pytest.mark.parametrize("reduction", ("none", "mean", "sum"))
def test_float64_probabilities_promote(reduction: Literal["none", "mean", "sum"]):
    """float64 probabilities of float32 logits give float64 losses, as PyTorch and optax; the
    logits' gradient keeps their dtype."""
    z = Z.astype(np.float32)
    p = RNG.dirichlet(np.ones(5), 7)
    expected = torch.nn.functional.cross_entropy(
        torch.tensor(z), torch.tensor(p), reduction=reduction
    )
    for backend in ("auto", "composed"):
        zt, pt = torch.tensor(z, requires_grad=True), torch.tensor(p, requires_grad=True)
        value = L.cross_entropy(zt, pt, reduction=reduction, backend=backend)
        value.sum().backward()
        assert value.dtype == torch.float64 and zt.grad is not None and pt.grad is not None
        assert (zt.grad.dtype, pt.grad.dtype) == (torch.float32, torch.float64)
        np.testing.assert_allclose(value.detach().numpy(), expected.numpy(), rtol=1e-6)
        found = J.cross_entropy(
            jnp.asarray(z), jnp.asarray(p), reduction=reduction, backend=backend
        )
        assert found.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(found), expected.numpy(), rtol=1e-6)


def test_nan_weights_and_value_dtype():
    for xp, convert in ((xp_torch, torch.tensor), (xp_jax, jnp.asarray)):
        good = np.asarray(_f.nan_weights(xp, convert(W)))
        np.testing.assert_array_equal(good, W)
        for bad in (np.inf, -1.0, np.nan):
            w = W.copy()
            w[1] = bad
            assert np.all(np.isnan(np.asarray(_f.nan_weights(xp, convert(w)))))
    for module in (torch, jnp):
        assert _f.value_dtype(module, module.float64) == module.float64
        for dtype in (module.float32, module.float16, module.bfloat16):
            assert _f.value_dtype(module, dtype) == module.float32
