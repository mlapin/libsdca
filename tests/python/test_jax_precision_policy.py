"""The precision policy of libsdca.jax (docs/API.md#precision), with x64 off and on.

Each setting runs in its own process: jax_enable_x64 is global.
"""

from __future__ import annotations

import os
import textwrap

import jax  # noqa: F401  (the jax test group)
import pytest
from child_process import run_python

PRELUDE = """
import jax, numpy as np
from unittest import mock
from libsdca import jax as sdca_jax
from named_losses import jax_loss as named_loss
rng = np.random.default_rng(5)
z = rng.normal(size=(6, 8)).astype(np.float32)
labels = np.array([0, 7, 2, 1, 4, 3])
mask = np.eye(8, dtype=bool)[labels]
mask[:, 5] = True
mask[5, 5] = False
def solve(loss, target, **options):
    f = lambda x: named_loss(x, target, loss, reduction="sum", **options)
    return jax.jit(jax.value_and_grad(f))(z)
def text(loss, target, **options):
    f = lambda x: named_loss(x, target, loss, **options)
    return jax.jit(jax.grad(f)).lower(z).as_text()
"""


def _run(code: str, x64: bool) -> None:
    env = dict(os.environ, JAX_ENABLE_X64="1" if x64 else "0", JAX_PLATFORMS="cpu")
    script = PRELUDE + textwrap.dedent(code) + "\nprint('ok')\n"
    result = run_python("-c", script, capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "ok" in result.stdout


def test_without_x64_defaults_keep_the_native_route():
    """Retained double defaults use native without x64; measured fp32 defaults use XLA.

    Explicit fp32 still needs double entries for the smoothed hinges; explicit float64 needs
    jax_enable_x64.
    """
    _run(
        """
        float32 = [("multiclass_hinge", labels, {}), ("topk_hinge_beta", labels, {"k": 3}),
                   ("multilabel_hinge", mask, {}), ("proximal_cross_entropy", labels, {}),
                   ("topk_cross_entropy", labels, {"k": 3})]
        smooth = [("multiclass_hinge", labels, {"gamma": 1.0}),
                  ("multilabel_hinge", mask, {"gamma": 1.0})]
        with mock.patch.object(sdca_jax, "_accelerator", return_value=True):
            for loss, target, options in float32 + smooth:
                fp32_default = loss in ("proximal_cross_entropy", "topk_cross_entropy")
                assert ("callback" in text(loss, target, **options)) == (not fp32_default), loss
                try:
                    text(loss, target, precision="float64", **options)
                    raise AssertionError("no error")
                except ValueError as error:
                    # float64 work needs jax_enable_x64 on every backend.
                    assert "jax_enable_x64" in str(error)
                single = text(loss, target, precision="float32", **options)
                assert ("callback" in single) == (options.get("gamma", 0) > 0), loss
        for loss, target, options in smooth:
            try:
                named_loss(z, target, loss, backend="jax", precision="float32", **options)
                raise AssertionError("no error")
            except ValueError as error:
                assert "jax_enable_x64" in str(error)
        """,
        x64=False,
    )


def test_without_x64_float32_proximal_rows_beyond_float32_give_nan():
    _run(
        """
        wide = z.copy()
        wide[2, 0], wide[2, 1] = 3e38, -3e38
        options = dict(backend="jax", precision="float32", reduction="none")
        values = np.asarray(named_loss(wide, labels, "proximal_cross_entropy", **options))
        assert np.isnan(values[2]) and np.isfinite(np.delete(values, 2)).all(), values
        # The hinges round such rows correctly in float32: the native float64 results.
        for loss in ("multiclass_hinge", "topk_hinge_alpha"):
            k = 2 if "topk" in loss else 1
            values = named_loss(wide, labels, loss, k=k, **options)
            native = named_loss(wide, labels, loss, k=k, backend="native", reduction="none")
            values, native = np.asarray(values), np.asarray(native)
            assert values[2] == native[2], (loss, values[2], native[2])
            np.testing.assert_allclose(values, native, rtol=64 * 2**-23, err_msg=loss)
        """,
        x64=False,
    )


@pytest.mark.parametrize(
    "loss,options",
    [
        ("multiclass_hinge", {}),
        ("topk_hinge_alpha", {"k": 3}),
        ("multilabel_hinge", {}),
        ("topk_cross_entropy", {"k": 3}),
        ("multiclass_hinge", {"gamma": 1.0}),
    ],
)
def test_with_x64_default_is_unchanged_float64(loss: str, options: dict[str, object]):
    _run(
        f"""
        if {loss!r} == "topk_cross_entropy":
            z = rng.normal(size=(6, 65)).astype(np.float32)  # wider retained double default
        target = mask if "multilabel" in {loss!r} else labels
        for backend in ("auto", "jax"):
            a = solve({loss!r}, target, backend=backend, **{options!r})
            b = solve({loss!r}, target, backend=backend, precision="float64", **{options!r})
            for x, y in zip(a, b):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        """,
        x64=True,
    )


def test_with_x64_float32_smoothed_hinges_form_entries_in_float64():
    _run(
        """
        for loss, target, options in [("multiclass_hinge", labels, {"gamma": 0.5}),
                                      ("topk_hinge_beta", labels, {"k": 3, "gamma": 0.5}),
                                      ("multilabel_hinge", mask, {"gamma": 0.5})]:
            a = solve(loss, target, backend="jax", **options)
            b = solve(loss, target, backend="jax", precision="float32", **options)
            for x, y in zip(a, b):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        """,
        x64=True,
    )
