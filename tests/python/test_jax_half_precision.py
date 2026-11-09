"""float16/bfloat16 logits in libsdca.jax against high-precision references.

References are computed at the represented 16-bit inputs: the float64 native losses
(libsdca.loss_and_grad, within 64 float64 eps of 100-digit values, numerics.md) and the
120- and 80-digit distillation references of test_distillation_references. The envelope is
the PyTorch one of numerics.md: one rounding to the output dtype, plus (float32 work) 100
float32 eps of max(|loss|, 1) and 2,000 float32 eps of the row's largest gradient entry,
plus one subnormal quantum of the gradient dtype. float64 work (precision="float64") keeps
64 float64 eps of each budget instead. float32 work sums the rows in float32: reduced values
also get 2 + log2(N) float32 roundings of the sum of the rows' magnitudes. float64 work sums
them in float64 and rounds once.
"""

from typing import Any

import jax
import jax.numpy as jnp
import libsdca
import numpy as np
import pytest
from libsdca.jax import distillation_loss, proximal_distillation_loss, sdca_loss
from test_distillation_references import proximal_reference, reference

jax.config.update("jax_enable_x64", True)

HALF = (jnp.float16, jnp.bfloat16)
EPS32, EPS64 = float(np.finfo(np.float32).eps), float(np.finfo(np.float64).eps)
VALUE_EPS, ROW_EPS = 100, 2000
WIDE_EPS = 64
# (loss, target kind, k, gamma): every loss family, top-k with k > 1, smooth and nonsmooth.
CASES = [
    ("cross_entropy", "index", 1, 0.0),
    ("cross_entropy", "mask", 1, 0.0),
    ("topk_cross_entropy", "index", 1, 0.0),
    ("topk_cross_entropy", "index", 3, 0.0),
    ("multiclass_hinge", "index", 1, 0.0),
    ("multiclass_hinge", "index", 1, 0.5),
    ("topk_hinge_alpha", "index", 3, 0.0),
    ("topk_hinge_alpha", "index", 3, 0.5),
    ("topk_hinge_beta", "index", 3, 0.0),
    ("topk_hinge_beta", "index", 3, 0.5),
    ("multilabel_hinge", "mask", 1, 0.0),
    ("multilabel_hinge", "mask", 1, 0.5),
    ("multilabel_cross_entropy", "mask", 1, 0.0),
    ("proximal_cross_entropy", "index", 1, 0.0),
    ("proximal_cross_entropy", "mask", 1, 0.0),
]
REDUCTIONS = ("none", "sum", "mean")
# Small rows (every reduction), and a batch whose mean gradient entries reach float16's
# subnormals (reduction mean).
SHAPES = {(6, 37): REDUCTIONS, (512, 64): ("mean",)}


def _targets(kind: str, n: int, c: int, rng: np.random.Generator) -> np.ndarray:
    labels = rng.integers(0, c, n)
    if kind == "index":
        return labels
    mask = rng.random((n, c)) < 0.3
    mask[np.arange(n), labels] = True
    mask[np.arange(n), (labels + 1) % c] = False
    return mask


def _upstream(reduction: str, n: int) -> np.ndarray:
    """The cotangent: per-row weights (none) or a scalar."""
    return np.linspace(0.5, 2.0, n) if reduction == "none" else np.array(1.5)


def _reduced(values: np.ndarray, gradients: np.ndarray, reduction: str) -> tuple[Any, Any, float]:
    """float64 reference rows reduced and scaled by the cotangent, and the rounding budget of
    a float32 sum of float32 rows (0 for reduction none)."""
    n = values.shape[0]
    u = _upstream(reduction, n)
    if reduction == "none":
        return values, gradients * u[:, None], 0.0
    divisor = n if reduction == "mean" else 1
    rounding = (2 + np.log2(n)) * EPS32 * np.abs(values).sum() / divisor
    return values.sum() / divisor, gradients * u / divisor, float(rounding)


def _run(f: Any, z: jax.Array, reduction: str) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    value, vjp = jax.vjp(f, z)
    (gradient,) = vjp(jnp.asarray(_upstream(reduction, z.shape[0]), value.dtype))
    return np.asarray(value, np.float64), np.asarray(gradient, np.float64), value, gradient


def assert_envelope(
    z: jax.Array,
    value: jax.Array,
    gradient: jax.Array,
    ref_value: np.ndarray,
    ref_gradient: np.ndarray,
    reduced: float,
    precision: str | None,
    message: str,
) -> None:
    assert value.dtype == jnp.float32, message
    assert gradient.dtype == z.dtype, message
    wide = precision == "float64"
    if wide:
        reduced = 0.0
    value_work = WIDE_EPS * EPS64 if wide else VALUE_EPS * EPS32
    row_work = WIDE_EPS * EPS64 if wide else ROW_EPS * EPS32
    out, dtype = np.finfo(np.float32), jnp.finfo(z.dtype)
    v, r = np.asarray(value, np.float64), np.asarray(ref_value, np.float64)
    bound = out.eps * np.abs(r) + value_work * np.maximum(np.abs(r), 1) + reduced + out.tiny
    assert np.all(np.abs(v - r) <= bound), f"{message} value {v} vs {r}"
    g, rg = np.asarray(gradient, np.float64), ref_gradient
    big = np.abs(rg).max(-1, keepdims=True)
    quantum = float(dtype.tiny) * float(dtype.eps)
    bound = float(dtype.eps) * np.abs(rg) + row_work * big + quantum
    excess = np.abs(g - rg) - bound
    assert excess.max() <= 0, f"{message} gradient: {np.abs(g - rg).max()} over the envelope"


class JaxHalfPrecisionTests:
    @pytest.mark.parametrize("dtype", HALF, ids=lambda dtype: np.dtype(dtype).name)
    @pytest.mark.parametrize("part", CASES, ids=lambda case: "-".join(map(str, case)))
    def test_every_loss_against_float64(
        self, part: tuple[Any, ...], dtype: Any, subtests: pytest.Subtests
    ):
        ran = False
        rng = np.random.default_rng(0)
        for (n, c), reductions in SHAPES.items():
            for loss, kind, k, gamma in CASES:
                target = _targets(kind, n, c, rng)
                source = 3 * rng.normal(size=(n, c))
                if (loss, kind, k, gamma) != part:
                    continue
                ran = True
                z = jnp.asarray(source, dtype)
                represented = np.asarray(z, np.float64)
                values, gradients = libsdca.loss_and_grad(
                    represented, target, loss, k=k, gamma=gamma
                )
                for backend in ("jax", "native"):
                    for precision in (None, "float64"):
                        for reduction in reductions:
                            message = (
                                f"{n}x{c} {loss} {kind} k={k} gamma={gamma} {np.dtype(dtype).name}"
                                f" {backend} {precision} {reduction}"
                            )
                            with subtests.test(message):

                                def f(x: jax.Array, **kw: Any) -> jax.Array:
                                    return sdca_loss(
                                        x,
                                        target,
                                        loss,
                                        k=k,
                                        gamma=gamma,
                                        backend=backend,
                                        precision=precision,
                                        reduction=reduction,
                                    )

                                *_, value, gradient = _run(jax.jit(f), z, reduction)
                                expected = _reduced(values, gradients, reduction)
                                assert_envelope(z, value, gradient, *expected, precision, message)
        assert ran, part

    @pytest.mark.parametrize("name", ["distillation", "proximal"])
    @pytest.mark.parametrize("part", HALF, ids=lambda dtype: np.dtype(dtype).name)
    def test_distillation_against_mpmath(self, part: Any, name: str, subtests: pytest.Subtests):
        ran = False
        rng = np.random.default_rng(1)
        source = 3 * rng.normal(size=(6, 9))
        teacher_source = source + rng.normal(size=source.shape)
        temperature, alpha = 2.0, 0.7
        for dtype in HALF:
            if dtype != part:
                continue
            ran = True
            z, t = jnp.asarray(source, dtype), jnp.asarray(teacher_source, dtype)
            s64, t64 = np.asarray(z, np.float64), np.asarray(t, np.float64)
            if name == "distillation":
                rows = reference(s64, t64, temperature)
            else:
                rows = proximal_reference(s64, t64, temperature, alpha)
            for precision in (None, "float64"):
                for reduction in REDUCTIONS:
                    message = f"{name} {np.dtype(dtype).name} {precision} {reduction}"
                    with subtests.test(message):

                        def f(x: jax.Array, **kw: Any) -> jax.Array:
                            if name == "distillation":
                                return distillation_loss(
                                    x, t, temperature, reduction=reduction, precision=precision
                                )
                            return proximal_distillation_loss(
                                x, t, temperature, alpha, reduction=reduction,
                                precision=precision,
                            )  # fmt: skip

                        *_, value, gradient = _run(jax.jit(f), z, reduction)
                        expected = _reduced(*rows, reduction)
                        assert_envelope(z, value, gradient, *expected, precision, message)
        assert ran, part

    @pytest.mark.parametrize("dtype", HALF, ids=lambda dtype: np.dtype(dtype).name)
    def test_loss_scaling_keeps_scaled_gradients(self, dtype: Any, subtests: pytest.Subtests):
        """A gradient below the 16-bit range survives a large cotangent: the gradient of
        scale * loss is the float32 gradient of the represented logits, rounded once."""
        z = jnp.asarray([[0.0, -20.0], [0.0, -24.0]], dtype)
        t = jnp.asarray([[0.0, -19.0], [0.0, -22.0]], dtype)
        losses = {
            "cross_entropy": lambda x: sdca_loss(x, jnp.array([0, 0]), "cross_entropy"),
            "distillation": lambda x: distillation_loss(x, t.astype(x.dtype)),
            "proximal": lambda x: proximal_distillation_loss(x, t.astype(x.dtype), 1.0, 0.5),
        }
        quantum = float(jnp.finfo(dtype).tiny) * float(jnp.finfo(dtype).eps)
        for name, loss in losses.items():
            for jit in (False, True):
                with subtests.test(f"{name} jit={jit}"):
                    grad = jax.grad(lambda x, f=loss: 10000 * f(x))
                    got = (jax.jit(grad) if jit else grad)(z)
                    wide = np.asarray(grad(z.astype(jnp.float32)), np.float64)
                    expected = np.asarray(jnp.asarray(wide, dtype), np.float64)
                    assert got.dtype == dtype
                    assert np.all(np.abs(expected) > 0), expected
                    error = np.abs(np.asarray(got, np.float64) - expected)
                    assert np.all(error <= float(jnp.finfo(dtype).eps) * np.abs(wide) + quantum)
