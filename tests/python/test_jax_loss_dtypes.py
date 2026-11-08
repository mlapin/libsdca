"""One dtype rule for every JAX loss: values promote(logits, float32), gradients logits."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike
from libsdca.jax import cross_entropy, distillation_loss, proximal_distillation_loss, sdca_loss

jax.config.update("jax_enable_x64", True)

DTYPES = (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64)
REDUCTIONS = ("none", "sum", "mean")
# (name, loss, target kind, keyword options): every loss, nonsmooth and smoothed.
CASES = [
    ("cross_entropy", "cross_entropy", "index", {}),
    ("cross_entropy mask", "cross_entropy", "mask", {}),
    ("topk_cross_entropy k=2", "topk_cross_entropy", "index", {"k": 2}),
    ("multiclass_hinge", "multiclass_hinge", "index", {}),
    ("multiclass_hinge gamma", "multiclass_hinge", "index", {"gamma": 0.5}),
    ("topk_hinge_alpha", "topk_hinge_alpha", "index", {"k": 2}),
    ("topk_hinge_beta gamma", "topk_hinge_beta", "index", {"k": 2, "gamma": 0.5}),
    ("multilabel_hinge", "multilabel_hinge", "mask", {}),
    ("multilabel_hinge gamma", "multilabel_hinge", "mask", {"gamma": 0.5}),
    ("multilabel_cross_entropy", "multilabel_cross_entropy", "mask", {}),
    ("proximal_cross_entropy", "proximal_cross_entropy", "index", {"alpha": 0.5}),
]


class JaxLossDtypeTests:
    def setup_method(self) -> None:
        self.rng = np.random.default_rng(0)

    def inputs(
        self, kind: str, dtype: DTypeLike, n: int = 5, c: int = 7
    ) -> tuple[jax.Array, jax.Array]:
        z = jnp.asarray(self.rng.standard_normal((n, c)), dtype)
        if kind == "index":
            return z, jnp.asarray(self.rng.integers(0, c, n))
        mask = self.rng.random((n, c)) < 0.4
        mask[:, 0], mask[:, 1] = True, False
        return z, jnp.asarray(mask)

    def check(
        self, fn: Callable[[jax.Array, str], jax.Array], z: jax.Array, reduction: str, message: str
    ) -> None:
        value = fn(z, reduction)
        assert value.dtype == jnp.promote_types(z.dtype, jnp.float32), message
        assert value.shape == ((z.shape[0],) if reduction == "none" else ()), message
        gradient = jax.grad(lambda x: fn(x, reduction).sum())(z)
        assert gradient.dtype == z.dtype, message
        assert bool(jnp.isfinite(value).all()), message

    def test_every_loss_dtype_and_reduction(self):
        for dtype in DTYPES:
            for name, loss, kind, options in CASES:
                z, target = self.inputs(kind, dtype)
                for reduction in REDUCTIONS:
                    self.check(
                        lambda x, r, t=target, o=options, n=loss: sdca_loss(
                            x, t, n, reduction=r, **o
                        ),
                        z,
                        reduction,
                        f"{dtype} {name} {reduction}",
                    )

    @pytest.mark.parametrize("part", DTYPES, ids=lambda dtype: np.dtype(dtype).name)
    def test_wrapper_and_teacher_losses(self, part: DTypeLike):
        ran = False
        for dtype in DTYPES:
            z, labels = self.inputs("index", dtype)
            teacher = jnp.asarray(self.rng.standard_normal(z.shape), dtype)
            if dtype != part:
                continue
            ran = True
            cases = {
                "cross_entropy": lambda x, r: cross_entropy(x, labels, reduction=r),
                "distillation": lambda x, r: distillation_loss(x, teacher, 2.0, reduction=r),
                "proximal_distillation": lambda x, r: proximal_distillation_loss(
                    x, teacher, 2.0, 0.5, reduction=r
                ),
            }
            for name, fn in cases.items():
                for reduction in REDUCTIONS:
                    self.check(fn, z, reduction, f"{dtype} {name} {reduction}")
        assert ran, part
