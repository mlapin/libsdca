"""The jax backend of libsdca.jax against the native losses, under jit, grad and vmap."""

import concurrent.futures
import contextlib
import functools
import inspect
import json
import math
import os
import textwrap
import time
from collections.abc import Iterator
from fractions import Fraction
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from child_process import run_python
from jax.typing import ArrayLike
from libsdca import (
    LOSSES,
    _jax_fusion,
    _jax_losses,
    distillation_loss_and_grad,
    loss_and_grad,
    proximal_distillation_loss_and_grad,
)
from libsdca import jax as sdca_jax
from libsdca.jax import sdca_loss
from topk_rows import CAPPED_GAPS, capped_reference, capped_rows

jax.config.update("jax_enable_x64", True)

TENSOR = [objective for objective in LOSSES if objective != "cross_entropy"]
SMOOTH = ["multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge"]
ALPHAS = [1e-300, 1e-18, 1e-6, 1e-2, 0.5, 1.0, 30.0, 1e4, 1e18, 1e300]
# {z0, z1, gamma}: margins 1 + z1 - z0 that cancel, h a overflowing or
# underflowing at huge or tiny gamma, and huge common offsets.
WIDE_PARTS = [(dtype, k) for dtype in ("float64", "float32") for k in (1, 2, 5)]
HINGE_CASES = [
    (1.0, 1e-20, 1e-20),
    (1.0, 2.0**-26, 1.0),
    (0.0, 1e200, 1e300),
    (1.0, 1e-200, 1e-300),
    (1e16, 1e16 + 2, 1.0),
    (1e16, 1e16 - 2, 1.0),
    (-1e16, -1e16 + 2, 1e16),
    (0.0, 0.0, 1e-310),
    (0.0, 0.5, 5e-324),
    (1.0, 0.0, 1e-310),
]


def options_of(objective: str, classes: int) -> list[dict[str, Any]]:
    """The k, gamma and alpha values that matter for an objective."""
    ks = sorted({1, 2, classes // 2, classes - 1} & set(range(1, classes)))
    if objective == "proximal_cross_entropy":
        return [dict(alpha=a) for a in ALPHAS]
    gammas = (0.0, 0.6, 1e-3, 30.0)
    if objective == "multilabel_cross_entropy":
        return [{}]
    if objective in ("multiclass_hinge", "multilabel_hinge"):
        return [dict(gamma=g) for g in gammas]
    if objective == "topk_cross_entropy":
        return [dict(k=k) for k in ks]
    return [dict(k=k, gamma=g) for k in ks for g in gammas]


def targets_of(
    objective: str, examples: int, classes: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Labels and/or masks with a positive and a negative label."""
    labels = rng.integers(0, classes, examples)
    mask = rng.random((examples, classes)) < 0.4
    rows = np.arange(examples)
    mask[rows, labels] = True
    mask[rows, (labels + 1) % classes] = False
    if objective == "proximal_cross_entropy":
        return [labels, mask]
    return [mask] if objective.startswith("multilabel") else [labels]


@contextlib.contextmanager
def traced_for(cpu: bool) -> Iterator[None]:
    """_jax_losses traced as for the CPU or not (mocked), by a new jit of loss_and_gradient: the
    module's own keeps its earlier traces."""
    fresh = jax.jit(
        functools.partial(inspect.unwrap(_jax_losses.loss_and_gradient)),
        static_argnames=("objective", "k", "width", "values_only"),
    )
    with (
        mock.patch.object(_jax_losses, "_cpu", return_value=cpu),
        mock.patch.object(_jax_losses, "loss_and_gradient", fresh),
    ):
        yield


def calls_native(text: str) -> bool:
    """Whether a lowering calls the native library: its FFI targets or a host callback."""
    return "@libsdca_loss" in text or "callback" in text


def run(
    z: np.ndarray,
    target: np.ndarray,
    objective: str,
    backend: str,
    weights: np.ndarray,
    jit: bool = False,
    **options: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-example values and the gradient of their weighted sum."""

    def f(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        per = sdca_loss(x, target, objective, backend=backend, reduction="none", **options)
        return (per * weights).sum(), per

    g = jax.value_and_grad(f, has_aux=True)
    (_, values), gradient = (jax.jit(g) if jit else g)(z)
    return np.asarray(values), np.asarray(gradient)


class JaxBackendTests:
    def check(
        self,
        z: np.ndarray,
        target: np.ndarray,
        objective: str,
        rtol: float,
        atol: float | np.ndarray,
        jit: bool = False,
        **options: Any,
    ) -> None:
        """Values and weighted gradients of the jax backend against loss_and_grad (outside XLA,
        whose CPU threads flush subnormals)."""
        weights = np.linspace(0.5, 2.0, len(z)).astype(z.dtype)
        reference_options = {"precision": "float64", **options}
        values, gradients = loss_and_grad(z, target, objective, **reference_options)
        expected = values.astype(z.dtype), gradients * weights[:, None]
        actual = run(z, target, objective, "jax", weights, jit, **options)
        message = f"{objective} {options} {z.dtype} {np.asarray(target).dtype}"
        atol = np.asarray(atol)
        for a, e, tol in zip(actual, expected, [atol, atol[..., None]], strict=True):
            assert a.dtype == e.dtype, message
            bad = ~(np.abs(a - e) <= tol + rtol * np.abs(e))
            assert not bad.any(), f"{message}: {a[bad][:4]} != {e[bad][:4]}"

    @pytest.mark.parametrize(
        "part", [(o, d) for o in TENSOR for d in ("float64", "float32")], ids=str
    )
    def test_random_scores_match_native(self, part: tuple[str, str]):
        ran = False
        rng = np.random.default_rng(31)
        for dtype, rtol in [(np.float64, 1e-12), (np.float32, 3e-7)]:
            for classes in [2, 3, 7, 40, 300]:
                # Rows at scales 1 and 100 in one batch.
                scale = np.repeat([1.0, 100.0], 24)
                z = (scale[:, None] * rng.normal(size=(48, classes))).astype(dtype)
                for objective in TENSOR:
                    if classes == 300 and objective != "proximal_cross_entropy":
                        continue
                    for target in targets_of(objective, 48, classes, rng):
                        if (objective, np.dtype(dtype).name) != part:
                            continue
                        ran = True
                        for options in options_of(objective, classes):
                            if objective == "proximal_cross_entropy" and dtype == np.float32:
                                # Keep this tight fp64-reference check; conformance separately
                                # checks the faster fp32 default against its work-precision bound.
                                options = {**options, "precision": "float64"}
                            atol = rtol * scale / (options.get("gamma") or 1.0)
                            self.check(z, target, objective, rtol, atol, **options)
        assert ran, part

    @pytest.mark.parametrize("part", WIDE_PARTS, ids=lambda part: f"{part[0]}-k{part[1]}")
    def test_wide_rows_solve_from_candidates_or_fall_back(self, part: tuple[str, int]):
        """Smooth hinges from the largest margins, top-k from candidates in the logits dtype,
        and the full solve where those do not hold (large gamma, integer ties)."""
        ran = False
        rng = np.random.default_rng(13)
        objectives = ["topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge"]
        for dtype, rtol in [(np.float64, 1e-12), (np.float32, 3e-7)]:
            for scores in [3 * rng.normal(size=(32, 300)), rng.integers(-2, 3, (32, 300))]:
                z = scores.astype(dtype)
                labels = rng.integers(0, 300, 32)
                mask = np.eye(300, dtype=bool)[labels] | (rng.random((32, 300)) < 0.01)
                mask[np.arange(32), (labels + 1) % 300] = False
                for k in [1, 2, 5]:
                    if (np.dtype(dtype).name, k) != part:
                        continue
                    ran = True
                    for objective in ["topk_hinge_alpha", "topk_hinge_beta"]:
                        self.check(z, labels, objective, rtol, rtol, jit=True, k=k)
                    for objective in objectives:
                        target = mask if objective.startswith("multilabel") else labels
                        for gamma in [1e-3, 0.5, 30.0]:
                            atol = rtol * 3 / min(gamma, 1.0)
                            self.check(z, target, objective, rtol, atol, k=k, gamma=gamma)
        assert ran, part
        if part != WIDE_PARTS[-1]:
            return
        # The candidates hold small gamma and give the full solve's result.
        z = jnp.asarray(3 * rng.normal(size=(8, 300)))
        positive = jnp.asarray(np.eye(300, dtype=bool)[rng.integers(0, 300, 8)])
        h = _jax_losses._margin(z, (z * positive).sum(-1, keepdims=True))
        gamma = jnp.float64(0.5)
        for objective in ["topk_hinge_alpha", "topk_hinge_beta"]:
            for k in [1, 2]:
                cap = gamma / k
                kind = objective + "_smooth"
                args = (h, positive, z.astype(np.float32), kind, k, gamma, cap)
                *fast, held = _jax_losses._smooth_single(*args, 16)
                *full, _ = _jax_losses._smooth_single(*args, None)
                assert bool(held.all()), objective
                for a, e in zip(fast, full, strict=True):
                    np.testing.assert_allclose(a, e, rtol=1e-14, atol=1e-16, err_msg=objective)

    def test_top_k_from_float32_keys(self):
        """_jax_losses.top_k against a stable sort, where float32 keys tie unequal float64
        entries inside, at and beyond the k-th, and with exact and -inf ties."""
        rng = np.random.default_rng(29)
        tiny = np.array([0.0, 1e-12, 2e-12, -1e-12, 3e-12])
        rows = [
            3 * rng.normal(size=40),
            np.repeat(1.0, 40) + np.resize(tiny, 40),
            rng.integers(-2, 3, 40).astype(np.float64),
            np.r_[np.full(37, -np.inf), 1.0, 1.0 + 1e-13, 2.0],
            np.r_[1e300, -1e300, np.full(38, -np.inf)],
        ]
        x = np.array([r[rng.permutation(40)] for r in rows])
        for k in [1, 2, 5, 38]:
            values, index = jax.jit(lambda v, k=k: _jax_losses.top_k(v, k))(x)
            order = np.argsort(-x, axis=-1, kind="stable")[:, :k]
            np.testing.assert_array_equal(np.asarray(index), order, err_msg=str(k))
            np.testing.assert_array_equal(values, np.take_along_axis(x, order, -1))
        for dtype in (jnp.float32, jnp.bfloat16, jnp.int8):
            v = jnp.asarray(rng.integers(-3, 4, (8, 40)), dtype)
            index = np.asarray(_jax_losses.top_k(v, 5)[1])
            order = np.argsort(-np.asarray(v, np.float64), axis=-1, kind="stable")[:, :5]
            np.testing.assert_array_equal(index, order, err_msg=str(dtype))

    def test_hinge_margins_tied_by_rounding(self):
        """Rounded margin ties match the native losses; top-1 selects the largest score."""
        half = 0.5 - 2.0**-54  # 1 + half - 0 rounds to 1.5 in float64
        rows = [
            [0.0, half, 0.5, -1.0, -2.0, -3.0],
            [0.0, 0.5, half, -1.0, -2.0, -3.0],
            [0.0, 0.25, half, 0.5, half, -3.0],
        ]
        z = np.array(rows)
        labels = np.zeros(len(z), int)
        for objective in ["topk_hinge_alpha", "topk_hinge_beta"]:
            for k in [1, 2, 3]:
                self.check(z, labels, objective, 1e-15, 1e-15, jit=True, k=k)
        # float32 scores beside a far label score: margins in float64 round them together.
        z = np.array([[-1e9, 0.5, np.nextafter(np.float32(0.5), np.float32(0)), 0.25]], np.float32)
        z = np.repeat(z[:, [0, 2, 1, 3]], 2, axis=0)
        for objective in ["topk_hinge_alpha", "topk_hinge_beta"]:
            for k in [1, 2]:
                self.check(z, np.zeros(2, int), objective, 1e-7, 1e-7, jit=True, k=k)

    def test_k1_hinge_orders_scores_before_rounding_margins(self):
        """The largest represented score wins even if a smaller score's margin rounds equal.
        The array implementation uses this rule on CPU and on other devices."""
        far = [[-1e10, -5000.0, 1.0, -7000.0], [0.0, 0.5, -1.0, 0.25]]
        tied = [[-1e10, 0.5, 1.0, -7000.0], [0.0, 0.5, -1.0, 0.25]]
        for rows in (far, tied):
            z = np.array(rows, np.float32)
            labels = np.zeros(2, int)
            for objective in ["topk_hinge_alpha", "topk_hinge_beta"]:

                def loss(x: jax.Array, objective: str = objective) -> jax.Array:
                    return sdca_loss(
                        x,
                        labels,
                        objective,
                        k=1,
                        backend="jax",
                        precision="float32",
                        reduction="sum",
                    )

                expected = np.zeros_like(z)
                expected[:, 0], expected[[0, 1], [2, 1]] = -1, 1
                for cpu in (True, False):
                    with traced_for(cpu):
                        gradient = jax.jit(jax.grad(loss))(z)
                    np.testing.assert_array_equal(gradient, expected, err_msg=f"{objective} {cpu}")

    @pytest.mark.long
    def test_off_the_cpu_without_its_conditionals(self):
        """Off the CPU (mocked), top_k is lax.top_k and the nonsmooth top-k hinges form every
        margin without the CPU's conditionals (one only where float32 keys pick k > 1 from
        wider margins), with the CPU's losses and gradients."""
        rng = np.random.default_rng(37)
        solve = inspect.unwrap(_jax_losses.loss_and_gradient)

        def traced(f: Any, cpu: bool, *args: Any) -> tuple[str, Any]:
            with traced_for(cpu):
                # New functions each time: make_jaxpr and jit trace them afresh.
                text = str(jax.make_jaxpr(lambda *a: f(*a))(*args))
                return text, jax.jit(lambda *a: f(*a))(*args)

        x = rng.normal(size=(8, 300))
        for cpu in (True, False):
            text, (values, index) = traced(lambda v: _jax_losses.top_k(v, 5), cpu, x)
            assert ("cond[" in text) == cpu and ("optimization_barrier" in text) == cpu
            order = np.argsort(-x, axis=-1, kind="stable")[:, :5]
            np.testing.assert_array_equal(index, order)
            np.testing.assert_array_equal(values, np.take_along_axis(x, order, -1))
        labels = rng.integers(0, 300, 32)
        mask = np.eye(300, dtype=bool)[labels] | (rng.random((32, 300)) < 0.01)
        mask[np.arange(32), (labels + 1) % 300] = False
        cases = [
            *[
                (o, k, 1.0, labels)
                for o in ("topk_hinge_alpha", "topk_hinge_beta")
                for k in (1, 2, 5)
            ],
            ("topk_cross_entropy", 2, 1.0, labels),
            ("topk_hinge_beta_smooth", 2, 0.5, labels),
            ("multilabel_hinge_smooth", 1, 0.5, mask),
            ("proximal_cross_entropy", 1, 1.0, mask),
        ]
        for dtype in (np.float32, np.float64):
            for scores in [3 * rng.normal(size=(32, 300)), rng.integers(-2, 3, (32, 300))]:
                z = scores.astype(dtype)
                for objective, k, gamma, target in cases:
                    scalars = np.array([gamma, gamma / k, 1.0, 0.0])
                    width = int(np.asarray(target).sum(-1).max()) if target.ndim == 2 else 1

                    def f(
                        z: jax.Array, t: jax.Array, o: str = objective, k: int = k, s: Any = scalars
                    ) -> Any:
                        return solve(z, t, s, objective=o, k=k, width=width)

                    (text, on), (other, off) = (traced(f, c, z, target) for c in (True, False))
                    message = f"{objective} k={k} {dtype.__name__}"
                    if objective in ("topk_hinge_alpha", "topk_hinge_beta"):
                        # Top-1 selects represented scores directly on both platforms;
                        # only k > 1 keeps the CPU margin-selection conditionals.
                        assert ("cond[" in text) == (k > 1), message
                        assert other.count("cond[") == (k > 1 and dtype == np.float32), message
                    # To the rounding of a sum (jaxlib 0.12 orders the k = 5 sum differently).
                    for a, e in zip(off, on, strict=True):
                        # The k largest margins sum in another order off the CPU: equal to
                        # rounding (an ulp of some float64 values with jaxlib 0.12, none on 0.6).
                        expected = np.asarray(e)
                        eps = 4 * float(np.finfo(expected.dtype).eps)
                        atol = eps * float(np.nanmax(np.abs(expected)))
                        np.testing.assert_allclose(
                            np.asarray(a), expected, rtol=eps, atol=atol, err_msg=message
                        )

    @pytest.mark.parametrize("part", ["float64", "float32"])
    def test_small_rows_and_integer_ties(self, part: str):
        """Every class in the full solve (XLA:CPU once took minutes to compile it from 10
        classes on), and top-k entropy among tied competitors."""
        ran = False
        rng = np.random.default_rng(17)
        for dtype, rtol in [(np.float64, 1e-12), (np.float32, 3e-7)]:
            for classes in [12, 20]:
                for scores in [
                    3 * rng.normal(size=(16, classes)),
                    rng.integers(-2, 3, (16, classes)),
                ]:
                    z = scores.astype(dtype)
                    labels = rng.integers(0, classes, 16)
                    if np.dtype(dtype).name != part:
                        continue
                    ran = True
                    for k in [2, 5]:
                        self.check(z, labels, "topk_cross_entropy", rtol, rtol * 3, jit=True, k=k)
                        for objective in ["topk_hinge_alpha", "topk_hinge_beta"]:
                            self.check(
                                z, labels, objective, rtol, rtol * 3, jit=True, k=k, gamma=0.5
                            )
        assert ran, part

    @pytest.mark.parametrize("part", TENSOR)
    def test_jit_and_vmap(self, part: str):
        ran = False
        rng = np.random.default_rng(7)
        batch = 3 * rng.normal(size=(3, 6, 7))
        masks = None
        for objective in TENSOR:
            options: dict[str, Any] = dict(k=2, gamma=0.6, alpha=0.9)
            labels = rng.integers(0, 7, (3, 6))
            masks = np.stack([targets_of("multilabel_hinge", 6, 7, rng)[0] for _ in range(3)])
            targets = masks if objective.startswith("multilabel") else labels
            if objective != part:
                continue
            ran = True
            self.check(batch[0], targets[0], objective, 1e-12, 1e-13, jit=True, **options)

            def per(
                x: ArrayLike,
                t: ArrayLike,
                objective: str = objective,
                options: dict[str, Any] = options,
            ) -> jax.Array:
                return sdca_loss(x, t, objective, backend="jax", **options)

            for in_axes, target in [((0, 0), targets), ((0, None), targets[0])]:

                def pick(
                    i: int, in_axes: tuple[int, int | None] = in_axes, target: np.ndarray = target
                ) -> np.ndarray:
                    return target[i] if in_axes[1] == 0 else target

                loop = [per(batch[i], pick(i)) for i in range(3)]
                mapped = jax.jit(jax.vmap(per, in_axes=in_axes))(batch, target)
                np.testing.assert_allclose(mapped, loop, rtol=1e-12, err_msg=objective)
                grads = jax.grad(lambda x, a=in_axes, t=target: jax.vmap(per, a)(x, t).sum())
                expected = [jax.grad(per)(batch[i], pick(i)) for i in range(3)]
                np.testing.assert_allclose(jax.jit(grads)(batch), expected, rtol=1e-13)
            nested = jax.jit(jax.vmap(jax.vmap(per)))(
                np.stack([batch] * 2), np.stack([targets] * 2)
            )
            np.testing.assert_allclose(nested[1], jax.vmap(per)(batch, targets), rtol=1e-12)
        assert ran, part
        if part != TENSOR[-1]:
            return
        # Traced masks: proximal_entropy pads its labels to every class.
        assert masks is not None
        z, mask = batch[0], masks[0]
        native = sdca_loss(z, mask, "proximal_cross_entropy", backend="native", reduction="none")
        traced = jax.jit(
            lambda x, t: sdca_loss(x, t, "proximal_cross_entropy", backend="jax", reduction="none")
        )(z, mask)
        np.testing.assert_allclose(traced, native, rtol=1e-13)
        # (examples, classes, height, width) with axis=1.
        z4, l4 = rng.normal(size=(2, 7, 3, 2)), rng.integers(0, 7, (2, 3, 2))
        for objective in ["topk_hinge_beta", "proximal_cross_entropy"]:
            values = [
                sdca_loss(z4, l4, objective, axis=1, k=2, backend=b, reduction="none")
                for b in ("native", "jax")
            ]
            np.testing.assert_allclose(values[1], values[0], rtol=1e-13)

    def test_constant_masks_with_a_moved_class_axis_under_jit(self):
        # jaxlib 0.6's XLA:CPU miscompiled a NumPy mask closed over under jit with axis=0:
        # the smooth multilabel hinge was off by 0.016 of 0.015 here.
        rng = np.random.default_rng(1)
        z = 0.1 * rng.normal(size=(33, 4))
        labels = rng.integers(0, 4, 33)
        mask = np.eye(4, dtype=bool)[labels] | (rng.random((33, 4)) < 0.3)
        mask &= ~np.eye(4, dtype=bool)[(labels + 1) % 4]
        for objective in TENSOR:
            target = mask if objective.startswith("multilabel") else np.eye(4, dtype=bool)[labels]
            for gamma in (0.0, 30.0):
                options: dict[str, Any] = dict(k=2, gamma=gamma, alpha=0.9, reduction="none")
                expected = sdca_loss(z, target, objective, backend="native", **options)
                for t in (target.T, np.ascontiguousarray(target.T)):

                    def f(x: jax.Array, t: np.ndarray = t) -> jax.Array:
                        return sdca_loss(x, t, objective, backend="jax", axis=0, **options)

                    np.testing.assert_allclose(
                        jax.jit(f)(z.T), expected, rtol=1e-13, atol=1e-15, err_msg=objective
                    )

    @pytest.mark.parametrize("part", [o for o in TENSOR if o != "proximal_cross_entropy"])
    def test_ties_select_the_native_subgradients(self, part: str):
        # Integer scores tie often; kinks and degenerate breakpoints must agree.
        ran = False
        rng = np.random.default_rng(4)
        for classes in [3, 6]:
            z = rng.integers(-2, 3, (200, classes)).astype(np.float64)
            for objective in TENSOR:
                if objective == "proximal_cross_entropy":
                    continue
                target = targets_of(objective, 200, classes, rng)[0]
                if objective != part:
                    continue
                ran = True
                for options in options_of(objective, classes):
                    rows = slice(None)
                    if objective == "topk_hinge_alpha":
                        # A mean margin of zero up to rounding admits either subgradient.
                        value = sdca_loss(z, target, objective, reduction="none", **options)
                        rows = np.abs(np.asarray(value)) > 1e-12
                    self.check(z[rows], target[rows], objective, 1e-12, 1e-12, **options)
        assert ran, part

    @pytest.mark.parametrize("part", TENSOR)
    def test_far_offsets_and_cancelling_scores(self, part: str):
        ran = False
        rows = [[1e16] * 3, [1e16, 1e16, 1e16 + 2], [700.0, 0.0, 1.0, 700.0, -0.3, 0.2]]
        for gap in [700.0, 1e16, 1e30]:
            rows += [[-gap, 0.0, 1.0, 2.0, -1.0], [gap, 0.0, 1.0, 2.0, -1.0]]
            rows.append([0.0, 1.0, gap, -0.3, 0.2])
        rows += [
            [0.0, 20.0, 20.0, -100.0],
            [0.0, 10.0, 10.0, 10.0, -100.0],
            [1000.0, 0.0, 0.0, -5.0],
        ]
        # proximal_cross_entropy: the rows of test_tensor_prox.
        prox = rows[:3] + [r for r in rows[3:] if r[1:] == [0.0, 1.0, 2.0, -1.0]]
        prox += [rows[-3], rows[-1]]
        prox += [[0.0, 0.0, 0.0, 0.0], [40.0, 0.0, 0.0, 0.0], [-40.0, 0.0, 0.0, 0.0]]
        # Hinge values of cancelling scores are ill-conditioned; top-k entropy keeps the mean.
        cancelling = [[0.0, 1e16, 1.0, -1e16], [0.0, 1e30, 1.0, -1e30], [0.0, 1e12, 0.0, -1e12]]
        # Rows of a width go in one batch: every row is solved independently.
        hinges = [o for o in TENSOR if o != "proximal_cross_entropy"]
        kinds = {"rows": hinges, "cancelling": ["topk_cross_entropy"]}
        kinds["prox"] = ["proximal_cross_entropy"]
        groups: dict[tuple[int, str], list[list[float]]] = {}
        for kind, members in [("rows", rows), ("cancelling", cancelling), ("prox", prox)]:
            for row in members:
                groups.setdefault((len(row), kind), []).append(row)
        for (classes, kind), group in groups.items():
            z = np.array(group)
            objectives = kinds[kind]
            mask = np.zeros_like(z, dtype=bool)
            mask[:, ::2] = True
            first = np.zeros(len(z), int)
            for objective in objectives:
                if objective != part:
                    continue
                ran = True
                keep = np.ones(len(z), bool)
                if objective == "topk_cross_entropy":
                    # The native top-k entropy loses free scores beside a far one; see below.
                    far = [max(r) >= 1e16 and r[0] < max(r) and r[1] == 1.0 for r in group]
                    keep = ~np.array(far)
                    if not keep.any():
                        continue
                targets = [first]
                if objective.startswith("multilabel"):
                    targets = [mask]
                elif objective == "proximal_cross_entropy":
                    targets = [first, first + 1, mask]
                settings = options_of(objective, classes)
                if objective in SMOOTH:
                    settings += [dict(s, gamma=g) for s in settings[::3] for g in (1e-20, 1e20)]
                for target in targets:
                    for options in settings:
                        self.check(z[keep], target[keep], objective, 1e-12, 1e-13, **options)
        assert ran, part

    @pytest.mark.parametrize("part", [o for o in TENSOR if "hinge" in o])
    def test_hinge_margins_and_smoothing_against_two_class_closed_forms(self, part: str):
        ran = False
        hinges = [o for o in TENSOR if "hinge" in o]
        for dtype in [np.float32, np.float64]:
            eps = float(np.finfo(dtype).eps)
            for z0, z1, smoothing in HINGE_CASES:
                z = np.array([[z0, z1]], dtype)
                if not np.isfinite(z).all():
                    continue
                if smoothing < np.finfo(np.float64).tiny and jax.default_backend() == "cpu":
                    continue  # XLA:CPU reads subnormal operands as zero; see JAX.md.
                h = 1 + Fraction(float(z[0, 1])) - Fraction(float(z[0, 0]))
                for objective, gamma in [(o, g) for o in hinges for g in (0.0, smoothing)]:
                    if objective != part:
                        continue
                    ran = True
                    smooth = gamma > 0
                    width = Fraction(gamma) * (2 if objective.startswith("multilabel") else 1)
                    v = min(Fraction(1), max(Fraction(0), h / width)) if smooth else h > 0
                    value = h * v - width * v * v / 2 if smooth else max(h, 0)
                    targets = [np.array([[True, False]])]
                    if "multilabel" not in objective:
                        targets.append(np.array([0]))
                    for target in targets:
                        actual, gradient = jax.jit(
                            jax.value_and_grad(
                                lambda x, t=target, o=objective, g=gamma: sdca_loss(
                                    x, t, o, gamma=g, backend="jax"
                                )
                            )
                        )(z)
                        message = f"{dtype} {objective} {z} {gamma} {target.dtype}"
                        np.testing.assert_allclose(
                            float(actual), float(value), rtol=2 * eps, atol=0, err_msg=message
                        )
                        np.testing.assert_allclose(
                            gradient[0],
                            [-float(v), float(v)],
                            rtol=2 * eps,
                            atol=0,
                            err_msg=message,
                        )
        assert ran, part

    def test_topk_entropy_free_scores_below_a_far_competitor(self):
        # The gradient does not depend on the gap; the native loss keeps it to 700.
        target = np.array([0])
        for k in [2, 3]:
            near = np.array([[-700.0, 1.0, 700.0, -0.3, 0.2]])
            expected = jax.grad(lambda x, k=k: sdca_loss(x, target, "topk_cross_entropy", k=k))(
                near
            )
            for gap in [1e16, 1e30]:
                z = np.array([[-gap, 1.0, gap, -0.3, 0.2]])
                actual = jax.grad(
                    lambda x, k=k: sdca_loss(x, target, "topk_cross_entropy", k=k, backend="jax")
                )(z)
                np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=0)

    def test_topk_entropy_confident_rows_within_64_eps(self):
        # Rows [0, -m, ..., -m] of label 0 cap no competitor: cross_entropy's loss
        # log1p((C - 1) e^-m) and entries, normal to m = 700. Log-domain odds rounded at the
        # magnitude of m: 154 eps at m = 261.5.
        eps = np.finfo(np.float64).eps
        gaps = 40.0 + 13.25 * np.arange(50)
        for classes, k in [(40, 2), (1000, 5)]:
            z = np.repeat(-gaps[:, None], classes, axis=1)
            z[:, 0] = 0
            target = np.zeros(len(gaps), np.int32)

            def loss(x: jax.Array, k: int = k, target: np.ndarray = target) -> jax.Array:
                return sdca_loss(
                    x, target, "topk_cross_entropy", k=k, backend="jax", reduction="none"
                )

            value = np.asarray(loss(jnp.asarray(z)))
            gradient = np.asarray(jax.grad(lambda x: loss(x).sum())(jnp.asarray(z)))
            for i, gap in enumerate(gaps):
                with mp.workdps(60):
                    e = mp.exp(-mp.mpf(float(gap)))
                    tail = (classes - 1) * e
                    expected = (mp.log1p(tail), -tail / (1 + tail), e / (1 + tail))
                actual = (value[i], gradient[i, 0], gradient[i, -1])
                names = ("loss", "target", "other")
                for name, got, ref in zip(names, actual, expected, strict=True):
                    error = abs(mp.mpf(float(got)) - ref) / abs(ref)
                    assert error <= 64 * eps, (name, classes, gap, error / eps)

    def test_topk_entropy_capped_confident_rows_within_64_eps(self, subtests: pytest.Subtests):
        # Log-domain odds and shifts put both backends up to 339 eps off at m = 649.5.
        eps = np.finfo(np.float64).eps
        for classes, k in [(40, 2), (1000, 5)]:
            z = jnp.asarray(capped_rows(classes))
            target = np.zeros(len(z), np.int32)
            expected = [capped_reference(gap, classes, k) for gap in CAPPED_GAPS]
            for backend in ("jax", "native"):

                def loss(
                    x: jax.Array, k: int = k, backend: str = backend, target: np.ndarray = target
                ) -> jax.Array:
                    return sdca_loss(
                        x, target, "topk_cross_entropy", k=k, backend=backend, reduction="none"
                    )

                value = np.asarray(loss(z))
                gradient = np.asarray(jax.grad(lambda x, f=loss: f(x).sum())(z))
                for i, gap in enumerate(CAPPED_GAPS):
                    actual = (value[i], gradient[i, 0], gradient[i, 1], gradient[i, -1])
                    names = ("loss", "target", "capped", "free")
                    with subtests.test(backend=backend, classes=classes, gap=gap):
                        for name, got, ref in zip(names, actual, expected[i], strict=True):
                            error = abs(mp.mpf(float(got)) - ref) / abs(ref)
                            assert error <= 64 * eps, (name, error / eps)

    def test_proximal_entropy_mean_without_overflow(self, subtests: pytest.Subtests):
        largest = np.finfo(np.float64).max
        logits = np.array([[0.0, -largest, -largest, 0.0]])
        positive = np.array([[True, True, True, False]])
        expected_value = float(Fraction.from_float(float(largest)) * 2 / 3)
        # The two far labels have vanishing probabilities. For the two visible
        # classes, stationarity gives p = sigmoid(4/3 - 2p) at alpha=1.
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if mid < 1 / (1 + math.exp(2 * mid - 4 / 3)):
                lo = mid
            else:
                hi = mid
        p = (lo + hi) / 2
        expected_gradient = [[p - 1 / 3, -1 / 3, -1 / 3, 1 - p]]
        for reduction in ("none", "mean", "sum"):

            def loss(
                x: jax.Array, upstream: float, reduction: str = reduction
            ) -> tuple[jax.Array, jax.Array]:
                values = sdca_loss(
                    x,
                    positive,
                    "proximal_cross_entropy",
                    backend="jax",
                    alpha=1.0,
                    reduction=reduction,
                )
                return values.sum() * upstream, values

            function = jax.value_and_grad(loss, has_aux=True)
            expected = np.array([expected_value]) if reduction == "none" else expected_value
            for compiled in (False, True):
                evaluate = jax.jit(function) if compiled else function
                for upstream in (1.0, 2.0**-10, 3.5):
                    with subtests.test(reduction=reduction, jit=compiled, upstream=upstream):
                        (_, value), gradient = evaluate(logits, upstream)
                        np.testing.assert_allclose(value, expected, rtol=4e-15, atol=0)
                        np.testing.assert_allclose(
                            gradient,
                            np.asarray(expected_gradient) * upstream,
                            rtol=1e-12,
                            atol=1e-14,
                        )

    def test_proximal_entropy_unsupported_float64_spread(self, subtests: pytest.Subtests):
        largest = np.finfo(np.float64).max
        logits = np.array([[largest, -largest, 0.0], [0.0, 0.0, 0.0]])
        labels = np.array([0, 0])

        def loss(x: jax.Array) -> tuple[jax.Array, jax.Array]:
            values = sdca_loss(x, labels, "proximal_cross_entropy", backend="jax", reduction="none")
            return values.sum(), values

        function = jax.value_and_grad(loss, has_aux=True)
        for compiled in (False, True):
            with subtests.test(jit=compiled):
                (_, values), gradient = (jax.jit(function) if compiled else function)(logits)
                assert np.isnan(values[0])
                assert np.isnan(gradient[0]).all()
                assert np.isfinite(values[1])
                assert np.isfinite(gradient[1]).all()

    def test_proximal_entropy_references_and_stationarity(self):
        # Equal label scores reach log 2; a^2 underflows at alpha = 1e300 (80-digit solve).
        cases = [
            (np.array([[1e16, 0.0, 0.0]]), np.array([[True, True, False]]), 1e50, math.log(2)),
            (np.zeros((1, 2)), np.array([0]), 1e300, 1.1762794618348152e-295),
        ]
        for z, target, alpha, expected in cases:
            value = float(
                sdca_loss(z, target, "proximal_cross_entropy", alpha=alpha, backend="jax")
            )
            assert abs(value / expected - 1.0) <= 1e-13
        # g = softmax(z - alpha g) - q, and the value is at most H(q, z).
        rng = np.random.default_rng(9)
        z = 4 * rng.normal(size=(32, 7))
        for target in targets_of("proximal_cross_entropy", 32, 7, rng):
            mask = target if target.dtype == bool else np.eye(7, dtype=bool)[target]
            q = mask / mask.sum(-1, keepdims=True)
            for alpha in [1e-3, 0.5, 20.0]:

                def f(x: jax.Array, target: np.ndarray = target, alpha: float = alpha) -> jax.Array:
                    return sdca_loss(
                        x,
                        target,
                        "proximal_cross_entropy",
                        alpha=alpha,
                        backend="jax",
                        reduction="none",
                    )

                value, vjp = jax.vjp(f, z)
                (g,) = vjp(np.ones(32))
                u = z - alpha * g
                np.testing.assert_allclose(jax.nn.softmax(u, -1) - q, g, rtol=0, atol=5e-14)
                cross = -(q * jax.nn.log_softmax(z, -1)).sum(-1)
                assert bool((value <= cross + 1e-14).all())

    @pytest.mark.long
    @pytest.mark.parametrize("part", [(o, c) for o in TENSOR for c in (2, 7, 40)], ids=str)
    def test_float32_precision(self, part: tuple[str, int]):
        """precision="float32" solves in float32, to float32 accuracy against float64."""
        ran = False
        rng = np.random.default_rng(29)
        for objective in TENSOR:
            for classes, scale in [(2, 1.0), (7, 10.0), (40, 1.0)]:
                z = (scale * rng.normal(size=(16, classes))).astype(np.float32)
                for target in targets_of(objective, 16, classes, rng):
                    if (objective, classes) != part:
                        continue
                    ran = True
                    for options in options_of(objective, classes)[::2]:
                        results = [
                            jax.jit(
                                jax.value_and_grad(
                                    lambda x, p=p, t=target, o=objective, s=options: sdca_loss(
                                        x, t, o, backend="jax", precision=p, reduction="sum", **s
                                    )
                                )
                            )(z)
                            for p in ("float32", "float64")
                        ]
                        tol = 2e-5 * scale / min(options.get("gamma") or 1.0, 1.0)
                        for single, f64 in zip(*results, strict=True):
                            assert single.dtype == np.float32
                            np.testing.assert_allclose(
                                single, f64, rtol=2e-5, atol=tol, err_msg=f"{objective} {options}"
                            )
        # A gamma or a row spread float32 cannot hold is solved in float64.
        z = np.array([[3e38, -3e38, 0.0], [1.0, 0.0, 2.0]], np.float32)
        for gamma in [1e-40, 1.0]:
            for objective in ["topk_hinge_beta", "proximal_cross_entropy"]:
                if (objective, 40) != part:
                    continue
                ran = True
                expected = sdca_loss(z, np.array([0, 1]), objective, gamma=gamma, reduction="none")
                actual = jax.jit(
                    lambda x, o=objective, g=gamma: sdca_loss(
                        x,
                        np.array([0, 1]),
                        o,
                        gamma=g,
                        backend="jax",
                        precision="float32",
                        reduction="none",
                    )
                )(z)
                np.testing.assert_allclose(actual, expected, rtol=3e-7, err_msg=objective)
        # Nonsmooth top-k hinges of large k: the target entry restores the zero sum of the
        # stored 1/k entries, rounded once.
        z = (3 * rng.normal(size=(32, 1000))).astype(np.float32)
        labels = rng.integers(0, 1000, 32)
        for objective in ["topk_hinge_alpha", "topk_hinge_beta"]:
            if (objective, 40) != part:
                continue
            ran = True
            for k in [500, 999]:
                gradient = jax.jit(
                    jax.grad(
                        lambda x, o=objective, k=k: sdca_loss(
                            x, labels, o, k=k, backend="jax", precision="float32", reduction="sum"
                        )
                    )
                )(z)
                g = np.asarray(gradient, np.float64)
                target = g[np.arange(32), labels]
                count = (g != 0).sum(-1) - (target != 0)
                assert (count > 0).any()
                np.testing.assert_allclose(target, -count / k, rtol=2.0**-23, err_msg=objective)
        assert ran, part

    def test_low_precision_inputs(self):
        rng = np.random.default_rng(2)
        z = 3 * rng.normal(size=(16, 9))
        for dtype, tol in [(jnp.float16, 4e-3), (jnp.bfloat16, 3e-2)]:
            for objective in TENSOR:
                target = targets_of(objective, 16, 9, rng)[0]
                options: dict[str, Any] = dict(k=3, gamma=0.5, backend="jax")

                def f(
                    x: jax.Array,
                    target: np.ndarray = target,
                    objective: str = objective,
                    options: dict[str, Any] = options,
                ) -> jax.Array:
                    return sdca_loss(x, target, objective, reduction="sum", **options)

                value, gradient = jax.value_and_grad(f)(jnp.asarray(z, dtype))
                expected = jax.value_and_grad(f)(jnp.asarray(z, dtype).astype(np.float32))
                assert gradient.dtype == dtype
                for a, e in zip((value, gradient), expected, strict=True):
                    np.testing.assert_allclose(
                        np.asarray(a, np.float32), e, rtol=tol, atol=tol * 16, err_msg=objective
                    )

    def test_invalid_traced_targets_give_nan(self):
        z = np.zeros((2, 3))
        cases = [
            ("topk_hinge_alpha", np.array([3, 0])),
            ("topk_hinge_alpha", np.array([[True, True, False], [True, False, False]])),
            ("multilabel_hinge", np.array([[True, True, True], [True, False, False]])),
            ("proximal_cross_entropy", np.array([[False, False, False], [True, False, False]])),
            ("proximal_cross_entropy", np.array([[True, True, True], [True, False, False]])),
        ]
        for objective, target in cases:
            values = jax.jit(
                lambda x, t, o=objective: sdca_loss(x, t, o, backend="jax", reduction="none")
            )(z, target)
            assert bool(jnp.isnan(values[0])), objective
            assert bool(jnp.isfinite(values[1])), objective

    def test_auto_backend_and_lowering(self):
        z = np.random.default_rng(3).normal(size=(4, 6))
        labels = np.array([0, 5, 2, 1])

        def lowered(**options: Any) -> str:
            f = jax.jit(lambda x: sdca_loss(x, labels, "topk_hinge_beta", k=2, **options))
            return jax.jit(jax.grad(f)).lower(z).as_text()

        # auto: jax on an accelerator; on the CPU for small calls, the nonsmooth multiclass and
        # multilabel hinges, else native.
        assert not calls_native(lowered())
        assert not calls_native(lowered(backend="jax"))
        wide = np.random.default_rng(4).normal(size=(64, 1024))
        rows = np.arange(64) % 1024
        mask = np.eye(1024, dtype=bool)[rows]

        def wide_lowered(objective: str, **options: Any) -> str:
            target = mask if objective.startswith("multilabel") else rows
            f = jax.jit(lambda x: sdca_loss(x, target, objective, **options))
            return jax.jit(jax.grad(f)).lower(wide).as_text()

        cpu = jax.default_backend() == "cpu"
        ffi_on = cpu and sdca_jax._jax_ffi.enabled()
        for objective, options, native in [
            ("topk_hinge_beta", dict(k=2), cpu),
            ("topk_hinge_alpha", dict(k=1), ffi_on),
            ("multiclass_hinge", {}, ffi_on),
            ("multiclass_hinge", dict(gamma=0.5), cpu),
            ("multilabel_hinge", {}, ffi_on),
            ("multilabel_hinge", dict(gamma=0.5), cpu),
            ("topk_cross_entropy", dict(k=2), cpu),
            ("proximal_cross_entropy", {}, cpu),
        ]:
            assert calls_native(wide_lowered(objective, **options)) == native, objective
        small = jax.jit(lambda x: sdca_loss(x, labels, "proximal_cross_entropy"))
        assert calls_native(small.lower(z).as_text()) == cpu
        # Off the CPU the native backend is a host callback.
        with mock.patch.object(sdca_jax, "_accelerator", return_value=True):
            assert not calls_native(lowered())
            assert "callback" in lowered(backend="native")
        try:
            from jax import export
        except ImportError:  # Early releases need absl for it.
            return
        # A host callback could not be exported.
        exported = export.export(
            jax.jit(lambda x: sdca_loss(x, labels, "proximal_cross_entropy", backend="jax"))
        )(jax.ShapeDtypeStruct(z.shape, z.dtype))
        np.testing.assert_allclose(
            exported.call(z), sdca_loss(z, labels, "proximal_cross_entropy"), rtol=1e-13
        )

    def test_without_x64(self):
        """Without fused CUDA kernels or x64, auto keeps float64 losses native."""
        code = textwrap.dedent(
            """
            from unittest import mock
            import jax, numpy as np
            from libsdca import jax as sdca_jax
            from libsdca.jax import sdca_loss
            z = np.random.default_rng(3).normal(size=(4, 6)).astype(np.float32)
            labels = np.array([0, 5, 2, 1])
            def text(**options):
                f = lambda x: sdca_loss(x, labels, "topk_hinge_alpha", k=2, **options)
                return jax.jit(jax.grad(f)).lower(z).as_text()
            with mock.patch.object(sdca_jax, "_accelerator", return_value=True):
                assert "callback" in text()
                assert "callback" not in text(precision="float32")
                assert "f64" not in text(precision="float32")
                # The smoothed hinges form their entries in float64: native without x64.
                assert "callback" in text(gamma=1.0, precision="float32")
            try:
                sdca_loss(z, labels, "topk_hinge_alpha", backend="jax")
                raise AssertionError("no error")
            except ValueError as error:
                assert "jax_enable_x64" in str(error)
            for objective in ["topk_hinge_alpha", "proximal_cross_entropy", "multilabel_hinge"]:
                target = np.eye(6, dtype=bool)[labels] if "multi" in objective else labels
                a = sdca_loss(z, target, objective, backend="jax", precision="float32")
                b = sdca_loss(z, target, objective, backend="native", precision="float32")
                np.testing.assert_allclose(a, b, rtol=2e-5, err_msg=objective)
            print("ok")
            """
        )
        env = dict(os.environ, JAX_ENABLE_X64="0", SDCA_JAX_CUDA="off")
        result = run_python("-c", code, capture_output=True, text=True, env=env, check=False)
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout


@pytest.mark.parametrize("dtype,gap", [(np.float32, 90.0), (np.float64, 710.0)])
def test_proximal_entropy_normal_aggregate_of_subnormal_terms(
    dtype: type[np.floating[Any]], gap: float
):
    # XLA:CPU flushes individual subnormals, but their normal aggregate must survive.
    z = np.full((1, 1000), -gap, dtype=dtype)
    z[0, 1] = 0
    positive = jnp.arange(1000)[None, :] == 1

    def run(x: jax.Array):
        alpha = jnp.asarray(1.0, dtype=x.dtype)
        return _jax_losses.proximal_entropy(x, positive, alpha, jnp.log(alpha), 1)

    value, gradient = jax.jit(run)(jnp.asarray(z))
    half = math.exp(-gap / 2)
    expected = half * (half * 999)
    assert float(value[0]) > 0
    rtol = 64 * float(np.finfo(dtype).eps)
    np.testing.assert_allclose(np.asarray(value), [expected], rtol=rtol, atol=0)
    np.testing.assert_allclose(np.asarray(gradient[:, 1]), [-expected], rtol=rtol, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distillation_independent_offsets(dtype: type[np.floating[Any]]):
    from test_distillation_references import reference

    huge = 1e30 if dtype == np.float32 else 1e300
    uniform = np.full(17, huge, dtype=dtype)
    varying = (np.arange(17) % 2 + 1).astype(dtype)
    student = np.stack([uniform, varying, -uniform])
    teacher = np.stack([varying, uniform, varying])
    expected_value, expected_gradient = reference(student, teacher, 1.0)
    x, t = jnp.asarray(student), jnp.asarray(teacher)
    value = sdca_jax.distillation_loss(x, t, 1.0, reduction="none")
    gradient = jax.grad(lambda z: sdca_jax.distillation_loss(z, t, 1.0, reduction="sum"))(x)
    eps = float(np.finfo(dtype).eps)
    np.testing.assert_allclose(value, expected_value, rtol=64 * eps, atol=0)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=64 * eps, atol=0)


def test_single_thread_cpu_reads_the_xla_thread_count():
    # 16 and 128 threads contain the text of one thread.
    for flags, single in [
        ("--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1", True),
        ("--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=16", False),
        ("intra_op_parallelism_threads=128", False),
        ("", False),
    ]:
        with mock.patch.dict(os.environ, {"XLA_FLAGS": flags}):
            os.environ.pop("SDCA_NUM_THREADS", None)
            os.environ.pop("OMP_NUM_THREADS", None)
            assert sdca_jax._single_thread_cpu() is single, flags


def fused_cases(rows: int, classes: int) -> dict[str, tuple[Any, np.ndarray, Any]]:
    """Per fenced loss family: the loss of (student, target), the target, and the float64
    native losses and gradients of float32 scores z."""
    rng = np.random.default_rng(5)
    labels = rng.integers(0, classes, rows)
    mask = rng.random((rows, classes)) < 0.01
    mask[np.arange(rows), labels] = True
    teacher = (rng.standard_normal((rows, classes)) * 3).astype(np.float32)

    def native(name: str, target: np.ndarray) -> Any:
        return lambda z: loss_and_grad(np.float64(z), target, name)

    return {
        "cross_entropy": (
            lambda z, y: sdca_loss(z, y, reduction="sum"),
            labels,
            native("cross_entropy", labels),
        ),
        "multilabel_cross_entropy": (
            lambda z, y: sdca_loss(z, y, "multilabel_cross_entropy", reduction="sum"),
            mask,
            native("multilabel_cross_entropy", mask),
        ),
        "distillation": (
            lambda z, t: sdca_jax.distillation_loss(z, t, 2.0, reduction="sum"),
            teacher,
            lambda z: distillation_loss_and_grad(
                np.float64(z), np.float64(teacher), temperature=2.0
            ),
        ),
        "proximal_distillation": (
            lambda z, t: sdca_jax.proximal_distillation_loss(z, t, 2.0, 0.5, reduction="sum"),
            teacher,
            lambda z: proximal_distillation_loss_and_grad(
                np.float64(z), np.float64(teacher), temperature=2.0, alpha=0.5
            ),
        ),
    }


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="the fences are CPU passes")
@pytest.mark.parametrize("fusion", ["fenced", "unfenced"])
@mock.patch.dict(os.environ, {"SDCA_JAX_FFI": "off"})
def test_cpu_fusion_setting_forces_either_variant(fusion: str):
    """SDCA_JAX_CPU_FUSION, read when a loss is traced, fences every pass of each fenced loss
    family above 16,384 scores or none (on any machine, any thread count); both variants
    differ only in summation order and stay within float32 rounding of float64 losses."""
    rows, classes = 48, 1000
    z = (np.random.default_rng(6).standard_normal((rows, classes)) * 3).astype(np.float32)
    eps = float(np.finfo(np.float32).eps)
    with mock.patch.dict(os.environ, {"SDCA_JAX_CPU_FUSION": fusion, "SDCA_NUM_THREADS": "1"}):
        for family, (loss, target, native) in fused_cases(rows, classes).items():
            with mock.patch.object(sdca_jax, "_fenced", wraps=sdca_jax._fenced) as fence:
                value, gradient = jax.jit(jax.value_and_grad(loss))(jnp.asarray(z), target)
            assert (fence.call_count > 0) == (fusion == "fenced"), family
            values, expected = native(z)
            assert float(value) == pytest.approx(values.sum(), rel=64 * eps), family
            scale = np.abs(expected).max()
            np.testing.assert_allclose(gradient, expected, rtol=0, atol=64 * eps * scale)
            # Small calls are never fenced.
            with mock.patch.object(sdca_jax, "_fenced", wraps=sdca_jax._fenced) as fence:
                jax.make_jaxpr(jax.value_and_grad(loss))(jnp.asarray(z[:8]), target[:8])
            assert fence.call_count == 0, family
        # distillation_loss without its gradient (distillation_value).
        loss, teacher, native = fused_cases(rows, classes)["distillation"]
        with mock.patch.object(sdca_jax, "_fenced", wraps=sdca_jax._fenced) as fence:
            value = jax.jit(loss)(jnp.asarray(z), teacher)
        assert (fence.call_count > 0) == (fusion == "fenced")
        assert float(value) == pytest.approx(native(z)[0].sum(), rel=64 * eps)


@mock.patch.dict(os.environ, {"SDCA_JAX_FFI": "off"})
def test_cpu_fusion_rejects_unknown_settings():
    z = jnp.zeros((64, 512), jnp.float32)
    with mock.patch.dict(os.environ, {"SDCA_JAX_CPU_FUSION": "fast"}):
        with pytest.raises(ValueError, match="SDCA_JAX_CPU_FUSION must be auto, fenced or"):
            _jax_fusion.mode()
        if jax.default_backend() == "cpu":
            with pytest.raises(ValueError, match="SDCA_JAX_CPU_FUSION must be auto, fenced or"):
                jax.make_jaxpr(lambda s: sdca_jax.distillation_loss(s, z))(z)
        else:
            # A CPU tuning switch must not affect the shared CUDA kernels.
            jax.make_jaxpr(lambda s: sdca_jax.distillation_loss(s, z))(z)
    with mock.patch.dict(os.environ, {"SDCA_JAX_CPU_FUSION": " Unfenced "}):
        assert _jax_fusion.mode() == "unfenced"
    with mock.patch.dict(os.environ, {"SDCA_JAX_CPU_FUSION": ""}):
        assert _jax_fusion.mode() == "auto"


@contextlib.contextmanager
def fresh_fusion(cache: Any) -> Iterator[None]:
    """auto with an empty decision cache in this process and its disk cache at cache."""
    environment = {"SDCA_JAX_CPU_FUSION": "auto", "SDCA_CACHE_DIR": str(cache)}
    with mock.patch.dict(os.environ, environment), mock.patch.dict(_jax_fusion._decisions):
        _jax_fusion._decisions.clear()
        yield


def test_cpu_fusion_auto_calibrates_once_and_caches(tmp_path: Any):
    """auto keeps the fences unless the unfenced variant takes at most MARGIN of their time;
    each calibrated family calibrates once (also with several threads asking), and its
    decision is cached on disk for the machine key."""
    timings = {
        "distillation": {"fenced_s": 1.0, "unfenced_s": 0.5, "seconds": 0.1},
        "cross_entropy": {"fenced_s": 1.0, "unfenced_s": 0.9, "seconds": 0.1},
    }
    calls: list[str] = []
    distillation = ("distillation", jnp.float32, False)

    def calibrate(family: str, dtype: str) -> dict[str, float]:
        calls.append(family)
        time.sleep(0.05)
        return timings[family]

    with fresh_fusion(tmp_path), mock.patch.object(_jax_fusion, "calibrate", calibrate):
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            fenced = list(pool.map(lambda _: _jax_fusion.fenced(*distillation), range(8)))
        assert fenced == [False] * 8 and calls == ["distillation"]
        assert _jax_fusion.fenced("cross_entropy", jnp.float64, False) is True
        # Proximal distillation is not timed: it keeps the heuristic.
        assert _jax_fusion.fenced("proximal_distillation", jnp.float32, False) is True
        assert calls == ["distillation", "cross_entropy"]
        records = [json.loads(p.read_text()) for p in tmp_path.glob("jax_cpu_fusion_*.json")]
        assert sorted((r["family"], r["dtype"], r["fenced"]) for r in records) == [
            ("cross_entropy", "float64", True),
            ("distillation", "float32", False),
        ]
        assert all(r["key"] == _jax_fusion.machine_key(False) for r in records)
    # A later process reads the records; 16-bit scores (bfloat16 and float16 alike) or another
    # key (here one thread) calibrate anew.
    with fresh_fusion(tmp_path), mock.patch.object(_jax_fusion, "calibrate", calibrate):
        assert _jax_fusion.fenced(*distillation) is False
        assert calls == ["distillation", "cross_entropy"]
        _jax_fusion.fenced("distillation", jnp.bfloat16, False)
        _jax_fusion.fenced("distillation", jnp.float16, False)
        _jax_fusion.fenced("distillation", jnp.float32, True)
        assert calls == ["distillation", "cross_entropy", "distillation", "distillation"]


def test_cpu_fusion_auto_falls_back_to_the_heuristic(tmp_path: Any, caplog: Any):
    """A failed calibration takes the heuristic (fenced, but the gradient of distillation on
    one thread), with a warning, and caches nothing on disk."""

    def calibrate(family: str, dtype: str) -> dict[str, float]:
        raise RuntimeError("no CPU device")

    def fenced(family: str, single_thread: bool) -> bool:
        return _jax_fusion.fenced(family, jnp.float32, single_thread)

    with fresh_fusion(tmp_path), mock.patch.object(_jax_fusion, "calibrate", calibrate):
        assert fenced("multilabel_cross_entropy", False) is True
        assert fenced("distillation", False) is True
        assert fenced("distillation", True) is False
        assert fenced("proximal_distillation", True) is False
        assert fenced("distillation_value", True) is True
        assert fenced("cross_entropy", True) is True
    assert "calibration of distillation (float32) failed (no CPU device)" in caplog.text
    assert not list(tmp_path.glob("jax_cpu_fusion_*"))


@pytest.mark.parametrize("enabled", [False, True])
def test_cpu_fusion_calibration_preserves_precision_context(enabled: bool):
    def timed(family: str, dtype: str) -> dict[str, Any]:
        assert bool(jax.config.jax_enable_x64) == enabled
        assert dtype == ("float64" if enabled else "float32")
        return {"checked": True}

    with _jax_fusion._precision_scope(enabled), mock.patch.object(_jax_fusion, "_timed", timed):
        dtype = _jax_fusion.calibration_dtype(jnp.float64 if enabled else jnp.float32)
        assert _jax_fusion.calibrate("distillation", dtype) == {"checked": True}


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="the fences are CPU passes")
def test_cpu_fusion_auto_calibrates_while_a_loss_is_traced(tmp_path: Any):
    """The first trace of a loss under auto calibrates its family on concrete arrays in a
    thread of its own, inside the caller's jit trace, and the traced loss is right."""
    rows, classes = 48, 1000
    z = (np.random.default_rng(7).standard_normal((rows, classes)) * 3).astype(np.float32)
    loss, labels, native = fused_cases(rows, classes)["cross_entropy"]
    with fresh_fusion(tmp_path), mock.patch.dict(os.environ):
        os.environ.pop("SDCA_NUM_THREADS", None)
        os.environ.pop("OMP_NUM_THREADS", None)
        value, gradient = jax.jit(jax.value_and_grad(loss))(jnp.asarray(z), labels)
        (record,) = [json.loads(p.read_text()) for p in tmp_path.glob("jax_cpu_fusion_*.json")]
    assert record["family"] == "cross_entropy" and record["dtype"] == "float32"
    assert record["key"]["cpu"]
    assert record["fenced_s"] > 0 and record["unfenced_s"] > 0 and record["seconds"] < 30
    assert record["fenced"] == (record["unfenced_s"] > _jax_fusion.MARGIN * record["fenced_s"])
    values, expected = native(z)
    eps = float(np.finfo(np.float32).eps)
    assert float(value) == pytest.approx(values.sum(), rel=64 * eps)
    np.testing.assert_allclose(gradient, expected, rtol=0, atol=64 * eps)


@pytest.mark.parametrize("backend", ["native", "jax"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_all_capped_mean_of_scores_near_finfo_max(backend: str, dtype: str):
    # k = C - 1 caps every competitor: the loss is the mean of the score differences, whose
    # terms (1.5 S here) overflow although the mean (5 S / 6) does not.
    big = float(np.finfo(dtype).max)
    x = jnp.asarray([[-big / 2, big, big, -big]], dtype=dtype)
    y = jnp.zeros(1, dtype=jnp.int32)
    value, gradient = jax.value_and_grad(
        lambda z: sdca_jax.topk_cross_entropy(z, y, 3, backend=backend)
    )(x)
    eps = float(np.finfo(dtype).eps)
    assert float(value) == pytest.approx(5 * (big / 6), rel=64 * eps)
    expected = np.array([[-1, 1 / 3, 1 / 3, 1 / 3]])
    np.testing.assert_allclose(np.asarray(gradient, np.float64), expected, rtol=64 * eps, atol=0)


@pytest.mark.parametrize("backend", ["native", "jax"])
def test_all_capped_mean_beyond_double_double_limit(backend: str):
    # LIMIT (docs/API.md, Known limits): differences at 2^200, 2^100 and 1 lose the 1 in the
    # double-double mean (exact 1/4); the documented result is log 5, not worse.
    h, m = 2.0**200, 2.0**100
    x = jnp.asarray([[m / 4, h, m, 1, -h]], dtype="float64")
    value = float(sdca_jax.topk_cross_entropy(x, jnp.zeros(1, dtype=jnp.int32), 4, backend=backend))
    exact = math.log1p(4 * math.exp(0.25))
    assert abs(value - exact) <= exact - math.log(5) + 1e-15


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="XLA:CPU flushes subnormals")
@pytest.mark.parametrize(
    ("loss", "entropy"),
    [
        (lambda z, **o: sdca_jax.cross_entropy(z, jnp.zeros(1, dtype=jnp.int32), **o), False),
        (
            lambda z, **o: sdca_jax.multilabel_cross_entropy(
                z, jnp.arange(z.shape[1])[None] == 0, **o
            ),
            False,
        ),
        (lambda z, **o: sdca_jax.distillation_loss(z, z, **o), True),
        (lambda z, **o: sdca_jax.proximal_distillation_loss(z, z, **o), True),
        (
            lambda z, **o: sdca_jax.topk_cross_entropy(z, jnp.zeros(1, dtype=jnp.int32), 1, **o),
            False,
        ),
    ],
    ids=["cross_entropy", "multilabel", "distillation", "proximal_distillation", "topk_k1"],
)
def test_float32_cpu_aggregate_tail(loss: Any, entropy: bool):
    # Each exp(-90) is a float32 subnormal that XLA:CPU flushes; the far-tail rescan keeps
    # their sum, a normal number. Equal student and teacher logits give
    # the teacher's entropy log1p(t) + 90 t / (1 + t).
    x = jnp.full((1, 1000), -90.0, dtype="float32").at[0, 0].set(0.0)
    tail = 999 * mp.exp(-90)
    exact = float(mp.log1p(tail) + (90 * tail / (1 + tail) if entropy else 0))
    eps = float(jnp.finfo(jnp.float32).eps)
    value, gradient = jax.value_and_grad(lambda z: loss(z, reduction="sum"))(x)
    assert float(value) == pytest.approx(exact, rel=64 * eps, abs=0)
    assert float(gradient[0, 0]) == pytest.approx(
        0 if entropy else -float(tail / (1 + tail)), rel=64 * eps, abs=0
    )
    # Working precision does not change the float32 output dtype.
    assert float(loss(x, precision="float64")) == float(np.float32(exact))
