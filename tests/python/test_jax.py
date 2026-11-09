"""The JAX losses against the native ones, under jit, grad and vmap."""

import os
import textwrap
from typing import Any, TypedDict
from unittest import mock

import jax
import jax.numpy as jnp
import libsdca
import mpmath
import numpy as np
import numpy.typing as npt
import pytest
from child_process import run_python
from jax.typing import ArrayLike
from libsdca import LOSSES, loss_and_grad
from libsdca import jax as sdca_jax
from libsdca.jax import (
    cross_entropy,
    distillation_loss,
    multiclass_hinge_loss,
    multilabel_cross_entropy,
    multilabel_hinge_loss,
    proximal_cross_entropy,
    proximal_distillation_loss,
    sdca_loss,
    topk_cross_entropy,
    topk_hinge_alpha_loss,
    topk_hinge_beta_loss,
    topk_hinge_loss,
)

jax.config.update("jax_enable_x64", True)


class Options(TypedDict):
    k: int
    gamma: float
    alpha: float


class Smoothing(TypedDict, total=False):
    gamma: float
    alpha: float


class Tolerance(TypedDict):
    rtol: float
    atol: float


OPTIONS: Options = {"k": 2, "gamma": 0.6, "alpha": 0.9}
TOLERANCES: dict[Any, Tolerance] = {
    np.float64: {"rtol": 1e-12, "atol": 1e-13},
    np.float32: {"rtol": 2e-6, "atol": 2e-7},
}


def problem(dtype: npt.DTypeLike, examples: int = 6, classes: int = 7, seed: int = 0):
    """Scores, class indices and masks with a positive and a negative label."""
    rng = np.random.default_rng(seed)
    z = (3 * rng.normal(size=(examples, classes))).astype(dtype)
    labels = rng.integers(0, classes, examples)
    mask = rng.random((examples, classes)) < 0.4
    rows = np.arange(examples)
    mask[rows, labels] = True
    mask[rows, (labels + 1) % classes] = False
    return z, labels, mask


def target_of(loss: str, labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return mask if loss.startswith("multilabel") else labels


def x64(enabled: bool) -> Any:
    """A thread-local x64 scope: jax.enable_x64 in newer releases, else jax.experimental's."""
    import jax.experimental

    # Not `or`: jaxlib 0.12's jax.enable_x64 is a config State, which has no truth value.
    scope = getattr(jax, "enable_x64", None)
    if scope is None:
        scope = getattr(jax.experimental, "enable_x64", None)
    assert scope is not None
    return scope(enabled)


class JaxLossTests:
    def test_traced_nonfinite_rows_give_nan_losses_and_gradients(self, subtests: pytest.Subtests):
        z = np.array([[0, -np.inf, -1], [0, np.inf, -1], [0, np.nan, -1], [0, 1, -1]])
        labels = np.zeros(4, dtype=int)
        for objective in LOSSES:
            with subtests.test(objective=objective):

                def loss(x: jax.Array):
                    values = sdca_loss(x, labels, objective, backend="jax", reduction="none")
                    return values.sum(), values

                (_, values), gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))(z)
                assert np.isnan(values[:3]).all()
                assert np.isnan(gradient[:3]).all()
                assert np.isfinite(values[3])
                assert np.isfinite(gradient[3]).all()
        for invalid_teacher in (False, True):
            with subtests.test(invalid_teacher=invalid_teacher):

                def distill(x: jax.Array, t: jax.Array):
                    values = distillation_loss(x, t, reduction="none")
                    return values.sum(), values

                clean = np.zeros_like(z)
                student, teacher = (clean, z) if invalid_teacher else (z, clean)
                (_, values), gradient = jax.jit(jax.value_and_grad(distill, has_aux=True))(
                    student, teacher
                )
                assert np.isnan(values[:3]).all()
                assert np.isnan(gradient[:3]).all()
                assert np.isfinite(values[3])
                assert np.isfinite(gradient[3]).all()

    def test_multilabel_entropy_mean_survives_overflowed_intermediates(
        self, subtests: pytest.Subtests
    ):
        dtypes: list[type[np.floating[Any]]] = [np.float32, np.float64]
        for dtype in dtypes:
            magnitude = np.finfo(dtype).max * 0.625
            for scores in ([0, -1, -1], [1, -1, -1]):
                z = np.array([scores], dtype=dtype) * dtype(magnitude)
                expected_value = float(z.max()) - sum(float(v / 3) for v in z[0])
                assert np.isfinite(expected_value)
                target = np.ones_like(z, dtype=bool)
                for transform in (lambda f: f, jax.jit):
                    with subtests.test(dtype=dtype, scores=scores, transform=transform):
                        value, gradient = transform(
                            jax.value_and_grad(lambda x: sdca_loss(x, target, backend="jax"))
                        )(z)
                        np.testing.assert_allclose(value, expected_value, rtol=3e-7)
                        np.testing.assert_allclose(gradient, [[2 / 3, -1 / 3, -1 / 3]], rtol=3e-7)

    def test_distillation_rescues_overflowed_centering_before_temperature(
        self, subtests: pytest.Subtests
    ):
        for dtype, magnitude, temperature in (
            (np.float32, 3e38, 1e10),
            (np.float64, 1e308, 1e100),
        ):
            z = np.array([[magnitude, -magnitude]], dtype=dtype)
            for teacher in (z, z[:, ::-1].copy()):
                same = np.array_equal(z, teacher)
                # The scaled gap is enormous: the teacher and student are
                # one-hot to working precision, while loss and gradient fit.
                expected_value = 0.0 if same else 2 * (float(z[0, 0]) / temperature)
                expected_gradient = np.zeros_like(z)
                if not same:
                    expected_gradient[0] = [1 / temperature, -1 / temperature]
                for transform in (lambda f: f, jax.jit):
                    with subtests.test(dtype=dtype, same=same, transform=transform):
                        value, gradient = transform(
                            jax.value_and_grad(
                                lambda x, t: distillation_loss(x, t, temperature=temperature)
                            )
                        )(z, teacher)
                        np.testing.assert_allclose(value, expected_value, rtol=3e-7)
                        np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-7)

    def test_narrow_integer_targets_do_not_wrap_class_positions(self, subtests: pytest.Subtests):
        z = np.zeros((1, 257), dtype=np.float64)
        z[0, 0] = 2.0
        denominator = np.exp(2.0) + 256
        expected_value = np.log(denominator) - 2.0
        expected_gradient = np.full_like(z, 1.0 / denominator)
        expected_gradient[0, 0] = -256 / denominator
        for dtype in (np.int8, np.uint8):
            for transform in (lambda f: f, jax.jit):
                with subtests.test(dtype=dtype, transform=transform):
                    value, gradient = transform(
                        jax.value_and_grad(lambda x, t: sdca_loss(x, t, backend="jax"))
                    )(z, np.array([0], dtype=dtype))
                    np.testing.assert_allclose(value, expected_value, rtol=1e-14)
                    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-14)
        invalid = jax.jit(jax.value_and_grad(lambda x, t: sdca_loss(x, t, backend="jax")))
        for target in (np.array([-1], dtype=np.int8), np.zeros_like(z, dtype=bool)):
            value, gradient = invalid(z, target)
            assert np.isnan(value)
            assert np.isnan(gradient).all()

    def test_native_losses_match_loss_and_grad(self):
        for dtype in [np.float64, np.float32]:
            z, labels, mask = problem(dtype)
            weights = np.linspace(0.5, 2.0, len(z)).astype(dtype)
            for loss in LOSSES:
                target = target_of(loss, labels, mask)
                values, gradients = loss_and_grad(z, target, loss, **OPTIONS)

                def weighted(x: jax.Array, t: jax.Array, loss: str = loss):
                    per = sdca_loss(x, t, loss, backend="native", reduction="none", **OPTIONS)
                    return (per * weights).sum(), per

                (_, actual), gradient = jax.jit(jax.value_and_grad(weighted, has_aux=True))(
                    z, target
                )
                assert (actual.dtype, gradient.dtype) == (dtype, dtype)
                tol = TOLERANCES[dtype]
                np.testing.assert_allclose(actual, values.astype(dtype), **tol, err_msg=loss)
                np.testing.assert_allclose(gradient, gradients * weights[:, None], **tol)

    def test_native_losses_use_the_available_cpus(self):
        z, labels, _ = problem(np.float32)
        threads: list[object] = []

        def spy(*args: Any, **kwargs: Any) -> Any:
            threads.append(kwargs["num_threads"])
            return loss_and_grad(*args, **kwargs)

        # The host callback's call; test_jax_ffi.py compares the FFI handlers with it.
        with mock.patch.object(sdca_jax, "loss_and_grad", spy):
            loss = jax.jit(lambda x: sdca_loss(x, labels, "multiclass_hinge", backend="native"))
            with mock.patch.dict(os.environ, {"SDCA_JAX_FFI": "off"}):
                os.environ.pop("SDCA_NUM_THREADS", None)
                loss(z).block_until_ready()
                os.environ["SDCA_NUM_THREADS"] = "2"
                loss(z + 1).block_until_ready()
        # 0 is automatic; None leaves the count to SDCA_NUM_THREADS.
        assert threads == [0, None]

    def test_native_values_take_the_x64_setting_of_the_trace(self):
        """The host returns the value dtype declared at trace time, which XLA's callback thread
        cannot read: inside thread-local x64 scopes and after x64 changes past a compile."""
        code = textwrap.dedent(
            """
            from unittest import mock
            import jax, jax.experimental, numpy as np
            from libsdca import loss_and_grad
            from libsdca import jax as sdca_jax
            from libsdca.jax import sdca_loss
            x64 = getattr(jax, "enable_x64", None)
            x64 = getattr(jax.experimental, "enable_x64", None) if x64 is None else x64
            z = np.random.default_rng(3).normal(size=(4, 6)).astype(np.float32)
            labels = np.array([0, 5, 2, 1])
            values, gradients = loss_and_grad(z, labels, "topk_hinge_beta", k=2)
            host, dtypes = sdca_jax._host, []
            def spy(*args, **kwargs):
                result = host(*args, **kwargs)
                dtypes.append(result[0].dtype)
                return result
            def loss(x):
                return sdca_loss(x, labels, "topk_hinge_beta", k=2, backend="native")
            def check(value, gradient):
                np.testing.assert_allclose(value, values.mean(), rtol=1e-6)
                np.testing.assert_allclose(gradient, gradients / 4, rtol=1e-6)
            with mock.patch.object(sdca_jax, "_host", spy):
                for enabled in (False, True):
                    with x64(enabled):
                        for f in (jax.value_and_grad(loss), jax.jit(jax.value_and_grad(loss))):
                            check(*f(z))
                        assert jax.jit(loss)(z).dtype == np.float32
                compiled = jax.jit(jax.value_and_grad(loss)).lower(z).compile()
                jax.config.update("jax_enable_x64", True)
                check(*compiled(z))
            # float32 values without x64, float64 inside its scope, float32 again compiled.
            assert dtypes == 3 * [np.float32] + 3 * [np.float64] + [np.float32], dtypes
            print("ok")
            """
        )
        # The host callback's dtypes; test_jax_ffi.py compares the FFI handlers with it.
        env = dict(os.environ, JAX_ENABLE_X64="0", SDCA_JAX_FFI="off")
        result = run_python("-c", code, capture_output=True, text=True, env=env, check=False)
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout
        # x64 on in this process: a disabled scope declares float32 values.
        z, labels, _ = problem(np.float32)
        expected = loss_and_grad(z, labels, "multiclass_hinge", gamma=0.6)
        for enabled in (False, True):
            with x64(enabled):
                loss = jax.jit(
                    jax.value_and_grad(
                        lambda x: sdca_loss(
                            x, labels, "multiclass_hinge", backend="native", gamma=0.6
                        )
                    )
                )
                value, gradient = loss(z)
                np.testing.assert_allclose(value, expected[0].mean(), rtol=1e-6)
                np.testing.assert_allclose(gradient, expected[1] / len(z), rtol=1e-6)

    def test_native_vmap_of_integer_labels_below_the_score_rank(self, subtests: pytest.Subtests):
        """vmap of native losses whose integer labels have the rank of the scores less one
        (batched or shared, the class axis inside the example): as their label masks."""
        rng = np.random.default_rng(11)
        z = rng.normal(size=(3, 2, 7, 4))
        labels = rng.integers(0, 7, (3, 2, 4))
        masks = np.moveaxis(np.eye(7, dtype=bool)[labels], -1, 2)
        for loss in [
            "multiclass_hinge",
            "topk_hinge_beta",
            "topk_cross_entropy",
            "proximal_cross_entropy",
        ]:

            def per(x: ArrayLike, t: ArrayLike, loss: str = loss) -> jax.Array:
                return sdca_loss(x, t, loss, backend="native", axis=1, reduction="sum", **OPTIONS)

            for in_axes, target, mask in [
                ((0, 0), labels, masks),
                ((0, None), labels[0], masks[0]),
            ]:

                def total(
                    x: jax.Array, t: ArrayLike, in_axes: tuple[int, int | None] = in_axes
                ) -> jax.Array:
                    return jax.vmap(per, in_axes)(x, t).sum()

                def pick(
                    i: int, in_axes: tuple[int, int | None] = in_axes, target: np.ndarray = target
                ) -> np.ndarray:
                    return target[i] if in_axes[1] == 0 else target

                with subtests.test(loss=loss, in_axes=in_axes):
                    mapped, grads = jax.jit(jax.vmap(per, in_axes)), jax.jit(jax.grad(total))
                    loop = [per(z[i], pick(i)) for i in range(3)]
                    np.testing.assert_allclose(mapped(z, target), loop, rtol=1e-12)
                    np.testing.assert_allclose(mapped(z, target), mapped(z, mask), rtol=1e-15)
                    expected = [jax.grad(per)(z[i], pick(i)) for i in range(3)]
                    np.testing.assert_allclose(grads(z, target), expected, rtol=1e-12)
                    np.testing.assert_allclose(grads(z, target), grads(z, mask), rtol=1e-15)

    def test_vmap_and_class_axis(self):
        z, labels, mask = problem(np.float64)
        batch = np.stack([problem(np.float64, seed=s)[0] for s in range(3)])
        batch_labels = np.stack([problem(np.float64, seed=s)[1] for s in range(3)])
        for loss, backend in [
            ("cross_entropy", "jax"),
            ("cross_entropy", "native"),
            ("topk_hinge_beta", "native"),
            ("proximal_cross_entropy", "native"),
            ("multilabel_hinge", "native"),
        ]:

            def per(x: ArrayLike, t: ArrayLike, loss: str = loss, backend: str = backend):
                return sdca_loss(x, t, loss, backend=backend, **OPTIONS)

            multilabel = loss.startswith("multilabel")
            batched = np.broadcast_to(mask, batch.shape) if multilabel else batch_labels
            shared = target_of(loss, labels, mask)
            for in_axes, targets in [((0, 0), batched), ((0, None), shared)]:

                def pick(
                    i: int, in_axes: tuple[int, int | None] = in_axes, targets: np.ndarray = targets
                ):
                    return targets[i] if in_axes[1] == 0 else targets

                loop = [per(batch[i], pick(i)) for i in range(3)]
                mapped = jax.jit(jax.vmap(per, in_axes=in_axes))(batch, targets)
                np.testing.assert_allclose(mapped, loop, rtol=1e-12, err_msg=loss)
                grads = jax.grad(lambda x, a=in_axes, t=targets: jax.vmap(per, a)(x, t).sum())
                expected = [jax.grad(per)(batch[i], pick(i)) for i in range(3)]
                np.testing.assert_allclose(jax.jit(grads)(batch), expected, rtol=1e-12)
            nested = jax.vmap(jax.vmap(per))(np.stack([batch] * 2), np.stack([batched] * 2))
            np.testing.assert_allclose(nested[1], jax.vmap(per)(batch, batched), rtol=1e-12)
        # (examples, classes, height, width) with axis=1.
        rng = np.random.default_rng(1)
        z4, l4 = rng.normal(size=(2, 7, 3, 2)), rng.integers(0, 7, (2, 3, 2))
        flat = np.moveaxis(z4, 1, -1).reshape(-1, 7)
        expected, _ = loss_and_grad(flat, l4.reshape(-1))
        for backend in ["native", "jax"]:
            values = sdca_loss(z4, l4, axis=1, backend=backend, reduction="none")
            np.testing.assert_allclose(np.asarray(values).reshape(-1), expected, rtol=1e-13)
        mask4 = np.moveaxis(np.eye(7, dtype=bool)[l4], -1, 1)
        np.testing.assert_allclose(sdca_loss(z4, mask4, axis=1), expected.mean(), rtol=1e-13)

    def test_jax_entropy_matches_native_to_any_order(self):
        for dtype in [np.float64, np.float32]:
            z, labels, mask = problem(dtype)
            for target in [labels, mask]:
                for f in [lambda x: x, jax.grad]:
                    native = f(lambda x, t=target: sdca_loss(x, t, backend="native"))(z)
                    actual = jax.jit(f(lambda x, t=target: sdca_loss(x, t, backend="jax")))(z)
                    np.testing.assert_allclose(actual, native, **TOLERANCES[dtype])
        z, labels, _ = problem(np.float64)
        hessian = jax.hessian(lambda x: sdca_loss(x, labels[:1], reduction="sum"))(z[:1])
        p = np.asarray(jax.nn.softmax(z[0]))
        np.testing.assert_allclose(
            hessian[0, :, 0, :], np.diag(p) - np.outer(p, p), rtol=1e-14, atol=1e-15
        )
        small = np.array([[0.0, 1e-10, -1e-10]])
        gradient = jax.grad(
            lambda x: sdca_loss(x, np.ones((1, 3), bool), "multilabel_cross_entropy")
        )
        np.testing.assert_allclose(gradient(small), small / 3, rtol=1e-6, atol=1e-20)
        half = sdca_loss(jnp.asarray(z, jnp.bfloat16), labels, reduction="none")
        assert half.dtype == jnp.float32

    def test_jax_entropy_derivatives_at_tied_maxima(self, subtests: pytest.Subtests):
        """Rows whose maximum ties (as all-zero logits): every order matches log_softmax, for
        labels at a tied maximum or not, on small calls and on large ones (another path)."""
        rng = np.random.default_rng(5)
        for classes in (6, 20000):
            rest = np.zeros(classes - 4)
            rows = [np.zeros(classes), np.r_[2, 2, 1, 0, rest], np.r_[1, 3, 3, 3, rest]]
            z = jnp.asarray(np.stack(rows))
            v, w = rng.normal(size=(2, *z.shape))
            for labels in (np.array([0, 0, 1]), np.array([3, 3, 0])):
                mask = np.eye(classes, dtype=bool)[labels]
                mask[:, 4] = True
                weights = mask / mask.sum(-1, keepdims=True)
                cases = (("labels", labels, np.eye(classes)[labels]), ("mask", mask, weights))
                for name, target, t in cases:

                    def reference(x: jax.Array, t: np.ndarray = t) -> jax.Array:
                        return -(t * jax.nn.log_softmax(x, -1)).sum()

                    def loss(x: jax.Array, target: np.ndarray = target) -> jax.Array:
                        return cross_entropy(x, target, reduction="sum")

                    def orders(f: Any) -> list[jax.Array]:
                        def jvp(x: jax.Array) -> jax.Array:
                            return jax.jvp(f, (x,), (v,))[1]

                        def second(x: jax.Array) -> jax.Array:
                            return jax.jvp(jvp, (x,), (w,))[1]

                        hvp = jax.jvp(jax.grad(f), (z,), (v,))[1]
                        third = jax.jvp(second, (z,), (v,))[1]
                        return [hvp, second(z), third]

                    with subtests.test(classes=classes, labels=labels.tolist(), target=name):
                        for actual, expected in zip(orders(loss), orders(reference), strict=True):
                            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)

    def test_native_derivatives_are_first_order_reverse_mode(self):
        z, labels, _ = problem(np.float64)

        for backend in ["native", "jax"]:

            def f(x: jax.Array, backend: str = backend):
                return sdca_loss(x, labels, "topk_hinge_beta", backend=backend, **OPTIONS)

            with pytest.raises(Exception, match="JVP"):
                jax.hessian(f)(z)
            with pytest.raises(TypeError):
                jax.jvp(f, (z,), (z,))

    def test_validation(self):
        z, labels, mask = problem(np.float64)
        cases = [
            ({"loss": "typo"}, "Unknown loss"),
            ({"backend": "torch"}, "Unknown backend"),
            ({"target": labels + 7}, "out of range"),
            ({"target": np.zeros_like(mask)}, "positive label"),
            ({"loss": "topk_hinge_beta", "target": mask}, "exactly one target"),
            ({"loss": "topk_hinge_beta", "gamma": 0.5, "target": mask}, "exactly one target"),
            ({"loss": "multiclass_hinge", "gamma": 0.5, "target": mask}, "exactly one target"),
            ({"loss": "topk_cross_entropy", "target": mask}, "exactly one target"),
            ({"loss": "multilabel_hinge", "target": np.ones_like(mask)}, "negative label"),
            ({"k": 7}, "k must be"),
            ({"gamma": -1.0}, "gamma"),
            ({"axis": 2}, "axis"),
            ({"logits": z.astype(int)}, "floating-point"),
            ({"logits": np.where(mask, np.nan, z)}, "finite"),
            ({"logits": z[:, :1], "target": labels * 0}, "two classes"),
            ({"target": labels.astype(float)}, "integer class indices"),
            ({"reduction": "max"}, "reduction"),
        ]
        for change, message in cases:
            arguments = {"logits": z, "target": labels, **change}
            with pytest.raises(ValueError, match=message):
                sdca_loss(arguments.pop("logits"), arguments.pop("target"), **arguments)
        # Traced targets reach the native checks on the host.
        with pytest.raises(Exception, match="positive label|one target"):
            jax.jit(lambda t: sdca_loss(z, t, "topk_hinge_beta", backend="native"))(
                labels + 7
            ).block_until_ready()

    def test_empty_batches_follow_torch(self, subtests: pytest.Subtests):
        # As libsdca.torch and torch.nn.functional: empty, a zero sum, a NaN mean, and a
        # zero gradient.
        z = jnp.zeros((0, 4))
        labels = jnp.zeros((0,), dtype=jnp.int32)
        calls = {
            "cross_entropy": lambda x, r: cross_entropy(x, labels, reduction=r),
            "topk_hinge_alpha": lambda x, r: topk_hinge_alpha_loss(x, labels, k=2, reduction=r),
            "distillation": lambda x, r: distillation_loss(x, x, reduction=r),
            "proximal_distillation": lambda x, r: proximal_distillation_loss(x, x, reduction=r),
        }
        for name, call in calls.items():
            with subtests.test(name):
                assert call(z, "none").shape == (0,)
                assert float(call(z, "sum")) == 0.0
                assert np.isnan(float(call(z, "mean")))
                assert jax.grad(lambda x, c=call: c(x, "sum"))(z).shape == z.shape

    def test_entropy_of_labels_keeps_tiny_losses_and_confident_gradients(self):
        rng = np.random.default_rng(5)
        z = (rng.normal(size=(64, 1000)) * 10).astype(np.float32)
        labels = rng.integers(0, 1000, 64)
        rows = np.arange(64)
        z[:32][rows[:32], labels[:32]] = z[:32].max(-1) + 30  # confident: losses near 1e-13
        z[32:40][rows[:8], labels[32:40]] = z[32:40].max(-1)  # tied with the maximum
        x = z.astype(np.float64)
        top = x.max(-1, keepdims=True)
        first = np.arange(1000) == x.argmax(-1)[:, None]
        mass = np.where(first, 0, np.exp(x - top)).sum(-1)
        expected = np.log1p(mass) + (top[:, 0] - x[rows, labels])
        p = np.exp(x - top) / (1 + mass[:, None])
        target = -np.where(np.arange(1000) == labels[:, None], 0, p).sum(-1)
        # Every row (64,000 scores: the formulas of large calls), and 16 confident or 16 partly
        # tied rows (16,000 scores: those of small calls).
        for k, f in ((None, cross_entropy), (1, topk_cross_entropy)):
            for part in (slice(None), slice(0, 16), slice(32, 48)):
                y = labels[part]

                def loss(
                    v: jax.Array, f: Any = f, k: int | None = k, y: np.ndarray = y
                ) -> tuple[jax.Array, jax.Array]:
                    values = (
                        f(v, y, reduction="none") if k is None else f(v, y, k, reduction="none")
                    )
                    return values.sum(), values

                (_, values), gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))(z[part])
                np.testing.assert_allclose(values, expected[part], rtol=1e-6)
                picked = gradient[np.arange(len(y)), y]
                np.testing.assert_allclose(picked, target[part], rtol=1e-6)
                # The other classes: exp of a rounded float32 difference to the maximum.
                np.testing.assert_allclose(
                    gradient, np.where(gradient < 0, gradient, p[part]), rtol=1e-5, atol=1e-30
                )
        # Traced labels out of range, negative ones included, and masks of top-k entropy at
        # k = 1 with other than one target give nan rows.
        step = jax.jit(jax.value_and_grad(lambda v, t: cross_entropy(v, t, reduction="sum")))
        for invalid in (-1, -100, 1000):
            value, gradient = step(z[:2], np.array([invalid, 3]))
            assert np.isnan(value) and np.isnan(gradient[0]).all()
            assert not np.isnan(gradient[1]).any()
        mask = np.zeros((3, 1000), bool)
        mask[0, :2] = mask[2, 5] = True
        values = jax.jit(lambda v, t: topk_cross_entropy(v, t, reduction="none"))(z[:3], mask)
        assert np.isnan(values).tolist() == [True, True, False]
        np.testing.assert_allclose(values[2], cross_entropy(z[2:3], np.array([5])), rtol=1e-6)

    def test_distillation(self):
        rng = np.random.default_rng(3)
        student, teacher = rng.normal(size=(2, 5, 4))
        for temperature in [0.5, 1.0, 3.0]:
            s, t = student / temperature, teacher / temperature
            p = np.exp(t - t.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            log_q = s - s.max(-1, keepdims=True)
            log_q -= np.log(np.exp(log_q).sum(-1, keepdims=True))
            q = np.exp(log_q)

            def f(x: jax.Array, y: jax.Array, temperature: float = temperature):
                return distillation_loss(x, y, temperature=temperature, reduction="sum")

            value, (gs, gt) = jax.jit(jax.value_and_grad(f, argnums=(0, 1)))(student, teacher)
            np.testing.assert_allclose(value, -(p * log_q).sum(), rtol=1e-14)
            expected_gs = (q - p) / temperature
            np.testing.assert_allclose(
                gs, expected_gs, rtol=1e-12, atol=1e-15 * float(np.abs(expected_gs).max())
            )
            np.testing.assert_array_equal(gt, 0.0)
        far = np.array([[1e30, 0.0, -1e30]])
        tiny = distillation_loss(far, far[:, ::-1], temperature=1e-20, reduction="none")
        assert bool(jnp.isfinite(tiny).all())
        same = jax.grad(lambda x: distillation_loss(x, student))(student)
        np.testing.assert_array_equal(same, 0.0)
        with pytest.raises(ValueError, match="matching shape"):
            distillation_loss(student, teacher[:, :3])
        with pytest.raises(ValueError, match="temperature"):
            distillation_loss(student, teacher, temperature=0.0)

    def test_distillation_second_derivatives_at_tied_maxima(self, subtests: pytest.Subtests):
        """Reverse over reverse mode of distillation at tied student maxima (and an untied
        row) matches that of the cross-entropy to softmax(teacher / T)."""
        rng = np.random.default_rng(8)
        student = np.array([np.zeros(6), [2, 2, 1, 0, 0, 0], [1, 3, 3, 3, 0, -1]])
        student = np.r_[student, rng.normal(size=(1, 6))]
        teacher = rng.normal(size=student.shape)
        for temperature in [0.5, 1.0, 3.0]:

            def loss(x: jax.Array, temperature: float = temperature) -> jax.Array:
                return distillation_loss(x, teacher, temperature, reduction="sum")

            def reference(x: jax.Array, temperature: float = temperature) -> jax.Array:
                p = jax.nn.softmax(teacher / temperature, -1)
                return -(p * jax.nn.log_softmax(x / temperature, -1)).sum()

            with subtests.test(temperature=temperature):
                actual = jax.jit(jax.jacrev(jax.grad(loss)))(student)
                expected = jax.jacrev(jax.grad(reference))(student)
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)

    def test_precision(self):
        z, labels, mask = problem(np.float32)
        for loss in LOSSES:
            target = target_of(loss, labels, mask)
            # The jax backend's float32 accuracy: test_jax_backend.
            backend = "auto" if loss == "cross_entropy" else "native"
            results = [
                jax.value_and_grad(
                    lambda x, p=p, t=target, n=loss, b=backend: sdca_loss(
                        x, t, n, precision=p, backend=b
                    )
                )(z)
                for p in ("float64", "float32", jnp.float64, jnp.float32)
            ]
            for f64, single, dt64, dt32 in zip(*results, strict=True):
                assert single.dtype == np.float32
                np.testing.assert_allclose(single, f64, rtol=2e-6, atol=2e-7, err_msg=loss)
                np.testing.assert_array_equal(dt64, f64)
                np.testing.assert_array_equal(dt32, single)
        # precision="float64" computes jax entropy and distillation in float64.
        entropy = sdca_loss(z, labels, precision="float64", reduction="none")
        double = sdca_loss(z.astype(np.float64), labels, reduction="none")
        np.testing.assert_array_equal(entropy, double.astype(np.float32))
        teacher = np.roll(z, 1, axis=1)
        options = {"temperature": 2.0, "reduction": "none"}
        f64 = distillation_loss(z, teacher, precision="float64", **options)
        double = distillation_loss(z.astype(np.float64), teacher.astype(np.float64), **options)
        np.testing.assert_array_equal(f64, double.astype(np.float32))
        for x, precision in (
            (z.astype(np.float64), "float32"),
            (z.astype(np.float64), jnp.float32),
            (z, "mixed"),
            (z, "half"),
        ):
            with pytest.raises(ValueError, match="precision"):
                sdca_loss(x, labels, precision=precision)
            with pytest.raises(ValueError, match="precision"):
                distillation_loss(x, x, precision=precision)

    def test_distillation_near_equality_and_tiny_temperatures(self):
        """Represented small differences survive; equal logits stay finite at any T."""
        for dtype, teacher, student in [
            (np.float32, [20.0, 0.0, 0.0], [20.0, 1e-7, 0.0]),
            (np.float64, [50.0, 0.0, 0.0], [50.0, 1e-16, 0.0]),
        ]:
            t, z = np.array([teacher], dtype), np.array([student], dtype)
            gradient = jax.grad(lambda x, t=t: distillation_loss(x, t, reduction="sum"))(z)
            mpmath.mp.dps = 100
            p = [mpmath.exp(mpmath.mpf(float(v))) for v in t[0]]
            q = [mpmath.exp(mpmath.mpf(float(v))) for v in z[0]]
            exact = [float(a / sum(q) - b / sum(p)) for a, b in zip(q, p, strict=True)]
            scale = max(abs(v) for v in exact)
            error = max(abs(float(g) - e) for g, e in zip(gradient[0], exact, strict=True))
            assert error / scale < 1e-5, str(dtype)
        for dtype in [np.float32, np.float64]:
            x = np.array([[1.0, 0.0], [3.0, 3.0]], dtype)
            # 1/T overflows below 5.6e-309: dividing by T must not become a multiply by inf.
            for temperature in [1e-20, 1e-100, 1e-300, 1e-310, 5e-324]:

                def f(v: jax.Array, temperature: float = temperature):
                    return distillation_loss(v, x, temperature=temperature)

                value, gradient = jax.value_and_grad(f)(x)
                assert abs(float(value) - np.log(2) / 2) <= 1e-6
                np.testing.assert_array_equal(gradient, 0.0)

    def test_distillation_ignores_a_common_offset(self):
        """A student offset from the teacher leaves the gradients unchanged, also where the
        offset rounds the differences of student and teacher logits."""
        rng = np.random.default_rng(2)
        teacher = (rng.integers(-(2**22), 2**22, size=(64, 10)) / 2**20).astype(np.float32)
        # On a coarse grid, so that every offset student below is exactly represented.
        student = np.round(teacher * 64) / 64 + rng.integers(-8, 8, size=teacher.shape) / 64
        student = student.astype(np.float32)
        for temperature, alpha in ((1.0, None), (4.0, None), (1.0, 1.0), (0.5, 0.1)):

            def f(x: jax.Array, temperature: float = temperature, alpha: float | None = alpha):
                if alpha is None:
                    return distillation_loss(x, teacher, temperature, reduction="sum")
                return proximal_distillation_loss(x, teacher, temperature, alpha, reduction="sum")

            step = jax.jit(jax.grad(f))
            expected = step(student)
            for offset in (32.0, 1024.0, 8192.0):
                shifted = step(student + np.float32(offset))
                np.testing.assert_allclose(shifted, expected, rtol=0, atol=1e-7)

    def test_distillation_wide_rows_every_entry_within_budget(self):
        """Every float32 distillation gradient entry at 32,768 classes stays within 64 eps of
        the row's largest entry (a zero-sum pivot formed from the others carried their
        roundings)."""
        rng = np.random.default_rng(7)
        student = rng.normal(size=(2, 32768))
        noise = rng.normal(size=(2, 32768))
        grad = jax.jit(jax.grad(lambda x, y: distillation_loss(x, y, 2.0, reduction="sum")))
        for spread in (0.3, 1e-3):
            s = student.astype(np.float32)
            t = (student + spread * noise).astype(np.float32)
            actual = np.asarray(grad(s, t), np.float64)
            expected = np.asarray(grad(s.astype(np.float64), t.astype(np.float64)), np.float64)
            scale = np.abs(expected).max(1, keepdims=True)
            assert np.all(np.abs(actual - expected) <= 64 * np.finfo(np.float32).eps * scale)

    @pytest.mark.parametrize("part", [0.5, 1.0, 4.0, "zero", "other"])
    def test_proximal_distillation_kkt_limits_and_zero_gradient(self, part: float | str):
        rng = np.random.default_rng(31)
        z, t = 3 * rng.normal(size=(2, 6, 9))
        for temperature in (0.5, 1.0, 4.0):
            if temperature != part:
                continue
            p = np.asarray(jax.nn.softmax(t / temperature))
            distill = distillation_loss(z, t, temperature, reduction="none")
            for alpha in (1e-4, 0.1, 1.0, 10.0, 1e4):

                def f(x: jax.Array, alpha: float = alpha, temperature: float = temperature):
                    values = proximal_distillation_loss(x, t, temperature, alpha, reduction="none")
                    return values.sum(), values

                (_, value), g = jax.jit(jax.value_and_grad(f, has_aux=True))(z)
                r = np.asarray(g) * temperature
                u = z / temperature - alpha * r
                pi = np.asarray(jax.nn.softmax(u))
                assert np.abs(pi - p - r).max() < 1e-13 * (1 + alpha)
                envelope = -(p * np.asarray(jax.nn.log_softmax(u))).sum(-1)
                envelope += alpha * (r * r).sum(-1) / 2
                np.testing.assert_allclose(value, envelope, rtol=4e-13)
                assert bool((value <= distill * (1 + 1e-12)).all())
            limit = proximal_distillation_loss(z, t, temperature, 1e-12, reduction="none")
            np.testing.assert_allclose(limit, distill, rtol=1e-11)
        if part == "zero":
            # Equal logits, also with subnormal teacher probabilities: an exactly zero gradient.
            far = np.array([[0.0, -720.0, -360.0, 1.0, -721.0]])
            for x in (z, far, far.astype(np.float32), z.astype(jnp.bfloat16)):
                for alpha in (1e-3, 1.0, 30.0):
                    g = jax.grad(lambda v, x=x, a=alpha: proximal_distillation_loss(v, x, 2.0, a))(
                        x
                    )
                    assert g.dtype == x.dtype
                    assert int(jnp.count_nonzero(g)) == 0
        if part != "other":
            return
        # vmap, the teacher's zero gradient, first-order only and validation.
        batched = jax.vmap(lambda a, b: proximal_distillation_loss(a, b, 2.0, 0.5))(
            z.reshape(2, 3, 9), t.reshape(2, 3, 9)
        )
        expected = proximal_distillation_loss(z, t, 2.0, 0.5, reduction="none").reshape(2, 3)
        np.testing.assert_allclose(batched, expected.mean(-1), rtol=1e-14)
        moved = proximal_distillation_loss(z.T, t.T, 2.0, 0.5, axis=0, reduction="none")
        np.testing.assert_allclose(moved, expected.reshape(-1), rtol=1e-14)
        gt = jax.grad(lambda b: proximal_distillation_loss(z, b), argnums=0)(t)
        np.testing.assert_array_equal(gt, 0.0)
        with pytest.raises(TypeError):
            jax.jvp(lambda v: proximal_distillation_loss(v, t), (z,), (z,))
        for alpha in (0.0, -1.0, np.nan, np.inf):
            with pytest.raises(ValueError, match="alpha"):
                proximal_distillation_loss(z, t, alpha=alpha)
        with pytest.raises(ValueError, match="matching shape"):
            proximal_distillation_loss(z, t[:, :3])
        nonfinite = jax.jit(lambda v: proximal_distillation_loss(v, t[:2], reduction="none"))(
            np.array([z[0], z[1] * np.inf])
        )
        assert np.isnan(nonfinite).tolist() == [False, True]

    def test_16_bit_logits_give_float32_values(self, subtests: pytest.Subtests):
        z, labels, _ = problem(np.float32)
        teacher = np.roll(z, 1, axis=1)
        for dtype in (jnp.float16, jnp.bfloat16):
            x, t = jnp.asarray(z, dtype), jnp.asarray(teacher, dtype)
            cases = {
                "sdca_loss": (lambda v: sdca_loss(v, labels, reduction="none"), jnp.float32),
                "cross_entropy": (
                    lambda v: cross_entropy(v, labels, reduction="none"),
                    jnp.float32,
                ),
                "distillation": (
                    lambda v: distillation_loss(v, t.astype(v.dtype), 2.0, reduction="none"),
                    jnp.float32,
                ),
                "proximal": (
                    lambda v: proximal_distillation_loss(
                        v, t.astype(v.dtype), 2.0, 0.5, reduction="none"
                    ),
                    jnp.float32,
                ),
                "hinge": (
                    lambda v: multiclass_hinge_loss(v, labels, reduction="none"),
                    jnp.float32,
                ),
            }
            for name, (f, out) in cases.items():
                with subtests.test(dtype=dtype, loss=name):
                    value, vjp = jax.vjp(f, x)
                    assert value.dtype == out
                    (gradient,) = vjp(jnp.ones_like(value))
                    assert gradient.dtype == dtype
                    reference = f(x.astype(np.float64))
                    np.testing.assert_allclose(
                        np.asarray(value, np.float64), reference, rtol=8e-3, atol=8e-3
                    )

    def test_reductions_round_once(self, subtests: pytest.Subtests):
        """float64 work sums the rows in float64 and rounds the sum once, within half a float32
        unit of 100-digit references; rows rounded to float32 first (or summed in float32)
        miss that by several units on 4,096 rows."""
        rng = np.random.default_rng(23)
        n, classes = 4096, 7
        z = (3 * rng.normal(size=(n, classes))).astype(np.float32)
        teacher = (z + rng.normal(size=z.shape)).astype(np.float32)
        labels = rng.integers(0, classes, n)
        mpmath.mp.dps = 100
        rows = [[mpmath.mpf(float(v)) for v in row] for row in z]
        entropy = [
            mpmath.log(sum(mpmath.exp(v) for v in r)) - r[y]
            for r, y in zip(rows, labels, strict=True)
        ]
        hinge = [
            max(max(1 + v - r[y] for j, v in enumerate(r) if j != y), 0)
            for r, y in zip(rows, labels, strict=True)
        ]
        tq = [[mpmath.mpf(float(v)) / 2 for v in row] for row in teacher]
        distillation = []
        for r, t in zip(rows, tq, strict=True):
            s = [v / 2 for v in r]
            p = [mpmath.exp(v) for v in t]
            total = sum(p)
            norm = mpmath.log(sum(mpmath.exp(v) for v in s))
            distillation.append(sum(a / total * (norm - b) for a, b in zip(p, s, strict=True)))
        cases = {
            "cross_entropy": (lambda x, r: cross_entropy(x, labels, **r), entropy),
            "hinge jax": (
                lambda x, r: multiclass_hinge_loss(x, labels, backend="jax", **r),
                hinge,
            ),
            "hinge native": (
                lambda x, r: multiclass_hinge_loss(x, labels, backend="native", **r),
                hinge,
            ),
            "distillation": (lambda x, r: distillation_loss(x, teacher, 2.0, **r), distillation),
        }
        for name, (f, exact) in cases.items():
            for reduction in ("sum", "mean"):
                with subtests.test(loss=name, reduction=reduction):
                    value = jax.jit(lambda x: f(x, dict(reduction=reduction, precision="float64")))(
                        z
                    )
                    assert value.dtype == jnp.float32
                    reference = mpmath.fsum(exact) / (n if reduction == "mean" else 1)
                    error = abs(mpmath.mpf(float(value)) - reference)
                    ulp = np.spacing(np.float32(abs(float(reference))))
                    assert error <= 0.5 * float(ulp) * (1 + 1e-9), (name, float(error / ulp))

    def test_native_losses_keep_subnormals_in_host_callbacks(self):
        # XLA:CPU flushes subnormals, also on the host callback thread.
        z, mask = np.array([[0.0, 20.0, 20.0, -100.0]]), np.array([[True, False, True, False]])
        expected, _ = loss_and_grad(z, mask, "proximal_cross_entropy", alpha=1e300)

        def per_row(x: jax.Array):
            return sdca_loss(x, mask, "proximal_cross_entropy", alpha=1e300, reduction="none")

        np.testing.assert_allclose(jax.jit(per_row)(z), expected, rtol=1e-15)
        # Subnormal gamma and alpha pass the wrapper's checks on that thread.
        cases: list[tuple[str, Smoothing]] = [
            ("topk_hinge_alpha", {"gamma": 1e-310}),
            ("multilabel_hinge", {"gamma": 5e-324}),
            ("proximal_cross_entropy", {"alpha": 1e-310}),
        ]
        for loss, options in cases:
            target = mask if loss.startswith("multilabel") else np.array([1])
            expected, gradient = loss_and_grad(z, target, loss, **options)

            def total(
                x: jax.Array,
                loss: str = loss,
                options: Smoothing = options,
                target: np.ndarray = target,
            ):
                return sdca_loss(x, target, loss, backend="native", reduction="sum", **options)

            value, actual = jax.jit(jax.value_and_grad(total))(z)
            np.testing.assert_allclose(value, expected.sum(), rtol=1e-15, err_msg=loss)
            np.testing.assert_allclose(actual, gradient, rtol=1e-15, err_msg=loss)

    def test_entropy_at_the_edge_of_the_dtype(self):
        for dtype, big in [(np.float32, 3e38), (np.float64, 1e308)]:
            z = np.array([[big, -big]], dtype)
            for backend in ["jax", "native"]:
                value, gradient = jax.value_and_grad(
                    lambda x, b=backend: sdca_loss(x, np.array([0]), backend=b)
                )(z)
                assert float(value) == 0.0, backend
                np.testing.assert_array_equal(gradient, 0.0)

    def test_hinge_margins_and_smoothing_against_two_class_closed_forms(self):
        from fractions import Fraction

        cases = [
            (1.0, 1e-20, 1e-20),
            (1.0, 2.0**-26, 1.0),
            (0.0, 1e200, 1e300),
            (1.0, 1e-200, 1e-300),
            (1e16, 1e16 + 2, 1.0),
        ]
        hinges = [o for o in LOSSES if "hinge" in o]
        for dtype in [np.float32, np.float64]:
            for z0, z1, gamma in cases:
                z = np.array([[z0, z1]], dtype)
                if not np.isfinite(z).all():
                    continue
                h = 1 + Fraction(float(z[0, 1])) - Fraction(float(z[0, 0]))
                for loss, smoothing in [(o, g) for o in hinges for g in (0.0, gamma)]:
                    width = Fraction(smoothing) * (2 if loss.startswith("multilabel") else 1)
                    slope = (
                        float(h > 0)
                        if width == 0
                        else float(min(Fraction(1), max(Fraction(0), h / width)))
                    )
                    val = (
                        float(max(h, 0))
                        if width == 0
                        else float(h * Fraction(slope) - width * Fraction(slope) ** 2 / 2)
                    )
                    target = (
                        np.array([[True, False]])
                        if loss.startswith("multilabel")
                        else np.array([0])
                    )
                    actual, grad = jax.jit(
                        jax.value_and_grad(
                            lambda x, t=target, name=loss, g=smoothing: sdca_loss(
                                x, t, name, gamma=g
                            )
                        )
                    )(z)
                    eps = float(np.finfo(dtype).eps)
                    message = f"{dtype} {loss} gamma={smoothing} {z}"
                    np.testing.assert_allclose(
                        float(actual), val, rtol=2 * eps, atol=0, err_msg=message
                    )
                    np.testing.assert_allclose(
                        grad[0], [-slope, slope], rtol=2 * eps, atol=0, err_msg=message
                    )

    def test_functional_helpers_and_names(self):
        z, labels, mask = problem(np.float64)
        for name in ("ce", "softmax", "svm", "topk_hinge_beta_smooth"):
            with pytest.raises(ValueError, match="Unknown loss"):
                sdca_loss(z, labels, name)
        np.testing.assert_allclose(cross_entropy(z, labels), sdca_loss(z, labels, "cross_entropy"))
        np.testing.assert_allclose(
            topk_cross_entropy(z, labels, 2), sdca_loss(z, labels, "topk_cross_entropy", k=2)
        )
        np.testing.assert_allclose(
            multiclass_hinge_loss(z, labels), sdca_loss(z, labels, "multiclass_hinge", k=2)
        )
        np.testing.assert_allclose(
            multiclass_hinge_loss(z, labels, 0.5),
            sdca_loss(z, labels, "topk_hinge_alpha", k=1, gamma=0.5),
        )
        for variant in ("alpha", "beta"):
            base = f"topk_hinge_{variant}"
            single = topk_hinge_alpha_loss if variant == "alpha" else topk_hinge_beta_loss
            for gamma in (0.0, 0.4):
                expected = sdca_loss(z, labels, base, k=2, gamma=gamma)
                np.testing.assert_allclose(
                    topk_hinge_loss(z, labels, 2, variant=variant, gamma=gamma), expected
                )
                np.testing.assert_allclose(single(z, labels, 2, gamma=gamma), expected)
        default = topk_hinge_loss  # the default variant is beta
        np.testing.assert_allclose(
            default(z, labels, 2), sdca_loss(z, labels, "topk_hinge_beta", k=2)
        )
        with pytest.raises(ValueError, match="variant"):
            topk_hinge_loss(z, labels, 2, variant="invalid")
        with pytest.raises(TypeError):
            # Deliberately an unknown keyword.
            topk_hinge_loss(z, labels, 2, mode="alpha")  # pyright: ignore[reportCallIssue]
        np.testing.assert_allclose(
            multilabel_cross_entropy(z, mask), sdca_loss(z, mask, "multilabel_cross_entropy")
        )
        np.testing.assert_allclose(
            multilabel_hinge_loss(z, mask), sdca_loss(z, mask, "multilabel_hinge")
        )
        np.testing.assert_allclose(
            multilabel_hinge_loss(z, mask, 0.6),
            sdca_loss(z, mask, "multilabel_hinge", gamma=0.6),
        )
        np.testing.assert_allclose(
            proximal_cross_entropy(z, labels, 0.7),
            sdca_loss(z, labels, "proximal_cross_entropy", alpha=0.7),
        )
        teacher = np.roll(z, 1, axis=1)
        np.testing.assert_allclose(
            distillation_loss(z, teacher, 3.0),
            distillation_loss(z, teacher, temperature=3.0),
        )


class PublicNamesTests:
    def test_star_import_exports_the_api_only(self):
        import libsdca.jax as module

        names: dict[str, Any] = {}
        exec("from libsdca.jax import *", names)
        assert set(names) - {"__builtins__"} == set(module.__all__)
        # Helpers from private modules stay private unless libsdca exports them too.
        leaked = {
            name
            for name, value in vars(module).items()
            if not name.startswith("_")
            and getattr(value, "__module__", "").startswith("libsdca")
            and name not in module.__all__
        }
        assert leaked <= set(libsdca.__all__)
