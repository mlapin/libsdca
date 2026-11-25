"""The CUDA FFI can run shared Triton plans without jax-triton, also under vmap."""

from __future__ import annotations

import os
import sys
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from child_process import run_python
from libsdca.jax import cross_entropy, distillation_loss


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("classes", "temperature", "reciprocal"),
    [
        (8193, 2.0, 0.5),
        (8193, 2.0**-1000, 2.0**1000),
        (8193, 2.0**1000, 2.0**-1000),
        (8193, 2.0**-1001, 0.0),
        (8193, 2.0**1001, 0.0),
        (8193, 0.75, 0.0),
        (4096, 2.0, 0.0),
        (4097, 2.0, 0.5),
    ],
)
def test_streamed_double_temperature_plan(
    classes: int, temperature: float, reciprocal: float
) -> None:
    pytest.importorskip("triton")
    from libsdca.triton import common, distill
    from libsdca.triton.math import bits

    call = common.Call(
        common.DISTILLATION,
        16,
        classes,
        "float64",
        "float64",
        temperature=temperature,
        saved=common.SAVE,
    )
    launches = common.plan(call).launches
    if classes <= 4096:
        (forward,) = launches
    else:
        forward, rescue, backward = launches
        assert rescue.kernel is distill._rescue_partition
        assert rescue.state == ("S",)
        assert dict(rescue.args)["TEMP"] == bits(temperature)
        assert "RECIPROCAL" not in dict(rescue.meta)
        assert backward.kernel is distill._backward
        assert dict(backward.meta)["RECIPROCAL"] == reciprocal
    assert forward.kernel is distill._forward
    assert dict(forward.meta)["RECIPROCAL"] == reciprocal


@pytest.mark.parametrize("classes", [10, 65, 255, 256, 257, 511, 512, 513, 1024, 4096, 4097, 32001])
def test_label_kernel_uses_the_same_row_geometry_across_batch_sizes(classes: int) -> None:
    from dataclasses import replace

    pytest.importorskip("triton")
    from libsdca.triton import common

    call = common.Call(common.CROSS_ENTROPY, 1, classes, "float32", "int64")
    for phase in (
        call,
        replace(call, grad=False, saved=common.SAVE),
        replace(call, values=False, saved=common.SAVED),
    ):
        expected = common.plan(phase)
        # Wide rows run as row chunks: partials, their merge, the gradient.
        chunked = 2 + (phase.grad and not phase.saved)
        assert len(expected.launches) == (1 if classes <= 4096 or phase.saved == 2 else chunked)
        for n in (255, 256, 511, 512, 513):
            other = common.plan(replace(phase, n=n))
            assert len(other.launches) == len(expected.launches)
            for mine, theirs in zip(other.launches, expected.launches, strict=True):
                assert mine.meta == theirs.meta
                assert mine.kernel == theirs.kernel
            assert [a.shape[1:] for a in other.arrays] == [a.shape[1:] for a in expected.arrays]


@pytest.mark.parametrize(
    ("rows", "classes"),
    [(513, c) for c in (7, 1000, 4096, 4097, 16401, 32000, 32769, 128256)]
    + [(1, 4097), (1170, 32000), (1171, 32000)],
)
@pytest.mark.parametrize(
    ("dtype", "target"), [("float32", "float32"), ("float32", "bool"), ("bfloat16", "uint8")]
)
def test_dense_ce_plans_share_arithmetic(rows: int, classes: int, dtype: str, target: str) -> None:
    from dataclasses import replace

    pytest.importorskip("triton")
    from libsdca.triton import common

    kind = common.MULTILABEL_CROSS_ENTROPY if target in ("bool", "uint8") else common.CROSS_ENTROPY
    call = common.Call(kind, rows, classes, dtype, target)
    for phase in (
        replace(call, grad=False),
        call,
        replace(call, grad=False, saved=common.SAVE),
        replace(call, values=False, saved=common.SAVED),
    ):
        expected = common.plan(phase)
        kernel_names = {step.kernel.fn.__name__ for step in expected.launches}
        # Wide float32-work masks run as chunks (no mask tables); probabilities from 512.
        wide_mask = phase.mask and classes > 4096
        assert "exponentials" not in kernel_names and "tiles" not in kernel_names
        arrays = {a.name: a for a in expected.arrays}
        # P: the chunk partials of probability and mask rows.
        assert set(arrays) <= {"V", "S", "G", "P"}
        chunked = (wide_mask or (not phase.mask and classes >= 512)) and (
            phase.saved != common.SAVED
        )
        assert ("P" in arrays) == chunked
        assert ("_mask_backward" in kernel_names) == (wide_mask and phase.grad)
        for step in expected.launches:
            assert len(step.writes) == len(set(step.writes))


@pytest.mark.cuda
def test_indexed_hinge_plan() -> None:
    from dataclasses import replace

    pytest.importorskip("triton")
    from libsdca.triton import common, topk

    for n in (1, 511, 512):
        for c in (63, 129, 513, 1000, 1024, 1025, 4096, 4097, 31999, 32000, 32001):
            call = common.Call(common.MULTICLASS_HINGE, n, c, "float32", "int64")
            for phase in (
                replace(call, saved=common.SAVE, grad=False),
                replace(call, saved=common.SAVED, values=False),
            ):
                # A program that owns the row can track its winner across column tiles; the
                # chunked kernels (no ROWS) only take streamed rows, whose winner they read.
                for launch in topk.plan(phase).launches:
                    flags = dict(launch.meta)
                    if "INDEX" in flags:
                        assert flags["INDEX"] == (flags.get("ROWS", 1) == 1)


@pytest.mark.cuda
def test_jit_vmap_without_jax_triton() -> None:
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("needs a CUDA device")
    result = run_python(
        "-m",
        "test_jax_cuda_ffi",
        text=True,
        capture_output=True,
        env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.cuda
@pytest.mark.parametrize("repeats", [1, 820])
def test_bfloat_multilabel_positive_residual(repeats: int) -> None:
    from libsdca.jax import multilabel_cross_entropy
    from masked_cases import bfloat_multilabel_positive_residual

    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("needs a CUDA device")
    x, m, u, reference = bfloat_multilabel_positive_residual()
    x, m = np.tile(x, (repeats, 1)), np.tile(m, (repeats, 1))
    u, reference = np.tile(u, repeats), np.tile(reference, repeats)

    @jax.jit
    def evaluate(
        z: jax.Array, mask: jax.Array, factors: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        values, pullback = jax.vjp(
            lambda scores: multilabel_cross_entropy(
                scores, mask, reduction="none", backend="kernel"
            ),
            z,
        )
        return values, pullback(factors)[0][:, 0], pullback(-factors)[0][:, 0]

    values, pos, neg = evaluate(jnp.asarray(x, jnp.bfloat16), jnp.asarray(m), jnp.asarray(u))
    assert np.isposinf(np.asarray(values)[2])
    for sign, actual in ((1, pos), (-1, neg)):
        expected = (reference * u * sign).astype(jnp.bfloat16).astype(np.float32)
        np.testing.assert_array_equal(np.asarray(actual, np.float32), expected)


@pytest.mark.cuda
@pytest.mark.parametrize("classes", [7, 32000])
def test_bfloat_multilabel_amplified_tail(classes: int) -> None:
    from libsdca.jax import multilabel_cross_entropy
    from masked_cases import amplified_multilabel_tail

    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("needs a CUDA device")
    x, target, u, reference = amplified_multilabel_tail(classes)
    z, mask = jnp.asarray(x, dtype=jnp.bfloat16), jnp.asarray(target)

    @jax.jit
    def evaluate(z: jax.Array, mask: jax.Array, upstream: jax.Array) -> tuple[jax.Array, jax.Array]:
        _, pullback = jax.vjp(
            lambda scores: multilabel_cross_entropy(
                scores, mask, reduction="none", backend="kernel"
            ),
            z,
        )
        return pullback(upstream)[0][:, 1], pullback(-upstream)[0][:, 1]

    expected = reference.astype(jnp.bfloat16).astype(np.float32)
    for sign, result in zip((1, -1), evaluate(z, mask, jnp.asarray(u)), strict=True):
        actual = np.asarray(result, dtype=np.float32)
        np.testing.assert_array_equal(actual, sign * expected)
        assert not np.signbit(actual[7::8]).any()
        assert not np.signbit(actual[-4])
        assert bool(np.signbit(actual[5])) == (sign == 1)


@pytest.mark.cuda
def test_reduced_ce_rounds_after_each_mapped_reduction() -> None:
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("needs a CUDA device")
    with jax.enable_x64(False):
        # In both slices the mean of rounded float32 rows differs by one ulp.
        other = np.array([[13, -17, -13], [-11, -13, 12]], dtype=np.float32)
        logits = jnp.stack((jnp.zeros_like(other), jnp.asarray(other)), axis=-1)
        labels = jnp.zeros(other.shape, jnp.int32)
        reference = np.logaddexp(0, other.astype(np.float64)).mean(axis=-1).astype(np.float32)
        mean = jax.jit(jax.vmap(cross_entropy))
        np.testing.assert_array_equal(mean(logits, labels), reference)

        # Two vmap axes, distinct class weights, ignored labels and saved statistics.
        z = jnp.stack((logits, -logits))
        y = jnp.stack((labels, labels.at[0, 0].set(-1))).at[:, :, 2].set(1)
        w = jnp.array([[[1.0, 2.0], [3.0, 1.0]], [[2.0, 1.0], [1.0, 4.0]]])

        scale = jnp.array([[2.0, -0.5], [-3.0, 1.0]])

        def loss(x: jax.Array, target: jax.Array, weight: jax.Array, s: jax.Array) -> jax.Array:
            return s * cross_entropy(x, target, weight=weight, ignore_index=-1)

        vg = jax.value_and_grad(loss)
        value, gradient = jax.jit(jax.vmap(jax.vmap(vg)))(z, y, w, scale)
        for i in range(2):
            for j in range(2):
                one, grad = vg(z[i, j], y[i, j], w[i, j], scale[i, j])
                np.testing.assert_array_equal(value[i, j], one)
                np.testing.assert_array_equal(gradient[i, j], grad)


@pytest.mark.cuda
def test_mapped_multilabel_reduction_and_cotangents() -> None:
    from libsdca.jax import multilabel_cross_entropy

    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("needs a CUDA device")
    rows, classes = 512, 513
    rng = np.random.default_rng(92)
    x = rng.normal(size=(2, rows, classes)).astype(np.float32)
    m = rng.uniform(size=x.shape) < 0.03
    m[:, :, 0] = True
    x[:, ::4, -1] = -np.inf
    m[:, ::4, -1] = False
    # The confident rows need their small leading gradient after a nonunit pullback.
    x[:, 1::4, :] = -70
    x[:, 1::4, 0] = 0
    m[:, 1::4, 1:] = False
    z, target = jnp.asarray(x), jnp.asarray(m)
    precise = x.astype(np.float64)
    peak = precise.max(axis=-1, keepdims=True)
    exponentials = np.exp(precise - peak)
    partition = exponentials.sum(axis=-1, keepdims=True)
    count = m.sum(axis=-1, keepdims=True)
    grad = exponentials / partition - m / count
    grad[:, 1::4, 0] = -(classes - 1) * np.exp(-70) / partition[:, 1::4, 0]
    expected = (
        np.log(partition) + np.where(m, peak - precise, 0).sum(axis=-1, keepdims=True) / count
    )
    expected[:, 1::4] = np.log1p((classes - 1) * np.exp(-70))

    def loss(scores: jax.Array, mask: jax.Array) -> jax.Array:
        return multilabel_cross_entropy(scores, mask, backend="kernel")

    def apply(scores: jax.Array, mask: jax.Array, scale: jax.Array) -> tuple[jax.Array, jax.Array]:
        value, pullback = jax.vjp(lambda xx: loss(xx, mask), scores)
        return value, pullback(scale)[0]

    mapped = jax.jit(jax.vmap(apply))
    plain = jax.jit(jax.vmap(loss))(z, target)
    for factors in ([1.0, 1.0], [0.25, -3.25], [2.0**-38, 0.0]):
        scale = np.asarray(factors, dtype=np.float32)
        value, gradient = mapped(z, target, jnp.asarray(scale))
        np.testing.assert_array_equal(value, plain)
        np.testing.assert_allclose(value, expected.mean(axis=1).ravel(), rtol=2e-7)
        np.testing.assert_allclose(
            gradient,
            (grad * (scale.astype(np.float64) / rows)[:, None, None]).astype(np.float32),
            rtol=4e-6,
            atol=np.finfo(np.float32).smallest_subnormal,
        )

    donated = jax.jit(jax.value_and_grad(loss), donate_argnums=(0,))
    value, gradient = donated(jnp.array(x[0], copy=True), target[0])
    np.testing.assert_array_equal(value, plain[0])
    np.testing.assert_allclose(gradient, (grad[0] / rows).astype(np.float32), rtol=4e-6)


@pytest.mark.cuda
def test_distillation_saved_normalizers() -> None:
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("needs a CUDA device")
    with jax.enable_x64(True):
        z = jax.random.normal(jax.random.key(33), (512, 1000), dtype=jnp.float32) * 3
        t = jax.random.normal(jax.random.key(42), z.shape, dtype=jnp.float32) * 3
        t = t.at[0].set(z[0]).at[1].set(z[1].at[0].add(1e-5))
        z = z.at[2].set(0).at[2, 0].set(190)
        t = t.at[2].set(z[2].at[1].add(1))
        # Equal partition sums with distinct distributions, then a small change
        # to one score. Both require an accurate difference of the normalizers.
        t = t.at[3].set(jnp.roll(z[3], 3))
        z = z.at[4, :2].set(jnp.array([7.0, -4.0], z.dtype))
        t = t.at[4].set(z[4].at[:2].set(jnp.array([-4.0, 7.001], z.dtype)))

        def evaluate(dtype: str) -> tuple[jax.Array, jax.Array]:
            def mean(student: jax.Array, teacher: jax.Array) -> jax.Array:
                return distillation_loss(student, teacher, temperature=2, precision=dtype)

            return jax.jit(jax.value_and_grad(mean))(z, t)

        value, gradient = evaluate("float32")
        expected, reference = evaluate("float64")
        np.testing.assert_allclose(value, expected, rtol=2e-7)
        masses = jax.nn.softmax(z.astype(jnp.float64) / 2) + jax.nn.softmax(
            t.astype(jnp.float64) / 2
        )
        eps = jnp.finfo(jnp.float32).eps
        error = jnp.abs(gradient.astype(jnp.float64) - reference)
        assert jnp.all(error <= 64 * eps * masses / (2 * len(z)) + 2.0**-149)
        np.testing.assert_array_equal(gradient[0], 0.0)
        bound = 4 * eps * jnp.max(jnp.abs(reference[3:5]), axis=1)
        assert jnp.all(jnp.max(error[3:5], axis=1) <= bound)


@pytest.mark.parametrize(
    ("count", "classes", "temperature", "selected"),
    [
        (1, 10, 0.7, True),
        (17, 129, 3.0, True),
        (4095, 1001, 2.0, True),
        (3, 4096, 1.0, True),
        (3, 4097, 1.0, False),
    ],
)
def test_bfloat_distillation_joint_selection(
    monkeypatch: pytest.MonkeyPatch, count: int, classes: int, temperature: float, selected: bool
) -> None:
    from libsdca import _jax_ops
    from libsdca._frontend import KINDS, Rows

    monkeypatch.setattr(_jax_ops, "platform", lambda scores: "gpu")
    monkeypatch.setattr(_jax_ops._jax_results, "available", True)
    scores = jax.ShapeDtypeStruct((count, classes), jnp.bfloat16)
    rows = Rows(KINDS["distillation"], temperature=temperature)

    def check(x: jax.Array) -> jax.Array:
        assert _jax_ops._joint_cuda(x, x, rows) is selected
        return x

    jax.eval_shape(check, scores)


@pytest.mark.cuda
@pytest.mark.parametrize(("count", "classes"), [(32768, 100), (4096, 1000)])
def test_bfloat_joint_distillation_vjp(count: int, classes: int) -> None:
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("needs a CUDA device")
    # Equal partitions for the swap give a signed analytic derivative. The
    # remaining rows exercise the finite ordinary and masked-score paths.
    rng = np.random.default_rng(1729)
    z = jnp.asarray(rng.normal(size=(4, classes)), jnp.bfloat16)
    t = jnp.asarray(rng.normal(size=(4, classes)), jnp.bfloat16)
    z = z.at[0].set(0).at[0, 0].set(1).at[1, -1].set(-jnp.inf)
    t = t.at[0].set(0).at[0, 1].set(1).at[1, -1].set(-jnp.inf)
    scores, teachers = jnp.tile(z, (count // 4, 1)), jnp.tile(t, (count // 4, 1))

    def loss(x: jax.Array, y: jax.Array) -> jax.Array:
        return distillation_loss(x, y, temperature=2.0)

    @jax.jit
    def scaled(x: jax.Array, y: jax.Array, scale: jax.Array) -> tuple[jax.Array, jax.Array]:
        value, pull = jax.vjp(lambda a: loss(a, y), x)
        return value, pull(scale)[0]

    with jax.enable_x64(False):
        value = jax.jit(loss)(scores, teachers)
        jointly, grad = jax.jit(jax.value_and_grad(loss))(scores, teachers)
        np.testing.assert_array_equal(jointly, value)
        diff = np.expm1(0.5)
        for factor in (1.0, 0.0, -0.75):
            primal, g = scaled(scores, teachers, jnp.float32(factor))
            np.testing.assert_array_equal(primal, value)
            swap = diff * factor / (2 * count * (classes + diff))
            expected = jnp.asarray([swap, -swap], jnp.bfloat16)
            np.testing.assert_array_equal(g[0, :2], expected)
            if factor == 1.0:
                np.testing.assert_array_equal(g, grad)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_streamed_partition_rescue_preserves_signed_pullback(dtype: str) -> None:
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("needs a CUDA device")
    # A swap leaves the partition unchanged. Opposing perturbations raise it
    # quadratically, including the gradient at the untouched columns.
    scores = np.zeros((3, 4097), np.float32)
    scores[0, 1], scores[1, 0] = 1.0, 13.25
    targets = scores.copy()
    targets[0, :2] = 1.0, 0.0
    targets[1, 1:3] = 2.0**-23, -(2.0**-23)

    def pull(x: jax.Array, y: jax.Array, cotangent: jax.Array) -> jax.Array:
        def loss(z: jax.Array) -> jax.Array:
            return distillation_loss(z, y, temperature=2.0, reduction="none", backend="kernel")

        _, backward = jax.vjp(loss, x)
        return backward(cotangent)[0]

    actual = np.asarray(
        jax.jit(pull)(
            jnp.asarray(scores, dtype=dtype),
            jnp.asarray(targets, dtype=dtype),
            jnp.array([1.0, -3.0, 1.0], dtype=jnp.float32),
        )
    )
    assert actual[0, 0] < 0 < actual[0, 1]
    np.testing.assert_array_equal(actual[0, 2:], 0)
    assert np.all(actual[1, 3:] < 0)
    np.testing.assert_array_equal(actual[2], 0)


@pytest.mark.cuda
def test_short_bfloat_value_is_independent_of_pullback() -> None:
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("needs a CUDA device")
    rng = np.random.default_rng(143)
    z = jnp.asarray(3.0 * rng.standard_normal((512, 100)), jnp.bfloat16)
    # The identical rows need a signed gradient rescue. Repeat each row so a
    # one-ulp difference in its value would survive the grouped mean.
    scores = jnp.broadcast_to(z[:8, None, :], (8, 512, 100))
    teachers = scores.at[1::2, :, :2].add(jnp.bfloat16(0.25))

    def row(x: jax.Array, t: jax.Array) -> jax.Array:
        return distillation_loss(x, t, temperature=2.0)

    def pull(x: jax.Array, t: jax.Array, u: jax.Array) -> tuple[jax.Array, jax.Array]:
        v, back = jax.vjp(lambda s: row(s, t), x)
        return v, back(u)[0]

    expected = jax.jit(jax.vmap(row))(scores, teachers)
    values, grads = jax.jit(jax.vmap(jax.value_and_grad(row)))(scores, teachers)
    timed = jax.jit(jax.vmap(pull, in_axes=(0, 0, None)))
    np.testing.assert_array_equal(values, expected)
    for factor in (1.0, 0.0, -0.75):
        upstream = jnp.float32(factor)
        actual, gradient = timed(scores, teachers, upstream)
        np.testing.assert_array_equal(actual, expected)
        if factor == 1.0:
            np.testing.assert_array_equal(gradient, grads)
        else:
            np.testing.assert_array_equal(
                gradient, jax.vmap(pull, in_axes=(0, 0, None))(scores, teachers, upstream)[1]
            )


def check(x64: bool) -> None:
    # Soft targets, saved statistics packed into 32-bit words without x64, and
    # an aliased gradient result. Distinct batch rows exercise the folded plan.
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        z = jnp.arange(30, dtype=dtype).reshape(2, 3, 5) / 16
        target = jnp.full_like(z, 0.2)

        def loss(x: jax.Array, y: jax.Array) -> jax.Array:
            return cross_entropy(x, y, reduction="none", backend="kernel").sum()

        evaluate = jax.jit(jax.vmap(jax.value_and_grad(loss)))
        assert "libsdca_cuda_kernels" in evaluate.lower(z, target).as_text()
        values, gradient = evaluate(z, target)
        reference = -jnp.sum(target * jax.nn.log_softmax(z), axis=(1, 2))
        np.testing.assert_allclose(values, reference, rtol=2e-6)
        np.testing.assert_allclose(gradient, jax.nn.softmax(z) - target, atol=2e-7)


if __name__ == "__main__":
    sys.modules["jax_triton"] = None  # pyright: ignore[reportArgumentType] - block the import
    for enabled in (False, True):
        check(enabled)


@pytest.mark.cuda
def test_parallel_counted_mean_ragged_groups() -> None:
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("needs a CUDA device")
    rows, classes = 16385, 9
    z = jnp.zeros((2, rows, classes), jnp.float32).at[0, -1, 0].set(3)
    labels = jnp.full((2, rows), -100, jnp.int32).at[0, ::3].set(0).at[0, -1].set(0)
    count = (rows + 2) // 3 + 1
    regular = np.log(classes)
    last = np.log1p((classes - 1) * np.exp(-3))

    def loss(scores: jax.Array, target: jax.Array) -> jax.Array:
        return cross_entropy(scores, target, ignore_index=-100)

    with jax.enable_x64(False):
        mapped = jax.jit(jax.vmap(jax.value_and_grad(loss)))
        values, gradients = mapped(z, labels)
    assert values[0] == np.float32(((count - 1) * regular + last) / count)
    assert jnp.isnan(values[1])
    np.testing.assert_array_equal(gradients[1], 0)
    np.testing.assert_allclose(gradients[0, 0, 0], -(1 - 1 / classes) / count, rtol=2e-7)


@pytest.mark.cuda
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_short_bfloat_ragged_group_reduction(reduction: Literal["mean", "sum"]) -> None:
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("needs a CUDA device")
    rows, classes = 16385, 65
    peaks = np.array([[1.0, 11.0], [3.0, -5.0]])
    scores = np.zeros((2, rows, classes), np.float32)
    scores[:, :, 0] = peaks[:, :1]
    scores[:, -1, 0] = peaks[:, 1]
    # Uniform teachers make each row's cross entropy log(exp(x) + C - 1) - x/C.
    losses = np.log(np.exp(peaks) + classes - 1) - peaks / classes
    reference = (rows - 1) * losses[:, 0] + losses[:, 1]
    if reduction == "mean":
        reference /= rows

    def loss(x: jax.Array) -> jax.Array:
        return distillation_loss(x, jnp.zeros_like(x), reduction=reduction, backend="kernel")

    with jax.enable_x64(False):
        mapped = jax.jit(jax.vmap(loss))
        z = jnp.asarray(scores, jnp.bfloat16)
        np.testing.assert_array_equal(mapped(z), reference.astype(np.float32))
        mixed = mapped(z.at[1, -1, 0].set(jnp.nan))
        assert mixed[0] == np.float32(reference[0])
        assert jnp.isnan(mixed[1])


@pytest.mark.parametrize(("classes", "gamma"), [(10, 0.5), (16, 3.0)])
def test_short_smooth_signed_weighted_vjp(classes: int, gamma: float) -> None:
    from hinge_order_cases import short_smooth_rows
    from libsdca.jax import multiclass_hinge_loss

    size = 65537 if jax.default_backend() == "gpu" else 19
    x, y, weights, upstream, expected, derivative = short_smooth_rows(classes, gamma, size)

    def apply(
        z: jax.Array, target: jax.Array, w: jax.Array, seed: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        def rows(scores: jax.Array) -> jax.Array:
            return multiclass_hinge_loss(
                scores, target, weight=w, gamma=gamma, ignore_index=-100, reduction="none"
            )

        values, pullback = jax.vjp(rows, z)
        return values, pullback(seed)[0]

    values, gradient = jax.jit(apply)(x, y, weights, upstream)
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(gradient, derivative)


@pytest.mark.cuda
def test_short_smooth_projection_plan() -> None:
    from dataclasses import replace

    pytest.importorskip("triton")
    from libsdca.triton import common, topk

    call = common.Call(common.MULTICLASS_HINGE, 65536, 10, "float32", "int64", gamma=0.5)
    for eligible, case in (
        (True, replace(call, c=2, gamma=2.0**-126)),
        (True, replace(call, c=16, gamma=2.0**127, ignore=True, weighted=True)),
        (True, replace(call, n=1)),
        (True, replace(call, c=17, dtype="float16")),
        (True, replace(call, c=1000, dtype="bfloat16")),
        (True, replace(call, c=1024)),
        (False, replace(call, c=1025)),
        (False, replace(call, precision=2)),
        (False, replace(call, gamma=2.0**-127)),
        (False, replace(call, gamma=2.0**128)),
    ):
        assert case.register_smooth_hinge is eligible
        for phase in (
            replace(case, grad=False, saved=common.SAVE),
            replace(case, values=False, saved=common.SAVED),
        ):
            plan = topk.plan(phase)
            # Streamed rows also launch the chunked kernels, which take no SHORT_SIMPLEX.
            flags = [dict(launch.meta) for launch in plan.launches]
            assert all(f["SHORT_SIMPLEX"] is eligible for f in flags if "SHORT_SIMPLEX" in f)
            assert not eligible or any("SHORT_SIMPLEX" in f for f in flags)


@pytest.mark.parametrize("classes", [513, 31999, 32001])
def test_hinge_winner_replays_signed_cotangents(classes: int) -> None:
    from libsdca.jax import multiclass_hinge_loss

    z = np.full((5, classes), -8, dtype=np.float32)
    z[:, 0] = 0
    z[:2, -1] = 2
    z[1, 511] = 2
    z[2, -1] = -1
    labels = jnp.zeros(5, dtype=jnp.int32)
    seed = np.array([-2, 4, -0.5, 1.5, 0], dtype=np.float32)

    def rows(scores: jax.Array) -> jax.Array:
        return multiclass_hinge_loss(scores, labels, reduction="none", backend="kernel")

    @jax.jit
    def evaluate(scores: jax.Array, upstream: jax.Array) -> tuple[jax.Array, ...]:
        values, pull = jax.vjp(rows, scores)
        return values, pull(upstream)[0], pull(-upstream)[0]

    expected = np.zeros_like(z)
    expected[:3, -1] = [seed[0], seed[1] / 2, seed[2] / 2]
    expected[1, 511] = seed[1] / 2
    expected[:3, 0] = [-seed[0], -seed[1], -seed[2] / 2]
    values, first, second = evaluate(jnp.asarray(z), jnp.asarray(seed))
    np.testing.assert_array_equal(values, [3, 3, 0, 0, 0])
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, -expected)
