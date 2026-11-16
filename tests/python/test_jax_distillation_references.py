"""JAX distillation regressions using the shared multiprecision cases."""

import os
from unittest import mock

import numpy as np
import pytest
from test_distillation_references import cases, check_reference, proximal_reference, reference


class JaxDistillationReferenceTests:
    def test_jax(self, subtests: pytest.Subtests):
        try:
            import jax
            import jax.experimental
        except ImportError:
            pytest.skip("JAX is unavailable")
        from libsdca.jax import distillation_loss

        # Newer JAX releases move enable_x64 out of jax.experimental.
        context = getattr(jax, "enable_x64", None)
        if context is None:
            context = getattr(jax.experimental, "enable_x64", None)
        assert context is not None
        with context():
            for dtype in (np.float32, np.float64):
                for temperature in (0.25, 1.0, 4.0):
                    student, teacher = cases(dtype, temperature)
                    precisions = ("float32", "float64") if dtype == np.float32 else ("float64",)
                    for precision in precisions:
                        with subtests.test(
                            dtype=dtype, temperature=temperature, precision=precision
                        ):

                            def loss(z: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
                                values = distillation_loss(
                                    z,
                                    t,
                                    temperature=temperature,
                                    reduction="none",
                                    precision=precision,
                                )
                                return values.sum(), values

                            (_, value), gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))(
                                student, teacher
                            )
                            check_reference(
                                np.asarray(value),
                                np.asarray(gradient),
                                student,
                                teacher,
                                temperature,
                            )

    @pytest.mark.parametrize("layout", ["tall", "wide"])
    @pytest.mark.parametrize(
        "dtype,precision",
        [
            ("float32", None),
            ("float32", "float64"),
            ("float64", None),
            ("float16", None),
            ("float16", "float64"),
            ("bfloat16", None),
            ("bfloat16", "float64"),
        ],
        ids=str,
    )
    @mock.patch.dict(os.environ, {"SDCA_JAX_FFI": "off"})
    def test_jax_fenced_and_unfenced_passes(
        self,
        layout: str,
        dtype: str,
        precision: str | None,
        monkeypatch: pytest.MonkeyPatch,
        subtests: pytest.Subtests,
    ):
        """The cases tiled (tall) or padded with classes of no mass (wide) past _SMALL scores,
        where the CPU passes may be fenced (SDCA_JAX_CPU_FUSION): both programs on every
        platform, forward plus backward and forward only. 16-bit logits (the float32 cases
        where they are finite) keep the envelope of test_jax_half_precision."""
        try:
            import jax
            import jax.experimental
        except ImportError:
            pytest.skip("JAX is unavailable")
        import jax.numpy as jnp
        import libsdca.jax as sdca_jax
        from test_jax_half_precision import assert_envelope

        if jax.default_backend() != "cpu":
            pytest.skip("the fences are CPU passes")
        context = getattr(jax, "enable_x64", None)
        if context is None:
            context = getattr(jax.experimental, "enable_x64", None)
        assert context is not None
        half = dtype in ("float16", "bfloat16")
        with context():
            for temperature in (0.25, 1.0, 4.0):
                source = cases(np.float32 if half else np.dtype(dtype).type, temperature)
                largest = float(jnp.finfo(dtype).max)
                finite = (np.abs(source[0]) <= largest).all(-1)
                finite &= (np.abs(source[1]) <= largest).all(-1)
                student, teacher = (
                    np.asarray(jnp.asarray(x[finite], dtype), np.float64) for x in source
                )
                value, gradient = reference(student, teacher, temperature)
                rows, classes = student.shape
                if layout == "tall":
                    copies = sdca_jax._SMALL // student.size + 1
                    student, teacher = np.tile(student, (copies, 1)), np.tile(teacher, (copies, 1))
                    value, gradient = np.tile(value, copies), np.tile(gradient, (copies, 1))
                else:
                    pad = np.full((rows, sdca_jax._SMALL // rows + 1 - classes), -1e4 * temperature)
                    student, teacher = np.hstack((student, pad)), np.hstack((teacher, pad))
                assert student.size > sdca_jax._SMALL
                z, t = jnp.asarray(student, dtype), jnp.asarray(teacher, dtype)
                for fenced in (True, False):
                    message = f"{layout} {dtype} {precision} T={temperature} fenced={fenced}"
                    with subtests.test(message):
                        monkeypatch.setenv(
                            "SDCA_JAX_CPU_FUSION", "fenced" if fenced else "unfenced"
                        )

                        def loss(
                            z: jax.Array, t: jax.Array, temperature: float = temperature
                        ) -> jax.Array:
                            return sdca_jax.distillation_loss(
                                z, t, temperature=temperature, reduction="none", precision=precision
                            )

                        def total(z: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
                            values = loss(z, t)
                            return values.sum(), values

                        with mock.patch.object(
                            sdca_jax, "_fenced", wraps=sdca_jax._fenced
                        ) as fence:
                            (_, values), gradients = jax.jit(
                                jax.value_and_grad(total, has_aux=True)
                            )(z, t)
                        assert (fence.call_count > 0) == fenced
                        assert np.all(np.asarray(gradients)[:, classes:] == 0)
                        gradients = gradients[:, :classes]
                        results = [(values, gradients)]
                        if fenced:  # forward-only ignores _fenced_distillation; check it once
                            results.append((jax.jit(loss)(z, t), None))
                        for v, g in results:
                            if half:
                                assert_envelope(z, v, g, value, gradient, 0.0, precision, message)
                            else:
                                check_reference(
                                    np.asarray(v),
                                    None if g is None else np.asarray(g),
                                    *source,
                                    temperature,
                                    (value, gradient),
                                )

    @pytest.mark.parametrize(
        "part", [(d, i) for d in ("float32", "float64") for i in range(5)], ids=str
    )
    def test_jax_proximal(self, part: tuple[str, int], subtests: pytest.Subtests):
        ran = False
        try:
            import jax
            import jax.experimental
        except ImportError:
            pytest.skip("JAX is unavailable")
        from libsdca.jax import proximal_distillation_loss

        # Newer JAX releases move enable_x64 out of jax.experimental.
        context = getattr(jax, "enable_x64", None)
        if context is None:
            context = getattr(jax.experimental, "enable_x64", None)
        assert context is not None
        rng = np.random.default_rng(0)
        student = rng.normal(size=(4, 7)) * 4
        teacher = student + rng.normal(size=student.shape) * np.array([[0], [1e-3], [1], [5]])
        # pi far above a teacher probability near the underflow (a -inf loss in float32).
        tiny = np.array([[0.0, 10.0, 0.0], [0.0, 10.0, -5.0]])
        gaps = {np.float32: 95.0, np.float64: 740.0}
        with context():
            # The machine epsilons of float32 and float64.
            for dtype, eps in ((np.float32, 2.0**-23), (np.float64, 2.0**-52)):
                cases = [(student, teacher, T, (1e-6, 0.3, 10.0, 1e6)) for T in (0.25, 1.0, 4.0)]
                far = np.array([[0.0, -gaps[dtype], 0.0], [0.0, -gaps[dtype], 1.0]])
                cases += [(tiny * T, far * T, T, (1.0, 100.0)) for T in (0.5, 1.0)]
                for i, (s, t, temperature, alphas) in enumerate(cases):
                    if (np.dtype(dtype).name, i) != part:
                        continue
                    ran = True
                    s, t = s.astype(dtype), t.astype(dtype)
                    for alpha in alphas:
                        with subtests.test(dtype=dtype, temperature=temperature, alpha=alpha):

                            def loss(
                                z: jax.Array,
                                t: jax.Array,
                                temperature: float = temperature,
                                alpha: float = alpha,
                            ) -> tuple[jax.Array, jax.Array]:
                                values = proximal_distillation_loss(
                                    z, t, temperature=temperature, alpha=alpha, reduction="none"
                                )
                                return values.sum(), values

                            (_, value), gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))(
                                s, t
                            )
                            ev, eg = proximal_reference(s, t, temperature, alpha)
                            np.testing.assert_allclose(value, ev, rtol=64 * eps, atol=0)
                            scale = np.abs(eg).max(-1, keepdims=True)
                            error = np.abs(np.asarray(gradient, np.float64) - eg)
                            assert np.all(error <= 64 * eps * scale + 1e-60)
        assert ran, part

    def test_jax_proximal_float32_confident_teacher(self):
        # Halley steps underflow in the float32 residual solve at teacher probabilities near
        # 1e-30: they once bisected the whole bracket, which lost most rows at T = 0.5.
        try:
            import jax
            import jax.experimental
        except ImportError:
            pytest.skip("JAX is unavailable")
        from libsdca.jax import proximal_distillation_loss

        context = getattr(jax, "enable_x64", None)
        if context is None:
            context = getattr(jax.experimental, "enable_x64", None)
        assert context is not None
        rng = np.random.default_rng(0)
        teacher = (rng.normal(size=(16, 1000)) * 10).astype(np.float32)
        student = (teacher + rng.normal(size=teacher.shape) * 0.3).astype(np.float32)
        with context():
            for temperature in (0.5, 1.0):

                def loss(z: jax.Array, t: jax.Array, temperature: float = temperature) -> jax.Array:
                    return proximal_distillation_loss(
                        z, t, temperature=temperature, alpha=1.0, reduction="sum"
                    )

                step = jax.jit(jax.value_and_grad(loss))
                value, gradient = step(student, teacher)
                expected, reference = step(student.astype(np.float64), teacher.astype(np.float64))
                np.testing.assert_allclose(value, expected, rtol=1e-6)
                np.testing.assert_allclose(gradient, reference, rtol=0, atol=1e-6)
