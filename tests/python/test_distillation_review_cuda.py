"""CUDA distillation regressions: rescue blocks at chunk and
program boundaries, padding at very high temperatures, a peaked row's dominant entry, the
centered dispatch guard of float32 rows, and the declared outputs of an aliased gradient."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _torch_grad(
    z: torch.Tensor, t: torch.Tensor, temperature: float, reuse: bool, reduction: str = "sum"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Value and gradient through the saved-state backward. reuse: the logits are a temporary
    with no other reference at backward, so backward writes the gradient over them; a hook
    on them checks that it did (the gradient arrives in their storage)."""
    from libsdca.torch import distillation_loss

    x = z.detach().clone().requires_grad_()
    options: dict[str, Any] = dict(temperature=temperature, reduction=reduction, backend="kernel")
    storage, seen = None, []
    if reuse:
        logits = x * 1
        storage = logits.data_ptr()
        logits.register_hook(lambda g: seen.append(g.data_ptr()))
        value = distillation_loss(logits, t, reuse_logits=True, **options)
        del logits
    else:
        value = distillation_loss(x, t, reuse_logits=False, **options)
    value.backward(torch.ones_like(value))
    if reuse:
        assert seen == [storage], "backward did not write the gradient over the logits"
    assert x.grad is not None
    return value.detach(), x.grad


def _jax_grad(z: torch.Tensor, t: torch.Tensor, temperature: float) -> tuple[float, np.ndarray]:
    """Value and gradient of a jitted value_and_grad (the joint CUDA path)."""
    jax = pytest.importorskip("jax")
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("needs a JAX GPU")
    import jax.numpy as jnp
    from libsdca.jax import distillation_loss

    def loss(x: Any, y: Any) -> Any:
        return distillation_loss(x, y, temperature=temperature, reduction="sum")

    with jax.enable_x64(True):
        dtype = {torch.bfloat16: jnp.bfloat16, torch.float32: jnp.float32}[z.dtype]
        x = jnp.asarray(z.float().cpu().numpy(), dtype)
        y = jnp.asarray(t.float().cpu().numpy(), dtype)
        value, grad = jax.jit(jax.value_and_grad(loss))(x, y)
    return float(value), np.asarray(grad, np.float64)


def _reference(
    z: torch.Tensor, t: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float64 values and gradients (masked classes removed)."""
    s = z.double().cpu() / temperature
    u = t.double().cpu() / temperature
    log_q = torch.log_softmax(s, -1)
    p = torch.softmax(u, -1)
    value = -torch.where(p > 0, p * log_q, 0.0).sum(-1)
    return value, (log_q.exp() - p) / temperature


@cuda
@pytest.mark.parametrize("classes", [10, 40])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rescue_blocks_stop_at_chunk_and_program_boundaries(
    classes: int, dtype: torch.dtype
) -> None:
    # Masked classes send rows to the general path; marks at 127 | 128 and 1023 | 1024
    # straddle a 128-row flag chunk and a 1024-row rescue program.
    generator = torch.Generator().manual_seed(classes)
    z = 3 * torch.randn(1100, classes, generator=generator)
    t = z + 0.5 * torch.randn(1100, classes, generator=generator)
    for r in (126, 127, 128, 129, 1022, 1023, 1024, 1025):
        z[r, 0] = t[r, 0] = -math.inf
    z, t = z.to("cuda", dtype), t.to("cuda", dtype)
    v_reuse, g_reuse = _torch_grad(z, t, 2.0, reuse=True)
    v_sep, g_sep = _torch_grad(z, t, 2.0, reuse=False)
    torch.testing.assert_close(g_reuse, g_sep, atol=0, rtol=0)
    torch.testing.assert_close(v_reuse, v_sep, atol=0, rtol=0)
    _, reference = _reference(z, t, 2.0)
    # The documented allowance: 64 (eps_out |g| + eps_work (p + q) / T min(1, gap / T)).
    s, u = z.double().cpu() / 2.0, t.double().cpu() / 2.0
    mass = (torch.softmax(s, -1) + torch.softmax(u, -1)) / 2.0
    finite = torch.isfinite(s) & torch.isfinite(u)
    gap = torch.where(finite, (s - u).abs(), 0.0).amax(-1, keepdim=True).clamp(max=1.0)
    allowance = 64 * (torch.finfo(dtype).eps * reference.abs() + 2.0**-23 * mass * gap)
    error = (g_sep.double().cpu() - reference).abs()
    assert torch.all(error <= allowance + 2.0**-149)


@cuda
@pytest.mark.parametrize("classes", [32, 33, 64])
def test_uniform_rows_at_a_temperature_far_above_the_padding(classes: int) -> None:
    z = torch.zeros(4, classes, device="cuda", dtype=torch.bfloat16)
    t = torch.full_like(z, 1e30)
    value, grad = _torch_grad(z, t, 1e31, reuse=False, reduction="none")
    np.testing.assert_allclose(value.double().cpu(), math.log(classes), rtol=1e-6)
    assert torch.count_nonzero(grad).item() == 0
    jvalue, jgrad = _jax_grad(z, t, 1e31)
    np.testing.assert_allclose(jvalue, 4 * math.log(classes), rtol=1e-6)
    assert np.count_nonzero(jgrad) == 0


@cuda
@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_peaked_row_keeps_its_dominant_entry(framework: str) -> None:
    # Student [0, 31 x -30], teacher [0, 31 x -31], T = 1: both dominant probabilities round
    # to 1 in float32; their difference is the row's largest gradient entry.
    z = torch.tensor([[0.0] + [-30.0] * 31], dtype=torch.bfloat16).repeat(4, 1).cuda()
    t = torch.tensor([[0.0] + [-31.0] * 31], dtype=torch.bfloat16).repeat(4, 1).cuda()
    expected = 31 * (math.exp(-31) - math.exp(-30))
    expected /= (1 + 31 * math.exp(-30)) * (1 + 31 * math.exp(-31))
    if framework == "torch":
        _, grad = _torch_grad(z, t, 1.0, reuse=False)
        grad = grad.double().cpu().numpy()
    else:
        _, grad = _jax_grad(z, t, 1.0)
    # The target entry within 8 BF16 ulps; the row sum within C u |g|_1.
    np.testing.assert_allclose(grad[:, 0], expected, rtol=8 * 2.0**-7)
    for row in grad:
        assert abs(math.fsum(row)) <= 32 * 2.0**-8 * math.fsum(abs(row))


@cuda
@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_float32_shifted_student_routes_alike(framework: str) -> None:
    # Practically uniform rows whose raw gap is 1 but whose centered gap is tiny.
    delta = 2.0**-18 + 2.0**-25
    t = torch.zeros(2, 32, dtype=torch.float32)
    t[:, 0] = delta
    results = []
    for shift in (1.0, 0.0):
        z = torch.full((2, 32), shift, dtype=torch.float32)
        if framework == "torch":
            _, grad = _torch_grad(z.cuda(), t.cuda(), 1.0, reuse=False)
            grad = grad.double().cpu().numpy()
        else:
            _, grad = _jax_grad(z, t, 1.0)
        results.append(grad)
    expected = -31 * math.expm1(delta) / (32 * (32 + math.expm1(delta)))
    # The documented allowance on centered logits: 64 (eps |g| + eps (p + q) min(1, gap)).
    eps = 2.0**-23
    allowance = 64 * (eps * abs(expected) + eps * (2 / 32) * delta)
    for grad in results:
        assert np.all(np.abs(grad[:, 0] - expected) <= allowance)
    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=allowance)


def test_aliased_gradient_is_a_declared_output():
    pytest.importorskip("triton")
    from libsdca.triton import common

    for classes, dtype in ((10, "bfloat16"), (100, "bfloat16"), (100, "float32")):
        for saved in (0, common.SAVED):
            call = common.Call(
                common.DISTILLATION, 64, classes, dtype, dtype, temperature=2.0,
                inplace=True, saved=saved, values=saved == 0, reduction=common.MEAN,
            )  # fmt: skip
            for launch in common.plan(call).launches:
                bound = dict(launch.args)
                name = getattr(getattr(launch.kernel, "fn", launch.kernel), "__name__", "")
                if bound.get("G") == common.Ref("Z") and name != "_reduce":
                    assert "G" in launch.state, launch.kernel


@cuda
@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_value_at_an_extreme_temperature_stays_in_range(framework: str) -> None:
    # T = 1e36 with scores near the float32 maximum: T times the teacher normalizer (1024)
    # exceeds float32, which must not truncate the value.
    z = torch.full((2, 1024), 2e38)
    z[:, :64] = 1.98e38
    z = z.to(torch.bfloat16)
    t = torch.full((2, 1024), 1.8e38).to(torch.bfloat16)
    expected, _ = _reference(z, t, 1e36)
    if framework == "torch":
        from libsdca.torch import distillation_loss

        value = distillation_loss(
            z.cuda(), t.cuda(), temperature=1e36, reduction="none", backend="kernel"
        )
        np.testing.assert_allclose(value.double().cpu(), expected, rtol=64 * 2.0**-23)
    else:
        value, grad = _jax_grad(z, t, 1e36)
        np.testing.assert_allclose(value, float(expected.sum()), rtol=64 * 2.0**-23)
        assert np.all(np.isfinite(grad))


# Float64 register rows whose maxima coincide (test_stability's collapse rows, release
# numerics' 4096 classes): (temperature, student, teacher).
_COMMON_PEAKS = [
    (1.0, [0.0, -18.3684003], [0.0, -9.18420014]),
    (1.0, [0.0, -55.1052009], [0.0, -27.5526004]),
    (
        1.0,
        [0.0, -55.1052009, -55.1052009, -56.1052009],
        [0.0, -27.5526004, -27.5526004, -28.0526004],
    ),
    (4.0, [0.0, -0.75], [0.0, -0.375]),
    (4.0, [0.0, -108.442876], [0.0, -54.2214382]),
    (2.0, [0.0] * 4096, [0.0] + [-1.0] * 4095),
]


@cuda
@pytest.mark.parametrize("framework", ["torch", "jax"])
@pytest.mark.parametrize("case", range(len(_COMMON_PEAKS)))
def test_float64_common_peaks_close_the_row_sum(framework: str, case: int) -> None:
    """The row's gradient sums to zero within its entries' roundings (C u |g|_1) and each
    entry stays within (64 + C) eps of the exact one. With common peaks the register rows
    skipped their q-times-sum correction: independently rounded entries summed to up to
    4 C u |g|_1, and at 4096 classes the log ratio of the partitions (0.5, rounded) left the
    others' entries (1e-4 of it) 4292 eps off."""
    import mpmath as mp

    temperature, student, teacher = _COMMON_PEAKS[case]
    z = torch.tensor([student], dtype=torch.float64, device="cuda")
    t = torch.tensor([teacher], dtype=torch.float64, device="cuda")
    if framework == "torch":
        _, grad = _torch_grad(z, t, temperature, reuse=False)
        grad = grad.cpu().numpy()[0]
    else:
        jax = pytest.importorskip("jax")
        if not any(d.platform == "gpu" for d in jax.devices()):
            pytest.skip("needs a JAX GPU")
        import jax.numpy as jnp
        from libsdca.jax import distillation_loss

        with jax.enable_x64(True):
            x = jnp.asarray(z.cpu().numpy(), jnp.float64)
            y = jnp.asarray(t.cpu().numpy(), jnp.float64)
            fn = jax.grad(
                lambda a: distillation_loss(a, y, temperature=temperature, reduction="sum")
            )
            grad = np.asarray(jax.jit(fn)(x), np.float64)[0]
    c = len(student)
    u = 2.0**-53
    assert abs(math.fsum(grad)) <= c * u * math.fsum(abs(grad)), grad[:4]
    with mp.workdps(50):
        s = [mp.mpf(v) / temperature for v in student]
        w = [mp.mpf(v) / temperature for v in teacher]
        zs, zt = mp.fsum(mp.e**v for v in s), mp.fsum(mp.e**v for v in w)
        for j in sorted({0, 1, c - 1}):
            exact = (mp.e ** s[j] / zs - mp.e ** w[j] / zt) / temperature
            assert abs(grad[j] - exact) <= (64 + c) * 2 * u * abs(exact), (j, grad[j], exact)


# Two-group rows of far tails, as test_triton_interpreted.test_distillation_far_tails:
# (dtype, classes, T, student gap, teacher gap in units of T, tied maxima at 0), the others
# at -gap T.
_FAR_TAILS = [
    ("float64", 10, 1.0, 740, 750, 1),
    ("float64", 1000, 0.5, 750, 740, 1),
    ("float64", 32768, 1.0, 740, 750, 1),
    ("float32", 8192, 1.0, 45, 100, 1),
    ("float32", 8192, 1.0, 50, 104, 1),
    ("float32", 8192, 1.0, 70, 120, 1),
    ("float32", 32768, 0.5, 104, 104, 1),
    ("float16", 32768, 0.5, 104, 104, 1),
    ("bfloat16", 32768, 0.5, 80, 90, 1),
    # Two tied maxima whose tails round away from 2 in float64 (equal sums).
    ("float32", 8192, 1.0, 50, 104, 2),
]


@cuda
@pytest.mark.parametrize("case", range(len(_FAR_TAILS)))
def test_far_tails_within_the_bound(case: int) -> None:
    """The value and every gradient entry within (64 + C) eps of the exact ones (and the
    smallest normal times eps below them). The rows exercise subnormal tail exponentials
    (float64 common peaks with the others beyond exp(-708)), streamed float32 rows with a
    teacher term below float32's range under a larger student term, equal tails beyond
    exp(-87) (the first chunk's share of the loss), and two tied maxima under the equal-tails
    shortcut. The row sums to zero within the entries' bounds."""
    import mpmath as mp
    from conformance.references import Block, Loss, reference

    name, c, temperature, gs, gt, lead = _FAR_TAILS[case]
    dtype = getattr(torch, name)
    z = torch.zeros((1, c), dtype=torch.float64)
    t = torch.zeros_like(z)
    z[0, lead:], t[0, lead:] = -gs * temperature, -gt * temperature
    z, t = z.to(dtype).cuda(), t.to(dtype).cuda()
    value, grad = _torch_grad(z, t, temperature, reuse=False, reduction="none")
    rest = Block(float(z[0, lead]), c - lead, teacher=float(t[0, lead]))
    row = (Block(0.0, lead, teacher=0.0), rest)
    with mp.workdps(40):
        ref = reference(Loss("distillation", temperature=temperature), row)
        work = torch.finfo(torch.float64 if name == "float64" else torch.float32)
        store = torch.finfo(dtype)
        budget = (64 + c) * work.eps
        g = grad.double().cpu()[0]
        checks = [(value.double().item(), ref.value, work.eps * work.tiny)]
        floor = store.eps * store.tiny
        checks += [(a, ref.gradient[0], floor) for a in g[:lead].unique().tolist()]
        checks += [(a, ref.gradient[1], floor) for a in g[lead:].unique().tolist()]
        for actual, exact, bottom in checks:
            assert abs(mp.mpf(actual) - exact) <= budget * abs(exact) + bottom, (actual, exact)
        size = lead * abs(ref.gradient[0]) + (c - lead) * abs(ref.gradient[1])
        total = math.fsum(g.tolist())
        assert abs(total) <= budget * size + c * floor, total
