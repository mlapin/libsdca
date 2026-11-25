"""launch_torch's direct launches (a CompiledKernel called with every argument in parameter
order, Triton's own launcher) equal the JIT's, call after call."""

from __future__ import annotations

from typing import Any

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

from libsdca import _torch_ops  # noqa: E402,F401
from libsdca.triton import common, launch_torch  # noqa: E402

CASES = [
    (common.CROSS_ENTROPY, 1, 0.0, 1.0, 0.1, "labels", 1000),
    (common.TOPK_CROSS_ENTROPY, 3, 0.0, 1.0, 0.0, "labels", 5000),
    (common.TOPK_HINGE_BETA, 2, 0.5, 1.0, 0.0, "labels", 100),
    (common.DISTILLATION, 1, 0.0, 2.0, 0.0, "teacher", 5000),
]


def _run(
    kind: int,
    k: int,
    gamma: float,
    temperature: float,
    eps: float,
    z: torch.Tensor,
    t: torch.Tensor,
    upstream: float,
):
    env = launch_torch.run(
        z, t, None, None, kind, k, gamma, temperature, eps, -100, False, 0, upstream, True, True
    )
    return env["V"].clone(), env["G"].clone()


@pytest.mark.parametrize("case", CASES, ids=lambda c: str(c[0]))
def test_cached_plans_match_fresh_plans(case: tuple):
    kind, k, gamma, temperature, eps, target, c = case
    generator = torch.Generator(device="cuda").manual_seed(0)
    calls = []
    for upstream in (1.0, -0.5, 3.0):
        z = torch.randn(33, c, device="cuda", generator=generator)
        if target == "labels":
            t = torch.randint(0, c, (33,), device="cuda", generator=generator)
        else:
            t = torch.randn(33, c, device="cuda", generator=generator)
        calls.append((z, t, upstream))
    launch_torch._prepared.cache_clear()
    direct = [_run(kind, k, gamma, temperature, eps, *call) for call in calls]
    launch_torch._prepared.cache_clear()
    jit = [_run(kind, k, gamma, temperature, eps, *call) for call in calls]
    launch_torch._prepared.cache_clear()
    for (v1, g1), (v2, g2) in zip(direct, jit, strict=True):
        torch.testing.assert_close(v1, v2, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(g1, g2, rtol=0, atol=0, equal_nan=True)


def test_direct_launch_uses_the_cotangent_dtype() -> None:
    generator = torch.Generator(device="cuda").manual_seed(241)
    scores = torch.randn(16, 96, device="cuda", dtype=torch.bfloat16, generator=generator)
    teacher = torch.randn_like(scores)
    factors = torch.tensor([0.375, -1.0, 0.0, 8.0], device="cuda").repeat(4)

    def gradient(upstream: torch.Tensor) -> torch.Tensor:
        return launch_torch.run(
            scores,
            teacher,
            None,
            upstream,
            common.DISTILLATION,
            1,
            0.0,
            2.0,
            0.0,
            -100,
            False,
            0,
            1.0,
            False,
            True,
        )["G"]

    launch_torch._prepared.cache_clear()
    reference = gradient(factors)
    assert reference.count_nonzero() > 0
    torch.testing.assert_close(gradient(factors.double()), reference, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["cross_entropy", "topk_hinge_alpha", "distillation"])
def test_eager_replays_on_the_current_stream(monkeypatch: pytest.MonkeyPatch, kind: str):
    from libsdca import torch as L

    module = _torch_ops.eager()
    if module is None or not hasattr(module, "cuda_install_plan"):
        pytest.skip("the eager CUDA plan launcher is unavailable")
    store = module.cuda_install_plan
    installed = []

    def record(*args: Any):
        accepted = store(*args)
        installed.append(accepted)
        return accepted

    monkeypatch.setattr(module, "cuda_install_plan", record)

    def evaluate(x: torch.Tensor, t: torch.Tensor):
        if kind == "cross_entropy":
            value = L.cross_entropy(x, t)
        elif kind == "topk_hinge_alpha":
            value = L.topk_hinge_loss(x, t, k=3, variant="alpha")
        else:
            value = L.distillation_loss(x, t, temperature=1.7)
        (g,) = torch.autograd.grad(value, x, torch.full_like(value, -0.75))
        return value.detach(), g

    n, c = 7, 773
    if kind == "cross_entropy":
        # A full native cache must not evict its smallest shape with each new install.
        # Vary a runtime scalar to fill it without specializing hundreds of kernels.
        larger = torch.randn(n + 1, c, device="cuda")
        labels = torch.arange(n + 1, device="cuda")
        for i in range(513):
            L.cross_entropy(larger, labels, ignore_index=-1000 - i)
        installed.clear()

    # On a new autograd worker the first backward establishes its CUDA context through
    # Triton, and the next can cache the native program in that context.
    for offset in (0, 1, 0, 1):
        x = torch.randn(n * c + offset, device="cuda")[offset:].reshape(n, c).requires_grad_()
        t = torch.randn_like(x) if kind == "distillation" else torch.arange(n, device="cuda")
        evaluate(x, t)
    if not any(installed):
        pytest.skip("the driver or Triton did not provide a supported native launch ABI")

    def unexpected(*args: Any):
        pytest.fail("a warmed eager call re-entered the Python launcher")

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for offset in (0, 1):
            x = torch.randn(n * c + offset, device="cuda")[offset:].reshape(n, c).requires_grad_()
            t = torch.randn_like(x) if kind == "distillation" else torch.arange(n, device="cuda")
            # The ordinary dispatcher launches the shared kernels to form the reference.
            with monkeypatch.context() as m:
                m.setattr(_torch_ops, "eager", lambda: None)
                reference = evaluate(x, t)
            with monkeypatch.context() as m:
                m.setattr(launch_torch._Launch, "__call__", unexpected)
                actual = evaluate(x, t)
            for a, b in zip(actual, reference, strict=True):
                torch.testing.assert_close(a, b)


@pytest.mark.parametrize(
    ("classes", "dtype"),
    [
        (100, torch.bfloat16),
        (1000, torch.float32),
        (8192, torch.float32),
        (8192, torch.float16),
        (8192, torch.bfloat16),
    ],
)
def test_compiled_backward_applies_row_cotangents(classes: int, dtype: torch.dtype) -> None:
    from libsdca import torch as losses

    generator = torch.Generator(device="cuda").manual_seed(127)
    scores = torch.randn(512, classes, device="cuda", dtype=dtype, generator=generator)
    target = torch.randn(scores.shape, device="cuda", dtype=dtype, generator=generator)
    if classes == 1000:
        target = target > 0
        loss = losses.multilabel_cross_entropy
    else:
        # A half-precision unit gradient can round to zero before a large cotangent revives it.
        scores[:, -1] = -20
        target[:, -1] = -torch.inf
        loss = losses.distillation_loss
    scores.requires_grad_()

    def rows(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return loss(z, t, reduction="none", reuse_logits=False)

    compiled = torch.compile(rows, fullgraph=True, dynamic=False, backend="aot_eager")
    value, reference = compiled(scores, target), rows(scores, target)
    torch.testing.assert_close(value, reference, rtol=0, atol=0)
    factors = torch.tensor([0, -4096, 0.125, 32], device="cuda").repeat(128)
    for upstream in (factors, -factors):
        actual = torch.autograd.grad(value, scores, upstream, retain_graph=True)[0]
        expected = torch.autograd.grad(reference, scores, upstream, retain_graph=True)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("rows", "classes"), [(64, 1000), (65, 113), (64, 1025), (129, 2049), (256, 4096), (257, 4097)]
)
@pytest.mark.parametrize("weighted", [False, True])
def test_compiled_fp32_mean_replays_actual_cotangent(
    rows: int, classes: int, weighted: bool
) -> None:
    import math

    from libsdca.torch import cross_entropy

    scores = torch.zeros((rows, classes), device="cuda")
    scores[:, 0] = 120
    labels = torch.arange(rows, device="cuda") % 2
    labels[::4] = -100
    scores[::4] = torch.nan
    weight = None
    if weighted:
        weight = torch.ones(classes, device="cuda")
        weight[0], weight[1] = 0.5, 1.25
    scores.requires_grad_()

    def loss(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return cross_entropy(z, y, weight=weight)

    torch._dynamo.reset()
    value = torch.compile(loss, fullgraph=True, dynamic=False)(scores, labels)
    # All competitors have the same tail. Form the winner's derivative from their mass.
    tail = math.exp(-120)
    count = rows - (rows + 3) // 4
    high = rows // 2
    low = count - high
    w0, w1 = (0.5, 1.25) if weighted else (1.0, 1.0)
    denominator = low * w0 + high * w1
    expected_value = math.log1p((classes - 1) * tail) + 120 * high * w1 / denominator
    torch.testing.assert_close(value, torch.full_like(value, expected_value), rtol=2e-7, atol=0)
    for factor in (0.0, 2.0**60, -1.25, math.inf, math.nan):
        upstream = torch.full_like(value, factor)
        (actual,) = torch.autograd.grad(value, scores, upstream, retain_graph=True)
        prob = tail / (1.0 + (classes - 1) * tail)
        low_row = torch.full((classes,), prob, device="cuda", dtype=torch.float64)
        low_row[0] = -(classes - 1) * prob
        high_row = torch.full_like(low_row, prob)
        high_row[0], high_row[1] = 1.0 / (1.0 + (classes - 1) * tail), prob - 1.0
        expected = torch.where(labels[:, None] == 0, low_row * w0, high_row * w1)
        expected = torch.where(labels[:, None] == -100, 0, expected * (factor / denominator))
        torch.testing.assert_close(actual, expected.float(), rtol=2e-6, atol=0, equal_nan=True)
        assert not torch.any((actual == 0) & torch.signbit(actual))


@pytest.mark.parametrize("distill", [False, True])
@pytest.mark.parametrize("reduced", [False, True])
def test_compiled_proximal_reuses_only_unit_pullbacks(distill: bool, reduced: bool) -> None:
    from libsdca import torch as losses

    gen = torch.Generator(device="cuda").manual_seed(581)
    scores = torch.randn(8, 65, device="cuda", generator=gen)
    scores[1, 0], scores[2, -1] = 100, -torch.inf
    target = torch.arange(8, device="cuda")
    target[::4] = -100
    if distill:
        target = torch.randn(scores.shape, device="cuda", generator=gen)
        target[1, 0], target[2, -1] = 100, -torch.inf
    weight = torch.linspace(0.25, 1.5, 65, device="cuda")
    scores.requires_grad_()

    def loss(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mode = "mean" if reduced else "none"
        if distill:
            return losses.proximal_distillation_loss(
                x, t, gamma=0.75, temperature=1.75, reduction=mode
            )
        return losses.proximal_cross_entropy(
            x, t, gamma=0.75, ignore_index=-100, weight=weight, reduction=mode
        )

    actual = torch.compile(loss, fullgraph=True, dynamic=False)(scores, target)
    expected = loss(scores, target)
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=0, equal_nan=True)
    for factor in (1.0, 0.0, 2.0**50, -1.25, 1.0, float("inf")):
        upstream = torch.full_like(actual, factor)
        if not reduced:
            upstream[::3] = 1.0
        a = torch.autograd.grad(actual, scores, upstream, retain_graph=True)[0]
        b = torch.autograd.grad(expected, scores, upstream, retain_graph=True)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


def test_small_compiled_labels_retain_the_gradient_after_a_zero_cotangent() -> None:
    from libsdca import torch as losses

    generator = torch.Generator(device="cuda").manual_seed(239)
    z = torch.randn(64, 1000, device="cuda", generator=generator)
    y = torch.arange(64, device="cuda")
    y[::6] = -100
    z[1::6, -3:] = -torch.inf
    z.requires_grad_()

    def loss(x: torch.Tensor) -> torch.Tensor:
        return losses.cross_entropy(x, y)

    compiled = torch.compile(loss, fullgraph=True, dynamic=False)
    actual, expected = compiled(z), loss(z)
    torch.testing.assert_close(actual, expected)
    for factor in (-12.3, 0.0, 0.125):
        cotangent = torch.full_like(actual, factor)
        g = torch.autograd.grad(actual, z, cotangent, retain_graph=True)[0]
        reference = torch.autograd.grad(expected, z, cotangent, retain_graph=True)[0]
        torch.testing.assert_close(g, reference, atol=1e-8, rtol=5e-6)


def test_short_bfloat_runtime_cotangent_factors():
    n, c = 512, 100
    gen = torch.Generator(device="cuda").manual_seed(42)
    scores = torch.randn(n, c, dtype=torch.bfloat16, device="cuda", generator=gen)
    teacher = torch.randn_like(scores)
    q = (scores.double() / 2).softmax(-1)
    p = (teacher.double() / 2).softmax(-1)
    for host, scalar in ((1 / n, None), (-0.75, None), (1.0, 1 / n), (1 / n, 1.0), (-2.0, 1 / n)):
        device = None if scalar is None else torch.tensor(scalar, device="cuda")
        result = launch_torch.run(
            scores,
            teacher,
            None,
            device,
            common.DISTILLATION,
            1,
            0.0,
            2.0,
            0.0,
            -100,
            False,
            0,
            host,
            False,
            True,
        )
        expected = (q - p) * (host * (scalar if scalar is not None else 1.0) / 2)
        torch.testing.assert_close(result["G"].double(), expected, rtol=0.0041, atol=1e-9)


def test_direct_launch_tracks_device_factor_storage():
    z = torch.randn(8, 12, device="cuda")
    t = torch.arange(8, device="cuda")
    unit = _run(common.CROSS_ENTROPY, 1, 0.0, 1.0, 0.0, z, t, 1.0)[1]
    for dtype in (torch.float16, torch.float64, torch.bfloat16, torch.float32):
        factor = torch.tensor(-0.5, device="cuda", dtype=dtype)
        actual = launch_torch.run(
            z,
            t,
            None,
            factor,
            common.CROSS_ENTROPY,
            1,
            0.0,
            1.0,
            0.0,
            -100,
            False,
            0,
            1.0,
            False,
            True,
        )
        torch.testing.assert_close(actual["G"], unit * -0.5, rtol=0, atol=0)
