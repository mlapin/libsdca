"""Direct calls of the SDCA_CPU_EXTENSION=on extension (_torch_entropy.cpp) against float64 references.

Cross-entropy with float32 exponentials (fp32) or float64, ignored targets and class weights as
torch.nn.functional.cross_entropy, top-k entropy in float32 exponentials against its float64
rows, upstream derivatives folded in before the one rounding of 16-bit gradients, distillation
(also with a common student/teacher offset), the proximal kernels' float32 gradients of 16-bit
logits, input errors, thread counts and per-call state.
"""

import math
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from libsdca import _cpu_entropy

pytestmark = pytest.mark.skipif(
    not _cpu_entropy.enabled(), reason="SDCA_CPU_EXTENSION=on builds the CPU extension"
)

DTYPES = (torch.float32, torch.bfloat16, torch.float16, torch.float64)
REDUCTIONS = {"none": 0, "mean": 1, "sum": 2}
EPS32 = torch.finfo(torch.float32).eps


@pytest.fixture(scope="module")
def ext() -> Any:
    return _cpu_entropy.operators()


def entropy_reference(z: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Losses and gradients in float64 from the competitors' maximum, accurate also for tiny
    losses of confident rows (log_softmax rounds those at the maximum's scale)."""
    z = z.double()
    label = z.gather(1, y[:, None]).squeeze(1)
    others = z.scatter(1, y[:, None], -math.inf)
    maximum = others.amax(1)
    log_sum = (others - maximum[:, None]).exp().sum(1).log()
    relative = (maximum - label) + log_sum
    loss = torch.where(
        relative > 0, relative + torch.log1p(torch.exp(-relative)), torch.log1p(relative.exp())
    )
    mass = torch.sigmoid(relative)
    grad = ((others - maximum[:, None]) - log_sum[:, None]).exp() * mass[:, None]
    return loss, grad.scatter(1, y[:, None], -mass[:, None])


def entropy_rows(
    dtype: torch.dtype, n: int = 96, c: int = 777
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordinary rows at three scales, confident rows (tiny losses) and huge spreads."""
    gen = torch.Generator().manual_seed(1)
    z = torch.randn(n, c, generator=gen, dtype=torch.float64)
    z[: n // 4] *= 10
    z[n // 4 : n // 2] *= 3
    y = torch.randint(0, c, (n,), generator=gen)
    rows = torch.arange(n)
    big = 60.0 if dtype == torch.float16 else 1e4
    z[n // 2 : 5 * n // 8, :5] = -big  # huge spreads
    confident = rows[5 * n // 8 : 3 * n // 4]
    z[confident, y[confident]] += 30  # losses near exp(-30)
    z[3 * n // 4 :] *= 0.01  # nearly uniform rows
    return z.to(dtype), y


def check_gradient(g: torch.Tensor, ref: torch.Tensor, y: torch.Tensor, ulps: float) -> None:
    """Target entries within ulps of the dtype (relative); competitors within ulps of the row's
    largest entry or one subnormal unit."""
    info = torch.finfo(g.dtype)
    g64 = g.double()
    target, ref_target = g64.gather(1, y[:, None]), ref.gather(1, y[:, None])
    assert (
        (target - ref_target).abs() <= ulps * info.eps * ref_target.abs() + info.tiny * info.eps
    ).all()
    scale = ref.abs().amax(1, keepdim=True)
    error = (g64 - ref).abs().scatter(1, y[:, None], 0)
    assert (error <= ulps * info.eps * scale + info.tiny * info.eps).all(), (error / scale).max()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fp32", [True, False])
def test_entropy_matches_reference(ext: Any, dtype: torch.dtype, fp32: bool) -> None:
    z, y = entropy_rows(dtype)
    loss_ref, grad_ref = entropy_reference(z, y)
    loss, stats, represented = ext.forward(z, y, 0, -100, None, fp32, True)
    assert represented and stats.shape == (3 * z.shape[0] + 1,) and stats.dtype == torch.float64
    assert loss.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    tolerance = 1e-13 if dtype == torch.float64 else 8 * EPS32
    assert ((loss.double() - loss_ref).abs() <= tolerance * loss_ref).all()
    upstream = torch.rand(
        z.shape[0], dtype=torch.float64, generator=torch.Generator().manual_seed(2)
    )
    g = ext.backward(z, y, stats, upstream.float(), 0, -100, None, fp32)
    assert g.dtype == dtype
    check_gradient(
        g, grad_ref * upstream.float().double()[:, None], y, 8 if dtype != torch.float64 else 64
    )
    for reduction, name in ((1, "mean"), (2, "sum")):
        value, stats, _ = ext.forward(z, y, reduction, -100, None, fp32, True)
        expected = loss_ref.mean() if name == "mean" else loss_ref.sum()
        assert abs(value.item() - expected.item()) <= tolerance * expected.item()
        g = ext.backward(z, y, stats, torch.tensor(3.0), reduction, -100, None, fp32)
        scale = 3.0 / z.shape[0] if name == "mean" else 3.0
        check_gradient(g, grad_ref * scale, y, 8 if dtype != torch.float64 else 64)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fp32", [True, False])
def test_upstream_scales_before_the_one_rounding(ext: Any, dtype: torch.dtype, fp32: bool) -> None:
    # A GradScaler-sized upstream: rounding first and then scaling would lose the small entries
    # (16-bit subnormals) and, for float16, overflow.
    z, y = entropy_rows(dtype, n=64)
    _, grad_ref = entropy_reference(z, y)
    _, stats, _ = ext.forward(z, y, 1, -100, None, fp32, True)
    g = ext.backward(z, y, stats, torch.tensor(65536.0), 1, -100, None, fp32)
    assert g.isfinite().all()
    if dtype == torch.float32:
        check_gradient(g, grad_ref * (65536.0 / 64), y, 8)
    else:
        # One 16-bit rounding: an ulp from that of the float64 reference.
        expected = (grad_ref * (65536.0 / 64)).to(dtype).double()
        ulp = torch.finfo(dtype).eps * expected.abs().clamp_min(torch.finfo(dtype).tiny)
        assert ((g.double() - expected).abs() <= ulp).all()


def cross_entropy_reference(
    z: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor | None,
    ignore_index: int,
    reduction: str,
    upstream: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z64 = z.double().nan_to_num(0.0).requires_grad_(True)
    w = None if weight is None else weight.double()
    loss = F.cross_entropy(z64, y, weight=w, ignore_index=ignore_index, reduction=reduction)
    (grad,) = torch.autograd.grad((loss * upstream.double()).sum(), z64)
    return loss.detach(), grad


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("weighted", ["none", "random", "zeros"])
@pytest.mark.parametrize("ignore_index", [-100, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_ignore_index_and_weights_match_pytorch(
    ext: Any,
    reduction: str,
    weighted: str,
    ignore_index: int,
    dtype: torch.dtype,
) -> None:
    gen = torch.Generator().manual_seed(3)
    n, c = 40, 11
    z = torch.randn(n, c, generator=gen, dtype=torch.float64).to(dtype)
    y = torch.randint(0, c, (n,), generator=gen)
    y[::5] = ignore_index
    # Ignored rows are neither read nor validated.
    z[::5] = math.nan
    weight = None
    if weighted != "none":
        weight = (torch.rand(c, generator=gen, dtype=torch.float64) + 0.5).to(dtype)
        if weighted == "zeros":
            weight[:4] = 0
    upstream = torch.rand(n if reduction == "none" else (), generator=gen, dtype=torch.float64)
    loss_ref, grad_ref = cross_entropy_reference(z, y, weight, ignore_index, reduction, upstream)
    code = REDUCTIONS[reduction]
    loss, stats, _ = ext.forward(z, y, code, ignore_index, weight, True, True)
    tolerance = 1e-12 if dtype == torch.float64 else 8 * EPS32
    torch.testing.assert_close(loss.double(), loss_ref, rtol=tolerance, atol=0)
    g = ext.backward(z, y, stats, upstream, code, ignore_index, weight, True)
    assert (g[::5] == 0).all()
    scale = grad_ref.abs().amax()
    eps = torch.finfo(dtype).eps
    assert ((g.double() - grad_ref).abs() <= 8 * eps * scale).all()


@pytest.mark.parametrize("ignore_index", [-100, 0])
def test_all_ignored_and_zero_weights(ext: Any, ignore_index: int) -> None:
    z = torch.randn(6, 4)
    y = torch.full((6,), ignore_index)
    for reduction, code in REDUCTIONS.items():
        loss, stats, represented = ext.forward(z, y, code, ignore_index, None, True, True)
        expected = F.cross_entropy(z, y, ignore_index=ignore_index, reduction=reduction)
        torch.testing.assert_close(loss, expected, equal_nan=True)
        assert represented
        upstream = torch.ones(6) if reduction == "none" else torch.tensor(1.0)
        assert (ext.backward(z, y, stats, upstream, code, ignore_index, None, True) == 0).all()
    # Every non-ignored target weighs zero: a NaN mean, as in PyTorch.
    y = torch.tensor([1, 1, ignore_index, 2, 1, 1])
    weight = torch.tensor([1.0, 0.0, 0.0, 1.0])
    loss, _, _ = ext.forward(z, y, 1, ignore_index, weight, True, True)
    assert loss.isnan() and F.cross_entropy(z, y, weight, ignore_index=ignore_index).isnan()


def test_input_errors(ext: Any) -> None:
    z = torch.randn(8, 5)
    y = torch.randint(0, 5, (8,))
    bad_target = y.clone()
    bad_target[3] = 5
    nonfinite = z.clone()
    nonfinite[6, 2] = math.inf
    finite_msg, target_msg = (
        "logits must be finite floating-point values",
        "Target index out of range",
    )
    for fp32 in (True, False):
        with pytest.raises(ValueError, match=finite_msg):
            ext.forward(nonfinite, y, 1, -100, None, fp32, False)
        with pytest.raises(ValueError, match=target_msg):
            ext.forward(z, bad_target, 1, -100, None, fp32, False)
        # The torch layer checks finiteness first.
        with pytest.raises(ValueError, match=finite_msg):
            ext.forward(nonfinite, bad_target, 1, -100, None, fp32, False)
    # Ignored rows are not validated.
    ignored = bad_target.clone()
    ignored[3] = -100
    nonfinite[6] = 0
    nonfinite[3] = math.nan
    assert ext.forward(nonfinite, ignored, 1, -100, None, True, False)[0].isfinite()
    with pytest.raises(ValueError, match="weight must be finite and nonnegative"):
        ext.forward(z, y, 1, -100, torch.tensor([1.0, -1, 1, 1, 1]), True, False)
    for call in (
        lambda z, y: ext.hinge_forward(z, y, 1, False, 1.0),
        lambda z, y: ext.hinge_forward(z, y, 1, True, 1.0),
        lambda z, y: ext.topk_forward(z, y, 1, 0, 2, 1.0, False),
        lambda z, y: ext.topk_forward(z, y, 1, 0, 2, 1.0, True),
        lambda z, y: ext.topk_forward(z, y, 1, 3, 2, 1.0, False),
    ):
        with pytest.raises(ValueError, match=finite_msg):
            call(z.clone().index_fill_(0, torch.tensor([6]), math.nan), bad_target)
        with pytest.raises(ValueError, match=target_msg):
            call(z, bad_target)
    for fp32 in (True, False):
        prox = ext.prox_entropy_forward
        with pytest.raises(ValueError, match=finite_msg):
            prox(z.clone().index_fill_(0, torch.tensor([6]), math.nan), y, 1.0, fp32)
        with pytest.raises(ValueError, match=finite_msg):
            prox(z.clone().index_fill_(0, torch.tensor([6]), math.nan), bad_target, 1.0, fp32)
        with pytest.raises(ValueError, match=target_msg):
            prox(z, bad_target, 1.0, fp32)
        positives = torch.zeros(8, 5, dtype=torch.bool)
        positives[:, 1] = True
        with pytest.raises(ValueError, match="positive label"):
            prox(z, positives.index_fill(0, torch.tensor([2]), False), 1.0, fp32)
        with pytest.raises(ValueError, match="negative label"):
            prox(z, positives.index_fill(0, torch.tensor([2]), True), 1.0, fp32)
    mask = torch.zeros(8, 5, dtype=torch.uint8)
    mask[:, 0] = 1
    with pytest.raises(ValueError, match=finite_msg):
        ext.multilabel_forward(
            z.clone().index_fill_(0, torch.tensor([2]), math.nan), mask, 1, 1, 1.0
        )
    empty = mask.clone()
    empty[4] = 0
    with pytest.raises(ValueError, match="Every example needs a positive label"):
        ext.multilabel_forward(z, empty, 1, 1, 1.0)
    full = mask.clone()
    full[4] = 1
    with pytest.raises(ValueError, match="This loss needs a negative label"):
        ext.multilabel_forward(z, full, 1, 1, 1.0)
    with pytest.raises(ValueError, match="Logits must be finite floating-point tensors"):
        ext.distillation_forward(nonfinite, z, 2.0, False, False)


@pytest.mark.parametrize("dtype", DTYPES)
def test_multilabel_entropy_without_stats_keeps_the_losses(ext: Any, dtype: torch.dtype) -> None:
    """Boolean-mask entropy without stats for a backward (need=False) gives the same losses,
    bit for bit: float32 exponentials, double lanes and rows beyond their ranges."""
    gen = torch.Generator().manual_seed(3)
    z = (3 * torch.randn(64, 37, generator=gen, dtype=torch.float64)).to(dtype)
    z[:8, 0] = -100 if dtype == torch.float16 else -1000  # beyond the lanes' spreads
    mask = torch.rand(64, 37, generator=gen) < 0.2
    mask[:, 1] = True
    mask[:, 2] = False
    for fp32 in (False, True):
        for reduction in (0, 1, 2):
            with_stats = ext.multilabel_forward(z, mask, reduction, 0, 0.0, fp32, True)[0]
            without = ext.multilabel_forward(z, mask, reduction, 0, 0.0, fp32, False)[0]
            assert torch.equal(with_stats, without), (fp32, reduction)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("k", [2, 5, 64])
def test_float32_topk_entropy_matches_double_rows(ext: Any, dtype: torch.dtype, k: int) -> None:
    """The fp32 top-k entropy rows (one sweep of float32 exponentials over blocks of 2048
    logits) against the extension's float64 rows: ordinary, confident and huge-spread rows,
    16-bit ties, labels at the block edges."""
    z, y = entropy_rows(dtype, c=4099)
    y[:4] = torch.tensor([0, 2047, 2048, 4098])
    upstream = torch.rand(z.shape[0], generator=torch.Generator().manual_seed(3))
    results = []
    for fp32 in (True, False):
        loss, stats = ext.topk_forward(z, y, 0, 0, k, 1.0, fp32)
        results.append((loss, ext.topk_backward(z, y, stats, upstream, 0, 0, k, 1.0, fp32)))
    (loss, g), (loss_ref, g_ref) = results
    assert loss.dtype == loss_ref.dtype == torch.float32
    assert ((loss.double() - loss_ref.double()).abs() <= 8 * EPS32 * loss_ref.double()).all()
    check_gradient(g, g_ref.double(), y, 8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_thread_count_invariance(ext: Any, dtype: torch.dtype) -> None:
    z, y = entropy_rows(dtype, n=256, c=3000)
    teacher = (z.double() + torch.randn(z.shape, dtype=torch.float64)).to(dtype)
    upstream = torch.rand(256)
    threads = torch.get_num_threads()
    results = []
    try:
        for count in (1, max(2, threads)):
            torch.set_num_threads(count)
            loss, stats, _ = ext.forward(z, y, 1, -100, None, True, True)
            value, saved = ext.distillation_forward(z, teacher, 2.0, False, True)
            results.append(
                (
                    loss,
                    ext.backward(z, y, stats, torch.tensor(1.0), 1, -100, None, True),
                    value,
                    ext.distillation_backward(z, teacher, saved, upstream, 2.0, False),
                )
            )
    finally:
        torch.set_num_threads(threads)
    for first, second in zip(*results, strict=True):
        assert torch.equal(first, second)


def test_no_state_between_calls(ext: Any) -> None:
    # Row statistics only (no logits-sized saved state), fresh per call.
    z, y = entropy_rows(torch.float32, n=512, c=4000)
    _, first, _ = ext.forward(z, y, 1, -100, None, True, True)
    snapshot = first.clone()
    _, second, _ = ext.forward(
        z[:256].contiguous(), y[:256].contiguous(), 0, -100, None, True, True
    )
    assert first.numel() == 3 * 512 + 1 and second.numel() == 3 * 256 + 1
    assert torch.equal(first, snapshot)
    assert ext.forward(z, y, 1, -100, None, True, False)[1] is None
    _, saved = ext.distillation_forward(z, z, 2.0, False, True)
    assert saved.shape == (512, 7)


def log_softmax_reference(x: torch.Tensor) -> torch.Tensor:
    """log_softmax of float64 rows, accurate near 0 (the tail is summed without its maximum)."""
    maximum, index = x.max(1, keepdim=True)
    tail = (x - maximum).exp().scatter(1, index, 0).sum(1, keepdim=True)
    return (x - maximum) - torch.log1p(tail)


def distillation_reference(
    student: torch.Tensor, teacher: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    s, t = student.double(), teacher.double()
    log_q = log_softmax_reference((s - s.amax(1, keepdim=True)) / temperature)
    log_p = log_softmax_reference((t - t.amax(1, keepdim=True)) / temperature)
    p, ratio = log_p.exp(), log_q - log_p
    u = torch.where(ratio.abs() < 1, p * torch.expm1(ratio), log_q.exp() - p)
    return -torch.where(p > 0, p * log_q, 0).sum(1), u / temperature


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize(
    ("temperature", "scale", "offset"),
    [(1.0, 1, 0), (2.0, 3, 0), (0.5, 1, 0), (4.0, 1, 1e4), (1.0, 20, 0)],
)
def test_distillation_matches_reference(
    ext: Any,
    dtype: torch.dtype,
    double: bool,
    temperature: float,
    scale: float,
    offset: float,
) -> None:
    if dtype == torch.float16 and (offset or scale > 3):
        pytest.skip("beyond float16 resolution")
    gen = torch.Generator().manual_seed(4)
    base = torch.randn(48, 1000, generator=gen, dtype=torch.float64) * scale
    student = (base + offset).to(dtype)
    teacher = (base + 0.1 * torch.randn(48, 1000, generator=gen, dtype=torch.float64)).to(dtype)
    value, stats = ext.distillation_forward(student, teacher, temperature, double, True)
    upstream = torch.rand(48, generator=gen, dtype=torch.float64) * 100
    g = ext.distillation_backward(student, teacher, stats, upstream, temperature, double)
    loss_ref, grad_ref = distillation_reference(student, teacher, temperature)
    work64 = double or dtype == torch.float64
    tolerance = 1e-12 if dtype == torch.float64 else (1 if work64 else 8) * EPS32
    assert ((value.double() - loss_ref).abs() <= tolerance * loss_ref).all()
    grad_ref = grad_ref * upstream[:, None]
    eps = torch.finfo(dtype).eps
    # float32 work: the float32 log-ratio of the represented logits. The float64 reference
    # rounds its log-ratio at the scale of the log-probabilities (a few 1e-14).
    bound = (
        1e-12 if dtype == torch.float64 else (1 if work64 or dtype != torch.float32 else 32) * eps
    )
    scale_ref = grad_ref.abs().amax(1, keepdim=True)
    assert ((g.double() - grad_ref).abs() <= bound * scale_ref).all()


def test_distillation_common_offset(ext: Any) -> None:
    # The gradient does not depend on a common student/teacher offset (regression:
    # float32 work rounded cs - ct at the offset's scale, 0.3% at an offset of 1e4).
    gen = torch.Generator().manual_seed(2)
    base = torch.randn(64, 10, generator=gen)
    teacher = base + torch.randn(64, 10, generator=gen) * 0.1
    for offset in (0.0, 300.0, 1e4):
        student = base + offset
        _, grad_ref = distillation_reference(student, teacher, 1.0)
        for double in (False, True):
            _, stats = ext.distillation_forward(student, teacher, 1.0, double, True)
            g = ext.distillation_backward(student, teacher, stats, torch.ones(64), 1.0, double)
            assert (g.double() - grad_ref).abs().max() <= 4 * EPS32 * grad_ref.abs().max()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("c", [7, 40])
def test_distillation_saved_gradient_beyond_the_correction_range(
    ext: Any, dtype: torch.dtype, c: int
) -> None:
    # Rows whose correction sum p exp(dn) - 1 overflows the work type (its log beyond 88 in
    # float32, 709 in float64) take float64 or distillation_stats, not NaN gradients
    # (regression: 1% of the rows of randn * 20 logits at T = 0.01 in float32). c = 7 takes
    # the small-row blocks first, c = 40 the per-row kernels.
    student = torch.zeros(2, c, dtype=torch.float64)
    teacher = torch.full((2, c), -708.0, dtype=torch.float64)
    student[:, -1], teacher[:, -1] = -708.0, 0.0
    student[0, :7] = torch.tensor([2.884, 14.892, 11.030, 15.019, 15.159, 7.772, -1.141])
    teacher[0, :7] = torch.tensor([2.281, 8.821, 11.924, 11.624, 4.790, 1.015, -1.775])
    for row, temperature in enumerate((0.01, 1.0)):
        s = student[row : row + 1].to(dtype)
        t = teacher[row : row + 1].to(dtype)
        value, g = ext.distillation_forward_grad(s, t, temperature)
        loss_ref, grad_ref = distillation_reference(s, t, temperature)
        assert loss_ref.item() > 400
        tolerance = 1e-12 if dtype == torch.float64 else 8 * EPS32
        assert ((value - loss_ref).abs() <= tolerance * loss_ref).all()
        # The saved gradient is the gradient times the temperature.
        bound = (1e-12 if dtype == torch.float64 else 32 * EPS32) * grad_ref.abs().max()
        assert (g.double() / temperature - grad_ref).abs().max() <= bound


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_distillation_opposed_rows(ext: Any, dtype: torch.dtype) -> None:
    # Student and teacher rank the classes in opposite orders: the shift -mean(delta) / T -
    # log1p(teacher's tail) is about 20 (float32 work, exp(s + shift) as exp(s) exp(shift))
    # and about 60 (beyond 32: float64), through the statistics and the saved gradient.
    ramp = torch.linspace(0, -20, 64, dtype=torch.float64)
    student = torch.stack([ramp, 1.5 * ramp]).to(dtype)
    teacher = student.flip(1).contiguous()
    for temperature in (1.0, 0.5):
        loss_ref, grad_ref = distillation_reference(student, teacher, temperature)
        _, stats = ext.distillation_forward(student, teacher, temperature, False, True)
        ones = torch.ones(2, dtype=torch.float64)
        gradients = [ext.distillation_backward(student, teacher, stats, ones, temperature, False)]
        if dtype == torch.float32:
            value, saved = ext.distillation_forward_grad(student, teacher, temperature)
            assert ((value - loss_ref).abs() <= 8 * EPS32 * loss_ref).all()
            gradients.append(saved / temperature)
        eps = torch.finfo(dtype).eps
        for g in gradients:
            bound = eps * grad_ref.abs() + 32 * EPS32 * grad_ref.abs().amax(1, keepdim=True)
            assert ((g.double() - grad_ref).abs() <= bound).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_distillation_upstream_before_rounding(ext: Any, dtype: torch.dtype) -> None:
    gen = torch.Generator().manual_seed(5)
    student = (torch.randn(32, 500, generator=gen) * 4).to(dtype)
    teacher = (torch.randn(32, 500, generator=gen) * 4).to(dtype)
    _, stats = ext.distillation_forward(student, teacher, 2.0, False, True)
    g = ext.distillation_backward(student, teacher, stats, torch.full((32,), 2048.0), 2.0, False)
    _, grad_ref = distillation_reference(student, teacher, 2.0)
    expected = (grad_ref * 2048).to(dtype).double()
    assert g.isfinite().all()
    ulp = torch.finfo(dtype).eps * expected.abs().clamp_min(torch.finfo(dtype).tiny)
    assert (
        (g.double() - expected).abs()
        <= ulp + 4 * EPS32 * grad_ref.abs().amax(1, keepdim=True) * 2048
    ).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_proximal_gradients_float32_for_16_bit(ext: Any, dtype: torch.dtype) -> None:
    z = torch.randn(8, 30).to(dtype)
    y = torch.randint(0, 30, (8,))
    _, gradient = ext.prox_entropy_forward(z, y, 0.5, True)
    assert gradient.dtype == torch.float32
    _, gradient = ext.prox_distillation_forward(z, z.flip(1).contiguous(), 2.0, 0.5)
    assert gradient.dtype == torch.float32
