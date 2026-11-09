"""Common calls of every PyTorch loss on the CPU paths of this process's SDCA_CPU_EXTENSION
(each backend, default and one thread) against float64 references (common_inputs.py), and
the contracts of ordinary misuse: invalid targets and logits raise, no_grad and
inference_mode give the values of grad mode, empty batches, autocast."""

import os
import random
from typing import Any

import common_inputs as ci
import pytest
import torch
import torch.nn.functional as F
from libsdca import _cpu
from libsdca.torch import sdca_loss

# The mode of the suite: the release gate's torch suites run with SDCA_CPU_EXTENSION on and off.
MODE = os.environ.get("SDCA_CPU_EXTENSION", "auto") or "auto"
CASES_PER_LOSS = 60
pytestmark = pytest.mark.skipif(not _cpu.AVAILABLE, reason="needs the native library")


def cases_of(loss: str, count: int = CASES_PER_LOSS) -> list[tuple[int, ci.Case]]:
    """The first count cases of loss in the stream of random_case(Random(i))."""
    found, i = [], 0
    while len(found) < count:
        case = ci.random_case(random.Random(i))
        if case.loss == loss:
            found.append((i, case))
        i += 1
    return found


@pytest.mark.parametrize("loss", ci.LOSSES)
def test_common_calls_match_references(loss: str):
    """Random reductions, upstream factors, layouts, dtypes, targets and forms; a failure
    names the case."""
    failures = []
    for index, case in cases_of(loss):
        for result in ci.check(case, extension_modes=(MODE,)):
            if not result.ok:
                failures.append(f"case {index} [{case.describe()}] {result.path}: {result.message}")
    assert not failures, "\n".join(failures)


def test_the_check_catches_a_dropped_upstream_factor(monkeypatch: pytest.MonkeyPatch):
    """A gradient that ignores the upstream factor (loss * w) fails the check."""
    case = ci.Case(
        "topk_hinge_beta",
        8,
        10,
        torch.float32,
        "sum",
        "contiguous",
        3,
        k=2,
        gamma=0.5,
        upstream=-2.0,
    )
    assert all(r.ok for r in ci.check(case, (MODE,), (None,)))
    scaled = _cpu.scaled
    monkeypatch.setattr(_cpu, "scaled", lambda g, up, *a: scaled(g, torch.ones_like(up), *a))
    native = [r for r in ci.check(case, (MODE,), (None,)) if "native" in r.path]
    assert native and not any(r.ok for r in native)


def test_the_references_agree_with_independent_ones():
    """The closed-form cross-entropy (weights, ignore_index), the envelope and the
    nonsmooth hinges against F.cross_entropy, a scalar solve and the mpmath references."""
    import test_topk_kernels as tk

    g = torch.Generator().manual_seed(0)
    z = torch.randn(40, 6, generator=g, dtype=torch.float64) * 3
    y = torch.randint(0, 6, (40,), generator=g)
    y[::4] = -100
    w = torch.rand(6, generator=g, dtype=torch.float64)
    for reduction in ("none", "mean", "sum"):
        case = ci.Case(
            "ce",
            40,
            6,
            torch.float64,
            reduction,
            "contiguous",
            0,
            weight=True,
            ignore=-100,
            upstream=0.5,
        )
        data = dict(z=z, target=y, teacher=None, weight=w, class_dim=-1)
        exp = ci.expected(case, data)
        assert exp is not None
        x = z.clone().requires_grad_()
        value = F.cross_entropy(x, y, w, reduction=reduction)
        value.backward(ci._upstream(case, value.detach()))
        torch.testing.assert_close(exp.value, value.detach(), rtol=1e-13, atol=0)
        torch.testing.assert_close(exp.grad, x.grad, rtol=1e-12, atol=1e-15)
    # Two classes: u1 - u0 = d solves d + 2 alpha (sigmoid(d) - t1) = x1 - x0.
    x, t, alpha = torch.tensor([[0.3, -1.7]], dtype=torch.float64), torch.tensor([[1.0, 0]]), 3.0
    value, grad = ci._envelope(x, t.double(), alpha)
    d = torch.tensor(0.0, dtype=torch.float64)
    for _ in range(100):
        d = d - (d + 2 * alpha * torch.sigmoid(d) - (x[0, 1] - x[0, 0])) / (
            1 + 2 * alpha * torch.sigmoid(d) * torch.sigmoid(-d)
        )
    torch.testing.assert_close(grad[0, 1], torch.sigmoid(d), rtol=1e-14, atol=0)
    # A confident row far from its target (mpmath, 50 digits).
    row = [26.125, 5.03125, -5.6875, -18.125, 16.125, 5.09375, 3.5625, 2.109375, 0.494140625]
    x = torch.tensor([[*row, -6.03125]], dtype=torch.float64)
    value, grad = ci._envelope(x, F.one_hot(torch.tensor([4]), 10).double(), 10.0)
    torch.testing.assert_close(value.item(), 3.193147301431146, rtol=1e-14, atol=0)
    expected = [0.49999993956443593, 5.1232157898486977e-8, 1.1335714344744246e-12]
    expected += [4.4968797532556213e-18, -0.50000006043556407]
    torch.testing.assert_close(grad[0, :5], torch.tensor(expected, dtype=torch.float64))
    # The nonsmooth hinges at rows without ties.
    rows = torch.randn(10, 7, generator=g, dtype=torch.float64)
    labels = torch.randint(0, 7, (10,), generator=g)
    for name in ("topk_hinge_alpha", "topk_hinge_beta"):
        values, grads, ambiguous = ci._hinge_rows(rows, labels, name, 3)
        assert not ambiguous.any()
        for row, label, v, gr in zip(rows.tolist(), labels.tolist(), values, grads, strict=True):
            rv, rg = tk.reference(row, label, name, 3)
            torch.testing.assert_close(v.item(), rv, rtol=1e-14, atol=1e-15)
            torch.testing.assert_close(gr, torch.tensor(rg, dtype=torch.float64))


LOSSES = (
    "cross_entropy",
    "multilabel_cross_entropy",
    "topk_cross_entropy",
    "multiclass_hinge",
    "topk_hinge_alpha",
    "topk_hinge_beta",
    "multilabel_hinge",
    "proximal_cross_entropy",
)
BACKENDS = ("auto", "native", "torch")


def _loss(
    name: str, z: torch.Tensor, target: torch.Tensor, backend: str, **options: Any
) -> torch.Tensor:
    k = 2 if name.startswith("topk") else 1
    return sdca_loss(z, target, name, backend=backend, k=k, **options)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("loss", LOSSES)
def test_invalid_targets_and_logits_raise(loss: str, backend: str):
    """Out-of-range and ignore-like labels, other label dtypes and shapes, empty or full
    masks and nonfinite logits raise ValueError, with or without gradients."""
    z = torch.randn(6, 5, generator=torch.Generator().manual_seed(1))
    y = torch.tensor([0, 1, 2, 3, 4, 0])
    bad_targets = [
        torch.full((6,), 5),
        torch.full((6,), -1),
        torch.full((6,), -100),
        y.int(),
        y.to(torch.uint8),
        y.float(),
        y[:5],
        torch.zeros(6, 5, dtype=torch.bool),
    ]
    if loss in ("multilabel_hinge", "proximal_cross_entropy"):
        bad_targets.append(torch.ones(6, 5, dtype=torch.bool))
    for target in bad_targets:
        with pytest.raises(ValueError):
            _loss(loss, z, target, backend)
    for value in (float("nan"), float("inf"), -float("inf")):
        bad = z.clone()
        bad[2, 3] = value
        with pytest.raises(ValueError):
            _loss(loss, bad.requires_grad_(), y, backend)
        with torch.no_grad(), pytest.raises(ValueError):
            _loss(loss, bad, y, backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("loss", LOSSES)
def test_no_grad_and_inference_mode_give_the_values(loss: str, backend: str):
    z = torch.randn(6, 5, generator=torch.Generator().manual_seed(2))
    y = torch.tensor([0, 1, 2, 3, 4, 0])
    for reduction in ("none", "mean", "sum"):
        expected = _loss(loss, z.clone().requires_grad_(), y, backend, reduction=reduction)
        with torch.no_grad():
            assert torch.equal(_loss(loss, z, y, backend, reduction=reduction), expected.detach())
        with torch.inference_mode():
            inference = z.clone()
            assert torch.equal(
                _loss(loss, inference, y, backend, reduction=reduction), expected.detach()
            )
        # Inference tensors are valid inputs outside inference mode.
        assert torch.equal(
            _loss(loss, inference, y, backend, reduction=reduction), expected.detach()
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("loss", LOSSES)
def test_empty_batches_follow_torch_functional(loss: str, backend: str):
    for reduction in ("none", "mean", "sum"):
        z = torch.randn(0, 5, requires_grad=True)
        value = _loss(loss, z, torch.zeros(0, dtype=torch.long), backend, reduction=reduction)
        value.sum().backward()
        assert z.grad is not None and z.grad.shape == (0, 5)
        if reduction == "none":
            assert value.shape == (0,)
        elif reduction == "mean":
            assert value.isnan()
        else:
            assert value.item() == 0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("loss", LOSSES)
def test_cpu_autocast_bfloat16(loss: str, backend: str):
    """A bfloat16 layer under CPU autocast: a float32 loss and gradients into the layer."""
    g = torch.Generator().manual_seed(3)
    layer = torch.nn.Linear(4, 5)
    x = torch.randn(6, 4, generator=g)
    y = torch.tensor([0, 1, 2, 3, 4, 0])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = layer(x)
        value = _loss(loss, logits, y, backend, gamma=0.5)
    assert logits.dtype == torch.bfloat16 and value.dtype == torch.float32
    value.backward()
    assert layer.weight.grad is not None and layer.weight.grad.isfinite().all()
    expected = _loss(loss, logits.detach().float(), y, "torch", gamma=0.5, precision="float64")
    torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("loss", [*LOSSES, "ce", "distillation", "proximal_distillation"])
def test_upstream_layouts(loss: str, backend: str):
    """loss(reduction="none").sum(), (loss * w).sum() with w broadcast and a strided upstream
    give the gradients of a dense upstream with the same values."""
    g = torch.Generator().manual_seed(4)
    z = torch.randn(2, 6, 5, generator=g)
    y = torch.randint(0, 5, (2, 6), generator=g)
    teacher = torch.randn(2, 6, 5, generator=g)

    def value(x: torch.Tensor) -> torch.Tensor:
        if loss == "ce":
            return ci.L.cross_entropy(x.movedim(-1, 1), y, reduction="none", backend=backend)
        if loss == "distillation":
            return ci.L.distillation_loss(x, teacher, 2.0, reduction="none")
        if loss == "proximal_distillation":
            return ci.L.proximal_distillation_loss(x, teacher, 2.0, 0.5, reduction="none")
        return _loss(loss, x, y, backend, reduction="none", gamma=0.5)

    w = torch.rand(2, 6, generator=g) - 0.3
    upstreams = {
        "sum": (lambda v: v.sum(), torch.ones(2, 6)),
        "broadcast": (lambda v: (v * w[:, :1]).sum(), w[:, :1].expand(2, 6)),
        "transposed": (lambda v: (v.t() * w.t().contiguous()).sum(), w),
        "strided": (lambda v: v.backward(torch.stack([w, w], -1)[..., 0]), w),
    }
    if loss.endswith("distillation") and backend != "auto":
        pytest.skip("distillation has no backend option")
    for name, (use, dense) in upstreams.items():
        x = z.clone().requires_grad_()
        out = use(value(x))
        if out is not None:
            out.backward()
        expected = z.clone().requires_grad_()
        value(expected).backward(dense.contiguous())
        torch.testing.assert_close(x.grad, expected.grad, rtol=1e-6, atol=1e-7, msg=name)


def _mask_entropy_tolerance(expected: torch.Tensor, work_eps: float) -> torch.Tensor:
    """One float32 rounding of each entry plus work_eps of the row's largest entry."""
    unit = torch.finfo(torch.float32).eps * expected.abs()
    return unit / 2 + work_eps * expected.abs().amax(-1, keepdim=True)


@pytest.mark.parametrize("backend", ("auto", "native"))
@pytest.mark.parametrize("precision", (None, "float64"))
def test_boolean_mask_entropy_gradients_round_once(backend: str, precision: str | None):
    """float32 gradients of boolean-mask cross-entropy: the float64 gradients rounded once
    (precision="float64"), within one rounding plus 2 float32 eps of the row's largest entry
    with float32 exponentials (precision=None, as TORCH.md). The bug of random case 1139,
    positives balancing the negatives rounded into float32, was ten float32 eps of an entry
    off but less than one of the row's largest: the float64 check catches it."""
    case = ci.random_case(random.Random(1139))
    z = ci.inputs(case)["z"].reshape(-1, case.c)
    mask = ci.inputs(case)["target"].reshape(-1, case.c)
    x = z.clone().requires_grad_()
    sdca_loss(x, mask, backend=backend, precision=precision, reduction="sum").backward()
    _, expected = ci._entropy_rows(z.double(), mask)
    exact = _mask_entropy_tolerance(expected, 64 * torch.finfo(torch.float64).eps)
    rounding = torch.where(mask, 0.0, expected.float().double() - expected).sum(-1)
    stale = torch.where(mask, expected - rounding[:, None] / mask.sum(-1, keepdim=True), expected)
    assert ((stale.float().double() - expected).abs() > exact).any()
    tol = exact
    if precision is None:
        tol = _mask_entropy_tolerance(expected, 2 * torch.finfo(torch.float32).eps)
    assert ((x.grad.double() - expected).abs() <= tol).all()
