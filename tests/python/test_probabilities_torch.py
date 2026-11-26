"""libsdca.torch.cross_entropy of probability targets: the CPU kernel against the mpmath
reference and torch.nn.functional.cross_entropy (class weights, label smoothing, shapes and
reductions), its backends, and its derivatives."""

import math
from collections.abc import Callable
from typing import Any

import libsdca.torch as L
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from probability_references import EPS, Family, families, gradient_ulps, loss_ulps, reference

DTYPES = ["float64", "float32", "bfloat16", "float16"]


def _tensors(
    z: np.ndarray, t: np.ndarray, options: dict[str, Any], dtype: str
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    kind = getattr(torch, dtype)
    x, p = torch.tensor(z).to(kind), torch.tensor(t).to(kind)
    w = options.get("weight")
    kw: dict[str, Any] = {"label_smoothing": options.get("label_smoothing", 0.0)}
    if w is not None:
        kw["weight"] = torch.tensor(w).to(kind)
    return x, p, kw


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("family", list(families()), ids=lambda f: f[0])
def test_kernel_accuracy_against_mpmath(family: Family, dtype: str):
    """The kernel's losses within 8 ulps of their dtype (float64, else float32), gradients
    within 8 ulps of the largest term of their row and, but for float16 (whose range drops
    the tails of confident rows) and targets equal to the softmax, of its largest entry."""
    name, z, t, options = family
    x, p, kw = _tensors(z, t, options, dtype)
    assert L.explain_backend(L.cross_entropy, x, p, **kw)[0] == "kernel"
    w = kw.get("weight")
    want, want_gradient, terms = reference(
        x.double().numpy(),
        p.double().numpy(),
        kw["label_smoothing"],
        None if w is None else w.double().numpy(),
    )
    x.requires_grad_()
    values = L.cross_entropy(x, p, reduction="none", **kw)
    (gradient,) = torch.autograd.grad(values.sum(), x)
    g = gradient.double().numpy()
    value_dtype = "float64" if dtype == "float64" else "float32"
    assert loss_ulps(values.detach().double().numpy(), want, value_dtype) <= 8
    assert gradient_ulps(g, want_gradient, terms, dtype) <= 8
    if name != "softmax" and dtype != "float16":
        largest = np.abs(want_gradient).max(1, keepdims=True)
        assert gradient_ulps(g, want_gradient, largest, dtype) <= 8


@pytest.mark.parametrize("family", list(families()), ids=lambda f: f[0])
def test_float64_matches_pytorch(family: Family):
    """float64 losses and gradients within 8 ulps of PyTorch's (of the largest term for the
    gradient), on rows where PyTorch has a value (no class masked)."""
    name, z, t, options = family
    if name == "masked":
        pytest.skip("PyTorch gives NaN at a masked class of target zero")
    x, p, kw = _tensors(z, t, options, "float64")
    ours, theirs = x.clone().requires_grad_(), x.clone().requires_grad_()
    a = L.cross_entropy(ours, p, reduction="none", **kw)
    b = F.cross_entropy(theirs, p, reduction="none", **kw)
    (ga,) = torch.autograd.grad(a.sum(), ours)
    (gb,) = torch.autograd.grad(b.sum(), theirs)
    scale = torch.maximum(ga.abs(), gb.abs()).amax(1, keepdim=True) + p.abs().amax(1, keepdim=True)
    assert ((a - b).abs() <= 8 * EPS["float64"] * b.abs().clamp_min(1e-300)).all() or name in (
        "confident",
        "softmax",
    )
    assert ((ga - gb).abs() <= 8 * EPS["float64"] * scale).all()


@pytest.mark.parametrize("backend", ["auto", "composed"])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_options_match_pytorch(backend: str, reduction: str):
    """Class weights, label smoothing, (N, C, H, W) and (C,) logits and targets of another
    dtype, as torch.nn.functional.cross_entropy computes them in float64."""
    g = torch.Generator().manual_seed(3)
    cases = [
        ((6, 5), {}),
        ((6, 5), {"label_smoothing": 0.2}),
        ((6, 5), {"weight": torch.rand(5, generator=g, dtype=torch.float64)}),
        ((2, 5, 3, 4), {"weight": torch.rand(5, generator=g, dtype=torch.float64),
                        "label_smoothing": 0.1}),
        ((5,), {"label_smoothing": 0.3}),
    ]  # fmt: skip
    for shape, kw in cases:
        z = torch.randn(shape, generator=g, dtype=torch.float64)
        dim = 0 if z.ndim == 1 else 1
        t = torch.softmax(torch.randn(shape, generator=g, dtype=torch.float64), dim)
        for target in (t, t.float()):
            ours, theirs = z.clone().requires_grad_(), z.clone().requires_grad_()
            a = L.cross_entropy(ours, target, reduction=reduction, backend=backend, **kw)
            b = F.cross_entropy(theirs, target.double(), reduction=reduction, **kw)
            assert a.shape == b.shape and a.dtype == torch.float64
            (ga,) = torch.autograd.grad(a.sum(), ours)
            (gb,) = torch.autograd.grad(b.sum(), theirs)
            torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-15)
            torch.testing.assert_close(ga, gb, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("gap", [2.0**-27, 2.0**-100])
def test_bfloat16_probability_gradient_near_half(gap: float):
    # Two active classes on a wider masked row. A float exp rounds their ratio
    # to one, but both signed gradients are normal and representable in BF16.
    scores = torch.full((1, 1000), -math.inf, dtype=torch.bfloat16)
    scores[0, :2] = torch.tensor([gap, 0])
    scores.requires_grad_()
    target = torch.zeros_like(scores)
    target[0, :2] = 0.5
    delta = 0.5 * math.tanh(gap * 0.5)
    for scale in (-2.0, -0.25, 0.0, 0.5, 1.0, 3.0):
        loss = L.cross_entropy(scores.clone(), target, reduction="none", backend="kernel")
        (grad,) = torch.autograd.grad(loss, scores, torch.full_like(loss, scale))
        expected = torch.zeros_like(scores)
        expected[0, :2] = torch.tensor([scale * delta, -scale * delta])
        torch.testing.assert_close(grad, expected, atol=0, rtol=0)


def test_backends():
    """CPU probability targets run on the kernel, also when the target requires gradients."""
    z, t = torch.randn(4, 6), torch.softmax(torch.randn(4, 6), 1)
    assert L.explain_backend(L.cross_entropy, z, t, backend="composed")[0] == "composed"
    kernel = L.cross_entropy(z, t, backend="kernel")
    torch.testing.assert_close(kernel, F.cross_entropy(z, t))
    t.requires_grad_()
    for backend in ("auto", "kernel"):
        assert L.explain_backend(L.cross_entropy, z, t, backend=backend)[0] == "kernel"
        actual = torch.autograd.grad(L.cross_entropy(z, t, backend=backend), t)[0]
        expected = torch.autograd.grad(F.cross_entropy(z, t), t)[0]
        torch.testing.assert_close(actual, expected)


def test_target_gradients_take_tensor_operations():
    z = torch.randn(4, 6, dtype=torch.float64)
    t = torch.softmax(torch.randn(4, 6, dtype=torch.float64), 1).requires_grad_()
    (ours,) = torch.autograd.grad(L.cross_entropy(z, t, label_smoothing=0.1, backend="composed"), t)
    (theirs,) = torch.autograd.grad(F.cross_entropy(z, t, label_smoothing=0.1), t)
    torch.testing.assert_close(ours, theirs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("backend", ["auto", "kernel"])
def test_cuda_probability_kernel_matches_pytorch(backend: L.Backend):
    z = torch.randn(4, 6, device="cuda", requires_grad=True)
    t = torch.softmax(torch.randn_like(z), 1)
    assert L.explain_backend(L.cross_entropy, z, t, backend=backend)[0] == "kernel"
    value = L.cross_entropy(z, t, backend=backend)
    expected = F.cross_entropy(z, t)
    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(
        torch.autograd.grad(value, z)[0], torch.autograd.grad(expected, z)[0]
    )


def test_double_backward():
    """Composed differentiates the gradient as PyTorch does; the kernels are first order."""
    z = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    t = torch.softmax(torch.randn(3, 5, dtype=torch.float64), 1)
    w = torch.rand(5, dtype=torch.float64)

    def hessian(f: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        (g,) = torch.autograd.grad(f(z), z, create_graph=True)
        rows = [
            torch.autograd.grad(g.flatten()[i], z, retain_graph=True)[0] for i in range(z.numel())
        ]
        return torch.stack(rows)

    ours = hessian(
        lambda x: L.cross_entropy(x, t, weight=w, label_smoothing=0.1, backend="composed")
    )
    theirs = hessian(lambda x: F.cross_entropy(x, t, weight=w, label_smoothing=0.1))
    torch.testing.assert_close(ours, theirs)
    with pytest.raises(RuntimeError, match="first derivatives only"):
        hessian(lambda x: L.cross_entropy(x, t, weight=w, label_smoothing=0.1))


def test_transforms_and_compile():
    z = torch.randn(4, 6)
    t = torch.softmax(torch.randn(4, 6), 1)
    want = torch.func.grad(lambda x: F.cross_entropy(x, t))(z)
    torch.testing.assert_close(torch.func.grad(lambda x: L.cross_entropy(x, t))(z), want)
    batched = torch.func.vmap(lambda x, p: L.cross_entropy(x, p))(
        z.expand(2, 4, 6), t.expand(2, 4, 6)
    )
    torch.testing.assert_close(batched, F.cross_entropy(z, t).expand(2))
    compiled = torch.compile(lambda x: L.cross_entropy(x, t), backend="eager")
    torch.testing.assert_close(compiled(z), F.cross_entropy(z, t))


def test_masked_classes():
    """A masked class of target zero adds nothing (+0.0 gradient for any upstream sign); of a
    positive target, the loss is +inf and its gradient -t."""
    z = torch.tensor([[0.5, -torch.inf, 1.0, 2.0]], dtype=torch.float64)
    t = torch.tensor([[0.25, 0.0, 0.25, 0.5]], dtype=torch.float64)
    for backend in ("auto", "composed"):
        for upstream in (1.0, -1.0, 0.0):
            x = z.clone().requires_grad_()
            value = L.cross_entropy(x, t, reduction="none", backend=backend)
            (g,) = torch.autograd.grad(value, x, torch.tensor([upstream], dtype=torch.float64))
            assert torch.isfinite(value).all() and g[0, 1] == 0 and not torch.signbit(g[0, 1])
        x = z.clone().requires_grad_()
        positive = torch.tensor([[0.25, 0.25, 0.0, 0.5]], dtype=torch.float64)
        value = L.cross_entropy(x, positive, reduction="none", backend=backend)
        (g,) = torch.autograd.grad(value.sum(), x)
        assert value.item() == torch.inf and g[0, 1] == -0.25


@pytest.mark.parametrize("backend", ["auto", "composed"])
def test_invalid_rows_are_nan(backend: str):
    """A negative target, a NaN logit or a row of masked classes makes its row NaN; the other
    rows keep their values (docs/API.md, Invalid rows)."""
    z, t = torch.randn(2, 4), torch.softmax(torch.randn(2, 4), 1)
    want = F.cross_entropy(z, t, reduction="none")
    bad = t.clone()
    bad[0, 1] = -0.25
    for logits, target, row in (
        (z, bad, 0),
        (z.clone().index_fill_(0, torch.tensor([1]), -torch.inf), t, 1),
    ):
        value = L.cross_entropy(logits, target, reduction="none", backend=backend)
        assert torch.isnan(value[row]) and torch.isclose(value[1 - row], want[1 - row])
    nan = z.clone()
    nan[1, 0] = torch.nan
    value = L.cross_entropy(nan, t, reduction="none", backend=backend)
    assert torch.isnan(value[1]) and torch.isclose(value[0], want[0])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_weighted_confident_row_limit(device: str):
    """docs/API.md, Known limits, "Class weights on confident probability-target rows": a
    class weight 2**100 on the target of [140, 0, ..., 0] brings the competitors' terms, which
    float32 work forms below its range, back into range. They are lost, by at most the weight
    times the competitors' terms below exp(-132) of the maximum; label targets and
    precision="float64" keep them."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    c, x, w = 1000, 140.0, 2.0**100
    s = (c - 1) * math.exp(-x)
    exact = {
        "value": w * math.log1p(s),
        "win": -w * s / (1 + s),
        "lose": w * math.exp(-x) / (1 + s),
    }

    def run(target: torch.Tensor, precision: str | None) -> dict[str, float]:
        z = torch.zeros(1, c, device=device)
        z[0, 0] = x
        weight = torch.ones(c, device=device)
        weight[0] = w
        z.requires_grad_()
        value = L.cross_entropy(z, target, weight=weight, reduction="none", backend="kernel",
                                precision=precision)  # fmt: skip
        (g,) = torch.autograd.grad(value.sum(), z)
        return {"value": value.item(), "win": g[0, 0].item(), "lose": g[0, 1].item()}

    probabilities = torch.zeros(1, c, device=device)
    probabilities[0, 0] = 1
    labels = torch.zeros(1, dtype=torch.long, device=device)
    for got in (run(labels, None), run(probabilities, "float64")):
        for key, ref in exact.items():
            assert got[key] == pytest.approx(float(np.float32(ref)), rel=1e-5), key
    got = run(probabilities, None)
    bound = w * (c - 1) * math.exp(-132.0)
    for key, ref in exact.items():
        assert abs(got[key] - ref) <= bound, key
