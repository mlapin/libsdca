"""Wide CUDA rows that run as parallel chunks (cross-entropy with labels, probabilities or
masks, multilabel hinges): ordinary rows against float64 references, and the rows a chunk marks
(NaN, +inf, masked classes, ties, extreme scores) through the streaming kernels."""

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from libsdca.torch import cross_entropy, multilabel_cross_entropy, multilabel_hinge_loss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
WIDTHS = [4097, 32000, 50257]
DENSE_WIDTHS = [600, 1000, 4096]


def _reference(
    z: torch.Tensor, target: torch.Tensor, **options: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    x = z.detach().double().requires_grad_()
    t = target.double() if target.is_floating_point() else target
    if "weight" in options:
        options["weight"] = options["weight"].double()
    value = F.cross_entropy(x, t, **options)
    value.sum().backward()  # one unit upstream per row under reduction="none"
    assert x.grad is not None
    return value.detach(), x.grad


def _scores(rows: int, classes: int, dtype: torch.dtype, special: bool = True) -> torch.Tensor:
    z = (3 * torch.randn(rows, classes, device="cuda")).to(dtype)
    if special:
        z[1, 7] = -float("inf")  # a masked class
        z[2, :] = 0.0  # ties everywhere
        z[3, 11] = 3e4  # an extreme score
    return z


@pytest.mark.parametrize("classes", WIDTHS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "target, options",
    [
        ("labels", {}),
        ("labels", {"label_smoothing": 0.1}),
        ("ignored", {"ignore_index": -100}),
        ("ignored", {"weight": True, "ignore_index": -100, "label_smoothing": 0.1}),
        ("probabilities", {}),
        ("probabilities", {"weight": True, "label_smoothing": 0.1}),
    ],
)
def test_chunked_cross_entropy_matches_float64(
    classes: int, dtype: torch.dtype, target: str, options: dict[str, Any]
) -> None:
    torch.manual_seed(classes)
    rows = 9
    z = _scores(rows, classes, dtype)
    y = torch.randint(0, classes, (rows,), device="cuda")
    y[1] = 3  # the masked class is a competitor
    if target == "ignored":
        y[::4] = -100
    t = torch.softmax(3 * torch.randn(rows, classes, device="cuda"), 1).to(dtype)
    if target == "probabilities":
        # torch gives NaN for a masked class without target mass (0 * -inf); libsdca 0.
        z[1, 7] = 0.0
    t = t if target == "probabilities" else y
    options = dict(options)
    if options.get("weight"):
        options["weight"] = (0.5 + torch.rand(classes, device="cuda")).to(dtype)
    x = z.clone().requires_grad_()
    value = cross_entropy(x, t, **options)
    value.backward()
    want, gradient = _reference(z, t, **options)
    torch.testing.assert_close(value.double(), want, rtol=2e-6, atol=0)
    scale = gradient.abs().max()
    resolution = 2.0**-8 if dtype == torch.bfloat16 else 2.0**-20
    assert x.grad is not None
    assert ((x.grad.double() - gradient).abs() <= resolution * scale).all()


@pytest.mark.parametrize("classes", DENSE_WIDTHS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_chunked_probabilities_narrow_rows(classes: int, dtype: torch.dtype) -> None:
    torch.manual_seed(classes)
    z = _scores(9, classes, dtype)
    z[1, 7] = 0.0
    t = torch.softmax(3 * torch.randn(9, classes, device="cuda"), 1).to(dtype)
    x = z.clone().requires_grad_()
    value = cross_entropy(x, t)
    value.backward()
    want, gradient = _reference(z, t)
    torch.testing.assert_close(value.double(), want, rtol=2e-6, atol=0)
    resolution = 2.0**-8 if dtype == torch.bfloat16 else 2.0**-20
    assert x.grad is not None
    assert ((x.grad.double() - gradient).abs() <= resolution * gradient.abs().max()).all()


@pytest.mark.parametrize("classes", [4097, 50257])
def test_chunked_cross_entropy_invalid_rows(classes: int) -> None:
    z = _scores(4, classes, torch.bfloat16, special=False)
    z[0, 5] = float("nan")
    z[1, 6] = float("inf")
    y = torch.tensor([1, 2, 3, 4], device="cuda")
    x = z.clone().requires_grad_()
    values = cross_entropy(x, y, reduction="none")
    values.sum().backward()
    assert values[:2].isnan().all() and values[2:].isfinite().all()
    assert x.grad is not None
    assert x.grad[:2].isnan().all() and x.grad[2:].isfinite().all()


def _multilabel_reference(z: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The nonsmooth multilabel hinge and its tie-sharing subgradient, float64."""
    z = z.double()
    inf = torch.tensor(float("inf"), device=z.device, dtype=z.dtype)
    low = torch.where(m, z, inf).min(1).values
    high = torch.where(m, -inf, z).max(1).values
    h = 1 + high - low
    slope = torch.where(h == 0, 0.5, (h > 0).double())
    at_low = m & (z == low[:, None])
    at_high = ~m & (z == high[:, None])
    g = at_high * (slope / at_high.sum(1))[:, None] - at_low * (slope / at_low.sum(1))[:, None]
    return h.clamp_min(0), g


@pytest.mark.parametrize("classes", WIDTHS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_chunked_multilabel_hinge_ties(classes: int, dtype: torch.dtype) -> None:
    torch.manual_seed(classes)
    rows = 8
    z = (3 * torch.randn(rows, classes, device="cuda")).round().to(dtype)  # many ties
    m = torch.rand(rows, classes, device="cuda") < 0.01
    m[:, 0], m[:, 1] = True, False
    x = z.clone().requires_grad_()
    values = multilabel_hinge_loss(x, m, reduction="none")
    values.sum().backward()
    want, gradient = _multilabel_reference(z, m)
    torch.testing.assert_close(values.double(), want, rtol=0, atol=0)
    assert x.grad is not None
    torch.testing.assert_close(x.grad.double(), gradient.to(dtype).double(), rtol=0, atol=0)


@pytest.mark.parametrize("classes", [4097, 50257])
def test_chunked_smooth_multilabel_hinge_matches_float64_work(classes: int) -> None:
    """The candidates' projection against the same loss in float64 work (all streamed)."""
    torch.manual_seed(classes)
    z = (3 * torch.randn(6, classes, device="cuda")).bfloat16()
    m = torch.rand(6, classes, device="cuda") < 0.01
    m[:, 0], m[:, 1] = True, False
    z[1, ~m[1]] = z[1, ~m[1]].clamp_max(-20)  # a separated row
    x = z.clone().requires_grad_()
    y = z.clone().double().requires_grad_()
    values = multilabel_hinge_loss(x, m, gamma=0.5, reduction="none")
    values.sum().backward()
    want = multilabel_hinge_loss(y, m, gamma=0.5, reduction="none")
    want.sum().backward()
    torch.testing.assert_close(values.double(), want.detach(), rtol=1e-6, atol=1e-6)
    assert x.grad is not None and y.grad is not None
    torch.testing.assert_close(x.grad.double(), y.grad.bfloat16().double(), rtol=0, atol=2.0**-8)


def _mask_reference(z: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """multilabel_cross_entropy and its gradient in float64: log sum exp(z) - mean z_m."""
    x = z.detach().double().requires_grad_()
    value = torch.logsumexp(x, 1) - torch.where(m, x, 0.0).sum(1) / m.sum(1)
    value.sum().backward()
    assert x.grad is not None
    return value.detach(), x.grad


@pytest.mark.parametrize("classes", WIDTHS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("save", [True, False])
def test_chunked_multilabel_cross_entropy_matches_float64(
    classes: int, dtype: torch.dtype, save: bool
) -> None:
    """Wide mask rows (aligned mask-table chunks where saved): ordinary rows, a masked
    negative, ties, extreme scores and a wide spread against float64."""
    torch.manual_seed(classes)
    rows = 10
    z = _scores(rows, classes, dtype)
    if dtype == torch.float16:
        z[3, 11] = 3e3
    z[4, 100:200] = -100.0  # within a chunk, e**-100 is below float32's normal range
    z[5, 0] = 90.0
    m = torch.rand(rows, classes, device="cuda") < 0.01
    m[:, 0], m[:, 1] = True, False
    m[1, 7] = False  # the masked class is a negative
    x = z.clone().requires_grad_(save)
    if save:
        values = multilabel_cross_entropy(x, m, reduction="none")
        values.sum().backward()
        gradient = x.grad
    else:
        # A joint value and gradient (no saved statistics): JAX's pullback, here the kernel.
        from libsdca import _frontend as f

        kind = f.KINDS["multilabel_cross_entropy"]
        values, gradient = torch.ops.libsdca.loss(
            z, m, None, None, kind, 1, 0.0, 1.0, 0.0, -100, False, 0, 1.0, True, True
        )
    want, grad = _mask_reference(z, m)
    torch.testing.assert_close(values.double(), want, rtol=2e-6, atol=0)
    resolution = torch.finfo(dtype).eps
    scale = grad.abs().amax(1, keepdim=True)
    assert gradient is not None
    assert ((gradient.double() - grad).abs() <= resolution * scale).all()


@pytest.mark.parametrize("classes", [4097, 32768, 32769])
@pytest.mark.parametrize("offset", [2.0**23, 2.0**24, -3.0e6])
@pytest.mark.parametrize("target", ["labels", "probabilities"])
def test_chunked_rows_far_from_zero(classes: int, offset: float, target: str) -> None:
    """Float32 rows whose scores sit far from zero (release numerics: x[0] = 1/eps, the
    others 1/eps - 1). A chunk's exponentials are of z - max, exact near the maximum at any
    scale (exp2 of z log2(e) - c, with c = max log2(e) rounded by up to 0.5, would be off by a
    4% sum at 4097 classes)."""
    torch.manual_seed(classes)
    z = torch.full((3, classes), offset - 1, device="cuda")
    z[:, 0] = offset
    z[2] = offset + torch.round(8 * torch.randn(classes, device="cuda")) / 8
    y = torch.tensor([0, 1, 2], device="cuda")
    if target == "probabilities":
        y = torch.softmax(torch.randn(3, classes, device="cuda"), 1)
    x = z.clone().requires_grad_()
    values = cross_entropy(x, y, reduction="none")
    values.sum().backward()
    want, grad = _reference(z, y, reduction="none")
    eps = torch.finfo(torch.float32).eps
    assert ((values.double() - want).abs() <= 64 * eps * want.abs()).all(), (values, want)
    assert x.grad is not None
    assert ((x.grad.double() - grad).abs() <= 64 * eps * grad.abs().amax(1, keepdim=True)).all()
