"""Row-strided CUDA scores (a slice x[:, a:b] of wider rows, at any element offset) reach the
cross-entropy kernels without a copy: values and gradients as for the contiguous scores, the
gradient written over strided logits (loss_) only inside the slice, and no allocation of a
contiguous copy."""

import pytest
import torch
import torch.nn.functional as F
from libsdca.torch import cross_entropy, multilabel_cross_entropy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
KINDS = {"labels": 0, "probabilities": 0, "mask": 6}


def _strided(z: torch.Tensor, offset: int, pad: int) -> torch.Tensor:
    """z as base[:, offset : offset + C] of a wider (N, C + pad) tensor."""
    base = torch.full((z.shape[0], z.shape[1] + pad), float("nan"), device=z.device, dtype=z.dtype)
    view = base[:, offset : offset + z.shape[1]]
    view.copy_(z)
    assert not view.is_contiguous() and view.stride() == (z.shape[1] + pad, 1)
    return view


def _target(kind: str, rows: int, classes: int, dtype: torch.dtype) -> torch.Tensor:
    if kind == "labels":
        return torch.randint(0, classes, (rows,), device="cuda")
    if kind == "probabilities":
        return torch.softmax(3 * torch.randn(rows, classes, device="cuda"), 1).to(dtype)
    m = torch.rand(rows, classes, device="cuda") < 0.01
    m[:, 0], m[:, 1] = True, False
    return m


def _loss(kind: str):
    return multilabel_cross_entropy if kind == "mask" else cross_entropy


@pytest.mark.parametrize("classes", [1000, 8193, 50257])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("kind", ["labels", "probabilities", "mask"])
@pytest.mark.parametrize("offset, pad", [(1, 1), (0, 64), (3, 5)])
def test_strided_scores_match_contiguous(
    classes: int, dtype: torch.dtype, kind: str, offset: int, pad: int
) -> None:
    torch.manual_seed(classes + offset)
    rows = 37
    z = (3 * torch.randn(rows, classes, device="cuda")).to(dtype)
    z[5, 9] = float("nan")  # a row the chunks mark: the streaming kernels, strided too
    t = _target(kind, rows, classes, dtype)
    loss = _loss(kind)
    x = z.clone().requires_grad_()
    want = loss(x, t, reduction="none")
    (want_grad,) = torch.autograd.grad(want.sum(), x)
    s = _strided(z, offset, pad).requires_grad_()
    got = loss(s, t, reduction="none")
    (grad,) = torch.autograd.grad(got.sum(), s)
    # Chunk boundaries follow each row's own alignment: sums may round differently.
    tol = 2e-6 if dtype == torch.float32 else 1e-5
    torch.testing.assert_close(got, want, rtol=tol, atol=tol, equal_nan=True)
    resolution = 2.0**-8 if dtype == torch.bfloat16 else 2.0**-20
    scale = want_grad.nan_to_num().abs().amax(1, keepdim=True)
    ok = (grad.double() - want_grad.double()).abs() <= resolution * scale
    assert (ok | (grad.isnan() & want_grad.isnan())).all()


@pytest.mark.parametrize("classes", [1000, 50257])
@pytest.mark.parametrize("kind", ["labels", "mask"])
def test_strided_gradient_over_the_scores(classes: int, kind: str) -> None:
    """loss_ (reuse_logits) writes the gradient into the strided rows, not between them."""
    torch.manual_seed(classes)
    rows, dtype = 19, torch.bfloat16
    z = (3 * torch.randn(rows, classes, device="cuda")).to(dtype)
    t = _target(kind, rows, classes, dtype)
    args = (None, None, KINDS[kind], 1, 0.0, 1.0, 0.0, -100, False, 0, 1.0, True, True)
    _, want = torch.ops.libsdca.loss(z, t, *args)
    s = _strided(z, 1, 3)
    base = s._base
    assert base is not None
    torch.ops.libsdca.loss_(s, t, *args)
    torch.testing.assert_close(s, want, rtol=0, atol=2.0**-8 * want.abs().max().item())
    outside = torch.ones_like(base, dtype=torch.bool)
    outside[:, 1 : 1 + classes] = False
    assert base[outside].isnan().all()


@pytest.mark.parametrize("kind", ["labels", "mask"])
def test_strided_forward_allocates_no_copy(kind: str) -> None:
    rows, classes, dtype = 512, 50257, torch.bfloat16
    z = (3 * torch.randn(rows, classes, device="cuda")).to(dtype)
    t = _target(kind, rows, classes, dtype)
    s = _strided(z, 1, 1)
    loss = _loss(kind)
    with torch.no_grad():
        want = loss(z, t)
        loss(s, t)  # compile and cache
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        got = loss(s, t)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
    assert peak < z.numel() * z.element_size() // 4
    torch.testing.assert_close(got, want, rtol=1e-5, atol=0)


def test_strided_labels_against_float64():
    torch.manual_seed(0)
    rows, classes = 64, 50257
    z = (3 * torch.randn(rows, classes, device="cuda")).bfloat16()
    y = torch.randint(0, classes, (rows,), device="cuda")
    s = _strided(z, 1, 1).requires_grad_()
    value = cross_entropy(s, y)
    (grad,) = torch.autograd.grad(value, s)
    x = z.double().requires_grad_()
    want = F.cross_entropy(x, y)
    (want_grad,) = torch.autograd.grad(want, x)
    torch.testing.assert_close(value.double(), want, rtol=2e-6, atol=0)
    assert ((grad.double() - want_grad).abs() <= 2.0**-8 * want_grad.abs().max()).all()
