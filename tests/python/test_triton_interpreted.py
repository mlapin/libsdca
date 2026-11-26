"""Release-numerics regressions of the CUDA Triton kernels, run on the CPU under Triton's
interpreter (interpreter.run; skipped without Triton): wide float32 cross-entropy rows far
from zero, near rows of wide float32 proximal distillation, float64 distillation rows whose
maxima coincide and distillation rows of far tails. The same rows run on CUDA in
test_chunked_rows_cuda.py, test_proximal_distillation_stream.py and
test_distillation_review_cuda.py."""

from __future__ import annotations

import pytest
import torch  # noqa: F401  (the torch group: the children launch through launch_torch)
from interpreter import run

PRELUDE = """
import math
import mpmath as mp
import torch
from conformance.references import Block, Loss, reference
from libsdca.triton import common, launch_torch

def launch(z, y, kind, gamma=0.0, temperature=1.0):
    env = launch_torch.run(z, y, None, None, kind, 1, gamma, temperature, 0.0, -100, False, 0,
                           1.0, True, True)
    return env["V"].double(), env["G"].double()
"""


@pytest.mark.parametrize("offset", [2.0**23, 2.0**24])
def test_wide_cross_entropy_far_from_zero(offset: float) -> None:
    """4097 classes, x[0] = offset, the others offset - 1 (labels 0 and 1): a chunk scale
    rounded by up to 0.5 needs its first-order correction to reach the exact loss 7.318."""
    run(
        PRELUDE
        + f"""
offset, c = {offset!r}, 4097
z = torch.full((2, c), offset - 1)
z[:, 0] = offset
y = torch.tensor([0, 1])
v, g = launch(z, y, common.CROSS_ENTROPY)
x = z.double().requires_grad_()
want = torch.nn.functional.cross_entropy(x, y, reduction="none")
want.sum().backward()
eps = torch.finfo(torch.float32).eps
assert ((v - want.detach()).abs() <= 64 * eps * want.detach().abs()).all(), (v, want)
assert ((g - x.grad).abs() <= 64 * eps * x.grad.abs().amax(1, keepdim=True)).all()
"""
    )


@pytest.mark.parametrize("gamma", [0.01, 1.0, 100.0])
def test_wide_proximal_distillation_near_rows(gamma: float) -> None:
    """4097 classes, student [0, -1, ...], teacher [0, -1.015625, ...], T = 2: the others'
    gradient (1.6e-6 of p) within (64 + C) eps = 4.96e-4 (relative)."""
    run(
        PRELUDE
        + f"""
gamma, c = {gamma!r}, 4097
z = torch.full((1, c), -1.0)
t = torch.full((1, c), -1.015625)
z[0, 0] = t[0, 0] = 0
v, g = launch(z, t, common.PROXIMAL_DISTILLATION, gamma, 2.0)
mp.mp.dps = 60
loss = Loss("proximal_distillation", temperature=2.0, alpha=gamma)
ref = reference(loss, (Block(0.0, 1, teacher=0.0), Block(-1.0, c - 1, teacher=-1.015625)))
budget = (64 + c) * torch.finfo(torch.float32).eps
checks = [(v[0].item(), ref.value), (g[0, 0].item(), ref.gradient[0])]
checks += [(a, ref.gradient[1]) for a in torch.unique(g[0, 1:]).tolist()]
for actual, exact in checks:
    assert abs(mp.mpf(actual) - exact) <= budget * abs(exact), (actual, exact)
"""
    )


def test_float64_distillation_common_peaks():
    """Float64 register rows whose maxima coincide: the row sum within C u |g|_1, and at 4096 classes the others' entries within (64 + C) eps (4292)."""
    run(
        PRELUDE
        + """
rows = [
    (1.0, [0.0, -18.3684003], [0.0, -9.18420014]),
    (1.0, [0.0, -55.1052009], [0.0, -27.5526004]),
    (1.0, [0.0, -55.1052009, -55.1052009, -56.1052009], [0.0, -27.5526004, -27.5526004, -28.0526004]),
    (4.0, [0.0, -108.442876], [0.0, -54.2214382]),
    (2.0, [0.0] * 4096, [0.0] + [-1.0] * 4095),
]
mp.mp.dps = 50
u = 2.0**-53
for temperature, student, teacher in rows:
    z = torch.tensor([student], dtype=torch.float64)
    t = torch.tensor([teacher], dtype=torch.float64)
    _, g = launch(z, t, common.DISTILLATION, 0.0, temperature)
    g = g[0].tolist()
    c = len(g)
    assert abs(math.fsum(g)) <= c * u * math.fsum(abs(a) for a in g), (student[:4], g[:4])
    s = [mp.mpf(a) / temperature for a in student]
    w = [mp.mpf(a) / temperature for a in teacher]
    zs, zt = mp.fsum(mp.e**a for a in s), mp.fsum(mp.e**a for a in w)
    for j in sorted({0, 1, c - 1}):
        exact = (mp.e**s[j] / zs - mp.e**w[j] / zt) / temperature
        assert abs(g[j] - exact) <= (64 + c) * 2 * u * abs(exact), (j, g[j], exact)
"""
    )


# Two-group distillation rows (dtype, classes, T, student gap, teacher gap in units of T): the
# others at -gap T, class 0 at 0 (conformance far tails).
FAR_TAILS = """
FAR = [
    # float64 common peaks, the others beyond exp(-708): entry 0 (the tails' mass
    # difference) comes from the tails' subnormal exponentials.
    (torch.float64, 10, 1.0, 740, 750, 1),
    (torch.float64, 1000, 0.5, 750, 740, 1),
    # Streamed float32 work (one read, _online): a teacher term below float32's range
    # under a larger student term (q 1e-20..1e-31 for p = 0); equal tails beyond exp(-87),
    # whose first chunk carries 1/4 of the loss.
    (torch.float32, 8192, 1.0, 45, 100, 1),
    (torch.float32, 8192, 1.0, 50, 104, 1),
    (torch.float32, 8192, 1.0, 70, 120, 1),
    (torch.float32, 8192, 1.0, 104, 104, 1),
    (torch.float16, 8192, 0.5, 104, 104, 1),
    # Two tied maxima: both tails round away from 2 in float64 (equal sums), and the
    # equal-tails shortcut must keep the maxima's entries (-3.9e-19) under a positive tail.
    (torch.float32, 8192, 1.0, 50, 104, 2),
]
"""


def test_distillation_far_tails():
    """Values and gradient entries of the far-tail rows within (64 + C) eps of their exact
    values (eps of the work, the float32 values' and the storage's smallest normal times its
    eps below them), the CUDA bound of release numerics."""
    run(
        PRELUDE
        + FAR_TAILS
        + """
mp.mp.dps = 40
for dtype, c, temperature, gs, gt, lead in FAR:
    z = torch.zeros((1, c), dtype=torch.float64)
    t = torch.zeros_like(z)
    z[0, lead:], t[0, lead:] = -gs * temperature, -gt * temperature
    z, t = z.to(dtype), t.to(dtype)
    v, g = launch(z, t, common.DISTILLATION, 0.0, temperature)
    loss = Loss("distillation", temperature=temperature)
    rest = Block(float(z[0, lead]), c - lead, teacher=float(t[0, lead]))
    row = (Block(0.0, lead, teacher=0.0), rest)
    ref = reference(loss, row)
    work = torch.float64 if dtype == torch.float64 else torch.float32
    store = torch.finfo(dtype)
    budget = (64 + c) * torch.finfo(work).eps
    out = torch.finfo(work)
    checks = [(v[0].item(), ref.value, out.eps * out.tiny)]
    floor = store.eps * store.tiny
    checks += [(a, ref.gradient[0], floor) for a in torch.unique(g[0, :lead]).tolist()]
    checks += [(a, ref.gradient[1], floor) for a in torch.unique(g[0, lead:]).tolist()]
    for actual, exact, bottom in checks:
        assert abs(mp.mpf(actual) - exact) <= budget * abs(exact) + bottom, (
            dtype, c, gs, gt, lead, actual, exact
        )
    # The row sums to zero within the entries' bounds.
    size = lead * abs(ref.gradient[0]) + (c - lead) * abs(ref.gradient[1])
    total = math.fsum(g[0].tolist())
    assert abs(total) <= budget * size + c * floor, (dtype, c, gs, gt, lead, total)
"""
    )


def test_wide_multilabel_cancelling_positive():
    """test_cuda's wide multilabel rows (16401 classes) whose maximum is a negative class: a
    positive within 1e-7 of its target 1 / |Y| (test_multilabel_entropy_mixed_rows) or 1.6%
    to 6.4% from it (test_wide_multilabel_entropy_small_positive_residual). The float32 pass
    is final there (64 eps of the row's largest entry, docs/API.md): each entry within 64 eps
    of its terms |up| (p_j + t_j) (at most the largest entry), the sign of p - 1 / |Y| kept;
    a confident positive at the maximum (float64 work) within 4e-6 of itself."""
    run(
        PRELUDE
        + """
kind = common.KINDS["multilabel_cross_entropy"]
eps = torch.finfo(torch.float32).eps

def check(z, mask, up, strict=()):
    env = launch_torch.run(z, mask, None, up, kind, 1, 0.0, 1.0, 0.0, -100, False, 0, 1.0,
                           True, True)
    g = env["G"].double()
    p = z.double().softmax(1)
    expected = p - mask.double() / mask.sum(1, keepdim=True)
    for i in strict:
        # A lone positive at the maximum: p - 1 = -(the others' mass), without cancellation.
        expected[i, 1] = -(p[i, :1].sum() + p[i, 2:].sum())
    expected = expected * up[:, None].double()
    error = (g - expected).abs()
    terms = up[:, None].abs() * (p + mask.double() / mask.sum(1, keepdim=True))
    bound = 64 * eps * torch.minimum(terms, expected.abs().amax(1, keepdim=True))
    for i in range(len(z)):
        if i in strict:
            assert (error[i] <= 4e-6 * expected[i].abs()).all(), i
        else:
            assert (error[i] <= bound[i]).all(), (i, (error[i] / bound[i]).max())
    return g

classes, positives = 16401, 32
z = torch.zeros((3, classes))
mask = torch.zeros_like(z, dtype=torch.bool)
mask[:, 1 : positives + 1] = True
z[:, 0] = math.log(2 * classes / positives)
z[:, -1] = -math.inf
z[1, 1] = math.log((math.exp(z[0, 0].item()) + classes - 3) / (positives - 1))
z[2, 1] = 70
mask[2, 2 : positives + 1] = False
check(z, mask, torch.tensor([1.0, 0.25, -2.0]), strict=(2,))

count = 7
z = torch.zeros((4, classes))
z[:, ::3] = -4.75
z[:, 0] = math.log(3 * classes / count)
z[:, 1] = -math.inf
other = z[0].double().exp().sum()
delta = torch.tensor([0.064, -0.064, 0.016, -0.016]).double()
z[:, 1] = ((1 + delta) * other / (count - 1 - delta)).log()
mask = torch.zeros_like(z, dtype=torch.bool)
mask[:, 1 : count + 1] = True
g = check(z, mask, torch.ones(4))
assert torch.equal(torch.sign(g[:, 1]), torch.sign(delta))
"""
    )
