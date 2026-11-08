"""Random common calls of the PyTorch losses against float64 references (test helper).

A Case is one call: a loss, its options, the shapes and dtype of the logits, a layout of
them (transposed, sliced, expanded, extra leading dims, class dimension not last), a target,
a reduction and an upstream factor. check() runs it on every CPU path (SDCA_CPU_EXTENSION
on and off, each backend, default and one thread) and compares the value and the gradient
of the logits' leaf with references of the represented inputs:

- cross-entropies and distillation: closed forms (class weights and ignore_index as in
  F.cross_entropy);
- proximal losses: the Moreau envelope from its Lambert W optimality condition, solved
  by bisection in float64;
- nonsmooth hinges: their definitions over the sorted margins;
- top-k cross-entropy and smoothed hinges: the mpmath references of test_topk_kernels and
  test_multilabel_kernels for up to SMALL classes; wider rows are compared with the
  float64 tensor backend instead.

Tolerances are in epsilons of the working arithmetic, relative to the scale of a row, plus
one rounding to the output dtype (TORCH.md: 16-bit gradients one unit of their dtype plus
2000 float32 eps). Nonsmooth hinge rows at a kink or with tied selections have several
subgradients: their gradients are not compared. test_torch_common_inputs.py runs a fixed
sample.
"""

from __future__ import annotations

import contextlib
import math
import os
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import libsdca.torch as L
import torch
import torch.nn.functional as F

SMALL = 48  # widest rows of the mpmath references
SDCA_LOSSES = (
    "cross_entropy",
    "multilabel_cross_entropy",
    "topk_cross_entropy",
    "multiclass_hinge",
    "topk_hinge_alpha",
    "topk_hinge_beta",
    "multilabel_hinge",
    "proximal_cross_entropy",
)
LOSSES = ("ce", *SDCA_LOSSES, "distillation", "proximal_distillation")
HINGES = ("multiclass_hinge", "topk_hinge_alpha", "topk_hinge_beta", "multilabel_hinge")
DTYPES = (torch.float32, torch.float64, torch.bfloat16, torch.float16)
# Tolerances in epsilons of the working arithmetic, relative to a row's scale.
VALUE_EPS = 64
GRAD_EPS = 64
PROX_EPS = 4096  # the solvers stop at a residual relative to the gradient's scale
HALF_GRAD_EPS32 = 2000  # TORCH.md: one unit of the 16-bit dtype plus 2000 float32 eps


@dataclass
class Case:
    loss: str
    n: int
    c: int
    dtype: torch.dtype
    reduction: str
    layout: str
    seed: int
    scale: float = 1.0
    offset: float = 0.0
    k: int = 1
    gamma: float = 0.0
    alpha: float = 1.0
    temperature: float = 1.0
    mask: bool = False  # boolean target
    positives: int = 1
    weight: bool = False
    ignore: int | None = None  # ignore_index for "ce" (some targets set to it)
    upstream: float = 1.0
    precision: str | None = None
    form: str = "function"  # function, named, module
    integer: bool = False  # integer-valued logits (ties)
    # Reduction none: the upstream's layout, "dense", "expanded" (one value, stride 0, as
    # loss.sum() gives) or "strided" (a view of every other entry).
    upstream_layout: str = "dense"
    extra: tuple[int, ...] = field(default_factory=tuple)  # dims after n, before classes

    def describe(self) -> str:
        defaults = Case("", 0, 0, torch.float32, "", "", 0)
        changed = {
            k: v
            for k, v in self.__dict__.items()
            if k in ("loss", "n", "c", "dtype", "reduction", "layout", "seed")
            or v != getattr(defaults, k)
        }
        return ", ".join(f"{k}={v}" for k, v in changed.items())


def random_case(rng: random.Random, large: bool = False) -> Case:
    """A random common call: ordinary shapes, scales, options and layouts."""
    loss = rng.choice(LOSSES)
    small = loss == "topk_cross_entropy" or (loss in HINGES and loss != "multiclass_hinge")
    shapes = [(1, 2), (1, 10), (2, 2), (3, 5), (8, 10), (16, 3), (5, 37), (64, 10), (33, 2)]
    if large and not small:
        shapes += [(1, 32000), (4, 5000), (2000, 3), (100000, 3), (256, 1000), (1, 128256)]
    elif large:
        shapes += [(256, 20), (1000, 5), (64, 40)]
    n, c = rng.choice(shapes)
    case = Case(
        loss=loss,
        n=n,
        c=c,
        dtype=rng.choices(DTYPES, weights=(4, 2, 2, 1))[0],
        reduction=rng.choice(("mean", "sum", "none")),
        layout=rng.choice(("contiguous",) * 4 + ("transposed", "sliced", "expanded", "dim")),
        seed=rng.randrange(2**31),
        scale=rng.choice((0.1, 1.0, 1.0, 3.0, 10.0)),
        offset=rng.choice((0.0, 0.0, 0.0, 5.0, -30.0)),
        upstream=rng.choice((1.0, 1.0, 0.5, -2.0, 3e-3, 1024.0)),
        precision=rng.choice((None, None, None, "float32", "float64")),
        form=rng.choice(("function", "function", "named", "module")),
        integer=rng.random() < 0.1,
    )
    if case.dtype == torch.float64 and case.precision == "float32":
        case.precision = None
    if rng.random() < 0.15 and n * c <= 100000:
        case.extra = (rng.choice((2, 3)),)
    if case.layout == "expanded" and case.n == 1:
        case.layout = "contiguous"
    _loss_options(case, rng)
    if case.reduction == "none":
        case.upstream_layout = rng.choice(("dense", "dense", "expanded", "strided"))
    return case


def _loss_options(case: Case, rng: random.Random) -> None:
    """The options and target kind of the case's loss."""
    loss, c = case.loss, case.c
    if loss in ("topk_cross_entropy", "topk_hinge_alpha", "topk_hinge_beta") and c > 2:
        case.k = min(rng.choice([1, 2, 2, 3, 5, c - 1]), c - 1)
    if loss in HINGES:
        case.gamma = rng.choice((0.0, 0.0, 0.1, 0.5, 1.0, 2.0))
    if loss in ("proximal_cross_entropy", "proximal_distillation"):
        case.alpha = rng.choice((1e-3, 0.1, 1.0, 1.0, 10.0, 100.0))
    if loss in ("distillation", "proximal_distillation"):
        case.temperature = rng.choice((0.5, 1.0, 1.0, 2.0, 4.0, 10.0))
    if loss in ("cross_entropy", "multilabel_cross_entropy", "multilabel_hinge"):
        case.mask = rng.random() < 0.6
    elif loss in SDCA_LOSSES:
        case.mask = rng.random() < 0.15
    if case.mask:
        if loss in ("cross_entropy", "multilabel_cross_entropy"):
            case.positives = rng.randint(1, c)
        elif loss in ("multilabel_hinge", "proximal_cross_entropy"):
            case.positives = rng.randint(1, c - 1)
    if loss == "ce":
        case.weight = rng.random() < 0.3
        if rng.random() < 0.3:
            case.ignore = rng.choice((-100, -100, 0, 2**40))
        if case.layout == "dim":
            case.layout = "contiguous"


# ----------------------------------------------------------------------------- inputs


def inputs(case: Case) -> dict[str, Any]:
    """Class-last logits, the target (class-last mask), the teacher, the class weights and
    the class dim of the call."""
    g = torch.Generator().manual_seed(case.seed)
    shape = (case.n, *case.extra, case.c)
    if case.integer:
        z = torch.randint(-3, 4, shape, generator=g).double()
    else:
        z = torch.randn(shape, generator=g, dtype=torch.float64) * case.scale + case.offset
    z = z.to(case.dtype)
    rows = shape[:-1]
    target: torch.Tensor
    if case.mask:
        order = torch.rand(*rows, case.c, generator=g).argsort(-1)
        target = order < case.positives
    else:
        target = torch.randint(0, case.c, rows, generator=g)
        if case.ignore is not None:
            drop = torch.rand(rows, generator=g) < 0.3
            target = torch.where(drop, torch.tensor(case.ignore), target)
    teacher = None
    if case.loss in ("distillation", "proximal_distillation"):
        teacher = torch.randn(shape, generator=g, dtype=torch.float64) * case.scale
        teacher = (teacher + case.offset).to(case.dtype)
        if torch.rand((), generator=g) < 0.2:
            teacher = teacher.requires_grad_()
    weight = None
    if case.weight:
        weight = torch.rand(case.c, generator=g, dtype=torch.float64) * 2
        weight[torch.rand(case.c, generator=g) < 0.2] = 0
        weight = weight.to(case.dtype)
    # The class dim of the call: last, except "ce" with extra dims ((N, C, d)) and "dim".
    class_dim = -1
    if case.loss == "ce" and case.extra:
        class_dim = 1
    elif case.layout == "dim":
        class_dim = 0
    if case.layout == "expanded" and case.n > 1:
        z = z[:1].expand(z.shape).clone()
        if teacher is not None:
            teacher = teacher[:1].expand(z.shape).clone().requires_grad_(teacher.requires_grad)
    return dict(z=z, target=target, teacher=teacher, weight=weight, class_dim=class_dim)


def view(case: Case, z: torch.Tensor, class_dim: int) -> tuple[torch.Tensor, Callable[..., Any]]:
    """A leaf and the function of it that gives the call's logits in the case's layout:
    the values of class-last z with the class dimension moved to class_dim."""
    want = z.movedim(-1, class_dim)
    if case.layout == "transposed" and want.ndim >= 2:
        leaf = want.transpose(0, -1).contiguous()
        return leaf, lambda x: x.transpose(0, -1)
    if case.layout == "sliced":
        leaf = torch.zeros((*want.shape[:-1], 2 * want.shape[-1]), dtype=want.dtype)
        leaf[..., ::2] = want
        return leaf, lambda x: x[..., ::2]
    if case.layout == "expanded" and want.shape[0] > 1 and class_dim != 0:
        return want[:1].clone(), lambda x: x.expand(want.shape)
    return want.clone(), lambda x: x


# ----------------------------------------------------------------------------- references


def _target_rows(target: torch.Tensor, c: int) -> torch.Tensor:
    """Target distribution rows: one-hot of indices or uniform over a mask's positives."""
    if target.dtype == torch.bool:
        t = target.double()
        return t / t.sum(-1, keepdim=True)
    return F.one_hot(target, c).double()


def _balanced(d: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """d (rows summing to zero) with the entry of each row's largest weight replaced by
    minus the sum of the others: p - t there would cancel when p is near t."""
    j = weights.argmax(-1, keepdim=True)
    return d.scatter(1, j, -d.scatter(1, j, 0.0).sum(-1, keepdim=True))


def _entropy_rows(z: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy with the target distribution and its gradient."""
    t = _target_rows(target, z.shape[-1])
    value = torch.logsumexp(z, -1) - (t * z).sum(-1)
    return value, _balanced(torch.softmax(z, -1) - t, t)


def _log_omega(log_x: torch.Tensor) -> torch.Tensor:
    """log W(exp(log_x)) of the Lambert W function: v + exp(v) = log_x by Newton steps from
    the right of the root (monotone convergence)."""
    v = torch.where(log_x > 1, log_x.clamp_min(1).log(), log_x)
    for _ in range(100):
        v = v - (v + v.exp() - log_x) / (1 + v.exp())
    return v


def _envelope(x: torch.Tensor, t: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """min_u lse(u) - t.u + |u - x|^2 / (2 alpha) per row: (value, softmax(u*) - t).

    At the minimum p = softmax(u*) has alpha p_j = W(alpha exp(x_j + alpha t_j - s)) with
    s = lse(u*) and sum p = 1, which decreases in s: bisection on s in float64."""
    shifted = math.log(alpha) + x + alpha * t

    def mass(s: torch.Tensor) -> torch.Tensor:
        return _log_omega(shifted - s[:, None]).exp().sum(-1) / alpha

    lo = torch.logsumexp(x, -1) - alpha - 1
    hi = torch.logsumexp(x, -1) + alpha + 1
    for _ in range(64):  # widen until mass(lo) >= 1 >= mass(hi)
        low, high = mass(lo) < 1, mass(hi) > 1
        if not bool((low | high).any()):
            break
        lo = torch.where(low, lo - (hi - lo), lo)
        hi = torch.where(high, hi + (hi - lo), hi)
    for _ in range(200):
        mid = (lo + hi) / 2
        above = mass(mid) > 1
        lo, hi = torch.where(above, mid, lo), torch.where(above, hi, mid)
    p = _log_omega(shifted - ((lo + hi) / 2)[:, None]).exp() / alpha
    p = p / p.sum(-1, keepdim=True)
    u = x - alpha * (p - t)
    value = torch.logsumexp(u, -1) - (t * u).sum(-1) + alpha * ((p - t) ** 2).sum(-1) / 2
    return value, _balanced(p - t, t)


def _hinge_rows(
    z: torch.Tensor, y: torch.Tensor, loss: str, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nonsmooth alpha/beta top-k hinge values, gradients, and rows with several
    subgradients (ties at the k-th margin or margins at a kink)."""
    n, c = z.shape
    h = 1 + z - z.gather(1, y[:, None])
    h = h.scatter(1, y[:, None], -math.inf)
    top, index = h.topk(min(k + 1, c - 1), dim=1)
    sel, sel_index = top[:, :k], index[:, :k]
    near = 1e-12 * (1 + z.abs().amax(1))
    if loss == "topk_hinge_alpha":
        mean = sel.mean(1)
        value = mean.clamp_min(0)
        a = (mean > 0).double()[:, None].expand(n, k) / k
        kink = mean.abs() <= near
        live = mean > -near
    else:
        value = sel.clamp_min(0).mean(1)
        a = (sel > 0).double() / k
        kink = (sel.abs() <= near[:, None]).any(1)
        live = top[:, k - 1] > -near
    tie = torch.zeros(n, dtype=torch.bool)
    if k + 1 < c:
        tie = ((top[:, k - 1] - top[:, k]).abs() <= near) & live
    grad = torch.zeros_like(z).scatter(1, sel_index, a)
    grad = grad.scatter(1, y[:, None], -a.sum(1, keepdim=True))
    return value, grad, tie | kink


def _multilabel_hinge_rows(
    z: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = torch.where(mask, z, math.inf)
    neg = torch.where(mask, -math.inf, z)
    low, low_i = pos.min(1)
    high, high_i = neg.max(1)
    margin = 1 + high - low
    on = (margin > 0).double()[:, None]
    grad = torch.zeros_like(z).scatter(1, low_i[:, None], -on).scatter(1, high_i[:, None], on)
    near = 1e-12 * (1 + z.abs().amax(1))
    tie = ((pos == low[:, None]).sum(1) > 1) | ((neg == high[:, None]).sum(1) > 1)
    ambiguous = (margin.abs() <= near) | (tie & (margin > -near))
    return margin.clamp_min(0), grad, ambiguous


def _mp_rows(
    z: torch.Tensor, target: torch.Tensor, loss: str, k: int, gamma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The mpmath / rational references of the test files, row by row."""
    import test_multilabel_kernels as ml
    import test_topk_kernels as tk

    values, grads = [], []
    for row, t in zip(z.tolist(), target.tolist(), strict=True):
        if loss == "multilabel_hinge":
            mask = t if isinstance(t, list) else [j == t for j in range(len(row))]
            v, g = ml.exact(row, mask, "multilabel_hinge_smooth", gamma)
        else:
            y = t.index(True) if isinstance(t, list) else t
            name = loss if loss == "topk_cross_entropy" else loss + "_smooth"
            v, g = tk.reference(row, y, name, k, gamma)
        values.append(v)
        grads.append(g)
    return torch.tensor(values, dtype=torch.float64), torch.tensor(grads, dtype=torch.float64)


def reference_rows(
    case: Case, z: torch.Tensor, target: torch.Tensor, teacher: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """float64 (rows,) values, (rows, C) gradients and ambiguous-gradient rows of class-last
    (rows, C) represented logits, or None where no independent reference applies."""
    loss, k, gamma = case.loss, case.k, case.gamma
    n, c = z.shape
    none = torch.zeros(n, dtype=torch.bool)
    if loss == "multiclass_hinge":
        loss, k = "topk_hinge_alpha", 1
    if loss in ("cross_entropy", "multilabel_cross_entropy") or (
        loss == "topk_cross_entropy" and k == 1
    ):
        return (*_entropy_rows(z, target), none)
    if loss == "proximal_cross_entropy":
        return (*_envelope(z, _target_rows(target, c), case.alpha), none)
    if loss in ("distillation", "proximal_distillation"):
        assert teacher is not None
        temp = case.temperature
        p = torch.softmax(teacher / temp, -1)
        if loss == "distillation":
            q = torch.log_softmax(z / temp, -1)
            return -(p * q).sum(-1), _balanced(q.exp() - p, p) / temp, none
        value, grad = _envelope(z / temp, p, case.alpha)
        return value, grad / temp, none
    indices = target if target.dtype != torch.bool else target.double().argmax(-1)
    if gamma == 0 and loss in ("topk_hinge_alpha", "topk_hinge_beta"):
        return _hinge_rows(z, indices, loss, k)
    if gamma == 0 and loss == "multilabel_hinge":
        mask = target if target.dtype == torch.bool else F.one_hot(target, c).bool()
        return _multilabel_hinge_rows(z, mask)
    if c <= SMALL and n <= 64:
        return (*_mp_rows(z, target, loss, k, gamma), none)
    return None


# ----------------------------------------------------------------------------- calls


def call(case: Case, logits: torch.Tensor, data: dict[str, Any], backend: str) -> torch.Tensor:
    """The loss of the case in its form (function, named function or module)."""
    loss, form = case.loss, case.form
    target, dim = data["target"], data["class_dim"]
    common: dict[str, Any] = dict(reduction=case.reduction, precision=case.precision)
    if loss == "ce":
        options = dict(common, backend=backend)
        if case.ignore is not None:
            options["ignore_index"] = case.ignore
        if form == "module":
            return L.CrossEntropyLoss(data["weight"], **options)(logits, target)
        return L.cross_entropy(logits, target, data["weight"], **options)
    if loss in ("distillation", "proximal_distillation"):
        options = dict(common, dim=dim)
        args: tuple[float, ...] = (case.temperature,)
        if loss == "proximal_distillation":
            args = (case.temperature, case.alpha)
        if form == "module":
            module = L.DistillationLoss if loss == "distillation" else L.ProximalDistillationLoss
            return module(*args, **options)(logits, data["teacher"])
        function = L.distillation_loss if loss == "distillation" else L.proximal_distillation_loss
        return function(logits, data["teacher"], *args, **options)
    options = dict(common, dim=dim, backend=backend)
    if form == "function":
        return L.sdca_loss(
            logits, target, loss, k=case.k, gamma=case.gamma, alpha=case.alpha, **options
        )
    if form == "module" and loss == "cross_entropy":
        return L.SDCALoss(loss, **options)(logits, target)
    named: dict[str, tuple[Callable[..., Any], Callable[..., Any], dict[str, Any]]] = {
        "cross_entropy": (L.multilabel_cross_entropy, L.MultilabelCrossEntropyLoss, {}),
        "multilabel_cross_entropy": (
            L.multilabel_cross_entropy,
            L.MultilabelCrossEntropyLoss,
            {},
        ),
        "topk_cross_entropy": (L.topk_cross_entropy, L.TopKCrossEntropyLoss, {"k": case.k}),
        "multiclass_hinge": (
            L.multiclass_hinge_loss,
            L.MulticlassHingeLoss,
            {"gamma": case.gamma},
        ),
        "topk_hinge_alpha": (
            L.topk_hinge_alpha_loss,
            L.TopKHingeAlphaLoss,
            {"k": case.k, "gamma": case.gamma},
        ),
        "topk_hinge_beta": (L.topk_hinge_loss, L.TopKHingeLoss, {"k": case.k, "gamma": case.gamma}),
        "multilabel_hinge": (
            L.multilabel_hinge_loss,
            L.MultilabelHingeLoss,
            {"gamma": case.gamma},
        ),
        "proximal_cross_entropy": (
            L.proximal_cross_entropy,
            L.ProximalCrossEntropyLoss,
            {"alpha": case.alpha},
        ),
    }
    function, module, extra = named[loss]
    if form == "module":
        return module(**extra, **options)(logits, target)
    return function(logits, target, **extra, **options)


@contextlib.contextmanager
def environment(extension: str, threads: int | None) -> Iterator[None]:
    """SDCA_CPU_EXTENSION (read at each call) and torch's thread count, restored after."""
    old_mode = os.environ.get("SDCA_CPU_EXTENSION")
    old_threads = torch.get_num_threads()
    os.environ["SDCA_CPU_EXTENSION"] = extension
    if threads is not None:
        torch.set_num_threads(threads)
    try:
        yield
    finally:
        if old_mode is None:
            os.environ.pop("SDCA_CPU_EXTENSION", None)
        else:
            os.environ["SDCA_CPU_EXTENSION"] = old_mode
        torch.set_num_threads(old_threads)


def paths(case: Case, extension_modes: tuple[str, ...] = ("on", "off")) -> list[tuple[str, str]]:
    """(SDCA_CPU_EXTENSION, backend) pairs of the case; distillation has no backend."""
    backends = ("auto",) if case.loss.endswith("distillation") else ("auto", "native", "torch")
    return [(mode, b) for mode in extension_modes for b in backends]


# ----------------------------------------------------------------------------- comparison


def _eps(case: Case) -> float:
    """Epsilon of the working arithmetic (float32 for every non-float64 call)."""
    double = case.dtype == torch.float64 or case.precision == "float64"
    return torch.finfo(torch.float64 if double else torch.float32).eps


def _out_eps(case: Case) -> float:
    """One rounding to the value dtype, promote(logits, float32)."""
    return torch.finfo(torch.float64 if case.dtype == torch.float64 else torch.float32).eps


def _unit(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """One unit in the last place of a 16-bit dtype at |x| (0 for wider dtypes)."""
    if dtype not in (torch.float16, torch.bfloat16):
        return torch.zeros_like(x)
    info = torch.finfo(dtype)
    mantissa = 7 if dtype == torch.bfloat16 else 10
    exponent = torch.floor(torch.log2(x.abs().clamp_min(info.tiny)))
    return torch.exp2(exponent - mantissa)


@dataclass
class Expected:
    value: torch.Tensor  # the reduced (or per-example) float64 value
    value_tol: torch.Tensor
    grad: torch.Tensor  # class-last gradient of the logits, upstream applied
    grad_tol: torch.Tensor
    ambiguous: int = 0


def _upstream(case: Case, value: torch.Tensor) -> torch.Tensor:
    """The upstream gradient: the case's scalar, or a fixed pattern of signs and sizes."""
    if case.reduction != "none":
        return torch.tensor(case.upstream, dtype=torch.float64)
    if case.upstream_layout == "expanded":
        return torch.full(value.shape, case.upstream, dtype=torch.float64)
    pattern = torch.tensor([1.0, -0.5, 2.0, 0.0, 0.25], dtype=torch.float64)
    flat = pattern.repeat(value.numel() // 5 + 1)[: value.numel()]
    return (flat * case.upstream).reshape(value.shape)


def _upstream_tensor(case: Case, value: torch.Tensor) -> torch.Tensor:
    """_upstream in the value's dtype and the case's upstream layout."""
    up = _upstream(case, value.detach().double()).to(value.dtype)
    if case.reduction != "none" or case.upstream_layout == "dense":
        return up
    if case.upstream_layout == "expanded":
        return torch.tensor(case.upstream, dtype=value.dtype).expand(value.shape)
    wide = torch.zeros((*up.shape[:-1], 2 * up.shape[-1]), dtype=up.dtype)
    wide[..., ::2] = up
    return wide[..., ::2]


def _grad_tol(case: Case, grads: torch.Tensor, factor: float = GRAD_EPS) -> torch.Tensor:
    """Per entry: factor eps of the row's largest entry plus two roundings (a gradient saved
    in the forward is rounded again by the upstream factor); for 16-bit gradients one unit
    of the dtype plus 2000 float32 eps of the row's largest entry."""
    row_max = grads.abs().amax(-1, keepdim=True)
    out = torch.finfo(case.dtype).eps
    tol = factor * _width(case) * _eps(case) * row_max + 2 * out * grads.abs() + 1e-300
    if case.dtype in (torch.float16, torch.bfloat16):
        eps32 = torch.finfo(torch.float32).eps
        tol = tol + _unit(grads, case.dtype) + HALF_GRAD_EPS32 * eps32 * row_max
    return tol.expand_as(grads)


def _reference_error(case: Case) -> float:
    """The rounding of a float64 reference gradient formed as a difference of
    probabilities (p - t, q - p): their sum is at most 2 (over T for distillation)."""
    if case.loss in HINGES or case.loss == "topk_cross_entropy" and case.k > 1:
        return 0.0
    scale = 2 / case.temperature if case.loss.endswith("distillation") else 2.0
    return (16 + 2 * math.sqrt(case.c)) * torch.finfo(torch.float64).eps * scale


def _width(case: Case) -> float:
    """Wide rows: exponent arguments near log(C) and sums of C terms round by about
    sqrt(C / 1000) times more (float32 eager PyTorch alike)."""
    return max(1.0, math.sqrt(case.c / 1000))


def _reduce(case: Case, rows: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Reduction of per-row values (or tolerances) with per-row weights (mean: weighted)."""
    if case.reduction == "none":
        return rows
    total = (rows * weights).sum()
    if case.reduction == "sum":
        return total
    return total / weights.sum()


def expected(case: Case, data: dict[str, Any]) -> Expected | None:
    """The float64 reference of the case (None: no independent reference)."""
    z = data["z"].double()  # class last, (n, *extra, c)
    shape, c = z.shape[:-1], z.shape[-1]
    flat = z.reshape(-1, c)
    target = data["target"]
    eps = _eps(case)
    if case.loss == "ce":
        labels = target.reshape(-1)
        row_w = torch.ones(labels.shape, dtype=torch.float64)
        if data["weight"] is not None:
            row_w = data["weight"].double()[labels.clamp(0, c - 1)]
        if case.ignore is not None:
            row_w = torch.where(labels == case.ignore, 0.0, row_w)
        values, grads = _entropy_rows(flat, labels.clamp(0, c - 1))
        per_row = VALUE_EPS * _width(case) * eps * (values.abs() + 1 + flat.abs().amax(-1))
        if case.reduction == "none":
            values, per_row = values * row_w, per_row * row_w
        value = _reduce(case, values, row_w).reshape(shape if case.reduction == "none" else ())
        tol = _reduce(case, per_row, row_w).reshape(value.shape) + _out_eps(case) * value.abs()
        up = _upstream(case, value)
        row_up = up.reshape(-1) if case.reduction == "none" else up.expand(row_w.shape)
        factor = row_up * row_w
        if case.reduction == "mean":
            factor = factor / row_w.sum()
            if row_w.sum() == 0:  # as F.cross_entropy: NaN, but zero for ignored rows
                ignored = labels == case.ignore if case.ignore is not None else row_w < 0
                factor = torch.where(ignored, 0.0, math.nan)
        grads = grads * factor[:, None]
        grad_tol = _grad_tol(case, grads) + _reference_error(case) * factor.abs()[:, None]
        grad_tol = grad_tol.nan_to_num(1.0).reshape(z.shape)
        return Expected(value, tol, grads.reshape(z.shape), grad_tol)
    teacher = data["teacher"]
    flat_target = target.reshape(-1, c) if target.dtype == torch.bool else target.reshape(-1)
    flat_teacher = None if teacher is None else teacher.detach().double().reshape(-1, c)
    rows = reference_rows(case, flat, flat_target, flat_teacher)
    if rows is None:
        return None
    values, grads, ambiguous = rows
    if flat_teacher is not None:
        scale = 1 + torch.cat([flat, flat_teacher], -1).abs().amax(-1) / case.temperature
    else:
        scale = 1 + flat.abs().amax(-1)
    proximal = case.loss.startswith("proximal")
    per_row = (PROX_EPS if proximal else VALUE_EPS) * _width(case) * eps * (values.abs() + scale)
    ones = torch.ones_like(values)
    value = _reduce(case, values, ones).reshape(shape if case.reduction == "none" else ())
    tol = _reduce(case, per_row, ones).reshape(value.shape) + _out_eps(case) * value.abs()
    up = _upstream(case, value)
    row_up = up.reshape(-1) if case.reduction == "none" else up.expand(values.shape)
    if case.reduction == "mean":
        row_up = row_up / values.numel()
    grads = grads * row_up[:, None]
    grad_tol = _grad_tol(case, grads, PROX_EPS if proximal else GRAD_EPS)
    grad_tol = grad_tol + _reference_error(case) * row_up.abs()[:, None]
    if case.loss in HINGES and case.gamma > 0:
        # A smoothed hinge's gradient projects the margins (rounded to eps of the scores'
        # scale) and divides by gamma: a 1-Lipschitz map in the 2-norm.
        margin = GRAD_EPS * eps * math.sqrt(c) * scale / case.gamma
        grad_tol = grad_tol + (margin * row_up.abs())[:, None]
    grad_tol = torch.where(ambiguous[:, None], math.inf, grad_tol)  # several subgradients
    return Expected(
        value, tol, grads.reshape(z.shape), grad_tol.reshape(z.shape), int(ambiguous.sum())
    )


@dataclass
class Result:
    path: str
    ok: bool
    message: str = ""
    value_err: float = 0.0  # error over tolerance (<= 1 passes)
    grad_err: float = 0.0


def run_path(
    case: Case, data: dict[str, Any], backend: str
) -> tuple[torch.Tensor, torch.Tensor | None, str]:
    """Value and leaf gradient of one call; exceptions and contract breaks as messages."""
    leaf, make = view(case, data["z"], data["class_dim"])
    leaf.requires_grad_(True)
    call_data = dict(data)
    dim = data["class_dim"]
    if data["teacher"] is not None:
        teacher = data["teacher"].detach().movedim(-1, dim).clone()
        call_data["teacher"] = teacher.requires_grad_(data["teacher"].requires_grad)
    if case.mask and dim != -1:
        call_data["target"] = data["target"].movedim(-1, dim)
    try:
        value = call(case, make(leaf), call_data, backend)
        if case.reduction != "none" and case.upstream == 1.0:
            value.backward()
        else:
            value.backward(_upstream_tensor(case, value))
    except Exception as error:  # noqa: BLE001 - reported as a failure of this path
        return torch.empty(0), None, f"{type(error).__name__}: {error}"
    if call_data["teacher"] is not None and call_data["teacher"].grad is not None:
        return value.detach(), None, "the teacher received a gradient"
    value_dtype = torch.float64 if case.dtype == torch.float64 else torch.float32
    if value.dtype != value_dtype:
        return value.detach(), None, f"value dtype {value.dtype}"
    if leaf.grad is None or leaf.grad.dtype != case.dtype or leaf.grad.shape != leaf.shape:
        return value.detach(), None, "gradient missing or of the wrong dtype or shape"
    return value.detach(), leaf.grad.detach(), ""


def _to_leaf(case: Case, data: dict[str, Any], grad: torch.Tensor) -> torch.Tensor:
    """A class-last gradient of the logits mapped to the leaf of the case's layout."""
    leaf, make = view(case, data["z"].double(), data["class_dim"])
    leaf.requires_grad_(True)
    (g,) = torch.autograd.grad(make(leaf), leaf, grad.movedim(-1, data["class_dim"]))
    return g


def _ratio(actual: torch.Tensor, expected: torch.Tensor, tol: torch.Tensor) -> float:
    """The largest error over its tolerance; NaN agrees only with NaN, inf only with inf."""
    if actual.shape != expected.shape:
        return math.inf
    err = (actual - expected).abs()
    same = (actual == expected) | (actual.isnan() & expected.isnan())  # also equal infinities
    err = torch.where(same, 0.0, err)
    err = torch.where(err.isnan(), math.inf, err)
    if err.numel() == 0:
        return 0.0
    return float(torch.where((err == 0) | tol.isinf(), 0.0, err / tol).max())


def check(
    case: Case,
    extension_modes: tuple[str, ...] = ("on", "off"),
    thread_counts: tuple[int | None, ...] = (None, 1),
) -> list[Result]:
    """Every CPU path of the case against the reference (or, without one, against the
    float64 tensor backend)."""
    data = inputs(case)
    exp = expected(case, data)
    if exp is None:
        base = Case(**{**case.__dict__, "precision": "float64"})
        with environment("off", None):
            value, grad, error = run_path(base, data, "torch")
        if error:
            return [Result("baseline", False, f"float64 tensor baseline: {error}")]
        assert grad is not None
        ref_value, ref_grad = value.double(), grad.double()
        value_tol = 1e3 * _eps(case) * (1 + ref_value.abs()) + _out_eps(case) * ref_value.abs()
        rows = ref_grad.abs().amax(-1, keepdim=True)
        grad_tol = 1e3 * _eps(case) * rows + 2 * torch.finfo(case.dtype).eps * ref_grad.abs()
        grad_tol = grad_tol + _unit(ref_grad, case.dtype) + 1e-300
        if case.loss in HINGES and case.gamma > 0:  # as in expected()
            flat = data["z"].double().reshape(-1, case.c)
            up = _upstream(case, ref_value)
            rows_up = up.reshape(-1) if case.reduction == "none" else up.expand(flat.shape[0])
            if case.reduction == "mean":
                rows_up = rows_up / flat.shape[0]
            margin = GRAD_EPS * _eps(case) * math.sqrt(case.c) / case.gamma
            margin = margin * (1 + flat.abs().amax(-1)) * rows_up.abs()
            margin = margin[:, None].expand(flat.shape).reshape(data["z"].shape)
            grad_tol = grad_tol + _to_leaf(case, data, margin).abs()
    else:
        ref_value, value_tol = exp.value, exp.value_tol
        ref_grad = _to_leaf(case, data, exp.grad)
        grad_tol = _to_leaf(case, data, exp.grad_tol.clamp_max(1e300)).abs()
    if case.layout == "expanded":  # autograd sums the rows' gradients in their dtype
        grad_tol = grad_tol + case.n * torch.finfo(case.dtype).eps * ref_grad.abs()
        # where it may overflow (16-bit), any result passes
        grad_tol = torch.where(ref_grad.abs() > torch.finfo(case.dtype).max / 2, math.inf, grad_tol)
    results = []
    for mode, backend in paths(case, extension_modes):
        for threads in thread_counts:
            name = f"ext={mode} backend={backend} threads={threads or 'default'}"
            with environment(mode, threads):
                value, grad, error = run_path(case, data, backend)
            if error:
                results.append(Result(name, False, error))
                continue
            assert grad is not None
            ve = _ratio(value.double(), ref_value, value_tol)
            ge = _ratio(grad.double(), ref_grad, grad_tol.expand_as(ref_grad))
            ok = ve <= 1 and ge <= 1
            message = "" if ok else f"value err/tol {ve:.3g}, gradient err/tol {ge:.3g}"
            results.append(Result(name, ok, message, ve, ge))
    return results
