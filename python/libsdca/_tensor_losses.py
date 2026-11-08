"""Pure-PyTorch tensor implementations of the SDCA losses and their projections
(the fallback and backend="torch" paths of libsdca.torch).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

import torch
from torch import nn

# Objectives of one target per example.
_SINGLE_TARGET = frozenset(
    [
        "topk_cross_entropy",
        "topk_hinge_alpha",
        "topk_hinge_beta",
        "topk_hinge_alpha_smooth",
        "topk_hinge_beta_smooth",
    ]
)


class _Saved(Protocol):
    """The autograd context in backward."""

    saved_tensors: tuple[torch.Tensor, ...]
    double: bool


class _Forward(Protocol):
    """The autograd context in _TensorEntropy.forward."""

    double: bool

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


def _work(z: torch.Tensor, double: bool = False) -> torch.Tensor:
    """Working logits: float64 when requested, float32 for 16-bit inputs, else unchanged."""
    if double:
        return z.double()
    if z.dtype in (torch.float16, torch.bfloat16):
        return z.float()
    return z


def _first(positive: torch.Tensor) -> torch.Tensor:
    """Index (..., 1) of the first positive entry."""
    # A bool-to-uint8 view miscompiles under Inductor (torch 2.14); a byte copy does not.
    return positive.to(torch.uint8).argmax(-1, keepdim=True)


def _normalizer(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row maximum, shifted scores z - maximum, and log-sum-exp of the shifted scores."""
    maximum, index = z.max(dim=-1, keepdim=True)
    shifted = z - maximum
    others = shifted.scatter(-1, index, -torch.inf)
    if not z.is_cuda:
        tail = others.exp().sum(dim=-1, keepdim=True)
        # At most n subnormal rounding errors, below eps of this tail.
        if bool((tail >= z.shape[-1] * torch.finfo(z.dtype).tiny).all()):
            return maximum, shifted, torch.log1p(tail)
    # Sum before rounding into the subnormal range: individual exponentials
    # may vanish even though their aggregate is representable.
    top = others.amax(dim=-1, keepdim=True)
    origin = torch.where(torch.isfinite(top), top, 0.0)
    half = (top * 0.5).exp()
    tail = half * (half * (others - origin).exp().sum(dim=-1, keepdim=True))
    # Some vector log1p implementations lose a subnormal quantum. Below
    # eps/2, the quadratic correction is below half a relative epsilon.
    lognorm = torch.where(tail < torch.finfo(z.dtype).eps / 2, tail, torch.log1p(tail))
    return maximum, shifted, lognorm


class _TensorEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _Forward, logits: torch.Tensor, positive: torch.Tensor, double: bool
    ) -> torch.Tensor:
        z = _work(logits, double)
        maximum, _, lognorm = _normalizer(z)
        counts = positive.sum(-1, dtype=z.dtype)
        # Masked before the sum: an infinite gap of a non-target is not 0 * inf.
        gap = maximum - z
        divisor = counts[:, None]
        # Normalize before summing: both a gap and its sum can overflow even
        # when their average is representable. Preserve small finite gaps.
        scaled = torch.where(torch.isfinite(gap), gap / divisor, maximum / divisor - z / divisor)
        gaps = torch.where(positive, scaled, 0.0).sum(-1)
        invalid = (counts == 0) | ~z.isfinite().all(-1)
        ctx.save_for_backward(logits, positive)
        ctx.double = double
        return (gaps + lognorm.squeeze(-1)).masked_fill(invalid, math.nan)

    @staticmethod
    def backward(ctx: _Saved, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (grad_output,) = grad_outputs
        logits, positive = ctx.saved_tensors
        z = _work(logits, ctx.double)
        invalid = (positive.sum(-1, keepdim=True) == 0) | ~z.isfinite().all(-1, keepdim=True)
        shifted = z - z.max(dim=-1, keepdim=True).values
        info, classes = torch.finfo(z.dtype), z.shape[-1]
        ordinary = not z.is_cuda and bool(
            (shifted.amin() > math.log(classes * info.tiny))
            & (grad_output.abs().amax() <= info.max / (2 * classes))
        )
        # Normal probabilities survive division by den <= classes. Bound the
        # upstream too, so accumulating positive-label differences cannot overflow.
        root = None if ordinary else (shifted * 0.5).exp()
        e = shifted.exp() if root is None else root.square()
        del shifted
        den = e.sum(-1, keepdim=True)
        anchor = _first(positive)
        delta = z - z.gather(-1, anchor)
        # The clamped unselected branches prevent expm1 overflow and keep
        # higher derivatives well-defined even at equal positive logits.
        differences = (
            torch.where(
                delta >= 0,
                -e * torch.expm1((-delta).clamp(max=1)),
                e.gather(-1, anchor) * torch.expm1(delta.clamp(max=1)),
            )
            / den
        )
        if root is None:
            neg = torch.where(positive, 0.0, e / den)
            count = positive.sum(-1, keepdim=True, dtype=z.dtype)
            correction = (
                torch.where(positive, differences, 0.0).sum(-1, keepdim=True)
                + neg.sum(-1, keepdim=True)
            ) / count
            gradient = torch.where(positive, differences - correction, neg)
            out = (gradient * grad_output[:, None]).masked_fill(invalid, math.nan)
            return out.to(logits.dtype), None, None
        negative_root = torch.where(positive, 0.0, root)
        largest = negative_root.amax(-1, keepdim=True)
        scale = torch.where(largest > 0, largest, 1.0)
        total = (negative_root / scale).square().sum(-1, keepdim=True)
        upstream = grad_output[:, None] / den
        negative_mass = scale * ((scale * upstream) * total)
        neg = negative_root * (negative_root * upstream)
        differences = differences * grad_output[:, None]
        count = positive.sum(-1, keepdim=True, dtype=z.dtype)
        correction = (
            torch.where(positive, differences, 0.0).sum(-1, keepdim=True) + negative_mass
        ) / count
        gradient = torch.where(positive, differences - correction, neg).masked_fill(
            invalid, math.nan
        )
        return gradient.to(logits.dtype), None, None


def _softplus(x: torch.Tensor) -> torch.Tensor:
    """log1p(exp(x)) without the linear cutoff of nn.functional.softplus."""
    return torch.where(x > 0, x + torch.log1p(torch.exp(-x)), torch.log1p(torch.exp(x)))


def _search(
    lo: torch.Tensor, hi: torch.Tensor, size: int, holds: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Last index in [lo, hi] (hi - lo <= size) of a monotone predicate that holds at lo."""
    for _ in range(size.bit_length()):
        mid = (lo + hi + 1) // 2
        ok = holds(mid)
        lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid - 1)
    return lo


def _next(points: torch.Tensor, index: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
    """points[index + 1], or -inf past last."""
    following = points.gather(-1, (index + 1).clamp(max=points.shape[-1] - 1))
    return following.masked_fill(index >= last, -math.inf)


def _knapsack(
    h: torch.Tensor,
    valid: torch.Tensor,
    cap: float,
    rhs: float,
    ordered: torch.Tensor | None = None,
    max_u: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Origin and correction of t with sum(clip(h - t, 0, cap)) = rhs over the valid entries.

    Bisection counts the entries at the cap, then the free ones, from the clipped mass
    relative to a represented score: a cap below the resolution of the scores still
    counts, and saturated far-off entries cannot cancel the free mass. ordered are the
    sorted valid scores if known; max_u bounds the capped count (cap * (max_u + 1) > rhs).
    """
    if ordered is None:
        ordered = h.masked_fill(~valid, -math.inf).sort(-1, descending=True).values
    count = valid.sum(-1, keepdim=True)

    def within(j: torch.Tensor, shift: float) -> torch.Tensor:
        # The mass at t = ordered[j - 1] + shift is at most rhs.
        origin = ordered.gather(-1, (j - 1).clamp(min=0))
        parts = torch.where(valid, ((h - origin) - shift).clamp(0, cap), 0.0)
        return parts.sum(-1, keepdim=True) <= rhs

    size = h.shape[-1]
    if max_u == 0:
        u = torch.zeros_like(count)
    else:
        span = size if max_u is None else min(size, max_u)
        hi = count if max_u is None else count.clamp(max=max_u)
        u = _search(torch.zeros_like(count), hi, span, lambda j: (j == 0) | within(j, -cap))
    end = _search(u, count, size, lambda j: (j == u) | within(j, 0.0))
    origin = ordered.gather(-1, (end - 1).clamp(min=0))
    position = torch.arange(1, size + 1, device=h.device)
    free = (position > u) & (position <= end)
    excess = torch.where(free, ordered - origin, 0.0).sum(-1, keepdim=True)
    excess = excess - (rhs - cap * u.to(h.dtype))
    correction = (excess / (end - u).clamp(min=1)).clamp(max=0)
    correction = correction.maximum(_next(ordered, end - 1, count - 1) - origin)
    # Without free entries the mass is flat: t = origin - cap caps the first u.
    return origin, torch.where(end > u, correction, -cap)


def _topk_cone(
    h: torch.Tensor, valid: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Threshold t, cap c of the projection clip(h - t, 0, c) onto 0 <= x_i <= sum(x)/k, and
    the sorted valid scores.

    All entries but one are valid. Bisection on the monotone stationarity residuals
    finds the number u of capped entries, then the number m of free ones.
    """
    size = h.shape[-1] - 1
    ordered = h.masked_fill(~valid, -math.inf).sort(-1, descending=True).values

    def above(t: torch.Tensor) -> torch.Tensor:
        return torch.where(valid, (h - t).clamp(min=0), 0.0).sum(-1, keepdim=True)

    def capped(j: torch.Tensor) -> torch.Tensor:
        # The cap level t + c = ordered[j] lies above the solution's.
        level = ordered.gather(-1, j.clamp(min=0))
        t = -above(level) / k
        return (j < 0) | (above(t) / k + 2 * t < level)

    start = torch.full_like(ordered[:, :1], -1, dtype=torch.int64)
    u = _search(start, start + k, k, capped) + 1
    position = torch.arange(h.shape[-1], device=h.device)
    head = torch.where(position < u, ordered, 0.0).sum(-1, keepdim=True)

    def free(j: torch.Tensor) -> torch.Tensor:
        # The threshold t = ordered[j] lies above the solution's.
        t = ordered.gather(-1, j.clamp(min=0))
        return (j < u) | (above(t) / k + 2 * t < (k * t + head) / u.clamp(min=1))

    index = _search(u - 1, torch.full_like(u, size - 1), size, free)
    m = torch.where(u == 0, (ordered > 0).sum(-1, keepdim=True), index - u + 1)
    rest = torch.where((position >= u) & (position < u + m), ordered, 0.0).sum(-1, keepdim=True)
    d = ((k - u) ** 2 + m * u).to(h.dtype)
    t = (u * rest - (k - u) * head) / d
    c = (m * head + (k - u) * rest) / d
    # Exactly the k largest are capped when the rest lie below their threshold.
    whole = _exact_sum(ordered[:, :k])
    top = ordered[:, k - 1 : k] - whole / k
    # As topk_cone_tolerance: 16 eps relative to scores below one keeps tiny margins.
    tol = 16 * torch.finfo(h.dtype).eps * ordered[:, :k].abs().sum(-1, keepdim=True).clamp(max=1)
    full = (u == k) | (top >= ordered[:, k : k + 1] - tol)
    t, c = torch.where(full, top, t), torch.where(full, whole / k, c)
    zero = whole <= tol
    return t.masked_fill(zero, 0), c.masked_fill(zero, 0), ordered


def _topk_simplex(h: torch.Tensor, valid: torch.Tensor, k: int, rhs: float) -> torch.Tensor:
    """Projection onto 0 <= x_i <= sum(x)/k, sum(x) <= rhs: the cone's unless its sum exceeds rhs."""
    t, c, ordered = _topk_cone(h, valid, k)
    origin, correction = _knapsack(h, valid, rhs / k, rhs, ordered, k)
    cone = (h - t).clamp(min=0).minimum(c)
    full = ((h - origin) - correction).clamp(0, rhs / k)
    return torch.where(k * c > rhs, full, cone).masked_fill(~valid, 0)


def _knapsack_le(
    h: torch.Tensor, valid: torch.Tensor, cap: float, rhs: float, max_u: int | None = None
) -> torch.Tensor:
    """Projection onto 0 <= x_i <= cap, sum(x) <= rhs over the valid entries."""
    box = torch.where(valid, h.clamp(0, cap), 0.0)
    if not h.is_cuda and bool((box.sum(-1, keepdim=True) <= rhs).all()):
        return box
    origin, correction = _knapsack(h, valid, cap, rhs, max_u=max_u)
    full = torch.where(valid, ((h - origin) - correction).clamp(0, cap), 0.0)
    return torch.where(box.sum(-1, keepdim=True) > rhs, full, box)


def _two_simplex(h: torch.Tensor, positive: torch.Tensor, rhs: float) -> torch.Tensor:
    """Projection onto x >= 0 whose positive and other entries have equal sums <= rhs."""
    # A cap of 2 rhs leaves no entry capped.
    split = [_knapsack(h, part, 2 * rhs, rhs, max_u=0) for part in (positive, ~positive)]
    (oa, ca), (ob, cb) = split
    simplex = torch.where(positive, (h - oa) - ca, (h - ob) - cb)
    # Below rhs, sum(max(h - t, 0)) over the positives = sum(max(h + t, 0)) over the others.
    points = torch.where(positive, h, -h).sort(-1, descending=True).values

    def balance(t: torch.Tensor) -> torch.Tensor:
        parts = torch.where(positive, (h - t).clamp(min=0), -(h + t).clamp(min=0))
        return parts.sum(-1, keepdim=True)

    last = torch.full_like(h[:, :1], h.shape[-1] - 1, dtype=torch.int64)
    index = _search(
        torch.zeros_like(last), last, h.shape[-1], lambda j: balance(points.gather(-1, j)) <= 0
    )
    origin, below = points.gather(-1, index), _next(points, index, last)
    left, right = positive & (h > below), ~positive & (-h <= below)
    parts = torch.where(left, h - origin, 0.0) - torch.where(right, h + origin, 0.0)
    excess = parts.sum(-1, keepdim=True)
    t = excess / (left.sum(-1, keepdim=True) + right.sum(-1, keepdim=True)).clamp(min=1)
    t = t.clamp(max=0).maximum(below - origin)
    equal = torch.where(positive, (h - origin) - t, (h + origin) + t)
    # Rounding of the thresholds' own terms, not a fixed floor, which would swallow tiny margins.
    scale = ((oa.abs() + ob.abs()) + (ca.abs() + cb.abs())).clamp(min=rhs)
    active = (oa + ob) + (ca + cb) >= -torch.finfo(h.dtype).eps * scale
    return torch.where(active, simplex, equal).clamp(0, rhs)


def _exact_sum(x: torch.Tensor, low: torch.Tensor | None = None) -> torch.Tensor:
    """Sum (plus low) over the last dimension in double-double, by pairwise TwoSum."""
    n = x.shape[-1]
    x = nn.functional.pad(x, (0, (1 << (n - 1).bit_length()) - n))
    low = torch.zeros_like(x[:, :1]) if low is None else low
    while x.shape[-1] > 1:
        a, b = x[:, 0::2], x[:, 1::2]
        x = a + b
        rounded = x - a
        # An infinite sum has no rounding error to carry (and inf - inf is nan).
        error = torch.where(x.isfinite(), (a - (x - rounded)) + (b - rounded), 0.0)
        low = low + error.sum(-1, keepdim=True)
    return x + low


def _offset_mean(z: torch.Tensor, positive: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Mean of z - z0 over the non-target entries, accumulated in double-double. The scores are
    scaled by a power of two first: differences of scores near finfo.max and their sum
    overflow."""
    scale = 2.0 ** -(2 * z.shape[-1]).bit_length()
    z0 = z0 * scale
    # The scaled scores as alpha of the subtractions: no pass of their own.
    total = torch.add(-z0, z, alpha=scale)
    rounded = torch.add(total, z, alpha=-scale)
    low = torch.add(rounded - total, z, alpha=scale) + (-z0 - rounded)
    low = low.masked_fill(positive | total.isinf(), 0)
    total = total.masked_fill(positive, 0)
    return _exact_sum(total, low.sum(-1, keepdim=True)) / (z.shape[-1] - 1) / scale


def _entropy_topk(
    z: torch.Tensor, positive: torch.Tensor, z0: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k entropy loss and its non-target gradient, the capped probabilities."""
    n = z.shape[-1]
    if k == n - 1:
        # Every competitor is capped: only their mean matters.
        mean = _offset_mean(z, positive, z0)
        loss, mass = _topk_tail(mean.clamp(max=0), mean + math.log(k), torch.full_like(mean, k))
        return loss.squeeze(-1), (mass / k).expand_as(z).masked_fill(positive, 0)
    others = z.masked_fill(positive, -math.inf)
    # The k largest competitors in order (k < n - 1), and the sum of the exponentials of
    # the others relative to the k-th: no full sort.
    ordered, index = others.topk(k, dim=-1)
    last = ordered[:, k - 1 :]
    rest = (others.scatter(-1, index, -math.inf) - last).exp().sum(-1, keepdim=True)
    largest = ordered[:, :1]
    position = torch.arange(k, device=z.device)

    def tail(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # ordered[u] and the sum of the exponentials of the competitors below it, relative
        # to it: free scores far below a capped one keep their differences.
        free = ordered.gather(-1, u)
        terms = torch.where(position > u, (ordered - free).exp(), 0.0).sum(-1, keepdim=True)
        return free, terms + (last - free).exp() * rest

    def capped(j: torch.Tensor) -> torch.Tensor:
        # The native active-set test relative to ordered[j]: it exceeds the threshold.
        threshold = tail(j.clamp(min=0))[1].log1p() - torch.log((k - j).to(z.dtype))
        eps = torch.finfo(z.dtype).eps
        return (j < 0) | (threshold + 32 * eps * threshold.abs().clamp(min=1) < 0)

    start = torch.full_like(largest, -1, dtype=torch.int64)
    u = _search(start, start + k - 1, k - 1, capped) + 1
    free, sums = tail(u)
    lse = sums.log1p()
    head = torch.where(position < u, ordered - largest, 0.0).sum(-1, keepdim=True)
    rest, offset, total = (k - u).to(z.dtype), largest - z0, sums + 1
    mean = (head + rest * ((free - largest) + lse - torch.log(rest))) / k
    # Rows of no capped competitor are cross-entropy's (as single_entropy_row); a positive
    # offset (a capped competitor far above the free ones) joins the mean.
    above = offset.clamp(min=0)
    scale = torch.where(u == 0, total, (mean + above).clamp(max=0).exp() * k)
    loss, mass = _topk_tail(offset - above, offset + mean + math.log(k), scale)
    # Each competitor cap * exp(z - free) / spread, at most cap, spread the free ones' sum of
    # exp(z - free) per uncapped slot: a log-domain shift would round at the magnitude of
    # log(mass).
    cap, spread = mass / k, total / rest
    share = ((others - free).minimum(spread.log()).exp() / spread).clamp(max=1)
    return loss.squeeze(-1), cap * share


def _topk_tail(
    offset: torch.Tensor, odds: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k entropy loss and mass of the odds, offset + log(scale) (offset <= 0) where they
    are below -36: such a tail is its own loss to half an eps, from squared half exponentials
    rounded once (as single_entropy_row); the log-domain odds round at the magnitude of the
    offset."""
    half = (offset * 0.5).exp()
    tail = half * (half * scale)
    far = odds < -36
    loss = torch.where(far, tail, _softplus(odds))
    return loss, torch.where(far, tail / (1 + tail), torch.exp(-_softplus(-odds)))


# Margins below this fraction of their terms cancel (sdca/math/compensated.h).
_MARGIN_CANCELLATION = 1 / 16


def _margin(z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """1 + z - z0; when it cancels, with the low parts of both roundings (TwoSum), as in loss.h."""
    d = z - z0
    h = d + 1
    cancels = h.abs() < _MARGIN_CANCELLATION * d.abs()
    if not z.is_cuda and not bool(cancels.any()):
        return h
    rounded = d - z
    low = (z - (d - rounded)) + (-z0 - rounded)
    rounded = h - d
    low = low + ((d - (h - rounded)) + (1 - rounded))
    return torch.where(cancels & low.isfinite(), h + low, h)


_scalars: dict[tuple[torch.device, torch.dtype, float, int | None], torch.Tensor] = {}


def _over(x: torch.Tensor, scalar: float) -> torch.Tensor:
    """x / scalar divided on the device: CUDA multiplies by 1/scalar for a Python scalar
    divisor, and 1/scalar overflows below 5.6e-309 (0 * inf = NaN)."""
    stream = torch.cuda.current_stream(x.device).cuda_stream if x.is_cuda else None
    key = (x.device, x.dtype, scalar, stream)
    denom = _scalars.get(key)
    if denom is None:
        if len(_scalars) > 64:
            _scalars.clear()
        # Inference must not leave tensors that autograd cannot save for training.
        with torch.inference_mode(False):
            denom = torch.full((), scalar, dtype=x.dtype, device=x.device)
        _scalars[key] = denom
    return x / denom


def _smooth_value(h: torch.Tensor, a: torch.Tensor, gamma: float) -> torch.Tensor:
    """Sum of (h a - a^2/2) / gamma, scaled by a / gamma first where the products leave the normal
    range (overflow or underflow)."""
    ha, aa = h * a, a * a / 2
    value, tiny = (ha - aa).abs(), torch.finfo(h.dtype).tiny / torch.finfo(h.dtype).eps
    normal = (value <= torch.finfo(h.dtype).max) & (value >= tiny)
    return torch.where(normal, _over(ha - aa, gamma), (h - a / 2) * _over(a, gamma)).sum(-1)


def _single_target(
    z: torch.Tensor, positive: torch.Tensor, objective: str, k: int, gamma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """A single-target loss and its gradient at the other classes (zero at the target)."""
    z0 = z.gather(-1, _first(positive))
    if objective == "topk_cross_entropy":
        return _entropy_topk(z, positive, z0, k)
    valid, h = ~positive, _margin(z, z0)
    if objective in ("topk_hinge_alpha", "topk_hinge_beta"):
        masked = h.masked_fill(positive, -math.inf)
        if k == 1:
            top, order = masked.max(dim=-1, keepdim=True)
        elif not masked.is_cuda and k + 1 <= masked.shape[-1]:
            top1, order1 = masked.topk(k + 1, dim=-1)
            if bool((top1[:, k - 1] > top1[:, k]).all()):
                top, order = top1[:, :k], order1[:, :k]
            else:
                top, order = masked.sort(dim=-1, descending=True, stable=True)
                top, order = top[:, :k], order[:, :k]
        else:
            top, order = masked.sort(dim=-1, descending=True, stable=True)
            top, order = top[:, :k], order[:, :k]
        if objective == "topk_hinge_beta":
            top = top.clamp(min=0)
        value = (top / k).sum(-1).clamp(min=0)
        chosen = (value[:, None] > 0).expand_as(top)
        if objective == "topk_hinge_beta":
            chosen = chosen & (top > 0)
        return value, torch.zeros_like(z).scatter(-1, order, chosen.to(z.dtype) / k)
    if objective == "topk_hinge_alpha_smooth" and k > 1:
        a = _topk_simplex(h, valid, k, gamma)
    else:
        a = _knapsack_le(h, valid, gamma / k, gamma, 0 if k == 1 else k)
    h = h.masked_fill(positive, 0)
    return _smooth_value(h, a, gamma), _over(a, gamma)


def _multilabel_hinge(
    z: torch.Tensor, positive: torch.Tensor, gamma: float, smooth: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multilabel hinge loss and gradient, relative to the first positive score."""
    low = z.masked_fill(~positive, math.inf).argmin(-1, keepdim=True)
    high = z.masked_fill(positive, -math.inf).argmax(-1, keepdim=True)
    zlow, zhigh = z.gather(-1, low), z.gather(-1, high)
    if not smooth:
        value = _margin(zhigh, zlow).clamp(min=0)
        sign = (value > 0).to(z.dtype)
        gradient = torch.zeros_like(z).scatter(-1, low, -sign).scatter(-1, high, sign)
        return value.squeeze(-1), gradient
    z0 = z.gather(-1, _first(positive))
    d = z - z0
    h = torch.where(positive, 0.5 - d, 0.5 + d)
    # A nearly cancelling largest pairwise margin goes in one entry, as in loss.h.
    scale = 1 + (zlow - z0).abs() + (zhigh - z0).abs()
    recenter = ((zhigh - zlow) + 1).abs() < _MARGIN_CANCELLATION * scale
    h = torch.where(recenter, torch.where(positive, zlow - z, _margin(z, zlow)), h)
    a = _two_simplex(h, positive, gamma)
    return _smooth_value(h, a, gamma), _over(torch.where(positive, -a, a), gamma)
