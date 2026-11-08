"""proximal_entropy on tensors: the block solve of
include/sdca/math/entropy.h (entropy_block_solver), batched over examples.

It works in float64 (or float32, with fewer fixed inner steps) and differs
from the native solve in that every block of a batch takes the same number
of root steps (rows that converged stay fixed), the inner residual solve
takes a fixed six (float32: four) bracketed Halley steps, and convergence is
checked on the host every few steps. Gradients are first order only.
"""

from __future__ import annotations

import math

import torch


def _lambert_w_exp(x: torch.Tensor) -> torch.Tensor:
    """W(exp(x)): Halley steps on exp(u) + u = x for u = log W from log(softplus(x))."""
    fp32 = x.dtype == torch.float32
    cut, large, steps = (-18.0, 1e7, 2) if fp32 else (-36.0, 1e16, 3)
    u = torch.nn.functional.softplus(x, threshold=30).log_()
    e = torch.empty_like(u)
    g = torch.empty_like(u)
    d = torch.empty_like(u)
    for _ in range(steps):
        torch.exp(u, out=e)
        torch.sub(e, x, out=g).add_(u)
        torch.add(e, 1.0, out=d)
        e.mul_(g).div_(d).mul_(-0.5).add_(d)
        g.div_(e)
        u.sub_(g)
    w = u.exp_()
    if not fp32:
        # A last Newton step on W itself: exp(u) loses the low bits of a large u.
        torch.log(w, out=d)
        torch.sub(w, x, out=g).add_(d).mul_(w)
        torch.add(w, 1.0, out=d)
        g.div_(d)
        w.sub_(g)
    # Far below, W(exp(x)) = exp(x) to the last bit; far above, x - log x
    # (where log(x) / x is below an ulp of x, and the steps above overflow).
    if x.is_cuda or bool((x > large).any()):
        torch.log(x.clamp(min=1.0), out=d)
        torch.sub(x, d, out=d)
        torch.where(x > large, d, w, out=w)
    if x.is_cuda or bool((x < cut).any()):
        torch.exp(x.clamp(max=0.0), out=d)
        torch.where(x < cut, d, w, out=w)
    return torch.where(x == math.inf, x, w, out=w)


def _probability(h: torch.Tensor, alpha: float, log_alpha: float) -> torch.Tensor:
    """W(exp(log alpha + h)) / alpha without an underflowing W."""
    arg = h + log_alpha
    w = _lambert_w_exp(arg)
    cut = -9.0 if h.dtype == torch.float32 else -18.0
    if not h.is_cuda and not bool((arg < cut).any()):
        return w.div_(alpha)
    small = (h - w).exp_()
    w.div_(alpha)
    return torch.where(arg < cut, small, w, out=w)


def _residual(
    t: torch.Tensor,
    alpha: float,
    q: torch.Tensor,
    log_alpha: float,
    log_q: torch.Tensor,
    prob: torch.Tensor | None = None,
) -> torch.Tensor:
    """r in [q - 1, q) with alpha r - log1p(-r / q) = t (entropy_residual)."""
    eps = torch.finfo(t.dtype).eps
    steps = 4 if t.dtype == torch.float32 else 6
    lo = q - 1
    half = alpha * q / 2
    far = (t > half + math.log(2)) | (t < -half - math.log(1.5))
    if prob is None:
        prob = _probability(log_q + (alpha * q - t), alpha, log_alpha)
    outer = torch.minimum(torch.maximum(q - prob, lo), q)
    # Near zero: a Pade start and Halley steps on log1p(-r / q), bracketed.
    s = torch.where(far, 0.0, t)
    inv_q = 1 / q
    linear = s / (alpha + inv_q)
    r = linear / (1 + (linear * inv_q) / (2 * (1 + alpha * q)))
    r = torch.minimum(torch.maximum(r, lo), torch.nextafter(q, torch.zeros_like(q)))
    low, high, tol = lo.expand_as(r), q.expand_as(r), 8 * eps * s.abs()
    for _ in range(steps):
        f = (alpha * r - torch.log1p(-r * inv_q)) - s
        inv_qr = 1 / (q - r)
        d1 = alpha + inv_qr
        denom = d1 - (f * inv_qr * inv_qr) / (2 * d1)
        step = torch.where(denom > 0, r - f / denom, r - f / d1)
        # A step below the resolution of r ends it: a subnormal r misses tol.
        active = (f.abs() > tol) & (step != r)
        low = torch.where(active & (f < 0), r, low)
        high = torch.where(active & (f >= 0), r, high)
        step = torch.where((low < step) & (step < high), step, (low + high) / 2)
        r = torch.where(active, step, r)
    r = torch.where(far, outer, r)
    return torch.where(t == 0, 0.0, torch.where(t <= alpha * lo + log_q, lo, r))


class _Block:
    """The residual sum f(dt) of a batch of entropy blocks and its derivatives."""

    def __init__(
        self, v: torch.Tensor, positive: torch.Tensor, alpha: float, single: bool = False
    ) -> None:
        self.v, self.positive, self.alpha = v, positive, alpha
        self.log_alpha = math.log(alpha)
        self.count = positive.sum(-1, keepdim=True)
        self.q = 1 / self.count.to(v.dtype)
        self.log_q = self.q.log()
        self.clip = alpha * (self.q - 1) + self.log_q
        self.label_max = v.masked_fill(~positive, -math.inf).amax(-1, keepdim=True)
        self.other_max = v.masked_fill(positive, -math.inf).amax(-1, keepdim=True)
        self.ref = torch.maximum(self.label_max, self.other_max)
        self.invalid = (
            (self.count < 1) | (self.count >= v.shape[-1]) | ~v.isfinite().all(-1, keepdim=True)
        )
        if not v.is_cuda:
            # CPU value checks also determine the packed label width.
            if bool(self.invalid.any()):
                raise ValueError(
                    "proximal_cross_entropy needs finite scores, labels and a non-target class"
                )
            width = int(self.count.max().item())
        else:
            width = 1 if single else v.shape[-1]
        self.labels: torch.Tensor | None
        if width == 1:
            self.labels = positive.to(torch.uint8).argmax(-1, keepdim=True)
            self.used = self.count > 0
            self.label_v = self.label_max
        elif width < v.shape[-1]:
            # The labels first, padded with other classes up to the largest label count.
            order = positive.to(torch.uint8).argsort(dim=-1, descending=True, stable=True)
            self.labels = order[:, :width]
            self.used = positive.gather(-1, self.labels)
            self.label_v = v.gather(-1, self.labels)
        else:
            self.labels = None
            self.used = positive
            self.label_v = v

    def evaluate(self, dt: torch.Tensor, derivatives: bool = True) -> tuple[torch.Tensor, ...]:
        """f, f', f'', magnitude, other mass, label residuals and other probabilities."""
        argument = (self.ref - self.label_v) + dt
        h = ((self.v - self.ref) - dt) + self.log_q
        if self.labels is None:
            p_all = _probability(
                torch.where(self.positive, h + self.alpha * self.q, h),
                self.alpha,
                self.log_alpha,
            )
            r_label = _residual(
                argument, self.alpha, self.q, self.log_alpha, self.log_q, prob=p_all
            )
            prob = p_all
        else:
            r_label = _residual(argument, self.alpha, self.q, self.log_alpha, self.log_q)
            prob = _probability(h, self.alpha, self.log_alpha)
        r_label = r_label.masked_fill(~self.used, 0.0)
        single = self.label_v.shape[-1] == 1
        r_sum = r_label if single else r_label.sum(-1, keepdim=True)
        r_abs = r_label.abs() if single else r_label.abs().sum(-1, keepdim=True)
        if not derivatives:
            prob.clamp_(max=1).masked_fill_(self.positive, 0.0)
            mass = prob.sum(-1, keepdim=True)
            magnitude = r_abs + mass
            f = r_sum - mass
            z = torch.zeros_like(f)
            return f, z, z, magnitude, mass, r_label, prob
        counted = (prob <= 1) & ~self.positive
        prob.clamp_(max=1).masked_fill_(self.positive, 0.0)
        mass = prob.sum(-1, keepdim=True)
        magnitude = r_abs + mass
        f = r_sum - mass
        slope = self.q - r_label
        label_denom = 1 + self.alpha * slope
        denom = prob * self.alpha
        denom.add_(1.0)
        label_d1 = torch.where(self.used & (argument >= self.clip), slope / label_denom, 0.0)
        d1 = torch.where(counted, prob / denom, 0.0)
        l_d1_sum = label_d1 if single else label_d1.sum(-1, keepdim=True)
        derivative = l_d1_sum + d1.sum(-1, keepdim=True)
        l_curv = label_d1 / (label_denom * label_denom)
        l_curv_sum = l_curv if single else l_curv.sum(-1, keepdim=True)
        d1.div_(denom).div_(denom)
        curvature = -l_curv_sum - d1.sum(-1, keepdim=True)
        return f, derivative, curvature, magnitude, mass, r_label, prob

    def label_log_probability(self, r_label: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """log p of the labels: log1p where r is small, else log(W / alpha) = h - W."""
        h = self.log_q + (self.alpha * self.q - ((self.ref - self.label_v) + dt))
        near = self.log_q + torch.log1p(-r_label / self.q)
        small = r_label <= self.q / 2
        if not h.is_cuda and bool((small | ~self.used).all()):
            return near.masked_fill(~self.used, 0.0)
        far = h - _lambert_w_exp(h + self.log_alpha)
        return torch.where(small, near, far).masked_fill(~self.used, 0.0)

    def residuals(self, r_label: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
        """The residuals of all classes: r at the labels, -p at the others."""
        if self.labels is None:
            return torch.where(self.positive, r_label, -prob)
        r = -prob
        return r.scatter(
            -1, self.labels, torch.where(self.used, r_label, r.gather(-1, self.labels))
        )

    def step(
        self, f: torch.Tensor, derivative: torch.Tensor, curvature: torch.Tensor, mass: torch.Tensor
    ) -> torch.Tensor:
        """Halley step, or a step in the log of the other mass where it exceeds one."""
        newton = -f / derivative
        correction = (f * curvature) / (2 * derivative * derivative)
        result = torch.where(
            correction.isfinite(), newton / (1 - correction.clamp(-0.5, 0.5)), newton
        )
        others = self.v.shape[-1] - self.count
        log_step = mass.log() + self.alpha * (mass - 1) / others
        jump = (f < 0) & (mass > 1) & log_step.isfinite() & (log_step > result)
        result = torch.where(jump, log_step, result)
        valid = (derivative > 0) & derivative.isfinite() & newton.isfinite()
        return torch.where(valid, result, 0.0)

    def bracket(self) -> tuple[torch.Tensor, torch.Tensor]:
        """dt where the largest probability is one (f <= 0) and every one <= 1/n (f >= 0)."""
        a, label, other = self.alpha, self.label_max - self.ref, self.other_max - self.ref
        lo = torch.maximum(label - a * (1 - self.q), other - a) + self.log_q
        hi = torch.maximum(label + a * self.q, other) + self.log_q + math.log(self.v.shape[-1])
        return lo, torch.where(lo <= hi, hi, lo)

    def start(self) -> torch.Tensor:
        """The softmax-limit threshold; log q for a confident single label."""
        eps = torch.finfo(self.v.dtype).eps
        shifted = self.v - self.ref
        first = shifted.argmax(-1, keepdim=True)
        tail = shifted.scatter(-1, first, -math.inf).exp().sum(-1, keepdim=True)
        tiny = (
            (self.count == 1)
            & (self.label_max == self.ref)
            & ((self.other_max - self.ref) + math.log(self.v.shape[-1]) < math.log(eps) / 2)
        )
        return torch.where(tiny, self.log_q, self.log_q + torch.log1p(tail))


def _solve(block: _Block) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The root dt of every block, its label residuals, other probabilities and failures."""
    eps, tiny = torch.finfo(block.v.dtype).eps, torch.finfo(block.v.dtype).tiny
    lo, hi = block.bracket()
    dt = torch.minimum(torch.maximum(block.start(), lo), hi)
    done = block.invalid.clone()
    # A nonzero step below the resolution of dt: dt is the closest root.
    resolved = torch.zeros_like(done)
    # CUDA needs the same iteration budget: clipped multilabel rows can need
    # more than 20 bisections even though ordinary rows converge in a few steps.
    check_cuda = (
        block.v.is_cuda
        and not torch.compiler.is_compiling()
        and not torch.cuda.is_current_stream_capturing()
    )
    for step in range(200):
        f, derivative, curvature, magnitude, mass, r_label, prob = block.evaluate(dt)
        done = done | (f.abs() <= 8 * eps * magnitude.clamp(min=tiny))
        lo = torch.where(~done & (f < 0), dt, lo)
        hi = torch.where(~done & (f >= 0), dt, hi)
        delta = block.step(f, derivative, curvature, mass)
        following = dt + delta
        resolved = resolved | (~done & (delta != 0) & (following == dt))
        done = done | resolved
        inside = (lo <= following) & (following <= hi) & (following != dt) & following.isfinite()
        following = torch.where(inside, following, lo + (hi - lo) / 2)
        done = done | (following == dt)
        # A completed batch already has its final evaluation; captured CUDA stays on-device.
        check = not block.v.is_cuda or (check_cuda and (step + 1) % 20 == 0)
        if check and bool(done.all()):
            break
        dt = torch.where(done, dt, following)
    else:
        f, _, _, magnitude, _, r_label, prob = block.evaluate(dt, derivatives=False)
    # The resolution of dt and of f is eps relative to at least tiny: they can underflow.
    width = 64 * eps * torch.maximum(lo.abs(), hi.abs()).clamp(min=tiny)
    far = f.abs() > 256 * eps * magnitude.clamp(min=tiny)
    failed = block.invalid | ~f.isfinite() | (~resolved & far & (hi - lo > width))
    if not block.v.is_cuda and bool(failed.any()):
        raise RuntimeError("Entropy normalization did not converge")
    return dt, r_label, prob, failed


def _recover_tiny_tail(
    block: _Block, value: torch.Tensor, a: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover an aggregate tail when the proximal perturbation is below precision."""
    info = torch.finfo(block.v.dtype)
    gap = block.other_max - block.label_max
    tail = gap + math.log(block.v.shape[-1] - 1)
    error = max(0.0, block.log_alpha) + tail
    bound = math.log(float(info.eps) / 8)
    tiny = (
        (block.count == 1)
        & gap.isfinite()
        & (gap < math.log(float(info.tiny)))
        & ((error < bound) | (error + tail < math.log(float(info.tiny)) + bound))
    )
    if not block.v.is_cuda and not bool(tiny.any()):
        return value, a
    # Double intermediates also avoid float32 exponential FTZ on CUDA.
    origin = torch.where(tiny, block.other_max, 0.0).double()
    terms = (block.v.double() - origin).masked_fill(block.positive | ~tiny, -math.inf).exp()
    half = (torch.where(tiny, gap, 0.0).double() / 2).exp()
    mass = half * (half * terms.sum(-1, keepdim=True))
    recovered = torch.where(block.positive, mass, -(half * terms) * half).to(a.dtype)
    return torch.where(tiny[:, 0], mass[:, 0].to(value.dtype), value), torch.where(
        tiny, recovered, a
    )


def proximal_entropy(
    z: torch.Tensor, positive: torch.Tensor, alpha: float, single: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss (examples,) and gradient -a of proximal_entropy at float64 or float32 scores z.

    The loss is min_u H(q, u) + |u - z|^2 / (2 alpha), a = (u - z) / alpha.
    Every row needs a positive label and a negative one.
    """
    if not (math.isfinite(alpha) and alpha > 0):
        raise ValueError("alpha must be finite and positive")
    v = z - z.amax(-1, keepdim=True)
    block = _Block(v, positive, alpha, single=single)
    dt, r_label, prob, failed = _solve(block)
    a = block.residuals(r_label, prob)
    # Zero-sum to one rounding: the label (or the largest entry) balances the
    # others. Labels that are all zero share the recovered mass of the others.
    mass = a.masked_fill(positive, 0.0).sum(-1, keepdim=True)
    labels_zero = (a.masked_fill(~positive, 0.0) == 0).all(-1, keepdim=True)
    share = (block.count > 1) & labels_zero & (mass != 0)
    a = torch.where(share & positive, -mass / block.count, a)
    pivot = torch.where(
        (block.count == 1) | share,
        positive.int().argmax(-1, keepdim=True),
        a.abs().argmax(-1, keepdim=True),
    )
    others = a.scatter(-1, pivot, 0.0)
    a = others.scatter(-1, pivot, -others.sum(-1, keepdim=True))
    # H from the label probabilities (u = v + alpha a cancels), and alpha a_j
    # first in the penalty (a_j^2 can underflow).
    # The corrected one-label mass survives an underflowed root residual.
    r_value = torch.where(block.count == 1, -mass, r_label)
    log_probability = block.label_log_probability(r_value, dt)
    cross = -log_probability.sum(-1) * block.q.squeeze(-1)
    # Several finite label gaps may overflow their sum while their mean fits.
    # Keep ordinary accumulation unchanged, rescaling only exceptional rows.
    scaled_cross = -(log_probability * block.q).sum(-1)
    cross = torch.where(cross.isfinite(), cross, scaled_cross)
    value = cross + ((alpha * a) * a).sum(-1) / 2
    value, a = _recover_tiny_tail(block, value, a)
    return value.masked_fill(failed.squeeze(-1), math.nan), (-a).masked_fill(failed, math.nan)


def proximal_distillation(
    log_p: torch.Tensor,
    log_q: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    logratio: torch.Tensor,
    u0: torch.Tensor,
    alpha: float,
    invalid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Moreau envelope min_u { H(p, softmax(u)) + |u - z_S/T|^2 / (2 alpha) } and residual -r = pi - p.

    Reuses _residual, _probability, and _lambert_w_exp on the cancellation-free log-ratio
    log(q / p) so equal student and teacher logits give an identically zero gradient.
    """
    log_alpha = math.log(alpha)
    eps, tiny = torch.finfo(log_q.dtype).eps, torch.finfo(log_q.dtype).tiny
    lo = -alpha * u0.amax(-1, keepdim=True).clamp(min=0.0)
    hi = -alpha * u0.amin(-1, keepdim=True).clamp(max=0.0)
    hi = torch.where(lo <= hi, hi, lo)
    omega = torch.zeros_like(lo).clamp(lo, hi)
    done = invalid | ((lo == 0) & (hi == 0))
    resolved = torch.zeros_like(done)

    def eval_omega(w: torch.Tensor, derivatives: bool = True) -> tuple[torch.Tensor, ...]:
        arg_res = w - logratio
        h = (log_q + alpha * p) - w
        prob = _probability(h, alpha, log_alpha).clamp(max=1.0)
        half = alpha * p / 2
        near = (p > tiny) & (arg_res <= half + math.log(2)) & (arg_res >= -half - math.log(1.5))
        p_safe = torch.where(near, p, 1.0)
        log_p_safe = torch.where(near, log_p, 0.0)
        r_near = _residual(
            torch.where(near, arg_res, 0.0), alpha, p_safe, log_alpha, log_p_safe, prob=prob
        )
        # A zero argument has a zero residual, also where p is subnormal.
        r = torch.where(near, r_near, (p - prob).masked_fill(arg_res == 0, 0.0))
        pi = torch.where(near, p - r_near, prob)
        f = r.sum(-1, keepdim=True)
        mag = r.abs().sum(-1, keepdim=True)
        if not derivatives:
            z = torch.zeros_like(f)
            return f, z, z, mag, r
        denom = 1.0 + alpha * pi
        d1 = (pi / denom).sum(-1, keepdim=True)
        d2 = -(pi / (denom * denom * denom)).sum(-1, keepdim=True)
        return f, d1, d2, mag, r

    max_steps = 20 if log_q.is_cuda else 100
    for _ in range(max_steps):
        f, d1, d2, mag, r = eval_omega(omega)
        done = done | (f.abs() <= 8 * eps * mag.clamp(min=tiny))
        lo = torch.where(~done & (f < 0), omega, lo)
        hi = torch.where(~done & (f >= 0), omega, hi)
        newton = -f / d1
        corr = (f * d2) / (2 * d1 * d1)
        delta = torch.where(corr.isfinite(), newton / (1 - corr.clamp(-0.5, 0.5)), newton)
        delta = torch.where((d1 > 0) & d1.isfinite() & newton.isfinite(), delta, 0.0)
        following = omega + delta
        resolved = resolved | (~done & (delta != 0) & (following == omega))
        done = done | resolved
        inside = (lo <= following) & (following <= hi) & (following != omega) & following.isfinite()
        following = torch.where(inside, following, lo + (hi - lo) / 2)
        done = done | (following == omega)
        if not log_q.is_cuda and bool(done.all()):
            break
        omega = torch.where(done, omega, following)
    else:
        f, _, _, mag, r = eval_omega(omega, derivatives=False)

    pivot = (p + q).argmax(-1, keepdim=True)
    others = r.scatter(-1, pivot, 0.0)
    r = others.scatter(-1, pivot, -others.sum(-1, keepdim=True))
    h = (log_q + alpha * p) - omega
    far_log_pi = h - _lambert_w_exp(h + log_alpha)
    near_log_pi = log_p + torch.log1p(-r / torch.where(p > 0, p, 1.0))
    # Far above a tiny p, -r / p overflows: log pi = log(p - r) directly.
    above = r < -p / 2
    near_log_pi = torch.where(above, torch.where(above, p - r, 1.0).log(), near_log_pi)
    log_pi = torch.where(r <= p / 2, near_log_pi, far_log_pi)
    cross = -torch.where(p > 0, p * log_pi, 0.0).sum(-1)
    value = cross + ((alpha * r) * r).sum(-1) / 2
    return value.masked_fill(invalid.squeeze(-1), math.nan), (-r).masked_fill(invalid, math.nan)
