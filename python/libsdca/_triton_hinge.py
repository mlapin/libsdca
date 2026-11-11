"""Fused CUDA integer-target hinge losses at k = 1, in float64 as include/sdca/loss.h.

At k = 1, topk_hinge_alpha and topk_hinge_beta are max(0, max_j h_j) with the margins
h_j = 1 + z_j - z_y, and their smooth forms project h onto {a >= 0, sum a <= gamma}.
Rows up to _MAX_BLOCK columns stay in registers; wider ones stream column chunks.
"""

import functools
from typing import Any

import torch
import triton
import triton.language as tl

from . import _cuda_launch, _triton_reduce

_REDUCTIONS = {"none": 0, "mean": 1, "sum": 2}
# Row values, two statistics per row and invalid flags, then the status word (no logs).
_LAYOUT = (0, -1, 3, 4)
_P, _I = _cuda_launch.POINTER, _cuda_launch.INT32
_gammas: dict[tuple[int, float, int | None], torch.Tensor] = {}
# Widest register-resident row. Wider rows pass over chunks of (columns, warps) per
# smoothing: the fastest of an H100 sweep from 4096 to 131072 classes.
_MAX_BLOCK = 1024
_CHUNKS = {False: (1024, 4), True: (1024, 4)}


@triton.jit
def _margin(z, label):
    """1 + z - label in float64; when it cancels, with the low parts of both roundings."""
    d = z - label
    r = d - z
    low = (z - (d - r)) + (-label - r)
    h = d + 1
    r = h - d
    low = low + ((d - (h - r)) + (1 - r))
    cancels = (tl.abs(h) < 0.0625 * tl.abs(d)) & ((low - low) == 0)
    return tl.where(cancels, h + low, h)


@triton.jit
def _margins(Z, Y, row, j, C: tl.constexpr):
    """Compensated margins 1 + z_j - z_y (as loss.h margin), -inf at the target and padding."""
    valid = j < C
    z = tl.load(Z + row * C + j, valid, other=0).to(tl.float64)
    y = tl.load(Y + row)
    in_range = (y >= 0) & (y < C)
    y = tl.minimum(tl.maximum(y, 0), C - 1)
    label = tl.load(Z + row * C + y).to(tl.float64)
    other = valid & (j != y)
    nonfinite = tl.sum((other & ((z - z) != 0)).to(tl.int32), 0)
    bad = (~in_range) | ((label - label) != 0) | (nonfinite > 0)
    return tl.where(other, _margin(z, label), -float("inf")), other, y, bad


@triton.jit
def _margins_rows(Z, Y, row, row_valid, j, C: tl.constexpr):
    """_margins across a [ROWS, B] register tile (B >= C)."""
    valid = row_valid[:, None] & (j < C)
    z = tl.load(Z + row[:, None] * C + j, mask=valid, other=0).to(tl.float64)
    y = tl.load(Y + row, mask=row_valid, other=0)
    in_range = row_valid & (y >= 0) & (y < C)
    y = tl.minimum(tl.maximum(y, 0), C - 1)
    label = tl.load(Z + row * C + y, mask=row_valid, other=0).to(tl.float64)
    other = valid & (j != y[:, None])
    nonfinite = tl.sum((other & ((z - z) != 0)).to(tl.int32), 1)
    bad = row_valid & ((~in_range) | ((label - label) != 0) | (nonfinite > 0))
    return tl.where(other, _margin(z, label[:, None]), -float("inf")), other, y, bad


@triton.jit
def _threshold(h, other, gamma):
    """Origin and correction of a = clip((h - origin) - correction, 0, gamma), sum a <= gamma."""
    box = tl.sum(tl.where(other, tl.minimum(tl.maximum(h, 0.0), gamma), 0.0), 0)
    top = tl.max(h, 0)
    # Michelot from the superset {g > -gamma} of gaps g = h - top: the free sum adds
    # same-signed terms and a tiny gamma keeps its mass at the top entry.
    g = h - top
    inside = other & (g > -gamma)
    count = tl.sum(inside.to(tl.int32), 0)
    s = (tl.sum(tl.where(inside, g, 0.0), 0) - gamma) / count
    # The set {g > level} only shrinks, as in exact arithmetic: a rounded s cannot
    # re-admit a gap and cycle. A scalar level keeps the loop free of a carried mask.
    level = tl.maximum(s, -gamma)
    previous = count + 1
    while count != previous:
        previous = count
        inside = other & (g > level)
        count = tl.sum(inside.to(tl.int32), 0)
        s = (tl.sum(tl.where(inside, g, 0.0), 0) - gamma) / count
        level = tl.maximum(level, s)
    project = box > gamma
    return tl.where(project, top, 0.0), tl.where(project, s, 0.0)


@triton.jit
def _threshold_rows(h, other, valid_row, gamma):
    """_threshold across a [ROWS, B] tile along axis 1."""
    box = tl.sum(tl.where(other, tl.minimum(tl.maximum(h, 0.0), gamma), 0.0), 1)
    project = valid_row & (box > gamma)
    top = tl.where(project, tl.max(h, 1), 0.0)
    g = h - top[:, None]
    inside = other & (g > -gamma)
    count = tl.sum(inside.to(tl.int32), 1)
    s = (tl.sum(tl.where(inside, g, 0.0), 1) - gamma) / tl.maximum(count, 1)
    level = tl.maximum(s, -gamma)
    active = project
    while tl.max(active.to(tl.int32), 0) > 0:
        previous = count
        inside = other & (g > level[:, None])
        new_count = tl.sum(inside.to(tl.int32), 1)
        new_s = (tl.sum(tl.where(inside, g, 0.0), 1) - gamma) / tl.maximum(new_count, 1)
        count = tl.where(active, new_count, count)
        s = tl.where(active, new_s, s)
        level = tl.where(active, tl.maximum(level, new_s), level)
        active = active & (count != previous)
    return top, tl.where(project, s, 0.0)


@triton.jit
def _terms(h, other, a, gamma, AXIS: tl.constexpr = 0):
    """Sum of (h a - a^2 / 2) / gamma over a > 0, as _smooth_value (scaled first off range)."""
    ha = h * a
    aa = a * a / 2
    v = tl.abs(ha - aa)
    normal = (v <= 1.7976931348623157e308) & (v >= 2.2250738585072014e-308 / 2.220446049250313e-16)
    term = tl.where(normal, (ha - aa) / gamma, (h - a / 2) * (a / gamma))
    return tl.sum(tl.where(other & (a > 0), term, 0.0), AXIS)


@triton.jit
def _chunks(Z, Y, row, GAMMA, C: tl.constexpr, B: tl.constexpr, SMOOTH: tl.constexpr):
    """Loss, two statistics and status of a row wider than B, in passes over B columns."""
    j = tl.arange(0, B)
    flags = tl.zeros([], tl.int32)
    top = tl.full([], -float("inf"), tl.float64)
    if SMOOTH:
        gamma = tl.load(GAMMA)
        box = tl.zeros([], tl.float64)
        # Michelot's first superset: the gaps above -gamma to the running top.
        count = tl.zeros([], tl.int32)
        total = tl.zeros([], tl.float64)
        for start in range(0, C, B):
            h, other, y, bad = _margins(Z, Y, row, start + j, C)
            flags = flags | bad.to(tl.int32)
            a = tl.minimum(tl.maximum(h, 0.0), gamma)
            box += tl.sum(tl.where(other, a, 0.0), 0)
            new = tl.maximum(top, tl.max(h, 0))
            # Shift the set's gaps to the new top; a gain of gamma leaves them all <= -gamma.
            shift = new - top
            keep = shift < gamma
            total = tl.where(keep, total - count * shift, 0.0)
            count = tl.where(keep, count, 0)
            g = h - new
            inside = other & (g > -gamma)
            count += tl.sum(inside.to(tl.int32), 0)
            total += tl.sum(tl.where(inside, g, 0.0), 0)
            top = new
        project = (box > gamma) & (flags == 0)
        # tau >= -gamma, so the first pass keeps a subset of the running superset.
        s = tl.maximum((total - gamma) / count, -gamma)
        searching = project
        while searching:
            count = tl.zeros([], tl.int32)
            total = tl.zeros([], tl.float64)
            least = tl.full([], float("inf"), tl.float64)
            for start in range(0, C, B):
                h, other, y, bad = _margins(Z, Y, row, start + j, C)
                g = h - top
                inside = other & (g > s)
                count += tl.sum(inside.to(tl.int32), 0)
                total += tl.sum(tl.where(inside, g, 0.0), 0)
                least = tl.minimum(least, tl.min(tl.where(inside, g, float("inf")), 0))
            s = (total - gamma) / count
            # The set is stable once all its gaps stay above the new threshold.
            searching = least <= s
        first = tl.where(project, top, 0.0)
        second = tl.where(project, s, 0.0)
        value = tl.zeros([], tl.float64)
        for start in range(0, C, B):
            h, other, y, bad = _margins(Z, Y, row, start + j, C)
            a = tl.minimum(tl.maximum((h - first) - second, 0.0), gamma)
            # Skip the float64 divisions of chunks without a positive entry (most, once projected).
            if tl.max((other & (a > 0)).to(tl.int32), 0) > 0:
                value += _terms(h, other, a, gamma)
    else:
        index = tl.zeros([], tl.int32)
        for start in range(0, C, B):
            k = start + j
            h, other, y, bad = _margins(Z, Y, row, k, C)
            flags = flags | bad.to(tl.int32)
            m = tl.max(h, 0)
            # The first largest margin: the first in its chunk, and the earliest chunk.
            index = tl.where(m > top, tl.min(tl.where(h == m, k, C), 0), index)
            top = tl.maximum(top, m)
        value = tl.maximum(top, 0.0)
        first = index.to(tl.float64)
        second = (value > 0).to(tl.float64)
    return value, first, second, flags


@triton.jit
def _forward_rows(
    Z, Y, S, GAMMA, N, C: tl.constexpr, B: tl.constexpr, SMOOTH: tl.constexpr, ROWS: tl.constexpr
):
    """_forward across a [ROWS, B] register tile (B >= C)."""
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    row_valid = row < N
    j = tl.arange(0, B)[None, :]
    h, other, _, bad = _margins_rows(Z, Y, row, row_valid, j, C)
    if SMOOTH:
        gamma = tl.load(GAMMA)
        origin, correction = _threshold_rows(h, other, row_valid & (~bad), gamma)
        a = tl.minimum(tl.maximum((h - origin[:, None]) - correction[:, None], 0.0), gamma)
        value = _terms(h, other, a, gamma, 1)
        first, second = origin, correction
    else:
        top = tl.max(h, 1)
        value = tl.maximum(top, 0.0)
        # The first largest margin per row, as the stable sort of loss.h.
        index = tl.min(tl.where(h == top[:, None], j, B), 1)
        first, second = index.to(tl.float64), (value > 0).to(tl.float64)
    tl.store(S + row, tl.where(bad, float("nan"), value), mask=row_valid)
    tl.store(S + N + row, first, mask=row_valid)
    tl.store(S + 2 * N + row, second, mask=row_valid)
    tl.store(S + 3 * N + row, bad.to(tl.float64), mask=row_valid)


@triton.jit(do_not_specialize=["N"])
def _forward(
    Z,
    Y,
    S,
    GAMMA,
    N,
    C: tl.constexpr,
    B: tl.constexpr,
    SMOOTH: tl.constexpr,
    ROWS: tl.constexpr = 1,
):
    if ROWS > 1:
        _forward_rows(Z, Y, S, GAMMA, N, C, B, SMOOTH, ROWS)
    else:
        row = tl.program_id(0)
        if B < C:
            value, first, second, bad = _chunks(Z, Y, row.to(tl.int64), GAMMA, C, B, SMOOTH)
            tl.store(S + N + row, first)
            tl.store(S + 2 * N + row, second)
        else:
            j = tl.arange(0, B)
            h, other, y, bad = _margins(Z, Y, row.to(tl.int64), j, C)
            if SMOOTH:
                gamma = tl.load(GAMMA)
                origin, correction = _threshold(h, other, gamma)
                a = tl.minimum(tl.maximum((h - origin) - correction, 0.0), gamma)
                value = _terms(h, other, a, gamma)
                tl.store(S + N + row, origin)
                tl.store(S + 2 * N + row, correction)
            else:
                top = tl.max(h, 0)
                value = tl.maximum(top, 0.0)
                # The first largest margin, as the stable sort of loss.h.
                index = tl.min(tl.where(h == top, j, B), 0)
                tl.store(S + N + row, index.to(tl.float64))
                tl.store(S + 2 * N + row, (value > 0).to(tl.float64))
        tl.store(S + row, tl.where(bad != 0, float("nan"), value))
        tl.store(S + 3 * N + row, bad.to(tl.float64))


@triton.jit
def _backward_rows(
    Z, Y, S, U, G, GAMMA, N, C: tl.constexpr, B: tl.constexpr,
    SMOOTH: tl.constexpr, REDUCTION: tl.constexpr, ROWS: tl.constexpr,
):  # fmt: skip
    """_backward across a [ROWS, B] register tile (B >= C)."""
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    row_valid = row < N
    j = tl.arange(0, B)[None, :]
    valid = row_valid[:, None] & (j < C)
    if REDUCTION == 0:
        weight = tl.load(U + row, mask=row_valid, other=0.0).to(tl.float64)
    else:
        weight = tl.load(U).to(tl.float64) + tl.zeros([ROWS], tl.float64)
    if REDUCTION == 1:
        weight = weight / N
    y = tl.minimum(tl.maximum(tl.load(Y + row, mask=row_valid, other=0), 0), C - 1)
    bad = tl.load(S + 3 * N + row, mask=row_valid, other=0.0) != 0
    if SMOOTH:
        gamma = tl.load(GAMMA)
        origin = tl.load(S + N + row, mask=row_valid, other=0.0)[:, None]
        correction = tl.load(S + 2 * N + row, mask=row_valid, other=0.0)[:, None]
        label = tl.load(Z + row * C + y, mask=row_valid, other=0).to(tl.float64)[:, None]
        z = tl.load(Z + row[:, None] * C + j, mask=valid, other=0).to(tl.float64)
        other = valid & (j != y[:, None])
        h = _margin(z, label)
        a = tl.minimum(tl.maximum((h - origin) - correction, 0.0), gamma)
        entry = tl.where(other, (a / gamma) * weight[:, None], 0.0)
        total = tl.sum(entry, 1)
        grad = tl.where(
            bad[:, None], float("nan"), tl.where(j == y[:, None], -total[:, None], entry)
        )
    else:
        index = tl.load(S + N + row, mask=row_valid, other=0.0).to(tl.int64)[:, None]
        w = (weight * tl.load(S + 2 * N + row, mask=row_valid, other=0.0))[:, None]
        grad = tl.where(
            bad[:, None], float("nan"), tl.where(j == index, w, tl.where(j == y[:, None], -w, 0.0))
        )
    tl.store(G + row[:, None] * C + j, grad.to(G.dtype.element_ty), mask=valid)


@triton.jit(do_not_specialize=["N"])
def _backward(
    Z,
    Y,
    S,
    U,
    G,
    GAMMA,
    N,
    C: tl.constexpr,
    B: tl.constexpr,
    SMOOTH: tl.constexpr,
    REDUCTION: tl.constexpr,
    ROWS: tl.constexpr = 1,
):
    if ROWS > 1:
        _backward_rows(Z, Y, S, U, G, GAMMA, N, C, B, SMOOTH, REDUCTION, ROWS)
    else:
        row = tl.program_id(0).to(tl.int64)
        j = tl.arange(0, B)
        if REDUCTION == 0:
            weight = tl.load(U + row).to(tl.float64)
        else:
            weight = tl.load(U).to(tl.float64)
        if REDUCTION == 1:
            weight = weight / N
        y = tl.minimum(tl.maximum(tl.load(Y + row), 0), C - 1)
        bad = tl.load(S + 3 * N + row) != 0
        if SMOOTH:
            gamma = tl.load(GAMMA)
            origin = tl.load(S + N + row)
            correction = tl.load(S + 2 * N + row)
            total = tl.zeros([], tl.float64)
            for start in range(0, C, B):
                k = start + j
                h, other, label, _ = _margins(Z, Y, row, k, C)
                a = tl.minimum(tl.maximum((h - origin) - correction, 0.0), gamma)
                # The unrounded entries, rounded once when stored.
                entry = tl.where(other, (a / gamma) * weight, 0.0)
                total += tl.sum(entry, 0)
                stored = tl.where(bad, float("nan"), entry)
                tl.store(G + row * C + k, stored.to(G.dtype.element_ty), (k < C) & (k != y))
            # The target entry: minus the competitors' unrounded sum, rounded once.
            tl.store(G + row * C + y, tl.where(bad, float("nan"), -total).to(G.dtype.element_ty))
        else:
            index = tl.load(S + N + row).to(tl.int64)
            w = weight * tl.load(S + 2 * N + row)
            for start in range(0, C, B):
                k = start + j
                gradient = tl.where(
                    bad, float("nan"), tl.where(k == index, w, tl.where(k == y, -w, 0.0))
                )
                tl.store(G + row * C + k, gradient.to(G.dtype.element_ty), k < C)


def _gamma(device: torch.device, gamma: float) -> torch.Tensor:
    # torch.full passes gamma to the device as a double, without a host copy.
    stream = torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None
    key = (device.index or 0, gamma, stream)
    if key not in _gammas:
        if len(_gammas) > 64:
            _gammas.clear()
        with torch.inference_mode(False):
            _gammas[key] = torch.full((1,), gamma, device=device, dtype=torch.float64)
    return _gammas[key]


def rows_per_cta(c: int) -> int:
    """Rows per program for short rows (c <= min(128, _MAX_BLOCK)), else 1."""
    block = max(16, triton.next_power_of_2(c))
    if c > 128 or block > _MAX_BLOCK:
        return 1
    return min(16, 512 // block)


def _shape(c: int, smooth: bool) -> tuple[int, int]:
    """Block width and warps of the program of one (or rows_per_cta) row of c columns."""
    rows = rows_per_cta(c)
    if rows > 1:
        block = max(16, triton.next_power_of_2(c))
        return block, 2 if rows * block <= 256 else 4
    block = triton.next_power_of_2(c)
    if block > _MAX_BLOCK:
        return _CHUNKS[smooth]
    return max(16, block), min(32, max(4, block // 1024))


def covers(logits: torch.Tensor) -> bool:
    """Whether one program per row (or packed short-row tile) computes the logits."""
    return logits.is_cuda and logits.shape[0] <= _cuda_launch.ROWS


def forward(
    logits: torch.Tensor, target: torch.Tensor, reduction: str, smooth: bool, gamma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss and row statistics; CUDA invalid values propagate NaNs asynchronously."""
    n, c = logits.shape
    with _cuda_launch.device_guard(logits):
        stats = torch.empty(_triton_reduce.size(n, 4), device=logits.device, dtype=torch.float64)
        # Loss values are at least float32.
        dtype = torch.promote_types(logits.dtype, torch.float32)
        output = torch.empty((n,) if reduction == "none" else (), device=logits.device, dtype=dtype)
        block, warps = _shape(c, smooth)
        rows = rows_per_cta(c)
        g = _gamma(logits.device, gamma)
        _forward[(triton.cdiv(n, rows),)](
            logits, target, stats, g, n, c, block, smooth, rows, num_warps=warps
        )
        _triton_reduce.launch(stats, output, n, _REDUCTIONS[reduction], _LAYOUT)
        if not logits.is_cuda and stats[4 * n].item():
            raise ValueError("logits must be finite and target indices in range")
        return output, stats


def backward(
    logits: torch.Tensor,
    target: torch.Tensor,
    stats: torch.Tensor,
    upstream: torch.Tensor,
    reduction: str,
    smooth: bool,
    gamma: float,
) -> torch.Tensor:
    """Gradient in the logits dtype, scaled by the upstream derivative in float64."""
    n, c = logits.shape
    with _cuda_launch.device_guard(logits):
        gradient = torch.empty_like(logits)
        block, warps = _shape(c, smooth)
        rows = rows_per_cta(c)
        _backward[(triton.cdiv(n, rows),)](
            logits,
            target,
            stats,
            upstream.contiguous(),
            gradient,
            _gamma(logits.device, gamma),
            n,
            c,
            block,
            smooth,
            _REDUCTIONS[reduction],
            rows,
            num_warps=warps,
        )
        return gradient


@functools.lru_cache(maxsize=_cuda_launch.CACHE)
def _compiled(
    device: int, dtype: torch.dtype, c: int, code: int, smooth: bool
) -> torch.Tensor | str:
    """C++ launch handles for (device, dtype, c, reduction, smooth), or why not."""
    with torch.cuda.device(device):
        z = torch.empty((1, 1), device="cuda", dtype=dtype)
        y = torch.empty(1, device="cuda", dtype=torch.int64)
        s = torch.empty(1, device="cuda", dtype=torch.float64)
        g = torch.empty(1, device="cuda", dtype=torch.float64)
        out = torch.empty(1, device="cuda", dtype=torch.promote_types(dtype, torch.float32))
        block, warps = _shape(c, smooth)
        rows = rows_per_cta(c)
        # Any int32 N: the kernels do not specialize on it. The C++ launcher passes a
        # float64 upstream derivative.
        k_fwd = _forward.warmup(z, y, s, g, 2, c, block, smooth, rows, num_warps=warps, grid=(1,))
        k_bwd = _backward.warmup(
            z, y, s, g, z, g, 2, c, block, smooth, code, rows, num_warps=warps, grid=(1,)
        )
        reduce = _triton_reduce.warmup(s, out, code, _LAYOUT)
        return _cuda_launch.handles(
            [(k_fwd, (_P, _P, _P, _P, _I))]
            + [(k, (_P, _P, _I)) for k in reduce]
            + [(k_bwd, (_P, _P, _P, _P, _P, _P, _I))],
            _triton_reduce.BLOCK,
            _triton_reduce.DIRECT_ROWS,
            0,
            rows,
            rows,
        )


def launcher(logits: torch.Tensor, code: int, smooth: bool) -> tuple[Any, torch.Tensor | str]:
    """(extension, kernel handles) for the C++ launch, else (None, why not)."""
    host, reason = _cuda_launch.status()
    if host is None:
        return None, reason
    device = logits.device.index or 0
    handles = _compiled(device, logits.dtype, logits.shape[1], code, smooth)
    return (None, handles) if isinstance(handles, str) else (host, handles)


def apply(
    logits: torch.Tensor, target: torch.Tensor, reduction: str, smooth: bool, gamma: float
) -> torch.Tensor | None:
    """C++ autograd and launches when the host launcher runs these kernels, else None."""
    code = _REDUCTIONS[reduction]
    host, handles = launcher(logits, code, smooth)
    if host is None:
        return None
    return host.hinge_cuda_apply(logits, target, _gamma(logits.device, gamma), code, handles)
