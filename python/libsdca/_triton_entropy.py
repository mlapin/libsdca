"""Fused CUDA integer-target entropy: double statistics, log-domain backward.

Rows up to _MAX_BLOCK columns stay in registers; wider ones stream column chunks
(_CHUNKS) with a running maximum. FP32 takes the elementwise exponentials and chunk lanes
in float32 and keeps the row statistics, losses and per-row factors in float64.
"""

import functools
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from . import _cuda_launch, _triton_reduce

_REDUCTIONS = {"none": 0, "mean": 1, "sum": 2}
# Row statistics: 3 per row (maximum and _row_factors), then values, logs and invalid
# flags (VALUE, LOG, BAD), the status word (STATUS) and the reduction partials; backward
# reads the first 3 * N.
_LAYOUT = (3, 4, 5, 6)
_FLT_MAX = tl.constexpr(3.4028234663852886e38)
_P, _I = _cuda_launch.POINTER, _cuda_launch.INT32
# Widest register-resident forward row; wider rows pass over chunks of (columns, warps)
# by FP32, tuned on H100 at 32000-128256 columns.
_MAX_BLOCK = 4096
_CHUNKS = {True: (2048, 4), False: (1024, 4)}
# Backward blocks per program of such FP32 rows: 64 columns per thread amortize each
# program's float64 row factors (A100, 8192-128256 columns).
_BWD_TILES = 4
# Widest rows: the backward grid has at most 65535 column blocks.
CLASSES = 2**25


@triton.jit
def _work(x, FP32: tl.constexpr):
    """x in the elementwise working type."""
    return x.to(tl.float32 if FP32 else tl.float64)


@triton.jit
def _exp32(x):
    """exp of float32 x <= 0: exp2 of the rounded t = x log2(e) times 1 + d, for the residual
    d = x - t ln(2) of a two-part ln(2). |d| <= 2**-23 |x|, so the omitted d**2 / 2 is far
    below float32 rounding. Below t = -150 (x = -inf included) it is 0; NaN stays NaN."""
    t = x * 1.4426950408889634
    r = tl.exp2(t)
    d = tl.fma(t, -0.693145751953125, x)
    d = tl.fma(t, -1.4286067653301870e-06, d)
    return tl.where(t < -150, 0.0, tl.fma(r, d, r))


@triton.jit
def _register(Z, row, y, C: tl.constexpr, B: tl.constexpr, FP32: tl.constexpr):
    """Largest competitor (nonfinite ones as +inf) and the sum of exp(z_j - maximum)."""
    j = tl.arange(0, B)
    valid = j < C
    z = _work(tl.load(Z + row * C + j, valid, other=0), FP32)
    other = valid & (j != y)
    z_checked = tl.where((z - z) == 0, z, float("inf"))
    maximum = tl.max(tl.where(other, z_checked, -float("inf")), 0)
    e = libdevice.exp(tl.where(other, z - maximum, -float("inf")))
    return maximum.to(tl.float64), tl.sum(e.to(tl.float64), 0)


@triton.jit
def _register_rows(Z, row, row_valid, y, C: tl.constexpr, B: tl.constexpr, FP32: tl.constexpr):
    """_register across a 2D tile of [ROWS, B] logits."""
    j = tl.arange(0, B)[None, :]
    valid = row_valid[:, None] & (j < C)
    z = _work(tl.load(Z + row[:, None] * C + j, valid, other=0), FP32)
    other = valid & (j != y[:, None])
    z_checked = tl.where((z - z) == 0, z, float("inf"))
    maximum = tl.max(tl.where(other, z_checked, -float("inf")), 1)
    e = libdevice.exp(tl.where(other, z - maximum[:, None], -float("inf")))
    return maximum.to(tl.float64), tl.sum(e.to(tl.float64), 1)


@triton.jit
def _entropy_values(maximum, label, sum_e):
    """Loss value and log(value) from maximum, label and sum_e (scalar or vector)."""
    log_sum = libdevice.log(sum_e)
    relative = (maximum - label) + log_sum
    half_tail = libdevice.exp(tl.minimum(maximum - label, 0.0) * 0.5)
    small = half_tail * (half_tail * sum_e)
    clamped = tl.minimum(tl.maximum(relative, -40.0), 40.0)
    mid = libdevice.log1p(libdevice.exp(clamped))
    value = tl.where(relative < -36, small, tl.where(relative > 36, relative, mid))
    inf_log = libdevice.log(maximum * 0.5 - label * 0.5) + 0.6931471805599453
    log_value = tl.where(
        relative < -36,
        relative,
        tl.where(relative == float("inf"), inf_log, libdevice.log(value)),
    )
    return log_sum, relative, value, log_value


@triton.jit
def _chunk(base, k, label, acc, top, C: tl.constexpr, FP32: tl.constexpr, MASKED: tl.constexpr):
    """_chunks' lanes and maximum after columns k; MASKED skips the label and columns
    from C, which only the label's chunk and the last one hold."""
    keep = (k < C) & (k != label)
    if MASKED:
        z = _work(tl.load(base + k, k < C, other=0), FP32)
        new = tl.maximum(top, tl.max(tl.where(keep, z, -float("inf")), 0))
    else:
        z = _work(tl.load(base + k), FP32)
        new = tl.maximum(top, tl.max(z, 0))
    if new > top:
        acc = acc * libdevice.exp(top - new)
        top = new
    # z * 0 is NaN exactly for nonfinite z; otherwise e + z * 0 is e.
    if MASKED:
        e = libdevice.exp(tl.where(keep, z - top, -float("inf")))
    elif FP32:
        e = _exp32(z - top)
    else:
        e = libdevice.exp(z - top)
    return acc + (e + z * 0), top


@triton.jit
def _chunks(Z, row, y, C: tl.constexpr, B: tl.constexpr, FP32: tl.constexpr):
    """As _register over chunks of B columns; lanes rescale when the maximum rises.

    Lanes take the working type (each sums at most C / B terms of at most one); their
    final sum is float64. A nonfinite score turns its lane NaN, reported as maximum +inf.
    """
    j = tl.arange(0, B)
    base = Z + row * C
    label = y.to(tl.int32)  # int32 column offsets: C <= CLASSES
    FULL: tl.constexpr = C // B * B
    split = label // B * B
    top = _work(tl.full([], -float("inf"), tl.float64), FP32)
    acc = _work(tl.zeros([B], tl.float64), FP32)
    for start in range(0, split, B):
        acc, top = _chunk(base, start + j, label, acc, top, C, FP32, False)
    if split < FULL:
        acc, top = _chunk(base, split + j, label, acc, top, C, FP32, True)
    for start in range(split + B, FULL, B):
        acc, top = _chunk(base, start + j, label, acc, top, C, FP32, False)
    if FULL < C:
        acc, top = _chunk(base, FULL + j, label, acc, top, C, FP32, True)
    total = tl.sum(acc.to(tl.float64), 0)
    return tl.where(total == total, top.to(tl.float64), float("inf")), total


@triton.jit
def _row_factors(maximum, sum_e, log_sum, label, relative, FULL64: tl.constexpr):
    """Backward's row factors after maximum: float64 storage keeps (log_sum, label); other
    storage (gradient exp(z - maximum) * p, target -mass) keeps (p = mass / sum_e, mass)."""
    if FULL64:
        first, second = log_sum, label
    else:
        second = 1.0 / (1.0 + libdevice.exp(-relative))
        first = second / sum_e
    return first, second


@triton.jit
def _store_rows(S, row, bad, maximum, first, second, R: tl.constexpr, mask=None):
    """Rows of R statistics start with maximum, first, second (NaN for invalid rows)."""
    nan = float("nan")
    tl.store(S + row * R, tl.where(bad, nan, maximum), mask)
    tl.store(S + row * R + 1, tl.where(bad, nan, first), mask)
    tl.store(S + row * R + 2, tl.where(bad, nan, second), mask)


@triton.jit(do_not_specialize=["N"])
def _forward(
    Z, Y, S, N, C: tl.constexpr, B: tl.constexpr, FP32: tl.constexpr, ROWS: tl.constexpr = 1
):
    if ROWS > 1:
        row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        row_valid = row < N
        y = tl.load(Y + row, row_valid, other=0)
        in_range = (y >= 0) & (y < C)
        y = tl.minimum(tl.maximum(y, 0), C - 1)
        label = tl.load(Z + row * C + y, row_valid, other=0).to(tl.float64)
        maximum, sum_e = _register_rows(Z, row, row_valid, y, C, B, FP32)
        mask = row_valid
    else:
        row = tl.program_id(0).to(tl.int64)
        y = tl.load(Y + row)
        in_range = (y >= 0) & (y < C)
        y = tl.minimum(tl.maximum(y, 0), C - 1)
        label = tl.load(Z + row * C + y).to(tl.float64)
        if B < C:
            maximum, sum_e = _chunks(Z, row, y, C, B, FP32)
        else:
            maximum, sum_e = _register(Z, row, y, C, B, FP32)
        mask = None
    bad = (~in_range) | ((label - label) != 0) | (maximum == float("inf"))
    log_sum, relative, value, log_value = _entropy_values(maximum, label, sum_e)
    first, second = _row_factors(
        maximum, sum_e, log_sum, label, relative, Z.dtype.element_ty == tl.float64
    )
    _store_rows(S, row, bad, maximum, first, second, 3, mask)
    nan = float("nan")
    tl.store(S + 3 * N + row, tl.where(bad, nan, value), mask)
    tl.store(S + 4 * N + row, tl.where(bad, nan, log_value), mask)
    tl.store(S + 5 * N + row, bad.to(tl.float64), mask)


@triton.jit
def _rows(
    z,
    maximum,
    first,
    second,
    col,
    y,
    sign,
    scale,
    WEIGHTED: tl.constexpr,
    FP32: tl.constexpr,
    VEC: tl.constexpr = False,
):
    """Gradient block of one row times sign * scale, rounded once into the logits dtype.

    scale is |upstream| * weight / divisor; float64 storage takes its square root, or its
    log when WEIGHTED.
    """
    out = z.dtype
    if z.dtype != tl.float64:
        # Bounded storage: scale stays below 2e77.
        factor = first * scale
        target = (-(second * scale) * sign).to(out)
        if FP32:
            half = libdevice.exp((z.to(tl.float32) - maximum.to(tl.float32)) * 0.5)
            half_factor = tl.where(factor > _FLT_MAX, _FLT_MAX, factor)
            half = half * libdevice.sqrt(half_factor).to(tl.float32)
            competitors = (half * half * sign.to(tl.float32)).to(out)
            any_wide = (
                (tl.max(tl.max((factor > _FLT_MAX).to(tl.int32), 1), 0) > 0)
                if VEC
                else (factor > _FLT_MAX)
            )
            if any_wide:
                wide = (libdevice.exp(z.to(tl.float64) - maximum) * factor * sign).to(out)
                competitors = tl.where(factor > _FLT_MAX, wide, competitors)
        else:
            competitors = (libdevice.exp(z.to(tl.float64) - maximum) * factor * sign).to(out)
        gradient = tl.where(col == y, target, competitors)
    else:
        log_sum, label = first, second
        relative = (maximum - label) + log_sum
        far = relative < -37
        clamped = tl.maximum(relative, -37.0)
        if WEIGHTED:
            log_mass = -libdevice.log1p(libdevice.exp(-clamped))
            comp_log = tl.where(far, z - label, (z - maximum) - log_sum + log_mass)
            log_probability = tl.where(col == y, tl.where(far, relative, log_mass), comp_log)
            magnitude = _triton_reduce._exp_full(log_probability + scale)
            gradient = tl.where(col == y, -magnitude, magnitude)
        else:
            mass = 1.0 / (1.0 + libdevice.exp(-clamped))
            factor = scale * libdevice.sqrt(mass / libdevice.exp(log_sum))
            half = tl.where(
                far,
                libdevice.exp((z - label) * 0.5) * scale,
                libdevice.exp((z - maximum) * 0.5) * factor,
            )
            half_mass = libdevice.exp(tl.minimum(maximum - label, 0.0) * 0.5) * scale
            target_mass = tl.where(
                far,
                half_mass * (half_mass * libdevice.exp(log_sum)),
                (scale * mass) * scale,
            )
            gradient = tl.where(col == y, -target_mass, half * half)
        gradient = gradient * sign
    return gradient


@triton.jit
def _store_gradient(
    G, row, col, valid, z, y, maximum, first, second, sign, scale, upstream, infinite,
    C: tl.constexpr, WEIGHTED: tl.constexpr, FP32: tl.constexpr, VEC: tl.constexpr = False,
):  # fmt: skip
    """_backward's gradient of the loaded columns col (where valid) of a row."""
    gradient = _rows(z, maximum, first, second, col, y, sign, scale, WEIGHTED, FP32, VEC)
    if z.dtype == tl.float64:
        gradient = tl.where(infinite, tl.where(col == y, -upstream, upstream), gradient)
    tl.store(G + row * C + col, gradient, valid)


@triton.jit
def _row_scale(
    S, U, row, raw, N, IGNORE: tl.constexpr, REDUCTION: tl.constexpr,
    WEIGHTED: tl.constexpr, NAMED: tl.constexpr, FULL64: tl.constexpr, mask=None,
):  # fmt: skip
    """Per-row upstream, sign, scale and infinite-upstream flag (scalar or [ROWS] vector)."""
    R: tl.constexpr = 4 if NAMED else 3
    maximum = tl.load(S + row * R, mask)
    first = tl.load(S + row * R + 1, mask)
    second = tl.load(S + row * R + 2, mask)
    upstream = (
        tl.load(U + row, mask).to(tl.float64) if REDUCTION == 0 else tl.load(U).to(tl.float64)
    )
    sign = tl.where(upstream < 0, -1.0, 1.0)
    magnitude = tl.abs(upstream)
    if NAMED:
        weight = tl.load(S + row * R + 3, mask)
        flag = tl.load(S + 5 * N)
        divisor = tl.load(S + 5 * N + 1) if REDUCTION == 1 else 1.0
        ignored = raw == IGNORE
        undefined = (flag == 3) | (weight < 0) | ((REDUCTION != 0) & (flag > 0))
        undefined = undefined | (~ignored & (REDUCTION == 1) & (divisor == 0))
        zero = ~undefined & (ignored | (weight == 0))
        if WEIGHTED and FULL64:
            scale = libdevice.log(weight) + libdevice.log(magnitude)
            if REDUCTION == 1:
                scale -= tl.load(S + 6 * N + 2)
            empty = -float("inf")
        elif FULL64:
            scale = libdevice.sqrt(magnitude) / libdevice.sqrt(divisor)
            empty = 0.0
        else:
            scale = (weight / divisor) * magnitude
            empty = 0.0
        scale = tl.where(undefined, float("nan"), tl.where(zero, empty, scale))
        infinite = (magnitude == float("inf")) & ~undefined & ~zero
        if not FULL64:
            maximum = tl.where(infinite, -float("inf"), maximum)
            first = tl.where(infinite, 1.0, first)
            second = tl.where(infinite, 1.0, second)
    else:
        if FULL64:
            scale = libdevice.sqrt(magnitude)
            if REDUCTION == 1:
                scale = scale / libdevice.sqrt(N.to(tl.float64))
        else:
            scale = magnitude / N.to(tl.float64) if REDUCTION == 1 else magnitude
        infinite = maximum < -float("inf")
    return maximum, first, second, upstream, sign, scale, infinite


@triton.jit(do_not_specialize=["N"])
def _backward(
    Z,
    Y,
    S,
    U,
    G,
    N,
    C: tl.constexpr,
    IGNORE: tl.constexpr,
    REDUCTION: tl.constexpr,
    B: tl.constexpr,
    WEIGHTED: tl.constexpr,
    NAMED: tl.constexpr,
    FP32: tl.constexpr,
    T: tl.constexpr,
    ROWS: tl.constexpr = 1,
):
    """Entropy gradient from the row statistics, B * T columns (or ROWS x B) per program."""
    W: tl.constexpr = WEIGHTED and NAMED
    FULL64: tl.constexpr = Z.dtype.element_ty == tl.float64
    if ROWS > 1:
        row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        row_valid = row < N
        col = tl.arange(0, B)[None, :]
        valid = row_valid[:, None] & (col < C)
        raw = tl.load(Y + row, row_valid, other=0)
        y = tl.minimum(tl.maximum(raw, 0), C - 1)
        z = tl.load(Z + row[:, None] * C + col, valid, other=0)
        maximum, first, second, upstream, sign, scale, infinite = _row_scale(
            S, U, row, raw, N, IGNORE, REDUCTION, WEIGHTED, NAMED, FULL64, row_valid
        )
        if REDUCTION != 0:
            upstream = upstream + tl.zeros([ROWS], tl.float64)
            sign = sign + tl.zeros([ROWS], tl.float64)
            scale = scale + tl.zeros([ROWS], tl.float64)
        _store_gradient(
            G, row[:, None], col, valid, z, y[:, None], maximum[:, None], first[:, None],
            second[:, None], sign[:, None], scale[:, None], upstream[:, None],
            infinite[:, None], C, W, FP32, True,
        )  # fmt: skip
    else:
        row = tl.program_id(0).to(tl.int64)
        start = tl.program_id(1) * (B * T)
        col = start + tl.arange(0, B)
        valid = col < C
        raw = tl.load(Y + row)
        y = tl.minimum(tl.maximum(raw, 0), C - 1)
        z = tl.load(Z + row * C + col, valid, other=0)
        maximum, first, second, upstream, sign, scale, infinite = _row_scale(
            S, U, row, raw, N, IGNORE, REDUCTION, WEIGHTED, NAMED, FULL64
        )
        _store_gradient(
            G, row, col, valid, z, y, maximum, first, second, sign, scale, upstream, infinite,
            C, W, FP32,
        )  # fmt: skip
        if T > 1:
            # Each block is read before it is written (in place: G is Z).
            if start + B * T <= C:
                for t in tl.static_range(1, T):
                    k = col + t * B
                    _store_gradient(
                        G, row, k, None, tl.load(Z + row * C + k), y, maximum, first, second,
                        sign, scale, upstream, infinite, C, W, FP32,
                    )  # fmt: skip
            else:
                for t in tl.static_range(1, T):
                    k = col + t * B
                    if start + t * B < C:
                        _store_gradient(
                            G, row, k, k < C, tl.load(Z + row * C + k, k < C, other=0), y,
                            maximum, first, second, sign, scale, upstream, infinite, C, W, FP32,
                        )  # fmt: skip


def _host():
    """The C++ launcher extension, or None."""
    return _cuda_launch.host()


def _warps(block: int) -> int:
    return min(32, max(1, block // 512))


def rows_per_cta(c: int) -> int:
    """Rows per program for short rows (c <= min(64, _MAX_BLOCK)), else 1."""
    block = max(16, triton.next_power_of_2(c))
    if c > 64 or block > _MAX_BLOCK:
        return 1
    return min(32, 512 // block)


def _shape(c: int, fp32: bool) -> tuple[int, int]:
    """Forward block width and warps of the program of one (or rows_per_cta) row of c columns."""
    block = max(16, triton.next_power_of_2(c))
    if block > _MAX_BLOCK:
        return _CHUNKS[fp32]
    rows = rows_per_cta(c)
    return block, max(2, _warps(block * rows)) if rows > 1 else _warps(block)


def _bwd_shape(c: int, fp32: bool) -> tuple[int, int, int]:
    """Backward column block, warps and blocks per program of rows of c columns."""
    rows = rows_per_cta(c)
    if rows > 1:
        block = max(16, triton.next_power_of_2(c))
        return block, max(2, _warps(block * rows)), 1
    block = min(1024, max(64, triton.next_power_of_2(c)))
    return block, _warps(block), _BWD_TILES if fp32 and c > _MAX_BLOCK else 1


def bwd_columns(c: int, fp32: bool) -> int:
    """Columns per backward program (its grid has cdiv(c, bwd_columns) of them per row)."""
    block, _, tiles = _bwd_shape(c, fp32)
    return block * tiles


def warmup_backward(z, y, s, u, c, code, fp32, ignore=0, weighted=False, named=False):
    """The compiled backward kernel of (dtype, c, reduction, options)."""
    block, warps, tiles = _bwd_shape(c, fp32)
    rows = rows_per_cta(c)
    return _backward.warmup(
        z,
        y,
        s,
        u,
        z,
        2,
        c,
        ignore,
        code,
        block,
        weighted,
        named,
        fp32,
        tiles,
        rows,
        num_warps=warps,
        grid=(1,),
    )


def launch_backward(
    logits, target, stats, upstream, code, fp32, ignore=0, weighted=False, named=False, out=None
):
    """Gradient through Triton's launcher; the C++ launcher mirrors this (cuda_backward). out,
    if given (logits itself included), receives it: each program reads each of its blocks
    before writing it."""
    n, c = logits.shape
    gradient = torch.empty_like(logits) if out is None else out
    block, warps, tiles = _bwd_shape(c, fp32)
    rows = rows_per_cta(c)
    _backward[(triton.cdiv(n, rows), triton.cdiv(c, block * tiles))](
        logits,
        target,
        stats,
        upstream.contiguous(),
        gradient,
        n,
        c,
        ignore,
        code,
        block,
        weighted,
        named,
        fp32,
        tiles,
        rows,
        num_warps=warps,
    )
    return gradient


def describe(c: int, fp32: bool) -> str:
    """The arithmetic and row layout of the forward kernel for c classes."""
    block = _shape(c, fp32)[0]
    layout = f"{block}-column chunks" if block < c else "rows in registers"
    return f"{'float32 exponentials' if fp32 else 'float64'}, {layout}"


@functools.lru_cache(maxsize=_cuda_launch.CACHE)
def _compiled(device: int, dtype: torch.dtype, c: int, code: int, fp32: bool) -> torch.Tensor | str:
    """C++ launch handles of the kernels for (device, dtype, c, reduction, fp32), or why not."""
    with torch.cuda.device(device):
        z = torch.empty((1, 1), device="cuda", dtype=dtype)
        y = torch.empty(1, device="cuda", dtype=torch.int64)
        s = torch.empty(1, device="cuda", dtype=torch.float64)
        out_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        out = torch.empty(1, device="cuda", dtype=out_dtype)
        (block, warps), b_bwd = _shape(c, fp32), bwd_columns(c, fp32)
        rows = rows_per_cta(c)
        # Any int32 N: the kernels do not specialize on it.
        k_fwd = _forward.warmup(z, y, s, 2, c, block, fp32, rows, num_warps=warps, grid=(1,))
        k_bwd = warmup_backward(z, y, s, out, c, code, fp32)
        reduce = _triton_reduce.warmup(s, out, code, _LAYOUT)
        return _cuda_launch.handles(
            [(k_fwd, (_P, _P, _P, _I))]
            + [(k, (_P, _P, _I)) for k in reduce]
            + [(k_bwd, (_P, _P, _P, _P, _P, _I))],
            _triton_reduce.BLOCK,
            _triton_reduce.DIRECT_ROWS,
            b_bwd,
            rows,
            rows,
        )


def launcher(logits: torch.Tensor, code: int, fp32: bool = False) -> tuple[Any, torch.Tensor | str]:
    """(extension, kernel handles) for the C++ launch, else (None, why not)."""
    host, reason = _cuda_launch.status()
    if host is None:
        return None, reason
    handles = _compiled(logits.get_device(), logits.dtype, logits.shape[1], code, fp32)
    return (None, handles) if isinstance(handles, str) else (host, handles)


def forward(
    logits: torch.Tensor, target: torch.Tensor, reduction: str, fp32: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate integer-target entropy in the fused pass and return loss and row statistics."""
    n, c = logits.shape
    code = _REDUCTIONS[reduction]
    host, handles = launcher(logits, code, fp32)
    if host is not None:
        return host.cuda_forward(logits, target, code, handles)
    with _cuda_launch.device_guard(logits):
        # One allocation holds saved row statistics and reduction scratch.
        stats = torch.empty(_triton_reduce.size(n, 6), device=logits.device, dtype=torch.float64)
        dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
        output = torch.empty((n,) if reduction == "none" else (), device=logits.device, dtype=dtype)
        block, warps = _shape(c, fp32)
        rows = rows_per_cta(c)
        _forward[(triton.cdiv(n, rows),)](
            logits, target, stats, n, c, block, fp32, rows, num_warps=warps
        )
        _triton_reduce.launch(stats, output, n, code, _LAYOUT)
        return output, stats


def backward(
    logits: torch.Tensor,
    target: torch.Tensor,
    stats: torch.Tensor,
    upstream: torch.Tensor,
    reduction: str,
    fp32: bool = False,
) -> torch.Tensor:
    """Scale by the upstream derivative before rounding the gradient into the input dtype."""
    code = _REDUCTIONS[reduction]
    host, handles = launcher(logits, code, fp32)
    if host is not None:
        return host.cuda_backward(logits, target, stats, upstream, code, handles)
    with _cuda_launch.device_guard(logits):
        return launch_backward(logits, target, stats, upstream, code, fp32)


def apply(
    logits: torch.Tensor, target: torch.Tensor, reduction: str, fp32: bool = False
) -> torch.Tensor | None:
    """Run fused CUDA entropy through C++ autograd when the host launcher is available."""
    code = _REDUCTIONS[reduction]
    host, handles = launcher(logits, code, fp32)
    return None if host is None else host.cuda_apply(logits, target, code, handles)
