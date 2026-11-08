"""CUDA entropy with ignored labels and optional class weights.

The bounded path accepts float16/bfloat16/float32 storage. With at
most 2**28 rows, weighted sums are below 1e87 and weight/upstream products
below 2e77. A double-underflowed term cannot be rescued into a float32
result: even multiplying 1e-308 by 2e77 and all 2**53 allowed entries is
below 1e-214, versus float32's smallest subnormal of about 1e-45. Mean
weight factors are at most one. Rows up to 4096 columns stay in registers;
wider rows stream column chunks up to CLASSES.

FP32 selects float32 exponentials for the bounded forward and, with half-width
storage, the backward; statistics and weighted reductions use float64. Float64
storage uses full-range exponential and log-domain reductions, including arbitrary finite nonnegative class weights.
One extra scalar stores the log denominator when its ordinary sum overflows.
"""

import functools

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from . import _cuda_launch, _triton_entropy, _triton_reduce

CLASSES = _triton_entropy.CLASSES
BLOCK = 1024
DIRECT_ROWS = 4096
_CODES = {"none": 0, "mean": 1, "sum": 2}


@triton.jit(do_not_specialize=["N"])
def _forward(
    Z,
    Y,
    W,
    S,
    N,
    C: tl.constexpr,
    IGNORE: tl.constexpr,
    WEIGHTED: tl.constexpr,
    B: tl.constexpr,
    FP32: tl.constexpr,
    ROWS: tl.constexpr = 1,
):
    if ROWS > 1:
        row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        row_valid = row < N
        raw = tl.load(Y + row, row_valid, other=IGNORE)
        ignored = raw == IGNORE
        in_range = (raw >= 0) & (raw < C)
        y = tl.minimum(tl.maximum(raw, 0), C - 1)
        label = tl.load(Z + row * C + y, row_valid, other=0).to(tl.float64)
        maximum, total = _triton_entropy._register_rows(Z, row, row_valid, y, C, B, FP32)
        mask = row_valid
    else:
        row = tl.program_id(0).to(tl.int64)
        raw = tl.load(Y + row)
        ignored = raw == IGNORE
        in_range = (raw >= 0) & (raw < C)
        y = tl.minimum(tl.maximum(raw, 0), C - 1)
        label = tl.load(Z + row * C + y).to(tl.float64)
        if B < C:
            maximum, total = _triton_entropy._chunks(Z, row, y, C, B, FP32)
        else:
            maximum, total = _triton_entropy._register(Z, row, y, C, B, FP32)
        mask = None
    # Match public validation priority: weight values, logits, then indices.
    bad = tl.where(~in_range & ~ignored, 1, 0)
    bad = tl.where(((label - label) != 0) | (maximum == float("inf")), 2, bad)
    log_sum = libdevice.log(total)
    relative = (maximum - label) + log_sum
    clamped = tl.minimum(tl.maximum(relative, -40.0), 40.0)
    mid = libdevice.log1p(libdevice.exp(clamped))
    if Z.dtype.element_ty == tl.float64:
        half = libdevice.exp(tl.minimum(maximum - label, 0.0) * 0.5)
        small = half * (half * total)
    else:
        small = libdevice.exp(tl.minimum(maximum - label, 0.0)) * total
    value = tl.where(relative < -36, small, tl.where(relative > 36, relative, mid))
    weight = tl.zeros_like(value) + 1.0
    if WEIGHTED:
        weight = tl.load(W + y, mask).to(tl.float64)
        if tl.program_id(0) == 0:
            j = tl.arange(0, B)
            inv = tl.zeros([], tl.int32)
            for start in range(0, C, B):
                k = start + j
                weights = tl.load(W + k, k < C, other=0.0).to(tl.float64)
                invalid = ((weights - weights) != 0) | (weights < 0)
                inv = inv | tl.max(invalid.to(tl.int32), 0)
            bad = tl.where((inv != 0) & (row == 0), 3, bad)
    weight = tl.where(ignored | ~(weight > 0), 0.0, weight)
    nan = float("nan")
    is_bad = bad > 0
    first, second = _triton_entropy._row_factors(
        maximum, total, log_sum, label, relative, Z.dtype.element_ty == tl.float64
    )
    _triton_entropy._store_rows(S, row, is_bad, maximum, first, second, 4, mask)
    # Each row's fourth statistic: its weight, or its validation status as a negative weight.
    tl.store(S + row * 4 + 3, tl.where(is_bad, -bad.to(tl.float64), weight), mask)
    weighted_value = tl.where(weight > 0, value * weight, 0.0)
    if Z.dtype.element_ty == tl.float64:
        inf_log = libdevice.log(maximum * 0.5 - label * 0.5) + 0.6931471805599453
        log_value = tl.where(
            relative < -36,
            relative,
            tl.where(relative == float("inf"), inf_log, libdevice.log(value)),
        )
        log_value = tl.where(weight > 0, log_value + libdevice.log(weight), -float("inf"))
        if WEIGHTED:
            under = (
                (value < 2.2250738585072014e-308)
                | (weighted_value < 2.2250738585072014e-308)
                | (weighted_value == float("inf"))
            )
            any_under = (tl.max(under.to(tl.int32), 0) > 0) if ROWS > 1 else under
            if any_under:
                full_val = _triton_reduce._exp_full(log_value)
                weighted_value = tl.where(under & (weight > 0), full_val, weighted_value)
        tl.store(S + 5 * N + 2 + row, log_value, mask)
    tl.store(S + 4 * N + row, tl.where(is_bad, nan, weighted_value), mask)


@triton.jit(do_not_specialize=["N"])
def _partials(S, OUT, N, REDUCTION: tl.constexpr, B: tl.constexpr, FULL64: tl.constexpr):
    block = tl.program_id(0)
    j = block * B + tl.arange(0, B)
    valid = j < N
    count = tl.cdiv(N, B)
    partial = S + 5 * N + 2 + (N + 1 if FULL64 else 0) + block
    values = tl.load(S + 4 * N + j, valid, other=0.0)
    w = tl.load(S + 4 * j + 3, valid, other=0.0)
    if REDUCTION == 0:
        tl.store(OUT + j, values, valid)
    tl.store(partial, tl.sum(values, 0))
    tl.store(partial + count, tl.sum(tl.maximum(w, 0.0), 0))
    tl.store(partial + 2 * count, tl.max(tl.maximum(-w, 0.0), 0))


@triton.jit(do_not_specialize=["N"])
def _log_weight_sum(S, N, B: tl.constexpr, divisor):
    """Log of nonnegative row weights, even when their ordinary sum overflows."""
    result = libdevice.log(divisor)
    if divisor == float("inf"):
        top = tl.zeros([B], tl.float64)
        for start in range(0, N, B):
            j = start + tl.arange(0, B)
            top = tl.maximum(top, tl.load(S + 4 * j + 3, j < N, other=0.0))
        largest = tl.max(top, 0)
        terms = tl.zeros([B], tl.float64)
        for start in range(0, N, B):
            j = start + tl.arange(0, B)
            w = tl.load(S + 4 * j + 3, j < N, other=0.0)
            terms += tl.maximum(w, 0.0) / largest
        result = libdevice.log(largest) + libdevice.log(tl.sum(terms, 0))
    return result


@triton.jit
def _invalid_weights(
    OUT, N, B: tl.constexpr, bad_flag, WEIGHTED: tl.constexpr, REDUCTION: tl.constexpr
):
    """Row 0 checks the class weights: an invalid one makes every row value NaN, as every
    gradient row in backward."""
    if WEIGHTED and REDUCTION == 0:
        if bad_flag == 3:
            tl.debug_barrier()  # after this program's own row stores
            for start in range(0, N, B):
                j = start + tl.arange(0, B)
                tl.store(OUT + j, tl.zeros([B], tl.float64) + float("nan"), j < N)


@triton.jit(do_not_specialize=["N"])
def _finish(
    S,
    OUT,
    N,
    REDUCTION: tl.constexpr,
    B: tl.constexpr,
    DIRECT: tl.constexpr,
    FULL64: tl.constexpr,
    WEIGHTED: tl.constexpr,
):
    count = N if DIRECT else tl.cdiv(N, B)
    total = tl.zeros([B], tl.float64)
    weight = tl.zeros([B], tl.float64)
    flag = tl.zeros([B], tl.float64)
    for start in range(0, count, B):
        j = start + tl.arange(0, B)
        valid = j < count
        if DIRECT:
            values = tl.load(S + 4 * N + j, valid, other=0.0)
            w = tl.load(S + 4 * j + 3, valid, other=0.0)
            total += values
            weight += tl.maximum(w, 0.0)
            flag = tl.maximum(flag, tl.maximum(-w, 0.0))
            if REDUCTION == 0:
                tl.store(OUT + j, values, valid)
        else:
            partial = S + 5 * N + 2 + (N + 1 if FULL64 else 0)
            total += tl.load(partial + j, valid, other=0.0)
            weight += tl.load(partial + count + j, valid, other=0.0)
            flag = tl.maximum(flag, tl.load(partial + 2 * count + j, valid, other=0.0))
    bad_flag = tl.max(flag, 0)
    divisor = tl.sum(weight, 0)
    _invalid_weights(OUT, N, B, bad_flag, WEIGHTED, REDUCTION)
    tl.store(S + 5 * N, bad_flag)
    tl.store(S + 5 * N + 1, divisor)
    log_divisor = tl.full([], 0.0, tl.float64)
    if FULL64 and WEIGHTED:
        if REDUCTION == 1:
            log_divisor = _log_weight_sum(S, N, B, divisor)
        tl.store(S + 6 * N + 2, log_divisor)
    if REDUCTION != 0:
        value = tl.sum(total, 0)
        if REDUCTION == 1:
            value = value / divisor
        if FULL64:
            # Rare range rescue, without changing ordinary reduction rounding.
            rescue = (value == float("inf")) | (tl.sum(total, 0) < 2.2250738585072014e-308 * N)
            if WEIGHTED:
                rescue = rescue | (value != value) | ((REDUCTION == 1) & (divisor == float("inf")))
            if rescue:
                logs = S + 5 * N + 2
                top = _triton_reduce._maximum(logs, N, B)
                terms = tl.zeros([B], tl.float64)
                for start in range(0, N, B):
                    j = start + tl.arange(0, B)
                    terms += libdevice.exp(tl.load(logs + j, j < N, other=-float("inf")) - top)
                log_total = top + libdevice.log(tl.sum(terms, 0))
                if REDUCTION == 1:
                    if WEIGHTED:
                        log_total -= log_divisor
                    else:
                        log_total -= libdevice.log(divisor)
                value = _triton_reduce._exp_full(log_total)
                value = tl.where(top == -float("inf"), 0.0, value)
            if REDUCTION == 1:
                value = tl.where(divisor == 0, float("nan"), value)
        tl.store(OUT, tl.where(bad_flag > 0, float("nan"), value))


@functools.lru_cache(maxsize=_cuda_launch.CACHE)
def _compiled(device, dtype, weight_dtype, c, code, ignore_index, fp32=True):
    with torch.cuda.device(device):
        z = torch.empty(1, device="cuda", dtype=dtype)
        y = torch.empty(1, device="cuda", dtype=torch.int64)
        s = torch.empty(1, device="cuda", dtype=torch.float64)
        full64 = dtype == torch.float64
        out = torch.empty(1, device="cuda", dtype=torch.float64 if full64 else torch.float32)
        weight = y if weight_dtype is None else torch.empty(1, device="cuda", dtype=weight_dtype)
        block, warps = _triton_entropy._shape(c, fp32)
        b_bwd = _triton_entropy.bwd_columns(c, fp32)
        rows = _triton_entropy.rows_per_cta(c)
        k_fwd = _forward.warmup(
            z,
            y,
            weight,
            s,
            2,
            c,
            ignore_index,
            weight_dtype is not None,
            block,
            fp32,
            rows,
            num_warps=warps,
            grid=(1,),
        )
        part = _partials.warmup(s, out, 2, code, BLOCK, full64, num_warps=4, grid=(1,))
        finish = _finish.warmup(
            s, out, 2, code, BLOCK, False, full64, weight_dtype is not None, num_warps=4, grid=(1,)
        )
        direct = _finish.warmup(
            s, out, 2, code, BLOCK, True, full64, weight_dtype is not None, num_warps=4, grid=(1,)
        )
        bwd = _triton_entropy.warmup_backward(
            z, y, s, out, c, code, fp32, ignore_index, weight_dtype is not None, True
        )
        p, i = _cuda_launch.POINTER, _cuda_launch.INT32
        return _cuda_launch.handles(
            [(k_fwd, (p, p, p, p, i))]
            + [(k, (p, p, i)) for k in (part, finish, direct)]
            + [(bwd, (p, p, p, p, p, i))],
            BLOCK,
            DIRECT_ROWS,
            b_bwd,
            rows,
            rows,
        )


def launcher(logits, weight, code, ignore_index, fp32=True):
    host, reason = _cuda_launch.status()
    if host is None:
        return None, reason
    device = logits.get_device()
    handles = _compiled(
        device,
        logits.dtype,
        None if weight is None else weight.dtype,
        logits.shape[1],
        code,
        ignore_index,
        fp32,
    )
    return (None, handles) if isinstance(handles, str) else (host, handles)


def forward(logits, target, weight, ignore_index, reduction, fp32=True):
    code = _CODES[reduction]
    host, handles = launcher(logits, weight, code, ignore_index, fp32)
    if host is not None:
        return host.weighted_cuda_forward(
            logits, target, target if weight is None else weight, code, handles
        )
    with _cuda_launch.device_guard(logits):
        return _triton_forward(logits, target, weight, ignore_index, code, fp32)


def stats_size(n: int, full64: bool) -> int:
    """Row statistics (4 per row), values, status and divisor, float64 logs, partials."""
    blocks = 0 if n <= DIRECT_ROWS else 3 * triton.cdiv(n, BLOCK)
    return (6 if full64 else 5) * n + 2 + int(full64) + blocks


def _triton_forward(logits, target, weight, ignore_index, code, fp32=True):
    n, c = logits.shape
    direct = n <= DIRECT_ROWS
    full64 = logits.dtype == torch.float64
    stats = torch.empty(stats_size(n, full64), device=logits.device, dtype=torch.float64)
    output = torch.empty(
        (n,) if code == 0 else (),
        device=logits.device,
        dtype=torch.float64 if full64 else torch.float32,
    )
    block, warps = _triton_entropy._shape(c, fp32)
    rows = _triton_entropy.rows_per_cta(c)
    _forward[(triton.cdiv(n, rows),)](
        logits,
        target,
        target if weight is None else weight,
        stats,
        n,
        c,
        ignore_index,
        weight is not None,
        block,
        fp32,
        rows,
        num_warps=warps,
    )
    if not direct:
        _partials[(triton.cdiv(n, BLOCK),)](stats, output, n, code, BLOCK, full64, num_warps=4)
    _finish[(1,)](stats, output, n, code, BLOCK, direct, full64, weight is not None, num_warps=4)
    return output, stats, None


def backward(
    logits, target, weight, stats, upstream, ignore_index, reduction, cuda_context, fp32=True
):
    code = _CODES[reduction]
    host, handles = launcher(logits, weight, code, ignore_index, fp32)
    if host is not None and cuda_context is not None:
        return host.weighted_cuda_backward(
            logits, target, stats, upstream, code, handles, cuda_context
        )
    with _cuda_launch.device_guard(logits):
        return _triton_entropy.launch_backward(
            logits, target, stats, upstream, code, fp32, ignore_index, weight is not None, True
        )


def apply(logits, target, weight, ignore_index, reduction, fp32=True):
    code = _CODES[reduction]
    host, handles = launcher(logits, weight, code, ignore_index, fp32)
    if host is None:
        return None
    return host.weighted_cuda_apply(
        logits,
        target,
        target if weight is None else weight,
        weight is not None,
        ignore_index,
        code,
        handles,
    )
