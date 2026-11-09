"""Deterministic multi-block reduction of the fused kernels' per-row losses.

The row statistics S hold the row values at S[VALUE * N:], their logs at S[LOG * N:]
(entropy; LOG < 0 for none), invalid-row flags at S[BAD * N:] and the status word at
S[STATUS * N], followed by four partials per BLOCK rows when N exceeds DIRECT_ROWS.
Up to DIRECT_ROWS rows, one program reduces them in chunks of DIRECT_BLOCK.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK = 2048
WARPS = 8
DIRECT_ROWS = 16384
DIRECT_BLOCK = 2048
DIRECT_WARPS = 8


@triton.jit
def _exp_full(x):
    half = libdevice.exp(x * 0.5)
    return half * half


@triton.jit(do_not_specialize=["N"])
def partials(
    S,
    OUT,
    N,
    VALUE: tl.constexpr,
    LOG: tl.constexpr,
    BAD: tl.constexpr,
    STATUS: tl.constexpr,
    REDUCTION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per-block maximum, sum, flag and log maximum; REDUCTION = 0 copies the values."""
    N = N.to(tl.int64)  # STATUS * N exceeds int32 within ROWS
    block = tl.program_id(0)
    j = block * BLOCK + tl.arange(0, BLOCK)
    valid = j < N
    count = tl.cdiv(N, BLOCK)
    W = S + STATUS * N + 1 + block
    values = tl.load(S + VALUE * N + j, valid, other=0)
    tl.store(W + 2 * count, tl.max(tl.load(S + BAD * N + j, valid, other=0), 0))
    if REDUCTION == 0:
        tl.store(OUT + j, values, valid)
    else:
        tl.store(W, tl.max(values, 0))
        tl.store(W + count, tl.sum(values, 0))
        if LOG >= 0:
            logs = tl.load(S + LOG * N + j, valid, other=-float("inf"))
            tl.store(W + 3 * count, tl.max(logs, 0))


@triton.jit
def _ordinary(
    S, N, maximum, total, VALUE: tl.constexpr, REDUCTION: tl.constexpr, BLOCK: tl.constexpr
):
    result = total
    if REDUCTION == 1:
        # A finite mean of values whose sum overflows.
        if (total == float("inf")) & (maximum < float("inf")):
            scaled = tl.zeros([BLOCK], tl.float64)
            for start in range(0, N, BLOCK):
                j = start + tl.arange(0, BLOCK)
                scaled += tl.load(S + VALUE * N + j, j < N, other=0) / maximum
            result = (tl.sum(scaled, 0) / N) * maximum
        else:
            result = total / N
    return result


@triton.jit
def _maximum(X, count, BLOCK: tl.constexpr):
    top = tl.full([BLOCK], -float("inf"), tl.float64)
    for start in range(0, count, BLOCK):
        j = start + tl.arange(0, BLOCK)
        top = tl.maximum(top, tl.load(X + j, j < count, other=-float("inf")))
    return tl.max(top, 0)


@triton.jit
def _log_domain(S, N, log_max, LOG: tl.constexpr, REDUCTION: tl.constexpr, BLOCK: tl.constexpr):
    terms = tl.zeros([BLOCK], tl.float64)
    for start in range(0, N, BLOCK):
        j = start + tl.arange(0, BLOCK)
        terms += libdevice.exp(tl.load(S + LOG * N + j, j < N, other=-float("inf")) - log_max)
    log_total = log_max + libdevice.log(tl.sum(terms, 0))
    if REDUCTION == 1:
        log_total = log_total - libdevice.log(N.to(tl.float64))
    return tl.where(log_max == -float("inf"), 0.0, _exp_full(log_total))


@triton.jit(do_not_specialize=["N"])
def finish(
    S,
    OUT,
    N,
    VALUE: tl.constexpr,
    LOG: tl.constexpr,
    BAD: tl.constexpr,
    STATUS: tl.constexpr,
    REDUCTION: tl.constexpr,
    BLOCK: tl.constexpr,
    DIRECT: tl.constexpr,
):
    """Status word and loss from the rows (DIRECT) or from the partials of BLOCK rows.

    DIRECT reads the rows in chunks of BLOCK. Entropy (LOG >= 0) sums logs outside the
    normal range: a sum of individually unrepresentable tails, or a mean of losses
    beyond double's range, can be finite.
    """
    N = N.to(tl.int64)  # STATUS * N exceeds int32 within ROWS
    if DIRECT:
        count = N
        maxima = S + VALUE * N
        sums = maxima
        flags = S + BAD * N
    else:
        count = tl.cdiv(N, BLOCK)
        maxima = S + STATUS * N + 1
        sums = maxima + count
        flags = maxima + 2 * count
    flag = tl.zeros([BLOCK], tl.float64)
    top = tl.zeros([BLOCK], tl.float64)
    total = tl.zeros([BLOCK], tl.float64)
    for start in range(0, count, BLOCK):
        j = start + tl.arange(0, BLOCK)
        valid = j < count
        flag = tl.maximum(flag, tl.load(flags + j, valid, other=0))
        if REDUCTION == 0:
            if DIRECT:
                tl.store(OUT + j, tl.load(sums + j, valid, other=0), valid)
        else:
            m = tl.load(maxima + j, valid, other=0)
            top = tl.maximum(top, m)
            if DIRECT:
                total += m
            else:
                total += tl.load(sums + j, valid, other=0)
    tl.store(S + STATUS * N, tl.max(flag, 0))
    if REDUCTION != 0:
        sum_total = tl.sum(total, 0)
        maximum = tl.max(top, 0)
        if sum_total != sum_total:
            result = sum_total
        elif LOG >= 0:
            if (maximum >= 2.2250738585072014e-308) & (maximum < float("inf")):
                result = _ordinary(S, N, maximum, sum_total, VALUE, REDUCTION, BLOCK)
            else:
                # Rare: the log maximum is read only here.
                if DIRECT:
                    log_max = _maximum(S + LOG * N, N, BLOCK)
                else:
                    log_max = _maximum(S + STATUS * N + 1 + 3 * count, count, BLOCK)
                result = _log_domain(S, N, log_max, LOG, REDUCTION, BLOCK)
        else:
            result = _ordinary(S, N, maximum, sum_total, VALUE, REDUCTION, BLOCK)
        tl.store(OUT, result)


def size(n: int, status: int) -> int:
    """Length of the float64 row statistics, with the status word and block partials."""
    return status * n + 1 + (4 * triton.cdiv(n, BLOCK) if n > DIRECT_ROWS else 0)


def launch(stats: torch.Tensor, out: torch.Tensor, n: int, code: int, layout: tuple) -> None:
    """Reduce through Triton's launcher; the C++ launcher mirrors this (reduce_rows)."""
    if n > DIRECT_ROWS:
        partials[(triton.cdiv(n, BLOCK),)](stats, out, n, *layout, code, BLOCK, num_warps=WARPS)
        finish[(1,)](stats, out, n, *layout, code, BLOCK, False, num_warps=WARPS)
    else:
        finish[(1,)](stats, out, n, *layout, code, DIRECT_BLOCK, True, num_warps=DIRECT_WARPS)


def warmup(stats: torch.Tensor, out: torch.Tensor, code: int, layout: tuple) -> list:
    """Compiled partials, multi-block finish and one-launch finish kernels."""
    return [
        partials.warmup(stats, out, 2, *layout, code, BLOCK, num_warps=WARPS, grid=(1,)),
        finish.warmup(stats, out, 2, *layout, code, BLOCK, False, num_warps=WARPS, grid=(1,)),
        finish.warmup(
            stats, out, 2, *layout, code, DIRECT_BLOCK, True, num_warps=DIRECT_WARPS, grid=(1,)
        ),
    ]
