"""Reduce wide row values in a fixed order and cast once to the public dtype."""

from __future__ import annotations

import triton
import triton.language as tl

from .._limits import FLOAT32
from . import common
from .common import Builder, Ref
from .math import add_parts, doubles

_BLOCK = (2048, 8)


@triton.jit
def _weights(Y, W, i, live, ignore, group, C: tl.constexpr, B: tl.constexpr,
             GROUPED: tl.constexpr, WEIGHTED: tl.constexpr, IGNORED: tl.constexpr,
             TERMS: tl.constexpr):  # fmt: skip
    if TERMS:
        return tl.load(Y + i, live, other=0).to(tl.float64)
    y = tl.load(Y + i, live, other=0)
    if WEIGHTED:
        index = tl.minimum(tl.maximum(y, 0), C - 1)
        if GROUPED:
            index += i // group * C
        w = tl.load(W + index, live, other=0).to(tl.float64)
    else:
        w = tl.full([B], 1.0, tl.float64)
    if IGNORED:
        w = tl.where(y == ignore, 0.0, w)
    return tl.where(live, w, 0.0)


@triton.jit
def _total(x, low, COMPENSATE: tl.constexpr):
    if COMPENSATE:
        value, error = tl.reduce((x, low), 0, add_parts)
        return tl.where(tl.abs(value) < float("inf"), value + error, value)
    else:
        return tl.sum(x, 0)


@triton.jit
def _sums(V, Y, W, N, ROWS, IGNORE, GROUP, scale, C: tl.constexpr, B: tl.constexpr,
          COUNTED: tl.constexpr, GROUPED: tl.constexpr, WEIGHTED: tl.constexpr,
          IGNORED: tl.constexpr, COMPENSATE: tl.constexpr, TERMS: tl.constexpr):  # fmt: skip
    total = tl.full([B], 0, tl.float64)
    count = tl.full([B], 0, tl.float64)
    low = tl.full([B], 0, tl.float64)
    count_low = tl.full([B], 0, tl.float64)
    base = tl.program_id(0).to(tl.int64) * N
    for start in range(0, N, B):
        r = start + tl.arange(0, B)
        i = base + r
        v = tl.load(V + i, r < N, other=0).to(tl.float64) * scale
        if COMPENSATE:
            total, low = add_parts(total, low, v, 0.0)
        else:
            total += v
        if COUNTED:
            w = (
                _weights(Y, W, i, r < N, IGNORE, GROUP, C, B, GROUPED, WEIGHTED, IGNORED, TERMS)
                * scale
            )
            if COMPENSATE:
                count, count_low = add_parts(count, count_low, w, 0.0)
            else:
                count += w
    numerator = _total(total, low, COMPENSATE)
    if COUNTED:
        divisor = _total(count, count_low, COMPENSATE)
    else:
        divisor = tl.cast(ROWS, tl.float64) * scale
    return numerator, divisor


@triton.jit(do_not_specialize=["N", "IGNORE", "GROUP", "ROWS", "OFFSET"])
def _reduce(V, Y, W, L, D, N, IGNORE, GROUP, ROWS, OFFSET, C: tl.constexpr, B: tl.constexpr,
            MEAN: tl.constexpr, COUNTED: tl.constexpr, GROUPED: tl.constexpr,
            WEIGHTED: tl.constexpr, IGNORED: tl.constexpr, COMPENSATE: tl.constexpr,
            TERMS: tl.constexpr, COUNT_OFFSET: tl.constexpr = 0,
            SPLIT: tl.constexpr = False):  # fmt: skip
    """The sum or mean of N values of V (of ROWS rows: a mean's divisor unless COUNTED).
    COUNT_OFFSET (constexpr) or SPLIT (OFFSET): V holds the partial sums, then their counts."""
    if COUNT_OFFSET:
        # The partial sums and counts occupy consecutive halves of one buffer.
        V = doubles(V)
        Y = V + COUNT_OFFSET
    if SPLIT:
        V = doubles(V)
        Y = V + OFFSET
    value, divisor = _sums(V, Y, W, N, ROWS, IGNORE, GROUP, 1.0, C, B, COUNTED, GROUPED,
                           WEIGHTED, IGNORED, COMPENSATE, TERMS)  # fmt: skip
    if MEAN:
        if COUNTED:
            tl.store(D + tl.program_id(0), divisor)
        if (tl.abs(value) == float("inf")) | (tl.abs(divisor) == float("inf")):
            # Scale before adding, including the label weights: either sum can
            # overflow FP64 while its quotient remains finite.
            scale = tl.exp2(-2.0 - tl.floor(tl.log2(tl.cast(N, tl.float64))))
            value, divisor = _sums(V, Y, W, N, ROWS, IGNORE, GROUP, scale, C, B, COUNTED,
                                   GROUPED, WEIGHTED, IGNORED, COMPENSATE, TERMS)  # fmt: skip
        value /= divisor
    tl.store(L + tl.program_id(0), value.to(L.dtype.element_ty))


@triton.jit(do_not_specialize=["N", "DIVISOR"])
def _sum_blocks(V, L, N, DIVISOR, PARTS: tl.constexpr, B: tl.constexpr,
                FINAL: tl.constexpr, MEAN: tl.constexpr):  # fmt: skip
    work = tl.program_id(0).to(tl.int64)
    row = work // PARTS
    i = (work % PARTS) * B + tl.arange(0, B)
    total = tl.sum(tl.load(doubles(V) + row * N + i, i < N, other=0.0), 0)
    if FINAL:
        if MEAN:
            total /= tl.cast(DIVISOR, tl.float64)
        tl.store(L + work, total.to(L.dtype.element_ty))
    else:
        tl.store(doubles(L) + work, total)


@triton.jit(do_not_specialize=["N", "IGNORE"])
def _counted_blocks(V, Y, L, N, IGNORE, PARTS: tl.constexpr, B: tl.constexpr):
    work = tl.program_id(0).to(tl.int64)
    r = (work % PARTS) * B + tl.arange(0, B)
    i = (work // PARTS) * N + r
    value = tl.load(doubles(V) + i, r < N, other=0.0)
    y = tl.load(Y + i, r < N, other=0)
    tl.store(doubles(L) + work, tl.sum(value, 0))
    tl.store(
        doubles(L) + tl.num_programs(0).to(tl.int64) + work,
        tl.sum(((r < N) & (y != IGNORE)).to(tl.int32), 0),
    )


def _parallel_row_sum(b: Builder, out: Ref, rows: int, groups: int) -> None:
    """Reduce independent groups of rows in parallel, retaining FP64 in intermediate blocks."""
    block = 1024
    source = Ref("V")
    count = rows
    while count > block:
        parts = common.cdiv(count, block)
        partial = b.array(f"partial_{count}", (groups, parts), "float64")
        args = dict(V=source, L=partial, N=count, DIVISOR=rows)
        b.launch(
            _sum_blocks, (groups * parts,), args, 4, PARTS=parts, B=block, FINAL=False, MEAN=False
        )
        source, count = partial, parts
    args = dict(V=source, L=out, N=count, DIVISOR=rows)
    b.launch(
        _sum_blocks,
        (groups,),
        args,
        4,
        PARTS=1,
        B=triton.next_power_of_2(count),
        FINAL=True,
        MEAN=b.call.reduction == common.MEAN,
    )


def _parallel(c: common.Call, rows: int) -> bool:
    """Whether a fixed tree of block sums (deterministic for a shape) replaces the serial
    one-program reduction: once that program has more 2048-row tiles than its eight warps,
    it runs bandwidth-bound on one SM. The tree has no
    scaled overflow retry, no compensation and no weight divisor: it takes float32-valued
    results of FP32-range inputs, whose FP64 sum of row values cannot overflow (for
    distillation, also an FP32-range temperature), and means with either a fixed divisor or an
    unweighted label count."""
    if rows <= _BLOCK[0] * _BLOCK[1] or not c.fp32 or (c.counted and c.weighted):
        return False
    if c.value_dtype != "float32" or c.target == "float64":
        return False
    return c.kind != common.DISTILLATION or FLOAT32.tiny <= c.temperature <= 1.0 / FLOAT32.tiny


def add(b: Builder, *, terms: bool = False, partials: int = 0) -> None:
    """Append the common reduction for one scalar per mapped example. partials: the row
    kernel's programs wrote that many sums of their values to T, then the sums of their
    label weights (one group)."""
    c = b.call
    rows = c.reduction_rows or c.n
    groups = c.n // rows if c.reduction_rows else 1
    shape = (groups,) if c.reduction_rows else ()
    out = b.array("L", shape, c.value_dtype, internal=False)
    if partials:
        sums = Ref("T")
        divisor = b.array("D", shape, "float64", internal=False) if c.counted else Ref("Z")
        block = min(_BLOCK[0], triton.next_power_of_2(partials))
        args = dict(
            V=sums, Y=sums, W=Ref("Z"), L=out, D=divisor, N=partials, IGNORE=c.ignore_index,
            GROUP=1, ROWS=rows, OFFSET=partials,
        )  # fmt: skip
        b.launch(
            _reduce, (1,), args, min(_BLOCK[1], max(1, block // 32)), C=c.c, B=block,
            MEAN=c.reduction == common.MEAN, COUNTED=c.counted, GROUPED=False, WEIGHTED=False,
            IGNORED=False, COMPENSATE=c.value_dtype == "float64", TERMS=c.counted, SPLIT=True,
        )  # fmt: skip
        return
    values = Ref("V")
    count_offset = 0
    if _parallel(c, rows) and not terms:
        if not c.counted:
            _parallel_row_sum(b, out, rows, groups)
            return
        block = 1024
        parts = common.cdiv(rows, block)
        count_offset = groups * parts
        sums = b.array("partial_values_counts", (2, groups, parts), "float64")
        args = dict(V=values, Y=Ref("Y"), L=sums, N=rows, IGNORE=c.ignore_index)
        b.launch(_counted_blocks, (groups * parts,), args, 4, B=block, PARTS=parts)
        values, rows, terms = sums, parts, True
    divisor = b.array("D", shape, "float64", internal=False) if c.counted or terms else Ref("Z")
    args = dict(
        V=values,
        Y=Ref("Y"),
        W=Ref("W" if c.weighted else "Z"),
        L=out,
        D=divisor,
        N=rows,
        GROUP=c.rows_per_group,
        IGNORE=c.ignore_index,
        ROWS=rows,
        OFFSET=0,
    )
    block = min(_BLOCK[0], triton.next_power_of_2(max(rows, 1)))
    b.launch(
        _reduce,
        (groups,),
        args,
        min(_BLOCK[1], max(1, block // 32)),
        C=c.c,
        B=block,
        MEAN=c.reduction == common.MEAN,
        COUNTED=c.counted or terms,
        GROUPED=bool(c.group),
        WEIGHTED=c.weighted,
        IGNORED=c.ignore,
        COMPENSATE=c.value_dtype == "float64",
        TERMS=terms,
        COUNT_OFFSET=count_offset,
    )
