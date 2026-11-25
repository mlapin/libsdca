"""Cross-entropy, as include/sdca/loss/ce.h: integer labels (class weights, ignore_index, label
smoothing), probability targets (class weights, label smoothing) and label masks
(multilabel_cross_entropy).

Label rows save four statistics (center M, the largest score; first = 1 / sum(exp(z - M));
the target total A; and the label's gradient Gy per unit upstream), from which the gradient is

    g_j = s (A exp(z_j - M) first - t_j),    g_y = s Gy (labels),

with s the row's gradient factor and t_j the row's target (eps / C w_j under smoothing). Label
rows compute the competitors' mass directly from their sum and their maximum relative to
the label. Dense rows also retain the mean target on the maxima and its tail correction
to preserve the gradient when a winning probability rounds to one. Float32 work uses float64
row sums. Short rows share a program (ROWS x B tiles) and rows up to 4096 classes stay in
registers. Unsmoothed float32-work label rows of 32 to 4096 classes (values and statistics)
and their gradients run as lane tiles of 16-byte vectors (_lane_row, _lane_backward): a row
over a few lanes, each thread finishing one row's float64 statistics; wider label rows of
many rows take one looped program per row (_looped_row). A label kernel whose values a sum
or mean reduces writes per-program partial sums instead (PARTIAL). Wider float32-work label, probability and mask rows run as parallel aligned row
chunks (rows.py): one read for the chunks' partial statistics and a float64 merge per row,
then one read and one write for the gradient; rows a chunk marks (NaN, +inf, masked
competitors under smoothing, extreme scores) take the streaming kernel alone. A wide mask
row's float32 pass is final where its maximum is a negative class (public's rule: the
gradient's norm is then at least that class's probability, which bounds the error
normwise); a positive at the maximum, whose gradient can cancel, takes the double
exponentials of the streaming kernel and _mask_backward. Float64 work streams.

One kernel computes a row's values, statistics and gradient. Register rows are read once.
Streamed joint calls read the row again, last chunk first. A backward that consumes saved
statistics distributes columns over a grid of blocks (common.SAVED). Label smoothing sums the
spread only for values; its float64 gaps take no weights without class weights.

A score of -inf masks its class (exponential +0.0, gradient +0.0). A masked label gives loss
+inf and gradient -1 there, as torch.nn.functional.cross_entropy; a masked class of positive
probability target gives +inf and -t; a masked positive gives the same infinite-loss limit. NaN, +inf, a row
of -inf only, a label out of range (unless ignored), a negative or nonfinite probability, and a
zero weight at a masked label give NaN rows.
"""

from __future__ import annotations

from typing import Any

import triton
import triton.language as tl

from . import common, rows
from .common import Builder, Call, Ref, pow2
from .math import (
    DBL_MAX,
    EXP_UNWIDE,
    EXP_WIDE,
    FLT_MAX,
    add_parts,
    doubles,
    exp,
    exp32,
    exp_parts,
    expm1,
    f64,
    finite,
    label_statistics,
    log,
    log1p,
    scale,
    target_row,
    wide_exp,
    work,
)
from .rows import chunks as row_chunks

LABELS, PROBABILITIES, MASK = tl.constexpr(0), tl.constexpr(1), tl.constexpr(2)
# Keep at most 4096 scores per program for a resident row. A streamed double
# chunk has the same 8 KiB of score payload as its float counterpart.
_REGISTER = 4096
STRIDED = True  # every kernel takes row-strided scores (Call.stride)
_CHUNKS = {True: (2048, 4), False: (1024, 4)}
_BACKWARD_TILES = 4


@triton.jit
def _weights(W, base, k, valid, WEIGHTED: tl.constexpr):
    """Class weights of columns k (1 without weights, 0 off the row), float64."""
    if WEIGHTED:
        w = tl.load(W + base + k, valid, other=0).to(tl.float64)
    else:
        w = valid.to(tl.float64)
    return w


@triton.jit
def _chunk(
    base,
    start,
    y,
    acc,
    top,
    gap,
    mass,
    shift,
    W,
    wbase,
    C: tl.constexpr,
    B: tl.constexpr,
    FP32: tl.constexpr,
    MASKED: tl.constexpr,
    GAP: tl.constexpr,
    MASS: tl.constexpr,
    POLICY: tl.constexpr,
):
    """Lanes and running maximum of the competitors after columns k; with GAP their sums of
    w_j (top - z_j) (with class weights per lane, else one shift per rise of top), with MASS
    of w_j. MASKED for the chunk of the label and the last one, the only ones with columns to
    skip."""
    k = start + tl.arange(0, B)
    keep = (k < C) & (k != y)
    if MASKED:
        z = work(tl.load(base + k, k < C, other=0, eviction_policy=POLICY), FP32)
        new = tl.maximum(top, tl.max(tl.where(keep, z, -float("inf")), 0))
    else:
        z = work(tl.load(base + k, eviction_policy=POLICY), FP32)
        new = tl.maximum(top, tl.max(z, 0))
    if new > top:
        acc = acc * exp(top - new)
        if GAP:
            rise = new.to(tl.float64) - top.to(tl.float64)
            if MASS:
                gap = gap + mass * rise
            else:
                # The competitors before this chunk: columns [0, start) but the label.
                shift = shift + rise * (start - (y < start).to(tl.int64))
        top = new
    # top is finite or +inf: e is +0.0 at -inf, NaN at NaN and at +inf.
    if MASKED:
        e = exp(tl.where(keep, z - top, -float("inf")))
    elif FP32:
        e = exp32(z - top)
    else:
        e = exp(z - top)
    if MASS:
        w = _weights(W, wbase, k, keep, True)
        if GAP:
            gap += tl.where(w > 0, w * (top.to(tl.float64) - z.to(tl.float64)), 0.0)
        mass += w
    elif GAP:
        drop = top.to(tl.float64) - z.to(tl.float64)
        gap += tl.where(keep, drop, 0.0) if MASKED else drop
    return acc + e, top, gap, mass, shift


@triton.jit
def _stream(
    Z,
    W,
    row,
    wbase,
    y,
    C: tl.constexpr,
    B: tl.constexpr,
    FP32: tl.constexpr,
    GAP: tl.constexpr,
    MASS: tl.constexpr,
    POLICY: tl.constexpr,
    SZ: tl.constexpr,
):
    """Competitor maximum (+inf for NaN or +inf among them, -inf when all are masked), the
    sum of exp(z - maximum) over chunks of B columns, sum w_j (top - z_j) (GAP) and sum w_j
    (MASS) over the competitors, and their running maximum top. The running maximum starts at
    the lowest finite value, so masked classes read first add +0.0."""
    base = Z + row * SZ
    FULL: tl.constexpr = C // B * B
    split = y // B * B
    if FP32:
        top = tl.full([], -FLT_MAX, tl.float32)
    else:
        top = tl.full([], -DBL_MAX, tl.float64)
    acc = work(tl.zeros([B], tl.float64), FP32)
    gap = tl.zeros([B], tl.float64)
    mass = tl.zeros([B], tl.float64)
    shift = tl.zeros([], tl.float64)
    for start in range(0, split, B):
        acc, top, gap, mass, shift = _chunk(
            base, start, y, acc, top, gap, mass, shift, W, wbase, C, B, FP32, False, GAP, MASS,
            POLICY,
        )  # fmt: skip
    if split < FULL:
        acc, top, gap, mass, shift = _chunk(
            base, split, y, acc, top, gap, mass, shift, W, wbase, C, B, FP32, True, GAP, MASS,
            POLICY,
        )  # fmt: skip
    for start in range(split + B, FULL, B):
        acc, top, gap, mass, shift = _chunk(
            base, start, y, acc, top, gap, mass, shift, W, wbase, C, B, FP32, False, GAP, MASS,
            POLICY,
        )  # fmt: skip
    if FULL < C:
        acc, top, gap, mass, shift = _chunk(
            base, FULL, y, acc, top, gap, mass, shift, W, wbase, C, B, FP32, True, GAP, MASS,
            POLICY,
        )  # fmt: skip
    total = tl.sum(acc.to(tl.float64), 0)
    maximum = tl.where(total == total, top.to(tl.float64), float("inf"))
    maximum = tl.where(total == 0, -float("inf"), maximum)
    return maximum, total, tl.sum(gap, 0) + shift, tl.sum(mass, 0), top.to(tl.float64)


# Wide float32-work rows run as parallel row chunks (rows.py): each (row, chunk) program
# writes its partial statistics (_LABEL_PART or _DENSE_PART float64 words), and one program
# per row merges them exactly in float64 (_labels and _dense with CHUNKS) before a lean
# gradient over the same aligned chunks (_chunk_backward). The chunks of ordinary rows take
# exp2 of (z - m) log2(e), m their maximum: the difference is exact near m at any scale. (An
# FMA z log2(e) - c with c = m log2(e) rounded is not: c is off by up to half its last place,
# 0.5 at |m| = 2**23, which a first-order correction of the sum cannot repair: 4097 classes
# at 2**23 would give a loss of 7.279 for 7.318.) A chunk with NaN, +inf, a gap that
# does not stay finite or extreme scores marks its row, which the merge then streams itself.
_PARTS = (2048, 4)
_DENSE_CHUNKED = 512
_CHUNK_BACKWARD = (2048, 4)
# Float32 probability chunks refine in float64 wherever one of their classes cancels; at about
# 1% of lanes per class almost every 2048-lane chunk does. Half-width chunks leave more of
# the row on the float32 path (ordinary label and 16-bit chunks keep _CHUNK_BACKWARD).
_CHUNK_REFINE = (1024, 4)
_LABEL_PART = tl.constexpr(5)  # competitors' maximum, sum exp(z - maximum), invalid, gap, mass
_LOG2E = tl.constexpr(1.4426950408889634)
_LEAN_MAX = tl.constexpr(2.0**100)  # |chunk maximum| of ordinary chunks
# Mask chunks keep every finite term of the partition: e**-80 is a normal float32.
_SPREAD = tl.constexpr(80.0)
# The anchor a mask row's statistics carry when its one float32 chunk pass is final (its
# maximum is a negative class): the lean chunk gradient writes it (real anchors are >= 0
# or -inf).
_LEAN_MASK = tl.constexpr(-2.0)
# Columns and warps of a program streaming a mask row in float64 (its merge, _mask_backward).
_MASK_ROW = (2048, 8)


@triton.jit
def _chunk_exp(drop):
    """exp(z - m) of a chunk's float32 scores z from drop = m - z, m their maximum (one per
    row): NaN for a row of -inf only (and at m = +inf), which _chunk_sum and the invalid flags
    absorb. drop is the gap sums' own difference, so the exponent is one multiply, as the FMA
    it replaces."""
    return tl.exp2(drop * -_LOG2E)


@triton.jit
def _chunk_sum(e, m):
    """sum(e) per row in float64 (0 for a chunk of -inf only)."""
    return tl.where(m == -float("inf"), 0.0, tl.sum(e, 1).to(tl.float64))


@triton.jit
def _label_chunk(z, k, y, W, wbase, valid, LEAN: tl.constexpr, GAP: tl.constexpr,
                 MASS: tl.constexpr):  # fmt: skip
    """_label_partial's statistics of a chunk of scores z (float32); LEAN: every column is in
    the row and none is the label."""
    if LEAN:
        keep = valid
    else:
        keep = valid & (k != y)
        z = tl.where(keep, z, -float("inf"))
    m = tl.max(z, 1)
    drop = m[:, None] - z
    total = _chunk_sum(_chunk_exp(drop), m)
    invalid = ~(total < float("inf")) | ~(tl.abs(m) < _LEAN_MAX) & (m > -float("inf"))
    gap = total * 0
    mass = total * 0
    if GAP:
        if MASS:
            w = tl.load(W + wbase + k, keep, other=0).to(tl.float32)
        else:
            w = keep.to(tl.float32)
        gap = tl.sum(tl.where(w > 0, w * drop, 0.0), 1).to(tl.float64)
        mass = tl.sum(w, 1).to(tl.float64)
        invalid |= ~(gap < float("inf"))
    return m, total, invalid, gap, mass


@triton.jit(do_not_specialize=["GROUP"])
def _label_partial(Z, Y, W, P, GROUP, C: tl.constexpr, B: tl.constexpr, CHUNKS: tl.constexpr,
                   GAP: tl.constexpr, MASS: tl.constexpr, SZ: tl.constexpr):  # fmt: skip
    """A label row chunk's competitors: their maximum m, sum exp(z - m), whether the chunk is
    not ordinary (marking its row), and with GAP sum w_j (m - z_j) and sum w_j (class weights
    with MASS, else 1 per competitor)."""
    row, block = rows.place(CHUNKS)
    shift, shared = rows.shared_shift(Z, Z, row, SZ, SZ)
    j, k, valid, inside = rows.span(block, shift, shared, B, C)
    y = tl.minimum(tl.maximum(tl.load(Y + row + tl.arange(0, 1)).to(tl.int64), 0), C - 1)
    first = block * B - shift
    label = (y >= first) & (y < first + B)
    wbase = (row // GROUP) * C
    if inside & ~tl.reshape(label, ()):
        z = rows.load(Z, row, j, shift, valid, SZ, False).to(tl.float32)
        m, total, invalid, gap, mass = _label_chunk(z, k, y[:, None], W, wbase, valid, True,
                                                    GAP, MASS)  # fmt: skip
    else:
        z = rows.load(Z, row, j, shift, valid, SZ, True).to(tl.float32)
        m, total, invalid, gap, mass = _label_chunk(z, k, y[:, None], W, wbase, valid, False,
                                                    GAP, MASS)  # fmt: skip
    out = doubles(P) + (row * CHUNKS + block) * _LABEL_PART + tl.arange(0, 1)
    tl.store(out, m.to(tl.float64))
    tl.store(out + 1, total)
    tl.store(out + 2, invalid.to(tl.float64))
    if GAP:
        tl.store(out + 3, gap)
        tl.store(out + 4, mass)


@triton.jit
def _merge_labels(P, row, CHUNKS: tl.constexpr, CP: tl.constexpr, GAP: tl.constexpr):
    """_stream's results from a row's chunk partials (_label_partial)."""
    c = tl.arange(0, CP)
    here = c < CHUNKS
    part = doubles(P) + (row * CHUNKS + c) * _LABEL_PART
    m = tl.load(part, here, other=-float("inf"))
    top = tl.max(m, 0)
    s = tl.load(part + 1, here, other=0.0)
    total = tl.sum(tl.where(m > -float("inf"), s * exp(m - top), 0.0), 0)
    # Rows the chunks marked, and rows of masked competitors only, take the streaming kernel.
    general = (tl.max(tl.load(part + 2, here, other=0.0), 0) != 0) | ~(total > 0)
    maximum = tl.where(total == 0, -float("inf"), top)
    gap = tl.zeros([], tl.float64)
    mass = tl.zeros([], tl.float64)
    if GAP:
        weight = tl.load(part + 4, here, other=0.0)
        rise = tl.where(weight > 0, weight * (top - m), 0.0)
        gap = tl.sum(tl.load(part + 3, here, other=0.0) + rise, 0)
        mass = tl.sum(weight, 0)
    return maximum, total, gap, mass, top, general


@triton.jit
def _label_row(Z, W, P, row, wbase, y, C: tl.constexpr, B: tl.constexpr, CHUNKS: tl.constexpr,
               CP: tl.constexpr, FP32: tl.constexpr, GAP: tl.constexpr, MASS: tl.constexpr,
               POLICY: tl.constexpr, SZ: tl.constexpr):  # fmt: skip
    """_stream's results of a row: its CHUNKS partials merged, or (a row a chunk marked, or
    no partials) the row streamed."""
    if CHUNKS > 0:
        maximum, total, gap, mass, top, general = _merge_labels(P, row, CHUNKS, CP, GAP)
        if general:
            maximum, total, gap, mass, top = _stream(
                Z, W, row, wbase, y, C, B, FP32, GAP, MASS, POLICY, SZ
            )
    else:
        maximum, total, gap, mass, top = _stream(
            Z, W, row, wbase, y, C, B, FP32, GAP, MASS, POLICY, SZ
        )
    return maximum, total, gap, mass, top


@triton.jit
def _label_grad(
    z,
    k,
    y,
    w,
    center,
    factor,
    s,
    gy,
    eps,
    dead,
    C: tl.constexpr,
    FP32: tl.constexpr,
    SMOOTH: tl.constexpr,
    WEIGHTED: tl.constexpr,
):
    """The gradient of label-row columns k of scores z (storage or work type); w the class
    weights under smoothing with weights."""
    if FP32 and z.dtype != tl.float64 and z.dtype != tl.float32 and _FAST_GRAD:
        # 16-bit storage: exp2 of the float32 difference (its error is far below the
        # stored gradient's), tails scaled as exp_parts.
        x = z.to(tl.float32) - center.to(tl.float32)
        tiny = x < -75
        e = tl.exp2(tl.where(tiny, 0.5 * x, x) * _LOG2E)
        g = e * (tl.where(tiny, e, 1.0) * factor.to(tl.float32))
    elif FP32 and z.dtype != tl.float64:
        # Scale the tail before rounding it to float32. Either exponential can be
        # normal even when their product is subnormal and the upstream is large.
        e, tail = exp_parts(z.to(tl.float32) - center.to(tl.float32))
        g = e * (tail * factor.to(tl.float32))
    else:
        g = exp(z.to(tl.float64) - center) * factor
    if SMOOTH:
        if WEIGHTED:
            g = g - (s * eps / C).to(g.dtype) * w.to(g.dtype)
        else:
            g = g - (s * eps / C).to(g.dtype)
    g = tl.where(k == y, (s * gy).to(g.dtype), g)
    # + 0.0: zero entries are +0.0 for either sign of the upstream derivative.
    return tl.where(dead, 0.0, g) + 0.0


@triton.jit
def _dense_grad(
    z,
    t,
    center,
    factor,
    s,
    norm,
    anchor,
    correction,
    FP32: tl.constexpr,
    TARGET: tl.constexpr,
    BFLOAT_STORAGE: tl.constexpr = False,
):
    """At a maximum, subtract the local target from the average target on the maxima first.
    The correction carries the mass of the remaining classes even as softmax rounds to one."""
    target = t.to(tl.float64) * norm
    if FP32 and TARGET != MASK and z.dtype != tl.float64:
        prob = exp32(z.to(tl.float32) - center.to(tl.float32)).to(tl.float64)
        term = prob * factor
        refine = (target > 0) & (tl.abs(term - target * s) < 0.0625 * tl.abs(target * s))
        prob = tl.where(refine, exp(z.to(tl.float64) - center), prob)
    else:
        # The same double exponential as the mask partition retains positive
        # cancellation residuals and finite tails amplified by an upstream factor.
        prob = exp(z.to(tl.float64) - center)
    if TARGET == MASK and not FP32:
        # At positive scores near log(sum(exp(z)) / |Y|), both terms in the
        # numerator are small. The relative partition residual is stored there.
        d = z.to(tl.float64) - anchor
        centered = ((expm1(tl.minimum(d, 0.25)) - correction) / (1 + correction)) * norm
        g = tl.where((t > 0) & (tl.abs(d) < 0.125), centered * s, prob * factor - target * s)
        # All positives at the maximum use the original tail-balanced pivot gradient.
        return (
            tl.where((anchor == -float("inf")) & (t > 0) & (z == center), correction * s, g) + 0.0
        )
    g = tl.where(z == center, (anchor - target + correction) * s, prob * factor - target * s)
    if FP32 and TARGET == MASK and BFLOAT_STORAGE:
        # Keep work underflows positive; a negative that rounds to zero only in BF16 keeps its sign.
        g = tl.where(tl.abs(g) <= 2.0**-150, 0.0, g)
    return g + 0.0


@triton.jit
def _put_labels(
    Z,
    W,
    G,
    row,
    wbase,
    k,
    y,
    center,
    factor,
    s,
    gy,
    eps,
    dead,
    C: tl.constexpr,
    FP32: tl.constexpr,
    WEIGHTED: tl.constexpr,
    SMOOTH: tl.constexpr,
    MASKED: tl.constexpr,
    STORE: tl.constexpr,
    SZ: tl.constexpr,
    SG: tl.constexpr,
):
    """Write the gradient of columns k of a streamed label row (read before written)."""
    if MASKED:
        z = tl.load(Z + row * SZ + k, k < C, other=0, eviction_policy="evict_first")
    else:
        z = tl.load(Z + row * SZ + k, eviction_policy="evict_first")
    w = _weights(W, wbase, k, k < C, True) if SMOOTH and WEIGHTED else 1.0
    g = _label_grad(z, k, y, w, center, factor, s, gy, eps, dead, C, FP32, SMOOTH, WEIGHTED)
    if MASKED:
        tl.store(G + row * SG + k, g.to(G.dtype.element_ty), k < C, eviction_policy=STORE)
    else:
        tl.store(G + row * SG + k, g.to(G.dtype.element_ty), eviction_policy=STORE)


@triton.jit
def _put_row(
    Z,
    W,
    G,
    row,
    wbase,
    y,
    center,
    factor,
    s,
    gy,
    eps,
    dead,
    C: tl.constexpr,
    B: tl.constexpr,
    FP32: tl.constexpr,
    WEIGHTED: tl.constexpr,
    SMOOTH: tl.constexpr,
    STORE: tl.constexpr,
    SZ: tl.constexpr,
    SG: tl.constexpr,
):
    """Write the gradient of a streamed label row, last chunk first: the chunks read last are
    the likeliest still in L2."""
    k = tl.arange(0, B)
    FULL: tl.constexpr = C // B * B
    if FULL < C:
        _put_labels(
            Z, W, G, row, wbase, FULL + k, y, center, factor, s, gy, eps, dead,
            C, FP32, WEIGHTED, SMOOTH, True, STORE, SZ, SG,
        )  # fmt: skip
    for i in range(0, FULL // B):
        _put_labels(
            Z, W, G, row, wbase, FULL - B - i * B + k, y, center, factor, s, gy, eps, dead,
            C, FP32, WEIGHTED, SMOOTH, False, STORE, SZ, SG,
        )  # fmt: skip


@triton.jit
def _scaled_gap(center, z, weight):
    """Scale a nonnegative gap even when subtracting its finite endpoints overflows."""
    gap = center - z
    scaled = tl.where(gap == float("inf"), weight * center - weight * z, weight * gap)
    return tl.where(weight == 0, 0.0, scaled)


@triton.jit
def _smoothed_value(Z, W, row, live, center, z_y, first, value, spread, mass, w_y, eps, group,
                    C: tl.constexpr, B: tl.constexpr, ROWS: tl.constexpr, FP32: tl.constexpr,
                    WEIGHTED: tl.constexpr, SZ: tl.constexpr):  # fmt: skip
    """Apply the target weights before rounding an otherwise overflowing score gap."""
    label_weight = (1 - eps) * w_y
    label_term = tl.where(label_weight == 0, 0.0, label_weight * value)
    uniform = (eps / C) * (spread - mass * log(first))
    if not FP32:
        # The operands are (ROWS,) blocks in a tile and scalars in the one-row path
        # (ROWS == 1); broadcast_to splats a scalar, where reshape cannot.
        recover = tl.broadcast_to(live & ~finite(uniform), (ROWS,))
        if tl.max(recover.to(tl.int32), 0) != 0:
            row2 = tl.reshape(tl.broadcast_to(row, (ROWS,)), (ROWS, 1))
            base = tl.reshape(tl.broadcast_to(group, (ROWS,)), (ROWS, 1)) * C
            lanes = tl.zeros([ROWS, B], tl.float64)
            for start in range(0, C, B):
                j = start + tl.arange(0, B)[None, :]
                valid = recover[:, None] & (j < C)
                z = tl.load(Z + row2 * SZ + j, valid, other=0).to(tl.float64)
                scale = tl.reshape(tl.broadcast_to(eps / C, (ROWS,)), (ROWS, 1))
                q = _weights(W, base, j, valid, WEIGHTED) * scale
                lanes += _scaled_gap(tl.reshape(tl.broadcast_to(center, (ROWS,)), (ROWS, 1)), z, q)
            gaps = tl.reshape(tl.sum(lanes, 1), uniform.shape)
            uniform = tl.where(
                tl.reshape(recover, uniform.shape), gaps - ((eps / C) * mass) * log(first), uniform
            )
        label_term = tl.where(
            value == float("inf"),
            _scaled_gap(center, z_y, label_weight) - label_weight * log(first),
            label_term,
        )
    return label_term + uniform


@triton.jit
def _narrow_row(Z, y, live, C: tl.constexpr, LOAD: tl.constexpr):
    """The competitors' float32 maximum, float64 sum of float32 exponentials and the label's
    score of rows of C columns at Z, one row per thread: no lanes of a row to reduce across,
    and the float64 row statistics computed once per row (a 2-D tile replicates them across a
    row's lanes). The second pass rereads the rows from L1."""
    m = tl.full(y.shape, -float("inf"), tl.float32)
    z_y = tl.zeros(y.shape, tl.float32)
    for k in tl.static_range(C):
        z = tl.load(Z + k, live, other=0, eviction_policy=LOAD).to(tl.float32)
        z_y = tl.where(y == k, z, z_y)
        m = tl.where(y == k, m, tl.maximum(m, tl.where(z < float("inf"), z, float("inf"))))
    t = tl.zeros(y.shape, tl.float64)
    for k in tl.static_range(C):
        z = tl.load(Z + k, live, other=0, eviction_policy=LOAD).to(tl.float32)
        t += tl.where(y == k, 0.0, exp(z - m)).to(tl.float64)
    return m, tl.where(m == -float("inf"), 0.0, t), z_y


# _lane_row's sums of float32 scores (SPLIT): 1 each vector's terms split exactly (rows of at
# most _LANE_EXACT classes: memory-bound there, so free); wider rows 3, a lane's terms in
# float32 (values within a few float32 ulps). 2: an exact split per lane (as accurate as 1,
# but slower).
_LANE_SPLIT = 1
_LANE_WIDE_SPLIT = 3
_LANE_EXACT = 256
# _label_grad of 16-bit scores: exp2 of the float32 difference (1) or the corrected exp (0).
_FAST_GRAD = tl.constexpr(1)
# _lane_row's exponentials: 0 exp2 of one FMA, 1 libdevice exp of z - maximum.
_LANE_SUM = tl.constexpr(0)


@triton.jit
def _lane_row(Z, row, live, y, C: tl.constexpr, LANES: tl.constexpr, PV: tl.constexpr,
              ALIGNED: tl.constexpr, SZ: tl.constexpr, LOAD: tl.constexpr,
              SPLIT: tl.constexpr):  # fmt: skip
    """The competitors' float32 maximum and float64 sum of exp(z - maximum) of label rows
    `row` ([1, R, 1, 1], labels y) over a lane tile (rows.lane_load): every per-element
    operation in float32, the row maximum and sum combined across lanes in float32 (exp2 of
    one FMA against the rounded scaled maximum, whose rounding the float64 total corrects)."""
    x, k, valid = rows.lane_load(Z, row, live, SZ, C, PV, LANES, ALIGNED, POLICY=LOAD)
    z = x.to(tl.float32)
    other = valid & (k != y)
    # NaN counts as +inf (an invalid row).
    top = tl.where(other, tl.where(z < float("inf"), z, float("inf")), -float("inf"))
    m = rows.lane_max(top)
    if _LANE_SUM == 1:
        e = exp(tl.where(other, z - m, -float("inf")))
    else:
        # exp2 of the float32 difference times log2(e): no overflow at any finite maximum.
        e = tl.exp2(tl.where(other, z - m, -float("inf")) * _LOG2E)
    VEC: tl.constexpr = 128 // Z.dtype.element_ty.primitive_bitwidth
    if SPLIT == 2 and Z.dtype.element_ty == tl.float32:
        # A lane's terms (at most 1, PV * VEC of them) split at a multiple of ulp(bias): the
        # high parts add exactly in float32, the low parts (below ulp(bias) / 2) with
        # negligible error; float64 across lanes only.
        bias: tl.constexpr = 1.5 * PV * VEC
        high = (e + bias) - bias
        lane = tl.sum(tl.sum(high, 3, keep_dims=True), 0, keep_dims=True).to(tl.float64)
        lane += tl.sum(tl.sum(e - high, 3, keep_dims=True), 0, keep_dims=True).to(tl.float64)
    elif SPLIT == 1 and Z.dtype.element_ty == tl.float32:
        # Each vector's terms (at most 1) split at a multiple of ulp(bias), whose high parts
        # add exactly in float32; float64 above.
        bias1: tl.constexpr = 1.5 * VEC
        high = (e + bias1) - bias1
        vector = tl.sum(high, 3, keep_dims=True).to(tl.float64)
        vector += tl.sum(e - high, 3, keep_dims=True).to(tl.float64)
        lane = tl.sum(vector, 0, keep_dims=True)
    elif SPLIT == 3:
        # A lane's terms in float32, float64 across lanes.
        lane = tl.sum(tl.sum(e, 3, keep_dims=True), 0, keep_dims=True).to(tl.float64)
    else:
        # 16-bit scores: each vector's terms (one 16-byte load) in float32, the vectors in
        # float64 (values within a few float32 ulps).
        lane = tl.sum(tl.sum(e, 3, keep_dims=True).to(tl.float64), 0, keep_dims=True)
    total = tl.sum(lane, 2, keep_dims=True)
    return m, tl.where(m > -float("inf"), total, 0.0)


@triton.jit
def _looped_row(Z, row, y, C: tl.constexpr, B: tl.constexpr, LOOP: tl.constexpr,
                SZ: tl.constexpr):  # fmt: skip
    """The competitors' float32 maximum and float64 sum of exp(z - maximum) of a label row,
    read once by one program in LOOP aligned chunks of B columns (rows.py): each lane keeps
    its own running maximum and float32 sum of exp(z - maximum) (differences first, as
    _chunk_exp), so that no chunk waits on a reduction across lanes; the lanes combine once,
    in float64."""
    shift, shared = rows.shared_shift(Z, Z, row, SZ, SZ)
    top = tl.full([1, B], -float("inf"), tl.float32)
    s = tl.zeros([1, B], tl.float32)
    for block in range(LOOP):
        j, k, valid, inside = rows.span(block, shift, shared, B, C)
        if inside:
            z = rows.load(Z, row, j, shift, valid, SZ, False).to(tl.float32)
        else:
            z = rows.load(Z, row, j, shift, valid, SZ, True).to(tl.float32)
        keep = valid & (k != y)
        # NaN counts as +inf: the row is then invalid.
        z = tl.where(keep, tl.where(z < float("inf"), z, float("inf")), -float("inf"))
        old = top
        top = tl.maximum(top, z)
        s = s * tl.where(old == top, 1.0, tl.exp2((old - top) * _LOG2E))
        s += tl.where(z == -float("inf"), 0.0, tl.exp2((z - top) * _LOG2E))
    m = tl.max(top, 1)
    lane = tl.where(top == -float("inf"), -float("inf"), top.to(tl.float64) - m[:, None])
    part = s.to(tl.float64) * exp(lane)
    total = tl.where(m == -float("inf"), 0.0, tl.sum(part, 1))
    return tl.reshape(m, ()), tl.reshape(total, ())


@triton.jit
def _row_sum(x):
    """The sum of x over its rows (any rank, 0-d included), float64."""
    if len(x.shape) == 0:
        total = x.to(tl.float64)
    else:
        total = tl.sum(tl.ravel(x.to(tl.float64)), 0)
    return total


@triton.jit(do_not_specialize=["N", "GROUP", "IGNORE", "UP", "TILES"])
def _labels(
    Z,
    Y,
    W,
    E,
    R,
    D,
    V,
    S,
    G,
    N,
    GROUP,
    IGNORE,
    UP,
    C: tl.constexpr,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    FP32: tl.constexpr,
    WEIGHTED: tl.constexpr,
    IGNORED: tl.constexpr,
    SMOOTH: tl.constexpr,
    ROWWISE: tl.constexpr,
    DIVIDED: tl.constexpr,
    VALUES: tl.constexpr,
    STATS: tl.constexpr,
    GRAD: tl.constexpr,
    LOAD: tl.constexpr = "",
    STORE: tl.constexpr = "",
    COMPACT: tl.constexpr = False,
    ALIAS: tl.constexpr = False,
    P=None,
    CHUNKS: tl.constexpr = 0,
    CP: tl.constexpr = 1,
    SZ: tl.constexpr = 0,
    SG: tl.constexpr = 0,
    NARROW: tl.constexpr = False,
    LANES: tl.constexpr = 0,
    PV: tl.constexpr = 1,
    ALIGNED: tl.constexpr = False,
    PARTIAL: tl.constexpr = False,
    T=None,
    TILES=1,
    LOOP: tl.constexpr = 0,
    GROUPS: tl.constexpr = 1,
    SPLIT: tl.constexpr = 1,
):
    """Row values (times the label's weight), and the statistics (STATS) or the gradient
    (GRAD) of integer-label rows, ROWS rows per tile (_label_tile). PARTIAL: each program
    takes the tiles pid, pid + programs, ... below TILES and writes the sum of its values
    and of their label weights (the divisor of a mean) to T[pid] and T[programs + pid]
    instead of the values, which _reduce adds up (reduce.add, partials)."""
    if PARTIAL:
        total = tl.zeros([], tl.float64)
        count = tl.zeros([], tl.float64)
        for tile in range(tl.program_id(0), TILES, tl.num_programs(0)):
            value, weight = _label_tile(
                tile, Z, Y, W, E, R, D, V, S, G, N, GROUP, IGNORE, UP, C, B, ROWS, FP32,
                WEIGHTED, IGNORED, SMOOTH, ROWWISE, DIVIDED, VALUES, STATS, GRAD, LOAD, STORE,
                COMPACT, ALIAS, P, CHUNKS, CP, SZ, SG, NARROW, LANES, PV, ALIGNED, PARTIAL, LOOP,
                GROUPS, SPLIT,
            )  # fmt: skip
            total += value
            count += weight
        T = doubles(T)
        tl.store(T + tl.program_id(0), total)
        tl.store(T + tl.num_programs(0) + tl.program_id(0), count)
    else:
        _label_tile(
            tl.program_id(0), Z, Y, W, E, R, D, V, S, G, N, GROUP, IGNORE, UP, C, B, ROWS,
            FP32, WEIGHTED, IGNORED, SMOOTH, ROWWISE, DIVIDED, VALUES, STATS, GRAD, LOAD,
            STORE, COMPACT, ALIAS, P, CHUNKS, CP, SZ, SG, NARROW, LANES, PV, ALIGNED, PARTIAL,
            LOOP, GROUPS, SPLIT,
        )  # fmt: skip


@triton.jit
def _label_tile(  # noqa: C901
    tile,
    Z,
    Y,
    W,
    E,
    R,
    D,
    V,
    S,
    G,
    N,
    GROUP,
    IGNORE,
    UP,
    C: tl.constexpr,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    FP32: tl.constexpr,
    WEIGHTED: tl.constexpr,
    IGNORED: tl.constexpr,
    SMOOTH: tl.constexpr,
    ROWWISE: tl.constexpr,
    DIVIDED: tl.constexpr,
    VALUES: tl.constexpr,
    STATS: tl.constexpr,
    GRAD: tl.constexpr,
    LOAD: tl.constexpr,
    STORE: tl.constexpr,
    COMPACT: tl.constexpr,
    ALIAS: tl.constexpr,
    P,
    CHUNKS: tl.constexpr,
    CP: tl.constexpr,
    SZ: tl.constexpr,
    SG: tl.constexpr,
    NARROW: tl.constexpr,
    LANES: tl.constexpr,
    PV: tl.constexpr,
    ALIGNED: tl.constexpr,
    PARTIAL: tl.constexpr,
    LOOP: tl.constexpr,
    GROUPS: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """One tile of _labels: ROWS rows from tile * ROWS. ALIAS: the scores are G, which the
    gradient overwrites. CHUNKS: merge a row's CHUNKS partials P (_label_partial) instead of
    reading it, but for a row a chunk marked. NARROW: each thread owns whole rows of C columns
    (_narrow_row). LANES: rows over LANES lanes of 16-byte vectors (_lane_row; values and
    statistics only). LOOP: one row per program in LOOP chunks of B columns (_looped_row;
    values and statistics only). Returns the sum of the tile's values and of their label weights
    (PARTIAL; else zeros)."""
    if ALIAS:
        Z = G
    S = doubles(S)
    # Smoothing: the spread sum_j w_j (center - z_j) for values, the weights' sum.
    GAP: tl.constexpr = SMOOTH and VALUES
    MASS: tl.constexpr = SMOOTH and WEIGHTED
    REGISTERS: tl.constexpr = (ROWS > 1 or B >= C) and CHUNKS == 0 and LANES == 0 and LOOP == 0
    spread = 0.0
    if LANES > 0:
        # GROUPS (at most LANES) groups of R rows: group t reads its rows over the LANES lanes
        # of each row, and lane t keeps their statistics, so that each thread finishes at
        # most one row (all of them with GROUPS = LANES).
        GROUP_ROWS: tl.constexpr = ROWS // GROUPS
        r = tl.arange(0, GROUP_ROWS)[None, :, None, None]
        lane = tl.arange(0, LANES)[None, None, :, None]
        first = tl.cast(tile, tl.int64) * ROWS
        row = first + lane * GROUP_ROWS + r
        live = (row < N) & (lane < GROUPS)
        y, inside, ignored, w_y = target_row(Y, W, row, live, C, IGNORE, GROUP, WEIGHTED, IGNORED)
        z_y = tl.load(Z + row * SZ + y, live, other=0).to(tl.float64)
        m32 = tl.full(row.shape, -float("inf"), tl.float32)
        total = tl.zeros(row.shape, tl.float64)
        for t in tl.static_range(GROUPS):
            group = first + t * GROUP_ROWS + r
            here = group < N
            label = tl.load(Y + group, here, other=0).to(tl.int64)
            label = tl.minimum(tl.maximum(label, 0), C - 1).to(tl.int32)
            m_t, total_t = _lane_row(Z, group, here, label, C, LANES, PV, ALIGNED, SZ, LOAD, SPLIT)
            m32 = tl.where(lane == t, m_t, m32)
            total = tl.where(lane == t, total_t, total)
        maximum = m32.to(tl.float64)
        mass = tl.full(row.shape, C, tl.float64)
    elif LOOP > 0:
        row = tl.cast(tile, tl.int64)
        live = row < N
        y, inside, ignored, w_y = target_row(Y, W, row, live, C, IGNORE, GROUP, WEIGHTED, IGNORED)
        z_y = tl.load(Z + row * SZ + y, live, other=0).to(tl.float64)
        m32, total = _looped_row(Z, row, y, C, B, LOOP, SZ)
        maximum = m32.to(tl.float64)
        mass = tl.full([], C, tl.float64)
    elif NARROW:
        row = tl.cast(tile, tl.int64) * ROWS + tl.arange(0, ROWS)
        live = row < N
        y, inside, ignored, w_y = target_row(Y, W, row, live, C, IGNORE, GROUP, WEIGHTED, IGNORED)
        m32, total, zy = _narrow_row(Z + row * SZ, y, live, C, LOAD)
        z_y = zy.to(tl.float64)
        maximum = m32.to(tl.float64)
        mass = tl.full([ROWS], C, tl.float64)
    elif REGISTERS:
        row = tl.cast(tile, tl.int64) * ROWS + tl.arange(0, ROWS)
        live = row < N
        y, inside, ignored, w_y = target_row(Y, W, row, live, C, IGNORE, GROUP, WEIGHTED, IGNORED)
        z_y = tl.load(Z + row * SZ + y, live, other=0).to(tl.float64)
        j = tl.arange(0, B)[None, :]
        valid = live[:, None] & (j < C)
        z = work(tl.load(Z + row[:, None] * SZ + j, valid, other=0), FP32)
        other = valid & (j != y[:, None])
        checked = tl.where(z < float("inf"), z, float("inf"))
        maximum = tl.max(tl.where(other, checked, -float("inf")), 1)
        e = exp(tl.where(other, z - maximum[:, None], -float("inf")))
        total = tl.sum(e.to(tl.float64), 1)
        total = tl.where(maximum == -float("inf"), 0.0, total)
        maximum = maximum.to(tl.float64)
        if MASS:
            w = _weights(W, (row // GROUP)[:, None] * C, j, valid, True)
            mass = tl.sum(w, 1)
            if GAP:
                center = tl.maximum(maximum, z_y)[:, None]
                spread = tl.sum(tl.where(w > 0, w * (center - z.to(tl.float64)), 0.0), 1)
        else:
            w = 1.0
            mass = tl.full([ROWS], C, tl.float64)
            if GAP:
                center = tl.maximum(maximum, z_y)[:, None]
                spread = tl.sum(tl.where(valid, center - z.to(tl.float64), 0.0), 1)
        tile = z, j, w, valid
    else:
        row = tl.cast(tile, tl.int64)
        live = row < N
        y, inside, ignored, w_y = target_row(Y, W, row, live, C, IGNORE, GROUP, WEIGHTED, IGNORED)
        z_y = tl.load(Z + row * SZ + y, live, other=0).to(tl.float64)
        maximum, total, gap, mass, top = _label_row(
            Z, W, P, row, (row // GROUP) * C, y, C, B, CHUNKS, CP, FP32, GAP, MASS, LOAD, SZ
        )
        if not MASS:
            mass = tl.full([], C - 1, tl.float64)
        if GAP:
            # Every term is >= 0: the competitors' gaps below the running maximum, its rise
            # to the center, and the label's.
            center = tl.maximum(maximum, z_y)
            spread = gap + mass * (center - top) + tl.where(w_y > 0, w_y * (center - z_y), 0.0)
        mass = mass + (w_y if MASS else 1.0)
    # An ignored row has no label: whatever it holds, it is a zero row.
    z_y = tl.where(ignored, 0.0, z_y)
    bad = (maximum == float("inf")) | ~(z_y < float("inf"))
    bad = (bad | ((maximum == -float("inf")) & (z_y == -float("inf"))) | ~inside) & ~ignored
    value, mass_off, first = label_statistics(maximum - z_y, total, ROWS == 1)
    center = tl.maximum(maximum, z_y)
    # Zero times the infinite loss of a masked label: NaN, as in PyTorch.
    bad = bad | (~ignored & (w_y == 0) & (z_y == -float("inf")))
    if SMOOTH:
        # Under smoothing every class has target probability: the log-softmax of each one.
        eps = tl.load(E + row // GROUP, live, other=0).to(tl.float64)
        a = (1 - eps) * w_y + eps * mass / C
        gy = eps * (mass - w_y) / C - a * mass_off
        if GAP:
            value = _smoothed_value(
                Z, W, row, live & ~bad & ~ignored, center, z_y, first, value, spread, mass,
                w_y, eps, row // GROUP, C, B, ROWS, FP32, WEIGHTED, SZ,
            )  # fmt: skip
            bad = bad | ((value != value) & ~ignored)
        else:
            # The spread is NaN only where the row is already invalid or the weights are NaN.
            bad = bad | ((mass != mass) & ~ignored)
        zero = ignored
    else:
        eps = tl.full(row.shape, 0.0, tl.float64)
        a = w_y
        gy = -w_y * mass_off
        value = w_y * value
        zero = ignored | ((w_y == 0) & ~bad)
    # A zero row: no loss, a zero gradient (dead: a = gy = 0).
    value = tl.where(zero, 0.0, value)
    a = tl.where(ignored, 0.0, a)
    gy = tl.where(zero, 0.0, gy)
    nan = float("nan")
    center = tl.where(bad, nan, center)
    first = tl.where(bad, nan, first)
    a = tl.where(bad, nan, a)
    gy = tl.where(bad, nan, gy)
    value = tl.where(bad, nan, value)
    if VALUES and not PARTIAL:
        tl.store(V + row, value.to(V.dtype.element_ty), live)
    if STATS:
        if COMPACT:
            # An unweighted, unsmoothed row needs only the inverse partition and the
            # label's gradient. Zero inverse partition identifies an ignored row.
            tl.store(S + row * 2, tl.where(ignored, 0.0, first), live)
            tl.store(S + row * 2 + 1, gy, live)
        else:
            tl.store(S + row * 4, center, live)
            tl.store(S + row * 4 + 1, first, live)
            tl.store(S + row * 4 + 2, a, live)
            tl.store(S + row * 4 + 3, gy, live)
    if GRAD:
        s = scale(UP, R, D, row, live, ROWWISE, DIVIDED)
        factor = s * a * first
        dead = (a == 0) & (gy == 0)
        if NARROW:
            for k in tl.static_range(C):
                g = _label_grad(
                    tl.load(Z + row * SZ + k, live, other=0), k, y, 1.0, center, factor, s, gy,
                    0.0, dead, C, FP32, False, False,
                )  # fmt: skip
                tl.store(G + row * SG + k, g.to(G.dtype.element_ty), live, eviction_policy=STORE)
        elif REGISTERS:
            z, j, w, valid = tile
            g = _label_grad(
                z,
                j,
                y[:, None],
                w,
                center[:, None],
                factor[:, None],
                s[:, None],
                gy[:, None],
                eps[:, None] if SMOOTH else 0.0,
                dead[:, None],
                C,
                FP32,
                SMOOTH,
                WEIGHTED,
            )
            tl.store(G + row[:, None] * SG + j, g.to(G.dtype.element_ty), valid)
        else:
            _put_row(
                Z, W, G, row, (row // GROUP) * C, y, center, factor, s, gy, eps, dead,
                C, B, FP32, WEIGHTED, SMOOTH, STORE, SZ, SG,
            )  # fmt: skip
    values = tl.zeros([], tl.float64)
    weights = tl.zeros([], tl.float64)
    if PARTIAL:
        values = _row_sum(tl.where(live, value, 0.0))
        weights = _row_sum(tl.where(live, w_y, 0.0))
    return values, weights


@triton.jit
def _targets(
    P,
    W,
    offsets,
    wbase,
    k,
    valid,
    eps,
    C: tl.constexpr,
    FP32: tl.constexpr,
    TARGET: tl.constexpr,
    WEIGHTED: tl.constexpr,
    SMOOTH: tl.constexpr,
):
    """The targets t of columns k in the work type (masks: 0 or 1) and the invalid ones."""
    if TARGET == MASK:
        t = work((tl.load(P + offsets, valid, other=0) != 0).to(tl.float32), FP32)
        bad = valid & (t < 0)
    else:
        t = work(tl.load(P + offsets, valid, other=0), FP32)
        bad = valid & (~(t >= 0) | ~finite(t))
        if SMOOTH:
            t = (1 - eps) * t + eps / C
        if WEIGHTED:
            t = t * work(tl.load(W + wbase + k, valid, other=0), FP32)
    return tl.where(valid, t, 0.0), bad


@triton.jit
def _dense_chunk(
    Z,
    Y,
    W,
    row,
    group,
    live,
    k,
    eps,
    top,
    tail,
    ties,
    top_mass,
    rest,
    gap,
    flags,
    seen,
    C: tl.constexpr,
    FP32: tl.constexpr,
    TARGET: tl.constexpr,
    WEIGHTED: tl.constexpr,
    SMOOTH: tl.constexpr,
    SZ: tl.constexpr,
    DOUBLE: tl.constexpr = False,
):
    """Scores and targets of columns k, and the running statistics after them. DOUBLE: the
    exponentials in float64 (rows that _dense_row streams after their chunks marked them)."""
    valid = live[:, None] & (k < C)
    offsets = row[:, None] * C + k
    z = work(tl.load(Z + row[:, None] * SZ + k, valid, other=-float("inf")), FP32)
    t, bad = _targets(
        Y, W, offsets, group[:, None] * C, k, valid, eps, C, FP32, TARGET, WEIGHTED, SMOOTH
    )
    if TARGET == MASK:
        # A boolean target has no invalid weights. One maximum distinguishes
        # all-masked tiles, finite tiles, and tiles containing NaN or +inf.
        peak = tl.max(tl.where(z == z, z, float("inf")), 1)
        flags = flags | (peak == float("inf")).to(tl.int32)
        seen = seen | (peak > -float("inf")).to(tl.int32)
        new = tl.maximum(top, peak)
    else:
        bad = bad | (z != z) | (z == float("inf"))
        flags = flags | tl.max(bad.to(tl.int32), 1)
        seen = seen | tl.max((z > -float("inf")).to(tl.int32), 1)
        new = tl.maximum(top, tl.max(tl.where(z < float("inf"), z, -float("inf")), 1))
    rise = new > top
    # Store mask gaps on a power-of-two scale: even a sum of C full-range differences
    # can be averaged before its finite value overflows.
    gap_scale: tl.constexpr = 0.5 / triton.next_power_of_2(C) if TARGET == MASK else 1.0
    center = new.to(tl.float64) * gap_scale
    gap += tl.where(
        rise & ((rest + top_mass) > 0),
        (rest + top_mass) * (center - top.to(tl.float64) * gap_scale),
        0.0,
    )
    tail = tl.where(
        rise, (tail + ties * EXP_WIDE) * exp(top.to(tl.float64) - new.to(tl.float64)), tail
    )
    rest += tl.where(rise, top_mass, 0.0)
    ties = tl.where(rise, 0, ties)
    top_mass = tl.where(rise, 0.0, top_mass)
    at_top = valid & (z == new[:, None])
    if (TARGET == MASK and FP32) or DOUBLE:
        # A positive target can nearly cancel a probability even on an ordinary
        # row. Its normalization must then retain more than float32 accuracy.
        e = wide_exp(z.to(tl.float64) - new.to(tl.float64)[:, None])
    else:
        e = wide_exp(z - new[:, None]).to(tl.float64)
    tail += tl.sum(tl.where(at_top, 0.0, e), 1)
    ties += tl.sum(at_top.to(tl.int32), 1)
    drop = center[:, None] - z.to(tl.float64) * gap_scale
    if TARGET == MASK:
        positive = t > 0
        top_mass += tl.sum((at_top & positive).to(tl.int32), 1)
        rest += tl.sum((~at_top & positive).to(tl.int32), 1)
        gap += tl.sum(tl.where(positive, drop, 0.0), 1)
    else:
        top_mass += tl.sum(tl.where(at_top, t, 0.0).to(tl.float64), 1)
        rest += tl.sum(tl.where(at_top, 0.0, t).to(tl.float64), 1)
        gap += tl.sum(tl.where(t > 0, t * drop, 0.0), 1)
    return z, t, new, tail, ties, top_mass, rest, gap, flags, seen


@triton.jit
def _dense_row(Z, Y, W, P, row, group, live, eps, top, tail, ties, top_mass, rest, gap, flags,
               seen, C: tl.constexpr, B: tl.constexpr, CHUNKS: tl.constexpr, CP: tl.constexpr,
               FP32: tl.constexpr, TARGET: tl.constexpr, WEIGHTED: tl.constexpr,
               SMOOTH: tl.constexpr, SZ: tl.constexpr):  # fmt: skip
    """Rows' running statistics over chunks of B columns: with CHUNKS, one row's merged
    partials (_dense_partial), or that row streamed here where a chunk marked it."""
    general = tl.full([], True, tl.int1)
    if CHUNKS > 0:
        m_top, m_tail, m_ties, m_top_mass, m_rest, m_gap, general = _merge_dense(
            P, tl.program_id(0).to(tl.int64), CHUNKS, CP
        )
        if TARGET == MASK:
            # Public's rule: a row whose maximum is a negative class has a gradient of norm at
            # least that class's probability, which bounds the float32 pass's error normwise.
            # A positive at the maximum can cancel its gradient: the float64 row instead.
            general |= m_top_mass > 0
        else:
            # The chunks' float32 exponentials flush terms below exp(-87.3) of their maximum
            # (at most C 2**-126 of the row's sum). They matter only in a row whose other terms
            # sum to less than C 2**-102 (2**64-scaled tails), which streams instead with
            # float64 exponentials: class weights can scale even its smallest terms into range.
            general |= (m_tail < C * 2.0**-38) & (m_ties < C)
    if general:
        for start in range(0, C, B):
            k = start + tl.arange(0, B)[None, :]
            _, _, top, tail, ties, top_mass, rest, gap, flags, seen = _dense_chunk(
                Z, Y, W, row, group, live, k, eps, top, tail, ties, top_mass, rest, gap,
                flags, seen, C, FP32, TARGET, WEIGHTED, SMOOTH, SZ, CHUNKS > 0,
            )  # fmt: skip
    elif CHUNKS > 0:
        top = (top.to(tl.float64) * 0 + m_top).to(top.dtype)  # pyright: ignore[reportPossiblyUnboundVariable] -- set under CHUNKS > 0
        tail = tail * 0 + m_tail  # pyright: ignore[reportPossiblyUnboundVariable] -- set under CHUNKS > 0
        ties = ties * 0 + m_ties  # pyright: ignore[reportPossiblyUnboundVariable] -- set under CHUNKS > 0
        top_mass = top_mass * 0 + m_top_mass  # pyright: ignore[reportPossiblyUnboundVariable] -- set under CHUNKS > 0
        rest = rest * 0 + m_rest  # pyright: ignore[reportPossiblyUnboundVariable] -- set under CHUNKS > 0
        if TARGET == MASK:
            # _dense_chunk's power-of-two scale of mask gaps.
            gap = gap * 0 + m_gap * (0.5 / triton.next_power_of_2(C))  # pyright: ignore[reportPossiblyUnboundVariable] -- set under CHUNKS > 0
        else:
            gap = gap * 0 + m_gap  # pyright: ignore[reportPossiblyUnboundVariable] -- set under CHUNKS > 0
        seen = flags * 0 + 1
    return top, tail, ties, top_mass, rest, gap, flags, seen, ~general


@triton.jit
def _mask_residual(Z, row, live, pivot, mass, C: tl.constexpr, B: tl.constexpr,
                   SZ: tl.constexpr):  # fmt: skip
    """sum(exp(z - pivot)) / mass - 1, carrying the low parts through cancellation."""
    high = -mass
    low = tl.full(mass.shape, 0.0, tl.float64)
    for start in range(0, C, B):
        k = start + tl.arange(0, B)[None, :]
        x = tl.load(Z + row[:, None] * SZ + k, live[:, None] & (k < C), other=-float("inf")).to(
            tl.float64
        )
        d = x - pivot[:, None]
        rounded = d - x
        error = (x - (d - rounded)) - (pivot[:, None] + rounded)
        e = exp(d)
        tile, carry = tl.reduce((e, tl.where(finite(d), e * error, 0.0)), 1, add_parts)
        high, low = add_parts(high, low, tile, carry)
    return (high + low) / mass


_DENSE_PART = tl.constexpr(7)  # maximum, ties, tail, top mass, rest, gap, invalid


@triton.jit
def _dense_chunk_stats(z, t, valid, MASKED: tl.constexpr):
    """_dense_partial's statistics of a chunk of scores z and targets t (float32, -inf and 0
    off the row)."""
    m = tl.max(z, 1)
    drop = m[:, None] - z
    e = _chunk_exp(drop)
    at = z == m[:, None]
    if MASKED:
        at = at & valid
    ties = tl.sum(at.to(tl.int32), 1).to(tl.float64)
    # Scaled by 2**64 as _dense's tails (wide_exp).
    tail = _chunk_sum(tl.where(at, 0.0, e), m) * EXP_WIDE
    high = tl.sum(tl.where(at, t, 0.0), 1).to(tl.float64)
    low = tl.sum(tl.where(at, 0.0, t), 1).to(tl.float64)
    gap = tl.sum(tl.where(t > 0, t * drop, 0.0), 1).to(tl.float64)
    least = tl.min(t, 1)
    invalid = ~(tail < float("inf")) | ~(low + high < float("inf")) | ~(gap < float("inf"))
    invalid |= (least < 0) | ~(tl.abs(m) < _LEAN_MAX)
    return m, ties, tail, high, low, gap, invalid


@triton.jit
def _mask_parts(a0, a1, a2, a3, a4, b0, b1, b2, b3, b4):
    return a0 + b0, a1 + b1, a2 + b2, a3 + b3, tl.minimum(a4, b4)


@triton.jit
def _mask_difference(x, origin):
    """Float32 x - origin and its rounding residual."""
    gap = x - origin
    rounded = gap - x
    low = (x - (gap - rounded)) + (-origin - rounded)
    # Beyond this range the float32 exponential (or squared half) is zero. Also
    # discard the undefined residual of an overflowing finite-score difference.
    return gap, tl.where(tl.abs(gap) < 256, low, 0.0)


@triton.jit
def _mask_exp_difference(x, origin):
    """Float32 exponential of a represented difference, retaining its rounding residual."""
    gap, low = _mask_difference(x, origin.to(tl.float32))
    value = exp(gap)
    # |low| <= 2^-16 here: exp(low) = 1 + low to substantially better than float eps.
    return (value + value * low).to(tl.float64)


@triton.jit
def _mask_chunk_stats(z, pos, valid, MASKED: tl.constexpr, WIDEN: tl.constexpr):
    """_dense_chunk_stats of a mask chunk (positives pos) in one reduction after the maximum:
    the ties and positives at it (packed), the positives, the tail and the gap, and the least
    finite score (a term below e**-_SPREAD of the maximum marks the chunk)."""
    m = tl.max(z, 1)
    if WIDEN:
        e = _mask_exp_difference(z, m[:, None])
        # Formed in the reduction, as before: held across the exponential, the difference
        # costs float32 storage's registers (sm_80: 72 -> 80, seven CTAs per SM to six).
        drop = None
    else:
        drop = m[:, None] - z
        e = _chunk_exp(drop)
    at = z == m[:, None]
    if MASKED:
        at = at & valid
    packed = at.to(tl.int32) + 4096 * (at & pos).to(tl.int32)
    count, positives, tail, gap, least = tl.reduce(
        (
            packed,
            pos.to(tl.int32),
            tl.where(at, 0.0, e),
            tl.where(pos, m[:, None] - z if WIDEN else drop, 0.0),
            tl.where(z > -float("inf"), z, m[:, None]),
        ),
        1,
        _mask_parts,
    )
    ties = (count % 4096).to(tl.float64)
    high = (count // 4096).to(tl.float64)
    low = positives.to(tl.float64) - high
    tail = tl.where(m == -float("inf"), 0.0, tail.to(tl.float64)) * EXP_WIDE
    gap = gap.to(tl.float64)
    invalid = ~(tail < float("inf")) | ~(gap < float("inf")) | ~(tl.abs(m) < _LEAN_MAX)
    invalid |= least < m - _SPREAD
    return m, ties, tail, high, low, gap, invalid


@triton.jit(do_not_specialize=["GROUP"])
def _dense_partial(Z, Y, W, E, P, GROUP, C: tl.constexpr, B: tl.constexpr,
                   CHUNKS: tl.constexpr, WEIGHTED: tl.constexpr, SMOOTH: tl.constexpr,
                   SZ: tl.constexpr, TARGET: tl.constexpr = PROBABILITIES):  # fmt: skip
    """A probability (or mask: TARGET) row chunk's statistics in float32 work, as
    _dense_chunk's from an empty row: its maximum m, the ties at m, sum 2**64 exp(z - m) over
    the rest, the targets at m and elsewhere, sum t (m - z) over t > 0, and whether the chunk
    is not ordinary (NaN, +inf, a negative or nonfinite target, an infinite gap, extreme
    scores, a mask term below float32's normal range: _dense then computes its row alone)."""
    row, block = rows.place(CHUNKS)
    if TARGET == MASK:
        shift, shared = rows.mask_shift(Z, Y, row, SZ, C)
        j, k, valid, inside = rows.span(block, shift, shared, B, C)
        VEC: tl.constexpr = 128 // Z.dtype.element_ty.primitive_bitwidth
        if inside:
            z = rows.load(Z, row, j, shift, valid, SZ, False).to(tl.float32)
            pos = rows.load_mask(Y, row, j, shift, valid, VEC, C, False)
            m, ties, tail, high, low, gap, invalid = _mask_chunk_stats(
                z, pos, valid, False, Z.dtype.element_ty == tl.float32
            )
        else:
            z = rows.load(Z, row, j, shift, valid, SZ, True, -float("inf")).to(tl.float32)
            pos = rows.load_mask(Y, row, j, shift, valid, VEC, C, True)
            m, ties, tail, high, low, gap, invalid = _mask_chunk_stats(
                z, pos, valid, True, Z.dtype.element_ty == tl.float32
            )
        _store_dense_part(P, row, block, CHUNKS, m, ties, tail, high, low, gap, invalid)
        return
    shift, shared = rows.shared_shift(Z, Y, row, SZ, C)
    j, k, valid, inside = rows.span(block, shift, shared, B, C)
    group = row // GROUP
    if inside:
        z = rows.load(Z, row, j, shift, valid, SZ, False).to(tl.float32)
        t = rows.load(Y, row, j, shift, valid, C, False).to(tl.float32)
    else:
        z = rows.load(Z, row, j, shift, valid, SZ, True, -float("inf")).to(tl.float32)
        t = rows.load(Y, row, j, shift, valid, C, True).to(tl.float32)
    if SMOOTH:
        eps = tl.load(E + group).to(tl.float32)
        t = (1 - eps) * t + eps / C
    if WEIGHTED:
        t = t * tl.load(W + group * C + k, valid, other=0).to(tl.float32)
    if SMOOTH or not inside:
        t = tl.where(valid, t, 0.0)
    if inside:
        m, ties, tail, high, low, gap, invalid = _dense_chunk_stats(z, t, valid, False)
    else:
        m, ties, tail, high, low, gap, invalid = _dense_chunk_stats(z, t, valid, True)
    _store_dense_part(P, row, block, CHUNKS, m, ties, tail, high, low, gap, invalid)


@triton.jit
def _store_dense_part(P, row, block, CHUNKS: tl.constexpr, m, ties, tail, high, low, gap,
                      invalid):  # fmt: skip
    out = doubles(P) + (row * CHUNKS + block) * _DENSE_PART + tl.arange(0, 1)
    tl.store(out, m.to(tl.float64))
    tl.store(out + 1, ties)
    tl.store(out + 2, tail)
    tl.store(out + 3, high)
    tl.store(out + 4, low)
    tl.store(out + 5, gap)
    tl.store(out + 6, invalid.to(tl.float64))


@triton.jit
def _merge_dense(P, row, CHUNKS: tl.constexpr, CP: tl.constexpr):
    """_dense's running statistics after a whole row, from its chunk partials."""
    c = tl.arange(0, CP)
    here = c < CHUNKS
    part = doubles(P) + (row * CHUNKS + c) * _DENSE_PART
    m = tl.load(part, here, other=-float("inf"))
    top = tl.max(m, 0)
    at = here & (m == top)
    count = tl.load(part + 1, here, other=0.0)
    tail = tl.load(part + 2, here, other=0.0)
    high = tl.load(part + 3, here, other=0.0)
    low = tl.load(part + 4, here, other=0.0)
    some = here & (m > -float("inf"))
    down = tl.where(at, 1.0, exp(tl.where(some, m - top, 0.0)))
    tail = tl.sum(tl.where(some, tl.where(at, tail, (tail + count * EXP_WIDE) * down), 0.0), 0)
    ties = tl.sum(tl.where(at, count, 0.0), 0).to(tl.int32)
    top_mass = tl.sum(tl.where(at, high, 0.0), 0)
    rest = tl.sum(tl.where(at, 0.0, high) + low, 0)
    weight = high + low
    rise = tl.where(weight > 0, weight * (top - m), 0.0)
    gap = tl.sum(tl.where(some, tl.load(part + 5, here, other=0.0) + rise, 0.0), 0)
    general = (tl.max(tl.load(part + 6, here, other=0.0), 0) != 0) | (top == -float("inf"))
    return top, tail, ties, top_mass, rest, gap, general


@triton.jit(do_not_specialize=["N", "GROUP", "UP"])
def _dense(
    Z,
    Y,
    W,
    E,
    R,
    D,
    V,
    S,
    G,
    N,
    GROUP,
    UP,
    C: tl.constexpr,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    FP32: tl.constexpr,
    TARGET: tl.constexpr,
    WEIGHTED: tl.constexpr,
    SMOOTH: tl.constexpr,
    ROWWISE: tl.constexpr,
    DIVIDED: tl.constexpr,
    VALUES: tl.constexpr,
    STATS: tl.constexpr,
    GRAD: tl.constexpr,
    ALIAS: tl.constexpr = False,
    P=None,
    CHUNKS: tl.constexpr = 0,
    CP: tl.constexpr = 1,
    SZ: tl.constexpr = 0,
    SG: tl.constexpr = 0,
):
    """Row values, and the statistics (STATS) or the gradient (GRAD), of probability or mask
    targets in one pass: the running maximum rescales the float64 exponential sum and the
    targets' gaps below it. ALIAS as in _labels."""
    if ALIAS:
        Z = G
    S = doubles(S)
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = row < N
    group = row // GROUP
    eps = work(tl.load(E + group, live, other=0), FP32)[:, None] if SMOOTH else 0.0
    if FP32:
        top = tl.full([ROWS], -FLT_MAX, tl.float32)
    else:
        top = tl.full([ROWS], -DBL_MAX, tl.float64)
    tail = tl.zeros([ROWS], tl.float64)
    top_mass = tail
    rest = tail
    gap = tail
    ties = tl.zeros([ROWS], tl.int32)
    flags = tl.zeros([ROWS], tl.int32)
    seen = flags
    REGISTERS: tl.constexpr = B >= C and CHUNKS == 0
    if REGISTERS:
        k = tl.arange(0, B)[None, :]
        z, t, top, tail, ties, top_mass, rest, gap, flags, seen = _dense_chunk(
            Z, Y, W, row, group, live, k, eps, top, tail, ties, top_mass, rest, gap, flags, seen,
            C, FP32, TARGET, WEIGHTED, SMOOTH, SZ,
        )  # fmt: skip
        tile = z, t, k
    else:
        top, tail, ties, top_mass, rest, gap, flags, seen, merged = _dense_row(
            Z, Y, W, P, row, group, live, eps, top, tail, ties, top_mass, rest, gap, flags, seen,
            C, B, CHUNKS, CP, FP32, TARGET, WEIGHTED, SMOOTH, SZ,
        )  # fmt: skip
    mass = top_mass + rest
    tail *= EXP_UNWIDE
    total = ties + tail
    lognorm = log1p((ties - 1).to(tl.float64) + tail)
    if TARGET == MASK:
        rescale: tl.constexpr = 2.0 * triton.next_power_of_2(C)
        value = gap * (rescale / mass) + lognorm
        a = tl.zeros([ROWS], tl.float64) + 1.0
        fourth = 1.0 / mass
        flags = flags | (mass == 0).to(tl.int32)
    else:
        value = gap + mass * lognorm
        a = mass
        fourth = tl.full([ROWS], 1.0, tl.float64)
    bad = (flags != 0) | (seen == 0)
    nan = float("nan")
    center = tl.where(bad, nan, top.to(tl.float64))
    first = tl.where(bad, nan, 1.0 / total)
    a = tl.where(bad, nan, a)
    fourth = tl.where(bad, nan, fourth)
    anchor = (top_mass / ties) * fourth
    correction = (rest * fourth - anchor * tail) / total
    if TARGET == MASK and not FP32 and (GRAD or STATS):
        refine = (rest > 0) & ~bad & live
        top_gradient = (anchor - fourth) + correction
        anchor = tl.full([ROWS], -float("inf"), tl.float64)
        correction = top_gradient
        if tl.max(refine.to(tl.int32), 0) > 0:
            pivot = center + log(total / mass)
            adjust = _mask_residual(Z, row, live, pivot, mass, C, B, SZ)
            anchor = tl.where(refine, pivot, anchor)
            correction = tl.where(refine, adjust, correction)
    if TARGET == MASK and CHUNKS > 0:
        # merged (_dense_row): the lean chunk gradient computes this row (_LEAN_MASK).
        anchor = tl.where(merged, _LEAN_MASK, anchor)  # pyright: ignore[reportPossiblyUnboundVariable]
    if VALUES:
        tl.store(V + row, tl.where(bad, nan, value).to(V.dtype.element_ty), live)
    if STATS:
        tl.store(S + row * 6, center, live)
        tl.store(S + row * 6 + 1, first, live)
        tl.store(S + row * 6 + 2, a, live)
        tl.store(S + row * 6 + 3, fourth, live)
        tl.store(S + row * 6 + 4, anchor, live)
        tl.store(S + row * 6 + 5, correction, live)
    if GRAD:
        s = scale(UP, R, D, row, live, ROWWISE, DIVIDED)
        factor = (s * a * first)[:, None]
        if REGISTERS:
            z, t, k = tile  # pyright: ignore[reportPossiblyUnboundVariable] -- same constexpr
            g = _dense_grad(
                z,
                t,
                center[:, None],
                factor,
                s[:, None],
                fourth[:, None],
                anchor[:, None],
                correction[:, None],
                FP32,
                TARGET,
                BFLOAT_STORAGE=Z.dtype.element_ty == tl.bfloat16,
            )
            tl.store(G + row[:, None] * SG + k, g.to(G.dtype.element_ty), live[:, None] & (k < C))
        else:
            # Last chunk first: the chunks read last are the likeliest still in L2.
            BLOCKS: tl.constexpr = (C + B - 1) // B
            for i in range(0, BLOCKS):
                k = (BLOCKS - 1 - i) * B + tl.arange(0, B)[None, :]
                valid = live[:, None] & (k < C)
                offsets = row[:, None] * C + k
                z = tl.load(Z + row[:, None] * SZ + k, valid, other=-float("inf"))
                t, _ = _targets(
                    Y, W, offsets, group[:, None] * C, k, valid, eps, C, FP32, TARGET,
                    WEIGHTED, SMOOTH,
                )  # fmt: skip
                g = _dense_grad(
                    z,
                    t,
                    center[:, None],
                    factor,
                    s[:, None],
                    fourth[:, None],
                    anchor[:, None],
                    correction[:, None],
                    FP32,
                    TARGET,
                    BFLOAT_STORAGE=Z.dtype.element_ty == tl.bfloat16,
                )
                tl.store(G + row[:, None] * SG + k, g.to(G.dtype.element_ty), valid)


@triton.jit
def _gradient(
    Z,
    Y,
    W,
    G,
    row,
    wbase,
    k,
    valid,
    y,
    center,
    factor,
    s,
    fourth,
    anchor,
    correction,
    eps,
    dead,
    C: tl.constexpr,
    FP32: tl.constexpr,
    TARGET: tl.constexpr,
    WEIGHTED: tl.constexpr,
    SMOOTH: tl.constexpr,
    SZ: tl.constexpr,
    SG: tl.constexpr,
):
    """Write the gradient of the loaded columns k (where valid) of rows row."""
    z = tl.load(Z + row * SZ + k, valid, other=0)
    if TARGET == LABELS:
        w = _weights(W, wbase, k, valid, True) if SMOOTH and WEIGHTED else 1.0
        g = _label_grad(z, k, y, w, center, factor, s, fourth, eps, dead, C, FP32, SMOOTH, WEIGHTED)
    else:
        t, _ = _targets(
            Y, W, row * C + k, wbase, k, valid, work(eps, FP32), C, FP32, TARGET, WEIGHTED, SMOOTH
        )
        g = _dense_grad(
            z,
            t,
            center,
            factor,
            s,
            fourth,
            anchor,
            correction,
            FP32,
            TARGET,
            BFLOAT_STORAGE=Z.dtype.element_ty == tl.bfloat16,
        )
    tl.store(G + row * SG + k, g.to(G.dtype.element_ty), valid)


@triton.jit
def _row_scale(UP, R, D, row, live, ROWWISE: tl.constexpr, DIVIDED: tl.constexpr):
    """math.scale with its row-independent part (upstream, a shared row_upstream, the
    divisor) computed once per thread: the same float64 operations in the same order."""
    if ROWWISE == 1:
        s = scale(UP, R, D, row, live, ROWWISE, DIVIDED)
    else:
        one = f64(UP)
        if ROWWISE == 2:
            one = one * tl.load(R).to(tl.float64)
        if DIVIDED:
            one = one / tl.load(D).to(tl.float64)
        s = one + tl.zeros(row.shape, tl.float64)
    return s


@triton.jit(do_not_specialize=["N", "GROUP", "UP"])
def _backward(
    Z,
    Y,
    W,
    E,
    R,
    D,
    S,
    G,
    N,
    GROUP,
    UP,
    C: tl.constexpr,
    B: tl.constexpr,
    TILES: tl.constexpr,
    ROWS: tl.constexpr,
    FP32: tl.constexpr,
    TARGET: tl.constexpr,
    WEIGHTED: tl.constexpr,
    SMOOTH: tl.constexpr,
    ROWWISE: tl.constexpr,
    DIVIDED: tl.constexpr,
    COMPACT: tl.constexpr = False,
    ALIAS: tl.constexpr = False,
    SZ: tl.constexpr = 0,
    SG: tl.constexpr = 0,
):
    """The gradient from the row statistics: B * TILES columns of a row, or ROWS rows of B
    columns, per program. Each block is read before it is written (G may be Z)."""
    if ALIAS:
        Z = G
    S = doubles(S)
    if ROWS > 1:
        row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        k = tl.arange(0, B)[None, :]
    else:
        row = tl.program_id(0).to(tl.int64)
        k = tl.program_id(1) * (B * TILES) + tl.arange(0, B)
    live = row < N
    if COMPACT:
        valid = live[:, None] & (k < C)
        z = tl.load(Z + row[:, None] * SZ + k, valid, other=-float("inf"))
        center = tl.max(z, 1).to(tl.float32)
        y = tl.minimum(tl.maximum(tl.load(Y + row, live, other=0), 0), C - 1).to(tl.int32)
        first = tl.load(S + row * 2, live, other=0.0)
        gy = tl.load(S + row * 2 + 1, live, other=0.0)
        s = _row_scale(UP, R, D, row, live, ROWWISE, DIVIDED)
        # The row factors rounded to float32 once per row (_label_grad rounds them so), not
        # once per element of the broadcast tile.
        g = _label_grad(
            z,
            k,
            y[:, None],
            1.0,
            center[:, None],
            (s * first).to(tl.float32)[:, None],
            1.0,
            (s * gy).to(tl.float32)[:, None],
            0.0,
            (first == 0.0)[:, None],
            C,
            FP32,
            False,
            False,
        )
        tl.store(G + row[:, None] * SG + k, g.to(G.dtype.element_ty), valid)
        return
    WIDTH: tl.constexpr = 4 if TARGET == LABELS else 6
    center = tl.load(S + row * WIDTH, live, other=0.0)
    first = tl.load(S + row * WIDTH + 1, live, other=0.0)
    a = tl.load(S + row * WIDTH + 2, live, other=0.0)
    fourth = tl.load(S + row * WIDTH + 3, live, other=0.0)
    s = _row_scale(UP, R, D, row, live, ROWWISE, DIVIDED)
    factor = s * a * first
    group = row // GROUP
    eps = tl.load(E + group, live, other=0).to(tl.float64) if SMOOTH else s * 0
    if TARGET == LABELS:
        y = tl.minimum(tl.maximum(tl.load(Y + row, live, other=0).to(tl.int64), 0), C - 1)
        anchor = a * 0.0
        correction = anchor
        dead = (a == 0) & (fourth == 0)
    else:
        anchor = tl.load(S + row * WIDTH + 4, live, other=0.0)
        correction = tl.load(S + row * WIDTH + 5, live, other=0.0)
        y = row * 0
        dead = a < -float("inf")
    if ROWS > 1:
        _gradient(
            Z,
            Y,
            W,
            G,
            row[:, None],
            group[:, None] * C,
            k,
            live[:, None] & (k < C),
            y[:, None],
            center[:, None],
            factor[:, None],
            s[:, None],
            fourth[:, None],
            anchor[:, None],
            correction[:, None],
            eps[:, None],
            dead[:, None],
            C,
            FP32,
            TARGET,
            WEIGHTED,
            SMOOTH,
            SZ,
            SG,
        )
    else:
        for t in tl.static_range(TILES):
            kt = k + t * B
            _gradient(
                Z,
                Y,
                W,
                G,
                row,
                group * C,
                kt,
                kt < C,
                y,
                center,
                factor,
                s,
                fourth,
                anchor,
                correction,
                eps,
                dead,
                C,
                FP32,
                TARGET,
                WEIGHTED,
                SMOOTH,
                SZ,
                SG,
            )


@triton.jit
def _lane_grad(Z, G, x, k, y, center, factor, s, gy, dead, valid, C: tl.constexpr,
               FP32: tl.constexpr, COMPACT: tl.constexpr):  # fmt: skip
    """_backward's label gradient of a lanes tile x ([PV, R, LANES, VEC]; columns k), its
    center recovered from the tile (COMPACT)."""
    if COMPACT:
        top = tl.where(valid, x.to(tl.float32), -float("inf"))
        top = tl.max(tl.max(tl.max(top, 3, keep_dims=True), 0, keep_dims=True), 2, keep_dims=True)
        center = top.to(tl.float64)
    g = _label_grad(x, k, y, 1.0, center, factor, s, gy, 0.0, dead, C, FP32, False, False)
    return g.to(G.dtype.element_ty)


@triton.jit(do_not_specialize=["N", "UP"])
def _lane_backward(Z, Y, R, D, S, G, N, UP, C: tl.constexpr, LANES: tl.constexpr,
                   PV: tl.constexpr, ROWS: tl.constexpr, FP32: tl.constexpr,
                   ROWWISE: tl.constexpr, DIVIDED: tl.constexpr, COMPACT: tl.constexpr = False,
                   ALIGNED: tl.constexpr = False, ALIAS: tl.constexpr = False,
                   SZ: tl.constexpr = 0, SG: tl.constexpr = 0):  # fmt: skip
    """_backward of unsmoothed label rows over the lanes tiles of _lane_row, ROWS rows per
    program: whole 16-byte vectors loaded and stored where the scores and the gradient share
    their positions (a row's partial first and last vectors are stored by column), else
    every column by itself. Each block is read before it is written (G may be Z)."""
    if ALIAS:
        Z = G
    S = doubles(S)
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)[None, :, None, None]
    live = row < N
    y = tl.minimum(tl.maximum(tl.load(Y + row, live, other=0).to(tl.int64), 0), C - 1)
    if COMPACT:
        first = tl.load(S + row * 2, live, other=0.0)
        gy = tl.load(S + row * 2 + 1, live, other=0.0)
        a = first * 0 + 1.0
        center = first
        dead = first == 0.0
    else:
        center = tl.load(S + row * 4, live, other=0.0)
        first = tl.load(S + row * 4 + 1, live, other=0.0)
        a = tl.load(S + row * 4 + 2, live, other=0.0)
        gy = tl.load(S + row * 4 + 3, live, other=0.0)
        dead = (a == 0) & (gy == 0)
    s = scale(UP, R, D, row, live, ROWWISE, DIVIDED)
    factor = s * a * first
    VEC: tl.constexpr = 128 // Z.dtype.element_ty.primitive_bitwidth
    zs = Z + row * SZ
    gs = G + row * SG
    lead = (tl.arange(0, PV)[:, None, None, None] * LANES
            + tl.arange(0, LANES)[None, None, :, None]) * VEC  # fmt: skip
    j = lead + tl.arange(0, VEC)[None, None, None, :]
    shift = rows.lane_shift(zs)
    same = (shift == rows.lane_shift(gs)) | ~live
    if ALIGNED:
        same &= (shift == 0) | ~live
    if tl.min(same.to(tl.int32)) == 1:
        k = j - shift
        valid = live & (k >= 0) & (k < C)
        x = tl.load(tl.multiple_of(zs - shift, (16, 16, 16, 16)) + j, live & (lead < shift + C),
                    other=0)  # fmt: skip
        g = _lane_grad(Z, G, x, k, y, center, factor, s, gy, dead, valid, C, FP32, COMPACT)
        whole = live & (lead >= shift) & (lead + VEC <= shift + C)
        base = tl.multiple_of(gs - shift, (16, 16, 16, 16))
        tl.store(base + j, g, whole)
        tl.store(base + j, g, valid & ~whole)
    else:
        k = j + (row * 0).to(tl.int32)
        valid = live & (k < C)
        x = tl.load(zs + k, valid, other=0)
        g = _lane_grad(Z, G, x, k, y, center, factor, s, gy, dead, valid, C, FP32, COMPACT)
        tl.store(gs + k, g, valid)


@triton.jit
def _lean_dense_grad(z, t, center, factor, s, fourth, anchor, correction, REFINE: tl.constexpr):
    """_dense_grad of probability targets in float32 arithmetic. 16-bit storage (REFINE 0):
    exp2 of one FMA with the row factor in the exponent (its rounding corrected in the scale),
    as accurate as the storage. Float32 storage: _dense_grad itself for a chunk with a class
    whose gradient cancels to below REFINE of its target term, or a row factor outside
    float32's comfortable range."""
    zf = z.to(tl.float32)
    c32 = center.to(tl.float32)
    s32 = s.to(tl.float32)
    target = t.to(tl.float32)
    if REFINE == 0:
        q = center * _LOG2E - tl.log2(tl.abs(factor))
        qf = q.to(tl.float32)
        # The rounding of q to float32 exceeds 64 only where |q| > 2**30: every 16-bit score
        # but the row's maxima is then at least 2**21 below them, and its term is exactly 0.
        # The clamp keeps tilt finite there (0 * inf would give NaN).
        tilt = tl.exp2(tl.minimum(tl.maximum(qf.to(tl.float64) - q, -64.0), 64.0).to(tl.float32))
        unit = tl.where(factor < 0, -tilt, tilt)
        unit = tl.where(factor == 0, 0.0, unit)
        term = tl.where(target > 0, target * s32, 0.0)
        g = tl.fma(tl.exp2(tl.fma(zf, _LOG2E, -qf)), unit, -term)
    else:
        target = target * fourth.to(tl.float32)
        term = target * s32
        g = exp32(zf - c32) * factor.to(tl.float32) - term
    g = tl.where(zf == c32, (anchor.to(tl.float32) - target + correction.to(tl.float32)) * s32, g)
    if REFINE > 0:
        bound = tl.maximum(tl.abs(s), tl.abs(factor))
        safe = (bound <= 2.0**60) & ((bound >= 2.0**-60) | (s == 0))
        need = ~safe | ((target > 0) & (tl.abs(g) < REFINE * tl.abs(term)))
        if tl.max(tl.max(need.to(tl.int32), 1), 0) != 0:
            g = _dense_grad(z, t, center, factor, s, fourth, anchor, correction, True,
                            PROBABILITIES).to(tl.float32)  # fmt: skip
    return g + 0.0


@triton.jit
def _chunk_grad(Z, Y, W, G, row, j, k, shift, valid, wbase, y, center, factor, s, fourth,
                anchor, correction, eps, dead, C: tl.constexpr, FP32: tl.constexpr,
                TARGET: tl.constexpr, WEIGHTED: tl.constexpr, SMOOTH: tl.constexpr,
                REFINE: tl.constexpr, MASKED: tl.constexpr, SZ: tl.constexpr,
                SG: tl.constexpr, LOADS: tl.constexpr):  # fmt: skip
    """The gradient of a chunk, stored at G's positions (whole vectors unless MASKED); LOADS
    masked: the scores and targets do not share them (strided scores), and load by column."""
    z = rows.load(Z, row, j, shift, valid, SZ, LOADS)
    if TARGET == LABELS:
        w = tl.load(W + wbase + k, valid, other=0) if SMOOTH and WEIGHTED else 1.0
        g = _label_grad(z, k, y, w, center, factor, s, fourth, eps, dead, C, FP32, SMOOTH,
                        WEIGHTED)  # fmt: skip
    elif TARGET == MASK and Z.dtype.element_ty == tl.float32:
        pos = rows.load_mask(Y, row, j, shift, valid, 4, C, LOADS)
        # A large upstream can bring an underflowed probability back into range.
        if tl.max(tl.abs(factor)) > 1.0:
            prob = exp(z.to(tl.float64) - center)
        else:
            prob = _mask_exp_difference(z, center)
        g = prob * factor - tl.where(pos, fourth * s, 0.0) + 0.0
    else:
        if TARGET == MASK:
            # The positives' target 1 / |Y| (fourth): _lean_dense_grad's probability.
            VEC: tl.constexpr = 128 // Z.dtype.element_ty.primitive_bitwidth
            t = rows.load_mask(Y, row, j, shift, valid, VEC, C, LOADS).to(tl.float32)
            t = t * fourth.to(tl.float32)
            fourth = fourth * 0 + 1.0
        else:
            t = rows.load(Y, row, j, shift, valid, C, LOADS).to(tl.float32)
        if SMOOTH:
            t = (1 - eps.to(tl.float32)) * t + eps.to(tl.float32) / C
        if WEIGHTED:
            t = t * tl.load(W + wbase + k, valid, other=0).to(tl.float32)
        t = tl.where(valid, t, 0.0)
        g = _lean_dense_grad(z, t, center, factor, s, fourth, anchor, correction, REFINE)
    rows.store(G, row, j, shift, valid, g, SG, MASKED)


@triton.jit(do_not_specialize=["N", "GROUP", "UP"])
def _chunk_backward(Z, Y, W, E, R, D, S, G, N, GROUP, UP, C: tl.constexpr, B: tl.constexpr,
                    FP32: tl.constexpr, TARGET: tl.constexpr, WEIGHTED: tl.constexpr,
                    SMOOTH: tl.constexpr, ROWWISE: tl.constexpr, DIVIDED: tl.constexpr,
                    CHUNKS: tl.constexpr, REFINE: tl.constexpr = 0.0625,
                    ALIAS: tl.constexpr = False, SZ: tl.constexpr = 0,
                    SG: tl.constexpr = 0, STRIDED: tl.constexpr = False):  # fmt: skip
    """_backward of a wide float32-work label, probability or lean mask row (_LEAN_MASK) over
    aligned row chunks
    (rows.py): one program per (row, chunk), each read before it is written (G may be Z).
    STRIDED (row-strided scores): chunks take G's positions, and scores (and targets) whose
    rows have another shift load by column; else rows not sharing one shift are masked."""
    if ALIAS:
        Z = G
    S = doubles(S)
    row, block = rows.place(CHUNKS)
    one = row + tl.arange(0, 1)
    live = one < N
    if STRIDED:
        shift = rows.misaligned(G, row, SG)
        same = rows.misaligned(Z, row, SZ) == shift
        if TARGET == PROBABILITIES:
            _, aligned = rows.shared_shift(G, Y, row, SG, C)
            same &= aligned
        elif TARGET == MASK:
            _, aligned = rows.mask_shift(G, Y, row, SG, C)
            same &= aligned
        j, k, valid, inside = rows.span(block, shift, shift == shift, B, C)
    elif TARGET == LABELS:
        shift, same = rows.shared_shift(Z, G, row, SZ, SG)
        j, k, valid, inside = rows.span(block, shift, same, B, C)
    else:
        if TARGET == MASK:
            shift, same = rows.mask_shift(Z, Y, row, SZ, C)
        else:
            shift, same = rows.shared_shift(Z, Y, row, SZ, C)
        _, also = rows.shared_shift(Z, G, row, SZ, SG)
        same &= also
        shift = tl.where(same, shift, 0)
        j, k, valid, inside = rows.span(block, shift, same, B, C)
    WIDTH: tl.constexpr = 4 if TARGET == LABELS else 6
    state = S + one * WIDTH
    center = tl.load(state)
    first = tl.load(state + 1)
    a = tl.load(state + 2)
    fourth = tl.load(state + 3)
    s = scale(UP, R, D, one, live, ROWWISE, DIVIDED)
    factor = s * a * first
    group = one // GROUP
    eps = tl.load(E + group).to(tl.float64) if SMOOTH else s * 0
    if TARGET == LABELS:
        y = tl.minimum(tl.maximum(tl.load(Y + one).to(tl.int64), 0), C - 1)
        anchor = a * 0.0
        correction = anchor
        dead = (a == 0) & (fourth == 0)
    else:
        anchor = tl.load(state + 4)
        correction = tl.load(state + 5)
        y = one * 0
        dead = a < -float("inf")
        if TARGET == MASK:
            # Rows of the float64 path are _mask_backward's (a separate launch).
            if tl.reshape(anchor != _LEAN_MASK, ()):
                return
            anchor = anchor * 0.0
    if inside & same:
        _chunk_grad(
            Z, Y, W, G, row, j, k, shift, valid, group[:, None] * C, y[:, None],
            center[:, None], factor[:, None], s[:, None], fourth[:, None], anchor[:, None],
            correction[:, None], eps[:, None], dead[:, None], C, FP32, TARGET, WEIGHTED,
            SMOOTH, REFINE, False, SZ, SG, False,
        )  # fmt: skip
    elif STRIDED and inside:
        _chunk_grad(
            Z, Y, W, G, row, j, k, shift, valid, group[:, None] * C, y[:, None],
            center[:, None], factor[:, None], s[:, None], fourth[:, None], anchor[:, None],
            correction[:, None], eps[:, None], dead[:, None], C, FP32, TARGET, WEIGHTED,
            SMOOTH, REFINE, False, SZ, SG, True,
        )  # fmt: skip
    else:
        _chunk_grad(
            Z, Y, W, G, row, j, k, shift, valid, group[:, None] * C, y[:, None],
            center[:, None], factor[:, None], s[:, None], fourth[:, None], anchor[:, None],
            correction[:, None], eps[:, None], dead[:, None], C, FP32, TARGET, WEIGHTED,
            SMOOTH, REFINE, True, SZ, SG, True,
        )  # fmt: skip


@triton.jit(do_not_specialize=["N", "GROUP", "UP"])
def _mask_backward(Z, Y, W, E, R, D, S, G, N, GROUP, UP, C: tl.constexpr, B: tl.constexpr,
                   FP32: tl.constexpr, ROWWISE: tl.constexpr, DIVIDED: tl.constexpr,
                   ALIAS: tl.constexpr = False, SZ: tl.constexpr = 0,
                   SG: tl.constexpr = 0):  # fmt: skip
    """The float64 gradient (_dense_grad) of the wide mask rows _chunk_backward leaves (not
    _LEAN_MASK): one program per row over its chunks; other rows return at once."""
    if ALIAS:
        Z = G
    S = doubles(S)
    row = tl.program_id(0).to(tl.int64)
    anchor = tl.load(S + row * 6 + 4)
    if anchor == _LEAN_MASK:
        return
    center = tl.load(S + row * 6)
    first = tl.load(S + row * 6 + 1)
    a = tl.load(S + row * 6 + 2)
    fourth = tl.load(S + row * 6 + 3)
    correction = tl.load(S + row * 6 + 5)
    s = scale(UP, R, D, row, row < N, ROWWISE, DIVIDED)
    factor = s * a * first
    for start in range(0, C, B):
        k = start + tl.arange(0, B)
        _gradient(
            Z, Y, W, G, row, 0, k, k < C, 0, center, factor, s, fourth, anchor, correction,
            s * 0, a < -float("inf"), C, FP32, MASK, False, False, SZ, SG,
        )  # fmt: skip


def rows_per_program(c: int) -> int:
    """Pack complete rows of at most 128 lanes into a 512-element tile; wider rows take a
    program each (packing two wider rows per program is slower)."""
    block = max(16, pow2(c))
    return max(1, 512 // block) if block <= 128 else 1


# Label rows (float32 work, no class weights or smoothing) of fewer than _NARROW classes, or
# of at most _REGISTER_ROW classes whose rows cannot be read as 16-byte vectors, take one row
# per thread (_labels NARROW, _NARROW_ROWS rows per program of _NARROW_WARPS warps). A packed
# 2-D tile spreads such a row over up to 32 lanes, which all repeat its float32 and float64
# statistics, which makes such a forward several times slower. Joint gradients use it only
# below _NARROW (strided stores).
_NARROW, _NARROW_ROWS, _NARROW_WARPS = 32, 128, 4
_REGISTER_ROW = 128
_ITEMSIZE = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}
# Elements of a packed backward tile of rows of at most 128 classes (4 warps).
_BACKWARD_TILE = 2048


def _narrow(call: Call) -> bool:
    if call.dense or call.mask or call.weighted or call.smoothed or not call.fp32:
        return False
    if call.c < _NARROW:
        return True
    row_bytes = (call.stride or call.c) * _ITEMSIZE.get(call.dtype, 16)
    return call.c <= _REGISTER_ROW and row_bytes % 16 != 0 and not call.grad


# Label rows of _NARROW to _LANE_MAX classes (float32 work, no smoothing; values and
# statistics) run as [PV, R, LANES, VEC] tiles of 16-byte vectors (_lane_row): about
# _LANE_ELEMS columns per lane, LANES lanes per row, _LANE_THREADS threads per program.
# _LANE_MAX by storage bytes (float32 lane sums above _LANE_EXACT classes).
_LANE_MAX = {2: 1024, 4: 1024}


def _lane_max(call: Call) -> int:
    return _LANE_MAX.get(_ITEMSIZE.get(call.dtype, 8), 0)


_LANE_ELEMS = {2: 32, 4: 32}
_LANE_THREADS = 128


# Label rows of fewer than _STATELESS classes (one per thread, _narrow) save no statistics:
# the gradient's own launch recomputes them from the logits it reads anyway, so training needs
# no float64 state (16 bytes per row) and the forward writes none. Off (0): the recomputing
# gradient (_labels NARROW GRAD, scalar loads) saves memory but is slower; it needs a
# lane-tile recompute first.
_STATELESS = 0


def _stateless(call: Call) -> bool:
    return (
        call.c < _STATELESS
        and not (call.dense or call.mask or call.weighted or call.smoothed)
        and call.fp32
    )


def _lane_shape(call: Call) -> tuple[int, int, int, int, bool] | None:
    """(LANES, PV, rows per tile (one per thread), warps, ALIGNED) of a label row of the lanes
    kernel, else None."""
    if call.dense or call.mask or call.smoothed or not call.fp32 or call.grad:
        return None
    if not _NARROW <= call.c <= _lane_max(call):
        return None
    return _lane_dims(call)


# Most row groups per lanes tile (unrolled: code size and registers grow with them).
_LANE_GROUPS = 8
# Fewest programs of a lanes forward before its tiles drop row groups.
_LANE_PROGRAMS = 512
# Unsmoothed label gradients from saved statistics of _LANE_BACKWARD to _LANE_MAX classes:
# _lane_backward (one program per R rows of the lanes tiles).
# Off: _backward is used instead.
_LANE_BACKWARD = 1 << 30


def _lane_dims(call: Call) -> tuple[int, int, int, int, bool]:
    lanes, pv, threads, aligned = rows.lane_dims(
        call.c, call.sz, _ITEMSIZE[call.dtype], _LANE_ELEMS[_ITEMSIZE[call.dtype]], _LANE_THREADS
    )
    return lanes, pv, threads, threads // 32, aligned


# Programs of a call whose label kernel sums its own values (_labels PARTIAL): enough to fill
# the GPU, few enough for one _reduce block.
_PARTIAL_PROGRAMS = 2048


def _partial(call: Call) -> bool:
    """Whether a label row kernel sums its values for the reduction itself (one group, float32
    work: partial sums of float32-range values cannot overflow float64)."""
    return (
        call.reduction != 0
        and call.values
        and call.fp32
        and call.reduction_rows in (0, call.n)
        # Wider rows: the persistent loop slows the one-row programs.
        and call.c <= _lane_max(call)
    )


def _warps(block: int) -> int:
    return min(32, max(1, block // 512))


def _forward_shape(call: Call) -> tuple[int, int, int]:
    """Choose a resident row or a bounded streaming tile from the width and work dtype."""
    c = call.c
    block = max(16, pow2(c))
    if block > _REGISTER:
        return (*_CHUNKS[call.fp32], 1)
    rows = rows_per_program(c)
    return block, max(2, _warps(block * rows)) if rows > 1 else _warps(block), rows


def _backward_shape(c: int, fp32: bool) -> tuple[int, int, int, int]:
    """Block, warps, tiles and rows of a backward program."""
    rows = rows_per_program(c)
    if rows > 1:
        block = max(16, pow2(c))
        return block, max(2, _warps(block * rows)), 1, rows
    block = min(1024, max(64, pow2(c)))
    return block, _warps(block), _BACKWARD_TILES if fp32 and c > _REGISTER else 1, 1


def _chunked_plan(
    b: Builder,
    call: Call,
    args: dict,
    flags: dict[str, Any],
    target: int,
    kernel: object,
    columns: int,
) -> common.Plan:
    """Rows of more than the register width: chunk partials, a per-row merge (with the values),
    and the chunked backward when a gradient is wanted."""
    n = call.n
    width, warps = _PARTS
    width = _chunk_width(call, width)
    chunks = row_chunks(call.c, width)
    partial = b.array("P", (n, chunks, 5 if target == 0 else 7), "float64")
    part = dict(args, P=partial)
    if target == 0:
        gap = call.smoothed and call.values
        b.launch(
            _label_partial, (n * chunks,), part, warps, C=call.c, B=width, CHUNKS=chunks,
            GAP=gap, MASS=gap and call.weighted, SZ=call.sz,
        )  # fmt: skip
    else:
        b.launch(
            _dense_partial, (n * chunks,), part, warps, C=call.c, B=width, CHUNKS=chunks,
            WEIGHTED=call.weighted, SMOOTH=call.smoothed, SZ=call.sz, TARGET=target,
        )  # fmt: skip
    args["S"] = b.stats(columns)
    merge = dict(args, P=partial, G=args["Z"])
    grid = (n,)
    if target == 0 and _partial(call):
        grid = _partials(b, merge, flags, n)
    b.launch(
        kernel, grid, merge, _MASK_ROW[1] if call.mask else 1, B=width, ROWS=1,
        CHUNKS=chunks, CP=pow2(chunks),
        **dict(flags, STATS=True, GRAD=False),
    )  # fmt: skip
    if call.grad and call.saved != common.SAVE:
        _chunk_backward_launch(b, args, flags)
    return b.plan()


def plan(call: Call) -> common.Plan:
    """V and G of cross_entropy (labels or probabilities) and multilabel_cross_entropy: a
    gradient from the statistics a forward saved is _backward alone."""
    n = call.n
    b = Builder(call)
    target = 2 if call.mask else 1 if call.dense else 0
    args = dict(b.inputs(), **b.results())
    flags = dict(b.flags(), SMOOTH=call.smoothed, TARGET=target)
    # A short row fits in one backward program, which can recover its maximum
    # from the logits it already loads. The saved width must also survive vmap.
    compact = (
        target == 0
        and call.dtype in ("float32", "bfloat16", "float16")
        and call.fp32
        and rows_per_program(call.c) > 1
        and not (call.weighted or call.smoothed)
    )
    flags["COMPACT"] = compact
    columns = 2 if compact else (4 if target == 0 else 6)
    # Arrays a launch does not read are the scores (the C++ plan replay binds every pointer).
    args.update(P=args["Z"], T=args["Z"], TILES=1)
    chunked = _chunked(call)
    if call.saved == common.SAVED:
        args["S"] = b.stats(columns)
        if chunked:
            _chunk_backward_launch(b, args, flags)
        else:
            _gradient_launch(b, args, flags)
        return b.plan()
    kernel = _labels if target == 0 else _dense
    loop = _looped(call) if target == 0 else None
    if loop is not None:
        width, warps = loop
        grid = (n,)
        if _partial(call):
            grid = _partials(b, args, flags, n)
        if call.saved == common.SAVE:
            args.update(S=b.stats(columns), G=args["Z"])
        else:
            args["S"] = args["Z"]
        b.launch(
            kernel, grid, args, warps, B=width, ROWS=1, LOOP=row_chunks(call.c, width),
            **dict(flags, STATS=call.saved == common.SAVE),
        )  # fmt: skip
        return b.plan()
    if chunked:
        return _chunked_plan(b, call, args, flags, target, kernel, columns)
    block, warps, rows = _forward_shape(call)
    lanes = _lane_shape(call) if target == 0 else None
    if lanes is not None:
        flags["LANES"], flags["PV"], threads, warps, flags["ALIGNED"] = lanes
        # Row groups per tile: all lanes finish a row each, unless that leaves fewer than
        # _LANE_PROGRAMS programs (rows of a group: one per lane).
        groups = min(flags["LANES"], _LANE_GROUPS)
        per = threads // flags["LANES"]
        while groups > 1 and common.cdiv(n, groups * per) < _LANE_PROGRAMS:
            groups //= 2
        flags["GROUPS"] = groups
        flags["SPLIT"] = _LANE_SPLIT if call.c <= _LANE_EXACT else _LANE_WIDE_SPLIT
        rows = groups * per
    elif _narrow(call):
        rows, warps = _NARROW_ROWS, _NARROW_WARPS
        flags["NARROW"] = True
    grid = (common.cdiv(n, rows),)
    if target == 0 and _partial(call):
        grid = _partials(b, args, flags, grid[0])
    if call.saved == common.SAVE and target == 0 and _stateless(call):
        # No saved statistics: the gradient recomputes them from the row it reads anyway.
        args.update(S=args["Z"], G=args["Z"])
        b.launch(kernel, grid, args, warps, B=block, ROWS=rows, STATS=False, **flags)
        return b.plan()
    if call.saved == common.SAVE:
        args.update(S=b.stats(columns), G=args["Z"])
        b.launch(kernel, grid, args, warps, B=block, ROWS=rows, STATS=True, **flags)
        return b.plan()
    # For a joint call the row statistics stay in the forward program.
    policy = dict(LOAD="evict_last", STORE="evict_first") if call.grad else {}
    args["S"] = args["Z"]
    b.launch(kernel, grid, args, warps, B=block, ROWS=rows, STATS=False, **policy, **flags)
    return b.plan()


# Wide label rows (more than _LANE_MAX classes, float32 work, no smoothing; values and
# statistics) of at least _LOOP_ROWS rows: one program per row reads it once in _LOOP chunks
# (columns, warps by storage size; _looped_row). Fewer rows run as parallel chunks and a merge.
# Off: the chunks are used instead.
_LOOP_ROWS = 1 << 40
_LOOP = {2: (2048, 4), 4: (2048, 8)}


def _looped(call: Call) -> tuple[int, int] | None:
    """(B, warps) of a wide label row read by one program, else None."""
    if call.dense or call.mask or call.smoothed or not call.fp32 or call.grad:
        return None
    if call.c <= _lane_max(call) or call.n < _LOOP_ROWS:
        return None
    return _LOOP[_ITEMSIZE[call.dtype]]


def _partials(b: Builder, args: dict, flags: dict, tiles: int) -> tuple[int]:
    """Make _labels sum its values (PARTIAL): at most _PARTIAL_PROGRAMS programs over the
    tiles, their sums in T for the reduction (reduce.add), no per-row values."""
    programs = min(tiles, _PARTIAL_PROGRAMS)
    b.drop("V")
    args.update(V=Ref("Z"), T=b.array("T", (2 * programs,), "float64"), TILES=tiles)
    flags["PARTIAL"] = True
    b.partials = programs
    return (programs,)


def _chunked(call: Call) -> bool:
    """Float32-work rows of parallel chunks (_PARTS): wide label and mask rows, and
    probability rows of at least _DENSE_CHUNKED classes (their register kernel's float64 sums
    cost more than a merge). Mask rows whose maximum is a positive class take the float64
    row (_dense, _mask_backward)."""
    if not call.fp32:
        return False
    if call.mask:
        return call.c > _REGISTER
    return call.c > _REGISTER or (call.dense and call.c >= _DENSE_CHUNKED)


def _chunk_width(call: Call, width: int) -> int:
    """At most width columns, fewer for a row that needs fewer (one chunk from its boundary)."""
    return min(width, max(256, pow2(call.c + rows.SLACK)))


def _chunk_backward_launch(b: Builder, args: dict, flags: dict) -> None:
    """_chunk_backward from the statistics S: one aligned read and write of the scores."""
    # Mask rows on this path have a negative maximum: public's normwise bound needs no
    # refinement of cancelling positives.
    refine = 0.0625 if b.call.dtype == "float32" and not b.call.mask else 0.0
    width, warps = _CHUNK_REFINE if refine and b.call.dense else _CHUNK_BACKWARD
    width = _chunk_width(b.call, width)
    chunks = row_chunks(b.call.c, width)
    b.launch(
        _chunk_backward, (b.call.n * chunks,), args, warps, B=width, CHUNKS=chunks,
        REFINE=refine, STRIDED=b.call.stride != 0, **flags,
    )  # fmt: skip
    if b.call.mask:
        # The rows the lean launch leaves (G written by both: state, for JAX).
        width, warps = _MASK_ROW
        b.launch(_mask_backward, (b.call.n,), args, warps, state=("G",), B=width, **flags)


def _gradient_launch(b: Builder, args: dict, flags: dict) -> None:
    """_backward from the statistics S: one read and one write of the scores."""
    n, c = b.call.n, b.call.c
    call = b.call
    if (
        not (call.dense or call.mask or call.smoothed)
        and call.fp32
        and _LANE_BACKWARD <= c <= _lane_max(call)
    ):
        lanes, pv, threads, warps, aligned = _lane_dims(call)
        rows = threads // lanes
        b.launch(
            _lane_backward, (common.cdiv(n, rows),), args, warps, LANES=lanes, PV=pv,
            ROWS=rows, ALIGNED=aligned, **flags,
        )  # fmt: skip
        return
    block, warps, tiles, rows = _backward_shape(c, b.call.fp32)
    if rows > 1 and (c < _NARROW or block * rows < _BACKWARD_TILE):
        rows, warps = max(rows, _BACKWARD_TILE // block), 4
    if b.call.mask and rows == 1:
        # Each block owns one column tile of a mask row; no row-wide reduction is needed.
        warps, tiles = 4, 1
    grid = (common.cdiv(n, rows), 1 if rows > 1 else common.cdiv(c, block * tiles))
    b.launch(
        _backward,
        grid,
        args,
        warps,
        B=block,
        TILES=tiles,
        ROWS=rows,
        **flags,
    )


def describe(call: Call) -> str:
    block = _forward_shape(call)[0]
    layout = f"{block}-column chunks" if block < call.c else "rows in registers"
    if not call.dense and not call.mask and _lane_shape(call) is not None:
        layout = "rows over lanes of 16-byte vectors"
    if call.mask:
        return f"float64 mask normalization, {layout}"
    return f"{'float32 exponentials' if call.fp32 else 'float64'}, {layout}"
