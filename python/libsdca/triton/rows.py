"""Row chunks shared by the streamed kernels.

A wide row is split into chunks of B columns, one program per (row, chunk) (program axes 0
and 1): chunks run in parallel and merge their partial statistics exactly afterwards (maxima,
sums rescaled to the merged maximum), instead of one program walking the row.

Each chunk is taken from the 16-byte boundary at or below the row's start, so that whole
chunks inside the row load and store whole vectors at any width and any base offset (a
width that is not a multiple of the vector, or a sliced tensor): only a row's first and
last chunks are masked. Rows of several arrays share chunk positions when their starts have
the same misalignment (`shift`); otherwise the chunks start at the row and are all masked.
Launch cdiv(C + 15, B) chunks per row: columns may begin up to 15 bytes before it.

Rows may be strided: every helper that addresses a row takes the row stride S of its array
(C for a contiguous one), so each row's shift is its own. A byte mask (masks of the
multilabel losses) shares the positions of the scores where its row, less the scores'
shift, is aligned to the scores' vector (mask_shift): its lanes then load whole vectors too.
"""

from __future__ import annotations

import triton
import triton.language as tl

from .math import INTERPRET

SLACK = 15  # bytes a row's first chunk may start before the row


def chunks(c: int, block: int) -> int:
    """Chunks of `block` columns per row of c columns (from the boundary below each row)."""
    return triton.cdiv(c + SLACK, block)


if INTERPRET:

    @triton.jit
    def misaligned(P, row, S: tl.constexpr):
        """The interpreter has no vectors: no shift."""
        return tl.zeros((), tl.int32)

    @triton.jit
    def mask_shift(Z, M, row, SZ: tl.constexpr, SM: tl.constexpr):
        """The interpreter has no vectors: no shift, masks always shared."""
        shift = tl.zeros((), tl.int32)
        return shift, shift == shift

else:

    @triton.jit
    def misaligned(P, row, S: tl.constexpr):
        """Elements between row's start in P (row stride S) and the 16-byte boundary at or
        below it."""
        size: tl.constexpr = P.dtype.element_ty.primitive_bitwidth // 8
        start = (P + row * S).to(tl.int64, bitcast=True)
        return ((start % 16) // size).to(tl.int32)

    @triton.jit
    def mask_shift(Z, M, row, SZ: tl.constexpr, SM: tl.constexpr):
        """The shift of row in the scores Z, and whether the byte mask M shares it: its row
        less that shift starts on a boundary of the scores' vector (16 bytes of scores), so
        that a chunk's mask loads whole vectors as well (load_mask)."""
        shift = misaligned(Z, row, SZ)
        VEC: tl.constexpr = 128 // Z.dtype.element_ty.primitive_bitwidth
        start = (M + row * SM).to(tl.int64, bitcast=True) - shift
        return shift, start % VEC == 0


@triton.jit
def place(CHUNKS: tl.constexpr):
    """(row, chunk) of program_id(0) in a 1-D grid of rows x CHUNKS programs, chunks fastest:
    consecutive programs read consecutive memory (more bandwidth than rows fastest), and the
    grid has no 65535 limit on rows."""
    pid = tl.program_id(0).to(tl.int64)
    return pid // CHUNKS, (pid % CHUNKS).to(tl.int32)


@triton.jit
def shared_shift(A, B, row, SA: tl.constexpr, SB: tl.constexpr):
    """The common misalignment of row in A and B (any element types of one size; row strides
    SA and SB), and whether they share it (else 0: masked chunks from the row's start)."""
    shift = misaligned(A, row, SA)
    if A.dtype.element_ty.primitive_bitwidth == B.dtype.element_ty.primitive_bitwidth:
        shared = shift == misaligned(B, row, SB)
    else:
        shared = shift != shift  # elements of different sizes never share positions
    return tl.where(shared, shift, 0), shared


@triton.jit
def span(block, shift, shared, B: tl.constexpr, C: tl.constexpr):
    """Chunk `block` of a row: window positions j ([1, B]), columns k = j - shift, which are
    in the row, and whether the whole chunk is (it then needs no mask)."""
    j = (block * B + tl.arange(0, B))[None, :]
    k = j - shift
    valid = (k >= 0) & (k < C)
    inside = shared & (block > 0) & ((block + 1) * B <= C)
    return j, k, valid, inside


@triton.jit
def load(P, row, j, shift, valid, S: tl.constexpr, MASKED: tl.constexpr, other=0):
    """Row `row` of P (row stride S) at window positions j: whole vectors unless MASKED.
    Kernels branch once on `inside` (span) and run their chunk body with MASKED False or
    True, so that the compiler sees the unmasked loads and stores of a whole chunk
    together."""
    base = P + row * S - shift
    if MASKED:
        x = tl.load(base + j, valid, other=other)
    else:
        x = tl.load(tl.multiple_of(base, 16) + j)
    return x


@triton.jit
def load_mask(M, row, j, shift, valid, VEC: tl.constexpr, S: tl.constexpr,
              MASKED: tl.constexpr):  # fmt: skip
    """Whether the byte mask M (row stride S) is set at window positions j of a chunk whose
    scores load vectors of VEC elements (mask_shift): whole VEC-byte vectors unless MASKED
    (False off the row)."""
    base = M + row * S - shift
    if MASKED:
        m = tl.load(base + j, valid, other=0)
    else:
        m = tl.load(tl.multiple_of(base, VEC) + j)
    return m != 0


@triton.jit
def store(P, row, j, shift, valid, value, S: tl.constexpr, MASKED: tl.constexpr):
    """Store value at window positions j of row `row` of P (row stride S): whole vectors
    unless MASKED."""
    base = P + row * S - shift
    if MASKED:
        tl.store(base + j, value.to(P.dtype.element_ty), valid)
    else:
        tl.store(tl.multiple_of(base, 16) + j, value.to(P.dtype.element_ty))


# Lane tiles: rows of up to a few thousand columns as [PV, R, LANES, VEC] tiles, VEC columns
# of one 16-byte vector, LANES lanes per row (contiguous vectors across lanes), PV vectors per
# lane and R rows. Each row is read from the 16-byte boundary at or below its start, a whole
# vector wherever it holds a column of the row (its other bytes lie in the same 16-byte
# block, so no read leaves the row's pages), so the loads stay 16-byte vectors at any width,
# row stride and offset. Reduce over axes (3, 0) within a lane, then 2 across a row's lanes
# (lane_max, lane_sum: keep_dims, so row statistics broadcast back onto the tile for free).


@triton.jit
def lane_shift(P):
    """Elements between the pointers P and the 16-byte boundaries at or below them (the
    real offset, also under the interpreter)."""
    size: tl.constexpr = P.dtype.element_ty.primitive_bitwidth // 8
    return ((P.to(tl.int64, bitcast=True) % 16) // size).to(tl.int32)


@triton.jit
def lane_columns(PV: tl.constexpr, LANES: tl.constexpr, VEC: tl.constexpr):
    """Window positions of a lane tile: each vector's first ([PV, 1, LANES, 1]) and every
    element's ([PV, 1, LANES, VEC])."""
    lead = (tl.arange(0, PV)[:, None, None, None] * LANES
            + tl.arange(0, LANES)[None, None, :, None]) * VEC  # fmt: skip
    return lead, lead + tl.arange(0, VEC)[None, None, None, :]


@triton.jit
def lane_load(P, row, live, S: tl.constexpr, C: tl.constexpr, PV: tl.constexpr,
              LANES: tl.constexpr, ALIGNED: tl.constexpr, other=0,
              POLICY: tl.constexpr = ""):  # fmt: skip
    """Rows `row` ([1, R, 1, 1], where live) of P (row stride S, C columns) as a lane tile:
    the scores x (other off the row), their columns k and whether they are in the row.
    ALIGNED: rows are 16-byte multiples and the window holds C columns only (no slack): a
    misaligned base then reads by column from each row's start (correct, unvectorized)."""
    VEC: tl.constexpr = 128 // P.dtype.element_ty.primitive_bitwidth
    start = P + row * S
    lead, j = lane_columns(PV, LANES, VEC)
    if ALIGNED:
        if lane_shift(P) == 0:
            k = j + (row * 0).to(tl.int32)
            x = tl.load(tl.multiple_of(start, (16, 16, 16, 16)) + j, live & (lead < C),
                        other=other, eviction_policy=POLICY)  # fmt: skip
        else:
            k = j + (row * 0).to(tl.int32)
            x = tl.load(start + k, live & (k < C), other=other, eviction_policy=POLICY)
    else:
        shift = lane_shift(start)
        k = j - shift
        x = tl.load(tl.multiple_of(start - shift, (16, 16, 16, 16)) + j,
                    live & (lead < shift + C), other=other, eviction_policy=POLICY)  # fmt: skip
    valid = live & (k >= 0) & (k < C)
    return x, k, valid


@triton.jit
def lane_max(x):
    """The maximum of each row of a lane tile, [1, R, 1, 1]."""
    return tl.max(tl.max(tl.max(x, 3, keep_dims=True), 0, keep_dims=True), 2, keep_dims=True)


@triton.jit
def lane_sum(x):
    """The sum of each row of a lane tile, [1, R, 1, 1] (in x's type)."""
    return tl.sum(tl.sum(tl.sum(x, 3, keep_dims=True), 0, keep_dims=True), 2, keep_dims=True)


def lane_dims(c: int, stride: int, itemsize: int, elems: int, threads: int):
    """(LANES, PV, threads, ALIGNED) of rows of c columns (row stride `stride`, itemsize bytes)
    at about `elems` columns per lane and at least `threads` threads per program."""
    vec = 16 // itemsize
    aligned = stride * itemsize % 16 == 0
    width = triton.next_power_of_2(c if aligned else c + vec - 1)
    lanes = max(1, width // elems)
    pv = max(1, width // (vec * lanes))
    lanes = width // (vec * pv)
    return lanes, pv, max(threads, lanes), aligned
