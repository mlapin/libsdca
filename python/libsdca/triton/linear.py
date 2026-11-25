"""Write a contiguous BF16 linear-head pullback from its FP32 accumulation and scalar cotangent."""

import torch
import triton
import triton.language as tl


@triton.jit
def _scale_bfloat(G, UP, OUT, n, FULL: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Preserve coalesced full-width accesses for every complete block even
    # when the tensor ends in a partial block.
    factor = tl.load(UP)
    if FULL or tl.program_id(0) < n // BLOCK:
        g = tl.load(G + offsets)
        tl.store(OUT + offsets, (g * factor).to(tl.bfloat16, fp_downcast_rounding="rtne"))
    else:
        live = offsets < n
        g = tl.load(G + offsets, live, other=0.0)
        tl.store(OUT + offsets, (g * factor).to(tl.bfloat16, fp_downcast_rounding="rtne"), live)


def scale_bfloat(gradient: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """One coalesced FP32 read and BF16 write per saved gradient entry."""
    n = gradient.numel()
    block = 1024
    out = torch.empty_like(gradient, dtype=torch.bfloat16)
    _scale_bfloat[(triton.cdiv(n, block),)](
        gradient, factor, out, n, n % block == 0, block, num_warps=4
    )
    return out
