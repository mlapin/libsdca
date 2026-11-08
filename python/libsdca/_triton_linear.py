"""Round a scaled float32 linear-loss gradient directly into 16-bit storage."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _scale_cast(X, U, Y, N, B: tl.constexpr):
    start = tl.program_id(0).to(tl.int64) * B
    j = start + tl.arange(0, B)
    factor = tl.load(U).to(tl.float32)
    # Match ATen: float32 multiplication, followed by one storage conversion. Only unmasked
    # blocks vectorize: N is not specialized.
    if start + B <= N:
        tl.store(Y + j, (tl.load(X + j) * factor).to(Y.dtype.element_ty))
    else:
        x = tl.load(X + j, j < N, other=0)
        tl.store(Y + j, (x * factor).to(Y.dtype.element_ty), j < N)


def scaled(x: torch.Tensor, upstream: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Contiguous CUDA float32 x times scalar upstream, rounded to dtype."""
    out = torch.empty(x.shape, device=x.device, dtype=dtype)
    if x.numel():
        with torch.cuda.device(x.device):
            _scale_cast[(triton.cdiv(x.numel(), 1024),)](
                x, upstream, out, x.numel(), 1024, num_warps=4, enable_fp_fusion=False
            )
    return out
