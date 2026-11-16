"""Conditional fp32 distillation kernels shared by Torch and JAX CUDA launch plans.

Ordinary, nonconfident rows use conventional fp32 softmax arithmetic. Near-equal,
confident, far-tail and nonfinite rows reuse the guarded implementation. The marker
2 in statistics slot 12 is outside its residual codes for C > 16. No extra buffer.
"""

# Triton JIT pointer arguments are not Python scalar types.
# pyright: reportMissingParameterType=false
import triton
import triton.language as tl

from ._triton_distillation import _backward as _precise_backward
from ._triton_distillation import _forward as _precise_forward


@triton.jit(do_not_specialize=["N"])
def _forward(
    S,
    T,
    TEMP,
    ST,
    N,
    C: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    N = N.to(tl.int64)
    temperature = tl.load(TEMP)
    j = tl.arange(0, B)
    s = tl.load(S + row * C + j, j < C, other=-float("inf")).to(tl.float32)
    t = tl.load(T + row * C + j, j < C, other=-float("inf")).to(tl.float32)
    sm, tm = tl.max(s, 0), tl.max(t, 0)
    s, t = (s - sm) / temperature, (t - tm) / temperature
    es, et = tl.exp(s), tl.exp(t)
    ss, ts = tl.sum(es, 0), tl.sum(et, 0)
    log_ss, log_ts = tl.log(ss), tl.log(ts)
    loss = log_ss - tl.sum(tl.where(j < C, et * s, 0.0), 0) / ts
    tl.store(ST + row, loss)
    tl.store(ST + N + row, 0.0)
    tl.store(ST + 2 * N + row, sm)
    tl.store(ST + 3 * N + row, tm)
    tl.store(ST + 4 * N + row, ss)
    tl.store(ST + 5 * N + row, ts)
    near = tl.max(tl.where(j < C, tl.abs(s - t), 0.0), 0) < 0.125
    # Bound normalized probabilities, including many tied maxima. Raw centered
    # exponents can be normal while division by their sum creates subnormals.
    far = tl.min(tl.where(j < C, tl.minimum(s - log_ss, t - log_ts), 0.0), 0) < -80.0
    bad = (tl.abs(sm) == float("inf")) | (tl.abs(tm) == float("inf"))
    bad = bad | (sm != sm) | (tm != tm) | (loss != loss) | (tl.abs(loss) == float("inf"))
    # A dominant student class loses log(1 + tail) and its gradient to
    # cancellation well before exponentials underflow.
    confident = ss < 2.0
    tl.store(ST + 12 * N + row, tl.where(near | far | bad | confident, -99.0, 2.0))


@triton.jit(do_not_specialize=["N"])
def _backward(
    S,
    T,
    TEMP,
    ST,
    U,
    G,
    N,
    C: tl.constexpr,
    REDUCTION: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    N = N.to(tl.int64)
    temperature = tl.load(TEMP)
    j = tl.arange(0, B)
    s = tl.load(S + row * C + j, j < C, other=-float("inf")).to(tl.float32)
    t = tl.load(T + row * C + j, j < C, other=-float("inf")).to(tl.float32)
    sm, tm = tl.load(ST + 2 * N + row), tl.load(ST + 3 * N + row)
    ss, ts = tl.load(ST + 4 * N + row), tl.load(ST + 5 * N + row)
    p = tl.exp((s - sm.to(tl.float32)) / temperature) / ss.to(tl.float32)
    q = tl.exp((t - tm.to(tl.float32)) / temperature) / ts.to(tl.float32)
    weight = tl.load(U + row) if REDUCTION == 0 else tl.load(U)
    weight = weight / temperature
    if REDUCTION == 1:
        weight = weight / N
    keep = j < C
    keep = keep & (tl.load(ST + 12 * N + row) == 2.0)
    tl.store(G + row * C + j, (p - q) * weight, keep)


@triton.jit(do_not_specialize=["N"])
def _rescue_forward(S, T, TEMP, ST, N, C: tl.constexpr, B: tl.constexpr, LARGE: tl.constexpr):
    N = N.to(tl.int64)
    if tl.load(ST + 12 * N + tl.program_id(0)) != 2.0:
        _precise_forward(S, T, TEMP, ST, N, C, B, True, LARGE, 2.0**-23, False, 1)


@triton.jit(do_not_specialize=["N"])
def _rescue_backward(
    S,
    T,
    TEMP,
    ST,
    U,
    G,
    N,
    C: tl.constexpr,
    B: tl.constexpr,
    LARGE: tl.constexpr,
    REDUCTION: tl.constexpr,
):
    N = N.to(tl.int64)
    if tl.load(ST + 12 * N + tl.program_id(0)) != 2.0:
        _precise_backward(S, T, TEMP, ST, U, G, N, C, B, True, LARGE, REDUCTION, 1)
