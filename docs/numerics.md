# Numerical notes

What libsdca 0.3.0 guarantees in floating point, how it is verified and where its
limits are. The measurements use an NVIDIA A100 with PyTorch 2.15.0 and Linux x86-64
CPUs; earlier ones also cover an H100 and macOS arm64. They are measured envelopes on the
stated inputs, not bounds for every representable input. Each bound names the test that
asserts it.

**In short**

- **No softmax collapse.** For a confident row, standard cross-entropy rounds the
  normalizer `1 + τ` (`τ` the sum of the competitors' `exp(-gap)`) to 1, so the loss and
  the target gradient become exactly zero while the competitors' gradients do not
  ([softmax collapse](softmax-collapse.md)). libsdca forms the loss as `log1p(τ)` and the
  target gradient from `τ` itself, so both stay nonzero until the exact value underflows
  the output dtype: the first zero is near gap 104 in float32 and 745 in float64, where
  PyTorch's cross-entropy returns exactly zero from 16.6 and 36.7. These two-class
  boundaries do not imply guarantees for every aggregate; see
  [aggregate-tail limitations](#aggregate-tail-limitations).
- **Every entry rounds once.** Each gradient entry is computed in the working precision,
  multiplied there by the upstream factor (a GradScaler's scale) and the mean's `1/N`, and
  only then rounded to the output dtype. An entry rounded to 16 bits before that scaling
  would be rounded twice, and every entry below the dtype's smallest subnormal (6e-8 in
  float16) would already be zero, which no scale can restore. With one rounding a scaled
  float16 gradient keeps entries that would otherwise vanish (float16 gap 20 with a scale of
  65504: `-1.35064e-4` against the exact `-1.35014e-4`; PyTorch returns 0), and the gradient
  of a row still sums to zero up to that one rounding. The exception is
  `linear_cross_entropy`'s saved gradients, formed at upstream derivative 1
  ([PyTorch notes](../python/TORCH.md#linear-layer-and-cross-entropy-without-the-logits)); `recompute=True` rounds once.
- **0 failures against 100-digit references** ([how it is verified](#how-it-is-verified)):
  30,145 CPU checks in all four dtypes (12,194 with the CPU extension off, 17,951 on) and
  13,720 CUDA checks. The independent proximal oracle passes 10,884 CPU and 6,734 CUDA cases. Targeted probes
  outside this corpus found the [aggregate-tail limitations](#aggregate-tail-limitations)
  below, so the zero-failure count is not a universal bound.

## What each precision guarantees

Every precision uses the same stable formulas; wider arithmetic adds accuracy, it is
not what avoids NaNs or lost residuals.

| | `"float32"`, float32 data | `"float64"`, float32 data (default) | float64 data |
| --- | --- | --- | --- |
| Coordinate steps | float (hinge projections in double) | double | double |
| `fit`'s iterate (`W`, duals) | float | double, rounded once into the returned float32 arrays at each evaluation | double |
| Losses | float; double where float cannot hold the parameters or a row's spread | double | double |
| Evaluation and `solved` | double | double | double |
| Entropy blocks of a coordinate step against the double solution | the scores' rounding and the solver's mass tolerance: ≤ (64 n + 16 max \|z\|) · 2⁻²⁴ of the block | ≤ 8 · 2⁻²⁴ of the block (measured: 1.86 float ulps) | reference |
| `multiclass_hinge` on standardized digits, lowest certified relative gap (5000 epochs) | 4.7e-6 | 3.5e-7 (tol 1e-6 in 2132 epochs; float64 data: 2100) | below 1e-7 (2610 epochs) |

The block bounds are those of `tests/cpp/regression.cpp`
(`float_storage_blocks_match_double_storage`), for the single steps of
`coordinate_update`, which store their result in the input's dtype. `"float32"` trades
these float-arithmetic errors for speed ([performance](performance.md#solver)).

A `fit` with float storage of its iterate loses the small updates: near the optimum,
most of an epoch's changes of `W` (`x_ij` times a dual's change) and of the duals are
below half an ulp of the entry they change and round away (89% of the updates of `W` and 75%
of the dual changes near the digits optimum), so the
iterate stalls. Rebuilding
`W` from the duals at each evaluation (`recompute_primal`) does not help: 4.5e-6 with it,
4.3e-6 without. Nor does a double `W` alone: float duals still stall at 1.6e-6 on
digits and 1.2e-6 on synthetic-10k at 5000 epochs. With `"float64"` the iterate is double
and each evaluation rounds it once into the returned float32 arrays and certifies those:
the floor is then that rounding (3.5e-7 on digits). `W` is returned rounded, not rebuilt
from the rounded duals: their rounding would enter every entry of `W` (that rebuild
alone held the gap at 3.3e-6).

### The `solved` certificate

`solved` means `P - D + e <= tol max(|P|, |D|)` ([API.md](API.md#results)): the primal at
the returned weights, the dual at the returned duals, both in double, with `e` the
evaluator's estimate of its own rounding. A gap below `-e` ends training as `failed`.
`test_certificate.py` checks every `solved` fit against a 50-digit primal.

### Projection precision

`projection_precision` selects how the multilabel hinge projection solves: `"float64"`
(the default) solves in double and keeps the result only if every entry is clear of its
threshold, else it resolves in `"extended"`, the hardened long-double solver. The top-k
hinge projections always use the extended solver. The two modes agree to 16 eps on
offsets to 1e30 and 2,000 integer-tie cases (`tests/cpp/regression.cpp`;
[API.md](API.md#projection-precision)).

## Representation limits

For logits `[0, -g]` and a correct target, the exact loss and gradient decay like
`exp(-g)` and become zero only when they underflow the output dtype:

| Logits / gradient dtype | libsdca target gradient: first zero | PyTorch CPU loss / target: first zero | PyTorch CUDA loss / target: first zero |
| --- | ---: | ---: | ---: |
| float16 | 17.34375 | 7.625 | 16.640625 |
| bfloat16 | 93 | 5.5625 | 16.75 |
| float32 | 103.97208 | 16.63553 | 16.63553 |
| float64 | 745.13322 | 36.73680 | 36.73680 |

These are the first represented input gaps measured with PyTorch 2.15.0,
without autocast, on CPU and A100. CUDA computes the 16-bit reference
normalization more widely than CPU, but still loses the target residual
well before its output underflows.

16-bit logits get float32 loss values. For two classes, the first represented gap
with zero loss is 104. With
`C - 1` equal competitors the limit moves out by about `log(C - 1)`. Gaps already rounded
away in the input logits cannot be recovered. `test_half_zero_gaps.py` asserts the first
zero of the 16-bit loss and target gradient (2 and 1000 classes), `test_float32_zero_gaps.py`
the float32 entropy losses at gaps 80 to 112 (subnormal values, then zero) and
`test_softmax_limits.py` the float64 values at gap 745. PyTorch's zeros are measured, not
tested.

## How it is verified

Comparing against PyTorch or against float64 does not measure the error: both are
inexact themselves, and float64 cross-entropy collapses from a gap of 36.7. The numerics
checks therefore compare every loss value and gradient entry with a reference computed in
100-digit arithmetic (mpmath) from the inputs as represented in their dtype, so the
rounding of the inputs does not count as error. The inputs target the cases that break
floating point: 2 to 32,769 classes (vector widths and one more or less), gaps to 110
(750 in float64) and through the output's subnormal range, large common offsets, ties,
±finfo.max, all tested dtypes and CPU/native/CUDA routes (CPU extension on and off, native
library and CUDA kernels). A record is one input on one backend and dtype; it passes when the value and every
gradient entry are within 64 eps of the output dtype of the reference (plus one subnormal
quantum near zero), and an overflow must give an infinity of the right sign. The largest
error is 0.678 of that budget on CPU and 0.622 on CUDA. Top-k cross-entropy
(k = 5, 64, 65) now includes CPU and CUDA, fused and tensor paths, and float64
subnormal gaps. Float64 rows use compensated sums within the 64-eps budget;
float32 work adds `C` working eps (and `C / 2` subnormal quanta for the native
library's float32 work). Top-k cases leave out ±finfo.max. The target mass is
retained independently of rounded competitor entries, including on the tensor
path (`test_topk_subnormal_mass.py`).

| Check | Result |
| --- | --- |
| Entropy and distillation stress: 2 to 32,769 classes, gaps to 110 (750 in float64) and through the subnormal range, offsets at the loss of input resolution, ±finfo.max (not in the top-k cases: see the large-spread limit below), all tested dtypes and CPU/native/CUDA routes, top-k cross-entropy on CPU and CUDA fused/tensor paths, against 100-digit references | 0 failures in 30,145 CPU and 13,720 CUDA candidate checks; largest budget fractions 0.678 and 0.622 respectively (CUDA distillation: 64 output eps plus `C` working eps) |
| Proximal oracle: `alpha` 1e-300 to 1e300, offsets, ties, multiple labels, up to 32,768 classes, against an independent mpmath solve | every case within one rounding plus 64 (float64 work) or 60 (float32 work) working eps, of the loss (values) and of the row's 1-norm (gradients) |
| Entropy blocks: KKT conditions of random blocks against a long double reference; `alpha` 1e-300 to 1e300 at the extremes | `tests/cpp/regression.cpp` (`entropy_block_kkt_randomized_*`, `entropy_block_extremes_and_validation`) |
| Warm hinge steps against the analytic cold step, old duals to 2¹⁰⁰ | within 4 eps of the block mass (`tests/cpp/stress.cpp`) |
| Projections at scales 1e-30 to 1e30 and near overflow | within 64 eps of KKT (`tests/cpp/prox/`, `tests/cpp/stress_projections.cpp`) |
| 16-bit gradients of every loss at 4096×1000, entry by entry against float64 | `test_half_gradient_rounding.py` |
| Ordinary inputs: every PyTorch CPU path, the NumPy API and `fit` against float64 and mpmath references | `test_torch_common_inputs.py`, `test_ordinary_inputs.py` |

The C++ tests also run under ASan, UBSan and ThreadSanitizer
(`tools/run_cpp_tests.sh --sanitize`), and the MATLAB tests under GNU Octave.

## Numerical contracts

These rules hold in every interface and backend; changing one can silently lose small
residuals.

- A coordinate update returns a new dual block; a loss gradient is evaluated at the given
  scores. Both keep the zero sum of the class derivatives.
- The storage type controls the stored arrays and the result type the reported
  arithmetic. Selected entropy intermediates are computed in wider precision.
- Each gradient entry is scaled by the upstream factor and the reduction before its one
  rounding to the output dtype.
- The Python and Rust functions validate inputs before calling the C ABI. The C++
  pointer-level algorithms assume valid buffers in the documented layout.
- The PyTorch and JAX tensor implementations are separate code; their tests compare them
  with the native losses and with independent references.

## PyTorch loss precision

Logits `[0, -gap]`, target 0, loss / target gradient:

| dtype, gap | PyTorch | libsdca |
| --- | --- | --- |
| bfloat16, 20 | `0 / 0` | `2.06115369e-9 / -2.066e-9` |
| float32, 20 | `0 / 0` | `2.06115369e-9 / -2.06115369e-9` |
| float64, 40 | `0 / 0` | `4.24835426e-18 / -4.24835426e-18` |
| float16 gap 20, GradScaler upstream 65504 | gradient `0` | `-1.35064e-4` (exact `-1.35014e-4`) |

libsdca's values are the 100-digit references rounded to the output dtype
(`test_softmax_limits.py`: one output unit with float64 work, and the 8 float32 eps of the
float32-exponential rows below with the default work). The hinges keep tiny margins too:
`multiclass_hinge` of `[1, 1e-8]` in float32 is `1e-8` (PyTorch: 0;
`test_softmax_limits.py`), and distillation with equal student and teacher logits has an
exactly zero gradient (PyTorch: up to 4.6e-9 in float32; `test_distillation.py`).

## Known limits

### Certificate and solver

- In double, a `tol` below about 1e-12 is often not certified even when the iterate is
  accurate; training then runs to its other limits.
- A large common center can hide small updates in the storage of `W` (float32 near 1e8,
  float64 near 1e16; with float32 data and `"float64"`, the float32 `W` returned); such a
  fit does not report `solved` (`test_centered_certificate.py`).
- Float32 storage of the returned arrays bounds the certified gap of a fit of float32 data
  to their rounding: 3.5e-7 relative on standardized digits (`"float64"`); with
  `"float32"`, the float iterate stalls higher (4.7e-6 there).

### Further precision limits

- **`"float32"` losses:** 16-bit logits with float32 work are within 100 float32 eps of
  `max(|loss|, 1)` and 2,000 eps of the row's largest gradient entry
  (`test_half_precision.py`).
- **Cross-entropy with float32 exponentials** (the default for 16- and 32-bit logits): float32
  exponentials with float64 row statistics; losses and target entries within 8 float32 eps
  of 60-digit references in the fused CUDA kernels (`test_fused_entropy.py`,
  `test_topk_kernels.py`, `test_multilabel_kernels.py`) and the native
  library (`test_native_tensor_losses.py`), and of float64 references in the CPU extension
  (`test_cpu_extension.py`). Top-k cross-entropy rows (`k <= 64` on CPU, `k <= 32` on CUDA)
  take the same float32 exponentials, within 8 float32 eps of float64 rows, and their target
  entry from the float64 row statistics, so subnormal target entries vanish where
  `cross_entropy`'s do (`test_float32_zero_gaps.py`). `precision="float64"` takes float64
  throughout.
- **CPU boolean-mask cross-entropy:** narrower logits with `precision=None` take float32
  exponentials with float64 row statistics, the positives in float64: float32 gradient
  entries are within one rounding plus 2 float32 eps of the row's largest entry (measured
  ≤ 0.94 eps against 400-digit references, 2 to 4,097 classes). With `precision="float64"`
  and for float64 logits every entry is the float64 gradient rounded once
  (`test_torch_common_inputs.py`). Row sums are compensated: well-fit rows (C − 2 equal
  small terms beside a 1) stay within 64 output eps (`test_multilabel_well_fit.py`).
- **Distillation:** within 64 eps of the logits' dtype of 80–120-digit references at `T`
  0.25–4 (`test_distillation_references.py`). Every gradient entry is formed on its own,
  from a compensated, centered log-ratio of student and teacher (`expm1` where
  they agree). Default CUDA uses float32 working arithmetic with float64 row
  statistics; explicit float64 work widens the log-ratio too. With float32 work, an
  entry where the student and teacher probabilities `q` and `p` nearly agree is within
  64 float32 eps of `(p + q) / T`, not of its own size; the CPU kernels compute rows
  whose mean log-ratio exceeds 32 in float64, CUDA keeps float32. At 128,256 classes
  the tested entries are within 64 eps of the row's largest one
  (`test_distillation.py`). Aggregate-tail exceptions are documented below.
- **Hinge margins** are formed relative to the label's score: differences between other
  scores below the resolution of a huge gap to the label are lost.
- **Top-k cross-entropy at large score spreads:** where some but not all competitors are
  capped (`k < C - 1`), the log-odds add score differences to the largest competitor and
  logarithms of order one in the working precision, so their error grows with the spread
  of the scores (not with a common offset). For `[0, S, -S, -S]`, label 0, `k = 2` (loss
  `log1p(2 sqrt(2))` = 1.3425 for every S), the float64 relative error is 5e-13 at S = 1e4,
  5e-10 at 1e8 and 9e-6 at 1e12; at 1e16 the result is `log 3` = 1.0986, and where the
  differences overflow (finfo.max) the loss is 0. Float32 results stay within float32
  resolution up to S = 1e8. The solver's primal loss (`fit` with `"topk_cross_entropy"`)
  is the same value, and the top-k entropy projection (`prox_topk_entropy`, used by the
  solver) computes its threshold the same way: for `[S, -S, -S]`, `k = 2`, its relative
  error is 2e-13 at S = 1e4 and 4e-6 at 1e12, and at `DBL_MAX` it returns 0. These
  values are pinned in `tests/cpp/stress.cpp` (`StressLimit`). With `k = C - 1` every
  competitor is capped and the loss takes the mean of the score differences: it avoids
  intermediate overflow when the exact mean is representable, and its double-double sum
  resolves about twice the working precision. Differences spanning more lose their small
  part: for `[M/4, H, M, 1, -H]`, `H = 2^200`, `M = 2^100`, label 0, `k = 4` (exact mean
  1/4), every route returns `log 5` = 1.6094 instead of 1.8142.
- **Top-k and proximal cross-entropy tails** retain the target mass independently of
  rounded competitor entries. A representable loss or target gradient can survive even
  when every individual competitor rounds to zero. JAX CPU still flushes subnormal
  values at the XLA interface; normal aggregates are retained.

### Aggregate-tail limitations

These inputs lie outside the validation corpus and fail its budget. They are not
regressions.

- **CUDA distillation, default FP32 work:** with 8,192 classes, student
  `[0, -110, ...]`, teacher `[0, -220, ...]` and temperature 1, the true target
  gradient is approximately `-1.38339e-44`, but the implementation returns zero.
  The loss `1.40130e-44` remains within one subnormal quantum. Each competitor
  gradient lies below FP32 range; residual terms round before their target
  aggregate is formed. The same failure appears at 32,000 and 128,256 classes,
  with target magnitudes about `5.40e-44` and `2.17e-43`. This exceeds the stated
  numerical budget. No effect on training is known at these scales.
- **JAX GPU, default FP32 work:** individually underflowed exponentials can be
  lost before their representable sum is formed. A 1,000-class single-positive
  mask with competitor gap 104 returns zero loss and target gradient instead of
  approximately `6.807e-43`. Distillation at temperature 2 and gap 208 has the
  same issue. BF16 gradient storage cannot retain these values, but the FP32
  loss is affected. Explicit float64 working precision passes these probes.
  This differs from the documented JAX CPU output-interface subnormal flushing.

### Platform ranges

The extended-precision retries use long double, which is wider than double only on x87
(Linux and Windows on x86-64 with GCC or Clang). Where it is double (Apple arm64, MSVC),
top-k entropy blocks solve to `C ‖x‖² = 1e20` and may raise `RuntimeError` beyond, and a
small mass beside a far-off pair in the two-simplex is resolved to a gap of 1e15
(`"extended"` projections: 1e13) instead of 1e16 (`tests/cpp/regression.cpp`). Inside
these ranges results agree with x87.

### Centered feature-space certificate

With a regularizer center `W0`, the evaluator forms `W - W0` directly, so a large common
center enters neither the objective values nor their error estimate
(`test_centered_certificate.py`).
