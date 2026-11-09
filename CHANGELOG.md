# Changelog

## 0.3.0 (2026-11-09)

### Highlights

- **New interfaces:** a Python package with a scikit-learn estimator, PyTorch and
  JAX losses, a Rust crate and a C ABI, sharing one set of objective, option and
  status names ([API.md](docs/API.md)).
- **Accuracy:** the losses and gradients keep tiny probability tails to the
  subnormal limit in the tested regimes, where PyTorch's cross-entropy returns zero.
  Default CUDA distillation and JAX GPU arithmetic retain documented extreme
  aggregate-tail exceptions ([numerics](docs/numerics.md#aggregate-tail-limitations)).
- **Speed:** fused CUDA and CPU kernels for every loss. On an A100, cross-entropy training
  on 8192×32000 and 8192×128256 logits takes 0.93–0.98× `torch.compile`'s time and
  0.33–0.41× eager PyTorch's. Top-5 cross-entropy takes 0.23–0.26× and distillation
  0.62–0.81× of compiled PyTorch for the same objective. On tall logits (10 to 1,000
  classes) libsdca is faster than `torch.compile` in 21 of 24 training cases. Slower cases
  are listed in [performance](docs/performance.md#where-libsdca-is-slower).
- **Certified training:** `solved` requires the duality gap of the returned
  weights and duals, evaluated in double, plus a first-order estimate of its
  rounding to meet `tol` ([API.md](docs/API.md#results)).

### New interfaces

- Python package `libsdca`: `fit` for every objective on features or kernels,
  with warm starts, float32 or float64 working precision, `loss_and_grad`,
  `coordinate_update` and the estimator `SDCAClassifier` (`libsdca.sklearn`).
- PyTorch losses as modules and functions ([TORCH.md](python/TORCH.md)):
  `CrossEntropyLoss`, a drop-in for `torch.nn.CrossEntropyLoss` with class
  weights, `ignore_index` and `(N, C, ...)` inputs; top-k cross-entropy; the
  multiclass, top-k and multilabel hinges; multilabel cross-entropy; proximal
  cross-entropy; distillation and proximal distillation; the generic `SDCALoss`;
  `linear_cross_entropy`, a language-model head's linear layer and cross-entropy in
  chunks of rows that never stores the logits.
  They run on CPU and CUDA tensors in all four float dtypes, under
  `torch.compile`, and accept `DTensor` inputs.
- JAX losses under `jit`, `grad` and `vmap` ([JAX.md](python/JAX.md)).
- Rust crate `libsdca` with the API of the Python package.
- A C ABI (`include/sdca/c_api.h`, `-DSDCA_BUILD_C_API=ON`) with versioned
  option structs and distinct error codes ([C.md](docs/C.md)).
- `projection_precision` (`"float64"` or `"extended"`) for the multilabel hinge
  projection ([numerics](docs/numerics.md#projection-precision)).
- `shrinking` (`"auto"`, `"off"`, `"heuristic"`, `"safe"`, `"both"`) for the hinge
  fits: liblinear-style shrinking, gap-safe screening, or both (the default);
  the entropy fits ignore it, and the certificate still covers every example
  ([API.md](docs/API.md#shrinking)). `struct sdca_options` gains it as its last
  field; structs of the earlier size take the default.
- `sdca::loss_and_gradient` in C++ and `matsdca_loss` in MATLAB.
- Loss and coordinate-step rows run on `num_threads` threads
  (`SDCA_NUM_THREADS`), with identical results for every thread count.
- Loss values have dtype `promote(logits dtype, float32)` and gradients keep the
  logits' dtype, in PyTorch and JAX.

### Performance

- **PyTorch on CUDA:** fused Triton kernels for every loss, without host
  synchronization (CUDA-graph capturable), launched from C++ autograd except for
  the proximal losses, which launch from Python. Cross-entropy
  keeps a few scalars per row instead of an `(N, C)` tensor between forward and
  backward, and its backward writes the gradient over the logits when nothing
  else holds them: an eager LLM-head training step (bfloat16, 8192×128256) peaked
  at 1066 MiB above its starting allocation, compared with 2068 MiB inside
  `torch.compile` and 8080 MiB for eager PyTorch. Rows of up to 2²⁵
  classes stream through column chunks; short rows (up to 64 classes in cross-entropy,
  up to 256 in top-k CE, masked CE and distillation) are packed several per thread
  block. Distillation inference skips the gradient statistics. `linear_cross_entropy`
  takes larger default chunks in bfloat16; `chunk_bytes` sets the memory budget
  ([measurements](docs/performance.md#pytorch-on-cuda)).
- **JAX on the GPU:** masked CE, and distillation on rows of bounded logit range, read
  the logits three times instead of five.
- **PyTorch on the CPU:** an ATen/OpenMP extension for every loss, built on first
  use for the ISA of PyTorch's own kernels, with C++ autograd nodes and
  `torch.compile(fullgraph=True)` support; the prebuilt native library serves as the
  fallback. Narrow cross-entropy, masked cross-entropy and top-k cross-entropy
  (`k <= 64`) rows use one sweep of float32 exponentials with float64 row statistics,
  and distillation forms every gradient entry independently
  ([details](python/TORCH.md#devices-and-backends)).
- **Proximal losses on the CPU:** the root of each row from a Taylor model of the
  probabilities about the last evaluation, in 1.0–1.3 evaluations of the row on
  average at `alpha <= 10`.
- **Shrinking:** hinge fits skip examples whose duals stay unchanged, step active
  examples only on their nonzero-dual classes, and screen out duals that the certified
  gap proves zero. Scheduled evaluations adapt to the shrunk epoch cost, cutting the
  evaluation overhead of tight fits ([performance](docs/performance.md#solver)).
- **Solver:** per-example steps update only the classes whose dual changed (epochs
  up to 4.5× faster with an external BLAS); entropy steps reuse their root solve
  (cross-entropy epochs 15–29% faster); `fit` stops at the epoch the gap reaches
  `tol` (cross-entropy fits to 1e-3 in 0.48–0.73 of the time); multithreaded gap
  evaluations with identical results.
- **Projections:** top-k cone, simplex, knapsack and top-k entropy searches take
  O(n log n) instead of O(kn): at n = 100k, k = 50k from seconds to milliseconds.
- **JAX:** the native losses on the CPU are an XLA FFI call into the native
  library (built on first use against jaxlib's headers; `SDCA_JAX_FFI`), with no
  host round trip: 0.045–0.44 of the host callback's time from 4096×1000 and
  0.13–0.69 at 256×10 with 32 threads, with the same results.
  `backend="auto"` on the CPU takes the jax backend where the native call's
  fixed cost dominates. On the CPU the entropy, distillation and
  `k = 1` hinge losses reduce each row in fused variadic passes (a row's sum with
  its tie and label counts) instead of materialized reductions.

### Numerical fixes

- No softmax collapse: entropy losses and dual updates stay accurate at high
  confidence, and the zero-sum constraint holds at any score offset.
- 16-bit gradients round once, after the upstream factor (a GradScaler's 65536 lost
  up to a third of subnormal entries). The target entry of the hinge and top-k
  gradients is the unrounded sum of the others: float16 `topk_cross_entropy` was
  off by up to 16%.
- Top-k entropy: accurate for large `C`, cap ties, large offsets and tiny tails;
  scores far below a cap keep their differences (dual errors were up to 0.17).
- Warm-started entropy and hinge steps keep the small residual between huge
  cancelling scores and duals.
- Multilabel cross-entropy sums its rows compensated: well-fit rows (many equal
  small terms beside a 1) lost up to 772 float64 eps of the loss.
- Hinge losses keep margins down to 1e-20, are invariant to a common offset, and
  stay finite at huge and tiny `gamma`.
- Knapsack, two-simplex, top-k cone and simplex projections and the Lambert W
  function: tiny scores and caps kept, no overflow near 1e308, no tolerance floors.
- Kernel fits evaluate the primal at the duals they return; kernels must be
  symmetric with a valid diagonal. `SDCAClassifier(warm_start=True)` handles a
  changed `C`.

Details and tested regimes: [numerics.md](docs/numerics.md).

### Breaking and behavior changes

- `solved` is a certificate: the gap, evaluated in double from `X'A`, plus a
  first-order estimate of its own rounding must meet `tol`. A failed step ends a fit as
  `failed`, leaving a consistent warm start, instead of raising.
- One build with IEEE semantics (`-fno-fast-math -ffp-contract=off`):
  `USE_FAST_MATH` and `SDCA_ACCURATE_MATH` are removed;
  `recompute_primal` replaces the latter at run time.
- `precision` controls both working arithmetic and the solver's iterate. For
  float32 data, the default `precision="float64"` now keeps `W` and the duals in
  double (`8 (d + n) m` additional bytes), rounding them once into the caller's
  float32 arrays at each evaluation and certifying the rounded values. This avoids
  float32 iterate stalling near the optimum (reaching `3.5e-7` relative gap on
  `digits`, vs. `4.7e-6` previously; [numerics](docs/numerics.md#what-each-precision-guarantees)).
  Consequently, float32 fits with default `precision` return tighter iterates than
  earlier trees; float64 fits and explicit `precision="float32"` fits are bitwise
  unchanged. In C++, `context.primal_variables` and `context.dual_variables` hold
  the double iterate for `<float, double>`, while the rounded results are written
  to the caller's buffers.
- Every loss checks that `k` is an integer in `[1, classes − 1]`.
- PyTorch: empty inputs return `torch.nn.functional`'s results instead of raising;
  `backend="torch"` always means tensor operations.
- MATLAB: `opts.precision` no longer accepts `'float'` (use `'single'`); model
  precisions read `'float32'`/`'float64'`; `matsdca_fit` defaults to
  `cross_entropy`; `topk_hinge_alpha` and `topk_hinge_beta` default to
  `gamma = 0` (`l2_hinge_topk` and `l2_topk_hinge` keep `gamma = 1`);
  `matsdca_prox` no longer modifies its input.
- C++: `l2_hinge_topk_smooth`, `l2_topk_hinge_smooth` and their `make_objective_*`
  functions take `(c, k, gamma)`; the nonnegative-feature solver is removed; the
  bundled CBLAS declarations moved to `include/sdca/math/cblas.h`.

### Build

- Python wheels for Linux x86-64 (manylinux_2_28) and macOS 11+ arm64 with a
  built-in BLAS; other platforms build the sdist.
- BLAS is optional: `SDCA_BLAS_BACKEND=AUTO|EXTERNAL|BUILTIN`.
- CMake 3.12+, an exported `libsdca::libsdca` target, opt-in tests and MATLAB.
- Tests live in `tests/cpp`, `tests/python` (pytest, parallel with `-n auto`) and
  `tests/matlab`; `tools/run_cpp_tests.sh --sanitize` adds ASan, UBSan and TSan.

## Earlier releases

See the [GitHub releases](https://github.com/mlapin/libsdca/releases).
