# Changelog

## 0.3.1 (2026-11-16)

- **JAX on CUDA:** `libsdca.jax` runs the same fused Triton kernels as PyTorch for every
  loss (Linux wheels include the handler), falling back to `jax.numpy` with a warning.
- **Shared CPU distillation:** PyTorch and JAX use the same fused CPU row kernels for
  distillation in all four dtypes; JAX proximal distillation uses the shared native solver.
  An overflowing native proximal-distillation loss now returns `+inf` with a finite
  gradient instead of `nan`.
- **`float32` work by default:** NumPy, scikit-learn, Rust and the C++ loss helpers default
  to `float32` work for `float32` inputs; `precision="float64"` keeps double work. Proximal
  cross-entropy and CUDA top-1 hinges use `float32` work for 16- and 32-bit inputs, promoting
  rows that need double ([API.md](docs/API.md#precision)).
- **Hinge ties:** Nonsmooth top-1 hinges select the largest represented score before forming
  the margin, so tiny score gaps no longer round to ties.
- **Aggregate tails:** CUDA distillation, proximal distillation and multilabel
  cross-entropy, and the JAX array losses, keep sums of individually underflowing
  exponentials.
- **Faster CUDA paths:** Adaptive distillation kernels and packed short hinge rows.
- **JAX CPU fusion:** `SDCA_JAX_CPU_FUSION=auto` times fused and fenced array programs once
  per machine and caches the choice ([JAX.md](python/JAX.md#cpu-pass-fusion)).
- **Packaging:** JAX-only installs no longer need PyTorch; the `torch` extra works with
  PyTorch 2.1; transient build failures no longer suppress retries for 24 hours. The
  version is read from one `VERSION` file.
- **Clearer errors:** PyTorch losses other than `cross_entropy` raise `ValueError` inside
  `torch.func` transforms, and all losses raise `ValueError` for tensors other than CPU or
  CUDA (MPS failed with a `TypeError`). `gamma` and `alpha` reject booleans. Fallback
  warnings point at the calling line.
- **Fixes:** `cross_entropy(..., backend="torch")` accepts nonfinite logits in ignored rows,
  as the other backends do. A solve whose final evaluation throws ends as `failed` instead
  of raising. The JAX CPU build falls back to portable flags when the native-ISA probe fails.
  `float64` proximal cross-entropy compensates its row sums: on 4,097–32,768 classes, values
  and gradients were up to `95×` the error bound and are now within it.
- **Docs and citation:** The README starts with the installation and shows the same two losses
  in every interface; [TORCH.md](python/TORCH.md#benchmarks) lists CUDA benchmarks;
  `CITATION.cff` prefers the TPAMI 2018 paper.
- **Known limits:** [API.md](docs/API.md#known-limits) lists the recorded `float64`
  cancellation errors (below `5e-17` absolute) and the behavior of finite rows at
  `±finfo.max`.

## 0.3.0 (2026-11-09)

### Highlights

- **Unified interfaces:** Python package (`fit`, `loss_and_grad`, `coordinate_update`),
  scikit-learn estimator (`SDCAClassifier`), PyTorch modules and functions, JAX loss
  functions, Rust crate, C ABI (`include/sdca/c_api.h`), C++11 headers, and MATLAB/Octave
  gateways sharing one set of objective, option and status names ([API.md](docs/API.md)).
- **Numerical stability:** Losses and gradients preserve tiny probability tails down to the
  subnormal limit (avoiding softmax collapse), and distillation gradients are exact zero at
  equal logits ([known limits](docs/API.md#known-limits)).
- **Fused CUDA and CPU kernels:** On an A100 (`8192 × 32000` and `8192 × 128256`),
  cross-entropy takes `0.93–0.98×` `torch.compile` time (`0.33–0.41×` eager), top-5
  cross-entropy `0.23–0.26×`, and distillation `0.62–0.81×`.
- **Certified convergence:** `status == "solved"` requires the duality gap of the returned
  weights and the feasible projection of the returned duals, evaluated in `double` with a
  first-order rounding estimate, to meet
  `tol` ([API.md](docs/API.md#results)).

### Interfaces and losses

- **PyTorch (`libsdca`, `libsdca.torch`):** `CrossEntropyLoss` (drop-in for
  `torch.nn.CrossEntropyLoss`, except integer class labels only and no label smoothing), `TopKCrossEntropyLoss`, `MulticlassHingeLoss`,
  `TopKHingeLoss` ($\alpha$ and $\beta$), `MultilabelCrossEntropyLoss`,
  `MultilabelHingeLoss`, `ProximalCrossEntropyLoss`, `DistillationLoss`,
  `ProximalDistillationLoss`, `SDCALoss`, and `LinearCrossEntropyLoss` (chunked linear head
  + cross-entropy without materializing `(N, V)` logits) across `float16`, `bfloat16`,
  `float32` and `float64` on CPU and CUDA, with `torch.compile(fullgraph=True)` and
  `DTensor` support ([TORCH.md](python/TORCH.md)).
- **JAX (`libsdca.jax`):** Classification and distillation losses for `jit`, `grad` and
  `vmap`, with zero-copy XLA FFI custom calls on CPU ([JAX.md](python/JAX.md)).
- **Python & scikit-learn:** `fit`, `loss_and_grad`, `distillation_loss_and_grad`,
  `proximal_distillation_loss_and_grad`, `coordinate_update`, and `SDCAClassifier`
  ([python/README.md](python/README.md)).
- **Rust, C, C++ and MATLAB:** Rust crate `libsdca` ([rust/README.md](rust/README.md)),
  versioned C ABI `libsdca::sdca_c` ([docs/C.md](docs/C.md)), CMake target
  `libsdca::libsdca` ([docs/CPP.md](docs/CPP.md)), and MEX gateways `matsdca_fit`,
  `matsdca_loss` and `matsdca_prox` ([docs/MATLAB.md](docs/MATLAB.md)).

### Performance and solver improvements

- **CUDA kernels:** Fused Triton kernels for all losses with C++ `cuLaunchKernel` autograd
  dispatch, multi-row CTA tiling for short rows, streaming tiles up to $2^{25}$ classes, and
  in-place backward gradient writes into unreferenced logits tensors.
- **CPU kernels:** JIT-compiled ATen/OpenMP C++ extension with C++ autograd nodes and SIMD
  `float32` exponentials + `float64` row statistics for narrow cross-entropy and top-*k*
  cross-entropy ($k \le 64$).
- **Shrinking and screening (`shrinking="auto"`):** Combines LIBLINEAR-style active-set
  shrinking with gap-safe screening for hinge fits, adapting the evaluation schedule to
  shrunk epoch work (`0.14–0.59×` `LinearSVC` runtime; [API.md](docs/API.md#shrinking)).
- **$O(n \log n)$ projections and Taylor root warm-starts:** Top-*k* cone, simplex, knapsack
  and top-*k* entropy projections run in $O(n \log n)$, and proximal / entropy root solves
  converge in `1.0–1.3` row evaluations on average.

### Numerics and precision

- **Single rounding:** 16-bit gradients are scaled by the upstream cotangent and reduction
  factor in working precision before a single rounding to the output dtype, and hinge / top-*k*
  target gradients are formed from the unrounded competitor sum.
- **Optional double iterate for `float32` fits (`precision="float64"`):** Keeps internal
  solver weights and duals in `double` (`8(d + n)m` bytes) and rounds once at each
  evaluation so `float32` fits reach `float64` gap tolerances down to `float32` output
  rounding (`3.5e-7` on `digits`; pass `precision="float32"` for pure `float32` speed and
  memory).
- **Compensated reductions & hardened projections:** Compensated sums in multilabel
  cross-entropy and distillation, warm-start residual preservation at large score offsets,
  and checked `float64` / extended-precision `multilabel_hinge` projections
  ([API.md](docs/API.md#projection-precision)).

### Packaging and build

- Portable `py3-none` wheels for Linux `x86_64` (`manylinux_2_28`) and macOS 11+ (`arm64`)
  with built-in BLAS; configurable BLAS (`SDCA_BLAS_BACKEND=AUTO|EXTERNAL|BUILTIN`);
  CMake 3.12+ package config; and GitHub Actions CI on Linux and macOS.
