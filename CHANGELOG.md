# Changelog

## 0.3.1 (2026-11-12)

- **CUDA hinges:** For `C <= 128`, the fused multiclass, smooth and top-*k* hinge kernels pack
  several rows per CTA, as the top-*k* cross-entropy kernels already do. Forward plus backward
  on an A100 (`bfloat16`): `multiclass_hinge` at `262144 × 10` takes `0.20` ms (was `0.52`),
  the smooth hinge `0.27` ms (was `2.21`), and `topk_hinge_alpha` ($k = 5$) `0.17` ms (was
  `2.76`). Values and gradients are unchanged, and wider rows keep the existing kernels.
- **Name and description:** The project is described everywhere by the technical report's
  title, *libsdca: Accurate and Fast Classification Losses for SDCA and Neural Networks*
  (README, PyPI, crates.io, module docs), and report links go to
  <https://mlapin.github.io/libsdca/report/>.
- **Citation:** `CITATION.cff` and the README cite the technical report and the TPAMI 2018
  paper; softmax collapse is credited to Prieto et al. (2025).
- **Docs:** Fixed GitHub math rendering, wide tables and line wrapping.
- **Packaging:** The crate is published through crates.io Trusted Publishing, and
  `tools/quality.py` checks the `CITATION.cff` version and date and the release-pinned links
  of the PyPI and crates.io pages.

## 0.3.0 (2026-11-09)

### Highlights

- **Unified interfaces:** Python package (`fit`, `loss_and_grad`, `coordinate_update`),
  scikit-learn estimator (`SDCAClassifier`), PyTorch modules and functions, JAX loss
  functions, Rust crate, C ABI (`include/sdca/c_api.h`), C++11 headers, and MATLAB/Octave
  gateways sharing one set of objective, option and status names ([API.md](docs/API.md)).
- **Numerical stability:** Losses and gradients preserve tiny probability tails down to the
  subnormal limit (avoiding softmax collapse), and distillation gradients are exact zero at
  equal logits ([numerics](docs/numerics.md),
  [known limits](docs/numerics.md#aggregate-tail-limitations)).
- **Fused CUDA and CPU kernels:** On an A100 (`8192 × 32000` and `8192 × 128256`),
  cross-entropy takes `0.93–0.98×` `torch.compile` time (`0.33–0.41×` eager), top-5
  cross-entropy `0.23–0.26×`, and distillation `0.62–0.81×` ([performance](docs/performance.md)).
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
- **Double iterate for `float32` fits (`precision="float64"` default):** Keeps internal
  solver weights and duals in `double` (`8(d + n)m` bytes) and rounds once at each
  evaluation so `float32` fits reach `float64` gap tolerances down to `float32` output
  rounding (`3.5e-7` on `digits`; pass `precision="float32"` for pure `float32` speed and
  memory; [numerics.md](docs/numerics.md#what-each-precision-guarantees)).
- **Compensated reductions & hardened projections:** Compensated sums in multilabel
  cross-entropy and distillation, warm-start residual preservation at large score offsets,
  and checked `float64` / extended-precision `multilabel_hinge` projections
  ([API.md](docs/API.md#projection-precision)).

### Packaging and build

- Portable `py3-none` wheels for Linux `x86_64` (`manylinux_2_28`) and macOS 11+ (`arm64`)
  with built-in BLAS; configurable BLAS (`SDCA_BLAS_BACKEND=AUTO|EXTERNAL|BUILTIN`);
  CMake 3.12+ package config; and GitHub Actions CI on Linux and macOS.
