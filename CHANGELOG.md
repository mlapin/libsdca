# Changelog

## 0.4.0 (2026-11-27)

Changes since 0.3.0. Every interface has a new API with one set of loss names, options and
defaults ([API.md](docs/API.md)). The [technical report](https://mlapin.github.io/libsdca/report/)
gives the benchmarks and numerical comparisons.

### Upgrading from 0.3.0

- **Names and parameters:** rename the losses and pass their parameters as below (top-*k*
  losses need `k`; PyTorch and JAX take hyperparameters by keyword).
- **`float32` data works in `float32`:** NumPy losses and gradients of `float32` inputs
  differ from 0.3.0's by up to about half a `float32` ulp, and fits of `float32` data can
  stop at a larger gap. `precision="float64"` reproduces 0.3.0's double work
  ([API.md](docs/API.md#precision)).
- **Invalid values no longer raise:** `NaN` or `+inf` logits and invalid rows of masked
  classes give `NaN` rows (NumPy warns once per call; `check_rows=True` raises), and a logit
  of `-inf` removes its class. Labels out of range raise PyTorch's
  `Target 7 is out of bounds.` in NumPy, Rust, C, C++ and synchronous CPU calls; CUDA and
  traced composed calls produce `NaN` rows.
- **Distillation teachers get gradients:** a teacher that requires gradients (PyTorch) or is
  differentiated (JAX) now receives PyTorch's gradient; detach it to keep 0.3.0's behaviour.
- **Linux wheels need `scipy-openblas32`:** they link its OpenBLAS, and `pip` installs it
  with them ([API.md](docs/API.md#blas)).
- **Zeros and empty batches:** gradient entries that were `-0.0` are `+0.0`, and NumPy batches
  of zero rows return empty arrays (0.3.0 raised `ValueError`).
- **Error texts:** every interface uses one text per error, PyTorch's where PyTorch has one.
- **Memory:** PyTorch backward writes the gradient over logits that nothing else references;
  code that keeps a reference to the logits gets a warning and a new gradient
  ([TORCH.md](docs/TORCH.md#memory)).

### Breaking changes

- **One loss vocabulary:** every interface names the losses `cross_entropy`,
  `topk_cross_entropy`, `multiclass_hinge`, `topk_hinge` (`variant="beta"`, the default, or
  `"alpha"`), `multilabel_hinge`, `multilabel_cross_entropy`, `proximal_cross_entropy`,
  `distillation` and `proximal_distillation`. Each takes only its own parameters (`k`,
  `variant`, `gamma`, `temperature`): a parameter of another loss is a `TypeError`. String
  aliases (`softmax`, `fp32`, ...) and the Python and Rust names `topk_hinge_alpha` and
  `topk_hinge_beta` are gone; the C kinds, C++ kinds and C++ objectives keep them.
- **Top-*k* losses need `k`:** `topk_cross_entropy` and `topk_hinge` have no default `k`
  (0.3.0 used `k = 1`); omitting it is a `TypeError` in Python, and `k = 0` is an invalid
  argument in C and C++ (Rust's `"topk_hinge".parse()` still gives `k = 1`).
- **`gamma` is the step of the proximal losses:** `proximal_cross_entropy` and
  `proximal_distillation` take their Moreau step as `gamma` (default `1`; required in C++'s `sdca::loss_parameters`)
  instead of `alpha`.
- **NumPy:** losses are objects: `fit(X, y, libsdca.TopKHinge(k=2))`,
  `loss_and_grad(logits, target, loss)` and `coordinate_update(logits, target, loss)`.
  `libsdca.Kernel(K)` marks a Gram matrix and `libsdca.Probabilities(P)` probability targets.
  `Solver`, `distillation_loss_and_grad` and `proximal_distillation_loss_and_grad` are gone,
  and `import libsdca` no longer exports the PyTorch losses.
- **scikit-learn:** `SDCAClassifier(loss, *, C, k, variant, gamma, tol, max_iter, eval_epoch,
  shrinking, kernel, precision, n_jobs, random_state, warm_start)`: `objective` is now
  `loss`, `max_epoch` is `max_iter`, and `recompute_primal` and `projection_precision` are
  gone.
- **PyTorch and JAX:** one function (and in PyTorch one module) per loss, with keyword-only
  hyperparameters. `SDCALoss`, `sdca_loss`, `TopKHingeAlphaLoss` and `TopKHingeBetaLoss` are
  gone: use `TopKHingeLoss(k=2, variant="alpha")`.
- **Classes on `dim=1` in every PyTorch loss** (`dim=0` for 1-D logits), as in
  `F.cross_entropy`; the losses other than `cross_entropy` defaulted to `dim=-1`. JAX keeps
  `axis=-1`.
- **Backends `"auto"`, `"kernel"` and `"composed"`:** `"auto"` (the default) and `"kernel"`
  run libsdca's kernels and raise `NotImplementedError` naming `backend="composed"` where no
  kernel computes a call, instead of falling back to tensor operations. `"composed"` runs
  framework operations on any device. The `"native"`, `"torch"` and `"jax"` backends are
  gone; `explain_backend` reports the path of a call in both frameworks.
- **Kernel derivatives are first order:** differentiating a kernel gradient again, or forward
  mode, raises; `backend="composed"` supports them
  ([TORCH.md](docs/TORCH.md#compilation-and-transformations)).
- **Invalid numerical inputs give `NaN`:** in PyTorch and JAX, `NaN` or `+inf` logits,
  invalid class weights and invalid rows of masked classes give `NaN` rows, with no host
  synchronization. NumPy's `loss_and_grad` writes `NaN` rows and issues one
  `InvalidRowsWarning` per call (`check_rows=True` raises).
- **`float32` work by default:** NumPy, scikit-learn, Rust and the C++ loss helpers work in
  `float32` for `float32` inputs; `precision="float64"` selects double work.
- **One array layout:** NumPy, Rust, C and C++ share row-major features `(n, d)`, weights
  `(m, d)` and duals `(n, m)` ([API.md](docs/API.md#array-layouts)). NumPy `Model.weights`
  and `center` were Fortran-ordered `(d, m)`.
- **Training options:** `eval_epoch = 0` evaluates only at the end, and `seed` sets the
  order of the examples. The statuses are `solved`, `no_progress`, `max_epoch` and `failed`
  (`none` and `solving` are gone).
- **C ABI 7** (0.3.0: 5): `sdca_loss_evaluate`, `sdca_loss_rows`, `sdca_dual_update`,
  `sdca_model_fit`, `sdca_model_scores`, `sdca_history_rows`, `sdca_max_extent` and
  `sdca_blas_backend` replace the 0.3.0 entry points, with the structs `sdca_loss`
  (`SDCA_LOSS_INIT(kind)`), `sdca_execution`, `sdca_targets`, `sdca_fit_options` and
  `sdca_input`. `sdca_loss_evaluate` returns `SDCA_INVALID_ROWS` (`1`) when it writes `NaN`
  rows ([C.md](docs/C.md)).
- **Rust:** `Loss` variants carry their parameters (`Loss::topk_hinge(2)`); a `Solver`
  builder trains and takes coordinate steps, and `Settings` hold the precision, threads and
  `check_rows`. `Loss::loss_and_grad` also returns a `#[must_use]` `LossReport`
  ([RUST.md](docs/RUST.md)).
- **C++:** the objective factories use the shared names and take `gamma`; the loss kinds are
  `sdca::kernel::Kind`, and `sdca::loss_kind`, `sdca::loss_name` and `sdca/loss_kind.h` are
  gone. Labels are 0-based, `make_input_feature(n, d, X)` and
  `make_input_kernel(n, n_train, K)` take the number of examples first, and the headers need
  no `logging.cpp` ([CPP.md](docs/CPP.md)).

### New features

- **Class weights, label smoothing and `ignore_index` everywhere:** NumPy, Rust and the C ABI
  take PyTorch's `cross_entropy` options, for the same losses as PyTorch and JAX
  ([API.md](docs/API.md#loss-evaluation-options)).
- **CPU losses without build tools:** where the PyTorch or JAX CPU extension cannot be built,
  CPU calls run the same kernels in the prebuilt native library after one `RuntimeWarning`
  ([TORCH.md](docs/TORCH.md#build-tools), [JAX.md](docs/JAX.md#build-tools)).
- **Probability targets for `cross_entropy`** in every interface, with class weights and label
  smoothing as in PyTorch, and PyTorch's gradient for targets that require one.
- **PyTorch cross-entropy parity:** `CrossEntropyLoss` and `cross_entropy` take label
  smoothing, probability targets and labels of any integer dtype; `LinearCrossEntropyLoss`
  takes `class_weight`, label smoothing and probability targets; the other single-label
  losses take `weight` and `ignore_index`. Every PyTorch loss works under `torch.func`.
- **Masked classes:** a score of `-inf` masks its class in every loss, interface and backend
  ([API.md](docs/API.md#masked-classes)).
- **Mean subgradient at kinks:** at ties of the nonsmooth hinges the gradient splits evenly
  among the tied classes, as optax and `torch.amax` do; rows without ties are unchanged
  ([API.md](docs/API.md#loss-definitions)).
- **`reuse_logits`** (PyTorch, default `True`): an eager backward writes the gradient over
  logits that nothing else references, and warns once when something does
  ([TORCH.md](docs/TORCH.md#memory)).
- **`libsdca.optax`:** optax's `softmax_cross_entropy_with_integer_labels`,
  `softmax_cross_entropy`, `safe_softmax_cross_entropy` and `multiclass_hinge_loss` with
  their signatures, on the libsdca kernels ([JAX.md](docs/JAX.md#optax-losses)).
- **JAX on CUDA:** `libsdca.jax` runs the same fused Triton kernels as PyTorch for every loss.
- **Composed losses:** `backend="composed"` compiles as one graph under `torch.compile` and
  avoids softmax collapse as the kernels do.
- **Batch scores:** `sdca_model_scores` (C), `sdca::batch_scores` (C++) and Rust's
  `Model::decision_function` score a batch with one BLAS `gemm`.
- **NumPy:** `fit` takes `num_threads` and `seed`.

### Behaviour changes

- **Labels out of range are errors, as in PyTorch** in NumPy, Rust, C and C++
  (`Target 7 is out of bounds.`).
- **Probability targets of a loss evaluation are taken as given,** as by PyTorch's
  `cross_entropy` (training still requires rows on the simplex).
- **One BLAS rule for every build:** `pip install`, the Rust crate and CMake link the first
  system CBLAS that links (Accelerate, OpenBLAS, CBLAS, BLAS), else the built-in products with
  a warning; `SDCA_BLAS` and `SDCA_BLAS_BACKEND` override it ([API.md](docs/API.md#blas)).
- **One set of CPU kernels:** PyTorch, JAX, NumPy, Rust, C and C++ share the same header-only
  kernels; results differ from 0.3.0 in the last bits.
- **Subnormals are kept:** the CPU kernels clear flush-to-zero around their calls, and their
  exponentials reach the subnormal range.
- **Reductions stay finite:** a `mean` of finite rows whose totals leave the range stays
  finite in PyTorch and JAX.
- **Distillation** forms its gradient from the log-ratio of student and teacher logits
  (Barron et al., 2026): equal logits give an exact zero gradient.
- **First-use builds** are cached in `SDCA_CACHE_DIR`; a failed build is skipped for 24 hours.
  JAX-only installs no longer need PyTorch.
- **NumPy and device tensors:** the NumPy API copies CUDA tensors to the host and warns once,
  pointing to `libsdca.torch`.

### Performance

- **Run-time instruction sets:** the CPU kernels compile scalar, AVX2 and AVX-512 code in
  one translation unit and choose at run time (NEON on arm64); `SDCA_CPU_CAPABILITY` lowers
  the choice ([API.md](docs/API.md#environment-variables)).
- **Threads:** loss evaluations split their rows by an estimate of their work
  ([API.md](docs/API.md#threads)).
- **One CPU kernel pass per training step:** a CPU sum or mean whose logits need a gradient
  computes values and gradient in one pass.
- **CUDA:** fused cross-entropy, distillation and linear cross-entropy kernels with
  specialized narrow and medium row paths, and no copies of leaf or referenced logits.

### Removals

- **`sdca/prox.h`:** `prox_two_simplex_sort`, `thresholds_two_simplex_sort` and undocumented
  internals are removed; `sdca::prox(t, first, last)` is documented
  ([CPP.md](docs/CPP.md#projections-and-proximal-operators)).
- **BLAS options (C++):** `BLAS_MKL`, `BLAS_ACCELERATE`, `BLAS_DEFAULT`, `USE_ILP64`,
  `USE_SEQUENTIAL` and `cmake/FindMKL.cmake` are gone (link MKL with `SDCA_BLAS=-lmkl_rt`).
- **MATLAB and Octave:** the MEX gateways, `docs/MATLAB.md` and `SDCA_BUILD_MATLAB` are gone.
- **`projection_precision`:** gone from every interface.
- **`SDCA_CUDA_LAUNCHER`:** gone; Triton launches the CUDA kernels.

## 0.3.0 (2026-11-09)

### Highlights

- **Unified interfaces:** Python package (`fit`, `loss_and_grad`, `coordinate_update`),
  scikit-learn estimator (`SDCAClassifier`), PyTorch modules and functions, JAX loss
  functions, Rust crate, C ABI (`include/sdca/c_api.h`) and C++11 headers sharing one set
  of objective, option and status names ([API.md](docs/API.md)).
- **Numerical stability:** Losses and gradients preserve tiny probability tails down to the
  subnormal limit (avoiding softmax collapse), and distillation gradients are exact zero at
  equal logits ([known limits](docs/API.md#known-limits)).
- **Fused CUDA and CPU kernels:** On an A100 (`8192 × 32000` and `8192 × 128256`),
  cross-entropy takes `0.93–0.98×` `torch.compile` time (`0.33–0.41×` eager), top-5
  cross-entropy `0.23–0.26×`, and distillation `0.62–0.81×`.
- **Duality-gap guarantee:** `status == "solved"` requires the duality gap of the returned
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
  `DTensor` support ([TORCH.md](docs/TORCH.md)).
- **JAX (`libsdca.jax`):** Classification and distillation losses for `jit`, `grad` and
  `vmap`, with zero-copy XLA FFI custom calls on CPU ([JAX.md](docs/JAX.md)).
- **Python & scikit-learn:** `fit`, `loss_and_grad`, `distillation_loss_and_grad`,
  `proximal_distillation_loss_and_grad`, `coordinate_update`, and `SDCAClassifier`
  ([docs/PYTHON.md](docs/PYTHON.md)).
- **Rust, C and C++:** Rust crate `libsdca` ([docs/RUST.md](docs/RUST.md)), versioned C
  ABI `libsdca::sdca_c` ([docs/C.md](docs/C.md)) and CMake target `libsdca::libsdca`
  ([docs/CPP.md](docs/CPP.md)).

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
  and checked `float64` / extended-precision `multilabel_hinge` projections.

### Packaging and build

- Portable `py3-none` wheels for Linux `x86_64` (`manylinux_2_28`) and macOS 11+ (`arm64`)
  with built-in BLAS; configurable BLAS (`SDCA_BLAS_BACKEND=AUTO|EXTERNAL|BUILTIN`);
  CMake 3.12+ package config; and GitHub Actions CI on Linux and macOS.
