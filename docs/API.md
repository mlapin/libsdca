# API reference

One set of loss names, option names, status codes and defaults is shared by C++, the C ABI
(`include/sdca/c_api.h`), Python, PyTorch, JAX, Rust and MATLAB. Interface guides:
[Python](../python/README.md), [PyTorch](../python/TORCH.md), [JAX](../python/JAX.md),
[Rust](../rust/README.md), [C++](CPP.md), [C](C.md) and [MATLAB](MATLAB.md).

## Loss definitions

For scores $z \in \mathbb{R}^m$, target class $y$, and non-target margins
$h_j = 1 + z_j - z_y$ ($j \neq y$) sorted as $h_{[1]} \ge h_{[2]} \ge \dots$
(or positive label set $Y \subset \lbrace 1, \dots, m \rbrace$ for multilabel losses):

| Name | Formula | Parameters | Use case |
| --- | --- | --- | --- |
| `cross_entropy` | $\log \sum_j \exp(z_j) - z_y$ | — | Multiclass softmax classification |
| `topk_cross_entropy` | $\max_{p \in \Delta_k} \langle a, p \rangle + H(\bar p)$ | `k` | Top-*k* entropy [4] (`cross_entropy` at $k=1$) |
| `multiclass_hinge` | $\max(0, h_{[1]})$ | `gamma` | Crammer–Singer SVM [1] |
| `topk_hinge_alpha` | $\max\big(0, \frac{1}{k}\sum_{i \le k} h_{[i]}\big)$ | `k`, `gamma` | Top-*k* SVM on average of $k$ worst margins [3] |
| `topk_hinge_beta` (`topk_hinge`) | $\frac{1}{k}\sum_{i \le k} \max(0, h_{[i]})$ | `k`, `gamma` | Top-*k* SVM on each of $k$ worst margins [4] |
| `multilabel_hinge` | $\max\big(0, 1 + \max_{j \notin Y} z_j - \min_{y \in Y} z_y\big)$ | `gamma` | Ranked multilabel SVM [2] |
| `multilabel_cross_entropy` | $\log \sum_j \exp(z_j) - \frac{1}{\lvert Y \rvert}\sum_{y \in Y} z_y$ | — | Cross-entropy to uniform target on $Y$ |
| `proximal_cross_entropy` | $\mathrm{env}_\alpha[\mathrm{CE}(\cdot, y)]{(z)}$ | `alpha` | Moreau envelope of CE [5] |
| `distillation` | $H\big(\mathrm{softmax}(t/T), \mathrm{softmax}(z/T)\big)$ | `temperature` | Teacher-logit distillation [6] |
| `proximal_distillation` | $\mathrm{env}_\alpha[H(p, \mathrm{softmax}(\cdot))]{(z/T)}$ | `temperature`, `alpha` | Moreau-damped distillation in $z/T$ |
| `linear_cross_entropy` | $\mathrm{CE}(h W^\top + b, y)$ | `chunk_bytes`, `recompute` | Chunked linear head + CE (PyTorch) |

- **Top-*k* entropy:** Over $j \neq y$, $a_j = z_j - z_y$, $s = \sum_j p_j$ and
  $\Delta_k = \lbrace p \ge 0 : s \le 1, p_j \le s/k \rbrace$;
  $H(\bar p) = -(1 - s)\log(1 - s) - \sum_j p_j \log p_j$ is the entropy of $\bar p = (1 - s, p)$.
- **Moreau envelope:**
  $\mathrm{env}_\alpha[f]{(x)} = \min_u \big\lbrace f(u) + \frac{1}{2\alpha}\Vert u - x \Vert_2^2 \big\rbrace$,
  where $H(p, q) = -\sum_j p_j \log q_j$ is the cross-entropy and $p = \mathrm{softmax}(t/T)$.
- **Hinge smoothing (`gamma`):** `gamma = 0` (default) is the nonsmooth hinge;
  `gamma > 0` is its Moreau envelope $\max_{p \in S} \langle h, p \rangle - \frac{\gamma}{2}\Vert p \Vert_2^2$,
  which has a $(1/\gamma)$-Lipschitz gradient [4] and lies below the nonsmooth loss by at most
  $\gamma / (2k)$ for the single-label hinges ($\gamma / 2$ for `multiclass_hinge`) and
  $\gamma$ for `multilabel_hinge`.
- **Top-*k* exclusion:** Top-*k* hinges exclude the target class $y$ when selecting the $k$
  largest competitor margins $h_{[1]}, \dots, h_{[k]}$ [4]. At $k = 1$,
  `topk_hinge_alpha` and `topk_hinge_beta` coincide with `multiclass_hinge`.

## Objectives and losses

### Core entry points

| Interface | Train a classifier | Loss and gradient | Coordinate step |
| --- | --- | --- | --- |
| NumPy | `fit(X, y, loss)` | `loss_and_grad(logits, target, loss)` | `coordinate_update(logits, target, loss)` |
| scikit-learn | `libsdca.sklearn.SDCAClassifier` | — | — |
| PyTorch | — | one function and module per loss (`libsdca.torch`) | — |
| JAX | — | one function per loss (`libsdca.jax`) | — |
| Rust | `Solver::fit` | `Loss::loss_and_grad` | `Solver::coordinate_update` |
| C++ | `make_solver(ctx).solve()` | `loss_and_gradient`, `distillation_loss` | `update_dual_variables` |
| C | `sdca_model_fit` | `sdca_loss_evaluate` | `sdca_dual_update` |
| MATLAB | `matsdca_fit` | `matsdca_loss` | — |

Public C++ headers: `sdca/solver.h`, `sdca/prox.h` and `sdca/loss.h`
(`sdca::detail` and `_`-prefixed Python names are internal).

### Names across interfaces

Every interface uses the canonical names and takes, for each loss, only its own parameters
(`k`, `gamma`, `alpha`, `temperature`); a parameter of another loss is an error.

| Canonical name | NumPy | PyTorch module, PyTorch / JAX function | Rust | C |
| --- | --- | --- | --- | --- |
| `cross_entropy` | `CrossEntropy()` | `CrossEntropyLoss`, `cross_entropy` | `Loss::CrossEntropy` | `SDCA_LOSS_CROSS_ENTROPY` |
| `topk_cross_entropy` | `TopKCrossEntropy(k)` | `TopKCrossEntropyLoss`, `topk_cross_entropy` | `Loss::TopKCrossEntropy { k }` | `SDCA_LOSS_TOPK_CROSS_ENTROPY` |
| `multiclass_hinge` | `MulticlassHinge(gamma)` | `MulticlassHingeLoss`, `multiclass_hinge_loss` | `Loss::MulticlassHinge { gamma }` | `SDCA_LOSS_MULTICLASS_HINGE` |
| `topk_hinge_alpha` | `TopKHingeAlpha(k, gamma)` | `TopKHingeLoss`, `topk_hinge_loss` (`variant="alpha"`) | `Loss::TopKHingeAlpha { k, gamma }` | `SDCA_LOSS_TOPK_HINGE_ALPHA` |
| `topk_hinge_beta` | `TopKHingeBeta(k, gamma)` | `TopKHingeLoss`, `topk_hinge_loss` (`variant="beta"`) | `Loss::TopKHingeBeta { k, gamma }` | `SDCA_LOSS_TOPK_HINGE_BETA` |
| `multilabel_hinge` | `MultilabelHinge(gamma)` | `MultilabelHingeLoss`, `multilabel_hinge_loss` | `Loss::MultilabelHinge { gamma }` | `SDCA_LOSS_MULTILABEL_HINGE` |
| `multilabel_cross_entropy` | `MultilabelCrossEntropy()` | `MultilabelCrossEntropyLoss`, `multilabel_cross_entropy` | `Loss::MultilabelCrossEntropy` | `SDCA_LOSS_MULTILABEL_CROSS_ENTROPY` |
| `proximal_cross_entropy` | `ProximalCrossEntropy(alpha)` | `ProximalCrossEntropyLoss`, `proximal_cross_entropy` | `Loss::ProximalCrossEntropy { alpha }` | `SDCA_LOSS_PROXIMAL_CROSS_ENTROPY` |
| `distillation` | `Distillation(temperature)` | `DistillationLoss`, `distillation_loss` | `Loss::Distillation { temperature }` | `SDCA_LOSS_DISTILLATION` |
| `proximal_distillation` | `ProximalDistillation(temperature, alpha)` | `ProximalDistillationLoss`, `proximal_distillation_loss` | `Loss::ProximalDistillation { temperature, alpha }` | `SDCA_LOSS_PROXIMAL_DISTILLATION` |
| `linear_cross_entropy` | — | `LinearCrossEntropyLoss`, `linear_cross_entropy` (PyTorch) | — | — |

C++ objectives use the canonical names with an `l2_` prefix (`l2_topk_hinge_beta`;
`*_objective(c, ...)` factories take `gamma`, default `0`), and MATLAB takes the canonical
names as `opts.objective` / `opts.loss`.

- **Targets:** The single-label losses take class labels (0-based; 1-based in MATLAB), the
  multilabel losses label sets or a boolean mask, and the distillation losses teacher
  logits. The training objective `cross_entropy` also takes target probabilities, rows of an
  `(n, num_classes)` matrix such as a teacher's softmax (`libsdca.Probabilities` in NumPy,
  `Targets::Probabilities` in Rust, `SDCA_TARGET_PROBABILITIES` in C). The proximal and
  distillation losses are not training objectives.

## Options

The solver options below have the same names and defaults in every interface (Rust: `Solver`
setters such as `with_tol`; C: `struct sdca_fit_options`; C++: `context.criteria`; the
scikit-learn estimator names `max_epoch` `max_iter` and `seed` `random_state`):

| Option | Default | Meaning |
| --- | --- | --- |
| `C` (`c` in Rust / C) | `1.0` | Loss weight in $0.5\Vert W \Vert_2^2 + C \sum_i \ell_i$ |
| `tol` | `1e-3` | Relative duality-gap stopping threshold |
| `max_epoch` | `1000` | Maximum passes over the training set |
| `eval_epoch` | `10` | Epochs between duality-gap evaluations; `0` evaluates only at the end |
| `eval_on_start` | `false` | Evaluate objective and gap before epoch 1 |
| `recompute_primal` | `true` | Rebuild $W = W_0 + X^\top A$ at each evaluation |
| `shrinking` | `"auto"` | Hinge shrinking and screening ([below](#shrinking)); the entropy objectives take only `"auto"` |
| `seed` | `0` | The order of the examples in the epochs |
| `num_classes` | largest label plus one | The number of classes |
| `center` | none | Regularize $0.5\Vert W - W_0 \Vert_2^2$ instead of $0.5\Vert W \Vert_2^2$ (features only) |
| `initial_dual` | none | Initial dual variables $A$ for warm starts; the library checks them and builds $W$ |

Kernel input is explicit: `libsdca.Kernel(K)` in NumPy, `kernel="precomputed"` in
scikit-learn, `Input::Kernel` in Rust, `SDCA_INPUT_KERNEL` in C, `make_input_kernel` in C++.
Every call also takes the execution settings `precision` ([below](#precision)) and
`num_threads` ([threads](#threads)); `projection_precision` belongs to `multilabel_hinge`
([below](#projection-precision)).

- **Evaluations (`eval_epoch`, `recompute_primal`):** Besides checking every
  `eval_epoch` epochs, a fit with `tol > 0` schedules up to two interpolated
  evaluations when the gap trend projects crossing `tol` early. With
  `recompute_primal=True` (default), feature fits rebuild $W = W_0 + X^\top A$
  before evaluating the primal objective so incremental update drift does not
  enter the gap (not for `float32` data with `precision="float64"`, whose `double` $W$ is
  evaluated as stored). A fit stopped by `max_epoch` reports `solved` when its last
  evaluation meets `tol`.
- **Kernel validation:** Every interface verifies a nonnegative
  diagonal, symmetry within $10^{-5}\sqrt{K_{ii}K_{jj}}$, and
  $\lvert K_{ij} \rvert \le (1 + 10^{-3})\sqrt{K_{ii}K_{jj}}$.
- **Feature scaling:** Because each coordinate step scales with $1/(C\Vert x_i \Vert_2^2)$,
  standardize or normalize features when example norms span orders of magnitude.

### Shrinking

`shrinking` accelerates hinge fits (`multiclass_hinge`, `topk_hinge_alpha`,
`topk_hinge_beta`, `multilabel_hinge`) by skipping dual variables that sit at zero
bounds (entropy objectives have strictly positive probabilities and ignore the option):

- `"heuristic"` (`Shrinking::Heuristic`, `SDCA_SHRINKING_HEURISTIC`): LIBLINEAR-style
  shrinking [7]. Epochs between full passes skip examples whose dual block did not change
  and restrict single-label updates to the target and nonzero-dual classes; full passes
  and evaluations run after `eval_epoch` full-pass equivalents of work (at most 50 epochs).
- `"safe"` (`Shrinking::Safe`, `SDCA_SHRINKING_SAFE`): Gap-safe screening [8] for the
  single-label hinges on feature inputs (no effect for `multilabel_hinge` or kernels). Uses
  strong convexity $\Vert W - W^\star \Vert_2 \le \sqrt{2\text{ gap}}$ to permanently prune
  dual variables proven to be zero at the optimum.
- `"both"` / `"auto"` (default): Combines `"heuristic"` and `"safe"`.
- `"off"`: Updates every class of every example on every epoch.

All modes evaluate the duality gap across all examples and classes, so `solved` carries the
same guarantee.

### Projection precision

`projection_precision` controls how the `multilabel_hinge` (two-simplex) projection is solved:

- `"float64"` (default; `ProjectionPrecision::Float64`, `SDCA_PROJECTION_FLOAT64`): Solves
  in `double` and verifies that every entry lies clear of its threshold error bound, falling
  back to the extended-precision solver only when a near-tie or extreme scale fails the check.
- `"extended"` (`ProjectionPrecision::Extended`, `SDCA_PROJECTION_EXTENDED`): Always runs
  the `long double` solver (slightly slower; reproducible on extreme spreads).

## Precision

Storage and working precision are separate. `precision=None` (default) follows the storage
dtype, with 16-bit storage computed in `float32`; `"float32"` or `"float64"` selects the
working precision explicitly:

| Storage dtype | `None` (default), `"float32"` | `"float64"` |
| --- | --- | --- |
| `float16`, `bfloat16` | `float32` work (NumPy casts 16-bit arrays to `float32`; C takes `float32` or `float64`) | `float64` work (JAX: see [backends](../python/JAX.md#backends)) |
| `float32` | `float32` work | `float64` work |
| `float64` | `float64` work (`"float32"`: NumPy casts to `float32` and returns `float32` arrays; PyTorch and JAX raise, since gradients keep the logits' dtype) | `float64` work |

MATLAB follows the class of `X` (`'single'` or `'double'`). In C the storage type is the
`dtype` argument and the work is `struct sdca_execution`'s `precision`
(`SDCA_PRECISION_DEFAULT`, `SDCA_PRECISION_FLOAT32` or `SDCA_PRECISION_FLOAT64`).

- **Accuracy contract:** `float32` work bounds the accuracy of the results; individual
  operations may run in `float64`. Row statistics of the entropy losses are accumulated in
  `float64`, and kernels promote rows or parameters that `float32` cannot represent. Some paths
  (single-label cross-entropy and smoothed hinges in the native library, proximal distillation
  on CPU) keep `float64` arithmetic throughout.
- **Outputs:** Weights, duals and gradients keep the storage dtype. Loss values are `double`
  in the native APIs and `promote(logits.dtype, float32)` in PyTorch and JAX.
- **`float32` fits:** Keeping $W$ and the duals in `float32` can stall near the optimum
  (updates below half an ulp are lost). `precision="float64"` (`<float, double>` in C++) keeps
  `double` iterates for tighter convergence at `8(d + n)m` extra bytes. Both modes check the
  gap of the returned arrays.
- **Compiler flags:** Build with IEEE semantics (`-fno-fast-math -ffp-contract=off`);
  `-ffast-math` or `-Ofast` breaks the nonfinite-input checks and compensated sums.

### Parameter domains and numerical limits

All loss and training entry points require `classes >= 2`, integer `1 <= k < classes`,
finite scores/features, and valid targets (0-based class indices, or 1-based in MATLAB;
boolean masks require at least one positive label, plus one negative label for
`multilabel_hinge` and `proximal_cross_entropy`; label sets in `fit` and `coordinate_update`
require a negative label for every objective).

| Parameter | Valid domain | Notes |
| --- | --- | --- |
| `C`, `alpha`, `temperature` | finite, strictly positive | `C`, `norm2` and `C * norm2` must be normal in the storage dtype, and `C` in working precision; `alpha` or `temperature` outside normal `float32` promotes to `float64` |
| `gamma` | finite, `>= 0` | `0` selects the nonsmooth hinge; positive values follow the rules of `alpha` |
| `tol` | finite, `>= 0` | `0` disables the `solved` stop: the fit ends with `max_epoch`, `no_progress` or `failed` (or a time limit in MATLAB) |
| `ignore_index` (PyTorch) | 64-bit integer | Matching targets are excluded from the loss and gradient |
| `weight` (PyTorch CE) | finite, `>= 0`, shape `(C,)` | Matches logits device and dtype (`float32` accepted for 16-bit logits under autocast) |

See [known limits](#known-limits) for accuracy bounds.

## Results

Training returns a status (`solved`, `no_progress`, `max_epoch` or `failed`) and an
evaluation `history` with the same fields in every interface: `epoch`, `primal`, `dual`,
`primal_loss`, `dual_loss`, `regularizer`, `relative_gap`, `accuracy` (the rank loss for the
multilabel losses), `cpu_time` and `wall_time`.

Only `status == "solved"` guarantees convergence:
$$P(W) - D(A) + e \le \text{tol} \cdot \max(|P(W)|, |D(A)|),$$
where $P(W)$ is evaluated in `double` at the returned weights, $D(A)$ at the feasible
projection of the returned duals, and $e$ is a first-order estimate of evaluation rounding.
If a numerical step fails or the evaluated gap falls below $-e$, training stops with
`failed` at the last consistent iterate. With a regularizer center $W_0$, $P(W)$ and
$D(A)$ evaluate the full centered objective directly from $W - W_0$ without cancelling
$\frac{1}{2}\Vert W_0 \Vert_2^2$.

Array layouts for $n$ examples, $d$ features and $m$ classes: Python returns `weights (d, m)`
column-major and `dual (n, m)` row-major; Rust returns both row-major (`Model::weights`,
`Model::dual`) and the history as `Evaluation` structs; MATLAB takes
features `(d, n)` and returns `W (d, m)` and `A (m, n)`.

## Distillation losses

`distillation_loss` and `proximal_distillation_loss` take student and teacher **logits** of the
same shape and dtype; the teacher receives no gradient. With $p = \mathrm{softmax}(t/T)$ and
$q = \mathrm{softmax}(z/T)$, the student gradient is $(q - p)/T$ (or $(\pi - p)/T$ at the
Moreau-envelope optimum $\pi = \mathrm{softmax}(u^\star)$ for `proximal_distillation_loss`),
with no $T^2$ multiplier. Because each entry is formed from a centered log-ratio via `expm1`,
identical student and teacher logits produce an **exact zero gradient** in every dtype.

```python
import torch
from libsdca import DistillationLoss, ProximalDistillationLoss

student = torch.randn(8, 10, requires_grad=True)
teacher = torch.randn(8, 10)
loss = DistillationLoss(temperature=4.0)(student, teacher)
prox = ProximalDistillationLoss(temperature=4.0, alpha=0.5)(student, teacher)
(loss + prox).backward()
```

In PyTorch, both run on fused Triton kernels on CUDA and the C++ extension on CPU
(first-order derivatives only). In JAX (`libsdca.jax`), both run on the same Triton
kernels on CUDA and use the shared CPU kernels where available, with array
fallbacks. Direct `jvp`/`jacfwd` of these custom-VJP losses raise. Ordinary
distillation supports reverse-over-reverse and `jax.hessian`; proximal distillation
supports `jax.hessian` (forward-over-reverse), but reverse-over-reverse is unsupported
because the array reference solve's dynamic loop cannot be transposed.

## Threads

Loss evaluations and coordinate steps process rows independently across up to `num_threads`
worker threads with **bitwise-identical results** for every thread count:

- **Default selection:** When `num_threads` is omitted (`None` in Python, `Threads::FromEnv`
  in Rust, `0` in C), the library reads `SDCA_NUM_THREADS` once and
  otherwise uses automatic threading, capped by the active
  OpenMP team (`torch.set_num_threads`) or `OMP_NUM_THREADS` and divided across concurrent
  calls. Passing `1` runs on the calling thread.
- **Work threshold:** A call times its first rows on the calling thread and spawns workers only
  when remaining work is estimated at $\ge 1\text{ ms}$ ($\ge 0.2\text{ ms}$ on an OpenMP team).
- **Training (`fit`):** Coordinate-ascent epochs are sequential; only duality-gap evaluations,
  gap-safe screening passes and kernel validation run in parallel.
- **Floating-point environment:** Every C ABI call executes with IEEE subnormals enabled and
  round-to-nearest (even if the caller set flush-to-zero, as XLA:CPU does) and restores the
  caller's environment on return.

## Environment variables

The runtime-compiled extension switches (`SDCA_CPU_EXTENSION`, `SDCA_CUDA_LAUNCHER`,
`SDCA_JAX_FFI`, `SDCA_JAX_CUDA`) take `auto` (default: compile and cache on first use, or
fall back with a one-time warning), `on` (`1`: raise `RuntimeError` if unavailable) or `off`
(`0`). A compiler error leaves `*.build-failed.txt` in the cache, which suppresses
retries for 24 hours (transient failures are not recorded); delete it to retry. The CUDA
switches fall back without a warning when Triton is newer than the verified range.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SDCA_NUM_THREADS` | unset (auto) | Default thread limit (`< 4096`; `0` = auto, `1` = serial) |
| `SDCA_CPU_EXTENSION` | `auto` | `auto \| on \| off`: JIT-compiled PyTorch CPU C++ extension (`libsdca.torch.explain_backend` reports status) |
| `SDCA_CUDA_LAUNCHER` | `auto` | `auto \| on \| off`: C++ `cuLaunchKernel` launcher for CUDA kernels (Triton 3.3–3.9) |
| `SDCA_JAX_FFI` | `auto` | `auto \| on \| off`: XLA FFI custom-call handler for native JAX CPU losses |
| `SDCA_JAX_CUDA` | `auto` | `auto \| on \| off`: fused Triton kernels of JAX losses on CUDA GPUs (Triton 3.3–3.9; `off` keeps `jax.numpy`) |
| `SDCA_JAX_CPU_FUSION` | `auto` | `auto \| fenced \| unfenced`: fusion of the JAX entropy and distillation CPU passes, read when a loss is traced; `auto` times both once per machine ([JAX.md](../python/JAX.md#cpu-pass-fusion)) |
| `SDCA_CACHE_DIR` | `$XDG_CACHE_HOME/libsdca` (`~/.cache/libsdca`) | Cache directory for compiled JAX FFI handlers and `SDCA_JAX_CPU_FUSION=auto` decisions |
| `TORCH_EXTENSIONS_DIR` | PyTorch cache | Cache directory for compiled PyTorch CPU and CUDA extensions |
| `OMP_NUM_THREADS` | unset | Caps automatic threads when no OpenMP runtime is loaded |
| `SDCA_LIBRARY_DIR` | unset | Override directory from which Python loads `libsdca_c.so` |
| `SDCA_BLAS_BACKEND` | `AUTO` | Build-time BLAS selection (`AUTO`, `EXTERNAL`, `BUILTIN`) for `pip install`, `python/build.py` and `rust/build.rs` (CMake: `-DSDCA_BLAS_BACKEND`) |
| `SDCA_BLAS` | unset | Explicit external BLAS linker flags (e.g. `-lopenblas` or `-L/opt/blas/lib -lblas`) |
| `SDCA_BUILD_NATIVE` | `ON` | Set to `OFF` during `pip install` to build a pure-Python wheel for tensor losses only |
| `CXX`, `AR`, `CUDA_HOME` | system default | Compiler, archiver, and CUDA toolkit path for native and extension builds |

CMake options (`cmake -D<option>=<value>`): `SDCA_BLAS_BACKEND` (`AUTO|EXTERNAL|BUILTIN`),
`SDCA_BUILD_C_API` (`OFF`), `SDCA_BUILD_MATLAB` (`OFF`), `SDCA_WARNINGS_AS_ERRORS` (`OFF`),
`BUILD_TESTING` (`OFF`), `USE_CHRONO` (`ON`), and MKL flags `USE_ILP64` (`ON`),
`USE_SEQUENTIAL` (`ON`), `USE_THREADING_SUFFIX` (`OFF`).

## Errors

| Interface | Invalid input or option | Numerical failure |
| --- | --- | --- |
| Python (NumPy, scikit-learn, PyTorch) | `ValueError` (or `TypeError`) | `RuntimeError` |
| JAX | `ValueError` eagerly; `JaxRuntimeError` (`native`) or `nan` (`jax`) under `jit` / `vmap` | `JaxRuntimeError` (`native`) or `nan` (`jax`) |
| Rust | `Error::InvalidArgument` | `Error::Native` |
| C++ | `std::invalid_argument` | `std::runtime_error` |
| C | `SDCA_INVALID_ARGUMENT` | `SDCA_FAILURE` (message in `sdca_last_error()`) |
| MATLAB | `LIBSDCA:*` error | `LIBSDCA:*` error |

## Known limits

- **Representation:** Losses and gradients are kept down to the output dtype's subnormal
  range: cross-entropy margins up to about `104` in `float32` and `745` in `float64`.
  XLA:CPU (JAX on CPU) flushes subnormals, so smaller values become `0` there.
- **`float32` work:** 16-bit inputs give values within `100` `float32` eps of
  $\max(|\text{loss}|, 1)$ and gradients within `2,000` `float32` eps of the row's largest
  entry. Cross-entropy with `float32` exponentials is within `8` `float32` eps of high-precision
  references. Pass `precision="float64"` for tighter bounds.
- **Distillation:** Relative error within `64` output eps for $T \in [0.25, 4]$. `float64` tails
  `740`–`750` below the leading logit are rounded to subnormals before summing.
- **Cancellation:** `float64` gradient entries that are differences of nearly equal
  probabilities (distillation where student and teacher nearly agree, and
  `multilabel_cross_entropy` $p - 1/|Y|$) stay within `5e-17` absolute (a fifth of eps) but
  can miss the componentwise relative bound (relative error up to `4e-11` on 4,097-class rows).
- **Extreme spreads:** Score differences near `±finfo.max` can overflow: on finite `float64`
  inputs, hinges, proximal losses and distillation ($T \le 1$) may return `nan`, `inf` or
  a wrong value, or raise `ValueError` for the derived scores. Top-*k* entropy loses
  relative accuracy at huge competitor spreads (`float64` relative error `9e-6` at
  $10^{12}$).
- **Solver:** Tolerances below about `1e-12` in `float64` cannot be guaranteed: the fit ends
  with `max_epoch` or `no_progress`. Large regularizer centers $W_0$ limit representable
  updates. Where `long double` is 64-bit (macOS `arm64`, MSVC), extended-precision retries
  equal `double`.
- **JAX without x64:** `backend="jax"` with `precision="float32"` has no `float64` fallback:
  `proximal_cross_entropy` rows whose score spread `float32` cannot hold return `nan`.

## References

1. K. Crammer and Y. Singer. On the algorithmic implementation of multiclass
   kernel-based vector machines. *JMLR*, 2001.
2. K. Crammer and Y. Singer. A family of additive online algorithms for
   category ranking. *JMLR*, 2003.
3. M. Lapin, M. Hein and B. Schiele. Top-k multiclass SVM. *NeurIPS*, 2015.
4. M. Lapin, M. Hein and B. Schiele. Loss functions for top-k error: analysis
   and insights. *CVPR*, 2016.
5. S. Shalev-Shwartz and T. Zhang. Stochastic dual coordinate ascent methods for
   regularized loss minimization. *JMLR*, 2013.
6. G. Hinton, O. Vinyals and J. Dean. Distilling the knowledge in a neural
   network. arXiv:1503.02531, 2015.
7. R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang and C.-J. Lin. LIBLINEAR:
   A library for large linear classification. *JMLR*, 2008.
8. E. Ndiaye, O. Fercoq, A. Gramfort and J. Salmon. Gap safe screening rules
   for sparsity enforcing penalties. *JMLR*, 2017.
