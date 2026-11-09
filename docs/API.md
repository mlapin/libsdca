# API reference

`libsdca` shares one set of loss names, option names, status codes and defaults
across C++, the C ABI (`include/sdca/c_api.h`), Python, PyTorch, JAX, Rust and
MATLAB. See the interface guides for language-specific examples:
[Python](../python/README.md), [PyTorch](../python/TORCH.md),
[JAX](../python/JAX.md), [Rust](../rust/README.md), [C++](CPP.md), [C](C.md)
and [MATLAB](MATLAB.md).

## Loss definitions

For scores $z \in \mathbb{R}^m$, target class $y$, and non-target margins
$h_j = 1 + z_j - z_y$ ($j \neq y$) sorted as $h_{[1]} \ge h_{[2]} \ge \dots$
(or positive label set $Y \subset \{1, \dots, m\}$ for multilabel losses):

| Name | Formula | Parameters | Use case |
| --- | --- | --- | --- |
| `cross_entropy` | $\log \sum_j \exp(z_j) - z_y$ | — | Standard multiclass classification |
| `topk_cross_entropy` | $\max_{p \in \Delta_k} \langle a, p \rangle - (1 - s)\log(1 - s) - \sum_j p_j \log p_j$, where $a_j = z_j - z_y$, $s = \sum_j p_j$, $\Delta_k = \{p \ge 0 : s \le 1,\, p_j \le s/k\}$ over $j \neq y$ | `k` | Top-$k$ entropy classification [4] (`cross_entropy` at $k=1$) |
| `multiclass_hinge` | $\max(0, h_{[1]})$ | `gamma` | Crammer–Singer multiclass SVM [1] |
| `topk_hinge_alpha` | $\max\!\big(0, \frac{1}{k}\sum_{i \le k} h_{[i]}\big)$ | `k`, `gamma` | Top-$k$ SVM on average of $k$ worst margins [3] |
| `topk_hinge_beta` (`topk_hinge`) | $\frac{1}{k}\sum_{i \le k} \max(0, h_{[i]})$ | `k`, `gamma` | Top-$k$ SVM on each of $k$ worst margins [4] |
| `multilabel_hinge` | $\max\!\big(0, 1 + \max_{j \notin Y} z_j - \min_{y \in Y} z_y\big)$ | `gamma` | Ranked multilabel SVM [2] |
| `multilabel_cross_entropy` | $\log \sum_j \exp(z_j) - \frac{1}{|Y|}\sum_{y \in Y} z_y$ | — | Softmax cross-entropy to uniform distribution over $Y$ |
| `proximal_cross_entropy` | $\min_u \big\{\operatorname{CE}(u, y) + \frac{1}{2\alpha}\|u - z\|_2^2\big\}$ | `alpha` | Moreau envelope of cross-entropy [5] (loss evaluation only) |
| `distillation` | $-\sum_j p_j \log q_j$, with $p = \operatorname{softmax}(t/T)$, $q = \operatorname{softmax}(z/T)$ | `temperature` | Knowledge distillation to teacher logits $t$ [6] |
| `proximal_distillation` | $\min_u \big\{H(p, \operatorname{softmax}(u)) + \frac{1}{2\alpha}\|u - z/T\|_2^2\big\}$ | `temperature`, `alpha` | Damped distillation step (Moreau envelope in $z/T$) |
| `linear_cross_entropy` | $\operatorname{CE}(h W^\top + b, y)$ evaluated in row chunks without materializing logits | `chunk_bytes`, `recompute` | Language-model / large-vocabulary classification head (PyTorch) |

- **Hinge smoothing (`gamma`):** `gamma = 0` (default) is the nonsmooth hinge;
  `gamma > 0` is its Moreau envelope $\max_{p \in S} \langle h, p \rangle - \frac{\gamma}{2}\|p\|_2^2$,
  which is within $\gamma / 2$ of the nonsmooth loss and has a $(1/\gamma)$-Lipschitz gradient [4].
- **Top-$k$ exclusion:** Top-$k$ hinges exclude the target class $y$ when selecting the $k$
  largest competitor margins $h_{[1]}, \dots, h_{[k]}$ [4, Prop. 11]. At $k = 1$,
  `topk_hinge_alpha` and `topk_hinge_beta` coincide with `multiclass_hinge`.

## Objectives and losses

### Core entry points

| Operation | Python (NumPy / sklearn) | PyTorch | JAX | Rust | C++ | C | MATLAB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Train classifier | `fit`, `SDCAClassifier` | — | — | `fit` | `make_solver(ctx).solve()` | `sdca_fit` | `matsdca_fit` |
| Loss & gradient | `loss_and_grad` | `SDCALoss`, `sdca_loss` | `libsdca.jax.sdca_loss` | `loss_and_grad` | `loss_and_gradient` | `sdca_loss_gradient` | `matsdca_loss` |
| Coordinate step | `coordinate_update` | — | — | `coordinate_update` | `objective.update_dual_variables` | `sdca_update_objective` | — |

The public C++ headers are `sdca/solver.h`, `sdca/prox.h` and `sdca/loss.h`
(`sdca::detail` and underscore-prefixed Python names are internal).

### Names across interfaces

| Canonical name | PyTorch module / function | JAX function | Rust | C++ objective / function | C ABI |
| --- | --- | --- | --- | --- | --- |
| `cross_entropy` | `CrossEntropyLoss`, `cross_entropy` | `cross_entropy` | `CrossEntropy` | `l2_entropy(c)` | `SDCA_CROSS_ENTROPY` |
| `topk_cross_entropy` | `TopKCrossEntropyLoss`, `topk_cross_entropy` | `topk_cross_entropy` | `TopKCrossEntropy` | `l2_entropy_topk(c, k)` | `SDCA_TOPK_CROSS_ENTROPY` |
| `multiclass_hinge` | `MulticlassHingeLoss`, `multiclass_hinge_loss` | `multiclass_hinge_loss` | `MulticlassHinge` | `l2_hinge_topk[_smooth]` | `SDCA_MULTICLASS_HINGE` |
| `topk_hinge_alpha` | `TopKHingeAlphaLoss`, `topk_hinge_alpha_loss` | `topk_hinge_alpha_loss` | `TopKHingeAlpha` | `l2_hinge_topk[_smooth]` | `SDCA_TOPK_HINGE_ALPHA` |
| `topk_hinge_beta` | `TopKHingeBetaLoss`, `topk_hinge_beta_loss` (`TopKHingeLoss`) | `topk_hinge_beta_loss` (`topk_hinge_loss`) | `TopKHingeBeta` | `l2_topk_hinge[_smooth]` | `SDCA_TOPK_HINGE_BETA` |
| `multilabel_hinge` | `MultilabelHingeLoss`, `multilabel_hinge_loss` | `multilabel_hinge_loss` | `MultilabelHinge` | `l2_multilabel_hinge[_smooth]` | `SDCA_MULTILABEL_HINGE` |
| `multilabel_cross_entropy` | `MultilabelCrossEntropyLoss`, `multilabel_cross_entropy` | `multilabel_cross_entropy` | `MultilabelCrossEntropy` | `l2_multilabel_entropy(c)` | `SDCA_MULTILABEL_CROSS_ENTROPY` |
| `proximal_cross_entropy` | `ProximalCrossEntropyLoss`, `proximal_cross_entropy` | `proximal_cross_entropy` | `Loss::ProximalCrossEntropy` | `loss_and_gradient` | `SDCA_PROXIMAL_CROSS_ENTROPY` |
| `distillation` | `DistillationLoss`, `distillation_loss` | `distillation_loss` | `distillation_loss_and_grad`, `fit_distillation` | `l2_distillation`, `distillation_loss[_and_gradient]` | `sdca_distillation_*`, `sdca_fit_distillation` |
| `proximal_distillation` | `ProximalDistillationLoss`, `proximal_distillation_loss` | `proximal_distillation_loss` | `proximal_distillation_loss_and_grad` | `proximal_distillation_loss[_and_gradient]` | `sdca_proximal_distillation_rows` |
| `linear_cross_entropy` | `LinearCrossEntropyLoss`, `linear_cross_entropy` | — | — | — | — |

In Python (`distillation_loss_and_grad`, `proximal_distillation_loss_and_grad`),
passing `objective="distillation"` to `fit` or `coordinate_update` trains on a
target probability matrix `(n, num_classes)`. Passing label sets or a boolean
mask with `cross_entropy` selects `multilabel_cross_entropy`.

## Options

| Option | Python / Rust | MATLAB | C++ | Default | Meaning |
| --- | --- | --- | --- | --- | --- |
| Loss weight | `C` / `c` | `C` | objective constructor | `1.0` | Regularization trade-off in $0.5\|W\|_2^2 + C \sum_i \ell_i$ |
| Top-$k$ | `k` | `k` | objective constructor | `1` | Integer in `[1, classes - 1]` for top-$k$ losses |
| Hinge smoothing | `gamma` | `gamma` | `_smooth` objective | `0.0` | `0`: nonsmooth hinge; `> 0`: Moreau smoothing step |
| Gap tolerance | `tol` | `tol` | `criteria.epsilon` | `1e-3` | Relative duality-gap stopping threshold |
| Epoch limit | `max_epoch` | `max_epoch` | `criteria.max_epoch` | `1000` | Maximum passes over the training set |
| Evaluation period | `eval_epoch` | `eval_epoch` | `criteria.eval_epoch` | `10` | Epochs between duality-gap evaluations |
| Evaluate at start | `eval_on_start` | `eval_on_start` | `criteria.eval_on_start` | `false` | Evaluate objective and gap before epoch 1 |
| Rebuild primal | `recompute_primal` | `recompute_primal` | `criteria.recompute_primal` | `true` | Recompute $W = W_0 + X^\top A$ from duals at each evaluation |
| Shrinking | `shrinking` | `shrinking` | `criteria.shrinking` | `"auto"` | Active-set shrinking and screening for hinge fits ([below](#shrinking)) |
| Projection mode | `projection_precision` | `projection_precision` | `criteria.projection` | `"float64"` | Multilabel hinge projection solver ([below](#projection-precision)) |
| Kernel input | `kernel` | `kernel` | `make_input_kernel` | `false` | Input $X$ is a symmetric positive semidefinite $(n, n)$ Gram matrix |
| Regularizer center | `center` | `center` | `context.primal_initial` | `None` | Regularize $0.5\|W - W_0\|_2^2$ instead of $0.5\|W\|_2^2$ |
| Warm start | `initial_dual` | `initial_dual` | dual buffer | `None` | Initial dual variables $A$ |
| Threads | `num_threads` | `num_threads` | `criteria.num_threads` | `SDCA_NUM_THREADS` or auto | Worker threads for loss rows, coordinate steps, and gap evaluations |
| Precision | `precision` | `precision` | `<Data, Result>` | `"float64"` | Working arithmetic and solver iterate precision ([below](#precision)) |

- **Evaluations (`eval_epoch`, `recompute_primal`):** Besides checking every
  `eval_epoch` epochs, a fit with `tol > 0` schedules up to two interpolated
  evaluations when the gap trend projects crossing `tol` early. With
  `recompute_primal=True` (default), feature fits rebuild $W = W_0 + X^\top A$
  before evaluating the primal objective so incremental update drift does not
  pollute the certificate.
- **Kernel validation (`kernel=True`):** Every interface verifies a nonnegative
  diagonal, symmetry within $10^{-5}\sqrt{K_{ii}K_{jj}}$, and
  $|K_{ij}| \le (1 + 10^{-3})\sqrt{K_{ii}K_{jj}}$.
- **Feature scaling:** Because each coordinate step scales with $1/(C\|x_i\|_2^2)$,
  standardize or normalize features when example norms span orders of magnitude.

### Shrinking

`shrinking` accelerates hinge fits (`multiclass_hinge`, `topk_hinge_alpha`,
`topk_hinge_beta`, `multilabel_hinge`) by skipping dual variables that sit at zero
bounds (entropy objectives have strictly positive probabilities and ignore the option):

- `"heuristic"` (`Shrinking::Heuristic`, `SDCA_SHRINKING_HEURISTIC`): LIBLINEAR-style
  shrinking. Epochs between full passes skip examples whose dual block did not change
  and restrict single-label updates to the target and nonzero-dual classes; full passes
  and evaluations run after `eval_epoch` full-pass equivalents of work (at most 50 epochs).
- `"safe"` (`Shrinking::Safe`, `SDCA_SHRINKING_SAFE`): Gap-safe screening on feature
  inputs. Uses strong convexity $\|W - W^\star\|_2 \le \sqrt{2\,\text{gap}}$ to permanently
  prune dual variables proven to be zero at the optimum.
- `"both"` / `"auto"` (default): Combines `"heuristic"` and `"safe"`.
- `"off"`: Updates every class of every example on every epoch.

All modes evaluate the duality gap over all examples and classes, so `solved`
carries the same certificate regardless of shrinking ([performance](performance.md#solver)).

### Projection precision

`projection_precision` controls how the `multilabel_hinge` (two-simplex) projection is solved:

- `"float64"` (default; `ProjectionPrecision::Float64`, `SDCA_PROJECTION_FLOAT64`): Solves
  in `double` and verifies that every entry lies clear of its threshold error bound, falling
  back to the extended-precision solver only when a near-tie or extreme scale fails the check.
- `"extended"` (`ProjectionPrecision::Extended`, `SDCA_PROJECTION_EXTENDED`): Always runs
  the hardened `long double` solver (`0–12%` slower at 1,000 classes, bitwise reproducible
  on extreme spreads; see [numerics.md](numerics.md#projection-precision)).

## Precision

Storage and working arithmetic are decoupled across all interfaces:

| Input / storage dtype | `precision` | Native (NumPy / Rust / C) | PyTorch | JAX |
| --- | --- | --- | --- | --- |
| `float16`, `bfloat16` | `None` (default) or `"float32"` | unsupported (cast to `float32`) | `float32` work (`float64` row stats in CE) | `float32` work |
| `float16`, `bfloat16` | `"float64"` | unsupported | `float64` work | `float64` work (requires `jax_enable_x64`) |
| `float32` | `None` (default) or `"float64"` | `float64` work & double iterate | `float32` work in CE (`float64` in hinges, or with `"float64"`) | `float32` work in CE (`float64` with `"float64"` and x64) |
| `float32` | `"float32"` | `float32` work & float iterate (1.6–1.9× faster CE fit) | `float32` work | `float32` work |
| `float64` | `None` or `"float64"` | `float64` work & double iterate (`"float32"` rejected) | `float64` work (`"float32"` rejected) | `float64` work (`"float32"` rejected) |

- **Outputs:** Returned weights, duals and gradients always keep the input storage dtype.
  Loss values are `double` in the native APIs and `promote(logits.dtype, float32)` in PyTorch
  and JAX (`float32` for 16- and 32-bit logits, `float64` for `float64`, matching `torch.autocast`).
- **Why `float32` training defaults to `precision="float64"`:** Keeping `W` and the duals in
  `float32` during training loses updates smaller than half an ulp near the optimum, stalling
  `float32` hinge fits around a $4.7\times 10^{-6}$ relative gap on `digits`. The default
  `"float64"` (`<float, double>` in C++) keeps `W` and the duals in `double` (`8(d + n)m` extra
  bytes), rounds them once into the caller's `float32` buffers at each evaluation, and certifies
  those rounded arrays (reaching $3.5\times 10^{-7}$ on `digits`; see
  [numerics.md](numerics.md#what-each-precision-guarantees)). Pass `precision="float32"` for
  pure `float32` speed and memory.
- **Compiler flags:** Build `libsdca` with IEEE semantics (`-fno-fast-math -ffp-contract=off`);
  `-ffast-math` / `-Ofast` breaks nonfinite-input detection and compensated reductions.

### Parameter domains and numerical limits

All loss and training entry points require `classes >= 2`, integer `1 <= k < classes`,
finite scores/features, and valid targets (0-based class indices, except 1-based in MATLAB;
boolean masks need at least one positive label, plus one negative label for `multilabel_hinge`
and `proximal_cross_entropy`).

| Parameter | Valid domain | Notes |
| --- | --- | --- |
| `C`, `alpha`, `temperature` | finite, strictly positive | `C` and `C * norm2` must be normal in working precision; `alpha` or `temperature` outside normal `float32` promotes to `float64` |
| `gamma` | finite, `>= 0` | `0` selects the nonsmooth hinge; positive values follow the rules of `alpha` |
| `tol` | finite, `>= 0` | `0` runs until `max_epoch` (or time limits in MATLAB); tolerances below ~`1e-12` in `float64` cannot be certified |
| `ignore_index` (PyTorch) | 64-bit integer | Matching targets are excluded from the loss and gradient |
| `weight` (PyTorch CE) | finite, `>= 0`, shape `(C,)` | Same device and dtype as logits (`float32` weights accepted for 16-bit logits under autocast) |

Tested accuracy bounds and representation limits are documented in
[numerics.md](numerics.md#known-limits).

## Results

Training returns a status (`none`, `solving`, `solved`, `no_progress`, `max_epoch`,
`max_cpu_time`, `max_wall_time`, `failed`) and an evaluation `history` (`epoch`,
`primal`, `dual`, `primal_loss`, `dual_loss`, `regularizer`, `relative_gap`,
`accuracy` / `rank_loss`, `cpu_time`, `wall_time`).

Only `status == "solved"` certifies convergence:
$$P(W) - D(A) + e \le \text{tol} \cdot \max(|P(W)|, |D(A)|),$$
where $P(W)$ and $D(A)$ are evaluated in `double` at the returned arrays and $e$ is a
first-order estimate of evaluation rounding ([numerics.md](numerics.md#the-solved-certificate)).
If a numerical step fails or the evaluated gap falls below $-e$, training stops with
`failed` at the last consistent iterate. With a regularizer center $W_0$, $P(W)$ and
$D(A)$ evaluate the full centered objective directly from $W - W_0$ without cancelling
$\frac{1}{2}\|W_0\|_2^2$.

Array layouts for $n$ examples, $d$ features and $m$ classes: Python returns `weights (d, m)`
column-major and `dual (n, m)` row-major; Rust returns both row-major; MATLAB takes
features `(d, n)` and returns `W (d, m)` and `A (m, n)`.

## Distillation losses

`distillation_loss` and `proximal_distillation_loss` take student and teacher **logits** of the
same shape and dtype; the teacher receives no gradient. With $p = \operatorname{softmax}(t/T)$ and
$q = \operatorname{softmax}(z/T)$, the student gradient is $(q - p)/T$ (or $(\pi - p)/T$ at the
Moreau-envelope optimum $\pi = \operatorname{softmax}(u^\star)$ for `proximal_distillation_loss`),
with no $T^2$ multiplier. Because each entry is formed from a centered log-ratio via `expm1`,
identical student and teacher logits produce an **exact zero gradient** in every dtype
([softmax cancellation](softmax-collapse.md#softmax-cancellation)).

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
(first-order derivatives only). In JAX (`libsdca.jax`), both run in `jax.numpy` on any
device (`jvp`/`jacfwd` raise, while reverse-over-reverse and `jax.hessian` are supported).

## Threads

Loss evaluations and coordinate steps process rows independently across up to `num_threads`
threads with **bitwise-identical results** for every thread count:

- **Default selection:** When `num_threads` is omitted (`None` in Python/Rust,
  `SDCA_THREADS_DEFAULT` in C), the library reads `SDCA_NUM_THREADS` once and otherwise uses
  automatic threading (`0`), capped by the active OpenMP team (`torch.set_num_threads`) or
  `OMP_NUM_THREADS` and divided across concurrent calls. Passing `1` runs on the calling thread.
- **Work threshold:** A call times its first rows on the calling thread and spawns workers only
  when remaining work is estimated at $\ge 1\text{ ms}$ ($\ge 0.2\text{ ms}$ on an OpenMP team).
- **Training (`fit`):** Coordinate-ascent epochs are sequential; only duality-gap evaluations
  and kernel validation run in parallel.
- **Floating-point environment:** Every C ABI call executes with IEEE subnormals enabled and
  round-to-nearest (even if the caller set flush-to-zero, as XLA:CPU does) and restores the
  caller's environment on return.

## Environment variables

For the runtime-compiled extension switches (`SDCA_CPU_EXTENSION`, `SDCA_CUDA_LAUNCHER`,
`SDCA_JAX_FFI`), `auto` (default) compiles and caches the extension on first use when build
tools are available, or falls back with a one-time warning (`*.build-failed.txt` suppresses
retries across processes for 24 h); `on` (`1`) requires the extension and raises
`RuntimeError` on failure; `off` (`0`) disables it silently.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SDCA_NUM_THREADS` | unset (auto) | Default thread limit (`< 4096`; `0` = auto, `1` = serial) for native losses, steps and evaluations |
| `SDCA_CPU_EXTENSION` | `auto` | `auto \| on \| off`: JIT-compiled PyTorch CPU C++ extension (`libsdca.torch.explain_backend` reports status) |
| `SDCA_CUDA_LAUNCHER` | `auto` | `auto \| on \| off`: C++ `cuLaunchKernel` autograd launcher for fused CUDA kernels (Triton 3.3–3.9) |
| `SDCA_JAX_FFI` | `auto` | `auto \| on \| off`: XLA FFI custom-call handler for native JAX CPU losses |
| `SDCA_CACHE_DIR` | `~/.cache/libsdca` | Cache directory for compiled JAX FFI handlers |
| `TORCH_EXTENSIONS_DIR` | PyTorch cache | Cache directory for compiled PyTorch CPU and CUDA extensions |
| `OMP_NUM_THREADS` | unset | Caps automatic threads when no OpenMP runtime is loaded |
| `SDCA_LIBRARY_DIR` | unset | Override directory from which Python loads `libsdca_c.so` |
| `SDCA_BLAS_BACKEND` | `AUTO` | Build-time BLAS selection (`AUTO`, `EXTERNAL`, `BUILTIN`) for `pip install`, `python/build.py`, `rust/build.rs`, and CMake |
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
| Python | `ValueError` (or `TypeError`) | `RuntimeError` |
| Rust | `Error::InvalidArgument` | `Error::Native` |
| C++ | `std::invalid_argument` | `std::runtime_error` |
| C | `SDCA_INVALID_ARGUMENT` | `SDCA_FAILURE` (message in `sdca_last_error()`) |
| MATLAB | `LIBSDCA:*` error | `LIBSDCA:*` error |

## References

1. K. Crammer and Y. Singer. On the algorithmic implementation of multiclass
   kernel-based vector machines. *JMLR*, 2001.
2. K. Crammer and Y. Singer. A family of additive online algorithms for
   category ranking. *JMLR*, 2003.
3. M. Lapin, M. Hein and B. Schiele. Top-k multiclass SVM. *NIPS*, 2015.
4. M. Lapin, M. Hein and B. Schiele. Loss functions for top-k error: analysis
   and insights. *CVPR*, 2016.
5. S. Shalev-Shwartz and T. Zhang. Stochastic dual coordinate ascent methods for
   regularized loss minimization. *JMLR*, 2013.
6. G. Hinton, O. Vinyals and J. Dean. Distilling the knowledge in a neural
   network. arXiv:1503.02531, 2015.
