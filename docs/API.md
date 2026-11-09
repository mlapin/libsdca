# API reference

The C++ headers, the C ABI (`include/sdca/c_api.h`), Python, Rust and
MATLAB use the same names and defaults for the same things; the tables below
list where an interface differs. The interface guides have examples:
[Python](../python/README.md), [PyTorch](../python/TORCH.md), [JAX](../python/JAX.md),
[Rust](../rust/README.md), [C++](CPP.md), [C](C.md) and [MATLAB](MATLAB.md). Python and Rust go through the C ABI, MATLAB
calls the C++ headers directly, and the PyTorch tensor backend is a separate
implementation of the same losses. JAX offers both `jax.numpy` implementations
and C ABI calls (an XLA FFI call on the CPU, a host callback elsewhere):
`backend="auto"` uses JAX for entropy on every device,
and for other losses where the required arithmetic is available: on accelerators,
and on the CPU for the nonsmooth multiclass and multilabel hinges and small calls.
See [JAX backends](../python/JAX.md#backends). The training functions take labels as
`y`, the loss functions targets as `target`, and the loss or objective is
selected by `loss` and `objective`.

## Entry points

| Task | Python | PyTorch | JAX | Rust | C | C++ | MATLAB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | `fit`, `SDCAClassifier` | | | `fit`, `fit_distillation` | `sdca_fit`, `sdca_fit_distillation` | `make_solver(context).solve()` | `matsdca_fit` |
| loss and gradient | `loss_and_grad` | `SDCALoss`, `sdca_loss` | `libsdca.jax.sdca_loss` | `loss_and_grad` | `sdca_loss_gradient` | `loss_and_gradient` | `matsdca_loss` |
| coordinate step | `coordinate_update` | | | `coordinate_update`, `coordinate_update_distillation` | `sdca_update_objective`, `sdca_update_distillation` | `update_dual_variables` of an objective | |
| distillation | `distillation_loss_and_grad` | `DistillationLoss`, `distillation_loss` | `libsdca.jax.distillation_loss` | `distillation_loss_and_grad` | `sdca_distillation_forward`, `sdca_distillation_backward` | `distillation_loss`, `distillation_loss_and_gradient` | |
| proximal distillation | `proximal_distillation_loss_and_grad` | `ProximalDistillationLoss`, `proximal_distillation_loss` | `libsdca.jax.proximal_distillation_loss` | `proximal_distillation_loss_and_grad` | `sdca_proximal_distillation_rows` | `proximal_distillation_loss`, `proximal_distillation_loss_and_gradient` | |

The supported C++ entry points are the `sdca/solver.h`, `sdca/prox.h` and `sdca/loss.h`
headers and their documented types and functions. `sdca::detail` and Python names that
start with an underscore are internal.

## Choosing a loss

| You want | Loss | PyTorch | NumPy / training |
| --- | --- | --- | --- |
| Ordinary multiclass classification | softmax cross-entropy | `CrossEntropyLoss()` | `"cross_entropy"` |
| Top-k accuracy (ambiguous or near-duplicate classes) | top-k entropy [4] | `TopKCrossEntropyLoss(k=5)` | `"topk_cross_entropy"`, `k=5` |
| A max-margin (SVM) classifier | multiclass hinge [1] | `MulticlassHingeLoss()` | `"multiclass_hinge"` |
| A margin loss with a Lipschitz gradient | smooth hinge [4] | `MulticlassHingeLoss(gamma=1.0)` | `"multiclass_hinge"`, `gamma=1.0` |
| A top-k margin on each of the k worst competitors | top-k hinge β [4] | `TopKHingeLoss(k=5)` | `"topk_hinge_beta"`, `k=5` |
| A top-k margin on the average of the k worst competitors | top-k hinge α [3] | `TopKHingeLoss(k=5, variant="alpha")` | `"topk_hinge_alpha"`, `k=5` |
| Several correct labels, ranked | multilabel hinge [2] | `MultilabelHingeLoss()` | `"multilabel_hinge"` |
| Several correct labels, as probabilities | multilabel cross-entropy | `MultilabelCrossEntropyLoss()` | `"multilabel_cross_entropy"` |
| A damped cross-entropy step | proximal cross-entropy [5] | `ProximalCrossEntropyLoss(alpha=0.5)` | `"proximal_cross_entropy"` (loss only) |
| To match a teacher's logits | distillation [6] | `DistillationLoss(temperature=4.0)` | `distillation_loss_and_grad`, `fit(X, P, "distillation")` |
| Distillation with a damped step | proximal distillation | `ProximalDistillationLoss(4.0, alpha=0.5)` | `proximal_distillation_loss_and_grad` |

Every hinge takes `gamma`: 0 (the default) is the nonsmooth hinge and `gamma > 0` its
smoothed version, a Moreau envelope with step `gamma`. A larger `gamma` gives a smoother
loss. `k` must lie in `[1, classes - 1]`; only the top-k losses use it. As `alpha → 0` the
proximal losses tend to their plain counterparts. The multilabel losses take a boolean
mask with at least one positive label per row; the multilabel hinge also needs one
negative.

## Loss definitions

Scores `z` in `R^m`, target `y`, and competitor margins `h_j = 1 + z_j - z_y` for `j ≠ y`,
sorted as `h_[1] ≥ h_[2] ≥ ...`. For label sets, `Y` holds the positive labels.

| Name | Definition |
| --- | --- |
| `cross_entropy` | `log Σ_j exp(z_j) - z_y` |
| `topk_cross_entropy` | `max_{p ∈ Δ_k} ⟨a, p⟩ - (1 - s) log(1 - s) - Σ_j p_j log p_j` with `a_j = z_j - z_y`, `s = Σ_j p_j` and the top-k simplex `Δ_k = {p ≥ 0 : s ≤ 1, p_j ≤ s / k}` over `j ≠ y`; `cross_entropy` at `k = 1` |
| `multiclass_hinge` | `max(0, h_[1])` |
| `topk_hinge_alpha` | `max(0, (1/k) Σ_{i≤k} h_[i])` |
| `topk_hinge_beta` | `(1/k) Σ_{i≤k} max(0, h_[i])` |
| hinges with `gamma > 0` | `max_{p ∈ S} ⟨h, p⟩ - (gamma/2) ‖p‖²`, with `S = Δ_k` for α and `S = {0 ≤ p_j ≤ 1/k, Σ_j p_j ≤ 1}` for β; within `gamma/2` of the nonsmooth loss, with a `1/gamma`-Lipschitz gradient |
| `multilabel_hinge` | `max(0, 1 + max_{j ∉ Y} z_j - min_{y ∈ Y} z_y)` |
| `multilabel_cross_entropy` | `log Σ_j exp(z_j) - (1/|Y|) Σ_{y ∈ Y} z_y`: cross-entropy to the uniform distribution on `Y`, not per-class sigmoid BCE |
| `proximal_cross_entropy` | `min_u CE(u, y) + ‖u - z‖² / (2 alpha)`; its gradient is the SDCA dual step |
| distillation | `-Σ_j softmax(t/T)_j log softmax(z/T)_j` for teacher logits `t`, without a `T²` factor |
| proximal distillation | `min_u H(softmax(t/T), softmax(u)) + ‖u - z/T‖² / (2 alpha)` |

The top-k hinges exclude the target from the top-k selection, as in the reduced problem
of [5, Prop. 11]; see the [top-k hinge convention](#top-k-hinge-convention).

## Objectives and losses

Every interface takes the same eight names (`topk_hinge` is an alias for
`topk_hinge_beta`, the default top-k hinge variant; MATLAB's `matsdca_fit` also
keeps its v0.2.0 names). The hinges are smoothed by `gamma`: `gamma = 0` (the
default) is the nonsmooth hinge and `gamma > 0` its Moreau-Yosida smoothing
(the Moreau envelope / proximal hinge loss with step `gamma`, analogous to step
`alpha > 0` in `proximal_cross_entropy` and `proximal_distillation_loss`). Only
the names with `topk` use `k`; every loss checks it
([domains](#parameter-domains-and-numerical-limits)).

| Name | Loss | C++ objective (v0.2.0 class) | C ABI | Rust |
| --- | --- | --- | --- | --- |
| `cross_entropy` | softmax cross-entropy | `l2_entropy(c)` | `SDCA_CROSS_ENTROPY` | `CrossEntropy` |
| `topk_cross_entropy` | top-k entropy; `cross_entropy` at `k = 1` | `l2_entropy_topk(c, k)` | `SDCA_TOPK_CROSS_ENTROPY` | `TopKCrossEntropy` |
| `multiclass_hinge` | multiclass hinge of Crammer and Singer | `l2_hinge_topk(c, 1)`, `l2_hinge_topk_smooth(c, 1, gamma)` | `SDCA_MULTICLASS_HINGE` | `MulticlassHinge` |
| `topk_hinge_alpha` | top-k hinge α: the hinge of the average of the `k` largest margins | `l2_hinge_topk(c, k)`, `l2_hinge_topk_smooth(c, k, gamma)` | `SDCA_TOPK_HINGE_ALPHA` | `TopKHingeAlpha` |
| `topk_hinge_beta` (`topk_hinge`) | top-k hinge β: the average of the `k` largest hinges | `l2_topk_hinge(c, k)`, `l2_topk_hinge_smooth(c, k, gamma)` | `SDCA_TOPK_HINGE_BETA` (`SDCA_TOPK_HINGE`) | `TopKHingeBeta` (`TopKHinge`) |
| `multilabel_hinge` | multilabel hinge | `l2_multilabel_hinge(c)`, `l2_multilabel_hinge_smooth(c, gamma)` | `SDCA_MULTILABEL_HINGE` | `MultilabelHinge` |
| `multilabel_cross_entropy` | cross-entropy with a uniform target over the labels | `l2_multilabel_entropy(c)` | `SDCA_MULTILABEL_CROSS_ENTROPY` | `MultilabelCrossEntropy` |
| `proximal_cross_entropy` | Moreau envelope of the cross-entropy with step `alpha` (loss only) | — | `SDCA_PROXIMAL_CROSS_ENTROPY` | `Loss::ProximalCrossEntropy` |

The C ABI ids are the row order (0-7) of `enum sdca_loss`; training takes ids
0-6. Python takes the names as strings (`LOSSES`, `OBJECTIVES`, and `HINGES` for
the four hinges), as do MATLAB's `opts.objective` and `opts.loss`; C++
`sdca::loss_kind` has the same names and ids. `cross_entropy` on label sets
trains `multilabel_cross_entropy`. At `k = 1`, `topk_hinge_alpha` and
`topk_hinge_beta` are `multiclass_hinge`.

MATLAB's `matsdca_fit` also accepts the names of v0.2.0, whose smoothing
default is `gamma = 1`: `softmax`, `l2_entropy_topk`, `msvm` (nonsmooth),
`msvm_smooth`, `l2_hinge_topk`, `l2_topk_hinge`, `mlsvm` (nonsmooth),
`mlsvm_smooth`, `mlsoftmax` and their `l2_multiclass_*`/`l2_multilabel_*`
synonyms (`matsdca_fit('help', 'objective')`). `model.objective` is the name as
given.

### PyTorch and JAX wrappers

Each wrapper is one name of the table; its options have the names and
defaults of `sdca_loss` (`k=1`, `gamma=0.0`, `alpha=1.0`).

| Module (PyTorch) | Function (PyTorch and JAX) | Name |
| --- | --- | --- |
| `CrossEntropyLoss` | `cross_entropy` | `cross_entropy` |
| `TopKCrossEntropyLoss(k)` | `topk_cross_entropy` | `topk_cross_entropy` |
| `MulticlassHingeLoss(gamma)` | `multiclass_hinge_loss` | `multiclass_hinge` |
| `TopKHingeLoss(k, *, variant="beta", gamma)` | `topk_hinge_loss` | `topk_hinge_beta`, or `topk_hinge_alpha` with `variant="alpha"` |
| `TopKHingeAlphaLoss(k, *, gamma)` | `topk_hinge_alpha_loss` | `topk_hinge_alpha` |
| `TopKHingeBetaLoss(k, *, gamma)` | `topk_hinge_beta_loss` | `topk_hinge_beta` |
| `MultilabelHingeLoss(gamma)` | `multilabel_hinge_loss` | `multilabel_hinge` |
| `MultilabelCrossEntropyLoss` | `multilabel_cross_entropy` | `multilabel_cross_entropy` |
| `ProximalCrossEntropyLoss(alpha)` | `proximal_cross_entropy` | `proximal_cross_entropy` |
| `SDCALoss(loss)` | `sdca_loss` | any of the eight |
| `LinearCrossEntropyLoss` (PyTorch only) | `linear_cross_entropy(hidden, weight, target, bias=None)` | `cross_entropy` of `F.linear(hidden, weight, bias)` |

The default top-k hinge is β (`variant="beta"`). The PyTorch
`CrossEntropyLoss`/`cross_entropy` are drop-in replacements for
`torch.nn`'s, with PyTorch's class axis, weights and `ignore_index`.
`linear_cross_entropy` takes the options of `cross_entropy` (the class weights as
`class_weight`), `chunk_bytes`, the memory budget of its chunks of logits, and `recompute`
(recompute the chunks in the backward instead of keeping the gradients); it never stores
the `(N, classes)` logits ([TORCH.md](../python/TORCH.md#linear-layer-and-cross-entropy-without-the-logits)).

## Distillation losses

NumPy (`distillation_loss_and_grad`, `proximal_distillation_loss_and_grad`),
PyTorch (`DistillationLoss`, `ProximalDistillationLoss`), JAX (`libsdca.jax`),
Rust (`distillation_loss_and_grad`, `proximal_distillation_loss_and_grad`),
C++ (`sdca::distillation_loss`, `sdca::proximal_distillation_loss`), and the
C ABI ([C](C.md#tensor-entry-points)). Soft-target SDCA training takes a
`(n, num_classes)` target probability matrix: Python `fit`/`coordinate_update` with
`objective="distillation"`, Rust `fit_distillation`/`coordinate_update_distillation`,
C `sdca_fit_distillation`/`sdca_update_distillation`, and C++ `sdca::l2_distillation`
with `sdca::distribution_output`. Both losses take student and teacher **logits** of the same
shape and dtype; the teacher gets no gradient. With `p = softmax(teacher / T)` and
`q = softmax(student / T)`, per example:

| Loss | Value | Gradient in `student` |
| --- | --- | --- |
| `distillation_loss(student, teacher, temperature=1.0)` | `H(p, q) = -sum_j p_j log q_j` | `(q - p) / T` |
| `proximal_distillation_loss(student, teacher, temperature=1.0, alpha=1.0)` | `min_u H(p, softmax(u)) + ‖u - student/T‖² / (2 alpha)` | `(pi - p) / T`, `pi = softmax(u*)` |

`temperature` and `alpha` must be finite and positive. The temperature divides both
logits after each is centered at its maximum; no `T²` factor is applied. The
proximal loss is the Moreau envelope of `distillation_loss` in `student / T`: it
lies below it and tends to it as `alpha -> 0+`. Both gradients are exactly zero at
equal represented logits. In PyTorch they are first-order (double backward raises).
In JAX `jvp` and `jacfwd` raise, but the gradient is `jax.numpy` code that `jax.hessian`
and reverse over reverse differentiate again. `reduction`, `dim` (JAX `axis`) and
`precision` work as in `sdca_loss`. The modules take the same options.

```python
import torch
from libsdca import DistillationLoss, ProximalDistillationLoss

student = torch.randn(8, 10, requires_grad=True)
teacher = torch.randn(8, 10)
loss = DistillationLoss(temperature=4.0)(student, teacher)
prox = ProximalDistillationLoss(temperature=4.0, alpha=0.5)(student, teacher)
(loss + prox).backward()
```

```python
import jax
from libsdca.jax import distillation_loss, proximal_distillation_loss

s, t = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 10))
grad = jax.grad(lambda x: proximal_distillation_loss(x, t, 4.0, alpha=0.5))(s)
```

Backends: on CUDA, one fused Triton kernel per loss; rows up to 4096 classes stay
in registers and wider rows stream column tiles. The C++ `cuLaunchKernel` launcher
launches `distillation_loss` when it is active, and Python launches the proximal
kernel. Without Triton 3.3 or newer, CUDA uses tensor operations. On the CPU, the
CPU extension (built on first use), else the native library. JAX uses `jax.numpy`
on every device. The proximal loss solves one scalar normalizer per row by safeguarded Halley steps
(at most 100 in the kernels, 200 in JAX; the CUDA tensor path takes a fixed 20); on the CPU a
Taylor model of the probabilities about the last evaluation gives the root near it.

Precision in PyTorch and JAX: float32 work for float32, float16 and bfloat16 logits,
float64 for float64 logits or `precision="float64"`; a `temperature` or `alpha` outside
the float32 normal range promotes to float64 (JAX: needs `jax_enable_x64`). The CPU
extension and the native library compute the proximal loss in float64. Values are
float64 for float64 logits and float32 otherwise. NumPy and Rust work in float64 by
default (`precision="float32"`, `Precision::Float32` select float32 work for float32
logits) and return float64 values; C++ `sdca::distillation_loss` and
`sdca::proximal_distillation_loss` work in and return `double`. Gradients keep the
logits dtype everywhere. Tested accuracy
(`test_distillation.py`, `test_distillation_references.py`): values and
gradients within 64 eps of the logits' dtype of an mpmath reference (gradients
relative to the row's largest entry), at temperatures 0.25-4 and `alpha` 1e-6-1e6
([numerics](numerics.md#further-precision-limits)).
Nonfinite logits raise `ValueError` on the CPU and in eager JAX; CUDA and traced
JAX return `nan` for that row.

## Top-k hinge convention

`topk_hinge_alpha` is the alpha variant: take the largest `k` **non-target**
margins `1 + z[j] - z[y]`, average them, then clip the average at zero.
`topk_hinge_beta` is the beta variant: clip each selected margin at zero before
averaging. A positive `gamma` smooths each with the corresponding conjugate
quadratic term.

This target exclusion follows the reduced coordinate problem in Proposition 11
of [Lapin et al., *Loss Functions for Top-k Error: Analysis and Insights*](https://arxiv.org/html/1512.00486v2).
It differs from a literal full-vector reading of the alpha loss in equations 6
and 12, where the target's zero margin participates in the top-k operation.
For `z=[0,0,-4]`, zero-based target `0`, and `k=2`, libsdca averages non-target
margins `[1,-3]` and returns `0`. Including the target margin gives
`top2([0,1,-3])=[1,0]`, so equation 6 gives `0.5`. With `gamma=1`,
libsdca's smoothed alpha loss is also `0`, while the full-vector simplex
form in equation 12 gives `0.25`. The nonsmooth beta value is unaffected by inserting
a zero margin, because its margins are clipped individually.

## Options

| Meaning | Python, Rust | MATLAB | C++ | Default |
| --- | --- | --- | --- | --- |
| loss weight | `C` / `c` | `c` or `C` | objective constructor | 1 |
| top-k | `k` | `k` | objective constructor | 1 |
| hinge smoothing (0: nonsmooth) | `gamma` | `gamma` (v0.2.0 names: 1) | the `_smooth` class | 0 |
| gap tolerance | `tol` | `epsilon` or `tol` | `criteria.epsilon` | 1e-3 |
| epoch limit | `max_epoch` | `max_epoch` | `criteria.max_epoch` | 1000 |
| evaluation period | `eval_epoch` (>= 1) | `eval_epoch` (0: only at the end) | `criteria.eval_epoch` (0: only at the end) | 10 |
| evaluate first | `eval_on_start` | `eval_on_start` | `criteria.eval_on_start` | false |
| rebuild `W = center + X' A` at each evaluation | `recompute_primal` | `recompute_primal` | `criteria.recompute_primal` | true |
| how the hinge projections solve ([below](#projection-precision)) | `projection_precision` | `projection_precision` | `criteria.projection` | `"float64"` |
| what the hinge epochs leave out ([below](#shrinking)) | `shrinking` | `shrinking` | `criteria.shrinking` | `"auto"` |
| Gram matrix input | `kernel` | `is_dual` or `kernel` | `make_input_kernel` | false |
| regularizer center | `center` | `W0` or `center` | `context.primal_initial` | none |
| warm start | `initial_dual` | `initial_dual` or `A` | the dual buffer | none |
| threads for the rows of losses and coordinate steps | `num_threads` (Python `None`, Rust `Option`) | `num_threads` (`matsdca_loss`) | C ABI `options.num_threads` (`SDCA_THREADS_DEFAULT`) | `SDCA_NUM_THREADS`, else automatic |
| threads of a fit's evaluations and kernel check (epochs are sequential) | `SDCA_NUM_THREADS` | `SDCA_NUM_THREADS` | `criteria.num_threads`; C ABI `options.num_threads` | `SDCA_NUM_THREADS`, else automatic |

`recompute_primal` (C ABI: the `sdca_options` field; sklearn: the
`SDCAClassifier` parameter) applies to feature input. Each evaluation of the
duality gap first rebuilds `W = center + X' A` (MATLAB: `W0 + X * A'`) from the duals. With
`false`, the evaluation uses the `W` maintained by the updates, which drifts
by rounding. The duals are the same either way until the next update.
Every interface defaults to `true`; it is a run-time option, not a build
property. A fit of float32 data in `precision="float64"` keeps `W` in double
([Precision](#precision)), where the updates drift only by double rounding, and
returns it rounded once: it does not rebuild `W` from the float32 duals
returned, whose rounding would enter every entry of `W`, so the option has no
effect there.

Besides every `eval_epoch` epochs, a fit with `tol > 0` evaluates up to twice
in between, at the first epoch where the certified relative gap, decreasing
at its rate between the last two evaluations (the first from a zero dual
start, whose relative gap is one), reaches `tol`. The stop stays certified by
an evaluation, and a history holds at most `max_epoch / eval_epoch + 2` rows
(one early evaluation with `eval_on_start` or when `max_epoch` is not a
multiple of `eval_epoch`, none with both).

A training Gram matrix must be symmetric positive semidefinite. Every
interface checks a nonnegative diagonal, symmetry to `1e-5 sqrt(K_ii K_jj)`
and `|K_ij| <= (1 + 1e-3) sqrt(K_ii K_jj)`, which rejects non-finite and
grossly indefinite kernels; it does not prove positive semidefiniteness, and
the certificate of an indefinite kernel is meaningless.

SDCA samples the examples uniformly, so rows of very different norms (a few
large rows among many small ones) converge slowly: the step of a row scales
with `1 / (C ||x_i||^2)`. Standardize or normalize the features, or scale the
kernel, when the norms spread over orders of magnitude. Tight tolerances on
nonsmooth hinges may also need a larger `max_epoch` than the default 1000 (for
example, `multiclass_hinge` on the 20,000-example benchmark set takes about
17,800 epochs to reach `tol=1e-6`).

### Shrinking

`shrinking` selects what the epochs of a hinge fit (`multiclass_hinge`,
`topk_hinge_alpha`, `topk_hinge_beta`, `multilabel_hinge`, smooth or not) leave
out. Most dual variables of a hinge sit at a bound: an example beyond its
margins has a zero block, and a single-label block is nonzero on few classes.

- `"heuristic"`: liblinear-style shrinking. The first epoch of a fit and the
  epoch of each scheduled evaluation update every class of every example; the
  epochs in between skip the examples whose last step left their duals
  unchanged, and step a single-label example on its label and its classes of
  nonzero dual alone (the scores and the update of `W` then cost those classes
  only). A skipped example whose margins break meanwhile is updated again at
  the next full pass. Those epochs cost a fraction of a full pass, so the
  evaluation at a multiple of `eval_epoch` waits until the epochs since the
  last one have taken as many steps as `eval_epoch` full passes, or for 50
  epochs (the C++ `criteria.full_pass_epoch`), and is never skipped at
  `max_epoch`: a full pass and an evaluation every 10 epochs would take most
  of the time of a tight fit. Without
  shrinking the schedule is every `eval_epoch` epochs, as before.
- `"safe"`: gap-safe screening (Ndiaye, Fercoq, Gramfort and Salmon, JMLR
  2017). The primal is 1-strongly convex in `W`, so the certified gap of an
  evaluation bounds `|W - W*| <= sqrt(2 gap)`, and with it every margin at the
  optimum. A zero dual that the bound proves zero at the optimum is left out
  for good: a whole example of the single-label hinges when all its margins
  stay negative, a class of `multiclass_hinge`, of `topk_hinge_beta` and of the
  k = 1 hinges when its margin does (or, nonsmooth with k = 1, when another
  margin stays above it). Screening runs after the evaluations that halve the
  certified gap since the last one (quarter it, and so on, after one that
  proves nothing), on feature input.
- `"both"`: the two; `"auto"`: the default, `"both"`; `"off"`: every class of
  every example in every epoch.

The entropy objectives update everything whatever the value: their duals
never reach the bounds. Every evaluation covers every example and class, so
`solved` certifies the gap of the returned weights and duals as without
shrinking; a fit that shrinks reaches the same objective to the certified
tolerance, with a different iterate (the heuristic changes the order of the
steps; screening alone only skips steps that change nothing, in practice the
same iterate). Results are deterministic and identical for every thread count.
What it saves is in [performance](performance.md#solver).

| Interface | Spelling | Takes it |
| --- | --- | --- |
| Python | `"auto"`, `"off"`, `"heuristic"`, `"safe"`, `"both"` | `fit`, `Solver.fit`, `SDCAClassifier` |
| Rust | `Shrinking::Auto`, `::Off`, `::Heuristic`, `::Safe`, `::Both` | `FitOptions` (`shrinking`) |
| C | `SDCA_SHRINKING_AUTO` (0), `_OFF` (1), `_HEURISTIC` (2), `_SAFE` (3), `_BOTH` (4) | `shrinking` of `struct sdca_options`, its last field; a struct of an earlier header (its `size` up to `warm`) takes `AUTO` |
| MATLAB | `'auto'`, `'off'`, `'heuristic'`, `'safe'`, `'both'` | `opts` of `matsdca_fit` (reported as `model.shrinking`) |
| C++ | `sdca::shrinking_mode::automatic`, `::off`, `::heuristic`, `::safe`, `::both` | `criteria.shrinking` (`criteria.full_pass_epoch`, 50: the most epochs between the heuristic's evaluations and full passes) |

Any other value is an invalid argument.

### Projection precision

`projection_precision` selects how the multilabel hinge projection solves: the
two-simplex of `multilabel_hinge` and its smooth form, in coordinate steps,
primal losses and loss functions. The top-k hinge projections
(`multiclass_hinge`, `topk_hinge_alpha`, `topk_hinge_beta`) run the hardened
extended-precision solver in both modes; the entropy objectives have no such
projection. It is separate from the working `precision`.

- `"float64"` (default) solves in double, then runs an acceptance check: every
  entry lies clear of its threshold by a multiple of that threshold's error
  bound, the active-sum case is clear, and no threshold dwarfs the score
  spread. A result that fails the check is solved again with the hardened
  extended-precision solver. The two modes agree to rounding.
- `"extended"` always runs the hardened extended-precision solver (long double
  sums, tolerances from each test's own terms).

`"float64"` exists for speed. At 10 classes a `multilabel_hinge` step takes 19%
longer than in v0.2.0 with `"float64"` and 64% longer with `"extended"`; at
1000 classes it is 0–5% faster with `"float64"` and 3–12% slower with
`"extended"`. With float32 storage at many classes the acceptance check mostly fails
and the steps run the extended solver. Pick `"extended"` to reproduce runs made
with it bit for bit, or for extreme inputs (scores spread over many orders of
magnitude, near ties at a threshold) where the check often fails and the double
attempt is wasted; a failed check gives the extended result either way. The
acceptance check and the measured errors are in
[numerics.md](numerics.md#projection-precision).

| Interface | Spelling | Takes it |
| --- | --- | --- |
| Python | `"float64"`, `"extended"` | `fit`, `loss_and_grad`, `coordinate_update`, the `Solver` methods, `SDCAClassifier` |
| Rust | `ProjectionPrecision::Float64`, `::Extended` | `FitOptions`, `LossOptions`, `UpdateOptions` (`projection_precision`) |
| C | `SDCA_PROJECTION_FLOAT64` (0), `SDCA_PROJECTION_EXTENDED` (1) | `projection_precision` of `struct sdca_options` |
| MATLAB | `'float64'`, `'extended'` | `opts` of `matsdca_fit` (reported as `model.projection_precision`) and `matsdca_loss` |
| C++ | `sdca::projection_precision::float64`, `::extended` | `criteria.projection`; the trailing argument of `loss_and_gradient`, `update_dual_variables` and `prox_two_simplex` |

Any other value is an invalid argument. The PyTorch and JAX losses do not take
the option; their native host paths use `"float64"`.

libsdca builds one native library with IEEE floating-point semantics
(`-fno-fast-math -ffp-contract=off`). Building it with `-ffast-math` or
`-Ofast` is unsupported: `-ffinite-math-only` removes the checks that reject
nonfinite inputs. The working arithmetic is chosen at run time with
`precision` (below).

## Threads

Loss functions (`loss_and_grad`, `matsdca_loss`, `sdca_loss_gradient`) and
coordinate steps (`coordinate_update`, `sdca_update_objective`) evaluate each
row independently across up to `num_threads` threads, producing bitwise-identical
results for every thread count:

- **Default thread count:** When `num_threads` is omitted (`None` in Python/Rust,
  `SDCA_THREADS_DEFAULT` in C), the library reads `SDCA_NUM_THREADS` once on first
  use and otherwise defaults to `0` (automatic). Automatic uses the CPUs available
  to the process (affinity mask on Linux; performance + efficiency cores on macOS),
  capped by a loaded OpenMP runtime's thread count (such as `torch.set_num_threads`)
  or `OMP_NUM_THREADS`, divided evenly across concurrent calls in flight. Passing
  `num_threads=1` (or `OMP_NUM_THREADS=1` under automatic) runs serially on the
  calling thread.
- **Small-input threshold:** A call times its first rows on the calling thread and
  spawns worker threads only when the remaining work is estimated at `>= 1 ms`
  (`>= 0.2 ms` on a loaded OpenMP team), so small inputs stay on the calling
  thread without thread-launch overhead.
- **OpenMP and PyTorch integration:** When an OpenMP team is active, rows are
  partitioned like an OpenMP static loop to preserve cache locality with
  surrounding PyTorch ops. Calls nested inside a multithreaded call or an OpenMP
  parallel region run serially. The PyTorch native CPU backend caps automatic
  threads at `torch.get_num_threads()`; the PyTorch CPU extension
  ([`SDCA_CPU_EXTENSION`](#environment-variables)) uses ATen's intra-op pool
  directly.
- **Training (`fit`):** Coordinate-ascent epochs are sequential (1 thread); only
  duality-gap evaluations and the initial kernel check run in parallel.

Every C ABI compute entry point (Python, Rust, and native JAX) executes in the
default IEEE floating-point environment—keeping subnormals enabled even if the
caller enabled flush-to-zero (`XLA:CPU` or `torch.set_flush_denormal`), with
round-to-nearest and traps disabled—and restores the caller's environment on
return (including on error). Direct C++ header and MATLAB callers use the
caller's floating-point environment.

## Precision

| Storage | Precision of the solve | Python `precision` | Rust `precision` | MATLAB `opts.precision` | C++ |
| --- | --- | --- | --- | --- | --- |
| `float` (`f32`) | `float` (`f32`) | `"float32"` (`"fp32"`, `"single"`, `np.float32`) | `f32` data, `Precision::Float32` | `single` data, `'single'` (`'float32'`, `'fp32'`) | `<float, float>` |
| `float` (`f32`) | `double` (`f64`) | `"float64"` (default for `float32`) | `f32` data, `Precision::Float64` (default) | `single` data, `'double'` (default) | `<float, double>` |
| `double` (`f64`) | `double` (`f64`) | `"float64"` (default for `float64`) | `f64` data, `Precision::Float64` (default) | `double` data, `'double'` (default) | `<double, double>` |

`precision` is the precision of the solve: the working arithmetic of training,
coordinate steps and losses, and the solver's iterate (the weights `W` and the
duals). The data and the returned arrays (weights, duals, gradients) keep the
input's dtype; the evaluation (history and certificate) computes in double in
every precision. With float32 data and `"float64"` (the default), a fit keeps
`W` and the duals in double copies of the returned float32 arrays, `8 (d + n) m`
bytes more for `d` features, `n` examples and `m` classes (`8 n m` on a kernel),
and each evaluation rounds them once into the returned arrays and certifies
those. Steps of float32 storage lose the updates below the rounding of `W` and of
the duals near the optimum, which stalls float32 hinge fits at a relative gap
of a few 1e-6 (4.7e-6 on the standardized digits); the double iterate reaches
the tolerances of float64 data in as many epochs, down to the rounding of the
returned values (3.5e-7 on digits). `"float32"` keeps the iterate in float, the
fast mode. The data are never copied.
On the tested corpus, `"float64"` working arithmetic keeps `float32`-storage
entropy blocks within a few float ulps of the double solution, and `"float32"`
working arithmetic makes the entropy steps 1.8–3.2× as fast; the hinge
steps compute in double either way and gain nothing
([limits](numerics.md#known-limits)). The loss
functions compute in double for `"float64"`, and in float for `"float32"`
(`matsdca_loss` with single data and `opts.precision = 'single'`), falling
back to double where float cannot hold `gamma`, `alpha` or a row's score
spread.
What each precision guarantees, and its measured accuracy, is in
[numerics.md](numerics.md#what-each-precision-guarantees).

### Storage and compute support

The native column below describes the C ABI and its NumPy/scikit-learn/Rust
frontends.
The tensor frontends have additional combinations; choosing a supported
combination does not guarantee finite output for every finite input.

| Storage dtype | Native compute | PyTorch tensor compute | JAX compute |
| --- | --- | --- | --- |
| `float16`, `bfloat16` | unsupported for native NumPy training/loss entry points | `float32`, `float64` | `float32`; `float64` with x64 enabled |
| `float32` | `float32`, `float64` | `float32`, `float64` | `float32`; `float64` with x64 enabled |
| `float64` | `float64` (`float32` compute rejected) | `float64` (`float32` compute rejected) | `float64` with x64 enabled (`float32` compute rejected) |

MATLAB calls the C++ headers directly and additionally accepts double storage
with float32 working arithmetic in fit and prox; this is not a supported C ABI
pair. `matsdca_loss` computes double scores in double for either `precision`. The prox gateway also retains its explicit long-double extension.

JAX's native backend adapts tensor storage to the native types; it is distinct
from passing a half array to the NumPy API. Native history and scalar loss
values are double; native weights, duals and gradients retain the selected
float32/float64 storage. Tensor gradients retain the logits dtype. Every
PyTorch and JAX loss value has dtype `promote(logits dtype, float32)`, as
under `torch.autocast`: float32 for 16-bit and float32 logits, float64 for
float64. See [PyTorch precision](../python/TORCH.md#precision)
for backend-specific defaults and promotion to float64 to avoid intermediate
overflow. JAX `backend="jax"` needs `jax_enable_x64=True` for float64 work;
`backend="auto"` can instead select the native backend for non-entropy losses
(always without float64, at `precision=None`).

### Parameter domains and numerical limits

With `m` classes, the public loss APIs require `m >= 2`, integer `1 <= k < m`,
finite scores, and valid targets. These checks also apply to generic options
unused by the selected loss. Integer targets are zero-based except in MATLAB,
where they are one-based. Boolean label masks must have at least one positive
label; multilabel hinges and `proximal_cross_entropy` also need a negative label. Reductions are `none`,
`sum`, or `mean` in the tensor APIs.

| Parameter | Accepted domain | Additional numerical restrictions |
| --- | --- | --- |
| `gamma` | finite and nonnegative; 0 is the nonsmooth hinge | A positive `gamma` follows the rules of `alpha`. |
| `alpha` for loss evaluation | finite, strictly positive Python/C double values | Internal projections and final outputs must be representable. Float32 paths may promote to float64; JAX needs x64 for that promotion. |
| Distillation `temperature` | finite and strictly positive | Working arithmetic must represent the scaled logits; half/bfloat16 gradients still round to their storage dtype. |
| Native training/coordinate `C`, positive `gamma` | finite and strictly positive | Must be normal after conversion to the working type. |
| PyTorch `ignore_index` | integer in `[-2**63, 2**63-1]` | May lie outside the class range; matching targets are omitted. |
| Nonzero feature norm squared / coordinate `norm2` | finite and positive | `norm2` and `C * norm2` must remain normal in storage. Zero feature rows have a separate zero-score update. |
| Training `tol` | finite and nonnegative | A smaller requested gap does not force a certificate below arithmetic resolution. |
| Cross-entropy `weight` (PyTorch named API) | finite, nonnegative vector of length `m`, on the logits device | Matching storage dtype, with float32 weights additionally accepted for half logits under autocast; zero-weight mean is undefined. |

Normal float32 magnitudes range from about `1.17549435e-38` to
`3.40282347e38`; float64 from `2.2250738585072014e-308` to
`1.7976931348623157e308`. These describe representation, not a guaranteed
safe interval for every product, difference, projection, or class count.
Float16 storage tops out at `65504`; bfloat16 at about `3.38953139e38`.
Wider compute cannot recover information already rounded from the input or
prevent a result from overflowing its output storage. Tests and empirical
ranges, including extreme `gamma`/`alpha` and score spreads, are listed in
[numerics](numerics.md#known-limits) and [PyTorch precision](../python/TORCH.md#precision).
Do not extrapolate their accuracy bounds to all combinations of finite inputs.

## Results

The status names (`none`, `solving`, `solved`, `no_progress`, `max_epoch`,
`max_cpu_time`, `max_wall_time`, `failed`) and the history fields (`epoch`,
`primal`, `dual`, `primal_loss`, `dual_loss`, `regularizer`,
`relative_gap`, `accuracy`, `cpu_time`, `wall_time`) are those of
`sdca::solver_status` and the C ABI. Only `solved` certifies the requested
gap: `P(W) - D(A) + e <= tol max(|P|, |D|)`, with `P` the primal objective
at the returned weights, `D` the dual objective at the returned duals (each
block's storage rounding repaired), both evaluated in double or wider, and
`e` a first-order estimate of that evaluation's rounding (rebuilding `W` changes
only `P`; `D` comes from the duals either way). Since `D(A) <= P* <= P(W)`,
`P(W) - P*` is then at most `tol max(|P|, |D|)` under the
[documented floating-point assumptions](numerics.md#the-solved-certificate); this
is not an interval-arithmetic guarantee, and the distance of `W` from the optimal
weights is not bounded. A
`tol` below what the evaluation can certify (about 1e-12 in double, more
with cancellation) never gives `solved`; training runs to its other limits.
A gap below `-e`, or a coordinate step that fails numerically, ends training
with `failed` and the last consistent iterate.

For feature input with a regularizer center `W0`, `P` includes the full
`0.5*||W-W0||^2` and `D` uses the matching full dual objective. These values
are evaluated without subtracting `0.5*||W0||^2` or adding it back to a
large negative number. The relative stopping scale therefore describes
the centered problem: a large center alone cannot make a fit report
`solved`.

Large centers still limit the weights that a storage type can represent:
float32 values near `1e8` have spacing 8, so an optimal update smaller than
that can be lost. Such a fit cannot certify a tight tolerance and may reach
`max_epoch`. Double storage and arithmetic are preferable for large centers;
even double has spacing 2 near `1e16`.

For the multilabel objectives the `accuracy` field holds the rank loss:
Python names it `rank_loss`, Rust keeps `accuracy` and also accepts
`rank_loss` in `History::get`, and MATLAB has a `rank_loss` field.

With `m` classes, Python returns weights `(d, m)` as a column-major (Fortran-order)
array and duals `(n, m)` row-major, Rust both row-major; MATLAB, following its column-major data `(d, n)`, returns
`W (d, m)` and `A (m, n)`. The letter `k` is reserved for the top-k
parameter.


## Environment variables

Read at run time (`SDCA_NUM_THREADS` and `OMP_NUM_THREADS` once on first native
call; `SDCA_CUDA_LAUNCHER` once on first fused CUDA call; `SDCA_CPU_EXTENSION` on
CPU loss calls; `SDCA_JAX_FFI` when tracing a native JAX CPU loss).

For the three runtime-compiled extension switches (`SDCA_CPU_EXTENSION`,
`SDCA_CUDA_LAUNCHER`, `SDCA_JAX_FFI`), `auto` (the default) compiles and caches the
extension on first use when its build tools are present, or falls back with a
one-time warning naming what is missing (a failed build warns once, writes a
`*.build-failed.txt` marker, and is skipped by subsequent processes for 24 h); `on`
(`1`) requires the extension and raises `RuntimeError` if tools are missing or the
build fails; `off` (`0`) disables it without warning; any other value raises
`ValueError`:

| Variable | Default | Values | What it does |
| --- | --- | --- | --- |
| `SDCA_NUM_THREADS` | unset: automatic | integer `< 4096`; `0` = automatic, `1` = calling thread, `> 1` = upper limit | Threads for native loss rows, coordinate steps, and fit evaluations when `num_threads` is not passed ([Threads](#threads)); out-of-range values are ignored. In the PyTorch native CPU backend, unset caps at `torch.get_num_threads()`. |
| `SDCA_CPU_EXTENSION` | `auto` | `auto`, `on`, `off` (`1`, `0`) | PyTorch CPU C++ extension (needs `ninja`, a C++17/C++20 compiler with OpenMP, and Python headers); falls back to the prebuilt native library. `explain_backend` reports the active path and reason. |
| `SDCA_CUDA_LAUNCHER` | `auto` | `auto`, `on`, `off` (`1`, `0`) | C++ `cuLaunchKernel` launcher for fused CUDA kernels on Triton 3.3–3.9 (needs `ninja`, a C++ compiler, Python headers, `CUDA_HOME` headers, and `libcuda.so.1`; `on` skips the Triton version check); falls back to launching the same kernels through Triton. |
| `SDCA_JAX_FFI` | `auto` | `auto`, `on`, `off` (`1`, `0`) | XLA FFI custom-call handlers for native JAX CPU losses (needs `jax.ffi`, `jaxlib` FFI headers, and a C++17 compiler); falls back to `jax.pure_callback` with identical results. |
| `SDCA_CACHE_DIR` | `$XDG_CACHE_HOME/libsdca` or `~/.cache/libsdca` | directory | Cache directory for compiled JAX FFI handlers (keyed by source, `jaxlib`, platform, compiler, and flags) and their `*.build-failed.txt` markers. |
| `OMP_NUM_THREADS` | unset | integer `< 4096` (first of list) | Caps automatic threads when no OpenMP runtime is loaded in the process; a loaded runtime's thread count takes precedence ([Threads](#threads)). |
| `SDCA_LIBRARY_DIR` | unset | directory | Directory from which Python loads `libsdca_c.so`, overriding `build/python` or the installed package library. |
| `TORCH_EXTENSIONS_DIR` | PyTorch's extension cache | directory | PyTorch cache directory for the CPU extension and CUDA launcher builds and their `*.build-failed.txt` markers. |
| `CXX` | `c++` (`g++` for `python/build.py`) | compiler command (may include flags) | C++ compiler for PyTorch extensions, JAX FFI handlers, `python/build.py`, and the Rust build. |
| `CUDA_HOME` | PyTorch's detection or `/usr/local/cuda` | CUDA toolkit directory | CUDA headers and libraries for the launcher build. |
| `TRITON_INTERPRET` | unset | `1` | Runs Triton kernels in Triton's CPU interpreter for testing (uses Triton math functions instead of `libdevice`). |

Read when building the native library (`pip install`, `python/build.py`, the
Rust crate's `build.rs`):

| Variable | Default | Values | What it does |
| --- | --- | --- | --- |
| `SDCA_BUILD_NATIVE` | `ON` | `ON`, `OFF` (`1`, `0`, `TRUE`, `FALSE` alike); others raise `ValueError` | `pip install` only: `OFF` builds a pure-Python wheel without the native library, for the tensor losses ([DEPENDENCIES.md](../DEPENDENCIES.md#python-configuration)). |
| `SDCA_BLAS_BACKEND` | `AUTO` | `AUTO`, `EXTERNAL`, `BUILTIN` | The BLAS: `AUTO` links the `SDCA_BLAS` libraries if set, else (Python builds) a system CBLAS (`blas`, then `openblas`) with every needed symbol, else the built-in BLAS; the Rust build's `AUTO` takes the built-in BLAS without `SDCA_BLAS`. `EXTERNAL` without a CBLAS fails. |
| `SDCA_BLAS` | unset | linker arguments, e.g. `-lopenblas` or `-L/opt/blas/lib -lblas` | The external BLAS libraries. |
| `SDCA_SOURCE_DIR` | the crate's `native/`, else the repository around it | directory | The C++ sources of the Rust build. |
| `AR` | `ar` | archiver command | The archiver of the Rust build. Cross compilation needs `CXX` and `AR` set to the target's tools. |

CMake options of the C++ packages (`cmake -D<option>=<value>`):

| Option | Default | What it does |
| --- | --- | --- |
| `SDCA_BLAS_BACKEND` | `AUTO` | `AUTO`, `EXTERNAL` or `BUILTIN` BLAS ([DEPENDENCIES.md](../DEPENDENCIES.md)) |
| `SDCA_BUILD_C_API` | `OFF` | build and install the C ABI library `libsdca::sdca_c` |
| `SDCA_BUILD_MATLAB` | `OFF` | build the MATLAB MEX interfaces |
| `SDCA_WARNINGS_AS_ERRORS` | `OFF` | treat the project's compiler warnings as errors |
| `BUILD_TESTING` | `OFF` | build the bundled GoogleTest suites |
| `USE_CHRONO` | `ON` | time with `std::chrono` instead of POSIX `gettimeofday` |
| `USE_ILP64`, `USE_SEQUENTIAL`, `USE_THREADING_SUFFIX` | `ON`, `ON`, `OFF` | with Intel MKL: 64-bit integers, the sequential layer, a threading suffix on binary names |

CMake looks for Intel MKL in `MKL_ROOT_DIR`, else in the `MKLROOT` environment
variable, else in `/opt/intel/mkl`.

## Errors

| | Invalid input or options | Failed computation |
| --- | --- | --- |
| C++ | `std::invalid_argument` | `std::runtime_error` |
| C | `SDCA_INVALID_ARGUMENT` | `SDCA_FAILURE` |
| Python | `ValueError` | `RuntimeError` |
| Rust | `Error::InvalidArgument` | `Error::Native` |
| MATLAB | `LIBSDCA:*` errors | `LIBSDCA:*` errors |

The C ABI, Python and Rust return the message of `sdca_last_error()` for the
errors the native code detects.

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
