# API reference

One set of loss names, option names, status codes and defaults is shared by C++, the C ABI
(`include/sdca/c_api.h`), Python, PyTorch, JAX and Rust. Interface guides:
[Python](PYTHON.md), [PyTorch](TORCH.md), [JAX](JAX.md),
[Rust](RUST.md), [C++](CPP.md) and [C](C.md).

Use an interface guide for runnable examples. This page defines the common contract.

| Looking for | Section |
| --- | --- |
| Choose a loss | [Loss definitions](#loss-definitions), [names across interfaces](#names-across-interfaces) |
| Configure training | [Options](#options), [results](#results) |
| Choose accuracy and execution | [Precision](#precision), [threads](#threads) |
| Diagnose a call | [Errors](#errors), [known limits](#known-limits), [environment variables](#environment-variables) |

## Loss definitions

For scores $z \in \mathbb{R}^m$, target class $y$, and hinge terms
$h_j = 1 + z_j - z_y$ ($j \neq y$) sorted as $h_{[1]} \ge h_{[2]} \ge \dots$
(or positive label set $Y \subset \lbrace 1, \dots, m \rbrace$ for multilabel losses):

| Name | Formula | Parameters | Use case |
| --- | --- | --- | --- |
| `cross_entropy` | $\log \sum_j \exp(z_j) - z_y$ | — | Multiclass softmax classification |
| `topk_cross_entropy` | $\max_{p \in \Delta_k} \langle a, p \rangle + H(\bar p)$ | `k` | Top-*k* entropy [4] (`cross_entropy` at $k=1$) |
| `multiclass_hinge` | $\max(0, h_{[1]})$ | `gamma` | Crammer–Singer SVM [1] |
| `topk_hinge` (`variant="beta"`, the default) | $\frac{1}{k}\sum_{i \le k} \max(0, h_{[i]})$ | `k`, `variant`, `gamma` | Top-*k* SVM on each of the $k$ largest hinge terms [4] |
| `topk_hinge` (`variant="alpha"`) | $\max\big(0, \frac{1}{k}\sum_{i \le k} h_{[i]}\big)$ | `k`, `variant`, `gamma` | Top-*k* SVM on the average of the $k$ largest hinge terms [3] |
| `multilabel_hinge` | $\max\big(0, 1 + \max_{j \notin Y} z_j - \min_{y \in Y} z_y\big)$ | `gamma` | Ranked multilabel SVM [2] |
| `multilabel_cross_entropy` | $\log \sum_j \exp(z_j) - \frac{1}{\lvert Y \rvert}\sum_{y \in Y} z_y$ | — | Cross-entropy to uniform target on $Y$ |
| `proximal_cross_entropy` | $\mathrm{env}_\gamma[\mathrm{CE}(\cdot, y)]{(z)}$ | `gamma` | Moreau envelope of CE [5] |
| `distillation` | $H\big(\mathrm{softmax}(t/T), \mathrm{softmax}(z/T)\big)$ | `temperature` | Teacher-logit distillation [6] |
| `proximal_distillation` | $\mathrm{env}_\gamma[H(p, \mathrm{softmax}(\cdot))]{(z/T)}$ | `temperature`, `gamma` | Moreau-damped distillation in $z/T$ |
| `linear_cross_entropy` | $\mathrm{CE}(h W^\top + b, y)$ | `chunk_bytes`, `recompute` | Chunked linear head + CE (PyTorch) |

- **Top-*k* entropy:** Over $j \neq y$, $a_j = z_j - z_y$, $s = \sum_j p_j$ and
  $\Delta_k = \lbrace p \ge 0 : s \le 1, p_j \le s/k \rbrace$;
  $H(\bar p) = -(1 - s)\log(1 - s) - \sum_j p_j \log p_j$ is the entropy of $\bar p = (1 - s, p)$.
- **Moreau envelope (`gamma` of the proximal losses):**
  $\mathrm{env}_\gamma[f]{(x)} = \min_u \big\lbrace f(u) + \frac{1}{2\gamma}\Vert u - x \Vert_2^2 \big\rbrace$
  with the step $\gamma > 0$ (default `1` in every interface; `SDCA_LOSS_INIT(kind)` in C;
  `0` is an error), where $H(p, q) = -\sum_j p_j \log q_j$ is the
  cross-entropy and $p = \mathrm{softmax}(t/T)$.
- **Hinge smoothing (`gamma`):** `gamma = 0` (default) is the nonsmooth hinge;
  `gamma > 0` is its Moreau envelope $\max_{p \in S} \langle h, p \rangle - \frac{\gamma}{2}\Vert p \Vert_2^2$,
  which has a $(1/\gamma)$-Lipschitz gradient [4] and lies below the nonsmooth loss by at most
  $\gamma / (2k)$ for the single-label hinges ($\gamma / 2$ for `multiclass_hinge`) and
  $\gamma$ for `multilabel_hinge`.
- **Top-*k* exclusion:** Top-*k* hinges exclude the target class $y$ when selecting the $k$
  largest competitor hinge terms $h_{[1]}, \dots, h_{[k]}$ [4]. At $k = 1$, both variants of
  `topk_hinge` coincide with `multiclass_hinge`.
- **Kinks of the nonsmooth hinges (`gamma = 0`):** the gradient is the mean of the one-sided
  subgradients, as `optax.losses.multiclass_hinge_loss` and `torch.amax` give it: the
  competitors are ranked by score before computing their hinge terms, and the remaining places
  at the $k$-th largest score split evenly among the competitors tied there
  (`multilabel_hinge`: the step splits among the tied largest negatives and the tied lowest
  positives), and a hinge at exactly zero counts half (`"beta"`: each term; `"alpha"`: their
  mean). At $k = 1$ a largest hinge term of exactly zero ties with the hinge's zero, as in optax:
  each of the $t$ competitors tied there takes $1/(t+1)$. For example `multiclass_hinge` of
  $z = (0, 0.5, 0.5, -3)$, $y = 0$ has the gradient $(-1, 0.5, 0.5, 0)$, of
  $z = (1, 0, -5, -5)$ the gradient $(-0.5, 0.5, 0, 0)$, and of $z = (1, 0, 0, -5)$ the
  gradient $(-2/3, 1/3, 1/3, 0)$.

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

Public C++ headers: `sdca/solver.h`, `sdca/prox.h` and `sdca/loss.h`
(`sdca::detail` and `_`-prefixed Python names are internal).

### Names across interfaces

Every interface uses the canonical names and takes, for each loss, only its own parameters
(`k`, `variant`, `gamma`, `temperature`); a parameter of another loss is an error. `gamma`
is the smoothing of the hinges and the step of the proximal losses.

#### Python interfaces

| Canonical name | NumPy loss object | PyTorch module / PyTorch and JAX function |
| --- | --- | --- |
| `cross_entropy` | `CrossEntropy()` | `CrossEntropyLoss`, `cross_entropy` |
| `topk_cross_entropy` | `TopKCrossEntropy(k)` | `TopKCrossEntropyLoss`, `topk_cross_entropy` |
| `multiclass_hinge` | `MulticlassHinge(gamma)` | `MulticlassHingeLoss`, `multiclass_hinge_loss` |
| `topk_hinge` | `TopKHinge(k, variant, gamma)` | `TopKHingeLoss`, `topk_hinge_loss` |
| `multilabel_hinge` | `MultilabelHinge(gamma)` | `MultilabelHingeLoss`, `multilabel_hinge_loss` |
| `multilabel_cross_entropy` | `MultilabelCrossEntropy()` | `MultilabelCrossEntropyLoss`, `multilabel_cross_entropy` |
| `proximal_cross_entropy` | `ProximalCrossEntropy(gamma)` | `ProximalCrossEntropyLoss`, `proximal_cross_entropy` |
| `distillation` | `Distillation(temperature)` | `DistillationLoss`, `distillation_loss` |
| `proximal_distillation` | `ProximalDistillation(temperature, gamma)` | `ProximalDistillationLoss`, `proximal_distillation_loss` |
| `linear_cross_entropy` | — | `LinearCrossEntropyLoss`, `linear_cross_entropy` (PyTorch) |

#### Native interfaces

| Canonical name | Rust | C |
| --- | --- | --- |
| `cross_entropy` | `Loss::CrossEntropy` | `SDCA_LOSS_CROSS_ENTROPY` |
| `topk_cross_entropy` | `Loss::TopKCrossEntropy { k }` | `SDCA_LOSS_TOPK_CROSS_ENTROPY` |
| `multiclass_hinge` | `Loss::MulticlassHinge { gamma }` | `SDCA_LOSS_MULTICLASS_HINGE` |
| `topk_hinge` | `Loss::TopKHinge { k, variant, gamma }` | `SDCA_LOSS_TOPK_HINGE_ALPHA`, `SDCA_LOSS_TOPK_HINGE_BETA` |
| `multilabel_hinge` | `Loss::MultilabelHinge { gamma }` | `SDCA_LOSS_MULTILABEL_HINGE` |
| `multilabel_cross_entropy` | `Loss::MultilabelCrossEntropy` | `SDCA_LOSS_MULTILABEL_CROSS_ENTROPY` |
| `proximal_cross_entropy` | `Loss::ProximalCrossEntropy { gamma }` | `SDCA_LOSS_PROXIMAL_CROSS_ENTROPY` |
| `distillation` | `Loss::Distillation { temperature }` | `SDCA_LOSS_DISTILLATION` |
| `proximal_distillation` | `Loss::ProximalDistillation { temperature, gamma }` | `SDCA_LOSS_PROXIMAL_DISTILLATION` |
| `linear_cross_entropy` | — | — |

The C kinds, C++ `sdca::kernel::Kind`s and C++ objectives name the two variants of `topk_hinge`
`topk_hinge_alpha` and `topk_hinge_beta`. C++ objectives use these names with an `l2_` prefix
(`l2_topk_hinge_beta`; `*_objective(c, ...)` factories take `gamma`, default `0`).

### Targets

The single-label losses take class labels (0-based), the
multilabel losses label sets or a boolean mask, and the distillation losses teacher logits.

`cross_entropy` also takes probability targets: rows of an `(n, num_classes)` matrix such
as a teacher's softmax (`libsdca.Probabilities` in NumPy, `SDCA_TARGET_PROBABILITIES` in
C, a floating-point target of the logits' shape in PyTorch and JAX), with the loss
$-\sum_j t_j \log \mathrm{softmax}(z)_j$ and the gradient $T\,\mathrm{softmax}(z) - t$,
$T = \sum_j t_j$. The targets of a loss are finite and nonnegative and need not sum to
one, as `torch.nn.functional.cross_entropy` takes them; every interface also applies class
weights and label smoothing to them ([below](#loss-evaluation-options)), as PyTorch does. The shared CPU kernels compute them
in double row arithmetic. As a training objective (`Targets::Probabilities` in Rust too),
each row must be finite, nonnegative and sum to one within $\gamma_{m+1} = (m+1)u/(1-(m+1)u)$ of the storage dtype (that of the features or scores; $u$ its unit roundoff: the rounding of a row normalized in that dtype), and the library
trains on a renormalized copy. The proximal and distillation losses are not training
objectives.

### Loss evaluation options

The loss evaluations (`loss_and_grad` and its counterparts) of every interface take the
options of PyTorch's `cross_entropy`, with its definitions. The same losses take each, in
every interface; a loss that does not take an option is an error (`TypeError` in Python,
"multiclass_hinge takes no parameter label_smoothing"):

| Option | Losses | Default | Meaning |
| --- | --- | --- | --- |
| `weight` | The single-label losses | none (every weight `1`) | Class weights $w$ `(num_classes,)`, finite and `>= 0`: a row of label $y$ has $w_y$ times its loss and gradient; `cross_entropy` takes the target weights $u_j = w_j t_j$ (labels: $t$ one-hot) |
| `label_smoothing` | `cross_entropy` | `0` | $\epsilon \in [0, 1]$: the targets $(1 - \epsilon) t + \epsilon / C$ before the weights, so the loss is $(1 - \epsilon)\,\mathrm{CE} + \frac{\epsilon}{C} \sum_j (\mathrm{lse}(z) - z_j)$ for $w = 1$ |
| `ignore_index` | The single-label losses | none (PyTorch `cross_entropy`: `-100`) | Rows of this label have the loss `0` and a gradient of `+0.0`, whatever their scores and weights; with probability targets only a negative value |

- **Names:** `weight=`, `label_smoothing=`, `ignore_index=` in NumPy, PyTorch and JAX;
  `Options::with_weight`, `with_label_smoothing`, `with_ignore_index` with
  `Loss::loss_and_grad_with_options` in Rust (labels are `usize`, so the ignored label is
  too); the fields `weight`, `label_smoothing`, `ignore` and `ignore_index` of
  `struct sdca_loss` in C. Training (`fit`, `coordinate_update`) takes none of them.
- **Reductions:** the NumPy, Rust and C evaluations return the losses of the rows. The `mean`
  of PyTorch and JAX divides by the sum of the weights of the labels of the rows not
  ignored, as PyTorch does.
- **Invalid weights:** a `NaN`, infinite or negative weight is an argument error in NumPy,
  Rust and C; PyTorch and JAX, which check no tensor values, give every row not ignored a
  `NaN` loss.

## Options

The solver options below have the same names and defaults in every interface:

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

Rust sets them with `Solver` setters such as `with_tol`, C with `struct sdca_fit_options`
and C++ with `context.criteria`. The scikit-learn estimator follows scikit-learn's
conventions: it names `max_epoch` `max_iter` and `seed` `random_state`, replaces
`initial_dual` with `warm_start=True`, and keeps the defaults of `eval_on_start`,
`recompute_primal`, `num_classes` and `center`.

Kernel input is explicit:

| Interface | Kernel input |
| --- | --- |
| NumPy | `libsdca.Kernel(K)` |
| scikit-learn | `kernel="precomputed"` |
| Rust | `Input::Kernel` |
| C | `SDCA_INPUT_KERNEL` |
| C++ | `make_input_kernel` |

The binding determines how to pass the execution settings `precision` ([below](#precision)) and
`num_threads` ([threads](#threads)).

- **Evaluations:** besides every `eval_epoch` epochs, a fit with `tol > 0` may evaluate up
  to twice more when the gap trend projects crossing `tol` early. A fit stopped by
  `max_epoch` reports `solved` when its last evaluation meets `tol`.
- **`recompute_primal`:** feature fits rebuild $W = W_0 + X^\top A$ before evaluating the
  primal, so update drift does not enter the gap (not for `float32` data with
  `precision="float64"`, whose `double` $W$ is evaluated as stored).
- **Kernel validation:** every interface requires a nonnegative diagonal, symmetry within
  $10^{-5}\sqrt{K_{ii}K_{jj}}$, and $\lvert K_{ij} \rvert \le (1 + 10^{-3})\sqrt{K_{ii}K_{jj}}$
  (C++: `criteria.kernel_symmetry` and `criteria.kernel_cauchy_schwarz`).
- **Feature scaling:** Because each coordinate step scales with $1/(C\Vert x_i \Vert_2^2)$,
  standardize or normalize features when example norms span orders of magnitude.

### Shrinking

`shrinking` accelerates hinge fits (`multiclass_hinge`, `topk_hinge`, `multilabel_hinge`)
by skipping dual variables that sit at zero
bounds (entropy objectives accept only `"auto"`):

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

## Precision

Storage and working precision are separate. `precision=None` (default) follows the storage
dtype, with 16-bit storage computed in `float32`; `"float32"` or `"float64"` selects the
working precision explicitly:

| Storage dtype | `None` (default), `"float32"` | `"float64"` |
| --- | --- | --- |
| `float16`, `bfloat16` | `float32` work (NumPy casts 16-bit arrays to `float32`; C takes `float32` or `float64`) | `float64` work (JAX: see [backends](JAX.md#backends)) |
| `float32` | `float32` work | `float64` work |
| `float64` | `float64` work (`"float32"`: NumPy casts to `float32` and returns `float32` arrays; PyTorch and JAX raise, since gradients keep the logits' dtype) | `float64` work |

In C the storage type is the
`dtype` argument and the work is `struct sdca_execution`'s `precision`
(`SDCA_PRECISION_DEFAULT`, `SDCA_PRECISION_FLOAT32` or `SDCA_PRECISION_FLOAT64`).

- **Accuracy contract:** `float32` work bounds the accuracy of the results; individual
  operations may run in `float64` (row statistics of the entropy losses, and rows or
  parameters that `float32` cannot represent).
- **Outputs:** Weights, duals and gradients keep the storage dtype. Loss values are `double`
  in the native APIs and `promote(logits.dtype, float32)` in PyTorch and JAX, or `float64`
  for `float64` class probabilities, whose gradient is then `float64` too (the logits'
  gradient keeps the logits' dtype). A `mean` of finite rows whose totals leave the range
  uses power-of-two scaling to recover a representable result. An overflowing sum of class
  weights on CUDA can still affect the divisor ([known limits](#known-limits)).
- **`float32` fits:** Keeping $W$ and the duals in `float32` can stall near the optimum
  (updates below half an ulp are lost). `precision="float64"` (`<float, double>` in C++) keeps
  `double` iterates for tighter convergence at `8(d + n)m` extra bytes. Both modes check the
  gap of the returned arrays.
- **Compiler flags:** Build with IEEE semantics (`-fno-fast-math -ffp-contract=off`);
  `-ffast-math` or `-Ofast` breaks the nonfinite-input checks and compensated sums.

### Parameter domains and numerical limits

All loss and training entry points require:

- `classes >= 2` and integer `1 <= k < classes`;
- finite scores/features (the losses also take scores of `-inf`, as
  [masked classes](#masked-classes));
- valid targets:
  - class indices are 0-based;
  - boolean masks require at least one positive label, plus one negative label for
    `multilabel_hinge`;
  - label sets in `fit` and `coordinate_update` require a negative label for every
    objective.

| Parameter | Valid domain | Notes |
| --- | --- | --- |
| `C`, `temperature`, `gamma` of the proximal losses | finite, strictly positive | `C`, `norm2` and `C * norm2` must be normal in the storage dtype, and `C` in working precision; a `gamma` or `temperature` outside normal `float32` promotes to `float64` |
| `gamma` of the hinges | finite, `>= 0` | `0` selects the nonsmooth hinge; positive values follow the rules of the proximal `gamma` |
| `tol` | finite, `>= 0` | `0` disables the `solved` stop: the fit ends with `max_epoch`, `no_progress` or `failed` |
| `ignore_index` | 64-bit integer (Rust: `usize`) | Matching targets are excluded from the loss and gradient ([options](#loss-evaluation-options)) |
| `weight` | finite, `>= 0`, shape `(C,)` | PyTorch: matches logits device and dtype (`float32` accepted for 16-bit logits under autocast) |
| `label_smoothing` | `[0, 1]` | `cross_entropy` only |

See [known limits](#known-limits) for accuracy bounds.

### Masked classes

A score of `-inf` masks its class. The loss of a row with masked classes is the limit of the
loss as those scores go to `-inf`, by one rule for every loss and interface: on the CPU,
in the CUDA kernels of PyTorch and JAX, and with `backend="composed"` (the CUDA kernels give
the infinite limits of the table's fourth row only for the masked labels of the
cross-entropy losses; their other such rows are still `NaN`):

| Scores of a row | Loss and gradient |
| --- | --- |
| `-inf` at a class that is not a label (a negative class, a class the teacher masks too) | The class is removed: its gradient entry is exactly `+0.0`, and the loss and the other entries are those of the loss on the remaining classes. |
| `-inf` at every class but the label | Loss `0` and a zero gradient. |
| Fewer than `k` other classes left (top-*k* losses) | The limit: `0` for `topk_cross_entropy` and `topk_hinge` of variant `"alpha"`; for variant `"beta"` with `r` other classes left, `r / k` times the loss of `k = r` and `gamma r / k`. |
| `-inf` at the label, at a positive class, or in the student where the teacher has probability, with another class left | Loss `+inf` and the limit of the gradient as those scores go to `-inf` ([below](#limits-of-the-gradient)), times the class weight and the upstream scale. A `sum` or `mean` over such a row is `+inf`. |
| `-inf` in every class (or a teacher row of `-inf` only); `NaN`; `+inf`; a zero class weight times an infinite loss | Invalid, as in PyTorch ([errors](#errors)). |

#### Limits of the gradient

The limit removes the other masked classes first; several masked labels, positive classes or
student classes go to `-inf` together (equal scores, so ties split as each loss splits them).
`r` is the number of classes left beside the label.

| Loss | Gradient of a row with a masked label, positive class or student class |
| --- | --- |
| `cross_entropy`, `linear_cross_entropy`, `topk_cross_entropy` (`k = 1`) | `-1` at the label and the softmax of the remaining classes elsewhere, as `torch.nn.functional.cross_entropy`. |
| `topk_cross_entropy` (`k > 1`) | `-1` at the label and the capped softmax `min(1 / k, exp(z_j - tau))` of the remaining classes (summing to one): the maximizer of the top-*k* entropy, whose mass goes to one. With `r < k` the loss is `0` with a zero gradient, as without the mask (the top-*k* simplex is `{0}`). |
| `multiclass_hinge`, `topk_hinge` (`gamma = 0`) | `1 / k` at the `k` largest remaining scores (the `k`-th place split evenly among its ties) and `-1` at the label. Variant `"alpha"` with `r < k`: loss `0`, zero gradient; variant `"beta"` with `r < k`: `1 / k` at each remaining class and `-r / k` at the label. |
| `multiclass_hinge`, `topk_hinge` (`gamma > 0`) | `a / gamma` at the remaining classes, `a` the Euclidean projection of their scores onto `{0 <= a_j <= gamma / k, sum a = gamma}` (both variants; the simplex at `k = 1`), and `-1` at the label; `r < k` as for `gamma = 0`. |
| `multilabel_hinge` | `gamma = 0`: `1` split evenly among the largest remaining negatives and `-1` among the masked positives. `gamma > 0`: `y / gamma` at the remaining negatives, `y` the projection of their scores onto `{y >= 0, sum y = gamma}`, and `-1 / m` at each of the `m` masked positives. Without a negative left the loss is `0`. |
| `multilabel_cross_entropy` | The softmax of the remaining classes minus `1 / n` at each of the `n` positives (`-1 / n` at a masked one), as soft-target `cross_entropy` with the uniform target on the positives. |
| `distillation` | `(q - p) / T` with `q = softmax(z / T)` on the classes the student keeps: `-p_j / T` where it masks a class. |
| `proximal_cross_entropy` | `-1` at the label and `pi` at the remaining classes: `gamma pi_j + log pi_j = z_j - t` with `sum pi = 1` (the proximal step on them toward no target). |
| `proximal_distillation` | `(pi - p) / T` at the classes the student keeps, `gamma pi_j + log pi_j = z_j / T + gamma p_j - t` with `sum pi = 1` over them, and `-p_j / T` where it masks a class. The envelope of the infinite loss is `+inf`; its gradient has this limit. |

Differences from PyTorch and JAX, and between them:

- PyTorch's `multi_margin_loss`, which averages the hinges of every class, gives a masked
  label the loss `+inf` and the gradient `1 / C` per other class (`-(C - 1) / C` at the
  label). The multiclass hinge here takes the largest hinge term, as
  `optax.losses.multiclass_hinge_loss` does, so its limit puts the whole step on the largest
  remaining score.
- Kinks: the nonsmooth hinges split the step evenly among ties and halve it at a zero hinge term,
  as `optax` and autograd through `torch.amax`, `torch.max()` or `jnp.max` do;
  `torch.max(dim)`, `torch.topk` and `jax.lax.top_k` would give it to the lowest index.
- A zero class weight times an infinite loss gives the loss `NaN`, as PyTorch, and an invalid
  row with a `NaN` gradient, where PyTorch's gradient is zero.
- A masked class of target `0` is removed (below); PyTorch and `optax` give `NaN` there.

- **Distillation:** A class the teacher masks has teacher probability `0`; a class both
  mask is removed. A class the student masks where the teacher has probability gives `+inf`
  (above); a teacher row of `-inf` only is invalid.
- **Weights and ignored rows:** A zero class weight times an infinite loss is invalid. Rows
  of an ignored label are excluded whatever they hold.
- **Probability targets:** A masked class of target `0` (after class weights and label
  smoothing) is removed, with a gradient of `+0.0`. One of a positive target gives the loss
  `+inf` and the gradient `-t` there and `T` times the softmax of the remaining classes
  elsewhere, as PyTorch, optax and a masked label do. Smoothing puts target probability on
  every class, so a masked class then gives `+inf`
  ([TORCH.md](TORCH.md#masked-classes), [JAX.md](JAX.md#masked-classes)).
- **Finite rows** give the same results as without masking.
- **Signed zeros:** Every zero gradient entry of the work arithmetic, in every loss (masked
  classes, zero rows such as hinges whose margins are met, and entries that underflow in the
  work dtype), is `+0.0` for either sign of the upstream derivative, as in PyTorch and JAX. A
  negative entry that rounds to zero only in a 16-bit storage dtype stays `-0.0`.
- **Scope:** `loss_and_grad` and its counterparts in every interface. Training
  (`fit`, `coordinate_update`) and the projections require finite inputs. The notes of
  each interface say which of its paths compute masked classes:
  [PyTorch](TORCH.md#masked-classes), [JAX](JAX.md#masked-classes),
  [NumPy](PYTHON.md#losses-and-gradients), [C](C.md#conventions),
  [C++](CPP.md#loss-values-and-gradients) and [Rust](RUST.md#losses-and-gradients).

## Results

Training returns a status and an evaluation `history` with the same fields in every
interface:

- **Status:** `solved`, `no_progress`, `max_epoch` or `failed`.
- **History fields:** `epoch`, `primal`, `dual`, `primal_loss`, `dual_loss`, `regularizer`,
  `relative_gap`, `accuracy` (the rank loss for the multilabel losses), `cpu_time` and
  `wall_time`. Rust exposes history rows as `Evaluation` structs.

Only `status == "solved"` guarantees convergence:

$$P(W) - D(A) + e \le \text{tol} \cdot \max(|P(W)|, |D(A)|),$$

where $P(W)$ is evaluated in `double` at the returned weights, $D(A)$ at the feasible
projection of the returned duals, and $e$ is a first-order estimate of evaluation rounding.

- **Failures:** If a numerical step fails or the evaluated gap falls below $-e$, training
  stops with `failed` at the last consistent iterate.
- **Regularizer center:** With a center $W_0$, $P(W)$ and $D(A)$ evaluate the full centered
  objective directly from $W - W_0$ without cancelling $\frac{1}{2}\Vert W_0 \Vert_2^2$.

### Array layouts

For $n$ examples, $d$ features and $m$ classes, NumPy, Rust, C and C++ share one layout.
Every array is dense and row-major (C order):

| Array | Shape | Contiguous in memory |
| --- | --- | --- |
| Features | `(n, d)` | The `d` features of an example |
| Weights, regularizer center | `(m, d)` | The weight vector of a class |
| Duals, scores | `(n, m)` | The `m` values of an example |
| Gram matrix | `(n, n)` | The kernel values of an example |

- **No copies:** The memory is the same in every interface. A binding hands its arrays to
  the native library as they are; none is copied, transposed or reordered on the way in
  or out. NumPy converts only an input of another dtype or one that is not C-contiguous.
- **Weights:** Row `c` is the weight vector of class `c`, the transpose of the $W$ of the
  objective. Scores are `X @ weights.T`, and `weights = center + dual.T @ X`.
- **Kernel fits** have no weight matrix; their scores are `K @ dual` for the kernel `K`
  `(n_test, n_train)` between test and training examples.

## Distillation losses

`distillation_loss` and `proximal_distillation_loss` take student and teacher **logits** of the
same shape; PyTorch and JAX promote differing floating-point dtypes, while native
interfaces require matching storage types. In PyTorch and JAX a teacher that is
differentiated gets the gradient of $H(p, q)$, $p_j(-\log q_j - H(p, q))/T$, as PyTorch's
autograd of `cross_entropy(z / T, softmax(t / T))`; for `proximal_distillation_loss` the same
at the envelope's optimum, $q = \pi$ (the inner problem is strongly convex, so its optimum
moves smoothly with $t$). The loss pays for it only when the teacher is differentiated.

With $p = \mathrm{softmax}(t/T)$ and
$q = \mathrm{softmax}(z/T)$, the student gradient is $(q - p)/T$ (or $(\pi - p)/T$ at the
Moreau-envelope optimum $\pi = \mathrm{softmax}(u^\star)$ for `proximal_distillation_loss`),
with no $T^2$ multiplier. The gradient is formed from the log-ratio of the student's and the
teacher's probabilities, so identical logits produce an **exact zero gradient** in every
dtype, and the gradient shrinks with their difference instead of stopping at a rounding
floor ([known limits](#known-limits)).

```python
import torch
from libsdca.torch import DistillationLoss, ProximalDistillationLoss

student = torch.randn(8, 10, requires_grad=True)
teacher = torch.randn(8, 10)
loss = DistillationLoss(temperature=4.0)(student, teacher)
prox = ProximalDistillationLoss(temperature=4.0, gamma=0.5)(student, teacher)
(loss + prox).backward()
```

- **PyTorch and JAX:** Both run on the CPU kernels and the fused Triton kernels on CUDA,
  with first derivatives only. With `backend="composed"`, `distillation_loss` differentiates
  at any order and in forward mode; `proximal_distillation_loss` stays first order.

## Projections and proximal operators

The coordinate steps of the solver and the hinge and top-*k* losses reduce to small convex
problems: projections onto simplex-like sets and their entropic counterparts. C++
(`sdca/prox.h`) exposes these operators directly; the other interfaces use them internally.

Each operator maps a vector $a$ to the minimizer $x$ of its problem. With
$q(x) = \frac12\langle x, x \rangle - \langle a, x \rangle$ (the Euclidean projection of $a$)
and $\sigma = \langle 1, x \rangle$:

| Operator | Minimizes | Subject to | Parameters |
| --- | --- | --- | --- |
| `knapsack_eq` | $q(x)$ | $\sigma = r$, $l \le x_i \le h$ | `lo`, `hi`, `rhs` |
| `knapsack_le` | $q(x)$ | $\sigma \le r$, $l \le x_i \le h$ | `lo`, `hi`, `rhs` |
| `knapsack_le_biased` | $q(x) + \frac{\rho}{2}\sigma^2$ | $\sigma \le r$, $l \le x_i \le h$ | `lo`, `hi`, `rhs`, `rho` |
| `topk_simplex` | $q(x)$ | $\sigma \le r$, $0 \le x_i \le \sigma / k$ | `k`, `rhs` |
| `topk_simplex_biased` | $q(x) + \frac{\rho}{2}\sigma^2$ | $\sigma \le r$, $0 \le x_i \le \sigma / k$ | `k`, `rhs`, `rho` |
| `topk_cone` | $q(x)$ | $0 \le x_i \le \sigma / k$ | `k` |
| `topk_cone_biased` | $q(x) + \frac{\rho}{2}\sigma^2$ | $0 \le x_i \le \sigma / k$ | `k`, `rho` |
| `entropy` | $\langle x, \log x \rangle - \langle a, x \rangle$ | $\sigma = r$, $0 \le x_i \le h$ | `hi`, `rhs` |
| `entropy_norm` | $\frac12\langle x, x \rangle + \langle x, \log x \rangle - \langle a, x \rangle$ | $\sigma = r$, $0 \le x_i \le h$ | `hi`, `rhs` |
| `topk_entropy` | $\langle x, \log x \rangle + (1 - \sigma)\log(1 - \sigma) - \langle a, x \rangle$ | $\sigma \le 1$, $0 \le x_i \le \sigma / k$ | `k` |
| `topk_entropy_biased` | $\frac{\alpha}{2}(\langle x, x \rangle + \sigma^2) + \langle x, \log x \rangle + (1 - \sigma)\log(1 - \sigma) - \langle a, x \rangle$ | $\sigma \le 1$, $0 \le x_i \le \sigma / k$ | `k`, `alpha` |

Three operators act on a vector split into two blocks, $a$ of length $p$ and $b$, with
minimizers $x$ and $y$:

| Operator | Minimizes | Subject to | Parameters |
| --- | --- | --- | --- |
| `two_simplex` | $\Vert x - a \Vert^2 + \Vert y - b \Vert^2$ | $\langle 1, x \rangle = \langle 1, y \rangle \le r$, $x, y \ge 0$ | `rhs` |
| `two_entropy` | $\frac{\alpha}{2}\Vert x - a/\alpha - 1/p \Vert^2 + \langle x, \log x \rangle + \frac{\alpha}{2}\Vert y - b/\alpha \Vert^2 + \langle y, \log y \rangle$ | $\langle 1, x \rangle + \langle 1, y \rangle = 1$, $x, y \ge 0$ | `alpha` |

- **Parameters and defaults:** `lo` $l = 0$, `hi` $h = 1$, `rhs` $r = 1$, `rho` $\rho = 1$,
  `alpha` $\alpha = 1$ and `k` $= 1$. With the defaults `knapsack_eq` is the projection onto
  the unit simplex.
- **Two-simplex accuracy:** `two_simplex` (and `multilabel_hinge`, built on it) solves in
  `double` and, at near-ties or extreme scales, again in `long double`.
- **Lambert function:** the entropic operators with a quadratic term evaluate $W(e^x)$, the
  solution $w$ of $w + \log w = x$, available as `sdca::lambert_w_exp`.
- **Calling them:** [CPP.md](CPP.md#projections-and-proximal-operators). The C ABI, NumPy,
  PyTorch, JAX and Rust have no projection entry point.

## BLAS

Every build selects the BLAS by one rule, from `src/native_build.txt`: `pip install`
(`setup.py`), `python/build.py`, the Rust crate and CMake, so one machine gives one backend
in every interface.

- **Selection:** `SDCA_BLAS_BACKEND=AUTO` (the default) links the first of
  `-framework Accelerate`, `-lopenblas`, `-lcblas` and `-lblas` (then the runtime names
  `libopenblas.so.0`, `libcblas.so.3` and `libblas.so.3`) against which a probe of the CBLAS
  calls links. `SDCA_BLAS` replaces the list with its own linker flags, `EXTERNAL` fails
  without a match, and `BUILTIN` takes the built-in products.
- **Wheels:** macOS wheels link Accelerate. Linux wheels link the OpenBLAS of the
  `scipy-openblas32` package (`SDCA_BLAS_BACKEND=PACKAGE`, Python builds only), which they
  require (about 8 MB), as NumPy and SciPy build against it. Its symbols carry the prefix
  `scipy_` (the macro `SDCA_CBLAS_PREFIX`); the NumPy binding loads its library from the
  installed package before libsdca's, without exporting its symbols to other modules.
- **Threads of the Linux wheels:** NumPy and libsdca each load an OpenBLAS (SciPy a third);
  each reads `OPENBLAS_NUM_THREADS` (else `OMP_NUM_THREADS`) and `OPENBLAS_THREAD_TIMEOUT`,
  and `threadpoolctl.threadpool_limits` sets them all. If alternating products of two copies
  slow down (busy-waiting threads), use fewer threads or a shorter timeout.
- **No BLAS found:** `AUTO` then uses the built-in products and warns: in the build (a
  Python build warning, a Cargo warning, a CMake warning), once per process before the first
  NumPy `fit` (a `RuntimeWarning`), and once in the C++ solver's log (at `level::warning`).
  `sdca_blas_backend()` reports `"external"` (`"external: OpenBLAS (scipy-openblas32)"` in
  the Linux wheels), `"builtin"` or `"fallback"`.
- **What it computes:** The solver's `scal` (accuracy vectors) and `dot` (dual losses of the
  smoothed hinges), and the scores of predictions: one `gemm` in `sdca_model_scores`, Rust's
  `decision_function` and C++'s `batch_scores` (NumPy's `decision_function` is NumPy's
  matrix product).
- **Reproducibility:** With the built-in products every result is the same bits for every
  thread count and instruction set. An external BLAS rounds its dot products and `gemm` as
  it will, so smoothed-hinge dual values (and with them the reported gap and, with `tol > 0`,
  the stopping epoch) and prediction scores can differ in the last bits between BLAS
  libraries, their versions and their thread settings (such as `OPENBLAS_NUM_THREADS`).

## Threads

Loss evaluations and coordinate steps process rows independently across up to `num_threads`
worker threads with **bitwise-identical results** for every thread count (the products of an
external BLAS round as it does; see [BLAS](#blas)):

- **Default selection:** When `num_threads` is omitted (`None` in Python, `Threads::FromEnv`
  in Rust, `0` in C), the library reads `SDCA_NUM_THREADS` once and
  otherwise uses automatic threading, capped by the active
  OpenMP team (`torch.set_num_threads`) or `OMP_NUM_THREADS` and divided across concurrent
  calls. Passing `1` runs on the calling thread.
- **Work split:** small calls stay on the calling thread; larger ones take threads in
  proportion to their estimated work. The split depends only on the call, the thread setting
  and the number of concurrent calls, and results are the same for every split.
- **Training (`fit`):** Coordinate-ascent epochs are sequential; only duality-gap evaluations,
  gap-safe screening passes and kernel validation run in parallel.
- **Floating-point environment:** Every C ABI call executes with IEEE subnormals enabled and
  round-to-nearest (even if the caller set flush-to-zero, as XLA:CPU does) and restores the
  caller's environment on return.

## Environment variables

The first-use builds of the CPU kernels (`SDCA_CPU_EXTENSION` for PyTorch, `SDCA_JAX_FFI`
for JAX) take `auto`, `on` (`1`) or `off` (`0`):

- **`auto`** (default) compiles and caches on first use.
- **`on`** also retries a build whose failure was recorded.
- **`off`** never builds or loads them (CPU calls take the native library, silently).
- **Failed builds:** A compiler error leaves `*.build-failed.txt` in the cache, which
  suppresses retries for 24 hours (transient failures are not recorded); delete it or set
  the switch to `on` to retry.
- **Without the CPU kernels** (switched off, no compiler or Python headers, a failed build),
  CPU calls with `backend="auto"` or `"kernel"` run the same kernels in the native library
  (`libsdca_c`, through `sdca_loss_rows`), after one `RuntimeWarning` unless switched off
  ([PyTorch](TORCH.md#build-tools), [JAX](JAX.md#build-tools)); only where that does not
  load either do they raise `NotImplementedError` naming `backend="composed"`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SDCA_NUM_THREADS` | unset (auto) | Default thread limit (`< 4096`; `0` = auto, `1` = serial) |
| `SDCA_CPU_CAPABILITY` | unset | `default \| avx2 \| avx512`: lowers the instruction set of the CPU loss kernels on x86-64 (never above the CPU's); unset, the CPU decides |
| `SDCA_CPU_EXTENSION` | `auto` | `auto \| on \| off`: build the PyTorch C++ extension for CPU kernels and direct CUDA launches. With `off`, CPU uses the native library and CUDA launches through Python |
| `SDCA_JAX_FFI` | `auto` | `auto \| on \| off`: first-use build of the JAX CPU kernels (an XLA FFI handler); without them, the native library |
| `SDCA_JAX_CUDA_FFI` | `auto` | `auto \| on \| off`: first-use build of the CUDA XLA FFI launcher when jax-triton 0.5 is unavailable |
| `SDCA_CACHE_DIR` | `$XDG_CACHE_HOME/libsdca` (`~/.cache/libsdca`) | Cache directory for the compiled CPU kernels and the JAX CUDA FFI launcher |
| `OMP_NUM_THREADS` | unset | Caps automatic threads when no OpenMP runtime is loaded |
| `SDCA_LIBRARY_DIR` | unset | Override directory from which Python loads `libsdca_c.so` |
| `SDCA_BLAS_BACKEND` | `AUTO` | Build-time BLAS selection (`AUTO`, `EXTERNAL`, `BUILTIN`; [BLAS](#blas)) for `pip install`, `python/build.py`, `rust/build.rs` and CMake |
| `SDCA_BLAS` | unset | Explicit external BLAS linker flags (e.g. `-lopenblas` or `-L/opt/blas/lib -lblas`), in place of the candidates |
| `SDCA_BUILD_NATIVE` | `ON` | Set to `OFF` during `pip install` to build a pure-Python wheel for tensor losses only |
| `CXX`, `AR` | system default | Compiler and archiver for native and first-use builds |

CMake options (`cmake -D<option>=<value>`):

| Option | Default |
| --- | --- |
| `SDCA_BLAS_BACKEND` | `AUTO` (`AUTO\|EXTERNAL\|BUILTIN`) |
| `SDCA_BLAS` | `$ENV{SDCA_BLAS}` (linker flags of an external CBLAS) |
| `SDCA_BUILD_C_API`, `SDCA_WARNINGS_AS_ERRORS`, `BUILD_TESTING` | `OFF` |
| `USE_CHRONO` | `ON` |

## Errors

| Interface | Invalid input or option | Invalid rows of a loss evaluation | Numerical failure |
| --- | --- | --- | --- |
| NumPy, scikit-learn | `ValueError` (or `TypeError`) | `nan` rows and one `InvalidRowsWarning` per call | `RuntimeError`; inspect the status of a returned fit |
| PyTorch | `TypeError` / `ValueError` for types, shapes and options | `nan` rows, silently | `RuntimeError` or `nan`, depending on the backend |
| JAX | `TypeError` / `ValueError` for types, shapes and options | `nan` rows, silently | `nan` rows; see [JAX errors](JAX.md#errors) |
| Rust | `Error::InvalidArgument` | `nan` rows and a `LossReport` with their count | `Error::Native` |
| C | `SDCA_INVALID_ARGUMENT` | `nan` rows and `SDCA_INVALID_ROWS` | `SDCA_FAILURE` (message in `sdca_last_error()`) |
| C++ | `std::invalid_argument` | `std::invalid_argument` (one row per call) | `std::runtime_error` |

PyTorch and JAX also raise `NotImplementedError` naming `backend="composed"` where
`backend="auto"` or `"kernel"` has no kernel for a call.

**Invalid rows.** Arguments, options and targets are checked before anything is written. A
label out of range (neither in `[0, classes)` nor the `ignore_index`), in a label array or
a label set, is an invalid argument, as in PyTorch: `Target 7 is out of bounds.` (the
`ValueError`, `Error::InvalidArgument`, `SDCA_INVALID_ARGUMENT` or `std::invalid_argument`
of the table). The PyTorch and JAX CPU kernels and eager composed CPU calls report the same
text (synchronous calls raise `IndexError`). CUDA and traced composed calls return NaN rows. A row of a loss evaluation that cannot be computed (`NaN` or `+inf` scores,
an invalid row of [masked classes](#masked-classes), an empty label set) gets a `nan` value
and `nan` gradients; the other rows hold their results. Probability targets of a loss
evaluation are taken as given, as by PyTorch's `cross_entropy` (training requires
distributions). The C ABI's `sdca_loss_evaluate` then returns
`SDCA_INVALID_ROWS` (a nonnegative code, every output written) with the count and the first
invalid row in `sdca_last_error()` and the optional `sdca_loss_report`. `check_rows`
(`sdca_execution.check_rows = 1` in C, `check_rows=True` in NumPy,
`Settings::with_check_rows` in Rust) checks every row before anything is written and makes
an invalid row an invalid-argument error instead, at the cost of one more pass over the
scores ([C conventions](C.md#conventions)).

## Known limits

Each entry gives the condition, its effect and, where one exists, the remedy.

- **Representation:** cross-entropy losses and gradients reach the output dtype's subnormal
  range (margins up to about `104` in `float32` and `745` in `float64`), the proximal losses'
  too. Beyond that:
  - a proximal row whose competitors' terms each underflow the working precision (though
    their sum would not) gets `0` for them;
  - in `float64`, `multilabel_cross_entropy`'s positive entries of rows with terms beyond
    `exp(-708)` err by up to $C/2$ units of the least subnormal.
- **Composed proximal distillation:** the loss value of a confident row can lose several units
  of the least subnormal.
- **JAX on the CPU:** the kernels keep subnormals, but XLA's own operations flush them. An
  XLA reduction of subnormal row losses, and the results of `backend="composed"`, are `0`
  below about `exp(-87.3)` (`float32`) or `exp(-708.4)` (`float64`); a flushed negative entry
  is `-0.0`.
- **JAX label smoothing on CUDA:** the CUDA kernels round `label_smoothing` to the logits'
  dtype, so `float32` or 16-bit logits with `precision="float64"` can differ from the CPU
  result by a few `float32` ulps.
- **Rows beyond the range:** a `sum` of rows that each underflow to `0` is `0`, and a row whose
  own loss exceeds `DBL_MAX` makes a `mean` `+inf`. If the sum of label weights overflows on
  CUDA, the weighted mean can be zero or `nan`.
- **PyTorch training memory on the CPU:** the eager forward of a sum or mean that needs a
  gradient computes it with the values, so the step holds the logits and their gradient
  (twice the logits). Other upstream gradients, `reduction="none"` and CUDA compute it in
  backward, over the logits where `reuse_logits` allows ([TORCH.md](TORCH.md#memory)).
- **`float32` work:** accuracy is about `64` `float32` eps of the value and of the row's
  largest gradient entry. 16-bit inputs add `100` eps of $\max(|\text{loss}|, 1)$ to values,
  and one 16-bit unit plus `2,000` eps of the row's largest entry to gradients. Remedy:
  `precision="float64"`.
- **Tiny probability targets:** `float32` work rounds `float64` targets to `float32`, so a row
  whose target mass is below about $C \cdot 1.2 \cdot 10^{-38}$ loses precision (loss `0` when
  every target underflows). Remedy: `float64` work.
- **Distillation:** in `float32` and `float64` work, each gradient entry $g_j$ is within `64`
  eps of $|g_j| + (p_j + q_j)/T \cdot \min(1, \Delta/T)$, where
  $\Delta = \max_j |(z_j - \max z) - (t_j - \max t)|$. Entries are exactly `0` where student and
  teacher logits agree. An entry whose own probabilities nearly agree while other classes
  differ is accurate to eps of the second term, not of itself.
- **Row sums:** losses invariant to adding a constant to a row return gradient rows that sum
  to zero up to rounding:
  $|\sum_j g_j| \le c\,(\epsilon_{out}\lVert g\rVert_1 + 2\epsilon_{work}T) + n\eta$.
  Here $\epsilon_{out}$ and $\eta$ are the eps and smallest subnormal of the output dtype
  ($2\eta$ per entry on CUDA; JAX and Optax add $n$ times the smallest normal), $\epsilon_{work}$
  the eps of the work, $n$ the classes, and $T$ the target mass where the work rounds the targets
  (probability targets, label smoothing, `multilabel_cross_entropy`; $0$ for labels).
  - Observed $c$: about `1` on the CPU in `float32` work (distillation `1.8`) and `11` in
    `float64`; on CUDA about `16` in `float32` and `128` in `float64`.
  - Example: `float32` logits `[0, 1]` with their rounded softmax as targets give the gradient
    `[0, 5.98e-9]`.
- **Cancellation:** `float64` gradient entries that are differences of nearly equal
  probabilities (distillation where student and teacher nearly agree) are accurate to eps of
  the probabilities, not of the difference.
- **Extreme spreads:** finite scores more than `finfo.max` apart overflow the margins.
  - A label (or every positive class) that far above the others gives loss `0` and a zero
    gradient.
  - A competitor that far above the label gives a zero nonsmooth hinge gradient and `nan` for
    the smoothed hinges, `proximal_cross_entropy` and `multilabel_cross_entropy`
    (`cross_entropy` gives `+inf`); distillation paths other than native CPU can give `nan`.
  - Cancelling margins (`[max, max/2, -max/2, 0]`, label second, `float32`) make the top-2
    alpha hinge `0` instead of `1`.
- **Sums near overflow:** losses that sum scores over the $C$ classes of a row can fail once the
  scores span more than `finfo.max / C`:
  - `topk_cross_entropy` with `k >= 3`: `nan` on the CPU where the loss exceeds `finfo.max`;
    on CUDA `0` with a zero gradient, or `nan`;
  - smoothed `multilabel_hinge` and `topk_hinge` (`k >= 2`): wrong losses and gradients on
    macOS `arm64`;
  - `multilabel_cross_entropy` and `proximal_distillation`: `nan`.

  `cross_entropy`, `distillation`, the nonsmooth hinges and the smoothed `multiclass_hinge`
  are not affected.
- **Top-*k* cross-entropy at extreme spreads:** with `k >= 2`, the CUDA kernels can return
  `nan` for rows whose finite scores span more than $2^{63}$ (about `9.2e18`), also with the
  label on top. Rows spanning many orders lose digits: `backend="composed"` from about
  $10^{16}$, and on CUDA a capped row as its log-odds round to `float64`. The CPU kernels
  keep their digits short of the sums near overflow.
- **Tied top scores with probability targets:** `cross_entropy` with probability targets
  can lose accuracy to cancellation on rows whose largest scores are tied.
- **Proximal losses in 16-bit storage:** with `float16` or `bfloat16` logits, the proximal
  losses lose competitor terms on very confident rows.
- **Smoothed top-*k* hinges on masked rows:** with masked classes, the smoothed `topk_hinge`
  can return wrong results on rows with scores from about $10^{16}$.
- **`backend="composed"`:** uses the same collapse-free formulas as the kernels, computed
  with the framework's tensor operations in the working precision, so it can lose accuracy
  on confident or very wide rows, near the subnormal range and at spreads near `finfo.max`.
  Some x86 CPUs flush `bfloat16` subnormal inputs of the matrix product in composed PyTorch
  `linear_cross_entropy`, which can zero a weight gradient.
- **Gradients near the subnormal range:** an entry in the work type's subnormal range is
  accurate to a few subnormal steps, and a large upstream factor scales that absolute error.
  On CUDA with `float32` work, entries up to $2^{24}$ times `float32`'s smallest normal lose
  bits.
- **Class weights on confident probability-target rows:** with `float32` work, competitor
  terms more than about `132` below the row's maximum (CPU; about `147` on CUDA rows of up to
  257 classes, and gradient entries beyond about `103.3`) are `0` before the class weight
  scales them. This matters only for weights above about $2^{30}$. Remedy: label targets or
  `precision="float64"`.
- **Solver:** tolerances below about `1e-12` in `float64` cannot be guaranteed (the fit ends
  with `max_epoch` or `no_progress`). Large regularizer centers $W_0$ limit the updates that
  can be represented. Where `long double` is 64-bit (macOS `arm64`, MSVC), the
  extended-precision two-simplex solve equals the `double` one.
- **External BLAS:** results depend in the last bits on the BLAS and its threads
  ([BLAS](#blas)); the built-in products are the same across thread counts and instruction
  sets.

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
