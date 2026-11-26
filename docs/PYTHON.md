# libsdca for Python

[All guides](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/README.md) ·
[Shared API reference](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md)

An accurate and fast implementation of classification losses: cross-entropy, multiclass and
multilabel SVM hinges, top-*k* losses and knowledge distillation, as PyTorch and JAX loss
layers, NumPy functions, and SDCA solvers for linear and kernel classifiers with a
scikit-learn estimator ([project page](https://mlapin.github.io/libsdca/),
[technical report](https://mlapin.github.io/libsdca/report/)).

Prebuilt wheels for Linux `x86_64` and macOS `arm64` need no compiler for the NumPy
interface; they require NumPy (and on Linux `scipy-openblas32`, which `pip` installs).
PyTorch, scikit-learn and JAX are optional.

## Installation

```bash
pip install libsdca              # NumPy interface
pip install 'libsdca[sklearn]'   # scikit-learn estimator SDCAClassifier
pip install 'libsdca[torch]'     # PyTorch loss layers
pip install 'libsdca[jax]'       # JAX CPU loss functions
pip install 'libsdca[jax-cuda]'  # JAX CUDA losses (Linux, NVIDIA driver 580+)
```

| Task | Entry point |
| --- | --- |
| Train a scikit-learn classifier | `libsdca.sklearn.SDCAClassifier` |
| Train with full control (warm starts, label sets, target probabilities, history) | `libsdca.fit` → `Model` |
| Evaluate losses and gradients on NumPy arrays | `libsdca.loss_and_grad` |
| PyTorch neural-network loss layers | `libsdca.torch.CrossEntropyLoss`, `TopKCrossEntropyLoss`, ... ([TORCH.md](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/TORCH.md)) |
| JAX loss functions (`jit`, `grad`, `vmap`) | `libsdca.jax.cross_entropy`, `topk_hinge_loss`, ... ([JAX.md](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/JAX.md)) |
| Single SDCA dual block update | `libsdca.coordinate_update` |

## Quick start

```python
import numpy as np
import libsdca

X = np.array([[1., 0.], [0., 1.], [-1., -1.]])
y = np.array([0, 1, 2])
model = libsdca.fit(X, y)
print(model.status, model.predict(X))
```

Only `model.status == "solved"` guarantees the requested duality gap.

The PyTorch losses replace `torch.nn.CrossEntropyLoss` and add the other losses
([TORCH.md](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/TORCH.md)):

```python
import torch
from libsdca.torch import CrossEntropyLoss, TopKHingeLoss

logits = torch.randn(32, 5, requires_grad=True)
target = torch.randint(5, (32,))
(CrossEntropyLoss()(logits, target) + TopKHingeLoss(k=2)(logits, target)).backward()
```

## Losses

A loss object accepts its own parameters as positional or keyword arguments. Passing a
parameter belonging to another loss raises `TypeError`, as does omitting `k`.
`libsdca.LOSSES` maps shared names to these classes:

| Loss | Parameters (defaults) | Targets |
| --- | --- | --- |
| `CrossEntropy()` | — | integer labels `(n,)`; `Probabilities` (finite, nonnegative; in `fit`, rows that sum to one within the rounding of a row normalized in `X`'s dtype, $\gamma_{m+1}$, renormalized) |
| `TopKCrossEntropy(k)` | `k` in `[1, m - 1]` | integer labels |
| `MulticlassHinge(gamma=0.0)` | `gamma >= 0` (`0`: nonsmooth) | integer labels |
| `TopKHinge(k, variant="beta", gamma=0.0)` | `k`, `"beta"` or `"alpha"`, `gamma` | integer labels |
| `MultilabelHinge(gamma=0.0)` | `gamma >= 0` (`0`: nonsmooth) | label sets |
| `MultilabelCrossEntropy()` | — | label sets |
| `ProximalCrossEntropy(gamma=1.0)` | the step `gamma > 0` | integer labels (`loss_and_grad` only) |
| `Distillation(temperature=1.0)` | `temperature > 0` | teacher logits `(n, m)` (`loss_and_grad` only) |
| `ProximalDistillation(temperature=1.0, gamma=1.0)` | `temperature`, the step `gamma > 0` | teacher logits (`loss_and_grad` only) |

Label sets are a boolean mask `(n, m)` or a sequence of integer label sequences; each set
needs a class outside it. A loss takes one kind of target: nothing is inferred from the
type of `y`.

## Classifier

```python
import numpy as np
from libsdca.sklearn import SDCAClassifier

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 20))
y = np.array(["cat", "dog", "emu"])[(X @ rng.normal(size=(20, 3))).argmax(axis=1)]

clf = SDCAClassifier("topk_cross_entropy", k=2, C=1.0).fit(X, y)
print(clf.predict(X[:5]), clf.score(X, y))
```

`SDCAClassifier` takes single-label loss names. Pass the loss's `k`, `variant`
(`topk_hinge`) and `gamma` as keywords; `None` selects its default. A parameter belonging to another loss raises
`TypeError` at `fit`, as for the loss objects.

| Setting | Default | Purpose |
| --- | --- | --- |
| `loss`, `C` | `"cross_entropy"`, `1.0` | Training objective and loss weight |
| `tol`, `max_iter`, `eval_epoch` | `1e-3`, `1000`, `10` | Gap target, epoch limit, evaluation interval |
| `kernel` | `"linear"` | Use `"precomputed"` for Gram matrices |
| `shrinking`, `warm_start` | `"auto"`, `False` | Hinge acceleration and reuse of dual variables |
| `precision`, `n_jobs`, `random_state` | `None` | Working precision, threads, random seed |

It supports `Pipeline`, `clone`, cross-validation and grid search, arbitrary sortable
labels, and scikit-learn's `check_estimator` (except checks supplying invalid Gram
matrices). `predict_proba` and `predict_log_proba` are available for `cross_entropy`.
If the solver stops before reaching the tolerance, `fit` returns the fitted estimator and
issues scikit-learn’s `ConvergenceWarning`. Inspect `status_` and `history_` for the stopping
reason and final gap.
A warm start reuses dual variables when the loss, dimensions, kernel mode and training labels
in row order agree with the previous fit. Changing the labels starts a fresh solve.
There is no intercept; append a constant feature for a regularized one.

### Fitted attributes

| Attribute | Meaning |
| --- | --- |
| `classes_` | Labels in score-column order |
| `coef_` | `(n_classes, n_features)`; binary fits return `(1, n_features)`, the margin weights; absent for precomputed kernels |
| `dual_coef_` | `(n_classes, n_samples)` |
| `n_features_in_`, `n_iter_` | Feature count and completed epochs |
| `status_`, `history_`, `model_` | Convergence status, evaluations, underlying `Model` |

## Training

```python
import numpy as np
import libsdca

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 20))
y_int = (X @ rng.normal(size=(20, 3))).argmax(axis=1)
model = libsdca.fit(X, y_int, libsdca.TopKHinge(k=2, variant="alpha", gamma=0.5), C=1.0)
print(model.status, model.epochs, model.history["relative_gap"][-1])
scores = model.decision_function(X)    # X @ model.weights.T
predicted = model.predict(X)
```

`fit(X, y, loss=CrossEntropy(), **options)` minimizes
`0.5 * ||W - W0||² + C * Σᵢ loss(W Xᵢ, yᵢ)` with no implicit intercept.
`C = 1 / (λn)` gives the averaged objective. Choose a training loss from
`libsdca.OBJECTIVES`; shared defaults are in
[solver options](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#options).

| Option | Default | Meaning |
| --- | --- | --- |
| `X` | — | Features `(n, d)`, or `libsdca.Kernel(K)` for a Gram matrix `(n, n)` |
| `y` | — | The loss's targets, or `libsdca.Probabilities(P)` for `CrossEntropy()` (distillation) |
| `C` | `1.0` | Loss weight (positive) |
| `tol`, `max_epoch`, `eval_epoch` | `1e-3`, `1000`, `10` | Relative duality-gap target, epoch limit, and evaluation period (`0`: only at the end) |
| `shrinking` | `"auto"` | Active-set shrinking and gap-safe screening of the hinges ([API.md](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#shrinking)) |
| `seed` | `0` | The order of the examples in the epochs |
| `center`, `initial_dual` | `None`, `None` | Regularizer center `W0` `(m, d)`, and warm-start duals `(n, m)` (checked by the library, which builds `W` from them) |
| `precision`, `num_threads` | `None`, `None` | Working precision (below) and threads of the evaluations |

### Results

| Field / method | Meaning |
| --- | --- |
| `loss`, `C`, `num_classes`, `kernel` | Configuration used by the fit |
| `weights`, `dual` | `(m, d)` weights, one row per class (`None` for kernels), `(n, m)` duals; C-contiguous, written by the native library in place ([layouts](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#array-layouts)) |
| `status`, `epochs` | Stop reason and completed passes; only `"solved"` guarantees the gap |
| `history` | Arrays indexed by the `libsdca.HISTORY` keys; `accuracy` is rank loss for multilabel fits |
| `decision_function`, `predict`, `score` | Scores, labels, and top-1 accuracy (top class in the target set for multilabel fits) |
| `predict_proba`, `predict_log_proba` | Available for the cross-entropy losses |

## Losses and gradients

```python
values, gradients = libsdca.loss_and_grad(scores, y_int, libsdca.TopKHinge(k=2, gamma=0.5))
values, gradients = libsdca.loss_and_grad(student, teacher, libsdca.Distillation(temperature=2.0))
```

- `loss_and_grad(logits, target, loss=CrossEntropy(), *, precision=None, num_threads=None,
  check_rows=False, weight=None, label_smoothing=None, ignore_index=None)` computes per-row
  losses `(n,)` (`float64`) and gradients `(n, m)` (the storage dtype) of any loss of
  `libsdca.LOSSES`. `weight`, `label_smoothing` and `ignore_index` are PyTorch's
  `cross_entropy` options, for the losses that take them
  ([options](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#loss-evaluation-options));
  `ignore_index` has no default (PyTorch's `cross_entropy` takes `-100`). A logit of `-inf` masks its class for every
  loss ([masked classes](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#masked-classes)).
  A label out of range raises `ValueError` (`Target 7 is out of bounds.`, as in PyTorch).
  Rows it cannot compute (`NaN` or `+inf` logits, invalid masked rows) are `NaN`, with one
  `InvalidRowsWarning` per call; `check_rows=True` raises `ValueError` instead. A batch of
  zero rows gives empty arrays.
- `coordinate_update(logits, target, loss=CrossEntropy(), *, previous=None, C=1.0,
  norm2=1.0, precision=None, num_threads=None)` performs one exact SDCA dual block step;
  `norm2` is a scalar or one squared norm per row, and `previous` must be feasible.

Row loops parallelize across `num_threads` threads (`None` uses `SDCA_NUM_THREADS` or all
available CPUs; `1` runs serially) with bitwise-identical results
([threads](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#threads),
[environment variables](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#environment-variables)).

Wrong types raise `TypeError` (a sparse matrix too: pass `X.toarray()`), invalid values
`ValueError`, and failed native computations `RuntimeError`. PyTorch tensors on CUDA or
another device are copied to the host with a one-time `RuntimeWarning`; `libsdca.torch`
keeps the losses on the device.

## Precision

Outputs (`weights`, `dual`, `gradients`) keep the storage dtype; loss values are `float64`.
`float32` and `float64` inputs keep their dtype, 16-bit floats are cast to `float32`
(exactly) and integers to `float64`.

| `precision` | Working precision |
| --- | --- |
| `None` | The storage precision |
| `"float32"` | Casts `float64` inputs to `float32` |
| `"float64"` | `double` |

For `float32` inputs, `precision="float64"` keeps the solver iterate in `double` for tighter
convergence, at the cost of `8(d + n)m` extra bytes and slower fits. Both modes check the
returned model's duality gap in `double`
([support matrix](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#precision)).

```python
X32 = X.astype(np.float32)
model = libsdca.fit(X32, y_int)                        # float32 arrays and working iterate
exact = libsdca.fit(X32, y_int, precision="float64")   # float32 arrays, double working iterate
```

## Building from source

```bash
pip install .            # links a system CBLAS if one is found, else the built-in products
```

See [DEPENDENCIES.md](https://github.com/mlapin/libsdca/blob/v0.4.0/DEPENDENCIES.md) for
`SDCA_BLAS_BACKEND` options (`AUTO`, `EXTERNAL`, `BUILTIN`). A build that finds no BLAS uses
the built-in products, and `fit` warns once (`RuntimeWarning`; [API.md](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#blas)).
