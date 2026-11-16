# libsdca for Python

A fast and accurate implementation of classification losses: cross-entropy, multiclass and
multilabel SVM hinges, top-*k* losses and knowledge distillation, as PyTorch and JAX loss
layers, NumPy functions, and SDCA solvers for linear and kernel classifiers with a
scikit-learn estimator ([project page](https://mlapin.github.io/libsdca/),
[technical report](https://mlapin.github.io/libsdca/report/)).

The native C++ library loads via `ctypes`. Prebuilt wheels for Linux `x86_64` and macOS
`arm64` require no compiler and depend only on NumPy; `import libsdca` does not import
PyTorch, scikit-learn or JAX until used. Small losses and gradients of confident
predictions are kept down to the output dtype's subnormal range (XLA:CPU flushes
subnormals to zero), and distillation gradients are exactly zero at equal logits ([known limits](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#known-limits)).

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
| PyTorch neural-network loss layers | `libsdca.torch.CrossEntropyLoss`, `TopKCrossEntropyLoss`, ... ([TORCH.md](https://github.com/mlapin/libsdca/blob/v0.4.0/python/TORCH.md)) |
| JAX loss functions (`jit`, `grad`, `vmap`) | `libsdca.jax.cross_entropy`, `topk_hinge_loss`, ... ([JAX.md](https://github.com/mlapin/libsdca/blob/v0.4.0/python/JAX.md)) |
| Single SDCA dual block update | `libsdca.coordinate_update` |

## Losses

A loss is an object with its own parameters only (a parameter of another loss is a
`TypeError`); `libsdca.LOSSES` maps the shared names to the classes:

| Loss | Parameters (defaults) | Targets |
| --- | --- | --- |
| `CrossEntropy()` | — | integer labels `(n,)`; `Probabilities` in `fit` |
| `TopKCrossEntropy(k)` | `k` in `[1, m - 1]` | integer labels |
| `MulticlassHinge(gamma=0.0)` | `gamma >= 0` (`0`: nonsmooth) | integer labels |
| `TopKHingeAlpha(k, gamma=0.0)`, `TopKHingeBeta(k, gamma=0.0)` | `k`, `gamma` | integer labels |
| `MultilabelHinge(gamma=0.0, projection_precision="float64")` | `gamma`, `"float64"` or `"extended"` | label sets |
| `MultilabelCrossEntropy()` | — | label sets |
| `ProximalCrossEntropy(alpha=1.0)` | `alpha > 0` | integer labels (`loss_and_grad` only) |
| `Distillation(temperature=1.0)` | `temperature > 0` | teacher logits `(n, m)` (`loss_and_grad` only) |
| `ProximalDistillation(temperature=1.0, alpha=1.0)` | `temperature`, `alpha` | teacher logits (`loss_and_grad` only) |

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

`SDCAClassifier(loss="cross_entropy", *, C=1.0, k=None, gamma=None, tol=1e-3,
max_iter=1000, eval_epoch=10, shrinking="auto", kernel="linear", precision=None,
n_jobs=None, random_state=None, warm_start=False)` takes the single-label losses by name;
`k` and `gamma` are the loss's (`None`: its default), and giving them to a loss that does not
take them raises `ValueError` at `fit`. It passes scikit-learn's `check_estimator` suite
(except checks that pass invalid Gram matrices to `kernel="precomputed"`) and supports
`Pipeline`, `clone`, `cross_val_score`, `GridSearchCV`, arbitrary sortable labels,
`predict_proba` / `predict_log_proba` (for `cross_entropy`), and precomputed Gram matrices.
Fitted attributes: `classes_`, `coef_` (`(n_classes, n_features)`, or `(1, n_features)`
for two classes, the weights of the margin `decision_function` returns; absent for
precomputed kernels), `dual_coef_ (n_classes, n_samples)`, `n_features_in_`, `n_iter_`
(epochs), `status_`, `history_`, and `model_`. There is no intercept: append a constant
feature to `X` for a regularized one.

## Training

```python
import libsdca

model = libsdca.fit(X, y_int, libsdca.TopKHingeAlpha(k=2, gamma=0.5), C=1.0)
print(model.status, model.epochs, model.history["relative_gap"][-1])
scores = model.decision_function(X)    # X @ model.weights
predicted = model.predict(X)
```

`libsdca.fit(X, y, loss=CrossEntropy(), *, C=1.0, tol=1e-3, max_epoch=1000, eval_epoch=10,
eval_on_start=False, recompute_primal=True, shrinking="auto", seed=0, num_classes=None,
center=None, initial_dual=None, precision=None, num_threads=None)` minimizes
`0.5 * ||W - W0||² + C * Σᵢ loss(Xᵢ W, yᵢ)` over the `(d, m)` weights `W`, with no implicit
intercept (`C = 1 / (λn)` gives the averaged form `λ/2 * ||W||² + (1/n) Σᵢ lossᵢ`), for a loss of
`libsdca.OBJECTIVES`. See [API.md](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#loss-definitions)
for the losses and options:

| Option | Default | Meaning |
| --- | --- | --- |
| `X` | — | Features `(n, d)`, or `libsdca.Kernel(K)` for a Gram matrix `(n, n)` |
| `y` | — | The loss's targets, or `libsdca.Probabilities(P)` for `CrossEntropy()` (distillation) |
| `C` | `1.0` | Loss weight (positive) |
| `tol`, `max_epoch`, `eval_epoch` | `1e-3`, `1000`, `10` | Relative duality-gap target, epoch limit, and evaluation period (`0`: only at the end) |
| `shrinking` | `"auto"` | Active-set shrinking and gap-safe screening of the hinges ([API.md](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#shrinking)) |
| `seed` | `0` | The order of the examples in the epochs |
| `center`, `initial_dual` | `None`, `None` | Regularizer center `W0` `(d, m)`, and warm-start duals `(n, m)` (checked by the library, which builds `W` from them) |
| `precision`, `num_threads` | `None`, `None` | Working precision (below) and threads of the evaluations |

`Model` holds `loss`, `C`, `num_classes`, `kernel`, `weights (d, m)` (`None` for a Gram
matrix), `dual (n, m)`, `status` (one of `libsdca.STATUS`; only `"solved"` guarantees the
duality gap), `epochs`, and `history` (the `libsdca.HISTORY` keys for every loss;
`accuracy` holds the rank loss of the multilabel losses); `decision_function`, `predict`,
`score` (top-1 accuracy, or the fraction whose top class is in the label set) and, for the
cross-entropy losses, `predict_proba` and `predict_log_proba`.

## Losses and gradients

```python
values, gradients = libsdca.loss_and_grad(scores, y_int, libsdca.TopKHingeBeta(k=2, gamma=0.5))
values, gradients = libsdca.loss_and_grad(student, teacher, libsdca.Distillation(temperature=2.0))
```

- `loss_and_grad(logits, target, loss=CrossEntropy(), *, precision=None, num_threads=None)`
  computes per-row losses `(n,)` (`float64`) and gradients `(n, m)` (the storage dtype) of
  any loss of `libsdca.LOSSES`.
- `coordinate_update(logits, target, loss=CrossEntropy(), *, previous=None, C=1.0,
  norm2=1.0, precision=None, num_threads=None)` performs one exact SDCA dual block step;
  `norm2` is a scalar or one squared norm per row, and `previous` must be feasible.

Row loops parallelize across `num_threads` threads (`None` uses `SDCA_NUM_THREADS` or all
available CPUs; `1` runs serially) with bitwise-identical results
([threads](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#threads),
[environment variables](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#environment-variables)).
Wrong types raise `TypeError` (a sparse matrix too: pass `X.toarray()`), invalid values
`ValueError`, and failed native computations `RuntimeError`.

## Precision

Outputs (`weights`, `dual`, `gradients`) keep the storage dtype: `float32` and `float64`
inputs keep theirs, 16-bit floats are cast to `float32` (exactly) and integers to `float64`;
loss values are `float64`. `precision=None` works in the storage precision; `"float32"`
casts `float64` inputs to `float32`; `"float64"` works in `double`. For `float32` inputs,
`precision="float64"` keeps the solver iterate in `double` for tighter convergence, at the
cost of `8(d + n)m` extra bytes and slower fits. Both modes check the returned model's
duality gap in `double`
([support matrix](https://github.com/mlapin/libsdca/blob/v0.4.0/docs/API.md#precision)).

```python
X32 = X.astype(np.float32)
model = libsdca.fit(X32, y_int)                        # float32 arrays and working iterate
exact = libsdca.fit(X32, y_int, precision="float64")   # float32 arrays, double working iterate
```

## PyTorch drop-in cross-entropy

`libsdca.torch.CrossEntropyLoss` is a drop-in for `torch.nn.CrossEntropyLoss`, with class
labels or probabilities, `weight`, `ignore_index`, `reduction` and `label_smoothing`.

```python
import torch
from libsdca.torch import CrossEntropyLoss

logits = torch.randn(32, 5, requires_grad=True)
target = torch.randint(5, (32,))
CrossEntropyLoss()(logits, target).backward()
```

See [TORCH.md](https://github.com/mlapin/libsdca/blob/v0.4.0/python/TORCH.md#replace-pytorch-cross-entropy).

## Building from source and testing

```bash
pip install -e .                                                   # editable install (uses system CBLAS if found)
python -m pytest -n auto --dist worksteal tests/python --group all # core, torch, jax, or all
python -m pytest tests/python --group all -m serial
```

See [DEPENDENCIES.md](https://github.com/mlapin/libsdca/blob/v0.4.0/DEPENDENCIES.md) for
`SDCA_BLAS_BACKEND` options (`AUTO`, `EXTERNAL`, `BUILTIN`).
