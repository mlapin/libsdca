# libsdca for Python

SDCA solvers for multiclass and multilabel classification: linear and kernel
SVMs (multiclass, top-k and multilabel hinge losses) and softmax classifiers
(cross-entropy and top-k entropy), the problems the multiclass solvers of
LIBLINEAR and LIBSVM address, plus the same losses as NumPy functions, PyTorch
layers and JAX functions. The native code is a C++
library loaded through `ctypes`; the wheels for Linux x86-64 and macOS
arm64 need no compiler (other platforms build the sdist with a C++11
compiler), and NumPy is the only required dependency. `import libsdca` loads neither
PyTorch nor scikit-learn, each imported on first use of its classes, nor
JAX (`import libsdca.jax`).

The losses are numerically stable where the standard framework implementations
are not: the loss and target gradient of a confident prediction stay nonzero to
the subnormal limit of the output dtype instead of collapsing to zero, gradient
entries are rounded once, and the distillation gradient is exactly zero at
equal student and teacher logits, in float16, bfloat16, float32 and float64 on
CPU and CUDA; the same arithmetic drives the SDCA solver
([numerics](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/numerics.md)).
The [known limits](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/numerics.md#known-limits)
list the exceptions: JAX on the CPU flushes results below the smallest normal
number to zero, default float32 work on a GPU can lose tails whose sum is
representable, and the saved gradients of `linear_cross_entropy` round twice.

```bash
pip install libsdca              # NumPy interface
pip install 'libsdca[sklearn]'   # with the scikit-learn estimator SDCAClassifier
pip install 'libsdca[torch]'     # with the PyTorch loss layers
pip install 'libsdca[jax]'       # with the JAX losses
```

| Task | API |
| --- | --- |
| Train a classifier, scikit-learn style | `SDCAClassifier` |
| Train with full control (label sets, warm starts, history) | `libsdca.fit` → `Model` |
| Loss values and gradients of NumPy scores | `libsdca.loss_and_grad` |
| A loss layer for neural networks | `libsdca.CrossEntropyLoss`, `libsdca.TopKCrossEntropyLoss` ([guide](https://github.com/mlapin/libsdca/blob/v0.3.0/python/TORCH.md)) |
| A loss for JAX (`jit`, `grad`, `vmap`) | `libsdca.jax.sdca_loss` ([guide](https://github.com/mlapin/libsdca/blob/v0.3.0/python/JAX.md)) |
| One SDCA coordinate step | `libsdca.coordinate_update` |

## Classifier

```python
import numpy as np
from libsdca import SDCAClassifier

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 20))
labels = (X @ rng.normal(size=(20, 3))).argmax(axis=1)   # 0, 1, 2
y = np.array(["cat", "dog", "emu"])[labels]

clf = SDCAClassifier("topk_cross_entropy", k=2, C=1.0).fit(X, y)
print(clf.predict(X[:5]), clf.score(X, y))
```

`SDCAClassifier` (`libsdca.sklearn`) is a scikit-learn estimator that passes
`check_estimator`: `fit`, `predict`, `decision_function`, `score`, and
`predict_proba` and `predict_log_proba` for `objective="cross_entropy"`. It works with pipelines, `clone`,
`cross_val_score` and `GridSearchCV`. Labels may be any sortable values. `kernel="precomputed"` trains on a Gram
matrix `K(train, train)` and predicts from `K(test, train)`. A fit that stops
before `tol` emits scikit-learn's `ConvergenceWarning`. Fitted attributes:
`classes_`, `coef_ (n_classes, n_features)`, `dual_coef_ (n_classes,
n_samples)`, `n_features_in_`, `n_iter_`, `status_`, `history_` and `model_`.

## Training

```python
import libsdca

model = libsdca.fit(X, labels, "topk_hinge_alpha", C=1.0, k=2, gamma=0.5)
print(model.status, model.epochs, model.history["relative_gap"][-1])
scores = model.decision_function(X)    # X @ model.weights
predicted = model.predict(X)
```

`libsdca.fit(X, y, objective="cross_entropy", *, ...)` minimizes
`0.5*||W||^2 + C*sum_i loss(X[i] @ W, y[i])` over the weights `W (d, m)` for `m` classes.
There is no intercept; append a constant feature. For
`lambda/2*||W||^2 + mean_i loss`, set `C = 1/(lambda*n)`.

| Objective | Loss | Options |
| --- | --- | --- |
| `cross_entropy` | softmax cross-entropy | |
| `topk_cross_entropy` | top-k entropy | `k` |
| `multiclass_hinge` | multiclass hinge (Crammer-Singer) | `gamma` |
| `topk_hinge_alpha`, `topk_hinge_beta` | the two top-k hinge losses | `k`, `gamma` |
| `multilabel_hinge` | multilabel hinge | `gamma` |
| `multilabel_cross_entropy` | softmax cross-entropy to the uniform distribution on the labels (not per-class binary cross-entropy) | |

Labels are integers in `[0, num_classes)`. The multilabel objectives take a
boolean mask `(n, num_classes)` or one list of labels per example, with
between 1 and `num_classes-1` labels each; `cross_entropy` on such targets is
`multilabel_cross_entropy`. Further options, with the defaults shared by every
interface:

| Option | Default | Meaning |
| --- | --- | --- |
| `num_classes` | `max(y)+1` | number of classes |
| `max_epoch` | 1000 | epoch limit |
| `eval_epoch` | 10 | epochs between duality-gap evaluations (up to two more fall between where the gap's trend reaches `tol`) |
| `tol` | 1e-3 | relative duality gap to stop at (`0`: run to the other limits) |
| `eval_on_start` | `False` | evaluate before the first epoch |
| `recompute_primal` | `True` | rebuild `W = center + X.T @ A` at each evaluation (feature input) |
| `projection_precision` | `"float64"` | `multilabel_hinge` projections in double with a checked result, or `"extended"` ([API.md](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/API.md#projection-precision)) |
| `kernel` | `False` | `X` is a Gram matrix `(n, n)`; predict from `K(test, train)` |
| `center` | `None` | regularize `0.5*||W-center||^2` instead (feature input) |
| `initial_dual` | `None` | warm start from dual variables, usually `model.dual` |
| `precision` | from `X.dtype` | see below |

The returned `Model` holds `weights (d, m)` (`None` for kernels), `dual (n,
m)`, `status`, `epochs`, `history` (one array per field: `epoch`, `primal`,
`dual`, `primal_loss`, `dual_loss`, `regularizer`, `relative_gap`,
`accuracy` (or `rank_loss` for multilabel objectives), `cpu_time`,
`wall_time`) and `params` (`C`, `k`, `gamma`, `num_classes`). Only `status == "solved"` certifies the
gap. A warm start is checked against the objective's feasible set and
rejected with a `ValueError` otherwise.

With `center=W0`, history and stopping use the full centered objective
`0.5*np.sum((model.weights-W0)**2) + C*np.sum(losses)` and its matching
dual. The regularizer is evaluated directly from the difference, avoiding
large cancelling constants. Large centers can still hide small weight
updates in float storage; choose `precision="float64"` when needed, and
inspect `status` rather than assuming a fixed epoch budget converged.

## Losses and gradients

```python
values, gradients = libsdca.loss_and_grad(scores, labels, "topk_hinge_beta", k=2, gamma=0.5)
```

`loss_and_grad(logits, target, loss="cross_entropy", *, k=1, gamma=0.0,
alpha=1.0, num_threads=None, projection_precision="float64", precision=None)` returns the per-row loss `(n,)` and its gradient `(n, m)` at the
given scores, for every objective above and `proximal_cross_entropy` (the Moreau
envelope of the entropy loss with parameter `alpha`). Targets are labels, a
boolean mask or label lists; entropy puts a uniform target on the positive
labels. Hinge losses return a subgradient at their kinks.

`coordinate_update(logits, y, objective="cross_entropy", *, previous=None, C=1.0,
norm2=1.0, k=1, gamma=0.0, num_threads=None, projection_precision="float64",
precision=None)` returns the dual blocks after one
exact SDCA coordinate step, for building custom solvers.

Both can split the rows among `num_threads` native threads: `None` takes
`SDCA_NUM_THREADS`, else 0: automatic (the available CPUs, as the work
allows); 1 runs on the calling thread, and the results are
identical for every count ([API.md](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/API.md#threads);
every variable: [environment variables](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/API.md#environment-variables)).

## Precision

Storage and working arithmetic are decoupled across all interfaces:
- **Storage**: Outputs (`weights`, `dual`, `gradients`) always keep the input storage dtype (`float32` or `float64`).
- **Compute**: `precision` is the precision of the solve: the working arithmetic and `fit`'s iterate (`"float32"` or `"float64"`, or aliases `"fp32"`, `"single"`, `"fp64"`, `"double"`, `np.float32`, `np.float64`, `float`). A float32 `fit` in `"float64"` keeps the weights and duals in double copies (`8 (d + n) m` bytes) and returns them rounded once, with the certificate on the returned values.

| Input `X.dtype` | `precision` | Storage | Working Arithmetic | Notes |
| --- | --- | --- | --- | --- |
| `float64` | `None` or `"float64"` | `float64` | `float64` | Default for `float64` |
| `float32` | `None` or `"float64"` | `float32` | `float64` | Default for `float32` (`fit`, `loss_and_grad`, `coordinate_update`) |
| `float32` | `"float32"` (`np.float32`, `"fp32"`) | `float32` | `float32` | Entropy steps 1.8–3.2× and `cross_entropy` training 1.6–1.9× faster than with `"float64"` on the tested corpus |

```python
X32 = X.astype(np.float32)
model = libsdca.fit(X32, labels)                   # float32 arrays, the solve in float64 (default)
fast = libsdca.fit(X32, labels, precision="float32")  # float32 arrays, the solve in float32
clf = SDCAClassifier(precision=np.float32).fit(X32, y)
```

Native NumPy entry points require float32 or float64 arrays; explicitly cast
half/bfloat16 inputs before training. Float64 storage with float32 compute is
rejected. See the [support matrix and parameter domains](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/API.md#precision).

The arithmetic applies to `fit`, `coordinate_update` and `loss_and_grad`
(whose `"float32"` computes in double where float cannot hold `gamma`,
`alpha` or a row's score spread); the hinge steps gain nothing from
`"float32"`. On the tested corpus, `"float64"` compute on `float32` data keeps
entropy blocks within a few float ulps of the double solution; `"float32"` is
accurate to the float rounding of the scores
([numerics](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/numerics.md#known-limits)). Outputs keep the storage
dtype; loss values are always float64. The PyTorch and JAX losses take
`precision` too; see [their working precision](https://github.com/mlapin/libsdca/blob/v0.3.0/python/TORCH.md#precision).

## Vanilla softmax in PyTorch

```python
import torch
from libsdca import CrossEntropyLoss

logits = torch.randn(32, 5, requires_grad=True)
target = torch.randint(5, (32,))
# Replace: criterion = torch.nn.CrossEntropyLoss()
criterion = CrossEntropyLoss()
criterion(logits, target).backward()
```

Use the same raw logits, integer labels, model and optimizer. The loss follows
the logits onto CPU or CUDA and preserves tiny confident-prediction losses and
target gradients; speed depends on the shape and device. See the
[compatibility notes](https://github.com/mlapin/libsdca/blob/v0.3.0/python/TORCH.md#replace-pytorch-cross-entropy)
and the [performance notes](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/performance.md).

## Building from source

`pip install .` in a source checkout builds the native library into the
package, using a system CBLAS when one is found (`SDCA_BLAS_BACKEND=BUILTIN`
or `EXTERNAL` forces one; see [DEPENDENCIES.md](https://github.com/mlapin/libsdca/blob/v0.3.0/DEPENDENCIES.md)).
`pip install -e .` does the same for development. Without installing,
`python python/build.py` builds `build/python/libsdca_c.so`, which
`PYTHONPATH=python` picks up (or the directory in `SDCA_LIBRARY_DIR`).
The package loads the library from `SDCA_LIBRARY_DIR` when it is set, else from
`build/python` of the checkout it is imported from when that file exists, and
only then the library installed with the package. After an editable install,
delete `build/python` (or set `SDCA_LIBRARY_DIR`) so that a stale build does not
shadow the installed library.

## Tests

```bash
pip install -e .                       # rerun after changing include/ or src/
python -m pytest -n auto --dist worksteal tests/python --group all   # core, torch, jax or all
python -m pytest tests/python --group all -m serial   # thread timing; core and all only
```

The suites check every objective against independent optimizers (L-BFGS,
SLSQP) and 90- to 100-digit references at extreme scores, the KKT conditions
of the coordinate steps, warm starts, layouts, input validation, the
scikit-learn estimator contract, the PyTorch layers (`gradcheck`,
`gradgradcheck`, CUDA when available) and the JAX losses.
