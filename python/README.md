# libsdca for Python

SDCA solvers for linear and kernel SVMs and softmax classifiers (`cross_entropy`,
`topk_cross_entropy`, `multiclass_hinge`, `topk_hinge_alpha`, `topk_hinge_beta`,
`multilabel_hinge`, `multilabel_cross_entropy`, `distillation`), plus differentiable
loss layers for PyTorch, JAX and NumPy.

The native C++ library loads via `ctypes`. Prebuilt wheels for Linux `x86_64` and macOS
`arm64` require no compiler and depend only on NumPy; `import libsdca` does not import
PyTorch, scikit-learn or JAX until used. All losses preserve tiny confident-prediction
losses and gradients down to the output dtype's subnormal limit and produce exact zero
distillation gradients at equal logits ([numerics](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/numerics.md),
[known limits](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/numerics.md#known-limits)).

```bash
pip install libsdca              # NumPy interface
pip install 'libsdca[sklearn]'   # scikit-learn estimator SDCAClassifier
pip install 'libsdca[torch]'     # PyTorch loss layers
pip install 'libsdca[jax]'       # JAX loss functions
```

| Task | Entry point |
| --- | --- |
| Train a scikit-learn classifier | `libsdca.SDCAClassifier` |
| Train with full control (warm starts, label sets, history) | `libsdca.fit` $\to$ `Model` |
| Evaluate losses and gradients on NumPy arrays | `libsdca.loss_and_grad`, `distillation_loss_and_grad` |
| PyTorch neural-network loss layers | `libsdca.CrossEntropyLoss`, `TopKCrossEntropyLoss`, ... ([TORCH.md](https://github.com/mlapin/libsdca/blob/v0.3.0/python/TORCH.md)) |
| JAX loss functions (`jit`, `grad`, `vmap`) | `libsdca.jax.sdca_loss`, ... ([JAX.md](https://github.com/mlapin/libsdca/blob/v0.3.0/python/JAX.md)) |
| Single SDCA dual block update | `libsdca.coordinate_update` |

## Classifier

```python
import numpy as np
from libsdca import SDCAClassifier

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 20))
y = np.array(["cat", "dog", "emu"])[(X @ rng.normal(size=(20, 3))).argmax(axis=1)]

clf = SDCAClassifier("topk_cross_entropy", k=2, C=1.0).fit(X, y)
print(clf.predict(X[:5]), clf.score(X, y))
```

`SDCAClassifier` (`libsdca.sklearn`) passes scikit-learn's `check_estimator` suite (except
checks that pass invalid Gram matrices to `kernel="precomputed"`) and supports `Pipeline`,
`clone`, `cross_val_score`, `GridSearchCV`, arbitrary sortable labels, `predict_proba` /
`predict_log_proba` (for `cross_entropy`), and precomputed Gram matrices. Fitted attributes:
`classes_`, `coef_ (n_classes, n_features)`, `dual_coef_ (n_classes, n_samples)`,
`n_features_in_`, `n_iter_`, `status_`, `history_`, and `model_`.

## Training

```python
import libsdca

model = libsdca.fit(X, y_int, "topk_hinge_alpha", C=1.0, k=2, gamma=0.5)
print(model.status, model.epochs, model.history["relative_gap"][-1])
scores = model.decision_function(X)    # X @ model.weights
predicted = model.predict(X)
```

`libsdca.fit(X, y, objective="cross_entropy", *, ...)` minimizes
$0.5\Vert W \Vert_2^2 + C \sum_i \ell(X_i W, y_i)$ over $W \in \mathbb{R}^{d \times m}$ (no implicit
intercept; set $C = 1 / (\lambda n)$ for $\frac{\lambda}{2}\Vert W \Vert_2^2 + \frac{1}{n}\sum_i \ell_i$).
See [API.md](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/API.md#loss-definitions)
for all objectives and options:

| Option | Default | Meaning |
| --- | --- | --- |
| `C` | `1.0` | Loss weight ($> 0$) |
| `k`, `gamma` | `1`, `0.0` | Top-*k* parameter (`[1, m - 1]`) and hinge smoothing (`0` = nonsmooth) |
| `tol`, `max_epoch`, `eval_epoch` | `1e-3`, `1000`, `10` | Relative duality-gap target, epoch limit, and evaluation period |
| `shrinking` | `"auto"` | Active-set shrinking and gap-safe screening for hinges ([API.md](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/API.md#shrinking)) |
| `projection_precision` | `"float64"` | `"float64"` (checked double) or `"extended"` ([API.md](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/API.md#projection-precision)) |
| `kernel`, `center`, `initial_dual` | `False`, `None`, `None` | Gram matrix `(n, n)` input, regularizer center $W_0$, and warm-start duals |
| `precision` | `"float64"` | Working arithmetic and solver iterate (`"float64"` or `"float32"`) |

`Model` exposes `weights (d, m)` (`None` for kernels), `dual (n, m)`, `status` (only
`"solved"` certifies the duality gap), `epochs`, `history`, and `params`.

## Losses and gradients

```python
values, gradients = libsdca.loss_and_grad(scores, y_int, "topk_hinge_beta", k=2, gamma=0.5)
```

- `loss_and_grad(logits, target, loss="cross_entropy", *, k=1, gamma=0.0, alpha=1.0, num_threads=None, projection_precision="float64", precision=None)`
  computes per-row losses `(n,)` and gradients `(n, m)` for any objective or
  `proximal_cross_entropy`.
- `distillation_loss_and_grad` and `proximal_distillation_loss_and_grad` evaluate logit
  distillation and its Moreau envelope.
- `coordinate_update(logits, y, objective="cross_entropy", *, previous=None, C=1.0, norm2=1.0, ...)`
  performs one exact SDCA dual block step.

Row loops parallelize across `num_threads` threads (`None` uses `SDCA_NUM_THREADS` or
automatic CPU detection; `1` runs serially) with bitwise-identical results
([threads](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/API.md#threads),
[environment variables](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/API.md#environment-variables)).

## Precision

Outputs (`weights`, `dual`, `gradients`) keep the input array's `float32` or `float64`
dtype; loss values are `float64`. For `float32` inputs, the default `precision="float64"`
keeps the solver iterate in `double` (`8(d + n)m` bytes) so convergence does not stall
above `1e-5`; pass `precision="float32"` for `1.6–1.9×` faster `cross_entropy` fits
([support matrix](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/API.md#precision),
[PyTorch precision](https://github.com/mlapin/libsdca/blob/v0.3.0/python/TORCH.md#precision)).

```python
X32 = X.astype(np.float32)
model = libsdca.fit(X32, y_int)                        # float32 arrays, double solve (default)
fast = libsdca.fit(X32, y_int, precision="float32")    # float32 arrays, float32 solve
```

## PyTorch drop-in cross-entropy

`CrossEntropyLoss` is a drop-in for `torch.nn.CrossEntropyLoss`, except it takes integer class
labels only and no label smoothing.

```python
import torch
from libsdca import CrossEntropyLoss

logits = torch.randn(32, 5, requires_grad=True)
target = torch.randint(5, (32,))
CrossEntropyLoss()(logits, target).backward()
```

See [TORCH.md](https://github.com/mlapin/libsdca/blob/v0.3.0/python/TORCH.md#replace-pytorch-cross-entropy)
and [performance.md](https://github.com/mlapin/libsdca/blob/v0.3.0/docs/performance.md).

## Building from source and testing

```bash
pip install -e .                                                   # editable install (uses system CBLAS if found)
python -m pytest -n auto --dist worksteal tests/python --group all # core, torch, jax, or all
python -m pytest tests/python --group all -m serial
```

See [DEPENDENCIES.md](https://github.com/mlapin/libsdca/blob/v0.3.0/DEPENDENCIES.md) for
`SDCA_BLAS_BACKEND` options (`AUTO`, `EXTERNAL`, `BUILTIN`).
